import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--wikipath", type=str, default="/home/agois/longformer_stuff/", 
        help="path to dir containing wikitext-103-raw/")
parser.add_argument("--outdir", default="tmp")
parser.add_argument("--pretrained-path", default=None, help="evaluate only")
parser.add_argument("--use_softmax", action="store_true")
parser.add_argument('--epochs', default='3', help="# ignored if max_steps>0")
parser.add_argument('--max-steps', default='3000',
                    help="e.g. 3000 steps // (245batches // 7 grad_accumulation_steps) + 1 = 429 epochs")
parser.add_argument('--train-batch-size', default='2')
parser.add_argument('--eval-batch-size', default='8')
parser.add_argument("--activation", type=str, default="entmax", choices=["entmax", "softmax", "extender", "extender-sim"])
parser.add_argument("--projectors_path", default=None, help="for extender only")
parser.add_argument("--centroids_path", default=None, help="for extender only")
parser.add_argument("--window-size", type=int, default=None, help="for extender-sim only")
parser.add_argument("--use-window", action='store_true', help="include window attention; for extender-sim only")
parser.add_argument("--use-clusters", action='store_true', help="include entmax clusters; for extender-sim only")
parser.add_argument("--use-bigbird", action='store_true', help="include bigbird attention; for extender-sim only")
parser.add_argument("--global-attention", action='store_true', help="CLS token connects everywhere; for extender-sim only")
parser.add_argument("--top_clusters", type=int, default=1, help="for extender-sim only")
parser.add_argument('--num-rand-blocks', type=int, default=None, help='for bigbird only; number of random blocks')
parser.add_argument('--block-size', type=int, default=1, help='for bigbird only; random block size')
args = parser.parse_args()

# print(args.outdir)
# print(args.wikipath)

import logging
import os
import math
import copy
import torch
from dataclasses import dataclass, field
from transformers import RobertaForMaskedLM, RobertaTokenizerFast, TextDataset, DataCollatorForLanguageModeling, Trainer
from transformers import TrainingArguments, HfArgumentParser
from transformers.modeling_longformer import LongformerSelfAttention

from extender.entmax_roberta import EntmaxRobertaForMaskedLM
from extender.extender_roberta import ExtenderRobertaForMaskedLM
from extender.extender_roberta_simulated import ExtenderSimRobertaForMaskedLM

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def pretrain_and_evaluate(args, model, tokenizer, eval_only, model_path):
    val_dataset = TextDataset(tokenizer=tokenizer,
                              file_path=args.val_datapath,  # .train_datapath .val_datapath,  todo revert change: should be val_datapath; model.config.output_attentions = True ; [(outputs[2][l][1]['q_avg'][0], outputs[2][l][1]['k_avg'][0]) for l in range(12)]
                              block_size=tokenizer.max_len)
    if eval_only:
        train_dataset = val_dataset
    else:
        logger.info(f'Loading and tokenizing training data is usually slow: {args.train_datapath}')
        train_dataset = TextDataset(tokenizer=tokenizer,
                                    file_path=args.train_datapath,
                                    block_size=tokenizer.max_len)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    trainer = Trainer(model=model, args=args, data_collator=data_collator,
                      train_dataset=train_dataset, eval_dataset=val_dataset, prediction_loss_only=True,)

    eval_loss = trainer.evaluate()
    eval_loss = eval_loss['eval_loss']
    # logger.info(f'Initial eval bpc: {eval_loss/math.log(2)}')  # prints in base-2, after having printed in base-e

    if not eval_only:
        trainer.train(model_path=model_path)
        trainer.save_model()

        eval_loss = trainer.evaluate()
        eval_loss = eval_loss['eval_loss']
        logger.info(f'Eval bpc after pretraining: {eval_loss/math.log(2)}')


@dataclass
class ModelArgs:
    attention_window: int = field(default=512, metadata={"help": "Size of attention window"})
    max_pos: int = field(default=512, metadata={"help": "Maximum position"})
    # max_pos: int = field(default=4096, metadata={"help": "Maximum position"})


parser = HfArgumentParser((TrainingArguments, ModelArgs,))


training_args, model_args = parser.parse_args_into_dataclasses(look_for_args_file=False, args=[
    '--output_dir', args.outdir,
    '--warmup_steps', '500',
    '--learning_rate', '0.00003',
    '--weight_decay', '0.01',
    '--adam_epsilon', '1e-6',
    '--max_steps', args.max_steps,  # e.g. 3000 steps // (245batches // 7 grad_accumulation_steps) + 1 = 429 epochs
    '--logging_steps', '500',
    '--save_steps', '500',
    '--max_grad_norm', '5.0',
    '--per_gpu_eval_batch_size', '8',
    '--per_gpu_train_batch_size', args.train_batch_size, #'2',  # 32GB gpu with fp32
    '--per_gpu_eval_batch_size', args.eval_batch_size,
    '--gradient_accumulation_steps', '32',
    '--num_train_epochs', args.epochs,  # ignored if max_steps>0
    '--evaluate_during_training',
    '--do_train',
    '--do_eval',
])
training_args.val_datapath = args.wikipath + 'wikitext-103-raw/wiki.valid.raw'
training_args.train_datapath = args.wikipath + 'wikitext-103-raw/wiki.train.raw'

if args.activation == "entmax":
    MaskedLMClass = EntmaxRobertaForMaskedLM
elif args.activation == "softmax":
    MaskedLMClass = RobertaForMaskedLM
elif args.activation == "extender":
    MaskedLMClass = ExtenderRobertaForMaskedLM
elif args.activation == "extender-sim":
    MaskedLMClass = ExtenderSimRobertaForMaskedLM
else:
    raise ValueError
activation_str = args.activation

if args.pretrained_path is None:
    # roberta_base = RobertaForMaskedLM.from_pretrained('roberta-base')
    my_roberta = MaskedLMClass.from_pretrained('roberta-base')
    roberta_base_tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    logger.info('Evaluating roberta-base (seqlen: 512) for refernece ...')
    pretrain_and_evaluate(training_args, my_roberta, roberta_base_tokenizer, eval_only=True, model_path=None)

    model_path = f'{training_args.output_dir}/'+activation_str+'-roberta-512' #roberta-base-{model_args.max_pos}'
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # logger.info(f'Converting roberta-base into roberta-base-{model_args.max_pos}')
    # model, tokenizer = create_long_model(
    #     save_model_to=model_path, attention_window=model_args.attention_window, max_pos=model_args.max_pos)
    logger.info(f'saving model to {model_path}')
    my_roberta.save_pretrained(model_path)
    roberta_base_tokenizer.save_pretrained(model_path)


    logger.info(f'Loading the model from {model_path}')
    tokenizer = RobertaTokenizerFast.from_pretrained(model_path)
    # model = RobertaLongForMaskedLM.from_pretrained(model_path)
    model = MaskedLMClass.from_pretrained(model_path)

    logger.info(f'Pretraining roberta-base-{model_args.max_pos} ... ')
    # training_args.max_steps = 3   ## <<<<<<<<<<<<<<<<<<<<<<<< REMOVE THIS <<<<<<<<<<<<<<<<<<<<<<<<
    pretrain_and_evaluate(training_args, model, tokenizer, eval_only=False, model_path=training_args.output_dir)

    logger.info(f'Saving model to {model_path}')
    model.save_pretrained(model_path)

else:
    model_path = args.pretrained_path
    logger.info(f'Loading the model from {model_path}')
    tokenizer = RobertaTokenizerFast.from_pretrained(model_path)

    if args.activation == "extender":
        model = MaskedLMClass.from_pretrained(model_path,
                                              projectors_path=args.projectors_path,
                                              centroids_path=args.centroids_path)
    elif args.activation == "extender-sim":
        model = MaskedLMClass.from_pretrained(model_path,
                                              window_size=args.window_size,
                                              projectors_path=args.projectors_path,
                                              centroids_path=args.centroids_path,
                                              use_window=args.use_window,
                                              use_clusters=args.use_clusters,
                                              use_bigbird=args.use_bigbird,
                                              global_attention=args.global_attention,
                                              top_clusters=args.top_clusters,
                                              num_rand_blocks=args.num_rand_blocks,
                                              block_size=args.block_size)
    else:
        model = MaskedLMClass.from_pretrained(model_path)
    pretrain_and_evaluate(training_args, model, tokenizer, eval_only=True, model_path=args.pretrained_path)
