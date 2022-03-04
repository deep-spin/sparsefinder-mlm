import argparse
import time

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
parser.add_argument("--output_dir", type=str, default="tmp")
parser.add_argument("--attn_variant", type=str, default="regular")
parser.add_argument("--projections_path", type=str, default=None)
parser.add_argument("--centroids_path", type=str, default=None)
parser.add_argument("--window_size", type=int, default=None)
parser.add_argument("--cls_size", type=int, default=None)
parser.add_argument("--top_clusters", type=int, default=1)
parser.add_argument("--bucket_size", type=int, default=0)
parser.add_argument("--block_size", type=int, default=1)
parser.add_argument("--num_random_blocks", type=int, default=0)
parser.add_argument("--finetune_part", type=str, default="all_but_projections")
parser.add_argument("--timeit", action="store_true")
args = parser.parse_args()

# print(args.outdir)
# print(args.wikipath)
import sys
sys.path.insert(0, ".")  # gambi to get extender import

from transformers import RobertaTokenizer
from extender.roberta_simulated import SimulatedRobertaForMaskedLM

import logging
import os
import math
import copy
import torch
from dataclasses import dataclass, field
from transformers import RobertaTokenizerFast, TextDataset, DataCollatorForLanguageModeling, Trainer
from transformers import TrainingArguments, HfArgumentParser

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def pretrain_and_evaluate(args, model, tokenizer, eval_only, model_path):
    val_dataset = TextDataset(tokenizer=tokenizer,
                              file_path=args.val_datapath,
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


def add_stats_to_result_dict(result, layers, metric='sparsities_gold', prefix="eval"):
    tensor_stats = get_average_results(get_list_of_metrics_per_layer(layers, metric=metric))
    result[prefix+"_"+metric+"_avg_all"] = tensor_stats.mean().item()
    result[prefix+"_"+metric+"_avg_per_layer"] = tensor_stats.mean(1).numpy()
    result[prefix+"_"+metric+"_layers_and_heads"] = tensor_stats.flatten().numpy()


def get_list_of_metrics_per_layer(layers, metric='sparsities_gold'):
    return [getattr(layer.attention.self, metric) for layer in layers]


def get_average_results(list_of_layers, avg='micro'):
    if avg == 'micro':  # (batch, heads) -> reduction=None
        # accum_batches.shape is (num_layers, num_batches, num_heads)
        accum_batches = torch.stack([torch.cat(list_of_tensors, dim=0) for list_of_tensors in list_of_layers])
        avg_dataset = accum_batches.mean(1)
    else:
        # avg_dataset.shape is (num_layers, num_heads)
        avg_dataset = torch.stack([torch.cat(list_of_tensors, dim=0).mean(0) for list_of_tensors in list_of_layers])
    return avg_dataset


def get_layer_centroids(attn_variant, centroids_path, num_layers, num_heads):
    from extender.sim_routing_transformer import KmeansAttention, load_kmeans_obj_from_state_dit

    layers_centroids = [None for _ in range(num_layers)]

    if attn_variant == 'extender' or attn_variant == 'extender_hf' or attn_variant == 'extender_rhf':
        assert centroids_path is not None
        print('Loading centroids from {}'.format(centroids_path))
        layers_centroids = torch.load(centroids_path, map_location=lambda storage, loc: storage)

    elif attn_variant == 'routing':
        assert centroids_path is not None
        print('Just using path to set num rounds and num centroids {}'.format(centroids_path))
        parts = centroids_path.split('.')[-2]
        num_rounds = int(parts.split('_')[1][:-1])
        num_centroids = int(parts.split('_')[2][:-1])
        layers_centroids = KmeansAttention(num_centroids, None, num_heads, num_rounds)
        layers_centroids = [layers_centroids for _ in range(num_layers)]

    elif attn_variant == 'routing_trained':
        assert centroids_path is not None
        layers_centroids = []
        print('Loading centroids from {}'.format(centroids_path))
        centroids_obj = torch.load(centroids_path, map_location=lambda storage, loc: storage)
        for idx in range(num_layers):
            layers_centroids.append(load_kmeans_obj_from_state_dit(os.path.basename(centroids_path),
                                                                   centroids_obj[idx], device='cuda'))

    elif attn_variant == 'routing_extended':
        assert centroids_path is not None
        print('Loading centroids from {}'.format(centroids_path))
        parts = centroids_path.split('.')[-2]  # pt_${rounds}r_${num_clusters}s_1n_shared
        num_rounds = int(parts.split('_')[1][:-1])
        num_centroids = int(parts.split('_')[2][:-1])
        ext_centroids = torch.load(centroids_path, map_location=lambda storage, loc: storage)
        layers_centroids = []
        for idx in range(num_layers):
            centroids = ext_centroids[idx][:, 0]  # get just the first run
            km = KmeansAttention(num_centroids, None, num_heads, num_rounds)
            km.kmeans.load_means(centroids)
            layers_centroids.append(km)

    return layers_centroids



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
    '--do_train',
    '--do_eval',
])
training_args.val_datapath = args.wikipath + 'wikitext-103-raw/wiki.valid.raw'
training_args.train_datapath = args.wikipath + 'wikitext-103-raw/wiki.train.raw'


activation_str = args.activation
if args.pretrained_path is None:
    from extender.entmax_roberta import EntmaxRobertaForMaskedLM
    # roberta_base = RobertaForMaskedLM.from_pretrained('roberta-base')
    my_roberta = EntmaxRobertaForMaskedLM.from_pretrained('roberta-base')
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
    model = EntmaxRobertaForMaskedLM.from_pretrained(model_path)

    logger.info(f'Pretraining roberta-base-{model_args.max_pos} ... ')
    # training_args.max_steps = 3   ## <<<<<<<<<<<<<<<<<<<<<<<< REMOVE THIS <<<<<<<<<<<<<<<<<<<<<<<<
    pretrain_and_evaluate(training_args, model, tokenizer, eval_only=False, model_path=training_args.output_dir)

    logger.info(f'Saving model to {model_path}')
    model.save_pretrained(model_path)

else:
    model_path = args.pretrained_path
    logger.info(f'Loading the model from {model_path}')
    tokenizer = RobertaTokenizerFast.from_pretrained(model_path)
    # tokenizer = RobertaTokenizer.from_pretrained(model_path)
    model = SimulatedRobertaForMaskedLM.from_pretrained(
        model_path,
        variant=args.attn_variant,
        projections_path=args.projections_path,
        centroids_path=args.centroids_path,
        window_size=args.window_size,
        cls_size=args.cls_size,
        top_clusters=args.top_clusters,
        bucket_size=args.bucket_size,
        compute_graph_stats=False,
        block_size=args.block_size,
        num_random_blocks=args.num_random_blocks
    )

    # re-init because it seems that huggingface is reinitializing params that are not in loaded_model.state_dict
    from pprint import pprint
    pprint(vars(args))
    if args.attn_variant != 'regular':
        print('loading projections from {}'.format(args.projections_path))
        projections = torch.load(args.projections_path, map_location=lambda storage, loc: storage)
        for i, layer in enumerate(model.base_model.encoder.layer):
            layer.attention.self.set_projections(projections[i])
    if args.attn_variant in ['extender', 'extender_hf', 'extender_rhf', 'routing_trained', 'routing_extended']:
        # centroids = torch.load(model_args.centroids_path, map_location=lambda storage, loc: storage)
        centroids = get_layer_centroids(args.attn_variant, args.centroids_path, 12, 12)
        for i, layer in enumerate(model.base_model.encoder.layer):
            layer.attention.self.set_centroids(centroids[i])

    if args.finetune_part == 'head':
        # Freeze the encoder
        for param in model.base_model.parameters():
            param.requires_grad = False

    elif args.finetune_part == 'all_but_projections':
        # Freeze projections only
        for i, layer in enumerate(model.base_model.encoder.layer):
            if layer.attention.self.headwise_proj_q is not None:
                for param in layer.attention.self.headwise_proj_q.parameters():
                    param.requires_grad = False
            if layer.attention.self.headwise_proj_k is not None:
                for param in layer.attention.self.headwise_proj_k.parameters():
                    param.requires_grad = False
            if layer.attention.self.centroids is not None:
                layer.attention.self.centroids.requires_grad = False

    val_dataset = TextDataset(tokenizer=tokenizer,
                              file_path=training_args.val_datapath,
                              block_size=tokenizer.model_max_length)
    train_dataset = val_dataset
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    trainer = Trainer(model=model, args=training_args, data_collator=data_collator,
                      train_dataset=train_dataset, eval_dataset=val_dataset)

    if not args.timeit:
        # compute graph stats for evaluation
        for i, layer in enumerate(trainer.model.base_model.encoder.layer):
            layer.attention.self.compute_graph_stats = True

    time_start = time.perf_counter()
    eval_loss = trainer.evaluate()
    time_elapsed = time.perf_counter() - time_start

    output_eval_file = os.path.join(
        args.output_dir,
        f"eval_results_entmax_roberta_{args.attn_variant}_{args.finetune_part}_{args.window_size}_{args.timeit}.txt"
    )
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    if not args.timeit:
        result = {'eval_loss': eval_loss['eval_loss']}
        layers = trainer.model.base_model.encoder.layer
        add_stats_to_result_dict(result, layers, metric='sparsities_gold')
        add_stats_to_result_dict(result, layers, metric='sparsities_pred')
        add_stats_to_result_dict(result, layers, metric='precisions')
        add_stats_to_result_dict(result, layers, metric='recalls')
        add_stats_to_result_dict(result, layers, metric='exacts')
        if 'hf' not in args.attn_variant:
            add_stats_to_result_dict(result, layers, metric='entropies_q')
            add_stats_to_result_dict(result, layers, metric='entropies_k')
    else:
        result = {
            'eval_loss': eval_loss['eval_loss'],
            'time_elapsed': time_elapsed,
            'sample_size': len(val_dataset),
            't_by_s': time_elapsed / len(val_dataset)
        }

    print('Saving to: ', output_eval_file)
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results entmax_roberta *****")
        for key, value in result.items():
            logger.info("  %s = %s", key, value)
            writer.write("%s = %s\n" % (key, value))
