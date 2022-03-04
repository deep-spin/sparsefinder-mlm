import argparse

from torch.utils.data import DataLoader

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
parser.add_argument("--finetune_part", type=str, default="all_but_projections")
args = parser.parse_args()

# print(args.outdir)
# print(args.wikipath)
import sys
sys.path.insert(0, ".")  # gambi to get extender import

from extender.roberta_simulated import SimulatedRobertaForMaskedLM

import logging
import os
import torch
from dataclasses import dataclass, field
from transformers import RobertaTokenizerFast, TextDataset, DataCollatorForLanguageModeling, Trainer
from transformers import TrainingArguments, HfArgumentParser

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_layer_centroids(attn_variant, centroids_path, num_layers, num_heads):
    from extender.sim_routing_transformer import KmeansAttention, load_kmeans_obj_from_state_dit

    layers_centroids = [None for _ in range(num_layers)]

    if attn_variant == 'extender':
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
    '--evaluate_during_training',
    '--do_train',
    '--do_eval',
])
training_args.val_datapath = args.wikipath + 'wikitext-103-raw/wiki.valid.raw'
training_args.train_datapath = args.wikipath + 'wikitext-103-raw/wiki.train.raw'


activation_str = args.activation
model_path = args.pretrained_path
logger.info(f'Loading the model from {model_path}')
tokenizer = RobertaTokenizerFast.from_pretrained(model_path)
model = SimulatedRobertaForMaskedLM.from_pretrained(
    model_path,
    variant=args.attn_variant,
    projections_path=args.projections_path,
    centroids_path=args.centroids_path,
    window_size=args.window_size,
    cls_size=args.cls_size,
    top_clusters=args.top_clusters,
    bucket_size=args.bucket_size,
    compute_graph_stats=False
)

# re-init because it seems that huggingface is reinitializing params that are not in loaded_model.state_dict
from pprint import pprint
pprint(vars(args))

if args.attn_variant != 'regular':
    print('loading projections from {}'.format(args.projections_path))
    projections = torch.load(args.projections_path, map_location=lambda storage, loc: storage)
    for i, layer in enumerate(model.base_model.encoder.layer):
        layer.attention.self.set_projections(projections[i])

if args.attn_variant in ['extender', 'routing_trained', 'routing_extended']:
    print('loading centroids from {}'.format(args.centroids_path))
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

# compute graph stats for evaluation
for i, layer in enumerate(model.base_model.encoder.layer):
    layer.attention.self.compute_graph_stats = True


block_size = 224
# block_size = tokenizer.max_len
val_dataset = TextDataset(tokenizer=tokenizer,
                          file_path=training_args.val_datapath,
                          block_size=block_size)
# val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True)

train_dataset = val_dataset
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
trainer = Trainer(model=model, args=training_args, data_collator=data_collator,
                  train_dataset=train_dataset, eval_dataset=val_dataset, prediction_loss_only=True,)
model.eval()
eval_dataloader = trainer.get_eval_dataloader(val_dataset)
plotted_samples = 0
from utils_plotting import plot_heatmap, plot_graph

manual_examples = [
    'find the collector rate on the incentive rate table .',
    'furthermore we must be ruthless dealing with corruption in the eu institutions .',
    'the governor of tyumen region will speak at the forum .',
    'however, what is armani polo ?',
    'you wonder what more people expect .',
    'There have been a large number of examples published where the requisite cation is arrive at by a variety of rearrangements .',
    'although the lobsters caught in lobster pots are usually 23 – 38 cm ( 9 – 15 in ) long and weigh 0 @.@ 7 – 2 @.@ 2 kg ( 1 @.@ 5 – 4 @.@ 9 lb )',
    'The cat sat on the mat .',
    'Mary did not slap the green witch .',
    'Mary did not slap the green witch !',
    'Did Mary slap the green witch ?',
    'there is also an economic motive .',
    'maybe the dress code is too stuffy .',
    'found in Taiwan . the wingspan is 24 - 28 mm .',
    'It goes on to plug a few diversified Fidelity funds by name .',
    'It declined to discuss its plans for upgrading its current product line',
    'The complicated language in the huge new law has muddied the fight .',
    'The 45-year-old former General Electric Co. executive figures it will be easier this time .',
    "Not his autograph ; power-hitter McGwire 's .",
    "many employees are working at its giant Renton, Wash. , plant .",
    'This market has been very badly damaged .',
    "But in the absence of panicky trading , its presence was never overtly felt .",
    "Prices of Treasury bonds tumbled in moderate to active trading .",
    "Short-term interest rates fell yesterday at the government 's weekly Treasury bill action .",
    "with Kim today as she got some expert opinions on the damage to her home",
    'joining peace talks between Israel and the Palestinians . The negotiations are',
]


saved_inputs = []
saved_attns = []

for mex in manual_examples:
    encoding = tokenizer(mex, return_tensors='pt', padding=False, truncation=False)
    input = encoding['input_ids']
    valid_seq_len = input.shape[-1]
    labels = torch.ones(valid_seq_len) * -100
    labels = labels.unsqueeze(0).long()
    inputs = {'input_ids': input, 'labels': labels}
    outputs = trainer.prediction_step(model, inputs, prediction_loss_only=True)

    gold_p = torch.stack([l.attention.self.gold_p for l in model.base_model.encoder.layer]).float()
    pred_p = torch.stack([l.attention.self.pred_p for l in model.base_model.encoder.layer]).float()
    pred_entmax_p = torch.stack([l.attention.self.pred_entmax_p for l in model.base_model.encoder.layer]).float()
    sparsities_gold = torch.stack([l.attention.self.sparsities_gold[0] for l in model.base_model.encoder.layer])
    sparsities_pred = torch.stack([l.attention.self.sparsities_pred[0] for l in model.base_model.encoder.layer])
    recalls = torch.stack([l.attention.self.recalls[0] for l in model.base_model.encoder.layer])

    gold_p = gold_p.transpose(0, 1).cpu().numpy()
    pred_p = pred_p.transpose(0, 1).cpu().numpy()
    pred_entmax_p = pred_entmax_p.transpose(0, 1).cpu().numpy()
    sparsities_gold = sparsities_gold.transpose(0, 1).cpu().numpy()
    sparsities_pred = sparsities_pred.transpose(0, 1).cpu().numpy()
    recalls = recalls.transpose(0, 1).cpu().numpy()

    batch_idx = 0
    tgt_ids = inputs["input_ids"][batch_idx].tolist()
    tokens = tokenizer.convert_ids_to_tokens(tgt_ids)
    tokens = [t.replace('Ġ', '') for t in tokens]

    tokens = tokens[:valid_seq_len]

    saved_inputs.append(tokens)
    saved_attns.append((gold_p[batch_idx], pred_p[batch_idx], pred_entmax_p[batch_idx]))

    output_path = 'plots/ex_{}_gold_{}l_{:.2f}s_{:.2f}r.png'.format(
        plotted_samples,
        len(tokens),
        sparsities_gold[batch_idx].mean(),
        recalls[batch_idx].mean()
    )
    print('plotting to {}'.format(output_path))
    # plot_heatmap(gold_p[batch_idx], tokens, tokens, output_path=output_path)

    output_path = 'plots/ex_{}_pred_mask_{}l_{:.2f}s_{:.2f}r.png'.format(
        plotted_samples,
        len(tokens),
        sparsities_pred[batch_idx].mean(),
        recalls[batch_idx].mean()
    )
    print('plotting to {}'.format(output_path))
    # plot_heatmap(pred_p[batch_idx], tokens, tokens, output_path=output_path)

    output_path = 'plots/ex_{}_pred_entmax_{}l_{:.2f}s_{:.2f}r.png'.format(
        plotted_samples,
        len(tokens),
        sparsities_pred[batch_idx].mean(),
        recalls[batch_idx].mean()
    )
    print('plotting to {}'.format(output_path))
    # plot_heatmap(pred_entmax_p[batch_idx], tokens, tokens, output_path=output_path)
    plotted_samples += 1

    for i, layer in enumerate(model.base_model.encoder.layer):
        layer.attention.self.reset_stats_attrs()


try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

def load_object(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def save_object(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)




output_path = '/media/hdd2/mtreviso/longformer_saved_attns/{}_{}_{}.pickle'.format(args.attn_variant, args.window_size, args.bucket_size)  # noqa
obj = {'inputs': saved_inputs, 'attns': saved_attns}
print('saving to ', output_path)
save_object(obj, output_path)

#
#
# for inputs in eval_dataloader:
#     outputs = trainer.prediction_step(model, inputs, prediction_loss_only=True)
#     print('Loss: {:.4f}'.format(outputs[0]), end='\r')
#
#     gold_p = torch.stack([l.attention.self.gold_p for l in model.base_model.encoder.layer]).float()
#     pred_p = torch.stack([l.attention.self.pred_p for l in model.base_model.encoder.layer]).float()
#     pred_entmax_p = torch.stack([l.attention.self.pred_entmax_p for l in model.base_model.encoder.layer]).float()
#     sparsities_gold = torch.stack([l.attention.self.sparsities_gold[0] for l in model.base_model.encoder.layer])
#     sparsities_pred = torch.stack([l.attention.self.sparsities_pred[0] for l in model.base_model.encoder.layer])
#     recalls = torch.stack([l.attention.self.recalls[0] for l in model.base_model.encoder.layer])
#
#     gold_p = gold_p.transpose(0, 1).cpu().numpy()
#     pred_p = pred_p.transpose(0, 1).cpu().numpy()
#     pred_entmax_p = pred_entmax_p.transpose(0, 1).cpu().numpy()
#     sparsities_gold = sparsities_gold.transpose(0, 1).cpu().numpy()
#     sparsities_pred = sparsities_pred.transpose(0, 1).cpu().numpy()
#     recalls = recalls.transpose(0, 1).cpu().numpy()
#
#     batch_idx = 1
#     valid_seq_len = 49
#
#     tgt_ids = inputs["input_ids"][1].tolist()
#     tokens = tokenizer.convert_ids_to_tokens(tgt_ids)
#     tokens = [t.replace('Ġ', '') for t in tokens]
#
#     tokens = tokens[:valid_seq_len]
#     output_path = 'plots/ex_{}_gold_{}l_{:.2f}s_{:.2f}r.png'.format(
#         plotted_samples,
#         len(tokens),
#         sparsities_gold[batch_idx].mean(),
#         recalls[batch_idx].mean()
#     )
#     print('plotting to {}'.format(output_path))
#     plot_heatmap(gold_p[batch_idx], tokens, tokens, output_path=output_path)
#
#     output_path = 'plots/ex_{}_pred_mask_{}l_{:.2f}s_{:.2f}r.png'.format(
#         plotted_samples,
#         len(tokens),
#         sparsities_pred[batch_idx].mean(),
#         recalls[batch_idx].mean()
#     )
#     print('plotting to {}'.format(output_path))
#     plot_heatmap(pred_p[batch_idx], tokens, tokens, output_path=output_path)
#
#     output_path = 'plots/ex_{}_pred_entmax_{}l_{:.2f}s_{:.2f}r.png'.format(
#         plotted_samples,
#         len(tokens),
#         sparsities_pred[batch_idx].mean(),
#         recalls[batch_idx].mean()
#     )
#     print('plotting to {}'.format(output_path))
#     plot_heatmap(pred_entmax_p[batch_idx], tokens, tokens, output_path=output_path)
#
#     import ipdb; ipdb.set_trace()
#
#     batch_size = gold_p.shape[0]
#     for batch_idx in range(batch_size):
#         # recover target tokens
#         tgt_ids = inputs["input_ids"][batch_idx].tolist()
#         tokens = tokenizer.convert_ids_to_tokens(tgt_ids)
#         tokens = [t.replace('Ġ', '') for t in tokens]
#
#         if len(tokens) > 30:
#             continue
#
#         output_path = 'plots/ex_{}_gold_{}l_{:.2f}s_{:.2f}r.png'.format(
#             plotted_samples,
#             len(tokens),
#             sparsities_gold[batch_idx].mean(),
#             recalls[batch_idx].mean()
#         )
#         print('plotting to {}'.format(output_path))
#         plot_heatmap(gold_p[batch_idx], tokens, tokens, output_path=output_path)
#
#         output_path = 'plots/ex_{}_pred_mask_{}l_{:.2f}s_{:.2f}r.png'.format(
#             plotted_samples,
#             len(tokens),
#             sparsities_pred[batch_idx].mean(),
#             recalls[batch_idx].mean()
#         )
#         print('plotting to {}'.format(output_path))
#         plot_heatmap(pred_p[batch_idx], tokens, tokens, output_path=output_path)
#
#         output_path = 'plots/ex_{}_pred_entmax_{}l_{:.2f}s_{:.2f}r.png'.format(
#             plotted_samples,
#             len(tokens),
#             sparsities_pred[batch_idx].mean(),
#             recalls[batch_idx].mean()
#         )
#         print('plotting to {}'.format(output_path))
#         plot_heatmap(pred_entmax_p[batch_idx], tokens, tokens, output_path=output_path)
#         plotted_samples += 1
#
#     if plotted_samples > 100:
#         exit()
#
#     for i, layer in enumerate(model.base_model.encoder.layer):
#         layer.attention.self.reset_stats_attrs()
