# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE
(Bert, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa)."""

import sys
sys.path.insert(0, ".")  # gambi to get extender import

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from transformers import GlueDataset, RobertaTokenizer
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    # new opts
    attn_variant: str = field(
        default="regular",
        metadata={"help": "attention variant to use in test-time",
                  "choices": "regular, extender, bigbird, longformer, openai_sparse, reformer, routing, "
                             "routing_trained, routing_extended, q_fixed, q_dynamic, reformer_multi"}
    )
    projections_path: str = field(
        default=None,
        metadata={"help": "Path to projections in torch format"}
    )
    centroids_path: str = field(
        default=None,
        metadata={"help": "Path to centroids in torch format"}
    )
    window_size: int = field(
        default=None,
        metadata={"help": "Window size to be used with non-regular attn variants (None = no window)"}
    )
    cls_size: int = field(
        default=None,
        metadata={"help": "CLS size to be used with non-regular attn variants "
                          "(None = no cls; 1 = first token; 2 first & last tokens)"}
    )
    top_clusters: int = field(
        default=1,
        metadata={"help": "To how many top-closest clusters a point will be assigned to"}
    )
    finetune_part: str = field(
        default='head',
        metadata={"help": "what to finetune"}
    )


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set seed
    set_seed(training_args.seed)
    tokenizer = RobertaTokenizer.from_pretrained(model_args.model_name_or_path)

    # Get datasets
    train_dataset = GlueDataset(data_args, tokenizer=tokenizer) if training_args.do_train else None
    eval_dataset = GlueDataset(data_args, tokenizer=tokenizer, mode='dev') if training_args.do_eval else None
    test_dataset = GlueDataset(data_args, tokenizer=tokenizer, mode='test') if training_args.do_eval else None

    def get_lens(x_dataset):
        return [sum(ex.attention_mask) for ex in x_dataset]
    train_lens = get_lens(train_dataset)
    print(vars(data_args))
    print('train:', '{}\t{}\t{}'.format(len(train_lens), np.sum(train_lens), np.mean(train_lens)))
    eval_lens = get_lens(eval_dataset)
    print('eval:', '{}\t{}\t{}'.format(len(eval_lens), np.sum(eval_lens), np.mean(eval_lens)))
    test_lens = get_lens(test_dataset)
    print('test:', '{}\t{}\t{}'.format(len(test_lens), np.sum(test_lens), np.mean(test_lens)))
    all_lens = train_lens + eval_lens + test_lens
    print('all:', '{}\t{}\t{}'.format(len(all_lens), np.sum(all_lens), np.mean(all_lens)))
    print('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(
        len(train_lens), np.sum(train_lens), np.mean(train_lens),
        len(eval_lens), np.sum(eval_lens), np.mean(eval_lens),
        len(test_lens), np.sum(test_lens), np.mean(test_lens),
        len(all_lens), np.sum(all_lens), np.mean(all_lens)
    ))
    print(train_dataset[0])
    exit()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
