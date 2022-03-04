# sparsefinder-mlm
Code for training and evaluating MLM models for our paper "Predicting Attention Sparsity in Transformers"

This code is based on https://github.com/allenai/longformer.


## Installation

To install in dev mode:
```bash
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
python3 setup.py develop
```

## Transformers

All implementations are in `extender/` folder. 

The main code is in: 
- `extender/roberta_simulated.py` for defining a transformer with different types of attention.
- `scripts/pretrain_entmax_roberta_512.py` 
	- for finetuning a RoBERTa model with entmax attention (`--pretrained_path` is `None`)
	- for evaluating a finetuned RoBERTa model with entmax attention (`--pretrained_path` is set to a finetuned model)
- `scripts/extract_entmax_attn.py` for extracting queries and keys tensors from a finetuned model.
