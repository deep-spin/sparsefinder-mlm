import argparse
import torch
from tqdm.auto import tqdm
from torch.utils.data.dataloader import DataLoader
from transformers import RobertaTokenizerFast, TextDataset
from extender.entmax_roberta import EntmaxRobertaForMaskedLM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wikipath", type=str, default="/home/agois/longformer_stuff/",
                        help="path to dir containing wikitext-103-raw/")
    parser.add_argument("--out", default="tmp", type=str, help="path to out file, will append -enc.pt")
    parser.add_argument("--model_path", required=True, help="path to trained entmax-model folder")
    parser.add_argument("--nsamples", default=500, type=int,
                        help="how many training samples to re-use for"
                             "dumping Q K vectors")
    parser.add_argument("--batchsize", default=10, type=int)
    parser.add_argument('--cpu', action='store_true', help='Use CPU only')
    args = parser.parse_args()

    args.val_datapath = args.wikipath + 'wikitext-103-raw/wiki.valid.raw'
    args.train_datapath = args.wikipath + 'wikitext-103-raw/wiki.train.raw'
    args.out += '_enc-attn.pt'

    return args


def format_qks_old(attn):
    """
    expects tuple with 1 elem per roberta-layer
    each elem is dict with keys 'src_q' 'src_k', each has 1 tensor
    tensor dims: (batch, head, length, dim)
    output is 2 tensors with dims: (batch, layer, head, length, dim)
    """
    qs, ks = attn[0]['src_q'].unsqueeze(1), attn[0]['src_k'].unsqueeze(1)

    for i in range(1, len(attn)):
        q, k = attn[i]['src_q'].unsqueeze(1), attn[i]['src_k'].unsqueeze(1)
        qs = torch.cat((qs, q), dim=1)
        ks = torch.cat((ks, k), dim=1)

    return {'src_q': qs, 'src_k': ks}


def format_qks(attn):
    """
    expects tuple with 1 elem per roberta-layer
    each elem is dict with keys 'src_q' 'src_k', each has 1 tensor
    tensor dims: (batch, head, length, dim)
    output is 2 lists of tensors with dims: (layer, head, length, dim)
    1 element in list is 1 batch element
    """
    qs, ks = attn[0]['src_q'].unsqueeze(1), attn[0]['src_k'].unsqueeze(1)

    for i in range(1, len(attn)):
        q, k = attn[i]['src_q'].unsqueeze(1), attn[i]['src_k'].unsqueeze(1)
        qs = torch.cat((qs, q), dim=1)
        ks = torch.cat((ks, k), dim=1)

    qs_list, ks_list = [], []
    assert qs.shape[0] == ks.shape[0]
    for i in range(qs.shape[0]):  # get rid of batch dim: 1 sample per list-element instead
        qs_list.append(qs[i, :, :, :, :])
        ks_list.append(ks[i, :, :, :, :])

    return {'src_q': qs_list, 'src_k': ks_list}


def get_attns(model, data_loader, args):
    device = model.device

    all_lengths_src = []
    enc_q = []
    enc_k = []
    with torch.no_grad():
        computed_samples = 0
        for inputs in tqdm(data_loader, total=args.nsamples/args.batchsize):
            inputs = inputs.to(device)
            if computed_samples >= args.nsamples:
                break
            output, qks = model(input_ids=inputs, output_attentions=True)
            attn = format_qks(qks)

            # todo lengths currently not ready for different lengths within same batch
            lengths_src = torch.Tensor([inputs.shape[1]]).repeat(inputs.shape[0]).to(inputs.device)
            q = [a.cpu() for a in attn['src_q']]
            k = [a.cpu() for a in attn['src_k']]

            all_lengths_src.append(lengths_src.cpu())
            enc_q.extend(q)
            enc_k.extend(k)

            computed_samples += inputs.shape[0]
    all_lengths_src = torch.cat(all_lengths_src, dim=0)

    return all_lengths_src, enc_q, enc_k


def main():
    args = parse_args()

    tokenizer = RobertaTokenizerFast.from_pretrained(args.model_path)
    model = EntmaxRobertaForMaskedLM.from_pretrained(args.model_path)
    if not args.cpu:
        model.cuda()
    train_dataset = TextDataset(tokenizer=tokenizer, file_path=args.train_datapath, block_size=tokenizer.max_len)
    # train_dataset = TextDataset(tokenizer=tokenizer, file_path=args.val_datapath, block_size=tokenizer.max_len)  # dump validation-representations for evaluation only
    data_loader = DataLoader(train_dataset, batch_size=args.batchsize)

    assert args.nsamples <= len(train_dataset), \
        "do not request more samples than full training set"

    all_lengths_src, enc_q, enc_k = get_attns(model, data_loader, args)

    print('Saving attentions from the encoder in {}...'.format(args.out))
    data = {'length_src': all_lengths_src, 'q': enc_q, 'k': enc_k}
    torch.save(data, args.out)


if __name__ == '__main__':
    main()
