import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np


class QKDataset(object):
    """
    Create a dataset with real query and key vectors.

    Each batch this class returns has Q and K from the heads in a given layer
    within an actual batch. Further batches will return the other layers and
    eventually a new actual batch.
    """

    def __init__(self, path, batch_size=16, to_cuda=True, eval_mode=False):
        data = torch.load(path)
        q = data['q']
        k = data['k']
        lengths = data['length_src']

        # the contents of Q and K are a list of tensors of different lengths
        # with shape (layer, head, length, dim)
        q = [item.transpose(0, 2) for item in q]
        k = [item.transpose(0, 2) for item in k]
        q = pad_sequence(q, batch_first=True).transpose(1, 3)
        k = pad_sequence(k, batch_first=True).transpose(1, 3)

        if not eval_mode:
            # leave 10% of the data for evaluating and the rest for training
            self.num_eval = int(0.1 * len(q))
            self.num_train = len(q) - self.num_eval
        else:
            # evaluate all data
            self.num_eval = len(q)
            self.num_train = 0

        # q and k are tensors of shape (num_items, layer, head, length, dim)
        self.train_q = q[:self.num_train]
        self.train_k = k[:self.num_train]
        self.train_length = lengths[:self.num_train]
        self.eval_q = q[self.num_train:]
        self.eval_k = k[self.num_train:]
        self.eval_length = lengths[self.num_train:]

        self.to_cuda = to_cuda
        self.d = q.shape[-1]
        self.num_layers = q.shape[1]
        self.num_heads = q.shape[2]
        self.batch_size = batch_size
        self.print_stats()

    def print_stats(self):
        print('batch size:', self.batch_size)
        print('train sents:', self.num_train)
        print('train tokens:', self.train_length.sum().item())
        print('train batches:', self.num_train // self.batch_size)
        print('eval sents:', self.num_eval)
        print('eval tokens:', self.eval_length.sum().item())
        print('eval batches:', self.num_eval // self.batch_size)

    def get_train_batch(self, layer: int):
        """
        Returns:
            A tuple of three tensors:
            q and k (batch, num_heads, max_length, dim)
            lengths (batch,)
        """
        for batch_start in range(0, self.num_train, self.batch_size):
            batch_end = batch_start + self.batch_size  # last batch may be smaller
            q = self.train_q[batch_start:batch_end, layer]
            k = self.train_k[batch_start:batch_end, layer]
            length = self.train_length[batch_start:batch_end]
            if self.to_cuda and torch.cuda.is_available():
                q = q.cuda()
                k = k.cuda()
                length = length.cuda()
            yield q, k, length

    def get_eval_batch(self, layer: int):
        """
        Returns:
            A tuple of three tensors:
            q and k (batch, num_heads, max_length, dim)
            lengths (batch,)
        """
        for batch_start in range(0, self.num_eval, self.batch_size):
            batch_end = batch_start + self.batch_size  # last batch may be smaller
            q = self.eval_q[batch_start:batch_end, layer]
            k = self.eval_k[batch_start:batch_end, layer]
            length = self.eval_length[batch_start:batch_end]
            if self.to_cuda and torch.cuda.is_available():
                q = q.cuda()
                k = k.cuda()
                length = length.cuda()
            yield q, k, length

    def get_clustering_data(self, layer, proj_q, proj_k, dataset="train"):
        if dataset == "train":
            get_batch_fn = self.get_train_batch
        elif dataset == "valid":
            get_batch_fn = self.get_eval_batch
        else:
            raise ValueError
        data = [[] for _ in range(self.num_heads)]
        with torch.no_grad():
            for q, k, length in get_batch_fn(layer):
                batch_size, num_heads, seq_len, _ = q.shape
                # (bs, nh, seq_len, d) -> (bs, nh, seq_len, r)
                q_low = proj_q(q)
                k_low = proj_k(k)
                # (bs, nh, seq_len, r) -> (nh, bs, seq_len, r)
                q_low = q_low.transpose(0, 1)
                k_low = k_low.transpose(0, 1)
                # (seq_len) -> (1, seq_len) -> (bs, seq_len)
                ar = torch.arange(seq_len, device=q.device)
                ar = ar.unsqueeze(0).expand(batch_size, -1)
                ix = ar < length.unsqueeze(1)
                for h in range(num_heads):
                    # (nh, bs, seq_len, r) -> (nh, bs*var_seq_len, r)
                    q_low_vectors = q_low[h, ix]
                    k_low_vectors = k_low[h, ix]
                    data[h].append(q_low_vectors.cpu().detach())
                    data[h].append(k_low_vectors.cpu().detach())
            # (nh, num_queries + num_keys, r)
            data = torch.stack([torch.cat(head) for head in data])
        data = data.cpu().numpy()
        return data
