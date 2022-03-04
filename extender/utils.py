import random

import numpy as np
import torch
from entmax import entmax15


def configure_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


def get_window_positions(n, window):
    """
    Create a tensor (n, window) such that the i-th row has the indices of the
    columns attended by it. E.g.,

    [[2, 0, 1],  # shifts as padding
     [0, 1, 2],
     [1, 2, 3],
     [2, 3, 4],
     ...]
    """
    half_window = window // 2

    # form indices for columns in an attention window pattern
    # e.g. [0, 1, 2], [1, 2, 3], [2, 3, 4] etc
    r = torch.arange(n).view(-1, 1)
    attended = r + torch.arange(-half_window, half_window + 1)

    # make the windows at the first and last few words the same size as in
    # the middle
    attended[attended < 0] += window
    attended[attended >= n] -= window

    return attended


def get_length_mask(lengths, max_length=None):
    """Create a (batch_size, max_length) boolean mask
    True for true positions and False for padding"""
    if max_length is None:
        max_length = lengths.max()
    r = torch.arange(max_length).unsqueeze(0).to(lengths.device)
    mask = r < lengths.unsqueeze(-1)
    return mask


def get_mask(num_positive, length):
    r = torch.arange(length).unsqueeze(0).to(num_positive.device)
    mask = r < num_positive.unsqueeze(1)
    return mask


def get_key_mask(num_positive, length=None):
    """Mask positions after num_positions, for each token/head/sample"""
    assert num_positive.ndim == 3
    if length is None:
        length = num_positive.shape[-1]
    r = torch.arange(length).view(1, 1, 1, -1).to(num_positive.device)
    mask = r < num_positive.unsqueeze(-1)
    return mask


def neighbours_mask(size, window_size):
    """Mask for neighbour positions.
    Args:
        size (int): squared tensor size
        window_size (int): how many elements to be considered as valid around
            the ith element (including ith).
    Returns:
        torch.Tensor: (size, size)
    """
    z = torch.ones(size, size, dtype=torch.uint8)
    mask = (torch.triu(z, diagonal=1 + window_size // 2)
            + torch.tril(z, diagonal=- window_size // 2))
    return z - mask


def dot_product_and_mask(q, k, lengths=None, mask_value=-float('inf')):
    """
    Args:
        q: tensor (batch, heads, n, dim)
        k: tensor (batch, heads, n, dim)
        lengths: tensor (batch,)
        mask_value: value for padding positions
    """
    # (batch, heads, n, n)
    dots = q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5

    if lengths is None:
        return dots

    # mask out padding positions - mask is (batch, n)
    mask = get_length_mask(lengths, q.shape[2])
    dots.masked_fill_(~mask.unsqueeze(1).unsqueeze(2), mask_value)

    return dots


def compute_gold_sparsity(p, lengths):
    """
    Compute the gold sparsity of the distribution `p`

    Args:
        p: tensor (batch, heads, n, n)
        lengths: tensor (batch,)
    """
    max_length = p.shape[2]

    # number of non-null keys for each query
    positive_p = p > 0
    mask = get_length_mask(lengths, max_length)
    keys_per_query = positive_p.sum(-1).masked_fill(~mask.unsqueeze(1), 0)
    num_possible = (lengths ** 2).sum()
    positive_ratio = keys_per_query.sum(-1).float() / num_possible
    sparsity = 1 - positive_ratio
    return sparsity.mean(0)


def get_ground_truth(dots, lengths, p=None):
    """
    Args:
        dots: tensor (batch, heads, n, n)
        lengths: tensor (batch,)
        p: optional pre-computed sparse attention
    """
    if p is None:
        p = entmax15(dots, dim=-1)

    # number of non-null keys for each query
    num_positive = (p > 0).sum(-1)
    inds = dots.argsort(-1, descending=True)

    # zero positive counts past sentence end
    mask = get_length_mask(lengths, p.shape[2])
    num_positive.masked_fill_(~mask.unsqueeze(1), 0)

    return inds, num_positive


def update_mean(current_mean, computed_samples, batch_mean, batch_size):
    """
    Computes an accumulated average in O(1).
    """
    assert (current_mean is None and computed_samples == 0) or \
           (current_mean is not None and computed_samples > 0), \
        '"current_mean is None" requires "computed samples==0" and vice-versa'
    if current_mean is None:
        return batch_mean
    else:
        updated_mean = (current_mean * computed_samples + batch_mean * batch_size) / (computed_samples + batch_size)
        return updated_mean


def append_tensor_cpu(original_tensor, batch):
    """Concat tensors in cpu"""
    batch = batch.cpu()
    if original_tensor is None:
        return batch
    else:
        return torch.cat([original_tensor, batch], dim=0)


def update_graph_stats(graph_stats, computed_samples, batch_graph_stats, batch_size):
    assert (graph_stats is None and computed_samples == 0) or \
           (graph_stats is not None and computed_samples > 0), \
        '"graph_stats is None" requires "computed samples==0" and vice-versa'

    batch_graph_stats['edge_distances'] = batch_graph_stats['edge_distances'].cpu()
    if graph_stats is None:
        return batch_graph_stats
    else:
        graph_stats['sparsity'] = update_mean(graph_stats['sparsity'], computed_samples,
                                              batch_graph_stats['sparsity'], batch_size)
        graph_stats['edge_distances'] = torch.cat([graph_stats['edge_distances'],
                                                   batch_graph_stats['edge_distances']], dim=0)
        return graph_stats

