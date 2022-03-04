import math

import torch
from torch import Tensor

from extender.utils import get_length_mask


def compute_bucket_entropy(
    buckets: torch.Tensor,
    return_distribution: bool = True
):
    """
    Computes the entropy for a specific bucketed tensor.

    Args:
        buckets: Tensor (batch, heads, n, projections)
        return_distribution (bool): whether to return the distribution of the
            number of elements per bucket
    """
    # TODO: return entropy per head
    values, counts = torch.unique(buckets, return_counts=True)
    p = counts.float() / counts.sum()
    log_p = torch.log(p)
    log_p[torch.isinf(log_p)] = 0
    entropy = -torch.sum(p * log_p).item()
    if return_distribution:
        return entropy, p
    return entropy


def add_global_attn_buckets(buckets_q, buckets_k):
    b, h, toks, rounds = buckets_q.shape
    cls_idx = 0
    device = buckets_q.device

    extra_bucket1_q = torch.ones(toks).to(device).int()
    extra_bucket1_q[cls_idx] = 0

    extra_bucket1_k = torch.zeros(toks).to(device).int()
    extra_bucket1_k[cls_idx] = 1

    # separate bucket only for CLS self-attention
    # initialize buckets with each tok alone in a bucket
    offset = 1  # reserve idx 0 to use later
    extra_bucket2_q = (torch.arange(toks) + offset).int()
    extra_bucket2_k = (torch.arange(toks) + toks + offset).int()
    extra_bucket2_q[cls_idx] = 0
    extra_bucket2_k[cls_idx] = 0

    extra_buckets_q = torch.stack((extra_bucket1_q, extra_bucket2_q)).transpose(0, 1)
    extra_buckets_k = torch.stack((extra_bucket1_k, extra_bucket2_k)).transpose(0, 1)

    extra_buckets_q = extra_buckets_q.expand(b, h, toks, 2)
    extra_buckets_k = extra_buckets_k.expand(b, h, toks, 2)

    buckets_q = torch.cat((buckets_q, extra_buckets_q), dim=-1)
    buckets_k = torch.cat((buckets_k, extra_buckets_k), dim=-1)

    return buckets_q, buckets_k


def compute_bucket_recall(
    buckets_q,
    buckets_k,
    positive_inds,
    num_positive,
    window_inds=None,
    window_only=False,
    use_global_attn=False
):
    """
    Computes accuracy as the ratio of "gold" keys attended by queries
    considering all buckets.

    Args:
        buckets_q: Tensor (batch, heads, n, projections)
        buckets_k: Tensor (batch, heads, n, projections)
        positive_inds: Tensor (batch, heads, n, k) indices of the keys sorted
            according to relevance to each query; may contain padding
        num_positive: Tensor (batch, heads, n) with the number of actual
            positive key per query
        lengths: Tensor (batch,) with the length of queries
        window_inds: Tensor (n, window_size) with the positions of keys attended
            by each query in all batch items and heads. The keys found here are
            added to the set found via bucketing.
        window_only: ignore buckets
    """
    if window_only:
        assert window_inds is not None, "must provide window_inds, if using only window"
        if use_global_attn:  # discard all buckets except those for global attention
            buckets_q, buckets_k = buckets_q[:, :, :, -2:], buckets_k[:, :, :, -2:]
    # max_num_positive = min(num_positive.max(), positive_inds.shape[1])
    max_num_positive = num_positive.max().item()
    max_num_positive = min(max_num_positive, positive_inds.shape[3])
    positive_inds = positive_inds[:, :, :, :max_num_positive]
    proj_dim = buckets_k.shape[-1]
    n = buckets_q.shape[2]

    # this has the buckets of the top keys for each query
    # (batch, heads, n, max_num_positive, projections)
    inds = positive_inds.unsqueeze(-1).expand(-1, -1, -1, -1, proj_dim)
    expanded_buckets = buckets_k.unsqueeze(3).expand(-1, -1, -1, max_num_positive, -1)
    pos_buckets = torch.gather(expanded_buckets, 2, inds)

    # matches of each query buckets with its corresponding keys
    # (batch, heads, n, num_positive, projections)
    matches = buckets_q.unsqueeze(3) == pos_buckets

    # consider matches in all rounds: (batch, heads, n, max_num_positive)
    if window_only and not use_global_attn:
        # ignore everything else
        found_keys = torch.zeros(matches.shape[:-1]).bool().to(matches.device)
    else:
        found_keys = matches.sum(-1) > 0

    if window_inds is not None:
        w = window_inds.shape[1]
        window_inds5d = window_inds.view(1, 1, n, w, 1)

        # compare (1, 1, n, window, 1) against (batch, heads, n, 1, num_pos)
        window_matches = window_inds5d == positive_inds.unsqueeze(-2)

        # sum across all window positions
        window_matches = window_matches.sum(-2)

        found_keys += window_matches.bool()

    # we are only interested in the top `num_positive` matches (the rest had
    # null attention in entmax)
    mask = get_length_mask(num_positive, max_num_positive)
    found_keys.masked_fill_(~mask, False)

    # it is expected to have a mismatch between per_head.mean() and recall_per_sentence.mean()
    # why?
    # because per_head.mean() is a micro average for each key and then macro-average across keys
    # to get a single number
    # in the latter, we are doing a single micro average for everything
    # to get the same numbers, we need to recover the counts by:
    # old_rec = (recall_per_head*total_positive_keys).sum(-1)/total_positive_keys.sum(-1)
    # old_rec.mean() == recall_per_sentence.mean()
    num_found_keys = found_keys.sum(-1).sum(-1)
    total_positive_keys = num_positive.sum(-1)
    recall_per_head = num_found_keys / total_positive_keys.float()
    return recall_per_head.mean(0)


def compute_bucket_sparsity(
    buckets_q: Tensor,
    buckets_k: Tensor,
    lengths: Tensor,
    window_inds: Tensor = None,
    window_only: bool = False,
    use_global_attn: bool = False
):
    """
    Compute the graph sparsity across all projection rounds; i.e., as if queries
    could look at any key it shared at least one bucket with.

    Args:
        buckets_q: Tensor (batch, heads, n, projections)
        buckets_k: Tensor (batch, heads, n, projections)
        lengths: Tensor (batch, )
        window_inds: Tensor (n, window_size) with the positions attended
           inside a window around each query
        counts_q: Tensor (heads, num_buckets) counting the number of
           queries in each bucket
        counts_k: Tensor (heads, num_buckets) counting the number of
            keys in each bucket
        window_only: ignore buckets
    """
    batch_size = buckets_q.shape[0]
    n = buckets_q.shape[2]

    if window_only and use_global_attn:  # discard all buckets except those for global attention
        buckets_q, buckets_k = buckets_q[:, :, :, -2:], buckets_k[:, :, :, -2:]

    # cross all Q and K to find out when they match
    # shape is (batch, heads, query, key, projection)
    qk = buckets_q.unsqueeze(3) == buckets_k.unsqueeze(2)

    # consider matches across any projection; (batch, heads, query, key)
    if window_only and not use_global_attn:
        qk = torch.zeros(qk.shape[:-1]).bool().to(qk.device)
    else:
        qk = qk.sum(4) > 0

    if window_inds is not None:
        r = torch.arange(n).view(-1, 1).to(qk.device)
        qk[:, :, r, window_inds] = True

    mask = get_length_mask(lengths, n)
    qk.masked_fill_(~mask.view(batch_size, 1, n, 1), False)
    qk.masked_fill_(~mask.view(batch_size, 1, 1, n), False)

    # compute distances between linked Q and K
    # last two columns of nonzero refer to Q and K indices
    nz = qk.nonzero(as_tuple=False)[:, -2:]
    diff = nz[:, 1] - nz[:, 0]

    links_per_head = qk.sum(3).sum(2)
    mean_links = links_per_head.float()
    sparsity = 1 - (mean_links / lengths.unsqueeze(1) ** 2)
    stats = {
        'sparsity': sparsity.mean(0),
        'edge_distances': diff,
    }
    return stats


def compute_bucket_counts(buckets_q, buckets_k, num_buckets, mask):
    buckets = torch.arange(num_buckets).view(1, 1, 1, 1, num_buckets)
    buckets = buckets.to(buckets_k.device)
    presences_k = buckets_k.unsqueeze(-1) == buckets
    presences_q = buckets_q.unsqueeze(-1) == buckets
    mask = mask.view(mask.shape[0], 1, -1, 1, 1)
    presences_k.masked_fill_(~mask, False)
    presences_q.masked_fill_(~mask, False)
    counts_k = presences_k.sum(2)
    counts_q = presences_q.sum(2)
    counts_q_per_head = counts_q.sum(-2).sum(0)
    counts_k_per_head = counts_k.sum(-2).sum(0)
    return counts_q_per_head, counts_k_per_head