import math

import torch


def quantize(x,
             bucket_size,
             mask,
             enforce_equal_size=False,
             return_lower_bounds=False,
             temperature=1.,
             ):
    """
    Args:
        x: projected query or key (batch, heads, n, projections)
        bucket_size: number of tokens per bucket
        mask: tensor (batch, seq_len) - necessary for masking padding
        enforce_equal_size: makes all buckets with the same number of tokens.
            If not, buckets will have tokens falling within the same interval
        return_lower_bounds: return also a tensor with the lowest value in each
            bucket
        temperature: coefficient to multiply projected values before tanh
            (only used with variable-sized buckets)

    Return:
        (buckets, inds, lower_bounds[Optional]):
            buckets (batch, heads, n, projections): the bucket each token has
                been assigned to at each projection
            lower_bounds [Optional] (num_intervals, rounds): the lowest value
                assigned to each bucket
    """
    batch_size, num_heads, n, num_projections = x.shape

    x = x.masked_fill(~mask.view(batch_size, 1, n, 1), float('inf'))
    num_intervals = math.ceil(n / bucket_size)

    if enforce_equal_size:
        modulo = n % bucket_size
        if modulo != 0:
            # pad
            shape = [batch_size, num_heads, bucket_size - modulo, num_projections]
            padding = torch.full(shape, float('inf')).to(x.device)
            x = torch.cat([x, padding], dim=2)

        # each round will have all tokens sorted by value
        inds = x.argsort(2)
        rev_inds = inds.argsort(2)
        buckets = rev_inds // bucket_size

        # remove the padding needed for having a multiple of `intervals`
        buckets = buckets[:, :, :n]

    else:
        step = 2 / num_intervals
        num_boundaries = num_intervals - 1
        boundaries = torch.linspace(-1 + step, 1 - step, num_boundaries)
        boundaries = boundaries.to(x.device)
        # torch.bucketize fails on tensors requiring gradient
        x = torch.tanh(temperature * x.detach())
        buckets = torch.bucketize(x, boundaries)

    if return_lower_bounds:
        raise NotImplementedError

    return buckets


def neighbours_mask(len_q, len_k, window_size, causal=False, device=None):
    """Mask for neighbour positions.

    Args:
        len_q (int): queries seq len
        len_k (int): keys seq len
        window_size (int): how many elements to be considered as valid around
            the ith element (including ith).

    Returns:
        torch.Tensor: (size, size)
    """
    z = torch.ones(len_q, len_k, dtype=torch.uint8, device=device)
    u_diag = 1 if causal else 1 + window_size // 2
    l_diag = - window_size // 2
    mask = torch.triu(z, diagonal=u_diag) + torch.tril(z, diagonal=l_diag)
    return (z - mask).bool()


def compute_sparsity(p, pad_mask, causal_mask=None, reduction='mean'):
    """
    Compute the sparsity of the distribution `p`

    Args:
        p: float tensor (batch, heads, n, n)
        pad_mask: bool tensor (batch, n)
        causal_mask: bool tensor  (n, n)
        reduction: str. `mean` will do macro average of samples in the batch (default)
                        `None` will return values for each sample
    """
    # we would need two src and trg pad_masks to calc cross attention stats perfectly
    # pairwise_mask = tgt_pad_mask.unsqueeze(-1) & src_pad_mask.unsqueeze(-2)
    pairwise_mask = pad_mask.unsqueeze(-1) & pad_mask.unsqueeze(1)
    pairwise_mask = pairwise_mask.unsqueeze(1)
    if causal_mask is not None:
        pairwise_mask = pairwise_mask & causal_mask.unsqueeze(0).unsqueeze(1)

    positive_p = (p > 0).masked_fill(~pairwise_mask, False)
    total_per_head = pairwise_mask.sum(-1).sum(-1).float()
    num_selected_per_head = positive_p.sum(-1).sum(-1).float()
    positive_ratio_per_head = num_selected_per_head / total_per_head
    sparsity = 1 - positive_ratio_per_head
    if reduction == 'mean':  # average of samples in the batch
        return sparsity.mean(0)
    return sparsity


def compute_recall(pred_p, gold_p, pad_mask, causal_mask=None, reduction='mean'):
    """
    Compute the recall between pred_p and gold_p selections

    Args:
        pred_p: float tensor (batch, heads, n, n)
        gold_p: float tensor (batch, heads, n, n)
        pad_mask: bool tensor (batch, n)
        causal_mask: bool tensor  (n, n)
        reduction: str. `mean` will do macro average of samples in the batch (default)
                        `None` will return values for each sample
    """
    pairwise_mask = pad_mask.unsqueeze(-1) & pad_mask.unsqueeze(1)
    pairwise_mask = pairwise_mask.unsqueeze(1)
    if causal_mask is not None:
        pairwise_mask = pairwise_mask & causal_mask.unsqueeze(0).unsqueeze(1)

    # compute (micro-average) recall for each batch item then average
    gold_p = (gold_p > 0).masked_fill(~pairwise_mask, False)
    pred_p = (pred_p > 0).masked_fill(~pairwise_mask, False)
    matches = pred_p & gold_p
    matches_per_head = matches.sum(-1).sum(-1).float()
    total_per_head = gold_p.sum(-1).sum(-1).float()
    recall_per_head = matches_per_head / total_per_head
    if reduction == 'mean':  # average of samples in the batch
        return recall_per_head.mean(0)
    return recall_per_head


def compute_precision(pred_p, gold_p, pad_mask, causal_mask=None, reduction='mean'):
    """
    Compute the precision between pred_p and gold_p selections for each head

    Args:
        pred_p: float tensor (batch, heads, n, n)
        gold_p: float tensor (batch, heads, n, n)
        pad_mask: bool tensor (batch, n)
        causal_mask: bool tensor  (n, n)
        reduction: str. `mean` will do macro average of samples in the batch (default)
                        `None` will return values for each sample
    """
    pairwise_mask = pad_mask.unsqueeze(-1) & pad_mask.unsqueeze(1)
    pairwise_mask = pairwise_mask.unsqueeze(1)
    if causal_mask is not None:
        pairwise_mask = pairwise_mask & causal_mask.unsqueeze(0).unsqueeze(1)

    gold_p = (gold_p > 0).masked_fill(~pairwise_mask, False)
    pred_p = (pred_p > 0).masked_fill(~pairwise_mask, False)
    matches = pred_p & gold_p
    matches_per_head = matches.sum(-1).sum(-1).float()
    total_per_head = pred_p.sum(-1).sum(-1).float()
    precision_per_head = matches_per_head / total_per_head
    if reduction == 'mean':  # average of samples in the batch
        return precision_per_head.mean(0)
    return precision_per_head


def compute_exact_fraction(pred_p, gold_p, pad_mask, causal_mask=None, reduction='mean'):
    """
    Compute the number of queries for which the entmax graph was recovered exact,
    i.e., when we get 100% recall due to entmax top-k property:

    a = entmax([1.3, 20.0, 19.5, 1.0]) = [0.0000, 0.6740, 0.3260, 0.0000]
    b = entmax([0.0, 20.0, 19.5, 0.0]) = [0.0000, 0.6740, 0.3260, 0.0000]

    Args:
        pred_p: float tensor (batch, heads, n, n)
        gold_p: float tensor (batch, heads, n, n)
        pad_mask: bool tensor (batch, n)
        causal_mask: bool tensor  (n, n)
    """
    pairwise_mask = pad_mask.unsqueeze(-1) & pad_mask.unsqueeze(1)
    pairwise_mask = pairwise_mask.unsqueeze(1)
    if causal_mask is not None:
        pairwise_mask = pairwise_mask & causal_mask.unsqueeze(0).unsqueeze(1)

    # compute (micro-average) recall for each batch item then average
    gold_p = (gold_p > 0).masked_fill(~pairwise_mask, False)
    pred_p = (pred_p > 0).masked_fill(~pairwise_mask, False)
    matches = pred_p & gold_p
    matches_per_query = matches.sum(-1).float()
    total_per_query = gold_p.sum(-1).float()
    recall_per_query = matches_per_query / total_per_query
    # get fraction of exact entmax distribution for each query vector
    # recall == 1.0 means an exact recovery of entmax dist for that vector
    exact_per_query = recall_per_query == 1.0
    valid_exact_per_query = exact_per_query.masked_fill(~pad_mask.unsqueeze(1), False)
    lengths = pad_mask.sum(-1).unsqueeze(-1).float()
    exact_per_head = valid_exact_per_query.sum(-1) / lengths
    if reduction == 'mean':  # average of samples in the batch
        return exact_per_head.mean(0)
    return exact_per_head


def compute_clusters_entropy(q_clustered, k_clustered, num_centroids, pad_mask, reduction='mean'):
    """
    Compute the clusters entropy across all projection rounds; i.e., as if queries
    could look at any key it shared at least one cluster with.

    Args:
        q_clustered: float tensor (batch, heads, n, num_runs)
        k_clustered: float tensor (batch, heads, n, num_runs)
        num_centroids: int
        pad_mask: bool tensor (batch, n)
        reduction: str. `mean` will do macro average of samples in the batch (default)
                        `None` will return values for each sample
    """
    lengths = pad_mask.int().sum(-1)
    arange_centroids = torch.arange(num_centroids).view(1, 1, 1, 1, -1).to(q_clustered.device)
    q_counts = (q_clustered.unsqueeze(-1) == arange_centroids).sum(dim=2)
    k_counts = (k_clustered.unsqueeze(-1) == arange_centroids).sum(dim=2)
    q_cluster_probs = q_counts.float() / lengths.view(-1, 1, 1, 1)
    k_cluster_probs = k_counts.float() / lengths.view(-1, 1, 1, 1)

    change_log_base = torch.log(torch.tensor(num_centroids, dtype=torch.float, device=q_clustered.device))
    q_logs = torch.log(q_cluster_probs)
    k_logs = torch.log(k_cluster_probs)
    q_logs[q_logs == float('-inf')] = 0
    k_logs[k_logs == float('-inf')] = 0
    q_cluster_ent = - (q_cluster_probs * q_logs / change_log_base).sum(-1)
    k_cluster_ent = - (k_cluster_probs * k_logs / change_log_base).sum(-1)

    if reduction == 'mean':
        entropies = {
            'q_avg': q_cluster_ent.mean(-1).mean(0),  # average runs and then average samples
            'k_avg': k_cluster_ent.mean(-1).mean(0),
            'q_min': q_cluster_ent.mean(-1).min(0)[0],  # average runs and then min across samples
            'k_min': k_cluster_ent.mean(-1).min(0)[0]
        }
    else:
        entropies = {
            'q_avg': q_cluster_ent.mean(-1),  # average runs only
            'k_avg': k_cluster_ent.mean(-1),
            'q_min': q_cluster_ent.min(-1)[0],  # mins across runs
            'k_min': k_cluster_ent.min(-1)[0]
        }
    return entropies
