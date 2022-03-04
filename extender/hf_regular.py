import math

import numpy as np
import torch
from entmax import entmax15
from torch import nn
from transformers.models.big_bird.modeling_big_bird import BigBirdBlockSparseAttention, BigBirdModel

from extender.attn_utils import neighbours_mask, compute_sparsity, compute_precision, compute_recall, \
    compute_exact_fraction
from extender.extender_layers import HeadwiseMLPProjection, HeadwiseLinearProjection


def torch_permutation(t):
    # https://discuss.pytorch.org/t/shuffling-a-tensor/25422/4
    idx = torch.randperm(t.nelement())
    return t.view(-1)[idx].view(t.size())


class RegularAttention(BigBirdBlockSparseAttention):

    def __init__(self,
                 config,
                 window_size=None,
                 cls_size=None,
                 top_clusters=1,
                 bucket_size=0,
                 compute_graph_stats=False
                 ):
        super().__init__(config)
        self.window_size = window_size
        self.cls_size = cls_size
        self.top_clusters = top_clusters
        self.compute_graph_stats = compute_graph_stats
        self.bucket_size = bucket_size
        self.headwise_proj_q = None
        self.headwise_proj_k = None
        self.centroids = None
        # stats
        self.sparsities_gold = []
        self.sparsities_pred = []
        self.recalls = []
        self.precisions = []
        self.exacts = []
        self.entropies_q = []
        self.entropies_k = []
        self.gold_p = None
        self.pred_p = None
        self.pred_entmax_p = None

    def set_projections(self, headwise_projections):
        if 'w1' in headwise_projections['q'].keys():
            proj_dim = headwise_projections['q']['w2'].shape[-1]
            hid_dim = headwise_projections['q']['w2'].shape[-2]
            self.headwise_proj_q = HeadwiseMLPProjection(self.num_attention_heads, self.attention_head_size, proj_dim, hid_dim)
            self.headwise_proj_k = HeadwiseMLPProjection(self.num_attention_heads, self.attention_head_size, proj_dim, hid_dim)
        else:
            proj_dim = headwise_projections['q']['w'].shape[-1]
            self.headwise_proj_q = HeadwiseLinearProjection(self.num_attention_heads, self.attention_head_size, proj_dim)
            self.headwise_proj_k = HeadwiseLinearProjection(self.num_attention_heads, self.attention_head_size, proj_dim)
        self.headwise_proj_q.load_state_dict(headwise_projections['q'])
        self.headwise_proj_k.load_state_dict(headwise_projections['k'])
        # freeze projectors
        for param in self.headwise_proj_q.parameters():
            param.requires_grad = False
        for param in self.headwise_proj_k.parameters():
            param.requires_grad = False
        if torch.cuda.is_available():
            self.headwise_proj_q = self.headwise_proj_q.cuda()
            self.headwise_proj_k = self.headwise_proj_k.cuda()

    def bigbird_block_sparse_attention(
            self,
            query_layer,
            key_layer,
            value_layer,
            band_mask,
            from_mask,
            to_mask,
            from_blocked_mask,
            to_blocked_mask,
            n_heads,
            n_rand_blocks,
            attention_head_size,
            from_block_size,
            to_block_size,
            batch_size,
            from_seq_len,
            to_seq_len,
            seed,
            plan_from_length,
            plan_num_rand_blocks,
            output_attentions,
    ):
        # get dims
        bsz, num_heads, tgt_len, head_dim = query_layer.shape
        _, _, src_len, _ = key_layer.shape
        attn_mask_penalty = -10000.0

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(attention_head_size)

        # pad_mask.shape is (bs, src_len)
        # attention_mask has 0 in valid positions and -10000.0 in invalid positions
        pad_mask = to_mask.squeeze(2).squeeze(1).bool()

        # unsqueeze pad and causal mask
        uns_pad_mask = pad_mask.unsqueeze(1).unsqueeze(2)

        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_mask = (1.0 - to_mask) * attn_mask_penalty
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = entmax15(attention_scores, dim=-1)

        # use original probs as gold_p and pred_p
        pred_p = attention_probs.clone()

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        if self.compute_graph_stats:
            src_len = from_seq_len
            tgt_len = to_seq_len
            cluster_mask = pred_p > 0
            # pad_mask.shape is (bs, src_len)
            pad_mask = to_mask.squeeze(2).squeeze(1).bool()
            cluster_mask = unblockify_attn(cluster_mask, from_block_size=from_block_size, to_block_size=to_block_size)

            # repeat num_heads if necessary
            if cluster_mask.shape[1] == 1:
                cluster_mask = cluster_mask.repeat(1, self.num_attention_heads, 1, 1)

            # add cls mask
            self.cls_size = 2
            k = self.cls_size * from_block_size
            cls_mask = torch.zeros(bsz, 1, tgt_len, src_len, device=pad_mask.device, dtype=torch.bool)
            cls_mask[:, :, :, :k] = True  # all attend to CLS
            cls_mask[:, :, :k, :] = True  # CLS attends to all
            # deal with padding
            nh = cls_mask.shape[1]
            lens = pad_mask.int().sum(-1)
            ar_s = torch.arange(src_len, device=cls_mask.device, dtype=torch.int)
            ar_t = torch.arange(tgt_len, device=cls_mask.device, dtype=torch.int)
            last_src = ar_s.unsqueeze(0) >= lens.unsqueeze(1)-k
            last_src = last_src.unsqueeze(1).unsqueeze(2).expand(-1, nh, tgt_len, -1)
            last_tgt = ar_t.unsqueeze(0) >= lens.unsqueeze(1)-k
            last_tgt = last_tgt.unsqueeze(1).unsqueeze(-1).expand(-1, nh, -1, src_len)
            cls_mask[last_src] = True  # all attend to last token
            cls_mask[last_tgt] = True  # last token attends to all
            cluster_mask |= cls_mask

            # add window mask
            self.window_size = 3
            w = self.window_size * from_block_size
            win_mask = neighbours_mask(tgt_len, src_len, window_size=w, device=pad_mask.device)
            win_mask = win_mask.unsqueeze(0).unsqueeze(1)
            cluster_mask |= win_mask

            # unsqueeze pad and causal mask
            uns_pad_mask = pad_mask.unsqueeze(1).unsqueeze(2)

            # final mask
            joint_mask = cluster_mask & uns_pad_mask.bool()

            # not exactly a prob distribution but it serves the purpose to calc matching as we only consider p > 0
            pred_p = joint_mask.float()

            # compute gold_p to calculate sparsity and recall
            gold_p = attention_probs

            # reduction=None to be able to compute micro-averaged results
            sps_gold_per_head = compute_sparsity(gold_p, pad_mask, causal_mask=None, reduction=None)
            sps_pred_per_head = compute_sparsity(pred_p, pad_mask, causal_mask=None, reduction=None)
            precs_per_head = compute_precision(pred_p, gold_p, pad_mask, causal_mask=None, reduction=None)
            recs_per_head = compute_recall(pred_p, gold_p, pad_mask, causal_mask=None, reduction=None)
            exacts_per_head = compute_exact_fraction(pred_p, gold_p, pad_mask, causal_mask=None, reduction=None)
            self.sparsities_gold.append(sps_gold_per_head.detach().cpu())
            self.sparsities_pred.append(sps_pred_per_head.detach().cpu())
            self.precisions.append(precs_per_head.detach().cpu())
            self.recalls.append(recs_per_head.detach().cpu())
            self.exacts.append(exacts_per_head.detach().cpu())
            self.gold_p = gold_p.detach().cpu()
            self.pred_p = pred_p.detach().cpu()

        return context_layer, attention_probs

    def forward(
            self,
            hidden_states,
            *args,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=False,
            **kwargs,
            # band_mask=None,
            # from_mask=None,
            # to_mask=None,
            # from_blocked_mask=None,
            # to_blocked_mask=None,
            # output_attentions=None,
    ):
        # Currently this `class` can't be used in decoder.

        batch_size, seqlen, _ = hidden_states.size()
        to_seq_length = from_seq_length = seqlen
        from_block_size = to_block_size = 1

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seqlen), device=hidden_states.device)
        blocked_encoder_mask, band_mask, from_mask, to_mask = BigBirdModel.create_masks_for_block_sparse_attn(
            attention_mask, self.block_size
        )
        from_blocked_mask = to_blocked_mask = blocked_encoder_mask
        extended_attention_mask = None

        assert from_seq_length % from_block_size == 0, "Query sided sequence length must be multiple of block size"
        assert to_seq_length % to_block_size == 0, "Key/Value sided sequence length must be multiple of block size"

        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        context_layer, attention_probs = self.bigbird_block_sparse_attention(
            query_layer,
            key_layer,
            value_layer,
            band_mask,
            from_mask,
            to_mask,
            from_blocked_mask,
            to_blocked_mask,
            self.num_attention_heads,
            self.num_random_blocks,
            self.attention_head_size,
            from_block_size,
            to_block_size,
            batch_size,
            from_seq_length,
            to_seq_length,
            seed=self.seed,
            plan_from_length=None,
            plan_num_rand_blocks=None,
            output_attentions=self.compute_graph_stats,
        )

        context_layer = context_layer.contiguous().view(batch_size, from_seq_length, -1)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


def unblockify_attn(att_dist, from_block_size=1, to_block_size=1, pad_mask=None, causal_mask=None):
    # (batch, heads, n_blocks, n_blocks) -> (batch, heads, n, n)
    att_dist = att_dist.repeat_interleave(to_block_size, dim=-1).repeat_interleave(from_block_size, dim=-2)
    # mask out padding and "future" positions
    if pad_mask is not None:
        pairwise_mask = pad_mask.unsqueeze(-1) & pad_mask.unsqueeze(1)
        pairwise_mask = pairwise_mask.unsqueeze(1)
        if causal_mask is not None:
            pairwise_mask = pairwise_mask & causal_mask.unsqueeze(0).unsqueeze(1)
        att_dist.masked_fill(~pairwise_mask, 0)
    return att_dist
