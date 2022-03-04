import math
from typing import Any

import torch
from torch.nn import Softmax
from transformers import RobertaForMaskedLM, RobertaForSequenceClassification, RobertaForQuestionAnswering
from transformers.models.bert.modeling_bert import BertSelfAttention
from entmax import entmax15

from extender.attn_utils import compute_sparsity, compute_precision, compute_recall, compute_exact_fraction, \
    compute_clusters_entropy, neighbours_mask, quantize
from extender.extender_layers import HeadwiseLinearProjection
from extender.hf_regular import RegularAttention
from extender.hf_sparsefinder import SparsefinderBlockSparseAttention
from extender.hf_bigbird import TorchBigBirdBlockSparseAttention


def robust_entmax15(x, *args, **kwargs):
    zero_attentions = ((x != float('-inf')).sum(dim=-1) == 0).bool().unsqueeze(-1)
    x = x.masked_fill(zero_attentions, 0)  # don't compute entries that only have -inf logits
    out = entmax15(x, *args, **kwargs)
    return out.masked_fill(zero_attentions, 0)  # entries with only -inf logits have prob=0 everywhere


def robust_softmax(x):
    zero_attentions = ((x != float('-inf')).sum(dim=-1) == 0).bool().unsqueeze(-1)
    x = x.masked_fill(zero_attentions, 0)  # don't compute entries that only have -inf logits
    out = Softmax(dim=-1)(x)
    return out.masked_fill(zero_attentions, 0)  # entries with only -inf logits have prob=0 everywhere


def init_simulated_self_attention(
        layer_modules,
        config,
        projections_path,
        centroids_path,
        variant='regular',
        window_size=None,
        cls_size=None,
        top_clusters=1,
        bucket_size=0,
        compute_graph_stats=False,
        block_size=1,
        num_random_blocks=0
):
    config.use_bias = True
    config.block_size = block_size
    config.num_random_blocks = num_random_blocks
    projections = [None for _ in range(len(layer_modules))]
    centroids = [None for _ in range(len(layer_modules))]
    # if variant != 'regular':
    #     projections = torch.load(projections_path, map_location=lambda storage, loc: storage)
    # if variant in ['extender', 'routing_trained', 'routing_extended']:
    #     centroids = torch.load(centroids_path, map_location=lambda storage, loc: storage)
    for i, layer in enumerate(layer_modules):
        if variant == 'extender_hf' or variant == 'extender_rhf':
            layer.attention.self = SparsefinderBlockSparseAttention(
                config,
                window_size=window_size,
                cls_size=cls_size,
                top_clusters=top_clusters,
                bucket_size=bucket_size,
                compute_graph_stats=compute_graph_stats,
                routing_variant=bool(variant == 'extender_rhf')
            )
        elif variant == 'bigbird_hf':
            layer.attention.self = TorchBigBirdBlockSparseAttention(
                config,
                window_size=window_size,
                cls_size=cls_size,
                top_clusters=top_clusters,
                bucket_size=bucket_size,
                compute_graph_stats=compute_graph_stats
            )
        elif variant == 'regular_hf':
            layer.attention.self = RegularAttention(
                config,
                window_size=window_size,
                cls_size=cls_size,
                top_clusters=top_clusters,
                bucket_size=bucket_size,
                compute_graph_stats=compute_graph_stats
            )
        else:
            layer.attention.self = SimulatedSelfAttention(
                config,
                projections[i],
                centroids[i],
                variant=variant,
                window_size=window_size,
                cls_size=cls_size,
                top_clusters=top_clusters,
                bucket_size=bucket_size,
                compute_graph_stats=compute_graph_stats
            )


class SimulatedRobertaForMaskedLM(RobertaForMaskedLM):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        init_simulated_self_attention(self.roberta.encoder.layer, config, **kwargs)


class SimulatedRobertaForSequenceClassification(RobertaForSequenceClassification):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        init_simulated_self_attention(self.roberta.encoder.layer, config, **kwargs)


class SimulatedRobertaForQuestionAnswering(RobertaForQuestionAnswering):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        init_simulated_self_attention(self.roberta.encoder.layer, config, **kwargs)


class SimulatedSelfAttention(BertSelfAttention):
    def __init__(self,
                 config,
                 projections,
                 centroids,
                 variant='regular',
                 window_size=None,
                 cls_size=None,
                 top_clusters=1,
                 bucket_size=0,
                 compute_graph_stats=False
                 ):
        super().__init__(config)
        self.variant = variant
        self.headwise_proj_q = None
        self.headwise_proj_k = None
        self.centroids = None
        self.window_size = window_size
        self.cls_size = cls_size
        self.top_clusters = top_clusters
        self.compute_graph_stats = compute_graph_stats
        self.bucket_size = bucket_size
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

    def reset_stats_attrs(self):
        self.sparsities_gold = []
        self.sparsities_pred = []
        self.recalls = []
        self.precisions = []
        self.exacts = []
        self.entropies_q = []
        self.entropies_k = []

    def set_projections(self, headwise_projections):
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

    def set_centroids(self, centroids):
        self.centroids = centroids
        if centroids is not None:
            self.centroids.requires_grad = False
            if torch.cuda.is_available():
                self.centroids = self.centroids.cuda()

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=False,
    ):
        """
        output_attention outputs projected q's and k's, instead of attention probs
        """
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # get dims
        bsz, num_heads, tgt_len, head_dim = query_layer.shape
        _, _, src_len, _ = key_layer.shape

        # pad_mask.shape is (bs, src_len)
        # attention_mask has 0 in valid positions and -10000.0 in invalid positions
        pad_mask = attention_mask.squeeze(2).squeeze(1) == 0

        # to be used later to calc stats
        pred_p = None
        gold_p = None
        pred_entmax_p = None
        q_clustered = None
        k_clustered = None
        num_centroids = 2  # small int just to not break the logic

        if self.variant != 'regular':
            # q_low.shape is (batch_size, num_heads, tgt_len, num_projections)
            q_low = self.headwise_proj_q(query_layer)
            # k_low.shape is (batch_size, num_heads, src_len, num_projections)
            k_low = self.headwise_proj_k(key_layer)

            if self.variant == 'extender':
                # add `batch` dimension (also, num_runs=1)
                # (batch_size, num_heads, num_runs, num_centroids, num_projections)
                expanded_centroids = self.centroids.unsqueeze(0).expand(bsz, -1, -1, -1, -1)
                num_centroids = self.centroids.shape[-2]

                # add `num_runs` dimension
                # (batch_size, num_heads, 1, q_seq_len, num_projections)
                expanded_q_low = q_low.unsqueeze(2)
                # (batch_size, num_heads, 1, k_seq_len, num_projections)
                expanded_k_low = k_low.unsqueeze(2)

                # q_dists.shape is (batch, num_heads, num_runs, q_seq_len, num_centroids)
                # q_dists = torch.cdist(expanded_q_low, expanded_centroids, p=2)
                # k_dists.shape is (batch, num_heads, num_runs, k_seq_len, num_centroids)
                # k_dists = torch.cdist(expanded_k_low, expanded_centroids, p=2)

                # more stable:
                q_diffs_sq = (expanded_q_low.unsqueeze(-2).double() - expanded_centroids.unsqueeze(-3).double())**2
                q_dists = torch.sum(q_diffs_sq, dim=-1)
                k_diffs_sq = (expanded_k_low.unsqueeze(-2).double() - expanded_centroids.unsqueeze(-3).double())**2
                k_dists = torch.sum(k_diffs_sq, dim=-1)

                # simulate extender mask
                # q_clustered.shape is (batch, num_heads, num_runs, q_seq_len)
                # q_clustered = torch.argmin(q_dists, dim=-1)
                # k_clustered.shape is (batch, num_heads, num_runs, k_seq_len)
                # k_clustered = torch.argmin(k_dists, dim=-1)
                # cluster_mask.shape is (batch, num_heads, q_seq_len, k_seq_len)
                # cluster_mask = q_clustered.transpose(2, 3) == k_clustered

                # simulate extender mask with the top_k strategy:
                _, q_clustered_topk = torch.topk(q_dists, self.top_clusters, dim=-1, largest=False)
                _, k_clustered_topk = torch.topk(k_dists, self.top_clusters, dim=-1, largest=False)
                cluster_mask_topk = q_clustered_topk.transpose(2, 3) == k_clustered_topk
                cluster_mask = cluster_mask_topk.any(-1)  # if at least one round is correct we have a match

            elif self.variant == 'distance_cos':
                threshold = self.bucket_size / 10.0 if self.bucket_size >= 0 else 0.5
                q_low_norm = q_low.norm(p=2, dim=-1).unsqueeze(-1)
                k_low_norm = k_low.norm(p=2, dim=-1).unsqueeze(-1)
                cos_sim = (q_low / q_low_norm) @ (k_low / k_low_norm).transpose(-1, -2)
                pairwise_ds = 1 - cos_sim
                cluster_mask = pairwise_ds < threshold

            elif self.variant == 'distance_l2':
                threshold = self.bucket_size / 10.0 if self.bucket_size >= 0 else 1.0
                pairwise_ds = torch.sum((q_low.unsqueeze(-2) - k_low.unsqueeze(-3)) ** 2, dim=-1)
                cluster_mask = pairwise_ds < threshold

            elif self.variant == 'q_fixed':
                bucket_size = math.ceil(src_len / self.bucket_size) if self.bucket_size > 0 else int(math.log2(src_len))
                buckets_q = quantize(q_low, bucket_size=bucket_size, mask=pad_mask, enforce_equal_size=True)
                buckets_k = quantize(k_low, bucket_size=bucket_size, mask=pad_mask, enforce_equal_size=True)
                cluster_mask = buckets_q.unsqueeze(3) == buckets_k.unsqueeze(2)
                cluster_mask = cluster_mask.any(-1)  # if at least one round is correct we have a match

            elif self.variant == 'q_dynamic':
                bucket_size = math.ceil(src_len / self.bucket_size) if self.bucket_size > 0 else int(math.log2(src_len))
                buckets_q = quantize(q_low, bucket_size=bucket_size, mask=pad_mask, enforce_equal_size=False)
                buckets_k = quantize(k_low, bucket_size=bucket_size, mask=pad_mask, enforce_equal_size=False)
                cluster_mask = buckets_q.unsqueeze(3) == buckets_k.unsqueeze(2)
                cluster_mask = cluster_mask.any(-1)  # if at least one round is correct we have a match

            elif self.variant == 'bigbird':
                from extender.sim_bigbird import bigbird_simulated_attention
                pairwise_mask = pad_mask.unsqueeze(-1) & pad_mask.unsqueeze(1)
                pairwise_mask = pairwise_mask.unsqueeze(1)
                bucket_size = self.bucket_size if self.bucket_size > 0 else min(3, src_len//8)
                cluster_mask = bigbird_simulated_attention(
                    pairwise_mask.cpu().numpy(),
                    num_attention_heads=num_heads,
                    num_rand_blocks=bucket_size,  # default from bigbird repo is 3
                    from_seq_length=tgt_len,
                    to_seq_length=src_len,
                    from_block_size=1,
                    to_block_size=1,
                    max_seq_len=src_len
                )
                cluster_mask = cluster_mask.to(pad_mask.device).bool()
            elif self.variant == 'longformer':
                from extender.sim_longformer import longformer_simulated_attention
                bucket_size = self.bucket_size if self.bucket_size > 0 else max(3, int(math.log2(src_len)))
                cluster_mask = longformer_simulated_attention(
                    pad_mask,
                    window_size=1,  # window will be added later
                    dilation=0,  # we do not consider dilation
                    # max_globals_per_sample=8,
                    max_globals_per_sample=bucket_size,
                    min_globals_per_sample=2,
                )
                cluster_mask = cluster_mask.to(pad_mask.device).bool().unsqueeze(1)
            elif self.variant == 'reformer':
                from extender.sim_reformer import reformer_simulated_attention
                bucket_size = self.bucket_size if self.bucket_size > 0 else None
                if bucket_size is not None:
                    kw = {'num_buckets': bucket_size}
                else:
                    kw = {'lsh_attn_chunk_length': max(1, int(math.log2(src_len)))}
                qk_low = torch.cat([q_low, k_low], dim=-1)
                cluster_mask = reformer_simulated_attention(
                    qk_low,
                    # lsh_attn_chunk_length=8,  # default from reformer is 4 or 8 for longer seqs
                    # lsh_attn_chunk_length=max(1, int(math.log2(tgt_len))),
                    # num_buckets=bucket_size,
                    num_hashes=1,  # if > 1, use any(-1) to represent matches
                    mask=pad_mask,
                    **kw
                )
                cluster_mask = cluster_mask.to(pad_mask.device).bool()
            elif self.variant == 'reformer_multi':
                from extender.sim_reformer import reformer_simulated_attention
                bucket_size = self.bucket_size if self.bucket_size > 0 else None
                if bucket_size is not None:
                    kw = {'num_buckets': bucket_size}
                else:
                    kw = {'lsh_attn_chunk_length': max(1, int(math.log2(src_len)))}
                qk_low = torch.cat([q_low, k_low], dim=-1)
                cluster_mask = reformer_simulated_attention(
                    qk_low,
                    # lsh_attn_chunk_length=8,  # default from reformer is 4 or 8 for longer seqs
                    # lsh_attn_chunk_length=max(1, int(math.log2(tgt_len))),
                    # num_buckets=bucket_size,
                    num_hashes=q_low.shape[-1],  # if > 1, use any(-1) to represent matches
                    mask=pad_mask,
                    **kw
                )
                cluster_mask = cluster_mask.to(pad_mask.device).bool()
            elif self.variant == 'reformer_fixed':
                from extender.sim_reformer import lsh_simulated_attention
                bucket_size = self.bucket_size if self.bucket_size > 0 else None
                if bucket_size is not None:
                    kw = {'num_buckets': bucket_size}
                else:
                    kw = {'lsh_attn_chunk_length': max(1, int(math.log2(src_len)))}
                cluster_mask = lsh_simulated_attention(
                    q_low,
                    k_low,
                    # lsh_attn_chunk_length=8,  # default from reformer is 4 or 8 for longer seqs
                    # lsh_attn_chunk_length=max(1, int(math.log2(tgt_len))),
                    # num_buckets=bucket_size,
                    num_hashes=q_low.shape[-1],  # if > 1, use any(-1) to represent matches
                    mask=pad_mask,
                    **kw
                )
                cluster_mask = cluster_mask.to(pad_mask.device).bool()
            elif self.variant in ['routing', 'routing_trained', 'routing_extended']:
                from extender.sim_routing_transformer import routing_simulated_attention
                num_centroids = self.centroids.num_clusters
                cluster_mask = routing_simulated_attention(q_low, k_low, self.centroids, topk_window_size=None)
                cluster_mask = cluster_mask.bool().to(q_low.device)
            else:
                raise NotImplementedError

            # repeat num_heads if necessary
            if cluster_mask.shape[1] == 1:
                cluster_mask = cluster_mask.repeat(1, num_heads, 1, 1)

            # add cls mask
            if self.cls_size is not None:
                cls_mask = torch.zeros(bsz, 1, tgt_len, src_len, device=pad_mask.device, dtype=torch.bool)
                cls_mask[:, :, :, 0] = True  # all attend to CLS
                cls_mask[:, :, 0, :] = True  # CLS attends to all
                if self.cls_size == 2:
                    nh = cls_mask.shape[1]
                    lens = pad_mask.int().sum(-1)
                    ar_s = torch.arange(src_len, device=cls_mask.device, dtype=torch.int)
                    ar_t = torch.arange(tgt_len, device=cls_mask.device, dtype=torch.int)
                    last_src = ar_s.unsqueeze(0) == lens.unsqueeze(1)-1
                    last_src = last_src.unsqueeze(1).unsqueeze(2).expand(-1, nh, tgt_len, -1)
                    last_tgt = ar_t.unsqueeze(0) == lens.unsqueeze(1)-1
                    last_tgt = last_tgt.unsqueeze(1).unsqueeze(-1).expand(-1, nh, -1, src_len)
                    cls_mask[last_src] = True  # all attend to last token
                    cls_mask[last_tgt] = True  # last token attends to all
                cluster_mask |= cls_mask

            if self.window_size is not None:
                win_mask = neighbours_mask(tgt_len, src_len, window_size=self.window_size, device=pad_mask.device)
                win_mask = win_mask.unsqueeze(0).unsqueeze(1)
                cluster_mask |= win_mask

            # unsqueeze pad and causal mask
            uns_pad_mask = pad_mask.unsqueeze(1).unsqueeze(2)

            # final mask
            joint_mask = cluster_mask & uns_pad_mask

            # not exactly a prob distribution but it serves the purpose to calc matching as we only consider p > 0
            pred_p = joint_mask.float()

            # do original attention and mask to perform simulation
            gold_logits = attention_scores + attention_mask

            # compute gold_p to calculate sparsity and recall
            gold_p = entmax15(gold_logits, dim=-1)

            # create attention_mask from joint_mask
            # 0.0 for valid positions and -10000.0 for invalid ones
            attention_mask = (1.0 - joint_mask.float()) * -10000.0

            # get pred_p using entmax with joint_mask
            pred_entmax_p = entmax15(attention_scores + attention_mask, dim=-1)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = entmax15(attention_scores, dim=-1)
        # attention_probs = robust_softmax(attention_scores)

        # use original probs as gold_p and pred_p
        if gold_p is None:
            gold_p = attention_probs.clone()
        if pred_p is None:
            pred_p = attention_probs.clone()
        if pred_entmax_p is None:
            pred_entmax_p = attention_probs.clone()

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # compute stats
        if self.compute_graph_stats:
            # small workaround for non-clustering variants
            if q_clustered is None or k_clustered is None:
                k_clustered = torch.ones(bsz, num_heads, 1, src_len, device=gold_p.device).long()
                q_clustered = torch.ones(bsz, num_heads, 1, tgt_len, device=gold_p.device).long()

            # reduction=None to be able to compute micro-averaged results
            sps_gold_per_head = compute_sparsity(gold_p, pad_mask, causal_mask=None, reduction=None)
            sps_pred_per_head = compute_sparsity(pred_p, pad_mask, causal_mask=None, reduction=None)
            precs_per_head = compute_precision(pred_p, gold_p, pad_mask, causal_mask=None, reduction=None)
            recs_per_head = compute_recall(pred_p, gold_p, pad_mask, causal_mask=None, reduction=None)
            exacts_per_head = compute_exact_fraction(pred_p, gold_p, pad_mask, causal_mask=None, reduction=None)
            ents_per_head = compute_clusters_entropy(q_clustered.transpose(-1, -2), k_clustered.transpose(-1, -2),
                                                     num_centroids, pad_mask, reduction=None)
            self.sparsities_gold.append(sps_gold_per_head.detach().cpu())
            self.sparsities_pred.append(sps_pred_per_head.detach().cpu())
            self.precisions.append(precs_per_head.detach().cpu())
            self.recalls.append(recs_per_head.detach().cpu())
            self.exacts.append(exacts_per_head.detach().cpu())
            self.entropies_q.append(ents_per_head['q_avg'].detach().cpu())
            self.entropies_k.append(ents_per_head['k_avg'].detach().cpu())

            self.gold_p = gold_p.detach().cpu()
            self.pred_p = pred_p.detach().cpu()
            self.pred_entmax_p = pred_entmax_p.detach().cpu()

        kqs = {'src_q': query_layer, 'src_k': key_layer}
        outputs = (context_layer, kqs) if output_attentions else (context_layer,)
        return outputs


if __name__ == '__main__':
    # roberta = EntmaxRobertaForMaskedLM(config="/home/agois/longformer_stuff/entmax-roberta-maia/config.json")
    roberta = SimulatedRobertaForMaskedLM.from_pretrained('roberta-base')
    inp = torch.remainder(torch.LongTensor(16, 512), 1000)
    out = roberta(input_ids=inp, output_attentions=True)
    print('bye')
