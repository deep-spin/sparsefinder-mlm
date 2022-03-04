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


class TorchBigBirdBlockSparseAttention(BigBirdBlockSparseAttention):

    def __init__(self,
                 config,
                 window_size=None,
                 cls_size=None,
                 top_clusters=1,
                 bucket_size=0,
                 compute_graph_stats=False,
                 routing_variant=False
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
        self.routing_variant = routing_variant
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
        self.rand_attn = None

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

    @staticmethod
    def _bigbird_block_rand_mask(
            from_seq_length, to_seq_length, from_block_size, to_block_size, num_rand_blocks, last_idx=-1, device=None
    ):
        """
        Create adjacency list of random attention.

        Args:
            from_seq_length: int. length of from sequence.
            to_seq_length: int. length of to sequence.
            from_block_size: int. size of block in from sequence.
            to_block_size: int. size of block in to sequence.
            num_rand_blocks: int. Number of random chunks per row.
            last_idx: if -1 then num_rand_blocks blocks chosen anywhere in to sequence,
            if positive then num_rand_blocks blocks chosen only up to last_idx.

        Returns:
            adjacency list of size from_seq_length//from_block_size-2 by num_rand_blocks
        """
        # using this method when from_seq_length in [1024, 3072, 4096]

        assert (
                from_seq_length // from_block_size == to_seq_length // to_block_size
        ), "Error the number of blocks needs to be same!"

        rand_attn = torch.zeros((from_seq_length // from_block_size - 2, num_rand_blocks), dtype=torch.int32, device=device)
        middle_seq = torch.arange(1, to_seq_length // to_block_size - 1, dtype=torch.int32, device=device)
        last = to_seq_length // to_block_size - 1
        if last_idx > (2 * to_block_size):
            last = (last_idx // to_block_size) - 1

        r = num_rand_blocks  # shorthand
        for i in range(1, from_seq_length // from_block_size - 1):
            start = i - 2
            end = i
            if i == 1:
                rand_attn[i - 1, :] = torch_permutation(middle_seq[2:last])[:r]
            elif i == 2:
                rand_attn[i - 1, :] = torch_permutation(middle_seq[3:last])[:r]
            elif i == from_seq_length // from_block_size - 3:
                rand_attn[i - 1, :] = torch_permutation(middle_seq[:last])[:r]
            # Missing -3: should have been sliced till last-3
            elif i == from_seq_length // from_block_size - 2:
                rand_attn[i - 1, :] = torch_permutation(middle_seq[:last])[:r]
            # Missing -4: should have been sliced till last-4
            else:
                if start > last:
                    start = last
                    rand_attn[i - 1, :] = torch_permutation(middle_seq[:start])[:r]
                elif (end + 1) == last:
                    rand_attn[i - 1, :] = torch_permutation(middle_seq[:start])[:r]
                else:
                    rand_attn[i - 1, :] = torch_permutation(
                        torch.cat((middle_seq[:start], middle_seq[end + 1:last]))
                    )[:r]
        return rand_attn


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

        # BigBird block-sparse attention as suggested in paper

        # ITC:
        #     global tokens: 2 x block_size
        #     window tokens: 3 x block_size
        #     random tokens: num_rand_tokens x block_size

        # ETC:
        #     global tokens: extra_globals_tokens + 2 x block_size
        #     window tokens: 3 x block_size
        #     random tokens: num_rand_tokens x block_size

        # Note:
        #     1) Currently, ETC is not supported.
        #     2) Window size is fixed to 3 blocks & it can be changed only by
        #     changing `block_size`.
        #     3) Number of global blocks are fixed (2 blocks here) & global tokens can be
        #     controlled only by `block_size`.

        # attention is calculated separately for q[0], q[1], q[2:-2], q[-2], q[-1] in order to use special trick of shifting tokens (for calculating sliding attention)
        # hence following code can be divided into 5 parts.
        if from_seq_len // from_block_size != to_seq_len // to_block_size:
            raise ValueError("Error the number of blocks needs to be same!")

        rsqrt_d = 1 / math.sqrt(attention_head_size)
        bsz = batch_size
        attn_mask_penalty = -10000.0

        # generate random attention and corresponding masks
        if from_seq_len in [512, 1024, 3072, 4096]:  # old plans used in paper
            if self.rand_attn is None:
                rand_attn = [
                    self._bigbird_block_rand_mask(
                        self.max_seqlen, self.max_seqlen, from_block_size, to_block_size, n_rand_blocks,
                        last_idx=from_seq_len, device=query_layer.device
                    )[: (from_seq_len // from_block_size - 2)]
                    for _ in range(n_heads)
                ]
                rand_attn = torch.stack(rand_attn, dim=0).long()
                rand_attn = rand_attn.unsqueeze(0).repeat(batch_size, 1, 1, 1)
                self.rand_attn = rand_attn
            else:
                rand_attn = self.rand_attn
        else:
            if plan_from_length is None:
                plan_from_length, plan_num_rand_blocks = self._get_rand_attn_plan(
                    from_seq_len, from_block_size, n_rand_blocks
                )

            rand_attn = self._bigbird_block_rand_mask_with_head(
                from_seq_length=from_seq_len,
                to_seq_length=to_seq_len,
                from_block_size=from_block_size,
                to_block_size=to_block_size,
                num_heads=n_heads,
                plan_from_length=plan_from_length,
                plan_num_rand_blocks=plan_num_rand_blocks,
            )
            rand_attn = np.stack(rand_attn, axis=0)
            rand_attn = torch.tensor(rand_attn, device=query_layer.device, dtype=torch.long)
            rand_attn.unsqueeze_(0)
            rand_attn = torch.cat([rand_attn for _ in range(batch_size)], dim=0)

        rand_mask = self._create_rand_mask_from_inputs(
            from_blocked_mask, to_blocked_mask, rand_attn, n_heads, n_rand_blocks, bsz, from_seq_len, from_block_size
        )

        blocked_query_matrix = query_layer.view(bsz, n_heads, from_seq_len // from_block_size, from_block_size, -1)
        blocked_key_matrix = key_layer.view(bsz, n_heads, to_seq_len // to_block_size, to_block_size, -1)
        blocked_value_matrix = value_layer.view(bsz, n_heads, to_seq_len // to_block_size, to_block_size, -1)

        # preparing block for randn attn
        # print(blocked_key_matrix.shape, rand_attn.shape)
        if blocked_key_matrix.shape[0] < rand_attn.shape[0]:
            rand_attn = rand_attn[:blocked_key_matrix.shape[0]]
        gathered_key = self.torch_gather_b2(blocked_key_matrix, rand_attn)
        gathered_key = gathered_key.view(
            bsz, n_heads, to_seq_len // to_block_size - 2, n_rand_blocks * to_block_size, -1
        )  # [bsz, n_heads, to_seq_len//to_block_size-2, n_rand_blocks, to_block_size, -1]
        gathered_value = self.torch_gather_b2(blocked_value_matrix, rand_attn)
        gathered_value = gathered_value.view(
            bsz, n_heads, to_seq_len // to_block_size - 2, n_rand_blocks * to_block_size, -1
        )  # [bsz, n_heads, to_seq_len//to_block_size-2, n_rand_blocks, to_block_size, -1]

        # 1st PART
        # 1st block (global block) attention scores
        # q[0] x (k[0], k[1], k[2], k[3], k[4] .... )

        # [bsz, n_heads, from_block_size, -1] x [bsz, n_heads, to_seq_len, -1] ==> [bsz, n_heads, from_block_size, to_seq_len]
        first_product = self.torch_bmm_nd_transpose(blocked_query_matrix[:, :, 0], key_layer, ndim=4)

        first_product = first_product * rsqrt_d
        first_product += (1.0 - to_mask) * attn_mask_penalty
        first_attn_weights = nn.functional.softmax(
            first_product, dim=-1
        )  # [bsz, n_heads, from_block_size, to_seq_len]

        # [bsz, n_heads, from_block_size, to_seq_len] x [bsz, n_heads, to_seq_len, -1] ==> [bsz, n_heads, from_block_size, -1]
        first_context_layer = self.torch_bmm_nd(first_attn_weights, value_layer, ndim=4)
        first_context_layer.unsqueeze_(2)

        # 2nd PART
        # 2nd block attention scores
        # q[1] x (sliding_keys, random_keys, global_keys)
        # sliding key blocks -> 2nd, 3rd blocks
        # global key blocks -> 1st block

        second_key_mat = torch.cat(
            [
                blocked_key_matrix[:, :, 0],
                blocked_key_matrix[:, :, 1],
                blocked_key_matrix[:, :, 2],
                blocked_key_matrix[:, :, -1],
                gathered_key[:, :, 0],
            ],
            dim=2,
        )  # [bsz, n_heads, (4+n_rand_blocks)*to_block_size, -1]
        second_value_mat = torch.cat(
            [
                blocked_value_matrix[:, :, 0],
                blocked_value_matrix[:, :, 1],
                blocked_value_matrix[:, :, 2],
                blocked_value_matrix[:, :, -1],
                gathered_value[:, :, 0],
            ],
            dim=2,
        )  # [bsz, n_heads, (4+n_rand_blocks)*to_block_size, -1]

        # [bsz, n_heads, from_block_size, -1] x [bsz, n_heads, (4+n_rand_blocks)*to_block_size, -1] ==> [bsz, n_heads, from_block_size, (4+n_rand_blocks)*to_block_size]
        second_product = self.torch_bmm_nd_transpose(blocked_query_matrix[:, :, 1], second_key_mat, ndim=4)
        second_seq_pad = torch.cat(
            [
                to_mask[:, :, :, : 3 * to_block_size],
                to_mask[:, :, :, -to_block_size:],
                to_mask.new_ones([bsz, 1, 1, n_rand_blocks * to_block_size]),
            ],
            dim=3,
        )
        second_rand_pad = torch.cat(
            [
                rand_mask.new_ones([bsz, n_heads, from_block_size, 4 * to_block_size]),
                rand_mask[:, :, 0],
            ],
            dim=3,
        )
        second_product = second_product * rsqrt_d
        second_product += (1.0 - torch.min(second_seq_pad, second_rand_pad)) * attn_mask_penalty
        second_attn_weights = nn.functional.softmax(
            second_product, dim=-1
        )  # [bsz, n_heads, from_block_size, (4+n_rand_blocks)*to_block_size]

        # [bsz, n_heads, from_block_size, (4+n_rand_blocks)*to_block_size] x [bsz, n_heads, (4+n_rand_blocks)*to_block_size, -1] ==> [bsz, n_heads, from_block_size, -1]
        second_context_layer = self.torch_bmm_nd(second_attn_weights, second_value_mat, ndim=4)

        second_context_layer.unsqueeze_(2)

        # 3rd PART
        # Middle blocks attention scores
        # q[-2:2] x (sliding_keys, random_keys, global_keys)
        # sliding attn is calculated using special trick of shifting tokens as discussed in paper
        # random keys are generated by taking random indices as per `rand_attn`
        # global keys -> 1st & last block

        exp_blocked_key_matrix = torch.cat(
            [blocked_key_matrix[:, :, 1:-3], blocked_key_matrix[:, :, 2:-2], blocked_key_matrix[:, :, 3:-1]], dim=3
        )  # [bsz, n_heads, from_seq_len//from_block_size-4, 3*to_block_size, -1]
        exp_blocked_value_matrix = torch.cat(
            [blocked_value_matrix[:, :, 1:-3], blocked_value_matrix[:, :, 2:-2], blocked_value_matrix[:, :, 3:-1]],
            dim=3,
        )  # [bsz, n_heads, from_seq_len//from_block_size-4, 3*to_block_size, -1]
        middle_query_matrix = blocked_query_matrix[:, :, 2:-2]

        # sliding attention scores for q[-2:2]
        # [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, -1] x [b, n_heads, from_seq_len//from_block_size-4, 3*to_block_size, -1]
        inner_band_product = self.torch_bmm_nd_transpose(middle_query_matrix, exp_blocked_key_matrix, ndim=5)
        #     ==> [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, 3*to_block_size]
        inner_band_product = inner_band_product * rsqrt_d

        # randn attention scores for q[-2:2]
        # [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, -1] x [bsz, n_heads, from_seq_len//from_block_size-4, n_rand_blocks*to_block_size, -1]
        rand_band_product = self.torch_bmm_nd_transpose(middle_query_matrix, gathered_key[:, :, 1:-1], ndim=5)
        #     ==> [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, n_rand_blocks*to_block_size]
        rand_band_product = rand_band_product * rsqrt_d

        # Including 1st block (since it's global)
        first_band_product = torch.einsum(
            "bhlqd,bhkd->bhlqk", middle_query_matrix, blocked_key_matrix[:, :, 0]
        )  # [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, -1] x [bsz, n_heads, to_block_size, -1] ==> [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, to_block_size]
        first_band_product = first_band_product * rsqrt_d

        # Including last block (since it's global)
        last_band_product = torch.einsum(
            "bhlqd,bhkd->bhlqk", middle_query_matrix, blocked_key_matrix[:, :, -1]
        )  # [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, -1] x [bsz, n_heads, to_block_size, -1] ==> [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, to_block_size]
        last_band_product = last_band_product * rsqrt_d

        # masking padded tokens
        inner_band_product += (1.0 - band_mask) * attn_mask_penalty
        first_band_product += (1.0 - to_mask[:, :, :, :to_block_size].unsqueeze(3)) * attn_mask_penalty
        last_band_product += (1.0 - to_mask[:, :, :, -to_block_size:].unsqueeze(3)) * attn_mask_penalty
        rand_band_product += (1.0 - rand_mask[:, :, 1:-1]) * attn_mask_penalty

        # completing attention scores matrix for all q[-2:2]
        band_product = torch.cat(
            [first_band_product, inner_band_product, rand_band_product, last_band_product], dim=-1
        )  # [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, (5+n_rand_blocks)*to_block_size]

        # safely doing softmax since attention matrix is completed
        attn_weights = nn.functional.softmax(
            band_product, dim=-1
        )  # [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, (5+n_rand_blocks)*to_block_size]

        # contribution of sliding keys
        # [bsz, n_heads, m//from_block_size-4, from_block_size, 3*to_block_size] x [bsz, n_heads, from_seq_len//from_block_size-4, 3*to_block_size, -1]
        context_layer = self.torch_bmm_nd(
            attn_weights[:, :, :, :, to_block_size : 4 * to_block_size], exp_blocked_value_matrix, ndim=5
        )
        #     ==> [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, -1]

        # adding contribution of random keys
        # [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, n_rand_blocks*to_block_size] x [bsz, n_heads, from_seq_len//from_block_size-4, n_rand_blocks*to_block_size, -1]
        context_layer += self.torch_bmm_nd(
            attn_weights[:, :, :, :, 4 * to_block_size : -to_block_size], gathered_value[:, :, 1:-1], ndim=5
        )
        #     ==> [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, -1]

        # adding contribution of global keys
        context_layer += torch.einsum(
            "bhlqk,bhkd->bhlqd", attn_weights[:, :, :, :, :to_block_size], blocked_value_matrix[:, :, 0]
        )  # [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, to_block_size] x [bsz, n_heads, to_block_size, -1] ==> [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, -1]
        context_layer += torch.einsum(
            "bhlqk,bhkd->bhlqd", attn_weights[:, :, :, :, -to_block_size:], blocked_value_matrix[:, :, -1]
        )  # [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, to_block_size] x [bsz, n_heads, to_block_size, -1] ==> [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, -1]

        # 4th PART
        # last 2nd token attention scores
        # q[-2] x (sliding_keys, random_keys, global_keys)
        # sliding key blocks -> last 3 blocks
        # global key block -> 1st block
        # random key block -> based on indices stored in `randn_attn`

        second_last_key_mat = torch.cat(
            [
                blocked_key_matrix[:, :, 0],
                blocked_key_matrix[:, :, -3],
                blocked_key_matrix[:, :, -2],
                blocked_key_matrix[:, :, -1],
                gathered_key[:, :, -1],
            ],
            dim=2,
        )  # [bsz, n_heads, (4+n_random_blocks)*to_block_size, -1]
        second_last_value_mat = torch.cat(
            [
                blocked_value_matrix[:, :, 0],
                blocked_value_matrix[:, :, -3],
                blocked_value_matrix[:, :, -2],
                blocked_value_matrix[:, :, -1],
                gathered_value[:, :, -1],
            ],
            dim=2,
        )  # [bsz, n_heads, (4+r)*to_block_size, -1]

        # [bsz, n_heads, from_block_size, -1] x [bsz, n_heads, (4+n_rand_blocks)*to_block_size, -1] ==> [bsz, n_heads, from_block_size, (4+n_rand_blocks)*to_block_size]
        second_last_product = self.torch_bmm_nd_transpose(blocked_query_matrix[:, :, -2], second_last_key_mat, ndim=4)
        second_last_seq_pad = torch.cat(
            [
                to_mask[:, :, :, :to_block_size],
                to_mask[:, :, :, -3 * to_block_size :],
                to_mask.new_ones([bsz, 1, 1, n_rand_blocks * to_block_size]),
            ],
            dim=3,
        )
        second_last_rand_pad = torch.cat(
            [
                rand_mask.new_ones([bsz, n_heads, from_block_size, 4 * to_block_size]),
                rand_mask[:, :, -1],
            ],
            dim=3,
        )
        second_last_product = second_last_product * rsqrt_d
        second_last_product += (1.0 - torch.min(second_last_seq_pad, second_last_rand_pad)) * attn_mask_penalty
        second_last_attn_weights = nn.functional.softmax(
            second_last_product, dim=-1
        )  # [bsz, n_heads, from_block_size, (4+n_rand_blocks)*to_block_size]

        # [bsz, n_heads, from_block_size, (4+n_rand_blocks)*to_block_size] x [bsz, n_heads, (4+n_rand_blocks)*to_block_size, -1] ==> [bsz, n_heads, from_block_size, -1]
        second_last_context_layer = self.torch_bmm_nd(second_last_attn_weights, second_last_value_mat, ndim=4)
        second_last_context_layer.unsqueeze_(2)

        # 5th PART
        # last block (global) attention scores
        # q[-1] x (k[0], k[1], k[2], k[3], .... )

        # [bsz, n_heads, from_block_size, -1] x [bsz, n_heads, to_seq_len, -1] ==> [bsz, n_heads, from_block_size, to_seq_len]
        last_product = self.torch_bmm_nd_transpose(blocked_query_matrix[:, :, -1], key_layer, ndim=4)
        last_product = last_product * rsqrt_d
        last_product += (1.0 - to_mask) * attn_mask_penalty
        last_attn_weights = nn.functional.softmax(last_product, dim=-1)  # [bsz, n_heads, from_block_size, n]

        # [bsz, n_heads, from_block_size, to_seq_len] x [bsz, n_heads, to_seq_len, -1] ==> [bsz, n_heads, from_block_size, -1]
        last_context_layer = self.torch_bmm_nd(last_attn_weights, value_layer, ndim=4)
        last_context_layer.unsqueeze_(2)

        # combining representations of all tokens
        context_layer = torch.cat(
            [first_context_layer, second_context_layer, context_layer, second_last_context_layer, last_context_layer],
            dim=2,
        )
        context_layer = context_layer.view((bsz, n_heads, from_seq_len, -1)) * from_mask
        context_layer = torch.transpose(context_layer, 1, 2)

        # this is just for visualizing; forward pass doesn't depend on following code
        if output_attentions:
            # TODO(PVP): need to verify if below code is correct
            attention_probs = torch.zeros(
                bsz, n_heads, from_seq_len, to_seq_len, dtype=torch.float, device=context_layer.device
            )

            # 1st query block
            # corresponding to `first_context_layer`
            attention_probs[:, :, :from_block_size, :] = first_attn_weights  # all keys global

            # 2nd query block
            # corresponding to `second_context_layer`
            attention_probs[:, :, from_block_size : 2 * from_block_size, : 3 * to_block_size] = second_attn_weights[
                                                                                                :, :, :, : 3 * to_block_size
                                                                                                ]  # 1st three key blocks (global + sliding)
            attention_probs[:, :, from_block_size : 2 * from_block_size, -to_block_size:] = second_attn_weights[
                                                                                            :, :, :, 3 * to_block_size : 4 * to_block_size
                                                                                            ]  # last key block (global)
            # random keys
            for p1, i1, w1 in zip(range(bsz), rand_attn, second_attn_weights):
                # p1, i1, w1 corresponds to batch_dim i.e. following operation is done for each sequence in batch
                for p2, i2, w2 in zip(range(n_heads), i1, w1):
                    # p2, i2, w2 corresponds to head_dim i.e. following operation is done for each heads
                    attn_probs_view = attention_probs.view(
                        bsz,
                        n_heads,
                        from_seq_len // from_block_size,
                        from_block_size,
                        to_seq_len // to_block_size,
                        to_block_size,
                        )
                    right_slice = w2[:, 4 * to_block_size :]
                    attn_probs_view[p1, p2, 1, :, i2[0]] = right_slice.view(
                        from_block_size, n_rand_blocks, to_block_size
                    )

            # Middle query blocks
            # corresponding to `context_layer`
            # sliding keys
            for q_idx in range(from_seq_len // from_block_size - 4):
                attn_probs_view = attention_probs.view(
                    bsz,
                    n_heads,
                    from_seq_len // from_block_size,
                    from_block_size,
                    to_seq_len // to_block_size,
                    to_block_size,
                    )[:, :, 2:-2, :, 1:-1, :]
                right_slice = attn_weights[:, :, q_idx, :, to_block_size : 4 * to_block_size]
                attn_probs_view[:, :, q_idx, :, q_idx : q_idx + 3, :] = right_slice.view(
                    bsz, n_heads, from_block_size, 3, to_block_size
                )  # inner_band_product
            # global keys (corresponding to 1st key block)
            attention_probs[:, :, 2 * from_block_size : -2 * from_block_size, :to_block_size] = attn_weights[
                                                                                                :, :, :, :, :to_block_size
                                                                                                ].view(
                bsz, n_heads, -1, to_block_size
            )  # first_band_product
            # global keys (corresponding to last key block)
            attention_probs[:, :, 2 * from_block_size : -2 * from_block_size, -to_block_size:] = attn_weights[
                                                                                                 :, :, :, :, -to_block_size:
                                                                                                 ].view(
                bsz, n_heads, -1, to_block_size
            )  # last_band_product
            # random keys
            for p1, i1, w1 in zip(range(bsz), rand_attn, attn_weights):
                # p1, i1, w1 corresponds to batch_dim i.e. following operation is done for each sequence in batch
                for p2, i2, w2 in zip(range(n_heads), i1, w1):
                    # p2, i2, w2 corresponds to head_dim i.e. following operation is done for each heads
                    for q_idx in range(1, len(i2) - 1):
                        attn_probs_view = attention_probs.view(
                            bsz,
                            n_heads,
                            from_seq_len // from_block_size,
                            from_block_size,
                            to_seq_len // to_block_size,
                            to_block_size,
                            )
                        right_slice = w2[q_idx - 1, :, 4 * to_block_size : -to_block_size]
                        attn_probs_view[p1, p2, q_idx + 1, :, i2[q_idx]] = right_slice.view(
                            from_block_size, n_rand_blocks, to_block_size
                        )

            # Second-last query block
            # corresponding to `second_last_context_layer`
            attention_probs[:, :, -2 * from_block_size : -from_block_size, :to_block_size] = second_last_attn_weights[
                                                                                             :, :, :, :to_block_size
                                                                                             ]  # 1st key block (global)
            attention_probs[
            :, :, -2 * from_block_size : -from_block_size, -3 * to_block_size :
            ] = second_last_attn_weights[
                :, :, :, to_block_size : 4 * to_block_size
                ]  # last three blocks (global + sliding)
            # random keys
            for p1, i1, w1 in zip(range(bsz), rand_attn, second_last_attn_weights):
                # p1, i1, w1 corresponds to batch_dim i.e. following operation is done for each sequence in batch
                for p2, i2, w2 in zip(range(n_heads), i1, w1):
                    # p2, i2, w2 corresponds to head_dim i.e. following operation is done for each heads
                    attn_probs_view = attention_probs.view(
                        bsz,
                        n_heads,
                        from_seq_len // from_block_size,
                        from_block_size,
                        to_seq_len // to_block_size,
                        to_block_size,
                        )
                    right_slice = w2[:, 4 * to_block_size :]
                    attn_probs_view[p1, p2, -2, :, i2[-1]] = right_slice.view(
                        from_block_size, n_rand_blocks, to_block_size
                    )

            # last query block
            # corresponding to `last_context_layer`
            attention_probs[:, :, -from_block_size:, :] = last_attn_weights  # all keys global

        else:
            attention_probs = None

        if self.compute_graph_stats:
            src_len = from_seq_len
            tgt_len = to_seq_len
            m = rand_attn.unsqueeze(-1) == torch.arange(to_seq_len // to_block_size).to(key_layer.device)
            cluster_mask = m.any(-2)
            # pad_mask.shape is (bs, src_len)
            pad_mask = to_mask.squeeze(2).squeeze(1).bool()
            cluster_mask = unblockify_attn(cluster_mask, from_block_size=from_block_size, to_block_size=to_block_size)
            ones = torch.ones(batch_size, self.num_attention_heads, to_block_size, src_len, device=cluster_mask.device, dtype=torch.bool)
            cluster_mask = torch.cat([ones, cluster_mask], dim=-2)
            cluster_mask = torch.cat([cluster_mask, ones], dim=-2)

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
            attn_logits = torch.matmul(query_layer, key_layer.transpose(-1, -2)) * rsqrt_d
            attn_logits += (1.0 - to_mask) * attn_mask_penalty
            gold_p = entmax15(attn_logits, dim=-1)
            # context_layer = torch.matmul(gold_p, value_layer)
            # context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            # new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
            # context_layer = context_layer.view(*new_context_layer_shape)

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
        from_block_size = to_block_size = self.block_size

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
