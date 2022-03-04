import math
import pickle
import torch
from torch import nn
from torch.nn import Softmax
from transformers import RobertaForMaskedLM
from transformers.modeling_bert import BertSelfAttention
from entmax import entmax15
from extender.bigbird import bigbird_simulated_attention
from extender.utils import get_window_positions
from extender.extender_layers import HeadwiseLinearProjection
from extender.group_projections import predict_clusters


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


class ExtenderSimRobertaForMaskedLM(RobertaForMaskedLM):
    def __init__(self, config, window_size, projectors_path, centroids_path,
                 use_window=False, use_clusters=False, use_bigbird=False,
                 global_attention=False, top_clusters=1,
                 num_rand_blocks=3, block_size=1):
        super().__init__(config)

        states = torch.load(projectors_path)
        # centroids_all_layers = torch.load(centroids_path)
        # num_clusters = centroids_all_layers[0].shape[2]

        for i, layer in enumerate(self.roberta.encoder.layer):
            state_dict_q = states[i]['q']
            state_dict_k = states[i]['k']
            # centroids = centroids_all_layers[i]
            l_centr_path = centroids_path[:-len('0l_shared.pickle')] + str(i) + 'l_shared.pickle'
            with open(l_centr_path, 'rb') as handle:
                centroids = pickle.load(handle)

            layer.attention.self = ExtenderSimSelfAttention(config,
                                                            window_size=window_size,
                                                            q_proj_state=state_dict_q,
                                                            k_proj_state=state_dict_k,
                                                            centroids=centroids,
                                                            use_window=use_window,
                                                            use_clusters=use_clusters,
                                                            use_bigbird=use_bigbird,
                                                            global_attention=global_attention,
                                                            top_clusters=top_clusters,
                                                            num_rand_blocks=num_rand_blocks,
                                                            block_size=block_size)


class ExtenderSimSelfAttention(BertSelfAttention):
    def __init__(self, config, window_size, q_proj_state,
                 k_proj_state, centroids,
                 use_window=False, use_clusters=False, use_bigbird=False,
                 global_attention=False, top_clusters=1,
                 num_rand_blocks=3, block_size=1):
        super().__init__(config)
        self.use_window = use_window
        self.use_clusters = use_clusters
        self.use_bigbird = use_bigbird
        self.global_attention = global_attention
        self.top_clusters = top_clusters
        self.num_rand_blocks = num_rand_blocks
        self.block_size = block_size

        self.window_size = window_size
        self.centroids = centroids
        num_heads, head_size, projection_dim = q_proj_state['w'].shape

        self.proj_q = HeadwiseLinearProjection(
            num_heads, head_size, projection_dim)
        self.proj_k = HeadwiseLinearProjection(
            num_heads, head_size, projection_dim)

        self.proj_q.load_state_dict(q_proj_state)
        self.proj_k.load_state_dict(k_proj_state)

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
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # simulate extender limitations
        extender_attn_mask = attention_mask.eq(0).int() # attention mask has 0 in valid positions and -10000 in invalid ones
        extender_attn_mask = extender_attn_mask.squeeze(2).squeeze(1)
        extender_mask = self.extender_mask(attention_scores.shape,
                                           attention_scores.device,
                                           q=query_layer,
                                           k=key_layer,
                                           attn_mask=extender_attn_mask)
        extender_mask = (~extender_mask).float()  # reverse 0's and 1's
        extender_mask = extender_mask.masked_fill(extender_mask == 1, float('-inf'))
        attention_scores += extender_mask

        # Normalize the attention scores to probabilities.
        # attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = robust_entmax15(attention_scores, dim=-1)
        # attention_probs = robust_softmax(attention_scores)

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

        kqs = {'src_q': query_layer, 'src_k': key_layer}
        outputs = (context_layer, kqs) if output_attentions else (context_layer,)
        return outputs

    def extender_mask(self, shape, device, q, k, attn_mask=None):
        head_mask = torch.zeros(shape).to(device)

        if self.use_window:
            head_mask += self.window_mask(shape, device)
        if self.use_clusters:
            head_mask += self.cluster_mask(shape, device, q, k)
        if self.global_attention:
            head_mask += self.global_mask(shape, device)
        if self.use_bigbird:
            head_mask += self.bigbird_mask(shape, device, attn_mask)

        return head_mask.bool()

    def bigbird_mask(self, shape, device, attention_mask):
        bs, nh, slen, slen = shape
        pairwise_mask = attention_mask.unsqueeze(-1) & attention_mask.unsqueeze(1)
        pairwise_mask.unsqueeze_(1)

        bbmask = bigbird_simulated_attention(
            pairwise_mask.cpu().numpy(),
            num_attention_heads=nh,
            num_rand_blocks=self.num_rand_blocks,  # default from bigbird repo is 3
            from_seq_length=slen,
            to_seq_length=slen,
            from_block_size=self.block_size,
            to_block_size=self.block_size,
            max_seq_len=slen
        )  # includes a window of length==3 by default + global attention to 1st+last token
        return bbmask.to(device)

    def global_mask(self, shape, device):
        batch, heads, q_len, k_len = shape
        window_mask = torch.zeros(shape).to(device)
        window_mask[:, :, 0, :] = 1  # CLS attends everywhere
        window_mask[:, :, :, 0] = 1  # all tokens attend CLS
        return window_mask

    def window_mask(self, shape, device):
        window_mask = torch.zeros(shape).to(device)

        if self.window_size is not None and self.window_size > 0:
            w = self.window_size
            _, _, n_toks, _ = shape

            window_mask += torch.diag(torch.ones(n_toks).to(device))
            for offset in range(1, w//2 + 1):
                window_mask += torch.diag(torch.ones(n_toks - offset).to(device), diagonal=offset)
                window_mask += torch.diag(torch.ones(n_toks - offset).to(device), diagonal=-offset)
            # force window of first few and last few words have size==w
            for idx in range(w//2):
                window_mask[:, :, idx, :w] = 1
                window_mask[:, :, n_toks - 1 - idx, -w:] = 1

            # # check if result matches quantization.py
            # window_inds = get_window_positions(n_toks, w)  # .to(device)
            # unique_inds = (window_inds+torch.arange(0, 1, 1/n_toks).unsqueeze(1)).unique().shape[0]
            # assert unique_inds == window_mask[0, 0, :, :].sum().item()  # count of elements matches count of quantization.py
            # assert (window_mask.gather(3, window_inds.unsqueeze(0).unsqueeze(0).to(device)) == 0).sum() == 0  # no 0's inside quantization.py's window

        return window_mask

    def cluster_mask(self, shape, device, q, k):
        low_q = self.proj_q(q)
        low_k = self.proj_k(k)

        buckets_q = predict_clusters(low_q, self.centroids, top_clusters=self.top_clusters, masked_lm_format=True)  # [:, :, :, 0]  # assume only 1 round
        buckets_k = predict_clusters(low_k, self.centroids, top_clusters=self.top_clusters, masked_lm_format=True)  # [:, :, :, 0]

        k_clusters, proj_dim = self.centroids[0].cluster_centers_[0].shape
        batch, heads, toks, top_clusters = buckets_q.shape
        all_keys_per_cluster, cluster_mask = None, None

        for i in range(top_clusters):
            k_per_c = (torch.arange(k_clusters).view(1, 1, -1, 1) ==
                       buckets_k[:, :, :, i].unsqueeze(-2).cpu()).float()
            all_keys_per_cluster = k_per_c if all_keys_per_cluster is None else k_per_c + all_keys_per_cluster

        for i in range(top_clusters):
            cluster_m = torch.gather(all_keys_per_cluster,
                                     dim=2,
                                     index=buckets_q[:, :, :, i].unsqueeze(-1).repeat(1, 1, 1, toks).long().cpu())

            cluster_mask = cluster_m if cluster_mask is None else cluster_m + cluster_mask

        # # only for assert
        # blank_cluster_mask = torch.zeros(shape)
        # for b in range(batch):
        #     for h in range(heads):
        #         keys_per_cluster = (torch.arange(k_clusters).unsqueeze(1) == buckets_k[b, h, :].unsqueeze(0).cpu()).float()
        #         assert torch.all(all_keys_per_cluster[b, h] == keys_per_cluster)
        #         blank_cluster_mask[b, h] = keys_per_cluster[buckets_q[b, h, :].long()]
        #         assert torch.all(cluster_mask[b, h] ==blank_cluster_mask[b, h])

        return cluster_mask.to(device)


if __name__ == '__main__':
    # roberta = EntmaxRobertaForMaskedLM(config="/home/agois/longformer_stuff/entmax-roberta-maia/config.json")
    roberta = ExtenderSimRobertaForMaskedLM.from_pretrained('roberta-base')

    inp = torch.remainder(torch.LongTensor(16, 512), 1000)  # random integers between 0-999; batch:16, len:512, (model dim: 64*12)
    out = roberta(input_ids=inp, output_attentions=True)
    print('bye')
