import torch
from transformers import RobertaForMaskedLM
from extender.extender_layers import ExtenderSelfAttention


class ExtenderRobertaForMaskedLM(RobertaForMaskedLM):
    def __init__(self, config, projectors_path, centroids_path):
        super().__init__(config)

        states = torch.load(projectors_path)
        centroids_all_layers = torch.load(centroids_path)
        num_clusters = centroids_all_layers[0].shape[2]

        for i, layer in enumerate(self.roberta.encoder.layer):
            state_dict_q = states[i]['q']
            state_dict_k = states[i]['k']
            centroids = centroids_all_layers[i]
            layer.attention.self = ExtenderSelfAttention.load(state_dict_q, state_dict_k,
                                                              query=layer.attention.self.query,
                                                              key=layer.attention.self.key,
                                                              value=layer.attention.self.value,
                                                              bucket_size=num_clusters, dropout=0.,
                                                              attn_func='entmax15', centroids=centroids)


if __name__ == '__main__':
    roberta = ExtenderRobertaForMaskedLM.from_pretrained('roberta-base',
                                                       projectors_path='/home/agois/longformer_stuff/entmax-roberta-maia/projections_dim4_shared.pickle',
                                                       centroids_path='/home/agois/longformer/centroids/kqs_enc-attn.pt_4r_8s_1n_shared.pickle')

    inp = torch.remainder(torch.LongTensor(16, 512), 1000)  # random integers between 0-999; batch:16, len:512, (model dim: 64*12)
    prediction_scores, (sparsity, entropies) = roberta(input_ids=inp, output_attentions=True)
    print('bye')
