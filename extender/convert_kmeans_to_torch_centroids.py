"""
Usage:
python3 convert_kmeans_to_torch_centroids.py --data $data -r 4 -s 4 -l 6 --share-projectors --cluster-rounds 1
python3 convert_kmeans_to_torch_centroids.py --data $data -r 4 -s 8 -l 6 --share-projectors --cluster-rounds 1
python3 convert_kmeans_to_torch_centroids.py --data $data -r 4 -s 12 -l 6 --share-projectors --cluster-rounds 1
python3 convert_kmeans_to_torch_centroids.py --data $data -r 4 -s 16 -l 6 --share-projectors --cluster-rounds 1
python3 convert_kmeans_to_torch_centroids.py --data $data -r 4 -s 20 -l 6 --share-projectors --cluster-rounds 1
"""

import argparse
import os
import pickle
import torch
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', dest='rounds', help='Hashing rounds', type=int, default=4)
    parser.add_argument('-s', dest='bucket_size', type=int, default=16, help='Bucket size')
    parser.add_argument('-l', '--num-layers', type=int)
    parser.add_argument('--data', type=str)
    parser.add_argument('--share-projectors', action='store_true', help='Whether to share projectors.')
    parser.add_argument('--cluster-rounds', type=int, default=1)
    parser.add_argument('--top_clusters', type=int, default=1, help='use "top_clusters" closest to each point.')
    args = parser.parse_args()

    centroids = []
    for layer in range(args.num_layers):
        # e.g. train20k-h8-new-attentions-40-1000000.pt_4r_4s_1n_0l_shared.pickle
        kmeans_path = "kmeans/{}_{}r_{}s_{}n_{}l_{}.pickle".format(
            os.path.basename(args.data),
            args.rounds,  # projected vectors size
            args.bucket_size,  # how many clusters
            args.cluster_rounds,  # how many runs
            layer,
            'shared' if args.share_projectors else 'indep'
        )
        print('Loading pretrained kmeans from: {}'.format(kmeans_path))
        with open(kmeans_path, 'rb') as handle:
            clusters_per_head = pickle.load(handle)
        centroids_l = np.stack([h.cluster_centers_ for h in clusters_per_head])
        # centroids_l.shape is (num_heads, num_runs, num_clusters, projection_size)
        centroids.append(torch.from_numpy(centroids_l))

    centroids_path = "centroids/{}_{}r_{}s_{}n_{}.pickle".format(
        os.path.basename(args.data),
        args.rounds,  # projected vectors size
        args.bucket_size,  # how many clusters
        args.cluster_rounds,  # how many runs
        'shared' if args.share_projectors else 'indep'
    )
    if not os.path.exists('centroids'):
        os.mkdir('centroids')
    print('Saving torch centroids to: {}'.format(centroids_path))
    torch.save(centroids, centroids_path)