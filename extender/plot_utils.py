import os

from matplotlib import pyplot as plt

import numpy as np


def plot_train_and_eval_loss(layer, title, train_and_eval_data):
    train_loss, eval_loss = zip(*train_and_eval_data)
    plt.figure()
    plt.title('Layer {}'.format(layer))
    plt.xlabel('steps')
    plt.ylabel(title)
    plt.plot(train_loss, label='Train')
    plt.plot(eval_loss, label='Eval')
    plt.legend()
    if not os.path.exists('figs/curves'):
        os.makedirs('figs/curves')
    plt.savefig('figs/curves/{}_{}.png'.format(title, layer), bbox_inches='tight')


def plot_distance_and_buckets(layer, num_heads, dist_x_buckets, plot_dist='euclidean', plot_type='hist', plot_bins=-1):
    rows = 1
    cols = num_heads // rows
    fig, axs = plt.subplots(rows, cols, figsize=(30, 3))
    for h in range(num_heads):
        i, j = h // cols, h % cols
        ax = axs[j] if rows == 1 else axs[i][j]
        x, y = zip(*dist_x_buckets[h])
        if plot_type == 'scatter':
            ax.plot(x, y, 'x')
        elif plot_type == 'hist':
            if plot_bins == -1:
                counts, bins = np.histogram(x, bins='auto')
            else:
                counts, bins = np.histogram(x, bins=plot_bins)
            ax.hist(x, bins=bins, weights=y)
            ax.set_yscale('log', nonposy='clip')
            ax.set_xlim([min(bins), max(bins)])
            ax.set_xticks(bins)
            ax.set_xticklabels(['{:.2f}'.format(b) for b in bins])
        else:
            raise NotImplementedError
        ax.set_title('Head {}'.format(h))
        ax.set_xlabel('{} distance'.format(plot_dist))
        ax.set_ylabel('number of buckets in common')
    fig.tight_layout()
    if not os.path.exists('figs/dist_x_buckets'):
        os.makedirs('figs/dist_x_buckets')
    figpath = 'figs/dist_x_buckets/layer{}.png'.format(layer)
    plt.savefig(figpath, bbox_inches='tight')