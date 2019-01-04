import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
import math
import torch.nn.functional as nn
from torch.autograd import Variable
import torch
from tensorflow.examples.tutorials.mnist import input_data


def P(z):
    mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)

    mb_size = 64
    Z_dim = 100
    X_dim = mnist.train.images.shape[1]
    y_dim = mnist.train.labels.shape[1]
    h_dim = 128
    c = 0
    lr = 1e-3

    def xavier_init(size):
        in_dim = size[0]
        xavier_stddev = 1. / np.sqrt(in_dim / 2.)
        return Variable(torch.randn(*size) * xavier_stddev, requires_grad=True)

    Wzh = xavier_init(size=[Z_dim, h_dim])
    bzh = Variable(torch.zeros(h_dim), requires_grad=True)

    Whx = xavier_init(size=[h_dim, X_dim])
    bhx = Variable(torch.zeros(X_dim), requires_grad=True)

    h = nn.relu(z @ Wzh + bzh.repeat(z.size(0), 1))
    X = nn.sigmoid(h @ Whx + bhx.repeat(h.size(0), 1))
    return X


def plot_a_numpy_array(a_numpy_array):
    if len(a_numpy_array.shape) != 2:
        side_size = int(math.sqrt(a_numpy_array.shape[0]))
        print(a_numpy_array.reshape(side_size, side_size).shape)
        plt.imshow(a_numpy_array.reshape(side_size, side_size), cmap='Greys_r')
        plt.show()
    else:
        plt.imshow(a_numpy_array, cmap='Greys_r')
        plt.show()


def plot_a_numpy_sample(samples, to_save=False, file_name=None):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    if to_save:
        if not os.path.exists('out/'):
            os.makedirs('out/')

        plt.savefig('out/' + file_name.replace('original_np_outs/', '').replace('.out', '') + '.png')
        plt.close(fig)
    else:
        plt.show()


def plot_a_numpy_file(file_name, to_save=False):
    samples = np.loadtxt(file_name, delimiter=',')
    if to_save:
        plot_a_numpy_sample(samples, to_save=to_save, file_name=file_name)
    else:
        plot_a_numpy_sample(samples, to_save=to_save)


def decode_a_z(an_z):
    if isinstance(an_z, np.ndarray):
        an_z = Variable(torch.from_numpy(an_z))
    samples = P(an_z).data.numpy()[:16]
    plot_a_numpy_sample(samples)


def decode_a_z_file(an_z_file):
    an_z = np.loadtxt(an_z_file, delimiter=',')
    decode_a_z(an_z)


# plot_a_numpy_file('original_np_outs/original_samples_99000.out')