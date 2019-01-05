from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os
from general_methods import get_current_time
import numpy as np
from torch.autograd import Variable
import pandas as pd
import general_methods as gm

running_time = get_current_time()

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=False, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=False, **kwargs)


class VAE(nn.Module):
    def __init__(self, save_z=False, save_path=None):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

        self.save_z = save_z
        self.save_path = save_path.replace('model.torch', '')

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        if self.save_z:
            # print(z.data.numpy().shape)
            save_name = self.save_path + 'z.txt'
            np.savetxt(save_name, z.data.numpy(), delimiter=',')
        # self.translate_z(z, x, to_save=True)
        return self.decode(z), mu, logvar

    def translate_z(self, z, data, to_save=False):
        if isinstance(z, str):
            z = np.loadtxt(z, delimiter=',')
            z = torch.from_numpy(z).float()
        elif isinstance(z, np.ndarray):
            z = torch.from_numpy(z).float()

        recon_batch = self.decode(z)
        n = min(data.size(0), 8)
        comparison = torch.cat([data[:n],
                                recon_batch.view(args.batch_size, 1, 28, 28)[:n]])

        if to_save:
            save_image(comparison.cpu(),
                       'translation.png', nrow=n)


def test(epoch, inverse=False, to_save=False):
    if inverse:
        save_prefix = 'results/inverse/'
    else:
        save_prefix = 'results/original/'
    save_path_prefix = save_prefix + running_time + '/'
    model.eval()
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            if inverse:
                data = 1 - data

            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 1, 28, 28)[:n]])

                if to_save:
                    if not os.path.exists(save_path_prefix):
                        os.makedirs(save_path_prefix)
                    save_image(comparison.cpu(),
                               save_path_prefix + 'reconstruction_' + str(epoch) + '.png', nrow=n)
            break


def z_vector_analyzer(z_path_1, z_path_2, z_path_3, z_path_4, z_path_5):
    z_1 = np.loadtxt(z_path_1, delimiter=',')
    z_2 = np.loadtxt(z_path_2, delimiter=',')
    z_3 = np.loadtxt(z_path_3, delimiter=',')
    z_4 = np.loadtxt(z_path_4, delimiter=',')
    z_5 = np.loadtxt(z_path_5, delimiter=',')
    # print(np.max(z_1 - z_2))
    return z_5


def translate_z(zs, model):
    if isinstance(zs, np.ndarray):
        zs = torch.from_numpy(zs).float()
    running_time = get_current_time()
    save_path_prefix = 'results/translation_z/' + running_time + '/'
    sample = model.decode(zs).cpu()
    if not os.path.exists(save_path_prefix):
        os.makedirs(save_path_prefix)
    save_image(sample.view(zs.shape[0], 1, 28, 28),
               save_path_prefix + 'z_translation' + '.png')


def generate_z_vectors(model, to_inverse):
    store_data_np = np.empty([0, 20])
    store_label_np = np.empty([0, ])

    for a_loader in [test_loader]:
        for i, (data, label) in enumerate(a_loader):
            if to_inverse:
                data = 1 - data
            mu, logvar = model.encode(data.view(-1, 784))
            z = model.reparameterize(mu, logvar)

            store_data_np = np.concatenate([store_data_np, z.data.numpy()], axis=0)
            store_label_np = np.concatenate([store_label_np, label], axis=0)

    store_label_np = store_label_np.astype(int)
    print('store data np: ', store_data_np.shape)
    np.savetxt('z_vectors.txt', store_data_np, delimiter='\t')
    np.savetxt('z_labels.txt', store_label_np, delimiter='\n', fmt='%i')


# model_path = 'results/original/2019-01-04-23-54-33/model.torch'
model_path = 'results/inverse/2019-01-04-23-53-46/model.torch'
# z_path = model_path.replace('model.torch', 'z.txt')

if 'inverse' in model_path:
    to_inverse = True
else:
    to_inverse = False

model = VAE(save_z=True, save_path=model_path)
model.load_state_dict(torch.load(model_path))

# model.eval()
# test(0, inverse=to_inverse)

arithmetic = z_vector_analyzer('results/original/2019-01-04-23-54-33/z.txt',
                               'results/original/2019-01-05-09-55-24/z.txt',
                               'results/inverse/2019-01-04-23-53-46/z.txt',
                               'results/inverse/2019-01-05-09-55-20/z.txt',
                               'results/original/2019-01-05-18-43-56/z.txt')

# model.eval()
# with torch.no_grad():
#     for i, (data, _) in enumerate(test_loader):
#         model.translate_z(arithmetic, data, to_save=True)
#         break

# zs = (np.zeros(shape=[1, 20]) * 100)
# translate_z(zs, model)

generate_z_vectors(model, to_inverse)
z_s = np.loadtxt('z_vectors.txt', delimiter='\t')
y_s = np.loadtxt('z_labels.txt', delimiter='\t')
gm.compute_TSNE_projection_of_latent_space(z_s[:10000], y_s[:10000])
