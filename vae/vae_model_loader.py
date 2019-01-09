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
from vae_models import VAE_Vanilla_Test, VAE_Conv

running_time = get_current_time()

model_path = 'results/conv/mixed/2019-01-07-20-36-50/model.torch'
# model_path = 'results/vanilla/original/2019-01-07-19-26-37/model.torch'

# model_path = 'results/inverse/2019-01-04-23-53-46/model.torch'
# z_path = model_path.replace('model.torch', 'z.txt')

model_type = 'vanilla'
if 'vanilla' in model_path:
    model_type = 'vanilla'
elif 'conv' in model_path:
    model_type = 'conv'
else:
    model_type = 'vanilla'

if 'mixed' in model_path:
    to_inverse = 'mixed'
else:
    if 'inverse' in model_path:
        to_inverse = True
    else:
        to_inverse = False

if to_inverse == 'mixed':
    save_prefix = 'results/' + model_type + '/mixed/'
else:
    if to_inverse:
        save_prefix = 'results/' + model_type + '/inverse/'
    else:
        save_prefix = 'results/' + model_type + '/original/'

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default='1', metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--lat_dim', type=int, default=20, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--img_size', type=int, default=28, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.img_size = 64
args.lat_dim = 100
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

transform = transforms.Compose([
        transforms.Scale(args.img_size),
        transforms.ToTensor()
        # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
if isinstance(to_inverse, bool):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transform),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transform),
        batch_size=args.batch_size, shuffle=False, **kwargs)
else:
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transform),
        batch_size=int(args.batch_size/2), shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transform),
        batch_size=int(args.batch_size/2), shuffle=False, **kwargs)


def test(epoch, to_save=False):
    save_path_prefix = save_prefix + running_time + '/'
    model.eval()
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            if to_inverse == 'mixed':
                data = torch.cat((data, 1 - data), 0)
            elif to_inverse:
                data = 1 - data

            data = data.to(device)
            recon_batch, mu, logvar, z = model(data)
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 1, args.img_size, args.img_size)[:n]])

                if to_save:
                    if not os.path.exists(save_path_prefix):
                        os.makedirs(save_path_prefix)
                    save_image(comparison.cpu(),
                               save_path_prefix + 'reconstruction_' + str(epoch) + '.png', nrow=n)
                break


def z_vector_analyzer(z_path_1):
    z_1 = np.loadtxt(z_path_1, delimiter=',')
    # z_2 = np.loadtxt(z_path_2, delimiter=',')
    # z_3 = np.loadtxt(z_path_3, delimiter=',')
    # z_4 = np.loadtxt(z_path_4, delimiter=',')
    # z_5 = np.loadtxt(z_path_5, delimiter=',')
    # print(np.max(z_1 - z_2))
    return z_1


def translate_z(zs, model):
    if isinstance(zs, np.ndarray):
        zs = torch.from_numpy(zs).float()
    running_time = get_current_time()
    save_path_prefix = 'results/translation_z/' + running_time + '/'
    sample = model.decode(zs).cpu()
    if not os.path.exists(save_path_prefix):
        os.makedirs(save_path_prefix)
    save_image(sample.view(zs.shape[0], 1, args.img_size, args.img_size),
               save_path_prefix + 'z_translation' + '.png')


def generate_z_vectors(model, to_inverse, up_to):
    store_data_np = np.empty([0, args.lat_dim])
    store_label_np = np.empty([0, ])

    for a_loader in [test_loader]:
        for i, (data, label) in enumerate(a_loader):
            if to_inverse == 'mixed':
                data = torch.cat((data, 1 - data), 0)
            elif to_inverse:
                data = 1 - data
            if model_type == 'conv':
                z, mu, logvar = model.encode(data)
            else:
                mu, logvar = model.encode(data)
                z = model.reparameterize(mu, logvar)

            store_data_np = np.concatenate([store_data_np, z.data.numpy()], axis=0)

            if to_inverse == 'mixed':
                label_half = label.data.numpy() + 0.5
                double_label = np.concatenate([label, label_half], axis=0)
                store_label_np = np.concatenate([store_label_np, double_label], axis=0)
            else:
                store_label_np = np.concatenate([store_label_np, label.data.numpy()], axis=0)

            if len(store_label_np) >= up_to:
                break

    store_label_np = store_label_np.astype(np.float32)
    print('store data np: ', store_data_np.shape)
    np.savetxt('z_vectors.txt', store_data_np, delimiter='\t')
    np.savetxt('z_labels.txt', store_label_np, delimiter='\n', fmt='%.1f')


if model_type == 'vanilla':
    print('Loading vanilla ...')
    model = VAE_Vanilla_Test(args=args, z_dim=args.lat_dim,
                             to_save_z=True, save_path=model_path.replace('model.torch', '')).to(device)
elif model_type == 'conv':
    print('Loading conv ...')
    model = VAE_Conv(args=args, z_dim=args.lat_dim, to_save_z=True, save_path=model_path.replace('model.torch', '')).to(device)

model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()
test(0)

arithmetic = z_vector_analyzer(model_path.replace('model.torch', 'z.txt'))

model.eval()
with torch.no_grad():
    for i, (data, _) in enumerate(test_loader):
        model.translate_z(arithmetic, data, to_save=True)
        break

# zs = (np.zeros(shape=[1, 20]) * 100)
# translate_z(zs, model)

up_to = 20000
generate_z_vectors(model, to_inverse, up_to)
z_s = np.loadtxt('z_vectors.txt', delimiter='\t')
y_s = np.loadtxt('z_labels.txt', delimiter='\t')
gm.compute_TSNE_projection_of_latent_space(z_s[:up_to], y_s[:up_to],
                                           save_path=model_path.replace('model.torch', 'tsne.png'))
