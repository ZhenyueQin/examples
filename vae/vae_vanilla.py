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
from vae_models import VAE_Vanilla, VAE_Conv

print(torch.__version__)

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
args.lat_dim = 100
args.img_size = 64
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")
print('device: ', device)

to_inverse = 'mixed'
model_type = 'vanilla'
if to_inverse == 'mixed':
    save_prefix = 'results/' + model_type + '/mixed/'
else:
    if to_inverse:
        save_prefix = 'results/' + model_type + '/inverse/'
    else:
        save_prefix = 'results/' + model_type + '/original/'


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

if model_type == 'vanilla':
    model = VAE_Vanilla(z_dim=args.lat_dim).to(device)
elif model_type == 'conv':
    model = VAE_Conv(args=args, z_dim=args.lat_dim).to(device)

print('is on cuda: ', next(model.parameters()).is_cuda)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar, z=None):
    BCE = F.binary_cross_entropy(recon_x.view(-1, args.img_size*args.img_size),
                                 x.view(-1, args.img_size*args.img_size)) * args.img_size * args.img_size * args.batch_size

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # if z is not None:
    #     z_loss =

    return BCE + KLD


def train(epoch, to_inverse=False):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        # print('img max: ', np.max(data.data.numpy()), '; img min: ', np.min(data.data.numpy()))
        if to_inverse == 'mixed':
            data = torch.cat((data, 1 - data), 0)
        elif to_inverse:
            data = 1 - data
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar, z = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            if to_inverse == 'mixed':
                data = torch.cat((data, 1 - data), 0)
            elif to_inverse:
                data = 1 - data

            data = data.to(device)
            recon_batch, mu, logvar, z = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 1, args.img_size, args.img_size)[:n]])
                if not os.path.exists(save_prefix + running_time + '/'):
                    os.makedirs(save_prefix + running_time + '/')
                save_image(comparison.cpu(),
                           save_prefix + running_time + '/' + 'reconstruction_' + str(epoch) + '.png', nrow=n)
                break

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

running_time = get_current_time()
print('Starting VAE normal at: ', running_time)

save_path_prefix = save_prefix + running_time + '/'

for epoch in range(1, args.epochs + 1):
    train(epoch, to_inverse=to_inverse)
    test(epoch)
    with torch.no_grad():
        sample = torch.randn(64, args.lat_dim).to(device)
        sample = model.decode(sample).cpu()
        if not os.path.exists(save_path_prefix):
            os.makedirs(save_path_prefix)
        save_image(sample.view(64, 1, args.img_size, args.img_size),
                    save_path_prefix + 'sample_' + str(epoch) + '.png')

model_path = save_path_prefix + 'model.torch'
torch.save(model.state_dict(), model_path)
