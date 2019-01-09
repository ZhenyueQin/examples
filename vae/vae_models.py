from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
from torchvision.utils import save_image


class VAE_Vanilla(nn.Module):
    def __init__(self, z_dim):
        super(VAE_Vanilla, self).__init__()

        self.fc1 = nn.Linear(64*64, 1000)
        self.fc21 = nn.Linear(1000, z_dim)
        self.fc22 = nn.Linear(1000, z_dim)
        self.fc3 = nn.Linear(z_dim, 1000)
        self.fc4 = nn.Linear(1000, 64*64)

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
        mu, logvar = self.encode(x.view(-1, 64*64))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z


class VAE_Vanilla_Test(nn.Module):
    def __init__(self, args,  z_dim, to_save_z=False, save_path=None):
        super(VAE_Vanilla_Test, self).__init__()

        self.fc1 = nn.Linear(args.img_size * args.img_size, 1000)
        self.fc21 = nn.Linear(1000, z_dim)
        self.fc22 = nn.Linear(1000, z_dim)
        self.fc3 = nn.Linear(z_dim, 1000)
        self.fc4 = nn.Linear(1000, args.img_size * args.img_size)

        self.to_save_z = to_save_z
        self.save_path = save_path.replace('model.torch', '')

        self.args = args

    def encode(self, x):
        h1 = F.relu(self.fc1(x.view(-1, self.args.img_size * self.args.img_size)))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 64 * 64))
        z = self.reparameterize(mu, logvar)
        if self.to_save_z:
            # print(z.data.numpy().shape)
            save_name = self.save_path + 'z.txt'
            np.savetxt(save_name, z.data.numpy(), delimiter=',')
        # self.translate_z(z, x, to_save=True)
        return self.decode(z), mu, logvar, z

    def translate_z(self, z, data, to_save=False):
        if isinstance(z, str):
            z = np.loadtxt(z, delimiter=',')
            z = torch.from_numpy(z).float()
        elif isinstance(z, np.ndarray):
            z = torch.from_numpy(z).float()

        recon_batch = self.decode(z)
        n = 8
        comparison = recon_batch.view(self.args.batch_size, 1, self.args.img_size, self.args.img_size)

        if to_save:
            save_image(comparison.cpu(),
                       'translation.png', nrow=n)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


# class UnFlatten(nn.Module):
#     def forward(self, input, size=1024):
#         print('input.view(input.size(0), size, 1, 1): ', input.view(input.size(0), size, 1, 1).shape)
#         return input.view(input.size(0), size, 1, 1)


class VAE_Conv(nn.Module):
    def __init__(self, args, z_dim=32, to_save_z=False, save_path=None):
        super(VAE_Conv, self).__init__()

        self.to_save_z = to_save_z
        self.save_path = save_path
        self.args = args

        d = 128

        self.encoder = nn.Sequential(
            nn.Conv2d(1, d, 4, 2, 1),
            nn.Conv2d(d, d * 2, 4, 2, 1),
            nn.BatchNorm2d(d * 2),
            nn.Conv2d(d * 2, d * 4, 4, 2, 1),
            nn.BatchNorm2d(d * 4),
            nn.Conv2d(d * 4, d * 8, 4, 2, 1),
            nn.BatchNorm2d(d * 8),
            nn.Conv2d(d * 8, 1, 4, 1, 0)
        )

        self.fc1 = nn.Linear(16384, 1000)
        self.fc2_1 = nn.Linear(1000, z_dim)
        self.fc2_2 = nn.Linear(1000, z_dim)
        self.fc3 = nn.Linear(z_dim, 100)

        self.conv1 = nn.Conv2d(1, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.conv4 = nn.Conv2d(d * 4, d * 8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d * 8)

        self.deconv1 = nn.ConvTranspose2d(100, d * 8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d * 8)
        self.deconv2 = nn.ConvTranspose2d(d * 8, d * 4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d * 4)
        self.deconv3 = nn.ConvTranspose2d(d * 4, d * 2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d * 2)
        self.deconv4 = nn.ConvTranspose2d(d * 2, d, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, 1, 4, 2, 1)

    def encoding(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = F.sigmoid(self.fc1(Flatten().forward(x)))
        return x

    def decoding(self, z):
        print('z: ', z.shape)
        z = z.view(-1, 100, 1, 1)
        x = F.relu(self.deconv1_bn(self.deconv1(z)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = F.sigmoid(self.deconv5(x))
        return x

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def bottleneck(self, h):
        mu, logvar = self.fc2_1(h), self.fc2_2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoding(x)
        z, mu, logvar = self.bottleneck(h)
        if self.to_save_z:
            # print(z.data.numpy().shape)
            save_name = self.save_path + 'z.txt'
            np.savetxt(save_name, z.data.numpy(), delimiter=',')
        return z, mu, logvar

    def decode(self, z):
        z = self.decoding(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        recons_z = self.decode(z)
        return recons_z, mu, logvar, z

    def translate_z(self, z, data, to_save=False):
        if isinstance(z, str):
            z = np.loadtxt(z, delimiter=',')
            z = torch.from_numpy(z).float()
        elif isinstance(z, np.ndarray):
            z = torch.from_numpy(z).float()

        recon_batch = self.decode(z)
        # n = min(data.size(0), 8)
        n = 8
        comparison = recon_batch.view(self.args.batch_size, 1, self.args.img_size, self.args.img_size)
        print('comparison: ', comparison.shape)
        if to_save:
            save_image(comparison.cpu(),
                       'translation.png', nrow=n)


