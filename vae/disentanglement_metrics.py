import torch
import numpy as np
from vae_models import VAE_Vanilla


class HigginsDisentanglementMetric:
    def __init__(self, z, a_VAE, L):
        self.a_VAE = a_VAE
        self.L = L
        self.z = z
        if not isinstance(self.z, torch.Tensor):
            self.z = torch.from_numpy(self.z)

    def get_a_z_diff(self, k):
        v_1 = torch.randn(self.z.shape)
        v_2 = torch.randn(self.z.shape)

        v_1[k] = v_2[k] = self.z[k]

        x_1 = self.a_VAE.decode(v_1)
        x_2 = self.a_VAE.decode(v_2)

        mu_1, logvar_1 = self.a_VAE.encode(x_1.view(-1, 64 * 64))
        z_1 = self.reparameterize(mu_1, logvar_1)

        mu_2, logvar_2 = self.a_VAE.encode(x_2.view(-1, 64 * 64))
        z_2 = self.reparameterize(mu_2, logvar_2)

        z_diff = np.abs((z_1 - z_2).data.numpy())
        return z_diff

    def get_L_z_diffs(self, k):
        z_diffs = np.zeros(shape=self.z.shape)
        for l in range(self.L):
            z_diffs += self.get_a_z_diff(k)
        return z_diffs / self.L

    def iterate_all_z_dims(self):
        z_diffs = np.zeros(shape=self.z.shape)
        for k in range(len(self.z)):
            z_diffs += self.get_L_z_diffs(k)
        return z_diffs / len(self.z)


z_path_1 = 'results/conv/mixed/2019-01-07-20-36-50/z.txt'
a_np_z = np.loadtxt(z_path_1, delimiter=',')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE_Vanilla(z_dim=a_np_z.shape[1]).to(device)
# print(a_np_z)
higgins_disentanglement_metric = HigginsDisentanglementMetric(a_np_z[0], device, 2)
higgins_disentanglement_metric.iterate_all_z_dims()