from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data


class DenoisingAE(nn.Module):
    def __init_(self, input_dim, latent_dim, output_dim):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, latent_dim),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, latent_dim),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, latent_dim),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, output_dim)
        )
    def forward(self, x):
        pass
