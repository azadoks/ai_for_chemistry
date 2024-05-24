# -*- coding: utf-8 -*-
"""Molecular variational auto-encoder.

Adapted from
https://github.com/aksub99/molecular-vae/blob/1e457410110f84a96124dbdf2d45fa318309b6ba/Molecular_VAE.ipynb.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ('Encoder', 'Decoder', 'VAE', 'loss')


def loss(x_decoded_mean, x, z_mean, z_logvar):
    xent_loss = F.binary_cross_entropy(x_decoded_mean, x, size_average=False)
    kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
    return xent_loss + kl_loss


class Encoder(nn.Module):
    """Encoder module."""
    def __init__(self, embedding_dim: int=463, latent_dim: int=292):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.conv_1 = nn.Conv1d(embedding_dim, 9, kernel_size=9)
        self.conv_2 = nn.Conv1d(9, 9, kernel_size=9)
        self.conv_3 = nn.Conv1d(9, 10, kernel_size=11)
        self.linear_0 = nn.Linear(610, 435)  # input_dim depends on embedding_dim via convolutions
        self.linear_1 = nn.Linear(435, latent_dim)
        self.linear_2 = nn.Linear(435, latent_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv_1(x))
        x = self.relu(self.conv_2(x))
        x = self.relu(self.conv_3(x))
        x = x.view(x.size(0), -1)
        x = F.selu(self.linear_0(x))
        x_mean = self.linear_1(x)
        x_logvar = self.linear_2(x)
        return x_mean, x_logvar


class Decoder(nn.Module):
    """Decoder module."""
    def __init__(self, embedding_dim: int=463, latent_dim: int=292, vocabulary_size: int=87):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.vocabulary_size = vocabulary_size
        self.linear_3 = nn.Linear(self.latent_dim, self.latent_dim)
        self.gru = nn.GRU(self.latent_dim, 501, 3, batch_first=True)
        self.linear_4 = nn.Linear(501, self.vocabulary_size)

    def forward(self, z):
        z = F.selu(self.linear_3(z))
        z = z.view(z.size(0), 1, z.size(-1)).repeat(1, self.embedding_dim, 1)
        output, hn = self.gru(z)
        out_reshape = output.contiguous().view(-1, output.size(-1))
        y0 = F.softmax(self.linear_4(out_reshape), dim=1)
        y = y0.contiguous().view(output.size(0), -1, y0.size(-1))
        return y


class VAE(nn.Module):
    """Variational auto-encoder module."""
    def __init__(self, embedding_dim: int=120, latent_dim: int=292, vocabulary_size: int=33):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.vocab_size = vocabulary_size
        self.encoder = Encoder(embedding_dim, latent_dim)
        self.decoder = Decoder(embedding_dim, latent_dim, vocabulary_size)

    def sampling(self, z_mean, z_logvar):
        epsilon = 1e-2 * torch.randn_like(z_logvar)
        return torch.exp(0.5 * z_logvar) * epsilon + z_mean

    def forward(self, x):
        z_mean, z_logvar = self.encoder(x)
        z = self.sampling(z_mean, z_logvar)
        return self.decoder(z), z_mean, z_logvar
