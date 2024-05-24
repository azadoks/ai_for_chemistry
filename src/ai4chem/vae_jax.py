# -*- coding: utf-8 -*-
"""Molecular variational auto-encoder.

Adapted from
https://github.com/aksub99/molecular-vae/blob/1e457410110f84a96124dbdf2d45fa318309b6ba/Molecular_VAE.ipynb.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn

__all__ = ('Encoder', 'Decoder', 'VAE')


class Encoder(nn.Module):
    """Encoder module."""

    latents: int

    @nn.compact
    def __call__(self, x):
        # self.conv_1 = nn.Conv1d(120, 9, kernel_size=9)
        x = nn.Conv(
            features=9,
            kernel_size=(9, 9),
            name='conv_1'
        )(x)
        x = nn.relu(x)
        # self.conv_2 = nn.Conv1d(9, 9, kernel_size=9)
        x = nn.Conv(
            features=9,
            kernel_size=(9, 9),
            name='conv_2'
        )(x)
        x = nn.relu(x)
        # self.conv_3 = nn.Conv1d(9, 10, kenel_size=11)
        x = nn.Conv(
            features=10,
            kernel_size=(11, 11),
            name='conv_3'
        )(x)
        x = nn.relu(x)
        x = x.flatten()
        # self.linear_0 = nn.Linear(70, 435)
        x = nn.Dense(
            features = 436,
            name='linear_0'
        )(x)
        x = nn.selu(x)
        # self.linear_1 = nn.Linear(435, 292)
        mean_x = nn.Dense(
            features=self.latents,
            name='linear_1'
        )(x)
        # self.linear_2 = nn.Linear(435, 292)
        logvar_x = nn.Dense(
            features=self.latents,
            name='linear_2'
        )(x)
        return mean_x, logvar_x


class Decoder(nn.Module):
    """Decoder module."""

    @nn.compact
    def __call__(self, z):
        # self.linear_3 = nn.Linear(292, 292)
        z = nn.Dense(
            features=292,
            name='linear_3'
        )(z)
        z = nn.selu(z)
        # self.gru = nn.GRU(292, 501, 3, batch_first=True)
        x, _ = nn.GRUCell(
            features=501,
            name='gru'
        )
        # self.linear_4 = nn.Linear(501, 33)
        x = nn.Dense(
            features=33,
            name='linear_4'
        )(x)
        x = nn.softmax(x)
        return x


class VAE(nn.Module):
    """Variational auto-encoder module."""

    latents: int = 120

    def setup(self):
        self.encoder = Encoder(self.latents)
        self.decoder = Decoder()

    def __call__(self, x, z_rng):
        mean, logvar = self.encoder(x)
        z = reparameterize(z_rng, mean, logvar)
        reconstructed_x = self.decoder(z)
        return reconstructed_x, mean, logvar

    def generate(self, z):
        return nn.sigmoid(self.decoder(z))

def reparameterize(rng, mean, logvar, eps_factor=1.0):
    """Reparameterize.

    Args:
        rng (ArrayLike): Random number generator key.
        mean (ArrayLike): Mean of x.
        logvar (ArrayLike): Log variance of x.
        eps_factor (float, optional): Factor on epsilon. Defaults to 1.0.
            Molecular-VAE uses 0.01.

    Returns:
        ArrayLike: z.
    """
    std = jnp.exp(0.5 * logvar)
    eps = eps_factor * jax.random.normal(rng, logvar.shape)
    return mean + eps * std
