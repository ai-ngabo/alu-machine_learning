#!/usr/bin/env python3
"""Variational Autoencoder"""

import tensorflow.keras as keras
from tensorflow.keras import backend as K


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder.

    Args:
        input_dims: integer, dimensions of the model input
        hidden_layers: list of integers, nodes in encoder hidden layers
        latent_dims: integer, dimensions of latent space

    Returns:
        encoder, decoder, auto
    """
    # Encoder
    encoder_input = keras.Input(shape=(input_dims,))
    x = encoder_input
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)
    mu = keras.layers.Dense(latent_dims, activation=None)(x)
    log_var = keras.layers.Dense(latent_dims, activation=None)(x)
    def sample(args):
        """Reparameterization trick"""
        mean, log_variance = args
        epsilon = K.random_normal(
            shape=(K.shape(mean)[0], latent_dims)
        )
        return mean + K.exp(log_variance / 2) * epsilon
    z = keras.layers.Lambda(sample)([mu, log_var])
    encoder = keras.Model(
        encoder_input,
        [z, mu, log_var]
    )
    # Decoder
    latent_input = keras.Input(shape=(latent_dims,))
    x = latent_input
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)
    decoder_output = keras.layers.Dense(
        input_dims,
        activation='sigmoid'
    )(x)
    decoder = keras.Model(latent_input, decoder_output)
    # Autoencoder
    auto_output = decoder(z)
    auto = keras.Model(encoder_input, auto_output)
    # KL Divergence Loss
    kl_loss = -0.5 * K.sum(
        1 + log_var - K.square(mu) - K.exp(log_var),
        axis=-1
    )
    auto.add_loss(K.mean(kl_loss))
    auto.compile(
        optimizer='adam',
        loss='binary_crossentropy'
    )
    return encoder, decoder, auto
