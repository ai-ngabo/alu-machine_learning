#!/usr/bin/env python3
"""Variational Autoencoder"""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Creates a variational autoencoder"""

    # Encoder
    inputs = keras.Input(shape=(input_dims,))
    x = inputs

    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)

    mu = keras.layers.Dense(latent_dims)(x)
    log_var = keras.layers.Dense(latent_dims)(x)

    def sampling(args):
        """Reparameterization trick"""
        mu, log_var = args
        epsilon = keras.backend.random_normal(
            shape=(keras.backend.shape(mu)[0], latent_dims)
        )
        return mu + keras.backend.exp(log_var / 2) * epsilon

    z = keras.layers.Lambda(sampling)([mu, log_var])

    encoder = keras.Model(inputs, [z, mu, log_var])

    # Decoder
    latent_inputs = keras.Input(shape=(latent_dims,))
    x = latent_inputs

    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)

    outputs = keras.layers.Dense(
        input_dims,
        activation='sigmoid'
    )(x)

    decoder = keras.Model(latent_inputs, outputs)

    # Autoencoder
    auto_outputs = decoder(z)
    auto = keras.Model(inputs, auto_outputs)

    # KL divergence loss
    kl_loss = -0.5 * keras.backend.sum(
        1 + log_var
        - keras.backend.square(mu)
        - keras.backend.exp(log_var),
        axis=-1
    )

    auto.add_loss(keras.backend.mean(kl_loss))

    auto.compile(
        optimizer='adam',
        loss='binary_crossentropy'
    )

    return encoder, decoder, auto
