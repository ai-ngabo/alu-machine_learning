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

    def sample(args):
        mu_, log_var_ = args

        epsilon = keras.backend.random_normal(
            shape=keras.backend.shape(mu_)
        )

        return (
            mu_
            + keras.backend.exp(log_var_ / 2)
            * epsilon
        )

    z = keras.layers.Lambda(sample)([mu, log_var])

    encoder = keras.Model(
        inputs,
        [z, mu, log_var]
    )

    # Decoder
    latent_inputs = keras.Input(shape=(latent_dims,))

    x = latent_inputs
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)

    outputs = keras.layers.Dense(
        input_dims,
        activation='sigmoid'
    )(x)

    decoder = keras.Model(
        latent_inputs,
        outputs
    )

    # Autoencoder
    z, mu, log_var = encoder(inputs)
    reconstructed = decoder(z)

    auto = keras.Model(
        inputs,
        reconstructed
    )

    # KL divergence
    kl_loss = keras.backend.exp(log_var)
    kl_loss += keras.backend.square(mu)
    kl_loss -= 1
    kl_loss -= log_var
    kl_loss = keras.backend.sum(kl_loss, axis=-1)
    kl_loss *= 0.5

    auto.add_loss(keras.backend.mean(kl_loss))

    auto.compile(
        optimizer='adam',
        loss='binary_crossentropy'
    )

    return encoder, decoder, auto
