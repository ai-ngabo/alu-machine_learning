#!/usr/bin/env python3
"""Variational Autoencoder"""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """creates a variational autoencoder"""

    # Encoder
    encoder_inputs = keras.Input(shape=(input_dims,))

    x = encoder_inputs
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)

    mu = keras.layers.Dense(
        latent_dims,
        activation=None
    )(x)

    log_var = keras.layers.Dense(
        latent_dims,
        activation=None
    )(x)

    def sampling(args):
        """reparameterization trick"""
        mu_, log_var_ = args

        epsilon = keras.backend.random_normal(
            shape=keras.backend.shape(mu_)
        )

        return (
            mu_
            + keras.backend.exp(log_var_ / 2)
            * epsilon
        )

    z = keras.layers.Lambda(
        sampling
    )([mu, log_var])

    encoder = keras.Model(
        encoder_inputs,
        [z, mu, log_var]
    )

    # Decoder
    decoder_inputs = keras.Input(shape=(latent_dims,))

    x = decoder_inputs
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(
            nodes,
            activation='relu'
        )(x)

    decoder_outputs = keras.layers.Dense(
        input_dims,
        activation='sigmoid'
    )(x)

    decoder = keras.Model(
        decoder_inputs,
        decoder_outputs
    )

    # Autoencoder
    encoded, mu, log_var = encoder(encoder_inputs)
    outputs = decoder(encoded)

    auto = keras.Model(
        encoder_inputs,
        outputs
    )

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
