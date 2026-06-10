#!/usr/bin/env python3
"""Variational Autoencoder"""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder

    Args:
        input_dims: dimensions of the model input
        hidden_layers: list containing the number of nodes for each
                       hidden layer in the encoder
        latent_dims: dimensions of the latent space representation

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
        mu_, log_var_ = args
        epsilon = keras.backend.random_normal(
            shape=(keras.backend.shape(mu_)[0], latent_dims)
        )
        return mu_ + keras.backend.exp(log_var_ / 2) * epsilon

    z = keras.layers.Lambda(
        sample,
        output_shape=(latent_dims,)
    )([mu, log_var])

    encoder = keras.Model(
        inputs=encoder_input,
        outputs=[z, mu, log_var]
    )

    # Decoder
    decoder_input = keras.Input(shape=(latent_dims,))

    x = decoder_input
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)

    decoder_output = keras.layers.Dense(
        input_dims,
        activation='sigmoid'
    )(x)

    decoder = keras.Model(
        inputs=decoder_input,
        outputs=decoder_output
    )

    # Autoencoder
    encoded, mu, log_var = encoder(encoder_input)
    reconstructed = decoder(encoded)

    auto = keras.Model(
        inputs=encoder_input,
        outputs=reconstructed
    )

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
