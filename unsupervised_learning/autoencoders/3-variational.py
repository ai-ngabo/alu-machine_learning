#!/usr/bin/env python3
"""Variational Autoencoder"""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder

    Args:
        input_dims (int): dimensions of the model input
        hidden_layers (list): number of nodes for each hidden layer
        latent_dims (int): dimensions of the latent space

    Returns:
        encoder, decoder, auto
    """

    # ====================
    # Encoder
    # ====================
    inputs = keras.Input(shape=(input_dims,))

    x = inputs
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)

    mu = keras.layers.Dense(latent_dims, activation=None)(x)
    log_var = keras.layers.Dense(latent_dims, activation=None)(x)

    def sampling(args):
        """Reparameterization trick"""
        mu_, log_var_ = args

        epsilon = keras.backend.random_normal(
            shape=(keras.backend.shape(mu_)[0], latent_dims)
        )

        return mu_ + keras.backend.exp(log_var_ / 2) * epsilon

    z = keras.layers.Lambda(sampling)([mu, log_var])

    encoder = keras.Model(
        inputs=inputs,
        outputs=[z, mu, log_var]
    )

    # ====================
    # Decoder
    # ====================
    latent_inputs = keras.Input(shape=(latent_dims,))

    x = latent_inputs
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)

    outputs = keras.layers.Dense(
        input_dims,
        activation='sigmoid'
    )(x)

    decoder = keras.Model(
        inputs=latent_inputs,
        outputs=outputs
    )

    # ====================
    # Autoencoder
    # ====================
    z, mu, log_var = encoder(inputs)
    reconstructed = decoder(z)

    auto = keras.Model(
        inputs=inputs,
        outputs=reconstructed
    )

    # Reconstruction loss
    reconstruction_loss = keras.losses.binary_crossentropy(
        inputs,
        reconstructed
    )
    reconstruction_loss = keras.backend.sum(
        reconstruction_loss,
        axis=-1
    )

    # KL divergence loss
    kl_loss = -0.5 * keras.backend.sum(
        1 + log_var
        - keras.backend.square(mu)
        - keras.backend.exp(log_var),
        axis=-1
    )

    # Total VAE loss
    vae_loss = keras.backend.mean(
        reconstruction_loss + kl_loss
    )

    auto.add_loss(vae_loss)
    auto.compile(optimizer='adam')

    return encoder, decoder, auto
