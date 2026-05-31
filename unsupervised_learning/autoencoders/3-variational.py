#!/usr/bin/env python3
"""Variational Autoencoder module."""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder.

    Args:
        input_dims (int): Dimensions of the model input.
        hidden_layers (list): Number of nodes for each hidden layer in the
                              encoder.
        latent_dims (int): Dimensions of the latent space representation.

    Returns:
        tuple: (encoder, decoder, auto) where:
            - encoder is the encoder model that outputs the latent
              representation, mean, and log variance
            - decoder is the decoder model
            - auto is the full autoencoder model
    """
    # Encoder
    encoder_input = keras.layers.Input(shape=(input_dims,))
    x = encoder_input
    # Add hidden layers to encoder
    for units in hidden_layers:
        x = keras.layers.Dense(units, activation='relu')(x)
    # Mean and log variance layers (no activation)
    mean = keras.layers.Dense(latent_dims, activation=None)(x)
    log_var = keras.layers.Dense(latent_dims, activation=None)(x)
    # Sampling function
    def sampling(args):
        """Sample from latent space using reparameterization trick."""
        mean, log_var = args
        epsilon = keras.backend.random_normal(
            shape=keras.backend.shape(mean),
            mean=0.0,
            stddev=1.0
        )
        return mean + keras.backend.exp(log_var / 2) * epsilon
    # Latent representation
    latent = keras.layers.Lambda(sampling, output_shape=(latent_dims,))([mean, log_var])
    # Encoder model outputs latent, mean, and log_var
    encoder = keras.models.Model(
        encoder_input,
        [latent, mean, log_var],
        name='encoder'
    )
    # Decoder
    decoder_input = keras.layers.Input(shape=(latent_dims,))
    x = decoder_input 
    # Add hidden layers to decoder (reversed)
    for units in reversed(hidden_layers):
        x = keras.layers.Dense(units, activation='relu')(x) 
    # Output layer with sigmoid activation
    output = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.models.Model(decoder_input, output, name='decoder')
    # Full autoencoder
    auto_input = keras.layers.Input(shape=(input_dims,))
    latent_output, mean_output, log_var_output = encoder(auto_input)
    decoded_output = decoder(latent_output)
    auto = keras.models.Model(auto_input, decoded_output, name='autoencoder')
    # Custom loss function (VAE loss = reconstruction loss + KL divergence)
    reconstruction_loss = keras.losses.binary_crossentropy(auto_input, decoded_output)
    reconstruction_loss *= input_dims
    kl_loss = 1 + log_var_output - keras.backend.square(mean_output) - keras.backend.exp(log_var_output)
    kl_loss = keras.backend.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = keras.backend.mean(reconstruction_loss + kl_loss)
    # Add loss to model and compile
    auto.add_loss(vae_loss)
    auto.compile(optimizer='adam')    
    return encoder, decoder, auto
