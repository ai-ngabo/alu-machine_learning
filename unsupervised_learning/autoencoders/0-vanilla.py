#!/usr/bin/env python3
"""Vanilla Autoencoder module."""

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a vanilla autoencoder.

    Args:
        input_dims (int): Dimensions of the model input.
        hidden_layers (list): Number of nodes for each hidden layer in the
                              encoder.
        latent_dims (int): Dimensions of the latent space representation.

    Returns:
        tuple: (encoder, decoder, auto) where:
            - encoder is the encoder model
            - decoder is the decoder model
            - auto is the full autoencoder model
    """
    # Encoder
    encoder_input = Input(shape=(input_dims,))
    x = encoder_input
    # Add hidden layers to encoder
    for units in hidden_layers:
        x = Dense(units, activation='relu')(x)
    # Latent layer
    latent = Dense(latent_dims, activation='relu')(x)
    encoder = Model(encoder_input, latent, name='encoder')
    # Decoder
    decoder_input = Input(shape=(latent_dims,))
    x = decoder_input
    # Add hidden layers to decoder (reversed)
    for units in reversed(hidden_layers):
        x = Dense(units, activation='relu')(x) 
    # Output layer with sigmoid activation
    output = Dense(input_dims, activation='sigmoid')(x)
    decoder = Model(decoder_input, output, name='decoder')  
    # Full autoencoder
    auto_input = Input(shape=(input_dims,))
    encoded = encoder(auto_input)
    decoded = decoder(encoded)
    auto = Model(auto_input, decoded, name='autoencoder')  
    # Compile the autoencoder
    auto.compile(optimizer='adam', loss='binary_crossentropy')
    return encoder, decoder, auto
