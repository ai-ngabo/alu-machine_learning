#!/usr/bin/env python3
"""Convolutional Autoencoder module."""

import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Creates a convolutional autoencoder.

    Args:
        input_dims (tuple): Dimensions of the model input (height, width,
                            channels).
        filters (list): Number of filters for each convolutional layer in
                        the encoder.
        latent_dims (tuple): Dimensions of the latent space representation.

    Returns:
        tuple: (encoder, decoder, auto) where:
            - encoder is the encoder model
            - decoder is the decoder model
            - auto is the full autoencoder model
    """
    # Encoder
    encoder_input = keras.layers.Input(shape=input_dims)
    x = encoder_input
    # Add convolutional layers with max pooling
    for num_filters in filters:
        x = keras.layers.Conv2D(
            num_filters,
            kernel_size=(3, 3),
            padding='same',
            activation='relu'
        )(x)
        x = keras.layers.MaxPooling2D(
            pool_size=(2, 2),
            padding='same'
        )(x)
    # Latent space
    latent = keras.layers.Conv2D(
        latent_dims[2],
        kernel_size=(3, 3),
        padding='same',
        activation='relu'
    )(x)
    encoder = keras.models.Model(encoder_input, latent, name='encoder')
    # Decoder
    decoder_input = keras.layers.Input(shape=latent_dims)
    x = decoder_input
    # Add convolutional layers with upsampling for all but last two
    for i, num_filters in enumerate(reversed(filters)):
        if i < len(filters) - 1:
            x = keras.layers.Conv2D(
                num_filters,
                kernel_size=(3, 3),
                padding='same',
                activation='relu'
            )(x)
            x = keras.layers.UpSampling2D(size=(2, 2))(x)
    # Second to last convolution with valid padding
    x = keras.layers.Conv2D(
        filters[0],
        kernel_size=(3, 3),
        padding='valid',
        activation='relu'
    )(x)
    x = keras.layers.UpSampling2D(size=(2, 2))(x)
    # Last convolution with sigmoid activation
    output = keras.layers.Conv2D(
        input_dims[2],
        kernel_size=(3, 3),
        padding='same',
        activation='sigmoid'
    )(x)
    decoder = keras.models.Model(decoder_input, output, name='decoder')
    # Full autoencoder
    auto_input = keras.layers.Input(shape=input_dims)
    encoded = encoder(auto_input)
    decoded = decoder(encoded)
    auto = keras.models.Model(auto_input, decoded, name='autoencoder')
    # Compile the autoencoder
    auto.compile(optimizer='adam', loss='binary_crossentropy')
    return encoder, decoder, auto
