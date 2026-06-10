#!/usr/bin/env python3
"""Variational Autoencoder"""


def autoencoder(input_dims, hidden_layers, latent_dims):
    """creates a variational autoencoder"""

    # ========= Encoder =========
    encoder_inputs = keras.Input(shape=(input_dims,))

    x = encoder_inputs
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)

    mu = keras.layers.Dense(latent_dims)(x)
    log_var = keras.layers.Dense(latent_dims)(x)

    def sampling(args):
        """reparameterization trick"""
        mu_, log_var_ = args
        epsilon = K.random_normal(
            shape=(K.shape(mu_)[0], latent_dims)
        )
        return mu_ + K.exp(log_var_ / 2) * epsilon

    z = keras.layers.Lambda(
        sampling,
        output_shape=(latent_dims,)
    )([mu, log_var])

    encoder = keras.Model(
        encoder_inputs,
        [z, mu, log_var]
    )

    # ========= Decoder =========
    decoder_inputs = keras.Input(shape=(latent_dims,))

    x = decoder_inputs
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)

    decoder_outputs = keras.layers.Dense(
        input_dims,
        activation='sigmoid'
    )(x)

    decoder = keras.Model(
        decoder_inputs,
        decoder_outputs
    )

    # ========= Autoencoder =========
    auto_inputs = encoder_inputs
    z, mu, log_var = encoder(auto_inputs)
    auto_outputs = decoder(z)

    auto = keras.Model(auto_inputs, auto_outputs)

    # KL divergence loss
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
