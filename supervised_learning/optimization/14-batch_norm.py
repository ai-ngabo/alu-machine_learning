#!/usr/bin/env python3
"""Batch normalization layer (TensorFlow)"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer

    Args:
        prev: activated output of previous layer
        n: number of nodes in layer
        activation: activation function

    Returns:
        activated tensor output
    """

    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")

    dense = tf.layers.Dense(
        units=n,
        kernel_initializer=init,
        use_bias=False
    )

    Z = dense(prev)

    gamma = tf.Variable(tf.ones([n]), trainable=True)
    beta = tf.Variable(tf.zeros([n]), trainable=True)

    mean, variance = tf.nn.moments(Z, axes=[0])

    Z_norm = tf.nn.batch_normalization(
        x=Z,
        mean=mean,
        variance=variance,
        offset=beta,
        scale=gamma,
        variance_epsilon=1e-8
    )

    return activation(Z_norm)
