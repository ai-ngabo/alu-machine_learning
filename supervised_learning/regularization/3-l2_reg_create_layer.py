#!/usr/bin/env python3
"""Create a layer with L2 regularization"""

import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a tensorflow layer that includes L2 regularization

    Args:
        prev: tensor containing the output of the previous layer
        n: number of nodes in the layer
        activation: activation function to use
        lambtha: L2 regularization parameter

    Returns:
        The output of the layer
    """
    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG"
    )

    layer = tf.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer,
        kernel_regularizer=tf.contrib.layers.l2_regularizer(lambtha)
    )

    return layer(prev)
