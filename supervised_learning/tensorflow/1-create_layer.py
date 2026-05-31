#!/usr/bin/env python3
"""TensorFlow layer creation module."""

import tensorflow as tf


def create_layer(prev, n, activation):
    """
    Creates a layer for a neural network.

    Args:
        prev (tf.Tensor): The tensor output of the previous layer.
        n (int): The number of nodes in the layer to create.
        activation (function): The activation function that the layer should
                               use.

    Returns:
        tf.Tensor: The tensor output of the layer.
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.dense(
        inputs=prev,
        units=n,
        activation=activation,
        kernel_initializer=init,
        name="layer"
    )
    return layer
