#!/usr/bin/env python3
"""TensorFlow forward propagation module."""

import tensorflow as tf

create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Creates the forward propagation graph for the neural network.

    Args:
        x (tf.Tensor): The placeholder for the input data.
        layer_sizes (list): List containing the number of nodes in each layer
                            of the network.
        activations (list): List containing the activation functions for each
                            layer of the network.

    Returns:
        tf.Tensor: The prediction of the network in tensor form.
    """
    prev = x
    for i in range(len(layer_sizes)):
        prev = create_layer(prev, layer_sizes[i], activations[i])
    return prev
