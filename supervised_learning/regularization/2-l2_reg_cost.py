#!/usr/bin/env python3
"""L2 Regularization Cost"""

import tensorflow as tf


def l2_reg_cost(cost):
    """
    Calculates the cost of a neural network with L2 regularization

    Args:
        cost: tensor containing the cost of the network
              without L2 regularization

    Returns:
        tensor containing the cost of the network
        accounting for L2 regularization
    """
    # Retrieve the global scalar regularization loss
    reg_losses = tf.losses.get_regularization_loss()

    # Explicitly add the scalar to the vector to return a matching array shape
    return cost + reg_losses
