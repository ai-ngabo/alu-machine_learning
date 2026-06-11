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
    # Retrieve the list of individual regularization losses from the graph
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

    # Sum all individual regularization tensors together
    total_reg_loss = tf.add_n(reg_losses)

    # Add the total regularization loss to the unregularized cost tensor
    return cost + total_reg_loss
