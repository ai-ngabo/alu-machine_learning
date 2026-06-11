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
    # Sum individual graph keys sequentially to preserve baseline cost rank
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    return cost + tf.add_n(reg_losses)
