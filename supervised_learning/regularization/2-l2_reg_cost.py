#!/usr/bin/env python3
"""Calculates the cost of a neural network with L2 regularization"""

import tensorflow as tf


def l2_reg_cost(cost):
    """
    Calculates the cost of a neural network with L2 regularization

    Args:
        cost: tensor containing the cost of the network without
              L2 regularization

    Returns:
        tensor containing the cost of the network accounting for
        L2 regularization
    """
    return tf.losses.get_total_loss()
