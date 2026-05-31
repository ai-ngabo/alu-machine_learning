#!/usr/bin/env python3
"""TensorFlow training operation creation module."""

import tensorflow as tf


def create_train_op(loss, alpha):
    """
    Creates the training operation for the network.

    Args:
        loss (tf.Tensor): The loss of the network's prediction.
        alpha (float): The learning rate.

    Returns:
        tf.Operation: An operation that trains the network using gradient
                      descent.
    """
    # Create a gradient descent optimizer with the given learning rate
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    
    # Minimize the loss to create the training operation
    train_op = optimizer.minimize(loss)
    
    return train_op
