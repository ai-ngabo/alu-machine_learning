#!/usr/bin/env python3
"""TensorFlow loss calculation module."""

import tensorflow as tf


def calculate_loss(y, y_pred):
    """
    Calculates the softmax cross-entropy loss of a prediction.

    Args:
        y (tf.Tensor): Placeholder for the labels of the input data (one-hot
                       encoded).
        y_pred (tf.Tensor): Tensor containing the network's predictions

    Returns:
        tf.Tensor: A tensor containing the loss of the prediction.
    """
    # Calculate softmax cross-entropy loss
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=y_pred)
    )
    return loss
