#!/usr/bin/env python3
"""Adam optimizer operation"""

import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    Creates training operation using Adam optimization

    Args:
        loss: loss tensor
        alpha: learning rate
        beta1: first moment weight
        beta2: second moment weight
        epsilon: small constant

    Returns:
        training operation
    """

    optimizer = tf.train.AdamOptimizer(
        learning_rate=alpha,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon
    )

    train_op = optimizer.minimize(loss)

    return train_op
