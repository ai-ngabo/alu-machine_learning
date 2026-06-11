#!/usr/bin/env python3
"""RMSProp optimizer operation"""

import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
    Creates training operation using RMSProp optimization

    Args:
        loss: loss tensor
        alpha: learning rate
        beta2: RMSProp decay rate
        epsilon: small number to avoid division by zero

    Returns:
        training operation
    """

    optimizer = tf.train.RMSPropOptimizer(
        learning_rate=alpha,
        decay=beta2,
        epsilon=epsilon
    )

    train_op = optimizer.minimize(loss)

    return train_op
