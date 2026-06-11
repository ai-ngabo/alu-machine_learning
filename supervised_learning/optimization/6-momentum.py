#!/usr/bin/env python3
"""Momentum optimizer operation"""

import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """
    Creates training operation using momentum optimization

    Args:
        loss: loss tensor
        alpha: learning rate
        beta1: momentum weight

    Returns:
        training operation
    """

    optimizer = tf.train.MomentumOptimizer(
        learning_rate=alpha,
        momentum=beta1
    )

    train_op = optimizer.minimize(loss)

    return train_op
