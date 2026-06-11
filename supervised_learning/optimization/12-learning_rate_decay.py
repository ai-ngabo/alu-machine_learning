#!/usr/bin/env python3
"""Learning rate decay operation (TensorFlow)"""

import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Creates inverse time decay learning rate operation

    Args:
        alpha: initial learning rate
        decay_rate: decay weight
        global_step: training step counter
        decay_step: step interval for decay

    Returns:
        learning rate tensor
    """

    learning_rate = tf.train.inverse_time_decay(
        learning_rate=alpha,
        global_step=global_step,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True
    )

    return learning_rate
