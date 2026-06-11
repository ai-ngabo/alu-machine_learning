#!/usr/bin/env python3
"""Learning rate decay (stepwise inverse time decay)"""

import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Updates learning rate using inverse time stepwise decay

    Args:
        alpha: initial learning rate
        decay_rate: decay weight
        global_step: number of gradient descent steps
        decay_step: interval before decay

    Returns:
        updated learning rate
    """

    step = global_step // decay_step

    alpha_new = alpha / (1 + decay_rate * step)

    return alpha_new
