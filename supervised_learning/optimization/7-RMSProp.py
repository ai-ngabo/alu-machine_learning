#!/usr/bin/env python3
"""RMSProp optimization update"""

import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Updates a variable using RMSProp optimization

    Args:
        alpha: learning rate
        beta2: RMSProp weight
        epsilon: small constant to avoid division by zero
        var: numpy.ndarray, variable to update
        grad: numpy.ndarray, gradient of var
        s: previous second moment

    Returns:
        updated variable, updated second moment
    """

    s_new = beta2 * s + (1 - beta2) * (grad ** 2)

    var_updated = var - alpha * grad / (np.sqrt(s_new) + epsilon)

    return var_updated, s_new
