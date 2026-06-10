#!/usr/bin/env python3
"""Momentum optimizer update"""

import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    Updates a variable using gradient descent with momentum

    Args:
        alpha: learning rate
        beta1: momentum weight
        var: numpy.ndarray, variable to update
        grad: numpy.ndarray, gradient of var
        v: previous first moment

    Returns:
        updated variable, updated moment
    """

    v_new = beta1 * v + (1 - beta1) * grad
    var_updated = var - alpha * v_new

    return var_updated, v_new
