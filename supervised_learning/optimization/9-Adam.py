#!/usr/bin/env python3
"""Adam optimization update"""

import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon,
                          var, grad, v, s, t):
    """
    Updates a variable using Adam optimization algorithm

    Args:
        alpha: learning rate
        beta1: first moment weight
        beta2: second moment weight
        epsilon: small constant
        var: numpy.ndarray, variable to update
        grad: numpy.ndarray, gradient
        v: first moment
        s: second moment
        t: time step

    Returns:
        updated var, updated v, updated s
    """

    # First moment update
    v_new = beta1 * v + (1 - beta1) * grad

    # Second moment update
    s_new = beta2 * s + (1 - beta2) * (grad ** 2)

    # Bias correction
    v_corrected = v_new / (1 - beta1 ** t)
    s_corrected = s_new / (1 - beta2 ** t)

    # Update variable
    var_updated = var - alpha * v_corrected / (np.sqrt(s_corrected) + epsilon)

    return var_updated, v_new, s_new
