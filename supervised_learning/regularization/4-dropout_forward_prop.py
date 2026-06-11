#!/usr/bin/env python3
"""Forward Propagation with Dropout"""

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conducts forward propagation using Dropout.

    Args:
        X: input data of shape (nx, m)
        weights: dictionary of weights and biases
        L: number of layers
        keep_prob: probability of keeping a node active

    Returns:
        Dictionary containing activations and dropout masks
    """
    cache = {"A0": X}

    for layer in range(1, L + 1):
        W = weights["W{}".format(layer)]
        b = weights["b{}".format(layer)]
        A_prev = cache["A{}".format(layer - 1)]

        Z = np.matmul(W, A_prev) + b

        if layer == L:
            exp_Z = np.exp(Z)
            cache["A{}".format(layer)] = (
                exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
            )
        else:
            A = np.tanh(Z)

            D = np.random.rand(*A.shape) < keep_prob
            A = A * D
            A = A / keep_prob

            cache["A{}".format(layer)] = A
            cache["D{}".format(layer)] = D

    return cache
