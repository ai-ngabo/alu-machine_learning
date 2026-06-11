#!/usr/bin/env python3
"""Gradient Descent with Dropout"""

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights of a neural network using gradient descent
    with Dropout regularization.

    Args:
        Y: one-hot labels of shape (classes, m)
        weights: dictionary of weights and biases
        cache: dictionary of activations and dropout masks
        alpha: learning rate
        keep_prob: probability of keeping a node active
        L: number of layers

    Returns:
        None
    """
    m = Y.shape[1]

    weights_copy = weights.copy()

    dZ = cache["A{}".format(L)] - Y

    for layer in range(L, 0, -1):
        A_prev = cache["A{}".format(layer - 1)]
        W = weights_copy["W{}".format(layer)]

        dW = np.matmul(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m

        weights["W{}".format(layer)] = W - alpha * dW
        weights["b{}".format(layer)] = (
            weights_copy["b{}".format(layer)] - alpha * db
        )

        if layer > 1:
            dZ = np.matmul(W.T, dZ)

            D = cache["D{}".format(layer - 1)]
            dZ = dZ * D
            dZ = dZ / keep_prob

            A = cache["A{}".format(layer - 1)]
            dZ = dZ * (1 - A ** 2) 