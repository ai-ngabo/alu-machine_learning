#!/usr/bin/env python3
"""Gradient Descent with L2 Regularization"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the weights and biases of a neural network using
    gradient descent with L2 regularization.

    Args:
        Y: one-hot labels of shape (classes, m)
        weights: dictionary containing weights and biases
        cache: dictionary containing activations
        alpha: learning rate
        lambtha: L2 regularization parameter
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

        dW = (np.matmul(dZ, A_prev.T) / m) + \
             (lambtha / m) * W
        db = np.sum(dZ, axis=1, keepdims=True) / m

        weights["W{}".format(layer)] = W - alpha * dW
        weights["b{}".format(layer)] = \
            weights_copy["b{}".format(layer)] - alpha * db

        if layer > 1:
            A_prev_layer = cache["A{}".format(layer - 1)]
            dZ = np.matmul(W.T, dZ) * (1 - A_prev_layer ** 2)
