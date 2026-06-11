#!/usr/bin/env python3
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conducts forward propagation using Dropout regularization.

    Parameters:
    - X: numpy.ndarray of shape (nx, m) containing the input data
    - weights: dictionary of the weights and biases of the neural network
    - L: number of layers in the network
    - keep_prob: probability that a node will be kept

    Returns:
    - cache: dictionary containing the outputs of each layer and the
             dropout mask used on each layer
    """
    cache = {}
    cache['A0'] = X

    for i in range(1, L + 1):
        W = weights[f'W{i}']
        b = weights[f'b{i}']
        A_prev = cache[f'A{i-1}']

        # Compute linear step
        Z = np.dot(W, A_prev) + b

        if i == L:
            # Softmax activation for the last layer
            exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            cache[f'A{i}'] = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
        else:
            # Tanh activation for hidden layers
            A = np.tanh(Z)
            
            # Generate dropout mask using a uniform distribution
            D = np.random.rand(A.shape[0], A.shape[1])
            D = (D < keep_prob).astype(int)
            
            # Inverted dropout scaling
            A = (A * D) / keep_prob
            
            cache[f'D{i}'] = D
            cache[f'A{i}'] = A

    return cache
