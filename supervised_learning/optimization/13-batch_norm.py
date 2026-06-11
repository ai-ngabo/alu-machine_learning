#!/usr/bin/env python3
"""Batch normalization"""

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalizes Z using batch normalization

    Args:
        Z: numpy.ndarray of shape (m, n)
        gamma: scale parameter (1, n)
        beta: shift parameter (1, n)
        epsilon: small constant for numerical stability

    Returns:
        normalized and scaled Z
    """

    mean = np.mean(Z, axis=0)
    variance = np.var(Z, axis=0)

    Z_norm = (Z - mean) / np.sqrt(variance + epsilon)

    return gamma * Z_norm + beta
