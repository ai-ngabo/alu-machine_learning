#!/usr/bin/env python3
"""
Module for performing same convolution on grayscale images.
"""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs a same convolution on grayscale images.

    Parameters:
    - images (numpy.ndarray): shape (m, h, w)
        Multiple grayscale images
    - kernel (numpy.ndarray): shape (kh, kw)
        Kernel for the convolution

    Returns:
    - numpy.ndarray: convolved images of shape (m, h, w)
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Calculate padding sizes
    pad_h = kh // 2
    pad_w = kw // 2

    # Pad images with zeros
    padded = np.pad(images, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
                    mode='constant')

    # Initialize output with same shape as input
    output = np.zeros((m, h, w))

    # Perform convolution with only two loops
    for i in range(h):
        for j in range(w):
            image_slice = padded[:, i:i+kh, j:j+kw]
            output[:, i, j] = np.sum(image_slice * kernel, axis=(1, 2))

    return output
