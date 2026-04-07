#!/usr/bin/env python3
"""
Module for performing convolution on grayscale images with custom padding.
"""

import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Performs a convolution on grayscale images with custom padding.

    Parameters:
    - images (numpy.ndarray): shape (m, h, w)
        Multiple grayscale images
    - kernel (numpy.ndarray): shape (kh, kw)
        Kernel for the convolution
    - padding (tuple): (ph, pw)
        Custom padding for height and width

    Returns:
    - numpy.ndarray: convolved images of shape (m, h + 2*ph - kh + 1, w + 2*pw - kw + 1)
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    # Pad images with zeros
    padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), mode='constant')

    # Output dimensions
    output_h = h + 2 * ph - kh + 1
    output_w = w + 2 * pw - kw + 1
    output = np.zeros((m, output_h, output_w))

    # Perform convolution with only two loops
    for i in range(output_h):
        for j in range(output_w):
            image_slice = padded[:, i:i+kh, j:j+kw]
            output[:, i, j] = np.sum(image_slice * kernel, axis=(1, 2))

    return output
