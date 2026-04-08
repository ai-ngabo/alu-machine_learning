#!/usr/bin/env python3
"""Module for pooling operations on images."""

import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Performs pooling on images.

    Args:
        images: numpy.ndarray with shape (m, h, w, c) containing multiple
                images
        kernel_shape: tuple of (kh, kw) containing kernel shape for pooling
        stride: tuple of (sh, sw) for stride in height and width
        mode: indicates type of pooling, either 'max' or 'avg'

    Returns:
        numpy.ndarray containing the pooled images with shape
        (m, oh, ow, c)
    """
    # Extract dimensions
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Calculate output dimensions
    output_h = (h - kh) // sh + 1
    output_w = (w - kw) // sw + 1

    # Initialize output array
    output = np.zeros((m, output_h, output_w, c))

    # Perform pooling using exactly two for loops
    for i in range(output_h):
        for j in range(output_w):
            # Calculate the region in the image
            h_start = i * sh
            h_end = h_start + kh
            w_start = j * sw
            w_end = w_start + kw

            # Extract region: shape (m, kh, kw, c)
            region = images[:, h_start:h_end, w_start:w_end, :]

            if mode == 'max':
                # Max pooling: take maximum value over height and width
                output[:, i, j, :] = np.max(region, axis=(1, 2))
            elif mode == 'avg':
                # Average pooling: take mean over height and width
                output[:, i, j, :] = np.mean(region, axis=(1, 2))
            else:
                raise ValueError("Mode must be either 'max' or 'avg'")

    return output
