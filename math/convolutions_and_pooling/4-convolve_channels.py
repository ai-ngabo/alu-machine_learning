#!/usr/bin/env python3
"""Module for image convolution with channels, stride, and padding options."""

import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images with channels.

    Args:
        images: numpy.ndarray with shape (m, h, w, c) containing multiple
                images
        kernel: numpy.ndarray with shape (kh, kw, c) containing convolution
                kernel
        padding: either a tuple of (ph, pw), 'same', or 'valid'
        stride: tuple of (sh, sw) for stride in height and width

    Returns:
        numpy.ndarray containing the convolved images with shape (m, oh, ow)
    """
    # Extract dimensions
    m, h, w, c = images.shape
    kh, kw, kc = kernel.shape
    sh, sw = stride

    # Validate kernel channels match image channels
    if kc != c:
        raise ValueError("Kernel channels must match image channels")

    # Handle padding
    if padding == 'same':
        # Calculate padding needed to keep output size same as input
        ph = ((h - 1) * sh + kh - h) // 2 + 1
        pw = ((w - 1) * sw + kw - w) // 2 + 1
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        # Padding is a tuple
        ph, pw = padding

    # Calculate output dimensions
    output_h = (h + 2 * ph - kh) // sh + 1
    output_w = (w + 2 * pw - kw) // sw + 1

    # Pad images
    padded_images = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant',
        constant_values=0
    )

    # Initialize output array
    output = np.zeros((m, output_h, output_w))

    # Perform convolution with stride using only two for loops
    for i in range(output_h):
        for j in range(output_w):
            # Calculate the region in the padded image
            h_start = i * sh
            h_end = h_start + kh
            w_start = j * sw
            w_end = w_start + kw

            # Extract region and perform element-wise multiplication
            # Region shape: (m, kh, kw, c)
            region = padded_images[:, h_start:h_end, w_start:w_end, :]

            # Multiply region with kernel and sum over spatial dimensions
            # and channels to produce single value per image
            output[:, i, j] = np.sum(region * kernel, axis=(1, 2, 3))

    return output
