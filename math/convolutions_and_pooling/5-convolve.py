#!/usr/bin/env python3
"""
Module for performing convolution on multi-channel images using multiple kernels.
"""

import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images using multiple kernels.

    Parameters:
    - images (numpy.ndarray): shape (m, h, w, c)
        Multiple images with channels
    - kernels (numpy.ndarray): shape (kh, kw, c, nc)
        Kernels for the convolution
    - padding (str or tuple): 'same', 'valid', or (ph, pw)
        Padding mode or custom padding
    - stride (tuple): (sh, sw)
        Stride for height and width

    Returns:
    - numpy.ndarray: convolved images of shape (m, out_h, out_w, nc)
    """
    m, h, w, c = images.shape
    kh, kw, kc, nc = kernels.shape
    sh, sw = stride

    if kc != c:
        raise ValueError("Kernel channels must match image channels")

    # Determine padding
    if isinstance(padding, tuple):
        ph, pw = padding
    elif padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2
        pw = ((w - 1) * sw + kw - w) // 2
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        raise ValueError("padding must be 'same', 'valid', or a tuple")

    # Pad images
    padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)), mode='constant')

    # Output dimensions
    out_h = (h + 2 * ph - kh) // sh + 1
    out_w = (w + 2 * pw - kw) // sw + 1
    output = np.zeros((m, out_h, out_w, nc))

    # Perform convolution with three loops
    for i in range(out_h):
        for j in range(out_w):
            vert_start = i * sh
            vert_end = vert_start + kh
            horiz_start = j * sw
            horiz_end = horiz_start + kw

            image_slice = padded[:, vert_start:vert_end, horiz_start:horiz_end, :]
            for k in range(nc):
                output[:, i, j, k] = np.sum(image_slice * kernels[:, :, :, k],
                                            axis=(1, 2, 3))

    return output
