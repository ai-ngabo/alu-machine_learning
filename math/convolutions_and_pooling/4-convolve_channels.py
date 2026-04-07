#!/usr/bin/env python3
"""
Module for performing convolution on multi-channel images with stride and padding.
"""

import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images with channels.

    Parameters:
    - images (numpy.ndarray): shape (m, h, w, c)
        Multiple images with channels
    - kernel (numpy.ndarray): shape (kh, kw, c)
        Kernel for the convolution
    - padding (str or tuple): 'same', 'valid', or (ph, pw)
        Padding mode or custom padding
    - stride (tuple): (sh, sw)
        Stride for height and width

    Returns:
    - numpy.ndarray: convolved images of shape (m, out_h, out_w)
    """
    m, h, w, c = images.shape
    kh, kw, kc = kernel.shape
    sh, sw = stride

    if kc != c:
        raise ValueError(
            "Kernel channels must match image channels"
        )

    # Determine padding
    if isinstance(padding, tuple):
        ph, pw = padding
    elif padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2
        pw = ((w - 1) * sw + kw - w) // 2
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        raise ValueError(
            "padding must be 'same', 'valid', or a tuple"
        )

    # Pad images
    padded = np.pad(
        images, ((0, 0), (ph, ph), (pw, pw), (0, 0)), mode='constant'
    )

    # Output dimensions
    out_h = (h + 2 * ph - kh) // sh + 1
    out_w = (w + 2 * pw - kw) // sw + 1
    output = np.zeros((m, out_h, out_w))

    # Perform convolution with only two loops
    for i in range(out_h):
        for j in range(out_w):
            vert_start = i * sh
            vert_end = vert_start + kh
            horiz_start = j * sw
            horiz_end = horiz_start + kw

            image_slice = padded[:, vert_start:vert_end,
                                 horiz_start:horiz_end, :]
            output[:, i, j] = np.sum(image_slice * kernel, axis=(1, 2, 3))

    return output
