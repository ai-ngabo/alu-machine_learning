#!/usr/bin/env python3
"""
Module for performing pooling on multi-channel images.
"""

import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Performs pooling on images.

    Parameters:
    - images (numpy.ndarray): shape (m, h, w, c)
        Multiple images with channels
    - kernel_shape (tuple): (kh, kw)
        Kernel shape for pooling
    - stride (tuple): (sh, sw)
        Stride for height and width
    - mode (str): 'max' or 'avg'
        Type of pooling

    Returns:
    - numpy.ndarray: pooled images of shape (m, out_h, out_w, c)
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Output dimensions
    out_h = (h - kh) // sh + 1
    out_w = (w - kw) // sw + 1
    output = np.zeros((m, out_h, out_w, c))

    # Perform pooling with only two loops
    for i in range(out_h):
        for j in range(out_w):
            vert_start = i * sh
            vert_end = vert_start + kh
            horiz_start = j * sw
            horiz_end = horiz_start + kw

            image_slice = images[:, vert_start:vert_end, horiz_start:horiz_end, :]
            if mode == 'max':
                output[:, i, j, :] = np.max(image_slice, axis=(1, 2))
            elif mode == 'avg':
                output[:, i, j, :] = np.mean(image_slice, axis=(1, 2))
            else:
                raise ValueError("mode must be 'max' or 'avg'")

    return output
