#!/usr/bin/env python3
import numpy as np

def convolve_grayscale_valid(images, kernel):
    """
    Performs a valid convolution on grayscale images.

    Parameters:
    - images: numpy.ndarray of shape (m, h, w)
        Multiple grayscale images
    - kernel: numpy.ndarray of shape (kh, kw)
        Kernel for the convolution

    Returns:
    - numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Output dimensions for valid convolution
    output_h = h - kh + 1
    output_w = w - kw + 1

    # Initialize output
    output = np.zeros((m, output_h, output_w))

    # Perform convolution with only two loops
    for i in range(output_h):
        for j in range(output_w):
            # Extract the slice of the image
            image_slice = images[:, i:i+kh, j:j+kw]
            # Element-wise multiply and sum over kernel
            output[:, i, j] = np.sum(image_slice * kernel, axis=(1, 2))

    return output
