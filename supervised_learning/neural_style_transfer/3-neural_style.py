#!/usr/bin/env python3
"""Defines class NST that performs tasks for neural style transfer."""

import numpy as np
import tensorflow as tf


class NST:
    """Performs tasks for Neural Style Transfer."""

    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """Initialize the NST class.

        Args:
            style_image: Image used as a style reference.
            content_image: Image used as a content reference.
            alpha: Weight for content cost.
            beta: Weight for style cost.
        """
        if type(style_image) is not np.ndarray or \
                len(style_image.shape) != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if type(content_image) is not np.ndarray or \
                len(content_image.shape) != 3:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        style_h, style_w, style_c = style_image.shape
        content_h, content_w, content_c = content_image.shape

        if style_h <= 0 or style_w <= 0 or style_c != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if content_h <= 0 or content_w <= 0 or content_c != 3:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if (type(alpha) is not float and type(alpha) is not int) \
                or alpha < 0:
            raise TypeError("alpha must be a non-negative number")

        if (type(beta) is not float and type(beta) is not int) \
                or beta < 0:
            raise TypeError("beta must be a non-negative number")

        tf.enable_eager_execution()

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def scale_image(image):
        """Rescale an image to a maximum side of 512 pixels.

        Args:
            image: Image to be rescaled.

        Returns:
            The scaled image.
        """
        if type(image) is not np.ndarray or len(image.shape) != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)"
            )

        h, w, c = image.shape

        if h <= 0 or w <= 0 or c != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if h > w:
            h_new = 512
            w_new = int(w * (512 / h))
        else:
            w_new = 512
            h_new = int(h * (512 / w))

        resized = tf.image.resize_bicubic(
            np.expand_dims(image, axis=0),
            size=(h_new, w_new)
        )

        rescaled = resized / 255
        rescaled = tf.clip_by_value(rescaled, 0, 1)

        return rescaled
