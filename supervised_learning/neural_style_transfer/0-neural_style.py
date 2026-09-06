#!/usr/bin/env python3
"""
Neural Style Transfer (NST) class implementation.
"""

import numpy as np
import tensorflow as tf


class NST:
    """
    Class for Neural Style Transfer tasks.
    """

    style_layers = ['block1_conv1', 'block2_conv1',
                    'block3_conv1', 'block4_conv1',
                    'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image,
                 alpha=1e4, beta=1):
        """
        Initialize NST instance.
        """
        if (not isinstance(style_image, np.ndarray) or
                style_image.ndim != 3 or
                style_image.shape[2] != 3):
            raise TypeError(
                "style_image must be a numpy.ndarray "
                "with shape (h, w, 3)"
            )
        if (not isinstance(content_image, np.ndarray) or
                content_image.ndim != 3 or
                content_image.shape[2] != 3):
            raise TypeError(
                "content_image must be a numpy.ndarray "
                "with shape (h, w, 3)"
            )
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        tf.compat.v1.enable_eager_execution()

        self.style_image = NST.scale_image(style_image)
        self.content_image = NST.scale_image(content_image)
        self.alpha = float(alpha)
        self.beta = float(beta)

    @staticmethod
    def scale_image(image):
        """
        Rescale image to max side 512 px, values in [0, 1].
        """
        if (not isinstance(image, np.ndarray) or
                image.ndim != 3 or
                image.shape[2] != 3):
            raise TypeError(
                "image must be a numpy.ndarray "
                "with shape (h, w, 3)"
            )

        image = tf.convert_to_tensor(image, dtype=tf.float32)
        image = tf.expand_dims(image, axis=0)

        h, w = image.shape[1], image.shape[2]
        if h > w:
            new_h, new_w = 512, int((w * 512) / h)
        else:
            new_h, new_w = int((h * 512) / w), 512

        image = tf.image.resize(
            image, (new_h, new_w),
            method=tf.image.ResizeMethod.BICUBIC
        )
        image = tf.clip_by_value(image / 255.0, 0.0, 1.0)
        return image
