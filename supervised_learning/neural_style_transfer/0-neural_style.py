#!/usr/bin/env python3
"""
Neural Style Transfer (NST) class implementation.
Handles preprocessing, Gram matrix, style cost, and content cost.
"""

import numpy as np
import tensorflow as tf


class NST:
    """
    Class for Neural Style Transfer tasks.
    """

    # Public class attributes
    style_layers = ['block1_conv1', 'block2_conv1',
                    'block3_conv1', 'block4_conv1',
                    'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Initialize NST instance.

        Args:
            style_image (np.ndarray): Style reference image (h, w, 3).
            content_image (np.ndarray): Content reference image (h, w, 3).
            alpha (float): Weight for content cost.
            beta (float): Weight for style cost.

        Raises:
            TypeError: If inputs are invalid.
        """
        if (not isinstance(style_image, np.ndarray) or
                style_image.ndim != 3 or style_image.shape[2] != 3):
            raise TypeError("style_image must be a numpy.ndarray with shape (h, w, 3)")
        if (not isinstance(content_image, np.ndarray) or
                content_image.ndim != 3 or content_image.shape[2] != 3):
            raise TypeError("content_image must be a numpy.ndarray with shape (h, w, 3)")
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

        Args:
            image (np.ndarray): Image of shape (h, w, 3).

        Returns:
            tf.Tensor: Scaled image of shape (1, h_new, w_new, 3).
        """
        if (not isinstance(image, np.ndarray) or
                image.ndim != 3 or image.shape[2] != 3):
            raise TypeError("image must be a numpy.ndarray with shape (h, w, 3)")

        image = tf.convert_to_tensor(image, dtype=tf.float32)
        image = tf.expand_dims(image, axis=0)

        h, w = image.shape[1], image.shape[2]
        scale = 512 / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)

        image = tf.image.resize(image, (new_h, new_w),
                                method=tf.image.ResizeMethod.BICUBIC)
        image = tf.clip_by_value(image / 255.0, 0.0, 1.0)
        return image

    @staticmethod
    def gram_matrix(input_tensor):
        """
        Compute Gram matrix for style representation.

        Args:
            input_tensor (tf.Tensor): Feature maps.

        Returns:
            tf.Tensor: Normalized Gram matrix.
        """
        if not isinstance(input_tensor, (tf.Tensor, tf.Variable)):
            raise TypeError("input_tensor must be a tensor")

        result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
        input_shape = tf.shape(input_tensor)
        num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
        return result / num_locations

    def layer_style_cost(self, style_output, gram_target):
        """
        Compute style cost for a single layer.

        Args:
            style_output (tf.Tensor): Style features.
            gram_target (tf.Tensor): Target Gram matrix.

        Returns:
            tf.Tensor: Style cost.
        """
        if not isinstance(style_output, (tf.Tensor, tf.Variable)):
            raise TypeError("style_output must be a tensor")
        if not isinstance(gram_target, (tf.Tensor, tf.Variable)):
            raise TypeError("gram_target must be a tensor")

        gram_style = NST.gram_matrix(style_output)
        return tf.reduce_mean(tf.square(gram_style - gram_target))

    def content_cost(self, content_output, generated_output):
        """
        Compute content cost between content and generated image.

        Args:
            content_output (tf.Tensor): Content features.
            generated_output (tf.Tensor): Generated features.

        Returns:
            tf.Tensor: Content cost.
        """
        if not isinstance(content_output, (tf.Tensor, tf.Variable)):
            raise TypeError("content_output must be a tensor")
        if not isinstance(generated_output, (tf.Tensor, tf.Variable)):
            raise TypeError("generated_output must be a tensor")

        return tf.reduce_mean(tf.square(content_output - generated_output))
