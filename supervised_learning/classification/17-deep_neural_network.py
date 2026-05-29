#!/usr/bin/env python3
"""
Deep Neural Network class for binary classification
"""
import numpy as np


class DeepNeuralNetwork:
    """
    Defines a deep neural network for binary classification
    """
    
    def __init__(self, nx, layers):
        """
        Class constructor for DeepNeuralNetwork
        
        Args:
            nx (int): Number of input features
            layers (list): List representing number of nodes in each layer
        
        Raises:
            TypeError: If nx is not an integer or layers is not a list or
                      layers elements are not all positive integers
            ValueError: If nx is less than 1
        """
        # Validate nx
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        
        # Validate layers
        if not isinstance(layers, list):
            raise TypeError("layers must be a list of positive integers")
        if len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(layer, int) and layer > 0 for layer in layers):
            raise TypeError("layers must be a list of positive integers")
        
        # Private instance attributes
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        
        # Initialize weights and biases using He et al. method
        for i in range(self.__L):
            if i == 0:
                # First layer: weights from input (nx) to first hidden layer
                weight_key = f"W{i + 1}"
                bias_key = f"b{i + 1}"
                
                # He et al. initialization with sqrt(2/n)
                self.__weights[weight_key] = np.random.randn(layers[i], nx) * np.sqrt(2 / nx)
                self.__weights[bias_key] = np.zeros((layers[i], 1))
            else:
                # Subsequent layers: weights from previous layer to current layer
                weight_key = f"W{i + 1}"
                bias_key = f"b{i + 1}"
                
                # He et al. initialization with sqrt(2/previous_layer_size)
                self.__weights[weight_key] = np.random.randn(layers[i], layers[i - 1]) * np.sqrt(2 / layers[i - 1])
                self.__weights[bias_key] = np.zeros((layers[i], 1))
    
    @property
    def L(self):
        """
        Getter for number of layers
        """
        return self.__L
    
    @property
    def cache(self):
        """
        Getter for cache dictionary
        """
        return self.__cache
    
    @property
    def weights(self):
        """
        Getter for weights dictionary
        """
        return self.__weights
