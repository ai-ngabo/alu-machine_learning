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
            layers (list): List representing number of nodes in each layer of the network
        
        Raises:
            TypeError: If nx is not an integer or layers is not a list or 
                      layers elements are not all positive integers
            ValueError: If nx is less than 1
        """
        # Validate nx
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        
        # Validate layers is a list
        if type(layers) is not list:
            raise TypeError("layers must be a list of positive integers")
        
        # Check for empty list
        if len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        
        # Private instance attributes
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        
        # ONE loop to validate layers and initialize weights/biases
        for i in range(self.__L):
            # Validate each layer element is a positive integer
            if type(layers[i]) is not int or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")
            
            # Weight initialization using He et al. method
            if i == 0:
                # First layer connects from input (nx) to first hidden layer (layers[0])
                self.__weights['W' + str(i + 1)] = np.random.randn(layers[i], nx) * np.sqrt(2 / nx)
            else:
                # Subsequent layers connect from previous layer (layers[i-1]) to current layer (layers[i])
                self.__weights['W' + str(i + 1)] = np.random.randn(layers[i], layers[i - 1]) * np.sqrt(2 / layers[i - 1])
            
            # Bias initialization (zeros)
            self.__weights['b' + str(i + 1)] = np.zeros((layers[i], 1))
    
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
