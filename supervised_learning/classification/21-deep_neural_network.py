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
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        
        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        
        for i in range(self.__L):
            if type(layers[i]) is not int or layers[i] < 1:
                raise TypeError("layers must be a list of positive integers")
            
            # He et al. initialization
            if i == 0:
                self.__weights['W{}'.format(i + 1)] = np.random.randn(layers[i], nx) * np.sqrt(2.0 / nx)
            else:
                self.__weights['W{}'.format(i + 1)] = np.random.randn(layers[i], layers[i - 1]) * np.sqrt(2.0 / layers[i - 1])
            
            self.__weights['b{}'.format(i + 1)] = np.zeros((layers[i], 1))
    
    @property
    def L(self):
        """Get the number of layers"""
        return self.__L
    
    @property
    def cache(self):
        """Get the cache dictionary"""
        return self.__cache
    
    @property
    def weights(self):
        """Get the weights dictionary"""
        return self.__weights
