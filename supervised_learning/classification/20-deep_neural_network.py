#!/usr/bin/env python3
"""Deep neural network module for binary classification"""

import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network performing binary classification"""
    
    def __init__(self, nx, layers):
        """Constructor for DeepNeuralNetwork class"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list:
            raise TypeError("layers must be a list of positive integers")
        for layer in layers:
            if type(layer) is not int or layer <= 0:
                raise TypeError("layers must be a list of positive integers")
        
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        
        for i in range(1, self.__L + 1):
            if i == 1:
                self.__weights['W' + str(i)] = (
                    np.random.randn(layers[i-1], nx) * np.sqrt(2 / nx)
                )
            else:
                self.__weights['W' + str(i)] = (
                    np.random.randn(layers[i-1], layers[i-2]) *
                    np.sqrt(2 / layers[i-2])
                )
            self.__weights['b' + str(i)] = np.zeros((layers[i-1], 1))
    
    @property
    def L(self):
        """Getter for private attribute L"""
        return self.__L
    
    @property
    def cache(self):
        """Getter for private attribute cache"""
        return self.__cache
    
    @property
    def weights(self):
        """Getter for private attribute weights"""
        return self.__weights
    
    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network"""
        self.__cache['A0'] = X
        
        for i in range(1, self.__L + 1):
            W = self.__weights['W' + str(i)]
            b = self.__weights['b' + str(i)]
            A_prev = self.__cache['A' + str(i - 1)]
            Z = np.matmul(W, A_prev) + b
            A = 1 / (1 + np.exp(-Z))
            self.__cache['A' + str(i)] = A
        
        return self.__cache['A' + str(self.__L)], self.__cache
    
    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        m = Y.shape[1]
        cost = -1/m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost
    
    def evaluate(self, X, Y):
        """Evaluates the neural network's predictions"""
        A, _ = self.forward_prop(X)
        predictions = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)
        return predictions, cost
