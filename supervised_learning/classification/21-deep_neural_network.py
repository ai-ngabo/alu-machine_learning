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

        # Loop #1
        for i in range(self.__L):
            if type(layers[i]) is not int or layers[i] < 1:
                raise TypeError("layers must be a list of positive integers")

            if i == 0:
                he_init = np.random.randn(layers[i], nx) * np.sqrt(2.0 / nx)
                self.__weights['W{}'.format(i + 1)] = he_init
            else:
                he_init = np.random.randn(layers[i], layers[i - 1])
                he_init = he_init * np.sqrt(2.0 / layers[i - 1])
                self.__weights['W{}'.format(i + 1)] = he_init

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

    def forward_prop(self, X):
        """
        Calculates forward propagation of the neural network
        """
        self.__cache['A0'] = X

        # Loop #2
        for i in range(self.__L):
            W = self.__weights['W{}'.format(i + 1)]
            b = self.__weights['b{}'.format(i + 1)]
            A_prev = self.__cache['A{}'.format(i)]

            Z = np.matmul(W, A_prev) + b
            A = 1 / (1 + np.exp(-Z))
            self.__cache['A{}'.format(i + 1)] = A

        return self.__cache['A{}'.format(self.__L)], self.__cache

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        """
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)) / m
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network's predictions
        """
        A, _ = self.forward_prop(X)
        prediction = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network
        """
        m = Y.shape[1]

        # Loop #3 - going backwards through layers
        for i in range(self.__L, 0, -1):
            A_curr = cache['A{}'.format(i)]
            A_prev = cache['A{}'.format(i - 1)]

            if i == self.__L:
                dz = A_curr - Y
            else:
                W_next = self.__weights['W{}'.format(i + 1)]
                dz_next = dz
                dz = np.matmul(W_next.T, dz_next) * (A_curr * (1 - A_curr))

            dw = np.matmul(dz, A_prev.T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m

            self.__weights['W{}'.format(i)] -= alpha * dw
            self.__weights['b{}'.format(i)] -= alpha * db
