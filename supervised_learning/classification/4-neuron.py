#!/usr/bin/env python3
"""
Module 3-neuron
Defines a single neuron performing binary classification.
"""

import numpy as np


class Neuron:
    """Defines a single neuron performing binary classification"""

    def __init__(self, nx):
        """
        Initialize a Neuron.

        Parameters
        ----------
        nx : int
            Number of input features to the neuron.

        Raises
        ------
        TypeError
            If nx is not an integer.
        ValueError
            If nx is less than 1.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be positive")

        # Private attributes
        self.__W = np.random.randn(1, nx)  # weights vector
        self.__b = 0                       # bias
        self.__A = 0                       # activated output

    @property
    def W(self):
        """Getter for weights vector"""
        return self.__W

    @property
    def b(self):
        """Getter for bias"""
        return self.__b

    @property
    def A(self):
        """Getter for activated output"""
        return self.__A

    def forward_prop(self, X):
        """
        Calculates forward propagation of the neuron.

        Parameters
        ----------
        X : numpy.ndarray
            Shape (nx, m) containing input data.

        Returns
        -------
        numpy.ndarray
            Activated output of the neuron (__A).
        """
        z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-z))  # sigmoid activation
        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression.

        Parameters
        ----------
        Y : numpy.ndarray
            Shape (1, m) containing correct labels.
        A : numpy.ndarray
            Shape (1, m) containing activated output.

        Returns
        -------
        float
            Logistic regression cost.
        """
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(Y * np.log(A) +
                                 (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neuron's predictions.

        Parameters
        ----------
        X : numpy.ndarray
            Shape (nx, m) containing input data.
        Y : numpy.ndarray
            Shape (1, m) containing correct labels.

        Returns
        -------
        tuple
            (prediction, cost)
            prediction: numpy.ndarray with shape (1, m)
            cost: float representing the cost of the network
        """
        A = self.forward_prop(X)
        prediction = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)
        return prediction, cost
