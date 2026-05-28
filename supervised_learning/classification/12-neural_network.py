#!/usr/bin/env python3
"""
Defines a neural network with one hidden layer performing
binary classification.
"""
import numpy as np


class NeuralNetwork:
    """
    Represents a neural network with one hidden layer.
    """

    def __init__(self, nx, nodes):
        """
        Initializes the neural network.

        Parameters:
        - nx: number of input features
        - nodes: number of nodes in the hidden layer
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # Hidden layer parameters
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0

        # Output neuron parameters
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Getter for weights matrix of the hidden layer."""
        return self.__W1

    @property
    def b1(self):
        """Getter for bias vector of the hidden layer."""
        return self.__b1

    @property
    def A1(self):
        """Getter for activated output of the hidden layer."""
        return self.__A1

    @property
    def W2(self):
        """Getter for weights matrix of the output neuron."""
        return self.__W2

    @property
    def b2(self):
        """Getter for bias of the output neuron."""
        return self.__b2

    @property
    def A2(self):
        """Getter for activated output of the output neuron."""
        return self.__A2

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network.

        Parameters:
        - X: numpy.ndarray with shape (nx, m) containing the input data

        Returns:
        - The private attributes __A1 and __A2, respectively
        """
        # Hidden layer forward propagation
        Z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))

        # Output layer forward propagation
        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))

        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression.

        Parameters:
        - Y: numpy.ndarray with shape (1, m) containing the correct labels
        - A: numpy.ndarray with shape (1, m) containing the activated outputs

        Returns:
        - The cost
        """
        m = Y.shape[1]
        loss = -(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        cost = np.sum(loss) / m
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network's predictions.

        Parameters:
        - X: numpy.ndarray with shape (nx, m) containing the input data
        - Y: numpy.ndarray with shape (1, m) containing the correct labels

        Returns:
        - The neuron's prediction and the cost of the network, respectively
        """
        _, A2 = self.forward_prop(X)
        cost = self.cost(Y, A2)

        # Threshold activations to binary results (0 or 1)
        prediction = np.where(A2 >= 0.5, 1, 0)

        return prediction, cost
    