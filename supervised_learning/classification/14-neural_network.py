#!/usr/bin/env python3
"""
Defines a neural network with one hidden layer performing
binary classification and includes full loop training.
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
        prediction = np.where(A2 >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network.

        Parameters:
        - X: numpy.ndarray with shape (nx, m) containing input data
        - Y: numpy.ndarray with shape (1, m) containing correct labels
        - A1: numpy.ndarray containing output of the hidden layer
        - A2: numpy.ndarray containing predicted output
        - alpha: learning rate
        """
        m = Y.shape[1]

        # Output layer backpropagation derivatives
        dZ2 = A2 - Y
        dW2 = np.matmul(dZ2, A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m

        # Hidden layer backpropagation derivatives
        dZ1 = np.matmul(self.__W2.T, dZ2) * (A1 * (1 - A1))
        dW1 = np.matmul(dZ1, X.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m

        # Update weights and biases
        self.__W2 -= alpha * dW2
        self.__b2 -= alpha * db2
        self.__W1 -= alpha * dW1
        self.__b1 -= alpha * db1

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Trains the neural network using forward prop and Gradient D

        Parameters:
        - X: numpy.ndarray with shape (nx, m) containing input data
        - Y: numpy.ndarray with shape (1, m) containing correct labels
        - iterations: number of iterations to train over
        - alpha: learning rate

        Returns:
        - The evaluation of the training data after training completes
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        for _ in range(iterations):
            A1, A2 = self.forward_prop(X)
            self.gradient_descent(X, Y, A1, A2, alpha)

        return self.evaluate(X, Y)
