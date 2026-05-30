#!/usr/bin/env python3
"""
Defines a deep neural network performing binary classification.
"""
import numpy as np


class DeepNeuralNetwork:
    """
    Represents a deep neural network.
    """

    def __init__(self, nx, layers):
        """
        Initializes the deep neural network.

        Parameters:
            nx (int): The number of input features.
            layers (list): The number of nodes in each layer.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.__L):
            if not isinstance(layers[i], int) or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")

            n_in = nx if i == 0 else layers[i - 1]
            n_out = layers[i]

            w_key = "W{}".format(i + 1)
            b_key = "b{}".format(i + 1)

            self.__weights[w_key] = (
                np.random.randn(n_out, n_in) * np.sqrt(2.0 / n_in)
            )
            self.__weights[b_key] = np.zeros((n_out, 1))

    @property
    def L(self):
        """Getter for the number of layers."""
        return self.__L

    @property
    def cache(self):
        """Getter for the intermediary values cache."""
        return self.__cache

    @property
    def weights(self):
        """Getter for the weights and biases dictionary."""
        return self.__weights

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network.

        Parameters:
            X (numpy.ndarray): Input data with shape (nx, m).

        Returns:
            The output of the neural network and the cache dictionary.
        """
        self.__cache["A0"] = X

        for i in range(self.__L):
            w_key = "W{}".format(i + 1)
            b_key = "b{}".format(i + 1)
            a_prev_key = "A{}".format(i)
            a_curr_key = "A{}".format(i + 1)

            W = self.__weights[w_key]
            b = self.__weights[b_key]
            A_prev = self.__cache[a_prev_key]

            Z = np.dot(W, A_prev) + b
            self.__cache[a_curr_key] = 1.0 / (1.0 + np.exp(-Z))

        return self.__cache["A{}".format(self.__L)], self.__cache

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression.

        Parameters:
            Y (numpy.ndarray): Correct labels with shape (1, m).
            A (numpy.ndarray): Activated output with shape (1, m).

        Returns:
            The cost value.
        """
        m = Y.shape[1]

        # Logistic regression cross-entropy cost formula:
        # loss = -(Y * log(A) + (1 - Y) * log(1.0000001 - A))
        loss = Y * np.log(A) + (1.0 - Y) * np.log(1.0000001 - A)
        cost = -1.0 / m * np.sum(loss)

        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network's predictions.

        Parameters:
            X (numpy.ndarray): Input data with shape (nx, m).
            Y (numpy.ndarray): Correct labels with shape (1, m).

        Returns:
            prediction (numpy.ndarray): Evaluated labels (1 or 0).
            cost (float): Cost value of the network.
        """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)

        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network.

        Parameters:
            Y (numpy.ndarray): Correct labels with shape (1, m).
            cache (dict): Dictionary containing intermediary network values.
            alpha (float): Learning rate.
        """
        m = Y.shape[1]
        dZ = None

        for i in range(self.__L, 0, -1):
            w_key = "W{}".format(i)
            b_key = "b{}".format(i)
            a_curr = cache["A{}".format(i)]
            a_prev = cache["A{}".format(i - 1)]

            if i == self.__L:
                dZ = a_curr - Y
            else:
                w_next = self.__weights["W{}".format(i + 1)]
                dZ = np.dot(w_next.T, dZ) * (a_curr * (1.0 - a_curr))

            dW = np.dot(dZ, a_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m

            self.__weights[w_key] = self.__weights[w_key] - (alpha * dW)
            self.__weights[b_key] = self.__weights[b_key] - (alpha * db)

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """
        Trains the deep neural network by updating weights and cache.

        Parameters:
            X (numpy.ndarray): Input data with shape (nx, m).
            Y (numpy.ndarray): Correct labels with shape (1, m).
            iterations (int): Total number of optimization passes.
            alpha (float): Learning rate parameter.
            verbose (bool): Whether to print cost information.
            graph (bool): Whether to graph cost over time.
            step (int): Training step window for output reporting.

        Returns:
            The evaluation results of the training data after optimization.
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")

        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be a positive integer and <= iterations")

        for itr in range(iterations):
            A, cache = self.forward_prop(X)
            if verbose and itr % step == 0:
                print("Cost after {} iterations: {}".format(
                    itr, self.cost(Y, A)))
            self.gradient_descent(Y, cache, alpha)

        A, _ = self.forward_prop(X)
        if verbose:
            print("Cost after {} iterations: {}".format(
                iterations, self.cost(Y, A)))

        return self.evaluate(X, Y)

    def save(self, filename):
        """
        Saves the instance object to a file in pickle format.

        Parameters:
            filename (str): Target save file name path string.
        """
        import pickle

        if not filename.endswith(".pkl"):
            filename += ".pkl"

        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """
        Loads a pickled DeepNeuralNetwork object from disk.

        Parameters:
            filename (str): File destination path location.

        Returns:
            DeepNeuralNetwork: Restored network object, or None 
        """
        import os
        import pickle

        if not os.path.exists(filename):
            return None

        with open(filename, "rb") as f:
            return pickle.load(f)
        