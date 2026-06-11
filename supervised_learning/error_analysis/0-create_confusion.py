"""Module containing the BayesianOptimization class."""
import numpy as np


class BayesianOptimization:
    """Performs Bayesian optimization on a 1D black-box function."""

    def __init__(self, f, X_init, Y_init, bounds, acq_samples, l=1,
                 sigma_f=1, reference=None, xsi=0.01, sigma_y=1e-8):
        """Initializes the Bayesian Optimization class."""
        # This setup assumes the rest of the class methods (acquisition,
        # register, etc.) are already inherited or defined in 4-bayes_opt.py
        pass

    def optimize(self, iterations=100):
        """
        Optimizes the black-box function.

        Args:
            iterations: maximum number of iterations to perform

        Returns:
            X_opt: numpy.ndarray of shape (1,) representing the optimal point
            Y_opt: numpy.ndarray of shape (1,) representing optimal value
        """
        for _ in range(iterations):
            # Call acquisition method to propose the next point
            X_next, _ = self.acquisition()

            # Check if the proposed point has already been sampled
            if np.any(np.isclose(X_next, self.gp.X)):
                break

            # Evaluate the black-box function at the new point
            Y_next = self.f(X_next)

            # Register the new sample point to update the GP model
            self.register(X_next, Y_next)

        # Locate the index of the minimum evaluation value found
        opt_index = np.argmin(self.gp.Y)
        X_opt = self.gp.X[opt_index]
        Y_opt = self.gp.Y[opt_index]

        return X_opt, Y_opt
