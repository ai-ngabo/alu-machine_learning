#!/usr/bin/env python3
""" baum_welch algorithm"""

import numpy as np


def baum_welch(Observations, Transition, Emission, Initial,
               iterations=1000):
    """Performs the Baum-Welch algorithm for a hidden Markov model."""
    if (not isinstance(Observations, np.ndarray) or
            not isinstance(Transition, np.ndarray) or
            not isinstance(Emission, np.ndarray) or
            not isinstance(Initial, np.ndarray)):
        return None, None

    if (Observations.ndim != 1 or Transition.ndim != 2 or
            Emission.ndim != 2 or Initial.ndim not in (1, 2)):
        return None, None

    M = Transition.shape[0]
    if (Transition.shape != (M, M) or
            Emission.shape[0] != M or
            Initial.size != M or Observations.size == 0):
        return None, None

    if (not np.issubdtype(Observations.dtype, np.integer) or
            np.any(Observations < 0) or
            np.any(Observations >= Emission.shape[1])):
        return None, None

    if (not np.all(np.isfinite(Transition)) or
            not np.all(np.isfinite(Emission)) or
            not np.all(np.isfinite(Initial)) or
            np.any(Transition < 0) or np.any(Emission < 0) or
            np.any(Initial < 0)):
        return None, None

    T = Observations.shape[0]
    N = Emission.shape[1]
    Initial = Initial.reshape(M)

    for _ in range(iterations):
        alpha = np.zeros((T, M))
        beta = np.zeros((T, M))

        alpha[0] = Initial * Emission[:, Observations[0]]
        scale = np.sum(alpha[0])
        if scale == 0:
            return None, None
        alpha[0] /= scale

        for t in range(1, T):
            alpha[t] = (
                alpha[t - 1] @ Transition
            ) * Emission[:, Observations[t]]
            scale = np.sum(alpha[t])
            if scale == 0:
                return None, None
            alpha[t] /= scale

        beta[T - 1] = 1

        for t in range(T - 2, -1, -1):
            beta[t] = Transition @ (
                Emission[:, Observations[t + 1]] * beta[t + 1]
            )
            scale = np.sum(beta[t])
            if scale != 0:
                beta[t] /= scale

        gamma = alpha * beta
        gamma_sum = np.sum(gamma, axis=1, keepdims=True)
        gamma_sum[gamma_sum == 0] = 1
        gamma /= gamma_sum

        xi_sum = np.zeros((M, M))
        gamma_den = np.sum(gamma[:-1], axis=0)

        for t in range(T - 1):
            xi = (
                alpha[t, :, None] * Transition *
                (Emission[:, Observations[t + 1]] * beta[t + 1])[None, :]
            )
            total = np.sum(xi)
            if total != 0:
                xi_sum += xi / total

        for i in range(M):
            if gamma_den[i] != 0:
                Transition[i] = xi_sum[i] / gamma_den[i]

        for i in range(M):
            denominator = np.sum(gamma[:, i])
            if denominator != 0:
                for j in range(N):
                    Emission[i, j] = np.sum(
                        gamma[Observations == j, i]
                    ) / denominator

    return Transition, Emission
