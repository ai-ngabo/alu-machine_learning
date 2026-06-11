#!/usr/bin/env python3
"""Determine whether to stop gradient descent early."""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    Determines if training should stop early.

    Args:
        cost: current validation cost
        opt_cost: lowest recorded validation cost
        threshold: minimum improvement threshold
        patience: patience count
        count: current count of epochs without sufficient improvement

    Returns:
        tuple (stop, count)
            stop: boolean indicating whether to stop early
            count: updated count
    """
    if opt_cost - cost > threshold:
        count = 0
    else:
        count += 1

    return (count >= patience, count)
