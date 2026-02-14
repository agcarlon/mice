"""
MICE: Multi-Iteration stochastiC Estimator

A gradient estimator for stochastic optimization that uses successive control
variates along the optimization path to reduce variance.
"""

from .core_impl import MICE  # noqa: F401
from .plot_mice import plot_mice  # noqa: F401

__version__ = "1.0.0"

