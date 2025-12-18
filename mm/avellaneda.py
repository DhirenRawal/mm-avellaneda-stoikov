"""
avellaneda.py

Implements the Avellaneda–Stoikov market making strategy.

This module provides:
  * reservation_price(...)
  * optimal_spread(...)
  * AvellanedaStoikovStrategy class

This is the "brains" of our market maker.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass


def reservation_price(S: float, q: float, gamma: float, sigma: float, tau: float) -> float:
    """
    Compute the reservation price r_t = S_t - q * gamma * sigma^2 * tau
    This shifts quotes up/down depending on inventory.
    """
    return S - q * gamma * (sigma**2) * tau


def optimal_spread(gamma: float, sigma: float, tau: float, k: float, A: float) -> float:
    """
    Compute the symmetric half-spread delta_t according to the
    standard Avellaneda–Stoikov approximation:

      delta = (gamma * sigma^2 * tau) / 2 + (1/gamma) * ln(1 + gamma/k)

    Returns the HALF-spread. Final quotes are r_t ± delta.

    Note: This is a common approximation of the A–S solution.
    """
    first_term = (gamma * (sigma**2) * tau) / 2
    second_term = (1.0 / gamma) * np.log(1 + gamma / k)
    return first_term + second_term


@dataclass
class ASParams:
    gamma: float       # risk aversion
    sigma: float       # volatility estimate
    k: float           # liquidity sensitivity parameter
    A: float           # base arrival rate parameter
    T: float           # time horizon
    dt: float          # time step size


class AvellanedaStoikovStrategy:
    """
    The strategy computes:
      * reservation price r_t
      * half-spread delta_t
      * final bid/ask quotes

    Inputs each step:
      * mid price S_t
      * current inventory q_t
      * current time t

    Output:
      * bid price
      * ask price
      * bid size
      * ask size
    """

    def __init__(self, params: ASParams, size: float = 1.0):
        self.p = params
        self.order_size = size

    def compute_quotes(self, S: float, q: float, t: float):
        """
        Compute bid/ask prices for the current step.
        """
        tau = max(self.p.T - t, 0.0)  # time to horizon

        # --- 1) Reservation price ---
        r_t = reservation_price(
            S=S,
            q=q,
            gamma=self.p.gamma,
            sigma=self.p.sigma,
            tau=tau,
        )

        # --- 2) Optimal half-spread ---
        delta = optimal_spread(
            gamma=self.p.gamma,
            sigma=self.p.sigma,
            tau=tau,
            k=self.p.k,
            A=self.p.A,
        )

        # --- 3) Final quote prices ---
        bid = r_t - delta
        ask = r_t + delta

        return bid, ask, self.order_size, self.order_size
