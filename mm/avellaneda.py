"""
avellaneda.py

Implements the Avellaneda–Stoikov market making strategy.

This module provides:
  * reservation_price(...)
  * optimal_spread(...)
  * ASParams dataclass
  * AvellanedaStoikovStrategy class

This is the "brains" of our market maker.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass


# ================================================================
# === Formula components =========================================
# ================================================================

def reservation_price(S: float, q: float, gamma: float, sigma: float, tau: float) -> float:
    """
    Compute: r_t = S_t - q * gamma * sigma^2 * tau
    The inventory penalty shifts quotes up/down.
    """
    return S - q * gamma * (sigma**2) * tau


def optimal_spread(gamma: float, sigma: float, tau: float, k: float, A: float) -> float:
    """
    Compute the symmetric half-spread delta:

        delta = (gamma * sigma^2 * tau) / 2  +  (1/gamma) * ln(1 + gamma/k)

    This is the standard Avellaneda–Stoikov approximation.
    Returns the HALF-spread. Final quotes are r_t ± delta.
    """
    first_term = (gamma * (sigma**2) * tau) / 2
    second_term = (1.0 / gamma) * np.log(1 + gamma / k)
    return first_term + second_term


# ================================================================
# === Parameters ==================================================
# ================================================================

@dataclass
class ASParams:
    gamma: float   # risk aversion
    sigma: float   # volatility estimate
    k: float       # liquidity sensitivity parameter
    A: float       # base arrival rate parameter (used in λ(δ))
    T: float       # time horizon
    dt: float      # simulation time step


# ================================================================
# === Strategy Class =============================================
# ================================================================

class AvellanedaStoikovStrategy:
    """
    The Avellaneda–Stoikov quoting strategy.

    At each step, given:
        S_t (mid price)
        q_t (inventory)
        t   (time)

    It computes:
        * reservation price r_t
        * optimal half-spread delta_t
        * bid/ask = r_t ± delta

    Returns:
        bid_price, ask_price, bid_size, ask_size
    """

    def __init__(self, params: ASParams, size: float = 1.0):
        # IMPORTANT for Day 10:
        # The execution layer expects .params
        self.params = params       # <-- required for ExecutionEngine
        self.p = params            # (keep old name for backward compatibility)

        # Order size
        self.order_size = size

    # --------------------------------------------------------------

    def compute_quotes(self, S: float, q: float, t: float):
        """
        Compute bid/ask prices for this simulation step.
        """

        # Remaining time
        tau = max(self.p.T - t, 0.0)

        # === 1) Reservation price ===
        r_t = reservation_price(
            S=S,
            q=q,
            gamma=self.p.gamma,
            sigma=self.p.sigma,
            tau=tau,
        )

        # === 2) Optimal half-spread ===
        delta = optimal_spread(
            gamma=self.p.gamma,
            sigma=self.p.sigma,
            tau=tau,
            k=self.p.k,
            A=self.p.A,
        )

        # === 3) Final quotes ===
        bid = r_t - delta
        ask = r_t + delta

        # return bid price, ask price, bid size, ask size
        return bid, ask, self.order_size, self.order_size
# ================================================================
# === Naive fixed-spread market maker ============================
# ================================================================

class NaiveMMStrategy:
    """
    Simple baseline MM strategy:

      * Always quotes bid = S_t - fixed_spread/2
                and ask = S_t + fixed_spread/2
      * Ignores inventory (no reservation-price shift)
      * Uses a fixed order size.

    We still attach a .params field so that the ExecutionEngine
    can read A, k, dt for the λ(δ) fill model.
    """

    def __init__(self, params: ASParams, fixed_spread: float, size: float = 1.0):
        # Re-use ASParams to carry A, k, dt (and gamma/sigma/T if needed)
        self.params = params
        self.p = params

        self.fixed_spread = fixed_spread
        self.order_size = size

    def compute_quotes(self, S: float, q: float, t: float):
        """
        Compute naive bid/ask:

            bid = S_t - fixed_spread / 2
            ask = S_t + fixed_spread / 2

        Ignores inventory q and time t.
        """
        half = self.fixed_spread / 2.0
        bid = S - half
        ask = S + half
        return bid, ask, self.order_size, self.order_size
