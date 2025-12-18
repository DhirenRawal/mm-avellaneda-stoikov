"""
simulator.py

Simple market simulator for our project.

It does three things:
  1) Evolve a mid price S_t using a random walk.
  2) Add random background LIMIT orders around the mid price.
  3) Add random background MARKET orders that hit the book.

We will later plug our market-making strategy into this environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .lob import OrderBook, Side


@dataclass
class SimulatorConfig:
    # Initial mid price
    S0: float = 100.0
    # Volatility parameter for the random walk (per time step)
    sigma: float = 0.1
    # Time step size (seconds)
    dt: float = 1.0

    # Background flow parameters
    background_limit_rate: float = 5.0   # expected # of limit orders per step
    background_mkt_rate: float = 1.0     # expected # of market orders per step

    # Microstructure-ish parameters
    tick_size: float = 0.01              # smallest price increment
    depth_per_order: float = 1.0         # quantity per background order


class SimpleMarketSimulator:
    """
    Simple, stylized market environment wrapped around an OrderBook.

    At each step:
      * update mid price with a Gaussian random shock
      * add some random limit orders around the mid
      * add some random market orders that hit the existing book
    """

    def __init__(self, book: OrderBook, config: SimulatorConfig):
        self.book = book
        self.cfg = config
        self.S = config.S0
        self.t = 0.0

    # ---------- price process ----------

    def _step_mid_price(self) -> None:
        """
        Update mid price using a simple random walk:
            S_{t+dt} = S_t + sigma * sqrt(dt) * N(0, 1)
        """
        dS = self.cfg.sigma * np.sqrt(self.cfg.dt) * np.random.randn()
        self.S += dS

    # ---------- background order flow ----------

    def _random_price_around_mid(self) -> float:
        """
        Choose a random price a few ticks around the current mid.
        This just makes the book look somewhat realistic.
        """
        # choose an integer k in [-5, 5]
        k = np.random.randint(-5, 6)
        price = self.S + k * self.cfg.tick_size
        # round to, say, 4 decimal places to avoid floating noise
        return round(price, 4)

    def _add_background_limit_orders(self) -> None:
        """
        Add a random number of small limit orders around the mid price.
        """
        lam = self.cfg.background_limit_rate
        n_orders = np.random.poisson(lam)

        for _ in range(n_orders):
            side = Side.BUY if np.random.rand() < 0.5 else Side.SELL
            price = self._random_price_around_mid()
            qty = self.cfg.depth_per_order
            self.book.add_limit_order(
                side=side,
                price=price,
                qty=qty,
                timestamp=self.t,
            )

    def _add_background_market_orders(self) -> None:
        """
        Add random market orders that hit the top of the book.
        """
        lam = self.cfg.background_mkt_rate
        n_orders = np.random.poisson(lam)

        for _ in range(n_orders):
            side = Side.BUY if np.random.rand() < 0.5 else Side.SELL
            qty = self.cfg.depth_per_order
            self.book.process_market_order(
                side=side,
                qty=qty,
                timestamp=self.t,
            )

    # ---------- public API ----------

    def step(self) -> Dict[str, Optional[float]]:
        """
        Advance the simulation by one time step.

        Returns a snapshot dict with:
          - t        : current time
          - mid      : mid price S_t
          - best_bid : current best bid
          - best_ask : current best ask
        """
        self.t += self.cfg.dt
        self._step_mid_price()
        self._add_background_limit_orders()
        self._add_background_market_orders()

        snapshot: Dict[str, Optional[float]] = {
            "t": self.t,
            "mid": self.S,
            "best_bid": self.book.best_bid(),
            "best_ask": self.book.best_ask(),
        }
        return snapshot
