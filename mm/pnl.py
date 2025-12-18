"""
pnl.py

PnL attribution helpers for the market maker.

We decompose equity changes into:
  * spread PnL     - profit from trading at better prices than mid
  * inventory PnL  - profit/loss from price moves while holding inventory
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .lob import Trade, Side


def trade_spread_pnl(mid: float, trade: Trade) -> float:
    """
    Spread PnL contribution from a single trade, using the mid price
    at the moment of the trade.

    If we BUY below mid or SELL above mid -> positive spread PnL.
    """
    if trade.taker_side is Side.BUY:
        # we bought qty at trade.price
        # good if we bought below mid
        return (mid - trade.price) * trade.qty
    else:
        # we sold qty at trade.price
        # good if we sold above mid
        return (trade.price - mid) * trade.qty


@dataclass
class PnLState:
    """
    Tracks cumulative PnL components.

    cumulative_spread_pnl:     sum of all spread PnL over time
    cumulative_inventory_pnl:  sum of all inventory PnL over time
    """
    cumulative_spread_pnl: float = 0.0
    cumulative_inventory_pnl: float = 0.0

    def update_for_step(
        self,
        mid_prev: Optional[float],
        mid_now: float,
        inventory_prev: float,
        trades: List[Trade],
    ) -> tuple[float, float]:
        """
        Update PnL for a single simulation step.

        mid_prev:        mid at previous step (None for first step)
        mid_now:         mid at current step
        inventory_prev:  our inventory at the *start* of the step
        trades:          list of trades that happened during this step

        Returns:
          (spread_pnl_step, inventory_pnl_step)
        """
        # 1) Spread PnL from trades this step
        spread_step = sum(trade_spread_pnl(mid_now, tr) for tr in trades)
        self.cumulative_spread_pnl += spread_step

        # 2) Inventory PnL from price move while holding inventory_prev
        inventory_step = 0.0
        if mid_prev is not None:
            inventory_step = inventory_prev * (mid_now - mid_prev)
            self.cumulative_inventory_pnl += inventory_step

        return spread_step, inventory_step
