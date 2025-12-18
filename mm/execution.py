"""
execution.py

Execution / risk layer for our market-making system.

Responsibilities:
  * Keep track of our cash and inventory.
  * Cancel existing quotes.
  * Ask the strategy for new bid/ask prices.
  * Submit limit orders via the OrderBook.
  * Update P&L from trades (immediate fills) and attribute PnL into:
      - spread PnL
      - inventory PnL
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict

from .lob import OrderBook, Side, Trade
from .avellaneda import AvellanedaStoikovStrategy
from .pnl import PnLState


@dataclass
class PositionState:
    """
    Tracks our financial state.

    cash:      cash balance
    inventory: number of units of the asset we hold
    """
    cash: float = 0.0
    inventory: float = 0.0

    def equity(self, mid: float) -> float:
        """
        Mark-to-market equity = cash + inventory * mid.
        """
        return self.cash + self.inventory * mid


class ExecutionEngine:
    """
    Execution layer that owns our quotes in the order book.

    It:
      * manages current bid/ask orders (their IDs)
      * updates cash and inventory based on trades
      * calls the A–S strategy to compute new quotes
      * tracks spread vs inventory PnL
    """

    def __init__(self, book: OrderBook, strategy: AvellanedaStoikovStrategy):
        self.book = book
        self.strategy = strategy

        self.state = PositionState()

        # current resting orders (if any)
        self.bid_order_id: Optional[int] = None
        self.ask_order_id: Optional[int] = None

        # PnL tracking
        self.pnl_state = PnLState()
        self.last_mid: Optional[float] = None

    # -------- internal helpers --------

    def _apply_trades(self, trades: List[Trade]) -> None:
        """
        Update cash and inventory given trades in which WE are the taker.
        In process_limit_order(), taker_side is the side of OUR incoming order.
        """
        for t in trades:
            if t.taker_side is Side.BUY:
                # we bought qty at price
                self.state.inventory += t.qty
                self.state.cash -= t.price * t.qty
            else:  # Side.SELL
                # we sold qty at price
                self.state.inventory -= t.qty
                self.state.cash += t.price * t.qty

    def _cancel_existing_quotes(self) -> None:
        """
        Cancel our current resting bid/ask (if any) in the book.
        """
        if self.bid_order_id is not None:
            self.book.cancel_order(self.bid_order_id)
            self.bid_order_id = None

        if self.ask_order_id is not None:
            self.book.cancel_order(self.ask_order_id)
            self.ask_order_id = None

    # -------- public API --------

    def update_quotes(self, mid: float, t: float) -> Dict[str, float]:
        """
        Called once per simulation step.

        Steps:
          1) Remember inventory at start of step (for inventory PnL).
          2) Cancel existing quotes.
          3) Ask strategy for new bid/ask prices and sizes.
          4) Submit limit BUY and SELL via process_limit_order().
          5) Update cash/inventory from any immediate trades.
          6) Compute spread + inventory PnL for this step.
          7) Store resting order IDs (if any).

        Returns a dict with:
          bid, ask, bid_qty, ask_qty,
          spread_pnl_step, inv_pnl_step,
          spread_pnl_cum, inv_pnl_cum
        """
        inventory_prev = self.state.inventory
        mid_prev = self.last_mid

        # 2) Cancel old quotes
        self._cancel_existing_quotes()

        # 3) Compute new quotes based on current inventory
        bid_px, ask_px, bid_qty, ask_qty = self.strategy.compute_quotes(
            S=mid,
            q=self.state.inventory,
            t=t,
        )

        # 4) Submit BUY limit (our bid)
        bid_resting, bid_trades = self.book.process_limit_order(
            side=Side.BUY,
            price=bid_px,
            qty=bid_qty,
            timestamp=t,
        )
        self._apply_trades(bid_trades)
        self.bid_order_id = bid_resting.id if bid_resting is not None else None

        # 4b) Submit SELL limit (our ask)
        ask_resting, ask_trades = self.book.process_limit_order(
            side=Side.SELL,
            price=ask_px,
            qty=ask_qty,
            timestamp=t,
        )
        self._apply_trades(ask_trades)
        self.ask_order_id = ask_resting.id if ask_resting is not None else None

        # 5) PnL attribution for this step
        all_trades = bid_trades + ask_trades
        spread_step, inv_step = self.pnl_state.update_for_step(
            mid_prev=mid_prev,
            mid_now=mid,
            inventory_prev=inventory_prev,
            trades=all_trades,
        )
        self.last_mid = mid

        # 6) Return quotes + PnL info for logging
        return {
            "bid": bid_px,
            "ask": ask_px,
            "bid_qty": bid_qty,
            "ask_qty": ask_qty,
            "spread_pnl_step": spread_step,
            "inv_pnl_step": inv_step,
            "spread_pnl_cum": self.pnl_state.cumulative_spread_pnl,
            "inv_pnl_cum": self.pnl_state.cumulative_inventory_pnl,
        }
