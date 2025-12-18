"""
execution.py

Execution / risk layer for our market-making system.

Responsibilities:
  * Keep track of our cash and inventory.
  * Cancel existing quotes.
  * Ask the strategy for new bid/ask prices.
  * Submit limit orders via the OrderBook.
  * Update P&L from any trades that happen immediately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

from .lob import OrderBook, Side, Trade
from .avellaneda import AvellanedaStoikovStrategy


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

    For now, we only handle immediate trades when we submit our limit orders
    (taker behavior). Later we can extend this to track passive fills too.
    """

    def __init__(self, book: OrderBook, strategy: AvellanedaStoikovStrategy):
        self.book = book
        self.strategy = strategy

        self.state = PositionState()

        # current resting orders (if any)
        self.bid_order_id: Optional[int] = None
        self.ask_order_id: Optional[int] = None

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
          1) Cancel existing quotes.
          2) Ask strategy for new bid/ask prices and sizes.
          3) Submit limit BUY and SELL via process_limit_order().
          4) Update cash/inventory from any immediate trades.
          5) Store resting order IDs (if any).

        Returns a dict with current quotes (for logging/printing).
        """
        # 1) Cancel old quotes
        self._cancel_existing_quotes()

        # 2) Compute new quotes based on current inventory
        bid_px, ask_px, bid_qty, ask_qty = self.strategy.compute_quotes(
            S=mid,
            q=self.state.inventory,
            t=t,
        )

        # 3) Submit BUY limit (our bid)
        bid_resting, bid_trades = self.book.process_limit_order(
            side=Side.BUY,
            price=bid_px,
            qty=bid_qty,
            timestamp=t,
        )
        self._apply_trades(bid_trades)
        if bid_resting is not None:
            self.bid_order_id = bid_resting.id
        else:
            self.bid_order_id = None

        # 4) Submit SELL limit (our ask)
        ask_resting, ask_trades = self.book.process_limit_order(
            side=Side.SELL,
            price=ask_px,
            qty=ask_qty,
            timestamp=t,
        )
        self._apply_trades(ask_trades)
        if ask_resting is not None:
            self.ask_order_id = ask_resting.id
        else:
            self.ask_order_id = None

        # 5) Return quotes for logging
        return {
            "bid": bid_px,
            "ask": ask_px,
            "bid_qty": bid_qty,
            "ask_qty": ask_qty,
        }
