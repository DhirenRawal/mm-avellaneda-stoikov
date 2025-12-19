"""
execution.py

Execution / risk layer for our market-making system.

Responsibilities:
  * Keep track of our cash and inventory.
  * Ask the strategy for new bid/ask prices.
  * Apply inventory risk controls (soft + hard limits).
  * Use an Avellaneda–Stoikov-style λ(δ) model to simulate fills:
        lambda_bid = A * exp(-k * (S_t - bid))
        lambda_ask = A * exp(-k * (ask - S_t))
    and draw Poisson fills each step.
  * Attribute PnL into:
      - spread PnL
      - inventory PnL
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict
from types import SimpleNamespace
import math

import numpy as np

from .lob import OrderBook, Side
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
    Execution layer that owns our quotes.

    It:
      * asks the A–S strategy for bid/ask quotes
      * applies inventory-based risk limits
      * simulates fills using λ(δ) with Poisson arrivals
      * updates cash and inventory
      * tracks spread vs inventory PnL
    """

    def __init__(
        self,
        book: OrderBook,
        strategy: AvellanedaStoikovStrategy,
        dt: Optional[float] = None,
        trade_size: float = 1.0,
    ):
        self.book = book
        self.strategy = strategy

        self.state = PositionState()

        # We no longer keep actual resting orders in the book for λ-fills,
        # but we still keep these for the hard-limit "flatten" behavior.
        self.bid_order_id: Optional[int] = None
        self.ask_order_id: Optional[int] = None

        # PnL tracking
        self.pnl_state = PnLState()
        self.last_mid: Optional[float] = None

        # --- risk control parameters ---
        # "Soft" limit: start widening spreads when |inventory| > soft_limit
        self.soft_limit: float = 10.0
        # "Hard" limit: if |inventory| > hard_limit, immediately flatten
        self.hard_limit: float = 20.0
        # How aggressively to widen spreads when beyond soft_limit
        self.spread_widen_factor: float = 0.5

        # --- λ(δ) arrival model parameters ---
        # Try to read A, k, dt from the A–S params; fall back to defaults.
        self.A: float = getattr(strategy.params, "A", 1.0)
        self.k: float = getattr(strategy.params, "k", 1.0)
        self.dt: float = dt if dt is not None else getattr(strategy.params, "dt", 1.0)
        self.trade_size: float = trade_size

    # -------- internal helpers --------

    def _cancel_existing_quotes(self) -> None:
        """
        Cancel our current resting bid/ask (if any) in the book.

        In the Day-10 λ(δ) model, we don't rely on actual resting orders
        for fills, but we still use the book for the hard-limit flattening
        logic, so we keep this around.
        """
        if self.bid_order_id is not None:
            self.book.cancel_order(self.bid_order_id)
            self.bid_order_id = None

        if self.ask_order_id is not None:
            self.book.cancel_order(self.ask_order_id)
            self.ask_order_id = None

    def _apply_inventory_risk_adjustment(
        self, bid_px: float, ask_px: float
    ) -> tuple[Optional[float], Optional[float]]:
        """
        Adjust quotes based on inventory.

        If |inventory| > hard_limit:
            return (None, None) to signal we should FLATTEN (kill-switch).

        If soft_limit < |inventory| <= hard_limit:
            widen spreads by pushing bid down and ask up.
        """
        inv = self.state.inventory

        # Hard limit -> we should flatten immediately (no quotes this step)
        if abs(inv) > self.hard_limit:
            return None, None

        # Soft limit -> widen spreads
        if abs(inv) > self.soft_limit:
            # how much to widen relative to current spread
            widen = self.spread_widen_factor * abs(inv / self.soft_limit)
            spread = ask_px - bid_px
            spread_adj = widen * spread

            bid_px -= spread_adj
            ask_px += spread_adj

        return bid_px, ask_px

    # -------- public API --------

    def update_quotes(self, mid: float, t: float) -> Dict[str, float]:
        """
        Called once per simulation step.

        Steps:
          1) Remember inventory at start of step (for inventory PnL).
          2) Cancel existing quotes (for completeness).
          3) Ask strategy for new bid/ask prices and sizes.
          4) Apply inventory-based risk adjustments to quotes.
          5) If hard limit breached, flatten inventory via a market order
             in the book and skip λ(δ) fills for this step.
          6) Otherwise, use λ(δ) to draw Poisson numbers of trades hitting
             our bid and ask, and update cash/inventory accordingly.
          7) Attribute PnL for this step.
          8) Return logging info.

        Returns a dict with:
          bid, ask, bid_qty, ask_qty,
          spread_pnl_step, inv_pnl_step,
          spread_pnl_cum, inv_pnl_cum,
          fills_bid, fills_ask,
          avg_trade_spread, lambda_bid, lambda_ask
        """
        inventory_prev = self.state.inventory
        mid_prev = self.last_mid if self.last_mid is not None else mid

        # 2) Cancel old quotes in the book (not essential for λ, but keeps book clean)
        self._cancel_existing_quotes()

        # 3) Compute base quotes from strategy (no risk adjustments yet)
        bid_px, ask_px, bid_qty, ask_qty = self.strategy.compute_quotes(
            S=mid,
            q=self.state.inventory,
            t=t,
        )

        # 4) Apply inventory-based risk adjustments
        risk_bid, risk_ask = self._apply_inventory_risk_adjustment(bid_px, ask_px)

        # 5) If hard limit exceeded -> flatten inventory via market order
        if risk_bid is None or risk_ask is None:
            inv = self.state.inventory
            if abs(inv) > 0:
                side = Side.SELL if inv > 0 else Side.BUY
                trades = self.book.process_market_order(
                    side=side,
                    qty=abs(inv),
                    timestamp=t,
                )

                # When flattening, we are the taker; reuse existing PnL logic.
                # Update cash/inventory from those trades.
                for tr in trades:
                    if tr.taker_side is Side.BUY:
                        # we bought qty at price
                        self.state.inventory += tr.qty
                        self.state.cash -= tr.price * tr.qty
                    else:  # Side.SELL
                        self.state.inventory -= tr.qty
                        self.state.cash += tr.price * tr.qty

                spread_step, inv_step = self.pnl_state.update_for_step(
                    mid_prev=mid_prev,
                    mid_now=mid,
                    inventory_prev=inventory_prev,
                    trades=trades,
                )
                self.last_mid = mid

                return {
                    "bid": None,
                    "ask": None,
                    "bid_qty": 0.0,
                    "ask_qty": 0.0,
                    "spread_pnl_step": spread_step,
                    "inv_pnl_step": inv_step,
                    "spread_pnl_cum": self.pnl_state.cumulative_spread_pnl,
                    "inv_pnl_cum": self.pnl_state.cumulative_inventory_pnl,
                    "fills_bid": 0.0,
                    "fills_ask": 0.0,
                    "avg_trade_spread": 0.0,
                    "lambda_bid": 0.0,
                    "lambda_ask": 0.0,
                }
            else:
                # No inventory but hard limit triggered; just don't quote.
                self.last_mid = mid
                return {
                    "bid": None,
                    "ask": None,
                    "bid_qty": 0.0,
                    "ask_qty": 0.0,
                    "spread_pnl_step": 0.0,
                    "inv_pnl_step": 0.0,
                    "spread_pnl_cum": self.pnl_state.cumulative_spread_pnl,
                    "inv_pnl_cum": self.pnl_state.cumulative_inventory_pnl,
                    "fills_bid": 0.0,
                    "fills_ask": 0.0,
                    "avg_trade_spread": 0.0,
                    "lambda_bid": 0.0,
                    "lambda_ask": 0.0,
                }

        # If we got here, we have adjusted quotes we can use
        bid_px, ask_px = risk_bid, risk_ask

        # 6) λ(δ) fill model: Poisson arrivals hitting OUR quotes.
        #    delta_bid = S_t - bid_price
        #    delta_ask = ask_price - S_t
        delta_bid = max(mid - bid_px, 0.0)
        delta_ask = max(ask_px - mid, 0.0)

        lambda_bid = self.A * math.exp(-self.k * delta_bid)
        lambda_ask = self.A * math.exp(-self.k * delta_ask)

        # Expected arrivals per step = lambda * dt
        n_bid_trades = np.random.poisson(lambda_bid * self.dt)
        n_ask_trades = np.random.poisson(lambda_ask * self.dt)

        # Total volume on each side, capped by our quote size
        max_bid_trades = int(bid_qty / self.trade_size) if self.trade_size > 0 else 0
        max_ask_trades = int(ask_qty / self.trade_size) if self.trade_size > 0 else 0

        n_bid_trades = min(n_bid_trades, max_bid_trades)
        n_ask_trades = min(n_ask_trades, max_ask_trades)

        filled_bid = n_bid_trades * self.trade_size
        filled_ask = n_ask_trades * self.trade_size

        # Update our position: we are effectively BUY at bid, SELL at ask
        if filled_bid > 0:
            self.state.inventory += filled_bid
            self.state.cash -= filled_bid * bid_px

        if filled_ask > 0:
            self.state.inventory -= filled_ask
            self.state.cash += filled_ask * ask_px

        # For PnLState, we just need objects with price, qty, taker_side.
        # We mark taker_side as OUR side (BUY when we buy, SELL when we sell),
        # to keep the sign convention consistent with _apply_trades().
        trades: List[object] = []
        if filled_bid > 0:
            trades.append(
                SimpleNamespace(price=bid_px, qty=filled_bid, taker_side=Side.BUY)
            )
        if filled_ask > 0:
            trades.append(
                SimpleNamespace(price=ask_px, qty=filled_ask, taker_side=Side.SELL)
            )

        spread_step, inv_step = self.pnl_state.update_for_step(
            mid_prev=mid_prev,
            mid_now=mid,
            inventory_prev=inventory_prev,
            trades=trades,
        )
        self.last_mid = mid

        # Fills per side and average spread at which we trade
        total_filled = filled_bid + filled_ask
        if total_filled > 0:
            # average (half) spread vs mid, quantity-weighted
            avg_spread = (
                (mid - bid_px) * filled_bid + (ask_px - mid) * filled_ask
            ) / total_filled
        else:
            avg_spread = 0.0

        return {
            "bid": bid_px,
            "ask": ask_px,
            "bid_qty": bid_qty,
            "ask_qty": ask_qty,
            "spread_pnl_step": spread_step,
            "inv_pnl_step": inv_step,
            "spread_pnl_cum": self.pnl_state.cumulative_spread_pnl,
            "inv_pnl_cum": self.pnl_state.cumulative_inventory_pnl,
            # Day-10 extras:
            "fills_bid": filled_bid,
            "fills_ask": filled_ask,
            "avg_trade_spread": avg_spread,
            "lambda_bid": lambda_bid,
            "lambda_ask": lambda_ask,
        }
