"""
lob.py

Limit Order Book and basic matching engine for our market making simulator.

Day 2: we implemented storing / cancelling orders and querying best bid/ask.
Day 3: we add matching logic for:
  * crossing limit orders
  * market orders

We still keep add_limit_order() as a "resting only" helper, and build
higher-level process_limit_order() and process_market_order() on top of it.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import Deque, Dict, List, Optional, Tuple


class Side(Enum):
    BUY = auto()
    SELL = auto()


@dataclass
class Order:
    """
    Simple limit order representation.
    - id:        unique order identifier
    - side:      BUY or SELL
    - price:     limit price
    - qty:       remaining quantity
    - timestamp: when the order was created (for FIFO within price level)
    """
    id: int
    side: Side
    price: float
    qty: float
    timestamp: float


@dataclass
class Trade:
    """
    Simple trade/fill record.

    taker_side:  side of the aggressive order that initiated the trade
    maker_id:    id of the resting (passive) order on the book that was hit
    price:       trade price
    qty:         traded quantity
    timestamp:   time of the trade
    """
    taker_side: Side
    maker_id: int
    price: float
    qty: float
    timestamp: float


class OrderBook:
    """
    Very simple limit order book:
    - bids: map price -> deque of BUY orders (highest price is best bid)
    - asks: map price -> deque of SELL orders (lowest price is best ask)

    We support:
      * adding a resting limit order (no matching) via add_limit_order()
      * cancelling an order by id
      * querying best bid / best ask
      * processing crossing limit orders
      * processing market orders
    """

    def __init__(self) -> None:
        # price -> deque[Order]
        self.bids: Dict[float, Deque[Order]] = {}
        self.asks: Dict[float, Deque[Order]] = {}

        # map order_id -> (side, price) so we can find and cancel it fast
        self._order_index: Dict[int, Tuple[Side, float]] = {}
        self._next_order_id: int = 1

    # ---------- internal helpers ----------

    def _get_book(self, side: Side) -> Dict[float, Deque[Order]]:
        return self.bids if side is Side.BUY else self.asks

    def _best_price(self, side: Side) -> Optional[float]:
        """
        Internal helper to get best price on a side.
        """
        book = self._get_book(side)
        if not book:
            return None
        if side is Side.BUY:
            return max(book.keys())
        else:
            return min(book.keys())

    def _pop_empty_level(self, side: Side, price: float) -> None:
        """
        Remove price level if its deque is empty.
        """
        book = self._get_book(side)
        level = book.get(price)
        if level is not None and not level:
            del book[price]

    def next_order_id(self) -> int:
        oid = self._next_order_id
        self._next_order_id += 1
        return oid

    # ---------- basic public API (Day 2) ----------

    def add_limit_order(
        self,
        side: Side,
        price: float,
        qty: float,
        timestamp: float,
    ) -> Order:
        """
        Add a new RESTING limit order to the book.
        This function assumes there is no crossing/matching; it is used
        by higher-level methods once any immediate trades are handled.

        Returns the created Order object.
        """
        order_id = self.next_order_id()
        order = Order(
            id=order_id,
            side=side,
            price=price,
            qty=qty,
            timestamp=timestamp,
        )

        book = self._get_book(side)
        if price not in book:
            book[price] = deque()
        book[price].append(order)

        self._order_index[order_id] = (side, price)
        return order

    def cancel_order(self, order_id: int) -> bool:
        """
        Cancel an existing order by id.
        Returns True if order was found and cancelled, False otherwise.
        """
        info = self._order_index.get(order_id)
        if info is None:
            return False

        side, price = info
        book = self._get_book(side)
        level = book.get(price)
        if level is None:
            # should not happen if index is consistent, but be defensive
            self._order_index.pop(order_id, None)
            return False

        # linear search within the price level deque
        for order in list(level):
            if order.id == order_id:
                level.remove(order)
                if not level:
                    # no more orders at this price
                    del book[price]
                break

        self._order_index.pop(order_id, None)
        return True

    def best_bid(self) -> Optional[float]:
        """
        Return the best bid price (highest bid) or None if no bids.
        """
        return self._best_price(Side.BUY)

    def best_ask(self) -> Optional[float]:
        """
        Return the best ask price (lowest ask) or None if no asks.
        """
        return self._best_price(Side.SELL)

    def depth_at_price(self, side: Side, price: float) -> float:
        """
        Total quantity available at a given price on a given side.
        """
        book = self._get_book(side)
        level = book.get(price)
        if level is None:
            return 0.0
        return sum(order.qty for order in level)

    # ---------- matching logic (Day 3) ----------

    def _match_against_book(
        self,
        taker_side: Side,
        qty: float,
        timestamp: float,
        limit_price: Optional[float] = None,
    ) -> Tuple[float, List[Trade]]:
        """
        Core matching function used by both market and limit orders.

        taker_side: side of the aggressive (incoming) order
        qty:        quantity of the taker order
        timestamp:  time of arrival
        limit_price: for limit orders, the worst acceptable price.
                     For market orders, this is None (accept any price).

        Returns:
          remaining_qty, trades
        """
        trades: List[Trade] = []

        # The taker hits the OPPOSITE side of the book.
        if taker_side is Side.BUY:
            book = self.asks
            book_side = Side.SELL
            # best ask = lowest ask price
            def next_price() -> Optional[float]:
                return min(book.keys()) if book else None
            price_cmp = lambda book_price: (
                True if limit_price is None else book_price <= limit_price
            )
        else:  # taker SELL
            book = self.bids
            book_side = Side.BUY
            # best bid = highest bid price
            def next_price() -> Optional[float]:
                return max(book.keys()) if book else None
            price_cmp = lambda book_price: (
                True if limit_price is None else book_price >= limit_price
            )

        while qty > 0 and book:
            best_price = next_price()
            if best_price is None:
                break

            # Check limit price constraint
            if limit_price is not None and not price_cmp(best_price):
                # best available price is worse than our limit; stop matching
                break

            level = book[best_price]
            if not level:
                del book[best_price]
                continue

            maker_order = level[0]
            trade_qty = min(qty, maker_order.qty)

            # record trade
            trades.append(
                Trade(
                    taker_side=taker_side,
                    maker_id=maker_order.id,
                    price=best_price,
                    qty=trade_qty,
                    timestamp=timestamp,
                )
            )

            # update quantities
            maker_order.qty -= trade_qty
            qty -= trade_qty

            if maker_order.qty <= 0:
                # fully filled maker order
                level.popleft()
                self._order_index.pop(maker_order.id, None)
                if not level:
                    del book[best_price]

        return qty, trades

    def process_market_order(
        self,
        side: Side,
        qty: float,
        timestamp: float,
    ) -> List[Trade]:
        """
        Process a MARKET order that immediately trades against the book.

        Market BUY hits asks; market SELL hits bids.

        Returns list of Trade objects created.
        """
        remaining_qty, trades = self._match_against_book(
            taker_side=side,
            qty=qty,
            timestamp=timestamp,
            limit_price=None,
        )
        # For a pure market order we ignore any unfilled remainder.
        _ = remaining_qty
        return trades

    def process_limit_order(
        self,
        side: Side,
        price: float,
        qty: float,
        timestamp: float,
    ) -> Tuple[Optional[Order], List[Trade]]:
        """
        Process a LIMIT order that may:
          * trade immediately against the opposite book (if crossing), and/or
          * leave a remaining portion resting on the book.

        Returns:
          (resting_order, trades)

        resting_order is:
          * an Order object if there is remaining qty that was added to the book
          * None if the order was fully filled and nothing is resting
        """
        # First, try to match as taker against opposite side.
        remaining_qty, trades = self._match_against_book(
            taker_side=side,
            qty=qty,
            timestamp=timestamp,
            limit_price=price,
        )

        resting_order: Optional[Order] = None
        if remaining_qty > 0:
            # Whatever is left becomes a resting limit order on our side.
            resting_order = self.add_limit_order(
                side=side,
                price=price,
                qty=remaining_qty,
                timestamp=timestamp,
            )

        return resting_order, trades
