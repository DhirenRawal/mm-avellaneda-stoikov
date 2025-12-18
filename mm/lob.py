"""
lob.py

Basic Limit Order Book data structures for our market making simulator.
Day 2: we only handle storing / cancelling orders and querying best bid/ask.
Matching and simulation will come later.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import Deque, Dict, Optional


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


class OrderBook:
    """
    Very simple limit order book:
    - bids: map price -> deque of BUY orders (highest price is best bid)
    - asks: map price -> deque of SELL orders (lowest price is best ask)

    For Day 2 we support:
      * adding a resting limit order (no matching yet)
      * cancelling an order by id
      * querying best bid / best ask
    """

    def __init__(self) -> None:
        # price -> deque[Order]
        self.bids: Dict[float, Deque[Order]] = {}
        self.asks: Dict[float, Deque[Order]] = {}

        # map order_id -> (side, price) so we can find and cancel it fast
        self._order_index: Dict[int, tuple[Side, float]] = {}
        self._next_order_id: int = 1

    # ---------- internal helpers ----------

    def _get_book(self, side: Side) -> Dict[float, Deque[Order]]:
        return self.bids if side is Side.BUY else self.asks

    # ---------- public API ----------

    def next_order_id(self) -> int:
        oid = self._next_order_id
        self._next_order_id += 1
        return oid

    def add_limit_order(
        self,
        side: Side,
        price: float,
        qty: float,
        timestamp: float,
    ) -> Order:
        """
        Add a new resting limit order to the book.
        (No crossing / matching logic yet; that will come in Day 3.)

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
        for i, order in enumerate(level):
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
        if not self.bids:
            return None
        return max(self.bids.keys())

    def best_ask(self) -> Optional[float]:
        """
        Return the best ask price (lowest ask) or None if no asks.
        """
        if not self.asks:
            return None
        return min(self.asks.keys())

    def depth_at_price(self, side: Side, price: float) -> float:
        """
        Total quantity available at a given price on a given side.
        """
        book = self._get_book(side)
        level = book.get(price)
        if level is None:
            return 0.0
        return sum(order.qty for order in level)
