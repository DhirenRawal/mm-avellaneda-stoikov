from mm.lob import OrderBook, Side


def test_add_and_best_prices():
    ob = OrderBook()

    # no bids/asks yet
    assert ob.best_bid() is None
    assert ob.best_ask() is None

    # add some bids
    ob.add_limit_order(Side.BUY, price=99.0, qty=5, timestamp=0.0)
    ob.add_limit_order(Side.BUY, price=100.0, qty=3, timestamp=1.0)

    # best bid should be highest price
    assert ob.best_bid() == 100.0

    # add some asks
    ob.add_limit_order(Side.SELL, price=101.0, qty=2, timestamp=2.0)
    ob.add_limit_order(Side.SELL, price=102.0, qty=4, timestamp=3.0)

    # best ask should be lowest price
    assert ob.best_ask() == 101.0


def test_cancel_order():
    ob = OrderBook()

    order = ob.add_limit_order(Side.BUY, price=100.0, qty=5, timestamp=0.0)
    assert ob.best_bid() == 100.0

    ok = ob.cancel_order(order.id)
    assert ok is True
    assert ob.best_bid() is None  # no more bids after cancel
