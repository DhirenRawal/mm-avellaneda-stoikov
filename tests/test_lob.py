from mm.lob import OrderBook, Side


def test_add_and_best_prices():
    ob = OrderBook()

    # no bids/asks yet
    assert ob.best_bid() is None
    assert ob.best_ask() is None

    # add some bids (resting only helper)
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


def test_crossing_limit_order_trades_then_rests():
    """
    A buy limit above best ask should trade immediately, then rest leftover.
    """
    ob = OrderBook()

    # Resting ask: 101 x 5
    ask_order = ob.add_limit_order(Side.SELL, price=101.0, qty=5, timestamp=0.0)

    # Incoming buy limit: price 102, qty 3 -> should hit ask at 101
    resting, trades = ob.process_limit_order(
        side=Side.BUY,
        price=102.0,
        qty=3.0,
        timestamp=1.0,
    )

    # All 3 should be traded against the ask at 101
    assert len(trades) == 1
    t = trades[0]
    assert t.price == 101.0
    assert t.qty == 3.0
    assert t.maker_id == ask_order.id

    # The incoming order was fully filled, so nothing should rest
    assert resting is None

    # The original ask now has only 2 qty left at 101
    assert ob.best_ask() == 101.0
    assert abs(ob.depth_at_price(Side.SELL, 101.0) - 2.0) < 1e-9


def test_market_order_hits_multiple_levels():
    """
    Market buy should walk the ask side, consuming orders from best ask upwards.
    """
    ob = OrderBook()

    # Build an ask book: 101 x 2, 102 x 3
    o1 = ob.add_limit_order(Side.SELL, price=101.0, qty=2.0, timestamp=0.0)
    o2 = ob.add_limit_order(Side.SELL, price=102.0, qty=3.0, timestamp=0.1)

    trades = ob.process_market_order(
        side=Side.BUY,
        qty=4.0,
        timestamp=1.0,
    )

    # Should fully consume 2 at 101 and 2 out of 3 at 102
    assert len(trades) == 2
    # First trade should be against best ask (101)
    assert trades[0].price == 101.0
    assert trades[0].qty == 2.0
    assert trades[0].maker_id == o1.id

    # Second trade at 102
    assert trades[1].price == 102.0
    assert trades[1].qty == 2.0
    assert trades[1].maker_id == o2.id

    # Remaining ask depth: 1 @ 102
    assert ob.best_ask() == 102.0
    assert abs(ob.depth_at_price(Side.SELL, 102.0) - 1.0) < 1e-9
