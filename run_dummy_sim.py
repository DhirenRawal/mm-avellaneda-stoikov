"""
run_dummy_sim.py

Quick sanity check for our SimpleMarketSimulator.

It runs a short simulation and prints:
  time, mid price, best bid, best ask
so we can see that the book and price process are evolving.
"""

from mm.lob import OrderBook
from mm.simulator import SimpleMarketSimulator, SimulatorConfig


def main():
    book = OrderBook()
    cfg = SimulatorConfig(
        S0=100.0,
        sigma=0.1,
        dt=1.0,
        background_limit_rate=5.0,
        background_mkt_rate=1.0,
        tick_size=0.01,
        depth_per_order=1.0,
    )
    sim = SimpleMarketSimulator(book, cfg)

    n_steps = 20
    for _ in range(n_steps):
        snap = sim.step()
        print(
            f"t={snap['t']:4.0f}  "
            f"mid={snap['mid']:7.3f}  "
            f"bid={snap['best_bid']}  "
            f"ask={snap['best_ask']}"
        )


if __name__ == "__main__":
    main()
