"""
run_mm_sim.py

End-to-end demo:
  * SimpleMarketSimulator creates price + background flow
  * Avellaneda–Stoikov strategy computes quotes
  * ExecutionEngine sends orders and tracks P&L
"""

from mm.lob import OrderBook
from mm.simulator import SimpleMarketSimulator, SimulatorConfig
from mm.avellaneda import ASParams, AvellanedaStoikovStrategy
from mm.execution import ExecutionEngine


def main():
    # 1) Order book and simulator
    book = OrderBook()
    sim_cfg = SimulatorConfig(
        S0=100.0,
        sigma=0.2,
        dt=1.0,
        background_limit_rate=5.0,
        background_mkt_rate=1.0,
        tick_size=0.01,
        depth_per_order=1.0,
    )
    sim = SimpleMarketSimulator(book, sim_cfg)

    # 2) Avellaneda–Stoikov strategy parameters
    as_params = ASParams(
        gamma=0.01,   # risk aversion (smaller = more risk-taking)
        sigma=0.2,    # volatility estimate (roughly match sim sigma)
        k=1.0,        # liquidity sensitivity
        A=50.0,       # base arrival rate (not used directly yet)
        T=100.0,      # trading horizon in seconds
        dt=sim_cfg.dt,
    )
    strat = AvellanedaStoikovStrategy(as_params, size=1.0)

    # 3) Execution engine
    engine = ExecutionEngine(book, strat)

    # 4) Run simulation
    n_steps = 50

    print(
        "t   mid      bid      ask    inv   cash       equity"
    )
    print("-" * 60)

    for _ in range(n_steps):
        snap = sim.step()
        t = snap["t"]
        mid = snap["mid"]

        quotes = engine.update_quotes(mid, t)
        equity = engine.state.equity(mid)

        print(
            f"{t:3.0f}  "
            f"{mid:7.3f}  "
            f"{quotes['bid']:7.3f}  "
            f"{quotes['ask']:7.3f}  "
            f"{engine.state.inventory:5.1f}  "
            f"{engine.state.cash:9.2f}  "
            f"{equity:9.2f}"
        )


if __name__ == "__main__":
    main()
