"""
run_strategy_compare.py

Day 11:
Compare Avellaneda–Stoikov vs a naive fixed-spread MM strategy.

We run many simulation paths for each strategy and report:
  * Mean final PnL
  * Std of final PnL
  * Sharpe (mean / std of step PnL)
  * Max drawdown (average over runs)
  * Average |inventory| over time
"""

from __future__ import annotations

import numpy as np

from mm.lob import OrderBook
from mm.simulator import SimpleMarketSimulator, SimulatorConfig
from mm.avellaneda import (
    ASParams,
    AvellanedaStoikovStrategy,
    NaiveMMStrategy,
)
from mm.execution import ExecutionEngine


# ---------- helper to run ONE path for a given strategy ----------

def run_single_path(strategy_factory, n_steps: int, seed: int = 0):
    """
    strategy_factory: function that returns a (strategy, sim_cfg) pair.

    Returns:
      dict with final_pnl, sharpe, max_drawdown, avg_abs_inventory
    """
    np.random.seed(seed)

    book = OrderBook()
    strategy, sim_cfg = strategy_factory()
    sim = SimpleMarketSimulator(book, sim_cfg)
    engine = ExecutionEngine(book, strategy, dt=sim_cfg.dt, trade_size=1.0)

    times = []
    equities = []
    abs_inv = []

    # run the path
    for _ in range(n_steps):
        snap = sim.step()
        t = snap["t"]
        mid = snap["mid"]

        quotes = engine.update_quotes(mid, t)
        eq = engine.state.equity(mid)

        times.append(t)
        equities.append(eq)
        abs_inv.append(abs(engine.state.inventory))

    equities = np.asarray(equities, dtype=float)
    abs_inv = np.asarray(abs_inv, dtype=float)

    # PnL and returns
    pnl_total = equities[-1] - equities[0]
    rets = np.diff(equities)
    if rets.std() > 0:
        sharpe = rets.mean() / rets.std() * np.sqrt(len(rets))
    else:
        sharpe = 0.0

    # max drawdown (simple)
    running_max = np.maximum.accumulate(equities)
    drawdown = equities - running_max
    max_dd = drawdown.min()

    avg_abs_inv = abs_inv.mean()

    return {
        "final_pnl": pnl_total,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "avg_abs_inventory": avg_abs_inv,
    }


# ---------- factories for each strategy --------------------------

def make_as_strategy():
    """Factory that builds an Avellaneda–Stoikov strategy + sim config."""
    sim_cfg = SimulatorConfig(
        S0=100.0,
        sigma=0.2,
        dt=1.0,
        background_limit_rate=5.0,
        background_mkt_rate=1.0,
        tick_size=0.01,
        depth_per_order=1.0,
    )

    params = ASParams(
        gamma=0.01,
        sigma=0.2,
        k=1.0,
        A=50.0,
        T=100.0,
        dt=sim_cfg.dt,
    )
    strat = AvellanedaStoikovStrategy(params, size=1.0)
    return strat, sim_cfg


def make_naive_strategy():
    """Factory that builds a naive fixed-spread strategy + sim config."""
    sim_cfg = SimulatorConfig(
        S0=100.0,
        sigma=0.2,
        dt=1.0,
        background_limit_rate=5.0,
        background_mkt_rate=1.0,
        tick_size=0.01,
        depth_per_order=1.0,
    )

    # reuse ASParams just to carry A, k, dt for λ(δ)
    params = ASParams(
        gamma=0.01,   # not used by NaiveMMStrategy
        sigma=0.2,    # not used by NaiveMMStrategy
        k=1.0,
        A=50.0,
        T=100.0,
        dt=sim_cfg.dt,
    )

    # choose a fixed spread similar to typical A–S spread
    fixed_spread = 0.5  # tweak this as you like
    strat = NaiveMMStrategy(params=params, fixed_spread=fixed_spread, size=1.0)
    return strat, sim_cfg


# ---------- main experiment --------------------------------------

def run_experiment(n_paths: int = 50, n_steps: int = 100):
    as_results = []
    naive_results = []

    for i in range(n_paths):
        # Different random seed for each run
        seed = 1234 + i

        as_res = run_single_path(make_as_strategy, n_steps=n_steps, seed=seed)
        naive_res = run_single_path(make_naive_strategy, n_steps=n_steps, seed=seed)

        as_results.append(as_res)
        naive_results.append(naive_res)

    def summarize(results, name: str):
        final_pnls = np.array([r["final_pnl"] for r in results])
        sharpes = np.array([r["sharpe"] for r in results])
        max_dds = np.array([r["max_drawdown"] for r in results])
        avg_abs_inv = np.array([r["avg_abs_inventory"] for r in results])

        print(f"\n===== {name} =====")
        print(f"# paths: {len(results)}")
        print(f"Mean final PnL      : {final_pnls.mean():8.3f}")
        print(f"Std final PnL       : {final_pnls.std():8.3f}")
        print(f"Mean Sharpe         : {sharpes.mean():8.3f}")
        print(f"Mean max drawdown   : {max_dds.mean():8.3f}")
        print(f"Mean |inv| over time: {avg_abs_inv.mean():8.3f}")

    summarize(as_results, "Avellaneda–Stoikov")
    summarize(naive_results, "Naive fixed-spread MM")


if __name__ == "__main__":
    run_experiment(n_paths=50, n_steps=100)
