"""Engine sanity checks — run before trusting any grid output."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from opt.scalp import grid
from opt.scalp.engine import ExecParams, simulate
from opt.scalp.strategies import STRATEGIES


def main() -> None:
    grid.load_all("BTCUSDT", "15m")
    G = grid._G
    n = len(G["index"])
    print(f"{n} 15m bars; subbars: {G['subbars'].shape}")
    nan_rows = int(np.isnan(G["subbars"][:, 0, 0]).sum())
    print(f"subbar rows without coverage: {nan_rows} ({nan_rows/n*100:.2f}%)")

    zeros = np.zeros(n, dtype=bool)

    # 1. no signals -> no trades, growth exactly 1.0
    ep = ExecParams()
    r = simulate(G["ohlc"], G["atr"], zeros, zeros, ep, funding=G["funding"])
    assert r.trades == 0 and abs(r.growth_x - 1.0) < 1e-12, r
    print("PASS: no signals -> no trades")

    # 2. random signals, fees on vs off: fee drag must be strictly negative
    rng = np.random.default_rng(7)
    sig = rng.random(n) < 0.01
    i0, i1 = grid.fold_bounds(("x", "2024-01-01", "2025-01-01"))
    ep_fees = ExecParams(entry_mode="taker", sl_atr=1.5, tp_atr=1.5)
    ep_free = ExecParams(entry_mode="taker", sl_atr=1.5, tp_atr=1.5,
                         maker_fee=0.0, taker_fee=0.0, slip=0.0)
    r_fees = simulate(G["ohlc"], G["atr"], sig, zeros, ep_fees,
                      start_i=i0, end_i=i1)
    r_free = simulate(G["ohlc"], G["atr"], sig, zeros, ep_free,
                      start_i=i0, end_i=i1)
    print(f"random longs 2024: fees {r_fees.growth_x:.4f}x vs "
          f"frictionless {r_free.growth_x:.4f}x (trades {r_fees.trades})")
    assert r_fees.growth_x < r_free.growth_x
    # random entries frictionless with symmetric brackets should be ~breakeven
    assert 0.5 < r_free.growth_x < 2.0, r_free
    print("PASS: fee drag negative; frictionless random ~breakeven")

    # 3. maker mode fills fewer entries than orders (some run away / cancel)
    ep_mk = ExecParams(entry_mode="maker", sl_atr=1.5, tp_atr=1.5)
    r_mk = simulate(G["ohlc"], G["atr"], sig, zeros, ep_mk, start_i=i0, end_i=i1)
    fill_rate = r_mk.maker_fills / max(1, r_mk.maker_orders)
    print(f"maker: {r_mk.maker_orders} orders -> {r_mk.maker_fills} fills "
          f"({fill_rate*100:.1f}%), postonly_cancels "
          f"{r_mk.exit_counts.get('postonly_cancel', 0)}")
    assert r_mk.maker_fills < r_mk.maker_orders

    # 4. lookahead guard: acting on a signal shifted one bar into the future
    # must NOT be reproducible by the engine itself — verify the engine acts
    # at i+1 by checking that entries never occur on the signal bar. Proxy:
    # a signal on the LAST bar produces no trade.
    sig_last = zeros.copy()
    sig_last[i1 - 1] = True
    r_last = simulate(G["ohlc"], G["atr"], sig_last, zeros,
                      ExecParams(entry_mode="taker"), start_i=i0, end_i=i1)
    assert r_last.trades == 0, r_last
    print("PASS: signal on final bar cannot act (decisions act next bar)")

    # 5. sub-bar replay changes both-hit ordering vs pure adverse-first:
    # with subs, some both-touched bars resolve to TP; without, SL always wins.
    sig2 = rng.random(n) < 0.02
    ep_tight = ExecParams(entry_mode="taker", sl_atr=0.7, tp_atr=0.7)
    r_subs = simulate(G["ohlc"], G["atr"], sig2, zeros, ep_tight,
                      start_i=i0, end_i=i1, subbars=G["subbars"])
    r_flat = simulate(G["ohlc"], G["atr"], sig2, zeros, ep_tight,
                      start_i=i0, end_i=i1, subbars=None)
    print(f"tight brackets 2024: subs win_rate {r_subs.win_rate:.1f}% vs "
          f"adverse-first {r_flat.win_rate:.1f}%")
    assert r_subs.win_rate > r_flat.win_rate
    print("PASS: sub-bar replay resolves ordering less pessimistically")

    # 6. speed
    strat = STRATEGIES["bb_reversion"]
    ls, ss, mel, mes = strat(G["df"], {"n": 16, "z_in": 2.0}, G["ctx"])
    t0 = time.monotonic()
    r = simulate(G["ohlc"], G["atr"], ls, ss, ExecParams(exit_on_mean=True),
                 mean_exit_long=mel, mean_exit_short=mes,
                 funding=G["funding"], subbars=G["subbars"])
    dt = time.monotonic() - t0
    print(f"full-history bb_reversion run: {dt*1000:.0f}ms, {r.trades} trades, "
          f"{r.growth_x:.3f}x")
    print("ALL SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
