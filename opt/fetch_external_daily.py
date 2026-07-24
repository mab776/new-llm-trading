"""Download + pin external daily context series (DXY, SPX) for the
cross-market context-vote probe (opt/probe_context_votes.py).

Writes history/external/{dxy,spx}_1d.csv — Date (session day, UTC-naive),
OHLC + Volume (0 for DXY). Pinned CSVs are the source of truth for sims:
re-running this script refreshes them, but probe results must always cite
the pinned file so folds are reproducible offline.

Causality convention (pre-declared in the probe): a daily bar dated D is
usable at 4h decision closes >= D+1 00:00 UTC (cash/futures sessions close
20:00-22:00 UTC; midnight next day is conservatively after both), and the
series forward-fills over weekends/holidays — exactly the staleness live
would see.

Run: /tmp/tmlvenv/bin/python opt/fetch_external_daily.py
"""
from __future__ import annotations

from pathlib import Path

import yfinance as yf

OUT = Path(__file__).parent.parent / "history" / "external"
TICKERS = {"dxy": "DX-Y.NYB", "spx": "^GSPC"}
START = "2019-12-01"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, ticker in TICKERS.items():
        df = yf.download(ticker, start=START, interval="1d",
                         auto_adjust=False, progress=False)
        if df is None or df.empty:
            raise SystemExit(f"{ticker}: empty download")
        # yfinance MultiIndex columns (field, ticker) -> flat
        if hasattr(df.columns, "levels"):
            df.columns = [c[0] for c in df.columns]
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index.name = "Date"
        path = OUT / f"{name}_1d.csv"
        df.to_csv(path)
        print(f"{name}: {len(df)} rows {df.index[0].date()} -> "
              f"{df.index[-1].date()} -> {path}")


if __name__ == "__main__":
    main()
