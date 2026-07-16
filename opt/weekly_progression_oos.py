"""Out-of-sample week-by-week equity progression (forward OOS window).

Same six studies as ``opt.weekly_progression`` but replayed over the genuinely
unseen forward window 2025-06-01 → 2026-04-30. The shipped configs were tuned
only on data through 2025-06-01 (in-sample), so this window is real
out-of-sample history — every candle post-dates the optimization. Indicators
are warmed up from earlier cached data; the portfolio starts fresh at the config
initial balance on 2025-06-01, so multiples read as "growth over the OOS window
alone".

Window note: the literal ask was 2025-06 → 2026-06, but Bitget futures history
has genuine candle holes in May 2026 (missing 2026-05-01 4h/1d and 2026-05-17 1h
candles for ETH and SOL). The data pipeline fails closed on gaps by design, so
the window is truncated to the last gap-free month boundary — 2026-04-30 —
common to all three assets, keeping the six studies on one comparable window.
BTC alone is gap-free through 2026-06.

Run:
    PYTHONPATH=. /tmp/tmlvenv/bin/python -m opt.weekly_progression_oos
Output:
    reports/weekly_progression_oos.xlsx
"""

from __future__ import annotations

from pathlib import Path

from opt.weekly_progression import collect_studies, build_workbook

OOS_START = "2025-06-01"
OOS_END = "2026-05-01"
DATA_END = "2026-04-30"
FUNDING_END = "2026-05-01"


def main() -> None:
    studies = collect_studies(
        OOS_START, OOS_END, data_end=DATA_END, funding_end=FUNDING_END,
    )
    out_path = Path("reports/weekly_progression_oos.xlsx")
    build_workbook(studies, out_path, f"{OOS_START} → {OOS_END} (out-of-sample)")
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
