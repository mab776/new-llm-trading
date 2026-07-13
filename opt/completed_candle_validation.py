"""Revalidate both profiles after enforcing completed-candle availability."""

from __future__ import annotations

import json
from pathlib import Path

from opt.cadence_correction import validate_profile


def main() -> None:
    result = {
        "method": (
            "bar-open normalized OHLCV; every primary/secondary input visible only "
            "after candle close; 1h sub-bar exits with trailing fixed intrabar and "
            "ratcheted once per completed 4h bar; maker entry; funding; liquidation; "
            "2bps market-exit slippage"
        ),
        "supersedes": "opt/cadence_correction_results.json",
        "standard": validate_profile("standard"),
        "aggressive": validate_profile("aggressive"),
    }
    path = Path("opt/completed_candle_results.json")
    path.write_text(json.dumps(result, indent=2, default=str) + "\n")
    for profile in ("standard", "aggressive"):
        row = result[profile]
        print(
            f"{profile}: continuous={row['continuous']['compound_x']:,.2f}x "
            f"DD={row['continuous']['reported_max_dd']:.2f}% "
            f"MTM-DD={row['continuous']['mark_to_market_max_dd']:.2f}% "
            f"test={row['held_out_test']['compound_x']:,.2f}x"
        )
    print(f"saved {path}")


if __name__ == "__main__":
    main()
