"""High-penalty turnover sensitivity for the walk-forward robustness study."""

from __future__ import annotations

import json
from pathlib import Path

from opt.walk_forward_retune import run


SEEDS = [17, 73, 211, 419, 887]
PENALTIES = [200.0, 500.0]


def main() -> None:
    cache = {}
    rows = []
    for penalty in PENALTIES:
        for seed in SEEDS:
            result = run(
                60, seed, entry_mode="maker", exit_granularity="sub",
                turnover_penalty=penalty, _evaluation_cache=cache,
            )
            rows.append(result)
            print(
                f"penalty={penalty:.0f} seed={seed} "
                f"ratio={result['growth_ratio']:.3f} "
                f"turnover={result['total_parameter_turnover']:.3f}"
            )
    payload = {
        "method": "60 candidates/window; shared cached TRAIN evaluations",
        "penalties": PENALTIES,
        "seeds": SEEDS,
        "runs": rows,
    }
    Path("opt/walk_forward_turnover_results.json").write_text(
        json.dumps(payload, indent=2)
    )


if __name__ == "__main__":
    main()
