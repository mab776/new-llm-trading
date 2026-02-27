#!/usr/bin/env python3
"""Convenience script to run the test suite."""

import subprocess
import sys


def main() -> int:
    """Run pytest with verbose output."""
    cmd = [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"]
    # Forward any extra CLI args (e.g. -k, --pdb)
    cmd.extend(sys.argv[1:])
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
