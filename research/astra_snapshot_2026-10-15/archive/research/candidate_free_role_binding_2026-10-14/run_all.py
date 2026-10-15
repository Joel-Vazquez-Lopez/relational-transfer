#!/usr/bin/env python3
"""Resume the four frozen candidate-free development configurations."""
from __future__ import annotations

import subprocess
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("candidate_free_pilot", ROOT / "pilot.py")
assert spec and spec.loader
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)
ARMS, STARTS, validate = pilot.ARMS, pilot.STARTS, pilot.validate


def main() -> None:
    validate()
    script = ROOT / "pilot.py"
    for start in STARTS:
        for arm in ARMS:
            adapter = ROOT / f"{start}_{arm}" / "adapter.pt"
            evaluation = ROOT / f"EVAL_{start}_{arm}.json"
            if not adapter.exists():
                subprocess.run([sys.executable, str(script), "train", "--start", start,
                                "--arm", arm], check=True)
            if not evaluation.exists():
                subprocess.run([sys.executable, str(script), "eval", "--start", start,
                                "--arm", arm], check=True)
    subprocess.run([sys.executable, str(ROOT / "analyze.py")], check=True)


if __name__ == "__main__":
    main()
