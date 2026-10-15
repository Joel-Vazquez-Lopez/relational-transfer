"""Resumable, two-worker launcher for the frozen Phase-1 factorial."""
from __future__ import annotations

import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "run_factorial.py"
SEEDS = (312001, 312002, 312003)
CELLS = ("cn1_p3", "cn1_p4", "cn2_p3", "cn2_p4")
JOBS = list(itertools.product(range(8), CELLS, SEEDS))
LOG = HERE / "RUN_LEDGER.jsonl"


def entry(data: dict) -> None:
    with LOG.open("a") as handle:
        handle.write(json.dumps({"timestamp": time.time(), **data}) + "\n")


def main() -> None:
    active = {}
    pending = [job for job in JOBS if not (HERE / "full" / f"family{job[0]}" / job[1] /
                                           f"seed_{job[2]}" / "metrics.json").exists()]
    print(json.dumps({"planned": len(JOBS), "pending": len(pending), "workers": 2}), flush=True)
    while pending or active:
        while pending and len(active) < 2:
            family, cell, seed = pending.pop(0)
            label = f"family{family}_{cell}_seed{seed}"
            log_path = HERE / "logs" / f"{label}.log"
            log_path.parent.mkdir(exist_ok=True)
            output = log_path.open("w")
            process = subprocess.Popen(
                [sys.executable, str(SCRIPT), "train", "--family", str(family),
                 "--cell", cell, "--seed", str(seed)],
                stdout=output, stderr=subprocess.STDOUT, cwd=HERE)
            active[process.pid] = (process, output, label, log_path)
            entry({"event": "start", "job": label, "pid": process.pid})
            print(f"START {label}", flush=True)
        time.sleep(5)
        for pid, (process, output, label, log_path) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            output.close()
            del active[pid]
            entry({"event": "finish", "job": label, "exit_code": code,
                   "log": str(log_path)})
            print(f"FINISH {label} exit={code}", flush=True)
            if code:
                for other, _, _, _ in active.values():
                    other.terminate()
                raise RuntimeError(f"Training failed: {label}; see {log_path}")
    print("ALL_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
