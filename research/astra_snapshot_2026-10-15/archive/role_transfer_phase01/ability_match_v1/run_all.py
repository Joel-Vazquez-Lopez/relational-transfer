"""Two-worker resumable launcher for ability-only calibration or full development."""
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

import match_experiment as exp

HERE = Path(__file__).resolve().parent
LEDGER = HERE / "RUN_LEDGER.jsonl"


def entry(data: dict) -> None:
    with LEDGER.open("a") as handle:
        handle.write(json.dumps({"timestamp": time.time(), **data}) + "\n")


def jobs(stage: str) -> list[tuple]:
    if stage == "calibration":
        return list(itertools.product(range(2), exp.CONDITIONS, exp.SEEDS, exp.WEIGHTS))
    choice = json.loads((HERE / "CALIBRATION.json").read_text())
    if not choice["calibration_match"]:
        raise RuntimeError("calibration did not meet the two-point ability gate")
    return list(itertools.product(range(8), exp.CONDITIONS, exp.SEEDS,
                                  (choice["selected_weight"],)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("calibration", "full"))
    args = ap.parse_args()
    exp.audit()
    all_jobs = jobs(args.stage)
    pending = [j for j in all_jobs if not (exp.dest(*j) / "metrics.json").exists()]
    active = {}
    print(json.dumps({"stage": args.stage, "planned": len(all_jobs),
                      "pending": len(pending), "workers": 2}), flush=True)
    while pending or active:
        while pending and len(active) < 2:
            family, condition, seed, weight = pending.pop(0)
            name = f"f{family}_{condition}_s{seed}_w{exp.label(weight)}"
            log = HERE / "logs" / f"{name}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            handle = log.open("a")
            process = subprocess.Popen([sys.executable, str(HERE / "match_experiment.py"), "train",
                                        "--family", str(family), "--condition", condition,
                                        "--seed", str(seed), "--weight", str(weight)],
                                       stdout=handle, stderr=subprocess.STDOUT, cwd=HERE)
            active[process.pid] = (process, handle, name, log)
            entry({"event": "start", "stage": args.stage, "job": name, "pid": process.pid})
            print(f"START {name}", flush=True)
        time.sleep(5)
        for pid, (process, handle, name, log) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            handle.close()
            del active[pid]
            entry({"event": "finish", "stage": args.stage, "job": name,
                   "exit_code": code, "log": str(log)})
            print(f"FINISH {name} exit={code}", flush=True)
            if code:
                for other, _, _, _ in active.values():
                    other.terminate()
                raise RuntimeError(f"training failed: {name}; inspect {log}")
    print("ALL_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
