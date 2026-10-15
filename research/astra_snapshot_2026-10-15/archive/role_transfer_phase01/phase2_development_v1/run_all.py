"""Two-worker resumable launcher for the frozen development configurations."""
from __future__ import annotations

import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

import experiment as exp

HERE = Path(__file__).resolve().parent
LEDGER = HERE / "RUN_LEDGER.jsonl"
JOBS = list(itertools.product(range(8), exp.CONDITIONS, exp.SEEDS, exp.ARMS))


def entry(data):
    with LEDGER.open("a") as handle:
        handle.write(json.dumps({"timestamp": time.time(), **data}) + "\n")


def main():
    exp.audit()
    pending = [job for job in JOBS
               if not (exp.output_dir(job[0], job[1], job[2], job[3], False, exp.SWAP_WEIGHT)
                       / "metrics.json").exists()]
    active = {}
    print(json.dumps({"planned": len(JOBS), "pending": len(pending), "workers": 2}), flush=True)
    while pending or active:
        while pending and len(active) < 2:
            family, condition, seed, arm = pending.pop(0)
            label = f"f{family}_{condition}_s{seed}_{arm}"
            log = HERE / "logs" / f"{label}.log"
            log.parent.mkdir(exist_ok=True)
            output = log.open("a")
            process = subprocess.Popen([
                sys.executable, str(HERE / "experiment.py"), "train",
                "--family", str(family), "--condition", condition,
                "--seed", str(seed), "--arm", arm],
                stdout=output, stderr=subprocess.STDOUT, cwd=HERE)
            active[process.pid] = (process, output, label, log)
            entry({"event": "start", "job": label, "pid": process.pid})
            print(f"START {label}", flush=True)
        time.sleep(5)
        for pid, (process, output, label, log) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            output.close()
            del active[pid]
            entry({"event": "finish", "job": label, "exit_code": code,
                   "log": str(log)})
            print(f"FINISH {label} exit={code}", flush=True)
            if code:
                for other, _, _, _ in active.values():
                    other.terminate()
                raise RuntimeError(f"Training failed: {label}; inspect {log}")
    print("ALL_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
