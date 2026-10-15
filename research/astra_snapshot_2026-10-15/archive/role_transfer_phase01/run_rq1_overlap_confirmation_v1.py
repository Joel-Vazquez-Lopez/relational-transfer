"""Run every frozen overlap-confirmation model; no interim outcome reading."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys

import rq1_overlap_confirmation_v1 as study

LOGS = study.ROOT / "logs"


def completed(world: str, seed: int, arm: str) -> bool:
    directory = study.model_path(world, seed, arm)
    metrics_path, checkpoint = directory / "metrics.json", directory / "model.pt"
    if not metrics_path.exists() and not checkpoint.exists():
        return False
    if not metrics_path.exists() or not checkpoint.exists():
        raise RuntimeError(f"Partial output: {directory}")
    data = json.loads(metrics_path.read_text())
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != data["model_sha256"]:
        raise RuntimeError(f"Checkpoint hash mismatch: {directory}")
    return True


def main() -> None:
    print(json.dumps(study.audit()), flush=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    plan = list(itertools.product((f"overlap{i}" for i in range(study.GRAPH_COUNT)), study.SEEDS))
    total = len(plan) * len(study.ARMS)
    for index, (world, seed) in enumerate(plan, 1):
        processes = []
        for arm in study.ARMS:
            if completed(world, seed, arm):
                continue
            log_path = LOGS / f"{world}_seed_{seed}_{arm}.log"
            if log_path.exists():
                raise FileExistsError(f"Existing log for incomplete model: {log_path}")
            stream = log_path.open("w")
            command = [sys.executable, str(study.HERE / "rq1_overlap_confirmation_v1.py"),
                       "train", "--world", world, "--seed", str(seed), "--arm", arm]
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            processes.append((arm, process, stream, log_path))
        print(f"Started matched triple {index}/{len(plan)}: {world} seed={seed}", flush=True)
        for arm, process, stream, log_path in processes:
            code = process.wait()
            stream.close()
            if code or not completed(world, seed, arm):
                raise RuntimeError(f"{world} seed={seed} arm={arm} failed; see {log_path}")
        done = sum(completed(w, s, arm) for w, s in plan for arm in study.ARMS)
        print(f"Completed {done}/{total} models", flush=True)
    result = study.summarize()
    (study.ROOT / "SUMMARY.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "graph_rows"}), flush=True)


if __name__ == "__main__":
    main()
