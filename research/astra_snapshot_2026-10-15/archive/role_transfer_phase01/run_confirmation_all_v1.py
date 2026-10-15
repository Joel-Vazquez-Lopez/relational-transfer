"""Run all sealed confirmation models; progress only, no interim outcome reading."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys

import confirmation_runner_v1 as study

ROOT = Path(__file__).resolve().parent / "role_aux_confirmation_v1"
LOGS = ROOT / "logs"


def completed(world: str, seed: int, arm: str) -> bool:
    directory = study.model_path(world, seed, arm)
    metrics_path = directory / "metrics.json"
    checkpoint = directory / "model.pt"
    if not metrics_path.exists() and not checkpoint.exists():
        return False
    if not metrics_path.exists() or not checkpoint.exists():
        raise RuntimeError(f"Partial output requires review: {directory}")
    metrics = json.loads(metrics_path.read_text())
    checksum = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if checksum != metrics["model_sha256"]:
        raise RuntimeError(f"Hash mismatch: {directory}")
    return True


def main() -> None:
    audit = study.audit()
    LOGS.mkdir(parents=True, exist_ok=True)
    plan = list(itertools.product(study.WORLDS, study.SEEDS))
    finished = sum(completed(world, seed, arm)
                   for world, seed in plan for arm in study.ARMS)
    print(json.dumps({"audit": audit, "already_complete": finished,
                      "planned": len(plan) * len(study.ARMS)}), flush=True)
    for triple_index, (world, seed) in enumerate(plan, 1):
        processes = []
        for arm in study.ARMS:
            if completed(world, seed, arm):
                continue
            log_path = LOGS / f"{world}_seed_{seed}_{arm}.log"
            if log_path.exists():
                raise FileExistsError(f"Existing log without valid output: {log_path}")
            stream = log_path.open("w")
            command = [sys.executable, str(Path(__file__).resolve().parent / "confirmation_runner_v1.py"),
                       "train", "--world", world, "--seed", str(seed), "--arm", arm]
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            processes.append((arm, process, stream, log_path))
        print(f"Started triple {triple_index}/{len(plan)}: {world} seed={seed} "
              f"new_models={len(processes)}", flush=True)
        failures = []
        for arm, process, stream, log_path in processes:
            code = process.wait()
            stream.close()
            if code != 0:
                failures.append({"arm": arm, "exit_code": code, "log": str(log_path)})
            elif not completed(world, seed, arm):
                failures.append({"arm": arm, "exit_code": code, "log": str(log_path),
                                 "reason": "missing or invalid output"})
        if failures:
            raise RuntimeError(f"Confirmation run failed at {world} seed={seed}: {failures}")
        finished = sum(completed(w, s, arm) for w, s in plan for arm in study.ARMS)
        print(f"Completed {finished}/{len(plan) * len(study.ARMS)} models", flush=True)
    print("All sealed confirmation models completed. Analyze with confirmation_runner_v1.py summarize.",
          flush=True)


if __name__ == "__main__":
    main()
