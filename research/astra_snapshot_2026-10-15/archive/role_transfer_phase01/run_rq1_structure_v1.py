"""Run the frozen RQ1 graph contrast, resuming only valid completed models."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys

import rq1_structure_v1 as study

LOGS = study.ROOT / "logs"


def completed(pair: str, group: str, seed: int) -> bool:
    directory = study.model_path(pair, group, seed)
    metrics, checkpoint = directory / "metrics.json", directory / "model.pt"
    if not metrics.exists() and not checkpoint.exists():
        return False
    if not metrics.exists() or not checkpoint.exists():
        raise RuntimeError(f"Incomplete model output: {directory}")
    data = json.loads(metrics.read_text())
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != data["model_sha256"]:
        raise RuntimeError(f"Checkpoint hash mismatch: {directory}")
    return True


def main() -> None:
    print(json.dumps(study.audit()), flush=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    configurations = list(itertools.product((f"pair{i}" for i in range(study.PAIR_COUNT)), study.SEEDS))
    total = 2 * len(configurations)
    for index, (pair, seed) in enumerate(configurations, 1):
        processes = []
        for group in ("sparse", "clustered"):
            if completed(pair, group, seed):
                continue
            log = LOGS / f"{pair}_{group}_{seed}.log"
            if log.exists():
                raise FileExistsError(f"Log exists for incomplete model: {log}")
            stream = log.open("w")
            command = [sys.executable, str(study.HERE / "rq1_structure_v1.py"), "train",
                       "--pair", pair, "--group", group, "--seed", str(seed)]
            processes.append((group, subprocess.Popen(command, stdout=stream,
                                                       stderr=subprocess.STDOUT), stream, log))
        print(f"Started {index}/{len(configurations)}: {pair} seed={seed}", flush=True)
        for group, process, stream, log in processes:
            code = process.wait()
            stream.close()
            if code or not completed(pair, group, seed):
                raise RuntimeError(f"{pair} {group} seed={seed} failed; see {log}")
        done = sum(completed(p, g, s) for p, s in configurations
                   for g in ("sparse", "clustered"))
        print(f"Completed {done}/{total} models", flush=True)
    output = study.summarize()
    (study.ROOT / "SUMMARY.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({key: value for key, value in output.items() if key != "pair_rows"}), flush=True)


if __name__ == "__main__":
    main()
