"""Run frozen matched-rewiring models without interim outcome inspection."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys

import rq1_rewire_v1 as study


def completed(index: int, arm: str, seed: int) -> bool:
    directory = study.model_path(index, arm, seed)
    metrics, checkpoint = directory / "metrics.json", directory / "model.pt"
    if not metrics.exists() and not checkpoint.exists():
        return False
    if not metrics.exists() or not checkpoint.exists():
        raise RuntimeError(f"Partial model output: {directory}")
    data = json.loads(metrics.read_text())
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != data["model_sha256"]:
        raise RuntimeError(f"Checkpoint hash mismatch: {directory}")
    return True


def main() -> None:
    print(json.dumps(study.audit()), flush=True)
    logs = study.ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    plan = list(itertools.product(range(study.FAMILIES), study.SEEDS))
    for batch, (index, seed) in enumerate(plan, 1):
        processes = []
        for arm in study.ARMS:
            if completed(index, arm, seed):
                continue
            log = logs / f"family{index}_{arm}_seed{seed}.log"
            if log.exists():
                raise FileExistsError(log)
            stream = log.open("w")
            command = [sys.executable, str(study.HERE / "rq1_rewire_v1.py"), "train",
                       "--family", str(index), "--arm", arm, "--seed", str(seed)]
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            processes.append((arm, process, stream, log))
        print(f"Started matched triple {batch}/{len(plan)}: family={index}, seed={seed}", flush=True)
        for arm, process, stream, log in processes:
            code = process.wait()
            stream.close()
            if code or not completed(index, arm, seed):
                raise RuntimeError(f"Failed family={index}, arm={arm}, seed={seed}; see {log}")
        print(f"Completed {batch * len(study.ARMS)}/{study.FAMILIES * len(study.SEEDS) * len(study.ARMS)} models", flush=True)
    result = study.summarize()
    (study.ROOT / "SUMMARY.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "family_rows"}, indent=2), flush=True)


if __name__ == "__main__": main()
