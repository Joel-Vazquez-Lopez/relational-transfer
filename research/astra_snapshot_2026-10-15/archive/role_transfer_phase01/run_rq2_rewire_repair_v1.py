"""Run the frozen fresh-graph repair test without interim outcome selection."""
from __future__ import annotations

import hashlib
import itertools
import json
import subprocess
import sys

import rq2_rewire_repair_v1 as study


def completed(index: int, condition: str, arm: str, seed: int) -> bool:
    directory = study.model_path(index, condition, arm, seed)
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
    total = study.FAMILIES * len(study.SEEDS) * len(study.CONDITIONS) * len(study.ARMS)
    for batch, (index, seed) in enumerate(plan, 1):
        # Two concurrent models avoid oversubscribing the CPU; each condition's
        # ordered/unordered comparison starts from matched initialization.
        for condition in study.CONDITIONS:
            processes = []
            for arm in study.ARMS:
                if completed(index, condition, arm, seed):
                    continue
                log = logs / f"family{index}_{condition}_{arm}_seed{seed}.log"
                if log.exists():
                    raise FileExistsError(log)
                stream = log.open("w")
                command = [sys.executable, str(study.HERE / "rq2_rewire_repair_v1.py"),
                           "train", "--family", str(index), "--condition", condition,
                           "--arm", arm, "--seed", str(seed)]
                process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
                processes.append((arm, process, stream, log))
            print(f"Started pair {batch}/{len(plan)}: family={index}, seed={seed}, condition={condition}", flush=True)
            for arm, process, stream, log in processes:
                code = process.wait()
                stream.close()
                if code or not completed(index, condition, arm, seed):
                    raise RuntimeError(f"Failed family={index}, condition={condition}, arm={arm}, seed={seed}; see {log}")
            done = sum(completed(i, c, a, s) for i, s in plan
                       for c in study.CONDITIONS for a in study.ARMS)
            print(f"Completed {done}/{total} models", flush=True)
    result = study.summarize()
    (study.ROOT / "SUMMARY.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "family_rows"}, indent=2), flush=True)


if __name__ == "__main__": main()
