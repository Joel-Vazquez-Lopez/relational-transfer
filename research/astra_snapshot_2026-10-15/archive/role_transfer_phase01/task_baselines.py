"""Training-free sanity checks on the controlled four-scene task."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

import all_pair_audit as phase

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "task_baselines.json"


def evaluate() -> dict:
    if OUT.exists():
        raise FileExistsError(OUT)
    phase.base.N_ID = 10
    phase.base.VOCAB = 12
    counts = {k: [0, 0] for k in ("position_first", "event_only", "role_only", "role_and_event")}
    pairs = list(itertools.combinations(range(10), 2))
    for index, (a, b) in enumerate(pairs):
        messages, scenes, labels, categories = phase.rows_for_pair(a, b, index)
        identities = scenes[..., :10].argmax(axis=-1)
        positions = scenes[..., 10:].reshape(512, 4, 3, 4, 2)
        for i in range(len(labels)):
            src, tgt, event, _ = messages[i]
            role_ok = []
            event_ok = []
            for j in range(4):
                src_idx = int(np.where(identities[i, j] == src)[0][0])
                tgt_idx = int(np.where(identities[i, j] == tgt)[0][0])
                source = positions[i, j, src_idx]
                target = positions[i, j, tgt_idx]
                speed_src = np.linalg.norm(source[-1] - source[0])
                speed_tgt = np.linalg.norm(target[-1] - target[0])
                role_ok.append(speed_src > speed_tgt)
                d_start = np.linalg.norm(source[0] - target[0])
                d_end = np.linalg.norm(source[-1] - target[-1])
                event_ok.append((d_end < d_start) == (event == 10))
            predictions = {
                "position_first": 0,
                "event_only": next(j for j, x in enumerate(event_ok) if x),
                "role_only": next(j for j, x in enumerate(role_ok) if x),
                "role_and_event": next(j for j in range(4) if role_ok[j] and event_ok[j]),
            }
            for name, predicted in predictions.items():
                counts[name][0] += int(predicted == labels[i])
                counts[name][1] += 1
    result = {name: {"accuracy": correct / total, "correct": correct, "n": total}
              for name, (correct, total) in counts.items()}
    OUT.write_text(json.dumps({"evaluation_seed": phase.TEST_SEED, "results": result}, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    evaluate()
