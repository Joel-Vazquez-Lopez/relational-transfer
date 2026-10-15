#!/usr/bin/env python3
"""Adaptive capacity check after the frozen tiny pilot failed to learn event binding."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import run as first

ROOT = Path(__file__).resolve().parent
SEED = 41
EPOCHS = 24
BATCH = 64
LR = 0.0007
DIM = 128
LAYERS = 4
HEADS = 4


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_size() -> None:
    first.DIM = DIM
    first.LAYERS = LAYERS
    first.HEADS = HEADS


def prepare() -> None:
    path = ROOT / "CAPACITY_PROTOCOL.json"
    if path.exists():
        raise FileExistsError(path)
    cfg = {
        "date": "2026-10-13", "status": "adaptive exploratory capacity check after first frozen pilot failed",
        "reason": "First 2-layer 64-dimensional scratch model learned in-domain operation tokens but failed to bind the intermediate and final entities. Increase capacity and updates once using a fixed predeclared setting; no new data or prompt optimization.",
        "arms": first.ARMS, "seed": SEED, "epochs": EPOCHS, "batch": BATCH,
        "learning_rate": LR, "dimension": DIM, "layers": LAYERS, "heads": HEADS,
        "diagnostic_gate": "If both forward and goal-first in-domain exact answer accuracy remain below 280/400, the task is not learned sufficiently to interpret their held-out difference. If learned, compare mixed accuracy and freely generated plan validity descriptively; any apparent gain requires a fresh locked dataset before confirmation.",
        "limits": "Adaptive on the same data after inspecting first pilot. One seed and synthetic controlled language; no LLM or general procedure-learning claim.",
        "data_sha256": sha(ROOT / "DATA.json"), "vocab_sha256": sha(ROOT / "VOCAB.json"),
        "original_protocol_sha256": sha(ROOT / "PROTOCOL.json"),
        "original_script_sha256": sha(ROOT / "run.py"),
        "script_sha256": sha(Path(__file__)),
    }
    path.write_text(json.dumps(cfg, indent=2) + "\n")
    print("Frozen adaptive capacity check", sha(path))


def validate() -> tuple[dict, dict, dict]:
    cfg = json.loads((ROOT / "CAPACITY_PROTOCOL.json").read_text())
    for key, path in (("data_sha256", ROOT / "DATA.json"),
                      ("vocab_sha256", ROOT / "VOCAB.json"),
                      ("original_protocol_sha256", ROOT / "PROTOCOL.json"),
                      ("original_script_sha256", ROOT / "run.py"),
                      ("script_sha256", Path(__file__))):
        assert cfg[key] == sha(path), key
    model_size()
    return cfg, first.read_data(), json.loads((ROOT / "VOCAB.json").read_text())


def train(arm: str) -> None:
    _, data, vocab = validate()
    dest = ROOT / f"CAP_MODEL_{arm}.pt"
    if dest.exists():
        raise FileExistsError(dest)
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    model = first.CausalLM(len(vocab))
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    rows = data["train"]
    rng = random.Random(SEED)
    order = list(range(len(rows)))
    losses, target_tokens, began = [], 0, time.perf_counter()
    model.train()
    for epoch in range(EPOCHS):
        rng.shuffle(order)
        for off in range(0, len(order), BATCH):
            selected = [rows[i] for i in order[off:off+BATCH]]
            ids, labels, count = first.batch(selected, arm, vocab)
            target_tokens += count
            logits = model(ids)
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, len(vocab)),
                                   labels[:, 1:].reshape(-1), ignore_index=-100)
            if not torch.isfinite(loss):
                raise FloatingPointError((arm, epoch, off))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach()))
        last = losses[-math.ceil(len(rows)/BATCH):]
        print(arm, "epoch", epoch + 1, "/", EPOCHS, "loss", round(sum(last)/len(last), 4),
              "elapsed", round(time.perf_counter() - began), flush=True)
    torch.save(model.state_dict(), dest)
    (ROOT / f"CAP_TRAIN_{arm}.json").write_text(json.dumps({
        "arm": arm, "protocol_sha256": sha(ROOT / "CAPACITY_PROTOCOL.json"),
        "checkpoint_sha256": sha(dest), "updates": len(losses),
        "target_tokens": target_tokens, "losses": losses,
        "elapsed_seconds": time.perf_counter() - began}, indent=2) + "\n")
    print("Saved", dest.name, flush=True)


@torch.inference_mode()
def evaluate(arm: str) -> None:
    _, data, vocab = validate()
    out = ROOT / f"CAP_EVAL_{arm}.json"
    if out.exists():
        raise FileExistsError(out)
    cp = ROOT / f"CAP_MODEL_{arm}.pt"
    trace = json.loads((ROOT / f"CAP_TRAIN_{arm}.json").read_text())
    assert trace["checkpoint_sha256"] == sha(cp)
    torch.set_num_threads(4)
    model = first.CausalLM(len(vocab))
    model.load_state_dict(torch.load(cp, map_location="cpu", weights_only=True))
    model.eval()
    results = {}
    for split in ("development", "in_domain_test", "heldout_mixed"):
        rows = data[split]
        emitted = first.generate(model, rows, vocab, arm)
        records = []
        for row, tokens in zip(rows, emitted):
            answer, unpadded = first.parse(tokens)
            expected = first.target(row, arm)[:-2] if arm != "direct" else []
            records.append({"id": row["id"], "pattern": row["pattern"],
                            "gold": row["answer"], "output": unpadded,
                            "answer": answer, "correct": answer == row["answer"],
                            "plan_valid": unpadded[:len(expected)] == expected if expected else None})
        results[split] = records
        print(arm, split, "correct", sum(r["correct"] for r in records), "/", len(rows),
              "plan", sum(r["plan_valid"] for r in records if r["plan_valid"] is not None), flush=True)
    out.write_text(json.dumps({"arm": arm,
                               "protocol_sha256": sha(ROOT / "CAPACITY_PROTOCOL.json"),
                               "checkpoint_sha256": sha(cp), "records": results}, indent=2) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("prepare", "train", "eval"))
    ap.add_argument("--arm", choices=first.ARMS)
    args = ap.parse_args()
    if args.mode == "prepare":
        prepare()
    else:
        if args.arm is None:
            ap.error("--arm is required")
        (train if args.mode == "train" else evaluate)(args.arm)
