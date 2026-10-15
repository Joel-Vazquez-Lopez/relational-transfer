#!/usr/bin/env python3
"""Development-only exploratory comparison of consolidation schedules.

This script deliberately reads only Stage B family_00 training and development
files. It never loads confirmation files or the reserved natural-language task.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
BENCH = (ROOT / "../relational_consolidation_stage_b_2026-10-12").resolve()
sys.path.insert(0, str(BENCH))
import benchmark as b  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def tokens(sentence: str) -> list[str]:
    return re.findall(r"[A-Za-z]+|[^\w\s]", sentence)


class RoleEncoder(nn.Module):
    def __init__(self, vocab_size: int, width: int, layers: int, heads: int, ff_width: int, max_len: int):
        super().__init__()
        self.word = nn.Embedding(vocab_size, width, padding_idx=0)
        self.position = nn.Embedding(max_len, width)
        block = nn.TransformerEncoderLayer(width, heads, ff_width, dropout=0.0, batch_first=True, norm_first=False)
        self.encoder = nn.TransformerEncoder(block, layers)
        self.output = nn.Linear(width, len(b.NAMES))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(ids.shape[1], device=ids.device)[None, :]
        hidden = self.word(ids) + self.position(positions)
        hidden = self.encoder(hidden, src_key_padding_mask=(ids == 0))
        return self.output(hidden[:, 0])


def data_bundle() -> dict:
    cfg = json.loads((ROOT / "PROTOCOL.json").read_text())
    family = cfg["development_family"]
    assert family == 0, "exploratory script permits development family 0 only"
    source = BENCH / "generated" / f"family_{family:02d}"
    train_path = source / "TRAIN_EVENTS.jsonl"
    dev_path = source / "DEVELOPMENT_EVENTS.jsonl"
    s2_path = source / "S2_TRAIN_DERIVATIONS.jsonl"
    train, dev, pairs = [read_jsonl(p) for p in (train_path, dev_path, s2_path)]
    assert all(e["split"] == "train" for e in train)
    assert all(e["split"] == "development" for e in dev)
    assert {tuple(e["unordered_pair"]) for e in train}.isdisjoint({tuple(e["unordered_pair"]) for e in dev})
    assert {p["source_event_id"] for p in pairs} == {e["event_id"] for e in train}
    # Fixed vocabulary uses training and train-derived text only, never dev.
    text = [b.render(e, form)["sentence"] for e in train for form in ("symbolic", "canonical")]
    text += [p["transformed"] for p in pairs]
    vocab = {"<PAD>": 0, "<ACTOR>": 1, "<PATIENT>": 2}
    for token in sorted({t for sentence in text for t in tokens(sentence)}):
        if token not in vocab:
            vocab[token] = len(vocab)
    max_len = max(len(tokens(s)) for s in text) + 1
    def encode(s: str, role: str) -> list[int]:
        ids = [vocab["<ACTOR>" if role == "actor" else "<PATIENT>"]]
        ids += [vocab[t] for t in tokens(s)]
        return ids + [0] * (max_len - len(ids))
    ordinary = []
    for ev in train:
        for form in ("symbolic", "canonical"):
            row = b.render(ev, form)
            for role in ("actor", "patient"):
                ordinary.append((encode(row["sentence"], role), row[f"gold_{role}_id"]))
    transformed = []
    ev_by_id = {e["event_id"]: e for e in train}
    for p in pairs:
        ev = ev_by_id[p["source_event_id"]]
        form = p["transformation"]
        row = b.render(ev, form)
        for role in ("actor", "patient"):
            source_role = ("patient" if role == "actor" else "actor") if form == "role_reversal" else role
            transformed.append((encode(p["source"], source_role),
                                encode(p["transformed"], role), row[f"gold_{role}_id"]))
    dev_rows = []
    for ev in dev:
        for form in b.FORMS:
            row = b.render(ev, form)
            for role in ("actor", "patient"):
                dev_rows.append((ev["event_id"], form, role, encode(row["sentence"], role), row[f"gold_{role}_id"]))
    return {"cfg": cfg, "vocab": vocab, "max_len": max_len,
            "ordinary": ordinary, "transformed": transformed, "dev": dev_rows,
            "source_sha256": {str(p.relative_to(BENCH)): sha(p) for p in (BENCH / "PROTOCOL.json", train_path, dev_path, s2_path)}}


def batch_ordinary(pool: list, indices: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.tensor([pool[i][0] for i in indices], dtype=torch.long), torch.tensor([pool[i][1] for i in indices], dtype=torch.long)


def batch_pair(pool: list, indices: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (torch.tensor([pool[i][0] for i in indices], dtype=torch.long),
            torch.tensor([pool[i][1] for i in indices], dtype=torch.long),
            torch.tensor([pool[i][2] for i in indices], dtype=torch.long))


def draw_batches(size: int, count: int, batch_size: int, seed: int) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    return rng.integers(0, size, size=(count, batch_size)).tolist()


@torch.no_grad()
def evaluate(model: nn.Module, dev: list) -> dict:
    model.eval()
    outcomes = defaultdict(dict)
    by_form = defaultdict(lambda: [0, 0])
    for start in range(0, len(dev), 256):
        rows = dev[start:start + 256]
        ids = torch.tensor([r[3] for r in rows], dtype=torch.long)
        answers = model(ids).argmax(-1).tolist()
        for row, answer in zip(rows, answers):
            event_id, form, role, _, gold = row
            correct = answer == gold
            outcomes[event_id][(form, role)] = correct
            by_form[form][0] += int(correct)
            by_form[form][1] += 1
    joint = sum(all(flags.values()) for flags in outcomes.values())
    return {"joint_events_correct": joint, "events": len(outcomes),
            "joint_accuracy": joint / len(outcomes),
            "by_form": {k: {"correct": c, "total": n, "accuracy": c / n} for k, (c, n) in sorted(by_form.items())}}


def train_step(model: nn.Module, optim: torch.optim.Optimizer, kind: str, batch: tuple, weight: float) -> float:
    model.train()
    optim.zero_grad(set_to_none=True)
    if kind == "ordinary":
        x, y = batch
        loss = F.cross_entropy(model(x), y)
    else:
        source, transformed, y = batch
        # The source and transformed questions target the same participant ID.
        ls, lt = model(source), model(transformed)
        loss = 0.5 * (F.cross_entropy(ls, y) + F.cross_entropy(lt, y))
        if kind == "consistency":
            p = F.softmax(ls, dim=-1)
            q = F.softmax(lt, dim=-1)
            loss = loss + weight * 0.5 * (
                F.kl_div(F.log_softmax(ls, dim=-1), q.detach(), reduction="batchmean") +
                F.kl_div(F.log_softmax(lt, dim=-1), p.detach(), reduction="batchmean"))
    loss.backward()
    optim.step()
    return float(loss.detach())


def new_model(bundle: dict, seed: int) -> RoleEncoder:
    torch.manual_seed(seed)
    cfg = bundle["cfg"]["model"]
    return RoleEncoder(len(bundle["vocab"]), cfg["width"], cfg["layers"], cfg["heads"], cfg["ff_width"], bundle["max_len"])


def run(seed: int, wake_only: bool = False) -> dict:
    bundle = data_bundle()
    cfg = bundle["cfg"]
    assert seed in cfg["seeds"]
    torch.set_num_threads(4)
    out = ROOT / "runs" / f"seed_{seed}"
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing run: {out}")
    out.mkdir(parents=True)
    (out / "RUN_CONFIG.json").write_text(json.dumps({"protocol_sha256": sha(ROOT / "PROTOCOL.json"),
                                                     "runner_sha256": sha(Path(__file__)),
                                                     "source_hashes": bundle["source_sha256"],
                                                     "seed": seed, "torch_version": torch.__version__,
                                                     "threads": torch.get_num_threads()}, indent=2) + "\n")
    wake_batches = draw_batches(len(bundle["ordinary"]), cfg["wake_steps"], cfg["batch_size"], seed + 11)
    ordinary_batches = draw_batches(len(bundle["ordinary"]), cfg["ordinary_continuation_batches"] + cfg["transformation_continuation_batches"], cfg["batch_size"], seed + 29)
    pair_batches = draw_batches(len(bundle["transformed"]), cfg["transformation_continuation_batches"], cfg["batch_size"], seed + 47)
    model = new_model(bundle, seed)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg["optimizer"]["learning_rate"], weight_decay=0.0)
    began = time.monotonic()
    wake_trace = []
    for i, indices in enumerate(wake_batches):
        loss = train_step(model, optim, "ordinary", batch_ordinary(bundle["ordinary"], indices), 0.0)
        if (i + 1) % 50 == 0:
            wake_trace.append({"step": i + 1, "loss": loss, "elapsed_seconds": round(time.monotonic() - began, 2)})
    wake_eval = evaluate(model, bundle["dev"])
    torch.save(model.state_dict(), out / "WAKE_MODEL.pt")
    status = {"wake": wake_eval, "wake_trace": wake_trace,
              "wake_seconds": round(time.monotonic() - began, 2),
              "wake_model_sha256": sha(out / "WAKE_MODEL.pt"),
              "vocab_size": len(bundle["vocab"]), "ordinary_rows": len(bundle["ordinary"]),
              "transformed_rows": len(bundle["transformed"]), "development_rows": len(bundle["dev"])}
    (out / "WAKE_RESULT.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps({"seed": seed, "wake": wake_eval, "seconds": status["wake_seconds"]}), flush=True)
    screen_pass = wake_eval["by_form"]["canonical"]["accuracy"] >= 0.90 and wake_eval["joint_accuracy"] <= 0.80
    if not screen_pass:
        (out / "SCREEN_FAILED.json").write_text(json.dumps({"reason": "wake headroom screen failed",
                                                         "canonical_accuracy": wake_eval["by_form"]["canonical"]["accuracy"],
                                                         "joint_accuracy": wake_eval["joint_accuracy"]}, indent=2) + "\n")
        print("wake headroom screen failed; no continuation arms run", flush=True)
        return status
    if wake_only:
        return status
    wake_state = torch.load(out / "WAKE_MODEL.pt", map_location="cpu", weights_only=True)
    arm_results = {}
    for arm in cfg["arms"]:
        arm_model = new_model(bundle, seed)
        arm_model.load_state_dict(wake_state)
        if arm == "stop":
            arm_results[arm] = {"evaluation": evaluate(arm_model, bundle["dev"]), "steps": 0}
            continue
        arm_optim = torch.optim.AdamW(arm_model.parameters(), lr=cfg["optimizer"]["learning_rate"], weight_decay=0.0)
        sequence = []
        n, m = cfg["ordinary_continuation_batches"], cfg["transformation_continuation_batches"]
        if arm == "extra_ordinary":
            sequence = [("ordinary", i) for i in range(n + m)]
        elif arm in ("augmentation", "s2_offline"):
            sequence = [("ordinary", i) for i in range(n)] + [("pair", i) for i in range(m)]
        elif arm == "s2_interleaved":
            sequence = [x for i in range(max(n, m)) for x in (("ordinary", i), ("pair", i)) if x[1] < (n if x[0] == "ordinary" else m)]
        else:
            raise ValueError(arm)
        losses = []
        arm_start = time.monotonic()
        for kind, i in sequence:
            if kind == "ordinary":
                batch = batch_ordinary(bundle["ordinary"], ordinary_batches[i])
                loss = train_step(arm_model, arm_optim, "ordinary", batch, 0.0)
            else:
                batch = batch_pair(bundle["transformed"], pair_batches[i])
                loss = train_step(arm_model, arm_optim, "consistency" if arm.startswith("s2_") else "pair", batch, cfg["consistency_weight"])
            if (len(losses) + 1) % 50 == 0:
                print(f"seed={seed} arm={arm} step={len(losses)+1}/{len(sequence)} loss={loss:.4f}", flush=True)
            losses.append(loss)
        arm_eval = evaluate(arm_model, bundle["dev"])
        arm_dir = out / arm
        arm_dir.mkdir()
        torch.save(arm_model.state_dict(), arm_dir / "MODEL.pt")
        arm_results[arm] = {"evaluation": arm_eval, "steps": len(sequence),
                            "ordinary_batches": sum(k == "ordinary" for k, _ in sequence),
                            "pair_batches": sum(k == "pair" for k, _ in sequence),
                            "loss_first": losses[0], "loss_last": losses[-1],
                            "seconds": round(time.monotonic() - arm_start, 2),
                            "model_sha256": sha(arm_dir / "MODEL.pt")}
        print(json.dumps({"seed": seed, "arm": arm, "joint": arm_eval["joint_accuracy"],
                          "forms": {k: v["accuracy"] for k, v in arm_eval["by_form"].items()},
                          "seconds": arm_results[arm]["seconds"]}), flush=True)
    result = {"seed": seed, "wake": wake_eval, "arms": arm_results,
              "source_hashes": bundle["source_sha256"], "protocol_sha256": sha(ROOT / "PROTOCOL.json")}
    (out / "RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("wake", "full"))
    p.add_argument("--seed", type=int, required=True)
    args = p.parse_args()
    run(args.seed, wake_only=args.action == "wake")
