#!/usr/bin/env python3
"""Tiny causal Transformer pilot: direct vs forward trace vs goal-first plan."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
ARMS = ("direct", "forward", "goal_first")
SEED = 41
EPOCHS = 6
BATCH = 64
LR = 0.001
DIM = 64
LAYERS = 2
HEADS = 4


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_data() -> dict:
    return json.loads((ROOT / "DATA.json").read_text())


def prompt(row: dict) -> list[str]:
    return ["<bos>", "facts"] + row["sentence"].split() + ["question"] + row["question"].split() + ["<out>"]


def target(row: dict, arm: str) -> list[str]:
    v1, v2 = row["verbs"]
    d1, d2 = row["operations"]
    b, c = row["middle"], row["answer"]
    if arm == "direct":
        body = ["answer", c]
    elif arm == "forward":
        body = ["goal", "execute", v1, d1, b, "then", v2, d2, c, "answer", c]
    else:
        body = ["goal", "need", v2, d2, "depends", v1, d1, "execute", b, c, "answer", c]
    return body + ["<eos>"]


def make_vocab(data: dict) -> dict:
    tokens = {"<pad>", "<eos>"}
    for rows in data.values():
        for row in rows:
            tokens.update(prompt(row))
            for arm in ARMS:
                tokens.update(target(row, arm))
    return {token: i for i, token in enumerate(sorted(tokens))}


class CausalLM(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.token = nn.Embedding(vocab_size, DIM)
        self.pos = nn.Embedding(160, DIM)
        layer = nn.TransformerEncoderLayer(DIM, HEADS, DIM * 4, dropout=0.0,
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, LAYERS, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(DIM)
        self.head = nn.Linear(DIM, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        n = ids.shape[1]
        x = self.token(ids) + self.pos(torch.arange(n, device=ids.device))[None, :, :]
        causal = torch.triu(torch.ones(n, n, dtype=torch.bool, device=ids.device), diagonal=1)
        x = self.blocks(x, mask=causal, src_key_padding_mask=(ids == 0))
        return self.head(self.norm(x))


def batch(rows: list[dict], arm: str, vocab: dict) -> tuple[torch.Tensor, torch.Tensor, int]:
    sequences, labels = [], []
    for row in rows:
        p, t = prompt(row), target(row, arm)
        ids = [vocab[x] for x in p + t]
        y = [-100] * len(p) + [vocab[x] for x in t]
        sequences.append(ids)
        labels.append(y)
    length = max(map(len, sequences))
    x = torch.tensor([s + [0] * (length - len(s)) for s in sequences])
    y = torch.tensor([z + [-100] * (length - len(z)) for z in labels])
    return x, y, sum(len(target(r, arm)) for r in rows)


def prepare() -> None:
    path = ROOT / "PROTOCOL.json"
    if path.exists():
        raise FileExistsError(path)
    data = read_data()
    vocab = make_vocab(data)
    (ROOT / "VOCAB.json").write_text(json.dumps(vocab, indent=2) + "\n")
    cfg = {
        "date": "2026-10-13", "status": "prospectively frozen local synthetic feasibility pilot",
        "question": "Does goal-first operation ordering improve two-hop answer transfer to unseen mixed F/R dependency compositions over direct and forward-trace training?",
        "design": "Four event facts with active/passive realization and a nested natural-language-like question. Training uses FF/RR two-hop compositions; heldout_mixed uses FR/RF. All primitives and verbs occur in training. Distinct generated prompts across splits. No natural-language or LLM generalization claim.",
        "arms": ARMS, "seed": SEED, "epochs": EPOCHS, "batch": BATCH,
        "learning_rate": LR, "dimension": DIM, "layers": LAYERS, "heads": HEADS,
        "matched": "Same generated train rows/order, architecture, initial parameter seed, optimizer, updates. Forward and goal-first both supervise operation order, intermediate entity, and final answer, but differ in output order and target-token count. Direct has fewer supervised targets. This is a feasibility pilot, not a supervision-matched causal test.",
        "primary": "Free-generated exact final answer on 400 heldout_mixed items per arm, plus in-domain test and output-plan validity. Report FF/RR and FR/RF separately, invalid outputs, target-token counts, and rule/position baselines.",
        "positive_screen": "Goal-first exceeds forward trace on heldout_mixed answer accuracy by >= 20/400 while maintaining in-domain accuracy within 20/400; and freely generated goal-first plans are valid for >=300/400 mixed items. Otherwise do not advance this formulation to a larger run. This threshold is an exploratory development gate, not a significance test.",
        "limits": "One seed, one tiny scratch-trained Transformer, symbolic names and four verbs, engineered grammar, no independent natural data. A win cannot establish generalized procedure learning, internal task states, or novelty beyond prior goal-first planning. Negative results may reflect model/data capacity or output design.",
        "benchmark_sha256": sha(ROOT / "benchmark.py"),
        "data_sha256": sha(ROOT / "DATA.json"),
        "audit_sha256": sha(ROOT / "DATA_AUDIT.json"),
        "vocab_sha256": sha(ROOT / "VOCAB.json"),
        "script_sha256": sha(Path(__file__)),
    }
    path.write_text(json.dumps(cfg, indent=2) + "\n")
    print("Frozen", len(data["train"]), "train examples, vocab", len(vocab), "protocol", sha(path))


def validate() -> tuple[dict, dict, dict]:
    cfg = json.loads((ROOT / "PROTOCOL.json").read_text())
    for key, path in (("benchmark_sha256", ROOT / "benchmark.py"),
                      ("data_sha256", ROOT / "DATA.json"),
                      ("audit_sha256", ROOT / "DATA_AUDIT.json"),
                      ("vocab_sha256", ROOT / "VOCAB.json"),
                      ("script_sha256", Path(__file__))):
        assert cfg[key] == sha(path), key
    return cfg, read_data(), json.loads((ROOT / "VOCAB.json").read_text())


def train(arm: str) -> None:
    cfg, data, vocab = validate()
    dest = ROOT / f"MODEL_{arm}.pt"
    if dest.exists():
        raise FileExistsError(dest)
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    model = CausalLM(len(vocab))
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
            ids, labels, count = batch(selected, arm, vocab)
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
        print(arm, "epoch", epoch + 1, "/", EPOCHS, "loss", round(sum(losses[-math.ceil(len(rows)/BATCH):])/math.ceil(len(rows)/BATCH), 4),
              "elapsed", round(time.perf_counter() - began), flush=True)
    torch.save(model.state_dict(), dest)
    (ROOT / f"TRAIN_{arm}.json").write_text(json.dumps({
        "arm": arm, "protocol_sha256": sha(ROOT / "PROTOCOL.json"),
        "checkpoint_sha256": sha(dest), "updates": len(losses),
        "target_tokens": target_tokens, "losses": losses,
        "elapsed_seconds": time.perf_counter() - began}, indent=2) + "\n")
    print("Saved", dest.name, flush=True)


@torch.inference_mode()
def generate(model: CausalLM, rows: list[dict], vocab: dict, arm: str) -> list[list[str]]:
    inverse = {v: k for k, v in vocab.items()}
    limit = max(len(target(r, arm)) for r in rows) + 2
    outputs: list[list[str] | None] = [None] * len(rows)
    by_length: dict[int, list[int]] = {}
    for i, row in enumerate(rows):
        by_length.setdefault(len(prompt(row)), []).append(i)
    for indices in by_length.values():
        for off in range(0, len(indices), 32):
            chosen = indices[off:off+32]
            ids = [[vocab[t] for t in prompt(rows[i])] for i in chosen]
            x = torch.tensor(ids)
            for _ in range(limit):
                next_id = model(x)[:, -1].argmax(dim=-1, keepdim=True)
                x = torch.cat([x, next_id], dim=1)
            for i, row_tokens in zip(chosen, x):
                outputs[i] = [inverse[int(t)] for t in row_tokens[len(ids[0]):]]
    assert all(x is not None for x in outputs)
    return outputs  # type: ignore[return-value]


def parse(tokens: list[str]) -> tuple[str | None, list[str]]:
    if "<eos>" in tokens:
        tokens = tokens[:tokens.index("<eos>")]
    names = [x for x in tokens if re.fullmatch(r"n\d+", x)]
    answer = None
    if "answer" in tokens:
        i = len(tokens) - 1 - tokens[::-1].index("answer")
        if i + 1 < len(tokens) and re.fullmatch(r"n\d+", tokens[i+1]):
            answer = tokens[i+1]
    return answer, tokens


@torch.inference_mode()
def evaluate(arm: str) -> None:
    cfg, data, vocab = validate()
    out = ROOT / f"EVAL_{arm}.json"
    if out.exists():
        raise FileExistsError(out)
    cp = ROOT / f"MODEL_{arm}.pt"
    train_info = json.loads((ROOT / f"TRAIN_{arm}.json").read_text())
    assert train_info["protocol_sha256"] == sha(ROOT / "PROTOCOL.json")
    assert train_info["checkpoint_sha256"] == sha(cp)
    torch.set_num_threads(4)
    model = CausalLM(len(vocab))
    model.load_state_dict(torch.load(cp, map_location="cpu", weights_only=True))
    model.eval()
    results = {}
    for split in ("development", "in_domain_test", "heldout_mixed"):
        rows = data[split]
        emitted = generate(model, rows, vocab, arm)
        records = []
        for row, tokens in zip(rows, emitted):
            answer, unpadded = parse(tokens)
            # Exact target-plan prefix, not an oracle-assisted answer.
            expected = target(row, arm)[:-2] if arm != "direct" else []
            plan = unpadded[:len(expected)]
            records.append({"id": row["id"], "pattern": row["pattern"],
                            "gold": row["answer"], "output": unpadded,
                            "answer": answer, "correct": answer == row["answer"],
                            "plan_valid": plan == expected if expected else None})
        results[split] = records
        print(arm, split, "correct", sum(r["correct"] for r in records), "/", len(rows),
              "plan", sum(r["plan_valid"] for r in records if r["plan_valid"] is not None), flush=True)
    out.write_text(json.dumps({"arm": arm, "protocol_sha256": sha(ROOT / "PROTOCOL.json"),
                               "checkpoint_sha256": sha(cp), "records": results}, indent=2) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("prepare", "train", "eval"))
    ap.add_argument("--arm", choices=ARMS)
    args = ap.parse_args()
    if args.mode == "prepare":
        prepare()
    else:
        if args.arm is None:
            ap.error("--arm is required")
        (train if args.mode == "train" else evaluate)(args.arm)
