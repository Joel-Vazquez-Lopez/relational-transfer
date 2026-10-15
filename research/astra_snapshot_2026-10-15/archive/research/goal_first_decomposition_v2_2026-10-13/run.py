#!/usr/bin/env python3
"""Matched question-operation learning with full versus masked-clause auxiliaries."""
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

ROOT = Path(__file__).resolve().parent
ARMS = ("joint_replay", "clause_decomposition")
SEEDS = (11, 23, 37, 41, 53)
EPOCHS = 10
BATCH = 128
LR = 0.001
DIM = 64
LAYERS = 2
HEADS = 4


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def data() -> dict:
    return json.loads((ROOT / "DATA.json").read_text())


def question_tokens(row: dict) -> list[str]:
    return row["question"].split()


def view(row: dict, step: int, masked: bool) -> list[str]:
    words = question_tokens(row)
    if masked:
        inner = words.index("the")
        assert words[inner:inner+2] == ["the", "person"]
        indices = set(range(inner, inner+5))
        if step == 2:
            indices = set(range(len(words))) - indices
        words = [word if i in indices else "<mask>" for i, word in enumerate(words)]
    return ["<cls>", f"<step{step}>"] + words


def vocab_from(rows: dict) -> dict:
    tokens = {"<pad>", "<mask>", "<cls>", "<step1>", "<step2>"}
    for split_rows in rows.values():
        for row in split_rows:
            tokens.update(question_tokens(row))
    return {token: i for i, token in enumerate(sorted(tokens))}


class Planner(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.token = nn.Embedding(vocab_size, DIM)
        self.pos = nn.Embedding(32, DIM)
        layer = nn.TransformerEncoderLayer(DIM, HEADS, DIM*4, dropout=0.0,
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, LAYERS, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(DIM)
        self.head = nn.Linear(DIM, 2)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        n = ids.shape[1]
        x = self.token(ids) + self.pos(torch.arange(n, device=ids.device))[None, :, :]
        x = self.blocks(x, src_key_padding_mask=(ids == 0))
        return self.head(self.norm(x[:, 0]))


def encoded(rows: list[dict], arm: str, vocab: dict, extra: bool) -> tuple[torch.Tensor, torch.Tensor]:
    texts, labels = [], []
    for row in rows:
        for step in (1, 2):
            texts.append(view(row, step, masked=(extra and arm == "clause_decomposition")))
            labels.append(0 if row["operations"][step-1] == "F" else 1)
    length = max(map(len, texts))
    ids = torch.tensor([[vocab[t] for t in text] + [vocab["<pad>"]] * (length-len(text))
                        for text in texts])
    y = torch.tensor(labels)
    return ids, y


def prepare() -> None:
    path = ROOT / "PROTOCOL.json"
    if path.exists():
        raise FileExistsError(path)
    rows = data()
    vocab = vocab_from(rows)
    (ROOT / "VOCAB.json").write_text(json.dumps(vocab, indent=2) + "\n")
    for split, split_rows in rows.items():
        for row in split_rows:
            for step in (1, 2):
                assert len(view(row, step, False)) == len(view(row, step, True))
    cfg = {
        "date": "2026-10-13", "status": "prospectively frozen targeted follow-up; new data bank, five paired seeds",
        "question": "Does training each operation with its irrelevant clause masked prevent an FF/RR correlation shortcut and improve FR/RF operation-plan composition at inference on the full question?",
        "scope": "Predict only two role-direction operations from a generated nested question. Verb recognition, participant binding, and execution are deliberately excluded. This is not full inverted CoT or an LLM repair.",
        "data": "Fresh data excludes all old exact sentence/question signatures. 6000 train FF/RR, 400 development FF/RR, 400 in-domain FF/RR, 400 mixed FR/RF. Same event grammar and name/verb vocabulary as old pilot.",
        "arms": ARMS, "seeds": SEEDS, "epochs": EPOCHS, "batch": BATCH,
        "learning_rate": LR, "dimension": DIM, "layers": LAYERS, "heads": HEADS,
        "matched": "Both arms use the same architecture, initial seed, training rows and order, optimizer, updates, and four direction labels per row. Primary views are full questions for both. Auxiliary views are a second full-question copy for joint_replay and a same-length masked-clause view for clause_decomposition. Thus input information differs by design, but labels, length, and gradient opportunities are matched.",
        "primary": "Free prediction on full unmasked questions: both operation directions correct on 400 fresh mixed items, each paired seed. Also report individual direction accuracy, in-domain both-correct, invalids (not applicable to classifier), and masked-view diagnostic. No oracle segmentation at primary inference.",
        "positive_screen": "All five seeds must reach >=360/400 in-domain both-correct for the comparison to be interpretable. Clause decomposition must exceed joint replay by >=40/400 mixed both-correct in at least four of five paired seeds, have a positive five-seed mean, and lose <=10/400 in-domain in each seed. This is a development signal; independent syntax and task-family tests still required.",
        "limits": "Masked auxiliary views are constructed from known query grammar, so the training intervention receives oracle clause boundaries. The generated grammar is narrow; success does not show general task decomposition, learnable segmentation, or internal procedure use. No entity execution is trained. Five seeds share the same test items.",
        "data_script_sha256": sha(ROOT / "data.py"),
        "data_sha256": sha(ROOT / "DATA.json"),
        "audit_sha256": sha(ROOT / "DATA_AUDIT.json"),
        "vocab_sha256": sha(ROOT / "VOCAB.json"),
        "script_sha256": sha(Path(__file__)),
    }
    path.write_text(json.dumps(cfg, indent=2) + "\n")
    print("Frozen v2", sha(path), "vocab", len(vocab))


def validate() -> tuple[dict, dict, dict]:
    cfg = json.loads((ROOT / "PROTOCOL.json").read_text())
    for key, path in (("data_script_sha256", ROOT / "data.py"),
                      ("data_sha256", ROOT / "DATA.json"),
                      ("audit_sha256", ROOT / "DATA_AUDIT.json"),
                      ("vocab_sha256", ROOT / "VOCAB.json"),
                      ("script_sha256", Path(__file__))):
        assert cfg[key] == sha(path), key
    return cfg, data(), json.loads((ROOT / "VOCAB.json").read_text())


def train(seed: int, arm: str) -> None:
    _, rows, vocab = validate()
    dest = ROOT / f"MODEL_{seed}_{arm}.pt"
    if dest.exists():
        raise FileExistsError(dest)
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    model = Planner(len(vocab))
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    train_rows = rows["train"]
    order = list(range(len(train_rows)))
    rng = random.Random(seed)
    losses, began = [], time.perf_counter()
    for epoch in range(EPOCHS):
        rng.shuffle(order)
        for off in range(0, len(order), BATCH):
            selected = [train_rows[i] for i in order[off:off+BATCH]]
            full_x, full_y = encoded(selected, arm, vocab, extra=False)
            extra_x, extra_y = encoded(selected, arm, vocab, extra=True)
            assert torch.equal(full_y, extra_y)
            assert full_x.shape == extra_x.shape
            logits = model(torch.cat([full_x, extra_x]))
            labels = torch.cat([full_y, extra_y])
            loss = F.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError((seed, arm, epoch, off))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach()))
        last = losses[-math.ceil(len(train_rows)/BATCH):]
        print(seed, arm, "epoch", epoch+1, "/", EPOCHS,
              "loss", round(sum(last)/len(last), 4),
              "elapsed", round(time.perf_counter()-began), flush=True)
    torch.save(model.state_dict(), dest)
    (ROOT / f"TRAIN_{seed}_{arm}.json").write_text(json.dumps({
        "seed": seed, "arm": arm, "protocol_sha256": sha(ROOT / "PROTOCOL.json"),
        "checkpoint_sha256": sha(dest), "updates": len(losses),
        "labels_seen": len(train_rows)*EPOCHS*4,
        "elapsed_seconds": time.perf_counter()-began, "losses": losses}, indent=2) + "\n")


@torch.inference_mode()
def evaluate(seed: int, arm: str) -> None:
    _, rows, vocab = validate()
    target = ROOT / f"EVAL_{seed}_{arm}.json"
    if target.exists():
        raise FileExistsError(target)
    cp = ROOT / f"MODEL_{seed}_{arm}.pt"
    trace = json.loads((ROOT / f"TRAIN_{seed}_{arm}.json").read_text())
    assert trace["checkpoint_sha256"] == sha(cp)
    torch.set_num_threads(4)
    model = Planner(len(vocab))
    model.load_state_dict(torch.load(cp, map_location="cpu", weights_only=True))
    model.eval()
    all_records = {}
    for split in ("development", "in_domain_test", "mixed_test"):
        results = []
        for off in range(0, len(rows[split]), BATCH):
            selected = rows[split][off:off+BATCH]
            full_x, _ = encoded(selected, arm, vocab, extra=False)
            mask_x, _ = encoded(selected, arm, vocab, extra=True)
            full_pred = model(full_x).argmax(dim=-1).tolist()
            mask_pred = model(mask_x).argmax(dim=-1).tolist()
            for i, row in enumerate(selected):
                full = ["F" if z == 0 else "R" for z in full_pred[2*i:2*i+2]]
                masked = ["F" if z == 0 else "R" for z in mask_pred[2*i:2*i+2]]
                results.append({"id": row["id"], "pattern": row["pattern"],
                                "gold": row["operations"], "pred": full,
                                "masked_pred": masked,
                                "both_correct": full == row["operations"],
                                "masked_both_correct": masked == row["operations"]})
        all_records[split] = results
        print(seed, arm, split, "full", sum(r["both_correct"] for r in results),
              "masked", sum(r["masked_both_correct"] for r in results), flush=True)
    target.write_text(json.dumps({"seed": seed, "arm": arm,
                                  "protocol_sha256": sha(ROOT / "PROTOCOL.json"),
                                  "checkpoint_sha256": sha(cp),
                                  "records": all_records}, indent=2) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("prepare", "train", "eval"))
    ap.add_argument("--seed", type=int, choices=SEEDS)
    ap.add_argument("--arm", choices=ARMS)
    args = ap.parse_args()
    if args.mode == "prepare":
        prepare()
    else:
        if args.seed is None or args.arm is None:
            ap.error("train/eval require --seed and --arm")
        (train if args.mode == "train" else evaluate)(args.seed, args.arm)
