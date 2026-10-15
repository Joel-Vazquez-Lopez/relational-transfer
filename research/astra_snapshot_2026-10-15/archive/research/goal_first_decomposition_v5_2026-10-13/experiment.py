#!/usr/bin/env python3
"""Learn clause selection and test a frozen v4 operation model on fresh items."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
V4 = ROOT.parent / "goal_first_decomposition_v4_2026-10-13"
PILOT = ROOT.parent / "goal_first_decomposition_pilot_2026-10-13"
V2 = ROOT.parent / "goal_first_decomposition_v2_2026-10-13"
V3 = ROOT.parent / "goal_first_decomposition_v3_2026-10-13"
sys.path.insert(0, str(PILOT))
sys.path.insert(0, str(V4))
import benchmark  # noqa: E402
from run import Planner, view, question_tokens  # noqa: E402

SEEDS = (11, 23, 37, 41, 53)
LENGTH = 14
BATCH = 128
EPOCHS = 10


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path):
    return json.loads(path.read_text())


def fresh_bank():
    previous = set()
    for folder in (PILOT, V2, V3, V4):
        bank = load(folder / "DATA.json")
        previous.update(benchmark.signature(row) for rows in bank.values() for row in rows)
    initial = len(previous)
    out = {}
    for split, patterns, seed in (("in_domain", ("FF", "RR"), 2081),
                                  ("mixed", ("FR", "RF"), 2082)):
        rng = random.Random(seed)
        rows = []
        while len(rows) < 400:
            row = benchmark.generate_one(rng, patterns[len(rows) % 2])
            signature = benchmark.signature(row)
            if signature in previous:
                continue
            previous.add(signature)
            assert benchmark.solve(row) == row["answer"]
            row["id"] = f"v5-{split}-{len(rows):04d}"
            rows.append(row)
        out[split] = rows
    assert len(previous) == initial + 800
    return out, initial


def gold_keep(row, step):
    words = question_tokens(row)
    inner = words.index("the")
    assert words[inner:inner+2] == ["the", "person"]
    return [inner <= i < inner+5 if step == 1 else not inner <= i < inner+5
            for i in range(len(words))]


def compressed(words, keep):
    out = []
    for token, retained in zip(words, keep):
        if retained:
            out.append(token)
        elif not out or out[-1] != "<other>":
            out.append("<other>")
    return out


def prepare():
    if (ROOT / "PROTOCOL.json").exists():
        raise FileExistsError("Protocol already frozen")
    bank, excluded = fresh_bank()
    (ROOT / "DATA.json").write_text(json.dumps(bank, indent=2) + "\n")
    for split_rows in bank.values():
        for row in split_rows:
            for step in (1, 2):
                assert ["<cls>", f"<step{step}>"] + compressed(question_tokens(row), gold_keep(row, step)) == view(row, step, True)
    protocol = {
        "date": "2026-10-13",
        "status": "exploratory test designed after v4; fresh generated examples and five selector seeds",
        "question": "Can a supervised token selector learn to derive the clause views from intact questions, then transfer its predicted views to a frozen local-operation model on unseen mixed operation pairs?",
        "data": "Train selectors on the 6000 v4 FF/RR questions. Test on 400 new FF/RR and 400 new FR/RF complete sentence/question pairs, excluding all earlier exact pair signatures. Mixed compositional patterns were unseen in selector and planner training.",
        "model": "2-layer 64-dimensional 4-head Transformer token classifier trained to keep/drop words for each requested step; no gold clause mask at test. Frozen v4 clause_decomposition planner checkpoints provide operation decisions.",
        "controls": "Record selector exact-mask accuracy, operation accuracy with predicted masks, oracle masks, and the planner's original full-question prediction. The same seed is used for selector and frozen planner, but this is not a fresh planner training comparison.",
        "positive_screen": "In at least four of five starts, exact masks >=380/400 for both unseen mixed operation pairs jointly and both operation choices >=380/400 with predicted masks. Full-question planner results remain reported as a control.",
        "limits": "The selector gets explicit gold keep/drop supervision derived from the generator. The syntax has only two templates, and exact question strings may recur because only full sentence/question pairs are excluded. Passing cannot show that a model invents decompositions, works on natural language, or has a general planning ability.",
        "seeds": SEEDS, "selector_epochs": EPOCHS, "selector_batch": BATCH,
        "prior_signatures_excluded": excluded,
        "padding_token_id": load(V4 / "VOCAB.json")["<pad>"],
        "data_sha256": sha(ROOT / "DATA.json"),
        "v4_protocol_sha256": sha(V4 / "PROTOCOL.json"),
        "v4_data_sha256": sha(V4 / "DATA.json"),
        "v4_vocab_sha256": sha(V4 / "VOCAB.json"),
        "script_sha256": sha(Path(__file__)),
    }
    (ROOT / "PROTOCOL.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print("Frozen v5", sha(ROOT / "PROTOCOL.json"), "excluded", excluded)


class Selector(nn.Module):
    def __init__(self, vocab_size, pad_id):
        super().__init__()
        self.pad_id = pad_id
        self.token = nn.Embedding(vocab_size, 64)
        self.pos = nn.Embedding(32, 64)
        layer = nn.TransformerEncoderLayer(64, 4, 256, dropout=0, batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(64)
        self.head = nn.Linear(64, 2)

    def forward(self, ids):
        length = ids.shape[1]
        x = self.token(ids) + self.pos(torch.arange(length, device=ids.device))[None, :, :]
        x = self.blocks(x, src_key_padding_mask=(ids == self.pad_id))
        return self.head(self.norm(x))


def examples(rows, vocab):
    xs, ys, metas = [], [], []
    for row in rows:
        for step in (1, 2):
            tokens = view(row, step, False)
            assert len(tokens) <= LENGTH
            xs.append([vocab[t] for t in tokens] + [vocab["<pad>"]] * (LENGTH-len(tokens)))
            ys.append([-100, -100] + [int(x) for x in gold_keep(row, step)] + [-100] * (LENGTH-len(tokens)))
            metas.append((row, step))
    return torch.tensor(xs), torch.tensor(ys), metas


def predict_planner(model, token_views, vocab):
    ids = torch.tensor([[vocab[t] for t in toks] + [vocab["<pad>"]] * (LENGTH-len(toks)) for toks in token_views])
    with torch.inference_mode():
        result = model(ids).argmax(-1).tolist()
    return ["F" if x == 0 else "R" for x in result]


def run(seed):
    protocol = load(ROOT / "PROTOCOL.json")
    assert protocol["script_sha256"] == sha(Path(__file__))
    for key, path in (("data_sha256", ROOT / "DATA.json"),
                      ("v4_protocol_sha256", V4 / "PROTOCOL.json"),
                      ("v4_data_sha256", V4 / "DATA.json"),
                      ("v4_vocab_sha256", V4 / "VOCAB.json")):
        assert protocol[key] == sha(path), key
    assert seed in SEEDS
    target = ROOT / f"EVAL_{seed}.json"
    if target.exists():
        raise FileExistsError(target)
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    vocab = load(V4 / "VOCAB.json")
    selector = Selector(len(vocab), vocab["<pad>"])
    opt = torch.optim.AdamW(selector.parameters(), lr=0.001)
    train_x, train_y, _ = examples(load(V4 / "DATA.json")["train"], vocab)
    order = list(range(len(train_x)))
    rng = random.Random(seed)
    began = time.perf_counter()
    for epoch in range(EPOCHS):
        selector.train()
        rng.shuffle(order)
        losses = []
        for off in range(0, len(order), BATCH):
            idx = order[off:off+BATCH]
            logits = selector(train_x[idx])
            loss = F.cross_entropy(logits.flatten(0, 1), train_y[idx].flatten())
            assert torch.isfinite(loss)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(selector.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach()))
        print(seed, "selector epoch", epoch+1, "loss", round(sum(losses)/len(losses), 5), flush=True)
    selector.eval()
    planner = Planner(len(vocab), vocab["<pad>"])
    checkpoint = V4 / f"MODEL_{seed}_clause_decomposition.pt"
    planner.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    planner.eval()
    all_records = {}
    for split, rows in load(ROOT / "DATA.json").items():
        records = []
        for off in range(0, len(rows), BATCH):
            selected = rows[off:off+BATCH]
            xs, ys, metas = examples(selected, vocab)
            with torch.inference_mode():
                keep_predictions = selector(xs).argmax(-1).tolist()
            learned_views, oracle_views, full_views = [], [], []
            exact_masks = []
            for (row, step), tags, gold in zip(metas, keep_predictions, ys.tolist()):
                words = question_tokens(row)
                mask = [bool(t) for t in tags[2:2+len(words)]]
                exact_masks.append(tags[2:2+len(words)] == gold[2:2+len(words)])
                learned_views.append(["<cls>", f"<step{step}>"] + compressed(words, mask))
                oracle_views.append(view(row, step, True))
                full_views.append(view(row, step, False))
            assert all(len(x) <= LENGTH for x in learned_views)
            learned = predict_planner(planner, learned_views, vocab)
            oracle = predict_planner(planner, oracle_views, vocab)
            full = predict_planner(planner, full_views, vocab)
            for i, row in enumerate(selected):
                records.append({"id": row["id"], "pattern": row["pattern"], "gold": row["operations"],
                                "selector_exact": exact_masks[2*i:2*i+2],
                                "predicted_mask_operations": learned[2*i:2*i+2],
                                "oracle_mask_operations": oracle[2*i:2*i+2],
                                "full_question_operations": full[2*i:2*i+2]})
        all_records[split] = records
        print(seed, split,
              "selector both", sum(all(r["selector_exact"]) for r in records),
              "learned both", sum(r["predicted_mask_operations"] == r["gold"] for r in records),
              "oracle both", sum(r["oracle_mask_operations"] == r["gold"] for r in records),
              "full both", sum(r["full_question_operations"] == r["gold"] for r in records), flush=True)
    target.write_text(json.dumps({"seed": seed, "protocol_sha256": sha(ROOT / "PROTOCOL.json"),
                                  "planner_checkpoint_sha256": sha(checkpoint),
                                  "elapsed_seconds": time.perf_counter()-began,
                                  "records": all_records}, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare()
    else:
        if args.seed is None:
            parser.error("run needs --seed")
        run(args.seed)
