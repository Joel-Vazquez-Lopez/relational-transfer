#!/usr/bin/env python3
"""Matched typed-goal input conditioning on a fresh QA-SRL Bank item set."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parent
OLDER = ROOT.parent / "goal_first_natural_routing_2026-10-13"
SOURCE = ROOT.parent / "natural_role_verifier_development_2026-10-13"
sys.path.insert(0, str(SOURCE))
import natural_train as prior  # noqa: E402
import data as bank  # noqa: E402

STARTS = ("fresh1", "fresh3")
ARMS = ("typed_goal", "constant_tag")
SEED = 910127
LR = 5e-5


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def norm(text):
    return prior.norm(text)


def splits():
    return json.loads((ROOT / "SPLITS.json").read_text())


def new_items():
    older = json.loads((OLDER / "SPLITS.json").read_text())
    excluded = {item["sentence_id"] for split in older.values() for item in split}
    groups = {}
    for item in bank.extract():
        if item["sentence_id"] not in excluded:
            groups.setdefault(item["sentence_id"], []).append(item)
    keys = sorted(groups, key=lambda key: hashlib.sha256(("typed-goal-20261013:" + key).encode()).hexdigest())
    chosen = []
    for key in keys:
        if len(chosen) >= 40:
            break
        chosen.append(groups[key][0])
    assert len(chosen) >= 40
    assert len({x["sentence_id"] for x in chosen}) == 40
    return chosen


def classifier(train_items):
    questions, labels = [], []
    for item in train_items:
        questions += [item["actor_question"], item["patient_question"]]
        labels += ["actor", "patient"]
    model = make_pipeline(TfidfVectorizer(analyzer="char", ngram_range=(2, 5), lowercase=True),
                          LogisticRegression(C=1, max_iter=2000, random_state=SEED))
    model.fit(questions, labels)
    return model


def role_predictions(items, trained):
    questions = [item["actor_question"] for item in items] + [item["patient_question"] for item in items]
    predicted = trained.predict(questions).tolist()
    return {(item["id"], role): prediction
            for (item, role), prediction in zip(
                [(item, "actor") for item in items] + [(item, "patient") for item in items], predicted)}


def prompt(tok, row, arm, predicted_role=None):
    if arm == "typed_goal":
        tag = (predicted_role or row["role"]).upper()
        assert tag in ("ACTOR", "PATIENT")
    else:
        tag = "NONE"
    system = "Give exactly: Actor: SPAN; Patient: SPAN; Answer: SPAN"
    user = (f"Sentence: {row['sentence']}\nCandidates: {', '.join(row['pair'])}\n"
            f"Requested role: {tag}\nQuestion: {row['question']}")
    return tok.apply_chat_template([{"role": "system", "content": system},
                                    {"role": "user", "content": user}],
                                   tokenize=False, add_generation_prompt=True) + "Actor:"


def suffix(row):
    return f" {row['actor']}; Patient: {row['patient']}; Answer: {row['gold']}"


def batch(tok, item, arm):
    entries = []
    for row in prior.rows(item):
        p = prompt(tok, row, arm)
        pids = tok.encode(p, add_special_tokens=False)
        full = tok.encode(p + suffix(row), add_special_tokens=False)
        assert full[:len(pids)] == pids
        tokens = full + [tok.eos_token_id]
        labels = [-100] * len(pids) + full[len(pids):] + [tok.eos_token_id]
        entries.append((tokens, labels))
    width = max(len(x) for x, _ in entries)
    ids, ys, masks = [], [], []
    for tokens, labels in entries:
        pad = width - len(tokens)
        ids.append([tok.pad_token_id] * pad + tokens)
        ys.append([-100] * pad + labels)
        masks.append([0] * pad + [1] * len(tokens))
    return torch.tensor(ids), torch.tensor(ys), torch.tensor(masks)


def parse(raw):
    m = re.match(r"^\s*(.*?)\s*;\s*Patient:\s*(.*?)\s*;\s*Answer:\s*(.*?)\s*$", raw, re.S)
    return (m.group(1), m.group(2), m.group(3)) if m else (None, None, None)


def prepare():
    if (ROOT / "PROTOCOL.json").exists():
        raise FileExistsError("Protocol already frozen")
    prior.validate()
    old = json.loads((OLDER / "SPLITS.json").read_text())
    data = {"train": old["train"], "fresh2": new_items()}
    (ROOT / "SPLITS.json").write_text(json.dumps(data, indent=2) + "\n")
    tok, model = prior.prior.base.prev.load("fresh1")
    for arm in ARMS:
        batch(tok, data["train"][0], arm)
    del model
    cfg = {
        "date": "2026-10-13", "status": "adaptive follow-up design frozen before training/inference on fresh2; human-review set untouched",
        "question": "Can supplying a predicted requested-role tag during training and inference make the model's own final answers use its actor/patient fields, beyond a constant-tag control?",
        "starts": STARTS, "arms": ARMS,
        "data": "Same 160 automatically mapped QA-SRL Bank training items as prior development, and a new 40-item, 40-sentence fresh2 set disjoint from all earlier local train/development/fresh items. Source human QA labels, actor/patient conversion unreviewed.",
        "matched": "Within each start, both arms use the same source adapter, train prompts, output targets, role fields, answer labels, candidate orders, 160 updates, optimizer, and target-token count. The only input difference is an ACTOR/PATIENT requested-role tag versus a constant NONE tag. At inference typed_goal uses a fixed question-only classifier trained on the 160 training items; no gold tag is supplied.",
        "router": "Character 2-5 gram TF-IDF plus C=1 logistic regression, identical hyperparameters to earlier router. Train on 320 question-role labels from train only. No fresh2 tuning.",
        "primary": "All four final answers correct per fresh2 item (actor/patient x both candidate orders), plus exact /160, both printed role fields correct, classifier role accuracy, answer-field consistency, and missing outputs. Compare typed_goal against constant_tag for each matched start. Report older fixed role_first and answer_only checkpoints descriptively on this bank if feasible.",
        "screen": "Typed_goal must beat constant_tag by >=8/40 all-four items and >=20/160 exact answers in BOTH starts, with no more than 8/160 loss of both-correct role fields, and classifier role >=76/80. Threshold is exploratory, not a significance test.",
        "limits": "Previous fresh-bank results motivated this adaptive design, but fresh2 sentences and outputs were not inspected before freeze. Questions are templatic and a simple rule can identify role. One small model family, two related starts, one source corpus, automatically mapped semantic labels, and extra input information relative to an untagged model. A win would not prove general reasoning or unique internal mechanism. Deterministic routing remains a strong baseline and may outperform this learned readout.",
        "seed": SEED, "lr": LR, "updates_per_checkpoint": len(data["train"]),
        "splits_sha256": sha(ROOT / "SPLITS.json"),
        "script_sha256": sha(Path(__file__)),
        "old_splits_sha256": sha(OLDER / "SPLITS.json"),
        "source_adapters_sha256": {s: sha(prior.prior.base.prev.source_path(s)) for s in STARTS},
    }
    (ROOT / "PROTOCOL.json").write_text(json.dumps(cfg, indent=2) + "\n")
    print("Frozen", sha(ROOT / "PROTOCOL.json"), "fresh2", len(data["fresh2"]), flush=True)


def validate():
    cfg = json.loads((ROOT / "PROTOCOL.json").read_text())
    for key, path in (("splits_sha256", ROOT / "SPLITS.json"),
                      ("script_sha256", Path(__file__)),
                      ("old_splits_sha256", OLDER / "SPLITS.json")):
        assert cfg[key] == sha(path), key
    for s in STARTS:
        assert cfg["source_adapters_sha256"][s] == sha(prior.prior.base.prev.source_path(s))
    return cfg


def train(start, arm):
    cfg = validate()
    dest = ROOT / f"{start}_{arm}"
    if dest.exists():
        raise FileExistsError(dest)
    torch.set_num_threads(2)
    tok, model = prior.prior.base.prev.load(start)
    torch.manual_seed(SEED)
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=LR)
    items = splits()["train"]
    order = list(range(len(items)))
    random.Random(SEED).shuffle(order)
    losses, tokens_seen = [], 0
    begun = time.perf_counter()
    model.train()
    for step, idx in enumerate(order, 1):
        ids, labels, attention = batch(tok, items[idx], arm)
        opt.zero_grad(set_to_none=True)
        logits = model(input_ids=ids, attention_mask=attention).logits
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                               labels[:, 1:].reshape(-1), ignore_index=-100)
        assert torch.isfinite(loss)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
        tokens_seen += int((labels != -100).sum())
        if step % 20 == 0:
            print(start, arm, step, "/", len(order), round(losses[-1], 4), flush=True)
    dest.mkdir()
    cp = dest / "adapter.pt"
    torch.save({n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}, cp)
    (dest / "TRAINING.json").write_text(json.dumps({
        "start": start, "arm": arm, "protocol_sha256": sha(ROOT / "PROTOCOL.json"),
        "checkpoint_sha256": sha(cp), "updates": len(order), "target_tokens": tokens_seen,
        "elapsed_seconds": time.perf_counter()-begun, "losses": losses}, indent=2) + "\n")
    print("Saved", start, arm, "tokens", tokens_seen, flush=True)


@torch.inference_mode()
def evaluate(start, arm):
    validate()
    path = ROOT / f"EVAL_{start}_{arm}_fresh2.json"
    if path.exists():
        raise FileExistsError(path)
    torch.set_num_threads(2)
    tok, model = prior.prior.base.prev.load(start)
    cp = ROOT / f"{start}_{arm}" / "adapter.pt"
    weights = torch.load(cp, map_location="cpu", weights_only=True)
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    assert weights.keys() == params.keys()
    for n, p in params.items():
        p.data.copy_(weights[n])
    model.eval()
    bank_data = splits()
    role_model = classifier(bank_data["train"])
    predicted = role_predictions(bank_data["fresh2"], role_model)
    rows = [row for item in bank_data["fresh2"] for row in prior.rows(item)]
    records = []
    for off in range(0, len(rows), 4):
        block = rows[off:off+4]
        prompts = [prompt(tok, row, arm, predicted[(row["id"], row["role"])]) for row in block]
        encoded = tok(prompts, padding=True, return_tensors="pt", add_special_tokens=False)
        generated = model.generate(**encoded, max_new_tokens=48, do_sample=False,
                                   pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        for row, tokens in zip(block, generated[:, encoded["input_ids"].shape[1]:]):
            raw = tok.decode(tokens, skip_special_tokens=True)
            actor, patient, answer = parse(raw)
            field = actor if row["role"] == "actor" else patient
            records.append({"id": row["id"], "role": row["role"], "order": row["order"],
                            "gold": row["gold"], "tag": predicted[(row["id"], row["role"])],
                            "tag_correct": predicted[(row["id"], row["role"])] == row["role"],
                            "raw": raw, "actor": actor, "patient": patient, "answer": answer,
                            "parsed": answer is not None,
                            "actor_correct": actor is not None and norm(actor) == norm(row["actor"]),
                            "patient_correct": patient is not None and norm(patient) == norm(row["patient"]),
                            "correct": answer is not None and norm(answer) == norm(row["gold"]),
                            "routed_correct": field is not None and norm(field) == norm(row["gold"]),
                            "answer_matches_field": answer is not None and field is not None and norm(answer) == norm(field)})
        if (off+4) % 40 == 0:
            print(start, arm, off+4, "/", len(rows), flush=True)
    path.write_text(json.dumps({"start": start, "arm": arm,
                                "protocol_sha256": sha(ROOT / "PROTOCOL.json"),
                                "checkpoint_sha256": sha(cp), "records": records}, indent=2) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("prepare", "train", "eval"))
    ap.add_argument("--start", choices=STARTS)
    ap.add_argument("--arm", choices=ARMS)
    args = ap.parse_args()
    if args.mode == "prepare":
        prepare()
    elif args.mode == "train":
        if not args.start or not args.arm:
            ap.error("train needs --start and --arm")
        train(args.start, args.arm)
    else:
        if not args.start or not args.arm:
            ap.error("eval needs --start and --arm")
        evaluate(args.start, args.arm)
