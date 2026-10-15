#!/usr/bin/env python3
"""Matched natural-role training: requested role before versus after role fields."""
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

ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent / "natural_role_verifier_development_2026-10-13"
sys.path.insert(0, str(PRIOR))
import natural_train as prior  # noqa: E402
import data as source  # noqa: E402

STARTS = ("fresh1", "fresh3")
ARMS = ("goal_first", "goal_after_roles")
LR = 5e-5
SEED = 910127


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def norm(s):
    return prior.norm(s)


def load_splits():
    return json.loads((ROOT / "SPLITS.json").read_text())


def reserve_items():
    old = prior.load_split()
    exclude = {x["sentence_id"] for x in old["train"] + old["development"]}
    grouped = {}
    for item in source.extract():
        if item["sentence_id"] not in exclude:
            grouped.setdefault(item["sentence_id"], []).append(item)
    group_ids = sorted(grouped, key=lambda key: hashlib.sha256(("goal-first-natural-20261013:" + key).encode()).hexdigest())
    chosen = []
    for group in group_ids:
        if len(chosen) >= 40:
            break
        chosen.extend(grouped[group])
    assert len(chosen) >= 40
    assert not {x["sentence_id"] for x in chosen} & exclude
    return chosen[:40]


def prompt(tok, row, arm):
    if arm == "goal_first":
        instruction = "Give exactly: Need: ACTOR or PATIENT; Actor: SPAN; Patient: SPAN; Answer: SPAN"
        opening = "Need:"
    else:
        instruction = "Give exactly: Actor: SPAN; Patient: SPAN; Need: ACTOR or PATIENT; Answer: SPAN"
        opening = "Actor:"
    user = (f"Sentence: {row['sentence']}\nCandidates: {', '.join(row['pair'])}\n"
            f"Question: {row['question']}")
    return tok.apply_chat_template([{"role": "system", "content": instruction},
                                    {"role": "user", "content": user}],
                                   tokenize=False, add_generation_prompt=True) + opening


def suffix(row, arm):
    role = row["role"].upper()
    if arm == "goal_first":
        return f" {role}; Actor: {row['actor']}; Patient: {row['patient']}; Answer: {row['gold']}"
    return f" {row['actor']}; Patient: {row['patient']}; Need: {role}; Answer: {row['gold']}"


def batch(tok, item, arm):
    entries = []
    for row in prior.rows(item):
        p = prompt(tok, row, arm)
        pids = tok.encode(p, add_special_tokens=False)
        full = tok.encode(p + suffix(row, arm), add_special_tokens=False)
        assert full[:len(pids)] == pids, (arm, row["id"])
        tokens = full + [tok.eos_token_id]
        labels = [-100] * len(pids) + full[len(pids):] + [tok.eos_token_id]
        entries.append((tokens, labels))
    width = max(len(tokens) for tokens, _ in entries)
    ids, targets, masks = [], [], []
    for tokens, labels in entries:
        pad = width - len(tokens)
        ids.append([tok.pad_token_id] * pad + tokens)
        targets.append([-100] * pad + labels)
        masks.append([0] * pad + [1] * len(tokens))
    return torch.tensor(ids), torch.tensor(targets), torch.tensor(masks)


def parse(raw, arm):
    if arm == "goal_first":
        match = re.match(r"^\s*(ACTOR|PATIENT)\s*;\s*Actor:\s*(.*?)\s*;\s*Patient:\s*(.*?)\s*;\s*Answer:\s*(.*?)\s*$", raw, re.S | re.I)
        return (match.group(1).upper(), match.group(2), match.group(3), match.group(4)) if match else (None, None, None, None)
    match = re.match(r"^\s*(.*?)\s*;\s*Patient:\s*(.*?)\s*;\s*Need:\s*(ACTOR|PATIENT)\s*;\s*Answer:\s*(.*?)\s*$", raw, re.S | re.I)
    return (match.group(3).upper(), match.group(1), match.group(2), match.group(4)) if match else (None, None, None, None)


def prepare():
    if (ROOT / "PROTOCOL.json").exists():
        raise FileExistsError("Protocol already frozen")
    prior.validate()
    old = prior.load_split()
    new = reserve_items()
    splits = {"train": old["train"], "development": old["development"], "fresh": new}
    (ROOT / "SPLITS.json").write_text(json.dumps(splits, indent=2) + "\n")
    tok, model = prior.prior.base.prev.load("fresh1")
    for arm in ARMS:
        batch(tok, splits["train"][0], arm)
    del model
    cfg = {
        "date": "2026-10-13", "status": "exploratory but prospective on fresh QA-SRL Bank items; no sealed human review accessed",
        "question": "Does producing the requested semantic role before extracting actor/patient fields improve direct answers over producing the same role label after the fields?",
        "design": "Both arms receive exactly the same 160 source training items, four actor/patient by candidate-order prompts per item, all gold role, requested-role, and answer labels, same source adapter within start, one epoch, optimizer, update order, and output field vocabulary. Only the order of the generated Need field changes. Token counts may differ slightly from tokenizer context; report them.",
        "starts": STARTS, "arms": ARMS, "train_n": len(splits["train"]),
        "development_n": len(splits["development"]), "fresh_n": len(splits["fresh"]),
        "primary": "On 40 new sentence-disjoint QA-SRL Bank items, count items with all four final answers correct (actor/patient by both candidate orders). Compare goal_first minus goal_after_roles within each start. Report exact/160, requested-role accuracy, actor/patient field accuracy, output validity, and deterministic routing from fields.",
        "screen": "Goal-first must gain at least 6/40 all-four-correct fresh items in BOTH starts, while requested-role parsing is at least 95% and joint actor/patient field correctness falls by no more than 8/160 prompts relative to goal_after_roles in either start. The threshold is an exploratory go/no-go screen, not a significance test.",
        "limits": "Actor/patient labels are a deterministic mapping of human QA-SRL question/answer annotations and have not been independently human-reviewed. This is one small Qwen2.5 model family with two related starts, original-train source distribution, participant overlap, and possible source-model pretraining overlap. A positive cannot establish a general reasoning method, novel internal state, or publication-grade natural-language repair. Existing answer-only and role-first checkpoints are secondary, not supervision-matched to these new arms.",
        "lr": LR, "seed": SEED, "updates_per_checkpoint": len(splits["train"]),
        "splits_sha256": sha(ROOT / "SPLITS.json"),
        "script_sha256": sha(Path(__file__)),
        "prior_splits_sha256": sha(PRIOR / "SPLITS.json"),
        "source_adapters_sha256": {start: sha(prior.prior.base.prev.source_path(start)) for start in STARTS},
    }
    (ROOT / "PROTOCOL.json").write_text(json.dumps(cfg, indent=2) + "\n")
    print("Frozen", sha(ROOT / "PROTOCOL.json"), "fresh", len(new), flush=True)


def validate():
    cfg = json.loads((ROOT / "PROTOCOL.json").read_text())
    assert cfg["script_sha256"] == sha(Path(__file__))
    assert cfg["splits_sha256"] == sha(ROOT / "SPLITS.json")
    assert cfg["prior_splits_sha256"] == sha(PRIOR / "SPLITS.json")
    for start in STARTS:
        assert cfg["source_adapters_sha256"][start] == sha(prior.prior.base.prev.source_path(start))
    return cfg


def train(start, arm):
    cfg = validate()
    target = ROOT / f"{start}_{arm}"
    if target.exists():
        raise FileExistsError(target)
    torch.set_num_threads(2)
    tok, model = prior.prior.base.prev.load(start)
    torch.manual_seed(SEED)
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=LR)
    items = load_splits()["train"]
    order = list(range(len(items)))
    random.Random(SEED).shuffle(order)
    losses, target_tokens = [], 0
    started = time.perf_counter()
    model.train()
    for step, index in enumerate(order, 1):
        ids, labels, mask = batch(tok, items[index], arm)
        opt.zero_grad(set_to_none=True)
        logits = model(input_ids=ids, attention_mask=mask).logits
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                               labels[:, 1:].reshape(-1), ignore_index=-100)
        assert torch.isfinite(loss)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
        target_tokens += int((labels != -100).sum())
        if step % 20 == 0:
            print(start, arm, step, "/", len(order), "loss", round(losses[-1], 4), flush=True)
    target.mkdir()
    cp = target / "adapter.pt"
    torch.save({n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}, cp)
    (target / "TRAINING.json").write_text(json.dumps({
        "start": start, "arm": arm, "protocol_sha256": sha(ROOT / "PROTOCOL.json"),
        "checkpoint_sha256": sha(cp), "updates": len(order), "examples": len(order) * 4,
        "target_tokens": target_tokens, "elapsed_seconds": time.perf_counter()-started,
        "losses": losses}, indent=2) + "\n")
    print("Saved", start, arm, "tokens", target_tokens, flush=True)


@torch.inference_mode()
def evaluate(start, arm, split):
    validate()
    target = ROOT / f"EVAL_{start}_{arm}_{split}.json"
    if target.exists():
        raise FileExistsError(target)
    torch.set_num_threads(2)
    tok, model = prior.prior.base.prev.load(start)
    cp = ROOT / f"{start}_{arm}" / "adapter.pt"
    saved = torch.load(cp, map_location="cpu", weights_only=True)
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    assert saved.keys() == params.keys()
    for n, p in params.items():
        p.data.copy_(saved[n])
    model.eval()
    all_rows = [row for item in load_splits()[split] for row in prior.rows(item)]
    records = []
    for off in range(0, len(all_rows), 4):
        block = all_rows[off:off+4]
        encoded = tok([prompt(tok, row, arm) for row in block], padding=True,
                      return_tensors="pt", add_special_tokens=False)
        generated = model.generate(**encoded, max_new_tokens=56, do_sample=False,
                                   pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        for row, tokens in zip(block, generated[:, encoded["input_ids"].shape[1]:]):
            raw = tok.decode(tokens, skip_special_tokens=True)
            need, actor, patient, answer = parse(raw, arm)
            role_field = actor if row["role"] == "actor" else patient
            records.append({"id": row["id"], "role": row["role"], "order": row["order"],
                            "gold": row["gold"], "raw": raw, "need": need,
                            "actor": actor, "patient": patient, "answer": answer,
                            "parsed": answer is not None,
                            "need_correct": need == row["role"].upper(),
                            "actor_correct": actor is not None and norm(actor) == norm(row["actor"]),
                            "patient_correct": patient is not None and norm(patient) == norm(row["patient"]),
                            "correct": answer is not None and norm(answer) == norm(row["gold"]),
                            "routed_correct": role_field is not None and norm(role_field) == norm(row["gold"]),
                            "answer_matches_field": answer is not None and role_field is not None and norm(answer) == norm(role_field)})
        if (off+4) % 40 == 0:
            print(start, arm, split, off+4, "/", len(all_rows), flush=True)
    target.write_text(json.dumps({"start": start, "arm": arm, "split": split,
                                  "protocol_sha256": sha(ROOT / "PROTOCOL.json"),
                                  "checkpoint_sha256": sha(cp), "records": records}, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "train", "eval"))
    parser.add_argument("--start", choices=STARTS)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--split", choices=("development", "fresh"))
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare()
    elif args.mode == "train":
        if args.start is None or args.arm is None:
            parser.error("train needs --start and --arm")
        train(args.start, args.arm)
    else:
        if args.start is None or args.arm is None or args.split is None:
            parser.error("eval needs --start, --arm, and --split")
        evaluate(args.start, args.arm, args.split)
