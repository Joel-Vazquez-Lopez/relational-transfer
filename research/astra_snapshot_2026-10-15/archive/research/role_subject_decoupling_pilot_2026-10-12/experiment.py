#!/usr/bin/env python3
"""Matched Qwen0.5B continuation: aligned versus decoupled syntax.

This study is exploratory. It never reads the reserved human-review challenge.
The test names, verbs, forms and endpoints are frozen before training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parents[1]
sys.path.insert(0, str(WORK))
import llm_role_independent_run as existing  # noqa: E402

TRAIN_PAIRS = (("Leo", "Emma"), ("Jake", "Kate"), ("Sam", "Luke"),
               ("Anna", "Eric"), ("Paul", "Sarah"), ("Adam", "Grace"))
TEST_PAIRS = (("Henry", "Peter"), ("Simon", "Laura"), ("Kevin", "Maria"),
              ("Robert", "Brian"), ("Jacob", "Thomas"), ("Daniel", "Emily"))
TRAIN_VERBS = ("pushed", "praised", "warned", "visited")
TEST_VERBS = ("greeted", "thanked", "hugged")
TRAIN_FORMS = {
    "aligned": ("active", "subject_cleft", "object_cleft"),
    "decoupled": ("active", "passive", "object_cleft"),
}
TEST_FORMS = ("active", "passive", "subject_cleft", "object_cleft",
              "relative_actor", "relative_patient")
QUESTIONS = {"actor": "Who performed the action?", "patient": "Who was {verb}?"}
STARTS = ("fresh1", "fresh3")
EPOCHS = 2
LR = 5e-5
SEED = 910127


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_name(start: str) -> str:
    return f"{start}_natural"


def source_path(start: str) -> Path:
    return existing.CHECKPOINTS[source_name(start)][1]


def events(pairs, verbs) -> list[dict]:
    out = []
    for pair_i, pair in enumerate(pairs):
        for verb in verbs:
            for direction, (actor, patient) in enumerate((pair, pair[::-1])):
                out.append({"id": f"p{pair_i}-{verb}-d{direction}",
                            "pair_index": pair_i, "pair": list(pair),
                            "direction": direction, "actor": actor,
                            "patient": patient, "verb": verb})
    return out


def sentence(e: dict, form: str) -> str:
    a, p, v = e["actor"], e["patient"], e["verb"]
    return {
        "active": f"{a} {v} {p}.",
        "passive": f"{p} was {v} by {a}.",
        "subject_cleft": f"It was {a} who {v} {p}.",
        "object_cleft": f"It was {p} that {a} {v}.",
        "relative_actor": f"The person who {v} {p} was {a}.",
        "relative_patient": f"The person who was {v} by {a} was {p}.",
    }[form]


def row(e: dict, form: str, role: str) -> dict:
    return {**e, "form": form, "role": role, "sentence": sentence(e, form),
            "question": QUESTIONS[role].format(verb=e["verb"]), "gold": e[role]}


def train_events() -> list[dict]:
    return events(TRAIN_PAIRS, TRAIN_VERBS)


def test_rows() -> list[dict]:
    return [row(e, form, role) for e in events(TEST_PAIRS, TEST_VERBS)
            for form in TEST_FORMS for role in QUESTIONS]


def prompt(tok, r: dict) -> str:
    return tok.apply_chat_template([
        {"role": "system", "content": "Answer with exactly one of the two candidate names. Output only the name."},
        {"role": "user", "content": f"Sentence: {r['sentence']}\nCandidates: {', '.join(r['pair'])}\nQuestion: {r['question']}"},
    ], tokenize=False, add_generation_prompt=True)


def check_names(tok) -> None:
    all_names = [n for pair in TRAIN_PAIRS + TEST_PAIRS for n in pair]
    assert len(all_names) == len(set(all_names)) == 24
    for n in all_names:
        assert len(tok.encode(n, add_special_tokens=False)) == 1, n
        assert len(tok.encode(" " + n, add_special_tokens=False)) == 1, n
    assert not set(TRAIN_VERBS) & set(TEST_VERBS)


def prepare() -> None:
    target = ROOT / "PROTOCOL.json"
    if target.exists():
        raise FileExistsError(target)
    tok, model = existing.load("fresh1_natural")
    check_names(tok)
    del model
    train = train_events()
    test = test_rows()
    assert len(train) == 48 and len(test) == 432
    assert not {x["actor"] for x in train} & {x["actor"] for x in test}
    cfg = {
        "status": "fixed_before_training_and_test_inference; exploratory",
        "date": "2026-10-12",
        "hypothesis": "Balanced exposure to passive clauses where grammatical subject is patient reduces the subject-answer shortcut created by active-only natural role supervision, beyond equal-volume aligned syntax exposure.",
        "starts": list(STARTS), "arms": list(TRAIN_FORMS),
        "source_adapters_sha256": {s: sha(source_path(s)) for s in STARTS},
        "train_pairs": TRAIN_PAIRS, "test_pairs": TEST_PAIRS,
        "train_verbs": TRAIN_VERBS, "test_verbs": TEST_VERBS,
        "train_forms": TRAIN_FORMS, "test_forms": TEST_FORMS,
        "questions": QUESTIONS, "epochs": EPOCHS, "updates_per_arm": len(train) * EPOCHS,
        "examples_per_update": 6, "lr": LR, "seed": SEED,
        "matched": "Both arms continue the same natural adapter within a start, use the same 48 directed events, both role questions, six supervised targets per update, two epochs, event order, optimizer, LR and LoRA parameters. Active and object cleft are shared; aligned adds subject cleft whereas decoupled adds passive.",
        "primary_endpoint": "On the 36 events with disjoint test names and verbs, count events whose actor and patient questions are both correct in the never-trained relative_patient construction. Compare decoupled minus aligned within each start.",
        "positive_screen": "At least +6/36 relative_patient both-roles-correct events in EACH start versus aligned, no more than 4/36 fewer both-roles-correct relative_actor events in either start, and no more than 5 percentage points less active-form question accuracy. All other cells and per-pair outcomes reported regardless.",
        "secondary": "Per-form role accuracy, both-roles-correct by form, six-form joint event correctness, and subject/first-mentioned response rates. The same items also evaluate uncontinued natural source adapters.",
        "limits": "One 0.5B Qwen family, two related starts, constructed language, no human natural-sentence challenge. The arms differ in construction as well as subject/actor correlation; a positive alone does not isolate a representation or establish a general LLM repair. Relative-clause forms introduce reference resolution. These test items were specified after inspecting the preceding pilot but before any inference on them.",
        "source_code_sha256": sha(Path(__file__)),
        "train_events_sha256": hashlib.sha256(json.dumps(train, sort_keys=True).encode()).hexdigest(),
        "test_rows_sha256": hashlib.sha256(json.dumps(test, sort_keys=True).encode()).hexdigest(),
    }
    target.write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"Frozen {len(train)} train events; {len(test)} test prompts", flush=True)


def validate() -> dict:
    cfg = json.loads((ROOT / "PROTOCOL.json").read_text())
    assert cfg["source_code_sha256"] == sha(Path(__file__))
    assert cfg["train_events_sha256"] == hashlib.sha256(json.dumps(train_events(), sort_keys=True).encode()).hexdigest()
    assert cfg["test_rows_sha256"] == hashlib.sha256(json.dumps(test_rows(), sort_keys=True).encode()).hexdigest()
    for s in STARTS:
        assert cfg["source_adapters_sha256"][s] == sha(source_path(s))
    return cfg


def load(start: str):
    tok, model = existing.load(source_name(start))
    check_names(tok)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok, model


def training_batch(tok, e: dict, arm: str):
    data = []
    for form in TRAIN_FORMS[arm]:
        for role in QUESTIONS:
            r = row(e, form, role)
            prefix = tok.encode(prompt(tok, r), add_special_tokens=False)
            answer = tok.encode(r["gold"], add_special_tokens=False)
            assert len(answer) == 1
            full = prefix + answer + [tok.eos_token_id]
            labels = [-100] * len(prefix) + answer + [tok.eos_token_id]
            data.append((full, labels))
    length = max(len(ids) for ids, _ in data)
    ids, labels, masks = [], [], []
    for tokens, targets in data:
        pad = length - len(tokens)
        ids.append([tok.pad_token_id] * pad + tokens)
        labels.append([-100] * pad + targets)
        masks.append([0] * pad + [1] * len(tokens))
    return torch.tensor(ids), torch.tensor(labels), torch.tensor(masks)


def train(start: str, arm: str) -> None:
    cfg = validate()
    target = ROOT / f"{start}_{arm}"
    if target.exists():
        raise FileExistsError(target)
    torch.set_num_threads(2)
    tok, model = load(start)
    torch.manual_seed(SEED)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=LR)
    events_ = train_events()
    order = list(range(len(events_)))
    rng = random.Random(SEED)
    losses = []
    started = time.perf_counter()
    model.train()
    for epoch in range(EPOCHS):
        rng.shuffle(order)
        for idx in order:
            ids, labels, mask = training_batch(tok, events_[idx], arm)
            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
            token_losses = F.cross_entropy(logits[:, :-1].contiguous().flatten(0, 1),
                                           labels[:, 1:].contiguous().flatten(),
                                           ignore_index=-100, reduction="none").view(6, -1)
            counts = (labels[:, 1:] != -100).sum(dim=1)
            loss = (token_losses.sum(dim=1) / counts).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError((start, arm, epoch, idx))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            if len(losses) % 24 == 0:
                print(f"{start}/{arm}: {len(losses)}/{len(events_)*EPOCHS}, "
                      f"loss={sum(losses[-24:])/24:.3f}, "
                      f"elapsed={time.perf_counter()-started:.0f}s", flush=True)
    target.mkdir()
    torch.save({name: p.detach().cpu() for name, p in model.named_parameters()
                if p.requires_grad}, target / "adapter.pt")
    (target / "TRAINING.json").write_text(json.dumps({
        "protocol_sha256": sha(ROOT / "PROTOCOL.json"),
        "source_adapter_sha256": cfg["source_adapters_sha256"][start],
        "checkpoint_sha256": sha(target / "adapter.pt"),
        "start": start, "arm": arm, "updates": len(losses),
        "examples": len(losses)*6, "elapsed_seconds": time.perf_counter()-started,
        "losses": losses,
    }, indent=2) + "\n")
    print(f"Saved {start}/{arm}", flush=True)


@torch.inference_mode()
def evaluate(start: str, arm: str) -> None:
    validate()
    name = f"{start}_{arm}"
    target = ROOT / f"EVAL_{name}.json"
    if target.exists():
        raise FileExistsError(target)
    torch.set_num_threads(2)
    tok, model = load(start)
    checkpoint = None
    if arm != "source":
        checkpoint = ROOT / name / "adapter.pt"
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        assert saved.keys() == params.keys()
        for n, p in params.items():
            p.data.copy_(saved[n])
    model.eval()
    rows = test_rows()
    outputs = []
    for offset in range(0, len(rows), 8):
        block = rows[offset:offset+8]
        prompts = [prompt(tok, r) for r in block]
        inputs = tok(prompts, padding=True, return_tensors="pt")
        logits = model(**inputs, use_cache=False).logits[:, -1]
        for r, v in zip(block, logits):
            answer = tok.decode([int(v.argmax())]).strip()
            outputs.append({"id": r["id"], "pair_index": r["pair_index"],
                            "direction": r["direction"], "verb": r["verb"],
                            "form": r["form"], "role": r["role"],
                            "gold": r["gold"], "answer": answer,
                            "correct": answer == r["gold"],
                            "valid": answer in r["pair"]})
        if (offset + 8) % 96 == 0:
            print(f"{name}: {min(offset+8,len(rows))}/{len(rows)}", flush=True)
    assert len(outputs) == len(rows)
    target.write_text(json.dumps({"name": name, "start": start, "arm": arm,
                                  "protocol_sha256": sha(ROOT / "PROTOCOL.json"),
                                  "checkpoint_sha256": sha(checkpoint) if checkpoint else sha(source_path(start)),
                                  "records": outputs}, indent=2) + "\n")
    print(f"Saved {target.name}: {sum(r['correct'] for r in outputs)}/{len(outputs)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "train", "eval"))
    parser.add_argument("--start", choices=STARTS)
    parser.add_argument("--arm", choices=("source", *TRAIN_FORMS))
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare()
        return
    if args.start is None or args.arm is None:
        parser.error("--start and --arm required")
    if args.mode == "train":
        if args.arm == "source":
            parser.error("cannot train source")
        train(args.start, args.arm)
    else:
        evaluate(args.start, args.arm)


if __name__ == "__main__":
    main()
