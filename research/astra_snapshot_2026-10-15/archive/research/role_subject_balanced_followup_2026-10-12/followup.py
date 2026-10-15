#!/usr/bin/env python3
"""Prospective fixed-budget mixed-form follow-up on new test names and verbs."""
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
sys.path.insert(0, str(ROOT.parent / "role_subject_decoupling_pilot_2026-10-12"))
import experiment as previous  # noqa: E402

TEST_PAIRS = (("John", "Mary"), ("James", "Lisa"), ("Michael", "Rachel"),
              ("Philip", "Karen"), ("Joseph", "Charles"), ("George", "Edward"))
TEST_VERBS = ("admired", "invited", "comforted")
FORMS = previous.TEST_FORMS
STARTS = previous.STARTS
ARMS = ("source", "aligned", "decoupled", "mixed")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def new_rows() -> list[dict]:
    return [previous.row(e, form, role)
            for e in previous.events(TEST_PAIRS, TEST_VERBS)
            for form in FORMS for role in previous.QUESTIONS]


def mixed_forms(event_index: int, epoch: int) -> tuple[str, ...]:
    cleft = "subject_cleft" if (event_index + epoch) % 2 == 0 else "object_cleft"
    return ("active", "passive", cleft)


def prepare() -> None:
    previous.validate()
    target = ROOT / "PROTOCOL.json"
    if target.exists():
        raise FileExistsError(target)
    names = {n for pair in TEST_PAIRS for n in pair}
    prior_names = {n for pair in previous.TRAIN_PAIRS + previous.TEST_PAIRS for n in pair}
    assert len(names) == 12 and not names & prior_names
    assert not set(TEST_VERBS) & (set(previous.TRAIN_VERBS) | set(previous.TEST_VERBS))
    tok, model = previous.load("fresh1")
    for name in names:
        assert len(tok.encode(name, add_special_tokens=False)) == 1
        assert len(tok.encode(" " + name, add_special_tokens=False)) == 1
    del model
    rows = new_rows()
    assert len(rows) == 432
    checkpoints = {}
    for start in STARTS:
        checkpoints[f"{start}_source"] = sha(previous.source_path(start))
        for arm in ("aligned", "decoupled"):
            checkpoints[f"{start}_{arm}"] = sha(previous.ROOT / f"{start}_{arm}" / "adapter.pt")
    cfg = {
        "status": "prospective_followup_frozen_before_mixed_training_or_any_new_test_inference",
        "date": "2026-10-12",
        "motivation": "The fixed first test found that passive-containing training helped passive and relative-patient answers but harmed active answers. Test whether covering both clefts while retaining passive exposure avoids this trade-off under the same update and example budget.",
        "new_test_pairs": TEST_PAIRS, "new_test_verbs": TEST_VERBS,
        "forms": FORMS, "questions": previous.QUESTIONS,
        "starts": STARTS, "evaluation_arms": ARMS,
        "mixed_training": "Continue the same natural adapter. On each of the same 48 directed train events in each of two epochs, train actor and patient questions for active, passive, and exactly one cleft. Alternate subject and object cleft by event index plus epoch, so each event sees each cleft once across the two epochs. Use 96 updates and six answer targets per update, with exactly the original optimizer, LR, event order and LoRA parameter set.",
        "comparison": "Compare newly trained mixed arm with both existing same-start aligned and decoupled checkpoints on the newly frozen 36-event lexical test. No checkpoint or prompt selection using this new test.",
        "primary_endpoint": "Sum of per-form both-roles-correct event counts across the six forms, out of 216 form-events, reported for each start. This rewards broad joint correctness rather than a single form.",
        "positive_screen": "In EACH start mixed must exceed the better of aligned and decoupled by at least 6/216 form-events; active, passive and relative-patient joint counts each must be within 3/36 of the best old arm on that form; report all cells regardless. A positive remains exploratory until natural human-checked confirmation.",
        "secondary": "Per-form actor/patient and joint counts, all-six-form event correctness, candidate validity, subject/first-mentioned answer rates, and per-pair results.",
        "limits": "The mixed arm is specified after seeing the first test, but all comparisons here use disjoint new names and verbs. One Qwen0.5B family, two related source starts and constructed sentences. Simpler form coverage, not a novel offline consolidation mechanism.",
        "source_script_sha256": sha(previous.ROOT / "experiment.py"),
        "source_protocol_sha256": sha(previous.ROOT / "PROTOCOL.json"),
        "source_checkpoint_sha256": checkpoints,
        "script_sha256": sha(Path(__file__)),
        "new_test_rows_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
    }
    target.write_text(json.dumps(cfg, indent=2) + "\n")
    print("Frozen 36 new events, 432 test prompts, and mixed-form criterion", flush=True)


def validate() -> dict:
    cfg = json.loads((ROOT / "PROTOCOL.json").read_text())
    assert cfg["script_sha256"] == sha(Path(__file__))
    assert cfg["source_script_sha256"] == sha(previous.ROOT / "experiment.py")
    assert cfg["source_protocol_sha256"] == sha(previous.ROOT / "PROTOCOL.json")
    assert cfg["new_test_rows_sha256"] == hashlib.sha256(json.dumps(new_rows(), sort_keys=True).encode()).hexdigest()
    for name, expected in cfg["source_checkpoint_sha256"].items():
        start, arm = name.split("_", 1)
        path = previous.source_path(start) if arm == "source" else previous.ROOT / name / "adapter.pt"
        assert sha(path) == expected, name
    return cfg


def batch(tok, event: dict, forms: tuple[str, ...]):
    entries = []
    for form in forms:
        for role in previous.QUESTIONS:
            r = previous.row(event, form, role)
            prefix = tok.encode(previous.prompt(tok, r), add_special_tokens=False)
            answer = tok.encode(r["gold"], add_special_tokens=False)
            assert len(answer) == 1
            ids = prefix + answer + [tok.eos_token_id]
            labels = [-100] * len(prefix) + answer + [tok.eos_token_id]
            entries.append((ids, labels))
    assert len(entries) == 6
    length = max(len(ids) for ids, _ in entries)
    ids, labels, masks = [], [], []
    for tokens, target in entries:
        pad = length - len(tokens)
        ids.append([tok.pad_token_id] * pad + tokens)
        labels.append([-100] * pad + target)
        masks.append([0] * pad + [1] * len(tokens))
    return torch.tensor(ids), torch.tensor(labels), torch.tensor(masks)


def train(start: str) -> None:
    cfg = validate()
    target = ROOT / f"{start}_mixed"
    if target.exists():
        raise FileExistsError(target)
    torch.set_num_threads(2)
    tok, model = previous.load(start)
    torch.manual_seed(previous.SEED)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=previous.LR)
    events = previous.train_events()
    order = list(range(len(events)))
    rng = random.Random(previous.SEED)
    losses = []
    started = time.perf_counter()
    model.train()
    for epoch in range(previous.EPOCHS):
        rng.shuffle(order)
        for idx in order:
            ids, labels, mask = batch(tok, events[idx], mixed_forms(idx, epoch))
            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
            token_losses = F.cross_entropy(logits[:, :-1].contiguous().flatten(0, 1),
                                           labels[:, 1:].contiguous().flatten(),
                                           ignore_index=-100, reduction="none").view(6, -1)
            counts = (labels[:, 1:] != -100).sum(dim=1)
            loss = (token_losses.sum(dim=1) / counts).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError((start, epoch, idx))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            if len(losses) % 24 == 0:
                print(f"{start}/mixed {len(losses)}/96 "
                      f"loss={sum(losses[-24:])/24:.3f} elapsed={time.perf_counter()-started:.0f}s", flush=True)
    target.mkdir()
    torch.save({name: p.detach().cpu() for name, p in model.named_parameters()
                if p.requires_grad}, target / "adapter.pt")
    (target / "TRAINING.json").write_text(json.dumps({
        "protocol_sha256": sha(ROOT / "PROTOCOL.json"),
        "source_adapter_sha256": cfg["source_checkpoint_sha256"][f"{start}_source"],
        "checkpoint_sha256": sha(target / "adapter.pt"),
        "updates": len(losses), "examples": len(losses)*6,
        "elapsed_seconds": time.perf_counter()-started,
        "losses": losses,
    }, indent=2) + "\n")
    print(f"Saved {start}/mixed", flush=True)


@torch.inference_mode()
def evaluate(start: str, arm: str) -> None:
    validate()
    name = f"{start}_{arm}"
    target = ROOT / f"EVAL_{name}.json"
    if target.exists():
        raise FileExistsError(target)
    torch.set_num_threads(2)
    tok, model = previous.load(start)
    checkpoint = previous.source_path(start)
    if arm != "source":
        checkpoint = (ROOT / name / "adapter.pt") if arm == "mixed" else (previous.ROOT / name / "adapter.pt")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        assert saved.keys() == params.keys()
        for n, p in params.items():
            p.data.copy_(saved[n])
    model.eval()
    rows = new_rows()
    outputs = []
    for offset in range(0, len(rows), 8):
        block = rows[offset:offset+8]
        prompts = [previous.prompt(tok, r) for r in block]
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
            print(f"{name}: {offset+8}/{len(rows)}", flush=True)
    assert len(outputs) == 432
    target.write_text(json.dumps({"name": name, "start": start, "arm": arm,
                                  "protocol_sha256": sha(ROOT / "PROTOCOL.json"),
                                  "checkpoint_sha256": sha(checkpoint),
                                  "records": outputs}, indent=2) + "\n")
    print(f"Saved {target.name}: {sum(r['correct'] for r in outputs)}/432", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "train", "eval"))
    parser.add_argument("--start", choices=STARTS)
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare()
    elif args.start is None:
        parser.error("--start required")
    elif args.mode == "train":
        train(args.start)
    elif args.arm is None:
        parser.error("--arm required for eval")
    else:
        evaluate(args.start, args.arm)


if __name__ == "__main__":
    main()
