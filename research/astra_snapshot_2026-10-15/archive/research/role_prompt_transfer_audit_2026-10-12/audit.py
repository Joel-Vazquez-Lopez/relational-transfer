#!/usr/bin/env python3
"""Prospective prompt-transfer diagnostic on existing Qwen0.5B checkpoints.

Read-only for all source checkpoints. Never accesses the reserved human set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parents[1]
sys.path.insert(0, str(WORK))
sys.path.insert(0, str(ROOT.parent / "role_subject_decoupling_pilot_2026-10-12"))
import experiment as prev  # noqa: E402

PAIRS = (("Alice", "Bob"), ("Bruce", "Jane"), ("David", "Amy"),
         ("Arthur", "Max"), ("Frank", "Rose"), ("Martin", "June"))
VERBS = ("helped", "followed", "protected")
FORMS = ("active", "passive", "subject_cleft", "object_cleft")
QUESTIONS = {
    "actor_performed": "Who performed the action?",
    "actor_did": "Who did the action?",
    "actor_carried": "Which person carried out the action described in the sentence?",
    "patient_verb": "Who was {verb}?",
    "patient_which": "Which person was {verb}?",
}
ARMS = ("source", "aligned", "decoupled", "mixed")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows() -> list[dict]:
    return [{**event, "form": form, "sentence": prev.sentence(event, form),
             "wording": wording, "role": wording.split("_")[0],
             "question": question.format(verb=event["verb"]),
             "gold": event[wording.split("_")[0]]}
            for event in prev.events(PAIRS, VERBS) for form in FORMS
            for wording, question in QUESTIONS.items()]


def checkpoint(start: str, arm: str) -> Path:
    if arm == "source":
        return prev.source_path(start)
    base = ROOT.parent / ("role_subject_balanced_followup_2026-10-12" if arm == "mixed"
                          else "role_subject_decoupling_pilot_2026-10-12")
    return base / f"{start}_{arm}" / "adapter.pt"


def make_protocol() -> None:
    target = ROOT / "PROTOCOL.json"
    if target.exists():
        raise FileExistsError(target)
    prior2 = (("John", "Mary"), ("James", "Lisa"), ("Michael", "Rachel"),
              ("Philip", "Karen"), ("Joseph", "Charles"), ("George", "Edward"))
    prior_verbs = set(prev.TRAIN_VERBS + prev.TEST_VERBS + ("admired", "invited", "comforted"))
    names = [n for pair in PAIRS for n in pair]
    assert len(names) == len(set(names)) == 12
    assert not set(names) & {n for pair in prev.TRAIN_PAIRS + prev.TEST_PAIRS + prior2 for n in pair}
    assert not set(VERBS) & prior_verbs
    tok, model = prev.load("fresh1")
    for name in names:
        assert len(tok.encode(name, add_special_tokens=False)) == 1, name
        assert len(tok.encode(" " + name, add_special_tokens=False)) == 1, name
    del model
    items = rows()
    assert len(items) == 720
    cfg = {
        "date": "2026-10-12",
        "status": "fixed_before_any_inference_on_new_items; read_only_existing_checkpoints",
        "motivation": "Determine whether wording sensitivity persists after natural adaptation and matched construction training, and whether it interacts with sentence form. No new training in this diagnostic.",
        "pairs": PAIRS, "verbs": VERBS, "forms": FORMS, "questions": QUESTIONS,
        "starts": prev.STARTS, "arms": ARMS,
        "known_vs_new_wording": "actor_performed and patient_verb were the earlier training questions. Other three questions are new to these continuations. Their exact semantics may still differ pragmatically; report each separately.",
        "primary": "For each checkpoint and form, report event-level correct-answer agreement across all three actor wordings, and count events where all three actor answers are correct. Compare actor wording contradictions on identical events, not only marginal accuracy.",
        "secondary": "Report individual role/question accuracy, both actor and patient answers correct across all five questions per form, worst-question accuracy, invalid output count, and candidate-relative score. Report each start and arm; no checkpoint selection.",
        "limits": "Constructed English only; one Qwen family; two related starts. Question paraphrases may differ pragmatically. Earlier result motivated this new diagnostic, so it is a new lexical test but not a blind theoretical discovery. Human-review challenge remains sealed.",
        "checkpoints_sha256": {f"{s}_{a}": sha(checkpoint(s, a)) for s in prev.STARTS for a in ARMS},
        "script_sha256": sha(Path(__file__)),
        "rows_sha256": hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest(),
    }
    target.write_text(json.dumps(cfg, indent=2) + "\n")
    print("Frozen 36 new events, 720 prompts/checkpoint", flush=True)


def validate() -> dict:
    cfg = json.loads((ROOT / "PROTOCOL.json").read_text())
    assert cfg["script_sha256"] == sha(Path(__file__))
    assert cfg["rows_sha256"] == hashlib.sha256(json.dumps(rows(), sort_keys=True).encode()).hexdigest()
    for name, expected in cfg["checkpoints_sha256"].items():
        start, arm = name.split("_", 1)
        assert sha(checkpoint(start, arm)) == expected, name
    return cfg


@torch.inference_mode()
def evaluate(start: str, arm: str) -> None:
    validate()
    target = ROOT / f"EVAL_{start}_{arm}.json"
    if target.exists():
        raise FileExistsError(target)
    torch.set_num_threads(2)
    tok, model = prev.load(start)
    cp = checkpoint(start, arm)
    if arm != "source":
        saved = torch.load(cp, map_location="cpu", weights_only=True)
        params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        assert saved.keys() == params.keys()
        for n, p in params.items():
            p.data.copy_(saved[n])
    model.eval()
    items = rows()
    out = []
    for offset in range(0, len(items), 8):
        block = items[offset:offset+8]
        prompts = [prev.prompt(tok, r) for r in block]
        inputs = tok(prompts, padding=True, return_tensors="pt")
        logits = model(**inputs, use_cache=False).logits[:, -1]
        for r, z in zip(block, logits):
            a, b = r["pair"]
            aid = tok.encode(a, add_special_tokens=False)[0]
            bid = tok.encode(b, add_special_tokens=False)[0]
            answer = tok.decode([int(z.argmax())]).strip()
            margin = float((z[aid] - z[bid]).item())
            out.append({"event": r["id"], "pair_index": r["pair_index"],
                        "form": r["form"], "wording": r["wording"],
                        "role": r["role"], "gold": r["gold"],
                        "answer": answer, "correct": answer == r["gold"],
                        "valid": answer in r["pair"], "first_minus_second_margin": margin})
        if (offset + 8) % 120 == 0:
            print(f"{start}_{arm}: {offset+8}/{len(items)}", flush=True)
    assert len(out) == 720
    target.write_text(json.dumps({"start": start, "arm": arm,
                                  "protocol_sha256": sha(ROOT / "PROTOCOL.json"),
                                  "checkpoint_sha256": sha(cp),
                                  "records": out}, indent=2) + "\n")
    print(f"Saved {target.name}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "eval"))
    parser.add_argument("--start", choices=prev.STARTS)
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    if args.mode == "prepare":
        make_protocol()
    else:
        if args.start is None or args.arm is None:
            parser.error("eval needs --start and --arm")
        evaluate(args.start, args.arm)
