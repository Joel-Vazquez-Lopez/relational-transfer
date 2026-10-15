"""Frozen cross-form actor test to audit position shortcuts in Qwen."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

import llm_role_internal_v1 as core

HERE = Path(__file__).resolve().parent
ROOT = HERE / "llm_role_structure_v1"
PAIRS = (("Daniel", "Rachel"), ("Emma", "Peter"), ("Laura", "Jack"),
         ("Kate", "William"), ("Sam", "Mike"), ("Mary", "Alex"),
         ("Lisa", "James"), ("Robert", "Anna"))
VERBS = ("insulted", "congratulated", "rescued", "invited")
FORMS = ("active", "passive", "object_cleft", "subject_cleft")
WORDINGS = ("generic", "performed")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows() -> list[dict]:
    result = []
    for pair_index, (a, b) in enumerate(PAIRS):
        for verb in VERBS:
            for direction, (actor, patient) in enumerate(((a, b), (b, a))):
                sentences = {
                    "active": f"{actor} {verb} {patient}.",
                    "passive": f"{patient} was {verb} by {actor}.",
                    "object_cleft": f"It was {patient} that {actor} {verb}.",
                    "subject_cleft": f"It was {actor} who {verb} {patient}.",
                }
                for form in FORMS:
                    for wording in WORDINGS:
                        result.append({"pair_index": pair_index, "verb": verb,
                                       "direction": direction, "actor": actor,
                                       "patient": patient, "form": form,
                                       "wording": wording,
                                       "sentence": sentences[form],
                                       "question": ("Who did the action?" if wording == "generic"
                                                    else "Who performed the action?")})
    assert len(result) == 512
    return result


def protocol() -> dict:
    return {"date": "2026-10-10", "status": "fixed_before_cross_form_run",
            "model": "Qwen2.5-1.5B-Instruct at commit 989aa7980e4cf806f80c7fef2b1adb7bc71aa306, no weight updates",
            "model_weights_sha256": sha(core.MODEL / "model.safetensors"),
            "source_hashes": {"shared": sha(HERE / "llm_role_internal_v1.py"),
                              "this": sha(HERE / "llm_role_structure_v1.py")},
            "design": "8 new pairings x 4 verbs x 2 actor directions = 64 events; each in active, passive, object cleft, subject cleft and two fixed actor questions = 512 related prompts",
            "outcome": "greedy first token must be exactly one of the names; report actor accuracy and last-name choice for each form and wording; event is the comparison unit",
            "interpretation": "consistent actor answers when actor appears first and last across forms would be stronger evidence of role use; a last-name rule across forms would indicate a positional shortcut",
            "limitations": "constructed English, one small pretrained model, four templates; phrase-level grammar and model pretraining are uncontrolled"}


def prepare() -> None:
    ROOT.mkdir(exist_ok=True)
    path = ROOT / "PROTOCOL.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(protocol(), indent=2) + "\n")
    (ROOT / "ITEMS.json").write_text(json.dumps(rows(), indent=2) + "\n")
    print(json.dumps({"protocol": str(path), "items": 512}, indent=2))


@torch.inference_mode()
def run() -> None:
    assert json.loads((ROOT / "PROTOCOL.json").read_text()) == protocol()
    items = json.loads((ROOT / "ITEMS.json").read_text())
    assert items == rows()
    destination = ROOT / "RESULT.json"
    if destination.exists():
        raise FileExistsError(destination)
    tokenizer, model = core.load_model()
    for name in {n for pair in PAIRS for n in pair}:
        assert len(tokenizer.encode(name, add_special_tokens=False)) == 1
        assert len(tokenizer.encode(" " + name, add_special_tokens=False)) == 1
    records = []
    for start in range(0, len(items), 4):
        batch = items[start:start + 4]
        inputs = tokenizer([core.make_prompt(tokenizer, row) for row in batch],
                           padding=True, return_tensors="pt")
        logits = model(**inputs, use_cache=False).logits[:, -1]
        for row, score in zip(batch, logits):
            first = tokenizer.decode([int(score.argmax())]).strip()
            records.append({**row, "first_token": first,
                            "valid": first in (row["actor"], row["patient"]),
                            "correct": first == row["actor"],
                            "last_name_choice": first == (row["actor"] if
                                row["sentence"].rfind(row["actor"]) >
                                row["sentence"].rfind(row["patient"]) else row["patient"]),
                            "actor_margin": core.margin_from_logits(
                                score, tokenizer, row["actor"], row["patient"])})
        if (start + len(batch)) % 64 == 0:
            print(f"cross-form {start + len(batch)}/{len(items)}", flush=True)
    destination.write_text(json.dumps({"protocol_sha256": sha(ROOT / "PROTOCOL.json"),
                                       "records": records}, indent=2) + "\n")


def summarize() -> dict:
    data = json.loads((ROOT / "RESULT.json").read_text())
    assert data["protocol_sha256"] == sha(ROOT / "PROTOCOL.json")
    records = data["records"]
    cells = {}
    for form in FORMS:
        for wording in WORDINGS:
            sub = [r for r in records if r["form"] == form and r["wording"] == wording]
            cells[f"{form}/{wording}"] = {
                "n": len(sub), "correct_actor": sum(r["correct"] for r in sub),
                "valid": sum(r["valid"] for r in sub),
                "last_name_choice": sum(r["last_name_choice"] for r in sub),
                "mean_actor_margin": sum(r["actor_margin"] for r in sub) / len(sub)}
    report = {"cells": cells}
    (ROOT / "SUMMARY.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run", "summarize"))
    args = parser.parse_args()
    if args.action == "prepare": prepare()
    elif args.action == "run": run()
    else: print(json.dumps(summarize(), indent=2))


if __name__ == "__main__":
    main()
