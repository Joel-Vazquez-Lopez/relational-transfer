"""Fixed authentic-sentence role choice evaluation on QA-SRL Gold candidates."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
MODEL = Path("/private/tmp/rolepilot_hf/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
# Screened against the source sentence and QA-SRL annotations before inference.
# Excluded duplicate predicates, unclear agent/patient spans, oblique relations,
# reciprocal/reflexive cases, and annotations where the answer is not in the text.
SELECTED = (1, 3, 4, 5, 6, 7, 8, 9, 11, 13, 15, 18, 19, 20, 21, 22,
            23, 25, 27, 28, 29, 30, 32, 33, 34, 38, 39, 43, 45, 46, 47,
            48, 50, 53, 56, 57, 58, 59, 61, 62, 66, 67, 68, 69, 70, 71,
            73, 76, 77, 80, 81, 83, 84, 85, 87, 90, 91, 92, 93, 95,
            96, 98, 99)
WORDINGS = {
    "actor": ("Which listed phrase refers to the person or group doing the target action?",
              "Which listed phrase is the actor of the target verb?"),
    "patient": ("Which listed phrase refers to the person or group affected by the target action?",
                "Which listed phrase is the patient of the target verb?"),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected() -> list[dict]:
    candidates = json.loads((ROOT / "QASRL_CANDIDATES.json").read_text())["rows"]
    rows = [candidates[i] for i in SELECTED]
    assert len(rows) == 63 and len({r["qasrl_id"] for r in rows}) == 63
    return rows


def items() -> list[dict]:
    rows = []
    for index, event in enumerate(selected()):
        for role in ("actor", "patient"):
            for wording in (0, 1):
                for order in (0, 1):
                    choices = ((event["actor"], event["patient"]) if order == 0
                               else (event["patient"], event["actor"]))
                    gold = "A" if choices[0] == event[role] else "B"
                    rows.append({"source_index": SELECTED[index], "id": event["id"],
                                 "partition": event["partition"], "verb": event["verb"],
                                 "sentence": event["sentence"], "actor": event["actor"],
                                 "patient": event["patient"], "role": role,
                                 "wording": wording, "order": order,
                                 "question": WORDINGS[role][wording],
                                 "choice_a": choices[0], "choice_b": choices[1],
                                 "gold": gold})
    assert len(rows) == 504
    return rows


def protocol() -> dict:
    return {
        "date": "2026-10-10", "status": "fixed_before_authentic_inference",
        "source": "QA-SRL Gold (plroit/qasrl-gs); 63 unique source sentences selected from 100 rule-filtered candidates by source/annotation inspection before inference",
        "source_candidate_sha256": sha(ROOT / "QASRL_CANDIDATES.json"),
        "model": "Qwen2.5-1.5B-Instruct, same fixed local snapshot, no weight updates",
        "model_weights_sha256": sha(MODEL / "model.safetensors"),
        "source_sha256": sha(ROOT / "qasrl_eval.py"),
        "design": "63 source sentences x actor/patient query x two fixed question wordings x both A/B choice orders = 504 prompts. Whole source sentences retained, with original corpus token spacing. Options reproduce annotated spans. Source is human annotated; candidate screening here was performed by an AI assistant, not independently checked by the student.",
        "scoring": "Greedy next token stripped, valid iff exactly A or B; accuracy against chosen annotated span; invalid counted wrong. Report per role, wording and option order, and complete eight-prompt event reliability.",
        "interpretation": "This tests role choice on selected authentic text, separately from constructed active/passive transfer. It cannot establish an unseen-pretraining relation or causal graph effect. Any publication claim needs independent human verification of all labels and a second model.",
    }


def prepare() -> None:
    path = ROOT / "QASRL_PROTOCOL.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(protocol(), indent=2) + "\n")
    (ROOT / "QASRL_ITEMS.json").write_text(json.dumps(items(), indent=2) + "\n")
    print(f"Prepared {len(items())} prompts from {len(selected())} sentences", flush=True)


def prompt(tokenizer, row: dict) -> str:
    content = (f"Sentence: {row['sentence']}\n"
               f"Target verb: {row['verb']}\n"
               f"Question: {row['question']}\n"
               f"A. {row['choice_a']}\nB. {row['choice_b']}")
    return tokenizer.apply_chat_template([
        {"role": "system", "content": "Answer the question by choosing exactly one option. Reply with only A or B."},
        {"role": "user", "content": content},
    ], tokenize=False, add_generation_prompt=True)


@torch.inference_mode()
def run() -> None:
    assert json.loads((ROOT / "QASRL_PROTOCOL.json").read_text()) == protocol()
    rows = json.loads((ROOT / "QASRL_ITEMS.json").read_text())
    assert rows == items()
    destination = ROOT / "QASRL_RESULT.json"
    if destination.exists():
        raise FileExistsError(destination)
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True,
                                               dtype=torch.float32).eval()
    records = []
    for start in range(0, len(rows), 4):
        batch = rows[start:start + 4]
        enc = tokenizer([prompt(tokenizer, row) for row in batch],
                        padding=True, return_tensors="pt")
        logits = model(**enc, use_cache=False).logits[:, -1]
        for row, score in zip(batch, logits):
            answer = tokenizer.decode([int(score.argmax())]).strip()
            records.append({**row, "answer": answer, "valid": answer in ("A", "B"),
                            "correct": answer == row["gold"]})
        if (start + len(batch)) % 40 == 0:
            print(f"authentic {start + len(batch)}/{len(rows)}", flush=True)
    destination.write_text(json.dumps({"protocol_sha256": sha(ROOT / "QASRL_PROTOCOL.json"),
                                       "records": records}, indent=2) + "\n")
    print(f"Saved {destination}", flush=True)


def summarize() -> None:
    obj = json.loads((ROOT / "QASRL_RESULT.json").read_text())
    assert obj["protocol_sha256"] == sha(ROOT / "QASRL_PROTOCOL.json")
    records = obj["records"]
    assert len(records) == 504
    cells = {}
    for role in ("actor", "patient"):
        for wording in (0, 1):
            for order in (0, 1):
                subset = [r for r in records if (r["role"], r["wording"], r["order"]) == (role, wording, order)]
                cells[f"{role}/{wording}/{order}"] = {"n": len(subset),
                    "correct": sum(r["correct"] for r in subset),
                    "valid": sum(r["valid"] for r in subset)}
    complete = sum(all(r["correct"] for r in records if r["id"] == event["id"])
                   for event in selected())
    report = {"n_sentences": 63, "n_prompts": 504, "cells": cells,
              "complete_eight_prompt_sentences": complete,
              "overall_correct": sum(r["correct"] for r in records),
              "overall_valid": sum(r["valid"] for r in records)}
    (ROOT / "QASRL_SUMMARY.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run", "summarize"))
    args = parser.parse_args()
    if args.action == "prepare": prepare()
    elif args.action == "run": run()
    else: summarize()


if __name__ == "__main__":
    main()
