#!/usr/bin/env python3
"""Frozen, no-training question-role and candidate scaffold diagnostic."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "natural_role_verifier_development_2026-10-13"
sys.path.insert(0, str(SOURCE))
import data as source_data  # noqa: E402
import natural_train as prior  # noqa: E402

STARTS = ("fresh1", "fresh3")
ARMS = ("role_first", "answer_only")
CONDITIONS = ("ordinary", "tagged", "no_candidates", "tagged_no_candidates")
SALT = "2026-10-13-question-role-tag-v1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def items() -> list[dict]:
    used = {item["id"] for group in ("train", "development")
            for item in source_data.split()[group]}
    pool = [item for item in source_data.extract()
            if item["id"] not in used
            and len(item["sentence"].split()) <= 20
            and len(item["actor"].split()) <= 3
            and len(item["patient"].split()) <= 3]
    pool.sort(key=lambda item: hashlib.sha256((SALT + ":" + item["id"]).encode()).hexdigest())
    chosen = pool[:40]
    assert len(chosen) == 40 and len({x["sentence_id"] for x in chosen}) == 40
    assert not {x["sentence_id"] for x in chosen} & {
        x["sentence_id"] for group in ("train", "development")
        for x in source_data.split()[group]}
    return chosen


def prompt(tok, row: dict, arm: str, condition: str) -> str:
    assert condition in CONDITIONS and arm in ARMS
    if arm == "answer_only":
        system, opening = "Give exactly: Answer: SPAN", "Answer:"
    else:
        system, opening = "Give exactly: Actor: SPAN; Patient: SPAN; Answer: SPAN", "Actor:"
    user = f"Sentence: {row['sentence']}\n"
    if condition not in ("no_candidates", "tagged_no_candidates"):
        user += f"Candidates: {', '.join(row['pair'])}\n"
    if condition in ("tagged", "tagged_no_candidates"):
        user += f"Requested role: {row['role'].upper()}\n"
    user += f"Question: {row['question']}"
    full = tok.apply_chat_template([{"role": "system", "content": system},
                                    {"role": "user", "content": user}],
                                   tokenize=False, add_generation_prompt=True) + opening
    if condition == "ordinary":
        assert full == prior.format_prompt(tok, row, arm)
    return full


def validate() -> dict:
    cfg = json.loads((ROOT / "PROTOCOL.json").read_text())
    assert cfg["script_sha256"] == sha(Path(__file__))
    assert cfg["items_sha256"] == sha(ROOT / "ITEMS.json")
    assert cfg["source_data_sha256"] == sha(SOURCE / "data.py")
    assert cfg["source_eval_sha256"] == sha(SOURCE / "natural_train.py")
    for key, value in cfg["checkpoint_sha256"].items():
        start, arm = key.split(":")
        assert value == sha(prior.checkpoint(start, arm))
    return cfg


def prepare() -> None:
    if (ROOT / "PROTOCOL.json").exists():
        raise FileExistsError(ROOT / "PROTOCOL.json")
    chosen = items()
    (ROOT / "ITEMS.json").write_text(json.dumps(chosen, indent=2) + "\n")
    cfg = {
        "date": "2026-10-13",
        "status": "prospectively frozen no-training development diagnostic; automatic QA-SRL role mapping remains provisional",
        "question": "Does an explicit requested-role tag improve direct answers from existing checkpoints, and do candidates scaffold the role fields?",
        "sample": "40 new original-train QA-SRL Bank items, disjoint from prior 160 train and 40 development sentences and excluded from QA-SRL Gold. Human QA annotation consensus, automatic actor/patient mapping; no independent human review. Deterministic SHA256 selection among remaining items with <=20 sentence tokens and <=3-token role spans.",
        "conditions": list(CONDITIONS),
        "arms": list(ARMS), "starts": list(STARTS),
        "prompts": "Native ordinary prompt; add Requested role: ACTOR/PATIENT before the unchanged original question, remove candidate line, or both. Same sentence, actor/patient questions, both candidate orders, fixed greedy decoding, 48 new tokens. No model training.",
        "primary": "For role_first with candidates, compare paired exact direct-answer counts for tagged versus ordinary on 160 prompts per start, and compare same-output routed-minus-direct gap. Report all-four-correct items, question-pair identical-answer rate, generated actor/patient field correctness, parse rates, and answer-other-field errors.",
        "secondary": "Answer-only tag effect tests generic explicit-role prompting. No-candidate conditions test reliance on supplied candidate spans but are an output/input distribution shift. Route from generated fields using metadata role is an oracle-selector diagnostic, not end-to-end QA.",
        "positive_screen": "A promising question-role interpretation signal requires tagged role_first direct accuracy >= ordinary +16/160 in each start, with routed field accuracy falling by no more than 8/160. A large residual routed-minus-direct gap under tagged prompts means question interpretation alone is insufficient. These are development decisions, not publication confirmation.",
        "limits": "This is one Qwen2.5-0.5B family, two starts, automatically mapped labels, and development data from the same source corpus. The 51 previously examined Gold items and both sealed human-review challenges are not evaluated. Explicit tags may help through prompt salience as well as role disambiguation; no-candidate performance may change because the training format included candidates.",
        "script_sha256": sha(Path(__file__)), "items_sha256": sha(ROOT / "ITEMS.json"),
        "source_data_sha256": sha(SOURCE / "data.py"),
        "source_eval_sha256": sha(SOURCE / "natural_train.py"),
        "checkpoint_sha256": {f"{start}:{arm}": sha(prior.checkpoint(start, arm))
                              for start in STARTS for arm in ARMS},
    }
    (ROOT / "PROTOCOL.json").write_text(json.dumps(cfg, indent=2) + "\n")
    print("Frozen", len(chosen), "new items and protocol", sha(ROOT / "PROTOCOL.json"))


@torch.inference_mode()
def evaluate(start: str, arm: str, condition: str) -> None:
    validate()
    target = ROOT / f"EVAL_{start}_{arm}_{condition}.json"
    if target.exists():
        raise FileExistsError(target)
    torch.set_num_threads(2)
    tok, model = prior.load_trained(start, arm)
    rows = [row for item in json.loads((ROOT / "ITEMS.json").read_text())
            for row in prior.rows(item)]
    result = []
    for off in range(0, len(rows), 4):
        block = rows[off:off+4]
        encoded = tok([prompt(tok, row, arm, condition) for row in block],
                      padding=True, return_tensors="pt", add_special_tokens=False)
        generated = model.generate(**encoded, max_new_tokens=48, do_sample=False,
                                   pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        for row, tokens in zip(block, generated[:, encoded["input_ids"].shape[1]:]):
            raw = tok.decode(tokens, skip_special_tokens=True)
            answer, actor, patient = prior.parse(raw, arm)
            result.append({"id": row["id"], "role": row["role"], "order": row["order"],
                           "gold": row["gold"], "candidate_pair": row["pair"],
                           "raw": raw, "answer": answer, "actor": actor, "patient": patient,
                           "parsed": answer is not None,
                           "correct": answer is not None and prior.norm(answer) == prior.norm(row["gold"]),
                           "actor_correct": actor is not None and prior.norm(actor) == prior.norm(row["actor"]),
                           "patient_correct": patient is not None and prior.norm(patient) == prior.norm(row["patient"]),
                           "routed_correct": (prior.norm(actor if row["role"] == "actor" else patient) == prior.norm(row["gold"]))
                                             if arm == "role_first" else None})
        if (off + 4) % 80 == 0:
            print(start, arm, condition, off + 4, "/", len(rows), flush=True)
    target.write_text(json.dumps({"start": start, "arm": arm, "condition": condition,
                                  "protocol_sha256": sha(ROOT / "PROTOCOL.json"),
                                  "checkpoint_sha256": sha(prior.checkpoint(start, arm)),
                                  "records": result}, indent=2) + "\n")
    print("Saved", target.name, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "eval"))
    parser.add_argument("--start", choices=STARTS)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--condition", choices=CONDITIONS)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare()
    else:
        if not args.start or not args.arm or not args.condition:
            parser.error("eval requires --start, --arm, and --condition")
        evaluate(args.start, args.arm, args.condition)
