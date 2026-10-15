#!/usr/bin/env python3
"""Prospective QA-SRL Gold direct-object probe for existing role-output adapters."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(ROOT.parent / "role_slot_routing_fresh_confirmation_2026-10-13"))
import confirm as prior  # noqa: E402

GOLD_ROOT = PROJECT / "llm_role_confirmation_v1" / "qasrl-gs" / "data" / "gold"
CANDIDATES = PROJECT / "llm_role_confirmation_v1" / "QASRL_CANDIDATES.json"
PARTS = ("wikinews.dev", "wikinews.test", "wikipedia.dev", "wikipedia.test")
ARMS = ("answer_first", "candidate_first", "slot_first")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(value):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", value.casefold())).strip()


def select_items():
    candidates = json.loads(CANDIDATES.read_text())["rows"]
    annotations = {}
    for part in PARTS:
        rows = list(csv.DictReader((GOLD_ROOT / f"{part}.gold.csv").open()))
        index = {}
        for row in rows:
            index.setdefault((row["qasrl_id"], int(row["verb_idx"])), []).append(row)
        annotations[part] = index
    selected = []
    for item in candidates:
        data = annotations[item["partition"]].get((item["qasrl_id"], item["verb_idx"]), ())
        actor_rows = [row for row in data if row["answer"] == item["actor"] and
                      row["question"] == item["actor_question"] and not row["prep"] and
                      row["obj"] == "someone"]
        patient_rows = [row for row in data if row["answer"] == item["patient"] and
                        row["question"] == item["patient_question"] and not row["prep"] and
                        not row["obj2"] and row["subj"] == "someone"]
        if actor_rows and patient_rows:
            selected.append(item)
    assert len(selected) == 51 and len({r["sentence"] for r in selected}) == 49
    return selected


def test_rows(items):
    rows = []
    for item in items:
        for role, question in (("actor", item["actor_question"]),
                               ("patient", item["patient_question"])):
            for order in (0, 1):
                pair = [item["actor"], item["patient"]]
                if order:
                    pair.reverse()
                rows.append({"id": item["id"], "sentence": item["sentence"],
                             "partition": item["partition"], "verb": item["verb"],
                             "pair": pair, "role": role, "order": order,
                             "question": question, "gold": item[role],
                             "actor": item["actor"], "patient": item["patient"]})
    assert len(rows) == 204
    return rows


def prompt(tok, row, arm):
    return prior.prompt(tok, row, arm)


def parse(raw, arm):
    if arm == "answer_first":
        match = re.match(r"^\s*(.*?)\s*;\s*Actor:\s*(.*?)\s*;\s*Patient:\s*(.*?)(?:\s*$)", raw, re.S)
        return (match.group(1), match.group(2), match.group(3)) if match else (None, None, None)
    if arm == "slot_first":
        match = re.match(r"^\s*(.*?)\s*;\s*Patient:\s*(.*?)\s*;\s*Answer:\s*(.*?)(?:\s*$)", raw, re.S)
        return (match.group(3), match.group(1), match.group(2)) if match else (None, None, None)
    match = re.match(r"^\s*(.*?)\s*;\s*Second:\s*(.*?)\s*;\s*Answer:\s*(.*?)(?:\s*$)", raw, re.S)
    return (match.group(3), None, None) if match else (None, None, None)


def prepare():
    target = ROOT / "PROTOCOL.json"
    if target.exists():
        raise FileExistsError(target)
    prior.validate()
    items = select_items()
    rows = test_rows(items)
    config = {
        "date": "2026-10-13",
        "status": "frozen before any inference on QA-SRL Gold with these six checkpoints",
        "source": "QA-SRL Gold, as downloaded in llm_role_confirmation_v1/qasrl-gs; human-annotated QA spans but automatic direct-object role mapping, not independently human-reviewed by the student",
        "selection": "All 51 rows from the prior 100 rule-filtered QA-SRL candidates where original gold metadata has an actor question with obj=someone and no preposition, and a patient question with subj=someone, no preposition, and no second object. These come from 49 unique source sentences. No item chosen using these checkpoints' predictions.",
        "prompts": "51 annotated predicates × actor/patient original gold questions × two candidate orders = 204 prompts/checkpoint. Original sentence text and candidate answer spans retained. Native trained output format requested in each arm; greedy decoding, max 48 new tokens for multiword spans.",
        "arms": list(ARMS), "starts": list(prior.STARTS),
        "primary": "Number of 51 predicates with exact gold actor and patient span on both candidate orders (four answers all correct). Normalize case, punctuation, and whitespace only. Every parse failure counts wrong.",
        "secondary": "Per-role, per-order exact span accuracy; parsed-output rate; slot-field accuracy; each source partition. Report all cells regardless of outcome.",
        "positive_screen": "For a provisional natural-data transfer signal, slot-first improves primary by at least 6/51 over BOTH controls in EACH start; parsed-output rate must be at least 80% in each slot-first start. This is a screening rule, not publication validation.",
        "limitations": "Question-answer pairs are human annotated, but the conversion to actor/patient is automatic and some predicates may not have canonical agent/patient semantics; candidate spans vary in length, 51 items are small and not all independent; output formats were trained on short synthetic names; natural QA-SRL may have appeared during base pretraining; no human review or second model family. Existing 63-item Qwen1.5B evaluation saw some of the same source sentences, but these six Qwen0.5B checkpoints have not been evaluated on this benchmark.",
        "source_file_sha256": {part: sha(GOLD_ROOT / f"{part}.gold.csv") for part in PARTS},
        "candidate_file_sha256": sha(CANDIDATES),
        "script_sha256": sha(Path(__file__)),
        "items_sha256": hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest(),
        "rows_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
        "checkpoint_sha256": {f"{s}_{a}": sha(prior.checkpoint(s, a)) for s in prior.STARTS for a in ARMS},
    }
    (ROOT / "ITEMS.json").write_text(json.dumps(items, indent=2) + "\n")
    target.write_text(json.dumps(config, indent=2) + "\n")
    print("Frozen", len(items), "items,", len(rows), "prompts/checkpoint", flush=True)


def validate():
    cfg = json.loads((ROOT / "PROTOCOL.json").read_text())
    prior.validate()
    assert cfg["script_sha256"] == sha(Path(__file__))
    assert cfg["candidate_file_sha256"] == sha(CANDIDATES)
    for part in PARTS:
        assert cfg["source_file_sha256"][part] == sha(GOLD_ROOT / f"{part}.gold.csv")
    items = select_items()
    rows = test_rows(items)
    assert cfg["items_sha256"] == hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest()
    assert cfg["rows_sha256"] == hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    for key, digest in cfg["checkpoint_sha256"].items():
        start, arm = key.split("_", 1)
        assert digest == sha(prior.checkpoint(start, arm))
    return cfg, rows


@torch.inference_mode()
def evaluate(start, arm):
    cfg, rows = validate()
    target = ROOT / f"EVAL_{start}_{arm}.json"
    if target.exists():
        raise FileExistsError(target)
    torch.set_num_threads(2)
    tok, model = prior.base.prev.load(start)
    saved = torch.load(prior.checkpoint(start, arm), map_location="cpu", weights_only=True)
    params = {name: param for name, param in model.named_parameters() if param.requires_grad}
    assert saved.keys() == params.keys()
    for name, param in params.items():
        param.data.copy_(saved[name])
    model.eval()
    out = []
    for off in range(0, len(rows), 4):
        block = rows[off:off+4]
        encoded = tok([prompt(tok, row, arm) for row in block], padding=True,
                      return_tensors="pt", add_special_tokens=False)
        generated = model.generate(**encoded, max_new_tokens=48, do_sample=False,
                                   pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        for row, tokens in zip(block, generated[:, encoded["input_ids"].shape[1]:]):
            raw = tok.decode(tokens, skip_special_tokens=True)
            answer, actor, patient = parse(raw, arm)
            out.append({"id": row["id"], "partition": row["partition"],
                        "role": row["role"], "order": row["order"],
                        "gold": row["gold"], "raw": raw, "answer": answer,
                        "actor": actor, "patient": patient,
                        "parsed": answer is not None,
                        "correct": answer is not None and normalize(answer) == normalize(row["gold"]),
                        "slot_joint_correct": actor is not None and patient is not None and
                                              normalize(actor) == normalize(row["actor"]) and
                                              normalize(patient) == normalize(row["patient"]),
                        "answer_matches_generated_slot": answer is not None and
                            normalize(answer) == normalize((actor if row["role"] == "actor" else patient) or "")})
        if off % 40 == 0:
            print(start, arm, min(off+4, len(rows)), "/", len(rows), flush=True)
    assert len(out) == 204
    target.write_text(json.dumps({"start": start, "arm": arm,
                                  "protocol_sha256": sha(ROOT / "PROTOCOL.json"),
                                  "checkpoint_sha256": cfg["checkpoint_sha256"][f"{start}_{arm}"],
                                  "records": out}, indent=2) + "\n")
    print("Saved", target.name, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "eval"))
    parser.add_argument("--start", choices=prior.STARTS)
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare()
    else:
        if not args.start or not args.arm:
            parser.error("eval requires --start and --arm")
        evaluate(args.start, args.arm)
