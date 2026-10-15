#!/usr/bin/env python3
"""Frozen-checkpoint Qwen role-equivariance diagnostic on new constructed items."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parents[1]
sys.path.insert(0, str(WORK))
import llm_role_independent_run as adapters  # noqa: E402
from llm_role_natural_prompt_screen import QWEN05  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def config() -> dict:
    return json.loads((ROOT / "PROTOCOL.json").read_text())


def events() -> list[dict]:
    cfg = config()
    records = []
    for pair_index, (a, b) in enumerate(cfg["pairs"]):
        for verb in cfg["verbs"]:
            for direction, (actor, patient) in enumerate(((a, b), (b, a))):
                records.append({"id": f"p{pair_index}-{verb}-d{direction}",
                                "pair_index": pair_index, "a": a, "b": b,
                                "verb": verb, "direction": direction,
                                "actor": actor, "patient": patient})
    assert len(records) == 36
    return records


def sentence(e: dict, form: str) -> str:
    a, p, v = e["actor"], e["patient"], e["verb"]
    return {
        "active": f"{a} {v} {p}.",
        "passive": f"{p} was {v} by {a}.",
        "object_cleft": f"It was {p} that {a} {v}.",
        "subject_cleft": f"It was {a} who {v} {p}.",
    }[form]


def items() -> list[dict]:
    cfg = config()
    rows = []
    for e in events():
        for form in cfg["forms"]:
            for wording, pattern in cfg["questions"].items():
                role = "actor" if wording.startswith("actor_") else "patient"
                rows.append({**e, "form": form, "wording": wording,
                             "role": role, "question": pattern.format(verb=e["verb"]),
                             "sentence": sentence(e, form), "gold": e[role]})
    assert len(rows) == 576
    return rows


def prompt(tok, row: dict, omitted: bool = False) -> str:
    s = "[omitted]" if omitted else row["sentence"]
    return tok.apply_chat_template([
        {"role": "system", "content": "Answer with exactly one of the two candidate names. Output only the name."},
        {"role": "user", "content": f"Sentence: {s}\nCandidates: {row['a']}, {row['b']}\nQuestion: {row['question']}"},
    ], tokenize=False, add_generation_prompt=True)


def name_ids(tok, name: str) -> list[int]:
    bare = tok.encode(name, add_special_tokens=False)
    spaced = tok.encode(" " + name, add_special_tokens=False)
    assert len(bare) == len(spaced) == 1
    return list(set((bare[0], spaced[0])))


def margin(logits, tok, e):
    actor = torch.logsumexp(logits[name_ids(tok, e["actor"])], 0)
    patient = torch.logsumexp(logits[name_ids(tok, e["patient"])], 0)
    return float(actor - patient)


def load_model(name: str):
    if name == "untouched_qwen05":
        tok = AutoTokenizer.from_pretrained(QWEN05, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(QWEN05, local_files_only=True, dtype=torch.float32).eval()
        return tok, model
    return adapters.load(name)


def eval_model(name: str) -> dict:
    cfg = config()
    if name not in cfg["models"]:
        raise ValueError(name)
    target = ROOT / f"{name}.json"
    if target.exists():
        raise FileExistsError(target)
    torch.set_num_threads(2)
    tok, model = load_model(name)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    for a, b in cfg["pairs"]:
        name_ids(tok, a)
        name_ids(tok, b)
    rows = items()
    answers = []
    @torch.inference_mode()
    def score(block: list[dict], omitted: bool) -> None:
        prompts = [prompt(tok, x, omitted) for x in block]
        inputs = tok(prompts, padding=True, return_tensors="pt")
        logits = model(**inputs, use_cache=False).logits[:, -1]
        for i, row in enumerate(block):
            record = {"id": row["id"], "pair_index": row["pair_index"],
                      "direction": row["direction"], "verb": row["verb"],
                      "form": row["form"], "wording": row["wording"],
                      "role": row["role"], "actor_margin": margin(logits[i], tok, row),
                      "omitted": omitted}
            if not omitted:
                token = int(logits[i].argmax())
                response = tok.decode([token]).strip()
                record.update({"answer": response, "gold": row["gold"],
                               "valid": response in (row["actor"], row["patient"]),
                               "correct": response == row["gold"]})
            answers.append(record)
    for i in range(0, len(rows), 8):
        score(rows[i:i+8], omitted=False)
        if (i + 8) % 96 == 0:
            print(f"{name}: present {min(i+8,len(rows))}/{len(rows)}", flush=True)
    # Only one omitted control per event/wording, since sentence form is absent.
    omitted_rows = [r for r in rows if r["form"] == "active"]
    for i in range(0, len(omitted_rows), 8):
        score(omitted_rows[i:i+8], omitted=True)
    report = summarize_records(answers)
    payload = {"model": name, "protocol_sha256": sha(ROOT / "PROTOCOL.json"),
               "script_sha256": sha(Path(__file__)),
               "model_source": str(QWEN05) if name == "untouched_qwen05" else str(adapters.CHECKPOINTS[name][1]),
               "model_source_sha256": sha(QWEN05 / "model.safetensors") if name == "untouched_qwen05" else sha(adapters.CHECKPOINTS[name][1]),
               "summary": report, "records": answers}
    target.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"model": name, "summary": report}, indent=2), flush=True)
    del model, tok
    gc.collect()
    return report


def summarize_records(records: list[dict]) -> dict:
    present = [r for r in records if not r["omitted"]]
    omitted = [r for r in records if r["omitted"]]
    per_event = defaultdict(list)
    cells = defaultdict(lambda: [0, 0])
    for r in present:
        per_event[r["id"]].append(r["correct"])
        key = f"{r['form']}/{r['role']}/{r['wording']}"
        cells[key][0] += int(r["correct"])
        cells[key][1] += 1
    assert len(per_event) == 36 and all(len(x) == 16 for x in per_event.values())
    joint = sum(all(x) for x in per_event.values())
    return {"joint_events": joint, "events": 36,
            "valid_outputs": sum(r["valid"] for r in present), "total_prompts": len(present),
            "all_prompt_accuracy": sum(r["correct"] for r in present) / len(present),
            "by_form": {f: {"correct": sum(r["correct"] for r in present if r["form"] == f),
                             "total": sum(r["form"] == f for r in present)}
                        for f in config()["forms"]},
            "mean_abs_present_actor_margin": sum(abs(r["actor_margin"]) for r in present)/len(present),
            "mean_abs_omitted_actor_margin": sum(abs(r["actor_margin"]) for r in omitted)/len(omitted),
            "mean_target_margin_present": sum(r["actor_margin"] * (1 if r["role"] == "actor" else -1) for r in present)/len(present),
            "mean_target_margin_omitted": sum(r["actor_margin"] * (1 if r["role"] == "actor" else -1) for r in omitted)/len(omitted),
            "cells": {k: {"correct": a, "total": n} for k, (a, n) in sorted(cells.items())}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=config()["models"])
    eval_model(parser.parse_args().model)
