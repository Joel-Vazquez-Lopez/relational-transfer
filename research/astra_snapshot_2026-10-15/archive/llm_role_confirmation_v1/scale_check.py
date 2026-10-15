"""Secondary same-family size check with unmodified Qwen2.5-0.5B-Instruct."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
import llm_role_confirmation_v1 as main_eval  # noqa: E402

MODEL = Path("/Users/joelvazquezlopez/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protocol() -> dict:
    return {
        "date": "2026-10-10", "status": "fixed_before_secondary_inference",
        "model": "unmodified Qwen2.5-0.5B-Instruct local snapshot",
        "model_weights_sha256": sha(MODEL / "model.safetensors"),
        "source_sha256": sha(ROOT / "scale_check.py"),
        "items_sha256": sha(ROOT / "ITEMS.json"),
        "design": "Same frozen 120 events and 480 prompts as 1.5B confirmation. Secondary same-family size check, not an independent architecture replication. Same exact first-answer-token scoring.",
        "interpretation": "Report all form-by-wording cells and compare descriptively with the 1.5B model; no pooled generalization claim.",
    }


def prepare() -> None:
    path = ROOT / "SCALE_PROTOCOL.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(protocol(), indent=2) + "\n")
    print(f"Prepared {path}")


@torch.inference_mode()
def run() -> None:
    assert json.loads((ROOT / "SCALE_PROTOCOL.json").read_text()) == protocol()
    rows = json.loads((ROOT / "ITEMS.json").read_text())
    assert rows == main_eval.items()
    destination = ROOT / "SCALE_RESULT.json"
    if destination.exists():
        raise FileExistsError(destination)
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    for name in {r["actor"] for r in rows} | {r["patient"] for r in rows}:
        assert len(tokenizer.encode(name, add_special_tokens=False)) == 1, name
        assert len(tokenizer.encode(" " + name, add_special_tokens=False)) == 1, name
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True,
                                               dtype=torch.float32).eval()
    records = []
    for start in range(0, len(rows), 4):
        batch = rows[start:start + 4]
        enc = tokenizer([main_eval.prompt(tokenizer, row) for row in batch],
                        padding=True, return_tensors="pt")
        scores = model(**enc, use_cache=False).logits[:, -1]
        for row, logits in zip(batch, scores):
            answer = tokenizer.decode([int(logits.argmax())]).strip()
            a = torch.logsumexp(logits[main_eval.name_ids(tokenizer, row["actor"])], dim=0)
            p = torch.logsumexp(logits[main_eval.name_ids(tokenizer, row["patient"])], dim=0)
            records.append({**row, "answer": answer,
                            "valid": answer in (row["actor"], row["patient"]),
                            "correct": answer == row["actor"],
                            "actor_margin": float((a - p).item())})
        if (start + len(batch)) % 40 == 0:
            print(f"0.5B {start + len(batch)}/{len(rows)}", flush=True)
    destination.write_text(json.dumps({"protocol_sha256": sha(ROOT / "SCALE_PROTOCOL.json"),
                                       "records": records}, indent=2) + "\n")


def summarize() -> None:
    obj = json.loads((ROOT / "SCALE_RESULT.json").read_text())
    assert obj["protocol_sha256"] == sha(ROOT / "SCALE_PROTOCOL.json")
    rows = obj["records"]
    cells = {}
    for form in ("active", "passive"):
        for wording in ("generic", "performed"):
            sub = [r for r in rows if (r["form"], r["wording"]) == (form, wording)]
            cells[f"{form}/{wording}"] = {"n": len(sub),
                "correct": sum(r["correct"] for r in sub),
                "valid": sum(r["valid"] for r in sub)}
    out = {"model": "Qwen2.5-0.5B-Instruct", "cells": cells}
    (ROOT / "SCALE_SUMMARY.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run", "summarize"))
    args = parser.parse_args()
    if args.action == "prepare": prepare()
    elif args.action == "run": run()
    else: summarize()


if __name__ == "__main__":
    cli()
