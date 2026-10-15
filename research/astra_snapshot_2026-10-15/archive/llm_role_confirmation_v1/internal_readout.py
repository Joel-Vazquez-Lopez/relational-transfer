"""Fixed intermediate-state logit-lens readout on the fresh Qwen role panel."""
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

MODEL = main_eval.MODEL
LAYERS = (0, 4, 8, 12, 16, 20, 24, 28)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protocol() -> dict:
    return {
        "date": "2026-10-10", "status": "fixed_before_internal_readout",
        "model": "unmodified Qwen2.5-1.5B-Instruct local snapshot",
        "model_weights_sha256": sha(MODEL / "model.safetensors"),
        "source_sha256": sha(ROOT / "internal_readout.py"),
        "items_sha256": sha(ROOT / "ITEMS.json"),
        "readout": "At the final prompt position after blocks 0,4,8,12,16,20,24,28, apply final norm (except on final state) and fixed LM head restricted to actor/patient name tokens. Save actor-minus-patient logsumexp logit margin; no fitted probe and no weight changes.",
        "primary_summary": "For each form x wording x layer, actor-positive fraction and mean actor margin over 120 prompts, plus by-set fractions at block20. On each event compare signs across four prompt variants. Full-layer actor margin should match the saved behavioral result.",
        "limits": "An intermediate logit lens is an approximate readout, and candidate-name preference is not an isolated semantic role representation. Prompt content changes with form and wording. A causal or construction-general role claim requires stronger controls.",
    }


def prepare() -> None:
    path = ROOT / "INTERNAL_PROTOCOL.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(protocol(), indent=2) + "\n")
    print(f"Prepared {path}")


@torch.inference_mode()
def run() -> None:
    assert json.loads((ROOT / "INTERNAL_PROTOCOL.json").read_text()) == protocol()
    rows = json.loads((ROOT / "ITEMS.json").read_text())
    assert rows == main_eval.items()
    dst = ROOT / "INTERNAL_RESULT.json"
    if dst.exists():
        raise FileExistsError(dst)
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True,
                                               dtype=torch.float32).eval()
    assert model.config.num_hidden_layers == 28
    records = []
    for start in range(0, len(rows), 4):
        batch = rows[start:start + 4]
        enc = tokenizer([main_eval.prompt(tokenizer, row) for row in batch],
                        padding=True, return_tensors="pt")
        output = model(**enc, use_cache=False, output_hidden_states=True)
        for i, row in enumerate(batch):
            ids_a = main_eval.name_ids(tokenizer, row["actor"])
            ids_p = main_eval.name_ids(tokenizer, row["patient"])
            margins = {}
            for layer in LAYERS:
                state = output.hidden_states[layer][i, -1]
                if layer != 28:
                    state = model.model.norm(state)
                ids = ids_a + ids_p
                scores = torch.mv(model.lm_head.weight[ids], state)
                margins[str(layer)] = float((torch.logsumexp(scores[:2], dim=0) -
                                             torch.logsumexp(scores[2:], dim=0)).item())
            records.append({"event": row["event"], "set": row["set"],
                            "pair": row["pair"], "form": row["form"],
                            "wording": row["wording"], "actor_margins": margins})
        if (start + len(batch)) % 40 == 0:
            print(f"internal {start + len(batch)}/{len(rows)}", flush=True)
    dst.write_text(json.dumps({"protocol_sha256": sha(ROOT / "INTERNAL_PROTOCOL.json"),
                               "records": records}, indent=2) + "\n")
    print(f"Saved {dst}")


def summarize() -> None:
    obj = json.loads((ROOT / "INTERNAL_RESULT.json").read_text())
    assert obj["protocol_sha256"] == sha(ROOT / "INTERNAL_PROTOCOL.json")
    rows = obj["records"]
    cells = {}
    for form in ("active", "passive"):
        for wording in ("generic", "performed"):
            sub = [r for r in rows if (r["form"], r["wording"]) == (form, wording)]
            layers = {}
            for layer in LAYERS:
                values = [r["actor_margins"][str(layer)] for r in sub]
                layers[str(layer)] = {"n": len(values), "actor_positive": sum(x > 0 for x in values),
                                      "mean_margin": sum(values) / len(values),
                                      "by_set_positive": [sum(r["actor_margins"][str(layer)] > 0
                                                              for r in sub if r["set"] == s)
                                                          for s in range(3)]}
            cells[f"{form}/{wording}"] = layers
    report = {"n_prompts": 480, "cells": cells}
    (ROOT / "INTERNAL_SUMMARY.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run", "summarize"))
    args = parser.parse_args()
    if args.action == "prepare": prepare()
    elif args.action == "run": run()
    else: summarize()


if __name__ == "__main__":
    cli()
