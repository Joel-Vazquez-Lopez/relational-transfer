"""Untouched 1.5B baseline on the already frozen EWT test, no training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import llm_role_independent_eval as data
from llm_role_independent_run import generated, margin, paraphrase_prompt
from llm_role_natural_contrast_screen import prompt as original_prompt
from llm_role_natural_prompt_screen import QWEN15

ROOT = Path(__file__).resolve().parent / "llm_role_independent_eval/qwen15"
EVENTS = Path(__file__).resolve().parent / "llm_role_independent_eval/EVENTS.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze():
    rows = data.validate()
    ROOT.mkdir(exist_ok=True)
    target = ROOT / "PROTOCOL.json"
    if target.exists():
        raise FileExistsError(target)
    p = {"status": "fixed_before_Qwen1.5B_EWT_inference",
         "date": "2026-10-12",
         "model": "untouched local Qwen2.5-1.5B-Instruct, no new training or intervention",
         "purpose": "Check whether candidate contrast has headroom in a larger local model on the already fixed EWT items. This is a measure-feasibility check, not a replication of the adaptation effect.",
         "measures": "Same primary D>0, omitted-sentence D>0, paraphrased D>0, and greedy strict direct actor/object spans as the 0.5B cross-corpus run. Same 75 events, no exclusions.",
         "limits": "EWT items and 0.5B results were inspected before this baseline; labels remain unreviewed syntactic proxies, and pretraining exposure is unknown.",
         "n_events": len(rows),
         "model_snapshot": str(QWEN15),
         "references": {"events_sha256": sha(EVENTS),
                        "selection_protocol_sha256": sha(data.ROOT / "PROTOCOL.json"),
                        "script_sha256": sha(Path(__file__))}}
    target.write_text(json.dumps(p, indent=2)+"\n")
    print("Frozen 1.5B baseline protocol")


def validate():
    rows = data.validate()
    p = json.loads((ROOT / "PROTOCOL.json").read_text())
    assert p["n_events"] == len(rows)
    assert p["references"]["events_sha256"] == sha(EVENTS)
    assert p["references"]["selection_protocol_sha256"] == sha(data.ROOT / "PROTOCOL.json")
    assert p["references"]["script_sha256"] == sha(Path(__file__))
    assert str(QWEN15) == p["model_snapshot"] and QWEN15.exists()
    return rows


@torch.inference_mode()
def run():
    rows = validate()
    target = ROOT / "EVALUATION.json"
    if target.exists():
        raise FileExistsError(target)
    torch.set_num_threads(2)
    tok = AutoTokenizer.from_pretrained(QWEN15, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(QWEN15, local_files_only=True,
                                                 dtype=torch.float32).eval()
    records = []
    for i, row in enumerate(rows, 1):
        rec = {"id": row["id"],
               "primary": margin(tok, model, row, original_prompt),
               "omitted": margin(tok, model, {**row, "sentence": "[omitted]"}, original_prompt),
               "paraphrase": margin(tok, model, row, paraphrase_prompt),
               "direct": {role: generated(tok, model, row, role) for role in ("actor", "other")}}
        records.append(rec)
        if i % 15 == 0:
            print(f"qwen15 {i}/{len(rows)}", flush=True)
    summary = {"primary": sum(r["primary"] > 0 for r in records),
               "omitted": sum(r["omitted"] > 0 for r in records),
               "paraphrase": sum(r["paraphrase"] > 0 for r in records),
               "direct_actor": sum(r["direct"]["actor"]["exact"] for r in records),
               "direct_other": sum(r["direct"]["other"]["exact"] for r in records),
               "direct_both": sum(r["direct"]["actor"]["exact"] and
                                  r["direct"]["other"]["exact"] for r in records)}
    target.write_text(json.dumps({"summary": summary, "records": records,
                                  "protocol_sha256": sha(ROOT / "PROTOCOL.json"),
                                  "events_sha256": sha(EVENTS)}, indent=2)+"\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "validate", "run"))
    action = parser.parse_args().action
    if action == "freeze":
        freeze()
    elif action == "validate":
        print(f"Validated {len(validate())} rows")
    else:
        run()
