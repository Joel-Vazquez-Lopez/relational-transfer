#!/usr/bin/env python3
"""Post hoc inference-format crossover on fixed answer-first/slot-first adapters."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "role_slot_routing_fresh_confirmation_2026-10-13"))
import confirm as prior  # noqa: E402


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows():
    return prior.test_rows()


def prepare():
    path = ROOT / "PROTOCOL.json"
    if path.exists():
        raise FileExistsError(path)
    fixed = prior.validate()
    config = {
        "date": "2026-10-13",
        "status": "exploratory, conceived after the main confirmation; frozen before crossover inference",
        "question": "Does the answer-first checkpoint recover slot-first performance when prompted to output actor/patient before its answer, without changing weights?",
        "starts": list(prior.STARTS),
        "checkpoint_arms": ["answer_first", "slot_first"],
        "prompt_format": "opposite trained format",
        "test_rows": "same 960 rows per checkpoint as prior fresh lexical confirmation",
        "primary": "both crossed actor_did and patient_verb answers correct in passive, subject cleft, and object cleft; 144 event-form groups/checkpoint",
        "secondary": "valid-answer fraction, per-form crossed joint, generated-slot joint, answer-to-slot agreement",
        "interpretation": "If answer-first with slot-first prompting approaches native slot-first, much of the effect may be an inference interface effect. If it does not, training or learned format remains relevant. Crossing formats is distribution shift; poor crossover alone does not isolate semantic training.",
        "limits": "Post hoc mechanism diagnostic on an already evaluated synthetic bank; not fresh confirmation, new training replication, or natural-language transfer.",
        "prior_protocol_sha256": sha(prior.ROOT / "PROTOCOL.json"),
        "prior_inference_script_sha256": fixed["script_sha256"],
        "script_sha256": sha(Path(__file__)),
        "test_rows_sha256": hashlib.sha256(json.dumps(rows(), sort_keys=True).encode()).hexdigest(),
        "checkpoint_sha256": {f"{s}_{a}": sha(prior.checkpoint(s, a)) for s in prior.STARTS for a in ("answer_first", "slot_first")},
    }
    path.write_text(json.dumps(config, indent=2) + "\n")
    print("Frozen", len(rows()), "rows/checkpoint", flush=True)


def validate():
    cfg = json.loads((ROOT / "PROTOCOL.json").read_text())
    prior.validate()
    assert cfg["script_sha256"] == sha(Path(__file__))
    assert cfg["prior_protocol_sha256"] == sha(prior.ROOT / "PROTOCOL.json")
    assert cfg["test_rows_sha256"] == hashlib.sha256(json.dumps(rows(), sort_keys=True).encode()).hexdigest()
    for key, digest in cfg["checkpoint_sha256"].items():
        start, arm = key.split("_", 1)
        assert sha(prior.checkpoint(start, arm)) == digest
    return cfg


@torch.inference_mode()
def evaluate(start, arm):
    cfg = validate()
    target = ROOT / f"EVAL_{start}_{arm}_cross.json"
    if target.exists():
        raise FileExistsError(target)
    prompt_arm = "slot_first" if arm == "answer_first" else "answer_first"
    torch.set_num_threads(2)
    tok, model = prior.base.prev.load(start)
    saved = torch.load(prior.checkpoint(start, arm), map_location="cpu", weights_only=True)
    params = {name: p for name, p in model.named_parameters() if p.requires_grad}
    assert saved.keys() == params.keys()
    for name, p in params.items():
        p.data.copy_(saved[name])
    model.eval()
    out = []
    dataset = rows()
    for off in range(0, len(dataset), 8):
        block = dataset[off:off+8]
        batch = tok([prior.base.prompt(tok, row, prompt_arm) for row in block], padding=True,
                    return_tensors="pt", add_special_tokens=False)
        generated = model.generate(**batch, max_new_tokens=13, do_sample=False,
                                   pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        for row, tokens in zip(block, generated[:, batch["input_ids"].shape[1]:]):
            raw = tok.decode(tokens, skip_special_tokens=True)
            answer, actor, patient = prior.base.parse_output(raw, prompt_arm)
            out.append({"event": row["id"], "pair_index": row["pair_index"],
                        "form": row["form"], "wording": row["wording"], "role": row["role"],
                        "gold": row["gold"], "raw": raw, "answer": answer,
                        "actor": actor, "patient": patient,
                        "valid": answer in row["pair"], "correct": answer == row["gold"],
                        "slot_joint_correct": actor == row["actor"] and patient == row["patient"],
                        "answer_consistent_with_slots": answer == (actor if row["role"] == "actor" else patient)})
        if (off + 8) % 240 == 0:
            print(start, arm, "cross", off + 8, "/", len(dataset), flush=True)
    assert len(out) == 960
    target.write_text(json.dumps({"start": start, "checkpoint_arm": arm,
                                  "prompt_arm": prompt_arm,
                                  "protocol_sha256": sha(ROOT / "PROTOCOL.json"),
                                  "checkpoint_sha256": cfg["checkpoint_sha256"][f"{start}_{arm}"],
                                  "records": out}, indent=2) + "\n")
    print("Saved", target.name, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "eval"))
    parser.add_argument("--start", choices=prior.STARTS)
    parser.add_argument("--arm", choices=("answer_first", "slot_first"))
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare()
    else:
        if not args.start or not args.arm:
            parser.error("eval requires --start and --arm")
        evaluate(args.start, args.arm)
