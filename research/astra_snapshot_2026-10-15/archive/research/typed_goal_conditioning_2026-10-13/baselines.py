#!/usr/bin/env python3
"""Secondary fixed-checkpoint comparison on fresh2, no retraining."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent / "natural_role_verifier_development_2026-10-13"
sys.path.insert(0, str(PRIOR))
import natural_train as prior  # noqa: E402

STARTS = ("fresh1", "fresh3")
ARMS = ("answer_only", "role_first")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    target = ROOT / "BASELINE_PROTOCOL.json"
    if target.exists():
        raise FileExistsError(target)
    cfg = {"status": "secondary comparison frozen before baseline inference on fresh2",
           "starts": STARTS, "arms": ARMS,
           "main_protocol_sha256": sha(ROOT / "PROTOCOL.json"),
           "splits_sha256": sha(ROOT / "SPLITS.json"),
           "script_sha256": sha(Path(__file__)),
           "checkpoint_sha256": {f"{s}_{a}": sha(prior.checkpoint(s, a)) for s in STARTS for a in ARMS},
           "limits": "These fixed checkpoints were trained on the same 160 source items but use different output targets and token counts; they are descriptive baselines, not the matched causal control."}
    target.write_text(json.dumps(cfg, indent=2) + "\n")
    print("Frozen", sha(target))


@torch.inference_mode()
def eval_one(start, arm):
    cfg = json.loads((ROOT / "BASELINE_PROTOCOL.json").read_text())
    assert cfg["main_protocol_sha256"] == sha(ROOT / "PROTOCOL.json")
    assert cfg["splits_sha256"] == sha(ROOT / "SPLITS.json")
    assert cfg["script_sha256"] == sha(Path(__file__))
    assert cfg["checkpoint_sha256"][f"{start}_{arm}"] == sha(prior.checkpoint(start, arm))
    dest = ROOT / f"BASELINE_{start}_{arm}_fresh2.json"
    if dest.exists():
        raise FileExistsError(dest)
    torch.set_num_threads(2)
    tok, model = prior.load_trained(start, arm)
    items = json.loads((ROOT / "SPLITS.json").read_text())["fresh2"]
    rows = [row for item in items for row in prior.rows(item)]
    records = []
    for off in range(0, len(rows), 4):
        block = rows[off:off+4]
        encoded = tok([prior.format_prompt(tok, row, arm) for row in block],
                      padding=True, return_tensors="pt", add_special_tokens=False)
        generated = model.generate(**encoded, max_new_tokens=48, do_sample=False,
                                   pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        for row, tokens in zip(block, generated[:, encoded["input_ids"].shape[1]:]):
            raw = tok.decode(tokens, skip_special_tokens=True)
            answer, actor, patient = prior.parse(raw, arm)
            field = actor if row["role"] == "actor" else patient
            records.append({"id": row["id"], "role": row["role"], "order": row["order"],
                            "gold": row["gold"], "raw": raw, "answer": answer,
                            "actor": actor, "patient": patient,
                            "correct": answer is not None and prior.norm(answer) == prior.norm(row["gold"]),
                            "routed_correct": field is not None and prior.norm(field) == prior.norm(row["gold"])})
        if (off+4) % 40 == 0:
            print(start, arm, off+4, "/", len(rows), flush=True)
    dest.write_text(json.dumps({"start": start, "arm": arm,
                                "baseline_protocol_sha256": sha(ROOT / "BASELINE_PROTOCOL.json"),
                                "checkpoint_sha256": sha(prior.checkpoint(start, arm)),
                                "records": records}, indent=2) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("prepare", "eval"))
    ap.add_argument("--start", choices=STARTS)
    ap.add_argument("--arm", choices=ARMS)
    args = ap.parse_args()
    if args.mode == "prepare":
        prepare()
    else:
        if not args.start or not args.arm:
            ap.error("eval needs --start and --arm")
        eval_one(args.start, args.arm)
