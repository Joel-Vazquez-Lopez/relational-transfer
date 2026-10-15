#!/usr/bin/env python3
"""Adaptive, no-training textual plan intervention for the capacity-check model."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

import capacity
import run as first

ROOT = Path(__file__).resolve().parent
CONDITIONS = ("own_plan", "gold_plan", "flipped_plan")
SPLITS = ("in_domain_test", "heldout_mixed")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare() -> None:
    path = ROOT / "INTERVENTION_PROTOCOL.json"
    if path.exists():
        raise FileExistsError(path)
    cfg = {
        "date": "2026-10-13", "status": "adaptive exploratory after two feasibility runs; no training",
        "question": "Does supplying correct operation tokens without entity answers improve final answering on held-out mixed compositions?",
        "design": "Prefix the goal-first assistant through the execute marker with (a) its own first eight emitted tokens, (b) the gold goal and operation tokens, or (c) the gold plan with both F/R directions flipped. Generate only intermediate names and final answer. The gold plan contains no middle or answer name. Same sentence/question/facts and checkpoint.",
        "splits": SPLITS, "conditions": CONDITIONS,
        "primary": "Exact final answer and exact middle entity for each condition; own-plan replay against original free generation; changed answer under flipped plan. Report in-domain and held-out separately, and do not call gold-plan performance end-to-end generalization.",
        "limits": "Chosen after seeing failure; same reused data, one scratch model, oracle operation information in gold_plan, free text prefix can alter generation. A change in answer is behavioral text dependence, not an internal-state mechanism.",
        "data_sha256": sha(ROOT / "DATA.json"),
        "vocab_sha256": sha(ROOT / "VOCAB.json"),
        "capacity_protocol_sha256": sha(ROOT / "CAPACITY_PROTOCOL.json"),
        "checkpoint_sha256": sha(ROOT / "CAP_MODEL_goal_first.pt"),
        "prior_eval_sha256": sha(ROOT / "CAP_EVAL_goal_first.json"),
        "script_sha256": sha(Path(__file__)),
    }
    path.write_text(json.dumps(cfg, indent=2) + "\n")
    print("Frozen adaptive textual intervention", sha(path))


def validate() -> tuple[dict, dict, dict]:
    cfg = json.loads((ROOT / "INTERVENTION_PROTOCOL.json").read_text())
    for key, path in (("data_sha256", ROOT / "DATA.json"),
                      ("vocab_sha256", ROOT / "VOCAB.json"),
                      ("capacity_protocol_sha256", ROOT / "CAPACITY_PROTOCOL.json"),
                      ("checkpoint_sha256", ROOT / "CAP_MODEL_goal_first.pt"),
                      ("prior_eval_sha256", ROOT / "CAP_EVAL_goal_first.json"),
                      ("script_sha256", Path(__file__))):
        assert cfg[key] == sha(path), key
    capacity.model_size()
    return cfg, first.read_data(), json.loads((ROOT / "VOCAB.json").read_text())


def prefix(row: dict, own: dict, condition: str) -> list[str]:
    gold = first.target(row, "goal_first")[:8]
    if condition == "own_plan":
        return own["output"][:8]
    if condition == "gold_plan":
        return gold
    altered = gold.copy()
    for i in (3, 6):
        altered[i] = "R" if altered[i] == "F" else "F"
    return altered


@torch.inference_mode()
def evaluate(condition: str) -> None:
    _, data, vocab = validate()
    path = ROOT / f"INTERVENTION_{condition}.json"
    if path.exists():
        raise FileExistsError(path)
    torch.set_num_threads(4)
    model = first.CausalLM(len(vocab))
    model.load_state_dict(torch.load(ROOT / "CAP_MODEL_goal_first.pt", map_location="cpu", weights_only=True))
    model.eval()
    prior = json.loads((ROOT / "CAP_EVAL_goal_first.json").read_text())["records"]
    inverse = {v: k for k, v in vocab.items()}
    results = {}
    for split in SPLITS:
        originals = {r["id"]: r for r in prior[split]}
        rows = data[split]
        records_by_id = {}
        groups: dict[int, list[dict]] = {}
        for row in rows:
            groups.setdefault(len(first.prompt(row)) + 8, []).append(row)
        for block_rows in groups.values():
            for off in range(0, len(block_rows), 32):
                block = block_rows[off:off+32]
                prefixes = [prefix(row, originals[row["id"]], condition) for row in block]
                assert all(len(pre) == 8 for pre in prefixes)
                sequences = [[vocab[x] for x in first.prompt(row) + pre]
                             for row, pre in zip(block, prefixes)]
                assert len(set(map(len, sequences))) == 1
                length = len(sequences[0])
                x = torch.tensor(sequences)
                for _ in range(6):
                    x = torch.cat([x, model(x)[:, -1].argmax(dim=-1, keepdim=True)], dim=1)
                for row, pre, tokens in zip(block, prefixes, x):
                    original = originals[row["id"]]
                    continuation = [inverse[int(t)] for t in tokens[length:]]
                    answer, output = first.parse(pre + continuation)
                    middle = continuation[0] if continuation and continuation[0].startswith("n") else None
                    records_by_id[row["id"]] = {"id": row["id"], "pattern": row["pattern"],
                        "gold": row["answer"], "middle_gold": row["middle"],
                        "prefix": pre, "continuation": continuation, "output": output,
                        "answer": answer, "middle": middle,
                        "correct": answer == row["answer"],
                        "middle_correct": middle == row["middle"],
                        "replay_answer_match": answer == original["answer"]}
        records = [records_by_id[row["id"]] for row in rows]
        results[split] = records
        print(condition, split, "answer", sum(r["correct"] for r in records),
              "middle", sum(r["middle_correct"] for r in records),
              "replay", sum(r["replay_answer_match"] for r in records), flush=True)
    path.write_text(json.dumps({"condition": condition,
                                "protocol_sha256": sha(ROOT / "INTERVENTION_PROTOCOL.json"),
                                "checkpoint_sha256": sha(ROOT / "CAP_MODEL_goal_first.pt"),
                                "records": results}, indent=2) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("prepare", "eval"))
    ap.add_argument("--condition", choices=CONDITIONS)
    args = ap.parse_args()
    if args.mode == "prepare":
        prepare()
    else:
        if not args.condition:
            ap.error("eval needs --condition")
        evaluate(args.condition)
