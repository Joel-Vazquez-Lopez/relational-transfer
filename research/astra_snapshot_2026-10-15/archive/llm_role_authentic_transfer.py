"""Matched natural-sentence evaluation of the existing Qwen 0.5B adapters.

The QA-SRL items were previously screened and used in a separate Qwen 1.5B
pilot. This is an exploratory external-domain check, not a blind confirmation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE / "llm_role_authentic_transfer"
SOURCE = HERE / "llm_role_confirmation_v1"
CONFIRM = HERE / "llm_role_repair_confirmation"
UNORDERED = HERE / "llm_role_unordered_control"
ARMS = ("pretrained", "ordinary", "ordered_roles", "unordered_pair")
WORLDS = ("fresh1", "fresh2", "fresh3")
SEEDS = (830211, 830212)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_items() -> list[dict]:
    rows = json.loads((SOURCE / "QASRL_ITEMS.json").read_text())
    assert len(rows) == 504
    assert len({row["id"] for row in rows}) == 63
    assert all(row["gold"] in ("A", "B") for row in rows)
    assert all(row["choice_a"] != row["choice_b"] for row in rows)
    counts = defaultdict(int)
    for row in rows:
        counts[row["id"]] += 1
        assert row["gold"] == ("A" if row["choice_a"] == row[row["role"]] else "B")
    assert set(counts.values()) == {8}
    return rows


def references() -> dict:
    files = {
        "source_items": SOURCE / "QASRL_ITEMS.json",
        "source_candidates": SOURCE / "QASRL_CANDIDATES.json",
        "source_protocol": SOURCE / "QASRL_PROTOCOL.json",
        "old_result": SOURCE / "QASRL_RESULT.json",
        "original_selection_code": SOURCE / "qasrl_eval.py",
        "confirmation_protocol": CONFIRM / "PROTOCOL.json",
        "unordered_protocol": UNORDERED / "PROTOCOL.json",
        "model_weights": Path("/Users/joelvazquezlopez/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775/model.safetensors"),
    }
    return {key: sha(path) for key, path in files.items()}


def protocol() -> dict:
    return {
        "status": "fixed_before_cross_adapter_natural_sentence_inference",
        "date": "2026-10-11",
        "dataset": "QA-SRL Gold, Wikinews and Wikipedia dev/test; 63 source sentences, 504 fixed prompts from prior exploratory Qwen 1.5B evaluation.",
        "dataset_source": "https://github.com/plroit/qasrl-gs",
        "reuse_disclosure": "These exact 63 sentences were previously evaluated on an unmodified Qwen2.5-1.5B model. They were selected by an AI assistant from 100 rule-filtered candidates before that inference, but selection and prior answers are known. No claim of a fresh blind test. The student_verified column in the prior review sheet is blank.",
        "question": "Does auxiliary role/participant supervision that protected synthetic patient-cleft accuracy also change performance on selected authentic sentences, relative to matched ordinary fine-tuning?",
        "model": "Local Qwen2.5-0.5B-Instruct weights and already trained LoRA adapters, no further updates.",
        "units": "Three fresh graph/name worlds x two matched random starts. Evaluate ordinary, ordered-role and unordered-participant arms in each; the unchanged pretrained model is scored once.",
        "primary": "Exact actor A/B accuracy over 63 sentences x two fixed actor question wordings x both option orders = 252 prompts per model; report six paired arm-minus-ordinary differences, equal-unit mean and each item/wording/order cell. No significance claim from treating prompts as independent.",
        "secondary": "Patient A/B accuracy on the same events, all-eight-prompts-correct event rate, option-order and wording sensitivity, invalid first-token rate, macro scores by Wikipedia/Wikinews. Report all results even if primary is negative.",
        "scoring": "Greedy first generated token after the fixed chat prompt; strip whitespace and accept only A or B. Any other token is invalid and wrong. Both A/B orders are shown. No post-hoc prompt edits or exclusions.",
        "interpretation": "Different domain and prompt style from synthetic training, with long sentences and multiword answer spans. A gain would support transfer to this selected natural-sentence role-choice task; a null result does not rule out gains under other natural evaluations. QA-SRL annotations and item suitability require independent human review before a paper-level claim. Prior Qwen 1.5B results and these items are known, so this is exploratory, not independent confirmation. Model pretraining overlap is unknown.",
        "references_sha256": references(),
        "script_sha256": sha(HERE / "llm_role_authentic_transfer.py"),
    }


def prepare() -> None:
    ROOT.mkdir(exist_ok=True)
    if (ROOT / "PROTOCOL.json").exists():
        raise FileExistsError(ROOT / "PROTOCOL.json")
    rows = source_items()
    (ROOT / "ITEMS.json").write_text(json.dumps(rows, indent=2) + "\n")
    (ROOT / "PROTOCOL.json").write_text(json.dumps(protocol(), indent=2) + "\n")
    events = {row["id"]: row for row in rows}
    with (ROOT / "HUMAN_REVIEW.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("id", "partition", "sentence", "target_verb", "actor_span",
                         "patient_span", "include_yes_no", "corrected_actor",
                         "corrected_patient", "reason_or_notes"))
        for row in events.values():
            writer.writerow((row["id"], row["partition"], row["sentence"], row["verb"],
                             row["actor"], row["patient"], "", "", "", ""))
    print(f"Frozen {len(events)} events and {len(rows)} prompts in {ROOT}")


def validate() -> tuple[dict, list[dict]]:
    saved = json.loads((ROOT / "PROTOCOL.json").read_text())
    assert saved == protocol(), "Protocol or referenced inputs changed"
    rows = json.loads((ROOT / "ITEMS.json").read_text())
    assert rows == source_items(), "Frozen items changed"
    return saved, rows


def prompt(tokenizer, row: dict) -> str:
    content = (f"Sentence: {row['sentence']}\n"
               f"Target verb: {row['verb']}\n"
               f"Question: {row['question']}\n"
               f"A. {row['choice_a']}\nB. {row['choice_b']}")
    return tokenizer.apply_chat_template([
        {"role": "system", "content": "Answer the question by choosing exactly one option. Reply with only A or B."},
        {"role": "user", "content": content},
    ], tokenize=False, add_generation_prompt=True)


def adapter_path(world: str, seed: int, arm: str) -> Path:
    if arm == "unordered_pair":
        return UNORDERED / f"{world}_{seed}_unordered_pair" / "adapter.pt"
    return CONFIRM / f"{world}_{seed}_{arm}" / "adapter.pt"


def run(world: str, seed: int, arm: str, limit: int | None = None) -> None:
    if arm == "pretrained":
        assert world == "fresh1" and seed == 0
    else:
        assert world in WORLDS and seed in SEEDS
    _, rows = validate()
    if limit is not None:
        assert 0 < limit <= len(rows)
        rows = rows[:limit]
    import torch
    import llm_role_repair_confirm as confirm
    import llm_role_repair_v3 as core

    label = f"{world}_{seed}_{arm}" + (f"_smoke{limit}" if limit else "")
    target = ROOT / f"{label}.json"
    if target.exists():
        raise FileExistsError(target)
    confirm.configure(world, seed)
    tokenizer, model, _ = core.load("pretrained" if arm == "pretrained" else "ordinary")
    model.eval()
    adapter_sha = None
    if arm != "pretrained":
        path = adapter_path(world, seed, arm)
        adapter_sha = sha(path)
        weights = torch.load(path, map_location="cpu", weights_only=True)["adapter"]
        trainable = {name: parameter for name, parameter in model.named_parameters()
                     if parameter.requires_grad}
        assert weights.keys() == trainable.keys()
        for name, parameter in trainable.items():
            assert parameter.shape == weights[name].shape
            parameter.data.copy_(weights[name])
    records = []
    with torch.inference_mode():
        for start in range(0, len(rows), 4):
            batch = rows[start:start + 4]
            enc = tokenizer([prompt(tokenizer, row) for row in batch],
                            padding=True, return_tensors="pt")
            logits = model(**enc, use_cache=False).logits[:, -1]
            for row, score in zip(batch, logits, strict=True):
                answer = tokenizer.decode([int(score.argmax())]).strip()
                records.append({"id": row["id"], "role": row["role"],
                                "wording": row["wording"], "order": row["order"],
                                "gold": row["gold"], "answer": answer,
                                "valid": answer in ("A", "B"), "correct": answer == row["gold"]})
            if (start + len(batch)) % 80 == 0:
                print(f"{label}: {start + len(batch)}/{len(rows)}", flush=True)
    output = {"label": label, "world": world, "seed": seed, "arm": arm,
              "n_items": len(rows), "protocol_sha256": sha(ROOT / "PROTOCOL.json"),
              "items_sha256": sha(ROOT / "ITEMS.json"), "adapter_sha256": adapter_sha,
              "records": records}
    target.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Saved {target}; correct={sum(r['correct'] for r in records)}/{len(records)}")


def analyze() -> None:
    _, items = validate()
    source_by_id = {row["id"]: row["partition"].split(".")[0] for row in items}
    results = {}
    for arm in ARMS:
        units = [("fresh1", 0)] if arm == "pretrained" else [
            (world, seed) for world in WORLDS for seed in SEEDS]
        for world, seed in units:
            path = ROOT / f"{world}_{seed}_{arm}.json"
            if not path.exists():
                continue
            obj = json.loads(path.read_text())
            assert obj["protocol_sha256"] == sha(ROOT / "PROTOCOL.json")
            assert obj["items_sha256"] == sha(ROOT / "ITEMS.json")
            assert len(obj["records"]) == 504
            if arm != "pretrained":
                assert obj["adapter_sha256"] == sha(adapter_path(world, seed, arm))
            cells = {}
            for role in ("actor", "patient"):
                part = [r for r in obj["records"] if r["role"] == role]
                cells[role] = {"correct": sum(r["correct"] for r in part),
                               "n": len(part), "invalid": sum(not r["valid"] for r in part)}
                for wording in (0, 1):
                    for order in (0, 1):
                        subset = [r for r in part if r["wording"] == wording and r["order"] == order]
                        cells[f"{role}/wording{wording}/order{order}"] = {
                            "correct": sum(r["correct"] for r in subset),
                            "n": len(subset), "invalid": sum(not r["valid"] for r in subset)}
            by_event = defaultdict(list)
            for row in obj["records"]:
                by_event[row["id"]].append(row)
            cells["all_eight_correct"] = sum(all(r["correct"] for r in group)
                                             for group in by_event.values())
            for source in ("wikinews", "wikipedia"):
                subset = [r for r in obj["records"] if source_by_id[r["id"]] == source]
                cells[source] = {"correct": sum(r["correct"] for r in subset),
                                 "n": len(subset), "invalid": sum(not r["valid"] for r in subset)}
            results[(world, seed, arm)] = cells
    paired = []
    for world in WORLDS:
        for seed in SEEDS:
            baseline = results.get((world, seed, "ordinary"))
            if baseline is None:
                continue
            for arm in ("ordered_roles", "unordered_pair"):
                other = results.get((world, seed, arm))
                if other is None:
                    continue
                paired.append({"world": world, "seed": seed, "arm": arm,
                               "actor_delta_pp": 100 * (other["actor"]["correct"] - baseline["actor"]["correct"]) / 252,
                               "patient_delta_pp": 100 * (other["patient"]["correct"] - baseline["patient"]["correct"]) / 252})
    means = {arm: {
        metric: sum(row[metric] for row in paired if row["arm"] == arm) /
        sum(row["arm"] == arm for row in paired)
        for metric in ("actor_delta_pp", "patient_delta_pp")}
        for arm in ("ordered_roles", "unordered_pair")
        if any(row["arm"] == arm for row in paired)}
    report = {"complete": len(results) == 19,
              "cells": {f"{w}/{s}/{a}": cell for (w, s, a), cell in results.items()},
              "paired_differences": paired,
              "equal_unit_mean_differences": means,
              "note": "Exploratory selected authentic-sentence evaluation; independent label review pending."}
    (ROOT / "ANALYSIS.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def batch() -> None:
    validate()
    jobs = [("fresh1", 0, "pretrained")] + [
        (world, seed, arm) for world in WORLDS for seed in SEEDS
        for arm in ("ordinary", "ordered_roles", "unordered_pair")]
    for world, seed, arm in jobs:
        output = ROOT / f"{world}_{seed}_{arm}.json"
        if output.exists():
            print(f"Already complete: {output}", flush=True)
            continue
        subprocess.run([sys.executable, str(HERE / "llm_role_authentic_transfer.py"),
                        "run", "--world", world, "--seed", str(seed), "--arm", arm], check=True)
    analyze()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "validate", "run", "batch", "analyze"))
    parser.add_argument("--world", choices=WORLDS, default="fresh1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arm", choices=ARMS, default="pretrained")
    parser.add_argument("--limit", type=int, help="Run a nonreportable small smoke check")
    args = parser.parse_args()
    if args.action == "prepare": prepare()
    elif args.action == "validate":
        validate(); print("Frozen protocol and 504 items validated")
    elif args.action == "run": run(args.world, args.seed, args.arm, args.limit)
    elif args.action == "batch": batch()
    else: analyze()
