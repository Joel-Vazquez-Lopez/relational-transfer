#!/usr/bin/env python3
"""Independent, read-only scorer for Astra's 21 synthetic frozen output units.

Usage:
    python3 audit.py /path/to/completed_adapter_diagnostic --output audit.json
Requires only Python's standard library. No inference, networking or training.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

FORMS = ("active", "passive", "patient_cleft")
WORLDS = ("fresh1", "fresh2", "fresh3")
SEEDS = (830211, 830212)
ARMS = ("ordinary", "ordered_roles", "unordered_pair")

def units():
    for world in WORLDS:
        yield world + "_0_pretrained", world, "pretrained", 0
        for seed in SEEDS:
            for arm in ARMS:
                yield f"{world}_{seed}_{arm}", world, arm, seed

def score_one(item, prediction):
    def answer(name, key):
        p = prediction[key]["parsed"]
        return bool(p["semantic_format_valid"] and p["name"] == name)
    first = item["actor"] if item["sentence"].find(item["actor"]) < item["sentence"].find(item["patient"]) else item["patient"]
    last = item["patient"] if first == item["actor"] else item["actor"]
    if item["sentence"].count(item["actor"]) != 1 or item["sentence"].count(item["patient"]) != 1:
        raise ValueError("Expected one appearance of each participant")
    ex = prediction["extraction"]["parsed"]
    valid = bool(ex["semantic_format_valid"])
    a = answer(item["actor"], "direct_actor")
    p = answer(item["patient"], "direct_patient")
    return {
        "actor_direct": int(a), "patient_direct": int(p),
        "both_direct": int(a and p),
        "actor_archived": int(answer(item["actor"], "archived")),
        "extraction_both": int(valid and ex["actor"] == item["actor"] and ex["patient"] == item["patient"]),
        "extraction_last": int(valid and ex["actor"] == last and ex["patient"] == first),
        "extraction_exact": int(ex["exact_requested_format"]),
    }

def audit(directory):
    directory = Path(directory)
    items_list = json.loads((directory / "FROZEN_ITEMS.json").read_text(encoding="utf8"))
    ids = [x["id"] for x in items_list]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate gold item IDs")
    gold = {x["id"]: x for x in items_list}
    summary = {}
    seen_files = set()
    for unit, world, arm, seed in units():
        path = directory / "full_outputs" / (unit + ".json")
        obj = json.loads(path.read_text(encoding="utf8"))
        if (obj["world"], obj["arm"], obj["seed"]) != (world, arm, seed):
            raise ValueError("Unit provenance mismatch: " + unit)
        if len(obj["records"]) != 96:
            raise ValueError("Unexpected number of prompts: " + unit)
        seen = set()
        by_form = defaultdict(lambda: defaultdict(int))
        for record in obj["records"]:
            item_id = record["id"]
            if item_id in seen or item_id not in gold:
                raise ValueError("Duplicate or unknown prediction: " + item_id)
            seen.add(item_id)
            item = gold[item_id]
            if item["world"] != world or item["form"] not in FORMS:
                raise ValueError("World/form mismatch: " + item_id)
            values = score_one(item, record)
            by_form[item["form"]]["n"] += 1
            for key, val in values.items():
                by_form[item["form"]][key] += val
        expected = {x["id"] for x in items_list if x["world"] == world}
        if seen != expected:
            raise ValueError("Missing or extra event IDs for " + unit)
        if any(by_form[f]["n"] != 32 for f in FORMS):
            raise ValueError("Unexpected form denominator: " + unit)
        summary[unit] = {"world":world, "arm":arm, "seed":seed, "forms":{f:dict(by_form[f]) for f in FORMS}}
        seen_files.add(path.name)
    return {"status":"independent_frozen_record_rescore", "units":len(summary),
            "sentence_model_comparisons":sum(sum(z["n"] for z in u["forms"].values()) for u in summary.values()),
            "summary":summary,
            "limitations":["Reuses archived parsed outputs rather than reparsing raw token streams.",
                           "No model reruns, causal inference, real-language generalization or independent family sampling.",
                           "Repeated prompts and forms are clustered by participant pair and world."]}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("completed_adapter_diagnostic", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    result = audit(args.completed_adapter_diagnostic)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf8")
    print("Verified", result["units"], "units and", result["sentence_model_comparisons"], "sentence-model comparisons")
    if not args.output:
        print(payload)

if __name__ == "__main__":
    main()
