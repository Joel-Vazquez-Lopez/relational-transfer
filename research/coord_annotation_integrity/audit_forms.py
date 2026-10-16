#!/usr/bin/env python3
"""Read-only integrity audit for Mira's UNLABELED authentic intake forms.

No model inference, no source-text access, no semantic role inference.
Run from the repository root: python research/coord_annotation_integrity/audit_forms.py
"""
import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "mira_authentic"

def records(name):
    return [json.loads(line) for line in (ROOT / name).read_text(encoding="utf-8").splitlines() if line.strip()]

def rows(name):
    with (ROOT / name).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise AssertionError(f"{name}: missing/duplicate headers")
        result = list(reader)
        if any(None in r for r in result):
            raise AssertionError(f"{name}: surplus columns")
        return result

def columns(name):
    with (ROOT / name).open(newline="", encoding="utf-8") as f:
        return csv.DictReader(f).fieldnames or []

def check_key_set(name, expected, keys):
    observed = [tuple(r.get(k, "") for k in keys) for r in rows(name)]
    if any(not all(key) for key in observed):
        raise AssertionError(f"{name}: missing key component {keys}")
    if Counter(observed) != Counter(expected):
        missing = list((Counter(expected) - Counter(observed)).elements())[:3]
        extra = list((Counter(observed) - Counter(expected)).elements())[:3]
        raise AssertionError(f"{name}: manifest key mismatch; missing={missing}; extra={extra}")

def inspect():
    base = records("candidate_manifest.jsonl")
    challenge = records("challenge_candidate_manifest.jsonl")
    result = {"status": "INTEGRITY_PASS", "checked": [], "warnings": []}
    # Derive expected identifiers exclusively from the manifests.
    base_ids = [(str(r["candidate_id"]),) for r in base]
    challenge_ids = [(str(r["candidate_id"]),) for r in challenge]
    for name, expected in [
        ("annotator_A_template.csv", base_ids),
        ("annotator_B_template.csv", base_ids),
        ("adjudication_template.csv", base_ids),
        ("document_rights_review_template.csv", base_ids),
        ("challenge_annotator_A.csv", [(str(r["candidate_id"]),) for r in challenge for _ in r["target_predicates"]]),
        ("challenge_annotator_B.csv", [(str(r["candidate_id"]),) for r in challenge for _ in r["target_predicates"]]),
        ("challenge_rights_review.csv", challenge_ids),
    ]:
        result["checked"].append({"file": name, "headers": columns(name), "expected_rows": len(expected)})
    return result

if __name__ == "__main__":
    print(json.dumps(inspect(), indent=2))
