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
    base_keys = [(str(r["candidate_id"]), str(r["sent_id"]), str(r["predicate_token_id"])) for r in base]
    challenge_keys = [(str(r["candidate_id"]), str(p["sent_id"]), str(p["token_id"])) for r in challenge for p in r["target_predicates"]]
    base_by_id = {str(r["candidate_id"]): r for r in base}
    challenge_by_id = {str(r["candidate_id"]): r for r in challenge}
    if len(base_by_id) != len(base) or len(challenge_by_id) != len(challenge):
        raise AssertionError("Duplicate manifest candidate IDs")
    for name in ("annotator_A_template.csv", "annotator_B_template.csv"):
        check_key_set(name, base_keys, ("candidate_id", "source_sentence_id", "predicate_token_id"))
        for row in rows(name):
            for k, v in row.items():
                if k not in ("candidate_id", "source_sentence_id", "predicate_token_id") and v.strip():
                    raise AssertionError(f"{name}: unexpected prefilled human annotation {k}")
        result["checked"].append(name)
    check_key_set("adjudication_template.csv", [(k,) for k in base_by_id], ("candidate_id",))
    for row in rows("adjudication_template.csv"):
        if any(v.strip() for k, v in row.items() if k != "candidate_id"):
            raise AssertionError("Base adjudication template contains prefilled fields")
    result["checked"].append("adjudication_template.csv")
    for name in ("challenge_annotator_A.csv", "challenge_annotator_B.csv"):
        check_key_set(name, challenge_keys, ("candidate_id", "source_sentence_id", "predicate_token_id"))
        for row in rows(name):
            if any(v.strip() for k, v in row.items() if k not in ("review_row_id", "candidate_id", "source_sentence_id", "predicate_token_id")):
                raise AssertionError(f"{name}: prefilled human annotation")
            if not row["review_row_id"].strip():
                raise AssertionError(f"{name}: blank review row ID")
        result["checked"].append(name)
    expected_base_rights = [(str(r["document_id"]), str(r["genre"]), str(r["source_commit"])) for r in base]
    check_key_set("document_rights_review_template.csv", expected_base_rights, ("document_id", "genre", "pinned_source_commit"))
    expected_challenge_rights = [(str(r["candidate_id"]), str(r["document_id"]), str(r["genre"])) for r in challenge]
    check_key_set("challenge_rights_review.csv", expected_challenge_rights, ("candidate_id", "document_id", "genre"))
    result["checked"] += ["document_rights_review_template.csv", "challenge_rights_review.csv"]
    result["warnings"].append("No challenge adjudication form exists yet; 36 challenge predicate slots require separate adjudication.")
    result["warnings"].append("This audits only manifest-to-form consistency, not source corpus identity or human labels.")
    return result

if __name__ == "__main__":
    print(json.dumps(inspect(), indent=2))
