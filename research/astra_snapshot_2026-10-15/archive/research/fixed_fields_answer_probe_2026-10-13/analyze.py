#!/usr/bin/env python3
"""Summarize adaptive answer-continuation cells without changing prior metrics."""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
cfg = json.loads((ROOT / "PROTOCOL.json").read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def norm(value):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", (value or "").casefold())).strip()


def cell(start, tag, fields):
    path = ROOT / f"EVAL_{start}_{tag}_{fields}.json"
    data = json.loads(path.read_text())
    assert data["protocol_sha256"] == sha(ROOT / "PROTOCOL.json")
    assert data["checkpoint_sha256"] == cfg["checkpoint_sha256"][start]
    rows = data["records"]
    by = {(r["id"], r["role"], r["order"]): r for r in rows}
    assert len(rows) == len(by) == 160
    ids = {r["id"] for r in rows}
    assert len(ids) == 40
    correct_fields = [r for r in rows if r["requested_field_correct"]]
    incorrect_fields = [r for r in rows if not r["requested_field_correct"]]
    return {"exact": sum(bool(r["correct"]) for r in rows),
            "four_correct": sum(all(by[(id_, role, order)]["correct"]
                                    for role in ("actor", "patient") for order in (0, 1)) for id_ in ids),
            "matches_cached_full_answer": sum(bool(r["matches_cached_full_answer"]) for r in rows),
            "requested_field_correct": len(correct_fields),
            "answer_matches_requested_field": sum(bool(r["answer_matches_requested_field"]) for r in rows),
            "answer_matches_other_field": sum(bool(r["answer_matches_other_field"]) for r in rows),
            "exact_when_requested_field_correct": sum(bool(r["correct"]) for r in correct_fields),
            "exact_when_requested_field_wrong": sum(bool(r["correct"]) for r in incorrect_fields),
            "requested_field_follow_when_correct": sum(bool(r["answer_matches_requested_field"]) for r in correct_fields),
            "requested_field_follow_when_wrong": sum(bool(r["answer_matches_requested_field"]) for r in incorrect_fields)}, by


out = {"protocol_sha256": sha(ROOT / "PROTOCOL.json"), "cells": {}, "paired": {}}
by_cache = {}
for start in cfg["starts"]:
    for tag in cfg["tags"]:
        for fields in cfg["fields"]:
            key = f"{start}/{tag}/{fields}"
            out["cells"][key], by_cache[key] = cell(start, tag, fields)
for start in cfg["starts"]:
    for fields in cfg["fields"]:
        a = by_cache[f"{start}/ordinary/{fields}"]
        b = by_cache[f"{start}/tagged/{fields}"]
        rescue = sum(not row["correct"] and b[key]["correct"] for key, row in a.items())
        harm = sum(row["correct"] and not b[key]["correct"] for key, row in a.items())
        changed = sum(norm(row["answer"]) != norm(b[key]["answer"]) for key, row in a.items())
        out["paired"][f"{start}/{fields}/tagged-vs-ordinary"] = {
            "rescue": rescue, "harm": harm, "net": rescue - harm, "answer_changed": changed}
    for tag in cfg["tags"]:
        a = by_cache[f"{start}/{tag}/original"]
        b = by_cache[f"{start}/{tag}/swapped"]
        changed = sum(norm(row["answer"]) != norm(b[key]["answer"]) for key, row in a.items())
        out["paired"][f"{start}/{tag}/swapped-vs-original"] = {"answer_changed": changed}
assert all(out["cells"][f"{start}/ordinary/original"]["matches_cached_full_answer"] == 160
           for start in cfg["starts"]), "Textual prefill did not replicate native answer generation"
(ROOT / "ANALYSIS.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
