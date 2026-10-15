#!/usr/bin/env python3
"""Compare off-format prompting with each checkpoint's native-format score."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NATIVE = ROOT.parent / "role_slot_routing_fresh_confirmation_2026-10-13"
FORMS = ("active", "passive", "subject_cleft", "object_cleft")
STARTS = ("fresh1", "fresh3")
ARMS = ("answer_first", "slot_first")
cfg = json.loads((ROOT / "PROTOCOL.json").read_text())
protocol_hash = hashlib.sha256((ROOT / "PROTOCOL.json").read_bytes()).hexdigest()
out = {"protocol_sha256": protocol_hash, "arms": {}}


def leading(raw):
    match = re.match(r"^\s*([A-Za-z]+)", raw)
    return match.group(1) if match else None


def tagged(raw, tag):
    match = re.search(r"\b" + re.escape(tag) + r":\s*([A-Za-z]+)", raw)
    return match.group(1) if match else None


def field(record, prompt_arm, kind):
    raw = record["raw"]
    if kind == "answer":
        return leading(raw) if prompt_arm == "answer_first" else tagged(raw, "Answer")
    if kind == "actor":
        return tagged(raw, "Actor") if prompt_arm == "answer_first" else leading(raw)
    if kind == "patient":
        return tagged(raw, "Patient")
    raise ValueError(kind)

for start in STARTS:
    for arm in ARMS:
        key = f"{start}_{arm}"
        cp_hash = cfg["checkpoint_sha256"][key]
        arm_result = {}
        for mode, path in (
            ("native", NATIVE / f"EVAL_{key}.json"),
            ("cross", ROOT / f"EVAL_{key}_cross.json"),
        ):
            data = json.loads(path.read_text())
            assert data["checkpoint_sha256"] == cp_hash
            assert data["protocol_sha256"] == (cfg["prior_protocol_sha256"] if mode == "native" else protocol_hash)
            records = data["records"]
            by = {(r["event"], r["form"], r["wording"]): r for r in records}
            assert len(records) == len(by) == 960
            events = sorted({r["event"] for r in records})
            assert len(events) == 48
            prompt_arm = arm if mode == "native" else ("slot_first" if arm == "answer_first" else "answer_first")
            def per_form(score_fn):
                return {form: sum(score_fn(by[(event, form, "actor_did")]) and
                                  score_fn(by[(event, form, "patient_verb")])
                                  for event in events) for form in FORMS}
            forms = {form: sum(by[(event, form, "actor_did")]["correct"] and
                               by[(event, form, "patient_verb")]["correct"]
                               for event in events) for form in FORMS}
            first_or_tagged = per_form(lambda r: field(r, prompt_arm, "answer") == r["gold"])
            routed = per_form(lambda r: field(r, prompt_arm, r["role"]) == r["gold"])
            arm_result[mode] = {
                "crossed_joint_by_form": forms,
                "primary_crossed_joint": sum(forms[form] for form in FORMS[1:]),
                "prefix_tolerant_answer_joint_by_form": first_or_tagged,
                "prefix_tolerant_answer_primary": sum(first_or_tagged[form] for form in FORMS[1:]),
                "routed_from_generated_fields_joint_by_form": routed,
                "routed_primary": sum(routed[form] for form in FORMS[1:]),
                "valid_answers": sum(r["valid"] for r in records),
                "slot_joint_outputs": sum(r["slot_joint_correct"] for r in records),
                "answer_field_present": sum(field(r, prompt_arm, "answer") is not None for r in records),
                "actor_field_present": sum(field(r, prompt_arm, "actor") is not None for r in records),
                "patient_field_present": sum(field(r, prompt_arm, "patient") is not None for r in records),
                "raw_output_examples": [r["raw"] for r in records[:3]],
            }
        arm_result["cross_minus_native"] = arm_result["cross"]["primary_crossed_joint"] - arm_result["native"]["primary_crossed_joint"]
        out["arms"][key] = arm_result

(ROOT / "ANALYSIS.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
