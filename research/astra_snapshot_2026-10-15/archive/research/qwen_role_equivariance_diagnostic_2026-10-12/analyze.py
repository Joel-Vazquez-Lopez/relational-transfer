#!/usr/bin/env python3
"""Descriptive analysis of the five already frozen Qwen evaluations.

This script does not load a model or use the reserved human-review challenge.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "PROTOCOL.json").read_text())
FORMS = CONFIG["forms"]
MODELS = [*CONFIG["models"], "untouched_qwen15"]
ACTOR_WORDINGS = ("actor_performed", "actor_acted")
PATIENT_WORDINGS = ("patient_verb", "patient_affected")


def load(name: str) -> dict:
    path = ROOT / f"{name}.json"
    data = json.loads(path.read_text())
    assert data["model"] == name
    assert data["protocol_sha256"] == hashlib.sha256((ROOT / "PROTOCOL.json").read_bytes()).hexdigest()
    assert len(data["records"]) == 720
    return data


def index(records: list[dict], omitted: bool) -> dict:
    return {(r["id"], r["form"] if not omitted else "omitted", r["wording"]): r
            for r in records if r["omitted"] == omitted}


def first_name(row: dict) -> str:
    a, b = CONFIG["pairs"][row["pair_index"]]
    actor = a if row["direction"] == 0 else b
    patient = b if row["direction"] == 0 else a
    return patient if row["form"] in ("passive", "object_cleft") else actor


def local_predicate_subject(row: dict) -> str:
    a, b = CONFIG["pairs"][row["pair_index"]]
    actor = a if row["direction"] == 0 else b
    patient = b if row["direction"] == 0 else a
    # In the object cleft, the embedded clause subject is the actor.
    # In the passive, the grammatical subject is the patient.
    return patient if row["form"] == "passive" else actor


def summarize(name: str) -> dict:
    data = load(name)
    rows = [r for r in data["records"] if not r["omitted"]]
    present = index(data["records"], False)
    omitted = index(data["records"], True)
    assert len(present) == 576 and len(omitted) == 144
    form_role = {}
    wording = {}
    candidate = {}
    local_subject = {}
    for form in FORMS:
        form_rows = [r for r in rows if r["form"] == form]
        local_subject[form] = {
            "local_subject_answers": sum(r["answer"] == local_predicate_subject(r) for r in form_rows),
            "first_mentioned_answers": sum(r["answer"] == first_name(r) for r in form_rows),
            "n": len(form_rows),
            "by_pair": [sum(r["answer"] == local_predicate_subject(r)
                            for r in form_rows if r["pair_index"] == i)
                        for i in range(len(CONFIG["pairs"]))],
        }
        for role in ("actor", "patient"):
            sub = [r for r in rows if r["form"] == form and r["role"] == role]
            form_role[f"{form}/{role}"] = {
                "correct": sum(r["correct"] for r in sub),
                "first_name_answers": sum(r["answer"] == first_name(r) for r in sub),
                "n": len(sub),
            }
        for question in CONFIG["questions"]:
            sub = [r for r in rows if r["form"] == form and r["wording"] == question]
            wording[f"{form}/{question}"] = {"correct": sum(r["correct"] for r in sub), "n": len(sub)}
        for actor_q, patient_q in zip(ACTOR_WORDINGS, PATIENT_WORDINGS):
            ds, lifts = [], []
            both = 0
            for event_id in {r["id"] for r in rows}:
                actor = present[(event_id, form, actor_q)]
                patient = present[(event_id, form, patient_q)]
                actor_null = omitted[(event_id, "omitted", actor_q)]
                patient_null = omitted[(event_id, "omitted", patient_q)]
                # The same two candidate names appear under both questions.
                # A positive contrast means actor wording favors the true actor
                # more than patient wording does; it need not mean either direct
                # answer is correct. The omitted contrast captures prompt/name bias.
                d = actor["actor_margin"] - patient["actor_margin"]
                d0 = actor_null["actor_margin"] - patient_null["actor_margin"]
                ds.append(d)
                lifts.append(d - d0)
                both += actor["correct"] and patient["correct"]
            candidate[f"{form}/{actor_q}+{patient_q}"] = {
                "contrast_positive": sum(x > 0 for x in ds),
                "context_lift_positive": sum(x > 0 for x in lifts),
                "both_direct_answers_correct": both,
                "mean_contrast": sum(ds) / len(ds),
                "mean_context_lift": sum(lifts) / len(lifts),
                "n": len(ds),
            }
    # Less strict descriptive measure: both roles correct with one explicit
    # wording in every form. This was not the frozen primary endpoint.
    explicit_joint = 0
    for event_id in {r["id"] for r in rows}:
        explicit_joint += all(present[(event_id, form, q)]["correct"]
                              for form in FORMS for q in ("actor_performed", "patient_verb"))
    return {"model": name, "summary": data["summary"], "form_role": form_role,
            "wording": wording, "candidate_contrast": candidate,
            "local_subject_vs_first_mention": local_subject,
            "explicit_wording_joint_events_exploratory": explicit_joint}


def paired_changes(source: str, natural: str) -> dict:
    a = index(load(source)["records"], False)
    b = index(load(natural)["records"], False)
    assert set(a) == set(b)
    by_cell = defaultdict(lambda: {"source_correct": 0, "natural_correct": 0,
                                   "gained": 0, "lost": 0, "n": 0})
    for key in a:
        left, right = a[key], b[key]
        for cell in (f"{left['form']}/{left['role']}", f"{left['form']}/{left['wording']}"):
            x = by_cell[cell]
            x["source_correct"] += left["correct"]
            x["natural_correct"] += right["correct"]
            x["gained"] += not left["correct"] and right["correct"]
            x["lost"] += left["correct"] and not right["correct"]
            x["n"] += 1
    return {"source": source, "natural": natural, "cells": dict(sorted(by_cell.items()))}


def main() -> None:
    result = {"status": "post_hoc_descriptive_analysis_of_frozen_exploratory_test",
              "models": {name: summarize(name) for name in MODELS},
              "matched_changes": [paired_changes("fresh1_source", "fresh1_natural"),
                                  paired_changes("fresh3_source", "fresh3_natural")]}
    (ROOT / "ANALYSIS.json").write_text(json.dumps(result, indent=2) + "\n")
    for name, x in result["models"].items():
        print(name, "joint", x["summary"]["joint_events"],
              "explicit_joint", x["explicit_wording_joint_events_exploratory"],
              "correct", sum(v["correct"] for v in x["form_role"].values()),
              "/576")
        for form in FORMS:
            parts = [x["form_role"][f"{form}/{role}"]["correct"] for role in ("actor", "patient")]
            print(" ", form, "actor/patient", *parts, "/72 each")
    print("Saved ANALYSIS.json")


if __name__ == "__main__":
    main()
