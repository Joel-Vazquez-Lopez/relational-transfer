#!/usr/bin/env python3
"""Read-only audit of saved, non-sealed role-output experiments."""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NATURAL = ROOT / "natural_role_verifier_development_2026-10-13"
SYNTHETIC = ROOT / "role_slot_routing_fresh_confirmation_2026-10-13"


def norm(value: str | None) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", (value or "").casefold())).strip()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def natural_cell(start: str, dataset: str, arm: str = "role_first", stress: str = "normal") -> dict:
    path = NATURAL / f"EVAL_{start}_{arm}_{dataset}_{stress}.json"
    data = json.loads(path.read_text())
    assert data["protocol_sha256"] == sha(NATURAL / "PROTOCOL.json")
    assert data["checkpoint_sha256"] == sha(NATURAL / f"{start}_{arm}" / "adapter.pt")
    records = data["records"]
    expected_items = 40 if dataset == "development" else 51
    assert len(records) == 4 * expected_items
    assert len({(r["id"], r["role"], r["order"]) for r in records}) == len(records)
    cross = Counter()
    failure_kind = Counter()
    rescued = []
    harmed = []
    by_id = {}
    keyed = {}
    by_role_order = Counter()
    for r in records:
        requested = r["actor"] if r["role"] == "actor" else r["patient"]
        routed = norm(requested) == norm(r["gold"])
        direct = bool(r["correct"])
        assert bool(r["routed_correct"]) == routed
        cross[f"route_{int(routed)}_direct_{int(direct)}"] += 1
        by_role_order[f"{r['role']}/order{r['order']}/route"] += routed
        by_role_order[f"{r['role']}/order{r['order']}/direct"] += direct
        by_id.setdefault(r["id"], []).append((routed, direct))
        keyed[(r["id"], r["role"], r["order"])] = r
        if routed and not direct:
            answer = norm(r["answer"])
            other = r["patient"] if r["role"] == "actor" else r["actor"]
            if not r["parsed"]:
                kind = "malformed"
            elif answer == norm(other):
                kind = "answered_with_other_role_field"
            elif answer == norm(r["candidate_pair"][0]):
                kind = "answered_with_first_candidate_other_than_requested_field"
            elif answer == norm(r["candidate_pair"][1]):
                kind = "answered_with_second_candidate_other_than_requested_field"
            else:
                kind = "other_or_ambiguous_answer"
            failure_kind[kind] += 1
            rescued.append({"id": r["id"], "role": r["role"], "order": r["order"],
                            "gold": r["gold"], "actor_field": r["actor"],
                            "patient_field": r["patient"], "answer": r["answer"],
                            "raw": r["raw"]})
        elif direct and not routed:
            harmed.append({"id": r["id"], "role": r["role"], "order": r["order"],
                           "gold": r["gold"], "actor_field": r["actor"],
                           "patient_field": r["patient"], "answer": r["answer"],
                           "raw": r["raw"]})
    assert len(by_id) == expected_items
    joint = Counter()
    for values in by_id.values():
        assert len(values) == 4
        joint[f"route_{int(all(v[0] for v in values))}_direct_{int(all(v[1] for v in values))}"] += 1
    question_pairs = Counter()
    for id_ in by_id:
        for order in (0, 1):
            a = keyed[(id_, "actor", order)]
            p = keyed[(id_, "patient", order)]
            same_answer = norm(a["answer"]) == norm(p["answer"])
            stable_fields = norm(a["actor"]) == norm(p["actor"]) and norm(a["patient"]) == norm(p["patient"])
            both_fields_correct = bool(a["routed_correct"] and p["routed_correct"])
            question_pairs[f"same_answer_{int(same_answer)}_stable_fields_{int(stable_fields)}_correct_fields_{int(both_fields_correct)}"] += 1
    items = json.loads((NATURAL / "SPLITS.json").read_text())["development"] if dataset == "development" else json.loads((ROOT / "role_slot_routing_qasrl_gold_2026-10-13" / "ITEMS.json").read_text())
    item_lookup = {item["id"]: item for item in items}
    for group in (rescued, harmed):
        for ex in group[:8]:
            item = item_lookup[ex["id"]]
            ex["sentence"] = item["sentence"]
            ex["question"] = item["actor_question"] if ex["role"] == "actor" else item["patient_question"]
    return {"source": str(path.relative_to(ROOT)), "sha256": sha(path),
            "prompt_pair_table": dict(cross), "item_pair_table": dict(joint),
            "by_role_order": dict(by_role_order), "question_pair_behavior": dict(question_pairs),
            "rescued_failure_kinds": dict(failure_kind),
            "rescued_examples": rescued[:8], "harmed_examples": harmed[:8]}


def synthetic_cell(start: str, arm: str) -> dict:
    path = SYNTHETIC / f"EVAL_{start}_{arm}.json"
    data = json.loads(path.read_text())
    assert data["protocol_sha256"] == sha(SYNTHETIC / "PROTOCOL.json")
    protocol = json.loads((SYNTHETIC / "PROTOCOL.json").read_text())
    assert data["checkpoint_sha256"] == protocol["checkpoint_sha256"][f"{start}_{arm}"]
    records = data["records"]
    assert len(records) == 960
    by = {(r["event"], r["form"], r["wording"]): r for r in records}
    assert len(by) == len(records)
    pairs = Counter()
    for event in {r["event"] for r in records}:
        for form in ("passive", "subject_cleft", "object_cleft"):
            actor = by[(event, form, "actor_did")]
            patient = by[(event, form, "patient_verb")]
            direct = bool(actor["correct"] and patient["correct"])
            routed = norm(actor["actor"]) == norm(actor["gold"]) and norm(patient["patient"]) == norm(patient["gold"])
            pairs[f"route_{int(routed)}_direct_{int(direct)}"] += 1
    assert sum(pairs.values()) == 144
    return {"source": str(path.relative_to(ROOT)), "sha256": sha(path), "crossed_pair_table": dict(pairs)}


out = {"scope": "existing outputs only; no sealed-set access or new inference",
       "natural": {}, "synthetic": {}}
for dataset in ("development", "gold_exploratory"):
    for start in ("fresh1", "fresh3"):
        out["natural"][f"{dataset}/{start}/role_first"] = natural_cell(start, dataset)
        for arm in ("verifier_clean", "verifier_noisy"):
            for stress in ("normal", "swapped"):
                out["natural"][f"{dataset}/{start}/{arm}/{stress}"] = natural_cell(start, dataset, arm, stress)
for start in ("fresh1", "fresh3"):
    for arm in ("answer_first", "slot_first"):
        out["synthetic"][f"{start}/{arm}"] = synthetic_cell(start, arm)
dest = Path(__file__).with_name("AUDIT.json")
dest.write_text(json.dumps(out, indent=2) + "\n")
for key, cell in out["natural"].items():
    if key.endswith("role_first") or key.endswith("verifier_noisy/swapped"):
        print(key, cell["prompt_pair_table"], cell["item_pair_table"], cell["rescued_failure_kinds"])
for key, cell in out["synthetic"].items():
    print(key, cell["crossed_pair_table"])
