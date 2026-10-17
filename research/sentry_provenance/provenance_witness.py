#!/usr/bin/env python3
"""Pre-inference event-source *prediction separation* certificate.

Input is metadata only (NO document text, no test gold, no LLM inference).
Rivals are explicitly restricted policies, NOT alleged internal LLM algorithms.
Observed error provenance is optional and scored only if independently VERIFIED.
"""
import argparse
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

POLICIES = ("first_event", "previous_event", "nearest_other", "lexically_closest_other", "only_other")


def validate(r):
    if set(r) - {"case", "events", "target", "observed_source", "provenance_status"}:
        raise ValueError("Unknown metadata field; no source text allowed")
    if not isinstance(r.get("case"), str) or not r["case"]:
        raise ValueError("Missing case ID")
    events = r.get("events")
    if not isinstance(events, list) or len(events) < 2:
        raise ValueError("Need two or more source event candidates")
    ids = set()
    ranks = set()
    for e in events:
        if not isinstance(e, dict) or set(e) != {"id", "order", "center", "overlap"}:
            raise ValueError("Events require ONLY id/order/center/overlap metadata")
        if not isinstance(e["id"], str) or not e["id"] or e["id"] in ids:
            raise ValueError("Duplicate or invalid event ID")
        if type(e["order"]) is not int or e["order"] < 0 or e["order"] in ranks:
            raise ValueError("Invalid/disputed source order")
        if type(e["center"]) not in (int, float) or type(e["overlap"]) not in (int, float):
            raise ValueError("Positions and lexical overlap must be numeric")
        if not (float('-inf') < e["center"] < float('inf') and 0 <= e["overlap"] <= 1):
            raise ValueError("Nonfinite positions or overlap outside [0,1]")
        ids.add(e["id"])
        ranks.add(e["order"])
    if ranks != set(range(len(events))):
        raise ValueError("Order must be a complete 0..N-1 permutation")
    if r.get("target") not in ids:
        raise ValueError("Target event not among candidates")
    status = r.get("provenance_status", "unverified")
    if status not in ("unverified", "ambiguous", "verified"):
        raise ValueError("Unknown provenance status")
    if status != "verified" and r.get("observed_source") is not None:
        raise ValueError("Cannot assign a source under ambiguous or unverified provenance")
    if status == "verified" and r.get("observed_source") not in ids:
        raise ValueError("Verified observed source must be an event in this exact context")


def unique_best(candidates, key, reverse=False):
    if not candidates:
        return None
    scored = [(key(e), e["id"]) for e in candidates]
    best = (max if reverse else min)(v for v, _ in scored)
    choices = [eid for val, eid in scored if val == best]
    return choices[0] if len(choices) == 1 else None


def predictions(r):
    validate(r)
    events = sorted(r["events"], key=lambda e: e["order"])
    target = next(e for e in events if e["id"] == r["target"])
    other = [e for e in events if e["id"] != target["id"]]
    before = [e for e in other if e["order"] < target["order"]]
    return {
        "first_event": events[0]["id"],
        "previous_event": unique_best(before, lambda e: e["order"], reverse=True),
        "nearest_other": unique_best(other, lambda e: abs(e["center"]-target["center"])),
        "lexically_closest_other": unique_best(other, lambda e: e["overlap"], reverse=True),
        "only_other": other[0]["id"] if len(events) == 2 else None,
    }


def audit(rows):
    if not isinstance(rows, list):
        raise ValueError("Top-level input must be a list")
    if len({r.get("case") for r in rows}) != len(rows):
        raise ValueError("Duplicate case ID")
    certs = [(r, predictions(r)) for r in rows]
    comparisons = {}
    for a, b in combinations(POLICIES, 2):
        common = [(x[a], x[b]) for _, x in certs if x[a] is not None and x[b] is not None]
        comparisons[a+" vs "+b] = {"jointly_defined": len(common),
                                    "disagreements": sum(v != w for v,w in common),
                                    "separating_cases": [r["case"] for r,p in certs
                                                         if p[a] is not None and p[b] is not None and p[a] != p[b]]}
    provenance = Counter(r.get("provenance_status", "unverified") for r,_ in certs)
    verified = [(r,p) for r,p in certs if r.get("provenance_status") == "verified"]
    fits = {k: {"scored_verified_rows": sum(p[k] is not None for r,p in verified),
                "matching_source": sum(p[k] == r["observed_source"] for r,p in verified if p[k] is not None)}
            for k in POLICIES}
    return {"kind": "metadata-only-preinference-restricted-rival-certificate",
            "cases": len(rows), "provenance_status_counts": dict(provenance),
            "policy_predictions": [{"case":r["case"], "target":r["target"], "predictions":p} for r,p in certs],
            "pairwise_distinguishability": comparisons,
            "observed_source_fit_if_independently_verified": fits,
            "claim_ceiling": "A nonzero disagreement is an opportunity for a bounded test, NOT proof of a semantic circuit. Zero disagreement means no witness for that pair of restricted algorithms; no new inference is justified solely to distinguish them."
            }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("metadata_json", type=Path, help="Local metadata ONLY; never add source text")
    args = ap.parse_args()
    print(json.dumps(audit(json.loads(args.metadata_json.read_text(encoding="utf-8"))), indent=2))


if __name__ == "__main__":
    main()
