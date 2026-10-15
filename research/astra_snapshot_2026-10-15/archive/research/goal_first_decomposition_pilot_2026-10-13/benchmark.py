#!/usr/bin/env python3
"""Deterministic, controlled two-hop event/query benchmark for a small pilot."""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NAMES = tuple(f"n{i}" for i in range(16))
VERBS = ("called", "helped", "thanked", "followed")
DIRS = ("F", "R")  # F: known actor -> patient; R: known patient -> actor
SPLITS = {"train": (4000, ("FF", "RR"), 201),
          "development": (400, ("FF", "RR"), 202),
          "in_domain_test": (400, ("FF", "RR"), 203),
          "heldout_mixed": (400, ("FR", "RF"), 204)}


def text_event(actor: str, patient: str, verb: str, passive: bool) -> str:
    return (f"{patient} was {verb} by {actor} ." if passive
            else f"{actor} {verb} {patient} .")


def generate_one(rng: random.Random, pattern: str) -> dict:
    a, b, c, d, e, f, g = rng.sample(NAMES, 7)
    v1, v2 = rng.sample(VERBS, 2)
    first = (a, b) if pattern[0] == "F" else (b, a)
    second = (b, c) if pattern[1] == "F" else (c, b)
    facts = [(first[0], first[1], v1, bool(rng.randrange(2))),
             (second[0], second[1], v2, bool(rng.randrange(2))),
             (d, e, v1, bool(rng.randrange(2))),
             (f, g, v2, bool(rng.randrange(2)))]
    rng.shuffle(facts)
    inner = (f"the person whom {a} {v1}" if pattern[0] == "F"
             else f"the person who {v1} {a}")
    question = (f"Who did {inner} {v2} ?" if pattern[1] == "F"
                else f"Who {v2} {inner} ?")
    sentence = " ".join(text_event(*fact) for fact in facts)
    return {"facts": facts, "sentence": sentence, "question": question,
            "start": a, "middle": b, "answer": c, "verbs": [v1, v2],
            "operations": [pattern[0], pattern[1]], "pattern": pattern}


def solve(row: dict) -> str:
    current = row["start"]
    for verb, direction in zip(row["verbs"], row["operations"]):
        matches = [(actor, patient) for actor, patient, predicate, _ in row["facts"]
                   if predicate == verb and (actor if direction == "F" else patient) == current]
        assert len(matches) == 1, row
        actor, patient = matches[0]
        current = patient if direction == "F" else actor
    return current


def signature(row: dict) -> str:
    return hashlib.sha256(json.dumps([row["sentence"], row["question"]]).encode()).hexdigest()


def make() -> dict:
    seen = set()
    result = {}
    for split, (count, patterns, seed) in SPLITS.items():
        rng = random.Random(seed)
        rows = []
        while len(rows) < count:
            row = generate_one(rng, patterns[len(rows) % len(patterns)])
            sig = signature(row)
            if sig in seen:
                continue
            assert solve(row) == row["answer"]
            seen.add(sig)
            row["id"] = f"{split}-{len(rows):04d}"
            rows.append(row)
        result[split] = rows
    assert len(seen) == sum(len(rows) for rows in result.values())
    assert not {x["pattern"] for x in result["train"]} & {x["pattern"] for x in result["heldout_mixed"]}
    return result


def audit(data: dict) -> dict:
    out = {}
    for split, rows in data.items():
        # These are deliberately weak response-position/name heuristics, not trained comparators.
        out[split] = {
            "n": len(rows), "patterns": dict(Counter(r["pattern"] for r in rows)),
            "answer_in_question": sum(r["answer"] in r["question"].split() for r in rows),
            "answer_is_start": sum(r["answer"] == r["start"] for r in rows),
            "first_fact_first_name": sum(r["answer"] == r["facts"][0][0] for r in rows),
            "first_fact_second_name": sum(r["answer"] == r["facts"][0][1] for r in rows),
            "last_fact_last_name": sum(r["answer"] == r["facts"][-1][1] for r in rows),
            "answer_name_top_frequency": Counter(r["answer"] for r in rows).most_common(1)[0],
        }
    return out


if __name__ == "__main__":
    if (ROOT / "DATA.json").exists():
        raise FileExistsError(ROOT / "DATA.json")
    data = make()
    (ROOT / "DATA.json").write_text(json.dumps(data, indent=2) + "\n")
    (ROOT / "DATA_AUDIT.json").write_text(json.dumps(audit(data), indent=2) + "\n")
    print(json.dumps(audit(data), indent=2))
