#!/usr/bin/env python3
"""Pure-stdlib observational equivalence audit for positional heuristics."""
import argparse
import json
from collections import defaultdict

def rules(row):
    first, last = row["participant_first"], row["participant_last"]
    role, subject = row["question_role"], row.get("grammatical_subject")
    if first == last or role not in ("actor", "patient"):
        raise ValueError("Distinct participants and actor/patient question required")
    other = last if subject == first else first if subject == last else None
    return {
        "first": first, "last": last,
        "subject": subject if subject in (first, last) else None,
        "position_by_question": first if role == "actor" else last,
        "reverse_position_by_question": last if role == "actor" else first,
        "subject_by_question": subject if role == "actor" else other,
        "reverse_subject_by_question": other if role == "actor" else subject,
    }

def audit(rows):
    if not rows:
        raise ValueError("Empty design")
    pred = [rules(r) for r in rows]
    names = list(pred[0])
    signatures = {name: tuple(p[name] for p in pred) for name in names}
    groups = defaultdict(list)
    for name, signature in signatures.items():
        groups[signature].append(name)
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i+1:]
             if signatures[a] != signatures[b]]
    remaining = set(pairs)
    chosen = []
    while remaining:
        gains = [(sum(pred[i][a] != pred[i][b] for a,b in remaining), -i, i)
                 for i in range(len(rows)) if i not in chosen]
        gain, _, idx = max(gains)
        if gain == 0:
            raise RuntimeError("Cannot separate signatures")
        chosen.append(idx)
        remaining = {(a,b) for a,b in remaining if pred[idx][a] == pred[idx][b]}
    return {"n_rows": len(rows), "n_rules": len(names),
            "identifiable_classes": len(groups),
            "equivalence_classes": sorted([sorted(g) for g in groups.values()]),
            "greedy_discriminating_row_indices": chosen,
            "pairwise_disagreement_counts": {
                a + "__" + b: sum(x != y for x,y in zip(signatures[a],signatures[b]))
                for a,b in pairs}}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl")
    args = parser.parse_args()
    with open(args.jsonl, encoding="utf8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    print(json.dumps(audit(rows), indent=2))
