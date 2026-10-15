#!/usr/bin/env python3
"""Surface-only symbolic parser/executor to verify task solvability."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NAME = r"n\d+"
VERB = r"called|helped|thanked|followed"


def parse_facts(sentence: str) -> list[tuple[str, str, str]]:
    out = []
    for part in sentence.split(" ."):
        part = part.strip()
        if not part:
            continue
        passive = re.fullmatch(rf"({NAME}) was ({VERB}) by ({NAME})", part)
        if passive:
            patient, verb, actor = passive.groups()
        else:
            active = re.fullmatch(rf"({NAME}) ({VERB}) ({NAME})", part)
            assert active, part
            actor, verb, patient = active.groups()
        out.append((actor, patient, verb))
    assert len(out) == 4
    return out


def parse_query(question: str) -> tuple[str, str, str, str, str]:
    inner_f = re.search(rf"the person whom ({NAME}) ({VERB})", question)
    inner_r = re.search(rf"the person who ({VERB}) ({NAME})", question)
    assert bool(inner_f) != bool(inner_r), question
    if inner_f:
        start, verb1 = inner_f.groups()
        direction1 = "F"
    else:
        verb1, start = inner_r.groups()  # type: ignore[union-attr]
        direction1 = "R"
    if question.startswith("Who did "):
        outer = re.fullmatch(rf"Who did .+ ({VERB}) \?", question)
        assert outer
        verb2, direction2 = outer.group(1), "F"
    else:
        outer = re.fullmatch(rf"Who ({VERB}) .+ \?", question)
        assert outer
        verb2, direction2 = outer.group(1), "R"
    return start, verb1, direction1, verb2, direction2


def solve(sentence: str, question: str) -> tuple[str, str]:
    facts = parse_facts(sentence)
    start, v1, d1, v2, d2 = parse_query(question)
    current = start
    middle = None
    for i, (verb, direction) in enumerate(((v1, d1), (v2, d2))):
        matches = [(a, p) for a, p, v in facts
                   if v == verb and (a if direction == "F" else p) == current]
        assert len(matches) == 1, (sentence, question)
        a, p = matches[0]
        current = p if direction == "F" else a
        if i == 0:
            middle = current
    assert middle is not None
    return middle, current


if __name__ == "__main__":
    data = json.loads((ROOT / "DATA.json").read_text())
    summary = {}
    for split, rows in data.items():
        correct = sum(solve(r["sentence"], r["question"]) == (r["middle"], r["answer"])
                      for r in rows)
        summary[split] = {"correct": correct, "total": len(rows)}
        assert correct == len(rows)
    (ROOT / "SYMBOLIC_AUDIT.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
