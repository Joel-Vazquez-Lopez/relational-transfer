#!/usr/bin/env python3
"""Extract sentence-disjoint, direct-object QA-SRL Bank training/development items."""
from __future__ import annotations

from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
import tarfile

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
SOURCE = PROJECT / "llm_role_natural_repair" / "data" / "qasrl-v2.tar"
GOLD_SENTENCES = PROJECT / "llm_role_confirmation_v1" / "qasrl-gs" / "data" / "sentences"
PARTS = ("wikinews.dev", "wikinews.test", "wikipedia.dev", "wikipedia.test")


def norm(text):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.casefold())).strip()


def heldout_sentences():
    result = set()
    for part in PARTS:
        with (GOLD_SENTENCES / f"{part}.full.csv").open() as file:
            result.update(norm(row["sentence"]) for row in csv.DictReader(file))
    return result


def consensus_span(question):
    counts = Counter(tuple(span) for judgment in question["answerJudgments"]
                     if judgment["isValid"] for span in judgment["spans"])
    if not counts:
        return None
    span, n = counts.most_common(1)[0]
    if n < 2 or any(count >= 2 and other != span for other, count in counts.items()):
        return None
    return span


def extract():
    excluded = heldout_sentences()
    result = []
    with tarfile.open(SOURCE) as archive:
        with gzip.GzipFile(fileobj=archive.extractfile("qasrl-v2/orig/train.jsonl.gz")) as stream:
            for line in stream:
                record = json.loads(line)
                tokens = record["sentenceTokens"]
                sentence = " ".join(tokens)
                if not 5 <= len(tokens) <= 25 or norm(sentence) in excluded:
                    continue
                for predicate in record["verbEntries"].values():
                    actor, patient = [], []
                    for question in predicate["questionLabels"].values():
                        slots = question["questionSlots"]
                        if (question["isPassive"] or question["isNegated"] or
                                slots["wh"] != "who" or slots["prep"] != "_" or
                                slots["obj2"] != "_"):
                            continue
                        span = consensus_span(question)
                        if span is None or not 0 <= span[0] < span[1] <= len(tokens) or span[1] - span[0] > 4:
                            continue
                        entry = (span, question["questionString"])
                        if slots["subj"] == "_" and slots["obj"] == "someone":
                            actor.append(entry)
                        if slots["subj"] == "someone" and slots["obj"] == "_":
                            patient.append(entry)
                    if len(actor) != 1 or len(patient) != 1:
                        continue
                    a, p = actor[0][0], patient[0][0]
                    if a[0] < p[1] and p[0] < a[1]:
                        continue
                    actor_text, patient_text = " ".join(tokens[a[0]:a[1]]), " ".join(tokens[p[0]:p[1]])
                    if norm(actor_text) == norm(patient_text):
                        continue
                    item = {"id": f"{record['sentenceId']}:{predicate['verbIndex']}",
                            "sentence_id": record["sentenceId"], "sentence": sentence,
                            "verb_index": predicate["verbIndex"],
                            "actor": actor_text, "patient": patient_text,
                            "actor_question": actor[0][1], "patient_question": patient[0][1]}
                    result.append(item)
    unique = {item["id"]: item for item in result}
    assert len(unique) == len(result)
    return result


def split():
    items = extract()
    groups = {}
    for item in items:
        groups.setdefault(item["sentence_id"], []).append(item)
    group_ids = sorted(groups, key=lambda value: hashlib.sha256(("20261013:" + value).encode()).hexdigest())
    train, development = [], []
    for group in group_ids:
        if len(train) < 160:
            train.extend(groups[group])
        elif len(development) < 40:
            development.extend(groups[group])
        else:
            break
    assert len(train) >= 160 and len(development) >= 40
    assert not {x["sentence_id"] for x in train} & {x["sentence_id"] for x in development}
    assert not {norm(x["sentence"]) for x in train} & {norm(x["sentence"]) for x in development}
    return {"train": train, "development": development,
            "source_candidates": len(items), "heldout_gold_sentences": len(heldout_sentences())}


if __name__ == "__main__":
    result = split()
    print(json.dumps({key: len(value) if isinstance(value, list) else value
                      for key, value in result.items()}, indent=2))
