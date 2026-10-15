#!/usr/bin/env python3
"""Prepare an untouched QA-SRL Bank test queue for blinded human role review."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path
import tarfile

import data

ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "SEALED_TEST_CANDIDATES.json"
MANIFEST = ROOT / "SEALED_TEST_MANIFEST.json"
REVIEW = ROOT / "SEALED_TEST_HUMAN_REVIEW.csv"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select():
    split = json.loads((ROOT / "SPLITS.json").read_text())
    excluded = data.heldout_sentences()
    excluded |= {data.norm(item["sentence"]) for items in (split["train"], split["development"]) for item in items}
    choices = []
    with tarfile.open(data.SOURCE) as archive:
      for source_partition in ("dev", "test"):
        with gzip.GzipFile(fileobj=archive.extractfile(f"qasrl-v2/orig/{source_partition}.jsonl.gz")) as stream:
            for line in stream:
                record = json.loads(line)
                tokens = record["sentenceTokens"]
                sentence = " ".join(tokens)
                if not 5 <= len(tokens) <= 45 or data.norm(sentence) in excluded:
                    continue
                for predicate in record["verbEntries"].values():
                    actor, patient = [], []
                    for question in predicate["questionLabels"].values():
                        slots = question["questionSlots"]
                        if (question["isPassive"] or question["isNegated"] or
                                slots["wh"] != "who" or slots["prep"] != "_" or
                                slots["obj2"] != "_"):
                            continue
                        span = data.consensus_span(question)
                        if span is None or not 0 <= span[0] < span[1] <= len(tokens) or span[1]-span[0] > 8:
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
                    actor_text = " ".join(tokens[a[0]:a[1]])
                    patient_text = " ".join(tokens[p[0]:p[1]])
                    if data.norm(actor_text) == data.norm(patient_text):
                        continue
                    choices.append({"id": f"{record['sentenceId']}:{predicate['verbIndex']}",
                                    "source_partition": source_partition,
                                    "sentence_id": record["sentenceId"], "sentence": sentence,
                                    "verb_index": predicate["verbIndex"],
                                    "actor": actor_text, "patient": patient_text,
                                    "actor_question": actor[0][1], "patient_question": patient[0][1]})
    choices.sort(key=lambda item: hashlib.sha256(("sealed-20261013:"+item["id"]).encode()).hexdigest())
    chosen, sentences = [], set()
    for item in choices:
        if item["sentence_id"] in sentences:
            continue
        sentences.add(item["sentence_id"])
        chosen.append(item)
        if len(chosen) == 60:
            break
    assert len(chosen) == 60, (len(chosen), len(choices))
    return chosen, len(choices)


def main():
    for path in (TARGET, MANIFEST, REVIEW):
        if path.exists():
            raise FileExistsError(path)
    selected, candidates = select()
    TARGET.write_text(json.dumps(selected, indent=2) + "\n")
    with REVIEW.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("id", "sentence", "actor_question", "patient_question",
                                                   "proposed_actor", "proposed_patient", "reviewed_actor",
                                                   "reviewed_patient", "status", "notes"))
        writer.writeheader()
        for item in selected:
            writer.writerow({"id": item["id"], "sentence": item["sentence"],
                             "actor_question": item["actor_question"],
                             "patient_question": item["patient_question"],
                             "proposed_actor": item["actor"], "proposed_patient": item["patient"],
                             "reviewed_actor": "", "reviewed_patient": "", "status": "", "notes": ""})
    MANIFEST.write_text(json.dumps({
        "date": "2026-10-13", "status": "sealed against model evaluation; awaiting independent human role review",
        "source": "QA-SRL Bank 2.0 orig/dev and orig/test JSONL files",
        "selection": "Same direct-object consensus-span rules as development, <=45 sentence tokens, <=8-token role spans; exclude all QA-SRL Gold source sentences and all training/development sentences; SHA256-sort IDs and take 60 unique sentences. No model outputs inspected or generated on these items.",
        "candidate_count_before_selection": candidates,
        "selected_items": len(selected),
        "selected_items_sha256": sha(TARGET), "review_csv_sha256": sha(REVIEW),
        "split_sha256": sha(ROOT / "SPLITS.json"),
        "source_tar_sha256": sha(data.SOURCE),
        "selection_script_sha256": sha(Path(__file__)),
        "rule": "Do not evaluate any model on this set until the training objective, checkpoints, scoring code, and positive rule are frozen, and independent human role review is complete. Keep disagreements and exclusions. Changing labels after seeing predictions invalidates final confirmation."
    }, indent=2) + "\n")
    print("Prepared sealed review queue:", len(selected), "items from", candidates, "candidates", flush=True)


if __name__ == "__main__":
    main()
