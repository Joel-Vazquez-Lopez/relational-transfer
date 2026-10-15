"""Prepare auditable authentic-sentence candidates from QA-SRL Gold."""
from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "qasrl-gs" / "data"
PARTS = ("wikinews.dev", "wikinews.test", "wikipedia.dev", "wikipedia.test")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract() -> list[dict]:
    result = []
    for part in PARTS:
        sent_path = DATA / "sentences" / f"{part}.full.csv"
        gold_path = DATA / "gold" / f"{part}.gold.csv"
        sentences = {row["qasrl_id"]: row["sentence"]
                     for row in csv.DictReader(sent_path.open())}
        actors, patients = defaultdict(list), defaultdict(list)
        for row in csv.DictReader(gold_path.open()):
            if (row["wh"] != "who" or row["is_passive"] != "False" or
                    row["is_negated"] != "False" or "~!~" in row["answer"]):
                continue
            key = (row["qasrl_id"], row["verb_idx"])
            if not row["subj"]:
                actors[key].append(row)
            if row["subj"] == "someone" and not row["obj"]:
                patients[key].append(row)
        for key in sorted(actors.keys() & patients.keys()):
            actor_answers = {row["answer"] for row in actors[key]}
            patient_answers = {row["answer"] for row in patients[key]}
            if len(actor_answers) != 1 or len(patient_answers) != 1:
                continue
            actor, patient = next(iter(actor_answers)), next(iter(patient_answers))
            sentence = sentences[key[0]]
            if (actor == patient or len(sentence.split()) > 45 or "[" in sentence or
                    not all(1 <= len(x.split()) <= 8 and x in sentence
                            for x in (actor, patient))):
                continue
            a = actors[key][0]
            p = patients[key][0]
            result.append({"id": f"{part}:{key[0]}:{key[1]}", "partition": part,
                           "qasrl_id": key[0], "verb_idx": int(key[1]),
                           "verb": a["verb"], "sentence": sentence,
                           "actor": actor, "patient": patient,
                           "actor_range": a["answer_range"],
                           "patient_range": p["answer_range"],
                           "actor_question": a["question"],
                           "patient_question": p["question"]})
    return result


def main() -> None:
    dst = ROOT / "QASRL_CANDIDATES.json"
    if dst.exists():
        raise FileExistsError(dst)
    rows = extract()
    dst.write_text(json.dumps({"source_repo": "https://github.com/plroit/qasrl-gs",
                               "source_files_sha256": {
                                   part: {"sentences": sha(DATA / "sentences" / f"{part}.full.csv"),
                                          "gold": sha(DATA / "gold" / f"{part}.gold.csv")}
                                   for part in PARTS},
                               "filter": "Who-question active nonnegated subject and object annotations on one predicate; unique single actor and patient answer; <=45 sentence tokens and <=8 answer tokens; no bracketed fragments. Candidate only, not human verified.",
                               "rows": rows}, indent=2) + "\n")
    print(f"{len(rows)} candidates saved to {dst}")


if __name__ == "__main__":
    main()
