"""Freeze and evaluate a cross-corpus English role test from UD EWT r2.18.

The UD labels are provisional proxies for semantic roles. Human review is
required before treating the aggregate as a semantic-role result.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE / "llm_role_independent_eval"
SOURCE = ROOT / "data/en_ewt-ud-test.conllu"
BANK_TRAIN = HERE / "llm_role_natural_repair/TRAIN.json"
BANK_DEV = HERE / "llm_role_natural_repair/DEV.json"
BANK_TEST = HERE / "llm_role_natural_repair/fresh_bank_test/EVENTS.json"
GOLD = HERE / "llm_role_natural_contrast_confirm/EVENTS.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(text: str) -> str:
    return " ".join(text.casefold().split())


def parse_sentences():
    for block in SOURCE.read_text().split("\n\n"):
        if not block.strip():
            continue
        meta = {}
        tokens = []
        for line in block.splitlines():
            if line.startswith("# ") and " = " in line:
                k, v = line[2:].split(" = ", 1)
                meta[k] = v
            elif line and not line.startswith("#"):
                fields = line.split("\t")
                if fields[0].isdigit():
                    tokens.append({"id": int(fields[0]), "form": fields[1],
                                   "upos": fields[3], "head": int(fields[6]),
                                   "dep": fields[7], "misc": fields[9]})
        if "sent_id" in meta and "text" in meta and tokens:
            yield meta, tokens


def phrase(head, tokens):
    # Keep only the noun phrase's simple, contiguous internal modifiers.
    allowed = {"det", "amod", "compound", "flat", "nummod", "nmod:poss"}
    ids = {head["id"]}
    while True:
        more = {t["id"] for t in tokens if t["head"] in ids and
                t["dep"] in allowed and t["upos"] != "PUNCT"}
        if more <= ids:
            break
        ids |= more
    if len(ids) > 8 or len(ids) != max(ids)-min(ids)+1:
        return None
    forms = [t["form"] for t in tokens if t["id"] in ids]
    return " ".join(forms)


def extract():
    seen = set()
    output = []
    for meta, ts in parse_sentences():
        if not 5 <= len(ts) <= 35 or normalized(meta["text"]) in seen:
            continue
        roots = [t for t in ts if t["head"] == 0]
        if len(roots) != 1 or roots[0]["upos"] != "VERB":
            continue
        verb = roots[0]
        children = [t for t in ts if t["head"] == verb["id"]]
        subjects = [t for t in children if t["dep"] == "nsubj"]
        objects = [t for t in children if t["dep"] == "obj"]
        if len(subjects) != 1 or len(objects) != 1:
            continue
        if subjects[0]["upos"] not in ("NOUN", "PROPN") or objects[0]["upos"] not in ("NOUN", "PROPN"):
            continue
        if any(t["dep"] in ("aux:pass", "nsubj:pass", "expl", "neg") or
               (t["dep"] == "advmod" and t["form"].casefold() in ("not", "n't", "never"))
               for t in children):
            continue
        actor = phrase(subjects[0], ts)
        other = phrase(objects[0], ts)
        if not actor or not other or normalized(actor) == normalized(other):
            continue
        if " " + normalized(actor) + " " not in " " + normalized(" ".join(t["form"] for t in ts)) + " ":
            continue
        if " " + normalized(other) + " " not in " " + normalized(" ".join(t["form"] for t in ts)) + " ":
            continue
        seen.add(normalized(meta["text"]))
        output.append({"id": meta["sent_id"], "sentence": meta["text"],
                       "verb": verb["form"], "actor_span": actor,
                       "other_span": other, "actor_head": subjects[0]["form"],
                       "other_head": objects[0]["form"],
                       "source": "UD_English-EWT-r2.18/test"})
    return output


def prepare():
    target = ROOT / "PROTOCOL.json"
    if target.exists():
        raise FileExistsError(target)
    prior = [json.loads(path.read_text()) for path in (BANK_TRAIN, BANK_DEV, BANK_TEST, GOLD)]
    prior_sentences = {normalized(row["sentence"]) for group in prior for row in group}
    rows = [row for row in extract() if normalized(row["sentence"]) not in prior_sentences]
    assert len(rows) >= 50, len(rows)
    (ROOT / "EVENTS.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (ROOT / "HUMAN_REVIEW.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "sentence", "target_verb", "provisional_actor", "provisional_other",
                         "actor_valid_yes_no_unsure", "other_valid_yes_no_unsure", "notes"])
        for r in rows:
            writer.writerow([r["id"], r["sentence"], r["verb"], r["actor_span"],
                             r["other_span"], "", "", ""])
    protocol = {
        "status": "fixed_before_any_EWT_model_inference",
        "date": "2026-10-12",
        "source": "UniversalDependencies/UD_English-EWT r2.18 test CoNLL-U",
        "source_url": "https://github.com/UniversalDependencies/UD_English-EWT/tree/r2.18",
        "selection": "All unique EWT test sentences with 5-35 tokens, one root VERB, exactly one nsubj and obj dependent with NOUN/PROPN heads, no root-verb passive/negation/expletive marker, and simple contiguous <=8-token noun phrases. Exclude exact normalized sentence overlaps with selected QA-SRL train, dev, Bank test, and Gold. No manual exclusions or model-based selection. UD subject/object are provisional proxies for actor/other roles.",
        "primary": "Paired candidate-relative role contrast D>0 for each sentence under the exact two QA-SRL evaluation questions. Compare same Qwen2.5-0.5B source and natural-only checkpoints for two starts. Report each item and start; no post-hoc threshold to declare success.",
        "controls": "Sentence-omitted D; directly generated exact actor and other spans under the same prompts; report these separately. Fixed secondary paraphrase: 'Which phrase names the doer of [verb]?' versus 'Which phrase names what [verb] affects directly?' Both ask for a phrase from the sentence. The paraphrase is a prompt-sensitivity test, not a new trained prompt.",
        "analysis": "Report raw counts, gains/losses for matched items, document source counts, and per-start scores. Prefer document-cluster uncertainty if a document key can be reliably extracted; do not treat two starts or multiple sentences from a document as independent models. Human review CSV remains blank and its absence must be stated in results.",
        "limits": "EWT is a syntactic treebank, not a semantic-role gold corpus. Subjects can be non-agentive and objects non-patient. Pretraining overlap unknown. The test source and design were chosen after QA-SRL successes, so cross-corpus evidence is exploratory even though items and measures are frozen before EWT inference.",
        "n_events": len(rows),
        "references": {"ewt_sha256": sha(SOURCE),
                       "bank_train_sha256": sha(BANK_TRAIN),
                       "bank_dev_sha256": sha(BANK_DEV),
                       "bank_test_sha256": sha(BANK_TEST),
                       "gold_sha256": sha(GOLD),
                       "events_sha256": sha(ROOT / "EVENTS.json"),
                       "script_sha256": sha(HERE / "llm_role_independent_eval.py")}}
    target.write_text(json.dumps(protocol, indent=2) + "\n")
    print(f"Frozen {len(rows)} EWT events before inference")


def validate():
    p = json.loads((ROOT / "PROTOCOL.json").read_text())
    paths = {"ewt_sha256": SOURCE, "bank_train_sha256": BANK_TRAIN,
             "bank_dev_sha256": BANK_DEV, "bank_test_sha256": BANK_TEST,
             "gold_sha256": GOLD, "events_sha256": ROOT / "EVENTS.json",
             "script_sha256": HERE / "llm_role_independent_eval.py"}
    for label, path in paths.items():
        assert p["references"][label] == sha(path), label
    rows = json.loads((ROOT / "EVENTS.json").read_text())
    assert len(rows) == p["n_events"] and rows == [
        row for row in extract() if normalized(row["sentence"]) not in {
            normalized(x["sentence"]) for path in (BANK_TRAIN, BANK_DEV, BANK_TEST, GOLD)
            for x in json.loads(path.read_text())}]
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "validate"))
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    else:
        print(f"Validated {len(validate())} events")
