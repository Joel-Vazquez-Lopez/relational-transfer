#!/usr/bin/env python3
"""Verify source integrity of Mira's OPEN, UNLABELED UD-EWT pilot.

Requires a locally obtained original en_ewt-ud-dev.conllu at pinned commit;
does not download, execute any models, infer semantic roles, or grant rights.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

SOURCE_COMMIT = "4a4d77f599ea53cc405f85d0cec4b2f14f81d42b"
SOURCE_BLOB_SHA1 = "120d04140cf4833690117bc74c08c8da861bbce6"
HERE = Path(__file__).resolve().parent

def load_lines(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def parse_ud(path):
    raw = path.read_bytes()
    actual = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    if actual != SOURCE_BLOB_SHA1:
        raise ValueError("SOURCE_GIT_BLOB_MISMATCH: expected %s, found %s" % (SOURCE_BLOB_SHA1, actual))
    sentences = {}
    active_doc = None
    for block in raw.decode("utf-8").split("\n\n"):
        sid = None
        local_doc = None
        tokens = {}
        for line in block.splitlines():
            if line.startswith("# newdoc id = "):
                local_doc = line[len("# newdoc id = "):]
            if line.startswith("# sent_id = "):
                sid = line[len("# sent_id = "):]
            if line and not line.startswith("# "):
                cols = line.split("\t")
                if len(cols) == 10 and cols[0].isdigit():
                    tokens[int(cols[0])] = {"form": cols[1], "lemma": cols[2], "upos": cols[3], "head": cols[6], "deprel": cols[7]}
        if local_doc:
            active_doc = local_doc
        if sid:
            if not active_doc or sid in sentences:
                raise ValueError("Missing document or duplicate sent_id: " + sid)
            sentences[sid] = {"document_id": active_doc, "tokens": tokens}
    return sentences, actual

def require(condition, message):
    if not condition:
        raise AssertionError(message)

def check_csv(filename, expected_n):
    with (HERE / filename).open(newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        require(rd.fieldnames is not None, "No CSV header: " + filename)
        rows = list(rd)
    require(len(rows) == expected_n, "Row-count mismatch: " + filename)
    require(all(None not in r for r in rows), "Unaligned CSV fields: " + filename)
    return rows

def main():
    a = argparse.ArgumentParser()
    a.add_argument("source_conllu", type=Path, help="Exact pinned UD English EWT dev .conllu")
    cfg = a.parse_args()
    corpus, source_blob = parse_ud(cfg.source_conllu)
    base = load_lines(HERE / "candidate_manifest.jsonl")
    challenge = load_lines(HERE / "challenge_candidate_manifest.jsonl")
    require(len(base) == 25 and len(challenge) == 24, "Manifest row counts changed")
    seen_docs = set()
    seen_sids = set()
    for record in base:
        cid = record["candidate_id"]
        require(record["source_commit"] == SOURCE_COMMIT, cid + ": source ref drift")
        require("gold" not in record and "sentence_text" not in record, cid + ": forbidden label or raw text")
        sid = record["sent_id"]
        require(sid in corpus, cid + ": missing sent_id")
        source = corpus[sid]
        require(source["document_id"] == record["document_id"], cid + ": doc mismatch")
        tok = source["tokens"].get(record["predicate_token_id"])
        require(tok is not None and tok["form"] == record["predicate_surface"] and tok["lemma"] == record["predicate_lemma"], cid + ": predicate mismatch")
        require(record["document_id"] not in seen_docs, cid + ": repeated document")
        seen_docs.add(record["document_id"])
        seen_sids.add(sid)
    counts = Counter()
    target_n = 0
    for row in challenge:
        cid = row["candidate_id"]
        require(row["source_commit"] == SOURCE_COMMIT, cid + ": source ref drift")
        require("gold" not in row and "sentence_text" not in row, cid + ": forbidden label or raw text")
        require(row["document_id"] not in seen_docs, cid + ": doc reused")
        seen_docs.add(row["document_id"])
        ids = row["sentence_ids"]
        predicates = row["target_predicates"]
        require(len(ids) in (1, 2) and len(predicates) in (1, 2), cid + ": invalid number of items")
        for sid in ids:
            require(sid in corpus, cid + ": source sentence missing")
            require(corpus[sid]["document_id"] == row["document_id"], cid + ": cross-document candidate")
            require(sid not in seen_sids, cid + ": reused sentence")
            seen_sids.add(sid)
        for p in predicates:
            sid = p["sent_id"]
            require(sid in ids, cid + ": predicate out of selected context")
            tok = corpus[sid]["tokens"].get(p["token_id"])
            require(tok is not None and tok["lemma"] == p["lemma"] and tok["form"] == p["surface"] and tok["upos"] == "VERB", cid + ": invalid predicate")
        cat = row["category"]
        counts[cat] += 1
        target_n += len(predicates)
        if cat == "repeated_lemma_nearby":
            require(len(ids) == len(predicates) == 2 and predicates[0]["lemma"] == predicates[1]["lemma"] and ids[0] != ids[1], cid + ": repeated-lemma control invalid")
        elif cat == "passive":
            require(len(ids) == 1 and len(predicates) == 1, cid + ": passive count")
            tid = str(predicates[0]["token_id"])
            require(any(t["head"] == tid and t["deprel"] == "nsubj:pass" for t in corpus[ids[0]]["tokens"].values()), cid + ": not a passive head")
        elif cat == "two_overt_event_candidates":
            require(len(ids) == 1 and len(predicates) == 2 and predicates[0]["token_id"] != predicates[1]["token_id"], cid + ": two event references invalid")
        elif cat == "pronoun_event_scope":
            require(len(ids) == len(predicates) == 1, cid + ": pronoun counts")
            toks = corpus[ids[0]]["tokens"].values()
            require(sum(t["upos"] == "VERB" for t in toks) >= 2 and any(t["upos"] == "PRON" for t in toks), cid + ": scope heuristic mismatch")
        else:
            raise AssertionError(cid + ": unknown category " + cat)
    require(target_n == 36 and len(seen_docs) == 49 and len(seen_sids) == 55, "Group/source counts changed")
    require(dict(counts) == {"repeated_lemma_nearby": 6, "passive": 6, "two_overt_event_candidates": 6, "pronoun_event_scope": 6}, "Category balance changed")
    for name, n in (("annotator_A_template.csv",25),("annotator_B_template.csv",25),("adjudication_template.csv",25),
                    ("document_rights_review_template.csv",25),("challenge_annotator_A.csv",36),
                    ("challenge_annotator_B.csv",36),("challenge_rights_review.csv",24)):
        check_csv(name, n)
    print(json.dumps({"status":"SOURCE_INTAKE_PASS","source_blob_sha1":source_blob,"base_documents":25,
      "challenge_documents":24,"total_distinct_documents":49,"total_source_sentences":55,
      "challenge_predicate_slots":36,"challenge_categories":counts,"human_semantic_gold":0,
      "source_text_rights":"PENDING_HUMAN_REVIEW"}, indent=2))

if __name__ == "__main__":
    main()
