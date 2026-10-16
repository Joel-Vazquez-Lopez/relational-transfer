#!/usr/bin/env python3
"""BX-006: independent read-only QA-SRL Gold development data integrity and answer-span audit.

No model inference or automatic conversion of QASRL questions to actor/patient gold.
Refer to https://github.com/plroit/qasrl-gs/tree/f7c64ae9b6fe48ff3910c3e59850a12ec278bf83
No original source text or raw corpus rows are included in this release.
"""
import argparse
import collections
import csv
import json
import re
from pathlib import Path

PINNED_SOURCE_COMMIT = "f7c64ae9b6fe48ff3910c3e59850a12ec278bf83"
SEP = "~!~"
RANGE = re.compile(r"^(\d+):(\d+)$")


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def parse_reference_spans(answer, answer_range, tokens):
    """Return source-index-backed alternatives; do not silently rewrite damaged gold text."""
    surfaces = [a.strip() for a in answer.split(SEP)]
    ranges = [r.strip() for r in answer_range.split(SEP)]
    if len(surfaces) != len(ranges) or not surfaces:
        return {"status": "COUNT_MISMATCH", "alternatives": []}
    alternatives = []
    for surface, coordinates in zip(surfaces, ranges):
        match = RANGE.fullmatch(coordinates)
        if not match:
            return {"status": "BAD_RANGE_FORMAT", "alternatives": alternatives}
        start, end = map(int, match.groups())
        if not (0 <= start < end <= len(tokens)):
            return {"status": "OUT_OF_BOUNDS", "alternatives": alternatives}
        source_slice = " ".join(tokens[start:end])
        alternatives.append({"start": start, "end": end, "surface_matches_source": surface == source_slice,
                             "answer_token_length": end-start, "source_slice": source_slice})
    return {"status": "EXACT" if all(x["surface_matches_source"] for x in alternatives)
            else "SOURCE_TEXT_DISAGREEMENT", "alternatives": alternatives}


def exact_span_match(response, span_record, source_tokens):
    """Conservative plain-string QA match to any admissible source span.

    Never use when reference is corrupt or disputed. This is diagnostic only;
    official QA-SRL evaluation and span-overlap scoring are separate.
    """
    if span_record["status"] != "EXACT":
        return None  # abstain instead of silently repairing damaged gold
    response = " ".join(response.strip().split()).casefold()
    return any(response == " ".join(source_tokens[a["start"]:a["end"]]).casefold()
               for a in span_record["alternatives"])


def audit(gold_rows, sentence_rows, *, partition):
    """Return non-text aggregate metadata only; no raw text or worker identifiers."""
    if ".test" in partition.lower():
        raise ValueError("BX-006 refuses test splits: prospective test evaluation is held back")
    sentences = {}
    for row in sentence_rows:
        sid = row.get("qasrl_id")
        if not sid or sid in sentences:
            raise ValueError("Missing or duplicate sentence identifier")
        sentences[sid] = row
    flags = collections.Counter()
    kinds = collections.Counter()
    qa_key = collections.Counter()
    annotated = set()
    n_alternatives = 0
    multi=0
    wh=collections.Counter()
    for r in gold_rows:
        sid=r.get("qasrl_id")
        if not sid or sid not in sentences:
            flags["MISSING_SENTENCE_JOIN"] += 1
            continue
        annotated.add(sid)
        wh[(r.get("wh") or "UNSPECIFIED").lower()] += 1
        tokenized=(sentences[sid].get("tokens") or "").split()
        parsed = parse_reference_spans(r.get("answer", ""),r.get("answer_range", ""),tokenized)
        kinds[parsed["status"]] += 1
        if parsed["status"] == "SOURCE_TEXT_DISAGREEMENT":
            flags["QUARANTINED_ENCODING_OR_LABEL_DISAGREEMENT"] += 1
        n_alternatives += len(parsed["alternatives"])
        multi += int(len(parsed["alternatives"])>1)
        key=(sid,r.get("verb_idx"),r.get("question"))
        qa_key[key] += 1
        try:
            vidx=int(r["verb_idx"])
            if not 0 <= vidx < len(tokenized):flags["PREDICATE_INDEX_OOB"] += 1
        except (KeyError,ValueError): flags["PREDICATE_INDEX_INVALID"]+=1
    duplicates=sum(int(v>1) for v in qa_key.values())
    if duplicates:flags["DUPLICATE_SENTENCE_VERB_QUESTION_KEYS"] = duplicates
    return {
        "experiment":"BX-006","status":"DEVELOPMENT_METADATA_AUDIT_ONLY",
        "source_commit":PINNED_SOURCE_COMMIT,"partition":partition,
        "gold_question_rows":len(gold_rows),"unique_annotated_sentence_ids":len(annotated),
        "sentence_reference_rows":len(sentence_rows),"source_missing_gold_joins":flags["MISSING_SENTENCE_JOIN"],
        "multiple_accepted_span_rows":multi,
        "multiple_accepted_span_fraction":(multi/len(gold_rows) if gold_rows else None),
        "total_accepted_answer_spans":n_alternatives,"reference_status_counts":dict(sorted(kinds.items())),
        "other_flags":dict(sorted(flags.items())),"wh_distribution":dict(sorted(wh.items())),
        "metric_contract":{
            "valid_answer_unit":"human-written QA question and zero-based source token span(s)",
            "answer_range":"start-inclusive/end-exclusive; ~!~ separates alternatives",
            "gold_conversion":"NO actor/patient label derivation",
            "score_policy":"quarantine SOURCE_TEXT_DISAGREEMENT; exact source-span alternatives only after integrity gate",
            "sampling_unit":"sentence ID, not verified unique document; no document-level uncertainty without document IDs",
            "model_results":"NONE","blind_test_claim":"NONE"},
        "cautions":["No source text redistributed", "Third-party text rights not automatically established by repo MIT license", "Human QA spans do not imply binary agent/patient roles", "Official QA-SRL evaluation should be used for primary published-task metrics"]
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--gold",required=True)
    p.add_argument("--sentences",required=True)
    p.add_argument("--partition",required=True)
    p.add_argument("--out")
    args=p.parse_args()
    res=audit(read_csv(args.gold),read_csv(args.sentences),partition=args.partition)
    result=json.dumps(res,indent=2,ensure_ascii=False)+"\n"
    if args.out: Path(args.out).write_text(result,encoding="utf-8")
    print(result)
if __name__ == "__main__":main()
