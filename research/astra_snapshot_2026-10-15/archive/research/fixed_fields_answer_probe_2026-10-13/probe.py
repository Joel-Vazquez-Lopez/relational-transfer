#!/usr/bin/env python3
"""Adaptive, no-training answer continuation with cached generated role fields."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "question_role_tag_diagnostic_2026-10-13"
sys.path.insert(0, str(SOURCE))
import diagnostic as prior_tag  # noqa: E402
source = ROOT.parent / "natural_role_verifier_development_2026-10-13"
sys.path.insert(0, str(source))
import natural_train as prior  # noqa: E402

STARTS = ("fresh1", "fresh3")
TAGS = ("ordinary", "tagged")
FIELDS = ("original", "swapped")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cached(start):
    path = SOURCE / f"EVAL_{start}_role_first_ordinary.json"
    data = json.loads(path.read_text())
    assert data["protocol_sha256"] == sha(SOURCE / "PROTOCOL.json")
    return path, {(r["id"], r["role"], r["order"]): r for r in data["records"]}


def validate():
    cfg = json.loads((ROOT / "PROTOCOL.json").read_text())
    assert cfg["script_sha256"] == sha(Path(__file__))
    assert cfg["items_sha256"] == sha(SOURCE / "ITEMS.json")
    for start in STARTS:
        assert cfg["checkpoint_sha256"][start] == sha(prior.checkpoint(start, "role_first"))
        assert cfg["cached_eval_sha256"][start] == sha(cached(start)[0])
    return cfg


def prepare():
    if (ROOT / "PROTOCOL.json").exists():
        raise FileExistsError(ROOT / "PROTOCOL.json")
    cfg = {
        "date": "2026-10-13", "status": "adaptive exploratory follow-up after the explicit-tag development screen failed",
        "question": "When originally generated actor/patient fields are held fixed, does an explicit requested-role cue improve the final answer, and do swapped fields causally influence it?",
        "data": "The same 40 new, non-sealed original-train QA-SRL Bank development items as the question-role-tag diagnostic. Automatic actor/patient mapping is not independently human-reviewed.",
        "design": "For each original role-first output, prefill the assistant with its cached Actor and Patient text up to Answer:. Compare ordinary versus explicit role-tagged user prompt, crossing original versus swapped cached fields. Same sentence, question, candidates, checkpoint, and decoding. No training. The ordinary/original cell should approximately reproduce the prior full-generation answer; if it does not, prefill tokenization or prompt replay limits interpretation.",
        "starts": list(STARTS), "tags": list(TAGS), "fields": list(FIELDS),
        "primary": "On original fields, compare paired exact-answer accuracy tagged minus ordinary; report separately for prompts where cached requested field matches the annotation. Measure final-answer agreement with requested and other fields, question-pair same-answer rate, and overlap of ordinary/original continuation with cached full-generation answer.",
        "secondary": "Swap cached fields while holding question and sentence fixed. Field-following and gold-answer accuracy are separate outcomes. Swapped fields can be incorrect or correct relative to the sentence; neither behavior is automatically preferable.",
        "limits": "Selected after inspecting earlier development results; exploratory only. Prefilling a reconstructed textual prefix can alter tokenization relative to original autoregressive generation. An explicit tag can change prompt processing beyond disambiguating question role. No internal-state claim and no sealed evaluation.",
        "script_sha256": sha(Path(__file__)),
        "items_sha256": sha(SOURCE / "ITEMS.json"),
        "checkpoint_sha256": {start: sha(prior.checkpoint(start, "role_first")) for start in STARTS},
        "cached_eval_sha256": {start: sha(cached(start)[0]) for start in STARTS},
    }
    (ROOT / "PROTOCOL.json").write_text(json.dumps(cfg, indent=2) + "\n")
    print("Frozen adaptive follow-up", sha(ROOT / "PROTOCOL.json"))


def prefill(tok, row, cache, tag, fields):
    assert tag in TAGS and fields in FIELDS
    base = prior_tag.prompt(tok, row, "role_first", "tagged" if tag == "tagged" else "ordinary")
    raw = cache["raw"]
    before, separator, _answer = raw.partition("Answer:")
    assert separator and cache["actor"] is not None and cache["patient"] is not None
    if fields == "original":
        prefix = before + separator
    else:
        prefix = f" {cache['patient']}; Patient: {cache['actor']}; Answer:"
    return base + prefix


@torch.inference_mode()
def evaluate(start, tag, fields):
    validate()
    output = ROOT / f"EVAL_{start}_{tag}_{fields}.json"
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(2)
    tok, model = prior.load_trained(start, "role_first")
    items = json.loads((SOURCE / "ITEMS.json").read_text())
    rows = [row for item in items for row in prior.rows(item)]
    _, cache = cached(start)
    records = []
    for off in range(0, len(rows), 4):
        block = rows[off:off+4]
        old = [cache[(r["id"], r["role"], r["order"])] for r in block]
        prompts = [prefill(tok, row, prior_row, tag, fields) for row, prior_row in zip(block, old)]
        encoded = tok(prompts, padding=True, return_tensors="pt", add_special_tokens=False)
        generated = model.generate(**encoded, max_new_tokens=24, do_sample=False,
                                   pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        for row, old_row, tokens in zip(block, old, generated[:, encoded["input_ids"].shape[1]:]):
            raw = tok.decode(tokens, skip_special_tokens=True)
            answer = raw.strip().split(";")[0].split("\n")[0].strip() or None
            actor = old_row["actor"] if fields == "original" else old_row["patient"]
            patient = old_row["patient"] if fields == "original" else old_row["actor"]
            requested = actor if row["role"] == "actor" else patient
            other = patient if row["role"] == "actor" else actor
            records.append({"id": row["id"], "role": row["role"], "order": row["order"],
                            "gold": row["gold"], "raw": raw, "answer": answer,
                            "cached_full_answer": old_row["answer"],
                            "actor_field": actor, "patient_field": patient,
                            "correct": answer is not None and prior.norm(answer) == prior.norm(row["gold"]),
                            "requested_field_correct": prior.norm(requested) == prior.norm(row["gold"]),
                            "answer_matches_requested_field": answer is not None and prior.norm(answer) == prior.norm(requested),
                            "answer_matches_other_field": answer is not None and prior.norm(answer) == prior.norm(other),
                            "matches_cached_full_answer": answer is not None and prior.norm(answer) == prior.norm(old_row["answer"])})
        if (off + 4) % 80 == 0:
            print(start, tag, fields, off + 4, "/", len(rows), flush=True)
    output.write_text(json.dumps({"start": start, "tag": tag, "fields": fields,
                                  "protocol_sha256": sha(ROOT / "PROTOCOL.json"),
                                  "checkpoint_sha256": sha(prior.checkpoint(start, "role_first")),
                                  "records": records}, indent=2) + "\n")
    print("Saved", output.name, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("prepare", "eval"))
    p.add_argument("--start", choices=STARTS)
    p.add_argument("--tag", choices=TAGS)
    p.add_argument("--fields", choices=FIELDS)
    a = p.parse_args()
    if a.mode == "prepare":
        prepare()
    else:
        if not a.start or not a.tag or not a.fields:
            p.error("eval requires start, tag, and fields")
        evaluate(a.start, a.tag, a.fields)
