"""Direct QA-SRL answer-span evaluation of the saved Qwen 0.5B adapters."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import unicodedata

HERE = Path(__file__).resolve().parent
ROOT = HERE / "llm_role_authentic_spans"
SOURCE = HERE / "llm_role_confirmation_v1"
FIRST = HERE / "llm_role_authentic_transfer"
CONFIRM = HERE / "llm_role_repair_confirmation"
UNORDERED = HERE / "llm_role_unordered_control"
WORLDS = ("fresh1", "fresh2", "fresh3")
SEEDS = (830211, 830212)
ARMS = ("pretrained", "ordinary", "ordered_roles", "unordered_pair")
MAX_NEW_TOKENS = 18


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_items() -> list[dict]:
    prior = json.loads((FIRST / "ITEMS.json").read_text())
    candidates = json.loads((SOURCE / "QASRL_CANDIDATES.json").read_text())["rows"]
    selected = {}
    for row in prior:
        selected[row["source_index"]] = row["id"]
    assert len(selected) == 63
    items = []
    for index, identity in selected.items():
        candidate = candidates[index]
        assert candidate["id"] == identity
        for role, question_key, answer_key in (
            ("subject_answer", "actor_question", "actor"),
            ("other_participant_answer", "patient_question", "patient"),
        ):
            answer = candidate[answer_key]
            assert answer in candidate["sentence"]
            assert answer != candidate["patient" if answer_key == "actor" else "actor"]
            items.append({"id": identity, "source_index": index,
                          "partition": candidate["partition"], "sentence": candidate["sentence"],
                          "verb": candidate["verb"], "verb_idx": candidate["verb_idx"],
                          "role": role, "question": candidate[question_key],
                          "gold_span": answer})
    assert len(items) == 126
    assert len({item["id"] for item in items}) == 63
    return items


def inputs_sha() -> dict:
    files = {
        "qa_srl_candidates": SOURCE / "QASRL_CANDIDATES.json",
        "prior_items": FIRST / "ITEMS.json",
        "prior_diagnostics": FIRST / "DIAGNOSTICS.json",
        "confirmation_protocol": CONFIRM / "PROTOCOL.json",
        "unordered_protocol": UNORDERED / "PROTOCOL.json",
        "model_weights": Path("/Users/joelvazquezlopez/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775/model.safetensors"),
    }
    return {name: sha(path) for name, path in files.items()}


def norm(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).lower().strip()
    value = re.sub(r"^[\s\"'`.,:;!?]+|[\s\"'`.,:;!?]+$", "", value)
    value = re.sub(r"\s+'\s*", "'", value)
    value = re.sub(r"\s+", " ", value)
    return value


def protocol() -> dict:
    return {
        "date": "2026-10-11", "status": "fixed_before_direct_span_inference",
        "purpose": "Remove A/B answer-position cues from the exploratory natural-language evaluation.",
        "source": "The same 63 previously selected QA-SRL Gold sentences. Use their original subject and direct-object question-answer annotations, with no A/B options.",
        "role_terms": "subject_answer is the answer to the active QA-SRL subject question; other_participant_answer is the answer to its direct-object question. The latter is not uniformly a semantic patient.",
        "prior_knowledge": "All source sentences and prior A/B outcomes are known. This is a follow-up diagnostic, not a blind external confirmation. An AI assistant inspected the items, but no independent student verification has been entered.",
        "units": "One untouched Qwen2.5-0.5B model plus ordinary, ordered-role and participant-only fine-tuned adapters for three graph worlds x two matched starts; no new training.",
        "prompt": "Fixed system instruction explaining QA-SRL's someone/something placeholders and requiring the exact noun phrase from the sentence, plus the full sentence, target verb, and original QA-SRL question. No candidate answers are presented.",
        "decoding": f"Greedy generation, at most {MAX_NEW_TOKENS} new tokens, stop at EOS. Keep raw text and all outputs including invalid ones.",
        "primary": "Exact normalized subject-answer span accuracy over 63 events per model. Report six paired arm-minus-ordinary differences and an equal-unit mean, alongside raw individual starts.",
        "secondary": "Exact normalized other-participant-answer accuracy; both roles correct per event; copied-substring and nonempty rates; Wikinews/Wikipedia breakdown. Report outputs for human error review.",
        "normalization": "Unicode NFKC, lowercase, trim boundary quote/punctuation characters, collapse whitespace and tokenized apostrophe spacing. No article removal, substring matching or aliases in primary exact score.",
        "limits": "Exact-span scoring is conservative when multiple phrase boundaries or referentially equivalent answers exist. Original QA-SRL annotations do not constitute independent verification for this selected subset. Model pretraining overlap is unknown. A null result may reflect prompt/domain transfer rather than absent role knowledge.",
        "inputs_sha256": inputs_sha(),
        "script_sha256": sha(HERE / "llm_role_authentic_spans.py"),
    }


def prepare() -> None:
    ROOT.mkdir(exist_ok=True)
    if (ROOT / "PROTOCOL.json").exists():
        raise FileExistsError(ROOT / "PROTOCOL.json")
    (ROOT / "ITEMS.json").write_text(json.dumps(make_items(), indent=2) + "\n")
    (ROOT / "PROTOCOL.json").write_text(json.dumps(protocol(), indent=2) + "\n")
    print("Frozen 63 source events x 2 original QA-SRL questions = 126 prompts/model")


def validate() -> list[dict]:
    assert json.loads((ROOT / "PROTOCOL.json").read_text()) == protocol()
    rows = json.loads((ROOT / "ITEMS.json").read_text())
    assert rows == make_items()
    return rows


def prompt(tokenizer, row: dict) -> str:
    return tokenizer.apply_chat_template([
        {"role": "system", "content": "In these questions, 'someone' and 'something' are placeholders, not literal answers. Identify who fills the missing role in the sentence. Copy only the exact noun phrase from the sentence. Do not include the verb or an explanation."},
        {"role": "user", "content": f"Sentence: {row['sentence']}\nTarget verb: {row['verb']}\nQuestion: {row['question']}"},
    ], tokenize=False, add_generation_prompt=True)


def adapter_path(world: str, seed: int, arm: str) -> Path:
    if arm == "unordered_pair":
        return UNORDERED / f"{world}_{seed}_unordered_pair" / "adapter.pt"
    return CONFIRM / f"{world}_{seed}_{arm}" / "adapter.pt"


def run(world: str, seed: int, arm: str, limit: int | None) -> None:
    if arm == "pretrained":
        assert world == "fresh1" and seed == 0
    else:
        assert world in WORLDS and seed in SEEDS
    rows = validate()
    if limit is not None:
        assert 0 < limit <= len(rows)
        rows = rows[:limit]
    import torch
    import llm_role_repair_confirm as confirm
    import llm_role_repair_v3 as core

    label = f"{world}_{seed}_{arm}" + (f"_smoke{limit}" if limit else "")
    path = ROOT / f"{label}.json"
    if path.exists():
        raise FileExistsError(path)
    confirm.configure(world, seed)
    tokenizer, model, _ = core.load("pretrained" if arm == "pretrained" else "ordinary")
    adapter_hash = None
    if arm != "pretrained":
        weights_path = adapter_path(world, seed, arm)
        adapter_hash = sha(weights_path)
        weights = torch.load(weights_path, map_location="cpu", weights_only=True)["adapter"]
        params = {name: param for name, param in model.named_parameters() if param.requires_grad}
        assert weights.keys() == params.keys()
        for name, param in params.items():
            assert param.shape == weights[name].shape
            param.data.copy_(weights[name])
    model.eval()
    results = []
    with torch.inference_mode():
        for start in range(0, len(rows), 4):
            batch = rows[start:start+4]
            encoded = tokenizer([prompt(tokenizer, row) for row in batch],
                                padding=True, return_tensors="pt")
            output = model.generate(**encoded, do_sample=False,
                                    max_new_tokens=MAX_NEW_TOKENS,
                                    pad_token_id=tokenizer.pad_token_id,
                                    eos_token_id=tokenizer.eos_token_id,
                                    use_cache=True)
            generated = output[:, encoded["input_ids"].shape[1]:]
            for row, token_ids in zip(batch, generated, strict=True):
                answer = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
                gold = row["gold_span"]
                results.append({"id": row["id"], "role": row["role"],
                                "source_index": row["source_index"],
                                "gold_span": gold, "answer": answer,
                                "correct": norm(answer) == norm(gold),
                                "copied": norm(answer) in norm(row["sentence"]),
                                "nonempty": bool(norm(answer))})
            if (start + len(batch)) % 32 == 0:
                print(f"{label}: {start + len(batch)}/{len(rows)}", flush=True)
    path.write_text(json.dumps({"world": world, "seed": seed, "arm": arm,
                                "protocol_sha256": sha(ROOT / "PROTOCOL.json"),
                                "items_sha256": sha(ROOT / "ITEMS.json"),
                                "adapter_sha256": adapter_hash,
                                "records": results}, indent=2) + "\n")
    print(f"Saved {path}: exact={sum(row['correct'] for row in results)}/{len(results)}")


def analyze() -> None:
    items = validate()
    expected = {(row["id"], row["role"]) for row in items}
    units = [("fresh1", 0, "pretrained")] + [
        (w, s, a) for w in WORLDS for s in SEEDS
        for a in ("ordinary", "ordered_roles", "unordered_pair")]
    cells = {}
    for world, seed, arm in units:
        path = ROOT / f"{world}_{seed}_{arm}.json"
        if not path.exists():
            continue
        obj = json.loads(path.read_text())
        assert obj["protocol_sha256"] == sha(ROOT / "PROTOCOL.json")
        assert obj["items_sha256"] == sha(ROOT / "ITEMS.json")
        records = obj["records"]
        assert len(records) == 126
        assert {(r["id"], r["role"]) for r in records} == expected
        if arm != "pretrained":
            assert obj["adapter_sha256"] == sha(adapter_path(world, seed, arm))
        by_id = defaultdict(list)
        for row in records:
            by_id[row["id"]].append(row)
        cells[f"{world}/{seed}/{arm}"] = {
            role: {"correct": sum(r["correct"] for r in records if r["role"] == role),
                   "copied": sum(r["copied"] for r in records if r["role"] == role),
                   "nonempty": sum(r["nonempty"] for r in records if r["role"] == role),
                   "n": 63}
            for role in ("subject_answer", "other_participant_answer")}
        cells[f"{world}/{seed}/{arm}"]["both_roles_correct"] = sum(
            all(r["correct"] for r in group) for group in by_id.values())
    paired = []
    for world in WORLDS:
        for seed in SEEDS:
            base = cells.get(f"{world}/{seed}/ordinary")
            if base is None:
                continue
            for arm in ("ordered_roles", "unordered_pair"):
                other = cells.get(f"{world}/{seed}/{arm}")
                if other is None:
                    continue
                paired.append({"world": world, "seed": seed, "arm": arm,
                               **{f"{role}_delta_pp": 100 *
                                  (other[role]["correct"] - base[role]["correct"]) / 63
                                  for role in ("subject_answer", "other_participant_answer")}})
    report = {"complete": len(cells) == 19, "cells": cells,
              "paired_differences": paired,
              "note": "Exploratory direct-span follow-up on previously selected QA-SRL items; exact match is conservative and independent human review remains pending."}
    (ROOT / "ANALYSIS.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"complete": report["complete"], "n_model_runs": len(cells),
                      "paired_differences": paired}, indent=2))


def batch() -> None:
    validate()
    jobs = [("fresh1", 0, "pretrained")] + [
        (w, s, a) for w in WORLDS for s in SEEDS
        for a in ("ordinary", "ordered_roles", "unordered_pair")]
    for world, seed, arm in jobs:
        output = ROOT / f"{world}_{seed}_{arm}.json"
        if output.exists():
            print(f"Already complete: {output}", flush=True)
            continue
        subprocess.run([sys.executable, str(HERE / "llm_role_authentic_spans.py"),
                        "run", "--world", world, "--seed", str(seed), "--arm", arm], check=True)
    analyze()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "validate", "run", "batch", "analyze"))
    parser.add_argument("--world", choices=WORLDS, default="fresh1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arm", choices=ARMS, default="pretrained")
    parser.add_argument("--limit", type=int, help="Run a nonreportable smoke check")
    args = parser.parse_args()
    if args.action == "prepare": prepare()
    elif args.action == "validate":
        validate(); print("Protocol and 126 items validated")
    elif args.action == "run": run(args.world, args.seed, args.arm, args.limit)
    elif args.action == "batch": batch()
    else: analyze()
