"""Same-sentence actor-versus-patient donor control on fresh Qwen items."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch

import llm_role_internal_v1 as core

HERE = Path(__file__).resolve().parent
ROOT = HERE / "llm_role_internal_v2"
PAIRS = (("Alice", "Bob"), ("Emma", "John"), ("Mary", "Tom"),
         ("Jack", "Anna"), ("Leo", "James"), ("Sarah", "Kate"),
         ("Mike", "Paul"), ("Max", "Alex"))
VERBS = ("greeted", "assisted", "warned", "admired")
VOICES = ("active", "passive")
WORDINGS = ("generic_actor", "performed_actor", "verb_patient")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows() -> list[dict]:
    result = []
    for pair_index, (a, b) in enumerate(PAIRS):
        for verb in VERBS:
            for direction, (actor, patient) in enumerate(((a, b), (b, a))):
                for voice in VOICES:
                    sentence = (f"{actor} {verb} {patient}." if voice == "active"
                                else f"{patient} was {verb} by {actor}.")
                    questions = {"generic_actor": "Who did the action?",
                                 "performed_actor": "Who performed the action?",
                                 "verb_patient": f"Who was {verb}?"}
                    for wording in WORDINGS:
                        result.append({"pair_index": pair_index, "verb": verb,
                                       "direction": direction, "actor": actor,
                                       "patient": patient, "voice": voice,
                                       "wording": wording, "sentence": sentence,
                                       "question": questions[wording],
                                       "gold": patient if wording == "verb_patient" else actor})
    assert len(result) == 384
    return result


def protocol() -> dict:
    return {"date": "2026-10-10", "status": "fixed_before_v2_run",
            "model": "Qwen2.5-1.5B-Instruct at commit 989aa7980e4cf806f80c7fef2b1adb7bc71aa306, no weight updates",
            "model_weights_sha256": sha(core.MODEL / "model.safetensors"),
            "source_hashes": {"v1": sha(HERE / "llm_role_internal_v1.py"),
                              "v2": sha(HERE / "llm_role_internal_v2.py")},
            "design": "8 fresh name pairs x 4 fresh reversible verbs x 2 directions x active/passive x 3 questions = 384 prompts, 64 events",
            "question_conditions": {"generic_actor": "Who did the action?",
                                    "performed_actor": "Who performed the action?",
                                    "verb_patient": "Who was [verb]?"},
            "selection": "generic actor first token wrong, performed actor first token correct, verb-specific patient first token correct, on exactly the same sentence; all valid outputs must be a name from the sentence",
            "causal_comparison": "replace the generic actor prompt's last-token residual stream after block20 with either the performed-actor donor or the verb-patient donor from the identical sentence; compare actor-minus-patient candidate logit margin",
            "analysis": "report all selected cases, donor answer rates, group means over eight name pairs, and exact paired sign flip if each pair contributes cases; no inference from the 384 dependent prompts as independent observations",
            "limits": "the entire state is transplanted, including question semantics; this tests whether a role-selective state can steer the answer, not an isolated role circuit or training repair"}


def prepare() -> None:
    ROOT.mkdir(exist_ok=True)
    path = ROOT / "PROTOCOL.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(protocol(), indent=2) + "\n")
    (ROOT / "ITEMS.json").write_text(json.dumps(rows(), indent=2) + "\n")
    print(json.dumps({"protocol": str(path), "items": 384}, indent=2))


def validate() -> list[dict]:
    assert json.loads((ROOT / "PROTOCOL.json").read_text()) == protocol()
    items = json.loads((ROOT / "ITEMS.json").read_text())
    assert items == rows()
    return items


@torch.inference_mode()
def run() -> None:
    items = validate()
    if (ROOT / "RESULT.json").exists():
        raise FileExistsError(ROOT / "RESULT.json")
    tokenizer, model = core.load_model()
    for name in {n for pair in PAIRS for n in pair}:
        assert len(tokenizer.encode(name, add_special_tokens=False)) == 1
        assert len(tokenizer.encode(" " + name, add_special_tokens=False)) == 1
    records = []
    h20 = []
    for start in range(0, len(items), 4):
        batch = items[start:start + 4]
        inputs = tokenizer([core.make_prompt(tokenizer, row) for row in batch],
                           padding=True, return_tensors="pt")
        out = model(**inputs, output_hidden_states=True, use_cache=False)
        for j, row in enumerate(batch):
            logits = out.logits[j, -1]
            answer = tokenizer.decode([int(logits.argmax())]).strip()
            records.append({**row, "first_token": answer,
                            "valid": answer in (row["actor"], row["patient"]),
                            "correct": answer == row["gold"],
                            "actor_margin": core.margin_from_logits(
                                logits, tokenizer, row["actor"], row["patient"])})
            h20.append(out.hidden_states[20][j, -1].cpu().numpy())
        if (start + len(batch)) % 48 == 0:
            print(f"readout {start + len(batch)}/{len(items)}", flush=True)
    lookup = {(r["pair_index"], r["verb"], r["direction"],
               r["voice"], r["wording"]): i for i, r in enumerate(records)}
    selected = []
    for i, row in enumerate(records):
        if row["wording"] != "generic_actor" or row["correct"] or not row["valid"]:
            continue
        common = row["pair_index"], row["verb"], row["direction"], row["voice"]
        actor = lookup[*common, "performed_actor"]
        patient = lookup[*common, "verb_patient"]
        if records[actor]["correct"] and records[patient]["correct"]:
            selected.append((i, actor, patient))
    patches = []
    for n, (recipient, actor_donor, patient_donor) in enumerate(selected, 1):
        row = records[recipient]
        actor = core.patched_margin(row, h20[actor_donor], 20, tokenizer, model)
        patient = core.patched_margin(row, h20[patient_donor], 20, tokenizer, model)
        patches.append({"recipient": recipient, "actor_donor": actor_donor,
                        "patient_donor": patient_donor, "actor_patch": actor,
                        "patient_patch": patient,
                        "actor_minus_patient_margin": actor["margin"] - patient["margin"]})
        if n % 8 == 0:
            print(f"patches {n}/{len(selected)}", flush=True)
    result = {"protocol_sha256": sha(ROOT / "PROTOCOL.json"),
              "records": records, "patches": patches}
    (ROOT / "RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"patches {len(selected)}/{len(selected)}", flush=True)


def summarize() -> dict:
    data = json.loads((ROOT / "RESULT.json").read_text())
    assert data["protocol_sha256"] == sha(ROOT / "PROTOCOL.json")
    records, patches = data["records"], data["patches"]
    cells = {}
    for wording in WORDINGS:
        for voice in VOICES:
            sub = [r for r in records if r["wording"] == wording and r["voice"] == voice]
            cells[f"{wording}/{voice}"] = {"correct": sum(r["correct"] for r in sub),
                                         "valid": sum(r["valid"] for r in sub),
                                         "n": len(sub)}
    by_pair = []
    for i in range(len(PAIRS)):
        sub = [p for p in patches if records[p["recipient"]]["pair_index"] == i]
        by_pair.append({"pair": i, "n": len(sub),
                        "mean_actor_minus_patient_margin": float(np.mean([
                            p["actor_minus_patient_margin"] for p in sub])) if sub else None})
    values = [p["mean_actor_minus_patient_margin"] for p in by_pair]
    if all(v is not None for v in values):
        observed = abs(float(np.mean(values)))
        p = sum(abs(float(np.mean(np.asarray(values) * signs))) >= observed - 1e-12
                for signs in itertools.product((-1, 1), repeat=len(values))) / 2 ** len(values)
    else:
        p = None
    report = {"cells": cells, "selected_cases": len(patches),
              "actor_donor_actor_answers": sum(p["actor_patch"]["correct"] for p in patches),
              "patient_donor_actor_answers": sum(p["patient_patch"]["correct"] for p in patches),
              "pair_rows": by_pair,
              "pair_equal_weight_mean_margin_difference": float(np.mean(values)) if p is not None else None,
              "exact_two_sided_pair_signflip_p": p}
    (ROOT / "SUMMARY.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run", "summarize"))
    args = parser.parse_args()
    if args.action == "prepare": prepare()
    elif args.action == "run": run()
    else: print(json.dumps(summarize(), indent=2))


if __name__ == "__main__":
    main()
