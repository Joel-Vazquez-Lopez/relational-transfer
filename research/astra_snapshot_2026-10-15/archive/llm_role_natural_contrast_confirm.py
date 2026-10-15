"""Frozen held-out role contrast on authentic QA-SRL Gold sentences."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE / "llm_role_natural_contrast_confirm"
SOURCE = HERE / "llm_role_authentic_spans"
SCREEN = HERE / "llm_role_natural_contrast_screen"
CONFIRM = HERE / "llm_role_repair_confirmation"
UNORDERED = HERE / "llm_role_unordered_control"
QWEN05 = Path("/Users/joelvazquezlopez/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775")
QWEN15 = Path("/private/tmp/rolepilot_hf/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
WORLDS = ("fresh1", "fresh2", "fresh3")
SEEDS = (830211, 830212)
ARMS = ("pretrained05", "pretrained15", "ordinary", "ordered_roles", "unordered_pair")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def events() -> list[dict]:
    items = json.loads((SOURCE / "ITEMS.json").read_text())
    dev_ids = set(json.loads((SCREEN / "qwen05.json").read_text())["records"][i]["id"]
                  for i in range(12))
    assert dev_ids == {r["id"] for r in json.loads((SCREEN / "qwen15.json").read_text())["records"]}
    grouped = defaultdict(dict)
    for row in items:
        if row["id"] not in dev_ids:
            grouped[row["id"]][row["role"]] = row
    result = []
    for identity, pair in grouped.items():
        a, b = pair["subject_answer"], pair["other_participant_answer"]
        result.append({"id": identity, "partition": a["partition"],
                       "sentence": a["sentence"], "verb": a["verb"],
                       "actor_span": a["gold_span"], "other_span": b["gold_span"]})
    assert len(result) == 51 and not dev_ids.intersection(r["id"] for r in result)
    return result


def reference_hashes() -> dict:
    refs = {
        "source_items": SOURCE / "ITEMS.json",
        "development_05": SCREEN / "qwen05.json",
        "development_15": SCREEN / "qwen15.json",
        "screen_code": HERE / "llm_role_natural_contrast_screen.py",
        "prompt_code": HERE / "llm_role_natural_prompt_screen.py",
        "model05_weights": QWEN05 / "model.safetensors",
        "model15_weights": QWEN15 / "model.safetensors",
        "confirmation_protocol": CONFIRM / "PROTOCOL.json",
        "unordered_protocol": UNORDERED / "PROTOCOL.json",
    }
    return {key: sha(path) for key, path in refs.items()}


def protocol() -> dict:
    return {
        "date": "2026-10-11", "status": "fixed_before_heldout_contrast_inference",
        "source": "Previously selected QA-SRL Gold 63 events: 12 development events, stratified three per source partition, used only for untouched-model screen; 51 remaining events for this comparison.",
        "development": "The untouched 0.5B model gave 10/12 positive raw-sum role contrasts; untouched 1.5B gave 12/12. These observations motivated this frozen held-out check, not any fine-tuned-arm selection.",
        "task": "For each event and each of two fixed questions (actor versus other direct participant), score each of the two annotated answer spans as a continuation with the sentence and target verb in context. No A/B options and no free generation.",
        "primary": "Role-contrast raw-sum margin D = [logP(actorSpan|actorQ)-logP(otherSpan|actorQ)] - [logP(actorSpan|otherQ)-logP(otherSpan|otherQ)]. Correct iff D>0, zero wrong. The same spans appear under both questions, cancelling much of the candidate prior and answer-length effect. Compare paired fine-tuned arms over 51 held-out events and six graph/start units; report every unit and equal-unit mean.",
        "secondary": "Mean-token-logprob version of D, raw margins, source partition, untouched 0.5B and 1.5B baselines. No-sentence baselines use the same question/target verb and candidates but replace the sentence with '[omitted]' as a lexical-prior diagnostic. No-sentence is run only on untouched models.",
        "criterion": "Interpret a transfer gain only if the untouched 0.5B model reaches >=70% on the 51 holdout events, the no-sentence baseline is at least 15 percentage points lower, and an auxiliary arm exceeds ordinary by >=10 points in >=5/6 paired units without a negative equal-unit mean margin shift. Otherwise report the observed direction without a transfer-repair claim.",
        "limits": "Candidate spans were drawn from QA-SRL Gold and AI-screened; no independent student verification. This is candidate-relative likelihood, not direct answer generation. One selected sentence set shared across all graph/start units; model pretraining overlap unknown. Other participant is not uniformly a semantic patient.",
        "reference_sha256": reference_hashes(),
        "script_sha256": sha(HERE / "llm_role_natural_contrast_confirm.py"),
    }


def prepare() -> None:
    ROOT.mkdir(exist_ok=True)
    if (ROOT / "PROTOCOL.json").exists():
        raise FileExistsError(ROOT / "PROTOCOL.json")
    (ROOT / "EVENTS.json").write_text(json.dumps(events(), indent=2) + "\n")
    (ROOT / "PROTOCOL.json").write_text(json.dumps(protocol(), indent=2) + "\n")
    print("Frozen 51 held-out events, two questions and two candidate spans")


def validate() -> list[dict]:
    assert json.loads((ROOT / "PROTOCOL.json").read_text()) == protocol()
    saved = json.loads((ROOT / "EVENTS.json").read_text())
    assert saved == events()
    return saved


def adapter_path(world: str, seed: int, arm: str) -> Path:
    return ((UNORDERED / f"{world}_{seed}_unordered_pair" / "adapter.pt") if arm == "unordered_pair"
            else CONFIRM / f"{world}_{seed}_{arm}" / "adapter.pt")


def run(world: str, seed: int, arm: str, no_sentence: bool = False,
        limit: int | None = None) -> None:
    if arm.startswith("pretrained"):
        assert world == "fresh1" and seed == 0
    else:
        assert world in WORLDS and seed in SEEDS and not no_sentence
    rows = validate()
    if limit is not None:
        assert 0 < limit <= len(rows)
        rows = rows[:limit]
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import llm_role_repair_confirm as confirm
    import llm_role_repair_v3 as core
    from llm_role_natural_contrast_screen import QUESTIONS, prompt, score

    label = f"{world}_{seed}_{arm}" + ("_no_sentence" if no_sentence else "")
    if limit is not None:
        label += f"_smoke{limit}"
    target = ROOT / f"{label}.json"
    if target.exists():
        raise FileExistsError(target)
    adapter_hash = None
    torch.set_num_threads(2)
    if arm == "pretrained15":
        tokenizer = AutoTokenizer.from_pretrained(QWEN15, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(QWEN15, local_files_only=True,
                                                   dtype=torch.float32).eval()
    else:
        confirm.configure(world, seed)
        tokenizer, model, _ = core.load("pretrained" if arm == "pretrained05" else "ordinary")
        if arm != "pretrained05":
            path = adapter_path(world, seed, arm)
            adapter_hash = sha(path)
            weights = torch.load(path, map_location="cpu", weights_only=True)["adapter"]
            params = {name: p for name, p in model.named_parameters() if p.requires_grad}
            assert weights.keys() == params.keys()
            for name, parameter in params.items():
                assert parameter.shape == weights[name].shape
                parameter.data.copy_(weights[name])
        model.eval()
    result = []
    for index, row in enumerate(rows, 1):
        context = {**row, "sentence": "[omitted]" if no_sentence else row["sentence"]}
        scores = {question: {candidate: score(tokenizer, model,
                                              prompt(tokenizer, context, question),
                                              row[f"{candidate}_span"])
                             for candidate in ("actor", "other")}
                  for question in QUESTIONS}
        raw = ((scores["actor"]["actor"][0] - scores["actor"]["other"][0])
               - (scores["other"]["actor"][0] - scores["other"]["other"][0]))
        mean = ((scores["actor"]["actor"][1] - scores["actor"]["other"][1])
                - (scores["other"]["actor"][1] - scores["other"]["other"][1]))
        result.append({"id": row["id"], "partition": row["partition"],
                       "actor_span": row["actor_span"], "other_span": row["other_span"],
                       "scores": scores, "margin_sum": raw, "margin_mean": mean,
                       "correct_sum": raw > 0, "correct_mean": mean > 0})
        if index % 10 == 0:
            print(f"{label}: {index}/{len(rows)}", flush=True)
    target.write_text(json.dumps({"label": label, "arm": arm, "world": world,
                                  "seed": seed, "no_sentence": no_sentence,
                                  "protocol_sha256": sha(ROOT / "PROTOCOL.json"),
                                  "events_sha256": sha(ROOT / "EVENTS.json"),
                                  "adapter_sha256": adapter_hash,
                                  "records": result}, indent=2) + "\n")
    print(f"Saved {label}: {sum(r['correct_sum'] for r in result)}/{len(result)}", flush=True)


def summarize() -> None:
    validate()
    full = [("fresh1", 0, "pretrained05", False),
            ("fresh1", 0, "pretrained15", False),
            ("fresh1", 0, "pretrained05", True),
            ("fresh1", 0, "pretrained15", True)] + [
                (w, s, a, False) for w in WORLDS for s in SEEDS
                for a in ("ordinary", "ordered_roles", "unordered_pair")]
    cells = {}
    for world, seed, arm, no_sentence in full:
        label = f"{world}_{seed}_{arm}" + ("_no_sentence" if no_sentence else "")
        path = ROOT / f"{label}.json"
        if not path.exists():
            continue
        obj = json.loads(path.read_text())
        assert obj["protocol_sha256"] == sha(ROOT / "PROTOCOL.json")
        assert obj["events_sha256"] == sha(ROOT / "EVENTS.json")
        if not arm.startswith("pretrained"):
            assert obj["adapter_sha256"] == sha(adapter_path(world, seed, arm))
        records = obj["records"]
        assert len(records) == 51
        cells[label] = {"n": 51, "correct_sum": sum(r["correct_sum"] for r in records),
                        "correct_mean": sum(r["correct_mean"] for r in records),
                        "mean_margin_sum": sum(r["margin_sum"] for r in records)/51,
                        "mean_margin_mean": sum(r["margin_mean"] for r in records)/51}
    pairs = []
    for world in WORLDS:
        for seed in SEEDS:
            baseline = cells.get(f"{world}_{seed}_ordinary")
            if baseline is None:
                continue
            for arm in ("ordered_roles", "unordered_pair"):
                x = cells.get(f"{world}_{seed}_{arm}")
                if x is None:
                    continue
                pairs.append({"world": world, "seed": seed, "arm": arm,
                              "accuracy_delta_pp": 100 * (x["correct_sum"]-baseline["correct_sum"])/51,
                              "margin_delta": x["mean_margin_sum"]-baseline["mean_margin_sum"]})
    report = {"complete": len(cells) == 22, "cells": cells, "paired_differences": pairs}
    (ROOT / "ANALYSIS.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def batch() -> None:
    validate()
    jobs = [("fresh1", 0, "pretrained05", False),
            ("fresh1", 0, "pretrained15", False),
            ("fresh1", 0, "pretrained05", True),
            ("fresh1", 0, "pretrained15", True)] + [
                (w, s, a, False) for w in WORLDS for s in SEEDS
                for a in ("ordinary", "ordered_roles", "unordered_pair")]
    for world, seed, arm, no_sentence in jobs:
        label = f"{world}_{seed}_{arm}" + ("_no_sentence" if no_sentence else "")
        if (ROOT / f"{label}.json").exists():
            print(f"Already complete: {label}", flush=True)
            continue
        args = [sys.executable, str(HERE / "llm_role_natural_contrast_confirm.py"),
                "run", "--world", world, "--seed", str(seed), "--arm", arm]
        if no_sentence:
            args.append("--no-sentence")
        subprocess.run(args, check=True)
    summarize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "validate", "run", "batch", "analyze"))
    parser.add_argument("--world", choices=WORLDS, default="fresh1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arm", choices=ARMS, default="pretrained05")
    parser.add_argument("--no-sentence", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.action == "prepare": prepare()
    elif args.action == "validate":
        validate(); print("Frozen 51 holdout events validated")
    elif args.action == "run": run(args.world, args.seed, args.arm,
                                   args.no_sentence, args.limit)
    elif args.action == "batch": batch()
    else: summarize()
