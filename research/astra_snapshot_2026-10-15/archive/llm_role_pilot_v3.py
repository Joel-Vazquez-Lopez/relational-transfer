"""Independent, fixed local Qwen role-prompt check on new events."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE / "llm_role_pilot_v3"
MODEL = "/private/tmp/rolepilot_hf/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
PAIRS = (("Clara", "Simon"), ("Priya", "Lucas"), ("Elena", "David"),
         ("Fatima", "Noah"), ("Sofia", "Isaac"), ("Hana", "Oliver"),
         ("Rosa", "Daniel"), ("Amina", "Peter"))
VERBS = ("photographed", "questioned", "interviewed", "followed")
FORMS = ("active", "passive")
PROMPTS = ("generic", "performed", "joint")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cases() -> list[dict]:
    rows = []
    for pair_index, (a, b) in enumerate(PAIRS):
        for verb in VERBS:
            for direction, (actor, patient) in enumerate(((a, b), (b, a))):
                event = f"{pair_index}:{verb}:{direction}"
                for form in FORMS:
                    sentence = (f"{actor} {verb} {patient}." if form == "active" else
                                f"{patient} was {verb} by {actor}.")
                    for prompt in PROMPTS:
                        question = {
                            "generic": "Who did the action?",
                            "performed": "Who performed the action?",
                            "joint": "Identify the person who acted and the person acted on.",
                        }[prompt]
                        rows.append(dict(event=event, pair_index=pair_index,
                                         verb=verb, direction=direction, actor=actor,
                                         patient=patient, form=form, prompt=prompt,
                                         sentence=sentence, question=question))
    assert len(rows) == 384 and len({r["event"] for r in rows}) == 64
    return rows


def protocol() -> dict:
    return {"status": "fixed_before_model_run", "date": "2026-10-09",
            "model": "Qwen2.5-1.5B-Instruct; local unmodified weights",
            "model_config_sha256": sha(Path(MODEL) / "config.json"),
            "source_sha256": sha(HERE / "llm_role_pilot_v3.py"),
            "design": "8 new name pairs x 4 reversible verbs x 2 directions = 64 events; active/passive x 3 prompts = 384 related measurements",
            "prompts": list(PROMPTS), "generation": "greedy, max 16 new tokens; batch size 4",
            "scoring": "one-name prompts require exact one-name output after terminal punctuation; joint prompt requires actor=NAME; patient=NAME, terminal punctuation tolerated; invalid outputs are errors and counted separately",
            "main_outcome": "event-level both active and passive actor answers correct for each one-name wording; descriptive, no significance test",
            "secondary": "accuracy by form and prompt, joint both roles correct, paired answer changes across the two actor wordings, first-name and last-name baselines",
            "scope": "exploratory transfer of evaluation idea to one pretrained model; no claim about unseen pretraining examples, training repair, or internal mechanism"}


def prepare() -> None:
    ROOT.mkdir(exist_ok=True)
    path = ROOT / "PROTOCOL.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(protocol(), indent=2) + "\n")
    (ROOT / "ITEMS.json").write_text(json.dumps(cases(), indent=2) + "\n")
    print(json.dumps({"protocol": str(path), "rows": 384, "events": 64}, indent=2))


def ask(tokenizer, row: dict) -> str:
    if row["prompt"] == "joint":
        system = "Answer only in this format: actor=NAME; patient=NAME. Use the names in the sentence."
    else:
        system = "Answer with exactly one of the two names in the sentence. Output only the name."
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": f"Sentence: {row['sentence']}\nQuestion: {row['question']}"}]
    return tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)


def judge(row: dict, raw: str) -> dict:
    names = {row["actor"], row["patient"]}
    clean = raw.strip().rstrip(". !\n")
    if row["prompt"] == "joint":
        match = re.fullmatch(r"actor=([A-Za-z]+);\s*patient=([A-Za-z]+)", clean)
        valid = bool(match and {match[1], match[2]} == names)
        return {"valid": valid, "correct": bool(valid and match[1] == row["actor"]
                                                 and match[2] == row["patient"]),
                "answer": [match[1], match[2]] if match else None}
    valid = clean in names
    return {"valid": valid, "correct": bool(valid and clean == row["actor"]),
            "answer": clean if valid else None}


def summary(rows: list[dict]) -> dict:
    cells = {}
    for prompt in PROMPTS:
        for form in FORMS:
            selected = [r for r in rows if r["prompt"] == prompt and r["form"] == form]
            cells[f"{prompt}/{form}"] = {"correct": sum(r["correct"] for r in selected),
                                         "valid": sum(r["valid"] for r in selected),
                                         "n": len(selected)}
    pairs = {(r["event"], r["prompt"], r["form"]): r for r in rows}
    both = {prompt: sum(pairs[e, prompt, "active"]["correct"] and
                        pairs[e, prompt, "passive"]["correct"]
                        for e in {r["event"] for r in rows})
            for prompt in PROMPTS}
    changed = sum(pairs[e, "generic", f]["answer"] !=
                  pairs[e, "performed", f]["answer"]
                  for e in {r["event"] for r in rows} for f in FORMS)
    return {"cells": cells, "both_forms_correct_out_of_64": both,
            "actor_wording_changed_answer_out_of_128": changed,
            "first_name_actor_baseline_out_of_128": sum(
                r["actor"] == r["sentence"].split()[0] for r in rows
                if r["prompt"] == "generic"),
            "last_name_actor_baseline_out_of_128": sum(
                r["actor"] == re.findall(r"[A-Z][a-z]+", r["sentence"])[-1]
                for r in rows if r["prompt"] == "generic")}


def run() -> None:
    assert json.loads((ROOT / "PROTOCOL.json").read_text()) == protocol()
    rows = json.loads((ROOT / "ITEMS.json").read_text())
    assert rows == cases()
    out = ROOT / "RESULT.json"
    if out.exists():
        raise FileExistsError(out)
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, local_files_only=True, dtype=torch.float32).eval()
    completed = []
    partial = ROOT / "PARTIAL.json"
    if partial.exists():
        completed = json.loads(partial.read_text())
        assert completed == rows[:len(completed)] or all(
            all(done.get(k) == row[k] for k in row)
            for done, row in zip(completed, rows))
    for start in range(len(completed), len(rows), 4):
        batch = rows[start:start + 4]
        inputs = tokenizer([ask(tokenizer, row) for row in batch],
                           padding=True, return_tensors="pt")
        with torch.inference_mode():
            generated = model.generate(**inputs, do_sample=False,
                                       max_new_tokens=16,
                                       pad_token_id=tokenizer.pad_token_id)
        decoded = tokenizer.batch_decode(
            generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        for row, raw in zip(batch, decoded):
            completed.append({**row, "raw": raw, **judge(row, raw)})
        if len(completed) % 32 == 0:
            partial.write_text(json.dumps(completed, indent=2) + "\n")
            print(f"completed {len(completed)}/384", flush=True)
    result = {"protocol_sha256": sha(ROOT / "PROTOCOL.json"),
              "summary": summary(completed), "rows": completed}
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run", "summarize"))
    args = parser.parse_args()
    if args.action == "prepare": prepare()
    elif args.action == "run": run()
    else: print(json.dumps(json.loads((ROOT / "RESULT.json").read_text())["summary"], indent=2))


if __name__ == "__main__":
    main()
