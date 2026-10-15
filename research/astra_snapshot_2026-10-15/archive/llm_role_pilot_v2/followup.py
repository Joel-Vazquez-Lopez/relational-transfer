"""Adaptive joint-role follow-up; see FOLLOWUP_PROTOCOL.json."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from run import ROOT, MODELS, cases

PROTOCOL_PATH = ROOT / "FOLLOWUP_PROTOCOL.json"
FORMAT = re.compile(r"^actor\s*=\s*([A-Za-z]+)\s*;\s*patient\s*=\s*([A-Za-z]+)\s*$", re.I)


def rows() -> list[dict]:
    unique = {}
    for row in cases():
        unique[(row["event_id"], row["voice"])] = {
            key: row[key] for key in ("event_id", "agent", "patient", "voice", "sentence")}
    output = []
    for item in unique.values():
        for condition in ("joint_no_example", "joint_with_example"):
            output.append({**item, "condition": condition})
    assert len(output) == 96
    return output


def prompt(tokenizer, row: dict) -> str:
    system = ("Identify the roles in the sentence. The actor performs the action; "
              "the patient is the person the action happens to. "
              "Answer using exactly this format: actor=<name>; patient=<name>. "
              "Write nothing else.")
    if row["condition"] == "joint_with_example":
        system += (" Example: In 'Tara was chased by Zed', the answer is "
                   "actor=Zed; patient=Tara.")
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": f"Sentence: {row['sentence']}"}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def summarize(data: list[dict]) -> dict:
    groups = defaultdict(list)
    for row in data:
        groups[(row["condition"], row["voice"])].append(row)
        groups[(row["condition"], "all")].append(row)
    return {"/".join(key): {
        "total": len(values),
        "valid": sum(r["valid"] for r in values),
        "actor_correct": sum(r["actor_correct"] for r in values),
        "patient_correct": sum(r["patient_correct"] for r in values),
        "joint_correct": sum(r["joint_correct"] for r in values),
    } for key, values in groups.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    args = parser.parse_args()
    model_id, path = MODELS[args.model]
    output_path = ROOT / f"{args.model}_followup_result.json"
    if output_path.exists():
        raise FileExistsError(output_path)
    torch.set_num_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True,
                                                 dtype=torch.float32).eval()
    data = rows()
    for start in range(0, len(data), 4):
        batch = data[start:start + 4]
        inputs = tokenizer([prompt(tokenizer, row) for row in batch],
                           padding=True, return_tensors="pt")
        with torch.inference_mode():
            result = model.generate(**inputs, do_sample=False, max_new_tokens=32,
                                    pad_token_id=tokenizer.pad_token_id)
        answers = tokenizer.batch_decode(result[:, inputs["input_ids"].shape[1]:],
                                         skip_special_tokens=True)
        for row, raw in zip(batch, answers):
            match = FORMAT.fullmatch(raw.strip())
            actor, patient = match.groups() if match else (None, None)
            row.update({"raw": raw, "parsed_actor": actor, "parsed_patient": patient,
                        "valid": bool(match),
                        "actor_correct": actor == row["agent"],
                        "patient_correct": patient == row["patient"],
                        "joint_correct": actor == row["agent"] and patient == row["patient"]})
        if (start + len(batch)) % 24 == 0:
            print(f"{model_id}: {start + len(batch)}/{len(data)} prompts", flush=True)
    output = {"model": model_id,
              "protocol_sha256": hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest(),
              "summary": summarize(data), "rows": data}
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output["summary"], indent=2), flush=True)
    print(f"Saved {output_path}", flush=True)


if __name__ == "__main__":
    main()
