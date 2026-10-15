"""Matched actor-question phrase check; see WORDING_PROTOCOL.json."""
from __future__ import annotations

import argparse
import hashlib
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from run import ROOT, MODELS, cases, judge, prompt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    args = parser.parse_args()
    model_id, path = MODELS[args.model]
    output_path = ROOT / f"{args.model}_wording_result.json"
    if output_path.exists():
        raise FileExistsError(output_path)
    rows = [dict(row) for row in cases()
            if row["role"] == "actor" and row["wording"] == "generic"]
    assert len(rows) == 48
    for row in rows:
        assert row["question"] == "Who did the action?"
        row["question"] = "Who performed the action?"
    torch.set_num_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True,
                                                 dtype=torch.float32).eval()
    for start in range(0, len(rows), 4):
        batch = rows[start:start + 4]
        inputs = tokenizer([prompt(tokenizer, row) for row in batch],
                           padding=True, return_tensors="pt")
        with torch.inference_mode():
            output = model.generate(**inputs, do_sample=False, max_new_tokens=8,
                                    pad_token_id=tokenizer.pad_token_id)
        answers = tokenizer.batch_decode(output[:, inputs["input_ids"].shape[1]:],
                                         skip_special_tokens=True)
        for row, raw in zip(batch, answers):
            answer, valid = judge(raw, row)
            row.update({"raw": raw, "answer": answer, "valid": valid,
                        "correct": valid and answer == row["gold"]})
        if (start + len(batch)) % 24 == 0:
            print(f"{model_id}: {start + len(batch)}/48 prompts", flush=True)
    previous = json.loads((ROOT / f"{args.model}_result.json").read_text())
    old = {(r["event_id"], r["voice"]): r for r in previous["rows"]
           if r["role"] == "actor" and r["wording"] == "generic"}
    assert len(old) == 48
    by_voice = {}
    for voice in ("active", "passive"):
        subset = [r for r in rows if r["voice"] == voice]
        by_voice[voice] = {
            "total": len(subset),
            "old_correct": sum(old[(r["event_id"], voice)]["correct"] for r in subset),
            "new_correct": sum(r["correct"] for r in subset),
            "changed_answer": sum(r["answer"] != old[(r["event_id"], voice)]["answer"]
                                  for r in subset),
            "new_last_name": sum(r["answer"] == r["sentence"].strip(".").split()[-1]
                                 for r in subset),
            "new_valid": sum(r["valid"] for r in subset),
        }
    result = {"model": model_id,
              "protocol_sha256": hashlib.sha256((ROOT / "WORDING_PROTOCOL.json").read_bytes()).hexdigest(),
              "summary": by_voice, "rows": rows}
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(by_voice, indent=2), flush=True)


if __name__ == "__main__":
    main()
