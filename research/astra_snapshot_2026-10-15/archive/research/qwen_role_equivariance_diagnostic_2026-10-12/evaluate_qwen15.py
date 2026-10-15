#!/usr/bin/env python3
"""Prospective larger-model supplement on the unchanged fixed item set."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import evaluate as base
from llm_role_natural_prompt_screen import QWEN15

ROOT = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    target = ROOT / "untouched_qwen15.json"
    if target.exists():
        raise FileExistsError(target)
    torch.set_num_threads(2)
    tok = AutoTokenizer.from_pretrained(QWEN15, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(QWEN15, local_files_only=True,
                                                dtype=torch.float32).eval()
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    for a, b in base.config()["pairs"]:
        base.name_ids(tok, a)
        base.name_ids(tok, b)
    rows = base.items()
    answers = []

    @torch.inference_mode()
    def score(block: list[dict], omitted: bool) -> None:
        prompts = [base.prompt(tok, x, omitted) for x in block]
        inputs = tok(prompts, padding=True, return_tensors="pt")
        logits = model(**inputs, use_cache=False).logits[:, -1]
        for i, row in enumerate(block):
            record = {"id": row["id"], "pair_index": row["pair_index"],
                      "direction": row["direction"], "verb": row["verb"],
                      "form": row["form"], "wording": row["wording"],
                      "role": row["role"],
                      "actor_margin": base.margin(logits[i], tok, row),
                      "omitted": omitted}
            if not omitted:
                response = tok.decode([int(logits[i].argmax())]).strip()
                record.update({"answer": response, "gold": row["gold"],
                               "valid": response in (row["actor"], row["patient"]),
                               "correct": response == row["gold"]})
            answers.append(record)

    for i in range(0, len(rows), 8):
        score(rows[i:i + 8], omitted=False)
        if (i + 8) % 96 == 0:
            print(f"qwen15: present {i+8}/{len(rows)}", flush=True)
    omitted_rows = [r for r in rows if r["form"] == "active"]
    for i in range(0, len(omitted_rows), 8):
        score(omitted_rows[i:i + 8], omitted=True)
    payload = {"model": "untouched_qwen15",
               "protocol_sha256": sha(ROOT / "PROTOCOL.json"),
               "supplement_sha256": sha(ROOT / "QWEN15_SUPPLEMENT.json"),
               "script_sha256": sha(Path(__file__)),
               "model_source": str(QWEN15),
               "model_source_sha256": sha(QWEN15 / "model.safetensors"),
               "summary": base.summarize_records(answers), "records": answers}
    target.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
