"""Greedy generation pilot; checks whether the model actually gives a name."""

import json
import os
import re
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_role_name_pilot import MODEL as DEFAULT_MODEL, PAIRS, VERBS

MODEL = os.environ.get("ROLE_PILOT_MODEL", DEFAULT_MODEL)


def main():
    torch.set_num_threads(4)
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, dtype=torch.float32).eval()
    rows = []
    for left, right in PAIRS:
        for verb in VERBS:
            for agent, patient in ((left, right), (right, left)):
                for voice in ("active", "passive"):
                    sentence = f"{agent} {verb} {patient}." if voice == "active" else f"{patient} was {verb} by {agent}."
                    style = os.environ.get("ROLE_PILOT_STYLE", "generic")
                    question = (
                        f"Who {verb} the other person? Reply with the name only."
                        if style == "verb_specific"
                        else "Who performed the action? Reply with the name only."
                    )
                    prompt = tok.apply_chat_template(
                        [
                            {"role": "system", "content": "Answer questions about sentences. Give only the person's name."},
                            {"role": "user", "content": f"Sentence: {sentence}\n{question}"},
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    rows.append({"sentence": sentence, "agent": agent, "patient": patient, "voice": voice, "verb": verb, "prompt": prompt})
    for start in range(0, len(rows), 8):
        batch = rows[start : start + 8]
        inputs = tok([r["prompt"] for r in batch], padding=True, return_tensors="pt")
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=16, do_sample=False, pad_token_id=tok.eos_token_id)
        continuations = tok.batch_decode(output[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        for row, generated in zip(batch, continuations):
            row["generated"] = generated.strip()
            names = re.findall(r"\b(?:" + re.escape(row["agent"]) + r"|" + re.escape(row["patient"]) + r")\b", generated)
            row["correct_prediction"] = len(names) == 1 and names[0] == row["agent"]
            row["unambiguous"] = len(names) == 1
            del row["prompt"]
        print(f"Generated {start + len(batch)}/{len(rows)}", flush=True)
    summary = {}
    for voice in ("active", "passive"):
        subset = [r for r in rows if r["voice"] == voice]
        summary[voice] = {
            "correct": sum(r["correct_prediction"] for r in subset),
            "unambiguous": sum(r["unambiguous"] for r in subset),
            "total": len(subset),
        }
    pairs = defaultdict(dict)
    for row in rows:
        pairs[(row["agent"], row["patient"], row["verb"])][row["voice"]] = row["correct_prediction"]
    summary["both_voices_correct"] = {"correct": sum(v["active"] and v["passive"] for v in pairs.values()), "total": len(pairs)}
    model_suffix = "qwen15" if "1.5B" in MODEL else "qwen05"
    style_suffix = "verb_specific" if os.environ.get("ROLE_PILOT_STYLE") == "verb_specific" else "generic"
    output_name = f"llm_role_generation_{model_suffix}_{style_suffix}_result.json"
    Path(output_name).write_text(json.dumps({"model": MODEL, "summary": summary, "rows": rows}, indent=2))
    print(json.dumps(summary, indent=2))
    print("Examples:")
    for row in rows[:12]:
        print(row["sentence"], "=>", repr(row["generated"]))


if __name__ == "__main__":
    main()
