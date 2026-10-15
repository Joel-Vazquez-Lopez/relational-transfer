"""Balanced, local-only feasibility check for agent identification.

Scores the next-token logit for A versus B; this is a forced-choice pilot,
not an unrestricted generation benchmark or a causal-intervention result.
"""

import json
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PAIRS = [("Alice", "Bob"), ("Emma", "Noah"), ("Nora", "Liam"), ("Maya", "Leo")]
VERBS = [("chased", "chased"), ("helped", "helped"), ("followed", "followed")]


def examples(tokenizer):
    for left, right in PAIRS:
        for past, participle in VERBS:
            for agent, patient in ((left, right), (right, left)):
                for voice in ("active", "passive"):
                    sentence = (
                        f"{agent} {past} {patient}."
                        if voice == "active"
                        else f"{patient} was {participle} by {agent}."
                    )
                    for option_a, option_b in ((left, right), (right, left)):
                        user = (
                            f"Sentence: {sentence}\n"
                            "Question: Who performed the action in the sentence?\n"
                            f"A. {option_a}\nB. {option_b}\n"
                            "Answer with A or B only."
                        )
                        prompt = tokenizer.apply_chat_template(
                            [
                                {"role": "system", "content": "Answer the question about the sentence. Reply with one letter only."},
                                {"role": "user", "content": user},
                            ],
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                        yield {
                            "sentence": sentence,
                            "voice": voice,
                            "agent": agent,
                            "patient": patient,
                            "verb": past,
                            "option_a": option_a,
                            "option_b": option_b,
                            "correct": "A" if agent == option_a else "B",
                            "prompt": prompt,
                        }


def main():
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, local_files_only=True, torch_dtype=torch.float32
    ).eval()
    a_id, b_id = tokenizer.encode("A", add_special_tokens=False), tokenizer.encode("B", add_special_tokens=False)
    assert len(a_id) == len(b_id) == 1
    rows = list(examples(tokenizer))
    for start in range(0, len(rows), 8):
        batch = rows[start : start + 8]
        tokenized = tokenizer(
            [row["prompt"] for row in batch], padding=True, return_tensors="pt"
        )
        with torch.inference_mode():
            logits = model(**tokenized).logits[:, -1, :]
        for row, score_a, score_b in zip(
            batch, logits[:, a_id[0]].tolist(), logits[:, b_id[0]].tolist()
        ):
            row["score_a"] = score_a
            row["score_b"] = score_b
            row["prediction"] = "A" if score_a > score_b else "B"
            row["correct_prediction"] = row["prediction"] == row["correct"]
            del row["prompt"]
        print(f"Scored {start + len(batch)}/{len(rows)}", flush=True)
    stats = defaultdict(list)
    for row in rows:
        stats[row["voice"]].append(row["correct_prediction"])
    summary = {key: {"correct": sum(values), "total": len(values), "accuracy": sum(values) / len(values)} for key, values in stats.items()}
    paired = defaultdict(dict)
    for row in rows:
        key = (row["verb"], row["agent"], row["patient"], row["option_a"])
        paired[key][row["voice"]] = row["correct_prediction"]
    summary["both_voices_correct"] = {
        "correct": sum(v.get("active", False) and v.get("passive", False) for v in paired.values()),
        "total": len(paired),
    }
    output = Path("llm_role_feasibility_result.json")
    output.write_text(json.dumps({"model": MODEL, "summary": summary, "rows": rows}, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
