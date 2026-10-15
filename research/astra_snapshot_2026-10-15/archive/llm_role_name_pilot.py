"""Forced-choice name-score pilot for active/passive agent identification."""

import json
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PAIRS = [
    ("Alice", "Bob"),
    ("Emma", "Leo"),
    ("John", "Mary"),
    ("Peter", "Jane"),
    ("Sarah", "David"),
    ("Anna", "Tom"),
]
VERBS = ["chased", "helped", "followed"]
QUESTIONS = [
    "Who performed the action in this sentence? Answer with the person's name only.",
    "Who is the agent, meaning the person doing the action? Answer with one name only.",
]


def main():
    torch.set_num_threads(4)
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, local_files_only=True, dtype=torch.float32
    ).eval()
    rows = []
    for left, right in PAIRS:
        for verb in VERBS:
            for agent, patient in ((left, right), (right, left)):
                for voice in ("active", "passive"):
                    sentence = (
                        f"{agent} {verb} {patient}."
                        if voice == "active"
                        else f"{patient} was {verb} by {agent}."
                    )
                    for question_id, question in enumerate(QUESTIONS):
                        messages = [
                            {"role": "system", "content": "Answer the question about the sentence accurately."},
                            {"role": "user", "content": f"Sentence: {sentence}\nQuestion: {question}"},
                        ]
                        rows.append(
                            {
                                "sentence": sentence,
                                "voice": voice,
                                "agent": agent,
                                "patient": patient,
                                "verb": verb,
                                "question_id": question_id,
                                "prompt": tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
                            }
                        )
    for start in range(0, len(rows), 8):
        batch = rows[start : start + 8]
        inp = tok([r["prompt"] for r in batch], padding=True, return_tensors="pt")
        with torch.inference_mode():
            logits = model(**inp).logits[:, -1, :]
        for index, row in enumerate(batch):
            def score(name):
                ids = [tok.encode(name, add_special_tokens=False), tok.encode(" " + name, add_special_tokens=False)]
                assert all(len(seq) == 1 for seq in ids)
                return torch.logsumexp(logits[index, [seq[0] for seq in ids]], dim=0).item()

            agent_score = score(row["agent"])
            patient_score = score(row["patient"])
            row["margin"] = agent_score - patient_score
            row["correct_prediction"] = row["margin"] > 0
            del row["prompt"]
        print(f"Scored {start + len(batch)}/{len(rows)}", flush=True)
    stats = defaultdict(list)
    for row in rows:
        stats[(row["voice"], row["question_id"])].append(row["correct_prediction"])
    summary = {
        f"{voice}_q{qid}": {"correct": sum(values), "total": len(values), "accuracy": sum(values) / len(values)}
        for (voice, qid), values in stats.items()
    }
    pairs = defaultdict(dict)
    for row in rows:
        pairs[(row["agent"], row["patient"], row["verb"], row["question_id"])][row["voice"]] = row["correct_prediction"]
    summary["both_voices_correct"] = {
        "correct": sum(value["active"] and value["passive"] for value in pairs.values()),
        "total": len(pairs),
    }
    output = Path("llm_role_name_pilot_result.json")
    output.write_text(json.dumps({"model": MODEL, "summary": summary, "rows": rows}, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
