"""Fixed local active/passive actor-patient pilot. See PROTOCOL.json."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import string

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
PROTOCOL = json.loads((ROOT / "PROTOCOL.json").read_text())
PAIRS = (("Mira", "Owen"), ("Leila", "Victor"),
         ("Nadia", "Felix"), ("Iris", "Amir"))
VERBS = ("chased", "helped", "followed")
MODELS = {
    "qwen15": ("Qwen/Qwen2.5-1.5B-Instruct",
               "/private/tmp/rolepilot_hf/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"),
    "qwen05": ("Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"),
}


def cases() -> list[dict]:
    rows = []
    for pair_index, (left, right) in enumerate(PAIRS):
        for verb in VERBS:
            for agent, patient in ((left, right), (right, left)):
                event_id = f"{pair_index}:{verb}:{agent}"
                for voice in ("active", "passive"):
                    sentence = (f"{agent} {verb} {patient}." if voice == "active"
                                else f"{patient} was {verb} by {agent}.")
                    for role in ("actor", "patient"):
                        for wording in ("generic", "verb_specific"):
                            if wording == "generic":
                                question = ("Who did the action?" if role == "actor"
                                            else "Who was acted on?")
                            else:
                                question = (f"Who {verb} the other person?" if role == "actor"
                                            else f"Who was {verb}?")
                            rows.append({"event_id": event_id, "pair": [left, right],
                                         "verb": verb, "agent": agent, "patient": patient,
                                         "voice": voice, "role": role, "wording": wording,
                                         "sentence": sentence, "question": question,
                                         "gold": agent if role == "actor" else patient})
    assert len(rows) == PROTOCOL["design"]["prompts_per_model"] == 192
    assert len({r["event_id"] for r in rows}) == 24
    return rows


def prompt(tokenizer, row: dict) -> str:
    messages = [
        {"role": "system", "content": "Answer the question with exactly one of the two names in the sentence. Output only the name."},
        {"role": "user", "content": f"Sentence: {row['sentence']}\nQuestion: {row['question']}"},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def judge(raw: str, row: dict) -> tuple[str, bool]:
    answer = raw.strip().strip(string.punctuation + " ")
    valid = answer in (row["agent"], row["patient"])
    return answer, valid


def summary(rows: list[dict]) -> dict:
    cells = defaultdict(list)
    for row in rows:
        for key in (("overall",), ("voice", row["voice"]),
                    ("role", row["role"]), ("wording", row["wording"]),
                    ("voice_role_wording", row["voice"], row["role"], row["wording"])):
            cells[key].append(row)
    groups = {"/".join(key): {"correct": sum(r["correct"] for r in values),
                             "valid": sum(r["valid"] for r in values),
                             "total": len(values),
                             "accuracy": sum(r["correct"] for r in values) / len(values)}
              for key, values in cells.items()}
    paired = defaultdict(dict)
    for row in rows:
        paired[(row["event_id"], row["role"], row["wording"])][row["voice"]] = row
    both_voices = sum(items["active"]["correct"] and items["passive"]["correct"]
                      for items in paired.values())
    wording_pairs = defaultdict(dict)
    for row in rows:
        wording_pairs[(row["event_id"], row["role"], row["voice"])][row["wording"]] = row
    both_wordings = sum(items["generic"]["correct"] and items["verb_specific"]["correct"]
                        for items in wording_pairs.values())
    first_name_correct = sum(row["gold"] == row["sentence"].split()[0] for row in rows)
    return {"groups": groups,
            "both_voices_correct": {"correct": both_voices, "total": len(paired)},
            "both_wordings_correct": {"correct": both_wordings, "total": len(wording_pairs)},
            "first_name_baseline": {"correct": first_name_correct, "total": len(rows)}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    args = parser.parse_args()
    model_id, path = MODELS[args.model]
    destination = ROOT / f"{args.model}_result.json"
    if destination.exists():
        raise FileExistsError(destination)
    torch.set_num_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True,
                                                 dtype=torch.float32).eval()
    rows = cases()
    for start in range(0, len(rows), 4):
        batch = rows[start:start + 4]
        prompts = [prompt(tokenizer, row) for row in batch]
        inputs = tokenizer(prompts, padding=True, return_tensors="pt")
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
            print(f"{model_id}: {start + len(batch)}/{len(rows)} prompts", flush=True)
    result = {"model": model_id, "protocol_sha256": hashlib.sha256(
        (ROOT / "PROTOCOL.json").read_bytes()).hexdigest(),
        "summary": summary(rows), "rows": rows}
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"model": model_id, "summary": result["summary"]}, indent=2), flush=True)
    print(f"Saved {destination}", flush=True)


if __name__ == "__main__":
    main()
