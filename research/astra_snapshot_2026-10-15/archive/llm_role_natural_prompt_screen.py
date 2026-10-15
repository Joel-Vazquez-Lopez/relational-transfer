"""Development-only prompt screen for authentic actor span extraction.

The 12 examples are stratified across four QA-SRL Gold source partitions.
No fine-tuned adapter is evaluated here; prompt choice uses untouched models only.
"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE / "llm_role_natural_prompt_screen"
ITEMS = HERE / "llm_role_authentic_spans/ITEMS.json"
QWEN05 = Path("/Users/joelvazquezlopez/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775")
QWEN15 = Path("/private/tmp/rolepilot_hf/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
PROMPTS = {
    "actor_plain": "Who did the action expressed by the target verb? Copy only the exact words naming that person or group from the sentence.",
    "actor_gloss": "Identify the actor of the target verb: the person or group that carries out that action. Reply with only the exact phrase from the sentence.",
    "actor_example": "Example: Sentence: The fox chased the dog. Target verb: chased. Actor: The fox. Now identify the actor of the target verb in the new sentence. Reply with only the exact phrase from that sentence.",
}


def rows() -> list[dict]:
    events = [row for row in json.loads(ITEMS.read_text()) if row["role"] == "subject_answer"]
    by_partition = defaultdict(list)
    for event in events:
        by_partition[event["partition"]].append(event)
    selected = [row for partition in sorted(by_partition) for row in by_partition[partition][:3]]
    assert len(selected) == 12 and len({x["id"] for x in selected}) == 12
    return selected


def prompt(tokenizer, row: dict, wording: str) -> str:
    return tokenizer.apply_chat_template([
        {"role": "system", "content": PROMPTS[wording]},
        {"role": "user", "content": f"Sentence: {row['sentence']}\nTarget verb: {row['verb']}\nAnswer:"},
    ], tokenize=False, add_generation_prompt=True)


def run(model_name: str) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from llm_role_authentic_spans import norm

    ROOT.mkdir(exist_ok=True)
    destination = ROOT / f"{model_name}.json"
    if destination.exists():
        raise FileExistsError(destination)
    model_path = {"qwen05": QWEN05, "qwen15": QWEN15}[model_name]
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
                                                dtype=torch.float32).eval()
    prompts = [(row, wording) for row in rows() for wording in PROMPTS]
    result = []
    with torch.inference_mode():
        for start in range(0, len(prompts), 4):
            batch = prompts[start:start+4]
            enc = tokenizer([prompt(tokenizer, row, wording) for row, wording in batch],
                            padding=True, return_tensors="pt")
            generated = model.generate(**enc, do_sample=False, max_new_tokens=18,
                                       pad_token_id=tokenizer.pad_token_id,
                                       eos_token_id=tokenizer.eos_token_id)
            new = generated[:, enc["input_ids"].shape[1]:]
            for (row, wording), tokens in zip(batch, new, strict=True):
                answer = tokenizer.decode(tokens, skip_special_tokens=True).strip()
                result.append({"id": row["id"], "partition": row["partition"],
                               "wording": wording, "gold": row["gold_span"],
                               "answer": answer, "exact": norm(answer) == norm(row["gold_span"]),
                               "copied": bool(norm(answer)) and norm(answer) in norm(row["sentence"])})
    destination.write_text(json.dumps({"model": model_name, "prompts": PROMPTS,
                                       "development_ids": [row["id"] for row in rows()],
                                       "records": result}, indent=2) + "\n")
    for wording in PROMPTS:
        subset = [r for r in result if r["wording"] == wording]
        print(model_name, wording, "exact", sum(r["exact"] for r in subset),
              "/", len(subset), "copied", sum(r["copied"] for r in subset), flush=True)


if __name__ == "__main__":
    run(sys.argv[1])
