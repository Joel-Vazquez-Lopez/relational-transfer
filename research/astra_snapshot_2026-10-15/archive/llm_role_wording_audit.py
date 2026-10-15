"""Fresh actor-question wording audit; fixed design before local inference."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_role_confirmation_v1 import prompt


HERE = Path(__file__).resolve().parent
ROOT = HERE / "llm_role_wording_audit"
MODELS = {
    "1.5b": Path("/private/tmp/rolepilot_hf/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"),
    "0.5b": Path("/Users/joelvazquezlopez/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"),
}
NAME_CANDIDATES = ("Stephen", "Alexander", "Sean", "Alan", "Allen", "Bruce", "Carl", "Louis", "Roy", "Juan", "Jose", "Carlos", "Ali", "Kevin", "Jason", "Scott", "Ryan", "Adam", "Tyler", "Leo", "Benjamin", "Frank", "Martin", "Oscar", "Isaac", "Ethan", "Albert", "Felix", "Simon", "Noah", "Arthur", "Luke")
VERBS = ("coached", "interviewed", "mentored", "consulted")
QUESTIONS = {
    "did": "Who did the action?",
    "performed": "Who performed the action?",
    "carried_out": "Which person carried out the action described in the sentence?",
    "actor": "Who is the actor in this sentence?",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare() -> None:
    ROOT.mkdir(exist_ok=True)
    if (ROOT / "PROTOCOL.json").exists():
        raise FileExistsError(ROOT / "PROTOCOL.json")
    tokenizers = {key: AutoTokenizer.from_pretrained(path, local_files_only=True)
                  for key, path in MODELS.items()}
    names = [name for name in NAME_CANDIDATES if all(
        len(tok.encode(name, add_special_tokens=False)) == 1 and
        len(tok.encode(" " + name, add_special_tokens=False)) == 1
        for tok in tokenizers.values())][:16]
    if len(names) < 16:
        raise ValueError(f"Only {len(names)} one-token names found")
    pairs = list(zip(names[::2], names[1::2]))
    rows = []
    for pair_id, ((left, right), verb, direction) in enumerate(itertools.product(pairs, VERBS, (0, 1))):
        actor, patient = ((left, right), (right, left))[direction]
        event = f"{names.index(left)//2}:{verb}:{direction}"
        for form, wording in itertools.product(("active", "passive"), QUESTIONS):
            sentence = (f"{actor} {verb} {patient}." if form == "active"
                        else f"{patient} was {verb} by {actor}.")
            rows.append({"pair": names.index(left)//2, "event": event,
                         "actor": actor, "patient": patient, "verb": verb,
                         "direction": direction, "form": form, "wording": wording,
                         "sentence": sentence, "question": QUESTIONS[wording]})
    assert len(rows) == 512 and len({row["event"] for row in rows}) == 64
    protocol = {
        "date": "2026-10-10", "status": "fixed_before_inference",
        "purpose": "Check whether the prior Qwen actor-question effect persists with two clearer phrasings and a fresh event set; compare two local Qwen2.5 instruction models.",
        "models": {key: {"path": str(path), "weights_sha256": sha(path / "model.safetensors")}
                   for key, path in MODELS.items()},
        "source_sha256": sha(HERE / "llm_role_wording_audit.py"),
        "design": "8 disjoint one-token name pairs x 4 new reversible verbs x 2 role directions = 64 events. Active/passive x four actor-question wordings = 8 prompts per event. Names, verbs, event instances new relative to confirmation set.",
        "names": names, "verbs": VERBS, "questions": QUESTIONS,
        "scoring": "Greedy next-token answer; exact actor name is correct. Record invalid names. Same fixed system instruction as prior confirmation.",
        "primary": "Accuracy and all-eight-correct rate by model, form and wording, reporting each of 8 pairs. The key sensitivity check is whether the active did-to-performed gap remains and whether the carried_out and actor questions match performed on active items. No significance threshold is predeclared for these exploratory new wording contrasts.",
        "limits": "Template English, one model family, no direct pretraining exposure audit; actor wording is a linguistic evaluation variable, not a training repair.",
    }
    (ROOT / "PROTOCOL.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (ROOT / "ITEMS.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(f"Prepared {len(rows)} prompts and 64 events with names: {names}", flush=True)


@torch.inference_mode()
def run(key: str) -> None:
    protocol = json.loads((ROOT / "PROTOCOL.json").read_text())
    assert sha(HERE / "llm_role_wording_audit.py") == protocol["source_sha256"]
    rows = json.loads((ROOT / "ITEMS.json").read_text())
    path = ROOT / f"RESULT_{key}.json"
    if path.exists():
        raise FileExistsError(path)
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(MODELS[key], local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODELS[key], local_files_only=True,
                                                dtype=torch.float32).eval()
    records = []
    for start in range(0, len(rows), 8):
        batch = rows[start:start + 8]
        inputs = tokenizer([prompt(tokenizer, row) for row in batch],
                           padding=True, return_tensors="pt")
        scores = model(**inputs, use_cache=False).logits[:, -1]
        for row, logits in zip(batch, scores):
            answer = tokenizer.decode([int(logits.argmax())]).strip()
            records.append({**row, "answer": answer,
                            "valid": answer in (row["actor"], row["patient"]),
                            "correct": answer == row["actor"]})
        if (start + len(batch)) % 64 == 0:
            print(f"{key}: {start + len(batch)}/{len(rows)}", flush=True)
    path.write_text(json.dumps({"protocol_sha256": sha(ROOT / "PROTOCOL.json"),
                                "records": records}, indent=2) + "\n")
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--model", choices=MODELS)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    else:
        if args.model is None:
            parser.error("--model is required for run")
        run(args.model)
