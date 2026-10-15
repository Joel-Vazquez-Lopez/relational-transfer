"""Fresh, fixed Qwen role-answer confirmation. Prepare before running inference."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE / "llm_role_confirmation_v1"
MODEL = Path("/private/tmp/rolepilot_hf/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
SETS = (
    {"pairs": (("Charlotte", "Richard"), ("Victoria", "Christopher"),
               ("Ruby", "Matthew"), ("Jordan", "Jessica"), ("David", "Sarah")),
     "verbs": ("advised", "greeted", "supported", "questioned")},
    {"pairs": (("Michael", "Jennifer"), ("Thomas", "Philip"),
               ("Joseph", "Michelle"), ("Charles", "Karen"), ("Edward", "Anthony")),
     "verbs": ("warned", "hugged", "pushed", "hired")},
    {"pairs": (("George", "Elizabeth"), ("Paul", "Crystal"),
               ("Mark", "Ann"), ("Andrew", "Carol"), ("Steven", "Rose")),
     "verbs": ("blamed", "ignored", "noticed", "photographed")},
)
QUESTIONS = {"generic": "Who did the action?", "performed": "Who performed the action?"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def items() -> list[dict]:
    rows = []
    for set_id, block in enumerate(SETS):
        for pair_id, (left, right) in enumerate(block["pairs"]):
            for verb, direction in itertools.product(block["verbs"], (0, 1)):
                actor, patient = ((left, right), (right, left))[direction]
                event = f"{set_id}:{pair_id}:{verb}:{direction}"
                for form, wording in itertools.product(("active", "passive"), QUESTIONS):
                    sentence = (f"{actor} {verb} {patient}." if form == "active"
                                else f"{patient} was {verb} by {actor}.")
                    rows.append({"set": set_id, "pair": f"{set_id}:{pair_id}",
                                 "event": event, "actor": actor, "patient": patient,
                                 "verb": verb, "direction": direction, "form": form,
                                 "wording": wording, "sentence": sentence,
                                 "question": QUESTIONS[wording]})
    assert len(rows) == 480 and len({r["event"] for r in rows}) == 120
    return rows


def protocol() -> dict:
    return {
        "date": "2026-10-10", "status": "fixed_before_inference",
        "model": "Qwen2.5-1.5B-Instruct, fixed local snapshot, unmodified",
        "model_weights_sha256": sha(MODEL / "model.safetensors"),
        "source_sha256": sha(HERE / "llm_role_confirmation_v1.py"),
        "design": "Three disjoint sets: each has 5 new name pairs x 4 new reversible verbs x 2 actor directions = 40 events. Each event has active/passive x two fixed actor questions = 4 related prompts. Names and verbs do not overlap earlier Qwen diagnostic panels.",
        "scoring": "Greedy next token decoded and stripped. Correct iff exactly actor name; invalid answers are wrong and counted separately. Also record forced actor-minus-patient candidate-token logit margin. Both names have one token with/without leading space.",
        "primary_comparison": "For each event, performed-minus-generic correctness in active and passive. Wording-by-form interaction is (active performed - active generic) - (passive performed - passive generic). Report all 3 sets, and pair-clustered mean difference plus exact two-sided pair sign-flip p over 15 pairs. A replicated wording effect requires active performed-minus-generic >= 20 percentage points in each set and overall; otherwise call it mixed or absent.",
        "secondary": "Four-prompt all-correct rate per event; both-voices-correct and both-wordings-correct rates; correctness and invalid counts by cell. Agreement without correctness is separately counted. First-name heuristic is a baseline.",
        "limits": "Constructed English; one small pretrained model; prompt forms reused from diagnostics, but all names, verbs, pairings and event instances are new. No claim about model pretraining exposure or other LLMs.",
    }


def prepare() -> None:
    ROOT.mkdir(exist_ok=True)
    path = ROOT / "PROTOCOL.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(protocol(), indent=2) + "\n")
    (ROOT / "ITEMS.json").write_text(json.dumps(items(), indent=2) + "\n")
    print(f"Prepared {len(items())} prompts, 120 events: {path}", flush=True)


def prompt(tokenizer, row: dict) -> str:
    return tokenizer.apply_chat_template([
        {"role": "system", "content": "Answer with exactly one of the two names in the sentence. Output only the name."},
        {"role": "user", "content": f"Sentence: {row['sentence']}\nQuestion: {row['question']}"},
    ], tokenize=False, add_generation_prompt=True)


def name_ids(tokenizer, name: str) -> list[int]:
    return [tokenizer.encode(name, add_special_tokens=False)[0],
            tokenizer.encode(" " + name, add_special_tokens=False)[0]]


@torch.inference_mode()
def run() -> None:
    assert json.loads((ROOT / "PROTOCOL.json").read_text()) == protocol()
    rows = json.loads((ROOT / "ITEMS.json").read_text())
    assert rows == items()
    destination = ROOT / "RESULT.json"
    if destination.exists():
        raise FileExistsError(destination)
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    for name in {n for block in SETS for pair in block["pairs"] for n in pair}:
        assert len(tokenizer.encode(name, add_special_tokens=False)) == 1, name
        assert len(tokenizer.encode(" " + name, add_special_tokens=False)) == 1, name
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True,
                                               dtype=torch.float32).eval()
    records = []
    for start in range(0, len(rows), 4):
        batch = rows[start:start + 4]
        inputs = tokenizer([prompt(tokenizer, row) for row in batch],
                           padding=True, return_tensors="pt")
        scores = model(**inputs, use_cache=False).logits[:, -1]
        for row, logits in zip(batch, scores):
            answer = tokenizer.decode([int(logits.argmax())]).strip()
            a = torch.logsumexp(logits[name_ids(tokenizer, row["actor"])], dim=0)
            p = torch.logsumexp(logits[name_ids(tokenizer, row["patient"])], dim=0)
            records.append({**row, "answer": answer,
                            "valid": answer in (row["actor"], row["patient"]),
                            "correct": answer == row["actor"],
                            "actor_margin": float((a - p).item())})
        if (start + len(batch)) % 40 == 0:
            print(f"confirmed {start + len(batch)}/{len(rows)}", flush=True)
    destination.write_text(json.dumps({"protocol_sha256": sha(ROOT / "PROTOCOL.json"),
                                       "records": records}, indent=2) + "\n")
    print(f"Saved {destination}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run"))
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    else:
        run()


if __name__ == "__main__":
    main()
