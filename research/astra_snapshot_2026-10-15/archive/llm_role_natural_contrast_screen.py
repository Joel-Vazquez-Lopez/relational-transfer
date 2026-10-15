"""Development-only span-likelihood contrast without A/B options."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_role_natural_prompt_screen import QWEN05, QWEN15, rows as dev_rows

HERE = Path(__file__).resolve().parent
ROOT = HERE / "llm_role_natural_contrast_screen"
ITEMS = HERE / "llm_role_authentic_spans/ITEMS.json"
QUESTIONS = {
    "actor": "Who is the actor of the target verb, meaning the person or group that carries out its action?",
    "other": "Who is the direct object of the target verb, meaning the other person or group involved in its action?",
}


def events() -> list[dict]:
    dev_ids = {row["id"] for row in dev_rows()}
    grouped = defaultdict(dict)
    for row in json.loads(ITEMS.read_text()):
        if row["id"] in dev_ids:
            grouped[row["id"]][row["role"]] = row
    result = []
    for identity, pair in grouped.items():
        a, b = pair["subject_answer"], pair["other_participant_answer"]
        result.append({"id": identity, "sentence": a["sentence"], "verb": a["verb"],
                       "actor_span": a["gold_span"], "other_span": b["gold_span"]})
    assert len(result) == 12
    return result


def prompt(tokenizer, row: dict, role: str) -> str:
    return tokenizer.apply_chat_template([
        {"role": "system", "content": "Answer with a phrase from the sentence."},
        {"role": "user", "content": f"Sentence: {row['sentence']}\nTarget verb: {row['verb']}\nQuestion: {QUESTIONS[role]}\nAnswer:"},
    ], tokenize=False, add_generation_prompt=True)


@torch.inference_mode()
def score(tokenizer, model, prefix: str, answer: str) -> tuple[float, float]:
    p = tokenizer.encode(prefix, add_special_tokens=False)
    a = tokenizer.encode(answer, add_special_tokens=False)
    assert p and a
    ids = torch.tensor([p + a])
    logits = model(input_ids=ids, use_cache=False).logits[0]
    continuation = torch.tensor(a)
    selected = logits[len(p)-1:len(p)+len(a)-1]
    value = selected.log_softmax(-1).gather(-1, continuation[:, None]).sum().item()
    return value, value / len(a)


def run(name: str) -> None:
    ROOT.mkdir(exist_ok=True)
    path = ROOT / f"{name}.json"
    if path.exists():
        raise FileExistsError(path)
    model_path = {"qwen05": QWEN05, "qwen15": QWEN15}[name]
    torch.set_num_threads(2)
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
                                               dtype=torch.float32).eval()
    records = []
    for row in events():
        scores = {question: {candidate: score(tok, model, prompt(tok, row, question),
                                              row[f"{candidate}_span"])
                             for candidate in ("actor", "other")}
                  for question in QUESTIONS}
        by_sum = ((scores["actor"]["actor"][0] - scores["actor"]["other"][0])
                  - (scores["other"]["actor"][0] - scores["other"]["other"][0]))
        by_mean = ((scores["actor"]["actor"][1] - scores["actor"]["other"][1])
                   - (scores["other"]["actor"][1] - scores["other"]["other"][1]))
        records.append({**row, "scores": scores, "contrast_sum": by_sum,
                        "contrast_mean": by_mean, "correct_sum": by_sum > 0,
                        "correct_mean": by_mean > 0})
    path.write_text(json.dumps({"model": name, "questions": QUESTIONS,
                                "records": records}, indent=2) + "\n")
    print(name, "sum", sum(r["correct_sum"] for r in records), "/", len(records),
          "mean", sum(r["correct_mean"] for r in records), "/", len(records), flush=True)


if __name__ == "__main__":
    run(sys.argv[1])
