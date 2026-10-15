"""Evaluate frozen, human-reviewed active/passive role pairs only.

This script refuses to run until freeze_review.py has created FROZEN_ITEMS.json.
One output file is written per checkpoint. Existing outputs are never replaced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

import llm_role_natural_repair as first
import llm_role_natural_repair_replication as second
from llm_role_natural_contrast_screen import score
from llm_role_natural_prompt_screen import QWEN05

ROOT = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(value: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", value.casefold()).split())


def surface_span(sentence: str, provisional: str) -> str:
    """Return the sentence's own capitalization and punctuation for a span."""
    match = re.search(re.escape(provisional), sentence, flags=re.I)
    if match:
        return match.group(0)
    # The final reviewed labels must be literal contiguous phrases. Stop
    # rather than silently score a different string.
    raise ValueError(f"Reviewed span {provisional!r} not literal in {sentence!r}")


def load_items():
    frozen_path = ROOT / "FROZEN_ITEMS.json"
    if not frozen_path.exists():
        raise RuntimeError("Independent human review and freeze_review.py are required before inference")
    obj = json.loads(frozen_path.read_text())
    assert obj["status"] == "frozen_before_challenge_inference"
    assert obj["review_sha256"] == sha(ROOT / "BLIND_HUMAN_REVIEW.csv")
    assert obj["protocol_sha256"] == sha(ROOT / "PROTOCOL.json")
    assert obj["n_accepted"] == len(obj["items"])
    return obj


def load_checkpoint(name: str, protocol: dict):
    cfg = protocol["checkpoints"][name]
    if name == "untouched_qwen05":
        tok = AutoTokenizer.from_pretrained(QWEN05, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            QWEN05, local_files_only=True, dtype=torch.float32).eval()
        return tok, model
    adapter = (ROOT / cfg["adapter"]).resolve()
    assert sha(adapter) == cfg["sha256"]
    tok, model = second.load_source() if name.startswith("fresh3") else first.load_source()
    source_name = name.endswith("_source")
    if not source_name:
        weights = torch.load(adapter, map_location="cpu", weights_only=True)
        params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        assert weights.keys() == params.keys()
        for key, value in weights.items():
            params[key].data.copy_(value)
    return tok, model.eval()


def make_prompt(tok, sentence: str, verb: str, question: str) -> str:
    return tok.apply_chat_template([
        {"role": "system", "content": "Answer with one exact phrase from the sentence, with no explanation."},
        {"role": "user", "content": f"Sentence: {sentence}\nTarget verb: {verb}\nQuestion: {question}\nAnswer:"},
    ], tokenize=False, add_generation_prompt=True)


@torch.inference_mode()
def generate(tok, model, prompt: str) -> str:
    prefix = tok.encode(prompt, add_special_tokens=False)
    generated = model.generate(input_ids=torch.tensor([prefix]),
                               max_new_tokens=24, do_sample=False,
                               pad_token_id=tok.eos_token_id,
                               eos_token_id=tok.eos_token_id,
                               use_cache=True)
    return tok.decode(generated[0, len(prefix):], skip_special_tokens=True).strip()


@torch.inference_mode()
def evaluate_form(tok, model, item: dict, form: str, wording: int, questions: dict):
    sentence = item["sentences"][form]
    actor = surface_span(sentence, item["actor_span"])
    other = surface_span(sentence, item["other_span"])
    actor_first = sentence.casefold().find(actor.casefold()) < sentence.casefold().find(other.casefold())
    prefix = {role: make_prompt(tok, sentence, item["verb"], questions[f"wording_{wording}_{role}"])
              for role in ("actor", "other")}
    likelihoods = {role: {candidate: score(tok, model, prefix[role], span)[0]
                           for candidate, span in (("actor", actor), ("other", other))}
                   for role in ("actor", "other")}
    margin = (likelihoods["actor"]["actor"] - likelihoods["actor"]["other"]
              - likelihoods["other"]["actor"] + likelihoods["other"]["other"])
    actor_answer = generate(tok, model, prefix["actor"])
    other_answer = generate(tok, model, prefix["other"])
    actor_exact = normalize(actor_answer) == normalize(actor)
    other_exact = normalize(other_answer) == normalize(other)
    patient = None
    if item["semantic_patient"] == "yes":
        pp = make_prompt(tok, sentence, item["verb"], questions[f"wording_{wording}_patient"])
        patient_answer = generate(tok, model, pp)
        patient = {"raw": patient_answer,
                   "exact": normalize(patient_answer) == normalize(other)}
    return {"sentence": sentence, "actor_gold": actor, "other_gold": other,
            "actor_is_first_mentioned_participant": actor_first,
            "candidate_margin": margin, "candidate_correct": margin > 0,
            "candidate_logprobs": likelihoods,
            "actor": {"raw": actor_answer, "exact": actor_exact},
            "other": {"raw": other_answer, "exact": other_exact},
            "patient": patient, "both_roles_correct": actor_exact and other_exact}


def summarize(records: list[dict]) -> dict:
    out = {}
    for form in ("active", "passive"):
        for wording in (1, 2):
            cells = [r["forms"][form][str(wording)] for r in records]
            patients = [x["patient"] for x in cells if x["patient"] is not None]
            out[f"{form}_wording{wording}"] = {
                "n": len(cells),
                "candidate_correct": sum(x["candidate_correct"] for x in cells),
                "actor_exact": sum(x["actor"]["exact"] for x in cells),
                "other_exact": sum(x["other"]["exact"] for x in cells),
                "both_roles_correct": sum(x["both_roles_correct"] for x in cells),
                "patient_exact": sum(x["exact"] for x in patients),
                "patient_n": len(patients),
                "first_participant_shortcut_actor_correct": sum(x["actor_is_first_mentioned_participant"] for x in cells),
            }
            for status in ("yes", "no", "unsure"):
                subset = [r["forms"][form][str(wording)] for r in records
                          if r["semantic_patient"] == status]
                out[f"{form}_wording{wording}_patient_{status}"] = {
                    "n": len(subset),
                    "candidate_correct": sum(x["candidate_correct"] for x in subset),
                    "actor_exact": sum(x["actor"]["exact"] for x in subset),
                    "other_exact": sum(x["other"]["exact"] for x in subset),
                    "both_roles_correct": sum(x["both_roles_correct"] for x in subset),
                }
    for wording in (1, 2):
        out[f"both_forms_wording{wording}"] = sum(
            r["forms"]["active"][str(wording)]["both_roles_correct"]
            and r["forms"]["passive"][str(wording)]["both_roles_correct"]
            for r in records)
    return out


def run(name: str) -> None:
    frozen = load_items()
    protocol = json.loads((ROOT / "PROTOCOL.json").read_text())
    target = ROOT / f"EVAL_{name}.json"
    if target.exists():
        raise FileExistsError(target)
    torch.set_num_threads(2)
    tok, model = load_checkpoint(name, protocol)
    records = []
    for i, item in enumerate(frozen["items"], 1):
        forms = {form: {str(wording): evaluate_form(tok, model, item, form, wording,
                                                   protocol["questions"])
                        for wording in (1, 2)}
                 for form in ("active", "passive")}
        records.append({"id": item["id"], "semantic_patient": item["semantic_patient"],
                        "forms": forms})
        if i % 5 == 0:
            print(f"{name}: {i}/{len(frozen['items'])}", flush=True)
    output = {"checkpoint": name, "frozen_items_sha256": sha(ROOT / "FROZEN_ITEMS.json"),
              "protocol_sha256": sha(ROOT / "PROTOCOL.json"),
              "summary": summarize(records), "records": records}
    target.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(output["summary"], indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", choices=("untouched_qwen05", "fresh1_source", "fresh1_natural",
                                                "fresh3_source", "fresh3_natural", "fresh1_copy_control"))
    run(parser.parse_args().checkpoint)
