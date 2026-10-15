"""Evaluate frozen cross-corpus EWT items; write one immutable file per checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

import llm_role_independent_eval as data
import llm_role_natural_repair as first
import llm_role_natural_repair_replication as second
from llm_role_natural_contrast_screen import prompt as original_prompt, score
from llm_role_natural_repair_fresh_direct import normalize

HERE = Path(__file__).resolve().parent
ROOT = HERE / "llm_role_independent_eval"
CHECKPOINTS = {
    "fresh1_source": ("fresh1", first.SOURCE_ADAPTER),
    "fresh1_natural": ("fresh1", HERE / "llm_role_natural_repair/controls_fresh1_830211/natural_only/adapter.pt"),
    "fresh3_source": ("fresh3", second.SOURCE),
    "fresh3_natural": ("fresh3", HERE / "llm_role_natural_repair/natural_only_fresh3_830212/adapter.pt"),
}
QUESTIONS = {
    "actor": "Which phrase in the sentence names the doer of the target verb?",
    "other": "Which phrase in the sentence names the direct object of the target verb?",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paraphrase_prompt(tok, row, role):
    return tok.apply_chat_template([
        {"role": "system", "content": "Answer with a phrase from the sentence."},
        {"role": "user", "content": f"Sentence: {row['sentence']}\nTarget verb: {row['verb']}\nQuestion: {QUESTIONS[role]}\nAnswer:"},
    ], tokenize=False, add_generation_prompt=True)


def freeze():
    rows = data.validate()
    target = ROOT / "EVAL_PROTOCOL.json"
    if target.exists():
        raise FileExistsError(target)
    p = {
        "status": "fixed_before_any_EWT_checkpoint_inference",
        "date": "2026-10-12",
        "n_events": len(rows),
        "checkpoints": {k: {"source_type": v[0], "adapter_sha256": sha(v[1])}
                        for k, v in CHECKPOINTS.items()},
        "primary": "For all EWT events and both starts, compare adapted versus source D>0 under the original QA-SRL actor and other questions, with the sentence present. A pairwise improvement is correct-after/wrong-before minus wrong-after/correct-before. No success threshold or item exclusion is defined post-inference.",
        "secondary": "Sentence-omitted D>0 under original questions; sentence-present D>0 under frozen paraphrase questions; direct greedy generated actor and other spans under original questions with strict normalized exact match. Report per-checkpoint and paired results. Do not use EWT labels for training or model selection.",
        "model": "Qwen2.5-0.5B-Instruct float32 CPU, each source adapter compared with its same-start natural-only adaptation; greedy decoding max 16 new tokens. Candidate contrast uses raw summed log probabilities exactly as in prior Bank experiment.",
        "limits": "UD subject/object labels are provisional semantic-role proxies; human review CSV is unfilled. EWT corpus was chosen after QA-SRL outcomes; pretraining overlap unknown. No new fine-tuning in this run.",
        "references": {"data_protocol_sha256": sha(ROOT / "PROTOCOL.json"),
                       "events_sha256": sha(ROOT / "EVENTS.json"),
                       "evaluator_sha256": sha(HERE / "llm_role_independent_run.py")}}
    target.write_text(json.dumps(p, indent=2) + "\n")
    print("Frozen EWT checkpoint protocol")


def validate():
    rows = data.validate()
    p = json.loads((ROOT / "EVAL_PROTOCOL.json").read_text())
    assert p["n_events"] == len(rows)
    assert p["references"]["data_protocol_sha256"] == sha(ROOT / "PROTOCOL.json")
    assert p["references"]["events_sha256"] == sha(ROOT / "EVENTS.json")
    assert p["references"]["evaluator_sha256"] == sha(HERE / "llm_role_independent_run.py")
    for name, (_, path) in CHECKPOINTS.items():
        assert p["checkpoints"][name]["adapter_sha256"] == sha(path)
    return rows


def load(name):
    which, path = CHECKPOINTS[name]
    tok, model = first.load_source() if which == "fresh1" else second.load_source()
    if not name.endswith("_source"):
        weights = torch.load(path, map_location="cpu", weights_only=True)
        params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        assert weights.keys() == params.keys()
        for key, value in weights.items():
            params[key].data.copy_(value)
    return tok, model.eval()


@torch.inference_mode()
def margin(tok, model, row, make_prompt):
    s = {role: {candidate: score(tok, model, make_prompt(tok, row, role),
                                  row[f"{candidate}_span"])[0]
                for candidate in ("actor", "other")}
         for role in ("actor", "other")}
    return s["actor"]["actor"] - s["actor"]["other"] - s["other"]["actor"] + s["other"]["other"]


@torch.inference_mode()
def generated(tok, model, row, role):
    prefix = tok.encode(original_prompt(tok, row, role), add_special_tokens=False)
    output = model.generate(input_ids=torch.tensor([prefix]), max_new_tokens=16,
                            do_sample=False, pad_token_id=tok.eos_token_id,
                            eos_token_id=tok.eos_token_id, use_cache=True)
    answer = tok.decode(output[0, len(prefix):], skip_special_tokens=True).strip()
    return {"raw": answer, "gold": row[f"{role}_span"],
            "exact": normalize(answer) == normalize(row[f"{role}_span"])}


def run(name):
    rows = validate()
    target = ROOT / f"{name}.json"
    if target.exists():
        raise FileExistsError(target)
    torch.set_num_threads(2)
    tok, model = load(name)
    records = []
    for i, row in enumerate(rows, 1):
        omitted = {**row, "sentence": "[omitted]"}
        record = {"id": row["id"],
                  "primary": margin(tok, model, row, original_prompt),
                  "omitted": margin(tok, model, omitted, original_prompt),
                  "paraphrase": margin(tok, model, row, paraphrase_prompt),
                  "direct": {role: generated(tok, model, row, role)
                             for role in ("actor", "other")}}
        records.append(record)
        if i % 15 == 0:
            print(f"{name} {i}/{len(rows)}", flush=True)
    obj = {"checkpoint": name, "eval_protocol_sha256": sha(ROOT / "EVAL_PROTOCOL.json"),
           "events_sha256": sha(ROOT / "EVENTS.json"), "adapter_sha256": sha(CHECKPOINTS[name][1]),
           "records": records,
           "summary": {"primary": sum(r["primary"] > 0 for r in records),
                       "omitted": sum(r["omitted"] > 0 for r in records),
                       "paraphrase": sum(r["paraphrase"] > 0 for r in records),
                       "direct_actor": sum(r["direct"]["actor"]["exact"] for r in records),
                       "direct_other": sum(r["direct"]["other"]["exact"] for r in records),
                       "direct_both": sum(r["direct"]["actor"]["exact"] and r["direct"]["other"]["exact"] for r in records)}}
    target.write_text(json.dumps(obj, indent=2) + "\n")
    print(json.dumps(obj["summary"], indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "validate", "run"))
    parser.add_argument("--checkpoint", choices=tuple(CHECKPOINTS))
    args = parser.parse_args()
    if args.action == "freeze":
        freeze()
    elif args.action == "validate":
        print(f"Validated {len(validate())} EWT events")
    else:
        if not args.checkpoint:
            parser.error("--checkpoint required")
        run(args.checkpoint)
