"""Post-EWT diagnostic: same natural spans and updates, without role questions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F

import llm_role_natural_repair as first
import llm_role_independent_eval as ewt
from llm_role_natural_contrast_screen import prompt, score

HERE = Path(__file__).resolve().parent
ROOT = HERE / "llm_role_copy_control"
ADAPTER = ROOT / "adapter.pt"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def copy_prompt(tok, event, span):
    return tok.apply_chat_template([
        {"role": "system", "content": "Answer with a phrase from the sentence."},
        {"role": "user", "content": f"Sentence: {event['sentence']}\nTarget verb: {event['verb']}\nQuestion: Copy this phrase exactly: {span}\nAnswer:"},
    ], tokenize=False, add_generation_prompt=True)


def batch(tok, event):
    entries = [first.encode_answer(tok, copy_prompt(tok, event, event[f"{role}_span"]),
                                   event[f"{role}_span"]) for role in ("actor", "other")]
    length = max(len(x[0]) for x in entries)
    ids, labels, masks = [], [], []
    for x, y in entries:
        pad = length - len(x)
        ids.append([tok.pad_token_id] * pad + x)
        labels.append([-100] * pad + y)
        masks.append([0] * pad + [1] * len(x))
    return torch.tensor(ids), torch.tensor(labels), torch.tensor(masks)


def freeze():
    if ROOT.exists():
        raise FileExistsError(ROOT)
    train, _, _ = first.validate()
    events = ewt.validate()
    ROOT.mkdir()
    protocol = {
        "status": "post_EWT_first_start_result_diagnostic_fixed_before_copy_training",
        "date": "2026-10-12",
        "design": "Start from the same fresh1 participant-only adapter. Use the identical ordered 256 natural QA-SRL train events, two gold answer spans per update, 256 AdamW updates at LR 5e-5, and no synthetic replay. Replace the actor/object question with an explicit copy request naming the exact gold span. This preserves natural sentences, answer strings, update count, and answer-format practice, but removes any requirement to infer role from the question. Prompt tokens are not identical and this is an imperfect matched control.",
        "evaluation": "Evaluate frozen 75-item EWT cross-corpus primary candidate contrast under original role questions, sentence-omitted contrast, and 130-item Bank candidate contrast. Compare source, copy control, and natural-role adaptation per item. No exclusions. The first EWT natural-only result was already seen, so this is diagnostic, not blind confirmation.",
        "interpretation": "A copy-control gain as large as role-adaptation gain would weaken a role-specific interpretation. A smaller gain would support, but not prove, a role-question contribution because training prompts differ.",
        "n_train": len(train), "n_ewt": len(events),
        "references": {"source_adapter_sha256": sha(first.SOURCE_ADAPTER),
                       "train_sha256": sha(first.ROOT / "TRAIN.json"),
                       "ewt_protocol_sha256": sha(ewt.ROOT / "PROTOCOL.json"),
                       "ewt_events_sha256": sha(ewt.ROOT / "EVENTS.json"),
                       "script_sha256": sha(HERE / "llm_role_copy_control.py")}}
    (ROOT / "PROTOCOL.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print("Frozen copy-control protocol")


def validate():
    train, _, _ = first.validate()
    events = ewt.validate()
    p = json.loads((ROOT / "PROTOCOL.json").read_text())
    refs = {"source_adapter_sha256": first.SOURCE_ADAPTER,
            "train_sha256": first.ROOT / "TRAIN.json",
            "ewt_protocol_sha256": ewt.ROOT / "PROTOCOL.json",
            "ewt_events_sha256": ewt.ROOT / "EVENTS.json",
            "script_sha256": HERE / "llm_role_copy_control.py"}
    assert all(p["references"][k] == sha(v) for k, v in refs.items())
    assert p["n_train"] == len(train) and p["n_ewt"] == len(events)
    return train, events


def train():
    examples, _ = validate()
    if ADAPTER.exists():
        raise FileExistsError(ADAPTER)
    torch.set_num_threads(2)
    tok, model = first.load_source()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=first.LR)
    losses = []
    start = time.perf_counter()
    model.train()
    for i, event in enumerate(examples, 1):
        ids, labels, masks = batch(tok, event)
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids=ids, attention_mask=masks, use_cache=False).logits
        targets = labels[:, 1:].contiguous()
        token_loss = F.cross_entropy(logits[:, :-1].contiguous().flatten(0, 1),
                                     targets.flatten(), ignore_index=-100,
                                     reduction="none").view(2, -1)
        loss = (token_loss.sum(dim=1)/(targets != -100).sum(dim=1)).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(i)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        if i % 32 == 0:
            print(f"copy control {i}/{len(examples)} elapsed {time.perf_counter()-start:.0f}s", flush=True)
    torch.save({n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad}, ADAPTER)
    (ROOT / "TRAINING.json").write_text(json.dumps({"protocol_sha256": sha(ROOT / "PROTOCOL.json"),
        "adapter_sha256": sha(ADAPTER), "losses": losses,
        "elapsed_seconds": time.perf_counter()-start}, indent=2)+"\n")


@torch.inference_mode()
def contrast(tok, model, row):
    scores = {role: {candidate: score(tok, model, prompt(tok, row, role),
                                      row[f"{candidate}_span"])[0]
                     for candidate in ("actor", "other")}
              for role in ("actor", "other")}
    return (scores["actor"]["actor"]-scores["actor"]["other"]
            -scores["other"]["actor"]+scores["other"]["other"])


def evaluate():
    _, events = validate()
    target = ROOT / "EVALUATION.json"
    if target.exists():
        raise FileExistsError(target)
    torch.set_num_threads(2)
    tok, model = first.load_source()
    weights = torch.load(ADAPTER, map_location="cpu", weights_only=True)
    params = {n:p for n,p in model.named_parameters() if p.requires_grad}
    assert params.keys() == weights.keys()
    for name, value in weights.items():
        params[name].data.copy_(value)
    model.eval()
    result = {}
    for label, rows in (("ewt", events), ("bank", json.loads(
            (first.ROOT / "fresh_bank_test/EVENTS.json").read_text()))):
        records = []
        for i, row in enumerate(rows, 1):
            rec = {"id": row["id"], "primary": contrast(tok, model, row)}
            if label == "ewt":
                rec["omitted"] = contrast(tok, model, {**row, "sentence": "[omitted]"})
            records.append(rec)
            if i % 25 == 0:
                print(f"copy control {label} {i}/{len(rows)}", flush=True)
        result[label] = records
    output = {"protocol_sha256": sha(ROOT / "PROTOCOL.json"),
              "adapter_sha256": sha(ADAPTER), "records": result,
              "summary": {k: {"primary": sum(x["primary"] > 0 for x in v),
                               "omitted": sum(x["omitted"] > 0 for x in v) if k == "ewt" else None}
                          for k,v in result.items()}}
    target.write_text(json.dumps(output, indent=2)+"\n")
    print(json.dumps(output["summary"], indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "validate", "train", "evaluate"))
    action = parser.parse_args().action
    if action == "freeze":
        freeze()
    elif action == "validate":
        validate(); print("validated")
    elif action == "train":
        train()
    else:
        evaluate()
