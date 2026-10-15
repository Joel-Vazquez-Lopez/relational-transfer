"""Second-start confirmation screen of natural-only role adaptation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

import llm_role_natural_repair as first
import llm_role_natural_repair_fresh as fresh
import llm_role_natural_repair_replication as rep
import llm_role_repair_confirm as confirm
import llm_role_repair_v3 as core
from llm_role_natural_contrast_screen import prompt, score


HERE = Path(__file__).resolve().parent
ROOT = HERE / "llm_role_natural_repair/natural_only_fresh3_830212"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare() -> None:
    first.validate()
    fresh.validate()
    rep.validate()
    ROOT.mkdir(exist_ok=True)
    if (ROOT / "PROTOCOL.json").exists():
        raise FileExistsError(ROOT / "PROTOCOL.json")
    p = {"status": "fixed_before_natural_only_second_start", "date": "2026-10-12",
         "question": "Does removing synthetic replay preserve patient-cleft transfer while retaining the authentic-role gain on the second starting adapter?",
         "design": "Start from the identical fresh3/830212 participant-only adapter as the mixed second-world run. Train on the same frozen 256 natural Bank examples, same actor and other-span objectives, order, 256 updates and LR 5e-5. Omit only the synthetic replay gradient. The decision follows inspection of first-start replay controls and the mixed second-world outcome; all evaluations are exploratory reuse of fixed items.",
         "criterion": "Useful repeat if candidate role-contrast accuracy rises by at least 7/130 Bank test events and patient-cleft accuracy loses no more than 5 percentage points (at most 37 of 750 items). Also report sentence-omitted margin, Gold51, and active/passive accuracy without exclusions.",
         "limits": "One model family, same Bank data and questions as earlier runs, no new blind dataset, human label verification pending. This is a recipe screen rather than publication-level confirmation.",
         "references": {"source_adapter_sha256": sha(rep.SOURCE),
                        "first_protocol_sha256": sha(first.ROOT / "PROTOCOL.json"),
                        "train_sha256": sha(first.ROOT / "TRAIN.json"),
                        "fresh_protocol_sha256": sha(fresh.ROOT / "PROTOCOL.json"),
                        "replication_protocol_sha256": sha(rep.ROOT / "PROTOCOL.json"),
                        "script_sha256": sha(HERE / "llm_role_natural_only_replication.py")}}
    (ROOT / "PROTOCOL.json").write_text(json.dumps(p, indent=2) + "\n")
    print("Frozen natural-only second-start screen", flush=True)


def validate():
    train, _, _ = first.validate()
    events = fresh.validate()
    rep.validate()
    p = json.loads((ROOT / "PROTOCOL.json").read_text())
    paths = {"source_adapter_sha256": rep.SOURCE,
             "first_protocol_sha256": first.ROOT / "PROTOCOL.json",
             "train_sha256": first.ROOT / "TRAIN.json",
             "fresh_protocol_sha256": fresh.ROOT / "PROTOCOL.json",
             "replication_protocol_sha256": rep.ROOT / "PROTOCOL.json",
             "script_sha256": HERE / "llm_role_natural_only_replication.py"}
    assert all(p["references"][k] == sha(v) for k, v in paths.items())
    return train, events


def train() -> None:
    examples, _ = validate()
    if (ROOT / "adapter.pt").exists():
        raise FileExistsError(ROOT / "adapter.pt")
    torch.set_num_threads(2)
    tok, model = rep.load_source()
    groups, _ = confirm.validate(rep.WORLD)
    confirm.configure(rep.WORLD, rep.SEED)
    optimizer = torch.optim.AdamW((v for v in model.parameters() if v.requires_grad), lr=first.LR)
    rng = np.random.default_rng(first.SEED)
    order = [int(i) for i in rng.permutation(len(groups))]
    losses = []
    started = time.perf_counter()
    model.train()
    for step, event in enumerate(examples, 1):
        if (step - 1) % len(groups) == 0 and step > 1:
            order = [int(i) for i in rng.permutation(len(groups))]
        ids, labels, masks = first.batch(tok, event, groups[order[(step-1) % len(groups)]])
        ids, labels, masks = ids[:2], labels[:2], masks[:2]
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids=ids, attention_mask=masks, use_cache=False).logits
        targets = labels[:, 1:].contiguous()
        per_token = F.cross_entropy(logits[:, :-1].contiguous().flatten(0, 1),
                                    targets.flatten(), ignore_index=-100,
                                    reduction="none").view(2, -1)
        loss = (per_token.sum(dim=1) / (targets != -100).sum(dim=1)).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(step)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        if step % 32 == 0:
            print(f"natural-only second {step}/{len(examples)} elapsed {time.perf_counter()-started:.0f}s", flush=True)
    torch.save({name: value.detach().cpu() for name, value in model.named_parameters()
                if value.requires_grad}, ROOT / "adapter.pt")
    (ROOT / "TRAINING.json").write_text(json.dumps({"protocol_sha256": sha(ROOT / "PROTOCOL.json"),
                                                    "elapsed_seconds": time.perf_counter()-started,
                                                    "losses": losses}, indent=2)+"\n")
    print("Natural-only second-start training complete", flush=True)


@torch.inference_mode()
def evaluate() -> None:
    _, events = validate()
    target = ROOT / "EVALUATION.json"
    if target.exists():
        raise FileExistsError(target)
    torch.set_num_threads(2)
    tok, model = rep.load_source()
    weights = torch.load(ROOT / "adapter.pt", map_location="cpu", weights_only=True)
    params = {name: value for name, value in model.named_parameters() if value.requires_grad}
    assert weights.keys() == params.keys()
    for name, param in params.items():
        param.data.copy_(weights[name])
    model.eval()
    records = []
    for i, event in enumerate(events, 1):
        margins = {}
        for condition in ("with", "omitted"):
            row = event if condition == "with" else {**event, "sentence": "[omitted]"}
            scores = {role: {candidate: score(tok, model, prompt(tok, row, role),
                                              event[f"{candidate}_span"])[0]
                             for candidate in ("actor", "other")}
                      for role in ("actor", "other")}
            margins[condition] = (scores["actor"]["actor"] - scores["actor"]["other"]
                                  - scores["other"]["actor"] + scores["other"]["other"])
        records.append({"id": event["id"], **margins,
                        "correct": margins["with"] > 0,
                        "omitted_correct": margins["omitted"] > 0})
        if i % 25 == 0:
            print(f"natural-only second fresh {i}/{len(events)}", flush=True)
    gold = []
    for row in json.loads(rep.GOLD.read_text()):
        scores = {role: {candidate: score(tok, model, prompt(tok, row, role),
                                          row[f"{candidate}_span"])[0]
                         for candidate in ("actor", "other")}
                  for role in ("actor", "other")}
        d = scores["actor"]["actor"] - scores["actor"]["other"] - scores["other"]["actor"] + scores["other"]["other"]
        gold.append({"id": row["id"], "margin": d, "correct": d > 0})
    _, synthetic = confirm.validate(rep.WORLD)
    confirm.configure(rep.WORLD, rep.SEED)
    selected = [r for r in synthetic if r["form"] in ("patient_cleft", "active", "passive")]
    answers = core.evaluate("natural_only_second", tok, model, selected)
    cells = {form: [sum(r["correct"] for r in answers if r["form"] == form),
                    sum(r["form"] == form for r in answers)]
             for form in ("patient_cleft", "active", "passive")}
    obj = {"protocol_sha256": sha(ROOT / "PROTOCOL.json"), "fresh": records,
           "gold": gold, "synthetic": cells,
           "fresh_correct": sum(r["correct"] for r in records),
           "omitted_correct": sum(r["omitted_correct"] for r in records),
           "gold_correct": sum(r["correct"] for r in gold)}
    target.write_text(json.dumps(obj, indent=2)+"\n")
    print(json.dumps({"fresh": [obj["fresh_correct"], len(records)],
                      "omitted": [obj["omitted_correct"], len(records)],
                      "gold": [obj["gold_correct"], len(gold)],
                      "synthetic": cells}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "validate", "train", "evaluate"))
    action = parser.parse_args().action
    if action == "prepare": prepare()
    elif action == "validate": validate(); print("validated")
    elif action == "train": train()
    else: evaluate()
