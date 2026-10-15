"""Matched one-start controls for natural supervision and synthetic replay."""
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
import llm_role_repair_confirm as confirm
import llm_role_repair_v3 as core
from llm_role_natural_contrast_screen import prompt, score


HERE = Path(__file__).resolve().parent
ROOT = HERE / "llm_role_natural_repair/controls_fresh1_830211"
ARMS = ("natural_only", "replay_only")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare() -> None:
    first.validate()
    fresh.validate()
    ROOT.mkdir(exist_ok=True)
    if (ROOT / "PROTOCOL.json").exists():
        raise FileExistsError(ROOT / "PROTOCOL.json")
    p = {"status": "fixed_before_control_training", "date": "2026-10-12",
         "question": "Do natural role examples drive the authentic-language gain, and does synthetic replay protect controlled cleft transfer?",
         "design": "Both controls start from the identical fresh1/830211 participant-only adapter used by the mixed natural+replay pilot. Same 256 training steps, LR 5e-5, deterministic natural event order and synthetic replay-group order. Natural-only uses the same two annotated Bank questions/answers per step; replay-only uses the same four original synthetic examples per step with the same 0.3 replay coefficient. The mixed arm was already run; these controls were planned after seeing its test scores and are exploratory.",
         "evaluation": "Freeze all 130 Bank test events, role-contrast and sentence-omitted controls, plus 750 patient clefts and 1,500 active/passive prompts from the same fresh1 held-out-pair test. Report raw paired counts against source and mixed. No item exclusion.",
         "interpretation": "If natural-only and mixed both improve fresh Bank while replay-only does not, authentic supervision is necessary for this observed gain. If mixed retains more cleft accuracy than natural-only at similar natural accuracy, replay helps retention. Neither comparison isolates role-specific supervision from prompt-format learning or generic natural-language exposure.",
         "limits": "Post-result one-start controls using the already inspected Bank test; no new blind confirmation, no human label verification, one model family.",
         "references": {"source_adapter_sha256": sha(first.SOURCE_ADAPTER),
                        "first_protocol_sha256": sha(first.ROOT / "PROTOCOL.json"),
                        "train_sha256": sha(first.ROOT / "TRAIN.json"),
                        "fresh_protocol_sha256": sha(fresh.ROOT / "PROTOCOL.json"),
                        "fresh_events_sha256": sha(fresh.ROOT / "EVENTS.json"),
                        "script_sha256": sha(HERE / "llm_role_natural_repair_controls.py")}}
    (ROOT / "PROTOCOL.json").write_text(json.dumps(p, indent=2) + "\n")
    print("Frozen natural-only and replay-only controls", flush=True)


def validate():
    train, _, _ = first.validate()
    events = fresh.validate()
    p = json.loads((ROOT / "PROTOCOL.json").read_text())
    paths = {"source_adapter_sha256": first.SOURCE_ADAPTER,
             "first_protocol_sha256": first.ROOT / "PROTOCOL.json",
             "train_sha256": first.ROOT / "TRAIN.json",
             "fresh_protocol_sha256": fresh.ROOT / "PROTOCOL.json",
             "fresh_events_sha256": fresh.ROOT / "EVENTS.json",
             "script_sha256": HERE / "llm_role_natural_repair_controls.py"}
    assert all(p["references"][k] == sha(v) for k, v in paths.items())
    return train, events


def run(arm: str) -> None:
    train, _ = validate()
    out = ROOT / arm
    if out.exists():
        raise FileExistsError(out)
    torch.set_num_threads(2)
    tok, model = first.load_source()
    groups, _ = confirm.validate("fresh1")
    confirm.configure("fresh1", 830211)
    optimizer = torch.optim.AdamW((v for v in model.parameters() if v.requires_grad), lr=first.LR)
    rng = np.random.default_rng(first.SEED)
    order = [int(i) for i in rng.permutation(len(groups))]
    losses = []
    started = time.perf_counter()
    model.train()
    for step, event in enumerate(train, 1):
        if (step - 1) % len(groups) == 0 and step > 1:
            order = [int(i) for i in rng.permutation(len(groups))]
        ids, labels, masks = first.batch(tok, event, groups[order[(step-1) % len(groups)]])
        start = 0 if arm == "natural_only" else 2
        end = 2 if arm == "natural_only" else 6
        ids, labels, masks = ids[start:end], labels[start:end], masks[start:end]
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids=ids, attention_mask=masks, use_cache=False).logits
        targets = labels[:, 1:].contiguous()
        token_loss = F.cross_entropy(logits[:, :-1].contiguous().flatten(0, 1),
                                     targets.flatten(), ignore_index=-100,
                                     reduction="none").view(end-start, -1)
        loss = (token_loss.sum(dim=1) / (targets != -100).sum(dim=1)).mean()
        if arm == "replay_only":
            loss = first.REPLAY_WEIGHT * loss
        if not torch.isfinite(loss):
            raise FloatingPointError(step)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        if step % 32 == 0:
            print(f"{arm} {step}/{len(train)} elapsed {time.perf_counter()-started:.0f}s", flush=True)
    out.mkdir()
    torch.save({name: value.detach().cpu() for name, value in model.named_parameters()
                if value.requires_grad}, out / "adapter.pt")
    (out / "TRAINING.json").write_text(json.dumps({"protocol_sha256": sha(ROOT / "PROTOCOL.json"),
                                                    "elapsed_seconds": time.perf_counter()-started,
                                                    "losses": losses}, indent=2)+"\n")
    print(f"{arm} training complete", flush=True)


@torch.inference_mode()
def evaluate(arm: str) -> None:
    _, events = validate()
    out = ROOT / arm
    target = out / "EVALUATION.json"
    if target.exists():
        raise FileExistsError(target)
    torch.set_num_threads(2)
    tok, model = first.load_source()
    weights = torch.load(out / "adapter.pt", map_location="cpu", weights_only=True)
    params = {name: value for name, value in model.named_parameters() if value.requires_grad}
    assert weights.keys() == params.keys()
    for name, param in params.items():
        param.data.copy_(weights[name])
    model.eval()
    records = []
    for i, event in enumerate(events, 1):
        margins = {}
        for condition in ("with", "omitted"):
            x = event if condition == "with" else {**event, "sentence": "[omitted]"}
            s = {role: {candidate: score(tok, model, prompt(tok, x, role),
                                         event[f"{candidate}_span"])[0]
                        for candidate in ("actor", "other")}
                 for role in ("actor", "other")}
            margins[condition] = (s["actor"]["actor"] - s["actor"]["other"]
                                  - s["other"]["actor"] + s["other"]["other"])
        records.append({"id": event["id"], **margins,
                        "correct": margins["with"] > 0,
                        "omitted_correct": margins["omitted"] > 0})
        if i % 25 == 0:
            print(f"{arm} fresh {i}/{len(events)}", flush=True)
    _, synthetic = confirm.validate("fresh1")
    confirm.configure("fresh1", 830211)
    selected = [r for r in synthetic if r["form"] in ("patient_cleft", "active", "passive")]
    answers = core.evaluate(arm, tok, model, selected)
    cells = {form: [sum(r["correct"] for r in answers if r["form"] == form),
                    sum(r["form"] == form for r in answers)]
             for form in ("patient_cleft", "active", "passive")}
    obj = {"arm": arm, "protocol_sha256": sha(ROOT / "PROTOCOL.json"),
           "records": records, "fresh_correct": sum(r["correct"] for r in records),
           "omitted_correct": sum(r["omitted_correct"] for r in records),
           "synthetic": cells}
    target.write_text(json.dumps(obj, indent=2)+"\n")
    print(json.dumps({"fresh": [obj["fresh_correct"], len(records)],
                      "omitted": [obj["omitted_correct"], len(records)],
                      "synthetic": cells}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "validate", "run", "evaluate"))
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    if args.action == "prepare": prepare()
    elif args.action == "validate": validate(); print("validated")
    else:
        if not args.arm:
            parser.error("--arm required")
        if args.action == "run": run(args.arm)
        else: evaluate(args.arm)
