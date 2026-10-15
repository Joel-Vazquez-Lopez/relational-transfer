"""Second-world matched continuation of the natural-data repair pilot."""
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
ROOT = HERE / "llm_role_natural_repair/replication_fresh3_830212"
WORLD = "fresh3"
SEED = 830212
SOURCE = HERE / f"llm_role_unordered_control/{WORLD}_{SEED}_unordered_pair/adapter.pt"
SYNTHETIC = HERE / f"llm_role_repair_confirmation/TEST_{WORLD}.json"
GOLD = HERE / "llm_role_natural_contrast_confirm/EVENTS.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare() -> None:
    first.validate()
    fresh.validate()
    ROOT.mkdir(exist_ok=True)
    if (ROOT / "PROTOCOL.json").exists():
        raise FileExistsError(ROOT / "PROTOCOL.json")
    p = {"status": "fixed_before_second_world_training", "date": "2026-10-12",
         "question": "Does the first natural-data-plus-replay effect recur from another participant-only synthetic checkpoint with different graph, names and initialization?",
         "source": f"Qwen2.5-0.5B-Instruct, participant-only adapter {WORLD}/{SEED}",
         "training": "Same frozen 256 Bank original training events and order as first pilot, same two role-answer losses, LR 5e-5, 256 updates, same 0.3 synthetic replay weight. Replay uses the separate fresh3 graph/name world. No new cleft forms or Bank test items are trained.",
         "evaluation": "Compare source and new checkpoint on the identical previously frozen 130-event Bank test, sentence-present and sentence-omitted candidate contrast; Gold 51; frozen fresh3 held-out-pair patient-cleft and familiar-form exact actor accuracy. Bank test has already been scored in the first pilot, so this tests starting-run stability rather than independent item selection.",
         "criterion": "Useful repeat if fresh Bank role contrast increases by at least 7/130 events, mean role margin and mean sentence-contribution margin increase, and synthetic patient-cleft accuracy loses no more than five percentage points. Report all scores even if rule fails.",
         "limits": "Second model initialization/graph but same Qwen family, same natural train data and evaluation items, and same data extraction errors. Not a fully independent multi-model confirmation.",
         "references": {"source_adapter_sha256": sha(SOURCE),
                        "first_protocol_sha256": sha(first.ROOT / "PROTOCOL.json"),
                        "train_sha256": sha(first.ROOT / "TRAIN.json"),
                        "fresh_protocol_sha256": sha(fresh.ROOT / "PROTOCOL.json"),
                        "fresh_events_sha256": sha(fresh.ROOT / "EVENTS.json"),
                        "gold_sha256": sha(GOLD),
                        "synthetic_sha256": sha(SYNTHETIC),
                        "script_sha256": sha(HERE / "llm_role_natural_repair_replication.py")}}
    (ROOT / "PROTOCOL.json").write_text(json.dumps(p, indent=2) + "\n")
    print("Frozen second-world replication protocol", flush=True)


def validate():
    first.validate()
    events = fresh.validate()
    p = json.loads((ROOT / "PROTOCOL.json").read_text())
    paths = {"source_adapter_sha256": SOURCE,
             "first_protocol_sha256": first.ROOT / "PROTOCOL.json",
             "train_sha256": first.ROOT / "TRAIN.json",
             "fresh_protocol_sha256": fresh.ROOT / "PROTOCOL.json",
             "fresh_events_sha256": fresh.ROOT / "EVENTS.json",
             "gold_sha256": GOLD,
             "synthetic_sha256": SYNTHETIC,
             "script_sha256": HERE / "llm_role_natural_repair_replication.py"}
    assert all(p["references"][k] == sha(path) for k, path in paths.items())
    return json.loads((first.ROOT / "TRAIN.json").read_text()), events


def load_source():
    confirm.configure(WORLD, SEED)
    tok, model, _ = core.load("ordinary")
    weights = torch.load(SOURCE, map_location="cpu", weights_only=True)["adapter"]
    params = {name: value for name, value in model.named_parameters() if value.requires_grad}
    assert weights.keys() == params.keys()
    for name, param in params.items():
        param.data.copy_(weights[name])
    return tok, model


def train() -> None:
    events, _ = validate()
    target = ROOT / "adapter.pt"
    if target.exists():
        raise FileExistsError(target)
    torch.set_num_threads(2)
    tok, model = load_source()
    groups, _ = confirm.validate(WORLD)
    confirm.configure(WORLD, SEED)
    optimizer = torch.optim.AdamW((v for v in model.parameters() if v.requires_grad), lr=first.LR)
    rng = np.random.default_rng(first.SEED)
    order = [int(i) for i in rng.permutation(len(groups))]
    losses = []
    start = time.perf_counter()
    model.train()
    for step, event in enumerate(events, 1):
        if (step - 1) % len(groups) == 0 and step > 1:
            order = [int(i) for i in rng.permutation(len(groups))]
        group = groups[order[(step - 1) % len(groups)]]
        ids, labels, mask = first.batch(tok, event, group)
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        targets = labels[:, 1:].contiguous()
        loss_tokens = F.cross_entropy(logits[:, :-1].contiguous().flatten(0, 1),
                                      targets.flatten(), ignore_index=-100,
                                      reduction="none").view(6, -1)
        per_item = loss_tokens.sum(dim=1) / (targets != -100).sum(dim=1)
        natural, replay = per_item[:2].mean(), per_item[2:].mean()
        loss = natural + first.REPLAY_WEIGHT * replay
        if not torch.isfinite(loss):
            raise FloatingPointError(step)
        loss.backward()
        optimizer.step()
        losses.append({"step": step, "natural": float(natural.detach()),
                       "replay": float(replay.detach()), "total": float(loss.detach())})
        if step % 32 == 0:
            print(f"replication training {step}/{len(events)} elapsed {time.perf_counter()-start:.0f}s", flush=True)
    torch.save({name: value.detach().cpu() for name, value in model.named_parameters()
                if value.requires_grad}, target)
    (ROOT / "TRAINING.json").write_text(json.dumps({"protocol_sha256": sha(ROOT / "PROTOCOL.json"),
                                                    "elapsed_seconds": time.perf_counter()-start,
                                                    "losses": losses}, indent=2)+"\n")
    print("Replication training complete", flush=True)


@torch.inference_mode()
def evaluate(arm: str) -> None:
    _, events = validate()
    target = ROOT / f"EVAL_{arm}.json"
    if target.exists():
        raise FileExistsError(target)
    torch.set_num_threads(2)
    tok, model = load_source()
    if arm == "new":
        weights = torch.load(ROOT / "adapter.pt", map_location="cpu", weights_only=True)
        params = {name: value for name, value in model.named_parameters() if value.requires_grad}
        assert weights.keys() == params.keys()
        for name, param in params.items():
            param.data.copy_(weights[name])
    model.eval()
    records = []
    for i, row in enumerate(events, 1):
        margins = {}
        for condition in ("with", "omitted"):
            item = row if condition == "with" else {**row, "sentence": "[omitted]"}
            scores = {role: {candidate: score(tok, model, prompt(tok, item, role),
                                              row[f"{candidate}_span"])[0]
                             for candidate in ("actor", "other")}
                      for role in ("actor", "other")}
            margins[condition] = (scores["actor"]["actor"] - scores["actor"]["other"]
                                  - scores["other"]["actor"] + scores["other"]["other"])
        records.append({"id": row["id"], **margins,
                        "correct": margins["with"] > 0,
                        "omitted_correct": margins["omitted"] > 0})
        if i % 25 == 0:
            print(f"{arm}: fresh {i}/{len(events)}", flush=True)
    gold_rows = json.loads(GOLD.read_text())
    gold = []
    for row in gold_rows:
        scores = {role: {candidate: score(tok, model, prompt(tok, row, role),
                                          row[f"{candidate}_span"])[0]
                         for candidate in ("actor", "other")}
                  for role in ("actor", "other")}
        margin = (scores["actor"]["actor"] - scores["actor"]["other"]
                  - scores["other"]["actor"] + scores["other"]["other"])
        gold.append({"id": row["id"], "margin": margin, "correct": margin > 0})
    synthetic = [r for r in json.loads(SYNTHETIC.read_text())
                 if r["form"] in ("patient_cleft", "active", "passive")]
    answers = core.evaluate(f"replication_{arm}", tok, model, synthetic)
    cells = {form: [sum(r["correct"] for r in answers if r["form"] == form),
                    sum(r["form"] == form for r in answers)]
             for form in ("patient_cleft", "active", "passive")}
    obj = {"arm": arm, "protocol_sha256": sha(ROOT / "PROTOCOL.json"),
           "fresh": records, "gold": gold, "synthetic": cells,
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
    parser.add_argument("--arm", choices=("source", "new"))
    args = parser.parse_args()
    if args.action == "prepare": prepare()
    elif args.action == "validate": validate(); print("validated")
    elif args.action == "train": train()
    else:
        if not args.arm:
            parser.error("--arm required")
        evaluate(args.arm)
