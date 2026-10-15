"""Ordinary fine-tuning trajectory: test whether early stopping explains auxiliary gains."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

import llm_role_repair_confirm as confirm
import llm_role_repair_v3 as core


HERE = Path(__file__).resolve().parent
ROOT = HERE / "llm_role_trajectory"
REFERENCE = HERE / "llm_role_repair_confirmation"
UNORDERED = HERE / "llm_role_unordered_control"
STEPS = (20, 40, 60, 80, 100, 120)
FORMS = ("active", "passive", "patient_cleft")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_rows(rows: list[dict], wordings: tuple[str, ...]) -> list[dict]:
    chosen = [row for row in rows if row["form"] in FORMS and row["wording"] in wordings]
    assert len(chosen) == 750 * len(wordings)
    return chosen


def protocol() -> dict:
    return {
        "status": "fixed_before_trajectory_training", "date": "2026-10-11",
        "question": "Can a shorter ordinary fine-tuning run reach the same patient-cleft and familiar-form accuracy trade-off as the ordered-role or participant-only auxiliary arms?",
        "source_sha256": sha(HERE / "llm_role_trajectory.py"),
        "core_source_sha256": sha(HERE / "llm_role_repair_v3.py"),
        "confirmation_source_sha256": sha(HERE / "llm_role_repair_confirm.py"),
        "reference_protocol_sha256": sha(REFERENCE / "PROTOCOL.json"),
        "unordered_protocol_sha256": sha(UNORDERED / "PROTOCOL.json"),
        "worlds": list(confirm.WORLD_NAMES), "seeds": list(confirm.SEEDS),
        "checkpoints": STEPS,
        "training": "Exact ordinary-loss setup, groups, update order, optimizer, LR 1e-4, corrected masks and LoRA initializations as the six-run confirmation. Run 120 updates and evaluate saved adapters at steps 20,40,60,80,100,120; no changes to examples or hyperparameters. Verify step-120 adapter exactly matches the prior ordinary result.",
        "screen": "At each checkpoint score all 25 withheld pairs x two directions x five verbs x active/passive/patient-cleft x the 'performed' wording = 750 prompts per graph/start (250 patient cleft, 500 familiar). Retain every raw prediction. No checkpoint selected using the other two wordings.",
        "selection": "On development graphs fresh1 and fresh2 (both starts), select one global ordinary checkpoint separately for ordered-role and unordered-participant comparisons: among checkpoints whose mean familiar accuracy on the screen is at least the corresponding auxiliary arm's mean familiar screen accuracy, take the highest mean patient-cleft accuracy; break ties toward the earlier step. If no checkpoint is eligible, report that explicitly. The criterion is a familiar-accuracy floor, not a full Pareto test.",
        "confirmation": "Evaluate the selected ordinary checkpoint(s), if any, on fresh3's two starts using the two nontraining question wordings ('did','carried_out') and all three primary forms (1500 prompts per start). Compare with previously frozen auxiliary arms on exactly those rows. Also report the full screen trajectory for all six units and the untouched baseline. Fresh3 was not used to choose the step, but its final-arm results were inspected before this follow-up was designed, so this is a held-out checkpoint choice rather than a wholly untouched study design.",
        "interpretation": "If a selected ordinary checkpoint matches or exceeds the auxiliary in both patient-cleft and familiar accuracy on both fresh3 starts, early stopping is sufficient for this tested trade-off. If not, the auxiliary may extend the tested trade-off, subject to other LR/regularization controls and one-model synthetic limits. Do not infer internal semantic-role mechanism from either result.",
    }


def prepare() -> None:
    ROOT.mkdir(exist_ok=True)
    path = ROOT / "PROTOCOL.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(protocol(), indent=2) + "\n")
    print("Frozen trajectory protocol: six units, six checkpoints, 750 screen prompts each", flush=True)


def state(model) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()
            if parameter.requires_grad}


def score(records: list[dict]) -> dict:
    out = {}
    for form in FORMS:
        selected = [row for row in records if row["form"] == form]
        out[form] = {"correct": sum(row["correct"] for row in selected),
                     "total": len(selected), "valid": sum(row["valid"] for row in selected)}
    return out


def run(world: str, seed: int) -> None:
    p = json.loads((ROOT / "PROTOCOL.json").read_text())
    assert p["source_sha256"] == sha(HERE / "llm_role_trajectory.py")
    assert p["core_source_sha256"] == sha(HERE / "llm_role_repair_v3.py")
    assert p["confirmation_source_sha256"] == sha(HERE / "llm_role_repair_confirm.py")
    assert p["reference_protocol_sha256"] == sha(REFERENCE / "PROTOCOL.json")
    assert p["unordered_protocol_sha256"] == sha(UNORDERED / "PROTOCOL.json")
    assert world in confirm.WORLD_NAMES and seed in confirm.SEEDS
    output = ROOT / f"{world}_{seed}_ordinary"
    if output.exists():
        raise FileExistsError(output)
    groups, all_rows = confirm.validate(world)
    rows = select_rows(all_rows, ("performed",))
    confirm.configure(world, seed)
    tokenizer, model, heads = core.load("ordinary")
    assert heads is None
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=core.LR)
    initial_adapter_hash = core.adapter_hash(model)
    reference = json.loads((REFERENCE / f"{world}_{seed}_ordinary" / "result.json").read_text())
    assert initial_adapter_hash == reference["training"]["initial_adapter_sha256"]
    rng = np.random.default_rng(core.SEED)
    order = [int(i) for _ in range(core.EPOCHS) for i in rng.permutation(len(groups))]
    assert len(order) == 120
    output.mkdir()
    diagnostics = []
    model.train()
    started = time.perf_counter()
    for step, group_id in enumerate(order, 1):
        tokens, labels, masks, _ = core.encode_group(tokenizer, groups[group_id])
        optimizer.zero_grad(set_to_none=True)
        result = model(input_ids=tokens, attention_mask=masks, labels=labels,
                       output_hidden_states=False, use_cache=False)
        loss = result.loss
        if not torch.isfinite(loss):
            raise FloatingPointError(step)
        loss.backward()
        optimizer.step()
        diagnostics.append({"step": step, "group": group_id, "language": float(loss.detach())})
        if step in STEPS:
            print(f"{world}/{seed}: trained {step}/120 updates in {time.perf_counter()-started:.0f}s", flush=True)
            checkpoint = output / f"step_{step:03d}"
            checkpoint.mkdir()
            torch.save({"adapter": state(model)}, checkpoint / "adapter.pt")
            records = core.evaluate(f"{world}_{seed}_ordinary_step_{step}", tokenizer, model, rows)
            (checkpoint / "result.json").write_text(json.dumps({
                "world": world, "seed": seed, "arm": "ordinary", "step": step,
                "protocol_sha256": sha(ROOT / "PROTOCOL.json"),
                "initial_adapter_sha256": initial_adapter_hash,
                "scores": score(records), "records": records}, indent=2) + "\n")
            model.train()
    previous = torch.load(REFERENCE / f"{world}_{seed}_ordinary" / "adapter.pt", map_location="cpu", weights_only=True)["adapter"]
    current = state(model)
    assert previous.keys() == current.keys()
    assert all(torch.equal(previous[key], current[key]) for key in previous), "Final adapter differs from previous ordinary run"
    assert all(a["group"] == b["group"] for a, b in zip(diagnostics, reference["training"]["diagnostics"], strict=True))
    (output / "TRAINING.json").write_text(json.dumps({"updates": 120,
                                                       "initial_adapter_sha256": initial_adapter_hash,
                                                       "step120_exact_reference_match": True,
                                                       "diagnostics": diagnostics}, indent=2) + "\n")
    print(f"{world}/{seed}: all six checkpoints done; step 120 exactly matches frozen ordinary adapter", flush=True)


def batch(seed: int) -> None:
    for world in confirm.WORLD_NAMES:
        output = ROOT / f"{world}_{seed}_ordinary" / "TRAINING.json"
        if output.exists():
            print(f"Already complete: {output}", flush=True)
            continue
        run(world, seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run", "batch"))
    parser.add_argument("--world", choices=confirm.WORLD_NAMES)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "batch":
        if args.seed not in confirm.SEEDS:
            parser.error("--seed must be a confirmation seed")
        batch(args.seed)
    else:
        if args.world is None or args.seed not in confirm.SEEDS:
            parser.error("--world and valid --seed required")
        run(args.world, args.seed)
