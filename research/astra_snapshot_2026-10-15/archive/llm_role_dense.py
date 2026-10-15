"""Dense 20-40 ordinary-training checkpoints for the early-stopping control."""
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
import llm_role_trajectory as trajectory


HERE = Path(__file__).resolve().parent
ROOT = HERE / "llm_role_dense"
TRAJECTORY = HERE / "llm_role_trajectory"
STEPS = (25, 30, 35)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protocol() -> dict:
    return {
        "status": "fixed_before_dense_training", "date": "2026-10-11",
        "question": "Does an ordinary checkpoint between 20 and 40 updates match the auxiliary arms' patient-cleft accuracy at their familiar-form accuracy?",
        "source_sha256": sha(HERE / "llm_role_dense.py"),
        "core_source_sha256": sha(HERE / "llm_role_repair_v3.py"),
        "confirmation_source_sha256": sha(HERE / "llm_role_repair_confirm.py"),
        "trajectory_source_sha256": sha(HERE / "llm_role_trajectory.py"),
        "trajectory_protocol_sha256": sha(TRAJECTORY / "PROTOCOL.json"),
        "development_worlds": ("fresh1", "fresh2"), "reserved_world": "fresh3",
        "seeds": confirm.SEEDS, "new_checkpoints": STEPS,
        "training": "Same ordinary loss, examples, optimizer, LR, update order, attention masks, LoRA initialization as the frozen trajectory. Evaluate at 25,30,35; train through 40 and require an exact match to its previously saved step-40 adapter.",
        "screen": "Same performed-wording 750 rows per graph/start as the frozen trajectory: 250 patient-cleft and 500 familiar active/passive. Retain every prediction.",
        "selection": "Combine the prior checkpoints 20,40,60,80,100,120 with new 25,30,35. On fresh1/fresh2 and both starts, separately for ordered and unordered auxiliary arms, require ordinary mean familiar accuracy at least that arm's mean familiar accuracy; among eligible steps choose highest mean patient accuracy, earliest step on ties. Keep fresh3 out of the choice. If a new 25/30/35 checkpoint is selected, run it on fresh3 and evaluate the two unseen question wordings with the same three forms. If an old checkpoint remains selected, retain its existing fresh3 confirmation.",
        "interpretation": "This follow-up was chosen after seeing the coarse 20-40 drop. It narrows, but cannot eliminate, unsampled-step or changed-learning-rate alternatives. Do not treat a participant-only effect as a semantic-role mechanism.",
    }


def prepare():
    ROOT.mkdir(exist_ok=True)
    path = ROOT / "PROTOCOL.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(protocol(), indent=2) + "\n")
    print("Frozen dense 25/30/35 checkpoint protocol", flush=True)


def run(world: str, seed: int):
    p = json.loads((ROOT / "PROTOCOL.json").read_text())
    for key, path in (("source_sha256", HERE / "llm_role_dense.py"),
                      ("core_source_sha256", HERE / "llm_role_repair_v3.py"),
                      ("confirmation_source_sha256", HERE / "llm_role_repair_confirm.py"),
                      ("trajectory_source_sha256", HERE / "llm_role_trajectory.py"),
                      ("trajectory_protocol_sha256", TRAJECTORY / "PROTOCOL.json")):
        assert p[key] == sha(path)
    assert world in confirm.WORLD_NAMES and seed in confirm.SEEDS
    output = ROOT / f"{world}_{seed}_ordinary"
    if output.exists():
        raise FileExistsError(output)
    groups, rows = confirm.validate(world)
    screen = trajectory.select_rows(rows, ("performed",))
    confirm.configure(world, seed)
    tokenizer, model, heads = core.load("ordinary")
    assert heads is None
    optimizer = torch.optim.AdamW([param for param in model.parameters() if param.requires_grad], lr=core.LR)
    previous = json.loads((TRAJECTORY / f"{world}_{seed}_ordinary" / "TRAINING.json").read_text())
    assert core.adapter_hash(model) == previous["initial_adapter_sha256"]
    rng = np.random.default_rng(core.SEED)
    order = [int(i) for _ in range(core.EPOCHS) for i in rng.permutation(len(groups))]
    output.mkdir()
    model.train()
    started = time.perf_counter()
    for step, group_id in enumerate(order[:40], 1):
        tokens, labels, masks, _ = core.encode_group(tokenizer, groups[group_id])
        optimizer.zero_grad(set_to_none=True)
        result = model(input_ids=tokens, attention_mask=masks, labels=labels,
                       output_hidden_states=False, use_cache=False)
        if not torch.isfinite(result.loss):
            raise FloatingPointError(step)
        result.loss.backward()
        optimizer.step()
        if step in STEPS:
            print(f"{world}/{seed}: checkpoint {step}, elapsed {time.perf_counter()-started:.0f}s", flush=True)
            checkpoint = output / f"step_{step:03d}"
            checkpoint.mkdir()
            torch.save({"adapter": trajectory.state(model)}, checkpoint / "adapter.pt")
            records = core.evaluate(f"{world}_{seed}_dense_{step}", tokenizer, model, screen)
            (checkpoint / "result.json").write_text(json.dumps({
                "world": world, "seed": seed, "step": step,
                "protocol_sha256": sha(ROOT / "PROTOCOL.json"),
                "scores": trajectory.score(records), "records": records}, indent=2) + "\n")
            model.train()
    target = torch.load(TRAJECTORY / f"{world}_{seed}_ordinary" / "step_040" / "adapter.pt",
                        map_location="cpu", weights_only=True)["adapter"]
    current = trajectory.state(model)
    assert target.keys() == current.keys()
    assert all(torch.equal(target[key], current[key]) for key in target)
    (output / "TRAINING.json").write_text(json.dumps({"updates": 40,
                                                       "step40_exact_match": True,
                                                       "protocol_sha256": sha(ROOT / "PROTOCOL.json")}, indent=2) + "\n")
    print(f"{world}/{seed}: dense checkpoints complete; step 40 exact match", flush=True)


def batch(seed: int):
    for world in ("fresh1", "fresh2"):
        target = ROOT / f"{world}_{seed}_ordinary" / "TRAINING.json"
        if target.exists():
            print(f"Already complete: {target}", flush=True)
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
            parser.error("valid --seed required")
        batch(args.seed)
    else:
        if args.world is None or args.seed not in confirm.SEEDS:
            parser.error("--world and valid --seed required")
        run(args.world, args.seed)
