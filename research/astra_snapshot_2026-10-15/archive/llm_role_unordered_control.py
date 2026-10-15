"""Matched control: classify the two participants without their semantic order."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import llm_role_repair_confirm as confirm
import llm_role_repair_v3 as core


HERE = Path(__file__).resolve().parent
ROOT = HERE / "llm_role_unordered_control"
REFERENCE = HERE / "llm_role_repair_confirmation"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protocol() -> dict:
    return {
        "status": "fixed_before_unordered_training",
        "date": "2026-10-11",
        "question": "Is the ordered-role cleft benefit specific to assigning actor and patient, or can an otherwise matched participant-identity auxiliary produce it?",
        "model": "Qwen2.5-0.5B-Instruct, same local snapshot and LoRA setup as the fresh-graph confirmation",
        "source_sha256": sha(HERE / "llm_role_unordered_control.py"),
        "core_source_sha256": sha(HERE / "llm_role_repair_v3.py"),
        "confirmation_source_sha256": sha(HERE / "llm_role_repair_confirm.py"),
        "reference_protocol_sha256": sha(REFERENCE / "PROTOCOL.json"),
        "worlds": list(confirm.WORLD_NAMES), "seeds": list(confirm.SEEDS),
        "design": "Same six graph/start units, training groups, labels, initialization, update order, model architecture, optimizer and 0.5 auxiliary weight as ordered-role confirmation. Only auxiliary targets change: heads predict the two participants in canonical index order (smaller ID, larger ID), independent of actor/patient order. The standard language-model answer loss is unchanged.",
        "primary": "Exact actor accuracy on the same 750 held-out patient-cleft prompts per unit. Compare unordered against both frozen ordinary and frozen ordered results. If unordered gives a similar gain to ordered, participant identity supervision is sufficient; if ordered consistently exceeds unordered, ordered roles add something beyond the extra heads and identity labels.",
        "secondary": "Familiar active/passive retention, actor-cleft control, seen/unseen verbs, question wording, all-three-wording correctness, invalid first tokens. Report every unit without exclusions.",
        "limits": "Follow-up was designed after inspecting ordered-versus-ordinary results. It is a mechanism-discrimination control, not a preregistered component of the original six-run directional replication. Three graph layouts and one small model family.",
    }


def prepare() -> None:
    ROOT.mkdir(exist_ok=True)
    path = ROOT / "PROTOCOL.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(protocol(), indent=2) + "\n")
    print("Frozen unordered participant-control protocol", flush=True)


def train(tokenizer, model, heads, groups: list[list[dict]]) -> dict:
    parameters = [p for p in model.parameters() if p.requires_grad] + list(heads.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=core.LR)
    initial_adapter_hash = core.adapter_hash(model)
    h = hashlib.sha256()
    for name, param in heads.named_parameters():
        h.update(name.encode())
        h.update(param.detach().numpy().tobytes())
    initial_head_hash = h.hexdigest()
    rng = np.random.default_rng(core.SEED)
    order = [int(i) for _ in range(core.EPOCHS) for i in rng.permutation(len(groups))]
    diagnostics = []
    model.train()
    heads.train()
    started = time.perf_counter()
    for step, group_id in enumerate(order, 1):
        group = groups[group_id]
        tokens, labels, masks, prompt_idx = core.encode_group(tokenizer, group)
        optimizer.zero_grad(set_to_none=True)
        result = model(input_ids=tokens, attention_mask=masks, labels=labels,
                       output_hidden_states=True, use_cache=False)
        states = result.hidden_states[-1][:, prompt_idx, :]
        lower_ids = torch.tensor([row["pair"][0] for row in group])
        upper_ids = torch.tensor([row["pair"][1] for row in group])
        auxiliary = (F.cross_entropy(heads[0](states), lower_ids) +
                     F.cross_entropy(heads[1](states), upper_ids)) / 2
        loss = result.loss + core.ORDERED_WEIGHT * auxiliary
        if not torch.isfinite(loss):
            raise FloatingPointError(step)
        loss.backward()
        optimizer.step()
        diagnostics.append({"step": step, "group": group_id,
                            "language": float(result.loss.detach()),
                            "unordered": float(auxiliary.detach()),
                            "total": float(loss.detach())})
        if step % 20 == 0:
            print(f"unordered_pair: {step}/{len(order)} updates, elapsed {time.perf_counter()-started:.0f}s, recent loss {np.mean([d['total'] for d in diagnostics[-20:]]):.3f}", flush=True)
    return {"updates": len(order), "train_examples": len(groups) * 4,
            "initial_adapter_sha256": initial_adapter_hash,
            "initial_head_sha256": initial_head_hash,
            "elapsed_seconds": time.perf_counter() - started,
            "diagnostics": diagnostics}


def run(world: str, seed: int) -> None:
    p = json.loads((ROOT / "PROTOCOL.json").read_text())
    assert p["source_sha256"] == sha(HERE / "llm_role_unordered_control.py")
    assert p["core_source_sha256"] == sha(HERE / "llm_role_repair_v3.py")
    assert p["confirmation_source_sha256"] == sha(HERE / "llm_role_repair_confirm.py")
    assert p["reference_protocol_sha256"] == sha(REFERENCE / "PROTOCOL.json")
    assert world in confirm.WORLD_NAMES and seed in confirm.SEEDS
    output = ROOT / f"{world}_{seed}_unordered_pair"
    if output.exists():
        raise FileExistsError(output)
    groups, rows = confirm.validate(world)
    confirm.configure(world, seed)
    tokenizer, model, heads = core.load("ordered_roles")
    training = train(tokenizer, model, heads, groups)
    ordered = json.loads((REFERENCE / f"{world}_{seed}_ordered_roles" / "result.json").read_text())
    assert training["initial_adapter_sha256"] == ordered["training"]["initial_adapter_sha256"]
    assert training["initial_head_sha256"] == ordered["training"]["initial_head_sha256"]
    records = core.evaluate(f"{world}_{seed}_unordered_pair", tokenizer, model, rows)
    output.mkdir()
    torch.save({"adapter": {name: param.detach().cpu() for name, param in model.named_parameters()
                            if param.requires_grad}, "heads": heads.state_dict()}, output / "adapter.pt")
    result = {"world": world, "seed": seed, "arm": "unordered_pair",
              "protocol_sha256": sha(ROOT / "PROTOCOL.json"),
              "reference_protocol_sha256": sha(REFERENCE / "PROTOCOL.json"),
              "training": training, "summary": core.summarize(records), "records": records}
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    patient = [row for row in records if row["form"] == "patient_cleft"]
    familiar = [row for row in records if row["form"] in ("active", "passive")]
    print(json.dumps({"world": world, "seed": seed,
                      "patient_cleft": [sum(row["correct"] for row in patient), len(patient)],
                      "familiar_forms": [sum(row["correct"] for row in familiar), len(familiar)]}), flush=True)


def batch(seeds: tuple[int, ...]) -> None:
    for world in confirm.WORLD_NAMES:
        for seed in seeds:
            output = ROOT / f"{world}_{seed}_unordered_pair" / "result.json"
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
        batch((args.seed,) if args.seed is not None else confirm.SEEDS)
    else:
        if args.world is None or args.seed is None:
            parser.error("--world and --seed required")
        run(args.world, args.seed)
