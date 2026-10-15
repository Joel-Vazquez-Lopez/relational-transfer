"""Fresh-graph, fresh-name and two-start confirmation of role-aware fine-tuning."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch
from transformers import AutoTokenizer

import llm_role_repair_v3 as core


HERE = Path(__file__).resolve().parent
ROOT = HERE / "llm_role_repair_confirmation"
GRAPH = HERE / "role_transfer_phase01/role_aux_confirmation_v1/GRAPH_MANIFEST.json"
WORLD_NAMES = {
    "fresh1": ("Ryan", "Emma", "Adam", "Emily", "Tyler", "Grace", "Leo", "Rachel", "Frank", "Maria"),
    "fresh2": ("Charlotte", "Richard", "Victoria", "Christopher", "Ruby", "Matthew", "Jordan", "Jessica", "David", "Sarah"),
    "fresh3": ("Michael", "Jennifer", "Thomas", "Michelle", "Joseph", "Karen", "Charles", "Elizabeth", "Paul", "Crystal"),
}
SEEDS = (830211, 830212)
ARMS = ("pretrained", "ordinary", "ordered_roles")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configure(world: str, seed: int) -> None:
    manifest = json.loads(GRAPH.read_text())
    edges = {tuple(e) for e in manifest["worlds"][world]["labeled_edges"]}
    assert len(edges) == 20 and all(a < b for a, b in edges)
    core.NAMES = WORLD_NAMES[world]
    core.SEED = seed
    core.edges = lambda: edges


def fixed_data(world: str):
    configure(world, SEEDS[0])
    training = core.train_groups()
    test = [row for row in core.test_rows() if not row["seen_pair"]]
    assert len(training) == 60 and len(test) == 3000
    return training, test


def protocol() -> dict:
    return {
        "status": "fixed_before_replication_training", "date": "2026-10-11",
        "hypothesis": "Relative to ordinary supervised fine-tuning, ordered actor/patient auxiliary training reduces incorrect actor choices for held-out pairs in patient-fronted clefts, while retaining high accuracy on active/passive forms.",
        "primary": "Ordered minus ordinary exact actor accuracy on patient_cleft prompts for 25 held-out pairs x 2 directions x 5 verbs x 3 fixed actor questions = 750 prompts per world/seed. Compare as six matched graph/start units. Report all units and the equal-unit mean. A directional replication requires positive differences in at least five of six units and an equal-unit mean gain of at least 10 percentage points.",
        "retention": "Report ordered-minus-ordinary accuracy across active/passive prompts on the same held-out pairs (1500 per unit). A mean loss exceeding 2 percentage points is a material trade-off, even if the primary effect replicates.",
        "secondary": "Actor-fronted cleft control; seen versus unseen verbs; all-three-wording correctness for each event; unchanged pretrained model once per name set; invalid-output rate. No post-hoc exclusions.",
        "model": "Qwen2.5-0.5B-Instruct local snapshot",
        "model_weights_sha256": sha(core.MODEL / "model.safetensors"),
        "worlds": {world: {"names": names, "edges": sorted([list(e) for e in json.loads(GRAPH.read_text())["worlds"][world]["labeled_edges"]])}
                   for world, names in WORLD_NAMES.items()},
        "seeds": SEEDS, "trained_arms": ("ordinary", "ordered_roles"),
        "training": "Same 60 four-example groups per world and arm: 20 seen pairs x 3 verbs x both actor directions x active/passive. Two epochs, 120 updates, LR 1e-4. Corrected attention masks for left-padded training. Same model, LoRA rank 4 and alpha 8 on q/v projections. Ordered arm adds two 10-way actor/patient heads at weight 0.5. Each matched pair uses the same start seed, adapter initialization, group order and examples.",
        "test": "Per world, all 25 held-out pairs x 2 directions x 5 verbs x 4 forms x 3 question wordings = 3000 prompts. Two verbs, both cleft forms and two question wordings are unseen in fine-tuning. Greedy next-token exact-name scoring. All raw answers retained.",
        "source_sha256": sha(HERE / "llm_role_repair_confirm.py"),
        "core_source_sha256": sha(HERE / "llm_role_repair_v3.py"),
        "graph_manifest_sha256": sha(GRAPH),
        "limits": "Three graph layouts, two starts each, disjoint ten-name vocabularies across worlds, one small model family. Reuse of three training verbs and construction templates from the exploratory pilot; not a cross-model or natural-corpus result. The six units are not wholly independent because both seeds for a graph share names and edges.",
    }


def prepare() -> None:
    ROOT.mkdir(exist_ok=True)
    path = ROOT / "PROTOCOL.json"
    if path.exists():
        raise FileExistsError(path)
    tokenizer = AutoTokenizer.from_pretrained(core.MODEL, local_files_only=True)
    all_names = [name for names in WORLD_NAMES.values() for name in names]
    assert len(set(all_names)) == 30 and not set(all_names).intersection(core.NAMES)
    for name in all_names:
        assert len(tokenizer.encode(name, add_special_tokens=False)) == 1, name
        assert len(tokenizer.encode(" " + name, add_special_tokens=False)) == 1, name
    path.write_text(json.dumps(protocol(), indent=2) + "\n")
    for world in WORLD_NAMES:
        groups, rows = fixed_data(world)
        (ROOT / f"TRAIN_{world}.json").write_text(json.dumps(groups, indent=2) + "\n")
        (ROOT / f"TEST_{world}.json").write_text(json.dumps(rows, indent=2) + "\n")
    print("Frozen 3 worlds x 2 starts, 60 groups and 3000 prompts per world", flush=True)


def validate(world: str):
    p = json.loads((ROOT / "PROTOCOL.json").read_text())
    assert p["source_sha256"] == sha(HERE / "llm_role_repair_confirm.py")
    assert p["core_source_sha256"] == sha(HERE / "llm_role_repair_v3.py")
    assert p["graph_manifest_sha256"] == sha(GRAPH)
    groups, rows = fixed_data(world)
    assert groups == json.loads((ROOT / f"TRAIN_{world}.json").read_text())
    assert rows == json.loads((ROOT / f"TEST_{world}.json").read_text())
    return groups, rows


def run(world: str, seed: int, arm: str) -> None:
    if arm == "pretrained" and seed != 0:
        raise ValueError("Use seed 0 for the unmodified baseline")
    if arm != "pretrained" and seed not in SEEDS:
        raise ValueError(seed)
    groups, rows = validate(world)
    configure(world, seed)
    output = ROOT / f"{world}_{seed}_{arm}"
    if output.exists():
        raise FileExistsError(output)
    tokenizer, model, heads = core.load(arm)
    training = None if arm == "pretrained" else core.train(arm, tokenizer, model, heads, groups)
    records = core.evaluate(f"{world}_{seed}_{arm}", tokenizer, model, rows)
    output.mkdir()
    if arm != "pretrained":
        torch.save({"adapter": {name: p.detach().cpu() for name, p in model.named_parameters()
                                if p.requires_grad},
                    "heads": heads.state_dict() if heads is not None else None}, output / "adapter.pt")
    summary = core.summarize(records)
    result = {"world": world, "seed": seed, "arm": arm,
              "protocol_sha256": sha(ROOT / "PROTOCOL.json"),
              "training": training, "summary": summary, "records": records}
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    primary = [row for row in records if row["form"] == "patient_cleft"]
    familiar = [row for row in records if row["form"] in ("active", "passive")]
    print(json.dumps({"world": world, "seed": seed, "arm": arm,
                      "patient_cleft": [sum(r["correct"] for r in primary), len(primary)],
                      "familiar_forms": [sum(r["correct"] for r in familiar), len(familiar)]}), flush=True)


def batch(arm: str) -> None:
    jobs = ((world, 0) for world in WORLD_NAMES) if arm == "pretrained" else (
        (world, seed) for world in WORLD_NAMES for seed in SEEDS)
    for world, seed in jobs:
        target = ROOT / f"{world}_{seed}_{arm}" / "result.json"
        if target.exists():
            print(f"Already complete: {target}", flush=True)
            continue
        subprocess.run([sys.executable, str(HERE / "llm_role_repair_confirm.py"), "run",
                        "--world", world, "--seed", str(seed), "--arm", arm], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run", "batch"))
    parser.add_argument("--world", choices=WORLD_NAMES)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "batch":
        if args.arm is None:
            parser.error("--arm is required")
        batch(args.arm)
    else:
        if args.world is None or args.seed is None or args.arm is None:
            parser.error("--world, --seed and --arm are required")
        run(args.world, args.seed, args.arm)
