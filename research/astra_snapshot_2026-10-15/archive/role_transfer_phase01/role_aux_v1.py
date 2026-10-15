"""Developmental ordered-role auxiliary objective with unordered-identity control."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import generator_gate_v1 as gate

base = gate.base
prior = gate.prior
ROOT = Path(__file__).resolve().parent / "role_aux_v1"
ARMS = ("ordered_roles", "unordered_identities")
WEIGHT = 0.5


class RoleListener(prior.Listener10):
    def __init__(self) -> None:
        super().__init__()
        self.first_head = nn.Linear(base.DIM, 10)
        self.second_head = nn.Linear(base.DIM, 10)


def objective(model: RoleListener, rows: list[tuple], arm: str,
              device: torch.device) -> tuple[torch.Tensor, dict]:
    assert arm in ARMS
    meanings = [row[0] for row in rows]
    messages = gate.oracle(meanings, device)
    candidates = torch.from_numpy(np.stack([row[1] for row in rows])).to(device)
    answers = torch.tensor([row[2] for row in rows], dtype=torch.long, device=device)
    query = model.query(messages)
    candidate = model.candidate(candidates.reshape(-1, 3, 18)).reshape(-1, 4, base.DIM)
    candidate = F.normalize(candidate, dim=-1)
    scores = torch.einsum("bd,bkd->bk", query, candidate) * 10
    native_loss = F.cross_entropy(scores, answers)
    selected = candidate[torch.arange(len(rows), device=device), answers]
    src = torch.tensor([m.src for m in meanings], dtype=torch.long, device=device)
    tgt = torch.tensor([m.tgt for m in meanings], dtype=torch.long, device=device)
    if arm == "ordered_roles":
        first, second = src, tgt
    else:
        first, second = torch.minimum(src, tgt), torch.maximum(src, tgt)
    first_q, first_c = model.first_head(query), model.first_head(selected)
    second_q, second_c = model.second_head(query), model.second_head(selected)
    aux_loss = sum((F.cross_entropy(first_q, first),
                    F.cross_entropy(first_c, first),
                    F.cross_entropy(second_q, second),
                    F.cross_entropy(second_c, second))) / 4
    total = native_loss + WEIGHT * aux_loss
    diagnostics = {"native": float(native_loss.detach()),
                   "auxiliary": float(aux_loss.detach()),
                   "query_exact": float(((first_q.argmax(-1) == first)
                                         & (second_q.argmax(-1) == second)).float().mean().detach()),
                   "scene_exact": float(((first_c.argmax(-1) == first)
                                         & (second_c.argmax(-1) == second)).float().mean().detach())}
    return total, diagnostics


def audit() -> dict:
    generator = gate.audit()
    mapping = gate.configure("new0")
    base.set_seed(gate.SEEDS[0])
    baseline = prior.Listener10()
    base.set_seed(gate.SEEDS[0])
    auxiliary = RoleListener()
    assert all(torch.equal(a, b) for a, b in zip(baseline.parameters(),
                                                  list(auxiliary.parameters())[:-4]))
    pool = base.make_pool("canonical")
    order = np.random.default_rng(np.random.SeedSequence([gate.SEEDS[0], 700, 0])).permutation(len(pool))
    schedule = gate.balanced_schedule(pool, order, gate.SEEDS[0], 0)
    rows = []
    for row_index, pool_index in enumerate(order[:32]):
        distractors = tuple(schedule[(row_index, member)] for member in range(4))
        row = gate.episode(pool[int(pool_index)], gate.SEEDS[0], row_index, distractors)
        rows.append(prior.mapped_row(row, mapping))
    outputs = {}
    for arm in ARMS:
        auxiliary.zero_grad(set_to_none=True)
        loss, diagnostics = objective(auxiliary, rows, arm, torch.device("cpu"))
        loss.backward()
        query_gradient = float(auxiliary.sym.grad.norm())
        scene_gradient = float(auxiliary.candidate.inp.weight.grad.norm())
        assert query_gradient > 0 and scene_gradient > 0
        outputs[arm] = {"initial_loss": float(loss.detach()),
                        "query_gradient_norm": query_gradient,
                        "scene_gradient_norm": scene_gradient, **diagnostics}
    return {"generator_worlds": list(generator["worlds"]), "initial_base_weights_matched": True,
            "auxiliary_weight": WEIGHT, "arms": outputs}


def train(world: str, arm: str, seed: int) -> dict:
    assert world in gate.WORLDS and arm in ARMS and seed in gate.SEEDS
    destination = ROOT / "full" / world / arm / f"seed_{seed}"
    if destination.exists():
        raise FileExistsError(destination)
    mapping = gate.configure(world)
    base.set_seed(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device("cpu")
    model = RoleListener().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=base.LR)
    pool = base.make_pool("canonical")
    diagnostics = []
    for epoch in range(gate.STEPS * base.BATCH // len(pool)):
        order = np.random.default_rng(np.random.SeedSequence([seed, 700, epoch])).permutation(len(pool))
        schedule = gate.balanced_schedule(pool, order, seed, epoch)
        for offset in range(0, len(pool), base.BATCH):
            batch = []
            for j, pool_index in enumerate(order[offset:offset + base.BATCH]):
                row_index = offset + j
                serial = epoch * len(pool) + row_index
                distractors = tuple(schedule[(row_index, member)] for member in range(4))
                row = gate.episode(pool[int(pool_index)], seed, serial, distractors)
                batch.append(prior.mapped_row(row, mapping))
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss, detail = objective(model, batch, arm, device)
            loss.backward()
            optimizer.step()
            diagnostics.append(detail)
            if len(diagnostics) % 500 == 0:
                recent = diagnostics[-100:]
                print(f"{world} {arm} seed={seed} step={len(diagnostics)} "
                      f"native={np.mean([r['native'] for r in recent]):.4f} "
                      f"aux={np.mean([r['auxiliary'] for r in recent]):.4f} "
                      f"q={np.mean([r['query_exact'] for r in recent]):.3f} "
                      f"c={np.mean([r['scene_exact'] for r in recent]):.3f}", flush=True)
    from all_pair_audit import rows_for_pair, score_pair  # noqa: E402
    import all_pair_audit as evaluator  # noqa: E402
    labeled_edges = {tuple(sorted((mapping[a], mapping[b])))
                     for a, b in gate.WORLDS[world]["edges"]}
    suites = {}
    for scene_seed in (260107, 260108):
        evaluator.TEST_SEED = scene_seed
        pairs = []
        for pair_index, (a, b) in enumerate(itertools.combinations(range(10), 2)):
            row = rows_for_pair(a, b, pair_index)
            row[0][:, 3] = gate.PAD_ID
            pairs.append({"pair": [a, b], "observed": (a, b) in labeled_edges,
                          **score_pair(model, row)})
        suites[str(scene_seed)] = pairs
    summaries = {
        str(scene_seed): {
            group: {metric: float(np.mean([row[metric] for row in pairs
                                           if row["observed"] == (group == "seen")]))
                    for metric in ("four_choice_accuracy", "binary_role_accuracy", "event_accuracy")}
            for group in ("seen", "unseen")}
        for scene_seed, pairs in suites.items()}
    result = {"world": world, "arm": arm, "seed": seed, "steps": gate.STEPS,
              "auxiliary_weight": WEIGHT, "mapping": mapping,
              "last_100_training": {key: float(np.mean([r[key] for r in diagnostics[-100:]]))
                                    for key in diagnostics[-1]},
              "summary": summaries, "evaluation": suites}
    destination.mkdir(parents=True)
    torch.save(model.state_dict(), destination / "model.pt")
    result["model_sha256"] = hashlib.sha256((destination / "model.pt").read_bytes()).hexdigest()
    (destination / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"completed": str(destination), "summary": summaries,
                      "last_100_training": result["last_100_training"]}), flush=True)
    return result


def summarize() -> dict:
    records = []
    for world, seed in itertools.product(gate.WORLDS, gate.SEEDS):
        baseline_path = (gate.OUTPUT / "full" / world / "balanced_distractor"
                         / f"seed_{seed}" / "metrics.json")
        paths = {arm: ROOT / "full" / world / arm / f"seed_{seed}" / "metrics.json"
                 for arm in ARMS}
        if not baseline_path.exists() or not all(path.exists() for path in paths.values()):
            continue
        baseline = json.loads(baseline_path.read_text())
        arms = {arm: json.loads(path.read_text()) for arm, path in paths.items()}
        scene = "260107"
        metrics = {"baseline": baseline, **arms}
        record = {"world": world, "seed": seed}
        for arm, result in metrics.items():
            record[arm] = {
                "seen_four_choice": result["summary"][scene]["seen"]["four_choice_accuracy"],
                "unseen_four_choice": result["summary"][scene]["unseen"]["four_choice_accuracy"],
                "unseen_binary_role": result["summary"][scene]["unseen"]["binary_role_accuracy"],
                "unseen_event": result["summary"][scene]["unseen"]["event_accuracy"],
                "weak_unseen_pairs": sum(not row["observed"] and row["binary_role_accuracy"] < .8
                                         for row in result["evaluation"][scene]),
            }
        record["ordered_minus_baseline"] = (record["ordered_roles"]["unseen_binary_role"]
                                             - record["baseline"]["unseen_binary_role"])
        record["ordered_minus_unordered"] = (record["ordered_roles"]["unseen_binary_role"]
                                              - record["unordered_identities"]["unseen_binary_role"])
        records.append(record)
    result = {"complete_triples": len(records), "planned_triples": 6, "records": records}
    if len(records) == 6:
        result["world_mean_contrasts"] = {
            world: {contrast: float(np.mean([row[contrast] for row in records
                                             if row["world"] == world]))
                    for contrast in ("ordered_minus_baseline", "ordered_minus_unordered")}
            for world in gate.WORLDS}
        result["development_advancement_gate"] = (
            all(value["ordered_minus_baseline"] > 0
                and value["ordered_minus_unordered"] > 0
                for value in result["world_mean_contrasts"].values())
            and all(row["ordered_roles"]["seen_four_choice"] >= .99 for row in records)
            and all(row["ordered_roles"]["unseen_event"]
                    >= row["baseline"]["unseen_event"] - .02 for row in records))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("audit")
    sub.add_parser("summarize")
    run = sub.add_parser("train")
    run.add_argument("--world", choices=gate.WORLDS, required=True)
    run.add_argument("--arm", choices=ARMS, required=True)
    run.add_argument("--seed", type=int, choices=gate.SEEDS, required=True)
    args = parser.parse_args()
    if args.command == "audit":
        print(json.dumps(audit(), indent=2))
    elif args.command == "summarize":
        print(json.dumps(summarize(), indent=2))
    else:
        train(args.world, args.arm, args.seed)


if __name__ == "__main__":
    main()
