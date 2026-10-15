"""Developmental generator gate: padding fixed in both arms, distractors varied.

This never writes to historical directories. See NEXT_GATE.md. The three graphs
and seeds below are developmental, not untouched publication confirmation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
STUDY = ROOT.parent / "d5_indirect_support"
sys.path.insert(0, str(STUDY))
import prospective_20261006 as prospective  # noqa: E402

EARLY_WORLDS = {key: dict(prospective.WORLDS[key]) for key in ("new0", "new2")}
import prospective_20261006_confirm as confirmation  # noqa: E402
import path2_control as prior  # noqa: E402
import run as base  # noqa: E402

WORLDS = {**EARLY_WORLDS, "confirm0": dict(confirmation.study.WORLDS["confirm0"])}
SEEDS = (261001, 261002)
ARMS = ("historical_distractor", "balanced_distractor")
STEPS = 6000
OUTPUT = ROOT / "generator_gate_v1"
PAD_ID = 12
VOCAB = 13


def configure(world: str) -> tuple[int, ...]:
    """Use one prespecified labeling per graph; evaluate every pair, not a focal pair."""
    info = WORLDS[world]
    a, b = info["shared1"]
    rest = [node for node in range(10) if node not in (a, b)]
    mapping = [None] * 10
    mapping[a], mapping[b] = 0, 1
    for label, node in enumerate(rest, 2):
        mapping[node] = label
    base.N_ID = 10
    base.VOCAB = VOCAB
    base.BASE_EDGES = info["edges"]
    base.MAPPINGS = {"canonical": tuple(range(10))}
    return tuple(mapping)


def oracle(meanings: list[base.Meaning], device: torch.device) -> torch.Tensor:
    ids = torch.tensor([[m.src, m.tgt, 10 + m.event, PAD_ID] for m in meanings],
                       dtype=torch.long, device=device)
    return F.one_hot(ids, VOCAB).float()


def balanced_schedule(pool: list[tuple], order: np.ndarray, seed: int,
                      epoch: int) -> dict[tuple[int, int], int]:
    """Balance each semantic candidate's allowed distractors within an epoch.

    Assignment is randomized inside each semantic group and does not use answer
    position or serial modulo. Keys are (position in epoch order, foil member).
    """
    groups: dict[tuple[int, int, int], list[tuple[int, int]]] = defaultdict(list)
    for row_index, pool_index in enumerate(order):
        family, (src, tgt, third), event, _, _ = pool[int(pool_index)]
        for member, meaning in enumerate(base.qset(family, src, tgt, third, event)):
            groups[(meaning.src, meaning.tgt, meaning.event)].append((row_index, member))
    assignment = {}
    for meaning, locations in sorted(groups.items()):
        allowed = [identity for identity in range(10) if identity not in meaning[:2]]
        if len(locations) % len(allowed):
            raise AssertionError(f"Cannot balance group {meaning}: {len(locations)}")
        rng = np.random.default_rng(np.random.SeedSequence(
            [seed, epoch, 801, *meaning]))
        values = np.tile(np.asarray(allowed), len(locations) // len(allowed))
        rng.shuffle(values)
        assignment.update(zip(locations, map(int, values)))
    assert len(assignment) == len(pool) * 4
    return assignment


def episode(spec: tuple, seed: int, serial: int,
            distractors: tuple[int, ...] | None) -> tuple:
    if distractors is None:
        return base.example(spec, seed, serial)
    family, (src, tgt, third), event, selected, answer = spec
    meanings = base.qset(family, src, tgt, third, event)
    rng = np.random.default_rng(np.random.SeedSequence([seed, serial, 11]))
    others = [member for member in range(4) if member != selected]
    rng.shuffle(others)
    members = others[:answer] + [selected] + others[answer:]
    scenes = []
    for position, member in enumerate(members):
        meaning = meanings[member]
        distractor = distractors[member]
        assert distractor not in (meaning.src, meaning.tgt)
        scene_rng = np.random.default_rng(np.random.SeedSequence([seed, serial, 20 + position]))
        scenes.append(base.scene(meaning, distractor, scene_rng))
    return meanings[selected], np.stack(scenes), answer, members.index(selected ^ 1)


def audit() -> dict:
    output = {"version": 1, "worlds": {}, "pad_id": PAD_ID,
              "development_seeds": SEEDS}
    for world in WORLDS:
        mapping = configure(world)
        pool = base.make_pool("canonical")
        assert len(pool) == 1280
        observed = {tuple(sorted((mapping[src], mapping[tgt])))
                    for _, (src, tgt, _), _, _, _ in pool}
        assert len(observed) == 20 and (0, 1) not in observed
        order = np.random.default_rng(np.random.SeedSequence([SEEDS[0], 700, 0])).permutation(len(pool))
        schedule = balanced_schedule(pool, order, SEEDS[0], 0)
        balance = defaultdict(Counter)
        for row_index, pool_index in enumerate(order):
            family, (src, tgt, third), event, selected, answer = pool[int(pool_index)]
            meanings = base.qset(family, src, tgt, third, event)
            assert selected in range(4) and answer in range(4)
            for member, meaning in enumerate(meanings):
                distractor = schedule[(row_index, member)]
                assert distractor not in (meaning.src, meaning.tgt)
                balance[(meaning.src, meaning.tgt, meaning.event)][distractor] += 1
        assert all(len(counts) == 8 and len(set(counts.values())) == 1
                   for counts in balance.values())
        assert len(balance) == 80
        # Same initialization and stochastic scene geometry in the two arms.
        base.set_seed(SEEDS[0])
        left = prior.Listener10()
        base.set_seed(SEEDS[0])
        right = prior.Listener10()
        assert all(torch.equal(a, b) for a, b in zip(left.parameters(), right.parameters()))
        assert left.sym.shape == (VOCAB, base.DIM)
        sample = pool[int(order[0])]
        row_a = episode(sample, SEEDS[0], 0, None)
        row_b = episode(sample, SEEDS[0], 0,
                        tuple(schedule[(0, member)] for member in range(4)))
        assert row_a[0] == row_b[0] and row_a[2:] == row_b[2:]
        assert oracle([row_a[0]], torch.device("cpu")).argmax(-1).tolist()[0][-1] == PAD_ID
        output["worlds"][world] = {"edges": len(observed), "pool": len(pool),
                                   "balanced_semantic_groups": len(balance),
                                   "distractor_counts_per_group": sorted(set(
                                       next(iter(balance.values())).values())),
                                   "mapping": mapping}
    return output


def train(world: str, arm: str, seed: int, steps: int, preflight: bool) -> dict:
    assert world in WORLDS and arm in ARMS and seed in SEEDS
    destination = OUTPUT / ("preflight" if preflight else "full") / world / arm / f"seed_{seed}"
    if destination.exists():
        raise FileExistsError(destination)
    mapping = configure(world)
    base.set_seed(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device("cpu")
    model = prior.Listener10().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=base.LR)
    pool = base.make_pool("canonical")
    if steps * base.BATCH % len(pool):
        raise ValueError("steps * batch must complete whole epochs")
    losses = []
    for epoch in range(steps * base.BATCH // len(pool)):
        order = np.random.default_rng(np.random.SeedSequence([seed, 700, epoch])).permutation(len(pool))
        schedule = (balanced_schedule(pool, order, seed, epoch)
                    if arm == "balanced_distractor" else None)
        for offset in range(0, len(pool), base.BATCH):
            batch = []
            for j, pool_index in enumerate(order[offset:offset + base.BATCH]):
                row_index = offset + j
                serial = epoch * len(pool) + row_index
                distractors = (tuple(schedule[(row_index, member)] for member in range(4))
                               if schedule is not None else None)
                row = episode(pool[int(pool_index)], seed, serial, distractors)
                batch.append(prior.mapped_row(row, mapping))
            messages = oracle([row[0] for row in batch], device)
            candidates = torch.from_numpy(np.stack([row[1] for row in batch])).to(device)
            labels = torch.tensor([row[2] for row in batch], dtype=torch.long, device=device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(messages, candidates), labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            if len(losses) % 500 == 0:
                print(f"{world} {arm} seed={seed} step={len(losses)} "
                      f"loss={np.mean(losses[-100:]):.4f}", flush=True)
    result = {"world": world, "arm": arm, "seed": seed, "steps": steps,
              "preflight": preflight, "last_100_train_loss": float(np.mean(losses[-100:])),
              "mapping": mapping, "pad_id": PAD_ID, "generator_version": 1}
    if not preflight:
        # Fixed all-pair crossing, using two scene seeds for rendering reliability.
        from all_pair_audit import rows_for_pair, score_pair  # noqa: E402
        import all_pair_audit as evaluator  # noqa: E402
        labeled_edges = {tuple(sorted((mapping[a], mapping[b]))) for a, b in WORLDS[world]["edges"]}
        suites = {}
        for scene_seed in (260107, 260108):
            evaluator.TEST_SEED = scene_seed
            pairs = []
            for pair_index, (a, b) in enumerate(itertools.combinations(range(10), 2)):
                row = rows_for_pair(a, b, pair_index)
                row[0][:, 3] = PAD_ID
                pairs.append({"pair": [a, b], "observed": (a, b) in labeled_edges,
                              **score_pair(model, row)})
            suites[str(scene_seed)] = pairs
        result["evaluation"] = suites
        result["summary"] = {
            str(scene_seed): {
                group: {metric: float(np.mean([row[metric] for row in pairs
                                               if row["observed"] == (group == "seen")]))
                        for metric in ("four_choice_accuracy", "binary_role_accuracy", "event_accuracy")}
                for group in ("seen", "unseen")}
            for scene_seed, pairs in suites.items()}
    destination.mkdir(parents=True)
    torch.save(model.state_dict(), destination / "model.pt")
    result["model_sha256"] = hashlib.sha256((destination / "model.pt").read_bytes()).hexdigest()
    (destination / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"completed": str(destination), "summary": result.get("summary"),
                      "last_100_train_loss": result["last_100_train_loss"]}), flush=True)
    return result


def summarize() -> dict:
    records = []
    for world, seed in itertools.product(WORLDS, SEEDS):
        paths = [OUTPUT / "full" / world / arm / f"seed_{seed}" / "metrics.json"
                 for arm in ARMS]
        if not all(path.exists() for path in paths):
            continue
        historical, balanced = [json.loads(path.read_text()) for path in paths]
        source = "260107"
        pair_metrics = {}
        for arm, result in zip(ARMS, (historical, balanced)):
            unseen = [row for row in result["evaluation"][source] if not row["observed"]]
            pair_metrics[arm] = {"weak_unseen_pairs_below_0.8": sum(
                row["binary_role_accuracy"] < 0.8 for row in unseen),
                "unseen_pairs": len(unseen)}
        records.append({"world": world, "seed": seed,
                        "seen_baseline": historical["summary"][source]["seen"],
                        "seen_balanced": balanced["summary"][source]["seen"],
                        "unseen_baseline": historical["summary"][source]["unseen"],
                        "unseen_balanced": balanced["summary"][source]["unseen"],
                        "delta_binary_role": (balanced["summary"][source]["unseen"]["binary_role_accuracy"]
                                              - historical["summary"][source]["unseen"]["binary_role_accuracy"]),
                        "weak_pair_counts": pair_metrics})
    result = {"complete_pairs": len(records), "planned_pairs": len(WORLDS) * len(SEEDS),
              "records": records}
    if len(records) == len(WORLDS) * len(SEEDS):
        result["familiar_gate_0.99"] = all(
            record[f"seen_{arm}"]["four_choice_accuracy"] >= 0.99
            for record in records for arm in ("baseline", "balanced"))
        result["world_mean_delta_binary_role"] = {
            world: float(np.mean([record["delta_binary_role"] for record in records
                                  if record["world"] == world])) for world in WORLDS}
        result["overall_mean_delta_binary_role"] = float(np.mean([
            record["delta_binary_role"] for record in records]))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("audit")
    sub.add_parser("summarize")
    run = sub.add_parser("train")
    run.add_argument("--world", choices=WORLDS, required=True)
    run.add_argument("--arm", choices=ARMS, required=True)
    run.add_argument("--seed", type=int, choices=SEEDS, required=True)
    run.add_argument("--steps", type=int, default=STEPS)
    run.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.command == "audit":
        print(json.dumps(audit(), indent=2))
    elif args.command == "summarize":
        print(json.dumps(summarize(), indent=2))
    else:
        train(args.world, args.arm, args.seed, args.steps, args.preflight)


if __name__ == "__main__":
    main()
