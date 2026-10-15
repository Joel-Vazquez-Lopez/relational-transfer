"""Phase 0 audit and Phase 1A all-pair reanalysis of saved role-transfer models.

Run from any directory with the ai-fingerprint Python environment. Historical
scripts, models, and result directories are only read. Output is written to
role_transfer_phase01. This is development analysis, not fresh confirmation.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
STUDY = PROJECT / "d5_indirect_support"
sys.path.insert(0, str(STUDY))

import prospective_20261006 as study  # noqa: E402
import run as base  # noqa: E402

# The confirmation module intentionally mutates study module globals on import.
SCREEN_WORLDS = dict(study.WORLDS)
SCREEN_SEEDS = tuple(study.SEEDS)
SCREEN_ROOT = study.RESULTS
import prospective_20261006_confirm as confirm  # noqa: E402
CONFIRM_WORLDS = dict(study.WORLDS)
CONFIRM_SEEDS = tuple(study.SEEDS)
CONFIRM_ROOT = study.RESULTS

COHORTS = {
    "screen": (SCREEN_WORLDS, SCREEN_SEEDS, SCREEN_ROOT),
    "confirmation_history": (CONFIRM_WORLDS, CONFIRM_SEEDS, CONFIRM_ROOT),
}
CONDITIONS = ("shared1", "shared2")
TEST_SEED = 260107
REPEATS = 4
OUT = ROOT / "all_pair_results.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def mapped_edges(world_info: dict, condition: str) -> set[tuple[int, int]]:
    a, b = world_info[condition]
    rest = [i for i in range(10) if i not in (a, b)]
    mapping = {a: 0, b: 1, **{old: new for new, old in enumerate(rest, 2)}}
    assert sorted(mapping.values()) == list(range(10))
    return {tuple(sorted((mapping[x], mapping[y]))) for x, y in world_info["edges"]}


def audit() -> dict:
    all_graphs: list[nx.Graph] = []
    records = []
    for cohort, (worlds, seeds, directory) in COHORTS.items():
        for world, info in worlds.items():
            graph = nx.Graph()
            graph.add_nodes_from(range(10))
            graph.add_edges_from(info["edges"])
            assert nx.is_connected(graph) and graph.number_of_edges() == 20
            assert all(degree == 4 for _, degree in graph.degree())
            assert not any(nx.is_isomorphic(graph, other) for other in all_graphs)
            all_graphs.append(graph)
            for condition in CONDITIONS:
                study.WORLDS = worlds
                study.configure(world)
                edges = mapped_edges(info, condition)
                graph_labeled = nx.Graph()
                graph_labeled.add_nodes_from(range(10))
                graph_labeled.add_edges_from(edges)
                pool = base.make_pool("canonical")
                # The pool is generated in canonical coordinates and then mapped
                # during training. Check each coordinate system separately.
                canonical_edges = {tuple(sorted((s, t))) for _, (s, t, _), _, _, _ in pool}
                assert canonical_edges == set(info["edges"])
                meanings = {(s, t, event) for _, (s, t, _), event, _, _ in pool}
                assert len(pool) == 1280 and len(meanings) == 80
                counts = {key: 0 for key in itertools.product(range(2), range(4), range(4))}
                for _, (_, _, _), event, selected, answer in pool:
                    counts[(event, selected, answer)] += 1
                assert set(counts.values()) == {40}  # 20 edges * two directions
                assert (0, 1) not in edges
                checkpoints = []
                for seed in seeds:
                    run_dir = directory / world / condition / f"seed_{seed}"
                    metrics = json.loads((run_dir / "metrics.json").read_text())
                    model_path = run_dir / "model.pt"
                    checksum = sha256(model_path)
                    assert checksum == metrics["model_sha256"]
                    checkpoints.append({"seed": seed, "model_sha256": checksum,
                                        "metrics_path": str(run_dir / "metrics.json"),
                                        "model_path": str(model_path),
                                        "historical_familiar_accuracy": metrics["familiar"]["accuracy"],
                                        "historical_selected_pair_accuracy": metrics["held"]["accuracy"]})
                records.append({"cohort": cohort, "world": world, "condition": condition,
                                "generator_seed": info["generator_seed"],
                                "canonical_edges": [list(e) for e in sorted(info["edges"])],
                                "labeled_edges": [list(e) for e in sorted(edges)],
                                "graph_connected": nx.is_connected(graph_labeled),
                                "observed_unordered_pairs": len(edges),
                                "unseen_unordered_pairs": 45 - len(edges),
                                "training_configurations": len(pool),
                                "distinct_meanings": len(meanings),
                                "counts_per_event_selected_position": 40,
                                "checkpoints": checkpoints})
    return {"status": "passed_with_documented_generator_limitations", "records": records,
            "worlds": len(all_graphs), "checkpoints": sum(len(r["checkpoints"]) for r in records),
            "documented_limitations": [
                "Fourth coded-message token uses entity-0 symbol embedding.",
                "Training distractor identity is correlated with serial, semantic foil, and answer position in base.example.",
                "In scenes, source usually moves substantially faster than target; scene source is perceptually cued.",
                "Condition relabeling alters which identity embeddings receive each graph position."
            ]}


def rows_for_pair(a: int, b: int, pair_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns messages, candidate scenes, labels, and semantic category maps."""
    messages = np.empty((512, 4), dtype=np.int64)
    scenes = np.empty((512, 4, 3, 18), dtype=np.float32)
    labels = np.empty(512, dtype=np.int64)
    categories = np.empty((512, 4), dtype=np.int64)
    slot = 0
    for direction, event, answer, distractor_rank, repeat in itertools.product(
        range(2), range(2), range(4), range(8), range(REPEATS)
    ):
        src, tgt = ((a, b) if direction == 0 else (b, a))
        distractor = [i for i in range(10) if i not in (a, b)][distractor_rank]
        semantic = base.qset("R", src, tgt, distractor, event)
        serial = pair_index * 512 + slot
        rng = np.random.default_rng(np.random.SeedSequence([TEST_SEED, pair_index, slot, 11]))
        other = [1, 2, 3]
        rng.shuffle(other)
        order = other[:answer] + [0] + other[answer:]
        assert len(set(order)) == 4
        for position, member in enumerate(order):
            scene_rng = np.random.default_rng(np.random.SeedSequence([TEST_SEED, serial, 20 + position]))
            scenes[slot, position] = base.scene(semantic[member], distractor, scene_rng)
        messages[slot] = (src, tgt, 10 + event, 0)
        labels[slot] = answer
        categories[slot] = order
        slot += 1
    assert slot == 512
    assert np.array_equal(np.bincount(labels, minlength=4), np.full(4, 128))
    return messages, scenes, labels, categories


@torch.no_grad()
def score_pair(model: torch.nn.Module, row: tuple[np.ndarray, ...]) -> dict:
    messages, scenes, labels, categories = row
    choices = np.empty(len(labels), dtype=np.int64)
    margins = np.empty(len(labels), dtype=np.float32)
    binary = np.empty(len(labels), dtype=np.float32)
    device = torch.device("cpu")
    model.eval()
    for start in range(0, len(labels), 128):
        stop = min(start + 128, len(labels))
        msg = torch.nn.functional.one_hot(torch.tensor(messages[start:stop]), base.VOCAB).float()
        candidates = torch.from_numpy(scenes[start:stop])
        scores = model(msg.to(device), candidates.to(device)).cpu().numpy()
        ix = np.arange(stop - start)
        current_labels = labels[start:stop]
        category = categories[start:stop]
        reverse_pos = np.argmax(category == 1, axis=1)
        margin = scores[ix, current_labels] - scores[ix, reverse_pos]
        margins[start:stop] = margin
        binary[start:stop] = (margin > 0).astype(np.float32) + .5 * (margin == 0)
        choices[start:stop] = category[ix, scores.argmax(axis=1)]
    fractions = np.bincount(choices, minlength=4) / len(choices)
    return {"n_renderings": len(choices), "four_choice_accuracy": float(fractions[0]),
            "binary_role_accuracy": float(binary.mean()),
            "event_accuracy": float(fractions[0] + fractions[1]),
            "error_fractions": {k: float(v) for k, v in zip(
                ("correct", "role_only", "event_only", "both"), fractions)},
            "mean_correct_minus_reverse_margin": float(margins.mean())}


def evaluate(output_path: Path) -> dict:
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    audit_path = ROOT / "phase0_audit.json"
    if not audit_path.exists():
        raise FileNotFoundError("Run phase0 first")
    passed = json.loads(audit_path.read_text())
    assert passed["status"] == "passed_with_documented_generator_limitations"
    torch.set_num_threads(4)
    results = []
    for item in passed["records"]:
        edges = {tuple(e) for e in item["labeled_edges"]}
        base.N_ID = 10
        base.VOCAB = 12
        for checkpoint in item["checkpoints"]:
            model = study.prior.Listener10()
            model.load_state_dict(torch.load(checkpoint["model_path"], map_location="cpu", weights_only=True))
            checkpoint["model"] = model
        pair_rows = []
        for pair_index, (a, b) in enumerate(itertools.combinations(range(10), 2)):
            row = rows_for_pair(a, b, pair_index)
            pair_rows.append({"pair": [a, b], "observed": (a, b) in edges,
                              "scores": {str(checkpoint["seed"]): score_pair(checkpoint["model"], row)
                                         for checkpoint in item["checkpoints"]}})
        for checkpoint in item["checkpoints"]:
            del checkpoint["model"]
            model_scores = [{"pair": pair["pair"], "observed": pair["observed"],
                             **pair["scores"][str(checkpoint["seed"])]} for pair in pair_rows]
            results.append({"cohort": item["cohort"], "world": item["world"],
                            "condition": item["condition"], "seed": checkpoint["seed"],
                            "model_sha256": checkpoint["model_sha256"],
                            "pairs": model_scores})
        print(f"Completed {item['cohort']} {item['world']} {item['condition']}", flush=True)
    output = {"protocol": "PROTOCOL.json", "audit": "phase0_audit.json",
              "evaluation_seed": TEST_SEED, "models": results}
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Wrote {output_path} ({len(results)} models)", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("phase0", "phase1a"))
    parser.add_argument("--evaluation-seed", type=int, default=260107)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    if args.command == "phase0":
        path = ROOT / "phase0_audit.json"
        if path.exists():
            raise FileExistsError(path)
        result = audit()
        path.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"status": result["status"], "worlds": result["worlds"],
                          "checkpoints": result["checkpoints"]}, indent=2))
    else:
        global TEST_SEED
        TEST_SEED = args.evaluation_seed
        evaluate(args.output)


if __name__ == "__main__":
    main()
