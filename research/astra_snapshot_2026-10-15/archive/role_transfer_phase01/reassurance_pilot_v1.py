"""One prespecified development-only pilot on old graph new1, new seed 261003."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import generator_gate_v1 as gate
import role_aux_v1 as auxiliary

ROOT = Path(__file__).resolve().parent / "reassurance_pilot_v1"
WORLD = "new1"
SEED = 261003
WORLD_INFO = {
    "edges": ((0, 2), (0, 3), (0, 4), (0, 9), (1, 2), (1, 6), (1, 7), (1, 8),
              (2, 6), (2, 7), (3, 5), (3, 8), (3, 9), (4, 5), (4, 7), (4, 8),
              (5, 7), (5, 9), (6, 8), (6, 9)),
    "shared1": (2, 3),
    "shared2": (2, 9),
    "generator_seed": 3,
}


def configure() -> None:
    gate.WORLDS[WORLD] = WORLD_INFO
    gate.SEEDS = (SEED,)
    gate.OUTPUT = ROOT / "baseline"
    auxiliary.ROOT = ROOT / "auxiliary"


def summarize() -> dict:
    baseline_path = ROOT / "baseline" / "full" / WORLD / "balanced_distractor" / f"seed_{SEED}" / "metrics.json"
    arm_paths = {arm: ROOT / "auxiliary" / "full" / WORLD / arm / f"seed_{SEED}" / "metrics.json"
                 for arm in auxiliary.ARMS}
    if not baseline_path.exists() or not all(path.exists() for path in arm_paths.values()):
        return {"complete": False, "available": [str(path) for path in [baseline_path, *arm_paths.values()]
                                                 if path.exists()]}
    documents = {"baseline": json.loads(baseline_path.read_text()),
                 **{arm: json.loads(path.read_text()) for arm, path in arm_paths.items()}}
    result = {"complete": True, "world": WORLD, "seed": SEED, "arms": {}}
    for arm, document in documents.items():
        result["arms"][arm] = {}
        for scene in ("260107", "260108"):
            data = document["summary"][scene]
            result["arms"][arm][scene] = {
                "seen_four_choice": data["seen"]["four_choice_accuracy"],
                "unseen_binary_role": data["unseen"]["binary_role_accuracy"],
                "unseen_four_choice": data["unseen"]["four_choice_accuracy"],
                "unseen_event": data["unseen"]["event_accuracy"],
                "weak_unseen_pairs": sum(not row["observed"] and row["binary_role_accuracy"] < .8
                                         for row in document["evaluation"][scene]),
            }
    main = {arm: result["arms"][arm]["260107"] for arm in result["arms"]}
    result["pilot_success"] = (
        main["ordered_roles"]["unseen_binary_role"] > main["baseline"]["unseen_binary_role"]
        and main["ordered_roles"]["unseen_binary_role"] > main["unordered_identities"]["unseen_binary_role"]
        and main["ordered_roles"]["seen_four_choice"] >= .99
        and main["ordered_roles"]["unseen_event"] >= main["baseline"]["unseen_event"] - .02)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "baseline", "ordered_roles",
                                             "unordered_identities", "summarize"))
    args = parser.parse_args()
    configure()
    if args.command == "audit":
        assert gate.WORLDS[WORLD]["edges"] == WORLD_INFO["edges"]
        gate.configure(WORLD)
        assert len(gate.base.make_pool("canonical")) == 1280
        print(json.dumps({"world": WORLD, "seed": SEED, "pool_size": 1280,
                          "fresh_confirmation_graph": False}, indent=2))
    elif args.command == "baseline":
        gate.train(WORLD, "balanced_distractor", SEED, gate.STEPS, False)
    elif args.command in auxiliary.ARMS:
        auxiliary.train(WORLD, args.command, SEED)
    else:
        print(json.dumps(summarize(), indent=2))


if __name__ == "__main__":
    main()
