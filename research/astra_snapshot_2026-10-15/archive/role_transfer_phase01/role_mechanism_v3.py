"""Repeat unchanged v2 subspace analysis on the two held-aside training starts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import confirmation_runner_v1 as confirmation
import role_mechanism_v1 as v1
import role_mechanism_v2 as v2

ROOT = Path(__file__).resolve().parent / "role_mechanism_v3"
SEEDS = (271002, 271003)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", choices=confirmation.WORLDS, required=True)
    args = parser.parse_args()
    destination = ROOT / f"{args.world}.json"
    if destination.exists():
        raise FileExistsError(destination)
    confirmation.audit()
    confirmation.configure_modules()
    train = v1.scene_suite(620107)
    episodes = v1.eval_rows()
    records = []
    for seed in SEEDS:
        v1.SEED = seed
        for arm in v2.ARMS:
            records.append(v2.one_model(args.world, arm, train, episodes))
    assert len(records) == 4
    output = {"protocol_sha256": hashlib.sha256((ROOT / "PROTOCOL.json").read_bytes()).hexdigest(),
              "records": records}
    destination.write_text(json.dumps(output, indent=2) + "\n")
    values = []
    for seed in SEEDS:
        arms = {r["arm"]:r for r in records if r["seed"] == seed}
        drops = {arm: z["results"]["ordinary"]["mean_margin"]
                 - z["results"]["role_order"]["mean_margin"]
                 for arm,z in arms.items()}
        values.append({"seed":seed,"unordered_drop":drops["unordered_identities"],
                       "ordered_drop":drops["ordered_roles"],
                       "contrast":drops["ordered_roles"]-drops["unordered_identities"]})
    print(json.dumps({"world":args.world,"starts":values},indent=2),flush=True)


if __name__ == "__main__":
    main()
