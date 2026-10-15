"""Development-only structure × objective study on previously examined graphs."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
PROJECT = PHASE.parent
sys.path.insert(0, str(PHASE))
sys.path.insert(0, str(PROJECT / "d5_indirect_support"))
import generator_gate_v1 as gate  # noqa: E402
import role_aux_v1 as auxiliary  # noqa: E402
import all_pair_audit as evaluator  # noqa: E402

base = gate.base
prior = gate.prior
SOURCE_MANIFEST = PHASE / "rq2_rewire_repair_v1" / "GRAPH_MANIFEST.json"
PROTOCOL = HERE / "PROTOCOL.json"
ARMS = ("standard", "participant_only", "ordered_roles", "role_swap", "permutation_control")
SEEDS = (321001, 321002)
CONDITIONS = ("sham", "high")
CHECKPOINTS = (1500, 3000, 4500, 6000)
P = torch.tensor((1, 0, 3, 2), dtype=torch.long)
WEIGHT = 0.5
SWAP_WEIGHT = 0.05
SCENES = (260107, 260108)
TRAIN_STEPS = 6000
PREFLIGHT_STEPS = 960


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest() -> dict:
    return json.loads(SOURCE_MANIFEST.read_text())


def protocol() -> dict:
    return json.loads(PROTOCOL.read_text())


def graph(index: int, condition: str) -> tuple[dict, list[list[int]]]:
    row = manifest()["families"][index]
    return row, row["conditions"][condition]["edges"]


def configure(index: int, condition: str) -> list[list[int]]:
    _, edges = graph(index, condition)
    base.N_ID = 10
    base.VOCAB = 13
    base.BASE_EDGES = tuple(map(tuple, edges))
    base.MAPPINGS = {"canonical": tuple(range(10))}
    gate.PAD_ID = 12
    gate.VOCAB = 13
    return edges


def batches(index: int, condition: str, seed: int):
    configure(index, condition)
    pool = base.make_pool("canonical")
    assert len(pool) == 1280
    for epoch in range(TRAIN_STEPS * base.BATCH // len(pool)):
        order = np.random.default_rng(np.random.SeedSequence([seed, 700, epoch])).permutation(len(pool))
        schedule = gate.balanced_schedule(pool, order, seed, epoch)
        for offset in range(0, len(pool), base.BATCH):
            rows = []
            for j, pool_index in enumerate(order[offset:offset + base.BATCH]):
                row_index = offset + j
                serial = epoch * len(pool) + row_index
                distractors = tuple(schedule[(row_index, member)] for member in range(4))
                rows.append(gate.episode(pool[int(pool_index)], seed, serial, distractors))
            yield rows


def tensors(rows: list[tuple]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    msg = gate.oracle([r[0] for r in rows], torch.device("cpu"))
    scenes = torch.from_numpy(np.stack([r[1] for r in rows]))
    labels = torch.tensor([r[2] for r in rows], dtype=torch.long)
    return msg, scenes, labels


def transformed(msg: torch.Tensor, scenes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Swap participant identities everywhere, then exchange paired candidates."""
    src = msg[:, 0].argmax(-1)
    tgt = msg[:, 1].argmax(-1)
    assert torch.all(src != tgt)
    out_msg = msg.clone()
    out_msg[:, 0] = msg[:, 1]
    out_msg[:, 1] = msg[:, 0]
    perm = torch.arange(10).expand(len(msg), -1).clone()
    perm.scatter_(1, src[:, None], tgt[:, None])
    perm.scatter_(1, tgt[:, None], src[:, None])
    out_scenes = scenes.clone()
    out_scenes[..., :10] = torch.gather(
        scenes[..., :10], -1, perm[:, None, None, :].expand(-1, 4, 3, -1))
    return out_msg, out_scenes[:, P]


def objective(model: auxiliary.RoleListener, rows: list[tuple], arm: str,
              swap_weight: float = SWAP_WEIGHT):
    if arm == "participant_only":
        return auxiliary.objective(model, rows, "unordered_identities", torch.device("cpu"))
    if arm == "ordered_roles":
        return auxiliary.objective(model, rows, "ordered_roles", torch.device("cpu"))
    msg, scenes, labels = tensors(rows)
    z = model(msg, scenes)
    native = F.cross_entropy(z, labels)
    if arm == "standard":
        return native, {"native": float(native.detach()), "auxiliary": 0.0}
    if arm == "role_swap":
        tmsg, tscenes = transformed(msg, scenes)
        tz = model(tmsg, tscenes)
        c1 = z[:, P] - z[:, P].mean(-1, keepdim=True)
        c2 = tz - tz.mean(-1, keepdim=True)
        extra = (c1 - c2).square().mean()
    elif arm == "permutation_control":
        tz = model(msg, scenes[:, P])
        extra = F.cross_entropy(tz, P[labels])
    else:
        raise ValueError(arm)
    weight = swap_weight if arm == "role_swap" else WEIGHT
    return native + weight * extra, {
        "native": float(native.detach()), "auxiliary": float(extra.detach())}


def eval_pair(model, a: int, b: int, pair_index: int, scene_seed: int, repeats: int = 1) -> dict:
    evaluator.TEST_SEED = scene_seed
    rows = evaluator.rows_for_pair(a, b, pair_index)
    rows[0][:, 3] = gate.PAD_ID
    if repeats == 1:
        rows = tuple(x[::4] for x in rows)
    return evaluator.score_pair(model, rows)


def evaluation(model, index: int, condition: str, scene_seed: int) -> dict:
    configure(index, condition)
    row, edges = graph(index, condition)
    focal = tuple(row["focal_pair"])
    seen = {tuple(edge) for edge in edges}
    allpairs = list(itertools.combinations(range(10), 2))
    assert focal not in seen
    nonfocal = [pair for pair in allpairs if pair not in seen and pair != focal][:8]
    familiar = [pair for pair in allpairs if pair in seen][:4]
    pairs = {pair: i for i, pair in enumerate(allpairs)}
    focal_score = eval_pair(model, *focal, pairs[focal], scene_seed, repeats=4)
    nonfocal_rows = [{"pair": pair, **eval_pair(model, *pair, pairs[pair], scene_seed)}
                     for pair in nonfocal]
    familiar_rows = [{"pair": pair, **eval_pair(model, *pair, pairs[pair], scene_seed)}
                     for pair in familiar]
    return {
        "focal": focal_score,
        "nonfocal_unseen_pairs": nonfocal_rows,
        "nonfocal_unseen_binary_role": float(np.mean([r["binary_role_accuracy"] for r in nonfocal_rows])),
        "familiar_four_choice": float(np.mean([r["four_choice_accuracy"] for r in familiar_rows])),
        "sample_sizes": {"focal": 512, "nonfocal_pairs": 8, "renderings_per_nonfocal_pair": 128,
                         "familiar_pairs": 4, "renderings_per_familiar_pair": 128},
    }


def output_dir(index: int, condition: str, seed: int, arm: str, preflight: bool,
               swap_weight: float) -> Path:
    kind = "preflight" if preflight else "full"
    label = (f"{arm}_weight_{str(swap_weight).replace('.', 'p')}"
             if preflight and arm == "role_swap" else arm)
    return HERE / kind / f"family{index}" / condition / f"seed_{seed}" / label


def audit() -> dict:
    manifest_data = manifest()
    assert len(manifest_data["families"]) == 8
    if PROTOCOL.exists():
        p = protocol()
        assert p["source_manifest_sha256"] == digest(SOURCE_MANIFEST)
        assert p["script_sha256"] == digest(Path(__file__))
    out = []
    for i, family in enumerate(manifest_data["families"]):
        focal = tuple(family["focal_pair"])
        for condition in CONDITIONS:
            edges = configure(i, condition)
            assert len(edges) == 20 and focal not in {tuple(e) for e in edges}
            pool = base.make_pool("canonical")
            assert len(pool) == 1280
            assert len({(spec[1][0], spec[1][1], spec[2]) for spec in pool}) == 80
            out.append({"family": i, "condition": condition, "training_rows": len(pool)})
    configure(0, "sham")
    rows = next(batches(0, "sham", SEEDS[0]))
    msg, scenes, labels = tensors(rows)
    tmsg, tscenes = transformed(msg, scenes)
    round_msg, round_scenes = transformed(tmsg, tscenes)
    assert torch.equal(round_msg, msg) and torch.equal(round_scenes, scenes)
    assert torch.equal(P[P[labels]], labels)
    # Every scene contains the two relevant participants and an unrelated distractor.
    ids = scenes[..., :10].argmax(-1)
    src = msg[:, 0].argmax(-1)[:, None, None]
    tgt = msg[:, 1].argmax(-1)[:, None, None]
    assert torch.all((ids == src).sum(-1) == 1)
    assert torch.all((ids == tgt).sum(-1) == 1)
    # Directly verify transformed gold scenes now describe the swapped target roles.
    gold = tscenes[torch.arange(len(labels)), P[labels]]
    gold_ids = gold[..., :10].argmax(-1)
    assert torch.all((gold_ids == tgt[:, 0]).sum(-1) == 1)
    assert torch.all((gold_ids == src[:, 0]).sum(-1) == 1)
    gradients = {}
    for arm in ARMS:
        base.set_seed(SEEDS[0])
        model = auxiliary.RoleListener()
        model.zero_grad(set_to_none=True)
        loss, detail = objective(model, rows, arm)
        assert torch.isfinite(loss)
        loss.backward()
        q = float(model.sym.grad.norm())
        c = float(model.candidate.inp.weight.grad.norm())
        assert q > 0 and c > 0 and math.isfinite(q) and math.isfinite(c)
        gradients[arm] = {"loss": float(loss.detach()), "query_grad": q,
                          "scene_grad": c, **detail}
    return {"valid": True, "graphs": out, "transformation_involution": True,
            "gold_label_permutation_valid": True, "gradients": gradients,
            "source_manifest_sha256": digest(SOURCE_MANIFEST)}


def train(index: int, condition: str, seed: int, arm: str, preflight: bool = False,
          swap_weight: float = SWAP_WEIGHT) -> dict:
    assert 0 <= index < 8 and condition in CONDITIONS and seed in SEEDS and arm in ARMS
    assert preflight or swap_weight == SWAP_WEIGHT
    audit()
    dest = output_dir(index, condition, seed, arm, preflight, swap_weight)
    final_path = dest / "metrics.json"
    if final_path.exists():
        raise FileExistsError(final_path)
    dest.mkdir(parents=True, exist_ok=True)
    configure(index, condition)
    base.set_seed(seed)
    torch.set_num_threads(2)
    model = auxiliary.RoleListener()
    optimizer = torch.optim.AdamW(model.parameters(), lr=base.LR)
    stop = PREFLIGHT_STEPS if preflight else TRAIN_STEPS
    marks = (PREFLIGHT_STEPS,) if preflight else CHECKPOINTS
    state_path = dest / "training_state.pt"
    start = 0
    trace = []
    if state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        assert state["arm"] == arm and state["seed"] == seed and state["condition"] == condition
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["rng"])
        start = state["step"]
        trace = state["trace"]
    t0 = time.monotonic()
    for step, rows in enumerate(batches(index, condition, seed), 1):
        if step <= start:
            continue
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, detail = objective(model, rows, arm, swap_weight)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite loss at {step}")
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            trace.append({"step": step, **detail})
        if step in marks:
            model.eval()
            checkpoints = {str(s): evaluation(model, index, condition, s) for s in SCENES}
            snapshot = dest / f"model_step_{step}.pt"
            torch.save(model.state_dict(), snapshot)
            payload = {"step": step, "arm": arm, "seed": seed, "condition": condition,
                       "family": index, "model": model.state_dict(),
                       "optimizer": optimizer.state_dict(), "rng": torch.get_rng_state(),
                       "trace": trace}
            (dest / f"checkpoint_{step}.json").write_text(json.dumps({
                "step": step, "model_sha256": digest(snapshot), "evaluation": checkpoints,
                "elapsed_seconds_since_resume": time.monotonic() - t0}, indent=2) + "\n")
            torch.save(payload, state_path)
            print(json.dumps({"checkpoint": str(dest), "step": step,
                              "focal": checkpoints["260107"]["focal"]["binary_role_accuracy"],
                              "familiar": checkpoints["260107"]["familiar_four_choice"]}), flush=True)
        if step == stop:
            break
    assert step == stop
    result = {"family": index, "condition": condition, "seed": seed, "arm": arm,
              "steps": stop, "preflight": preflight,
              "swap_weight": swap_weight if arm == "role_swap" else None,
              "source_manifest_sha256": digest(SOURCE_MANIFEST),
              "protocol_sha256": digest(PROTOCOL) if PROTOCOL.exists() else None,
              "script_sha256": digest(Path(__file__)),
              "model_sha256": digest(dest / f"model_step_{stop}.pt"),
              "checkpoint_steps": list(marks), "loss_trace": trace,
              "elapsed_seconds_since_resume": time.monotonic() - t0}
    final_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"completed": str(dest), "elapsed_seconds": result["elapsed_seconds_since_resume"]}), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("audit", "train", "preflight"))
    parser.add_argument("--family", type=int, default=0)
    parser.add_argument("--condition", choices=CONDITIONS, default="sham")
    parser.add_argument("--seed", type=int, choices=SEEDS, default=SEEDS[0])
    parser.add_argument("--arm", choices=ARMS, default="role_swap")
    parser.add_argument("--swap-weight", type=float, default=SWAP_WEIGHT)
    args = parser.parse_args()
    if args.action == "audit":
        print(json.dumps(audit(), indent=2))
    else:
        train(args.family, args.condition, args.seed, args.arm,
              args.action == "preflight", args.swap_weight)


if __name__ == "__main__":
    main()
