"""Compare unordered participant labels with the two frozen reference arms."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT.parent / "llm_role_repair_confirmation"
PROTOCOL_HASH = hashlib.sha256((ROOT / "PROTOCOL.json").read_bytes()).hexdigest()
REFERENCE_HASH = hashlib.sha256((REFERENCE / "PROTOCOL.json").read_bytes()).hexdigest()
WORLDS = ("fresh1", "fresh2", "fresh3")
SEEDS = (830211, 830212)
FORMS = ("active", "passive", "actor_cleft", "patient_cleft")


def load(path):
    return json.loads(path.read_text())


def score(rows, form):
    selected = [r for r in rows if r["form"] == form]
    return {"correct": sum(r["correct"] for r in selected), "total": len(selected),
            "valid": sum(r["valid"] for r in selected)}


def main():
    units = []
    for world in WORLDS:
        for seed in SEEDS:
            control = load(ROOT / f"{world}_{seed}_unordered_pair" / "result.json")
            ordinary = load(REFERENCE / f"{world}_{seed}_ordinary" / "result.json")
            ordered = load(REFERENCE / f"{world}_{seed}_ordered_roles" / "result.json")
            assert control["protocol_sha256"] == PROTOCOL_HASH
            assert control["reference_protocol_sha256"] == REFERENCE_HASH
            assert control["training"]["updates"] == ordered["training"]["updates"] == 120
            assert control["training"]["train_examples"] == ordered["training"]["train_examples"] == 240
            assert control["training"]["initial_adapter_sha256"] == ordered["training"]["initial_adapter_sha256"]
            assert control["training"]["initial_head_sha256"] == ordered["training"]["initial_head_sha256"]
            for reference in (ordinary, ordered):
                for a, b in zip(control["records"], reference["records"], strict=True):
                    assert {k: v for k, v in a.items() if k not in ("answer", "valid", "correct")} == \
                           {k: v for k, v in b.items() if k not in ("answer", "valid", "correct")}
            metrics = {}
            for arm, result in (("ordinary", ordinary), ("unordered_pair", control), ("ordered_roles", ordered)):
                metrics[arm] = {"forms": {form: score(result["records"], form) for form in FORMS},
                                "familiar": {"correct": sum(r["correct"] for r in result["records"]
                                                    if r["form"] in ("active", "passive")), "total": 1500},
                                "invalid": sum(not r["valid"] for r in result["records"])}
            units.append({"world": world, "seed": seed, "metrics": metrics})
    aggregate = {}
    for arm in ("ordinary", "unordered_pair", "ordered_roles"):
        aggregate[arm] = {form: {"correct": sum(u["metrics"][arm]["forms"][form]["correct"] for u in units),
                                  "total": 4500} for form in FORMS}
        aggregate[arm]["familiar"] = {"correct": sum(u["metrics"][arm]["familiar"]["correct"] for u in units),
                                      "total": 9000}
    (ROOT / "ANALYSIS.json").write_text(json.dumps({"protocol_sha256": PROTOCOL_HASH,
                                                     "units": units, "aggregate": aggregate}, indent=2) + "\n")
    for unit in units:
        m = unit["metrics"]
        print(unit["world"], unit["seed"], *(f"{arm}={m[arm]['forms']['patient_cleft']['correct']}/750,{m[arm]['familiar']['correct']}/1500"
                                               for arm in ("ordinary", "unordered_pair", "ordered_roles")))


if __name__ == "__main__":
    main()
