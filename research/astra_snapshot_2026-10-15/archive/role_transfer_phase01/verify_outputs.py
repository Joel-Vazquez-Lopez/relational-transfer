"""Check coverage, metric identities, and saved artifact hashes."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
OUT = ROOT / "ARTIFACT_MANIFEST.json"


def digest(path: Path) -> dict:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return {"path": str(path.relative_to(PROJECT)), "bytes": path.stat().st_size,
            "sha256": h.hexdigest()}


def check_suite(name: str) -> dict:
    suite = json.loads((ROOT / name).read_text())
    expected_pairs = {tuple(p) for p in itertools.combinations(range(10), 2)}
    assert len(suite["models"]) == 42
    model_keys = set()
    for model in suite["models"]:
        key = (model["cohort"], model["world"], model["condition"], model["seed"])
        assert key not in model_keys
        model_keys.add(key)
        pairs = model["pairs"]
        assert len(pairs) == 45
        assert {tuple(p["pair"]) for p in pairs} == expected_pairs
        assert sum(p["observed"] for p in pairs) == 20
        for pair in pairs:
            assert pair["n_renderings"] == 512
            fractions = pair["error_fractions"]
            assert abs(sum(fractions.values()) - 1) < 1e-10
            assert abs(pair["four_choice_accuracy"] - fractions["correct"]) < 1e-10
            assert abs(pair["event_accuracy"] - fractions["correct"] - fractions["role_only"]) < 1e-10
            assert 0 <= pair["binary_role_accuracy"] <= 1
    return {"models": len(suite["models"]), "model_pair_cases": 42 * 45,
            "renderings_per_pair": 512}


def main() -> None:
    checks = {name: check_suite(name) for name in ("all_pair_results.json", "all_pair_replication.json")}
    source_paths = [PROJECT / "ASTRA_RESEARCH_PLAN_20261006.md"] + [
        PROJECT / "d5_indirect_support" / name for name in
        ("run.py", "path2_control.py", "prospective_20261006.py",
         "prospective_20261006_confirm.py", "component_diagnostic_20261006.py",
         "audit_20261006.py")]
    artifact_paths = [p for p in ROOT.iterdir() if p.is_file() and p.name != OUT.name]
    artifacts = [digest(p) for p in sorted(source_paths + artifact_paths)]
    result = {"status": "verified", "coverage_checks": checks, "artifacts": artifacts,
              "note": "Manifest captures final local file contents. All analyses are developmental; protocol was amended for adaptive checks."}
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "coverage_checks": checks,
                      "artifact_count": len(artifacts)}, indent=2))


if __name__ == "__main__":
    main()
