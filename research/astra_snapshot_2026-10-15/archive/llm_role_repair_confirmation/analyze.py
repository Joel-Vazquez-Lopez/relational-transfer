"""Verify the frozen replication and summarize paired outcomes."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROTOCOL = json.loads((ROOT / "PROTOCOL.json").read_text())
PROTOCOL_HASH = hashlib.sha256((ROOT / "PROTOCOL.json").read_bytes()).hexdigest()
WORLDS = tuple(PROTOCOL["worlds"])
SEEDS = tuple(PROTOCOL["seeds"])


def load(world, seed, arm):
    path = ROOT / f"{world}_{seed}_{arm}" / "result.json"
    result = json.loads(path.read_text())
    assert (result["world"], result["seed"], result["arm"]) == (world, seed, arm)
    assert result["protocol_sha256"] == PROTOCOL_HASH
    assert len(result["records"]) == 3000
    return result


def score(rows, predicate):
    chosen = [row for row in rows if predicate(row)]
    return {"correct": sum(row["correct"] for row in chosen),
            "total": len(chosen), "valid": sum(row["valid"] for row in chosen)}


def metrics(result):
    rows = result["records"]
    by_form = {form: score(rows, lambda r, f=form: r["form"] == f)
               for form in ("active", "passive", "actor_cleft", "patient_cleft")}
    by_verb = {str(seen): score(rows, lambda r, v=seen: r["seen_verb"] == v)
               for seen in (True, False)}
    by_wording = {wording: score(rows, lambda r, w=wording: r["wording"] == w)
                  for wording in ("performed", "did", "carried_out")}
    form_events = defaultdict(list)
    for row in rows:
        form_events[(row["event"], row["form"])].append(row["correct"])
    assert len(form_events) == 1000 and all(len(group) == 3 for group in form_events.values())
    all_three = {form: sum(all(correct) for (event, f), correct in form_events.items() if f == form)
                 for form in by_form}
    return {"forms": by_form,
            "familiar": score(rows, lambda r: r["form"] in ("active", "passive")),
            "seen_verb": by_verb, "wording": by_wording,
            "all_three_wordings_correct_by_form": all_three,
            "invalid": sum(not row["valid"] for row in rows)}


def delta(a, b):
    assert a["total"] == b["total"]
    return (a["correct"] - b["correct"]) / a["total"]


def main():
    analysis = {"protocol_sha256": PROTOCOL_HASH, "units": [], "baseline": {}}
    for world in WORLDS:
        baseline = load(world, 0, "pretrained")
        analysis["baseline"][world] = metrics(baseline)
        for seed in SEEDS:
            ordinary = load(world, seed, "ordinary")
            ordered = load(world, seed, "ordered_roles")
            assert ordinary["training"]["updates"] == ordered["training"]["updates"] == 120
            assert ordinary["training"]["train_examples"] == ordered["training"]["train_examples"] == 240
            assert ordinary["training"]["initial_adapter_sha256"] == ordered["training"]["initial_adapter_sha256"]
            for source in (ordinary, ordered):
                for saved, fixed in zip(source["records"], baseline["records"], strict=True):
                    assert {k: v for k, v in saved.items() if k not in ("answer", "valid", "correct")} == \
                           {k: v for k, v in fixed.items() if k not in ("answer", "valid", "correct")}
            om, rm = metrics(ordinary), metrics(ordered)
            pair_deltas = {}
            for pair in {tuple(r["pair"]) for r in ordinary["records"]}:
                condition = lambda r, p=pair: tuple(r["pair"]) == p and r["form"] == "patient_cleft"
                pair_deltas["-".join(map(str, pair))] = (score(ordered["records"], condition)["correct"] -
                                                         score(ordinary["records"], condition)["correct"])
            assert len(pair_deltas) == 25
            analysis["units"].append({"world": world, "seed": seed,
                                      "ordinary": om, "ordered_roles": rm,
                                      "patient_cleft_gain": delta(rm["forms"]["patient_cleft"], om["forms"]["patient_cleft"]),
                                      "familiar_gain": delta(rm["familiar"], om["familiar"]),
                                      "patient_pair_delta_correct_out_of_30": pair_deltas})
    units = analysis["units"]
    analysis["decision"] = {
        "positive_units": sum(u["patient_cleft_gain"] > 0 for u in units),
        "equal_unit_patient_gain": sum(u["patient_cleft_gain"] for u in units) / len(units),
        "equal_unit_familiar_gain": sum(u["familiar_gain"] for u in units) / len(units),
    }
    analysis["decision"]["frozen_directional_replication"] = (
        analysis["decision"]["positive_units"] >= 5 and
        analysis["decision"]["equal_unit_patient_gain"] >= 0.10)
    analysis["decision"]["material_familiar_loss"] = analysis["decision"]["equal_unit_familiar_gain"] < -0.02
    (ROOT / "ANALYSIS.json").write_text(json.dumps(analysis, indent=2) + "\n")
    print(json.dumps(analysis["decision"], indent=2))


if __name__ == "__main__":
    main()
