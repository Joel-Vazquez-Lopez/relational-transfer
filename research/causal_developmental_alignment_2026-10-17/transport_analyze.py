"""Predeclared analysis of the role-vs-position transport effects."""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
LOCK = json.loads((HERE / "TRANSPORT_LOCK.json").read_text())
LAYER_IDS = LOCK["layers_zero_indexed"]
AA = "b_agent_active_first"
AP = "b_agent_passive_second"
PA = "b_patient_active_second"
PP = "b_patient_passive_first"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def contrasts(family):
    rows = {(r["layer"], r["site"], r["donor"]): r["margin_b_minus_a"]
            for r in family["patches"]}
    result = []
    for layer in LAYER_IDS:
        per_site = {}
        for site in ("period", "prefix"):
            a, b, c, d = (rows[(layer, site, name)] for name in (AA, AP, PA, PP))
            per_site[site] = {
                "role_D": ((a + b) - (c + d)) / 2,
                "first_name_position_P": ((a + d) - (b + c)) / 2,
                "voice_V": ((a + c) - (b + d)) / 2,
                "four_raw_margins": {AA: a, AP: b, PA: c, PP: d},
            }
        self_margin = rows[(layer, "period", "receiver_cleft_a_agent")]
        unrelated = rows[(layer, "period", "same_answer_unrelated")]
        result.append({"layer": layer, **per_site,
                       "self_patch_margin": self_margin,
                       "same_answer_unrelated_effect": unrelated - self_margin})
    return result


def summarize(model_record):
    cases = []
    for fam in model_record["results"]:
        profile = contrasts(fam)
        d = sum(x["period"]["role_D"] for x in profile) / len(profile)
        p = sum(x["period"]["first_name_position_P"] for x in profile) / len(profile)
        w = sum(x["prefix"]["role_D"] for x in profile) / len(profile)
        cases.append({"family_id": fam["family"]["id"], "profile": profile,
                      "mean_role_D": d, "mean_position_P": p, "mean_wrong_site_D": w,
                      "passes_case_rule": d > 0 and d > abs(p) and abs(w) < 0.5 * d})
    d = sum(x["mean_role_D"] for x in cases) / len(cases)
    p = sum(x["mean_position_P"] for x in cases) / len(cases)
    w = sum(x["mean_wrong_site_D"] for x in cases) / len(cases)
    return {"cases": cases, "mean_role_D": d, "mean_position_P": p,
            "mean_wrong_site_D": w,
            "development_pass": d > 0.5 and d > abs(p) and abs(w) < 0.5 * d,
            "evaluation_pass": all(x["passes_case_rule"] for x in cases)}


def main(split):
    assert split in ("development", "evaluation")
    outpath = HERE / ("TRANSPORT_DEV_DECISION.json" if split == "development"
                      else "TRANSPORT_EVAL_ANALYSIS.json")
    assert not outpath.exists(), "refusing overwrite"
    models = {}
    for model in LOCK["models"]:
        path = HERE / f"TRANSPORT_PATCH_{model}_{split}.json"
        data = json.loads(path.read_text())
        assert data["lock_sha256"] == sha(HERE / "TRANSPORT_LOCK.json")
        models[model] = {"patch_sha256": sha(path), **summarize(data)}
    passed = all(models[m]["development_pass" if split == "development" else "evaluation_pass"]
                 for m in LOCK["models"])
    result = {"status": f"frozen_transport_{split}_analysis", "split": split,
              "lock_sha256": sha(HERE / "TRANSPORT_LOCK.json"),
              "script_sha256": sha(__file__), "models": models, "pass": passed,
              "notes": ["D/P/V are algebraic contrasts of patched answer margins, not recovered internal operation order.",
                        "The wrong-site criterion compares magnitudes: abs(wrong D) < 0.5*primary D.",
                        "Same-answer unrelated effects are diagnostic and do not automatically fail role transport."]}
    outpath.write_text(json.dumps(result, indent=2) + "\n")
    print(split, "pass", passed)
    for m, x in models.items():
        print(m, "D", round(x["mean_role_D"], 4), "P", round(x["mean_position_P"], 4),
              "wrong D", round(x["mean_wrong_site_D"], 4))


if __name__ == "__main__":
    main(sys.argv[1])
