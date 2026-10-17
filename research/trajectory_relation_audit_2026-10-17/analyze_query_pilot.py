"""Apply QUERY_LOCK.json's fixed metrics and retain all families and failures."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
LOCK = json.loads((HERE/"QUERY_LOCK.json").read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cosine(a, b):
    denominator = np.linalg.norm(a, axis=-1)*np.linalg.norm(b, axis=-1)
    out = np.full(denominator.shape, np.nan)
    np.divide(np.sum(a*b, axis=-1), denominator, out=out, where=denominator > 1e-12)
    return np.clip(out, -1, 1)


def main():
    models = {}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    labels = {"qwen05": "Qwen 0.5B (diagnostic)", "qwen15": "Qwen 1.5B",
              "qwen7b": "Qwen 7B", "llama3b": "Llama 3B"}
    for ax, key in zip(axes.flat, LOCK["models"]):
        meta_path = HERE/f"QUERY_{key}.json"
        states_path = HERE/f"QUERY_STATES_{key}.npz"
        meta = json.loads(meta_path.read_text())
        assert meta["lock_sha256"] == sha(HERE/"QUERY_LOCK.json")
        assert meta["script_sha256"] == sha(HERE/"run_query_pilot.py")
        assert meta["states_sha256"] == sha(states_path)
        source = np.load(states_path)
        raw = source["states"].astype(np.float64)
        norm = source["normalized_states"].astype(np.float64)
        g = source["readout_axes"].astype(np.float64)
        assert list(source["questions"]) == list(LOCK["questions"])
        assert list(source["conditions"]) == LOCK["conditions"]
        assert list(source["family_ids"]) == [f["id"] for f in LOCK["families"]]
        a, b = raw[:, :, 0]-raw[:, :, 1], raw[:, :, 2]-raw[:, :, 3]
        an, bn = norm[:, :, 0]-norm[:, :, 1], norm[:, :, 2]-norm[:, :, 3]
        unit = g/np.linalg.norm(g, axis=-1, keepdims=True)
        direction = unit[:, None, None, None, :]
        dot_a, dot_b = np.sum(an*direction, axis=-1), np.sum(bn*direction, axis=-1)
        ap, bp = an-dot_a[..., None]*direction, bn-dot_b[..., None]*direction
        raw_cos, norm_cos, proj_cos = cosine(a, b), cosine(an, bn), cosine(ap, bp)
        energy = (dot_a**2/(np.sum(an*an, axis=-1)+1e-15) +
                  dot_b**2/(np.sum(bn*bn, axis=-1)+1e-15))/2
        depth = np.arange(raw.shape[3])/(raw.shape[3]-1)
        late = depth >= 0.75
        means = np.nanmean(raw_cos[:, :, :, 1][:, :, late], axis=-1)
        projected_means = np.nanmean(proj_cos[:, :, :, 1][:, :, late], axis=-1)
        deltas = means[:, 0]-means[:, 1]
        delta_projected = projected_means[:, 0]-projected_means[:, 1]
        before = raw[:, 0, :, :, 0]
        after = raw[:, 1, :, :, 0]
        discrepancy = np.linalg.norm(before-after, axis=-1)/np.maximum(
            (np.linalg.norm(before, axis=-1)+np.linalg.norm(after, axis=-1))/2, 1e-12)
        max_discrepancy = float(np.max(discrepancy))
        technical_ok = max_discrepancy < 0.001
        counts = {task: 0 for task in LOCK["questions"]}
        competent = []
        family_rows = []
        for i, (saved, planned) in enumerate(zip(meta["families"], LOCK["families"])):
            assert saved["family"] == planned
            flags = []
            for task_record, task_key in zip(saved["tasks"], LOCK["questions"]):
                assert task_record["question_key"] == task_key
                for row, condition in zip(task_record["rows"], LOCK["conditions"]):
                    assert row["condition"] == condition
                    assert row["sentence"] == LOCK["sentences"][condition].format(**planned)
                    assert row["question"] == LOCK["questions"][task_key].format(**planned)
                    agent_a = condition.startswith("a_agent")
                    expected_a = agent_a if task_key == "agent" or condition.endswith("active") else not agent_a
                    assert row["expected"] == planned["a" if expected_a else "b"]
                    ok = bool(row["greedy_correct"] and row["score_correct"])
                    flags.append(ok)
                    counts[task_key] += ok
            fully = all(flags)
            assert fully == saved["fully_competent"]
            if fully:
                competent.append(i)
            family_rows.append({"id": planned["id"], "verb": planned["verb"],
                                "correct_answers": sum(flags), "fully_competent": fully,
                                "late_cos_agent": float(means[i, 0]),
                                "late_cos_first": float(means[i, 1]),
                                "delta": float(deltas[i]),
                                "delta_after_axis_removal": float(delta_projected[i])})
        competent_first = float(np.mean(means[competent, 1])) if competent else None
        passed = technical_ok and len(competent) >= 2 and float(np.mean(deltas)) > 0.25 and competent_first < 0
        # Scalar first-token output baseline, evaluated at final normalized states.
        margins = np.sum(norm*g[:, None, None, None, None, :], axis=-1)
        signed_product = np.sign((margins[:, :, 0, -1, 1]-margins[:, :, 1, -1, 1]) *
                                 (margins[:, :, 2, -1, 1]-margins[:, :, 3, -1, 1]))
        observed_sign = np.sign(raw_cos[:, :, -1, 1])
        models[key] = {
            "raw_sha256": sha(meta_path), "states_sha256": sha(states_path),
            "correct_answers_by_question": counts, "answers_per_question": 32,
            "fully_competent_family_ids": [LOCK["families"][i]["id"] for i in competent],
            "all_family_mean_late_agent_cos": float(np.mean(means[:, 0])),
            "all_family_mean_late_first_cos": float(np.mean(means[:, 1])),
            "all_family_mean_delta": float(np.mean(deltas)),
            "delta_positive_family_count": int(np.sum(deltas > 0)),
            "all_family_mean_delta_axis_removed": float(np.mean(delta_projected)),
            "mean_late_energy_removed": float(np.mean(energy[:, :, :, 1][:, :, late])),
            "competent_subset_mean_first_cos": competent_first,
            "max_prequestion_relative_discrepancy": max_discrepancy,
            "technical_invariance_passed": technical_ok,
            "descriptive_rule_passed": bool(passed),
            "final_geometry_sign_matches_scalar_output_response": int(np.sum(signed_product == observed_sign)),
            "scalar_output_response_comparisons": int(signed_product.size),
            "families": family_rows,
        }
        for q, (name, color) in enumerate((("Agent query", "#3B6FB6"), ("First-person query", "#D25B32"))):
            values = raw_cos[:, q, :, 1]
            for line in values:
                ax.plot(depth, line, color=color, alpha=0.17, lw=0.8)
            mean_curve = np.full(len(depth), np.nan)
            mean_curve[1:] = np.nanmean(values[:, 1:], axis=0)
            ax.plot(depth, mean_curve, color=color, label=name, lw=2.5)
        ax.axhline(0, color="#666", lw=0.7)
        ax.axvspan(0.75, 1, color="#777", alpha=0.06)
        ax.set_title(f"{labels[key]}  ·  answers {counts['agent']}/32, {counts['first']}/32", loc="left", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.15)
        ax.set_xlim(0, 1)
        ax.set_ylim(-1.05, 1.05)
    for ax in axes[-1]:
        ax.set_xlabel("Normalized layer depth")
    for ax in axes[:, 0]:
        ax.set_ylabel("Active/passive role-response cosine")
    handles, legend = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend, loc="upper center", bbox_to_anchor=(0.5, 0.935), ncol=2, frameon=False)
    fig.suptitle("Does the relationship between paths follow the question?", fontsize=16, weight="bold", y=0.99)
    fig.text(0.5, 0.947, "Eight fresh constructed families · every correct and incorrect answer retained", ha="center", fontsize=10, color="#555")
    fig.text(0.5, 0.012, "Thin lines: individual families. Thick lines: means. Shaded quarter: frozen summary window. Geometry alone does not identify an algorithm.",
             ha="center", fontsize=8.5, color="#555")
    fig.tight_layout(rect=(0, 0.04, 1, 0.87))
    decision = {"status": "bounded_query_measurement_diagnostic_complete",
                "lock_sha256": sha(HERE/"QUERY_LOCK.json"), "analysis_script_sha256": sha(__file__),
                "primary_descriptive_rule_passed": all(models[m]["descriptive_rule_passed"] for m in LOCK["primary_models"]),
                "models": models}
    target = HERE/"QUERY_SUMMARY.json"
    assert not target.exists(), "refusing overwrite"
    target.write_text(json.dumps(decision, indent=2)+"\n")
    for extension in ("png", "pdf"):
        output = HERE/f"QUERY_GEOMETRY.{extension}"
        assert not output.exists(), "refusing overwrite"
        fig.savefig(output, dpi=200, facecolor="white")
    print(json.dumps({"pass": decision["primary_descriptive_rule_passed"],
                      "models": {k: {name: row[name] for name in (
                          "correct_answers_by_question", "fully_competent_family_ids",
                          "all_family_mean_late_agent_cos", "all_family_mean_late_first_cos",
                          "all_family_mean_delta", "all_family_mean_delta_axis_removed",
                          "max_prequestion_relative_discrepancy", "descriptive_rule_passed",
                          "final_geometry_sign_matches_scalar_output_response")}
                                 for k, row in models.items()}}, indent=2))


if __name__ == "__main__":
    main()
