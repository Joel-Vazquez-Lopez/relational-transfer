"""Reproduce cross-voice response angles and the readout-axis diagnostic."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "family_dependent_computation_2026-10-17"
MODELS = ("qwen05", "qwen15", "qwen7b", "llama3b")
LABELS = ("Qwen 0.5B · failed", "Qwen 1.5B", "Qwen 7B", "Llama 3B")
COLORS = ("#8F8F8F", "#3B6FB6", "#D25B32", "#23846A")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cosine(a, b):
    denominator = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    out = np.full(denominator.shape, np.nan)
    np.divide(np.sum(a*b, axis=-1), denominator, out=out, where=denominator > 1e-12)
    return np.clip(out, -1, 1)


def project_out(x, axis):
    return x - (x @ axis)[..., None] * axis


def serial(x):
    return np.where(np.isfinite(x), x, None).tolist()


def main():
    data = {}
    for key in MODELS:
        source = SOURCE / f"PATH_VIEW_{key}.npz"
        raw = np.load(source)["states"].astype(np.float64)
        assert raw.shape[:1] == (4,) and raw.shape[2] == 2
        readout_path = HERE / f"READOUT_{key}.npz"
        meta = json.loads((HERE / f"READOUT_{key}.json").read_text())
        assert sha(source) == meta["source_states_sha256"]
        assert sha(readout_path) == meta["output_sha256"]
        z = np.load(readout_path)
        normed = z["normalized_states"].astype(np.float64)
        g = z["readout_weights"][0] - z["readout_weights"][1]
        g = g / np.linalg.norm(g)
        a, b = raw[0] - raw[1], raw[2] - raw[3]
        an, bn = normed[0] - normed[1], normed[2] - normed[3]
        ap, bp = project_out(an, g), project_out(bn, g)
        c_raw, c_norm, c_proj = cosine(a, b), cosine(an, bn), cosine(ap, bp)
        energy = ((an@g)**2/(np.sum(an**2, axis=-1)+1e-15) +
                  (bn@g)**2/(np.sum(bn**2, axis=-1)+1e-15))/2
        rng = np.random.default_rng(1701026)
        random_means = []
        for _ in range(256):
            axis = rng.normal(size=len(g))
            axis /= np.linalg.norm(axis)
            cr = cosine(project_out(an, axis), project_out(bn, axis))
            random_means.append(float(np.mean(cr[-7:, 1])))
        raw_logit_margins = normed @ (z["readout_weights"][0]-z["readout_weights"][1])
        data[key] = {
            "source_sha256": sha(source), "readout_sha256": sha(readout_path),
            "cos_raw": serial(c_raw), "cos_final_norm": serial(c_norm),
            "cos_without_first_token_axis": serial(c_proj),
            "mean_response_energy_on_axis": energy.tolist(),
            "first_token_logit_margins_A_minus_B": raw_logit_margins.tolist(),
            "late_seven_blocks": {
                "event_cos_raw": float(np.mean(c_raw[-7:, 0])),
                "answer_cos_raw": float(np.mean(c_raw[-7:, 1])),
                "answer_cos_final_norm": float(np.mean(c_norm[-7:, 1])),
                "answer_cos_axis_removed": float(np.mean(c_proj[-7:, 1])),
                "answer_fraction_energy_removed": float(np.mean(energy[-7:, 1])),
                "random_axis_mean_cos": float(np.mean(random_means)),
                "random_axis_descriptive_2_5_97_5_percentiles": np.quantile(random_means, [0.025, 0.975]).tolist(),
            },
        }
    output = HERE / "RELATION_AUDIT.json"
    assert not output.exists(), "refusing overwrite"
    output.write_text(json.dumps({"status": "exploratory_one_selected_family",
                                  "script_sha256": sha(__file__), "models": data}, indent=2) + "\n")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for key, label, color in zip(MODELS, LABELS, COLORS):
        row = data[key]
        x = np.linspace(0, 1, len(row["cos_raw"]))
        style = "--" if key == "qwen05" else "-"
        axes[0, 0].plot(x, np.array(row["cos_raw"], dtype=float)[:, 0], style, color=color, label=label, lw=2)
        axes[0, 1].plot(x, np.array(row["cos_raw"], dtype=float)[:, 1], style, color=color, lw=2)
        axes[1, 0].plot(x, np.array(row["cos_without_first_token_axis"], dtype=float)[:, 1], style, color=color, lw=2)
        axes[1, 1].plot(x, np.array(row["mean_response_energy_on_axis"])[:, 1]*100, style, color=color, lw=2)
    titles = ["Before the question: responses oppose", "At the answer: successful models align",
              "Answer site after removing one readout axis", "How much response energy that axis removes"]
    for i, (ax, title) in enumerate(zip(axes.flat, titles)):
        ax.set_title(title, fontsize=11, loc="left")
        ax.spines[["right", "top"]].set_visible(False)
        ax.grid(alpha=0.15)
        ax.set_xlim(0, 1)
        if i != 3:
            ax.axhline(0, color="#555", lw=0.7)
            ax.set_ylim(-1.05, 1.05)
            ax.set_ylabel("Active/passive response cosine")
        else:
            ax.set_ylabel("Mean squared norm removed (%)")
            ax.set_ylim(bottom=0)
        if i >= 2:
            ax.set_xlabel("Normalized layer depth")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=4, frameon=False)
    fig.suptitle("Relationships between paths reveal what size curves hide", fontsize=16, weight="bold", y=0.99)
    fig.text(0.5, 0.94, "Miriam / Adrian ‘pushed’ · one case selected after the behavioral gate", ha="center", fontsize=10, color="#555")
    fig.text(0.5, 0.013, "Exploratory: alignment can reflect distributed answer coding. Removing one output axis does not remove all answer information.",
             ha="center", fontsize=8.5, color="#555")
    fig.tight_layout(rect=(0, 0.04, 1, 0.86))
    for suffix in ("png", "pdf"):
        path = HERE / f"RELATION_AUDIT.{suffix}"
        assert not path.exists(), "refusing overwrite"
        fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    print(json.dumps({key: row["late_seven_blocks"] for key, row in data.items()}, indent=2))


if __name__ == "__main__":
    main()
