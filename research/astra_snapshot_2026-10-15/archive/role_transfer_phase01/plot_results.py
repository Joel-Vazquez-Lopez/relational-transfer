"""Publication-style descriptive figure for the all-pair development audit."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "all_pair_overview.png"


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    data = json.loads((ROOT / "SUMMARY.json").read_text())["all_pairs"]
    models = data["models"]
    full = json.loads((ROOT / "all_pair_results.json").read_text())
    unseen = [p["binary_role_accuracy"] for m in full["models"]
              for p in m["pairs"] if not p["observed"]]
    colors = {"screen": "#2764a5", "confirmation_history": "#c26b23"}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7), constrained_layout=True)
    ax = axes[0]
    ax.plot([.45, 1.01], [.45, 1.01], color="0.65", linestyle="--", linewidth=1)
    for cohort, label in (("screen", "original 30 models"),
                          ("confirmation_history", "later 12 models")):
        subset = [m for m in models if m["cohort"] == cohort]
        ax.scatter([m["unseen_role_mean"] for m in subset],
                   [m["focal_role"] for m in subset], label=label,
                   color=colors[cohort], s=35, alpha=.85, edgecolor="white", linewidth=.35)
    ax.set(xlim=(.65, 1.01), ylim=(.45, 1.02),
           xlabel="Mean role accuracy over all 25 unseen pairs",
           ylabel="Role accuracy on selected pair (0,1)",
           title="Selected-pair scores can hide other failures")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.grid(alpha=.15)

    ax = axes[1]
    bins = np.arange(.45, 1.051, .05)
    ax.hist(unseen, bins=bins, color="#2764a5", edgecolor="white", linewidth=.7)
    ax.axvline(.8, color="#b34436", linestyle="--", linewidth=1.5,
               label="80% descriptive threshold")
    ax.set(xlim=(.45, 1.01), xlabel="Role accuracy for one model–pair case",
           ylabel="Number of model–pair cases",
           title="1,050 unseen model–pair cases")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=.15)
    fig.suptitle("Development audit of 42 saved Transformer models", fontsize=13)
    fig.text(.01, .005, "Pairs share graphs and models; counts are descriptive, not independent replications.",
             fontsize=8, color="0.35")
    fig.savefig(OUT, dpi=200, facecolor="white")
    print(OUT)


if __name__ == "__main__":
    main()
