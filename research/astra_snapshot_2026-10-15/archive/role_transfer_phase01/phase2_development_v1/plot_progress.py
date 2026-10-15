"""Plot run progress and the user-requested, one-family provisional peek."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "PHASE2_PROGRESS.png"


def main():
    counts = np.zeros((8, 2), dtype=int)
    for family in range(8):
        for j, condition in enumerate(("sham", "high")):
            counts[family, j] = len(list((HERE / "full" / f"family{family}" / condition).rglob("metrics.json")))
    completed = int(counts.sum())

    values = {}
    for arm in ("participant_only", "ordered_roles"):
        values[arm] = {}
        for condition in ("sham", "high"):
            scores = []
            for seed in (321001, 321002):
                file = HERE / "full" / "family0" / condition / f"seed_{seed}" / arm / "checkpoint_6000.json"
                if file.exists():
                    data = json.loads(file.read_text())
                    scores.append(data["evaluation"]["260107"]["focal"]["binary_role_accuracy"] * 100)
            values[arm][condition] = scores

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(11.2, 5.2), facecolor="#fbfbfa")
    gs = fig.add_gridspec(1, 2, width_ratios=[0.88, 1.12], left=.07, right=.96, top=.81, bottom=.17, wspace=.29)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    fig.text(.07, .94, "Role-transfer development run", fontsize=20, weight="bold", color="#1b263b")
    fig.text(.07, .88, f"{completed} of 160 models complete ({completed/160:.0%})  •  One-family scores are provisional", fontsize=11.5, color="#4a5568")

    ax0.imshow(counts / 10, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax0.set_xticks([0, 1], ["Sham", "High rewire"])
    ax0.set_yticks(range(8), [f"Family {i+1}" for i in range(8)])
    ax0.set_title("Model completion by graph condition", loc="left", pad=15, fontsize=12, weight="bold")
    ax0.tick_params(length=0)
    for i in range(8):
        for j in range(2):
            ax0.text(j, i, f"{counts[i,j]}/10", ha="center", va="center",
                     color="white" if counts[i,j] >= 6 else "#17365d", weight="bold", fontsize=10)
    for edge in np.arange(-.5, 8, 1): ax0.axhline(edge, color="white", lw=2)
    ax0.axvline(.5, color="white", lw=2)

    colors = {"participant_only": "#d55e00", "ordered_roles": "#0072b2"}
    labels = {"participant_only": "Participant-only", "ordered_roles": "Ordered roles"}
    for arm in ("participant_only", "ordered_roles"):
        sham, high = values[arm]["sham"], values[arm]["high"]
        if len(sham) == 2 and len(high) == 2:
            ax1.plot([0, 1], [np.mean(sham), np.mean(high)], marker="o", lw=2.8,
                     ms=8, color=colors[arm], label=labels[arm], zorder=4)
            for k in range(2):
                ax1.plot([0, 1], [sham[k], high[k]], color=colors[arm], alpha=.22, lw=1.2, zorder=2)
                ax1.scatter([0, 1], [sham[k], high[k]], color=colors[arm], alpha=.48, s=22, zorder=3)
    ax1.set_xticks([0, 1], ["Sham graph", "High rewire graph"])
    ax1.set_xlim(-.18, 1.18)
    ax1.set_ylim(50, 103)
    ax1.set_ylabel("Correct actor/patient choice (%)")
    ax1.set_title("Family 1 only: withheld pair", loc="left", pad=15, fontsize=12, weight="bold")
    ax1.grid(axis="y", alpha=.15)
    ax1.legend(frameon=False, loc="lower left", fontsize=10)
    ax1.text(.98, .03, "Large dots: mean of 2 starts\nSmall dots: individual starts", transform=ax1.transAxes,
             fontsize=9.5, color="#4a5568", va="bottom", ha="right")
    fig.text(.07, .065,
             "Interim development view. Other graph families, performance matching, and ceiling checks are pending.",
             fontsize=10, color="#4a5568")
    fig.savefig(OUT, dpi=180, facecolor=fig.get_facecolor())
    print(OUT)


if __name__ == "__main__":
    main()
