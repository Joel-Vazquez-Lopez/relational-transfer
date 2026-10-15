"""Standalone descriptive figures for the Qwen role evaluation."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
one = json.loads((ROOT / "SUMMARY.json").read_text())
small = json.loads((ROOT / "SCALE_SUMMARY.json").read_text())
internal = json.loads((ROOT / "INTERNAL_SUMMARY.json").read_text())

labels = ("Active\ngeneric", "Active\nperformed", "Passive\ngeneric", "Passive\nperformed")
keys = (("active", "generic"), ("active", "performed"),
        ("passive", "generic"), ("passive", "performed"))
large_values = [sum(one["cells"][f"set{s}/{form}/{wording}"]["correct"] for s in range(3)) / 120
                for form, wording in keys]
small_values = [small["cells"][f"{form}/{wording}"]["correct"] / 120
                for form, wording in keys]

plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 160})
fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.1), constrained_layout=True)
x = np.arange(4)
width = .36
axes[0].bar(x-width/2, large_values, width, label="Qwen2.5-1.5B", color="#2459a6")
axes[0].bar(x+width/2, small_values, width, label="Qwen2.5-0.5B", color="#b76829")
axes[0].set_xticks(x, labels)
axes[0].set_ylim(0, 1.12)
axes[0].set_ylabel("Correct actor answers / 120 events")
axes[0].set_title("Same events, different model responses")
axes[0].legend(frameon=False, loc="upper left")
for i, values in enumerate((large_values, small_values)):
    for j, value in enumerate(values):
        axes[0].text(j + (-width/2 if i == 0 else width/2), value+.025,
                     f"{round(value*120)}/120", ha="center", va="bottom", fontsize=8)

layers = [0, 4, 8, 12, 16, 20, 24, 28]
colors = {"active/generic": "#b03436", "active/performed": "#2459a6",
          "passive/generic": "#3b8b60", "passive/performed": "#8055a5"}
for key, color in colors.items():
    values = [internal["cells"][key][str(layer)]["actor_positive"] / 120
              for layer in layers]
    axes[1].plot(layers, values, marker="o", lw=1.9, ms=4, color=color,
                 label=key.replace("/", ", "))
axes[1].axhline(.5, color="0.65", lw=1, ls="--")
axes[1].set_xticks(layers)
axes[1].set_ylim(0, 1.05)
axes[1].set_xlabel("Transformer block completed")
axes[1].set_ylabel("Actor-favoring candidate readout")
axes[1].set_title("Fixed intermediate readout, Qwen2.5-1.5B")
axes[1].legend(frameon=False, fontsize=8, loc="lower left")

fig.suptitle("Role answers vary with wording, sentence form, and model", fontsize=12)
fig.savefig(ROOT / "RQ3_OVERVIEW.png", bbox_inches="tight")
fig.savefig(ROOT / "RQ3_OVERVIEW.pdf", bbox_inches="tight")
print(ROOT / "RQ3_OVERVIEW.png")
