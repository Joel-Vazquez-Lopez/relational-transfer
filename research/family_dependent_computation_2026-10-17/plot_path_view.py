"""Plot full-dimensional descriptive role-reversal separation for f07."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
LOCK = json.loads((HERE / "PATH_VIEW_LOCK.json").read_text())
COLORS = {"qwen05": "#9A9A9A", "qwen15": "#3B6FB6",
          "qwen7b": "#D25B32", "llama3b": "#23846A"}
LABELS = {"qwen05": "Qwen 0.5B · failed screen", "qwen15": "Qwen 1.5B",
          "qwen7b": "Qwen 7B", "llama3b": "Llama 3B"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    data = {}
    for key in LOCK["models"]:
        raw = HERE / f"PATH_VIEW_{key}.npz"
        meta = json.loads((HERE / f"PATH_VIEW_{key}.json").read_text())
        assert meta["states_sha256"] == sha(raw)
        assert meta["path_view_lock_sha256"] == sha(HERE / "PATH_VIEW_LOCK.json")
        loaded = np.load(raw)
        arr = loaded["states"].astype(np.float32)
        assert list(loaded["conditions"]) == LOCK["conditions"]
        assert list(loaded["sites"]) == LOCK["sites"]
        assert arr.shape[0] == 4 and arr.shape[2] == 2
        curves = {}
        for voice, indexes in (("active", (0, 1)), ("passive", (2, 3))):
            a, b = arr[list(indexes)]
            delta = np.linalg.norm(a-b, axis=-1)
            scale = (np.linalg.norm(a, axis=-1)+np.linalg.norm(b, axis=-1))/2
            curves[voice] = (delta/np.maximum(scale, 1e-8)).tolist()
        data[key] = {"stages": arr.shape[1], "hidden_dim": arr.shape[-1],
                     "curves": curves,
                     "endpoints": {voice: {site: round(curves[voice][-1][si], 5)
                                          for si, site in enumerate(LOCK["sites"])}
                                   for voice in ("active", "passive")}}

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), sharex=True, sharey="col")
    for ri, voice in enumerate(("active", "passive")):
        for ci, site in enumerate(LOCK["sites"]):
            ax = axes[ri, ci]
            for key in LOCK["models"]:
                y = np.array(data[key]["curves"][voice])[:, ci]
                x = np.linspace(0, 1, len(y))
                ax.plot(x, y, color=COLORS[key], linewidth=2.2 if key != "qwen05" else 1.6,
                        linestyle="--" if key == "qwen05" else "-", label=LABELS[key])
            ax.set_title(("Active" if voice == "active" else "Passive") + " · " +
                         ("event end" if site == "event_end" else "answer prompt end"),
                         fontsize=12, loc="left")
            ax.grid(alpha=0.18)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_xlim(0, 1)
            if ri == 1:
                ax.set_xlabel("Normalized layer depth  (embedding → final block)")
            if ci == 0:
                ax.set_ylabel("Relative state separation")
    for ci, site in enumerate(LOCK["sites"]):
        max_value = max(max(np.array(data[key]["curves"][voice])[:, ci])
                        for key in LOCK["models"] for voice in ("active", "passive"))
        axes[0, ci].set_ylim(0, max_value * 1.08)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=4,
               frameon=False, fontsize=10)
    fig.suptitle("How the role-swap difference develops across layers", x=0.5, y=0.99,
                 fontsize=17, fontweight="bold")
    fig.text(0.5, 0.945, "One selected family: Miriam / Adrian, ‘pushed’ · full-dimensional residual states",
             ha="center", fontsize=10, color="#555555")
    fig.text(0.5, 0.012,
             "Descriptive only: lexical exchange and tokenization remain confounded; depth is not computation time.",
             ha="center", fontsize=9, color="#666666")
    fig.tight_layout(rect=(0, 0.035, 1, 0.86), h_pad=1.5, w_pad=1.4)
    png = HERE / "PATH_VIEW_LINES.png"
    pdf = HERE / "PATH_VIEW_LINES.pdf"
    assert not png.exists() and not pdf.exists(), "refusing overwrite"
    fig.savefig(png, dpi=220, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    out = HERE / "PATH_VIEW_METRICS.json"
    assert not out.exists(), "refusing overwrite"
    out.write_text(json.dumps({"status": "descriptive_lines_only",
                               "lock_sha256": sha(HERE / "PATH_VIEW_LOCK.json"),
                               "script_sha256": sha(Path(__file__)), "models": data,
                               "png_sha256": sha(png), "pdf_sha256": sha(pdf)}, indent=2) + "\n")
    print(json.dumps({key: val["endpoints"] for key, val in data.items()}, indent=2))


if __name__ == "__main__":
    main()
