"""Exploratory block-write analysis of already exposed, saved query states.

No model load, new inference, causal intervention, layer selection or gate change.
All eight families and both questions are retained. Rounding bounds concern the
saved float16 capture only; they are not statistical intervals.
"""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "trajectory_relation_audit_2026-10-17"
MODELS = ("qwen05", "qwen15", "qwen7b", "llama3b")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cos(a, b):
    denom = np.linalg.norm(a, axis=-1)*np.linalg.norm(b, axis=-1)
    out = np.full(denom.shape, np.nan)
    np.divide(np.sum(a*b, axis=-1), denom, out=out, where=denom > 1e-12)
    return np.clip(out, -1, 1)


def saved_rounding_bound(x):
    # Conservative half of the larger adjacent-float16 spacing, componentwise.
    up = np.nextafter(x, np.float16(np.inf)).astype(np.float64)
    down = np.nextafter(x, np.float16(-np.inf)).astype(np.float64)
    value = x.astype(np.float64)
    return 0.5*np.maximum(up-value, value-down)


def cosine_bounds(a, b, bound_a, bound_b):
    c = cos(a, b)
    norm_a, norm_b = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    alpha = bound_a/np.maximum(norm_a, 1e-30)
    beta = bound_b/np.maximum(norm_b, 1e-30)
    valid = (alpha < 1) & (beta < 1) & np.isfinite(c)
    theta = np.arccos(np.nan_to_num(c))
    error = np.arcsin(np.minimum(alpha, 1))+np.arcsin(np.minimum(beta, 1))
    lo = np.where(valid, np.cos(np.minimum(np.pi, theta+error)), -1)
    hi = np.where(valid, np.cos(np.maximum(0, theta-error)), 1)
    return lo, hi, alpha, beta


def safe_list(x):
    return np.where(np.isfinite(x), x, None).tolist()


def phase_summary(state_cos, write_cos, ua, ub, lo, hi, depth):
    summaries = {}
    for name, mask in (("early", depth <= 1/3),
                       ("middle", (depth > 1/3) & (depth <= 2/3)),
                       ("late", depth > 2/3)):
        a, b = ua[:, mask], ub[:, mask]
        denom = np.linalg.norm(a)*np.linalg.norm(b)
        summaries[name] = {
            "blocks_one_based": (np.where(mask)[0]+1).tolist(),
            "mean_state_cos": float(np.nanmean(state_cos[:, mask])),
            "mean_write_cos": float(np.nanmean(write_cos[:, mask])),
            "pooled_energy_weighted_write_cos": float(np.sum(a*b)/denom) if denom else None,
            "mean_write_cos_capture_rounding_bounds": [float(np.mean(lo[:, mask])), float(np.mean(hi[:, mask]))],
        }
    return summaries


def main():
    result, plot_data, ledger = {}, {}, []
    manifest = []
    for model in MODELS:
        path = SOURCE/f"QUERY_STATES_{model}.npz"
        meta = json.loads((SOURCE/f"QUERY_{model}.json").read_text())
        assert sha(path) == meta["states_sha256"]
        source = np.load(path)
        captured = source["states"]
        assert captured.dtype == np.float16
        x = captured.astype(np.float64)
        rounding = saved_rounding_bound(captured)
        a, b = x[:, :, 0]-x[:, :, 1], x[:, :, 2]-x[:, :, 3]
        # Axes: family, question, block/stage, token site, hidden coordinate.
        ua, ub = np.diff(a, axis=2), np.diff(b, axis=2)
        assert np.max(np.abs(a[:, :, 0]+ua.sum(axis=2)-a[:, :, -1])) < 1e-9
        assert np.max(np.abs(b[:, :, 0]+ub.sum(axis=2)-b[:, :, -1])) < 1e-9
        da_bound = rounding[:, :, 0]+rounding[:, :, 1]
        db_bound = rounding[:, :, 2]+rounding[:, :, 3]
        ea = np.linalg.norm(da_bound[:, :, 1:]+da_bound[:, :, :-1], axis=-1)
        eb = np.linalg.norm(db_bound[:, :, 1:]+db_bound[:, :, :-1], axis=-1)
        state_cos = cos(a[:, :, 1:], b[:, :, 1:])
        write_cos = cos(ua, ub)
        lo, hi, alpha, beta = cosine_bounds(ua, ub, ea, eb)
        common, flipped = (ua+ub)/2, (ua-ub)/2
        common_energy = np.sum(common**2, axis=-1)
        flipped_energy = np.sum(flipped**2, axis=-1)
        assert np.allclose(common_energy+flipped_energy,
                           (np.sum(ua**2, axis=-1)+np.sum(ub**2, axis=-1))/2)
        block_depth = np.arange(1, ua.shape[2]+1)/ua.shape[2]
        model_summary = {"source_sha256": sha(path), "blocks": ua.shape[2],
                         "cautions": ["Exposed cases; exploratory reanalysis.",
                                      "Whole block writes combine attention and MLP effects.",
                                      "Llama previously failed prefix invariance; remains technically flagged.",
                                      "Bound addresses saved float16 rounding, not weight quantization or execution errors."],
                         "by_question": {}}
        for qi, question in enumerate(source["questions"]):
            sites = {}
            for si, site in enumerate(source["sites"]):
                sc, wc = state_cos[:, qi, :, si], write_cos[:, qi, :, si]
                sites[str(site)] = phase_summary(sc, wc, ua[:, qi, :, si], ub[:, qi, :, si],
                                                lo[:, qi, :, si], hi[:, qi, :, si], block_depth)
                for fi, family in enumerate(source["family_ids"]):
                    for li in range(len(block_depth)):
                        total = common_energy[fi, qi, li, si]+flipped_energy[fi, qi, li, si]
                        ledger.append({"model": model, "family": str(family), "question": str(question),
                                       "site": str(site), "block": li+1,
                                       "state_cos": float(sc[fi, li]), "write_cos": float(wc[fi, li]),
                                       "write_cos_lower_capture_rounding": float(lo[fi, qi, li, si]),
                                       "write_cos_upper_capture_rounding": float(hi[fi, qi, li, si]),
                                       "active_write_norm": float(np.linalg.norm(ua[fi, qi, li, si])),
                                       "passive_write_norm": float(np.linalg.norm(ub[fi, qi, li, si])),
                                       "active_relative_rounding_bound": float(alpha[fi, qi, li, si]),
                                       "passive_relative_rounding_bound": float(beta[fi, qi, li, si]),
                                       "shared_sign_write_energy_fraction": float(common_energy[fi, qi, li, si]/total) if total else None})
            model_summary["by_question"][str(question)] = sites
        result[model] = model_summary
        plot_data[model] = {"depth": block_depth,
                            "state": np.nanmean(state_cos[:, :, :, 1], axis=0),
                            "write": np.nanmean(write_cos[:, :, :, 1], axis=0),
                            "lower": np.mean(lo[:, :, :, 1], axis=0),
                            "upper": np.mean(hi[:, :, :, 1], axis=0)}
        manifest.append({"path": str(path.relative_to(HERE.parent.parent)), "sha256": sha(path)})
    output = HERE/"LAYER_WRITE_RESULTS.json"
    assert not output.exists(), "refusing overwrite"
    output.write_text(json.dumps({"status": "exploratory_saved_state_block_write_audit",
                                  "date": "2026-10-18", "script_sha256": sha(__file__),
                                  "sources": manifest, "models": result}, indent=2)+"\n")
    with (HERE/"LAYER_WRITE_LEDGER.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ledger[0]))
        writer.writeheader()
        writer.writerows(ledger)
    labels = {"qwen05": "Qwen 0.5B · diagnostic", "qwen15": "Qwen 1.5B · competence limited",
              "qwen7b": "Qwen 7B", "llama3b": "Llama 3B · prefix control failed"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    for ax, model in zip(axes.flat, MODELS):
        curves = plot_data[model]
        for qi, (question, color) in enumerate((("Action question", "#3B6FB6"), ("First-person question", "#D25B32"))):
            ax.plot(curves["depth"], curves["state"][qi], "--", color=color, alpha=0.55, lw=1.4,
                    label=question+": stored state")
            ax.plot(curves["depth"], curves["write"][qi], color=color, lw=2,
                    label=question+": block write")
            ax.fill_between(curves["depth"], curves["lower"][qi], curves["upper"][qi], color=color, alpha=0.10)
        ax.set_title(labels[model], loc="left", fontsize=11)
        ax.axhline(0, color="#777", lw=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.13)
        ax.set_xlim(0, 1)
        ax.set_ylim(-1.05, 1.05)
    for ax in axes[:, 0]:
        ax.set_ylabel("Active/passive response cosine")
    for ax in axes[-1]:
        ax.set_xlabel("Normalized block depth")
    handles, legends = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legends, loc="upper center", bbox_to_anchor=(0.5, 0.934), ncol=2,
               frameon=False, fontsize=9)
    fig.suptitle("What a layer changes can differ from what the residual stream stores", weight="bold", fontsize=15, y=0.99)
    fig.text(0.5, 0.947, "All eight exposed families retained · whole-block differences · no new model inference", ha="center", fontsize=10, color="#555")
    fig.text(0.5, 0.012, "Bands bound saved float16 rounding only; they are not confidence intervals. Write alignment does not establish causal use or semantic roles.",
             ha="center", fontsize=8.5, color="#555")
    fig.tight_layout(rect=(0, 0.04, 1, 0.85))
    for ext in ("png", "pdf"):
        fig.savefig(HERE/f"LAYER_WRITES.{ext}", dpi=200, facecolor="white")
    plt.close(fig)
    print(json.dumps({m: d["by_question"]["agent"]["answer_prompt_end"] for m, d in result.items()}, indent=2))
    print("Recorded", len(ledger), "family/question/block/site rows")


if __name__ == "__main__":
    main()
