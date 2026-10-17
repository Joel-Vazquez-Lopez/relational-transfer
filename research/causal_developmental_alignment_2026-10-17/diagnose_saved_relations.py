"""Diagnostic reanalysis of saved, exposed role-swap states. No inference.

Quantifies how much step-Gram similarity survives excluding nearby layers,
conditioning on a generic raw-path template, and comparing labeled token sites.
All analyses are exploratory and use the same four previously seen examples.
"""
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
MODELS = ("qwen7b", "llama3b")
SITES = ("subject", "verb", "object", "final")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def corr(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / den) if den > 1e-12 else None


def unit_steps(path):
    v = np.diff(path.astype(np.float64), axis=0)
    n = np.linalg.norm(v, axis=-1)
    return v / np.maximum(n[:, None], 1e-12), n


def mean_pairwise(rows):
    values = [corr(rows[i], rows[j]) for i, j in itertools.combinations(range(len(rows)), 2)]
    return float(np.mean([v for v in values if v is not None]))


def describe_grams(role, unrelated, raw):
    n = role.shape[-1]
    lag = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    combined = np.concatenate([role, unrelated], axis=0)
    out = {}
    for minimum_lag in (1, 2, 3, 5, 8, 12):
        mask = np.triu(lag >= minimum_lag, 1)
        rr = [g[mask] for g in role]
        uu = [g[mask] for g in unrelated]
        ru = [corr(a, b) for a in rr for b in uu]
        out[str(minimum_lag)] = {
            "n_entries": int(mask.sum()),
            "role_within": mean_pairwise(rr),
            "unrelated_within": mean_pairwise(uu),
            "role_unrelated": float(np.mean(ru)),
        }
    # The raw-prompt template is separate from all contrast curves. This is
    # descriptive conditioning, not a fitted held-out correction.
    raw_template = raw.mean(axis=0)
    mask = np.triu(np.ones((n, n), dtype=bool), 1)
    r_resid = [(g - raw_template)[mask] for g in role]
    u_resid = [(g - raw_template)[mask] for g in unrelated]
    out["raw_template_residual"] = {
        "role_to_raw_template_corr": float(np.mean([corr(g[mask], raw_template[mask]) for g in role])),
        "unrelated_to_raw_template_corr": float(np.mean([corr(g[mask], raw_template[mask]) for g in unrelated])),
        "role_within": mean_pairwise(r_resid),
        "unrelated_within": mean_pairwise(u_resid),
        "role_unrelated": float(np.mean([corr(a, b) for a in r_resid for b in u_resid])),
        "role_residual_to_raw_ratio": float(np.linalg.norm(role - raw_template, axis=(1, 2)).mean() /
                                             np.linalg.norm(role, axis=(1, 2)).mean()),
        "unrelated_residual_to_raw_ratio": float(np.linalg.norm(unrelated - raw_template, axis=(1, 2)).mean() /
                                                  np.linalg.norm(unrelated, axis=(1, 2)).mean()),
    }
    # Test whether the diagonal/lag pattern alone captures pairwise Gram
    # entries. A pooled template would overfit; lag has only n-1 parameters.
    pooled = combined.mean(axis=0)
    lag_template = np.zeros_like(pooled)
    for k in range(1, n):
        lag_template[lag == k] = pooled[lag == k].mean()
    lag_template[np.diag_indices(n)] = 1
    target = combined[:, mask]
    predictor = np.broadcast_to(lag_template[mask], target.shape)
    baseline = float(np.mean(target))
    out["pooled_lag_template"] = {
        "pooled_r2_offdiagonal": float(1 - np.sum((target - predictor) ** 2) /
                                         np.sum((target - baseline) ** 2)),
        "lag_1_mean": float(pooled[lag == 1].mean()),
        "lag_2_mean": float(pooled[lag == 2].mean()),
        "lag_5_mean": float(pooled[lag == 5].mean()),
        "lag_10_mean": float(pooled[lag == 10].mean()),
    }
    return out


def cross_site_summary(curves):
    # Cross-site relations are unavailable to a single-site step Gram.
    # Same-layer step cosine is the diagonal of the cross-site Gram.
    out = {}
    for left, right in ((0, 2), (0, 1), (1, 3)):
        key = f"{SITES[left]}_{SITES[right]}"
        curves_cos = []
        for z in curves:
            u, _ = unit_steps(z[:, left, :])
            v, _ = unit_steps(z[:, right, :])
            curves_cos.append(np.sum(u * v, axis=-1))
        c = np.stack(curves_cos)
        out[key] = {
            "mean_first_seven_steps": float(c[:, :7].mean()),
            "mean_middle_seven_steps": float(c[:, 10:17].mean()),
            "mean_last_seven_steps": float(c[:, -7:].mean()),
            "mean_by_step": c.mean(axis=0).round(5).tolist(),
            "within_case_curves": c.round(5).tolist(),
        }
    return out


def exact_match_permutation(a, b):
    n = len(a)
    sim = np.array([[corr(x, y) for y in b] for x in a])
    observed = float(np.diag(sim).mean())
    null = np.array([np.mean([sim[i, p[i]] for i in range(n)])
                     for p in itertools.permutations(range(n))])
    return {"matched_mean": observed,
            "mismatched_mean": float((sim.sum() - np.trace(sim)) / (n * (n - 1))),
            "exact_permutation_upper_tail_fraction": float(np.mean(null >= observed - 1e-12)),
            "permutations": len(null), "similarity_matrix": sim.round(5).tolist()}


def main():
    arrays = {m: np.load(HERE / f"PATH_STATES_{m}.npz")["states"].astype(np.float32)
              for m in MODELS}
    grams = {}
    diagnostics = {}
    for m, x in arrays.items():
        role = x[:, 0] - x[:, 1]
        unrelated = np.stack([x[i, 0] - x[(i + 1) % 4, 0] for i in range(4)])
        raw = x[:, 0]
        diagnostics[m] = {}
        grams[m] = {}
        for si, site in enumerate(SITES):
            gs = {}
            for cat, curves in (("role", role), ("unrelated", unrelated), ("raw", raw)):
                gs[cat] = np.stack([unit_steps(z[:, si, :])[0] @ unit_steps(z[:, si, :])[0].T
                                    for z in curves])
            diagnostics[m][site] = describe_grams(gs["role"], gs["unrelated"], gs["raw"])
            grams[m][site] = gs
        diagnostics[m]["cross_site"] = {
            "role": cross_site_summary(role),
            "unrelated": cross_site_summary(unrelated),
            "raw": cross_site_summary(raw),
        }
    matched = {}
    for site in SITES:
        mask = np.triu(np.ones((28, 28), dtype=bool), 1)
        matched[site] = {}
        for cat in ("role", "unrelated"):
            a = grams["qwen7b"][site][cat][:, mask]
            b = grams["llama3b"][site][cat][:, mask]
            matched[site][cat] = exact_match_permutation(a, b)
    result = {
        "status": "exploratory_reanalysis_of_exposed_four_cases_no_model_inference",
        "source_sha256": {m: sha(HERE / f"PATH_STATES_{m}.npz") for m in MODELS},
        "script_sha256": sha(__file__),
        "definitions": {
            "role": "base minus reversal in same lexical family",
            "unrelated": "base family i minus base family i+1 modulo four",
            "raw": "unmodified base prompt trajectory",
            "gram": "cosine between adjacent-layer contrast-state updates",
            "lag_mask": "off-diagonal Gram entries with |step_i-step_j| at least given lag",
            "raw_template_residual": "subtract mean raw-prompt Gram within model and token site; descriptive only",
            "cross_site": "same-layer cosine between steps at labeled token sites",
        },
        "diagnostics": diagnostics,
        "cross_model_exact_case_match": matched,
        "limitations": [
            "Four exposed examples and no independent voice or other task contrast.",
            "Unrelated control has different lexical content; cross-site patterns can be lexical.",
            "Raw-template subtraction can remove real structure and is not validated on holdouts.",
            "Permutation fractions are descriptive with only 24 mappings, not population p-values.",
        ],
    }
    out = HERE / "SAVED_RELATION_DIAGNOSTICS.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(out)
    for m in MODELS:
        z = diagnostics[m]["final"]
        print(m, "final", {k: z[k] for k in ("1", "3", "8", "raw_template_residual", "pooled_lag_template")})
        for cat in ("role", "unrelated"):
            print(m, cat, "subject_object", diagnostics[m]["cross_site"][cat]["subject_object"]
                  ["mean_first_seven_steps"])
    print("cross-model", {site: {cat: matched[site][cat]["exact_permutation_upper_tail_fraction"]
                                  for cat in ("role", "unrelated")} for site in SITES})


if __name__ == "__main__":
    main()
