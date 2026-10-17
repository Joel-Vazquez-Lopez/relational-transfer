"""Exact calibration: a surface-keyed answer copier mimics response alignment.

This is a hand-defined program, not a learned model or fit to the LLM curves.
The latent state contains first-mentioned identity plus a query-specific answer
looked up from (voice, first identity, question). It has no stored role variable
before the query. It is not evidence that an LLM uses this algorithm.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent


def cosine(a, b):
    return np.sum(a*b, axis=-1)/(np.linalg.norm(a, axis=-1)*np.linalg.norm(b, axis=-1))


def main():
    x = np.linspace(0, 1, 29)
    beta = 0.05 + 4/(1+np.exp(-20*(x-0.7)))
    epsilon = 0.1
    common = np.array([2., 0., 0., 0.])
    first_direction = np.array([0., 1., 0., 0.])
    output_axis = np.array([0., 0., 1., 0.])
    answer_direction = epsilon*output_axis + np.sqrt(1-epsilon**2)*np.array([0., 0., 0., 1.])
    # Full table is the algorithm; no role variable is carried between stages.
    lookup = {(voice, first, query): (first if query == "first" or voice == "active" else -first)
              for voice in ("active", "passive") for first in (-1, 1) for query in ("agent", "first")}
    data = {}
    for query in ("agent", "first"):
        states = {}
        answers = []
        for voice in ("active", "passive"):
            for true_agent in (1, -1):
                first = true_agent if voice == "active" else -true_agent
                answer = lookup[voice, first, query]
                state = common + first*first_direction + beta[:, None]*answer*answer_direction
                states[voice, true_agent] = state
                assert int(np.sign(state[-1]@output_axis)) == answer
                answers.append({"voice": voice, "A_is_agent": true_agent == 1,
                                "answer": "A" if answer == 1 else "B"})
        a = states["active", 1]-states["active", -1]
        b = states["passive", 1]-states["passive", -1]
        ap = a-(a@output_axis)[:, None]*output_axis
        bp = b-(b@output_axis)[:, None]*output_axis
        norms = [np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)]
        assert np.allclose(norms[0], norms[1])
        data[query] = {"cosine": cosine(a, b).tolist(),
                       "cosine_after_direct_output_axis_removal": cosine(ap, bp).tolist(),
                       "response_norms": [n.tolist() for n in norms],
                       "answers": answers}
    # Both query tasks are correct, yet their relation cosines have opposite endpoints.
    assert data["agent"]["cosine"][0] < -0.99
    assert data["agent"]["cosine"][-1] > 0.88
    assert data["agent"]["cosine_after_direct_output_axis_removal"][-1] > 0.88
    assert np.allclose(data["first"]["cosine"], -1)
    assert np.allclose(data["agent"]["response_norms"], data["first"]["response_norms"])
    output = {"status": "analytical_counterexample_not_llm_evidence", "epsilon": epsilon,
              "query_tasks": data,
              "interpretation": "The same norm curves, a negative-to-positive cross-voice angle change, and survival after direct-output-axis removal can all arise from a voice-and-order lookup followed by distributed answer coding. This does not establish a reusable query-independent semantic role representation."}
    path = HERE / "COPYING_COUNTEREXAMPLE.json"
    assert not path.exists(), "refusing overwrite"
    path.write_text(json.dumps(output, indent=2)+"\n")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9))
    axes[0].plot(x, data["agent"]["response_norms"][0], color="#3B6FB6", lw=4, label="Agent query")
    axes[0].plot(x, data["first"]["response_norms"][0], "--", color="#D25B32", lw=2, label="First-name query")
    axes[0].set_title("Identical response-size lines")
    axes[0].set_ylabel("Response norm")
    axes[1].plot(x, data["agent"]["cosine"], color="#3B6FB6", lw=2, label="Agent query")
    axes[1].plot(x, data["first"]["cosine"], "--", color="#D25B32", lw=2, label="First-name query")
    axes[1].set_title("Different relations between responses")
    axes[1].set_ylabel("Active/passive response cosine")
    axes[2].plot(x, data["agent"]["cosine"], color="#3B6FB6", lw=4, label="Before removal")
    axes[2].plot(x, data["agent"]["cosine_after_direct_output_axis_removal"], "--", color="#23846A", lw=2, label="Output axis removed")
    axes[2].set_title("The copying explanation survives")
    axes[2].set_ylabel("Agent-query response cosine")
    for i, ax in enumerate(axes):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.15)
        ax.set_xlabel("Ordered computational stage")
        ax.legend(frameon=False, fontsize=8)
        if i:
            ax.set_ylim(-1.1, 1.1)
            ax.axhline(0, color="#777", lw=0.7)
    fig.suptitle("Calibration: a simple answer lookup can imitate an encouraging path relation", weight="bold", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    for ext in ("png", "pdf"):
        target = HERE / f"COPYING_COUNTEREXAMPLE.{ext}"
        assert not target.exists(), "refusing overwrite"
        fig.savefig(target, dpi=200, facecolor="white")
    print(json.dumps({q: {"first_cos": v["cosine"][0], "final_cos": v["cosine"][-1],
                          "final_after_axis_removal": v["cosine_after_direct_output_axis_removal"][-1]}
                      for q, v in data.items()}, indent=2))


if __name__ == "__main__":
    main()
