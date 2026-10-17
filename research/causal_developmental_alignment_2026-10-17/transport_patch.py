"""Frozen native sentence-event-state transport patching. One model and split/run.

Usage: python transport_patch.py qwen7b|llama3b development|evaluation
Do not invoke evaluation unless TRANSPORT_DEV_DECISION.json says pass.
"""
import hashlib
import json
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load

from factorial_gate import make_prompt
from role_path_pilot import MODELS
from transport_gate import LOCK, build

HERE = Path(__file__).parent
LAYERS = tuple(LOCK["layers_zero_indexed"])
PRIMARY = "period"
WRONG = "prefix"
FACTORS = LOCK["seven_conditions"][:4]
DONORS = FACTORS + ["same_answer_unrelated"]
SELF_PATCH_TOLERANCE_NATS = 0.05  # Technical assertion frozen before patches.


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def indices(tok, text, sentence):
    marker = "Sentence: " + sentence
    start = text.index(marker)
    period_char = start + len(marker) - 1
    prefix_char = start + len("Sentence")
    encoded = tok._tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    ids = tok.encode(text, add_special_tokens=False)
    assert encoded["input_ids"] == ids
    offsets = encoded["offset_mapping"]
    def covering(char):
        hit = [i for i, (s, e) in enumerate(offsets) if s <= char < e]
        assert len(hit) == 1, (char, hit)
        return hit[0]
    return ids, {PRIMARY: covering(period_char), WRONG: covering(prefix_char)}


class Probe(nn.Module):
    def __init__(self, original, donor=None, target=None):
        super().__init__()
        self.original = original
        self.donor = donor
        self.target = target
        self.captured = None
        self.use_sliding = getattr(original, "use_sliding", False)

    def __call__(self, x, mask=None, cache=None):
        y = self.original(x, mask, cache)
        self.captured = y
        if self.donor is not None:
            vec = self.donor.reshape(1, 1, -1)
            y = mx.concatenate([y[:, :self.target, :], vec,
                                y[:, self.target + 1:, :]], axis=1)
        return y


def forward(model, ids, patch=None, capture=False):
    blocks = model.model.layers
    chosen = range(len(blocks)) if capture else ([patch[0]] if patch else [])
    wrapped = []
    for li in chosen:
        old = blocks[li]
        probe = Probe(old, patch[2] if patch and li == patch[0] else None,
                      patch[1] if patch and li == patch[0] else None)
        blocks[li] = probe
        wrapped.append((li, old, probe))
    try:
        logits = model(mx.array([ids]))[0]
        mx.eval(logits)
        states = []
        if capture:
            for _, _, probe in wrapped:
                mx.eval(probe.captured)
                states.append(probe.captured)
        return logits, states
    finally:
        for li, old, _ in wrapped:
            blocks[li] = old


def capture_states(model, ids, sites):
    _, states = forward(model, ids, capture=True)
    emb = model.model.embed_tokens(mx.array([ids]))
    mx.eval(emb)
    stages = [emb] + states
    arr = np.empty((len(stages), 2, int(emb.shape[-1])), dtype=np.float16)
    live = {}
    for li, state in enumerate(stages):
        for si, site in enumerate((PRIMARY, WRONG)):
            vec = state[0, sites[site], :]
            mx.eval(vec)
            live[(li, site)] = vec  # Preserve native dtype for interventions.
            arr[li, si] = np.asarray(vec).astype(np.float16)
    return arr, live


def margin(model, tok, ids, a, b, patch=None):
    scores = {}
    for name in (a, b):
        continuation = tok.encode(name, add_special_tokens=False)
        logits, _ = forward(model, ids + continuation, patch=patch)
        values = [logits[len(ids)-1+j, token] - mx.logsumexp(logits[len(ids)-1+j])
                  for j, token in enumerate(continuation)]
        score = mx.sum(mx.stack(values))
        mx.eval(score)
        scores[name] = float(score.item())
    return scores[b] - scores[a], scores


def family_run(model, tok, fam, gate_rows):
    prompts = {}
    states = {}
    live_states = {}
    metadata = {}
    for condition in LOCK["seven_conditions"]:
        sentence, question, expected = build(fam, condition)
        p = make_prompt(tok, sentence, question)
        ids, sites = indices(tok, p, sentence)
        prompts[condition] = ids
        states[condition], live_states[condition] = capture_states(model, ids, sites)
        metadata[condition] = {"prompt": p, "sentence": sentence, "expected": expected,
                               "token_count": len(ids), "sites": sites}
        mx.clear_cache()
    target = "receiver_cleft_a_agent"
    ids = prompts[target]
    site = metadata[target]["sites"]
    base_row = gate_rows[target]
    base_margin = base_row["full_name_logprob"][fam["b"]] - base_row["full_name_logprob"][fam["a"]]
    reverse_row = gate_rows["receiver_cleft_b_agent"]
    reverse_margin = reverse_row["full_name_logprob"][fam["b"]] - reverse_row["full_name_logprob"][fam["a"]]
    rows = []
    for layer in LAYERS:
        self_donor = live_states[target][(layer+1, PRIMARY)]
        self_margin, self_scores = margin(model, tok, ids, fam["a"], fam["b"],
                                          patch=(layer, site[PRIMARY], self_donor))
        assert abs(self_margin - base_margin) <= SELF_PATCH_TOLERANCE_NATS, (
            "self patch discrepancy", fam["id"], layer, self_margin, base_margin)
        layer_rows = [{"layer": layer, "donor": target, "site": PRIMARY,
                       "margin_b_minus_a": self_margin, "scores": self_scores}]
        for condition in DONORS:
            for patch_site in ((PRIMARY, WRONG) if condition in FACTORS else (PRIMARY,)):
                donor = live_states[condition][(layer+1, patch_site)]
                value, scores = margin(model, tok, ids, fam["a"], fam["b"],
                                       patch=(layer, site[patch_site], donor))
                layer_rows.append({"layer": layer, "donor": condition, "site": patch_site,
                                   "margin_b_minus_a": value, "scores": scores})
        rows.extend(layer_rows)
        print(fam["id"], "layer", layer, "done", flush=True)
        mx.clear_cache()
    labels = list(LOCK["seven_conditions"])
    vectors = np.stack([states[k] for k in labels])
    return {"family": fam, "recipient_baseline_margin_b_minus_a": base_margin,
            "reversed_recipient_margin_b_minus_a": reverse_margin,
            "metadata": metadata, "patches": rows}, vectors, labels


def main(key, split):
    assert key in LOCK["models"] and split in ("development", "evaluation")
    if split == "evaluation":
        decision = json.loads((HERE / "TRANSPORT_DEV_DECISION.json").read_text())
        assert decision["pass"], "development gate failed; evaluation patches forbidden"
    out = HERE / f"TRANSPORT_PATCH_{key}_{split}.json"
    raw = HERE / f"TRANSPORT_STATES_{key}_{split}.npz"
    assert not out.exists() and not raw.exists(), "refusing overwrite"
    gate_path = HERE / f"TRANSPORT_GATE_{key}.json"
    gate = json.loads(gate_path.read_text())
    assert gate["lock_sha256"] == sha(HERE / "TRANSPORT_LOCK.json")
    eligible = [f for f in gate["families"] if f["family"]["split"] == split and f["competent"]]
    assert eligible, "no competent family in split"
    model_path = MODELS[key]
    model, tok = load(str(model_path), tokenizer_config={"trust_remote_code": False})
    model.eval()
    mx.set_cache_limit(256 * 1024 * 1024)
    results, arrays, labels = [], [], None
    for f in eligible:
        r, v, labels = family_run(model, tok, f["family"], {row["condition"]: row for row in f["rows"]})
        results.append(r)
        arrays.append(v)
        print(key, split, f["family"]["id"], "complete", flush=True)
    np.savez_compressed(raw, states=np.stack(arrays),
                        family_ids=np.array([r["family"]["id"] for r in results]),
                        conditions=np.array(labels), sites=np.array([PRIMARY, WRONG]))
    record = {"status": "native_prequestion_event_state_patching", "model": key,
              "split": split, "lock_sha256": sha(HERE / "TRANSPORT_LOCK.json"),
              "script_sha256": sha(__file__), "gate_sha256": sha(gate_path),
              "model_config_sha256": sha(model_path / "config.json"),
              "state_shape": "[competent_family, seven_conditions, embedding_plus_28_blocks, period_and_prefix, hidden_width]",
              "states_sha256": sha(raw), "results": results}
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(out, raw, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
