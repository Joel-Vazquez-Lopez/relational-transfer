"""Frozen question-conditioned response diagnostic; no causal interventions."""
import hashlib
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "causal_developmental_alignment_2026-10-17"
sys.path.insert(0, str(SOURCE))
from factorial_gate import examine, make_prompt  # noqa: E402
from role_path_pilot import forward  # noqa: E402
from extract_readout import MODELS  # noqa: E402

LOCK = json.loads((HERE / "QUERY_LOCK.json").read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def get_positions(tok, prompt, sentence, ids):
    end = prompt.index("Sentence: " + sentence) + len("Sentence: " + sentence) - 1
    assert prompt[end] == "."
    enc = tok._tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=False)
    assert enc["input_ids"] == ids
    overlaps = [i for i, (lo, hi) in enumerate(enc["offset_mapping"]) if lo <= end < hi]
    assert len(overlaps) == 1
    return [overlaps[0], len(ids)-1]


def selected_weights(model, first_ids):
    ix = mx.array(first_ids)
    if model.args.tie_word_embeddings:
        return model.model.embed_tokens(ix), model.model.embed_tokens.as_linear
    head = model.lm_head
    if hasattr(head, "scales"):
        rows = mx.dequantize(head.weight[ix], head.scales[ix], head.biases[ix],
                             group_size=head.group_size, bits=head.bits)
    else:
        rows = head.weight[ix]
    return rows, head


def main(key):
    assert key in LOCK["models"]
    out = HERE / f"QUERY_{key}.json"
    raw = HERE / f"QUERY_STATES_{key}.npz"
    assert not out.exists() and not raw.exists(), "refusing overwrite"
    start = time.monotonic()
    model, tok = load(str(MODELS[key]), tokenizer_config={"trust_remote_code": False})
    model.eval()
    mx.set_cache_limit(256*1024*1024)
    all_states, all_normed, axes, families = [], [], [], []
    for family in LOCK["families"]:
        names = [family["a"], family["b"]]
        first_ids = [tok.encode(n, add_special_tokens=False)[0] for n in names]
        assert first_ids[0] != first_ids[1]
        weights, head = selected_weights(model, first_ids)
        mx.eval(weights)
        w = np.asarray(weights).astype(np.float64)
        axes.append(w[0]-w[1])
        tasks, fam_states, fam_normed = [], [], []
        for question_key, question_template in LOCK["questions"].items():
            rows, states, normed_states = [], [], []
            for condition in LOCK["conditions"]:
                sentence = LOCK["sentences"][condition].format(**family)
                question = question_template.format(**family)
                agent_a = condition.startswith("a_agent")
                active = condition.endswith("active")
                expected_a = agent_a if question_key == "agent" or active else not agent_a
                expected = names[0 if expected_a else 1]
                prompt = make_prompt(tok, sentence, question)
                ids = tok.encode(prompt, add_special_tokens=False)
                row = examine(model, tok, prompt, names, expected)
                positions = get_positions(tok, prompt, sentence, ids)
                _, layers = forward(model, ids, capture=True)
                embedding = model.model.embed_tokens(mx.array([ids]))
                mx.eval(embedding)
                captured = np.empty((len(layers)+1, 2, embedding.shape[-1]), dtype=np.float16)
                for stage, state in enumerate([embedding]+layers):
                    for site, pos in enumerate(positions):
                        captured[stage, site] = np.asarray(state[0, pos]).astype(np.float16)
                zn = model.model.norm(mx.array(captured.astype(np.float32)))
                native = head(zn[-1, 1][None, :])[:, mx.array(first_ids)]
                mx.eval(zn, native)
                zn_np = np.asarray(zn).astype(np.float32)
                error = float(np.max(np.abs(zn_np[-1, 1].astype(np.float64)@w.T-np.asarray(native)[0])))
                assert error < 0.05, f"output head mismatch {error}"
                row.update({"question_key": question_key, "question": question,
                            "condition": condition, "sentence": sentence,
                            "positions": positions, "selected_head_max_error": error})
                rows.append(row)
                states.append(captured)
                normed_states.append(zn_np.astype(np.float16))
                print(key, family["id"], question_key, condition,
                      repr(row["generated"]), "expected", expected,
                      "margin", round(row["score_margin"], 3), flush=True)
            tasks.append({"question_key": question_key, "rows": rows})
            fam_states.append(np.stack(states))
            fam_normed.append(np.stack(normed_states))
            mx.clear_cache()
        all_states.append(np.stack(fam_states))
        all_normed.append(np.stack(fam_normed))
        families.append({"family": family, "tasks": tasks,
                         "fully_competent": all(r["greedy_correct"] and r["score_correct"]
                                                for t in tasks for r in t["rows"])})
    np.savez_compressed(raw, states=np.stack(all_states), normalized_states=np.stack(all_normed),
                        readout_axes=np.stack(axes),
                        family_ids=np.array([f["id"] for f in LOCK["families"]]),
                        questions=np.array(list(LOCK["questions"])),
                        conditions=np.array(LOCK["conditions"]), sites=np.array(LOCK["sites"]))
    result = {"status": "frozen_query_geometry_pilot_complete", "model": key,
              "lock_sha256": sha(HERE/"QUERY_LOCK.json"), "script_sha256": sha(__file__),
              "scorer_sha256": sha(SOURCE/"factorial_gate.py"),
              "forward_helper_sha256": sha(SOURCE/"role_path_pilot.py"),
              "config_sha256": sha(MODELS[key]/"config.json"),
              "states_sha256": sha(raw), "families": families,
              "elapsed_seconds": time.monotonic()-start,
              "state_shape": "family, question, sentence condition, stage, site, hidden dimension"}
    out.write_text(json.dumps(result, indent=2)+"\n")
    print(key, "complete", "fully competent", sum(f["fully_competent"] for f in families), "/8", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
