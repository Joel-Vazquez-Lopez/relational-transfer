"""Capture all-layer descriptive states for the single selected f07 family."""
import hashlib
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load

SOURCE = Path(__file__).resolve().parents[1] / "causal_developmental_alignment_2026-10-17"
sys.path.insert(0, str(SOURCE))
from factorial_gate import make_prompt  # noqa: E402
from role_path_pilot import forward  # noqa: E402

HERE = Path(__file__).resolve().parent
GATE = json.loads((HERE / "GATE_LOCK.json").read_text())
LOCK = json.loads((HERE / "PATH_VIEW_LOCK.json").read_text())
RESEARCH = HERE.parent
MODELS = {
    "qwen05": RESEARCH / "stronger_model_development_2026-10-15/models/qwen05",
    "qwen15": RESEARCH / "stronger_model_development_2026-10-15/models/qwen15",
    "qwen7b": RESEARCH / "stronger_model_development_2026-10-15/models/qwen7b",
    "llama3b": RESEARCH / "llama_gemma_cross_family_2026-10-15/model",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def site_indices(tok, text, ids, sentence):
    marker = "Sentence: " + sentence
    assert marker in text
    dot_pos = text.index(marker) + len(marker) - 1
    assert text[dot_pos] == "."
    enc = tok._tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    assert enc["input_ids"] == ids
    overlaps = [i for i, (start, end) in enumerate(enc["offset_mapping"])
                if start <= dot_pos < end]
    assert len(overlaps) == 1, (dot_pos, overlaps)
    return [overlaps[0], len(ids) - 1]


def main(key):
    assert key in LOCK["models"]
    raw = HERE / f"PATH_VIEW_{key}.npz"
    meta = HERE / f"PATH_VIEW_{key}.json"
    assert not raw.exists() and not meta.exists(), "refusing overwrite"
    fam = next(x for x in GATE["families"] if x["id"] == LOCK["family_id"])
    model_path = MODELS[key]
    model, tok = load(str(model_path), tokenizer_config={"trust_remote_code": False})
    model.eval()
    mx.set_cache_limit(256 * 1024 * 1024)
    vectors = []
    metadata = []
    for condition in LOCK["conditions"]:
        sentence = GATE["sentence_templates"][condition].format(**fam)
        question = GATE["question"].format(**fam)
        text = make_prompt(tok, sentence, question)
        ids = tok.encode(text, add_special_tokens=False)
        positions = site_indices(tok, text, ids, sentence)
        _, layers = forward(model, ids, capture=True)
        embedding = model.model.embed_tokens(mx.array([ids]))
        mx.eval(embedding)
        states = [embedding] + layers
        arr = np.empty((len(states), len(positions), int(embedding.shape[-1])), dtype=np.float16)
        for stage, state in enumerate(states):
            for site, pos in enumerate(positions):
                arr[stage, site] = np.asarray(state[0, pos, :]).astype(np.float16)
        vectors.append(arr)
        metadata.append({"condition": condition, "sentence": sentence, "prompt": text,
                         "token_count": len(ids), "positions": positions})
        mx.clear_cache()
        print(key, condition, arr.shape, positions, flush=True)
    combined = np.stack(vectors)
    np.savez_compressed(raw, states=combined,
                        conditions=np.array(LOCK["conditions"]), sites=np.array(LOCK["sites"]))
    record = {"model": key, "status": "descriptive_path_capture_only",
              "path_view_lock_sha256": sha(HERE / "PATH_VIEW_LOCK.json"),
              "gate_lock_sha256": sha(HERE / "GATE_LOCK.json"),
              "script_sha256": sha(__file__), "states_sha256": sha(raw),
              "config_sha256": sha(model_path / "config.json"),
              "shape": list(combined.shape), "metadata": metadata}
    meta.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
