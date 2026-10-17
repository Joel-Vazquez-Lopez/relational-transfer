"""Read output weights and apply final RMSNorm to already captured states.

No prompt forward pass, generation, patch or model download. This exploratory
diagnostic removes one known direct first-token readout axis; it does not remove
all possible answer-related information.
"""
import hashlib
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
SOURCE = RESEARCH / "family_dependent_computation_2026-10-17"
MODELS = {
    "qwen05": RESEARCH / "stronger_model_development_2026-10-15/models/qwen05",
    "qwen15": RESEARCH / "stronger_model_development_2026-10-15/models/qwen15",
    "qwen7b": RESEARCH / "stronger_model_development_2026-10-15/models/qwen7b",
    "llama3b": RESEARCH / "llama_gemma_cross_family_2026-10-15/model",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(key):
    target = HERE / f"READOUT_{key}.npz"
    metadata_path = HERE / f"READOUT_{key}.json"
    assert not target.exists() and not metadata_path.exists(), "refusing overwrite"
    source = SOURCE / f"PATH_VIEW_{key}.npz"
    old_meta = json.loads((SOURCE / f"PATH_VIEW_{key}.json").read_text())
    assert sha(source) == old_meta["states_sha256"]
    states = np.load(source)["states"].astype(np.float32)
    model, tok = load(str(MODELS[key]), tokenizer_config={"trust_remote_code": False})
    model.eval()
    mx.set_cache_limit(256 * 1024 * 1024)
    names = ["Miriam", "Adrian"]
    name_ids = [tok.encode(n, add_special_tokens=False) for n in names]
    first_ids = [ids[0] for ids in name_ids]
    assert first_ids[0] != first_ids[1]
    indexes = mx.array(first_ids)
    if model.args.tie_word_embeddings:
        rows = model.model.embed_tokens(indexes)
        head = model.model.embed_tokens.as_linear
    else:
        layer = model.lm_head
        head = layer
        if hasattr(layer, "scales"):
            rows = mx.dequantize(layer.weight[indexes], layer.scales[indexes],
                                 layer.biases[indexes], group_size=layer.group_size,
                                 bits=layer.bits)
        else:
            rows = layer.weight[indexes]
    normalized = model.model.norm(mx.array(states))
    mx.eval(rows, normalized)
    weights = np.asarray(rows).astype(np.float64)
    normalized_np = np.asarray(normalized).astype(np.float32)
    # Technical verification against the native head at the four final states.
    probe = normalized[:, -1, 1, :]
    native = head(probe)[:, indexes]
    mx.eval(native)
    predicted = normalized_np[:, -1, 1, :].astype(np.float64) @ weights.T
    max_error = float(np.max(np.abs(predicted - np.asarray(native))))
    assert max_error < 0.05, f"selected-weight/head mismatch: {max_error}"
    np.savez_compressed(target, normalized_states=normalized_np, readout_weights=weights)
    info = {"model": key, "status": "saved_state_readout_diagnostic_only",
            "source_states_sha256": sha(source), "script_sha256": sha(__file__),
            "config_sha256": sha(MODELS[key] / "config.json"),
            "names": names, "name_token_ids": name_ids, "first_token_ids": first_ids,
            "selected_head_max_abs_error": max_error, "output_sha256": sha(target),
            "limitation": "Only one first-answer-token margin axis is removed. Earlier-layer use of final RMSNorm/head is a logit-lens convention, not the true downstream computation. Saved residual states are float16 captures."}
    metadata_path.write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
