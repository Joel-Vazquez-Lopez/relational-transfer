"""Frozen four-model behavioral gate for family-dependent computation study."""
import hashlib
import json
import sys
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

SOURCE = Path(__file__).resolve().parents[1] / "causal_developmental_alignment_2026-10-17"
sys.path.insert(0, str(SOURCE))
from factorial_gate import examine, make_prompt  # noqa: E402

HERE = Path(__file__).resolve().parent
LOCK = json.loads((HERE / "GATE_LOCK.json").read_text())
RESEARCH = HERE.parent
MODEL_PATHS = {
    "qwen05": RESEARCH / "stronger_model_development_2026-10-15/models/qwen05",
    "qwen15": RESEARCH / "stronger_model_development_2026-10-15/models/qwen15",
    "qwen7b": RESEARCH / "stronger_model_development_2026-10-15/models/qwen7b",
    "llama3b": RESEARCH / "llama_gemma_cross_family_2026-10-15/model",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(key):
    assert key in LOCK["models"]
    out = HERE / f"GATE_{key}.json"
    assert not out.exists(), f"refusing overwrite: {out}"
    model_path = MODEL_PATHS[key]
    model, tok = load(str(model_path), tokenizer_config={"trust_remote_code": False})
    model.eval()
    mx.set_cache_limit(256 * 1024 * 1024)
    families = []
    for fam in LOCK["families"]:
        rows = []
        for condition in LOCK["conditions"]:
            sentence = LOCK["sentence_templates"][condition].format(**fam)
            question = LOCK["question"].format(**fam)
            expected = fam["a"] if condition.startswith("a_agent") else fam["b"]
            prompt = make_prompt(tok, sentence, question)
            row = examine(model, tok, prompt, [fam["a"], fam["b"]], expected)
            row.update({"condition": condition, "sentence": sentence, "question": question})
            rows.append(row)
            print(key, fam["id"], condition, repr(row["generated"]),
                  "expected", expected, "margin", round(row["score_margin"], 3), flush=True)
        competent = all(row["greedy_correct"] and row["score_correct"] for row in rows)
        families.append({"family": fam, "rows": rows, "competent": competent})
        mx.clear_cache()
    result = {
        "status": "family_competence_gate_complete", "model": key,
        "lock_sha256": sha(HERE / "GATE_LOCK.json"),
        "script_sha256": sha(__file__),
        "scorer_sha256": sha(SOURCE / "factorial_gate.py"),
        "config_sha256": sha(model_path / "config.json"),
        "families": families,
    }
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(key, "competent", sum(f["competent"] for f in families), "/", len(families), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
