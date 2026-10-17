"""Fresh-family behavioral competence gate for the frozen transport study."""
import hashlib
import json
import sys
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

from factorial_gate import examine, make_prompt
from role_path_pilot import MODELS

HERE = Path(__file__).parent
LOCK = json.loads((HERE / "TRANSPORT_LOCK.json").read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build(fam, condition):
    sentence = LOCK["sentences"][condition].format(**fam)
    question = ("Who photographed someone?" if condition == "same_answer_unrelated"
                else LOCK["question"].format(verb=fam["verb"]))
    expected = fam["a"] if condition in (
        "b_patient_active_second", "b_patient_passive_first", "receiver_cleft_a_agent"
    ) else fam["b"]
    return sentence, question, expected


def main(key):
    assert key in LOCK["models"]
    out = HERE / f"TRANSPORT_GATE_{key}.json"
    assert not out.exists(), f"refusing overwrite: {out}"
    model_path = MODELS[key]
    model, tok = load(str(model_path), tokenizer_config={"trust_remote_code": False})
    model.eval()
    mx.set_cache_limit(256 * 1024 * 1024)
    families = []
    for fam in LOCK["families"]:
        rows = []
        for condition in LOCK["seven_conditions"]:
            sentence, question, expected = build(fam, condition)
            row = examine(model, tok, make_prompt(tok, sentence, question),
                          [fam["a"], fam["b"]], expected)
            row.update({"condition": condition, "sentence": sentence, "question": question})
            rows.append(row)
            print(key, fam["id"], condition, row["generated"],
                  "expected", expected, "margin", round(row["score_margin"], 3), flush=True)
        families.append({"family": fam, "rows": rows,
                         "competent": all(r["greedy_correct"] and r["score_correct"] for r in rows)})
        mx.clear_cache()
    result = {"status": "frozen_transport_competence_gate_complete", "model": key,
              "lock_sha256": sha(HERE / "TRANSPORT_LOCK.json"),
              "script_sha256": sha(__file__), "scorer_sha256": sha(HERE / "factorial_gate.py"),
              "config_sha256": sha(model_path / "config.json"), "families": families}
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(out, "competent", sum(f["competent"] for f in families), "/", len(families))


if __name__ == "__main__":
    main(sys.argv[1])
