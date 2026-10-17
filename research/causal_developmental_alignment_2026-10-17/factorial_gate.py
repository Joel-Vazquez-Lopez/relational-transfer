"""Frozen local competence/readout gate for role-swap x voice factorial.

Usage: MODEL=qwen7b|llama3b python factorial_gate.py MODEL
Run one model per process. Writes raw JSON with every prompt, answer and score.
"""
import hashlib
import json
import re
import sys
from pathlib import Path

import mlx.core as mx
from mlx_lm import generate, load

from role_path_pilot import CASES as OLD_CASES, MODELS, SYSTEM, prompt as old_prompt

HERE = Path(__file__).parent
LOCK = json.loads((HERE / "FACTORIAL_GATE_LOCK.json").read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize(x):
    return re.sub(r"[^a-z]+", "", x.lower())


def cell(family, r, v, frame):
    a, b, verb = family["a"], family["b"], family["verb"]
    agent, patient = (b, a) if r else (a, b)
    sentence = LOCK["frames"][frame]["passive" if v else "active"].format(
        agent=agent, patient=patient, verb=verb
    )
    return sentence, agent, patient


def make_prompt(tok, sentence, question):
    user = f"Sentence: {sentence}\nQuestion: {question}"
    return tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True
    )


def full_name_logprob(model, tok, prompt_ids, name):
    # Continuation tokens are scored as a *sequence*, fixing first-token
    # aliases such as Llama's partial 'Cl'/'Ell'. No length normalization.
    continuation = tok.encode(name, add_special_tokens=False)
    if not continuation:
        raise ValueError(f"empty candidate for {name}")
    full = prompt_ids + continuation
    logits = model(mx.array([full]))[0]
    values = []
    for j, token_id in enumerate(continuation):
        row = logits[len(prompt_ids) - 1 + j]
        values.append(row[token_id] - mx.logsumexp(row))
    score = mx.sum(mx.stack(values))
    mx.eval(score)
    return float(score.item()), continuation


def examine(model, tok, p, names, expected):
    ids = tok.encode(p, add_special_tokens=False)
    generated = generate(model, tok, p, max_tokens=10).strip()
    scores = {}
    tokens = {}
    for name in names:
        scores[name], tokens[name] = full_name_logprob(model, tok, ids, name)
    rival = names[1] if expected == names[0] else names[0]
    return {
        "prompt": p, "prompt_token_count": len(ids), "generated": generated,
        "expected": expected, "rival": rival, "full_name_logprob": scores,
        "candidate_token_ids": tokens,
        "greedy_correct": normalize(generated) == normalize(expected),
        "score_correct": scores[expected] > scores[rival],
        "score_margin": scores[expected] - scores[rival]
    }


def main(key):
    assert key in LOCK["models"]
    out = HERE / f"FACTORIAL_GATE_{key}.json"
    assert not out.exists(), f"refusing overwrite: {out}"
    model_path = MODELS[key]
    model, tok = load(str(model_path), tokenizer_config={"trust_remote_code": False})
    model.eval()
    mx.set_cache_limit(256 * 1024 * 1024)
    old = []
    for case in OLD_CASES:
        for reverse in (False, True):
            names = case["names"]
            expected = names[int(reverse)]
            row = examine(model, tok, old_prompt(tok, case, reverse), names, expected)
            row.update({"case_id": case["id"], "reverse": reverse})
            old.append(row)
            print(key, "validation", case["id"], reverse, row["generated"],
                  round(row["score_margin"], 3), flush=True)
        mx.clear_cache()
    valid = all(x["greedy_correct"] and x["score_correct"] for x in old)
    families = []
    if valid:
        for fam in LOCK["families"]:
            rows = []
            for frame in LOCK["frames"]:
                for r in (0, 1):
                    for v in (0, 1):
                        sentence, agent, patient = cell(fam, r, v, frame)
                        for role, expected in (("agent", agent), ("patient", patient)):
                            q = LOCK["questions"][role].format(verb=fam["verb"])
                            row = examine(model, tok, make_prompt(tok, sentence, q),
                                          [fam["a"], fam["b"]], expected)
                            row.update({"frame": frame, "r": r, "v": v,
                                        "role": role, "sentence": sentence, "question": q})
                            rows.append(row)
                            print(key, fam["id"], frame, r, v, role,
                                  row["generated"], round(row["score_margin"], 3), flush=True)
            families.append({"family": fam, "rows": rows,
                             "competent": all(x["greedy_correct"] and x["score_correct"] for x in rows),
                             "correct_count": sum(x["greedy_correct"] and x["score_correct"] for x in rows)})
            mx.clear_cache()
    result = {
        "status": "factorial_baseline_gate_complete", "model": key,
        "lock_sha256": sha(HERE / "FACTORIAL_GATE_LOCK.json"),
        "script_sha256": sha(__file__), "config_sha256": sha(model_path / "config.json"),
        "old_scorer_validation_passed": valid, "old_validation": old,
        "families": families,
        "limits": ["No new hidden-state capture or causal patch is part of this gate.",
                   "Full-name log probabilities are teacher-forced continuation scores without EOS; greedy generations are separately checked."]
    }
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(out, "validation", valid, "competent", sum(f["competent"] for f in families),
          "/", len(families), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
