#!/usr/bin/env python3
"""Learn requested role from question text, then route fixed model role fields."""
import argparse
import hashlib
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parent
STARTS = ("fresh1", "fresh3")
ARMS = ("goal_first", "goal_after_roles", "role_first")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def norm(value):
    import re
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", (value or "").casefold())).strip()


def question_rows(items):
    return [(item["id"], "actor", item["actor_question"])
            for item in items] + [(item["id"], "patient", item["patient_question"])
                              for item in items]


def prepare():
    path = ROOT / "ROUTER_PROTOCOL.json"
    if path.exists():
        raise FileExistsError(path)
    cfg = {
        "status": "post hoc system follow-up, prospectively frozen before router training and new routing scores; same fresh bank already seen in matched-arm analysis",
        "question": "Can a learned question-to-role normalizer convert printed actor/patient fields into an end-to-end answer system without oracle requested-role labels?",
        "training": "Train a fixed character 2-5 gram TF-IDF plus C=1 logistic classifier on 320 actor/patient questions from the 160 training items. Input is question text only. No tuning or fresh labels used for fitting.",
        "evaluation": "Predict requested role for 80 questions from 40 fresh items, report exact question-string overlap with training. Use that predicted role to select a printed field from each fixed model output; score 160 prompts per checkpoint, both candidate orders. Missing fields are errors. Compare against model's own direct answer and separately report the oracle-role route as a ceiling.",
        "screen": "On each of two starts, role classifier >=76/80 fresh questions and routed role_first checkpoint improves exact answers by >=20/160 over that checkpoint's own direct answer. This is an exploratory system screen, not an independent model-training confirmation.",
        "limits": "The source actor/patient mapping is unreviewed; question forms are repetitive, and previous matched-arm results used this fresh bank. A positive is an end-to-end system result only within this small source distribution. It is not a novel general planner, a learned internal state, or a natural-domain publication claim.",
        "splits_sha256": sha(ROOT / "SPLITS.json"),
        "main_protocol_sha256": sha(ROOT / "PROTOCOL.json"),
        "baseline_protocol_sha256": sha(ROOT / "BASELINE_PROTOCOL.json"),
        "script_sha256": sha(Path(__file__)),
    }
    path.write_text(json.dumps(cfg, indent=2) + "\n")
    print("Frozen", sha(path))


def run():
    cfg = json.loads((ROOT / "ROUTER_PROTOCOL.json").read_text())
    for key, path in (("splits_sha256", ROOT / "SPLITS.json"),
                      ("main_protocol_sha256", ROOT / "PROTOCOL.json"),
                      ("baseline_protocol_sha256", ROOT / "BASELINE_PROTOCOL.json"),
                      ("script_sha256", Path(__file__))):
        assert cfg[key] == sha(path), key
    splits = json.loads((ROOT / "SPLITS.json").read_text())
    train = question_rows(splits["train"])
    fresh = question_rows(splits["fresh"])
    assert len(train) == 320 and len(fresh) == 80
    classifier = make_pipeline(TfidfVectorizer(analyzer="char", ngram_range=(2, 5), lowercase=True),
                               LogisticRegression(C=1, max_iter=2000, random_state=910127))
    classifier.fit([x[2] for x in train], [x[1] for x in train])
    predictions = classifier.predict([x[2] for x in fresh]).tolist()
    router = {(item, role): pred for (item, role, _), pred in zip(fresh, predictions)}
    assert len(router) == 80
    seen = {x[2] for x in train}
    summary = {
        "protocol_sha256": sha(ROOT / "ROUTER_PROTOCOL.json"),
        "question_count": len(fresh),
        "correct_questions": sum(gold == pred for (_, gold, _), pred in zip(fresh, predictions)),
        "exact_seen_questions": sum(q in seen for _, _, q in fresh),
        "novel_question_count": sum(q not in seen for _, _, q in fresh),
        "novel_question_correct": sum(q not in seen and gold == pred for (_, gold, q), pred in zip(fresh, predictions)),
        "question_records": [{"id": item, "gold": role, "question": question, "predicted": pred,
                              "seen_in_training": question in seen}
                             for (item, role, question), pred in zip(fresh, predictions)],
        "systems": {},
    }
    for start in STARTS:
        summary["systems"][start] = {}
        for arm in ARMS:
            path = (ROOT / f"BASELINE_{start}_{arm}_fresh.json" if arm == "role_first"
                    else ROOT / f"EVAL_{start}_{arm}_fresh.json")
            data = json.loads(path.read_text())
            rows = data["records"]
            assert len(rows) == 160
            outputs = []
            for row in rows:
                predicted_role = router[(row["id"], row["role"])]
                field = row[predicted_role]
                success = field is not None and norm(field) == norm(row["gold"])
                oracle = row[row["role"]] is not None and norm(row[row["role"]]) == norm(row["gold"])
                outputs.append({"id": row["id"], "role": row["role"], "order": row["order"],
                                "gold": row["gold"], "predicted_role": predicted_role,
                                "field": field, "correct": success, "oracle_correct": oracle,
                                "model_direct_correct": row["correct"]})
            items = {r["id"] for r in outputs}
            by_key = {(r["id"], r["role"], r["order"]): r for r in outputs}
            assert len(items) == 40 and len(by_key) == 160
            def all_four(field):
                return sum(all(by_key[(item, role, order)][field]
                               for role in ("actor", "patient") for order in (0, 1))
                           for item in items)
            summary["systems"][start][arm] = {
                "source_sha256": sha(path),
                "direct_exact": sum(r["model_direct_correct"] for r in outputs),
                "routed_exact": sum(r["correct"] for r in outputs),
                "oracle_role_routed_exact": sum(r["oracle_correct"] for r in outputs),
                "direct_all_four": all_four("model_direct_correct"),
                "routed_all_four": all_four("correct"),
                "oracle_role_routed_all_four": all_four("oracle_correct"),
                "records": outputs,
            }
    summary["screen_pass"] = (
        summary["correct_questions"] >= 76 and
        all(summary["systems"][start]["role_first"]["routed_exact"] -
            summary["systems"][start]["role_first"]["direct_exact"] >= 20
            for start in STARTS))
    (ROOT / "ROUTER_ANALYSIS.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"question_correct": summary["correct_questions"],
                      "novel_question_correct": summary["novel_question_correct"],
                      "systems": {start: {arm: {k: v for k, v in values.items()
                                                if k not in ("records", "source_sha256")}
                                          for arm, values in arms.items()}
                                  for start, arms in summary["systems"].items()},
                      "screen_pass": summary["screen_pass"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "run"))
    args = parser.parse_args()
    (prepare if args.mode == "prepare" else run)()
