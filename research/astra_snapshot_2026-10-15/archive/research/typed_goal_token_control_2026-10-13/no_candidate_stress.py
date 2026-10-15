"""Post hoc frozen stress: remove the candidate line from both fixed models."""
import hashlib
import importlib.util
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
TYPED = ROOT.parent / "typed_goal_conditioning_2026-10-13"
spec = importlib.util.spec_from_file_location("typed_goal_source", TYPED / "run.py")
typed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(typed)
STARTS = ("fresh1", "fresh3")
ARMS = ("typed_goal", "null_two_token")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def prepare():
    path = ROOT / "NO_CANDIDATE_PROTOCOL.json"
    if path.exists():
        raise FileExistsError(path)
    cfg = {
        "date": "2026-10-13", "status": "post hoc stress designed after full fresh3 candidate-supported result; frozen before no-candidate inference",
        "question": "Does the typed input benefit survive removing the supplied answer candidates at inference?",
        "items": "The same 20 fresh3 sentences; one actor question and one patient question per sentence, giving 40 distinct prompts. No new-item confirmation.",
        "intervention": "Delete only the Candidates line from the test prompt. Keep Sentence, Requested role, Question, system instruction, fixed classifier, checkpoints, and greedy decoding. No retraining or calibration.",
        "outcomes": "Exact final answers/40, both printed role fields correct/40, final answer agrees with requested printed field/40, parsed/40. Compare arms within start. Diagnostic only; no prespecified pass threshold.",
        "limits": "Out-of-distribution prompt removal, same already-used fresh3 items, automatic unreviewed labels, one model family. A failure would not negate the candidate-supported result; a success would still need new candidate-free training and external tests.",
        "source_protocol_sha256": sha(ROOT / "PROTOCOL.json"),
        "source_splits_sha256": sha(ROOT / "SPLITS.json"),
        "script_sha256": sha(Path(__file__)),
        "checkpoint_sha256": {f"{s}:{a}": sha((TYPED / f"{s}_typed_goal" / "adapter.pt") if a == "typed_goal"
                                                 else (ROOT / f"{s}_null_two_token" / "adapter.pt"))
                               for s in STARTS for a in ARMS},
    }
    path.write_text(json.dumps(cfg, indent=2) + "\n")
    print("Frozen", sha(path))


def prompt(tok, row, arm, predicted_role):
    tag = predicted_role.upper() if arm == "typed_goal" else "NO ROLE"
    system = "Give exactly: Actor: SPAN; Patient: SPAN; Answer: SPAN"
    user = (f"Sentence: {row['sentence']}\nRequested role: {tag}\nQuestion: {row['question']}")
    return tok.apply_chat_template([{"role": "system", "content": system},
                                    {"role": "user", "content": user}],
                                   tokenize=False, add_generation_prompt=True) + "Actor:"


@torch.inference_mode()
def evaluate(start, arm):
    cfg = load(ROOT / "NO_CANDIDATE_PROTOCOL.json")
    assert cfg["source_protocol_sha256"] == sha(ROOT / "PROTOCOL.json")
    assert cfg["source_splits_sha256"] == sha(ROOT / "SPLITS.json")
    assert cfg["script_sha256"] == sha(Path(__file__))
    dest = ROOT / f"NO_CANDIDATE_{start}_{arm}.json"
    if dest.exists():
        raise FileExistsError(dest)
    torch.set_num_threads(2)
    tok, model = typed.prior.prior.base.prev.load(start)
    cp = ((TYPED / f"{start}_typed_goal" / "adapter.pt") if arm == "typed_goal"
          else (ROOT / f"{start}_null_two_token" / "adapter.pt"))
    assert cfg["checkpoint_sha256"][f"{start}:{arm}"] == sha(cp)
    saved = torch.load(cp, map_location="cpu", weights_only=True)
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    assert saved.keys() == params.keys()
    for n, p in params.items():
        p.data.copy_(saved[n])
    model.eval()
    bank = load(ROOT / "SPLITS.json")
    predictor = typed.classifier(bank["train"])
    predicted = typed.role_predictions(bank["fresh3"], predictor)
    rows = [r for item in bank["fresh3"] for r in typed.prior.rows(item) if r["order"] == 0]
    assert len(rows) == 40
    records = []
    for off in range(0, len(rows), 4):
        block = rows[off:off+4]
        prompts = [prompt(tok, r, arm, predicted[(r["id"], r["role"])]) for r in block]
        encoded = tok(prompts, padding=True, return_tensors="pt", add_special_tokens=False)
        generated = model.generate(**encoded, max_new_tokens=48, do_sample=False,
                                   pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        for row, tokens in zip(block, generated[:, encoded["input_ids"].shape[1]:]):
            raw = tok.decode(tokens, skip_special_tokens=True)
            actor, patient, answer = typed.parse(raw)
            field = actor if row["role"] == "actor" else patient
            records.append({"id": row["id"], "role": row["role"], "gold": row["gold"],
                            "raw": raw, "parsed": answer is not None,
                            "actor_correct": actor is not None and typed.norm(actor) == typed.norm(row["actor"]),
                            "patient_correct": patient is not None and typed.norm(patient) == typed.norm(row["patient"]),
                            "correct": answer is not None and typed.norm(answer) == typed.norm(row["gold"]),
                            "answer_matches_field": answer is not None and field is not None
                                                    and typed.norm(answer) == typed.norm(field)})
        print(start, arm, off+4, "/", len(rows), flush=True)
    dest.write_text(json.dumps({"start": start, "arm": arm,
                                "protocol_sha256": sha(ROOT / "NO_CANDIDATE_PROTOCOL.json"),
                                "checkpoint_sha256": sha(cp), "records": records}, indent=2) + "\n")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("prepare", "eval"))
    ap.add_argument("--start", choices=STARTS)
    ap.add_argument("--arm", choices=ARMS)
    args = ap.parse_args()
    if args.mode == "prepare":
        prepare()
    else:
        assert args.start and args.arm
        evaluate(args.start, args.arm)
