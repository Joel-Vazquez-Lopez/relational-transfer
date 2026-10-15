#!/usr/bin/env python3
"""Post hoc candidate-tuple diagnostic; no training or sealed data."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import boost  # noqa: E402

FORMS = ("active", "object_cleft")
ARMS = ("old_sft", "old_coupled")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def cases():
    return [(event, form) for event in boost.pilot.prev.events(boost.PAIRS, boost.VERBS)
            for form in FORMS]


def prefix(tok, event: dict, form: str, include_sentence: bool):
    sentence = boost.pilot.prev.sentence(event, form)
    body = (f"Sentence: {sentence}\n" if include_sentence else "")
    body += (f"Candidates: {', '.join(event['pair'])}\n"
             "Identify the actor and patient of the action. Give both names.")
    return tok.apply_chat_template([
        {"role": "system", "content": "Answer in exactly this format: Actor: NAME; Patient: NAME"},
        {"role": "user", "content": body},
    ], tokenize=False, add_generation_prompt=True) + "Actor:"


def prepare():
    target = ROOT / "TUPLE_PROTOCOL.json"
    if target.exists():
        raise FileExistsError(target)
    data = {
        "date": "2026-10-12",
        "status": "post_hoc_diagnostic_fixed_before_tuple_inference; prior_direct_answers_seen",
        "question": "Can the same checkpoints prefer the correct ordered (actor, patient) tuple over the reversed tuple without separate actor/patient question wording? Compare active and object cleft; also score a sentence-omitted control for candidate-order preference.",
        "arms": ARMS, "starts": boost.pilot.STARTS, "forms": FORMS,
        "cases": len(cases()),
        "metric": "Teacher-forced log-likelihood of ' NAME; Patient: NAME' after a fixed 'Actor:' prefix. Correct-minus-reversed tuple log score, and whether its sign is positive. Same two names in opposite positions have the same token count. Report sentence-present and sentence-omitted scores; do not interpret a positive score as a verified internal event representation.",
        "limitations": "Post hoc on the same already-inspected constructed bank. Prompt/format itself may change behavior. Candidate-relative scores may reflect name/order priors.",
        "script_sha256": sha(Path(__file__)),
        "boost_protocol_sha256": sha(ROOT / "PROTOCOL.json"),
        "checkpoint_sha256": {
            f"{s}_{a}": sha(boost.checkpoint(s, a))
            for s in boost.pilot.STARTS for a in ARMS
        },
    }
    target.write_text(json.dumps(data, indent=2) + "\n")
    print("Saved tuple diagnostic protocol", flush=True)


def validate():
    cfg = json.loads((ROOT / "TUPLE_PROTOCOL.json").read_text())
    assert cfg["script_sha256"] == sha(Path(__file__))
    assert cfg["boost_protocol_sha256"] == sha(ROOT / "PROTOCOL.json")
    for k, expected in cfg["checkpoint_sha256"].items():
        s, a = k.split("_", 1)
        assert sha(boost.checkpoint(s, a)) == expected
    return cfg


def make_entries(tok, event: dict, form: str, include_sentence: bool):
    p = prefix(tok, event, form, include_sentence)
    pids = tok.encode(p, add_special_tokens=False)
    names = (event["actor"], event["patient"])
    out = []
    for is_correct, (a, b) in ((True, names), (False, names[::-1])):
        suffix = f" {a}; Patient: {b}"
        full = tok.encode(p + suffix, add_special_tokens=False)
        assert full[:len(pids)] == pids
        assert len(full) > len(pids)
        out.append((is_correct, full, [-100]*len(pids) + full[len(pids):]))
    assert len(out[0][1]) == len(out[1][1])
    return out


@torch.inference_mode()
def evaluate(start: str, arm: str):
    cfg = validate()
    target = ROOT / f"TUPLE_{start}_{arm}.json"
    if target.exists():
        raise FileExistsError(target)
    torch.set_num_threads(2)
    tok, model = boost.pilot.prev.load(start)
    saved = torch.load(boost.checkpoint(start, arm), map_location="cpu", weights_only=True)
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    assert saved.keys() == params.keys()
    for n, p in params.items():
        p.data.copy_(saved[n])
    model.eval()
    output = []
    for i, (event, form) in enumerate(cases()):
        for include_sentence in (True, False):
            entries = make_entries(tok, event, form, include_sentence)
            length = max(len(ids) for _, ids, _ in entries)
            ids = torch.tensor([[tok.pad_token_id]*(length-len(x[1])) + x[1] for x in entries])
            labels = torch.tensor([[-100]*(length-len(x[1])) + x[2] for x in entries])
            mask = (ids != tok.pad_token_id).long()
            logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
            losses = F.cross_entropy(logits[:, :-1].contiguous().flatten(0, 1),
                                     labels[:, 1:].contiguous().flatten(),
                                     ignore_index=-100, reduction="none").view(2, -1)
            scores = (-losses.sum(dim=1)).tolist()
            output.append({
                "event": event["id"], "form": form,
                "sentence_present": include_sentence,
                "correct_tuple_log_score": scores[0],
                "reversed_tuple_log_score": scores[1],
                "correct_minus_reversed": scores[0]-scores[1],
                "correct_preferred": scores[0] > scores[1],
            })
        if (i+1) % 24 == 0:
            print(f"{start}_{arm}: {i+1}/{len(cases())} cases", flush=True)
    target.write_text(json.dumps({
        "start": start, "arm": arm,
        "protocol_sha256": sha(ROOT / "TUPLE_PROTOCOL.json"),
        "checkpoint_sha256": cfg["checkpoint_sha256"][f"{start}_{arm}"],
        "records": output,
    }, indent=2) + "\n")
    print(f"Saved {target.name}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "eval"))
    parser.add_argument("--start", choices=boost.pilot.STARTS)
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare()
    else:
        if args.start is None or args.arm is None:
            parser.error("eval needs --start and --arm")
        evaluate(args.start, args.arm)
