#!/usr/bin/env python3
"""Prospective matched object-cleft exposure control for the role-coupling pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parents[1]
sys.path.insert(0, str(ROOT.parent / "role_prompt_equivariance_pilot_2026-10-12"))
import pilot  # noqa: E402

PAIRS = (("Harry", "Susan"), ("Alan", "Ada"), ("Gary", "Albert"),
         ("Louis", "Mark"), ("Ian", "Carl"), ("Tony", "Tim"))
VERBS = ("challenged", "stopped", "encouraged")
ARMS = ("old_sft", "old_coupled", "boost_sft", "boost_coupled")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def forms(index: int, epoch: int) -> tuple[str, ...]:
    other = "passive" if (index + epoch) % 2 == 0 else "subject_cleft"
    return ("active", "object_cleft", other)


def rows() -> list[dict]:
    return [pilot.question_row(event, form, name, qtext)
            for event in pilot.prev.events(PAIRS, VERBS)
            for form in pilot.FORMS for name, qtext in pilot.TEST_Q.items()]


def checkpoint(start: str, arm: str) -> Path:
    if arm.startswith("old_"):
        old = "balanced_sft" if arm == "old_sft" else "role_coupled"
        return pilot.ROOT / f"{start}_{old}" / "adapter.pt"
    return ROOT / f"{start}_{arm}" / "adapter.pt"


def prepare() -> None:
    target = ROOT / "PROTOCOL.json"
    if target.exists():
        raise FileExistsError(target)
    prior4 = (("Alice", "Bob"), ("Bruce", "Jane"), ("David", "Amy"),
              ("Arthur", "Max"), ("Frank", "Rose"), ("Martin", "June"))
    names = [n for pair in PAIRS for n in pair]
    old_names = {n for pair in (pilot.prev.TRAIN_PAIRS + pilot.prev.TEST_PAIRS +
                               pilot.mixed.TEST_PAIRS + pilot.TEST_PAIRS + prior4)
                 for n in pair}
    assert len(names) == len(set(names)) == 12
    assert not set(names) & old_names
    assert not set(VERBS) & set(pilot.prev.TRAIN_VERBS + pilot.prev.TEST_VERBS +
                                pilot.mixed.TEST_VERBS + pilot.TEST_VERBS +
                                ("helped", "followed", "protected"))
    tok, model = pilot.prev.load("fresh1")
    for name in names:
        assert len(tok.encode(name, add_special_tokens=False)) == 1
        assert len(tok.encode(" " + name, add_special_tokens=False)) == 1
    del model
    items = rows()
    assert len(items) == 720
    cfg = {
        "date": "2026-10-12",
        "status": "frozen_before_boost_training_or_any_new_test_inference",
        "reason": "The preceding role/prompt pilot made only 16-23 of 48 TRAIN object-cleft patient decisions correctly. Test whether higher object-cleft exposure under the same answer-target and update budget explains the failure.",
        "pairs": PAIRS, "verbs": VERBS,
        "starts": pilot.STARTS, "arms": ARMS,
        "questions": pilot.TEST_Q, "forms_test": pilot.FORMS,
        "train_schedule": "Each directed training event is seen for two epochs. At each update: active, object cleft, and alternating passive/subject cleft, 3 forms x 3 questions = 9 answer targets. Each event gets active and object cleft twice, passive and subject cleft once. The old schedule used active and passive twice, subject/object cleft once.",
        "matched": "Boosted SFT and boosted coupled begin from the same natural source within each start, use the original 48 directed events, 96 updates, 864 answer targets, questions, seed, LR, optimizer and LoRA configuration. The coupled objective is exactly the earlier frozen pilot objective; only the shared form schedule changes.",
        "primary": "For each start and arm, object-cleft event correctness on both held-out actor_carried and patient_which questions, out of 36. Compare boosted SFT with old SFT, and boosted coupled with boosted SFT. Report all four forms and all five questions.",
        "screen": "A simple exposure explanation is supported if boosted SFT improves object-cleft held-out two-role correctness by at least 8/36 in EACH start versus old SFT while active and passive held-out two-role correctness each fall by at most 4/36. A separate role-specific gain requires boosted coupled to improve over boosted SFT by at least 6/144 across all forms in EACH start with no object-cleft regression. These are pilot screens, not publication claims.",
        "limits": "One Qwen0.5B family, two related starts, constructed English and one new lexical bank. Boosting object clefts necessarily reduces passive exposure under a fixed budget, so any result is a distribution trade-off. This protocol follows inspection of the preceding test, but the new names and verbs were unused.",
        "script_sha256": sha(Path(__file__)),
        "pilot_script_sha256": sha(pilot.ROOT / "pilot.py"),
        "pilot_protocol_sha256": sha(pilot.ROOT / "PROTOCOL.json"),
        "source_checkpoint_sha256": {s: sha(pilot.prev.source_path(s)) for s in pilot.STARTS},
        "old_checkpoint_sha256": {f"{s}_{a}": sha(checkpoint(s, a))
                                  for s in pilot.STARTS for a in ("old_sft", "old_coupled")},
        "rows_sha256": hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest(),
    }
    target.write_text(json.dumps(cfg, indent=2) + "\n")
    print("Frozen new 36-event lexical bank and object-cleft budget screen", flush=True)


def validate() -> dict:
    cfg = json.loads((ROOT / "PROTOCOL.json").read_text())
    assert cfg["script_sha256"] == sha(Path(__file__))
    assert cfg["pilot_script_sha256"] == sha(pilot.ROOT / "pilot.py")
    assert cfg["pilot_protocol_sha256"] == sha(pilot.ROOT / "PROTOCOL.json")
    assert cfg["rows_sha256"] == hashlib.sha256(json.dumps(rows(), sort_keys=True).encode()).hexdigest()
    for s in pilot.STARTS:
        assert cfg["source_checkpoint_sha256"][s] == sha(pilot.prev.source_path(s))
        for a in ("old_sft", "old_coupled"):
            assert cfg["old_checkpoint_sha256"][f"{s}_{a}"] == sha(checkpoint(s, a))
    return cfg


def train(start: str, arm: str) -> None:
    cfg = validate()
    assert arm in ("boost_sft", "boost_coupled")
    target = ROOT / f"{start}_{arm}"
    if target.exists():
        raise FileExistsError(target)
    torch.set_num_threads(2)
    tok, model = pilot.prev.load(start)
    torch.manual_seed(pilot.SEED)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=pilot.LR)
    events = pilot.prev.train_events()
    order = list(range(len(events)))
    rng = random.Random(pilot.SEED)
    traces = []
    started = time.perf_counter()
    model.train()
    for epoch in range(pilot.EPOCHS):
        rng.shuffle(order)
        for idx in order:
            event = events[idx]
            ids, labels, mask, positions = pilot.batch(tok, event, forms(idx, epoch))
            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
            token_losses = F.cross_entropy(logits[:, :-1].contiguous().flatten(0, 1),
                                           labels[:, 1:].contiguous().flatten(),
                                           ignore_index=-100, reduction="none").view(9, -1)
            counts = (labels[:, 1:] != -100).sum(dim=1)
            ce = (token_losses.sum(dim=1) / counts).mean()
            js = contrast = torch.zeros((), dtype=ce.dtype)
            if arm == "boost_coupled":
                pair_ids = tuple(tok.encode(n, add_special_tokens=False)[0] for n in event["pair"])
                actor_index = event["pair"].index(event["actor"])
                js, contrast = pilot.coupled_loss(logits, positions, pair_ids, actor_index)
            loss = ce + pilot.LAMBDA_JS*js + pilot.LAMBDA_CONTRAST*contrast
            if not torch.isfinite(loss):
                raise FloatingPointError((start, arm, epoch, idx))
            loss.backward()
            optimizer.step()
            traces.append({"ce": float(ce.detach()), "js": float(js.detach()),
                           "contrast": float(contrast.detach()), "loss": float(loss.detach())})
            if len(traces) % 24 == 0:
                print(f"{start}/{arm}: {len(traces)}/96 "
                      f"loss={sum(x['loss'] for x in traces[-24:])/24:.3f} "
                      f"elapsed={time.perf_counter()-started:.0f}s", flush=True)
    target.mkdir()
    torch.save({n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
               target / "adapter.pt")
    (target / "TRAINING.json").write_text(json.dumps({
        "protocol_sha256": sha(ROOT / "PROTOCOL.json"),
        "source_sha256": cfg["source_checkpoint_sha256"][start],
        "checkpoint_sha256": sha(target / "adapter.pt"),
        "start": start, "arm": arm, "updates": len(traces),
        "answer_targets": len(traces)*9, "elapsed_seconds": time.perf_counter()-started,
        "traces": traces}, indent=2) + "\n")
    print(f"Saved {start}_{arm}", flush=True)


@torch.inference_mode()
def evaluate(start: str, arm: str) -> None:
    validate()
    target = ROOT / f"EVAL_{start}_{arm}.json"
    if target.exists():
        raise FileExistsError(target)
    cp = checkpoint(start, arm)
    torch.set_num_threads(2)
    tok, model = pilot.prev.load(start)
    saved = torch.load(cp, map_location="cpu", weights_only=True)
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    assert saved.keys() == params.keys()
    for n, p in params.items():
        p.data.copy_(saved[n])
    model.eval()
    items = rows()
    out = []
    for offset in range(0, len(items), 8):
        block = items[offset:offset+8]
        inputs = tok([pilot.prev.prompt(tok, r) for r in block],
                     padding=True, return_tensors="pt")
        logits = model(**inputs, use_cache=False).logits[:, -1]
        for r, z in zip(block, logits):
            answer = tok.decode([int(z.argmax())]).strip()
            out.append({"event": r["id"], "pair_index": r["pair_index"],
                        "form": r["form"], "wording": r["wording"],
                        "role": r["role"], "gold": r["gold"],
                        "answer": answer, "correct": answer == r["gold"],
                        "valid": answer in r["pair"]})
        if (offset+8) % 120 == 0:
            print(f"{start}_{arm}: {offset+8}/{len(items)}", flush=True)
    assert len(out) == 720
    target.write_text(json.dumps({"start": start, "arm": arm,
                                  "protocol_sha256": sha(ROOT / "PROTOCOL.json"),
                                  "checkpoint_sha256": sha(cp),
                                  "records": out}, indent=2) + "\n")
    print(f"Saved {target.name}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "train", "eval"))
    parser.add_argument("--start", choices=pilot.STARTS)
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare()
    elif args.start is None or args.arm is None:
        parser.error("train/eval require --start and --arm")
    elif args.mode == "train":
        train(args.start, args.arm)
    else:
        evaluate(args.start, args.arm)
