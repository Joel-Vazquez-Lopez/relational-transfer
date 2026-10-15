#!/usr/bin/env python3
"""Matched question-balanced SFT versus role/prompt-coupled Qwen0.5B pilot."""
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
sys.path.insert(0, str(WORK))
sys.path.insert(0, str(ROOT.parent / "role_subject_decoupling_pilot_2026-10-12"))
sys.path.insert(0, str(ROOT.parent / "role_subject_balanced_followup_2026-10-12"))
import experiment as prev  # noqa: E402
import followup as mixed  # noqa: E402

TEST_PAIRS = (("Steve", "Tom"), ("Bill", "Alex"), ("Jack", "Nick"),
              ("Chris", "Ann"), ("Ben", "Joe"), ("Mike", "Fred"))
TEST_VERBS = ("pulled", "rescued", "guided")
FORMS = ("active", "passive", "subject_cleft", "object_cleft")
TRAIN_Q = {
    "actor_performed": "Who performed the action?",
    "actor_did": "Who did the action?",
    "patient_verb": "Who was {verb}?",
}
TEST_Q = {**TRAIN_Q,
          "actor_carried": "Which person carried out the action described in the sentence?",
          "patient_which": "Which person was {verb}?"}
STARTS = prev.STARTS
ARMS = ("balanced_sft", "role_coupled")
EPOCHS = prev.EPOCHS
LR = prev.LR
SEED = prev.SEED
LAMBDA_JS = 0.5
LAMBDA_CONTRAST = 0.25


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def question_row(event: dict, form: str, qname: str, qtext: str) -> dict:
    role = qname.split("_")[0]
    return {**event, "form": form, "sentence": prev.sentence(event, form),
            "wording": qname, "role": role,
            "question": qtext.format(verb=event["verb"]), "gold": event[role]}


def test_rows() -> list[dict]:
    return [question_row(event, form, qname, qtext)
            for event in prev.events(TEST_PAIRS, TEST_VERBS)
            for form in FORMS for qname, qtext in TEST_Q.items()]


def train_rows(event: dict, forms: tuple[str, ...]) -> list[dict]:
    return [question_row(event, form, qname, qtext)
            for form in forms for qname, qtext in TRAIN_Q.items()]


def prepare() -> None:
    target = ROOT / "PROTOCOL.json"
    if target.exists():
        raise FileExistsError(target)
    prior3 = (("Alice", "Bob"), ("Bruce", "Jane"), ("David", "Amy"),
              ("Arthur", "Max"), ("Frank", "Rose"), ("Martin", "June"))
    names = [n for pair in TEST_PAIRS for n in pair]
    prior_names = {n for pair in prev.TRAIN_PAIRS + prev.TEST_PAIRS + mixed.TEST_PAIRS + prior3
                   for n in pair}
    assert len(names) == len(set(names)) == 12
    assert not set(names) & prior_names
    assert not set(TEST_VERBS) & set(prev.TRAIN_VERBS + prev.TEST_VERBS + mixed.TEST_VERBS +
                                    ("helped", "followed", "protected"))
    tok, model = prev.load("fresh1")
    for name in names:
        assert len(tok.encode(name, add_special_tokens=False)) == 1, name
        assert len(tok.encode(" " + name, add_special_tokens=False)) == 1, name
    del model
    train_events = prev.train_events()
    test = test_rows()
    assert len(train_events) == 48 and len(test) == 720
    cfg = {
        "date": "2026-10-12",
        "status": "frozen_before_training_or_test_inference; exploratory_pilot",
        "motivation": "Fresh wording audit found that prior role training improved patient answers but lost actor accuracy on a question absent from training. Test whether coupling role answers across wording and form helps beyond training on the same questions with ordinary SFT.",
        "starts": STARTS, "arms": ARMS,
        "train_pairs": prev.TRAIN_PAIRS, "train_verbs": prev.TRAIN_VERBS,
        "test_pairs": TEST_PAIRS, "test_verbs": TEST_VERBS,
        "train_question_wordings": TRAIN_Q, "test_question_wordings": TEST_Q,
        "train_forms": "For each of 48 events per epoch, active, passive, and one cleft (subject/object alternate by event index and epoch). 3 forms x 3 questions = 9 answer targets per update; 2 epochs=96 updates.",
        "test_forms": FORMS,
        "matched": "Both arms begin from the same natural-role adapter in each start, use identical event order, examples, tokens, questions, forms, 96 optimizer updates, AdamW, LR and LoRA parameters. Only the auxiliary objective differs.",
        "losses": "Balanced SFT minimizes mean first-answer-token plus EOS cross-entropy on all nine targets. Role-coupled uses the same CE plus 0.5 times within-role Jensen-Shannon dispersion of the two-candidate next-token distributions across all event forms/questions, plus 0.25 times a softplus(2-D) contrast, where D is mean actor-minus-patient candidate margin on actor questions minus that margin on patient questions. Gold CE anchors role polarity and both event directions are trained.",
        "lambda_js": LAMBDA_JS, "lambda_contrast": LAMBDA_CONTRAST,
        "primary": "On 36 new directed events x four forms =144 event-forms per checkpoint, count event-forms where BOTH held-out actor_carried and held-out patient_which questions are correct. Compare role_coupled minus balanced_sft within each start.",
        "positive_screen": "At least +8/144 held-out two-role-correct event-forms in EACH start; active actor_did, active patient_verb, and passive actor_performed each no worse than -3/36 versus balanced SFT; report all individual cells, worst wording, all-five and invalid outputs regardless. A passing pilot would need generic-margin control and natural/second-family confirmation before a repair claim.",
        "secondary": "Per-form, per-wording correct /36; held-out actor and patient joint /36 by form; all-five questions correct /36 by form; actor-question contradictions and valid output rate.",
        "limits": "Constructed English; one Qwen0.5B family and two related starts; held-out question forms were chosen after earlier Qwen observations but this lexical test is new. Only one objective weight chosen prospectively; any success remains exploratory.",
        "source_sha256": {s: sha(prev.source_path(s)) for s in STARTS},
        "prior_code_sha256": {
            "experiment": sha(prev.ROOT / "experiment.py"),
            "followup": sha(mixed.ROOT / "followup.py"),
        },
        "script_sha256": sha(Path(__file__)),
        "train_events_sha256": hashlib.sha256(json.dumps(train_events, sort_keys=True).encode()).hexdigest(),
        "test_rows_sha256": hashlib.sha256(json.dumps(test, sort_keys=True).encode()).hexdigest(),
    }
    target.write_text(json.dumps(cfg, indent=2) + "\n")
    print("Frozen 48 train events, 36 new test events, 720 prompts/checkpoint", flush=True)


def validate() -> dict:
    cfg = json.loads((ROOT / "PROTOCOL.json").read_text())
    assert cfg["script_sha256"] == sha(Path(__file__))
    assert cfg["prior_code_sha256"]["experiment"] == sha(prev.ROOT / "experiment.py")
    assert cfg["prior_code_sha256"]["followup"] == sha(mixed.ROOT / "followup.py")
    assert cfg["train_events_sha256"] == hashlib.sha256(
        json.dumps(prev.train_events(), sort_keys=True).encode()).hexdigest()
    assert cfg["test_rows_sha256"] == hashlib.sha256(
        json.dumps(test_rows(), sort_keys=True).encode()).hexdigest()
    for start in STARTS:
        assert cfg["source_sha256"][start] == sha(prev.source_path(start))
    return cfg


def batch(tok, event: dict, forms: tuple[str, ...]):
    entries = []
    for r in train_rows(event, forms):
        prefix = tok.encode(prev.prompt(tok, r), add_special_tokens=False)
        answer = tok.encode(r["gold"], add_special_tokens=False)
        assert len(answer) == 1
        entries.append((prefix + answer + [tok.eos_token_id],
                        [-100]*len(prefix) + answer + [tok.eos_token_id], len(prefix)))
    assert len(entries) == 9
    length = max(len(toks) for toks, _, _ in entries)
    ids, labels, mask, pred_pos = [], [], [], []
    for toks, target, prefix_length in entries:
        pad = length - len(toks)
        ids.append([tok.pad_token_id]*pad + toks)
        labels.append([-100]*pad + target)
        mask.append([0]*pad + [1]*len(toks))
        pred_pos.append(pad + prefix_length - 1)
    return (torch.tensor(ids), torch.tensor(labels), torch.tensor(mask),
            torch.tensor(pred_pos))


def coupled_loss(logits: torch.Tensor, positions: torch.Tensor,
                 pair_ids: tuple[int, int], actor_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    row_indices = torch.arange(9)
    z = logits[row_indices, positions][:, list(pair_ids)]
    probs = F.softmax(z, dim=-1)
    actor_rows = torch.tensor([0, 1, 3, 4, 6, 7])
    patient_rows = torch.tensor([2, 5, 8])
    actor_probs = probs[actor_rows]
    patient_probs = probs[patient_rows]
    def dispersion(group: torch.Tensor) -> torch.Tensor:
        mean = group.mean(dim=0).clamp_min(1e-8)
        return (group * (group.clamp_min(1e-8).log() - mean.log())).sum(dim=-1).mean()
    js = dispersion(actor_probs) + dispersion(patient_probs)
    patient_index = 1 - actor_index
    actor_margin = (z[actor_rows, actor_index] - z[actor_rows, patient_index]).mean()
    patient_margin = (z[patient_rows, actor_index] - z[patient_rows, patient_index]).mean()
    contrast = F.softplus(2.0 - (actor_margin - patient_margin))
    return js, contrast


def train(start: str, arm: str) -> None:
    cfg = validate()
    target = ROOT / f"{start}_{arm}"
    if target.exists():
        raise FileExistsError(target)
    torch.set_num_threads(2)
    tok, model = prev.load(start)
    torch.manual_seed(SEED)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=LR)
    events = prev.train_events()
    order = list(range(len(events)))
    rng = random.Random(SEED)
    traces = []
    started = time.perf_counter()
    model.train()
    for epoch in range(EPOCHS):
        rng.shuffle(order)
        for idx in order:
            event = events[idx]
            forms = mixed.mixed_forms(idx, epoch)
            ids, labels, mask, positions = batch(tok, event, forms)
            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
            token_losses = F.cross_entropy(logits[:, :-1].contiguous().flatten(0, 1),
                                           labels[:, 1:].contiguous().flatten(),
                                           ignore_index=-100, reduction="none").view(9, -1)
            counts = (labels[:, 1:] != -100).sum(dim=1)
            ce = (token_losses.sum(dim=1) / counts).mean()
            js = contrast = torch.zeros((), dtype=ce.dtype)
            if arm == "role_coupled":
                pair_ids = tuple(tok.encode(n, add_special_tokens=False)[0] for n in event["pair"])
                actor_index = event["pair"].index(event["actor"])
                js, contrast = coupled_loss(logits, positions, pair_ids, actor_index)
            loss = ce + LAMBDA_JS*js + LAMBDA_CONTRAST*contrast
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
        "source_sha256": cfg["source_sha256"][start],
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
    cp = ROOT / f"{start}_{arm}" / "adapter.pt"
    torch.set_num_threads(2)
    tok, model = prev.load(start)
    saved = torch.load(cp, map_location="cpu", weights_only=True)
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    assert saved.keys() == params.keys()
    for n, p in params.items():
        p.data.copy_(saved[n])
    model.eval()
    items = test_rows()
    out = []
    for offset in range(0, len(items), 8):
        block = items[offset:offset+8]
        prompts = [prev.prompt(tok, r) for r in block]
        inputs = tok(prompts, padding=True, return_tensors="pt")
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
    parser.add_argument("--start", choices=STARTS)
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
