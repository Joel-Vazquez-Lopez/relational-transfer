"""Candidate-free role-binding development test on original QA-SRL questions."""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
TYPED = PROJECT / "research/typed_goal_conditioning_2026-10-13"
sys.path.insert(0, str(TYPED))
sys.path.insert(0, str(PROJECT))
spec = importlib.util.spec_from_file_location("typed_goal_development_run", TYPED / "run.py")
assert spec and spec.loader
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
import llm_role_natural_repair as bank  # noqa: E402

STARTS = ("fresh1", "fresh3")
ARMS = ("typed", "neutral")
SEED = 910214
LR = 5e-5
DEV_SOURCE = PROJECT / "llm_role_natural_repair/data/qasrl-v2/orig/dev.jsonl.gz"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dev_items() -> list[dict]:
    existing = bank.extract("dev")
    targets = {row["id"][4:]: row for row in existing}
    found = {}
    with gzip.open(DEV_SOURCE, "rt") as stream:
        for line in stream:
            record = json.loads(line)
            for verb_id, predicate in record["verbEntries"].items():
                key = f"{record['sentenceId']}:{verb_id}"
                if key not in targets:
                    continue
                target = targets[key]
                actor_q, patient_q = set(), set()
                for label in predicate["questionLabels"].values():
                    slot = label["questionSlots"]
                    if slot["wh"] != "who" or label["isPassive"] or label["isNegated"]:
                        continue
                    span = bank.accepted_span(label)
                    if span is None:
                        continue
                    if slot["subj"] == "_" and tuple(target["actor_range"]) == span:
                        actor_q.add(label["questionString"])
                    if (slot["subj"] == "someone" and slot["obj"] == "_"
                            and tuple(target["other_range"]) == span):
                        patient_q.add(label["questionString"])
                if actor_q and patient_q:
                    found[key] = {"id": target["id"], "sentence_id": record["sentenceId"],
                                  "sentence": target["sentence"], "verb": target["verb"],
                                  "actor": target["actor_span"], "patient": target["other_span"],
                                  "actor_question": sorted(actor_q)[0],
                                  "patient_question": sorted(patient_q)[0]}
    groups = defaultdict(list)
    for row in found.values():
        groups[row["sentence_id"]].append(row)
    keys = sorted(groups, key=lambda key: hashlib.sha256(("candidate-free-dev:" + key).encode()).hexdigest())
    chosen = [sorted(groups[key], key=lambda row: row["id"])[0] for key in keys[:100]]
    assert len(chosen) == 100 and len({x["sentence_id"] for x in chosen}) == 100
    train = prior.splits()["train"]
    assert not {prior.norm(x["sentence"]) for x in train}.intersection(
        prior.norm(x["sentence"]) for x in chosen)
    return chosen


def rows(item: dict) -> list[dict]:
    return [{"id": item["id"], "sentence": item["sentence"], "role": role,
             "question": item[f"{role}_question"], "gold": item[role],
             "actor": item["actor"], "patient": item["patient"]}
            for role in ("actor", "patient")]


def prompt(tok, row: dict, arm: str, predicted_role: str | None = None) -> str:
    tag = (predicted_role or row["role"]).upper() if arm == "typed" else "NONE"
    assert tag in ("ACTOR", "PATIENT", "NONE")
    user = (f"Sentence: {row['sentence']}\nRequested role: {tag}\n"
            f"Question: {row['question']}")
    return tok.apply_chat_template([
        {"role": "system", "content": "Give exactly: Actor: SPAN; Patient: SPAN; Answer: SPAN"},
        {"role": "user", "content": user},
    ], tokenize=False, add_generation_prompt=True) + "Actor:"


def batch(tok, item: dict, arm: str):
    encoded = []
    for row in rows(item):
        prefix = prompt(tok, row, arm)
        pids = tok.encode(prefix, add_special_tokens=False)
        full = tok.encode(prefix + prior.suffix(row), add_special_tokens=False)
        assert full[:len(pids)] == pids
        tokens = full + [tok.eos_token_id]
        labels = [-100] * len(pids) + full[len(pids):] + [tok.eos_token_id]
        encoded.append((tokens, labels))
    width = max(len(tokens) for tokens, _ in encoded)
    ids, ys, masks = [], [], []
    for tokens, labels in encoded:
        pad = width - len(tokens)
        ids.append([tok.pad_token_id] * pad + tokens)
        ys.append([-100] * pad + labels)
        masks.append([0] * pad + [1] * len(tokens))
    return torch.tensor(ids), torch.tensor(ys), torch.tensor(masks)


def preflight() -> None:
    train = prior.splits()["train"]
    dev = dev_items()
    tok, model = prior.prior.prior.base.prev.load("fresh1")
    torch.set_num_threads(2)
    for arm in ARMS:
        ids, labels, mask = batch(tok, train[0], arm)
        logits = model(input_ids=ids, attention_mask=mask).logits
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                               labels[:, 1:].reshape(-1), ignore_index=-100)
        assert torch.isfinite(loss)
        loss.backward()
        grad = sum(float(p.grad.detach().abs().sum()) for p in model.parameters()
                   if p.requires_grad and p.grad is not None)
        assert grad > 0
        model.zero_grad(set_to_none=True)
    print(json.dumps({"train": len(train), "dev": len(dev), "arms": ARMS,
                      "actor_before_patient": sum(row["sentence"].find(row["actor"]) <
                                                  row["sentence"].find(row["patient"]) for row in dev),
                      "finite_nonzero_gradient": True}), flush=True)


def prepare() -> None:
    assert not (ROOT / "PROTOCOL.json").exists()
    preflight()
    train = prior.splits()["train"]
    dev = dev_items()
    split = {"train": train, "dev": dev}
    (ROOT / "SPLITS.json").write_text(json.dumps(split, indent=2) + "\n")
    cfg = {
        "status": "frozen before candidate-free training and dev inference",
        "date": "2026-10-14", "phase": "exploratory development, not publication confirmation",
        "question": "Can a typed requested-role cue still improve end-to-end role answers when the model must extract participants from natural sentences rather than select supplied candidates?",
        "design": "Two matched arms (ACTOR/PATIENT cue vs neutral NONE cue), two starts, original QA-SRL Bank questions and consensus answer spans. No candidate spans in training or evaluation. Same 160 prior natural training items, two role questions per item, 160 updates, same role-first output targets and optimizer. Compare final answer and printed role-field accuracy separately.",
        "development": "100 deterministic one-predicate-per-sentence QA-SRL Bank original-dev items, sentence-disjoint from train. This source was used in earlier broad development, so the set is not independent confirmation. The automatic actor/other direct-participant mapping has no independent human review; almost all selected sentences put actor first.",
        "primary": "Both actor and patient direct final answers exact per sentence /100; also individual exact /200. Typed-minus-neutral gain >=10/100 both-correct and >=20/200 exact in both starts, without >10/200 loss in both printed role fields, is an exploratory positive screen.",
        "secondary": "Printed actor/patient exact, answer agreement with requested printed field, invalid schema, role-tag classifier accuracy. The classifier trains on training questions only. No candidate-list lines appear in prompts.",
        "limits": "This isolates a candidate-free input condition and cue comparison, not an original general method. QA-SRL labels are human question/answer annotations but our semantic mapping is automatic. Actor-first skew, one 0.5B model family, two related starts, previously developed source corpus, and unknown pretraining overlap preclude a natural-language repair claim.",
        "starts": STARTS, "arms": ARMS, "seed": SEED, "learning_rate": LR,
        "updates_per_checkpoint": len(train), "dev_items": len(dev),
        "script_sha256": sha(Path(__file__)),
        "analysis_sha256": sha(ROOT / "analyze.py"),
        "runner_sha256": sha(ROOT / "run_all.py"),
        "splits_sha256": sha(ROOT / "SPLITS.json"),
        "source_dev_sha256": sha(DEV_SOURCE),
        "prior_protocol_sha256": sha(TYPED / "PROTOCOL.json"),
        "source_adapters_sha256": {
            start: sha(prior.prior.prior.base.prev.source_path(start)) for start in STARTS},
    }
    (ROOT / "PROTOCOL.json").write_text(json.dumps(cfg, indent=2) + "\n")
    print("FROZEN", sha(ROOT / "PROTOCOL.json"), flush=True)


def validate() -> dict:
    cfg = json.loads((ROOT / "PROTOCOL.json").read_text())
    for field, path in (("script_sha256", Path(__file__)),
                        ("analysis_sha256", ROOT / "analyze.py"),
                        ("runner_sha256", ROOT / "run_all.py"),
                        ("splits_sha256", ROOT / "SPLITS.json"),
                        ("source_dev_sha256", DEV_SOURCE),
                        ("prior_protocol_sha256", TYPED / "PROTOCOL.json")):
        assert cfg[field] == sha(path), field
    for start in STARTS:
        assert cfg["source_adapters_sha256"][start] == sha(prior.prior.prior.base.prev.source_path(start))
    return cfg


def train(start: str, arm: str) -> None:
    validate()
    dest = ROOT / f"{start}_{arm}"
    dest.mkdir(exist_ok=True)
    final = dest / "adapter.pt"
    if final.exists():
        raise FileExistsError(final)
    tok, model = prior.prior.prior.base.prev.load(start)
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=LR)
    order = list(range(160))
    random.Random(SEED).shuffle(order)
    train_items = json.loads((ROOT / "SPLITS.json").read_text())["train"]
    assert len(train_items) == 160
    state_path = dest / "state.pt"
    step0, losses, target_tokens = 0, [], 0
    if state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        assert (state["start"], state["arm"], state["protocol_sha256"]) == (start, arm, sha(ROOT / "PROTOCOL.json"))
        params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        assert state["trainable"].keys() == params.keys()
        for name, param in params.items():
            param.data.copy_(state["trainable"][name])
        opt.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["rng"])
        step0, losses, target_tokens = state["step"], state["losses"], state["target_tokens"]
    begun = time.perf_counter()
    model.train()
    for step, index in enumerate(order, 1):
        if step <= step0:
            continue
        ids, labels, mask = batch(tok, train_items[index], arm)
        opt.zero_grad(set_to_none=True)
        logits = model(input_ids=ids, attention_mask=mask).logits
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                               labels[:, 1:].reshape(-1), ignore_index=-100)
        assert torch.isfinite(loss)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
        target_tokens += int((labels != -100).sum())
        if step % 40 == 0:
            torch.save({"start": start, "arm": arm, "step": step,
                        "protocol_sha256": sha(ROOT / "PROTOCOL.json"),
                        "trainable": {n: p.detach().cpu() for n, p in model.named_parameters()
                                      if p.requires_grad}, "optimizer": opt.state_dict(),
                        "rng": torch.get_rng_state(), "losses": losses,
                        "target_tokens": target_tokens}, state_path)
            print(start, arm, step, "/160", round(losses[-1], 4), flush=True)
    torch.save({n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}, final)
    (dest / "TRAINING.json").write_text(json.dumps({"start": start, "arm": arm,
        "protocol_sha256": sha(ROOT / "PROTOCOL.json"), "checkpoint_sha256": sha(final),
        "updates": 160, "target_tokens": target_tokens, "losses": losses,
        "elapsed_seconds_this_session": time.perf_counter()-begun}, indent=2) + "\n")
    print("SAVED", start, arm, flush=True)


@torch.inference_mode()
def evaluate(start: str, arm: str) -> None:
    validate()
    out = ROOT / f"EVAL_{start}_{arm}.json"
    if out.exists():
        raise FileExistsError(out)
    torch.set_num_threads(2)
    tok, model = prior.prior.prior.base.prev.load(start)
    cp = ROOT / f"{start}_{arm}" / "adapter.pt"
    weights = torch.load(cp, map_location="cpu", weights_only=True)
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    assert weights.keys() == params.keys()
    for name, param in params.items():
        param.data.copy_(weights[name])
    model.eval()
    split = json.loads((ROOT / "SPLITS.json").read_text())
    classifier = prior.classifier(split["train"])
    predictions = prior.role_predictions(split["dev"], classifier)
    all_rows = [row for item in split["dev"] for row in rows(item)]
    output = []
    for off in range(0, len(all_rows), 4):
        block = all_rows[off:off+4]
        prompts = [prompt(tok, row, arm, predictions[(row["id"], row["role"])]) for row in block]
        encoded = tok(prompts, padding=True, return_tensors="pt", add_special_tokens=False)
        generated = model.generate(**encoded, max_new_tokens=48, do_sample=False,
                                   pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        for row, tokens in zip(block, generated[:, encoded["input_ids"].shape[1]:]):
            raw = tok.decode(tokens, skip_special_tokens=True)
            actor, patient, answer = prior.parse(raw)
            field = actor if row["role"] == "actor" else patient
            output.append({"id": row["id"], "role": row["role"], "gold": row["gold"],
                           "tag": predictions[(row["id"], row["role"])],
                           "tag_correct": predictions[(row["id"], row["role"])] == row["role"],
                           "raw": raw, "actor": actor, "patient": patient, "answer": answer,
                           "parsed": answer is not None,
                           "actor_correct": actor is not None and prior.norm(actor) == prior.norm(row["actor"]),
                           "patient_correct": patient is not None and prior.norm(patient) == prior.norm(row["patient"]),
                           "correct": answer is not None and prior.norm(answer) == prior.norm(row["gold"]),
                           "routed_correct": field is not None and prior.norm(field) == prior.norm(row["gold"]),
                           "answer_matches_field": answer is not None and field is not None
                                                   and prior.norm(answer) == prior.norm(field)})
        if (off+4) % 40 == 0:
            print(start, arm, off+4, "/", len(all_rows), flush=True)
    out.write_text(json.dumps({"start": start, "arm": arm,
                               "protocol_sha256": sha(ROOT / "PROTOCOL.json"),
                               "checkpoint_sha256": sha(cp), "records": output}, indent=2) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("preflight", "prepare", "train", "eval"))
    ap.add_argument("--start", choices=STARTS)
    ap.add_argument("--arm", choices=ARMS)
    args = ap.parse_args()
    if args.mode == "preflight":
        preflight()
    elif args.mode == "prepare":
        prepare()
    elif args.mode == "train":
        assert args.start and args.arm
        train(args.start, args.arm)
    else:
        assert args.start and args.arm
        evaluate(args.start, args.arm)
