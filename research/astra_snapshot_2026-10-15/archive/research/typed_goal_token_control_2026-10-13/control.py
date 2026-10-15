#!/usr/bin/env python3
"""Token-matched null control and new-sentence confirmation for typed role input."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
TYPED = ROOT.parent / "typed_goal_conditioning_2026-10-13"
OLDER = ROOT.parent / "goal_first_natural_routing_2026-10-13"
spec = importlib.util.spec_from_file_location("typed_goal_source", TYPED / "run.py")
typed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(typed)

STARTS = ("fresh1", "fresh3")
ARMS = ("typed_goal", "null_two_token")
SEED = typed.SEED


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def fresh_bank():
    old = load(OLDER / "SPLITS.json")
    used = {x["sentence_id"] for rows in old.values() for x in rows}
    used.update(x["sentence_id"] for x in load(TYPED / "SPLITS.json")["fresh2"])
    groups = {}
    for item in typed.bank.extract():
        if item["sentence_id"] not in used:
            groups.setdefault(item["sentence_id"], []).append(item)
    order = sorted(groups, key=lambda k: hashlib.sha256(("typed-confirm-20261013:" + k).encode()).hexdigest())
    items = [groups[k][0] for k in order[:20]]
    assert len(items) == len({x["sentence_id"] for x in items}) == 20
    return items


def null_prompt(tok, row):
    system = "Give exactly: Actor: SPAN; Patient: SPAN; Answer: SPAN"
    user = (f"Sentence: {row['sentence']}\nCandidates: {', '.join(row['pair'])}\n"
            f"Requested role: NO ROLE\nQuestion: {row['question']}")
    return tok.apply_chat_template([{"role": "system", "content": system},
                                    {"role": "user", "content": user}],
                                   tokenize=False, add_generation_prompt=True) + "Actor:"


def null_batch(tok, item):
    entries = []
    for row in typed.prior.rows(item):
        p = null_prompt(tok, row)
        pids = tok.encode(p, add_special_tokens=False)
        full = tok.encode(p + typed.suffix(row), add_special_tokens=False)
        assert full[:len(pids)] == pids
        tokens = full + [tok.eos_token_id]
        labels = [-100] * len(pids) + full[len(pids):] + [tok.eos_token_id]
        entries.append((tokens, labels))
    width = max(len(t) for t, _ in entries)
    ids, ys, masks = [], [], []
    for tokens, labels in entries:
        pad = width - len(tokens)
        ids.append([tok.pad_token_id] * pad + tokens)
        ys.append([-100] * pad + labels)
        masks.append([0] * pad + [1] * len(tokens))
    return torch.tensor(ids), torch.tensor(ys), torch.tensor(masks)


def prepare():
    if (ROOT / "PROTOCOL.json").exists():
        raise FileExistsError("Protocol already frozen")
    typed.validate()
    rows = {"train": typed.splits()["train"], "fresh3": fresh_bank()}
    (ROOT / "SPLITS.json").write_text(json.dumps(rows, indent=2) + "\n")
    tok, model = typed.prior.prior.base.prev.load("fresh1")
    token_counts = {}
    for arm in ARMS:
        attention_tokens = targets = 0
        for item in rows["train"]:
            _, y, mask = typed.batch(tok, item, "typed_goal") if arm == "typed_goal" else null_batch(tok, item)
            attention_tokens += int(mask.sum())
            targets += int((y != -100).sum())
        token_counts[arm] = {"nonpad_input_plus_target_tokens": attention_tokens, "supervised_target_tokens": targets}
    assert token_counts["typed_goal"] == token_counts["null_two_token"]
    del model
    cfg = {
        "date": "2026-10-13", "status": "post hoc token-control design, frozen before training or any fresh3 inference; earlier typed checkpoint fixed",
        "question": "Does typed requested-role conditioning beat a null two-token tag with exactly matched input and output token counts on new natural sentences?",
        "starts": STARTS, "arms": ARMS,
        "data": "Same 160 QA-SRL Bank training items as prior typed study; 20 new sentence-disjoint items excluded from all earlier local banks. Automatic actor/patient mapping remains unreviewed.",
        "comparison": "Reuse the already trained typed_goal checkpoint. Train a fresh null_two_token control from the same paired source adapter and item order, identical Actor/Patient/Answer output targets, 160 updates, optimizer, and target tokens. Null prompt has 'Requested role: NO ROLE', two Qwen tokens, exactly matching ACTOR/PATIENT token count in all train prompts. At inference typed gets the same fixed question-only classifier prediction as before. Both receive the same sentence, original QA-SRL question, candidates, and decoding.",
        "token_counts": token_counts,
        "primary": "All four answers correct per fresh3 item, exact/80, both printed fields correct, answer-field agreement, and classifier role accuracy. Compare typed_goal with null_two_token in each start.",
        "screen": "Typed_goal gains >=4/20 all-four items and >=10/80 exact answers in each start, with no loss >4/80 in both-correct role fields; classifier >=38/40 correct roles. This is an exploratory independent-item confirmation of the matched effect, not a new model-family replication.",
        "limits": "The null control was designed after seeing the first positive result. Fresh3 examples were not inspected before this freeze. One model family, related starts, original-train QA-SRL source distribution, automatic semantic role mapping, candidate support, and templatic questions. Same 160 training items as all prior natural pilots. No causal claim about internal states or broad LLM reasoning.",
        "seed": SEED, "lr": typed.LR, "updates_per_checkpoint": len(rows["train"]),
        "splits_sha256": sha(ROOT / "SPLITS.json"),
        "script_sha256": sha(Path(__file__)),
        "typed_protocol_sha256": sha(TYPED / "PROTOCOL.json"),
        "typed_script_sha256": sha(TYPED / "run.py"),
        "typed_checkpoint_sha256": {s: sha(TYPED / f"{s}_typed_goal" / "adapter.pt") for s in STARTS},
        "source_adapters_sha256": {s: sha(typed.prior.prior.base.prev.source_path(s)) for s in STARTS},
    }
    (ROOT / "PROTOCOL.json").write_text(json.dumps(cfg, indent=2) + "\n")
    print("Frozen", sha(ROOT / "PROTOCOL.json"), "counts", token_counts, flush=True)


def validate():
    cfg = load(ROOT / "PROTOCOL.json")
    for key, path in (("splits_sha256", ROOT / "SPLITS.json"),
                      ("script_sha256", Path(__file__)),
                      ("typed_protocol_sha256", TYPED / "PROTOCOL.json"),
                      ("typed_script_sha256", TYPED / "run.py")):
        assert cfg[key] == sha(path), key
    for s in STARTS:
        assert cfg["typed_checkpoint_sha256"][s] == sha(TYPED / f"{s}_typed_goal" / "adapter.pt")
        assert cfg["source_adapters_sha256"][s] == sha(typed.prior.prior.base.prev.source_path(s))
    return cfg


def train(start):
    cfg = validate()
    dest = ROOT / f"{start}_null_two_token"
    if dest.exists():
        raise FileExistsError(dest)
    torch.set_num_threads(2)
    tok, model = typed.prior.prior.base.prev.load(start)
    torch.manual_seed(SEED)
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=typed.LR)
    items = load(ROOT / "SPLITS.json")["train"]
    order = list(range(len(items)))
    random.Random(SEED).shuffle(order)
    losses, targets = [], 0
    begun = time.perf_counter()
    model.train()
    for step, idx in enumerate(order, 1):
        x, y, mask = null_batch(tok, items[idx])
        opt.zero_grad(set_to_none=True)
        logits = model(input_ids=x, attention_mask=mask).logits
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                               y[:, 1:].reshape(-1), ignore_index=-100)
        assert torch.isfinite(loss)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
        targets += int((y != -100).sum())
        if step % 20 == 0:
            print(start, "null_two_token", step, "/", len(order), round(losses[-1], 4), flush=True)
    assert targets == cfg["token_counts"]["null_two_token"]["supervised_target_tokens"]
    dest.mkdir()
    cp = dest / "adapter.pt"
    torch.save({n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}, cp)
    (dest / "TRAINING.json").write_text(json.dumps({"start": start, "arm": "null_two_token",
        "protocol_sha256": sha(ROOT / "PROTOCOL.json"), "checkpoint_sha256": sha(cp),
        "updates": len(order), "target_tokens": targets, "elapsed_seconds": time.perf_counter()-begun,
        "losses": losses}, indent=2) + "\n")
    print("Saved", start, "null_two_token", flush=True)


@torch.inference_mode()
def evaluate(start, arm):
    validate()
    dest = ROOT / f"EVAL_{start}_{arm}_fresh3.json"
    if dest.exists():
        raise FileExistsError(dest)
    torch.set_num_threads(2)
    tok, model = typed.prior.prior.base.prev.load(start)
    cp = (TYPED / f"{start}_typed_goal" / "adapter.pt" if arm == "typed_goal"
          else ROOT / f"{start}_null_two_token" / "adapter.pt")
    saved = torch.load(cp, map_location="cpu", weights_only=True)
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    assert saved.keys() == params.keys()
    for n, p in params.items():
        p.data.copy_(saved[n])
    model.eval()
    bank = load(ROOT / "SPLITS.json")
    predictor = typed.classifier(bank["train"])
    predicted = typed.role_predictions(bank["fresh3"], predictor)
    rows = [row for item in bank["fresh3"] for row in typed.prior.rows(item)]
    records = []
    for off in range(0, len(rows), 4):
        block = rows[off:off+4]
        prompts = [(typed.prompt(tok, row, "typed_goal", predicted[(row["id"], row["role"])])
                    if arm == "typed_goal" else null_prompt(tok, row)) for row in block]
        encoded = tok(prompts, padding=True, return_tensors="pt", add_special_tokens=False)
        generated = model.generate(**encoded, max_new_tokens=48, do_sample=False,
                                   pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        for row, tokens in zip(block, generated[:, encoded["input_ids"].shape[1]:]):
            raw = tok.decode(tokens, skip_special_tokens=True)
            actor, patient, answer = typed.parse(raw)
            field = actor if row["role"] == "actor" else patient
            records.append({"id": row["id"], "role": row["role"], "order": row["order"],
                            "gold": row["gold"], "tag": predicted[(row["id"], row["role"])],
                            "tag_correct": predicted[(row["id"], row["role"])] == row["role"],
                            "raw": raw, "actor": actor, "patient": patient, "answer": answer,
                            "parsed": answer is not None,
                            "actor_correct": actor is not None and typed.norm(actor) == typed.norm(row["actor"]),
                            "patient_correct": patient is not None and typed.norm(patient) == typed.norm(row["patient"]),
                            "correct": answer is not None and typed.norm(answer) == typed.norm(row["gold"]),
                            "routed_correct": field is not None and typed.norm(field) == typed.norm(row["gold"]),
                            "answer_matches_field": answer is not None and field is not None and typed.norm(answer) == typed.norm(field)})
        if (off+4) % 40 == 0:
            print(start, arm, off+4, "/", len(rows), flush=True)
    dest.write_text(json.dumps({"start": start, "arm": arm, "protocol_sha256": sha(ROOT / "PROTOCOL.json"),
                                "checkpoint_sha256": sha(cp), "records": records}, indent=2) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("prepare", "train", "eval"))
    ap.add_argument("--start", choices=STARTS)
    ap.add_argument("--arm", choices=ARMS)
    args = ap.parse_args()
    if args.mode == "prepare":
        prepare()
    elif args.mode == "train":
        if not args.start:
            ap.error("train needs --start")
        train(args.start)
    else:
        if not args.start or not args.arm:
            ap.error("eval needs --start and --arm")
        evaluate(args.start, args.arm)
