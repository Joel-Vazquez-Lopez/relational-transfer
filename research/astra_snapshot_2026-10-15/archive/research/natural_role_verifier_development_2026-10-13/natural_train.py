#!/usr/bin/env python3
"""Matched natural role-answer development pilot with imperfect-draft verification."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "role_slot_routing_fresh_confirmation_2026-10-13"))
import confirm as prior  # noqa: E402
import data  # noqa: E402

ARMS = ("answer_only", "candidate_first", "role_first", "verifier_clean", "verifier_noisy")
STARTS = ("fresh1", "fresh3")
LR = 5e-5
SEED = 910127
GOLD_DIAGNOSTIC = ROOT.parent / "role_slot_routing_qasrl_gold_2026-10-13"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def norm(value):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", (value or "").casefold())).strip()


def rows(item):
    result = []
    for role, question in (("actor", item["actor_question"]),
                           ("patient", item["patient_question"])):
        for order in (0, 1):
            pair = [item["actor"], item["patient"]]
            if order:
                pair.reverse()
            result.append({"id": item["id"], "sentence": item["sentence"],
                           "question": question, "pair": pair, "role": role, "order": order,
                           "gold": item[role], "actor": item["actor"],
                           "patient": item["patient"]})
    return result


def swapped_train_draft(row):
    digest = hashlib.sha256(("corrupt-v1:" + row["id"] + ":" + row["role"] +
                             ":" + str(row["order"])).encode()).digest()
    return digest[0] % 2 == 0


def format_prompt(tok, row, arm, draft=None):
    if arm == "answer_only":
        system, opening = "Give exactly: Answer: SPAN", "Answer:"
    elif arm == "candidate_first":
        system, opening = "Give exactly: First: SPAN; Second: SPAN; Answer: SPAN", "First:"
    else:
        system, opening = "Give exactly: Actor: SPAN; Patient: SPAN; Answer: SPAN", "Actor:"
    user = f"Sentence: {row['sentence']}\nCandidates: {', '.join(row['pair'])}\n"
    if arm.startswith("verifier_"):
        assert draft is not None
        system = "Verify the draft roles against the sentence. " + system
        user += f"Draft roles: Actor: {draft[0]}; Patient: {draft[1]}\n"
    user += f"Question: {row['question']}"
    prompt = tok.apply_chat_template([{"role": "system", "content": system},
                                      {"role": "user", "content": user}],
                                     tokenize=False, add_generation_prompt=True)
    return prompt + opening


def target(row, arm):
    if arm == "answer_only":
        return f" {row['gold']}"
    if arm == "candidate_first":
        return f" {row['pair'][0]}; Second: {row['pair'][1]}; Answer: {row['gold']}"
    return f" {row['actor']}; Patient: {row['patient']}; Answer: {row['gold']}"


def draft_for_train(row, arm):
    if arm == "verifier_clean":
        return row["actor"], row["patient"]
    if arm == "verifier_noisy":
        return ((row["patient"], row["actor"]) if swapped_train_draft(row)
                else (row["actor"], row["patient"]))
    return None


def batch(tok, item, arm):
    encoded = []
    for row in rows(item):
        prefix = format_prompt(tok, row, arm, draft_for_train(row, arm))
        prefix_ids = tok.encode(prefix, add_special_tokens=False)
        full_ids = tok.encode(prefix + target(row, arm), add_special_tokens=False)
        assert full_ids[:len(prefix_ids)] == prefix_ids, (arm, row["id"])
        tokens = full_ids + [tok.eos_token_id]
        labels = [-100] * len(prefix_ids) + full_ids[len(prefix_ids):] + [tok.eos_token_id]
        encoded.append((tokens, labels))
    width = max(len(tokens) for tokens, _ in encoded)
    ids, labels, masks = [], [], []
    for tokens, goals in encoded:
        padding = width - len(tokens)
        ids.append([tok.pad_token_id] * padding + tokens)
        labels.append([-100] * padding + goals)
        masks.append([0] * padding + [1] * len(tokens))
    return torch.tensor(ids), torch.tensor(labels), torch.tensor(masks)


def parse(raw, arm):
    if arm == "answer_only":
        answer = raw.strip().split(";")[0].strip()
        return (answer or None, None, None)
    if arm == "candidate_first":
        match = re.match(r"^\s*(.*?)\s*;\s*Second:\s*(.*?)\s*;\s*Answer:\s*(.*?)\s*$", raw, re.S)
        return (match.group(3), None, None) if match else (None, None, None)
    match = re.match(r"^\s*(.*?)\s*;\s*Patient:\s*(.*?)\s*;\s*Answer:\s*(.*?)\s*$", raw, re.S)
    return (match.group(3), match.group(1), match.group(2)) if match else (None, None, None)


def load_split():
    return json.loads((ROOT / "SPLITS.json").read_text())


def checkpoint(start, arm):
    return ROOT / f"{start}_{arm}" / "adapter.pt"


def prepare():
    protocol = ROOT / "PROTOCOL.json"
    if protocol.exists():
        raise FileExistsError(protocol)
    prior.validate()
    splits = data.split()
    tok, model = prior.base.prev.load("fresh1")
    for arm in ARMS:
        batch(tok, splits["train"][0], arm)
    del model
    (ROOT / "SPLITS.json").write_text(json.dumps(splits, indent=2) + "\n")
    corrupted = sum(swapped_train_draft(row) for item in splits["train"] for row in rows(item))
    cfg = {
        "date": "2026-10-13",
        "status": "frozen natural-data development pilot; no final publication test selected or opened",
        "question": "Does natural training with verified role drafts improve answer accuracy and robustness over answer-only, candidate-first, role-first, and clean-draft controls?",
        "data": "QA-SRL Bank 2.0 original train split, human QA annotations with automatic direct-object role mapping; 160 sentence-disjoint train items and 40 development items. Exclude every sentence in the local QA-SRL Gold corpus. Previously examined 51 QA-SRL Gold items are exploratory out-of-domain diagnostics only.",
        "starts": list(STARTS), "arms": list(ARMS),
        "prompts": "Original QA questions; both candidate orders; actor and patient requested. Four examples per item, one training update per item, one epoch, deterministic order.",
        "training": "All arms initialize from the same natural-role source adapter within each start. Role-first and both verifier arms have identical gold output targets. Verifier-clean sees gold draft roles in the user prompt; verifier-noisy sees a hash-fixed mix of correct and swapped drafts. Both must output corrected gold roles and gold answer. Candidate-first is a structured answer-last control. Answer-only has fewer target tokens; report budgets.",
        "corruption": {"method": "swap actor and patient in the draft input only, selected by SHA256 parity of item/role/order; never change gold output", "corrupted_train_rows": corrupted, "total_train_rows": len(splits["train"]) * 4},
        "evaluation": "Evaluate all arms on the 40 source-development items and the 51 already examined QA-SRL Gold exploratory items, each in both candidate orders and actor/patient questions. Verifier arms receive the role-first arm's generated draft fields for the identical item and question; malformed drafts become Unknown/Unknown. Also swap the draft fields at evaluation as a frozen stress condition. Exact span accuracy, all-four-correct, role-field accuracy, candidate-order bias, invalid outputs, and answer-draft agreement all reported. No test result is a final paper confirmation.",
        "primary_development": "All four answers correct per item on 40 source-development items. Compare verifier-noisy against verifier-clean and role-first, and direct against role-first deterministic routing; all items and starts reported.",
        "screen": "A promising development signal requires verifier-noisy to beat verifier-clean by at least 4/40 all-four-correct on source-development in EACH start, parsed rate >=80%, and no drop greater than 4/40 relative to role-first on unswapped draft inputs. This screen is only for deciding whether to pursue final human-reviewed confirmation.",
        "limits": "Automatic role mapping is not independent human verification; Gold 51 was previously examined, and this is a small-model development experiment. Two-pass verifier has extra inference compute relative to single-pass baselines. Participant identities are not disjoint; sentences are. The final sealed human-reviewed evaluation remains to be built separately.",
        "learning_rate": LR, "seed": SEED, "updates_per_checkpoint": len(splits["train"]),
        "splits_sha256": sha(ROOT / "SPLITS.json"), "data_script_sha256": sha(ROOT / "data.py"),
        "experiment_script_sha256": sha(Path(__file__)),
        "source_adapters_sha256": {s: sha(prior.base.prev.source_path(s)) for s in STARTS},
        "exploratory_gold_protocol_sha256": sha(GOLD_DIAGNOSTIC / "PROTOCOL.json"),
    }
    protocol.write_text(json.dumps(cfg, indent=2) + "\n")
    print("Frozen", len(splits["train"]), "train,", len(splits["development"]),
          "development,", corrupted, "corrupted draft rows", flush=True)


def validate():
    cfg = json.loads((ROOT / "PROTOCOL.json").read_text())
    assert cfg["splits_sha256"] == sha(ROOT / "SPLITS.json")
    assert cfg["data_script_sha256"] == sha(ROOT / "data.py")
    assert cfg["experiment_script_sha256"] == sha(Path(__file__))
    assert cfg["exploratory_gold_protocol_sha256"] == sha(GOLD_DIAGNOSTIC / "PROTOCOL.json")
    for start in STARTS:
        assert cfg["source_adapters_sha256"][start] == sha(prior.base.prev.source_path(start))
    return cfg


def train(start, arm):
    cfg = validate()
    target_dir = ROOT / f"{start}_{arm}"
    if target_dir.exists():
        raise FileExistsError(target_dir)
    target_dir.mkdir()
    torch.set_num_threads(2)
    tok, model = prior.base.prev.load(start)
    torch.manual_seed(SEED)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=LR)
    items = load_split()["train"]
    order = list(range(len(items)))
    random.Random(SEED).shuffle(order)
    began = time.perf_counter()
    losses, target_tokens = [], 0
    model.train()
    for step, index in enumerate(order, 1):
        ids, labels, mask = batch(tok, items[index], arm)
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids=ids, attention_mask=mask).logits
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                               labels[:, 1:].reshape(-1), ignore_index=-100)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        target_tokens += int((labels != -100).sum())
        if step % 20 == 0:
            print(start, arm, step, "/", len(order), "loss", round(losses[-1], 4), flush=True)
    weights = target_dir / "adapter.pt"
    torch.save({name: p.detach().cpu() for name, p in model.named_parameters() if p.requires_grad}, weights)
    (target_dir / "TRAINING.json").write_text(json.dumps({
        "start": start, "arm": arm, "protocol_sha256": sha(ROOT / "PROTOCOL.json"),
        "source_sha256": cfg["source_adapters_sha256"][start],
        "checkpoint_sha256": sha(weights), "updates": len(order),
        "examples": len(order) * 4, "target_tokens": target_tokens,
        "elapsed_seconds": time.perf_counter() - began, "losses": losses}, indent=2) + "\n")
    print("Saved", start, arm, flush=True)


def load_trained(start, arm):
    tok, model = prior.base.prev.load(start)
    saved = torch.load(checkpoint(start, arm), map_location="cpu", weights_only=True)
    params = {name: p for name, p in model.named_parameters() if p.requires_grad}
    assert saved.keys() == params.keys()
    for name, param in params.items():
        param.data.copy_(saved[name])
    model.eval()
    return tok, model


def eval_items(which):
    if which == "development":
        return load_split()["development"]
    assert which == "gold_exploratory"
    return json.loads((GOLD_DIAGNOSTIC / "ITEMS.json").read_text())


@torch.inference_mode()
def evaluate(start, arm, which, stress):
    validate()
    target_file = ROOT / f"EVAL_{start}_{arm}_{which}_{stress}.json"
    if target_file.exists():
        raise FileExistsError(target_file)
    if arm.startswith("verifier_"):
        draft_file = ROOT / f"EVAL_{start}_role_first_{which}_normal.json"
        assert draft_file.exists()
        draft_data = json.loads(draft_file.read_text())
        drafts = {(r["id"], r["role"], r["order"]): (r["actor"] or "Unknown", r["patient"] or "Unknown")
                  for r in draft_data["records"]}
    else:
        drafts = {}
        assert stress == "normal"
    torch.set_num_threads(2)
    tok, model = load_trained(start, arm)
    all_rows = [row for item in eval_items(which) for row in rows(item)]
    result = []
    for off in range(0, len(all_rows), 4):
        block = all_rows[off:off+4]
        row_drafts = []
        for row in block:
            draft = drafts.get((row["id"], row["role"], row["order"]))
            if draft is not None and stress == "swapped":
                draft = draft[::-1]
            row_drafts.append(draft)
        encoded = tok([format_prompt(tok, row, arm, draft) for row, draft in zip(block, row_drafts)],
                      padding=True, return_tensors="pt", add_special_tokens=False)
        generated = model.generate(**encoded, max_new_tokens=48, do_sample=False,
                                   pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        for row, draft, tokens in zip(block, row_drafts,
                                      generated[:, encoded["input_ids"].shape[1]:]):
            raw = tok.decode(tokens, skip_special_tokens=True)
            answer, actor, patient = parse(raw, arm)
            result.append({"id": row["id"], "role": row["role"], "order": row["order"],
                           "gold": row["gold"], "candidate_pair": row["pair"],
                           "draft": draft, "raw": raw, "answer": answer,
                           "actor": actor, "patient": patient,
                           "parsed": answer is not None,
                           "correct": answer is not None and norm(answer) == norm(row["gold"]),
                           "actor_correct": actor is not None and norm(actor) == norm(row["actor"]),
                           "patient_correct": patient is not None and norm(patient) == norm(row["patient"]),
                           "routed_correct": (norm(actor if row["role"] == "actor" else patient) == norm(row["gold"]))
                                             if arm not in ("answer_only", "candidate_first") else None})
        if (off + 4) % 80 == 0:
            print(start, arm, which, stress, off + 4, "/", len(all_rows), flush=True)
    target_file.write_text(json.dumps({"start": start, "arm": arm, "dataset": which,
                                       "stress": stress, "protocol_sha256": sha(ROOT / "PROTOCOL.json"),
                                       "checkpoint_sha256": sha(checkpoint(start, arm)),
                                       "records": result}, indent=2) + "\n")
    print("Saved", target_file.name, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "train", "eval"))
    parser.add_argument("--start", choices=STARTS)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--dataset", choices=("development", "gold_exploratory"))
    parser.add_argument("--stress", choices=("normal", "swapped"), default="normal")
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare()
    else:
        if not args.start or not args.arm:
            parser.error("train/eval require --start and --arm")
        if args.mode == "train":
            train(args.start, args.arm)
        else:
            if not args.dataset:
                parser.error("eval requires --dataset")
            evaluate(args.start, args.arm, args.dataset, args.stress)
