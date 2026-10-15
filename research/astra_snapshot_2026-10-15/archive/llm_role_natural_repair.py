"""Exploratory, source-separated natural-role adaptation of the local Qwen pilot."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

import llm_role_repair_confirm as confirm
import llm_role_repair_v3 as core
from llm_role_natural_contrast_screen import prompt as natural_prompt, score


HERE = Path(__file__).resolve().parent
ROOT = HERE / "llm_role_natural_repair"
BANK = ROOT / "data/qasrl-v2/orig"
GOLD = HERE / "llm_role_natural_contrast_confirm/EVENTS.json"
SOURCE_ADAPTER = HERE / "llm_role_unordered_control/fresh1_830211_unordered_pair/adapter.pt"
SYNTHETIC = HERE / "llm_role_repair_confirmation/TEST_fresh1.json"
TRAIN_N = 256
SEED = 710261
LR = 5e-5
REPLAY_WEIGHT = 0.3


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def accepted_span(label: dict) -> tuple[int, int] | None:
    votes = [tuple(j["spans"][0]) for j in label["answerJudgments"]
             if j["isValid"] and len(j["spans"]) == 1]
    if len(votes) < 2:
        return None
    span, n = Counter(votes).most_common(1)[0]
    return span if n >= 2 else None


def extract(split: str) -> list[dict]:
    rows = []
    with gzip.open(BANK / f"{split}.jsonl.gz", "rt") as stream:
        for line in stream:
            item = json.loads(line)
            tokens = item["sentenceTokens"]
            if not 5 <= len(tokens) <= 45:
                continue
            for verb_id, verb_entry in item["verbEntries"].items():
                actor, other = set(), set()
                for label in verb_entry["questionLabels"].values():
                    slots = label["questionSlots"]
                    if slots["wh"] != "who" or label["isPassive"] or label["isNegated"]:
                        continue
                    span = accepted_span(label)
                    if span is None or not 1 <= span[1] - span[0] <= 8:
                        continue
                    if slots["subj"] == "_":
                        actor.add(span)
                    elif slots["subj"] == "someone" and slots["obj"] == "_":
                        other.add(span)
                if len(actor) != 1 or len(other) != 1:
                    continue
                a, b = next(iter(actor)), next(iter(other))
                if a == b or max(a[0], b[0]) < min(a[1], b[1]):
                    continue
                actor_text = " ".join(tokens[a[0]:a[1]])
                other_text = " ".join(tokens[b[0]:b[1]])
                if actor_text.casefold() == other_text.casefold():
                    continue
                rows.append({"id": f"{split}:{item['sentenceId']}:{verb_id}",
                             "sentence": " ".join(tokens),
                             "verb": tokens[int(verb_id)],
                             "actor_span": actor_text, "other_span": other_text,
                             "actor_range": list(a), "other_range": list(b)})
    return rows


def prepare() -> None:
    ROOT.mkdir(exist_ok=True)
    if (ROOT / "PROTOCOL.json").exists():
        raise FileExistsError(ROOT / "PROTOCOL.json")
    all_train = extract("train")
    all_dev = extract("dev")
    gold = json.loads(GOLD.read_text())
    gold_sentences = {x["sentence"].casefold().strip() for x in gold}
    gold_docs = {":".join(x["id"].split(":")[2:4]) for x in gold}
    def doc_id(row: dict) -> str | None:
        parts = row["id"].split(":")
        return ":".join(parts[2:4]) if parts[1] == "Wiki1k" else None
    assert all(x["sentence"].casefold().strip() not in gold_sentences for x in all_train)
    assert not {doc_id(x) for x in all_train}.intersection(gold_docs)
    all_dev = [x for x in all_dev if doc_id(x) not in gold_docs and
               x["sentence"].casefold().strip() not in gold_sentences]
    assert all(x["id"].startswith(("train:", "dev:")) for x in all_train + all_dev)
    train = sorted(all_train, key=lambda r: hashlib.sha256(f"{SEED}:{r['id']}".encode()).hexdigest())[:TRAIN_N]
    assert len(train) == TRAIN_N
    assert not {r["id"].split(":", 2)[2] for r in train}.intersection(
        r["id"].split(":", 2)[2] for r in all_dev)
    (ROOT / "TRAIN.json").write_text(json.dumps(train, indent=2) + "\n")
    (ROOT / "DEV.json").write_text(json.dumps(all_dev, indent=2) + "\n")
    protocol = {
        "status": "fixed_before_training", "date": "2026-10-11",
        "purpose": "Screen whether authentic role-supervised examples plus synthetic replay improve the previously weak authentic contrast while retaining synthetic cleft transfer.",
        "design": "Exploratory continuation of one Qwen2.5-0.5B-Instruct participant-only adapter (fresh1, seed 830211). Use 256 original QA-SRL Bank 2.0 train events. Evaluate on separate Bank dev events, 51 previously inspected QA-SRL Gold events, and the original frozen synthetic held-out-pair test. One seed and a previously inspected Gold set cannot establish a publication-level repair.",
        "data_rule": "Active, non-negated WHO subject and direct-object questions for the same verb; each answer span receives at least two agreeing valid judgments; unique, non-overlapping spans up to eight tokens; sentence 5-45 tokens. Bank train only; Gold and Bank dev excluded; exact normalized-sentence overlap and Wiki1k article-ID overlap with Gold are zero. Deterministic hash sample of 256 train events. Do not select on Gold results.",
        "training": "One epoch of 256 updates at LR 5e-5. Each step uses both fixed actor and other-participant questions with gold span answers. Synthetic replay uses the same original four-example group, with 0.3 times its language-model loss; replay groups cycle through a seeded permutation. Source adapter and all data/checkpoint hashes are fixed here. No new synthetic clefts are trained.",
        "primary": "Candidate-relative raw-sum role contrast D>0 on the 51 Gold events, reported beside untouched 0.5B and source participant-only adapter. This is an exploratory screen because these Gold events were examined in prior experiments. A useful signal requires at least +5/51 correct over the source adapter and no drop of more than 5 percentage points on patient clefts or familiar synthetic forms. Also report sentence-omitted contrast, mean margins, and direct-span exact answers to detect prompt-format gains.",
        "secondary": "TQA dev contrast and exact actor/other answers; frozen synthetic patient-cleft, active and passive accuracy. This is a test of natural-data adaptation with rehearsal, not an isolated proof of role-specific objective or novelty.",
        "references": {"bank_train_sha256": sha(BANK / "train.jsonl.gz"),
                       "bank_dev_sha256": sha(BANK / "dev.jsonl.gz"),
                       "gold_events_sha256": sha(GOLD),
                       "source_adapter_sha256": sha(SOURCE_ADAPTER),
                       "synthetic_test_sha256": sha(SYNTHETIC),
                       "source_code_sha256": sha(HERE / "llm_role_natural_repair.py")},
        "train_count": len(train), "dev_count": len(all_dev),
        "all_qualifying_bank_train": len(all_train), "seed": SEED,
        "lr": LR, "replay_weight": REPLAY_WEIGHT,
    }
    (ROOT / "PROTOCOL.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(f"Frozen {len(train)} train, {len(all_dev)} dev; {len(all_train)} qualifying Bank train events", flush=True)


def validate() -> tuple[list[dict], list[dict], dict]:
    p = json.loads((ROOT / "PROTOCOL.json").read_text())
    for label, path in (("bank_train_sha256", BANK / "train.jsonl.gz"),
                        ("bank_dev_sha256", BANK / "dev.jsonl.gz"),
                        ("gold_events_sha256", GOLD),
                        ("source_adapter_sha256", SOURCE_ADAPTER),
                        ("synthetic_test_sha256", SYNTHETIC),
                        ("source_code_sha256", HERE / "llm_role_natural_repair.py")):
        assert p["references"][label] == sha(path), label
    train = json.loads((ROOT / "TRAIN.json").read_text())
    dev = json.loads((ROOT / "DEV.json").read_text())
    assert len(train) == TRAIN_N and train == sorted(extract("train"), key=lambda r: hashlib.sha256(f"{SEED}:{r['id']}".encode()).hexdigest())[:TRAIN_N]
    gold = json.loads(GOLD.read_text())
    gold_sentences = {x["sentence"].casefold().strip() for x in gold}
    gold_docs = {":".join(x["id"].split(":")[2:4]) for x in gold}
    def excluded(row: dict) -> bool:
        parts = row["id"].split(":")
        return (row["sentence"].casefold().strip() in gold_sentences or
                (parts[1] == "Wiki1k" and ":".join(parts[2:4]) in gold_docs))
    assert dev == [x for x in extract("dev") if not excluded(x)]
    return train, dev, p


def load_source():
    confirm.configure("fresh1", 830211)
    tok, model, _ = core.load("ordinary")
    saved = torch.load(SOURCE_ADAPTER, map_location="cpu", weights_only=True)["adapter"]
    params = {name: value for name, value in model.named_parameters() if value.requires_grad}
    assert saved.keys() == params.keys()
    for name, param in params.items():
        param.data.copy_(saved[name])
    return tok, model


def encode_answer(tok, prefix: str, answer: str):
    p = tok.encode(prefix, add_special_tokens=False)
    a = tok.encode(answer, add_special_tokens=False)
    assert p and a and len(a) <= 16
    return p + a + [tok.eos_token_id], [-100] * len(p) + a + [tok.eos_token_id]


def batch(tok, event: dict, group: list[dict]):
    natural = [encode_answer(tok, natural_prompt(tok, event, role), event[f"{role}_span"])
               for role in ("actor", "other")]
    replay = [encode_answer(tok, core.prompt(tok, item), item["gold"]) for item in group]
    entries = natural + replay
    length = max(len(x[0]) for x in entries)
    tokens, labels, masks = [], [], []
    for ids, target in entries:
        pad = length - len(ids)
        tokens.append([tok.pad_token_id] * pad + ids)
        labels.append([-100] * pad + target)
        masks.append([0] * pad + [1] * len(ids))
    return torch.tensor(tokens), torch.tensor(labels), torch.tensor(masks)


def run() -> None:
    train, _, p = validate()
    out = ROOT / "mixed_replay"
    if out.exists():
        raise FileExistsError(out)
    tok, model = load_source()
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    optimizer = torch.optim.AdamW((v for v in model.parameters() if v.requires_grad), lr=LR)
    groups, _ = confirm.validate("fresh1")
    confirm.configure("fresh1", 830211)
    rng = np.random.default_rng(SEED)
    replay_order = [int(i) for i in rng.permutation(len(groups))]
    losses = []
    started = time.perf_counter()
    model.train()
    for step, event in enumerate(train, 1):
        if (step - 1) % len(groups) == 0 and step > 1:
            replay_order = [int(i) for i in rng.permutation(len(groups))]
        group = groups[replay_order[(step - 1) % len(groups)]]
        ids, labels, mask = batch(tok, event, group)
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        shifted_logits = logits[:, :-1].contiguous()
        shifted_labels = labels[:, 1:].contiguous()
        loss_tokens = F.cross_entropy(shifted_logits.flatten(0, 1), shifted_labels.flatten(),
                                      ignore_index=-100, reduction="none").view(6, -1)
        counts = (shifted_labels != -100).sum(dim=1)
        losses_per_item = loss_tokens.sum(dim=1) / counts
        natural_loss = losses_per_item[:2].mean()
        replay_loss = losses_per_item[2:].mean()
        total = natural_loss + REPLAY_WEIGHT * replay_loss
        if not torch.isfinite(total):
            raise FloatingPointError(step)
        total.backward()
        optimizer.step()
        losses.append({"step": step, "natural": float(natural_loss.detach()),
                       "replay": float(replay_loss.detach()), "total": float(total.detach())})
        if step % 32 == 0:
            print(f"training {step}/{len(train)} elapsed {time.perf_counter()-started:.0f}s; natural {np.mean([x['natural'] for x in losses[-32:]]):.3f}; replay {np.mean([x['replay'] for x in losses[-32:]]):.3f}", flush=True)
    out.mkdir()
    torch.save({name: value.detach().cpu() for name, value in model.named_parameters() if value.requires_grad}, out / "adapter.pt")
    (out / "TRAINING.json").write_text(json.dumps({"protocol_sha256": sha(ROOT / "PROTOCOL.json"),
                                                    "elapsed_seconds": time.perf_counter() - started,
                                                    "updates": len(train), "losses": losses}, indent=2) + "\n")
    print("Training complete", flush=True)


@torch.inference_mode()
def evaluate() -> None:
    _, dev, _ = validate()
    out = ROOT / "mixed_replay"
    target = out / "EVALUATION.json"
    if target.exists():
        raise FileExistsError(target)
    tok, model = load_source()
    weights = torch.load(out / "adapter.pt", map_location="cpu", weights_only=True)
    params = {name: value for name, value in model.named_parameters() if value.requires_grad}
    assert weights.keys() == params.keys()
    for name, param in params.items():
        param.data.copy_(weights[name])
    model.eval()

    def contrast(events, label):
        records = []
        for i, event in enumerate(events, 1):
            scores = {role: {candidate: score(tok, model, natural_prompt(tok, event, role),
                                              event[f"{candidate}_span"])[0]
                             for candidate in ("actor", "other")}
                      for role in ("actor", "other")}
            d = (scores["actor"]["actor"] - scores["actor"]["other"]
                 - scores["other"]["actor"] + scores["other"]["other"])
            records.append({"id": event["id"], "margin": d, "correct": d > 0})
            if i % 25 == 0:
                print(f"{label} {i}/{len(events)}", flush=True)
        return records

    gold = json.loads(GOLD.read_text())
    natural = contrast(gold, "gold")
    development = contrast(dev, "dev")
    synthetic = json.loads(SYNTHETIC.read_text())
    selected = [r for r in synthetic if r["form"] in ("patient_cleft", "active", "passive")]
    responses = core.evaluate("mixed_replay", tok, model, selected)
    cells = {form: {"correct": sum(r["correct"] for r in responses if r["form"] == form),
                    "total": sum(r["form"] == form for r in responses)}
             for form in ("patient_cleft", "active", "passive")}
    result = {"protocol_sha256": sha(ROOT / "PROTOCOL.json"),
              "adapter_sha256": sha(out / "adapter.pt"),
              "gold": natural, "dev": development, "synthetic": cells,
              "gold_correct": sum(r["correct"] for r in natural),
              "dev_correct": sum(r["correct"] for r in development)}
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"gold": [result["gold_correct"], len(natural)],
                      "dev": [result["dev_correct"], len(development)],
                      "synthetic": cells}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "validate", "run", "evaluate"))
    args = parser.parse_args()
    if args.action == "prepare": prepare()
    elif args.action == "validate": validate(); print("validated")
    elif args.action == "run": run()
    else: evaluate()
