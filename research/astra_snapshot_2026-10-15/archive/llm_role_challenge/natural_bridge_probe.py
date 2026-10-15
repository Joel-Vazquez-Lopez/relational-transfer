"""Exploratory natural-syntax probe on a frozen pretrained Transformer.

Only UD EWT train/dev are read. The reserved active/passive human-review challenge
and EWT test are deliberately excluded.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "natural_bridge_probe"
DATA = ROOT / "data"
MODEL = Path.home() / ".cache/huggingface/hub/models--distilbert-base-uncased/snapshots/12040accade4e8a0f71eabdb258fecc2e7e948be"
SEEDS = (17, 29, 43)
MAX_TOKENS = 45
MODEL_MAX = 96


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse(path: Path) -> list[dict]:
    rows = []
    for block in path.read_text().split("\n\n"):
        lines = block.splitlines()
        sent_id = next((s[12:] for s in lines if s.startswith("# sent_id = ")), None)
        toks = []
        for line in lines:
            fields = line.split("\t")
            if len(fields) != 10 or not fields[0].isdigit() or not fields[6].isdigit():
                continue
            toks.append({"id": int(fields[0]), "form": fields[1], "lemma": fields[2],
                         "upos": fields[3], "head": int(fields[6]), "dep": fields[7]})
        if not toks or len(toks) > MAX_TOKENS or not sent_id:
            continue
        by_id = {t["id"]: t for t in toks}
        if len(by_id) != len(toks):
            continue
        for verb in toks:
            if verb["upos"] != "VERB":
                continue
            children = [t for t in toks if t["head"] == verb["id"]]
            deps: dict[str, list[dict]] = {}
            for child in children:
                deps.setdefault(child["dep"], []).append(child)
            voice = None
            if len(deps.get("nsubj", [])) == len(deps.get("obj", [])) == 1 and not deps.get("nsubj:pass"):
                voice = "active"
                actor, other = deps["nsubj"][0], deps["obj"][0]
            elif len(deps.get("nsubj:pass", [])) == len(deps.get("obl:agent", [])) == 1:
                voice = "passive"
                actor, other = deps["obl:agent"][0], deps["nsubj:pass"][0]
                # A by-phrase marks the explicit agent in this restricted probe.
                if not any(x["head"] == actor["id"] and x["form"].casefold() == "by" for x in toks):
                    continue
            if voice is None:
                continue
            indexes = {t["id"]: i for i, t in enumerate(toks)}
            a, b, v = indexes[actor["id"]], indexes[other["id"]], indexes[verb["id"]]
            if len({a, b, v}) != 3:
                continue
            ordered = sorted((a, b))
            rows.append({"id": f"{sent_id}:{verb['id']}", "tokens": [t["form"] for t in toks],
                         "verb_index": v, "verb_lemma": verb["lemma"],
                         "first_index": ordered[0], "second_index": ordered[1],
                         "actor_is_first": int(a == ordered[0]), "voice": voice,
                         "actor_head": actor["form"], "other_head": other["form"]})
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids)), "Target-verb IDs must be unique"
    return rows


def select(rows: list[dict], n: int, seed: int) -> list[dict]:
    return sorted(rows, key=lambda x: hashlib.sha256(f"{seed}:{x['id']}".encode()).hexdigest())[:n]


def prepare() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    train_path = DATA / "en_ewt-ud-train.conllu"
    dev_path = DATA / "en_ewt-ud-dev.conllu"
    train, dev = parse(train_path), parse(dev_path)
    ta, tp = ([x for x in train if x["voice"] == v] for v in ("active", "passive"))
    da, dp = ([x for x in dev if x["voice"] == v] for v in ("active", "passive"))
    n = len(tp)
    assert n >= 100 and len(ta) >= 2 * n and len(dp) >= 10
    assert not {x["id"] for x in train} & {x["id"] for x in dev}
    manifest = {"status": "frozen_before_model_inference", "date": "2026-10-12",
                "model": str(MODEL), "model_config_sha256": sha(MODEL / "config.json"),
                "train_sha256": sha(train_path), "dev_sha256": sha(dev_path),
                "source_code_sha256": sha(Path(__file__)),
                "counts": {"train_active": len(ta), "train_passive_agent": n,
                           "dev_active": len(da), "dev_passive_agent": len(dp)},
                "design": "Frozen DistilBERT encoder plus L2 logistic probe. Active-only uses 2N active examples. Balanced uses N active plus all N explicit-agent passives. N is the passive train count. Three fixed active-subsampling seeds. Same total training examples. EWT dev only for evaluation; no selection on outcomes.",
                "features": "Contextual first-minus-second candidate head states, their elementwise interactions with target-verb state, and two relative-position scalars. No gold voice or dependency labels are provided to the probe.",
                "primary": "Passive-dev actor-head choice accuracy: balanced minus active-only, each averaged equally over three fixed seeds. Also report active-dev accuracy and first-participant baseline. No significance claim from the small passive sample.",
                "limitations": "Syntactic dependency labels are proxies for semantic roles. This probe predicts which of two gold candidate heads is actor; it does not generate spans, train the Transformer weights, or test the reserved human-review challenge. Active/passive counts and labels should be manually checked before using as paper evidence.",
                "seeds": list(SEEDS), "n_per_voice": n,
                "sample_ids": {str(seed): {
                    "active_only": [x["id"] for x in select(ta, 2 * n, seed)],
                    "balanced_active": [x["id"] for x in select(ta, n, seed)]}
                    for seed in SEEDS},
                "dev_ids": [x["id"] for x in dev]}
    OUT.mkdir()
    (OUT / "PROTOCOL.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"protocol": str(OUT / "PROTOCOL.json"), "counts": manifest["counts"]}))


def load() -> tuple[dict, list[dict], list[dict]]:
    p = json.loads((OUT / "PROTOCOL.json").read_text())
    assert p["source_code_sha256"] == sha(Path(__file__))
    assert p["model_config_sha256"] == sha(MODEL / "config.json")
    train_path, dev_path = DATA / "en_ewt-ud-train.conllu", DATA / "en_ewt-ud-dev.conllu"
    assert p["train_sha256"] == sha(train_path) and p["dev_sha256"] == sha(dev_path)
    train, dev = parse(train_path), parse(dev_path)
    assert p["dev_ids"] == [x["id"] for x in dev]
    return p, train, dev


@torch.inference_mode()
def embed(rows: list[dict]) -> dict[str, np.ndarray]:
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModel.from_pretrained(MODEL, local_files_only=True)
    model.eval()
    torch.set_num_threads(4)
    out = {}
    for index, row in enumerate(rows, 1):
        encoded = tok(row["tokens"], is_split_into_words=True, return_tensors="pt",
                      truncation=True, max_length=MODEL_MAX)
        word_ids = encoded.word_ids(batch_index=0)
        wanted = (row["first_index"], row["second_index"], row["verb_index"])
        offsets = [word_ids.index(w) if w in word_ids else None for w in wanted]
        if any(v is None for v in offsets):
            continue
        state = model(**encoded).last_hidden_state[0]
        first, second, verb = (state[i].numpy() for i in offsets)
        delta = first - second
        relative = np.array([(row["first_index"] - row["verb_index"]) / MAX_TOKENS,
                             (row["second_index"] - row["verb_index"]) / MAX_TOKENS], dtype=np.float32)
        out[row["id"]] = np.concatenate((delta, delta * verb, relative)).astype(np.float32)
        if index % 250 == 0:
            print(f"embedded {index}/{len(rows)}", flush=True)
    return out


def run() -> None:
    p, train, dev = load()
    target = OUT / "RESULTS.json"
    if target.exists():
        raise FileExistsError(target)
    by_id = {x["id"]: x for x in train + dev}
    needed = {i for seed in SEEDS for arm in ("active_only", "balanced_active")
              for i in p["sample_ids"][str(seed)][arm]}
    needed |= {x["id"] for x in train if x["voice"] == "passive"}
    needed |= {x["id"] for x in dev}
    features = embed([by_id[i] for i in sorted(needed)])
    usable_dev = [x for x in dev if x["id"] in features]
    assert len([x for x in usable_dev if x["voice"] == "passive"]) >= 10
    summaries = []
    for seed in SEEDS:
        for arm in ("active_only", "balanced"):
            ids = p["sample_ids"][str(seed)]["active_only" if arm == "active_only" else "balanced_active"]
            if arm == "balanced":
                ids = ids + [x["id"] for x in train if x["voice"] == "passive"]
            ids = [i for i in ids if i in features]
            assert len(ids) >= 0.95 * 2 * p["n_per_voice"]
            x = np.stack([features[i] for i in ids])
            y = np.array([by_id[i]["actor_is_first"] for i in ids])
            # Active-only data may include a few nonfirst actors, so both classes exist.
            assert len(set(y)) == 2
            scale = StandardScaler().fit(x)
            clf = LogisticRegression(C=0.05, max_iter=1000, random_state=seed).fit(scale.transform(x), y)
            for voice in ("active", "passive"):
                items = [r for r in usable_dev if r["voice"] == voice]
                predicted = clf.predict(scale.transform(np.stack([features[r["id"]] for r in items])))
                rows = [{"id": r["id"], "correct": bool(int(predicted[j]) == r["actor_is_first"]),
                         "gold_actor_first": bool(r["actor_is_first"])} for j, r in enumerate(items)]
                summaries.append({"seed": seed, "arm": arm, "voice": voice,
                                  "train_count": len(ids), "train_actor_first": int(y.sum()),
                                  "correct": sum(v["correct"] for v in rows), "total": len(rows),
                                  "items": rows})
            print(f"completed seed={seed} arm={arm}", flush=True)
    baselines = {voice: {"first_candidate_correct": sum(r["actor_is_first"] for r in usable_dev if r["voice"] == voice),
                         "total": sum(r["voice"] == voice for r in usable_dev)} for voice in ("active", "passive")}
    result = {"protocol_sha256": sha(OUT / "PROTOCOL.json"), "feature_count": len(features),
              "baselines": baselines, "results": summaries}
    target.write_text(json.dumps(result, indent=2) + "\n")
    for voice in ("active", "passive"):
        print(voice, "first-candidate baseline", baselines[voice])
        for arm in ("active_only", "balanced"):
            rows = [x for x in summaries if x["voice"] == voice and x["arm"] == arm]
            print(arm, [(x["correct"], x["total"]) for x in rows])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "run"))
    args = parser.parse_args()
    {"prepare": prepare, "run": run}[args.mode]()
