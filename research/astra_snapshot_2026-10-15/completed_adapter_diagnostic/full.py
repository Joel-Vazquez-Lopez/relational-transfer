"""Frozen development-only behavioral comparison for matched 0.5B adapters.

Commands: prepare, run, analyze. No command trains or downloads a model.
All scientific inputs and this source are hashed in FROZEN_PROTOCOL.json.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import llm_role_repair_confirm as confirm  # noqa: E402
import llm_role_repair_v3 as core  # noqa: E402

CONFIRM = ROOT / "llm_role_repair_confirmation"
UNORDERED = ROOT / "llm_role_unordered_control"
PROTOCOL = HERE / "FROZEN_PROTOCOL.json"
ITEMS = HERE / "FROZEN_ITEMS.json"
OUT = HERE / "full_outputs"
ANALYSIS = HERE / "ANALYSIS.json"
WORLDS = tuple(confirm.WORLD_NAMES)
SEEDS = confirm.SEEDS
ARMS = ("pretrained", "ordinary", "ordered_roles", "unordered_pair")
VERBS = ("praised", "criticized")
FORMS = ("active", "passive", "patient_cleft")
MAX_NEW_TOKENS = 20
BATCH = 4
ACTOR_QUESTION = "Who performed the action described in the sentence?"
PATIENT_QUESTION = "Who received the action described in the sentence?"
DIRECT_SYSTEM = "Answer with exactly one person's name. Do not add any other words."
EXTRACTION_SYSTEM = "Extract the actor and patient from the sentence. Reply exactly: Actor: <name>; Patient: <name>."
DIRECT_SEMANTIC = re.compile(r"^\s*([A-Za-z]+)\.?\s*$")
EXTRACT_SEMANTIC = re.compile(r"^\s*Actor:\s*([A-Za-z]+);\s*Patient:\s*([A-Za-z]+)\.?\s*$")
DIRECT_EXACT = re.compile(r"[A-Za-z]+")
EXTRACT_EXACT = re.compile(r"Actor: [A-Za-z]+; Patient: [A-Za-z]+")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(path)
    temp = path.with_name(path.name + ".tmp")
    if temp.exists():
        raise FileExistsError(temp)
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def prompt(tokenizer, system: str, user: str) -> str:
    return tokenizer.apply_chat_template([
        {"role": "system", "content": system}, {"role": "user", "content": user}
    ], tokenize=False, add_generation_prompt=True)


def sentence(actor: str, patient: str, verb: str, form: str) -> str:
    if form == "active":
        return f"{actor} {verb} {patient}."
    if form == "passive":
        return f"{patient} was {verb} by {actor}."
    if form == "patient_cleft":
        return f"It was {patient} whom {actor} {verb}."
    raise ValueError(form)


def build_items(tokenizer) -> list[dict]:
    graph = json.loads(confirm.GRAPH.read_text())
    out = []
    for world in WORLDS:
        names = confirm.WORLD_NAMES[world]
        train_edges = {tuple(pair) for pair in graph["worlds"][world]["labeled_edges"]}
        heldout = [pair for pair in itertools.combinations(range(10), 2)
                   if pair not in train_edges]
        assert len(train_edges) == 20 and len(heldout) == 25
        chosen = heldout[:8]
        assert len(chosen) == 8 and not (set(chosen) & train_edges)
        for left, right in chosen:
            for actor_id, patient_id in ((left, right), (right, left)):
                actor, patient = names[actor_id], names[patient_id]
                for verb in VERBS:
                    for form in FORMS:
                        sent = sentence(actor, patient, verb, form)
                        row = {"id": f"{world}_{left}_{right}_{actor_id}_{verb}_{form}",
                               "world": world, "pair": [left, right], "actor_id": actor_id,
                               "patient_id": patient_id, "actor": actor,
                               "patient": patient, "verb": verb, "form": form,
                               "sentence": sent}
                        row["archived_prompt"] = core.prompt(tokenizer, {
                            "sentence": sent, "question": core.QUESTION_TRAIN})
                        row["direct_actor_prompt"] = prompt(
                            tokenizer, DIRECT_SYSTEM,
                            f"Sentence: {sent}\nQuestion: {ACTOR_QUESTION}")
                        row["direct_patient_prompt"] = prompt(
                            tokenizer, DIRECT_SYSTEM,
                            f"Sentence: {sent}\nQuestion: {PATIENT_QUESTION}")
                        row["extraction_prompt"] = prompt(
                            tokenizer, EXTRACTION_SYSTEM, f"Sentence: {sent}")
                        out.append(row)
    return out


def validate_items(items: list[dict]) -> None:
    graph = json.loads(confirm.GRAPH.read_text())
    if len(items) != 288 or len({r["id"] for r in items}) != len(items):
        raise ValueError("Item count or duplicate ID")
    for world in WORLDS:
        subset = [r for r in items if r["world"] == world]
        pairs = {tuple(r["pair"]) for r in subset}
        train = {tuple(e) for e in graph["worlds"][world]["labeled_edges"]}
        if len(subset) != 96 or len(pairs) != 8 or pairs & train:
            raise ValueError(f"Coverage or training-edge leakage: {world}")
        for pair in pairs:
            rows = [r for r in subset if tuple(r["pair"]) == pair]
            cells = {(r["actor_id"], r["verb"], r["form"]) for r in rows}
            expected = set(itertools.product(pair, VERBS, FORMS))
            if cells != expected:
                raise ValueError(f"Incomplete pair: {world}/{pair}")
        for row in subset:
            if row["actor"] != confirm.WORLD_NAMES[world][row["actor_id"]] or row["patient"] != confirm.WORLD_NAMES[world][row["patient_id"]]:
                raise ValueError("Name ID mismatch")
            if row["sentence"] != sentence(row["actor"], row["patient"], row["verb"], row["form"]):
                raise ValueError("Sentence rendering mismatch")
            if row["verb"] in core.TRAIN_VERBS + core.TEST_VERBS or row["verb"] in ("photographed", "rescued"):
                raise ValueError("Verb overlaps historical or smoke panel")
            if row["sentence"] in ("Ryan photographed Emma.", "Emma was photographed by Ryan.",
                                   "Grace rescued Tyler.", "Tyler was rescued by Grace."):
                raise ValueError("Smoke sentence overlap")


def checkpoint(arm: str, world: str, seed: int) -> Path:
    base = UNORDERED if arm == "unordered_pair" else CONFIRM
    return base / f"{world}_{seed}_{arm}" / "adapter.pt"


def prepare() -> None:
    if PROTOCOL.exists() or ITEMS.exists() or OUT.exists():
        raise FileExistsError("A frozen protocol, item file, or output directory already exists")
    tokenizer = AutoTokenizer.from_pretrained(core.MODEL, local_files_only=True)
    items = build_items(tokenizer)
    validate_items(items)
    cp = json.loads((CONFIRM / "PROTOCOL.json").read_text())
    up = json.loads((UNORDERED / "PROTOCOL.json").read_text())
    reference_hashes = {
        "model_weights": sha(core.MODEL / "model.safetensors"),
        "tokenizer_json": sha(core.MODEL / "tokenizer.json"),
        "confirmation_protocol": sha(CONFIRM / "PROTOCOL.json"),
        "unordered_protocol": sha(UNORDERED / "PROTOCOL.json"),
        "core_source": sha(ROOT / "llm_role_repair_v3.py"),
        "confirmation_source": sha(ROOT / "llm_role_repair_confirm.py"),
        "unordered_source": sha(ROOT / "llm_role_unordered_control.py"),
        "graph_manifest": sha(confirm.GRAPH),
        "full_source": sha(Path(__file__)),
    }
    if (reference_hashes["model_weights"] != cp["model_weights_sha256"] or
        reference_hashes["core_source"] != cp["core_source_sha256"] or
        reference_hashes["confirmation_source"] != cp["source_sha256"] or
        reference_hashes["graph_manifest"] != cp["graph_manifest_sha256"] or
        reference_hashes["unordered_source"] != up["source_sha256"] or
        reference_hashes["confirmation_protocol"] != up["reference_protocol_sha256"]):
        raise ValueError("Frozen historical source/base lineage mismatch")
    checkpoints = {}
    for world in WORLDS:
        for seed in SEEDS:
            for arm in ARMS[1:]:
                path = checkpoint(arm, world, seed)
                if not path.exists():
                    raise FileNotFoundError(path)
                checkpoints[f"{world}_{seed}_{arm}"] = sha(path)
    protocol = {
        "status": "frozen_before_full_development_inference", "date": "2026-10-15",
        "purpose": "Behavioral interface comparison after adaptation; no internal mechanism claim",
        "worlds": WORLDS, "seeds": SEEDS, "arms": ARMS,
        "selection": "First eight lexicographically sorted unordered pairs absent from each world's 20 training edges",
        "verbs": VERBS, "forms": FORMS, "max_new_tokens_each_generated_interface": MAX_NEW_TOKENS,
        "batch_size": BATCH, "actor_question": ACTOR_QUESTION,
        "patient_question": PATIENT_QUESTION, "direct_system": DIRECT_SYSTEM,
        "extraction_system": EXTRACTION_SYSTEM,
        "parser": "Direct: exactly one ASCII alphabetic name, surrounding whitespace and at most one terminal period. Extraction: exactly Actor: NAME; Patient: NAME, surrounding whitespace and at most one terminal period. Reject extra text/names. Record exact requested-format separately. Special tokens removed only for semantic parsing; raw output retained.",
        "primary": "Same-event semantic actor, patient, and joint accuracy under two strong direct questions versus one explicit joint extraction response; pair-level discordance and construction-specific differences by matched arm/world/seed.",
        "secondary": "Archived training-compatible direct actor question, greedy first token, as historical compatibility measure only.",
        "analysis": "All units reported. Paired bootstrap resamples eight unordered entity pairs within each world, sharing resample across arms and interfaces; average the two seeds per graph, and count base only once. CIs exploratory because only three worlds. Invalid formats count wrong. No prompt-level independence claim.",
        "inference_budget": "Two direct calls of up to 20 new tokens versus one joint extraction call of up to 20 new tokens; compute and output budgets differ. Archived question uses one forward pass and one first-token answer.",
        "reference_sha256": reference_hashes, "checkpoint_sha256": checkpoints,
        "item_sha256": None,
    }
    write_new(ITEMS, items)
    protocol["item_sha256"] = sha(ITEMS)
    write_new(PROTOCOL, protocol)
    print(json.dumps({"locked": str(PROTOCOL), "items": len(items),
                      "models": 21, "prompts": 8064,
                      "protocol_sha256": sha(PROTOCOL)}), flush=True)


def validate_frozen() -> tuple[dict, list[dict]]:
    protocol = json.loads(PROTOCOL.read_text())
    items = json.loads(ITEMS.read_text())
    validate_items(items)
    if sha(ITEMS) != protocol["item_sha256"]:
        raise ValueError("Frozen item hash mismatch")
    paths = {
        "model_weights": core.MODEL / "model.safetensors",
        "tokenizer_json": core.MODEL / "tokenizer.json",
        "confirmation_protocol": CONFIRM / "PROTOCOL.json",
        "unordered_protocol": UNORDERED / "PROTOCOL.json",
        "core_source": ROOT / "llm_role_repair_v3.py",
        "confirmation_source": ROOT / "llm_role_repair_confirm.py",
        "unordered_source": ROOT / "llm_role_unordered_control.py",
        "graph_manifest": confirm.GRAPH,
        "full_source": Path(__file__),
    }
    for label, expected in protocol["reference_sha256"].items():
        if sha(paths[label]) != expected:
            raise ValueError(f"Frozen {label} hash mismatch")
    for identity, expected in protocol["checkpoint_sha256"].items():
        world, seed, arm = identity.split("_", 2)
        if sha(checkpoint(arm, world, int(seed))) != expected:
            raise ValueError(f"Frozen checkpoint hash mismatch: {identity}")
    tokenizer = AutoTokenizer.from_pretrained(core.MODEL, local_files_only=True)
    if items != build_items(tokenizer):
        raise ValueError("Frozen prompts differ from source generation")
    return protocol, items


def load_arm(arm: str, world: str, seed: int):
    confirm.configure(world, seed)
    kind = "pretrained" if arm == "pretrained" else (
        "ordinary" if arm == "ordinary" else "ordered_roles")
    started = time.perf_counter()
    tokenizer, model, heads = core.load(kind)
    if arm != "pretrained":
        state = torch.load(checkpoint(arm, world, seed), map_location="cpu", weights_only=True)
        if set(state) != {"adapter", "heads"}:
            raise ValueError("Checkpoint top-level keys")
        current = {name: param for name, param in model.named_parameters() if param.requires_grad}
        if set(current) != set(state["adapter"]) or len(current) != 96:
            raise ValueError("Adapter key mismatch")
        with torch.no_grad():
            for name, param in current.items():
                source = state["adapter"][name]
                if source.shape != param.shape or source.dtype != param.dtype:
                    raise ValueError(f"Adapter tensor mismatch: {name}")
                param.copy_(source)
        if heads is None:
            if state["heads"] is not None:
                raise ValueError("Unexpected heads")
        else:
            if state["heads"] is None or set(heads.state_dict()) != set(state["heads"]):
                raise ValueError("Head key mismatch")
            heads.load_state_dict(state["heads"], strict=True)
    model.eval()
    return tokenizer, model, time.perf_counter() - started


def parse_direct(clean: str) -> dict:
    match = DIRECT_SEMANTIC.fullmatch(clean)
    return {"name": match.group(1) if match else None,
            "semantic_format_valid": bool(match),
            "exact_requested_format": bool(DIRECT_EXACT.fullmatch(clean))}


def parse_extraction(clean: str) -> dict:
    match = EXTRACT_SEMANTIC.fullmatch(clean)
    return {"actor": match.group(1) if match else None,
            "patient": match.group(2) if match else None,
            "semantic_format_valid": bool(match),
            "exact_requested_format": bool(EXTRACT_EXACT.fullmatch(clean))}


@torch.inference_mode()
def infer_batch(tokenizer, model, batch: list[dict]) -> tuple[list[dict], dict]:
    records = [{"id": row["id"]} for row in batch]
    timings = {}
    for field in ("archived", "direct_actor", "direct_patient", "extraction"):
        prompts = [row[f"{field}_prompt"] for row in batch]
        started = time.perf_counter()
        inputs = tokenizer(prompts, padding=True, return_tensors="pt")
        if field == "archived":
            token_ids = model(**inputs, use_cache=False).logits[:, -1].argmax(-1).tolist()
            for i, token_id in enumerate(token_ids):
                raw = tokenizer.decode([token_id], skip_special_tokens=False)
                records[i][field] = {"raw_output": raw, "raw_token_ids": [token_id],
                                     "generated_tokens": 1, "parsed": parse_direct(raw.strip())}
        else:
            outputs = model.generate(**inputs, do_sample=False, max_new_tokens=MAX_NEW_TOKENS,
                                     use_cache=True, pad_token_id=tokenizer.pad_token_id)
            suffix = outputs[:, inputs["input_ids"].shape[1]:]
            for i, ids in enumerate(suffix):
                token_ids = ids.tolist()
                raw = tokenizer.decode(ids, skip_special_tokens=False)
                clean = tokenizer.decode(ids, skip_special_tokens=True)
                parser = parse_extraction if field == "extraction" else parse_direct
                eos = tokenizer.eos_token_id
                generated_count = (token_ids.index(eos) + 1 if eos in token_ids
                                   else len(token_ids))
                records[i][field] = {"raw_output": raw, "clean_output": clean,
                                     "raw_token_ids": token_ids,
                                     "generated_tokens": generated_count,
                                     "batch_suffix_tokens": len(token_ids),
                                     "parsed": parser(clean)}
        timings[field] = time.perf_counter() - started
    return records, timings


def unit_jobs():
    for world in WORLDS:
        yield world, 0, "pretrained"
        for seed in SEEDS:
            for arm in ARMS[1:]:
                yield world, seed, arm


def run() -> None:
    protocol, items = validate_frozen()
    OUT.mkdir(exist_ok=True)
    protocol_hash = sha(PROTOCOL)
    for world, seed, arm in unit_jobs():
        path = OUT / f"{world}_{seed}_{arm}.json"
        if path.exists():
            data = json.loads(path.read_text())
            if (data["protocol_sha256"] != protocol_hash or data["world"] != world or
                data["seed"] != seed or data["arm"] != arm or len(data["records"]) != 96):
                raise ValueError(f"Existing output identity mismatch: {path}")
            print(f"Already complete {path.name}", flush=True)
            continue
        tokenizer, model, load_seconds = load_arm(arm, world, seed)
        subset = [r for r in items if r["world"] == world]
        all_records = []
        timings = defaultdict(float)
        started = time.perf_counter()
        for start in range(0, len(subset), BATCH):
            records, elapsed = infer_batch(tokenizer, model, subset[start:start+BATCH])
            all_records.extend(records)
            for field, seconds in elapsed.items():
                timings[field] += seconds
            if (start + BATCH) % 24 == 0:
                print(f"{world}/{seed}/{arm}: {start+BATCH}/96, {time.perf_counter()-started:.1f}s", flush=True)
        data = {"status": "development_inference", "protocol_sha256": protocol_hash,
                "world": world, "seed": seed, "arm": arm,
                "checkpoint_sha256": (None if arm == "pretrained" else
                                      protocol["checkpoint_sha256"][f"{world}_{seed}_{arm}"]),
                "load_seconds": load_seconds, "inference_seconds": time.perf_counter()-started,
                "interface_seconds": dict(timings), "records": all_records}
        write_new(path, data)
        print(f"Saved {path.name}, {data['inference_seconds']:.1f}s", flush=True)
        del tokenizer, model
        gc.collect()


def item_score(item: dict, record: dict) -> dict:
    a = record["direct_actor"]["parsed"]["name"] == item["actor"]
    p = record["direct_patient"]["parsed"]["name"] == item["patient"]
    x = record["extraction"]["parsed"]
    xa, xp = x["actor"] == item["actor"], x["patient"] == item["patient"]
    return {"qa_actor": a, "qa_patient": p, "qa_both": a and p,
            "ex_actor": xa, "ex_patient": xp, "ex_both": xa and xp,
            "archived_actor": record["archived"]["parsed"]["name"] == item["actor"],
            "qa_actor_malformed": not record["direct_actor"]["parsed"]["semantic_format_valid"],
            "qa_patient_malformed": not record["direct_patient"]["parsed"]["semantic_format_valid"],
            "ex_malformed": not x["semantic_format_valid"],
            "qa_actor_nonexact": not record["direct_actor"]["parsed"]["exact_requested_format"],
            "qa_patient_nonexact": not record["direct_patient"]["parsed"]["exact_requested_format"],
            "ex_nonexact": not x["exact_requested_format"],
            "archived_malformed": not record["archived"]["parsed"]["semantic_format_valid"]}


def rates(rows: list[dict]) -> dict:
    keys = list(rows[0]["score"])
    result = {key: sum(r["score"][key] for r in rows)/len(rows) for key in keys}
    for role in ("actor", "patient", "both"):
        direct = f"qa_{role}"; extract = f"ex_{role}"
        result[f"{role}_newly_correct_extraction"] = sum(
            r["score"][extract] and not r["score"][direct] for r in rows)
        result[f"{role}_newly_incorrect_extraction"] = sum(
            r["score"][direct] and not r["score"][extract] for r in rows)
        result[f"{role}_ex_minus_qa"] = result[extract] - result[direct]
    result["n"] = len(rows)
    return result


def paired_bootstrap(world_rows: dict, arm: str, form: str, role: str,
                     repeats: int = 2000) -> dict:
    # Resample unordered pairs within each world. Preserve all repeated forms,
    # directions, verbs and seeds in a sampled pair; count the base only once.
    rng = np.random.default_rng(15102026)
    observed = []
    distributions = []
    for world in WORLDS:
        rows = [r for r in world_rows[world] if r["form"] == form and r["arm"] == arm]
        pairs = sorted({tuple(r["pair"]) for r in rows})
        if len(pairs) != 8:
            raise ValueError("Bootstrap requires eight pairs")
        pair_deltas = []
        for pair in pairs:
            chosen = [r for r in rows if tuple(r["pair"]) == pair]
            by_seed = defaultdict(list)
            for r in chosen:
                by_seed[r["seed"]].append(r)
            seed_deltas = [rates(group)[f"{role}_ex_minus_qa"] for group in by_seed.values()]
            pair_deltas.append(float(np.mean(seed_deltas)))
        observed.append(float(np.mean(pair_deltas)))
        distributions.append(np.array(pair_deltas))
    boot = []
    for _ in range(repeats):
        values = [float(np.mean(pairs[rng.integers(0, 8, 8)])) for pairs in distributions]
        boot.append(float(np.mean(values)))
    return {"estimate": float(np.mean(observed)), "world_estimates": dict(zip(WORLDS, observed)),
            "exploratory_pair_cluster_ci_95": [float(np.quantile(boot, .025)),
                                                   float(np.quantile(boot, .975))],
            "bootstrap_repeats": repeats}


def analyze() -> None:
    _, items = validate_frozen()
    if ANALYSIS.exists():
        raise FileExistsError(ANALYSIS)
    lookup = {item["id"]: item for item in items}
    joined = []
    timings = []
    for world, seed, arm in unit_jobs():
        path = OUT / f"{world}_{seed}_{arm}.json"
        data = json.loads(path.read_text())
        if data["protocol_sha256"] != sha(PROTOCOL) or len(data["records"]) != 96:
            raise ValueError(f"Invalid output: {path}")
        ids = [r["id"] for r in data["records"]]
        expected_ids = [r["id"] for r in items if r["world"] == world]
        if ids != expected_ids:
            raise ValueError(f"Incomplete or reordered output: {path}")
        timings.append({"world": world, "seed": seed, "arm": arm,
                        "load_seconds": data["load_seconds"],
                        "inference_seconds": data["inference_seconds"],
                        "interface_seconds": data["interface_seconds"]})
        for record in data["records"]:
            item = lookup[record["id"]]
            joined.append({"id": item["id"], "world": world, "seed": seed,
                           "arm": arm, "pair": item["pair"], "form": item["form"],
                           "score": item_score(item, record)})
    if len(joined) != 2016:
        raise ValueError("Full output coverage")
    unit = {}
    by_form = {}
    world_rows = {world: [r for r in joined if r["world"] == world] for world in WORLDS}
    for world, seed, arm in unit_jobs():
        rows = [r for r in world_rows[world] if r["seed"] == seed and r["arm"] == arm]
        key = f"{world}_{seed}_{arm}"
        unit[key] = rates(rows)
        by_form[key] = {form: rates([r for r in rows if r["form"] == form]) for form in FORMS}
    bootstrap = {}
    for arm in ARMS:
        for form in FORMS:
            for role in ("actor", "patient", "both"):
                bootstrap[f"{arm}_{form}_{role}"] = paired_bootstrap(
                    world_rows, arm, form, role)
    report = {"status": "development_behavioral_diagnostic",
              "protocol_sha256": sha(PROTOCOL), "source_sha256": sha(Path(__file__)),
              "unit": unit, "by_form": by_form, "exploratory_bootstrap": bootstrap,
              "timing": timings, "total_sentence_model_comparisons": len(joined),
              "limits": "Synthetic development items; one base per graph, two related seeds per graph, three worlds. Two strong direct generations versus one joint extraction generation are not compute-matched. Behavioral differences do not identify internal mechanisms. Pair-cluster CIs are exploratory with three worlds."}
    write_new(ANALYSIS, report)
    print(f"Saved {ANALYSIS}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run", "analyze"))
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "analyze": analyze}[args.action]()
