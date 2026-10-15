"""Inference-only format and checkpoint preflight for the local role adapters.

This file does not import or read any historical test rows. It refuses to
overwrite its own output and does not train or download a model.
"""
from __future__ import annotations

import gc
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import llm_role_repair_confirm as confirm  # noqa: E402
import llm_role_repair_v3 as core  # noqa: E402

CONFIRM = ROOT / "llm_role_repair_confirmation"
UNORDERED = ROOT / "llm_role_unordered_control"
WORLD = "fresh1"
SEED = 830211
ARMS = ("pretrained", "ordinary", "ordered_roles", "unordered_pair")
OUTPUT = HERE / "SMOKE.json"

# New development events. Each event is rendered in both voices.
EVENTS = (
    {"id": "dev_photo", "actor": "Ryan", "patient": "Emma", "verb": "photographed",
     "active": "Ryan photographed Emma.", "passive": "Emma was photographed by Ryan."},
    {"id": "dev_rescue", "actor": "Grace", "patient": "Tyler", "verb": "rescued",
     "active": "Grace rescued Tyler.", "passive": "Tyler was rescued by Grace."},
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_provenance() -> dict:
    cp = json.loads((CONFIRM / "PROTOCOL.json").read_text())
    up = json.loads((UNORDERED / "PROTOCOL.json").read_text())
    expected = {
        "model_weights": (core.MODEL / "model.safetensors", cp["model_weights_sha256"]),
        "confirmation_source": (ROOT / "llm_role_repair_confirm.py", cp["source_sha256"]),
        "core_source": (ROOT / "llm_role_repair_v3.py", cp["core_source_sha256"]),
        "graph_manifest": (confirm.GRAPH, cp["graph_manifest_sha256"]),
        "unordered_source": (ROOT / "llm_role_unordered_control.py", up["source_sha256"]),
    }
    actual = {}
    for label, (path, wanted) in expected.items():
        got = sha(path)
        if got != wanted:
            raise ValueError(f"{label} SHA mismatch: {got} != {wanted}")
        actual[label] = got
    cp_hash = sha(CONFIRM / "PROTOCOL.json")
    up_hash = sha(UNORDERED / "PROTOCOL.json")
    if up["reference_protocol_sha256"] != cp_hash:
        raise ValueError("Unordered protocol references a different confirmation protocol")
    if up["core_source_sha256"] != actual["core_source"]:
        raise ValueError("Unordered protocol references a different core source")
    if up["confirmation_source_sha256"] != actual["confirmation_source"]:
        raise ValueError("Unordered protocol references a different confirmation source")
    if not (core.MODEL / "tokenizer.json").exists():
        raise FileNotFoundError("Local tokenizer missing")
    return {"sha256": actual, "confirmation_protocol_sha256": cp_hash,
            "unordered_protocol_sha256": up_hash, "base_snapshot": str(core.MODEL)}


def checkpoint_path(arm: str) -> Path:
    if arm == "unordered_pair":
        return UNORDERED / f"{WORLD}_{SEED}_{arm}" / "adapter.pt"
    return CONFIRM / f"{WORLD}_{SEED}_{arm}" / "adapter.pt"


def load_arm(arm: str, provenance: dict):
    confirm.configure(WORLD, 0 if arm == "pretrained" else SEED)
    load_kind = "pretrained" if arm == "pretrained" else (
        "ordinary" if arm == "ordinary" else "ordered_roles")
    started = time.perf_counter()
    tokenizer, model, heads = core.load(load_kind)
    model.eval()
    info = {"arm": arm, "load_seconds": None, "adapter_tensors": 0,
            "head_tensors": 0, "checkpoint_sha256": None}
    if arm != "pretrained":
        path = checkpoint_path(arm)
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if set(saved) != {"adapter", "heads"}:
            raise ValueError(f"Unexpected checkpoint top-level keys for {arm}")
        current = {name: parameter for name, parameter in model.named_parameters()
                   if parameter.requires_grad}
        saved_adapter = saved["adapter"]
        if set(saved_adapter) != set(current):
            raise ValueError(f"Adapter key mismatch for {arm}: "
                             f"missing={set(current)-set(saved_adapter)}, "
                             f"extra={set(saved_adapter)-set(current)}")
        with torch.no_grad():
            for name, parameter in current.items():
                value = saved_adapter[name]
                if value.shape != parameter.shape or value.dtype != parameter.dtype:
                    raise ValueError(f"Adapter tensor mismatch for {arm}: {name}")
                parameter.copy_(value)
        saved_heads = saved["heads"]
        if heads is None:
            if saved_heads is not None:
                raise ValueError(f"Unexpected heads for {arm}")
        else:
            if saved_heads is None or set(saved_heads) != set(heads.state_dict()):
                raise ValueError(f"Head key mismatch for {arm}")
            heads.load_state_dict(saved_heads, strict=True)
            info["head_tensors"] = len(saved_heads)
        info["adapter_tensors"] = len(saved_adapter)
        info["checkpoint_sha256"] = sha(path)
        result_path = path.with_name("result.json")
        result = json.loads(result_path.read_text())
        expected_protocol = (provenance["unordered_protocol_sha256"] if arm == "unordered_pair"
                             else provenance["confirmation_protocol_sha256"])
        if result["protocol_sha256"] != expected_protocol:
            raise ValueError(f"Result/protocol mismatch for {arm}")
        if (result["world"], result["seed"], result["arm"]) != (WORLD, SEED, arm):
            raise ValueError(f"Result/checkpoint identity mismatch for {arm}")
    info["load_seconds"] = time.perf_counter() - started
    return tokenizer, model, info


def rows() -> list[dict]:
    result = []
    for event in EVENTS:
        for form in ("active", "passive"):
            result.append({"id": f"{event['id']}_{form}", "event_id": event["id"],
                           "form": form, "actor": event["actor"],
                           "patient": event["patient"], "sentence": event[form]})
    return result


def qa_prompt(tokenizer, row: dict) -> str:
    return core.prompt(tokenizer, {"sentence": row["sentence"],
                                   "question": core.QUESTION_TRAIN})


def extraction_prompt(tokenizer, row: dict) -> str:
    return tokenizer.apply_chat_template([
        {"role": "system", "content":
         "Extract the actor and patient from the sentence. Reply exactly: Actor: <name>; Patient: <name>."},
        {"role": "user", "content": f"Sentence: {row['sentence']}"},
    ], tokenize=False, add_generation_prompt=True)


@torch.inference_mode()
def smoke_arm(tokenizer, model, items: list[dict]) -> tuple[list[dict], dict]:
    qa_prompts = [qa_prompt(tokenizer, row) for row in items]
    started = time.perf_counter()
    inputs = tokenizer(qa_prompts, padding=True, return_tensors="pt")
    logits = model(**inputs, use_cache=False).logits[:, -1, :]
    qa_tokens = logits.argmax(-1).tolist()
    qa_seconds = time.perf_counter() - started

    extraction_prompts = [extraction_prompt(tokenizer, row) for row in items]
    started = time.perf_counter()
    inputs = tokenizer(extraction_prompts, padding=True, return_tensors="pt")
    outputs = model.generate(**inputs, do_sample=False, max_new_tokens=20,
                             use_cache=True, pad_token_id=tokenizer.pad_token_id)
    generated = outputs[:, inputs["input_ids"].shape[1]:]
    extraction_seconds = time.perf_counter() - started

    records = []
    for i, row in enumerate(items):
        first = tokenizer.decode([qa_tokens[i]], skip_special_tokens=False)
        direct = first.strip()
        raw = tokenizer.decode(generated[i], skip_special_tokens=False)
        clean = tokenizer.decode(generated[i], skip_special_tokens=True).strip()
        match = re.fullmatch(r"Actor:\s*([A-Za-z]+);\s*Patient:\s*([A-Za-z]+)", clean)
        parsed = None if match is None else {"actor": match.group(1),
                                             "patient": match.group(2)}
        records.append({**row,
                        "direct_qa": {"prompt": qa_prompts[i], "first_token_id": qa_tokens[i],
                                      "raw_first_token": first, "parsed_actor": direct if direct in
                                      (row["actor"], row["patient"]) else None},
                        "role_extraction": {"prompt": extraction_prompts[i],
                                            "generated_token_ids": generated[i].tolist(),
                                            "raw_output": raw, "clean_output": clean,
                                            "parsed": parsed}})
    return records, {"direct_qa_seconds": qa_seconds,
                     "role_extraction_seconds": extraction_seconds,
                     "role_extraction_max_new_tokens": 20}


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    provenance = verify_provenance()
    items = rows()
    if len(items) != 4:
        raise AssertionError("Smoke must contain exactly two events x two voices")
    report = {"status": "technical_smoke_only_not_scientific_evidence",
              "world": WORLD, "seed": SEED, "provenance": provenance,
              "events": list(EVENTS), "arms": []}
    for arm in ARMS:
        tokenizer, model, load_info = load_arm(arm, provenance)
        records, timing = smoke_arm(tokenizer, model, items)
        report["arms"].append({"load": load_info, "timing": timing,
                               "records": records})
        print(json.dumps({"arm": arm, "load_seconds": round(load_info["load_seconds"], 2),
                          **{key: round(value, 2) if isinstance(value, float) else value
                             for key, value in timing.items()}}), flush=True)
        del tokenizer, model
        gc.collect()
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
