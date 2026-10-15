"""Fixed readout and middle-layer swap diagnostic for local Qwen2.5-1.5B."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE / "llm_role_internal_v1"
MODEL = Path("/private/tmp/rolepilot_hf/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
PAIRS = (("Grace", "Henry"), ("Emily", "Jacob"), ("Harry", "Lisa"),
         ("Aaron", "Eric"), ("Maria", "Peter"), ("Rachel", "Robert"),
         ("Sam", "Susan"), ("William", "Laura"))
VERBS = ("praised", "criticized", "thanked", "visited")
VOICES = ("active", "passive")
WORDINGS = ("generic", "performed")
LENS_LAYERS = (0, 4, 8, 12, 16, 20, 24, 28)
PATCH_LAYERS = (14, 20)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows() -> list[dict]:
    result = []
    for pair_index, (a, b) in enumerate(PAIRS):
        for verb in VERBS:
            for direction, (actor, patient) in enumerate(((a, b), (b, a))):
                for voice in VOICES:
                    sentence = (f"{actor} {verb} {patient}." if voice == "active"
                                else f"{patient} was {verb} by {actor}.")
                    for wording in WORDINGS:
                        question = ("Who did the action?" if wording == "generic"
                                    else "Who performed the action?")
                        result.append({"pair_index": pair_index, "verb": verb,
                                       "direction": direction, "actor": actor,
                                       "patient": patient, "voice": voice,
                                       "wording": wording, "sentence": sentence,
                                       "question": question})
    assert len(result) == 256
    return result


def protocol() -> dict:
    return {"date": "2026-10-10", "status": "fixed_before_hidden_state_run",
            "model": "Qwen2.5-1.5B-Instruct at commit 989aa7980e4cf806f80c7fef2b1adb7bc71aa306, no weight updates",
            "model_weights_sha256": sha(MODEL / "model.safetensors"),
            "model_config_sha256": sha(MODEL / "config.json"),
            "source_sha256": sha(HERE / "llm_role_internal_v1.py"),
            "design": "8 new name pairs x 4 new reversible verbs x 2 directions x active/passive x 2 fixed actor questions = 256 prompts, 64 events",
            "token_rule": "both names must be one token, with and without leading space; score each name by logsumexp of the two token logits",
            "generation_measure": "greedy first answer token decoded and stripped; correct only if it is exactly the actor name, invalid counted separately",
            "readout": "at final prompt token after 0,4,8,12,16,20,24,28 blocks, apply final model norm and fixed LM head to intermediate states; report actor-minus-patient candidate logit margin and positive fraction, without fitting a probe",
            "main_diagnostic": "among prompts with a wrong generic first-token answer but a correct performed answer for the same event and voice, report generic actor-positive fraction at fixed layers 12,16,20,24 and final28; descriptive only",
            "intervention": "for all generic-wrong/performed-correct cases, replace last-token state after block14 and block20 in generic prompt with state from performed prompt on same event; compare with state from performed prompt on direction-reversed event with same names, verb and voice",
            "intervention_outcome": "correct-donor minus reversed-donor actor-versus-patient next-token margin, plus first-token actor accuracy, separately by layer; selected cases are conditional and findings exploratory",
            "limits": "logit lens is an imperfect readout; whole-state patch transfers more than role information; output differences do not prove absence/presence of a complete role representation; one small model and constructed sentences"}


def prepare() -> None:
    ROOT.mkdir(exist_ok=True)
    path = ROOT / "PROTOCOL.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(protocol(), indent=2) + "\n")
    (ROOT / "ITEMS.json").write_text(json.dumps(rows(), indent=2) + "\n")
    print(json.dumps({"protocol": str(path), "items": 256}, indent=2))


def validate() -> list[dict]:
    assert json.loads((ROOT / "PROTOCOL.json").read_text()) == protocol()
    items = json.loads((ROOT / "ITEMS.json").read_text())
    assert items == rows()
    return items


def load_model():
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, local_files_only=True, dtype=torch.float32).eval()
    assert model.config.num_hidden_layers == 28
    for name in {n for pair in PAIRS for n in pair}:
        assert len(tokenizer.encode(name, add_special_tokens=False)) == 1
        assert len(tokenizer.encode(" " + name, add_special_tokens=False)) == 1
    return tokenizer, model


def make_prompt(tokenizer, row: dict) -> str:
    return tokenizer.apply_chat_template([
        {"role": "system", "content": "Answer with exactly one of the two names in the sentence. Output only the name."},
        {"role": "user", "content": f"Sentence: {row['sentence']}\nQuestion: {row['question']}"}],
        tokenize=False, add_generation_prompt=True)


def candidate_ids(tokenizer, name: str) -> list[int]:
    return [tokenizer.encode(name, add_special_tokens=False)[0],
            tokenizer.encode(" " + name, add_special_tokens=False)[0]]


def margin_from_logits(logits: torch.Tensor, tokenizer, actor: str,
                       patient: str) -> float:
    actor_score = torch.logsumexp(logits[candidate_ids(tokenizer, actor)], dim=0)
    patient_score = torch.logsumexp(logits[candidate_ids(tokenizer, patient)], dim=0)
    return float((actor_score - patient_score).item())


def lens_margin(model, hidden: torch.Tensor, tokenizer,
                actor: str, patient: str, final: bool) -> float:
    state = hidden if final else model.model.norm(hidden)
    ids = candidate_ids(tokenizer, actor) + candidate_ids(tokenizer, patient)
    logits = torch.mv(model.lm_head.weight[ids], state)
    return float((torch.logsumexp(logits[:2], dim=0) -
                  torch.logsumexp(logits[2:], dim=0)).item())


@torch.inference_mode()
def initial_pass(items: list[dict], tokenizer, model) -> None:
    destination = ROOT / "READOUT.json"
    if destination.exists():
        raise FileExistsError(destination)
    records = []
    h14, h20 = [], []
    for start in range(0, len(items), 4):
        batch = items[start:start + 4]
        inputs = tokenizer([make_prompt(tokenizer, row) for row in batch],
                           padding=True, return_tensors="pt")
        output = model(**inputs, output_hidden_states=True, use_cache=False)
        for j, row in enumerate(batch):
            logits = output.logits[j, -1]
            first = tokenizer.decode([int(logits.argmax())]).strip()
            margins = {str(layer): lens_margin(
                model, output.hidden_states[layer][j, -1], tokenizer,
                row["actor"], row["patient"], final=(layer == 28))
                       for layer in LENS_LAYERS}
            assert abs(margins["28"] - margin_from_logits(
                logits, tokenizer, row["actor"], row["patient"])) < 1e-4
            records.append({**row, "first_token": first,
                            "valid": first in (row["actor"], row["patient"]),
                            "correct": first == row["actor"],
                            "candidate_margins": margins})
            h14.append(output.hidden_states[14][j, -1].cpu().numpy())
            h20.append(output.hidden_states[20][j, -1].cpu().numpy())
        if (start + len(batch)) % 32 == 0:
            print(f"readout {start + len(batch)}/{len(items)}", flush=True)
    np.savez_compressed(ROOT / "PATCH_DONORS.npz",
                        h14=np.stack(h14), h20=np.stack(h20))
    destination.write_text(json.dumps(records, indent=2) + "\n")


@torch.inference_mode()
def patched_margin(row: dict, donor: np.ndarray, layer: int,
                   tokenizer, model) -> dict:
    inputs = tokenizer(make_prompt(tokenizer, row), return_tensors="pt")
    block = model.model.layers[layer - 1]
    donor_tensor = torch.from_numpy(donor)

    def replace(_module, _input, output):
        altered = output.clone()
        altered[:, -1, :] = donor_tensor
        return altered

    handle = block.register_forward_hook(replace)
    try:
        output = model(**inputs, use_cache=False)
    finally:
        handle.remove()
    logits = output.logits[0, -1]
    answer = tokenizer.decode([int(logits.argmax())]).strip()
    return {"answer": answer, "correct": answer == row["actor"],
            "margin": margin_from_logits(logits, tokenizer,
                                         row["actor"], row["patient"])}


def patch_pass(items: list[dict], tokenizer, model) -> None:
    assert (ROOT / "READOUT.json").exists()
    path = ROOT / "PATCHES.json"
    if path.exists():
        raise FileExistsError(path)
    readout = json.loads((ROOT / "READOUT.json").read_text())
    assert len(readout) == len(items)
    for item, record in zip(items, readout):
        assert all(record[k] == v for k, v in item.items())
    donors = np.load(ROOT / "PATCH_DONORS.npz")
    lookup = {(row["pair_index"], row["verb"], row["direction"],
               row["voice"], row["wording"]): i
              for i, row in enumerate(readout)}
    selected = []
    for i, row in enumerate(readout):
        if row["wording"] != "generic" or row["correct"]:
            continue
        counterpart = lookup[row["pair_index"], row["verb"],
                             row["direction"], row["voice"], "performed"]
        if readout[counterpart]["correct"]:
            selected.append((i, counterpart))
    records = []
    for n, (recipient, correct_donor) in enumerate(selected, 1):
        row = readout[recipient]
        reverse_donor = lookup[row["pair_index"], row["verb"],
                               1 - row["direction"], row["voice"], "performed"]
        one = {"recipient": recipient, "correct_donor": correct_donor,
               "reverse_donor": reverse_donor,
               "ordinary_margin": row["candidate_margins"]["28"],
               "ordinary_answer": row["first_token"],
               "patches": {}}
        for layer in PATCH_LAYERS:
            states = donors[f"h{layer}"]
            one["patches"][str(layer)] = {
                "correct_donor": patched_margin(row, states[correct_donor],
                                                layer, tokenizer, model),
                "reverse_donor": patched_margin(row, states[reverse_donor],
                                                layer, tokenizer, model)}
        records.append(one)
        if n % 8 == 0:
            print(f"patches {n}/{len(selected)}", flush=True)
    path.write_text(json.dumps({"selected_cases": len(selected),
                                "records": records}, indent=2) + "\n")
    print(f"patches {len(selected)}/{len(selected)}", flush=True)


def summarize() -> dict:
    readout = json.loads((ROOT / "READOUT.json").read_text())
    patches = json.loads((ROOT / "PATCHES.json").read_text())
    cells = {}
    for wording in WORDINGS:
        for voice in VOICES:
            selected = [r for r in readout
                        if r["wording"] == wording and r["voice"] == voice]
            cells[f"{wording}/{voice}"] = {
                "n": len(selected), "correct": sum(r["correct"] for r in selected),
                "valid": sum(r["valid"] for r in selected),
                "actor_positive_at_layer": {
                    str(layer): sum(r["candidate_margins"][str(layer)] > 0
                                    for r in selected) for layer in LENS_LAYERS}}
    selected_readout = [readout[r["recipient"]] for r in patches["records"]]
    latent = {str(layer): sum(row["candidate_margins"][str(layer)] > 0
                              for row in selected_readout)
              for layer in LENS_LAYERS}
    intervention = {}
    for layer in PATCH_LAYERS:
        key = str(layer)
        records = patches["records"]
        intervention[key] = {
            "n": len(records),
            "correct_donor_actor_answers": sum(r["patches"][key]["correct_donor"]["correct"]
                                               for r in records),
            "reversed_donor_actor_answers": sum(r["patches"][key]["reverse_donor"]["correct"]
                                                for r in records),
            "mean_correct_minus_reversed_margin": float(np.mean([
                r["patches"][key]["correct_donor"]["margin"] -
                r["patches"][key]["reverse_donor"]["margin"] for r in records]))
            if records else None}
    return {"cells": cells, "discordant_cases": len(selected_readout),
            "generic_actor_positive_in_discordant_cases_by_layer": latent,
            "intervention": intervention}


@torch.inference_mode()
def smoke() -> None:
    tokenizer, model = load_model()
    row = {"sentence": "Alice praised Bob.", "question": "Who did the action?",
           "actor": "Alice", "patient": "Bob"}
    inputs = tokenizer(make_prompt(tokenizer, row), return_tensors="pt")
    with torch.inference_mode():
        output = model(**inputs, output_hidden_states=True, use_cache=False)
    margin = lens_margin(model, output.hidden_states[28][0, -1], tokenizer,
                         "Alice", "Bob", final=True)
    assert abs(margin - margin_from_logits(output.logits[0, -1], tokenizer,
                                          "Alice", "Bob")) < 1e-4
    changed = patched_margin(row, np.zeros(1536, dtype=np.float32),
                             14, tokenizer, model)
    print(json.dumps({"layers": len(output.hidden_states) - 1,
                      "ordinary_margin": margin,
                      "zero_patch": changed}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("smoke", "prepare", "readout", "patch", "summarize"))
    args = parser.parse_args()
    if args.action == "smoke": smoke()
    elif args.action == "prepare": prepare()
    elif args.action == "readout":
        items = validate()
        tokenizer, model = load_model()
        initial_pass(items, tokenizer, model)
    elif args.action == "patch":
        items = validate()
        tokenizer, model = load_model()
        patch_pass(items, tokenizer, model)
    else:
        report = summarize()
        (ROOT / "SUMMARY.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
