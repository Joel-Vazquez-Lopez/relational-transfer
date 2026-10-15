"""Exploratory direct-answer check on the separately frozen Bank test events."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import torch

import llm_role_natural_repair as repair
import llm_role_natural_repair_fresh as fresh
from llm_role_natural_contrast_screen import prompt


ROOT = Path(__file__).resolve().parent / "llm_role_natural_repair/fresh_bank_test"


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold().strip().strip(" .,'\"!?"))


@torch.inference_mode()
def run(arm: str) -> None:
    events = fresh.validate()
    output = ROOT / f"direct_{arm}.json"
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(2)
    tok, model = repair.load_source()
    if arm == "new":
        weights = torch.load(repair.ROOT / "mixed_replay/adapter.pt", map_location="cpu", weights_only=True)
        params = {name: value for name, value in model.named_parameters() if value.requires_grad}
        assert weights.keys() == params.keys()
        for name, param in params.items():
            param.data.copy_(weights[name])
    model.eval()
    records = []
    for i, event in enumerate(events, 1):
        answers = {}
        for role in ("actor", "other"):
            prefix = tok.encode(prompt(tok, event, role), add_special_tokens=False)
            generated = model.generate(input_ids=torch.tensor([prefix]),
                                       max_new_tokens=16, do_sample=False,
                                       pad_token_id=tok.eos_token_id,
                                       eos_token_id=tok.eos_token_id,
                                       use_cache=True)
            answer = tok.decode(generated[0, len(prefix):], skip_special_tokens=True).strip()
            answers[role] = {"raw": answer, "gold": event[f"{role}_span"],
                             "exact": normalize(answer) == normalize(event[f"{role}_span"])}
        records.append({"id": event["id"], "answers": answers})
        if i % 25 == 0:
            print(f"{arm}: {i}/{len(events)}", flush=True)
    summary = {role: sum(x["answers"][role]["exact"] for x in records)
               for role in ("actor", "other")}
    summary["both"] = sum(all(x["answers"][r]["exact"] for r in ("actor", "other"))
                          for x in records)
    obj = {"arm": arm,
           "fresh_protocol_sha256": hashlib.sha256((ROOT / "PROTOCOL.json").read_bytes()).hexdigest(),
           "events_sha256": hashlib.sha256((ROOT / "EVENTS.json").read_bytes()).hexdigest(),
           "source_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "summary": summary, "records": records,
           "limits": "Direct generation was added after seeing the candidate-relative fresh-test result. Exact span matching is strict and may mark a semantically equivalent shorter phrase wrong. One checkpoint per condition."}
    output.write_text(json.dumps(obj, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("source", "new"), required=True)
    run(parser.parse_args().arm)
