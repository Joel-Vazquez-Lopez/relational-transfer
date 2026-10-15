"""Strict generated-span diagnostic for both natural-only pilot checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

import llm_role_natural_repair as first
import llm_role_natural_repair_fresh as fresh
import llm_role_natural_repair_replication as rep
from llm_role_natural_contrast_screen import prompt
from llm_role_natural_repair_fresh_direct import normalize


HERE = Path(__file__).resolve().parent


@torch.inference_mode()
def run(start: str) -> None:
    events = fresh.validate()
    if start == "fresh1":
        root = HERE / "llm_role_natural_repair/controls_fresh1_830211/natural_only"
        tok, model = first.load_source()
    else:
        root = HERE / "llm_role_natural_repair/natural_only_fresh3_830212"
        tok, model = rep.load_source()
    target = root / "DIRECT.json"
    if target.exists():
        raise FileExistsError(target)
    torch.set_num_threads(2)
    weights = torch.load(root / "adapter.pt", map_location="cpu", weights_only=True)
    params = {name: value for name, value in model.named_parameters() if value.requires_grad}
    assert weights.keys() == params.keys()
    for name, param in params.items():
        param.data.copy_(weights[name])
    model.eval()
    rows = []
    for i, event in enumerate(events, 1):
        answers = {}
        for role in ("actor", "other"):
            prefix = tok.encode(prompt(tok, event, role), add_special_tokens=False)
            generated = model.generate(input_ids=torch.tensor([prefix]), max_new_tokens=16,
                                       do_sample=False, pad_token_id=tok.eos_token_id,
                                       eos_token_id=tok.eos_token_id, use_cache=True)
            answer = tok.decode(generated[0, len(prefix):], skip_special_tokens=True).strip()
            answers[role] = {"raw": answer, "gold": event[f"{role}_span"],
                             "exact": normalize(answer) == normalize(event[f"{role}_span"])}
        rows.append({"id": event["id"], "answers": answers})
        if i % 25 == 0:
            print(f"{start}: {i}/{len(events)}", flush=True)
    summary = {role: sum(x["answers"][role]["exact"] for x in rows)
               for role in ("actor", "other")}
    summary["both"] = sum(all(x["answers"][role]["exact"] for role in ("actor", "other"))
                          for x in rows)
    target.write_text(json.dumps({"start": start, "summary": summary, "records": rows,
                                  "fresh_protocol_sha256": hashlib.sha256(
                                      (fresh.ROOT / "PROTOCOL.json").read_bytes()).hexdigest(),
                                  "adapter_sha256": hashlib.sha256((root / "adapter.pt").read_bytes()).hexdigest(),
                                  "source_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                  "limits": "Post-screen strict direct-generation diagnostic, not part of the frozen candidate-contrast gate."},
                                 indent=2)+"\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", choices=("fresh1", "fresh3"), required=True)
    run(parser.parse_args().start)
