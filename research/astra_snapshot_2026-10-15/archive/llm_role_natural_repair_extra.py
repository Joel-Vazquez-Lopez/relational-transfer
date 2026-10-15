"""Post-screen diagnostics for the exploratory natural-role adaptation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import torch

import llm_role_natural_repair as repair
from llm_role_natural_contrast_screen import prompt, score


HERE = Path(__file__).resolve().parent
ROOT = HERE / "llm_role_natural_repair"
EVENTS = HERE / "llm_role_natural_contrast_confirm/EVENTS.json"
NEW_ADAPTER = ROOT / "mixed_replay/adapter.pt"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold().strip().strip(" .,'\"!?"))


def load(arm: str):
    tok, model = repair.load_source()
    if arm == "new":
        saved = torch.load(NEW_ADAPTER, map_location="cpu", weights_only=True)
        params = {name: value for name, value in model.named_parameters() if value.requires_grad}
        assert saved.keys() == params.keys()
        for name, param in params.items():
            param.data.copy_(saved[name])
    model.eval()
    return tok, model


@torch.inference_mode()
def run(arm: str) -> None:
    repair.validate()
    path = ROOT / f"EXTRA_{arm}.json"
    if path.exists():
        raise FileExistsError(path)
    torch.set_num_threads(2)
    tok, model = load(arm)
    events = json.loads(EVENTS.read_text())
    records = []
    for i, row in enumerate(events, 1):
        omitted = {**row, "sentence": "[omitted]"}
        scores = {role: {candidate: score(tok, model, prompt(tok, omitted, role),
                                          row[f"{candidate}_span"])[0]
                         for candidate in ("actor", "other")}
                  for role in ("actor", "other")}
        no_sentence_margin = (scores["actor"]["actor"] - scores["actor"]["other"]
                              - scores["other"]["actor"] + scores["other"]["other"])
        answers = {}
        for role in ("actor", "other"):
            prefix = tok.encode(prompt(tok, row, role), add_special_tokens=False)
            result = model.generate(input_ids=torch.tensor([prefix]),
                                    max_new_tokens=16, do_sample=False,
                                    pad_token_id=tok.eos_token_id,
                                    eos_token_id=tok.eos_token_id,
                                    use_cache=True)
            generated = tok.decode(result[0, len(prefix):], skip_special_tokens=True).strip()
            answers[role] = {"raw": generated,
                             "exact": normalize(generated) == normalize(row[f"{role}_span"])}
        records.append({"id": row["id"], "no_sentence_margin": no_sentence_margin,
                        "no_sentence_correct": no_sentence_margin > 0,
                        "answers": answers})
        if i % 10 == 0:
            print(f"{arm}: {i}/{len(events)}", flush=True)
    result = {"arm": arm, "source_code_sha256": sha(HERE / "llm_role_natural_repair_extra.py"),
              "repair_protocol_sha256": sha(ROOT / "PROTOCOL.json"),
              "events_sha256": sha(EVENTS),
              "adapter_sha256": sha(NEW_ADAPTER if arm == "new" else repair.SOURCE_ADAPTER),
              "records": records,
              "no_sentence_correct": sum(x["no_sentence_correct"] for x in records),
              "actor_exact": sum(x["answers"]["actor"]["exact"] for x in records),
              "other_exact": sum(x["answers"]["other"]["exact"] for x in records),
              "both_exact": sum(all(x["answers"][role]["exact"] for role in ("actor", "other"))
                                for x in records)}
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("no_sentence_correct", "actor_exact", "other_exact", "both_exact")}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("source", "new"), required=True)
    run(parser.parse_args().arm)
