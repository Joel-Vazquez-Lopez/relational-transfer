"""Post-pilot, separately frozen Bank test check for natural-role adaptation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

import llm_role_natural_repair as repair
from llm_role_natural_contrast_screen import prompt, score


HERE = Path(__file__).resolve().parent
ROOT = HERE / "llm_role_natural_repair/fresh_bank_test"
GOLD = HERE / "llm_role_natural_contrast_confirm/EVENTS.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def document_id(row: dict) -> str | None:
    parts = row["id"].split(":")
    return ":".join(parts[2:4]) if parts[1] == "Wiki1k" else None


def rows() -> list[dict]:
    gold = json.loads(GOLD.read_text())
    used_docs = {":".join(x["id"].split(":")[2:4]) for x in gold}
    used_sentences = {x["sentence"].casefold().strip() for x in gold}
    train = json.loads((repair.ROOT / "TRAIN.json").read_text())
    dev = json.loads((repair.ROOT / "DEV.json").read_text())
    used_docs.update(document_id(x) for x in train + dev if document_id(x))
    used_sentences.update(x["sentence"].casefold().strip() for x in train + dev)
    result = [x for x in repair.extract("test")
              if document_id(x) not in used_docs and
              x["sentence"].casefold().strip() not in used_sentences]
    assert len(result) >= 100
    return result


def prepare() -> None:
    ROOT.mkdir(exist_ok=True)
    if (ROOT / "PROTOCOL.json").exists():
        raise FileExistsError(ROOT / "PROTOCOL.json")
    selected = rows()
    (ROOT / "EVENTS.json").write_text(json.dumps(selected, indent=2) + "\n")
    p = {
        "status": "fixed_before_fresh_bank_test_inference",
        "date": "2026-10-12",
        "design": "After observing the positive 51-item Gold exploratory pilot, evaluate both its source participant-only checkpoint and the natural-data-plus-replay checkpoint on all qualifying QA-SRL Bank 2.0 original test events from unseen articles/sentences. This is a post-pilot test selection and one training run, not a preregistered independent replication.",
        "selection": "Apply the identical strict actor/direct-participant extraction rule used for training. Include all Bank orig test events surviving exact normalized-sentence and Wiki1k source-article exclusions against Gold, selected training, and development items. No test result or model score used in selection.",
        "primary": "Paired candidate-relative role contrast D>0 on each event. A useful directional fresh-test signal is at least seven additional correct events for the new checkpoint, a positive mean paired margin change, and synthetic cleft loss no worse than five percentage points (already frozen in the original pilot). Report all items and source domains.",
        "control": "Repeat contrast with the sentence replaced by [omitted], retaining verb and candidate spans. Compare with-sentence minus omitted accuracy and mean margin for each model; a gain solely in the omitted condition is not evidence of better use of sentence context.",
        "limits": "Bank original annotations have at least two agreeing judgments under our extraction rule but have not been independently reviewed by the student. Multiple predicates may come from one sentence/article. These events are not independent training replications. Source and new checkpoints have one shared model/start; current Gold test was already used in study design.",
        "references": {"bank_test_sha256": sha(repair.BANK / "test.jsonl.gz"),
                       "gold_events_sha256": sha(GOLD),
                       "repair_protocol_sha256": sha(repair.ROOT / "PROTOCOL.json"),
                       "source_adapter_sha256": sha(repair.SOURCE_ADAPTER),
                       "new_adapter_sha256": sha(repair.ROOT / "mixed_replay/adapter.pt"),
                       "script_sha256": sha(HERE / "llm_role_natural_repair_fresh.py")},
        "n_events": len(selected),
    }
    (ROOT / "PROTOCOL.json").write_text(json.dumps(p, indent=2) + "\n")
    print(f"Frozen {len(selected)} Bank test events", flush=True)


def validate() -> list[dict]:
    repair.validate()
    p = json.loads((ROOT / "PROTOCOL.json").read_text())
    refs = p["references"]
    for key, path in (("bank_test_sha256", repair.BANK / "test.jsonl.gz"),
                      ("gold_events_sha256", GOLD),
                      ("repair_protocol_sha256", repair.ROOT / "PROTOCOL.json"),
                      ("source_adapter_sha256", repair.SOURCE_ADAPTER),
                      ("new_adapter_sha256", repair.ROOT / "mixed_replay/adapter.pt"),
                      ("script_sha256", HERE / "llm_role_natural_repair_fresh.py")):
        assert refs[key] == sha(path), key
    saved = json.loads((ROOT / "EVENTS.json").read_text())
    assert saved == rows() and len(saved) == p["n_events"]
    return saved


@torch.inference_mode()
def run(arm: str) -> None:
    events = validate()
    target = ROOT / f"{arm}.json"
    if target.exists():
        raise FileExistsError(target)
    torch.set_num_threads(2)
    tok, model = repair.load_source()
    if arm == "new":
        weights = torch.load(repair.ROOT / "mixed_replay/adapter.pt", map_location="cpu", weights_only=True)
        params = {name: value for name, value in model.named_parameters() if value.requires_grad}
        assert weights.keys() == params.keys()
        for name, param in params.items():
            param.data.copy_(weights[name])
    model.eval()
    result = []
    for i, row in enumerate(events, 1):
        margins = {}
        for condition in ("with_sentence", "omitted"):
            x = row if condition == "with_sentence" else {**row, "sentence": "[omitted]"}
            s = {role: {candidate: score(tok, model, prompt(tok, x, role),
                                         row[f"{candidate}_span"])[0]
                        for candidate in ("actor", "other")}
                 for role in ("actor", "other")}
            margins[condition] = (s["actor"]["actor"] - s["actor"]["other"]
                                  - s["other"]["actor"] + s["other"]["other"])
        result.append({"id": row["id"], "with_sentence": margins["with_sentence"],
                       "omitted": margins["omitted"],
                       "correct": margins["with_sentence"] > 0,
                       "omitted_correct": margins["omitted"] > 0})
        if i % 25 == 0:
            print(f"{arm}: {i}/{len(events)}", flush=True)
    obj = {"arm": arm, "protocol_sha256": sha(ROOT / "PROTOCOL.json"),
           "events_sha256": sha(ROOT / "EVENTS.json"), "records": result,
           "correct": sum(x["correct"] for x in result),
           "omitted_correct": sum(x["omitted_correct"] for x in result)}
    target.write_text(json.dumps(obj, indent=2) + "\n")
    print(f"{arm}: {obj['correct']}/{len(result)} with sentence, {obj['omitted_correct']}/{len(result)} omitted", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "validate", "run"))
    parser.add_argument("--arm", choices=("source", "new"))
    args = parser.parse_args()
    if args.action == "prepare": prepare()
    elif args.action == "validate": validate(); print("validated")
    else:
        if not args.arm:
            parser.error("--arm required")
        run(args.arm)
