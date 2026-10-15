"""Score dense-selected ordinary checkpoints on fresh3 unused question wordings."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import llm_role_repair_confirm as confirm  # noqa: E402
import llm_role_repair_v3 as core  # noqa: E402


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(seed):
    selection = json.loads((HERE / "SELECTION.json").read_text())
    assert selection["dense_protocol_sha256"] == sha(HERE / "PROTOCOL.json")
    steps = sorted({item["selected"]["step"] for item in selection["choices"].values()
                    if item["selected"] and item["selected"]["step"] in (25, 30, 35)})
    assert steps == [25, 30]
    groups, all_rows = confirm.validate("fresh3")
    assert len(groups) == 60
    rows = [r for r in all_rows if r["form"] in ("active", "passive", "patient_cleft")
            and r["wording"] in ("did", "carried_out")]
    assert len(rows) == 1500
    confirm.configure("fresh3", seed)
    for step in steps:
        output = HERE / f"confirm_fresh3_{seed}_step_{step:03d}.json"
        if output.exists():
            print(f"Already complete: {output}", flush=True)
            continue
        tokenizer, model, heads = core.load("ordinary")
        assert heads is None
        adapter = torch.load(HERE / f"fresh3_{seed}_ordinary" / f"step_{step:03d}" / "adapter.pt",
                             map_location="cpu", weights_only=True)["adapter"]
        found = 0
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                assert name in adapter and parameter.shape == adapter[name].shape
                parameter.data.copy_(adapter[name])
                found += 1
        assert found == len(adapter)
        records = core.evaluate(f"confirm_fresh3_{seed}_step_{step}", tokenizer, model, rows)
        scores = {}
        for form in ("active", "passive", "patient_cleft"):
            subset = [r for r in records if r["form"] == form]
            scores[form] = {"correct": sum(r["correct"] for r in subset), "total": len(subset)}
        output.write_text(json.dumps({"world": "fresh3", "seed": seed, "step": step,
                                      "selection_sha256": sha(HERE / "SELECTION.json"),
                                      "dense_protocol_sha256": sha(HERE / "PROTOCOL.json"),
                                      "scores": scores, "records": records}, indent=2) + "\n")
        print(json.dumps({"seed": seed, "step": step, "scores": scores}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(830211, 830212), required=True)
    main(parser.parse_args().seed)
