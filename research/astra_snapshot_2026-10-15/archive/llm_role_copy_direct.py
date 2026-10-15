"""Post-result diagnostic: strict direct answers from the copy-only control."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

import llm_role_copy_control as copy
import llm_role_independent_eval as ewt
import llm_role_natural_repair as first
from llm_role_natural_contrast_screen import prompt
from llm_role_natural_repair_fresh_direct import normalize

ROOT = Path(__file__).resolve().parent / "llm_role_copy_control"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.inference_mode()
def main():
    copy.validate()
    rows = ewt.validate()
    target = ROOT / "DIRECT_EWT.json"
    if target.exists():
        raise FileExistsError(target)
    torch.set_num_threads(2)
    tok, model = first.load_source()
    weights = torch.load(copy.ADAPTER, map_location="cpu", weights_only=True)
    params = {name: value for name, value in model.named_parameters() if value.requires_grad}
    assert params.keys() == weights.keys()
    for name, value in weights.items():
        params[name].data.copy_(value)
    model.eval()
    records = []
    for i, row in enumerate(rows, 1):
        answers = {}
        for role in ("actor", "other"):
            p = tok.encode(prompt(tok, row, role), add_special_tokens=False)
            generated = model.generate(input_ids=torch.tensor([p]), max_new_tokens=16,
                                       do_sample=False, pad_token_id=tok.eos_token_id,
                                       eos_token_id=tok.eos_token_id, use_cache=True)
            answer = tok.decode(generated[0, len(p):], skip_special_tokens=True).strip()
            answers[role] = {"raw": answer, "gold": row[f"{role}_span"],
                             "exact": normalize(answer) == normalize(row[f"{role}_span"])}
        records.append({"id": row["id"], "answers": answers})
        if i % 25 == 0:
            print(f"copy direct {i}/{len(rows)}", flush=True)
    summary = {role: sum(r["answers"][role]["exact"] for r in records)
               for role in ("actor", "other")}
    summary["both"] = sum(r["answers"]["actor"]["exact"] and
                          r["answers"]["other"]["exact"] for r in records)
    target.write_text(json.dumps({"status": "post_result_diagnostic", "records": records,
                                  "summary": summary, "protocol_sha256": sha(ROOT / "PROTOCOL.json"),
                                  "ewt_events_sha256": sha(ewt.ROOT / "EVENTS.json"),
                                  "adapter_sha256": sha(copy.ADAPTER),
                                  "source_code_sha256": sha(Path(__file__))}, indent=2)+"\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
