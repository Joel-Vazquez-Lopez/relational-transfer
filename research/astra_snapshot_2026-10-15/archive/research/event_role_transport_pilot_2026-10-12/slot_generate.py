#!/usr/bin/env python3
"""Post hoc free-generation slot diagnostic on already-inspected lexical bank."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import run  # noqa: E402
import slot_check  # noqa: E402


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    target=ROOT/"SLOT_GENERATE_PROTOCOL.json"
    if target.exists(): raise FileExistsError(target)
    slot_check.validate()
    cfg={"date":"2026-10-12",
         "status":"post_hoc_free_generation_diagnostic; tuple-rank and direct-QA results seen",
         "cases":"36 held-out directed lexical events × all four forms = 144/checkpoint",
         "prompt":"The same structured slot prompt as training, ending with Actor:",
         "decoding":"greedy, max_new_tokens=7, no sampling; parse first actor and patient candidate names only",
         "metric":"Exact both-role slot generation; actor and patient separately; unparseable outputs counted wrong",
         "limits":"The lexical bank and output behavior were already inspected; exploratory output-level check, not independent confirmation or internal mechanism evidence.",
         "script_sha256":sha(Path(__file__)),
         "main_protocol_sha256":sha(ROOT/"PROTOCOL.json"),
         "slot_check_protocol_sha256":sha(ROOT/"SLOT_CHECK_PROTOCOL.json"),
         "checkpoints":{f"{s}_{a}":sha(ROOT/f"{s}_{a}"/"adapter.pt")
                        for s in run.STARTS for a in run.ARMS}}
    target.write_text(json.dumps(cfg,indent=2)+"\n")
    print("Frozen free-generation slot diagnostic",flush=True)


def validate():
    cfg=json.loads((ROOT/"SLOT_GENERATE_PROTOCOL.json").read_text())
    assert cfg["script_sha256"]==sha(Path(__file__))
    assert cfg["main_protocol_sha256"]==sha(ROOT/"PROTOCOL.json")
    assert cfg["slot_check_protocol_sha256"]==sha(ROOT/"SLOT_CHECK_PROTOCOL.json")
    for key,h in cfg["checkpoints"].items(): assert sha(ROOT/key/"adapter.pt")==h
    return cfg


@torch.inference_mode()
def evaluate(start,arm):
    cfg=validate(); target=ROOT/f"GENERATE_{start}_{arm}.json"
    if target.exists(): raise FileExistsError(target)
    torch.set_num_threads(2)
    tok,model=run.prev.load(start)
    cp=ROOT/f"{start}_{arm}"/"adapter.pt"
    saved=torch.load(cp,map_location="cpu",weights_only=True)
    params={n:p for n,p in model.named_parameters() if p.requires_grad}
    assert saved.keys()==params.keys()
    for n,p in params.items(): p.data.copy_(saved[n])
    model.eval(); out=[]; cases=slot_check.cases()
    for off in range(0,len(cases),8):
        block=cases[off:off+8]
        prompts=[slot_check.prefix(tok,e,f,True) for e,f in block]
        batch=tok(prompts,padding=True,return_tensors="pt",add_special_tokens=False)
        generated=model.generate(**batch,max_new_tokens=7,do_sample=False,
                                 pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
        suffixes=generated[:,batch["input_ids"].shape[1]:]
        for (e,f),tokens in zip(block,suffixes):
            raw=tok.decode(tokens,skip_special_tokens=True)
            m=re.match(r"^\s*([A-Za-z]+)\s*;\s*Patient:\s*([A-Za-z]+)",raw)
            actor=m.group(1) if m else None
            patient=m.group(2) if m else None
            out.append({"event":e["id"],"form":f,"raw":raw,"actor":actor,"patient":patient,
                        "actor_correct":actor==e["actor"],
                        "patient_correct":patient==e["patient"],
                        "joint_correct":actor==e["actor"] and patient==e["patient"]})
        if (off+8)%48==0: print(start,arm,off+8,"/",len(cases),flush=True)
    assert len(out)==144
    target.write_text(json.dumps({"start":start,"arm":arm,
        "protocol_sha256":sha(ROOT/"SLOT_GENERATE_PROTOCOL.json"),
        "checkpoint_sha256":cfg["checkpoints"][f"{start}_{arm}"],
        "records":out},indent=2)+"\n")
    print("Saved",target.name,flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("mode",choices=("prepare","eval"))
    ap.add_argument("--start",choices=run.STARTS);ap.add_argument("--arm",choices=run.ARMS)
    args=ap.parse_args()
    if args.mode=="prepare":prepare()
    else:
        if args.start is None or args.arm is None:ap.error("eval requires start and arm")
        evaluate(args.start,args.arm)
