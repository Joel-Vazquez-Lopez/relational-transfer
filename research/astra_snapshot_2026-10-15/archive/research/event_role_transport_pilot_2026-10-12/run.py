#!/usr/bin/env python3
"""Exploratory matched slot-supervision versus event-role transport pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parents[1]
sys.path.insert(0, str(ROOT.parent / "role_prompt_equivariance_pilot_2026-10-12"))
sys.path.insert(0, str(ROOT.parent / "role_subject_decoupling_pilot_2026-10-12"))
import pilot  # noqa: E402
import experiment as prev  # noqa: E402

PAIRS = (("Charlie", "Dave"), ("Joy", "Kyle"), ("Meg", "Pat"),
         ("Ray", "Ted"), ("Val", "Will"), ("Andy", "Carol"))
VERBS = ("grabbed", "called", "poked")
TRAIN_FORMS = ("active", "passive", "object_cleft")
TEST_FORMS = pilot.FORMS
TRAIN_Q = pilot.TRAIN_Q
TEST_Q = pilot.TEST_Q
STARTS = prev.STARTS
ARMS = ("slot_sft", "transport")
EPOCHS = 2
LR = prev.LR
SEED = prev.SEED
L_INV = 0.5
L_SWAP = 0.5
L_ROUTE = 0.5


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train_units():
    all_events = prev.train_events()
    return [(all_events[i], all_events[i+1]) for i in range(0, len(all_events), 2)]


def test_rows():
    return [pilot.question_row(e, f, q, txt)
            for e in prev.events(PAIRS, VERBS)
            for f in TEST_FORMS for q, txt in TEST_Q.items()]


def slot_prefix(tok, event, form):
    return tok.apply_chat_template([
        {"role": "system", "content": "Give the two event roles exactly as Actor: NAME; Patient: NAME"},
        {"role": "user", "content":
         f"Sentence: {prev.sentence(event, form)}\nCandidates: {', '.join(event['pair'])}\nIdentify the event roles."},
    ], tokenize=False, add_generation_prompt=True) + "Actor:"


def entry(tok, event, form, qname=None):
    if qname is not None:
        r = pilot.question_row(event, form, qname, TRAIN_Q[qname])
        p = prev.prompt(tok, r)
        prefix = tok.encode(p, add_special_tokens=False)
        ans = tok.encode(r["gold"], add_special_tokens=False)
        assert len(ans) == 1
        full = prefix + ans + [tok.eos_token_id]
        labels = [-100]*len(prefix) + ans + [tok.eos_token_id]
        return full, labels, {"answer": len(prefix)-1}
    p = slot_prefix(tok, event, form)
    s1 = f" {event['actor']}; Patient:"
    s2 = f" {event['patient']}"
    a = tok.encode(p, add_special_tokens=False)
    b = tok.encode(p+s1, add_special_tokens=False)
    c = tok.encode(p+s1+s2, add_special_tokens=False)
    assert b[:len(a)] == a and c[:len(b)] == b
    assert len(tok.encode(" " + event["actor"], add_special_tokens=False)) == 1
    assert len(tok.encode(" " + event["patient"], add_special_tokens=False)) == 1
    assert len(c) == len(b)+1
    full = c + [tok.eos_token_id]
    labels = [-100]*len(a) + c[len(a):] + [tok.eos_token_id]
    return full, labels, {"actor": len(a)-1, "patient": len(b)-1}


def pack(tok, pair):
    entries, places = [], []
    for event in pair:
        for form in TRAIN_FORMS:
            group = {}
            for q in TRAIN_Q:
                item = entry(tok, event, form, q)
                group[q] = len(entries)
                entries.append(item)
            group["slot"] = len(entries)
            entries.append(entry(tok, event, form))
            places.append(group)
    assert len(entries) == 24 and len(places) == 6
    length = max(len(x[0]) for x in entries)
    ids, labs, masks, positions = [], [], [], []
    for toks, labels, pos in entries:
        pad = length-len(toks)
        ids.append([tok.pad_token_id]*pad+toks)
        labs.append([-100]*pad+labels)
        masks.append([0]*pad+[1]*len(toks))
        positions.append({k: v+pad for k, v in pos.items()})
    return torch.tensor(ids), torch.tensor(labs), torch.tensor(masks), positions, places


def js_pair(x, y):
    m = (x+y)/2
    return ((x * (x.clamp_min(1e-8).log()-m.clamp_min(1e-8).log())).sum() +
            (y * (y.clamp_min(1e-8).log()-m.clamp_min(1e-8).log())).sum())/2


def transport_losses(logits, positions, places, qa_ids, slot_ids):
    probs = {}
    for gi, group in enumerate(places):
        for q in TRAIN_Q:
            row = group[q]
            z = logits[row, positions[row]["answer"], list(qa_ids)]
            probs[(gi, q)] = F.softmax(z, dim=-1)
        row = group["slot"]
        for role in ("actor", "patient"):
            z = logits[row, positions[row][role], list(slot_ids)]
            probs[(gi, role)] = F.softmax(z, dim=-1)
    inv = swap = route = torch.zeros((), dtype=logits.dtype)
    for direction in range(2):
        gs = [direction*3+i for i in range(3)]
        for role in ("actor", "patient"):
            inv = inv + js_pair(probs[(gs[0], role)], probs[(gs[1], role)])
            inv = inv + js_pair(probs[(gs[0], role)], probs[(gs[2], role)])
        for g in gs:
            route = route + js_pair(probs[(g, "actor")], probs[(g, "actor_performed")])
            route = route + js_pair(probs[(g, "actor")], probs[(g, "actor_did")])
            route = route + js_pair(probs[(g, "patient")], probs[(g, "patient_verb")])
    for fi in range(3):
        swap = swap + js_pair(probs[(fi, "actor")], probs[(fi+3, "patient")])
        swap = swap + js_pair(probs[(fi, "patient")], probs[(fi+3, "actor")])
    return inv/8, swap/6, route/18


def prepare():
    target = ROOT/"PROTOCOL.json"
    if target.exists(): raise FileExistsError(target)
    tok, model = prev.load("fresh1")
    for name in (n for pair in PAIRS for n in pair):
        assert len(tok.encode(name, add_special_tokens=False)) == 1, name
        assert len(tok.encode(" "+name, add_special_tokens=False)) == 1, name
    all_prior = prev.TRAIN_PAIRS + prev.TEST_PAIRS + pilot.TEST_PAIRS
    assert not ({n for p in PAIRS for n in p} & {n for p in all_prior for n in p})
    assert not set(VERBS) & (set(prev.TRAIN_VERBS) | set(prev.TEST_VERBS) | set(pilot.TEST_VERBS))
    unit = train_units()[0]
    pack(tok, unit)
    del model
    data = {
        "date":"2026-10-12", "status":"exploratory; frozen before training or inference on this new lexical bank",
        "hypothesis":"Role-slot invariance, reversal, and answer grounding improve strict held-out two-role QA beyond identical QA and slot supervision.",
        "arms":ARMS,"starts":STARTS,"train_forms":TRAIN_FORMS,"test_forms":TEST_FORMS,
        "train_question_wordings":TRAIN_Q,"test_question_wordings":TEST_Q,
        "test_pairs":PAIRS,"test_verbs":VERBS,"train_units":len(train_units()),
        "epochs":EPOCHS,"updates":len(train_units())*EPOCHS,"rows_per_update":24,
        "lambdas":{"invariance":L_INV,"reversal":L_SWAP,"grounding":L_ROUTE},
        "matched":"Both arms see identical 24 supervised rows per update: two directions × three forms × (three QA + ordered slot tuple); same answer labels, updates, source adapters, optimizer, event order. Only transport arm adds distribution constraints.",
        "primary":"Event-forms /144 where both held-out actor_carried and patient_which answers are correct on 36 new events × four forms. Compare transport minus slot_sft in each start.",
        "screen":"At least +8/144 strict joint in both starts, no worse than -4/36 active or passive strict joint, and positive object-cleft strict joint in both starts. Full claim needs independent natural and second-family confirmation.",
        "limitations":"One 0.5B family, two related starts, constructed English; even a positive screen is not a conference-level repair claim. No human-review data accessed.",
        "source_sha256":{s:sha(prev.source_path(s)) for s in STARTS},
        "script_sha256":sha(Path(__file__)),
        "train_sha256":hashlib.sha256(json.dumps(train_units(),sort_keys=True).encode()).hexdigest(),
        "test_sha256":hashlib.sha256(json.dumps(test_rows(),sort_keys=True).encode()).hexdigest(),
    }
    target.write_text(json.dumps(data,indent=2)+"\n")
    print("Frozen",len(train_units()),"training units and",len(test_rows()),"test prompts",flush=True)


def validate():
    cfg=json.loads((ROOT/"PROTOCOL.json").read_text())
    assert cfg["script_sha256"]==sha(Path(__file__))
    assert cfg["train_sha256"]==hashlib.sha256(json.dumps(train_units(),sort_keys=True).encode()).hexdigest()
    assert cfg["test_sha256"]==hashlib.sha256(json.dumps(test_rows(),sort_keys=True).encode()).hexdigest()
    for s in STARTS: assert cfg["source_sha256"][s]==sha(prev.source_path(s))
    return cfg


def train(start, arm):
    cfg=validate(); target=ROOT/f"{start}_{arm}"
    if target.exists(): raise FileExistsError(target)
    torch.set_num_threads(2)
    tok,model=prev.load(start)
    torch.manual_seed(SEED)
    opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=LR)
    units=train_units(); order=list(range(len(units))); rng=random.Random(SEED)
    trace=[]; began=time.perf_counter(); model.train()
    for epoch in range(EPOCHS):
        rng.shuffle(order)
        for idx in order:
            pair=units[idx]
            ids,labels,mask,positions,places=pack(tok,pair)
            opt.zero_grad(set_to_none=True)
            logits=model(input_ids=ids,attention_mask=mask,use_cache=False).logits
            losses=F.cross_entropy(logits[:,:-1].contiguous().flatten(0,1),
                                   labels[:,1:].contiguous().flatten(),ignore_index=-100,
                                   reduction="none").view(24,-1)
            counts=(labels[:,1:]!=-100).sum(dim=1)
            ce=(losses.sum(dim=1)/counts).mean()
            inv=swap=route=torch.zeros((),dtype=ce.dtype)
            if arm=="transport":
                names=pair[0]["pair"]
                qa_ids=tuple(tok.encode(n,add_special_tokens=False)[0] for n in names)
                slot_ids=tuple(tok.encode(" "+n,add_special_tokens=False)[0] for n in names)
                inv,swap,route=transport_losses(logits,positions,places,qa_ids,slot_ids)
            loss=ce+L_INV*inv+L_SWAP*swap+L_ROUTE*route
            if not torch.isfinite(loss): raise FloatingPointError((start,arm,idx))
            loss.backward(); opt.step()
            trace.append({"ce":float(ce.detach()),"inv":float(inv.detach()),
                          "swap":float(swap.detach()),"route":float(route.detach())})
            if len(trace)%12==0:
                print(f"{start}/{arm} {len(trace)}/48 elapsed={time.perf_counter()-began:.0f}s",flush=True)
    target.mkdir()
    torch.save({n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad},target/"adapter.pt")
    (target/"TRAINING.json").write_text(json.dumps({
        "protocol_sha256":sha(ROOT/"PROTOCOL.json"),"source_sha256":cfg["source_sha256"][start],
        "checkpoint_sha256":sha(target/"adapter.pt"),"start":start,"arm":arm,
        "updates":len(trace),"supervised_rows":len(trace)*24,
        "elapsed_seconds":time.perf_counter()-began,"trace":trace},indent=2)+"\n")
    print("Saved",target.name,flush=True)


@torch.inference_mode()
def evaluate(start,arm):
    validate(); target=ROOT/f"EVAL_{start}_{arm}.json"
    if target.exists(): raise FileExistsError(target)
    torch.set_num_threads(2)
    tok,model=prev.load(start)
    cp=ROOT/f"{start}_{arm}"/"adapter.pt"
    saved=torch.load(cp,map_location="cpu",weights_only=True)
    params={n:p for n,p in model.named_parameters() if p.requires_grad}
    assert saved.keys()==params.keys()
    for n,p in params.items(): p.data.copy_(saved[n])
    model.eval(); out=[]; rows=test_rows()
    for off in range(0,len(rows),8):
        block=rows[off:off+8]
        inputs=tok([prev.prompt(tok,r) for r in block],padding=True,return_tensors="pt")
        logits=model(**inputs,use_cache=False).logits[:,-1]
        for r,z in zip(block,logits):
            ans=tok.decode([int(z.argmax())]).strip()
            out.append({"event":r["id"],"form":r["form"],"wording":r["wording"],
                        "role":r["role"],"gold":r["gold"],"answer":ans,
                        "correct":ans==r["gold"],"valid":ans in r["pair"]})
        if (off+8)%180==0: print(start,arm,off+8,"/",len(rows),flush=True)
    assert len(out)==720
    target.write_text(json.dumps({"start":start,"arm":arm,
        "protocol_sha256":sha(ROOT/"PROTOCOL.json"),"checkpoint_sha256":sha(cp),
        "records":out},indent=2)+"\n")
    print("Saved",target.name,flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("mode",choices=("prepare","train","eval"))
    ap.add_argument("--start",choices=STARTS); ap.add_argument("--arm",choices=ARMS)
    args=ap.parse_args()
    if args.mode=="prepare": prepare()
    elif args.start is None or args.arm is None: ap.error("train/eval require start and arm")
    elif args.mode=="train": train(args.start,args.arm)
    else: evaluate(args.start,args.arm)
