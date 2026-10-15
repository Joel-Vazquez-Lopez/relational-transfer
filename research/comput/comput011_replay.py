#!/usr/bin/env python3
"""COMPUT-011 concise independent replay. Full audited implementation in conversation ZIP."""
import argparse,collections,hashlib,json,re
from pathlib import Path
SHA="653a33a0a56f34a84c5c39649dda7bdb100990462daea0390cfcb2fcb8d7ea27"
ACTIVE=re.compile(r"^([A-Z][a-z]+) ([a-z]+ed) ([A-Z][a-z]+)\.$")
PASSIVE=re.compile(r"^([A-Z][a-z]+) was ([a-z]+ed) by ([A-Z][a-z]+)\.$")
def features(s,q):
    a=ACTIVE.fullmatch(s);p=PASSIVE.fullmatch(s)
    if bool(a)==bool(p):raise ValueError("Unknown sentence grammar")
    first,verb,last=(a or p).groups()
    w=("generic_actor" if q=="Who did the action?" else
       "performed_actor" if q=="Who performed the action?" else
       "verb_patient" if q==f"Who was {verb}?" else None)
    if not w:raise ValueError("Unknown question")
    return (first,last,"active" if a else "passive",w,verb)
def analyze(path):
    raw=Path(path).read_bytes();sha=hashlib.sha256(raw).hexdigest()
    if sha!=SHA:raise ValueError("Original source SHA-256 mismatch: "+sha)
    data=json.loads(raw);rows=data["records"]
    if len(rows)!=384:raise ValueError("Expected 384 records")
    grouped=collections.defaultdict(list);entries=[];seen=set()
    for r in rows:
        first,last,voice,w,verb=features(r["sentence"],r["question"])
        key=(r["pair_index"],r["direction"],verb,voice,w)
        if key in seen:raise ValueError("Duplicate")
        seen.add(key)
        if (voice,verb,w)!=(r["voice"],r["verb"],r["wording"]):raise ValueError("Metadata mismatch")
        if {first,last}!={r["actor"],r["patient"]}:raise ValueError("Names mismatch")
        if r["valid"]!=(r["first_token"] in (first,last)) or r["correct"]!=(r["first_token"]==r["gold"]):raise ValueError("Flag mismatch")
        entries.append((r,first,last,voice,w,verb))
        grouped[(r["pair_index"],r["direction"],verb,voice)].append(r)
    if len(grouped)!=128 or any(len(v)!=3 for v in grouped.values()):raise ValueError("Unbalanced coverage")
    def fitted(rows):
        counts=collections.defaultdict(collections.Counter)
        for r,f,l,v,w,verb in rows:counts[(w,v)]["first" if r["first_token"]==f else "last"]+=1
        return {k:"first" if c["first"]>c["last"] else "last" for k,c in counts.items()}
    def eval_fit(test,fit):
        return sum(r["first_token"]==(f if fit.get((w,v),"last")=="first" else l) for r,f,l,v,w,verb in test)
    fit=fitted(entries);last=sum(r["first_token"]==l for r,f,l,v,w,verb in entries)
    exceptions=[{"pair_index":r["pair_index"],"verb":verb,"direction":r["direction"],"sentence":r["sentence"],"question":r["question"],"model_answer":r["first_token"],"baseline_answer":f if fit[(w,v)]=="first" else l} for r,f,l,v,w,verb in entries if r["first_token"]!=(f if fit[(w,v)]=="first" else l)]
    loo={}
    for unit in ("pair_index","verb"):
        acc=0
        for g in sorted({r[unit] for r in rows}):
            train=[e for e in entries if e[0][unit]!=g];test=[e for e in entries if e[0][unit]==g]
            acc+=eval_fit(test,fitted(train))
        loo[unit]=acc
    flips=collections.Counter(); same=0
    for k,rr in grouped.items():
        d={r["wording"]:r for r in rr}
        flips[k[3]]+=d["generic_actor"]["first_token"]!=d["performed_actor"]["first_token"]
        same+=d["generic_actor"]["first_token"]==d["verb_patient"]["first_token"]
    out={"sha256":sha,"records":len(entries),"model_correct":sum(r["correct"] for r in rows),
         "model_valid":sum(r["valid"] for r in rows),"always_last_agreement":last,
         "question_voice_agreement":eval_fit(entries,fit),"first_cells":[list(k) for k,v in fit.items() if v=="first"],
         "exceptions":exceptions,"exceptions_by_verb":dict(collections.Counter(x["verb"] for x in exceptions)),
         "leave_one_out_agreement":loo,"actor_question_paraphrase_flips_by_voice":dict(flips),
         "generic_actor_patient_identical_answers":same}
    assert (out["model_correct"],last,out["question_voice_agreement"],len(exceptions))==(240,336,368,16)
    return out
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("source");p.add_argument("--output");a=p.parse_args()
    s=json.dumps(analyze(a.source),indent=2)+"\n"
    if a.output:Path(a.output).write_text(s)
    else:print(s)
