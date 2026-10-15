#!/usr/bin/env python3
"""COMPUT-009: transparent closed-grammar behavioral falsification calibration.
Fully synthetic; no LLM inference, sealed data, or gold-label predictor features.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path

PAIRS=(("Amy","Ben"),("Clara","David"),("Eva","Frank"),
       ("Grace","Henry"),("Iris","Jack"),("Kate","Leo"))
VERBS=("praised","warned","assisted","thanked")
FORMS=("active","passive","subject_cleft","object_cleft")
VARIANTS=("canonical","as_for_actor","as_for_patient",
          "regarding_actor","regarding_patient","parenthetical")
NAME_RE=re.compile(r"\b[A-Z][a-z]+\b")
EXCLUDE=frozenset(("As","Regarding","It","Who"))
QUESTIONS={"actor":"Who performed the action?","patient":"Who received the action?"}

def core_sentence(actor,patient,verb,form):
    return {
        "active":f"{actor} {verb} {patient}.",
        "passive":f"{patient} was {verb} by {actor}.",
        "subject_cleft":f"It was {actor} who {verb} {patient}.",
        "object_cleft":f"It was {patient} whom {actor} {verb}.",
    }[form]

def transform(core,actor,patient,form,variant):
    if variant=="canonical": return core
    if variant.startswith("as_for_") or variant.startswith("regarding_"):
        prefix_type,role=variant.rsplit("_",1)
        prefix="As for" if prefix_type=="as_for" else "Regarding"
        return f"{prefix} {actor if role=='actor' else patient}, {core}"
    if variant=="parenthetical":
        if form=="active":return core.replace(f"{actor} ",f"{actor}, unexpectedly, ",1)
        if form=="passive":return core.replace(f"{patient} ",f"{patient}, unexpectedly, ",1)
        if form=="subject_cleft":return core.replace(f"{actor} who ",f"{actor}, unexpectedly, who ",1)
        if form=="object_cleft":return core.replace(f"{actor} ",f"{actor}, unexpectedly, ",1)
    raise ValueError(variant)

def generate():
    rows=[]
    for pair_id,(p,q) in enumerate(PAIRS):
        for verb in VERBS:
            for direction,(actor,patient) in enumerate(((p,q),(q,p))):
                for form in FORMS:
                    core=core_sentence(actor,patient,verb,form)
                    for variant in VARIANTS:
                        sentence=transform(core,actor,patient,form,variant)
                        for role,question in QUESTIONS.items():
                            rows.append({"pair_id":pair_id,"predicate":verb,"orientation":direction,
                                         "form":form,"variant":variant,"sentence":sentence,
                                         "question":question,"gold":actor if role=="actor" else patient})
    return rows

def extract_mentions(text):
    return [x for x in NAME_RE.findall(text) if x not in EXCLUDE]

def question_role(question):
    return {v:k for k,v in QUESTIONS.items()}[question]

def construction(text):
    if re.search(r"\bwhom\b",text):return "object_cleft"
    if re.search(r"\bwho\b",text):return "subject_cleft"
    if re.search(r"\bwas (?:praised|warned|assisted|thanked) by\b",text):return "passive"
    if any(re.search(r"\b"+v+r"\b",text) for v in VERBS):return "active"
    raise ValueError("Construction not found")

def order_router(text,question):
    names=list(dict.fromkeys(extract_mentions(text)))
    if len(names)!=2:return None
    actor_first=construction(text) in ("active","subject_cleft")
    if question_role(question)=="patient":actor_first=not actor_first
    return names[0] if actor_first else names[1]

def after_first_comma(text,question):
    return order_router(text.split(", ",1)[1] if ", " in text else text,question)

def prefix_and_parenthetical_router(text,question):
    text=re.sub(r"^(?:As for|Regarding) [A-Z][a-z]+, ","",text)
    text=text.replace(", unexpectedly, "," ")
    return order_router(text,question)

PREDICTORS={"global_order":order_router,
            "after_first_comma":after_first_comma,
            "fixed_surface_grammar":prefix_and_parenthetical_router}

def result(rows):
    cells={}
    for variant in VARIANTS:
        rr=[r for r in rows if r["variant"]==variant]
        cells[variant]={name:{
            "correct":sum(f(r["sentence"],r["question"])==r["gold"] for r in rr),
            "abstain":sum(f(r["sentence"],r["question"]) is None for r in rr),
            "total":len(rr)} for name,f in PREDICTORS.items()}
    paired={}
    for marker in ("as_for","regarding"):
        groups={}
        for r in rows:
            if r["variant"] not in (marker+"_actor",marker+"_patient"):continue
            k=(r["pair_id"],r["predicate"],r["orientation"],r["form"],r["question"])
            groups.setdefault(k,{})[r["variant"].rsplit("_",1)[1]]=r
        assert all(set(pair)=={"actor","patient"} for pair in groups.values())
        flips=stable=0
        for pair in groups.values():
            a,b=pair["actor"],pair["patient"]
            flips+=order_router(a["sentence"],a["question"])!=order_router(b["sentence"],b["question"])
            stable+=prefix_and_parenthetical_router(a["sentence"],a["question"])==prefix_and_parenthetical_router(b["sentence"],b["question"])
        paired[marker]={"comparisons":len(groups),"global_order_flips":flips,
                         "fixed_surface_grammar_stable":stable}
    return {"dataset_type":"synthetic_generated","n":len(rows),
            "n_source_events":len(rows)//(len(VARIANTS)*len(QUESTIONS)),
            "per_variant":cells,"counterfactuals":paired,
            "limit":"Closed-grammar handcrafted heuristic success is not an internal relational mechanism."}

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--results",type=Path);p.add_argument("--stimuli",type=Path)
    a=p.parse_args();rows=generate();report=result(rows)
    if a.results:a.results.write_text(json.dumps(report,indent=2)+"\n",encoding="utf8")
    if a.stimuli:a.stimuli.write_text("".join(json.dumps(r)+"\n" for r in rows),encoding="utf8")
    print(json.dumps(report,indent=2))
