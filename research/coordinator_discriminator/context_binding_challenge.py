#!/usr/bin/env python3
"""RELATE-001: same-query contextual-binding transplant calibration.

Programmed oracle hypotheses only: NO neural checkpoints or pretrained inference.
Every recipient/donor pair keeps queried verb and role identical.
"""
import json
from collections import Counter
from itertools import permutations, product

NAMES = ("Ada", "Bea", "Cal", "Dee")
VERBS = ("praised", "criticized")
ROLES = ("actor", "patient")
KINDS = ("predicate_swap", "participant_swap", "event_order_swap", "voice_flip")

def worlds():
    for p in permutations(NAMES):
        yield [dict(verb=VERBS[0],actor=p[0],patient=p[1]),
               dict(verb=VERBS[1],actor=p[2],patient=p[3])]

def position(world, verb):
    slots = [i for i,e in enumerate(world) if e["verb"] == verb]
    if len(slots) != 1: raise ValueError("Non-unique predicate")
    return slots[0]

def answer(world, question):
    return world[position(world,question[0])][question[1]]

def render(world, voice):
    if voice == "active":
        s = [f"{e['actor']} {e['verb']} {e['patient']}." for e in world]
    elif voice == "passive":
        s = [f"{e['patient']} was {e['verb']} by {e['actor']}." for e in world]
    else: raise ValueError(voice)
    return " ".join(s)

def transform(world, kind, queried_verb):
    donor = [dict(e) for e in world]
    if kind == "predicate_swap":
        donor[0]["verb"],donor[1]["verb"]=donor[1]["verb"],donor[0]["verb"]
    elif kind == "participant_swap":
        e=donor[position(donor,queried_verb)]
        e["actor"],e["patient"]=e["patient"],e["actor"]
    elif kind == "event_order_swap":
        donor.reverse()
    elif kind == "voice_flip":
        pass
    else: raise ValueError(kind)
    return donor

def idealized_patch(hypothesis, recipient, donor, query):
    """Idealized *functional* hypotheses; not a neural state intervention."""
    if hypothesis=="query_lexical_only":return answer(recipient,query)
    if hypothesis=="event_slot_address":return recipient[position(donor,query[0])][query[1]]
    if hypothesis=="donor_answer_content":return answer(donor,query)
    raise ValueError(hypothesis)

HYPOTHESES=("query_lexical_only","event_slot_address","donor_answer_content")

def generate():
    rows=[]
    for permutation_id,recipient in enumerate(worlds()):
        for voice,verb,role in product(("active","passive"),VERBS,ROLES):
            query=(verb,role)
            for kind in KINDS:
                donor=transform(recipient,kind,verb)
                donor_voice=("passive" if voice=="active" else "active") if kind=="voice_flip" else voice
                rows.append({"permutation_id":permutation_id,"voice":voice,"query_verb":verb,
                  "query_role":role,"intervention":kind,
                  "recipient":render(recipient,voice),"donor":render(donor,donor_voice),
                  "recipient_gold":answer(recipient,query),"donor_gold":answer(donor,query),
                  "recipient_slot":position(recipient,verb),"donor_slot":position(donor,verb),
                  "predictions":{h:idealized_patch(h,recipient,donor,query) for h in HYPOTHESES}})
    return rows

def summarize(rows):
    return {kind:{h:dict(Counter("change" if r["predictions"][h]!=r["recipient_gold"] else "same"
      for r in rows if r["intervention"]==kind)) for h in HYPOTHESES} for kind in KINDS}

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",help="Output development-only paired cases as JSONL")
    a=ap.parse_args()
    rows=generate()
    if a.out:
        with open(a.out,"w",encoding="utf8") as f:
            for r in rows:f.write(json.dumps(r,sort_keys=True)+"\n")
    print(json.dumps({"n":len(rows),"permutations":len({r["permutation_id"] for r in rows}),
                      "signatures":summarize(rows)},indent=2,sort_keys=True))
