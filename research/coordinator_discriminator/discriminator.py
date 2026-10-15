#!/usr/bin/env python3
"""Known-mechanism control calibration with independent cache/raw edits.

Deterministic oracles, not trained LMs or interventions on neural states.
"""
import json
from collections import Counter
from itertools import permutations, product

NAMES = ("Ada", "Bea", "Cal", "Dee")
VERBS = ("praised", "criticized")
ROLES = ("actor", "patient")

def worlds():
    for t in permutations(NAMES):
        yield {VERBS[0]: {"actor":t[0], "patient":t[1]},
               VERBS[1]: {"actor":t[2], "patient":t[3]}}

def ask(world, query):
    return world[query[0]][query[1]]

def edit(world, verb, role, answer):
    result = {v:dict(event) for v,event in world.items()}
    result[verb][role] = answer
    return result

def cached(cache, raw, query):
    return ask(cache, query)

def reconstructed(cache, raw, query):
    return ask(raw, query)

def shortcut(cache, raw, query):
    return NAMES[0]

def hybrid(cache, raw, query):
    return ask(cache if query[1] == "actor" else raw, query)

CONTROLS = {"cached":cached, "reconstructed":reconstructed,
            "shortcut":shortcut, "hybrid":hybrid}

def experiment():
    rows=[]
    for world_id,world in enumerate(worlds()):
        for verb,role in product(VERBS, ROLES):
            query=(verb,role)
            orig=ask(world,query)
            donors=[n for n in NAMES if n!=orig]
            for cache_donor in donors:
                for raw_donor in donors:
                    if cache_donor == raw_donor:
                        continue
                    changed_cache=edit(world,verb,role,cache_donor)
                    changed_raw=edit(world,verb,role,raw_donor)
                    for name,policy in CONTROLS.items():
                        base=policy(world,world,query)
                        c=policy(changed_cache,world,query)
                        r=policy(world,changed_raw,query)
                        crossed=policy(changed_cache,changed_raw,query)
                        # Require selective interventions as well as donor agreement.
                        # Raw donor coincidence alone generates false positives on
                        # a constant-name shortcut.
                        label=("cache" if c!=base and r==base and crossed==cache_donor
                               else "raw" if r!=base and c==base and crossed==raw_donor
                               else "other")
                        rows.append({"world_id":world_id,"control":name,
                            "query":[verb,role],"original":orig,"baseline":base,
                            "cache_donor":cache_donor,"raw_donor":raw_donor,
                            "cache_only":c,"raw_only":r,"crossed":crossed,
                            "label":label})
    return rows

def summarize(rows):
    out={}
    for name in CONTROLS:
        records=[row for row in rows if row["control"]==name]
        out[name]={"n":len(records),
           "unintervened_accuracy":sum(row["baseline"]==row["original"] for row in records)/len(records),
           "cache_effect":sum(row["baseline"]!=row["cache_only"] for row in records),
           "raw_effect":sum(row["baseline"]!=row["raw_only"] for row in records),
           "crossed_labels":dict(Counter(row["label"] for row in records))}
    return out

if __name__=="__main__":
    print(json.dumps(summarize(experiment()),indent=2,sort_keys=True))
