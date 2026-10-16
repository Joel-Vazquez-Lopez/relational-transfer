"""Read-only QA-SRL Gold development shortcut diagnostics; no source text emitted."""
import argparse,csv,json
from collections import Counter,defaultdict

def load(path):
    with open(path,encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))

def audit(gold,sentences):
    lookup={r["qasrl_id"]:r for r in sentences}
    if len(lookup)!=len(sentences):raise ValueError("duplicate sentence IDs")
    c=Counter(); by_sentence=defaultdict(list)
    for row in gold:
        sid=row.get("qasrl_id")
        if sid not in lookup:
            c["missing_join"]+=1;continue
        tokens=lookup[sid]["tokens"].split()
        try:
            verb=int(row["verb_idx"])
            assert 0<=verb<len(tokens)
        except (ValueError,KeyError,AssertionError):
            c["invalid_predicate"]+=1;continue
        ranges=row.get("answer_range","").split("~!~")
        if len(ranges)!=1:
            c["multiple_spans"]+=1;continue
        try:
            a,b=map(int,ranges[0].split(":"))
            assert 0<=a<b<=len(tokens)
        except (ValueError,AssertionError):
            c["invalid_span"]+=1;continue
        if " ".join(tokens[a:b])!=row.get("answer",""):
            c["surface_disagreement"]+=1;continue
        c["eligible"]+=1
        by_sentence[sid].append(verb)
        c["first_token"]+=int(a==0 and b==1)
        c["last_token"]+=int(a==len(tokens)-1 and b==len(tokens))
        c["immediately_before_verb"]+=int(a==verb-1 and b==verb)
        c["immediately_after_verb"]+=int(a==verb+1 and b==verb+2)
        width=b-a
        c["same_answer_surface_multiple_positions"]+=int(sum(
            tokens[i:i+width]==tokens[a:b] for i in range(len(tokens)-width+1))>1)
    c["sentence_clusters"]=len(by_sentence)
    c["sentences_with_multiple_predicates"]=sum(len(set(v))>1 for v in by_sentence.values())
    return {"status":"development_only_no_model_results","counts":dict(c),
            "caveats":["QA-SRL questions are not universal actor/patient labels",
            "position heuristics are diagnostic, not competitive parsers",
            "same-surface matches do not prove correct occurrence selection",
            "sentence IDs are not proven document-independent",
            "multi-span and inconsistent references are excluded and counted"]}

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--gold",required=True);p.add_argument("--sentences",required=True)
    p.add_argument("--partition",default="dev")
    a=p.parse_args()
    if "test" in a.partition.lower():raise ValueError("test partition not allowed")
    print(json.dumps(audit(load(a.gold),load(a.sentences)),indent=2))
