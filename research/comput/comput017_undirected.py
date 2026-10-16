#!/usr/bin/env python3
"""Independent label-free co-occurrence graph baseline for Mira V5.

Predictor reads passage+question ONLY. It does not parse subject/object, passive voice,
actor/patient, gold, stage indexing, source metadata or generator bits.
"""
import collections, json, pathlib, re, argparse
NAMES = re.compile(r'\b[A-Z][a-z]+\b')
WORDS = re.compile(r'\b[a-z]+\b')
AND_Q = re.compile(r'^Who ([a-z]+) the person who both ([a-z]+) ([A-Z][a-z]+) and ([a-z]+) ([A-Z][a-z]+)\?$')

def undirected_graph(passage):
    graph=collections.defaultdict(lambda:collections.defaultdict(set))
    parts=re.split(r'\.\s*|;\s*',passage.strip().rstrip('.'))
    for part in parts:
        part=part.removeprefix('Then, ').strip()
        if not part:continue
        names=NAMES.findall(part)
        pred=[w for w in WORDS.findall(part) if w not in {'was','by'}]
        if len(names)!=2 or len(pred)!=1:
            raise ValueError(('not two names/one predicate',part,names,pred))
        a,b=names;v=pred[0]
        graph[v][a].add(b)
        graph[v][b].add(a)
    return graph

def predict(passage,question):
    graph=undirected_graph(passage)
    m=AND_Q.fullmatch(question)
    if m:
        target,v1,a1,v2,a2=m.groups()
        mids=graph[v1][a1] & graph[v2][a2]
        if len(mids)!=1:return None
        candidates=graph[target][next(iter(mids))]
        return next(iter(candidates)) if len(candidates)==1 else None
    if not(question.startswith('Who ') and question.endswith('?')):return None
    text=question[4:-1]
    parts=text.split(' the person who ')
    if len(parts)<2:return None
    last=parts[-1].split()
    if len(last)!=2:return None
    anchor=last[-1]
    verbs=list(reversed(parts[:-1]+[last[0]]))
    current={anchor}
    for v in verbs:
        possible=set()
        for name in current:possible.update(graph[v][name])
        if len(possible)!=1:return None
        current=possible
    return next(iter(current))

def audit(inputs, gold):
    source=[json.loads(l) for l in pathlib.Path(inputs).read_text().splitlines() if l]
    targets={r['id']:r['gold'] for r in (json.loads(l) for l in pathlib.Path(gold).read_text().splitlines() if l)}
    assert len(source)==len(targets)==26496
    total=correct=abstain=0
    errors=[]
    for r in source:
        guess=predict(r['passage'],r['question']);label=targets[r['id']]
        total+=1;correct+=int(guess==label);abstain+=int(guess is None)
        if guess!=label and len(errors)<15:errors.append({'id':r['id'],'guess':guess,'reference':label})
    return {'n':total,'correct':correct,'abstain':abstain,'errors':errors,
            'model':'undirected name-verb cooccurrence, no syntactic direction'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('model_inputs');p.add_argument('private_gold');p.add_argument('--output');a=p.parse_args()
    out=audit(a.model_inputs,a.private_gold)
    txt=json.dumps(out,indent=2)+'\n'
    if a.output:pathlib.Path(a.output).write_text(txt)
    else:print(txt)
