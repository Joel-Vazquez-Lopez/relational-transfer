"""BX-007: open-development QA-SRL question-contrast audit, no model calls.

Upstream QA-SRL Gold: plroit/qasrl-gs at
f7c64ae9b6fe48ff3910c3e59850a12ec278bf83.
Read only official .dev CSVs. No actor/patient recast, no source text export.
"""
import argparse, csv, hashlib, json, re
from collections import Counter, defaultdict

WH = ('who', 'what')
HALFOPEN = re.compile(r'^(\d+):(\d+)$')


def read(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def norm(s):
    return ' '.join(s.casefold().split())


def _span(row, tokens):
    if '~!~' in row.get('answer_range', '') or '~!~' in row.get('answer', ''):
        return None, 'multi_span'
    m = HALFOPEN.fullmatch(row.get('answer_range', ''))
    if not m:
        return None, 'bad_range'
    a,b=map(int,m.groups())
    if not (0 <= a < b <= len(tokens)):
        return None, 'range_bounds'
    if ' '.join(tokens[a:b]) != row.get('answer', ''):
        return None, 'source_answer_mismatch'
    # Ambiguous occurrences make string-only scoring unable to locate the gold span.
    matches = sum(tokens[k:k+(b-a)] == tokens[a:b]
                  for k in range(len(tokens)-(b-a)+1))
    if matches != 1:
        return None, 'repeated_surface'
    return (a,b,row['answer']), None


def eligible_pairs(gold, sentences):
    """Return source-bound development-only question contrasts and exclusion counts.

    Text remains in memory, never included in the emitted summary.
    """
    lookup = {}
    for r in sentences:
        sid=r.get('qasrl_id')
        if not sid or sid in lookup:
            raise ValueError('missing/duplicate source sentence id')
        lookup[sid] = (r['tokens'].split(), r)
    grouped=defaultdict(list)
    exclusions=Counter()
    for r in gold:
        sid=r.get('qasrl_id'); wh=norm(r.get('wh', ''))
        if wh not in WH:
            exclusions['other_wh']+=1;continue
        if sid not in lookup:
            exclusions['missing_sentence']+=1;continue
        tokens=lookup[sid][0]
        try:
            vi=int(r['verb_idx'])
        except (KeyError,ValueError):
            exclusions['bad_verb']+=1;continue
        if not 0<=vi<len(tokens):
            exclusions['bad_verb']+=1;continue
        span,err=_span(r,tokens)
        if err:
            exclusions[err]+=1;continue
        question=r.get('question','').strip()
        if not question:
            exclusions['empty_question']+=1;continue
        grouped[(sid,vi)].append((wh,question,span,r.get('worker_id',''),r.get('assign_id','')))
    eligible=[]
    for (sid,vi),items in grouped.items():
        q_counts=Counter(norm(row[1]) for row in items)
        filtered=[x for x in items if q_counts[norm(x[1])]==1]
        exclusions['duplicate_question_rows_excluded']+=len(items)-len(filtered)
        for i in range(len(filtered)):
            for j in range(i+1,len(filtered)):
                wh1,q1,(a,b,ans1),worker1,assign1=filtered[i]
                wh2,q2,(c,d,ans2),worker2,assign2=filtered[j]
                if wh1 != wh2 or norm(q1)==norm(q2):
                    continue
                if not (b<=c or d<=a):
                    exclusions['overlap_pair']+=1;continue
                if norm(ans1)==norm(ans2):
                    exclusions['equal_answer_surface_pair']+=1;continue
                # Canonical ordering is question-normalized lexicographic only.
                z=sorted([(q1,a,b,ans1),(q2,c,d,ans2)],key=lambda x:(norm(x[0]),x[1],x[2]))
                key=json.dumps([sid,vi,wh1,[(x[0],x[1],x[2]) for x in z]],ensure_ascii=False)
                eligible.append({'sid':sid,'vi':vi,'wh':wh1,'qid':hashlib.sha256(key.encode()).hexdigest(),
                                 'questions':(z[0][0],z[1][0]),'ranges':((z[0][1],z[0][2]),(z[1][1],z[1][2])),
                                 'answers':(z[0][3],z[1][3]),
                                 'same_worker':bool(worker1 and worker2 and worker1==worker2),
                                 'same_assignment':bool(assign1 and assign2 and assign1==assign2)})
    return eligible, dict(exclusions)


def select_panel(pairs, quota_per_source_wh=20):
    """One pair per sentence; separate 20 who/20 what per source if available.

    Selection is deterministic SHA-256 rank of sentence + pair fingerprint, not
    stratified by document (source sentence IDs are not documented article IDs).
    """
    candidates=defaultdict(list)
    for p in pairs:
        key=(p['source'],p['wh'])
        rank=hashlib.sha256((p['source']+'|'+p['sid']+'|'+p['qid']).encode()).hexdigest()
        candidates[key].append((rank,p))
    output=[];used=set();counts={}
    for key in sorted(candidates):
        selected=[]
        for _,p in sorted(candidates[key]):
            if (p['source'],p['sid']) in used:
                continue
            selected.append(p)
            used.add((p['source'],p['sid']))
            if len(selected)>=quota_per_source_wh:break
        counts['/'.join(key)]=len(selected)
        output+=selected
    return output,counts


def audit_sources(sources, *, quota=20):
    """sources mapping {source-name: (gold-rows,sentence-rows)}; labels never exported."""
    if not sources or any('.test' in source.lower() for source in sources):
        raise ValueError('explicit dev sources only; no test files')
    allp=[];per={}
    for source,(gold,sentences) in sorted(sources.items()):
        if not source.endswith('.dev'):
            raise ValueError('dev partition must be explicit')
        pairs,ex=eligible_pairs(gold,sentences)
        for p in pairs:p['source']=source
        allp+=pairs
        per[source]={'gold_rows':len(gold),'source_sentence_rows':len(sentences),
                     'eligible_pairs':len(pairs),
                     'same_worker_pairs':sum(p['same_worker'] for p in pairs),
                     'same_assignment_pairs':sum(p['same_assignment'] for p in pairs),
                     'distinct_source_sentences':len(set(p['sid'] for p in pairs)),
                     'pairs_by_wh':dict(Counter(p['wh'] for p in pairs)), 'excluded':ex}
    selected,quotas=select_panel(allp,quota)
    # No questions, source sentences or answer strings leave this audit.
    return {'status':'OPEN_DEVELOPMENT_NO_MODEL_RESULTS','sources':per,
            'eligible_total_pairs':len(allp),
            'eligible_distinct_source_sentences':len(set((p['source'],p['sid']) for p in allp)),
            'selected_groups':len(selected),'selected_by_source_wh':quotas,
            'sha256_selected_private_pair_ids':hashlib.sha256('\n'.join(sorted(x['qid'] for x in selected)).encode()).hexdigest(),
            'pair_prompt_contract':'Identical original sentence and verb, two different original human questions, same wh type, two disjoint singleton exact source spans with different nonrepeated text.',
            'primary_future_metrics':'Both correct, exactly one correct, neither correct; exact same-answer collapse; invalid format separately.',
            'warning':'Data are open development; source sentence IDs are not verified document clusters; gold-answer range and qid remain scorer-only; no actual LLM inference or novelty claim.'}


def pair_scores(answer1,answer2,gold1,gold2):
    """Strict text diagnostic only, valid where singleton exact spans are verified.

    Not an official QA-SRL metric; source-occurrence ambiguity prefiltered.
    """
    x=norm(answer1);y=norm(answer2)
    c1=int(x==norm(gold1));c2=int(y==norm(gold2))
    return {'correct_count':c1+c2,'both_correct':bool(c1 and c2),
            'same_answer_collapse':bool(x and y and x==y),
            'both_nonempty':bool(x and y),'one_correct':(c1+c2)==1}


def main():
    p=argparse.ArgumentParser();p.add_argument('--wikinews-gold',required=True)
    p.add_argument('--wikinews-sentences',required=True);p.add_argument('--wikipedia-gold',required=True)
    p.add_argument('--wikipedia-sentences',required=True);p.add_argument('--quota',type=int,default=20)
    p.add_argument('--output');a=p.parse_args()
    if a.quota<1:raise ValueError('quota must be positive')
    sources={'wikinews.dev':(read(a.wikinews_gold),read(a.wikinews_sentences)),
             'wikipedia.dev':(read(a.wikipedia_gold),read(a.wikipedia_sentences))}
    result=audit_sources(sources,quota=a.quota)
    text=json.dumps(result,indent=2,ensure_ascii=False)+'\n'
    if a.output:
        with open(a.output,'w',encoding='utf8') as f:f.write(text)
    print(text)
if __name__=='__main__':main()
