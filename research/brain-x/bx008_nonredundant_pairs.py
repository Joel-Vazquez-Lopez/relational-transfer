"""BX-008: read-only comparison against Sentry structural QA pairs and balanced panel.

Requires the pinned original QA-SRL Gold DEV files; does not export text, answers,
workers, sentence IDs, or gold. No model inference or new semantic role labels.
"""
from collections import Counter, defaultdict
import argparse
import hashlib
import json
import re

from bx007_qasrl_contrast import eligible_pairs, read, norm


def structural_sentry_signatures(rows):
    """Sentry PR29's original syntax-slot screen, not semantic-role gold.

    Intentionally does not add token-boundary criteria that Sentry did not use.
    """
    groups = defaultdict(lambda: {'subject': [], 'object': []})
    for r in rows:
        if r.get('wh') not in ('who', 'what') or r.get('is_passive') != 'False' or r.get('is_negated') != 'False':
            continue
        if r.get('prep') or r.get('obj2') or '~!~' in r.get('answer_range', ''):
            continue
        rg = r.get('answer_range', '')
        if not re.fullmatch(r'\d+:\d+', rg):
            continue
        try:
            a, b = [int(x) for x in rg.split(':')]
        except ValueError:
            continue
        if a < 0 or a >= b:
            continue
        slot = None
        if not r.get('subj') and r.get('obj') in ('someone', 'something'):
            slot = 'subject'
        elif r.get('subj') in ('someone', 'something') and not r.get('obj'):
            slot = 'object'
        if slot:
            groups[(r['qasrl_id'], r['verb_idx'])][slot].append(r)
    result = set()
    for (sid, verb), slots in groups.items():
        aa = slots['subject']; bb = slots['object']
        if len(aa) != 1 or len(bb) != 1:
            continue
        a, b = map(int, aa[0]['answer_range'].split(':'))
        c, d = map(int, bb[0]['answer_range'].split(':'))
        if not (b <= c or d <= a):
            continue
        questions = tuple(sorted((norm(r['question']), r['answer_range']) for r in (aa[0], bb[0])))
        result.add((sid, str(verb), questions))
    return result


def signature(p):
    return (p['sid'], str(p['vi']), tuple(sorted((norm(q), f'{a}:{b}')
        for q, (a,b) in zip(p['questions'], p['ranges']))))


def gold_order(p):
    """Privileged GOLD SPAN position; never part of model-visible prompts."""
    if p['ranges'][0][0] == p['ranges'][1][0]:
        raise ValueError('Two disjoint spans cannot start at one position')
    return 'early' if p['ranges'][0][0] < p['ranges'][1][0] else 'late'


def rank(p):
    return hashlib.sha256((p['source'] + '|' + p['sid'] + '|' + p['qid']).encode()).hexdigest()


def match_one_wh(group, per_side=10, reserved=None):
    """Maximal bipartite matching: 10 early+10 late unique source-sentence IDs.

    A sentence may support several candidate pairs; it cannot fill two slots.
    reserved is the cross-WH sentence-use set for this source.
    """
    if per_side < 1:
        raise ValueError('Need positive quota')
    reserved = reserved or set()
    by_side = {x: sorted((p for p in group if gold_order(p) == x and p['sid'] not in reserved),
                         key=lambda p: (rank(p), p['qid'])) for x in ('early','late')}
    # Process the less numerous class first, but use augmenting paths to resolve ties.
    order = sorted(('early', 'late'), key=lambda d: len({p['sid'] for p in by_side[d]}))
    slots = [d for d in order for _ in range(per_side)]
    assigned = {}  # sid -> (slot-index, selected candidate)
    def augment(slot, seen):
        side = slots[slot]
        for candidate in by_side[side]:
            sid = candidate['sid']
            if sid in seen: continue
            seen.add(sid)
            if sid not in assigned or augment(assigned[sid][0], seen):
                assigned[sid] = (slot, candidate)
                return True
        return False
    for i in range(len(slots)):
        augment(i, set())
    selected = [p for _slot, p in assigned.values()]
    counts = Counter(gold_order(p) for p in selected)
    if counts['early'] != per_side or counts['late'] != per_side:
        return [], {'status':'BLOCKED_INSUFFICIENT_POSITION_BALANCE',
                    'early':counts['early'],'late':counts['late'], 'required_per_side':per_side}
    if len({p['sid'] for p in selected}) != len(selected):
        raise AssertionError('Bipartite matching unexpectedly reused a sentence')
    return (sorted(selected, key=lambda p:(p['source'],p['wh'],rank(p))),
            {'status':'READY_DEVELOPMENT_SELECTION','early':counts['early'],'late':counts['late']})


def audit(sources, per_side=10):
    """Aggregate only. This script is NOT a model scorer or a rights verifier."""
    if not sources or any(not s.endswith('.dev') for s in sources):
        raise ValueError('Only explicitly labelled .dev source files')
    output = {}
    selected = []
    for source, (gold, sentences) in sorted(sources.items()):
        pairs, _ = eligible_pairs(gold, sentences)
        for p in pairs: p['source'] = source
        other = structural_sentry_signatures(gold)
        overlap = [p for p in pairs if signature(p) in other]
        order = Counter((p['wh'], gold_order(p)) for p in pairs)
        used = set()
        quotas = {}
        for wh in ('who','what'):
            pp, info = match_one_wh([p for p in pairs if p['wh'] == wh], per_side, used)
            quotas[wh] = info
            if info['status'] != 'READY_DEVELOPMENT_SELECTION':
                return {'status':'BLOCKED', 'reason':f'{source}/{wh}', 'availability':quotas}
            used.update(x['sid'] for x in pp)
            selected += pp
        output[source] = {
            'bx_pairs':len(pairs), 'sentry_structural_pairs':len(other),
            'exact_pair_overlap':len(overlap),
            'bx_only':len(pairs)-len(overlap),
            'sentry_only':len(other)-len(overlap),
            'lex_question_first_gold_position':{
                f'{wh}_{side}':order[wh,side] for wh in ('who','what') for side in ('early','late')},
            'balanced_selection_counts':quotas,
            'selected_unique_sentence_ids':len(used),
            'same_worker_pairs':sum(p['same_worker'] for p in pairs),
        }
    selected_ids = sorted(p['qid'] for p in selected)
    return {'status':'DEVELOPMENT_METADATA_ONLY_NOT_APPROVED_MODEL_RUN',
            'sources':output,'all_bx_pairs':sum(z['bx_pairs'] for z in output.values()),
            'total_exact_pair_overlap':sum(z['exact_pair_overlap'] for z in output.values()),
            'selected_balanced_pairs':len(selected),
            'selected_pair_ids_sha256':hashlib.sha256('\n'.join(selected_ids).encode()).hexdigest(),
            'source_gold_order_is_privileged':True,
            'no_model_results':True,
            'warning':'Human QA paired within same workers/assignments; no independent human role gold. Source sentence IDs not proven article/document IDs. Unknown test exposure must be reviewed. Selector needs genuine upstream data cold replay and rights approval before inference.'}


def main():
    p = argparse.ArgumentParser()
    for x in ('wikinews','wikipedia'):
        p.add_argument(f'--{x}-gold', required=True)
        p.add_argument(f'--{x}-sentences', required=True)
    p.add_argument('--per-side',type=int,default=10)
    p.add_argument('--output')
    a=p.parse_args()
    sources={f'{n}.dev':(read(getattr(a,n+'_gold')),read(getattr(a,n+'_sentences'))) for n in ('wikinews','wikipedia')}
    result=audit(sources,per_side=a.per_side)
    text=json.dumps(result,indent=2,sort_keys=True)+'\n'
    if a.output:
        with open(a.output,'w',encoding='utf8') as f:f.write(text)
    print(text)
    if result['status']=='BLOCKED':raise SystemExit(2)
if __name__=='__main__':main()
