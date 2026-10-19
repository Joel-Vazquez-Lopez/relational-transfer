#!/usr/bin/env python3
"""BX-009: frozen-output, read-only wrong-event provenance audit.

No inference, no text access/download, no actor/patient recast, no repair of gold.
Input JSONL supplied only by an authorized owner. Output contains aggregate counts,
input hashes and explicit gate statuses, never raw sentences, entities or responses.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

ARMS = ('bare', 'bystander')
ROLES = ('agent', 'patient')
CATEGORIES = ('correct', 'other_event_same_role', 'own_event_other_role',
              'other_event_other_role', 'bystander', 'ambiguous_candidate',
              'out_of_set', 'empty')


def normalize(s):
    if not isinstance(s, str):
        raise ValueError('Answer/name must be UTF-8 string')
    return ' '.join(s.strip().casefold().split())


def _read(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def classify(record, answer):
    """String-source attribution, not proof of latent retrieval or semantic identity."""
    a = normalize(answer)
    if not a:
        return 'empty'
    target = record['target_event']
    role = record['target_role']
    other = 'e1' if target == 'e2' else 'e2'
    opposite = 'patient' if role == 'agent' else 'agent'
    candidates = [(record['roles'][target][role], 'correct'),
                  (record['roles'][other][role], 'other_event_same_role'),
                  (record['roles'][target][opposite], 'own_event_other_role'),
                  (record['roles'][other][opposite], 'other_event_other_role')]
    candidates += [(name, 'bystander') for name in record['bystanders']]
    matches = {kind for name, kind in candidates if normalize(name) == a}
    if len(matches) > 1:
        return 'ambiguous_candidate'
    return next(iter(matches)) if matches else 'out_of_set'


def _strict_manifest(row):
    expected = {'case_id', 'document_id', 'source_kind', 'source_review', 'rights_review',
                'target_event', 'target_role', 'target_order', 'roles', 'bystanders',
                'question_sha256', 'prompt_sha256', 'source_prompt_invariance_review'}
    if set(row) != expected:
        raise ValueError(f'manifest schema mismatch {sorted(expected - set(row))} {sorted(set(row) - expected)}')
    if row['target_event'] not in ('e1','e2') or row['target_role'] not in ROLES or row['target_order'] not in (1,2):
        raise ValueError('invalid target event, role or order')
    if set(row['roles']) != {'e1','e2'} or any(set(row['roles'][k]) != set(ROLES) for k in ('e1','e2')):
        raise ValueError('need original adjudicated event×role square, not synthetic labels inferred from QA')
    if not all(normalize(name) for x in row['roles'].values() for name in x.values()):
        raise ValueError('blank gold argument name')
    if not isinstance(row['bystanders'], list) or not row['bystanders'] or not all(normalize(x) for x in row['bystanders']):
        raise ValueError('no original bystander list')
    if not isinstance(row['source_kind'],str) or row['source_kind'] not in ('human_original','human_rewritten','synthetic_constructed','unverified'):
        raise ValueError('explicit source kind required')
    if row['source_review'] not in ('independent_human_adjudicated', 'owner_only', 'pending'):
        raise ValueError('explicit original gold/referent review status required')
    if row['rights_review'] not in ('approved_for_this_analysis', 'pending', 'not_approved'):
        raise ValueError('source rights review status required')
    if row['source_prompt_invariance_review'] not in ('independently_verified','owner_asserted','pending'):
        raise ValueError('prompt invariance status required')
    if set(row['prompt_sha256']) != set(ARMS):
        raise ValueError('two arms required')
    for v in [row['question_sha256'], *row['prompt_sha256'].values()]:
        if not isinstance(v,str) or len(v)!=64 or any(c not in '0123456789abcdef' for c in v):
            raise ValueError('missing/malformed fixed SHA-256')
    if not row['case_id'] or not row['document_id'] or not isinstance(row['case_id'],str) or not isinstance(row['document_id'],str):
        raise ValueError('nonempty case and document identifiers required')


def analyze(manifest, predictions, *, pinned_revision, owner_reported_threshold=8):
    if owner_reported_threshold < 1:
        raise ValueError('threshold must be positive')
    M = {}
    for r in manifest:
        _strict_manifest(r)
        if r['case_id'] in M:
            raise ValueError('duplicate manifest case ID')
        M[r['case_id']] = r
    if not M:
        raise ValueError('no cases')
    P = {}
    for p in predictions:
        if set(p) != {'case_id','arm','model_revision','response','prompt_sha256','question_sha256'}:
            raise ValueError('prediction schema includes unapproved fields or is incomplete')
        k = (p['case_id'],p['arm'])
        if p['case_id'] not in M or p['arm'] not in ARMS or k in P:
            raise ValueError('missing manifest key, unknown arm or duplicate prediction')
        if p['model_revision'] != pinned_revision:
            raise ValueError('mixed/unpinned model revisions')
        m = M[p['case_id']]
        if p['prompt_sha256'] != m['prompt_sha256'][p['arm']] or p['question_sha256'] != m['question_sha256']:
            raise ValueError('prompt/question hash mismatch; paired question may differ')
        if not isinstance(p['response'], str):
            raise ValueError('missing raw text response')
        P[k] = p
    if set(P) != {(key, arm) for key in M for arm in ARMS}:
        raise ValueError('missing predictions for a matched pair')
    totals = {arm: Counter() for arm in ARMS}
    matrices = Counter(); by_target_order = defaultdict(Counter)
    by_role = defaultdict(Counter); by_document = defaultdict(Counter)
    any_collision = 0; nonpaper = 0
    for case_id, m in M.items():
        labels = [normalize(x) for event in m['roles'].values() for x in event.values()] + [normalize(x) for x in m['bystanders']]
        any_collision += int(len(set(labels)) < len(labels))
        ready = (m['source_kind']=='human_original' and m['source_review']=='independent_human_adjudicated'
                 and m['rights_review']=='approved_for_this_analysis' and m['source_prompt_invariance_review']=='independently_verified')
        nonpaper += int(not ready)
        a,b = [classify(m,P[case_id,arm]['response']) for arm in ARMS]
        totals['bare'][a] += 1; totals['bystander'][b] += 1
        matrices[f'{a} -> {b}'] += 1
        by_target_order[str(m['target_order'])][b] += 1
        by_role[m['target_role']][b] += 1
        by_document[m['document_id']][b] += 1
    harm = sum(n for key,n in matrices.items() if key.startswith('correct -> ') and key != 'correct -> correct')
    rescue = sum(n for key,n in matrices.items() if key.endswith(' -> correct') and key != 'correct -> correct')
    other_event = totals['bystander']['other_event_same_role']+totals['bystander']['other_event_other_role']
    return {
        'status':'AUTHOR_OUTPUT_REANALYSIS_DESCRIPTIVE_ONLY' if nonpaper else 'METADATA_GATES_DECLARED_PASS_INDEPENDENT_SOURCE_REVIEW_STILL_REQUIRED',
        'experiment':'BX-009', 'model_revision':pinned_revision,
        'n_cases':len(M),'n_document_groups':len(by_document),
        'source_rights_gold_or_prompt_review_not_independently_certified_cases':nonpaper,
        'candidate_surface_collision_cases':any_collision,
        'arms':{arm:dict(sorted((k,int(totals[arm][k])) for k in CATEGORIES)) for arm in ARMS},
        'paired_transition':dict(sorted(matrices.items())),
        'rescue':rescue,'harm':harm,'net_correct_gain':rescue-harm,
        'bystander_condition_wrong_event_outputs':other_event,
        'bystander_condition_wrong_event_same_role':totals['bystander']['other_event_same_role'],
        'bystander_condition_wrong_event_other_role':totals['bystander']['other_event_other_role'],
        'owner_reported_threshold':owner_reported_threshold,
        'meets_owner_reported_wrong_event_threshold':other_event >= owner_reported_threshold,
        'by_target_order':{k:dict(sorted(v.items())) for k,v in sorted(by_target_order.items())},
        'by_target_role':{k:dict(sorted(v.items())) for k,v in sorted(by_role.items())},
        'document_level_error_summary':dict(Counter(sum(v[k] for k in CATEGORIES if k != 'correct') for v in by_document.values())),
        'warnings':[
            'A string matching a participant is not proof of that retrieval circuit.',
            'Repeated names/cross-event co-reference require human review; this parser only flags identical surface forms.',
            'Document IDs/source rights/referents/prompt invariance are self-declared; hashes do not authenticate originals.',
            'Owner predeclared threshold must not be changed based on this audit.',
            'Novelty/replication claims require frozen, previously unseen human-source data and strong copied-retrieval rivals.'
        ]}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',required=True);p.add_argument('--predictions',required=True)
    p.add_argument('--model-revision',required=True);p.add_argument('--out');p.add_argument('--threshold',type=int,default=8)
    a=p.parse_args()
    result=analyze(_read(a.manifest),_read(a.predictions),pinned_revision=a.model_revision,owner_reported_threshold=a.threshold)
    result['manifest_file_sha256']=sha(a.manifest)
    result['predictions_file_sha256']=sha(a.predictions)
    txt=json.dumps(result,indent=2,sort_keys=True)+'\n'
    if a.out:Path(a.out).write_text(txt,encoding='utf-8')
    print(txt)

if __name__=='__main__':main()
