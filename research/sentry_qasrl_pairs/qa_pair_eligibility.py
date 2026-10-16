#!/usr/bin/env python3
"""Development-only QA-SRL Gold QA-pair eligibility audit.

Measures whether one original human QA predicate has both a narrowly defined
subject-slot question and an object-slot question with different singleton
answer spans. This does NOT infer semantic actor/patient, document identity,
answer utility, or a natural-directional-model result. No text is exported.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re

UPSTREAM_COMMIT = 'f7c64ae9b6fe48ff3910c3e59850a12ec278bf83'
UPSTREAM_GIT_BLOBS = {
    'wikinews': '2171f970fc6bc34b6847a0274f1a62e388b8a4df',
    'wikipedia': '8ccc9450490a4ce3fac7be86ea53f99493e1f75d',
}
REQUIRED = ('qasrl_id', 'verb_idx', 'question', 'answer_range', 'wh',
            'subj', 'obj', 'obj2', 'prep', 'is_passive', 'is_negated')
RANGE = re.compile(r'^(\d+):(\d+)$')


def role_slot(row):
    """Strict syntactic QA template GAP, not human semantic role label."""
    if row['wh'] not in ('who', 'what'):
        return None, 'non_person_or_thing_wh'
    if row['is_passive'] != 'False':
        return None, 'passive_or_unknown_voice'
    if row['is_negated'] != 'False':
        return None, 'negated_or_unknown_polarity'
    if row['prep'] or row['obj2']:
        return None, 'preposition_or_second_object'
    if '~!~' in row['answer_range']:
        return None, 'multiple_answer_spans'
    m = RANGE.fullmatch(row['answer_range'])
    if not m or int(m.group(1)) >= int(m.group(2)):
        return None, 'invalid_or_empty_span'
    s, o = row['subj'], row['obj']
    if not s and o in ('someone', 'something'):
        return 'subject_slot_question', None
    if s in ('someone', 'something') and not o:
        return 'object_slot_question', None
    return None, 'no_unique_structural_gap'


def spans_overlap(a, b):
    a0, a1 = [int(v) for v in a.split(':')]
    b0, b1 = [int(v) for v in b.split(':')]
    return not (a1 <= b0 or b1 <= a0)


def audit_rows(rows, source):
    groups = defaultdict(lambda: {'subject_slot_question': [], 'object_slot_question': []})
    skips, slots = Counter(), Counter()
    predicates = set()
    num = 0
    for r in rows:
        num += 1
        if set(REQUIRED) - set(r):
            raise ValueError('Missing original QA-SRL Gold schema columns')
        key = (source, r['qasrl_id'], r['verb_idx'])
        predicates.add(key)
        slot, reason = role_slot(r)
        if not slot:
            skips[reason] += 1
            continue
        slots[slot] += 1
        groups[key][slot].append(r)
    paired, qualified, ambiguous, overlap = 0, 0, 0, 0
    question_rows, same_workers, other_workers = 0, 0, 0
    subject_before_object, object_before_subject = 0, 0
    all_sentences = set()
    wh = Counter()
    for key, g in groups.items():
        subj, obj = g['subject_slot_question'], g['object_slot_question']
        if not subj or not obj:
            continue
        paired += 1
        su = {x['answer_range'] for x in subj}
        ob = {x['answer_range'] for x in obj}
        # Do not arbitrate independently answered or contradictory spans.
        if len(su) != 1 or len(ob) != 1 or len(subj) != 1 or len(obj) != 1:
            ambiguous += 1
            continue
        if spans_overlap(next(iter(su)), next(iter(ob))):
            overlap += 1
            continue
        qualified += 1
        question_rows += len(subj) + len(obj)
        if int(next(iter(su)).split(':')[0]) < int(next(iter(ob)).split(':')[0]):
            subject_before_object += 1
        else:
            object_before_subject += 1
        all_sentences.add(key[1])
        who = subj[0]['wh']; what = obj[0]['wh']
        wh['who_who' if who == what == 'who' else
           'what_what' if who == what == 'what' else 'mixed'] += 1
        if subj[0].get('worker_id') == obj[0].get('worker_id'):
            same_workers += 1
        else:
            other_workers += 1
    return dict(source=source, gold_rows=num, source_predicate_keys=len(predicates),
                retained_slot_question_rows=dict(slots),
                predicates_with_both_gaps=paired,
                strict_unambiguous_pairs=qualified,
                distinct_sentence_ids_with_pairs=len(all_sentences),
                paired_human_question_rows=question_rows,
                subject_gold_span_starts_before_object=subject_before_object,
                object_gold_span_starts_before_subject=object_before_subject,
                rejected_pairs_multiple_ambiguous_answers=ambiguous,
                rejected_pairs_overlapping_spans=overlap,
                pair_wh_types=dict(wh),
                same_worker_pairs=same_workers,
                different_worker_pairs=other_workers,
                filtered_row_reasons=dict(skips))


def git_blob_sha(data):
    return hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()


def load_dev(path, source, check_upstream=True):
    path = Path(path)
    if path.name != f'{source}.dev.gold.csv':
        raise ValueError('Only exact official DEV gold basenames supported; no test data')
    raw = path.read_bytes()
    if check_upstream and git_blob_sha(raw) != UPSTREAM_GIT_BLOBS[source]:
        raise ValueError(f'Source content Git blob mismatch: {source}')
    # newline='' preserves Python csv module quoting semantics
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        rd = csv.DictReader(f)
        if not rd.fieldnames or set(REQUIRED) - set(rd.fieldnames):
            raise ValueError('QA-SRL original schema mismatch')
        result = audit_rows(rd, source)
    result['git_blob_sha1'] = git_blob_sha(raw)
    result['sha256_local_file'] = hashlib.sha256(raw).hexdigest()
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--wikinews-dev-gold', type=Path, required=True)
    ap.add_argument('--wikipedia-dev-gold', type=Path, required=True)
    ap.add_argument('--summary-json', type=Path)
    args = ap.parse_args()
    data = [load_dev(args.wikinews_dev_gold, 'wikinews'),
            load_dev(args.wikipedia_dev_gold, 'wikipedia')]
    result = {'upstream_commit': UPSTREAM_COMMIT,
              'source_files': data,
              'total_qualifying_predicate_pairs': sum(x['strict_unambiguous_pairs'] for x in data),
              'total_sentence_ids_with_qualifying_pairs': sum(x['distinct_sentence_ids_with_pairs'] for x in data),
              'interpretation': 'Syntactic QA question-slot pair availability, NOT actor/patient gold or a directionality-model score.',
              'dataset_scope': 'Original human-written QA-SRL Gold DEV only; no text, spans, or questions exported.'}
    text = json.dumps(result, indent=2, sort_keys=True)+'\n'
    print(text, end='')
    if args.summary_json:
        if args.summary_json.exists():
            raise SystemExit('Refuse overwrite of existing summary')
        args.summary_json.write_text(text, encoding='utf-8')

if __name__ == '__main__':
    main()
