#!/usr/bin/env python3
"""COMPUT-023: read-only development audit of SemEval-2010 Task 8 training data.

No test split, model inference, human comment, or source text is exported.
Gold-TYPE-conditioned direction is an ORACLE DIAGNOSTIC, not a usable raw-text model.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

RELATIONS = {"Cause-Effect", "Instrument-Agency", "Product-Producer", "Content-Container",
             "Entity-Origin", "Entity-Destination", "Component-Whole", "Member-Collection", "Message-Topic"}
REL_LABEL = re.compile(r'^([A-Za-z-]+)\((e1,e2|e2,e1)\)$')
SENTENCE = re.compile(r'^(\d+)\s+"([\s\S]*)"$')

def load_training(text):
    rows = []
    for chunk in re.split(r'\n\s*\n', text.replace('\r\n', '\n').replace('\r','\n')):
        if not chunk.strip():
            continue
        lines = chunk.strip().split('\n')
        m = SENTENCE.fullmatch(lines[0])
        if m is None or len(lines) < 2:
            raise ValueError('Invalid numbered input or missing gold label')
        rid, sentence = int(m.group(1)), m.group(2)
        label = lines[1].strip()
        if label != 'Other':
            lm = REL_LABEL.fullmatch(label)
            if lm is None or lm.group(1) not in RELATIONS:
                raise ValueError(f'Invalid relation label at {rid}')
        if any(sentence.count(tag) != 1 for tag in ('<e1>', '</e1>', '<e2>', '</e2>')):
            raise ValueError(f'Invalid entity markup at {rid}')
        if sentence.index('<e1>') >= sentence.index('<e2>'):
            raise ValueError(f'Entity mention order unexpected at {rid}')
        rows.append({'id':rid, 'sentence':sentence, 'label':label})
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate item IDs')
    return rows

def base_sentence(sentence):
    return ' '.join(re.sub(r'</?e[12]>', '', sentence).lower().split())

def audit(rows):
    labels=Counter(r['label'] for r in rows)
    by_type=defaultdict(Counter)
    by_base=defaultdict(list)
    by_markup=Counter()
    for row in rows:
        y=row['label'];by_markup[' '.join(row['sentence'].lower().split())]+=1
        by_base[base_sentence(row['sentence'])].append(row)
        if y != 'Other':
            m=REL_LABEL.fullmatch(y)
            by_type[m.group(1)][m.group(2)]+=1
    types={}
    for name in sorted(RELATIONS):
        v=by_type[name];a=v['e1,e2'];b=v['e2,e1'];n=a+b
        types[name]={'e1_to_e2':a,'e2_to_e1':b,'count':n,
                     'majority_direction_accuracy_if_type_gold':max(a,b)/n if n else None,
                     'maximum_balanced_subset_size':2*min(a,b)}
    eligible=sum(v['count'] for v in types.values())
    oracle=sum(max(v['e1_to_e2'],v['e2_to_e1']) for v in types.values())
    pairs=sum(v['maximum_balanced_subset_size'] for v in types.values())
    forward=sum(v['e1_to_e2'] for v in types.values())
    # Group the SAME untagged source sentence into one fold, not whole documents.
    group_fold={k:i%5 for i,k in enumerate(by_base)}
    folds=[];hit=0;alln=0
    for f in range(5):
        train=defaultdict(Counter);test=[]
        for row in rows:
            if row['label']=='Other':continue
            role=REL_LABEL.fullmatch(row['label'])
            if group_fold[base_sentence(row['sentence'])] == f:test.append((role.group(1),role.group(2)))
            else:train[role.group(1)][role.group(2)]+=1
        good=sum(int(direction == ('e1,e2' if train[t]['e1,e2'] >= train[t]['e2,e1'] else 'e2,e1'))
                 for t,direction in test)
        folds.append({'fold':f,'n':len(test),'correct':good})
        hit+=good;alln+=len(test)
    repeated=[v for v in by_base.values() if len(v)>1]
    return {
        'task':'SemEval-2010 Task 8, TRAINING ONLY, 19 label task',
        'n_rows':len(rows),'n_unique_ids':len({r['id'] for r in rows}),
        'n_undirected_relation_items':eligible,'n_other':labels['Other'],
        'by_relation_type':types,
        'global_forward_fraction_nonother':forward/eligible,
        'gold_type_oracle_majority_orientation':{'correct':oracle,'n':eligible,'accuracy':oracle/eligible,
          'privileged_type_WARNING':'Uses correct human relation type at scoring time; NOT standalone model baseline'},
        'gold_type_oracle_5fold_untagged_sentence_grouped':{'correct':hit,'n':alln,'accuracy':hit/alln,'folds':folds,
          'grouping_WARNING':'Only identical untagged source sentences grouped; NOT documents'},
        'max_type_direction_balanced_n':pairs,
        'identical_untagged_sentence_additional_rows':sum(len(v)-1 for v in repeated),
        'identical_untagged_sentence_groups':len(repeated),
        'identical_untagged_groups_with_multiple_gold_labels':sum(len({x['label'] for x in g})>1 for g in repeated),
        'identical_entity_marked_input_additional_rows':sum(n-1 for n in by_markup.values() if n>1),
        'critical_warning':'Distinct marked pairs in a sentence are not necessarily conflicting human labels. No inference.'
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument('training_txt',type=Path)
    p.add_argument('--out',type=Path)
    p.add_argument('--expect-original-8000',action='store_true')
    args=p.parse_args()
    if 'TEST' in args.training_txt.name.upper():p.error('Only TRAINING data allowed')
    rows=load_training(args.training_txt.read_text(encoding='utf-8-sig'))
    if args.expect_original_8000 and len(rows)!=8000:p.error('Expected original 8000 rows')
    result=audit(rows)
    body=json.dumps(result,sort_keys=True,indent=2)+'\n'
    if args.out:args.out.write_text(body)
    else:print(body)
if __name__=='__main__':main()
