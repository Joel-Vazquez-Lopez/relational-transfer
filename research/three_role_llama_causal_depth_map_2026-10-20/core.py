"""Frozen full-name scoring and six-answer eligibility; no model initialization."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
LENGTH = 128
MAX_NEW = 12
WORKER_CAP = 32


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()


def save(path, obj):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


def append(path, obj):
    with Path(path).open('a') as f:
        f.write(json.dumps(obj, allow_nan=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


def lse(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.logaddexp.reduce(x))


def full_score(row, get, decode):
    paths, classes = {}, {}
    for name in row['names']:
        paths[name] = []
        for seq in sorted({tuple(x) for x in row['candidate_sequences'][name]}):
            terms = []
            for j, token in enumerate(seq):
                z = get(seq[:j])
                terms.append(float(z[token]) - lse(z))
            paths[name].append(dict(tokens=list(seq), log_probability=sum(terms), terms=terms))
        classes[name] = lse([x['log_probability'] for x in paths[name]])
    tokens, eos = [], None
    for _ in range(MAX_NEW):
        token = int(np.argmax(get(tuple(tokens))))
        if token in row['eos_ids']:
            eos = token
            break
        tokens.append(token)
    text = decode(tokens)
    actual = next((n for n in row['names'] if text.strip() == n), None)
    gold = row['gold']
    margin = classes[gold] - max(v for n,v in classes.items() if n != gold)
    return dict(generated_text=text, generated_tokens=tokens, eos_id=eos, terminated=eos is not None,
                actual=actual, native_correct=bool(eos is not None and actual == gold),
                format_compliant=bool(eos is not None and actual is not None),
                gold=gold, full_name_margin=margin, score_correct=bool(margin > 0),
                candidate_log_probabilities=classes, candidate_paths=paths)


def eligible_pairs(rows):
    """Nine unordered transposition edges per family; all nine ordered form pairs."""
    import itertools
    lookup = {(r['family'], tuple(r['assignment']), r['form'], r['query_role']): r for r in rows}
    roles = ['sender','sent_person','destination_person']
    result = []
    for family in sorted({r['family'] for r in rows}):
        for a in itertools.permutations(range(3)):
            for i,j in itertools.combinations(range(3),2):
                b = list(a); b[i],b[j] = b[j],b[i]; b = tuple(b)
                if a >= b:
                    continue
                for fa,fb in itertools.product(['active','passive','goal_fronted'], repeat=2):
                    group = [lookup[family,p,f,q] for p,f in [(a,fa),(b,fb)] for q in roles]
                    result.append(dict(family=family, recipient=list(a), donor=list(b), recipient_form=fa, donor_form=fb,
                                       swapped_roles=[roles[i],roles[j]], invariant_role=next(q for q in roles if q not in [roles[i],roles[j]]),
                                       answer_ids=[r['id'] for r in group], native_correct=sum(r['native_correct'] for r in group),
                                       eligible=all(r['native_correct'] for r in group)))
    return result
