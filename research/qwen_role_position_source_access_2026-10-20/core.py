"""Declared factorial, scoring and serialization; no model initialization."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
LENGTH, WORKER_CAP, MAX_NEW = 128, 32, 4
CUTS = [0, 7, 14, 21, 28]
ROLES = ['sender', 'sent_person', 'destination_person']
FORMS = ['active', 'passive', 'goal_fronted']

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''): h.update(b)
    return h.hexdigest()

def array_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float32).tobytes()).hexdigest()

def save(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)

def append(path, obj):
    with Path(path).open('a') as f:
        f.write(json.dumps(obj, allow_nan=False) + '\n'); f.flush(); os.fsync(f.fileno())

def source(names, assignment, form):
    a, b, c = [names[i] for i in assignment[:3]]
    return f'{a} sent {b} to {c}.' if form == 'active' else f'{b} was sent to {c} by {a}.' if form == 'passive' else f'To {c}, {a} sent {b}.'

def surface_roles(form):
    return [0, 1, 2] if form == 'active' else [1, 2, 0] if form == 'passive' else [2, 0, 1]

def source_authority(aa, bb, ba, ab):
    """All arguments are B-answer minus A-answer logits; no clipping."""
    denominator = bb - aa
    return (ba - ab) / denominator if denominator >= 1.0 else None

def score_generation(row, get, decode):
    first = get(())
    tokens, eos = [], None
    for _ in range(MAX_NEW):
        t = int(np.argmax(get(tuple(tokens))))
        if t in row['eos_ids']:
            eos = t; break
        tokens.append(t)
    text = decode(tokens)
    name = next((n for n in row['names'] if text.strip() == n), None)
    return dict(generated_tokens=tokens, generated_text=text, eos_id=eos,
                terminated=eos is not None, actual=name,
                native_correct=eos is not None and name == row['gold'],
                first_token=int(np.argmax(first)),
                first_token_correct=int(np.argmax(first)) == row['candidate_ids'][row['gold']],
                name_logits={n: float(first[t]) for n, t in row['candidate_ids'].items()})
