"""Full candidate-sequence scoring plus actual greedy completion.

No model imports. get_logits(prefix) must return logits following that exact
continuation, under the same masks and numerical shape as the primary prompt.
"""
import numpy as np
from core import lse

MAX_NEW_TOKENS=4

def full_score(get_logits,row,decode):
    first=np.asarray(get_logits(())); cached={():first}
    def at(prefix):
        prefix=tuple(prefix)
        if prefix not in cached: cached[prefix]=np.asarray(get_logits(prefix))
        return cached[prefix]
    path_scores={}; classes={}
    for name in row['names']:
        # Distinct paths only: never double count duplicate tokenizations.
        sequences=sorted({tuple(x) for x in row['candidate_sequences'][name]})
        assert sequences and all(seq for seq in sequences)
        path_scores[name]=[]
        for seq in sequences:
            value=0.; terms=[]
            for j,token in enumerate(seq):
                logits=at(seq[:j]); term=float(logits[token])-lse(logits)
                assert np.isfinite(term)
                value+=term; terms.append(term)
            path_scores[name].append(dict(token_ids=list(seq),log_probability=value,token_log_probabilities=terms))
        classes[name]=lse([x['log_probability'] for x in path_scores[name]])
    tokens=[]; termination=None
    for j in range(MAX_NEW_TOKENS):
        token=int(np.argmax(at(tokens)))
        if token in row['eos_ids']:
            termination=token; break
        tokens.append(token)
    text=decode(tokens)
    actual=next((n for n in row['names'] if text.strip()==n),None)
    gold=row['expected']; rival=next(n for n in row['names'] if n!=gold)
    margin=classes[gold]-classes[rival]
    terminated=termination is not None
    top=np.argsort(first)[-20:][::-1]
    return dict(expected=gold,argmax_id=int(np.argmax(first)),argmax_token=decode([int(np.argmax(first))]),
        actual=actual,generated_text=text,generated_token_ids=tokens,eos_id=termination,
        terminated=terminated,greedy_correct=bool(terminated and actual==gold),
        score_correct=bool(margin>0),correct=bool(terminated and actual==gold and margin>0),margin=margin,
        full_name_log_probability=classes,candidate_paths=path_scores,
        top20=[dict(id=int(i),token=decode([int(i)]),logit=float(first[i])) for i in top],
        scoring_scope='Probability of the natural full-name prefix, summed over distinct space/no-space paths; no EOS score. Greedy correctness separately requires observed EOS within four tokens.',
        unique_logit_prefixes=[list(p) for p in sorted(cached)])
