"""Post hoc common-item comparison. No inference or sealed-set access."""
import json
from pathlib import Path
from audit_cached_outputs import norm
ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
out = {}
for start in ['fresh1', 'fresh3']:
    arms = {}
    for arm in ['neutral', 'typed']:
        rows = json.loads((PROJECT / f'research/candidate_free_role_binding_2026-10-14/EVAL_{start}_{arm}.json').read_text())['records']
        arms[arm] = {(r['id'], r['role']): r for r in rows}
        assert len(arms[arm]) == len(rows)
    assert set(arms['neutral']) == set(arms['typed'])
    ids = sorted(set(k[0] for k in arms['neutral']))
    eligible = [i for i in ids if all(arms[a][i,r]['actor'] is not None and arms[a][i,r]['patient'] is not None and arms[a][i,r]['tag_correct'] for a in arms for r in ['actor','patient'])]
    counts = {a: sum((norm(arms[a][i,'actor']['actor']),norm(arms[a][i,'actor']['patient'])) != (norm(arms[a][i,'patient']['actor']),norm(arms[a][i,'patient']['patient'])) for i in eligible) for a in arms}
    out[start] = {'common_parsable_correct_tag_event_count':len(eligible),'events_with_changed_role_fields':counts}
result = {'status':'post_hoc_development_only','limitations':'Requested role, question wording, and typed cue change together. Printed outputs, not hidden states. Not a causal interference test.','results':out}
(ROOT/'COMMON_PAIR_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(out,indent=2))
