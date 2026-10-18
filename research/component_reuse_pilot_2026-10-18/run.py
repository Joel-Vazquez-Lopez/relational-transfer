"""Single frozen local pilot. Run once; scientific gates never adapt to outcomes."""
import contextlib
import hashlib
import itertools
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import generate, load

HERE = Path(__file__).resolve().parent
MODEL = HERE.parent / 'stronger_model_development_2026-10-15/models/qwen7b'
LOCK = json.loads((HERE/'PROTOCOL.json').read_text())
PARTS = ['incoming', 'attention', 'mlp', 'outgoing']
SITES = ['person_a', 'person_b', 'verb', 'period', 'verb_previous', 'period_previous']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, value):
    (HERE/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def expected_person(condition, question):
    agent = 'a' if condition.endswith('_a') else 'b'
    first = agent if condition.startswith(('active', 'cleft')) else ('b' if agent == 'a' else 'a')
    other = lambda x: 'b' if x == 'a' else 'a'
    return {'agent': agent, 'patient': other(agent), 'first': first, 'last': other(first)}[question]


def build(tok, fam, condition, question, mapping):
    sentence = LOCK['sentences'][condition].format(**fam)
    q = LOCK['questions'][question].format(**fam)
    codes = LOCK['code_maps'][mapping]
    user = LOCK['user_template'].format(sentence=sentence, question=q,
                                       code_a=codes['a'], code_b=codes['b'], **fam)
    prompt = tok.apply_chat_template([{'role': 'system', 'content': LOCK['system']},
                                     {'role': 'user', 'content': user}],
                                    tokenize=False, add_generation_prompt=True)
    ids = tok.encode(prompt, add_special_tokens=False)
    enc = tok._tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=False)
    assert enc['input_ids'] == ids
    start = prompt.index('Sentence: '+sentence)+len('Sentence: ')
    def at(char):
        hits = [i for i, (a,b) in enumerate(enc['offset_mapping']) if a <= char < b]
        assert len(hits) == 1, (char, hits)
        return hits[0]
    sites = {key: at(start+sentence.index(value)+len(value)-1)
             for key, value in [('person_a',fam['a']), ('person_b',fam['b']), ('verb',fam['verb'])]}
    sites['period'] = at(start+len(sentence)-1)
    for key in ['verb', 'period']:
        sites[key+'_previous'] = sites[key]-1
    return {'prompt': prompt, 'ids': ids, 'sites': sites, 'sentence': sentence,
            'question': q, 'expected': codes[expected_person(condition,question)],
            'site_tokens': {s: tok.decode([ids[i]]) for s,i in sites.items()}}


class Operation(nn.Module):
    def __init__(self, original, parent, key, patch):
        super().__init__()
        self.original = original
        # Plain dict avoids registering a circular Module reference.
        self.record = parent
        self.key = key
        self.patch = patch

    def __call__(self, *args, **kwargs):
        y = self.original(*args, **kwargs)
        self.record[self.key] = y
        if self.patch is not None:
            pos, vector = self.patch
            v = mx.array(vector).astype(y.dtype).reshape(1,1,-1)
            y = mx.concatenate([y[:,:pos], v, y[:,pos+1:]], axis=1)
        return y


class Block(nn.Module):
    def __init__(self, original, record):
        super().__init__()
        self.original = original
        self.record = record
        self.use_sliding = getattr(original,'use_sliding',False)

    def __call__(self, x, *args, **kwargs):
        self.record['incoming'] = x
        y = self.original(x,*args,**kwargs)
        self.record['outgoing'] = y
        return y


@contextlib.contextmanager
def instrument(model, capture=False, patch=None):
    blocks = model.model.layers
    chosen = range(len(blocks)) if capture else ([patch['layer']] if patch else [])
    old, records = [], {}
    for li in chosen:
        block = blocks[li]
        attention, mlp = block.self_attn, block.mlp
        record = {}
        records[li] = record
        def replacement(component):
            if patch and li == patch['layer'] and component == patch['component']:
                return patch['position'], patch['vector']
            return None
        block.self_attn = Operation(attention,record,'attention',replacement('attention'))
        block.mlp = Operation(mlp,record,'mlp',replacement('mlp'))
        blocks[li] = Block(block,record)
        old.append((li,block,attention,mlp))
    try:
        yield records
    finally:
        for li,block,attention,mlp in old:
            block.self_attn, block.mlp = attention,mlp
            blocks[li] = block


def forward(model, tok, codes, data, patch=None, capture=False):
    with instrument(model,capture,patch) as records:
        logits = model(mx.array([data['ids']]))[0,-1].astype(mx.float32)
        logp = logits-mx.logsumexp(logits)
        best = mx.argmax(logits)
        selected = logp[mx.array(codes)]
        mx.eval(selected,best)
        best_id = int(best.item())
        score = {'logp': dict(zip(['X','Y'],map(float,np.asarray(selected)))),
                 'greedy_token_id': best_id, 'greedy_token': tok.decode([best_id])}
        captured = None
        if capture:
            vectors = []
            positions = [data['sites'][site] for site in SITES]
            for li in range(len(model.model.layers)):
                row = []
                for part in PARTS:
                    v = records[li][part][0,mx.array(positions),:].astype(mx.float32)
                    mx.eval(v)
                    row.append(np.asarray(v).copy())
                vectors.append(np.stack(row))
            captured = np.stack(vectors)
    return score,captured


def oriented(score, expected):
    other = 'Y' if expected == 'X' else 'X'
    return score['logp'][other]-score['logp'][expected]


def relative(a,b):
    return float(np.linalg.norm((a-b).astype(np.float64))/max(np.linalg.norm(b.astype(np.float64)),1e-30))


def main():
    assert not (HERE/'RUN_STARTED.json').exists(), 'refusing a second run'
    started = time.monotonic()
    metadata = {'protocol_sha256': sha(HERE/'PROTOCOL.json'), 'runner_sha256':sha(__file__),
                'model_config_sha256':sha(MODEL/'config.json'), 'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    save('RUN_STARTED.json', metadata)
    model,tok = load(str(MODEL),tokenizer_config={'trust_remote_code':False})
    model.eval()
    mx.set_cache_limit(256*1024*1024)
    code_ids = [tok.encode(c,add_special_tokens=False) for c in ['X','Y']]
    assert all(len(x)==1 for x in code_ids) and code_ids[0]!=code_ids[1]
    code_ids = [x[0] for x in code_ids]
    metadata['code_token_ids'] = code_ids
    prompts,baselines = {},{}
    for fam,condition,q,mapping in itertools.product(LOCK['families'],LOCK['sentences'],LOCK['questions'],LOCK['code_maps']):
        key = (fam['id'],condition,q,mapping)
        data = build(tok,fam,condition,q,mapping)
        score,_ = forward(model,tok,code_ids,data)
        generated = generate(model,tok,data['prompt'],max_tokens=10).strip()
        row = dict(family=fam['id'],condition=condition,question_key=q,mapping=mapping,
                   **data,**score,generated=generated)
        row['margin_correct'] = -oriented(score,data['expected'])
        row['pass'] = generated == data['expected'] and row['margin_correct']>0
        prompts[key],baselines[key] = data,row
        print('baseline',*key,repr(generated),data['expected'],round(row['margin_correct'],3),flush=True)
        mx.clear_cache()
    assert len(baselines)==192
    save('BASELINES.json', {'metadata':metadata,'rows':list(baselines.values())})
    eligible = {}
    units = []
    for fam,orientation in itertools.product(LOCK['families'],LOCK['orientations']):
        cells = LOCK['orientations'][orientation]
        rows = [baselines[(fam['id'],cells[c],q,m)] for c,q,m in itertools.product(
            ['receiver','target','voice_control'],LOCK['questions'],LOCK['code_maps'])]
        yes = all(row['pass'] for row in rows)
        eligible[(fam['id'],orientation)] = yes
        units.append({'family':fam['id'],'orientation':orientation,'eligible':yes,
                      'correct':sum(row['pass'] for row in rows),'total':len(rows)})
    development = [f for f in LOCK['families'] if f['split']=='development' and eligible[(f['id'],'a')]]
    evaluation = [f for f in LOCK['families'] if f['split']=='evaluation']
    competence = bool(development) and all(any(eligible[(f['id'],o)] for o in LOCK['orientations']) for f in evaluation)
    save('COMPETENCE.json', {'units':units,'causal_gate_pass':competence,
                            'baseline_correct':sum(r['pass'] for r in baselines.values()),'baseline_total':192})
    print('COMPETENCE',competence,units,flush=True)
    arrays, tech = {},{'reconstruction':[],'prefix_invariance':[],'identity':[]}
    def capture_family(fam):
        for condition in LOCK['sentences']:
            data = prompts[(fam['id'],condition,'agent','normal')]
            _,array = forward(model,tok,code_ids,data,capture=True)
            arrays[(fam['id'],condition)] = array
            errors = [relative(array[l,0]+array[l,1]+array[l,2],array[l,3]) for l in range(array.shape[0])]
            tech['reconstruction'].append({'family':fam['id'],'condition':condition,'max_relative_error':max(errors),'by_layer':errors})
            print('capture',fam['id'],condition,'reconstruction',max(errors),flush=True)
            mx.clear_cache()
    for fam in LOCK['families']:
        if fam['split']!='development':
            continue
        capture_family(fam)
        reference = arrays[(fam['id'],'active_a')]
        for q,m in itertools.product(LOCK['questions'],LOCK['code_maps']):
            _,array = forward(model,tok,code_ids,prompts[(fam['id'],'active_a',q,m)],capture=True)
            errors = [relative(array[l,p,s],reference[l,p,s]) for l,p,s in itertools.product(
                range(array.shape[0]),range(4),range(6))]
            tech['prefix_invariance'].append({'family':fam['id'],'question':q,'mapping':m,'max_relative_error':max(errors)})
            mx.clear_cache()
    def technical_pass():
        return (all(r['max_relative_error']<=0.002 for r in tech['reconstruction']) and
                all(r['max_relative_error']<0.001 for r in tech['prefix_invariance']) and
                all(r['margin_discrepancy']<=0.05 for r in tech['identity']))
    patchrows = []
    def patch(fam,orientation,candidate,q,m,donor,vector,site=None,phase='development'):
        receiver = LOCK['orientations'][orientation]['receiver']
        key = (fam['id'],receiver,q,m)
        data = prompts[key]
        score,_ = forward(model,tok,code_ids,data,patch={**candidate,'position':data['sites'][site or candidate['site']],'vector':vector})
        effect = oriented(score,data['expected'])-oriented(baselines[key],data['expected'])
        row = {'phase':phase,'family':fam['id'],'orientation':orientation,**candidate,
               'patch_site':site or candidate['site'],'question':q,'mapping':m,'donor':donor,
               'baseline_code':data['expected'],'baseline_margin':oriented(baselines[key],data['expected']),
               'patched_margin':oriented(score,data['expected']),'effect':effect,**score}
        patchrows.append(row)
        assert len(patchrows)<=LOCK['max_scored_patch_conditions']
        if donor=='self':
            tech['identity'].append({'phase':phase,'family':fam['id'],'orientation':orientation,
                                     **candidate,'question':q,'mapping':m,'margin_discrepancy':abs(effect)})
        mx.clear_cache()
        return row
    def vector(fid,condition,candidate,site=None):
        return arrays[(fid,condition)][candidate['layer'],PARTS.index(candidate['component']),SITES.index(site or candidate['site'])]
    candidates = []
    decision = 'STOP_COMPETENCE' if not competence else ('STOP_TECHNICAL' if not technical_pass() else 'DEVELOPMENT')
    if decision=='DEVELOPMENT':
        for layer,component,site in itertools.product(LOCK['candidate_grid']['layers_zero_indexed'],
                                                     LOCK['candidate_grid']['components'],LOCK['candidate_grid']['sites']):
            candidate = {'layer':layer,'component':component,'site':site}
            rows = []
            for fam,q,donor in itertools.product(development,['agent','first'],['target','voice_control','self']):
                cells = LOCK['orientations']['a']
                condition = cells['receiver'] if donor=='self' else cells[donor]
                rows.append(patch(fam,'a',candidate,q,'normal',donor,vector(fam['id'],condition,candidate)))
            R = float(np.mean([r['effect'] for r in rows if r['donor']=='target' and r['question']=='agent']))
            P = float(np.mean([abs(r['effect']) for r in rows if r['donor']=='target' and r['question']=='first']))
            V = float(np.mean([abs(r['effect']) for r in rows if r['donor']=='voice_control' and r['question']=='agent']))
            c = {**candidate,'R':R,'P':P,'V':V,'selection_score':R-P-V,'qualifies':R>1 and R>2*P and R>2*V}
            candidates.append(c)
            print('development',c,flush=True)
        decision = 'STOP_TECHNICAL' if not technical_pass() else 'SELECT'
    selected = None
    if decision=='SELECT':
        qualified = [c for c in candidates if c['qualifies']]
        if qualified:
            best = sorted(qualified,key=lambda c:(-c['selection_score'],c['layer'],c['component'],c['site']))[0]
            selected = {k:best[k] for k in ['layer','component','site']}
            decision = 'EVALUATION'
        else:
            decision = 'STOP_NO_COMPONENT'
    # Written before any evaluation captures or interventions.
    save('DEVELOPMENT_DECISION.json',{'decision':decision,'selected':selected,'candidates':candidates,
                                     'technical_checks':tech,'patch_count':len(patchrows)})
    save('PATCHES_DEVELOPMENT.json',patchrows)
    evaluations = []
    if decision=='EVALUATION':
        for fam in evaluation:
            capture_family(fam)
        if not technical_pass():
            decision = 'STOP_TECHNICAL'
        else:
            for fam,orientation in itertools.product(evaluation,LOCK['orientations']):
                if not eligible[(fam['id'],orientation)]:
                    continue
                cells = LOCK['orientations'][orientation]
                receiver = vector(fam['id'],cells['receiver'],selected)
                target = vector(fam['id'],cells['target'],selected)
                donors = {key:vector(fam['id'],cells['receiver'] if key=='self' else cells[key],selected)
                          for key in ['self','target','voice_control','role_matched']}
                neighbor = selected['site']+'_previous'
                donors['neighbor'] = vector(fam['id'],cells['target'],selected,neighbor)
                sham_seeds = {}
                for i in range(3):
                    text = json.dumps([fam['id'],orientation,selected,i],sort_keys=True)
                    seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8],'big')
                    direction = np.random.default_rng(seed).normal(size=receiver.shape).astype(np.float32)
                    donors[f'sham_{i}'] = receiver+direction*(np.linalg.norm(target-receiver)/np.linalg.norm(direction))
                    sham_seeds[f'sham_{i}'] = seed
                for m in LOCK['code_maps']:
                    rows = []
                    for q,(donor,v) in itertools.product(LOCK['questions'],donors.items()):
                        rows.append(patch(fam,orientation,selected,q,m,donor,v,
                                          site=neighbor if donor=='neighbor' else None,phase='evaluation'))
                    def get(d,q):
                        return next(r for r in rows if r['donor']==d and r['question']==q)
                    switches = {}
                    for d in ['target','role_matched','self','voice_control']:
                        switches[d] = {}
                        for q in LOCK['questions']:
                            r = get(d,q)
                            wanted = ('Y' if r['baseline_code']=='X' else 'X') if d in ['target','role_matched'] and q in ['agent','patient'] else r['baseline_code']
                            switches[d][q] = {'expected':wanted,'observed':r['greedy_token'],
                                              'correct':r['greedy_token'].strip()==wanted}
                    role_effect = float(np.mean([get('target',q)['effect'] for q in ['agent','patient']]))
                    order_effect = max(abs(get('target',q)['effect']) for q in ['first','last'])
                    sham_effect = max(abs(float(np.mean([get(f'sham_{i}',q)['effect'] for q in ['agent','patient']]))) for i in range(3))
                    criteria = {'codes':all(v['correct'] for d in switches.values() for v in d.values()),
                                'role_effect_over_1':role_effect>1,'role_over_twice_order':role_effect>2*order_effect,
                                'role_over_shams_plus_half':role_effect>sham_effect+0.5}
                    evaluations.append({'family':fam['id'],'orientation':orientation,'mapping':m,
                                        'answers':switches,'role_effect':role_effect,'max_order_effect':order_effect,
                                        'max_absolute_sham_role_effect':sham_effect,'sham_seeds':sham_seeds,
                                        'criteria':criteria,'pass':all(criteria.values())})
                    print('evaluation',fam['id'],orientation,m,criteria,flush=True)
            decision = ('STOP_TECHNICAL' if not technical_pass() else
                        'BOUNDED_PASS' if evaluations and all(e['pass'] for e in evaluations) else 'BOUNDED_FAIL')
    archive = {'__'.join(k):v for k,v in arrays.items()}
    archive.update(parts=np.array(PARTS),sites=np.array(SITES))
    np.savez_compressed(HERE/'COMPONENT_STATES.npz',**archive)
    save('PATCHES.json',patchrows)
    save('TECHNICAL.json',tech)
    save('RESULT.json',{'metadata':metadata,'decision':decision,'selected':selected,'evaluations':evaluations,
                        'candidate_count':len(candidates),'patch_count':len(patchrows),
                        'causal_gate_pass':competence,'technical_pass':technical_pass(),
                        'state_sha256':sha(HERE/'COMPONENT_STATES.npz'),
                        'elapsed_seconds':time.monotonic()-started})
    print('FINAL',decision,'patches',len(patchrows),'seconds',round(time.monotonic()-started,1),flush=True)


if __name__=='__main__':
    main()
