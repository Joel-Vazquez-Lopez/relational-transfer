"""Saved-data/design audit only. Never imports a model runtime or runs inference.

Run with the existing stronger_model_development_2026-10-15/.venv/bin/python.
Drafts are not an approved/frozen experimental protocol. --tokenize is local-only.
"""
import argparse
import csv
import hashlib
import itertools
import json
import re
from collections import Counter
from pathlib import Path
import zipfile

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OLD = ROOT / 'research/trajectory_relation_audit_2026-10-17'
MODELS = {
    'qwen7b': ('research/stronger_model_development_2026-10-15/models/qwen7b', 3584),
    'llama3b': ('research/llama_gemma_cross_family_2026-10-15/model', 3072),
}
FAMILIES = [
    dict(id='a01', split='development', a='Ada', b='Boris', c='Clara', d='Neil',
         base='recommend', past='recommended', agent_target='a'),
    dict(id='a02', split='development', a='Mara', b='Oscar', c='Hugo', d='Rose',
         base='refer', past='referred', agent_target='d'),
    dict(id='a03', split='held_out', a='Leah', b='Felix', c='Simon', d='Vera',
         base='recommend', past='recommended', agent_target='d'),
    dict(id='a04', split='held_out', a='Eva', b='Colin', c='Ivan', d='Ruby',
         base='refer', past='referred', agent_target='a'),
]
SYSTEM = ('Use only the provided sentence. For participant questions, report roles explicitly '
          'assigned by that sentence. For mention-order questions, use the order of names '
          'in the text. Reply with only Yes or No.')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2) + '\n')


def saved_audit():
    # R=+1 means A is Agent; V=+1 active; Q=+1 Agent question.
    cells = list(itertools.product([1, -1], repeat=3))
    X = np.array([[1, r, v, q, r*v, r*q, v*q, r*v*q] for r,v,q in cells])
    order = np.array([r*v for r,v,q in cells])
    answer = np.array([r if q == 1 else r*v for r,v,q in cells])
    predicted = X @ np.array([0, .5, 0, 0, .5, .5, 0, -.5])
    assert np.array_equal(answer, predicted)
    assert np.array_equal(order, X[:,4])
    cert = dict(status='algebraic_design_fact_not_new_model_result',
                columns=['1','R','V','Q','RV','RQ','VQ','RVQ'],
                rank=int(np.linalg.matrix_rank(X)),
                rank_after_order_and_answer=int(np.linalg.matrix_rank(np.column_stack([X, order, answer]))),
                first_identity='O=R*V',
                expected_answer_identity='Y=(R+R*Q+R*V-R*V*Q)/2',
                rows=[dict(R=r,V=v,Q=q,O=int(o),Y=int(y))
                      for (r,v,q),o,y in zip(cells,order,answer)],
                consequence='Old full factorial is saturated; surface first identity and answer identity are algebraically aliased with its interactions. More families repeat this alias.')
    write_json('OLD_DESIGN_CERTIFICATE.json', cert)
    summary = json.loads((OLD/'QUERY_SUMMARY.json').read_text())
    models = {}
    pairs = []
    for model in ['qwen7b','llama3b']:
        raw = json.loads((OLD/f'QUERY_{model}.json').read_text())
        n_same_answer_role = n_same_question = n_both_pass = 0
        for family in raw['families']:
            fam = family['family']
            rows = [row for task in family['tasks'] for row in task['rows']]
            for u,v in itertools.combinations(rows,2):
                ur = u['condition'].startswith('a_agent')
                vr = v['condition'].startswith('a_agent')
                if u['expected'] == v['expected'] and ur != vr:
                    sameq = u['question_key'] == v['question_key']
                    both = all(r['greedy_correct'] and r['score_correct'] for r in [u,v])
                    n_same_answer_role += 1
                    n_same_question += int(sameq)
                    n_both_pass += int(both)
                    pairs.append(dict(model=model, family=fam['id'],
                                      u=u['condition']+':'+u['question_key'],
                                      v=v['condition']+':'+v['question_key'],
                                      same_question=sameq, both_pass=both,
                                      expected=u['expected']))
        keep = ['correct_answers_by_question','fully_competent_family_ids',
                'all_family_mean_delta','max_prequestion_relative_discrepancy',
                'technical_invariance_passed','descriptive_rule_passed',
                'final_geometry_sign_matches_scalar_output_response']
        models[model] = {k:summary['models'][model][k] for k in keep}
        models[model]['same_answer_different_agent_pairs'] = n_same_answer_role
        models[model]['of_these_same_question'] = n_same_question
        models[model]['of_these_both_answers_pass'] = n_both_pass
    with (HERE/'OLD_PAIR_COVERAGE.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(pairs[0]));w.writeheader();w.writerows(pairs)
    paths = [OLD/'QUERY_LOCK.json',OLD/'QUERY_SUMMARY.json']
    paths += [OLD/f'QUERY_{m}.json' for m in MODELS]
    paths += [OLD/f'QUERY_STATES_{m}.npz' for m in MODELS]
    paths += [ROOT/'research/component_reuse_pilot_2026-10-18/COMPONENT_STATES.npz']
    paths += [ROOT/f'research/trajectory_specificity_benchmark_2026-10-18/STATES_{m}.npz' for m in MODELS]
    paths += [ROOT/f'research/{d}/{fn}' for d,fn in [
        ('component_reuse_pilot_2026-10-18','REPORT.md'),
        ('prefix_shape_audit_2026-10-18','REPORT.md'),
        ('prefix_shape_audit_2026-10-18','SOURCE_MANIFEST.json'),
        ('trajectory_specificity_benchmark_2026-10-18','FINDINGS.md'),
        ('trajectory_specificity_benchmark_2026-10-18','MANIFEST.json'),
    ]]
    inventory=[]
    for p in paths:
        item=dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size,sha256=digest(p))
        if p.suffix == '.npz':
            item['arrays']={}
            with zipfile.ZipFile(p) as z:
                for name in z.namelist():
                    with z.open(name) as f:
                        version=np.lib.format.read_magic(f)
                        reader = { (1,0): np.lib.format.read_array_header_1_0,
                                   (2,0): np.lib.format.read_array_header_2_0 }[version]
                        shape,fortran,dtype=reader(f)
                    item['arrays'][name]=dict(shape=list(shape),dtype=str(dtype),fortran=fortran)
            if p.parent == OLD:
                model=p.stem.replace('QUERY_STATES_','')
                assert item['sha256'] == summary['models'][model]['states_sha256']
        inventory.append(item)
    write_json('SAVED_DATA_AUDIT.json',dict(
        status='existing_files_and_design_only_no_new_inference',
        models=models,inventory=inventory,
        notes=['Pairs are correlated within families, not independent replications.',
               'Different Agent identity does not hold voice and first identity fixed.',
               'Llama historical failed invariance remains failed; no retroactive repair.',
               'NPZ headers and byte hashes inspected, not every scalar revalidated.']))


def draft_rows():
    rows=[]
    for f in FAMILIES:
        for k,p in itertools.product(['within','across'],[0,1]):
            a,b,c,d=(f[x] for x in 'abcd')
            x,y=(b,c) if p == 0 else (c,b)
            sentence=(f'{a} {f["past"]} {x} and {y} to {d}.' if k == 'within'
                      else f'{a} {f["past"]} {x} to {y} and {d}.')
            themes=[x,y] if k == 'within' else [x]
            goals=[d] if k == 'within' else [y,d]
            questions={
                'agent': f'Did {f[f["agent_target"]]} {f["base"]} someone?',
                'theme': f'Was {b} {f["past"]} to someone?',
                'goal': f'Was someone {f["past"]} to {b}?',
                'order': f'In the sentence, was {b} mentioned before {c}?',
            }
            truth={'agent':f['agent_target']=='a','theme':b in themes,
                   'goal':b in goals,'order':p==0}
            for q,question in questions.items():
                rows.append(dict(id=f'{f["id"]}_{k}_{p}_{q}',family=f['id'],
                    split=f['split'],frame=k,swap=p,question_key=q,
                    sentence=sentence,question=question,system=SYSTEM,
                    expected='Yes' if truth[q] else 'No',
                    annotations=dict(agent=a,theme=themes,goal=goals),
                    annotation_status='constructed_gold_pending_independent_human_review'))
    for f in FAMILIES:
        rs=[r for r in rows if r['family']==f['id'] and r['question_key']=='agent']
        assert len({tuple(r['sentence'].replace('.','').split()) for r in rs})==4
        assert len({tuple(sorted(r['sentence'].split())) for r in rs})==1
        assert len({r['expected'] for r in rs}) == 1
        for p in [0,1]:
            texts=[r['sentence'] for r in rs if r['swap']==p]
            assert texts[0].split()[:3] == texts[1].split()[:3]
    assert Counter(r['expected'] for r in rows)=={'Yes':32,'No':32}
    for split in ['development','held_out']:
        assert Counter(r['expected'] for r in rows if r['split']==split)=={'Yes':16,'No':16}
    # An explicit positional/marker-keyed competitor, consuming only surface strings.
    def copier(sentence, question):
        s=sentence.rstrip('.').split()
        within=s[3]=='and'
        assert s[3] in ['and','to'] and s[5] in ['and','to']
        if question.startswith('Did '):
            truth=question.split()[1]==s[0]
        elif question.startswith('Was someone '):
            target=question.rstrip('?').split()[-1]
            truth=target in ([s[6]] if within else [s[4],s[6]])
        elif question.startswith('Was '):
            target=question.split()[1]
            truth=target in ([s[2],s[4]] if within else [s[2]])
        else:
            match=re.fullmatch(r'In the sentence, was (\w+) mentioned before (\w+)\?',question)
            assert match
            truth=s.index(match[1])<s.index(match[2])
        return 'Yes' if truth else 'No'
    copying_correct=sum(copier(r['sentence'],r['question'])==r['expected'] for r in rows)
    assert copying_correct==64
    (HERE/'DRAFT_STIMULI.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    # Explicit restricted rival witnesses. Not fitted to model states.
    pred=[]
    for q in ['agent','theme','goal','order']:
        cells={}
        for r in rows:
            if r['family']=='a01' and r['question_key']==q:
                cells[(r['frame'],r['swap'])]=1 if r['expected']=='Yes' else -1
        answer_I=(cells['across',1]-cells['across',0])-(cells['within',1]-cells['within',0])
        pred.append(dict(question=q,additive_lexical_frame_question_I=0,
                         label_only_answer_writer_I=answer_I,
                         syntax_aware_copy_accuracy='all_gold_labels_by_set_membership',
                         interpretation='I is a vector contrast; answer-writer scalar multiplies an arbitrary fixed output vector.'))
    write_json('PREDICTION_WITNESSES.json',dict(status='toy_predictions_not_LM_findings',rows=pred,
        surface_string_copier_correct=copying_correct,surface_string_copier_total=len(rows),
        implication='All behavioral labels are compatible with a marker/position-keyed program. This pilot cannot uniquely identify a semantic mechanism.'))
    return rows


def tokenize(rows):
    from transformers import AutoTokenizer
    allrows=[]; summary={}
    for m,(path,width) in MODELS.items():
        tok=AutoTokenizer.from_pretrained(ROOT/path,local_files_only=True,trust_remote_code=False)
        out=[]
        for r in rows:
            user='Sentence: '+r['sentence']+'\nQuestion: '+r['question']
            prompt=tok.apply_chat_template([dict(role='system',content=SYSTEM),
                       dict(role='user',content=user)],tokenize=False,add_generation_prompt=True)
            enc=tok(prompt,add_special_tokens=False,return_offsets_mapping=True)
            ids=enc['input_ids']; offsets=enc['offset_mapping']
            start=prompt.index('Sentence: ')+len('Sentence: ')
            end=start+len(r['sentence'])
            qs=prompt.index('\nQuestion: ',end)+len('\nQuestion: ')
            sites=[i for i,(s,e) in enumerate(offsets) if s<end and e>start]
            qsites=[i for i,(s,e) in enumerate(offsets) if s<len(prompt) and e>qs and s<qs+len(r['question'])]
            assert len(ids)<128
            out.append(dict(id=r['id'],model=m,prompt=prompt,token_ids=ids,
                sentence_positions=sites,question_positions=qsites,
                original_prompt_end=len(ids)-1,pad_to=128,
                retained_positions=sorted(set(sites+qsites+[len(ids)-1]))))
        by={r['id']:r for r in out}; tests=[]
        for f in FAMILIES:
            for q in ['agent','theme','goal','order']:
                group=[by[f'{f["id"]}_{k}_{p}_{q}'] for k,p in itertools.product(['within','across'],[0,1])]
                # Names selected without activations: each occupies one space-prefixed token in both tokenizers.
                assert all(len(tok.encode(' '+f[x],add_special_tokens=False))==1 for x in 'bcd')
                assert len({len(r['token_ids']) for r in group}) == 1
                assert len({tuple(r['sentence_positions']) for r in group}) == 1
                first_cues=[]
                for p in [0,1]:
                    left=by[f'{f["id"]}_within_{p}_{q}']['token_ids']
                    right=by[f'{f["id"]}_across_{p}_{q}']['token_ids']
                    diffs=[i for i,(x,y) in enumerate(zip(left,right)) if x!=y]
                    assert len(diffs)==2 # only and/to exchange
                    first_cues.append(min(diffs))
                assert first_cues[0] == first_cues[1]
                tests.append(dict(family=f['id'],question=q,first_frame_cue_position=first_cues[0],
                                  same_positions=True,only_and_to_frame_difference=True))
        aliases={label:{s:tok.encode(s,add_special_tokens=False) for s in forms}
                 for label,forms in {'Yes':['Yes',' Yes','yes',' yes'],
                                     'No':['No',' No','no',' no']}.items()}
        assert all(len(ids)==1 for variants in aliases.values() for ids in variants.values())
        retained=sum(len(r['retained_positions']) for r in out)
        # Upper bound: embedding +28 outgoing residuals and28 attention/MLP writes at every retained site.
        storage=retained*(29+28*2)*width*4
        summary[m]=dict(model_path=path,prompts=len(out),
            prompt_token_range=[min(len(r['token_ids']) for r in out),max(len(r['token_ids']) for r in out)],
            candidate_alias_ids=aliases,checks=tests,retained_token_sites=retained,
            float32_storage_upper_bound_bytes=storage,
            tokenizer_sha256=digest(ROOT/path/'tokenizer.json'),
            chat_template_sha256=hashlib.sha256(tok.chat_template.encode()).hexdigest())
        allrows+=out
    (HERE/'DRAFT_TOKENIZATION.jsonl').write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in allrows))
    write_json('TOKENIZER_AUDIT.json',dict(status='local_tokenization_only_no_model_execution',models=summary))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tokenize',action='store_true');args=ap.parse_args()
    saved_audit();rows=draft_rows()
    if args.tokenize:tokenize(rows)
    write_json('MANIFEST.json',dict(status='DRAFT_PENDING_REVIEW_NO_INFERENCE_AUTHORIZED',
        new_model_forwards=0,original_audit_commit='7cfb40634baf9783f7a1c0a057cde8feb82dbed9',
        latest_evidence_commit='f59a798d2a2c76f6cf9ba4e73541cbf5b94fdf1a',
        proposed_models=list(MODELS),families=FAMILIES,unique_prompts_per_model=64,
        proposed_main_forwards=128,proposed_validation_forwards=4,
        paid_calls=0,downloads=0,
        files={p.name:dict(bytes=p.stat().st_size,sha256=digest(p))
               for p in sorted(HERE.iterdir()) if p.is_file() and p.name not in ['MANIFEST.json','GITHUB_PRESERVATION.json']}))
    print(json.dumps(dict(status='design_audit_complete',new_model_forwards=0,
                          stimuli=len(rows),files=[p.name for p in HERE.iterdir() if p.is_file()])))


if __name__ == '__main__':main()
