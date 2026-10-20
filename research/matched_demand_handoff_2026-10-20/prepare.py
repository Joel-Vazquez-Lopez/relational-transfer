"""CPU-only, outcome-blind tokenizer, annotation and budget preparation."""
import itertools
import importlib.metadata
import re
import shutil
from transformers import AutoTokenizer
from core import *

def main():
    assert not (HERE/'LOCK.json').exists()
    previous = json.loads((ROOT/'research/three_role_llama_native_2026-10-20/LOCK.json').read_text())
    model = previous['model']
    tok = AutoTokenizer.from_pretrained(str(ROOT/model['path']), local_files_only=True, trust_remote_code=False)
    role_questions = ['Who is the sender?', 'Who is being sent?', 'Who is the recipient?']
    order_questions = ['Who is mentioned first?', 'Who is mentioned second?', 'Who is mentioned third?']
    assert all(len(tok.encode(q, add_special_tokens=False)) == 5 for q in role_questions+order_questions)
    system = 'Read the sentence and answer the question about that sentence. Return only the person\'s name, copied exactly from the sentence. Do not explain.'
    rows, checks = [], []
    eos = json.loads((ROOT/model['path']/'config.json').read_text())['eos_token_id']
    for family, names in [('u01', ['Alice','Bob','Carol']), ('u02', ['Dave','Erin','Frank'])]:
        candidate_ids = {n: tok.encode(' '+n, add_special_tokens=False)[0] for n in names}
        assert all(tok.encode(' '+n, add_special_tokens=False) == [candidate_ids[n]] for n in names)
        for form, assignment in itertools.product(FORMS, itertools.permutations(range(3))):
            sentence = source(names, assignment, form)
            parsed = re.fullmatch(r'(\w+) sent (\w+) to (\w+)\.', sentence) if form == 'active' else re.fullmatch(r'To (\w+), (\w+) sent (\w+)\.', sentence)
            role_names = list(parsed.groups()) if form == 'active' else [parsed[2],parsed[3],parsed[1]]
            assert role_names == [names[i] for i in assignment]
            sr = surface_roles(form)
            for demand, questions in [('role', role_questions), ('ordinal', order_questions)]:
                for qi, question in enumerate(questions):
                    role_index = qi if demand == 'role' else sr[qi]
                    gold = names[assignment[role_index]]
                    prompt = tok.apply_chat_template([dict(role='system', content=system), dict(role='user',content='Sentence: '+sentence+'\nQuestion: '+question)],tokenize=False,add_generation_prompt=True)+'Answer:'
                    enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
                    ids, offsets = enc['input_ids'], enc['offset_mapping']
                    start = prompt.index('Sentence: ')+len('Sentence: ')
                    at = lambda p: next(i for i,(lo,hi) in enumerate(offsets) if lo <= p < hi)
                    anchors = [at(start+m.end()-1) for m in re.finditer(r'\b(?:'+'|'.join(names)+r')\b', sentence)]
                    assert len(anchors) == 3
                    for anchor, ri in zip(anchors, sr):
                        assert ids[anchor] == candidate_ids[names[assignment[ri]]]
                    for n,t in candidate_ids.items():
                        assert tok.encode(prompt+' '+n, add_special_tokens=False) == ids+[t]
                    assert len(ids)+MAX_NEW < LENGTH
                    row = dict(id=f'{family}_{form}_{"".join(map(str,assignment))}_{demand}{qi}',family=family,form=form,assignment=list(assignment),source=sentence,question=question,demand=demand,query_index=qi,role_index=role_index,gold=gold,names=names,candidate_ids=candidate_ids,token_ids=ids,prompt=prompt,pad=87,eos_ids=eos,period=at(start+len(sentence)-1),anchors=anchors)
                    rows.append(row)
                    checks.append(dict(id=row['id'],prompt_tokens=len(ids),question_tokens=5,complete_single_token_names=True,parsed_roles=role_names))
    assert len(rows)==144
    lookup = {(r['family'],r['form'],tuple(r['assignment']),r['demand'],r['query_index']):r for r in rows}
    units = []
    for family, form in itertools.product(['u01','u02'],FORMS):
        for edge, b in enumerate([(0,2,1),(1,0,2),(2,1,0)]):
            queries=[]
            for demand,qi in itertools.product(['role','ordinal'],range(3)):
                a = lookup[family,form,(0,1,2),demand,qi]; z = lookup[family,form,b,demand,qi]
                assert len(a['token_ids'])==len(z['token_ids'])
                assert a['period']==z['period']
                different=[i for i,(x,y) in enumerate(zip(a['token_ids'],z['token_ids'])) if x!=y]
                assert set(different) <= set(a['anchors']) and len(different)==2
                queries.append(dict(a=a['id'],b=z['id'],demand=demand,query_index=qi,role_index=a['role_index'],changed=a['gold']!=z['gold']))
            units.append(dict(id=f'{family}_{form}_e{edge}',family=family,form=form,edge=edge,assignment_a=[0,1,2],assignment_b=list(b),queries=queries))
    for family, form, assignment in itertools.product(['u01','u02'],FORMS,itertools.permutations(range(3))):
        group=[lookup[family,form,assignment,d,q] for d,q in itertools.product(['role','ordinal'],range(3))]
        assert len({len(r['token_ids']) for r in group})==1
        assert all(r['token_ids'][:r['period']+1]==group[0]['token_ids'][:r['period']+1] for r in group)
        for ri in range(3):
            assert lookup[family,form,assignment,'role',ri]['gold']==lookup[family,form,assignment,'ordinal',surface_roles(form).index(ri)]['gold']
    # No inference or outcome-driven source selection occurs here.
    save(HERE/'INPUTS.json',rows); save(HERE/'UNITS.json',units); save(HERE/'TOKENIZER_AUDIT.json',checks)
    prior=ROOT/'research/three_role_llama_native_2026-10-20'
    shutil.copyfile(prior/'CONTROL_INPUT.json',HERE/'CONTROL_INPUT.json')
    shutil.copyfile(prior/'REFERENCE_CONTROL.npz',HERE/'REFERENCE_CONTROL.npz')
    from test_design import main as test
    test()
    for name,value in model['files'].items(): assert sha(ROOT/name)==value['sha256']
    save(HERE/'SOURCE_MANIFEST.json',dict(annotation='AI-only authored, regex-parsed and tokenizer-checked. No human validation.',families=['u01','u02'],names=[['Alice','Bob','Carol'],['Dave','Erin','Frank']],native_prompts=144,source_cases=24,edges=12,scientific_generations=432,exact_question_length=5,unused_old_evaluation_families=['t03','t04'],reference_origin='research/three_role_llama_native_2026-10-20/REFERENCE_CONTROL.npz',previous_failure_commit='97e53a21ea2780b6510c808a51accbe37de4e77d',freshness='New authored combinations and question demands; common names may have appeared elsewhere. No model outputs on this frozen factorial were inspected. Not a new corpus sample.'))
    files=['core.py','prepare.py','worker.py','runtime.py','run.py','analyze.py','test_design.py','PROTOCOL.md','INPUTS.json','UNITS.json','TOKENIZER_AUDIT.json','CPU_PREFLIGHT.json','SOURCE_MANIFEST.json','CONTROL_INPUT.json','REFERENCE_CONTROL.npz']
    save(HERE/'LOCK.json',dict(files={n:sha(HERE/n) for n in files},native_sources=previous['native_sources'],model=model,max_model_forwards=2630,physical_length=LENGTH,max_new_tokens=MAX_NEW,cuts=CUTS,primary_cut=14,runtime={p:importlib.metadata.version(p) for p in ['mlx','mlx-lm','numpy','transformers']},new_outcomes_seen=False))
    print(json.dumps(dict(rows=len(rows),units=len(units),scientific_generations=432,token_lengths=sorted({len(r['token_ids']) for r in rows}))))

if __name__=='__main__': main()
