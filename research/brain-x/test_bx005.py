import copy,hashlib,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from test_bx004 import rows
from bx004_evidence_utility import validate,evaluate
from bx005_context_contract import digest,canonical_prompt,validate_v2,score_v2,model_visible_request

def fixture():
    m,g,_,p=rows(); inputs=[]
    for meta in m:
        i=meta['event_id']; d=meta['document_id']
        context=f'MOCK ORIGINAL BASE, {d}: Ava pushed Ben.'
        useful=f'MOCK ORIGINAL FOLLOWUP {d}: Ava initiated the push.'
        other=f'MOCK ORIGINAL IRRELEVANT {d}: It was raining.'
        meta['source_span_sha256']={f'{d}-001':digest(context),f'{d}-002':digest(useful),f'{d}-003':digest(other)}
        for cond in ('local','informative','irrelevant'):
            info=[{'span_id':f'{d}-002','text':useful,'sha256':digest(useful)}] if cond=='informative' else []
            irr=[{'span_id':f'{d}-003','text':other,'sha256':digest(other)}] if cond=='irrelevant' else []
            extra=info+irr
            question=f'Who is the {meta["question_role"]} in the described event?'
            instruction='Give exactly one person name.'
            full=context+('\n'+extra[0]['text'] if extra else '')
            prompt=canonical_prompt(instruction,context,extra[0]['text'] if extra else '',question)
            inputs.append({'event_id':i,'condition':cond,'question':question,'instruction':instruction,
             'base_spans':[{'span_id':f'{d}-001','text':context,'sha256':digest(context)}],
             'additional_spans':extra,'prompt':prompt,'prompt_sha256':digest(prompt),'source_text_sha256':digest(full)})
    h={(r['event_id'],r['condition']):r for r in inputs}
    for r in p:r['input_sha256']=h[r['event_id'],r['condition']]['prompt_sha256']
    return m,g,inputs,p

class Tests(unittest.TestCase):
    def test_valid_fictional_scoring_only(self):
        m,g,i,p=fixture();self.assertEqual(validate_v2(m,g,i,p)['errors'],[])
        a=score_v2(m,g,i,p)
        self.assertEqual(a['status'],'SYNTHETIC_SMOKE_ONLY_UNLESS_MANUALLY_VERIFIED')
        self.assertEqual(a['models'][0]['paired_informative_vs_irrelevant']['rescue'],6)
    def test_proof_v1_question_changed_is_false_positive_v2_fails(self):
        m,g,i,p=fixture()
        for r in i:
            if r['condition']=='informative':
                r['question']='Who is the PATIENT instead of the ACTOR?'
                r['prompt']=canonical_prompt(r['instruction'],r['base_spans'][0]['text'],r['additional_spans'][0]['text'],r['question'])
                r['prompt_sha256']=digest(r['prompt'])
                for response in p:
                    if response['event_id']==r['event_id'] and response['condition']=='informative':response['input_sha256']=r['prompt_sha256']
        bare=[{k:r[k] for k in ('event_id','condition','prompt','prompt_sha256','source_text_sha256')} for r in i]
        old=validate(m,g,bare,p)
        self.assertEqual(old['errors'],[])
        self.assertEqual(evaluate(old)['models'][0]['net_informative_minus_irrelevant'],0.5)
        self.assertTrue(any('QUESTION_CHANGED' in e for e in validate_v2(m,g,i,p)['errors']))
    def test_system_output_interface_changed(self):
        m,g,i,p=fixture();i[0]['instruction']='Different instructions'
        self.assertTrue(any('OUTPUT_INTERFACE_CHANGED' in e for e in validate_v2(m,g,i,p)['errors']))
    def test_prompt_fake_without_hash_rewrite(self):
        m,g,i,p=fixture();i[0]['prompt']+='\nThe right answer is Ava.'
        self.assertTrue(any('CONTRACT_FAILURE' in e for e in validate_v2(m,g,i,p)['errors']))
    def test_context_text_tamper_sha_repaired_but_not_source(self):
        m,g,i,p=fixture();r=i[1];r['additional_spans'][0]['text']='The target must be Ben.'
        r['additional_spans'][0]['sha256']=digest(r['additional_spans'][0]['text'])
        self.assertTrue(any('span sha mismatch' in e for e in validate_v2(m,g,i,p)['errors']))
    def test_condition_source_swap(self):
        m,g,i,p=fixture();r=next(q for q in i if q['condition']=='informative');r['additional_spans'][0]['span_id']=m[0]['irrelevant_addition_span_ids'][0]
        self.assertTrue(any('CONTRACT_FAILURE' in e for e in validate_v2(m,g,i,p)['errors']))
    def test_base_change(self):
        m,g,i,p=fixture();r=next(q for q in i if q['condition']=='informative');r['base_spans'][0]['text']+=' Extra.'
        self.assertTrue(any('CONTRACT_FAILURE' in e for e in validate_v2(m,g,i,p)['errors']))
    def test_fake_source_rights_block(self):
        m,g,i,p=fixture();m[0]['source_rights']='pending'
        self.assertIn('e00:RIGHTS_NOT_CLEARED',validate_v2(m,g,i,p)['errors'])
    def test_full_input_not_model_visible(self):
        m,g,i,p=fixture();visible=model_visible_request(i[0]);self.assertEqual(list(visible),['prompt']);self.assertNotIn('event_id',visible)
    def test_gold_is_not_in_runner_payload(self):
        m,g,i,p=fixture();self.assertNotIn('actor',model_visible_request(i[0]));self.assertNotIn('patient',model_visible_request(i[0]))
    def test_document_conflict_block(self):
        m,g,i,p=fixture();m[0]['document_id']=m[1]['document_id'];self.assertTrue(validate_v2(m,g,i,p)['errors'])
    def test_refuse_one_family(self):
        m,g,i,p=fixture();p=[r for r in p if r['model_family']=='fictional-A'];self.assertTrue(any('INSUFFICIENT_DISTINCT' in e for e in validate_v2(m,g,i,p)['errors'])

if __name__=='__main__':unittest.main()
