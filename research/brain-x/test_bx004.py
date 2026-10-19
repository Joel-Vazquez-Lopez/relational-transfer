import hashlib, json, unittest
from bx004_evidence_utility import validate,evaluate,score_response,candidate_intake

def rows():
    manifest=[];gold=[];inputs=[];responses=[]
    for i in range(12):
        eid=f'e{i:02}';doc=f'd{i:02}';role='actor' if i%2==0 else 'patient'
        manifest.append({'event_id':eid,'document_id':doc,'source_commit':'mock_0001',
          'source_rights':'approved_for_internal_model_evaluation','contrast_adjudication':'human_verified_independently_informative_vs_irrelevant',
          'gold_adjudication':'completed_two_blind_humans_plus_independent_adjudicator','local_evidence_status':'insufficient' if i%3==0 else 'sufficient','question_role':role,
          'context_provenance':'unaltered_original_document_subspans','local_span_ids':[f'{doc}-001'],
          'informative_addition_span_ids':[f'{doc}-002'],'irrelevant_addition_span_ids':[f'{doc}-003']})
        gold.append({'event_id':eid,'document_id':doc,'actor':'Ava','patient':'Ben','annotators':['synthetic_A','synthetic_B'],'adjudicator':'synthetic_C'})
        correct='Ava' if role=='actor' else 'Ben';wrong='Ben' if role=='actor' else 'Ava'
        for cond in ('local','informative','irrelevant'):
            prompt=f'MOCK example {i}, {cond}, question {role}'
            h=hashlib.sha256(prompt.encode()).hexdigest()
            inputs.append({'event_id':eid,'condition':cond,'prompt':prompt,'prompt_sha256':h,'source_text_sha256':'synthetic_fixture'})
            for fam in ('fictional-A','fictional-B'):
                resp=correct if (cond=='informative' or (i%2==0)) else wrong
                responses.append({'model_family':fam,'model_revision':'mock_only','event_id':eid,'condition':cond,'response':resp,'input_sha256':h})
    return manifest,gold,inputs,responses

class BX004Tests(unittest.TestCase):
    def test_valid_synthetic_fixture(self):
        v=validate(*rows())
        self.assertEqual(v['errors'],[])
        score=evaluate(v)
        self.assertEqual(score['status'],'EXPLORATORY_NOT_MECHANISM')
        self.assertEqual(len(score['models']),2)
        self.assertEqual(score['models'][0]['paired_informative_vs_irrelevant']['rescue'],6)
        self.assertEqual(score['models'][0]['paired_informative_vs_irrelevant']['harm'],0)
    def test_no_real_humans_means_block(self):
        m,g,i,p=rows();m[0]['gold_adjudication']='not_yet_reviewed'
        self.assertEqual(evaluate(validate(m,g,i,p))['status'],'BLOCKED')
    def test_license_block(self):
        m,g,i,p=rows();m[0]['source_rights']='pending'
        self.assertIn('e00:RIGHTS_NOT_CLEARED',validate(m,g,i,p)['errors'])
    def test_leakage_block(self):
        m,g,i,p=rows();i[0]['gold']='Ava'
        self.assertTrue(any('MODEL_INPUT_EXTRA_FIELDS' in e for e in validate(m,g,i,p)['errors']))
    def test_missing_model_family(self):
        m,g,i,p=rows();p=[r for r in p if r['model_family']=='fictional-A']
        self.assertIn('INSUFFICIENT_DISTINCT_MODEL_FAMILIES',validate(m,g,i,p)['errors'])
    def test_duplicate_group(self):
        m,g,i,p=rows();m[1]['document_id']=m[0]['document_id'];g[1]['document_id']=m[0]['document_id']
        self.assertEqual(evaluate(validate(m,g,i,p))['status'],'EXPLORATORY_NOT_MECHANISM')
        self.assertEqual(validate(m,g,i,p)['documents'],11)
    def test_missing_condition(self):
        m,g,i,p=rows();p=[r for r in p if not (r['event_id']=='e00' and r['condition']=='irrelevant' and r['model_family']=='fictional-A')]
        self.assertTrue(any('MISSING_RESPONSE' in e for e in validate(m,g,i,p)['errors']))
    def test_response_prompt_hash_disagreement(self):
        m,g,i,p=rows();p[0]['input_sha256']='wrong'
        self.assertTrue(any('RESPONSE_PROMPT_HASH_MISMATCH' in e for e in validate(m,g,i,p)['errors'])
    def test_unannotated_candidate_gate(self):
        items=[{'document_id':f'd{i}','genre':'weblog','human_verified':False,'rights':'underlying_text_per_document_review_required'} for i in range(25)]
        v=candidate_intake(items)
        self.assertEqual((v['candidate_sentences'],v['documents'],v['fully_eligible_for_bx004']), (25,25,0))
    def test_reject_unsupported_rewrite(self):
        m,g,i,p=rows();m[0]['context_provenance']='synthetic_paraphrase'
        self.assertIn('e00:CONTEXT_PROVENANCE_UNVERIFIED',validate(m,g,i,p)['errors'])
    def test_reject_overlapping_conditions(self):
        m,g,i,p=rows();m[0]['irrelevant_addition_span_ids']=m[0]['informative_addition_span_ids']
        self.assertIn('e00:SOURCE_SPAN_CONDITION_OVERLAP',validate(m,g,i,p)['errors'])
    def test_name_compliance(self):
        g={'actor':'Ava','patient':'Ben'}
        self.assertEqual(score_response('Ava',g,'actor'),(True,True))
        self.assertEqual(score_response('Ava; Ben',g,'actor'),(False,False))
        self.assertEqual(score_response('Ben',g,'actor'),(False,True))
        self.assertEqual(score_response('Claire',g,'actor'),(False,True))

if __name__=='__main__': unittest.main()
