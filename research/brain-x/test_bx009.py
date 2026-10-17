import copy
import hashlib
import unittest
from bx009_source_provenance import analyze,classify,normalize
H='a'*64
J='b'*64
K='c'*64

def fixture(n=8):
    m=[];p=[]
    for i in range(n):
        d={'case_id':f'item-{i}', 'document_id':f'doc-{i//2}', 'source_kind':'unverified',
           'source_review':'owner_only','rights_review':'pending','source_prompt_invariance_review':'owner_asserted',
           'target_event':'e2' if i%2 else 'e1', 'target_role':'agent', 'target_order':2 if i%2 else 1,
           'roles':{'e1':{'agent':f'A{i}','patient':f'B{i}'},'e2':{'agent':f'C{i}','patient':f'D{i}'}},
           'bystanders':[f'Z{i}'], 'question_sha256':H,'prompt_sha256':{'bare':J,'bystander':K}}
        m.append(d)
        target='C'+str(i) if i%2 else 'A'+str(i)
        competitor='A'+str(i) if i%2 else 'C'+str(i)
        for arm in ('bare','bystander'):
            response=target if arm=='bare' or i!=1 else competitor
            p.append({'case_id':f'item-{i}','arm':arm,'model_revision':'mock-model-not-real',
                      'response':response,'prompt_sha256':J if arm=='bare' else K,'question_sha256':H})
    return m,p

class BX009Tests(unittest.TestCase):
    def test_nominal_mock(self):
        m,p=fixture();r=analyze(m,p,pinned_revision='mock-model-not-real')
        self.assertEqual((r['n_cases'],r['n_document_groups'],r['harm'],r['rescue']), (8,4,1,0))
        self.assertEqual(r['bystander_condition_wrong_event_same_role'],1)
        self.assertFalse(r['meets_owner_reported_wrong_event_threshold'])
        self.assertIn('DESCRIPTIVE_ONLY',r['status'])
    def test_same_role_wrong_event(self):
        m,_=fixture(1);self.assertEqual(classify(m[0],'C0'),'other_event_same_role')
    def test_other_role_wrong_event(self):
        m,_=fixture(1);self.assertEqual(classify(m[0],'D0'),'other_event_other_role')
    def test_own_other_role(self):
        m,_=fixture(1);self.assertEqual(classify(m[0],'B0'),'own_event_other_role')
    def test_bystander(self):
        m,_=fixture(1);self.assertEqual(classify(m[0],'Z0'),'bystander')
    def test_correct(self):
        m,_=fixture(1);self.assertEqual(classify(m[0],'A0'),'correct')
    def test_empty(self):
        m,_=fixture(1);self.assertEqual(classify(m[0],' '),'empty')
    def test_out_of_set(self):
        m,_=fixture(1);self.assertEqual(classify(m[0],'Because someone said it was true'),'out_of_set')
    def test_name_collision(self):
        m,_=fixture(1);m[0]['roles']['e2']['agent']='A0';self.assertEqual(classify(m[0],'A0'),'ambiguous_candidate')
    def test_hash_mismatch(self):
        m,p=fixture();p[0]['question_sha256']='d'*64
        with self.assertRaisesRegex(ValueError,'hash mismatch'):analyze(m,p,pinned_revision='mock-model-not-real')
    def test_duplicate_prediction(self):
        m,p=fixture();p.append(copy.deepcopy(p[0]))
        with self.assertRaisesRegex(ValueError,'duplicate prediction'):analyze(m,p,pinned_revision='mock-model-not-real')
    def test_missing_one_arm(self):
        m,p=fixture();p.pop()
        with self.assertRaisesRegex(ValueError,'missing predictions'):analyze(m,p,pinned_revision='mock-model-not-real')
    def test_mixed_revisions(self):
        m,p=fixture();p[0]['model_revision']='something-else'
        with self.assertRaisesRegex(ValueError,'mixed/unpinned'):analyze(m,p,pinned_revision='mock-model-not-real')
    def test_duplicate_manifest_id(self):
        m,p=fixture();m.append(copy.deepcopy(m[0]))
        with self.assertRaisesRegex(ValueError,'duplicate manifest'):analyze(m,p,pinned_revision='mock-model-not-real')
    def test_strict_source_status(self):
        m,p=fixture();m[0]['source_review']='invented_review'
        with self.assertRaisesRegex(ValueError,'review status'):analyze(m,p,pinned_revision='mock-model-not-real')
    def test_no_raw_names_in_output(self):
        m,p=fixture();r=analyze(m,p,pinned_revision='mock-model-not-real');self.assertNotIn('A0',str(r));self.assertNotIn('item-0',str(r))
    def test_unreviewed_names_counted(self):
        m,p=fixture();self.assertEqual(analyze(m,p,pinned_revision='mock-model-not-real')['source_rights_gold_or_prompt_review_not_independently_certified_cases'],8)
    def test_paired_recovery(self):
        m,p=fixture();p[0]['response']='C0';p[1]['response']='A0';r=analyze(m,p,pinned_revision='mock-model-not-real')
        self.assertEqual((r['rescue'],r['harm'],r['net_correct_gain']),(1,1,0))
    def test_normalization_does_not_fuzzy_change_name(self):
        self.assertEqual(normalize('  ALICE   Q.  '),'alice q.')
        m,_=fixture(1);self.assertEqual(classify(m[0],'A0.'),'out_of_set')

if __name__=='__main__':unittest.main()
