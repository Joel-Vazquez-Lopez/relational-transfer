"""COMPUT-027: source-free regression tests for the archived-output audit."""
import unittest
from comput027_frozen_factorial_audit import parse_surface_roles,audit_one,audit_pair

def toy(model='qwen',fail=False):
    family={'id':'a_b_thanked','a':'Ada','b':'Bo','verb':'thanked'}
    rows=[]
    for frame in ('canonical','agent_first_passive_cleft'):
        for v in (0,1):
            sentence=('Ada thanked Bo.' if v==0 else
                      'Bo was thanked by Ada.' if frame=='canonical'
                      else 'It was by Ada that Bo was thanked.')
            for role,answer in [('agent','Ada'),('patient','Bo')]:
                guessed='Ada' if fail and role=='patient' and sentence!='Bo was thanked by Ada.' else answer
                rows.append(dict(prompt=f'{model}:{sentence}:{role}',sentence=sentence,
                    generated=guessed,expected=answer,rival='Bo' if answer=='Ada' else 'Ada',
                    score_margin=1 if guessed==answer else -1,greedy_correct=guessed==answer,
                    candidate_token_ids={'Ada':[1],'Bo':[2]},role=role,frame=frame,
                    r=0,v=v))
    return {'families':[{'family':family,'rows':rows}]}

class Tests(unittest.TestCase):
    def test_active_parser(self):
        self.assertEqual(parse_surface_roles('Ada thanked Bo.')['patient'],'Bo')
    def test_passive_parser(self):
        self.assertEqual(parse_surface_roles('Bo was thanked by Ada.')['agent'],'Ada')
    def test_cleft_parser(self):
        self.assertEqual(parse_surface_roles('It was by Ada that Bo was thanked.')['patient'],'Bo')
    def test_reject_non_template(self):
        with self.assertRaises(ValueError):
            parse_surface_roles('Ada maybe thanked Bo.')
    def test_repeated_active_collapses(self):
        a=audit_one(toy())
        self.assertEqual((a['n_recorded'],a['distinct']['n'],a['n_duplicate_active_rows']),(8,6,2))
    def test_first_mention_bias(self):
        a=audit_one(toy(fail=True))
        self.assertEqual(a['gold_first']['wrong'],0)
        self.assertEqual(a['gold_second']['wrong'],2)
        self.assertEqual(a['errors_choose_first'],2)
    def test_corrupted_duplicate_rejected(self):
        z=toy();r=z['families'][0]['rows'][4]
        r['generated']='Bo';r['greedy_correct']=False;r['score_margin']=-1
        with self.assertRaises(ValueError):audit_one(z)
    def test_corrupt_gold_rejected(self):
        z=toy();z['families'][0]['rows'][0]['expected']='Bo'
        with self.assertRaises(ValueError):audit_one(z)
    def test_other_role_errors(self):
        a=audit_one(toy(fail=True))
        self.assertEqual(a['errors_choose_other_role'],2)
        self.assertEqual(a['template_parser_correct'],6)
    def test_cross_model_semantic_conditions(self):
        r=audit_pair(toy('qwen'),toy('llama',fail=True))
        self.assertEqual(r['matched_semantic_cells'],6)
        self.assertEqual(r['agreement']['llama_only_wrong'],2)
    def test_aggregate_output_contains_no_source_prompts(self):
        r=audit_pair(toy('qwen'),toy('llama',fail=True))
        self.assertNotIn('prompt',str(r))
if __name__=='__main__':
    unittest.main()
