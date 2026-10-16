import unittest
from bx007_qasrl_contrast import eligible_pairs, audit_sources, select_panel, pair_scores

class TestBX007(unittest.TestCase):
    def example(self):
        s=[{'qasrl_id':'S1','tokens':'Alice gave Bob a parcel .'}]
        g=[{'qasrl_id':'S1','verb_idx':'1','wh':'who','question':'Who gave a parcel?','answer_range':'0:1','answer':'Alice'},
           {'qasrl_id':'S1','verb_idx':'1','wh':'who','question':'Who received a parcel?','answer_range':'2:3','answer':'Bob'}]
        return g,s
    def test_good_pair(self):
        g,s=self.example(); p,e=eligible_pairs(g,s); self.assertEqual(len(p),1);self.assertEqual(p[0]['wh'],'who')
    def test_reject_answer_overlap(self):
        g,s=self.example();g[1]['answer_range']='0:1';g[1]['answer']='Alice';self.assertEqual(len(eligible_pairs(g,s)[0]),0)
    def test_reject_different_question_wh(self):
        g,s=self.example();g[1]['wh']='what';self.assertEqual(len(eligible_pairs(g,s)[0]),0)
    def test_reject_duplicate_question_worker(self):
        g,s=self.example();g.append(g[1].copy());self.assertEqual(len(eligible_pairs(g,s)[0]),0)
    def test_reject_multi_answer(self):
        g,s=self.example();g[1]['answer']='Bob~!~parcel';g[1]['answer_range']='2:3~!~4:5';self.assertEqual(len(eligible_pairs(g,s)[0]),0)
    def test_reject_corrupt_source_answer(self):
        g,s=self.example();g[1]['answer']='B?b';self.assertEqual(len(eligible_pairs(g,s)[0]),0)
    def test_reject_repeated_surface(self):
        g,s=self.example();s[0]['tokens']='Alice gave Bob to Bob .';g[1]['answer_range']='2:3';self.assertEqual(len(eligible_pairs(g,s)[0]),0)
    def test_question_blind_answer_fails_both(self):
        r=pair_scores('Alice','Alice','Alice','Bob');self.assertTrue(r['same_answer_collapse']);self.assertFalse(r['both_correct']);self.assertEqual(r['correct_count'],1)
    def test_empty_outputs_not_counted_as_collapse(self):
        r=pair_scores('', '', 'Alice','Bob');self.assertFalse(r['same_answer_collapse']);self.assertFalse(r['both_nonempty'])
    def test_correct_both_requires_distinct_answers(self):
        r=pair_scores('Alice','Bob','Alice','Bob');self.assertTrue(r['both_correct']);self.assertFalse(r['same_answer_collapse'])
    def test_test_split_guard(self):
        g,s=self.example()
        with self.assertRaises(ValueError):audit_sources({'wikinews.test':(g,s)})
    def test_small_source_audit_no_text(self):
        g,s=self.example();r=audit_sources({'wikinews.dev':(g,s)},quota=1)
        self.assertEqual(r['selected_groups'],1);self.assertNotIn('Alice',str(r));self.assertEqual(r['eligible_total_pairs'],1)
    def test_fixed_selection_one_per_sentence(self):
        g,s=self.example();extra=[dict(r) for r in g]; extra[1]['question']='Who got the parcel?'
        p,_=eligible_pairs(g+extra[1:],s)
        for x in p:x['source']='wikinews.dev'
        selected,c=select_panel(p,quota_per_source_wh=10)
        self.assertEqual(len(selected),1)

if __name__=='__main__':unittest.main()
