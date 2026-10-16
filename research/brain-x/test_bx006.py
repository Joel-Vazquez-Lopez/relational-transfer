import unittest
from bx006_qasrl_gold_audit import audit, parse_reference_spans, exact_span_match

class BX006Tests(unittest.TestCase):
    def test_multiple_alternatives(self):
        row=parse_reference_spans('Amy~!~Lee','0:1~!~2:3','Amy and Lee ran'.split())
        self.assertEqual(row['status'],'EXACT')
        self.assertTrue(exact_span_match('Lee',row,'Amy and Lee ran'.split()))
        self.assertFalse(exact_span_match('and',row,'Amy and Lee ran'.split()))
    def test_zero_based_half_open(self):
        result=parse_reference_spans('and Lee','1:3','Amy and Lee'.split())
        self.assertEqual(result['status'],'EXACT')
        self.assertEqual(result['alternatives'][0]['end'],3)
    def test_count_mismatch(self):
        self.assertEqual(parse_reference_spans('Amy~!~Lee','0:1','Amy and Lee'.split())['status'],'COUNT_MISMATCH')
    def test_bad_range(self):
        self.assertEqual(parse_reference_spans('Amy','banana','Amy Lee'.split())['status'],'BAD_RANGE_FORMAT')
    def test_bad_bounds(self):
        self.assertEqual(parse_reference_spans('Lee','1:9','Amy Lee'.split())['status'],'OUT_OF_BOUNDS')
    def test_corrupt_accent_abstain(self):
        parsed=parse_reference_spans('Garc?s','0:1',['Garcés'])
        self.assertEqual(parsed['status'],'SOURCE_TEXT_DISAGREEMENT')
        self.assertIsNone(exact_span_match('Garcés',parsed,['Garcés']))
    def test_legacy_actor_patient_not_imputed(self):
        report=audit([dict(qasrl_id='s1',verb_idx='1',wh='who',question='Who went?',answer='Amy',answer_range='0:1')],
                     [dict(qasrl_id='s1',tokens='Amy went',sentence='Amy went')],partition='toy.dev')
        self.assertEqual(report['multiple_accepted_span_rows'],0)
        self.assertEqual(report['metric_contract']['gold_conversion'],'NO actor/patient label derivation')
    def test_duplicate_same_question_detected(self):
        g=dict(qasrl_id='s1',verb_idx='1',wh='who',question='Who went?',answer='Amy',answer_range='0:1')
        result=audit([g,g.copy()],[dict(qasrl_id='s1',tokens='Amy went')],partition='toy.dev')
        self.assertEqual(result['other_flags']['DUPLICATE_SENTENCE_VERB_QUESTION_KEYS'],1)
    def test_no_test_split(self):
        with self.assertRaises(ValueError):audit([],[],partition='wikinews.test')
    def test_repeated_surface_not_position_certificate(self):
        toks='Amy saw Amy'.split()
        parsed=parse_reference_spans('Amy','2:3',toks)
        self.assertTrue(exact_span_match('Amy',parsed,toks))
        # Text-only outputs cannot certify WHICH occurrence was selected.
        self.assertNotEqual(parsed['alternatives'][0]['start'],0)

if __name__=='__main__':unittest.main()
