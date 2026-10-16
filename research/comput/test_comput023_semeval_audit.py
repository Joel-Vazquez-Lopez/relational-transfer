import unittest
from comput023_semeval_audit import load_training, audit, base_sentence

S='''1 "Alice drove <e1>cars</e1> to the <e2>market</e2>."
Entity-Destination(e1,e2)
Comment: toy fixture

2 "Alice drove <e1>cars</e1> to the <e2>market</e2>."
Entity-Destination(e2,e1)
Comment: toy fixture - deliberately conflicting same pair

3 "The <e1>cat</e1> followed the <e2>dog</e2>."
Other
Comment:

4 "The <e1>cat</e1> followed the <e2>dog</e2>."
Other
Comment:
'''
class Tests(unittest.TestCase):
    def test_source_schema_and_marks(self):
        rows=load_training(S);self.assertEqual(len(rows),4)
        self.assertEqual(rows[0]['label'],'Entity-Destination(e1,e2)')
    def test_stats_never_use_comment(self):
        o=audit(load_training(S));self.assertEqual(o['n_rows'],4)
        self.assertEqual(o['n_other'],2)
        self.assertEqual(o['identical_entity_marked_input_additional_rows'],2)
    def test_direction_balancing(self):
        o=audit(load_training(S));self.assertEqual(o['max_type_direction_balanced_n'],2)
        self.assertEqual(o['gold_type_oracle_majority_orientation']['accuracy'],.5)
    def test_repeated_bare_different_tags_not_contradiction(self):
        t='''1 "A <e1>girl</e1> plays a <e2>violin</e2>."\nInstrument-Agency(e2,e1)\nComment:\n\n2 "A <e1>girl</e1> plays a violin."\nOther\nComment:'''
        with self.assertRaises(ValueError):load_training(t)
    def test_reject_duplicate_id(self):
        with self.assertRaises(ValueError):load_training(S.replace('2 "','1 "'))
    def test_reject_early_e2_marker(self):
        with self.assertRaises(ValueError):load_training(S.replace('<e1>cars</e1> to the <e2>market</e2>','<e2>cars</e2> to the <e1>market</e1>'))
    def test_source_span_normalization(self):
        self.assertEqual(base_sentence('A <e1>girl</e1> plays her <e2>violin</e2>.'),'a girl plays her violin.')
    def test_no_other_as_direction(self):
        o=audit(load_training(S));self.assertEqual(o['n_undirected_relation_items'],2)
if __name__=='__main__':unittest.main()
