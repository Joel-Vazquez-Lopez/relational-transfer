import unittest
from provenance_certificate import Case, first_fallback, other_event, immediate_predecessor, conditional_provenance, two_event_table, three_event_table, audit

class ProvenanceCertificateTests(unittest.TestCase):
    def test_two_event_wrong_answer_is_other_event_under_closed_candidates(self):
        for t in (0,1):
            labels=('alpha','beta')
            out=labels[1-t]
            self.assertEqual(conditional_provenance(labels,t,out), f'OTHER_EVENT_{2-t}')
    def test_two_event_second_target_confounded_in_both_orders(self):
        for row in two_event_table():
            if row['target']==row['order'][1]:
                self.assertEqual(row['first_fallback'],row['other_event'])
    def test_two_event_first_target_separates_only_restricted_policies(self):
        for row in two_event_table():
            if row['target']==row['order'][0]:
                self.assertNotEqual(row['first_fallback'],row['other_event'])
    def test_three_event_last_target_first_vs_previous_separable(self):
        rows=three_event_table()
        self.assertEqual(sum(x['rivals_disagree'] for x in rows),6)
        self.assertTrue(all(x['rivals_disagree'] for x in rows if x['target_index']==2))
    def test_out_of_candidate_is_not_other_event(self):
        self.assertEqual(conditional_provenance(('Alice','Bob'),1,'Carol'),'OUTSIDE_ROLE_CANDIDATES')
    def test_repeat_surface_form_rejected(self):
        with self.assertRaises(AssertionError):
            conditional_provenance(('Pat','Pat'),1,'Pat')
    def test_full_certificate(self):
        cert=audit()
        self.assertEqual(cert['two_event_second_target_disagree_count'],0)
        self.assertEqual(cert['three_event_late_target_first_vs_predecessor_disagree_count'],6)

if __name__=='__main__':unittest.main()
