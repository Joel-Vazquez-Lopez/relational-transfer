import unittest
from comput015_v2_shortcut import predict,audit
class V2ShortcutTests(unittest.TestCase):
    def test_active_actor_and_patient(self):
        self.assertEqual(predict('Ari000 nudged Bea000. Cai000 warned Dee000.',"For the event 'nudged', who is the actor?"),'Ari000')
        self.assertEqual(predict('Bea000 nudged Cai000. Dee000 warned Ari000.',"For the event 'nudged', who is the patient?"),'Cai000')
    def test_passive_and_clause_order(self):
        self.assertEqual(predict('Dee000 was nudged by Cai000. Bea000 was warned by Ari000.',"For the event 'nudged', who is the actor?"),'Cai000')
        self.assertEqual(predict('Ari000 warned Dee000. Bea000 nudged Cai000.',"For the event 'nudged', who is the actor?"),'Bea000')
    def test_role_changing_donor_failure_retained(self):
        q="For the event 'nudged', who is the actor?"
        self.assertEqual(predict('Ari000 nudged Bea000. Cai000 warned Dee000.',q),'Ari000')
        self.assertEqual(predict('Bea000 nudged Ari000. Cai000 warned Dee000.',q),'Ari000')
    def test_fail_closed(self):
        with self.assertRaises(ValueError):predict('Alice hugged Bob.',"Who did the action?")
        with self.assertRaises(ValueError):predict('Ari000 nudged Bea000.',"For the event 'nudged', who is the actor?")
    def test_fixture_score(self):
        b='Ari000 nudged Bea000. Cai000 warned Dee000.'
        r={'pair_id':'t','family_id':'F000','condition':'target_role_swap','split':'open_development',
           'base_context':b,'donor_context':'Bea000 nudged Ari000. Cai000 warned Dee000.',
           'query':"For the event 'nudged', who is the actor?",'base_gold':'Ari000','donor_gold':'Bea000'}
        counts=audit([r])['by_condition']['target_role_swap']
        self.assertEqual((counts['base_correct'],counts['donor_correct']),(1,0))
if __name__=='__main__':unittest.main()
