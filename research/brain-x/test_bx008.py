import unittest
from bx008_nonredundant_pairs import structural_sentry_signatures, signature, gold_order, match_one_wh, audit
from bx007_qasrl_contrast import eligible_pairs

def entry(sid,wh,qa,qb,qa_start,qb_start,source='wikipedia.dev'):
    return {'source':source,'sid':sid,'vi':1,'wh':wh,'qid':sid+'_'+wh+'_'+str(qa_start),
            'questions':(qa,qb),'ranges':((qa_start,qa_start+1),(qb_start,qb_start+1)),
            'answers':('A','B'),'same_worker':True,'same_assignment':True}

def slotrow(q,rg,slot):
    r={'qasrl_id':'s1','verb_idx':'1','question':q,'answer_range':rg,'wh':'who',
       'is_passive':'False','is_negated':'False','prep':'','obj2':'',
       'subj':'','obj':'','worker_id':'w1'}
    if slot=='subj':r['obj']='someone'
    else:r['subj']='someone'
    return r

class TestBX008(unittest.TestCase):
    def test_sentry_same_original_exact_pair(self):
        x,y=slotrow('Who traveled?', '0:1','subj'), slotrow('Who did someone see?', '2:3','obj')
        sig=structural_sentry_signatures([x,y]);self.assertEqual(len(sig),1)
        p={'sid':'s1','vi':1,'questions':('Who traveled?','Who did someone see?'),
           'ranges':((0,1),(2,3))}
        self.assertIn(signature(p),sig)
    def test_reject_passive_slot(self):
        a,b=slotrow('Who traveled?','0:1','subj'),slotrow('Who did someone see?','2:3','obj')
        a['is_passive']='True';self.assertFalse(structural_sentry_signatures([a,b]))
    def test_reject_worker_dupes(self):
        a,b=slotrow('Who traveled?','0:1','subj'),slotrow('Who did someone see?','2:3','obj')
        self.assertFalse(structural_sentry_signatures([a,b,b.copy()]))
    def test_early_late_canonical(self):
        self.assertEqual(gold_order(entry('s','who','A?','B?',1,5)),'early')
        self.assertEqual(gold_order(entry('s','who','A?','B?',5,1)),'late')
    def test_reject_equal_position(self):
        with self.assertRaises(ValueError):gold_order(entry('s','who','A?','B?',1,1))
    def test_balanced_matching_nominal(self):
        items=[entry('e'+str(i),'who','A?','B?',1,3) for i in range(10)]+[entry('l'+str(i),'who','A?','B?',3,1) for i in range(10)]
        out,state=match_one_wh(items,per_side=10)
        self.assertEqual(state['status'],'READY_DEVELOPMENT_SELECTION')
        self.assertEqual(len(out),20)
        self.assertEqual((state['early'],state['late']),(10,10))
    def test_shared_sentence_cross_side_reassignment(self):
        items=[entry('shared','who','A?','B?',1,3),entry('shared','who','A?','B?',3,1),
               entry('exclusive','who','A?','B?',1,3)]
        out,state=match_one_wh(items,per_side=1)
        self.assertEqual(state['status'],'READY_DEVELOPMENT_SELECTION')
        self.assertEqual(set(x['sid'] for x in out),{'shared','exclusive'})
    def test_quota_shortage_fails_closed(self):
        items=[entry('only','who','A?','B?',1,3)]
        out,state=match_one_wh(items,per_side=1)
        self.assertEqual(state['status'],'BLOCKED_INSUFFICIENT_POSITION_BALANCE')
        self.assertEqual(out,[])
    def test_exclude_used_sentence_cross_wh(self):
        items=[entry('reuse','what','A?','B?',1,3),entry('fresh','what','A?','B?',1,3),
               entry('later','what','A?','B?',3,1)]
        out,state=match_one_wh(items,per_side=1,reserved={'reuse'})
        self.assertEqual(state['status'],'READY_DEVELOPMENT_SELECTION')
        self.assertNotIn('reuse',[x['sid'] for x in out])
    def test_invalid_quota(self):
        with self.assertRaises(ValueError):match_one_wh([],per_side=0)
    def test_audit_fails_for_small_source(self):
        gold=[slotrow('Who traveled?','0:1','subj'),slotrow('Who did someone see?','2:3','obj')]
        sentences=[{'qasrl_id':'s1','tokens':'Alice traveled Bob'}]
        result=audit({'wikinews.dev':(gold,sentences)},per_side=1)
        self.assertEqual(result['status'],'BLOCKED')
    def test_no_test_partition(self):
        with self.assertRaises(ValueError):audit({'wikinews.test':([],[])})

if __name__=='__main__':unittest.main()
