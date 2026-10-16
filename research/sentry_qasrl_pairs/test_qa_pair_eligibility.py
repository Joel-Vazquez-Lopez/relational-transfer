import io
import unittest
from qa_pair_eligibility import audit_rows, git_blob_sha, load_dev, role_slot, spans_overlap
from pathlib import Path
from tempfile import TemporaryDirectory


def item(sent='s1', verb='3', gap='subject', span='1:2', **override):
    row = dict(qasrl_id=sent, verb_idx=verb, question='Who did something?', answer_range=span,
               wh='who', subj='' if gap == 'subject' else 'someone',
               obj='something' if gap == 'subject' else '',
               obj2='', prep='', is_passive='False', is_negated='False', worker_id='w1')
    row.update(override)
    return row


class AuditorTests(unittest.TestCase):
    def test_eligible_fixed_predicate_two_original_questions(self):
        out = audit_rows([item(gap='subject', span='0:1'), item(gap='object', span='2:3')], 'wikinews')
        self.assertEqual(out['strict_unambiguous_pairs'], 1)
        self.assertEqual(out['paired_human_question_rows'], 2)
        self.assertEqual(out['distinct_sentence_ids_with_pairs'], 1)
        self.assertEqual(out['subject_gold_span_starts_before_object'], 1)
        self.assertEqual(out['object_gold_span_starts_before_subject'], 0)

    def test_not_across_different_predicates_or_sentences(self):
        out = audit_rows([item(gap='subject'), item(sent='s2', gap='object'),
                          item(sent='s3', gap='subject'),item(sent='s3', verb='4', gap='object')], 'wikipedia')
        self.assertEqual(out['strict_unambiguous_pairs'], 0)

    def test_multispan_abstains(self):
        z = audit_rows([item(gap='subject',span='0:1~!~3:4'),item(gap='object',span='2:3')], 'wikinews')
        self.assertEqual(z['strict_unambiguous_pairs'], 0)
        self.assertEqual(z['filtered_row_reasons']['multiple_answer_spans'], 1)

    def test_active_no_neg_no_prep_no_second_obj(self):
        for field, value in [('is_passive','True'), ('is_negated','True'), ('prep','to'), ('obj2','someone'), ('wh','where')]:
            r = item(**{field:value})
            self.assertIsNone(role_slot(r)[0], field)

    def test_ambiguous_independent_answers_rejected(self):
        rows=[item(gap='subject',span='0:1'),item(gap='subject',span='4:5'),item(gap='object',span='2:3')]
        out=audit_rows(rows,'wikinews')
        self.assertEqual(out['predicates_with_both_gaps'],1)
        self.assertEqual(out['rejected_pairs_multiple_ambiguous_answers'],1)
        self.assertEqual(out['strict_unambiguous_pairs'],0)

    def test_overlapping_roles_rejected(self):
        out=audit_rows([item(gap='subject',span='0:2'),item(gap='object',span='1:3')],'wikipedia')
        self.assertEqual(out['strict_unambiguous_pairs'],0)
        self.assertEqual(out['rejected_pairs_overlapping_spans'],1)
        self.assertTrue(spans_overlap('0:2','1:3'))

    def test_reverse_answer_span_order_survives(self):
        out=audit_rows([item(gap='subject',span='8:9'),item(gap='object',span='2:3')],'wikipedia')
        self.assertEqual(out['strict_unambiguous_pairs'],1)
        self.assertEqual(out['subject_gold_span_starts_before_object'],0)
        self.assertEqual(out['object_gold_span_starts_before_subject'],1)

    def test_mixed_wh_not_semantic_binary(self):
        out=audit_rows([item(gap='subject',span='0:1',wh='who'),item(gap='object',span='2:3',wh='what')],'wikinews')
        self.assertEqual(out['strict_unambiguous_pairs'],1)
        self.assertEqual(out['pair_wh_types']['mixed'],1)

    def test_content_pinning_and_test_file_rejection(self):
        with TemporaryDirectory() as d:
            p=Path(d)/'wikipedia.test.gold.csv';p.write_text('irrelevant')
            with self.assertRaises(ValueError): load_dev(p,'wikipedia')
            p2=Path(d)/'wikipedia.dev.gold.csv';p2.write_text('not official')
            with self.assertRaises(ValueError): load_dev(p2,'wikipedia')
        self.assertEqual(git_blob_sha(b'hello\n'),'ce013625030ba8dba906f756967f9e9ca394464a')

    def test_missing_schema_and_bad_spans(self):
        with self.assertRaises(ValueError): audit_rows([{'qasrl_id':'x'}],'wikinews')
        self.assertIsNone(role_slot(item(span='3:3'))[0])
        self.assertIsNone(role_slot(item(span='wrong'))[0])

if __name__ == '__main__': unittest.main()
