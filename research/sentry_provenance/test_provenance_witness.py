import unittest
from provenance_witness import audit, predictions


def event(id, order, center, overlap):
    return dict(id=id, order=order, center=center, overlap=overlap)


def case(events=None, target="E1", **extra):
    return dict(case="toy", target=target, events=events or
                [event("E0", 0, 0, .4), event("E1", 1, 10, .9)], **extra)


class WitnessTests(unittest.TestCase):
    def test_two_event_later_target_is_totally_aliased(self):
        p=predictions(case())
        self.assertEqual(set(p.values()), {"E0"})
        a=audit([case()])
        self.assertTrue(all(x["disagreements"] == 0 for x in a["pairwise_distinguishability"].values()))

    def test_target_first_means_first_is_correct_not_wrong_other(self):
        p=predictions(case(target="E0"))
        self.assertEqual(p["first_event"], "E0")
        self.assertEqual(p["only_other"], "E1")
        self.assertIsNone(p["previous_event"])

    def test_three_event_middle_target_separates_first_from_nearest(self):
        r=case([event("E0",0,0,.1),event("E1",1,10,.1),event("E2",2,11,.9)])
        p=predictions(r)
        self.assertEqual(p["first_event"], "E0")
        self.assertEqual(p["previous_event"], "E0")
        self.assertEqual(p["nearest_other"], "E2")
        self.assertEqual(p["lexically_closest_other"], "E2")
        self.assertIsNone(p["only_other"])

    def test_four_events_can_yield_three_different_wrong_sources(self):
        r=case([event("E0",0,0,.1),event("E1",1,4,.9),
                event("ET",2,20,.1), event("E3",3,21,.2)],target="ET")
        p=predictions(r)
        self.assertEqual((p["first_event"],p["previous_event"],p["nearest_other"],p["lexically_closest_other"]),
                         ("E0","E1","E3","E1"))

    def test_ambiguous_provenance_cannot_be_mislabeled(self):
        with self.assertRaisesRegex(ValueError, "Cannot assign"):
            predictions(case(observed_source="E0",provenance_status="ambiguous"))

    def test_verified_provenance_count_not_inferred_from_text(self):
        a=audit([case(observed_source="E0",provenance_status="verified")])
        self.assertEqual(a["observed_source_fit_if_independently_verified"]["first_event"]["matching_source"],1)
        self.assertEqual(a["provenance_status_counts"]["verified"],1)

    def test_unverified_has_no_source_fit(self):
        self.assertEqual(audit([case()])["observed_source_fit_if_independently_verified"]["first_event"]["scored_verified_rows"],0)

    def test_ties_abstain_and_do_not_create_false_witness(self):
        p=predictions(case([event("E0",0,0,.8),event("E1",1,10,.1), event("E2",2,20,.8)]))
        self.assertIsNone(p["nearest_other"])
        self.assertIsNone(p["lexically_closest_other"])

    def test_duplicate_event_ids_fail(self):
        with self.assertRaises(ValueError): predictions(case([event("E0",0,0,.4),event("E0",1,10,.8)],target="E0"))

    def test_wrong_gold_like_keys_fail(self):
        bad=case();bad["gold_answer"]="Alice"
        with self.assertRaisesRegex(ValueError,"no source text"):
            predictions(bad)

    def test_duplicate_cases_fail(self):
        with self.assertRaisesRegex(ValueError,"Duplicate case"):
            audit([case(),case()])

    def test_out_of_range_overlap_fails(self):
        with self.assertRaises(ValueError):predictions(case([event("E0",0,0,1.5),event("E1",1,2,.4)]))

    def test_nonconsecutive_rank_fails(self):
        with self.assertRaises(ValueError):predictions(case([event("E0",0,0,.5),event("E1",2,2,.5)]))


if __name__ == "__main__":
    unittest.main()
