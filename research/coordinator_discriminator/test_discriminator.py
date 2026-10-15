import unittest
from discriminator import experiment, summarize

class KnownMechanisms(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=experiment()
        cls.s=summarize(cls.rows)

    def test_counts(self):
        self.assertEqual(len(self.rows),24*4*6*4)

    def test_same_unintervened_answers(self):
        for name in ("cached","reconstructed","hybrid"):
            self.assertEqual(self.s[name]["unintervened_accuracy"],1)

    def test_opposed_donors(self):
        self.assertEqual(self.s["cached"]["crossed_labels"],{"cache":576})
        self.assertEqual(self.s["reconstructed"]["crossed_labels"],{"raw":576})
        self.assertEqual(self.s["shortcut"]["crossed_labels"],{"other":576})
        self.assertEqual(self.s["hybrid"]["crossed_labels"],{"cache":288,"raw":288})

    def test_selective_cuts(self):
        for name,expected in (("cached",(576,0)),("reconstructed",(0,576)),
                              ("hybrid",(288,288)),("shortcut",(0,0))):
            observed=(self.s[name]["cache_effect"],self.s[name]["raw_effect"])
            self.assertEqual(observed,expected)

if __name__=="__main__":
    unittest.main()
