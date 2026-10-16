"""Unit tests for independent frozen output rescoring. Run: python3 -m unittest discover -s research/coordinator_audit -p 'test_*.py' -v"""
import unittest
from independent_adapter_rescore import score_one

def item(form="active"):
    return {"actor":"Alice","patient":"Bob","sentence":"Alice praised Bob.","form":form}

def pred(actor="Alice", patient="Bob", extracted=("Alice","Bob"), valid=True):
    return {
        "archived":{"parsed":{"name":actor,"semantic_format_valid":True}},
        "direct_actor":{"parsed":{"name":actor,"semantic_format_valid":True}},
        "direct_patient":{"parsed":{"name":patient,"semantic_format_valid":valid}},
        "extraction":{"parsed":{"actor":extracted[0],"patient":extracted[1],
                               "semantic_format_valid":True,"exact_requested_format":False}},
    }

class TestIndependentAudit(unittest.TestCase):
    def test_semantically_correct_nonexact_extraction(self):
        v=score_one(item(),pred())
        self.assertEqual(v["both_direct"],1)
        self.assertEqual(v["extraction_both"],1)
        self.assertEqual(v["extraction_exact"],0)
        self.assertEqual(v["extraction_last"],0)
    def test_last_name_shortcut_can_be_wrong(self):
        v=score_one(item(),pred(extracted=("Bob","Alice")))
        self.assertEqual(v["extraction_both"],0)
        self.assertEqual(v["extraction_last"],1)
    def test_format_invalid_not_scored_correct(self):
        v=score_one(item(),pred(valid=False))
        self.assertEqual(v["patient_direct"],0)
        self.assertEqual(v["both_direct"],0)
    def test_duplicate_participant_surface_rejected(self):
        with self.assertRaises(ValueError):
            score_one({"actor":"Alice","patient":"Bob","sentence":"Alice praised Bob and Alice."},pred())

if __name__=="__main__":
    unittest.main()
