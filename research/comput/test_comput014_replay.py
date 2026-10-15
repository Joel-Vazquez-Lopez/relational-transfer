import unittest
from comput014_replay import template,classified
class TestComput014(unittest.TestCase):
    def test_closed_grammar(self):
        self.assertEqual(template("Alice praised Bob."),("Alice","Bob"))
        self.assertEqual(template("Bob was criticized by Alice."),("Alice","Bob"))
        self.assertEqual(template("It was Bob whom Alice praised."),("Alice","Bob"))
        self.assertIsNone(template("As for Bob, Alice praised Bob."))
    def entry(self,a,p,e):
        def direct(n):
            return {"clean_output":n,"parsed":{"name":n,"semantic_format_valid":True}}
        return {"id":"toy","direct_actor":direct(a),"direct_patient":direct(p),
                "extraction":{"clean_output":f"Actor: {e[0]}; Patient: {e[1]}",
                    "parsed":{"actor":e[0],"patient":e[1],"semantic_format_valid":True}}}
    def item(self):
        return {"actor":"Alice","patient":"Bob","sentence":"Alice praised Bob."}
    def test_collapse_complementary_reversal(self):
        c=classified(self.entry("Alice","Alice",("Bob","Alice")),self.item())
        self.assertEqual((c["qa_same"],c["ex_complement"],c["ex_reversed"]),(1,1,1))
    def test_duplicate_extraction_not_complement(self):
        c=classified(self.entry("Alice","Bob",("Alice","Alice")),self.item())
        self.assertEqual((c["ex_valid"],c["ex_duplicate"],c["ex_complement"]),(1,1,0))
    def test_does_not_consume_gold_in_template(self):
        self.assertEqual(template(self.item()["sentence"]),("Alice","Bob"))
        self.assertEqual(template(self.item()["sentence"]),template(self.item()["sentence"]))
    def test_parser_disagreement_raises(self):
        r=self.entry("Alice","Bob",("Alice","Bob"))
        r["direct_actor"]["parsed"]["name"]="Bob"
        with self.assertRaises(ValueError):classified(r,self.item())
if __name__=="__main__":unittest.main()
