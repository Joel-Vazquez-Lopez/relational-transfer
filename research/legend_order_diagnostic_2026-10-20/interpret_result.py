"""Editorial interpretation after the frozen analysis, with no further inference."""
import json
from core import HERE,save
def main():
    r=json.loads((HERE/'RESULTS.json').read_text())
    assert r['candidate']=='negative_category' and r['counts']['negative_category']['matches']==32
    decision={'decision':'NARROW','supported_operational_account':'negative-category-linked response default','matches':32,'cases':32,'last_option_matches':16,'correct_relational_answers':16,'complete_four_answer_units':0,'units':8,'causal_task_ready':False,'scope':'One checkpoint, two fresh authored active-sentence/question families and A/B codes. The pilot discriminates category-linked output from a fixed last-option policy. It does not discriminate semantic negative preference from syntax-aware copying of the symbol associated with the words not entailed.','example_surviving_nonsemantic_algorithm':'Return the symbol assigned to the not-entailed legend key without reading the sentence.','no_new_mechanism_or_novelty_claim':True,'no_further_inference':True}
    save(HERE/'SCIENTIFIC_DECISION.json',decision)
    report=HERE/'REPORT.md';text=report.read_text();heading,rest=text.split('\n',1)
    section='''## Main finding: category-linked output, not a last-position policy

**NARROW:** Qwen emitted the code assigned to **not entailed on32/32 fresh cases**. This held for both sentence families, both legend orders and both A/B mappings. When not-entailed appeared first, it chose that first option on16/16 cases. The pure last-option prediction matched16/32 and failed the frozen qualification criterion. Correct relational answering was16/32, with0/8 complete four-answer units; the assay remains unready for causal interventions.

The model's output tracks the association between a legend key and its assigned code. This does **not** establish that it semantically judged every relation false or lacked relational information. A syntax-aware rule such as `return legend["not entailed"]`, ignoring the sentence, produces exactly the same observed choices as a negative-category prior. Those explanations remain indistinguishable here. The pilot resolves the proposed last-option rival at the behavioral level without identifying an internal mechanism.

The numerical compatibility control reproduced its old full output vector exactly (maximum error0). All33 collected payloads passed independent score, argmax, shape, hash and journal checks. Every wrong answer and full output vector is retained. Earlier failed gates and the larger assay's NO-GO decision remain unchanged.

'''
    report.write_text(heading+'\n\n'+section+rest)
    print(json.dumps(decision,indent=2))
if __name__=='__main__':main()
