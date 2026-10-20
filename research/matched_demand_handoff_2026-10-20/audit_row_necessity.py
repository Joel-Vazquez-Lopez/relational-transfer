"""Independent pure-Python enumeration of every saved old-panel row lesion."""
import itertools
from core import *

def main():
    report=json.loads((HERE/'ROW_NECESSITY.json').read_text());checks=0
    for model,folder in [('qwen7b','three_role_native_development_2026-10-20'),('llama3b','three_role_llama_native_2026-10-20')]:
        path=ROOT/'research'/folder/'ANSWERS.jsonl'
        assert sha(path)==report['old_panels'][model]['input_sha256'];checks+=1
        rows=[json.loads(l) for l in path.read_text().splitlines()]
        for entry in report['old_panels'][model]['groups']:
            rr=sorted([r for r in rows if r['source']==entry['source']],key=lambda r:ROLES.index(r['query_role']))
            names=sorted(rr[0]['candidate_log_probabilities'])
            for outcome in entry['outcomes']:
                removed=ROLES.index(outcome['removed']);scores=[]
                for p in itertools.permutations(names):
                    score=sum(rr[q]['candidate_log_probabilities'][p[q]] for q in range(3) if q!=removed)
                    scores.append((score,p))
                scores.sort(reverse=True);margin=scores[0][0]-scores[1][0]
                assert abs(margin-outcome['margin'])<1e-10;checks+=1
                correct=margin>1e-8 and all(n==r['gold'] for n,r in zip(scores[0][1],rr))
                assert correct==outcome['correct'];checks+=1
    save(HERE/'ROW_NECESSITY_AUDIT.json',dict(passed=True,checks=checks,model_forwards=0,enumeration='Independent pure-Python ordered candidate permutations; raw saved full-name scores; no NumPy assignment routine.'))
    print(json.dumps(dict(passed=True,checks=checks)))

if __name__=='__main__':main()
