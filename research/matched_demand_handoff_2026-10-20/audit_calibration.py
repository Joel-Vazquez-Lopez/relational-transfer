"""Independent saved-score calibration, train/test and assignment checks."""
import itertools
import re
from core import *

def main():
    result=json.loads((HERE/'CALIBRATION_RIVALS.json').read_text())
    checks=0;details={}
    def check(ok,label):
        nonlocal checks
        assert ok,label;checks+=1
    for model,analysis in result['models'].items():
        path=ROOT/analysis['input'];check(sha(path)==analysis['input_sha256'],'Input hash')
        rows=[json.loads(l) for l in path.read_text().splitlines()]
        groups={}
        for r in rows:groups.setdefault(r['source'],[]).append(r)
        summary={}
        for method,data in analysis['methods'].items():
            comparisons=[]
            for e in data['entries']:
                rr=sorted(groups[e['source']],key=lambda r:ROLES.index(r['query_role']))
                names=sorted(rr[0]['candidate_log_probabilities'])
                raw=np.array([[r['candidate_log_probabilities'][n] for n in names] for r in rr])
                if method=='raw_independent':fixed=raw
                elif method=='within_source_three_question_column_centering':fixed=raw-np.mean(raw,axis=0)
                else:
                    train=next(t for t in analysis['training'] if t['source']==e['source'])
                    if method=='other_form_lexical_prior':
                        sources=train['lexical_training_sources']
                        check(all(groups[s][0]['family']==e['family'] and groups[s][0]['form']!=e['form'] for s in sources),'Lexical split')
                        expected=np.mean([[r['candidate_log_probabilities'][n] for n in names] for s in sources for r in groups[s]],axis=0)
                        check(np.max(abs(expected-np.array(train['lexical_bias'])))<1e-12,'Lexical prior recomputation')
                        fixed=raw-expected
                    else:
                        sources=train['surface_training_sources']
                        check(all(groups[s][0]['family']!=e['family'] and groups[s][0]['form']==e['form'] for s in sources),'Surface family split')
                        samples=[]
                        for s in sources:
                            names_s=sorted(groups[s][0]['candidate_log_probabilities'])
                            mentions=[m.group() for m in re.finditer(r'\b(?:'+'|'.join(names_s)+r')\b',s)]
                            for row in groups[s]:samples.append([row['candidate_log_probabilities'][n] for n in mentions])
                        expected=np.mean(samples,axis=0)
                        check(np.max(abs(expected-np.array(train['surface_bias'])))<1e-12,'Surface prior recomputation')
                        mentions=[m.group() for m in re.finditer(r'\b(?:'+'|'.join(names)+r')\b',e['source'])]
                        fixed=raw-np.array([expected[mentions.index(n)] for n in names])
                winners=np.argmax(fixed,axis=1)
                check([names[i] for i in winners]==e['answers'],'Independent predictions')
                margins=[fixed[q,w]-max(fixed[q,j] for j in range(3) if j!=w) for q,w in enumerate(winners)]
                correct=[bool(names[w]==r['gold'] and m>1e-8) for w,m,r in zip(winners,margins,rr)]
                check(correct==e['correct'],'Independent correctness')
                comparisons.append(dict(family=e['family'],form=e['form'],native_complete=all(r['native_correct'] for r in rr),new_complete=all(correct),native_correct=sum(r['native_correct'] for r in rr),new_correct=sum(correct),rescued_answers=sum(not r['native_correct'] and c for r,c in zip(rr,correct)),damaged_answers=sum(r['native_correct'] and not c for r,c in zip(rr,correct))))
                perms=list(itertools.permutations(range(3)))
                a=np.array([sum(raw[q,p[q]] for q in range(3)) for p in perms])
                b=np.array([sum(fixed[q,p[q]] for q in range(3)) for p in perms])
                check(np.max(abs((a-a[0])-(b-b[0])))<1e-10,'All assignment gaps invariant to column calibration')
            summary[method]=dict(rescued_answers=sum(x['rescued_answers'] for x in comparisons),damaged_answers=sum(x['damaged_answers'] for x in comparisons),by_family_form=[dict(family=f,form=form,answers=sum(x['new_correct'] for x in comparisons if x['family']==f and x['form']==form),complete_sources=sum(x['new_complete'] for x in comparisons if x['family']==f and x['form']==form)) for f in ['t01','t02'] for form in ['active','passive','goal_fronted']])
        details[model]=summary
    save(HERE/'CALIBRATION_AUDIT.json',dict(passed=True,checks=checks,model_forwards=0,details=details,limitations='All underlying outcomes exposed before this exploratory analysis. Column correction preserves assignment gaps algebraically; this is not independent confirmation of assignment decoding or a causal attribution of neural error.'))
    print(json.dumps(dict(passed=True,checks=checks,details=details),indent=2))

if __name__=='__main__':main()
