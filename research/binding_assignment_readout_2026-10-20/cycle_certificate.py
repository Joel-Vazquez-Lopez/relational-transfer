"""Independent five-cycle certificate from source text and saved score tables."""
import hashlib,json,re
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent
QUESTIONS=['Who sent someone to another person?','Who was sent to another person?','Who was someone sent to?']

def parse(source):
    if source.startswith('To '):
        m=re.fullmatch(r'To (\w+), (\w+) sent (\w+)\.',source);assert m;return [m[2],m[3],m[1]]
    if ' was sent ' in source:
        m=re.fullmatch(r'(\w+) was sent to (\w+) by (\w+)\.',source);assert m;return [m[3],m[1],m[2]]
    m=re.fullmatch(r'(\w+) sent (\w+) to (\w+)\.',source);assert m;return list(m.groups())

def main():
    ledger=[];checks=0
    for model,dirname in [('qwen7b','three_role_native_development_2026-10-20'),('llama3b','three_role_llama_native_2026-10-20')]:
        p=ROOT/'research'/dirname;raw=[json.loads(s) for s in (p/'ANSWERS.jsonl').read_text().splitlines()]
        published=json.loads((HERE/f'{model}_RESULTS.json').read_text());by_source={r['source']:r for r in published['unit_outcomes']}
        groups={}
        for r in raw:groups.setdefault(r['source'],{})[r['question']]=r
        assert len(groups)==36;checks+=1
        for source,d in sorted(groups.items()):
            role_names=parse(source);assert len(set(role_names))==3 and set(d)==set(QUESTIONS)
            t=np.array([[d[q]['candidate_log_probabilities'][n] for n in role_names] for q in QUESTIONS])
            pair=[float(t[i,i]+t[j,j]-t[i,j]-t[j,i]) for i,j in [(0,1),(0,2),(1,2)]]
            diag=float(np.trace(t));cycles=[diag-float(t[0,1]+t[1,2]+t[2,0]),diag-float(t[0,2]+t[1,0]+t[2,1])]
            margins=pair+cycles;assert min(margins)>1e-8
            assert abs(min(margins)-by_source[source]['gold_assignment_margin'])<1e-8
            assert by_source[source]['structured_correct'];checks+=3
            # No greedy/permutation decoder is called here. The five nonidentity permutation cycles exhaust three-role rivals.
            ledger.append(dict(model=model,source=source,independently_parsed_role_names=role_names,two_role_swap_margins=pair,three_role_cycle_margins=cycles,minimum_margin=min(margins)))
        for f,h in published['source_files'].items():assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h;checks+=1
    out=dict(passed=True,n_checks=checks,sources=72,scope='Source-text parsing and five algebraic rival margins independently verify the complete-assignment claim from already native-audited probability inputs. Not new inference, independent data or a significance test.',ledger=ledger)
    (HERE/'CYCLE_CERTIFICATE.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='ledger'},indent=2))
if __name__=='__main__':main()
