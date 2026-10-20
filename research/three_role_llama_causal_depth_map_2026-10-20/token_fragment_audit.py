"""Posthoc exact subtoken-splice description of all unexpected outputs; no inference."""
import json
from pathlib import Path
from transformers import AutoTokenizer
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent
ORDER={'active':[0,1,2],'passive':[1,2,0],'goal_fronted':[2,0,1]}
def main():
    rows={r['id']:r for r in json.loads((HERE/'INPUTS.json').read_text())};lock=json.loads((HERE/'LOCK.json').read_text())
    tok=AutoTokenizer.from_pretrained(str(ROOT/lock['model']['path']),local_files_only=True,trust_remote_code=False)
    answers=[json.loads(s) for s in (HERE/'ANSWERS.jsonl').read_text().splitlines()];out=[]
    for a in answers:
        r,d=rows[a['recipient_id']],rows[a['donor_input_id']];text=a['generated_text'].strip()
        if a['terminated'] and text in set(r['names'])|set(d['names']):continue
        rn=[r['names'][r['assignment'][i]] for i in ORDER[r['form']]];dn=[d['names'][d['assignment'][i]] for i in ORDER[d['form']]]
        matches=[];prefix_matches=[]
        for i,n in enumerate(rn):
            prefix=tok.decode(r['candidate_sequences'][n][0][:-1])
            for j,m in enumerate(dn):
                suffix=tok.decode(d['candidate_sequences'][m][0][-1:]);splice=(prefix+suffix).strip()
                if splice in set(r['names'])|set(d['names']) or not splice:continue
                meta=dict(recipient_name=n,donor_name=m,recipient_prefix=prefix,donor_last_subtoken=suffix,splice=splice,surface_aligned=i==j)
                if text==splice:matches.append(meta)
                elif not a['terminated'] and text.startswith(splice+' '):prefix_matches.append(meta)
        out.append(dict(id=a['id'],block=a['block'],condition=a['condition'],generated_text=a['generated_text'],terminated=a['terminated'],exact_splice_matches=matches,nonterminated_splice_prefixes=prefix_matches))
    result=dict(scope='Posthoc token-fragment provenance compatibility,not a causal explanation or a new success gate. Enumerates every recipient-name prefix excluding its last subtoken concatenated with every donor-name last subtoken. Candidate paths are the already fixed leading-space paths. Exact matches do not prove a literal splicing algorithm;the full-state patch changes contextual features too.',unexpected_outputs=len(out),exact_splice_outputs=sum(bool(r['exact_splice_matches']) for r in out),surface_aligned_exact=sum(any(m['surface_aligned'] for m in r['exact_splice_matches']) for r in out),nonterminated_splice_prefixes=sum(bool(r['nonterminated_splice_prefixes']) for r in out),ledger=out,model_forwards=0)
    (HERE/'TOKEN_FRAGMENT_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='ledger'},indent=2))
if __name__=='__main__':main()
