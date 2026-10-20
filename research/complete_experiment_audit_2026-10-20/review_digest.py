"""Print bounded documentary excerpts for explicit human/agent review."""
from pathlib import Path
import json,re,sys
HERE=Path(__file__).resolve().parent
rows=json.loads((HERE/'REVIEW_TEXT.json').read_text())
a,b=map(int,sys.argv[1:3]);budget=int(sys.argv[3]) if len(sys.argv)>3 else 2100
for i,r in enumerate(rows[a:b],a):
 text=r['text'];lines=text.splitlines();selected=[]
 # Retain opening result, numerical tables, and conclusion/limitation paragraphs.
 intro='\n'.join(lines[:12]);selected.append(intro)
 tables='\n'.join(l for l in lines if l.startswith('|'))
 if tables:selected.append(tables[:int(budget*.55)])
 paras=text.split('\n\n');pattern=r'(?i)(conclu|decision|failed|does not|did not|cannot|negative|inconclusive|stop|not establish|not support|confound|gate|limitation|no.go|narrow)'
 candidates=[p for p in paras if re.search(pattern,p) and not p.startswith('SOURCE:') and len(p)<1800]
 tail='\n\n'.join(candidates[-5:]);selected.append(tail)
 excerpt='\n'.join(selected)
 if len(excerpt)>budget:
  first=int(budget*.65);excerpt=excerpt[:first]+'\n...[selected excerpt]...\n'+excerpt[-(budget-first):]
 print('\n###',i,r['key'],'FULL_CHARS',r['chars'],'\n'+excerpt)
