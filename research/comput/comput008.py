#!/usr/bin/env python3
"""COMPUT-008: synthetic counterbalanced shortcut audit; stdlib only."""
import itertools,json,argparse
PAIRS=[("Amy","Ben"),("Clara","David"),("Eva","Frank"),("Grace","Henry"),("Iris","Jack"),("Kate","Leo")]
VERBS=("praised","warned","assisted","thanked")
FORMS=("active","passive","subject_cleft","object_cleft")
RULES=("first","last","first_for_actor","last_for_actor","construction_position")
def sentence(actor,patient,verb,form):
 return {"active":f"{actor} {verb} {patient}.","passive":f"{patient} was {verb} by {actor}.","subject_cleft":f"It was {actor} who {verb} {patient}.","object_cleft":f"It was {patient} whom {actor} {verb}."}[form]
def generate():
 rows=[]
 for pair,(x,y) in enumerate(PAIRS):
  for verb in VERBS:
   for direction,(actor,patient) in enumerate(((x,y),(y,x))):
    for form in FORMS:
     s=sentence(actor,patient,verb,form);order=sorted((x,y),key=s.index)
     for q in ("actor","patient"):
      rows.append({"pair_id":pair,"verb":verb,"direction":direction,"form":form,"question":q,"sentence":s,"first":order[0],"last":order[1],"gold":actor if q=="actor" else patient})
 return rows
def predict(row,rule):
 f,l,q=row["first"],row["last"],row["question"]
 if rule=="first":return f
 if rule=="last":return l
 if rule=="first_for_actor":return f if q=="actor" else l
 if rule=="last_for_actor":return l if q=="actor" else f
 if rule=="construction_position":return predict(row,"first_for_actor" if row["form"] in ("active","subject_cleft") else "last_for_actor")
 raise ValueError(rule)
def audit(rows):
 return {"n":len(rows),"cells":{form:{rule:{"correct":sum(predict(r,rule)==r["gold"] for r in rows if r["form"]==form),"n":sum(r["form"]==form for r in rows)} for rule in RULES} for form in FORMS},
 "note":"All stimuli generated with one fixed surface template per construction; construction_position is an explicitly programmed shortcut, not learned model behavior."}
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--output");args=p.parse_args()
 result=audit(generate());out=json.dumps(result,indent=2)
 if args.output:
  with open(args.output,"w",encoding="utf8") as f:f.write(out+"\n")
 print(out)
