"""Compass 13: incomplete two-swap shortcut baseline.
Run from same directory as exp13_curriculum.py.
No model training or pretrained inference.
"""
from collections import Counter
from exp13_curriculum import PAIRS,dataset,final_state

def heuristic(ops,query,rule):
    if rule=='omit-first':
        return final_state(ops[1:])[query]
    if rule=='omit-middle':
        return final_state([ops[0],ops[2]])[query]
    if rule=='omit-last':
        return final_state(ops[:2])[query]
    if rule=='last-only':
        return final_state(ops[2:])[query]
    if rule=='partial-vote':
        outputs=[final_state(ops[1:])[query],
                 final_state([ops[0],ops[2]])[query],
                 final_state(ops[:2])[query]]
        return Counter(outputs).most_common(1)[0][0]
    raise ValueError(rule)

def audit():
    rows=[r for r in dataset() if r['length']==3]
    results={}
    for rule in ('omit-first','omit-middle','omit-last','last-only','partial-vote'):
        good=sum(heuristic([PAIRS[i] for i in r['slots']],r['q'],rule)==r['y'] for r in rows)
        four=[r for r in rows if r['cycle_type']=='4']
        four_good=sum(heuristic([PAIRS[i] for i in r['slots']],r['q'],rule)==r['y'] for r in four)
        complete=sum(all(heuristic([PAIRS[i] for i in rows[j]['slots']],q,rule)==rows[j+q]['y']
                         for q in range(4))
                     for j in range(0,len(rows),4))
        results[rule]={'correct':good,'total':len(rows),
                       'four_cycle_correct':four_good,'four_cycle_total':len(four),
                       'complete_worlds':complete,'world_total':len(rows)//4}
    assert results['partial-vote']['correct']==504
    assert results['partial-vote']['four_cycle_correct']==264
    assert results['partial-vote']['complete_worlds']==0
    print(results)
    return results

if __name__=='__main__':
    audit()
