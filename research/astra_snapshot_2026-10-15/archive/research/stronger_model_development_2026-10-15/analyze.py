"""Paired, cluster-resampled exploratory summaries; no sealed evaluation access."""
import collections
import hashlib
import json
import random
import math
import statistics
from pathlib import Path
from design import ROOT, OLD, sentence, sha
from run import jobs, file_sha

CFG=json.loads((ROOT/'PROTOCOL.json').read_text())

def grouped(rows,keys):
    d=collections.defaultdict(list)
    for r in rows:d[tuple(r[k] for k in keys)].append(r)
    return d

def contrast_correct(r):
    return r['actor_margin']*(1 if r['role']=='actor' else -1)>0

def joint_rows(rows,score):
    out=[]
    for key,g in grouped(rows,['id','form','wording','order']).items():
        assert len(g)==2 and {r['role'] for r in g}=={'actor','patient'}
        a=next(r for r in g if r['role']=='actor');p=next(r for r in g if r['role']=='patient')
        r=(a['actor_margin']-p['actor_margin'])/2
        out.append(dict(id=a['id'],pair_index=a['pair_index'],form=a['form'],wording=a['wording'],order=a['order'],
                        joint=int(score(a) and score(p)),both_wrong=int(not score(a) and not score(p)),
                        r=r,reversed=int(r<0),tie=int(r==0)))
    return out

def legacy_pairs(rows):
    out=[]
    for wording,pword in [('actor_performed','patient_verb'),('actor_acted','patient_affected')]:
        g=[dict(r,wording=wording) for r in rows if r['wording'] in [wording,pword]]
        out+=joint_rows(g,lambda r:r['correct'])
    return out

def nrate(rows,key):
    return dict(n=len(rows),count=sum(r[key] for r in rows),rate=sum(r[key] for r in rows)/len(rows))

def summary(rows):
    result={}
    for panel,rr in grouped(rows,['panel']).items():
        panel=panel[0]
        jj=legacy_pairs(rr) if panel=='legacy' else joint_rows(rr,contrast_correct if panel=='strong_list' else lambda r:r['correct'])
        result[panel]={'by_form':{},'by_form_wording':{},'by_form_role_wording':{},'all_form_joint':{}}
        for (form,),g in grouped(rr,['form']).items():
            j=[x for x in jj if x['form']==form]
            result[panel]['by_form'][form]=dict(n=len(g),correct=sum(x['correct'] for x in g),valid=sum(x['valid'] for x in g),
                contrast_correct=sum(contrast_correct(x) for x in g),joint=nrate(j,'joint'),question_contrast_reversed=nrate(j,'reversed'),both_wrong=nrate(j,'both_wrong'))
        for (form,wording),j in grouped(jj,['form','wording']).items():
            result[panel]['by_form_wording'][form+'/'+wording]={k:nrate(j,k) for k in ['joint','reversed','both_wrong']}
            result[panel]['by_form_wording'][form+'/'+wording]['r_distribution']=dict(mean=statistics.mean(x['r'] for x in j),median=statistics.median(x['r'] for x in j),minimum=min(x['r'] for x in j),maximum=max(x['r'] for x in j))
        for key,g in grouped(rr,['form','role','wording']).items():
            result[panel]['by_form_role_wording']['/'.join(key)]=dict(n=len(g),correct=sum(x['correct'] for x in g),valid=sum(x['valid'] for x in g),contrast_correct=sum(contrast_correct(x) for x in g))
        for (wording,order),g in grouped(jj,['wording','order']).items():
            eg=list(grouped(g,['id']).values());assert all(len(x)==4 for x in eg)
            result[panel]['all_form_joint'][wording+'/'+order]=dict(events=len(eg),correct=sum(all(r['joint'] for r in x) for x in eg))
    return result

def ci(v):
    """v: six independent participant-pair cluster means; retain all paired rows."""
    assert len(v)==6
    rng=random.Random(CFG['analysis']['bootstrap_seed'])
    draws=sorted(sum(v[rng.randrange(6)] for _ in range(6))/6 for _ in range(CFG['analysis']['bootstrap_replicates']))
    return dict(estimate=sum(v)/6,cluster_bootstrap_95=[draws[249],draws[9749]],clusters=6,cluster_values=v)

def gaps(j,key):
    out=[]
    for pair in range(6):
        g=[r for r in j if r['pair_index']==pair]
        a=[r[key] for r in g if r['form']=='active'];o=[r[key] for r in g if r['form']=='object_cleft']
        assert len(a)==len(o)>0
        out.append(sum(o)/len(o)-sum(a)/len(a))
    return out

def heuristic(rows):
    out={}
    for (form,),g in grouped(rows,['form']).items():
        scores=collections.defaultdict(lambda:dict(correct=0,agreement=0,error_agreement=0))
        errors=sum(not r['correct'] for r in g)
        for r in g:
            first=r['patient'] if form in ['passive','object_cleft'] else r['actor']
            last=r['actor'] if first==r['patient'] else r['patient']
            subject=r['patient'] if form=='passive' else r['actor']
            other=r['actor'] if subject==r['patient'] else r['patient']
            predictions={'always_first_mention':first,'always_last_mention':last,
                         'first_mention_as_actor':first if r['role']=='actor' else last,
                         'last_mention_as_actor':last if r['role']=='actor' else first,
                         'event_grammatical_subject_as_actor':subject if r['role']=='actor' else other,
                         'event_grammatical_subject_as_patient':other if r['role']=='actor' else subject}
            if r['order'] in ['ab','ba']:
                names=(r['a'],r['b']) if r['order']=='ab' else (r['b'],r['a'])
                predictions.update(always_candidate_first=names[0],always_candidate_second=names[1])
            for name,pred in predictions.items():
                scores[name]['correct']+=pred==r['gold']; scores[name]['agreement']+=pred==r['answer']
                scores[name]['error_agreement']+=(not r['correct'] and pred==r['answer'])
        out[form]=dict(n=len(g),model_errors=errors,heuristics=dict(scores))
    return out

def main():
    allrows={};report={'status':'exploratory, previously examined constructed events; fresh review not evaluated','models':{},'paired_uncertainty':{},'heuristics':{},'validation':{}}
    for name in CFG['model_order']:
        p=ROOT/f'{name}.jsonl'
        rows=[json.loads(l) for l in p.read_text().splitlines()]
        expected={r['key'] for r in jobs(CFG)}
        assert len(rows)==1152 and {r['key'] for r in rows}==expected
        assert all(math.isfinite(r['actor_margin']) for r in rows)
        assert json.loads((ROOT/f'{name}_COMPLETE.json').read_text())['sha256']==sha(p)
        allrows[name]=rows;report['models'][name]=summary(rows)
        report['models'][name]['compute']=dict(prompt_tokens=sum(r['prompt_tokens'] for r in rows),
            candidate_free_generated_tokens=sum(len(r['generated_token_ids']) for r in rows if r['panel']=='strong_free'),
            candidate_free_length_limit_hits=sum(not r['ended_eos'] for r in rows if r['panel']=='strong_free'),
            complete=json.loads((ROOT/f'{name}_COMPLETE.json').read_text()))
        free=[r for r in rows if r['panel']=='strong_free']
        report['heuristics'][name]=heuristic(free)
        for wording in CFG['strong_questions']:
            j=joint_rows([r for r in free if r['wording']==wording],lambda r:r['correct'])
            report['paired_uncertainty'][name+'/'+wording+'/object_minus_active_joint']=ci(gaps(j,'joint'))
            report['paired_uncertainty'][name+'/'+wording+'/object_minus_active_reversal']=ci(gaps(j,'reversed'))
            for form in CFG['forms']:
                for metric in ['joint','reversed']:
                    v=[]
                    for pair in range(6):
                        g=[r[metric] for r in j if r['pair_index']==pair and r['form']==form]
                        v.append(sum(g)/len(g))
                    report['paired_uncertainty'][name+'/'+wording+'/'+form+'/'+metric]=ci(v)
        lst=[r for r in rows if r['panel']=='strong_list']
        orderpairs=list(grouped(lst,['id','form','wording','role']).values())
        assert all(len(g)==2 for g in orderpairs)
        report['models'][name]['candidate_order']=dict(comparisons=len(orderpairs),contrast_choice_flips=sum(contrast_correct(g[0])!=contrast_correct(g[1]) for g in orderpairs),
            next_token_answer_changes=sum(g[0]['answer']!=g[1]['answer'] for g in orderpairs))
        report['models'][name]['candidate_order']['by_form']={form:dict(comparisons=len(g),contrast_choice_flips=sum(contrast_correct(x[0])!=contrast_correct(x[1]) for x in g)) for form,g in grouped_order(orderpairs).items()}
    for wording in CFG['strong_questions']:
        for metric in ['joint','reversed']:
            j1=joint_rows([r for r in allrows['qwen15'] if r['panel']=='strong_free' and r['wording']==wording],lambda r:r['correct'])
            j7=joint_rows([r for r in allrows['qwen7b'] if r['panel']=='strong_free' and r['wording']==wording],lambda r:r['correct'])
            report['paired_uncertainty']['7b_minus_15/'+wording+'/object_active_interaction/'+metric]=ci([a-b for a,b in zip(gaps(j7,metric),gaps(j1,metric))])
    report['individual_transitions']={}
    report['paired_model_transitions']={}
    low={r['key']:r for r in allrows['qwen15']}
    for (panel,form),rr in grouped(allrows['qwen7b'],['panel','form']).items():
        report['paired_model_transitions'][panel+'/'+form]=dict(n=len(rr),
            newly_correct=[r['key'] for r in rr if r['correct'] and not low[r['key']]['correct']],
            newly_wrong=[r['key'] for r in rr if not r['correct'] and low[r['key']]['correct']])
    for name,rows in allrows.items():
        legacy={ (r['id'],r['form'],r['role']):r for r in rows if r['panel']=='legacy' and r['wording'] in ['actor_performed','patient_verb']}
        free=[r for r in rows if r['panel']=='strong_free' and r['wording']=='lexical']
        report['individual_transitions'][name]=dict(note='legacy to strong changes BOTH wording and answer interface and removes candidates; cannot isolate a cause',
            newly_correct=[dict(id=r['id'],form=r['form'],role=r['role'],before=legacy[r['id'],r['form'],r['role']]['answer'],after=r['answer']) for r in free if r['correct'] and not legacy[r['id'],r['form'],r['role']]['correct']],
            newly_wrong=[dict(id=r['id'],form=r['form'],role=r['role'],before=legacy[r['id'],r['form'],r['role']]['answer'],after=r['answer']) for r in free if not r['correct'] and legacy[r['id'],r['form'],r['role']]['correct']])
    report['archived_fp32_comparison']={}
    for name,oldname in [('qwen05','untouched_qwen05'),('qwen15','untouched_qwen15')]:
        old=json.loads((OLD/f'{oldname}.json').read_text())['records']
        old={ (r['id'],r['form'],r['wording']):r for r in old if not r['omitted']}
        rr=[r for r in allrows[name] if r['panel']=='legacy']
        report['archived_fp32_comparison'][name]=dict(note='same examples and prompt construction; backend/precision changed; NOT a scaling comparison',
            n=len(rr),gains=sum(r['correct'] and not old[r['id'],r['form'],r['wording']]['correct'] for r in rr),
            losses=sum(not r['correct'] and old[r['id'],r['form'],r['wording']]['correct'] for r in rr),
            by_form={form:dict(old_correct=sum(x['correct'] for x in old.values() if x['form']==form),new_correct=sum(x['correct'] for x in rr if x['form']==form)) for form in CFG['forms']})
    for rel,h in CFG['old_hashes'].items():assert file_sha(ROOT.parent/rel)==h
    closed=json.loads((ROOT.parent/'FRONTIER_DIAGNOSTICS_CLOSED_2026-10-15.json').read_text())
    for rel,h in closed['sha256'].items():assert file_sha(ROOT.parents[1]/rel)==h
    for panel in ['legacy','strong_free','strong_list']:
        prompt_maps=[{r['key']:(r['prompt_sha256'],r['token_ids_sha256']) for r in rr if r['panel']==panel} for rr in allrows.values()]
        assert prompt_maps[0]==prompt_maps[1]==prompt_maps[2],panel
    report['validation']=dict(old_source_hashes_unchanged=True,source_lock=json.loads((ROOT/'SOURCE_LOCK.json').read_text()),
        closed_manifest_sha256=file_sha(ROOT.parent/'FRONTIER_DIAGNOSTICS_CLOSED_2026-10-15.json'),
        raw_record_count=sum(map(len,allrows.values())),new_review_inference_count=0,closed_files_unchanged=len(closed['sha256']),identical_prompt_text_and_token_ids_across_sizes=True,
        note='No sealed data, no new reviewed-set evaluation, no fine-tuning; closure manifest unchanged. Bootstrap intervals are conditional on only six reused name pairs.')
    (ROOT/'ANALYSIS.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['models','paired_uncertainty','archived_fp32_comparison']},indent=2))

def grouped_order(orderpairs):
    out=collections.defaultdict(list)
    for pair in orderpairs:out[pair[0]['form']].append(pair)
    return out

if __name__=='__main__':main()
