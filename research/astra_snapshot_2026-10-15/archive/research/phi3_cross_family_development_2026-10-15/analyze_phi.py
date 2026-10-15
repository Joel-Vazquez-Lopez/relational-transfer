"""Exploratory cross-family paired analysis; reads only explicitly named old outputs."""
import collections,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
QWEN=ROOT.parent/'stronger_model_development_2026-10-15'
sys.path.insert(0,str(QWEN))
from analyze import summary,joint_rows,legacy_pairs,ci,grouped,contrast_correct,gaps
from run import file_sha

def units(rows,score):
    return legacy_pairs(rows) if rows[0]['panel']=='legacy' else joint_rows(rows,score)

def heuristic(rows):
    report={}
    for (form,),g in grouped(rows,['form']).items():
        scores=collections.defaultdict(lambda:dict(agreement=0,correct=0,error_agreement=0))
        for r in g:
            first=r['patient'] if form in ['passive','object_cleft'] else r['actor']
            last=r['actor'] if first==r['patient'] else r['patient']
            subj=r['patient'] if form=='passive' else r['actor'];other=r['actor'] if subj==r['patient'] else r['patient']
            preds=dict(always_actor=r['actor'],always_patient=r['patient'],always_event_subject=subj,
                always_first=first,always_last=last,first_as_actor=first if r['role']=='actor' else last,
                last_as_actor=last if r['role']=='actor' else first,
                event_subject_as_actor=subj if r['role']=='actor' else other,
                event_subject_as_patient=other if r['role']=='actor' else subj)
            for n,v in preds.items():
                scores[n]['agreement']+=v==r['answer'];scores[n]['correct']+=v==r['gold']
                scores[n]['error_agreement']+=not r['correct'] and v==r['answer']
        report[form]=dict(n=len(g),errors=sum(not r['correct'] for r in g),invalid=sum(not r['valid'] for r in g),heuristics=dict(scores))
    return report

def wording(rows):
    j=joint_rows(rows,lambda r:r['correct']);by={}
    for (form,),g in grouped(j,['form']).items():
        a={x['id']:x for x in g if x['wording']=='lexical'};b={x['id']:x for x in g if x['wording']=='explicit'}
        gains=[k for k in a if b[k]['joint'] and not a[k]['joint']];losses=[k for k in a if a[k]['joint'] and not b[k]['joint']]
        v=[]
        for pair in range(6):
            ids=[k for k in a if a[k]['pair_index']==pair]
            v.append(sum(b[k]['joint']-a[k]['joint'] for k in ids)/len(ids))
        by[form]=dict(newly_correct=gains,newly_incorrect=losses,uncertainty=ci(v))
    return by

def main():
    cfg=json.loads((ROOT/'PROTOCOL.json').read_text())
    for f,h in json.loads((ROOT/'SOURCE_LOCK.json').read_text()).items():assert file_sha(ROOT/f)==h
    for f,h in cfg['qwen_checksums'].items():
        if f in {'NEW_REVIEW.csv','NEW_PROPOSED_LABELS.json','REVIEW_INSTRUCTIONS.txt'}:continue
        assert file_sha(QWEN/f)==h, f
    closed=json.loads((ROOT.parent/'FRONTIER_DIAGNOSTICS_CLOSED_2026-10-15.json').read_text())
    for rel,h in closed['sha256'].items():assert file_sha(ROOT.parents[1]/rel)==h
    raw=ROOT/'phi3.jsonl';rows=[json.loads(l) for l in raw.read_text().splitlines()]
    assert len(rows)==1728 and len({r['key'] for r in rows})==1728
    assert file_sha(raw)==json.loads((ROOT/'COMPLETE.json').read_text())['sha256']
    assert all(math.isfinite(x['actor_margin']) for x in rows)
    # AB legacy subset exactly matches the old semantic panel. Qwen generation on legacy/list is one-token only.
    matched=[r for r in rows if r['panel']!='legacy' or r['order']=='ab']
    assert len(matched)==1152
    models={'phi3':matched}
    for n in ['qwen05','qwen15','qwen7b']:models[n]=[json.loads(l) for l in (QWEN/f'{n}.jsonl').read_text().splitlines()]
    assert all({r['key'] for r in rr}=={r['key'] for r in matched} for rr in models.values())
    report=dict(status='exploratory reused constructed data; no new/sealed data evaluated',
        phi_summary=summary(matched),models={},wording_transitions={},heuristics={},uncertainty={},
        candidate_order={},generation_controls={},paired_model_transitions={})
    for n,rr in models.items():
        free=[r for r in rr if r['panel']=='strong_free'];j=joint_rows(free,lambda r:r['correct'])
        pairs=grouped(free,['id','form','wording','order'])
        for x in j:
            pair=pairs[(x['id'],x['form'],x['wording'],x['order'])]
            x['both_valid']=all(r['valid'] for r in pair)
            x['contrast_joint']=all(contrast_correct(r) for r in pair)
        report['models'][n]=summary(rr)
        legacy=legacy_pairs([dict(r,correct=contrast_correct(r)) for r in rr if r['panel']=='legacy'])
        report['models'][n]['legacy_candidate_decisions']={form+'/'+w:dict(n=len(gg),joint=sum(x['joint'] for x in gg),reversed=sum(x['reversed'] for x in gg)) for (form,w),gg in grouped(legacy,['form','wording']).items()}
        report['wording_transitions'][n]=wording(free);report['heuristics'][n]=heuristic(free)
        report['generation_controls'][n]=dict(candidate_free_prompts=len(free),valid=sum(r['valid'] for r in free),
            length_limit=sum(not r['ended_eos'] for r in free),generated_tokens=sum(len(r['generated_token_ids']) for r in free),
            candidate_free_prompt_tokens=sum(r['prompt_tokens'] for r in free),
            invalid_responses=[{k:r[k] for k in ['key','answer','gold','ended_eos']} for r in free if not r['valid']])
        for (form,w),gg in grouped(j,['form','wording']).items():
            key=form+'/'+w
            report['models'][n].setdefault('contrast_answer_separation',{})[key]=dict(events=len(gg),
                positive_contrast_but_joint_failure=sum(x['r']>0 and not x['joint'] for x in gg),
                compliant_positive_contrast_but_joint_failure=sum(x['r']>0 and not x['joint'] and x['both_valid'] for x in gg),
                contrast_joint_correct_but_generation_joint_wrong=sum(x['contrast_joint'] and not x['joint'] for x in gg),
                negative_contrast_and_joint_failure=sum(x['r']<0 and not x['joint'] for x in gg))
            for metric in ['joint','reversed']:
                v=[]
                for pair in range(6):
                    vals=[x[metric] for x in gg if x['pair_index']==pair];v.append(sum(vals)/len(vals))
                report['uncertainty'][n+'/'+key+'/'+metric]=ci(v)
        for w in cfg['strong_questions']:
            jj=[x for x in j if x['wording']==w]
            for metric in ['joint','reversed']:report['uncertainty'][n+'/'+w+'/object_minus_active/'+metric]=ci(gaps(jj,metric))
        order_data=rows if n=='phi3' else rr
        for panel in ['strong_list','legacy']:
            groups=list(grouped([r for r in order_data if r['panel']==panel],['id','form','wording','role']).values())
            if not all(len(g)==2 for g in groups):continue
            cells={}
            for form in cfg['forms']:
                gg=[g for g in groups if g[0]['form']==form]
                cells[form]=dict(pairs=len(gg),contrast_choice_flips=sum(contrast_correct(g[0])!=contrast_correct(g[1]) for g in gg),
                    full_answer_changes=sum(g[0]['answer']!=g[1]['answer'] for g in gg) if n=='phi3' else None)
            report['candidate_order'][n+'/'+panel]=cells
    phi={r['key']:r for r in matched}
    for n,rr in models.items():
        if n=='phi3':continue
        for (form,w),gg in grouped([r for r in rr if r['panel']=='strong_free'],['form','wording']).items():
            report['paired_model_transitions'][n+'/'+form+'/'+w]=dict(n=len(gg),
                phi_newly_correct=[r['key'] for r in gg if phi[r['key']]['correct'] and not r['correct']],
                phi_newly_incorrect=[r['key'] for r in gg if not phi[r['key']]['correct'] and r['correct']])
    for w in cfg['strong_questions']:
        for metric in ['joint','reversed']:
            key='phi3/'+w+'/object_minus_active/'+metric;other='qwen7b/'+w+'/object_minus_active/'+metric
            a=report['uncertainty'][key]['cluster_values'];b=report['uncertainty'][other]['cluster_values']
            report['uncertainty']['phi_minus_qwen7b/'+w+'/object_active_interaction/'+metric]=ci([x-y for x,y in zip(a,b)])
    # Secondary exploratory follow-up: motivated by the completed stronger-interface panel,
    # after observing its crossover; not a preregistered confirmatory contrast.
    for form in cfg['forms']:
        a=report['wording_transitions']['phi3'][form]['uncertainty']['cluster_values']
        b=report['wording_transitions']['qwen7b'][form]['uncertainty']['cluster_values']
        report['uncertainty']['exploratory_phi_minus_qwen7b/wording_change/'+form]=ci([x-y for x,y in zip(a,b)])
    report['exploratory_followup_note']='Family difference in lexical-to-explicit joint-accuracy changes was added after inspecting the complete 1152-prompt AB/strong panel while the prespecified reversed legacy controls were still running. It is exploratory; no prompts, exclusions, seeds or primary measures changed.'
    report['phi_full_generation_by_cell']={}
    for key,g in grouped(rows,['panel','form','role','wording','order']).items():
        report['phi_full_generation_by_cell']['/'.join(key)]=dict(n=len(g),correct=sum(r['correct'] for r in g),valid=sum(r['valid'] for r in g),
            first_token_correct=sum(r['first_answer']==r['gold'] for r in g),contrast_correct=sum(contrast_correct(r) for r in g))
    report['phi_token_usage']=dict(prompt_tokens=sum(r['prompt_tokens'] for r in rows),
        generated_non_eos_tokens=sum(len(r['generated_token_ids']) for r in rows),
        additional_candidate_scoring_prefill_tokens=sum(r['prompt_tokens']+1 for r in rows),
        full_vs_first_token_answer_changes=sum(r['answer']!=r['first_answer'] for r in rows))
    report['validation']=dict(phi_records=1728,matched_records_per_model=1152,qwen_record_files_unchanged=True,closed_files_unchanged=len(closed['sha256']),
        model_checksum_verified=True,new_review_inference_count=0,comparability='Same event IDs and semantic prompt text, native templates differ. Strong_free generation is matched. Legacy/list Qwen generation is one token; compare candidate-score choices separately, not against full Phi generations. No raw cross-family likelihood comparison.')
    (ROOT/'ANALYSIS.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
