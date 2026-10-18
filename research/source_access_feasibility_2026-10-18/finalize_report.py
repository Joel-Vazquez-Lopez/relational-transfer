"""Post-outcome reporting clarification only; no scoring or inference changes."""
import json
from pathlib import Path
from core import save,sha

HERE=Path(__file__).resolve().parent

def main():
    lock=json.loads((HERE/'LOCK.json').read_text())
    verification={n:dict(expected=h,actual=sha(HERE/n),unchanged=sha(HERE/n)==h) for n,h in lock['files'].items()}
    assert all(x['unchanged'] for x in verification.values())
    save(HERE/'FROZEN_FILES_POST_RUN.json',dict(all_unchanged=True,files=verification))
    p=HERE/'REPORT.md'; text=p.read_text()
    heading='## Interpretation correction: the competence scorer was inadequate'
    if heading not in text:
        first,rest=text.split('\n',1)
        text=first+'\n\n'+heading+'\n\n'+(
            '**This is a scoring-design failure, not a demonstrated linguistic failure.** '
            'The frozen endpoint passed 6/8 baselines, but both rejected outputs were the first token `N` of the possible multi-token answer `Nina`. '
            'Preflight accepted a single-token leading-space alias while the no-leading-space name requires two tokens. '
            'No continuation was observed, so neither full-answer correctness nor full-answer failure is established on those two cases. '
            'The machine stop label below is preserved; [POSTMORTEM.md](POSTMORTEM.md) explains the investigator error. '
            '**Zero substantive source-access cuts, zero evaluation calls and zero Llama calls were executed.** '
            'All 13 frozen input/code/protocol/test artifacts remained byte-identical. No adaptive rescue followed.\n')+rest
        p.write_text(text)
    manifest={}
    for f in sorted(HERE.rglob('*')):
        if f.is_file() and f.name not in ['RESULT_MANIFEST.json','GITHUB_PRESERVATION.json'] and '__pycache__' not in f.parts:
            manifest[str(f.relative_to(HERE))]=dict(bytes=f.stat().st_size,sha256=sha(f))
    save(HERE/'RESULT_MANIFEST.json',dict(files=manifest,total_bytes=sum(x['bytes'] for x in manifest.values())))
    print('Clarified report; all frozen files unchanged; manifest refreshed')

if __name__=='__main__': main()
