"""Authenticated metadata probe. Credentials remain process-local and are never serialized."""
import json,logging,os,shlex,sys
from pathlib import Path
import httpx
ROOT=Path(__file__).resolve().parent
WORK=ROOT.parents[1]
BASE='https://api.berget.ai'
MODEL='google/gemma-4-31B-it'

def load_key():
    path=WORK/'.env.local'
    if not path.is_file():raise RuntimeError('Local credential file missing')
    os.chmod(path,0o600)
    values=[]
    with path.open() as f:
        for line in f:
            line=line.strip()
            if line.startswith('export '):line=line[7:]
            if '=' not in line:continue
            name,value=line.split('=',1)
            if name.strip()!='BERGET_API_KEY':continue
            parts=shlex.split(value,comments=True,posix=True)
            if len(parts)!=1:raise RuntimeError('Credential format invalid')
            values.append(parts[0])
    if len(values)!=1 or not values[0] or any(ord(c)<33 or ord(c)>126 for c in values[0]):
        raise RuntimeError('Credential format invalid')
    return values[0]

def safe_json(path,value,key):
    text=json.dumps(value,indent=2)+'\n'
    if key in text:raise RuntimeError('Credential redaction guard triggered')
    path.write_text(text)

def client(key):
    logging.getLogger('httpx').disabled=True;logging.getLogger('httpcore').disabled=True
    return httpx.Client(base_url=BASE,headers={'Authorization':'Bearer '+key},timeout=60,follow_redirects=False,trust_env=False)

def main():
    key=load_key()
    with client(key) as c:
        response=c.get('/v1/models')
        if response.status_code!=200:
            print(json.dumps({'authenticated_metadata_connectivity':False,'http_status':response.status_code,'paid_inference_requests':0}));return 1
        data=response.json();entries=data.get('data',[])
        target=[x for x in entries if x.get('id')==MODEL]
        report={'authenticated_metadata_connectivity':True,'http_status':200,'paid_inference_requests':0,
                'target_models':[{k:x[k] for k in ['id','name','pricing','capabilities','context_length','context_window','status'] if k in x} for x in target]}
        safe_json(ROOT/'CONNECTIVITY.json',report,key)
        print(json.dumps(report));return 0

if __name__=='__main__':
    try:sys.exit(main())
    except Exception as exc:
        # Never serialize exception text, request objects, headers, or environment values.
        print(json.dumps({'connectivity_probe_failed':True,'error_type':type(exc).__name__}));sys.exit(1)
