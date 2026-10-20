"""Create the pre-output lock after generator preflight, before model initialization."""
from pathlib import Path
import hashlib,json,sys,importlib.metadata,datetime
HERE=Path(__file__).resolve().parent
FILES=['factorial_data.py','preflight.py','legacy_model.py','legacy_renderer.py','run_factorial.py','analyze_factorial.py','freeze_factorial.py','FACTORIAL_PROTOCOL.md','REUSED_CODE.json','PREFLIGHT.json','ANALYTICAL_PREFLIGHT_REVISION.json','PREINFERENCE_TECHNICAL_REPAIR.json']
def main():
    output=HERE/'FROZEN_LOCK.json';assert not output.exists(),'Refuse overwrite frozen lock'
    pre=json.loads((HERE/'PREFLIGHT.json').read_text());assert pre['status']=='PASS_PRETRAINING_DESIGN'
    assert pre['model_initializations']==pre['model_forwards']==0
    lock=dict(status='FROZEN_BEFORE_MODEL_INITIALIZATION',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),preflight_status=pre['status'],frozen_files={p:hashlib.sha256((HERE/p).read_bytes()).hexdigest() for p in FILES},python=sys.version,numpy=importlib.metadata.version('numpy'),torch=importlib.metadata.version('torch'),updates=3000,batch_size=128,checkpoints=[0,100,500,1500,3000],technical_seed=7283999,development_seed=7284101,confirmation_seeds=list(range(7284201,7284209)),resources=dict(wall_seconds=5400,peak_rss_bytes=int(1.5*1024**3),artifact_bytes=1024**3,disk_reserve_bytes=5*1024**3),training_models_maximum=54,paid_api_calls=0,large_model_downloads=0)
    output.write_text(json.dumps(lock,indent=2,sort_keys=True)+'\n');print(json.dumps(lock,indent=2))
if __name__=='__main__':main()
