"""Download ONLY the approved pinned public Llama weight, with validated resumable HTTP ranges."""
import concurrent.futures,hashlib,json,shutil,time
from pathlib import Path
import httpx
ROOT=Path(__file__).resolve().parent
SIZE=1807496278
EXPECTED='d75e1ee0ea653cc5b76191ec934c7c0d568e94d4e47846619f1f4bc715b7b265'
URL='https://huggingface.co/mlx-community/Llama-3.2-3B-Instruct-4bit/resolve/7f0dc925e0d0afb0322d96f9255cfddf2ba5636e/model.safetensors'
CHUNK=256*1024*1024
def main():
    model=ROOT/'model';parts=model/'.parts';parts.mkdir(exist_ok=True)
    seed=list((model/'.cache/huggingface/download').glob('*.incomplete'))
    assert len(seed)<=1
    if seed:
        # The previous ordinary HTTPS stream wrote a contiguous prefix.
        with seed[0].open('rb') as f:
            for i,start in enumerate(range(0,SIZE,CHUNK)):
                data=f.read(min(CHUNK,SIZE-start))
                if not data:break
                p=parts/f'{i:02d}.part'
                if not p.exists() or p.stat().st_size<len(data):p.write_bytes(data)
    def one(pair):
        i,start=pair;end=min(SIZE,start+CHUNK)-1;p=parts/f'{i:02d}.part';n=p.stat().st_size if p.exists() else 0
        for attempt in range(6):
            if n==end-start+1:return p
            assert n<end-start+1
            try:
                with httpx.stream('GET',URL,headers={'Range':f'bytes={start+n}-{end}'},follow_redirects=True,timeout=30) as response:
                    assert response.status_code==206,(i,response.status_code)
                    assert response.headers.get('content-range')==f'bytes {start+n}-{end}/{SIZE}'
                    with p.open('ab') as f:
                        for block in response.iter_bytes(1024*1024):
                            f.write(block);n+=len(block)
                assert n==end-start+1
                return p
            except (httpx.HTTPError,OSError) as exc:
                n=p.stat().st_size if p.exists() else 0
                print(json.dumps({'chunk':i,'retry':attempt+1,'error_type':type(exc).__name__}),flush=True);time.sleep(min(2**attempt,8))
        raise RuntimeError(f'incomplete chunk {i}')
    started=time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        paths=list(pool.map(one,enumerate(range(0,SIZE,CHUNK))))
    temp=model/'model.safetensors.assembling';h=hashlib.sha256()
    with temp.open('wb') as f:
        for p in paths:
            with p.open('rb') as source:
                for block in iter(lambda:source.read(8*1024*1024),b''):
                    f.write(block);h.update(block)
    assert temp.stat().st_size==SIZE and h.hexdigest()==EXPECTED
    temp.replace(model/'model.safetensors')
    (ROOT/'DOWNLOAD_VALIDATION.json').write_text(json.dumps({'bytes':SIZE,'sha256':h.hexdigest(),'pinned_LFS_hash_matches':True,'method':'public HTTPS ranges; 6 workers; retained prior contiguous prefix','seconds':time.monotonic()-started},indent=2)+'\n')
    # Remove only redundant transfer fragments created in this experiment, after checksum validation.
    shutil.rmtree(parts)
    for p in seed:p.unlink()
    print('Verified selected pinned model weight.',flush=True)
if __name__=='__main__':main()
