"""Download ONLY the selected pinned public weight, with validated resumable HTTP ranges."""
import concurrent.futures,hashlib,json,shutil,time
from pathlib import Path
import httpx
ROOT=Path(__file__).resolve().parent
SIZE=2149696167
EXPECTED='8d75680621a09474f6601e9176f2f61f92a5e4c079d68d583901f51699fda50a'
URL='https://huggingface.co/mlx-community/Phi-3-mini-4k-instruct-4bit/resolve/5b3819ed6317784fb20eddeae9bed984f778d0d0/model.safetensors'
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
