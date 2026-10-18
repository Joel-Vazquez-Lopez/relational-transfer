"""Explicit reproducibility setup; installs only into sibling isolated runtime."""
from pathlib import Path
import sys,subprocess,urllib.request,hashlib
P=Path(__file__).resolve().parent.parent/'transformer_comparison_feasibility_2026-10-18'
requirements=['jax==0.4.35','jaxlib==0.4.35','dm-haiku==0.0.13','chex==0.1.87','jmp==0.0.4','absl-py==2.3.1','ml-dtypes==0.5.4','opt-einsum==3.4.0','einops==0.8.1','tabulate==0.9.0','toolz==1.0.0','https://github.com/google-deepmind/tracr/archive/9ce2b8c82b6ba10e62e86cf6f390e7536d4fd2cd.zip']
subprocess.run([sys.executable,'-m','pip','install','--target',str(P/'runtime'),'--no-deps',*requirements],check=True)
u='https://raw.githubusercontent.com/alansun17904/interp-equiv/0bdc4f6ff45c9e8319d2450bcd63b082b4e167e9/left/encoding.py'
b=urllib.request.urlopen(u,timeout=30).read();sha=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
assert sha=='abd3db7367317c51974a265026efb1649a4ec76a';(P/'upstream').mkdir(exist_ok=True);(P/'upstream/encoding.py').write_bytes(b)
print('Isolated runtime and exact baseline source ready')
