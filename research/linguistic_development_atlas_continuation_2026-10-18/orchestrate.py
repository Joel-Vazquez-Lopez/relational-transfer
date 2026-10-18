"""Fixed administrative ordering; no adaptive prompts or extra inference."""
import subprocess
import sys
from pathlib import Path

here=Path(__file__).resolve().parent
for phase in ['development','held_out']:
    for model in ['qwen7b','llama3b']:
        with (here/f'RUN_{model}_{phase}.log').open('x') as stream:
            subprocess.run([sys.executable,str(here/'run.py'),'--model',model,'--phase',phase],
                           stdout=stream,stderr=subprocess.STDOUT,check=True)
subprocess.run([sys.executable,str(here/'analyze.py')],check=True)
