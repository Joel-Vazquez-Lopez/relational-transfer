"""Fetch pinned small source files for static inspection only; never import them."""
import base64
import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
SELECT = {
    'cl-tohoku/OI-Subspace': ['README.md'],
    'redwoodresearch/Easy-Transformer': ['README.md', 'experiments.py', 'easy_transformer/ioi_dataset.py', 'easy_transformer/ioi_utils.py'],
    'constanzafierro/multilingual_factual_memorization': ['README.md', 'causal_analysis.py', 'patching_experiments.py', 'dataset/data_utils.py', 'inference/run_evaluation.py', 'inference/run_inference.py', 'attention_knockout.py'],
    'technion-cs-nlp/llm-arithmetic-heuristics': ['README.md', '.gitignore', 'prompt_generation.py', 'evaluation_utils.py', 'activation_patching.py', 'script_circuit_localization.py', 'llm-arithmetic-analysis-main.ipynb', 'pythia-heuristics-analysis-notebook.ipynb'],
    'HumanCompatibleAI/leela-interp': ['README.md', '.gitignore', 'scripts/make_puzzles.py', 'scripts/make_corruptions.py', 'scripts/run_global_patching.py', 'scripts/run_single_head.py', 'notebooks/other_figures.ipynb', 'notebooks/act_patching.ipynb'],
    'kmeng01/rome': ['README.md', 'experiments/causal_trace.py', 'dsets/knowns.py', 'notebooks/average_causal_effects.ipynb', 'globals.yml'],
}

def main():
    manifest = []
    total = 0
    for repo, paths in SELECT.items():
        meta = json.loads((HERE / (repo.replace('/', '__') + '.json')).read_text())
        known = {x['path']: x for x in meta['tree'] if x['type'] == 'blob'}
        for path in paths:
            entry = known[path]
            if entry['size'] > 20 * 1024 ** 2:
                raise ValueError('per-file cap')
            dest = HERE / 'sources' / repo / path
            if dest.exists():
                b = dest.read_bytes()
            else:
                content = subprocess.check_output(['/opt/homebrew/bin/gh', 'api', 'repos/' + repo + '/git/blobs/' + entry['sha']], timeout=60)
                payload = json.loads(content)
                if payload['encoding'] != 'base64':
                    raise ValueError('blob encoding')
                b = base64.b64decode(payload['content'])
            git = hashlib.sha1(b'blob ' + str(len(b)).encode() + b'\0' + b).hexdigest()
            if git != entry['sha'] or len(b) != entry['size']:
                raise ValueError('blob mismatch')
            total += len(b)
            if total > 200 * 1024 ** 2:
                raise ValueError('total cap')
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b)
            manifest.append(dict(repo=repo, commit=meta['head'], path=path, git_blob=git, sha256=hashlib.sha256(b).hexdigest(), bytes=len(b), local=str(dest.relative_to(HERE))))
            print(repo, path, len(b), flush=True)
    (HERE / 'FETCH_MANIFEST.json').write_text(json.dumps(dict(total_bytes=total, files=manifest), indent=2) + '\n')

if __name__ == '__main__':
    main()
