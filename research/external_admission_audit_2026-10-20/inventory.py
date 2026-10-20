"""Bounded repository metadata inventory. Never executes remote code or weights."""
import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPOS = ['cl-tohoku/OI-Subspace', 'redwoodresearch/Easy-Transformer', 'constanzafierro/multilingual_factual_memorization', 'technion-cs-nlp/llm-arithmetic-heuristics', 'HumanCompatibleAI/leela-interp', 'kmeng01/rome']

def api(endpoint):
    b = subprocess.check_output(['/opt/homebrew/bin/gh', 'api', endpoint], timeout=60)
    if len(b) > 20 * 1024 * 1024:
        raise ValueError('metadata size cap')
    return json.loads(b)

def main():
    out = []
    for repo in REPOS:
        stem = repo.replace('/', '__')
        meta = api('repos/' + repo)
        head = api('repos/' + repo + '/commits/' + meta['default_branch'])['sha']
        tree = api('repos/' + repo + '/git/trees/' + head + '?recursive=1')
        if tree.get('truncated'):
            raise ValueError('truncated tree: ' + repo)
        releases = api('repos/' + repo + '/releases?per_page=100')
        rec = dict(repo=repo, head=head, default_branch=meta['default_branch'], license=meta.get('license'), tree=tree['tree'], releases=releases)
        b = (json.dumps(rec, indent=2) + '\n').encode()
        (HERE / (stem + '.json')).write_bytes(b)
        blobs = [x for x in tree['tree'] if x['type'] == 'blob']
        data = [dict(path=x['path'], bytes=x.get('size')) for x in blobs if Path(x['path']).suffix.lower() in ['.json', '.jsonl', '.csv', '.tsv', '.npz', '.npy', '.pt', '.pkl', '.pickle', '.h5', '.hdf5', '.zip', '.tar', '.gz']]
        out.append(dict(repo=repo, head=head, files=len(blobs), metadata_sha256=hashlib.sha256(b).hexdigest(), possible_data=data, release_assets=[dict(release=r['tag_name'], assets=[dict(name=a['name'], bytes=a['size'], url=a['browser_download_url']) for a in r['assets']]) for r in releases]))
        print(json.dumps(out[-1]), flush=True)
    (HERE / 'REPOSITORY_INVENTORY.json').write_text(json.dumps(out, indent=2) + '\n')

if __name__ == '__main__':
    main()
