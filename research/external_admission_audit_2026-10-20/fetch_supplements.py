"""Bounded static supplements; HTTP data only. No model downloads or execution."""
import hashlib
import json
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
URLS = {
    'dai_software.zip': 'https://aclanthology.org/attachments/2024.emnlp-main.967.software.zip',
    'figshare_share.html': 'https://figshare.com/s/adc80845c00b67c8fce5',
    'rome_data.html': 'https://rome.baulab.info/data/',
}

def main():
    out = []
    for name, url in URLS.items():
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Research-artifact-audit/1.0'})
            with urllib.request.urlopen(req, timeout=45) as response:
                b = response.read(20 * 1024 ** 2 + 1)
                if len(b) > 20 * 1024 ** 2:
                    raise ValueError('per-file cap; incomplete response discarded')
                headers = dict(response.headers)
                final_url = response.url
            (HERE / name).write_bytes(b)
            out.append(dict(name=name, url=url, final_url=final_url, bytes=len(b), sha256=hashlib.sha256(b).hexdigest(), headers=headers, status='fetched'))
        except Exception as e:
            out.append(dict(name=name, url=url, status='failed', error=str(e)))
        print(json.dumps(out[-1]), flush=True)
    (HERE / 'SUPPLEMENT_FETCH.json').write_text(json.dumps(out, indent=2) + '\n')

if __name__ == '__main__':
    main()
