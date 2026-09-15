"""Fetch pinned official inference files only; record every downloaded hash."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request
import urllib.parse

ROOT = Path(__file__).resolve().parents[1]
COMMIT = json.loads((ROOT / 'env-twist-inference.json').read_text())['commit']
BASE = f'https://raw.githubusercontent.com/YanjieZe/TWIST/{COMMIT}/'

def fetch(entry):
    path = entry['path']
    dst = ROOT / 'third_party/TWIST' / path
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        payload = urllib.request.urlopen(BASE + urllib.parse.quote(path), timeout=90).read()
        dst.write_bytes(payload)
    payload = dst.read_bytes()
    blob = hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest()
    if blob != entry['sha']: raise ValueError(f'Pinned Git blob mismatch: {path}')
    return {'path': path, 'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest(), 'git_blob_sha1': blob}

def main():
    tree = json.load(urllib.request.urlopen(f'https://api.github.com/repos/YanjieZe/TWIST/git/trees/{COMMIT}?recursive=1'))['tree']
    paths = [x['path'] for x in tree if x['type'] == 'blob' and (
        x['path'].startswith('assets/g1/meshes/') or x['path'].startswith('deploy_real/') or
        x['path'] in ('assets/g1/g1_sim2sim_with_wrist_roll.xml', 'assets/twist_general_motion_tracker.pt', 'LICENSE', 'README.md'))]
    total = sum(x.get('size', 0) for x in tree if x['path'] in paths)
    assert total < 500 * 1024**2, total
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(fetch, [x for x in tree if x['path'] in paths]))
    out = ROOT / 'records/twist/provenance.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'commit': COMMIT, 'source': BASE, 'files': records,
        'bundled_motion_files': [x['path'] for x in tree if x['path'].endswith(('.pkl','.npz','.npy'))]}, indent=2))
    print('TWIST_FETCHED', len(records), sum(r['bytes'] for r in records), flush=True)

if __name__ == '__main__': main()
