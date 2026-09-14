"""Preserve raw baseline; write actor-correspondence-corrected derivative run."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from combat.identity import stabilize_actor_identity


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    manifest=json.loads((a.input/'manifest.json').read_text(encoding='utf-8'));a.output.mkdir(parents=True)
    for r in manifest['samples']:
        path=a.input/(r['sample_id']+'.npz')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=r['sha256']:raise ValueError('Input hash mismatch')
        with np.load(path,allow_pickle=False) as z:
            fixed,indices,audit=stabilize_actor_identity(z['positions'])
            features=z['features'][np.arange(len(indices))[:,None],indices]
            if not np.array_equal(features[...,:66].reshape(fixed.shape),fixed):raise ValueError('Feature/position mismatch')
            target=a.output/path.name
            np.savez_compressed(target,positions=fixed,features=features,fps=z['fps'],actor_permutation=indices)
        r['raw_sha256']=r['sha256'];r['sha256']=hashlib.sha256(target.read_bytes()).hexdigest();r['identity_audit']=audit
    manifest['postprocess']='two-state Viterbi actor correspondence; no smoothing or physical optimization'
    manifest['raw_run']=str(a.input.resolve())
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps([r['identity_audit'] for r in manifest['samples']],indent=2))


if __name__=='__main__':main()
