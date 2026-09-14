"""Audit pair shape, synchronization and finiteness without altering source arrays."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from combat.protocol import load_manifest, source_split_check


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--min-frames',type=int,default=16)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    kept=[];excluded=[];counts=Counter()
    for row in load_manifest(a.manifest):
        arrays=[np.load(path,allow_pickle=False,mmap_mode='r') for path in row['motion_paths']]
        shapes=[list(x.shape) for x in arrays];reasons=[]
        if any(x.ndim!=2 or x.shape[1]!=492 for x in arrays):reasons.append('unexpected_raw_shape')
        if any(len(x)<a.min_frames for x in arrays):reasons.append('too_short')
        if len(arrays[0])!=len(arrays[1]):reasons.append('unequal_person_frame_counts')
        if any(not np.isfinite(x).all() for x in arrays):reasons.append('nonfinite')
        audit={'sample_id':row['sample_id'],'split':row['split'],'shapes':shapes,'reasons':reasons}
        if reasons:
            excluded.append(audit);continue
        row['raw_shapes']=shapes
        row['motion_sha256']=[hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in row['motion_paths']]
        kept.append(row);counts[row['split']]+=1
        if row.get('weak_tags'):counts[row['split']+'_weak_combat']+=1
    check=source_split_check(kept)
    summary={'input_manifest_sha256':hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
             'minimum_raw_frames':a.min_frames,'counts':dict(counts),'excluded_count':len(excluded),
             'exclusion_reasons':dict(Counter(r for x in excluded for r in x['reasons'])),
             'source_split_check':check,'raw_motion_modified':False,
             'scope':'shape/finite-value data integrity only; not contact or physical feasibility'}
    a.output.mkdir(parents=True)
    (a.output/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in kept),encoding='utf-8')
    (a.output/'excluded.json').write_text(json.dumps(excluded,indent=2),encoding='utf-8')
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
