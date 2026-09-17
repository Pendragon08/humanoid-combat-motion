"""Finite source-only backtracking, with unchanged v1 thresholds.

Recorded after v1 data inspection but before ANY new policy execution.
This is development of a gate, not an independently validated selection method.
"""
import json
from pathlib import Path
import joblib
import numpy as np
from prepare_pbhc_quality_gate import ROOT,REC,OUT,sha,check

def main():
    target=REC/'selection.json';folder=OUT/'backtracking_selected'
    if target.exists() or folder.exists():raise FileExistsError('Selection is immutable')
    folder.mkdir()
    cfg=json.loads((REC/'frozen_config.json').read_text())
    protocol=dict(scales=[1.,.75,.5,.25],fallback='raw_source',
        ordering='Largest correction scale passing ALL unchanged data-only checks, never policy-score selection',
        development='Backtracking added after v1 data-only foot-XY rejections; before new execution; not preregistered before all development',
        config_sha256=sha(ROOT/'configs/pbhc_quality_gate_v1.json'),script_sha256=sha(Path(__file__)))
    (REC/'selection_protocol.json').write_text(json.dumps(protocol,indent=2))
    inputs=json.loads((REC/'manifest.json').read_text())['records']
    inventory=json.loads((ROOT/'runs/pbhc_seed_panel_20260916/inventory.json').read_text())['motions']
    rows=[]
    for entry in inputs:
        source=Path(entry['source']);assert sha(source)==entry['source_sha256']
        with np.load(source,allow_pickle=False) as z:
            names=list(z['joint_names']);raw=np.c_[z['body_pos_w'][:,0],z['body_quat_w'][:,0],z['joint_pos']].astype(float)
        candidatepath=Path(entry['candidate']);assert sha(candidatepath)==entry['candidate_sha256']
        with np.load(candidatepath,allow_pickle=False) as z:candidate=z['qpos']
        case=next(c for c in inventory if c['name']==entry['name'])
        contact=np.asarray(next(iter(joblib.load(case['source_path']).values()))['contact_mask'])
        ids=np.minimum(np.rint(np.arange(len(raw))/50*case['source_fps']).astype(int),len(contact)-1);contact=contact[ids]
        trials=[];selected=raw;chosen=0.
        # Data-only finite search: store all attempted scales, no hiding rejections.
        for scale in protocol['scales']:
            q=raw+scale*(candidate-raw)
            gate=check(raw,q,names,contact,entry['onset_s'],cfg)
            trials.append(dict(scale=scale,gate=gate))
            if gate['pass_gate']:selected=q;chosen=scale;break
        path=folder/f'sample_{entry["name"]}.npz';np.savez_compressed(path,qpos=selected,fps=50.,joint_names=np.array(names))
        row=dict(name=entry['name'],selected_scale=chosen,status='eligible_for_probe' if chosen else 'raw_fallback',
            selected_input=str(path),selected_sha256=sha(path),source=str(source),source_sha256=sha(source),
            candidate_sha256=entry['candidate_sha256'],trials=trials)
        rows.append(row);print(entry['name'],chosen,row['status'],flush=True)
    target.write_text(json.dumps(dict(protocol=protocol,records=rows,used_execution_scores=False),indent=2))

if __name__=='__main__':main()
