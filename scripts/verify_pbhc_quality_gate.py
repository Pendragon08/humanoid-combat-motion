"""Recompute source gates, preserve all rejected candidates and test unsafe inputs."""
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
from prepare_pbhc_quality_gate import ROOT,REC,OUT,sha,check

def main():
    target=REC/'verification.json'
    if target.exists():raise FileExistsError(target)
    cfg=json.loads((REC/'frozen_config.json').read_text())
    selection=json.loads((REC/'selection.json').read_text())
    manifest=json.loads((REC/'manifest.json').read_text())
    inventory=json.loads((ROOT/'runs/pbhc_seed_panel_20260916/inventory.json').read_text())['motions']
    records=[];negative=[]
    assert len(selection['records'])==4
    for row in selection['records']:
        parent=next(x for x in manifest['records'] if x['name']==row['name'])
        for key in ['source','selected_input']:assert sha(Path(row[key]))==row['source_sha256' if key=='source' else 'selected_sha256']
        with np.load(row['source'],allow_pickle=False) as z:
            names=list(z['joint_names']);raw=np.c_[z['body_pos_w'][:,0],z['body_quat_w'][:,0],z['joint_pos']].astype(float)
        with np.load(row['selected_input'],allow_pickle=False) as z:
            selected=z['qpos'];assert list(z['joint_names'])==names and float(z['fps'])==50
        assert selected.shape==raw.shape
        case=next(c for c in inventory if c['name']==row['name'])
        c=np.asarray(next(iter(joblib.load(case['source_path']).values()))['contact_mask'])
        ids=np.minimum(np.rint(np.arange(len(raw))/50*case['source_fps']).astype(int),len(c)-1);c=c[ids]
        gate=check(raw,selected,names,c,parent['onset_s'],cfg)
        assert gate['pass_gate']==(row['status']=='eligible_for_probe')
        if row['status']=='eligible_for_probe':
            assert row['trials'][-1]['gate']['checks']==gate['checks']
            for k,v in gate['metrics'].items():np.testing.assert_allclose(v,row['trials'][-1]['gate']['metrics'][k],atol=1e-10)
        else:np.testing.assert_array_equal(selected,raw)
        with np.load(OUT/'references'/f'sample_{row["name"]}_actor0.npz',allow_pickle=False) as converted:
            np.testing.assert_allclose(converted['joint_pos'],selected[:,7:],atol=1e-6)
            np.testing.assert_allclose(converted['body_pos_w'][:,0],selected[:,:3],atol=1e-6)
            assert len(converted['joint_pos'])==len(raw)
        records.append(dict(name=row['name'],scale=row['selected_scale'],gate=gate,frames=len(raw),input_sha256=row['selected_sha256']))
        # Fault injection: a one-frame splice and a global downward shift must not pass.
        splice=selected.copy();j=names.index('left_knee_joint');splice[5,7+j]+=.1
        sg=check(raw,splice,names,c,parent['onset_s'],cfg)
        assert not sg['pass_gate'] and not sg['checks']['joint_delta_acceleration_rad_s2']
        shifted=selected.copy();shifted[:,2]-=.01;hg=check(raw,shifted,names,c,parent['onset_s'],cfg)
        assert not hg['pass_gate'] and not hg['checks']['root_no_downshift']
        upper=selected.copy();upper[5,7+names.index('left_elbow_joint')]+=.05;ug=check(raw,upper,names,c,parent['onset_s'],cfg)
        assert not ug['pass_gate'] and not ug['checks']['upper_joints_unchanged']
        negative.append(dict(name=row['name'],one_frame_splice_rejected=True,downshift_rejected=True,upper_body_change_rejected=True))
    old=json.loads((ROOT/'records/pbhc_followup_20260916/metrics_index.json').read_text())
    for p,h in old.items():assert sha(ROOT/p)==h
    output=dict(records=records,negative_controls=negative,unchanged_historical_metric_files=len(old),
        selection_used_execution_scores=False,script_sha256=sha(Path(__file__)),
        limitations='Shared gate implementation; deterministic verification not independent algorithm review. No dynamics/semantic guarantee.')
    target.write_text(json.dumps(output,indent=2));print('QUALITY_GATE_VERIFIED',len(records),'sources,',len(negative)*3,'negative controls')

if __name__=='__main__':main()
