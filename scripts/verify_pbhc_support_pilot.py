"""Verify all source/output hashes and data-preservation invariants; plot raw traces."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
REC=ROOT/'records/pbhc_support_repair_20260917'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    assert (REC/'COMPLETE.json').exists()
    dest=REC/'verification.json'
    if dest.exists():raise FileExistsError(dest)
    protocol=json.loads((REC/'protocol.json').read_text());manifest=json.loads((REC/'manifest.json').read_text())
    rows=[]
    for row in protocol['records']:
        assert sha(Path(row['source']))==row['source_sha256']
        assert sha(Path(row['input']))==row['input_sha256']
        with np.load(row['source'],allow_pickle=False) as original,np.load(row['input'],allow_pickle=False) as candidate:
            q=candidate['qpos'];q0=np.c_[original['body_pos_w'][:,0],original['body_quat_w'][:,0],original['joint_pos']]
            assert q.shape==q0.shape and len(q)==row['frames'] and float(candidate['fps'])==50
            names=list(candidate['joint_names']);assert names==list(original['joint_names'])
            np.testing.assert_array_equal(q[:,:2],q0[:,:2]);np.testing.assert_array_equal(q[:,3:7],q0[:,3:7])
            np.testing.assert_allclose(q[:,2]-q0[:,2],row['initial_ground_shift_m'],atol=1e-12)
            active=np.arange(len(q))/50>=row['onset_s']
            np.testing.assert_array_equal(q[active,7:],q0[active,7:])
            upper=[i for i,n in enumerate(names) if not any(x in n for x in ['hip_','knee_','ankle_'])]
            np.testing.assert_array_equal(q[:,7+np.array(upper)],q0[:,7+np.array(upper)])
            delta=np.abs(q[:,7:]-q0[:,7:]);assert delta.max()<=.120001
            if row['arm']=='ground_align':np.testing.assert_array_equal(q[:,7:],q0[:,7:])
            converted=ROOT/'runs/pbhc_support_repair_20260917/references'/f'sample_{row["name"]}_{row["arm"]}_actor0.npz'
            with np.load(converted,allow_pickle=False) as z:
                np.testing.assert_allclose(z['joint_pos'],q[:,7:],atol=1e-6)
                np.testing.assert_allclose(z['body_pos_w'][:,0],q[:,:3],atol=1e-6)
                assert len(z['joint_pos'])==len(q)
            rows.append(dict(name=row['name'],arm=row['arm'],invariants_pass=True,max_joint_delta_rad=float(delta.max()),converted_sha256=sha(converted)))
    for job in manifest['jobs']:
        for key in ['checkpoint','reference','original_reference']:assert sha(ROOT/job[key])==job[key+'_sha256']
        metrics=json.loads((ROOT/job['output']/'metrics.json').read_text())
        assert metrics['trace_sha256']==sha(ROOT/job['output']/'support_trace.npz')
        assert metrics['thresholds']=={'joint_mae_rad':.3,'root_error_m':.15,'body8_error_m':.15,'minimum_root_height_m':.35}
    old_index=json.loads((ROOT/'records/pbhc_followup_20260916/metrics_index.json').read_text())
    for file,digest in old_index.items():assert sha(ROOT/file)==digest,file
    write=dict(rows=rows,unchanged_prior_metric_files=len(old_index),candidate_evaluations_verified=len(manifest['jobs']),
               training_updates=0,script_sha256=sha(Path(__file__)))
    dest.write_text(json.dumps(write,indent=2))
    names=sorted({r['name'] for r in rows});fig,axes=plt.subplots(4,2,figsize=(13,12))
    for k,name in enumerate(names):
        for arm,color in [('raw','#4c566a'),('ground_align','#d08770'),('prep_support','#5e81ac')]:
            folder=ROOT/('runs/pbhc_support_diagnostic_20260917/'+name if arm=='raw' else 'runs/pbhc_support_repair_20260917/evaluation/'+name+'_'+arm)
            with np.load(folder/'support_trace.npz',allow_pickle=False) as z:
                valid=z['alive'][:,0].astype(bool);n=valid.sum();t=np.arange(1,n+1)/float(z['fps'])
                rawref=ROOT/'runs/pbhc_beyondmimic_20260916/references_full'/f'sample_{name}_actor0.npz'
                with np.load(rawref,allow_pickle=False) as ref:
                    root=z['root_state'][valid,0,:3]-z['env_origins'][0]
                    error=np.linalg.norm(root-ref['body_pos_w'][1:n+1,0],axis=-1)
                axes[k,0].plot(t,error,label=arm,color=color)
                if arm=='raw':
                    sn=list(z['sensor_body_names'])
                    for s,c in [('left','#5e81ac'),('right','#d08770')]:
                        f=z['net_contact_forces'][valid,0,sn.index(s+'_ankle_roll_link')]
                        axes[k,1].plot(t,np.linalg.norm(f,axis=-1),label=s+' foot',color=c)
        axes[k,0].axhline(.15,color='red',ls='--',label='0.15 m threshold')
        axes[k,0].set(title=name+' | Original-target root error (env0)',ylabel='m',xlabel='s')
        axes[k,1].set(title=name+' | Raw-policy foot normal forces (env0)',ylabel='N',xlabel='s')
        for ax in axes[k]:ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('Fixed policies, nominal conditions, development sources only. Curves stop at first termination.',fontsize=11)
    fig.tight_layout();fig.savefig(REC/'support_comparison.png',dpi=160);plt.close(fig)
    print('SUPPORT_PILOT_VERIFIED',len(rows),'candidates;',len(old_index),'old metrics unchanged')

if __name__=='__main__':main()
