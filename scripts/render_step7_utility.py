"""Same held-out reference and environment0 across the three seed0 policies."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import mujoco
import imageio.v2 as imageio
from combat.robot import Robot
from scripts.render_results import panel

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
directories=[Path('runs/g1_tracking_eval_multiref_test4_sample_000_primed'),
             Path('runs/step7_eval_generic_seed0_sample_000'),Path('runs/step7_eval_contact_seed0_sample_000')]
records=[np.load(r/'rollout_env0.npz',allow_pickle=False) for r in directories]
for z in records[1:]:
    np.testing.assert_allclose(z['reference_qpos'],records[0]['reference_qpos'],atol=1e-5)
    np.testing.assert_array_equal(z['joint_names'],records[0]['joint_names'])
robot=Robot();m,d=robot.model,robot.data
mapping=[list(records[0]['joint_names']).index(n) for n in robot.names]
alive=[int(z['alive'].sum()) for z in records];fps=float(records[0]['fps'])
if min(alive)<1:raise ValueError('Missing initial state')
a.output.mkdir(parents=True)
camera=mujoco.MjvCamera();camera.distance=3.6;camera.azimuth=145;camera.elevation=-12
camera.lookat[:]=[records[0]['reference_qpos'][0,0],records[0]['reference_qpos'][0,1],1.0]
with mujoco.Renderer(m,height=480,width=640) as renderer:
    with imageio.get_writer(a.output/'step7_utility.mp4',fps=fps/4,codec='libx264',quality=8,macro_block_size=16) as writer:
        for t in range(len(records[0]['qpos'])):
            panels=[]
            for index in range(4):
                ref=index==0;z=records[max(0,index-1)]
                frame=min(t,len(z['reference_qpos'])-1 if ref else alive[index-1]-1)
                q=z['reference_qpos' if ref else 'qpos'][frame]
                d.qpos[:7]=q[:7];d.qpos[7:]=q[7:][mapping];d.qvel[:]=0;mujoco.mj_forward(m,d)
                renderer.update_scene(d,camera)
                title=['HELD-OUT REFERENCE','A: REAL SOURCE','B: REAL + GENERIC','C: REAL + CONTACT'][index]
                status='FIRST FAILURE: FROZEN' if not ref and t>=alive[index-1] else f't={frame/fps:.2f}s | 4x slow'
                panels.append(panel(renderer.render(),title,status))
            image=np.concatenate([np.concatenate(panels[:2],axis=1),np.concatenate(panels[2:],axis=1)],axis=0)
            writer.append_data(image)
            if t==18:imageio.imwrite(a.output/'step7_utility_preview.png',image)
(a.output/'provenance.json').write_text(json.dumps({'source_evaluations':[str(x) for x in directories],
 'source_rollout_sha256':[hashlib.sha256((x/'rollout_env0.npz').read_bytes()).hexdigest() for x in directories],
 'selection':'first sorted held-out clip, fixed environment0, controller seed0; no success selection',
 'alive_frames':alive,'playback_speed':.25,'freeze_after_first_failure':True,
 'rendering':'MuJoCo rendering of actual recorded Isaac states; not MuJoCo policy deployment'},indent=2))
print('STEP7_UTILITY_RENDER_COMPLETE',flush=True)
