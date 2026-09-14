"""Matched target/policy rows for fixed training and held-out example0."""
import argparse
import json
from pathlib import Path
import numpy as np
import mujoco
import imageio.v2 as imageio
from combat.robot import Robot
from scripts.render_results import panel
p=argparse.ArgumentParser();p.add_argument('--train',type=Path,required=True)
p.add_argument('--test',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
evaluations=[a.train,a.test];records=[np.load(r/'rollout_env0.npz',allow_pickle=False) for r in evaluations]
robot=Robot();m,d=robot.model,robot.data
maps=[[list(z['joint_names']).index(n) for n in robot.names] for z in records]
alive=[int(z['alive'].sum()) for z in records];fps=float(records[0]['fps'])
a.output.mkdir(parents=True)
camera=mujoco.MjvCamera();camera.distance=3.6;camera.azimuth=145;camera.elevation=-12
with mujoco.Renderer(m,height=480,width=640) as renderer:
    with imageio.get_writer(a.output/'multireference_diagnostic.mp4',fps=fps/4,codec='libx264',quality=8,macro_block_size=16) as writer:
        for t in range(max(len(z['qpos']) for z in records)):
            rows=[]
            for index,z in enumerate(records):
                images=[]
                for actual in (False,True):
                    frame=min(t,alive[index]-1 if actual else len(z['reference_qpos'])-1)
                    q=z['qpos' if actual else 'reference_qpos'][frame]
                    d.qpos[:7]=q[:7];d.qpos[7:]=q[7:][maps[index]];d.qvel[:]=0;mujoco.mj_forward(m,d)
                    camera.lookat[:]=[z['reference_qpos'][0,0],z['reference_qpos'][0,1],1.0]
                    renderer.update_scene(d,camera)
                    title=('HELD-OUT' if index else 'TRAINING')+(' | LEARNED POLICY' if actual else ' | MATCHED REFERENCE')
                    status='FIRST FAILURE: FROZEN' if actual and t>=alive[index] else f't={frame/fps:.2f}s | 4x slow'
                    images.append(panel(renderer.render(),title,status))
                rows.append(np.concatenate(images,axis=1))
            image=np.concatenate(rows,axis=0);writer.append_data(image)
            if t==18:imageio.imwrite(a.output/'multireference_preview.png',image)
(a.output/'provenance.json').write_text(json.dumps({'source_evaluations':[str(x) for x in evaluations],
    'selection':'first sorted training source and first sorted held-out source, environment0; no success selection',
    'alive_frames':alive,'rendering':'MuJoCo rendering of recorded Isaac states; no sim2sim policy deployment'},indent=2))
print('MULTIREFERENCE_RENDER_COMPLETE',flush=True)
