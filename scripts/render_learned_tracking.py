"""Visual replay of recorded Isaac PPO trajectories; never re-simulates them."""
import argparse
import json
from pathlib import Path
import numpy as np
import mujoco
import imageio.v2 as imageio
from combat.robot import Robot
from scripts.render_results import panel


def main():
    p=argparse.ArgumentParser();p.add_argument('--early',type=Path,required=True)
    p.add_argument('--extended',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    z=[np.load(x/'rollout_env0.npz',allow_pickle=False) for x in [a.early,a.extended]]
    metrics=[json.loads((x/'metrics.json').read_text()) for x in [a.early,a.extended]]
    robot=Robot();m,d=robot.model,robot.data
    maps=[[list(x['joint_names']).index(n) for n in robot.names] for x in z]
    alive_lengths=[int(x['alive'].sum()) for x in z]
    fps=float(z[0]['fps']);length=len(z[0]['qpos'])
    a.output.mkdir(parents=True)
    camera=mujoco.MjvCamera();camera.distance=2.7;camera.azimuth=145;camera.elevation=-12
    with mujoco.Renderer(m,height=480,width=640) as renderer:
        with imageio.get_writer(a.output/'learned_tracking_diagnostic.mp4',fps=fps/4,codec='libx264',quality=8,macro_block_size=16) as writer:
            for t in range(length):
                images=[]
                for col in range(3):
                    j=max(col-1,0);actual=col>0
                    frame=min(t,alive_lengths[j]-1) if actual else t
                    q=z[j]['qpos' if actual else 'reference_qpos'][frame]
                    d.qpos[:7]=q[:7];d.qpos[7:]=q[7:][maps[j]];d.qvel[:]=0
                    mujoco.mj_forward(m,d)
                    camera.lookat[:]=[z[0]['reference_qpos'][0,0],z[0]['reference_qpos'][0,1],.7]
                    renderer.update_scene(d,camera)
                    title=['REFERENCE REPLAY','PPO: 100 ITERATIONS','PPO: 1000 MORE ITERATIONS'][col]
                    status='FAILED: frame frozen' if actual and t>=alive_lengths[j] else f't={t/fps:.2f}s | 4x slow'
                    images.append(panel(renderer.render(),title,status+' | recorded Isaac trajectory' if actual else status))
                frame=np.concatenate(images,axis=1);writer.append_data(frame)
                if t==min(18,length-1):imageio.imwrite(a.output/'learned_tracking_preview.png',frame)
    (a.output/'provenance.json').write_text(json.dumps({'source_evaluations':[str(a.early),str(a.extended)],
        'selection':'environment 0, predetermined; no success selection','alive_frames':alive_lengths,
        'completion_fraction':[x['completion_fraction'] for x in metrics],
        'rendering':'MuJoCo forward-kinematic visualization of Isaac states, not cross-simulator control',
        'speed':'quarter speed; frozen at first tracking failure; excludes automatic resets'},indent=2))
    print('LEARNED_TRACKING_RENDER_COMPLETE',a.output,flush=True)


if __name__=='__main__':main()
