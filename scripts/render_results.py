"""Render reference replay and physical rollout as distinct videos with visible labels."""
import argparse
from pathlib import Path
import json
import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
from combat.robot import Robot, ROOT

def panel(frame,title,subtitle):
    im=Image.fromarray(frame);draw=ImageDraw.Draw(im)
    draw.rectangle((0,0,im.width,64),fill=(15,25,40))
    try:font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    except OSError:font=ImageFont.load_default()
    draw.text((12,8),title,font=font,fill='white');draw.text((12,34),subtitle,font=font,fill=(180,208,235))
    return np.asarray(im)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);args=p.parse_args()
    out=ROOT/'runs'/args.run;media=out/'media';media.mkdir(exist_ok=True)
    z=np.load(out/'motions.npz',allow_pickle=False);robot=Robot();m=robot.model;d=robot.data
    camera=mujoco.MjvCamera();camera.distance=3.1;camera.azimuth=135;camera.elevation=-12
    # Predetermined first window of each test source, no success-based cherry-picking.
    ids=[]
    for source in sorted(set(z['source_ids'])):
        ids.append(int(np.flatnonzero(z['source_ids']==source)[0]))
    selection={'cases':ids,'rule':'first evaluated case from each held-out source, independent of outcomes'}
    (media/'selection.json').write_text(json.dumps(selection,indent=2))
    with mujoco.Renderer(m,height=480,width=640) as renderer:
        with imageio.get_writer(media/'reference_comparison.mp4',fps=30,codec='libx264',quality=8,macro_block_size=16) as writer:
            for case in ids:
                seq=z['qpos'][case]
                for t in range(len(seq[0])):
                    panels=[]
                    for arm,title in [(0,'VIDEO-DERIVED SEED'),(1,'DIFFUSION AUGMENTATION'),(3,'JOINT + GROUND REFINEMENT')]:
                        d.qpos[:]=seq[arm,t];d.qvel[:]=0;mujoco.mj_forward(m,d)
                        camera.lookat[:]=[seq[0,t,0],seq[0,t,1],.8]
                        renderer.update_scene(d,camera)
                        panels.append(panel(renderer.render(),title,f'REFERENCE REPLAY | source {z["source_ids"][case]}'))
                    frame=np.concatenate(panels,axis=1);writer.append_data(frame)
                    if case==ids[0] and t==len(seq[0])//2:imageio.imwrite(media/'reference_preview.png',frame)
        with imageio.get_writer(media/'physical_pd_diagnostic.mp4',fps=30,codec='libx264',quality=8,macro_block_size=16) as writer:
            for case in ids:
                seq=z['qpos'][case];rolls=[]
                for arm in [0,1,3]:
                    result,poses=robot.diagnostic_rollout(seq[arm],record=True);rolls.append((result,poses))
                n=max(len(poses) for _,poses in rolls)
                for t in range(n):
                    panels=[]
                    for (result,poses),label in zip(rolls,['VIDEO SEED','DIFFUSION','REFINED']):
                        q=poses[min(t,len(poses)-1)];d.qpos[:]=q;d.qvel[:]=0;mujoco.mj_forward(m,d)
                        camera.lookat[:]=[seq[0,0,0],seq[0,0,1],.65];camera.distance=3.6
                        renderer.update_scene(d,camera)
                        status='FALL' if result['pd_fall_time_s'] is not None and t/30>=result['pd_fall_time_s'] else 'NO FALL FLAG YET'
                        panels.append(panel(renderer.render(),label+' | '+status,'FREE-BASE PD | no learned balance policy'))
                    writer.append_data(np.concatenate(panels,axis=1))
    print('RENDER_COMPLETE',media)

if __name__=='__main__':main()
