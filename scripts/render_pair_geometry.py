"""Side-by-side first-frame geometry audit; explicitly not a physical rollout."""
import argparse
from pathlib import Path
import numpy as np
import mujoco
from PIL import Image,ImageDraw
from combat.twist_tracking import scene
from combat.paired_refinement import penetration
p=argparse.ArgumentParser();p.add_argument('--case',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
z=np.load(a.case/'refinement/geometry_audit.npz');m,_=scene(2);d=mujoco.MjData(m)
r=mujoco.Renderer(m,height=544,width=960);cam=mujoco.MjvCamera();cam.azimuth=135;cam.elevation=-15;cam.distance=3.
cam.lookat[:]=np.mean(z['original_qpos'][0].reshape(2,32)[:,:3],axis=0);cam.lookat[2]=.65
panels=[]
for key,label in [('original_qpos','ORIGINAL geometry'),('corrected_qpos','CORRECTED geometry')]:
    d.qpos[:]=z[key][0];mujoco.mj_forward(m,d);depth=penetration(m,d)
    r.update_scene(d,camera=cam);im=Image.fromarray(r.render());draw=ImageDraw.Draw(im)
    draw.rectangle((0,0,960,58),fill=(20,30,45));draw.text((10,8),label+' | first reference frame | NOT a physics rollout',fill='white')
    draw.text((10,30),f'Penetration ground/self/pair (m): {np.round(depth,4)}',fill='white');panels.append(im)
canvas=Image.new('RGB',(1920,544));canvas.paste(panels[0],(0,0));canvas.paste(panels[1],(960,0));a.output.parent.mkdir(parents=True,exist_ok=True);canvas.save(a.output);r.close();print('PAIR_GEOMETRY_IMAGE',a.output)
