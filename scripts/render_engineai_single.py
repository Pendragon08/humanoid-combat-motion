"""Render saved Isaac trajectories using SDK geometry, with explicit provenance.

This is visualization of recorded states, not a new MuJoCo execution test.
"""
import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--kind',choices=['reference','isaac_pd','isaac_ppo'],required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    with np.load(a.input,allow_pickle=False) as z:
        if a.kind=='reference':q=z['qpos'];names=z['joint_names'].tolist();fps=float(z['fps']);alive=np.ones(len(q),bool)
        elif a.kind=='isaac_ppo':q=z['qpos'];names=z['joint_names'].tolist();fps=float(z['fps']);alive=z['alive']
        else:
            # Diagnostic joint order comes from the official Isaac witness.
            names=json.loads((ROOT/'records/engineai_single_v1/witness.json').read_text())['joints'];fps=50.
            body=json.loads((ROOT/'records/engineai_single_v1/witness.json').read_text())['bodies'].index('LINK_BASE')
            q=np.c_[z['actual_root_pos'],z['body_quat_w'][:,body],z['actual_joint_pos']];alive=np.ones(len(q),bool)
    tree=ET.parse(ROOT/'cache/engineai_overlay/pm01_kinematic.xml');world=tree.getroot().find('worldbody')
    ET.SubElement(world,'geom',type='plane',size='6 6 .1',rgba='.82 .84 .87 1')
    model=mujoco.MjModel.from_xml_string(ET.tostring(tree.getroot(),encoding='unicode'));data=mujoco.MjData(model)
    indexes=[names.index(model.joint(i).name) for i in range(1,model.njnt)]
    renderer=mujoco.Renderer(model,height=480,width=640);camera=mujoco.MjvCamera();camera.distance=3.3;camera.azimuth=135;camera.elevation=-15
    opt=mujoco.MjvOption();opt.geomgroup[3]=1
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with imageio.get_writer(str(a.output),fps=25,codec='libx264',quality=7) as writer:
        for t in range(0,len(q),max(1,round(fps/25))):
            if a.kind=='isaac_ppo' and not alive[t]:break
            data.qpos[:7]=q[t,:7];data.qpos[7:]=q[t,7:][indexes];mujoco.mj_forward(model,data)
            camera.lookat[:]=[q[0,0],q[0,1],.65]
            renderer.update_scene(data,camera,scene_option=opt)
            im=Image.fromarray(renderer.render());draw=ImageDraw.Draw(im);draw.rectangle((0,0,640,53),fill=(20,25,35))
            draw.text((12,8),f'PM01 | {a.kind} | t={t/fps:.2f}s',fill='white')
            draw.text((12,29),'REFERENCE ONLY' if a.kind=='reference' else 'Recorded Isaac physics states; SDK geometry visualization',fill='white')
            writer.append_data(np.asarray(im))
    renderer.close();print('VIDEO_SAVED',a.output)
if __name__=='__main__':main()
