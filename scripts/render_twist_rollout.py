"""Render a saved physical rollout (not a fresh policy evaluation)."""
import argparse
from pathlib import Path
import numpy as np
import mujoco
import imageio
from PIL import Image,ImageDraw

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--label',required=True)
    a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
    z=np.load(a.run/'rollout.npz');model=mujoco.MjModel.from_xml_path(str(a.run/'scene.xml'))
    data=mujoco.MjData(model);renderer=mujoco.Renderer(model,height=544,width=960)
    cam=mujoco.MjvCamera();cam.azimuth=135;cam.elevation=-17;cam.distance=3.5 if z['qpos'].shape[1]==2 else 2.7
    with imageio.get_writer(a.output,fps=50) as w:
        for t,q in enumerate(z['qpos']):
            data.qpos[:]=q.flatten();mujoco.mj_forward(model,data)
            cam.lookat[:]=q[:,:3].mean(0);cam.lookat[2]=.7
            renderer.update_scene(data,camera=cam);img=Image.fromarray(renderer.render());draw=ImageDraw.Draw(img)
            draw.rectangle((0,0,960,64),fill=(20,30,45))
            draw.text((12,8),a.label,fill='white')
            draw.text((12,27),f'Saved actual TWIST physics rollout | t={(t+1)/50:.2f}s | not a kinematic target animation',fill='white')
            draw.text((12,46),'Joint MAE(rad): '+str(np.round(z['joint_error'][t],3))+' | Root error(m): '+str(np.round(z['root_error'][t],3)),fill='white')
            if t==len(z['qpos'])//3:img.save(a.output.with_suffix('.jpg'))
            w.append_data(np.array(img))
    renderer.close();print('TWIST_VIDEO',a.output)

if __name__=='__main__':main()
