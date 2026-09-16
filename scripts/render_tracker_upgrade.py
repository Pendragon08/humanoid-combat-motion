"""Render saved executed states, never substitute reference animation."""
import argparse,json
from pathlib import Path
import mujoco,numpy as np,imageio
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    m=json.loads((a.run/'metrics.json').read_text());z=np.load(a.run/'rollout.npz')
    xml=Path(m['model_path']) if 'model_path' in m else ROOT/'third_party/TWIST2/assets/g1/g1_sim2sim_29dof.xml'
    model=mujoco.MjModel.from_xml_path(str(xml));model.vis.global_.offwidth=960;model.vis.global_.offheight=544
    d=mujoco.MjData(model);r=mujoco.Renderer(model,height=544,width=960);cam=mujoco.MjvCamera();cam.azimuth=135;cam.elevation=-17;cam.distance=3.
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with imageio.get_writer(a.output,fps=25) as writer:
        for t in range(0,len(z['qpos']),2):
            d.qpos[:]=z['qpos'][t];mujoco.mj_forward(model,d);cam.lookat[:]=d.qpos[:3];cam.lookat[2]=.7
            r.update_scene(d,camera=cam);im=Image.fromarray(r.render());draw=ImageDraw.Draw(im);draw.rectangle((0,0,960,56),fill=(20,30,45))
            draw.text((12,8),m['tracker']+' | '+a.run.name,fill='white')
            draw.text((12,30),f'Actual saved physics | t={(t+1)*.02:.2f}s | joint={z["joint_error"][t]:.3f}rad | root={z["root_error"][t]:.3f}m',fill='white')
            writer.append_data(np.asarray(im))
    r.close();print('SAVED_PHYSICS_VIDEO',a.output)
if __name__=='__main__':main()
