"""All four environment0 replays: reference, primary, no-push and nominal."""
import hashlib
import json
from pathlib import Path
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image,ImageDraw
from combat.robot import Robot

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'runs/pbhc_beyondmimic_20260916'
OUT=ROOT/'runs/pbhc_followup_20260916'
dest=OUT/'videos';dest.mkdir(parents=True,exist_ok=True)
manifest=json.loads((ROOT/'records/pbhc_followup_20260916/manifest.json').read_text())
robot=Robot();model,data=robot.model,robot.data
cam=mujoco.MjvCamera();cam.distance=3.2;cam.azimuth=135;cam.elevation=-15
results=[]
with mujoco.Renderer(model,height=240,width=320) as renderer:
    for name in sorted(set(j['name'] for j in manifest['jobs'])):
        target=dest/(name+'.mp4');receipt=dest/(name+'.json')
        if target.exists() and receipt.exists():results.append(json.loads(receipt.read_text()));continue
        if target.exists():raise FileExistsError('Partial video needs inspection')
        paths=[BASE/'evaluation'/name/'model_1999',
               OUT/'evaluation'/(name+'_no_push_model_1999'),OUT/'evaluation'/(name+'_nominal_model_1999')]
        traces=[np.load(p/'rollout.npz',allow_pickle=False) for p in paths]
        order=[list(z['joint_names']).index(n) for n in robot.names for z in traces[:1]]
        for z in traces[1:]:np.testing.assert_allclose(z['reference'],traces[0]['reference'],atol=1e-5)
        lengths=[int(z['alive'][:,0].sum()) for z in traces];assert min(lengths)>0
        with imageio.get_writer(target,fps=25,codec='libx264',quality=7) as writer:
            for t in range(0,len(traces[0]['qpos']),2):
                panels=[]
                for col,label in enumerate(['REFERENCE','RANDOMIZED','NO PUSH','NOMINAL']):
                    z=traces[max(0,col-1)];frame=t if col==0 else min(t,lengths[col-1]-1)
                    q=z['reference' if col==0 else 'qpos'][frame]
                    data.qpos[:7]=q[:7];data.qpos[7:]=q[7:][order];data.qvel[:]=0
                    mujoco.mj_forward(model,data)
                    cam.lookat[:]=[traces[0]['reference'][0,0],traces[0]['reference'][0,1],.7]
                    renderer.update_scene(data,cam)
                    image=Image.fromarray(renderer.render());draw=ImageDraw.Draw(image);draw.rectangle((0,0,320,44),fill=(20,30,45))
                    draw.text((5,4),label+' | '+name,fill='white')
                    label2='TERMINATED: FROZEN' if col and t>=lengths[col-1] else f't={(frame+1)/50:.2f}s | env0'
                    draw.text((5,24),label2,fill='white');panels.append(np.asarray(image))
                picture=np.concatenate(panels,axis=1);writer.append_data(picture)
                if t==20:imageio.imwrite(dest/(name+'.png'),picture)
        record={'name':name,'file':str(target),'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
                'scope':'Visual rendering of recorded Isaac states, no MuJoCo policy re-execution',
                'selection':'environment0, all four sources, no success selection','alive_frames':lengths}
        receipt.write_text(json.dumps(record,indent=2));results.append(record)
        for z in traces:z.close()
(dest/'COMPLETE.json').write_text(json.dumps(results,indent=2))
print('FOLLOWUP_REPLAYS_COMPLETE',len(results))
