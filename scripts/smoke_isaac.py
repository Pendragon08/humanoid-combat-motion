"""Actual gravity/contact rollout, not import-only validation."""
import json
import faulthandler
from pathlib import Path
faulthandler.enable()
faulthandler.dump_traceback_later(90,repeat=True)
from isaacsim import SimulationApp
print('ISAAC_STAGE constructor',flush=True)
experience=Path(__file__).resolve().parents[1]/'third_party/IsaacLab/apps/isaaclab.python.headless.kit'
app=SimulationApp({'headless':True,'width':128,'height':128,'multi_gpu':False,'sync_loads':False},experience=str(experience))
try:
    print('ISAAC_STAGE app_started',flush=True)
    import numpy as np
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import DynamicCuboid
    world=World(stage_units_in_meters=1.0,physics_dt=1/120,rendering_dt=1/60)
    print('ISAAC_STAGE world_created',flush=True)
    world.scene.add_ground_plane(restitution=0.0)
    cube=world.scene.add(DynamicCuboid(prim_path='/World/TestCube',name='cube',position=np.array([0.,0.,1.]),scale=np.array([.1,.1,.1]),mass=1.))
    print('ISAAC_STAGE reset',flush=True)
    world.reset()
    before=cube.get_world_pose()[0].copy()
    for _ in range(240):world.step(render=False)
    after=cube.get_world_pose()[0].copy()
    assert np.isfinite(after).all() and .03<after[2]<.08 and before[2]>.9,(before,after)
    print('ISAAC_PHYSICS_WITNESS '+json.dumps({'start':before.tolist(),'end':after.tolist(),'steps':240,'contact':'cube_on_ground','robot_tracking':False}),flush=True)
finally:
    faulthandler.cancel_dump_traceback_later()
    app.close()
