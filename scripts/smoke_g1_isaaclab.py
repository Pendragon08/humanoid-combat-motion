"""Real articulated G1 physics witness; not a learned tracking evaluation."""
import argparse
from pathlib import Path
from isaaclab.app import AppLauncher

parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,required=True)
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
launcher=AppLauncher(args,sync_loads=False,multi_gpu=False)
app=launcher.app

import json
import torch
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene,InteractiveSceneCfg
from isaaclab.utils import configclass
from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG
from isaacsim.core.api.objects import GroundPlane


@configclass
class Scene(InteractiveSceneCfg):
    robot=G1_CYLINDER_CFG.replace(prim_path='{ENV_REGEX_NS}/Robot')


def main():
    torch.manual_seed(0)
    sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=.005,device=args.device))
    GroundPlane('/World/Ground',size=20,z_position=0)
    scene_cfg=Scene(num_envs=4,env_spacing=3)
    scene_cfg.robot.spawn.usd_dir=str(Path(__file__).resolve().parents[1]/'cache/usd/g1_tracking')
    scene=InteractiveScene(scene_cfg)
    sim.reset()
    robot=scene['robot']
    initial=robot.data.root_pos_w.detach().cpu().clone()
    for _ in range(200):
        robot.set_joint_position_target(robot.data.default_joint_pos)
        scene.write_data_to_sim();sim.step(render=False);scene.update(.005)
    if not torch.isfinite(robot.data.joint_pos).all():raise FloatingPointError('Nonfinite G1 state')
    # Seeded GPU backward witness also exercises the exact training interpreter.
    net=torch.nn.Linear(29,16,device=args.device)
    loss=net(robot.data.joint_pos.detach()).square().mean();loss.backward()
    result={'witness':'G1_ISAACLAB_PHYSICS_WITNESS','physics_steps':200,'robots':4,
            'initial_root_position':initial.tolist(),'final_root_position':robot.data.root_pos_w.detach().cpu().tolist(),
            'joint_names':robot.joint_names,'body_names':robot.body_names,'gradient_loss':loss.item(),
            'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),
            'learned_tracking':False,'claim':'articulation, contact simulator and CUDA backward only; no learned policy'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('G1_ISAACLAB_PHYSICS_WITNESS '+json.dumps(result),flush=True)


try:
    main()
except Exception:
    import traceback
    import sys
    failure=traceback.format_exc()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.with_suffix('.error.txt').write_text(failure,encoding='utf-8')
    print(failure,file=sys.stderr,flush=True)
    raise
finally:
    # Remove Isaac Lab's timeline stop callback before SimulationApp closes the
    # stage; otherwise the callback intentionally enters its paused render loop.
    sim_utils.SimulationContext.clear_instance()
    # This headless witness creates no Replicator writers or render products.
    import faulthandler
    faulthandler.dump_traceback_later(20, repeat=False)
    app.close(wait_for_replicator=False)
    faulthandler.cancel_dump_traceback_later()
