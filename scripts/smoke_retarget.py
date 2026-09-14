"""Seeded native G1 inverse-kinematics witness, no dynamics claim."""
import json
from pathlib import Path
import numpy as np
import mujoco
import mink


def main():
    np.random.seed(0)
    root=Path(__file__).resolve().parents[1]
    model=mujoco.MjModel.from_xml_path(str(root/'third_party/GMR/assets/unitree_g1/g1_mocap_29dof.xml'))
    configuration=mink.Configuration(model)
    task=mink.FrameTask('left_wrist_yaw_link','body',position_cost=1.0,orientation_cost=0.0,lm_damping=1.0)
    target=configuration.get_transform_frame_to_world('left_wrist_yaw_link','body')
    desired=target.translation()+np.array([.02,0,.01])
    task.set_target(mink.SE3.from_rotation_and_translation(target.rotation(),desired))
    before=float(np.linalg.norm(configuration.get_transform_frame_to_world('left_wrist_yaw_link','body').translation()-desired))
    for _ in range(40):
        velocity=mink.solve_ik(configuration,[task],.01,'daqp',damping=1e-3,limits=[mink.ConfigurationLimit(model)])
        configuration.integrate_inplace(velocity,.01)
    after=float(np.linalg.norm(configuration.get_transform_frame_to_world('left_wrist_yaw_link','body').translation()-desired))
    print(json.dumps({'position_error_before_m':before,'position_error_after_m':after}),flush=True)
    if not np.isfinite(configuration.data.qpos).all() or after>=before*.1:raise RuntimeError('IK failed witness')
    result={'witness':'GMR_IK_WITNESS','nq':model.nq,'nv':model.nv,'before_m':before,'after_m':after,
            'mujoco':mujoco.__version__,'solver':'daqp','learned_tracking':False,'physics_evaluation':False}
    dest=root/'records/paired_v1/retarget_environment_witness.json';dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(result,indent=2),encoding='utf-8');print('GMR_IK_WITNESS '+json.dumps(result),flush=True)


if __name__=='__main__':main()
