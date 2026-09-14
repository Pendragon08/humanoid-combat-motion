"""Pinned G1 model, explicit joint mapping and separate kinematic/dynamic diagnostics."""
from pathlib import Path
import json
import xml.etree.ElementTree as ET
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[1]

def make_scene():
    tree = ET.parse(ROOT / "assets/g1/g1_29dof.xml")
    root = tree.getroot()
    ET.SubElement(root, "option", timestep="0.002", gravity="0 0 -9.81", integrator="implicitfast", iterations="50")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1920", offheight="1080", azimuth="135", elevation="-15")
    ET.SubElement(visual, "headlight", diffuse="0.7 0.7 0.7", ambient="0.3 0.3 0.3", specular="0.1 0.1 0.1")
    asset = root.find("asset")
    ET.SubElement(asset, "texture", name="checker", type="2d", builtin="checker", rgb1=".18 .23 .3", rgb2=".26 .31 .38", width="512", height="512")
    ET.SubElement(asset, "material", name="floor", texture="checker", texrepeat="4 4", reflectance=".05")
    world = root.find("worldbody")
    ET.SubElement(world, "geom", name="floor", type="plane", size="200 200 .1", material="floor", friction=".8 .005 .0001")
    ET.SubElement(world, "light", pos="1 -2 4", dir="-1 1 -3", diffuse=".8 .8 .8")
    # Mark physically relevant end points; sites are visual/measurement markers only.
    for side in ["left", "right"]:
        foot = world.find(f".//body[@name='{side}_ankle_roll_link']")
        ET.SubElement(foot, "site", name=f"{side}_sole", pos=".035 0 -.035", size=".008", rgba="0 .8 .8 1")
        hand = world.find(f".//body[@name='{side}_wrist_yaw_link']")
        ET.SubElement(hand, "site", name=f"{side}_fist", pos=".10 0 0", size=".02", rgba=".9 .3 .1 1")
    target = ROOT / "assets/g1/pilot_scene.xml"
    tree.write(target, encoding="unicode")
    return target

class Robot:
    def __init__(self):
        path = ROOT / "assets/g1/pilot_scene.xml"
        if not path.exists():
            path = make_scene()
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)
        m = self.model
        assert (m.nq, m.nv, m.nu) == (36, 35, 29), (m.nq, m.nv, m.nu)
        self.names = [m.joint(i).name for i in range(1, m.njnt)]
        assert len(self.names) == 29
        src = ET.parse(ROOT / "assets/gmr_source_order.xml")
        self.source_names = [j.attrib["name"] for j in src.findall(".//worldbody//joint") if j.get("type", "hinge") != "free"]
        assert set(self.names) == set(self.source_names)
        self.mapping = [self.source_names.index(x) for x in self.names]
        self.limits = m.jnt_range[1:].copy()
        urdf = ET.parse(ROOT / "assets/gmr_limits.urdf")
        vel = {j.attrib["name"]: float(j.find("limit").attrib["velocity"]) for j in urdf.findall("joint") if j.find("limit") is not None}
        self.velocity_limits = np.array([vel[n] for n in self.names])
        # A research screening threshold, not a manufacturer acceleration specification.
        self.acceleration_limits = np.full(29, 100.0)
        self.joint_to_actuator = np.array([list(m.actuator_trnid[:, 0]).index(i) for i in range(1, 30)])
        self.torque_limits = np.minimum(np.abs(m.actuator_ctrlrange[self.joint_to_actuator]).max(axis=1), np.abs(m.jnt_actfrcrange[1:]).max(axis=1))
        self.feet = [m.site(f"{s}_sole").id for s in ["left", "right"]]
        self.fists = [m.site(f"{s}_fist").id for s in ["left", "right"]]
        self.foot_bodies = [m.body(f"{s}_ankle_roll_link").id for s in ["left", "right"]]

    def remap(self, qpos):
        q = np.asarray(qpos, dtype=np.float64).copy()
        assert q.ndim == 2 and q.shape[1] == 36 and np.isfinite(q).all()
        q[:, 7:] = qpos[:, 7:][:, self.mapping]
        norms = np.linalg.norm(q[:, 3:7], axis=1)
        assert np.min(norms) > .5 and np.max(norms) < 1.5
        q[:, 3:7] /= norms[:, None]
        # q and -q are equivalent; enforce continuous quaternion signs.
        for t in range(1, len(q)):
            if q[t, 3:7] @ q[t-1, 3:7] < 0:
                q[t, 3:7] *= -1
        return q

    def forward(self, q):
        self.data.qpos[:] = q
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        return self.data

    def velocities(self, q, fps):
        v = np.zeros((len(q), self.model.nv))
        for t in range(len(q)-1):
            mujoco.mj_differentiatePos(self.model, v[t], 1/fps, q[t], q[t+1])
        v[-1] = v[-2]
        return v

    def geometry(self, q):
        feet, fists, com, penetration, self_hit = [], [], [], [], []
        for pose in q:
            d = self.forward(pose)
            feet.append(d.site_xpos[self.feet].copy())
            fists.append(d.site_xpos[self.fists].copy())
            com.append(d.subtree_com[1].copy())
            penetration.append(max(0., -min([float(c.dist) for c in d.contact[:d.ncon]] or [0.])))
            self_hit.append(any(c.dist < -.002 and self.model.geom_bodyid[c.geom1] != 0 and self.model.geom_bodyid[c.geom2] != 0 for c in d.contact[:d.ncon]))
        return dict(feet=np.array(feet), fists=np.array(fists), com=np.array(com), penetration=np.array(penetration), self_collision=np.array(self_hit))

    def diagnostic_rollout(self, q, fps=30., record=False):
        """Free-base torque-limited PD, zero root assistance. This is NOT a learned policy.

        A negative result diagnoses this controller+motion pair, not all possible controllers.
        Init at reference pose and velocity; no pose resets during the rollout.
        """
        m = self.model
        d = mujoco.MjData(m)
        vel = self.velocities(q, fps)
        d.qpos[:] = q[0]
        d.qvel[:] = vel[0]
        mujoco.mj_forward(m, d)
        kp = np.array([100. if any(s in n for s in ["hip", "knee"]) else 40. if "ankle" in n else 60. if "waist" in n else 25. if "wrist" not in n else 8. for n in self.names])
        kd = 2 * np.sqrt(kp) * .15
        duration = (len(q)-1) / fps
        steps = int(np.ceil(duration/m.opt.timestep))
        joint_errors, root_errors, saturation = [], [], []
        poses, fall_time, next_frame = [], None, 0.
        numerical_failure = False
        for step in range(steps+1):
            t = min(d.time * fps, len(q)-1)
            lo = min(int(t), len(q)-2)
            a = np.clip(t-lo, 0, 1)
            target = (1-a)*q[lo, 7:] + a*q[lo+1, 7:]
            target_v = (1-a)*vel[lo, 6:] + a*vel[lo+1, 6:]
            command = kp*(target-d.qpos[7:]) + kd*(target_v-d.qvel[6:])
            saturation.append(np.mean(np.abs(command) > self.torque_limits))
            d.ctrl[self.joint_to_actuator] = np.clip(command, -self.torque_limits, self.torque_limits)
            # No virtual force at the floating root; actuator torques are the only controls.
            before_step_time = float(d.time)
            mujoco.mj_step(m, d)
            numerical_failure |= bool(d.time < before_step_time or not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all())
            refroot = (1-a)*q[lo, :3]+a*q[lo+1, :3]
            joint_errors.append(np.mean((d.qpos[7:]-target)**2))
            root_errors.append(np.sum((d.qpos[:3]-refroot)**2))
            upright = float(d.xmat[1].reshape(3, 3)[2, 2])
            if fall_time is None and (d.qpos[2] < .35 or upright < .25 or not np.isfinite(d.qpos).all()):
                fall_time = float(d.time)
            if record and d.time >= next_frame:
                poses.append(d.qpos.copy())
                next_frame += 1/fps
        result = {"pd_joint_rmse_rad": float(np.sqrt(np.mean(joint_errors))), "pd_root_rmse_m": float(np.sqrt(np.mean(root_errors))), "pd_fall": fall_time is not None, "pd_fall_time_s": fall_time, "pd_numerical_failure": bool(numerical_failure), "pd_torque_saturation_fraction": float(np.mean(saturation)), "pd_full_horizon_success": bool(not numerical_failure and fall_time is None and np.sqrt(np.mean(joint_errors)) < .35 and np.sqrt(np.mean(root_errors)) < .25), "duration_s": duration}
        return result, np.array(poses)

    def metadata(self):
        return {"names": self.names, "source_names": self.source_names, "source_to_target_indices": self.mapping, "joint_limits_rad": self.limits.tolist(), "velocity_limits_rad_s": self.velocity_limits.tolist(), "velocity_source": "GMR bundled G1 URDF, model-version dependent", "acceleration_limits_rad_s2": self.acceleration_limits.tolist(), "acceleration_source": "research screening threshold; NOT verified hardware specification", "torque_limits_nm": self.torque_limits.tolist(), "joint_limit_source": "pinned Unitree MuJoCo G1 model", "quaternion": "wxyz", "frame": "z-up world", "units": "m, rad, s"}
