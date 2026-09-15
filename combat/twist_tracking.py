"""Offline TWIST inference adapter; official constants/functions extracted by AST.

Upstream MIT source is fetched/pinned separately. No Redis, GUI or robot SDK is
needed. Physics/PD/observation/history remain the upstream deployment contract.
"""
import ast
import copy
from collections import deque
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
import torch
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / 'third_party/TWIST'
XML = UPSTREAM / 'assets/g1/g1_sim2sim_with_wrist_roll.xml'

def official_contract():
    source = UPSTREAM / 'deploy_real/server_low_level_g1_sim.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    scope = {'np': np, 'deque': deque}
    params = UPSTREAM / 'deploy_real/data_utils/params.py'
    exec(compile(params.read_text(encoding='utf-8'), str(params), 'exec'), scope)
    rotfile = UPSTREAM / 'deploy_real/data_utils/rot_utils.py'
    quat_fn = next(n for n in ast.parse(rotfile.read_text(encoding='utf-8')).body
                   if isinstance(n, ast.FunctionDef) and n.name == 'quatToEuler')
    exec(compile(ast.Module(body=[quat_fn], type_ignores=[]), str(rotfile), 'exec'), scope)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in (
            'extract_mimic_obs_to_body_and_wrist', 'aggregate_wrist_dof_pos'):
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), scope)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    names = {'num_actions', 'sim_dt', 'sim_decimation', 'default_dof_pos',
             'stiffness', 'damping', 'torque_limits', 'action_scale', 'ankle_idx',
             'n_mimic_obs', 'n_proprio'}
    cfg = SimpleNamespace()
    scope['self'] = cfg
    for node in init.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Attribute) and node.targets[0].attr in names:
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), scope)
    assert cfg.n_proprio == 105 and cfg.num_actions == 23
    return cfg, scope

def scene(actors):
    tree = ET.parse(XML); root = tree.getroot()
    root.find('compiler').set('meshdir', str(XML.parent / 'meshes'))
    root.find('visual/global').set('offwidth', '960')
    root.find('visual/global').set('offheight', '544')
    # Keyframe is unused; its qpos length is specific to the single robot.
    for key in list(root.findall('keyframe')): root.remove(key)
    robot_world = next(w for w in root.findall('worldbody') if w.find('body') is not None)
    original = copy.deepcopy(robot_world.find('body')); robot_world.remove(robot_world.find('body'))
    sections = {name: list(root.find(name)) for name in ('actuator', 'sensor')}
    for name in sections: root.find(name).clear()
    for actor in range(actors):
        prefix = f'a{actor}_'
        body = copy.deepcopy(original)
        for e in body.iter():
            if 'name' in e.attrib: e.set('name', prefix + e.get('name'))
            if actor == 1 and e.tag == 'geom' and e.get('group') == '1': e.set('rgba', '.25 .45 .85 1')
        robot_world.append(body)
        for name, elems in sections.items():
            for original_elem in elems:
                e = copy.deepcopy(original_elem)
                for attr in ('name', 'joint', 'site', 'objname'):
                    if attr in e.attrib: e.set(attr, prefix + e.get(attr))
                root.find(name).append(e)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    assert model.nq == 32 * actors and model.nu == 25 * actors
    return model, ET.tostring(root, encoding='unicode')

class Controller:
    def __init__(self, model, actor, policy, device):
        self.cfg, self.scope = official_contract()
        self.prefix = f'a{actor}_'; self.policy = policy; self.device = device
        self.names = [model.joint(i).name[len(self.prefix):] for i in range(model.njnt)
                      if model.joint(i).name.startswith(self.prefix) and model.jnt_type[i] != 0]
        self.jids = np.array([model.joint(self.prefix+n).id for n in self.names])
        self.qids = model.jnt_qposadr[self.jids]; self.vids = model.jnt_dofadr[self.jids]
        self.aids = np.array([model.actuator(self.prefix+n).id for n in self.names])
        root_id = model.joint(self.prefix+'pelvis').id
        self.rootq = int(model.jnt_qposadr[root_id]); self.rootv = int(model.jnt_dofadr[root_id])
        self.body_ids = np.array([i for i, n in enumerate(self.names) if 'wrist' not in n])
        assert self.body_ids.tolist() == list(range(19)) + list(range(20,24))
        self.history = deque([np.zeros(105) for _ in range(10)], maxlen=10)
        self.last_action = np.zeros(23, dtype=np.float32)
        self.pd_target = self.scope['DEFAULT_MIMIC_OBS']['g1'][8:].copy()

    def target(self, data, mimic):
        cfg = self.cfg
        q = data.qpos[self.qids].astype(np.float32)
        v = data.qvel[self.vids].astype(np.float32)[self.body_ids]
        v[cfg.ankle_idx] = 0
        quat = data.sensor(self.prefix+'orientation').data.astype(np.float32)
        ang = data.sensor(self.prefix+'angular-velocity').data.astype(np.float32)
        rpy = self.scope['quatToEuler'](quat)
        proprio = np.concatenate([ang*.25, rpy[:2], q[self.body_ids]-cfg.default_dof_pos,
                                  v*.05, self.last_action])
        command, wrists = self.scope['extract_mimic_obs_to_body_and_wrist'](np.asarray(mimic,dtype=np.float32))
        full = np.concatenate([command, proprio])
        obs = np.concatenate([full, np.asarray(self.history).flatten()])
        assert obs.shape == (1155,) and np.isfinite(obs).all()
        self.history.append(full)
        with torch.inference_mode():
            action = self.policy(torch.tensor(obs, dtype=torch.float32, device=self.device)[None]).cpu().numpy().reshape(-1)
        assert action.shape == (23,) and np.isfinite(action).all()
        self.last_action = action.copy()
        body = np.clip(action,-10,10)*cfg.action_scale + cfg.default_dof_pos
        self.pd_target = self.scope['aggregate_wrist_dof_pos'](body, wrists)
        return obs

    def torque(self, data):
        # Upstream extract_data() casts both to float32 before PD arithmetic.
        raw = (self.pd_target-data.qpos[self.qids].astype(np.float32))*self.cfg.stiffness-data.qvel[self.vids].astype(np.float32)*self.cfg.damping
        data.ctrl[self.aids] = np.clip(raw,-self.cfg.torque_limits,self.cfg.torque_limits)

def reference(path, ctrl, seconds=5):
    if path is None:
        default = ctrl.scope['DEFAULT_MIMIC_OBS']['g1']
        n = int(seconds*50)+1
        q = np.tile(np.r_[0,0,default[0],1,0,0,0,default[8:]], (n,1))
        return {'qpos': q, 'mimic': np.tile(default,(n,1)), 'velocity': np.zeros((n,31)),
                'dropped_joints': [], 'source': 'official DEFAULT_MIMIC_OBS standing command'}
    z = np.load(path, allow_pickle=False)
    assert abs(float(np.asarray(z['fps']).item())-50) < 1e-6
    names = list(z['joint_names']); ids = [names.index(n) for n in ctrl.names]
    bodies = list(z['body_names']); pelvis = bodies.index('pelvis')
    pos = z['body_pos_w'][:,pelvis]; quat = z['body_quat_w'][:,pelvis]
    assert np.max(abs(np.linalg.norm(quat,axis=1)-1)) < 1e-3
    rot = Rotation.from_quat(quat[:,[1,2,3,0]])
    vel = rot.inv().apply(z['body_lin_vel_w'][:,pelvis])
    ang = rot.inv().apply(z['body_ang_vel_w'][:,pelvis])
    euler = np.array([ctrl.scope['quatToEuler'](r) for r in quat])
    joints = z['joint_pos'][:,ids]
    mimic = np.concatenate([pos[:,2:3],euler,vel,ang[:,2:3],joints],axis=1)
    velocity = np.concatenate([z['body_lin_vel_w'][:,pelvis],ang,z['joint_vel'][:,ids]],axis=1)
    return {'qpos': np.concatenate([pos,quat,joints],axis=1), 'mimic': mimic,
            'velocity': velocity, 'dropped_joints': [n for n in names if n not in ctrl.names], 'source': str(path)}
