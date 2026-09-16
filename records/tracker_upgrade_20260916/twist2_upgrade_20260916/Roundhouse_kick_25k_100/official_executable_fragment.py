rpy = quatToEuler(quat)
obs_body_dof_vel = dof_vel.copy()
obs_body_dof_vel[self.ankle_idx] = 0.0
obs_proprio = np.concatenate([ang_vel * 0.25, rpy[:2], dof_pos - self.default_dof_pos, obs_body_dof_vel * 0.05, self.last_action])
obs_full = np.concatenate([action_mimic, obs_proprio])
obs_hist = np.array(self.proprio_history_buf).flatten()
self.proprio_history_buf.append(obs_full)
future_obs = action_mimic.copy()
obs_buf = np.concatenate([obs_full, obs_hist, future_obs])
assert obs_buf.shape[0] == self.total_obs_size, f'Expected {self.total_obs_size} obs, got {obs_buf.shape[0]}'
obs_tensor = torch.from_numpy(obs_buf).float().unsqueeze(0).to(self.device)
with torch.no_grad():
    raw_action = self.policy(obs_tensor).cpu().numpy().squeeze()
self.last_action = raw_action
raw_action = np.clip(raw_action, -10.0, 10.0)
scaled_actions = raw_action * self.action_scale
pd_target = scaled_actions + self.default_dof_pos
torque = (pd_target - dof_pos) * self.stiffness - dof_vel * self.damping
torque = np.clip(torque, -self.torque_limits, self.torque_limits)
