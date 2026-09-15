"""Two fixed references, environment0, actual Isaac states; no success selection."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import mujoco
import imageio.v2 as imageio
from combat.robot import Robot
from scripts.render_results import panel

p = argparse.ArgumentParser()
p.add_argument('--comparison', choices=['interface', 'sampling'], required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--selected-four', action='store_true', help='Show every selected training reference, in frozen order')
a = p.parse_args()
if a.output.exists():
    raise FileExistsError(a.output)
jobs = ['single_upstream', 'single_uniform'] if a.comparison == 'interface' else ['four_uniform', 'four_adaptive']
refs = [('eval', 'sample_000'), ('dev', 'sample_000')]
if a.selected_four:
    protocol = json.loads(Path('runs/tracking_diagnostics_20260915_refs/protocol.json').read_text())
    refs = [('eval', r['clip_id']) for r in protocol['references']]
directories = [[Path(f'runs/tracking_diagnostic_{split}_{job}_{sample}') for job in jobs] for split, sample in refs]
records = [[np.load(path / 'rollout_env0.npz', allow_pickle=False) for path in row] for row in directories]
for row in records:
    np.testing.assert_allclose(row[0]['reference_qpos'], row[1]['reference_qpos'], atol=1e-5)
    np.testing.assert_array_equal(row[0]['joint_names'], row[1]['joint_names'])
robot = Robot(); model, data = robot.model, robot.data
mapping = [list(records[0][0]['joint_names']).index(n) for n in robot.names]
fps = float(records[0][0]['fps'])
alive = [[int(z['alive'].sum()) for z in row] for row in records]
if min(v for row in alive for v in row) < 1:
    raise ValueError('Missing initial state')
frames = min(len(z['qpos']) for row in records for z in row)
a.output.mkdir(parents=True)
camera = mujoco.MjvCamera(); camera.distance = 3.6; camera.azimuth = 145; camera.elevation = -12
with mujoco.Renderer(model, height=300 if a.selected_four else 360, width=400 if a.selected_four else 480) as renderer:
    with imageio.get_writer(a.output / 'comparison.mp4', fps=fps/4, codec='libx264', quality=8, macro_block_size=16) as writer:
        for t in range(frames):
            panels = []
            for row_id, row in enumerate(records):
                camera.lookat[:] = [row[0]['reference_qpos'][0, 0], row[0]['reference_qpos'][0, 1], 1.0]
                for column in range(3):
                    reference = column == 0
                    z = row[max(0, column-1)]
                    frame = t if reference else min(t, alive[row_id][column-1]-1)
                    q = z['reference_qpos' if reference else 'qpos'][frame]
                    data.qpos[:7] = q[:7]; data.qpos[7:] = q[7:][mapping]; data.qvel[:] = 0
                    mujoco.mj_forward(model, data); renderer.update_scene(data, camera)
                    prefix = ('TRAIN ' + refs[row_id][1].split('_')[-1]) if a.selected_four else ('SHARED TRAIN' if row_id == 0 else 'DEVELOPMENT')
                    title = prefix + ': ' + ('REFERENCE' if reference else jobs[column-1])
                    frozen = not reference and t >= alive[row_id][column-1]
                    status = 'FIRST FAILURE: FROZEN' if frozen else f't={frame/fps:.2f}s | 4x slow'
                    panels.append(panel(renderer.render(), title, status))
            picture = np.concatenate([np.concatenate(panels[i:i+3], axis=1) for i in range(0, len(panels), 3)], axis=0)
            writer.append_data(picture)
            if t == 18:
                imageio.imwrite(a.output / 'preview.png', picture)
sources = [path for row in directories for path in row]
(a.output / 'provenance.json').write_text(json.dumps({'comparison': a.comparison,
    'source_evaluations': [str(path) for path in sources],
    'source_rollout_sha256': [hashlib.sha256((path/'rollout_env0.npz').read_bytes()).hexdigest() for path in sources],
    'selection': ('all four selected training references in frozen order' if a.selected_four else 'first selected training reference and first development reference') + '; environment0; seed0; no success selection',
    'alive_frames': alive, 'playback_speed': .25, 'freeze_after_first_failure': True,
    'rendering': 'MuJoCo rendering of recorded Isaac states, not cross-simulator policy execution'}, indent=2))
print('TRACKING_DIAGNOSTIC_RENDER_COMPLETE', a.comparison)
