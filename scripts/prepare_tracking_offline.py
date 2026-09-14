"""Make narrowly scoped, audited offline entrypoints from pinned BeyondMimic.

The PPO algorithm, rewards, observations and robot configuration are unchanged.
Generated upstream-derived files stay in third_party and retain their headers.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / 'third_party/whole_body_tracking'


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'Upstream drift; expected one occurrence: {old[:90]}')
    return text.replace(old, new, 1)


def main():
    records = []
    train = UPSTREAM / 'scripts/rsl_rl/train.py'
    source = train.read_text(encoding='utf-8')
    result = replace_once(source,
        'parser.add_argument("--registry_name", type=str, required=True, help="The name of the wand registry.")',
        'parser.add_argument("--motion_file", type=str, required=True, help="Local reference NPZ; no external upload.")')
    result = replace_once(result,'app_launcher = AppLauncher(args_cli)',
                          'app_launcher = AppLauncher(args_cli, sync_loads=False, multi_gpu=False)')
    start = result.index('    # load the motion file from the wandb registry')
    end = result.index('    # specify directory for logging experiments', start)
    result = result[:start] + '''    # Local-only reference loading. No W&B API, artifact upload or telemetry.
    import pathlib
    motion_path = pathlib.Path(args_cli.motion_file).resolve(strict=True)
    registry_name = None
    env_cfg.commands.motion.motion_file = str(motion_path)
    agent_cfg.logger = "tensorboard"

''' + result[end:]
    output = train.with_name('train_local.py')
    output.write_text(result, encoding='utf-8')
    records.append({'source': str(train.relative_to(ROOT)), 'output': str(output.relative_to(ROOT)),
                    'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
                    'output_sha256': hashlib.sha256(result.encode()).hexdigest()})
    replay = UPSTREAM / 'scripts/csv_to_npz.py'
    source = replay.read_text(encoding='utf-8')
    result = replace_once(source,
        'parser.add_argument("--output_name", type=str, required=True, help="The name of the motion npz file.")',
        'parser.add_argument("--output_file", type=str, required=True, help="Local output NPZ.")')
    result = replace_once(result,'app_launcher = AppLauncher(args_cli)',
                          'app_launcher = AppLauncher(args_cli, sync_loads=False, multi_gpu=False)')
    result = replace_once(result,
        '            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",',
        '            # No remote sky texture needed for motion conversion.')
    start = result.index('            np.savez("/tmp/motion.npz", **log)')
    end = result.index('\n\n\ndef main():', start)
    result = result[:start] + '''            from pathlib import Path
            target = Path(args_cli.output_file).resolve()
            if target.exists():
                raise FileExistsError(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target, **log)
            print(f"LOCAL_REFERENCE_SAVED {target} frames={len(log['joint_pos'])}", flush=True)
            break
''' + result[end:]
    output = replay.with_name('csv_to_npz_local.py')
    output.write_text(result, encoding='utf-8')
    records.append({'source': str(replay.relative_to(ROOT)), 'output': str(output.relative_to(ROOT)),
                    'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
                    'output_sha256': hashlib.sha256(result.encode()).hexdigest()})
    dest = ROOT / 'records/paired_v1/tracking_offline_adaptations.json'
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({'changes': records, 'algorithm_changed': False,
                               'note': 'Replay assigns states kinematically; it is not a tracking success test.'}, indent=2), encoding='utf-8')
    print(dest)


if __name__ == '__main__':
    main()
