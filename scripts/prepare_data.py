"""Schema-checked raw G1 references -> reproducible, source-disjoint windows."""
import json
import hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from combat.robot import Robot, ROOT, make_scene

def main():
    make_scene()
    robot = Robot()
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    split = {s: [] for s in ["train", "validation", "test"]}
    records = []
    exclusions = []
    original_fps = {}
    window = 64
    for entry in manifest["motions"]:
        path = ROOT / entry["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
        z = np.load(path, allow_pickle=False)
        assert "qpos" in z.files and "fps" in z.files, (path, z.files)
        fps = float(np.asarray(z["fps"]).reshape(-1)[0])
        q = robot.remap(z["qpos"])
        original_fps[entry["id"]] = fps
        assert 10 <= fps <= 240, (path, fps)
        if abs(fps-30.) > 1e-6:
            old = np.arange(len(q))/fps
            new = np.arange(int(np.floor(old[-1]*30))+1)/30.
            interp = np.stack([np.interp(new, old, q[:, j]) for j in range(36)], axis=1)
            rotations = Rotation.from_quat(q[:, [4, 5, 6, 3]])
            interp[:, [4, 5, 6, 3]] = Slerp(old, rotations)(new).as_quat()
            q = interp
        if len(q) < window:
            exclusions.append({"id": entry["id"], "reason": "shorter than 64 frames", "frames": len(q)})
            continue
        starts = np.arange(0, len(q)-window+1, 32)
        if len(starts) > 48:
            starts = starts[np.linspace(0, len(starts)-1, 48).astype(int)]
        style = 2 if "太极" in entry["title"] or "太極" in entry["title"] else 1 if "南拳" in entry["title"] else 0
        for start in starts:
            clip = q[start:start+window].copy()
            clip[:, :2] -= clip[0, :2]
            if np.any(np.abs(np.diff(clip[:, :3], axis=0)) > .5):
                exclusions.append({"id": entry["id"], "start": int(start), "reason": "root jump >0.5m in one 30Hz frame"})
                continue
            # A single common initial-height calibration for every arm; do not spawn
            # some baselines inside the ground because retarget and simulator assets differ.
            d = robot.forward(clip[0])
            feet = np.flatnonzero(np.isin(robot.model.geom_bodyid, robot.foot_bodies) & (robot.model.geom_type == 2) & (robot.model.geom_contype > 0))
            assert len(feet) == 8
            lowest = np.min(d.geom_xpos[feet, 2] - robot.model.geom_size[feet, 0])
            ground_shift = float(np.clip(.002-lowest, -.12, .12))
            clip[:, 2] += ground_shift
            idx = len(records)
            split[entry["split"]].append(idx)
            records.append({"qpos": clip.astype(np.float32), "id": entry["id"], "start": int(start), "style": style, "group": entry["source_group"], "initial_ground_shift_m": ground_shift})
    arrays = {}
    groups = {}
    for s, ids in split.items():
        assert ids, "Empty split " + s
        arrays[s] = np.stack([records[i]["qpos"] for i in ids])
        arrays[s+"_style"] = np.array([records[i]["style"] for i in ids], dtype=np.int64)
        arrays[s+"_ids"] = np.array([records[i]["id"] for i in ids])
        arrays[s+"_starts"] = np.array([records[i]["start"] for i in ids])
        arrays[s+"_initial_ground_shift_m"] = np.array([records[i]["initial_ground_shift_m"] for i in ids])
        groups[s] = sorted({records[i]["group"] for i in ids})
    assert not set(groups["train"]) & set(groups["test"])
    assert not set(groups["train"]) & set(groups["validation"])
    assert not set(groups["validation"]) & set(groups["test"])
    train = arrays["train"][:, :, 7:]
    arrays["mean"] = train.mean(axis=(0, 1))
    arrays["std"] = np.maximum(train.std(axis=(0, 1)), .1)
    np.savez_compressed(ROOT / "data/windows.npz", **arrays)
    audit = {"window_frames": window, "fps": 30, "style_labels": ["Changquan / 长拳", "Nanquan / 南拳", "Taijiquan / 太极拳"], "counts": {s: len(arrays[s]) for s in split}, "source_groups": groups, "exclusions": exclusions, "source_clip_count": len(manifest["motions"]), "raw_frame_total": sum(len(np.load(ROOT/e["path"], allow_pickle=False)["qpos"]) for e in manifest["motions"]), "robot": robot.metadata(), "limits": "Source video titles used for separation; same actor or routine may occur in distinct videos. Not a certified actor-disjoint benchmark."}
    (ROOT / "audit/data_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    audit['native_fps_by_clip'] = original_fps
    audit['resampling'] = '30Hz; linear positions/joints, quaternion SLERP; timestamps preserved. Actual NPZ fps field overrides README example.'
    audit['initial_ground_alignment'] = 'For every window and all experimental arms, apply one fixed vertical offset placing the lowest initial foot collision sphere 2mm above ground, clipped to +/-12cm. No per-frame grounding of the video baseline. This controls initial asset-ground mismatch.'
    (ROOT / "audit/data_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print("DATA_PREPARED", audit["counts"], "raw_frames", audit["raw_frame_total"])
    print("MODEL", robot.model.nq, robot.model.nv, "TORQUE", robot.torque_limits)

if __name__ == "__main__":
    main()
