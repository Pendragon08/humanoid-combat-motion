"""Render episode 0 from the completed official SampleEps recordings.

The MuJoCo model is used only for forward-kinematic display of states already
recorded in Isaac Gym. This is not a MuJoCo policy rollout or Sim-to-Sim test.
"""

import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import joblib
import mujoco
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
UP = ROOT / "third_party/PBHC"
RUN = ROOT / "runs/pbhc_official_reproduction_20260917/official_horse_punch"
DEST = ROOT / "reports/media/pbhc_official_20260918"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_episode(folder):
    files = sorted(folder.glob("SampleEps_*_1000x332-332.pkl"))
    if len(files) != 1:
        raise RuntimeError(f"Expected exactly one 1000-episode source in {folder}: {files}")
    source = files[0]
    episode = joblib.load(source)["motion0"]
    arrays = {key: np.asarray(episode[key]) for key in ("root_trans_offset", "root_rot", "dof", "terminate")}
    if not (len(arrays["dof"]) == len(arrays["root_rot"]) == len(arrays["root_trans_offset"]) == 332):
        raise RuntimeError("Unexpected official evaluation shape")
    return source, arrays, float(episode["fps"])


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    video = DEST / "horse_stance_punch_provided_vs_retrained.mp4"
    manifest = DEST / "horse_stance_punch_provided_vs_retrained.json"
    if video.exists() or manifest.exists():
        raise FileExistsError("Existing result must be reviewed before rerendering")

    provided_source, provided, fps = load_episode(UP / "example/pretrained_horse_stance_punch/motions")
    retrained_source, retrained, fps2 = load_episode(RUN / "motions")
    if abs(fps - 50) > 0.01 or abs(fps2 - fps) > 0.01:
        raise RuntimeError(f"Unexpected frame rates: {fps}, {fps2}")

    model = mujoco.MjModel.from_xml_path(str(UP / "description/robots/g1/g1_23dof_lock_wrist.xml"))
    if model.nq != 30 or model.nu != 23:
        raise RuntimeError(f"Unexpected model: nq={model.nq}, nu={model.nu}")
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.azimuth = 135
    camera.elevation = -13
    camera.distance = 2.65
    camera.lookat[:] = np.r_[retrained["root_trans_offset"][0, :2], 0.75]
    panels = [("AUTHOR CHECKPOINT", provided), ("NEW 50K-UPDATE CHECKPOINT", retrained)]
    first_terminate = [int(np.flatnonzero(x["terminate"])[0]) if np.any(x["terminate"]) else None for _, x in panels]

    with mujoco.Renderer(model, width=480, height=480) as renderer:
        with imageio.get_writer(video, fps=25, codec="libx264", quality=7, macro_block_size=16) as writer:
            for t in range(0, 332, 2):
                images = []
                for (label, x), term in zip(panels, first_terminate):
                    pos = x["root_trans_offset"][t]
                    quat_xyzw = x["root_rot"][t]
                    data.qpos[:3] = pos
                    data.qpos[3:7] = np.r_[quat_xyzw[3], quat_xyzw[:3]]
                    data.qpos[7:] = x["dof"][t]
                    data.qvel[:] = 0
                    mujoco.mj_forward(model, data)
                    renderer.update_scene(data, camera)
                    im = Image.fromarray(renderer.render())
                    draw = ImageDraw.Draw(im)
                    draw.rectangle((0, 0, 480, 43), fill=(18, 30, 45))
                    draw.text((10, 7), f"{label} | {t / fps:.2f} s", fill="white")
                    if term is not None and t >= term:
                        draw.text((10, 25), "Recording termination flag reached", fill="#ffcc99")
                    images.append(np.asarray(im))
                writer.append_data(np.concatenate(images, axis=1))

    record = {
        "video": str(video.relative_to(ROOT)),
        "video_sha256": sha(video),
        "selection": "motion0 from each of the two completed official 1000-episode SampleEps outputs, selected before viewing",
        "sources": [str(provided_source.relative_to(ROOT)), str(retrained_source.relative_to(ROOT))],
        "source_sha256": [sha(provided_source), sha(retrained_source)],
        "source_fps": fps,
        "video_fps": 25,
        "frames": 166,
        "first_termination_frame": first_terminate,
        "rendering": "MuJoCo forward-kinematic display of recorded Isaac Gym states; no MuJoCo policy execution",
        "script_sha256": sha(Path(__file__)),
    }
    manifest.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print("PBHC_OFFICIAL_RESULT_VIDEO", json.dumps(record))


if __name__ == "__main__":
    main()
