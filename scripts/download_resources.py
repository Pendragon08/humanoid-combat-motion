"""Download only a pinned martial-arts subset and required G1 assets, not full datasets."""
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
import xml.etree.ElementTree as ET
import requests

ROOT = Path(__file__).resolve().parents[1]
HF_REV = "ff09f3795dc99bef73b21413b03fa8470917a22c"
UNITREE_REV = "1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d"
GMR_REV = "bb1bbe40774794fceb2a7c579a3464a28e68c844"
DATASET = "LuluCao/KungfuAthleteBot"
GROUND = "kungfu_athlete_for_g1_29dof/datasets/org_smooth/ground_smoothed"

def get(url):
    for attempt in range(4):
        try:
            r = requests.get(url, timeout=(15, 90))
            r.raise_for_status()
            return r.content
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))

def save(url, relative):
    p = ROOT / relative
    p.parent.mkdir(parents=True, exist_ok=True)
    content = get(url) if not p.exists() else p.read_bytes()
    if not p.exists():
        p.write_bytes(content)
    return {"path": str(relative), "url": url, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}

def main():
    (ROOT / "data").mkdir(exist_ok=True)
    hfraw = f"https://huggingface.co/datasets/{DATASET}/resolve/{HF_REV}/"
    title_record = save(hfraw + "Action%20File%20Title.csv", "data/source_titles.csv")
    text = (ROOT / "data/source_titles.csv").read_text(encoding="utf-8-sig").replace("\r\r\n", "\n")
    titles = list(csv.DictReader(io.StringIO(text)))
    listing = json.loads(get(f"https://huggingface.co/api/datasets/{DATASET}/tree/{HF_REV}/{GROUND}"))
    available = {x["path"].split("/")[-1] for x in listing}
    rows = []
    for row in titles:
        number = row["Data Number"].split(".")[0]
        title = row["Source Video Title"]
        if number not in available or not re.search("长拳|長拳|南拳|太极拳|太極拳", title) or re.search("剑|刀|棍|枪", title):
            continue
        group = re.sub(r"-Scene-.*", "", title)
        split = "test" if "长拳三段单练" in group else "validation" if "长拳五段单练" in group else "train"
        rows.append({"id": number, "title": title, "source_group": group, "split": split, "kind": "video_reconstructed_and_GMR_retargeted", "paired": False})

    def motion(row):
        num = row["id"]
        record = save(hfraw + f"{GROUND}/{num}/{num}.npz", f"data/raw/{num}.npz")
        print("DATA", num, row["split"], record["bytes"], flush=True)
        return {**row, **record}

    with ThreadPoolExecutor(max_workers=4) as pool:
        motions = list(pool.map(motion, rows))
    assetroot = f"https://raw.githubusercontent.com/unitreerobotics/unitree_mujoco/{UNITREE_REV}/"
    assets = [save(assetroot + "unitree_robots/g1/g1_29dof.xml", "assets/g1/g1_29dof.xml"), save(assetroot + "LICENSE", "assets/UNITREE_LICENSE")]
    xml = ET.parse(ROOT / "assets/g1/g1_29dof.xml")
    files = sorted({x.attrib["file"] for x in xml.findall(".//asset/mesh")})
    def mesh(name):
        return save(assetroot + "unitree_robots/g1/meshes/" + name, "assets/g1/meshes/" + name)
    with ThreadPoolExecutor(max_workers=4) as pool:
        assets += list(pool.map(mesh, files))
    assets += [save(f"https://raw.githubusercontent.com/YanjieZe/GMR/{GMR_REV}/assets/unitree_g1/g1_mocap_29dof.xml", "assets/gmr_source_order.xml"), save(f"https://raw.githubusercontent.com/YanjieZe/GMR/{GMR_REV}/assets/unitree_g1/g1_custom_collision_29dof.urdf", "assets/gmr_limits.urdf"), save(f"https://raw.githubusercontent.com/YanjieZe/GMR/{GMR_REV}/assets/unitree_g1/LICENSE", "assets/GMR_ASSET_LICENSE")]
    card = save(hfraw + "README.md", "data/DATASET_CARD.md")
    manifest = {"hf_revision": HF_REV, "unitree_revision": UNITREE_REV, "gmr_revision": GMR_REV, "source": "KungFuAthleteBot; public martial-arts video -> GVHMR -> GMR -> manually filtered G1 reference", "selection": "Ground folder intersect source-title regex; no weapon titles. Source-title split; performer identity unknown.", "dataset_license": "Apache-2.0 per pinned dataset card; upstream video and body-model terms still apply", "motions": motions, "assets": assets, "metadata": [title_record, card]}
    (ROOT / "data/manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print("DOWNLOAD_COMPLETE", len(motions), "motions", sum(x["bytes"] for x in motions + assets) / 1e6, "MB")

if __name__ == "__main__":
    main()
