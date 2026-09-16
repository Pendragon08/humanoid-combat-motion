"""Build a pinned, isolated compatibility overlay; never edit upstream or Isaac."""
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    src=ROOT/'third_party/engineai_rl_lab/source/engineai_rl_lab/engineai_rl_lab'
    dst=ROOT/'cache/engineai_overlay/engineai_rl_lab'
    if dst.exists(): raise FileExistsError(dst)
    shutil.copytree(src,dst,ignore=shutil.ignore_patterns('assets','__pycache__'))
    changes=[]
    for rel in ['__init__.py','tasks/__init__.py','tasks/tracking/config/pm01/__init__.py','tasks/tracking/config/t800/__init__.py']:
        (dst/rel).write_text('# Explicit task import; no automatic registration in compatibility overlay.\n')
        changes.append(rel)
    p=dst/'tasks/tracking/config/pm01/flat_env_cfg.py'
    s=p.read_text().replace('from engineai_rl_lab.tasks.tracking.config.pm01.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE','LOW_FREQ_SCALE = 0.5  # Unused low-frequency variant; avoid incompatible runner config import.')
    p.write_text(s);changes.append(str(p.relative_to(dst)))
    sdk=ROOT/'third_party/engineai_robotics_native_sdk/assets/resource/robot/pm01_edu/xml'
    tree=ET.parse(sdk/'serial_pm01_edu.xml');root=tree.getroot()
    # Expand includes. Primitive collision shapes and inertias stay unchanged.
    for parent in list(root.iter()):
        for inc in list(parent.findall('include')):
            file=inc.attrib['file']
            if file!='assets.xml':
                included=ET.parse(sdk/file).getroot()
                for child in list(included):parent.append(child)
            parent.remove(inc)
    for parent in list(root.iter()):
        for child in list(parent):
            if child.tag in ['asset','keyframe'] or (child.tag=='geom' and child.get('type')=='mesh'):
                parent.remove(child)
    for side,offset in [('L','0.03 -0.02 -0.14'),('R','0.03 0.02 -0.14')]:
        body=root.find(f".//body[@name='LINK_ELBOW_YAW_{side}']")
        ET.SubElement(body,'site',name=f'hand_{side}',pos=offset,size='0.01')
    dest=ROOT/'cache/engineai_overlay/pm01_kinematic.xml'
    tree.write(dest,encoding='unicode')
    (dst.parent/'overlay_manifest.json').write_text(json.dumps({'changes':changes,'model':'official SDK kinematics/inertias/primitive collisions, visual meshes removed; added hand target sites','spec':json.loads((ROOT/'configs/engineai_single_spec.json').read_text())},indent=2))
    print('ENGINEAI_OVERLAY_PREPARED',dst)

if __name__=='__main__': main()
