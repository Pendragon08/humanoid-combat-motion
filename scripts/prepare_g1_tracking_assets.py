"""Extract only the official tracking release's G1 resources; record hashes."""
import hashlib
import json
from pathlib import Path
import tarfile

root=Path(__file__).resolve().parents[1]
archive=root/'cache/downloads/unitree_description.tar.gz'
dest=root/'third_party/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets'
selected=[]
with tarfile.open(archive,'r:gz') as tar:
    for member in tar.getmembers():
        if not (member.name.startswith('unitree_description/urdf/g1/') or
                member.name.startswith('unitree_description/meshes/g1/') or
                member.name == 'unitree_description/package.xml'):
            continue
        target=(dest/member.name).resolve()
        if not target.is_relative_to(dest.resolve()) or not (member.isfile() or member.isdir()):
            raise ValueError('Unsafe or unsupported archive member: '+member.name)
        if member.isdir(): target.mkdir(parents=True,exist_ok=True);continue
        content=tar.extractfile(member).read()
        if target.exists() and target.read_bytes()!=content:raise FileExistsError('Different existing asset: '+str(target))
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content)
        selected.append({'path':str(target.relative_to(root)),'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()})
record={'source':'https://storage.googleapis.com/qiayuanl_robot_descriptions/unitree_description.tar.gz',
        'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'files':selected,
        'note':'Vendor resources remain excluded from repository; hashes only.'}
(root/'records/paired_v1/g1_tracking_assets.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print('G1_ASSETS_EXTRACTED',len(selected))
