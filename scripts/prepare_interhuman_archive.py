"""Extract checksummed sources safely, recording unofficial annotation transport."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import zipfile


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()


def unpack(path,dest,expected,pattern):
    if sha(path)!=expected:raise ValueError('Archive hash mismatch: '+str(path))
    with zipfile.ZipFile(path) as z:
        members=[m for m in z.infolist() if not m.is_dir()]
        required=sum(m.file_size for m in members)
        if shutil.disk_usage(dest).free < required+8*1024**3:
            raise RuntimeError('Insufficient disk headroom (8GiB reserve)')
        if z.testzip() is not None:raise ValueError('ZIP CRC failure')
        for m in members:
            p=PurePosixPath(m.filename)
            if p.is_absolute() or '..' in p.parts or not re.fullmatch(pattern,m.filename):
                raise ValueError('Unexpected archive path: '+m.filename)
            target=(dest/str(p)).resolve()
            if dest.resolve() not in target.parents:raise ValueError('Path escaped target')
            raw=z.read(m)
            if target.exists():
                if hashlib.sha256(target.read_bytes()).digest()!=hashlib.sha256(raw).digest():
                    raise ValueError('Existing file differs: '+str(target))
            else:
                target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
    return len(members)


def main():
    p=argparse.ArgumentParser();p.add_argument('--motions',type=Path,required=True)
    p.add_argument('--annots',type=Path,required=True);p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True);a=p.parse_args()
    audit=json.loads(a.audit.read_text(encoding='utf-8'))
    if len(audit['records'])!=50 or not all(r.get('byte_identical') for r in audit['records']):
        raise ValueError('Annotation transport audit incomplete')
    a.data.mkdir(parents=True,exist_ok=True)
    n=unpack(a.motions,a.data,'ad951d24f95fff550d964c8f977eb99da47264b81e3080312c95046e69fd6325',r'motions_processed/person[12]/\d+\.npy')
    k=unpack(a.annots,a.data,audit['mirror_sha256'],r'annots/\d+\.txt')
    record={'motion_files':n,'annotation_files':k,'motion_archive_sha256':sha(a.motions),
            'motion_source':'official InterHuman Google Drive','annotation_transport':audit,
            'annotation_verification':'50/50 official files byte-identical; remaining mirror files not individually verified',
            'license':'Original InterHuman LICENSE.md governs. Mirror MIT tag is not adopted. No dataset redistribution.'}
    (a.data/'installation_provenance.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps({'motion_files':n,'annotation_files':k}),flush=True)


if __name__=='__main__':main()
