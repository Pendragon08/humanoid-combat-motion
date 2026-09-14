"""Validate and unpack the official pinned binary within this project only."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

root=Path(__file__).resolve().parents[1]
archive=root/'cache/downloads/isaac-sim-4.5.0.zip.partial'
dest=root/'third_party/isaacsim-4.5.0'
spec=json.loads((root/'env-isaaclab.json').read_text())
if dest.exists():raise FileExistsError('Refusing to overwrite existing simulator')
with zipfile.ZipFile(archive) as z:
    entries=z.infolist()
    if len(entries)!=spec['isaacsim_file_count'] or sum(i.file_size for i in entries)!=spec['isaacsim_uncompressed_bytes']:
        raise RuntimeError('Archive metadata differs from audited central directory')
    for entry in entries:
        resolved=(dest/entry.filename).resolve()
        if not resolved.is_relative_to(dest.resolve()):raise RuntimeError('Unsafe archive path')
    required=sum(i.file_size for i in entries)+8*1024**3
    if shutil.disk_usage(root).free<required:raise RuntimeError('Insufficient space with 8GiB reserve')
    bad=z.testzip()
    if bad:raise RuntimeError('Archive CRC failed: '+bad)
digest=hashlib.sha256()
with archive.open('rb') as f:
    for block in iter(lambda:f.read(1024**2),b''):digest.update(block)
dest.mkdir()
subprocess.run(['unzip','-q',str(archive),'-d',str(dest)],check=True)
record={'sha256':digest.hexdigest(),'bytes':archive.stat().st_size,'crc':'passed','installed_to':str(dest),'runtime_validation':'pending'}
(root/'records/paired_v1/isaac_binary.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record))
