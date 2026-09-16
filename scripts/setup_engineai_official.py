"""Build a pinned isolated environment; keep the old simulator/env untouched."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import hashlib
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REC = ROOT / 'records/engineai_official_v1'
SPEC = ROOT / 'configs/engineai_official_env.json'

def run(cmd, **kw):
    print('RUN', cmd, flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True, **kw)

def main():
    spec = json.loads(SPEC.read_text())
    REC.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(ROOT).free / 2**30
    if free < spec['storage']['minimum_free_gib']:
        raise RuntimeError(f'Insufficient storage reserve: {free:.1f} GiB')
    sha = hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    (REC / 'build_spec.json').write_text(json.dumps({'spec':spec,'canonical_sha256':sha,'free_gib_before':free}, indent=2))
    src = ROOT / 'third_party/IsaacLab-engineai-official'
    if not src.exists():
        url='https://codeload.github.com/isaac-sim/IsaacLab/tar.gz/'+spec['isaac_lab_commit']
        archive=ROOT/'cache/IsaacLab-engineai-official.tar.gz'
        print('Downloading pinned archive',url,flush=True)
        with urllib.request.urlopen(url,timeout=60) as response,archive.open('wb') as out:shutil.copyfileobj(response,out)
        src.mkdir()
        with tarfile.open(archive) as tar:
            for member in tar.getmembers():
                parts=Path(member.name).parts
                if len(parts)<2:continue
                dest=src.joinpath(*parts[1:])
                if src.resolve() not in dest.resolve().parents:raise ValueError('Unsafe archive path')
                if member.isdir():dest.mkdir(parents=True,exist_ok=True)
                elif member.isfile():
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    with tar.extractfile(member) as source,dest.open('wb') as target:shutil.copyfileobj(source,target)
        (src/'SOURCE.json').write_text(json.dumps({'commit':spec['isaac_lab_commit'],'url':url,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest()},indent=2))
    if json.loads((src/'SOURCE.json').read_text())['commit']!=spec['isaac_lab_commit']:raise RuntimeError('Wrong source revision')
    venv = ROOT / '.venv-engineai-official'
    if not venv.exists():
        run(['/data/code/ouyanghao/oyh/world-model-risk-planning/.env/bin/python','-m','venv',str(venv)])
    py = str(venv / 'bin/python')
    env = dict(os.environ)
    env.pop('PYTHONPATH',None)
    env.update(TMPDIR=str(ROOT/'cache/engineai-official/tmp'),PIP_CACHE_DIR=str(ROOT/'cache/engineai-official/pip'),PIP_DISABLE_PIP_VERSION_CHECK='1')
    Path(env['TMPDIR']).mkdir(parents=True,exist_ok=True)
    run([py,'-m','pip','install','--no-cache-dir','--index-url','https://pypi.org/simple','pip==25.0.1','setuptools==75.8.0','wheel==0.45.1','toml'],env=env)
    run([py,'-m','pip','install','--no-cache-dir','torch==2.7.0','torchvision==0.22.0','--index-url','https://download.pytorch.org/whl/cu128'],env=env)
    # Constrain the heavy packages so the resolver never upgrades CUDA/Torch implicitly.
    constraints = ROOT/'cache/engineai-official/constraints.txt'
    constraints.write_text('torch==2.7.0+cu128\ntorchvision==0.22.0+cu128\nnumpy==1.26.4\npin==2.7.0\nhpp-fcl==2.4.4\neigenpy==3.5.1\n')
    cmd=[py,'-m','pip','install','--no-cache-dir','--index-url','https://pypi.org/simple','-c',str(constraints)]
    for name in ['isaaclab','isaaclab_assets','isaaclab_tasks','isaaclab_rl']:
        cmd+=['-e',str(src/'source'/name)]
    cmd+=['rsl-rl-lib==5.0.1','onnxscript>=0.5','tensorboard','numpy==1.26.4','pin==2.7.0','hpp-fcl==2.4.4','wheel==0.45.1']
    wheelhouse=ROOT/'cache/engineai-official/wheels'
    if wheelhouse.exists():cmd+=['--find-links',str(wheelhouse)]
    run(cmd,env=env)
    (REC/'INSTALL_COMPLETE.json').write_text(json.dumps({'spec_sha256':sha,'free_gib_after':shutil.disk_usage(ROOT).free/2**30,'status':'installed; physics witness still required'},indent=2))

if __name__=='__main__':main()
