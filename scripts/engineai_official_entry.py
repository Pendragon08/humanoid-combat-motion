"""Activate isolated package .pth files while retaining Kit's own Python stdlib."""
from pathlib import Path
import runpy
import site
import sys
from engineai_official_packages import activate

root=Path(__file__).resolve().parents[1]
site.addsitedir(str(root/'.venv-engineai-official/lib/python3.10/site-packages'))
isolated=str(root/'.venv-engineai-official/lib/python3.10/site-packages')
sys.path.remove(isolated)
sys.path.insert(0,isolated)
activate()
# Kit's ML extension adds its own older dependency bundle during app startup.
# Resolve the declared numerical stack first, as in an external Isaac Python env.
import torch
import torchvision
import numpy
import gymnasium
assert torch.__version__=='2.7.0+cu128',torch.__file__
print('ISOLATED_RUNTIME',sys.version,torch.__version__,torch.__file__,flush=True)
script=sys.argv[1]
sys.argv=sys.argv[1:]
runpy.run_path(script,run_name='__main__')
