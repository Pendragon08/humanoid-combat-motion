"""Keep declared numerical packages outside Kit's bundled dependency namespace."""
import importlib.abc
import importlib.machinery
from pathlib import Path
import sys

ISOLATED=str(Path(__file__).resolve().parents[1]/'.venv-engineai-official/lib/python3.10/site-packages')
PACKAGES={'torch','torchvision','tensordict','numpy','gymnasium','scipy','onnx','onnxscript','onnx_ir','triton','sympy'}

class IsolatedPackageFinder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.partition('.')[0] not in PACKAGES:return None
        return importlib.machinery.PathFinder.find_spec(fullname,[ISOLATED] if path is None else path)

def activate():
    # Reinsert after Kit startup, which prepends its fast namespace importer.
    sys.meta_path[:]=[v for v in sys.meta_path if not isinstance(v,IsolatedPackageFinder)]
    sys.meta_path.insert(0,IsolatedPackageFinder())
