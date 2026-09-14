"""Build an isolated small overlay, keeping the borrowed PyTorch environment read-only."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
spec = json.loads((root / "env-spec.json").read_text())
spec_hash = hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:8]
for name in ["cache/tmp", "audit", ".aris/compute"]:
    (root / name).mkdir(parents=True, exist_ok=True)
assert sys.version_info[:2] == (3, 10)
assert (root / "audit/disk_cleanup.json").exists(), "Clean and audit disks before setup."
assert (root / "audit/system_cleanup.json").exists()
env = dict(os.environ, TMPDIR=str(root / "cache/tmp"), PIP_NO_CACHE_DIR="1")
if not (root / ".venv/bin/python").exists():
    subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", str(root / ".venv")], check=True, env=env)
for phase in spec["pip_phases"]:
    subprocess.run([str(root / ".venv/bin/python"), "-m", "pip", "install", "--no-cache-dir", *phase], check=True, env=env)
(root / ".aris/compute/nlp03.md").write_text(f"# Environment ledger\n\nSpec hash: {spec_hash}\n\nBorrowed interpreter: {spec['borrowed_interpreter']} (read-only).\nOverlay: {root}/.venv\n\nInvocation (working directory {root}):\n```sh\nbash scripts/run.sh scripts/smoke.py\n```\n\nStatus: installed; kernel witness and fresh-agent document validation pending.\n")
print("ENV_SPEC_HASH", spec_hash)
