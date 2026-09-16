"""Acquire the official archive for local non-commercial research; no redistribution."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request
import zipfile

URL = 'https://vcai.mpi-inf.mpg.de/projects/remos/ReMocap.zip'
LICENSE = 'https://raw.githubusercontent.com/anindita127/ReMoS/main/README.md'


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    a = p.parse_args()
    a.root.mkdir(parents=True, exist_ok=True)
    a.receipt.parent.mkdir(parents=True, exist_ok=True)
    license_text = urllib.request.urlopen(LICENSE, timeout=60).read()
    (a.root / 'upstream_README_license.md').write_bytes(license_text)
    archive = a.root / 'ReMocap.zip'
    if not archive.exists():
        partial = archive.with_suffix('.zip.partial')
        # Restart rather than trust servers that ignore Range requests.
        with urllib.request.urlopen(URL, timeout=120) as r, partial.open('wb') as f:
            expected = int(r.headers.get('Content-Length', 0))
            total = 0
            last = time.monotonic()
            while True:
                block = r.read(4 * 1024 * 1024)
                if not block:
                    break
                f.write(block)
                total += len(block)
                if time.monotonic() - last > 15:
                    print(json.dumps({'downloaded_bytes': total, 'expected_bytes': expected}), flush=True)
                    last = time.monotonic()
            if expected and total != expected:
                raise ValueError('Truncated archive')
        partial.replace(archive)
    dest = (a.root / 'extracted').resolve()
    with zipfile.ZipFile(archive) as z:
        bad = z.testzip()
        if bad:
            raise ValueError('CRC mismatch: ' + bad)
        for item in z.infolist():
            target = (dest / item.filename).resolve()
            if dest not in target.parents and target != dest:
                raise ValueError('Unsafe archive path')
            if (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('Archive symlink rejected')
        z.extractall(dest)
        entries = len(z.infolist())
    receipt = dict(source_url=URL, license_url=LICENSE,
                   license_sha256=hashlib.sha256(license_text).hexdigest(),
                   terms='noncommercial research only; no redistribution including derivatives',
                   archive_sha256=sha256(archive), archive_bytes=archive.stat().st_size,
                   archive_entries=entries, crc_verified=True, extracted_path=str(dest),
                   data_uploaded_to_github=False, acquisition_complete=True)
    a.receipt.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print('REMOCAP_ACQUISITION_COMPLETE ' + json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
