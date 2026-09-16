"""Validate custom confirmation provenance without launching or changing experiments."""
import argparse
import json
from pathlib import Path
from combat.benchmark_readiness import check_manifest, digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/benchmark_protocol_v1.json')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = root / a.config
    cfg = json.loads(config_path.read_text(encoding='utf8'))
    result = check_manifest(root, root / cfg['manifest_path'], root / cfg['annotation_path'])
    result['protocol_sha256'] = digest(config_path)
    result['tracks'] = cfg['tracks']
    target = Path(a.output)
    if target.exists():
        raise FileExistsError('Do not overwrite a readiness receipt: ' + str(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(0 if result['confirmation_manifest_ready'] else 2)


if __name__ == '__main__':
    main()
