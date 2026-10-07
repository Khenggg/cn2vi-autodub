"""Print the checked-in cloud install plan; never download or import model libraries."""
import argparse
import json
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--field', choices=['assets', 'profiles'])
    args = parser.parse_args()
    plan = json.loads((args.project / 'config/cloud-runtime.json').read_text(encoding='utf-8'))
    lock = json.loads((args.project / 'benchmarks/models.lock.json').read_text(encoding='utf-8'))
    if plan.get('schema_version') != 1 or lock.get('schema_version') != 1:
        raise ValueError('Unsupported install plan or model lock')
    known = {m['id']: m for m in lock['models']}
    for field in ('assets', 'profiles'):
        values = plan[field]
        if not values or len(set(values)) != len(values) or any(not re.fullmatch(r'[a-z0-9][a-z0-9_-]*', v) for v in values):
            raise ValueError('Invalid install selection')
    if not set(plan['assets']) <= known.keys():
        raise ValueError('Install plan references unknown model assets')
    if not set(plan['profiles']) <= {'asr', 'tts', 'vision', 'separation', 'bandit'}:
        raise ValueError('Install plan references unknown environments')
    if args.field:
        print('\n'.join(plan[args.field]))
    else:
        size = sum(f.get('bytes') or 0 for name in plan['assets'] for f in known[name].get('files', []))
        print(json.dumps({**plan, 'known_model_download_bytes': size, 'unknown_size_files': sum(f.get('bytes') is None for name in plan['assets'] for f in known[name].get('files', [])), 'known_model_download_gib': round(size / 1024**3, 2)}, indent=2))

if __name__ == '__main__':
    main()
