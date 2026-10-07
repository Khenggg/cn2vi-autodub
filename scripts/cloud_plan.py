"""Print the checked-in cloud install plan; never download or import model libraries."""
import argparse
import json
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--field', choices=['assets', 'profiles', 'pipeline_version'])
    parser.add_argument('--profiles', help='Comma-separated explicit profile choices to filter or set')
    parser.add_argument('--profile', help='Explicit single profile choice')
    args = parser.parse_args()
    plan = json.loads((args.project / 'config/cloud-runtime.json').read_text(encoding='utf-8'))
    lock = json.loads((args.project / 'benchmarks/models.lock.json').read_text(encoding='utf-8'))
    if plan.get('schema_version') != 1 or lock.get('schema_version') != 1:
        raise ValueError('Unsupported install plan or model lock')
    known = {m['id']: m for m in lock['models']}

    # Allow known profiles including upcoming profiles diarization, punctuation, indextts
    known_profiles = {
        'asr', 'tts', 'vision', 'separation', 'bandit',
        'diarization', 'punctuation', 'indextts',
    }
    # Dynamically discover any profile requirement files present in requirements/
    req_dir = args.project / 'requirements'
    if req_dir.is_dir():
        for req_file in req_dir.glob('*.txt'):
            stem = req_file.stem
            for prefix in ('runtime-', 'bench-'):
                if stem.startswith(prefix):
                    stem = stem[len(prefix):]
            if stem != 'worker-common':
                known_profiles.add(stem)

    if args.profiles:
        explicit = [p.strip() for p in args.profiles.split(',') if p.strip()]
        if not explicit or len(set(explicit)) != len(explicit):
            raise ValueError('Invalid explicit profiles selection')
        plan['profiles'] = explicit
    elif args.profile:
        explicit = [args.profile.strip()]
        if not explicit[0]:
            raise ValueError('Invalid explicit profile selection')
        plan['profiles'] = explicit

    if plan.get('pipeline_version') == 'CN2VI-V2':
        if any(p not in plan['profile_assets'] for p in plan['profiles']):
            raise ValueError('V2 production installs only ASR, TTS and CUDA vision profiles')
        plan['assets'] = list(dict.fromkeys(asset for profile in plan['profiles']
                                           for asset in plan['profile_assets'][profile]))
    for field in ('assets', 'profiles'):
        values = plan[field]
        if not values or len(set(values)) != len(values) or any(not re.fullmatch(r'[a-z0-9][a-z0-9_-]*', v) for v in values):
            raise ValueError('Invalid install selection')
    missing_assets = set(plan['assets']) - known.keys()
    if missing_assets:
        raise ValueError(f"Install plan references unknown model assets: {sorted(missing_assets)}")
    unknown_envs = set(plan['profiles']) - known_profiles
    if unknown_envs:
        raise ValueError(f"Install plan references unknown environments: {sorted(unknown_envs)}")

    if args.field:
        print(plan.get('pipeline_version', 'legacy') if args.field == 'pipeline_version' else '\n'.join(plan[args.field]))
    else:
        size = sum(f.get('bytes') or 0 for name in plan['assets'] for f in known[name].get('files', []))
        print(json.dumps({**plan, 'known_model_download_bytes': size, 'unknown_size_files': sum(f.get('bytes') is None for name in plan['assets'] for f in known[name].get('files', [])), 'known_model_download_gib': round(size / 1024**3, 2)}, indent=2))

if __name__ == '__main__':
    main()
