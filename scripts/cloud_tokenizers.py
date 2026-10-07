"""Extract only verified Punkt tokenizer resources during explicit cloud setup."""
import argparse
import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath

from autodub.adapters.common import asset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models-root', type=Path, required=True)
    parser.add_argument('--cache-root', type=Path, required=True)
    args = parser.parse_args()
    folder, _manifest = asset({'models_root': str(args.models_root)}, 'nltk-tokenizers')
    target = (args.cache_root / 'nltk_data' / 'tokenizers').resolve()
    if not target.is_relative_to(args.cache_root.resolve()):
        raise ValueError('Tokenizer cache root escapes through a symlink')
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(folder / 'punkt_tab.zip') as archive:
        for item in archive.infolist():
            name = PurePosixPath(item.filename)
            if (name.is_absolute() or '..' in name.parts or '\\' in item.filename
                    or not name.parts or name.parts[0] != 'punkt_tab'
                    or stat.S_ISLNK(item.external_attr >> 16)):
                raise ValueError('Unexpected tokenizer archive member')
            destination = target.joinpath(*name.parts)
            if not destination.resolve().is_relative_to(target):
                raise ValueError('Tokenizer path escapes its cache')
            if item.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(item) as source, destination.open('wb') as output:
                    shutil.copyfileobj(source, output)
    print('Verified Punkt tokenizer cache prepared; no inference was run.')

if __name__ == '__main__':
    main()
