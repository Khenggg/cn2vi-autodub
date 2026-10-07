import os
import subprocess
import sys
from pathlib import Path

from autodub.model_assets import fetch_asset, load_manifest
from autodub.model_runner import ModelRunner


def pinned_source(tmp_path):
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    files = {
        ".gitignore": "__pycache__/\n",
        "fireredasr2s/__init__.py": "",
        "fireredasr2s/fireredpunc/__init__.py": "",
        "fireredasr2s/fireredpunc/punc.py":
            "class FireRedPunc:\n    def process(self):\n        return None\n",
    }
    for name, content in files.items():
        path = upstream / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    subprocess.run(["git", "-C", str(upstream), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(upstream), "add", "."], check=True)
    subprocess.run(["git", "-C", str(upstream), "-c", "user.name=Fixture",
                    "-c", "user.email=fixture@example.invalid", "commit", "-qm", "fixture"], check=True)
    revision = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    models = tmp_path / "models"
    fetch_asset({"id": "firered-code", "kind": "git", "repo": str(upstream),
                 "revision": revision}, models)
    return models


def test_cloud_import_check_preserves_pinned_source(tmp_path):
    models = pinned_source(tmp_path)
    repo = Path(__file__).resolve().parents[1]
    code = (
        "import importlib.util,sys; from pathlib import Path; "
        "spec=importlib.util.spec_from_file_location('check',sys.argv[1]); "
        "module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); "
        "module.configure_offline=lambda config:None; "
        "module.check('punctuation',Path(sys.argv[2]))"
    )
    subprocess.run([sys.executable, "-c", code, str(repo / "scripts/v1_model_check.py"), str(models)],
                   env={**os.environ, "PYTHONPATH": str(repo / "src"),
                        "PYTHONDONTWRITEBYTECODE": "0"}, check=True)
    assert load_manifest(models / "firered-code")["id"] == "firered-code"
    assert not list((models / "firered-code/source").rglob("__pycache__"))


def test_real_worker_import_preserves_pinned_source(tmp_path, monkeypatch):
    models = pinned_source(tmp_path)
    package_root = tmp_path / "worker"
    package = package_root / "autodub"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "model_worker.py").write_text('''
import argparse,json,sys
from pathlib import Path
parser=argparse.ArgumentParser()
parser.add_argument('--request'); parser.add_argument('--result')
args=parser.parse_args()
request=json.loads(Path(args.request).read_text())
sys.path.insert(0,str(Path(request['config']['models_root'])/'firered-code'/'source'))
from fireredasr2s.fireredpunc.punc import FireRedPunc
artifact=Path(request['config']['output_dir'])/'import-result.json'
artifact.write_text(json.dumps({'imported':callable(FireRedPunc.process)}))
Path(args.result).write_text(json.dumps({'schema_version':1,'stage':request['stage'],
    'result':{'artifacts':[str(artifact)]}}))
''')
    source = tmp_path / "source.bin"
    source.write_bytes(b"fixture")
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "0")
    runner = ModelRunner({"package_root": str(package_root), "models_root": str(models),
                          "interpreters": {"punctuation": sys.executable}})
    result = runner.run("V1_PUNCTUATION", source, {"output_dir": str(tmp_path / "output")})
    assert Path(result["artifacts"][0]).is_file()
    assert load_manifest(models / "firered-code")["id"] == "firered-code"
    assert not list((models / "firered-code/source").rglob("__pycache__"))
