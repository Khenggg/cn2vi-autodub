"""Explicit provisioning on an authorized execution machine, never on UI startup."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

from .media import file_sha256
from .pipeline import write_json

ARTIFACTS = {
    "detector": {
        "filename": "PP-OCRv6_det_small.onnx",
        "source": "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx/PP-OCRv6/det/PP-OCRv6_det_small.onnx",
        "sha256": "090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f",
    },
    "recognizer": {
        "filename": "PP-OCRv6_rec_small.onnx",
        "source": "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx/PP-OCRv6/rec/PP-OCRv6_rec_small.onnx",
        "sha256": "6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884",
    },
}
SOURCE_REVISION = "0700743fc8bfb3943cd8e40e84547f14c7d15dfc"


def prepare(destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": 1, "family": "PP-OCRv6-small"}
    for name, row in ARTIFACTS.items():
        final = destination / row["filename"]
        if not final.exists() or file_sha256(final) != row["sha256"]:
            temporary = final.with_suffix(".part")
            try:
                with (
                    urllib.request.urlopen(row["source"], timeout=90) as source,
                    temporary.open("wb") as target,
                ):
                    while block := source.read(1024 * 1024):
                        target.write(block)
                if file_sha256(temporary) != row["sha256"]:
                    raise ValueError(
                        f"Download hash mismatch: {name}; no alternative model will be used"
                    )
                temporary.replace(final)
            finally:
                temporary.unlink(missing_ok=True)
        manifest[name] = {
            "path": final.name,
            "sha256": row["sha256"],
            "source": row["source"],
            "license": "Apache-2.0",
            "registry_revision": SOURCE_REVISION,
        }
    for filename in ("LICENSE", "MODEL_LICENSES.md"):
        url = f"https://raw.githubusercontent.com/RapidAI/RapidOCR/{SOURCE_REVISION}/python/{filename}"
        data = urllib.request.urlopen(url, timeout=30).read()
        (destination / filename).write_bytes(data)
    write_json(destination / "manifest.json", manifest)
    print(json.dumps({"manifest": str(destination / "manifest.json"), "hashes_verified": True}))


def main():
    parser = argparse.ArgumentParser(
        description="Provision the frozen OCR pair on the execution machine"
    )
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--confirm-execution-machine", action="store_true")
    args = parser.parse_args()
    if not args.confirm_execution_machine:
        parser.error(
            "Do not download weights on the local development machine; use --confirm-execution-machine only on the authorized execution host"
        )
    prepare(args.destination)


if __name__ == "__main__":
    main()
