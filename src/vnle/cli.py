"""UI-only startup requires only Python; analysis dependencies are opt-in."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .domain import Request
from .media import probe
from .pipeline import DEFAULT_CONFIG, analyze, validate_config
from .server import serve


def main():
    parser = argparse.ArgumentParser(description="VNLE visual text discovery prototype")
    sub = parser.add_subparsers(dest="command", required=True)
    ui = sub.add_parser("serve", help="Preview, manual subtitle exclusion and event review")
    ui.add_argument("--data-root", type=Path, default=Path("data/vnle"))
    ui.add_argument("--port", type=int, default=18083)
    ui.add_argument("--enable-analysis", action="store_true")
    ui.add_argument("--model-manifest", type=Path)
    run = sub.add_parser("analyze", help="Run only on the authorized execution machine")
    run.add_argument("video", type=Path)
    run.add_argument("--request", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--model-manifest", type=Path, required=True)
    for command in (ui, run):
        command.add_argument("--ffprobe", default="ffprobe")
        command.add_argument("--config", type=Path)
    args = parser.parse_args()
    config = (
        validate_config(json.loads(args.config.read_text(encoding="utf-8")))
        if args.config
        else DEFAULT_CONFIG
    )
    if args.command == "serve":
        if args.enable_analysis and not args.model_manifest:
            parser.error("--enable-analysis requires --model-manifest")
        serve(
            args.data_root,
            args.port,
            args.ffprobe,
            args.model_manifest,
            config,
            args.enable_analysis,
        )
    else:
        media = probe(args.video, args.ffprobe)
        request = Request.parse(
            json.loads(args.request.read_text(encoding="utf-8")), media["duration_s"]
        )
        result = analyze(
            args.video,
            media,
            request,
            args.output,
            args.model_manifest,
            config,
            lambda update: print(json.dumps(update), flush=True),
        )
        if result["status"] != "COMPLETED_UNVERIFIED":
            raise SystemExit(result["error"] or result["status"])


if __name__ == "__main__":
    main()
