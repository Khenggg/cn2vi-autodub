#!/usr/bin/env python3
"""Run Phase 0 GPU Benchmark on RTX 5060 Ti for CN2VI AutoDub.

Measures Peak VRAM, RTF (Real-Time Factor), throughput, and output quality across:
- Qwen3-ASR (Chinese Speech Recognition)
- BandIt ERB48 (Speech / Music / SFX separation)
- VieNeu v3 Turbo (Vietnamese Neural TTS)
- RapidOCR PP-OCRv6 (Hardcoded subtitle detection)
- LaMa ONNX (Subtitle inpainting)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

VENVS = {
    "core": "/opt/autodub/venvs/core/bin/python",
    "asr": "/opt/autodub/venvs/asr/bin/python",
    "tts": "/opt/autodub/venvs/tts/bin/python",
    "bandit": "/opt/autodub/venvs/bandit/bin/python",
    "vision": "/opt/autodub/venvs/vision/bin/python",
}

DEFAULT_VIDEO = "/data/uploads/3b998daa3f9c400db25035bfb1c6707b/66a0a891f9884195a5782eb39ed4114f/source.mp4"
BENCH_DIR = Path("/data/benchmarks")
RESULTS_DIR = Path("/data/results")


def probe_video(video_path: Path) -> dict:
    cmd = [
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration,size,bit_rate:stream=width,height,r_frame_rate,codec_name",
        "-of", "json", str(video_path)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(res.stdout)
    fmt = data.get("format", {})
    video_stream = next((s for s in data.get("streams", []) if s.get("width")), {})
    return {
        "duration_s": float(fmt.get("duration", 0)),
        "size_mb": round(int(fmt.get("size", 0)) / (1024 * 1024), 2),
        "width": video_stream.get("width"),
        "height": video_stream.get("height"),
        "codec": video_stream.get("codec_name"),
    }


def prepare_configs(video_path: Path, meta: dict) -> Path:
    cfg_dir = BENCH_DIR / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. ASR config: test first 60 seconds (or full) of audio
    asr_cfg = {
        "models_root": "/data/models",
        "cache_root": "/data/cache",
        "output_dir": str(RESULTS_DIR / "asr_artifacts"),
        "start_ms": 0,
        "end_ms": min(int(meta["duration_s"] * 1000), 60000),
        "chunk_ms": 60000,
        "overlap_ms": 1000,
        "device": "cuda:0",
        "dtype": "bfloat16"
    }
    (cfg_dir / "asr.json").write_text(json.dumps(asr_cfg, indent=2), encoding="utf-8")

    # 2. Bandit config: 1 window of 10s
    bandit_cfg = {
        "models_root": "/data/models",
        "cache_root": "/data/cache",
        "output_dir": str(RESULTS_DIR / "bandit_artifacts"),
        "windows": [{"start_ms": 0, "end_ms": min(int(meta["duration_s"] * 1000), 10000)}],
        "device": "cuda:0",
        "batch_size": 1
    }
    (cfg_dir / "bandit.json").write_text(json.dumps(bandit_cfg, indent=2), encoding="utf-8")

    # 3. TTS config: sample Vietnamese sentence
    tts_cfg = {
        "models_root": "/data/models",
        "cache_root": "/data/cache",
        "output_dir": str(RESULTS_DIR / "tts_artifacts"),
        "text": "Chào bạn, đây là thử nghiệm giọng đọc tự động tiếng Việt bằng mô hình trí tuệ nhân tạo trên máy chủ GPU.",
        "voice_id": "Mai Anh",
        "device": "cuda:0",
        "target_ms": 6000,
        "dtype": "bfloat16"
    }
    (cfg_dir / "tts.json").write_text(json.dumps(tts_cfg, indent=2), encoding="utf-8")

    # 4. OCR config
    width = meta.get("width") or 1280
    height = meta.get("height") or 720
    ocr_cfg = {
        "models_root": "/data/models",
        "cache_root": "/data/cache",
        "output_dir": str(RESULTS_DIR / "ocr_artifacts"),
        "roi": {"x": 0, "y": int(height * 0.75), "width": width, "height": int(height * 0.25)},
        "start_ms": 0,
        "end_ms": min(int(meta["duration_s"] * 1000), 5000),
        "sample_fps": 2,
        "text_score": 0.5
    }
    (cfg_dir / "ocr.json").write_text(json.dumps(ocr_cfg, indent=2), encoding="utf-8")

    # Suite Plan
    plan = {
        "schema_version": 1,
        "hourly_rate_vnd": 3000,
        "jobs": [
            {
                "id": "probe-source",
                "stage": "probe",
                "input": str(video_path),
                "python": VENVS["core"],
                "timeout_seconds": 60
            },
            {
                "id": "asr-qwen",
                "stage": "asr",
                "input": str(video_path),
                "config": str(cfg_dir / "asr.json"),
                "output": str(RESULTS_DIR / "asr_report.json"),
                "python": VENVS["asr"],
                "timeout_seconds": 300
            },
            {
                "id": "bandit-erb48",
                "stage": "bandit",
                "input": str(video_path),
                "config": str(cfg_dir / "bandit.json"),
                "output": str(RESULTS_DIR / "bandit_report.json"),
                "python": VENVS["bandit"],
                "timeout_seconds": 300
            },
            {
                "id": "tts-vieneu",
                "stage": "tts",
                "input": str(video_path),
                "config": str(cfg_dir / "tts.json"),
                "output": str(RESULTS_DIR / "tts_report.json"),
                "python": VENVS["tts"],
                "timeout_seconds": 300
            },
            {
                "id": "ocr-rapidocr",
                "stage": "ocr",
                "input": str(video_path),
                "config": str(cfg_dir / "ocr.json"),
                "output": str(RESULTS_DIR / "ocr_report.json"),
                "python": VENVS["vision"],
                "timeout_seconds": 300
            }
        ]
    }
    plan_path = BENCH_DIR / "suite.json"
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return plan_path


def main():
    parser = argparse.ArgumentParser(description="Run Phase 0 GPU Benchmark Suite")
    parser.add_argument("--video", type=Path, default=Path(DEFAULT_VIDEO), help="Target video file")
    parser.add_argument("--dry-run", action="store_true", help="Validate configuration only")
    args = parser.parse_args()

    video_path = args.video.resolve()
    if not video_path.is_file():
        print(f"[ERROR] Video file not found: {video_path}")
        sys.exit(1)

    print(f"=== [Phase 0] GPU Benchmark on NVIDIA RTX 5060 Ti ===")
    print(f"Target Video: {video_path}")
    meta = probe_video(video_path)
    print(f"Video Info: {meta['duration_s']:.1f}s ({meta['duration_s']/60:.1f}m), {meta['width']}x{meta['height']}, {meta['size_mb']} MB")

    plan_path = prepare_configs(video_path, meta)
    print(f"Suite Plan generated at: {plan_path}")

    # Set model cache envs
    env = os.environ.copy()
    env["HF_HOME"] = "/data/cache/huggingface"
    env["TORCH_HOME"] = "/data/cache/torch"
    env["XDG_CACHE_HOME"] = "/data/cache"

    summary_path = RESULTS_DIR / "suite_summary.json"
    if summary_path.is_file():
        summary_path.unlink()

    cmd = [
        VENVS["core"], "-m", "autodub.bench_suite",
        "--plan", str(plan_path),
        "--summary", str(summary_path)
    ]
    if args.dry_run:
        cmd.append("--dry-run")

    import threading
    stop_monitor = threading.Event()
    def monitor_progress():
        last_states = {}
        while not stop_monitor.is_set():
            if summary_path.is_file():
                try:
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    for job in summary.get("jobs", []):
                        jid = job.get("id")
                        st = job.get("status")
                        if jid and st != last_states.get(jid):
                            last_states[jid] = st
                            if st == "RUNNING":
                                print(f"  [RUNNING] {jid} ({job.get('stage')})...", flush=True)
                            elif st in {"MEASURED", "READY"}:
                                ms = job.get("elapsed_ms", 0)
                                print(f"  [OK] {jid} ({job.get('stage')}) finished in {ms/1000:.2f}s", flush=True)
                            elif st in {"FAILED", "TIMED_OUT"}:
                                print(f"  [FAIL] {jid} ({job.get('stage')}): {job.get('error_type')}", flush=True)
                except Exception:
                    pass
            stop_monitor.wait(1.5)

    monitor_thread = threading.Thread(target=monitor_progress, daemon=True)
    monitor_thread.start()

    print("\n--- Starting Benchmark Execution ---")
    start_time = time.time()
    try:
        res = subprocess.run(cmd, env=env)
    finally:
        stop_monitor.set()
        monitor_thread.join(timeout=2)

    elapsed = time.time() - start_time
    print(f"\n--- Benchmark Finished in {elapsed:.2f}s (Exit code: {res.returncode}) ---")

    # Read and print summary
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        print(f"\nOverall Status: {summary.get('status')}")
        print("\nStage Results Table:")
        print(f"  {'Status':<10} {'Job ID':<18} {'Stage':<10} {'Time (s)':<10} {'Peak VRAM':<15} {'Peak RAM':<12}")
        print("  " + "-" * 75)
        for job in summary.get("jobs", []):
            job_id = job.get("id")
            stage = job.get("stage")
            status = job.get("status")
            elapsed_ms = job.get("elapsed_ms", 0)
            peak_gpu_mb = job.get("peak_gpu_mb")
            peak_ram_mb = round(job.get("peak_ram_bytes", 0) / (1024 * 1024), 1)
            vram_str = f"{peak_gpu_mb} MB" if peak_gpu_mb else "N/A"
            ram_str = f"{peak_ram_mb} MB"
            print(f"  {status:<10} {job_id:<18} {stage:<10} {elapsed_ms/1000:>8.2f}s  {vram_str:>12}  {ram_str:>10}")

            if status == "FAILED":
                rep_path = Path(job.get("report", ""))
                if rep_path.is_file():
                    try:
                        rep = json.loads(rep_path.read_text(encoding="utf-8"))
                        err_msg = rep.get("error_message") or rep.get("error_type")
                        if err_msg:
                            print(f"    --> Error Details: {err_msg}")
                    except Exception:
                        pass
        print("  " + "-" * 75)

    sys.exit(res.returncode)


if __name__ == "__main__":
    main()
