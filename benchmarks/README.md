# Phase 0 benchmark

Adapters thật đã có cho `asr`, `align`, `bandit`, `tts`, `ocr`, `inpaint` (LaMa) và `propainter`. Chạy mỗi stage trong profile environment riêng, process mới để giải phóng model sau khi đo. Inference dùng assets local đã xác minh revision/hash.

```sh
python -m autodub.model_assets list --lock benchmarks/models.lock.json
python -m autodub.model_assets fetch --lock benchmarks/models.lock.json --root /data/models
python -m autodub.benchmark --stage asr --input /data/inputs/sample.mp4 \
  --config /data/configs/asr.json --output /data/results/asr.json
```

Lệnh inference phải dùng Python của profile tương ứng trong [cloud runbook](../docs/CLOUD_RUNBOOK.md). [Corpus validator](../docs/BENCHMARK_CORPUS.md) và [plan generator](../docs/BENCHMARK_PLAN.md) sinh config/suite từ source đã kiểm tra. [Suite runner](../docs/BENCHMARK_RUNNER.md) chạy tuần tự và dry-run không nạp model. [CPU smoke](../docs/CPU_SMOKE.md) dùng clip tổng hợp có nhãn, không cần thuê GPU.

Adapter có contract `run(source: pathlib.Path, config: dict) -> dict`, chạy inference thật và trả:

```json
{
  "model_revision": "exact upstream commit or model revision",
  "weights_sha256": "actual checksum of model weights manifest",
  "quality_metrics": {"cer": 0.0, "alignment_p95_error_ms": 0}
}
```

Schema trên chỉ minh họa, không phải kết quả đo. Chỉ số phải phù hợp với stage và ground truth; chưa có ground truth thì ghi `null`, không tự điền số tốt. Kết quả `MEASURED` không đồng nghĩa quality pass. RTF dùng thời gian inference trên số ms thực xử lý; stage wall time gồm cold load/import/probe. RAM lấy mẫu 500 ms có thể bỏ lỡ peak rất ngắn; VRAM device bao gồm process khác, allocator metrics được ghi riêng nếu có. Nhãn `synthetic_smoke` và `representative_quality_pass: false` đi xuyên config, report và summary.

Chuẩn bị bộ video 10 phút có thoại rõ, nhạc, scream/cry/laugh/breath, overlap, subtitle nền tĩnh/chuyển động và cảnh tối. Giữ source SHA-256, transcript/timing chuẩn, thuật ngữ, vùng chữ và đánh giá SFX. ProPainter có [gate crop/mask/dense frames riêng](../docs/PROPAINTER_BENCHMARK.md). Chưa gọi DashScope hoặc chạy CUDA inference; đã tải và chạy một số weights CPU, ghi trong VALIDATION.

Gates cần đo: ASR CER/RTF; aligner median/p95 timing error; Bandit leakage và SFX damage; TTS pronunciation/duration retry; OCR recall/false positive trong ROI; inpainting residual/flicker/VRAM; E2E wall time và chi phí. Các mục tiêu 35 phút/120 phút và 5.000 VND là mục tiêu thiết kế, chưa được xác nhận.
