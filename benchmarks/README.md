# Phase 0 benchmark

Runner đã có; adapter cho ASR/Bandit/TTS/inpaint và phép đo chất lượng cần được triển khai, chạy trên cloud trước khi bật Phase A. Không có adapter giả trả kết quả thành công. Chạy mỗi stage trong process mới để giải phóng model sau khi đo.

```sh
python -m autodub.benchmark --stage probe --input golden/representative.mp4 --output benchmark-results/probe.json
python benchmarks/asr_bench.py --input golden/representative.mp4 --adapter local_adapters.qwen:run --config local-config.json --output benchmark-results/asr.json
```

`local_adapters.qwen:run` trong ví dụ chưa tồn tại. Adapter phải là hàm `run(source: pathlib.Path, config: dict) -> dict`, chạy inference thật và trả:

```json
{
  "model_revision": "exact upstream commit or model revision",
  "weights_sha256": "actual checksum of model weights manifest",
  "quality_metrics": {"cer": 0.0, "alignment_p95_error_ms": 0}
}
```

Schema trên chỉ minh họa, không phải kết quả đo. Chỉ số phải phù hợp với stage và ground truth; chưa có ground truth thì ghi `null`, không tự điền số tốt. Kết quả `MEASURED` không đồng nghĩa quality pass. Runner ghi wall time, RTF, RAM RSS lấy mẫu của process và children, VRAM dùng của GPU đầu tiên (bao gồm process khác). Lấy mẫu 500 ms có thể bỏ lỡ peak rất ngắn; GPU adapter cần bổ sung peak allocator đo được vào quality/measurement manifest riêng.

Chuẩn bị bộ video 10 phút có thoại rõ, nhạc, scream/cry/laugh/breath, overlap, subtitle nền tĩnh/chuyển động và cảnh tối. Giữ source SHA-256, transcript/timing chuẩn, thuật ngữ, vùng chữ và đánh giá SFX. Kiểm chứng tên model/phiên bản/giá trong tài liệu trước khi sử dụng. Không tải weights hoặc gọi API ở bước tạo skeleton.

Gates cần đo: ASR CER/RTF; aligner median/p95 timing error; Bandit leakage và SFX damage; TTS pronunciation/duration retry; OCR recall/false positive trong ROI; inpainting residual/flicker/VRAM; E2E wall time và chi phí. Các mục tiêu 35 phút/120 phút và 5.000 VND là mục tiêu thiết kế, chưa được xác nhận.
