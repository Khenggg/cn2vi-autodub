# Tạo suite benchmark từ corpus

`autodub.bench_plan` kiểm tra manifest corpus bằng FFprobe rồi sinh `suite.json` và cấu hình riêng cho từng case/stage. Chương trình kiểm tra coverage representative trước khi ghi file, sau đó gọi suite dry-run để kiểm tra schema, đường dẫn và Python interpreter. Bước này không nạp adapter, GPU, weights hay tải model.

```sh
python -m autodub.bench_plan \
  --corpus /data/golden/corpus.json \
  --output-dir /data/benchmarks/run-001 \
  --models-root /data/models \
  --cache-root /data/cache \
  --venv-root /opt/autodub/venvs \
  --hourly-rate-vnd 50000

python -m autodub.bench_suite --plan /data/benchmarks/run-001/suite.json --dry-run
python -m autodub.bench_suite --plan /data/benchmarks/run-001/suite.json
```

`--venv-root` chứa bốn environment đã cài sẵn: `asr`, `tts`, `bandit` và `vision`. ASR/alignment dùng `asr`; OCR/inpainting dùng `vision`. Bộ tạo plan ghi đường dẫn interpreter theo hệ điều hành. Nếu environment chưa tồn tại, các file plan vẫn được giữ lại nhưng kết quả tạo plan là `PREFLIGHT_BLOCKED` để có thể hoàn tất cài đặt rồi chạy lại dry-run.

Có thể truyền `--ffmpeg-bin` và `--ffprobe-bin`; mặc định lần lượt lấy từ `FFMPEG_BIN`/`FFPROBE_BIN` hoặc tên executable trong `PATH`. Mỗi config nhận `models_root`, `cache_root`, `output_dir`, `ffmpeg_bin` và `ffprobe_bin`. TTS dùng thêm `text`, `target_ms`, preset `Mai Anh` và `cuda:0`. Khi case có text nhưng thiếu `target_ms`, generator dùng độ dài interval của case làm target.

Output directory chứa `suite.json`, `generation-manifest.json` và `configs/<job>.json`. Mỗi job ghi report `<job>.json` cùng thư mục artifact kề bên `<job>.artifacts`. ASR tạo đường dẫn transcript mà job alignment cùng case sử dụng; OCR tạo manifest cho job inpainting. Source trong plan là đường dẫn tuyệt đối đã được kiểm tra hash.

Mặc định generator tạo ASR, alignment và Bandit cho mỗi case; TTS chỉ chạy khi case có `tts_text`, còn OCR và inpainting chỉ chạy khi có ROI. Job bị bỏ qua được ghi trong `generation-manifest.json` với lý do như `missing_tts_text` hoặc `missing_roi`. Reference transcript và word timing thiếu được ghi vào `quality_evidence_missing`; generator không điền dữ liệu giả. Cấu hình TTS lấy `target_ms` từ case hoặc interval như trên.

`generation-manifest.json` là snapshot xác định của corpus đã kiểm tra, jobs, config hashes và các stage bị bỏ qua; các khóa `env`, `env_vars`, `environment` và `environment_variables` bị loại bỏ. `suite.json` chứa SHA-256 của snapshot để kiểm tra lại nội dung.

Manifest representative phải hợp lệ, đủ bảy category và ít nhất mười phút case coverage. Synthetic smoke chỉ được dùng khi truyền `--allow-synthetic`; các job vẫn được tạo để kiểm tra wiring nhưng manifest, suite và quality evidence luôn đánh dấu `representative_quality_pass: false`. Kết quả `MEASURED` luôn cần quality review và không phải quality pass.
