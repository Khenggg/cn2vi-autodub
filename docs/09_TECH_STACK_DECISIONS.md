# 09 — Stack và ADR

Snapshot nguồn là revision nghiên cứu, **không là dependency lock đã thử**. Không cài lại CN2VI hoặc copy môi trường SubAI. Version packaging cuối chỉ khóa sau prototype+Windows clean-env test.

| Thành phần | Candidate/version policy | License và giới hạn |
|---|---|---|
| Python | 3.12x64 baseline; kiểm wheel matrix thật | PSF; orchestration/schema/SQLite |
| FFmpeg/ffprobe | Stable build có NVENC/libass; hash+configure khóa, không tự dùngHEAD | LGPL/GPL theo build, no nonfree redistribution default |
| PyAV | Stable wheel tương thích Python/FFmpeg; source snapshot hiện có | BSD-3-Clause own wrapper; linked FFmpeg terms riêng; PTS/native decode |
| OpenCV | 4.x (docs4.13.0 khi truy cập); wheels thường không CUDA | Apache-2.0 current; LK/geometry/resize/native, custom CUDA build không default |
| RapidOCR | Snapshot0700743…, chọn stable release có model registry tương ứng | Apache-2.0; code/model attribution riêng |
| PP-OCR | v6small ứng viên, v6tiny/v5mobile đối chứng | Official smallONNXApache-2.0; artifact SHA/config/dictionary phải khóa |
| ORT | Stable CUDA wheel khớp CUDAmajor/cuDNNmajor theo matrix | MIT; CPU partition/profiling/IOBinding kiểm thật |
| libass | Stable compatible FFmpeg; font hashes | ISC; renderer CPU, FreeType/HarfBuzz/font dependencies audit |
| PySceneDetect | Docs0.7.1/source81c414c…; versionAPI khóa sau thử | BSD-3-Clause; optional scene cues, không secondfull pass mặc định |
| pybind11 | Chỉ đưa vào khi cần customC++ | BSD-3-Clause; GIL release phải explicit |
| Cython | Candidate cho hotspot numeric loops đã đo | Apache-2.0; typed code/nogil không tự parallel |
| Nuitka | Standalone packaging sau quality gates | Apache-2.0 core; features/commercial tooling audit riêng |
| SQLite/JSON | Schema/event/TM/resume | SQLite public-domain; không DBserver mặc định |
| Translation/semantic | Adapter chưa chốt provider/model | Unknown quality/cost/license cho tới screening |

CUDA không chốt theo template cloud mới nhất. ORT/cuDNNmajor phải khớp; DirectML sustained engineering theo docsMicrosoft, WinML đang tiếp nhận tính năng mới. Nếu thửDirectML, env riêng và device_id kiểm đúng discreteGPU, không default0 trên laptop dualGPU.

## ADR

**ADR01 — Python orchestration, native kernels (ACCEPTED for prototype).** Dữ liệu nhỏ + mature engines trước. SourceOpenCV PyAllowThreads/PyAVcython.nogil cho thấy GIL không giữ toàn decode/CV. Pythonpixel loops/object copying vẫn có thể nghẽn. Đổi sangB khi profiler+AB chứng minh.

**ADR02 — Events làm intermediate durable (ACCEPTED).** IDs+PTS+revision+hash tách OCR/dịch/render; resume không trộn input/model/settings. SQLiteTM+JSONreport, chưa cần messagebroker/cloudmicroservices.

**ADR03 — Algorithm-first, OCR-on-candidates (ACCEPTED design, QUALITY UNPROVEN).** Scene/tile changes + detector watchdog; ML nhận chữ không thay geometry/timing. Short-event coverage benchmark trước chọncadence.

**ADR04 — NativeCPU renderer+NVENC baseline (PROVISIONAL).** Dễ triển khai; route CPU ASS phải đo. compositor GPU native không fake zero-copy; upgrade khi bottleneck thật.

**ADR05 — Không heavyweight resident models (ACCEPTED MVP).** Không CoTracker/SAM2/TransDETR/VLMdefault. Event-levelAI optional cóbudget và review, adapterreplaceable.

**ADR06 — SubAI độc lập, ModeA (ACCEPTED requirement).** Không sửatool; ModeB chờ compatibility evidence và phê duyệt.

**ADR07 — Không copyleft/unknown code copy vô điều kiện (ACCEPTED).** GPL/AGPL vẫn cho phép thương mại khi đáp ứng nghĩa vụ; không gọi chúng cấm bán. Scope derivative/distribution/network compliance phải xét riêng. Hiện VNLE chưa chọn distribution license, nên manga repos LEARNONLY, không portcode dưới tênADAPT để né license. Cách algorithm độc lập dùng OpenCV phải phân biệt source derivative.

**ADR08 — Không sử dụng code/model cũ mặc định (ACCEPTED).** CN2VI archived; SubAI logs chỉ là observations, không testschoVNLE. Research task không installweights/infer/buildproduct.

## Packaging sau này

Windows folder distribution trước onefile để giảm startup/unpack và debugDLL; lock interpreter/wheels/FFmpeg/fonts/model hash/VCruntime/license notices. Long Unicode paths, offline install/no auto downloads, incompatible driver errors và resume corruption test. Nuitka không tăng tốc ORTkernel ngoài và C++rewrite không sửa copy CPU→GPU tự động. Không tạo installer trong nhiệm vụ này.

## ADR09 — Manual subtitle exclusion: ACCEPTED

User khoanh vùng trước detection/OCR. Loại ngay ở input, không chạy model để tìm sub rồi bỏ. Transform/interval/hash thuộc manifest/cache. UI khoanh vùng là prototype tương lai, chưa được triển khai trong nhiệm vụ nghiên cứu.
