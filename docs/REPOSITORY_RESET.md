# Reset repository — kết quả và khôi phục

Ngày 09/10/2026. Repo đã xác nhận bằng Git root và remote: `D:\Video`, `https://github.com/Khenggg/cn2vi-autodub.git`. Nhánh trước `codex/v2-smart-voiceover`, HEAD `7c7ed60`; tracked working tree sạch trước reset. Có untracked tài liệu/script SubAI, results và subai-python, cùng ignored cache/credentials/media.

## Bản lưu đã xác minh

- Nhánh lưu: `codex/backup-cn2vi-before-vnle-20261009-152059`; nhánh làm việc mới: `codex/vnle-research-reset`.
- Kho riêng: `D:\Video\.legacy-backup\cn2vi-20261009-152059`. `verified-copy/` chứa 9.640 tệp, 1.258.417.187 byte, đã so SHA-256 từng tệp với nguồn **trước khi đưa khỏi active tree**.
- `original-tree/` giữ nguyên các thành phần được chuyển khỏi repo, không xóa vật lý. Cùng volume, đường nguồn/đích được xác minh nằm trong `D:\Video` trước mọi Move-Item.
- `history.bundle` giữ toàn bộ refs Git lúc backup và đã qua `git bundle verify`. `.git` vẫn giữ nguyên, không reset/clean hoặc rewrite history.
- `git-status.txt`, `tracked-files.txt`, `uncommitted.patch`, `reset-plan.json`, `sha256-manifest.json` ghi trạng thái/kế hoạch trước reset. Tracked patch rỗng; untracked được lưu riêng cùng bản sao.

Kho lưu có thể chứa key/token/media và mã nghiên cứu riêng. `.legacy-backup/` nằm trong .gitignore, không commit/upload. Bản lưu trong cùng ổ bảo vệ khỏi reset nhầm, **không phải backup chống hỏng ổ**; chưa có bản offsite trong nhiệm vụ này.

## Thành phần đã loại khỏi active product

Toàn bộ `src/autodub`, frontend, tests/benchmarks CN2VI, config/requirements, scripts triển khai/model/cloud/SubAI, docs CN2VI và tài liệu SubAI cũ, results, subai-python, workflow CI cũ, Docker/compose/install/pyproject/locks/env mẫu, output và hf_token root được lưu rồi chuyển vào `original-tree`. Danh sách chính xác ở reset-plan.json; source ASR/TTS/separation/ducking/dialogue pipelines không còn trong working tree sản phẩm.

README/AGENTS/.gitignore được viết lại; bản trước nằm trong verified-copy và Git history. `.gitattributes` giữ cấu hình text/binary thông dụng. Không thay remote hoặc push. Ứng dụng SubAI, package Windows và tài sản **ngoài repo** không bị sửa bởi reset này.

## Phần local giữ nguyên tại chỗ

`.cache`, `.venv`, `.playwright-cli`, các pytest/ruff cache không dùng làm dependency sản phẩm mới, đều Gitignored. Inventory riêng ghi 83.437 tệp nhìn thấy và 2 lỗi truy cập; **không tuyên bố đã sao lưu hết các thư mục này**. Thư mục không truy cập được giữ nguyên, không xóa. Credentials/video riêng trong cache vẫn ở chỗ cũ. Repo sạch về source/product và Git scope, không phải mọi byte local bị xóa.

## Khôi phục an toàn vào checkout khác trong kho riêng

Không chạy các lệnh này trong nhiệm vụ nghiên cứu; đây là hướng dẫn khi cần. Không dùng reset --hard/clean lên nhánh VNLE.

```powershell
$restoreRoot = 'D:\Video\.legacy-backup\cn2vi-20261009-152059\restored-code'
git -C 'D:\Video' worktree add $restoreRoot 'codex/backup-cn2vi-before-vnle-20261009-152059'
robocopy 'D:\Video\.legacy-backup\cn2vi-20261009-152059\verified-copy' $restoreRoot /E /R:1 /W:1
```

`robocopy /E` bổ sung bản dirty/untracked/private đã lưu; **không dùng /MIR hoặc /PURGE**. Git worktree chứa code gốc, bản sao khôi phục metadata/untracked; thư mục cache được giữ nguyên riêng và cần ánh xạ lại đường dẫn nếu chạy lại CN2VI. Không chép credentials vào Git, không dùng env cũ để nghiệm thu VNLE.

Nếu Git database gốc mất nhưng bundle còn: `git clone --branch codex/backup-cn2vi-before-vnle-20261009-152059 history.bundle restored-from-bundle` từ kho riêng, rồi chép verified-copy như trên. Kiểm manifest hash trước dùng; bundle chỉ chứa lịch sử Git, không thay bản lưu untracked/ignored.

## Cấu trúc mới

README, AGENTS, .gitignore, .gitattributes và docs00–13 cùng báo cáo reset/nghiên cứu. Không có code sản phẩm, dependency lock hoặc CI giả chạy test cũ. Tài liệu có budgets/bench plan/quality gates, chưa inference hoặc installweights. Static validation tài liệu khác với chứng nhận runtime.
