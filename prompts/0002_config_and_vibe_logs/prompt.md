Name:        Tạo VibeConfig Override & Luật Log Vibe Scripts
Description: Tạo class `VibeConfig` tự động kế thừa và chỉ cho phép override đúng tên biến của `src/config.py`, đồng thời bổ sung luật `LOG_VIBE_SCRIPTS.md` theo dõi các chức năng cải thiện và script trong `ROOT/src/vibe_scripts/`.
Purpose:     Loại bỏ việc copy-paste cấu hình thủ công giữa các lab (Fail-Fast khi sai tên biến) và lưu vết minh bạch toàn bộ script/cải tiến do AI Agent tạo.
Input:
- `ROOT/src/config.py`
- `ROOT/labs/lab1/vibe/config.py`
- `ROOT/.agents/rules/logs/`
- `ROOT/src/vibe_scripts/`
Output:
- `ROOT/.agents/rules/logs/LOG_VIBE_SCRIPTS.md` và `ROOT/logs/vibe_scripts/README.md`
- `ROOT/src/vibe_scripts/config_override.py` và `ROOT/test/test_config_override.py`
- Refactor `ROOT/labs/lab1/vibe/config.py`
How to do:   Tuân thủ `.agents/rules/` (lưu plan, Dual Testing Smoke + Normal, kiểm tra link, tách 2 atomic commits).
Examples:    `cfg = VibeConfig(LR=1e-3, EPOCH=20, extras={"BEST_METRIC": "val_loss"})`
Reference:
- `ROOT/.agents/rules/BEHAVIORAL_LAYER.md`
- `ROOT/.agents/rules/PLANNING.md`
- `ROOT/.agents/rules/logs/README.md`
