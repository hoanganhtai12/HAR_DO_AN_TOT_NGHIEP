# WSL GPU Inference Service

Thư mục này chạy trong Ubuntu/WSL, không chạy trên Windows.

- `friend_model/bundle/esp.pt` đã có sẵn.
- Khi nhận model ASUS, chép `asus.pt` vào `friend_model/bundle/asus.pt`.
- `wsl_inference_service.py` nhận ba receiver qua HTTP và gọi cùng API model:
  `predict(rx1, rx2, rx3)`.

## Cài API dependencies

```bash
source ~/env/bin/activate
cd ~/har_inference/wsl_service
python -m pip install -r requirements_wsl_service.txt
```

## Test model ESP trực tiếp

```bash
python test_local_model.py --source esp
```

Kết quả random chỉ chứng minh Mamba/GPU/bundle chạy được.

## Chạy API cho Windows

```bash
python -m uvicorn wsl_inference_service:app --host 0.0.0.0 --port 8001
```

Giữ terminal này mở. Trong PowerShell Windows kiểm tra:

```powershell
Invoke-RestMethod http://127.0.0.1:8001/health
```

Khi `bundle_available.esp = True`, Windows có thể bật `USE_ESP_WSL_MODEL = True`.
