# Cập nhật package model mới của bạn bạn

## Windows

Giữ các file gốc ở root project:

```text
config.py, predictor.py, preprocess.py, realtime_ai_app.py
```

Chỉ đổi `HAR_LABELS` trong config sang thứ tự model mới. Không chép `model.py` hoặc `predict.py` của bạn bạn vào root Windows.

## WSL

Copy toàn bộ `wsl_service/` trong project sang:

```text
~/har_inference/wsl_service/
```

Lệnh an toàn (ghi đè code WSL và bundle):

```bash
source ~/env/bin/activate
mkdir -p ~/har_inference/wsl_service
cp -r "/mnt/c/Users/hoang/Downloads/ĐATN/realtime_har_wsl/wsl_service/." \
  ~/har_inference/wsl_service/
cd ~/har_inference/wsl_service
python -m pip install -r requirements_wsl_service.txt
python test_local_model.py --source esp
python test_local_model.py --source asus
```

Sau khi cả hai test thành công:

```bash
python -m uvicorn wsl_inference_service:app --host 0.0.0.0 --port 8001
```

Trên Windows:

```powershell
Invoke-RestMethod http://127.0.0.1:8001/health
python tools\test_wsl_bridge.py --source esp
python tools\test_wsl_bridge.py --source asus
```

Chỉ khi test bridge thành công, đổi trong Windows `config.py`:

```python
USE_ESP_WSL_MODEL = True
USE_ASUS_WSL_MODEL = True
FALLBACK_TO_RANDOM_ON_MODEL_ERROR = False
PRINT_MODEL_INPUT_TENSORS = False
```
