# Realtime HAR — Windows Dashboard + WSL GPU Mamba

## Kiến trúc đã đổi

```text
Windows
ESP/ASUS Collection
        ↓ TCP JSON Lines
realtime_ai_app.py
        ↓
preprocess.py: seq + amplitude + PCHIP
        ↓
predictor.py: reorder/bỏ sub
        ↓ HTTP localhost:8001
WSL Ubuntu + NVIDIA GPU
wsl_service/wsl_inference_service.py
        ↓
friend_model.predict(rx1, rx2, rx3)
        ↓
Dashboard WebSocket + SQLite
```

Windows **không cần** `torch`, `mamba-ssm` hoặc `causal-conv1d`. Windows vẫn dùng `unpack.dll` cho ASUS.
WSL mới chạy Torch/Mamba/GPU.

## Bốn file/thư mục đã sửa hoặc thêm

```text
config.py                   # cờ bật WSL model + URL WSL + subcarrier
predictor.py                # gửi tensor ESP/ASUS sang WSL bằng HTTP
realtime_ai_app.py          # SQLite chỉ lưu input_timestamp dạng thời gian bình thường
wsl_service/                # FastAPI + model Mamba chạy trong Ubuntu
```

## 1. Chạy Windows ở chế độ mock trước

Trong `config.py`, giữ:

```python
USE_ESP_WSL_MODEL = False
USE_ASUS_WSL_MODEL = False
```

Khi đó pipeline vẫn chạy như trước: in tensor và trả prediction random. Test shape:

```powershell
python tools\test_model_input_shapes.py
python tools\test_asus_reorder_and_drop.py
```

## 2. Cài/copy service vào Ubuntu WSL

Giả sử project Windows được giải nén tại:

```text
C:\Users\hoang\Downloads\realtime_har_wsl_bridge
```

Mở Ubuntu rồi chạy:

```bash
source ~/env/bin/activate
mkdir -p ~/har_inference
cp -r /mnt/c/Users/hoang/Downloads/realtime_har_wsl_bridge/wsl_service ~/har_inference/
cd ~/har_inference/wsl_service
python -m pip install -r requirements_wsl_service.txt
```

Test model ESP trực tiếp trong WSL:

```bash
python test_local_model.py --source esp
```

Kết quả với random chỉ xác nhận bundle ESP + Mamba + GPU chạy được.

Chạy API WSL, giữ terminal này mở:

```bash
python -m uvicorn wsl_inference_service:app --host 0.0.0.0 --port 8001
```

## 3. Kiểm tra Windows gọi được WSL

Mở PowerShell tại thư mục project Windows:

```powershell
Invoke-RestMethod http://127.0.0.1:8001/health
python tools\test_wsl_bridge.py --source esp
```

`test_wsl_bridge.py` gửi tensor random `(1,56,1000)` x 3; nhãn random-test không có ý nghĩa HAR, chỉ xác nhận Windows ↔ WSL ↔ GPU ↔ model đã thông.

## 4. Bật ESP model thật

Sau khi bước 3 chạy được, sửa trong `config.py` Windows:

```python
USE_ESP_WSL_MODEL = True
PRINT_MODEL_INPUT_TENSORS = False
```

Giữ ASUS mock:

```python
USE_ASUS_WSL_MODEL = False
```

Sau đó chạy server Windows như cũ:

```powershell
python -m pip install -r requirements.txt
python -m uvicorn realtime_ai_app:app --host 0.0.0.0 --port 8000
```

## 5. Bật ASUS sau này

Khi bạn bạn gửi model ASUS cùng kiến trúc API:

1. Chép file vào WSL:

```bash
cp /mnt/c/DUONG_DAN/asus.pt ~/har_inference/wsl_service/friend_model/bundle/asus.pt
```

2. Kiểm tra model ASUS bằng input random:

```bash
cd ~/har_inference/wsl_service
source ~/env/bin/activate
python test_local_model.py --source asus
```

3. Nếu ASUS model được train với đúng `(4,56,1000)`, bật trong Windows `config.py`:

```python
USE_ASUS_WSL_MODEL = True
```

Không sửa Dashboard, SQLite, `preprocess.py` hoặc `unpack.dll`.

## Input contract

```text
ESP Windows preprocess : (64,1000)
ESP WSL model input    : (1,56,1000) cho mỗi esp1/esp2/esp3

ASUS Windows preprocess: (4,64,1000)
ASUS WSL model input   : (4,56,1000) cho mỗi asus1/asus2/asus3
```

ASUS luôn reorder trước rồi bỏ subcarrier theo `ASUS_DROP_SUBCARRIER_INDICES`; index bỏ tính sau reorder.

## SQLite

Bảng `prediction_logs` mới lưu:

```text
id, web_updated_at, source, input_timestamp, latency_ms, action, confidence
```

`input_timestamp` là chuỗi thời gian bình thường. Không lưu `input_timestamp_us` vào SQLite.

Khi đang dùng database cũ, dừng server và xóa file DB cũ **một lần** trước khi chạy bản mới:

```powershell
Remove-Item .\realtime_log.db
```
