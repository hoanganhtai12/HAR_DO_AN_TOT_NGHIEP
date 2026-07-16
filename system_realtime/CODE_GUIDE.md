# Hướng dẫn đọc code — bản Windows + WSL GPU

## 1. `config.py`

Nơi thường sửa:

- `ESP_HOST`, `ESP_PORT`, `ASUS_HOST`, `ASUS_PORT`: Collection TCP.
- MAC mappings của sáu receiver.
- `WINDOW_SIZE`, `RAW_BUFFER_SIZE`, `STEP_SIZE`.
- `ESP_DROP_SUBCARRIER_INDICES`.
- `ASUS_REORDER_SUBCARRIER_INDICES`, `ASUS_DROP_SUBCARRIER_INDICES`.
- `USE_ESP_WSL_MODEL`, `USE_ASUS_WSL_MODEL`.
- `WSL_INFERENCE_URL` nếu port WSL thay đổi.

## 2. `realtime_ai_app.py`

- Nhận JSON TCP từ Collection.
- Giữ sáu deque raw packet.
- Đủ data thì gọi `preprocess.py`, sau đó `predictor.py`.
- Lưu kết quả SQLite và broadcast WebSocket.
- SQLite chỉ lưu `input_timestamp` đã format thành thời gian bình thường.

## 3. `preprocess.py`

Không gọi GPU/model. Chỉ làm:

```text
seq 12-bit -> target 1000 seq -> amplitude -> NaN packet mất -> PCHIP
```

Output mỗi device:

```text
ESP  : (64,1000)
ASUS : (4,64,1000)
```

## 4. `predictor.py` (Windows)

- ESP: `(64,1000) -> (1,56,1000)`.
- ASUS: `(4,64,1000) -> reorder -> drop -> (4,56,1000)`.
- Nếu cờ WSL `False`: in tensor + random output.
- Nếu cờ WSL `True`: POST `source, rx1, rx2, rx3` tới Ubuntu port 8001.

## 5. `wsl_service/wsl_inference_service.py` (Ubuntu)

- Nhận HTTP từ Windows.
- Kiểm tra shape.
- Gọi một API chung `friend_model.predict.predict(rx1,rx2,rx3)`.
- `A=1` tự chọn `bundle/esp.pt`; `A=4` tự chọn `bundle/asus.pt`.
- Trả `label_id`, `label`, `probability 0..1` cho Windows.

## 6. Luồng khi ESP chạy thật

```text
ESP JSON
-> Windows raw deque
-> PCHIP ESP (64,1000)
-> predictor Windows (1,56,1000)
-> HTTP WSL
-> Haar + z-score + Mamba GPU
-> label/probability
-> Windows Dashboard + SQLite
```
