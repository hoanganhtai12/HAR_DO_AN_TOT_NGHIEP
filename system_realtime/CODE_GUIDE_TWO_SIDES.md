# Kiến trúc hai phía: Windows + WSL

## 1. Windows làm gì?

Windows giữ toàn bộ phần liên quan thiết bị và Dashboard:

```text
ESP/ASUS Collection TCP JSON Lines
    -> realtime_ai_app.py: buffer 1200 raw packet / device
    -> preprocess.py: chọn 1000 seq chung + amplitude + packet thiếu NaN + PCHIP
    -> predictor.py: ESP bỏ sub; ASUS reorder rồi bỏ sub
    -> HTTP localhost:8001
    -> Dashboard WebSocket + SQLite
```

Không đặt `torch`, `mamba_ssm`, `model.py` hay checkpoint vào code chạy Windows.
`unpack.dll` ASUS vẫn chỉ chạy Windows.

## 2. WSL làm gì?

WSL chỉ giữ model và GPU:

```text
wsl_service/friend_model/
  predict.py      API do nhóm AI cung cấp
  model.py        WavDualMamba
  preprocess.py   Haar DWT + z-score do nhóm AI cung cấp
  bundle/esp.pt
  bundle/asus.pt

wsl_inference_service.py
  POST /predict
  source='esp'  -> predict_esp(rx1, rx2, rx3)
  source='asus' -> predict_asus(rx1, rx2, rx3)
```

## 3. Hai file cùng tên `preprocess.py` KHÁC NHAU

- `Windows/preprocess.py`: PCHIP, seq, unpack ASUS, tạo amplitude 64 subcarrier.
- `WSL/wsl_service/friend_model/preprocess.py`: Haar DWT + z-score đúng như lúc train.

Không copy đè file này lên file kia.

## 4. Hợp đồng tensor

```text
ESP Windows -> WSL:  rx1/rx2/rx3, mỗi tensor (1, 56, 1000)
ASUS Windows -> WSL: rx1/rx2/rx3, mỗi tensor (4, 56, 1000)
```

Không có batch dimension khi gửi HTTP. Model WSL tự thêm batch dimension sau Haar/z-score.

## 5. Hợp đồng kết quả

```json
{
  "label_id": 1,
  "label": "Sit Down",
  "probability": 0.94
}
```

Windows đổi probability thành percent để Dashboard dùng, nhưng SQLite/WebSocket vẫn lưu confidence `0..1`.

## 6. Thứ tự label (cả ESP và ASUS)

1 Sit Down
2 Stand Up
3 Pick Up Object
4 Fall
5 Lie Still
6 Walk
7 Run
8 Empty Room

Windows không ánh xạ lại `label_id` khi nhận model thật; dùng luôn `label` WSL trả.
