# Nhận dạng hành động từ CSI Wi-Fi — Hướng dẫn cho nhóm IoT

Model đã train sẵn. Bên IoT chỉ cần gọi 1 hàm: `predict_esp(...)` (thiết bị 1
anten) hoặc `predict_asus(...)` (4 anten). Đưa vào 3 mảng CSI, nhận về tên hành
động + độ tin cậy.

## 1. Cài đặt (cần GPU NVIDIA + Linux/WSL)

`mamba-ssm` chỉ chạy trên **Linux** (Windows dùng WSL2) với **GPU NVIDIA**.
Khuyến nghị **Ubuntu 24.04** (có sẵn Python 3.12, khớp wheel `cp312` bên dưới).
Cách chắc chắn nhất là dùng **wheel dựng sẵn** — KHÔNG build từ nguồn.

Bí quyết tránh lỗi `undefined symbol`: **torch và wheel phải cùng C++ ABI**.
torch 2.7 (PyPI) dùng ABI = TRUE, nên phải lấy wheel tag `...cxx11abiTRUE`.

```bash
# gói hệ thống — build-essential cho trình biên dịch C (triton cần lúc import)
sudo apt install -y python3.12-venv python3-dev build-essential curl

python3 -m venv env && source env/bin/activate      # Python 3.12
pip install --upgrade pip
pip install torch==2.7.0                              # PyPI, ABI = TRUE, kèm CUDA 12.6
pip install numpy PyWavelets einops packaging        # packaging: mamba cần, --no-deps bỏ qua

# Tải 2 wheel khớp: torch2.7 · cp312 · cxx11abiTRUE  (từ GitHub releases)
#   https://github.com/Dao-AILab/causal-conv1d/releases
#   https://github.com/state-spaces/mamba/releases
pip install --no-deps causal_conv1d-*+cu12torch2.7cxx11abiTRUE-cp312-*.whl
pip install --no-deps mamba_ssm-*+cu12torch2.7cxx11abiTRUE-cp312-*.whl
```

> `--no-deps` để pip KHÔNG tự nâng torch (nếu không sẽ kéo bản khác về, lệch ABI).
> Kiểm tra ngay (chỉ cần mỗi file `test_install.py`, chưa cần gói model):
> `python test_install.py` → phải ra `TAT CA OK`.
> Trên **Kaggle/Colab** (GPU) thì đơn giản hơn: `pip install mamba-ssm causal-conv1d --no-build-isolation`.

## 2. Đặt file

Giải nén gói, đặt cạnh code của bạn. Có 2 phần:

**Phần của gói (không sửa):**
```
predict.py  model.py  preprocess.py  bundle/esp.pt   (và/hoặc bundle/asus.pt)
```

**Phần của bạn:** một file `.py` do bạn tự viết (vd `iot_main.py`) đặt **cùng thư
mục** với các file trên, trong đó `import` và gọi hàm predict (xem Bước 3).

```
iot_main.py        ← code CỦA BẠN: import predict_esp/asus, truyền dữ liệu, nhận kết quả
predict.py         ┐
model.py           ├ của gói — để nguyên
preprocess.py      │
bundle/esp.pt      ┘  (asus.pt nếu bạn dùng thiết bị 4 anten)
```

> 3 file `.py` của gói + thư mục `bundle/` luôn đi cùng nhau. Chạy `iot_main.py`
> ngay trong thư mục này để hàm predict tự tìm thấy `bundle/`.

## 3. Gọi hàm predict

Chọn hàm theo thiết bị của bạn:

```python
from predict import predict_esp        # ESP  — 1 anten/receiver
label, name, percent = predict_esp(rx1, rx2, rx3)
print(f"{name}  {percent*100:.1f}%")   # vd: Walk  94.1%
```

```python
from predict import predict_asus       # ASUS — 4 anten/receiver
label, name, percent = predict_asus(rx1, rx2, rx3)
```

**Vòng real-time** — phần đọc CSI là của bạn; cứ gom đủ 1000 gói cho 3 receiver
thì gọi 1 lần:

```python
from predict import predict_esp

while True:
    rx1, rx2, rx3 = doc_csi_tu_hardware()      # ← code CỦA BẠN, trả 3 mảng đúng shape
    label, name, percent = predict_esp(rx1, rx2, rx3)
    print(f"{name}  {percent*100:.1f}%")
```

**Input** — `rx1, rx2, rx3` là 3 receiver (thứ tự cố định 0, 1, 2), mỗi cái là
mảng biên độ shape `(anten, 56 subcarrier, 1000 gói)`:

| Hàm | shape mỗi rx |
|---|---|
| `predict_esp`  | `(1, 56, 1000)` |
| `predict_asus` | `(4, 56, 1000)` |

> 🔴 Phải đúng **56 subcarrier** (không phải 64). Gọi nhầm hàm (sai số anten)
> sẽ báo lỗi ngay.

**Output** — 3 giá trị:

| | Kiểu | Ý nghĩa |
|---|---|---|
| `label` | int 1..8 | mã hành động |
| `name` | str | tên hành động |
| `percent` | float 0..1 | độ tin cậy (× 100 ra %) |

8 hành động (label 1..8): Sit Down · Stand Up · Pick Up Object · Fall ·
Lie Still · Walk · Run · Empty Room.

> Lần gọi đầu chậm vài giây (nạp model lên GPU), các lần sau rất nhanh.
