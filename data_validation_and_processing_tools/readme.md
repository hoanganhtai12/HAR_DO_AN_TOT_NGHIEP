# Installation

## 1. Tạo môi trường Python

Chỉ cần thực hiện một lần:

```cmd
python -m venv .venv
```

---

## 2. Kích hoạt môi trường

Tùy theo cấu hình của VS Code, môi trường có thể được tự động kích hoạt khi mở terminal trong thư mục dự án.

Nếu chưa được kích hoạt, chạy:

```cmd
.\.venv\Scripts\Activate.ps1
```

Sau khi kích hoạt thành công, terminal sẽ hiển thị tiền tố `(.venv)`.

Ví dụ:

```text
(.venv) PS D:\workspace\HUST\Personal\WiFi-Sensing\Code\csi-extractor
```

---

## 3. Cài đặt thư viện

Chỉ cần thực hiện một lần:

```cmd
pip install -r requirements.txt
```

---

## 4. Biên dịch `unpack.dll`

Nếu thư mục dự án chưa có file `unpack.dll`, thực hiện:

```cmd
gcc -O3 -shared unpack.c -o unpack.dll
```

---

# Chuẩn bị dữ liệu

Trước khi chạy các công cụ trong dự án, cần tạo thư mục dữ liệu:

```text
data/
```

Thư mục này sẽ là thư mục gốc chứa toàn bộ dữ liệu CSI và video thu được.

Ví dụ:

```text
project/
│
├── data/
│   ├── 1_1_1_4_4_10_sitdown_standup_0618_111303/
│   ├── 1_1_1_4_4_10_sitdown_standup_0618_112015/
│   └── ...
│
├── csi_config.json
├── requirements.txt
└── ...
```

Sau khi thu dữ liệu, sao chép các thư mục dữ liệu vào bên trong thư mục `data`.

Mặc định file `csi_config.json` được cấu hình:

```json
{
    "database_root": ".\\data"
}
```

Do đó hệ thống sẽ tự động tìm dữ liệu trong thư mục `data` nằm cùng cấp với mã nguồn dự án.

Nếu dữ liệu được lưu ở vị trí khác, hãy cập nhật giá trị `database_root` trong file `csi_config.json`.


# Running Scripts

## 1. Vẽ đồ thị từ file `.bin`

```cmd
python .\plot_bin.py
```

---

## 2. Vẽ đồ thị từ file `.npy`

```cmd
python .\plot_npy.py
```

---

## 3. Cắt video theo nhãn hành động

```cmd
python .\cut_video.py
```

---

## 4. Chuyển đổi dữ liệu CSI từ `.bin` sang `.npy`

```cmd
python .\csi_export_npy.py
```

---

## 5. Nhận CSI thời gian thực từ ASUS qua TCP

### 5.1. Khởi động TCP Server

```cmd
python .\test_csi_server.py
```

### 5.2. Khởi động công cụ vẽ thời gian thực

```cmd
python .\test_csi_realtime_plot.py
```

---

# Project Configuration

Cấu hình hệ thống được lưu trong file:

```text
csi_config.json
```

File này chứa:

* Đường dẫn cơ sở dữ liệu CSI (`database_root`)
* Chế độ debug (`debug`)
* Mô tả cấu trúc dữ liệu (`data_structure`)

Ví dụ:

```json
{
    "database_root": ".\\data",
    "debug": false,
    "data_structure": {
        ...
    }
}
```

Chi tiết xem phần **Cấu hình `csi_config.json`** trong tài liệu dự án.

# Cấu hình `csi_config.json`

File `csi_config.json` chứa toàn bộ cấu hình hệ thống, bao gồm:

- Đường dẫn cơ sở dữ liệu CSI.
- Chế độ debug: In ra nhiều thông tin hơn
- Mô tả cấu trúc dữ liệu (`data_structure`).

## Ví dụ

```json
{
    "database_root": ".\\data",
    "data_structure": {
        "sitdown": {
            "idx": 1,
            "room": [1],
            "session": [1],
            "user": [4],
            "pos": [4],
            "setup": [1],
            "repeat": [1,2,3,4,5,6,7,8,9,10]
        }
    }
}
```

---

## Trường `database_root`

Đường dẫn thư mục gốc chứa cơ sở dữ liệu CSI.

Ví dụ:

```json
{
    "database_root": "D:\\Dataset\\CSI"
}
```

hoặc

```json
{
    "database_root": ".\\data"
}
```

Đường dẫn tương đối sẽ được tính từ thư mục làm việc hiện tại của chương trình.

---

## Trường `data_structure`

Mô tả tập dữ liệu sẽ được sử dụng trong quá trình truy vấn, xuất dữ liệu hoặc huấn luyện mô hình.

Mỗi action được biểu diễn bởi một object có cấu trúc:

```json
{
    "<action_name>": {
        "idx": <action_id>,
        "room": [...],
        "session": [...],
        "user": [...],
        "pos": [...],
        "setup": [...],
        "repeat": [...]
    }
}
```

### Ý nghĩa các trường

| Trường    | Ý nghĩa                                |
| --------- | -------------------------------------- |
| `idx`     | Mã định danh của action                |
| `room`    | Danh sách phòng thu dữ liệu            |
| `session` | Danh sách phiên thu                    |
| `user`    | Danh sách người tham gia               |
| `pos`     | Danh sách vị trí                       |
| `setup`   | Danh sách hướng hoặc cấu hình thiết bị |
| `repeat`  | Danh sách lần lặp                      |

Ví dụ:

```json
{
    "pickup": {
        "idx": 3,
        "room": [1,2],
        "session": [1],
        "user": [1,2,3],
        "pos": [1,2,3],
        "setup": [1],
        "repeat": [1,2,3,4,5]
    }
}
```

Tương ứng với:

* Action: `pickup`
* Action ID: `3`
* Room: `1`, `2`
* Session: `1`
* User: `1`, `2`, `3`
* Position: `1`, `2`, `3`
* Setup: `1`
* Repeat: `1` đến `5`

---

## Danh sách Action mặc định

| ID | Action    |
| -- | --------- |
| 1  | sitdown   |
| 2  | standup   |
| 3  | pickup    |
| 4  | fall      |
| 5  | liestill  |
| 6  | walk      |
| 7  | run       |
| 8  | emptyroom |

---

## Quy tắc xử lý

Các module xử lý dữ liệu sẽ tự động duyệt toàn bộ tổ hợp:

```text
room × session × user × pos × setup × repeat
```

được khai báo trong `data_structure`.

Ví dụ:

```json
{
    "fall": {
        "idx": 4,
        "room": [1],
        "session": [1],
        "user": [1,2],
        "pos": [1,2],
        "setup": [1,2],
        "repeat": [1,2,3]
    }
}
```

Tổng số tổ hợp được tạo:

```text
1 × 1 × 2 × 2 × 2 × 3 = 24
```

---

## Giá trị rỗng

Nếu một trường là danh sách rỗng:

```json
{
    "room": []
}
```

thì vòng lặp không xảy ra, nên không truy xuất dữ liệu của hành động này.

---

## Khuyến nghị

* Không thay đổi giá trị `idx` của các action đã tồn tại.
* Sử dụng tên action duy nhất.
* Kiểm tra kỹ các danh sách `room`, `user`, `pos`, `setup`, `repeat` trước khi xuất dữ liệu để tránh bỏ sót mẫu.
* Nên quản lý toàn bộ cấu hình dataset trong `csi_config.json` thay vì sửa trực tiếp mã nguồn Python.

# Cấu trúc folder data đã tính toán pha, biên độ

- sshar là folder gốc.
- Bên trong chứa các folder phòng: ví dụ room_01
- Trong phòng có 2 folder là loại thiết bị: `asus` hoặc `esp`
- Bên trong mỗi loại thiết bị có 3 folder cho từng thiết bị: `rx_01`, `rx_02`, `rx_03`
- Trong mỗi thiết bị có 10 folder là `subject_01` đến `subject_10` tương ứng với 10 người.
- Bên trong mỗi người có 1280 file numpy, 1 nửa số file chứa giá trị biên độ, 1 nửa chứa giá trị pha.
- Cấu trúc tên file như sau: `amp_act{aa}_pos{pp}_dir{dd}_rep{rr}`:
    + `act{aa}` là hành động thứ `aa` (00 - 08)
    + `pos{pp}` là vị trí thứ `pp` (00-08)
    + `dir{dd}` là chiều của hành động. `00` là không có chiều, `01` là thuận kim đồng hồ, `02` là ngược chiều kim đồng hồ
    + `rep{rr}` là lần lặp lại `rr` (00-05 đối với `dd` = 1 hoặc 2, 00-10 đối với `dd` = 0)

# WiFi-HAR Dataset Structure

Dataset đã được tiền xử lý và lưu dưới dạng các file NumPy (`.npy`) chứa dữ liệu CSI dưới hai dạng:

* Amplitude (biên độ)
* Phase (pha)

## Directory Structure

```text
sshar/
│
├── room_01/
│   ├── asus/
│   │   ├── rx_01/
│   │   │   ├── subject_01/
│   │   │   │   ├── amp_act01_pos00_dir00_rep00.npy
│   │   │   │   ├── pha_act01_pos00_dir00_rep00.npy
│   │   │   │   └── ...
│   │   │   ├── subject_02/
│   │   │   └── ...
│   │   ├── rx_02/
│   │   └── rx_03/
│   │
│   └── esp/
│       ├── rx_01/
│       ├── rx_02/
│       └── rx_03/
│
├── room_02/
├── room_03/
└── ...
```

## Folder Description

### Root Folder

The dataset root directory is:

```text
sshar/
```

### Room

Each room corresponds to a different data collection environment:

```text
room_01
room_02
room_03
...
```

### Device Type

Each room contains CSI data collected from two device platforms:

```text
asus
esp
```

### Receiver Device

Each platform contains data from three receiver devices:

```text
rx_01
rx_02
rx_03
```

### Subject

Each receiver contains data collected from ten participants:

```text
subject_01
subject_02
...
subject_10
```

### Data Files

Each subject directory contains a total of 1280 NumPy files:

* 640 amplitude files (`amp_*.npy`)
* 640 phase files (`pha_*.npy`)

Each file corresponds to a specific activity instance.

---

# File Naming Convention

Amplitude files:

```text
amp_act{aa}_pos{pp}_dir{dd}_rep{rr}.npy
```

Phase files:

```text
pha_act{aa}_pos{pp}_dir{dd}_rep{rr}.npy
```

Example:

```text
amp_act06_pos03_dir01_rep04.npy
pha_act06_pos03_dir01_rep04.npy
```

---

# Parameter Description

| Field     | Description        |
| --------- | ------------------ |
| `act{aa}` | Activity label     |
| `pos{pp}` | Position index     |
| `dir{dd}` | Movement direction |
| `rep{rr}` | Repetition index   |

## Position

```text
pos = 01 - 08
```

corresponding to 9 predefined positions in the room.

## Direction

| Value | Description       |
| ----- | ----------------- |
| `00`  | No direction      |
| `01`  | Clockwise         |
| `02`  | Counter-clockwise |

## Repetition

For activities with:

```text
dir = 00
```

the repetition index is:

```text
rep = 01 - 10
```

(total 11 repetitions)

For activities with:

```text
dir = 01 or 02
```

the repetition index is:

```text
rep = 00 - 05
```

(total 6 repetitions for each direction)

---

# Activity Labels

| Activity ID | Activity Name  | Valid Direction |
| ----------- | -------------- | --------------- |
| `01`        | Sit Down       | `00`            |
| `02`        | Stand Up       | `00`            |
| `03`        | Pick Up Object | `00`            |
| `04`        | Fall           | `01`, `02`      |
| `05`        | Lie Still      | `01`, `02`      |
| `06`        | Walk           | `01`, `02`      |
| `07`        | Run            | `01`, `02`      |
| `08`        | Empty Room     | `00`            |

## Static Activities

The following activities do not have movement direction:

| ID   | Activity       |
| ---- | -------------- |
| `01` | Sit Down       |
| `02` | Stand Up       |
| `03` | Pick Up Object |
| `08` | Empty Room     |

For these activities:

```text
dir = 00
rep = 01 - 10
```

---

## Directional Activities

The following activities have movement direction:

| ID   | Activity  |
| ---- | --------- |
| `04` | Fall      |
| `05` | Lie Still |
| `06` | Walk      |
| `07` | Run       |

Direction values:

| Direction | Meaning           |
| --------- | ----------------- |
| `01`      | Clockwise         |
| `02`      | Counter-clockwise |

For each direction:

```text
rep = 01 - 05
```

---

# Example

File:

```text
amp_act06_pos03_dir01_rep04.npy
```

Meaning:

| Field      | Value     |
| ---------- | --------- |
| Data Type  | Amplitude |
| Activity   | Walk      |
| Position   | 03        |
| Direction  | Clockwise |
| Repetition | 04        |

---

# Dataset Statistics

For each:

```text
room_xx/
    device_type/
        rx_xx/
            subject_xx/
```

there are:

* 640 amplitude files
* 640 phase files
* 1280 files in total

All files are stored as:

```python
numpy.ndarray
```

and contain CSI amplitude or CSI phase data after preprocessing.
