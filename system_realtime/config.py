"""Cấu hình tập trung cho Realtime HAR CSI.

Windows chạy Collection, preprocess, Dashboard và SQLite.
WSL Ubuntu chạy model Mamba trên GPU qua HTTP nội bộ.
Chủ yếu sửa file này khi đổi MAC, IP/port, buffer, subcarrier hoặc bật model.
"""

from pathlib import Path

# =========================================================
# PATH CỦA PROJECT WINDOWS
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
STATIC_INDEX = BASE_DIR / "static" / "index.html"
DB_PATH = BASE_DIR / "realtime_log.db"
UNPACK_DLL_PATH = BASE_DIR / "unpack.dll"

# =========================================================
# KẾT NỐI ĐẾN HAI COLLECTION (TCP JSON Lines)
# =========================================================
ESP_HOST, ESP_PORT = "127.0.0.1", 9201
ASUS_HOST, ASUS_PORT = "127.0.0.1", 9100
RECONNECT_DELAY_SEC = 2.0
SOURCE_TIMEOUT_SEC = 5.0

# =========================================================
# MAC THẬT -> ID LOGIC
# =========================================================
ESP_MAC_TO_ID = {
    "D0:CF:13:ED:2E:EC": "esp1",
    "D0:CF:13:EB:8A:9C": "esp2",
    "D0:CF:13:EC:49:04": "esp3",
}
ASUS_MAC_TO_ID = {
    "04:D4:C4:B5:8E:7C": "asus1",
    "04:D4:C4:B8:76:64": "asus2",
    "04:D4:C4:1C:0A:C4": "asus3",
}
GROUP_DEVICES = {
    "esp": ("esp1", "esp2", "esp3"),
    "asus": ("asus1", "asus2", "asus3"),
}

# =========================================================
# BUFFER, SEQ VÀ CỬA SỔ THỜI GIAN
# =========================================================
WINDOW_SIZE = 1000
RAW_BUFFER_SIZE = 1100
STEP_SIZE = 300
SEQ_MODULO = 4096
SEQ_HALF_RANGE = SEQ_MODULO // 2

# =========================================================
# PACKET MẤT + PCHIP
# =========================================================
INTERPOLATION_METHOD = "pchip"
MAX_CONSECUTIVE_MISSING = 10
MAX_MISSING_RATIO = 0.1
USE_TIMESTAMP_CHECK = False
MAX_TIMESTAMP_SKEW_US = 20_000

# =========================================================
# LABELS FALLBACK / MOCK — KHÔNG DÙNG LABELS JSON Ở WINDOWS
# =========================================================
# Model thật trong WSL trả label string của chính bundle model.
# Mảng này chỉ dùng khi đang mock hoặc WSL service lỗi rồi fallback random.
HAR_LABELS = (
    "Sit Down",
    "Stand Up",
    "Pick Up Object",
    "Fall",
    "Lie Still",
    "Walk",
    "Run",
    "Empty Room",
)

# =========================================================
# WSL GPU INFERENCE BRIDGE
# =========================================================
# Ubuntu/WSL chạy wsl_service/wsl_inference_service.py tại port 8001.
# Windows gọi 127.0.0.1:8001 qua loopback, không cần mở port LAN.
# WSL_INFERENCE_URL = "http://127.0.0.1:8001/predict"
WSL_HEALTH_URL = "http://172.27.58.229:8001/health"


WSL_INFERENCE_URL = "http://172.27.58.229:8001/predict"
WSL_INFERENCE_TIMEOUT_SEC = 45.0

# False: in tensor rồi trả random prediction để test pipeline.
# True : gửi tensor qua HTTP tới WSL để model Mamba chạy GPU thật.
USE_ESP_WSL_MODEL = True
USE_ASUS_WSL_MODEL = True

# WSL service/model lỗi thì không làm chết Dashboard. Server sẽ in lỗi và fallback random.
FALLBACK_TO_RANDOM_ON_MODEL_ERROR = False

# Khi mock/fallback random, có in rx1/rx2/rx3 để kiểm tra tensor.
# Khi đã chạy model thật nên đặt False để terminal không chậm.
PRINT_MODEL_INPUT_TENSORS = True
PRINT_FULL_MODEL_INPUT_TENSORS = False
MOCK_PROBABILITY_MIN = 0.50
MOCK_PROBABILITY_MAX = 0.99

# =========================================================
# ASUS: JSON raw -> unpack.dll -> PCHIP (4,64,1000)
# =========================================================
ASUS_ANTENNAS = ("c0", "c1", "c2", "c3")
ASUS_RAW_SUBCARRIERS = 64
ASUS_SUBCARRIERS = ASUS_RAW_SUBCARRIERS

# Tham số unpack Nexmon. Không đổi nếu dataset train dùng cùng unpack.c.
ASUS_NFFT = 64
ASUS_NMAN = 12
ASUS_NEXP = 6
ASUS_NBITS = 10
ASUS_AUTOSCALE = 0
ASUS_SHFT = 0
ASUS_FMT = 0

# Raw order từ Nexmon: 0..+31, -32..-1.
# Model order:             -32..-1, 0..+31.
# Đây là index trong tensor RAW (4,64,1000) để làm reorder theo trục subcarrier.
ASUS_REORDER_SUBCARRIER_INDICES: tuple[int, ...] = (
    *range(32, 64),
    *range(0, 32),
)

# INDEX NÀY TÍNH SAU KHI ĐÃ REORDER.
# Cấu hình hiện tại giữ -28..-1, +1..+28 => 56 subcarrier.
# Bỏ -32,-31,-30,-29,0,+29,+30,+31 => index 0,1,2,3,32,61,62,63.
ASUS_MODEL_SUBCARRIERS = 56
ASUS_DROP_SUBCARRIER_INDICES: tuple[int, ...] | None = (
    0, 1, 2, 3,
    32,
    61, 62, 63,
)

# =========================================================
# ESP: PCHIP (64,1000) -> model WSL (1,56,1000)
# =========================================================
ESP_SUBCARRIERS = 64
ESP_MODEL_SUBCARRIERS = 56

# Bỏ index 28 và 57..63 theo model ESP của bạn bạn.
ESP_DROP_SUBCARRIER_INDICES: tuple[int, ...] = (
    28,
    57, 58, 59, 60, 61, 62, 63,
)
