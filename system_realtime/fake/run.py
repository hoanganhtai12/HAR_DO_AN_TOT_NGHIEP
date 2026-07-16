"""Chạy đồng thời ESP32-Collection giả và Nexmon-Collection (ASUS) giả.

Thay vì mở hai cửa sổ PowerShell riêng, file này chạy cả hai trong một tiến
trình, mỗi collection nằm trên một thread daemon.

Yêu cầu: đặt file này CÙNG THƯ MỤC với:
    - esp_fake.py
    - asus_fake_bin.py
    - tcp_stream_server.py

Cách dùng:
    python run_collections.py            # chạy cả ESP (9201) và ASUS (9100)
    python run_collections.py --only esp
    python run_collections.py --only asus

Nhấn Ctrl+C để dừng cả hai.

Lưu ý: hai collection vẫn mở hai TCP server ở hai port khác nhau (ESP 9201,
ASUS 9100), giống hệt khi chạy riêng. Gộp ở đây chỉ để tiện một cửa sổ.
"""

from __future__ import annotations

import argparse
import threading
import time

import esp_fake
import asus_fake_bin


def run_esp() -> None:
    try:
        esp_fake.main()
    except Exception as exc:  # noqa: BLE001
        print(f"[run_collections] ESP collection dừng do lỗi: {exc}")


def run_asus() -> None:
    try:
        asus_fake_bin.main()
    except Exception as exc:  # noqa: BLE001
        print(f"[run_collections] ASUS collection dừng do lỗi: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chạy đồng thời ESP và ASUS collection giả trong một tiến trình."
    )
    parser.add_argument(
        "--only",
        choices=("esp", "asus", "both"),
        default="both",
        help="Chọn collection để chạy. Mặc định both.",
    )
    args = parser.parse_args()

    threads: list[threading.Thread] = []

    if args.only in ("esp", "both"):
        threads.append(threading.Thread(target=run_esp, name="esp-collection", daemon=True))

    if args.only in ("asus", "both"):
        threads.append(threading.Thread(target=run_asus, name="asus-collection", daemon=True))

    for thread in threads:
        thread.start()
        time.sleep(0.3)  # cách nhau chút để log hai server không chen vào nhau.

    print("[run_collections] Đã khởi động. Nhấn Ctrl+C để dừng.")

    try:
        # Giữ tiến trình chính sống để các thread daemon tiếp tục chạy.
        while any(thread.is_alive() for thread in threads):
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[run_collections] Nhận Ctrl+C, đang dừng...")
        # Thread là daemon nên sẽ tự kết thúc khi tiến trình chính thoát.


if __name__ == "__main__":
    main()