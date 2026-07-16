"""Test Windows -> WSL HTTP bridge bằng tensor random.

Chạy từ PowerShell trong project Windows:
    python tools\test_wsl_bridge.py --source esp

- Cần WSL service đang chạy port 8001.
- ESP cần bundle/esp.pt (đã có trong wsl_service).
- ASUS chỉ chạy được sau khi chép bundle/asus.pt vào WSL.
- Dữ liệu random chỉ kiểm tra đường truyền + model, không có ý nghĩa action.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import WSL_HEALTH_URL, WSL_INFERENCE_TIMEOUT_SEC, WSL_INFERENCE_URL  # noqa: E402


def get_json(url: str) -> object:
    with urlopen(url, timeout=10.0) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=("esp", "asus"), default="esp")
    args = parser.parse_args()

    try:
        print("[WINDOWS] health:")
        print(json.dumps(get_json(WSL_HEALTH_URL), ensure_ascii=False, indent=2))
    except URLError as exc:
        raise SystemExit(
            "Không kết nối được WSL service. Mở Ubuntu và chạy "
            "python -m uvicorn wsl_inference_service:app --host 0.0.0.0 --port 8001"
        ) from exc

    shape = (1, 56, 1000) if args.source == "esp" else (4, 56, 1000)
    rng = np.random.default_rng(0)
    payload = {
        "source": args.source,
        "rx1": rng.random(shape, dtype=np.float32).tolist(),
        "rx2": rng.random(shape, dtype=np.float32).tolist(),
        "rx3": rng.random(shape, dtype=np.float32).tolist(),
    }
    request = Request(
        WSL_INFERENCE_URL,
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=WSL_INFERENCE_TIMEOUT_SEC) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"WSL service trả HTTP {exc.code}: {detail}") from exc

    print("[WINDOWS] result:")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
