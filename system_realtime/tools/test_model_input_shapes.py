"""Test shape input model và mock predict, không cần model thật."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from predictor import (
    predict_asus,
    predict_esp,
    prepare_asus_for_friend_model,
    prepare_esp_for_friend_model,
)


def main() -> None:
    esp = np.ones((64, 1000), dtype=np.float32)
    asus = np.ones((4, 64, 1000), dtype=np.float32)

    esp_input = prepare_esp_for_friend_model(esp)
    asus_input = prepare_asus_for_friend_model(asus)

    print("ESP model input :", esp_input.shape)   # (1, 56, 1000)
    print("ASUS model input:", asus_input.shape) # (4, 64, 1000) hiện tại

    # predict_* sẽ in tensor và trả random result khi model thật đang tắt.
    print("ESP mock result :", predict_esp(esp, esp, esp))
    print("ASUS mock result:", predict_asus(asus, asus, asus))


if __name__ == "__main__":
    main()
