"""Test trực tiếp model bạn bạn trong WSL với tensor random.

ESP:  python test_local_model.py --source esp
ASUS: python test_local_model.py --source asus

Tensor random chỉ chứng minh Mamba/GPU/bundle chạy được, không có ý nghĩa HAR.
"""
from __future__ import annotations

import argparse
import numpy as np

from friend_model.predict import predict_asus, predict_esp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=("esp", "asus"), default="esp")
    args = parser.parse_args()

    shape = (1, 56, 1000) if args.source == "esp" else (4, 56, 1000)
    rng = np.random.default_rng(0)
    rx1, rx2, rx3 = (rng.random(shape, dtype=np.float32) for _ in range(3))

    predict_fn = predict_esp if args.source == "esp" else predict_asus
    print(f"source={args.source}")
    print(f"rx shape={shape}")
    label_id, label, probability = predict_fn(rx1, rx2, rx3)
    print(f"label_id={label_id}")
    print(f"label={label}")
    print(f"probability={probability:.6f}")
    print(f"percent={probability * 100.0:.2f}%")


if __name__ == "__main__":
    main()
