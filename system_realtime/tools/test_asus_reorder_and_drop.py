"""Test đúng thứ tự ASUS: raw 0..31,-32..-1 -> reorder -> bỏ 8 sub.

Chạy:
    python tools\\test_asus_reorder_and_drop.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import (
    ASUS_DROP_SUBCARRIER_INDICES,
    ASUS_MODEL_SUBCARRIERS,
    ASUS_REORDER_SUBCARRIER_INDICES,
    WINDOW_SIZE,
)
from predictor import drop_asus_subcarriers, reorder_asus_subcarriers


def main() -> None:
    # Mỗi raw subcarrier mang giá trị đúng bằng raw index của nó để dễ theo dõi.
    x = np.zeros((4, 64, WINDOW_SIZE), dtype=np.float32)
    for raw_index in range(64):
        x[:, raw_index, :] = raw_index

    ordered = reorder_asus_subcarriers(x)
    final = drop_asus_subcarriers(x)

    print("Raw order index:      0..31, 32..63")
    print("Signed raw labels:    0..31, -32..-1")
    print("Reorder raw indices:", list(ASUS_REORDER_SUBCARRIER_INDICES))
    print("Drop after reorder:", ASUS_DROP_SUBCARRIER_INDICES)
    print("Ordered shape:", ordered.shape)
    print("Final shape:", final.shape)
    print("Final raw positions (antenna 0, time 0):")
    print(final[0, :, 0].astype(int).tolist())

    expected_raw_positions = list(range(36, 64)) + list(range(1, 29))
    actual = final[0, :, 0].astype(int).tolist()
    assert actual == expected_raw_positions, (actual, expected_raw_positions)
    assert final.shape == (4, ASUS_MODEL_SUBCARRIERS, WINDOW_SIZE)
    print("OK: giữ signed subcarrier -28..-1, +1..+28.")


if __name__ == "__main__":
    main()
