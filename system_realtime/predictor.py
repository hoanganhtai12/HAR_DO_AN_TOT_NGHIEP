"""Adapter inference cho Realtime HAR.

Windows chỉ làm:
    raw JSON -> amplitude/PCHIP -> reorder/bỏ sub -> HTTP tới WSL.

WSL Ubuntu + GPU làm:
    Haar + z-score + Mamba model -> label_id, label, probability.

Mỗi receiver gửi riêng để API model bạn bạn giữ đúng hợp đồng:
    predict(rx1, rx2, rx3) -> label_id, label, probability.
"""

from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

from config import (
    ASUS_DROP_SUBCARRIER_INDICES,
    ASUS_MODEL_SUBCARRIERS,
    ASUS_RAW_SUBCARRIERS,
    ASUS_REORDER_SUBCARRIER_INDICES,
    ESP_DROP_SUBCARRIER_INDICES,
    ESP_MODEL_SUBCARRIERS,
    ESP_SUBCARRIERS,
    FALLBACK_TO_RANDOM_ON_MODEL_ERROR,
    HAR_LABELS,
    MOCK_PROBABILITY_MAX,
    MOCK_PROBABILITY_MIN,
    PRINT_FULL_MODEL_INPUT_TENSORS,
    PRINT_MODEL_INPUT_TENSORS,
    USE_ASUS_WSL_MODEL,
    USE_ESP_WSL_MODEL,
    WINDOW_SIZE,
    WSL_INFERENCE_TIMEOUT_SEC,
    WSL_INFERENCE_URL,
)

# Chỉ dùng khi đang mock/fallback.
_RNG = np.random.default_rng()


# =========================================================
# HÀM DÙNG CHUNG
# =========================================================

def _validate_tensor(
    x: np.ndarray,
    expected_shape: tuple[int, ...],
    device_name: str,
) -> np.ndarray:
    """Ép float32, kiểm tra shape, NaN/Inf và contiguous trước khi gọi model."""
    arr = np.asarray(x, dtype=np.float32)
    if arr.shape != expected_shape:
        raise ValueError(
            f"{device_name} tensor sai shape {arr.shape}; cần {expected_shape}."
        )
    if not np.isfinite(arr).all():
        raise ValueError(
            f"{device_name} tensor còn NaN/Inf; PCHIP phải hoàn tất trước khi predict."
        )
    return np.ascontiguousarray(arr, dtype=np.float32)


def _drop_subcarriers(
    x: np.ndarray,
    *,
    raw_subcarriers: int,
    model_subcarriers: int,
    subcarrier_axis: int,
    drop_indices: tuple[int, ...] | list[int] | None,
    device_name: str,
) -> np.ndarray:
    """Bỏ subcarrier trên một trục và giữ nguyên thứ tự các sub còn lại."""
    if model_subcarriers > raw_subcarriers:
        raise ValueError(
            f"{device_name}: model_subcarriers không thể lớn hơn raw_subcarriers."
        )

    if model_subcarriers == raw_subcarriers:
        if drop_indices not in (None, (), []):
            raise ValueError(
                f"{device_name}: model dùng đủ {raw_subcarriers} subcarrier, "
                "nên ASUS/ESP_DROP_SUBCARRIER_INDICES phải là None."
            )
        return np.ascontiguousarray(x, dtype=np.float32)

    if drop_indices is None:
        raise RuntimeError(
            f"Chưa cấu hình danh sách subcarrier cần bỏ cho {device_name}."
        )

    drop = np.asarray(drop_indices, dtype=np.int64)
    expected_drop_count = raw_subcarriers - model_subcarriers
    if drop.ndim != 1 or drop.size != expected_drop_count:
        raise ValueError(
            f"{device_name} phải bỏ đúng {expected_drop_count} subcarrier; "
            f"nhận {drop.size}."
        )
    if np.unique(drop).size != drop.size:
        raise ValueError(f"{device_name}: danh sách index bỏ bị trùng.")
    if np.any(drop < 0) or np.any(drop >= raw_subcarriers):
        raise ValueError(
            f"{device_name}: index bỏ phải nằm trong 0..{raw_subcarriers - 1}."
        )

    keep_mask = np.ones(raw_subcarriers, dtype=bool)
    keep_mask[drop] = False
    slices: list[Any] = [slice(None)] * x.ndim
    slices[subcarrier_axis] = keep_mask
    return np.ascontiguousarray(x[tuple(slices)], dtype=np.float32)


def _print_model_inputs(source: str, rx1: np.ndarray, rx2: np.ndarray, rx3: np.ndarray) -> None:
    """In tensor dùng cho mock/fallback để kiểm tra shape và giá trị."""
    print(f"\n[MOCK MODEL] {source.upper()} input tensors")
    options = {"threshold": np.inf} if PRINT_FULL_MODEL_INPUT_TENSORS else {}
    with np.printoptions(**options):
        for name, x in (("rx1", rx1), ("rx2", rx2), ("rx3", rx3)):
            print(f"{name}: shape={x.shape}, dtype={x.dtype}")
            print(x)


def _random_prediction(
    source: str,
    rx1: np.ndarray,
    rx2: np.ndarray,
    rx3: np.ndarray,
) -> tuple[int, str, float]:
    """Mock/fallback: in tensor nếu bật rồi trả label ngẫu nhiên và percent 0..100."""
    if PRINT_MODEL_INPUT_TENSORS:
        _print_model_inputs(source, rx1, rx2, rx3)
    else:
        print(
            f"[MOCK MODEL] {source.upper()}: "
            f"rx1={rx1.shape}, rx2={rx2.shape}, rx3={rx3.shape}"
        )

    label_index = int(_RNG.integers(0, len(HAR_LABELS)))
    probability = float(_RNG.uniform(MOCK_PROBABILITY_MIN, MOCK_PROBABILITY_MAX))
    label_id = label_index + 1
    label = HAR_LABELS[label_index]
    percent = probability * 100.0
    print(
        f"[MOCK MODEL] {source.upper()} result: "
        f"label_id={label_id}, label={label}, percent={percent:.2f}%"
    )
    return label_id, label, percent


def _normalize_model_result(result: Any, source: str) -> tuple[int, str, float]:
    """Chuẩn hóa output WSL/model thành (label_id, label, percent 0..100)."""
    if not isinstance(result, (tuple, list)) or len(result) != 3:
        raise RuntimeError(
            f"Model {source} phải trả 3 giá trị: label_id, label, probability."
        )

    label_id, label, probability = result
    label_id = int(label_id)
    probability = float(probability)
    percent = probability * 100.0 if 0.0 <= probability <= 1.0 else probability

    if not label:
        if not 1 <= label_id <= len(HAR_LABELS):
            raise RuntimeError(f"label_id={label_id} ngoài range 1..{len(HAR_LABELS)}.")
        label = HAR_LABELS[label_id - 1]

    return label_id, str(label), percent


def _call_wsl_friend_predict(
    source: str,
    rx1: np.ndarray,
    rx2: np.ndarray,
    rx3: np.ndarray,
) -> tuple[int, str, float]:
    """Gửi ba tensor của một source sang WSL và nhận kết quả model GPU.

    Payload JSON:
        source: "esp" | "asus"
        rx1/rx2/rx3: nested list float32

    WSL service trả:
        {"label_id": 1..8, "label": "walk", "probability": 0..1}
    """
    payload = {
        "source": source,
        "rx1": rx1.tolist(),
        "rx2": rx2.tolist(),
        "rx3": rx3.tolist(),
    }
    request = Request(
        WSL_INFERENCE_URL,
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=WSL_INFERENCE_TIMEOUT_SEC) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"WSL service trả HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(
            "Không kết nối được WSL inference service. "
            "Kiểm tra Ubuntu đang chạy port 8001."
        ) from exc
    except TimeoutError as exc:
        raise RuntimeError(
            f"WSL inference timeout sau {WSL_INFERENCE_TIMEOUT_SEC:.0f}s."
        ) from exc

    if not isinstance(response_data, dict):
        raise RuntimeError("WSL service trả JSON không phải object.")
    try:
        result = (
            response_data["label_id"],
            response_data["label"],
            response_data["probability"],
        )
    except KeyError as exc:
        raise RuntimeError(f"WSL service thiếu key response: {exc}") from exc

    return _normalize_model_result(result, source.upper())


# =========================================================
# ESP: (64,1000) -> (1,56,1000)
# =========================================================

def drop_esp_subcarriers(esp_tensor: np.ndarray) -> np.ndarray:
    """(64,1000) -> (56,1000), bỏ index 28 và 57..63."""
    x = _validate_tensor(esp_tensor, (ESP_SUBCARRIERS, WINDOW_SIZE), "ESP")
    out = _drop_subcarriers(
        x,
        raw_subcarriers=ESP_SUBCARRIERS,
        model_subcarriers=ESP_MODEL_SUBCARRIERS,
        subcarrier_axis=0,
        drop_indices=ESP_DROP_SUBCARRIER_INDICES,
        device_name="ESP",
    )
    expected = (ESP_MODEL_SUBCARRIERS, WINDOW_SIZE)
    if out.shape != expected:
        raise RuntimeError(f"ESP sau khi bỏ sub có shape {out.shape}; cần {expected}.")
    return out


def prepare_esp_for_friend_model(esp_tensor: np.ndarray) -> np.ndarray:
    """(64,1000) -> bỏ sub -> (56,1000) -> thêm A=1 -> (1,56,1000)."""
    x56 = drop_esp_subcarriers(esp_tensor)
    return np.ascontiguousarray(x56[np.newaxis, :, :], dtype=np.float32)


def predict_esp(
    esp1_csi: np.ndarray,
    esp2_csi: np.ndarray,
    esp3_csi: np.ndarray,
) -> tuple[int, str, float]:
    """Predict nhóm ESP. WSL bật thì dùng GPU; tắt/lỗi thì mock random."""
    rx1 = prepare_esp_for_friend_model(esp1_csi)
    rx2 = prepare_esp_for_friend_model(esp2_csi)
    rx3 = prepare_esp_for_friend_model(esp3_csi)

    if USE_ESP_WSL_MODEL:
        try:
            return _call_wsl_friend_predict("esp", rx1, rx2, rx3)
        except Exception as exc:
            if not FALLBACK_TO_RANDOM_ON_MODEL_ERROR:
                raise
            print(f"[ESP WSL MODEL] lỗi: {exc}. Chuyển sang random test.")

    return _random_prediction("esp", rx1, rx2, rx3)


# =========================================================
# ASUS: raw order -> signed reorder -> bỏ sub -> (4,S,1000)
# =========================================================

def reorder_asus_subcarriers(asus_tensor: np.ndarray) -> np.ndarray:
    """Sắp trục raw 0..+31,-32..-1 thành signed -32..-1,0..+31."""
    x = _validate_tensor(
        asus_tensor,
        (4, ASUS_RAW_SUBCARRIERS, WINDOW_SIZE),
        "ASUS",
    )
    reorder = np.asarray(ASUS_REORDER_SUBCARRIER_INDICES, dtype=np.int64)
    if reorder.ndim != 1 or reorder.size != ASUS_RAW_SUBCARRIERS:
        raise ValueError(
            "ASUS_REORDER_SUBCARRIER_INDICES phải có đúng "
            f"{ASUS_RAW_SUBCARRIERS} index."
        )
    if np.unique(reorder).size != ASUS_RAW_SUBCARRIERS:
        raise ValueError("ASUS_REORDER_SUBCARRIER_INDICES có index trùng.")
    if np.any(reorder < 0) or np.any(reorder >= ASUS_RAW_SUBCARRIERS):
        raise ValueError(
            "ASUS_REORDER_SUBCARRIER_INDICES phải nằm trong "
            f"0..{ASUS_RAW_SUBCARRIERS - 1}."
        )
    return np.ascontiguousarray(x[:, reorder, :], dtype=np.float32)


def drop_asus_subcarriers(asus_tensor: np.ndarray) -> np.ndarray:
    """Reorder ASUS trước rồi bỏ sub theo ASUS_DROP_SUBCARRIER_INDICES.

    Drop index được tính trên thứ tự sau reorder.
    Cấu hình hiện tại: (4,64,1000) -> (4,56,1000).
    """
    ordered = reorder_asus_subcarriers(asus_tensor)
    out = _drop_subcarriers(
        ordered,
        raw_subcarriers=ASUS_RAW_SUBCARRIERS,
        model_subcarriers=ASUS_MODEL_SUBCARRIERS,
        subcarrier_axis=1,
        drop_indices=ASUS_DROP_SUBCARRIER_INDICES,
        device_name="ASUS",
    )
    expected = (4, ASUS_MODEL_SUBCARRIERS, WINDOW_SIZE)
    if out.shape != expected:
        raise RuntimeError(f"ASUS sau reorder/bỏ sub có shape {out.shape}; cần {expected}.")
    return out


def prepare_asus_for_friend_model(asus_tensor: np.ndarray) -> np.ndarray:
    """(4,64,1000) -> reorder -> bỏ sub -> (4,56,1000)."""
    return drop_asus_subcarriers(asus_tensor)


def predict_asus(
    asus1_csi: np.ndarray,
    asus2_csi: np.ndarray,
    asus3_csi: np.ndarray,
) -> tuple[int, str, float]:
    """Predict nhóm ASUS. WSL bật thì dùng GPU; tắt/lỗi thì mock random."""
    rx1 = prepare_asus_for_friend_model(asus1_csi)
    rx2 = prepare_asus_for_friend_model(asus2_csi)
    rx3 = prepare_asus_for_friend_model(asus3_csi)

    if USE_ASUS_WSL_MODEL:
        try:
            return _call_wsl_friend_predict("asus", rx1, rx2, rx3)
        except Exception as exc:
            if not FALLBACK_TO_RANDOM_ON_MODEL_ERROR:
                raise
            print(f"[ASUS WSL MODEL] lỗi: {exc}. Chuyển sang random test.")

    return _random_prediction("asus", rx1, rx2, rx3)
