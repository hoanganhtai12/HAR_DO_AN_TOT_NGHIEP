"""FastAPI bridge: Windows gửi tensor đã preprocess, WSL chạy Mamba trên GPU.

Windows gửi POST /predict:
    source: "esp" | "asus"
    rx1, rx2, rx3: mỗi tensor float32 đã có 56 subcarrier.

WSL chỉ chạy Haar DWT + z-score + Mamba qua package friend_model.
"""
from __future__ import annotations

import threading
import traceback
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from friend_model.predict import predict_asus, predict_esp

# Input mỗi receiver, không có batch dimension.
EXPECTED_SHAPES: dict[str, tuple[int, int, int]] = {
    "esp": (1, 56, 1000),
    "asus": (4, 56, 1000),
}

FRIEND_MODEL_DIR = Path(__file__).resolve().parent / "friend_model"
BUNDLE_DIR = FRIEND_MODEL_DIR / "bundle"

app = FastAPI(title="WSL GPU CSI Inference")
predict_lock = threading.Lock()


class PredictRequest(BaseModel):
    """JSON body mà Windows predictor.py gửi sang WSL."""
    source: Literal["esp", "asus"]
    rx1: list[Any]
    rx2: list[Any]
    rx3: list[Any]


def _bundle_exists(source: str) -> bool:
    """Báo trạng thái esp.pt/asus.pt cho endpoint /health."""
    return (BUNDLE_DIR / f"{source}.pt").is_file()


def _validate_tensor(
    name: str,
    value: list[Any],
    expected_shape: tuple[int, int, int],
) -> np.ndarray:
    """JSON nested-list -> float32 contiguous, kiểm tra đúng contract model."""
    x = np.asarray(value, dtype=np.float32)
    if x.shape != expected_shape:
        raise ValueError(f"{name} sai shape {tuple(x.shape)}; cần {expected_shape}.")
    if not np.isfinite(x).all():
        raise ValueError(f"{name} có NaN/Inf; PCHIP phía Windows chưa hoàn tất.")
    return np.ascontiguousarray(x, dtype=np.float32)


@app.get("/health")
def health() -> dict[str, Any]:
    """Windows dùng để kiểm tra WSL, GPU và hai checkpoint."""
    return {
        "ok": True,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "expected_shape_per_receiver": EXPECTED_SHAPES,
        "bundle_available": {
            "esp": _bundle_exists("esp"),
            "asus": _bundle_exists("asus"),
        },
    }


@app.post("/predict")
def predict_api(request: PredictRequest) -> dict[str, Any]:
    """Gọi đúng predict_esp hoặc predict_asus theo source, không auto-detect mơ hồ."""
    try:
        expected_shape = EXPECTED_SHAPES[request.source]
        rx1 = _validate_tensor("rx1", request.rx1, expected_shape)
        rx2 = _validate_tensor("rx2", request.rx2, expected_shape)
        rx3 = _validate_tensor("rx3", request.rx3, expected_shape)

        print(
            f"[WSL] {request.source}: "
            f"rx1={rx1.shape}, rx2={rx2.shape}, rx3={rx3.shape}"
        )

        with predict_lock:
            if request.source == "esp":
                label_id, label, probability = predict_esp(rx1, rx2, rx3)
            else:
                label_id, label, probability = predict_asus(rx1, rx2, rx3)

        response = {
            "label_id": int(label_id),
            "label": str(label),
            "probability": float(probability),
        }
        print(f"[WSL] {request.source} result: {response}")
        return response

    except Exception as exc:
        print("\n[WSL] LỖI INFERENCE:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc
