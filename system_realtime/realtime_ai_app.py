"""Realtime HAR Server.

6 deque JSON raw -> chọn 1000 seq -> amplitude + PCHIP -> tensor từng device
-> predict_asus/predict_esp -> SQLite + WebSocket Dashboard.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

from config import (
    ASUS_HOST,
    ASUS_MAC_TO_ID,
    ASUS_PORT,
    DB_PATH,
    ESP_HOST,
    ESP_MAC_TO_ID,
    ESP_PORT,
    GROUP_DEVICES,
    RAW_BUFFER_SIZE,
    RECONNECT_DELAY_SEC,
    SOURCE_TIMEOUT_SEC,
    STATIC_INDEX,
    STEP_SIZE,
    WINDOW_SIZE,
)
from predictor import predict_asus, predict_esp
from preprocess import (
    PreparedGroup,
    WindowNotReady,
    WindowQualityError,
    packet_seq,
    prepare_asus_group,
    prepare_esp_group,
)

if RAW_BUFFER_SIZE < WINDOW_SIZE:
    raise ValueError("RAW_BUFFER_SIZE phải >= WINDOW_SIZE.")

if not 1 <= STEP_SIZE <= WINDOW_SIZE:
    raise ValueError("STEP_SIZE phải trong 1..WINDOW_SIZE.")


# =========================================================
# FASTAPI / RUNTIME STATE
# =========================================================

app = FastAPI(title="Realtime CSI HAR")

# Các Dashboard đang kết nối WebSocket.
ws_clients: set[WebSocket] = set()

# Chỉ cho một coroutine broadcast tại một thời điểm.
broadcast_lock = asyncio.Lock()

# History mới nhất giữ trong RAM.
prediction_history: deque[dict[str, Any]] = deque(maxlen=100)

# Các background task của server.
background_tasks: list[asyncio.Task] = []


# =========================================================
# 6 DEQUE RAW JSON
# =========================================================

class DeviceState:
    """Trạng thái runtime của một receiver vật lý: esp1...asus3."""

    def __init__(self, device_name: str, source: str) -> None:
        self.device_name = device_name
        self.source = source

        # Giữ packet JSON raw gần nhất.
        self.buffer: deque[dict[str, Any]] = deque(maxlen=RAW_BUFFER_SIZE)

        # Dùng để xác định thiết bị online/offline.
        self.last_seen_monotonic = 0.0

        # Thống kê packet.
        self.packet_count_total = 0
        self.packet_count_rate = 0
        self.rate_t0 = time.monotonic()
        self.packet_rate = 0.0

        # Số packet mới kể từ lần prediction gần nhất.
        self.new_since_prediction = 0

    def append(self, packet: dict[str, Any]) -> None:
        """Nhận một packet JSON hợp lệ và thêm vào buffer."""
        packet_seq(packet)

        now = time.monotonic()

        self.buffer.append(packet)
        self.last_seen_monotonic = now

        self.packet_count_total += 1
        self.packet_count_rate += 1
        self.new_since_prediction += 1

        elapsed = now - self.rate_t0

        if elapsed >= 1.0:
            self.packet_rate = round(self.packet_count_rate / elapsed, 2)
            self.packet_count_rate = 0
            self.rate_t0 = now

    def online(self) -> bool:
        """True nếu thiết bị vừa gửi packet trong SOURCE_TIMEOUT_SEC gần nhất."""
        return (
            self.last_seen_monotonic > 0
            and time.monotonic() - self.last_seen_monotonic <= SOURCE_TIMEOUT_SEC
        )

    def ready(self) -> bool:
        """True nếu buffer đã đủ raw packet để tạo window."""
        return len(self.buffer) >= RAW_BUFFER_SIZE


class GroupState:
    """Trạng thái chung cho ba ESP hoặc ba ASUS."""

    def __init__(self, source: str) -> None:
        self.source = source

        # True khi đang preprocess/inference.
        self.pending = False

        # Kết quả prediction mới nhất.
        self.action: str | None = None
        self.confidence: float | None = None

        # Thành công: số mili-giây.
        # Lỗi model/WSL: None -> Dashboard hiển thị --, SQLite lưu NULL.
        self.latency_ms: float | None = None

        self.web_updated_at: str | None = None
        self.input_timestamp_us: int | None = None

        self.start_seq: int | None = None
        self.end_seq: int | None = None

        self.last_error: str | None = None
        self.last_quality: dict[str, Any] = {}

    @property
    def device_names(self) -> tuple[str, str, str]:
        """Thứ tự receiver cố định, phải khớp lúc train model."""
        return GROUP_DEVICES[self.source]

    def ready(self) -> bool:
        return all(devices[name].ready() for name in self.device_names)

    def can_predict(self) -> bool:
        return (
            self.ready()
            and not self.pending
            and min(
                devices[name].new_since_prediction
                for name in self.device_names
            )
            >= STEP_SIZE
        )

    def snapshot(self) -> dict[str, Any]:
        """Trạng thái gửi Dashboard."""
        members = [devices[name] for name in self.device_names]

        online_count = sum(member.online() for member in members)
        ready_count = sum(member.ready() for member in members)

        if online_count == 3:
            status = "online"
        elif online_count > 0:
            status = "partial"
        else:
            status = "waiting"

        return {
            "status": status,
            "action": self.action,
            "confidence": self.confidence,
            "latency_ms": self.latency_ms,
            "web_updated_at": self.web_updated_at,
            "input_timestamp_us": self.input_timestamp_us,
            "input_timestamp_text": format_timestamp_us(
                self.input_timestamp_us
            ),
            "online_devices": online_count,
            "total_devices": 3,
            "ready_devices": ready_count,
            "window_size": WINDOW_SIZE,
            "raw_buffer_size": RAW_BUFFER_SIZE,
            "step_size": STEP_SIZE,
            "prediction_pending": self.pending,
            "start_seq": self.start_seq,
            "end_seq": self.end_seq,
            "last_error": self.last_error,
            "quality": self.last_quality,
        }


devices = {
    "esp1": DeviceState("esp1", "esp"),
    "esp2": DeviceState("esp2", "esp"),
    "esp3": DeviceState("esp3", "esp"),
    "asus1": DeviceState("asus1", "asus"),
    "asus2": DeviceState("asus2", "asus"),
    "asus3": DeviceState("asus3", "asus"),
}

groups = {
    "esp": GroupState("esp"),
    "asus": GroupState("asus"),
}


# =========================================================
# SQLITE
# =========================================================

def init_db() -> None:
    """Tạo bảng SQLite nếu chưa tồn tại."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                web_updated_at TEXT NOT NULL,
                source TEXT NOT NULL,
                input_timestamp TEXT,
                latency_ms REAL,
                action TEXT,
                confidence REAL
            )
            """
        )


def insert_log(row: dict[str, Any]) -> None:
    """Ghi một prediction vào SQLite."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO prediction_logs
            (
                web_updated_at,
                source,
                input_timestamp,
                latency_ms,
                action,
                confidence
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                row["web_updated_at"],
                row["source"],
                row.get("input_timestamp"),
                row.get("latency_ms"),
                row.get("action"),
                row.get("confidence"),
            ),
        )


def read_logs(limit: int) -> list[dict[str, Any]]:
    """Đọc lịch sử prediction từ SQLite."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row

        rows = conn.execute(
            """
            SELECT
                id,
                web_updated_at,
                source,
                input_timestamp,
                latency_ms,
                action,
                confidence
            FROM prediction_logs
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    result = [dict(row) for row in rows]
    result.reverse()

    for row in result:
        row["input_timestamp_text"] = row.get("input_timestamp")

    return result


# =========================================================
# WEB / WEBSOCKET
# =========================================================

def format_timestamp_us(timestamp_us: int | None) -> str | None:
    """Đổi Unix microsecond thành chuỗi giờ dễ đọc."""
    if timestamp_us is None:
        return None

    try:
        return datetime.fromtimestamp(
            int(timestamp_us) / 1_000_000
        ).strftime("%H:%M:%S %d/%m/%Y")

    except (TypeError, ValueError, OSError):
        return None


def build_snapshot() -> dict[str, Any]:
    """Tạo JSON trạng thái đầy đủ gửi Dashboard."""
    return {
        "type": "snapshot",
        "server_time": datetime.now().strftime("%H:%M:%S %d/%m/%Y"),
        "esp": groups["esp"].snapshot(),
        "asus": groups["asus"].snapshot(),
        "history": list(prediction_history),
    }


async def broadcast(message: dict[str, Any]) -> None:
    """Gửi JSON đến toàn bộ browser Dashboard."""
    dead: list[WebSocket] = []

    async with broadcast_lock:
        for ws in list(ws_clients):
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)

        for ws in dead:
            ws_clients.discard(ws)


@app.get("/", response_class=HTMLResponse)
async def dashboard() -> str:
    return STATIC_INDEX.read_text(encoding="utf-8")


@app.get("/api/status")
async def api_status() -> JSONResponse:
    return JSONResponse(build_snapshot())


@app.get("/api/history")
async def api_history(
    limit: int = Query(50, ge=1, le=500)
) -> JSONResponse:
    return JSONResponse(
        {
            "type": "history",
            "history": await asyncio.to_thread(read_logs, limit),
        }
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Kênh WebSocket giữa server và Dashboard."""
    await websocket.accept()

    ws_clients.add(websocket)

    await websocket.send_json(build_snapshot())

    try:
        while True:
            await websocket.receive_text()

    except WebSocketDisconnect:
        pass

    finally:
        ws_clients.discard(websocket)


# =========================================================
# PREPROCESS + MODEL
# =========================================================

def _quality_to_json(prepared: PreparedGroup) -> dict[str, Any]:
    """Đổi thông tin quality sang dict JSON."""
    return {
        name: {
            "missing_count": item.missing_count,
            "missing_ratio": item.missing_ratio,
            "max_consecutive_missing": item.max_consecutive_missing,
        }
        for name, item in prepared.quality.items()
    }


def _predict_worker(
    source: str,
    buffers: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Chạy trong worker thread: raw JSON -> preprocess -> model."""

    if source == "asus":
        prepared = prepare_asus_group(
            buffers["asus1"],
            buffers["asus2"],
            buffers["asus3"],
        )

        label_id, label, percent = predict_asus(
            *prepared.as_tuple(("asus1", "asus2", "asus3"))
        )

    elif source == "esp":
        prepared = prepare_esp_group(
            buffers["esp1"],
            buffers["esp2"],
            buffers["esp3"],
        )

        label_id, label, percent = predict_esp(
            *prepared.as_tuple(("esp1", "esp2", "esp3"))
        )

    else:
        raise ValueError(f"source không hợp lệ: {source}")

    return {
        "label": label,
        "label_id": label_id,
        "percent": percent,
        "confidence": percent / 100.0,
        "input_timestamp_us": prepared.input_timestamp_us,
        "start_seq": prepared.start_seq,
        "end_seq": prepared.end_seq,
        "quality": _quality_to_json(prepared),
    }


async def run_group_prediction(
    source: str,
    buffers: dict[str, list[dict[str, Any]]],
) -> None:
    """Chạy một prediction và cập nhật Dashboard/SQLite.

    Thành công:
        latency_ms = preprocessing + gửi WSL + model GPU + nhận kết quả.

    Lỗi model/WSL:
        latency_ms = None -> SQLite lưu NULL.
    """

    group = groups[source]

    # Timer chỉ được dùng khi prediction thành công.
    t0 = time.perf_counter()

    input_timestamp_us: int | None = None
    latency_ms: float | None = None

    try:
        result = await asyncio.to_thread(
            _predict_worker,
            source,
            buffers,
        )

        label = result["label"]
        confidence = result["confidence"]

        input_timestamp_us = result["input_timestamp_us"]

        group.start_seq = result["start_seq"]
        group.end_seq = result["end_seq"]

        group.last_quality = result["quality"]
        group.last_error = None

        # Chỉ có kết quả model thành công mới tính latency.
        latency_ms = round(
            (time.perf_counter() - t0) * 1000.0,
            2,
        )

    except (WindowNotReady, WindowQualityError) as exc:
        # Window chưa đủ/chất lượng không tốt: không log prediction.
        group.last_error = str(exc)
        group.pending = False

        print(f"[PREPROCESS] {source}: {exc}")

        await broadcast(build_snapshot())
        return

    except FileNotFoundError as exc:
        # Không có model: không có latency inference.
        label = "no_model"
        confidence = None
        latency_ms = None

        group.last_error = str(exc)

        print(f"[AI] {source}: {exc}")

    except Exception as exc:
        # Lỗi WSL, timeout HTTP, lỗi Mamba, lỗi shape...
        # Không xem thời gian lỗi là inference latency.
        label = "predict_error"
        confidence = None
        latency_ms = None

        group.last_error = str(exc)

        print(f"[AI] {source} error: {exc}")

    web_updated_at = datetime.now().strftime("%H:%M:%S %d/%m/%Y")

    group.action = label
    group.confidence = confidence
    group.latency_ms = latency_ms

    group.web_updated_at = web_updated_at
    group.input_timestamp_us = input_timestamp_us

    group.pending = False

    row = {
        "source": source,
        "action": label,
        "confidence": confidence,

        # Thành công: float ms.
        # Lỗi: None -> SQLite tự lưu NULL.
        "latency_ms": latency_ms,

        "web_updated_at": web_updated_at,

        # SQLite chỉ lưu timestamp dạng dễ đọc.
        "input_timestamp": format_timestamp_us(input_timestamp_us),
        "input_timestamp_text": format_timestamp_us(input_timestamp_us),
    }

    prediction_history.append(row)

    await asyncio.to_thread(insert_log, row)

    await broadcast(
        {
            "type": "prediction",
            "prediction": row,
            "snapshot": build_snapshot(),
        }
    )


async def accept_packet(source: str, packet: dict[str, Any]) -> None:
    """Nhận JSON từ TCP Collection, map MAC -> device ID, rồi trigger prediction."""

    mac = str(packet.get("device_id", "")).upper()

    device_name = (
        ESP_MAC_TO_ID if source == "esp" else ASUS_MAC_TO_ID
    ).get(mac)

    if device_name is None:
        print(f"[TCP] bỏ packet {source}: MAC chưa khai báo: {mac}")
        return

    try:
        devices[device_name].append(packet)

    except ValueError as exc:
        print(f"[TCP] bỏ packet {device_name}: {exc}")
        return

    group = groups[source]

    if group.can_predict():
        # Snapshot raw buffer để worker thread xử lý ổn định.
        buffers = {
            name: list(devices[name].buffer)
            for name in group.device_names
        }

        group.pending = True

        for name in group.device_names:
            devices[name].new_since_prediction = 0

        asyncio.create_task(
            run_group_prediction(source, buffers)
        )


# =========================================================
# TCP JSON LINES CLIENT
# =========================================================

async def tcp_receiver(source: str, host: str, port: int) -> None:
    """TCP client nhận JSON Lines từ Collection ESP hoặc ASUS."""

    while True:
        try:
            print(f"[TCP] {source}: connecting {host}:{port}")

            reader, writer = await asyncio.open_connection(host, port)

            print(f"[TCP] {source}: connected")

            try:
                while True:
                    line = await reader.readline()

                    if not line:
                        raise ConnectionError("collector closed connection")

                    try:
                        packet = json.loads(
                            line.decode("utf-8").strip()
                        )

                    except (
                        UnicodeDecodeError,
                        json.JSONDecodeError,
                    ) as exc:
                        print(f"[TCP] {source}: JSON lỗi: {exc}")
                        continue

                    if not isinstance(packet, dict):
                        continue

                    if source == "esp" and packet.get("type") != "csi_data":
                        continue

                    await accept_packet(source, packet)

            finally:
                writer.close()
                await writer.wait_closed()

        except asyncio.CancelledError:
            raise

        except Exception as exc:
            print(
                f"[TCP] {source}: {exc}; "
                f"retry sau {RECONNECT_DELAY_SEC}s"
            )

            await asyncio.sleep(RECONNECT_DELAY_SEC)


async def status_broadcast_loop() -> None:
    """Mỗi giây broadcast trạng thái mới nhất lên Dashboard."""

    while True:
        await asyncio.sleep(1.0)

        await broadcast(build_snapshot())


# =========================================================
# STARTUP / SHUTDOWN
# =========================================================

@app.on_event("startup")
async def startup() -> None:
    """Khởi tạo DB, history và TCP/background task."""

    init_db()

    prediction_history.extend(
        await asyncio.to_thread(read_logs, 100)
    )

    background_tasks.extend(
        [
            asyncio.create_task(
                tcp_receiver("esp", ESP_HOST, ESP_PORT)
            ),
            asyncio.create_task(
                tcp_receiver("asus", ASUS_HOST, ASUS_PORT)
            ),
            asyncio.create_task(status_broadcast_loop()),
        ]
    )


@app.on_event("shutdown")
async def shutdown() -> None:
    """Dừng gọn các background task."""

    for task in background_tasks:
        task.cancel()

    await asyncio.gather(
        *background_tasks,
        return_exceptions=True,
    )