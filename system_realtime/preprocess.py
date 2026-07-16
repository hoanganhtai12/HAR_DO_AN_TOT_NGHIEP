"""Tiền xử lý CSI theo seq: JSON raw -> amplitude -> PCHIP -> tensor từng device.

Thiết kế:
- Mỗi device giữ deque 1200 JSON raw trong app.
- Lấy seq CUỐI của từng buffer (packet mới nhất đang giữ).
- Chọn seq cuối CŨ NHẤT trong ba device làm end_seq chung,
  tức là mốc mới nhất mà ba device đều có cơ hội đã đi tới.
- Từ end_seq lùi 999 mốc để tạo window gồm đúng 1000 seq.
- Packet có thật: chuyển sang amplitude.
- Packet mất/lỗi: để NaN.
- PCHIP nội suy amplitude theo chiều thời gian.
- Chưa ghép ba device ở file này.
"""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.interpolate import PchipInterpolator

from config import (
    ASUS_ANTENNAS,
    ASUS_AUTOSCALE,
    ASUS_FMT,
    ASUS_NBITS,
    ASUS_NEXP,
    ASUS_NFFT,
    ASUS_NMAN,
    ASUS_SHFT,
    ASUS_SUBCARRIERS,
    ESP_SUBCARRIERS,
    INTERPOLATION_METHOD,
    MAX_CONSECUTIVE_MISSING,
    MAX_MISSING_RATIO,
    RAW_BUFFER_SIZE,
    SEQ_HALF_RANGE,
    SEQ_MODULO,
    UNPACK_DLL_PATH,
    WINDOW_SIZE,
)


class WindowNotReady(RuntimeError):
    """Chưa đủ JSON raw để tạo window 1000 seq."""


class WindowQualityError(RuntimeError):
    """Window có packet mất/lỗi vượt ngưỡng, không nên suy luận."""


@dataclass(frozen=True)
class WindowPlan:
    """Kế hoạch chọn một window seq chung cho ba thiết bị cùng source.

    start_seq/end_seq: mốc raw seq 12-bit của cửa sổ 1000 packet.
    target_seqs      : danh sách 1000 seq theo đúng thứ tự thời gian logic.
    packets_by_device: map ``seq -> packet JSON`` để tra nhanh packet tại từng mốc.
    """

    # Seq đầu = end_seq - (WINDOW_SIZE - 1), tính theo modulo 4096.
    start_seq: int
    # Seq cuối chung: seq cuối cũ nhất trong ba raw buffer.
    end_seq: int
    # 1000 seq liên tiếp, dùng làm trục time T của tensor.
    target_seqs: tuple[int, ...]
    # Dữ liệu raw từng thiết bị sau khi map theo seq.
    packets_by_device: dict[str, dict[int, Mapping[str, Any]]]


@dataclass(frozen=True)
class DeviceQuality:
    """Chất lượng một device trong window vừa tạo.

    Các giá trị được gửi về Dashboard/SQLite để biết window có bị mất packet không.
    """

    # Tổng số seq không có packet thật hoặc packet sai format.
    missing_count: int
    # missing_count / WINDOW_SIZE.
    missing_ratio: float
    # Số packet mất liên tiếp dài nhất.
    max_consecutive_missing: int


@dataclass(frozen=True)
class PreparedGroup:
    """Tensor độc lập từng device, chưa stack theo trục receiver R."""

    tensors: dict[str, np.ndarray]
    start_seq: int
    end_seq: int
    input_timestamp_us: int | None
    quality: dict[str, DeviceQuality]

    def as_tuple(self, device_names: tuple[str, str, str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Trả ba tensor theo thứ tự cố định mà model yêu cầu.

        Ví dụ ASUS phải luôn gọi ``asus1, asus2, asus3``;
        không được đảo thứ tự receiver so với lúc train model.
        """
        return tuple(self.tensors[name] for name in device_names)  # type: ignore[return-value]


# =========================================================
# SEQ 12-BIT: SO SÁNH MODULO 4096
# =========================================================

def packet_seq(packet: Mapping[str, Any]) -> int:
    """Đọc và kiểm tra seq raw 0..4095 từ JSON packet."""
    try:
        seq = int(packet["seq"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Packet thiếu hoặc sai trường `seq`.") from exc
    if not 0 <= seq < SEQ_MODULO:
        raise ValueError(f"seq={seq} ngoài khoảng 0..{SEQ_MODULO - 1}.")
    return seq


def seq_is_newer(candidate: int, reference: int) -> bool:
    """True nếu candidate mới hơn reference trong chu kỳ seq modulo 4096.

    Chỉ dùng khi hai seq cần so sánh cách nhau nhỏ hơn 2048 packet,
    điều kiện đúng với buffer 1200 packet của hệ thống này.
    """
    delta = (candidate - reference) % SEQ_MODULO
    return 0 < delta < SEQ_HALF_RANGE


def newest_seq(seqs: Iterable[int]) -> int:
    """Lấy seq mới nhất theo thứ tự modulo, không dùng max() trực tiếp."""
    iterator = iter(seqs)
    try:
        newest = next(iterator)
    except StopIteration as exc:
        raise ValueError("Không có seq để so sánh.") from exc

    for seq in iterator:
        if seq_is_newer(seq, newest):
            newest = seq
    return newest


def oldest_seq(seqs: Iterable[int]) -> int:
    """Lấy seq cũ nhất theo thứ tự modulo, không dùng min() trực tiếp.

    Ví dụ khi seq quay vòng 4095 -> 0:
        4080, 4090, 15
    thì 4080 mới là mốc cũ nhất theo thời gian logic.

    Điều kiện: các seq so sánh cách nhau nhỏ hơn SEQ_HALF_RANGE (= 2048),
    phù hợp với RAW_BUFFER_SIZE = 1200 trong hệ thống này.
    """
    iterator = iter(seqs)
    try:
        oldest = next(iterator)
    except StopIteration as exc:
        raise ValueError("Không có seq để so sánh.") from exc

    for seq in iterator:
        # Nếu oldest hiện tại lại mới hơn seq, seq phải là mốc cũ hơn.
        if seq_is_newer(oldest, seq):
            oldest = seq
    return oldest


def has_reached(last_seq: int, required_seq: int) -> bool:
    """Kiểm tra device đã nhận tới required_seq hay chưa theo modulo."""
    return last_seq == required_seq or seq_is_newer(last_seq, required_seq)


# =========================================================
# CHỌN 1000 MỐC SEQ: KẾT THÚC Ở DEVICE CHẬM NHẤT
# =========================================================

def _as_packet_list(buffer: Sequence[Mapping[str, Any]], device_name: str) -> list[Mapping[str, Any]]:
    """Sao chép deque thành list ổn định và kiểm tra buffer đã đủ 1200 raw packet."""
    packets = list(buffer)
    if len(packets) < RAW_BUFFER_SIZE:
        raise WindowNotReady(
            f"{device_name} mới có {len(packets)}/{RAW_BUFFER_SIZE} packet raw."
        )

    # append ở app đã kiểm tra seq, nhưng kiểm tra lại giúp hàm dùng được độc lập.
    for packet in packets:
        packet_seq(packet)
    return packets


def select_window_plan(
    buffers: Mapping[str, Sequence[Mapping[str, Any]]],
) -> WindowPlan:
    """Chọn window 1000 seq bám theo dữ liệu mới nhất chung của ba device.

    Quy tắc chọn window:
    1. Lấy seq cuối của mỗi buffer, tức packet mới nhất đang giữ của device đó.
    2. Chọn seq cuối CŨ NHẤT theo modulo 4096 làm ``end_seq`` chung.
       Đây là mốc mới nhất mà không vượt quá device chậm nhất.
    3. Lùi 999 seq từ ``end_seq`` để có ``start_seq``. Như vậy window có
       đúng 1000 mốc từ start_seq -> end_seq.

    Ví dụ không quay vòng:
        esp1: 1000 ... 2199
        esp2: 1030 ... 2229
        esp3: 1020 ... 2219

        seq cuối cũ nhất = 2199
        end_seq   = 2199
        start_seq = 1200
        target    = 1200 ... 2199

    Không dùng ``min(last_seqs)`` vì seq là 12-bit và có thể quay vòng
    4095 -> 0. Packet thiếu bên trong window vẫn được xử lý ở bước sau
    bằng NaN + PCHIP; nhưng window không được thiếu dữ liệu ở hai biên.
    """
    if len(buffers) != 3:
        raise ValueError("Mỗi group cần đúng 3 buffer device.")

    packet_lists = {
        name: _as_packet_list(buffer, name)
        for name, buffer in buffers.items()
    }

    # Packet cuối deque là packet mới nhất đang giữ của từng device.
    last_seqs = {
        name: packet_seq(packets[-1])
        for name, packets in packet_lists.items()
    }

    # Device chậm nhất quyết định mốc kết thúc chung.
    end_seq = oldest_seq(last_seqs.values())

    # Window có WINDOW_SIZE mốc, gồm cả start_seq và end_seq.
    start_seq = (end_seq - WINDOW_SIZE + 1) % SEQ_MODULO

    for name, packets in packet_lists.items():
        first_seq = packet_seq(packets[0])
        last_seq = packet_seq(packets[-1])

        # Dù end_seq được chọn từ ba last_seq, kiểm tra lại để bảo vệ
        # trường hợp dữ liệu vào không theo đúng thứ tự tăng seq.
        if not has_reached(last_seq, end_seq):
            raise WindowNotReady(
                f"{name} mới đến seq={last_seq}; chưa đến end_seq={end_seq}."
            )

        # start_seq phải còn nằm trong phạm vi lịch sử còn giữ ở deque.
        # has_reached(start_seq, first_seq) == True nghĩa là start_seq
        # bằng hoặc mới hơn mốc đầu buffer. Nếu False thì device này đã
        # bỏ mất phần đầu window vì nó chạy nhanh hơn các device còn lại.
        if not has_reached(start_seq, first_seq):
            raise WindowNotReady(
                f"{name} chỉ còn từ seq={first_seq}; "
                f"không đủ lịch sử để lấy window {start_seq} -> {end_seq}."
            )

    # Trục time của model luôn đi xuôi theo thời gian logic, không đảo ngược:
    # start_seq, start_seq+1, ..., end_seq (qua 4095 -> 0 khi cần).
    target_seqs = tuple(
        (start_seq + offset) % SEQ_MODULO
        for offset in range(WINDOW_SIZE)
    )

    # Buffer < SEQ_MODULO nên raw seq không lặp qua hai vòng trong cùng deque.
    # Nếu trùng seq do duplicate packet, giữ packet đến sau cùng.
    maps: dict[str, dict[int, Mapping[str, Any]]] = {}
    for name, packets in packet_lists.items():
        by_seq: dict[int, Mapping[str, Any]] = {}
        for packet in packets:
            by_seq[packet_seq(packet)] = packet
        maps[name] = by_seq

    return WindowPlan(
        start_seq=start_seq,
        end_seq=end_seq,
        target_seqs=target_seqs,
        packets_by_device=maps,
    )


# =========================================================
# ASUS: uint32 raw -> I/Q -> AMPLITUDE (4, 64) / PACKET
# =========================================================

class NexmonCUnpacker:
    """Wrapper cho unpack_float_acphy trong unpack.dll.

    Input : uint32[64] của một antenna.
    Output: int32[64, 2], mỗi hàng [I, Q].
    """

    def __init__(self, dll_path=UNPACK_DLL_PATH):
        if not dll_path.exists():
            raise FileNotFoundError(f"Không thấy DLL: {dll_path}")
        if os.name != "nt":
            raise OSError(
                "unpack.dll là DLL Windows. Hãy chạy ASUS pipeline trên Windows x64 "
                "hoặc biên dịch unpack.c thành .so cho Linux."
            )

        self._dll = ctypes.WinDLL(str(dll_path))
        self._func = self._dll.unpack_float_acphy
        self._func.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_int32),
        ]
        self._func.restype = None

    def unpack_iq(self, raw_uint32: Iterable[int]) -> np.ndarray:
        raw = np.ascontiguousarray(np.asarray(list(raw_uint32), dtype=np.uint32))
        if raw.shape != (ASUS_NFFT,):
            raise ValueError(
                f"Một antenna ASUS cần {ASUS_NFFT} uint32, nhận shape={raw.shape}."
            )
        output = np.empty(ASUS_NFFT * 2, dtype=np.int32)
        self._func(
            ASUS_NBITS,
            ASUS_AUTOSCALE,
            ASUS_SHFT,
            ASUS_FMT,
            ASUS_NMAN,
            ASUS_NEXP,
            ASUS_NFFT,
            raw.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        )
        return output.reshape(ASUS_NFFT, 2)


_unpacker: NexmonCUnpacker | None = None


def get_unpacker() -> NexmonCUnpacker:
    global _unpacker
    if _unpacker is None:
        _unpacker = NexmonCUnpacker()
    return _unpacker


def asus_packet_to_amplitude(packet: Mapping[str, Any]) -> np.ndarray:
    """JSON ASUS một packet -> amplitude float32 (4, 64)."""
    csi = packet.get("csi")
    if not isinstance(csi, Mapping):
        raise ValueError("Packet ASUS thiếu object csi.")

    frame = np.empty((len(ASUS_ANTENNAS), ASUS_SUBCARRIERS), dtype=np.float32)
    unpacker = get_unpacker()
    for antenna_index, antenna_name in enumerate(ASUS_ANTENNAS):
        raw = csi.get(antenna_name)
        if not isinstance(raw, list):
            raise ValueError(f"Packet ASUS thiếu csi.{antenna_name}.")
        iq = unpacker.unpack_iq(raw).astype(np.float32, copy=False)
        # unpack.c trả [I, Q]; model dùng amplitude = sqrt(I^2 + Q^2).
        frame[antenna_index, :] = np.hypot(iq[:, 0], iq[:, 1])
    return frame


# =========================================================
# ESP: [Q, I] -> AMPLITUDE (64,) / PACKET
# =========================================================

def esp_packet_to_amplitude(packet: Mapping[str, Any]) -> np.ndarray:
    """JSON ESP một packet -> amplitude float32 (64,)."""
    csi = np.asarray(packet.get("csi"), dtype=np.float32)
    if csi.shape != (ESP_SUBCARRIERS, 2):
        raise ValueError(
            f"CSI ESP cần shape ({ESP_SUBCARRIERS}, 2) theo [Q, I], nhận {csi.shape}."
        )
    return np.hypot(csi[:, 0], csi[:, 1]).astype(np.float32, copy=False)


# =========================================================
# PCHIP AMPLITUDE + KIỂM TRA MẤT GÓI
# =========================================================

def _max_consecutive_true(mask: np.ndarray) -> int:
    """Tìm độ dài đoạn True liên tiếp dài nhất trong mask packet mất."""
    max_run = run = 0
    for value in mask.tolist():
        if value:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    return max_run


def check_missing_quality(missing_mask: np.ndarray, device_name: str) -> DeviceQuality:
    """Chấp nhận/từ chối window trước khi PCHIP.

    PCHIP chỉ nội suy tốt ở giữa chuỗi; không nội suy ngoại suy ở đầu/cuối.
    Vì vậy window bị thiếu ở đầu hoặc cuối sẽ bị loại thẳng.
    """
    missing_count = int(missing_mask.sum())
    missing_ratio = missing_count / float(WINDOW_SIZE)
    max_gap = _max_consecutive_true(missing_mask)

    if missing_mask[0] or missing_mask[-1]:
        raise WindowQualityError(
            f"{device_name}: mất packet tại đầu/cuối window, không PCHIP ngoại suy."
        )
    if missing_ratio > MAX_MISSING_RATIO:
        raise WindowQualityError(
            f"{device_name}: mất {missing_count}/{WINDOW_SIZE} ({missing_ratio:.2%}), "
            f"vượt ngưỡng {MAX_MISSING_RATIO:.2%}."
        )
    if max_gap > MAX_CONSECUTIVE_MISSING:
        raise WindowQualityError(
            f"{device_name}: mất liên tiếp {max_gap} packet, "
            f"vượt ngưỡng {MAX_CONSECUTIVE_MISSING}."
        )
    return DeviceQuality(missing_count, missing_ratio, max_gap)


def pchip_interpolate_amplitude(tensor: np.ndarray, missing_mask: np.ndarray) -> np.ndarray:
    """Nội suy PCHIP amplitude theo trục thời gian cuối.

    tensor ASUS: (4, 64, T)
    tensor ESP : (64, T)
    missing_mask: (T,), True tại packet mất/lỗi.
    """
    if INTERPOLATION_METHOD != "pchip":
        raise ValueError(f"Không hỗ trợ nội suy: {INTERPOLATION_METHOD}")

    result = np.asarray(tensor, dtype=np.float32).copy()
    if not missing_mask.any():
        return result

    valid_idx = np.flatnonzero(~missing_mask)
    missing_idx = np.flatnonzero(missing_mask)
    if valid_idx.size < 2:
        raise WindowQualityError("Không đủ packet thật để nội suy PCHIP.")

    interpolator = PchipInterpolator(
        valid_idx,
        result[..., valid_idx],
        axis=-1,
        extrapolate=False,
    )
    result[..., missing_idx] = interpolator(missing_idx).astype(np.float32, copy=False)

    if not np.isfinite(result).all():
        raise WindowQualityError("PCHIP xong vẫn còn NaN/Inf trong tensor.")
    return result


# =========================================================
# TẠO TENSOR RIÊNG TỪ WINDOW PLAN
# =========================================================

def _build_asus_tensor(
    packets_by_seq: Mapping[int, Mapping[str, Any]],
    target_seqs: Sequence[int],
    device_name: str,
) -> tuple[np.ndarray, DeviceQuality]:
    """Tạo tensor một ASUS theo đúng 1000 mốc seq đã chọn.

    Packet thật  -> unpack DLL -> I/Q -> amplitude (4,64).
    Packet mất   -> cột time tương ứng giữ NaN, sau đó PCHIP.
    Output cuối  -> (4,64,1000), không còn NaN nếu qua kiểm tra chất lượng.
    """
    tensor = np.full(
        (len(ASUS_ANTENNAS), ASUS_SUBCARRIERS, WINDOW_SIZE),
        np.nan,
        dtype=np.float32,
    )
    missing = np.zeros(WINDOW_SIZE, dtype=bool)

    for time_index, seq in enumerate(target_seqs):
        packet = packets_by_seq.get(seq)
        if packet is None:
            missing[time_index] = True
            continue
        try:
            tensor[:, :, time_index] = asus_packet_to_amplitude(packet)
        except ValueError:
            # Packet malformed được coi như packet mất để PCHIP nếu còn trong ngưỡng.
            missing[time_index] = True

    quality = check_missing_quality(missing, device_name)
    return pchip_interpolate_amplitude(tensor, missing), quality


def _build_esp_tensor(
    packets_by_seq: Mapping[int, Mapping[str, Any]],
    target_seqs: Sequence[int],
    device_name: str,
) -> tuple[np.ndarray, DeviceQuality]:
    """Tạo tensor một ESP theo đúng 1000 mốc seq đã chọn.

    Packet thật  -> [Q,I] -> amplitude (64,).
    Packet mất   -> cột time tương ứng giữ NaN, sau đó PCHIP.
    Output cuối  -> (64,1000), không còn NaN nếu qua kiểm tra chất lượng.
    """
    tensor = np.full((ESP_SUBCARRIERS, WINDOW_SIZE), np.nan, dtype=np.float32)
    missing = np.zeros(WINDOW_SIZE, dtype=bool)

    for time_index, seq in enumerate(target_seqs):
        packet = packets_by_seq.get(seq)
        if packet is None:
            missing[time_index] = True
            continue
        try:
            tensor[:, time_index] = esp_packet_to_amplitude(packet)
        except ValueError:
            missing[time_index] = True

    quality = check_missing_quality(missing, device_name)
    return pchip_interpolate_amplitude(tensor, missing), quality


def _input_timestamp_us(plan: WindowPlan, device_names: tuple[str, str, str]) -> int | None:
    """Mốc input: min timestamp của packet thật cuối cùng trong ba device."""
    timestamps: list[int] = []
    for name in device_names:
        by_seq = plan.packets_by_device[name]
        for seq in reversed(plan.target_seqs):
            packet = by_seq.get(seq)
            if packet is None:
                continue
            try:
                timestamps.append(int(packet["timestamp"]))
                break
            except (KeyError, TypeError, ValueError):
                continue
    return min(timestamps) if timestamps else None


def prepare_asus_group(
    asus1_buffer: Sequence[Mapping[str, Any]],
    asus2_buffer: Sequence[Mapping[str, Any]],
    asus3_buffer: Sequence[Mapping[str, Any]],
) -> PreparedGroup:
    """Tiền xử lý cả nhóm ASUS.

    1. Chọn 1000 seq chung.
    2. Tạo tensor riêng cho asus1/asus2/asus3.
    3. Ghi metadata chất lượng và input timestamp.

    Tensor ở đây vẫn (4,64,1000); reorder/bỏ sub nằm trong predictor.py.
    """
    names = ("asus1", "asus2", "asus3")
    plan = select_window_plan(dict(zip(names, (asus1_buffer, asus2_buffer, asus3_buffer))))
    tensors: dict[str, np.ndarray] = {}
    quality: dict[str, DeviceQuality] = {}
    for name in names:
        tensor, report = _build_asus_tensor(plan.packets_by_device[name], plan.target_seqs, name)
        tensors[name] = tensor
        quality[name] = report
    return PreparedGroup(tensors, plan.start_seq, plan.end_seq, _input_timestamp_us(plan, names), quality)


def prepare_esp_group(
    esp1_buffer: Sequence[Mapping[str, Any]],
    esp2_buffer: Sequence[Mapping[str, Any]],
    esp3_buffer: Sequence[Mapping[str, Any]],
) -> PreparedGroup:
    """Tiền xử lý cả nhóm ESP.

    Tensor ở đây vẫn (64,1000); bỏ 8 sub và thêm antenna A=1 nằm trong predictor.py.
    """
    names = ("esp1", "esp2", "esp3")
    plan = select_window_plan(dict(zip(names, (esp1_buffer, esp2_buffer, esp3_buffer))))
    tensors: dict[str, np.ndarray] = {}
    quality: dict[str, DeviceQuality] = {}
    for name in names:
        tensor, report = _build_esp_tensor(plan.packets_by_device[name], plan.target_seqs, name)
        tensors[name] = tensor
        quality[name] = report
    return PreparedGroup(tensors, plan.start_seq, plan.end_seq, _input_timestamp_us(plan, names), quality)


# Hàm tiện dùng nếu chỉ cần ba tensor, chưa cần metadata start_seq/quality/timestamp.
def prepare_asus_tensors(asus1_buffer, asus2_buffer, asus3_buffer):
    """Wrapper ngắn: trả (asus1_tensor, asus2_tensor, asus3_tensor)."""
    group = prepare_asus_group(asus1_buffer, asus2_buffer, asus3_buffer)
    return group.as_tuple(("asus1", "asus2", "asus3"))


def prepare_esp_tensors(esp1_buffer, esp2_buffer, esp3_buffer):
    """Wrapper ngắn: trả (esp1_tensor, esp2_tensor, esp3_tensor)."""
    group = prepare_esp_group(esp1_buffer, esp2_buffer, esp3_buffer)
    return group.as_tuple(("esp1", "esp2", "esp3"))
