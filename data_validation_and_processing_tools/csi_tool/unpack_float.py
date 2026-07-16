import numpy as np
from typing import Tuple
import ctypes
import os

# Nếu chạy trên Windows
# ifdef _WIN32
# define EXPORT __declspec(dllexport)
# else
# define EXPORT
# endif

# Tải thư viện động C
lib = ctypes.CDLL("./unpack.dll")

lib.unpack_float_acphy.argtypes = [
    ctypes.c_int,  # nbits
    ctypes.c_int,  # autoscale
    ctypes.c_int,  # shft
    ctypes.c_int,  # fmt
    ctypes.c_int,  # nman
    ctypes.c_int,  # nexp
    ctypes.c_int,  # nfft

    np.ctypeslib.ndpointer(dtype=np.uint32, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),
]

lib.unpack_float_acphy.restype = None


def unpack_float_acphy_c(
    nbits,
    autoscale,
    shft,
    fmt,
    nman,
    nexp,
    nfft,
    H
):
    H = np.ascontiguousarray(H, dtype=np.uint32)
    Hout = np.zeros(nfft * 2, dtype=np.int32)

    lib.unpack_float_acphy(
        nbits,
        autoscale,
        shft,
        fmt,
        nman,
        nexp,
        nfft,
        H,
        Hout
    )

    return Hout


def unpack_float(format_type: int, nfft: int, H: np.ndarray) -> np.ndarray:
    """Wrapper cho unpack_float_acphy - hỗ trợ format 0 & 1"""
    
    if format_type not in [0, 1]:
        raise ValueError(f"format can only be 0 or 1, got {format_type}")
    
    if len(H) < nfft:
        raise ValueError(f"Length of H ({len(H)}) must be at least nfft ({nfft})")
    
    if format_type == 0:
        return unpack_float_acphy_c(10, 0, 0, 1, 9, 5, nfft, H)
    else:  # format_type == 1
        return unpack_float_acphy_c(10, 0, 0, 1, 12, 6, nfft, H)


def process_csi_data(input_data, num_subcarriers, num_antennas=4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Xử lý dữ liệu CSI từ byte array → magnitude & phase, phân rã độc lập theo từng ăng-ten.
    
    Format mặc định = 1 (12 bit mantissa, 6 bit exponent)
    
    Args:
        input_data: Mảng byte (little endian) hoặc uint32 array.
        num_subcarriers (int): TỔNG số lượng subcarrier của tất cả các ăng-ten cộng lại.
        num_antennas (int): Số lượng ăng-ten thu (mặc định = 4).
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: (magnitude, phase)
            - magnitude: Biên độ, shape = (num_subcarriers,)
            - phase: Pha (radian), shape = (num_subcarriers,)
    """
    
    if num_subcarriers % num_antennas != 0:
        raise ValueError(f"Tổng số subcarrier ({num_subcarriers}) phải chia hết cho số lượng ăng-ten ({num_antennas})")
        
    subc_per_ant = num_subcarriers // num_antennas

    # ===== CASE 1: đã là uint32 array =====
    if isinstance(input_data, np.ndarray) and input_data.dtype == np.uint32:
        if input_data.size != num_subcarriers:
            raise ValueError(
                f"Invalid uint32 array size: "
                f"expected {num_subcarriers}, got {input_data.size}"
            )
        H = input_data

    # ===== CASE 2: raw bytes =====
    else:
        expected_bytes = num_subcarriers * 4
        if len(input_data) != expected_bytes:
            raise ValueError(
                f"Invalid byte_data size: "
                f"expected {expected_bytes}, got {len(input_data)}"
            )
        H = np.frombuffer(input_data, dtype='<u4')
    
    all_magnitudes = []
    all_phases = []
    
    # Vòng lặp duyệt qua từng ăng-ten và giải nén độc lập
    for i in range(num_antennas):
        start_idx = i * subc_per_ant
        end_idx = (i + 1) * subc_per_ant
        H_ant = H[start_idx:end_idx]
        
        # Giải nén (format = 1 fixed) với tham số nfft là số subcarrier của 1 ăng-ten
        result = unpack_float(format_type=1, nfft=subc_per_ant, H=H_ant)
        
        # Tách I & Q cho ăng-ten hiện tại
        I = result[0::2].astype(np.float64)
        Q = result[1::2].astype(np.float64)
        
        # Tính magnitude & phase
        magnitude = np.sqrt(I**2 + Q**2)
        phase = np.arctan2(Q, I)
        
        all_magnitudes.append(magnitude)
        all_phases.append(phase)
        
    # Tổng hợp (móc nối) kết quả của tất cả các ăng-ten lại thành một mảng duy nhất
    final_magnitude = np.concatenate(all_magnitudes)
    final_phase = np.concatenate(all_phases)
    
    return final_magnitude, final_phase