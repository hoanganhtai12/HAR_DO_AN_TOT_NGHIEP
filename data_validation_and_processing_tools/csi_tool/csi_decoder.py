import os
import numpy as np


from .unpack_float import process_csi_data
from .config import get_debug

DEBUG = get_debug()

def extract_csi_matrix(data_inputs: list, dev_type: str, from_buffer: bool = False) -> list:
    results = []

    for rx_idx, input_source in enumerate(data_inputs):
        if not from_buffer:
            if not os.path.exists(input_source):
                print(f"[Lỗi] Không tìm thấy file: {input_source}")
                results.append(None)
                continue

        if DEBUG: print(f"Đang giải mã ma trận ({dev_type.upper()}) cho Rx{rx_idx + 1}...")

        if dev_type == 'esp':
            esp_dtype = np.dtype([
                ('seq', '<u2'), ('timestamp', '<u8'), ('channel', '<u2'),
                ('agc', 'u1'), ('fft', 'u1'), ('noise', 'i1'), ('rssi', 'i1'),
                ('payload', 'i1', (128,))
            ])
            data = np.frombuffer(input_source, dtype=esp_dtype) if from_buffer else np.fromfile(input_source, dtype=esp_dtype)

            Q_float = data['payload'][:, 0::2].astype(np.float32)
            I_float = data['payload'][:, 1::2].astype(np.float32)
            amplitude = np.sqrt(I_float**2 + Q_float**2)
            phase = np.arctan2(Q_float, I_float)

            results.append({
                'rx_index': rx_idx,
                'timestamp': data['timestamp'],
                'amplitude': amplitude,
                'phase': phase,
                'num_antennas': 1
            })

        elif dev_type == 'asus':
            asus_dtype = np.dtype([
                ('seq', '<u2'),
                ('timestamp', '<u8'),
                ('channel', '<u2'),
                ('agc_gain', 'u1', (4,)),
                ('rssi', 'i1', (4,)),
                ('payload', '<u4', (256,))
            ])

            data = (
                np.frombuffer(input_source, dtype=asus_dtype)
                if from_buffer
                else np.fromfile(input_source, dtype=asus_dtype)
            )

            csi_raw = data['payload']
            N_packets = csi_raw.shape[0]

            # Shape:
            # (antenna, packet, subcarrier)
            amplitude = np.zeros((4, N_packets, 64), dtype=np.float32)
            phase = np.zeros((4, N_packets, 64), dtype=np.float32)

            last_amp = 0
            last_amp2 = 0
            last_amp3 = 0
            last_amp4 = 0
            last_raw = 0
            for i in range(N_packets):
                payload_u32 = csi_raw[i]   # đã là <u4, shape (256,)
                magnitude_full, phase_full = process_csi_data(payload_u32, 256)

                # TODO: comment here
                # if abs(magnitude_full[15] - last_amp) >= 1000 and (magnitude_full[15] != 0) and (last_raw != 0):
                #     print(f"packet {i}: {hex(last_raw)} -  {hex(payload_u32[15])} = {abs(magnitude_full[15] - last_amp)}")
                #     magnitude_full[15] = last_amp

                # last_amp = magnitude_full[15]
                # last_raw = payload_u32[15]

                # if abs(magnitude_full[15+64*(2-1)] - last_amp2) >= 1000 and (magnitude_full[15+64*(2-1)] != 0) and (last_amp2 != 0):
                #     magnitude_full[15+64*(2-1)] = last_amp2
                # last_amp2 = magnitude_full[15+64*(2-1)]

                # if abs(magnitude_full[15+64*(3-1)] - last_amp3) >= 1000 and (magnitude_full[15+64*(3-1)] != 0) and (last_amp3 != 0):
                #     magnitude_full[15+64*(3-1)] = last_amp3
                # last_amp3 = magnitude_full[15+64*(3-1)]

                # if abs(magnitude_full[15+64*(4-1)] - last_amp4) >= 1000 and (magnitude_full[15+64*(4-1)] != 0) and (last_amp4 != 0):
                #     magnitude_full[15+64*(4-1)] = last_amp4
                # last_amp4 = magnitude_full[15+64*(4-1)]

                # TODO: end comment here
                
                magnitude_full = magnitude_full.reshape(4, 64)
                phase_full = phase_full.reshape(4, 64)

                amplitude[:, i, :] = magnitude_full
                phase[:, i, :] = phase_full

            results.append({
                'rx_index': rx_idx,
                'timestamp': data['timestamp'],
                'amplitude': amplitude,
                'phase': phase,
                'num_antennas': 4
            })

    if DEBUG: print("[Thành công] Đã trích xuất xong mảng đa chiều vào bộ nhớ RAM!")
    return results
