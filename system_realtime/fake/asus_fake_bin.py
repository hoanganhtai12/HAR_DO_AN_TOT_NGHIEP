# collection_stub/nexmon_collection_stub.py
#
# Mô phỏng Nexmon-Collection (ASUS) khi CHƯA có thiết bị thật.
#
# Quy ước gửi CSI:
#   - Mỗi round gửi lần lượt ASUS1 -> ASUS2 -> ASUS3.
#   - Ba packet trong cùng một round có CÙNG seq và CÙNG timestamp.
#   - Gửi xong cả ba thì seq mới tăng 1; 4095 -> 0.
#   - PACKET_RATE_HZ_PER_DEVICE là tần số của MỖI ASUS.
#     Ví dụ 200.0 Hz => trong 5 giây mỗi ASUS gửi xấp xỉ 1000 packet,
#     tổng cộng 600 JSON Lines/giây cho ba ASUS.

import random
import time

from tcp_stream_server import TcpStreamServer


# Phải khớp ASUS_PORT trong realtime_ai_app.py.
ASUS_PORT = 9100

# ===== CẤU HÌNH FAKE STREAM =====
# Tần số gửi của MỖI ASUS (không phải tổng cả 3 ASUS).
# Với window 5 giây và T=1000, để 200.0 Hz.
PACKET_RATE_HZ_PER_DEVICE = 200.0

# None: chọn ngẫu nhiên một số 0..4095 khi khởi động.
# Hoặc đặt số cố định, ví dụ START_SEQ = 0.
START_SEQ = None

CSI_SUBCARRIER_COUNT = 64
ANTENNA_COUNT = 4
SEQ_MODULO = 4096

NEXMON_CHANNEL = 157
NEXMON_BW = 20

# Thứ tự list này chính là thứ tự JSON được gửi trong mỗi round: ASUS1 -> ASUS2 -> ASUS3.
ASUS_MACS = [
    "04:D4:C4:B5:8E:7C",
    "04:D4:C4:B8:76:64",
    "04:D4:C4:1C:0A:C4",
]


def unix_now_us() -> int:
    return time.time_ns() // 1_000


def pack_qi_to_uint32(q: int, i: int) -> int:
    """Pack 1 cặp Q/I thành uint32: Q ở 16 bit thấp, I ở 16 bit cao."""
    return (q & 0xFFFF) | ((i & 0xFFFF) << 16)


def fake_csi_uint32_values() -> list[int]:
    """Tạo 64 giá trị CSI uint32 cho 1 antenna."""
    values = []
    for _ in range(CSI_SUBCARRIER_COUNT):
        q = random.randint(-32768, 32767)
        i = random.randint(-32768, 32767)
        values.append(pack_qi_to_uint32(q, i))
    return values


def fake_nexmon_csi() -> dict:
    """Tạo CSI cho 4 antenna c0/c1/c2/c3, mỗi antenna 64 uint32."""
    return {
        f"c{ant}": fake_csi_uint32_values()
        for ant in range(ANTENNA_COUNT)
    }


def fake_agc() -> list[int]:
    return [random.randint(0, 255) for _ in range(ANTENNA_COUNT)]


def fake_rssi() -> list[int]:
    return [random.randint(-80, -30) for _ in range(ANTENNA_COUNT)]


def resolve_start_seq() -> int:
    """Lấy seq khởi tạo hợp lệ trong khoảng 0..4095."""
    if START_SEQ is None:
        return random.randrange(SEQ_MODULO)

    start_seq = int(START_SEQ)
    if not 0 <= start_seq < SEQ_MODULO:
        raise ValueError(f"START_SEQ phải nằm trong 0..{SEQ_MODULO - 1}, nhận được {START_SEQ!r}")
    return start_seq


def main():
    if PACKET_RATE_HZ_PER_DEVICE <= 0:
        raise ValueError("PACKET_RATE_HZ_PER_DEVICE phải lớn hơn 0")

    server = TcpStreamServer(
        host="127.0.0.1",
        port=ASUS_PORT,
        name="Nexmon-Collection",
    )

    server.start()

    seq = resolve_start_seq()
    interval_sec = 1.0 / float(PACKET_RATE_HZ_PER_DEVICE)
    next_round_at = time.perf_counter()

    print(
        f"[Nexmon-Collection] Gửi ASUS1 -> ASUS2 -> ASUS3, "
        f"{PACKET_RATE_HZ_PER_DEVICE:g} Hz/thiết bị, seq bắt đầu {seq}, port {ASUS_PORT}."
    )

    while True:
        # Giữ tần số theo từng round; mỗi round luôn có đủ ASUS1, ASUS2, ASUS3.
        now = time.perf_counter()
        wait_sec = next_round_at - now
        if wait_sec > 0:
            time.sleep(wait_sec)
        elif now - next_round_at > interval_sec:
            # Không cố "bù" các round bị trễ để tránh gửi burst packet.
            next_round_at = now

        # Ba ASUS trong cùng round cùng seq và timestamp để dễ kiểm thử đồng bộ.
        round_timestamp = unix_now_us()
        round_seq = seq

        # Duyệt theo ASUS_MACS, không random thứ tự.
        for device_id in ASUS_MACS:
            packet = {
                "device_id": device_id,
                "seq": round_seq,
                "timestamp": round_timestamp,
                "bw": NEXMON_BW,
                "ch": NEXMON_CHANNEL,
                "agc": fake_agc(),
                "rssi": fake_rssi(),
                "csi": fake_nexmon_csi(),
            }
            server.send_packet(packet)

        # Chỉ tăng seq sau khi đã gửi xong ASUS1, ASUS2, ASUS3 của cùng round.
        seq = (seq + 1) % SEQ_MODULO
        next_round_at += interval_sec


if __name__ == "__main__":
    main()
