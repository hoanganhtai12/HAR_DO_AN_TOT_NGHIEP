# collection_stub/esp32_collection_stub.py
#
# Mô phỏng ESP32-Collection khi CHƯA có thiết bị thật.
#
# Quy ước gửi CSI:
#   - Mỗi round gửi lần lượt ESP1 -> ESP2 -> ESP3.
#   - Ba packet trong cùng một round có CÙNG seq và CÙNG timestamp.
#   - Gửi xong cả ba thì seq mới tăng 1; 4095 -> 0.
#   - PACKET_RATE_HZ_PER_DEVICE là tần số của MỖI ESP.
#     Ví dụ 200.0 Hz => trong 5 giây mỗi ESP gửi xấp xỉ 1000 packet,
#     tổng cộng tối đa 600 JSON Lines/giây cho ba ESP.

import random
import time

from tcp_stream_server import TcpStreamServer


# Phải khớp ESP_PORT trong realtime_ai_app.py.
ESP_PORT = 9201

# Nếu True: 3 ESP gửi CSI ngay khi server chạy, không cần lệnh uart_control.
AUTO_CONNECT = True

# ===== CẤU HÌNH FAKE STREAM =====
# Tần số gửi của MỖI ESP (không phải tổng cả 3 ESP).
# Với window 5 giây và T=1000, để 200.0 Hz.
PACKET_RATE_HZ_PER_DEVICE = 200.0

# None: chọn ngẫu nhiên một số 0..4095 khi khởi động.
# Hoặc đặt số cố định, ví dụ START_SEQ = 0.
START_SEQ = None

SEQ_MODULO = 4096
FAKE_COM_PORTS = ["COM3", "COM4", "COM5", "COM6", "COM7", "COM8"]
CSI_PAIR_COUNT = 64

# Thứ tự list này chính là thứ tự JSON được gửi trong mỗi round: ESP1 -> ESP2 -> ESP3.
ESP_MACS = [
    "D0:CF:13:ED:2E:EC",
    "D0:CF:13:EB:8A:9C",
    "D0:CF:13:EC:49:04",
]


def unix_now_us() -> int:
    return time.time_ns() // 1_000


def fake_csi_values() -> list[list[int]]:
    """Tạo fake CSI 64 cặp Q/I: [[q0, i0], [q1, i1], ...]."""
    return [
        [random.randint(-128, 127), random.randint(-128, 127)]
        for _ in range(CSI_PAIR_COUNT)
    ]


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
        port=ESP_PORT,
        name="ESP32-Collection",
    )

    devices = {
        mac: {
            # AUTO_CONNECT=True -> gửi data ngay; False -> chờ lệnh connect.
            "connected": AUTO_CONNECT,
            "com": "COM3" if AUTO_CONNECT else None,
            "baudrate": 115200,
        }
        for mac in ESP_MACS
    }

    def send_com_list():
        server.send_packet({
            "type": "com_list",
            "ports": FAKE_COM_PORTS,
        })

    def send_uart_status(device_id: str, status: str = "updated", message: str = ""):
        packet = {
            "type": "uart_status",
            "device_id": device_id,
            "status": status,
            "config": devices.get(device_id),
        }

        if message:
            packet["message"] = message

        server.send_packet(packet)

    def handle_message(message: dict):
        """
        Nhận lệnh từ Management.

        Message hỗ trợ:
        - {"type":"get_com_ports"}
        - {"type":"uart_control","action":"connect","device_id":"...","com":"COM3","baudrate":115200}
        - {"type":"uart_control","action":"disconnect","device_id":"..."}
        """
        print("[ESP32-Collection] RX:", message)

        msg_type = message.get("type")
        action = message.get("action")

        if msg_type == "get_com_ports" or action == "refresh_com_ports":
            send_com_list()
            return

        if msg_type not in {"uart_control", "configure_device"} and action not in {
            "connect",
            "disconnect",
            "configure_device",
        }:
            server.send_packet({
                "type": "control_ack",
                "status": "ignored",
                "message": "Message type/action không hỗ trợ",
                "raw": message,
            })
            return

        device_id = message.get("device_id") or message.get("uartId")
        if device_id is not None:
            device_id = str(device_id).strip().upper()

        if device_id not in devices:
            server.send_packet({
                "type": "uart_status",
                "status": "error",
                "message": f"Thiết bị không hợp lệ: {device_id}",
                "device_id": device_id,
            })
            return

        if action == "disconnect" or message.get("enabled") is False:
            devices[device_id]["connected"] = False
            send_uart_status(device_id, status="disconnected")
            return

        # connect/configure_device
        com = message.get("com") or message.get("port") or devices[device_id]["com"]
        baudrate = message.get("baudrate") or message.get("baudRate") or devices[device_id]["baudrate"]

        if com not in FAKE_COM_PORTS:
            send_uart_status(
                device_id,
                status="error",
                message=f"COM không nằm trong danh sách fake: {com}",
            )
            return

        devices[device_id]["com"] = com
        devices[device_id]["baudrate"] = int(baudrate)
        devices[device_id]["connected"] = True
        send_uart_status(device_id, status="connected")

    server.set_message_handler(handle_message)
    server.set_client_connected_handler(send_com_list)
    server.start()

    seq = resolve_start_seq()
    interval_sec = 1.0 / float(PACKET_RATE_HZ_PER_DEVICE)
    next_round_at = time.perf_counter()

    if AUTO_CONNECT:
        print(
            f"[ESP32-Collection] AUTO_CONNECT bật: gửi ESP1 -> ESP2 -> ESP3, "
            f"{PACKET_RATE_HZ_PER_DEVICE:g} Hz/thiết bị, seq bắt đầu {seq}, port {ESP_PORT}."
        )

    while True:
        # Duyệt theo ESP_MACS, không random thứ tự.
        connected_devices = [
            device_id
            for device_id in ESP_MACS
            if devices[device_id].get("connected")
        ]

        if not connected_devices:
            time.sleep(0.1)
            next_round_at = time.perf_counter()
            continue

        # Giữ tần số theo từng round; một round bao gồm packet của tất cả ESP đang connected.
        now = time.perf_counter()
        wait_sec = next_round_at - now
        if wait_sec > 0:
            time.sleep(wait_sec)
        elif now - next_round_at > interval_sec:
            # Không cố "bù" các round bị trễ để tránh gửi burst packet.
            next_round_at = now

        # Ba thiết bị trong cùng round cùng seq và timestamp để dễ kiểm thử đồng bộ.
        round_timestamp = unix_now_us()
        round_seq = seq

        for device_id in connected_devices:
            packet = {
                "type": "csi_data",
                "device_id": device_id,
                "seq": round_seq,
                "timestamp": round_timestamp,
                "radio": {
                    "rssi": random.randint(-80, -30),
                    "channel": random.choice([1, 6, 11]),
                    "agc_gain": random.randint(0, 3),
                    "fft_gain": random.randint(0, 3),
                    "noise_floor": random.randint(-100, -85),
                },
                "csi": fake_csi_values(),
            }
            server.send_packet(packet)

        # Chỉ tăng seq sau khi đã gửi xong ESP1, ESP2, ESP3 của cùng round.
        seq = (seq + 1) % SEQ_MODULO
        next_round_at += interval_sec


if __name__ == "__main__":
    main()
