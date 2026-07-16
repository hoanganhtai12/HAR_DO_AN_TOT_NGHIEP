"""
server2server.py

Kiến trúc:

    Client A1 \
    Client A2 ----> Server A ----forward----> Client B1
    Client A3 /                           \-> Client B2

- Server A:
    + TCP server
    + tối đa 3 client
    + nhận JSON kết thúc bởi '\n'

- Server B:
    + TCP server
    + tối đa 2 client
    + nhận dữ liệu forward từ A

Python >= 3.8
"""

import socket
import threading


HOST = "0.0.0.0"

PORT_A = 9001
PORT_B = 9100

MAX_CLIENT_A = 3
MAX_CLIENT_B = 2

BUFFER_SIZE = 4096


# ============================================================
# Global state
# ============================================================

clients_a = []
clients_a_lock = threading.Lock()

clients_b = []
clients_b_lock = threading.Lock()


# ============================================================
# Forward sang tất cả client ở server B
# ============================================================

def forward_to_b(data: bytes):

    dead_clients = []

    with clients_b_lock:

        if len(clients_b) == 0:
            print("[B] No client connected")
            return

        for conn in clients_b:

            try:

                conn.sendall(data)

            except Exception as e:

                print(f"[B] Send error: {e}")

                dead_clients.append(conn)

        # remove dead client
        for conn in dead_clients:

            try:
                conn.close()
            except:
                pass

            if conn in clients_b:
                clients_b.remove(conn)


# ============================================================
# Handle client của Server A
# ============================================================

def handle_client_a(conn: socket.socket, addr):

    print(f"[A] Client connected: {addr}")

    buffer = ""

    try:

        while True:

            data = conn.recv(BUFFER_SIZE)

            if not data:
                break

            buffer += data.decode(
                "utf-8",
                errors="ignore"
            )

            while "\n" in buffer:

                line, buffer = buffer.split(
                    "\n",
                    1
                )

                line = line.strip()

                if not line:
                    continue

                json_msg = line + "\n"

                # forward sang B clients
                forward_to_b(
                    json_msg.encode()
                )

    except Exception as e:

        print(f"[A] Error: {e}")

    finally:

        print(f"[A] Client disconnected: {addr}")

        with clients_a_lock:

            if conn in clients_a:
                clients_a.remove(conn)

        try:
            conn.close()
        except:
            pass


# ============================================================
# Handle client của Server B
# ============================================================

def handle_client_b(conn: socket.socket, addr):

    print(f"[B] Client connected: {addr}")

    try:

        while True:

            data = conn.recv(BUFFER_SIZE)

            if not data:
                break

            print(
                f"[B] RX from B-client: "
                f"{data.decode(errors='ignore').strip()}"
            )

    except Exception as e:

        print(f"[B] Error: {e}")

    finally:

        print(f"[B] Client disconnected: {addr}")

        with clients_b_lock:

            if conn in clients_b:
                clients_b.remove(conn)

        try:
            conn.close()
        except:
            pass


# ============================================================
# Server A
# ============================================================

def server_a_thread():

    server = socket.socket(
        socket.AF_INET,
        socket.SOCK_STREAM
    )

    server.setsockopt(
        socket.SOL_SOCKET,
        socket.SO_REUSEADDR,
        1
    )

    server.bind((HOST, PORT_A))

    server.listen()

    print(f"[A] Listening on {PORT_A}")

    while True:

        conn, addr = server.accept()

        with clients_a_lock:

            if len(clients_a) >= MAX_CLIENT_A:

                print("[A] Reject client: max reached")

                conn.close()

                continue

            clients_a.append(conn)

        t = threading.Thread(
            target=handle_client_a,
            args=(conn, addr),
            daemon=True
        )

        t.start()


# ============================================================
# Server B
# ============================================================

def server_b_thread():

    server = socket.socket(
        socket.AF_INET,
        socket.SOCK_STREAM
    )

    server.setsockopt(
        socket.SOL_SOCKET,
        socket.SO_REUSEADDR,
        1
    )

    server.bind((HOST, PORT_B))

    server.listen()

    print(f"[B] Listening on {PORT_B}")

    while True:

        conn, addr = server.accept()

        with clients_b_lock:

            if len(clients_b) >= MAX_CLIENT_B:

                print("[B] Reject extra client")

                conn.close()

                continue

            clients_b.append(conn)

        t = threading.Thread(
            target=handle_client_b,
            args=(conn, addr),
            daemon=True
        )

        t.start()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    threading.Thread(
        target=server_a_thread,
        daemon=True
    ).start()

    threading.Thread(
        target=server_b_thread,
        daemon=True
    ).start()

    print("Server2Server running...")

    while True:

        try:

            threading.Event().wait(1)

        except KeyboardInterrupt:

            print("Exit")

            break