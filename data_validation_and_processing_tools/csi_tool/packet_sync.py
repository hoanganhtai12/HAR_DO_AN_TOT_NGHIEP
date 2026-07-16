import os
import sys
import struct
from datetime import datetime
from .config import get_debug

DEBUG = get_debug()

SEQ_MAX = 4096
HEADER_SIZE = 10
TIMESTAMP_WINDOW_US = 1_000_000


# =========================================================
# UTIL
# =========================================================

def seq_add(seq, step=1):
    return (seq + step) % SEQ_MAX


def seq_diff(a, b):
    return (a - b) % SEQ_MAX


def read_header(packet):
    if len(packet) < HEADER_SIZE:
        return None, None

    return struct.unpack("<HQ", packet[:HEADER_SIZE])


def create_fake_packet(seq, packet_size, timestamp=0):
    packet = bytearray(packet_size)

    struct.pack_into("<H", packet, 0, seq)
    struct.pack_into("<Q", packet, 2, int(timestamp))

    return bytes(packet)


# =========================================================
# FIND FIRST PACKET
# =========================================================

def find_first_packet_fast(
    file_path,
    target_time,
    packet_size
):
    if not os.path.exists(file_path):

        print(f"[ERROR] File not found: {file_path}")

        return None, None

    file_size = os.path.getsize(file_path)

    total_packets = file_size // packet_size

    if total_packets == 0:
        return None, None

    best_seq = None
    best_ts = None

    best_index = -1

    with open(file_path, "rb") as f:

        # =================================================
        # sanity check
        # =================================================

        f.seek(0)

        header = f.read(HEADER_SIZE)

        if len(header) < HEADER_SIZE:
            return None, None

        seq0, ts0 = struct.unpack("<HQ", header)

        dt = datetime.fromtimestamp(ts0 / 1_000_000)

        if DEBUG: print(f"[INFO] File 1 starts at: {ts0} = {dt}")

        if dt.year < 2026:

            print("!!! WARNING: abnormal timestamp !!!")

            sys.exit(1)

        # =================================================
        # binary search
        # =================================================

        low = 0
        high = total_packets - 1

        while low <= high:

            mid = (low + high) // 2

            f.seek(mid * packet_size)

            header = f.read(HEADER_SIZE)

            if len(header) < HEADER_SIZE:
                break

            seq, ts = struct.unpack("<HQ", header)

            if ts >= target_time:

                best_seq = seq
                best_ts = ts

                best_index = mid

                high = mid - 1

            else:

                low = mid + 1

    if best_index == -1:

        print("[FAIL] Cannot find target packet.")

        return None, None

    delta = best_ts - target_time

    if DEBUG: print(
        f"[FOUND] "
        f"SEQ={best_seq} "
        f"TS={best_ts} "
        f"delta_us={delta}"
    )

    return best_seq, best_ts


# =========================================================
# READ FIRST HEADER
# =========================================================

def read_first_header(file_path, packet_size):

    with open(file_path, "rb") as f:

        packet = f.read(packet_size)

    if len(packet) < packet_size:
        return None, None

    return read_header(packet)


# =========================================================
# NORMALIZE SINGLE FILE
# =========================================================

def normalize_single_file(
    file_path,
    packet_size,
    global_start_seq
):
    output = bytearray()

    stats = {
        "prepend_fake": 0,
        "missing_fake": 0,
        "duplicate_skip": 0,
        "out_of_order_skip": 0,
        "real_packets": 0,
    }

    with open(file_path, "rb") as f:

        # =================================================
        # first packet
        # =================================================

        first_packet = f.read(packet_size)

        if len(first_packet) < packet_size:
            return b"", stats

        first_seq, first_ts = read_header(first_packet)

        # =================================================
        # prepend fake packets
        # =================================================

        seq_cursor = global_start_seq

        while seq_cursor != first_seq:

            fake_packet = create_fake_packet(
                seq_cursor,
                packet_size,
                first_ts
            )

            output.extend(fake_packet)

            stats["prepend_fake"] += 1

            seq_cursor = seq_add(seq_cursor)

        # append first real packet

        output.extend(first_packet)

        stats["real_packets"] += 1

        prev_seq = first_seq
        prev_ts = first_ts

        # =================================================
        # loop
        # =================================================

        while True:

            current_packet = f.read(packet_size)

            if len(current_packet) < packet_size:
                break

            current_seq, current_ts = read_header(current_packet)

            diff = seq_diff(current_seq, prev_seq)

            # duplicate

            if diff == 0:

                stats["duplicate_skip"] += 1

                continue

            # out of order

            if diff > (SEQ_MAX // 2):

                stats["out_of_order_skip"] += 1

                continue

            # missing packets

            if diff > 1:

                for step in range(1, diff):

                    fake_seq = seq_add(prev_seq, step)

                    fake_packet = create_fake_packet(
                        fake_seq,
                        packet_size,
                        prev_ts
                    )

                    output.extend(fake_packet)

                    stats["missing_fake"] += 1

            # append real packet

            output.extend(current_packet)

            stats["real_packets"] += 1

            prev_seq = current_seq
            prev_ts = current_ts

    return bytes(output), stats


# =========================================================
# NORMALIZE 3 FILES
# =========================================================

def normalize_3_files(
    file1,
    file2,
    file3,
    packet_size
):
    files = [file1, file2, file3]

    first_headers = []

    # for fp in files:
    for idx, fp in enumerate(files):
        seq, ts = read_first_header(fp, packet_size)

        if seq is None:
            return None
        
        if DEBUG: print(
        f"[File {idx+1}:] "
        f" Start timestamp = {ts} = {datetime.fromtimestamp(ts / 1_000_000)}"
        f" Start seq = {seq}"
        )

        first_headers.append((seq, ts))

    global_start_seq = min(
        seq for seq, _ in first_headers
    )


    if DEBUG: print(
        f"[NORMALIZE] "
        f"global_start_seq = {global_start_seq}"
    )

    normalized_buffers = []

    for i, fp in enumerate(files):

        if DEBUG: print("\n" + "=" * 60)

        if DEBUG: print(f"[NORMALIZE] file{i+1}")

        buffer, stats = normalize_single_file(
            fp,
            packet_size,
            global_start_seq
        )

        if DEBUG: print(stats)

        normalized_buffers.append(buffer)

    return normalized_buffers


# =========================================================
# FIND PACKET BY SEQ
# =========================================================

def find_packet_by_seq(
    buffer_bytes,
    packet_size,
    target_seq,
    target_ts
):
    total_packets = len(buffer_bytes) // packet_size

    best_index = None
    best_ts = None
    best_delta = None

    for idx in range(total_packets):

        start = idx * packet_size

        packet = buffer_bytes[
            start:start + packet_size
        ]

        seq, ts = read_header(packet)

        if seq != target_seq:
            continue

        delta = abs(ts - target_ts)

        if delta <= TIMESTAMP_WINDOW_US:

            if (
                best_delta is None
                or delta < best_delta
            ):

                best_delta = delta
                best_index = idx
                best_ts = ts

    return best_index, best_ts


# =========================================================
# CUT NORMALIZED BUFFER
# =========================================================

def cut_normalized_buffer(
    buffer_bytes,
    packet_size,
    start_index,
    packet_count
):
    total_packets = len(buffer_bytes) // packet_size

    if start_index is None:
        return b""

    # =====================================================
    # full file
    # =====================================================

    if packet_count == 0:

        return buffer_bytes[
            start_index * packet_size:
        ]

    # =====================================================
    # fixed packet count
    # =====================================================

    end_index = min(
        total_packets,
        start_index + packet_count
    )

    output = bytearray(
        buffer_bytes[
            start_index * packet_size:
            end_index * packet_size
        ]
    )

    packets_added = len(output) // packet_size

    # padding if needed

    if packets_added < packet_count:

        if packets_added > 0:

            last_packet = output[-packet_size:]

            last_seq, last_ts = read_header(
                last_packet
            )

        else:

            last_seq = 0
            last_ts = 0

        current_seq = last_seq

        while packets_added < packet_count:

            current_seq = seq_add(current_seq)

            fake_packet = create_fake_packet(
                current_seq,
                packet_size,
                last_ts
            )

            output.extend(fake_packet)

            packets_added += 1

    return bytes(output)


# =========================================================
# MAIN CUT
# =========================================================

def sync_and_cut_3_files(
    normalized_buffers,
    start_seq,
    start_ts,
    packet_size,
    dev_type,
    base_out_name,
    output_dir,
    packet_count,
    is_save=True
):
    ram_buffers = []

    for i, buffer_bytes in enumerate(normalized_buffers):
        rx_name = f"{dev_type}{i+1}"

        if DEBUG: print("\n" + "=" * 60)
        if DEBUG: print(f"[PROCESS] {rx_name}")
        # =================================================
        # find start index
        # =================================================
        start_index, matched_ts = find_packet_by_seq(
            buffer_bytes,
            packet_size,
            start_seq,
            start_ts
        )

        if start_index is None:
            print(
                f"[FAIL] Cannot find seq={start_seq}"
            )
            ram_buffers.append(b"")
            continue

        delta = matched_ts - start_ts
        if DEBUG: print(
            f"[SYNC] "
            f"matched_ts={matched_ts} "
            f"delta_us={delta}"
        )
        # =================================================
        # cut
        # =================================================
        cut_buffer = cut_normalized_buffer(
            buffer_bytes,
            packet_size,
            start_index,
            packet_count
        )

        ram_buffers.append(cut_buffer)
        # =================================================
        # save
        # =================================================
        if is_save:
            out_dir = os.path.join(
                output_dir,
                rx_name
            )

            os.makedirs(
                out_dir,
                exist_ok=True
            )

            out_path = os.path.join(
                out_dir,
                base_out_name
            )

            with open(out_path, "wb") as f_out:

                f_out.write(cut_buffer)

            print(f"[SAVE] {out_path}")

    return ram_buffers