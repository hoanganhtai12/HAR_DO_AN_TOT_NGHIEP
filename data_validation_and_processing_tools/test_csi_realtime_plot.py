import asyncio
import ctypes
import json
import math
import threading
import time
from collections import deque

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore


# =====================================================
# CONFIG
# =====================================================

B_HOST = "127.0.0.1"
B_PORT = 9100

TARGET_DEVICES = [
    "04:d4:c4:b5:8e:7c",
    "04:d4:c4:b8:76:64",
    "04:d4:c4:1c:0a:c4",
]

CSI_SUB_INDEX = 15

SEND_FREQ = 200

# delay giữ 30 giây
WINDOW_SECONDS = 30
MAX_DELAY_POINTS = SEND_FREQ * WINDOW_SECONDS

# CSI chỉ giữ 1000 điểm gần nhất
MAX_CSI_POINTS = 5000
AMP_MAX = 10000

# =====================================================
# CSI UNPACK
# =====================================================

def unpack_csi_amplitude(x: int,
                         nman=12,
                         nexp=6,
                         nbits=10) -> float:

    x = ctypes.c_uint32(x).value

    iq_mask = ctypes.c_uint32(
        (1 << (nman - 1)) - 1
    ).value

    e_mask = ctypes.c_uint32(
        (1 << nexp) - 1
    ).value

    e_p = 1 << (nexp - 1)

    sgnr_mask = ctypes.c_uint32(
        1 << (nexp + 2 * nman - 1)
    ).value

    sgni_mask = ctypes.c_uint32(
        sgnr_mask >> nman
    ).value

    # =====================================
    # extract
    # =====================================

    vi = ctypes.c_uint32(
        (x >> (nexp + nman)) & iq_mask
    ).value

    vq = ctypes.c_uint32(
        (x >> nexp) & iq_mask
    ).value

    e = ctypes.c_int32(
        x & e_mask
    ).value

    if e >= e_p:
        e -= (e_p << 1)

    # =====================================
    # sign extend
    # =====================================

    if x & sgnr_mask:

        vi = ctypes.c_int32(
            vi | 0xFFFFF000
        ).value

    else:

        vi = ctypes.c_int32(vi).value

    if x & sgni_mask:

        vq = ctypes.c_int32(
            vq | 0xFFFFF000
        ).value

    else:

        vq = ctypes.c_int32(vq).value

    # =====================================
    # scale
    # =====================================

    shift = e + nbits

    if shift > 20:
        shift = 20

    if shift < -20:
        shift = -20

    if shift < -nman:

        vi = 0
        vq = 0

    elif shift < 0:

        vi = ctypes.c_int32(
            vi >> (-shift)
        ).value

        vq = ctypes.c_int32(
            vq >> (-shift)
        ).value

    else:

        vi = ctypes.c_int32(
            vi << shift
        ).value

        vq = ctypes.c_int32(
            vq << shift
        ).value

    # =====================================
    # amplitude
    # =====================================

    amp = math.sqrt(
        float(vi * vi + vq * vq)
    )

    if math.isnan(amp) or math.isinf(amp):
        return 0

    if amp > AMP_MAX:
        amp = AMP_MAX

    return amp


# =====================================================
# BUFFER
# =====================================================

delay_x = deque(
    maxlen=MAX_DELAY_POINTS
)

delay_y = deque(
    maxlen=MAX_DELAY_POINTS
)

csi_buffers = {}

for dev in TARGET_DEVICES:

    csi_buffers[dev] = deque(
        maxlen=MAX_CSI_POINTS
    )

packet_counter = 0


# =====================================================
# TCP RECEIVER
# =====================================================

async def tcp_receiver():

    global packet_counter

    print(
        f"[CLIENT] Connecting to "
        f"{B_HOST}:{B_PORT}"
    )

    reader, writer = await asyncio.open_connection(
        B_HOST,
        B_PORT
    )

    print("[CLIENT] Connected")

    try:

        while True:

            line_raw = await reader.readline()

            if not line_raw:
                print("[CLIENT] Server disconnected")
                break

            recv_ts_us = time.time_ns() // 1000

            try:

                text = line_raw.decode(
                    "utf-8",
                    errors="replace"
                ).strip()

                packet = json.loads(text)

                device_id = packet.get(
                    "device_id"
                )

                # =================================
                # FILTER DEVICE
                # =================================

                if device_id not in TARGET_DEVICES:
                    continue

                # =================================
                # DELAY
                # =================================

                packet_ts = int(
                    packet["timestamp"] // 1000
                )

                delta_us = (
                    recv_ts_us - packet_ts
                )

                delta_ms = (
                    delta_us / 1_000
                )

                if delta_ms > 1000:
                    delta_ms = 1000

                if delta_ms < 0:
                    delta_ms = 0

                delay_x.append(
                    packet_counter
                )

                delay_y.append(
                    delta_ms
                )

                # =================================
                # CSI
                # =================================

                csi = packet.get(
                    "csi",
                    {}
                )

                c0 = csi.get(
                    "c0",
                    []
                )

                if len(c0) <= CSI_SUB_INDEX:
                    continue

                packed_value = c0[
                    CSI_SUB_INDEX
                ]

                if not isinstance(
                    packed_value,
                    int
                ):
                    continue

                amp = unpack_csi_amplitude(
                    packed_value
                )

                csi_buffers[
                    device_id
                ].append(amp)

                packet_counter += 1

                # log mỗi giây
                if (
                    packet_counter %
                    SEND_FREQ
                ) == 0:

                    print(
                        f"packet="
                        f"{packet_counter}"
                        f"time_ms="
                        f"{delta_ms}"
                    )

            except Exception as e:

                print(
                    "[ERROR]",
                    e
                )

    finally:

        writer.close()

        await writer.wait_closed()


def start_async_loop():

    loop = asyncio.new_event_loop()

    asyncio.set_event_loop(loop)

    loop.run_until_complete(
        tcp_receiver()
    )


# =====================================================
# GUI
# =====================================================

app = QtWidgets.QApplication([])

win = pg.GraphicsLayoutWidget(
    title="Realtime CSI + Delay Monitor"
)

win.resize(1600, 900)


# =====================================================
# DELAY PLOT
# =====================================================

plot_delay = win.addPlot(
    title="Timestamp Delay"
)

plot_delay.setLabel(
    "left",
    "Delay (sec)"
)

plot_delay.setLabel(
    "bottom",
    "Packet Index"
)

plot_delay.showGrid(
    x=True,
    y=True
)

plot_delay.setYRange(
    0,
    5000
)

curve_delay = plot_delay.plot(
    pen=pg.mkPen(width=2)
)

win.nextRow()


# =====================================================
# CSI PLOTS
# =====================================================

csi_curves = {}
csi_plots = {}

for dev in TARGET_DEVICES:

    plot = win.addPlot(
        title=(
            f"CSI c0"
            f"[{CSI_SUB_INDEX}]"
            f" - {dev}"
        )
    )

    plot.setLabel(
        "left",
        "Amplitude"
    )

    plot.setLabel(
        "bottom",
        "Rolling Index"
    )

    plot.showGrid(
        x=True,
        y=True
    )

    # FIXED SCALE
    plot.setYRange(
        0,
        AMP_MAX
    )

    # FIXED X RANGE
    plot.setXRange(
        0,
        MAX_CSI_POINTS
    )

    curve = plot.plot(
        pen=pg.mkPen( 'g', width=2)
    )

    csi_curves[dev] = curve
    csi_plots[dev] = plot

    win.nextRow()

win.show()


# =====================================================
# UPDATE GUI
# =====================================================

def update_plot():

    # =====================================
    # DELAY
    # =====================================

    if len(delay_x) > 0:

        x = np.array(
            delay_x,
            dtype=np.float32
        )

        y = np.array(
            delay_y,
            dtype=np.float32
        )

        curve_delay.setData(
            x,
            y
        )

    # =====================================
    # CSI
    # =====================================

    for dev in TARGET_DEVICES:

        buf = csi_buffers[dev]

        if len(buf) == 0:
            continue

        y = np.array(
            buf,
            dtype=np.float32
        )

        # rolling local index
        x = np.arange(
            len(y),
            dtype=np.float32
        )

        csi_curves[dev].setData(
            x,
            y
        )


timer = QtCore.QTimer()

timer.timeout.connect(
    update_plot
)

# ~20 FPS GUI
timer.start(50)


# =====================================================
# START TCP THREAD
# =====================================================

tcp_thread = threading.Thread(
    target=start_async_loop,
    daemon=True
)

tcp_thread.start()


# =====================================================
# RUN GUI
# =====================================================

QtWidgets.QApplication.instance().exec_()