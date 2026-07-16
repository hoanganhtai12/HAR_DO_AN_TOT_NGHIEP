import os
import numpy as np

from scipy.interpolate import PchipInterpolator
from scipy.signal import butter, filtfilt

# ==================================================
# CONFIG
# ==================================================

SRC_ROOT = r"data/sshar"
DST_ROOT = r"data/sshar_amp_interp"

# Supported:
#   esp
#   asus
DEVICE_TYPE = "asus"

FS = 200
CUTOFF = 20


# ==================================================
# VALID SUBCARRIERS
# ==================================================

# ESP32:
# Keep:
#   0 ~ 27
#   29 ~ 56
#
# Remove:
#   28
#   57 ~ 63
ESP32_VALID_SUBCARRIERS = (
    list(range(0, 28)) +
    list(range(29, 57))
)


# ASUS
#
# Sau khi reorder:
#   -32 ... -1 1 ... 28 0 29 30 31
#
# Bỏ:
#   -32 -31 -30 -29
#   0   +29 +30 +31
#
# Giữ:
#   -28 ... -1
#    1  ... +28
ASUS_VALID_SUBCARRIERS = list(range(4, 60))

# ==================================================
# PCHIP INTERPOLATION
# ==================================================
def pchip_interpolate_zeros(signal):
    """
    Nội suy packet lỗi (=0) bằng PCHIP

    Input:
        signal: (T,)

    Output:
        signal: (T,)
    """

    signal = np.asarray(
        signal,
        dtype=np.float32,
    )

    valid_idx = np.where(
        signal > 1e-5
    )[0]

    if len(valid_idx) < 2:
        return signal

    interp = PchipInterpolator(
        valid_idx,
        signal[valid_idx],
        extrapolate=True,
    )

    result = signal.copy()
    invalid_idx = np.where(
        signal <= 1e-5
    )[0]

    result[invalid_idx] = interp(
        invalid_idx
    )

    return result.astype(
        np.float32
    )


# ==================================================
# HAMPEL FILTER
# ==================================================

def hampel_filter(
    x,
    window_size=5,
    n_sigma=3,
):
    """
    Hampel filter
    """

    x = np.asarray(
        x,
        dtype=np.float32,
    )

    y = x.copy()
    n = len(x)
    k = 1.4826

    for i in range(n):
        start = max(
            0,
            i - window_size,
        )
        end = min(
            n,
            i + window_size + 1,
        )
        window = x[start:end]
        median = np.median(
            window
        )
        mad = np.median(
            np.abs(
                window - median
            )
        )
        threshold = (
            n_sigma * k * mad
        )
        if abs(
            x[i] - median
        ) > threshold:
            y[i] = median

    return y


# ==================================================
# BUTTERWORTH
# ==================================================

def butter_lowpass_filter(
    x,
    cutoff,
    fs,
    order=4,
):
    """
    Butterworth low-pass
    """

    nyquist = fs * 0.5

    normal_cutoff = (
        cutoff / nyquist
    )

    b, a = butter(
        order,
        normal_cutoff,
        btype="low",
    )
    return filtfilt(
        b,
        a,
        x,
    )


# ==================================================
# NORMALIZATION
# ==================================================

def normalize_zscore(x):
    """
    Z-score normalization
    """

    mean = np.mean(x)

    std = np.std(x)
    return (
        x - mean
    ) / (
        std + 1e-6
    )

# ==================================================
# SELECT VALID SUBCARRIERS
# ==================================================
def select_valid_subcarriers(x):
    """
    Input:
        (num_ant, num_sub, num_time)

    Output:

    ESP:
        - Keep 56 valid subcarriers
        - Order:
            -32 ... +31 (đã đúng)

    ASUS:
        - Keep all subcarriers
        - Reorder:
            0 ... 31 -32 ... -1
                ↓
            -32 ... -1 0 ... 31
    """

    device = DEVICE_TYPE.lower()
    if device == "esp":
        return x[
            :,
            ESP32_VALID_SUBCARRIERS,
            :
        ]
    elif device == "asus":
        # ----------------------------------------------
        # Reorder:
        #
        # 0 ... 31 -32 ... -1
        #
        # ->
        #
        # -32 ... -1 0 ... 31
        # ----------------------------------------------
        x = np.concatenate(
            (
                x[:, 32:, :],
                x[:, :32, :]
            ),
            axis=1
        )

        return x[
            :,
            ASUS_VALID_SUBCARRIERS,
            :
        ]
    else:
        raise ValueError(
            f"Unsupported DEVICE_TYPE: {DEVICE_TYPE}"
        )

# ==================================================
# PREPROCESS
# ==================================================

def preprocess_amplitude(
    x,
):
    """
    Input:
        (num_ant, num_sub, num_time)

    Output:
        ESP:
            (num_ant, 56, num_time)

        ASUS:
            (num_ant, num_sub, num_time)

    Pipeline

        PCHIP

            ↓

        Select valid subcarriers
    """

    num_ant, num_sub, num_time = (
        x.shape
    )

    out = np.zeros_like(
        x,
        dtype=np.float32,
    )

    for ant in range(
        num_ant
    ):

        for sub in range(
            num_sub
        ):

            sig = x[
                ant,
                sub,
                :
            ]

            sig = pchip_interpolate_zeros(
                sig
            )

            out[
                ant,
                sub,
                :
            ] = sig

    # Chọn các subcarrier hợp lệ
    out = select_valid_subcarriers(
        out
    )

    return out


# ==================================================
# MAIN
# ==================================================

os.makedirs(
    DST_ROOT,
    exist_ok=True,
)

total_files = 0

for room in os.listdir(
    SRC_ROOT
):

    room_path = os.path.join(
        SRC_ROOT,
        room,
    )

    if not os.path.isdir(
        room_path
    ):
        continue

    src_device_path = os.path.join(
        room_path,
        DEVICE_TYPE,
    )

    if not os.path.exists(
        src_device_path
    ):
        continue

    dst_device_path = os.path.join(
        DST_ROOT,
        room,
        DEVICE_TYPE,
    )

    os.makedirs(
        dst_device_path,
        exist_ok=True,
    )

    print(
        f"\nProcessing {room}"
    )

    for rx in os.listdir(
        src_device_path
    ):

        src_rx_path = os.path.join(
            src_device_path,
            rx,
        )

        if not os.path.isdir(
            src_rx_path
        ):
            continue

        dst_rx_path = os.path.join(
            dst_device_path,
            rx,
        )

        os.makedirs(
            dst_rx_path,
            exist_ok=True,
        )

        for subject in os.listdir(
            src_rx_path
        ):

            src_subject_path = os.path.join(
                src_rx_path,
                subject,
            )

            if not os.path.isdir(
                src_subject_path
            ):
                continue

            dst_subject_path = os.path.join(
                dst_rx_path,
                subject,
            )

            os.makedirs(
                dst_subject_path,
                exist_ok=True,
            )

            for file_name in os.listdir(
                src_subject_path
            ):

                if not file_name.startswith(
                    "amp_"
                ):
                    continue

                src_file = os.path.join(
                    src_subject_path,
                    file_name,
                )

                dst_file = os.path.join(
                    dst_subject_path,
                    file_name,
                )

                try:

                    x = np.load(
                        src_file
                    ).astype(
                        np.float32
                    )

                    x = preprocess_amplitude(
                        x
                    )

                    np.save(
                        dst_file,
                        x,
                    )

                    total_files += 1

                    if total_files % 100 == 0:

                        print(
                            f"Processed: {total_files}"
                        )

                except Exception as e:

                    print(
                        "\nERROR:"
                    )

                    print(
                        src_file
                    )

                    print(
                        str(e)
                    )

print()
print("=" * 80)
print(
    f"Amplitude files processed: {total_files}"
)
print(
    f"Output folder: {DST_ROOT}"
)
print("Finished.")