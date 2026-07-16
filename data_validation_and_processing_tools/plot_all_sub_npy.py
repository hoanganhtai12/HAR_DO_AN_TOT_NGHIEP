import os
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import PchipInterpolator


# ==================================================
# CONFIG
# ==================================================

# Có nội suy PCHIP trước khi vẽ hay không
ENABLE_PCHIP = False


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
# SUBCARRIER LABEL
# ==================================================

def subcarrier_label(idx):
    """
    Convert index 0~63 to:

    00 -- -32
    ...
    31 -- -1
    32 -- +0
    ...
    63 -- +31
    """

    logical = idx - 32

    if logical >= 0:
        logical_str = f"+{logical}"
    else:
        logical_str = str(logical)

    return f"{idx:02d} -- {logical_str}"


# ==================================================
# PLOT
# ==================================================

def plot_all_subcarriers(
    npy_file,
    output_dir="image",
):
    """
    npy shape:
        (Antenna, Subcarrier, Time)

    Output:
        4 images / antenna
        Each image contains 16 subcarriers.
    """

    data = np.load(npy_file)

    print("Shape =", data.shape)

    if data.ndim != 3:
        raise ValueError(
            f"Expected shape (Antenna, Subcarrier, Time), got {data.shape}"
        )

    # ==================================================
    # Optional PCHIP interpolation
    # ==================================================
    if ENABLE_PCHIP:

        print("Applying PCHIP interpolation...")

        num_antennas, num_subcarriers, _ = data.shape

        for ant in range(num_antennas):

            for sub in range(num_subcarriers):

                data[
                    ant,
                    sub,
                    :
                ] = pchip_interpolate_zeros(
                    data[
                        ant,
                        sub,
                        :
                    ]
                )

        print("Done.")

    num_antennas, num_subcarriers, num_time = data.shape

    print("Antenna    :", num_antennas)
    print("Subcarrier :", num_subcarriers)
    print("Time       :", num_time)

    if num_subcarriers != 64:
        print(
            "Warning: Code is designed for 64 subcarriers."
        )

    t = np.arange(num_time)

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    base_name = Path(
        npy_file
    ).stem

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    group_size = 16

    # ==================================================
    # Each antenna
    # ==================================================
    for ant_idx in range(
        num_antennas
    ):

        # ==================================================
        # 4 figures
        # ==================================================
        for group in range(4):

            start = (
                group *
                group_size
            )

            end = min(
                start +
                group_size,
                num_subcarriers,
            )

            fig, axes = plt.subplots(
                group_size,
                1,
                figsize=(18, 22),
                sharex=True,
            )

            if group_size == 1:
                axes = [axes]

            # ==============================================
            # Plot each subcarrier
            # ==============================================
            for row, sc in enumerate(
                range(start, end)
            ):

                ax = axes[row]

                ax.plot(
                    t,
                    data[
                        ant_idx,
                        sc,
                        :
                    ],
                    color="blue",
                    linewidth=0.8,
                )

                ax.set_ylabel(
                    subcarrier_label(sc),
                    rotation=0,
                    fontsize=9,
                    labelpad=45,
                )

                ax.grid(True)

            axes[-1].set_xlabel(
                "Time Index"
            )

            title = (
                f"Antenna {ant_idx}"
                f" | "
                f"Subcarrier "
                f"{start:02d} ~ {end-1:02d}"
            )

            if ENABLE_PCHIP:
                title += " (PCHIP)"

            fig.suptitle(
                title,
                fontsize=16,
                fontweight="bold",
            )

            plt.tight_layout(
                rect=[0, 0, 1, 0.98]
            )

            suffix = (
                "_pchip"
                if ENABLE_PCHIP
                else "_raw"
            )

            save_path = os.path.join(
                output_dir,
                f"{base_name}"
                f"{suffix}"
                f"_ant{ant_idx}"
                f"_part{group+1}"
                f"_{timestamp}.png",
            )

            plt.savefig(
                save_path,
                dpi=300,
                bbox_inches="tight",
            )

            plt.close(fig)

            print(
                f"Saved: {save_path}"
            )


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":

    npy_file = (
        r"D:\workspace\HUST\Personal\WiFi-Sensing\Code\csi-extractor\data\sshar_amp_interp\room_01\asus\rx_01\subject_04\amp_act01_pos04_dir00_rep01.npy"
    )

    print(
        f"Processing: {npy_file}"
    )

    plot_all_subcarriers(
        npy_file=npy_file,
        output_dir="image",
    )