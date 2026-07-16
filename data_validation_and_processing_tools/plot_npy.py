import os
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt


def plot_subcarrier(
    npy_file,
    subcarrier_idx=15,
    output_dir="image"
):
    """
    npy shape:
        (Antenna, Subcarrier, Time)
    """

    data = np.load(npy_file)

    print("Shape =", data.shape)

    if data.ndim != 3:
        raise ValueError(
            f"Expected shape (Antenna, Subcarrier, Time), got {data.shape}"
        )

    num_antennas, num_subcarriers, num_time = data.shape

    if not (0 <= subcarrier_idx < num_subcarriers):
        raise ValueError(
            f"subcarrier_idx={subcarrier_idx} out of range "
            f"(0~{num_subcarriers - 1})"
        )

    fig, axes = plt.subplots(
        num_antennas,
        1,
        figsize=(12, 2.5 * num_antennas),
        sharex=True
    )

    if num_antennas == 1:
        axes = [axes]

    t = np.arange(num_time)

    for ant_idx in range(num_antennas):

        signal = data[ant_idx, subcarrier_idx, :]

        axes[ant_idx].plot(t, signal)

        axes[ant_idx].set_ylabel(
            f"Ant {ant_idx}"
        )

        axes[ant_idx].grid(True)

    axes[-1].set_xlabel("Time Index")

    fig.suptitle(
        f"Subcarrier {subcarrier_idx}"
    )

    plt.tight_layout()

    # ==========================
    # Tạo thư mục image
    # ==========================
    os.makedirs(output_dir, exist_ok=True)

    # ==========================
    # Tạo tên file:
    # amp_act01_pos02_dir00_rep01_YYYYMMDD_HHMMSS.png
    # ==========================
    base_name = Path(npy_file).stem

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    filename = (
        f"{base_name}_{timestamp}.png"
    )

    save_path = os.path.join(
        output_dir,
        filename
    )

    # ==========================
    # Lưu ảnh
    # ==========================
    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Saved image: {save_path}")


if __name__ == "__main__":

    npy_file = (
        r"D:\workspace\HUST\Personal\WiFi-Sensing\Code\csi-extractor\data"
        r"\sshar\room_01\asus\rx_00\subject_07"
        r"\amp_act01_pos02_dir00_rep01.npy"
    )

    plot_subcarrier(
        npy_file=npy_file,
        subcarrier_idx=15,
        output_dir="image"
    )