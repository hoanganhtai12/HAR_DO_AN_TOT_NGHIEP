import os
import numpy as np

from .config import get_database_path
from .folder_loader import (
    resolve_data_folder,
    search_and_select_folder
)
from .events import get_time_from_event
from .packet_sync import (
    find_first_packet_fast,
    sync_and_cut_3_files,
    normalize_3_files,
    read_first_header
)

from .csi_decoder import extract_csi_matrix
from .plotter import plot_csi_data


# =========================================================
# CONFIG
# =========================================================

NUMBER_PACKET_TO_TEST = 1000

DEVICE_CONFIG = {
    'esp': {
        'packet_size': 144,
    },
    'asus': {
        'packet_size': 1044,
    }
}


# =========================================================
# MAIN
# =========================================================

def main():

    # =====================================================
    # select device
    # =====================================================

    device_map = {
        '1': 'esp',
        '2': 'asus',
        'esp': 'esp',
        'asus': 'asus'
    }

    while True:
        dev_choice = input(
            "1. Chọn loại thiết bị: "
            "1) esp32  "
            "2) asus : "
        ).strip().lower()
        dev_type = device_map.get(dev_choice)
        if dev_type:
            break

        print(
            "   -> Thiết bị không hợp lệ, "
            "vui lòng nhập 1 hoặc 2."
        )

    pkt_size = DEVICE_CONFIG[dev_type]['packet_size']

    # =====================================================
    # select database folder
    # =====================================================

    db_root = get_database_path()

    path = search_and_select_folder(db_root)

    if path is None:
        input("\nNhấn Enter để thoát chương trình...")
        return

    original_path = path
    path = resolve_data_folder(path)
    if path != original_path:
        print(
            f"[Chú ý] "
            f"Tự động sử dụng thư mục dữ liệu thực: {path}"
        )

    # =====================================================
    # plot mode
    # =====================================================

    print("\n2. Chọn chế độ vẽ:")
    print("   1) Vẽ FULL FILE")
    print("   2) Vẽ đoạn N packets")

    plot_mode = input(
        "Chọn 1 hoặc 2 [1]: "
    ).strip() or '1'

    if plot_mode not in ['1', '2']:
        print(
            "   -> Lựa chọn không hợp lệ, "
            "mặc định FULL FILE."
        )
        plot_mode = '1'

    # =====================================================
    # FULL FILE MODE
    # =====================================================

    if plot_mode == '1':
        number_packet = 0
        print(
            "\n[FULL FILE MODE] "
            "Sẽ xử lý toàn bộ file."
        )

        action = None
        repeat = None

    # =====================================================
    # N PACKET MODE
    # =====================================================

    else:
        number_packet = NUMBER_PACKET_TO_TEST
        print(
            f"\n[N PACKET MODE] "
            f"Sẽ xử lý {number_packet} packets."
        )

        action_map = {
            '1': 'sitdown',
            '2': 'standup',
            '3': 'pickup',
            '4': 'fall',
            '5': 'liestill',
            '6': 'walk',
            '7': 'run',
            '8': 'emptyroom'
        }

        print("\n3. Chọn hành động:")
        print("   1) sitdown        2) standup")
        print("   3) pickup         4) fall")
        print("   5) liesstill      6) walk")
        print("   7) run            8) emptyroom")

        while True:
            action_choice = input(
                "Chọn 1-8: "
            ).strip()

            action = action_map.get(action_choice)
            if action:
                break
            print(
                "   -> Lựa chọn không hợp lệ, "
                "vui lòng chọn 1 đến 8."
            )
        # =================================================
        # repeat index
        # =================================================
        while True:
            try:
                repeat = int(
                    input(
                        "4. Nhập chỉ số lần lặp thứ "
                        "(VD: 1, 2, 3...): "
                    ).strip()
                )
                break
            except ValueError:
                print(
                    "   -> Lỗi: "
                    "Vui lòng nhập một số nguyên."
                )
    # =====================================================
    # force plot mode
    # =====================================================
    is_save = False

    # =====================================================
    # info
    # =====================================================
    print("\n" + "-" * 60)
    print(" ĐANG CHẠY CHỨC NĂNG: EXARRAY ")
    print("-" * 60)
    output_dir = "Cut_Data"

    # =====================================================
    # build output filename
    # =====================================================
    folder_name = os.path.basename(
        os.path.normpath(path)
    )

    parts = folder_name.split('_')
    if len(parts) >= 7:
        prefix_5 = "_".join(parts[:5])
        time_suffix = "_".join(parts[-2:])
    else:
        prefix_5 = "1_1_1_1_1"
        time_suffix = "0000_000000"
    if number_packet == 0:
        base_name = f"{prefix_5}_FULL_{time_suffix}"
    else:
        base_name = (
            f"{prefix_5}_"
            f"{repeat}_"
            f"{action}_"
            f"{time_suffix}"
        )

    base_out_name = f"{base_name}.bin"
    print(
        f"[*] Tên file đồng bộ: "
        f"{base_out_name}"
    )

    # =====================================================
    # paths
    # =====================================================
    event_file = os.path.join(
        path,
        "action_events.csv"
    )
    f1 = os.path.join(
        path,
        f"raw_{dev_type}1.bin"
    )
    f2 = os.path.join(
        path,
        f"raw_{dev_type}2.bin"
    )
    f3 = os.path.join(
        path,
        f"raw_{dev_type}3.bin"
    )

    # =====================================================
    # normalize all files
    # =====================================================
    print("\n" + "=" * 60)
    print(" NORMALIZE 3 FILES ")
    print("=" * 60)

    normalized_buffers = normalize_3_files(
        f1,
        f2,
        f3,
        pkt_size
    )

    if normalized_buffers is None:
        print("[FAIL] normalize failed")
        input("\nNhấn Enter để thoát...")
        return

    # =====================================================
    # determine start packet
    # =====================================================

    # -----------------------------------------------------
    # FULL FILE MODE
    # -----------------------------------------------------
    if number_packet == 0:
        found_seq, found_ts = read_first_header(
            f1,
            pkt_size
        )

        if found_seq is None:
            print(
                "[FAIL] "
                "Cannot read first packet."
            )
            return

        print(
            f"[FULL FILE MODE] "
            f"start_seq={found_seq} "
            f"start_ts={found_ts}"
        )

    # -----------------------------------------------------
    # N PACKET MODE
    # -----------------------------------------------------

    else:
        time_input = get_time_from_event(
            event_file,
            action,
            repeat
        )

        if time_input is None:
            print(
                "\n[Kết thúc] "
                "Dừng chương trình do không "
                "tìm thấy mốc thời gian."
            )
            input("Nhấn Enter để thoát...")

            return

        found_seq, found_ts = find_first_packet_fast(
            f1,
            time_input,
            pkt_size
        )

        if found_seq is None:
            print(
                "\n[Kết thúc] "
                "Không tìm thấy packet neo."
            )
            input("Nhấn Enter để thoát...")
            return

    # =====================================================
    # sync and cut
    # =====================================================
    print("\n" + "=" * 60)
    print(" SYNC AND CUT ")
    print("=" * 60)

    ram_buffers = sync_and_cut_3_files(
        normalized_buffers,
        found_seq,
        found_ts,
        pkt_size,
        dev_type,
        base_out_name,
        output_dir,
        number_packet,
        is_save=is_save
    )

    # =====================================================
    # extract CSI
    # =====================================================

    csi_matrices = extract_csi_matrix(
        ram_buffers,
        dev_type,
        from_buffer=True
    )

    # =====================================================
    # plot option
    # =====================================================

    print("\n" + "=" * 60)
    print(
        " TÙY CHỌN VẼ DỮ LIỆU CSI "
        "(ĐỒNG BỘ MULTI-DEVICE)"
    )
    print("=" * 60)

    preprocess_choice = input(
        "\n1. Có muốn vẽ thêm "
        "tiền xử lý không? (y/N): "
    ).strip().lower()

    plot_preprocess = (preprocess_choice != 'n')

    # =====================================================
    # subcarrier option
    # =====================================================
    print("\n2. Vẽ subcarrier:")
    print(
        "   - Để trống hoặc nhập Enter: "
        "vẽ tất cả (64 subcarrier)"
    )
    print(
        "   - Nhập số 0-63: "
        "vẽ chỉ 1 subcarrier"
    )
    subcarrier_input = input(
        "Lựa chọn (Enter=all, 0-63=single): "
    ).strip()

    if subcarrier_input == "":
        plot_single_sub = None
    else:
        try:
            sub_idx = int(subcarrier_input)
            if 0 <= sub_idx < 64:
                plot_single_sub = sub_idx
            else:
                print(
                    f"   -> Giá trị {sub_idx} "
                    f"ngoài phạm vi. "
                    f"Sẽ vẽ tất cả."
                )
                plot_single_sub = None
        except ValueError:
            print(
                "   -> Lựa chọn không hợp lệ. "
                "Sẽ vẽ tất cả."
            )
            plot_single_sub = None

    # =====================================================
    # build devices list
    # =====================================================
    devices_list = []
    if csi_matrices:
        for i, csi_data in enumerate(csi_matrices):
            if csi_data is not None:
                devices_list.append({
                    'name': f'{dev_type.upper()}_{i + 1}',
                    'dev_type': dev_type,
                    'csi_matrices': [csi_data]
                })

    if not devices_list:
        print(
            "\n[Lỗi] "
            "Không có dữ liệu CSI hợp lệ."
        )
        return

    print(
        f"\n[*] Đang gửi dữ liệu "
        f"của {len(devices_list)} "
        f"thiết bị sang Plotter..."
    )

    plot_csi_data(
        base_name,
        devices_list,
        plot_preprocess=plot_preprocess,
        plot_single_sub=plot_single_sub
    )

    # =====================================================
    # log shape
    # =====================================================
    if (
        csi_matrices
        and len(csi_matrices) > 0
        and csi_matrices[0] is not None
    ):

        amp = csi_matrices[0]['amplitude']
        if isinstance(amp, np.ndarray):
            shape_str = str(amp.shape)
        else:
            shape_str = (
                f"{len(amp)} antennas, "
                f"shape: {amp[0].shape}"
            )
        print(
            f"\n[INFO] Rx1 amplitude shape: "
            f"{shape_str}"
        )


# =========================================================
# ENTRY
# =========================================================

if __name__ == "__main__":
    main()