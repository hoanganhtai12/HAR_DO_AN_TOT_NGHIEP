import os
import numpy as np

from .config import get_database_path
from .events import get_time_from_event
from .packet_sync import (
    find_first_packet_fast,
    sync_and_cut_3_files,
    normalize_3_files
)

from .csi_decoder import extract_csi_matrix
from .config import get_data_structure, get_debug

DEBUG = get_debug()
# =========================================================
# CONFIG
# =========================================================
SUBCARRIER_NUM = 64
PACKET_COUNT = 1000
DEVICE_CONFIG = {
    'esp': {
        'packet_size': 144,
    },
    'asus': {
        'packet_size': 1044,
    }
}

# So lan repeat toi da o truong hop dir = 0, dir = 1
MAX_REPEAT = [10, 5]    

# DEVICE_MAP = {
#     1: 'esp',
#     2: 'asus',
# }

DEVICE_MAP = {
    2: 'asus'
}

DATA_STRUCTURE = None

def setup_db_struct(data_structure):
    global DATA_STRUCTURE
    DATA_STRUCTURE = data_structure

def get_action_name_by_idx(idx):
    global DATA_STRUCTURE

    for action_name, action_info in DATA_STRUCTURE.items():
        if action_info["idx"] == idx:
            return action_name

    return None

def get_folder(db_root: str, dev_type:str, action_name: str, room: int, setup: int, session: int, user_idx: int, pos: int) -> str:
    criteria = [str(room), str(setup), str(session), str(user_idx), str(pos)]
    matched_folders = []

    for folder_name in os.listdir(db_root):
        folder_path = os.path.join(db_root, folder_name)
        if not os.path.isdir(folder_path):
            continue
        # ví dụ 1 folder name: 1_1_1_6_6_5_run_0618_163523
        parts = folder_name.split('_')
        if len(parts) < 7:
            continue

        is_match = True
        for i in range(5):
            if criteria[i] is not None and criteria[i] != parts[i]:
                is_match = False
                break
        
        if action_name not in parts[5:]:
            is_match = False
            
        if is_match:
            if os.path.exists(os.path.join(folder_path, "action_events.csv")) \
                and os.path.exists(os.path.join(folder_path, f"raw_{dev_type}1.bin")) \
                and os.path.exists(os.path.join(folder_path, f"raw_{dev_type}2.bin")) \
                and os.path.exists(os.path.join(folder_path, f"raw_{dev_type}3.bin")):
                matched_folders.append(folder_name)

    if not matched_folders:
        print(
            f"[ERROR]   -> Không tìm thư mục khớp với:"
            f"room: {room}, setup: {setup}, session: {session}, user_idx: {user_idx}, pos: {pos}"
        )
        return None

    print(f"---------------[Thành công] Tìm thấy {len(matched_folders)} thư mục khớp. Lựa chọn thư mục: {matched_folders[0]}")
    return os.path.join(db_root, matched_folders[0])

def query_data_by_options(db_root:str, dev_idx, action_idx, repeat_idx, room_idx, setup_idx, session_idx, user_idx, pos_idx):
    # =====================================================
    # select query options
    # =====================================================
    dev_type = DEVICE_MAP.get(dev_idx)
    if dev_type == None:
        print(
            "[ERROR]   -> Thiết bị không hợp lệ, "
            "vui lòng chọn 1 hoặc 2"
        )
        return None

    action_name = get_action_name_by_idx(action_idx)
    if action_name == None:
        print(
        "[ERROR]   -> Hành động không hợp lệ, "
        "vui lòng chọn 1 đến 8."
        )
        return None

    pkt_size = DEVICE_CONFIG[dev_type]['packet_size']

    path = get_folder(db_root, dev_type, action_name, room_idx, setup_idx, session_idx, user_idx, pos_idx)
    if path is None:
        return None

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
    normalized_buffers = normalize_3_files(
        f1,
        f2,
        f3,
        pkt_size
    )

    if normalized_buffers is None:
        print("[FAIL] normalize failed")
        input("\nNhấn Enter để thoát...")
        return None

    # =====================================================
    # determine start packet
    # =====================================================
    time_input = get_time_from_event(
        event_file,
        action_name,
        repeat_idx
    )

    if time_input is None:
        print(
            "\n[ERROR] "
            "Dừng chương trình do không "
            "tìm thấy mốc thời gian."
        )
        return None

    found_seq, found_ts = find_first_packet_fast(
        f1,
        time_input,
        pkt_size
    )

    if found_seq is None:
        print(
            "\n[ERROR] "
            "Không tìm thấy packet neo."
        )
        return None

    # =====================================================
    # sync and cut
    # =====================================================
    if DEBUG: print("\n" + "=" * 60)
    if DEBUG: print(" SYNC AND CUT ")
    if DEBUG: print("=" * 60)

    ram_buffers = sync_and_cut_3_files(
        normalized_buffers,
        found_seq,
        found_ts,
        pkt_size,
        dev_type,
        "",
        "",
        PACKET_COUNT,
        is_save=False
    )

    # =====================================================
    # extract CSI
    # =====================================================

    csi_matrices = extract_csi_matrix(
        ram_buffers,
        dev_type,
        from_buffer=True
    )

    # hàm extrict_csi_matric có code như sau:
    # ...
    # result = []
    # ...
    # results.append({
    #             'rx_index': rx_idx,
    #             'timestamp': data['timestamp'],
    #             'amplitude': amplitude,
    #             'phase': phase,
    #             'num_antennas': 1 hoặc 4
    #         })
    # ...
    # return result

    return csi_matrices


def export_numpy():
    global DATA_STRUCTURE
    DATA_STRUCTURE = get_data_structure()

    root_dir = get_database_path()
    if not os.path.exists(root_dir):
        print(
            f"[ERROR]   -> Không tìm thấy đường dẫn: {root_dir}"
        )
        return None
    
    for action_name, action_info in DATA_STRUCTURE.items():
        print(f"[INFO] Trich xuat data cho action: {action_name}")

        action_idx = action_info['idx']

        for room_idx in action_info['room']:
            for session_idx in action_info['session']:
                for user_idx in action_info['user']:
                    for pos_idx in action_info['pos']:
                        for repeat_idx in action_info['repeat']:
                            for setup_idx in action_info['setup']:
                                for devtype_idx, devtype_name in DEVICE_MAP.items():
                                    csi_data = query_data_by_options(
                                        root_dir,
                                        devtype_idx,
                                        action_idx,
                                        repeat_idx,
                                        room_idx,
                                        setup_idx,
                                        session_idx,
                                        user_idx,
                                        pos_idx
                                    )

                                    if csi_data is None:
                                        continue


                                    if action_info['dir'] == 0:
                                        direction = 0
                                        repeat_i = repeat_idx + (setup_idx - 1) * len(action_info['repeat'])

                                        # xu ly giup chap nhan truong hop emptyroom 80 lan lap
                                        pos_i = int((repeat_i - 1) // MAX_REPEAT[action_info['dir']]) + pos_idx
                                        repeat_i = ((repeat_i - 1) % MAX_REPEAT[action_info['dir']]) + 1
                                    else:
                                        direction = setup_idx
                                        repeat_i = repeat_idx
                                        pos_i = pos_idx

                                    # print(f"save as: repeat_idx:{repeat_idx} ---> dir: {direction} pos: {pos_i} repeat: {repeat_i}")

                                    for item in csi_data:

                                        rx_idx = item["rx_index"]

                                        amp = np.asarray(item["amplitude"])
                                        pha = np.asarray(item["phase"])

                                        #
                                        # ESP:
                                        # (Time, Subcarrier)
                                        #
                                        if amp.ndim == 2:
                                            amp = amp[np.newaxis, :, :]

                                        if pha.ndim == 2:
                                            pha = pha[np.newaxis, :, :]

                                        #
                                        # Sau bước trên mọi dữ liệu đều phải là:
                                        # (Antenna, Time, Subcarrier)
                                        #
                                        if amp.ndim != 3:
                                            raise ValueError(
                                                f"Unexpected amplitude shape: {amp.shape}"
                                            )

                                        if pha.ndim != 3:
                                            raise ValueError(
                                                f"Unexpected phase shape: {pha.shape}"
                                            )

                                        #
                                        # (Antenna, Time, Subcarrier)
                                        # -> (Antenna, Subcarrier, Time)
                                        #
                                        amp_tensor = np.transpose(amp, (0, 2, 1))
                                        pha_tensor = np.transpose(pha, (0, 2, 1))

                                        save_dir = (
                                            f"{root_dir}\\sshar\\room_{room_idx:02d}\\"
                                            f"{devtype_name}\\rx_{rx_idx:02d}\\subject_{user_idx:02d}"
                                        )

                                        os.makedirs(save_dir, exist_ok=True)

                                        filename_amp = (
                                            f"{save_dir}\\"
                                            f"amp_act{action_idx:02d}_"
                                            f"pos{pos_i:02d}_"
                                            f"dir{direction:02d}_"
                                            f"rep{repeat_i:02d}.npy"
                                        )

                                        filename_pha = (
                                            f"{save_dir}\\"
                                            f"pha_act{action_idx:02d}_"
                                            f"pos{pos_i:02d}_"
                                            f"dir{direction:02d}_"
                                            f"rep{repeat_i:02d}.npy"
                                        )

                                        np.save(filename_amp, amp_tensor)
                                        np.save(filename_pha, pha_tensor)

                                        if DEBUG: print(
                                            f"[SAVE] "
                                            f"DEV={devtype_name} "
                                            f"RX={rx_idx} "
                                            f"RAW={amp.shape} "
                                            f"SAVED={amp_tensor.shape}"
                                        )