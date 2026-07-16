import os


def get_optional_input(prompt_text: str) -> str:
    val = input(prompt_text).strip()
    return val if val != "" else None


def resolve_data_folder(path: str) -> str:
    """Resolve nested dataset folder when the selected folder contains a single subfolder."""
    if os.path.exists(os.path.join(path, "action_events.csv")):
        return path

    if not os.path.isdir(path):
        return path

    candidates = []
    for entry in os.listdir(path):
        child = os.path.join(path, entry)
        if os.path.isdir(child) and os.path.exists(os.path.join(child, "action_events.csv")):
            candidates.append(child)

    if len(candidates) == 1:
        return candidates[0]

    return path


def search_and_select_folder(db_root: str) -> str:
    if not os.path.exists(db_root):
        print(f"[Lỗi] Không tìm thấy thư mục Database gốc: {db_root}")
        return None

    print("\n" + "=" * 50)
    print(" BỘ LỌC TÌM KIẾM DỮ LIỆU DATABASE")
    print(" (Mẹo: Bấm Enter để bỏ qua tiêu chí nếu không cần thiết)")
    print("=" * 50)

    room = get_optional_input(" -> Chỉ số phòng (room)      : ")
    setup = get_optional_input(" -> Chỉ số setup (setup)     : ")
    session = get_optional_input(" -> Chỉ số phiên (session)   : ")
    user_idx = get_optional_input(" -> Chỉ số người (user)      : ")
    pos = get_optional_input(" -> Chỉ số vị trí (pos)      : ")

    criteria = [room, setup, session, user_idx, pos]
    matched_folders = []

    for folder_name in os.listdir(db_root):
        folder_path = os.path.join(db_root, folder_name)
        if not os.path.isdir(folder_path):
            continue
        parts = folder_name.split('_')
        if len(parts) < 5:
            continue

        is_match = True
        for i in range(5):
            if criteria[i] is not None and criteria[i] != parts[i]:
                is_match = False
                break
        if is_match:
            matched_folders.append(folder_name)

    if not matched_folders:
        print("\n[Thất bại] Không tìm thấy folder nào khớp.")
        return None

    if len(matched_folders) == 1:
        print(f"\n[Thành công] Tìm thấy duy nhất 1 thư mục khớp: {matched_folders[0]}")
        return os.path.join(db_root, matched_folders[0])

    print(f"\n[Kết quả] Tìm thấy {len(matched_folders)} thư mục phù hợp:")
    for idx, f_name in enumerate(matched_folders):
        print(f"  {idx + 1}. {f_name}")

    while True:
        choice = input(f"\n -> Chọn số thứ tự thư mục muốn xử lý (1 - {len(matched_folders)}): ").strip()
        try:
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(matched_folders):
                return os.path.join(db_root, matched_folders[choice_idx])
            print(" [!] Lựa chọn nằm ngoài phạm vi danh sách.")
        except ValueError:
            print(" [!] Vui lòng nhập một số nguyên hợp lệ.")
