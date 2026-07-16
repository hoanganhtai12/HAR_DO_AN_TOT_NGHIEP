import json
import os

CONFIG_FILE = "csi_config.json"


def load_config() -> dict:
    if not os.path.exists(CONFIG_FILE):
        return {}

    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def get_database_path() -> str:
    config_data = load_config()

    saved_path = config_data.get("database_root", "")

    if saved_path and os.path.exists(saved_path):
        print("\n[*] Đã tải cấu hình Database gốc từ lần chạy trước:")
        print(f" -> {saved_path}")

        choice = input(
            "Bấm [Enter] để sử dụng, hoặc gõ 'n' để đổi đường dẫn mới: "
        ).strip().lower()

        if choice != "n":
            return saved_path

    while True:
        print("\n" + "=" * 50)

        new_path = input(
            " [Cài đặt] Nhập đường dẫn thư mục Database gốc: "
        ).strip().strip('"\'')
        
        if os.path.exists(new_path):

            config_data["database_root"] = new_path

            try:
                with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                    json.dump(
                        config_data,
                        f,
                        indent=4,
                        ensure_ascii=False
                    )

                print(" -> [Thành công] Đã lưu đường dẫn vĩnh viễn!")

            except Exception as e:
                print(
                    f" -> [Cảnh báo] Không thể lưu file cấu hình: {e}"
                )

            return new_path

        print(" -> [Lỗi] Đường dẫn không tồn tại.")


def get_data_structure() -> dict:
    config_data = load_config()

    data_structure = config_data.get("data_structure")

    if data_structure is None:
        raise RuntimeError(
            "'data_structure' không tồn tại trong csi_config.json"
        )

    return data_structure


def get_debug() -> bool:
    """
    Đọc cờ debug từ file cấu hình.
    Mặc định trả về False nếu không tồn tại.
    """

    config_data = load_config()

    return bool(
        config_data.get("debug", False)
    )