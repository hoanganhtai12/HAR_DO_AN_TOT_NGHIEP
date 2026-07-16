import os
import pandas as pd
from datetime import datetime

from .config import get_debug

DEBUG = get_debug()

def get_time_from_event(event_file: str, action_name: str, repeat_idx: int) -> int:
    if not os.path.exists(event_file):
        print(f"[Lỗi] không tìm thấy file event: {event_file}")
        return None
    try:
        df = pd.read_csv(event_file) if event_file.endswith('.csv') else pd.read_excel(event_file)
        required_columns = ['action_name', 'repeat_index', 'start_unix_us']
        for col in required_columns:
            if col not in df.columns:
                print(f"[Lỗi] File event thiếu cột: '{col}'")
                return None

        condition = (df['action_name'] == action_name) & (df['repeat_index'] == repeat_idx)
        filtered_df = df[condition]
        if filtered_df.empty:
            print(f"[Cảnh báo] Không tìm thấy '{action_name}' lần lặp {repeat_idx}.")
            return None

        if DEBUG: print(f"Event: {action_name} - lần lặp: {repeat_idx} - bắt đầu quanh:{int(float(filtered_df['start_unix_us'].iloc[0]))} = {datetime.fromtimestamp(int(float(filtered_df['start_unix_us'].iloc[0])) / 1_000_000)}")
        return int(float(filtered_df['start_unix_us'].iloc[0]))
    except Exception as e:
        print(f"[Lỗi xử lý Dataframe] {e}")
        return None
