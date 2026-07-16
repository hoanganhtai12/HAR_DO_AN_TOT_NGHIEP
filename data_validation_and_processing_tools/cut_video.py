"""
video_cutter.py (v5)
====================
Cắt chính xác theo elapsed_us từ action_events,
map qua video_index để lấy frame_no, không padding.
"""
import time

start_time = time.time()
import argparse, csv, subprocess, sys, shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def check_ffmpeg_installed():
    """Kiểm tra xem FFmpeg và FFprobe đã được cài đặt và thêm vào PATH chưa."""
    has_ffmpeg = shutil.which("ffmpeg") is not None
    has_ffprobe = shutil.which("ffprobe") is not None
    
    if not has_ffmpeg or not has_ffprobe:
        print("\n" + "="*70)
        print("[LỖI NGHIÊM TRỌNG] Không tìm thấy FFmpeg hoặc FFprobe trên hệ thống!")
        print("Chương trình này bắt buộc phải có bộ công cụ FFmpeg để xử lý video.\n")
        print("CÁCH KHẮC PHỤC TRÊN WINDOWS:")
        print("1. Tải FFmpeg tại: https://github.com/BtbN/FFmpeg-Builds/releases (chọn bản win64-gpl.zip)")
        print("2. Giải nén file vừa tải về.")
        print("3. Tìm thư mục 'bin' (bên trong chứa các file ffmpeg.exe, ffprobe.exe).")
        print("4. Thêm đường dẫn của thư mục 'bin' này vào biến môi trường PATH của Windows.")
        print("5. Khởi động lại Terminal / VS Code và chạy lại chương trình.")
        print("="*70 + "\n")
        sys.exit(1)


def find_video(session_dir: Path) -> Path:
    for ext in ["mp4", "avi", "mkv"]:
        p = session_dir / f"video.{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"Không tìm thấy video: {session_dir}")


def load_action_events(session_dir: Path) -> list[dict]:
    with open(session_dir / "action_events.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_video_index(session_dir: Path) -> list[dict]:
    rows = []
    with open(session_dir / "video_index.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "frame_no":   int(row["frame_no"]),
                "elapsed_us": float(row["elapsed_us"]),
            })
    return rows


def get_video_fps(video_path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    # Thêm shell=True đôi khi giúp Windows tìm thấy file thực thi nếu PATH bị lỗi nhẹ
    output = subprocess.check_output(cmd, text=True, shell=False).strip()
    num, den = output.split("/")
    return float(num) / float(den)


def find_frame(video_index: list[dict], target_us: float, mode: str) -> int:
    lo, hi = 0, len(video_index) - 1
    if mode == "start":
        result = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if video_index[mid]["elapsed_us"] <= target_us:
                result = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return video_index[result]["frame_no"]
    else:  # end
        result = len(video_index) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if video_index[mid]["elapsed_us"] >= target_us:
                result = mid
                hi = mid - 1
            else:
                lo = mid + 1
        return video_index[result]["frame_no"]


def cut_one(
    action:      dict,
    video_path:  Path,
    video_index: list[dict],
    fps:         float,
    segments_dir: Path,
) -> tuple[str, bool, str]:
    idx  = int(action["action_index"])
    rep  = int(action["repeat_index"])
    pos  = action["position_id"]
    name = action["action_name"]
    label = f"[{idx:03d}] {name} rep{rep} pos{pos}"

    frame_start = find_frame(video_index, float(action["start_elapsed_us"]), mode="start")
    frame_end   = find_frame(video_index, float(action["end_elapsed_us"]),   mode="end")

    start_sec = (frame_start - 1) / fps
    end_sec   = (frame_end   - 1) / fps
    duration  = end_sec - start_sec

    out_path = segments_dir / f"{idx:03d}_{name}_rep{rep}_pos{pos}.mp4"

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_sec:.6f}",
        "-i",  str(video_path),
        "-t",  f"{duration:.6f}",
        "-c",  "copy",
        "-avoid_negative_ts", "1",
        str(out_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", shell=False)
    if result.returncode != 0:
        return label, False, result.stderr.splitlines()[-1] if result.stderr else "Unknown FFmpeg Error"

    size_kb = out_path.stat().st_size // 1024
    return label, True, f"frame {frame_start}→{frame_end} | {start_sec:.3f}s→{end_sec:.3f}s | {size_kb} KB"


def cut_video(session_dir: Path, workers: int = 4):
    video_path   = find_video(session_dir)
    actions      = load_action_events(session_dir)
    video_index  = load_video_index(session_dir)
    fps          = get_video_fps(video_path)

    segments_dir = session_dir / "segments"
    segments_dir.mkdir(exist_ok=True)

    print(f"Video   : {video_path}  ({fps:.3f} fps)")
    print(f"Frames  : {video_index[-1]['frame_no']}  |  Actions: {len(actions)}  |  Workers: {workers}\n")

    futures = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for action in actions:
            future = executor.submit(cut_one, action, video_path, video_index, fps, segments_dir)
            futures[future] = int(action["action_index"])

        results = {}
        for future in as_completed(futures):
            results[futures[future]] = future.result()

    ok_count = err_count = 0
    for idx in sorted(results):
        label, ok, msg = results[idx]
        if ok:
            print(f"  OK  {label} | {msg}")
            ok_count += 1
        else:
            print(f"  ERR {label} | {msg}")
            err_count += 1

    print(f"\nHoàn tất → {segments_dir}  ({ok_count} OK, {err_count} lỗi)")


def main():
    # Kiểm tra FFmpeg trước khi thực thi bất kỳ tính toán nào
    check_ffmpeg_installed()
    
    path_input = input("Đường dẫn file: ").strip().strip('"\'')
    if not path_input:
        sys.exit("Đường dẫn file không được để trống!")

    file_path = Path(path_input)
    d = file_path.parent

    if not d.is_dir():
        sys.exit(f"Không tìm thấy folder: {d}")

    workers = 4
    print(f"-> Thư mục xử lý: {d}")
    print(f"-> Số luồng chạy: {workers}")
    
    cut_video(d, workers=workers)


if __name__ == "__main__":
    main()
    elapsed = time.time() - start_time
    print(f"\nTổng thời gian: {elapsed:.2f}s")