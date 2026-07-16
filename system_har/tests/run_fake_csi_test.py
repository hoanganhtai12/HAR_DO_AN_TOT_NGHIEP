# main.py

import threading
import time
from app.services.config_service import ConfigService
from app.services.session_service import SessionService
from app.services.scenario_service import ScenarioService
from app.services.audio_cue_service import AudioCueService
from app.adapters.webcam_adapter import WebcamAdapter
from app.services.video_service import VideoService
from app.services.csi_service import CsiService  # Import CsiService
from app.core.time_utils import perf_now  # Import perf_now to get session start time

def record_video(session_dir, camera_cfg, stop_event, video_ready_event, session_t0):
    cam = WebcamAdapter(camera_index=camera_cfg["camera_index"])

    video = VideoService(
        session_dir=session_dir,
        fps=camera_cfg["fps"],
        width=camera_cfg["width"],
        height=camera_cfg["height"],
        session_t0=session_t0
    )

    frame_interval = 1.0 / camera_cfg["fps"]
    first_frame_written = False

    try:
        cam.open()

        # Bỏ các frame đầu để camera ổn định
        for _ in range(20):
            cam.read_frame()
            time.sleep(0.03)

        video.open()

        print("Video recording started")

        while not stop_event.is_set():
            loop_start = time.perf_counter()  # Sử dụng perf_counter thay vì time.time()

            ok, frame = cam.read_frame()

            if ok:
                video.write_frame(frame)

                if not first_frame_written:
                    first_frame_written = True
                    video_ready_event.set()
                    print("Video ready: first frame written")
            else:
                print("Không đọc được frame từ camera")

            elapsed = time.perf_counter() - loop_start  # Sử dụng perf_counter thay vì time.time()
            time.sleep(max(0, frame_interval - elapsed))

    finally:
        cam.close()
        video.close()
        print("Video recording stopped")


def main():
    session_config = ConfigService().load_session_config()

    action_plan = ScenarioService().build_action_plan(
        scenario_name=session_config["scenario"],
        repeat_count=session_config["repeat_count"],
        position_id=session_config["position_id"]
    )

    print("Total actions:", len(action_plan))

    session_info, session_dir = SessionService().create_session(session_config)

    print("Session ID:", session_info["session_id"])
    print("Session dir:", session_dir)

    camera_cfg = session_config["devices"]["camera"]

    stop_video_event = threading.Event()
    video_ready_event = threading.Event()

    session_t0 = perf_now()  # Đảm bảo lấy thời gian bắt đầu khi bắt đầu thu thập

    # Khởi tạo CsiService và bắt đầu thu thập dữ liệu CSI
    csi_service = CsiService(session_dir, session_t0)  # Truyền session_t0 vào CsiService
    csi_service.start_csi_collection()

    video_thread = None

    if camera_cfg.get("enabled", True):
        video_thread = threading.Thread(
            target=record_video,
            args=(
                session_dir,
                camera_cfg,
                stop_video_event,
                video_ready_event,
                session_t0  # Truyền session_t0 vào video recording
            ),
            daemon=True
        )

        video_thread.start()

        print("Waiting for video ready...")

        if not video_ready_event.wait(timeout=10):
            stop_video_event.set()

            if video_thread:
                video_thread.join()

            raise RuntimeError("Camera chưa ghi được frame đầu tiên sau 10 giây")

        # Cho video ghi thêm một đoạn ngắn trước khi phát audio
        time.sleep(0.5)

    try:
        audio = AudioCueService(session_dir)
        audio.run_action_plan(action_plan, session_t0=session_t0)

    finally:
        stop_video_event.set()

        if video_thread:
            video_thread.join()

        # Dừng thu thập CSI khi hoàn thành
        csi_service.stop_csi_collection()

    print("Done")
    print("Video:", session_dir / "video.mp4")
    print("Video index:", session_dir / "video_index.csv")
    print("Action events:", session_dir / "action_events.csv")
    print("CSI data:", session_dir / "raw_eth.csv")


if __name__ == "__main__":
    main()