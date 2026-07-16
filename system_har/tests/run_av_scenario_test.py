# import threading
# import time

# from app.services.config_service import ConfigService
# from app.services.session_service import SessionService
# from app.services.scenario_service import ScenarioService
# from app.services.audio_cue_service import AudioCueService
# from app.adapters.webcam_adapter import WebcamAdapter
# from app.services.video_service import VideoService


# def record_video(session_dir, camera_cfg, stop_event):
#     cam = WebcamAdapter(camera_index=camera_cfg["camera_index"])

#     video = VideoService(
#         session_dir=session_dir,
#         fps=camera_cfg["fps"],
#         width=camera_cfg["width"],
#         height=camera_cfg["height"]
#     )

#     frame_interval = 1.0 / camera_cfg["fps"]

#     try:
#         cam.open()
#         video.open()

#         print("Video recording started")

#         while not stop_event.is_set():
#             loop_start = time.time()

#             ok, frame = cam.read_frame()

#             if ok:
#                 video.write_frame(frame)
#             else:
#                 print("Không đọc được frame từ camera")

#             elapsed = time.time() - loop_start
#             sleep_time = max(0, frame_interval - elapsed)
#             time.sleep(sleep_time)

#     finally:
#         cam.close()
#         video.close()
#         print("Video recording stopped")


# def main():
#     # 1. Đọc config
#     session_config = ConfigService().load_session_config()

#     # 2. Tạo action plan trước
#     action_plan = ScenarioService().build_action_plan(
#         scenario_name=session_config["scenario"],
#         repeat_count=session_config["repeat_count"],
#         position_id=session_config["position_id"]
#     )

#     print("Total actions:", len(action_plan))

#     # 3. Tạo session
#     session_info, session_dir = SessionService().create_session(session_config)

#     print("Session ID:", session_info["session_id"])
#     print("Session dir:", session_dir)

#     # 4. Chuẩn bị camera
#     camera_cfg = session_config["devices"]["camera"]
#     stop_video_event = threading.Event()

#     video_thread = None

#     if camera_cfg.get("enabled", True):
#         video_thread = threading.Thread(
#             target=record_video,
#             args=(session_dir, camera_cfg, stop_video_event),
#             daemon=True
#         )

#         video_thread.start()
#         time.sleep(1.0)

#     # 5. Chạy audio cue
#     try:
#         audio = AudioCueService(session_dir)
#         audio.run_action_plan(action_plan)

#     finally:
#         # 6. Dừng video sau khi audio xong
#         stop_video_event.set()

#         if video_thread:
#             video_thread.join()

#     print("Done")
#     print("Video:", session_dir / "video.mp4")
#     print("Video index:", session_dir / "video_index.csv")
#     print("Action events:", session_dir / "action_events.csv")


# if __name__ == "__main__":
#     main()

# bản sửa 2 chưa được

# import threading
# import time

# from app.core.time_utils import perf_now
# from app.services.config_service import ConfigService
# from app.services.session_service import SessionService
# from app.services.scenario_service import ScenarioService
# from app.services.audio_cue_service import AudioCueService
# from app.adapters.webcam_adapter import WebcamAdapter
# from app.services.video_service import VideoService


# def record_video(session_dir, camera_cfg, stop_event, session_t0):
#     cam = WebcamAdapter(camera_index=camera_cfg["camera_index"])

#     video = VideoService(
#         session_dir=session_dir,
#         fps=camera_cfg["fps"],
#         width=camera_cfg["width"],
#         height=camera_cfg["height"],
#         session_t0=session_t0
#     )

#     frame_interval = 1.0 / camera_cfg["fps"]

#     try:
#         cam.open()

#         # Warm-up camera: bỏ vài frame đầu để camera ổn định
#         for _ in range(10):
#             cam.read_frame()
#             time.sleep(0.03)

#         video.open()

#         print("Video recording started")

#         while not stop_event.is_set():
#             loop_start = time.time()

#             ok, frame = cam.read_frame()

#             if ok:
#                 video.write_frame(frame)
#             else:
#                 print("Không đọc được frame từ camera")

#             elapsed = time.time() - loop_start
#             time.sleep(max(0, frame_interval - elapsed))

#     finally:
#         cam.close()
#         video.close()
#         print("Video recording stopped")


# def main():
#     session_config = ConfigService().load_session_config()

#     action_plan = ScenarioService().build_action_plan(
#         scenario_name=session_config["scenario"],
#         repeat_count=session_config["repeat_count"],
#         position_id=session_config["position_id"]
#     )

#     print("Total actions:", len(action_plan))

#     session_info, session_dir = SessionService().create_session(session_config)

#     print("Session ID:", session_info["session_id"])
#     print("Session dir:", session_dir)

#     camera_cfg = session_config["devices"]["camera"]
#     stop_video_event = threading.Event()

#     # Mốc thời gian chung cho cả audio, video, CSI sau này
#     session_t0 = perf_now()

#     video_thread = None

#     if camera_cfg.get("enabled", True):
#         video_thread = threading.Thread(
#             target=record_video,
#             args=(session_dir, camera_cfg, stop_video_event, session_t0),
#             daemon=True
#         )
#         video_thread.start()

#         # Đợi camera mở và ghi được frame đầu tiên
#         time.sleep(1.5)

#     try:
#         audio = AudioCueService(session_dir)
#         audio.run_action_plan(action_plan, session_t0=session_t0)

#     finally:
#         stop_video_event.set()

#         if video_thread:
#             video_thread.join()

#     print("Done")
#     print("Video:", session_dir / "video.mp4")
#     print("Video index:", session_dir / "video_index.csv")
#     print("Action events:", session_dir / "action_events.csv")


# if __name__ == "__main__":
#     main()

import threading
import time

from app.core.time_utils import perf_now
from app.services.config_service import ConfigService
from app.services.session_service import SessionService
from app.services.scenario_service import ScenarioService
from app.services.audio_cue_service import AudioCueService
from app.adapters.webcam_adapter import WebcamAdapter
from app.services.video_service import VideoService


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
            loop_start = time.time()

            ok, frame = cam.read_frame()

            if ok:
                video.write_frame(frame)

                if not first_frame_written:
                    first_frame_written = True
                    video_ready_event.set()
                    print("Video ready: first frame written")
            else:
                print("Không đọc được frame từ camera")

            elapsed = time.time() - loop_start
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

    session_t0 = perf_now()

    video_thread = None

    if camera_cfg.get("enabled", True):
        video_thread = threading.Thread(
            target=record_video,
            args=(
                session_dir,
                camera_cfg,
                stop_video_event,
                video_ready_event,
                session_t0
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

    print("Done")
    print("Video:", session_dir / "video.mp4")
    print("Video index:", session_dir / "video_index.csv")
    print("Action events:", session_dir / "action_events.csv")


if __name__ == "__main__":
    main()