import threading
import time

from app.services.session_service import SessionService
from app.services.audio_cue_service import AudioCueService
from app.adapters.webcam_adapter import WebcamAdapter
from app.services.video_service import VideoService

def record_video(session_dir, duration=10):
    cam = WebcamAdapter()
    cam.open()

    video = VideoService(session_dir)
    video.open()

    start = time.time()
    while time.time() - start < duration:
        ok, frame = cam.read_frame()
        if ok:
            video.write_frame(frame)

    cam.close()
    video.close()

session_info, session_dir = SessionService().create_session()
print(session_info)

audio = AudioCueService(session_dir)

video_thread = threading.Thread(target=record_video, args=(session_dir, 10))
audio_thread = threading.Thread(target=audio.run_script)

video_thread.start()
audio_thread.start()

video_thread.join()
audio_thread.join()