from app.services.session_service import SessionService
from app.services.audio_cue_service import AudioCueService

session_info, session_dir = SessionService().create_session()
print(session_info)

audio = AudioCueService(session_dir)
audio.run_script()