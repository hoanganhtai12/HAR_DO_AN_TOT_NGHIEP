from app.services.config_service import ConfigService
from app.services.session_service import SessionService
from app.services.scenario_service import ScenarioService
from app.services.audio_cue_service import AudioCueService


def main():
    # 1. Đọc session_config.json
    session_config = ConfigService().load_session_config()

    # 2. Tạo session folder
    session_info, session_dir = SessionService().create_session(session_config)

    print("Session ID:", session_info["session_id"])
    print("Session dir:", session_dir)

    # 3. Sinh action plan từ scenario + repeat_count + position_id
    action_plan = ScenarioService().build_action_plan(
        scenario_name=session_config["scenario"],
        repeat_count=session_config["repeat_count"],
        position_id=session_config["position_id"]
    )

    print("Total actions:", len(action_plan))

    # 4. Chạy audio cue và ghi action_events.csv
    AudioCueService(session_dir).run_action_plan(action_plan)

    print("Done")


if __name__ == "__main__":
    main()