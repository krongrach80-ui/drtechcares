"""Configuration & environment settings for Dr.Tech.Care Edge Vision Server."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

EDGE_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = EDGE_ROOT.parent


def _load_env_files() -> None:
    """Load edge-vision/.env and workspace root .env if present."""
    candidates = [WORKSPACE_ROOT / ".env", EDGE_ROOT / ".env"]
    try:
        from dotenv import load_dotenv  # type: ignore

        for env_path in candidates:
            if env_path.exists():
                load_dotenv(dotenv_path=env_path, override=False)
        return
    except ImportError:
        pass

    for env_path in candidates:
        if not env_path.exists():
            continue
        try:
            for raw_line in env_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
        except OSError:
            continue


_load_env_files()


class EdgeVisionSettings(BaseModel):
    """Clinical-grade settings for 1080p60 camera, RTMW3D-l, One-Euro, and InsightFace."""

    APP_NAME: str = "Dr.Tech.Care Clinical 3D Edge Vision"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    CORS_ORIGINS: List[str] = Field(
        default_factory=lambda: [
            "http://localhost:5173",
            "http://localhost:4173",
            "http://127.0.0.1:5173",
            "https://drtechcares-platform.krongrach80.workers.dev",
            "*",
        ]
    )
    KIOSK_ID: str = "KIOSK-OPD-01"

    # Camera parameters (1920x1080 or 1280x720 at 60 FPS with locked optical focus)
    CAMERA_INDEX: int = 0
    CAMERA_WIDTH: int = 1920
    CAMERA_HEIGHT: int = 1080
    CAMERA_FALLBACK_WIDTH: int = 1280
    CAMERA_FALLBACK_HEIGHT: int = 720
    CAMERA_FPS: int = 60
    AUTOFOCUS_OFF: int = 0
    FIXED_FOCUS_2M: int = 35
    AUTO_EXPOSURE_MANUAL: float = 0.25
    FIXED_EXPOSURE: float = -6.0
    MOCK_VIDEO_PATH: Optional[str] = None
    WS_TARGET_HZ: float = 60.0

    # RTMW3D-l (133 Wholebody 3D) model & runtime settings
    RTMW3D_MODE: str = "performance"
    RTMW3D_TO_OPENPOSE_KEYPOINTS: bool = False
    USE_GPU_IF_AVAILABLE: bool = True

    # One-Euro Filter parameters for 3D (X, Y, Z) jitter removal
    ONE_EURO_MIN_CUTOFF: float = 1.2
    ONE_EURO_BETA: float = 0.08
    ONE_EURO_D_CUTOFF: float = 1.0

    # 2-Second Standing Calibration & Anti-Cheating thresholds
    CALIBRATION_DURATION_SEC: float = 2.0
    CALIBRATION_STABILITY_MAX_TILT_DEG: float = 12.0
    TRUNK_LEAN_MAX_DEG: float = 10.0
    SHOULDER_HIKE_MAX_CM: float = 3.0
    KNEE_VALGUS_MAX_DEG: float = 8.0
    HEEL_LIFT_MAX_CM: float = 2.5
    MAX_ANGULAR_VELOCITY_DEG_S: float = 145.0
    FALL_DROP_VELOCITY_THRESHOLD: float = 0.85

    # InsightFace (buffalo_l / ArcFace 512-D) 120-Frame 5-Angle Enrollment & Matcher
    INSIGHTFACE_MODEL_NAME: str = "buffalo_l"
    FACE_MATCH_THRESHOLD: float = 0.62
    FACE_TOTAL_FRAMES: int = 120
    FACE_BUCKET_TARGETS: Dict[str, int] = Field(
        default_factory=lambda: {
            "center": 40,
            "left": 20,
            "right": 20,
            "up": 20,
            "down": 20,
        }
    )

    # Continuous Anti-Cheat & Identity Guard thresholds
    MULTI_PERSON_SKELETON_MIN_SCORE: float = 0.60
    MULTI_PERSON_MIN_TORSO_HEIGHT_RATIO: float = 0.25
    MULTI_PERSON_FACE_MIN_DET_SCORE: float = 0.65
    MULTI_PERSON_FACE_MIN_PX: int = 45
    MULTI_PERSON_TTS_COOLDOWN_SEC: float = 4.0
    IDENTITY_CHECK_FRAME_INTERVAL: int = 8
    IMPOSTOR_FACE_MIN_DET_SCORE: float = 0.70
    IMPOSTOR_SIMILARITY_MIN: float = 0.68
    IMPOSTOR_CONSECUTIVE_MISSES: int = 5
    UNLOCK_VERIFY_SIMILARITY_MIN: float = 0.72
    UNLOCK_CONSECUTIVE_MATCHES: int = 3

    # Supabase Cloud integration
    SUPABASE_URL: str = ""
    SUPABASE_ANON_KEY: str = ""
    SUPABASE_SERVICE_ROLE_KEY: str = ""

    @classmethod
    def from_env(cls) -> "EdgeVisionSettings":
        """Build settings from environment variables with sane clinical defaults."""
        mock_raw = os.getenv("MOCK_VIDEO_PATH")
        mock_val = mock_raw.strip() if mock_raw and mock_raw.strip() else None

        return cls(
            APP_HOST=os.getenv("EDGE_VISION_HOST", "0.0.0.0"),
            APP_PORT=int(os.getenv("EDGE_VISION_PORT", "8000")),
            KIOSK_ID=os.getenv("KIOSK_ID", "KIOSK-OPD-01"),
            CAMERA_INDEX=int(os.getenv("CAMERA_INDEX", "0")),
            CAMERA_WIDTH=int(os.getenv("CAMERA_WIDTH", "1920")),
            CAMERA_HEIGHT=int(os.getenv("CAMERA_HEIGHT", "1080")),
            CAMERA_FPS=int(os.getenv("CAMERA_FPS", "60")),
            AUTOFOCUS_OFF=int(os.getenv("AUTOFOCUS_OFF", "0")),
            FIXED_FOCUS_2M=int(os.getenv("FIXED_FOCUS_2M", "35")),
            MOCK_VIDEO_PATH=mock_val,
            WS_TARGET_HZ=float(os.getenv("WS_TARGET_HZ", "60.0")),
            RTMW3D_MODE=os.getenv("RTMW3D_MODE", "performance"),
            ONE_EURO_MIN_CUTOFF=float(os.getenv("ONE_EURO_MIN_CUTOFF", "1.2")),
            ONE_EURO_BETA=float(os.getenv("ONE_EURO_BETA", "0.08")),
            ONE_EURO_D_CUTOFF=float(os.getenv("ONE_EURO_D_CUTOFF", "1.0")),
            CALIBRATION_DURATION_SEC=float(os.getenv("CALIBRATION_DURATION_SEC", "2.0")),
            TRUNK_LEAN_MAX_DEG=float(os.getenv("TRUNK_LEAN_MAX_DEG", "10.0")),
            SHOULDER_HIKE_MAX_CM=float(os.getenv("SHOULDER_HIKE_MAX_CM", "3.0")),
            KNEE_VALGUS_MAX_DEG=float(os.getenv("KNEE_VALGUS_MAX_DEG", "8.0")),
            HEEL_LIFT_MAX_CM=float(os.getenv("HEEL_LIFT_MAX_CM", "2.5")),
            MAX_ANGULAR_VELOCITY_DEG_S=float(os.getenv("MAX_ANGULAR_VELOCITY_DEG_S", "145.0")),
            FALL_DROP_VELOCITY_THRESHOLD=float(os.getenv("FALL_DROP_VELOCITY_THRESHOLD", "0.85")),
            FACE_MATCH_THRESHOLD=float(os.getenv("FACE_MATCH_THRESHOLD", "0.62")),
            SUPABASE_URL=os.getenv("SUPABASE_URL") or os.getenv("VITE_SUPABASE_URL") or "",
            SUPABASE_ANON_KEY=os.getenv("SUPABASE_ANON_KEY") or os.getenv("VITE_SUPABASE_ANON_KEY") or "",
            SUPABASE_SERVICE_ROLE_KEY=os.getenv("SUPABASE_SERVICE_ROLE_KEY") or "",
        )


@lru_cache(maxsize=1)
def get_settings() -> EdgeVisionSettings:
    """Return cached settings instance."""
    return EdgeVisionSettings.from_env()

