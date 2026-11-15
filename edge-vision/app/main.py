"""FastAPI entrypoint for Dr.Tech.Care Clinical 3D Edge Vision Server."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes_enroll import router as enroll_router
from app.api.ws_kiosk import KioskRuntimeCoordinator, router as ws_router
from app.config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("edge_vision.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    coordinator = KioskRuntimeCoordinator(settings)
    app.state.coordinator = coordinator
    await coordinator.startup()
    try:
        yield
    finally:
        await coordinator.shutdown()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.APP_NAME,
        version="1.1.0",
        description=(
            "Clinical-grade 3D rehabilitation telemetry using MediaPipe Pose Heavy / RTMW3D-l, "
            "dual 2D/3D One-Euro Filter, True 3D World Vector Biomechanics, and InsightFace 512-D Face ID."
        ),
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(ws_router)
    app.include_router(enroll_router)

    @app.get("/health")
    async def health_check() -> Dict[str, Any]:
        coordinator: KioskRuntimeCoordinator = app.state.coordinator
        w, h = coordinator.camera.resolution
        return {
            "status": "ok",
            "service": "drtechcares-edge-vision",
            "kiosk_id": settings.KIOSK_ID,
            "engine": coordinator.pose_estimator.backend_name,
            "pose_backend": settings.POSE_BACKEND,
            "kiosk_pose_mode": settings.KIOSK_POSE_MODE,
            "mediapipe_config": settings.get_mediapipe_pose_config(),
            "face_engine": coordinator.face_manager.backend_name,
            "camera_mode": coordinator.camera.mode,
            "camera_resolution": f"{w}x{h}",
            "fps": coordinator.effective_fps,
            "enrolled_candidates": coordinator.face_manager.candidate_count,
            "fsm_state": coordinator.fsm.fsm_state,
            "movement_phase": coordinator.fsm.movement_phase,
            "hold_progress_pct": coordinator.fsm.hold_progress_pct,
            "wholebody_keypoints": 133,
        }

    @app.get("/api/telemetry/latest")
    @app.get("/telemetry")
    async def latest_telemetry() -> Dict[str, Any]:
        """Return latest real-time clinical biomechanics telemetry snapshot."""
        coordinator: KioskRuntimeCoordinator = app.state.coordinator
        payload = coordinator._last_telemetry or coordinator.step_once(include_frame=False)
        return {
            "kiosk_state": payload.kiosk_state,
            "fsm_state": payload.fsm_state,
            "movement_phase": payload.movement_phase,
            "current_rom_deg": payload.telemetry_analysis.current_rom_deg,
            "target_rom_deg": payload.telemetry_analysis.target_rom_deg,
            "hold_progress_pct": payload.hold_progress_pct,
            "hold_remaining_sec": payload.rep_counter.hold_remaining_sec,
            "hold_timer_frozen": payload.telemetry_analysis.hold_timer_frozen,
            "stability_score": payload.stability_score,
            "is_form_valid": payload.telemetry_analysis.is_form_valid,
            "compensation_detected": payload.telemetry_analysis.compensation_detected,
            "compensation_warnings": payload.compensation_warnings,
            "telemetry": payload.model_dump(exclude={"frame_base64"}),
        }

    return app


app = create_app()
