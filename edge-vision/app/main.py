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
        version="1.0.0",
        description=(
            "Clinical-grade 3D rehabilitation telemetry using RTMW3D-l (133 Wholebody 3D), "
            "One-Euro Filter, and InsightFace buffalo_l 512-D Face ID."
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
            "face_engine": coordinator.face_manager.backend_name,
            "camera_mode": coordinator.camera.mode,
            "camera_resolution": f"{w}x{h}",
            "fps": coordinator.effective_fps,
            "enrolled_candidates": coordinator.face_manager.candidate_count,
            "fsm_state": coordinator.fsm.fsm_state,
            "wholebody_keypoints": 133,
        }

    return app


app = create_app()
