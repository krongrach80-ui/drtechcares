"""REST API routes for 120-frame Face ID enrollment, 1:N verification, Supabase sync & exercises."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from fastapi import APIRouter, HTTPException, Request

from app.biomechanics.exercises import list_all_exercises
from app.schemas.telemetry import (
    EnrollFinalizeRequest,
    EnrollFrameRequest,
    EnrollStartRequest,
    ExercisePrescriptionSchema,
    FaceEnrollmentStateSchema,
    FaceID120Response,
    VerifyFaceRequest,
    VerifyFaceResponse,
)

router = APIRouter(prefix="/api", tags=["edge-vision"])


@router.get("/exercises")
async def get_exercises(scope: str = "core") -> Dict[str, Any]:
    """List clinical rehabilitation exercises (pass ?scope=all to include Elderly Care & MOPH modules)."""
    items = list_all_exercises(include_elderly_moph=(scope.strip().lower() == "all"))
    return {"count": len(items), "exercises": items}


@router.post("/enroll/start", response_model=FaceEnrollmentStateSchema)
async def enroll_start(body: EnrollStartRequest, request: Request) -> FaceEnrollmentStateSchema:
    """Create a 120-frame 5-angle InsightFace enrollment session."""
    coordinator = request.app.state.coordinator
    return coordinator.face_manager.start_enrollment(
        patient_id=body.patient_id,
        hn=body.hn,
        full_name=body.full_name,
    )


@router.post("/enroll/frame", response_model=FaceEnrollmentStateSchema)
async def enroll_frame(body: EnrollFrameRequest, request: Request) -> FaceEnrollmentStateSchema:
    """Submit a browser frame (base64) or pull live camera frame into 120-frame InsightFace aggregator."""
    coordinator = request.app.state.coordinator
    frame = None
    if body.image_base64:
        frame = coordinator.face_manager.decode_base64_frame(body.image_base64)
    if frame is None:
        _, frame = coordinator.camera.read_latest()
    if frame is None:
        # Construct minimal synthetic frame if requested synchronously before camera tick
        frame = np.full((240, 240, 3), 120, dtype=np.uint8)

    try:
        return coordinator.face_manager.process_enrollment_frame(
            session_id=body.session_id,
            frame_bgr=frame,
            forced_bucket=body.forced_bucket,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/enroll/finalize", response_model=FaceID120Response)
async def enroll_finalize(body: EnrollFinalizeRequest, request: Request) -> FaceID120Response:
    """Aggregate 120 ArcFace 512-D vectors and persist to Supabase `patient_biometrics`."""
    coordinator = request.app.state.coordinator
    try:
        return coordinator.face_manager.finalize_enrollment(
            session_id=body.session_id,
            patient_id=body.patient_id,
            hn=body.hn,
            persist_to_supabase=body.persist_to_supabase,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/biometrics/verify", response_model=VerifyFaceResponse)
async def verify_face(body: VerifyFaceRequest, request: Request) -> VerifyFaceResponse:
    """Run real-time 1:N InsightFace ArcFace matching (< 100ms in RAM)."""
    coordinator = request.app.state.coordinator
    frame = None
    if body.image_base64:
        frame = coordinator.face_manager.decode_base64_frame(body.image_base64)
    if frame is None and body.use_edge_camera:
        _, frame = coordinator.camera.read_latest()
    if frame is None:
        raise HTTPException(status_code=400, detail="No valid camera or image frame provided.")

    feat = coordinator.face_manager.extract_face_features(frame)
    if feat is None:
        return VerifyFaceResponse(
            matched=False,
            patient=None,
            similarity=0.0,
            latency_ms=0.0,
            candidate_count=coordinator.face_manager.candidate_count,
        )

    match = coordinator.face_manager.match_1_to_n(
        feat["embedding"],
        yaw_deg=float(feat["yaw_deg"]),
        pitch_deg=float(feat["pitch_deg"]),
        roll_deg=float(feat.get("roll_deg", 0.0)),
    )
    if match.id:
        coordinator.fsm.set_patient_matched(match, auto_start_calibration=True)

    return VerifyFaceResponse(
        matched=bool(match.id),
        patient=match if match.id else None,
        similarity=match.similarity,
        latency_ms=match.latency_ms,
        candidate_count=coordinator.face_manager.candidate_count,
    )


@router.post("/biometrics/sync")
async def sync_biometrics(request: Request) -> Dict[str, Any]:
    """Force reload of enrolled ArcFace 512-D templates from Supabase Cloud."""
    coordinator = request.app.state.coordinator
    count = coordinator.face_manager.sync_from_supabase()
    return {
        "status": "ok",
        "synced_count": count,
        "backend": coordinator.face_manager.backend_name,
    }


@router.post("/kiosk/prescription")
async def set_kiosk_prescription(body: ExercisePrescriptionSchema, request: Request) -> Dict[str, Any]:
    """Set active clinical exercise prescription and trigger 2-second calibration."""
    coordinator = request.app.state.coordinator
    coordinator.fsm.set_prescription(body, start_calibration=True)
    return {
        "status": "ok",
        "fsm_state": coordinator.fsm.fsm_state,
        "exercise_id": coordinator.fsm.rule.id,
        "target_rom_deg": coordinator.fsm.prescription.target_rom_deg,
    }

