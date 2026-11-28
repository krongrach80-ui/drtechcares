"""WebSocket /ws/kiosk runtime coordinator streaming 30-60 Hz 3D biomechanics telemetry."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from typing import Any, Dict, Optional, Set, Tuple

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.biomechanics.state_machine import ClinicalRehabFSM
from app.config import EdgeVisionSettings
from app.schemas.telemetry import (
    EdgeKioskTelemetryPayload,
    ExercisePrescriptionSchema,
    FaceEnrollmentStateSchema,
    HandRaiseTelemetry,
    IdentityGuardSchema,
    PatientMatchSchema,
)
from app.vision.camera import ThreadedCameraCapture
from app.vision.face_120 import InsightFace120Manager
from app.vision.pose_3d import (
    RTMW3DPoseEstimator,
    map_mediapipe33_to_wholebody133,
    wholebody133_to_world_meters,
)

logger = logging.getLogger("edge_vision.ws_kiosk")
router = APIRouter()


class KioskRuntimeCoordinator:
    """Coordinates camera thread, RTMW3D-l pose estimator, InsightFace matcher, and FSM loop."""

    def __init__(self, settings: EdgeVisionSettings) -> None:
        self.settings = settings
        self.camera = ThreadedCameraCapture(settings)
        self.pose_estimator = RTMW3DPoseEstimator(settings)
        self.face_manager = InsightFace120Manager(settings)
        self.fsm = ClinicalRehabFSM(settings)

        self._clients: Set[WebSocket] = set()
        self._stream_video_clients: Set[WebSocket] = set()
        self._task: Optional[asyncio.Task[None]] = None
        self._running = False
        self._started_at: float = time.perf_counter()
        self._effective_fps: float = 0.0
        self._last_face_check_at: float = 0.0
        self._last_telemetry: Optional[EdgeKioskTelemetryPayload] = None
        self._last_frame_b64: Optional[str] = None
        self._mock_pose_override: Optional[str] = None
        self._latest_face_enroll_state: Optional[FaceEnrollmentStateSchema] = None
        self._client_pose_2d: Optional[np.ndarray] = None
        self._client_pose_world_3d: Optional[np.ndarray] = None
        self._client_vis_33: Optional[np.ndarray] = None
        self._client_person_count: int = 1
        self._client_landmarks_at: float = -999.0
        self._state_lock = threading.RLock()

    @property
    def effective_fps(self) -> float:
        return round(self._effective_fps or self.camera.fps or 30.0, 1)

    async def startup(self) -> None:
        self._started_at = time.perf_counter()
        self.camera.start()
        self.face_manager.sync_from_supabase()
        self._running = True
        self._task = asyncio.create_task(self._telemetry_loop(), name="edge-ws-telemetry-loop")
        logger.info(
            "Edge Vision Coordinator started (camera=%s, pose=%s, face=%s)",
            self.camera.mode,
            self.pose_estimator.backend_name,
            self.face_manager.backend_name,
        )

    async def shutdown(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        self.camera.stop()
        self.pose_estimator.close()

    def set_mock_pose_override(self, pose: Optional[str]) -> None:
        self._mock_pose_override = pose

    @staticmethod
    def _parse_mediapipe_landmark_array(raw: Any) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if not isinstance(raw, list) or len(raw) < 33:
            return None
        pts = np.zeros((33, 3), dtype=np.float64)
        vis = np.ones((33,), dtype=np.float64)
        for i in range(33):
            item = raw[i]
            if isinstance(item, dict):
                pts[i, 0] = float(item.get("x", 0.0))
                pts[i, 1] = float(item.get("y", 0.0))
                pts[i, 2] = float(item.get("z", 0.0))
                vis[i] = float(item.get("score", item.get("visibility", 1.0)))
        return pts, np.clip(vis, 0.0, 1.0)

    def step_once(self, include_frame: bool = False) -> EdgeKioskTelemetryPayload:
        """Execute one synchronous vision + biomechanics + FSM tick (useful for loop & tests)."""
        now_sec = time.perf_counter() - self._started_at
        _, frame = self.camera.read_latest()
        is_mock = self.camera.mode == "synthetic_mock"

        has_fresh_client_pose = (
            self._client_pose_2d is not None
            and self._client_vis_33 is not None
            and (now_sec - self._client_landmarks_at) <= 2.0
        )
        err_code: Optional[str] = None
        err_message: Optional[str] = None
        if has_fresh_client_pose:
            pose_landmarks_2d = self._client_pose_2d
            vis_33 = self._client_vis_33
            xyz, scores = map_mediapipe33_to_wholebody133(pose_landmarks_2d, vis_33)
            if self._client_pose_world_3d is not None:
                pose_world_landmarks_3d = self._client_pose_world_3d
                world_xyz, _ = map_mediapipe33_to_wholebody133(pose_world_landmarks_3d, vis_33)
            else:
                pose_world_landmarks_3d = pose_landmarks_2d
                world_xyz = wholebody133_to_world_meters(xyz)
            skeleton_count = self._client_person_count
            skeleton_bboxes = []
            effective_camera_mode = "live_browser_bridge"
        else:
            pose_frame = self.pose_estimator.estimate_clinical_with_crowd(
                frame_bgr=frame,
                timestamp_sec=now_sec,
                exercise_id=self.fsm.rule.id,
                is_calibrating=(self.fsm.fsm_state == "CALIBRATE_2S"),
                mock_pose_override=self._mock_pose_override,
                is_mock_mode=is_mock,
            )
            xyz = pose_frame.wholebody_2d_133
            world_xyz = pose_frame.wholebody_world_133
            scores = pose_frame.scores_133
            pose_landmarks_2d = pose_frame.pose_landmarks_2d
            pose_world_landmarks_3d = pose_frame.pose_world_landmarks_3d
            vis_33 = pose_frame.visibilities_33
            skeleton_count = pose_frame.skeleton_count
            skeleton_bboxes = pose_frame.valid_bboxes
            effective_camera_mode = self.camera.mode

            if not is_mock and self._mock_pose_override is None:
                if self.camera.mode == "camera_disconnected" or frame is None:
                    err_code = "CAMERA_DISCONNECTED"
                    err_message = "Camera Disconnected: ไม่พบการเชื่อมต่อกล้องเว็บแคม"
                elif pose_frame.backend_used == "AI_MODEL_ERROR":
                    err_code = "AI_MODEL_ERROR"
                    err_message = "AI Model Error: โหลดโมเดล MediaPipe PoseLandmarker ล้มเหลว"
                elif pose_frame.skeleton_count == 0:
                    err_code = "NO_POSE_DETECTED"
                    err_message = "ไม่พบผู้ใช้งานในเฟรมกล้อง กรุณาขยับเข้าหน้ากล้อง"

        # Face ID 120 active enrollment or real-time 1:N authentication
        active_sid = self.face_manager.active_session_id
        face_enroll_state: Optional[FaceEnrollmentStateSchema] = None
        if active_sid and frame is not None:
            try:
                face_enroll_state = self.face_manager.process_enrollment_frame(active_sid, frame)
                self._latest_face_enroll_state = face_enroll_state
            except KeyError:
                self._latest_face_enroll_state = None
        elif (
            self.fsm.fsm_state == "FACE_AUTH"
            and frame is not None
            and self.face_manager.candidate_count > 0
            and (now_sec - self._last_face_check_at) >= 0.2
        ):
            self._last_face_check_at = now_sec
            feat = self.face_manager.extract_face_features(frame)
            if feat is not None:
                match = self.face_manager.match_1_to_n(
                    feat["embedding"],
                    yaw_deg=float(feat["yaw_deg"]),
                    pitch_deg=float(feat["pitch_deg"]),
                    roll_deg=float(feat.get("roll_deg", 0.0)),
                )
                if match.id:
                    self.fsm.set_patient_matched(match, auto_start_calibration=True)

        # Continuous Anti-Cheat & Identity Guard (Rule 1 + Rule 2) during active rehab
        if has_fresh_client_pose and frame is None:
            identity_guard = IdentityGuardSchema(
                person_count=skeleton_count,
                status="MULTI_PERSON" if skeleton_count > 1 else "VERIFIED",
                is_counting_allowed=(skeleton_count <= 1),
            )
        elif self.fsm.fsm_state != "FACE_AUTH" and not active_sid:
            identity_guard = self.face_manager.evaluate_identity_guard(
                frame_bgr=frame,
                primary_xyz=xyz,
                primary_scores=scores,
                skeleton_count=skeleton_count,
                skeleton_bboxes=skeleton_bboxes,
                mock_override=self._mock_pose_override,
            )
        else:
            identity_guard = IdentityGuardSchema(person_count=skeleton_count)

        has_valid_landmarks = skeleton_count > 0 and err_code is None
        keypoints_3d = self.pose_estimator.to_keypoint_schemas(xyz, scores) if has_valid_landmarks else []
        pose_landmarks = (
            self.pose_estimator.to_world_landmark_schemas(pose_landmarks_2d, vis_33)
            if has_valid_landmarks
            else []
        )
        pose_world_landmarks = (
            self.pose_estimator.to_world_landmark_schemas(pose_world_landmarks_3d, vis_33)
            if has_valid_landmarks
            else []
        )
        hand_status = (
            self.pose_estimator.extract_hand_raise_telemetry(xyz, scores)
            if has_valid_landmarks
            else HandRaiseTelemetry()
        )
        frame_b64: Optional[str] = None
        if include_frame and frame is not None:
            frame_b64 = self.camera.encode_jpeg_base64(frame, max_width=640, quality=68)
            if frame_b64:
                self._last_frame_b64 = frame_b64
        elif self._stream_video_clients and frame is not None:
            frame_b64 = self._last_frame_b64

        payload = self.fsm.step(
            pts=xyz,
            scores=scores,
            timestamp_sec=now_sec,
            fps=self.effective_fps,
            camera_mode=effective_camera_mode,
            hand_status=hand_status,
            keypoints_3d=keypoints_3d,
            face_enrollment=face_enroll_state or self._latest_face_enroll_state,
            frame_base64=frame_b64,
            identity_guard=identity_guard,
            world_pts=world_xyz,
            pose_world_landmarks=pose_world_landmarks,
            pose_landmarks=pose_landmarks,
            error_code=err_code,
            error_message=err_message,
        )
        self._last_telemetry = payload
        return payload

    async def _telemetry_loop(self) -> None:
        target_hz = min(60.0, max(15.0, float(self.settings.WS_TARGET_HZ)))
        interval = 1.0 / target_hz
        fps_start = time.perf_counter()
        tick_count = 0

        while self._running:
            t0 = time.perf_counter()
            if self._clients:
                need_video = len(self._stream_video_clients) > 0 and (tick_count % 2 == 0)
                payload = await asyncio.to_thread(self.step_once, need_video)
                base_dict = payload.model_dump()
                dead: Set[WebSocket] = set()
                for ws in list(self._clients):
                    try:
                        if ws in self._stream_video_clients:
                            await ws.send_json(base_dict)
                        else:
                            slim = dict(base_dict)
                            slim["frame_base64"] = None
                            await ws.send_json(slim)
                    except Exception:
                        dead.add(ws)
                for ws in dead:
                    self._clients.discard(ws)
                    self._stream_video_clients.discard(ws)

            tick_count += 1
            now = time.perf_counter()
            if now - fps_start >= 1.0:
                self._effective_fps = tick_count / (now - fps_start)
                tick_count = 0
                fps_start = now

            elapsed = time.perf_counter() - t0
            await asyncio.sleep(max(0.002, interval - elapsed))

    async def handle_client_command(self, ws: WebSocket, raw_text: str) -> None:
        try:
            msg: Dict[str, Any] = json.loads(raw_text)
        except json.JSONDecodeError:
            await ws.send_json({"type": "error", "message": "Invalid JSON command"})
            return

        cmd = str(msg.get("type") or msg.get("action") or "").strip().lower()
        if not cmd and isinstance(msg.get("prescription"), dict):
            cmd = "prescription"
        if cmd == "ping":
            await ws.send_json({"type": "pong", "ts_ms": int(time.time() * 1000)})
            return

        if cmd == "prescription":
            rx_raw = (
                msg.get("data")
                if isinstance(msg.get("data"), dict)
                else msg.get("prescription")
                if isinstance(msg.get("prescription"), dict)
                else msg
            )
            def _safe_float(val: Any, default: float) -> float:
                if val is None or val == "":
                    return default
                try:
                    return float(val)
                except (TypeError, ValueError):
                    return default

            def _safe_opt_float(val: Any) -> Optional[float]:
                if val is None or val == "":
                    return None
                try:
                    return float(val)
                except (TypeError, ValueError):
                    return None

            def _safe_int(val: Any, default: int) -> int:
                if val is None or val == "":
                    return default
                try:
                    return int(val)
                except (TypeError, ValueError):
                    return default

            def _safe_opt_int(val: Any) -> Optional[int]:
                if val is None or val == "":
                    return None
                try:
                    return int(val)
                except (TypeError, ValueError):
                    return None

            target_rom_raw = rx_raw.get("target_rom_deg")
            if target_rom_raw is None:
                target_rom_raw = rx_raw.get("minAngleDeg")
            hold_sec_raw = rx_raw.get("hold_sec")
            if hold_sec_raw is None:
                hold_sec_raw = rx_raw.get("holdDurationSec")
            reps_raw = rx_raw.get("target_reps")
            if reps_raw is None:
                reps_raw = rx_raw.get("reps")

            rx = ExercisePrescriptionSchema(
                exercise_id=str(rx_raw.get("exercise_id") or rx_raw.get("base_code") or rx_raw.get("id") or "ELDERLY_SH_ABD"),
                exercise_name=str(rx_raw.get("exercise_name") or rx_raw.get("name") or "Clinical Exercise"),
                target_joint=str(rx_raw.get("target_joint") or "right_shoulder"),
                prescribed_side=(
                    str(rx_raw.get("prescribed_side") or rx_raw.get("side")).strip()
                    if (rx_raw.get("prescribed_side") or rx_raw.get("side"))
                    else None
                ),
                target_rom_deg=_safe_float(target_rom_raw, 110.0),
                hold_sec=_safe_float(hold_sec_raw, 3.0),
                target_reps=max(1, _safe_int(reps_raw, 5)),
                max_safe_rom_deg=_safe_opt_float(rx_raw.get("max_safe_rom_deg")),
                affected_side=(str(rx_raw.get("affected_side")).strip() if rx_raw.get("affected_side") else None),
                care_group=(str(rx_raw.get("care_group")).strip() if rx_raw.get("care_group") else None),
                motor_assessment_score=_safe_opt_int(rx_raw.get("motor_assessment_score")),
            )
            start_calib = bool(msg.get("start_calibration", True))
            with self._state_lock:
                self.fsm.set_prescription(rx, start_calibration=start_calib)
            await ws.send_json({"type": "ack", "command": "prescription", "fsm_state": self.fsm.fsm_state})
            return

        if cmd == "start_calibration":
            self.fsm.start_calibration()
            await ws.send_json({"type": "ack", "command": "start_calibration", "fsm_state": self.fsm.fsm_state})
            return

        if cmd == "lock_patient":
            pdata = msg.get("patient") if isinstance(msg.get("patient"), dict) else msg
            pid = str(pdata.get("patient_id") or pdata.get("id") or "").strip()
            if pid:
                hn_val = str(pdata.get("hn") or "")
                name_val = str(pdata.get("full_name") or pdata.get("name") or pid)
                master_emb = pdata.get("master_embedding") if isinstance(pdata.get("master_embedding"), list) else None
                angle_embs = pdata.get("angle_embeddings") if isinstance(pdata.get("angle_embeddings"), dict) else None
                self.face_manager.lock_active_patient(
                    patient_id=pid,
                    hn=hn_val,
                    full_name=name_val,
                    master_embedding=master_emb,
                    angle_embeddings=angle_embs,
                )
                self.fsm.set_patient_matched(
                    PatientMatchSchema(id=pid, hn=hn_val, name=name_val, similarity=1.0),
                    auto_start_calibration=bool(msg.get("start_calibration", False)),
                )
            await ws.send_json(
                {
                    "type": "ack",
                    "command": "lock_patient",
                    "patient_id": pid or None,
                    "fsm_state": self.fsm.fsm_state,
                }
            )
            return

        if cmd == "reset_session":
            return_auth = bool(msg.get("return_to_auth", False))
            self.face_manager.reset_identity_guard(clear_patient=return_auth)
            self.fsm.reset_session(return_to_auth=return_auth)
            await ws.send_json({"type": "ack", "command": "reset_session", "fsm_state": self.fsm.fsm_state})
            return

        if cmd == "emergency_stop":
            reason = str(msg.get("reason") or "Manual Kiosk Emergency Stop Button Pressed")
            self.fsm.trigger_emergency(reason)
            await ws.send_json({"type": "ack", "command": "emergency_stop", "fsm_state": self.fsm.fsm_state})
            return

        if cmd == "mock_pose":
            pose_val = msg.get("pose")
            self.set_mock_pose_override(str(pose_val) if pose_val else None)
            await ws.send_json({"type": "ack", "command": "mock_pose", "pose": self._mock_pose_override})
            return

        if cmd == "config":
            if msg.get("stream_video") is True:
                self._stream_video_clients.add(ws)
            elif msg.get("stream_video") is False:
                self._stream_video_clients.discard(ws)
            await ws.send_json({"type": "ack", "command": "config"})
            return

        if cmd == "client_landmarks":
            parsed_2d = self._parse_mediapipe_landmark_array(msg.get("pose_landmarks"))
            if parsed_2d is not None:
                parsed_world = self._parse_mediapipe_landmark_array(msg.get("pose_world_landmarks"))
                with self._state_lock:
                    self._client_pose_2d, self._client_vis_33 = parsed_2d
                    self._client_pose_world_3d = parsed_world[0] if parsed_world is not None else None
                    self._client_person_count = max(1, int(msg.get("person_count") or 1))
                    self._client_landmarks_at = time.perf_counter() - self._started_at
            return

        if cmd == "start_face_enroll":
            state = self.face_manager.start_enrollment(
                patient_id=msg.get("patient_id"),
                hn=msg.get("hn"),
                full_name=msg.get("full_name"),
            )
            self._latest_face_enroll_state = state
            await ws.send_json({"type": "ack", "command": "start_face_enroll", "session_id": state.session_id})
            return

    async def register_ws(self, ws: WebSocket) -> None:
        await ws.accept()
        self._clients.add(ws)
        self._stream_video_clients.add(ws)
        # Send immediate initial telemetry frame (with video frame if live) so client sees backend ready in <10ms
        initial = self.step_once(include_frame=True)
        await ws.send_json(initial.model_dump())

    def unregister_ws(self, ws: WebSocket) -> None:
        self._clients.discard(ws)
        self._stream_video_clients.discard(ws)


@router.websocket("/ws/kiosk")
async def ws_kiosk_endpoint(websocket: WebSocket) -> None:
    """Real-time 30-60 Hz WebSocket stream connecting Kiosk UI with RTMW3D & InsightFace."""
    coordinator: KioskRuntimeCoordinator = websocket.app.state.coordinator
    await coordinator.register_ws(websocket)
    try:
        while True:
            raw_msg = await websocket.receive_text()
            await coordinator.handle_client_command(websocket, raw_msg)
    except WebSocketDisconnect:
        coordinator.unregister_ws(websocket)
    except Exception as exc:
        logger.debug("WebSocket session closed: %s", exc)
        coordinator.unregister_ws(websocket)
