"""Clinical Finite State Machine with Hysteresis, Movement Phases (IDLE->READY->CONCENTRIC/ECCENTRIC->HOLD->REP_COMPLETE), and Hold Timer Freeze."""

from __future__ import annotations

import logging
import time
from typing import List, Optional

import numpy as np

from app.biomechanics.calibrator import StandingCalibrator
from app.biomechanics.exercises import ClinicalExerciseRule, resolve_exercise_rule
from app.biomechanics.solver_3d import Biomechanics3DSolver
from app.config import EdgeVisionSettings
from app.schemas.telemetry import (
    CalibrationStateSchema,
    EdgeKioskTelemetryPayload,
    ExercisePrescriptionSchema,
    FaceEnrollmentStateSchema,
    HandRaiseTelemetry,
    IdentityGuardSchema,
    Keypoint3D,
    MovementPhase,
    PatientMatchSchema,
    RepCounterSchema,
    TelemetryAnalysisSchema,
    UIDirectivesSchema,
)

logger = logging.getLogger("edge_vision.fsm")

THAI_COMPENSATION_PROMPTS = {
    "TRUNK_LEAN": "โปรดหลังตรง อย่าเอียงลำตัวช่วยยก",
    "SHOULDER_HIKE": "ลดหัวไหล่ลง ระวังอย่ายักไหล่ช่วย",
    "SHOULDER_TILT": "ปรับแนวหัวไหล่ทั้งสองข้างให้เสมอกัน",
    "PELVIC_TILT": "คุมแนวสะโพกให้นิ่ง อย่าเอียงสะโพกช่วย",
    "ELBOW_FLEXION": "กรุณาเหยียดข้อศอกให้ตรงขณะทำท่า",
    "KNEE_VALGUS": "จัดแนวหัวเข่าให้ตรง อย่าให้เข่าบิดเข้าด้านใน",
    "HEEL_LIFT": "กรุณาวางส้นเท้าทั้งสองข้างให้ติดพื้น",
    "TOO_FAST": "เคลื่อนไหวเร็วเกินไป กรุณาค่อยๆ ทำช้าๆ เพื่อความปลอดภัย",
    "LOW_VISIBILITY": "มองเห็นข้อต่อไม่ชัดเจน กรุณาจัดตำแหน่งหน้ากล้องให้เหมาะสม",
}


class ClinicalRehabFSM:
    """
    Controls session flow across all clinical states with:
    - Exercise Phase tracking: IDLE -> READY -> CONCENTRIC / ECCENTRIC -> HOLD -> REP_COMPLETE
    - Hysteresis entry/exit thresholds preventing boundary flicker & double counting
    - Instant Hold Timer Freeze when solver_3d detects compensation or low visibility
    """

    def __init__(self, settings: EdgeVisionSettings) -> None:
        self.settings = settings
        self.calibrator = StandingCalibrator(
            duration_sec=settings.CALIBRATION_DURATION_SEC,
            stability_max_tilt_deg=settings.CALIBRATION_STABILITY_MAX_TILT_DEG,
            min_visibility=settings.CALIBRATION_MIN_VISIBILITY,
            min_torso_height_ratio=settings.CALIBRATION_MIN_TORSO_HEIGHT_RATIO,
            max_torso_height_ratio=settings.CALIBRATION_MAX_TORSO_HEIGHT_RATIO,
        )
        self.solver = Biomechanics3DSolver(settings)

        self.fsm_state: str = "FACE_AUTH"
        self.movement_phase: MovementPhase = "IDLE"
        self.patient: Optional[PatientMatchSchema] = None
        self.prescription = ExercisePrescriptionSchema()
        self.rule: ClinicalExerciseRule = resolve_exercise_rule(self.prescription.exercise_id)

        self.valid_reps: int = 0
        self.invalid_reps: int = 0
        self.target_reps: int = int(self.prescription.target_reps)
        self.hold_sec: float = float(self.prescription.hold_sec)
        self.hold_remaining_sec: float = self.hold_sec

        self._hold_accumulated_sec: float = 0.0
        self._hold_last_ts: Optional[float] = None
        self._hold_timer_frozen: bool = False
        self._rep_complete_emitted: bool = False
        self._prev_rom_deg: Optional[float] = None
        self._last_fault_counted = False
        self._emergency_reason: Optional[str] = None
        self._paused_from_state: Optional[str] = None
        self._last_multi_person_tts_ts: float = -10.0
        self._last_tts: str = "กรุณามองกล้องเพื่อยืนยันตัวตนด้วยใบหน้า"

    @property
    def hold_progress_pct(self) -> float:
        if self.fsm_state in ("REP_DONE", "SET_FINISHED"):
            return 100.0
        denom = max(float(self.hold_sec), 1e-3)
        return round(float(np.clip((self._hold_accumulated_sec / denom) * 100.0, 0.0, 100.0)), 1)

    def _reset_hold_timer(self) -> None:
        self._hold_accumulated_sec = 0.0
        self._hold_last_ts = None
        self._hold_timer_frozen = False
        self.hold_remaining_sec = self.hold_sec

    def set_patient_matched(self, match: PatientMatchSchema, auto_start_calibration: bool = True) -> None:
        self.patient = match
        if auto_start_calibration and self.fsm_state == "FACE_AUTH":
            self.start_calibration()

    def set_prescription(self, rx: ExercisePrescriptionSchema, start_calibration: bool = True) -> None:
        self.prescription = rx
        self.rule = resolve_exercise_rule(rx.exercise_id)
        self.valid_reps = 0
        self.invalid_reps = 0
        self.target_reps = max(1, int(rx.target_reps or self.rule.default_reps))
        self.hold_sec = max(0.5, float(rx.hold_sec or self.rule.default_hold_sec))
        self._reset_hold_timer()
        self._rep_complete_emitted = False
        self._prev_rom_deg = None
        self._last_fault_counted = False
        self._emergency_reason = None
        self.solver.reset()

        if start_calibration:
            self.start_calibration()
        elif self.calibrator.is_calibrated:
            self.fsm_state = "ACTIVE"
            self.movement_phase = "READY"

    def start_calibration(self) -> None:
        self.calibrator.reset()
        self.solver.reset()
        self._reset_hold_timer()
        self._rep_complete_emitted = False
        self._prev_rom_deg = None
        self._paused_from_state = None
        self._last_multi_person_tts_ts = -10.0
        self.fsm_state = "CALIBRATE_2S"
        self.movement_phase = "IDLE"
        self._last_tts = "กรุณายืนตัวตรงนิ่ง 2 วินาที เพื่อคาลิเบรตสัดส่วนร่างกาย"

    def trigger_emergency(self, reason: str) -> None:
        self.fsm_state = "EMERGENCY"
        self.movement_phase = "IDLE"
        self._emergency_reason = reason
        self._last_tts = "ตรวจพบความเสี่ยงฉุกเฉิน ระบบหยุดการฝึกและแจ้งเตือนเจ้าหน้าที่แล้ว"
        pid = self.patient.id if self.patient else None
        self.solver.dispatch_emergency_alert_supabase(pid, reason)

    def reset_session(self, return_to_auth: bool = True) -> None:
        self.calibrator.reset()
        self.solver.reset()
        self.valid_reps = 0
        self.invalid_reps = 0
        self._reset_hold_timer()
        self._rep_complete_emitted = False
        self._prev_rom_deg = None
        self._last_fault_counted = False
        self._emergency_reason = None
        self._paused_from_state = None
        self._last_multi_person_tts_ts = -10.0
        self.movement_phase = "IDLE"
        if return_to_auth:
            self.patient = None
            self.fsm_state = "FACE_AUTH"
            self._last_tts = "กรุณามองกล้องเพื่อยืนยันตัวตนด้วยใบหน้า"
        else:
            self.fsm_state = "CALIBRATE_2S"
            self._last_tts = "เริ่มคาลิเบรตสัดส่วนร่างกายใหม่ กรุณายืนตรง 2 วินาที"

    def _kiosk_state_alias(self) -> str:
        mapping = {
            "FACE_AUTH": "AUTH_FRAMING",
            "CALIBRATE_2S": "CALIBRATING",
            "ACTIVE": "ACTIVE_REP",
            "HOLD": "HOLD_PHASE",
            "REP_DONE": "REP_COMPLETED",
            "SET_FINISHED": "SET_FINISHED",
            "EMERGENCY": "EMERGENCY_HALT",
            "PAUSED_MULTI_PERSON": "PAUSED_MULTI_PERSON",
            "PAUSED_IMPOSTOR": "PAUSED_IMPOSTOR",
        }
        return mapping.get(self.fsm_state, "ACTIVE_REP")

    def _target_rom_deg(self) -> float:
        return float(self.prescription.target_rom_deg or self.rule.min_angle_deg)

    def _reached_target(self, rom_deg: float) -> bool:
        """Hysteresis-aware entry check (ACTIVE -> HOLD)."""
        return self.rule.is_in_target_zone(
            rom_deg,
            currently_holding=False,
            target_override=self._target_rom_deg(),
        )

    def _still_in_hold_zone(self, rom_deg: float) -> bool:
        """Hysteresis-aware hold retention check preventing flicker near threshold."""
        return self.rule.is_in_target_zone(
            rom_deg,
            currently_holding=True,
            target_override=self._target_rom_deg(),
        )

    def _returned_to_rest(self, rom_deg: float) -> bool:
        return self.rule.is_in_rest_zone(rom_deg)

    def _update_active_movement_phase(self, rom_deg: float) -> MovementPhase:
        """Determine IDLE / READY / CONCENTRIC / ECCENTRIC phase during ACTIVE tracking."""
        if self._returned_to_rest(rom_deg):
            return "READY"
        if self._prev_rom_deg is None:
            return "CONCENTRIC"
        delta = rom_deg - self._prev_rom_deg
        moving_toward_target = (delta <= -0.5) if self.rule.is_descending_target else (delta >= 0.5)
        moving_toward_rest = (delta >= 0.5) if self.rule.is_descending_target else (delta <= -0.5)
        if moving_toward_rest and not moving_toward_target:
            return "ECCENTRIC"
        return "CONCENTRIC"

    def step(
        self,
        pts: np.ndarray,
        scores: np.ndarray,
        timestamp_sec: float,
        fps: float,
        camera_mode: str,
        hand_status: HandRaiseTelemetry,
        keypoints_3d: List[Keypoint3D],
        face_enrollment: Optional[FaceEnrollmentStateSchema] = None,
        frame_base64: Optional[str] = None,
        identity_guard: Optional[IdentityGuardSchema] = None,
        world_pts: Optional[np.ndarray] = None,
        pose_world_landmarks: Optional[List[Keypoint3D]] = None,
        pose_landmarks: Optional[List[Keypoint3D]] = None,
        error_code: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> EdgeKioskTelemetryPayload:
        """Advance FSM one frame and emit complete WebSocket telemetry payload."""
        guard = identity_guard or IdentityGuardSchema()

        no_live_pose = (
            camera_mode != "synthetic_mock"
            and not error_code
            and len(keypoints_3d) == 0
            and len(pose_landmarks or []) == 0
        )
        effective_err_code = error_code or ("NO_POSE_DETECTED" if no_live_pose else None)
        effective_err_msg = error_message or (
            "ไม่พบโครงกระดูกผู้ใช้งานในเฟรมกล้อง กรุณาขยับตัวเข้ามาหน้ากล้อง"
            if no_live_pose
            else None
        )

        if effective_err_code:
            self.calibrator.estimated_distance_m = 0.0
            halt_calib = CalibrationStateSchema(
                is_calibrated=self.calibrator.is_calibrated,
                progress=0.0,
                remaining_sec=self.settings.CALIBRATION_DURATION_SEC,
                message_th=effective_err_msg or "ไม่สามารถตรวจจับท่าทางได้",
                distance_status="OCCLUDED",
                estimated_distance_m=0.0,
                min_visibility=0.0,
                missing_joints=[],
                bone_lengths=self.calibrator.bone_lengths,
            )
            idle_analysis = TelemetryAnalysisSchema(
                current_rom_deg=0.0,
                target_rom_deg=self._target_rom_deg(),
                rom_completion_pct=0.0,
                is_form_valid=False,
                visibility_ok=False,
                compensation_detected=["NONE"],
                compensation_warnings=[effective_err_msg] if effective_err_msg else [],
                pain_alert=False,
            )
            highlight = "RED" if effective_err_code in ("CAMERA_DISCONNECTED", "AI_MODEL_ERROR") else "AMBER"
            return EdgeKioskTelemetryPayload(
                kiosk_state=self._kiosk_state_alias(),  # type: ignore[arg-type]
                fsm_state=self.fsm_state,  # type: ignore[arg-type]
                timestamp_ms=int(time.time() * 1000),
                fps=round(fps, 1),
                engine_backend="MediaPipe PoseLandmarker Heavy + OneEuroFilter",
                camera_mode=camera_mode,
                movement_phase="IDLE",
                hold_progress_pct=self.hold_progress_pct,
                stability_score=0.0,
                compensation_warnings=idle_analysis.compensation_warnings,
                patient=self.patient,
                telemetry_analysis=idle_analysis,
                rep_counter=RepCounterSchema(
                    valid_reps=self.valid_reps,
                    invalid_reps=self.invalid_reps,
                    hold_remaining_sec=round(self.hold_remaining_sec, 2),
                    hold_progress_pct=self.hold_progress_pct,
                    hold_timer_frozen=self._hold_timer_frozen,
                    movement_phase="IDLE",
                    target_reps=self.target_reps,
                ),
                ui_directives=UIDirectivesSchema(
                    skeleton_highlight_joint=self.prescription.target_joint or self.rule.target_joint,
                    highlight_color=highlight,  # type: ignore[arg-type]
                    show_target_arc=False,
                ),
                tts_voice_th=effective_err_msg or self._last_tts,
                calibration=halt_calib,
                hand_status=HandRaiseTelemetry(),
                face_enrollment=face_enrollment,
                identity_guard=guard,
                keypoints_3d=[],
                pose_landmarks=[],
                pose_world_landmarks=[],
                frame_base64=frame_base64,
                error_code=effective_err_code,
                error_message=effective_err_msg,
            )

        live_dist_m = self.calibrator.estimate_live_distance_m(pts, scores)
        calib_state = CalibrationStateSchema(
            is_calibrated=self.calibrator.is_calibrated,
            progress=1.0 if self.calibrator.is_calibrated else 0.0,
            remaining_sec=0.0 if self.calibrator.is_calibrated else self.settings.CALIBRATION_DURATION_SEC,
            bone_lengths=self.calibrator.bone_lengths,
            estimated_distance_m=live_dist_m,
        )

        # Continuous Anti-Cheat & Identity Guard enforcement (Rule 1 + Rule 2)
        just_unlocked = False
        if self.fsm_state not in ("FACE_AUTH", "EMERGENCY", "SET_FINISHED"):
            if guard.status == "MULTI_PERSON":
                if self.fsm_state not in ("PAUSED_MULTI_PERSON", "PAUSED_IMPOSTOR"):
                    self._paused_from_state = self.fsm_state
                self.fsm_state = "PAUSED_MULTI_PERSON"
                self.movement_phase = "IDLE"
                self._reset_hold_timer()
                if (timestamp_sec - self._last_multi_person_tts_ts) >= self.settings.MULTI_PERSON_TTS_COOLDOWN_SEC:
                    self._last_multi_person_tts_ts = timestamp_sec
                self._last_tts = (
                    "ตรวจพบมีบุคคลอยู่หลายคนในเฟรม "
                    "กรุณาให้คนไข้ยืนหน้ากล้องเพียงลำพัง ระบบจึงจะนับรอบการฝึกต่อครับ"
                )
            elif guard.status == "IMPOSTOR":
                if self.fsm_state not in ("PAUSED_MULTI_PERSON", "PAUSED_IMPOSTOR"):
                    self._paused_from_state = self.fsm_state
                self.fsm_state = "PAUSED_IMPOSTOR"
                self.movement_phase = "IDLE"
                self._reset_hold_timer()
                self._last_tts = (
                    "คุณไม่ใช่คนไข้ที่ลงทะเบียนไว้ ระบบจะไม่นับรอบการฝึก "
                    "กรุณาให้คนไข้ตัวจริงเข้าฝึกครับ"
                )
            elif self.fsm_state in ("PAUSED_MULTI_PERSON", "PAUSED_IMPOSTOR") and guard.is_counting_allowed:
                # Auto-unlock after 1 person + 3 consecutive face matches
                resume_target = (
                    "ACTIVE"
                    if self.calibrator.is_calibrated
                    else (self._paused_from_state or "CALIBRATE_2S")
                )
                if resume_target in ("HOLD", "REP_DONE", "PAUSED_MULTI_PERSON", "PAUSED_IMPOSTOR"):
                    resume_target = "ACTIVE" if self.calibrator.is_calibrated else "CALIBRATE_2S"
                self.fsm_state = resume_target
                self.movement_phase = "READY" if resume_target == "ACTIVE" else "IDLE"
                self._paused_from_state = None
                self._reset_hold_timer()
                self._last_tts = "ยืนยันตัวตนคนไข้ถูกต้องแล้ว สามารถฝึกต่อได้ครับ"
                just_unlocked = True

        # Calibration stage (only advances when not paused by identity guard)
        if self.fsm_state == "CALIBRATE_2S":
            calib_state = self.calibrator.update(
                pts,
                scores,
                timestamp_sec,
                world_pts=world_pts,
                required_indices=self.rule.required_landmark_indices(),
            )
            self.movement_phase = "IDLE"
            if not just_unlocked:
                self._last_tts = calib_state.message_th
            if calib_state.is_calibrated:
                self.fsm_state = "ACTIVE"
                self.movement_phase = "READY"
                self._last_tts = f"เริ่มท่า {self.rule.name_th} ช้าๆ ให้ถึงเป้าหมายครับ"

        # Biomechanical 3D evaluation
        target_rom = self._target_rom_deg()
        analysis, emergency_reason = self.solver.analyze(
            pts=pts,
            rule=self.rule,
            calibrator=self.calibrator,
            timestamp_sec=timestamp_sec,
            target_rom_override=target_rom,
            world_pts=world_pts,
            scores=scores,
        )

        # Safety fall override (active once exercising or calibrating)
        if emergency_reason and self.fsm_state not in ("FACE_AUTH", "EMERGENCY"):
            self.trigger_emergency(emergency_reason)
            analysis.pain_alert = True

        # State transitions
        if self.fsm_state == "FACE_AUTH":
            self.movement_phase = "IDLE"
            self._last_tts = (
                face_enrollment.guidance_th
                if face_enrollment and face_enrollment.active
                else "กรุณามองกล้องเพื่อยืนยันตัวตนด้วยใบหน้า"
            )
        elif self.fsm_state == "ACTIVE":
            self._hold_timer_frozen = False
            self.movement_phase = self._update_active_movement_phase(analysis.current_rom_deg)
            if not analysis.is_form_valid:
                first_fault = analysis.compensation_detected[0]
                self._last_tts = THAI_COMPENSATION_PROMPTS.get(first_fault, "ปรับท่าทางให้ถูกต้อง")
                if not self._last_fault_counted and self._reached_target(analysis.current_rom_deg):
                    self.invalid_reps += 1
                    self._last_fault_counted = True
            else:
                self._last_fault_counted = False
                if self._reached_target(analysis.current_rom_deg):
                    self.fsm_state = "HOLD"
                    self.movement_phase = "HOLD"
                    self._hold_accumulated_sec = 0.0
                    self._hold_last_ts = float(timestamp_sec)
                    self._hold_timer_frozen = False
                    self.hold_remaining_sec = self.hold_sec
                    self._last_tts = f"ดีมากครับ! ค้างท่าไว้ {int(round(self.hold_remaining_sec))} วินาที"
                elif not just_unlocked:
                    self._last_tts = f"{self.rule.name_th} ({analysis.current_rom_deg:.1f}° / {target_rom:.0f}°)"

        elif self.fsm_state == "HOLD":
            self.movement_phase = "HOLD"
            dt = max(0.0, float(timestamp_sec) - (self._hold_last_ts if self._hold_last_ts is not None else float(timestamp_sec)))
            self._hold_last_ts = float(timestamp_sec)

            if not analysis.is_form_valid:
                # Freeze Hold Timer immediately (preserve accumulated hold progress until form is corrected)
                self._hold_timer_frozen = True
                first_fault = analysis.compensation_detected[0]
                self._last_tts = THAI_COMPENSATION_PROMPTS.get(first_fault, "รักษาท่าทางให้ตรงระหว่างค้างท่า")
            elif not self._still_in_hold_zone(analysis.current_rom_deg):
                # Exited below hysteresis boundary: reset hold timer and return to ACTIVE
                self.fsm_state = "ACTIVE"
                self.movement_phase = "ECCENTRIC"
                self._reset_hold_timer()
                self._last_tts = "แขน/ขาตกลงเล็กน้อย กรุณายกกลับเข้าสู่เป้าหมาย"
            else:
                self._hold_timer_frozen = False
                self._hold_accumulated_sec = min(self.hold_sec, self._hold_accumulated_sec + dt)
                self.hold_remaining_sec = max(0.0, self.hold_sec - self._hold_accumulated_sec)
                if self.hold_remaining_sec <= 0.0:
                    self.valid_reps += 1
                    self.fsm_state = "REP_DONE"
                    self.movement_phase = "REP_COMPLETE"
                    self._rep_complete_emitted = True
                    self._hold_last_ts = None
                    hold_label = f"{self.hold_sec:g}"
                    lower_phrase = (
                        "ค่อยๆ เอาขาลงช้าๆ ครับ"
                        if self.rule.metric_kind in ("knee_extension", "hip_flexion", "heel_raise")
                        else "ค่อยๆ เอาแขนลงช้าๆ ครับ"
                    )
                    if self.valid_reps >= self.target_reps:
                        self._last_tts = f"ครบ {hold_label} วินาทีแล้ว เก่งมากครับ! ครบ {self.target_reps} ครั้งแล้ว {lower_phrase}"
                    else:
                        self._last_tts = f"ครบ {hold_label} วินาทีแล้ว เยี่ยมมากครับ ครบ {self.valid_reps}/{self.target_reps} ครั้ง {lower_phrase}"
                else:
                    self._last_tts = f"ค้างท่าไว้ {self.hold_remaining_sec:.1f} วินาที"

        elif self.fsm_state == "REP_DONE":
            self._hold_timer_frozen = False
            if self._returned_to_rest(analysis.current_rom_deg):
                if self.valid_reps >= self.target_reps:
                    self.fsm_state = "SET_FINISHED"
                    self.movement_phase = "REP_COMPLETE"
                    self._last_tts = f"ยินดีด้วยครับ ทำท่า {self.rule.name_th} ครบทั้งเซตแล้ว"
                else:
                    self.fsm_state = "ACTIVE"
                    self.movement_phase = "READY"
                    self._reset_hold_timer()
                    self._rep_complete_emitted = False
                    self._last_tts = f"เริ่มครั้งที่ {self.valid_reps + 1} ได้เลยครับ"
            else:
                if self._still_in_hold_zone(analysis.current_rom_deg):
                    self.movement_phase = "REP_COMPLETE"
                else:
                    self._rep_complete_emitted = False
                    self.movement_phase = "ECCENTRIC"

        elif self.fsm_state == "SET_FINISHED":
            self.movement_phase = "REP_COMPLETE"

        self._prev_rom_deg = analysis.current_rom_deg

        # Sync hold progress & freeze state onto analysis schema as well
        hold_pct = self.hold_progress_pct
        analysis.hold_progress_pct = hold_pct
        analysis.hold_timer_frozen = self._hold_timer_frozen

        # UI highlight color
        highlight_color = "GREEN"
        if self.fsm_state in ("EMERGENCY", "PAUSED_IMPOSTOR") or analysis.pain_alert:
            highlight_color = "RED"
        elif (
            self.fsm_state == "PAUSED_MULTI_PERSON"
            or not analysis.is_form_valid
            or self._hold_timer_frozen
        ):
            highlight_color = "AMBER"

        return EdgeKioskTelemetryPayload(
            kiosk_state=self._kiosk_state_alias(),  # type: ignore[arg-type]
            fsm_state=self.fsm_state,  # type: ignore[arg-type]
            timestamp_ms=int(time.time() * 1000),
            fps=round(fps, 1),
            engine_backend="MediaPipe Heavy / RTMW3D-l + InsightFace + OneEuroFilter",
            camera_mode=camera_mode,
            movement_phase=self.movement_phase,
            hold_progress_pct=hold_pct,
            stability_score=analysis.stability_score,
            compensation_warnings=analysis.compensation_warnings,
            patient=self.patient,
            telemetry_analysis=analysis,
            rep_counter=RepCounterSchema(
                valid_reps=self.valid_reps,
                invalid_reps=self.invalid_reps,
                hold_remaining_sec=round(self.hold_remaining_sec, 2),
                hold_progress_pct=hold_pct,
                hold_timer_frozen=self._hold_timer_frozen,
                movement_phase=self.movement_phase,
                target_reps=self.target_reps,
            ),
            ui_directives=UIDirectivesSchema(
                skeleton_highlight_joint=self.prescription.target_joint or self.rule.target_joint,
                highlight_color=highlight_color,  # type: ignore[arg-type]
                show_target_arc=True,
            ),
            tts_voice_th=self._last_tts,
            calibration=calib_state,
            hand_status=hand_status,
            face_enrollment=face_enrollment,
            identity_guard=guard,
            keypoints_3d=keypoints_3d,
            pose_landmarks=pose_landmarks or [],
            pose_world_landmarks=pose_world_landmarks or [],
            frame_base64=frame_base64,
        )

