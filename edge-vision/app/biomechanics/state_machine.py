"""Clinical Finite State Machine: FACE_AUTH -> CALIBRATE_2S -> ACTIVE -> HOLD -> REP_DONE | EMERGENCY."""

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
    PatientMatchSchema,
    RepCounterSchema,
    UIDirectivesSchema,
)

logger = logging.getLogger("edge_vision.fsm")

THAI_COMPENSATION_PROMPTS = {
    "TRUNK_LEAN": "โปรดหลังตรง อย่าเอียงลำตัวช่วยยก",
    "SHOULDER_HIKE": "ลดหัวไหล่ลง ระวังอย่ายักไหล่ช่วย",
    "KNEE_VALGUS": "จัดแนวหัวเข่าให้ตรง อย่าให้เข่าบิดเข้าด้านใน",
    "HEEL_LIFT": "กรุณาวางส้นเท้าทั้งสองข้างให้ติดพื้น",
    "TOO_FAST": "เคลื่อนไหวเร็วเกินไป กรุณาค่อยๆ ทำช้าๆ เพื่อความปลอดภัย",
}


class ClinicalRehabFSM:
    """
    Controls session flow across all 7 clinical states and generates Thai voice feedback
    plus React UI telemetry directives at 30-60 Hz.
    """

    def __init__(self, settings: EdgeVisionSettings) -> None:
        self.settings = settings
        self.calibrator = StandingCalibrator(
            duration_sec=settings.CALIBRATION_DURATION_SEC,
            stability_max_tilt_deg=settings.CALIBRATION_STABILITY_MAX_TILT_DEG,
        )
        self.solver = Biomechanics3DSolver(settings)

        self.fsm_state: str = "FACE_AUTH"
        self.patient: Optional[PatientMatchSchema] = None
        self.prescription = ExercisePrescriptionSchema()
        self.rule: ClinicalExerciseRule = resolve_exercise_rule(self.prescription.exercise_id)

        self.valid_reps: int = 0
        self.invalid_reps: int = 0
        self.target_reps: int = int(self.prescription.target_reps)
        self.hold_sec: float = float(self.prescription.hold_sec)
        self.hold_remaining_sec: float = self.hold_sec

        self._hold_start_ts: Optional[float] = None
        self._last_fault_counted = False
        self._emergency_reason: Optional[str] = None
        self._paused_from_state: Optional[str] = None
        self._last_multi_person_tts_ts: float = -10.0
        self._last_tts: str = "กรุณามองกล้องเพื่อยืนยันตัวตนด้วยใบหน้า"

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
        self.hold_remaining_sec = self.hold_sec
        self._hold_start_ts = None
        self._last_fault_counted = False
        self._emergency_reason = None
        self.solver.reset()

        if start_calibration:
            self.start_calibration()
        elif self.calibrator.is_calibrated:
            self.fsm_state = "ACTIVE"

    def start_calibration(self) -> None:
        self.calibrator.reset()
        self.solver.reset()
        self.hold_remaining_sec = self.hold_sec
        self._hold_start_ts = None
        self._paused_from_state = None
        self._last_multi_person_tts_ts = -10.0
        self.fsm_state = "CALIBRATE_2S"
        self._last_tts = "กรุณายืนตัวตรงนิ่ง 2 วินาที เพื่อคาลิเบรตสัดส่วนร่างกาย"

    def trigger_emergency(self, reason: str) -> None:
        self.fsm_state = "EMERGENCY"
        self._emergency_reason = reason
        self._last_tts = "ตรวจพบความเสี่ยงฉุกเฉิน ระบบหยุดการฝึกและแจ้งเตือนเจ้าหน้าที่แล้ว"
        pid = self.patient.id if self.patient else None
        self.solver.dispatch_emergency_alert_supabase(pid, reason)

    def reset_session(self, return_to_auth: bool = True) -> None:
        self.calibrator.reset()
        self.solver.reset()
        self.valid_reps = 0
        self.invalid_reps = 0
        self.hold_remaining_sec = self.hold_sec
        self._hold_start_ts = None
        self._last_fault_counted = False
        self._emergency_reason = None
        self._paused_from_state = None
        self._last_multi_person_tts_ts = -10.0
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

    def _reached_target(self, rom_deg: float) -> bool:
        target = float(self.prescription.target_rom_deg or self.rule.min_angle_deg)
        if self.rule.is_descending_target:
            return rom_deg <= target + 6.0
        return rom_deg >= target - 8.0

    def _returned_to_rest(self, rom_deg: float) -> bool:
        rest = self.rule.rest_angle_deg
        if self.rule.is_descending_target:
            return rom_deg >= rest - 18.0
        return rom_deg <= rest + 22.0

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
    ) -> EdgeKioskTelemetryPayload:
        """Advance FSM one frame and emit complete WebSocket telemetry payload."""
        guard = identity_guard or IdentityGuardSchema()

        calib_state = CalibrationStateSchema(
            is_calibrated=self.calibrator.is_calibrated,
            progress=1.0 if self.calibrator.is_calibrated else 0.0,
            remaining_sec=0.0 if self.calibrator.is_calibrated else self.settings.CALIBRATION_DURATION_SEC,
            bone_lengths=self.calibrator.bone_lengths,
        )

        # Continuous Anti-Cheat & Identity Guard enforcement (Rule 1 + Rule 2)
        just_unlocked = False
        if self.fsm_state not in ("FACE_AUTH", "EMERGENCY", "SET_FINISHED"):
            if guard.status == "MULTI_PERSON":
                if self.fsm_state not in ("PAUSED_MULTI_PERSON", "PAUSED_IMPOSTOR"):
                    self._paused_from_state = self.fsm_state
                self.fsm_state = "PAUSED_MULTI_PERSON"
                self._hold_start_ts = None
                self.hold_remaining_sec = self.hold_sec
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
                self._hold_start_ts = None
                self.hold_remaining_sec = self.hold_sec
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
                self._paused_from_state = None
                self._hold_start_ts = None
                self.hold_remaining_sec = self.hold_sec
                self._last_tts = "ยืนยันตัวตนคนไข้ถูกต้องแล้ว สามารถฝึกต่อได้ครับ"
                just_unlocked = True

        # Calibration stage (only advances when not paused by identity guard)
        if self.fsm_state == "CALIBRATE_2S":
            calib_state = self.calibrator.update(pts, scores, timestamp_sec)
            if not just_unlocked:
                self._last_tts = calib_state.message_th
            if calib_state.is_calibrated:
                self.fsm_state = "ACTIVE"
                self._last_tts = f"เริ่มท่า {self.rule.name_th} ช้าๆ ให้ถึงเป้าหมายครับ"

        # Biomechanical 3D evaluation
        target_rom = float(self.prescription.target_rom_deg or self.rule.min_angle_deg)
        analysis, emergency_reason = self.solver.analyze(
            pts=pts,
            rule=self.rule,
            calibrator=self.calibrator,
            timestamp_sec=timestamp_sec,
            target_rom_override=target_rom,
        )

        # Safety fall override (active once exercising or calibrating)
        if emergency_reason and self.fsm_state not in ("FACE_AUTH", "EMERGENCY"):
            self.trigger_emergency(emergency_reason)
            analysis.pain_alert = True

        # State transitions
        if self.fsm_state == "FACE_AUTH":
            self._last_tts = (
                face_enrollment.guidance_th
                if face_enrollment and face_enrollment.active
                else "กรุณามองกล้องเพื่อยืนยันตัวตนด้วยใบหน้า"
            )
        elif self.fsm_state == "ACTIVE":
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
                    self._hold_start_ts = timestamp_sec
                    self.hold_remaining_sec = self.hold_sec
                    self._last_tts = f"ดีมากครับ! ค้างท่าไว้ {int(round(self.hold_remaining_sec))} วินาที"
                elif not just_unlocked:
                    self._last_tts = f"{self.rule.name_th} ({analysis.current_rom_deg:.0f}° / {target_rom:.0f}°)"

        elif self.fsm_state == "HOLD":
            if not analysis.is_form_valid:
                # Reset hold if form breaks significantly during isometric hold
                self._hold_start_ts = timestamp_sec
                self.hold_remaining_sec = self.hold_sec
                first_fault = analysis.compensation_detected[0]
                self._last_tts = THAI_COMPENSATION_PROMPTS.get(first_fault, "รักษาท่าทางให้ตรงระหว่างค้างท่า")
            elif not self._reached_target(analysis.current_rom_deg):
                self.fsm_state = "ACTIVE"
                self._hold_start_ts = None
                self.hold_remaining_sec = self.hold_sec
                self._last_tts = "แขน/ขาตกลงเล็กน้อย กรุณายกกลับเข้าสู่เป้าหมาย"
            else:
                elapsed = max(0.0, timestamp_sec - (self._hold_start_ts or timestamp_sec))
                self.hold_remaining_sec = max(0.0, self.hold_sec - elapsed)
                if self.hold_remaining_sec <= 0.0:
                    self.valid_reps += 1
                    self.fsm_state = "REP_DONE"
                    self._hold_start_ts = None
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
            if self._returned_to_rest(analysis.current_rom_deg):
                if self.valid_reps >= self.target_reps:
                    self.fsm_state = "SET_FINISHED"
                    self._last_tts = f"ยินดีด้วยครับ ทำท่า {self.rule.name_th} ครบทั้งเซตแล้ว"
                else:
                    self.fsm_state = "ACTIVE"
                    self.hold_remaining_sec = self.hold_sec
                    self._last_tts = f"เริ่มครั้งที่ {self.valid_reps + 1} ได้เลยครับ"

        # UI highlight color
        highlight_color = "GREEN"
        if self.fsm_state in ("EMERGENCY", "PAUSED_IMPOSTOR") or analysis.pain_alert:
            highlight_color = "RED"
        elif self.fsm_state == "PAUSED_MULTI_PERSON" or not analysis.is_form_valid:
            highlight_color = "AMBER"

        return EdgeKioskTelemetryPayload(
            kiosk_state=self._kiosk_state_alias(),  # type: ignore[arg-type]
            fsm_state=self.fsm_state,  # type: ignore[arg-type]
            timestamp_ms=int(time.time() * 1000),
            fps=round(fps, 1),
            engine_backend="RTMW3D-l + InsightFace + OneEuroFilter",
            camera_mode=camera_mode,
            patient=self.patient,
            telemetry_analysis=analysis,
            rep_counter=RepCounterSchema(
                valid_reps=self.valid_reps,
                invalid_reps=self.invalid_reps,
                hold_remaining_sec=round(self.hold_remaining_sec, 2),
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
            frame_base64=frame_base64,
        )

