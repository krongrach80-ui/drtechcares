"""Clinical 3D joint angle solver, anti-cheating detectors, and emergency fall alert."""

from __future__ import annotations

import logging
import math
from typing import List, Optional, Tuple

import numpy as np

from app.biomechanics.calibrator import StandingCalibrator
from app.biomechanics.exercises import ClinicalExerciseRule
from app.config import EdgeVisionSettings
from app.schemas.telemetry import BoneLengths3D, TelemetryAnalysisSchema

logger = logging.getLogger("edge_vision.solver_3d")

try:
    from supabase import Client, create_client  # type: ignore
except ImportError:  # pragma: no cover
    Client = None  # type: ignore
    create_client = None  # type: ignore


def angle_between_vectors_3d(u: np.ndarray, v: np.ndarray) -> float:
    """Compute true 3D Euclidean vector angle (in degrees) using arccos(u·v / ||u||||v||)."""
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu < 1e-6 or nv < 1e-6:
        return 0.0
    cos_theta = float(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))
    return float(math.degrees(math.acos(cos_theta)))


def joint_angle_3d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Compute internal 3D angle at vertex b formed by segments (b->a) and (b->c)."""
    return angle_between_vectors_3d(a - b, c - b)


class Biomechanics3DSolver:
    """Evaluates real 3D joint ROM, angular velocity, 5 compensation patterns, and fall safety."""

    def __init__(self, settings: EdgeVisionSettings) -> None:
        self.settings = settings
        self._prev_rom_deg: Optional[float] = None
        self._prev_hip_y: Optional[float] = None
        self._prev_ts: Optional[float] = None
        self._emergency_sent_for_episode = False

    def reset(self) -> None:
        self._prev_rom_deg = None
        self._prev_hip_y = None
        self._prev_ts = None
        self._emergency_sent_for_episode = False

    def _compute_primary_rom(
        self,
        pts: np.ndarray,
        rule: ClinicalExerciseRule,
        spine_vertical_unit: np.ndarray,
        baseline_heel_y: float,
    ) -> float:
        mid_shoulder = 0.5 * (pts[5] + pts[6])
        mid_hip = 0.5 * (pts[11] + pts[12])
        trunk_down = mid_hip - mid_shoulder

        side = rule.side if rule.side in ("left", "right") else "right"
        sh_idx = 5 if side == "left" else 6
        el_idx = 7 if side == "left" else 8
        wr_idx = 9 if side == "left" else 10
        hp_idx = 11 if side == "left" else 12
        kn_idx = 13 if side == "left" else 14
        an_idx = 15 if side == "left" else 16
        toe_idx = 17 if side == "left" else 20
        heel_idx = 19 if side == "left" else 22

        kind = rule.metric_kind
        if kind == "shoulder_elevation":
            upper_arm = pts[el_idx] - pts[sh_idx]
            return angle_between_vectors_3d(upper_arm, trunk_down)

        if kind == "elbow_flexion":
            return joint_angle_3d(pts[sh_idx], pts[el_idx], pts[wr_idx])

        if kind == "knee_extension":
            if rule.id == "ex-26":
                left_k = joint_angle_3d(pts[11], pts[13], pts[15])
                right_k = joint_angle_3d(pts[12], pts[14], pts[16])
                return 0.5 * (left_k + right_k)
            return joint_angle_3d(pts[hp_idx], pts[kn_idx], pts[an_idx])

        if kind == "hip_flexion":
            # Internal 3D angle shoulder -> hip -> knee (standing ~175°, lifted knee ~85°)
            return joint_angle_3d(pts[sh_idx], pts[hp_idx], pts[kn_idx])

        if kind == "heel_raise":
            # Measure foot pitch elevation angle in 3D from heel(19/22)->toe(17/20) vs horizontal + heel lift
            foot_vec = pts[toe_idx] - pts[heel_idx]
            horiz = np.array([foot_vec[0], 0.0, foot_vec[2]], dtype=np.float64)
            pitch_deg = angle_between_vectors_3d(foot_vec, horiz)
            lift_bonus = max(0.0, (baseline_heel_y - pts[heel_idx, 1]) * 350.0)
            return float(np.clip(pitch_deg + lift_bonus, 0.0, 95.0))

        if kind == "neck_lateral_tilt":
            head_axis = pts[0] - mid_shoulder
            return angle_between_vectors_3d(head_axis, spine_vertical_unit)

        # balance_trunk_sway / trunk_flexion
        live_spine = mid_shoulder - mid_hip
        return angle_between_vectors_3d(live_spine, spine_vertical_unit)

    def analyze(
        self,
        pts: np.ndarray,
        rule: ClinicalExerciseRule,
        calibrator: StandingCalibrator,
        timestamp_sec: float,
        target_rom_override: Optional[float] = None,
    ) -> Tuple[TelemetryAnalysisSchema, Optional[str]]:
        """
        Compute full 3D biomechanics, 5 anti-cheat compensations, and fall emergency trigger.
        Returns `(TelemetryAnalysisSchema, emergency_reason_or_none)`.
        """
        bones: Optional[BoneLengths3D] = calibrator.bone_lengths
        if bones is not None:
            spine_unit = np.array(bones.spine_vertical_unit, dtype=np.float64)
            cm_per_unit = bones.cm_per_unit
            base_l_sh_y = calibrator.baseline_left_shoulder_y
            base_r_sh_y = calibrator.baseline_right_shoulder_y
            base_l_heel_y = calibrator.baseline_left_heel_y
            base_r_heel_y = calibrator.baseline_right_heel_y
        else:
            spine_unit = np.array([0.0, -1.0, 0.0], dtype=np.float64)
            cm_per_unit = 196.0
            base_l_sh_y = float(pts[5, 1])
            base_r_sh_y = float(pts[6, 1])
            base_l_heel_y = float(pts[19, 1])
            base_r_heel_y = float(pts[22, 1])

        rom_deg = self._compute_primary_rom(pts, rule, spine_unit, 0.5 * (base_l_heel_y + base_r_heel_y))
        target_rom = float(target_rom_override) if target_rom_override is not None else rule.min_angle_deg
        if rule.is_descending_target:
            denom = max(rule.rest_angle_deg - target_rom, 1e-3)
            pct = float(np.clip(((rule.rest_angle_deg - rom_deg) / denom) * 100.0, 0.0, 100.0))
        else:
            denom = max(target_rom - rule.rest_angle_deg, 1e-3)
            pct = float(np.clip(((rom_deg - rule.rest_angle_deg) / denom) * 100.0, 0.0, 100.0))

        # Angular velocity (deg/s) + hip drop velocity (units/s)
        hip_y = float(0.5 * (pts[11, 1] + pts[12, 1]))
        ang_vel = 0.0
        hip_drop_vel = 0.0
        if self._prev_ts is not None and timestamp_sec > self._prev_ts:
            dt = max(timestamp_sec - self._prev_ts, 1e-3)
            if self._prev_rom_deg is not None:
                ang_vel = abs(rom_deg - self._prev_rom_deg) / dt
            if self._prev_hip_y is not None:
                hip_drop_vel = (hip_y - self._prev_hip_y) / dt

        self._prev_rom_deg = rom_deg
        self._prev_hip_y = hip_y
        self._prev_ts = float(timestamp_sec)

        # 1. TRUNK_LEAN > 10°
        mid_shoulder = 0.5 * (pts[5] + pts[6])
        mid_hip = 0.5 * (pts[11] + pts[12])
        trunk_tilt_deg = angle_between_vectors_3d(mid_shoulder - mid_hip, spine_unit)

        # 2. SHOULDER_HIKE > 3.0 cm (active shoulder rises vertically above standing baseline)
        if rule.side == "left":
            hike_cm = max(0.0, (base_l_sh_y - float(pts[5, 1])) * cm_per_unit)
        elif rule.side == "right":
            hike_cm = max(0.0, (base_r_sh_y - float(pts[6, 1])) * cm_per_unit)
        else:
            hike_cm = max(
                0.0,
                max(base_l_sh_y - float(pts[5, 1]), base_r_sh_y - float(pts[6, 1])) * cm_per_unit,
            )

        # 3. KNEE_VALGUS > 8.0° (medial knee collapse in frontal plane relative to hip->ankle line)
        l_valgus = angle_between_vectors_3d(
            np.array([pts[13, 0] - pts[11, 0], pts[13, 1] - pts[11, 1], 0.0]),
            np.array([pts[15, 0] - pts[11, 0], pts[15, 1] - pts[11, 1], 0.0]),
        )
        r_valgus = angle_between_vectors_3d(
            np.array([pts[14, 0] - pts[12, 0], pts[14, 1] - pts[12, 1], 0.0]),
            np.array([pts[16, 0] - pts[12, 0], pts[16, 1] - pts[12, 1], 0.0]),
        )
        # Penalize inward medial shift (left knee moving -x toward midline, right knee moving +x toward midline)
        l_inward = (pts[11, 0] + pts[15, 0]) * 0.5 - pts[13, 0]
        r_inward = pts[14, 0] - (pts[12, 0] + pts[16, 0]) * 0.5
        knee_valgus_deg = max(
            l_valgus if l_inward > 0.008 else 0.0,
            r_valgus if r_inward > 0.008 else 0.0,
        )

        # 4. HEEL_LIFT > 2.5 cm using keypoints 19 (L heel), 22 (R heel) & toes (17, 20)
        l_heel_lift = max(0.0, (base_l_heel_y - float(pts[19, 1])) * cm_per_unit)
        r_heel_lift = max(0.0, (base_r_heel_y - float(pts[22, 1])) * cm_per_unit)
        heel_lift_cm = max(l_heel_lift, r_heel_lift)

        # Gather active anti-cheating violations configured for this exercise
        compensations: List[str] = []
        checks = rule.anti_cheat_checks
        if "TRUNK_LEAN" in checks and trunk_tilt_deg > self.settings.TRUNK_LEAN_MAX_DEG:
            compensations.append("TRUNK_LEAN")
        if "SHOULDER_HIKE" in checks and hike_cm > self.settings.SHOULDER_HIKE_MAX_CM:
            compensations.append("SHOULDER_HIKE")
        if "KNEE_VALGUS" in checks and knee_valgus_deg > self.settings.KNEE_VALGUS_MAX_DEG:
            compensations.append("KNEE_VALGUS")
        if "HEEL_LIFT" in checks and heel_lift_cm > self.settings.HEEL_LIFT_MAX_CM:
            compensations.append("HEEL_LIFT")
        if "TOO_FAST" in checks and ang_vel > self.settings.MAX_ANGULAR_VELOCITY_DEG_S:
            compensations.append("TOO_FAST")

        is_form_valid = len(compensations) == 0
        if not compensations:
            compensations = ["NONE"]

        # Safety / Fall override: sudden downward hip/head drop or hip collapsing near floor
        emergency_reason: Optional[str] = None
        if (hip_y > 0.80 and trunk_tilt_deg > 28.0) or (
            hip_drop_vel > self.settings.FALL_DROP_VELOCITY_THRESHOLD
        ):
            emergency_reason = (
                f"Fall/Collapse Safety Alert on {rule.code} "
                f"(hip_y={hip_y:.2f}, drop_vel={hip_drop_vel:.2f}, tilt={trunk_tilt_deg:.1f}°)"
            )

        schema = TelemetryAnalysisSchema(
            current_rom_deg=round(rom_deg, 1),
            target_rom_deg=round(target_rom, 1),
            rom_completion_pct=round(pct, 1),
            trunk_tilt_deg=round(trunk_tilt_deg, 1),
            shoulder_hike_cm=round(hike_cm, 2),
            knee_valgus_deg=round(knee_valgus_deg, 1),
            heel_lift_cm=round(heel_lift_cm, 2),
            angular_velocity_deg_s=round(ang_vel, 1),
            is_form_valid=is_form_valid,
            compensation_detected=compensations,
            pain_alert=emergency_reason is not None,
        )
        return schema, emergency_reason

    def dispatch_emergency_alert_supabase(
        self,
        patient_id: Optional[str],
        reason: str,
    ) -> bool:
        """Write fall/pain emergency record directly into Supabase `public.emergency_alerts`."""
        if self._emergency_sent_for_episode:
            return False
        self._emergency_sent_for_episode = True

        key = self.settings.SUPABASE_SERVICE_ROLE_KEY or self.settings.SUPABASE_ANON_KEY
        if create_client is None or not self.settings.SUPABASE_URL or not key:
            logger.warning("Emergency alert logged locally (Supabase credentials absent): %s", reason)
            return False

        try:
            client = create_client(self.settings.SUPABASE_URL, key)
            payload = {
                "kiosk_id": self.settings.KIOSK_ID,
                "patient_id": patient_id,
                "reason": reason[:500],
                "status": "OPEN",
            }
            client.table("emergency_alerts").insert(payload).execute()
            logger.critical("Emergency alert persisted to Supabase: %s", reason)
            return True
        except Exception as exc:  # pragma: no cover
            logger.error("Failed persisting emergency_alerts to Supabase: %s", exc)
            return False

