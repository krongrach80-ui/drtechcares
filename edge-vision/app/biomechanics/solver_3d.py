"""Clinical 3D joint angle solver, torso-plane & pelvic/shoulder tilt detectors, and safety guard."""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from app.biomechanics.calibrator import StandingCalibrator
from app.biomechanics.exercises import (
    ClinicalExerciseRule,
    are_hips_reliable,
    compute_shoulder_elevation_with_vertical_fallback,
)
from app.config import EdgeVisionSettings
from app.schemas.telemetry import BoneLengths3D, TelemetryAnalysisSchema
from app.vision.pose_3d import map_mediapipe33_to_wholebody133

logger = logging.getLogger("edge_vision.solver_3d")

try:
    from supabase import Client, create_client  # type: ignore
except ImportError:  # pragma: no cover
    Client = None  # type: ignore
    create_client = None  # type: ignore

COMPENSATION_WARNINGS_TH: Dict[str, str] = {
    "TRUNK_LEAN": "ลำตัวเอียงเกินกำหนด โปรดตั้งลำตัวให้ตรง",
    "SHOULDER_HIKE": "ห้ามยกหัวไหล่ขึ้นช่วย",
    "SHOULDER_TILT": "แนวหัวไหล่ทั้งสองข้างไม่สมดุล ระวังการเอียงไหล่ช่วย",
    "PELVIC_TILT": "แนวสะโพกเอียง ระวังการบิดหรือเอียงสะโพกชดเชยท่า",
    "ELBOW_FLEXION": "กรุณาเหยียดข้อศอกให้ตรงขณะยกแขน",
    "KNEE_VALGUS": "หัวเข่าบิดเข้าด้านใน โปรดจัดแนวเข่าให้ตรง",
    "HEEL_LIFT": "ส้นเท้าลอยจากพื้น โปรดวางส้นเท้าให้แนบสนิท",
    "TOO_FAST": "เคลื่อนไหวเร็วเกินไป กรุณาผ่อนจังหวะลงให้ช้าและสม่ำเสมอ",
    "LOW_VISIBILITY": "มองเห็นข้อต่อไม่ชัดเจน กรุณาขยับให้อยู่กลางกล้องในที่แสงสว่างเพียงพอ",
}


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


def _as_wholebody_133(pts: np.ndarray) -> np.ndarray:
    """Accept either (133, 3) Wholebody coords or (33, 3) MediaPipe coords seamlessly."""
    arr = np.asarray(pts, dtype=np.float64)
    if arr.shape[0] == 33:
        mapped, _ = map_mediapipe33_to_wholebody133(arr)
        return mapped
    return arr


def _coronal_tilt_deg(left_pt: np.ndarray, right_pt: np.ndarray) -> float:
    """3D-invariant coronal tilt angle (degrees) relative to camera/ground horizontal plane."""
    delta = left_pt - right_pt
    horiz = math.hypot(float(delta[0]), float(delta[2]))
    if horiz < 1e-6:
        return 90.0 if abs(float(delta[1])) > 1e-6 else 0.0
    return float(abs(math.degrees(math.atan2(abs(float(delta[1])), horiz))))


def compute_shoulder_tilt_deg(pts: np.ndarray) -> float:
    """Calculate bilateral shoulder coronal tilt in degrees (works with 33 or 133 landmarks)."""
    wb = _as_wholebody_133(pts)
    return _coronal_tilt_deg(wb[5], wb[6])


def compute_pelvic_tilt_deg(pts: np.ndarray) -> float:
    """Calculate bilateral pelvic (ASIS/hip) coronal tilt in degrees (33 or 133 landmarks)."""
    wb = _as_wholebody_133(pts)
    return _coronal_tilt_deg(wb[11], wb[12])


def compute_torso_plane(
    pts: np.ndarray,
    reference_up: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Construct the patient's 3D anatomical Torso Plane (origin=mid-hip, up=spine axis,
    lateral=left-to-right shoulder/hip axis, normal=anterior facing vector) from 3D landmarks.
    Eliminates perspective angle error when patient stands rotated sideways toward Kiosk.
    """
    wb = _as_wholebody_133(pts)
    mid_shoulder = 0.5 * (wb[5] + wb[6])
    mid_hip = 0.5 * (wb[11] + wb[12])
    spine_vec = mid_shoulder - mid_hip
    spine_norm = float(np.linalg.norm(spine_vec))
    up_axis = spine_vec / spine_norm if spine_norm > 1e-6 else np.array([0.0, -1.0, 0.0], dtype=np.float64)

    sh_lat = wb[5] - wb[6]
    hp_lat = wb[11] - wb[12]
    lateral_raw = 0.65 * sh_lat + 0.35 * hp_lat
    # Orthonormalize lateral axis against spine up_axis
    lateral_proj = lateral_raw - float(np.dot(lateral_raw, up_axis)) * up_axis
    lat_norm = float(np.linalg.norm(lateral_proj))
    lateral_axis = (
        lateral_proj / lat_norm if lat_norm > 1e-6 else np.array([1.0, 0.0, 0.0], dtype=np.float64)
    )

    anterior_raw = np.cross(lateral_axis, up_axis)
    ant_norm = float(np.linalg.norm(anterior_raw))
    normal = anterior_raw / ant_norm if ant_norm > 1e-6 else np.array([0.0, 0.0, -1.0], dtype=np.float64)

    ref_up = reference_up if reference_up is not None else np.array([0.0, -1.0, 0.0], dtype=np.float64)
    tilt_deg = angle_between_vectors_3d(up_axis, ref_up)
    yaw_deg = float(math.degrees(math.atan2(float(normal[0]), max(1e-6, -float(normal[2])))))

    return {
        "origin": mid_hip,
        "mid_shoulder": mid_shoulder,
        "up_axis": up_axis,
        "lateral_axis": lateral_axis,
        "normal": normal,
        "tilt_deg": float(tilt_deg),
        "yaw_deg": float(yaw_deg),
    }


class Biomechanics3DSolver:
    """Evaluates real 3D joint ROM, angular velocity, Torso/Shoulder/Pelvic posture checks, and fall safety."""

    def __init__(self, settings: EdgeVisionSettings) -> None:
        self.settings = settings
        self._prev_rom_deg: Optional[float] = None
        self._prev_hip_y: Optional[float] = None
        self._prev_ts: Optional[float] = None
        self._emergency_sent_for_episode = False
        self._rom_window: Deque[float] = deque(maxlen=max(5, int(settings.STABILITY_WINDOW_FRAMES)))

    def reset(self) -> None:
        self._prev_rom_deg = None
        self._prev_hip_y = None
        self._prev_ts = None
        self._emergency_sent_for_episode = False
        self._rom_window.clear()

    def _compute_primary_rom(
        self,
        pts: np.ndarray,
        rule: ClinicalExerciseRule,
        spine_vertical_unit: np.ndarray,
        baseline_heel_y: float,
        scores: Optional[np.ndarray] = None,
        pts_2d: Optional[np.ndarray] = None,
    ) -> float:
        mid_shoulder = 0.5 * (pts[5] + pts[6])
        mid_hip = 0.5 * (pts[11] + pts[12])

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
            return compute_shoulder_elevation_with_vertical_fallback(
                pts_3d=pts,
                side=side,
                scores=scores,
                min_visibility=float(self.settings.MIN_LANDMARK_VISIBILITY),
                pts_2d=pts_2d,
            )

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

    def _compute_stability_score(
        self,
        rom_deg: float,
        ang_vel: float,
        compensations: List[str],
        min_vis: float,
    ) -> float:
        """Compute 0-100 posture & motion smoothness score from ROM second-differences and form."""
        self._rom_window.append(float(rom_deg))
        jitter_penalty = 0.0
        if len(self._rom_window) >= 4:
            arr = np.asarray(self._rom_window, dtype=np.float64)
            second_diff = np.diff(arr, n=2)
            jitter_std = float(np.std(second_diff))
            jitter_penalty = min(35.0, jitter_std * 2.2)

        vel_excess = max(0.0, ang_vel - self.settings.MAX_ANGULAR_VELOCITY_DEG_S)
        vel_penalty = min(25.0, vel_excess * 0.25)
        comp_penalty = min(40.0, 16.0 * len([c for c in compensations if c != "NONE"]))
        vis_penalty = max(0.0, (self.settings.MIN_LANDMARK_VISIBILITY - min_vis) * 60.0)
        score = float(np.clip(100.0 - jitter_penalty - vel_penalty - comp_penalty - vis_penalty, 0.0, 100.0))
        return round(score, 1)

    def analyze(
        self,
        pts: np.ndarray,
        rule: ClinicalExerciseRule,
        calibrator: StandingCalibrator,
        timestamp_sec: float,
        target_rom_override: Optional[float] = None,
        world_pts: Optional[np.ndarray] = None,
        scores: Optional[np.ndarray] = None,
    ) -> Tuple[TelemetryAnalysisSchema, Optional[str]]:
        """
        Compute full 3D biomechanics from `pose_world_landmarks` (when available) or `(133, 3)` `pts`,
        detecting compensations (Trunk Lean, Shoulder Tilt/Hike, Pelvic Tilt, Elbow Flexion,
        Knee Valgus, Heel Lift, Speed, Visibility) plus Stability Score and Fall Emergency trigger.
        """
        pts_wb = _as_wholebody_133(pts)
        world_wb = _as_wholebody_133(world_pts) if world_pts is not None else pts_wb
        scores_wb: Optional[np.ndarray] = None
        if scores is not None and len(scores) > 0:
            sc_raw = np.asarray(scores, dtype=np.float64).reshape(-1)
            if sc_raw.shape[0] == 33:
                _, scores_wb = map_mediapipe33_to_wholebody133(
                    np.asarray(pts, dtype=np.float64),
                    sc_raw,
                )
            else:
                scores_wb = sc_raw

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
            base_l_sh_y = float(pts_wb[5, 1])
            base_r_sh_y = float(pts_wb[6, 1])
            base_l_heel_y = float(pts_wb[19, 1])
            base_r_heel_y = float(pts_wb[22, 1])

        # Visibility gating across active exercise landmarks
        req_indices = rule.required_landmark_indices()
        if scores_wb is not None and len(scores_wb) > 0:
            valid_req = [i for i in req_indices if i < scores_wb.shape[0]]
            min_vis = float(np.min(scores_wb[valid_req])) if valid_req else float(np.min(scores_wb))
        else:
            min_vis = 1.0
        visibility_ok = min_vis >= float(self.settings.MIN_LANDMARK_VISIBILITY)
        hips_visible = are_hips_reliable(
            world_wb,
            scores=scores_wb,
            min_visibility=float(self.settings.MIN_LANDMARK_VISIBILITY),
            pts_2d=pts_wb,
        )

        # True 3D ROM calculation using world landmarks (with vertical fallback when hips out of frame)
        world_heel_y = float(0.5 * (world_wb[19, 1] + world_wb[22, 1]))
        rom_deg = self._compute_primary_rom(
            world_wb,
            rule,
            spine_unit,
            world_heel_y if world_pts is not None else 0.5 * (base_l_heel_y + base_r_heel_y),
            scores=scores_wb,
            pts_2d=pts_wb,
        )
        target_rom = float(target_rom_override) if target_rom_override is not None else rule.min_angle_deg
        if rule.is_descending_target:
            denom = max(rule.rest_angle_deg - target_rom, 1e-3)
            pct = float(np.clip(((rule.rest_angle_deg - rom_deg) / denom) * 100.0, 0.0, 100.0))
        else:
            denom = max(target_rom - rule.rest_angle_deg, 1e-3)
            pct = float(np.clip(((rom_deg - rule.rest_angle_deg) / denom) * 100.0, 0.0, 100.0))

        # Angular velocity (deg/s) + hip drop velocity (units/s from image coords)
        hip_y = float(0.5 * (pts_wb[11, 1] + pts_wb[12, 1]))
        ang_vel = 0.0
        hip_drop_vel = 0.0
        if self._prev_ts is not None and timestamp_sec > self._prev_ts:
            dt = max(timestamp_sec - self._prev_ts, 1e-3)
            if self._prev_rom_deg is not None:
                ang_vel = abs(rom_deg - self._prev_rom_deg) / dt
            if hips_visible and self._prev_hip_y is not None:
                hip_drop_vel = (hip_y - self._prev_hip_y) / dt

        self._prev_rom_deg = rom_deg
        self._prev_hip_y = hip_y if hips_visible else None
        self._prev_ts = float(timestamp_sec)

        # 1. Shoulder Coronal Tilt, Torso Plane & TRUNK_LEAN > 10°
        shoulder_tilt_deg = compute_shoulder_tilt_deg(world_wb)
        if hips_visible:
            torso_plane = compute_torso_plane(world_wb, reference_up=spine_unit)
            trunk_tilt_deg = float(torso_plane["tilt_deg"])
            pelvic_tilt_deg = compute_pelvic_tilt_deg(world_wb)
        else:
            # Seated close to webcam without visible hips: estimate upper trunk lean from shoulder level
            trunk_tilt_deg = round(shoulder_tilt_deg * 0.7, 1) if rule.category in ("shoulder", "arm", "neck") else 0.0
            pelvic_tilt_deg = 0.0

        if rule.side == "left":
            hike_cm = max(0.0, (base_l_sh_y - float(pts_wb[5, 1])) * cm_per_unit)
        elif rule.side == "right":
            hike_cm = max(0.0, (base_r_sh_y - float(pts_wb[6, 1])) * cm_per_unit)
        else:
            hike_cm = max(
                0.0,
                max(base_l_sh_y - float(pts_wb[5, 1]), base_r_sh_y - float(pts_wb[6, 1])) * cm_per_unit,
            )

        # Active side elbow extension angle (11/13/15 equivalent -> 5/7/9 left, 6/8/10 right)
        side = rule.side if rule.side in ("left", "right") else "right"
        sh_idx = 5 if side == "left" else 6
        el_idx = 7 if side == "left" else 8
        wr_idx = 9 if side == "left" else 10
        elbow_flexion_deg = joint_angle_3d(world_wb[sh_idx], world_wb[el_idx], world_wb[wr_idx])

        # 3. KNEE_VALGUS > 8.0° (medial knee collapse in frontal plane relative to hip->ankle line)
        l_valgus = angle_between_vectors_3d(
            np.array([world_wb[13, 0] - world_wb[11, 0], world_wb[13, 1] - world_wb[11, 1], 0.0]),
            np.array([world_wb[15, 0] - world_wb[11, 0], world_wb[15, 1] - world_wb[11, 1], 0.0]),
        )
        r_valgus = angle_between_vectors_3d(
            np.array([world_wb[14, 0] - world_wb[12, 0], world_wb[14, 1] - world_wb[12, 1], 0.0]),
            np.array([world_wb[16, 0] - world_wb[12, 0], world_wb[16, 1] - world_wb[12, 1], 0.0]),
        )
        l_inward = (world_wb[11, 0] + world_wb[15, 0]) * 0.5 - world_wb[13, 0]
        r_inward = world_wb[14, 0] - (world_wb[12, 0] + world_wb[16, 0]) * 0.5
        knee_valgus_deg = max(
            l_valgus if l_inward > 0.008 else 0.0,
            r_valgus if r_inward > 0.008 else 0.0,
        )

        # 4. HEEL_LIFT > 2.5 cm using foot keypoints 19 (L heel), 22 (R heel)
        l_heel_lift = max(0.0, (base_l_heel_y - float(pts_wb[19, 1])) * cm_per_unit)
        r_heel_lift = max(0.0, (base_r_heel_y - float(pts_wb[22, 1])) * cm_per_unit)
        heel_lift_cm = max(l_heel_lift, r_heel_lift)

        # Gather active anti-cheating violations configured for this exercise
        compensations: List[str] = []
        checks = rule.anti_cheat_checks
        if not visibility_ok:
            compensations.append("LOW_VISIBILITY")
        if "TRUNK_LEAN" in checks and trunk_tilt_deg > self.settings.TRUNK_LEAN_MAX_DEG:
            compensations.append("TRUNK_LEAN")
        if "SHOULDER_HIKE" in checks and (
            hike_cm > self.settings.SHOULDER_HIKE_MAX_CM
            or shoulder_tilt_deg > self.settings.SHOULDER_TILT_MAX_DEG
        ):
            compensations.append("SHOULDER_HIKE")
            if shoulder_tilt_deg > self.settings.SHOULDER_TILT_MAX_DEG:
                compensations.append("SHOULDER_TILT")
        if (
            hips_visible
            and ("PELVIC_TILT" in checks or rule.category in ("leg", "balance", "trunk", "shoulder"))
            and pelvic_tilt_deg > self.settings.PELVIC_TILT_MAX_DEG
        ):
            compensations.append("PELVIC_TILT")
        if (
            rule.requires_straight_elbow
            and rom_deg > 25.0
            and elbow_flexion_deg < self.settings.ELBOW_EXTENSION_MIN_DEG
        ):
            compensations.append("ELBOW_FLEXION")
        if "KNEE_VALGUS" in checks and knee_valgus_deg > self.settings.KNEE_VALGUS_MAX_DEG:
            compensations.append("KNEE_VALGUS")
        if "HEEL_LIFT" in checks and heel_lift_cm > self.settings.HEEL_LIFT_MAX_CM:
            compensations.append("HEEL_LIFT")
        if "TOO_FAST" in checks and ang_vel > self.settings.MAX_ANGULAR_VELOCITY_DEG_S:
            compensations.append("TOO_FAST")

        is_form_valid = len(compensations) == 0
        warnings_th = [COMPENSATION_WARNINGS_TH[c] for c in compensations if c in COMPENSATION_WARNINGS_TH]
        if not compensations:
            compensations = ["NONE"]

        stability_score = self._compute_stability_score(
            rom_deg=rom_deg,
            ang_vel=ang_vel,
            compensations=compensations,
            min_vis=min_vis,
        )

        # Safety / Fall override: sudden downward hip/head drop or hip collapsing near floor
        emergency_reason: Optional[str] = None
        if hips_visible and (
            (hip_y > 0.80 and trunk_tilt_deg > 28.0)
            or (hip_drop_vel > self.settings.FALL_DROP_VELOCITY_THRESHOLD)
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
            torso_plane_tilt_deg=round(trunk_tilt_deg, 1),
            shoulder_tilt_deg=round(shoulder_tilt_deg, 1),
            pelvic_tilt_deg=round(pelvic_tilt_deg, 1),
            shoulder_hike_cm=round(hike_cm, 2),
            elbow_flexion_deg=round(elbow_flexion_deg, 1),
            knee_valgus_deg=round(knee_valgus_deg, 1),
            heel_lift_cm=round(heel_lift_cm, 2),
            angular_velocity_deg_s=round(ang_vel, 1),
            min_landmark_visibility=round(min_vis, 3),
            visibility_ok=visibility_ok,
            stability_score=stability_score,
            is_form_valid=is_form_valid,
            compensation_detected=compensations,
            compensation_warnings=warnings_th,
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

