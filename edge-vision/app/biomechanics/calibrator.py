"""2-Second Standing Calibration capturing patient-specific 3D bone proportions and camera distance."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from app.schemas.telemetry import BoneLengths3D, CalibrationStateSchema

CALIBRATION_JOINT_LABELS: Dict[int, str] = {
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle",
}


class StandingCalibrator:
    """
    Requires the patient to stand upright facing the webcam for 2.0 seconds
    with clear joint visibility and safe camera distance before counting reps,
    extracting anthropometric 3D bone lengths and baseline posture metrics.
    """

    def __init__(
        self,
        duration_sec: float = 2.0,
        stability_max_tilt_deg: float = 12.0,
        min_visibility: float = 0.70,
        min_torso_height_ratio: float = 0.16,
        max_torso_height_ratio: float = 0.55,
    ) -> None:
        self.duration_sec = float(duration_sec)
        self.stability_max_tilt_deg = float(stability_max_tilt_deg)
        self.min_visibility = float(min_visibility)
        self.min_torso_height_ratio = float(min_torso_height_ratio)
        self.max_torso_height_ratio = float(max_torso_height_ratio)
        self._start_ts: Optional[float] = None
        self._samples: List[np.ndarray] = []
        self._world_samples: List[np.ndarray] = []
        self._bone_lengths: Optional[BoneLengths3D] = None
        self.baseline_left_shoulder_y: float = 0.275
        self.baseline_right_shoulder_y: float = 0.275
        self.baseline_left_heel_y: float = 0.935
        self.baseline_right_heel_y: float = 0.935
        self.baseline_shoulder_tilt_deg: float = 0.0
        self.baseline_pelvic_tilt_deg: float = 0.0
        self.baseline_shoulder_delta_y: float = 0.0
        self.baseline_spine_angle_deg: float = 0.0
        self.estimated_distance_m: float = 2.0

    @property
    def is_calibrated(self) -> bool:
        return self._bone_lengths is not None

    @property
    def bone_lengths(self) -> Optional[BoneLengths3D]:
        return self._bone_lengths

    def reset(self) -> None:
        self._start_ts = None
        self._samples.clear()
        self._world_samples.clear()
        self._bone_lengths = None

    @staticmethod
    def _dist3(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    @staticmethod
    def _coronal_tilt_deg(left_pt: np.ndarray, right_pt: np.ndarray) -> float:
        """Compute coronal tilt from horizontal axis in degrees (level = 0°)."""
        delta = left_pt - right_pt
        horiz = math.hypot(float(delta[0]), float(delta[2]))
        if horiz < 1e-6:
            return 90.0 if abs(float(delta[1])) > 1e-6 else 0.0
        return float(abs(math.degrees(math.atan2(abs(float(delta[1])), horiz))))

    @staticmethod
    def _has_visible_hips(pts: np.ndarray, scores: np.ndarray, min_visibility: float) -> bool:
        """Return True when both hips are clearly visible and anatomically below shoulders."""
        if scores.size < 17:
            return False
        if float(scores[11]) < min_visibility or float(scores[12]) < min_visibility:
            return False
        mid_hip_y = 0.5 * float(pts[11, 1] + pts[12, 1])
        mid_sh_y = 0.5 * float(pts[5, 1] + pts[6, 1])
        return mid_hip_y <= 0.96 and (mid_hip_y - mid_sh_y) >= 0.08

    @staticmethod
    def _spine_tilt_from_vertical(pts: np.ndarray) -> float:
        mid_shoulder = 0.5 * (pts[5] + pts[6])
        mid_hip = 0.5 * (pts[11] + pts[12])
        spine = mid_shoulder - mid_hip
        norm = float(np.linalg.norm(spine))
        if norm < 1e-6:
            return 90.0
        # Image-space vertical up is (0, -1, 0)
        up = np.array([0.0, -1.0, 0.0], dtype=np.float64)
        cos_val = float(np.clip(np.dot(spine / norm, up), -1.0, 1.0))
        return float(math.degrees(math.acos(cos_val)))

    def evaluate_framing_and_distance(
        self,
        pts: np.ndarray,
        scores: np.ndarray,
        required_indices: Optional[Sequence[int]] = None,
    ) -> Tuple[bool, str, float, float, List[str]]:
        """
        Verify required landmarks meet visibility >= threshold and patient stands within
        recommended distance range from Kiosk camera.
        Returns `(ok, status_code, estimated_distance_m, min_vis, missing_joints)`.
        """
        req = tuple(required_indices) if required_indices is not None else (5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
        missing: List[str] = []
        min_score = 1.0
        if scores.size >= 17:
            for idx in req:
                sc = float(scores[idx])
                if sc < min_score:
                    min_score = sc
                if sc < self.min_visibility:
                    missing.append(CALIBRATION_JOINT_LABELS.get(idx, f"kp_{idx}"))
        else:
            min_score = 0.0
            missing = ["full_body"]

        if missing:
            return (False, "OCCLUDED", self.estimated_distance_m, min_score, missing)

        req_pts = pts[list(req), :2]
        if (
            float(np.min(req_pts[:, 0])) < 0.02
            or float(np.max(req_pts[:, 0])) > 0.98
            or float(np.min(req_pts[:, 1])) < 0.01
            or float(np.max(req_pts[:, 1])) > 0.99
        ):
            return (False, "OUT_OF_FRAME", self.estimated_distance_m, min_score, [])

        hips_req = 11 in req or 12 in req
        hips_visible = self._has_visible_hips(pts, scores, self.min_visibility)

        if not hips_req and not hips_visible:
            shoulder_width = max(self._dist3(pts[5], pts[6]), 1e-4)
            est_dist_m = round(float(np.clip(2.0 * (0.19 / shoulder_width), 0.6, 5.0)), 2)
            if shoulder_width < 0.08:
                return (False, "TOO_FAR", est_dist_m, min_score, [])
            if shoulder_width > 0.72:
                return (False, "TOO_CLOSE", est_dist_m, min_score, [])
            return (True, "OK", est_dist_m, min_score, [])

        mid_shoulder = 0.5 * (pts[5] + pts[6])
        mid_hip = 0.5 * (pts[11] + pts[12])
        torso_height = max(self._dist3(mid_shoulder, mid_hip), 1e-4)
        est_dist_m = round(float(np.clip(2.0 * (0.265 / torso_height), 0.6, 5.0)), 2)

        if torso_height < self.min_torso_height_ratio:
            return (False, "TOO_FAR", est_dist_m, min_score, [])
        if torso_height > self.max_torso_height_ratio:
            return (False, "TOO_CLOSE", est_dist_m, min_score, [])

        return (True, "OK", est_dist_m, min_score, [])

    def estimate_live_distance_m(self, pts: np.ndarray, scores: np.ndarray) -> float:
        """Compute dynamic real-time subject distance (meters) from live 2D shoulders/torso."""
        if pts is None or scores is None or pts.shape[0] < 13 or scores.size < 13:
            self.estimated_distance_m = 0.0
            return 0.0
        if float(scores[5]) < 0.25 or float(scores[6]) < 0.25:
            self.estimated_distance_m = 0.0
            return 0.0

        hips_visible = self._has_visible_hips(pts, scores, 0.25)
        if hips_visible:
            mid_sh = 0.5 * (pts[5] + pts[6])
            mid_hip = 0.5 * (pts[11] + pts[12])
            torso_h = max(self._dist3(mid_sh, mid_hip), 1e-4)
            est = round(float(np.clip(2.0 * (0.265 / torso_h), 0.5, 5.0)), 2)
        else:
            sh_w = max(self._dist3(pts[5], pts[6]), 1e-4)
            est = round(float(np.clip(2.0 * (0.19 / sh_w), 0.5, 5.0)), 2)
        self.estimated_distance_m = est
        return est

    def _compute_bone_lengths(
        self,
        stacked: np.ndarray,
        world_stacked: Optional[np.ndarray] = None,
    ) -> BoneLengths3D:
        med = np.median(stacked, axis=0)
        med_world = np.median(world_stacked, axis=0) if world_stacked is not None and len(world_stacked) > 0 else med

        l_upper = self._dist3(med[5], med[7])
        r_upper = self._dist3(med[6], med[8])
        l_fore = self._dist3(med[7], med[9])
        r_fore = self._dist3(med[8], med[10])
        l_thigh = self._dist3(med[11], med[13])
        r_thigh = self._dist3(med[12], med[14])
        l_shin = self._dist3(med[13], med[15])
        r_shin = self._dist3(med[14], med[16])

        shoulder_width = max(self._dist3(med[5], med[6]), 0.05)
        mid_shoulder = 0.5 * (med[5] + med[6])
        mid_hip = 0.5 * (med[11] + med[12])
        has_anatomical_torso = float(mid_hip[1] - mid_shoulder[1]) >= 0.08 and float(mid_hip[1]) <= 0.96
        torso_vec = mid_shoulder - mid_hip
        raw_torso_height = float(np.linalg.norm(torso_vec))

        if has_anatomical_torso and raw_torso_height > 1e-5:
            torso_height = max(raw_torso_height, 0.05)
            unit = torso_vec / torso_height
            cm_from_torso = 52.0 / torso_height
            cm_from_shoulder = 38.0 / shoulder_width
            cm_per_unit = float(np.clip(0.7 * cm_from_torso + 0.3 * cm_from_shoulder, 40.0, 450.0))
            self.estimated_distance_m = round(float(np.clip(2.0 * (0.265 / max(torso_height, 1e-4)), 0.6, 5.0)), 2)
        else:
            torso_height = max(shoulder_width * 1.38, 0.05)
            unit = np.array([0.0, -1.0, 0.0], dtype=np.float64)
            cm_per_unit = float(np.clip(38.0 / shoulder_width, 40.0, 450.0))
            self.estimated_distance_m = round(float(np.clip(2.0 * (0.19 / shoulder_width), 0.6, 5.0)), 2)

        self.baseline_left_shoulder_y = float(med[5, 1])
        self.baseline_right_shoulder_y = float(med[6, 1])
        self.baseline_left_heel_y = float(med[19, 1])
        self.baseline_right_heel_y = float(med[22, 1])
        self.baseline_shoulder_tilt_deg = self._coronal_tilt_deg(med_world[5], med_world[6])
        self.baseline_pelvic_tilt_deg = (
            self._coronal_tilt_deg(med_world[11], med_world[12]) if has_anatomical_torso else 0.0
        )
        self.baseline_shoulder_delta_y = float(med_world[5, 1] - med_world[6, 1])
        self.baseline_spine_angle_deg = float(self._spine_tilt_from_vertical(med_world))

        avg_upper = max(0.5 * (l_upper + r_upper), shoulder_width * 0.82)
        avg_fore = max(0.5 * (l_fore + r_fore), shoulder_width * 0.72)
        avg_thigh = 0.5 * (l_thigh + r_thigh)
        avg_shin = 0.5 * (l_shin + r_shin)

        return BoneLengths3D(
            upper_arm_len=round(avg_upper, 5),
            forearm_len=round(avg_fore, 5),
            thigh_len=round(avg_thigh, 5),
            shin_len=round(avg_shin, 5),
            shoulder_width=round(shoulder_width, 5),
            torso_height=round(torso_height, 5),
            cm_per_unit=round(cm_per_unit, 3),
            spine_vertical_unit=(
                round(float(unit[0]), 6),
                round(float(unit[1]), 6),
                round(float(unit[2]), 6),
            ),
            baseline_shoulder_y=round(float(mid_shoulder[1]), 5),
            baseline_heel_y=round(float(0.5 * (med[19, 1] + med[22, 1])), 5),
            baseline_hip_y=round(float(mid_hip[1]), 5),
            baseline_shoulder_tilt_deg=round(self.baseline_shoulder_tilt_deg, 1),
            baseline_pelvic_tilt_deg=round(self.baseline_pelvic_tilt_deg, 1),
            resting_shoulder_delta_y=round(self.baseline_shoulder_delta_y, 5),
            resting_spine_angle_deg=round(self.baseline_spine_angle_deg, 2),
            arm_to_torso_ratio=round((avg_upper + avg_fore) / torso_height, 4),
            leg_to_torso_ratio=round((avg_thigh + avg_shin) / torso_height, 4),
            estimated_distance_m=self.estimated_distance_m,
        )

    def force_calibrate_from_pose(
        self,
        pts: np.ndarray,
        world_pts: Optional[np.ndarray] = None,
    ) -> BoneLengths3D:
        """Finalize calibration immediately from a valid reference pose (helpful for tests)."""
        self._samples = [pts.copy()]
        if world_pts is not None:
            self._world_samples = [world_pts.copy()]
        self._bone_lengths = self._compute_bone_lengths(
            np.stack(self._samples, axis=0),
            np.stack(self._world_samples, axis=0) if self._world_samples else None,
        )
        return self._bone_lengths

    def update(
        self,
        pts: np.ndarray,
        scores: np.ndarray,
        timestamp_sec: float,
        world_pts: Optional[np.ndarray] = None,
        required_indices: Optional[Sequence[int]] = None,
    ) -> CalibrationStateSchema:
        """Accumulate stable standing frames until duration_sec elapses."""
        if self._bone_lengths is not None:
            return CalibrationStateSchema(
                is_calibrated=True,
                progress=1.0,
                remaining_sec=0.0,
                message_th="คาลิเบรตสัดส่วนร่างกายสำเร็จ พร้อมเริ่มฝึกกายภาพบำบัด",
                distance_status="OK",
                estimated_distance_m=self._bone_lengths.estimated_distance_m,
                min_visibility=1.0,
                missing_joints=[],
                bone_lengths=self._bone_lengths,
            )

        framing_ok, dist_status, est_dist_m, min_score, missing_joints = self.evaluate_framing_and_distance(
            pts, scores, required_indices=required_indices
        )
        if not framing_ok:
            self._start_ts = None
            self._samples.clear()
            self._world_samples.clear()
            if dist_status == "TOO_FAR":
                msg = f"คุณยืนไกลกล้อง ({est_dist_m:.1f} ม.) กรุณาขยับเข้ามาใกล้กล้อง (ระยะแนะนำ 1.5-2.5 ม.)"
            elif dist_status == "TOO_CLOSE":
                msg = f"คุณยืนใกล้กล้องเกินไป ({est_dist_m:.1f} ม.) กรุณาถอยหลังเล็กน้อยให้เห็นข้อต่อครบถ้วน"
            elif dist_status == "OUT_OF_FRAME":
                msg = "กรุณาขยับให้อยู่กึ่งกลางกล้อง เห็นร่างกายตั้งแต่ศีรษะถึงข้อเท้า"
            else:
                msg = "กรุณาถอยยืนในที่แสงสว่างเพียงพอ ให้กล้องเห็นข้อต่อสำคัญครบทุกจุด"
            return CalibrationStateSchema(
                is_calibrated=False,
                progress=0.0,
                remaining_sec=round(self.duration_sec, 1),
                message_th=msg,
                distance_status=dist_status,  # type: ignore[arg-type]
                estimated_distance_m=est_dist_m,
                min_visibility=round(min_score, 3),
                missing_joints=missing_joints,
                bone_lengths=None,
            )

        posture_ref = world_pts if world_pts is not None else pts
        sh_tilt = self._coronal_tilt_deg(posture_ref[5], posture_ref[6])
        req_check = tuple(required_indices) if required_indices is not None else (5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
        hips_visible = self._has_visible_hips(pts, scores, self.min_visibility)
        if (11 in req_check or 12 in req_check) or hips_visible:
            tilt_deg = self._spine_tilt_from_vertical(pts)
            pelvic_tilt = self._coronal_tilt_deg(posture_ref[11], posture_ref[12])
            worst_tilt = max(tilt_deg, sh_tilt, pelvic_tilt)
        else:
            worst_tilt = sh_tilt

        if worst_tilt > self.stability_max_tilt_deg:
            self._start_ts = None
            self._samples.clear()
            self._world_samples.clear()
            return CalibrationStateSchema(
                is_calibrated=False,
                progress=0.0,
                remaining_sec=round(self.duration_sec, 1),
                message_th=f"ลำตัวหรือไหล่/สะโพกเอียง {worst_tilt:.0f}° กรุณายืนตรงนิ่ง 2 วินาทีเพื่อคาลิเบรต",
                distance_status="OK",
                estimated_distance_m=est_dist_m,
                min_visibility=round(min_score, 3),
                missing_joints=[],
                bone_lengths=None,
            )

        if self._start_ts is None:
            self._start_ts = float(timestamp_sec)
            self._samples.clear()
            self._world_samples.clear()

        self._samples.append(pts.copy())
        if world_pts is not None:
            self._world_samples.append(world_pts.copy())
        elapsed = max(0.0, float(timestamp_sec) - self._start_ts)
        progress = float(min(1.0, elapsed / max(self.duration_sec, 0.1)))
        remaining = max(0.0, self.duration_sec - elapsed)

        if elapsed >= self.duration_sec and len(self._samples) >= 3:
            self._bone_lengths = self._compute_bone_lengths(
                np.stack(self._samples, axis=0),
                np.stack(self._world_samples, axis=0) if len(self._world_samples) == len(self._samples) else None,
            )
            return CalibrationStateSchema(
                is_calibrated=True,
                progress=1.0,
                remaining_sec=0.0,
                message_th="คาลิเบรตสัดส่วน 3 มิติเรียบร้อย เริ่มเคลื่อนไหวได้เลยครับ",
                distance_status="OK",
                estimated_distance_m=self._bone_lengths.estimated_distance_m,
                min_visibility=round(min_score, 3),
                missing_joints=[],
                bone_lengths=self._bone_lengths,
            )

        return CalibrationStateSchema(
            is_calibrated=False,
            progress=round(progress, 3),
            remaining_sec=round(remaining, 1),
            message_th=f"กำลังวัดสัดส่วนกระดูก 3 มิติ กรุณายืนนิ่งอีก {remaining:.1f} วินาที",
            distance_status="OK",
            estimated_distance_m=est_dist_m,
            min_visibility=round(min_score, 3),
            missing_joints=[],
            bone_lengths=None,
        )

