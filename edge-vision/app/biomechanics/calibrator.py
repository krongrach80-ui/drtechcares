"""2-Second Standing Calibration capturing patient-specific 3D bone proportions."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from app.schemas.telemetry import BoneLengths3D, CalibrationStateSchema


class StandingCalibrator:
    """
    Requires the patient to stand upright facing the webcam for 2.0 seconds
    before counting rehabilitation reps, extracting 3D bone lengths and spine baseline.
    """

    def __init__(self, duration_sec: float = 2.0, stability_max_tilt_deg: float = 12.0) -> None:
        self.duration_sec = float(duration_sec)
        self.stability_max_tilt_deg = float(stability_max_tilt_deg)
        self._start_ts: Optional[float] = None
        self._samples: List[np.ndarray] = []
        self._bone_lengths: Optional[BoneLengths3D] = None
        self.baseline_left_shoulder_y: float = 0.275
        self.baseline_right_shoulder_y: float = 0.275
        self.baseline_left_heel_y: float = 0.935
        self.baseline_right_heel_y: float = 0.935

    @property
    def is_calibrated(self) -> bool:
        return self._bone_lengths is not None

    @property
    def bone_lengths(self) -> Optional[BoneLengths3D]:
        return self._bone_lengths

    def reset(self) -> None:
        self._start_ts = None
        self._samples.clear()
        self._bone_lengths = None

    @staticmethod
    def _dist3(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

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

    def _compute_bone_lengths(self, stacked: np.ndarray) -> BoneLengths3D:
        med = np.median(stacked, axis=0)

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
        torso_vec = mid_shoulder - mid_hip
        torso_height = max(float(np.linalg.norm(torso_vec)), 0.05)

        if torso_height > 1e-5:
            unit = torso_vec / torso_height
        else:
            unit = np.array([0.0, -1.0, 0.0], dtype=np.float64)

        # Blend torso height (52 cm avg adult) and biacromial shoulder width (38 cm avg)
        cm_from_torso = 52.0 / torso_height
        cm_from_shoulder = 38.0 / shoulder_width
        cm_per_unit = float(np.clip(0.7 * cm_from_torso + 0.3 * cm_from_shoulder, 40.0, 450.0))

        self.baseline_left_shoulder_y = float(med[5, 1])
        self.baseline_right_shoulder_y = float(med[6, 1])
        self.baseline_left_heel_y = float(med[19, 1])
        self.baseline_right_heel_y = float(med[22, 1])

        return BoneLengths3D(
            upper_arm_len=round(0.5 * (l_upper + r_upper), 5),
            forearm_len=round(0.5 * (l_fore + r_fore), 5),
            thigh_len=round(0.5 * (l_thigh + r_thigh), 5),
            shin_len=round(0.5 * (l_shin + r_shin), 5),
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
        )

    def force_calibrate_from_pose(self, pts: np.ndarray) -> BoneLengths3D:
        """Finalize calibration immediately from a valid reference pose (helpful for tests)."""
        self._samples = [pts.copy()]
        self._bone_lengths = self._compute_bone_lengths(np.stack(self._samples, axis=0))
        return self._bone_lengths

    def update(
        self,
        pts: np.ndarray,
        scores: np.ndarray,
        timestamp_sec: float,
    ) -> CalibrationStateSchema:
        """Accumulate stable standing frames until duration_sec elapses."""
        if self._bone_lengths is not None:
            return CalibrationStateSchema(
                is_calibrated=True,
                progress=1.0,
                remaining_sec=0.0,
                message_th="คาลิเบรตสัดส่วนร่างกายสำเร็จ พร้อมเริ่มฝึกกายภาพบำบัด",
                bone_lengths=self._bone_lengths,
            )

        required_indices = (5, 6, 11, 12, 13, 14, 15, 16)
        min_score = float(np.min(scores[list(required_indices)])) if scores.size >= 17 else 0.0
        if min_score < 0.35:
            self._start_ts = None
            self._samples.clear()
            return CalibrationStateSchema(
                is_calibrated=False,
                progress=0.0,
                remaining_sec=round(self.duration_sec, 1),
                message_th="กรุณาถอยยืนให้กล้องเห็นร่างกายตั้งแต่ศีรษะถึงข้อเท้า",
                bone_lengths=None,
            )

        tilt_deg = self._spine_tilt_from_vertical(pts)
        if tilt_deg > self.stability_max_tilt_deg:
            self._start_ts = None
            self._samples.clear()
            return CalibrationStateSchema(
                is_calibrated=False,
                progress=0.0,
                remaining_sec=round(self.duration_sec, 1),
                message_th=f"ลำตัวเอียง {tilt_deg:.0f}° กรุณายืนตรงนิ่ง 2 วินาทีเพื่อคาลิเบรต",
                bone_lengths=None,
            )

        if self._start_ts is None:
            self._start_ts = float(timestamp_sec)
            self._samples.clear()

        self._samples.append(pts.copy())
        elapsed = max(0.0, float(timestamp_sec) - self._start_ts)
        progress = float(min(1.0, elapsed / max(self.duration_sec, 0.1)))
        remaining = max(0.0, self.duration_sec - elapsed)

        if elapsed >= self.duration_sec and len(self._samples) >= 3:
            self._bone_lengths = self._compute_bone_lengths(np.stack(self._samples, axis=0))
            return CalibrationStateSchema(
                is_calibrated=True,
                progress=1.0,
                remaining_sec=0.0,
                message_th="คาลิเบรตสัดส่วน 3 มิติเรียบร้อย เริ่มเคลื่อนไหวได้เลยครับ",
                bone_lengths=self._bone_lengths,
            )

        return CalibrationStateSchema(
            is_calibrated=False,
            progress=round(progress, 3),
            remaining_sec=round(remaining, 1),
            message_th=f"กำลังวัดสัดส่วนกระดูก 3 มิติ กรุณายืนนิ่งอีก {remaining:.1f} วินาที",
            bone_lengths=None,
        )

