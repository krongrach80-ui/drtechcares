"""Complete 26-exercise clinical biomechanics ruleset synced with Dr.Tech.Care."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np


MetricKind = Literal[
    "shoulder_elevation",
    "elbow_flexion",
    "knee_extension",
    "hip_flexion",
    "heel_raise",
    "neck_lateral_tilt",
    "balance_trunk_sway",
    "trunk_flexion",
]


@dataclass(frozen=True)
class ClinicalExerciseRule:
    id: str
    code: str
    name_th: str
    name_en: str
    category: str
    side: Literal["left", "right", "both"]
    pattern: str
    metric_kind: MetricKind
    target_joint: str
    min_angle_deg: float
    max_angle_deg: float
    rest_angle_deg: float
    default_hold_sec: float
    default_sets: int
    default_reps: int
    anti_cheat_checks: Tuple[str, ...] = ("TRUNK_LEAN", "SHOULDER_HIKE", "TOO_FAST")
    hysteresis_enter_margin_deg: float = 6.0
    hysteresis_exit_margin_deg: float = 14.0
    rest_hysteresis_margin_deg: float = 20.0

    @property
    def is_descending_target(self) -> bool:
        """True when active ROM decreases from rest angle (e.g., elbow flexion 145 -> 45)."""
        return self.min_angle_deg < self.rest_angle_deg

    @property
    def requires_straight_elbow(self) -> bool:
        """True when straight-arm shoulder raises should guard against excessive elbow bend."""
        return self.metric_kind == "shoulder_elevation" and self.pattern == "shoulder"

    def target_threshold_enter(self, target_override: Optional[float] = None) -> float:
        """Target threshold required to enter HOLD (ascending >= target-enter, descending <= target+enter)."""
        target = float(target_override) if target_override is not None else float(self.min_angle_deg)
        if self.is_descending_target:
            return target + float(self.hysteresis_enter_margin_deg)
        return target - float(self.hysteresis_enter_margin_deg)

    def target_threshold_exit(self, target_override: Optional[float] = None) -> float:
        """Wider hysteresis exit boundary to prevent oscillation chatter while holding target."""
        target = float(target_override) if target_override is not None else float(self.min_angle_deg)
        if self.is_descending_target:
            return target + float(self.hysteresis_exit_margin_deg)
        return target - float(self.hysteresis_exit_margin_deg)

    def is_in_target_zone(
        self,
        rom_deg: float,
        currently_holding: bool = False,
        target_override: Optional[float] = None,
    ) -> bool:
        """Hysteresis-stabilized target zone check comparing ascending vs descending state boundaries."""
        thresh = (
            self.target_threshold_exit(target_override)
            if currently_holding
            else self.target_threshold_enter(target_override)
        )
        if self.is_descending_target:
            return rom_deg <= thresh
        return rom_deg >= thresh

    def is_in_rest_zone(self, rom_deg: float) -> bool:
        """Hysteresis-stabilized return-to-rest boundary required before counting next rep."""
        rest = float(self.rest_angle_deg)
        margin = float(self.rest_hysteresis_margin_deg)
        if self.is_descending_target:
            return rom_deg >= rest - margin
        return rom_deg <= rest + margin

    def required_landmark_indices(self) -> Tuple[int, ...]:
        """Return 133-topology body indices needed for reliable ROM and compensation tracking."""
        # Upper-body exercises only require shoulders + active arm/head so seated users (hips out of frame) track smoothly.
        if self.metric_kind in ("shoulder_elevation", "elbow_flexion", "neck_lateral_tilt"):
            base = [5, 6]
        else:
            base = [5, 6, 11, 12]
        if self.metric_kind in ("shoulder_elevation", "elbow_flexion"):
            if self.side in ("left", "both"):
                base.extend([7, 9])
            if self.side in ("right", "both"):
                base.extend([8, 10])
        elif self.metric_kind in ("knee_extension", "hip_flexion", "heel_raise", "balance_trunk_sway"):
            if self.side in ("left", "both"):
                base.extend([13, 15])
            if self.side in ("right", "both"):
                base.extend([14, 16])
        elif self.metric_kind == "neck_lateral_tilt":
            base.append(0)
        return tuple(dict.fromkeys(base))


def are_hips_reliable(
    pts: np.ndarray,
    scores: Optional[np.ndarray] = None,
    min_visibility: float = 0.70,
    pts_2d: Optional[np.ndarray] = None,
) -> bool:
    """Return True only when both left and right hips are clearly visible and anatomically below shoulders."""
    arr = np.asarray(pts, dtype=np.float64)
    is_mp33 = arr.shape[0] == 33
    l_sh_idx, r_sh_idx = (11, 12) if is_mp33 else (5, 6)
    l_hp_idx, r_hp_idx = (23, 24) if is_mp33 else (11, 12)
    if arr.shape[0] <= max(l_hp_idx, r_hp_idx):
        return False

    if scores is not None and len(scores) > max(l_hp_idx, r_hp_idx):
        sc = np.asarray(scores, dtype=np.float64).reshape(-1)
        if float(sc[l_hp_idx]) < float(min_visibility) or float(sc[r_hp_idx]) < float(min_visibility):
            return False

    ref_2d = np.asarray(pts_2d, dtype=np.float64) if pts_2d is not None else arr
    if pts_2d is not None and ref_2d.shape[0] > max(l_hp_idx, r_hp_idx):
        mid_hip_y = 0.5 * float(ref_2d[l_hp_idx, 1] + ref_2d[r_hp_idx, 1])
        mid_sh_y = 0.5 * float(ref_2d[l_sh_idx, 1] + ref_2d[r_sh_idx, 1])
        if mid_hip_y > 0.96 or (mid_hip_y - mid_sh_y) < 0.08:
            return False

    mid_sh_3d = 0.5 * (arr[l_sh_idx] + arr[r_sh_idx])
    mid_hp_3d = 0.5 * (arr[l_hp_idx] + arr[r_hp_idx])
    return float(mid_hp_3d[1] - mid_sh_3d[1]) >= 0.06


def compute_shoulder_elevation_with_vertical_fallback(
    pts_3d: np.ndarray,
    side: str = "right",
    scores: Optional[np.ndarray] = None,
    min_visibility: float = 0.70,
    pts_2d: Optional[np.ndarray] = None,
) -> float:
    """
    Compute 3D shoulder flexion/abduction angle against torso-down vector when hips are visible,
    or seamlessly fall back to gravity Vertical Reference Vector [0, 1, 0] for seated desk users.
    Supports both (133, 3) Wholebody and (33, 3) MediaPipe arrays.
    """
    arr = np.asarray(pts_3d, dtype=np.float64)
    is_mp33 = arr.shape[0] == 33
    use_left = side == "left"

    sh_idx = (11 if use_left else 12) if is_mp33 else (5 if use_left else 6)
    el_idx = (13 if use_left else 14) if is_mp33 else (7 if use_left else 8)
    wr_idx = (15 if use_left else 16) if is_mp33 else (9 if use_left else 10)
    l_sh_idx, r_sh_idx = (11, 12) if is_mp33 else (5, 6)
    l_hp_idx, r_hp_idx = (23, 24) if is_mp33 else (11, 12)

    vertical_down = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    if are_hips_reliable(arr, scores=scores, min_visibility=min_visibility, pts_2d=pts_2d):
        mid_shoulder = 0.5 * (arr[l_sh_idx] + arr[r_sh_idx])
        mid_hip = 0.5 * (arr[l_hp_idx] + arr[r_hp_idx])
        trunk_vec = mid_hip - mid_shoulder
        trunk_norm = float(np.linalg.norm(trunk_vec))
        ref_down = trunk_vec / trunk_norm if trunk_norm > 1e-6 else vertical_down
    else:
        # Seated close to webcam without hips visible: anchor reference axis along gravity [0, 1, 0].
        ref_down = vertical_down

    sh = arr[sh_idx]
    el = arr[el_idx]
    wr = arr[wr_idx]
    upper_arm = el - sh
    full_arm = wr - sh
    u_norm = float(np.linalg.norm(upper_arm))
    f_norm = float(np.linalg.norm(full_arm))

    def _deg(vec: np.ndarray, norm_val: float) -> Optional[float]:
        if norm_val < 1e-5:
            return None
        cos_t = float(np.clip(np.dot(vec / norm_val, ref_down), -1.0, 1.0))
        return float(math.degrees(math.acos(cos_t)))

    deg_upper = _deg(upper_arm, u_norm)
    deg_full = _deg(full_arm, f_norm)

    if deg_upper is not None and u_norm >= 0.04:
        return round(deg_upper, 2)
    if deg_full is not None:
        return round(deg_full, 2)
    if deg_upper is not None:
        return round(deg_upper, 2)
    return 0.0


ALL_26_EXERCISES: List[ClinicalExerciseRule] = [
    ClinicalExerciseRule(
        id="ex-01",
        code="SH_FLEX_R",
        name_th="ยกแขนตรงไปด้านหน้า (ขวา)",
        name_en="Right Shoulder Flexion",
        category="shoulder",
        side="right",
        pattern="shoulder",
        metric_kind="shoulder_elevation",
        target_joint="right_shoulder",
        min_angle_deg=110.0,
        max_angle_deg=180.0,
        rest_angle_deg=35.0,
        default_hold_sec=3.0,
        default_sets=2,
        default_reps=5,
        anti_cheat_checks=("TRUNK_LEAN", "SHOULDER_HIKE", "TOO_FAST"),
    ),
    ClinicalExerciseRule(
        id="ex-02",
        code="SH_FLEX_L",
        name_th="ยกแขนตรงไปด้านหน้า (ซ้าย)",
        name_en="Left Shoulder Flexion",
        category="shoulder",
        side="left",
        pattern="shoulder",
        metric_kind="shoulder_elevation",
        target_joint="left_shoulder",
        min_angle_deg=110.0,
        max_angle_deg=180.0,
        rest_angle_deg=35.0,
        default_hold_sec=3.0,
        default_sets=2,
        default_reps=5,
        anti_cheat_checks=("TRUNK_LEAN", "SHOULDER_HIKE", "TOO_FAST"),
    ),
    ClinicalExerciseRule(
        id="ex-03",
        code="SH_ABD_R",
        name_th="กางแขนขวาด้านข้าง",
        name_en="Right Shoulder Abduction",
        category="shoulder",
        side="right",
        pattern="shoulder",
        metric_kind="shoulder_elevation",
        target_joint="right_shoulder",
        min_angle_deg=95.0,
        max_angle_deg=175.0,
        rest_angle_deg=35.0,
        default_hold_sec=3.0,
        default_sets=2,
        default_reps=5,
        anti_cheat_checks=("TRUNK_LEAN", "SHOULDER_HIKE", "TOO_FAST"),
    ),
    ClinicalExerciseRule(
        id="ex-04",
        code="SH_ABD_L",
        name_th="กางแขนซ้ายด้านข้าง",
        name_en="Left Shoulder Abduction",
        category="shoulder",
        side="left",
        pattern="shoulder",
        metric_kind="shoulder_elevation",
        target_joint="left_shoulder",
        min_angle_deg=95.0,
        max_angle_deg=175.0,
        rest_angle_deg=35.0,
        default_hold_sec=3.0,
        default_sets=2,
        default_reps=5,
        anti_cheat_checks=("TRUNK_LEAN", "SHOULDER_HIKE", "TOO_FAST"),
    ),
    ClinicalExerciseRule(
        id="ex-05",
        code="ELB_FLEX_R",
        name_th="งอข้อศอกเข้าหาหัวไหล่ (ขวา)",
        name_en="Right Elbow Flexion (Biceps Curl)",
        category="arm",
        side="right",
        pattern="elbow",
        metric_kind="elbow_flexion",
        target_joint="right_elbow",
        min_angle_deg=35.0,
        max_angle_deg=75.0,
        rest_angle_deg=145.0,
        default_hold_sec=2.0,
        default_sets=2,
        default_reps=8,
        anti_cheat_checks=("TRUNK_LEAN", "SHOULDER_HIKE", "TOO_FAST"),
    ),
    ClinicalExerciseRule(
        id="ex-06",
        code="ELB_FLEX_L",
        name_th="งอข้อศอกเข้าหาหัวไหล่ (ซ้าย)",
        name_en="Left Elbow Flexion (Biceps Curl)",
        category="arm",
        side="left",
        pattern="elbow",
        metric_kind="elbow_flexion",
        target_joint="left_elbow",
        min_angle_deg=35.0,
        max_angle_deg=75.0,
        rest_angle_deg=145.0,
        default_hold_sec=2.0,
        default_sets=2,
        default_reps=8,
        anti_cheat_checks=("TRUNK_LEAN", "SHOULDER_HIKE", "TOO_FAST"),
    ),
    ClinicalExerciseRule(
        id="ex-07",
        code="SH_PEND_R",
        name_th="แกว่งแขนลูกตุ้มผ่อนคลายไหล่ (ขวา)",
        name_en="Right Shoulder Pendulum Swing",
        category="shoulder",
        side="right",
        pattern="timer",
        metric_kind="shoulder_elevation",
        target_joint="right_shoulder",
        min_angle_deg=15.0,
        max_angle_deg=90.0,
        rest_angle_deg=15.0,
        default_hold_sec=15.0,
        default_sets=2,
        default_reps=1,
        anti_cheat_checks=("TOO_FAST",),
    ),
    ClinicalExerciseRule(
        id="ex-08",
        code="SH_PEND_L",
        name_th="แกว่งแขนลูกตุ้มผ่อนคลายไหล่ (ซ้าย)",
        name_en="Left Shoulder Pendulum Swing",
        category="shoulder",
        side="left",
        pattern="timer",
        metric_kind="shoulder_elevation",
        target_joint="left_shoulder",
        min_angle_deg=15.0,
        max_angle_deg=90.0,
        rest_angle_deg=15.0,
        default_hold_sec=15.0,
        default_sets=2,
        default_reps=1,
        anti_cheat_checks=("TOO_FAST",),
    ),
    ClinicalExerciseRule(
        id="ex-09",
        code="SH_SCAP_R",
        name_th="ยกแขนแนว 45 องศา (ขวา)",
        name_en="Right Shoulder Scaption (V-Raise)",
        category="shoulder",
        side="right",
        pattern="shoulder",
        metric_kind="shoulder_elevation",
        target_joint="right_shoulder",
        min_angle_deg=90.0,
        max_angle_deg=165.0,
        rest_angle_deg=35.0,
        default_hold_sec=3.0,
        default_sets=2,
        default_reps=6,
        anti_cheat_checks=("TRUNK_LEAN", "SHOULDER_HIKE", "TOO_FAST"),
    ),
    ClinicalExerciseRule(
        id="ex-10",
        code="SH_SCAP_L",
        name_th="ยกแขนแนว 45 องศา (ซ้าย)",
        name_en="Left Shoulder Scaption (V-Raise)",
        category="shoulder",
        side="left",
        pattern="shoulder",
        metric_kind="shoulder_elevation",
        target_joint="left_shoulder",
        min_angle_deg=90.0,
        max_angle_deg=165.0,
        rest_angle_deg=35.0,
        default_hold_sec=3.0,
        default_sets=2,
        default_reps=6,
        anti_cheat_checks=("TRUNK_LEAN", "SHOULDER_HIKE", "TOO_FAST"),
    ),
    ClinicalExerciseRule(
        id="ex-11",
        code="SH_OVH_R",
        name_th="ชูแขนเหนือศีรษะค้างไว้ (ขวา)",
        name_en="Right Overhead Reach Hold",
        category="shoulder",
        side="right",
        pattern="shoulder",
        metric_kind="shoulder_elevation",
        target_joint="right_shoulder",
        min_angle_deg=135.0,
        max_angle_deg=180.0,
        rest_angle_deg=40.0,
        default_hold_sec=3.0,
        default_sets=2,
        default_reps=5,
        anti_cheat_checks=("TRUNK_LEAN", "SHOULDER_HIKE", "TOO_FAST"),
    ),
    ClinicalExerciseRule(
        id="ex-12",
        code="SH_OVH_L",
        name_th="ชูแขนเหนือศีรษะค้างไว้ (ซ้าย)",
        name_en="Left Overhead Reach Hold",
        category="shoulder",
        side="left",
        pattern="shoulder",
        metric_kind="shoulder_elevation",
        target_joint="left_shoulder",
        min_angle_deg=135.0,
        max_angle_deg=180.0,
        rest_angle_deg=40.0,
        default_hold_sec=3.0,
        default_sets=2,
        default_reps=5,
        anti_cheat_checks=("TRUNK_LEAN", "SHOULDER_HIKE", "TOO_FAST"),
    ),
    ClinicalExerciseRule(
        id="ex-13",
        code="KNEE_EXT_R",
        name_th="นั่งเตะเหยียดขาตรง (ขวา)",
        name_en="Right Seated Knee Extension",
        category="leg",
        side="right",
        pattern="knee",
        metric_kind="knee_extension",
        target_joint="right_knee",
        min_angle_deg=155.0,
        max_angle_deg=180.0,
        rest_angle_deg=110.0,
        default_hold_sec=3.0,
        default_sets=2,
        default_reps=6,
        anti_cheat_checks=("TRUNK_LEAN", "KNEE_VALGUS", "HEEL_LIFT", "TOO_FAST"),
    ),
    ClinicalExerciseRule(
        id="ex-14",
        code="KNEE_EXT_L",
        name_th="นั่งเตะเหยียดขาตรง (ซ้าย)",
        name_en="Left Seated Knee Extension",
        category="leg",
        side="left",
        pattern="knee",
        metric_kind="knee_extension",
        target_joint="left_knee",
        min_angle_deg=155.0,
        max_angle_deg=180.0,
        rest_angle_deg=110.0,
        default_hold_sec=3.0,
        default_sets=2,
        default_reps=6,
        anti_cheat_checks=("TRUNK_LEAN", "KNEE_VALGUS", "HEEL_LIFT", "TOO_FAST"),
    ),
    ClinicalExerciseRule(
        id="ex-15",
        code="HIP_FLEX_R",
        name_th="ยกเข่าสูงอยู่กับที่ (ขวา)",
        name_en="Right Marching / High Knee Lift",
        category="leg",
        side="right",
        pattern="hip",
        metric_kind="hip_flexion",
        target_joint="right_hip",
        min_angle_deg=75.0,
        max_angle_deg=120.0,
        rest_angle_deg=155.0,
        default_hold_sec=2.0,
        default_sets=2,
        default_reps=6,
        anti_cheat_checks=("TRUNK_LEAN", "KNEE_VALGUS", "TOO_FAST"),
    ),
    ClinicalExerciseRule(
        id="ex-16",
        code="HIP_FLEX_L",
        name_th="ยกเข่าสูงอยู่กับที่ (ซ้าย)",
        name_en="Left Marching / High Knee Lift",
        category="leg",
        side="left",
        pattern="hip",
        metric_kind="hip_flexion",
        target_joint="left_hip",
        min_angle_deg=75.0,
        max_angle_deg=120.0,
        rest_angle_deg=155.0,
        default_hold_sec=2.0,
        default_sets=2,
        default_reps=6,
        anti_cheat_checks=("TRUNK_LEAN", "KNEE_VALGUS", "TOO_FAST"),
    ),
    ClinicalExerciseRule(
        id="ex-17",
        code="HEEL_RAISE_R",
        name_th="เขย่งส้นเท้าช้าๆ (ขวา)",
        name_en="Right Standing Calf / Heel Raise",
        category="leg",
        side="right",
        pattern="heel_raise",
        metric_kind="heel_raise",
        target_joint="right_ankle",
        min_angle_deg=45.0,
        max_angle_deg=95.0,
        rest_angle_deg=25.0,
        default_hold_sec=2.0,
        default_sets=2,
        default_reps=8,
        anti_cheat_checks=("TRUNK_LEAN", "KNEE_VALGUS", "TOO_FAST"),
    ),
    ClinicalExerciseRule(
        id="ex-18",
        code="HEEL_RAISE_L",
        name_th="เขย่งส้นเท้าช้าๆ (ซ้าย)",
        name_en="Left Standing Calf / Heel Raise",
        category="leg",
        side="left",
        pattern="heel_raise",
        metric_kind="heel_raise",
        target_joint="left_ankle",
        min_angle_deg=45.0,
        max_angle_deg=95.0,
        rest_angle_deg=25.0,
        default_hold_sec=2.0,
        default_sets=2,
        default_reps=8,
        anti_cheat_checks=("TRUNK_LEAN", "KNEE_VALGUS", "TOO_FAST"),
    ),
    ClinicalExerciseRule(
        id="ex-19",
        code="NECK_TILT_R",
        name_th="เอียงศีรษะยืดคอด้านขวา",
        name_en="Right Neck Lateral Tilt Stretch",
        category="neck",
        side="right",
        pattern="neck_tilt",
        metric_kind="neck_lateral_tilt",
        target_joint="neck",
        min_angle_deg=18.0,
        max_angle_deg=45.0,
        rest_angle_deg=8.0,
        default_hold_sec=3.0,
        default_sets=2,
        default_reps=5,
        anti_cheat_checks=("TRUNK_LEAN", "SHOULDER_HIKE", "TOO_FAST"),
    ),
    ClinicalExerciseRule(
        id="ex-20",
        code="NECK_TILT_L",
        name_th="เอียงศีรษะยืดคอด้านซ้าย",
        name_en="Left Neck Lateral Tilt Stretch",
        category="neck",
        side="left",
        pattern="neck_tilt",
        metric_kind="neck_lateral_tilt",
        target_joint="neck",
        min_angle_deg=18.0,
        max_angle_deg=45.0,
        rest_angle_deg=8.0,
        default_hold_sec=3.0,
        default_sets=2,
        default_reps=5,
        anti_cheat_checks=("TRUNK_LEAN", "SHOULDER_HIKE", "TOO_FAST"),
    ),
    ClinicalExerciseRule(
        id="ex-21",
        code="BAL_STAND_R",
        name_th="ยืนทรงตัวลงน้ำหนักขาขวา",
        name_en="Single-Leg Balance Hold (Right)",
        category="balance",
        side="right",
        pattern="hold_balance",
        metric_kind="balance_trunk_sway",
        target_joint="right_hip",
        min_angle_deg=0.0,
        max_angle_deg=18.0,
        rest_angle_deg=25.0,
        default_hold_sec=10.0,
        default_sets=2,
        default_reps=1,
        anti_cheat_checks=("TRUNK_LEAN", "KNEE_VALGUS"),
    ),
    ClinicalExerciseRule(
        id="ex-22",
        code="BAL_STAND_L",
        name_th="ยืนทรงตัวลงน้ำหนักขาซ้าย",
        name_en="Single-Leg Balance Hold (Left)",
        category="balance",
        side="left",
        pattern="hold_balance",
        metric_kind="balance_trunk_sway",
        target_joint="left_hip",
        min_angle_deg=0.0,
        max_angle_deg=18.0,
        rest_angle_deg=25.0,
        default_hold_sec=10.0,
        default_sets=2,
        default_reps=1,
        anti_cheat_checks=("TRUNK_LEAN", "KNEE_VALGUS"),
    ),
    ClinicalExerciseRule(
        id="ex-23",
        code="TRUNK_SIT_STRETCH",
        name_th="นั่งยืดหลังตรงผ่อนคลาย 10 วินาที",
        name_en="Seated Posture & Spine Alignment Hold",
        category="trunk",
        side="both",
        pattern="timer",
        metric_kind="trunk_flexion",
        target_joint="spine",
        min_angle_deg=0.0,
        max_angle_deg=12.0,
        rest_angle_deg=20.0,
        default_hold_sec=10.0,
        default_sets=2,
        default_reps=1,
        anti_cheat_checks=("TRUNK_LEAN",),
    ),
    ClinicalExerciseRule(
        id="ex-24",
        code="SH_WALL_SLIDE_R",
        name_th="ไถแขนขึ้นกำแพงด้านหน้า (ขวา)",
        name_en="Right Shoulder Wall Slide",
        category="shoulder",
        side="right",
        pattern="shoulder",
        metric_kind="shoulder_elevation",
        target_joint="right_shoulder",
        min_angle_deg=115.0,
        max_angle_deg=180.0,
        rest_angle_deg=40.0,
        default_hold_sec=3.0,
        default_sets=2,
        default_reps=5,
        anti_cheat_checks=("TRUNK_LEAN", "SHOULDER_HIKE", "TOO_FAST"),
    ),
    ClinicalExerciseRule(
        id="ex-25",
        code="SH_WALL_SLIDE_L",
        name_th="ไถแขนขึ้นกำแพงด้านหน้า (ซ้าย)",
        name_en="Left Shoulder Wall Slide",
        category="shoulder",
        side="left",
        pattern="shoulder",
        metric_kind="shoulder_elevation",
        target_joint="left_shoulder",
        min_angle_deg=115.0,
        max_angle_deg=180.0,
        rest_angle_deg=40.0,
        default_hold_sec=3.0,
        default_sets=2,
        default_reps=5,
        anti_cheat_checks=("TRUNK_LEAN", "SHOULDER_HIKE", "TOO_FAST"),
    ),
    ClinicalExerciseRule(
        id="ex-26",
        code="SIT_TO_STAND",
        name_th="ลุกจากเก้าอี้และนั่งลงช้าๆ (สควอท)",
        name_en="Sit-to-Stand Functional Squat",
        category="leg",
        side="both",
        pattern="knee",
        metric_kind="knee_extension",
        target_joint="right_knee",
        min_angle_deg=160.0,
        max_angle_deg=180.0,
        rest_angle_deg=110.0,
        default_hold_sec=2.0,
        default_sets=2,
        default_reps=6,
        anti_cheat_checks=("TRUNK_LEAN", "KNEE_VALGUS", "HEEL_LIFT", "TOO_FAST"),
    ),
]

EXERCISE_BY_ID: Dict[str, ClinicalExerciseRule] = {rule.id: rule for rule in ALL_26_EXERCISES}
EXERCISE_BY_CODE: Dict[str, ClinicalExerciseRule] = {rule.code: rule for rule in ALL_26_EXERCISES}

# Kiosk slug aliases used in useKioskSocket & PhysicalTherapy views
EXERCISE_ALIASES: Dict[str, str] = {
    "right_shoulder_abduction": "ex-03",
    "left_shoulder_abduction": "ex-04",
    "right_shoulder_flexion": "ex-01",
    "left_shoulder_flexion": "ex-02",
    "elbow_flexion_right": "ex-05",
    "elbow_flexion_left": "ex-06",
    "knee_extension_right": "ex-13",
    "knee_extension_left": "ex-14",
    "chair_squat": "ex-26",
    "sit_to_stand": "ex-26",
}


def resolve_exercise_rule(identifier: str) -> ClinicalExerciseRule:
    """Resolve exercise by id ('ex-03'), code ('SH_ABD_R'), or frontend alias."""
    norm = (identifier or "").strip()
    if norm in EXERCISE_BY_ID:
        return EXERCISE_BY_ID[norm]
    if norm.upper() in EXERCISE_BY_CODE:
        return EXERCISE_BY_CODE[norm.upper()]
    alias_id = EXERCISE_ALIASES.get(norm.lower())
    if alias_id and alias_id in EXERCISE_BY_ID:
        return EXERCISE_BY_ID[alias_id]
    return EXERCISE_BY_ID["ex-03"]


def list_all_exercises() -> List[Dict[str, object]]:
    """Return serializable metadata for all 26 exercises."""
    return [
        {
            "id": r.id,
            "code": r.code,
            "name_th": r.name_th,
            "name_en": r.name_en,
            "category": r.category,
            "side": r.side,
            "target_joint": r.target_joint,
            "min_angle_deg": r.min_angle_deg,
            "max_angle_deg": r.max_angle_deg,
            "rest_angle_deg": r.rest_angle_deg,
            "default_hold_sec": r.default_hold_sec,
            "default_reps": r.default_reps,
            "anti_cheat_checks": list(r.anti_cheat_checks),
            "hysteresis_enter_margin_deg": r.hysteresis_enter_margin_deg,
            "hysteresis_exit_margin_deg": r.hysteresis_exit_margin_deg,
        }
        for r in ALL_26_EXERCISES
    ]

