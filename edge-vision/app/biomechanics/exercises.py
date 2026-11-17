"""Complete 26-exercise clinical Stroke ADL & MOPH Elderly biomechanics ruleset synced with Dr.Tech.Care."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np


MetricKind = Literal[
    "shoulder_elevation",
    "cross_body_reach",
    "scapular_protraction",
    "elbow_flexion",
    "bilateral_elbow_push",
    "hand_to_head",
    "forearm_rotation",
    "shoulder_rotation",
    "wrist_finger_extension",
    "thumb_pinch",
    "shoulder_shrug_roll",
    "bilateral_arm_raise",
    "neck_lateral_tilt",
    "neck_yaw_turn",
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
        """True when active ROM decreases from rest angle (e.g., elbow flexion 145 -> 55)."""
        return self.min_angle_deg < self.rest_angle_deg

    @property
    def requires_straight_elbow(self) -> bool:
        """True when straight-arm shoulder raises should guard against excessive elbow bend."""
        return (
            self.metric_kind in ("shoulder_elevation", "scapular_protraction")
            and self.pattern == "shoulder"
        )

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
        """Return 133-topology body indices needed for reliable upper-body ROM and compensation tracking."""
        base = [5, 6]
        if self.metric_kind in (
            "shoulder_elevation",
            "cross_body_reach",
            "scapular_protraction",
            "elbow_flexion",
            "bilateral_elbow_push",
            "hand_to_head",
            "forearm_rotation",
            "shoulder_rotation",
            "wrist_finger_extension",
            "thumb_pinch",
            "bilateral_arm_raise",
        ):
            if self.side in ("left", "both"):
                base.extend([7, 9])
            if self.side in ("right", "both"):
                base.extend([8, 10])
        elif self.metric_kind == "shoulder_shrug_roll":
            base.extend([3, 4])
        elif self.metric_kind in ("neck_lateral_tilt", "neck_yaw_turn"):
            base.extend([0, 3, 4])
        if self.metric_kind == "hand_to_head":
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
    # --- Category A: Stroke ADL Shoulder & Arm (12) ---
    ClinicalExerciseRule(
        id="STROKE_SH_FLEX_R",
        code="STROKE_SH_FLEX_R",
        name_th="[Stroke ADL] ยกแขนขวาไปด้านหน้า (ฝึกเอื้อม/สอดแขนเสื้อ)",
        name_en="Stroke ADL: Right Shoulder Forward Flexion",
        category="upper_limb",
        side="right",
        pattern="shoulder",
        metric_kind="shoulder_elevation",
        target_joint="right_shoulder",
        min_angle_deg=90.0,
        max_angle_deg=165.0,
        rest_angle_deg=30.0,
        default_hold_sec=3.0,
        default_sets=2,
        default_reps=6,
        anti_cheat_checks=("TRUNK_LEAN", "SHOULDER_HIKE", "ELBOW_FLEXION", "TOO_FAST"),
    ),
    ClinicalExerciseRule(
        id="STROKE_SH_FLEX_L",
        code="STROKE_SH_FLEX_L",
        name_th="[Stroke ADL] ยกแขนซ้ายไปด้านหน้า (ฝึกเอื้อม/สอดแขนเสื้อ)",
        name_en="Stroke ADL: Left Shoulder Forward Flexion",
        category="upper_limb",
        side="left",
        pattern="shoulder",
        metric_kind="shoulder_elevation",
        target_joint="left_shoulder",
        min_angle_deg=90.0,
        max_angle_deg=165.0,
        rest_angle_deg=30.0,
        default_hold_sec=3.0,
        default_sets=2,
        default_reps=6,
        anti_cheat_checks=("TRUNK_LEAN", "SHOULDER_HIKE", "ELBOW_FLEXION", "TOO_FAST"),
    ),
    ClinicalExerciseRule(
        id="STROKE_SH_ABD_R",
        code="STROKE_SH_ABD_R",
        name_th="[Stroke ADL] กางแขนขวาด้านข้าง (ฝึกกางแขนสวมเสื้อ)",
        name_en="Stroke ADL: Right Shoulder Abduction",
        category="upper_limb",
        side="right",
        pattern="shoulder",
        metric_kind="shoulder_elevation",
        target_joint="right_shoulder",
        min_angle_deg=85.0,
        max_angle_deg=160.0,
        rest_angle_deg=30.0,
        default_hold_sec=3.0,
        default_sets=2,
        default_reps=6,
        anti_cheat_checks=("TRUNK_LEAN", "SHOULDER_HIKE", "TOO_FAST"),
    ),
    ClinicalExerciseRule(
        id="STROKE_SH_ABD_L",
        code="STROKE_SH_ABD_L",
        name_th="[Stroke ADL] กางแขนซ้ายด้านข้าง (ฝึกกางแขนสวมเสื้อ)",
        name_en="Stroke ADL: Left Shoulder Abduction",
        category="upper_limb",
        side="left",
        pattern="shoulder",
        metric_kind="shoulder_elevation",
        target_joint="left_shoulder",
        min_angle_deg=85.0,
        max_angle_deg=160.0,
        rest_angle_deg=30.0,
        default_hold_sec=3.0,
        default_sets=2,
        default_reps=6,
        anti_cheat_checks=("TRUNK_LEAN", "SHOULDER_HIKE", "TOO_FAST"),
    ),
    ClinicalExerciseRule(
        id="STROKE_CROSS_BODY_R",
        code="STROKE_CROSS_BODY_R",
        name_th="[Stroke ADL] เอื้อมมือขวาแตะไหล่ซ้าย (ฝึกดึงแขนเสื้อ/ใส่เสื้อคลุม)",
        name_en="Stroke ADL: Right Cross-Body Shoulder Reach",
        category="upper_limb",
        side="right",
        pattern="shoulder",
        metric_kind="cross_body_reach",
        target_joint="right_shoulder",
        min_angle_deg=75.0,
        max_angle_deg=145.0,
        rest_angle_deg=25.0,
        default_hold_sec=3.0,
        default_sets=2,
        default_reps=6,
        anti_cheat_checks=("TRUNK_LEAN", "SHOULDER_HIKE", "TOO_FAST"),
    ),
    ClinicalExerciseRule(
        id="STROKE_CROSS_BODY_L",
        code="STROKE_CROSS_BODY_L",
        name_th="[Stroke ADL] เอื้อมมือซ้ายแตะไหล่ขวา (ฝึกดึงแขนเสื้อ/ใส่เสื้อคลุม)",
        name_en="Stroke ADL: Left Cross-Body Shoulder Reach",
        category="upper_limb",
        side="left",
        pattern="shoulder",
        metric_kind="cross_body_reach",
        target_joint="left_shoulder",
        min_angle_deg=75.0,
        max_angle_deg=145.0,
        rest_angle_deg=25.0,
        default_hold_sec=3.0,
        default_sets=2,
        default_reps=6,
        anti_cheat_checks=("TRUNK_LEAN", "SHOULDER_HIKE", "TOO_FAST"),
    ),
    ClinicalExerciseRule(
        id="STROKE_SH_ROT_R",
        code="STROKE_SH_ROT_R",
        name_th="[Stroke ADL] หมุนข้อไหล่ขวาเข้า-ออก (อ้อมใส่เสื้อ/ติดกระดุม)",
        name_en="Stroke ADL: Right Shoulder External/Internal Rotation",
        category="upper_limb",
        side="right",
        pattern="rotation",
        metric_kind="shoulder_rotation",
        target_joint="right_shoulder",
        min_angle_deg=45.0,
        max_angle_deg=90.0,
        rest_angle_deg=15.0,
        default_hold_sec=3.0,
        default_sets=2,
        default_reps=6,
        anti_cheat_checks=("TRUNK_LEAN", "TOO_FAST"),
        hysteresis_enter_margin_deg=5.0,
        hysteresis_exit_margin_deg=10.0,
        rest_hysteresis_margin_deg=12.0,
    ),
    ClinicalExerciseRule(
        id="STROKE_SH_ROT_L",
        code="STROKE_SH_ROT_L",
        name_th="[Stroke ADL] หมุนข้อไหล่ซ้ายเข้า-ออก (อ้อมใส่เสื้อ/ติดกระดุม)",
        name_en="Stroke ADL: Left Shoulder External/Internal Rotation",
        category="upper_limb",
        side="left",
        pattern="rotation",
        metric_kind="shoulder_rotation",
        target_joint="left_shoulder",
        min_angle_deg=45.0,
        max_angle_deg=90.0,
        rest_angle_deg=15.0,
        default_hold_sec=3.0,
        default_sets=2,
        default_reps=6,
        anti_cheat_checks=("TRUNK_LEAN", "TOO_FAST"),
        hysteresis_enter_margin_deg=5.0,
        hysteresis_exit_margin_deg=10.0,
        rest_hysteresis_margin_deg=12.0,
    ),
    ClinicalExerciseRule(
        id="STROKE_SCAP_PROTRACT_R",
        code="STROKE_SCAP_PROTRACT_R",
        name_th="[Stroke ADL] เหยียดแขนขวาดันไปด้านหน้าระดับอก (ป้องกันข้อไหล่หลุด)",
        name_en="Stroke ADL: Right Scapular Protraction Punch",
        category="upper_limb",
        side="right",
        pattern="shoulder",
        metric_kind="scapular_protraction",
        target_joint="right_shoulder",
        min_angle_deg=70.0,
        max_angle_deg=120.0,
        rest_angle_deg=25.0,
        default_hold_sec=3.0,
        default_sets=2,
        default_reps=6,
        anti_cheat_checks=("TRUNK_LEAN", "ELBOW_FLEXION", "TOO_FAST"),
        hysteresis_enter_margin_deg=5.0,
        hysteresis_exit_margin_deg=10.0,
        rest_hysteresis_margin_deg=15.0,
    ),
    ClinicalExerciseRule(
        id="STROKE_SCAP_PROTRACT_L",
        code="STROKE_SCAP_PROTRACT_L",
        name_th="[Stroke ADL] เหยียดแขนซ้ายดันไปด้านหน้าระดับอก (ป้องกันข้อไหล่หลุด)",
        name_en="Stroke ADL: Left Scapular Protraction Punch",
        category="upper_limb",
        side="left",
        pattern="shoulder",
        metric_kind="scapular_protraction",
        target_joint="left_shoulder",
        min_angle_deg=70.0,
        max_angle_deg=120.0,
        rest_angle_deg=25.0,
        default_hold_sec=3.0,
        default_sets=2,
        default_reps=6,
        anti_cheat_checks=("TRUNK_LEAN", "ELBOW_FLEXION", "TOO_FAST"),
        hysteresis_enter_margin_deg=5.0,
        hysteresis_exit_margin_deg=10.0,
        rest_hysteresis_margin_deg=15.0,
    ),
    ClinicalExerciseRule(
        id="STROKE_BOBATH_ARM_RAISE",
        code="STROKE_BOBATH_ARM_RAISE",
        name_th="[Stroke & สธ.] ประสานมือสองข้างชูขึ้นเหนือศีรษะ (ท่า Bobath พยุงแขนอ่อนแรง)",
        name_en="Stroke & MOPH: Bobath Bilateral Clasped Arm Raise",
        category="upper_limb",
        side="both",
        pattern="bilateral",
        metric_kind="bilateral_arm_raise",
        target_joint="right_shoulder",
        min_angle_deg=105.0,
        max_angle_deg=175.0,
        rest_angle_deg=35.0,
        default_hold_sec=3.0,
        default_sets=2,
        default_reps=6,
        anti_cheat_checks=("TRUNK_LEAN", "TOO_FAST"),
    ),
    ClinicalExerciseRule(
        id="STROKE_BOBATH_FORWARD_PUSH",
        code="STROKE_BOBATH_FORWARD_PUSH",
        name_th="[Stroke ADL] ประสานมือสองข้างดันเหยียดแขนไปด้านหน้า (ลดเกร็งพับศอก)",
        name_en="Stroke ADL: Bobath Bilateral Forward Elbow Push",
        category="upper_limb",
        side="both",
        pattern="elbow",
        metric_kind="bilateral_elbow_push",
        target_joint="right_shoulder",
        min_angle_deg=135.0,
        max_angle_deg=180.0,
        rest_angle_deg=95.0,
        default_hold_sec=3.0,
        default_sets=2,
        default_reps=6,
        anti_cheat_checks=("TRUNK_LEAN", "TOO_FAST"),
    ),
    # --- Category B: Stroke ADL Elbow, Forearm, Wrist & Fingers (10) ---
    ClinicalExerciseRule(
        id="STROKE_ELB_FLEX_EXT_R",
        code="STROKE_ELB_FLEX_EXT_R",
        name_th="[Stroke ADL] งอ-เหยียดข้อศอกขวา (ดึงแขนเสื้อ/ป้อนอาหาร)",
        name_en="Stroke ADL: Right Elbow Flexion & Extension",
        category="upper_limb",
        side="right",
        pattern="elbow",
        metric_kind="elbow_flexion",
        target_joint="right_shoulder",
        min_angle_deg=55.0,
        max_angle_deg=165.0,
        rest_angle_deg=145.0,
        default_hold_sec=2.0,
        default_sets=2,
        default_reps=8,
        anti_cheat_checks=("TRUNK_LEAN", "SHOULDER_HIKE", "TOO_FAST"),
        hysteresis_enter_margin_deg=6.0,
        hysteresis_exit_margin_deg=14.0,
        rest_hysteresis_margin_deg=25.0,
    ),
    ClinicalExerciseRule(
        id="STROKE_ELB_FLEX_EXT_L",
        code="STROKE_ELB_FLEX_EXT_L",
        name_th="[Stroke ADL] งอ-เหยียดข้อศอกซ้าย (ดึงแขนเสื้อ/ป้อนอาหาร)",
        name_en="Stroke ADL: Left Elbow Flexion & Extension",
        category="upper_limb",
        side="left",
        pattern="elbow",
        metric_kind="elbow_flexion",
        target_joint="left_shoulder",
        min_angle_deg=55.0,
        max_angle_deg=165.0,
        rest_angle_deg=145.0,
        default_hold_sec=2.0,
        default_sets=2,
        default_reps=8,
        anti_cheat_checks=("TRUNK_LEAN", "SHOULDER_HIKE", "TOO_FAST"),
        hysteresis_enter_margin_deg=6.0,
        hysteresis_exit_margin_deg=14.0,
        rest_hysteresis_margin_deg=25.0,
    ),
    ClinicalExerciseRule(
        id="STROKE_HAND_TO_HEAD_R",
        code="STROKE_HAND_TO_HEAD_R",
        name_th="[Stroke ADL] ยกมือขวาแตะศีรษะ (ฝึกสวมเสื้อทางศีรษะ/หวีผม)",
        name_en="Stroke ADL: Right Hand-to-Head Grooming Reach",
        category="upper_limb",
        side="right",
        pattern="elbow",
        metric_kind="hand_to_head",
        target_joint="right_shoulder",
        min_angle_deg=95.0,
        max_angle_deg=160.0,
        rest_angle_deg=35.0,
        default_hold_sec=3.0,
        default_sets=2,
        default_reps=6,
        anti_cheat_checks=("TRUNK_LEAN", "SHOULDER_HIKE", "TOO_FAST"),
    ),
    ClinicalExerciseRule(
        id="STROKE_HAND_TO_HEAD_L",
        code="STROKE_HAND_TO_HEAD_L",
        name_th="[Stroke ADL] ยกมือซ้ายแตะศีรษะ (ฝึกสวมเสื้อทางศีรษะ/หวีผม)",
        name_en="Stroke ADL: Left Hand-to-Head Grooming Reach",
        category="upper_limb",
        side="left",
        pattern="elbow",
        metric_kind="hand_to_head",
        target_joint="left_shoulder",
        min_angle_deg=95.0,
        max_angle_deg=160.0,
        rest_angle_deg=35.0,
        default_hold_sec=3.0,
        default_sets=2,
        default_reps=6,
        anti_cheat_checks=("TRUNK_LEAN", "SHOULDER_HIKE", "TOO_FAST"),
    ),
    ClinicalExerciseRule(
        id="STROKE_FOREARM_TURN_R",
        code="STROKE_FOREARM_TURN_R",
        name_th="[Stroke ADL] พลิกหงาย-คว่ำฝ่ามือขวา (ฝึกถือช้อน/บิดลูกบิดประตู)",
        name_en="Stroke ADL: Right Forearm Supination & Pronation",
        category="upper_limb",
        side="right",
        pattern="wrist",
        metric_kind="forearm_rotation",
        target_joint="right_shoulder",
        min_angle_deg=50.0,
        max_angle_deg=95.0,
        rest_angle_deg=18.0,
        default_hold_sec=2.5,
        default_sets=2,
        default_reps=8,
        anti_cheat_checks=("TOO_FAST",),
        hysteresis_enter_margin_deg=6.0,
        hysteresis_exit_margin_deg=12.0,
        rest_hysteresis_margin_deg=15.0,
    ),
    ClinicalExerciseRule(
        id="STROKE_FOREARM_TURN_L",
        code="STROKE_FOREARM_TURN_L",
        name_th="[Stroke ADL] พลิกหงาย-คว่ำฝ่ามือซ้าย (ฝึกถือช้อน/บิดลูกบิดประตู)",
        name_en="Stroke ADL: Left Forearm Supination & Pronation",
        category="upper_limb",
        side="left",
        pattern="wrist",
        metric_kind="forearm_rotation",
        target_joint="left_shoulder",
        min_angle_deg=50.0,
        max_angle_deg=95.0,
        rest_angle_deg=18.0,
        default_hold_sec=2.5,
        default_sets=2,
        default_reps=8,
        anti_cheat_checks=("TOO_FAST",),
        hysteresis_enter_margin_deg=6.0,
        hysteresis_exit_margin_deg=12.0,
        rest_hysteresis_margin_deg=15.0,
    ),
    ClinicalExerciseRule(
        id="STROKE_WRIST_FINGER_R",
        code="STROKE_WRIST_FINGER_R",
        name_th="[Stroke ADL] กระดกข้อมือ & กาง-กำนิ้วมือขวา (ลดเกร็งนิ้วมือ)",
        name_en="Stroke ADL: Right Wrist Extension & Finger Open/Close",
        category="upper_limb",
        side="right",
        pattern="wrist",
        metric_kind="wrist_finger_extension",
        target_joint="right_shoulder",
        min_angle_deg=42.0,
        max_angle_deg=85.0,
        rest_angle_deg=15.0,
        default_hold_sec=2.5,
        default_sets=2,
        default_reps=8,
        anti_cheat_checks=("TOO_FAST",),
        hysteresis_enter_margin_deg=5.0,
        hysteresis_exit_margin_deg=10.0,
        rest_hysteresis_margin_deg=12.0,
    ),
    ClinicalExerciseRule(
        id="STROKE_WRIST_FINGER_L",
        code="STROKE_WRIST_FINGER_L",
        name_th="[Stroke ADL] กระดกข้อมือ & กาง-กำนิ้วมือซ้าย (ลดเกร็งนิ้วมือ)",
        name_en="Stroke ADL: Left Wrist Extension & Finger Open/Close",
        category="upper_limb",
        side="left",
        pattern="wrist",
        metric_kind="wrist_finger_extension",
        target_joint="left_shoulder",
        min_angle_deg=42.0,
        max_angle_deg=85.0,
        rest_angle_deg=15.0,
        default_hold_sec=2.5,
        default_sets=2,
        default_reps=8,
        anti_cheat_checks=("TOO_FAST",),
        hysteresis_enter_margin_deg=5.0,
        hysteresis_exit_margin_deg=10.0,
        rest_hysteresis_margin_deg=12.0,
    ),
    ClinicalExerciseRule(
        id="STROKE_THUMB_PINCH_R",
        code="STROKE_THUMB_PINCH_R",
        name_th="[Stroke ADL] จีบนิ้วโป้งแตะปลายนิ้วมือขวา (ฝึกติดกระดุมเสื้อ/รูดซิป)",
        name_en="Stroke ADL: Right Thumb-to-Finger Fine Pinch",
        category="upper_limb",
        side="right",
        pattern="wrist",
        metric_kind="thumb_pinch",
        target_joint="right_shoulder",
        min_angle_deg=35.0,
        max_angle_deg=80.0,
        rest_angle_deg=14.0,
        default_hold_sec=2.5,
        default_sets=2,
        default_reps=8,
        anti_cheat_checks=("TOO_FAST",),
        hysteresis_enter_margin_deg=5.0,
        hysteresis_exit_margin_deg=10.0,
        rest_hysteresis_margin_deg=12.0,
    ),
    ClinicalExerciseRule(
        id="STROKE_THUMB_PINCH_L",
        code="STROKE_THUMB_PINCH_L",
        name_th="[Stroke ADL] จีบนิ้วโป้งแตะปลายนิ้วมือซ้าย (ฝึกติดกระดุมเสื้อ/รูดซิป)",
        name_en="Stroke ADL: Left Thumb-to-Finger Fine Pinch",
        category="upper_limb",
        side="left",
        pattern="wrist",
        metric_kind="thumb_pinch",
        target_joint="left_shoulder",
        min_angle_deg=35.0,
        max_angle_deg=80.0,
        rest_angle_deg=14.0,
        default_hold_sec=2.5,
        default_sets=2,
        default_reps=8,
        anti_cheat_checks=("TOO_FAST",),
        hysteresis_enter_margin_deg=5.0,
        hysteresis_exit_margin_deg=10.0,
        rest_hysteresis_margin_deg=12.0,
    ),
    # --- Category C: MOPH Elderly & Stroke Neck & Shoulder Girdle (4) ---
    ClinicalExerciseRule(
        id="MOPH_NECK_TILT_R",
        code="MOPH_NECK_TILT_R",
        name_th="[สธ. ผู้สูงอายุ] ยืดกล้ามเนื้อคอและบ่า เอียงคอขวาช้าๆ",
        name_en="MOPH Elderly: Gentle Right Lateral Neck Stretch",
        category="neck_trunk",
        side="right",
        pattern="neck",
        metric_kind="neck_lateral_tilt",
        target_joint="spine",
        min_angle_deg=18.0,
        max_angle_deg=45.0,
        rest_angle_deg=8.0,
        default_hold_sec=3.0,
        default_sets=2,
        default_reps=5,
        anti_cheat_checks=("SHOULDER_HIKE", "TOO_FAST"),
        hysteresis_enter_margin_deg=4.0,
        hysteresis_exit_margin_deg=8.0,
        rest_hysteresis_margin_deg=10.0,
    ),
    ClinicalExerciseRule(
        id="MOPH_NECK_TILT_L",
        code="MOPH_NECK_TILT_L",
        name_th="[สธ. ผู้สูงอายุ] ยืดกล้ามเนื้อคอและบ่า เอียงคอซ้ายช้าๆ",
        name_en="MOPH Elderly: Gentle Left Lateral Neck Stretch",
        category="neck_trunk",
        side="left",
        pattern="neck",
        metric_kind="neck_lateral_tilt",
        target_joint="spine",
        min_angle_deg=18.0,
        max_angle_deg=45.0,
        rest_angle_deg=8.0,
        default_hold_sec=3.0,
        default_sets=2,
        default_reps=5,
        anti_cheat_checks=("SHOULDER_HIKE", "TOO_FAST"),
        hysteresis_enter_margin_deg=4.0,
        hysteresis_exit_margin_deg=8.0,
        rest_hysteresis_margin_deg=10.0,
    ),
    ClinicalExerciseRule(
        id="MOPH_NECK_TURN_BILATERAL",
        code="MOPH_NECK_TURN_BILATERAL",
        name_th="[Stroke & สธ.] หันศีรษะมองซ้าย-ขวาช้าๆ (กระตุ้นการรับรู้ภาพด้านที่อ่อนแรง)",
        name_en="Stroke & MOPH: Bilateral Neck Rotation Scanning",
        category="neck_trunk",
        side="both",
        pattern="neck",
        metric_kind="neck_yaw_turn",
        target_joint="spine",
        min_angle_deg=30.0,
        max_angle_deg=75.0,
        rest_angle_deg=12.0,
        default_hold_sec=2.0,
        default_sets=2,
        default_reps=6,
        anti_cheat_checks=("SHOULDER_HIKE", "TOO_FAST"),
        hysteresis_enter_margin_deg=5.0,
        hysteresis_exit_margin_deg=10.0,
        rest_hysteresis_margin_deg=12.0,
    ),
    ClinicalExerciseRule(
        id="MOPH_SHOULDER_SHRUG_ROLL",
        code="MOPH_SHOULDER_SHRUG_ROLL",
        name_th="[สธ. ผู้สูงอายุ] ท่ายักไหล่และหมุนหัวไหล่ผ่อนคลายบ่า",
        name_en="MOPH Elderly: Shoulder Shrug & Scapular Roll",
        category="neck_trunk",
        side="both",
        pattern="shoulder",
        metric_kind="shoulder_shrug_roll",
        target_joint="right_shoulder",
        min_angle_deg=32.0,
        max_angle_deg=65.0,
        rest_angle_deg=12.0,
        default_hold_sec=2.0,
        default_sets=2,
        default_reps=8,
        anti_cheat_checks=("TRUNK_LEAN", "TOO_FAST"),
        hysteresis_enter_margin_deg=5.0,
        hysteresis_exit_margin_deg=10.0,
        rest_hysteresis_margin_deg=12.0,
    ),
]

ALL_CLINICAL_EXERCISES: List[ClinicalExerciseRule] = [*ALL_26_EXERCISES]

EXERCISE_BY_ID: Dict[str, ClinicalExerciseRule] = {rule.id: rule for rule in ALL_CLINICAL_EXERCISES}
EXERCISE_BY_CODE: Dict[str, ClinicalExerciseRule] = {rule.code: rule for rule in ALL_CLINICAL_EXERCISES}

# Canonical + legacy compatibility alias lookup map
EXERCISE_ALIASES: Dict[str, str] = {
    "right_shoulder_abduction": "STROKE_SH_ABD_R",
    "left_shoulder_abduction": "STROKE_SH_ABD_L",
    "right_shoulder_flexion": "STROKE_SH_FLEX_R",
    "left_shoulder_flexion": "STROKE_SH_FLEX_L",
    "elbow_flexion_right": "STROKE_ELB_FLEX_EXT_R",
    "elbow_flexion_left": "STROKE_ELB_FLEX_EXT_L",
    "stroke_sh_flex": "STROKE_SH_FLEX_R",
    "stroke_sh_abd": "STROKE_SH_ABD_R",
    "stroke_cross_body": "STROKE_CROSS_BODY_R",
    "stroke_scap_protract": "STROKE_SCAP_PROTRACT_R",
    "stroke_elb_flex_ext": "STROKE_ELB_FLEX_EXT_R",
    "stroke_hand_to_head": "STROKE_HAND_TO_HEAD_R",
    "stroke_forearm_turn": "STROKE_FOREARM_TURN_R",
    "stroke_sh_rot": "STROKE_SH_ROT_R",
    "stroke_wrist_finger": "STROKE_WRIST_FINGER_R",
    "stroke_thumb_pinch": "STROKE_THUMB_PINCH_R",
    "moph_neck_tilt": "MOPH_NECK_TILT_R",
    "moph_neck_turn_bilateral": "MOPH_NECK_TURN_BILATERAL",
    "moph_shoulder_shrug_roll": "MOPH_SHOULDER_SHRUG_ROLL",
    "moph_bilateral_arm_raise": "STROKE_BOBATH_ARM_RAISE",
    # Legacy numeric IDs remapped safely to upper-body canonical exercises
    "ex-01": "STROKE_SH_FLEX_R",
    "ex-02": "STROKE_SH_FLEX_L",
    "ex-03": "STROKE_SH_ABD_R",
    "ex-04": "STROKE_SH_ABD_L",
    "ex-05": "STROKE_ELB_FLEX_EXT_R",
    "ex-06": "STROKE_ELB_FLEX_EXT_L",
    "ex-07": "STROKE_SCAP_PROTRACT_R",
    "ex-08": "STROKE_SCAP_PROTRACT_L",
    "ex-09": "STROKE_SH_FLEX_R",
    "ex-10": "STROKE_SH_FLEX_L",
    "ex-11": "STROKE_SH_ROT_R",
    "ex-12": "STROKE_SH_ROT_L",
    "ex-19": "MOPH_NECK_TILT_R",
    "ex-20": "MOPH_NECK_TILT_L",
    "ex-27": "STROKE_SH_FLEX_R",
    "ex-28": "STROKE_SH_FLEX_L",
    "ex-29": "STROKE_SH_ABD_R",
    "ex-30": "STROKE_SH_ABD_L",
    "ex-31": "STROKE_ELB_FLEX_EXT_R",
    "ex-32": "STROKE_ELB_FLEX_EXT_L",
    "ex-33": "STROKE_SH_ROT_R",
    "ex-34": "STROKE_SH_ROT_L",
    "ex-35": "STROKE_WRIST_FINGER_R",
    "ex-36": "STROKE_WRIST_FINGER_L",
    "ex-37": "MOPH_NECK_TILT_R",
    "ex-38": "MOPH_NECK_TILT_L",
    "ex-39": "MOPH_SHOULDER_SHRUG_ROLL",
    "ex-40": "STROKE_BOBATH_ARM_RAISE",
}


def resolve_exercise_rule(identifier: str) -> ClinicalExerciseRule:
    """Resolve exercise by canonical ID/code ('STROKE_SH_FLEX_R'), frontend alias, or legacy ID."""
    norm = (identifier or "").strip()
    if norm in EXERCISE_BY_ID:
        return EXERCISE_BY_ID[norm]
    if norm.upper() in EXERCISE_BY_CODE:
        return EXERCISE_BY_CODE[norm.upper()]
    alias_id = EXERCISE_ALIASES.get(norm.lower())
    if alias_id and alias_id in EXERCISE_BY_ID:
        return EXERCISE_BY_ID[alias_id]
    return EXERCISE_BY_ID["STROKE_SH_ABD_R"]


def list_all_exercises(include_stroke_moph: bool = True) -> List[Dict[str, object]]:
    """Return serializable metadata for all 26 standardized Stroke ADL & MOPH upper-body exercises."""
    source = ALL_26_EXERCISES
    return [
        {
            "id": r.id,
            "code": r.code,
            "name_th": r.name_th,
            "name_en": r.name_en,
            "category": r.category,
            "side": r.side,
            "metric_kind": r.metric_kind,
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
        for r in source
    ]
