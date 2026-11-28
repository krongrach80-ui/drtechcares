"""Complete 26-exercise clinical Elderly Care & MOPH biomechanics ruleset synced with Dr.Tech.Care."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
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

    def is_in_rest_zone(self, rom_deg: float, target_override: Optional[float] = None) -> bool:
        """Hysteresis-stabilized return-to-rest boundary required before next rep."""
        rest = float(self.rest_angle_deg)
        margin = float(self.rest_hysteresis_margin_deg)
        if self.is_descending_target:
            return rom_deg >= rest - margin
        target = float(target_override) if target_override is not None else float(self.min_angle_deg)
        # Ascending exercises with elevated rest posture (e.g. ELDERLY_CLASPED_FORWARD_PUSH rest=95 vs target=135)
        # must allow returning to rest_angle_deg + margin; standard 0-start exercises enforce strict < 45% target.
        if rest >= 50.0 and target > rest:
            return rom_deg <= max(target * 0.45, rest + margin)
        return rom_deg < target * 0.45

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


ALL_15_EXERCISES: List[ClinicalExerciseRule] = [
    # --- Category A: Elderly Rehab Shoulder & Arm (7 Unified Master Exercises) ---
    ClinicalExerciseRule(
        id="ELDERLY_SH_FLEX",
        code="ELDERLY_SH_FLEX",
        name_th="[ผู้สูงอายุ] ยกแขนไปด้านหน้า (ฝึกเอื้อม/สอดแขนเสื้อ)",
        name_en="Elderly Care: Forward Shoulder Flexion",
        category="upper_limb",
        side="right",
        pattern="shoulder",
        metric_kind="shoulder_elevation",
        target_joint="right_shoulder",
        min_angle_deg=90.0,
        max_angle_deg=180.0,
        rest_angle_deg=30.0,
        default_hold_sec=3.0,
        default_sets=2,
        default_reps=6,
        anti_cheat_checks=("TRUNK_LEAN", "SHOULDER_HIKE", "ELBOW_FLEXION", "TOO_FAST"),
    ),
    ClinicalExerciseRule(
        id="ELDERLY_SH_ABD",
        code="ELDERLY_SH_ABD",
        name_th="[ผู้สูงอายุ] กางแขนด้านข้าง (ฝึกกางแขนสวมเสื้อ)",
        name_en="Elderly Care: Shoulder Abduction",
        category="upper_limb",
        side="right",
        pattern="shoulder",
        metric_kind="shoulder_elevation",
        target_joint="right_shoulder",
        min_angle_deg=85.0,
        max_angle_deg=180.0,
        rest_angle_deg=30.0,
        default_hold_sec=3.0,
        default_sets=2,
        default_reps=6,
        anti_cheat_checks=("TRUNK_LEAN", "SHOULDER_HIKE", "TOO_FAST"),
    ),
    ClinicalExerciseRule(
        id="ELDERLY_CROSS_BODY",
        code="ELDERLY_CROSS_BODY",
        name_th="[ผู้สูงอายุ] เอื้อมมือข้ามลำตัวแตะไหล่ตรงข้าม (ฝึกดึงแขนเสื้อ/ใส่เสื้อคลุม)",
        name_en="Elderly Care: Cross-Body Shoulder Reach",
        category="upper_limb",
        side="right",
        pattern="shoulder",
        metric_kind="cross_body_reach",
        target_joint="right_shoulder",
        min_angle_deg=75.0,
        max_angle_deg=180.0,
        rest_angle_deg=25.0,
        default_hold_sec=3.0,
        default_sets=2,
        default_reps=6,
        anti_cheat_checks=("TRUNK_LEAN", "SHOULDER_HIKE", "TOO_FAST"),
    ),
    ClinicalExerciseRule(
        id="ELDERLY_SH_ROT",
        code="ELDERLY_SH_ROT",
        name_th="[ผู้สูงอายุ] หมุนข้อไหล่เข้า-ออก (อ้อมใส่เสื้อ/ติดกระดุม)",
        name_en="Elderly Care: Shoulder External/Internal Rotation",
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
        id="ELDERLY_SCAP_PROTRACT",
        code="ELDERLY_SCAP_PROTRACT",
        name_th="[ผู้สูงอายุ] เหยียดแขนดันไปด้านหน้าระดับอก (ป้องกันข้อไหล่หลุด)",
        name_en="Elderly Care: Scapular Protraction Punch",
        category="upper_limb",
        side="right",
        pattern="shoulder",
        metric_kind="scapular_protraction",
        target_joint="right_shoulder",
        min_angle_deg=70.0,
        max_angle_deg=180.0,
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
        id="ELDERLY_CLASPED_ARM_RAISE",
        code="ELDERLY_CLASPED_ARM_RAISE",
        name_th="[สธ. ผู้สูงอายุ] ประสานมือสองข้างชูขึ้นเหนือศีรษะ (ยืดเหยียดแขนและไหล่สองข้าง)",
        name_en="MOPH Elderly: Bilateral Clasped Overhead Arm Raise",
        category="upper_limb",
        side="both",
        pattern="bilateral",
        metric_kind="bilateral_arm_raise",
        target_joint="right_shoulder",
        min_angle_deg=105.0,
        max_angle_deg=180.0,
        rest_angle_deg=35.0,
        default_hold_sec=3.0,
        default_sets=2,
        default_reps=6,
        anti_cheat_checks=("TRUNK_LEAN", "TOO_FAST"),
    ),
    ClinicalExerciseRule(
        id="ELDERLY_CLASPED_FORWARD_PUSH",
        code="ELDERLY_CLASPED_FORWARD_PUSH",
        name_th="[ผู้สูงอายุ] ประสานมือสองข้างดันเหยียดแขนไปด้านหน้า (ลดเกร็งพับศอก)",
        name_en="Elderly Care: Bilateral Clasped Forward Elbow Push",
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
    # --- Category B: Elderly Care Elbow, Forearm, Wrist & Fingers (5 Unified Master Exercises) ---
    ClinicalExerciseRule(
        id="ELDERLY_ELB_FLEX_EXT",
        code="ELDERLY_ELB_FLEX_EXT",
        name_th="[ผู้สูงอายุ] งอ-เหยียดข้อศอก (ดึงแขนเสื้อ/ป้อนอาหาร)",
        name_en="Elderly Care: Elbow Flexion & Extension",
        category="upper_limb",
        side="right",
        pattern="elbow",
        metric_kind="elbow_flexion",
        target_joint="right_shoulder",
        min_angle_deg=95.0,
        max_angle_deg=180.0,
        rest_angle_deg=25.0,
        default_hold_sec=2.0,
        default_sets=2,
        default_reps=8,
        anti_cheat_checks=("TRUNK_LEAN", "SHOULDER_HIKE", "TOO_FAST"),
        hysteresis_enter_margin_deg=6.0,
        hysteresis_exit_margin_deg=14.0,
        rest_hysteresis_margin_deg=14.0,
    ),
    ClinicalExerciseRule(
        id="ELDERLY_HAND_TO_HEAD",
        code="ELDERLY_HAND_TO_HEAD",
        name_th="[ผู้สูงอายุ] ยกมือแตะศีรษะ (ฝึกสวมเสื้อทางศีรษะ/หวีผม)",
        name_en="Elderly Care: Hand-to-Head Grooming Reach",
        category="upper_limb",
        side="right",
        pattern="elbow",
        metric_kind="hand_to_head",
        target_joint="right_shoulder",
        min_angle_deg=95.0,
        max_angle_deg=180.0,
        rest_angle_deg=35.0,
        default_hold_sec=3.0,
        default_sets=2,
        default_reps=6,
        anti_cheat_checks=("TRUNK_LEAN", "SHOULDER_HIKE", "TOO_FAST"),
    ),
    ClinicalExerciseRule(
        id="ELDERLY_FOREARM_TURN",
        code="ELDERLY_FOREARM_TURN",
        name_th="[ผู้สูงอายุ] พลิกหงาย-คว่ำฝ่ามือ (ฝึกถือช้อน/บิดลูกบิดประตู)",
        name_en="Elderly Care: Forearm Supination & Pronation",
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
        id="ELDERLY_WRIST_FINGER",
        code="ELDERLY_WRIST_FINGER",
        name_th="[ผู้สูงอายุ] กระดกข้อมือ & กาง-กำนิ้วมือ (ลดเกร็งนิ้วมือ)",
        name_en="Elderly Care: Wrist Extension & Finger Open/Close",
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
        id="ELDERLY_THUMB_PINCH",
        code="ELDERLY_THUMB_PINCH",
        name_th="[ผู้สูงอายุ] จีบนิ้วโป้งแตะปลายนิ้วมือ (ฝึกติดกระดุมเสื้อ/รูดซิป)",
        name_en="Elderly Care: Thumb-to-Finger Fine Pinch",
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
    # --- Category C: MOPH Elderly Neck & Shoulder Girdle (3 Unified Master Exercises) ---
    ClinicalExerciseRule(
        id="MOPH_NECK_TILT",
        code="MOPH_NECK_TILT",
        name_th="[สธ. ผู้สูงอายุ] ยืดกล้ามเนื้อคอและบ่า เอียงคอช้าๆ",
        name_en="MOPH Elderly: Gentle Lateral Neck Stretch",
        category="neck_trunk",
        side="both",
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
        id="MOPH_NECK_TURN",
        code="MOPH_NECK_TURN",
        name_th="[สธ. ผู้สูงอายุ] หันศีรษะมองซ้าย-ขวาช้าๆ (กระตุ้นการรับรู้ภาพด้านที่อ่อนแรง)",
        name_en="MOPH Elderly: Neck Rotation Visual Scanning",
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

# Backward-compatible symbol referencing the 15 Unified Master Exercises
ALL_26_EXERCISES: List[ClinicalExerciseRule] = ALL_15_EXERCISES

ALL_CLINICAL_EXERCISES: List[ClinicalExerciseRule] = [*ALL_15_EXERCISES]

EXERCISE_BY_ID: Dict[str, ClinicalExerciseRule] = {rule.id: rule for rule in ALL_CLINICAL_EXERCISES}
EXERCISE_BY_CODE: Dict[str, ClinicalExerciseRule] = {rule.code: rule for rule in ALL_CLINICAL_EXERCISES}

# Canonical + legacy compatibility alias lookup map (returns effective code with optional side suffix)
EXERCISE_ALIASES: Dict[str, str] = {
    "right_shoulder_abduction": "ELDERLY_SH_ABD_RIGHT",
    "left_shoulder_abduction": "ELDERLY_SH_ABD_LEFT",
    "right_shoulder_flexion": "ELDERLY_SH_FLEX_RIGHT",
    "left_shoulder_flexion": "ELDERLY_SH_FLEX_LEFT",
    "elbow_flexion_right": "ELDERLY_ELB_FLEX_EXT_RIGHT",
    "elbow_flexion_left": "ELDERLY_ELB_FLEX_EXT_LEFT",
    "elderly_sh_flex": "ELDERLY_SH_FLEX",
    "elderly_sh_abd": "ELDERLY_SH_ABD",
    "elderly_cross_body": "ELDERLY_CROSS_BODY",
    "elderly_scap_protract": "ELDERLY_SCAP_PROTRACT",
    "elderly_elb_flex_ext": "ELDERLY_ELB_FLEX_EXT",
    "elderly_hand_to_head": "ELDERLY_HAND_TO_HEAD",
    "elderly_forearm_turn": "ELDERLY_FOREARM_TURN",
    "elderly_sh_rot": "ELDERLY_SH_ROT",
    "elderly_wrist_finger": "ELDERLY_WRIST_FINGER",
    "elderly_thumb_pinch": "ELDERLY_THUMB_PINCH",
    "moph_neck_tilt": "MOPH_NECK_TILT",
    "moph_neck_turn": "MOPH_NECK_TURN",
    "moph_neck_turn_bilateral": "MOPH_NECK_TURN_BOTH",
    "moph_shoulder_shrug_roll": "MOPH_SHOULDER_SHRUG_ROLL_BOTH",
    "moph_bilateral_arm_raise": "ELDERLY_CLASPED_ARM_RAISE_BOTH",
    # Legacy numeric IDs remapped safely to upper-body canonical exercises
    "ex-01": "ELDERLY_SH_FLEX_RIGHT",
    "ex-02": "ELDERLY_SH_FLEX_LEFT",
    "ex-03": "ELDERLY_SH_ABD_RIGHT",
    "ex-04": "ELDERLY_SH_ABD_LEFT",
    "ex-05": "ELDERLY_ELB_FLEX_EXT_RIGHT",
    "ex-06": "ELDERLY_ELB_FLEX_EXT_LEFT",
    "ex-07": "ELDERLY_SCAP_PROTRACT_RIGHT",
    "ex-08": "ELDERLY_SCAP_PROTRACT_LEFT",
    "ex-09": "ELDERLY_SH_FLEX_RIGHT",
    "ex-10": "ELDERLY_SH_FLEX_LEFT",
    "ex-11": "ELDERLY_SH_ROT_RIGHT",
    "ex-12": "ELDERLY_SH_ROT_LEFT",
    "ex-19": "MOPH_NECK_TILT_RIGHT",
    "ex-20": "MOPH_NECK_TILT_LEFT",
    "ex-27": "ELDERLY_SH_FLEX_RIGHT",
    "ex-28": "ELDERLY_SH_FLEX_LEFT",
    "ex-29": "ELDERLY_SH_ABD_RIGHT",
    "ex-30": "ELDERLY_SH_ABD_LEFT",
    "ex-31": "ELDERLY_ELB_FLEX_EXT_RIGHT",
    "ex-32": "ELDERLY_ELB_FLEX_EXT_LEFT",
    "ex-33": "ELDERLY_SH_ROT_RIGHT",
    "ex-34": "ELDERLY_SH_ROT_LEFT",
    "ex-35": "ELDERLY_WRIST_FINGER_RIGHT",
    "ex-36": "ELDERLY_WRIST_FINGER_LEFT",
    "ex-37": "MOPH_NECK_TILT_RIGHT",
    "ex-38": "MOPH_NECK_TILT_LEFT",
    "ex-39": "MOPH_SHOULDER_SHRUG_ROLL_BOTH",
    "ex-40": "ELDERLY_CLASPED_ARM_RAISE_BOTH",
}


def parse_exercise_identifier(
    identifier: str,
    side_override: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    """Extract canonical base exercise ID and side ('left' | 'right' | 'both') from any code/alias."""
    norm = (identifier or "").strip()
    mapped = EXERCISE_ALIASES.get(norm.lower(), norm)
    upper = mapped.upper()
    parsed_side: Optional[str] = None

    if upper == "MOPH_NECK_TURN_BILATERAL":
        base = "MOPH_NECK_TURN"
        parsed_side = "both"
    elif upper.endswith("_LEFT"):
        base = upper[:-5]
        parsed_side = "left"
    elif upper.endswith("_RIGHT"):
        base = upper[:-6]
        parsed_side = "right"
    elif upper.endswith("_BOTH"):
        base = upper[:-5]
        parsed_side = "both"
    elif upper.endswith("_L") and len(upper) > 2:
        base = upper[:-2]
        parsed_side = "left"
    elif upper.endswith("_R") and len(upper) > 2:
        base = upper[:-2]
        parsed_side = "right"
    else:
        base = upper

    if side_override:
        so = side_override.strip().lower()
        if so in ("left", "l"):
            parsed_side = "left"
        elif so in ("right", "r"):
            parsed_side = "right"
        elif so in ("both", "b", "bilateral"):
            parsed_side = "both"

    return base, parsed_side


def resolve_exercise_rule(
    identifier: str,
    side_override: Optional[str] = None,
) -> ClinicalExerciseRule:
    """Resolve unified master exercise by ID/code, effective code ('ELDERLY_SH_FLEX_LEFT'), or legacy alias."""
    base_code, parsed_side = parse_exercise_identifier(identifier, side_override=side_override)
    base_rule = (
        EXERCISE_BY_ID.get(base_code)
        or EXERCISE_BY_CODE.get(base_code)
        or EXERCISE_BY_ID["ELDERLY_SH_ABD"]
    )
    resolved_side = parsed_side or base_rule.side
    if resolved_side == base_rule.side:
        return base_rule
    if base_rule.target_joint == "spine":
        resolved_joint = "spine"
    elif resolved_side == "left":
        resolved_joint = "left_shoulder"
    else:
        resolved_joint = "right_shoulder"
    return replace(base_rule, side=resolved_side, target_joint=resolved_joint)


def list_all_exercises(include_elderly_moph: bool = True) -> List[Dict[str, object]]:
    """Return serializable metadata for all 15 unified Elderly Care & MOPH upper-body master exercises."""
    source = ALL_15_EXERCISES
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
