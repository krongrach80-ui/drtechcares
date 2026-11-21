"""MediaPipe Pose (Heavy) & RTMW3D-l 3D pose estimator with dual 2D/3D timestamped One-Euro Filter."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import signal as scipy_signal  # Clinical DSP verification & cutoff helpers

from app.config import EdgeVisionSettings
from app.schemas.telemetry import HandRaiseTelemetry, Keypoint3D

logger = logging.getLogger("edge_vision.pose_3d")

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore

try:
    import mediapipe as mp  # type: ignore
except ImportError:  # pragma: no cover
    mp = None  # type: ignore

try:
    from mediapipe.tasks import python as mp_python  # type: ignore
    from mediapipe.tasks.python import vision as mp_vision  # type: ignore
except ImportError:  # pragma: no cover
    mp_python = None  # type: ignore
    mp_vision = None  # type: ignore

try:
    import onnxruntime as ort  # type: ignore
except ImportError:  # pragma: no cover
    ort = None  # type: ignore

try:
    from rtmlib import Wholebody3d  # type: ignore
except ImportError:  # pragma: no cover
    Wholebody3d = None  # type: ignore

TOTAL_WHOLEBODY_KEYPOINTS = 133
TOTAL_MEDIAPIPE_KEYPOINTS = 33

# Mapping from MediaPipe 33 Pose landmark index -> COCO-Wholebody 133 index
MEDIAPIPE_TO_WHOLEBODY_INDEX: Dict[int, int] = {
    0: 0,   # nose
    2: 1,   # left_eye
    5: 2,   # right_eye
    7: 3,   # left_ear
    8: 4,   # right_ear
    11: 5,  # left_shoulder
    12: 6,  # right_shoulder
    13: 7,  # left_elbow
    14: 8,  # right_elbow
    15: 9,  # left_wrist
    16: 10, # right_wrist
    23: 11, # left_hip
    24: 12, # right_hip
    25: 13, # left_knee
    26: 14, # right_knee
    27: 15, # left_ankle
    28: 16, # right_ankle
    29: 19, # left_heel
    30: 22, # right_heel
    31: 17, # left_big_toe
    32: 20, # right_big_toe
}

WHOLEBODY_TO_MEDIAPIPE_INDEX: Dict[int, int] = {wb: mp_i for mp_i, wb in MEDIAPIPE_TO_WHOLEBODY_INDEX.items()}

BODY_KEYPOINT_NAMES: Dict[int, str] = {
    0: "nose",
    1: "left_eye",
    2: "right_eye",
    3: "left_ear",
    4: "right_ear",
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
    17: "left_big_toe",
    18: "left_small_toe",
    19: "left_heel",
    20: "right_big_toe",
    21: "right_small_toe",
    22: "right_heel",
}


def keypoint_name(index: int) -> str:
    if index in BODY_KEYPOINT_NAMES:
        return BODY_KEYPOINT_NAMES[index]
    if 23 <= index <= 90:
        return f"face_{index - 23}"
    if 91 <= index <= 111:
        return f"left_hand_{index - 91}"
    if 112 <= index <= 132:
        return f"right_hand_{index - 112}"
    return f"kp_{index}"


class OneEuroFilter:
    """
    Real-time vectorized One-Euro Filter (1€ Filter) using actual frame timestamps (`perf_counter`).
    Supports 2D `(N, 2/3)` `pose_landmarks`, 3D `(N, 3)` `pose_world_landmarks`, and scalar signals,
    with visibility gating to prevent occluded/low-confidence landmarks from corrupting derivatives.
    """

    def __init__(
        self,
        min_cutoff: float = 1.15,
        beta: float = 0.06,
        d_cutoff: float = 1.0,
        min_visibility: float = 0.0,
    ) -> None:
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.min_visibility = float(min_visibility)
        self._x_prev: Optional[np.ndarray] = None
        self._dx_prev: Optional[np.ndarray] = None
        self._last_ts: Optional[float] = None

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = None
        self._last_ts = None

    @staticmethod
    def _alpha(cutoff: np.ndarray | float, dt: float) -> np.ndarray | float:
        tau = 1.0 / (2.0 * math.pi * np.maximum(cutoff, 1e-4))
        return 1.0 / (1.0 + tau / dt)

    def __call__(
        self,
        values: np.ndarray | float,
        timestamp_sec: Optional[float] = None,
        visibility: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        return self.filter(values, timestamp_sec=timestamp_sec, visibility=visibility)

    def filter(
        self,
        values: np.ndarray | float,
        timestamp_sec: Optional[float] = None,
        visibility: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        ts = float(time.perf_counter() if timestamp_sec is None else timestamp_sec)
        x = np.asarray(values, dtype=np.float64)
        scalar_mode = x.ndim == 0
        if scalar_mode:
            x = x.reshape(1, 1)
        elif x.ndim == 1:
            x = x.reshape(1, -1)

        if self._x_prev is None or self._x_prev.shape != x.shape or self._last_ts is None:
            self._x_prev = x.copy()
            self._dx_prev = np.zeros_like(x)
            self._last_ts = ts
            out = x.copy()
            return out.reshape(()) if scalar_mode else out

        raw_dt = ts - self._last_ts
        dt = max(1e-4, min(0.25, raw_dt if raw_dt > 0.0 else 1.0 / 60.0))
        self._last_ts = ts

        # Hold prior smoothed coordinate when landmark visibility drops below threshold
        effective_x = x
        vis_mask: Optional[np.ndarray] = None
        if visibility is not None and self.min_visibility > 0.0:
            vis_arr = np.asarray(visibility, dtype=np.float64).reshape(-1)
            if vis_arr.shape[0] == x.shape[0]:
                vis_mask = vis_arr >= self.min_visibility
                if not np.all(vis_mask):
                    effective_x = x.copy()
                    effective_x[~vis_mask] = self._x_prev[~vis_mask]

        dx = (effective_x - self._x_prev) / dt
        alpha_d = self._alpha(self.d_cutoff, dt)
        dx_hat = alpha_d * dx + (1.0 - alpha_d) * self._dx_prev

        speed = np.linalg.norm(dx_hat, axis=-1, keepdims=True)
        cutoff = self.min_cutoff + self.beta * speed
        alpha = self._alpha(cutoff, dt)
        x_hat = alpha * effective_x + (1.0 - alpha) * self._x_prev

        if vis_mask is not None and not np.all(vis_mask):
            x_hat[~vis_mask] = self._x_prev[~vis_mask]
            dx_hat[~vis_mask] = 0.0

        self._x_prev = x_hat.copy()
        self._dx_prev = dx_hat.copy()
        return x_hat.reshape(()) if scalar_mode else x_hat


def filter_landmarks_by_visibility(
    landmarks_xyz: np.ndarray,
    visibilities: Optional[np.ndarray],
    min_visibility: float = 0.70,
    fallback_xyz: Optional[np.ndarray] = None,
    required_indices: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Screen unreliable landmarks with visibility < `min_visibility` prior to biomechanics calculation.
    Returns `(cleaned_xyz, valid_mask, min_required_visibility)`.
    """
    pts = np.asarray(landmarks_xyz, dtype=np.float64).copy()
    n = pts.shape[0]
    if visibilities is None:
        return pts, np.ones((n,), dtype=bool), 1.0

    vis = np.asarray(visibilities, dtype=np.float64).reshape(-1)
    if vis.shape[0] < n:
        padded_v = np.ones((n,), dtype=np.float64)
        padded_v[: vis.shape[0]] = vis
        vis = padded_v
    elif vis.shape[0] > n:
        vis = vis[:n]

    valid_mask = vis >= float(min_visibility)
    if not np.all(valid_mask):
        if fallback_xyz is not None and np.asarray(fallback_xyz).shape == pts.shape:
            fb = np.asarray(fallback_xyz, dtype=np.float64)
            pts[~valid_mask] = fb[~valid_mask]
        else:
            pts[~valid_mask] = np.nan

    if required_indices is not None:
        req = [int(i) for i in required_indices if 0 <= int(i) < n]
        min_req = float(np.min(vis[req])) if req else float(np.min(vis))
    else:
        min_req = float(np.min(vis)) if n > 0 else 0.0
    return pts, valid_mask, min_req


class OneEuroFilter3D(OneEuroFilter):
    """
    Vectorized One-Euro Filter across all 133 (X, Y, Z) coordinates:
    - Eliminates micro-jitter while holding isometric targets (min_cutoff = 1.15 Hz).
    - Adapts cutoff proportionally to 3D velocity (beta = 0.06) for near-zero fast-motion lag.
    """

    def __init__(
        self,
        num_keypoints: int = TOTAL_WHOLEBODY_KEYPOINTS,
        min_cutoff: float = 1.15,
        beta: float = 0.06,
        d_cutoff: float = 1.0,
        min_visibility: float = 0.0,
    ) -> None:
        super().__init__(
            min_cutoff=min_cutoff,
            beta=beta,
            d_cutoff=d_cutoff,
            min_visibility=min_visibility,
        )
        self.num_keypoints = int(num_keypoints)
        # Precompute reference SciPy bilinear filter check for diagnostic telemetry
        self._scipy_b, self._scipy_a = scipy_signal.butter(1, min(0.95, self.min_cutoff / 30.0), btype="low")

    def filter(
        self,
        points_xyz: np.ndarray,
        timestamp_sec: Optional[float] = None,
        visibility: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Smooth `(N, 3)` float coordinate array at `timestamp_sec` (monotonic seconds).
        Returns `(N, 3)` filtered float64 numpy array.
        """
        x = np.asarray(points_xyz, dtype=np.float64)
        if x.shape != (self.num_keypoints, 3):
            padded = np.zeros((self.num_keypoints, 3), dtype=np.float64)
            rows = min(self.num_keypoints, x.shape[0])
            cols = min(3, x.shape[1]) if x.ndim == 2 else 0
            if cols > 0 and rows > 0:
                padded[:rows, :cols] = x[:rows, :cols]
            x = padded
        vis = visibility
        if vis is not None:
            v_arr = np.asarray(vis, dtype=np.float64).reshape(-1)
            if v_arr.shape[0] != self.num_keypoints:
                v_pad = np.ones((self.num_keypoints,), dtype=np.float64)
                cnt = min(self.num_keypoints, v_arr.shape[0])
                v_pad[:cnt] = v_arr[:cnt]
                vis = v_pad
        return super().filter(x, timestamp_sec=timestamp_sec, visibility=vis)


def generate_synthetic_133_pose(
    t_sec: float,
    exercise_id: str = "STROKE_SH_ABD_R",
    is_calibrating: bool = False,
    mock_pose_override: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate realistic 133-point 3D normalized coordinates `(133, 3)` + scores `(133,)`
    for mock/headless validation (x, y in normalized [0..1] view space, z in meters relative to hip center).
    """
    pts = np.zeros((TOTAL_WHOLEBODY_KEYPOINTS, 3), dtype=np.float64)
    scores = np.full((TOTAL_WHOLEBODY_KEYPOINTS,), 0.96, dtype=np.float64)

    # Neutral standing torso centered at (0.50, y)
    pts[0] = [0.50, 0.16, -0.04]  # nose
    pts[1] = [0.515, 0.145, -0.05]  # left_eye
    pts[2] = [0.485, 0.145, -0.05]  # right_eye
    pts[3] = [0.535, 0.155, -0.01]  # left_ear
    pts[4] = [0.465, 0.155, -0.01]  # right_ear
    pts[5] = [0.585, 0.275, 0.00]  # left_shoulder
    pts[6] = [0.415, 0.275, 0.00]  # right_shoulder
    pts[11] = [0.555, 0.540, 0.00]  # left_hip
    pts[12] = [0.445, 0.540, 0.00]  # right_hip
    pts[13] = [0.555, 0.735, -0.02]  # left_knee
    pts[14] = [0.445, 0.735, -0.02]  # right_knee
    pts[15] = [0.555, 0.910, 0.00]  # left_ankle
    pts[16] = [0.445, 0.910, 0.00]  # right_ankle

    # Feet (heels & toes 17..22)
    pts[17] = [0.562, 0.938, -0.07]  # left_big_toe
    pts[18] = [0.576, 0.938, -0.06]  # left_small_toe
    pts[19] = [0.548, 0.935, 0.04]  # left_heel
    pts[20] = [0.438, 0.938, -0.07]  # right_big_toe
    pts[21] = [0.424, 0.938, -0.06]  # right_small_toe
    pts[22] = [0.452, 0.935, 0.04]  # right_heel

    # Default neutral arms hanging down
    upper_arm = 0.145
    forearm = 0.135
    l_angle_rad = math.radians(8.0)
    r_angle_rad = math.radians(8.0)
    l_forward_z = 0.0
    r_forward_z = 0.0

    if not is_calibrating:
        cycle = 0.5 * (1.0 - math.cos(t_sec * 1.45))  # 0..1 smooth periodic motion
        if mock_pose_override == "raise_left":
            l_angle_rad = math.radians(145.0)
        elif mock_pose_override == "raise_right":
            r_angle_rad = math.radians(145.0)
        elif mock_pose_override == "fall":
            pts[0, 1] += 0.36
            pts[5, 1] += 0.32
            pts[6, 1] += 0.32
            pts[11, 1] += 0.28
            pts[12, 1] += 0.28
        elif "left" in exercise_id.lower() or exercise_id.upper().endswith("_L"):
            l_angle_rad = math.radians(12.0 + cycle * 145.0)
            if "flex" in exercise_id.lower() or "scap" in exercise_id.lower():
                l_forward_z = -0.22 * cycle
        elif "bilateral" in exercise_id.lower() or "bobath" in exercise_id.lower() or "both" in exercise_id.lower():
            l_angle_rad = math.radians(12.0 + cycle * 135.0)
            r_angle_rad = math.radians(12.0 + cycle * 135.0)
        else:
            r_angle_rad = math.radians(12.0 + cycle * 145.0)
            if "flex" in exercise_id.lower() or "scap" in exercise_id.lower():
                r_forward_z = -0.22 * cycle

    # Left arm (shoulder 5 -> elbow 7 -> wrist 9)
    pts[7] = [
        pts[5, 0] + math.sin(l_angle_rad) * upper_arm,
        pts[5, 1] + math.cos(l_angle_rad) * upper_arm,
        pts[5, 2] + l_forward_z * 0.5,
    ]
    pts[9] = [
        pts[7, 0] + math.sin(l_angle_rad) * forearm,
        pts[7, 1] + math.cos(l_angle_rad) * forearm,
        pts[7, 2] + l_forward_z * 0.5,
    ]

    # Right arm (shoulder 6 -> elbow 8 -> wrist 10)
    pts[8] = [
        pts[6, 0] - math.sin(r_angle_rad) * upper_arm,
        pts[6, 1] + math.cos(r_angle_rad) * upper_arm,
        pts[6, 2] + r_forward_z * 0.5,
    ]
    pts[10] = [
        pts[8, 0] - math.sin(r_angle_rad) * forearm,
        pts[8, 1] + math.cos(r_angle_rad) * forearm,
        pts[8, 2] + r_forward_z * 0.5,
    ]

    # 68 face landmarks clustered cleanly around nose (indices 23..90)
    for idx in range(23, 91):
        theta = ((idx - 23) / 68.0) * 2.0 * math.pi
        radius_x = 0.035 if idx < 40 else 0.018
        radius_y = 0.045 if idx < 40 else 0.022
        pts[idx] = [pts[0, 0] + math.cos(theta) * radius_x, pts[0, 1] + math.sin(theta) * radius_y, pts[0, 2]]

    # 21 left-hand landmarks anchored on left_wrist (91..111)
    for f_i in range(21):
        finger = max(0, (f_i - 1) // 4)
        joint = (f_i - 1) % 4 + 1 if f_i > 0 else 0
        dx = (finger - 2) * 0.008
        dy = -joint * 0.012
        pts[91 + f_i] = [pts[9, 0] + dx, pts[9, 1] + dy, pts[9, 2]]

    # 21 right-hand landmarks anchored on right_wrist (112..132)
    for f_i in range(21):
        finger = max(0, (f_i - 1) // 4)
        joint = (f_i - 1) % 4 + 1 if f_i > 0 else 0
        dx = (finger - 2) * 0.008
        dy = -joint * 0.012
        pts[112 + f_i] = [pts[10, 0] + dx, pts[10, 1] + dy, pts[10, 2]]

    return pts, scores


def map_mediapipe33_to_wholebody133(
    landmarks_33: np.ndarray,
    visibilities_33: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Map 33 MediaPipe Pose landmarks into the 133-point Wholebody3d layout."""
    lm = np.asarray(landmarks_33, dtype=np.float64)
    vis = (
        np.asarray(visibilities_33, dtype=np.float64).reshape(-1)
        if visibilities_33 is not None
        else np.ones((TOTAL_MEDIAPIPE_KEYPOINTS,), dtype=np.float64)
    )
    out = np.zeros((TOTAL_WHOLEBODY_KEYPOINTS, 3), dtype=np.float64)
    out_sc = np.zeros((TOTAL_WHOLEBODY_KEYPOINTS,), dtype=np.float64)

    for mp_idx, wb_idx in MEDIAPIPE_TO_WHOLEBODY_INDEX.items():
        if mp_idx < lm.shape[0]:
            out[wb_idx] = lm[mp_idx, :3]
            out_sc[wb_idx] = float(vis[mp_idx]) if mp_idx < vis.shape[0] else 1.0

    # Derive small toes from big toes and hips/ears cleanly if absent
    out[18] = out[17] + np.array([0.012, 0.0, 0.0], dtype=np.float64)
    out_sc[18] = out_sc[17]
    out[21] = out[20] + np.array([-0.012, 0.0, 0.0], dtype=np.float64)
    out_sc[21] = out_sc[20]

    # Left & right hands (indices 91..111, 112..132) anchored on wrists + MediaPipe finger tips (17, 19, 21 / 18, 20, 22)
    l_wrist = out[9]
    l_tip = lm[19, :3] if lm.shape[0] > 19 else l_wrist
    l_dir = l_tip - l_wrist
    for i in range(21):
        frac = ((i - 1) % 4 + 1) / 4.0 if i > 0 else 0.0
        out[91 + i] = l_wrist + l_dir * frac
        out_sc[91 + i] = out_sc[9]

    r_wrist = out[10]
    r_tip = lm[20, :3] if lm.shape[0] > 20 else r_wrist
    r_dir = r_tip - r_wrist
    for i in range(21):
        frac = ((i - 1) % 4 + 1) / 4.0 if i > 0 else 0.0
        out[112 + i] = r_wrist + r_dir * frac
        out_sc[112 + i] = out_sc[10]

    # Face contour (23..90) anchored around nose/ears
    nose = out[0]
    for idx in range(23, 91):
        out[idx] = nose
        out_sc[idx] = out_sc[0]

    return out, np.clip(out_sc, 0.0, 1.0)


def wholebody133_to_mediapipe33(
    xyz_133: np.ndarray,
    scores_133: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project 133 Wholebody keypoints into 33 MediaPipe Pose landmarks + visibilities."""
    wb = np.asarray(xyz_133, dtype=np.float64)
    sc = (
        np.asarray(scores_133, dtype=np.float64).reshape(-1)
        if scores_133 is not None
        else np.ones((TOTAL_WHOLEBODY_KEYPOINTS,), dtype=np.float64)
    )
    out = np.zeros((TOTAL_MEDIAPIPE_KEYPOINTS, 3), dtype=np.float64)
    vis = np.zeros((TOTAL_MEDIAPIPE_KEYPOINTS,), dtype=np.float64)

    for mp_idx, wb_idx in MEDIAPIPE_TO_WHOLEBODY_INDEX.items():
        if wb_idx < wb.shape[0]:
            out[mp_idx] = wb[wb_idx, :3]
            vis[mp_idx] = float(sc[wb_idx]) if wb_idx < sc.shape[0] else 1.0

    # Fill eye inner/outer & mouth neighbors cleanly from nose/eyes
    out[1] = out[2]
    out[3] = out[2]
    out[4] = out[5]
    out[6] = out[5]
    out[9] = out[0] + np.array([0.01, 0.02, 0.0], dtype=np.float64)
    out[10] = out[0] + np.array([-0.01, 0.02, 0.0], dtype=np.float64)
    for f_i in (1, 3, 4, 6, 9, 10):
        vis[f_i] = vis[0]

    # Hand distal landmarks (17,19,21 left; 18,20,22 right) from 91..132 or wrists
    if wb.shape[0] >= 133:
        out[17], out[19], out[21] = wb[108], wb[99], wb[95]
        out[18], out[20], out[22] = wb[129], wb[120], wb[116]
        for mp_h in (17, 19, 21):
            vis[mp_h] = vis[15]
        for mp_h in (18, 20, 22):
            vis[mp_h] = vis[16]

    return out, np.clip(vis, 0.0, 1.0)


def wholebody133_to_world_meters(xyz_133: np.ndarray, scale_m: float = 1.96) -> np.ndarray:
    """Convert normalized image-space `(133, 3)` coords to isotropic hip-centered meters."""
    pts = np.asarray(xyz_133, dtype=np.float64)
    mid_hip = 0.5 * (pts[11] + pts[12])
    return (pts - mid_hip) * float(scale_m)


@dataclass
class PoseEstimationFrame:
    """Unified smoothed 2D (`pose_landmarks`) + 3D (`pose_world_landmarks`) frame container."""

    pose_landmarks_2d: np.ndarray        # (33, 3) filtered normalized 2D + relative Z
    pose_world_landmarks_3d: np.ndarray  # (33, 3) filtered metric 3D (meters, hip-centered)
    visibilities_33: np.ndarray          # (33,) visibility scores in [0..1]
    wholebody_2d_133: np.ndarray         # (133, 3) display keypoints
    wholebody_world_133: np.ndarray      # (133, 3) true 3D metric keypoints (meters)
    scores_133: np.ndarray               # (133,) scores
    skeleton_count: int
    valid_bboxes: List[List[int]]
    visibility_ok: bool
    backend_used: str


class MediaPipePoseEngine:
    """
    Clinical MediaPipe Pose engine supporting both legacy `mp.solutions.pose.Pose`
    and MediaPipe Tasks `PoseLandmarker`, paired with separate 2D & 3D `OneEuroFilter` smoothers.
    """

    def __init__(self, settings: EdgeVisionSettings) -> None:
        self.settings = settings
        self.config = settings.get_mediapipe_pose_config()
        # Keep One-Euro smoothing responsive down to visibility >= 0.15 so moving joints (elbows/wrists)
        # never freeze during partial occlusion or quick turns.
        self.filter_2d = OneEuroFilter(
            min_cutoff=settings.ONE_EURO_2D_MIN_CUTOFF,
            beta=settings.ONE_EURO_2D_BETA,
            d_cutoff=settings.ONE_EURO_D_CUTOFF,
            min_visibility=0.15,
        )
        self.filter_3d = OneEuroFilter(
            min_cutoff=settings.ONE_EURO_MIN_CUTOFF,
            beta=settings.ONE_EURO_BETA,
            d_cutoff=settings.ONE_EURO_D_CUTOFF,
            min_visibility=0.15,
        )
        self._solutions_pose: Optional[Any] = None
        self._tasks_landmarker: Optional[Any] = None
        self._last_ts_ms: int = 0
        self._backend_label: str = "MediaPipe Unavailable"
        self._init_mediapipe()

    @property
    def is_ready(self) -> bool:
        return self._solutions_pose is not None or self._tasks_landmarker is not None

    @property
    def backend_label(self) -> str:
        return self._backend_label

    def reset_filters(self) -> None:
        self.filter_2d.reset()
        self.filter_3d.reset()

    def close(self) -> None:
        if self._solutions_pose is not None:
            try:
                self._solutions_pose.close()
            except Exception:
                pass
            self._solutions_pose = None
        if self._tasks_landmarker is not None:
            try:
                self._tasks_landmarker.close()
            except Exception:
                pass
            self._tasks_landmarker = None

    def _init_mediapipe(self) -> None:
        if mp is None:
            self._backend_label = "AI Model Error (mediapipe package missing)"
            return

        cfg = self.config
        det_conf = min(0.5, float(cfg["min_detection_confidence"]))
        pres_conf = min(0.5, float(cfg["min_presence_confidence"]))
        track_conf = min(0.5, float(cfg["min_tracking_confidence"]))

        # Primary: MediaPipe Tasks Vision API PoseLandmarker (mediapipe 1.2.3+) with auto-downloaded heavy .task asset
        model_path = self.settings.resolve_mediapipe_model_path(int(cfg["model_complexity"]))
        if model_path is not None and mp_python is not None and mp_vision is not None:
            try:
                base_opts = mp_python.BaseOptions(model_asset_path=str(model_path))
                options = mp_vision.PoseLandmarkerOptions(
                    base_options=base_opts,
                    running_mode=mp_vision.RunningMode.VIDEO,
                    num_poses=int(cfg["num_poses"]),
                    min_pose_detection_confidence=det_conf,
                    min_pose_presence_confidence=pres_conf,
                    min_tracking_confidence=track_conf,
                    output_segmentation_masks=False,
                )
                self._tasks_landmarker = mp_vision.PoseLandmarker.create_from_options(options)
                self._backend_label = (
                    f"MediaPipe PoseLandmarker ({model_path.name}, complexity={cfg['model_complexity']}, "
                    f"det={det_conf:.1f}, track={track_conf:.1f})"
                )
                logger.info("Initialized %s", self._backend_label)
                return
            except Exception as exc:
                logger.warning("Failed initializing MediaPipe PoseLandmarker: %s", exc)

        # Legacy fallback only if mediapipe.tasks is unavailable on an older environment
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
            try:
                self._solutions_pose = mp.solutions.pose.Pose(
                    static_image_mode=False,
                    model_complexity=int(cfg["model_complexity"]),
                    smooth_landmarks=bool(cfg["smooth_landmarks"]),
                    enable_segmentation=bool(cfg["enable_segmentation"]),
                    min_detection_confidence=det_conf,
                    min_tracking_confidence=track_conf,
                )
                self._backend_label = (
                    f"MediaPipe Pose Solutions (complexity={cfg['model_complexity']}, "
                    f"det={det_conf:.1f}, track={track_conf:.1f})"
                )
                logger.info("Initialized %s", self._backend_label)
                return
            except Exception as exc:
                logger.warning("Failed initializing mp.solutions.pose.Pose: %s", exc)

        self._backend_label = "AI Model Error (PoseLandmarker asset unavailable)"

    @staticmethod
    def _landmarks_to_numpy(landmark_list: Sequence[Any]) -> Tuple[np.ndarray, np.ndarray]:
        pts = np.zeros((TOTAL_MEDIAPIPE_KEYPOINTS, 3), dtype=np.float64)
        vis = np.ones((TOTAL_MEDIAPIPE_KEYPOINTS,), dtype=np.float64)
        cnt = min(TOTAL_MEDIAPIPE_KEYPOINTS, len(landmark_list))
        for i in range(cnt):
            lm = landmark_list[i]
            pts[i, 0] = float(getattr(lm, "x", 0.0))
            pts[i, 1] = float(getattr(lm, "y", 0.0))
            pts[i, 2] = float(getattr(lm, "z", 0.0))
            v = getattr(lm, "visibility", None)
            p = getattr(lm, "presence", None)
            if v is not None:
                vis[i] = float(v)
            elif p is not None:
                vis[i] = float(p)
        return pts, vis

    def infer_frame(
        self,
        frame_bgr: np.ndarray,
        timestamp_sec: float,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, int, List[List[int]]]]:
        """
        Run MediaPipe Pose on a BGR image and return smoothed:
        `(pose_2d_33, pose_world_3d_33, visibility_33, person_count, bboxes_1280x720)`.
        """
        if not self.is_ready or frame_bgr is None:
            return None

        if cv2 is not None:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        else:
            rgb = frame_bgr[:, :, ::-1].copy()

        raw_2d_list: List[Tuple[np.ndarray, np.ndarray]] = []
        raw_3d_list: List[np.ndarray] = []

        if self._solutions_pose is not None:
            results = self._solutions_pose.process(rgb)
            if results.pose_landmarks and results.pose_world_landmarks:
                pts2, vis2 = self._landmarks_to_numpy(results.pose_landmarks.landmark)
                pts3, _ = self._landmarks_to_numpy(results.pose_world_landmarks.landmark)
                raw_2d_list.append((pts2, vis2))
                raw_3d_list.append(pts3)
        elif self._tasks_landmarker is not None and mp is not None:
            rgb_c = np.ascontiguousarray(rgb, dtype=np.uint8)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_c)
            ts_ms = max(self._last_ts_ms + 1, int(round(float(timestamp_sec) * 1000.0)))
            self._last_ts_ms = ts_ms
            res = self._tasks_landmarker.detect_for_video(mp_img, ts_ms)
            if res.pose_landmarks and res.pose_world_landmarks:
                count = min(len(res.pose_landmarks), len(res.pose_world_landmarks))
                for idx in range(count):
                    pts2, vis2 = self._landmarks_to_numpy(res.pose_landmarks[idx])
                    pts3, _ = self._landmarks_to_numpy(res.pose_world_landmarks[idx])
                    raw_2d_list.append((pts2, vis2))
                    raw_3d_list.append(pts3)

        if not raw_2d_list:
            return None

        # Evaluate valid human candidates & pick largest/most confident primary patient
        valid_bboxes: List[List[int]] = []
        candidates: List[Tuple[float, int, List[int]]] = []
        min_body_score = float(self.settings.MULTI_PERSON_SKELETON_MIN_SCORE)
        min_torso_ratio = float(self.settings.MULTI_PERSON_MIN_TORSO_HEIGHT_RATIO)
        for idx, (pts2, vis2) in enumerate(raw_2d_list):
            body_vis = float(np.mean(vis2[[11, 12, 13, 14, 15, 16, 23, 24, 25, 26]]))
            torso_span = float(
                max(pts2[[11, 12, 23, 24], 1]) - min(pts2[[11, 12, 23, 24], 1])
            )
            full_span = float(max(pts2[:, 1]) - min(pts2[:, 1]))
            xs = np.clip(pts2[:, 0], 0.0, 1.0)
            ys = np.clip(pts2[:, 1], 0.0, 1.0)
            bbox = [
                int(round(max(0.0, float(np.min(xs)) - 0.04) * 1280)),
                int(round(max(0.0, float(np.min(ys)) - 0.06) * 720)),
                int(round(min(1.0, float(np.max(xs)) + 0.04) * 1280)),
                int(round(min(1.0, float(np.max(ys)) + 0.04) * 720)),
            ]
            if body_vis >= min_body_score and max(torso_span, full_span * 0.5) >= min_torso_ratio:
                candidates.append((body_vis * max(full_span, 0.1), idx, bbox))
                valid_bboxes.append(bbox)

        if candidates:
            candidates.sort(key=lambda c: c[0], reverse=True)
            primary_idx = candidates[0][1]
        else:
            primary_idx = 0
            pts2_fallback = raw_2d_list[0][0]
            xs = np.clip(pts2_fallback[:, 0], 0.0, 1.0)
            ys = np.clip(pts2_fallback[:, 1], 0.0, 1.0)
            valid_bboxes = [[
                int(round(max(0.0, float(np.min(xs)) - 0.04) * 1280)),
                int(round(max(0.0, float(np.min(ys)) - 0.06) * 720)),
                int(round(min(1.0, float(np.max(xs)) + 0.04) * 1280)),
                int(round(min(1.0, float(np.max(ys)) + 0.04) * 720)),
            ]]

        primary_2d, primary_vis = raw_2d_list[primary_idx]
        primary_3d = raw_3d_list[primary_idx]

        # Dual timestamped One-Euro filtering (separate 2D image filter + 3D world metric filter)
        smoothed_2d = self.filter_2d.filter(primary_2d, timestamp_sec=timestamp_sec, visibility=primary_vis)
        smoothed_3d = self.filter_3d.filter(primary_3d, timestamp_sec=timestamp_sec, visibility=primary_vis)

        return smoothed_2d, smoothed_3d, primary_vis, max(1, len(valid_bboxes)), valid_bboxes


class RTMW3DPoseEstimator:
    """
    Unified Clinical 3D Pose Estimator supporting MediaPipe Pose (Heavy, complexity=2,
    dual 2D/3D One-Euro filtering & visibility gating) and RTMW3D-l Wholebody3d fallback.
    """

    def __init__(self, settings: EdgeVisionSettings) -> None:
        self.settings = settings
        self.filter_2d = OneEuroFilter(
            min_cutoff=settings.ONE_EURO_2D_MIN_CUTOFF,
            beta=settings.ONE_EURO_2D_BETA,
            d_cutoff=settings.ONE_EURO_D_CUTOFF,
            min_visibility=settings.MIN_LANDMARK_VISIBILITY,
        )
        self.filter_3d = OneEuroFilter3D(
            num_keypoints=TOTAL_WHOLEBODY_KEYPOINTS,
            min_cutoff=settings.ONE_EURO_MIN_CUTOFF,
            beta=settings.ONE_EURO_BETA,
            d_cutoff=settings.ONE_EURO_D_CUTOFF,
        )
        self.world_filter_3d = OneEuroFilter3D(
            num_keypoints=TOTAL_WHOLEBODY_KEYPOINTS,
            min_cutoff=settings.ONE_EURO_MIN_CUTOFF,
            beta=settings.ONE_EURO_BETA,
            d_cutoff=settings.ONE_EURO_D_CUTOFF,
        )
        self.mp_engine: Optional[MediaPipePoseEngine] = None
        self.latest_frame: Optional[PoseEstimationFrame] = None
        self._model = None
        self._device = "cpu"
        self._backend_name = "MediaPipe / RTMW3D-l (Pending Init)"
        self._load_model()

    @property
    def backend_name(self) -> str:
        return self._backend_name

    def close(self) -> None:
        if self.mp_engine is not None:
            self.mp_engine.close()

    def _resolve_device(self) -> str:
        if self.settings.USE_GPU_IF_AVAILABLE and ort is not None:
            try:
                providers = ort.get_available_providers()
                if "CUDAExecutionProvider" in providers:
                    return "cuda"
            except Exception:
                pass
        return "cpu"

    def _load_model(self) -> None:
        self._device = self._resolve_device()
        preferred = (self.settings.POSE_BACKEND or "mediapipe").strip().lower()

        if preferred in ("mediapipe", "auto"):
            self.mp_engine = MediaPipePoseEngine(self.settings)
            if self.mp_engine.is_ready:
                self._backend_name = self.mp_engine.backend_label
                return

        if Wholebody3d is None:
            if self.mp_engine is not None:
                self._backend_name = f"Synthetic133-3D ({self.mp_engine.backend_label})"
            else:
                self._backend_name = f"Synthetic133-3D (rtmlib not installed, target={self._device})"
            logger.warning("No native pose runtime loaded; using synthetic 133-point 3D fallback.")
            return

        # Attempt CUDA first if available, then fall back cleanly to CPU ONNXRuntime
        tried: List[str] = [self._device] if self._device == "cpu" else ["cuda", "cpu"]
        for dev in tried:
            try:
                self._model = Wholebody3d(
                    to_openpose=self.settings.RTMW3D_TO_OPENPOSE_KEYPOINTS,
                    mode=self.settings.RTMW3D_MODE,
                    backend="onnxruntime",
                    device=dev,
                )
                self._device = dev
                self._backend_name = f"RTMW3D-l Wholebody3d ({dev.upper()} ONNXRuntime)"
                logger.info("Loaded Wholebody3d RTMW3D-l on %s.", dev)
                return
            except Exception as exc:
                logger.warning("Failed initializing Wholebody3d on %s: %s", dev, exc)

        self._model = None
        self._backend_name = "Synthetic133-3D (Model Fallback)"

    @staticmethod
    def _bbox_from_norm_kpts(norm_xyz: np.ndarray, scores: np.ndarray) -> List[int]:
        """Compute [x1, y1, x2, y2] in 1280x720 clinical display space from normalized keypoints."""
        body_mask = scores[:23] >= 0.25 if scores.shape[0] >= 23 else scores >= 0.25
        pts = norm_xyz[:23][body_mask] if np.any(body_mask) else norm_xyz[:17]
        min_x = float(np.clip(np.min(pts[:, 0]) - 0.04, 0.0, 1.0))
        min_y = float(np.clip(np.min(pts[:, 1]) - 0.06, 0.0, 1.0))
        max_x = float(np.clip(np.max(pts[:, 0]) + 0.04, 0.0, 1.0))
        max_y = float(np.clip(np.max(pts[:, 1]) + 0.04, 0.0, 1.0))
        return [
            int(round(min_x * 1280)),
            int(round(min_y * 720)),
            int(round(max_x * 1280)),
            int(round(max_y * 720)),
        ]

    def estimate_clinical_with_crowd(
        self,
        frame_bgr: Optional[np.ndarray],
        timestamp_sec: float,
        exercise_id: str = "STROKE_SH_ABD_R",
        is_calibrating: bool = False,
        mock_pose_override: Optional[str] = None,
        is_mock_mode: bool = False,
    ) -> PoseEstimationFrame:
        """
        Estimate smoothed 2D `pose_landmarks`, 3D `pose_world_landmarks`, visibility scores,
        and Wholebody-133 keypoints for the primary subject + valid human count.
        """
        if (
            self.mp_engine is not None
            and self.mp_engine.is_ready
            and frame_bgr is not None
            and not is_mock_mode
            and mock_pose_override is None
        ):
            mp_out = self.mp_engine.infer_frame(frame_bgr, timestamp_sec)
            if mp_out is not None:
                sm_2d_33, sm_3d_33, vis_33, skel_cnt, bboxes = mp_out
                wb_2d, wb_scores = map_mediapipe33_to_wholebody133(sm_2d_33, vis_33)
                wb_world, _ = map_mediapipe33_to_wholebody133(sm_3d_33, vis_33)
                min_core_vis = float(np.min(vis_33[[11, 12, 13, 14, 15, 16, 23, 24]]))
                frame_res = PoseEstimationFrame(
                    pose_landmarks_2d=sm_2d_33,
                    pose_world_landmarks_3d=sm_3d_33,
                    visibilities_33=vis_33,
                    wholebody_2d_133=wb_2d,
                    wholebody_world_133=wb_world,
                    scores_133=wb_scores,
                    skeleton_count=skel_cnt,
                    valid_bboxes=bboxes,
                    visibility_ok=min_core_vis >= float(self.settings.MIN_LANDMARK_VISIBILITY),
                    backend_used=self.mp_engine.backend_label,
                )
                self.latest_frame = frame_res
                return frame_res
        raw_xyz: Optional[np.ndarray] = None
        scores: Optional[np.ndarray] = None
        valid_bboxes: List[List[int]] = []

        if self._model is not None and frame_bgr is not None and not is_mock_mode and mock_pose_override is None:
            try:
                h, w = frame_bgr.shape[:2]
                out = self._model(frame_bgr)
                kpts_arr = np.asarray(out[0], dtype=np.float64)
                sc_arr = (
                    np.asarray(out[1], dtype=np.float64)
                    if len(out) > 1
                    else np.ones((kpts_arr.shape[0], TOTAL_WHOLEBODY_KEYPOINTS))
                )
                if kpts_arr.ndim == 3 and kpts_arr.shape[0] > 0:
                    candidates: List[Tuple[int, np.ndarray, np.ndarray, float, List[int]]] = []
                    for idx in range(kpts_arr.shape[0]):
                        pk = kpts_arr[idx]
                        ps = sc_arr[idx]
                        norm = np.zeros((TOTAL_WHOLEBODY_KEYPOINTS, 3), dtype=np.float64)
                        cnt = min(TOTAL_WHOLEBODY_KEYPOINTS, pk.shape[0])
                        norm[:cnt, 0] = pk[:cnt, 0] / max(1.0, float(w))
                        norm[:cnt, 1] = pk[:cnt, 1] / max(1.0, float(h))
                        if pk.shape[1] >= 3:
                            norm[:cnt, 2] = pk[:cnt, 2]
                        sc_vec = np.zeros((TOTAL_WHOLEBODY_KEYPOINTS,), dtype=np.float64)
                        sc_cnt = min(TOTAL_WHOLEBODY_KEYPOINTS, ps.shape[0])
                        sc_vec[:sc_cnt] = ps[:sc_cnt]

                        body_mean = float(np.mean(sc_vec[:17]))
                        torso_pts = norm[[5, 6, 11, 12], 1]
                        torso_height_ratio = float(np.max(torso_pts) - np.min(torso_pts))
                        full_height_ratio = float(np.max(norm[:17, 1]) - np.min(norm[:17, 1]))
                        bbox = self._bbox_from_norm_kpts(norm, sc_vec)

                        if (
                            body_mean >= self.settings.MULTI_PERSON_SKELETON_MIN_SCORE
                            and max(torso_height_ratio, full_height_ratio * 0.5)
                            >= self.settings.MULTI_PERSON_MIN_TORSO_HEIGHT_RATIO
                        ):
                            candidates.append((idx, norm, sc_vec, body_mean * max(full_height_ratio, 0.1), bbox))

                    if candidates:
                        # Primary subject is highest confidence * size
                        candidates.sort(key=lambda c: c[3], reverse=True)
                        raw_xyz = candidates[0][1]
                        scores = candidates[0][2]
                        valid_bboxes = [c[4] for c in candidates]
                    else:
                        best_idx = int(np.argmax(np.mean(sc_arr, axis=1)))
                        pk = kpts_arr[best_idx]
                        ps = sc_arr[best_idx]
                        raw_xyz = np.zeros((TOTAL_WHOLEBODY_KEYPOINTS, 3), dtype=np.float64)
                        cnt = min(TOTAL_WHOLEBODY_KEYPOINTS, pk.shape[0])
                        raw_xyz[:cnt, 0] = pk[:cnt, 0] / max(1.0, float(w))
                        raw_xyz[:cnt, 1] = pk[:cnt, 1] / max(1.0, float(h))
                        if pk.shape[1] >= 3:
                            raw_xyz[:cnt, 2] = pk[:cnt, 2]
                        scores = np.zeros((TOTAL_WHOLEBODY_KEYPOINTS,), dtype=np.float64)
                        sc_cnt = min(TOTAL_WHOLEBODY_KEYPOINTS, ps.shape[0])
                        scores[:sc_cnt] = ps[:sc_cnt]
                        valid_bboxes = [self._bbox_from_norm_kpts(raw_xyz, scores)]
            except Exception as exc:
                logger.debug("RTMW3D frame inference warning: %s", exc)

        if raw_xyz is None or scores is None:
            if not is_mock_mode and mock_pose_override is None:
                # Never fabricate a procedural skeleton during live clinical mode.
                if self.mp_engine is not None:
                    self.mp_engine.reset_filters()
                if frame_bgr is None:
                    err_backend = "CAMERA_DISCONNECTED"
                elif not (self.mp_engine is not None and self.mp_engine.is_ready) and self._model is None:
                    err_backend = "AI_MODEL_ERROR"
                else:
                    err_backend = self.mp_engine.backend_label if self.mp_engine is not None else self._backend_name

                empty_frame = PoseEstimationFrame(
                    pose_landmarks_2d=np.zeros((0, 3), dtype=np.float64),
                    pose_world_landmarks_3d=np.zeros((0, 3), dtype=np.float64),
                    visibilities_33=np.zeros((TOTAL_MEDIAPIPE_KEYPOINTS,), dtype=np.float64),
                    wholebody_2d_133=np.zeros((TOTAL_WHOLEBODY_KEYPOINTS, 3), dtype=np.float64),
                    wholebody_world_133=np.zeros((TOTAL_WHOLEBODY_KEYPOINTS, 3), dtype=np.float64),
                    scores_133=np.zeros((TOTAL_WHOLEBODY_KEYPOINTS,), dtype=np.float64),
                    skeleton_count=0,
                    valid_bboxes=[],
                    visibility_ok=False,
                    backend_used=err_backend,
                )
                self.latest_frame = empty_frame
                return empty_frame

            raw_xyz, scores = generate_synthetic_133_pose(
                t_sec=timestamp_sec,
                exercise_id=exercise_id,
                is_calibrating=is_calibrating,
                mock_pose_override=mock_pose_override,
            )
            if mock_pose_override == "multi_person":
                valid_bboxes = [[240, 80, 580, 680], [720, 100, 1040, 670]]
            else:
                valid_bboxes = [self._bbox_from_norm_kpts(raw_xyz, scores)]

        filtered_xyz = self.filter_3d.filter(raw_xyz, timestamp_sec, visibility=scores)
        filtered_world_133 = self.world_filter_3d.filter(
            wholebody133_to_world_meters(filtered_xyz),
            timestamp_sec,
            visibility=scores,
        )
        mp_2d_33, vis_33 = wholebody133_to_mediapipe33(filtered_xyz, scores)
        mp_3d_33, _ = wholebody133_to_mediapipe33(filtered_world_133, scores)
        core_body_idx = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        min_core_vis = float(np.min(scores[core_body_idx])) if scores.shape[0] >= 17 else 1.0

        skeleton_count = max(1, len(valid_bboxes))
        backend_used = (
            self.mp_engine.backend_label
            if (self.mp_engine is not None and not is_mock_mode and mock_pose_override is None and self._model is None)
            else self._backend_name
        )
        frame_res = PoseEstimationFrame(
            pose_landmarks_2d=mp_2d_33,
            pose_world_landmarks_3d=mp_3d_33,
            visibilities_33=vis_33,
            wholebody_2d_133=filtered_xyz,
            wholebody_world_133=filtered_world_133,
            scores_133=scores,
            skeleton_count=skeleton_count,
            valid_bboxes=valid_bboxes,
            visibility_ok=min_core_vis >= float(self.settings.MIN_LANDMARK_VISIBILITY),
            backend_used=backend_used,
        )
        self.latest_frame = frame_res
        return frame_res

    def estimate_with_crowd(
        self,
        frame_bgr: Optional[np.ndarray],
        timestamp_sec: float,
        exercise_id: str = "STROKE_SH_ABD_R",
        is_calibrating: bool = False,
        mock_pose_override: Optional[str] = None,
        is_mock_mode: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, int, List[List[int]]]:
        """
        Estimate smoothed 133 `(X, Y, Z)` keypoints for the primary subject PLUS valid human count
        (filtering reflections/posters via confidence >= 0.60 and torso span >= 25% of frame height).
        Returns `(filtered_xyz (133,3), scores (133,), skeleton_count, skeleton_bboxes_1280x720)`.
        """
        res = self.estimate_clinical_with_crowd(
            frame_bgr=frame_bgr,
            timestamp_sec=timestamp_sec,
            exercise_id=exercise_id,
            is_calibrating=is_calibrating,
            mock_pose_override=mock_pose_override,
            is_mock_mode=is_mock_mode,
        )
        return res.wholebody_2d_133, res.scores_133, res.skeleton_count, res.valid_bboxes

    def estimate(
        self,
        frame_bgr: Optional[np.ndarray],
        timestamp_sec: float,
        exercise_id: str = "STROKE_SH_ABD_R",
        is_calibrating: bool = False,
        mock_pose_override: Optional[str] = None,
        is_mock_mode: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate smoothed 133 `(X, Y, Z)` normalized keypoints and confidence scores.
        Returns `(filtered_xyz (133,3), scores (133,))`.
        """
        filtered_xyz, scores, _, _ = self.estimate_with_crowd(
            frame_bgr=frame_bgr,
            timestamp_sec=timestamp_sec,
            exercise_id=exercise_id,
            is_calibrating=is_calibrating,
            mock_pose_override=mock_pose_override,
            is_mock_mode=is_mock_mode,
        )
        return filtered_xyz, scores

    @staticmethod
    def to_keypoint_schemas(xyz: np.ndarray, scores: np.ndarray) -> List[Keypoint3D]:
        return [
            Keypoint3D(
                index=idx,
                name=keypoint_name(idx),
                x=round(float(xyz[idx, 0]), 5),
                y=round(float(xyz[idx, 1]), 5),
                z=round(float(xyz[idx, 2]), 5),
                score=round(float(scores[idx]), 4),
            )
            for idx in range(TOTAL_WHOLEBODY_KEYPOINTS)
        ]

    @staticmethod
    def to_world_landmark_schemas(world_33: np.ndarray, vis_33: np.ndarray) -> List[Keypoint3D]:
        """Serialize 33 true metric `pose_world_landmarks` (meters, hip-centered)."""
        cnt = min(TOTAL_MEDIAPIPE_KEYPOINTS, world_33.shape[0])
        return [
            Keypoint3D(
                index=idx,
                name=f"mp_{idx}",
                x=round(float(world_33[idx, 0]), 5),
                y=round(float(world_33[idx, 1]), 5),
                z=round(float(world_33[idx, 2]), 5),
                score=round(float(vis_33[idx]), 4) if idx < vis_33.shape[0] else 1.0,
            )
            for idx in range(cnt)
        ]

    @staticmethod
    def extract_hand_raise_telemetry(xyz: np.ndarray, scores: np.ndarray) -> HandRaiseTelemetry:
        """
        Compute real-time hand raise & mirrored screen zone selection for PhysicalCardGame
        and kiosk greeting using 133-point wrist/finger coordinates (indices 91..132).
        """
        l_sh_y = float(xyz[5, 1]) if scores[5] > 0.2 else 0.35
        r_sh_y = float(xyz[6, 1]) if scores[6] > 0.2 else 0.35

        # Left hand centroid (91..111)
        l_hand_pts = xyz[91:112]
        l_hand_sc = scores[91:112]
        l_valid = bool(float(scores[9]) > 0.2 or float(np.mean(l_hand_sc)) > 0.25)
        l_palm = np.mean(l_hand_pts[[0, 5, 9, 13, 17]], axis=0) if np.mean(l_hand_sc) > 0.25 else xyz[9]
        l_raised = bool(l_valid and float(l_palm[1]) < l_sh_y + 0.08)

        # Right hand centroid (112..132)
        r_hand_pts = xyz[112:133]
        r_hand_sc = scores[112:133]
        r_valid = bool(float(scores[10]) > 0.2 or float(np.mean(r_hand_sc)) > 0.25)
        r_palm = np.mean(r_hand_pts[[0, 5, 9, 13, 17]], axis=0) if np.mean(r_hand_sc) > 0.25 else xyz[10]
        r_raised = bool(r_valid and float(r_palm[1]) < r_sh_y + 0.08)

        zone = "NONE"
        if l_raised and r_raised:
            zone = "BOTH"
        elif r_raised:
            # Mirror check: user's right hand maps to screen right side (screenX >= 0.5)
            mirrored_x = 1.0 - float(r_palm[0])
            zone = "RIGHT" if mirrored_x >= 0.45 else "LEFT"
        elif l_raised:
            mirrored_x = 1.0 - float(l_palm[0])
            zone = "LEFT" if mirrored_x <= 0.55 else "RIGHT"

        left_hand_kpts = [
            Keypoint3D(
                index=91 + i,
                name=f"left_hand_{i}",
                x=round(float(xyz[91 + i, 0]), 5),
                y=round(float(xyz[91 + i, 1]), 5),
                z=round(float(xyz[91 + i, 2]), 5),
                score=round(float(scores[91 + i]), 4),
            )
            for i in range(21)
        ]
        right_hand_kpts = [
            Keypoint3D(
                index=112 + i,
                name=f"right_hand_{i}",
                x=round(float(xyz[112 + i, 0]), 5),
                y=round(float(xyz[112 + i, 1]), 5),
                z=round(float(xyz[112 + i, 2]), 5),
                score=round(float(scores[112 + i]), 4),
            )
            for i in range(21)
        ]

        return HandRaiseTelemetry(
            left_raised=l_raised,
            right_raised=r_raised,
            active_card_zone=zone,  # type: ignore[arg-type]
            left_palm_2d=(round(float(l_palm[0]), 4), round(float(l_palm[1]), 4)),
            right_palm_2d=(round(float(r_palm[0]), 4), round(float(r_palm[1]), 4)),
            left_hand_landmarks=left_hand_kpts,
            right_hand_landmarks=right_hand_kpts,
        )


ClinicalPoseEstimator = RTMW3DPoseEstimator
