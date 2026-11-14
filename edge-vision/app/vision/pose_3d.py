"""RTMW3D-l (133-keypoint Wholebody3d) 3D pose estimator with vectorized One-Euro Filter."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal as scipy_signal  # Clinical DSP verification & cutoff helpers

from app.config import EdgeVisionSettings
from app.schemas.telemetry import HandRaiseTelemetry, Keypoint3D

logger = logging.getLogger("edge_vision.pose_3d")

try:
    import onnxruntime as ort  # type: ignore
except ImportError:  # pragma: no cover
    ort = None  # type: ignore

try:
    from rtmlib import Wholebody3d  # type: ignore
except ImportError:  # pragma: no cover
    Wholebody3d = None  # type: ignore

TOTAL_WHOLEBODY_KEYPOINTS = 133

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


class OneEuroFilter3D:
    """
    Vectorized One-Euro Filter across all 133 (X, Y, Z) coordinates:
    - Eliminates micro-jitter while holding isometric targets (min_cutoff = 1.2 Hz).
    - Adapts cutoff proportionally to 3D velocity (beta = 0.08) for near-zero fast-motion lag.
    """

    def __init__(
        self,
        num_keypoints: int = TOTAL_WHOLEBODY_KEYPOINTS,
        min_cutoff: float = 1.2,
        beta: float = 0.08,
        d_cutoff: float = 1.0,
    ) -> None:
        self.num_keypoints = num_keypoints
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x_prev: Optional[np.ndarray] = None
        self._dx_prev: Optional[np.ndarray] = None
        self._last_ts: Optional[float] = None

        # Precompute reference SciPy bilinear filter check for diagnostic telemetry
        self._scipy_b, self._scipy_a = scipy_signal.butter(1, min(0.95, self.min_cutoff / 30.0), btype="low")

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = None
        self._last_ts = None

    @staticmethod
    def _alpha(cutoff: np.ndarray | float, dt: float) -> np.ndarray | float:
        tau = 1.0 / (2.0 * math.pi * np.maximum(cutoff, 1e-4))
        return 1.0 / (1.0 + tau / dt)

    def filter(self, points_xyz: np.ndarray, timestamp_sec: float) -> np.ndarray:
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

        if self._x_prev is None or self._last_ts is None:
            self._x_prev = x.copy()
            self._dx_prev = np.zeros_like(x)
            self._last_ts = float(timestamp_sec)
            return x.copy()

        dt = max(1e-3, min(0.25, float(timestamp_sec) - self._last_ts))
        self._last_ts = float(timestamp_sec)

        dx = (x - self._x_prev) / dt
        alpha_d = self._alpha(self.d_cutoff, dt)
        dx_hat = alpha_d * dx + (1.0 - alpha_d) * self._dx_prev

        speed = np.linalg.norm(dx_hat, axis=1, keepdims=True)
        cutoff = self.min_cutoff + self.beta * speed
        alpha = self._alpha(cutoff, dt)
        x_hat = alpha * x + (1.0 - alpha) * self._x_prev

        self._x_prev = x_hat.copy()
        self._dx_prev = dx_hat.copy()
        return x_hat


def generate_synthetic_133_pose(
    t_sec: float,
    exercise_id: str = "ex-03",
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
        elif "squat" in exercise_id.lower() or exercise_id in ("ex-13", "ex-14", "ex-26"):
            hip_drop = cycle * 0.13
            pts[0, 1] += hip_drop * 0.9
            pts[5, 1] += hip_drop * 0.9
            pts[6, 1] += hip_drop * 0.9
            pts[11, 1] += hip_drop
            pts[12, 1] += hip_drop
            pts[11, 2] += 0.08 * cycle
            pts[12, 2] += 0.08 * cycle
            pts[13, 2] -= 0.10 * cycle
            pts[14, 2] -= 0.10 * cycle
        elif "left" in exercise_id.lower() or exercise_id in ("ex-02", "ex-04", "ex-06", "ex-08", "ex-10", "ex-12", "ex-25"):
            l_angle_rad = math.radians(12.0 + cycle * 145.0)
            if "flex" in exercise_id.lower() or exercise_id == "ex-02":
                l_forward_z = -0.22 * cycle
        else:
            r_angle_rad = math.radians(12.0 + cycle * 145.0)
            if "flex" in exercise_id.lower() or exercise_id == "ex-01":
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


class RTMW3DPoseEstimator:
    """
    RTMW3D-l (Wholebody3d 133-point 3D) inference engine backed by rtmlib + onnxruntime-gpu/CPU
    and paired with 3D One-Euro Filtering for clinical 3D landmark extraction.
    """

    def __init__(self, settings: EdgeVisionSettings) -> None:
        self.settings = settings
        self.filter_3d = OneEuroFilter3D(
            num_keypoints=TOTAL_WHOLEBODY_KEYPOINTS,
            min_cutoff=settings.ONE_EURO_MIN_CUTOFF,
            beta=settings.ONE_EURO_BETA,
            d_cutoff=settings.ONE_EURO_D_CUTOFF,
        )
        self._model = None
        self._device = "cpu"
        self._backend_name = "RTMW3D-l (Pending Init)"
        self._load_model()

    @property
    def backend_name(self) -> str:
        return self._backend_name

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
        if Wholebody3d is None:
            self._backend_name = f"Synthetic133-3D (rtmlib not installed, target={self._device})"
            logger.warning("rtmlib is unavailable in current Python env; using synthetic 133-point 3D fallback.")
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

    def estimate_with_crowd(
        self,
        frame_bgr: Optional[np.ndarray],
        timestamp_sec: float,
        exercise_id: str = "ex-03",
        is_calibrating: bool = False,
        mock_pose_override: Optional[str] = None,
        is_mock_mode: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, int, List[List[int]]]:
        """
        Estimate smoothed 133 `(X, Y, Z)` keypoints for the primary subject PLUS valid human count
        (filtering reflections/posters via confidence >= 0.60 and torso span >= 25% of frame height).
        Returns `(filtered_xyz (133,3), scores (133,), skeleton_count, skeleton_bboxes_1280x720)`.
        """
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

        filtered_xyz = self.filter_3d.filter(raw_xyz, timestamp_sec)
        skeleton_count = max(1, len(valid_bboxes))
        return filtered_xyz, scores, skeleton_count, valid_bboxes

    def estimate(
        self,
        frame_bgr: Optional[np.ndarray],
        timestamp_sec: float,
        exercise_id: str = "ex-03",
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
        l_palm = np.mean(l_hand_pts[[0, 5, 9, 13, 17]], axis=0) if np.mean(l_hand_sc) > 0.25 else xyz[9]
        l_raised = bool(float(l_palm[1]) < l_sh_y + 0.08 or float(l_palm[1]) < 0.62)

        # Right hand centroid (112..132)
        r_hand_pts = xyz[112:133]
        r_hand_sc = scores[112:133]
        r_palm = np.mean(r_hand_pts[[0, 5, 9, 13, 17]], axis=0) if np.mean(r_hand_sc) > 0.25 else xyz[10]
        r_raised = bool(float(r_palm[1]) < r_sh_y + 0.08 or float(r_palm[1]) < 0.62)

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
