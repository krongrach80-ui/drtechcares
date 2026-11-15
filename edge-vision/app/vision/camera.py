"""Threaded OpenCV webcam & mock video capture with locked focus/exposure for clinical ROM precision."""

from __future__ import annotations

import base64
import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from app.config import EdgeVisionSettings

logger = logging.getLogger("edge_vision.camera")

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore


class ThreadedCameraCapture:
    """
    Clinical camera capture thread:
    - Opens 1920x1080 (fallback 1280x720) at 60 FPS (CAP_PROP_FPS = 60).
    - Disables autofocus (CAP_PROP_AUTOFOCUS = 0) & locks fixed 2.0m focus so apparent body scale never drifts.
    - Supports MOCK_VIDEO_PATH (video file path or "SYNTHETIC") when testing without a physical webcam.
    """

    def __init__(self, settings: EdgeVisionSettings) -> None:
        self.settings = settings
        self._cap: Optional[object] = None
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._frame: Optional[np.ndarray] = None
        self._frame_id: int = 0
        self._fps: float = 0.0
        self._started_at: float = time.perf_counter()
        self._mode: str = "uninitialized"
        self._width: int = settings.CAMERA_WIDTH
        self._height: int = settings.CAMERA_HEIGHT
        self._last_reopen_attempt: float = 0.0

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def fps(self) -> float:
        return round(self._fps, 1)

    @property
    def resolution(self) -> Tuple[int, int]:
        return (self._width, self._height)

    def start(self) -> None:
        if self._running:
            return
        self._started_at = time.perf_counter()
        self._open_source()
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, name="edge-camera-thread", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.5)
        self._thread = None
        if self._cap is not None:
            try:
                self._cap.release()  # type: ignore[attr-defined]
            except Exception:
                pass
            self._cap = None

    def _open_source(self) -> None:
        mock_path = self.settings.MOCK_VIDEO_PATH
        if mock_path and mock_path.upper() == "SYNTHETIC":
            self._mode = "synthetic_mock"
            self._width = 1280
            self._height = 720
            logger.info("Camera running in SYNTHETIC mock mode (MOCK_VIDEO_PATH=SYNTHETIC).")
            return

        if cv2 is None:
            self._mode = "synthetic_mock"
            self._width = 1280
            self._height = 720
            logger.warning("OpenCV (cv2) not installed; falling back to SYNTHETIC mock frames.")
            return

        if mock_path:
            resolved = Path(mock_path)
            if not resolved.exists():
                logger.warning("MOCK_VIDEO_PATH=%s not found; using SYNTHETIC mock frames.", mock_path)
                self._mode = "synthetic_mock"
                self._width = 1280
                self._height = 720
                return
            cap = cv2.VideoCapture(str(resolved))
            if cap.isOpened():
                self._cap = cap
                self._mode = f"mock_video:{resolved.name}"
                self._width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
                self._height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
                logger.info("Opened mock video file %s (%dx%d).", resolved, self._width, self._height)
                return

        # Physical camera open (prefer DirectShow on Windows for fast 60fps & manual focus control)
        cap = None
        if os.name == "nt":
            cap = cv2.VideoCapture(self.settings.CAMERA_INDEX, cv2.CAP_DSHOW)
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(self.settings.CAMERA_INDEX)

        if not cap or not cap.isOpened():
            logger.warning("No physical webcam detected on index %d; switching to SYNTHETIC mock mode.", self.settings.CAMERA_INDEX)
            self._cap = None
            self._mode = "synthetic_mock"
            self._width = 1280
            self._height = 720
            return

        # Enforce MJPG for high-bandwidth 1080p60 over USB
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.settings.CAMERA_WIDTH))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.settings.CAMERA_HEIGHT))
        cap.set(cv2.CAP_PROP_FPS, float(self.settings.CAMERA_FPS))

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if actual_w < 1280 or actual_h < 720:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.settings.CAMERA_FALLBACK_WIDTH))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.settings.CAMERA_FALLBACK_HEIGHT))
            cap.set(cv2.CAP_PROP_FPS, float(self.settings.CAMERA_FPS))
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or self.settings.CAMERA_FALLBACK_WIDTH)
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or self.settings.CAMERA_FALLBACK_HEIGHT)

        # Clinical optical lock: Disable Auto-Focus & Auto-Exposure to eliminate ROM focal distortion
        cap.set(cv2.CAP_PROP_AUTOFOCUS, float(self.settings.AUTOFOCUS_OFF))
        cap.set(cv2.CAP_PROP_FOCUS, float(self.settings.FIXED_FOCUS_2M))
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, float(self.settings.AUTO_EXPOSURE_MANUAL))
        cap.set(cv2.CAP_PROP_EXPOSURE, float(self.settings.FIXED_EXPOSURE))

        self._cap = cap
        self._width = actual_w
        self._height = actual_h
        self._mode = "live_webcam"
        logger.info(
            "Webcam locked at %dx%d @ %.0f FPS target (autofocus=%d, fixed_focus=%d).",
            actual_w,
            actual_h,
            cap.get(cv2.CAP_PROP_FPS) or self.settings.CAMERA_FPS,
            self.settings.AUTOFOCUS_OFF,
            self.settings.FIXED_FOCUS_2M,
        )

    def _generate_synthetic_frame(self, t_sec: float) -> np.ndarray:
        """Render clean synthetic clinical test frame for offline verification without a webcam."""
        h, w = 720, 1280
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :, 0] = 34
        frame[:, :, 1] = 28
        frame[:, :, 2] = 16
        if cv2 is not None:
            cx, cy = w // 2, 190
            cv2.circle(frame, (cx, cy), 42, (180, 225, 210), 2)
            cv2.line(frame, (cx - 80, 270), (cx + 80, 270), (160, 240, 190), 3)
            cv2.line(frame, (cx, 270), (cx, 450), (160, 240, 190), 3)
            angle = np.sin(t_sec * 1.4) * 0.5 + 0.5
            arm_x = int(cx - 80 - 110 * np.sin(angle * np.pi * 0.8))
            arm_y = int(270 + 110 * np.cos(angle * np.pi * 0.8))
            cv2.line(frame, (cx - 80, 270), (arm_x, arm_y), (80, 220, 130), 4)
            cv2.putText(
                frame,
                f"DRTECHCARES MOCK CAMERA ({t_sec:.1f}s)",
                (32, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (140, 240, 210),
                2,
            )
        return frame

    def _capture_loop(self) -> None:
        target_interval = 1.0 / max(15.0, float(self.settings.CAMERA_FPS))
        fps_window_start = time.perf_counter()
        fps_counter = 0

        while self._running:
            loop_start = time.perf_counter()
            frame: Optional[np.ndarray] = None

            if (
                self._cap is None
                and cv2 is not None
                and not self.settings.MOCK_VIDEO_PATH
                and (loop_start - self._last_reopen_attempt) >= 2.5
            ):
                self._last_reopen_attempt = loop_start
                self._open_source()

            if self._cap is not None and cv2 is not None:
                ok, grabbed = self._cap.read()  # type: ignore[attr-defined]
                if not ok and self._mode.startswith("mock_video"):
                    # Loop mock video cleanly
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # type: ignore[attr-defined]
                    ok, grabbed = self._cap.read()  # type: ignore[attr-defined]
                if ok and grabbed is not None:
                    frame = grabbed
            else:
                frame = self._generate_synthetic_frame(loop_start - self._started_at)

            if frame is not None:
                with self._lock:
                    self._frame = frame
                    self._frame_id += 1
                fps_counter += 1

            now = time.perf_counter()
            window_elapsed = now - fps_window_start
            if window_elapsed >= 0.5:
                self._fps = fps_counter / window_elapsed
                fps_counter = 0
                fps_window_start = now

            elapsed = time.perf_counter() - loop_start
            sleep_for = target_interval - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

    def read_latest(self) -> Tuple[int, Optional[np.ndarray]]:
        with self._lock:
            if self._frame is None:
                return (self._frame_id, None)
            return (self._frame_id, self._frame.copy())

    def encode_jpeg_bytes(self, frame: np.ndarray, max_width: int = 640, quality: int = 70) -> Optional[bytes]:
        if cv2 is None or frame is None:
            return None
        try:
            h, w = frame.shape[:2]
            resized = frame
            if w > max_width:
                scale = max_width / float(w)
                resized = cv2.resize(frame, (max_width, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            if not ok:
                return None
            return buf.tobytes()
        except Exception:
            return None

    def encode_jpeg_base64(self, frame: np.ndarray, max_width: int = 640, quality: int = 70) -> Optional[str]:
        jpeg_bytes = self.encode_jpeg_bytes(frame, max_width=max_width, quality=quality)
        if not jpeg_bytes:
            return None
        return base64.b64encode(jpeg_bytes).decode("ascii")

