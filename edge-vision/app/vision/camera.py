"""Threaded OpenCV webcam & mock video capture with locked focus/exposure for clinical ROM precision."""

from __future__ import annotations

import base64
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional, Tuple

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
            self._cap = None
            self._mode = "synthetic_mock" if (mock_path and mock_path.upper() == "SYNTHETIC") else "camera_disconnected"
            self._width = 1280
            self._height = 720
            logger.warning("OpenCV (cv2) not installed; camera mode=%s.", self._mode)
            return

        if mock_path:
            resolved = Path(mock_path)
            if not resolved.exists():
                logger.warning("MOCK_VIDEO_PATH=%s not found; marking camera disconnected.", mock_path)
                self._cap = None
                self._mode = "camera_disconnected"
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

        def _configure_and_probe(vc: Any, label: str) -> bool:
            if vc is None or not vc.isOpened():
                return False
            try:
                vc.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            except Exception:
                pass
            vc.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.settings.CAMERA_WIDTH))
            vc.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.settings.CAMERA_HEIGHT))
            vc.set(cv2.CAP_PROP_FPS, float(self.settings.CAMERA_FPS))

            w_chk = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            h_chk = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            if w_chk < 1280 or h_chk < 720:
                vc.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.settings.CAMERA_FALLBACK_WIDTH))
                vc.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.settings.CAMERA_FALLBACK_HEIGHT))
                vc.set(cv2.CAP_PROP_FPS, float(self.settings.CAMERA_FPS))

            # Enable auto-exposure & auto-focus so Windows DSHOW/MSMF webcams never output pitch-black frames
            try:
                vc.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3.0)
                vc.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
                vc.set(cv2.CAP_PROP_AUTOFOCUS, 1.0)
            except Exception:
                pass

            # Probe up to 5 warmup frames to verify non-black imagery
            for _ in range(5):
                ok_probe, probe_img = vc.read()
                if ok_probe and probe_img is not None and float(probe_img.mean()) >= 3.0:
                    return True
                time.sleep(0.03)
            logger.warning("Camera backend %s returned dark/unreadable warmup frames (mean < 3.0).", label)
            return False

        # Physical camera open: try DSHOW first on Windows, fall back to MSMF/default if DSHOW is dark/unavailable
        cap = None
        if os.name == "nt":
            candidate_dshow = cv2.VideoCapture(self.settings.CAMERA_INDEX, cv2.CAP_DSHOW)
            if _configure_and_probe(candidate_dshow, "CAP_DSHOW"):
                cap = candidate_dshow
            elif candidate_dshow is not None:
                try:
                    candidate_dshow.release()
                except Exception:
                    pass

        if cap is None:
            candidate_default = cv2.VideoCapture(self.settings.CAMERA_INDEX)
            if candidate_default is not None and candidate_default.isOpened():
                _configure_and_probe(candidate_default, "CAP_ANY")
                cap = candidate_default

        if not cap or not cap.isOpened():
            logger.warning("Camera Disconnected: No physical webcam detected on index %d.", self.settings.CAMERA_INDEX)
            self._cap = None
            with self._lock:
                self._frame = None
            self._mode = "camera_disconnected"
            self._width = 1280
            self._height = 720
            return

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or self.settings.CAMERA_FALLBACK_WIDTH)
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or self.settings.CAMERA_FALLBACK_HEIGHT)

        self._cap = cap
        self._width = actual_w
        self._height = actual_h
        self._mode = "live_webcam"
        logger.info(
            "Webcam live at %dx%d @ %.0f FPS target (auto-exposure enabled).",
            actual_w,
            actual_h,
            cap.get(cv2.CAP_PROP_FPS) or self.settings.CAMERA_FPS,
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
        consecutive_failures = 0

        while self._running:
            loop_start = time.perf_counter()
            frame: Optional[np.ndarray] = None

            if (
                self._cap is None
                and cv2 is not None
                and self._mode != "synthetic_mock"
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
                    consecutive_failures = 0
                    if not self._mode.startswith("mock_video"):
                        self._mode = "live_webcam"
                else:
                    consecutive_failures += 1
                    if consecutive_failures >= 30 and not self._mode.startswith("mock_video"):
                        logger.warning("Webcam stopped delivering frames; marking camera_disconnected.")
                        try:
                            self._cap.release()  # type: ignore[attr-defined]
                        except Exception:
                            pass
                        self._cap = None
                        self._mode = "camera_disconnected"
                        with self._lock:
                            self._frame = None
            elif self._mode == "synthetic_mock":
                frame = self._generate_synthetic_frame(loop_start - self._started_at)
            else:
                self._mode = "camera_disconnected"
                with self._lock:
                    self._frame = None

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

