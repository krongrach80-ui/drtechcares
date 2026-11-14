"""InsightFace (buffalo_l / ArcFace 512-D) 120-Frame 5-Angle Enrollment & <100ms 1:N Matcher."""

from __future__ import annotations

import base64
import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.config import EdgeVisionSettings
from app.schemas.telemetry import (
    FaceEnrollmentStateSchema,
    FaceID120Response,
    IdentityDetectedBoxSchema,
    IdentityGuardSchema,
    PatientMatchSchema,
)

logger = logging.getLogger("edge_vision.face_120")

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore

try:
    import onnxruntime as ort  # type: ignore
except ImportError:  # pragma: no cover
    ort = None  # type: ignore

try:
    from insightface.app import FaceAnalysis  # type: ignore
except ImportError:  # pragma: no cover
    FaceAnalysis = None  # type: ignore

try:
    from supabase import create_client  # type: ignore
except ImportError:  # pragma: no cover
    create_client = None  # type: ignore


BUCKET_ORDER: Tuple[str, ...] = ("center", "left", "right", "up", "down")


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-12:
        out = np.zeros_like(arr, dtype=np.float64)
        if out.size > 0:
            out[0] = 1.0
        return out
    return arr / norm


def classify_face_bucket(yaw_deg: float, pitch_deg: float) -> str:
    if abs(yaw_deg) >= abs(pitch_deg) * 1.15:
        if yaw_deg <= -8.0:
            return "left"
        if yaw_deg >= 8.0:
            return "right"
    if pitch_deg <= -6.0:
        return "up"
    if pitch_deg >= 6.0:
        return "down"
    if yaw_deg <= -8.0:
        return "left"
    if yaw_deg >= 8.0:
        return "right"
    return "center"


@dataclass
class CachedPatientBiometric:
    patient_id: str
    hn: str
    full_name: str
    master_embedding: np.ndarray
    angle_embeddings: Dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class FaceEnrollmentSession:
    session_id: str
    patient_id: Optional[str] = None
    hn: Optional[str] = None
    full_name: Optional[str] = None
    bucket_targets: Dict[str, int] = field(
        default_factory=lambda: {"center": 40, "left": 20, "right": 20, "up": 20, "down": 20}
    )
    bucket_samples: Dict[str, List[np.ndarray]] = field(
        default_factory=lambda: {"center": [], "left": [], "right": [], "up": [], "down": []}
    )
    sharpness_scores: List[float] = field(default_factory=list)
    best_frontal_avatar: str = ""
    best_frontal_score: float = -1.0
    last_capture_at: float = 0.0


class InsightFace120Manager:
    """
    Clinical InsightFace (buffalo_l / ArcFace 512-D) Engine:
    - 120-frame multi-angle enrollment (center: 40, left: 20, right: 20, up: 20, down: 20).
    - Syncs `public.patient_biometrics` & `public.patients` in Supabase Cloud.
    - Performs vectorized 1:N patient cosine matching in < 100ms in RAM.
    """

    def __init__(self, settings: EdgeVisionSettings) -> None:
        self.settings = settings
        self._face_app: Any = None
        self._supabase: Any = None
        self._patients: List[CachedPatientBiometric] = []
        self._matrix: Optional[np.ndarray] = None
        self._enroll_sessions: Dict[str, FaceEnrollmentSession] = {}
        self._active_session_id: Optional[str] = None
        self._backend_name = "InsightFace (Uninitialized)"

        # Continuous Anti-Cheat & Identity Guard state
        self.active_patient: Optional[CachedPatientBiometric] = None
        self._guard_frame_counter: int = 0
        self._consecutive_impostor_misses: int = 0
        self._consecutive_unlock_matches: int = 0
        self._locked_guard_status: str = "VERIFIED"
        self._last_verified_sim: float = 0.96
        self._impostor_logged_for_episode: bool = False

        self._init_insightface()
        self._init_supabase()

    @property
    def backend_name(self) -> str:
        return self._backend_name

    @property
    def candidate_count(self) -> int:
        return len(self._patients)

    @property
    def active_session_id(self) -> Optional[str]:
        return self._active_session_id

    def _init_insightface(self) -> None:
        if FaceAnalysis is None:
            self._backend_name = "ArcFace-512 Fallback (insightface not installed)"
            logger.warning("insightface package not installed; using deterministic image embedding fallback.")
            return

        providers = ["CPUExecutionProvider"]
        if self.settings.USE_GPU_IF_AVAILABLE and ort is not None:
            try:
                avail = ort.get_available_providers()
                if "CUDAExecutionProvider" in avail:
                    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            except Exception:
                pass

        for prov_list in ([providers[0], "CPUExecutionProvider"] if providers[0] != "CPUExecutionProvider" else [["CPUExecutionProvider"]]):
            try:
                app = FaceAnalysis(name=self.settings.INSIGHTFACE_MODEL_NAME, providers=prov_list)
                ctx_id = 0 if "CUDAExecutionProvider" in prov_list else -1
                app.prepare(ctx_id=ctx_id, det_size=(640, 640), det_thresh=0.5)
                self._face_app = app
                self._backend_name = f"InsightFace {self.settings.INSIGHTFACE_MODEL_NAME} ArcFace 512-D ({prov_list[0]})"
                logger.info("Initialized %s", self._backend_name)
                return
            except Exception as exc:
                logger.warning("InsightFace init warning (%s): %s", prov_list, exc)

        self._face_app = None
        self._backend_name = "ArcFace-512 Fallback"

    def _init_supabase(self) -> None:
        key = self.settings.SUPABASE_SERVICE_ROLE_KEY or self.settings.SUPABASE_ANON_KEY
        if create_client is None or not self.settings.SUPABASE_URL or not key:
            logger.info("Supabase client disabled or credentials missing; Face 1:N cache runs locally.")
            return
        try:
            self._supabase = create_client(self.settings.SUPABASE_URL, key)
            logger.info("Connected to Supabase Cloud (%s).", self.settings.SUPABASE_URL)
        except Exception as exc:
            logger.warning("Failed connecting to Supabase Cloud: %s", exc)
            self._supabase = None

    def sync_from_supabase(self) -> int:
        """Load enrolled 512-D embeddings from `patient_biometrics` + `patients` into memory."""
        if self._supabase is None:
            return len(self._patients)
        try:
            patients_res = (
                self._supabase.table("patients")
                .select("id, hn, full_name, registration_status")
                .eq("registration_status", "active")
                .execute()
            )
            patient_map: Dict[str, Dict[str, Any]] = {
                str(row["id"]): row for row in (patients_res.data or []) if row.get("id")
            }

            bio_res = self._supabase.table("patient_biometrics").select("*").execute()
            loaded: List[CachedPatientBiometric] = []
            for row in bio_res.data or []:
                pid = str(row.get("patient_id") or "")
                master = row.get("master_embedding")
                if not pid or not isinstance(master, list) or len(master) != 512:
                    continue
                p_info = patient_map.get(pid, {})
                raw_angles = row.get("angle_embeddings") or {}
                angle_vecs: Dict[str, np.ndarray] = {}
                for bucket in BUCKET_ORDER:
                    vec = raw_angles.get(bucket)
                    if isinstance(vec, list) and len(vec) == 512:
                        angle_vecs[bucket] = l2_normalize(np.asarray(vec, dtype=np.float64))
                loaded.append(
                    CachedPatientBiometric(
                        patient_id=pid,
                        hn=str(row.get("hn") or p_info.get("hn") or ""),
                        full_name=str(p_info.get("full_name") or f"Patient {row.get('hn', pid)}"),
                        master_embedding=l2_normalize(np.asarray(master, dtype=np.float64)),
                        angle_embeddings=angle_vecs,
                    )
                )

            self._patients = loaded
            self._matrix = (
                np.vstack([p.master_embedding for p in loaded]) if loaded else None
            )
            logger.info("Synchronized %d enrolled patients into 1:N ArcFace RAM index.", len(loaded))
            return len(loaded)
        except Exception as exc:
            logger.warning("Supabase biometric sync warning: %s", exc)
            return len(self._patients)

    def decode_base64_frame(self, image_base64: Optional[str]) -> Optional[np.ndarray]:
        if not image_base64 or cv2 is None:
            return None
        payload = image_base64.split(",", 1)[1] if "," in image_base64 else image_base64
        try:
            raw = base64.b64decode(payload)
            arr = np.frombuffer(raw, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return frame
        except Exception:
            return None

    def _compute_sharpness(self, frame_bgr: np.ndarray) -> float:
        if cv2 is not None:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            return float(cv2.Laplacian(gray, cv2.CV_64F).var())
        gray = np.mean(frame_bgr.astype(np.float64), axis=2)
        return float(np.var(np.diff(gray, axis=1)) + np.var(np.diff(gray, axis=0)))

    def _estimate_pose_from_kps(self, kps: np.ndarray) -> Tuple[float, float, float]:
        """Compute approximate head yaw/pitch/roll degrees from 5 InsightFace facial landmarks."""
        l_eye, r_eye, nose, l_mouth, r_mouth = kps[0], kps[1], kps[2], kps[3], kps[4]
        eye_dist = max(1e-3, float(np.linalg.norm(r_eye - l_eye)))
        eye_mid = 0.5 * (l_eye + r_eye)
        mouth_mid = 0.5 * (l_mouth + r_mouth)
        roll = math.degrees(math.atan2(float(r_eye[1] - l_eye[1]), max(1e-3, float(r_eye[0] - l_eye[0]))))
        yaw = ((float(nose[0] - eye_mid[0])) / (eye_dist * 0.5)) * 32.0
        vertical_span = max(1e-3, float(mouth_mid[1] - eye_mid[1]))
        nose_ratio = (float(nose[1] - eye_mid[1])) / vertical_span
        pitch = (nose_ratio - 0.55) * 48.0
        return (round(yaw, 2), round(pitch, 2), round(roll, 2))

    def _deterministic_fallback_embedding(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Structured grayscale spatial patch projection (512-D normalized) when buffalo_l is unavailable."""
        h, w = frame_bgr.shape[:2]
        cx, cy = w // 2, h // 2
        half_w, half_h = max(32, min(w, h) // 4), max(32, min(w, h) // 4)
        crop = frame_bgr[max(0, cy - half_h) : min(h, cy + half_h), max(0, cx - half_w) : min(w, cx + half_w)]
        gray = np.mean(crop, axis=2, dtype=np.float64)
        # Pool crop onto 16x32 grid = 512 dims
        row_bins = np.array_split(gray, 16, axis=0)
        pooled: List[float] = []
        for rb in row_bins:
            col_bins = np.array_split(rb, 32, axis=1)
            for cb in col_bins:
                pooled.append(float(np.mean(cb)) if cb.size > 0 else 0.0)
        vec = np.asarray(pooled, dtype=np.float64)
        vec = vec - float(np.mean(vec))
        return l2_normalize(vec)

    def extract_face_features(
        self, frame_bgr: np.ndarray
    ) -> Optional[Dict[str, Any]]:
        """
        Detect primary face and return 512-D normalized ArcFace embedding, head pose, sharpness, and avatar data URL.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return None
        sharpness = self._compute_sharpness(frame_bgr)

        if self._face_app is not None:
            faces = self._face_app.get(frame_bgr)
            if not faces:
                return None
            # Largest bounding box = standing kiosk patient closest to camera
            primary = max(
                faces,
                key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])),
            )
            emb = getattr(primary, "normed_embedding", None)
            if emb is None and getattr(primary, "embedding", None) is not None:
                emb = l2_normalize(primary.embedding)
            if emb is None:
                return None
            emb_norm = l2_normalize(np.asarray(emb, dtype=np.float64))

            if getattr(primary, "pose", None) is not None:
                pitch_deg, yaw_deg, roll_deg = [round(float(v), 2) for v in primary.pose[:3]]
            elif getattr(primary, "kps", None) is not None:
                yaw_deg, pitch_deg, roll_deg = self._estimate_pose_from_kps(np.asarray(primary.kps))
            else:
                yaw_deg, pitch_deg, roll_deg = (0.0, 0.0, 0.0)

            avatar_url = ""
            if cv2 is not None:
                x1, y1, x2, y2 = [int(v) for v in primary.bbox[:4]]
                pad_x = max(12, int((x2 - x1) * 0.18))
                pad_y = max(12, int((y2 - y1) * 0.22))
                crop = frame_bgr[
                    max(0, y1 - pad_y) : min(frame_bgr.shape[0], y2 + pad_y),
                    max(0, x1 - pad_x) : min(frame_bgr.shape[1], x2 + pad_x),
                ]
                if crop.size > 0:
                    resized = cv2.resize(crop, (320, 320), interpolation=cv2.INTER_AREA)
                    ok, jpg = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
                    if ok:
                        avatar_url = "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode("ascii")

            return {
                "embedding": emb_norm,
                "yaw_deg": yaw_deg,
                "pitch_deg": pitch_deg,
                "roll_deg": roll_deg,
                "sharpness": round(sharpness, 2),
                "avatar_url": avatar_url,
            }

        # Deterministic offline fallback when InsightFace weights are absent (e.g. mock testing)
        return {
            "embedding": self._deterministic_fallback_embedding(frame_bgr),
            "yaw_deg": 0.0,
            "pitch_deg": 0.0,
            "roll_deg": 0.0,
            "sharpness": round(sharpness, 2),
            "avatar_url": "",
        }

    def match_1_to_n(
        self, query_embedding: np.ndarray, yaw_deg: float = 0.0, pitch_deg: float = 0.0
    ) -> PatientMatchSchema:
        """
        Match a 512-D query embedding against all enrolled patients in RAM in < 100ms.
        Combines master centroid cosine similarity + matching head angle similarity.
        """
        start_ns = time.perf_counter_ns()
        if not self._patients or self._matrix is None:
            return PatientMatchSchema(similarity=0.0, latency_ms=0.0)

        q = l2_normalize(query_embedding)
        sims = self._matrix @ q
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        best_patient = self._patients[best_idx]

        # Fuse closest angle centroid when patient turns head naturally
        pose_bucket = classify_face_bucket(yaw_deg, pitch_deg)
        angle_vec = best_patient.angle_embeddings.get(pose_bucket)
        if angle_vec is not None:
            angle_score = float(np.dot(angle_vec, q))
            best_score = max(best_score, 0.45 * best_score + 0.55 * angle_score)

        elapsed_ms = round((time.perf_counter_ns() - start_ns) / 1_000_000.0, 3)
        if best_score < self.settings.FACE_MATCH_THRESHOLD:
            return PatientMatchSchema(similarity=round(best_score, 4), latency_ms=elapsed_ms)

        self.lock_active_patient(
            patient_id=best_patient.patient_id,
            hn=best_patient.hn,
            full_name=best_patient.full_name,
        )

        return PatientMatchSchema(
            id=best_patient.patient_id,
            hn=best_patient.hn,
            name=best_patient.full_name,
            similarity=round(best_score, 4),
            latency_ms=elapsed_ms,
        )

    def reset_identity_guard(self, clear_patient: bool = False) -> None:
        """Reset consecutive impostor/unlock streak counters."""
        self._guard_frame_counter = 0
        self._consecutive_impostor_misses = 0
        self._consecutive_unlock_matches = 0
        self._locked_guard_status = "VERIFIED"
        self._last_verified_sim = 0.96
        self._impostor_logged_for_episode = False
        if clear_patient:
            self.active_patient = None

    def lock_active_patient(
        self,
        patient_id: str,
        hn: Optional[str] = None,
        full_name: Optional[str] = None,
        master_embedding: Optional[List[float]] = None,
        angle_embeddings: Optional[Dict[str, List[float]]] = None,
    ) -> None:
        """Store active patient's 512-D embeddings in RAM for Continuous 1:1 Identity Guard."""
        found = next((p for p in self._patients if p.patient_id == patient_id or (hn and p.hn == hn)), None)
        if found is not None and not master_embedding:
            self.active_patient = found
        elif master_embedding and len(master_embedding) == 512:
            angles: Dict[str, np.ndarray] = {}
            for b, vec in (angle_embeddings or {}).items():
                if isinstance(vec, list) and len(vec) == 512:
                    angles[b] = l2_normalize(np.asarray(vec, dtype=np.float64))
            locked = CachedPatientBiometric(
                patient_id=patient_id,
                hn=hn or (found.hn if found else ""),
                full_name=full_name or (found.full_name if found else patient_id),
                master_embedding=l2_normalize(np.asarray(master_embedding, dtype=np.float64)),
                angle_embeddings=angles or (found.angle_embeddings if found else {}),
            )
            self.active_patient = locked
            self._patients = [p for p in self._patients if p.patient_id != patient_id] + [locked]
            self._matrix = np.vstack([p.master_embedding for p in self._patients])
        elif found is not None:
            self.active_patient = found
        else:
            # Minimal record when DB cache doesn't contain 512-D vector yet
            placeholder = CachedPatientBiometric(
                patient_id=patient_id,
                hn=hn or "",
                full_name=full_name or patient_id,
                master_embedding=l2_normalize(np.ones((512,), dtype=np.float64)),
                angle_embeddings={},
            )
            self.active_patient = placeholder
        self.reset_identity_guard(clear_patient=False)

    def dispatch_impersonation_audit_log(
        self,
        patient_id: Optional[str],
        similarity: float,
        reason: str = "IMPERSONATION_BLOCKED",
    ) -> bool:
        """Write IMPERSONATION_BLOCKED audit log to Supabase `public.audit_logs` once per episode."""
        if self._impostor_logged_for_episode:
            return False
        self._impostor_logged_for_episode = True

        if self._supabase is None:
            logger.warning(
                "IMPERSONATION_BLOCKED logged locally (Supabase unavailable): patient=%s sim=%.3f",
                patient_id,
                similarity,
            )
            return False
        try:
            self._supabase.table("audit_logs").insert(
                {
                    "event_type": "BIOMETRIC_SECURITY",
                    "actor_role": "kiosk_edge_vision",
                    "actor_id": self.settings.KIOSK_ID,
                    "actor_name": self.settings.KIOSK_ID,
                    "action": "IMPERSONATION_BLOCKED",
                    "target": "patient_session",
                    "target_id": patient_id,
                    "details": {
                        "patient_id": patient_id,
                        "similarity": round(similarity, 4),
                        "reason": reason,
                    },
                }
            ).execute()
            logger.warning("Persisted IMPERSONATION_BLOCKED to Supabase audit_logs for patient=%s", patient_id)
            return True
        except Exception as exc:  # pragma: no cover
            logger.error("Failed persisting audit_logs IMPERSONATION_BLOCKED: %s", exc)
            return False

    def _cosine_vs_active(self, query_embedding: np.ndarray, yaw_deg: float, pitch_deg: float) -> float:
        if self.active_patient is None:
            return 1.0
        q = l2_normalize(query_embedding)
        score = float(np.dot(self.active_patient.master_embedding, q))
        bucket = classify_face_bucket(yaw_deg, pitch_deg)
        angle_vec = self.active_patient.angle_embeddings.get(bucket)
        if angle_vec is not None:
            angle_score = float(np.dot(angle_vec, q))
            score = max(score, 0.45 * score + 0.55 * angle_score)
        return score

    @staticmethod
    def _is_face_bound_to_skeleton(
        face_bbox_norm: Tuple[float, float, float, float],
        primary_xyz: np.ndarray,
        primary_scores: np.ndarray,
    ) -> bool:
        """
        Skeleton-to-Face Spatial Binding:
        Verify that the face box overlaps RTMW3D head landmarks (nose=0, left_eye=1, right_eye=2)
        so a patient sitting off to the side cannot let someone else exercise in front of the camera.
        """
        fx1, fy1, fx2, fy2 = face_bbox_norm
        pad_x = max(0.08, (fx2 - fx1) * 0.35)
        pad_y = max(0.08, (fy2 - fy1) * 0.35)
        head_indices = (0, 1, 2)
        checked = 0
        for idx in head_indices:
            if idx < primary_xyz.shape[0] and idx < primary_scores.shape[0] and primary_scores[idx] >= 0.20:
                checked += 1
                hx = float(primary_xyz[idx, 0])
                hy = float(primary_xyz[idx, 1])
                if (fx1 - pad_x) <= hx <= (fx2 + pad_x) and (fy1 - pad_y) <= hy <= (fy2 + pad_y):
                    return True
        return checked == 0

    def _detect_valid_faces(
        self,
        frame_bgr: Optional[np.ndarray],
        primary_xyz: np.ndarray,
        mock_override: Optional[str],
        face_detections_override: Optional[List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        min_det = self.settings.MULTI_PERSON_FACE_MIN_DET_SCORE
        min_px = self.settings.MULTI_PERSON_FACE_MIN_PX

        if face_detections_override is not None:
            valid: List[Dict[str, Any]] = []
            for f in face_detections_override:
                det_sc = float(f.get("det_score", 0.95))
                if det_sc < min_det:
                    continue
                bbox = f.get("bbox_1280") or f.get("bbox") or [460, 80, 620, 240]
                bw = float(bbox[2]) - float(bbox[0])
                bh = float(bbox[3]) - float(bbox[1])
                if bw < min_px or bh < min_px:
                    continue
                bn = f.get("bbox_norm")
                if bn is None:
                    bn = (bbox[0] / 1280.0, bbox[1] / 720.0, bbox[2] / 1280.0, bbox[3] / 720.0)
                emb = f.get("embedding")
                valid.append(
                    {
                        "embedding": l2_normalize(np.asarray(emb, dtype=np.float64)) if emb is not None else None,
                        "det_score": det_sc,
                        "bbox_1280": [int(v) for v in bbox[:4]],
                        "bbox_norm": tuple(float(v) for v in bn[:4]),
                        "yaw_deg": float(f.get("yaw_deg", 0.0)),
                        "pitch_deg": float(f.get("pitch_deg", 0.0)),
                    }
                )
            return valid

        if self._face_app is not None and frame_bgr is not None and frame_bgr.size > 0 and mock_override is None:
            fh, fw = frame_bgr.shape[:2]
            raw_faces = self._face_app.get(frame_bgr)
            valid = []
            for f in raw_faces:
                det_sc = float(getattr(f, "det_score", 1.0))
                x1, y1, x2, y2 = [float(v) for v in f.bbox[:4]]
                if det_sc < min_det or (x2 - x1) < min_px or (y2 - y1) < min_px:
                    continue
                emb = getattr(f, "normed_embedding", None)
                if emb is None and getattr(f, "embedding", None) is not None:
                    emb = l2_normalize(f.embedding)
                if getattr(f, "pose", None) is not None:
                    pitch_deg, yaw_deg = float(f.pose[0]), float(f.pose[1])
                elif getattr(f, "kps", None) is not None:
                    yaw_deg, pitch_deg, _ = self._estimate_pose_from_kps(np.asarray(f.kps))
                else:
                    yaw_deg, pitch_deg = (0.0, 0.0)

                bbox_norm = (x1 / max(1.0, fw), y1 / max(1.0, fh), x2 / max(1.0, fw), y2 / max(1.0, fh))
                bbox_1280 = [
                    int(round(bbox_norm[0] * 1280)),
                    int(round(bbox_norm[1] * 720)),
                    int(round(bbox_norm[2] * 1280)),
                    int(round(bbox_norm[3] * 720)),
                ]
                valid.append(
                    {
                        "embedding": l2_normalize(np.asarray(emb, dtype=np.float64)) if emb is not None else None,
                        "det_score": det_sc,
                        "bbox_1280": bbox_1280,
                        "bbox_norm": bbox_norm,
                        "yaw_deg": yaw_deg,
                        "pitch_deg": pitch_deg,
                    }
                )
            return valid

        # Fallback/synthetic path when buffalo_l isn't loaded or mock override active
        nx = float(primary_xyz[0, 0]) if primary_xyz.shape[0] > 0 else 0.5
        ny = float(primary_xyz[0, 1]) if primary_xyz.shape[0] > 0 else 0.16
        head_norm = (max(0.0, nx - 0.06), max(0.0, ny - 0.08), min(1.0, nx + 0.06), min(1.0, ny + 0.08))
        head_1280 = [
            int(round(head_norm[0] * 1280)),
            int(round(head_norm[1] * 720)),
            int(round(head_norm[2] * 1280)),
            int(round(head_norm[3] * 720)),
        ]
        if mock_override == "multi_person":
            emb_main = (
                self.active_patient.master_embedding.copy()
                if self.active_patient is not None
                else l2_normalize(np.ones((512,), dtype=np.float64))
            )
            return [
                {
                    "embedding": emb_main,
                    "det_score": 0.96,
                    "bbox_1280": [380, 60, 500, 210],
                    "bbox_norm": (0.29, 0.08, 0.39, 0.29),
                    "yaw_deg": 0.0,
                    "pitch_deg": 0.0,
                },
                {
                    "embedding": -emb_main,
                    "det_score": 0.93,
                    "bbox_1280": [810, 80, 930, 220],
                    "bbox_norm": (0.63, 0.11, 0.72, 0.30),
                    "yaw_deg": 0.0,
                    "pitch_deg": 0.0,
                },
            ]
        if mock_override == "impostor":
            base = (
                self.active_patient.master_embedding
                if self.active_patient is not None
                else l2_normalize(np.ones((512,), dtype=np.float64))
            )
            return [
                {
                    "embedding": -base,
                    "det_score": 0.95,
                    "bbox_1280": head_1280,
                    "bbox_norm": head_norm,
                    "yaw_deg": 0.0,
                    "pitch_deg": 0.0,
                }
            ]

        if self.active_patient is not None and frame_bgr is not None and frame_bgr.size > 0:
            emb = self._deterministic_fallback_embedding(frame_bgr)
        elif self.active_patient is not None:
            emb = self.active_patient.master_embedding
        else:
            emb = l2_normalize(np.ones((512,), dtype=np.float64))
        return [
            {
                "embedding": emb,
                "det_score": 0.95,
                "bbox_1280": head_1280,
                "bbox_norm": head_norm,
                "yaw_deg": 0.0,
                "pitch_deg": 0.0,
            }
        ]

    def evaluate_identity_guard(
        self,
        frame_bgr: Optional[np.ndarray],
        primary_xyz: np.ndarray,
        primary_scores: np.ndarray,
        skeleton_count: int = 1,
        skeleton_bboxes: Optional[List[List[int]]] = None,
        mock_override: Optional[str] = None,
        face_detections_override: Optional[List[Dict[str, Any]]] = None,
        force_identity_check: bool = False,
    ) -> IdentityGuardSchema:
        """
        Continuous Anti-Cheat & Identity Guard:
        - Rule 1: MULTI_PERSON_DETECTED (skeleton_count > 1 or face_count > 1) -> lock reps, amber boxes, 3-frame unlock.
        - Rule 2: IMPOSTOR_DETECTED (every 8 frames, spatial skeleton-to-face binding, 5 consecutive < 0.68 -> red box + Supabase audit_logs).
        """
        self._guard_frame_counter += 1
        skel_boxes = skeleton_bboxes or [[320, 80, 640, 680]]
        faces_info = self._detect_valid_faces(frame_bgr, primary_xyz, mock_override, face_detections_override)
        face_count = len(faces_info)
        person_count = max(skeleton_count, face_count)

        auth_label = (
            f"{self.active_patient.hn} {self.active_patient.full_name}".strip()
            if self.active_patient is not None
            else "คนไข้หลัก"
        )

        # Rule 1: Multi-person dual-detector verification
        if skeleton_count > 1 or face_count > 1 or mock_override == "multi_person":
            self._locked_guard_status = "MULTI_PERSON"
            self._consecutive_unlock_matches = 0
            boxes: List[IdentityDetectedBoxSchema] = []
            source_boxes = skel_boxes if len(skel_boxes) > 1 else [f["bbox_1280"] for f in faces_info]
            if len(source_boxes) <= 1:
                source_boxes = [[240, 80, 580, 680], [720, 100, 1040, 670]]
            for idx, b in enumerate(source_boxes):
                boxes.append(
                    IdentityDetectedBoxSchema(
                        bbox=b,
                        label=auth_label if idx == 0 else "บุคคลเกินในเฟรม",
                        is_authorized=(idx == 0),
                    )
                )
            return IdentityGuardSchema(
                status="MULTI_PERSON",
                person_count=max(2, person_count),
                face_count=face_count,
                match_confidence=round(self._last_verified_sim, 3),
                is_counting_allowed=False,
                alert_title_th="ตรวจพบมีบุคคลอยู่หลายคนในเฟรม",
                alert_subtitle_th="กรุณาให้คนไข้ยืนหน้ากล้องเพียงลำพัง ระบบจึงจะนับรอบการฝึกต่อครับ",
                detected_boxes=boxes,
            )

        # If no patient is locked yet, stay verified while still guarding Rule 1 above
        if self.active_patient is None:
            return IdentityGuardSchema(
                status="VERIFIED",
                person_count=person_count,
                face_count=face_count,
                match_confidence=round(self._last_verified_sim, 3),
                is_counting_allowed=True,
                detected_boxes=[
                    IdentityDetectedBoxSchema(bbox=skel_boxes[0], label=auth_label, is_authorized=True)
                ],
            )

        should_run_1_1 = (
            force_identity_check
            or self._locked_guard_status != "VERIFIED"
            or mock_override is not None
            or face_detections_override is not None
            or (self._guard_frame_counter % max(1, self.settings.IDENTITY_CHECK_FRAME_INTERVAL) == 1)
        )

        if should_run_1_1:
            if face_count == 0:
                if self._locked_guard_status in ("MULTI_PERSON", "IMPOSTOR"):
                    self._consecutive_unlock_matches = 0
                else:
                    return IdentityGuardSchema(
                        status="FACE_NOT_VISIBLE",
                        person_count=person_count,
                        face_count=0,
                        match_confidence=round(self._last_verified_sim, 3),
                        is_counting_allowed=True,
                        alert_title_th="ไม่พบใบหน้าในมุมกล้อง",
                        alert_subtitle_th="กรุณาหันหน้าเข้าหากล้องเป็นระยะ",
                        detected_boxes=[
                            IdentityDetectedBoxSchema(bbox=skel_boxes[0], label=auth_label, is_authorized=True)
                        ],
                    )
            else:
                primary_face = max(
                    faces_info,
                    key=lambda f: float((f["bbox_1280"][2] - f["bbox_1280"][0]) * (f["bbox_1280"][3] - f["bbox_1280"][1])),
                )
                det_sc = float(primary_face["det_score"])
                emb = primary_face.get("embedding")
                sim = (
                    self._cosine_vs_active(emb, float(primary_face["yaw_deg"]), float(primary_face["pitch_deg"]))
                    if emb is not None
                    else 0.0
                )
                bound = self._is_face_bound_to_skeleton(primary_face["bbox_norm"], primary_xyz, primary_scores)
                self._last_verified_sim = sim

                # Rule 2: Impostor detection or spatial mismatch between face & active skeleton
                is_impostor_frame = (
                    mock_override == "impostor"
                    or (det_sc >= self.settings.IMPOSTOR_FACE_MIN_DET_SCORE and (sim < self.settings.IMPOSTOR_SIMILARITY_MIN or not bound))
                )
                if is_impostor_frame:
                    self._consecutive_impostor_misses += 1
                    self._consecutive_unlock_matches = 0
                    if (
                        mock_override == "impostor"
                        or self._consecutive_impostor_misses >= self.settings.IMPOSTOR_CONSECUTIVE_MISSES
                    ):
                        self._locked_guard_status = "IMPOSTOR"
                        self.dispatch_impersonation_audit_log(
                            patient_id=self.active_patient.patient_id,
                            similarity=sim,
                            reason="SPATIAL_MISMATCH" if not bound else "IMPERSONATION_BLOCKED",
                        )
                elif bound and sim >= self.settings.UNLOCK_VERIFY_SIMILARITY_MIN and skeleton_count == 1 and face_count == 1:
                    self._consecutive_impostor_misses = 0
                    if self._locked_guard_status in ("MULTI_PERSON", "IMPOSTOR"):
                        self._consecutive_unlock_matches += 1
                        if self._consecutive_unlock_matches >= self.settings.UNLOCK_CONSECUTIVE_MATCHES:
                            self._locked_guard_status = "VERIFIED"
                            self._impostor_logged_for_episode = False
                else:
                    self._consecutive_unlock_matches = 0

        if self._locked_guard_status == "IMPOSTOR":
            return IdentityGuardSchema(
                status="IMPOSTOR",
                person_count=person_count,
                face_count=face_count,
                match_confidence=round(max(0.0, self._last_verified_sim), 3),
                is_counting_allowed=False,
                alert_title_th="คุณไม่ใช่คนไข้ที่ลงทะเบียนไว้",
                alert_subtitle_th="ระบบจะไม่นับรอบการฝึก กรุณาให้คนไข้ตัวจริงเข้าฝึกครับ",
                detected_boxes=[
                    IdentityDetectedBoxSchema(
                        bbox=skel_boxes[0],
                        label="✕ ไม่ใช่คนไข้ที่ลงทะเบียน",
                        is_authorized=False,
                    )
                ],
            )

        if self._locked_guard_status == "MULTI_PERSON":
            return IdentityGuardSchema(
                status="MULTI_PERSON",
                person_count=person_count,
                face_count=face_count,
                match_confidence=round(max(0.0, self._last_verified_sim), 3),
                is_counting_allowed=False,
                alert_title_th="ตรวจพบมีบุคคลอยู่หลายคนในเฟรม",
                alert_subtitle_th=f"กำลังยืนยันตัวตนคนไข้หลัก ({self._consecutive_unlock_matches}/{self.settings.UNLOCK_CONSECUTIVE_MATCHES} เฟรม)",
                detected_boxes=[
                    IdentityDetectedBoxSchema(bbox=skel_boxes[0], label=auth_label, is_authorized=True)
                ],
            )

        return IdentityGuardSchema(
            status="VERIFIED",
            person_count=person_count,
            face_count=face_count,
            match_confidence=round(max(0.0, self._last_verified_sim), 3),
            is_counting_allowed=True,
            detected_boxes=[
                IdentityDetectedBoxSchema(bbox=skel_boxes[0], label=auth_label, is_authorized=True)
            ],
        )

    def start_enrollment(
        self, patient_id: Optional[str] = None, hn: Optional[str] = None, full_name: Optional[str] = None
    ) -> FaceEnrollmentStateSchema:
        session_id = f"enroll-{uuid.uuid4().hex[:10]}"
        session = FaceEnrollmentSession(
            session_id=session_id,
            patient_id=patient_id,
            hn=hn,
            full_name=full_name,
            bucket_targets=dict(self.settings.FACE_BUCKET_TARGETS),
        )
        self._enroll_sessions[session_id] = session
        self._active_session_id = session_id
        return self._session_to_state(session, active=True, face_detected=False)

    def _session_to_state(
        self,
        session: FaceEnrollmentSession,
        active: bool = True,
        yaw: float = 0.0,
        pitch: float = 0.0,
        roll: float = 0.0,
        sharpness: float = 0.0,
        face_detected: bool = False,
        phase: str = "scanning",
    ) -> FaceEnrollmentStateSchema:
        counts = {b: len(session.bucket_samples[b]) for b in BUCKET_ORDER}
        total = sum(counts.values())
        next_target = next(
            (b for b in BUCKET_ORDER if counts[b] < session.bucket_targets[b]),
            "center",
        )
        guidance_map = {
            "center": f"มองตรงเข้ากล้อง ({counts['center']}/{session.bucket_targets['center']} เฟรม)",
            "left": f"หันศีรษะไปทางซ้ายเล็กน้อย ({counts['left']}/{session.bucket_targets['left']} เฟรม)",
            "right": f"หันศีรษะไปทางขวาเล็กน้อย ({counts['right']}/{session.bucket_targets['right']} เฟรม)",
            "up": f"เงยหน้าขึ้นเล็กน้อย ({counts['up']}/{session.bucket_targets['up']} เฟรม)",
            "down": f"ก้มหน้าลงเล็กน้อย ({counts['down']}/{session.bucket_targets['down']} เฟรม)",
        }
        return FaceEnrollmentStateSchema(
            active=active,
            session_id=session.session_id,
            phase=phase,  # type: ignore[arg-type]
            frames_captured=total,
            target_frames=self.settings.FACE_TOTAL_FRAMES,
            bucket_counts=counts,
            bucket_targets=dict(session.bucket_targets),
            current_pose=next_target,
            yaw_deg=yaw,
            pitch_deg=pitch,
            roll_deg=roll,
            sharpness=sharpness,
            face_detected=face_detected,
            guidance_th="สแกนครบ 120 เฟรมเรียบร้อยแล้ว" if total >= self.settings.FACE_TOTAL_FRAMES else guidance_map[next_target],
        )

    def process_enrollment_frame(
        self,
        session_id: str,
        frame_bgr: np.ndarray,
        forced_bucket: Optional[str] = None,
    ) -> FaceEnrollmentStateSchema:
        session = self._enroll_sessions.get(session_id)
        if session is None:
            raise KeyError(f"Enrollment session not found: {session_id}")

        feat = self.extract_face_features(frame_bgr)
        if feat is None:
            return self._session_to_state(session, active=True, face_detected=False)

        yaw = float(feat["yaw_deg"])
        pitch = float(feat["pitch_deg"])
        roll = float(feat["roll_deg"])
        sharpness = float(feat["sharpness"])

        bucket = forced_bucket or classify_face_bucket(yaw, pitch)
        # Allow completing remaining quota when camera has limited yaw range during mock/testing
        if len(session.bucket_samples[bucket]) >= session.bucket_targets[bucket]:
            open_buckets = [b for b in BUCKET_ORDER if len(session.bucket_samples[b]) < session.bucket_targets[b]]
            if not open_buckets:
                return self._session_to_state(
                    session, True, yaw, pitch, roll, sharpness, True, phase="completed"
                )
            if forced_bucket is None and self._face_app is None:
                bucket = open_buckets[0]
            else:
                return self._session_to_state(session, True, yaw, pitch, roll, sharpness, True, phase="scanning")

        session.bucket_samples[bucket].append(feat["embedding"])
        session.sharpness_scores.append(sharpness)
        session.last_capture_at = time.time()

        if bucket == "center" and sharpness > session.best_frontal_score and feat.get("avatar_url"):
            session.best_frontal_score = sharpness
            session.best_frontal_avatar = str(feat["avatar_url"])

        total = sum(len(session.bucket_samples[b]) for b in BUCKET_ORDER)
        phase = "completed" if total >= self.settings.FACE_TOTAL_FRAMES else "scanning"
        return self._session_to_state(session, True, yaw, pitch, roll, sharpness, True, phase=phase)

    def finalize_enrollment(
        self,
        session_id: str,
        patient_id: Optional[str] = None,
        hn: Optional[str] = None,
        persist_to_supabase: bool = True,
    ) -> FaceID120Response:
        session = self._enroll_sessions.get(session_id)
        if session is None:
            raise KeyError(f"Enrollment session not found: {session_id}")

        all_vecs: List[np.ndarray] = []
        for bucket in BUCKET_ORDER:
            all_vecs.extend(session.bucket_samples[bucket])
        if not all_vecs:
            raise ValueError("No valid face frames captured in enrollment session.")

        master_vec = l2_normalize(np.mean(np.vstack(all_vecs), axis=0))
        angle_embeddings: Dict[str, List[float]] = {}
        for bucket in BUCKET_ORDER:
            vecs = session.bucket_samples[bucket]
            centroid = l2_normalize(np.mean(np.vstack(vecs), axis=0)) if vecs else master_vec
            angle_embeddings[bucket] = [round(float(v), 6) for v in centroid.tolist()]

        master_list = [round(float(v), 6) for v in master_vec.tolist()]
        mean_sharp = float(np.mean(session.sharpness_scores)) if session.sharpness_scores else 80.0
        diversity = sum(1 for b in BUCKET_ORDER if session.bucket_samples[b])
        quality_score = round(min(0.99, 0.75 + min(0.16, mean_sharp / 380.0) + diversity * 0.016), 3)
        counts = {b: len(session.bucket_samples[b]) for b in BUCKET_ORDER}
        frames_captured = sum(counts.values())

        target_pid = patient_id or session.patient_id
        target_hn = hn or session.hn or ""
        supabase_synced = False

        if persist_to_supabase and self._supabase is not None and target_pid:
            try:
                self._supabase.table("patient_biometrics").upsert(
                    {
                        "patient_id": target_pid,
                        "hn": target_hn,
                        "master_embedding": master_list,
                        "angle_embeddings": angle_embeddings,
                        "frames_captured": frames_captured,
                        "quality_score": quality_score,
                        "avatar_url": session.best_frontal_avatar or None,
                    },
                    on_conflict="patient_id",
                ).execute()
                self._supabase.table("patients").update(
                    {
                        "face_biometric_status": "VERIFIED",
                        "face_frames_count": frames_captured,
                        "face_image_url": session.best_frontal_avatar or None,
                    }
                ).eq("id", target_pid).execute()
                supabase_synced = True
                self.sync_from_supabase()
            except Exception as exc:
                logger.warning("Failed persisting FaceID120 to Supabase: %s", exc)

        # Register into local RAM cache immediately even without Supabase round-trip
        if target_pid:
            cached = CachedPatientBiometric(
                patient_id=target_pid,
                hn=target_hn,
                full_name=session.full_name or target_hn or target_pid,
                master_embedding=master_vec,
                angle_embeddings={b: l2_normalize(np.asarray(angle_embeddings[b])) for b in BUCKET_ORDER},
            )
            self._patients = [p for p in self._patients if p.patient_id != target_pid] + [cached]
            self._matrix = np.vstack([p.master_embedding for p in self._patients])

        if self._active_session_id == session_id:
            self._active_session_id = None

        return FaceID120Response(
            session_id=session_id,
            master_embedding=master_list,
            angle_embeddings=angle_embeddings,
            frames_captured=frames_captured,
            quality_score=quality_score,
            avatar_data_url=session.best_frontal_avatar,
            bucket_counts=counts,
            supabase_synced=supabase_synced,
        )

