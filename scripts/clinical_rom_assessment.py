import math
import os
import time
import urllib.request
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

try:
    from PIL import Image, ImageDraw, ImageFont

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

MODEL_PATH = os.path.join("models", "pose_landmarker_heavy.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
)
HAND_MODEL_PATH = os.path.join("models", "hand_landmarker.task")
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

HAND21_FINGER_CHAINS: dict[str, tuple[int, int, int, int]] = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
HAND21_FINGER_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "thumb": (11, 158, 245),
    "index": (248, 189, 56),
    "middle": (129, 185, 16),
    "ring": (250, 139, 167),
    "pinky": (182, 114, 244),
}
HAND21_PALM_RING: tuple[int, ...] = (0, 5, 9, 13, 17)
PINCH_LABELS_TH: dict[str, str] = {
    "index": "นิ้วชี้",
    "middle": "นิ้วกลาง",
    "ring": "นิ้วนาง",
    "pinky": "นิ้วก้อย",
}


def ensure_task_model(model_path: str, model_url: str, label: str) -> str:
    """ตรวจสอบไฟล์โมเดล MediaPipe Tasks และดาวน์โหลดอัตโนมัติหากยังไม่มี"""
    model_dir = os.path.dirname(model_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    if not os.path.exists(model_path):
        print(f"[INFO] Downloading {label} model to {model_path} ...")
        urllib.request.urlretrieve(model_url, model_path)
        print(f"[INFO] {label} download complete.")
    return model_path


def ensure_pose_landmarker_model(
    model_path: str = MODEL_PATH,
    model_url: str = MODEL_URL,
) -> str:
    """ตรวจสอบไฟล์โมเดล MediaPipe Pose Landmarker Heavy และดาวน์โหลดอัตโนมัติหากยังไม่มี"""
    return ensure_task_model(model_path, model_url, "PoseLandmarker Heavy")


def create_pose_landmarker(
    model_path: str = MODEL_PATH,
    model_url: str = MODEL_URL,
) -> vision.PoseLandmarker:
    """สร้างอินสแตนซ์ PoseLandmarker (Tasks Vision API) ในโหมด VIDEO ความละเอียดสูง"""
    resolved_path = ensure_pose_landmarker_model(model_path=model_path, model_url=model_url)
    base_options = python.BaseOptions(model_asset_path=resolved_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.7,
        min_pose_presence_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    return vision.PoseLandmarker.create_from_options(options)


def create_hand_landmarker(
    model_path: str = HAND_MODEL_PATH,
    model_url: str = HAND_MODEL_URL,
) -> vision.HandLandmarker | None:
    """สร้างอินสแตนซ์ MediaPipe HandLandmarker 21 จุด (รองรับ 2 มือ + พิกัด 3 มิติ world_landmarks)"""
    try:
        resolved_path = ensure_task_model(model_path, model_url, "HandLandmarker 21-pt")
        base_options = python.BaseOptions(model_asset_path=resolved_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.45,
            min_hand_presence_confidence=0.45,
            min_tracking_confidence=0.45,
        )
        return vision.HandLandmarker.create_from_options(options)
    except Exception as exc:
        print(f"[WARN] HandLandmarker 21-pt unavailable ({exc}); continuing with PoseLandmarker only.")
        return None


class OneEuroFilter:
    """
    อัลกอริทึม 1€ Filter มาตรฐานงานวิจัยและเครื่องมือแพทย์
    ใช้อัตราการเปลี่ยนแปลงตามเวลาจริง (dt) เพื่อตัดอาการสั่นโดยไม่ทำให้ภาพหน่วง
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.05, d_cutoff: float = 1.0) -> None:
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev: np.ndarray | None = None
        self.dx_prev: np.ndarray | None = None
        self.t_prev: float | None = None

    def _smoothing_factor(self, dt: float, cutoff: np.ndarray | float) -> np.ndarray | float:
        r = 2.0 * math.pi * cutoff * dt
        return r / (r + 1.0)

    def apply(self, x, t: float | None = None) -> np.ndarray:
        if t is None:
            t = time.perf_counter()
        x_arr = np.asarray(x, dtype=np.float64)

        if self.x_prev is None or self.x_prev.shape != x_arr.shape or self.t_prev is None:
            self.x_prev = x_arr
            self.dx_prev = np.zeros_like(x_arr, dtype=np.float64)
            self.t_prev = t
            return x_arr

        dt = t - self.t_prev
        if dt <= 1e-6:
            return self.x_prev

        # คำนวณความเร็วการเคลื่อนที่และกรองสัญญาณความเร็ว
        a_d = self._smoothing_factor(dt, self.d_cutoff)
        dx = (x_arr - self.x_prev) / dt
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev

        # ปรับค่า Cutoff อัตโนมัติตามความเร็ว (ขยับช้า = นิ่งสนิท, ขยับเร็ว = ตามทันทันที)
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self._smoothing_factor(dt, cutoff)
        x_hat = a * x_arr + (1.0 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

    def reset(self) -> None:
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None


def calculate_3d_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """คำนวณมุมจริงในปริภูมิ 3 มิติ (0-180 องศา) จาก Dot Product โดยมีจุด b เป็นจุดหมุนข้อต่อ"""
    ba = a - b
    bc = c - b
    norm_ba = float(np.linalg.norm(ba))
    norm_bc = float(np.linalg.norm(bc))
    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return 0.0
    cosine_angle = float(np.dot(ba, bc)) / (norm_ba * norm_bc)
    cosine_angle = float(np.clip(cosine_angle, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine_angle)))


def calculate_2d_tilt_angle(p1: np.ndarray, p2: np.ndarray) -> float:
    """วัดมุมความเอียงของเส้นเชื่อมจุด 2 จุด เทียบกับแนวระนาบ (ตรวจจับการยกไหล่/เอียงตัว)"""
    dy = abs(float(p1[1]) - float(p2[1]))
    dx = abs(float(p1[0]) - float(p2[0]))
    if dx < 1e-6:
        return 90.0
    return float(np.degrees(np.arctan2(dy, dx)))


def _load_thai_font(size: int):
    if not PIL_AVAILABLE:
        return None
    candidates = [
        r"C:\Windows\Fonts\leelawui.ttf",
        r"C:\Windows\Fonts\LeelawUI.ttf",
        r"C:\Windows\Fonts\tahoma.ttf",
        r"C:\Windows\Fonts\Tahoma.ttf",
        "/usr/share/fonts/truetype/tlwg/Garuda.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansThai-Regular.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return None


def draw_hud_text(
    frame: np.ndarray,
    text: str,
    pos: tuple[int, int],
    color_bgr: tuple[int, int, int] = (255, 255, 255),
    font_size: int = 20,
    fallback_ascii: str | None = None,
) -> np.ndarray:
    """แสดงข้อความภาษาไทย/อังกฤษที่คมชัดบน OpenCV (ใช้ Pillow หากมีฟอนต์ไทย มิฉะนั้นใช้ ASCII Fallback)"""
    font = _load_thai_font(font_size) if PIL_AVAILABLE else None
    if font is not None:
        rgb_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(rgb_img)
        color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
        draw.text(pos, text, font=font, fill=color_rgb)
        return cv2.cvtColor(np.asarray(rgb_img), cv2.COLOR_RGB2BGR)

    label = fallback_ascii if fallback_ascii is not None else text.encode("ascii", "ignore").decode("ascii")
    cv2.putText(
        frame,
        label,
        (pos[0], pos[1] + font_size),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.5, font_size / 30.0),
        color_bgr,
        2,
    )
    return frame


def draw_joint_angle_arc(
    frame: np.ndarray,
    joint_pt: tuple[int, int],
    ref_pt: tuple[int, int],
    moving_pt: tuple[int, int],
    angle_deg: float,
    color_bgr: tuple[int, int, int],
    radius: int = 44,
) -> None:
    """วาดเส้นโค้ง (Arc) แสดงมุมที่ข้อต่อไหล่แบบ Real-time ระหว่างแกนลำตัวกับแขน"""
    ref_ang = math.degrees(math.atan2(ref_pt[1] - joint_pt[1], ref_pt[0] - joint_pt[0]))
    mov_ang = math.degrees(math.atan2(moving_pt[1] - joint_pt[1], moving_pt[0] - joint_pt[0]))
    delta = (mov_ang - ref_ang + 540.0) % 360.0 - 180.0

    overlay = frame.copy()
    cv2.ellipse(
        overlay,
        joint_pt,
        (radius, radius),
        0.0,
        ref_ang,
        ref_ang + delta,
        color_bgr,
        -1,
    )
    cv2.addWeighted(overlay, 0.28, frame, 0.72, 0.0, frame)
    cv2.ellipse(
        frame,
        joint_pt,
        (radius, radius),
        0.0,
        ref_ang,
        ref_ang + delta,
        color_bgr,
        3,
    )
    cv2.putText(
        frame,
        f"{angle_deg:.1f} deg",
        (joint_pt[0] - 45, joint_pt[1] - radius - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
    )


def analyze_hand_21_metrics(
    pts_3d: np.ndarray,
    side_label: str = "Right",
) -> dict[str, object]:
    """วิเคราะห์ชีวกลศาสตร์มือ 21 จุดจาก MediaPipe HandLandmarker (นิ้ว 5 นิ้ว, การจีบนิ้ว, มุมหมุนฝ่ามือ 3 มิติ)"""
    wrist = pts_3d[0]
    index_mcp = pts_3d[5]
    middle_mcp = pts_3d[9]
    pinky_mcp = pts_3d[17]
    palm_width = max(1e-4, float(np.linalg.norm(index_mcp - pinky_mcp)))

    extended_count = 0
    finger_ext_pcts: dict[str, float] = {}
    for fname, (c0, c1, c2, c3) in HAND21_FINGER_CHAINS.items():
        if fname == "thumb":
            pip_ang = calculate_3d_angle(pts_3d[c0], pts_3d[c1], pts_3d[c2])
            dip_ang = calculate_3d_angle(pts_3d[c1], pts_3d[c2], pts_3d[c3])
            ext_pct = float(np.clip(((0.5 * (pip_ang + dip_ang)) - 115.0) / 55.0 * 100.0, 0.0, 100.0))
            is_ext = ext_pct >= 55.0
        else:
            mcp_ang = calculate_3d_angle(wrist, pts_3d[c0], pts_3d[c1])
            pip_ang = calculate_3d_angle(pts_3d[c0], pts_3d[c1], pts_3d[c2])
            dip_ang = calculate_3d_angle(pts_3d[c1], pts_3d[c2], pts_3d[c3])
            mean_ang = 0.25 * mcp_ang + 0.50 * pip_ang + 0.25 * dip_ang
            ext_pct = float(np.clip((mean_ang - 90.0) / 80.0 * 100.0, 0.0, 100.0))
            is_ext = pip_ang >= 145.0 and ext_pct >= 62.0
        finger_ext_pcts[fname] = round(ext_pct, 1)
        if is_ext:
            extended_count += 1

    # True 3D palm normal for signed pronation/supination roll [-90..+90 deg]
    v_palm = middle_mcp - wrist
    u_palm = pinky_mcp - index_mcp
    normal = np.cross(v_palm, u_palm)
    n_mag = float(np.linalg.norm(normal))
    side_sign = 1.0 if side_label.lower() == "right" else -1.0
    if n_mag > 1e-6:
        up_comp = (-float(normal[1]) * side_sign) / n_mag
        cam_comp = (-float(normal[2]) * side_sign) / n_mag
        side_comp = abs(float(normal[0])) / n_mag
        roll_deg = float(np.clip(math.degrees(math.atan2(up_comp * 0.72 + cam_comp * 0.68, max(0.08, side_comp))), -90.0, 90.0))
    else:
        roll_deg = 0.0

    palm_label = "หงายฝ่ามือ" if roll_deg >= 28.0 else "คว่ำฝ่ามือ" if roll_deg <= -28.0 else "สันมือตรง"

    thumb_tip = pts_3d[4]
    pinch_map = {
        "index": float(np.linalg.norm(thumb_tip - pts_3d[8])) / palm_width,
        "middle": float(np.linalg.norm(thumb_tip - pts_3d[12])) / palm_width,
        "ring": float(np.linalg.norm(thumb_tip - pts_3d[16])) / palm_width,
        "pinky": float(np.linalg.norm(thumb_tip - pts_3d[20])) / palm_width,
    }
    best_pinch = min(pinch_map, key=pinch_map.get)
    active_pinch = best_pinch if pinch_map[best_pinch] <= 0.42 else None

    return {
        "extended_count": extended_count,
        "finger_ext_pcts": finger_ext_pcts,
        "roll_deg": round(roll_deg, 1),
        "palm_label_th": palm_label,
        "active_pinch": active_pinch,
        "active_pinch_th": PINCH_LABELS_TH[active_pinch] if active_pinch else None,
    }


def draw_hand_21_overlay(
    frame: np.ndarray,
    pts_2d: np.ndarray,
    metrics: dict[str, object],
) -> None:
    """วาดโครงกระดูกมือและนิ้วทั้ง 5 นิ้วละเอียด 21 จุด แยกสีตามนิ้วพร้อมเส้นเชื่อมการจีบนิ้ว"""
    palm_poly = np.array([pts_2d[idx] for idx in HAND21_PALM_RING], dtype=np.int32)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [palm_poly], (180, 140, 40))
    cv2.addWeighted(overlay, 0.18, frame, 0.82, 0.0, frame)
    cv2.polylines(frame, [palm_poly], True, (220, 200, 90), 2)

    wrist_pt = tuple(pts_2d[0].astype(int))
    for fname, chain in HAND21_FINGER_CHAINS.items():
        color = HAND21_FINGER_COLORS_BGR[fname]
        first_pt = tuple(pts_2d[chain[0]].astype(int))
        cv2.line(frame, wrist_pt, first_pt, color, 2)
        for i in range(len(chain) - 1):
            p_a = tuple(pts_2d[chain[i]].astype(int))
            p_b = tuple(pts_2d[chain[i + 1]].astype(int))
            cv2.line(frame, p_a, p_b, color, 3)

    active_pinch = metrics.get("active_pinch")
    if active_pinch in ("index", "middle", "ring", "pinky"):
        tip_idx = {"index": 8, "middle": 12, "ring": 16, "pinky": 20}[str(active_pinch)]
        cv2.line(
            frame,
            tuple(pts_2d[4].astype(int)),
            tuple(pts_2d[tip_idx].astype(int)),
            (130, 240, 80),
            4,
        )

    for i in range(21):
        pt = tuple(pts_2d[i].astype(int))
        is_tip = i in (4, 8, 12, 16, 20)
        is_wrist = i == 0
        if is_wrist:
            color = (240, 210, 60)
        elif i <= 4:
            color = HAND21_FINGER_COLORS_BGR["thumb"]
        elif i <= 8:
            color = HAND21_FINGER_COLORS_BGR["index"]
        elif i <= 12:
            color = HAND21_FINGER_COLORS_BGR["middle"]
        elif i <= 16:
            color = HAND21_FINGER_COLORS_BGR["ring"]
        else:
            color = HAND21_FINGER_COLORS_BGR["pinky"]
        radius = 6 if (is_tip or is_wrist) else 4
        cv2.circle(frame, pt, radius + 1, (255, 255, 255), -1)
        cv2.circle(frame, pt, radius - 1, color, -1)


def main() -> None:
    # ตั้งค่า MediaPipe Tasks Vision API (PoseLandmarker Heavy + HandLandmarker 21-pt) สำหรับงานคลินิก
    landmarker = create_pose_landmarker()
    hand_landmarker = create_hand_landmarker()

    # สร้าง 1€ Filter แยกสำหรับพิกัด 2D (วาดจอ), 3D (คำนวณมุมจริง) และตัวเลขมุม
    filter_2d = OneEuroFilter(min_cutoff=0.8, beta=0.04)
    filter_3d = OneEuroFilter(min_cutoff=0.6, beta=0.03)
    angle_filter = OneEuroFilter(min_cutoff=0.5, beta=0.02)
    hand_filters_2d = [OneEuroFilter(min_cutoff=1.2, beta=0.06), OneEuroFilter(min_cutoff=1.2, beta=0.06)]
    hand_filters_3d = [OneEuroFilter(min_cutoff=1.0, beta=0.05), OneEuroFilter(min_cutoff=1.0, beta=0.05)]

    # พารามิเตอร์สำหรับท่ากายภาพ: กางแขนซ้าย (Left Shoulder Abduction: 23 -> 11 -> 13)
    TARGET_MIN_ANGLE = 85.0
    TARGET_MAX_ANGLE = 180.0
    REQUIRED_HOLD_TIME = 3.0
    VISIBILITY_THRESHOLD = 0.75  # คัดกรองความชัดเจนขั้นต่ำ 0.75
    MAX_SHOULDER_TILT_DEG = 12.0
    MIN_ELBOW_EXTENSION_DEG = 150.0

    reps = 0
    hold_time = 0.0
    start_time = time.perf_counter()
    last_frame_time = start_time
    last_timestamp_ms = 0
    rep_completed_in_current_hold = False

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            now = time.perf_counter()
            dt = max(0.0, now - last_frame_time)
            last_frame_time = now

            # สร้าง timestamp_ms แบบเพิ่มขึ้นต่อเนื่องทุกเฟรมตามข้อกำหนดโหมด VIDEO
            timestamp_ms = int((now - start_time) * 1000)
            if timestamp_ms <= last_timestamp_ms:
                timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = timestamp_ms

            # พลิกภาพกระจกให้ผู้ป่วยดูตามได้ง่าย
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb_frame = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = landmarker.detect_for_video(mp_image, timestamp_ms)
            hand_results = (
                hand_landmarker.detect_for_video(mp_image, timestamp_ms)
                if hand_landmarker is not None
                else None
            )

            feedback_th = f"เตรียมพร้อม: กางแขนซ้ายขึ้นระดับ {TARGET_MIN_ANGLE:.0f}-{TARGET_MAX_ANGLE:.0f} องศา"
            feedback_ascii = f"Ready: Abduct left arm to {TARGET_MIN_ANGLE:.0f}-{TARGET_MAX_ANGLE:.0f} deg"
            status_color = (220, 220, 220)
            shoulder_angle = 0.0
            hand_summary_th = "Hand21: รอตรวจจับนิ้วมือ"
            hand_summary_ascii = "Hand21: Waiting for hand"

            if results.pose_landmarks and results.pose_world_landmarks:
                raw_2d = results.pose_landmarks[0]
                raw_3d = results.pose_world_landmarks[0]

                # ตรวจสอบ visibility ของทุกจุดที่ใช้ (11, 12, 13, 15, 23)
                req_indices = [11, 12, 13, 15, 23]
                visibilities = [
                    raw_2d[i].visibility if raw_2d[i].visibility is not None else 0.0
                    for i in req_indices
                ]

                if min(visibilities) < VISIBILITY_THRESHOLD:
                    # ระงับการวัดค่าชั่วคราวและหยุดเวลา Hold Timer ทันที
                    feedback_th = "แจ้งเตือน: มองเห็นข้อต่อไม่ชัดเจน (< 0.75) กรุณาจัดตำแหน่งกล้องใหม่"
                    feedback_ascii = "Alert: Low visibility (< 0.75) - Reposition camera"
                    status_color = (0, 165, 255)
                else:
                    pts_2d = np.array([[lm.x * w, lm.y * h] for lm in raw_2d], dtype=np.float64)
                    pts_3d = np.array([[lm.x, lm.y, lm.z] for lm in raw_3d], dtype=np.float64)

                    smooth_2d = filter_2d.apply(pts_2d, now)
                    smooth_3d = filter_3d.apply(pts_3d, now)

                    # 1. มุมกางแขนซ้าย 3 มิติ (23 -> 11 -> 13) จาก pose_world_landmarks
                    raw_shoulder_angle = calculate_3d_angle(smooth_3d[23], smooth_3d[11], smooth_3d[13])
                    shoulder_angle = float(angle_filter.apply(raw_shoulder_angle, now))

                    # 2. มุมเหยียดข้อศอกซ้าย 3 มิติ (11 -> 13 -> 15)
                    elbow_angle = calculate_3d_angle(smooth_3d[11], smooth_3d[13], smooth_3d[15])

                    # 3. มุมความเอียงระหว่างไหล่ซ้าย (11) และไหล่ขวา (12) เทียบแนวระนาบ
                    shoulder_tilt = calculate_2d_tilt_angle(smooth_2d[11], smooth_2d[12])

                    # ตรวจจับการชดเชยท่า (Compensation Detection)
                    is_compensated = False
                    if shoulder_tilt > MAX_SHOULDER_TILT_DEG:
                        feedback_th = f"ห้ามยกไหล่หรือเอียงตัวช่วย (เอียง {shoulder_tilt:.1f} องศา)"
                        feedback_ascii = f"Warning: Do not shrug or tilt ({shoulder_tilt:.1f} deg)"
                        status_color = (0, 0, 255)
                        is_compensated = True
                    elif elbow_angle < MIN_ELBOW_EXTENSION_DEG and shoulder_angle > 25.0:
                        feedback_th = f"กรุณาเหยียดแขนให้ตรง (มุมข้อศอก {elbow_angle:.0f} องศา)"
                        feedback_ascii = f"Warning: Straighten elbow ({elbow_angle:.0f} deg)"
                        status_color = (0, 0, 255)
                        is_compensated = True

                    in_target_zone = TARGET_MIN_ANGLE <= shoulder_angle <= TARGET_MAX_ANGLE

                    if is_compensated:
                        # หยุดเวลา Hold Timer ทันทีเมื่อตรวจพบการชดเชยท่า
                        pass
                    elif in_target_zone:
                        if not rep_completed_in_current_hold:
                            hold_time += dt
                            feedback_th = (
                                f"เยี่ยมมาก! ค้างท่าต่อเนื่อง ({hold_time:.1f} / {REQUIRED_HOLD_TIME:.1f} วินาที)"
                            )
                            feedback_ascii = f"Good! Hold steady ({hold_time:.1f} / {REQUIRED_HOLD_TIME:.1f}s)"
                            status_color = (0, 220, 0)
                            if hold_time >= REQUIRED_HOLD_TIME:
                                reps += 1
                                rep_completed_in_current_hold = True
                                hold_time = REQUIRED_HOLD_TIME
                                feedback_th = "สำเร็จ 1 ครั้ง! ค่อยๆ ลดแขนลงข้างลำตัวได้"
                                feedback_ascii = "Rep Completed! Lower arm slowly"
                        else:
                            feedback_th = "ครบเวลาแล้ว ค่อยๆ ลดแขนลงข้างลำตัวเพื่อเริ่มครั้งถัดไป"
                            feedback_ascii = "Done! Lower arm to prepare next rep"
                            status_color = (255, 200, 0)
                    else:
                        # แขนตก/หลุดจากช่วงเป้าหมายก่อนครบเวลาให้รีเซ็ตเวลาใหม่ทันที
                        if not rep_completed_in_current_hold:
                            hold_time = 0.0
                        if shoulder_angle < TARGET_MIN_ANGLE:
                            if shoulder_angle < 35.0:
                                rep_completed_in_current_hold = False
                                hold_time = 0.0
                            feedback_th = f"กางแขนขึ้นสู่ช่วงเป้าหมาย ({TARGET_MIN_ANGLE:.0f}-{TARGET_MAX_ANGLE:.0f} องศา)"
                            feedback_ascii = f"Raise arm to target ({TARGET_MIN_ANGLE:.0f}-{TARGET_MAX_ANGLE:.0f} deg)"
                            status_color = (255, 190, 0)
                        else:
                            feedback_th = "กางแขนสูงเกินไป ลดระดับลงเล็กน้อย"
                            feedback_ascii = "Too high! Lower arm slightly"
                            status_color = (0, 140, 255)

                    # พิกัดข้อต่อ 2 มิติที่กรองแล้วสำหรับวาด Clinical Overlay
                    p_sh_l = tuple(smooth_2d[11].astype(int))
                    p_sh_r = tuple(smooth_2d[12].astype(int))
                    p_el_l = tuple(smooth_2d[13].astype(int))
                    p_wr_l = tuple(smooth_2d[15].astype(int))
                    p_hip_l = tuple(smooth_2d[23].astype(int))

                    tilt_color = (0, 0, 255) if shoulder_tilt > MAX_SHOULDER_TILT_DEG else (255, 255, 0)
                    cv2.line(frame, p_sh_l, p_sh_r, tilt_color, 3)

                    arm_color = (
                        (0, 0, 255)
                        if is_compensated
                        else (0, 230, 0)
                        if in_target_zone
                        else (0, 165, 255)
                    )
                    cv2.line(frame, p_hip_l, p_sh_l, (200, 200, 200), 4)
                    cv2.line(frame, p_sh_l, p_el_l, arm_color, 6)
                    cv2.line(frame, p_el_l, p_wr_l, arm_color, 6)

                    # วาดเส้นโค้ง Arc แสดงมุมที่ข้อต่อไหล่พร้อมตัวเลขทศนิยม 1 ตำแหน่ง
                    draw_joint_angle_arc(frame, p_sh_l, p_hip_l, p_el_l, shoulder_angle, arm_color)

                    for pt in (p_sh_l, p_sh_r, p_el_l, p_wr_l, p_hip_l):
                        cv2.circle(frame, pt, 8, (255, 255, 255), -1)
                        cv2.circle(frame, pt, 6, arm_color, -1)
            else:
                hold_time = 0.0
                rep_completed_in_current_hold = False
                filter_2d.reset()
                filter_3d.reset()
                angle_filter.reset()

            # ตรวจจับและวาดนิ้วมือละเอียด 21 จุดจาก MediaPipe HandLandmarker (2D + 3D world landmarks)
            if hand_results and hand_results.hand_landmarks:
                hand_labels_th: list[str] = []
                hand_labels_ascii: list[str] = []
                for hi, raw_hand_2d in enumerate(hand_results.hand_landmarks[:2]):
                    pts_h2d = np.array([[lm.x * w, lm.y * h] for lm in raw_hand_2d], dtype=np.float64)
                    smooth_h2d = hand_filters_2d[hi].apply(pts_h2d, now)
                    if hand_results.hand_world_landmarks and hi < len(hand_results.hand_world_landmarks):
                        raw_h3d = hand_results.hand_world_landmarks[hi]
                        pts_h3d = np.array([[lm.x, lm.y, lm.z] for lm in raw_h3d], dtype=np.float64)
                    else:
                        pts_h3d = np.array([[lm.x, lm.y, lm.z] for lm in raw_hand_2d], dtype=np.float64)
                    smooth_h3d = hand_filters_3d[hi].apply(pts_h3d, now)
                    side_cat = "Right"
                    if hand_results.handedness and hi < len(hand_results.handedness) and hand_results.handedness[hi]:
                        side_cat = hand_results.handedness[hi][0].category_name
                    h_metrics = analyze_hand_21_metrics(smooth_h3d, side_label=side_cat)
                    draw_hand_21_overlay(frame, smooth_h2d, h_metrics)
                    pinch_txt = f" | จีบ{h_metrics['active_pinch_th']}" if h_metrics["active_pinch_th"] else ""
                    hand_labels_th.append(
                        f"มือ{hi + 1}: เหยียด {h_metrics['extended_count']}/5 นิ้ว | {h_metrics['palm_label_th']} ({h_metrics['roll_deg']:+.0f} deg){pinch_txt}"
                    )
                    hand_labels_ascii.append(
                        f"H{hi + 1}: {h_metrics['extended_count']}/5 ext | Roll {h_metrics['roll_deg']:+.0f} deg"
                    )
                if hand_labels_th:
                    hand_summary_th = "  /  ".join(hand_labels_th)
                    hand_summary_ascii = "  /  ".join(hand_labels_ascii)
            else:
                for hf in hand_filters_2d:
                    hf.reset()
                for hf in hand_filters_3d:
                    hf.reset()

            # แผงควบคุม Clinical HUD ด้านบนจอ
            cv2.rectangle(frame, (0, 0), (w, 152), (24, 24, 32), -1)
            cv2.putText(frame, f"REPS: {reps}", (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.35, (255, 255, 255), 3)
            cv2.putText(
                frame,
                f"ROM: {shoulder_angle:.1f} deg  |  Target: {TARGET_MIN_ANGLE:.0f}-{TARGET_MAX_ANGLE:.0f} deg",
                (24, 92),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (160, 255, 190),
                2,
            )

            # หลอด Progress Bar สำหรับ Hold Timer
            bar_x, bar_y, bar_w, bar_h = 340, 20, 380, 32
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (70, 70, 80), -1)
            fill_ratio = max(0.0, min(1.0, hold_time / REQUIRED_HOLD_TIME))
            bar_fill_color = (0, 0, 255) if status_color == (0, 0, 255) else (0, 220, 100)
            cv2.rectangle(
                frame,
                (bar_x, bar_y),
                (bar_x + int(bar_w * fill_ratio), bar_y + bar_h),
                bar_fill_color,
                -1,
            )
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 2)
            cv2.putText(
                frame,
                f"HOLD: {hold_time:.1f}s / {REQUIRED_HOLD_TIME:.1f}s",
                (bar_x + 95, bar_y + 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

            # แถบสถานะแจ้งเตือนความถูกต้องของท่าทาง
            frame = draw_hud_text(
                frame,
                f"สถานะ: {feedback_th}",
                (470, 74),
                color_bgr=status_color,
                font_size=22,
                fallback_ascii=f"Status: {feedback_ascii}",
            )
            frame = draw_hud_text(
                frame,
                f"นิ้วมือ 21 จุด: {hand_summary_th}",
                (24, 118),
                color_bgr=(250, 220, 120),
                font_size=18,
                fallback_ascii=hand_summary_ascii,
            )

            cv2.imshow("Clinical ROM Assessment (High-Precision 3D + Hand21)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        landmarker.close()
        if hand_landmarker is not None:
            hand_landmarker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
