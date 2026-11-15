import cv2
import mediapipe as mp
import numpy as np


class LandmarkSmoother:
    """
    คลาสสำหรับลดอาการสั่น (Jittering) ของจุด Landmark ทั้ง 33 จุด
    รองรับทั้งแบบคงที่ (Standard EMA) และปรับตามความเร็ว (Adaptive EMA)
    """

    def __init__(
        self,
        alpha: float = 0.3,
        adaptive: bool = True,
        min_alpha: float = 0.1,
        max_alpha: float = 0.8,
        velocity_scale: float = 15.0,
    ) -> None:
        self.alpha = alpha                  # ค่าความสมูทพื้นฐาน (ยิ่งน้อยยิ่งสมูท แต่หน่วงขึ้น)
        self.adaptive = adaptive            # เปิด/ปิดโหมดปรับค่าตามความเร็วการขยับ
        self.min_alpha = min_alpha          # ค่า alpha ต่ำสุดขณะอยู่นิ่ง (เน้นนิ่งสนิท)
        self.max_alpha = max_alpha          # ค่า alpha สูงสุดขณะขยับเร็ว (เน้นตามทันทันที)
        self.velocity_scale = velocity_scale # ตัวคูณความไวต่อการเคลื่อนที่
        self.smoothed_landmarks: np.ndarray | None = None

    def apply(self, current_landmarks_list) -> np.ndarray:
        """แปลงข้อมูล Landmark ทั้ง 33 จุดเป็น NumPy Array ขนาด (33, 3) -> [x, y, z] แล้วทำ Smoothing"""
        current = np.array(
            [[lm.x, lm.y, lm.z] for lm in current_landmarks_list],
            dtype=np.float64,
        )

        # เฟรมแรกที่ตรวจจับเจอ ให้ใช้ค่าเริ่มต้นทันที
        if self.smoothed_landmarks is None or self.smoothed_landmarks.shape != current.shape:
            self.smoothed_landmarks = current
            return self.smoothed_landmarks

        if self.adaptive:
            # คำนวณระยะการขยับ (Velocity) ของแต่ละจุดเทียบกับเฟรมก่อนหน้า
            velocity = np.linalg.norm(current - self.smoothed_landmarks, axis=1, keepdims=True)

            # คำนวณค่า alpha แยกตามแต่ละจุด: ขยับน้อยใช้ min_alpha, ขยับเร็วพุ่งไปหา max_alpha
            dynamic_alpha = np.clip(
                self.min_alpha + (velocity * self.velocity_scale),
                self.min_alpha,
                self.max_alpha,
            )
            self.smoothed_landmarks = (
                (dynamic_alpha * current) + ((1.0 - dynamic_alpha) * self.smoothed_landmarks)
            )
        else:
            # สมการ Exponential Moving Average (EMA) แบบคงที่: S_t = alpha * Y_t + (1 - alpha) * S_{t-1}
            self.smoothed_landmarks = (
                (self.alpha * current) + ((1.0 - self.alpha) * self.smoothed_landmarks)
            )

        return self.smoothed_landmarks

    def reset(self) -> None:
        """รีเซ็ตค่าอ้างอิงเมื่อไม่พบตัวบุคคล"""
        self.smoothed_landmarks = None


def draw_smooth_stickman(canvas: np.ndarray, landmarks_array: np.ndarray) -> None:
    """ฟังก์ชันวาดตัวละคร Stickman ด้วยพิกัด NumPy Array (33, 3) ลงบนพื้นหลังสีขาว"""
    h, w, _ = canvas.shape

    def to_pixel(pt: np.ndarray) -> tuple[int, int]:
        return (int(pt[0] * w), int(pt[1] * h))

    # ดึงพิกัดจุดสำคัญตาม Index ของ MediaPipe Pose (ครบทั้งแขน ขา ลำตัว และศีรษะ)
    l_shoulder, r_shoulder = landmarks_array[11], landmarks_array[12]
    l_elbow, r_elbow = landmarks_array[13], landmarks_array[14]
    l_wrist, r_wrist = landmarks_array[15], landmarks_array[16]
    l_hip, r_hip = landmarks_array[23], landmarks_array[24]
    l_knee, r_knee = landmarks_array[25], landmarks_array[26]
    l_ankle, r_ankle = landmarks_array[27], landmarks_array[28]

    # คำนวณจุดกึ่งกลางคอและสะโพก
    neck = (l_shoulder + r_shoulder) / 2.0
    hip_center = (l_hip + r_hip) / 2.0

    l_shoulder_px = to_pixel(l_shoulder)
    r_shoulder_px = to_pixel(r_shoulder)
    l_hip_px = to_pixel(l_hip)
    r_hip_px = to_pixel(r_hip)
    neck_px = to_pixel(neck)
    hip_px = to_pixel(hip_center)

    # คำนวณขนาดและตำแหน่งหัวจากความกว้างไหล่
    shoulder_width = np.linalg.norm(
        np.array(l_shoulder_px, dtype=np.float64) - np.array(r_shoulder_px, dtype=np.float64)
    )
    head_radius = max(int(shoulder_width * 0.38), 16)
    head_center = (neck_px[0], max(head_radius + 10, neck_px[1] - head_radius - 12))

    body_color = (40, 40, 40)
    left_arm_color = (230, 110, 0)
    right_arm_color = (0, 110, 230)
    thickness = 6

    # 1. วาดเส้นบ่า สะโพก และแกนกลางลำตัว
    cv2.line(canvas, l_shoulder_px, r_shoulder_px, body_color, thickness)
    cv2.line(canvas, l_hip_px, r_hip_px, body_color, thickness)
    cv2.line(canvas, neck_px, hip_px, body_color, thickness)

    # 2. วาดแขนซ้ายและขวาจากข้อไหล่จริง
    cv2.line(canvas, l_shoulder_px, to_pixel(l_elbow), left_arm_color, thickness)
    cv2.line(canvas, to_pixel(l_elbow), to_pixel(l_wrist), left_arm_color, thickness)
    cv2.line(canvas, r_shoulder_px, to_pixel(r_elbow), right_arm_color, thickness)
    cv2.line(canvas, to_pixel(r_elbow), to_pixel(r_wrist), right_arm_color, thickness)

    # 3. วาดขาซ้ายและขวาจากสะโพก
    cv2.line(canvas, l_hip_px, to_pixel(l_knee), body_color, thickness)
    cv2.line(canvas, to_pixel(l_knee), to_pixel(l_ankle), body_color, thickness)
    cv2.line(canvas, r_hip_px, to_pixel(r_knee), body_color, thickness)
    cv2.line(canvas, to_pixel(r_knee), to_pixel(r_ankle), body_color, thickness)

    # 4. วาดจุดข้อต่อสำคัญ
    for joint in (
        l_shoulder,
        r_shoulder,
        l_elbow,
        r_elbow,
        l_wrist,
        r_wrist,
        l_hip,
        r_hip,
        l_knee,
        r_knee,
        l_ankle,
        r_ankle,
    ):
        cv2.circle(canvas, to_pixel(joint), 7, (16, 185, 129), -1)

    # 5. วาดศีรษะและเส้นคอ
    cv2.line(canvas, neck_px, (head_center[0], head_center[1] + head_radius), body_color, thickness)
    cv2.circle(canvas, head_center, head_radius, body_color, thickness)


def main() -> None:
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,          # เปิดฟิลเตอร์ชั้นแรกของ MediaPipe
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    smoother = LandmarkSmoother(alpha=0.25, adaptive=True)
    enable_custom_smooth = True

    cap = cv2.VideoCapture(0)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # พลิกภาพซ้าย-ขวาให้เหมือนส่องกระจก
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            # สร้างพื้นหลังสีขาว
            canvas = np.ones_like(frame) * 255

            if results.pose_landmarks:
                raw_landmarks = results.pose_landmarks.landmark
                if enable_custom_smooth:
                    landmarks_to_draw = smoother.apply(raw_landmarks)
                else:
                    landmarks_to_draw = np.array(
                        [[lm.x, lm.y, lm.z] for lm in raw_landmarks],
                        dtype=np.float64,
                    )

                draw_smooth_stickman(canvas, landmarks_to_draw)
            else:
                smoother.reset()

            # แสดงสถานะโหมด Smoothing และค่า alpha บนมุมจอ
            if not enable_custom_smooth:
                mode_text = f"OFF (Base alpha={smoother.alpha:.2f})"
                mode_color = (0, 0, 200)
            elif smoother.adaptive:
                mode_text = (
                    f"Adaptive EMA (min={smoother.min_alpha:.2f}, max={smoother.max_alpha:.2f}, "
                    f"alpha={smoother.alpha:.2f})"
                )
                mode_color = (0, 120, 0)
            else:
                mode_text = f"Fixed EMA (alpha={smoother.alpha:.2f})"
                mode_color = (180, 100, 0)

            cv2.putText(
                canvas,
                f"Smoothing: {mode_text}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                mode_color,
                2,
            )
            cv2.putText(
                canvas,
                "Keys: [1] On/Off | [2] EMA/Adaptive | [+/-] Alpha | [q] Quit",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (90, 90, 90),
                1,
            )

            cv2.imshow("Smooth Stickman Pose", canvas)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("1"):
                enable_custom_smooth = not enable_custom_smooth
                if not enable_custom_smooth:
                    smoother.reset()
            elif key == ord("2"):
                smoother.adaptive = not smoother.adaptive
            elif key in (ord("+"), ord("=")):
                smoother.alpha = min(0.95, round(smoother.alpha + 0.05, 2))
            elif key in (ord("-"), ord("_")):
                smoother.alpha = max(0.05, round(smoother.alpha - 0.05, 2))
    finally:
        pose.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

