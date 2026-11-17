"""Pydantic v2 telemetry, calibration, biomechanics, and Face ID 120 schemas."""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple
from pydantic import BaseModel, ConfigDict, Field


class Keypoint3D(BaseModel):
    """Single 3D keypoint in the 133-point Wholebody3d topology."""

    index: int
    name: str = ""
    x: float
    y: float
    z: float = 0.0
    score: float = 1.0


class BoneLengths3D(BaseModel):
    """Calibrated anatomical segment lengths captured during 2-second standing calibration."""

    upper_arm_len: float
    forearm_len: float
    thigh_len: float
    shin_len: float
    shoulder_width: float
    torso_height: float
    cm_per_unit: float
    spine_vertical_unit: Tuple[float, float, float]
    baseline_shoulder_y: float
    baseline_heel_y: float
    baseline_hip_y: float = 0.54
    baseline_shoulder_tilt_deg: float = 0.0
    baseline_pelvic_tilt_deg: float = 0.0
    resting_shoulder_delta_y: float = 0.0
    resting_spine_angle_deg: float = 0.0
    arm_to_torso_ratio: float = 0.0
    leg_to_torso_ratio: float = 0.0
    estimated_distance_m: float = 2.0


class CalibrationStateSchema(BaseModel):
    """2-Second Standing Calibration live state."""

    is_calibrated: bool = False
    progress: float = 0.0
    remaining_sec: float = 2.0
    message_th: str = "กรุณายืนตรงหันหน้าเข้ากล้อง 2 วินาทีเพื่อคาลิเบรตสัดส่วนร่างกาย"
    distance_status: Literal["OK", "TOO_CLOSE", "TOO_FAR", "OUT_OF_FRAME", "OCCLUDED"] = "OK"
    estimated_distance_m: float = 2.0
    min_visibility: float = 1.0
    missing_joints: List[str] = Field(default_factory=list)
    bone_lengths: Optional[BoneLengths3D] = None


class HandRaiseTelemetry(BaseModel):
    """Real-time 133-keypoint hand raise & PhysicalCardGame LEFT/RIGHT selection telemetry."""

    left_raised: bool = False
    right_raised: bool = False
    active_card_zone: Literal["LEFT", "RIGHT", "BOTH", "NONE"] = "NONE"
    left_palm_2d: Optional[Tuple[float, float]] = None
    right_palm_2d: Optional[Tuple[float, float]] = None
    left_hand_landmarks: List[Keypoint3D] = Field(default_factory=list)
    right_hand_landmarks: List[Keypoint3D] = Field(default_factory=list)


class FaceEnrollmentStateSchema(BaseModel):
    """Live InsightFace 120-frame 5-angle enrollment telemetry."""

    active: bool = False
    session_id: Optional[str] = None
    phase: Literal["idle", "scanning", "computing", "completed"] = "idle"
    frames_captured: int = 0
    target_frames: int = 120
    bucket_counts: Dict[str, int] = Field(
        default_factory=lambda: {"center": 0, "left": 0, "right": 0, "up": 0, "down": 0}
    )
    bucket_targets: Dict[str, int] = Field(
        default_factory=lambda: {"center": 40, "left": 20, "right": 20, "up": 20, "down": 20}
    )
    current_pose: str = "center"
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    sharpness: float = 0.0
    face_detected: bool = False
    guidance_th: str = "กรุณาวางใบหน้าให้อยู่ในกรอบวงกลม"


MovementPhase = Literal[
    "IDLE",
    "READY",
    "CONCENTRIC",
    "HOLD",
    "ECCENTRIC",
    "REP_COMPLETE",
]


class TelemetryAnalysisSchema(BaseModel):
    """Biomechanical joint angle and anti-cheating analysis metrics."""

    current_rom_deg: float = 0.0
    target_rom_deg: float = 150.0
    rom_completion_pct: float = 0.0
    trunk_tilt_deg: float = 0.0
    torso_plane_tilt_deg: float = 0.0
    shoulder_tilt_deg: float = 0.0
    pelvic_tilt_deg: float = 0.0
    shoulder_hike_cm: float = 0.0
    elbow_flexion_deg: float = 180.0
    knee_valgus_deg: float = 0.0
    heel_lift_cm: float = 0.0
    angular_velocity_deg_s: float = 0.0
    min_landmark_visibility: float = 1.0
    visibility_ok: bool = True
    stability_score: float = 100.0
    hold_progress_pct: float = 0.0
    hold_timer_frozen: bool = False
    is_form_valid: bool = True
    compensation_detected: List[str] = Field(default_factory=lambda: ["NONE"])
    compensation_warnings: List[str] = Field(default_factory=list)
    pain_alert: bool = False
    safety_guard_alert: Optional[str] = None
    safety_guard_banner_th: Optional[str] = None
    spasticity_warning: bool = False
    contraindication_clamped: bool = False
    consecutive_compensations: int = 0
    fatigue_rest_recommended: bool = False
    fatigue_rest_remaining_sec: float = 0.0
    suggested_reduced_rom_deg: Optional[float] = None


class RepCounterSchema(BaseModel):
    """Finite state machine repetition counters and hold timer."""

    valid_reps: int = 0
    invalid_reps: int = 0
    hold_remaining_sec: float = 3.0
    hold_progress_pct: float = 0.0
    hold_timer_frozen: bool = False
    movement_phase: MovementPhase = "IDLE"
    target_reps: int = 10


class UIDirectivesSchema(BaseModel):
    """Directives for Kiosk UI skeleton highlight and ROM fan."""

    skeleton_highlight_joint: str = "right_shoulder"
    highlight_color: Literal["GREEN", "AMBER", "RED"] = "GREEN"
    show_target_arc: bool = True


class PatientMatchSchema(BaseModel):
    """Patient identity resolved in < 100ms via InsightFace 1:N cosine matching."""

    id: Optional[str] = None
    hn: Optional[str] = None
    name: Optional[str] = None
    similarity: float = 0.0
    latency_ms: float = 0.0


class ExercisePrescriptionSchema(BaseModel):
    """Active exercise prescription payload sent from frontend or REST."""

    model_config = ConfigDict(extra="allow")

    exercise_id: str = "STROKE_SH_ABD"
    exercise_name: str = "[Stroke ADL] กางแขนด้านข้าง (ฝึกกางแขนสวมเสื้อ)"
    target_joint: str = "right_shoulder"
    prescribed_side: Optional[str] = None
    target_rom_deg: float = 150.0
    hold_sec: float = 3.0
    target_reps: int = 10
    max_safe_rom_deg: Optional[float] = None
    affected_side: Optional[str] = None
    stroke_type: Optional[str] = None
    motor_assessment_score: Optional[int] = None


class IdentityDetectedBoxSchema(BaseModel):
    """Bounding box for every person/face detected in frame (1280x720 coordinate space)."""

    bbox: List[int]
    label: str
    is_authorized: bool = True


class IdentityGuardSchema(BaseModel):
    """Continuous anti-cheat & patient identity lock telemetry emitted every frame."""

    status: Literal["VERIFIED", "MULTI_PERSON", "IMPOSTOR", "FACE_NOT_VISIBLE"] = "VERIFIED"
    person_count: int = 1
    face_count: int = 1
    match_confidence: float = 0.96
    is_counting_allowed: bool = True
    alert_title_th: str = ""
    alert_subtitle_th: str = ""
    detected_boxes: List[IdentityDetectedBoxSchema] = Field(default_factory=list)


class EdgeKioskTelemetryPayload(BaseModel):
    """Complete 30-60 Hz WebSocket payload streamed to React DRTECHCARES frontend."""

    kiosk_state: Literal[
        "AUTH_FRAMING",
        "CALIBRATING",
        "ACTIVE_REP",
        "HOLD_PHASE",
        "REP_COMPLETED",
        "SET_FINISHED",
        "EMERGENCY_HALT",
        "ACTIVE",
        "HOLD",
        "REP_DONE",
        "PAUSED_MULTI_PERSON",
        "PAUSED_IMPOSTOR",
        "PAUSED_ANTI_CHEAT",
        "EMERGENCY",
    ]
    fsm_state: Literal[
        "FACE_AUTH",
        "CALIBRATE_2S",
        "ACTIVE",
        "HOLD",
        "REP_DONE",
        "SET_FINISHED",
        "PAUSED_MULTI_PERSON",
        "PAUSED_IMPOSTOR",
        "PAUSED_ANTI_CHEAT",
        "EMERGENCY",
    ]
    timestamp_ms: int
    fps: float
    engine_backend: str = "MediaPipe Heavy / RTMW3D-l + InsightFace + OneEuroFilter"
    camera_mode: str = "live_webcam"
    movement_phase: MovementPhase = "IDLE"
    hold_progress_pct: float = 0.0
    stability_score: float = 100.0
    compensation_warnings: List[str] = Field(default_factory=list)
    patient: Optional[PatientMatchSchema] = None
    identity_guard: IdentityGuardSchema = Field(default_factory=IdentityGuardSchema)
    telemetry_analysis: TelemetryAnalysisSchema
    rep_counter: RepCounterSchema
    ui_directives: UIDirectivesSchema
    tts_voice_th: str
    calibration: CalibrationStateSchema
    hand_status: HandRaiseTelemetry
    face_enrollment: Optional[FaceEnrollmentStateSchema] = None
    keypoints_3d: List[Keypoint3D] = Field(default_factory=list)
    pose_landmarks: List[Keypoint3D] = Field(default_factory=list)
    pose_world_landmarks: List[Keypoint3D] = Field(default_factory=list)
    frame_base64: Optional[str] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None


class EnrollStartRequest(BaseModel):
    patient_id: Optional[str] = None
    hn: Optional[str] = None
    full_name: Optional[str] = None
    use_edge_camera: bool = True


class EnrollFrameRequest(BaseModel):
    session_id: str
    image_base64: Optional[str] = None
    forced_bucket: Optional[Literal["center", "left", "right", "up", "down"]] = None


class EnrollFinalizeRequest(BaseModel):
    session_id: str
    patient_id: Optional[str] = None
    hn: Optional[str] = None
    persist_to_supabase: bool = True


class FaceID120Response(BaseModel):
    session_id: str
    master_embedding: List[float]
    angle_embeddings: Dict[str, List[float]]
    frames_captured: int
    quality_score: float
    avatar_data_url: str = ""
    bucket_counts: Dict[str, int]
    supabase_synced: bool = False


class VerifyFaceRequest(BaseModel):
    image_base64: Optional[str] = None
    use_edge_camera: bool = True


class VerifyFaceResponse(BaseModel):
    matched: bool
    patient: Optional[PatientMatchSchema] = None
    similarity: float = 0.0
    latency_ms: float = 0.0
    candidate_count: int = 0

