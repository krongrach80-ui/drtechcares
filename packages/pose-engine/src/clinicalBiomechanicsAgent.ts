import { POSE_INDEX } from './armJointProcessor';
import type { ExerciseConfig, PoseLandmark } from './types';

export type KioskBiomechanicsState =
  | 'AUTH_FRAMING'
  | 'CALIBRATING'
  | 'ACTIVE_REP'
  | 'HOLD_PHASE'
  | 'REP_COMPLETED'
  | 'SET_FINISHED'
  | 'EMERGENCY_HALT';

export type CompensationCode =
  | 'NONE'
  | 'TRUNK_LEAN'
  | 'SHOULDER_HIKE'
  | 'KNEE_VALGUS'
  | 'TOO_FAST'
  | 'INCOMPLETE_ROM';

export type HighlightJoint =
  | 'none'
  | 'left_shoulder'
  | 'right_shoulder'
  | 'spine'
  | 'left_knee'
  | 'right_knee';

export type HighlightColor = 'GREEN' | 'AMBER' | 'RED';

export interface ClinicalTelemetryPacket {
  auth_status: {
    verified: boolean;
    patient_id?: string;
    confidence: number;
    liveness: boolean;
  };
  spatial_framing: {
    distance_m: number;
    full_body_visible: boolean;
    feet_visible?: boolean;
    lighting_score?: number;
  };
  exercise_prescription: {
    exercise_id?: string;
    exercise_name?: string;
    target_joint?: 'left_shoulder' | 'right_shoulder' | 'left_knee' | 'right_knee' | 'spine' | 'hip';
    target_rom_deg: number;
    hold_sec: number;
    target_reps: number;
    max_safe_rom_deg?: number;
    max_safe_velocity_deg_per_sec?: number;
  };
  live_telemetry: {
    primary_angle_deg: number;
    trunk_tilt_deg: number;
    contralateral_shoulder_elevation_cm?: number;
    shoulder_elevation_cm?: number;
    knee_valgus_deg?: number;
    heel_lifted?: boolean;
    pelvic_tilt_deg?: number;
    angular_velocity_deg_per_sec?: number;
    angular_velocity_deg_s?: number;
    hold_elapsed_sec?: number;
    pain_score?: number;
    facial_pain_detected?: boolean;
    rapid_vertical_drop_detected?: boolean;
    dt_sec?: number;
  };
}

export interface ClinicalBiomechanicsDecision {
  kiosk_state: KioskBiomechanicsState;
  telemetry_analysis: {
    current_rom_deg: number;
    target_rom_deg: number;
    rom_completion_pct: number;
    is_form_valid: boolean;
    compensation_detected: CompensationCode[];
    pain_alert: boolean;
  };
  rep_counter: {
    valid_reps: number;
    invalid_reps: number;
    hold_remaining_sec: number;
  };
  ui_directives: {
    skeleton_highlight_joint: HighlightJoint;
    highlight_color: HighlightColor;
    show_target_arc: boolean;
  };
  tts_voice_th: string;
}

export interface BiomechanicsCounterState {
  valid_reps: number;
  invalid_reps: number;
  hold_elapsed_sec: number;
  awaiting_return_to_rest?: boolean;
  rep_invalidated_in_cycle?: boolean;
  reached_attempt_threshold?: boolean;
}

type CanonicalTargetJoint = 'left_shoulder' | 'right_shoulder' | 'left_knee' | 'right_knee' | 'spine' | 'hip';

export function inferTargetJointFromExerciseName(exerciseName?: string): CanonicalTargetJoint {
  const n = (exerciseName ?? '').toLowerCase();
  if (n.includes('left') && (n.includes('knee') || n.includes('ankle') || n.includes('squat'))) {
    return 'left_knee';
  }
  if (n.includes('เข่าซ้าย')) return 'left_knee';
  if (n.includes('right') && (n.includes('knee') || n.includes('ankle') || n.includes('squat'))) {
    return 'right_knee';
  }
  if (n.includes('เข่าขวา') || n.includes('squat') || n.includes('ลุกนั่ง')) return 'right_knee';
  if (n.includes('left') || n.includes('ซ้าย')) return 'left_shoulder';
  if (n.includes('spine') || n.includes('trunk') || n.includes('หลัง') || n.includes('เอียงตัว')) {
    return 'spine';
  }
  return 'right_shoulder';
}

function resolveHighlightJoint(targetJoint?: CanonicalTargetJoint): HighlightJoint {
  if (!targetJoint) return 'right_shoulder';
  if (targetJoint === 'hip') return 'spine';
  return targetJoint;
}

/**
 * Stateless/pure evaluator step matching the Clinical Biomechanics & Physical Therapy AI Agent protocol.
 * Supports both internal telemetry packets and external RTMPose / ZED SDK JSON payloads.
 * Priority: 1. SAFETY_OVERRIDE -> 2. AUTH_AND_FRAMING -> 3. BIOMECHANICAL_EVALUATION & ANTI-CHEATING.
 */
export function evaluateClinicalBiomechanicsPacket(
  packet: ClinicalTelemetryPacket,
  prevState: BiomechanicsCounterState = { valid_reps: 0, invalid_reps: 0, hold_elapsed_sec: 0 },
): { decision: ClinicalBiomechanicsDecision; nextState: BiomechanicsCounterState } {
  const { auth_status, spatial_framing, exercise_prescription, live_telemetry } = packet;
  const dtSec = Math.max(0.01, live_telemetry.dt_sec ?? 0.25);
  const targetRom = Math.max(10, exercise_prescription.target_rom_deg);
  const maxSafeRom = exercise_prescription.max_safe_rom_deg ?? Math.min(185, targetRom + 25);
  const maxSafeVelocity = exercise_prescription.max_safe_velocity_deg_per_sec ?? 95;
  const romPct = Math.max(0, Math.min(120, Math.round((live_telemetry.primary_angle_deg / targetRom) * 100)));
  const inferredJoint =
    exercise_prescription.target_joint ?? inferTargetJointFromExerciseName(exercise_prescription.exercise_name);
  const primaryJoint = resolveHighlightJoint(inferredJoint);

  const shoulderElevationCm =
    live_telemetry.contralateral_shoulder_elevation_cm ?? live_telemetry.shoulder_elevation_cm ?? 0;
  const angularVelocityDegPerSec =
    live_telemetry.angular_velocity_deg_per_sec ?? live_telemetry.angular_velocity_deg_s ?? 0;
  const kneeValgusDeg = live_telemetry.knee_valgus_deg ?? 0;
  const painScore = live_telemetry.pain_score ?? 0;
  const facialPain = Boolean(live_telemetry.facial_pain_detected) || painScore >= 0.7;

  const state: BiomechanicsCounterState = {
    valid_reps: prevState.valid_reps,
    invalid_reps: prevState.invalid_reps,
    hold_elapsed_sec: live_telemetry.hold_elapsed_sec ?? prevState.hold_elapsed_sec,
    awaiting_return_to_rest: prevState.awaiting_return_to_rest ?? false,
    rep_invalidated_in_cycle: prevState.rep_invalidated_in_cycle ?? false,
    reached_attempt_threshold: prevState.reached_attempt_threshold ?? false,
  };

  // 1. SAFETY_OVERRIDE (Highest Priority - Evaluated Every Frame)
  const hyperextended = live_telemetry.primary_angle_deg > maxSafeRom;
  const painStop = painScore >= 0.7 || (facialPain && Math.abs(angularVelocityDegPerSec) <= 10);

  if (live_telemetry.rapid_vertical_drop_detected || hyperextended || painStop) {
    return {
      decision: {
        kiosk_state: 'EMERGENCY_HALT',
        telemetry_analysis: {
          current_rom_deg: Number(live_telemetry.primary_angle_deg.toFixed(1)),
          target_rom_deg: targetRom,
          rom_completion_pct: romPct,
          is_form_valid: false,
          compensation_detected: ['NONE'],
          pain_alert: true,
        },
        rep_counter: {
          valid_reps: state.valid_reps,
          invalid_reps: state.invalid_reps,
          hold_remaining_sec: Number(Math.max(0, exercise_prescription.hold_sec - state.hold_elapsed_sec).toFixed(1)),
        },
        ui_directives: {
          skeleton_highlight_joint: primaryJoint,
          highlight_color: 'RED',
          show_target_arc: false,
        },
        tts_voice_th: live_telemetry.rapid_vertical_drop_detected
          ? 'หยุดพักก่อนนะครับ เจ้าหน้าที่กำลังเข้าไปดูแลครับ'
          : 'หยุดพักทันทีนะครับ รู้สึกเจ็บตรงไหนไหมครับ',
      },
      nextState: { ...state, hold_elapsed_sec: 0 },
    };
  }

  // 2. AUTH_AND_FRAMING (Pre-Exercise Setup)
  if (!auth_status.verified || auth_status.confidence < 0.85 || !auth_status.liveness) {
    return {
      decision: {
        kiosk_state: 'AUTH_FRAMING',
        telemetry_analysis: {
          current_rom_deg: Number(live_telemetry.primary_angle_deg.toFixed(1)),
          target_rom_deg: targetRom,
          rom_completion_pct: romPct,
          is_form_valid: false,
          compensation_detected: ['NONE'],
          pain_alert: false,
        },
        rep_counter: {
          valid_reps: state.valid_reps,
          invalid_reps: state.invalid_reps,
          hold_remaining_sec: exercise_prescription.hold_sec,
        },
        ui_directives: {
          skeleton_highlight_joint: 'none',
          highlight_color: 'AMBER',
          show_target_arc: false,
        },
        tts_voice_th: 'กรุณาสแกนใบหน้ายืนยันตัวตนก่อนเริ่มฝึกครับ',
      },
      nextState: state,
    };
  }

  const distanceOk = spatial_framing.distance_m >= 1.8 && spatial_framing.distance_m <= 2.2;
  const framingOk = spatial_framing.full_body_visible && spatial_framing.feet_visible !== false;

  if (!distanceOk || !framingOk) {
    let framingVoice = 'ขยับให้เห็นทั้งตัวและเท้าในกรอบครับ';
    if (!distanceOk && spatial_framing.distance_m < 1.8) {
      framingVoice = 'ถอยหลังอีกนิด ให้อยู่ระยะ 2 เมตรครับ';
    } else if (!distanceOk && spatial_framing.distance_m > 2.2) {
      framingVoice = 'เดินเข้ามาอีกนิด ให้อยู่ระยะ 2 เมตรครับ';
    }
    return {
      decision: {
        kiosk_state: 'AUTH_FRAMING',
        telemetry_analysis: {
          current_rom_deg: Number(live_telemetry.primary_angle_deg.toFixed(1)),
          target_rom_deg: targetRom,
          rom_completion_pct: romPct,
          is_form_valid: false,
          compensation_detected: ['NONE'],
          pain_alert: false,
        },
        rep_counter: {
          valid_reps: state.valid_reps,
          invalid_reps: state.invalid_reps,
          hold_remaining_sec: exercise_prescription.hold_sec,
        },
        ui_directives: {
          skeleton_highlight_joint: 'spine',
          highlight_color: 'AMBER',
          show_target_arc: false,
        },
        tts_voice_th: framingVoice,
      },
      nextState: state,
    };
  }

  // If all target reps already finished
  if (state.valid_reps >= exercise_prescription.target_reps) {
    return {
      decision: {
        kiosk_state: 'SET_FINISHED',
        telemetry_analysis: {
          current_rom_deg: Number(live_telemetry.primary_angle_deg.toFixed(1)),
          target_rom_deg: targetRom,
          rom_completion_pct: romPct,
          is_form_valid: true,
          compensation_detected: ['NONE'],
          pain_alert: false,
        },
        rep_counter: {
          valid_reps: state.valid_reps,
          invalid_reps: state.invalid_reps,
          hold_remaining_sec: 0,
        },
        ui_directives: {
          skeleton_highlight_joint: primaryJoint,
          highlight_color: 'GREEN',
          show_target_arc: false,
        },
        tts_voice_th: 'เก่งมากครับ ฝึกครบเซ็ตเรียบร้อยแล้ว',
      },
      nextState: state,
    };
  }

  // 3. BIOMECHANICAL_EVALUATION & ANTI-CHEATING
  const compensations: CompensationCode[] = [];
  let highlightJoint: HighlightJoint = primaryJoint;
  let primaryCorrectionVoice = '';

  if (Math.abs(live_telemetry.trunk_tilt_deg) > 10) {
    compensations.push('TRUNK_LEAN');
    highlightJoint = 'spine';
    primaryCorrectionVoice = 'ยืดตัวตรงครับ อย่าเอียงตัวช่วยยก';
  }
  if (shoulderElevationCm > 3) {
    compensations.push('SHOULDER_HIKE');
    if (!primaryCorrectionVoice) {
      highlightJoint = inferredJoint === 'right_shoulder' ? 'right_shoulder' : 'left_shoulder';
      primaryCorrectionVoice = 'ผ่อนไหล่ลงครับ อย่ายักไหล่ขึ้น';
    }
  }
  if (Math.abs(kneeValgusDeg) > 8) {
    compensations.push('KNEE_VALGUS');
    if (!primaryCorrectionVoice) {
      highlightJoint = inferredJoint === 'right_knee' ? 'right_knee' : 'left_knee';
      primaryCorrectionVoice = 'ระวังเข่าบิดเข้าด้านในครับ';
    }
  }
  if (Math.abs(angularVelocityDegPerSec) > maxSafeVelocity) {
    compensations.push('TOO_FAST');
    if (!primaryCorrectionVoice) {
      primaryCorrectionVoice = 'ค่อยๆ เคลื่อนไหวช้าๆ อย่างควบคุมครับ';
    }
  }
  if (live_telemetry.heel_lifted && !primaryCorrectionVoice) {
    compensations.push('KNEE_VALGUS');
    primaryCorrectionVoice = 'กดส้นเท้าติดพื้นไว้ครับ';
  }
  if ((live_telemetry.pelvic_tilt_deg ?? 0) > 8 && !primaryCorrectionVoice) {
    compensations.push('TRUNK_LEAN');
    highlightJoint = 'spine';
    primaryCorrectionVoice = 'รักษาระดับสะโพกให้ตรงครับ';
  }

  const restThreshold = targetRom * 0.35;
  const attemptThreshold = targetRom * 0.5;
  const romGate = targetRom * 0.95;
  const isLowerBody = primaryJoint.includes('knee') || primaryJoint.includes('hip');
  const lowerLimbInstructionTh = isLowerBody ? 'ค่อยๆ เอาขาลงช้าๆ' : 'ค่อยๆ เอาแขนลงช้าๆ';
  const holdDurationLabel = Number(exercise_prescription.hold_sec.toFixed(1).replace(/\.0$/, ''));

  if (
    compensations.length > 0 &&
    live_telemetry.primary_angle_deg >= targetRom * 0.8 &&
    live_telemetry.primary_angle_deg < romGate
  ) {
    compensations.push('INCOMPLETE_ROM');
  }

  // Handle waiting for return to start after completing or invalidating a rep
  if (state.awaiting_return_to_rest) {
    if (live_telemetry.primary_angle_deg <= restThreshold) {
      state.awaiting_return_to_rest = false;
      state.rep_invalidated_in_cycle = false;
      state.reached_attempt_threshold = false;
      state.hold_elapsed_sec = 0;
    } else {
      return {
        decision: {
          kiosk_state: 'REP_COMPLETED',
          telemetry_analysis: {
            current_rom_deg: Number(live_telemetry.primary_angle_deg.toFixed(1)),
            target_rom_deg: targetRom,
            rom_completion_pct: romPct,
            is_form_valid: compensations.length === 0,
            compensation_detected: compensations.length > 0 ? compensations : ['NONE'],
            pain_alert: false,
          },
          rep_counter: {
            valid_reps: state.valid_reps,
            invalid_reps: state.invalid_reps,
            hold_remaining_sec: 0,
          },
          ui_directives: {
            skeleton_highlight_joint: primaryJoint,
            highlight_color: 'GREEN',
            show_target_arc: true,
          },
          tts_voice_th: `ครบ ${holdDurationLabel} วินาทีแล้ว ${lowerLimbInstructionTh} ครบ ${state.valid_reps} ครั้งแล้วครับ`,
        },
        nextState: state,
      };
    }
  }

  if (live_telemetry.primary_angle_deg >= attemptThreshold) {
    state.reached_attempt_threshold = true;
  }

  // Anti-cheating rejection
  if (compensations.length > 0) {
    if (!state.rep_invalidated_in_cycle && live_telemetry.primary_angle_deg >= attemptThreshold) {
      state.invalid_reps += 1;
      state.rep_invalidated_in_cycle = true;
    }
    return {
      decision: {
        kiosk_state: 'ACTIVE_REP',
        telemetry_analysis: {
          current_rom_deg: Number(live_telemetry.primary_angle_deg.toFixed(1)),
          target_rom_deg: targetRom,
          rom_completion_pct: romPct,
          is_form_valid: false,
          compensation_detected: compensations,
          pain_alert: false,
        },
        rep_counter: {
          valid_reps: state.valid_reps,
          invalid_reps: state.invalid_reps,
          hold_remaining_sec: Number(
            Math.max(0, exercise_prescription.hold_sec - state.hold_elapsed_sec).toFixed(1),
          ),
        },
        ui_directives: {
          skeleton_highlight_joint: highlightJoint,
          highlight_color: 'AMBER',
          show_target_arc: true,
        },
        tts_voice_th: primaryCorrectionVoice,
      },
      nextState: state,
    };
  }

  // ROM Gate + Hold Phase Evaluation (tolerance ±0.2s)
  if (live_telemetry.primary_angle_deg >= romGate) {
    state.hold_elapsed_sec += dtSec;
    const holdToleranceTarget = Math.max(0, exercise_prescription.hold_sec - 0.2);
    const holdRemaining = Math.max(0, exercise_prescription.hold_sec - state.hold_elapsed_sec);

    if (state.hold_elapsed_sec >= holdToleranceTarget) {
      state.valid_reps += 1;
      state.hold_elapsed_sec = 0;
      state.awaiting_return_to_rest = true;
      state.reached_attempt_threshold = false;
      const finished = state.valid_reps >= exercise_prescription.target_reps;
      return {
        decision: {
          kiosk_state: finished ? 'SET_FINISHED' : 'REP_COMPLETED',
          telemetry_analysis: {
            current_rom_deg: Number(live_telemetry.primary_angle_deg.toFixed(1)),
            target_rom_deg: targetRom,
            rom_completion_pct: romPct,
            is_form_valid: true,
            compensation_detected: ['NONE'],
            pain_alert: false,
          },
          rep_counter: {
            valid_reps: state.valid_reps,
            invalid_reps: state.invalid_reps,
            hold_remaining_sec: 0,
          },
          ui_directives: {
            skeleton_highlight_joint: primaryJoint,
            highlight_color: 'GREEN',
            show_target_arc: !finished,
          },
          tts_voice_th: finished
            ? `ครบ ${holdDurationLabel} วินาทีแล้ว เยี่ยมมากครับ ครบ ${state.valid_reps} ครั้ง จบเซ็ตแล้ว ${lowerLimbInstructionTh} ครับ`
            : `ครบ ${holdDurationLabel} วินาทีแล้ว ดีมากครับ ครบ ${state.valid_reps} ครั้ง ${lowerLimbInstructionTh} ครับ`,
        },
        nextState: state,
      };
    }

    return {
      decision: {
        kiosk_state: 'HOLD_PHASE',
        telemetry_analysis: {
          current_rom_deg: Number(live_telemetry.primary_angle_deg.toFixed(1)),
          target_rom_deg: targetRom,
          rom_completion_pct: romPct,
          is_form_valid: true,
          compensation_detected: ['NONE'],
          pain_alert: false,
        },
        rep_counter: {
          valid_reps: state.valid_reps,
          invalid_reps: state.invalid_reps,
          hold_remaining_sec: Number(holdRemaining.toFixed(1)),
        },
        ui_directives: {
          skeleton_highlight_joint: primaryJoint,
          highlight_color: 'GREEN',
          show_target_arc: true,
        },
        tts_voice_th: `ดีมากครับ ค้างไว้ ${Math.ceil(holdRemaining)} วินาที`,
      },
      nextState: state,
    };
  }

  // Incomplete ROM detection if patient dropped back before completing hold
  if (state.reached_attempt_threshold && live_telemetry.primary_angle_deg <= restThreshold) {
    state.reached_attempt_threshold = false;
    state.hold_elapsed_sec = 0;
    state.invalid_reps += 1;
    return {
      decision: {
        kiosk_state: 'ACTIVE_REP',
        telemetry_analysis: {
          current_rom_deg: Number(live_telemetry.primary_angle_deg.toFixed(1)),
          target_rom_deg: targetRom,
          rom_completion_pct: romPct,
          is_form_valid: false,
          compensation_detected: ['INCOMPLETE_ROM'],
          pain_alert: false,
        },
        rep_counter: {
          valid_reps: state.valid_reps,
          invalid_reps: state.invalid_reps,
          hold_remaining_sec: exercise_prescription.hold_sec,
        },
        ui_directives: {
          skeleton_highlight_joint: primaryJoint,
          highlight_color: 'AMBER',
          show_target_arc: true,
        },
        tts_voice_th: 'ยกให้สุดมุมแล้วค้างไว้สักครู่นะครับ',
      },
      nextState: state,
    };
  }

  state.hold_elapsed_sec = 0;
  return {
    decision: {
      kiosk_state: 'ACTIVE_REP',
      telemetry_analysis: {
        current_rom_deg: Number(live_telemetry.primary_angle_deg.toFixed(1)),
        target_rom_deg: targetRom,
        rom_completion_pct: romPct,
        is_form_valid: true,
        compensation_detected: ['NONE'],
        pain_alert: false,
      },
      rep_counter: {
        valid_reps: state.valid_reps,
        invalid_reps: state.invalid_reps,
        hold_remaining_sec: exercise_prescription.hold_sec,
      },
      ui_directives: {
        skeleton_highlight_joint: primaryJoint,
        highlight_color: 'GREEN',
        show_target_arc: true,
      },
      tts_voice_th: 'ค่อยๆ ยกขึ้นจนถึงเส้นเป้าหมายครับ',
    },
    nextState: state,
  };
}

/**
 * Derives a ClinicalTelemetryPacket from live 3D landmarks for real-time evaluation.
 */
export function buildTelemetryPacketFromLandmarks(params: {
  landmarks: readonly PoseLandmark[];
  prevLandmarks?: readonly PoseLandmark[] | null;
  exercise: ExerciseConfig;
  primaryAngleDeg: number;
  prevAngleDeg?: number | null;
  dtSec?: number;
  authConfidence?: number;
}): ClinicalTelemetryPacket {
  const { landmarks, prevLandmarks, exercise, primaryAngleDeg, prevAngleDeg, dtSec = 0.25 } = params;
  const lSh = landmarks[POSE_INDEX.LEFT_SHOULDER];
  const rSh = landmarks[POSE_INDEX.RIGHT_SHOULDER];
  const lHip = landmarks[POSE_INDEX.LEFT_HIP];
  const rHip = landmarks[POSE_INDEX.RIGHT_HIP];
  const nose = landmarks[POSE_INDEX.NOSE];
  const lAnkle = landmarks[POSE_INDEX.LEFT_ANKLE];
  const rAnkle = landmarks[POSE_INDEX.RIGHT_ANKLE];
  const lKnee = landmarks[POSE_INDEX.LEFT_KNEE];
  const rKnee = landmarks[POSE_INDEX.RIGHT_KNEE];

  let trunkTiltDeg = 0;
  let shoulderElevCm = 0;
  if (lSh && rSh && lHip && rHip) {
    const midShX = (lSh.x + rSh.x) / 2;
    const midShY = (lSh.y + rSh.y) / 2;
    const midHipX = (lHip.x + rHip.x) / 2;
    const midHipY = (lHip.y + rHip.y) / 2;
    trunkTiltDeg =
      (Math.atan2(midShX - midHipX, Math.max(0.05, midHipY - midShY)) * 180) / Math.PI;
    // Approximate cm from normalized body height (~165cm full body)
    shoulderElevCm = Math.abs(lSh.y - rSh.y) * 140;
  }

  let kneeValgusDeg = 0;
  if (lHip && lKnee && lAnkle) {
    kneeValgusDeg = Math.abs((lKnee.x - (lHip.x + lAnkle.x) / 2) * 90);
  }
  if (rHip && rKnee && rAnkle) {
    const rValgus = Math.abs((rKnee.x - (rHip.x + rAnkle.x) / 2) * 90);
    kneeValgusDeg = Math.max(kneeValgusDeg, rValgus);
  }

  let rapidDrop = false;
  if (prevLandmarks && nose && lHip && rHip) {
    const prevHip = prevLandmarks[POSE_INDEX.LEFT_HIP];
    const currHipY = (lHip.y + rHip.y) / 2;
    if (prevHip && currHipY - prevHip.y > 0.22) {
      rapidDrop = true;
    }
  }

  // Approximate distance & framing from body span in normalized camera coordinates
  const feetVisible = Boolean(
    lAnkle &&
      rAnkle &&
      (lAnkle.visibility ?? 0) >= 0.45 &&
      (rAnkle.visibility ?? 0) >= 0.45 &&
      lAnkle.y <= 0.98 &&
      rAnkle.y <= 0.98,
  );
  const fullBodyVisible = Boolean(
    nose && lSh && rSh && lHip && rHip && (nose.visibility ?? 0) >= 0.5 && (lHip.visibility ?? 0) >= 0.5,
  );
  const bodyHeightNorm = nose && lAnkle ? Math.max(0.2, Math.abs(lAnkle.y - nose.y)) : 0.72;
  const estDistanceM = Number((1.45 / bodyHeightNorm).toFixed(2));

  const targetJoint: ClinicalTelemetryPacket['exercise_prescription']['target_joint'] =
    exercise.detection.jointGroup === 'knee' || exercise.detection.jointGroup === 'ankle'
      ? exercise.detection.side === 'left'
        ? 'left_knee'
        : 'right_knee'
      : exercise.detection.jointGroup === 'trunk'
        ? 'spine'
        : exercise.detection.side === 'left'
          ? 'left_shoulder'
          : 'right_shoulder';

  const angularVelocity =
    prevAngleDeg !== null && prevAngleDeg !== undefined
      ? (primaryAngleDeg - prevAngleDeg) / Math.max(0.05, dtSec)
      : 0;

  return {
    auth_status: {
      verified: true,
      confidence: params.authConfidence ?? 0.95,
      liveness: true,
    },
    spatial_framing: {
      distance_m: estDistanceM,
      full_body_visible: fullBodyVisible,
      feet_visible: feetVisible,
      lighting_score: 0.9,
    },
    exercise_prescription: {
      exercise_id: exercise.id,
      exercise_name: `${exercise.name.en} (${exercise.name.th})`,
      target_joint: targetJoint,
      target_rom_deg: exercise.detection.targetMinDeg,
      hold_sec: exercise.detection.holdSec,
      target_reps: exercise.detection.defaultReps,
    },
    live_telemetry: {
      primary_angle_deg: primaryAngleDeg,
      trunk_tilt_deg: Number(trunkTiltDeg.toFixed(1)),
      contralateral_shoulder_elevation_cm: Number(shoulderElevCm.toFixed(1)),
      shoulder_elevation_cm: Number(shoulderElevCm.toFixed(1)),
      knee_valgus_deg: Number(kneeValgusDeg.toFixed(1)),
      angular_velocity_deg_per_sec: Number(angularVelocity.toFixed(1)),
      angular_velocity_deg_s: Number(angularVelocity.toFixed(1)),
      pain_score: 0,
      rapid_vertical_drop_detected: rapidDrop,
      dt_sec: dtSec,
    },
  };
}

export class ClinicalBiomechanicsAgent {
  private counterState: BiomechanicsCounterState = {
    valid_reps: 0,
    invalid_reps: 0,
    hold_elapsed_sec: 0,
  };

  reset(): void {
    this.counterState = {
      valid_reps: 0,
      invalid_reps: 0,
      hold_elapsed_sec: 0,
      awaiting_return_to_rest: false,
      rep_invalidated_in_cycle: false,
      reached_attempt_threshold: false,
    };
  }

  evaluate(packet: ClinicalTelemetryPacket): ClinicalBiomechanicsDecision {
    const { decision, nextState } = evaluateClinicalBiomechanicsPacket(packet, this.counterState);
    this.counterState = nextState;
    return decision;
  }
}

