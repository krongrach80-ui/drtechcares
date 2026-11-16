import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { CLINICAL_EXERCISES } from '@drtechcares/pose-engine';
import type { TodayPrescribedExerciseItem } from '@drtechcares/api-client';
import type {
  CalibrationState,
  HandRaiseTelemetry,
  IdentityGuardState,
  Keypoint3DPoint,
} from './useEdgeVisionSocket';

export type KioskState =
  | 'AUTH_FRAMING'
  | 'CALIBRATING'
  | 'ACTIVE_REP'
  | 'HOLD_PHASE'
  | 'REP_COMPLETED'
  | 'SET_FINISHED'
  | 'EMERGENCY_HALT'
  | 'PAUSED_MULTI_PERSON'
  | 'PAUSED_IMPOSTOR';

export type CompensationCode =
  | 'NONE'
  | 'TRUNK_LEAN'
  | 'SHOULDER_HIKE'
  | 'KNEE_VALGUS'
  | 'HEEL_LIFT'
  | 'TOO_FAST'
  | 'INCOMPLETE_ROM'
  | 'WRONG_LIMB'
  | 'WRONG_PLANE';

export type HighlightJoint =
  | 'none'
  | 'left_shoulder'
  | 'right_shoulder'
  | 'spine'
  | 'left_knee'
  | 'right_knee';

export type HighlightColor = 'GREEN' | 'AMBER' | 'RED';

export interface PatientSummary {
  id: string | null;
  name: string | null;
  hn?: string | null;
  similarity?: number;
  latency_ms?: number;
}

export interface TelemetryAnalysis {
  current_rom_deg: number;
  target_rom_deg: number;
  rom_completion_pct: number;
  trunk_tilt_deg?: number;
  shoulder_hike_cm?: number;
  knee_valgus_deg?: number;
  heel_lift_cm?: number;
  angular_velocity_deg_s?: number;
  is_form_valid: boolean;
  compensation_detected: CompensationCode[];
  pain_alert: boolean;
}

export interface RepCounter {
  valid_reps: number;
  invalid_reps: number;
  hold_remaining_sec: number;
  target_reps?: number;
}

export interface UIDirectives {
  skeleton_highlight_joint: HighlightJoint;
  highlight_color: HighlightColor;
  show_target_arc: boolean;
}

export interface KioskDecisionPayload {
  kiosk_state: KioskState;
  fsm_state?: string;
  fps?: number;
  engine_backend?: string;
  camera_mode?: string;
  patient: PatientSummary | null;
  telemetry_analysis: TelemetryAnalysis;
  rep_counter: RepCounter;
  ui_directives: UIDirectives;
  tts_voice_th: string;
  error_code?: string | null;
  error_message?: string | null;
  calibration?: CalibrationState;
  hand_status?: HandRaiseTelemetry;
  identity_guard?: IdentityGuardState;
  keypoints_3d?: Keypoint3DPoint[];
  pose_landmarks?: Keypoint3DPoint[];
  pose_world_landmarks?: Keypoint3DPoint[];
  frame_base64: string | null;
}

export type ExpectedMovementPlane = 'flexion' | 'abduction' | 'any';

export interface ExercisePrescriptionOption {
  exercise_id: string;
  catalog_exercise_id?: string;
  schedule_id?: string;
  name_th: string;
  name_en: string;
  target_joint: Exclude<HighlightJoint, 'none'>;
  expected_plane?: ExpectedMovementPlane;
  target_rom_deg: number;
  hold_sec: number;
  target_reps: number;
  target_sets?: number;
}

export type DemoPresetId =
  | 'FACE_SCAN'
  | 'ACTIVE_REP'
  | 'TRUNK_LEAN'
  | 'WRONG_LIMB'
  | 'WRONG_PLANE'
  | 'HOLD_PHASE'
  | 'REP_COMPLETED'
  | 'SET_FINISHED'
  | 'MULTI_PERSON'
  | 'IMPOSTOR'
  | 'EMERGENCY_HALT';

export function resolveExpectedPlaneForExercise(item: {
  exercise_id?: string;
  catalog_exercise_id?: string;
  name_th?: string;
  name_en?: string;
  target_joint?: string;
  expected_plane?: ExpectedMovementPlane;
}): ExpectedMovementPlane {
  if (
    item.expected_plane === 'flexion' ||
    item.expected_plane === 'abduction' ||
    item.expected_plane === 'any'
  ) {
    return item.expected_plane;
  }
  if (item.target_joint && item.target_joint !== 'left_shoulder' && item.target_joint !== 'right_shoulder') {
    return 'any';
  }
  const found = CLINICAL_EXERCISES.find(
    (c) =>
      c.id === item.catalog_exercise_id ||
      c.code === item.catalog_exercise_id ||
      c.id === item.exercise_id ||
      c.code === item.exercise_id,
  );
  if (found?.detection.expectedPlane) {
    return found.detection.expectedPlane;
  }
  const combined = `${item.name_th ?? ''} ${item.name_en ?? ''} ${item.exercise_id ?? ''}`.toLowerCase();
  if (
    !combined.includes('elbow') &&
    !combined.includes('ข้อศอก') &&
    (combined.includes('ด้านหน้า') || combined.includes('ข้างหน้า') || combined.includes('flexion'))
  ) {
    return 'flexion';
  }
  if (
    combined.includes('ด้านข้าง') ||
    combined.includes('กางแขน') ||
    combined.includes('abduction')
  ) {
    return 'abduction';
  }
  return 'any';
}

export function getPrescribedPlanePromptTh(expectedPlane: ExpectedMovementPlane): {
  requiredPlaneTh: string;
  wrongPlaneTh: string;
  bannerAlertTh: string;
  voicePromptTh: string;
} {
  if (expectedPlane === 'flexion') {
    return {
      requiredPlaneTh: 'ยกแขนตรงไปด้านหน้า',
      wrongPlaneTh: 'กางแขนออกด้านข้าง',
      bannerAlertTh: 'ท่านี้ต้องยกแขนตรงไปด้านหน้าเท่านั้น (ห้ามกางแขนออกด้านข้าง)',
      voicePromptTh: 'กรุณายกแขนตรงไปด้านหน้านะครับ ท่านี้ห้ามกางแขนออกด้านข้าง ระบบจะไม่นับรอบครับ',
    };
  }
  if (expectedPlane === 'abduction') {
    return {
      requiredPlaneTh: 'กางแขนออกด้านข้าง',
      wrongPlaneTh: 'ยกแขนไปด้านหน้า',
      bannerAlertTh: 'ท่านี้ต้องกางแขนออกด้านข้างเท่านั้น (ห้ามยกแขนไปด้านหน้า)',
      voicePromptTh: 'กรุณากางแขนออกด้านข้างนะครับ ท่านี้ห้ามยกแขนไปด้านหน้า ระบบจะไม่นับรอบครับ',
    };
  }
  return {
    requiredPlaneTh: 'ยกแขนตามทิศทางที่กำหนด',
    wrongPlaneTh: 'เคลื่อนไหวผิดทิศทาง',
    bannerAlertTh: 'กรุณายกแขนตามทิศทางที่กำหนดครับ',
    voicePromptTh: 'กรุณายกแขนตามทิศทางที่กำหนดนะครับ',
  };
}

export function getPrescribedLimbPromptTh(
  targetJoint: ExercisePrescriptionOption['target_joint'],
): {
  requiredLimbTh: string;
  wrongLimbTh: string;
  bannerAlertTh: string;
  voicePromptTh: string;
} {
  if (targetJoint === 'left_shoulder') {
    return {
      requiredLimbTh: 'แขนซ้าย',
      wrongLimbTh: 'แขนขวา',
      bannerAlertTh: 'กรุณายกแขนซ้าย (ท่านี้ต้องทำด้วยแขนซ้าย)',
      voicePromptTh: 'กรุณายกแขนซ้ายนะครับ ท่านี้ต้องทำด้วยแขนซ้าย ระบบจะไม่นับรอบหากยกแขนขวาครับ',
    };
  }
  if (targetJoint === 'right_shoulder') {
    return {
      requiredLimbTh: 'แขนขวา',
      wrongLimbTh: 'แขนซ้าย',
      bannerAlertTh: 'กรุณายกแขนขวา (ท่านี้ต้องทำด้วยแขนขวา)',
      voicePromptTh: 'กรุณายกแขนขวานะครับ ท่านี้ต้องทำด้วยแขนขวา ระบบจะไม่นับรอบหากยกแขนซ้ายครับ',
    };
  }
  if (targetJoint === 'left_knee') {
    return {
      requiredLimbTh: 'ขาซ้าย',
      wrongLimbTh: 'ขาขวา',
      bannerAlertTh: 'กรุณาใช้ขาซ้าย (ท่านี้ต้องทำด้วยขาซ้าย)',
      voicePromptTh: 'กรุณาทำท่าด้วยขาซ้ายตามที่กำหนดนะครับ',
    };
  }
  return {
    requiredLimbTh: 'ขาขวา',
    wrongLimbTh: 'ขาซ้าย',
    bannerAlertTh: 'กรุณาใช้ขาขวาตามท่าที่กำหนด',
    voicePromptTh: 'กรุณาทำท่าตามข้างที่กำหนดนะครับ',
  };
}

export type ConnectionStatus = 'connecting' | 'connected' | 'offline';

export const EXERCISE_OPTIONS: ExercisePrescriptionOption[] = [
  {
    exercise_id: 'right_shoulder_abduction',
    name_th: 'กางแขนขวาด้านข้าง',
    name_en: 'Right Shoulder Abduction',
    target_joint: 'right_shoulder',
    expected_plane: 'abduction',
    target_rom_deg: 150,
    hold_sec: 3,
    target_reps: 10,
  },
  {
    exercise_id: 'left_shoulder_abduction',
    name_th: 'กางแขนซ้ายด้านข้าง',
    name_en: 'Left Shoulder Abduction',
    target_joint: 'left_shoulder',
    expected_plane: 'abduction',
    target_rom_deg: 150,
    hold_sec: 3,
    target_reps: 10,
  },
  {
    exercise_id: 'chair_squat',
    name_th: 'ลุกนั่งฝึกกล้ามเนื้อขา',
    name_en: 'Chair Squat Rehabilitation',
    target_joint: 'right_knee',
    target_rom_deg: 100,
    hold_sec: 2,
    target_reps: 10,
  },
];

export const DEMO_PATIENT: PatientSummary = {
  id: 'PT-8841',
  name: 'คุณสมชาย ใจดี',
};

function resolveTargetJointForTodayExercise(
  item: TodayPrescribedExerciseItem,
  catalogMatch?: (typeof CLINICAL_EXERCISES)[number],
): Exclude<HighlightJoint, 'none'> {
  if (catalogMatch) {
    const group = catalogMatch.detection.jointGroup;
    if (group === 'knee' || group === 'hip' || group === 'ankle') {
      return catalogMatch.detection.side === 'left' ? 'left_knee' : 'right_knee';
    }
    if (group === 'trunk' || group === 'neck') {
      return 'spine';
    }
    return catalogMatch.detection.side === 'left' ? 'left_shoulder' : 'right_shoulder';
  }
  const combined = `${item.nameTh} ${item.nameEn} ${item.targetJoint}`.toLowerCase();
  if (
    combined.includes('เข่า') ||
    combined.includes('ขา') ||
    combined.includes('knee') ||
    combined.includes('squat')
  ) {
    return combined.includes('ซ้าย') || combined.includes('left') ? 'left_knee' : 'right_knee';
  }
  if (combined.includes('ซ้าย') || combined.includes('left')) {
    return 'left_shoulder';
  }
  return 'right_shoulder';
}

export function mapTodayExercisesToPrescriptions(
  items: readonly TodayPrescribedExerciseItem[],
): ExercisePrescriptionOption[] {
  const physioItems = items.filter(
    (item) =>
      item.activityType !== 'minigame' &&
      !item.exerciseId.includes('card') &&
      !item.exerciseId.includes('minigame'),
  );
  const sourceItems = physioItems.length > 0 ? physioItems : items;
  if (sourceItems.length === 0) {
    return EXERCISE_OPTIONS;
  }

  return sourceItems.map((item) => {
    const found = CLINICAL_EXERCISES.find(
      (c) => c.id === item.exerciseId || c.code === item.exerciseId,
    );
    const target_joint = resolveTargetJointForTodayExercise(item, found);
    const target_rom_deg = found
      ? Math.min(
          160,
          Math.max(90, Math.round((found.detection.targetMinDeg + found.detection.targetMaxDeg) / 2)),
        )
      : target_joint.includes('knee')
        ? 100
        : 150;

    return {
      exercise_id: item.id || item.exerciseId,
      catalog_exercise_id: item.exerciseId,
      schedule_id: item.id,
      name_th: item.nameTh,
      name_en: item.nameEn,
      target_joint,
      expected_plane:
        found?.detection.expectedPlane ??
        resolveExpectedPlaneForExercise({
          exercise_id: item.exerciseId,
          name_th: item.nameTh,
          name_en: item.nameEn,
          target_joint,
        }),
      target_rom_deg,
      hold_sec: item.holdSec || (found?.detection.holdSec ?? 3),
      target_reps: item.reps || (found?.detection.defaultReps ?? 10),
      target_sets: item.sets || 1,
    };
  });
}

export interface DemoSynthesisParams {
  preset: DemoPresetId;
  romDeg: number;
  trunkTiltDeg: number;
  distanceM: number;
  exercise: ExercisePrescriptionOption;
  validReps: number;
  invalidReps: number;
  holdRemainingSec: number;
  faceMatched?: boolean;
  patientOverride?: PatientSummary | null;
}

export function synthesizeDemoPayload({
  preset,
  romDeg,
  trunkTiltDeg,
  distanceM,
  exercise,
  validReps,
  invalidReps,
  holdRemainingSec,
  faceMatched = false,
  patientOverride,
}: DemoSynthesisParams): KioskDecisionPayload {
  const activePatient = patientOverride ?? DEMO_PATIENT;
  const targetRom = exercise.target_rom_deg;
  const clampedRom = Math.max(0, Math.min(180, Number(romDeg.toFixed(1))));
  const romPct = Math.max(0, Math.min(120, Math.round((clampedRom / targetRom) * 100)));

  if (preset === 'MULTI_PERSON') {
    const authLabel = activePatient.hn
      ? `${activePatient.hn} ${activePatient.name || 'คนไข้หลัก'}`
      : (activePatient.name || 'คนไข้หลัก');
    return {
      kiosk_state: 'PAUSED_MULTI_PERSON',
      fsm_state: 'PAUSED_MULTI_PERSON',
      patient: activePatient,
      telemetry_analysis: {
        current_rom_deg: clampedRom,
        target_rom_deg: targetRom,
        rom_completion_pct: romPct,
        is_form_valid: false,
        compensation_detected: ['NONE'],
        pain_alert: false,
      },
      rep_counter: {
        valid_reps: validReps,
        invalid_reps: invalidReps,
        hold_remaining_sec: exercise.hold_sec,
      },
      ui_directives: {
        skeleton_highlight_joint: 'none',
        highlight_color: 'AMBER',
        show_target_arc: false,
      },
      identity_guard: {
        status: 'MULTI_PERSON',
        person_count: 2,
        face_count: 2,
        match_confidence: 0.91,
        is_counting_allowed: false,
        alert_title_th: 'ตรวจพบมีบุคคลอยู่หลายคนในเฟรม',
        alert_subtitle_th: 'กรุณาให้คนไข้ยืนหน้ากล้องเพียงลำพัง ระบบจึงจะนับรอบการฝึกต่อครับ',
        detected_boxes: [
          { bbox: [220, 90, 560, 670], label: authLabel, is_authorized: true },
          { bbox: [730, 120, 1060, 665], label: 'บุคคลเกินในเฟรม', is_authorized: false },
        ],
      },
      tts_voice_th: 'ตรวจพบมีบุคคลอยู่หลายคนในเฟรม กรุณาให้คนไข้ยืนหน้ากล้องเพียงลำพัง ระบบจึงจะนับรอบการฝึกต่อครับ',
      frame_base64: null,
    };
  }

  if (preset === 'IMPOSTOR') {
    return {
      kiosk_state: 'PAUSED_IMPOSTOR',
      fsm_state: 'PAUSED_IMPOSTOR',
      patient: activePatient,
      telemetry_analysis: {
        current_rom_deg: clampedRom,
        target_rom_deg: targetRom,
        rom_completion_pct: romPct,
        is_form_valid: false,
        compensation_detected: ['NONE'],
        pain_alert: false,
      },
      rep_counter: {
        valid_reps: validReps,
        invalid_reps: invalidReps,
        hold_remaining_sec: exercise.hold_sec,
      },
      ui_directives: {
        skeleton_highlight_joint: 'none',
        highlight_color: 'RED',
        show_target_arc: false,
      },
      identity_guard: {
        status: 'IMPOSTOR',
        person_count: 1,
        face_count: 1,
        match_confidence: 0.34,
        is_counting_allowed: false,
        alert_title_th: 'คุณไม่ใช่คนไข้ที่ลงทะเบียนไว้',
        alert_subtitle_th: 'ระบบจะไม่นับรอบการฝึก กรุณาให้คนไข้ตัวจริงเข้าฝึกครับ',
        detected_boxes: [
          { bbox: [420, 80, 860, 680], label: '✕ ไม่ใช่คนไข้ที่ลงทะเบียน', is_authorized: false },
        ],
      },
      tts_voice_th: 'คุณไม่ใช่คนไข้ที่ลงทะเบียนไว้ ระบบจะไม่นับรอบการฝึก กรุณาให้คนไข้ตัวจริงเข้าฝึกครับ',
      frame_base64: null,
    };
  }

  if (preset === 'EMERGENCY_HALT') {
    return {
      kiosk_state: 'EMERGENCY_HALT',
      patient: activePatient,
      telemetry_analysis: {
        current_rom_deg: clampedRom,
        target_rom_deg: targetRom,
        rom_completion_pct: romPct,
        is_form_valid: false,
        compensation_detected: ['NONE'],
        pain_alert: true,
      },
      rep_counter: {
        valid_reps: validReps,
        invalid_reps: invalidReps,
        hold_remaining_sec: exercise.hold_sec,
      },
      ui_directives: {
        skeleton_highlight_joint: exercise.target_joint,
        highlight_color: 'RED',
        show_target_arc: false,
      },
      tts_voice_th: 'หยุดพักทันทีนะครับ ระบบกำลังเรียกเจ้าหน้าที่ดูแล',
      frame_base64: null,
    };
  }

  if (preset === 'FACE_SCAN') {
    const distanceOk = distanceM >= 1.8 && distanceM <= 2.2;
    return {
      kiosk_state: 'AUTH_FRAMING',
      patient: faceMatched ? activePatient : null,
      telemetry_analysis: {
        current_rom_deg: 12.0,
        target_rom_deg: targetRom,
        rom_completion_pct: 8,
        is_form_valid: false,
        compensation_detected: ['NONE'],
        pain_alert: false,
      },
      rep_counter: {
        valid_reps: 0,
        invalid_reps: 0,
        hold_remaining_sec: exercise.hold_sec,
      },
      ui_directives: {
        skeleton_highlight_joint: 'none',
        highlight_color: faceMatched && distanceOk ? 'GREEN' : 'AMBER',
        show_target_arc: false,
      },
      tts_voice_th: !faceMatched
        ? 'กรุณามองตรงเข้ากรอบวงกลม เพื่อยืนยันตัวตนอัตโนมัติครับ'
        : distanceM < 1.8
          ? 'ยืนยันตัวตนสำเร็จ กรุณาถอยหลังเล็กน้อยให้อยู่ระยะ 2 เมตรครับ'
          : distanceM > 2.2
            ? 'ยืนยันตัวตนสำเร็จ กรุณาเดินเข้ามาอีกนิดให้อยู่ระยะ 2 เมตรครับ'
            : 'ยืนยันตัวตนสำเร็จ ระยะยืนเหมาะสม พร้อมเริ่มฝึกได้เลยครับ',
      frame_base64: null,
    };
  }

  const isLowerBody =
    exercise.target_joint.includes('knee') || exercise.target_joint.includes('hip');
  const lowerLimbInstructionTh = isLowerBody
    ? 'ค่อยๆ เอาขาลงช้าๆ ครับ'
    : 'ค่อยๆ เอาแขนลงช้าๆ ครับ';

  if (preset === 'SET_FINISHED') {
    const patientName = activePatient.name || 'ผู้ป่วย';
    return {
      kiosk_state: 'SET_FINISHED',
      patient: activePatient,
      telemetry_analysis: {
        current_rom_deg: targetRom,
        target_rom_deg: targetRom,
        rom_completion_pct: 100,
        is_form_valid: true,
        compensation_detected: ['NONE'],
        pain_alert: false,
      },
      rep_counter: {
        valid_reps: exercise.target_reps,
        invalid_reps: invalidReps,
        hold_remaining_sec: 0,
      },
      ui_directives: {
        skeleton_highlight_joint: exercise.target_joint,
        highlight_color: 'GREEN',
        show_target_arc: false,
      },
      tts_voice_th: `ครบ ${exercise.hold_sec} วินาทีแล้ว เยี่ยมมากครับ ${patientName} ฝึกครบ ${exercise.target_reps} ครั้ง จบเซ็ตแล้ว ${lowerLimbInstructionTh}`,
      frame_base64: null,
    };
  }

  if (preset === 'REP_COMPLETED') {
    return {
      kiosk_state: 'REP_COMPLETED',
      patient: activePatient,
      telemetry_analysis: {
        current_rom_deg: Math.max(clampedRom, targetRom * 0.96),
        target_rom_deg: targetRom,
        rom_completion_pct: Math.max(romPct, 96),
        is_form_valid: true,
        compensation_detected: ['NONE'],
        pain_alert: false,
      },
      rep_counter: {
        valid_reps: validReps,
        invalid_reps: invalidReps,
        hold_remaining_sec: 0,
      },
      ui_directives: {
        skeleton_highlight_joint: exercise.target_joint,
        highlight_color: 'GREEN',
        show_target_arc: true,
      },
      tts_voice_th: `ครบ ${exercise.hold_sec} วินาทีแล้ว ดีมากครับ ครบ ${validReps} ครั้ง ${lowerLimbInstructionTh}`,
      frame_base64: null,
    };
  }

  if (preset === 'WRONG_LIMB') {
    const limbInfo = getPrescribedLimbPromptTh(exercise.target_joint);
    return {
      kiosk_state: 'ACTIVE_REP',
      patient: activePatient,
      telemetry_analysis: {
        current_rom_deg: clampedRom,
        target_rom_deg: targetRom,
        rom_completion_pct: romPct,
        is_form_valid: false,
        compensation_detected: ['WRONG_LIMB'],
        pain_alert: false,
      },
      rep_counter: {
        valid_reps: validReps,
        invalid_reps: Math.max(1, invalidReps),
        hold_remaining_sec: exercise.hold_sec,
      },
      ui_directives: {
        skeleton_highlight_joint: exercise.target_joint,
        highlight_color: 'AMBER',
        show_target_arc: true,
      },
      tts_voice_th: limbInfo.voicePromptTh,
      frame_base64: null,
    };
  }

  if (preset === 'WRONG_PLANE') {
    const expectedPlane = resolveExpectedPlaneForExercise(exercise);
    const planeInfo = getPrescribedPlanePromptTh(expectedPlane);
    return {
      kiosk_state: 'ACTIVE_REP',
      patient: activePatient,
      telemetry_analysis: {
        current_rom_deg: 0,
        target_rom_deg: targetRom,
        rom_completion_pct: 0,
        is_form_valid: false,
        compensation_detected: ['WRONG_PLANE'],
        pain_alert: false,
      },
      rep_counter: {
        valid_reps: validReps,
        invalid_reps: Math.max(1, invalidReps),
        hold_remaining_sec: exercise.hold_sec,
      },
      ui_directives: {
        skeleton_highlight_joint: exercise.target_joint,
        highlight_color: 'AMBER',
        show_target_arc: true,
      },
      tts_voice_th: planeInfo.voicePromptTh,
      frame_base64: null,
    };
  }

  const trunkLeaning = preset === 'TRUNK_LEAN' || Math.abs(trunkTiltDeg) > 10.0;
  if (trunkLeaning) {
    const compensations: CompensationCode[] =
      clampedRom < targetRom * 0.95 ? ['TRUNK_LEAN', 'INCOMPLETE_ROM'] : ['TRUNK_LEAN'];
    return {
      kiosk_state: 'ACTIVE_REP',
      patient: activePatient,
      telemetry_analysis: {
        current_rom_deg: clampedRom,
        target_rom_deg: targetRom,
        rom_completion_pct: romPct,
        is_form_valid: false,
        compensation_detected: compensations,
        pain_alert: false,
      },
      rep_counter: {
        valid_reps: validReps,
        invalid_reps: Math.max(1, invalidReps),
        hold_remaining_sec: exercise.hold_sec,
      },
      ui_directives: {
        skeleton_highlight_joint: 'spine',
        highlight_color: 'AMBER',
        show_target_arc: true,
      },
      tts_voice_th: 'ยืดตัวตรงครับ อย่าเอียงลำตัวช่วยยกนะครับ',
      frame_base64: null,
    };
  }

  const inHoldPhase = preset === 'HOLD_PHASE' || clampedRom >= targetRom * 0.95;
  if (inHoldPhase) {
    const rem = Math.max(0.1, Number(holdRemainingSec.toFixed(1)));
    return {
      kiosk_state: 'HOLD_PHASE',
      patient: activePatient,
      telemetry_analysis: {
        current_rom_deg: clampedRom,
        target_rom_deg: targetRom,
        rom_completion_pct: romPct,
        is_form_valid: true,
        compensation_detected: ['NONE'],
        pain_alert: false,
      },
      rep_counter: {
        valid_reps: validReps,
        invalid_reps: invalidReps,
        hold_remaining_sec: rem,
      },
      ui_directives: {
        skeleton_highlight_joint: exercise.target_joint,
        highlight_color: 'GREEN',
        show_target_arc: true,
      },
      tts_voice_th: `ดีมากครับ ค้างท่าไว้ ${Math.ceil(rem)} วินาที`,
      frame_base64: null,
    };
  }

  return {
    kiosk_state: 'ACTIVE_REP',
    patient: activePatient,
    telemetry_analysis: {
      current_rom_deg: clampedRom,
      target_rom_deg: targetRom,
      rom_completion_pct: romPct,
      is_form_valid: true,
      compensation_detected: ['NONE'],
      pain_alert: false,
    },
    rep_counter: {
      valid_reps: validReps,
      invalid_reps: invalidReps,
      hold_remaining_sec: exercise.hold_sec,
    },
    ui_directives: {
      skeleton_highlight_joint: exercise.target_joint,
      highlight_color: 'GREEN',
      show_target_arc: true,
    },
    tts_voice_th: `ค่อยๆ เคลื่อนไหวไปยังเป้าหมาย ${targetRom} องศาครับ`,
    frame_base64: null,
  };
}

/**
 * Builds an ordered, deduplicated list of Kiosk Edge Vision WebSocket endpoints
 * across env overrides, localStorage config, and localhost / 127.0.0.1 fallbacks.
 */
export function resolveKioskWsCandidates(defaultWsUrl: string): string[] {
  const candidates: string[] = [];
  const pushUnique = (val?: string | null) => {
    const trimmed = val?.trim();
    if (trimmed && !candidates.includes(trimmed)) {
      candidates.push(trimmed);
    }
  };
  pushUnique(import.meta.env.VITE_KIOSK_WS_URL as string | undefined);
  pushUnique(import.meta.env.VITE_EDGE_VISION_WS as string | undefined);
  if (typeof window !== 'undefined' && window.localStorage) {
    try {
      pushUnique(window.localStorage.getItem('drtechcares_edge_ws_url'));
    } catch {
      // Ignore storage access errors
    }
  }
  pushUnique(defaultWsUrl);
  pushUnique('ws://localhost:8000/ws/kiosk');
  pushUnique('ws://127.0.0.1:8000/ws/kiosk');
  return candidates;
}

export function useKioskSocket(
  wsUrl = 'ws://localhost:8000/ws/kiosk',
  options?: {
    prescribedExercises?: ExercisePrescriptionOption[];
    patientOverride?: PatientSummary | null;
  },
) {
  const wsCandidates = useMemo(() => resolveKioskWsCandidates(wsUrl), [wsUrl]);
  const [activeWsUrl, setActiveWsUrl] = useState<string>(() => wsCandidates[0] ?? wsUrl);

  const exerciseOptions = useMemo(
    () =>
      options?.prescribedExercises && options.prescribedExercises.length > 0
        ? options.prescribedExercises
        : EXERCISE_OPTIONS,
    [options?.prescribedExercises],
  );
  const hasExplicitContext = Boolean(
    options?.prescribedExercises?.length || options?.patientOverride,
  );

  const [selectedExercise, setSelectedExerciseState] = useState<ExercisePrescriptionOption>(
    () => exerciseOptions[0],
  );
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>('connecting');
  const [isDemoMode, setIsDemoMode] = useState<boolean>(false);
  const [livePayload, setLivePayload] = useState<KioskDecisionPayload | null>(null);

  // Interactive Demo Simulator state
  const [demoPreset, setRawDemoPresetState] = useState<DemoPresetId>('ACTIVE_REP');
  const [manualHoldCountdown, setManualHoldCountdown] = useState<boolean>(false);
  const [demoRomDeg, setDemoRomDeg] = useState<number>(0);

  const setDemoPresetState = useCallback((next: DemoPresetId) => {
    setManualHoldCountdown(false);
    setRawDemoPresetState(next);
  }, []);
  const [demoTrunkTiltDeg, setDemoTrunkTiltDeg] = useState<number>(0);
  const [demoDistanceM, setDemoDistanceM] = useState<number>(0);
  const [demoValidReps, setDemoValidReps] = useState<number>(() => (hasExplicitContext ? 0 : 4));
  const [demoInvalidReps, setDemoInvalidReps] = useState<number>(0);
  const [demoHoldRemainingSec, setDemoHoldRemainingSec] = useState<number>(
    () => exerciseOptions[0].hold_sec,
  );
  const [demoFaceMatched, setDemoFaceMatched] = useState<boolean>(false);
  const [demoStreakFrames, setDemoStreakFrames] = useState<number>(2);
  const [autoCycleDemo, setAutoCycleDemo] = useState<boolean>(false);

  const wsRef = useRef<WebSocket | null>(null);
  const userToggledModeRef = useRef<boolean>(false);

  const sendExercisePrescription = useCallback((rx: ExercisePrescriptionOption) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(
        JSON.stringify({
          type: 'config',
          stream_video: true,
        }),
      );
      ws.send(
        JSON.stringify({
          type: 'prescription',
          include_frame: true,
          data: {
            exercise_id: rx.catalog_exercise_id ?? rx.exercise_id,
            exercise_name: `${rx.name_en} (${rx.name_th})`,
            target_joint: rx.target_joint,
            target_rom_deg: rx.target_rom_deg,
            hold_sec: rx.hold_sec,
            target_reps: rx.target_reps,
          },
        }),
      );
    }
  }, []);

  useEffect(() => {
    if (!exerciseOptions.some((opt) => opt.exercise_id === selectedExercise.exercise_id)) {
      const first = exerciseOptions[0];
      setSelectedExerciseState(first);
      setDemoValidReps(0);
      setDemoInvalidReps(0);
      setDemoHoldRemainingSec(first.hold_sec);
      setDemoPresetState('ACTIVE_REP');
      sendExercisePrescription(first);
    }
  }, [exerciseOptions, selectedExercise.exercise_id, sendExercisePrescription]);

  const setSelectedExercise = useCallback(
    (rx: ExercisePrescriptionOption) => {
      setSelectedExerciseState(rx);
      setDemoValidReps(0);
      setDemoInvalidReps(0);
      setDemoPresetState('ACTIVE_REP');
      setDemoHoldRemainingSec(rx.hold_sec);
      sendExercisePrescription(rx);
    },
    [sendExercisePrescription],
  );

  // WebSocket client with automatic exponential backoff reconnect
  useEffect(() => {
    let disposed = false;
    let attempt = 0;
    let reconnectTimer: number | null = null;

    const connect = () => {
      if (disposed) return;
      setConnectionStatus('connecting');

      try {
        const targetWsUrl = wsCandidates[attempt % wsCandidates.length] ?? wsUrl;
        setActiveWsUrl(targetWsUrl);
        // Trigger CORS + Private Network Access preflight prior to loopback WebSocket upgrade
        const healthUrl = targetWsUrl
          .replace(/^ws:/i, 'http:')
          .replace(/^wss:/i, 'https:')
          .replace(/\/ws\/kiosk.*$/i, '/health');
        void fetch(healthUrl, { mode: 'cors' }).catch(() => undefined);
        const ws = new WebSocket(targetWsUrl);
        wsRef.current = ws;

        ws.onopen = () => {
          if (disposed) return;
          attempt = 0;
          setConnectionStatus('connected');
          if (!userToggledModeRef.current) {
            setIsDemoMode(false);
          }
          ws.send(JSON.stringify({ type: 'config', stream_video: true }));
          if (options?.patientOverride?.id) {
            ws.send(
              JSON.stringify({
                type: 'lock_patient',
                patient: {
                  id: options.patientOverride.id,
                  hn: options.patientOverride.hn ?? '',
                  full_name: options.patientOverride.name ?? options.patientOverride.id,
                },
                start_calibration: false,
              }),
            );
          }
          sendExercisePrescription(selectedExercise);
        };

        ws.onmessage = (event) => {
          if (disposed) return;
          try {
            const data = JSON.parse(event.data) as KioskDecisionPayload;
            if (data && data.kiosk_state && data.telemetry_analysis) {
              setLivePayload(data);
            }
          } catch {
            // Ignore malformed non-JSON frame
          }
        };

        ws.onerror = () => {
          if (disposed) return;
          ws.close();
        };

        ws.onclose = () => {
          if (disposed) return;
          wsRef.current = null;
          setConnectionStatus('offline');
          const delayMs = Math.min(10000, 1500 * Math.pow(1.6, attempt));
          attempt += 1;
          reconnectTimer = window.setTimeout(connect, delayMs);
        };
      } catch {
        setConnectionStatus('offline');
        const delayMs = Math.min(10000, 1500 * Math.pow(1.6, attempt));
        attempt += 1;
        reconnectTimer = window.setTimeout(connect, delayMs);
      }
    };

    connect();
    return () => {
      disposed = true;
      if (reconnectTimer !== null) {
        window.clearTimeout(reconnectTimer);
      }
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [wsCandidates, wsUrl, sendExercisePrescription]);

  // Keep Edge Vision active_patient locked whenever authenticated patient updates
  useEffect(() => {
    const ws = wsRef.current;
    if (connectionStatus !== 'connected' || !ws || ws.readyState !== WebSocket.OPEN) return;
    const pid = options?.patientOverride?.id;
    if (!pid) return;
    ws.send(
      JSON.stringify({
        type: 'lock_patient',
        patient: {
          id: pid,
          hn: options?.patientOverride?.hn ?? '',
          full_name: options?.patientOverride?.name ?? pid,
        },
        start_calibration: false,
      }),
    );
  }, [
    connectionStatus,
    options?.patientOverride?.id,
    options?.patientOverride?.hn,
    options?.patientOverride?.name,
  ]);

  // Face scan animated streak in FACE_SCAN preset
  useEffect(() => {
    if (!isDemoMode || demoPreset !== 'FACE_SCAN') return;
    setDemoFaceMatched(false);
    setDemoStreakFrames(1);
    const timer = window.setInterval(() => {
      setDemoStreakFrames((prev) => {
        if (prev >= 5) {
          setDemoFaceMatched(true);
          return 5;
        }
        const next = prev + 1;
        if (next >= 5) setDemoFaceMatched(true);
        return next;
      });
    }, 550);
    return () => window.clearInterval(timer);
  }, [isDemoMode, demoPreset]);

  // Hold phase smooth wall-clock countdown when manually inspecting HOLD_PHASE
  useEffect(() => {
    if (!isDemoMode || demoPreset !== 'HOLD_PHASE' || !manualHoldCountdown || autoCycleDemo) return;
    const holdDurationSec = selectedExercise.hold_sec;
    const startedAt = performance.now();
    setDemoHoldRemainingSec(holdDurationSec);
    const interval = window.setInterval(() => {
      const elapsedSec = (performance.now() - startedAt) / 1000;
      const remaining = Number(Math.max(0, holdDurationSec - elapsedSec).toFixed(1));
      setDemoHoldRemainingSec(remaining);
      if (remaining <= 0) {
        window.clearInterval(interval);
        setManualHoldCountdown(false);
        setDemoValidReps((r) => {
          const next = Math.min(selectedExercise.target_reps, r + 1);
          setRawDemoPresetState(
            next >= selectedExercise.target_reps ? 'SET_FINISHED' : 'REP_COMPLETED',
          );
          return next;
        });
      }
    }, 50);
    return () => window.clearInterval(interval);
  }, [
    isDemoMode,
    demoPreset,
    manualHoldCountdown,
    autoCycleDemo,
    selectedExercise.hold_sec,
    selectedExercise.target_reps,
  ]);

  // Smooth continuous auto-simulation loop when Auto-Cycle Demo is toggled ON
  useEffect(() => {
    if (!isDemoMode || !autoCycleDemo) return;
    let phaseT = 0;
    let holdAccumSec = 0;
    let repCelebrateRemainingSec = 0;
    let lastTickAt = performance.now();
    const interval = window.setInterval(() => {
      const now = performance.now();
      const dtSec = Math.min(0.25, Math.max(0.001, (now - lastTickAt) / 1000));
      lastTickAt = now;

      if (repCelebrateRemainingSec > 0) {
        repCelebrateRemainingSec = Math.max(0, repCelebrateRemainingSec - dtSec);
        return;
      }

      const target = selectedExercise.target_rom_deg;
      const currentSine = (Math.sin(phaseT - Math.PI / 2) + 1) / 2;
      const currentRom = Math.round(currentSine * (target + 4));
      const inPeakHoldWindow = currentRom >= target * 0.95 && holdAccumSec < selectedExercise.hold_sec;

      if (!inPeakHoldWindow) {
        phaseT += dtSec * 0.95;
      }
      const sine = (Math.sin(phaseT - Math.PI / 2) + 1) / 2;
      const rom = Math.round(sine * (target + 4));
      setDemoRomDeg(rom);

      if (rom >= target * 0.95) {
        holdAccumSec += dtSec;
        const rem = Math.max(0.1, Number((selectedExercise.hold_sec - holdAccumSec).toFixed(1)));
        setDemoHoldRemainingSec(rem);
        setDemoPresetState('HOLD_PHASE');
        if (holdAccumSec >= selectedExercise.hold_sec) {
          holdAccumSec = 0;
          phaseT = 0;
          repCelebrateRemainingSec = 3.5;
          setDemoHoldRemainingSec(0);
          setDemoValidReps((r) => {
            const next = Math.min(selectedExercise.target_reps, r + 1);
            if (next >= selectedExercise.target_reps) {
              setDemoPresetState('SET_FINISHED');
              setAutoCycleDemo(false);
            } else {
              setDemoPresetState('REP_COMPLETED');
            }
            return next;
          });
        }
      } else {
        holdAccumSec = 0;
        setDemoHoldRemainingSec(selectedExercise.hold_sec);
        setDemoPresetState(demoTrunkTiltDeg > 10 ? 'TRUNK_LEAN' : 'ACTIVE_REP');
      }
    }, 16);
    return () => window.clearInterval(interval);
  }, [
    isDemoMode,
    autoCycleDemo,
    selectedExercise.target_rom_deg,
    selectedExercise.hold_sec,
    selectedExercise.target_reps,
    demoTrunkTiltDeg,
    setDemoPresetState,
  ]);

  const setDemoPreset = useCallback(
    (preset: DemoPresetId) => {
      setAutoCycleDemo(false);
      setManualHoldCountdown(preset === 'HOLD_PHASE');
      setRawDemoPresetState(preset);
      switch (preset) {
        case 'FACE_SCAN':
          setDemoTrunkTiltDeg(1.5);
          setDemoRomDeg(12);
          break;
        case 'ACTIVE_REP':
          setDemoTrunkTiltDeg(2.2);
          setDemoRomDeg(Math.round(selectedExercise.target_rom_deg * 0.72));
          break;
        case 'TRUNK_LEAN':
          setDemoTrunkTiltDeg(16.4);
          setDemoRomDeg(Math.round(selectedExercise.target_rom_deg * 0.88));
          setDemoInvalidReps((r) => Math.max(1, r));
          break;
        case 'WRONG_LIMB':
          setDemoTrunkTiltDeg(2.0);
          setDemoRomDeg(12);
          setDemoInvalidReps((r) => Math.max(1, r));
          break;
        case 'WRONG_PLANE':
          setDemoTrunkTiltDeg(2.0);
          setDemoRomDeg(0);
          setDemoInvalidReps((r) => Math.max(1, r));
          break;
        case 'HOLD_PHASE':
          setDemoTrunkTiltDeg(1.8);
          setDemoRomDeg(selectedExercise.target_rom_deg);
          setDemoHoldRemainingSec(selectedExercise.hold_sec);
          break;
        case 'REP_COMPLETED':
          setDemoTrunkTiltDeg(1.5);
          setDemoRomDeg(selectedExercise.target_rom_deg);
          setDemoValidReps((r) => Math.min(selectedExercise.target_reps, r + 1));
          setDemoHoldRemainingSec(0);
          break;
        case 'SET_FINISHED':
          setDemoTrunkTiltDeg(1.2);
          setDemoRomDeg(selectedExercise.target_rom_deg);
          setDemoValidReps(selectedExercise.target_reps);
          setDemoHoldRemainingSec(0);
          break;
        case 'EMERGENCY_HALT':
          setDemoTrunkTiltDeg(4.0);
          break;
        case 'MULTI_PERSON':
          setDemoTrunkTiltDeg(2.0);
          break;
        case 'IMPOSTOR':
          setDemoTrunkTiltDeg(2.0);
          break;
      }
    },
    [selectedExercise],
  );

  const toggleDemoMode = useCallback(
    (next?: boolean) => {
      userToggledModeRef.current = true;
      const targetNext = typeof next === 'boolean' ? next : !isDemoMode;
      setIsDemoMode(targetNext);
      if (!targetNext) {
        setDemoRomDeg(0);
        setDemoTrunkTiltDeg(0);
        setDemoDistanceM(0);
      } else {
        setDemoRomDeg(15);
        setDemoTrunkTiltDeg(2.0);
        setDemoDistanceM(2.0);
      }
    },
    [isDemoMode],
  );

  const triggerEmergencyStop = useCallback(() => {
    setAutoCycleDemo(false);
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN && !isDemoMode) {
      ws.send(JSON.stringify({ type: 'emergency_stop', reason: 'Kiosk Emergency Stop Pressed' }));
      return;
    }
    userToggledModeRef.current = true;
    setIsDemoMode(true);
    setDemoPresetState('EMERGENCY_HALT');
  }, [isDemoMode]);

  const resetDemoSession = useCallback(() => {
    setAutoCycleDemo(false);
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN && !isDemoMode) {
      ws.send(JSON.stringify({ type: 'reset_session', return_to_auth: false }));
      return;
    }
    setDemoValidReps(0);
    setDemoInvalidReps(0);
    setDemoTrunkTiltDeg(1.8);
    setDemoRomDeg(Math.round(selectedExercise.target_rom_deg * 0.65));
    setDemoPresetState('ACTIVE_REP');
  }, [isDemoMode, selectedExercise.target_rom_deg]);

  const completeCurrentExercise = useCallback(() => {
    setAutoCycleDemo(false);
    setDemoTrunkTiltDeg(1.2);
    setDemoRomDeg(selectedExercise.target_rom_deg);
    setDemoValidReps(selectedExercise.target_reps);
    setDemoHoldRemainingSec(0);
    setDemoPresetState('SET_FINISHED');
  }, [selectedExercise.target_rom_deg, selectedExercise.target_reps]);

  const demoPayload = synthesizeDemoPayload({
    preset: demoPreset,
    romDeg: demoRomDeg,
    trunkTiltDeg: demoTrunkTiltDeg,
    distanceM: demoDistanceM,
    exercise: selectedExercise,
    validReps: demoValidReps,
    invalidReps: demoInvalidReps,
    holdRemainingSec: demoHoldRemainingSec,
    faceMatched: demoFaceMatched,
    patientOverride: options?.patientOverride,
  });

  const sendClientLandmarks = useCallback(
    (
      poseLandmarks: ReadonlyArray<{ x: number; y: number; z?: number; visibility?: number }>,
      worldLandmarks?: ReadonlyArray<{ x: number; y: number; z?: number; visibility?: number }>,
      personCount = 1,
    ) => {
      const ws = wsRef.current;
      if (!ws || ws.readyState !== WebSocket.OPEN) return;
      ws.send(
        JSON.stringify({
          type: 'client_landmarks',
          person_count: personCount,
          pose_landmarks: poseLandmarks.map((lm) => ({
            x: Number(lm.x.toFixed(4)),
            y: Number(lm.y.toFixed(4)),
            z: Number((lm.z ?? 0).toFixed(4)),
            score: Number((lm.visibility ?? 1).toFixed(3)),
          })),
          pose_world_landmarks: worldLandmarks?.map((lm) => ({
            x: Number(lm.x.toFixed(4)),
            y: Number(lm.y.toFixed(4)),
            z: Number((lm.z ?? 0).toFixed(4)),
            score: Number((lm.visibility ?? 1).toFixed(3)),
          })),
        }),
      );
    },
    [],
  );

  const isEdgeSyntheticMock = Boolean(
    connectionStatus === 'connected' && livePayload?.camera_mode === 'synthetic_mock',
  );
  const isEdgeDirectCamera = Boolean(
    !isDemoMode &&
      connectionStatus === 'connected' &&
      livePayload &&
      livePayload.camera_mode !== 'live_browser_bridge' &&
      livePayload.camera_mode !== 'synthetic_mock',
  );
  const isEdgeLiveCameraActive = Boolean(
    !isDemoMode &&
      connectionStatus === 'connected' &&
      livePayload &&
      livePayload.camera_mode !== 'synthetic_mock',
  );
  const isHttpsRemoteOrigin =
    typeof window !== 'undefined' &&
    window.location.protocol === 'https:' &&
    !/^(localhost|127\.0\.0\.1)$/i.test(window.location.hostname);

  // In live mode, reflect Edge Vision's direct telemetry (including CAMERA_DISCONNECTED /
  // AI_MODEL_ERROR / NO_POSE_DETECTED states) instead of reverting to synthetic demoPayload.
  const payload: KioskDecisionPayload =
    !isDemoMode &&
    connectionStatus === 'connected' &&
    livePayload &&
    livePayload.camera_mode !== 'live_browser_bridge'
      ? livePayload
      : demoPayload;

  return {
    payload,
    connectionStatus,
    isDemoMode,
    isEdgeDirectCamera,
    isEdgeLiveCameraActive,
    isEdgeSyntheticMock,
    activeWsUrl,
    isHttpsRemoteOrigin,
    sendClientLandmarks,
    toggleDemoMode,
    exerciseOptions,
    selectedExercise,
    setSelectedExercise,
    demoPreset,
    setDemoPreset,
    setDemoPresetState,
    demoRomDeg,
    setDemoRomDeg,
    demoTrunkTiltDeg,
    setDemoTrunkTiltDeg,
    demoDistanceM,
    setDemoDistanceM,
    demoValidReps,
    setDemoValidReps,
    demoInvalidReps,
    demoHoldRemainingSec,
    setDemoHoldRemainingSec,
    demoFaceMatched,
    demoStreakFrames,
    autoCycleDemo,
    setAutoCycleDemo,
    triggerEmergencyStop,
    resetDemoSession,
    completeCurrentExercise,
  };
}

