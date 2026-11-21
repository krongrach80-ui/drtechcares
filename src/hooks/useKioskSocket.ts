import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  CLINICAL_EXERCISES,
  mergeDuplicateSideExercises,
  normalizeStrokeExerciseId,
  parseExerciseCodeAndSide,
  resolveActiveExerciseSide,
  resolveSideAdjustedExerciseCopy,
  sortExercisesStrokeUpperFirst,
  type ExerciseConfig,
  type ExerciseSideCode,
} from '@drtechcares/pose-engine';
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
  universalHealthLinkCode?: string | null;
  similarity?: number;
  latency_ms?: number;
  max_safe_rom_deg?: number;
  affected_side?: 'LEFT' | 'RIGHT' | 'BOTH' | 'NONE';
  stroke_type?: string;
  motor_assessment_score?: number;
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
  safety_guard_alert?: string | null;
  safety_guard_banner_th?: string | null;
  spasticity_warning?: boolean;
  contraindication_clamped?: boolean;
  consecutive_compensations?: number;
  fatigue_rest_recommended?: boolean;
  fatigue_rest_remaining_sec?: number;
  suggested_reduced_rom_deg?: number | null;
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

export type ExpectedMovementPlane =
  | 'flexion'
  | 'abduction'
  | 'rotation'
  | 'cross_body'
  | 'scap_protract'
  | 'hand_to_head'
  | 'forearm_turn'
  | 'wrist_finger'
  | 'thumb_pinch'
  | 'shrug_roll'
  | 'neck_turn'
  | 'bilateral'
  | 'bilateral_push'
  | 'any';

export interface ExercisePrescriptionOption {
  exercise_id: string;
  catalog_exercise_id?: string;
  base_code?: string;
  effective_code?: string;
  prescribed_side?: ExerciseSideCode;
  schedule_id?: string;
  name_th: string;
  name_en: string;
  instruction_th?: string;
  instruction_en?: string;
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

const VALID_EXPECTED_PLANES = new Set<ExpectedMovementPlane>([
  'flexion',
  'abduction',
  'rotation',
  'cross_body',
  'scap_protract',
  'hand_to_head',
  'forearm_turn',
  'wrist_finger',
  'thumb_pinch',
  'shrug_roll',
  'neck_turn',
  'bilateral',
  'bilateral_push',
  'any',
]);

export function resolveExpectedPlaneForExercise(item: {
  exercise_id?: string;
  catalog_exercise_id?: string;
  name_th?: string;
  name_en?: string;
  target_joint?: string;
  expected_plane?: ExpectedMovementPlane;
}): ExpectedMovementPlane {
  if (item.expected_plane && VALID_EXPECTED_PLANES.has(item.expected_plane)) {
    return item.expected_plane;
  }
  const rawId = (item.catalog_exercise_id || item.exercise_id || '').trim();
  const normalizedId = rawId ? normalizeStrokeExerciseId(rawId) : '';
  const found = rawId
    ? CLINICAL_EXERCISES.find(
      (c) =>
        c.id === normalizedId ||
        c.code === normalizedId ||
        c.id === item.catalog_exercise_id ||
        c.code === item.catalog_exercise_id ||
        c.id === item.exercise_id ||
        c.code === item.exercise_id,
    )
    : undefined;
  if (found?.detection.expectedPlane) {
    return found.detection.expectedPlane as ExpectedMovementPlane;
  }
  const combined = `${item.name_th ?? ''} ${item.name_en ?? ''} ${item.exercise_id ?? ''}`.toLowerCase();
  if (combined.includes('neck_turn') || combined.includes('หันคอ')) {
    return 'neck_turn';
  }
  if (combined.includes('thumb_pinch') || combined.includes('จีบนิ้ว') || combined.includes('pinch')) {
    return 'thumb_pinch';
  }
  if (combined.includes('forearm_turn') || combined.includes('คว่ำ') || combined.includes('หงายมือ')) {
    return 'forearm_turn';
  }
  if (combined.includes('hand_to_head') || combined.includes('แตะศีรษะ') || combined.includes('หวีผม')) {
    return 'hand_to_head';
  }
  if (combined.includes('scap_protract') || combined.includes('ยื่นมือเอื้อมไปด้านหน้า') || combined.includes('protract')) {
    return 'scap_protract';
  }
  if (combined.includes('cross_body') || combined.includes('ไขว้หน้าอก')) {
    return 'cross_body';
  }
  if (combined.includes('forward_push') || combined.includes('ดันแขนตรงไปด้านหน้า')) {
    return 'bilateral_push';
  }
  if (item.target_joint && item.target_joint !== 'left_shoulder' && item.target_joint !== 'right_shoulder') {
    return 'any';
  }
  if (combined.includes('rotation') || combined.includes('หมุนข้อไหล่')) {
    return 'rotation';
  }
  if (combined.includes('wrist') || combined.includes('finger') || combined.includes('กระดกข้อมือ')) {
    return 'wrist_finger';
  }
  if (combined.includes('shrug') || combined.includes('ยักไหล่')) {
    return 'shrug_roll';
  }
  if (combined.includes('bilateral') || combined.includes('ประสานมือ') || combined.includes('arm_raise')) {
    return 'bilateral';
  }
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
  if (expectedPlane === 'cross_body') {
    return {
      requiredPlaneTh: 'วาดแขนไขว้ผ่านหน้าอก',
      wrongPlaneTh: 'กางแขนออกด้านนอก',
      bannerAlertTh: 'กรุณาวาดแขนไขว้ผ่านหน้าอกเข้าหาไหล่ฝั่งตรงข้ามครับ',
      voicePromptTh: 'กรุณายกแขนวาดไขว้ผ่านหน้าอกแตะไหล่ฝั่งตรงข้ามนะครับ',
    };
  }
  if (expectedPlane === 'scap_protract') {
    return {
      requiredPlaneTh: 'เหยียดศอกตรงยื่นมือไปด้านหน้า',
      wrongPlaneTh: 'งอข้อศอกหรือเอียงตัว',
      bannerAlertTh: 'กรุณาเหยียดข้อศอกให้ตรงแล้วยื่นหัวไหล่ไปด้านหน้าระดับอกครับ',
      voicePromptTh: 'กรุณาเหยียดศอกตรงแล้วยื่นมือไปด้านหน้าระดับหน้าอกนะครับ',
    };
  }
  if (expectedPlane === 'bilateral_push') {
    return {
      requiredPlaneTh: 'ประสานมือดันเหยียดศอกไปด้านหน้า',
      wrongPlaneTh: 'ยกแขนไม่สมมาตร',
      bannerAlertTh: 'กรุณาประสานมือทั้ง 2 ข้างแล้วดันเหยียดข้อศอกตรงไปด้านหน้าครับ',
      voicePromptTh: 'กรุณาประสานมือแล้วดันเหยียดข้อศอกตรงไปด้านหน้านะครับ',
    };
  }
  if (expectedPlane === 'hand_to_head') {
    return {
      requiredPlaneTh: 'ยกมือขึ้นแตะศีรษะ',
      wrongPlaneTh: 'ก้มศีรษะลงหามือ',
      bannerAlertTh: 'ตั้งศีรษะให้ตรงแล้วยกมือขึ้นแตะบริเวณศีรษะนะครับ',
      voicePromptTh: 'ตั้งศีรษะให้ตรงแล้วค่อยๆ ยกมือขึ้นแตะศีรษะนะครับ',
    };
  }
  if (expectedPlane === 'forearm_turn') {
    return {
      requiredPlaneTh: 'คว่ำและหงายฝ่ามือในกรอบกล้อง',
      wrongPlaneTh: 'กางข้อศอกหนีลำตัว',
      bannerAlertTh: 'กรุณาแนบศอกข้างลำตัวและพลิกคว่ำ-หงายฝ่ามือช้าๆ ครับ',
      voicePromptTh: 'กรุณาหนีบข้อศอกข้างลำตัวและพลิกหงายฝ่ามือสลับคว่ำมือนะครับ',
    };
  }
  if (expectedPlane === 'thumb_pinch') {
    return {
      requiredPlaneTh: 'จีบปลายนิ้วโป้งแตะนิ้วเป็นรูปวงกลม',
      wrongPlaneTh: 'มองไม่เห็นปลายนิ้วชัดเจน',
      bannerAlertTh: 'กรุณายกมือในกรอบกล้องและจีบปลายนิ้วโป้งแตะนิ้วทีละนิ้วครับ',
      voicePromptTh: 'กรุณายกมือขึ้นให้ชัดแล้วจีบปลายนิ้วโป้งแตะปลายนิ้วเป็นวงกลมนะครับ',
    };
  }
  if (expectedPlane === 'neck_turn') {
    return {
      requiredPlaneTh: 'หันศีรษะมองซ้าย-ขวาอย่างช้าๆ',
      wrongPlaneTh: 'หันไหล่ตามศีรษะ',
      bannerAlertTh: 'กรุณาตั้งไหล่นิ่งและค่อยๆ หันศีรษะมองซ้าย-ขวาอย่างนุ่มนวลครับ',
      voicePromptTh: 'ตั้งไหล่ให้นิ่งแล้วค่อยๆ หันศีรษะมองซ้ายขวาสลับกันนะครับ',
    };
  }
  return {
    requiredPlaneTh: 'เคลื่อนไหวตามทิศทางที่กำหนด',
    wrongPlaneTh: 'เคลื่อนไหวผิดทิศทาง',
    bannerAlertTh: 'กรุณาเคลื่อนไหวตามทิศทางและท่าทางที่กำหนดครับ',
    voicePromptTh: 'กรุณาเคลื่อนไหวตามทิศทางที่กำหนดนะครับ',
  };
}

export function getPrescribedLimbPromptTh(
  targetJoint: ExercisePrescriptionOption['target_joint'],
  expectedPlane?: ExpectedMovementPlane,
  prescribedSide?: ExerciseSideCode,
): {
  requiredLimbTh: string;
  wrongLimbTh: string;
  bannerAlertTh: string;
  voicePromptTh: string;
} {
  if (
    prescribedSide === 'BOTH' ||
    expectedPlane === 'bilateral' ||
    expectedPlane === 'bilateral_push'
  ) {
    return {
      requiredLimbTh: 'แขนทั้ง 2 ข้างพร้อมกัน',
      wrongLimbTh: 'แขนเพียงข้างเดียว',
      bannerAlertTh: 'ท่านี้ฝึกทั้งสองข้างพร้อมกัน กรุณายกแขนทั้ง 2 ข้างขึ้นพร้อมกันครับ',
      voicePromptTh: 'ท่านี้ฝึกทั้งสองข้างพร้อมกัน กรุณายกแขนทั้ง 2 ข้างขึ้นพร้อมกันนะครับ',
    };
  }
  if (prescribedSide === 'LEFT' || (prescribedSide !== 'RIGHT' && targetJoint === 'left_shoulder')) {
    return {
      requiredLimbTh: 'แขนซ้าย',
      wrongLimbTh: 'แขนขวา',
      bannerAlertTh: 'ท่านี้ฝึกเฉพาะข้างซ้าย! กรุณาลดแขนขวาลง และยกเฉพาะแขนซ้ายข้างเดียว',
      voicePromptTh: 'ท่านี้ฝึกเฉพาะข้างซ้าย กรุณาลดแขนขวาลง และยกเฉพาะแขนซ้ายข้างเดียวนะครับ',
    };
  }
  if (prescribedSide === 'RIGHT' || targetJoint === 'right_shoulder') {
    return {
      requiredLimbTh: 'แขนขวา',
      wrongLimbTh: 'แขนซ้าย',
      bannerAlertTh: 'ท่านี้ฝึกเฉพาะข้างขวา! กรุณาลดแขนซ้ายลง และยกเฉพาะแขนขวาข้างเดียว',
      voicePromptTh: 'ท่านี้ฝึกเฉพาะข้างขวา กรุณาลดแขนซ้ายลง และยกเฉพาะแขนขวาข้างเดียวนะครับ',
    };
  }
  return {
    requiredLimbTh: 'ศีรษะ/ลำตัวและแขนทั้ง 2 ข้าง',
    wrongLimbTh: 'ไม่สมมาตร',
    bannerAlertTh: 'กรุณาทำท่าทางตามแนวแกนกลางและข้างที่กำหนด',
    voicePromptTh: 'กรุณาทำท่าตามทิศทางที่กำหนดนะครับ',
  };
}

export type ConnectionStatus = 'connecting' | 'connected' | 'offline';

export const EXERCISE_OPTIONS: ExercisePrescriptionOption[] = [
  {
    exercise_id: 'STROKE_SH_FLEX_RIGHT',
    catalog_exercise_id: 'STROKE_SH_FLEX',
    base_code: 'STROKE_SH_FLEX',
    effective_code: 'STROKE_SH_FLEX_RIGHT',
    prescribed_side: 'RIGHT',
    name_th: '[Stroke ADL] ยกแขนขวาไปด้านหน้า (ฝึกเอื้อม/สอดแขนเสื้อ)',
    name_en: 'Stroke ADL: Forward Shoulder Flexion (Right)',
    target_joint: 'right_shoulder',
    expected_plane: 'flexion',
    target_rom_deg: 125,
    hold_sec: 3,
    target_reps: 10,
  },
  {
    exercise_id: 'STROKE_SH_ABD_RIGHT',
    catalog_exercise_id: 'STROKE_SH_ABD',
    base_code: 'STROKE_SH_ABD',
    effective_code: 'STROKE_SH_ABD_RIGHT',
    prescribed_side: 'RIGHT',
    name_th: '[Stroke ADL] กางแขนขวาด้านข้าง (ฝึกกางแขนสวมเสื้อ)',
    name_en: 'Stroke ADL: Shoulder Abduction (Right)',
    target_joint: 'right_shoulder',
    expected_plane: 'abduction',
    target_rom_deg: 115,
    hold_sec: 3,
    target_reps: 10,
  },
  {
    exercise_id: 'STROKE_BOBATH_ARM_RAISE',
    catalog_exercise_id: 'STROKE_BOBATH_ARM_RAISE',
    base_code: 'STROKE_BOBATH_ARM_RAISE',
    effective_code: 'STROKE_BOBATH_ARM_RAISE_BOTH',
    prescribed_side: 'BOTH',
    name_th: '[Stroke & สธ.] ประสานมือสองข้างชูขึ้นเหนือศีรษะ (ท่า Bobath พยุงแขนอ่อนแรง)',
    name_en: 'Stroke & MOPH: Bobath Bilateral Clasped Arm Raise',
    target_joint: 'right_shoulder',
    expected_plane: 'bilateral',
    target_rom_deg: 135,
    hold_sec: 3,
    target_reps: 10,
  },
];

export const DEMO_PATIENT: PatientSummary = {
  id: 'PT-8841',
  name: 'คุณสมชาย ใจดี',
};

export const POSE_TEST_PATIENT: PatientSummary = {
  id: 'HN-TEST-POSE',
  hn: 'HN-TEST-POSE',
  name: 'ผู้ทดลองระบบ (โหมดทดสอบท่ากายภาพ 15 ท่า)',
  max_safe_rom_deg: 180,
  affected_side: 'BOTH',
  stroke_type: 'SANDBOX_POSE_TEST',
  motor_assessment_score: 6,
};

function findCatalogExerciseConfig(rawCodeOrId?: string | null): ExerciseConfig | undefined {
  const raw = (rawCodeOrId ?? '').trim();
  if (!raw) return undefined;
  const parsed = parseExerciseCodeAndSide(raw);
  const canonicalId = normalizeStrokeExerciseId(parsed.baseCode || raw);
  return CLINICAL_EXERCISES.find(
    (c) =>
      c.id === canonicalId ||
      c.code === canonicalId ||
      c.id === parsed.baseCode ||
      c.code === parsed.baseCode ||
      c.id === raw ||
      c.code === raw,
  );
}

/** Builds all 15 unified clinical exercises in 1..15 order for Pose Test Mode (10 reps/exercise). */
export function buildAll15ClinicalExercisePrescriptions(
  catalogOverrides?: ReadonlyArray<{
    id: string;
    code: string;
    category?: string;
    name: { th: string; en: string; cn?: string };
    instruction: { th: string; en: string; cn?: string };
    detection?: ExerciseConfig['detection'];
  }>,
): ExercisePrescriptionOption[] {
  const sourcePool: ExerciseConfig[] =
    catalogOverrides && catalogOverrides.length > 0
      ? mergeDuplicateSideExercises(
          sortExercisesStrokeUpperFirst(
            catalogOverrides.map((ex) => {
              const master = findCatalogExerciseConfig(ex.code || ex.id);
              return (
                master ?? {
                  id: ex.id,
                  code: ex.code,
                  category: (ex.category as ExerciseConfig['category']) ?? 'upper_limb',
                  clinicalFocus: 'stroke_adl',
                  name: {
                    th: ex.name.th,
                    en: ex.name.en,
                    cn: ex.name.cn ?? ex.name.en,
                  },
                  instruction: {
                    th: ex.instruction.th,
                    en: ex.instruction.en,
                    cn: ex.instruction.cn ?? ex.instruction.en,
                  },
                  detection: ex.detection ?? CLINICAL_EXERCISES[0].detection,
                }
              );
            }),
          ),
        )
      : [...CLINICAL_EXERCISES];

  const todayItems: TodayPrescribedExerciseItem[] = sourcePool.map((ex, idx) => {
    const defaultSide: ExerciseSideCode =
      ex.detection.side === 'both'
        ? 'BOTH'
        : ex.detection.side === 'left'
          ? 'LEFT'
          : 'RIGHT';
    const baseCode = ex.code || ex.id;
    return {
      id: `pose-test-${idx + 1}-${baseCode}_${defaultSide}`,
      exerciseId: ex.id,
      exerciseCode: baseCode,
      effectiveCode: `${baseCode}_${defaultSide}`,
      side: defaultSide,
      nameTh: ex.name.th,
      nameEn: ex.name.en,
      instructionTh: ex.instruction.th,
      instructionEn: ex.instruction.en,
      targetJoint: ex.detection.jointGroup,
      activityType: 'physio_exercise' as const,
      holdSec: ex.detection.holdSec ?? 3,
      sets: 1,
      reps: 10,
      status: 'ACTIVE' as const,
    };
  });

  return mapTodayExercisesToPrescriptions(todayItems).map((rx, idx) => {
    const base = sourcePool[idx];
    const side: ExerciseSideCode = rx.prescribed_side ?? 'RIGHT';
    const baseCode = base?.code ?? rx.base_code ?? rx.catalog_exercise_id ?? rx.exercise_id;
    return {
      ...rx,
      exercise_id: `${baseCode}_${side}`,
      catalog_exercise_id: base?.id ?? rx.catalog_exercise_id ?? baseCode,
      base_code: baseCode,
      effective_code: `${baseCode}_${side}`,
      prescribed_side: side,
      target_reps: 10,
      target_sets: 1,
    };
  });
}

/** Switches an exercise prescription dynamically to LEFT, RIGHT, or BOTH while updating copy & target joint. */
export function applyExerciseSideOverride(
  rx: ExercisePrescriptionOption,
  nextSide: ExerciseSideCode,
): ExercisePrescriptionOption {
  const baseCode =
    rx.base_code ||
    rx.catalog_exercise_id ||
    parseExerciseCodeAndSide(rx.effective_code || rx.exercise_id).baseCode ||
    rx.exercise_id;
  const master = findCatalogExerciseConfig(baseCode) ?? findCatalogExerciseConfig(rx.exercise_id);
  const copy = master ? resolveSideAdjustedExerciseCopy(master, nextSide) : null;
  const keepSpine = rx.target_joint === 'spine' || master?.detection.jointGroup === 'neck';
  const nextTargetJoint: ExercisePrescriptionOption['target_joint'] = keepSpine
    ? 'spine'
    : nextSide === 'LEFT'
      ? 'left_shoulder'
      : 'right_shoulder';

  const masterPlane = master?.detection.expectedPlane as ExpectedMovementPlane | undefined;
  const nextExpectedPlane: ExpectedMovementPlane | undefined =
    nextSide === 'BOTH'
      ? masterPlane ?? rx.expected_plane
      : masterPlane === 'bilateral'
        ? 'flexion'
        : rx.expected_plane === 'bilateral'
          ? 'flexion'
          : masterPlane ?? rx.expected_plane;

  return {
    ...rx,
    exercise_id: `${baseCode}_${nextSide}`,
    catalog_exercise_id: master?.id ?? rx.catalog_exercise_id ?? baseCode,
    base_code: baseCode,
    effective_code: `${baseCode}_${nextSide}`,
    prescribed_side: nextSide,
    target_joint: nextTargetJoint,
    expected_plane: nextExpectedPlane,
    name_th: copy?.nameTh ?? rx.name_th,
    name_en: copy?.nameEn ?? rx.name_en,
    instruction_th: copy?.instructionTh ?? rx.instruction_th,
    instruction_en: copy?.instructionEn ?? rx.instruction_en,
  };
}

function resolveTargetJointForTodayExercise(
  item: TodayPrescribedExerciseItem,
  catalogMatch?: (typeof CLINICAL_EXERCISES)[number],
  resolvedSide?: ExerciseSideCode,
): Exclude<HighlightJoint, 'none'> {
  if (catalogMatch) {
    const group = catalogMatch.detection.jointGroup;
    if (group === 'trunk' || group === 'neck') {
      return 'spine';
    }
    if (resolvedSide === 'LEFT') return 'left_shoulder';
    if (resolvedSide === 'RIGHT') return 'right_shoulder';
    return catalogMatch.detection.side === 'left' ? 'left_shoulder' : 'right_shoulder';
  }
  const combined = `${item.nameTh} ${item.nameEn} ${item.targetJoint}`.toLowerCase();
  if (combined.includes('คอ') || combined.includes('neck') || combined.includes('spine')) {
    return 'spine';
  }
  if (resolvedSide === 'LEFT' || combined.includes('ซ้าย') || combined.includes('left')) {
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
    const parsed = parseExerciseCodeAndSide(
      item.exerciseCode || item.effectiveCode || item.exerciseId,
      item.side,
    );
    const canonicalId = normalizeStrokeExerciseId(parsed.baseCode || item.exerciseId);
    const found = CLINICAL_EXERCISES.find(
      (c) =>
        c.id === canonicalId ||
        c.code === canonicalId ||
        c.id === parsed.baseCode ||
        c.code === parsed.baseCode ||
        c.id === item.exerciseId ||
        c.code === item.exerciseId,
    );
    const inferredFromMeta = resolveActiveExerciseSide(undefined, item);
    const resolvedSide = parsed.hasExplicitSide ? parsed.side : inferredFromMeta ?? parsed.side;
    const effectiveCode =
      resolvedSide !== parsed.side && parsed.baseCode
        ? `${parsed.baseCode}_${resolvedSide}`
        : parsed.effectiveCode;
    const target_joint = resolveTargetJointForTodayExercise(item, found, resolvedSide);
    const isLowRangeUpper =
      found &&
      (found.detection.jointGroup === 'neck' ||
        found.detection.jointGroup === 'wrist' ||
        found.detection.expectedPlane === 'rotation' ||
        found.detection.expectedPlane === 'scap_protract' ||
        found.detection.expectedPlane === 'forearm_turn' ||
        found.detection.expectedPlane === 'thumb_pinch' ||
        found.detection.expectedPlane === 'neck_turn' ||
        found.detection.expectedPlane === 'shrug_roll');
    const defaultMidDeg = found
      ? found.detection.targetRomDeg ?? Math.round((found.detection.targetMinDeg + found.detection.targetMaxDeg) / 2)
      : 125;
    const target_rom_deg = found
      ? isLowRangeUpper
        ? Math.max(25, defaultMidDeg)
        : Math.min(165, Math.max(85, defaultMidDeg))
      : 125;

    const sideCopy = found ? resolveSideAdjustedExerciseCopy(found, resolvedSide) : null;

    return {
      exercise_id: item.id || effectiveCode,
      catalog_exercise_id: found?.id ?? canonicalId,
      base_code: found?.code ?? canonicalId,
      effective_code: effectiveCode,
      prescribed_side: resolvedSide,
      schedule_id: item.id,
      name_th: sideCopy?.nameTh ?? found?.name.th ?? item.nameTh,
      name_en: sideCopy?.nameEn ?? found?.name.en ?? item.nameEn,
      instruction_th: found?.instruction.th ?? item.instructionTh,
      instruction_en: found?.instruction.en ?? item.instructionEn,
      target_joint,
      expected_plane:
        (found?.detection.expectedPlane as ExpectedMovementPlane | undefined) ??
        resolveExpectedPlaneForExercise({
          exercise_id: canonicalId,
          catalog_exercise_id: found?.id ?? canonicalId,
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
          { bbox: [420, 80, 860, 680], label: 'ไม่ใช่คนไข้ที่ลงทะเบียน', is_authorized: false },
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
    const expectedPlane = resolveExpectedPlaneForExercise(exercise);
    const activeSide = resolveActiveExerciseSide(
      exercise,
      undefined,
      exercise.target_joint,
      expectedPlane,
    );
    const limbInfo = getPrescribedLimbPromptTh(
      exercise.target_joint,
      expectedPlane,
      activeSide,
    );
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

  const trunkLeaning =
    preset === 'TRUNK_LEAN' ||
    (preset !== 'HOLD_PHASE' && Math.abs(trunkTiltDeg) > 10.0) ||
    (preset === 'HOLD_PHASE' && Math.abs(trunkTiltDeg) > 18.0);
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
        trunk_tilt_deg: Number(trunkTiltDeg.toFixed(1)),
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
    const rem = Math.max(0, Number(holdRemainingSec.toFixed(1)));
    return {
      kiosk_state: 'HOLD_PHASE',
      patient: activePatient,
      telemetry_analysis: {
        current_rom_deg: clampedRom,
        target_rom_deg: targetRom,
        rom_completion_pct: romPct,
        trunk_tilt_deg: Number(trunkTiltDeg.toFixed(1)),
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
      tts_voice_th:
        rem <= 0
          ? isLowerBody
            ? 'ค้างครบเวลาแล้ว ค่อยๆ เอาขาลงเพื่อนับเป็น 1 ครั้งครับ'
            : 'ค้างครบเวลาแล้ว ค่อยๆ เอาแขนลงเพื่อนับเป็น 1 ครั้งครับ'
          : `ดีมากครับ ค้างท่าไว้ ${Math.ceil(rem)} วินาที`,
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
    initialSelectedExerciseId?: string | null;
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

  const resolveMatchingExercise = useCallback(
    (codeOrId?: string | null): ExercisePrescriptionOption | undefined => {
      const clean = (codeOrId ?? '').trim().toUpperCase();
      if (!clean) return undefined;
      const parsedClean = parseExerciseCodeAndSide(clean).baseCode;
      return exerciseOptions.find((opt) => {
        const optIds = [opt.exercise_id, opt.base_code, opt.catalog_exercise_id, opt.effective_code]
          .filter((v): v is string => Boolean(v))
          .map((v) => v.toUpperCase());
        return (
          optIds.includes(clean) ||
          optIds.includes(parsedClean) ||
          optIds.some((v) => parseExerciseCodeAndSide(v).baseCode === parsedClean)
        );
      });
    },
    [exerciseOptions],
  );

  const [selectedExercise, setSelectedExerciseState] = useState<ExercisePrescriptionOption>(
    () => resolveMatchingExercise(options?.initialSelectedExerciseId) ?? exerciseOptions[0],
  );
  const lastAppliedInitialExRef = useRef<string | null | undefined>(options?.initialSelectedExerciseId);
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

  const patientOverrideRef = useRef(options?.patientOverride);
  patientOverrideRef.current = options?.patientOverride;
  const selectedExerciseRef = useRef(selectedExercise);
  selectedExerciseRef.current = selectedExercise;

  const sendExercisePrescription = useCallback((rx: ExercisePrescriptionOption) => {
    const ws = wsRef.current;
    const pOverride = patientOverrideRef.current;
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
            exercise_id: rx.effective_code ?? rx.catalog_exercise_id ?? rx.exercise_id,
            base_code: rx.base_code ?? rx.catalog_exercise_id ?? undefined,
            prescribed_side: rx.prescribed_side ?? undefined,
            exercise_name: `${rx.name_en} (${rx.name_th})`,
            target_joint: rx.target_joint,
            target_rom_deg:
              typeof pOverride?.max_safe_rom_deg === 'number' &&
                pOverride.max_safe_rom_deg > 0 &&
                !rx.target_joint.includes('knee')
                ? Math.min(rx.target_rom_deg, pOverride.max_safe_rom_deg)
                : rx.target_rom_deg,
            hold_sec: rx.hold_sec,
            target_reps: rx.target_reps,
            max_safe_rom_deg: pOverride?.max_safe_rom_deg ?? undefined,
            affected_side: pOverride?.affected_side ?? undefined,
            stroke_type: pOverride?.stroke_type ?? undefined,
            motor_assessment_score: pOverride?.motor_assessment_score ?? undefined,
          },
        }),
      );
    }
  }, []);

  useEffect(() => {
    if (options?.initialSelectedExerciseId !== lastAppliedInitialExRef.current) {
      lastAppliedInitialExRef.current = options?.initialSelectedExerciseId;
      const matched = resolveMatchingExercise(options?.initialSelectedExerciseId);
      if (matched && matched.exercise_id !== selectedExercise.exercise_id) {
        setSelectedExerciseState(matched);
        setDemoValidReps(0);
        setDemoInvalidReps(0);
        setDemoHoldRemainingSec(matched.hold_sec);
        setDemoPresetState('ACTIVE_REP');
        sendExercisePrescription(matched);
        return;
      }
    }
    if (!exerciseOptions.some((opt) => opt.exercise_id === selectedExercise.exercise_id)) {
      const first = resolveMatchingExercise(options?.initialSelectedExerciseId) ?? exerciseOptions[0];
      setSelectedExerciseState(first);
      setDemoValidReps(0);
      setDemoInvalidReps(0);
      setDemoHoldRemainingSec(first.hold_sec);
      setDemoPresetState('ACTIVE_REP');
      sendExercisePrescription(first);
    }
  }, [
    exerciseOptions,
    selectedExercise.exercise_id,
    options?.initialSelectedExerciseId,
    resolveMatchingExercise,
    sendExercisePrescription,
  ]);

  const setSelectedExercise = useCallback(
    (rx: ExercisePrescriptionOption) => {
      setSelectedExerciseState(rx);
      setDemoValidReps(0);
      setDemoInvalidReps(0);
      setDemoPresetState('ACTIVE_REP');
      setDemoHoldRemainingSec(rx.hold_sec);
      // Clear stale rep counter on livePayload so switching exercises in Edge mode never cascade-completes
      setLivePayload((prev) =>
        prev
          ? {
              ...prev,
              kiosk_state: 'ACTIVE_REP',
              rep_counter: {
                valid_reps: 0,
                invalid_reps: 0,
                hold_remaining_sec: rx.hold_sec,
              },
            }
          : prev,
      );
      sendExercisePrescription(rx);
    },
    [sendExercisePrescription],
  );

  // WebSocket client with automatic exponential backoff reconnect
  useEffect(() => {
    const isRemoteHttps =
      typeof window !== 'undefined' &&
      window.location.protocol === 'https:' &&
      !/^(localhost|127\.0\.0\.1)$/i.test(window.location.hostname);

    const allowedWsCandidates = wsCandidates.filter(
      (candidate) => !isRemoteHttps || !/^ws:\/\//i.test(candidate.trim()),
    );
    if (allowedWsCandidates.length === 0) {
      // Avoid mixed-content WebSocket blocks and reconnect loops on remote HTTPS cloud deployments
      setConnectionStatus('offline');
      return;
    }

    const MAX_RECONNECT_ATTEMPTS = 5;
    let disposed = false;
    let attempt = 0;
    let reconnectTimer: number | null = null;
    let healthAbortController: AbortController | null = null;

    const scheduleReconnect = () => {
      if (disposed || attempt >= MAX_RECONNECT_ATTEMPTS) return;
      if (reconnectTimer !== null) {
        window.clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
      const delayMs = Math.min(10000, 1500 * Math.pow(1.6, attempt));
      attempt += 1;
      reconnectTimer = window.setTimeout(connect, delayMs);
    };

    const connect = () => {
      if (disposed) return;
      setConnectionStatus('connecting');

      try {
        const targetWsUrl = allowedWsCandidates[attempt % allowedWsCandidates.length] ?? wsUrl;
        setActiveWsUrl(targetWsUrl);
        // Trigger CORS + Private Network Access preflight prior to loopback WebSocket upgrade
        const healthUrl = targetWsUrl
          .replace(/^ws:/i, 'http:')
          .replace(/^wss:/i, 'https:')
          .replace(/\/ws\/kiosk.*$/i, '/health');
        healthAbortController?.abort();
        healthAbortController = typeof AbortController !== 'undefined' ? new AbortController() : null;
        void fetch(
          healthUrl,
          healthAbortController ? { mode: 'cors', signal: healthAbortController.signal } : { mode: 'cors' },
        ).catch(() => undefined);
        const ws = new WebSocket(targetWsUrl);
        wsRef.current = ws;

        ws.onopen = () => {
          if (disposed || wsRef.current !== ws) return;
          attempt = 0;
          setConnectionStatus('connected');
          if (!userToggledModeRef.current) {
            setIsDemoMode(false);
          }
          ws.send(JSON.stringify({ type: 'config', stream_video: true }));
          const pOverride = patientOverrideRef.current;
          if (pOverride?.id) {
            ws.send(
              JSON.stringify({
                type: 'lock_patient',
                patient: {
                  id: pOverride.id,
                  hn: pOverride.hn ?? '',
                  full_name: pOverride.name ?? pOverride.id,
                },
                start_calibration: false,
              }),
            );
          }
          sendExercisePrescription(selectedExerciseRef.current);
        };

        ws.onmessage = (event) => {
          if (disposed || wsRef.current !== ws) return;
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
          if (disposed || wsRef.current !== ws) return;
          ws.close();
        };

        ws.onclose = () => {
          if (wsRef.current !== ws) return;
          wsRef.current = null;
          setConnectionStatus('offline');
          scheduleReconnect();
        };
      } catch {
        setConnectionStatus('offline');
        scheduleReconnect();
      }
    };

    connect();
    return () => {
      disposed = true;
      healthAbortController?.abort();
      if (reconnectTimer !== null) {
        window.clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
      if (wsRef.current) {
        const activeWs = wsRef.current;
        wsRef.current = null;
        activeWs.close();
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

