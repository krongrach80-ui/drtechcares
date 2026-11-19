import React, { useEffect, useMemo, useRef, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import {
  FaceLandmarker,
  FilesetResolver,
  HandLandmarker,
  PoseLandmarker,
  type NormalizedLandmark,
} from '@mediapipe/tasks-vision';
import {
  alignFaceTo112,
  computeEyeRollAngleDeg,
  evaluateLivenessAndPose,
  extractArcFaceOnnxEmbedding,
  extractGeometricAnthropometricRatios,
  isArcFaceSessionReady,
  preloadArcFaceOnnx,
  selectClinicalCameraDeviceId,
  selectPrimaryFace,
} from '@drtechcares/face-client';
import {
  detectBiometricVectorEngine,
} from '@drtechcares/smart-biometric-core';
import {
  BIOMETRIC_ENGINE_ID,
  captureCanonicalFaceSnapshot,
  extractCanonicalFaceDescriptor,
} from '../lib/biometricEngine';
import {
  computeObjectCoverLayout,
  computeOwnerFaceBoxMetrics,
  isPoseSkeletonLinkedToOwnerFace,
  verifyOwnerFace1to1,
  type OwnerBiometricTemplate,
  type OwnerFaceBoxMetrics,
  type OwnerLockStatus,
} from '../lib/ownerBiometricLock';
import {
  AutoBoneBaselineCalibrator,
  ClinicalHandOneEuroFilter,
  ClinicalOneEuroLandmarkFilter,
  HOLD_COMPLETED_WAIT_LOWER_BANNER_TH,
  HAND21_CONNECTIONS,
  PHASE_2_LOWER_ARM_PROMPT_BANNER_TH,
  POSE_INDEX,
  computeJointAngle3D,
  evaluateActiveWristsLowered,
  formatRepScoredBannerTh,
  resolveActiveExerciseSide,
  solveStrokeClinicalJointBiomechanics,
  stepStrokeHoldAndRepStateMachine,
  type ExerciseSideCode,
  type StrokeAnatomyBaseline,
  type StrokeHoldRepState,
  type StrokeSideViolationCode,
  type StrokeTargetOverlayHint,
} from '@drtechcares/pose-engine';
import {
  HAND_OUT_OF_FRAME_BADGE_TH,
  SKELETON_JOINT_MIN_SCORE,
  applyKinematicSkeletonLock,
  computeCanvasShoulderWidth,
  evaluatePoseCanvasSkeletonSafety,
  isJointValidInFrame,
  mapVideoPointToCoverCanvas,
  validateTetheredHandLandmarks21,
} from '../lib/poseSkeletonSafety';
import { inferMascotPoseFromSpeechText } from '@drtechcares/kiosk-input';
import {
  Activity,
  AlertOctagon,
  AlertTriangle,
  ArrowLeft,
  ArrowRight,
  Camera,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Crosshair,
  Eye,
  EyeOff,
  Gamepad2,
  Mic,
  Pause,
  Play,
  PlusCircle,
  RotateCcw,
  Ruler,
  Shield,
  SkipBack,
  SkipForward,
  SlidersHorizontal,
  Timer,
  Volume2,
  VolumeX,
  X,
} from 'lucide-react';
import {
  ExpandedCoachModal,
  Zone2PipCoachOverlay,
  Zone4CoachStageAndSteps,
} from './AnimatedRehabCoachModel';
import {
  DrTechCareMascot,
  MASCOT_ENCOURAGING_DURATION_MS,
  MASCOT_GREETING_DURATION_MS,
  resolveMascotPose,
} from './DrTechCareMascot';
import { resolveExercisePostureSpec } from './ExercisePostureGuide';
import { FaceScanOverlay } from './FaceScanOverlay';
import { SkeletonCanvas } from './SkeletonCanvas';
import {
  VoiceCoachBanner,
  cancelAllKioskVoiceSpeech,
  readStoredKioskVoiceMuted,
  sanitizeSpeechText,
  subscribeKioskVoiceMuted,
  translateVoiceCoachText,
  writeStoredKioskVoiceMuted,
  type VoiceCoachLanguage,
} from './VoiceCoachBanner';
import {
  POSE_TEST_PATIENT,
  applyExerciseSideOverride,
  buildAll15ClinicalExercisePrescriptions,
  getPrescribedLimbPromptTh,
  getPrescribedPlanePromptTh,
  resolveExpectedPlaneForExercise,
  useKioskSocket,
  type DemoPresetId,
  type ExercisePrescriptionOption,
  type ExpectedMovementPlane,
  type PatientSummary,
} from '../hooks/useKioskSocket';
import { isUpperBodyVisibleInFrame } from '../hooks/useEdgeVisionSocket';

export {
  isUpperBodyVisibleInFrame,
  mapVideoPointToCoverCanvas,
  isJointValidInFrame,
  applyKinematicSkeletonLock,
  validateTetheredHandLandmarks21,
  evaluatePoseCanvasSkeletonSafety,
  HAND_OUT_OF_FRAME_BADGE_TH,
};

interface SmartRehabKioskUIProps {
  patient?: PatientSummary | null;
  ownerBiometrics?: OwnerBiometricTemplate | null;
  prescribedExercises?: ExercisePrescriptionOption[];
  isStaffAuthorized?: boolean;
  isPoseTestMode?: boolean;
  initialExerciseCode?: string | null;
  onBackToAdminPortal?: () => void;
  onExerciseCompleted?: (exercise: ExercisePrescriptionOption) => void;
  onAllExercisesCompleted?: () => void;
  onEmergencyStop?: (info: { reason: string; patientId?: string; exerciseNameTh?: string }) => void;
  onOpenStaffPortal?: () => void;
  onOpenLegacyKiosk?: () => void;
}

const PRIVACY_MODE_STORAGE_KEY = 'drtechcares_kiosk_privacy_v2';
const SKELETON_OVERLAY_STORAGE_KEY = 'drtechcares_kiosk_show_skeleton_v1';

const DEMO_PRESETS: { id: DemoPresetId; label: string; shortLabel: string }[] = [
  { id: 'FACE_SCAN', label: 'Face Scan Login', shortLabel: '1. Face Scan' },
  { id: 'ACTIVE_REP', label: 'Active Exercise', shortLabel: '2. Active' },
  { id: 'TRUNK_LEAN', label: 'Cheating / Trunk Lean', shortLabel: '3. Trunk Lean' },
  { id: 'WRONG_LIMB', label: 'Wrong Arm (ยกผิดข้าง)', shortLabel: '3b. Wrong Arm' },
  { id: 'WRONG_PLANE', label: 'Wrong Plane (ยกผิดทิศทาง)', shortLabel: '3c. Wrong Plane' },
  { id: 'HOLD_PHASE', label: 'Hold Phase', shortLabel: '4. Hold' },
  { id: 'REP_COMPLETED', label: 'Rep Completed', shortLabel: '5. Rep +' },
  { id: 'SET_FINISHED', label: 'Set Finished', shortLabel: '6. Finished' },
  { id: 'MULTI_PERSON', label: 'Multi-Person Alert (2+ คน)', shortLabel: '7. Multi-Person' },
  { id: 'IMPOSTOR', label: 'Impostor Blocked (ผิดคน)', shortLabel: '8. Impostor' },
];

export type FormAssessmentRuleKey =
  | 'limb'
  | 'plane'
  | 'trunk'
  | 'posture'
  | 'velocity'
  | 'identity';

export interface FormAssessmentRulesConfig {
  enabled: boolean;
  limb: boolean;
  plane: boolean;
  trunk: boolean;
  posture: boolean;
  velocity: boolean;
  identity: boolean;
}

export const DEFAULT_FORM_ASSESSMENT_RULES: FormAssessmentRulesConfig = {
  enabled: true,
  limb: true,
  plane: true,
  trunk: true,
  posture: true,
  velocity: true,
  identity: true,
};

const FORM_ASSESSMENT_STORAGE_KEY = 'drtechcares_form_assessment_rules_v1';

export function isFormRuleActive(
  config: FormAssessmentRulesConfig,
  rule: FormAssessmentRuleKey,
): boolean {
  return Boolean(config.enabled && config[rule]);
}

export interface LiveJointMeasurement {
  romDeg: number | null;
  rawPrimaryRomDeg?: number | null;
  leftArmRom?: number | null;
  rightArmRom?: number | null;
  trunkTiltDeg: number;
  sideViolation?: StrokeSideViolationCode;
  sideViolationBannerTh?: string | null;
  wrongSideRaised: boolean;
  oppositeRomDeg: number | null;
  wrongSideBannerTh: string | null;
  wrongPlaneRaised: boolean;
  wrongPlaneBannerTh: string | null;
  xyzVector?: { x: number; y: number; z: number } | null;
  lowVisibilityAlertTh?: string | null;
  postureCompensated?: boolean;
  postureBannerTh?: string | null;
  elbowAngleDeg?: number | null;
  shoulderTiltDeg?: number;
  shoulderHikeRatio?: number;
  neckDropDeg?: number;
  compensations?: string[];
  overlayHint?: StrokeTargetOverlayHint | null;
  headY?: number | null;
  shoulderMidY?: number | null;
}

export interface TorsoAnchorState {
  leftShoulder: NormalizedLandmark;
  rightShoulder: NormalizedLandmark;
  nose?: NormalizedLandmark;
  spanX: number;
}

const LOWER_BODY_POSE_INDICES = [
  POSE_INDEX.LEFT_HIP,
  POSE_INDEX.RIGHT_HIP,
  POSE_INDEX.LEFT_KNEE,
  POSE_INDEX.RIGHT_KNEE,
  POSE_INDEX.LEFT_ANKLE,
  POSE_INDEX.RIGHT_ANKLE,
] as const;

/**
 * Validates anatomical torso landmarks and prevents MediaPipe self-occlusion collapse
 * (e.g., shoulders/hips jumping onto an outstretched hand approaching the camera lens).
 */
export function sanitizeAndStabilizePoseLandmarks(
  rawLandmarks: readonly NormalizedLandmark[],
  prevAnchor: TorsoAnchorState | null,
  targetJoint: ExercisePrescriptionOption['target_joint'],
): { landmarks: NormalizedLandmark[]; anchor: TorsoAnchorState | null; occludedFallback: boolean } {
  const stabilized = rawLandmarks.map((lm) => ({ ...lm }));
  const lSh = stabilized[POSE_INDEX.LEFT_SHOULDER];
  const rSh = stabilized[POSE_INDEX.RIGHT_SHOULDER];
  const nose = stabilized[POSE_INDEX.NOSE];
  const lWrist = stabilized[POSE_INDEX.LEFT_WRIST];
  const rWrist = stabilized[POSE_INDEX.RIGHT_WRIST];

  let occludedFallback = false;
  let nextAnchor = prevAnchor;

  const hasShoulderVis = Boolean(
    lSh && rSh && (lSh.visibility ?? 0) >= 0.15 && (rSh.visibility ?? 0) >= 0.15,
  );
  const rawSpanX = lSh && rSh ? Math.abs(lSh.x - rSh.x) : 0;

  const isWristCollapsedCluster = (wrist: NormalizedLandmark | undefined) => {
    if (!wrist || (wrist.visibility ?? 0) < 0.35 || !lSh || !rSh) return false;
    const dL = Math.hypot(lSh.x - wrist.x, lSh.y - wrist.y);
    const dR = Math.hypot(rSh.x - wrist.x, rSh.y - wrist.y);
    return dL < 0.075 && dR < 0.075;
  };

  const shouldersAnatomical =
    hasShoulderVis &&
    rawSpanX >= 0.085 &&
    !isWristCollapsedCluster(rWrist) &&
    !isWristCollapsedCluster(lWrist);

  if (shouldersAnatomical && lSh && rSh) {
    nextAnchor = {
      leftShoulder: { ...lSh },
      rightShoulder: { ...rSh },
      nose:
        nose && (nose.visibility ?? 0) >= 0.25
          ? { ...nose }
          : prevAnchor?.nose
            ? { ...prevAnchor.nose }
            : undefined,
      spanX: rawSpanX,
    };
  } else if (prevAnchor) {
    occludedFallback = true;
    stabilized[POSE_INDEX.LEFT_SHOULDER] = {
      ...prevAnchor.leftShoulder,
      visibility: Math.max(0.65, prevAnchor.leftShoulder.visibility ?? 0.65),
    };
    stabilized[POSE_INDEX.RIGHT_SHOULDER] = {
      ...prevAnchor.rightShoulder,
      visibility: Math.max(0.65, prevAnchor.rightShoulder.visibility ?? 0.65),
    };
    if (prevAnchor.nose) {
      const midX = (prevAnchor.leftShoulder.x + prevAnchor.rightShoulder.x) / 2;
      const midY = (prevAnchor.leftShoulder.y + prevAnchor.rightShoulder.y) / 2;
      const noseOk =
        nose &&
        (nose.visibility ?? 0) >= 0.25 &&
        Math.hypot(nose.x - midX, nose.y - midY) <= prevAnchor.spanX * 1.85;
      if (!noseOk) {
        stabilized[POSE_INDEX.NOSE] = {
          ...prevAnchor.nose,
          visibility: Math.max(0.65, prevAnchor.nose.visibility ?? 0.65),
        };
      }
    }
  }

  const effLSh = stabilized[POSE_INDEX.LEFT_SHOULDER];
  const effRSh = stabilized[POSE_INDEX.RIGHT_SHOULDER];

  // Suppress anatomically impossible chest/hand-height hallucinated hips/knees/ankles across all exercises
  const maxShY = effLSh && effRSh ? Math.max(effLSh.y, effRSh.y) : 0.5;
  for (const lowerIdx of LOWER_BODY_POSE_INDICES) {
    const lm = stabilized[lowerIdx];
    const isHipIdx = lowerIdx === POSE_INDEX.LEFT_HIP || lowerIdx === POSE_INDEX.RIGHT_HIP;
    const maxHallucinatedOffset = isHipIdx ? 0.05 : 0.12;
    if (lm && lm.y <= maxShY + maxHallucinatedOffset) {
      stabilized[lowerIdx] = { ...lm, visibility: 0 };
    }
  }

  // Repair overhead out-of-frame wrist foldback when hand exits the top of the camera frame
  if (targetJoint === 'left_shoulder' || targetJoint === 'right_shoulder') {
    const shIdx =
      targetJoint === 'left_shoulder' ? POSE_INDEX.LEFT_SHOULDER : POSE_INDEX.RIGHT_SHOULDER;
    const elbowIdx =
      targetJoint === 'left_shoulder' ? POSE_INDEX.LEFT_ELBOW : POSE_INDEX.RIGHT_ELBOW;
    const wristIdx =
      targetJoint === 'left_shoulder' ? POSE_INDEX.LEFT_WRIST : POSE_INDEX.RIGHT_WRIST;
    const activeSh = stabilized[shIdx];
    const activeElbow = stabilized[elbowIdx];
    const activeWrist = stabilized[wristIdx];
    if (
      activeSh &&
      activeElbow &&
      (activeElbow.visibility ?? 0) >= 0.3 &&
      activeElbow.y < activeSh.y - 0.08 &&
      activeWrist &&
      activeWrist.y > activeElbow.y + 0.08 &&
      (activeWrist.visibility ?? 0) < 0.65
    ) {
      stabilized[wristIdx] = {
        x: activeElbow.x + (activeElbow.x - activeSh.x) * 0.85,
        y: activeElbow.y + (activeElbow.y - activeSh.y) * 0.85,
        z:
          activeElbow.z !== undefined && activeSh.z !== undefined
            ? activeElbow.z + (activeElbow.z - activeSh.z) * 0.85
            : (activeElbow.z ?? 0),
        visibility: Math.max(0.55, activeElbow.visibility ?? 0.55),
      };
    }
  }

  return { landmarks: stabilized, anchor: nextAnchor, occludedFallback };
}

export type ExerciseJointMode =
  | 'elbow'
  | 'cross_body'
  | 'scap_protract'
  | 'bobath_push'
  | 'hand_to_head'
  | 'forearm_turn'
  | 'wrist_finger'
  | 'thumb_pinch'
  | 'shoulder_rotation'
  | 'shoulder_shrug'
  | 'bilateral_raise'
  | 'neck'
  | 'neck_turn'
  | 'default';

export interface ExerciseMetaHint {
  exercise_id?: string;
  catalog_exercise_id?: string;
  effective_code?: string;
  code?: string;
  prescribed_side?: ExerciseSideCode | string;
  side?: string | null;
  target_side?: string | null;
  name_th?: string;
  name_en?: string;
  target_joint?: string;
  expected_plane?: ExpectedMovementPlane;
}

export function resolveExerciseSideFromMeta(
  exerciseMeta?: ExerciseMetaHint,
  targetJoint?: string,
  expectedPlane?: ExpectedMovementPlane,
): ExerciseSideCode | undefined {
  const canonical = resolveActiveExerciseSide(
    exerciseMeta,
    undefined,
    targetJoint,
    expectedPlane,
  );
  if (canonical) return canonical;
  const jointMode = classifyExerciseJointMode(exerciseMeta);
  if (
    jointMode === 'bilateral_raise' ||
    jointMode === 'bobath_push' ||
    jointMode === 'shoulder_shrug' ||
    jointMode === 'neck_turn'
  ) {
    return 'BOTH';
  }
  return undefined;
}

export function classifyExerciseJointMode(
  exerciseMeta?: ExerciseMetaHint,
): ExerciseJointMode {
  if (!exerciseMeta) return 'default';
  const id = (exerciseMeta.catalog_exercise_id || exerciseMeta.exercise_id || '').toLowerCase();
  const name = `${exerciseMeta.name_th ?? ''} ${exerciseMeta.name_en ?? ''}`.toLowerCase();
  if (
    id.includes('hand_to_head') ||
    name.includes('แตะศีรษะ') ||
    name.includes('หวีผม') ||
    name.includes('hand-to-head')
  ) {
    return 'hand_to_head';
  }
  if (
    id.includes('elb') ||
    name.includes('ข้อศอก') ||
    name.includes('elbow')
  ) {
    return 'elbow';
  }
  if (
    id.includes('cross_body') ||
    name.includes('ไขว้หน้าอก') ||
    name.includes('cross-body')
  ) {
    return 'cross_body';
  }
  if (
    id.includes('scap_protract') ||
    name.includes('ผลักสะบัก') ||
    name.includes('ยื่นมือเอื้อมไปด้านหน้า') ||
    name.includes('protraction')
  ) {
    return 'scap_protract';
  }
  if (
    id.includes('forward_push') ||
    name.includes('ดันแขนตรงไปด้านหน้า') ||
    name.includes('forward push')
  ) {
    return 'bobath_push';
  }
  if (
    id.includes('arm_raise') ||
    id.includes('bilateral_arm_raise') ||
    name.includes('ประสานมือ') ||
    name.includes('bilateral')
  ) {
    return 'bilateral_raise';
  }
  if (
    id.includes('sh_rot') ||
    name.includes('หมุนข้อไหล่') ||
    name.includes('shoulder rotation')
  ) {
    return 'shoulder_rotation';
  }
  if (
    id.includes('forearm_turn') ||
    name.includes('คว่ำ') ||
    name.includes('หงายมือ') ||
    name.includes('supination')
  ) {
    return 'forearm_turn';
  }
  if (
    id.includes('thumb_pinch') ||
    name.includes('จีบนิ้ว') ||
    name.includes('pinch')
  ) {
    return 'thumb_pinch';
  }
  if (
    id.includes('wrist_finger') ||
    name.includes('กระดกข้อมือ') ||
    name.includes('wrist & finger')
  ) {
    return 'wrist_finger';
  }
  if (
    id.includes('shrug_roll') ||
    name.includes('ยักไหล่') ||
    name.includes('shoulder shrug')
  ) {
    return 'shoulder_shrug';
  }
  if (
    id.includes('neck_turn') ||
    name.includes('หันคอ') ||
    name.includes('neck rotation')
  ) {
    return 'neck_turn';
  }
  if (
    id.includes('neck') ||
    name.includes('เอียงคอ') ||
    name.includes('neck')
  ) {
    return 'neck';
  }
  return 'default';
}

export function measureLiveJointAngleAndTilt(
  landmarks: readonly NormalizedLandmark[],
  targetJoint: ExercisePrescriptionOption['target_joint'],
  expectedPlane: ExpectedMovementPlane = 'any',
  worldLandmarks?: readonly NormalizedLandmark[],
  exerciseMeta?: ExerciseMetaHint,
  strokeOptions?: {
    handLandmarksBySide?: {
      left?: readonly NormalizedLandmark[];
      right?: readonly NormalizedLandmark[];
    };
    baseline?: StrokeAnatomyBaseline;
    targetRomDeg?: number;
  },
): LiveJointMeasurement {
  const UPPER_MIN_VIS = 0.02;
  const jointMode = classifyExerciseJointMode(exerciseMeta);
  if (targetJoint !== 'left_knee' && targetJoint !== 'right_knee') {
    const inferredSide = resolveExerciseSideFromMeta(exerciseMeta, targetJoint, expectedPlane);
    const solved = solveStrokeClinicalJointBiomechanics({
      landmarks,
      worldLandmarks,
      handLandmarksBySide: strokeOptions?.handLandmarksBySide,
      targetJoint,
      exerciseSide: inferredSide,
      expectedPlane,
      targetRomDeg: strokeOptions?.targetRomDeg,
      jointMode,
      baseline: strokeOptions?.baseline,
    });
    if (solved.wrongSideRaised) {
      solved.wrongSideBannerTh =
        solved.sideViolationBannerTh ??
        getPrescribedLimbPromptTh(targetJoint, expectedPlane, inferredSide).bannerAlertTh;
    }
    if (solved.wrongPlaneRaised) {
      solved.wrongPlaneBannerTh = getPrescribedPlanePromptTh(expectedPlane).bannerAlertTh;
    }
    return solved;
  }

  const lSh = landmarks[POSE_INDEX.LEFT_SHOULDER];
  const rSh = landmarks[POSE_INDEX.RIGHT_SHOULDER];
  const lHip = landmarks[POSE_INDEX.LEFT_HIP];
  const rHip = landmarks[POSE_INDEX.RIGHT_HIP];
  let trunkTiltDeg = 2.0;
  if (lSh && rSh && lHip && rHip && (lHip.visibility ?? 0) >= 0.2 && (rHip.visibility ?? 0) >= 0.2) {
    const midShX = (lSh.x + rSh.x) / 2;
    const midShY = (lSh.y + rSh.y) / 2;
    const midHipX = (lHip.x + rHip.x) / 2;
    const midHipY = (lHip.y + rHip.y) / 2;
    trunkTiltDeg = Math.abs(
      (Math.atan2(midShX - midHipX, Math.max(0.05, midHipY - midShY)) * 180) / Math.PI,
    );
  }
  let romDeg: number | null = null;
  let lowVisibilityAlertTh: string | null = null;

  if (targetJoint === 'left_knee' || targetJoint === 'right_knee') {
    const isLeft = targetJoint === 'left_knee';
    const hipIdx = isLeft ? POSE_INDEX.LEFT_HIP : POSE_INDEX.RIGHT_HIP;
    const kneeIdx = isLeft ? POSE_INDEX.LEFT_KNEE : POSE_INDEX.RIGHT_KNEE;
    const ankleIdx = isLeft ? POSE_INDEX.LEFT_ANKLE : POSE_INDEX.RIGHT_ANKLE;
    const hip = landmarks[hipIdx];
    const knee = landmarks[kneeIdx];
    const ankle = landmarks[ankleIdx];
    const KNEE_JOINT_MIN_VIS = 0.55;
    const hasShouldersForOrder = Boolean(
      lSh && rSh && (lSh.visibility ?? 0) >= 0.35 && (rSh.visibility ?? 0) >= 0.35,
    );
    const maxShY = hasShouldersForOrder && lSh && rSh ? Math.max(lSh.y, rSh.y) : 0;
    const anatomicalOrderOk = Boolean(
      hip &&
      knee &&
      ankle &&
      (!hasShouldersForOrder || hip.y > maxShY + 0.10) &&
      knee.y >= hip.y - 0.08 &&
      ankle.y > knee.y + 0.06,
    );
    const inFrameKneeJoint = Boolean(
      hip &&
      knee &&
      ankle &&
      (hip.visibility ?? 0) >= KNEE_JOINT_MIN_VIS &&
      (knee.visibility ?? 0) >= KNEE_JOINT_MIN_VIS &&
      (ankle.visibility ?? 0) >= KNEE_JOINT_MIN_VIS &&
      hip.y <= 0.96 &&
      knee.y <= 0.96 &&
      ankle.y <= 0.96 &&
      anatomicalOrderOk,
    );
    if (inFrameKneeJoint && hip && knee && ankle) {
      const wHip = worldLandmarks?.[hipIdx];
      const wKnee = worldLandmarks?.[kneeIdx];
      const wAnkle = worldLandmarks?.[ankleIdx];
      const hasValidWorldKnee = Boolean(
        wHip &&
        wKnee &&
        wAnkle &&
        Math.hypot(wHip.x - wKnee.x, wHip.y - wKnee.y, (wHip.z ?? 0) - (wKnee.z ?? 0)) >= 0.08 &&
        Math.hypot(wAnkle.x - wKnee.x, wAnkle.y - wKnee.y, (wAnkle.z ?? 0) - (wKnee.z ?? 0)) >= 0.08,
      );
      romDeg =
        hasValidWorldKnee && wHip && wKnee && wAnkle
          ? computeJointAngle3D(wHip, wKnee, wAnkle)
          : computeJointAngle3D(hip, knee, ankle);
    } else {
      lowVisibilityAlertTh = 'มองไม่เห็นสะโพก-เข่า-ข้อเท้าชัดเจน กรุณาถอยห่างจากกล้องให้เห็นช่วงข้อต่อครบ';
    }
  }

  return {
    romDeg,
    rawPrimaryRomDeg: romDeg,
    trunkTiltDeg: Number(trunkTiltDeg.toFixed(1)),
    wrongSideRaised: false,
    oppositeRomDeg: null,
    wrongSideBannerTh: null,
    wrongPlaneRaised: false,
    wrongPlaneBannerTh: null,
    xyzVector: null,
    lowVisibilityAlertTh,
    postureCompensated: false,
    postureBannerTh: null,
    elbowAngleDeg: null,
    shoulderTiltDeg: 0,
    headY: landmarks[POSE_INDEX.NOSE] && (landmarks[POSE_INDEX.NOSE].visibility ?? 0) >= UPPER_MIN_VIS ? landmarks[POSE_INDEX.NOSE].y : null,
    shoulderMidY: lSh && rSh ? (lSh.y + rSh.y) / 2 : null,
  };
}

export interface UnifiedCoachingBannerSpec {
  tone: 'emerald' | 'sky' | 'amber' | 'rose' | 'slate';
  titleTh: string;
  subtitleTh: string;
  badgeTh?: string;
  suggestedRomDeg?: number | null;
  fatigueRestSec?: number;
  showSafetyAcknowledge?: boolean;
}

/**
 * Resolves a single non-overlapping coaching banner rendered cleanly BELOW the camera
 * viewport using canonical exercise side resolution so BOTH/LEFT exercises never show right-arm warnings.
 */
export function resolveUnifiedKioskCoachingBanner(params: {
  exercise: ExerciseMetaHint & { name_th: string; name_en: string; hold_sec: number; target_reps: number; target_rom_deg: number };
  expectedPlane: ExpectedMovementPlane;
  kioskState: string;
  wrongLimbAlert: boolean;
  wrongPlaneAlert: boolean;
  trunkLeanAlert: boolean;
  shoulderHikeAlert: boolean;
  tooFastAlert: boolean;
  isMultiPerson: boolean;
  isImpostor: boolean;
  ownerRehabGuardAlert?: { text: string; status: string } | null;
  strokeSafetyAlert?: { type: string; messageTh: string; suggestedRom?: number } | null;
  fatigueRestSec?: number;
  safetyGuardBannerTh?: string | null;
  framingHintTh?: string | null;
  repSuccessToastTh?: string | null;
  holdRemainingSec: number;
  holdTargetSec: number;
  currentRomDeg: number;
  targetRomDeg: number;
  validReps: number;
  targetReps: number;
  calibrationMessageTh?: string | null;
}): UnifiedCoachingBannerSpec {
  const activeSide =
    resolveExerciseSideFromMeta(params.exercise, params.exercise.target_joint, params.expectedPlane) ??
    (/both|ทั้งสองข้าง|ทั้ง\s*2\s*ข้าง|ประสานมือ|bobath/i.test(
      `${params.exercise.name_th} ${params.exercise.name_en}`,
    )
      ? 'BOTH'
      : undefined);

  if (params.ownerRehabGuardAlert && params.ownerRehabGuardAlert.status === 'MULTI_PERSON') {
    return {
      tone: 'amber',
      titleTh: params.ownerRehabGuardAlert.text,
      subtitleTh: 'กรุณาให้ผู้ป่วยเจ้าของบัญชีฝึกเพียงลำพังหน้ากล้องครับ',
      badgeTh: 'PAUSED',
    };
  }

  if (params.isImpostor || params.isMultiPerson) {
    return {
      tone: params.isImpostor ? 'rose' : 'amber',
      titleTh: params.isImpostor
        ? 'ผู้ใช้งานไม่ใช่คนไข้ที่ลงทะเบียนไว้'
        : 'พบมากกว่า 1 คนในเฟรม • ระบบหยุดนับชั่วคราว',
      subtitleTh: params.isImpostor
        ? 'กรุณาให้คนไข้ตัวจริงเข้าฝึกครับ ระบบจะไม่นับรอบให้บุคคลอื่น'
        : 'กรุณาให้คนไข้ยืนหน้ากล้องเพียงลำพัง ระบบจึงจะนับรอบการฝึกต่อครับ',
      badgeTh: params.isImpostor ? 'LOCKED' : 'PAUSED',
    };
  }

  if (params.wrongLimbAlert) {
    if (activeSide === 'BOTH') {
      return {
        tone: 'rose',
        titleTh: 'ท่านี้ต้องยกแขนทั้ง 2 ข้างพร้อมกัน (Both Arms)',
        subtitleTh: 'กรุณายกแขนทั้งข้างซ้ายและข้างขวาขึ้นพร้อมกันให้ได้ระดับเดียวกันครับ',
        badgeTh: 'BOTH ARMS',
      };
    }
    if (activeSide === 'LEFT') {
      return {
        tone: 'rose',
        titleTh: 'ท่านี้ฝึกเฉพาะข้างซ้าย (Left Arm Only)',
        subtitleTh: 'กรุณาลดแขนขวาลงข้างลำตัว และยกเฉพาะแขนซ้ายข้างเดียวครับ',
        badgeTh: 'LEFT ONLY',
      };
    }
    return {
      tone: 'rose',
      titleTh: 'ท่านี้ฝึกเฉพาะข้างขวา (Right Arm Only)',
      subtitleTh: 'กรุณาลดแขนซ้ายลงข้างลำตัว และยกเฉพาะแขนขวาข้างเดียวครับ',
      badgeTh: 'RIGHT ONLY',
    };
  }

  if (params.wrongPlaneAlert) {
    const planePrompt = getPrescribedPlanePromptTh(params.expectedPlane);
    return {
      tone: 'amber',
      titleTh: `เคลื่อนไหวผิดทิศทาง — กรุณา${planePrompt.requiredPlaneTh}`,
      subtitleTh: planePrompt.bannerAlertTh,
      badgeTh: `กำลัง${planePrompt.wrongPlaneTh}`,
    };
  }

  if (params.repSuccessToastTh && params.repSuccessToastTh.includes('ได้ +1 แต้มแล้ว')) {
    return {
      tone: 'emerald',
      titleTh: params.repSuccessToastTh,
      subtitleTh: 'ค้างครบเวลาแล้ว! ลดแขนลงข้างลำตัวเพื่อเริ่มนับรอบถัดไป',
      badgeTh: `${params.validReps}/${params.targetReps} REPS • ครบ!`,
    };
  }

  const waitLowerRequested = Boolean(
    params.repSuccessToastTh &&
      (params.repSuccessToastTh.includes('เอาแขนลง') || params.repSuccessToastTh.includes('ค้างครบ')),
  );
  if (waitLowerRequested || (params.kioskState === 'HOLD_PHASE' && params.holdRemainingSec <= 0)) {
    return {
      tone: 'sky',
      titleTh: 'ค้างครบเวลาแล้ว! ค่อยๆ เอาแขนลงข้างลำตัวเพื่อนับ +1 ครั้ง',
      subtitleTh: 'ลดแขนลงกลับสู่ท่าพักช้าๆ เพื่อเตรียมพร้อมรอบถัดไป',
      badgeTh: 'WAIT LOWER',
    };
  }

  if (params.repSuccessToastTh || params.kioskState === 'REP_COMPLETED' || params.kioskState === 'SET_FINISHED') {
    return {
      tone: 'emerald',
      titleTh:
        params.repSuccessToastTh ??
        `ได้ +1 แต้มแล้ว! (รวม ${params.validReps}/${params.targetReps} ครั้ง) • กรุณาเอาแขนลงข้างลำตัวเพื่อเริ่มนับรอบถัดไป`,
      subtitleTh: 'ลดแขนลงกลับสู่ท่าพักเพื่อเริ่มนับรอบถัดไปครับ',
      badgeTh: `${params.validReps}/${params.targetReps} REPS`,
    };
  }

  if (params.kioskState === 'HOLD_PHASE') {
    const elapsedSec = Math.max(0, params.holdTargetSec - params.holdRemainingSec);
    return {
      tone: 'emerald',
      titleTh: `กำลังค้างท่าเป้าหมาย — ${elapsedSec.toFixed(1)} / ${params.holdTargetSec.toFixed(1)} วินาที`,
      subtitleTh: 'ค้างไว้ให้นิ่งที่สุดครับ ถึงเป้าแล้วจะบวก +1 ครั้งทันที',
      badgeTh: `${params.holdRemainingSec.toFixed(1)}s LEFT`,
    };
  }

  if (
    params.trunkLeanAlert ||
    params.shoulderHikeAlert ||
    params.tooFastAlert ||
    (params.strokeSafetyAlert && params.strokeSafetyAlert.type !== 'emergency') ||
    params.safetyGuardBannerTh ||
    (params.fatigueRestSec ?? 0) > 0
  ) {
    const titleTh =
      params.strokeSafetyAlert?.messageTh ??
      params.safetyGuardBannerTh ??
      (params.trunkLeanAlert
        ? 'ลำตัวเอียงมากกว่า 18° — กรุณาค่อยๆ ปรับลำตัวให้ตรงครับ'
        : params.shoulderHikeAlert
          ? 'ผ่อนคลายหัวไหล่ลงเล็กน้อยครับ'
          : 'เคลื่อนไหวช้าลงนิดนึงครับ ไม่ต้องฝืนยกสูงเกินไปนะครับ');
    const subtitleTh = 'ตัวนับเวลายังคงทำงานต่อตามปกติครับ';
    return {
      tone: 'amber',
      titleTh,
      subtitleTh,
      suggestedRomDeg: params.strokeSafetyAlert?.suggestedRom ?? null,
      fatigueRestSec: params.fatigueRestSec ?? 0,
      showSafetyAcknowledge: Boolean(params.strokeSafetyAlert || (params.fatigueRestSec ?? 0) > 0),
    };
  }

  if (params.framingHintTh || params.calibrationMessageTh) {
    return {
      tone: 'amber',
      titleTh: params.framingHintTh ?? params.calibrationMessageTh ?? 'กรุณาจัดตำแหน่งร่างกายหน้ากล้อง',
      subtitleTh: 'ยืนห่างกล้อง ~1.8–2.2 เมตร ให้กล้องเห็นศีรษะ ไหล่ และลำตัวชัดเจน',
    };
  }

  const movePrompt =
    activeSide === 'BOTH'
      ? 'ยกแขนทั้ง 2 ข้างขึ้นพร้อมกันถึงมุมเป้าหมาย'
      : activeSide === 'LEFT'
        ? 'ยกแขนข้างซ้ายขึ้นถึงมุมเป้าหมาย'
        : activeSide === 'RIGHT'
          ? 'ยกแขนข้างขวาขึ้นถึงมุมเป้าหมาย'
          : 'เคลื่อนไหวตามท่าทางเป้าหมายอย่างต่อเนื่อง';

  return {
    tone: 'emerald',
    titleTh: movePrompt,
    subtitleTh: `มุมปัจจุบัน ${Math.round(params.currentRomDeg)}° / เป้าหมาย ${Math.round(params.targetRomDeg)}° • ค้าง ${params.holdTargetSec.toFixed(1)} วินาที แล้วลดแขนลงเพื่อนับ +1 ครั้ง`,
    badgeTh: `${Math.round(params.currentRomDeg)}° / ${Math.round(params.targetRomDeg)}°`,
  };
}

/**
 * Smooths noisy joint ROM readings with a compact anti-jitter deadband and instant
 * fast-pursuit slew rate so gauge arcs and degree readouts lock onto rapid arm motion.
 */
export function stepSmoothRomAngle(
  currentRom: number | null,
  targetRom: number,
  dtMs: number,
): number {
  const safeTarget = Math.max(0, Math.min(180, targetRom));
  if (currentRom === null || !Number.isFinite(currentRom)) {
    return safeTarget;
  }
  const safeDt = Math.min(100, Math.max(1, dtMs));
  const rawDelta = safeTarget - currentRom;
  // Tight deadband of 0.35 deg suppresses stationary jitter while locking onto motion without lag
  if (Math.abs(rawDelta) < 0.35) {
    return currentRom;
  }
  // Instant fast-pursuit filter (tau: 3..12ms) + high slew cap (1800 deg/s)
  const absDelta = Math.abs(rawDelta);
  const adaptiveTau = absDelta >= 12 ? 3 : absDelta >= 4 ? 6 : 12;
  const alpha = 1 - Math.exp(-safeDt / adaptiveTau);
  const emaStep = rawDelta * alpha;
  const maxStep = (1800 * safeDt) / 1000;
  const step = emaStep > 0 ? Math.min(emaStep, maxStep) : Math.max(emaStep, -maxStep);
  return Math.max(0, Math.min(180, currentRom + step));
}

export interface LandmarkSmootherOptions {
  alpha?: number;
  adaptive?: boolean;
  minAlpha?: number;
  maxAlpha?: number;
  velocityScale?: number;
}

/**
 * Dual-mode EMA & Velocity-based Adaptive EMA smoother over (N, 3) pose [x, y, z] coordinates.
 */
export class LandmarkSmoother {
  alpha: number;
  adaptive: boolean;
  minAlpha: number;
  maxAlpha: number;
  velocityScale: number;
  private smoothedCoords: Array<[number, number, number]> | null = null;

  constructor(options: LandmarkSmootherOptions = {}) {
    this.alpha = options.alpha ?? 0.3;
    this.adaptive = options.adaptive ?? true;
    this.minAlpha = options.minAlpha ?? 0.1;
    this.maxAlpha = options.maxAlpha ?? 0.8;
    this.velocityScale = options.velocityScale ?? 15.0;
  }

  apply(currentLandmarks: readonly NormalizedLandmark[]): NormalizedLandmark[] {
    if (currentLandmarks.length === 0) return [];
    if (!this.smoothedCoords || this.smoothedCoords.length !== currentLandmarks.length) {
      this.smoothedCoords = currentLandmarks.map((lm) => [lm.x, lm.y, lm.z ?? 0]);
      return currentLandmarks.map((lm) => ({ ...lm }));
    }

    const nextCoords: Array<[number, number, number]> = [];
    const output: NormalizedLandmark[] = new Array(currentLandmarks.length);

    for (let i = 0; i < currentLandmarks.length; i += 1) {
      const lm = currentLandmarks[i]!;
      const prev = this.smoothedCoords[i]!;
      const curX = lm.x;
      const curY = lm.y;
      const curZ = lm.z ?? 0;

      let effectiveAlpha = this.alpha;
      if (this.adaptive) {
        const velocity = Math.hypot(curX - prev[0], curY - prev[1], curZ - prev[2]);
        effectiveAlpha = Math.max(
          this.minAlpha,
          Math.min(this.maxAlpha, this.minAlpha + velocity * this.velocityScale),
        );
      }

      const sx = effectiveAlpha * curX + (1 - effectiveAlpha) * prev[0];
      const sy = effectiveAlpha * curY + (1 - effectiveAlpha) * prev[1];
      const sz = effectiveAlpha * curZ + (1 - effectiveAlpha) * prev[2];
      nextCoords.push([sx, sy, sz]);
      output[i] = {
        ...lm,
        x: sx,
        y: sy,
        z: sz,
      };
    }

    this.smoothedCoords = nextCoords;
    return output;
  }

  reset(): void {
    this.smoothedCoords = null;
  }
}

/**
 * Zero-delay 1:1 landmark follower: MediaPipe VIDEO mode already applies its internal C++ One-Euro
 * filter, so only sub-pixel stationary tremor (< 0.0015 norm) is damped while active movement snaps 1:1.
 */
export function interpolatePoseLandmarks(
  prev: NormalizedLandmark[] | null,
  next: readonly NormalizedLandmark[],
  dtMs: number,
): NormalizedLandmark[] {
  if (!prev || prev.length !== next.length) {
    return next.map((lm) => ({ ...lm }));
  }
  const safeDt = Math.min(100, Math.max(1, dtMs));
  const jitterAlpha = Math.min(1, 1 - Math.exp(-safeDt / 22));
  return next.map((target, idx) => {
    const cur = prev[idx];
    if (!cur) return { ...target };
    const dx = target.x - cur.x;
    const dy = target.y - cur.y;
    const dist2D = Math.hypot(dx, dy);
    const alpha = dist2D < 0.0015 ? jitterAlpha : dist2D < 0.0025 ? 0.9 : 1.0;
    if (alpha >= 1) {
      return { ...target };
    }
    const vis =
      target.visibility !== undefined
        ? (cur.visibility ?? target.visibility) +
        (target.visibility - (cur.visibility ?? target.visibility)) * alpha
        : cur.visibility;
    return {
      x: cur.x + dx * alpha,
      y: cur.y + dy * alpha,
      z: cur.z + (target.z - cur.z) * alpha,
      visibility: vis,
    };
  });
}

/**
 * High-speed 60FPS Frame-to-Frame Linear Interpolation (`alpha = 0.42`) paired with
 * One-Euro Filter coordinates so live skeletal lines glue tightly to the patient with zero stutter.
 */
export function lerpPoseLandmarksAlpha(
  prev: NormalizedLandmark[] | null,
  targetList: readonly NormalizedLandmark[],
  alpha = 0.42,
): NormalizedLandmark[] {
  if (!prev || prev.length !== targetList.length) {
    return targetList.map((lm) => ({ ...lm }));
  }
  const safeAlpha = Math.max(0.05, Math.min(1, alpha));
  return targetList.map((target, idx) => {
    const cur = prev[idx];
    if (!cur || (cur.visibility ?? 0) < 0.35) return { ...target };
    return {
      x: cur.x + (target.x - cur.x) * safeAlpha,
      y: cur.y + (target.y - cur.y) * safeAlpha,
      z: (cur.z ?? 0) + ((target.z ?? 0) - (cur.z ?? 0)) * safeAlpha,
      visibility: target.visibility ?? cur.visibility,
    };
  });
}

const LIVE_BODY_BONELINE_PAIRS: readonly [number, number][] = [
  [POSE_INDEX.LEFT_SHOULDER, POSE_INDEX.RIGHT_SHOULDER],
  [POSE_INDEX.LEFT_SHOULDER, POSE_INDEX.LEFT_ELBOW],
  [POSE_INDEX.LEFT_ELBOW, POSE_INDEX.LEFT_WRIST],
  [POSE_INDEX.RIGHT_SHOULDER, POSE_INDEX.RIGHT_ELBOW],
  [POSE_INDEX.RIGHT_ELBOW, POSE_INDEX.RIGHT_WRIST],
  [POSE_INDEX.LEFT_SHOULDER, POSE_INDEX.LEFT_HIP],
  [POSE_INDEX.RIGHT_SHOULDER, POSE_INDEX.RIGHT_HIP],
  [POSE_INDEX.LEFT_HIP, POSE_INDEX.RIGHT_HIP],
  [POSE_INDEX.LEFT_HIP, POSE_INDEX.LEFT_KNEE],
  [POSE_INDEX.LEFT_KNEE, POSE_INDEX.LEFT_ANKLE],
  [POSE_INDEX.RIGHT_HIP, POSE_INDEX.RIGHT_KNEE],
  [POSE_INDEX.RIGHT_KNEE, POSE_INDEX.RIGHT_ANKLE],
];

const LIVE_KEYPOINT_INDICES: readonly number[] = [
  POSE_INDEX.LEFT_SHOULDER,
  POSE_INDEX.RIGHT_SHOULDER,
  POSE_INDEX.LEFT_ELBOW,
  POSE_INDEX.RIGHT_ELBOW,
  POSE_INDEX.LEFT_WRIST,
  POSE_INDEX.RIGHT_WRIST,
  POSE_INDEX.LEFT_HIP,
  POSE_INDEX.RIGHT_HIP,
  POSE_INDEX.LEFT_KNEE,
  POSE_INDEX.RIGHT_KNEE,
  POSE_INDEX.LEFT_ANKLE,
  POSE_INDEX.RIGHT_ANKLE,
];

const LIVE_HAND_BONELINE_PAIRS: readonly [number, number][] = [
  [POSE_INDEX.LEFT_WRIST, POSE_INDEX.LEFT_PINKY],
  [POSE_INDEX.LEFT_WRIST, POSE_INDEX.LEFT_INDEX],
  [POSE_INDEX.LEFT_WRIST, POSE_INDEX.LEFT_THUMB],
  [POSE_INDEX.LEFT_PINKY, POSE_INDEX.LEFT_INDEX],
  [POSE_INDEX.RIGHT_WRIST, POSE_INDEX.RIGHT_PINKY],
  [POSE_INDEX.RIGHT_WRIST, POSE_INDEX.RIGHT_INDEX],
  [POSE_INDEX.RIGHT_WRIST, POSE_INDEX.RIGHT_THUMB],
  [POSE_INDEX.RIGHT_PINKY, POSE_INDEX.RIGHT_INDEX],
];

const LIVE_FOOT_BONELINE_PAIRS: readonly [number, number][] = [
  [POSE_INDEX.LEFT_ANKLE, POSE_INDEX.LEFT_HEEL],
  [POSE_INDEX.LEFT_ANKLE, POSE_INDEX.LEFT_FOOT_INDEX],
  [POSE_INDEX.LEFT_HEEL, POSE_INDEX.LEFT_FOOT_INDEX],
  [POSE_INDEX.RIGHT_ANKLE, POSE_INDEX.RIGHT_HEEL],
  [POSE_INDEX.RIGHT_ANKLE, POSE_INDEX.RIGHT_FOOT_INDEX],
  [POSE_INDEX.RIGHT_HEEL, POSE_INDEX.RIGHT_FOOT_INDEX],
];

const LIVE_EXTREMITY_KEYPOINT_INDICES: readonly number[] = [
  POSE_INDEX.LEFT_PINKY,
  POSE_INDEX.RIGHT_PINKY,
  POSE_INDEX.LEFT_INDEX,
  POSE_INDEX.RIGHT_INDEX,
  POSE_INDEX.LEFT_THUMB,
  POSE_INDEX.RIGHT_THUMB,
  POSE_INDEX.LEFT_HEEL,
  POSE_INDEX.RIGHT_HEEL,
  POSE_INDEX.LEFT_FOOT_INDEX,
  POSE_INDEX.RIGHT_FOOT_INDEX,
];

const UPPER_BODY_KEYPOINTS = new Set<number>([
  POSE_INDEX.LEFT_SHOULDER,
  POSE_INDEX.RIGHT_SHOULDER,
  POSE_INDEX.LEFT_ELBOW,
  POSE_INDEX.RIGHT_ELBOW,
  POSE_INDEX.LEFT_WRIST,
  POSE_INDEX.RIGHT_WRIST,
  POSE_INDEX.LEFT_PINKY,
  POSE_INDEX.RIGHT_PINKY,
  POSE_INDEX.LEFT_INDEX,
  POSE_INDEX.RIGHT_INDEX,
  POSE_INDEX.LEFT_THUMB,
  POSE_INDEX.RIGHT_THUMB,
]);

export function drawLivePoseOverlayOnCanvas(
  canvas: HTMLCanvasElement | null,
  landmarksList: readonly (readonly NormalizedLandmark[])[],
  opts: {
    targetJoint: ExercisePrescriptionOption['target_joint'];
    expectedPlane?: ExpectedMovementPlane;
    exerciseMeta?: ExerciseMetaHint;
    romDeg: number | null;
    targetRomDeg: number;
    holdRatio: number;
    trunkTiltDeg: number;
    isMultiPerson: boolean;
    isImpostorOrEmergency: boolean;
    sideViolation?: StrokeSideViolationCode;
    wrongSideRaised?: boolean;
    wrongSideBannerTh?: string | null;
    wrongPlaneRaised?: boolean;
    wrongPlaneBannerTh?: string | null;
    lowVisibilityAlertTh?: string | null;
    postureBannerTh?: string | null;
    repSuccessBannerTh?: string | null;
    overlayHint?: StrokeTargetOverlayHint | null;
    handLandmarks21?: readonly NormalizedLandmark[];
    videoWidth?: number;
    videoHeight?: number;
    canvasWidth?: number;
    canvasHeight?: number;
    smoothingStatusText?: string;
    showSkeleton?: boolean;
    isPrivacyMode?: boolean;
    ownerLock?: {
      ownerName: string;
      faceBox: OwnerFaceBoxMetrics | null;
      extraFaceBoxes?: OwnerFaceBoxMetrics[];
      ownerFaceVerified: boolean;
      faceCount: number;
    };
  },
): void {
  if (!canvas) return;
  const w =
    opts.canvasWidth && opts.canvasWidth > 0
      ? opts.canvasWidth
      : canvas.clientWidth || 640;
  const h =
    opts.canvasHeight && opts.canvasHeight > 0
      ? opts.canvasHeight
      : canvas.clientHeight || 480;
  const dpr = Math.min(typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1, 2);
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  // Draw Continuous 1:1 Account Owner Face Lock HUD + Biometric Tether onto full-bleed overlay
  if (opts.ownerLock && opts.showSkeleton !== false) {
    const { ownerName, faceBox, extraFaceBoxes = [], ownerFaceVerified, faceCount } = opts.ownerLock;
    const drawRehabFaceBox = (box: OwnerFaceBoxMetrics, hex: string, label: string, dashed = false) => {
      const x1 = Math.max(4, box.xMin * w);
      const y1 = Math.max(4, box.yMin * h);
      const bw = Math.max(24, box.width * w);
      const bh = Math.max(28, box.height * h);
      ctx.save();
      ctx.strokeStyle = hex;
      ctx.lineWidth = 3;
      ctx.shadowColor = hex;
      ctx.shadowBlur = 10;
      if (dashed) ctx.setLineDash([7, 5]);
      ctx.beginPath();
      ctx.roundRect(x1, y1, bw, bh, 14);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.font = '800 12px "Sarabun", sans-serif';
      const lw = ctx.measureText(label).width + 18;
      const ly = Math.max(6, y1 - 24);
      ctx.fillStyle = hex;
      ctx.beginPath();
      ctx.roundRect(x1, ly, lw, 22, 8);
      ctx.fill();
      ctx.fillStyle = '#FFFFFF';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillText(label, x1 + 9, ly + 11);
      ctx.restore();
    };
    if (faceBox) {
      if (faceCount > 1) {
        drawRehabFaceBox(faceBox, '#F59E0B', 'พบมากกว่า 1 คนในเฟรม', true);
      } else if (ownerFaceVerified) {
        drawRehabFaceBox(faceBox, '#10B981', `เจ้าของบัญชี: ${ownerName}`, false);
      } else {
        drawRehabFaceBox(faceBox, '#EF4444', `ไม่ใช่เจ้าของบัญชี (${ownerName})`, false);
      }
    }
    for (const extra of extraFaceBoxes) {
      drawRehabFaceBox(extra, '#EF4444', 'บุคคลอื่น', true);
    }

    // Glowing Emerald Dashed Biometric Tether linking Owner Face Box to active wrists
    if (
      ownerFaceVerified &&
      faceBox &&
      faceCount === 1 &&
      landmarksList.length > 0 &&
      isUpperBodyVisibleInFrame(landmarksList[0])
    ) {
      const vwT = opts.videoWidth && opts.videoWidth > 0 ? opts.videoWidth : w;
      const vhT = opts.videoHeight && opts.videoHeight > 0 ? opts.videoHeight : h;
      const covT = computeObjectCoverLayout(vwT, vhT, w, h);
      const p0 = landmarksList[0];
      const fAnchorLX = faceBox.xMin * w + faceBox.width * w * 0.2;
      const fAnchorRX = faceBox.xMax * w - faceBox.width * w * 0.2;
      const fAnchorY = faceBox.centerY * h;
      for (const [wIdx, ax] of [
        [POSE_INDEX.LEFT_WRIST, fAnchorLX],
        [POSE_INDEX.RIGHT_WRIST, fAnchorRX],
      ] as const) {
        const wLm = p0[wIdx];
        if (!wLm) continue;
        const wx = covT.offsetX + (1 - wLm.x) * covT.renderW;
        const wy = covT.offsetY + wLm.y * covT.renderH;
        if (!isJointValidInFrame({ lm: wLm, canvasPt: { x: wx, y: wy }, canvasW: w, canvasH: h, minScore: SKELETON_JOINT_MIN_SCORE })) {
          continue;
        }
        ctx.save();
        ctx.strokeStyle = '#10B981';
        ctx.lineWidth = 2.2;
        ctx.shadowColor = '#10B981';
        ctx.shadowBlur = 8;
        ctx.setLineDash([6, 4]);
        ctx.beginPath();
        ctx.moveTo(ax, fAnchorY);
        ctx.lineTo(wx, wy);
        ctx.stroke();
        ctx.restore();
      }
    }
  }

  if (
    opts.showSkeleton === false ||
    landmarksList.length === 0 ||
    !isUpperBodyVisibleInFrame(landmarksList[0])
  ) {
    return;
  }

  // Match CSS object-cover scale-x-[-1] full-bleed framing so mirrored landmarks align 1:1 with zero black bars
  const vw = opts.videoWidth && opts.videoWidth > 0 ? opts.videoWidth : w;
  const vh = opts.videoHeight && opts.videoHeight > 0 ? opts.videoHeight : h;
  const rawCanvasX = (normX: number) => mapVideoPointToCoverCanvas(normX, 0, vw, vh, w, h, true).x;
  const rawCanvasY = (normY: number) => mapVideoPointToCoverCanvas(0, normY, vw, vh, w, h, true).y;
  const toCanvasPt = (_idx: number, lm: NormalizedLandmark) =>
    mapVideoPointToCoverCanvas(lm.x, lm.y, vw, vh, w, h, true);
  const minVisFor = (idx: number) => {
    if (
      idx === POSE_INDEX.LEFT_ELBOW ||
      idx === POSE_INDEX.RIGHT_ELBOW ||
      idx === POSE_INDEX.LEFT_WRIST ||
      idx === POSE_INDEX.RIGHT_WRIST ||
      idx === POSE_INDEX.LEFT_PINKY ||
      idx === POSE_INDEX.RIGHT_PINKY ||
      idx === POSE_INDEX.LEFT_INDEX ||
      idx === POSE_INDEX.RIGHT_INDEX ||
      idx === POSE_INDEX.LEFT_THUMB ||
      idx === POSE_INDEX.RIGHT_THUMB
    ) {
      return SKELETON_JOINT_MIN_SCORE;
    }
    return UPPER_BODY_KEYPOINTS.has(idx) ? 0.35 : 0.55;
  };

  const activeColor =
    opts.isImpostorOrEmergency || opts.wrongSideRaised
      ? '#EF4444'
      : opts.postureBannerTh || opts.wrongPlaneRaised || opts.isMultiPerson || opts.trunkTiltDeg > 18 || opts.lowVisibilityAlertTh
        ? '#F59E0B'
        : '#10B981';

  const primary = landmarksList[0];
  const isUpperLimbOrSpine =
    opts.targetJoint === 'left_shoulder' ||
    opts.targetJoint === 'right_shoulder' ||
    opts.targetJoint === 'spine';
  const lShLm = primary[POSE_INDEX.LEFT_SHOULDER];
  const rShLm = primary[POSE_INDEX.RIGHT_SHOULDER];
  const maxShY =
    lShLm && rShLm ? Math.max(lShLm.y, rShLm.y) : (lShLm?.y ?? rShLm?.y ?? 0.5);
  const lHipLm = primary[POSE_INDEX.LEFT_HIP];
  const rHipLm = primary[POSE_INDEX.RIGHT_HIP];
  const hipsCleanlyBelowShoulders = Boolean(
    lHipLm &&
    rHipLm &&
    (lHipLm.visibility ?? 0) >= 0.55 &&
    (rHipLm.visibility ?? 0) >= 0.55 &&
    lHipLm.y > maxShY + 0.16 &&
    rHipLm.y > maxShY + 0.16 &&
    Math.abs(lHipLm.x - rHipLm.x) >= 0.09,
  );
  const resolvedDrawSide = resolveExerciseSideFromMeta(
    opts.exerciseMeta,
    opts.targetJoint,
    opts.expectedPlane,
  );
  const isBilateralMode =
    resolvedDrawSide === 'BOTH' ||
    opts.expectedPlane === 'bilateral' ||
    opts.expectedPlane === 'bilateral_push';

  const activeWristLm =
    resolvedDrawSide === 'BOTH'
      ? undefined
      : resolvedDrawSide === 'LEFT' || opts.targetJoint === 'left_shoulder'
        ? primary[POSE_INDEX.LEFT_WRIST]
        : opts.targetJoint === 'right_shoulder'
          ? primary[POSE_INDEX.RIGHT_WRIST]
          : undefined;

  const isOppositeArmIdx = (idx: number) => {
    if (isBilateralMode) return false;
    if (resolvedDrawSide === 'LEFT' || opts.targetJoint === 'left_shoulder') {
      return (
        idx === POSE_INDEX.RIGHT_ELBOW ||
        idx === POSE_INDEX.RIGHT_WRIST ||
        idx === POSE_INDEX.RIGHT_PINKY ||
        idx === POSE_INDEX.RIGHT_INDEX ||
        idx === POSE_INDEX.RIGHT_THUMB
      );
    }
    if (resolvedDrawSide === 'RIGHT' || opts.targetJoint === 'right_shoulder') {
      return (
        idx === POSE_INDEX.LEFT_ELBOW ||
        idx === POSE_INDEX.LEFT_WRIST ||
        idx === POSE_INDEX.LEFT_PINKY ||
        idx === POSE_INDEX.LEFT_INDEX ||
        idx === POSE_INDEX.LEFT_THUMB
      );
    }
    return false;
  };

  const skeletonSafety = evaluatePoseCanvasSkeletonSafety({
    landmarks: primary,
    videoW: vw,
    videoH: vh,
    canvasW: w,
    canvasH: h,
    isMirrored: true,
    minScoreForIdx: (idx) => (isOppositeArmIdx(idx) ? SKELETON_JOINT_MIN_SCORE : minVisFor(idx)),
  });

  const isPointDrawable = (idx: number): boolean => {
    const lm = primary[idx];
    if (!lm) return false;
    if (!skeletonSafety.validMask[idx]) return false;
    const isKnee = idx === POSE_INDEX.LEFT_KNEE || idx === POSE_INDEX.RIGHT_KNEE;
    const isFootOrAnkle =
      idx === POSE_INDEX.LEFT_ANKLE ||
      idx === POSE_INDEX.RIGHT_ANKLE ||
      idx === POSE_INDEX.LEFT_HEEL ||
      idx === POSE_INDEX.RIGHT_HEEL ||
      idx === POSE_INDEX.LEFT_FOOT_INDEX ||
      idx === POSE_INDEX.RIGHT_FOOT_INDEX;
    const isHip = idx === POSE_INDEX.LEFT_HIP || idx === POSE_INDEX.RIGHT_HIP;
    if (isUpperLimbOrSpine) {
      if (isKnee) return false;
      if (isFootOrAnkle && (lm.visibility ?? 0) < 0.55) return false;
      if (isHip && !hipsCleanlyBelowShoulders) return false;
    }
    const minV = minVisFor(idx);
    if ((lm.visibility ?? 0) < minV) return false;
    if (
      isOppositeArmIdx(idx) &&
      !opts.wrongSideRaised &&
      activeWristLm &&
      (activeWristLm.visibility ?? 0) >= SKELETON_JOINT_MIN_SCORE &&
      Math.hypot(lm.x - activeWristLm.x, lm.y - activeWristLm.y) < 0.08
    ) {
      return false;
    }
    return true;
  };

  const isHighlightedEdge = (a: number, b: number) => {
    const isLeftEdge =
      (a === POSE_INDEX.LEFT_SHOULDER && b === POSE_INDEX.LEFT_ELBOW) ||
      (a === POSE_INDEX.LEFT_ELBOW && b === POSE_INDEX.LEFT_WRIST);
    const isRightEdge =
      (a === POSE_INDEX.RIGHT_SHOULDER && b === POSE_INDEX.RIGHT_ELBOW) ||
      (a === POSE_INDEX.RIGHT_ELBOW && b === POSE_INDEX.RIGHT_WRIST);
    if (isBilateralMode) {
      return isLeftEdge || isRightEdge;
    }
    if (resolvedDrawSide === 'LEFT' || opts.targetJoint === 'left_shoulder') {
      return isLeftEdge;
    }
    if (resolvedDrawSide === 'RIGHT' || opts.targetJoint === 'right_shoulder') {
      return isRightEdge;
    }
    if (opts.targetJoint === 'left_knee') {
      return (
        (a === POSE_INDEX.LEFT_HIP && b === POSE_INDEX.LEFT_KNEE) ||
        (a === POSE_INDEX.LEFT_KNEE && b === POSE_INDEX.LEFT_ANKLE)
      );
    }
    if (opts.targetJoint === 'right_knee') {
      return (
        (a === POSE_INDEX.RIGHT_HIP && b === POSE_INDEX.RIGHT_KNEE) ||
        (a === POSE_INDEX.RIGHT_KNEE && b === POSE_INDEX.RIGHT_ANKLE)
      );
    }
    return (
      (a === POSE_INDEX.LEFT_SHOULDER && b === POSE_INDEX.RIGHT_SHOULDER) ||
      (a === POSE_INDEX.LEFT_SHOULDER && b === POSE_INDEX.LEFT_HIP) ||
      (a === POSE_INDEX.RIGHT_SHOULDER && b === POSE_INDEX.RIGHT_HIP)
    );
  };

  const isOppositeArmEdge = (a: number, b: number) => {
    const isLeftEdge =
      (a === POSE_INDEX.LEFT_SHOULDER && b === POSE_INDEX.LEFT_ELBOW) ||
      (a === POSE_INDEX.LEFT_ELBOW && b === POSE_INDEX.LEFT_WRIST);
    const isRightEdge =
      (a === POSE_INDEX.RIGHT_SHOULDER && b === POSE_INDEX.RIGHT_ELBOW) ||
      (a === POSE_INDEX.RIGHT_ELBOW && b === POSE_INDEX.RIGHT_WRIST);
    if (opts.sideViolation === 'RIGHT_ARM_RAISED_IN_LEFT_MODE') return isRightEdge;
    if (opts.sideViolation === 'LEFT_ARM_RAISED_IN_RIGHT_MODE') return isLeftEdge;
    if (opts.sideViolation === 'SINGLE_ARM_IN_BOTH_MODE') return isLeftEdge || isRightEdge;
    if (opts.targetJoint === 'left_shoulder') return isRightEdge;
    if (opts.targetJoint === 'right_shoulder') return isLeftEdge;
    return false;
  };

  // Primary patient live skeleton overlay
  ctx.save();
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  for (const [aIdx, bIdx] of LIVE_BODY_BONELINE_PAIRS) {
    if (!isPointDrawable(aIdx) || !isPointDrawable(bIdx) || !skeletonSafety.shouldDrawBone(aIdx, bIdx)) {
      continue;
    }
    const a = primary[aIdx]!;
    const b = primary[bIdx]!;
    const aPt = toCanvasPt(aIdx, a);
    const bPt = toCanvasPt(bIdx, b);
    const emphasized = isHighlightedEdge(aIdx, bIdx);
    const isWrongSideArmEdge = Boolean(opts.wrongSideRaised && isOppositeArmEdge(aIdx, bIdx));
    const isWrongRaisedEdge = Boolean(
      isWrongSideArmEdge || (opts.wrongPlaneRaised && emphasized),
    );

    const strokeHex = isWrongRaisedEdge
      ? '#EF4444'
      : emphasized
        ? activeColor
        : 'rgba(56, 189, 248, 0.9)';
    ctx.shadowBlur = 0;
    ctx.setLineDash([]);
    if (isWrongRaisedEdge || emphasized) {
      ctx.beginPath();
      ctx.moveTo(aPt.x, aPt.y);
      ctx.lineTo(bPt.x, bPt.y);
      ctx.lineWidth = 13;
      ctx.globalAlpha = 0.24;
      ctx.strokeStyle = strokeHex;
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    ctx.beginPath();
    if (isWrongSideArmEdge) {
      ctx.setLineDash([8, 6]);
    }
    ctx.moveTo(aPt.x, aPt.y);
    ctx.lineTo(bPt.x, bPt.y);
    ctx.lineWidth = isWrongRaisedEdge ? 6 : emphasized ? 6 : 3.5;
    ctx.strokeStyle = strokeHex;
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // Draw Stickman head circle & neck stem only in Privacy Mode (avoid covering user's real face in live camera)
  if (opts.isPrivacyMode) {
    const noseLm = primary[POSE_INDEX.NOSE];
    const lEarLm = primary[POSE_INDEX.LEFT_EAR];
    const rEarLm = primary[POSE_INDEX.RIGHT_EAR];
    const headLm =
      noseLm && (noseLm.visibility ?? 0) >= 0.18
        ? noseLm
        : lEarLm && rEarLm && (lEarLm.visibility ?? 0) >= 0.18 && (rEarLm.visibility ?? 0) >= 0.18
          ? {
            x: (lEarLm.x + rEarLm.x) / 2,
            y: (lEarLm.y + rEarLm.y) / 2,
            z: ((lEarLm.z ?? 0) + (rEarLm.z ?? 0)) / 2,
          }
          : null;
    let shoulderSpanPx = vw * Math.max(w / vw, h / vh) * 0.18;
    let neckPt: { x: number; y: number } | null = null;
    if (
      lShLm &&
      rShLm &&
      (lShLm.visibility ?? 0) >= 0.18 &&
      (rShLm.visibility ?? 0) >= 0.18
    ) {
      const lShPt = toCanvasPt(POSE_INDEX.LEFT_SHOULDER, lShLm);
      const rShPt = toCanvasPt(POSE_INDEX.RIGHT_SHOULDER, rShLm);
      shoulderSpanPx = Math.max(44, Math.hypot(lShPt.x - rShPt.x, lShPt.y - rShPt.y));
      neckPt = { x: (lShPt.x + rShPt.x) / 2, y: (lShPt.y + rShPt.y) / 2 };

      ctx.shadowBlur = 0;
      ctx.strokeStyle = 'rgba(13, 148, 136, 0.85)';
      ctx.lineWidth = 3.2;
      ctx.beginPath();
      ctx.moveTo(lShPt.x, lShPt.y);
      ctx.lineTo(neckPt.x, neckPt.y);
      ctx.lineTo(rShPt.x, rShPt.y);
      ctx.stroke();

      ctx.beginPath();
      ctx.arc(neckPt.x, neckPt.y, 4.5, 0, Math.PI * 2);
      ctx.fillStyle = '#FFFFFF';
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = activeColor;
      ctx.stroke();
    }

    if (headLm) {
      const headPt = toCanvasPt(POSE_INDEX.NOSE, headLm as NormalizedLandmark);
      const headRadius = Math.max(18, Math.min(44, shoulderSpanPx * 0.28));

      if (neckPt) {
        const dx = headPt.x - neckPt.x;
        const dy = headPt.y - neckPt.y;
        const dist = Math.hypot(dx, dy);
        if (dist > headRadius + 2) {
          const edgeX = headPt.x - (dx / dist) * headRadius;
          const edgeY = headPt.y - (dy / dist) * headRadius;
          ctx.beginPath();
          ctx.moveTo(neckPt.x, neckPt.y);
          ctx.lineTo(edgeX, edgeY);
          ctx.lineWidth = 3.2;
          ctx.strokeStyle = 'rgba(13, 148, 136, 0.9)';
          ctx.stroke();
        }
      }

      ctx.beginPath();
      ctx.arc(headPt.x, headPt.y, headRadius, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(15, 23, 42, 0.55)';
      ctx.fill();
      ctx.lineWidth = 3.2;
      ctx.strokeStyle = activeColor;
      ctx.stroke();
    }
  }
  ctx.shadowBlur = 0;

  // Hand wrist-rotation & foot stance bones
  const drawExtremityPairs = (
    pairs: readonly [number, number][],
    leftStroke: string,
    rightStroke: string,
    isLeftFn: (idx: number) => boolean,
  ) => {
    ctx.lineWidth = 2.2;
    for (const [aIdx, bIdx] of pairs) {
      if (!isPointDrawable(aIdx) || !isPointDrawable(bIdx) || !skeletonSafety.shouldDrawBone(aIdx, bIdx)) continue;
      const aLm = primary[aIdx]!;
      const bLm = primary[bIdx]!;
      const aPt = toCanvasPt(aIdx, aLm);
      const bPt = toCanvasPt(bIdx, bLm);
      ctx.beginPath();
      ctx.moveTo(aPt.x, aPt.y);
      ctx.lineTo(bPt.x, bPt.y);
      ctx.strokeStyle = isLeftFn(aIdx) ? leftStroke : rightStroke;
      ctx.stroke();
    }
  };
  drawExtremityPairs(
    LIVE_HAND_BONELINE_PAIRS,
    '#818CF8',
    '#34D399',
    (idx) =>
      idx === POSE_INDEX.LEFT_WRIST ||
      idx === POSE_INDEX.LEFT_PINKY ||
      idx === POSE_INDEX.LEFT_INDEX ||
      idx === POSE_INDEX.LEFT_THUMB,
  );
  drawExtremityPairs(
    LIVE_FOOT_BONELINE_PAIRS,
    '#38BDF8',
    '#38BDF8',
    () => true,
  );

  // Body joint keypoints (facial landmarks excluded for clean Stickman head)
  for (const idx of LIVE_KEYPOINT_INDICES) {
    if (!isPointDrawable(idx)) continue;
    const lm = primary[idx]!;
    if (lm.x < -0.02 || lm.x > 1.02 || lm.y < -0.02 || lm.y > 1.02) {
      continue;
    }
    const pt = toCanvasPt(idx, lm);
    ctx.beginPath();
    ctx.arc(pt.x, pt.y, 6, 0, Math.PI * 2);
    ctx.fillStyle = '#FFFFFF';
    ctx.fill();
    ctx.lineWidth = 2.5;
    ctx.strokeStyle = activeColor;
    ctx.stroke();
  }

  // Hand (wrist-twist rotation) & Foot keypoint nodes
  for (const idx of LIVE_EXTREMITY_KEYPOINT_INDICES) {
    if (!isPointDrawable(idx)) continue;
    const lm = primary[idx]!;
    if (lm.x < -0.02 || lm.x > 1.02 || lm.y < -0.02 || lm.y > 1.02) {
      continue;
    }
    const isFoot =
      idx === POSE_INDEX.LEFT_HEEL ||
      idx === POSE_INDEX.RIGHT_HEEL ||
      idx === POSE_INDEX.LEFT_FOOT_INDEX ||
      idx === POSE_INDEX.RIGHT_FOOT_INDEX;
    const isLeftHand =
      idx === POSE_INDEX.LEFT_PINKY ||
      idx === POSE_INDEX.LEFT_INDEX ||
      idx === POSE_INDEX.LEFT_THUMB;
    const pt = toCanvasPt(idx, lm);
    ctx.beginPath();
    ctx.arc(pt.x, pt.y, isFoot ? 4.5 : 4, 0, Math.PI * 2);
    ctx.fillStyle = isFoot ? '#38BDF8' : isLeftHand ? '#818CF8' : '#34D399';
    ctx.fill();
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = '#FFFFFF';
    ctx.stroke();
  }

  // High-precision 21-point HandLandmarker skeleton overlay strictly tethered to validated body wrist
  let hand21OutOfFrame = false;
  if (opts.handLandmarks21 && opts.handLandmarks21.length >= 21) {
    const chosenWristSide: 'left' | 'right' = (() => {
      if (resolvedDrawSide === 'LEFT' || opts.targetJoint === 'left_shoulder') return 'left';
      if (resolvedDrawSide === 'RIGHT' || opts.targetJoint === 'right_shoulder') return 'right';
      const r0 = mapVideoPointToCoverCanvas(
        opts.handLandmarks21[0].x,
        opts.handLandmarks21[0].y,
        vw,
        vh,
        w,
        h,
        true,
      );
      const lWrPt = skeletonSafety.points[POSE_INDEX.LEFT_WRIST];
      const rWrPt = skeletonSafety.points[POSE_INDEX.RIGHT_WRIST];
      const dL = skeletonSafety.validMask[POSE_INDEX.LEFT_WRIST] && lWrPt
        ? Math.hypot(r0.x - lWrPt.x, r0.y - lWrPt.y)
        : Number.POSITIVE_INFINITY;
      const dR = skeletonSafety.validMask[POSE_INDEX.RIGHT_WRIST] && rWrPt
        ? Math.hypot(r0.x - rWrPt.x, r0.y - rWrPt.y)
        : Number.POSITIVE_INFINITY;
      return dL <= dR ? 'left' : 'right';
    })();

    const handTether = validateTetheredHandLandmarks21({
      handLandmarks21: opts.handLandmarks21,
      poseLandmarks: primary,
      wristSide: chosenWristSide,
      videoW: vw,
      videoH: vh,
      canvasW: w,
      canvasH: h,
      isMirrored: true,
      armSafety: skeletonSafety,
    });
    if (handTether.valid) {
      ctx.save();
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.strokeStyle = activeColor;
      ctx.lineWidth = 3;
      ctx.shadowColor = activeColor;
      ctx.shadowBlur = 8;
      for (const [a, b] of HAND21_CONNECTIONS) {
        const ptA = handTether.canvasPoints[a];
        const ptB = handTether.canvasPoints[b];
        if (!ptA || !ptB) continue;
        ctx.beginPath();
        ctx.moveTo(ptA.x, ptA.y);
        ctx.lineTo(ptB.x, ptB.y);
        ctx.stroke();
      }
      ctx.shadowBlur = 0;
      for (let i = 0; i < 21; i += 1) {
        const pt = handTether.canvasPoints[i];
        if (!pt) continue;
        const isFingertip = i === 4 || i === 8 || i === 12 || i === 16 || i === 20;
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, isFingertip ? 4.5 : 3, 0, Math.PI * 2);
        ctx.fillStyle = isFingertip ? activeColor : '#FFFFFF';
        ctx.fill();
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = isFingertip ? '#FFFFFF' : activeColor;
        ctx.stroke();
      }
      ctx.restore();
    } else if (handTether.outOfFrame) {
      hand21OutOfFrame = true;
    }
  }

  // Show compact badge if active wrist/hand left camera frame while elbow remains inside frame
  const activeHandOutOfFrame =
    hand21OutOfFrame ||
    (isBilateralMode
      ? skeletonSafety.leftWristOutOfFrame || skeletonSafety.rightWristOutOfFrame
      : resolvedDrawSide === 'LEFT' || opts.targetJoint === 'left_shoulder'
        ? skeletonSafety.leftWristOutOfFrame
        : resolvedDrawSide === 'RIGHT' || opts.targetJoint === 'right_shoulder'
          ? skeletonSafety.rightWristOutOfFrame
          : false);
  if (activeHandOutOfFrame) {
    const anchorElbowIdx =
      resolvedDrawSide === 'LEFT' || opts.targetJoint === 'left_shoulder'
        ? POSE_INDEX.LEFT_ELBOW
        : POSE_INDEX.RIGHT_ELBOW;
    const elbowOk = skeletonSafety.validMask[anchorElbowIdx];
    const elbowPt = elbowOk ? skeletonSafety.points[anchorElbowIdx] : null;
    ctx.save();
    ctx.font = '800 12px "Sarabun", sans-serif';
    const bw = Math.min(w - 24, ctx.measureText(HAND_OUT_OF_FRAME_BADGE_TH).width + 20);
    const bh = 26;
    const bx = elbowPt ? Math.max(bw / 2 + 10, Math.min(w - bw / 2 - 10, elbowPt.x)) : w / 2;
    const by = elbowPt ? Math.max(24, Math.min(h - 24, elbowPt.y - 26)) : 52;
    ctx.fillStyle = 'rgba(217, 119, 6, 0.94)';
    ctx.strokeStyle = '#FDE68A';
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    ctx.roundRect(bx - bw / 2, by - bh / 2, bw, bh, 13);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = '#FFFFFF';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(HAND_OUT_OF_FRAME_BADGE_TH, bx, by, bw - 12);
    ctx.restore();
  }

  // Glowing Clinical Target Zone Circle + Dashed Trajectory Ray
  if (opts.overlayHint) {
    const tgtPt = {
      x: rawCanvasX(opts.overlayHint.targetNorm.x),
      y: rawCanvasY(opts.overlayHint.targetNorm.y),
    };
    const curPt = opts.overlayHint.currentNorm
      ? {
        x: rawCanvasX(opts.overlayHint.currentNorm.x),
        y: rawCanvasY(opts.overlayHint.currentNorm.y),
      }
      : null;
    const inTargetZone =
      opts.romDeg !== null &&
      opts.romDeg >= opts.targetRomDeg * 0.82 &&
      !opts.postureBannerTh &&
      !opts.wrongPlaneRaised;
    const zoneColor = opts.postureBannerTh ? '#F59E0B' : inTargetZone ? '#10B981' : '#38BDF8';
    ctx.save();
    if (curPt) {
      ctx.beginPath();
      ctx.setLineDash([6, 5]);
      ctx.moveTo(curPt.x, curPt.y);
      ctx.lineTo(tgtPt.x, tgtPt.y);
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = zoneColor;
      ctx.globalAlpha = 0.72;
      ctx.stroke();
      ctx.setLineDash([]);
    }
    ctx.shadowColor = zoneColor;
    ctx.shadowBlur = inTargetZone ? 22 : 14;
    ctx.globalAlpha = inTargetZone ? 0.34 : 0.18;
    ctx.fillStyle = zoneColor;
    ctx.beginPath();
    ctx.arc(tgtPt.x, tgtPt.y, inTargetZone ? 24 : 20, 0, Math.PI * 2);
    ctx.fill();
    ctx.globalAlpha = 0.96;
    ctx.lineWidth = 3;
    ctx.strokeStyle = zoneColor;
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(tgtPt.x, tgtPt.y, 5, 0, Math.PI * 2);
    ctx.fillStyle = '#FFFFFF';
    ctx.fill();
    ctx.restore();
  }

  // On-body floating angle badge & radial hold indicator
  const overlayJointMode = classifyExerciseJointMode(opts.exerciseMeta);
  const isHandOrWristOverlay =
    overlayJointMode === 'wrist_finger' ||
    overlayJointMode === 'forearm_turn' ||
    overlayJointMode === 'thumb_pinch';
  const anchorIdx =
    opts.targetJoint === 'left_shoulder'
      ? isHandOrWristOverlay
        ? POSE_INDEX.LEFT_WRIST
        : overlayJointMode === 'elbow' || overlayJointMode === 'shoulder_rotation'
          ? POSE_INDEX.LEFT_ELBOW
          : POSE_INDEX.LEFT_SHOULDER
      : opts.targetJoint === 'right_shoulder'
        ? isHandOrWristOverlay
          ? POSE_INDEX.RIGHT_WRIST
          : overlayJointMode === 'elbow' || overlayJointMode === 'shoulder_rotation'
            ? POSE_INDEX.RIGHT_ELBOW
            : POSE_INDEX.RIGHT_SHOULDER
        : opts.targetJoint === 'left_knee'
          ? POSE_INDEX.LEFT_KNEE
          : opts.targetJoint === 'right_knee'
            ? POSE_INDEX.RIGHT_KNEE
            : POSE_INDEX.NOSE;
  const anchorLm = primary[anchorIdx] ?? primary[POSE_INDEX.NOSE];
  if (anchorLm && opts.romDeg !== null) {
    const anchorPt = toCanvasPt(
      primary[anchorIdx] ? anchorIdx : POSE_INDEX.NOSE,
      anchorLm,
    );

    // Draw real-time geometric angle arc directly at the active joint
    if (
      opts.targetJoint === 'left_shoulder' ||
      opts.targetJoint === 'right_shoulder' ||
      opts.targetJoint === 'left_knee' ||
      opts.targetJoint === 'right_knee'
    ) {
      const isLeftSide =
        opts.targetJoint === 'left_shoulder' || opts.targetJoint === 'left_knee';
      let refLm: NormalizedLandmark | undefined;
      let movLm: NormalizedLandmark | undefined;
      if (opts.targetJoint === 'left_shoulder' || opts.targetJoint === 'right_shoulder') {
        if (overlayJointMode === 'elbow' || overlayJointMode === 'shoulder_rotation') {
          refLm = primary[isLeftSide ? POSE_INDEX.LEFT_SHOULDER : POSE_INDEX.RIGHT_SHOULDER];
          movLm = primary[isLeftSide ? POSE_INDEX.LEFT_WRIST : POSE_INDEX.RIGHT_WRIST];
        } else if (!isHandOrWristOverlay) {
          refLm = primary[isLeftSide ? POSE_INDEX.LEFT_HIP : POSE_INDEX.RIGHT_HIP];
          movLm = primary[isLeftSide ? POSE_INDEX.LEFT_ELBOW : POSE_INDEX.RIGHT_ELBOW];
        }
      } else {
        refLm = primary[isLeftSide ? POSE_INDEX.LEFT_HIP : POSE_INDEX.RIGHT_HIP];
        movLm = primary[isLeftSide ? POSE_INDEX.LEFT_ANKLE : POSE_INDEX.RIGHT_ANKLE];
      }

      if (movLm && (movLm.visibility ?? 0) >= 0.18) {
        const movPt = toCanvasPt(anchorIdx, movLm);
        const refPt =
          refLm && (refLm.visibility ?? 0) >= 0.18
            ? toCanvasPt(anchorIdx, refLm)
            : { x: anchorPt.x, y: anchorPt.y + 80 };
        const aRef = Math.atan2(refPt.y - anchorPt.y, refPt.x - anchorPt.x);
        const aMov = Math.atan2(movPt.y - anchorPt.y, movPt.x - anchorPt.x);
        let deltaAng = aMov - aRef;
        while (deltaAng <= -Math.PI) deltaAng += Math.PI * 2;
        while (deltaAng > Math.PI) deltaAng -= Math.PI * 2;
        const anticlockwise = deltaAng < 0;

        ctx.save();
        ctx.beginPath();
        ctx.moveTo(anchorPt.x, anchorPt.y);
        ctx.arc(anchorPt.x, anchorPt.y, 32, aRef, aRef + deltaAng, anticlockwise);
        ctx.closePath();
        ctx.globalAlpha = 0.24;
        ctx.fillStyle = activeColor;
        ctx.fill();
        ctx.globalAlpha = 0.92;
        ctx.beginPath();
        ctx.arc(anchorPt.x, anchorPt.y, 32, aRef, aRef + deltaAng, anticlockwise);
        ctx.lineWidth = 3;
        ctx.strokeStyle = activeColor;
        ctx.stroke();
        ctx.restore();
      }
    }

    const jx = Math.max(64, Math.min(w - 64, anchorPt.x));
    const jy = Math.max(54, Math.min(h - 40, anchorPt.y - 42));

    if (opts.holdRatio > 0) {
      ctx.beginPath();
      ctx.arc(jx, jy, 34, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * Math.min(1, opts.holdRatio));
      ctx.strokeStyle = '#10B981';
      ctx.lineWidth = 5;
      ctx.stroke();
    }

    const badgeText = `${opts.romDeg.toFixed(1)}° / ${opts.targetRomDeg}°`;
    ctx.font = '800 14px "Sarabun", sans-serif';
    const tw = ctx.measureText(badgeText).width;
    const bw = Math.max(82, tw + 20);
    const bh = 30;
    const bx = jx - bw / 2;
    const by = jy - bh / 2;

    ctx.fillStyle = 'rgba(15, 23, 42, 0.85)';
    ctx.strokeStyle = activeColor;
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    ctx.roundRect(bx, by, bw, bh, 15);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = '#FFFFFF';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(badgeText, jx, jy);
  }

  // All warning & status banners are rendered cleanly BELOW the camera viewport
  // via resolveUnifiedKioskCoachingBanner so nothing ever blocks the patient's body.
  // Multi-person camera bounding boxes when 2+ people enter view
  if (opts.isMultiPerson || landmarksList.length > 1) {
    landmarksList.forEach((person, pIdx) => {
      let minX = w;
      let maxX = 0;
      let minY = h;
      let maxY = 0;
      let any = false;
      for (const idx of LIVE_KEYPOINT_INDICES) {
        const lm = person[idx];
        if (!lm || (lm.visibility ?? 0) < minVisFor(idx)) continue;
        const pt = toCanvasPt(idx, lm);
        const cx = pt.x;
        const cy = pt.y;
        minX = Math.min(minX, cx);
        maxX = Math.max(maxX, cx);
        minY = Math.min(minY, cy);
        maxY = Math.max(maxY, cy);
        any = true;
      }
      if (!any) return;
      const x1 = Math.max(8, minX - 24);
      const y1 = Math.max(8, minY - 28);
      const bw = Math.min(w - x1 - 8, Math.max(72, maxX - minX + 48));
      const bh = Math.min(h - y1 - 8, Math.max(96, maxY - minY + 44));
      const boxHex = pIdx === 0 ? '#10B981' : '#F59E0B';
      const label = pIdx === 0 ? 'คนไข้หลัก' : 'บุคคลเกินในเฟรม';

      ctx.strokeStyle = boxHex;
      ctx.lineWidth = 3;
      ctx.setLineDash(pIdx === 0 ? [] : [8, 5]);
      ctx.strokeRect(x1, y1, bw, bh);
      ctx.setLineDash([]);

      ctx.font = '800 12px "Sarabun", sans-serif';
      const lw = ctx.measureText(label).width + 16;
      const ly = Math.max(6, y1 - 24);
      ctx.fillStyle = boxHex;
      ctx.fillRect(x1, ly, lw, 22);
      ctx.fillStyle = '#FFFFFF';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillText(label, x1 + 8, ly + 11);
    });
  }

  if (opts.smoothingStatusText) {
    ctx.font = '700 11px "Sarabun", sans-serif';
    const textW = ctx.measureText(opts.smoothingStatusText).width;
    const pillW = Math.min(w - 28, textW + 18);
    const pillH = 22;
    const pillX = 14;
    const pillY = h - 32;
    ctx.fillStyle = 'rgba(15, 23, 42, 0.72)';
    ctx.beginPath();
    ctx.roundRect(pillX, pillY, pillW, pillH, 11);
    ctx.fill();
    ctx.fillStyle = 'rgba(226, 232, 240, 0.96)';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText(opts.smoothingStatusText, pillX + 9, pillY + pillH / 2, pillW - 18);
  }
  ctx.restore();
}

export function SmartRehabKioskUI({
  patient: patientProp,
  ownerBiometrics: ownerBiometricsProp,
  prescribedExercises,
  isStaffAuthorized = false,
  isPoseTestMode = false,
  initialExerciseCode = null,
  onBackToAdminPortal,
  onExerciseCompleted,
  onAllExercisesCompleted,
  onEmergencyStop,
  onOpenStaffPortal,
  onOpenLegacyKiosk,
}: SmartRehabKioskUIProps) {
  const [testSideOverrides, setTestSideOverrides] = useState<Record<string, ExerciseSideCode>>({});
  const effectivePatient = useMemo(
    () => (patientProp !== undefined ? patientProp : isPoseTestMode ? POSE_TEST_PATIENT : undefined),
    [patientProp, isPoseTestMode],
  );
  const baseExercisesList = useMemo(() => {
    if (prescribedExercises && prescribedExercises.length > 0) {
      return prescribedExercises;
    }
    if (isPoseTestMode) {
      return buildAll15ClinicalExercisePrescriptions();
    }
    return undefined;
  }, [prescribedExercises, isPoseTestMode]);

  const effectivePrescribedExercises = useMemo(() => {
    if (!baseExercisesList) return undefined;
    if (Object.keys(testSideOverrides).length === 0) return baseExercisesList;
    return baseExercisesList.map((rx) => {
      const baseKey = rx.base_code ?? rx.catalog_exercise_id ?? rx.exercise_id;
      const sideOv = testSideOverrides[baseKey];
      return sideOv ? applyExerciseSideOverride(rx, sideOv) : rx;
    });
  }, [baseExercisesList, testSideOverrides]);

  const socketOptions = useMemo(
    () =>
      effectivePrescribedExercises || effectivePatient !== undefined || initialExerciseCode
        ? {
            prescribedExercises: effectivePrescribedExercises,
            patientOverride: effectivePatient,
            initialSelectedExerciseId: initialExerciseCode,
          }
        : undefined,
    [effectivePrescribedExercises, effectivePatient, initialExerciseCode],
  );

  const {
    payload,
    connectionStatus,
    isDemoMode,
    isEdgeDirectCamera,
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
    setDemoHoldRemainingSec,
    setDemoValidReps,
    demoStreakFrames,
    demoFaceMatched,
    autoCycleDemo,
    setAutoCycleDemo,
    triggerEmergencyStop,
    resetDemoSession,
    completeCurrentExercise,
  } = useKioskSocket('ws://localhost:8000/ws/kiosk', socketOptions);

  const sendClientLandmarksRef = useRef(sendClientLandmarks);
  sendClientLandmarksRef.current = sendClientLandmarks;

  const [voiceMuted, setVoiceMuted] = useState<boolean>(() => readStoredKioskVoiceMuted());
  useEffect(() => subscribeKioskVoiceMuted(setVoiceMuted), []);
  const handleToggleVoiceMute = () => {
    setVoiceMuted((prev) => {
      const next = writeStoredKioskVoiceMuted(!prev);
      if (next) {
        cancelAllKioskVoiceSpeech();
      }
      return next;
    });
  };
  const [zone5VoiceLang, setZone5VoiceLang] = useState<VoiceCoachLanguage>(() => {
    if (typeof window === 'undefined') return 'th';
    try {
      const saved = window.localStorage.getItem('drtechcares_voice_coach_lang_v1');
      return saved === 'en' || saved === 'zh' ? saved : 'th';
    } catch {
      return 'th';
    }
  });
  const handleSelectZone5VoiceLang = (nextLang: VoiceCoachLanguage) => {
    setZone5VoiceLang(nextLang);
    if (typeof window !== 'undefined') {
      try {
        window.localStorage.setItem('drtechcares_voice_coach_lang_v1', nextLang);
      } catch {
        // Ignore storage errors in restricted contexts
      }
    }
  };
  const [toolsPopoverOpen, setToolsPopoverOpen] = useState<boolean>(false);
  const [showPipCoach, setShowPipCoach] = useState<boolean>(true);
  const [coachModalOpen, setCoachModalOpen] = useState<boolean>(false);
  const [drawerOpen, setDrawerOpen] = useState(() => !patientProp);
  const [formRules, setFormRules] = useState<FormAssessmentRulesConfig>(() => {
    try {
      const saved =
        typeof window !== 'undefined' ? window.localStorage.getItem(FORM_ASSESSMENT_STORAGE_KEY) : null;
      if (saved) {
        const parsed = JSON.parse(saved) as Partial<FormAssessmentRulesConfig>;
        return { ...DEFAULT_FORM_ASSESSMENT_RULES, ...parsed };
      }
    } catch {
      // Ignore storage read errors
    }
    return DEFAULT_FORM_ASSESSMENT_RULES;
  });
  const [formChecklistExpanded, setFormChecklistExpanded] = useState<boolean>(true);
  const [completedIds, setCompletedIds] = useState<string[]>([]);
  const [livePoseActive, setLivePoseActive] = useState<boolean>(true);
  const [isPrivacyMode, setIsPrivacyMode] = useState<boolean>(() => {
    try {
      const saved =
        typeof window !== 'undefined'
          ? window.localStorage.getItem(PRIVACY_MODE_STORAGE_KEY)
          : null;
      if (saved === 'true') return true;
      if (saved === 'false') return false;
    } catch {
      // Ignore storage read errors
    }
    return false;
  });
  const [showSkeletonOverlay, setShowSkeletonOverlay] = useState<boolean>(() => {
    try {
      const saved =
        typeof window !== 'undefined'
          ? window.localStorage.getItem(SKELETON_OVERLAY_STORAGE_KEY)
          : null;
      if (saved === 'false') return false;
      if (saved === 'true') return true;
    } catch {
      // Ignore storage read errors
    }
    return true;
  });
  const [liveFramingHintTh, setLiveFramingHintTh] = useState<string | null>(null);
  const [strokeSafetyAlert, setStrokeSafetyAlert] = useState<{
    type: 'emergency' | 'spasticity' | 'fatigue';
    messageTh: string;
    suggestedRom?: number;
  } | null>(null);
  const [fatigueRestSec, setFatigueRestSec] = useState<number>(0);
  const [reducedRomOverrideDeg, setReducedRomOverrideDeg] = useState<number | null>(null);
  const [ownerRehabGuardAlert, setOwnerRehabGuardAlert] = useState<{
    status: OwnerLockStatus;
    text: string;
  } | null>(null);
  const isPoseTestModeRef = useRef<boolean>(isPoseTestMode);
  isPoseTestModeRef.current = isPoseTestMode;
  const ownerBiometricsRef = useRef<OwnerBiometricTemplate | null | undefined>(
    isPoseTestMode ? null : ownerBiometricsProp,
  );
  ownerBiometricsRef.current = isPoseTestMode ? null : ownerBiometricsProp;
  const manualResetNonceRef = useRef<number>(0);

  const effectiveTargetRomDeg = useMemo(() => {
    const isLowerLimb = selectedExercise.target_joint.includes('knee');
    const safeLimit =
      typeof patientProp?.max_safe_rom_deg === 'number' &&
        patientProp.max_safe_rom_deg > 0 &&
        !isLowerLimb
        ? patientProp.max_safe_rom_deg
        : 180;
    const manualReduction = reducedRomOverrideDeg ?? 180;
    return Math.max(25, Math.min(selectedExercise.target_rom_deg, safeLimit, manualReduction));
  }, [selectedExercise.target_rom_deg, selectedExercise.target_joint, patientProp?.max_safe_rom_deg, reducedRomOverrideDeg]);

  const effectiveTargetRomRef = useRef(effectiveTargetRomDeg);
  effectiveTargetRomRef.current = effectiveTargetRomDeg;
  const maxSafeRomRef = useRef<number | null>(
    typeof patientProp?.max_safe_rom_deg === 'number' && patientProp.max_safe_rom_deg > 0
      ? patientProp.max_safe_rom_deg
      : null,
  );
  maxSafeRomRef.current =
    typeof patientProp?.max_safe_rom_deg === 'number' && patientProp.max_safe_rom_deg > 0
      ? patientProp.max_safe_rom_deg
      : null;

  useEffect(() => {
    setReducedRomOverrideDeg(null);
    setStrokeSafetyAlert(null);
    setFatigueRestSec(0);
  }, [selectedExercise.exercise_id]);

  useEffect(() => {
    if (fatigueRestSec <= 0) return;
    const timer = window.setInterval(() => {
      setFatigueRestSec((prev) => {
        if (prev <= 1) {
          setStrokeSafetyAlert((cur) => (cur?.type === 'fatigue' ? null : cur));
          return 0;
        }
        return prev - 1;
      });
    }, 1000);
    return () => window.clearInterval(timer);
  }, [fatigueRestSec]);

  const isPrivacyModeRef = useRef(isPrivacyMode);
  isPrivacyModeRef.current = isPrivacyMode;
  const showSkeletonOverlayRef = useRef(showSkeletonOverlay);
  showSkeletonOverlayRef.current = showSkeletonOverlay;

  const canShowManualRepButton = useMemo(() => {
    if (isDemoMode || isStaffAuthorized) return true;
    if (typeof window === 'undefined') return false;
    const params = new URLSearchParams(window.location.search);
    return params.get('debug') === '1' || params.get('debug') === 'true';
  }, [isDemoMode, isStaffAuthorized]);

  const handleTogglePrivacyMode = () => {
    const next = !isPrivacyMode;
    setIsPrivacyMode(next);
    try {
      window.localStorage.setItem(PRIVACY_MODE_STORAGE_KEY, String(next));
    } catch {
      // Ignore storage write errors
    }
    if (!next && isDemoMode) {
      setAutoCycleDemo(false);
      setLivePoseActive(true);
      toggleDemoMode(false);
    }
  };

  const handleToggleSkeletonOverlay = () => {
    const next = !showSkeletonOverlay;
    setShowSkeletonOverlay(next);
    try {
      window.localStorage.setItem(SKELETON_OVERLAY_STORAGE_KEY, String(next));
    } catch {
      // Ignore storage write errors
    }
  };

  const handleToggleDemoMode = () => {
    if (isDemoMode) {
      setAutoCycleDemo(false);
      setLivePoseActive(true);
      toggleDemoMode(false);
    } else {
      setLivePoseActive(false);
      setLiveFramingHintTh(null);
      toggleDemoMode(true);
    }
  };
  const [cameraReady, setCameraReady] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [repSuccessToastTh, setRepSuccessToastTh] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const poseCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const videoContainerRef = useRef<HTMLDivElement | null>(null);
  const canvasSizeRef = useRef<{ w: number; h: number }>({ w: 640, h: 480 });
  const holdProgressBarRef = useRef<HTMLDivElement | null>(null);
  const holdTimerTextRef = useRef<HTMLDivElement | null>(null);

  // Zero-stale-closure 60FPS Refs for Keypoints, Hold Timer & Rep State Machine
  const rawKeypointsRef = useRef<NormalizedLandmark[] | null>(null);
  const smoothedRenderKeypointsRef = useRef<NormalizedLandmark[] | null>(null);
  const allRawKeypointsRef = useRef<NormalizedLandmark[][]>([]);
  const latestKinematicsRef = useRef<LiveJointMeasurement | null>(null);
  const holdElapsedSecRef = useRef<number>(0);
  const completedRepsRef = useRef<number>(0);
  const repPhaseRef = useRef<StrokeHoldRepState['phase']>('PHASE_1_HOLDING');
  const loweredFrameStreakRef = useRef<number>(0);
  const readyBannerUntilRef = useRef<number>(0);
  const waitLowerStartedAtRef = useRef<number>(0);
  const cooldownStartedAtRef = useRef<number>(0);
  const repCelebrateUntilRef = useRef<number>(0);
  const lastFrameTimeRef = useRef<number>(0);
  const repSuccessToastRef = useRef<string | null>(null);
  const [holdElapsedUiSec, setHoldElapsedUiSec] = useState<number>(0);
  const [isWaitLowerArmUi, setIsWaitLowerArmUi] = useState<boolean>(false);

  useEffect(() => {
    completedRepsRef.current = 0;
    holdElapsedSecRef.current = 0;
    repPhaseRef.current = 'PHASE_1_HOLDING';
    loweredFrameStreakRef.current = 0;
    readyBannerUntilRef.current = 0;
    waitLowerStartedAtRef.current = 0;
    cooldownStartedAtRef.current = 0;
    repCelebrateUntilRef.current = 0;
    rawKeypointsRef.current = null;
    smoothedRenderKeypointsRef.current = null;
    allRawKeypointsRef.current = [];
    repSuccessToastRef.current = null;
    setHoldElapsedUiSec(0);
    setIsWaitLowerArmUi(false);
    setRepSuccessToastTh(null);
    if (holdProgressBarRef.current) {
      holdProgressBarRef.current.style.width = '0%';
    }
    if (holdTimerTextRef.current) {
      holdTimerTextRef.current.textContent = '0.0s';
    }
  }, [selectedExercise.exercise_id, selectedExercise.hold_sec]);

  useEffect(() => {
    if (
      demoValidReps === 0 &&
      completedRepsRef.current > 0 &&
      repPhaseRef.current === 'PHASE_1_HOLDING' &&
      holdElapsedSecRef.current === 0
    ) {
      completedRepsRef.current = 0;
      holdElapsedSecRef.current = 0;
      repPhaseRef.current = 'PHASE_1_HOLDING';
      loweredFrameStreakRef.current = 0;
      readyBannerUntilRef.current = 0;
      waitLowerStartedAtRef.current = 0;
    } else if (demoValidReps > completedRepsRef.current) {
      completedRepsRef.current = demoValidReps;
    }
  }, [demoValidReps]);

  // Keep `<canvas>` dimensions synchronized 1:1 with `<div ref={videoContainerRef}>` via ResizeObserver
  useEffect(() => {
    if (typeof ResizeObserver === 'undefined') return;
    const container = videoContainerRef.current;
    if (!container) return;
    const syncDims = () => {
      const cw = Math.round(container.clientWidth || 640);
      const ch = Math.round(container.clientHeight || 480);
      if (cw > 0 && ch > 0) {
        canvasSizeRef.current = { w: cw, h: ch };
        const cvs = poseCanvasRef.current;
        if (cvs) {
          const dpr = Math.min(window.devicePixelRatio || 1, 2);
          const targetW = Math.round(cw * dpr);
          const targetH = Math.round(ch * dpr);
          if (cvs.width !== targetW || cvs.height !== targetH) {
            cvs.width = targetW;
            cvs.height = targetH;
          }
        }
      }
    };
    syncDims();
    const ro = new ResizeObserver(syncDims);
    ro.observe(container);
    return () => ro.disconnect();
  }, [livePoseActive, isEdgeDirectCamera, cameraError]);

  const onExerciseCompletedRef = useRef(onExerciseCompleted);
  const onAllExercisesCompletedRef = useRef(onAllExercisesCompleted);
  const selectedExerciseRef = useRef(selectedExercise);
  const demoValidRepsRef = useRef(demoValidReps);
  const demoPresetRef = useRef(demoPreset);
  const formRulesRef = useRef<FormAssessmentRulesConfig>(formRules);
  const countingLockedRef = useRef(false);

  onExerciseCompletedRef.current = onExerciseCompleted;
  onAllExercisesCompletedRef.current = onAllExercisesCompleted;
  selectedExerciseRef.current = selectedExercise;
  demoValidRepsRef.current = demoValidReps;
  demoPresetRef.current = demoPreset;
  formRulesRef.current = formRules;
  countingLockedRef.current =
    !isPoseTestMode &&
    isFormRuleActive(formRules, 'identity') &&
    (payload.kiosk_state === 'PAUSED_MULTI_PERSON' ||
      payload.kiosk_state === 'PAUSED_IMPOSTOR' ||
      payload.identity_guard?.is_counting_allowed === false);

  useEffect(() => {
    try {
      if (typeof window !== 'undefined') {
        window.localStorage.setItem(FORM_ASSESSMENT_STORAGE_KEY, JSON.stringify(formRules));
      }
    } catch {
      // Ignore storage write errors
    }
  }, [formRules]);

  useEffect(() => {
    if (
      (demoPreset === 'WRONG_LIMB' && !isFormRuleActive(formRules, 'limb')) ||
      (demoPreset === 'WRONG_PLANE' && !isFormRuleActive(formRules, 'plane')) ||
      (demoPreset === 'TRUNK_LEAN' && !isFormRuleActive(formRules, 'trunk')) ||
      (demoPreset === 'MULTI_PERSON' && !isFormRuleActive(formRules, 'identity'))
    ) {
      setDemoPresetState('ACTIVE_REP');
    }
  }, [formRules, demoPreset, setDemoPresetState]);

  const toggleMasterFormAssessment = () => {
    setFormRules((prev) => {
      const nextEnabled = !prev.enabled;
      if (nextEnabled) {
        setFormChecklistExpanded(true);
      }
      return { ...prev, enabled: nextEnabled };
    });
  };

  const toggleSingleFormRule = (rule: FormAssessmentRuleKey) => {
    setFormRules((prev) => {
      if (!prev.enabled) {
        return { ...prev, enabled: true, [rule]: true };
      }
      return { ...prev, [rule]: !prev[rule] };
    });
  };

  const resetAllFormRules = () => {
    setFormRules(DEFAULT_FORM_ASSESSMENT_RULES);
    setFormChecklistExpanded(true);
  };

  const { kiosk_state, patient, telemetry_analysis, rep_counter, ui_directives, tts_voice_th } = payload;
  const compensations = telemetry_analysis.compensation_detected;
  // Strictly passive trunk lean indicator (> 18 deg) in live mode; shoulder hike never blocks or penalizes live timer.
  const trunkLeanAlert =
    isFormRuleActive(formRules, 'trunk') &&
    (isDemoMode ? compensations.includes('TRUNK_LEAN') : (telemetry_analysis.trunk_tilt_deg ?? 0) > 18);
  const shoulderHikeAlert =
    isFormRuleActive(formRules, 'posture') && isDemoMode && compensations.includes('SHOULDER_HIKE');
  const tooFastAlert = isFormRuleActive(formRules, 'velocity') && isDemoMode && compensations.includes('TOO_FAST');
  const wrongLimbAlert = isFormRuleActive(formRules, 'limb') && compensations.includes('WRONG_LIMB');
  const wrongPlaneAlert = isFormRuleActive(formRules, 'plane') && compensations.includes('WRONG_PLANE');
  const activeExpectedPlane = resolveExpectedPlaneForExercise(selectedExercise);
  const isHoldPhase = kiosk_state === 'HOLD_PHASE';
  const isRepCelebration = kiosk_state === 'REP_COMPLETED' || kiosk_state === 'SET_FINISHED';
  const isEmergency = kiosk_state === 'EMERGENCY_HALT';
  const isMultiPersonState =
    kiosk_state === 'PAUSED_MULTI_PERSON' || payload.identity_guard?.status === 'MULTI_PERSON';
  const isImpostorState =
    kiosk_state === 'PAUSED_IMPOSTOR' || payload.identity_guard?.status === 'IMPOSTOR';
  const unifiedCoachingBanner = resolveUnifiedKioskCoachingBanner({
    exercise: selectedExercise,
    expectedPlane: activeExpectedPlane,
    kioskState: kiosk_state,
    wrongLimbAlert,
    wrongPlaneAlert,
    trunkLeanAlert,
    shoulderHikeAlert,
    tooFastAlert,
    isMultiPerson: isMultiPersonState,
    isImpostor: isImpostorState,
    ownerRehabGuardAlert,
    strokeSafetyAlert,
    fatigueRestSec,
    safetyGuardBannerTh: telemetry_analysis.safety_guard_banner_th,
    framingHintTh: liveFramingHintTh,
    repSuccessToastTh,
    holdRemainingSec: rep_counter.hold_remaining_sec,
    holdTargetSec: selectedExercise.hold_sec,
    currentRomDeg: telemetry_analysis.current_rom_deg,
    targetRomDeg: effectiveTargetRomDeg,
    validReps: rep_counter.valid_reps,
    targetReps: selectedExercise.target_reps,
    calibrationMessageTh:
      !isDemoMode && livePoseActive && !isEdgeDirectCamera && !cameraReady && !cameraError
        ? 'กำลังเปิดกล้องตรวจจับท่าทาง...'
        : payload.calibration && !payload.calibration.is_calibrated
          ? payload.calibration.message_th
          : null,
  });

  const [voiceCoachSpeaking, setVoiceCoachSpeaking] = useState<boolean>(false);
  const [mascotGreetingActive, setMascotGreetingActive] = useState<boolean>(true);
  const [mascotEncouragingActive, setMascotEncouragingActive] = useState<boolean>(false);
  const prevMascotRepsRef = useRef<number>(rep_counter.valid_reps);
  const encouragingTimerRef = useRef<number | null>(null);

  useEffect(() => {
    setMascotGreetingActive(true);
    const greetTimer = window.setTimeout(() => {
      setMascotGreetingActive(false);
    }, MASCOT_GREETING_DURATION_MS);
    return () => window.clearTimeout(greetTimer);
  }, [patient?.id]);

  useEffect(() => {
    prevMascotRepsRef.current = 0;
  }, [selectedExercise.exercise_id]);

  useEffect(() => {
    const scoredToast =
      typeof repSuccessToastTh === 'string' &&
      (repSuccessToastTh.includes('+1') || repSuccessToastTh.includes('สำเร็จ'));
    const gainedRep = rep_counter.valid_reps > prevMascotRepsRef.current;
    if (
      gainedRep ||
      kiosk_state === 'REP_COMPLETED' ||
      kiosk_state === 'SET_FINISHED' ||
      scoredToast
    ) {
      prevMascotRepsRef.current = rep_counter.valid_reps;
      setMascotEncouragingActive(true);
      if (encouragingTimerRef.current !== null) {
        window.clearTimeout(encouragingTimerRef.current);
      }
      encouragingTimerRef.current = window.setTimeout(() => {
        encouragingTimerRef.current = null;
        setMascotEncouragingActive(false);
      }, MASCOT_ENCOURAGING_DURATION_MS);
    } else if (rep_counter.valid_reps < prevMascotRepsRef.current) {
      prevMascotRepsRef.current = rep_counter.valid_reps;
    }
  }, [rep_counter.valid_reps, kiosk_state, repSuccessToastTh]);

  useEffect(() => {
    return () => {
      if (encouragingTimerRef.current !== null) {
        window.clearTimeout(encouragingTimerRef.current);
        encouragingTimerRef.current = null;
      }
    };
  }, []);

  const handleEmergencyStopClick = (customReason?: string) => {
    const reasonMsg =
      typeof customReason === 'string' && customReason.trim().length > 0
        ? customReason.trim()
        : `กดปุ่มหยุดฉุกเฉินระหว่างฝึกท่า ${selectedExercise.name_th}`;
    if (reasonMsg.includes('เสียสมดุล')) {
      setStrokeSafetyAlert({
        type: 'emergency',
        messageTh: 'ตรวจพบการเสียสมดุล กรุณานั่งพักและจับราวจับ',
      });
    }
    triggerEmergencyStop();
    onEmergencyStop?.({
      reason: reasonMsg,
      patientId: patient?.id ?? patientProp?.id ?? undefined,
      exerciseNameTh: selectedExercise.name_th,
    });
  };
  const triggerBalanceEmergencyRef = useRef(handleEmergencyStopClick);
  triggerBalanceEmergencyRef.current = handleEmergencyStopClick;

  const markCurrentExerciseCompleted = () => {
    completeCurrentExercise();
    if (!completedIds.includes(selectedExercise.exercise_id)) {
      setCompletedIds((prev) =>
        prev.includes(selectedExercise.exercise_id)
          ? prev
          : [...prev, selectedExercise.exercise_id],
      );
      onExerciseCompletedRef.current?.(selectedExercise);
    }
  };

  const exerciseListSignature = exerciseOptions.map((item) => item.exercise_id).join('|');
  useEffect(() => {
    setCompletedIds([]);
  }, [exerciseListSignature]);

  const currentExerciseIndex = Math.max(
    0,
    exerciseOptions.findIndex((opt) => opt.exercise_id === selectedExercise.exercise_id),
  );
  const allExercisesCompleted =
    exerciseOptions.length > 0 && completedIds.length >= exerciseOptions.length;
  const nextUncompletedExercise = useMemo(() => {
    const afterCurrent = exerciseOptions.find(
      (opt, idx) =>
        idx > currentExerciseIndex &&
        opt.exercise_id !== selectedExercise.exercise_id &&
        !completedIds.includes(opt.exercise_id),
    );
    if (afterCurrent) return afterCurrent;
    return exerciseOptions.find(
      (opt) => opt.exercise_id !== selectedExercise.exercise_id && !completedIds.includes(opt.exercise_id),
    );
  }, [exerciseOptions, currentExerciseIndex, selectedExercise.exercise_id, completedIds]);

  // Mark exercise completed once target reps are reached
  useEffect(() => {
    if (rep_counter.valid_reps < selectedExercise.target_reps) return;
    if (completedIds.includes(selectedExercise.exercise_id)) return;
    setCompletedIds((prev) =>
      prev.includes(selectedExercise.exercise_id) ? prev : [...prev, selectedExercise.exercise_id],
    );
    onExerciseCompletedRef.current?.(selectedExercise);
  }, [rep_counter.valid_reps, selectedExercise, completedIds]);

  // Auto-advance to the next uncompleted exercise after completing one set
  useEffect(() => {
    if (!completedIds.includes(selectedExercise.exercise_id)) return;
    if (allExercisesCompleted) {
      if (!onAllExercisesCompletedRef.current) return;
      const doneTimer = window.setTimeout(() => {
        onAllExercisesCompletedRef.current?.();
      }, 2200);
      return () => window.clearTimeout(doneTimer);
    }
    if (!nextUncompletedExercise) return;
    const nextTimer = window.setTimeout(() => {
      setSelectedExercise(nextUncompletedExercise);
    }, 1800);
    return () => window.clearTimeout(nextTimer);
  }, [
    completedIds,
    selectedExercise.exercise_id,
    allExercisesCompleted,
    nextUncompletedExercise,
    setSelectedExercise,
  ]);

  // Live MediaPipe PoseLandmarker camera tracking (skip when Python Edge Vision owns the physical webcam)
  useEffect(() => {
    if (!livePoseActive || autoCycleDemo || isEmergency || isEdgeDirectCamera) {
      setCameraReady(false);
      return;
    }

    let active = true;
    let stream: MediaStream | null = null;
    let rafId = 0;
    let inferenceTimerId = 0;
    let poseLandmarker: PoseLandmarker | null = null;
    let faceLandmarker: FaceLandmarker | null = null;
    let handLandmarker: HandLandmarker | null = null;
    let faceAlignCanvas: HTMLCanvasElement | null = null;
    let faceRoiCanvas: HTMLCanvasElement | null = null;
    let aiCaptureCanvas: HTMLCanvasElement | null = null;
    let lastFaceTs = 0;
    let lastHandTs = 0;
    let ownerFaceCycle = 0;
    let cachedOwnerVerified = true;
    let cachedOwnerFaceCount = 0;
    let cachedOwnerMissingStreak = 0;
    let cachedOwnerFaceBox: OwnerFaceBoxMetrics | null = null;
    let cachedOwnerExtraBoxes: OwnerFaceBoxMetrics[] = [];
    let lastOwnerRehabKey = '';
    let lastValidPoseAt = 0;
    let lastDetectTs = 0;
    let lastFrameAt = 0;
    let smoothedRom: number | null = 0;
    let smoothedTilt = 0;
    let singlePersonRecoveryFrames = 0;
    let trackedExId = selectedExerciseRef.current.exercise_id;
    let trackedResetNonce = manualResetNonceRef.current;
    let torsoAnchor: TorsoAnchorState | null = null;
    const landmarkSmoother = new LandmarkSmoother({
      alpha: 0.25,
      adaptive: true,
      minAlpha: 0.1,
      maxAlpha: 0.85,
      velocityScale: 15.0,
    });
    const pose2dOneEuroFilter = new ClinicalOneEuroLandmarkFilter(1.15, 0.06, 1.0);
    const worldOneEuroFilter = new ClinicalOneEuroLandmarkFilter(1.15, 0.06, 1.0);
    const handOneEuroFilter = new ClinicalHandOneEuroFilter(1.15, 0.06, 1.0);
    const boneBaselineCalibrator = new AutoBoneBaselineCalibrator();
    let latestHandLandmarksBySide: { left?: NormalizedLandmark[]; right?: NormalizedLandmark[] } = {};
    let latestOverlayHint: StrokeTargetOverlayHint | null = null;
    let smoothingMode: 'one_euro' | 'adaptive_ema' | 'fixed_ema' = 'one_euro';
    let enableCustomSmooth = true;
    const formatSmoothingHud = () => {
      if (!enableCustomSmooth) {
        return `Smoothing: OFF • [1] On/Off [2] Mode [+/-] Alpha`;
      }
      if (smoothingMode === 'one_euro') {
        return `Smoothing: 1€ Filter (Heavy 3D+2D) • [1] On/Off [2] Mode [+/-] Alpha`;
      }
      if (smoothingMode === 'adaptive_ema') {
        return `Smoothing: Adaptive EMA (min=${landmarkSmoother.minAlpha.toFixed(2)}, max=${landmarkSmoother.maxAlpha.toFixed(2)}) • [1] On/Off [2] Mode [+/-] Alpha`;
      }
      return `Smoothing: Fixed EMA (alpha=${landmarkSmoother.alpha.toFixed(2)}) • [1] On/Off [2] Mode [+/-] Alpha`;
    };
    const handleSmoothingKeyDown = (event: KeyboardEvent) => {
      const targetEl = event.target as HTMLElement | null;
      if (
        targetEl &&
        (targetEl.tagName === 'INPUT' ||
          targetEl.tagName === 'TEXTAREA' ||
          targetEl.tagName === 'SELECT' ||
          targetEl.isContentEditable)
      ) {
        return;
      }
      if (event.key === '1') {
        enableCustomSmooth = !enableCustomSmooth;
        if (!enableCustomSmooth) {
          landmarkSmoother.reset();
          pose2dOneEuroFilter.reset();
          worldOneEuroFilter.reset();
          handOneEuroFilter.reset();
        }
      } else if (event.key === '2') {
        smoothingMode =
          smoothingMode === 'one_euro'
            ? 'adaptive_ema'
            : smoothingMode === 'adaptive_ema'
              ? 'fixed_ema'
              : 'one_euro';
        landmarkSmoother.adaptive = smoothingMode === 'adaptive_ema';
      } else if (event.key === '+' || event.key === '=') {
        landmarkSmoother.alpha = Math.min(0.95, Number((landmarkSmoother.alpha + 0.05).toFixed(2)));
      } else if (event.key === '-' || event.key === '_') {
        landmarkSmoother.alpha = Math.max(0.05, Number((landmarkSmoother.alpha - 0.05).toFixed(2)));
      }
    };
    window.addEventListener('keydown', handleSmoothingKeyDown);
    let occludedFrames = 0;
    let lastVideoTime = -1;
    let lastDetectWallMs = 0;
    let wrongSideStreak = 0;
    let wrongPlaneStreak = 0;
    let sustainedSwayFrames = 0;
    let consecutiveCompEvents = 0;
    let prevHeadY: number | null = null;
    let prevShoulderMidY: number | null = null;
    let prevSafetySampleAt = 0;
    let prevVelocityRom: number | null = null;
    let lastSpasticityAt = 0;
    let multiPersonStartAtMs = 0;
    let latestRawPrimaryRomDeg: number | null = null;
    let latestLeftArmRom: number | null = null;
    let latestRightArmRom: number | null = null;
    let targetRomDeg: number | null = null;
    let latestLandmarks: NormalizedLandmark[] | null = null;
    let renderPrimaryLandmarks: NormalizedLandmark[] | null = null;
    let latestAllLandmarks: NormalizedLandmark[][] = [];
    let latestPoseCount = 0;
    let latestRawTiltDeg = 0;
    let latestTrunkTiltDeg = 0;
    let latestSideViolation: StrokeSideViolationCode = 'NONE';
    let latestWrongSideRaised = false;
    let latestWrongSideBannerTh: string | null = null;
    let latestWrongPlaneRaised = false;
    let latestWrongPlaneBannerTh: string | null = null;
    let latestLowVisibilityAlertTh: string | null = null;
    let latestPostureCompensated = false;
    let latestPostureBannerTh: string | null = null;
    let lastEmittedRom = -999;
    let lastEmittedTilt = -999;
    let lastEmittedHold = -999;
    let lastEmittedDistanceM = -1;
    let lastEmittedFramingHint: string | null = null;
    let lastRomEmitAt = 0;
    let lastTiltEmitAt = 0;
    let lastHoldEmitAt = 0;
    let lastDistEmitAt = 0;
    let lastEdgeBridgeSentAt = 0;

    const cleanup = () => {
      window.removeEventListener('keydown', handleSmoothingKeyDown);
      landmarkSmoother.reset();
      pose2dOneEuroFilter.reset();
      worldOneEuroFilter.reset();
      handOneEuroFilter.reset();
      boneBaselineCalibrator.reset();
      rawKeypointsRef.current = null;
      smoothedRenderKeypointsRef.current = null;
      allRawKeypointsRef.current = [];
      setLiveFramingHintTh(null);
      if (inferenceTimerId) {
        window.clearInterval(inferenceTimerId);
        inferenceTimerId = 0;
      }
      if (rafId) {
        cancelAnimationFrame(rafId);
        rafId = 0;
      }
      if (poseCanvasRef.current) {
        const ctx = poseCanvasRef.current.getContext('2d');
        ctx?.clearRect(0, 0, poseCanvasRef.current.width, poseCanvasRef.current.height);
      }
      if (stream) {
        stream.getTracks().forEach((t) => t.stop());
        stream = null;
      }
      if (poseLandmarker) {
        try {
          poseLandmarker.close();
        } catch {
          // ignore close errors
        }
        poseLandmarker = null;
      }
      if (faceLandmarker) {
        try {
          faceLandmarker.close();
        } catch {
          // ignore close errors
        }
        faceLandmarker = null;
      }
      if (handLandmarker) {
        try {
          handLandmarker.close();
        } catch {
          // ignore close errors
        }
        handLandmarker = null;
      }
      setOwnerRehabGuardAlert(null);
    };

    async function initPoseCamera() {
      try {
        if (typeof navigator === 'undefined' || !navigator.mediaDevices?.getUserMedia) {
          setCameraError('เบราว์เซอร์ไม่รองรับกล้อง (สามารถฝึกผ่านโหมดควบคุมได้)');
          return;
        }
        setCameraError(null);
        // Brief pause allows the previous step's camera stream to release cleanly on Windows
        await new Promise((r) => setTimeout(r, 180));
        if (!active) return;

        const deviceId = await selectClinicalCameraDeviceId();
        if (!active) return;

        const baseVideoConstraints: MediaTrackConstraints = {
          width: { ideal: 640 },
          height: { ideal: 360 },
          aspectRatio: { ideal: 16 / 9 },
          frameRate: { ideal: 60, max: 120 },
          facingMode: 'user',
        };
        stream = await navigator.mediaDevices.getUserMedia({
          video: deviceId
            ? {
              deviceId: { exact: deviceId },
              ...baseVideoConstraints,
            }
            : baseVideoConstraints,
          audio: false,
        });
        if (!active || !videoRef.current) {
          cleanup();
          return;
        }
        const videoTrack = stream.getVideoTracks()[0];
        const caps = videoTrack?.getCapabilities?.() as { zoom?: { min?: number } } | undefined;
        if (videoTrack && caps?.zoom && typeof caps.zoom.min === 'number') {
          await videoTrack
            .applyConstraints({
              advanced: [{ zoom: caps.zoom.min } as MediaTrackConstraintSet],
            })
            .catch(() => undefined);
        }
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        if (!active) {
          cleanup();
          return;
        }
        setCameraReady(true);

        let vision;
        for (const wasmBase of [
          '/mediapipe/wasm',
          '/models/wasm',
          'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304/wasm',
        ]) {
          try {
            vision = await FilesetResolver.forVisionTasks(wasmBase);
            break;
          } catch {
            // Try next WASM base URL
          }
        }
        if (!vision || !active) return;

        const POSE_MODEL_PATHS = [
          'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task',
          '/models/pose_landmarker_full.task',
          '/models/pose_landmarker.task',
        ];
        for (const minConf of [0.7, 0.5] as const) {
          for (const modelPath of POSE_MODEL_PATHS) {
            for (const delegate of ['GPU', 'CPU', undefined] as const) {
              try {
                const created = await PoseLandmarker.createFromOptions(vision, {
                  baseOptions: {
                    modelAssetPath: modelPath,
                    ...(delegate ? { delegate } : {}),
                  },
                  runningMode: 'VIDEO',
                  numPoses: 1,
                  minPoseDetectionConfidence: minConf,
                  minPosePresenceConfidence: minConf,
                  minTrackingConfidence: minConf,
                });
                if (!active) {
                  created.close();
                  return;
                }
                poseLandmarker = created;
                break;
              } catch {
                // Try fallback
              }
            }
            if (poseLandmarker) break;
          }
          if (poseLandmarker) break;
        }
        if (!poseLandmarker || !active) return;

        // Lightweight FaceLandmarker for Continuous 1:1 Account Owner Face Lock every 5 frames
        try {
          for (const delegate of ['GPU', 'CPU', undefined] as const) {
            try {
              const fl = await FaceLandmarker.createFromOptions(vision, {
                baseOptions: delegate
                  ? { modelAssetPath: '/models/face_landmarker.task', delegate }
                  : { modelAssetPath: '/models/face_landmarker.task' },
                runningMode: 'VIDEO',
                numFaces: 2,
                outputFaceBlendshapes: true,
                minFaceDetectionConfidence: 0.45,
                minFacePresenceConfidence: 0.45,
                minTrackingConfidence: 0.45,
              });
              if (!active) {
                fl.close();
                return;
              }
              faceLandmarker = fl;
              break;
            } catch {
              // try next delegate
            }
          }
        } catch {
          faceLandmarker = null;
        }

        // 21-point HandLandmarker for Wrist/Finger extension, Forearm pronation/supination, and Thumb pinch
        try {
          for (const delegate of ['GPU', 'CPU', undefined] as const) {
            try {
              const hl = await HandLandmarker.createFromOptions(vision, {
                baseOptions: delegate
                  ? { modelAssetPath: '/models/hand_landmarker.task', delegate }
                  : { modelAssetPath: '/models/hand_landmarker.task' },
                runningMode: 'VIDEO',
                numHands: 2,
                minHandDetectionConfidence: 0.45,
                minHandPresenceConfidence: 0.45,
                minTrackingConfidence: 0.45,
              });
              if (!active) {
                hl.close();
                return;
              }
              handLandmarker = hl;
              break;
            } catch {
              // try next delegate
            }
          }
        } catch {
          handLandmarker = null;
        }

        const emitPresetState = (nextPreset: DemoPresetId) => {
          if (demoPresetRef.current !== nextPreset) {
            demoPresetRef.current = nextPreset;
            setDemoPresetState(nextPreset);
          }
        };

        const resetExerciseTrackingState = (nextExId: string) => {
          trackedExId = nextExId;
          torsoAnchor = null;
          landmarkSmoother.reset();
          pose2dOneEuroFilter.reset();
          worldOneEuroFilter.reset();
          handOneEuroFilter.reset();
          boneBaselineCalibrator.reset();
          latestHandLandmarksBySide = {};
          latestOverlayHint = null;
          rawKeypointsRef.current = null;
          smoothedRenderKeypointsRef.current = null;
          allRawKeypointsRef.current = [];
          occludedFrames = 0;
          wrongSideStreak = 0;
          wrongPlaneStreak = 0;
          sustainedSwayFrames = 0;
          consecutiveCompEvents = 0;
          prevHeadY = null;
          prevShoulderMidY = null;
          prevSafetySampleAt = 0;
          prevVelocityRom = null;
          latestRawPrimaryRomDeg = null;
          latestLeftArmRom = null;
          latestRightArmRom = null;
          targetRomDeg = null;
          smoothedRom = 0;
          smoothedTilt = 0;
          lastValidPoseAt = 0;
          latestPoseCount = 0;
          latestRawTiltDeg = 0;
          latestTrunkTiltDeg = 0;
          latestSideViolation = 'NONE';
          latestWrongSideRaised = false;
          latestWrongSideBannerTh = null;
          latestWrongPlaneRaised = false;
          latestWrongPlaneBannerTh = null;
          latestLowVisibilityAlertTh = null;
          latestPostureCompensated = false;
          latestPostureBannerTh = null;
          lastEmittedRom = -999;
          lastEmittedTilt = -999;
          lastEmittedHold = -999;
          lastRomEmitAt = 0;
          lastTiltEmitAt = 0;
          lastHoldEmitAt = 0;
          multiPersonStartAtMs = 0;
          holdElapsedSecRef.current = 0;
          completedRepsRef.current = 0;
          repPhaseRef.current = 'PHASE_1_HOLDING';
          loweredFrameStreakRef.current = 0;
          readyBannerUntilRef.current = 0;
          waitLowerStartedAtRef.current = 0;
          cooldownStartedAtRef.current = 0;
          repCelebrateUntilRef.current = 0;
          repSuccessToastRef.current = null;
          setHoldElapsedUiSec(0);
          setIsWaitLowerArmUi(false);
          setRepSuccessToastTh(null);
        };

        // Decoupled AI Inference Loop (downscaled 384x216 capture canvas for 30+ FPS CPU/GPU speed)
        const runAiInference = () => {
          if (!active || !videoRef.current || !poseLandmarker) return;
          const curEx = selectedExerciseRef.current;
          const curExpectedPlane = resolveExpectedPlaneForExercise(curEx);
          if (trackedExId !== curEx.exercise_id || trackedResetNonce !== manualResetNonceRef.current) {
            trackedResetNonce = manualResetNonceRef.current;
            resetExerciseTrackingState(curEx.exercise_id);
          }

          const v = videoRef.current;
          const now = performance.now();
          if (
            v.readyState < 2 ||
            v.videoWidth <= 0 ||
            (v.currentTime === lastVideoTime && now - lastDetectWallMs < 32)
          ) {
            return;
          }
          lastVideoTime = v.currentTime;
          lastDetectWallMs = now;

          try {
            const ts = Math.max(lastDetectTs + 1, Math.round(now));
            lastDetectTs = ts;

            if (!aiCaptureCanvas) {
              aiCaptureCanvas = document.createElement('canvas');
            }
            const capW = 384;
            const capH = Math.max(
              216,
              Math.round(capW * ((v.videoHeight || 360) / (v.videoWidth || 640))),
            );
            if (aiCaptureCanvas.width !== capW || aiCaptureCanvas.height !== capH) {
              aiCaptureCanvas.width = capW;
              aiCaptureCanvas.height = capH;
            }
            const aiCtx = aiCaptureCanvas.getContext('2d', { alpha: false });
            let poseSourceForFrame: HTMLVideoElement | HTMLCanvasElement = v;
            if (aiCtx) {
              aiCtx.drawImage(v, 0, 0, capW, capH);
              poseSourceForFrame = aiCaptureCanvas;
            }

            let poseRes;
            try {
              poseRes = poseLandmarker.detectForVideo(poseSourceForFrame, ts);
            } catch {
              poseRes = poseLandmarker.detectForVideo(v, ts);
              poseSourceForFrame = v;
            }

            const allLandmarks = poseRes?.landmarks ?? [];
            latestPoseCount = allLandmarks.length;

            const stepARules = formRulesRef.current;
            const stepACheckIdentity = !isPoseTestModeRef.current && isFormRuleActive(stepARules, 'identity');

            // Active Single-Person Exercise Lock:
            // Once logged in, ALWAYS treat single-person frames (personCount <= 1) as verified owner so
            // low-res 1.5m-3.0m distance crops NEVER lock out reps or flash "ไม่ใช่เจ้าของบัญชี".
            // Only pause for multi-person intrusion when >= 2 distinct people are continuously present > 1.5s.
            ownerFaceCycle += 1;
            if (faceLandmarker && stepACheckIdentity && (ownerFaceCycle === 1 || ownerFaceCycle % 8 === 0)) {
              try {
                const fTs = Math.max(lastFaceTs + 1, lastDetectTs + 1, Math.round(now));
                lastFaceTs = fTs;
                const fRes = faceLandmarker.detectForVideo(v, fTs);
                const rawFaces = fRes.faceLandmarks ?? [];
                const faceSel = selectPrimaryFace(rawFaces);
                const primaryFaceLm = faceSel.primaryFace ?? rawFaces[0] ?? null;
                const effFaceCount =
                  rawFaces.length > 1 ? Math.max(1, faceSel.effectiveFaceCount) : rawFaces.length;
                cachedOwnerFaceCount = effFaceCount;
                const { w: cw, h: ch } = canvasSizeRef.current;

                if (!primaryFaceLm || effFaceCount === 0) {
                  cachedOwnerMissingStreak += 1;
                  cachedOwnerFaceBox = null;
                  cachedOwnerExtraBoxes = [];
                } else {
                  cachedOwnerMissingStreak = 0;
                  cachedOwnerFaceBox = computeOwnerFaceBoxMetrics(
                    primaryFaceLm,
                    v.videoWidth,
                    v.videoHeight,
                    cw,
                    ch,
                  );
                  cachedOwnerExtraBoxes = rawFaces
                    .filter((lm) => lm !== primaryFaceLm)
                    .map((lm) => computeOwnerFaceBoxMetrics(lm, v.videoWidth, v.videoHeight, cw, ch))
                    .filter((b): b is OwnerFaceBoxMetrics => b !== null);
                }
              } catch {
                // keep loop resilient across transient face detection frames
              }
            }

            const personCountInFrame = Math.max(latestPoseCount, cachedOwnerFaceCount);
            if (personCountInFrame >= 2 && stepACheckIdentity) {
              if (multiPersonStartAtMs === 0) {
                multiPersonStartAtMs = now;
              }
              if (now - multiPersonStartAtMs >= 1500) {
                cachedOwnerVerified = false;
                if (lastOwnerRehabKey !== 'MULTI_PERSON') {
                  lastOwnerRehabKey = 'MULTI_PERSON';
                  setOwnerRehabGuardAlert({
                    status: 'MULTI_PERSON',
                    text: 'พบมากกว่า 1 คนในเฟรม กรุณาให้เจ้าของบัญชีฝึกเพียงลำพัง',
                  });
                }
              }
            } else {
              multiPersonStartAtMs = 0;
              cachedOwnerVerified = true;
              if (lastOwnerRehabKey !== '') {
                lastOwnerRehabKey = '';
                setOwnerRehabGuardAlert(null);
              }
            }

            const multiPersonSustained =
              personCountInFrame >= 2 &&
              multiPersonStartAtMs > 0 &&
              now - multiPersonStartAtMs >= 1500;
            if (multiPersonSustained && stepACheckIdentity) {
              singlePersonRecoveryFrames = 0;
              rawKeypointsRef.current = null;
              smoothedRenderKeypointsRef.current = null;
              allRawKeypointsRef.current = allLandmarks;
              landmarkSmoother.reset();
              pose2dOneEuroFilter.reset();
              worldOneEuroFilter.reset();
              handOneEuroFilter.reset();
              latestHandLandmarksBySide = {};
              latestOverlayHint = null;
              latestSideViolation = 'NONE';
              latestWrongSideRaised = false;
              latestWrongSideBannerTh = null;
            } else if (demoPresetRef.current !== 'IMPOSTOR' || !stepACheckIdentity) {
              const rawLandmarks = allLandmarks[0] ?? null;
              const upperBodyVisible = Boolean(
                rawLandmarks && isUpperBodyVisibleInFrame(rawLandmarks),
              );
              if (rawLandmarks && upperBodyVisible) {
                if (lastEmittedFramingHint !== null) {
                  lastEmittedFramingHint = null;
                  setLiveFramingHintTh(null);
                }
                const stabilizedResult = sanitizeAndStabilizePoseLandmarks(
                  rawLandmarks,
                  torsoAnchor,
                  curEx.target_joint,
                );
                torsoAnchor = stabilizedResult.anchor;
                if (stabilizedResult.occludedFallback) {
                  occludedFrames += 1;
                  if (occludedFrames > 12) {
                    torsoAnchor = null;
                    occludedFrames = 0;
                  }
                } else {
                  occludedFrames = 0;
                }
                const { landmarks: kinematicSafeLandmarks } = applyKinematicSkeletonLock({
                  prevSmoothed: rawKeypointsRef.current,
                  rawLandmarks: stabilizedResult.landmarks,
                  videoW: v.videoWidth || 640,
                  videoH: v.videoHeight || 480,
                  canvasW: canvasSizeRef.current.w,
                  canvasH: canvasSizeRef.current.h,
                  isMirrored: true,
                });
                const applyOneEuro = (
                  flt: ClinicalOneEuroLandmarkFilter,
                  src: readonly NormalizedLandmark[],
                ): NormalizedLandmark[] =>
                  flt.filterLandmarks(src, now).map((lm, idx) => ({
                    x: lm.x,
                    y: lm.y,
                    z: lm.z ?? src[idx]?.z ?? 0,
                    // Preserve strict kinematic invalidation (`visibility === 0` when OUT_OF_FRAME / bone violation)
                    visibility: (src[idx]?.visibility ?? 0) < 0.05 ? 0 : (lm.visibility ?? src[idx]?.visibility ?? 0),
                  }));
                const landmarks = !enableCustomSmooth
                  ? kinematicSafeLandmarks
                  : smoothingMode === 'one_euro'
                    ? applyOneEuro(pose2dOneEuroFilter, kinematicSafeLandmarks)
                    : landmarkSmoother.apply(kinematicSafeLandmarks).map((lm, idx) => ({
                        ...lm,
                        visibility: (kinematicSafeLandmarks[idx]?.visibility ?? 0) < 0.05 ? 0 : lm.visibility,
                      }));
                rawKeypointsRef.current = landmarks;
                allRawKeypointsRef.current = [landmarks, ...allLandmarks.slice(1)];
                lastValidPoseAt = now;

                const lShPt = landmarks[POSE_INDEX.LEFT_SHOULDER];
                const rShPt = landmarks[POSE_INDEX.RIGHT_SHOULDER];
                if (lShPt && rShPt) {
                  const spanX = Math.abs(lShPt.x - rShPt.x);
                  const estDistM = Number(
                    Math.max(0.6, Math.min(3.5, 0.42 / Math.max(0.1, spanX))).toFixed(1),
                  );
                  if (
                    estDistM !== lastEmittedDistanceM &&
                    (lastEmittedDistanceM <= 0 || now - lastDistEmitAt >= 150)
                  ) {
                    lastEmittedDistanceM = estDistM;
                    lastDistEmitAt = now;
                    setDemoDistanceM(estDistM);
                  }
                }

                if (demoPresetRef.current === 'MULTI_PERSON') {
                  singlePersonRecoveryFrames += 1;
                  if (singlePersonRecoveryFrames >= 3) {
                    singlePersonRecoveryFrames = 0;
                    emitPresetState('ACTIVE_REP');
                  }
                }
                boneBaselineCalibrator.update(landmarks);
                const rawWorldLandmarks = poseRes?.worldLandmarks?.[0];
                const worldLandmarks =
                  rawWorldLandmarks && enableCustomSmooth
                    ? applyOneEuro(worldOneEuroFilter, rawWorldLandmarks)
                    : rawWorldLandmarks;

                const loopJointMode = classifyExerciseJointMode(curEx);
                const needsHandLandmarker =
                  loopJointMode === 'wrist_finger' ||
                  loopJointMode === 'forearm_turn' ||
                  loopJointMode === 'thumb_pinch';
                if (handLandmarker && needsHandLandmarker) {
                  try {
                    const hTs = Math.max(lastHandTs + 1, lastDetectTs + 1, Math.round(now));
                    lastHandTs = hTs;
                    const handRes = handLandmarker.detectForVideo(poseSourceForFrame, hTs);
                    const detectedHands = handRes.landmarks ?? [];
                    const detectedHandedness = handRes.handedness ?? [];
                    const nextHands: { left?: NormalizedLandmark[]; right?: NormalizedLandmark[] } = {};
                    const poseLWrist = landmarks[POSE_INDEX.LEFT_WRIST];
                    const poseRWrist = landmarks[POSE_INDEX.RIGHT_WRIST];

                    detectedHands.forEach((rawHand, hi) => {
                      if (!rawHand || rawHand.length < 21) return;
                      const hWrist = rawHand[0];
                      let side: 'left' | 'right' = hi === 0 ? 'right' : 'left';
                      if (poseLWrist && poseRWrist && hWrist) {
                        const dLeft = Math.hypot(hWrist.x - poseLWrist.x, hWrist.y - poseLWrist.y);
                        const dRight = Math.hypot(hWrist.x - poseRWrist.x, hWrist.y - poseRWrist.y);
                        side = dLeft <= dRight ? 'left' : 'right';
                      } else {
                        const cat = detectedHandedness[hi]?.[0]?.categoryName?.toLowerCase();
                        if (cat === 'left' || cat === 'right') {
                          side = cat;
                        }
                      }
                      const smoothedHand = enableCustomSmooth
                        ? handOneEuroFilter.filterHand(side, rawHand, now).map((lm, li) => ({
                          x: lm.x,
                          y: lm.y,
                          z: lm.z ?? rawHand[li]?.z ?? 0,
                          visibility: rawHand[li]?.visibility ?? 1,
                        }))
                        : rawHand;
                      nextHands[side] = smoothedHand;
                    });
                    latestHandLandmarksBySide = nextHands;
                  } catch {
                    // keep loop resilient if a hand detection frame fails
                  }
                } else {
                  latestHandLandmarksBySide = {};
                }

                if (now - lastEdgeBridgeSentAt >= 65) {
                  lastEdgeBridgeSentAt = now;
                  sendClientLandmarksRef.current(landmarks, worldLandmarks, latestPoseCount);
                }
                const measured = measureLiveJointAngleAndTilt(
                  landmarks,
                  curEx.target_joint,
                  curExpectedPlane,
                  worldLandmarks,
                  curEx,
                  {
                    handLandmarksBySide: latestHandLandmarksBySide,
                    baseline: boneBaselineCalibrator.getBaseline() ?? undefined,
                    targetRomDeg: curEx.target_rom_deg,
                  },
                );
                latestKinematicsRef.current = measured;
                latestOverlayHint = measured.overlayHint ?? null;
                latestRawTiltDeg = measured.trunkTiltDeg;
                wrongSideStreak = measured.wrongSideRaised ? wrongSideStreak + 1 : 0;
                wrongPlaneStreak = measured.wrongPlaneRaised ? wrongPlaneStreak + 1 : 0;
                const activeExSideInFrame = resolveExerciseSideFromMeta(
                  curEx,
                  curEx.target_joint,
                  curExpectedPlane,
                );
                const sideDebounceFrames =
                  activeExSideInFrame === 'BOTH' ||
                  measured.sideViolation === 'SINGLE_ARM_IN_BOTH_MODE'
                    ? 6
                    : 1;
                const sideViolationConfirmed =
                  Boolean(measured.wrongSideRaised) && wrongSideStreak >= sideDebounceFrames;
                latestSideViolation = sideViolationConfirmed
                  ? (measured.sideViolation ?? 'NONE')
                  : 'NONE';
                latestWrongSideRaised = sideViolationConfirmed;
                latestWrongSideBannerTh = sideViolationConfirmed
                  ? measured.wrongSideBannerTh
                  : null;
                latestWrongPlaneRaised = wrongPlaneStreak >= 2;
                latestWrongPlaneBannerTh = measured.wrongPlaneBannerTh;
                latestLowVisibilityAlertTh = measured.lowVisibilityAlertTh ?? null;
                latestPostureCompensated = Boolean(measured.postureCompensated);
                latestPostureBannerTh = measured.postureBannerTh ?? null;
                latestRawPrimaryRomDeg = measured.rawPrimaryRomDeg ?? measured.romDeg;
                latestLeftArmRom = measured.leftArmRom ?? null;
                latestRightArmRom = measured.rightArmRom ?? null;

                // Stroke Safety Guards in Browser AI Mode
                const safetyDtSec = prevSafetySampleAt > 0 ? Math.max(0.03, (now - prevSafetySampleAt) / 1000) : 0.1;
                const headDropVel =
                  prevHeadY !== null && measured.headY !== null && measured.headY !== undefined
                    ? (measured.headY - prevHeadY) / safetyDtSec
                    : 0;
                const shDropVel =
                  prevShoulderMidY !== null && measured.shoulderMidY !== null && measured.shoulderMidY !== undefined
                    ? (measured.shoulderMidY - prevShoulderMidY) / safetyDtSec
                    : 0;
                const angVel =
                  prevVelocityRom !== null && latestRawPrimaryRomDeg !== null
                    ? Math.abs(latestRawPrimaryRomDeg - prevVelocityRom) / safetyDtSec
                    : 0;

                if (measured.headY !== null && measured.headY !== undefined) prevHeadY = measured.headY;
                if (measured.shoulderMidY !== null && measured.shoulderMidY !== undefined) prevShoulderMidY = measured.shoulderMidY;
                if (latestRawPrimaryRomDeg !== null) prevVelocityRom = latestRawPrimaryRomDeg;
                prevSafetySampleAt = now;

                if (measured.trunkTiltDeg > 28) {
                  sustainedSwayFrames += 1;
                } else {
                  sustainedSwayFrames = 0;
                }
                if (sustainedSwayFrames >= 4 || headDropVel > 0.72 || shDropVel > 0.72) {
                  triggerBalanceEmergencyRef.current('POSTURAL_SWAY_AND_FALL_GUARD: ตรวจพบการเสียสมดุล กรุณานั่งพักและจับราวจับ');
                  return;
                }

                const maxSafeDeg = maxSafeRomRef.current;
                const overSafeLimit =
                  maxSafeDeg !== null &&
                  !curEx.target_joint.includes('knee') &&
                  latestRawPrimaryRomDeg !== null &&
                  latestRawPrimaryRomDeg > maxSafeDeg + 8;
                if (angVel > 195 || overSafeLimit) {
                  if (now - lastSpasticityAt > 2500) {
                    lastSpasticityAt = now;
                    setStrokeSafetyAlert({
                      type: 'spasticity',
                      messageTh: 'ช้าลงนิดนึงครับ ไม่ต้องฝืนยกสูงเกินไปนะครับ',
                    });
                  }
                } else if (now - lastSpasticityAt > 3500) {
                  setStrokeSafetyAlert((cur) => (cur?.type === 'spasticity' ? null : cur));
                }
                // Passive trunk lean indicator only (> 18 deg): never reset hold timer or trigger fatigue rest interruption!
                consecutiveCompEvents = 0;
              } else {
                wrongSideStreak = 0;
                wrongPlaneStreak = 0;
                occludedFrames = 0;
                latestSideViolation = 'NONE';
                latestWrongSideRaised = false;
                latestWrongPlaneRaised = false;
                latestLowVisibilityAlertTh = null;
                latestPostureCompensated = false;
                latestPostureBannerTh = null;
                latestRawPrimaryRomDeg = null;
                latestLeftArmRom = null;
                latestRightArmRom = null;
                targetRomDeg = null;
                smoothedRom = 0;
                smoothedTilt = 0;
                rawKeypointsRef.current = null;
                smoothedRenderKeypointsRef.current = null;
                allRawKeypointsRef.current = [];
                torsoAnchor = null;
                landmarkSmoother.reset();
                pose2dOneEuroFilter.reset();
                worldOneEuroFilter.reset();
                if (lastEmittedRom !== 0) {
                  lastEmittedRom = 0;
                  lastRomEmitAt = now;
                  setDemoRomDeg(0);
                }
                if (lastEmittedTilt !== 0) {
                  lastEmittedTilt = 0;
                  lastTiltEmitAt = now;
                  setDemoTrunkTiltDeg(0);
                }
                if (lastEmittedDistanceM !== 0) {
                  lastEmittedDistanceM = 0;
                  lastDistEmitAt = now;
                  setDemoDistanceM(0);
                }
                const hintMsg = 'กรุณาถอยห่างให้เห็นช่วงไหล่และลำตัว';
                if (lastEmittedFramingHint !== hintMsg) {
                  lastEmittedFramingHint = hintMsg;
                  setLiveFramingHintTh(hintMsg);
                }
              }
            }
          } catch {
            // Ignore transient inference errors
          }
        };

        // Dedicated 60FPS Render + Lerp + Hold & Rep Engine loop (decoupled from AI inference interval)
        const render60FpsLoop = () => {
          if (!active || !videoRef.current || !poseLandmarker) return;
          const curEx = selectedExerciseRef.current;
          const curExpectedPlane = resolveExpectedPlaneForExercise(curEx);
          if (trackedExId !== curEx.exercise_id || trackedResetNonce !== manualResetNonceRef.current) {
            trackedResetNonce = manualResetNonceRef.current;
            resetExerciseTrackingState(curEx.exercise_id);
          }

          const v = videoRef.current;
          const now = performance.now();
          const wallDtMs = lastFrameAt > 0 ? Math.min(120, Math.max(1, now - lastFrameAt)) : 16.7;
          const dtSec = wallDtMs / 1000;
          lastFrameAt = now;
          lastFrameTimeRef.current = now;

          const activeRules = formRulesRef.current;
          const checkLimb = isFormRuleActive(activeRules, 'limb');
          const checkPlane = isFormRuleActive(activeRules, 'plane');
          const checkTrunk = isFormRuleActive(activeRules, 'trunk');
          const checkIdentity = isFormRuleActive(activeRules, 'identity');
          const holdTargetSec = Math.max(0.5, curEx.hold_sec);
          const { w: canvasW, h: canvasH } = canvasSizeRef.current;
          const multiPersonConfirmed =
            checkIdentity &&
            (latestPoseCount > 1 || cachedOwnerFaceCount > 1) &&
            multiPersonStartAtMs > 0 &&
            now - multiPersonStartAtMs >= 1500;

          if (multiPersonConfirmed) {
            emitPresetState('MULTI_PERSON');
            drawLivePoseOverlayOnCanvas(poseCanvasRef.current, allRawKeypointsRef.current, {
              targetJoint: curEx.target_joint,
              expectedPlane: curExpectedPlane,
              exerciseMeta: curEx,
              romDeg: targetRomDeg !== null ? smoothedRom : null,
              targetRomDeg: curEx.target_rom_deg,
              holdRatio: holdElapsedSecRef.current / holdTargetSec,
              trunkTiltDeg: 0,
              isMultiPerson: true,
              isImpostorOrEmergency: false,
              videoWidth: v.videoWidth,
              videoHeight: v.videoHeight,
              canvasWidth: canvasW,
              canvasHeight: canvasH,
              showSkeleton: showSkeletonOverlayRef.current,
              isPrivacyMode: isPrivacyModeRef.current,
            });
          } else if (demoPresetRef.current === 'IMPOSTOR' && checkIdentity) {
            drawLivePoseOverlayOnCanvas(poseCanvasRef.current, allRawKeypointsRef.current, {
              targetJoint: curEx.target_joint,
              expectedPlane: curExpectedPlane,
              exerciseMeta: curEx,
              romDeg: targetRomDeg !== null ? smoothedRom : null,
              targetRomDeg: curEx.target_rom_deg,
              holdRatio: 0,
              trunkTiltDeg: 0,
              isMultiPerson: false,
              isImpostorOrEmergency: true,
              videoWidth: v.videoWidth,
              videoHeight: v.videoHeight,
              canvasWidth: canvasW,
              canvasHeight: canvasH,
              showSkeleton: showSkeletonOverlayRef.current,
              isPrivacyMode: isPrivacyModeRef.current,
            });
          } else if (rawKeypointsRef.current) {
            // 60FPS Frame-to-Frame Linear Interpolation (alpha = 0.42) glued directly onto One-Euro landmarks
            smoothedRenderKeypointsRef.current = lerpPoseLandmarksAlpha(
              smoothedRenderKeypointsRef.current,
              rawKeypointsRef.current,
              0.42,
            );
            const renderLandmarksList: NormalizedLandmark[][] = [
              smoothedRenderKeypointsRef.current,
              ...allRawKeypointsRef.current.slice(1),
            ];

            const effRawTiltDeg = checkTrunk ? latestRawTiltDeg : 0;
            const effSideViolation: StrokeSideViolationCode = checkLimb ? latestSideViolation : 'NONE';
            const effWrongSideRaised = checkLimb ? latestWrongSideRaised : false;
            const effWrongSideBannerTh = checkLimb ? latestWrongSideBannerTh : null;
            const effWrongPlaneRaised = checkPlane ? latestWrongPlaneRaised : false;
            const effWrongPlaneBannerTh = checkPlane ? latestWrongPlaneBannerTh : null;
            // Trunk lean (> 18 deg) & posture are strictly passive visual indicators; NEVER reduce, pause, or reset holdElapsedSec!
            const effPostureCompensated = false;
            const effPostureBannerTh =
              checkTrunk && latestRawTiltDeg > 18
                ? 'ลำตัวเอียงมากกว่า 18° กรุณาค่อยๆ ปรับลำตัวให้ตรง'
                : null;
            targetRomDeg =
              effWrongPlaneRaised ||
              (effWrongSideRaised && effSideViolation !== 'SINGLE_ARM_IN_BOTH_MODE')
                ? 0
                : latestRawPrimaryRomDeg;

            const tiltDelta = effRawTiltDeg - smoothedTilt;
            if (!checkTrunk) {
              smoothedTilt = 0;
            } else if (Math.abs(tiltDelta) >= 0.4) {
              const tiltAlpha = 1 - Math.exp(-wallDtMs / 120);
              smoothedTilt = Math.max(0, smoothedTilt + tiltDelta * tiltAlpha);
            }
            latestTrunkTiltDeg = Number(smoothedTilt.toFixed(1));
            if (now - lastTiltEmitAt >= 90 && Math.abs(latestTrunkTiltDeg - lastEmittedTilt) >= 0.3) {
              lastEmittedTilt = latestTrunkTiltDeg;
              lastTiltEmitAt = now;
              setDemoTrunkTiltDeg(latestTrunkTiltDeg);
            }

            if (targetRomDeg !== null) {
              smoothedRom = stepSmoothRomAngle(smoothedRom, targetRomDeg, wallDtMs);
              const nextEmitRom = Number(smoothedRom.toFixed(1));
              const hitBoundary =
                (nextEmitRom <= 0.1 || Math.abs(nextEmitRom - curEx.target_rom_deg) <= 0.2) &&
                Math.abs(nextEmitRom - lastEmittedRom) >= 0.1;
              if (now - lastRomEmitAt >= 48 && (Math.abs(nextEmitRom - lastEmittedRom) >= 0.3 || hitBoundary)) {
                lastEmittedRom = nextEmitRom;
                lastRomEmitAt = now;
                setDemoRomDeg(nextEmitRom);
              }
            } else {
              smoothedRom = 0;
              if (lastEmittedRom !== 0) {
                lastEmittedRom = 0;
                lastRomEmitAt = now;
                setDemoRomDeg(0);
              }
            }

            const blockCounting =
              multiPersonConfirmed ||
              (demoPresetRef.current === 'MULTI_PERSON' && checkIdentity) ||
              countingLockedRef.current;

            const curLoopJointMode = classifyExerciseJointMode(curEx);
            const curExSide = resolveExerciseSideFromMeta(curEx, curEx.target_joint, curExpectedPlane);
            const isSmallRomEx =
              effectiveTargetRomRef.current <= 70 ||
              curLoopJointMode === 'neck' ||
              curLoopJointMode === 'neck_turn' ||
              curLoopJointMode === 'wrist_finger' ||
              curLoopJointMode === 'forearm_turn' ||
              curLoopJointMode === 'thumb_pinch';
            const activeLms = smoothedRenderKeypointsRef.current ?? rawKeypointsRef.current;
            const armWristLowered = evaluateActiveWristsLowered({
              landmarks: activeLms,
              exerciseSide: curExSide,
              targetJoint: curEx.target_joint,
              isSmallRomOrKnee: isSmallRomEx || curEx.target_joint.includes('knee'),
            });

            // Detect raised arm whose wrist temporarily clipped above/outside camera frame edge ("มืออยู่นอกกรอบกล้อง")
            const isSideWristClippedRaised = (side: 'left' | 'right'): boolean => {
              const sh = activeLms[side === 'left' ? POSE_INDEX.LEFT_SHOULDER : POSE_INDEX.RIGHT_SHOULDER];
              const el = activeLms[side === 'left' ? POSE_INDEX.LEFT_ELBOW : POSE_INDEX.RIGHT_ELBOW];
              const wr = activeLms[side === 'left' ? POSE_INDEX.LEFT_WRIST : POSE_INDEX.RIGHT_WRIST];
              if (!sh || (sh.visibility ?? 0) < 0.3) return false;
              const armElevated =
                Boolean(el && (el.visibility ?? 0) >= 0.3 && el.y <= sh.y + 0.14) ||
                (targetRomDeg ?? 0) >= effectiveTargetRomRef.current * 0.55;
              const wristOutOrHigh =
                !wr ||
                (wr.visibility ?? 0) < 0.45 ||
                wr.y <= 0.06 ||
                wr.x <= 0.04 ||
                wr.x >= 0.96;
              return armElevated && wristOutOrHigh;
            };
            const handsOutOfFrameRaised =
              !curEx.target_joint.includes('knee') &&
              curEx.target_joint !== 'spine' &&
              (curExSide === 'LEFT'
                ? isSideWristClippedRaised('left')
                : curExSide === 'RIGHT'
                  ? isSideWristClippedRaised('right')
                  : curExSide === 'BOTH'
                    ? isSideWristClippedRaised('left') || isSideWristClippedRaised('right')
                    : false);

            // Step Option A Rep Engine: Hold -> Immediately +1 Rep & Lock 100% -> Lower Arm -> Ready for Next Rep
            const repStep = stepStrokeHoldAndRepStateMachine({
              state: {
                phase: repPhaseRef.current,
                holdElapsedSec: holdElapsedSecRef.current,
                completedReps: completedRepsRef.current,
                cooldownStartedAtMs: cooldownStartedAtRef.current,
                repCelebrateUntilMs: repCelebrateUntilRef.current,
                loweredFrameStreak: loweredFrameStreakRef.current,
                readyBannerUntilMs: readyBannerUntilRef.current,
                waitLowerStartedAtMs: waitLowerStartedAtRef.current,
              },
              activeRomDeg: targetRomDeg !== null ? smoothedRom : null,
              leftArmRomDeg: latestLeftArmRom,
              rightArmRomDeg: latestRightArmRom,
              exerciseCode: curEx.exercise_id,
              exerciseSide: curExSide,
              targetRomDeg: effectiveTargetRomRef.current,
              targetHoldSec: holdTargetSec,
              targetReps: curEx.target_reps,
              dtSec,
              nowMs: now,
              isKneeExercise: curEx.target_joint.includes('knee'),
              isSmallRomExercise: isSmallRomEx,
              armWristLowered,
              handsOutOfFrameRaised,
              sideViolation: effSideViolation,
              wrongSideBannerTh: effWrongSideBannerTh,
              wrongPlaneRaised: effWrongPlaneRaised,
              wrongPlaneBannerTh: effWrongPlaneBannerTh,
              isCompensatedNow: effPostureCompensated,
              postureBannerTh: effPostureBannerTh,
              blockCounting,
            });

            repPhaseRef.current = repStep.nextState.phase;
            holdElapsedSecRef.current = repStep.nextState.holdElapsedSec;
            completedRepsRef.current = repStep.nextState.completedReps;
            loweredFrameStreakRef.current = repStep.nextState.loweredFrameStreak ?? 0;
            readyBannerUntilRef.current = repStep.nextState.readyBannerUntilMs ?? 0;
            waitLowerStartedAtRef.current = repStep.nextState.waitLowerStartedAtMs ?? 0;
            cooldownStartedAtRef.current = repStep.nextState.cooldownStartedAtMs;
            repCelebrateUntilRef.current = repStep.nextState.repCelebrateUntilMs;

            const isWaitingLower = repPhaseRef.current === 'PHASE_2_WAIT_LOWER_TO_SCORE';

            // Direct 60FPS DOM updates for jitter-free Hold Progress Bar & Elapsed Timer Text
            if (holdProgressBarRef.current) {
              holdProgressBarRef.current.style.width = `${repStep.holdProgressPct}%`;
            }
            if (holdTimerTextRef.current) {
              holdTimerTextRef.current.textContent = isWaitingLower
                ? `${holdTargetSec.toFixed(1)}s (ครบ!)`
                : `${repStep.nextState.holdElapsedSec.toFixed(1)}s`;
            }

            if (repStep.justCompletedRep) {
              lastEmittedHold = repStep.holdRemainingSec;
              lastHoldEmitAt = now;
              setDemoHoldRemainingSec(repStep.holdRemainingSec);
              setHoldElapsedUiSec(Number(repStep.nextState.holdElapsedSec.toFixed(1)));
              setIsWaitLowerArmUi(isWaitingLower);
              setDemoValidReps(repStep.nextState.completedReps);
              const scoredMsg =
                repStep.bannerTh ??
                formatRepScoredBannerTh(repStep.nextState.completedReps, curEx.target_reps);
              if (repSuccessToastRef.current !== scoredMsg) {
                repSuccessToastRef.current = scoredMsg;
                setRepSuccessToastTh(scoredMsg);
              }
              emitPresetState(
                repStep.nextState.completedReps >= curEx.target_reps ? 'SET_FINISHED' : 'REP_COMPLETED',
              );
            } else if (repStep.unlockedToReady) {
              lastEmittedHold = repStep.holdRemainingSec;
              lastHoldEmitAt = now;
              setDemoHoldRemainingSec(repStep.holdRemainingSec);
              setHoldElapsedUiSec(0);
              setIsWaitLowerArmUi(false);
              repSuccessToastRef.current = null;
              setRepSuccessToastTh(null);
              emitPresetState('ACTIVE_REP');
            } else if (repStep.holdCompleteWaitLower || isWaitingLower) {
              lastEmittedHold = 0;
              if (now - lastHoldEmitAt >= 90) {
                lastHoldEmitAt = now;
                setDemoHoldRemainingSec(0);
                setHoldElapsedUiSec(holdTargetSec);
                setIsWaitLowerArmUi(true);
              }
              const waitMsg =
                repStep.bannerTh ??
                formatRepScoredBannerTh(repStep.nextState.completedReps, curEx.target_reps);
              if (repSuccessToastRef.current !== waitMsg) {
                repSuccessToastRef.current = waitMsg;
                setRepSuccessToastTh(waitMsg);
              }
              emitPresetState('HOLD_PHASE');
            } else {
              if (repSuccessToastRef.current !== null) {
                repSuccessToastRef.current = null;
                setRepSuccessToastTh(null);
              }

              if (
                now - lastHoldEmitAt >= 80 &&
                (repStep.holdRemainingSec !== lastEmittedHold ||
                  Math.abs(repStep.nextState.holdElapsedSec - (holdTargetSec - lastEmittedHold)) >= 0.05)
              ) {
                lastEmittedHold = repStep.holdRemainingSec;
                lastHoldEmitAt = now;
                setDemoHoldRemainingSec(repStep.holdRemainingSec);
                setHoldElapsedUiSec(Number(repStep.nextState.holdElapsedSec.toFixed(1)));
                setIsWaitLowerArmUi(false);
              }

              if (repStep.statusKind === 'WRONG_LIMB') {
                emitPresetState('WRONG_LIMB');
              } else if (repStep.statusKind === 'WRONG_PLANE') {
                emitPresetState('WRONG_PLANE');
              } else if (repStep.statusKind === 'HOLD_PHASE') {
                emitPresetState('HOLD_PHASE');
              } else if (repStep.statusKind === 'SET_FINISHED') {
                emitPresetState('SET_FINISHED');
              } else {
                emitPresetState('ACTIVE_REP');
              }
            }

            drawLivePoseOverlayOnCanvas(poseCanvasRef.current, renderLandmarksList, {
              targetJoint: curEx.target_joint,
              expectedPlane: curExpectedPlane,
              exerciseMeta: curEx,
              romDeg: targetRomDeg !== null ? smoothedRom : null,
              targetRomDeg: curEx.target_rom_deg,
              holdRatio: repStep.holdProgressPct / 100,
              trunkTiltDeg: latestTrunkTiltDeg,
              isMultiPerson: (multiPersonConfirmed || demoPresetRef.current === 'MULTI_PERSON') && checkIdentity,
              isImpostorOrEmergency: countingLockedRef.current,
              sideViolation: effSideViolation,
              wrongSideRaised: effWrongSideRaised,
              wrongSideBannerTh: effWrongSideBannerTh,
              wrongPlaneRaised: effWrongPlaneRaised,
              wrongPlaneBannerTh: effWrongPlaneBannerTh,
              lowVisibilityAlertTh: latestLowVisibilityAlertTh,
              postureBannerTh: effPostureBannerTh,
              repSuccessBannerTh: repSuccessToastRef.current,
              overlayHint: latestOverlayHint,
              handLandmarks21: (() => {
                const activeSide = resolveExerciseSideFromMeta(curEx, curEx.target_joint, curExpectedPlane);
                if (activeSide === 'LEFT') return latestHandLandmarksBySide.left;
                if (activeSide === 'RIGHT') return latestHandLandmarksBySide.right;
                return latestHandLandmarksBySide.right ?? latestHandLandmarksBySide.left;
              })(),
              videoWidth: v.videoWidth,
              videoHeight: v.videoHeight,
              canvasWidth: canvasW,
              canvasHeight: canvasH,
              smoothingStatusText: formatSmoothingHud(),
              showSkeleton: showSkeletonOverlayRef.current,
              isPrivacyMode: isPrivacyModeRef.current,
              ownerLock: checkIdentity && (cachedOwnerFaceBox || cachedOwnerExtraBoxes.length > 0)
                ? {
                  ownerName: ownerBiometricsRef.current?.fullName ?? payload.patient?.name ?? 'ผู้ป่วย',
                  faceBox: cachedOwnerFaceBox,
                  extraFaceBoxes: cachedOwnerExtraBoxes,
                  ownerFaceVerified: !multiPersonConfirmed,
                  faceCount: multiPersonConfirmed ? Math.max(2, cachedOwnerFaceCount) : 1,
                }
                : undefined,
            });
          } else {
            smoothedRenderKeypointsRef.current = null;
            const holdPctKeep = Math.min(100, (holdElapsedSecRef.current / holdTargetSec) * 100);
            if (holdProgressBarRef.current) holdProgressBarRef.current.style.width = `${holdPctKeep}%`;
            if (holdTimerTextRef.current) {
              holdTimerTextRef.current.textContent =
                repPhaseRef.current === 'PHASE_2_WAIT_LOWER_TO_SCORE'
                  ? `${holdTargetSec.toFixed(1)}s (ครบ!)`
                  : `${holdElapsedSecRef.current.toFixed(1)}s`;
            }
            drawLivePoseOverlayOnCanvas(poseCanvasRef.current, [], {
              targetJoint: curEx.target_joint,
              expectedPlane: curExpectedPlane,
              exerciseMeta: curEx,
              romDeg: null,
              targetRomDeg: curEx.target_rom_deg,
              holdRatio: holdPctKeep / 100,
              trunkTiltDeg: 0,
              isMultiPerson: false,
              isImpostorOrEmergency: countingLockedRef.current,
              videoWidth: v.videoWidth,
              videoHeight: v.videoHeight,
              canvasWidth: canvasW,
              canvasHeight: canvasH,
              showSkeleton: showSkeletonOverlayRef.current,
              isPrivacyMode: isPrivacyModeRef.current,
              ownerLock: checkIdentity && (cachedOwnerFaceBox || cachedOwnerExtraBoxes.length > 0)
                ? {
                  ownerName: ownerBiometricsRef.current?.fullName ?? payload.patient?.name ?? 'ผู้ป่วย',
                  faceBox: cachedOwnerFaceBox,
                  extraFaceBoxes: cachedOwnerExtraBoxes,
                  ownerFaceVerified: !multiPersonConfirmed,
                  faceCount: multiPersonConfirmed ? Math.max(2, cachedOwnerFaceCount) : 1,
                }
                : undefined,
            });
          }

          if (active) {
            rafId = requestAnimationFrame(render60FpsLoop);
          }
        };

        runAiInference();
        inferenceTimerId = window.setInterval(runAiInference, 28);
        rafId = requestAnimationFrame(render60FpsLoop);
      } catch {
        if (active) {
          setCameraError('ไม่พบกล้องหรือไม่สามารถเปิดกล้องได้');
        }
      }
    }

    void initPoseCamera();
    return () => {
      active = false;
      cleanup();
    };
  }, [
    livePoseActive,
    autoCycleDemo,
    isEmergency,
    isEdgeDirectCamera,
    setDemoDistanceM,
    setDemoHoldRemainingSec,
    setDemoPresetState,
    setDemoRomDeg,
    setDemoTrunkTiltDeg,
    setDemoValidReps,
  ]);

  const patientRevealed = Boolean(
    patient?.name && (isPoseTestMode || kiosk_state !== 'AUTH_FRAMING' || demoFaceMatched),
  );

  const activePoseTestIndex = Math.max(
    0,
    exerciseOptions.findIndex((opt) => opt.exercise_id === selectedExercise.exercise_id),
  );
  const activePoseTestSide: ExerciseSideCode =
    resolveExerciseSideFromMeta(selectedExercise, selectedExercise.target_joint, activeExpectedPlane) ??
    'RIGHT';

  const resetPoseTestProgress = (nextHoldSec: number) => {
    manualResetNonceRef.current += 1;
    completedRepsRef.current = 0;
    holdElapsedSecRef.current = 0;
    repPhaseRef.current = 'PHASE_1_HOLDING';
    loweredFrameStreakRef.current = 0;
    readyBannerUntilRef.current = 0;
    waitLowerStartedAtRef.current = 0;
    repSuccessToastRef.current = null;
    setDemoValidReps(0);
    setDemoHoldRemainingSec(nextHoldSec);
    setHoldElapsedUiSec(0);
    setIsWaitLowerArmUi(false);
    setRepSuccessToastTh(null);
    if (holdProgressBarRef.current) {
      holdProgressBarRef.current.style.width = '0%';
    }
    if (holdTimerTextRef.current) {
      holdTimerTextRef.current.textContent = '0.0s';
    }
  };

  const handleSelectPoseTestExercise = (exerciseId: string) => {
    const nextRx = exerciseOptions.find((opt) => opt.exercise_id === exerciseId);
    if (!nextRx) return;
    setSelectedExercise(nextRx);
    resetPoseTestProgress(nextRx.hold_sec);
  };

  const handleStepPoseTestExercise = (delta: -1 | 1) => {
    if (exerciseOptions.length === 0) return;
    const nextIdx = (activePoseTestIndex + delta + exerciseOptions.length) % exerciseOptions.length;
    const nextRx = exerciseOptions[nextIdx];
    setSelectedExercise(nextRx);
    resetPoseTestProgress(nextRx.hold_sec);
  };

  const handleSwitchPoseTestSide = (nextSide: ExerciseSideCode) => {
    const baseKey =
      selectedExercise.base_code ?? selectedExercise.catalog_exercise_id ?? selectedExercise.exercise_id;
    const nextRx = applyExerciseSideOverride(selectedExercise, nextSide);
    setTestSideOverrides((prev) => ({ ...prev, [baseKey]: nextSide }));
    setSelectedExercise(nextRx);
    resetPoseTestProgress(nextRx.hold_sec);
  };

  const handleResetPoseTestReps = () => {
    resetPoseTestProgress(selectedExercise.hold_sec);
    resetDemoSession();
  };

  const checklistItems: {
    ruleKey: FormAssessmentRuleKey;
    label: string;
    ok: boolean;
    ruleActive: boolean;
    warnText: string;
  }[] = [
      ...((selectedExercise.target_joint === 'left_shoulder' ||
        selectedExercise.target_joint === 'right_shoulder' ||
        selectedExercise.prescribed_side !== undefined)
        ? (() => {
          const checkSide = resolveExerciseSideFromMeta(
            selectedExercise,
            selectedExercise.target_joint,
            activeExpectedPlane,
          );
          if (selectedExercise.target_joint === 'spine' && !checkSide) return [];
          return [
            {
              ruleKey: 'limb' as const,
              label:
                checkSide === 'BOTH'
                  ? 'ยกพร้อมกันทั้ง 2 ข้าง (ซ้าย + ขวา)'
                  : checkSide === 'LEFT'
                    ? 'ยกเฉพาะแขนซ้ายตามท่าที่กำหนด'
                    : 'ยกเฉพาะแขนขวาตามท่าที่กำหนด',
              ok: !wrongLimbAlert && !isEmergency,
              ruleActive: isFormRuleActive(formRules, 'limb'),
              warnText:
                checkSide === 'BOTH'
                  ? 'กรุณายกพร้อมกันทั้ง 2 ข้าง'
                  : checkSide === 'LEFT'
                    ? 'กรุณายกแขนซ้าย'
                    : 'กรุณายกแขนขวา',
            },
            ...(activeExpectedPlane === 'flexion' || activeExpectedPlane === 'abduction'
              ? [
                {
                  ruleKey: 'plane' as const,
                  label:
                    activeExpectedPlane === 'flexion'
                      ? 'ทิศทางถูกต้อง (ยกแขนไปด้านหน้าเท่านั้น)'
                      : 'ทิศทางถูกต้อง (กางแขนออกด้านข้างเท่านั้น)',
                  ok: !wrongPlaneAlert && !isEmergency,
                  ruleActive: isFormRuleActive(formRules, 'plane'),
                  warnText:
                    activeExpectedPlane === 'flexion' ? 'ห้ามกางด้านข้าง' : 'ห้ามยกด้านหน้า',
                },
              ]
              : []),
          ];
        })()
        : []),
      {
        ruleKey: 'trunk',
        label: 'ลำตัวตั้งตรง ไม่เอียงชดเชย',
        ok: !trunkLeanAlert && !isEmergency,
        ruleActive: isFormRuleActive(formRules, 'trunk'),
        warnText: 'เอียงลำตัว',
      },
      {
        ruleKey: 'posture',
        label: 'ผ่อนหัวไหล่ ไม่ยักไหล่ขึ้น',
        ok: !shoulderHikeAlert && !isEmergency,
        ruleActive: isFormRuleActive(formRules, 'posture'),
        warnText: 'ยักไหล่',
      },
      {
        ruleKey: 'velocity',
        label: 'ความเร็วเคลื่อนไหวสม่ำเสมอ',
        ok: !tooFastAlert && !isEmergency,
        ruleActive: isFormRuleActive(formRules, 'velocity'),
        warnText: 'เร็วเกินไป',
      },
      {
        ruleKey: 'identity',
        label: 'ยืนยันตัวตนคนไข้ตัวจริง (1 คนในเฟรม)',
        ok:
          !isFormRuleActive(formRules, 'identity') ||
          (kiosk_state !== 'PAUSED_MULTI_PERSON' &&
            kiosk_state !== 'PAUSED_IMPOSTOR' &&
            payload.identity_guard?.is_counting_allowed !== false &&
            !isEmergency),
        ruleActive: isFormRuleActive(formRules, 'identity'),
        warnText:
          kiosk_state === 'PAUSED_IMPOSTOR' || payload.identity_guard?.status === 'IMPOSTOR'
            ? 'ไม่ใช่คนไข้'
            : 'พบหลายคน',
      },
    ];
  const hasDisabledFormRule =
    !formRules.enabled || checklistItems.some((item) => !item.ruleActive);

  const isOnlineReady = !isDemoMode && (connectionStatus === 'connected' || livePoseActive);
  const displayedDistanceM =
    !isDemoMode && isEdgeDirectCamera
      ? (payload.calibration?.estimated_distance_m ?? 0)
      : demoDistanceM;
  const distanceLabel =
    displayedDistanceM > 0
      ? `${displayedDistanceM.toFixed(1)} ม.${displayedDistanceM >= 1.4 && displayedDistanceM <= 2.6 ? ' (เหมาะสม)' : ''}`
      : 'รอจับระยะ...';
  const currentUiSide =
    resolveExerciseSideFromMeta(
      selectedExercise,
      selectedExercise.target_joint,
      activeExpectedPlane,
    ) ?? activePoseTestSide;
  const targetSideLabel =
    currentUiSide === 'BOTH'
      ? 'ทั้งสองข้าง (Both)'
      : currentUiSide === 'LEFT'
        ? 'ซ้าย (Left)'
        : 'ขวา (Right)';
  const activeRom = Math.max(0, telemetry_analysis.current_rom_deg);
  const holdTargetSec = Math.max(0.5, selectedExercise.hold_sec);
  const liveHoldElapsed = isWaitLowerArmUi
    ? holdTargetSec
    : holdElapsedUiSec > 0
      ? holdElapsedUiSec
      : Math.max(0, holdTargetSec - rep_counter.hold_remaining_sec);
  const holdPct =
    isWaitLowerArmUi || kiosk_state === 'REP_COMPLETED' || kiosk_state === 'SET_FINISHED'
      ? 100
      : Math.max(0, Math.min(100, (liveHoldElapsed / holdTargetSec) * 100));
  const romPct =
    effectiveTargetRomDeg > 0
      ? Math.min(100, Math.max(0, Math.round((activeRom / effectiveTargetRomDeg) * 100)))
      : Math.min(100, Math.max(0, Math.round(telemetry_analysis.rom_completion_pct)));
  const postureSpec = resolveExercisePostureSpec(selectedExercise, 'th');
  const coreChecklistItems = [
    {
      ruleKey: 'trunk' as const,
      displayLabel: 'ลำตัวตั้งตรง ไม่เอียง',
      baseItem: checklistItems.find((item) => item.ruleKey === 'trunk'),
    },
    {
      ruleKey: 'posture' as const,
      displayLabel: 'ผ่อนหัวไหล่ ไม่ยกเกร็ง',
      baseItem: checklistItems.find((item) => item.ruleKey === 'posture'),
    },
    {
      ruleKey: 'identity' as const,
      displayLabel: 'ยืนยันตัวตนคนไข้ (1 คนในเฟรม)',
      baseItem: checklistItems.find((item) => item.ruleKey === 'identity'),
    },
  ].filter(
    (entry): entry is {
      ruleKey: 'trunk' | 'posture' | 'identity';
      displayLabel: string;
      baseItem: (typeof checklistItems)[number];
    } => Boolean(entry.baseItem),
  );
  const rawVoiceTextTh =
    strokeSafetyAlert?.messageTh ?? telemetry_analysis.safety_guard_banner_th ?? tts_voice_th;
  const cleanSpeechText = sanitizeSpeechText(rawVoiceTextTh);
  const displayedVoiceSubtitle = sanitizeSpeechText(translateVoiceCoachText(cleanSpeechText, zone5VoiceLang));
  const isMascotSpeaking = !voiceMuted && voiceCoachSpeaking;
  const isMascotExplaining =
    coachModalOpen ||
    kiosk_state === 'ACTIVE_REP' ||
    kiosk_state === 'HOLD_PHASE' ||
    kiosk_state === 'CALIBRATING' ||
    Boolean(unifiedCoachingBanner?.titleTh);
  const baseMascotPose = resolveMascotPose({
    isGreeting: mascotGreetingActive,
    isEncouraging: mascotEncouragingActive,
    isSpeaking: isMascotSpeaking,
    isExplaining: isMascotExplaining,
  });
  const mascotPose =
    !mascotEncouragingActive && isMascotSpeaking
      ? inferMascotPoseFromSpeechText(cleanSpeechText)
      : baseMascotPose;

  return (
    <div className="h-[100dvh] max-h-[100dvh] w-full overflow-hidden bg-gradient-to-b from-slate-50 via-white to-emerald-50/40 text-slate-800 flex flex-col justify-between p-2.5 md:p-3 gap-2 select-none">
      {/* ZONE 1: Compact 2-Row Clinical White-Blue-Green Header & Controls */}
      <header
        data-testid={isPoseTestMode ? 'pose-test-switcher-bar' : undefined}
        className="shrink-0 bg-white border-2 border-sky-100 rounded-2xl px-3 py-2 shadow-sm flex flex-col gap-1.5"
      >
        {/* Row 1: Brand, Status Pill, Quick Reset, Return to Admin Portal */}
        <div className="flex items-center justify-between gap-2 flex-nowrap">
          <div className="flex items-center gap-2 min-w-0">
            <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-gradient-to-br from-teal-500 to-emerald-600 text-white shadow-xs">
              <Shield className="h-4 w-4" />
            </div>
            <div className="flex items-baseline gap-2 min-w-0 truncate">
              <h1 className="font-extrabold text-blue-950 text-base md:text-lg truncate">
                Dr.Tech.Care • {isPoseTestMode ? 'โหมดทดสอบท่าทาง' : 'Smart Rehab Kiosk'}
              </h1>
              {patientRevealed && patient && (
                <span className="hidden sm:inline text-xs md:text-sm font-bold text-sky-700 truncate">
                  {patient.name} ({patient.id})
                </span>
              )}
            </div>
          </div>

          <div className="flex items-center gap-2 shrink-0">
            <div
              className={`inline-flex items-center gap-2 rounded-full border px-3 py-1 font-bold text-xs md:text-sm whitespace-nowrap ${
                isOnlineReady
                  ? 'bg-emerald-50 border-emerald-400 text-emerald-800'
                  : 'bg-amber-50 border-amber-400 text-amber-800'
              }`}
            >
              <span
                className={`inline-block w-2.5 h-2.5 rounded-full shrink-0 ${
                  isOnlineReady ? 'bg-emerald-500 animate-status-halo' : 'bg-amber-500'
                }`}
              />
              <span>
                {isOnlineReady
                  ? 'ออนไลน์ (ระบบกล้องพร้อม)'
                  : isDemoMode
                    ? 'โหมดจำลอง'
                    : 'พักกล้องชั่วคราว'}
              </span>
            </div>

            <button
              type="button"
              onClick={isPoseTestMode ? handleResetPoseTestReps : resetDemoSession}
              className="inline-flex items-center gap-1.5 bg-sky-50 hover:bg-sky-100 border border-sky-300 text-blue-800 font-bold text-xs md:text-sm px-3 py-1 rounded-xl cursor-pointer active:scale-95 transition whitespace-nowrap"
            >
              <RotateCcw className="w-3.5 h-3.5 shrink-0" />
              <span>รีเซ็ต (0/{selectedExercise.target_reps})</span>
            </button>

            {onBackToAdminPortal || onOpenStaffPortal ? (
              <button
                type="button"
                onClick={onBackToAdminPortal ?? onOpenStaffPortal}
                className="inline-flex items-center gap-1.5 bg-blue-600 hover:bg-blue-700 text-white font-extrabold text-xs md:text-sm px-3.5 py-1.5 rounded-xl shadow-sm cursor-pointer active:scale-95 transition whitespace-nowrap"
              >
                <ArrowLeft className="w-4 h-4 shrink-0" />
                <span>กลับหน้า Admin Portal</span>
              </button>
            ) : (
              <button
                type="button"
                onClick={() => handleEmergencyStopClick()}
                className="bg-rose-600 hover:bg-rose-700 text-white font-extrabold text-xs md:text-sm px-3.5 py-1.5 rounded-xl shadow-sm cursor-pointer active:scale-95 transition whitespace-nowrap"
              >
                หยุดฉุกเฉิน
              </button>
            )}
          </div>
        </div>

        {/* Row 2: Exercise Dropdown (1-15) + Prev/Next Buttons + Left/Right/Both Side Pills */}
        <div className="flex items-center justify-between gap-2 flex-wrap md:flex-nowrap">
          <label htmlFor="pose-test-exercise-select" className="sr-only">
            เลือกท่าที่จะทดลอง 1–{exerciseOptions.length}
          </label>
          <select
            id="pose-test-exercise-select"
            aria-label={`เลือกท่าที่จะทดลอง 1–${exerciseOptions.length}`}
            value={selectedExercise.exercise_id}
            onChange={(e) => {
              if (isPoseTestMode) {
                handleSelectPoseTestExercise(e.target.value);
              } else {
                const nextRx = exerciseOptions.find((opt) => opt.exercise_id === e.target.value);
                if (nextRx) setSelectedExercise(nextRx);
              }
            }}
            className="flex-1 min-w-[200px] bg-slate-50 border-2 border-blue-200 text-blue-950 font-extrabold text-sm md:text-base rounded-xl px-3 py-1.5 focus:border-blue-500 focus:outline-none cursor-pointer"
          >
            {exerciseOptions.map((opt, idx) => (
              <option key={opt.exercise_id} value={opt.exercise_id}>
                {completedIds.includes(opt.exercise_id) && !isPoseTestMode ? '[ครบ]' : `${idx + 1}.`} {opt.name_th}
              </option>
            ))}
          </select>

          <button
            type="button"
            onClick={() => {
              if (isPoseTestMode) {
                handleStepPoseTestExercise(-1);
              } else if (exerciseOptions.length > 0) {
                const prevIdx = (currentExerciseIndex - 1 + exerciseOptions.length) % exerciseOptions.length;
                setSelectedExercise(exerciseOptions[prevIdx]);
              }
            }}
            className="inline-flex items-center gap-1.5 bg-blue-50 hover:bg-blue-100 border border-blue-300 text-blue-800 font-extrabold text-xs md:text-sm px-3 py-1.5 rounded-xl shrink-0 cursor-pointer active:scale-95 transition whitespace-nowrap"
          >
            <SkipBack className="w-3.5 h-3.5 shrink-0" />
            <span>ก่อนหน้า</span>
          </button>

          <button
            type="button"
            onClick={() => {
              if (isPoseTestMode) {
                handleStepPoseTestExercise(1);
              } else if (exerciseOptions.length > 0) {
                const nextIdx = (currentExerciseIndex + 1) % exerciseOptions.length;
                setSelectedExercise(exerciseOptions[nextIdx]);
              }
            }}
            className="inline-flex items-center gap-1.5 bg-blue-50 hover:bg-blue-100 border border-blue-300 text-blue-800 font-extrabold text-xs md:text-sm px-3 py-1.5 rounded-xl shrink-0 cursor-pointer active:scale-95 transition whitespace-nowrap"
          >
            <span>ถัดไป</span>
            <SkipForward className="w-3.5 h-3.5 shrink-0" />
          </button>

          {isPoseTestMode ? (
            <div
              className="flex items-center gap-1.5 shrink-0"
              role="group"
              aria-label="เลือกข้างที่ต้องการทดสอบ"
            >
              {(
                [
                  { code: 'LEFT' as const, label: 'ซ้าย (L)' },
                  { code: 'RIGHT' as const, label: 'ขวา (R)' },
                  { code: 'BOTH' as const, label: 'ทั้งสองข้าง (Both)' },
                ] as const
              ).map((pill) => {
                const isActive = activePoseTestSide === pill.code;
                return (
                  <button
                    key={pill.code}
                    type="button"
                    aria-pressed={isActive}
                    onClick={() => handleSwitchPoseTestSide(pill.code)}
                    className={`cursor-pointer active:scale-95 transition whitespace-nowrap rounded-xl ${
                      isActive
                        ? 'bg-emerald-600 border-2 border-emerald-700 text-white font-extrabold text-xs md:text-sm px-3.5 py-1.5 shadow-sm'
                        : 'bg-white hover:bg-slate-50 border border-slate-300 text-slate-700 font-bold text-xs md:text-sm px-3 py-1.5'
                    }`}
                  >
                    {pill.label}
                  </button>
                );
              })}
            </div>
          ) : (
            <div className="flex items-center gap-1.5 shrink-0">
              <span className="bg-emerald-600 border-2 border-emerald-700 text-white font-extrabold text-xs md:text-sm px-3.5 py-1.5 rounded-xl shadow-sm whitespace-nowrap">
                {targetSideLabel}
              </span>
              {nextUncompletedExercise && (
                <button
                  type="button"
                  onClick={() => {
                    markCurrentExerciseCompleted();
                    setSelectedExercise(nextUncompletedExercise);
                  }}
                  className="inline-flex items-center gap-1 bg-blue-600 hover:bg-blue-700 text-white font-extrabold text-xs md:text-sm px-3 py-1.5 rounded-xl shadow-sm cursor-pointer whitespace-nowrap"
                >
                  <span>ไปท่าถัดไป</span>
                  <ArrowRight className="w-3.5 h-3.5 shrink-0" />
                </button>
              )}
              {!nextUncompletedExercise && onAllExercisesCompleted && (
                <button
                  type="button"
                  onClick={() => {
                    markCurrentExerciseCompleted();
                    onAllExercisesCompleted();
                  }}
                  className="inline-flex items-center gap-1.5 bg-emerald-600 hover:bg-emerald-700 text-white font-extrabold text-xs md:text-sm px-3 py-1.5 rounded-xl shadow-sm cursor-pointer whitespace-nowrap"
                >
                  <Gamepad2 className="w-3.5 h-3.5 shrink-0" />
                  <span>เข้าสู่มินิเกม</span>
                  <ArrowRight className="w-3.5 h-3.5 shrink-0" />
                </button>
              )}
            </div>
          )}
        </div>
      </header>

      {/* Camera Hardware Alert Banner (only shown if local camera access fails) */}
      {!isDemoMode && !isEdgeDirectCamera && cameraError && (
        <div
          role="alert"
          className="shrink-0 flex items-center justify-between gap-2 rounded-xl border-2 border-rose-300 bg-rose-50 px-3 py-1.5 text-rose-900 shadow-xs animate-toast-in animate-gentle-shake"
        >
          <div className="flex min-w-0 items-center gap-2">
            <AlertTriangle className="h-4 w-4 shrink-0 text-rose-600" />
            <p className="text-xs md:text-sm font-extrabold truncate">
              ไม่สามารถเปิดกล้องเว็บแคมได้: {cameraError}
            </p>
          </div>
          <button
            type="button"
            onClick={() => {
              setCameraError(null);
              setLivePoseActive(false);
              window.setTimeout(() => setLivePoseActive(true), 120);
            }}
            className="inline-flex items-center gap-1 rounded-lg bg-rose-600 px-3 py-1 text-xs font-extrabold text-white hover:bg-rose-500 shrink-0 cursor-pointer"
          >
            <RotateCcw className="h-3.5 w-3.5" />
            <span>ลองเปิดกล้องใหม่</span>
          </button>
        </div>
      )}

      {allExercisesCompleted && onAllExercisesCompleted && (
        <div className="shrink-0 flex items-center justify-between gap-2 rounded-xl border-2 border-emerald-400 bg-emerald-50 px-3 py-1.5 shadow-xs animate-toast-in">
          <div className="flex items-center gap-2 min-w-0">
            <CheckCircle2 className="h-4 w-4 shrink-0 text-emerald-600" />
            <p className="text-xs md:text-sm font-extrabold text-emerald-950 truncate">
              ทำท่ากายภาพครบทั้งหมด {exerciseOptions.length} ท่าแล้ว!
            </p>
          </div>
          <button
            type="button"
            onClick={onAllExercisesCompleted}
            className="inline-flex items-center gap-1.5 rounded-lg bg-emerald-600 px-3 py-1 text-xs font-extrabold text-white shadow-xs hover:bg-emerald-500 shrink-0 cursor-pointer"
          >
            <Gamepad2 className="h-3.5 w-3.5" />
            <span>เข้าสู่มินิเกม</span>
            <ArrowRight className="h-3.5 w-3.5 shrink-0" />
          </button>
        </div>
      )}

      {/* ZONE 2: Giant Flexible Camera Viewport (Takes 100% of remaining 100dvh vertical space) */}
      <section
        aria-label="หน้าจอกล้องวิเคราะห์ท่าทางสด"
        className="flex-1 min-h-0 relative w-full rounded-2xl overflow-hidden border-2 border-emerald-400 bg-slate-900 shadow-md [&>div]:!aspect-auto [&>div]:!h-full [&>div]:!w-full [&>div]:!min-h-0 [&>div]:!rounded-none [&>div]:!border-0 [&>div]:!bg-transparent [&>div]:!shadow-none [&>div>div:nth-child(2)]:hidden"
      >
        <SkeletonCanvas
          payload={payload}
          exercise={selectedExercise}
          isDemoMode={isDemoMode}
          distanceM={
            !isDemoMode && isEdgeDirectCamera
              ? (payload.calibration?.estimated_distance_m ?? 0)
              : demoDistanceM
          }
          trunkTiltDeg={
            !isDemoMode && isEdgeDirectCamera
              ? (payload.telemetry_analysis?.trunk_tilt_deg ?? 0)
              : demoTrunkTiltDeg
          }
          hasLiveCamera={!isEdgeDirectCamera && livePoseActive && !cameraError}
          isPrivacyMode={isPrivacyMode}
          showSkeletonOverlay={showSkeletonOverlay}
          framingHintTh={liveFramingHintTh}
          liveVideoOverlay={
            !isEdgeDirectCamera && livePoseActive && !cameraError ? (
              <div
                ref={videoContainerRef}
                className={`relative h-full w-full overflow-hidden ${isPrivacyMode ? 'bg-[#050811]' : 'bg-emerald-950'
                  }`}
              >
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className={`w-full h-full object-cover scale-x-[-1] transition-opacity duration-200 ${isPrivacyMode ? 'opacity-0' : 'opacity-100'
                    }`}
                />
                <canvas
                  ref={poseCanvasRef}
                  aria-label="Live Pose Skeleton & Angle Overlay"
                  className="pointer-events-none absolute inset-0 w-full h-full object-cover"
                />
              </div>
            ) : undefined
          }
        >
          {kiosk_state === 'AUTH_FRAMING' && (
            <FaceScanOverlay
              verified={demoFaceMatched || Boolean(patient?.id)}
              streakFrames={demoStreakFrames}
              requiredFrames={5}
              distanceM={demoDistanceM}
              patientName={patient?.name}
            />
          )}

          {isEmergency && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="absolute inset-0 z-30 flex flex-col items-center justify-center bg-rose-950/85 p-4 text-center text-white backdrop-blur-sm"
            >
              <AlertOctagon className="h-14 w-14 text-rose-400" />
              <h2 className="mt-2 text-2xl font-extrabold">หยุดพักฉุกเฉิน (EMERGENCY HALT)</h2>
              <p className="mt-1 max-w-md text-sm font-extrabold text-rose-100">
                {strokeSafetyAlert?.type === 'emergency'
                  ? strokeSafetyAlert.messageTh
                  : telemetry_analysis.safety_guard_banner_th ||
                  'ตรวจพบการเสียสมดุล กรุณานั่งพักและจับราวจับ • ระบบแจ้งเตือนเจ้าหน้าที่แล้ว'}
              </p>
              <button
                type="button"
                onClick={() => {
                  setStrokeSafetyAlert(null);
                  setFatigueRestSec(0);
                  resetDemoSession();
                }}
                className="mt-4 inline-flex items-center gap-2 rounded-xl bg-white px-5 py-2 text-sm font-bold text-rose-700 shadow-lg hover:bg-rose-50 cursor-pointer"
              >
                <RotateCcw className="h-4 w-4" />
                รีเซ็ตสถานะเพื่อฝึกต่อ
              </button>
            </motion.div>
          )}

          {/* Top-Left Standing Distance Badge */}
          <div className="pointer-events-none absolute top-2.5 left-3 z-20">
            <div className="inline-flex items-center gap-1.5 bg-white/95 border-2 border-emerald-500 text-emerald-900 font-extrabold text-sm md:text-base px-3.5 py-1 rounded-full shadow">
              <Ruler className="w-4 h-4 text-emerald-600 shrink-0" />
              <span>ระยะยืน: {distanceLabel}</span>
            </div>
          </div>

          {/* Picture-in-Picture Live Animated Rehab Coach in Corner of Camera */}
          <Zone2PipCoachOverlay
            exercise={selectedExercise}
            targetSide={currentUiSide}
            visible={showPipCoach}
            onToggleVisible={() => setShowPipCoach((prev) => !prev)}
          />

          {/* Top-Right Side Target Badge & Live Angle Box */}
          <div className="pointer-events-none absolute top-2.5 right-3 z-20 flex flex-col items-end gap-1.5">
            <div className="inline-flex items-center gap-1.5 bg-blue-900/90 border border-sky-300 text-white font-extrabold text-sm md:text-base px-3.5 py-1 rounded-full shadow">
              <Crosshair className="w-4 h-4 text-sky-300 shrink-0" />
              <span>เป้าหมาย: {targetSideLabel}</span>
            </div>
            <div className="bg-white/95 border-2 border-blue-600 text-blue-950 font-black text-xl md:text-2xl px-3.5 py-1 rounded-xl shadow-md tabular-mono">
              {Math.round(activeRom)}° / {effectiveTargetRomDeg}°
            </div>
          </div>

          {/* Bottom Coaching Banner Inside Camera */}
          <div className="absolute bottom-2 left-3 right-3 z-20">
            <div
              role="status"
              data-testid="kiosk-unified-coaching-banner"
              className={`rounded-xl border-2 px-4 py-1.5 shadow-lg text-center transition-all duration-200 ${
                unifiedCoachingBanner.tone === 'rose'
                  ? 'bg-rose-50/95 border-rose-500 text-rose-900'
                  : unifiedCoachingBanner.tone === 'amber'
                    ? 'bg-amber-50/95 border-amber-500 text-amber-950'
                    : 'bg-white/95 border-emerald-500 text-blue-950'
              }`}
            >
              <div className="flex flex-wrap items-center justify-center gap-2">
                {(unifiedCoachingBanner.tone === 'rose' || unifiedCoachingBanner.tone === 'amber') ? (
                  <AlertTriangle className="w-4 h-4 shrink-0" />
                ) : isWaitLowerArmUi || holdPct >= 100 ? (
                  <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" />
                ) : (
                  <Timer className="w-4 h-4 text-emerald-600 shrink-0" />
                )}
                <p className="text-sm md:text-base font-extrabold leading-tight">
                  {unifiedCoachingBanner.tone === 'rose' || unifiedCoachingBanner.tone === 'amber'
                    ? `${unifiedCoachingBanner.titleTh}${unifiedCoachingBanner.subtitleTh ? ` — ${unifiedCoachingBanner.subtitleTh}` : ''}`
                    : isWaitLowerArmUi || holdPct >= 100
                      ? 'ค้างครบเวลา! +1 ครั้ง ค่อยๆ ลดแขนลงช้าๆ'
                      : isHoldPhase
                        ? `ค้างท่าไว้ ${holdTargetSec.toFixed(1)} วินาที! (ทำดีมาก!)`
                        : `ค่อยๆ ยกแขนขึ้นช้าๆ ให้ถึงเป้าหมาย ${effectiveTargetRomDeg} องศา (ทำดีมาก!)`}
                </p>
                {unifiedCoachingBanner.suggestedRomDeg ? (
                  <button
                    type="button"
                    onClick={() => {
                      const nextDeg = unifiedCoachingBanner.suggestedRomDeg ?? 90;
                      setReducedRomOverrideDeg(nextDeg);
                      setStrokeSafetyAlert(null);
                    }}
                    className="rounded-lg bg-blue-600 px-2.5 py-0.5 text-xs font-extrabold text-white shadow-xs hover:bg-blue-700 cursor-pointer"
                  >
                    ลดเป้าหมายเป็น {unifiedCoachingBanner.suggestedRomDeg}°
                  </button>
                ) : null}
                {unifiedCoachingBanner.showSafetyAcknowledge && (
                  <button
                    type="button"
                    onClick={() => {
                      setStrokeSafetyAlert(null);
                      setFatigueRestSec(0);
                    }}
                    className="rounded-lg border border-blue-400 bg-white px-2.5 py-0.5 text-xs font-extrabold text-blue-900 hover:bg-blue-50 cursor-pointer"
                  >
                    {(unifiedCoachingBanner.fatigueRestSec ?? 0) > 0 ? 'พร้อมฝึกต่อ' : 'รับทราบ'}
                  </button>
                )}
              </div>
            </div>
          </div>
        </SkeletonCanvas>
      </section>

      {/* ZONE 3: 3 Oversized Scoreboard Cards */}
      <section
        aria-label="สรุปผลคะแนนการทำกายภาพ"
        className="shrink-0 grid grid-cols-3 gap-2.5"
      >
        {/* Card 1: Completed Reps */}
        <div
          className={`relative overflow-hidden bg-white border-2 rounded-2xl px-3 py-2 text-center shadow-sm flex flex-col items-center justify-between transition-all duration-300 ${
            isRepCelebration
              ? 'border-emerald-500 ring-2 ring-emerald-300 shadow-md scale-[1.01]'
              : 'border-blue-200'
          }`}
        >
          <AnimatePresence>
            {isRepCelebration && (
              <motion.div
                key={`ripple-${rep_counter.valid_reps}`}
                initial={{ scale: 0.85, opacity: 0.65 }}
                animate={{ scale: 1.35, opacity: 0 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.75, ease: 'easeOut' }}
                className="pointer-events-none absolute inset-0 rounded-2xl border-4 border-emerald-500"
              />
            )}
          </AnimatePresence>
          <p className="text-blue-900 font-extrabold text-sm md:text-base leading-tight">
            จำนวนครั้งสำเร็จ
          </p>
          <div
            key={`rep-readout-${rep_counter.valid_reps}`}
            className={`text-blue-700 font-black text-2xl md:text-3xl tabular-mono leading-tight my-0.5 ${
              rep_counter.valid_reps > 0 ? 'animate-rep-pop' : ''
            }`}
          >
            {rep_counter.valid_reps} / {selectedExercise.target_reps} ครั้ง
          </div>
          <div className="w-full flex items-center justify-center gap-1">
            {Array.from({ length: Math.min(10, Math.max(1, selectedExercise.target_reps)) }).map((_, idx) => (
              <span
                key={idx}
                className={`h-2 flex-1 max-w-[16px] rounded-full transition-all duration-300 ${
                  idx < rep_counter.valid_reps
                    ? `bg-emerald-500 scale-110 shadow-xs ${
                        idx === rep_counter.valid_reps - 1 && isRepCelebration ? 'animate-rep-pop' : ''
                      }`
                    : 'bg-sky-100'
                }`}
              />
            ))}
          </div>
        </div>

        {/* Card 2: Target Hold Timer */}
        <div
          className={`bg-emerald-50/90 border-2 rounded-2xl px-3 py-2 text-center shadow-sm flex flex-col items-center justify-between transition-all duration-300 ${
            isWaitLowerArmUi || holdPct >= 100
              ? 'border-emerald-500 ring-2 ring-emerald-300 shadow-md'
              : 'border-emerald-400'
          }`}
        >
          <p className="inline-flex items-center justify-center gap-1.5 text-emerald-950 font-extrabold text-sm md:text-base leading-tight">
            <Timer className="w-4 h-4 text-emerald-700 shrink-0" />
            <span>ค้างท่าเป้าหมาย</span>
          </p>
          <div className="text-emerald-700 font-black text-2xl md:text-3xl tabular-mono leading-tight my-0.5 flex items-baseline justify-center gap-1">
            <span
              ref={holdTimerTextRef}
              data-testid="kiosk-hold-timer-display"
              className={isWaitLowerArmUi || holdPct >= 100 ? 'animate-rep-pop' : ''}
            >
              {isWaitLowerArmUi || holdPct >= 100
                ? `${holdTargetSec.toFixed(1)}s (ครบ!)`
                : `${liveHoldElapsed.toFixed(1)}s`}
            </span>
            <span className="text-sm md:text-lg font-extrabold text-emerald-700/80">
              / {holdTargetSec.toFixed(1)}s
            </span>
          </div>
          <div
            className={`w-full h-2 bg-emerald-200 rounded-full overflow-hidden transition-shadow duration-200 ${
              isWaitLowerArmUi || holdPct >= 100 ? 'animate-status-halo' : ''
            }`}
          >
            <div
              ref={holdProgressBarRef}
              className={`h-full rounded-full transition-[width,background-color] duration-150 ease-linear ${
                isWaitLowerArmUi || holdPct >= 100 ? 'bg-emerald-500' : 'bg-emerald-600'
              }`}
              style={{
                width: `${holdPct}%`,
                willChange: 'width',
              }}
            />
          </div>
        </div>

        {/* Card 3: ROM Semi-Circular Gauge */}
        <div
          className={`bg-white border-2 rounded-2xl px-3 py-2 text-center shadow-sm flex flex-col items-center justify-center transition-all duration-300 ${
            romPct >= 100 ? 'border-emerald-500 ring-2 ring-emerald-200 shadow-md' : 'border-teal-200'
          }`}
        >
          <p className="text-teal-950 font-extrabold text-sm md:text-base leading-tight">
            มุมการเคลื่อนไหว ROM
          </p>
          <div className="flex items-center justify-center gap-2 mt-0.5">
            <svg viewBox="0 0 120 66" className="w-16 h-9 md:w-20 md:h-11 overflow-visible shrink-0" aria-hidden="true">
              <defs>
                <linearGradient id="kioskRomClinicalGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor="#0284C7" />
                  <stop offset="100%" stopColor="#10B981" />
                </linearGradient>
              </defs>
              <path
                d="M 12 58 A 48 48 0 0 1 108 58"
                fill="none"
                stroke="#e2e8f0"
                strokeWidth="12"
                strokeLinecap="round"
              />
              <path
                d="M 12 58 A 48 48 0 0 1 108 58"
                fill="none"
                stroke="url(#kioskRomClinicalGrad)"
                strokeWidth="12"
                strokeLinecap="round"
                strokeDasharray="150.8"
                strokeDashoffset={150.8 * (1 - romPct / 100)}
                className="transition-[stroke-dashoffset,stroke] duration-200 ease-out"
              />
            </svg>
            <div className="text-left leading-tight">
              <div className="text-teal-700 font-black text-2xl md:text-3xl tabular-mono leading-none">
                {telemetry_analysis.current_rom_deg.toFixed(1)}°
              </div>
              <p className="text-slate-600 text-xs font-bold mt-0.5">
                (เป้าหมาย {effectiveTargetRomDeg}°)
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* ZONE 4: Compact Step-by-Step Guide & 3-Item Checklist */}
      <section
        aria-label="ขั้นตอนการฝึกและตารางตรวจสอบเช็คลิสต์"
        className="shrink-0 bg-white border-2 border-sky-100 rounded-2xl px-3.5 py-2 shadow-sm"
      >
        <h2
          key={`${selectedExercise.exercise_id}-${currentUiSide}`}
          className="flex items-center gap-1.5 text-sm md:text-base font-extrabold text-blue-950 mb-1.5 truncate animate-view-enter"
        >
          <Activity className="w-4 h-4 text-blue-700 shrink-0" />
          <span className="truncate">{selectedExercise.name_th} — {targetSideLabel}</span>
        </h2>

        <div className="grid grid-cols-12 gap-2.5 items-stretch">
          {/* Left 3 cols + Middle 6 cols: Synchronized Live Coach Model & 3 Step Cards (Zero Truncation) */}
          <Zone4CoachStageAndSteps
            exercise={selectedExercise}
            targetSide={currentUiSide}
            steps={postureSpec.steps}
            holdTargetSec={holdTargetSec}
            targetRomDeg={effectiveTargetRomDeg}
            onExpandPreview={() => setCoachModalOpen(true)}
          />

          {/* Right 3 cols: Compact 3-Item Live Checklist */}
          <div className="col-span-3 bg-emerald-50/60 border border-emerald-200 rounded-2xl px-2.5 py-1.5 flex flex-col justify-between space-y-1">
            <div className="flex items-center justify-between gap-1.5 pb-0.5 border-b border-emerald-200/70">
              <span className="text-[11px] md:text-xs font-black text-emerald-950 truncate">
                ตารางตรวจสอบเช็คลิสต์
              </span>
              <div className="flex items-center gap-1 shrink-0">
                <button
                  type="button"
                  onClick={toggleMasterFormAssessment}
                  aria-pressed={formRules.enabled}
                  className={`rounded-md px-1.5 py-0.5 text-[10px] font-extrabold cursor-pointer transition ${
                    formRules.enabled
                      ? 'bg-emerald-600 text-white'
                      : 'bg-slate-300 text-slate-700'
                  }`}
                >
                  {formRules.enabled ? 'เปิด' : 'ปิด'}
                </button>
                {hasDisabledFormRule && (
                  <button
                    type="button"
                    onClick={resetAllFormRules}
                    className="rounded-md bg-blue-600 px-1.5 py-0.5 text-[10px] font-extrabold text-white cursor-pointer"
                  >
                    เปิดหมด
                  </button>
                )}
              </div>
            </div>

            {coreChecklistItems.map(({ ruleKey, displayLabel, baseItem }) => (
              <button
                key={ruleKey}
                type="button"
                onClick={() => toggleSingleFormRule(ruleKey)}
                aria-pressed={baseItem.ruleActive}
                className="w-full flex items-center justify-between gap-1.5 text-left cursor-pointer hover:bg-emerald-100/50 rounded-md px-1 py-0.5 transition"
              >
                <span className="inline-flex items-center gap-1.5 min-w-0">
                  {!baseItem.ruleActive ? (
                    <Pause className="w-3.5 h-3.5 text-slate-400 shrink-0" />
                  ) : baseItem.ok ? (
                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600 shrink-0" />
                  ) : (
                    <AlertTriangle className="w-3.5 h-3.5 text-amber-500 shrink-0" />
                  )}
                  <span
                    className={`text-[11px] md:text-xs font-bold leading-tight truncate ${
                      !baseItem.ruleActive ? 'text-slate-400 line-through' : 'text-slate-800'
                    }`}
                  >
                    {displayLabel}
                  </span>
                </span>
                <span
                  className={`shrink-0 rounded-full px-1.5 py-0.5 text-[10px] font-extrabold ${
                    !baseItem.ruleActive
                      ? 'bg-slate-200 text-slate-600'
                      : baseItem.ok
                        ? 'bg-emerald-600 text-white'
                        : 'bg-amber-500 text-white'
                  }`}
                >
                  {!baseItem.ruleActive ? 'ปิดอยู่' : baseItem.ok ? 'ผ่าน' : baseItem.warnText}
                </span>
              </button>
            ))}
          </div>
        </div>
      </section>

      {/* ZONE 5: Compact Accessible Voice Coach Footer */}
      <footer className="shrink-0 relative bg-gradient-to-r from-teal-700 via-emerald-700 to-blue-800 text-white rounded-2xl px-4 py-1.5 flex items-center justify-between gap-3 shadow-sm">
        {/* Left: Dr. TechCare Mascot + Microphone badge + Label + Live Spoken Subtitle */}
        <div className="flex items-center gap-2.5 min-w-0 flex-1">
          <DrTechCareMascot
            pose={mascotPose}
            isSpeaking={isMascotSpeaking}
            size="sm"
            className="w-14 h-16 shrink-0 mr-3 animate-view-enter"
            testId="kiosk-voice-coach-mascot"
          />
          <span
            aria-hidden="true"
            className="inline-flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-white/15"
          >
            <Mic className="w-4 h-4 text-white" />
          </span>
          <span className="text-emerald-200 font-extrabold text-xs md:text-sm shrink-0">
            THAI AI VOICE COACH:
          </span>
          <p className="text-white font-bold text-sm md:text-base truncate">
            &ldquo;{displayedVoiceSubtitle}&rdquo;
          </p>
        </div>

        {/* Right: Audio Toggle, Thai/EN Language Pills, and Floating Clinical Tools Popover */}
        <div className="flex items-center gap-1.5 shrink-0">
          <button
            type="button"
            onClick={handleToggleVoiceMute}
            aria-pressed={voiceMuted}
            data-testid="kiosk-voice-mute-toggle"
            className={`font-extrabold text-xs md:text-sm px-3 py-1 rounded-lg cursor-pointer transition whitespace-nowrap inline-flex items-center gap-1.5 ${
              voiceMuted
                ? 'bg-amber-400 text-slate-950 hover:bg-amber-300'
                : 'bg-white/20 hover:bg-white/30 text-white'
            }`}
          >
            {voiceMuted ? <VolumeX className="h-3.5 w-3.5" /> : <Volume2 className="h-3.5 w-3.5" />}
            <span>{voiceMuted ? 'ปิดเสียง' : 'เสียงพากย์'}</span>
          </button>

          <button
            type="button"
            onClick={() => handleSelectZone5VoiceLang('th')}
            aria-pressed={zone5VoiceLang === 'th'}
            className={`px-2.5 py-1 rounded-lg text-xs md:text-sm font-extrabold cursor-pointer transition ${
              zone5VoiceLang === 'th'
                ? 'bg-white text-teal-900 shadow-xs'
                : 'bg-teal-900/50 hover:bg-teal-900/80 text-white'
            }`}
          >
            ไทย
          </button>

          <button
            type="button"
            onClick={() => handleSelectZone5VoiceLang('en')}
            aria-pressed={zone5VoiceLang === 'en'}
            className={`px-2.5 py-1 rounded-lg text-xs md:text-sm font-extrabold cursor-pointer transition ${
              zone5VoiceLang === 'en'
                ? 'bg-white text-teal-900 shadow-xs'
                : 'bg-teal-900/50 hover:bg-teal-900/80 text-white'
            }`}
          >
            EN
          </button>

          <button
            type="button"
            onClick={() => setToolsPopoverOpen((prev) => !prev)}
            aria-expanded={toolsPopoverOpen}
            title="ตั้งค่ากล้อง / โหมดจำลอง"
            className="bg-white/15 hover:bg-white/25 text-white font-bold px-2.5 py-1 rounded-lg text-xs cursor-pointer transition inline-flex items-center gap-1"
          >
            <SlidersHorizontal className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">ตั้งค่า</span>
          </button>
        </div>

        {/* Preserve live TTS & Speech Queue Engine invisibly without altering layout height */}
        <div className="sr-only" aria-hidden="true">
          <VoiceCoachBanner
            key={zone5VoiceLang}
            textTh={cleanSpeechText}
            kioskState={kiosk_state}
            highlightColor={ui_directives.highlight_color}
            muted={voiceMuted}
            onToggleMute={handleToggleVoiceMute}
            onSpeakingChange={setVoiceCoachSpeaking}
          />
        </div>

        {/* Floating Staff Utility & Demo Popover (Zero Impact on 100dvh Layout Height) */}
        {toolsPopoverOpen && (
          <div className="fixed bottom-14 right-3 z-40 w-[min(94vw,680px)] rounded-2xl border-2 border-sky-200 bg-white/98 p-3 text-slate-800 shadow-2xl backdrop-blur-md space-y-2.5 animate-popup-spring">
            <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-200 pb-2">
              <span className="inline-flex items-center gap-1.5 text-xs font-extrabold text-blue-950">
                <SlidersHorizontal className="w-3.5 h-3.5 text-blue-700" />
                <span>เครื่องมือผู้ดูแล &amp; การแสดงผลกล้อง</span>
              </span>
              <button
                type="button"
                onClick={() => setToolsPopoverOpen(false)}
                className="inline-flex items-center gap-1 rounded-lg bg-slate-100 px-2.5 py-1 text-xs font-bold text-slate-700 hover:bg-slate-200 cursor-pointer"
              >
                <span>ปิดหน้าต่าง</span>
                <X className="w-3.5 h-3.5" />
              </button>
            </div>

            <div className="flex flex-wrap items-center gap-1.5 text-xs">
              <button
                type="button"
                onClick={() => {
                  const nextLive = !livePoseActive;
                  setLivePoseActive(nextLive);
                  if (nextLive && isDemoMode) {
                    setAutoCycleDemo(false);
                    toggleDemoMode(false);
                  }
                }}
                className={`inline-flex items-center gap-1.5 rounded-xl border px-2.5 py-1.5 font-bold cursor-pointer ${
                  livePoseActive
                    ? 'border-emerald-300 bg-emerald-50 text-emerald-800'
                    : 'border-slate-300 bg-slate-100 text-slate-600'
                }`}
              >
                <Camera className="h-3.5 w-3.5" />
                <span>{livePoseActive ? 'กล้อง AI: ON' : 'กล้อง AI: OFF'}</span>
              </button>

              <button
                type="button"
                onClick={handleTogglePrivacyMode}
                aria-pressed={isPrivacyMode}
                className={`inline-flex items-center gap-1.5 rounded-xl border px-2.5 py-1.5 font-bold cursor-pointer ${
                  isPrivacyMode
                    ? 'border-emerald-500 bg-slate-900 text-emerald-300'
                    : 'border-sky-200 bg-sky-50 text-blue-900'
                }`}
              >
                {isPrivacyMode ? <EyeOff className="h-3.5 w-3.5" /> : <Eye className="h-3.5 w-3.5" />}
                <span>{isPrivacyMode ? 'โหมดความเป็นส่วนตัว' : 'โหมดภาพกล้องจริง'}</span>
              </button>

              <button
                type="button"
                onClick={handleToggleSkeletonOverlay}
                aria-pressed={showSkeletonOverlay}
                className={`inline-flex items-center gap-1.5 rounded-xl border px-2.5 py-1.5 font-bold cursor-pointer ${
                  showSkeletonOverlay
                    ? 'border-teal-300 bg-teal-50 text-teal-800'
                    : 'border-slate-300 bg-slate-100 text-slate-600'
                }`}
              >
                <Activity className="h-3.5 w-3.5" />
                <span>{showSkeletonOverlay ? 'โครงกระดูก: ON' : 'โครงกระดูก: OFF'}</span>
              </button>

              <button
                type="button"
                onClick={handleToggleDemoMode}
                className={`inline-flex items-center gap-1.5 rounded-xl border px-2.5 py-1.5 font-bold cursor-pointer ${
                  isDemoMode
                    ? 'border-blue-400 bg-blue-100 text-blue-950'
                    : 'border-slate-300 bg-slate-100 text-slate-700'
                }`}
              >
                <SlidersHorizontal className="h-3.5 w-3.5" />
                <span>{isDemoMode ? 'Demo: ON' : isEdgeDirectCamera ? 'Edge: ON' : 'Demo: OFF'}</span>
              </button>

              {canShowManualRepButton && (
                <button
                  type="button"
                  onClick={() => setDemoPreset('REP_COMPLETED')}
                  disabled={rep_counter.valid_reps >= selectedExercise.target_reps}
                  className="inline-flex items-center gap-1 rounded-xl border border-blue-300 bg-blue-50 px-2.5 py-1.5 font-bold text-blue-900 disabled:opacity-50 cursor-pointer"
                >
                  <PlusCircle className="h-3.5 w-3.5" />
                  <span>+1 ครั้ง</span>
                </button>
              )}

              {onOpenLegacyKiosk && (
                <button
                  type="button"
                  onClick={onOpenLegacyKiosk}
                  className="rounded-xl border border-slate-300 bg-white px-2.5 py-1.5 font-bold text-slate-700 hover:bg-slate-50 cursor-pointer"
                >
                  สแกนเต็มระบบ
                </button>
              )}

              <button
                type="button"
                onClick={() => handleEmergencyStopClick()}
                className="inline-flex items-center gap-1 rounded-xl bg-rose-600 px-3 py-1.5 font-extrabold text-white hover:bg-rose-500 cursor-pointer"
              >
                <AlertOctagon className="h-3.5 w-3.5" />
                <span>หยุดฉุกเฉิน</span>
              </button>
            </div>

            {/* Interactive Demo Simulator Controls inside Popover */}
            {isDemoMode && (
              <div className="space-y-2 border-t border-slate-200 pt-2">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <button
                    type="button"
                    onClick={() => setDrawerOpen((o) => !o)}
                    className="flex items-center gap-1.5 text-xs font-extrabold text-blue-900 cursor-pointer"
                  >
                    <span>Demo Simulator Controls</span>
                    {drawerOpen ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronUp className="h-3.5 w-3.5" />}
                  </button>
                  <div className="flex items-center gap-1.5">
                    <button
                      type="button"
                      onClick={() => setAutoCycleDemo((a) => !a)}
                      className={`inline-flex items-center gap-1 rounded-full px-3 py-1 text-xs font-bold cursor-pointer ${
                        autoCycleDemo ? 'bg-emerald-600 text-white' : 'bg-sky-100 text-blue-900'
                      }`}
                    >
                      {autoCycleDemo ? <Pause className="h-3 w-3" /> : <Play className="h-3 w-3" />}
                      <span>{autoCycleDemo ? 'หยุดจำลอง' : 'เล่นจำลอง'}</span>
                    </button>
                    <button
                      type="button"
                      onClick={resetDemoSession}
                      className="inline-flex items-center gap-1 rounded-full border border-slate-300 bg-white px-2.5 py-1 text-xs font-bold text-slate-700 cursor-pointer"
                    >
                      <RotateCcw className="h-3 w-3" />
                      รีเซ็ต
                    </button>
                  </div>
                </div>

                {drawerOpen && (
                  <div className="space-y-2">
                    <div className="flex flex-wrap gap-1">
                      {DEMO_PRESETS.map((p) => {
                        const active = demoPreset === p.id;
                        return (
                          <button
                            key={p.id}
                            type="button"
                            onClick={() => setDemoPreset(p.id)}
                            className={`rounded-lg px-2.5 py-1 text-[11px] font-bold cursor-pointer ${
                              active
                                ? p.id === 'IMPOSTOR'
                                  ? 'bg-rose-600 text-white'
                                  : p.id === 'TRUNK_LEAN' ||
                                      p.id === 'MULTI_PERSON' ||
                                      p.id === 'WRONG_LIMB' ||
                                      p.id === 'WRONG_PLANE'
                                    ? 'bg-amber-500 text-white'
                                    : 'bg-blue-600 text-white'
                                : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
                            }`}
                          >
                            {p.label}
                          </button>
                        );
                      })}
                    </div>
                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5 text-xs">
                      <label className="flex flex-col gap-0.5">
                        <div className="flex justify-between font-bold">
                          <span>ROM</span>
                          <span className="text-blue-700">{demoRomDeg.toFixed(0)}°</span>
                        </div>
                        <input
                          type="range"
                          min={0}
                          max={180}
                          step={1}
                          value={demoRomDeg}
                          onChange={(e) => {
                            setAutoCycleDemo(false);
                            setDemoRomDeg(Number(e.target.value));
                          }}
                          className="accent-blue-600"
                        />
                      </label>
                      <label className="flex flex-col gap-0.5">
                        <div className="flex justify-between font-bold">
                          <span>เอียงลำตัว</span>
                          <span className="text-emerald-700">{demoTrunkTiltDeg.toFixed(1)}°</span>
                        </div>
                        <input
                          type="range"
                          min={0}
                          max={25}
                          step={0.5}
                          value={demoTrunkTiltDeg}
                          onChange={(e) => {
                            setAutoCycleDemo(false);
                            setDemoTrunkTiltDeg(Number(e.target.value));
                          }}
                          className="accent-amber-500"
                        />
                      </label>
                      <label className="flex flex-col gap-0.5">
                        <div className="flex justify-between font-bold">
                          <span>ระยะยืน</span>
                          <span className="text-teal-700">{demoDistanceM.toFixed(2)} ม.</span>
                        </div>
                        <input
                          type="range"
                          min={1.4}
                          max={2.6}
                          step={0.05}
                          value={demoDistanceM}
                          onChange={(e) => setDemoDistanceM(Number(e.target.value))}
                          className="accent-emerald-600"
                        />
                      </label>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </footer>

      <ExpandedCoachModal
        open={coachModalOpen}
        exercise={selectedExercise}
        targetSide={currentUiSide}
        targetSideLabel={targetSideLabel}
        steps={postureSpec.steps}
        onClose={() => setCoachModalOpen(false)}
      />
    </div>
  );
}

