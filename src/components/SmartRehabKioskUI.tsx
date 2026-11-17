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
  extractHybrid512Descriptor,
  extractLegacyV1Hybrid512Descriptor,
  flipImageDataHorizontally,
  flipLandmarksHorizontally,
  isArcFaceSessionReady,
  preloadArcFaceOnnx,
  selectClinicalCameraDeviceId,
  selectPrimaryFace,
} from '@drtechcares/face-client';
import {
  detectBiometricVectorEngine,
  isLegacyV1HybridVector,
} from '@drtechcares/smart-biometric-core';
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
  HAND21_CONNECTIONS,
  POSE_INDEX,
  computeJointAngle3D,
  solveStrokeClinicalJointBiomechanics,
  stepStrokeHoldAndRepStateMachine,
  type StrokeAnatomyBaseline,
  type StrokeHoldRepState,
  type StrokeSideViolationCode,
  type StrokeTargetOverlayHint,
} from '@drtechcares/pose-engine';
import {
  Activity,
  AlertOctagon,
  AlertTriangle,
  Camera,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Eye,
  EyeOff,
  Gamepad2,
  HeartPulse,
  Pause,
  Play,
  PlusCircle,
  RotateCcw,
  SlidersHorizontal,
  Sparkles,
  Timer,
  UserCheck,
  Wifi,
  WifiOff,
} from 'lucide-react';
import { ExercisePostureGuide, ExercisePostureSvg } from './ExercisePostureGuide';
import { FaceScanOverlay } from './FaceScanOverlay';
import { RomGauge } from './RomGauge';
import { SkeletonCanvas } from './SkeletonCanvas';
import { VoiceCoachBanner } from './VoiceCoachBanner';
import {
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

export { isUpperBodyVisibleInFrame };

interface SmartRehabKioskUIProps {
  patient?: PatientSummary | null;
  ownerBiometrics?: OwnerBiometricTemplate | null;
  prescribedExercises?: ExercisePrescriptionOption[];
  isStaffAuthorized?: boolean;
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
  name_th?: string;
  name_en?: string;
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
    const inferredSide: 'LEFT' | 'RIGHT' | 'BOTH' | undefined =
      expectedPlane === 'bilateral' ||
      expectedPlane === 'bilateral_push' ||
      jointMode === 'bilateral_raise' ||
      jointMode === 'bobath_push'
        ? 'BOTH'
        : targetJoint === 'left_shoulder'
          ? 'LEFT'
          : targetJoint === 'right_shoulder'
            ? 'RIGHT'
            : undefined;
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
        solved.sideViolationBannerTh ?? getPrescribedLimbPromptTh(targetJoint, expectedPlane).bannerAlertTh;
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
    if (!cur) return { ...target };
    return {
      x: cur.x + (target.x - cur.x) * safeAlpha,
      y: cur.y + (target.y - cur.y) * safeAlpha,
      z: (cur.z ?? 0) + ((target.z ?? 0) - (cur.z ?? 0)) * safeAlpha,
      visibility: target.visibility ?? cur.visibility,
    };
  });
}

/**
 * Exact `object-cover` + `scale-x-[-1]` coordinate projection from raw camera resolution / normalized
 * coords onto the synced `<canvas>` element without black bars or lateral drift.
 */
export function mapVideoPointToCoverCanvas(
  vx: number,
  vy: number,
  videoW: number,
  videoH: number,
  canvasW: number,
  canvasH: number,
  isMirrored = true,
): { x: number; y: number } {
  const cw = canvasW > 0 ? canvasW : 640;
  const ch = canvasH > 0 ? canvasH : 480;
  const vw = videoW > 0 ? videoW : cw;
  const vh = videoH > 0 ? videoH : ch;
  const px = vx <= 1.05 ? vx * vw : vx;
  const py = vy <= 1.05 ? vy * vh : vy;
  const scale = Math.max(cw / vw, ch / vh);
  const scaledW = vw * scale;
  const scaledH = vh * scale;
  const offsetX = (cw - scaledW) / 2;
  const offsetY = (ch - scaledH) / 2;
  const rawCanvasX = px * scale + offsetX;
  const canvasY = py * scale + offsetY;
  const canvasX = isMirrored ? cw - rawCanvasX : rawCanvasX;
  return { x: canvasX, y: canvasY };
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
        drawRehabFaceBox(faceBox, '#F59E0B', '⚠️ พบมากกว่า 1 คนในเฟรม', true);
      } else if (ownerFaceVerified) {
        drawRehabFaceBox(faceBox, '#10B981', `✓ เจ้าของบัญชี: ${ownerName}`, false);
      } else {
        drawRehabFaceBox(faceBox, '#EF4444', `⛔ ไม่ใช่เจ้าของบัญชี (${ownerName})`, false);
      }
    }
    for (const extra of extraFaceBoxes) {
      drawRehabFaceBox(extra, '#EF4444', '⚠️ บุคคลอื่น', true);
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
        if (!wLm || (wLm.visibility ?? 0) < 0.35) continue;
        const wx = covT.offsetX + (1 - wLm.x) * covT.renderW;
        const wy = covT.offsetY + wLm.y * covT.renderH;
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
  const minVisFor = (idx: number) => (UPPER_BODY_KEYPOINTS.has(idx) ? 0.35 : 0.55);

  const activeColor =
    opts.isImpostorOrEmergency || opts.wrongSideRaised
      ? '#EF4444'
      : opts.postureBannerTh || opts.wrongPlaneRaised || opts.isMultiPerson || opts.trunkTiltDeg > 12 || opts.lowVisibilityAlertTh
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
  const activeWristLm =
    opts.targetJoint === 'left_shoulder'
      ? primary[POSE_INDEX.LEFT_WRIST]
      : opts.targetJoint === 'right_shoulder'
        ? primary[POSE_INDEX.RIGHT_WRIST]
        : undefined;

  const isOppositeArmIdx = (idx: number) => {
    if (opts.targetJoint === 'left_shoulder') {
      return (
        idx === POSE_INDEX.RIGHT_ELBOW ||
        idx === POSE_INDEX.RIGHT_WRIST ||
        idx === POSE_INDEX.RIGHT_PINKY ||
        idx === POSE_INDEX.RIGHT_INDEX ||
        idx === POSE_INDEX.RIGHT_THUMB
      );
    }
    if (opts.targetJoint === 'right_shoulder') {
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

  const isPointDrawable = (idx: number): boolean => {
    const lm = primary[idx];
    if (!lm) return false;
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
      // Never draw hallucinated off-screen knee/ankle lines during seated upper-body camera exercises,
      // while preserving genuinely visible in-frame feet for full-body stance verification.
      if (isKnee) return false;
      if (isFootOrAnkle) {
        const inFrame = lm.x >= 0.02 && lm.x <= 0.98 && lm.y >= 0.02 && lm.y <= 0.98;
        if (!inFrame || (lm.visibility ?? 0) < 0.55) return false;
      }
      if (isHip && !hipsCleanlyBelowShoulders) return false;
    }
    const minV = isOppositeArmIdx(idx) ? 0.28 : minVisFor(idx);
    if ((lm.visibility ?? 0) < minV) return false;
    // Prevent phantom opposite elbow/wrist keypoints collapsed onto the reaching hand
    if (
      isOppositeArmIdx(idx) &&
      !opts.wrongSideRaised &&
      activeWristLm &&
      (activeWristLm.visibility ?? 0) >= 0.28 &&
      Math.hypot(lm.x - activeWristLm.x, lm.y - activeWristLm.y) < 0.08
    ) {
      return false;
    }
    return true;
  };

  const isBilateralMode =
    opts.expectedPlane === 'bilateral' || opts.expectedPlane === 'bilateral_push';

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
    if (opts.targetJoint === 'left_shoulder') {
      return isLeftEdge;
    }
    if (opts.targetJoint === 'right_shoulder') {
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
    if (!isPointDrawable(aIdx) || !isPointDrawable(bIdx)) {
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
      if (!isPointDrawable(aIdx) || !isPointDrawable(bIdx)) continue;
      const aLm = primary[aIdx]!;
      const bLm = primary[bIdx]!;
      if (
        aLm.x < -0.02 || aLm.x > 1.02 || aLm.y < -0.02 || aLm.y > 1.02 ||
        bLm.x < -0.02 || bLm.x > 1.02 || bLm.y < -0.02 || bLm.y > 1.02
      ) {
        continue;
      }
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

  // High-precision 21-point HandLandmarker skeleton overlay for wrist / forearm / finger exercises
  if (opts.handLandmarks21 && opts.handLandmarks21.length >= 21) {
    ctx.save();
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = activeColor;
    ctx.lineWidth = 3;
    ctx.shadowColor = activeColor;
    ctx.shadowBlur = 8;
    for (const [a, b] of HAND21_CONNECTIONS) {
      const lmA = opts.handLandmarks21[a];
      const lmB = opts.handLandmarks21[b];
      if (!lmA || !lmB) continue;
      const ptA = { x: rawCanvasX(lmA.x), y: rawCanvasY(lmA.y) };
      const ptB = { x: rawCanvasX(lmB.x), y: rawCanvasY(lmB.y) };
      ctx.beginPath();
      ctx.moveTo(ptA.x, ptA.y);
      ctx.lineTo(ptB.x, ptB.y);
      ctx.stroke();
    }
    ctx.shadowBlur = 0;
    for (let i = 0; i < 21; i += 1) {
      const lm = opts.handLandmarks21[i];
      if (!lm) continue;
      const pt = { x: rawCanvasX(lm.x), y: rawCanvasY(lm.y) };
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

  // On-body alert badge when patient raises in wrong movement plane (e.g. side abduction instead of forward flexion)
  if (opts.wrongPlaneRaised && !opts.wrongSideRaised) {
    const activeElbowIdx =
      opts.targetJoint === 'left_shoulder' ? POSE_INDEX.LEFT_ELBOW : POSE_INDEX.RIGHT_ELBOW;
    const activeShIdx =
      opts.targetJoint === 'left_shoulder' ? POSE_INDEX.LEFT_SHOULDER : POSE_INDEX.RIGHT_SHOULDER;
    const activeLm = primary[activeElbowIdx] ?? primary[activeShIdx] ?? primary[POSE_INDEX.NOSE];
    const planeInfo = getPrescribedPlanePromptTh(opts.expectedPlane ?? 'any');
    const warnText = `✗ ${planeInfo.wrongPlaneTh} • กรุณา${planeInfo.requiredPlaneTh}`;
    ctx.font = '800 13px "Sarabun", sans-serif';
    const wtw = ctx.measureText(warnText).width;
    const wbw = Math.min(w - 24, wtw + 24);
    const wbh = 32;
    const wx = activeLm
      ? Math.max(wbw / 2 + 12, Math.min(w - wbw / 2 - 12, rawCanvasX(activeLm.x)))
      : w / 2;
    const wy = activeLm ? Math.max(88, Math.min(h - 36, rawCanvasY(activeLm.y) - 34)) : 96;

    ctx.fillStyle = 'rgba(220, 38, 38, 0.92)';
    ctx.strokeStyle = '#FCA5A5';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.roundRect(wx - wbw / 2, wy - wbh / 2, wbw, wbh, 16);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = '#FFFFFF';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(warnText, wx, wy);
  }

  // Prominent Strict Side Isolation alert banner + on-limb badge when wrong arm / single-arm in BOTH mode
  if (opts.wrongSideRaised) {
    const limbInfo = getPrescribedLimbPromptTh(opts.targetJoint, opts.expectedPlane);
    const bannerText = opts.wrongSideBannerTh || limbInfo.bannerAlertTh;
    ctx.font = '800 14px "Sarabun", sans-serif';
    const bTw = ctx.measureText(bannerText).width;
    const bBw = Math.min(w - 24, bTw + 28);
    const bBh = 36;
    const bX = w / 2;
    const bY = 34;

    ctx.fillStyle = 'rgba(220, 38, 38, 0.95)';
    ctx.strokeStyle = '#FCA5A5';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.roundRect(bX - bBw / 2, bY - bBh / 2, bBw, bBh, 18);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = '#FFFFFF';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(bannerText, bX, bY, bBw - 20);
  } else if (opts.repSuccessBannerTh && !opts.wrongPlaneRaised && !opts.postureBannerTh) {
    ctx.font = '800 14px "Sarabun", sans-serif';
    const gTw = ctx.measureText(opts.repSuccessBannerTh).width;
    const gBw = Math.min(w - 24, gTw + 28);
    const gBh = 34;
    ctx.fillStyle = 'rgba(5, 150, 105, 0.94)';
    ctx.strokeStyle = '#6EE7B7';
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    ctx.roundRect(w / 2 - gBw / 2, 16, gBw, gBh, 17);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = '#FFFFFF';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(opts.repSuccessBannerTh, w / 2, 33, gBw - 18);
  }

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
      const label = pIdx === 0 ? '✓ คนไข้หลัก' : '⚠ บุคคลเกินในเฟรม';

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

  const clinicalAlertBanner = !opts.wrongSideRaised && !opts.wrongPlaneRaised
    ? opts.postureBannerTh
      ? { text: `⚠ ${opts.postureBannerTh}`, bg: 'rgba(220, 38, 38, 0.92)', border: '#FCA5A5' }
      : opts.lowVisibilityAlertTh
        ? { text: `⚠ ${opts.lowVisibilityAlertTh}`, bg: 'rgba(217, 119, 6, 0.92)', border: '#FDE68A' }
        : null
    : null;
  if (clinicalAlertBanner) {
    ctx.font = '800 13px "Sarabun", sans-serif';
    const tw = ctx.measureText(clinicalAlertBanner.text).width;
    const bw = Math.min(w - 24, tw + 26);
    const bh = 32;
    const bx = w / 2;
    const by = h - 68;
    ctx.fillStyle = clinicalAlertBanner.bg;
    ctx.strokeStyle = clinicalAlertBanner.border;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.roundRect(bx - bw / 2, by - bh / 2, bw, bh, 16);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = '#FFFFFF';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(clinicalAlertBanner.text, bx, by, bw - 18);
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
  onExerciseCompleted,
  onAllExercisesCompleted,
  onEmergencyStop,
  onOpenStaffPortal,
  onOpenLegacyKiosk,
}: SmartRehabKioskUIProps) {
  const socketOptions = useMemo(
    () =>
      prescribedExercises || patientProp !== undefined
        ? { prescribedExercises, patientOverride: patientProp }
        : undefined,
    [prescribedExercises, patientProp],
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

  const [voiceMuted, setVoiceMuted] = useState(false);
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
  const ownerBiometricsRef = useRef<OwnerBiometricTemplate | null | undefined>(ownerBiometricsProp);
  ownerBiometricsRef.current = ownerBiometricsProp;

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
  const repPhaseRef = useRef<StrokeHoldRepState['phase']>('WAIT_RAISE');
  const cooldownStartedAtRef = useRef<number>(0);
  const repCelebrateUntilRef = useRef<number>(0);
  const lastFrameTimeRef = useRef<number>(0);
  const repSuccessToastRef = useRef<string | null>(null);

  useEffect(() => {
    completedRepsRef.current = 0;
    holdElapsedSecRef.current = 0;
    repPhaseRef.current = 'WAIT_RAISE';
    cooldownStartedAtRef.current = 0;
    repCelebrateUntilRef.current = 0;
    rawKeypointsRef.current = null;
    smoothedRenderKeypointsRef.current = null;
    allRawKeypointsRef.current = [];
    repSuccessToastRef.current = null;
    setRepSuccessToastTh(null);
    if (holdProgressBarRef.current) {
      holdProgressBarRef.current.style.width = '0%';
    }
    if (holdTimerTextRef.current) {
      holdTimerTextRef.current.textContent = `${selectedExercise.hold_sec.toFixed(1)}s`;
    }
  }, [selectedExercise.exercise_id, selectedExercise.hold_sec]);

  useEffect(() => {
    if (demoValidReps === 0 && completedRepsRef.current > 0 && repPhaseRef.current !== 'COOLDOWN') {
      completedRepsRef.current = 0;
      holdElapsedSecRef.current = 0;
      repPhaseRef.current = 'WAIT_RAISE';
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
  const trunkLeanAlert = isFormRuleActive(formRules, 'trunk') && compensations.includes('TRUNK_LEAN');
  const shoulderHikeAlert = isFormRuleActive(formRules, 'posture') && compensations.includes('SHOULDER_HIKE');
  const tooFastAlert = isFormRuleActive(formRules, 'velocity') && compensations.includes('TOO_FAST');
  const wrongLimbAlert = isFormRuleActive(formRules, 'limb') && compensations.includes('WRONG_LIMB');
  const wrongPlaneAlert = isFormRuleActive(formRules, 'plane') && compensations.includes('WRONG_PLANE');
  const activeExpectedPlane = resolveExpectedPlaneForExercise(selectedExercise);
  const isHoldPhase = kiosk_state === 'HOLD_PHASE';
  const isRepCelebration = kiosk_state === 'REP_COMPLETED' || kiosk_state === 'SET_FINISHED';
  const isEmergency = kiosk_state === 'EMERGENCY_HALT';

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
    let lastFatigueTriggerAt = 0;
    let latestRawPrimaryRomDeg: number | null = null;
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
          holdElapsedSecRef.current = 0;
          completedRepsRef.current = 0;
          repPhaseRef.current = 'WAIT_RAISE';
          cooldownStartedAtRef.current = 0;
          repCelebrateUntilRef.current = 0;
          repSuccessToastRef.current = null;
          setRepSuccessToastTh(null);
        };

        // Decoupled AI Inference Loop (downscaled 384x216 capture canvas for 30+ FPS CPU/GPU speed)
        const runAiInference = () => {
          if (!active || !videoRef.current || !poseLandmarker) return;
          const curEx = selectedExerciseRef.current;
          const curExpectedPlane = resolveExpectedPlaneForExercise(curEx);
          if (trackedExId !== curEx.exercise_id) {
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
            const stepACheckIdentity = isFormRuleActive(stepARules, 'identity');

            // Continuous 1:1 Account Owner Face Lock every 8 pose cycles (~250ms)
            ownerFaceCycle += 1;
            const activeOwnerBio = ownerBiometricsRef.current;
            if (activeOwnerBio?.biometricEngine === 'arcface-onnx' && !isArcFaceSessionReady()) {
              void preloadArcFaceOnnx();
            }
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
                  // Do NOT stall rep counting when a single patient stands at 1.8-2.2m exercise distance
                  if (latestPoseCount === 1) {
                    cachedOwnerVerified = true;
                    if (lastOwnerRehabKey === 'NO_OWNER_FACE') {
                      lastOwnerRehabKey = '';
                      setOwnerRehabGuardAlert(null);
                    }
                  }
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

                  if (effFaceCount > 1) {
                    cachedOwnerVerified = false;
                    const key = 'MULTI_PERSON';
                    if (lastOwnerRehabKey !== key) {
                      lastOwnerRehabKey = key;
                      setOwnerRehabGuardAlert({
                        status: 'MULTI_PERSON',
                        text: '⚠️ พบผู้อื่นอยู่ในเฟรม กรุณาให้เจ้าของบัญชีอยู่หน้ากล้องเพียงคนเดียว',
                      });
                    }
                  } else if (!activeOwnerBio || activeOwnerBio.masterEmbedding.length === 0) {
                    cachedOwnerVerified = true;
                    if (lastOwnerRehabKey !== '') {
                      lastOwnerRehabKey = '';
                      setOwnerRehabGuardAlert(null);
                    }
                  } else {
                    if (!faceAlignCanvas) faceAlignCanvas = document.createElement('canvas');
                    if (!faceRoiCanvas) faceRoiCanvas = document.createElement('canvas');
                    const crop = alignFaceTo112(v, primaryFaceLm, faceAlignCanvas, faceRoiCanvas);
                    if (crop) {
                      const flippedCrop = flipImageDataHorizontally(crop);
                      const flippedLm = flipLandmarksHorizontally(primaryFaceLm);
                      const ownerEngine = detectBiometricVectorEngine(
                        activeOwnerBio.masterEmbedding,
                        activeOwnerBio.biometricEngine,
                      );
                      const livePose = evaluateLivenessAndPose(primaryFaceLm, null, effFaceCount);
                      const rollDeg = computeEyeRollAngleDeg(primaryFaceLm, v.videoWidth, v.videoHeight);
                      const probeRatios = extractGeometricAnthropometricRatios(primaryFaceLm);

                      const applyOwnerResult = (matched: boolean, hasTemplate: boolean) => {
                        cachedOwnerVerified = !hasTemplate || matched;
                        if (!cachedOwnerVerified) {
                          const key = `IMPOSTOR:${activeOwnerBio.fullName}`;
                          if (lastOwnerRehabKey !== key) {
                            lastOwnerRehabKey = key;
                            setOwnerRehabGuardAlert({
                              status: 'IMPOSTOR_FACE',
                              text: `⛔ คุณไม่ใช่เจ้าของบัญชี (${activeOwnerBio.fullName}) ไม่สามารถทำรายการแทนกันได้`,
                            });
                          }
                        } else if (lastOwnerRehabKey !== '') {
                          lastOwnerRehabKey = '';
                          setOwnerRehabGuardAlert(null);
                        }
                      };

                      if (ownerEngine === 'arcface-onnx' && isArcFaceSessionReady()) {
                        const cropCopy = {
                          width: crop.width,
                          height: crop.height,
                          data: new Uint8ClampedArray(crop.data),
                        };
                        void extractArcFaceOnnxEmbedding(cropCopy)
                          .then((arcVec) => {
                            const ownerRes = verifyOwnerFace1to1({
                              probeVector: arcVec,
                              probeEngine: 'arcface-onnx',
                              yawDeg: -livePose.yawDeg,
                              pitchDeg: livePose.pitchDeg,
                              rollDeg,
                              probeRatios,
                              ownerTemplate: activeOwnerBio,
                            });
                            applyOwnerResult(ownerRes.matched, ownerRes.hasTemplate);
                          })
                          .catch(() => { });
                      } else if (ownerEngine !== 'arcface-onnx') {
                        const probeVec = extractHybrid512Descriptor(crop, primaryFaceLm);
                        const mirroredVec = extractHybrid512Descriptor(flippedCrop, flippedLm);
                        const ownerIsLegacyV1 = isLegacyV1HybridVector(activeOwnerBio.masterEmbedding);
                        const legacyV1ProbeVector = ownerIsLegacyV1
                          ? extractLegacyV1Hybrid512Descriptor(crop, primaryFaceLm)
                          : undefined;
                        const mirroredLegacyV1ProbeVector = ownerIsLegacyV1
                          ? extractLegacyV1Hybrid512Descriptor(flippedCrop, flippedLm)
                          : undefined;
                        const ownerRes = verifyOwnerFace1to1({
                          probeVector: probeVec,
                          mirroredProbeVector: mirroredVec,
                          probeEngine: 'hybrid-512d',
                          ...(legacyV1ProbeVector ? { legacyV1ProbeVector } : {}),
                          ...(mirroredLegacyV1ProbeVector
                            ? { mirroredLegacyV1ProbeVector }
                            : {}),
                          yawDeg: -livePose.yawDeg,
                          pitchDeg: livePose.pitchDeg,
                          rollDeg,
                          probeRatios,
                          ownerTemplate: activeOwnerBio,
                        });
                        applyOwnerResult(ownerRes.matched, ownerRes.hasTemplate);
                      }
                    }
                  }
                }
              } catch {
                // keep loop resilient across transient face detection frames
              }
            }

            let ownerPoseLinked = true;
            if (stepACheckIdentity && cachedOwnerFaceBox && allLandmarks[0]) {
              const { w: cwPose, h: chPose } = canvasSizeRef.current;
              const covPose = computeObjectCoverLayout(v.videoWidth, v.videoHeight, cwPose, chPose);
              ownerPoseLinked = isPoseSkeletonLinkedToOwnerFace(allLandmarks[0], cachedOwnerFaceBox, covPose);
              if (!ownerPoseLinked && lastOwnerRehabKey === '') {
                lastOwnerRehabKey = 'INTRUDER_HAND';
                setOwnerRehabGuardAlert({
                  status: 'INTRUDER_HAND',
                  text: '⚠️ ตรวจพบท่าทางจากบุคคลอื่น กรุณาให้เจ้าของบัญชีทำกายภาพด้วยตนเองเท่านั้น',
                });
              } else if (ownerPoseLinked && lastOwnerRehabKey === 'INTRUDER_HAND' && cachedOwnerVerified) {
                lastOwnerRehabKey = '';
                setOwnerRehabGuardAlert(null);
              }
            }

            if (latestPoseCount > 1 && stepACheckIdentity) {
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
                const applyOneEuro = (
                  flt: ClinicalOneEuroLandmarkFilter,
                  src: readonly NormalizedLandmark[],
                ): NormalizedLandmark[] =>
                  flt.filterLandmarks(src, now).map((lm, idx) => ({
                    x: lm.x,
                    y: lm.y,
                    z: lm.z ?? src[idx]?.z ?? 0,
                    visibility: lm.visibility ?? src[idx]?.visibility ?? 0,
                  }));
                const landmarks = !enableCustomSmooth
                  ? stabilizedResult.landmarks
                  : smoothingMode === 'one_euro'
                    ? applyOneEuro(pose2dOneEuroFilter, stabilizedResult.landmarks)
                    : landmarkSmoother.apply(stabilizedResult.landmarks);
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
                // Instant block on frame 1 when wrong arm or single-arm in BOTH mode detected!
                latestSideViolation = measured.sideViolation ?? 'NONE';
                latestWrongSideRaised = Boolean(measured.wrongSideRaised);
                latestWrongSideBannerTh = measured.wrongSideBannerTh;
                latestWrongPlaneRaised = wrongPlaneStreak >= 2;
                latestWrongPlaneBannerTh = measured.wrongPlaneBannerTh;
                latestLowVisibilityAlertTh = measured.lowVisibilityAlertTh ?? null;
                latestPostureCompensated = Boolean(measured.postureCompensated);
                latestPostureBannerTh = measured.postureBannerTh ?? null;
                latestRawPrimaryRomDeg = measured.rawPrimaryRomDeg ?? measured.romDeg;

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

                if (measured.trunkTiltDeg > 18) {
                  sustainedSwayFrames += 1;
                } else {
                  sustainedSwayFrames = 0;
                }
                if (sustainedSwayFrames >= 3 || headDropVel > 0.65 || shDropVel > 0.65) {
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

                const hasStrokeCompensation =
                  (measured.shoulderTiltDeg ?? 0) > 13.5 || measured.trunkTiltDeg > 12;
                if (hasStrokeCompensation && !measured.wrongSideRaised) {
                  if (now - lastFatigueTriggerAt > 450) {
                    consecutiveCompEvents += 1;
                    lastFatigueTriggerAt = now;
                    if (consecutiveCompEvents >= 3) {
                      const nextSuggested = Math.max(30, Math.round(effectiveTargetRomRef.current - 15));
                      setStrokeSafetyAlert({
                        type: 'fatigue',
                        messageTh: `พบการยักไหล่/เอียงลำตัวช่วยต่อเนื่อง (${consecutiveCompEvents} ครั้ง) แนะนำลดเป้าหมายเป็น ${nextSuggested}° หรือพัก 15 วินาที`,
                        suggestedRom: nextSuggested,
                      });
                      setFatigueRestSec((cur) => (cur > 0 ? cur : 15));
                      consecutiveCompEvents = 0;
                    }
                  }
                } else if (!hasStrokeCompensation && (latestRawPrimaryRomDeg ?? 0) >= 25) {
                  consecutiveCompEvents = 0;
                }
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
          if (trackedExId !== curEx.exercise_id) {
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
          const checkPosture = isFormRuleActive(activeRules, 'posture');
          const checkIdentity = isFormRuleActive(activeRules, 'identity');
          const holdTargetSec = Math.max(0.5, curEx.hold_sec);
          const { w: canvasW, h: canvasH } = canvasSizeRef.current;

          if (latestPoseCount > 1 && checkIdentity) {
            holdElapsedSecRef.current = 0;
            if (holdProgressBarRef.current) holdProgressBarRef.current.style.width = '0%';
            if (holdTimerTextRef.current) holdTimerTextRef.current.textContent = `${holdTargetSec.toFixed(1)}s`;
            if (lastEmittedHold !== curEx.hold_sec) {
              lastEmittedHold = curEx.hold_sec;
              lastHoldEmitAt = now;
              setDemoHoldRemainingSec(curEx.hold_sec);
            }
            emitPresetState('MULTI_PERSON');
            drawLivePoseOverlayOnCanvas(poseCanvasRef.current, allRawKeypointsRef.current, {
              targetJoint: curEx.target_joint,
              expectedPlane: curExpectedPlane,
              exerciseMeta: curEx,
              romDeg: targetRomDeg !== null ? smoothedRom : null,
              targetRomDeg: curEx.target_rom_deg,
              holdRatio: 0,
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
            holdElapsedSecRef.current = 0;
            if (holdProgressBarRef.current) holdProgressBarRef.current.style.width = '0%';
            if (holdTimerTextRef.current) holdTimerTextRef.current.textContent = `${holdTargetSec.toFixed(1)}s`;
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
            const effPostureCompensated = checkPosture ? latestPostureCompensated : false;
            const effPostureBannerTh = checkPosture ? latestPostureBannerTh : null;
            targetRomDeg =
              effWrongSideRaised || effWrongPlaneRaised ? 0 : latestRawPrimaryRomDeg;

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
              (demoPresetRef.current === 'MULTI_PERSON' && checkIdentity) ||
              countingLockedRef.current ||
              (checkIdentity &&
                lastOwnerRehabKey !== '' &&
                lastOwnerRehabKey !== 'NO_OWNER_FACE');

            const isCompensatedNow =
              (checkTrunk && latestTrunkTiltDeg > 12) || effPostureCompensated;

            // Step Ref-Driven Clinical 60FPS Hold & Rep State Machine (Zero Stale Closures)
            const repStep = stepStrokeHoldAndRepStateMachine({
              state: {
                phase: repPhaseRef.current,
                holdElapsedSec: holdElapsedSecRef.current,
                completedReps: completedRepsRef.current,
                cooldownStartedAtMs: cooldownStartedAtRef.current,
                repCelebrateUntilMs: repCelebrateUntilRef.current,
              },
              activeRomDeg: targetRomDeg !== null ? smoothedRom : null,
              targetRomDeg: effectiveTargetRomRef.current,
              targetHoldSec: holdTargetSec,
              targetReps: curEx.target_reps,
              dtSec,
              nowMs: now,
              isKneeExercise: curEx.target_joint.includes('knee'),
              sideViolation: effSideViolation,
              wrongSideBannerTh: effWrongSideBannerTh,
              wrongPlaneRaised: effWrongPlaneRaised,
              wrongPlaneBannerTh: effWrongPlaneBannerTh,
              isCompensatedNow,
              postureBannerTh: effPostureBannerTh,
              blockCounting,
            });

            repPhaseRef.current = repStep.nextState.phase;
            holdElapsedSecRef.current = repStep.nextState.holdElapsedSec;
            completedRepsRef.current = repStep.nextState.completedReps;
            cooldownStartedAtRef.current = repStep.nextState.cooldownStartedAtMs;
            repCelebrateUntilRef.current = repStep.nextState.repCelebrateUntilMs;

            // Direct 60FPS DOM updates for jitter-free Hold Progress Bar & Countdown Text
            if (holdProgressBarRef.current) {
              holdProgressBarRef.current.style.width = `${repStep.holdProgressPct}%`;
            }
            if (holdTimerTextRef.current) {
              holdTimerTextRef.current.textContent = `${repStep.holdRemainingSec.toFixed(1)}s`;
            }

            if (repStep.justCompletedRep) {
              lastEmittedHold = 0;
              lastHoldEmitAt = now;
              setDemoHoldRemainingSec(0);
              setDemoValidReps(repStep.nextState.completedReps);
              const toastText = '✓ นับสำเร็จ +1 ครั้ง! ค่อยๆ ลดแขนลง';
              if (repSuccessToastRef.current !== toastText) {
                repSuccessToastRef.current = toastText;
                setRepSuccessToastTh(toastText);
              }
              emitPresetState(
                repStep.nextState.completedReps >= curEx.target_reps ? 'SET_FINISHED' : 'REP_COMPLETED',
              );
            } else {
              if (
                repSuccessToastRef.current !== null &&
                now > repCelebrateUntilRef.current &&
                repPhaseRef.current !== 'COOLDOWN'
              ) {
                repSuccessToastRef.current = null;
                setRepSuccessToastTh(null);
              }
              if (
                now - lastHoldEmitAt >= 90 &&
                repStep.holdRemainingSec !== lastEmittedHold
              ) {
                lastEmittedHold = repStep.holdRemainingSec;
                lastHoldEmitAt = now;
                setDemoHoldRemainingSec(repStep.holdRemainingSec);
              }
              if (repStep.statusKind === 'WRONG_LIMB') {
                emitPresetState('WRONG_LIMB');
              } else if (repStep.statusKind === 'WRONG_PLANE') {
                emitPresetState('WRONG_PLANE');
              } else if (repStep.statusKind === 'POSTURE_COMPENSATED') {
                emitPresetState('TRUNK_LEAN');
              } else if (repStep.statusKind === 'HOLD_PHASE') {
                emitPresetState('HOLD_PHASE');
              } else if (repStep.statusKind === 'REP_COMPLETED') {
                emitPresetState('REP_COMPLETED');
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
              isMultiPerson: demoPresetRef.current === 'MULTI_PERSON' && checkIdentity,
              isImpostorOrEmergency: countingLockedRef.current || (checkIdentity && !cachedOwnerVerified),
              sideViolation: effSideViolation,
              wrongSideRaised: effWrongSideRaised,
              wrongSideBannerTh: effWrongSideBannerTh,
              wrongPlaneRaised: effWrongPlaneRaised,
              wrongPlaneBannerTh: effWrongPlaneBannerTh,
              lowVisibilityAlertTh: latestLowVisibilityAlertTh,
              postureBannerTh: effPostureBannerTh,
              repSuccessBannerTh:
                now <= repCelebrateUntilRef.current ? '✓ นับสำเร็จ +1 ครั้ง! ค่อยๆ ลดแขนลง' : null,
              overlayHint: latestOverlayHint,
              handLandmarks21:
                curEx.target_joint === 'left_shoulder'
                  ? latestHandLandmarksBySide.left
                  : curEx.target_joint === 'right_shoulder'
                    ? latestHandLandmarksBySide.right
                    : (latestHandLandmarksBySide.right ?? latestHandLandmarksBySide.left),
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
                  ownerFaceVerified: cachedOwnerVerified,
                  faceCount: cachedOwnerFaceCount,
                }
                : undefined,
            });
          } else {
            smoothedRenderKeypointsRef.current = null;
            if (holdProgressBarRef.current) holdProgressBarRef.current.style.width = '0%';
            if (holdTimerTextRef.current) holdTimerTextRef.current.textContent = `${holdTargetSec.toFixed(1)}s`;
            drawLivePoseOverlayOnCanvas(poseCanvasRef.current, [], {
              targetJoint: curEx.target_joint,
              expectedPlane: curExpectedPlane,
              exerciseMeta: curEx,
              romDeg: null,
              targetRomDeg: curEx.target_rom_deg,
              holdRatio: 0,
              trunkTiltDeg: 0,
              isMultiPerson: false,
              isImpostorOrEmergency: checkIdentity && !cachedOwnerVerified,
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
                  ownerFaceVerified: cachedOwnerVerified,
                  faceCount: cachedOwnerFaceCount,
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

  const patientRevealed = Boolean(patient?.name && (kiosk_state !== 'AUTH_FRAMING' || demoFaceMatched));

  const checklistItems: {
    ruleKey: FormAssessmentRuleKey;
    label: string;
    ok: boolean;
    ruleActive: boolean;
    warnText: string;
  }[] = [
      ...(selectedExercise.target_joint === 'left_shoulder' ||
        selectedExercise.target_joint === 'right_shoulder'
        ? [
          {
            ruleKey: 'limb' as const,
            label:
              selectedExercise.target_joint === 'left_shoulder'
                ? 'ยกเฉพาะแขนซ้ายตามท่าที่กำหนด'
                : 'ยกเฉพาะแขนขวาตามท่าที่กำหนด',
            ok: !wrongLimbAlert && !isEmergency,
            ruleActive: isFormRuleActive(formRules, 'limb'),
            warnText:
              selectedExercise.target_joint === 'left_shoulder'
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
        ]
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

  return (
    <div className="clinical-mist-bg relative min-h-screen w-full overflow-x-hidden text-[#0F4C81]">
      {/* 1. Ambient Breathing Aurora Background */}
      <div className="pointer-events-none fixed inset-0 overflow-hidden">
        <motion.div
          animate={{ x: [0, 36, -18, 0], y: [0, -24, 20, 0], scale: [1, 1.08, 0.96, 1] }}
          transition={{ repeat: Infinity, duration: 18, ease: 'easeInOut' }}
          className="absolute -left-24 -top-24 h-96 w-96 rounded-full bg-[#A7F3D0]/55 blur-3xl"
        />
        <motion.div
          animate={{ x: [0, -40, 24, 0], y: [0, 28, -16, 0], scale: [1, 0.94, 1.07, 1] }}
          transition={{ repeat: Infinity, duration: 22, ease: 'easeInOut' }}
          className="absolute -bottom-24 -right-24 h-[440px] w-[440px] rounded-full bg-[#BAE6FD]/60 blur-3xl"
        />
        <motion.div
          animate={{ opacity: [0.25, 0.45, 0.25] }}
          transition={{ repeat: Infinity, duration: 10, ease: 'easeInOut' }}
          className="absolute left-1/2 top-1/3 h-80 w-80 -translate-x-1/2 rounded-full bg-[#E0F2FE]/70 blur-3xl"
        />
      </div>

      {/* Main Container */}
      <div className="relative z-10 mx-auto flex min-h-screen max-w-[1600px] flex-col justify-between gap-4 p-3 sm:p-5">
        {/* 2. Top Status Bar (Glass Header) */}
        <header className="glass-card flex flex-wrap items-center justify-between gap-3 rounded-3xl px-4 py-3 sm:px-6">
          {/* Left: Logo + Edge Vision Connection Status */}
          <div className="flex flex-wrap items-center gap-3">
            <div className="flex items-center gap-2.5">
              <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-gradient-to-br from-[#0284C7] to-[#10B981] text-white shadow-md">
                <HeartPulse className="h-6 w-6" />
              </div>
              <div>
                <h1 className="text-lg font-extrabold tracking-tight text-[#0F4C81] sm:text-xl">
                  DRTECH<span className="text-[#0D9488]">CARES</span>
                </h1>
                <p className="text-[11px] font-semibold text-[#0284C7]">Smart Rehabilitation Kiosk</p>
              </div>
            </div>

            <div
              className={`inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs font-bold ${!isDemoMode && (connectionStatus === 'connected' || livePoseActive)
                  ? 'border-emerald-200 bg-emerald-50 text-[#0D9488]'
                  : 'border-sky-200 bg-sky-50 text-[#0284C7]'
                }`}
            >
              <span className="relative flex h-2.5 w-2.5">
                <span
                  className={`absolute inline-flex h-full w-full animate-ping rounded-full opacity-75 ${!isDemoMode && (connectionStatus === 'connected' || livePoseActive)
                      ? 'bg-[#10B981]'
                      : 'bg-[#38BDF8]'
                    }`}
                />
                <span
                  className={`relative inline-flex h-2.5 w-2.5 rounded-full ${!isDemoMode && (connectionStatus === 'connected' || livePoseActive)
                      ? 'bg-[#10B981]'
                      : 'bg-[#0284C7]'
                    }`}
                />
              </span>
              {!isDemoMode && (connectionStatus === 'connected' || livePoseActive) ? (
                <>
                  <Wifi className="h-3.5 w-3.5" />
                  <span>
                    {connectionStatus === 'connected'
                      ? 'Connected to Edge Vision (Heavy 3D)'
                      : 'Live AI Heavy 3D Active'}
                  </span>
                </>
              ) : !isDemoMode ? (
                <>
                  <WifiOff className="h-3.5 w-3.5" />
                  <span>Edge Vision & Camera Paused</span>
                </>
              ) : (
                <>
                  <WifiOff className="h-3.5 w-3.5" />
                  <span>
                    {connectionStatus === 'connected'
                      ? 'Demo Simulator (Live Paused)'
                      : 'Demo Simulator Mode'}
                  </span>
                </>
              )}
            </div>
          </div>

          {/* Center: Patient Profile Badge */}
          <AnimatePresence>
            {patientRevealed && patient && (
              <motion.div
                initial={{ opacity: 0, y: -10, scale: 0.95 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: -10 }}
                className="flex items-center gap-3 rounded-2xl border border-emerald-200/90 bg-white/90 px-4 py-1.5 shadow-sm"
              >
                <div className="flex h-8 w-8 items-center justify-center rounded-full bg-emerald-100 text-[#0D9488]">
                  <UserCheck className="h-4 w-4" />
                </div>
                <div className="leading-tight">
                  <p className="text-sm font-bold text-[#0F4C81]">{patient.name}</p>
                  <p className="tabular-mono text-[11px] font-semibold text-[#0284C7]">
                    {patient.id} • ความคืบหน้า {rep_counter.valid_reps}/{selectedExercise.target_reps} ครั้ง
                  </p>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Right: Live Camera Toggle + Privacy Mode Switch + Skeleton Overlay Switch + Demo Mode Switch + Portal Switch + Emergency Stop */}
          <div className="flex flex-wrap items-center gap-2">
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
              className={`inline-flex items-center gap-1.5 rounded-2xl border px-3 py-2 text-xs font-bold transition-all sm:text-sm ${livePoseActive
                  ? 'border-emerald-300 bg-emerald-50 text-[#0D9488]'
                  : 'border-slate-200 bg-white/80 text-slate-600 hover:bg-slate-50'
                }`}
            >
              <Camera className="h-4 w-4" />
              <span>{livePoseActive ? 'กล้อง AI: ON' : 'กล้อง AI: OFF'}</span>
            </button>

            <button
              type="button"
              onClick={handleTogglePrivacyMode}
              aria-pressed={isPrivacyMode}
              title="สลับระหว่างโหมดความเป็นส่วนตัว (พื้นหลังดำ) และโหมดแสดงภาพกล้องจริง"
              className={`inline-flex items-center gap-1.5 rounded-2xl border px-3 py-2 text-xs font-bold transition-all cursor-pointer sm:text-sm ${isPrivacyMode
                  ? 'border-slate-700 bg-slate-900 text-emerald-300 shadow-sm hover:bg-slate-800'
                  : 'border-sky-300 bg-sky-50 text-[#0F4C81] hover:bg-sky-100'
                }`}
            >
              {isPrivacyMode ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
              <span>
                {isPrivacyMode
                  ? 'โหมดความเป็นส่วนตัว (พื้นหลังดำ)'
                  : 'โหมดแสดงภาพกล้องจริง'}
              </span>
            </button>

            <button
              type="button"
              onClick={handleToggleSkeletonOverlay}
              aria-pressed={showSkeletonOverlay}
              title="สลับการแสดงหรือซ่อนเส้นโครงกระดูกและมุมข้อต่อบนหน้าจอ"
              className={`inline-flex items-center gap-1.5 rounded-2xl border px-3 py-2 text-xs font-bold transition-all cursor-pointer sm:text-sm ${showSkeletonOverlay
                  ? 'border-teal-300 bg-teal-50 text-[#0D9488] hover:bg-teal-100'
                  : 'border-slate-200 bg-white/80 text-slate-600 hover:bg-slate-50'
                }`}
            >
              <Activity className="h-4 w-4" />
              <span>
                {showSkeletonOverlay
                  ? 'แสดง/ซ่อนเส้นโครงกระดูก: ON'
                  : 'แสดง/ซ่อนเส้นโครงกระดูก: OFF'}
              </span>
            </button>

            <button
              type="button"
              onClick={handleToggleDemoMode}
              className={`inline-flex items-center gap-2 rounded-2xl border px-3.5 py-2 text-xs font-bold transition-all sm:text-sm ${isDemoMode
                  ? 'border-[#38BDF8] bg-[#E0F2FE] text-[#0F4C81]'
                  : 'border-emerald-300 bg-emerald-50 text-[#0D9488]'
                }`}
            >
              <SlidersHorizontal className="h-4 w-4" />
              <span>
                {isDemoMode
                  ? 'Demo Simulator: ON'
                  : isEdgeDirectCamera
                    ? 'Live Edge: ON'
                    : 'Demo Simulator: OFF'}
              </span>
            </button>

            {onOpenLegacyKiosk && (
              <button
                type="button"
                onClick={onOpenLegacyKiosk}
                className="rounded-2xl border border-sky-200 bg-white/80 px-3 py-2 text-xs font-bold text-[#0F4C81] hover:bg-sky-50"
              >
                โหมดสแกนเต็มระบบ
              </button>
            )}

            {onOpenStaffPortal && (
              <button
                type="button"
                onClick={onOpenStaffPortal}
                className="rounded-2xl border border-sky-200 bg-white/80 px-3 py-2 text-xs font-bold text-[#0284C7] hover:bg-sky-50"
              >
                Staff Portal
              </button>
            )}

            <button
              type="button"
              onClick={() => handleEmergencyStopClick()}
              className="inline-flex items-center gap-2 rounded-2xl bg-[#EF4444] px-4 py-2 text-xs font-extrabold text-white shadow-md transition-transform hover:bg-red-600 active:scale-95 sm:text-sm"
            >
              <AlertOctagon className="h-4 w-4" />
              <span>หยุดฉุกเฉิน</span>
            </button>
          </div>
        </header>

        {/* Camera Hardware Alert Banner (only shown if local camera access fails) */}
        {!isDemoMode && !isEdgeDirectCamera && cameraError && (
          <div
            role="alert"
            className="glass-card flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-rose-300/90 bg-rose-50/95 px-4 py-2.5 text-rose-950 shadow-sm"
          >
            <div className="flex min-w-0 flex-1 items-center gap-2.5">
              <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-rose-500/15 text-rose-700">
                <AlertTriangle className="h-4 w-4" />
              </div>
              <p className="text-xs font-extrabold text-rose-950">
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
              className="inline-flex items-center gap-1.5 rounded-xl bg-rose-600 px-3 py-1.5 text-xs font-extrabold text-white shadow-sm hover:bg-rose-500"
            >
              <RotateCcw className="h-3.5 w-3.5" />
              <span>ลองเปิดกล้องใหม่</span>
            </button>
          </div>
        )}

        {/* Completion Banner when all prescribed exercises are finished */}
        {(strokeSafetyAlert && strokeSafetyAlert.type !== 'emergency') ||
          (telemetry_analysis.safety_guard_banner_th && !isEmergency) ||
          fatigueRestSec > 0 ? (
          <div className="glass-card flex flex-wrap items-center justify-between gap-3 rounded-3xl border-2 border-amber-400 bg-amber-50/95 px-5 py-3.5 text-amber-950 shadow-md">
            <div className="flex items-center gap-3 min-w-0">
              <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl bg-amber-500 text-white shadow">
                <AlertTriangle className="h-5 w-5" />
              </div>
              <div className="min-w-0">
                <p className="text-sm font-extrabold">
                  {strokeSafetyAlert?.messageTh ??
                    telemetry_analysis.safety_guard_banner_th ??
                    'โหมดพักความปลอดภัยสำหรับผู้ป่วย Stroke (Auto Rest)'}
                </p>
                {fatigueRestSec > 0 && (
                  <p className="text-xs font-bold text-amber-800">
                    พักผ่อนคลายกล้ามเนื้อ {fatigueRestSec} วินาที — จับราวจับหรือนั่งพักก่อนฝึกต่อได้ครับ
                  </p>
                )}
              </div>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              {(strokeSafetyAlert?.suggestedRom ?? telemetry_analysis.suggested_reduced_rom_deg) ? (
                <button
                  type="button"
                  onClick={() => {
                    const nextDeg =
                      strokeSafetyAlert?.suggestedRom ??
                      telemetry_analysis.suggested_reduced_rom_deg ??
                      90;
                    setReducedRomOverrideDeg(nextDeg);
                    setStrokeSafetyAlert(null);
                  }}
                  className="rounded-xl bg-amber-600 px-3.5 py-2 text-xs font-extrabold text-white shadow-sm hover:bg-amber-500 cursor-pointer"
                >
                  ลดเป้าหมายเป็น{' '}
                  {strokeSafetyAlert?.suggestedRom ?? telemetry_analysis.suggested_reduced_rom_deg}°
                </button>
              ) : null}
              <button
                type="button"
                onClick={() => {
                  setStrokeSafetyAlert(null);
                  setFatigueRestSec(0);
                }}
                className="rounded-xl border border-amber-300 bg-white px-3.5 py-2 text-xs font-extrabold text-amber-900 hover:bg-amber-100 cursor-pointer"
              >
                {fatigueRestSec > 0 ? 'พร้อมฝึกต่อ' : 'รับทราบ'}
              </button>
            </div>
          </div>
        ) : null}

        {allExercisesCompleted && onAllExercisesCompleted && (
          <div className="glass-card flex flex-wrap items-center justify-between gap-3 rounded-3xl border-2 border-emerald-400 bg-emerald-50/95 px-5 py-3.5 shadow-lg">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-emerald-500 text-white shadow">
                <CheckCircle2 className="h-5 w-5" />
              </div>
              <div>
                <p className="text-sm font-extrabold text-emerald-950">
                  ทำท่ากายภาพบำบัดครบทั้งหมด {exerciseOptions.length} ท่าแล้ว!
                </p>
                <p className="text-xs font-semibold text-emerald-800">
                  ระบบกำลังพาเข้าสู่ขั้นตอนสุดท้าย: มินิเกมฝึกสมองและการ์ดกายภาพ...
                </p>
              </div>
            </div>
            <button
              type="button"
              onClick={onAllExercisesCompleted}
              className="inline-flex items-center gap-2 rounded-2xl bg-gradient-to-r from-emerald-600 to-teal-600 px-5 py-2.5 text-xs font-extrabold text-white shadow-md hover:from-emerald-500 hover:to-teal-500 cursor-pointer sm:text-sm"
            >
              <Gamepad2 className="h-4 w-4" />
              <span>เข้าสู่มินิเกม (ขั้นตอนสุดท้าย) ➔</span>
            </button>
          </div>
        )}

        {/* 3. Main Clinical Workspace: 16:9 Landscape (~65% Left / ~35% Right), 9:16 Portrait Responsive */}
        <main className="grid flex-1 grid-cols-1 gap-4 lg:grid-cols-12">
          {/* Left Stage (~65% width): Live AI Camera + Exercise Posture Guide Below Camera */}
          <section className="flex flex-col gap-4 lg:col-span-8 lg:self-start">
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
                    className={`relative h-full w-full overflow-hidden rounded-2xl ${isPrivacyMode ? 'bg-[#050811]' : 'bg-emerald-950'
                      }`}
                  >
                    <video
                      ref={videoRef}
                      autoPlay
                      playsInline
                      muted
                      className={`absolute inset-0 h-full w-full object-cover scale-x-[-1] transition-opacity duration-200 ${isPrivacyMode ? 'opacity-0' : 'opacity-100'
                        }`}
                    />
                    <canvas
                      ref={poseCanvasRef}
                      aria-label="Live Pose Skeleton & Angle Overlay"
                      className="pointer-events-none absolute inset-0 h-full w-full"
                    />
                    {repSuccessToastTh && !ownerRehabGuardAlert && (
                      <div className="pointer-events-none absolute inset-x-4 bottom-4 z-20 flex justify-center">
                        <div className="rounded-2xl border-2 border-emerald-300 bg-emerald-600/95 px-5 py-2.5 text-sm font-black text-white shadow-xl backdrop-blur-md sm:text-base">
                          {repSuccessToastTh}
                        </div>
                      </div>
                    )}
                    {ownerRehabGuardAlert && (
                      <div className="pointer-events-none absolute inset-x-4 bottom-4 z-20 flex justify-center">
                        <div
                          className={`rounded-2xl border-2 px-4 py-2 text-xs sm:text-sm font-black shadow-xl backdrop-blur-md ${ownerRehabGuardAlert.status === 'IMPOSTOR_FACE' ||
                              ownerRehabGuardAlert.status === 'INTRUDER_HAND'
                              ? 'border-rose-300 bg-rose-600/95 text-white'
                              : 'border-amber-200 bg-amber-500/95 text-white'
                            }`}
                        >
                          {ownerRehabGuardAlert.text}
                        </div>
                      </div>
                    )}
                    <div className="pointer-events-none absolute left-4 top-16 z-10 inline-flex items-center gap-1.5 rounded-full border border-emerald-300 bg-white/90 px-3 py-1 text-[11px] font-bold text-[#0F4C81] shadow-sm backdrop-blur-md">
                      <span className="h-2 w-2 animate-pulse rounded-full bg-emerald-400" />
                      <span>
                        {cameraReady ? '📷 กล้องตรวจจับท่าทางสด (Live AI Pose)' : '⏳ กำลังเปิดกล้อง...'}
                      </span>
                    </div>
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
                  className="absolute inset-0 z-30 flex flex-col items-center justify-center bg-rose-950/85 p-6 text-center text-white backdrop-blur-sm"
                >
                  <AlertOctagon className="h-20 w-20 text-rose-400" />
                  <h2 className="mt-4 text-3xl font-extrabold">หยุดพักฉุกเฉิน (EMERGENCY HALT)</h2>
                  <p className="mt-2 max-w-md text-lg font-extrabold text-rose-100">
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
                    className="mt-6 inline-flex items-center gap-2 rounded-2xl bg-white px-6 py-3 text-sm font-bold text-rose-700 shadow-lg hover:bg-rose-50 cursor-pointer"
                  >
                    <RotateCcw className="h-4 w-4" />
                    รีเซ็ตสถานะเพื่อฝึกต่อ
                  </button>
                </motion.div>
              )}
            </SkeletonCanvas>

            {/* Exercise-Specific Animated Posture Guide directly below Camera */}
            <ExercisePostureGuide
              exercise={selectedExercise}
              currentRomDeg={telemetry_analysis.current_rom_deg}
              targetRomDeg={effectiveTargetRomDeg}
              kioskState={kiosk_state}
            />
          </section>

          {/* Right Telemetry Panel (~35% width) */}
          <aside className="flex flex-col gap-4 lg:col-span-4">
            {/* Card 1: Exercise Prescription Header */}
            <div className="glass-card rounded-3xl p-5">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <span className="inline-flex items-center gap-1.5 rounded-full bg-sky-100/90 px-3 py-1 text-[11px] font-bold text-[#0284C7]">
                    <Sparkles className="h-3.5 w-3.5" />
                    ท่าที่ {currentExerciseIndex + 1} จาก {exerciseOptions.length} • เซต{' '}
                    {selectedExercise.target_sets ?? 1} × {selectedExercise.target_reps} ครั้ง
                  </span>
                  <h2 className="mt-2 text-xl font-extrabold text-[#0F4C81]">{selectedExercise.name_th}</h2>
                  <p className="text-xs font-semibold text-[#0284C7]">{selectedExercise.name_en}</p>
                  {(selectedExercise.target_joint === 'left_shoulder' ||
                    selectedExercise.target_joint === 'right_shoulder') && (
                      <div className="mt-2 flex flex-wrap gap-1.5">
                        <div
                          className={`inline-flex items-center gap-1.5 rounded-xl border px-3 py-1 text-xs font-extrabold ${selectedExercise.target_joint === 'left_shoulder'
                              ? 'border-orange-300 bg-orange-50 text-orange-800'
                              : 'border-emerald-300 bg-emerald-50 text-emerald-800'
                            }`}
                        >
                          <span>✋ ท่านี้ต้องยกเฉพาะ:</span>
                          <span>
                            {activeExpectedPlane === 'bilateral' || activeExpectedPlane === 'bilateral_push'
                              ? 'ทั้ง 2 ข้างพร้อมกัน (Both Arms Together)'
                              : selectedExercise.target_joint === 'left_shoulder'
                                ? 'แขนซ้าย (Left Arm Only)'
                                : 'แขนขวา (Right Arm Only)'}
                          </span>
                        </div>
                        {(activeExpectedPlane === 'flexion' || activeExpectedPlane === 'abduction') && (
                          <div className="inline-flex items-center gap-1.5 rounded-xl border border-sky-300 bg-sky-50 px-3 py-1 text-xs font-extrabold text-[#0F4C81]">
                            <span>
                              {activeExpectedPlane === 'flexion'
                                ? '⬆️ ทิศทาง: ยกตรงไปด้านหน้า • ตรวจจับมุม 3 มิติ (แกน X-Y-Z)'
                                : '↔️ ทิศทาง: กางออกด้านข้าง • ตรวจจับมุม 3 มิติ (แกน X-Y-Z)'}
                            </span>
                          </div>
                        )}
                      </div>
                    )}
                </div>

                {/* Animated Miniature Guide Illustration (synced to selected exercise) */}
                <div className="flex h-16 w-16 shrink-0 items-center justify-center rounded-2xl border border-sky-100 bg-gradient-to-br from-sky-50 to-emerald-50">
                  <ExercisePostureSvg exercise={selectedExercise} compact />
                </div>
              </div>

              <div className="mt-3 flex max-h-36 flex-wrap gap-1.5 overflow-y-auto pr-1">
                {exerciseOptions.map((opt, idx) => {
                  const active = opt.exercise_id === selectedExercise.exercise_id;
                  const done = completedIds.includes(opt.exercise_id);
                  return (
                    <button
                      key={opt.exercise_id}
                      type="button"
                      onClick={() => setSelectedExercise(opt)}
                      className={`inline-flex items-center gap-1 rounded-xl px-2.5 py-1.5 text-xs font-bold transition-colors ${active
                          ? 'bg-[#0F4C81] text-white shadow-sm'
                          : done
                            ? 'bg-emerald-100 text-emerald-900 border border-emerald-300'
                            : 'bg-sky-50 text-[#0F4C81] hover:bg-sky-100'
                        }`}
                    >
                      <span>{done ? '✓' : `${idx + 1}.`}</span>
                      <span>{opt.name_th}</span>
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Card 2: Hero ROM Angle Gauge */}
            <RomGauge
              currentRomDeg={telemetry_analysis.current_rom_deg}
              targetRomDeg={effectiveTargetRomDeg}
              completionPct={
                effectiveTargetRomDeg > 0
                  ? Math.min(100, Math.round((telemetry_analysis.current_rom_deg / effectiveTargetRomDeg) * 100))
                  : telemetry_analysis.rom_completion_pct
              }
              kioskState={kiosk_state}
              highlightColor={ui_directives.highlight_color}
              holdRemainingSec={rep_counter.hold_remaining_sec}
              holdTargetSec={selectedExercise.hold_sec}
            />

            {/* Card 3: Repetition & Hold Counter */}
            <div className="glass-card relative overflow-hidden rounded-3xl p-5">
              {/* Emerald Shockwave Ripple on Rep Completion */}
              <AnimatePresence>
                {isRepCelebration && (
                  <motion.div
                    key={`ripple-${rep_counter.valid_reps}`}
                    initial={{ scale: 0.75, opacity: 0.65 }}
                    animate={{ scale: 1.45, opacity: 0 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.85, ease: 'easeOut' }}
                    className="pointer-events-none absolute inset-0 rounded-3xl border-4 border-[#10B981]"
                  />
                )}
              </AnimatePresence>

              <div className="grid grid-cols-2 gap-3">
                {/* Valid Reps */}
                <div className="rounded-2xl border border-emerald-100 bg-gradient-to-br from-white to-emerald-50/70 p-3.5">
                  <p className="text-xs font-bold text-[#0D9488]">จำนวนครั้งที่ถูกต้อง</p>
                  <div className="mt-1 flex items-baseline gap-1.5">
                    <motion.span
                      key={rep_counter.valid_reps}
                      initial={{ y: -10, scale: 1.25, opacity: 0.4 }}
                      animate={{ y: 0, scale: 1, opacity: 1 }}
                      transition={{ type: 'spring', stiffness: 320, damping: 18 }}
                      className="tabular-mono text-3xl font-extrabold text-[#0F4C81]"
                    >
                      {rep_counter.valid_reps}
                    </motion.span>
                    <span className="tabular-mono text-sm font-bold text-[#0F4C81]/60">
                      / {selectedExercise.target_reps} ครั้ง
                    </span>
                  </div>
                </div>

                {/* Live Hold Timer */}
                <div
                  className={`rounded-2xl border p-3.5 transition-colors ${isHoldPhase
                      ? 'border-emerald-300 bg-emerald-500 text-white'
                      : 'border-sky-100 bg-gradient-to-br from-white to-sky-50/70 text-[#0F4C81]'
                    }`}
                >
                  <p className={`flex items-center gap-1 text-xs font-bold ${isHoldPhase ? 'text-emerald-50' : 'text-[#0284C7]'}`}>
                    <Timer className="h-3.5 w-3.5" />
                    ค้างท่าเป้าหมาย
                  </p>
                  <div
                    ref={holdTimerTextRef}
                    className="tabular-mono mt-1 text-3xl font-extrabold"
                  >
                    {rep_counter.hold_remaining_sec.toFixed(1)}s
                  </div>
                  {(() => {
                    const holdTargetSec = Math.max(0.5, selectedExercise.hold_sec);
                    const holdPct =
                      kiosk_state === 'REP_COMPLETED' || kiosk_state === 'SET_FINISHED'
                        ? 100
                        : isHoldPhase || rep_counter.hold_remaining_sec < holdTargetSec
                          ? Math.max(
                            0,
                            Math.min(100, ((holdTargetSec - rep_counter.hold_remaining_sec) / holdTargetSec) * 100),
                          )
                          : 0;
                    return (
                      <div
                        className={`mt-2 h-2 w-full overflow-hidden rounded-full ${isHoldPhase ? 'bg-white/30' : 'bg-sky-100'
                          }`}
                      >
                        <div
                          ref={holdProgressBarRef}
                          className={`h-full rounded-full ${isHoldPhase ? 'bg-white' : 'bg-gradient-to-r from-[#0284C7] to-[#10B981]'
                            }`}
                          style={{
                            width: `${holdPct}%`,
                            willChange: 'width',
                          }}
                        />
                      </div>
                    );
                  })()}
                </div>
              </div>

              {/* Rep Visual Dots */}
              <div className="mt-3.5 flex items-center gap-1.5">
                {Array.from({ length: selectedExercise.target_reps }).map((_, idx) => {
                  const done = idx < rep_counter.valid_reps;
                  return (
                    <motion.div
                      key={idx}
                      animate={{ scale: done ? [1, 1.15, 1] : 1 }}
                      transition={{ duration: 0.25 }}
                      className={`h-2.5 flex-1 rounded-full transition-colors ${done ? 'bg-[#10B981]' : 'bg-sky-100'
                        }`}
                    />
                  );
                })}
              </div>

              {/* Action Controls: Manual Test Rep (Debug/Demo/Staff only) / Complete Current Exercise -> Next or Final Minigame */}
              <div className="mt-3.5 flex flex-col gap-2">
                {canShowManualRepButton && (
                  <button
                    type="button"
                    onClick={() => setDemoPreset('REP_COMPLETED')}
                    disabled={rep_counter.valid_reps >= selectedExercise.target_reps}
                    className="inline-flex w-full items-center justify-center gap-1.5 rounded-xl border border-sky-200 bg-sky-50 px-3 py-2 text-xs font-bold text-[#0F4C81] hover:bg-sky-100 disabled:opacity-50 cursor-pointer"
                  >
                    <PlusCircle className="h-3.5 w-3.5" />
                    <span>+1 ครั้ง (นับครั้งสำเร็จ)</span>
                  </button>
                )}

                {nextUncompletedExercise ? (
                  <button
                    type="button"
                    onClick={() => {
                      markCurrentExerciseCompleted();
                      setSelectedExercise(nextUncompletedExercise);
                    }}
                    className="inline-flex w-full items-center justify-center gap-1.5 rounded-xl bg-gradient-to-r from-[#0284C7] to-[#10B981] px-3 py-2.5 text-xs font-extrabold text-white shadow hover:opacity-95 cursor-pointer"
                  >
                    <CheckCircle2 className="h-4 w-4" />
                    <span>ทำท่านี้ครบแล้ว • ไปท่าถัดไป ({nextUncompletedExercise.name_th}) ➔</span>
                  </button>
                ) : (
                  onAllExercisesCompleted && (
                    <button
                      type="button"
                      onClick={() => {
                        markCurrentExerciseCompleted();
                        onAllExercisesCompleted();
                      }}
                      className="inline-flex w-full items-center justify-center gap-1.5 rounded-xl bg-gradient-to-r from-emerald-600 to-teal-600 px-3 py-2.5 text-xs font-extrabold text-white shadow hover:from-emerald-500 hover:to-teal-500 cursor-pointer"
                    >
                      <Gamepad2 className="h-4 w-4" />
                      <span>ทำกายภาพครบทุกท่าแล้ว • เข้าสู่มินิเกม (ขั้นตอนสุดท้าย) ➔</span>
                    </button>
                  )
                )}
              </div>
            </div>

            {/* Card 4: Biomechanical Form Checklist */}
            <div className="glass-card rounded-3xl p-5">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <button
                  type="button"
                  onClick={() => setFormChecklistExpanded((prev) => !prev)}
                  className="flex items-center gap-2 text-left text-sm font-extrabold text-[#0F4C81] hover:text-[#0284C7] cursor-pointer"
                  aria-expanded={formChecklistExpanded}
                >
                  <Activity
                    className={`h-4 w-4 shrink-0 ${formRules.enabled ? 'text-[#0D9488]' : 'text-slate-400'}`}
                  />
                  <span>การประเมินท่าทางแบบเรียลไทม์</span>
                </button>

                <div className="flex items-center gap-1.5">
                  <button
                    type="button"
                    onClick={toggleMasterFormAssessment}
                    aria-pressed={formRules.enabled}
                    title={formRules.enabled ? 'กดเพื่อปิดระบบประเมินท่าทางแบบเรียลไทม์' : 'กดเพื่อเปิดระบบประเมินท่าทางแบบเรียลไทม์'}
                    className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] font-extrabold transition-colors cursor-pointer ${formRules.enabled
                        ? 'border-emerald-300 bg-emerald-50 text-[#0D9488] hover:bg-emerald-100'
                        : 'border-slate-300 bg-slate-100 text-slate-600 hover:bg-slate-200'
                      }`}
                  >
                    <span
                      className={`inline-flex h-3.5 w-6 items-center rounded-full p-0.5 transition-colors ${formRules.enabled ? 'bg-[#10B981] justify-end' : 'bg-slate-400 justify-start'
                        }`}
                    >
                      <span className="h-2.5 w-2.5 rounded-full bg-white shadow-sm" />
                    </span>
                    <span>{formRules.enabled ? 'เปิดประเมิน' : 'ปิดประเมิน'}</span>
                  </button>

                  <button
                    type="button"
                    onClick={() => setFormChecklistExpanded((prev) => !prev)}
                    aria-label={formChecklistExpanded ? 'ซ่อนรายการประเมินท่าทาง' : 'เปิดดูรายการประเมินท่าทาง'}
                    className="inline-flex items-center gap-1 rounded-full border border-sky-200 bg-white/80 px-2.5 py-1 text-[11px] font-bold text-[#0F4C81] hover:bg-sky-50 cursor-pointer"
                  >
                    <span>{formChecklistExpanded ? 'ซ่อน' : 'เปิดดู'}</span>
                    {formChecklistExpanded ? (
                      <ChevronUp className="h-3.5 w-3.5" />
                    ) : (
                      <ChevronDown className="h-3.5 w-3.5" />
                    )}
                  </button>
                </div>
              </div>

              <AnimatePresence initial={false}>
                {formChecklistExpanded ? (
                  <motion.div
                    key="form-checklist-expanded"
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.2, ease: 'easeOut' }}
                    className="overflow-hidden"
                  >
                    <div className="mt-2.5 mb-2.5 flex flex-wrap items-center justify-between gap-2 text-xs">
                      <span className="tabular-mono font-bold text-[#0284C7]">
                        {isFormRuleActive(formRules, 'trunk')
                          ? `เอียงลำตัว ${demoTrunkTiltDeg.toFixed(1)}°`
                          : 'เอียงลำตัว: ปิดเช็ก'}
                      </span>
                      <div className="flex items-center gap-2">
                        <span className="text-[11px] font-semibold text-slate-500">
                          กดแต่ละหัวข้อเพื่อเปิด/ปิด
                        </span>
                        {hasDisabledFormRule && (
                          <button
                            type="button"
                            onClick={resetAllFormRules}
                            className="inline-flex items-center gap-1 rounded-full border border-sky-200 bg-sky-50 px-2 py-0.5 text-[11px] font-bold text-[#0284C7] hover:bg-sky-100 cursor-pointer"
                          >
                            <RotateCcw className="h-3 w-3" />
                            <span>เปิดทั้งหมด</span>
                          </button>
                        )}
                      </div>
                    </div>

                    {!formRules.enabled && (
                      <div className="mb-2.5 flex items-center justify-between gap-2 rounded-2xl border border-slate-200 bg-slate-50 px-3.5 py-2 text-xs font-semibold text-slate-600">
                        <span>ปิดการประเมินท่าทางอยู่ • นับรอบตามมุมองศาอย่างเดียว</span>
                        <button
                          type="button"
                          onClick={toggleMasterFormAssessment}
                          className="shrink-0 rounded-full bg-emerald-600 px-2.5 py-1 text-[11px] font-bold text-white hover:bg-emerald-500 cursor-pointer"
                        >
                          กดเปิดประเมิน
                        </button>
                      </div>
                    )}

                    <div className="space-y-2">
                      {checklistItems.map((item) => (
                        <button
                          key={item.label}
                          type="button"
                          onClick={() => toggleSingleFormRule(item.ruleKey)}
                          aria-pressed={item.ruleActive}
                          className={`flex w-full items-center justify-between gap-2 rounded-2xl border px-3.5 py-2.5 text-left text-sm font-semibold transition-colors cursor-pointer ${!item.ruleActive
                              ? 'border-dashed border-slate-200 bg-slate-50/70 text-slate-400 hover:border-slate-300 hover:bg-slate-100/70'
                              : item.ok
                                ? 'border-emerald-100 bg-emerald-50/60 text-[#0F4C81] hover:bg-emerald-50'
                                : 'border-amber-300 bg-amber-50 text-amber-900 hover:bg-amber-100/70'
                            }`}
                        >
                          <span className="flex items-center gap-2">
                            {!item.ruleActive ? (
                              <Pause className="h-4 w-4 shrink-0 text-slate-400" />
                            ) : item.ok ? (
                              <CheckCircle2 className="h-4 w-4 shrink-0 text-[#10B981]" />
                            ) : (
                              <AlertTriangle className="h-4 w-4 shrink-0 text-[#F59E0B]" />
                            )}
                            <span className={!item.ruleActive ? 'line-through decoration-slate-300' : ''}>
                              {item.label}
                            </span>
                          </span>
                          <span
                            className={`shrink-0 rounded-full px-2 py-0.5 text-[11px] font-bold ${!item.ruleActive
                                ? 'bg-slate-200 text-slate-600'
                                : item.ok
                                  ? 'bg-emerald-100 text-[#0D9488]'
                                  : 'bg-amber-200 text-amber-900'
                              }`}
                          >
                            {!item.ruleActive ? 'ปิดอยู่' : item.ok ? 'ผ่าน' : item.warnText}
                          </span>
                        </button>
                      ))}
                    </div>
                  </motion.div>
                ) : (
                  <div className="mt-2 flex items-center justify-between text-xs font-semibold text-slate-500">
                    <span>
                      {formRules.enabled
                        ? `กำลังเปิดประเมิน (${checklistItems.filter((item) => item.ruleActive).length}/${checklistItems.length} หัวข้อ)`
                        : 'ปิดการประเมินท่าทางอยู่ • นับรอบตามมุมองศาอย่างเดียว'}
                    </span>
                    <button
                      type="button"
                      onClick={() => setFormChecklistExpanded(true)}
                      className="font-bold text-[#0284C7] hover:underline cursor-pointer"
                    >
                      คลิกเพื่อเปิดดูรายการ
                    </button>
                  </div>
                )}
              </AnimatePresence>
            </div>
          </aside>
        </main>

        {/* 4. Bottom Floating Dock: Thai AI Coach + Demo Simulator Drawer */}
        <footer className="space-y-3">
          <VoiceCoachBanner
            textTh={
              strokeSafetyAlert?.messageTh ??
              telemetry_analysis.safety_guard_banner_th ??
              tts_voice_th
            }
            kioskState={kiosk_state}
            highlightColor={ui_directives.highlight_color}
            muted={voiceMuted}
            onToggleMute={() => setVoiceMuted((m) => !m)}
          />

          {/* Interactive Demo Simulator Floating Control Drawer */}
          {isDemoMode && (
            <div className="glass-card rounded-3xl px-4 py-3 sm:px-6">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <button
                  type="button"
                  onClick={() => setDrawerOpen((o) => !o)}
                  className="flex items-center gap-2 text-xs font-extrabold uppercase tracking-wider text-[#0F4C81] sm:text-sm"
                >
                  <SlidersHorizontal className="h-4 w-4 text-[#0284C7]" />
                  <span>Demo Simulator Controls (พรีวิวสถานะ &amp; มุมข้อต่อสด)</span>
                  {drawerOpen ? <ChevronDown className="h-4 w-4" /> : <ChevronUp className="h-4 w-4" />}
                </button>

                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() => setAutoCycleDemo((a) => !a)}
                    className={`inline-flex items-center gap-1.5 rounded-full px-3.5 py-1.5 text-xs font-bold transition-colors ${autoCycleDemo
                        ? 'bg-[#10B981] text-white'
                        : 'bg-sky-100 text-[#0F4C81] hover:bg-sky-200'
                      }`}
                  >
                    {autoCycleDemo ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}
                    <span>{autoCycleDemo ? 'หยุดจำลองอัตโนมัติ' : 'เล่นจำลองต่อเนื่อง'}</span>
                  </button>
                  <button
                    type="button"
                    onClick={resetDemoSession}
                    className="inline-flex items-center gap-1 rounded-full border border-sky-200 bg-white px-3 py-1.5 text-xs font-bold text-[#0F4C81] hover:bg-sky-50"
                  >
                    <RotateCcw className="h-3.5 w-3.5" />
                    รีเซ็ต
                  </button>
                </div>
              </div>

              {drawerOpen && (
                <div className="mt-3 space-y-3 border-t border-sky-100 pt-3">
                  {/* State Switch Pills */}
                  <div className="flex flex-wrap gap-1.5">
                    {DEMO_PRESETS.map((p) => {
                      const active = demoPreset === p.id;
                      return (
                        <button
                          key={p.id}
                          type="button"
                          onClick={() => setDemoPreset(p.id)}
                          className={`rounded-xl px-3 py-1.5 text-xs font-bold transition-all ${active
                              ? p.id === 'IMPOSTOR'
                                ? 'bg-[#EF4444] text-white shadow-sm'
                                : p.id === 'TRUNK_LEAN' ||
                                  p.id === 'MULTI_PERSON' ||
                                  p.id === 'WRONG_LIMB' ||
                                  p.id === 'WRONG_PLANE'
                                  ? 'bg-[#F59E0B] text-white shadow-sm'
                                  : 'bg-[#0284C7] text-white shadow-sm'
                              : 'bg-white/80 text-[#0F4C81] hover:bg-sky-50'
                            }`}
                        >
                          {p.label}
                        </button>
                      );
                    })}
                  </div>

                  {/* Sliders: current_rom_deg, trunk_tilt_deg, distance_m */}
                  <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
                    <label className="flex flex-col gap-1">
                      <div className="flex justify-between text-xs font-bold">
                        <span>มุมข้อต่อ (current_rom_deg)</span>
                        <span className="tabular-mono text-[#0284C7]">{demoRomDeg.toFixed(0)}°</span>
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
                        className="accent-[#0284C7]"
                      />
                    </label>

                    <label className="flex flex-col gap-1">
                      <div className="flex justify-between text-xs font-bold">
                        <span>เอียงลำตัว (trunk_tilt_deg)</span>
                        <span
                          className={`tabular-mono ${demoTrunkTiltDeg > 10 ? 'text-[#F59E0B]' : 'text-[#0D9488]'
                            }`}
                        >
                          {demoTrunkTiltDeg.toFixed(1)}°
                        </span>
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
                        className="accent-[#F59E0B]"
                      />
                    </label>

                    <label className="flex flex-col gap-1">
                      <div className="flex justify-between text-xs font-bold">
                        <span>ระยะยืนหน้าตู้ (distance_m)</span>
                        <span className="tabular-mono text-[#0D9488]">{demoDistanceM.toFixed(2)} ม.</span>
                      </div>
                      <input
                        type="range"
                        min={1.4}
                        max={2.6}
                        step={0.05}
                        value={demoDistanceM}
                        onChange={(e) => setDemoDistanceM(Number(e.target.value))}
                        className="accent-[#10B981]"
                      />
                    </label>
                  </div>
                </div>
              )}
            </div>
          )}
        </footer>
      </div>
    </div>
  );
}

