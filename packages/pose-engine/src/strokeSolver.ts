import { POSE_INDEX, computeJointAngle3D } from './armJointProcessor';
import type { PoseLandmark } from './types';

export const HAND21_INDEX = {
  WRIST: 0,
  THUMB_CMC: 2,
  THUMB_TIP: 4,
  INDEX_MCP: 5,
  INDEX_TIP: 8,
  MIDDLE_MCP: 9,
  MIDDLE_TIP: 12,
  RING_TIP: 16,
  PINKY_MCP: 17,
  PINKY_TIP: 20,
} as const;

export const HAND21_CONNECTIONS: readonly [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
  [5, 9], [9, 13], [13, 17],
];

export type StrokeJointMode =
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

export interface StrokeAnatomyBaseline {
  shoulderWidth: number;
  upperArmLen: number;
  forearmLen: number;
  restingShoulderDeltaY: number;
  restingSpineAngleDeg: number;
  restingMidShoulderY: number;
  restingEarShoulderGapRatio: number;
  sampleCount: number;
}

export interface StrokeTargetOverlayHint {
  pivotNorm: { x: number; y: number };
  targetNorm: { x: number; y: number };
  currentNorm?: { x: number; y: number };
  inTargetZone: boolean;
  labelTh: string;
}

export type StrokeSideViolationCode =
  | 'NONE'
  | 'RIGHT_ARM_RAISED_IN_LEFT_MODE'
  | 'LEFT_ARM_RAISED_IN_RIGHT_MODE'
  | 'SINGLE_ARM_IN_BOTH_MODE';

export type StrokeRepPhase = 'WAIT_RAISE' | 'HOLDING' | 'COOLDOWN';

export interface StrokeHoldRepState {
  phase: StrokeRepPhase;
  holdElapsedSec: number;
  completedReps: number;
  cooldownStartedAtMs: number;
  repCelebrateUntilMs: number;
}

export interface StrokeHoldRepStepInput {
  state: StrokeHoldRepState;
  activeRomDeg: number | null;
  targetRomDeg: number;
  targetHoldSec: number;
  targetReps: number;
  dtSec: number;
  nowMs: number;
  isKneeExercise?: boolean;
  sideViolation: StrokeSideViolationCode;
  wrongSideBannerTh: string | null;
  wrongPlaneRaised: boolean;
  wrongPlaneBannerTh: string | null;
  isCompensatedNow: boolean;
  postureBannerTh: string | null;
  blockCounting?: boolean;
}

export interface StrokeHoldRepStepResult {
  nextState: StrokeHoldRepState;
  holdProgressPct: number;
  holdRemainingSec: number;
  justCompletedRep: boolean;
  bannerTh: string | null;
  statusKind:
    | 'IDLE'
    | 'ACTIVE_REP'
    | 'HOLD_PHASE'
    | 'REP_COMPLETED'
    | 'SET_FINISHED'
    | 'WRONG_LIMB'
    | 'WRONG_PLANE'
    | 'POSTURE_COMPENSATED';
}

export interface StrokePoseSolveInput {
  landmarks: readonly PoseLandmark[];
  worldLandmarks?: readonly PoseLandmark[];
  handLandmarksBySide?: {
    left?: readonly PoseLandmark[];
    right?: readonly PoseLandmark[];
  };
  targetJoint: 'left_shoulder' | 'right_shoulder' | 'spine' | 'left_knee' | 'right_knee' | 'none';
  exerciseSide?: 'LEFT' | 'RIGHT' | 'BOTH';
  expectedPlane?: string;
  targetRomDeg?: number;
  jointMode: StrokeJointMode;
  baseline?: StrokeAnatomyBaseline;
}

export interface StrokePoseSolveResult {
  romDeg: number | null;
  rawPrimaryRomDeg: number | null;
  leftArmRom: number | null;
  rightArmRom: number | null;
  oppositeRomDeg: number | null;
  trunkTiltDeg: number;
  shoulderTiltDeg: number;
  shoulderHikeRatio: number;
  neckDropDeg: number;
  elbowAngleDeg: number | null;
  sideViolation: StrokeSideViolationCode;
  sideViolationBannerTh: string | null;
  wrongSideRaised: boolean;
  wrongSideBannerTh: string | null;
  wrongPlaneRaised: boolean;
  wrongPlaneBannerTh: string | null;
  postureCompensated: boolean;
  postureBannerTh: string | null;
  lowVisibilityAlertTh: string | null;
  xyzVector: { x: number; y: number; z: number } | null;
  headY: number | null;
  shoulderMidY: number | null;
  compensations: string[];
  isFormValid: boolean;
  overlayHint: StrokeTargetOverlayHint | null;
}

function clamp(v: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, v));
}

function dist2D(a: PoseLandmark, b: PoseLandmark): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

/** Computes signed/unsigned 2D torso angle relative to vertical axis. */
function computeSignedSpineAngleDeg(landmarks: readonly PoseLandmark[]): number {
  const lSh = landmarks[POSE_INDEX.LEFT_SHOULDER];
  const rSh = landmarks[POSE_INDEX.RIGHT_SHOULDER];
  const lHip = landmarks[POSE_INDEX.LEFT_HIP];
  const rHip = landmarks[POSE_INDEX.RIGHT_HIP];
  if (
    lSh &&
    rSh &&
    lHip &&
    rHip &&
    (lHip.visibility ?? 0) >= 0.2 &&
    (rHip.visibility ?? 0) >= 0.2
  ) {
    const midShX = (lSh.x + rSh.x) / 2;
    const midShY = (lSh.y + rSh.y) / 2;
    const midHipX = (lHip.x + rHip.x) / 2;
    const midHipY = (lHip.y + rHip.y) / 2;
    return (Math.atan2(midShX - midHipX, Math.max(0.05, midHipY - midShY)) * 180) / Math.PI;
  }
  if (lSh && rSh && (lSh.visibility ?? 0) >= 0.35 && (rSh.visibility ?? 0) >= 0.35) {
    return (
      ((Math.atan2(rSh.y - lSh.y, Math.max(0.11, Math.abs(lSh.x - rSh.x))) * 180) / Math.PI) *
      0.55
    );
  }
  return 0;
}

/** Builds an immediate stateless anatomical baseline fallback from a single pose frame. */
export function computeDefaultAnatomyBaseline(
  landmarks: readonly PoseLandmark[],
): StrokeAnatomyBaseline {
  const lSh = landmarks[POSE_INDEX.LEFT_SHOULDER];
  const rSh = landmarks[POSE_INDEX.RIGHT_SHOULDER];
  const lEl = landmarks[POSE_INDEX.LEFT_ELBOW];
  const rEl = landmarks[POSE_INDEX.RIGHT_ELBOW];
  const lWr = landmarks[POSE_INDEX.LEFT_WRIST];
  const rWr = landmarks[POSE_INDEX.RIGHT_WRIST];
  const lEar = landmarks[POSE_INDEX.LEFT_EAR];
  const rEar = landmarks[POSE_INDEX.RIGHT_EAR];

  const shoulderWidth =
    lSh && rSh ? Math.max(0.10, Math.abs(lSh.x - rSh.x)) : 0.18;
  const measuredUpperL = lSh && lEl ? dist2D(lSh, lEl) : 0;
  const measuredUpperR = rSh && rEl ? dist2D(rSh, rEl) : 0;
  const measuredForeL = lEl && lWr ? dist2D(lEl, lWr) : 0;
  const measuredForeR = rEl && rWr ? dist2D(rEl, rWr) : 0;

  const upperArmLen = Math.max(shoulderWidth * 0.82, measuredUpperL, measuredUpperR);
  const forearmLen = Math.max(shoulderWidth * 0.72, measuredForeL, measuredForeR);
  const restingShoulderDeltaY = lSh && rSh ? lSh.y - rSh.y : 0;
  const restingSpineAngleDeg = computeSignedSpineAngleDeg(landmarks);
  const restingMidShoulderY = lSh && rSh ? (lSh.y + rSh.y) / 2 : 0.42;

  const lGap = lSh && lEar ? Math.max(0.02, lSh.y - lEar.y) : shoulderWidth * 0.55;
  const rGap = rSh && rEar ? Math.max(0.02, rSh.y - rEar.y) : shoulderWidth * 0.55;
  const restingEarShoulderGapRatio = ((lGap + rGap) / 2) / Math.max(0.08, shoulderWidth);

  return {
    shoulderWidth,
    upperArmLen,
    forearmLen,
    restingShoulderDeltaY,
    restingSpineAngleDeg,
    restingMidShoulderY,
    restingEarShoulderGapRatio,
    sampleCount: 1,
  };
}

/**
 * Automatic Bone-Length Baseline Calibrator capturing patient-specific anatomy
 * across initial neutral frames without forcing a fixed standing pause.
 */
export class AutoBoneBaselineCalibrator {
  private baseline: StrokeAnatomyBaseline | null = null;

  reset(): void {
    this.baseline = null;
  }

  getBaseline(): StrokeAnatomyBaseline | null {
    return this.baseline ? { ...this.baseline } : null;
  }

  update(landmarks: readonly PoseLandmark[]): StrokeAnatomyBaseline {
    const snap = computeDefaultAnatomyBaseline(landmarks);
    if (!this.baseline) {
      this.baseline = snap;
      return { ...this.baseline };
    }

    const lSh = landmarks[POSE_INDEX.LEFT_SHOULDER];
    const rSh = landmarks[POSE_INDEX.RIGHT_SHOULDER];
    const lWr = landmarks[POSE_INDEX.LEFT_WRIST];
    const rWr = landmarks[POSE_INDEX.RIGHT_WRIST];
    const alpha = this.baseline.sampleCount < 20 ? 0.18 : 0.04;

    const shouldersLevel = Boolean(
      lSh && rSh && Math.abs(lSh.y - rSh.y) < snap.shoulderWidth * 0.12,
    );
    const armsNearRest = Boolean(
      lSh &&
        rSh &&
        (!lWr || lWr.y > lSh.y + this.baseline.upperArmLen * 0.35) &&
        (!rWr || rWr.y > rSh.y + this.baseline.upperArmLen * 0.35),
    );

    if (shouldersLevel) {
      this.baseline.shoulderWidth =
        this.baseline.shoulderWidth * (1 - alpha) + snap.shoulderWidth * alpha;
      // Preserve true anatomical maximum while arm hangs calmly without foreshortening shrinkage
      this.baseline.upperArmLen = Math.max(
        this.baseline.shoulderWidth * 0.82,
        armsNearRest
          ? this.baseline.upperArmLen * (1 - alpha) + snap.upperArmLen * alpha
          : this.baseline.upperArmLen,
      );
      this.baseline.forearmLen = Math.max(
        this.baseline.shoulderWidth * 0.72,
        armsNearRest
          ? this.baseline.forearmLen * (1 - alpha) + snap.forearmLen * alpha
          : this.baseline.forearmLen,
      );
      if (this.baseline.sampleCount < 25 && armsNearRest) {
        this.baseline.restingShoulderDeltaY =
          this.baseline.restingShoulderDeltaY * (1 - alpha) + snap.restingShoulderDeltaY * alpha;
        this.baseline.restingSpineAngleDeg =
          this.baseline.restingSpineAngleDeg * (1 - alpha) + snap.restingSpineAngleDeg * alpha;
        this.baseline.restingMidShoulderY =
          this.baseline.restingMidShoulderY * (1 - alpha) + snap.restingMidShoulderY * alpha;
        this.baseline.restingEarShoulderGapRatio =
          this.baseline.restingEarShoulderGapRatio * (1 - alpha) +
          snap.restingEarShoulderGapRatio * alpha;
      }
    }
    this.baseline.sampleCount += 1;
    return { ...this.baseline };
  }
}

/** Checks whether 21-point hand landmarks are dense and spread enough to compute distal ROM. */
function hasUsableHand21(hand: readonly PoseLandmark[] | undefined): hand is readonly PoseLandmark[] {
  if (!hand || hand.length < 21) return false;
  const wrist = hand[HAND21_INDEX.WRIST];
  const indexTip = hand[HAND21_INDEX.INDEX_TIP];
  const pinkyMcp = hand[HAND21_INDEX.PINKY_MCP];
  if (!wrist || !indexTip || !pinkyMcp) return false;
  const minVis = Math.min(wrist.visibility ?? 1, indexTip.visibility ?? 1, pinkyMcp.visibility ?? 1);
  if (minVis < 0.25) return false;
  const palmSpan = Math.max(dist2D(wrist, pinkyMcp), dist2D(wrist, indexTip));
  return palmSpan >= 0.02;
}

/**
 * Computes clinical elevation ROM assisted by normalized wrist/elbow height within torso bounds
 * so stroke patients reach 90%-105% of targetRomGoal between mid-chest and shoulder level.
 */
function computeAssistedElevationRom(params: {
  rawAngleDeg: number | null;
  shoulder: PoseLandmark | undefined;
  elbow: PoseLandmark | undefined;
  wrist: PoseLandmark | undefined;
  hipY: number;
  torsoSpan: number;
  shoulderSpanX: number;
  targetGoalDeg: number;
  enabled: boolean;
}): number | null {
  const { rawAngleDeg, shoulder, elbow, wrist, hipY, torsoSpan, shoulderSpanX, targetGoalDeg, enabled } = params;
  if (!enabled || !shoulder || !wrist || (wrist.visibility ?? 0) < 0.2) {
    return rawAngleDeg;
  }
  const safeTorso = Math.max(torsoSpan, shoulderSpanX * 1.15, 0.14);
  const wristLiftRatio = clamp((hipY - wrist.y) / safeTorso, 0, 1.45);
  const elbowVis = elbow && (elbow.visibility ?? 0) >= 0.2;
  const elbowLiftRatio = elbowVis ? clamp(((hipY - elbow.y) / safeTorso) * 1.12, 0, 1.45) : 0;
  const effectiveLift = Math.max(wristLiftRatio, wristLiftRatio * 0.58 + elbowLiftRatio * 0.42);
  const forwardZGain = shoulder.z != null && wrist.z != null
    ? clamp((shoulder.z - wrist.z) / Math.max(0.12, shoulderSpanX), 0, 1.2)
    : 0;

  const base = rawAngleDeg ?? 0;
  // Preserve low resting goniometer readings (< 22 deg / below chest zone) without artificial floor
  if (effectiveLift < 0.42 || (base < 22 && effectiveLift < 0.48)) {
    return rawAngleDeg;
  }

  // Map chest-to-shoulder height cleanly to ~80%-105% of targetGoalDeg
  const chestToShoulderT = clamp((effectiveLift - 0.50) / (0.88 - 0.50), 0, 1);
  const progressRatio = 0.76 + chestToShoulderT * 0.27;
  const zBoostRatio = effectiveLift >= 0.48 && chestToShoulderT < 0.9 ? forwardZGain * 0.08 : 0;
  const assistedDeg = clamp(
    (progressRatio + zBoostRatio) * targetGoalDeg,
    0,
    targetGoalDeg * 1.05,
  );
  return Number(clamp(Math.max(base, assistedDeg), 0, 180).toFixed(1));
}

/**
 * Fallback ROM calculation using MediaPipe 33-pt Pose wrist/elbow/hand-anchor landmarks
 * when the 21-point HandLandmarker is inactive, too distant, or occluded.
 */
function computePoseDistalFallbackRom(params: {
  landmarks: readonly PoseLandmark[];
  side: 'left' | 'right';
  mode: 'forearm_turn' | 'wrist_finger' | 'thumb_pinch';
  shoulderSpanX: number;
  forearmLen: number;
  targetGoalDeg: number;
}): number | null {
  const { landmarks, side, mode, shoulderSpanX, forearmLen, targetGoalDeg } = params;
  const sh = landmarks[side === 'left' ? POSE_INDEX.LEFT_SHOULDER : POSE_INDEX.RIGHT_SHOULDER];
  const el = landmarks[side === 'left' ? POSE_INDEX.LEFT_ELBOW : POSE_INDEX.RIGHT_ELBOW];
  const wr = landmarks[side === 'left' ? POSE_INDEX.LEFT_WRIST : POSE_INDEX.RIGHT_WRIST];
  const pinky = landmarks[side === 'left' ? POSE_INDEX.LEFT_PINKY : POSE_INDEX.RIGHT_PINKY];
  const index = landmarks[side === 'left' ? POSE_INDEX.LEFT_INDEX : POSE_INDEX.RIGHT_INDEX];
  const thumb = landmarks[side === 'left' ? POSE_INDEX.LEFT_THUMB : POSE_INDEX.RIGHT_THUMB];

  if (!el || !wr || (wr.visibility ?? 0) < 0.2) return null;
  const safeFore = Math.max(0.08, forearmLen, shoulderSpanX * 0.65);

  if (mode === 'thumb_pinch') {
    if (thumb && index && (thumb.visibility ?? 0) >= 0.15 && (index.visibility ?? 0) >= 0.15) {
      const pinchSpan = dist2D(thumb, index) / Math.max(0.04, shoulderSpanX * 0.32);
      const liftBonus = el.y > wr.y ? clamp((el.y - wr.y) / safeFore, 0, 0.5) * 20 : 0;
      const pinchClosePct = clamp(1.35 - pinchSpan, 0, 1.05);
      return Number(clamp(pinchClosePct * targetGoalDeg * 1.05 + liftBonus, 0, Math.max(targetGoalDeg * 1.15, 75)).toFixed(1));
    }
  }

  if (mode === 'forearm_turn' && index && pinky) {
    const palmTilt = Math.abs(index.x - pinky.x) / Math.max(0.025, shoulderSpanX * 0.22);
    const wristLift = clamp((el.y - wr.y) / safeFore, -0.2, 1.1);
    const turnScore = clamp(palmTilt * 0.65 + Math.max(0, wristLift) * 0.55, 0, 1.1);
    return Number(clamp(turnScore * targetGoalDeg * 1.02, 0, Math.max(targetGoalDeg * 1.1, 85)).toFixed(1));
  }

  // Generic visible forearm/wrist readiness & distal motion ROM fallback
  const handAnchor = index ?? thumb ?? pinky;
  const wristUp = clamp((el.y + safeFore * 0.25 - wr.y) / (safeFore * 0.95), 0, 1.05);
  const shElevate = sh ? clamp((sh.y + safeFore * 0.9 - wr.y) / (safeFore * 1.15), 0, 1.05) : 0;
  const wristFlex = handAnchor
    ? clamp(Math.abs(handAnchor.y - wr.y) / Math.max(0.03, shoulderSpanX * 0.16), 0, 1.1)
    : 0;
  const combined = Math.max(wristUp * 0.72 + wristFlex * 0.38, shElevate * 0.95);
  return Number(clamp(combined * targetGoalDeg * 1.02, 0, Math.max(targetGoalDeg * 1.1, 90)).toFixed(1));
}

/**
 * Computes clinical 3D joint angles and Stroke compensation detections across all
 * 26 standardized Stroke ADL & MOPH Elderly upper-body exercises.
 */
export function solveStrokeClinicalJointBiomechanics(
  input: StrokePoseSolveInput,
): StrokePoseSolveResult {
  const {
    landmarks,
    worldLandmarks,
    handLandmarksBySide,
    targetJoint,
    exerciseSide,
    expectedPlane = 'any',
    jointMode,
  } = input;
  const baseline = input.baseline ?? computeDefaultAnatomyBaseline(landmarks);
  const UPPER_MIN_VIS = 0.02;

  const lSh = landmarks[POSE_INDEX.LEFT_SHOULDER];
  const rSh = landmarks[POSE_INDEX.RIGHT_SHOULDER];
  const nose = landmarks[POSE_INDEX.NOSE];
  const lEar = landmarks[POSE_INDEX.LEFT_EAR];
  const rEar = landmarks[POSE_INDEX.RIGHT_EAR];
  const lEye = landmarks[POSE_INDEX.LEFT_EYE];
  const rEye = landmarks[POSE_INDEX.RIGHT_EYE];

  const shoulderSpanX = baseline.shoulderWidth;
  const armLen = baseline.upperArmLen + baseline.forearmLen;
  const targetRomGoal = input.targetRomDeg ?? 110;

  const liveSpineDeg = computeSignedSpineAngleDeg(landmarks);
  const trunkTiltDeg = Math.abs(liveSpineDeg);
  const spineDeviationDeg = Math.abs(liveSpineDeg - baseline.restingSpineAngleDeg);

  const shoulderTiltDeg =
    lSh && rSh && (lSh.visibility ?? 0) >= 0.35 && (rSh.visibility ?? 0) >= 0.35
      ? Number(
          Math.abs(
            (Math.atan2(rSh.y - lSh.y, Math.max(0.10, Math.abs(lSh.x - rSh.x))) * 180) / Math.PI,
          ).toFixed(1),
        )
      : 0;

  const hipsVisible = Boolean(
    lSh &&
      rSh &&
      landmarks[POSE_INDEX.LEFT_HIP] &&
      landmarks[POSE_INDEX.RIGHT_HIP] &&
      (landmarks[POSE_INDEX.LEFT_HIP].visibility ?? 0) >= 0.2 &&
      (landmarks[POSE_INDEX.RIGHT_HIP].visibility ?? 0) >= 0.2 &&
      (landmarks[POSE_INDEX.LEFT_HIP].y + landmarks[POSE_INDEX.RIGHT_HIP].y) / 2 <= 0.94 &&
      (landmarks[POSE_INDEX.LEFT_HIP].y + landmarks[POSE_INDEX.RIGHT_HIP].y) / 2 - (lSh.y + rSh.y) / 2 >= 0.15,
  );
  const torsoSpanY =
    hipsVisible && lSh && rSh
      ? Math.abs(
          (landmarks[POSE_INDEX.LEFT_HIP].y + landmarks[POSE_INDEX.RIGHT_HIP].y) / 2 -
            (lSh.y + rSh.y) / 2,
        )
      : 0;
  const midShY = lSh && rSh ? (lSh.y + rSh.y) / 2 : (lSh?.y ?? rSh?.y ?? 0.35);
  const midHipY = hipsVisible
    ? (landmarks[POSE_INDEX.LEFT_HIP].y + landmarks[POSE_INDEX.RIGHT_HIP].y) / 2
    : midShY + Math.max(torsoSpanY, shoulderSpanX * 1.48);
  const effTorsoSpan = hipsVisible ? Math.max(torsoSpanY, shoulderSpanX * 1.15) : shoulderSpanX * 1.48;

  const hasExplicitTargetRom = typeof input.targetRomDeg === 'number' && Number.isFinite(input.targetRomDeg) && input.targetRomDeg > 0;
  const assistElevationEnabled =
    hasExplicitTargetRom ||
    jointMode === 'scap_protract' ||
    jointMode === 'bobath_push' ||
    jointMode === 'bilateral_raise';

  const enhanceSideElevationRom = (side: 'left' | 'right', rawAngleDeg: number | null): number | null => {
    const shPt = landmarks[side === 'left' ? POSE_INDEX.LEFT_SHOULDER : POSE_INDEX.RIGHT_SHOULDER];
    const elPt = landmarks[side === 'left' ? POSE_INDEX.LEFT_ELBOW : POSE_INDEX.RIGHT_ELBOW];
    const wrPt = landmarks[side === 'left' ? POSE_INDEX.LEFT_WRIST : POSE_INDEX.RIGHT_WRIST];
    return computeAssistedElevationRom({
      rawAngleDeg,
      shoulder: shPt,
      elbow: elPt,
      wrist: wrPt,
      hipY: midHipY,
      torsoSpan: effTorsoSpan,
      shoulderSpanX,
      targetGoalDeg: targetRomGoal,
      enabled: assistElevationEnabled,
    });
  };

  const computeForwardFlexionWithForeshortening = (
    side: 'left' | 'right',
  ): {
    angleDeg: number | null;
    xyzVector: { x: number; y: number; z: number } | null;
    elbowAngleDeg: number | null;
  } => {
    const shIdx = side === 'left' ? POSE_INDEX.LEFT_SHOULDER : POSE_INDEX.RIGHT_SHOULDER;
    const elIdx = side === 'left' ? POSE_INDEX.LEFT_ELBOW : POSE_INDEX.RIGHT_ELBOW;
    const wrIdx = side === 'left' ? POSE_INDEX.LEFT_WRIST : POSE_INDEX.RIGHT_WRIST;
    const sh = landmarks[shIdx];
    const el = landmarks[elIdx];
    const wr = landmarks[wrIdx];
    if (!sh || (sh.visibility ?? 0) < UPPER_MIN_VIS) {
      return { angleDeg: null, xyzVector: null, elbowAngleDeg: null };
    }

    const wristUpRef = hipsVisible
      ? Math.max(torsoSpanY * 1.02, shoulderSpanX * 1.88)
      : shoulderSpanX * 2.72;

    const measureSegment = (
      pt: PoseLandmark | undefined,
      isWrist: boolean,
    ): { angleDeg: number; xyzVector: { x: number; y: number; z: number } } | null => {
      if (!pt || (pt.visibility ?? 0) < UPPER_MIN_VIS) return null;
      const dx = (pt.x - sh.x) * 1.22;
      const dy = pt.y - sh.y;
      const dzSensor = ((pt.z ?? 0) - (sh.z ?? 0)) * 1.22;

      const refLen = isWrist ? wristUpRef : wristUpRef * 0.54;
      const len2D = Math.hypot(dx, dy);
      const dzSphere = Math.sqrt(Math.max(0, refLen * refLen - dx * dx - dy * dy));
      const isPlaneRestricted = expectedPlane === 'flexion' || expectedPlane === 'abduction';
      const sagittalWeight =
        isPlaneRestricted || Math.abs(dx) <= shoulderSpanX * 0.45
          ? 1
          : Math.max(0, 1 - (Math.abs(dx) - shoulderSpanX * 0.45) / (shoulderSpanX * 0.45));

      const projRatio = clamp(
        dist2D(sh, el && (el.visibility ?? 0) >= UPPER_MIN_VIS ? el : pt) /
          Math.max(0.08, baseline.upperArmLen),
        0,
        1,
      );
      const boneForeshorten =
        expectedPlane !== 'abduction' && dy <= shoulderSpanX * 0.45
          ? clamp((shoulderSpanX * 0.45 - dy) / (shoulderSpanX * 0.25), 0, 1) *
            clamp((0.82 - projRatio) / 0.35, 0, 1)
          : 0;
      const foreshortenedForward =
        expectedPlane !== 'abduction' && dy <= shoulderSpanX * 0.38
          ? Math.max(
              boneForeshorten,
              clamp((shoulderSpanX * 0.38 - dy) / (shoulderSpanX * 0.25), 0, 1) *
                clamp((refLen * 0.68 - len2D) / (refLen * 0.26), 0, 1),
            )
          : 0;
      const downForwardWeight =
        dy > 0
          ? expectedPlane === 'abduction'
            ? 0
            : Math.max(clamp(Math.abs(dzSensor) / 0.14, 0, 1), foreshortenedForward)
          : 1;
      const absDz = Math.max(
        Math.abs(dzSensor),
        dzSphere * sagittalWeight * downForwardWeight,
      );
      const dz = (dzSensor > 1e-4 ? 1 : -1) * absDz;
      const horizMagXZ = Math.hypot(dx, absDz);
      const mag3D = Math.hypot(dx, dy, dz);
      const angle3D = mag3D >= 0.02 ? (Math.atan2(horizMagXZ, dy) * 180) / Math.PI : null;
      const xyz2DSphere =
        mag3D >= 0.02
          ? {
              x: Number((dx / mag3D).toFixed(3)),
              y: Number((dy / mag3D).toFixed(3)),
              z: Number((dz / mag3D).toFixed(3)),
            }
          : null;

      if (worldLandmarks) {
        const wSh = worldLandmarks[shIdx];
        const wPt = worldLandmarks[isWrist ? wrIdx : elIdx];
        if (wSh && wPt) {
          const wDx = wPt.x - wSh.x;
          const wDy = wPt.y - wSh.y;
          const wDz = (wPt.z ?? 0) - (wSh.z ?? 0);
          const wLen3D = Math.hypot(wDx, wDy, wDz);
          if (wLen3D >= 0.08) {
            const worldAngleDeg = (Math.atan2(Math.hypot(wDx, wDz), wDy) * 180) / Math.PI;
            const stronglyForeshortenedFront =
              expectedPlane !== 'abduction' &&
              angle3D !== null &&
              angle3D >= 65 &&
              dy <= shoulderSpanX * 0.45 &&
              Math.abs(dx) <= shoulderSpanX * 0.85 &&
              len2D <= refLen * 0.68;
            if (stronglyForeshortenedFront && worldAngleDeg < angle3D - 18 && xyz2DSphere) {
              return {
                angleDeg: Number((0.15 * worldAngleDeg + 0.85 * angle3D).toFixed(1)),
                xyzVector: xyz2DSphere,
              };
            }
            return {
              angleDeg: worldAngleDeg,
              xyzVector: {
                x: Number((wDx / wLen3D).toFixed(3)),
                y: Number((wDy / wLen3D).toFixed(3)),
                z: Number((wDz / wLen3D).toFixed(3)),
              },
            };
          }
        }
      }
      if (angle3D === null || !xyz2DSphere) return null;
      return { angleDeg: angle3D, xyzVector: xyz2DSphere };
    };

    const wristMeasure = measureSegment(wr, true);
    const elbowMeasure = measureSegment(el, false);
    let angleDeg: number | null = null;
    if (wristMeasure && elbowMeasure) {
      angleDeg = 0.75 * wristMeasure.angleDeg + 0.25 * elbowMeasure.angleDeg;
    } else if (wristMeasure) {
      angleDeg = wristMeasure.angleDeg;
    } else if (elbowMeasure) {
      angleDeg = elbowMeasure.angleDeg;
    }

    let elbowAngleDeg: number | null = null;
    if (
      el &&
      wr &&
      (sh.visibility ?? 0) >= 0.35 &&
      (el.visibility ?? 0) >= 0.35 &&
      (wr.visibility ?? 0) >= 0.35
    ) {
      const wSh = worldLandmarks?.[shIdx];
      const wEl = worldLandmarks?.[elIdx];
      const wWr = worldLandmarks?.[wrIdx];
      const validW = Boolean(
        wSh &&
          wEl &&
          wWr &&
          Math.hypot(wSh.x - wEl.x, wSh.y - wEl.y, (wSh.z ?? 0) - (wEl.z ?? 0)) >= 0.08 &&
          Math.hypot(wWr.x - wEl.x, wWr.y - wEl.y, (wWr.z ?? 0) - (wEl.z ?? 0)) >= 0.08,
      );
      elbowAngleDeg =
        validW && wSh && wEl && wWr ? computeJointAngle3D(wSh, wEl, wWr) : computeJointAngle3D(sh, el, wr);
    }

    return {
      angleDeg: angleDeg !== null ? Number(clamp(angleDeg, 0, 180).toFixed(1)) : null,
      xyzVector: wristMeasure?.xyzVector ?? elbowMeasure?.xyzVector ?? null,
      elbowAngleDeg,
    };
  };

  const computeCoronalAbduction = (
    side: 'left' | 'right',
  ): {
    angleDeg: number | null;
    xyzVector: { x: number; y: number; z: number } | null;
    elbowAngleDeg: number | null;
    lateralDxRatio: number;
  } => {
    const res = computeForwardFlexionWithForeshortening(side);
    const shIdx = side === 'left' ? POSE_INDEX.LEFT_SHOULDER : POSE_INDEX.RIGHT_SHOULDER;
    const wrIdx = side === 'left' ? POSE_INDEX.LEFT_WRIST : POSE_INDEX.RIGHT_WRIST;
    const elIdx = side === 'left' ? POSE_INDEX.LEFT_ELBOW : POSE_INDEX.RIGHT_ELBOW;
    const sh = landmarks[shIdx];
    const distal =
      landmarks[wrIdx] && (landmarks[wrIdx].visibility ?? 0) >= UPPER_MIN_VIS
        ? landmarks[wrIdx]
        : landmarks[elIdx];
    if (!sh || !distal) {
      return { ...res, lateralDxRatio: 0 };
    }
    const lateralDxRatio = Math.abs(distal.x - sh.x) / Math.max(0.08, shoulderSpanX);
    const spineRad = (liveSpineDeg * Math.PI) / 180;
    const spineDownX = -Math.sin(spineRad);
    const spineDownY = Math.cos(spineRad);
    const vx = distal.x - sh.x;
    const vy = distal.y - sh.y;
    const vn = Math.hypot(vx, vy);
    const coronalDeg =
      vn >= 0.02
        ? (Math.acos(clamp((vx * spineDownX + vy * spineDownY) / vn, -1, 1)) * 180) / Math.PI
        : 0;
    const blended = res.angleDeg !== null ? 0.55 * coronalDeg + 0.45 * res.angleDeg : coronalDeg;
    return {
      angleDeg: Number(clamp(blended, 0, 180).toFixed(1)),
      xyzVector: res.xyzVector,
      elbowAngleDeg: res.elbowAngleDeg,
      lateralDxRatio,
    };
  };

  const computeElbowFlexionRom = (
    side: 'left' | 'right',
  ): { flexionDeg: number | null; xyzVector: { x: number; y: number; z: number } | null } => {
    const shIdx = side === 'left' ? POSE_INDEX.LEFT_SHOULDER : POSE_INDEX.RIGHT_SHOULDER;
    const elIdx = side === 'left' ? POSE_INDEX.LEFT_ELBOW : POSE_INDEX.RIGHT_ELBOW;
    const wrIdx = side === 'left' ? POSE_INDEX.LEFT_WRIST : POSE_INDEX.RIGHT_WRIST;
    const sh = landmarks[shIdx];
    const el = landmarks[elIdx];
    const wr = landmarks[wrIdx];
    if (
      !sh ||
      !el ||
      !wr ||
      (sh.visibility ?? 0) < UPPER_MIN_VIS ||
      (el.visibility ?? 0) < UPPER_MIN_VIS ||
      (wr.visibility ?? 0) < UPPER_MIN_VIS
    ) {
      return { flexionDeg: null, xyzVector: null };
    }

    const wSh = worldLandmarks?.[shIdx];
    const wEl = worldLandmarks?.[elIdx];
    const wWr = worldLandmarks?.[wrIdx];
    const validW = Boolean(
      wSh &&
        wEl &&
        wWr &&
        Math.hypot(wSh.x - wEl.x, wSh.y - wEl.y, (wSh.z ?? 0) - (wEl.z ?? 0)) >= 0.08 &&
        Math.hypot(wWr.x - wEl.x, wWr.y - wEl.y, (wWr.z ?? 0) - (wEl.z ?? 0)) >= 0.08,
    );
    const interiorDeg = validW && wSh && wEl && wWr ? computeJointAngle3D(wSh, wEl, wWr) : computeJointAngle3D(sh, el, wr);
    const baseFlex = clamp(180 - interiorDeg, 0, 155);

    // Proximity fallback: bringing wrist within 1.15 * shoulderWidth of shoulder/chin achieves 100% targetGoal
    const wrToOwnShRatio = dist2D(wr, sh) / Math.max(0.10, shoulderSpanX);
    const isUpwardCurl = wr.y <= el.y + shoulderSpanX * 0.32;
    const proxScore = isUpwardCurl
      ? wrToOwnShRatio <= 1.15
        ? 1.0 + clamp((1.15 - wrToOwnShRatio) / 0.9, 0, 0.08)
        : clamp((1.65 - wrToOwnShRatio) / (1.65 - 1.15), 0, 1)
      : 0;
    const proxFlex = proxScore * Math.max(targetRomGoal, 95);

    const dx = wr.x - el.x;
    const dy = wr.y - el.y;
    const dz = (wr.z ?? 0) - (el.z ?? 0);
    const fMag = Math.hypot(dx, dy, dz);
    return {
      flexionDeg: Number(clamp(Math.max(baseFlex, proxFlex), 0, 155).toFixed(1)),
      xyzVector:
        fMag >= 0.02
          ? {
              x: Number((dx / fMag).toFixed(3)),
              y: Number((dy / fMag).toFixed(3)),
              z: Number((dz / fMag).toFixed(3)),
            }
          : null,
    };
  };

  const primarySide: 'left' | 'right' = targetJoint === 'left_shoulder' ? 'left' : 'right';
  const oppositeSide: 'left' | 'right' = primarySide === 'left' ? 'right' : 'left';
  const activeHand21 = handLandmarksBySide?.[primarySide];

  let romDeg: number | null = null;
  let rawPrimaryRomDeg: number | null = null;
  let leftArmRom: number | null = null;
  let rightArmRom: number | null = null;
  let oppositeRomDeg: number | null = null;
  let sideViolation: StrokeSideViolationCode = 'NONE';
  let sideViolationBannerTh: string | null = null;
  let wrongSideRaised = false;
  let wrongSideBannerTh: string | null = null;
  let wrongPlaneRaised = false;
  let wrongPlaneBannerTh: string | null = null;
  let postureCompensated = false;
  let postureBannerTh: string | null = null;
  let lowVisibilityAlertTh: string | null = null;
  let elbowAngleDeg: number | null = null;
  let xyzVector: { x: number; y: number; z: number } | null = null;
  let neckDropDeg = 0;
  let overlayHint: StrokeTargetOverlayHint | null = null;

  const shIdx = primarySide === 'left' ? POSE_INDEX.LEFT_SHOULDER : POSE_INDEX.RIGHT_SHOULDER;
  const oppShIdx = oppositeSide === 'left' ? POSE_INDEX.LEFT_SHOULDER : POSE_INDEX.RIGHT_SHOULDER;
  const elIdx = primarySide === 'left' ? POSE_INDEX.LEFT_ELBOW : POSE_INDEX.RIGHT_ELBOW;
  const wrIdx = primarySide === 'left' ? POSE_INDEX.LEFT_WRIST : POSE_INDEX.RIGHT_WRIST;
  const activeSh = landmarks[shIdx];
  const oppSh = landmarks[oppShIdx];
  const activeEl = landmarks[elIdx];
  const activeWr = landmarks[wrIdx];

  // --- Dispatch exact clinical equation per jointMode ---
  const computeCrossBodySide = (side: 'left' | 'right'): number | null => {
    const sPt = landmarks[side === 'left' ? POSE_INDEX.LEFT_SHOULDER : POSE_INDEX.RIGHT_SHOULDER];
    const oPt = landmarks[side === 'left' ? POSE_INDEX.RIGHT_SHOULDER : POSE_INDEX.LEFT_SHOULDER];
    const wPt = landmarks[side === 'left' ? POSE_INDEX.LEFT_WRIST : POSE_INDEX.RIGHT_WRIST];
    if (!sPt || !oPt || !wPt || (wPt.visibility ?? 0) < UPPER_MIN_VIS) return null;
    const distToOpp = dist2D(wPt, oPt) / Math.max(0.10, shoulderSpanX);
    // Keep hanging/lap-rest arms below trigger threshold until wrist elevates out of resting hang
    const armLiftedOutOfRest = wPt.y <= sPt.y + effTorsoSpan * 0.72;
    const rawReach =
      distToOpp <= 1.15
        ? 1.0 + clamp((1.15 - distToOpp) / 0.85, 0, 0.08)
        : clamp((1.65 - distToOpp) / (1.65 - 1.15), 0, 1);
    const reachScore = armLiftedOutOfRest ? rawReach : rawReach * 0.25;
    return Number(clamp(reachScore * Math.max(targetRomGoal, 90), 0, Math.max(120, targetRomGoal * 1.1)).toFixed(1));
  };

  const computeHandToHeadSide = (side: 'left' | 'right'): number | null => {
    const sPt = landmarks[side === 'left' ? POSE_INDEX.LEFT_SHOULDER : POSE_INDEX.RIGHT_SHOULDER];
    const wPt = landmarks[side === 'left' ? POSE_INDEX.LEFT_WRIST : POSE_INDEX.RIGHT_WRIST];
    if (!sPt || !wPt || (wPt.visibility ?? 0) < UPPER_MIN_VIS) return null;
    const crownNorm = {
      x: nose ? nose.x : lSh && rSh ? (lSh.x + rSh.x) / 2 : sPt.x,
      y: nose ? Math.max(0.05, nose.y - shoulderSpanX * 0.22) : Math.max(0.05, sPt.y - shoulderSpanX * 0.80),
    };
    const noseAnchor = nose ?? crownNorm;
    const headDistRatio = Math.min(dist2D(wPt, crownNorm), dist2D(wPt, noseAnchor)) / Math.max(0.10, shoulderSpanX);
    const armLifted = wPt.y <= sPt.y + effTorsoSpan * 0.65;
    const proxScore = armLifted
      ? headDistRatio <= 1.15
        ? 1.0 + clamp((1.15 - headDistRatio) / 0.85, 0, 0.08)
        : clamp((1.68 - headDistRatio) / (1.68 - 1.15), 0, 1)
      : 0;
    const liftRes = computeForwardFlexionWithForeshortening(side);
    const goalScale = Math.max(targetRomGoal, 95);
    return Number(clamp(Math.max(proxScore * goalScale, (liftRes.angleDeg ?? 0) * 0.65), 0, Math.max(145, goalScale * 1.1)).toFixed(1));
  };

  const computeScapProtractSide = (
    side: 'left' | 'right',
  ): { deg: number | null; elbowDeg: number | null; xyz: { x: number; y: number; z: number } | null } => {
    const res = computeForwardFlexionWithForeshortening(side);
    const assistedDeg = enhanceSideElevationRom(side, res.angleDeg);
    const sPt = landmarks[side === 'left' ? POSE_INDEX.LEFT_SHOULDER : POSE_INDEX.RIGHT_SHOULDER];
    const elPt = landmarks[side === 'left' ? POSE_INDEX.LEFT_ELBOW : POSE_INDEX.RIGHT_ELBOW];
    const wPt = landmarks[side === 'left' ? POSE_INDEX.LEFT_WRIST : POSE_INDEX.RIGHT_WRIST];
    if (assistedDeg === null || !sPt) {
      return { deg: null, elbowDeg: res.elbowAngleDeg, xyz: res.xyzVector };
    }
    const distal = wPt ?? elPt;
    const depthBoost = distal ? Math.max(0, ((sPt.z ?? 0) - (distal.z ?? 0)) * 42) : 0;
    return {
      deg: Number(clamp(Math.max((res.angleDeg ?? 0) * 0.92 + depthBoost, assistedDeg), 0, Math.max(120, targetRomGoal * 1.1)).toFixed(1)),
      elbowDeg: res.elbowAngleDeg,
      xyz: res.xyzVector,
    };
  };

  const computeRotationSide = (side: 'left' | 'right'): { deg: number | null; driftRatio: number } => {
    const sPt = landmarks[side === 'left' ? POSE_INDEX.LEFT_SHOULDER : POSE_INDEX.RIGHT_SHOULDER];
    const elPt = landmarks[side === 'left' ? POSE_INDEX.LEFT_ELBOW : POSE_INDEX.RIGHT_ELBOW];
    const wPt = landmarks[side === 'left' ? POSE_INDEX.LEFT_WRIST : POSE_INDEX.RIGHT_WRIST];
    if (!sPt || !elPt || !wPt || (elPt.visibility ?? 0) < UPPER_MIN_VIS || (wPt.visibility ?? 0) < UPPER_MIN_VIS) {
      return { deg: null, driftRatio: 0 };
    }
    const driftRatio = Math.abs(elPt.x - sPt.x) / Math.max(0.08, shoulderSpanX);
    const dx = Math.abs(wPt.x - elPt.x);
    const safeFore = Math.max(0.07, baseline.forearmLen);
    const sweepRatio = clamp(dx / safeFore, 0, 1);
    const dz = Math.abs((wPt.z ?? 0) - (elPt.z ?? 0));
    const rotBySin = ((Math.asin(sweepRatio) * 180) / Math.PI) * 0.92;
    const rotByAtan = ((Math.atan2(dx, Math.max(0.03, dz)) * 180) / Math.PI) * 0.95;
    const sweepTarget = sweepRatio * Math.max(targetRomGoal, 55) * 1.25;
    return {
      deg: Number(clamp(Math.max(rotBySin, rotByAtan, sweepTarget), 0, Math.max(85, targetRomGoal * 1.15)).toFixed(1)),
      driftRatio,
    };
  };

  if (jointMode === 'cross_body') {
    const lCross = computeCrossBodySide('left');
    const rCross = computeCrossBodySide('right');
    leftArmRom = lCross;
    rightArmRom = rCross;
    if (exerciseSide === 'BOTH') {
      if (lCross !== null && rCross !== null) {
        rawPrimaryRomDeg = Number((0.5 * (lCross + rCross)).toFixed(1));
        romDeg = rawPrimaryRomDeg;
      } else {
        lowVisibilityAlertTh = 'กรุณาให้เห็นหัวไหล่และมือทั้ง 2 ข้างในกรอบกล้อง';
      }
    } else if (activeSh && oppSh && activeWr && (activeWr.visibility ?? 0) >= UPPER_MIN_VIS) {
      rawPrimaryRomDeg = primarySide === 'left' ? lCross : rCross;
      oppositeRomDeg = primarySide === 'left' ? rCross : lCross;
      romDeg = rawPrimaryRomDeg;
      overlayHint = {
        pivotNorm: { x: activeSh.x, y: activeSh.y },
        targetNorm: { x: oppSh.x, y: oppSh.y },
        currentNorm: { x: activeWr.x, y: activeWr.y },
        inTargetZone: (rawPrimaryRomDeg ?? 0) >= targetRomGoal * 0.82,
        labelTh: 'แตะไหล่ตรงข้าม',
      };
    } else {
      lowVisibilityAlertTh = 'กรุณาให้เห็นหัวไหล่ทั้ง 2 ข้างและมือในกรอบกล้อง';
    }
  } else if (jointMode === 'hand_to_head') {
    const lHead = computeHandToHeadSide('left');
    const rHead = computeHandToHeadSide('right');
    leftArmRom = lHead;
    rightArmRom = rHead;
    if (nose && lEar && rEar && lSh && rSh) {
      const earMidY = (lEar.y + rEar.y) / 2;
      const eyeMidY = lEye && rEye ? (lEye.y + rEye.y) / 2 : earMidY;
      const dropTan = (nose.y - eyeMidY) / Math.max(0.05, shoulderSpanX * 0.35);
      const noseLowerDeg = Math.max(0, (Math.atan(dropTan) * 180) / Math.PI - 12);
      neckDropDeg = Number(noseLowerDeg.toFixed(1));
    }
    if (exerciseSide === 'BOTH') {
      if (lHead !== null && rHead !== null) {
        rawPrimaryRomDeg = Number((0.5 * (lHead + rHead)).toFixed(1));
        romDeg = rawPrimaryRomDeg;
      } else {
        lowVisibilityAlertTh = 'กรุณาให้เห็นศีรษะและมือทั้ง 2 ข้างในกรอบกล้อง';
      }
    } else if (activeSh && activeWr && (activeWr.visibility ?? 0) >= UPPER_MIN_VIS) {
      rawPrimaryRomDeg = primarySide === 'left' ? lHead : rHead;
      oppositeRomDeg = primarySide === 'left' ? rHead : lHead;
      romDeg = rawPrimaryRomDeg;
      const crownNorm = {
        x: nose ? nose.x : lSh && rSh ? (lSh.x + rSh.x) / 2 : activeSh.x,
        y: nose ? Math.max(0.05, nose.y - shoulderSpanX * 0.28) : Math.max(0.05, activeSh.y - shoulderSpanX * 0.85),
      };
      overlayHint = {
        pivotNorm: { x: activeSh.x, y: activeSh.y },
        targetNorm: crownNorm,
        currentNorm: { x: activeWr.x, y: activeWr.y },
        inTargetZone: (rawPrimaryRomDeg ?? 0) >= targetRomGoal * 0.82,
        labelTh: 'ยกมือแตะศีรษะ',
      };
    } else {
      lowVisibilityAlertTh = 'กรุณาให้เห็นศีรษะและมือข้างที่ฝึกในกรอบกล้อง';
    }
  } else if (jointMode === 'scap_protract') {
    const lScap = computeScapProtractSide('left');
    const rScap = computeScapProtractSide('right');
    leftArmRom = lScap.deg;
    rightArmRom = rScap.deg;
    const pScap = primarySide === 'left' ? lScap : rScap;
    elbowAngleDeg = pScap.elbowDeg;
    xyzVector = pScap.xyz;
    if (exerciseSide === 'BOTH' && lScap.deg !== null && rScap.deg !== null) {
      rawPrimaryRomDeg = Number((0.5 * (lScap.deg + rScap.deg)).toFixed(1));
      romDeg = rawPrimaryRomDeg;
    } else if (pScap.deg !== null && activeSh) {
      const wr = activeWr ?? activeEl;
      rawPrimaryRomDeg = pScap.deg;
      romDeg = rawPrimaryRomDeg;
      if (elbowAngleDeg !== null && elbowAngleDeg < 130 && rawPrimaryRomDeg >= 40) {
        postureCompensated = true;
        postureBannerTh = `กรุณาเหยียดข้อศอกให้ตรงขณะยื่นไหล่ (${elbowAngleDeg.toFixed(0)}°)`;
      }
      overlayHint = {
        pivotNorm: { x: activeSh.x, y: activeSh.y },
        targetNorm: { x: activeSh.x, y: activeSh.y - shoulderSpanX * 0.08 },
        currentNorm: wr ? { x: wr.x, y: wr.y } : undefined,
        inTargetZone: rawPrimaryRomDeg >= targetRomGoal * 0.82,
        labelTh: 'ยื่นแขนตรงระดับอก',
      };
    } else {
      lowVisibilityAlertTh = 'มองเห็นหัวไหล่และแขนไม่ชัดเจน กรุณาอยู่ในกรอบกล้อง';
    }
  } else if (jointMode === 'bobath_push') {
    const lElev = computeForwardFlexionWithForeshortening('left');
    const rElev = computeForwardFlexionWithForeshortening('right');
    const lPush = enhanceSideElevationRom('left', lElev.angleDeg);
    const rPush = enhanceSideElevationRom('right', rElev.angleDeg);
    leftArmRom = lPush;
    rightArmRom = rPush;
    const lExt =
      lSh && landmarks[POSE_INDEX.LEFT_ELBOW] && landmarks[POSE_INDEX.LEFT_WRIST]
        ? computeJointAngle3D(lSh, landmarks[POSE_INDEX.LEFT_ELBOW], landmarks[POSE_INDEX.LEFT_WRIST])
        : null;
    const rExt =
      rSh && landmarks[POSE_INDEX.RIGHT_ELBOW] && landmarks[POSE_INDEX.RIGHT_WRIST]
        ? computeJointAngle3D(rSh, landmarks[POSE_INDEX.RIGHT_ELBOW], landmarks[POSE_INDEX.RIGHT_WRIST])
        : null;
    elbowAngleDeg =
      lExt !== null && rExt !== null ? 0.5 * (lExt + rExt) : (lExt ?? rExt ?? null);
    if (lPush !== null && rPush !== null) {
      rawPrimaryRomDeg = Number(clamp(0.5 * (lPush + rPush), 0, 145).toFixed(1));
      xyzVector = rElev.xyzVector ?? lElev.xyzVector;
    } else if (lPush !== null || rPush !== null) {
      rawPrimaryRomDeg = Number(clamp(lPush ?? rPush ?? 0, 0, 145).toFixed(1));
    } else {
      lowVisibilityAlertTh = 'กรุณาให้กล้องเห็นแขนทั้ง 2 ข้างขณะดันไปด้านหน้า';
    }
    romDeg = rawPrimaryRomDeg;
    if (lSh && rSh) {
      const midX = (lSh.x + rSh.x) / 2;
      const midY = (lSh.y + rSh.y) / 2;
      overlayHint = {
        pivotNorm: { x: midX, y: midY },
        targetNorm: { x: midX, y: midY - shoulderSpanX * 0.05 },
        inTargetZone: (rawPrimaryRomDeg ?? 0) >= targetRomGoal * 0.82,
        labelTh: 'ประสานมือดันตรงไปด้านหน้า',
      };
    }
  } else if (jointMode === 'elbow') {
    const primaryRes = computeElbowFlexionRom(primarySide);
    const oppRes = computeElbowFlexionRom(oppositeSide);
    const lRaiseRes = computeForwardFlexionWithForeshortening('left');
    const rRaiseRes = computeForwardFlexionWithForeshortening('right');
    if (exerciseSide === 'BOTH') {
      const lEl = primarySide === 'left' ? primaryRes.flexionDeg : oppRes.flexionDeg;
      const rEl = primarySide === 'right' ? primaryRes.flexionDeg : oppRes.flexionDeg;
      if (lEl !== null && rEl !== null) {
        rawPrimaryRomDeg = Number((0.5 * (lEl + rEl)).toFixed(1));
      } else {
        rawPrimaryRomDeg = Number(((lEl ?? rEl ?? 0)).toFixed(1));
      }
      leftArmRom = lEl;
      rightArmRom = rEl;
    } else {
      rawPrimaryRomDeg = primaryRes.flexionDeg;
      oppositeRomDeg = oppRes.flexionDeg;
    }
    xyzVector = primaryRes.xyzVector;
    romDeg = rawPrimaryRomDeg;
    if (activeSh && activeEl) {
      overlayHint = {
        pivotNorm: { x: activeEl.x, y: activeEl.y },
        targetNorm: { x: activeSh.x, y: activeSh.y + shoulderSpanX * 0.12 },
        currentNorm: activeWr ? { x: activeWr.x, y: activeWr.y } : undefined,
        inTargetZone: (rawPrimaryRomDeg ?? 0) >= targetRomGoal * 0.82,
        labelTh: exerciseSide === 'BOTH' ? 'พับศอกทั้ง 2 ข้าง' : 'พับศอกเข้าหาไหล่',
      };
    }
  } else if (jointMode === 'shoulder_rotation') {
    const lRot = computeRotationSide('left');
    const rRot = computeRotationSide('right');
    leftArmRom = lRot.deg;
    rightArmRom = rRot.deg;
    const pRot = primarySide === 'left' ? lRot : rRot;
    if (exerciseSide === 'BOTH') {
      if (lRot.deg !== null && rRot.deg !== null) {
        rawPrimaryRomDeg = Number((0.5 * (lRot.deg + rRot.deg)).toFixed(1));
        romDeg = rawPrimaryRomDeg;
      } else {
        lowVisibilityAlertTh = 'กรุณาให้เห็นข้อศอกและข้อมือทั้ง 2 ข้างชัดเจนในกรอบกล้อง';
      }
    } else if (pRot.deg !== null && activeSh && activeEl && activeWr) {
      if (pRot.driftRatio > 0.45) {
        postureCompensated = true;
        postureBannerTh = 'กรุณาหนีบข้อศอกให้ชิดข้างลำตัวไว้ครับ';
      }
      rawPrimaryRomDeg = pRot.deg;
      romDeg = rawPrimaryRomDeg;
      const outDir = primarySide === 'left' ? 1 : -1;
      overlayHint = {
        pivotNorm: { x: activeEl.x, y: activeEl.y },
        targetNorm: {
          x: clamp(activeEl.x + outDir * baseline.forearmLen * 0.9, 0.05, 0.95),
          y: activeEl.y,
        },
        currentNorm: { x: activeWr.x, y: activeWr.y },
        inTargetZone: rawPrimaryRomDeg >= targetRomGoal * 0.82,
        labelTh: 'กางปลายแขนออกด้านข้าง',
      };
    } else {
      lowVisibilityAlertTh = 'กรุณาให้เห็นข้อศอกและข้อมือชัดเจนในกรอบกล้อง';
    }
  } else if (jointMode === 'wrist_finger') {
    if (hasUsableHand21(activeHand21)) {
      const w = activeHand21[HAND21_INDEX.WRIST];
      const midMcp = activeHand21[HAND21_INDEX.MIDDLE_MCP];
      const palmLen = Math.max(0.02, dist2D(w, midMcp));
      const tips = [
        HAND21_INDEX.THUMB_TIP,
        HAND21_INDEX.INDEX_TIP,
        HAND21_INDEX.MIDDLE_TIP,
        HAND21_INDEX.RING_TIP,
        HAND21_INDEX.PINKY_TIP,
      ];
      const avgRatio =
        tips.reduce((acc, idx) => acc + dist2D(activeHand21[idx], w), 0) / tips.length / palmLen;
      const spreadScore = clamp((avgRatio - 0.88) / (1.72 - 0.88), 0, 1.08);
      const dorsiflexBonus = Math.max(0, (w.y - midMcp.y) / palmLen) * 32;
      rawPrimaryRomDeg = Number(clamp(spreadScore * Math.max(75, targetRomGoal) + dorsiflexBonus, 0, 95).toFixed(1));
      romDeg = rawPrimaryRomDeg;
      overlayHint = {
        pivotNorm: { x: w.x, y: w.y },
        targetNorm: { x: midMcp.x, y: Math.max(0.05, w.y - palmLen * 1.65) },
        currentNorm: { x: activeHand21[HAND21_INDEX.INDEX_TIP].x, y: activeHand21[HAND21_INDEX.INDEX_TIP].y },
        inTargetZone: rawPrimaryRomDeg >= targetRomGoal * 0.82,
        labelTh: 'กางนิ้วทั้ง 5 ออกกว้าง',
      };
    } else {
      const fallbackRom = computePoseDistalFallbackRom({
        landmarks,
        side: primarySide,
        mode: 'wrist_finger',
        shoulderSpanX,
        forearmLen: baseline.forearmLen,
        targetGoalDeg: targetRomGoal,
      });
      if (fallbackRom !== null) {
        rawPrimaryRomDeg = fallbackRom;
        romDeg = rawPrimaryRomDeg;
      } else {
        lowVisibilityAlertTh = 'กรุณายกมือและข้อมือเข้ามาให้กล้องมองเห็นชัดเจน';
      }
    }
  } else if (jointMode === 'forearm_turn') {
    if (hasUsableHand21(activeHand21)) {
      const thumbCmc = activeHand21[HAND21_INDEX.THUMB_CMC];
      const pinkyMcp = activeHand21[HAND21_INDEX.PINKY_MCP];
      const palmSpan = Math.max(0.015, dist2D(thumbCmc, pinkyMcp));
      const horizFlipRatio = clamp(Math.abs(thumbCmc.x - pinkyMcp.x) / palmSpan, 0, 1);
      const deg = ((Math.acos(1 - horizFlipRatio) * 180) / Math.PI) * 1.15;
      rawPrimaryRomDeg = Number(clamp(deg, 0, 95).toFixed(1));
      romDeg = rawPrimaryRomDeg;
      overlayHint = {
        pivotNorm: { x: activeHand21[0].x, y: activeHand21[0].y },
        targetNorm: { x: thumbCmc.x, y: thumbCmc.y },
        inTargetZone: rawPrimaryRomDeg >= targetRomGoal * 0.82,
        labelTh: 'พลิกหงาย-คว่ำฝ่ามือ',
      };
    } else {
      const fallbackRom = computePoseDistalFallbackRom({
        landmarks,
        side: primarySide,
        mode: 'forearm_turn',
        shoulderSpanX,
        forearmLen: baseline.forearmLen,
        targetGoalDeg: targetRomGoal,
      });
      if (fallbackRom !== null) {
        rawPrimaryRomDeg = fallbackRom;
        romDeg = rawPrimaryRomDeg;
      } else {
        lowVisibilityAlertTh = 'กรุณายกปลายแขนและข้อมือเข้ามาให้กล้องมองเห็นชัดเจน';
      }
    }
  } else if (jointMode === 'thumb_pinch') {
    if (hasUsableHand21(activeHand21)) {
      const thumbTip = activeHand21[HAND21_INDEX.THUMB_TIP];
      const idxTip = activeHand21[HAND21_INDEX.INDEX_TIP];
      const midTip = activeHand21[HAND21_INDEX.MIDDLE_TIP];
      const palmW = Math.max(0.02, dist2D(activeHand21[HAND21_INDEX.INDEX_MCP], activeHand21[HAND21_INDEX.PINKY_MCP]));
      const pinchDist = Math.min(dist2D(thumbTip, idxTip), dist2D(thumbTip, midTip));
      const pinchRatio = pinchDist / palmW;
      const pinchScore = clamp((0.92 - pinchRatio) / (0.92 - 0.22), 0, 1.05);
      rawPrimaryRomDeg = Number(clamp(pinchScore * Math.max(68, targetRomGoal * 1.05), 0, 82).toFixed(1));
      romDeg = rawPrimaryRomDeg;
      overlayHint = {
        pivotNorm: { x: idxTip.x, y: idxTip.y },
        targetNorm: { x: (thumbTip.x + idxTip.x) / 2, y: (thumbTip.y + idxTip.y) / 2 },
        currentNorm: { x: thumbTip.x, y: thumbTip.y },
        inTargetZone: rawPrimaryRomDeg >= targetRomGoal * 0.82,
        labelTh: 'จีบนิ้วโป้งแตะปลายนิ้ว',
      };
    } else {
      const fallbackRom = computePoseDistalFallbackRom({
        landmarks,
        side: primarySide,
        mode: 'thumb_pinch',
        shoulderSpanX,
        forearmLen: baseline.forearmLen,
        targetGoalDeg: targetRomGoal,
      });
      if (fallbackRom !== null) {
        rawPrimaryRomDeg = fallbackRom;
        romDeg = rawPrimaryRomDeg;
      } else {
        lowVisibilityAlertTh = 'กรุณายกมือระดับอกให้เห็นข้อมือชัดเจนในกรอบกล้อง';
      }
    }
  } else if (jointMode === 'neck') {
    if (lSh && rSh && (lSh.visibility ?? 0) >= UPPER_MIN_VIS && (rSh.visibility ?? 0) >= UPPER_MIN_VIS) {
      let tiltDeg = 0;
      let eyeTiltDeg = 0;
      if (lEye && rEye && (lEye.visibility ?? 0) >= 0.25 && (rEye.visibility ?? 0) >= 0.25) {
        const shRad = Math.atan2(rSh.y - lSh.y, lSh.x - rSh.x);
        const eyeRad = Math.atan2(rEye.y - lEye.y, lEye.x - rEye.x);
        let diffDeg = ((eyeRad - shRad) * 180) / Math.PI;
        while (diffDeg > 180) diffDeg -= 360;
        while (diffDeg < -180) diffDeg += 360;
        eyeTiltDeg = Math.abs(diffDeg);
        tiltDeg = eyeTiltDeg;
      }
      if (eyeTiltDeg < 2 && nose && (nose.visibility ?? 0) >= UPPER_MIN_VIS) {
        const neckMidX = (lSh.x + rSh.x) / 2;
        const neckMidY = (lSh.y + rSh.y) / 2;
        const lateralTiltRad = Math.atan2(nose.x - neckMidX, Math.max(0.05, neckMidY - nose.y));
        const noseTiltDeg = ((Math.abs(lateralTiltRad) * 180) / Math.PI) * 1.25;
        tiltDeg = Math.max(tiltDeg, noseTiltDeg);
      }
      // Apply 1.6x clinical sensitivity gain above neutral jitter threshold so natural 12-15 deg reaches target
      const gainedNeckTilt = tiltDeg >= 3.5 ? tiltDeg * 1.6 : tiltDeg;
      rawPrimaryRomDeg = Number(clamp(gainedNeckTilt, 0, 65).toFixed(1));
      romDeg = rawPrimaryRomDeg;
      if (nose) {
        overlayHint = {
          pivotNorm: { x: (lSh.x + rSh.x) / 2, y: (lSh.y + rSh.y) / 2 },
          targetNorm: { x: nose.x + (primarySide === 'left' ? 0.06 : -0.06), y: nose.y + 0.03 },
          inTargetZone: rawPrimaryRomDeg >= targetRomGoal * 0.82,
          labelTh: 'เอียงคอช้าๆ',
        };
      }
    }
  } else if (jointMode === 'neck_turn') {
    if (nose && (nose.visibility ?? 0) >= UPPER_MIN_VIS) {
      const gain = 1.6;
      if (lEar && rEar && (lEar.visibility ?? 0) >= UPPER_MIN_VIS && (rEar.visibility ?? 0) >= UPPER_MIN_VIS) {
        const midEarX = (lEar.x + rEar.x) / 2;
        const earSpan = Math.max(0.04, dist2D(lEar, rEar));
        const yawRatio = Math.abs(nose.x - midEarX) / (earSpan * 0.5);
        rawPrimaryRomDeg = Number(clamp(yawRatio * 58 * (yawRatio >= 0.08 ? gain : 1), 0, 75).toFixed(1));
      } else if (lEye && rEye && (lEye.visibility ?? 0) >= 0.25 && (rEye.visibility ?? 0) >= 0.25) {
        const midEyeX = (lEye.x + rEye.x) / 2;
        const eyeSpan = Math.max(0.03, dist2D(lEye, rEye));
        const yawRatio = Math.abs(nose.x - midEyeX) / (eyeSpan * 0.8);
        rawPrimaryRomDeg = Number(clamp(yawRatio * 55 * (yawRatio >= 0.08 ? gain : 1), 0, 75).toFixed(1));
      } else if (lSh && rSh) {
        const midShX = (lSh.x + rSh.x) / 2;
        const yawRatio = Math.abs(nose.x - midShX) / (shoulderSpanX * 0.35);
        rawPrimaryRomDeg = Number(clamp(yawRatio * 55 * (yawRatio >= 0.08 ? gain : 1), 0, 75).toFixed(1));
      }
      romDeg = rawPrimaryRomDeg;
      if (lSh && rSh) {
        overlayHint = {
          pivotNorm: { x: (lSh.x + rSh.x) / 2, y: nose.y },
          targetNorm: { x: clamp(nose.x + 0.08, 0.1, 0.9), y: nose.y },
          inTargetZone: (rawPrimaryRomDeg ?? 0) >= targetRomGoal * 0.82,
          labelTh: 'หันศีรษะซ้าย-ขวา',
        };
      }
    }
  } else if (jointMode === 'shoulder_shrug') {
    if (lSh && rSh) {
      const curMidShY = (lSh.y + rSh.y) / 2;
      const liftVsBaseline = Math.max(0, baseline.restingMidShoulderY - curMidShY) / Math.max(0.08, shoulderSpanX);
      const earLift =
        lEar || rEar
          ? Math.max(
              0,
              baseline.restingEarShoulderGapRatio -
                (((lEar ? Math.max(0, lSh.y - lEar.y) : 0.1) +
                  (rEar ? Math.max(0, rSh.y - rEar.y) : 0.1)) /
                  2) /
                  Math.max(0.08, shoulderSpanX),
            )
          : 0;
      const baseShrug = Math.max(liftVsBaseline * 240, earLift * 210);
      const shrugScore = baseShrug >= 4 ? baseShrug * 1.6 : baseShrug;
      rawPrimaryRomDeg = Number(clamp(shrugScore, 0, 75).toFixed(1));
      romDeg = rawPrimaryRomDeg;
      overlayHint = {
        pivotNorm: { x: (lSh.x + rSh.x) / 2, y: curMidShY },
        targetNorm: { x: (lSh.x + rSh.x) / 2, y: Math.max(0.05, curMidShY - shoulderSpanX * 0.18) },
        inTargetZone: rawPrimaryRomDeg >= targetRomGoal * 0.82,
        labelTh: 'ยักไหล่ขึ้นหาใบหู',
      };
    }
  } else if (jointMode === 'bilateral_raise') {
    const lRes = computeForwardFlexionWithForeshortening('left');
    const rRes = computeForwardFlexionWithForeshortening('right');
    const lBoost = enhanceSideElevationRom('left', lRes.angleDeg);
    const rBoost = enhanceSideElevationRom('right', rRes.angleDeg);
    leftArmRom = lBoost;
    rightArmRom = rBoost;
    if (lBoost !== null && rBoost !== null) {
      rawPrimaryRomDeg = Number((0.5 * (lBoost + rBoost)).toFixed(1));
      xyzVector = rRes.xyzVector ?? lRes.xyzVector;
    } else if (lBoost !== null || rBoost !== null) {
      rawPrimaryRomDeg = Number((lBoost ?? rBoost ?? 0).toFixed(1));
    } else {
      lowVisibilityAlertTh = 'กรุณายืน/นั่งให้กล้องเห็นแขนทั้ง 2 ข้างชัดเจน';
    }
    romDeg = rawPrimaryRomDeg;
    if (lSh && rSh) {
      const midX = (lSh.x + rSh.x) / 2;
      const midY = (lSh.y + rSh.y) / 2;
      overlayHint = {
        pivotNorm: { x: midX, y: midY },
        targetNorm: { x: midX, y: Math.max(0.06, midY - baseline.upperArmLen * 1.15) },
        inTargetZone: (rawPrimaryRomDeg ?? 0) >= targetRomGoal * 0.82,
        labelTh: 'ประสานมือยกแขนเหนือศีรษะ',
      };
    }
  } else if (targetJoint === 'left_shoulder' || targetJoint === 'right_shoulder') {
    const pRes =
      expectedPlane === 'abduction'
        ? computeCoronalAbduction(primarySide)
        : computeForwardFlexionWithForeshortening(primarySide);
    const oRes =
      expectedPlane === 'abduction'
        ? computeCoronalAbduction(oppositeSide)
        : computeForwardFlexionWithForeshortening(oppositeSide);
    const boostedPrimary = enhanceSideElevationRom(primarySide, pRes.angleDeg);
    rawPrimaryRomDeg = boostedPrimary;
    oppositeRomDeg = oRes.angleDeg;
    leftArmRom = primarySide === 'left' ? boostedPrimary : oRes.angleDeg;
    rightArmRom = primarySide === 'right' ? boostedPrimary : oRes.angleDeg;
    xyzVector = pRes.xyzVector;
    elbowAngleDeg = pRes.elbowAngleDeg;

    if (rawPrimaryRomDeg !== null) {
      if (expectedPlane === 'abduction' && rawPrimaryRomDeg >= 42 && activeSh) {
        const latRatio =
          'lateralDxRatio' in pRes && typeof pRes.lateralDxRatio === 'number'
            ? pRes.lateralDxRatio
            : 0;
        const distalRef = activeWr && (activeWr.visibility ?? 0) >= UPPER_MIN_VIS ? activeWr : activeEl;
        const notPureOverhead = !distalRef || distalRef.y >= activeSh.y - shoulderSpanX * 0.55;
        if (latRatio < 0.35 && (rawPrimaryRomDeg <= 145 || notPureOverhead)) {
          wrongPlaneRaised = true;
          wrongPlaneBannerTh = 'ท่านี้ต้องกางแขนออกด้านข้างเท่านั้น (ห้ามยกแขนไปด้านหน้า)';
        }
      } else if (expectedPlane === 'flexion' && rawPrimaryRomDeg >= 42 && activeSh) {
        const lIsPositiveX = !lSh || !rSh || lSh.x >= rSh.x;
        const outwardSign = lIsPositiveX
          ? primarySide === 'right'
            ? -1
            : 1
          : primarySide === 'right'
            ? 1
            : -1;
        const elOutLat = activeEl ? ((activeEl.x - activeSh.x) * outwardSign) / shoulderSpanX : 0;
        const wrOutLat = activeWr ? ((activeWr.x - activeSh.x) * outwardSign) / shoulderSpanX : 0;
        const distal = activeWr && (activeWr.visibility ?? 0) >= UPPER_MIN_VIS ? activeWr : activeEl;
        const dxAbs = distal ? Math.abs(distal.x - activeSh.x) : 0;
        const dyAbs = distal ? Math.abs(distal.y - activeSh.y) : 0;
        const dzAbs = distal ? Math.abs((distal.z ?? 0) - (activeSh.z ?? 0)) : 0;
        const forwardProxy = Math.max(dyAbs, dzAbs * 0.65);
        if (
          ((wrOutLat >= 0.85 && elOutLat >= 0.45) || elOutLat >= 0.70) &&
          dxAbs > forwardProxy * 1.15 &&
          rawPrimaryRomDeg <= 135
        ) {
          wrongPlaneRaised = true;
          wrongPlaneBannerTh = 'ท่านี้ต้องยกแขนตรงไปด้านหน้าเท่านั้น (ห้ามกางแขนออกด้านข้าง)';
        }
      }
    }

    if (exerciseSide === 'BOTH') {
      if (leftArmRom !== null && rightArmRom !== null) {
        rawPrimaryRomDeg = Number((0.5 * (leftArmRom + rightArmRom)).toFixed(1));
      } else {
        rawPrimaryRomDeg = leftArmRom ?? rightArmRom;
      }
      if (rawPrimaryRomDeg === null) {
        lowVisibilityAlertTh = 'มองเห็นข้อต่อแขนทั้ง 2 ข้างไม่ชัดเจน กรุณาขยับให้อยู่ในกรอบกล้อง';
      }
      romDeg = wrongPlaneRaised ? 0 : rawPrimaryRomDeg;
    } else {
      if (rawPrimaryRomDeg === null) {
        lowVisibilityAlertTh = 'มองเห็นข้อต่อแขนไม่ชัดเจน (< 0.70) กรุณาขยับให้อยู่ในกรอบกล้อง';
      }
      romDeg = wrongPlaneRaised ? 0 : rawPrimaryRomDeg;
    }

    if (activeSh) {
      const radGoal = (clamp(targetRomGoal, 30, 165) * Math.PI) / 180;
      const sideSign = primarySide === 'left' ? 1 : -1;
      const targetOffsetX =
        expectedPlane === 'abduction'
          ? sideSign * Math.sin(radGoal) * baseline.upperArmLen * 1.3
          : sideSign * baseline.upperArmLen * 0.12;
      const targetOffsetY = Math.cos(radGoal) * baseline.upperArmLen * 1.3;
      overlayHint = {
        pivotNorm: { x: activeSh.x, y: activeSh.y },
        targetNorm: {
          x: clamp(activeSh.x + targetOffsetX, 0.04, 0.96),
          y: clamp(activeSh.y + targetOffsetY, 0.04, 0.94),
        },
        currentNorm: activeWr ? { x: activeWr.x, y: activeWr.y } : undefined,
        inTargetZone: !wrongPlaneRaised && (rawPrimaryRomDeg ?? 0) >= targetRomGoal * 0.82,
        labelTh:
          exerciseSide === 'BOTH'
            ? expectedPlane === 'abduction'
              ? 'กางแขนทั้ง 2 ข้างพร้อมกัน'
              : 'ยกแขนทั้ง 2 ข้างพร้อมกัน'
            : expectedPlane === 'abduction'
              ? 'กางแขนถึงจุดเป้าหมาย'
              : 'ยกแขนตรงถึงเป้าหมาย',
      };
    }
  } else if (targetJoint === 'spine') {
    rawPrimaryRomDeg = Number(Math.max(0, 15 - trunkTiltDeg).toFixed(1));
    romDeg = rawPrimaryRomDeg;
  }

  // --- Strict Side Isolation Engine (LEFT / RIGHT / BOTH) ---
  const isUpperLimbIsolatedExercise =
    jointMode !== 'neck' &&
    jointMode !== 'neck_turn' &&
    jointMode !== 'shoulder_shrug' &&
    targetJoint !== 'spine' &&
    targetJoint !== 'left_knee' &&
    targetJoint !== 'right_knee' &&
    targetJoint !== 'none';

  if (isUpperLimbIsolatedExercise) {
    if (leftArmRom === null) {
      leftArmRom = computeForwardFlexionWithForeshortening('left').angleDeg;
    }
    if (rightArmRom === null) {
      rightArmRom = computeForwardFlexionWithForeshortening('right').angleDeg;
    }
    if (oppositeRomDeg === null) {
      oppositeRomDeg = primarySide === 'left' ? rightArmRom : leftArmRom;
    }

    const effectiveSideMode: 'LEFT' | 'RIGHT' | 'BOTH' =
      exerciseSide ??
      (jointMode === 'bilateral_raise' || jointMode === 'bobath_push'
        ? 'BOTH'
        : targetJoint === 'left_shoulder'
          ? 'LEFT'
          : 'RIGHT');

    const isOppositeWristRaisedAboveShoulder = (side: 'left' | 'right'): boolean => {
      const shP = landmarks[side === 'left' ? POSE_INDEX.LEFT_SHOULDER : POSE_INDEX.RIGHT_SHOULDER];
      const wrP = landmarks[side === 'left' ? POSE_INDEX.LEFT_WRIST : POSE_INDEX.RIGHT_WRIST];
      const elP = landmarks[side === 'left' ? POSE_INDEX.LEFT_ELBOW : POSE_INDEX.RIGHT_ELBOW];
      if (!shP || !wrP || !elP || (wrP.visibility ?? 0) < UPPER_MIN_VIS || (shP.visibility ?? 0) < UPPER_MIN_VIS) {
        return false;
      }
      // Only flag opposite side when resting arm tolerance (>60 deg) or wrist genuinely rises above shoulder height
      return wrP.y < shP.y - 0.02 && wrP.y < elP.y - 0.025;
    };

    if (effectiveSideMode === 'LEFT') {
      const rRom = rightArmRom ?? 0;
      const rightRaised = rRom > 60 || isOppositeWristRaisedAboveShoulder('right');
      if (rightRaised) {
        sideViolation = 'RIGHT_ARM_RAISED_IN_LEFT_MODE';
        sideViolationBannerTh =
          '⚠️ ท่านี้ฝึกเฉพาะข้างซ้าย! กรุณาลดแขนขวาลง และยกเฉพาะแขนซ้ายข้างเดียว';
        wrongSideRaised = true;
        wrongSideBannerTh = sideViolationBannerTh;
        romDeg = 0;
      }
    } else if (effectiveSideMode === 'RIGHT') {
      const lRom = leftArmRom ?? 0;
      const leftRaised = lRom > 60 || isOppositeWristRaisedAboveShoulder('left');
      if (leftRaised) {
        sideViolation = 'LEFT_ARM_RAISED_IN_RIGHT_MODE';
        sideViolationBannerTh =
          '⚠️ ท่านี้ฝึกเฉพาะข้างขวา! กรุณาลดแขนซ้ายลง และยกเฉพาะแขนขวาข้างเดียว';
        wrongSideRaised = true;
        wrongSideBannerTh = sideViolationBannerTh;
        romDeg = 0;
      }
    } else if (effectiveSideMode === 'BOTH') {
      const lVal = leftArmRom ?? 0;
      const rVal = rightArmRom ?? 0;
      const oneArmRaisedOnly = (lVal > 60 && rVal < 22) || (rVal > 60 && lVal < 22);
      const asymmetricSpread = Math.max(lVal, rVal) >= 48 && Math.abs(lVal - rVal) > 55;
      if (oneArmRaisedOnly || asymmetricSpread) {
        sideViolation = 'SINGLE_ARM_IN_BOTH_MODE';
        sideViolationBannerTh =
          '⚠️ ท่านี้ต้องทำพร้อมกันทั้ง 2 ข้าง! กรุณายกแขนทั้งซ้ายและขวาขึ้นพร้อมกัน';
        wrongSideRaised = true;
        wrongSideBannerTh = sideViolationBannerTh;
        romDeg = 0;
      }
    }
    if (wrongSideRaised && overlayHint) {
      overlayHint = { ...overlayHint, inTargetZone: false };
    }
  }

  // --- Stroke Compensation Checks (Exact clinical thresholds & user alerts) ---
  let shoulderHikeRatio = 0;
  const compensations: string[] = [];

  if (
    lSh &&
    rSh &&
    (lSh.visibility ?? 0) >= 0.35 &&
    (rSh.visibility ?? 0) >= 0.35
  ) {
    const rawDeltaY = lSh.y - rSh.y;
    const netDeltaY = input.baseline ? rawDeltaY - baseline.restingShoulderDeltaY : rawDeltaY;
    // Left arm lifted -> check if left shoulder shifted significantly upward (lower y => netDeltaY negative)
    // Right arm lifted -> check if right shoulder shifted significantly upward (lower y => netDeltaY positive)
    const activeHikePx =
      targetJoint === 'left_shoulder'
        ? Math.max(0, -netDeltaY)
        : targetJoint === 'right_shoulder'
          ? Math.max(0, netDeltaY)
          : Math.abs(netDeltaY);
    shoulderHikeRatio = Number((activeHikePx / Math.max(0.08, shoulderSpanX)).toFixed(3));
  }

  if (
    trunkTiltDeg > 12 &&
    (!input.baseline || spineDeviationDeg > 8 || trunkTiltDeg > 14) &&
    jointMode !== 'neck' &&
    jointMode !== 'neck_turn'
  ) {
    compensations.push('TRUNK_LEAN');
    postureCompensated = true;
    postureBannerTh = '⚠️ กรุณาตั้งลำตัวให้ตรง ไม่เอนตัวช่วยยกแขน';
  }

  if (
    jointMode !== 'shoulder_shrug' &&
    jointMode !== 'neck' &&
    jointMode !== 'neck_turn' &&
    (rawPrimaryRomDeg ?? 0) >= 20 &&
    !wrongSideRaised &&
    shoulderHikeRatio > 0.14
  ) {
    compensations.push('SHOULDER_HIKE');
    postureCompensated = true;
    postureBannerTh = '⚠️ อย่ายักไหล่ช่วยครับ ผ่อนคลายหัวไหล่ลง';
  }

  if (
    !wrongSideRaised &&
    !wrongPlaneRaised &&
    (rawPrimaryRomDeg ?? 0) >= 40 &&
    (expectedPlane === 'flexion' || expectedPlane === 'abduction') &&
    elbowAngleDeg !== null &&
    elbowAngleDeg < 138 &&
    !postureCompensated
  ) {
    compensations.push('ELBOW_FLEXION_CHEAT');
    postureCompensated = true;
    postureBannerTh = `กรุณาเหยียดข้อศอกให้ตรง (${elbowAngleDeg.toFixed(0)}°)`;
  }

  if (jointMode === 'hand_to_head' && neckDropDeg > 15) {
    compensations.push('NECK_DROP');
    postureCompensated = true;
    postureBannerTh = '⚠️ มองตรงข้างหน้าครับ ให้ยกมือขึ้นหาศีรษะแทนการก้มคอ';
  }

  if (wrongSideRaised) compensations.push('WRONG_LIMB');
  if (wrongPlaneRaised) compensations.push('WRONG_PLANE');
  if (lowVisibilityAlertTh) compensations.push('LOW_VISIBILITY');

  const headY = nose && (nose.visibility ?? 0) >= 0.35 ? nose.y : null;
  const shoulderMidY =
    lSh && rSh && (lSh.visibility ?? 0) >= 0.35 && (rSh.visibility ?? 0) >= 0.35
      ? (lSh.y + rSh.y) / 2
      : null;

  return {
    romDeg,
    rawPrimaryRomDeg,
    leftArmRom,
    rightArmRom,
    oppositeRomDeg,
    trunkTiltDeg: Number(trunkTiltDeg.toFixed(1)),
    shoulderTiltDeg,
    shoulderHikeRatio,
    neckDropDeg,
    elbowAngleDeg,
    sideViolation,
    sideViolationBannerTh,
    wrongSideRaised,
    wrongSideBannerTh,
    wrongPlaneRaised,
    wrongPlaneBannerTh,
    postureCompensated,
    postureBannerTh,
    lowVisibilityAlertTh,
    xyzVector,
    headY,
    shoulderMidY,
    compensations: compensations.length > 0 ? compensations : ['NONE'],
    isFormValid: compensations.length === 0,
    overlayHint,
  };
}

/**
 * Forgiving Clinical 60FPS hysteresis hold & rep state machine:
 * - Zero stale closures (mutates/returns explicit state values)
 * - Easy Hold Entry (>= 0.68 * targetRom) & Keep Holding (>= 0.55 * targetRom)
 * - Accelerated 1.25x hold accumulation when >= 0.85 * targetRom
 * - Progressive Hold Accumulation: dips below 0.55x decay ultra-slowly (dt * 0.15) instead of wiping to 0
 * - Advisory-Only Form Warnings (`SHOULDER_HIKE` / `TRUNK_LEAN` / `isCompensatedNow`) never freeze or block reps
 * - Immediate stop on strict side violations (`LEFT`/`RIGHT`/`BOTH`)
 * - Fast COOLDOWN unlock after `0.7s` (700ms) OR when lowered `< 0.72 * targetRom`.
 */
export function stepStrokeHoldAndRepStateMachine(
  input: StrokeHoldRepStepInput,
): StrokeHoldRepStepResult {
  const safeTargetHold = Math.max(0.5, input.targetHoldSec || 3.0);
  const safeTargetRom = Math.max(15, input.targetRomDeg || 110);
  const dt = Math.max(0, Math.min(0.25, input.dtSec));
  const rom = input.activeRomDeg;

  const next: StrokeHoldRepState = { ...input.state };

  if (next.completedReps >= input.targetReps) {
    return {
      nextState: next,
      holdProgressPct: 100,
      holdRemainingSec: 0,
      justCompletedRep: false,
      bannerTh: '🎉 ฝึกครบเซ็ตเรียบร้อยแล้ว เยี่ยมมากครับ!',
      statusKind: 'SET_FINISHED',
    };
  }

  // 1. Strict Side Isolation Violation -> Stop hold timer immediately & never count
  if (input.sideViolation !== 'NONE') {
    next.holdElapsedSec = 0;
    if (next.phase !== 'COOLDOWN') {
      next.phase = 'WAIT_RAISE';
    }
    return {
      nextState: next,
      holdProgressPct: 0,
      holdRemainingSec: Number(safeTargetHold.toFixed(1)),
      justCompletedRep: false,
      bannerTh: input.wrongSideBannerTh,
      statusKind: 'WRONG_LIMB',
    };
  }

  // 2. Wrong Movement Plane Violation -> Stop hold timer & alert
  if (input.wrongPlaneRaised) {
    next.holdElapsedSec = 0;
    if (next.phase !== 'COOLDOWN') {
      next.phase = 'WAIT_RAISE';
    }
    return {
      nextState: next,
      holdProgressPct: 0,
      holdRemainingSec: Number(safeTargetHold.toFixed(1)),
      justCompletedRep: false,
      bannerTh: input.wrongPlaneBannerTh,
      statusKind: 'WRONG_PLANE',
    };
  }

  // 3. COOLDOWN phase -> Fast clinical unlock after 0.7s OR when lowered < 0.72 * targetRom
  if (next.phase === 'COOLDOWN') {
    const cooldownElapsedMs = input.nowMs - next.cooldownStartedAtMs;
    const isKneeFlex = Boolean(input.isKneeExercise && safeTargetRom < 140);
    const unlockedByReturn =
      rom !== null &&
      (isKneeFlex ? rom > safeTargetRom + 20 : rom < safeTargetRom * 0.72);

    if (cooldownElapsedMs >= 700 || (cooldownElapsedMs >= 150 && unlockedByReturn)) {
      next.phase = 'WAIT_RAISE';
      next.holdElapsedSec = 0;
      return {
        nextState: next,
        holdProgressPct: 0,
        holdRemainingSec: Number(safeTargetHold.toFixed(1)),
        justCompletedRep: false,
        bannerTh: null,
        statusKind: 'ACTIVE_REP',
      };
    }

    return {
      nextState: next,
      holdProgressPct: 100,
      holdRemainingSec: 0,
      justCompletedRep: false,
      bannerTh: '✓ นับสำเร็จ +1 ครั้ง! ค่อยๆ ลดแขนลง',
      statusKind: 'REP_COMPLETED',
    };
  }

  if (rom === null || !Number.isFinite(rom) || input.blockCounting) {
    // Freeze softly without wiping progress on dropped frames or active face mismatch pause
    const pct = Math.min(100, (next.holdElapsedSec / safeTargetHold) * 100);
    const rem = Math.max(0, safeTargetHold - next.holdElapsedSec);
    return {
      nextState: next,
      holdProgressPct: Number(pct.toFixed(2)),
      holdRemainingSec: Number(rem.toFixed(1)),
      justCompletedRep: false,
      bannerTh: null,
      statusKind: next.phase === 'HOLDING' ? 'HOLD_PHASE' : 'ACTIVE_REP',
    };
  }

  // 4. Forgiving Clinical Hysteresis & Progressive Accumulation
  const isKneeFlex = Boolean(input.isKneeExercise && safeTargetRom < 140);
  const inBoostZone = isKneeFlex ? rom <= safeTargetRom + 12 : rom >= safeTargetRom * 0.85;
  const inEnterZone = isKneeFlex ? rom <= safeTargetRom + 20 : rom >= safeTargetRom * 0.68;
  const inMaintainZone = isKneeFlex ? rom <= safeTargetRom + 28 : rom >= safeTargetRom * 0.55;
  const holdGainDt = inBoostZone ? dt * 1.25 : dt;

  // Note: input.isCompensatedNow (e.g. SHOULDER_HIKE / TRUNK_LEAN) is ADVISORY-ONLY and never halts counting!
  if (next.phase === 'WAIT_RAISE') {
    if (inEnterZone) {
      next.phase = 'HOLDING';
      next.holdElapsedSec = Math.min(safeTargetHold, next.holdElapsedSec + holdGainDt);
    } else if (next.holdElapsedSec > 0) {
      // Progressive retention: decay ultra-slowly so temporary tremor never erases effort
      next.holdElapsedSec = Math.max(0, next.holdElapsedSec - dt * 0.15);
    }
  } else if (next.phase === 'HOLDING') {
    if (inMaintainZone) {
      next.holdElapsedSec = Math.min(safeTargetHold, next.holdElapsedSec + holdGainDt);
    } else {
      // Never reset holdElapsedSec straight to 0 mid-hold; decay ultra-slowly at 0.15x
      next.holdElapsedSec = Math.max(0, next.holdElapsedSec - dt * 0.15);
      if (next.holdElapsedSec <= 0) {
        next.phase = 'WAIT_RAISE';
      }
    }
  }

  // 5. Guaranteed Rep Trigger when targetHoldSec reached
  if (next.holdElapsedSec >= safeTargetHold - 1e-4) {
    next.completedReps = Math.min(input.targetReps, next.completedReps + 1);
    next.holdElapsedSec = safeTargetHold;
    next.phase = 'COOLDOWN';
    next.cooldownStartedAtMs = input.nowMs;
    next.repCelebrateUntilMs = input.nowMs + 1200;
    return {
      nextState: next,
      holdProgressPct: 100,
      holdRemainingSec: 0,
      justCompletedRep: true,
      bannerTh: '✓ นับสำเร็จ +1 ครั้ง! ค่อยๆ ลดแขนลง',
      statusKind: next.completedReps >= input.targetReps ? 'SET_FINISHED' : 'REP_COMPLETED',
    };
  }

  const pct = Math.min(100, (next.holdElapsedSec / safeTargetHold) * 100);
  const rem = Math.max(0, safeTargetHold - next.holdElapsedSec);
  return {
    nextState: next,
    holdProgressPct: Number(pct.toFixed(2)),
    holdRemainingSec: Number(rem.toFixed(1)),
    justCompletedRep: false,
    bannerTh: input.isCompensatedNow && input.postureBannerTh ? input.postureBannerTh : null,
    statusKind: next.phase === 'HOLDING' ? 'HOLD_PHASE' : 'ACTIVE_REP',
  };
}
