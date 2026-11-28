import { POSE_INDEX, computeJointAngle3D } from './armJointProcessor';
import {
  HAND21_INDEX,
  analyzeHand21Biomechanics,
  hasUsableHand21Landmarks,
  type Hand21Biomechanics,
} from './handFingerTracker';
import type { PoseLandmark } from './types';

export type ElderlyJointMode =
  | 'elbow'
  | 'cross_body'
  | 'scap_protract'
  | 'clasped_push'
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

export interface ElderlyAnatomyBaseline {
  shoulderWidth: number;
  upperArmLen: number;
  forearmLen: number;
  restingShoulderDeltaY: number;
  restingSpineAngleDeg: number;
  restingMidShoulderY: number;
  restingMidShoulderZ?: number;
  restingEarShoulderGapRatio: number;
  sampleCount: number;
}

export interface ElderlyTargetOverlayHint {
  pivotNorm: { x: number; y: number };
  targetNorm: { x: number; y: number };
  currentNorm?: { x: number; y: number };
  inTargetZone: boolean;
  labelTh: string;
}

export type ElderlySideViolationCode =
  | 'NONE'
  | 'RIGHT_ARM_RAISED_IN_LEFT_MODE'
  | 'LEFT_ARM_RAISED_IN_RIGHT_MODE'
  | 'SINGLE_ARM_IN_BOTH_MODE';

export type ElderlyRepPhase =
  | 'PHASE_1_HOLDING'
  | 'PHASE_2_WAIT_LOWER_TO_SCORE'
  | 'PHASE_3_SCORED_READY';

export interface ElderlyHoldRepState {
  phase: ElderlyRepPhase;
  holdElapsedSec: number;
  completedReps: number;
  cooldownStartedAtMs: number;
  repCelebrateUntilMs: number;
  loweredFrameStreak?: number;
  readyBannerUntilMs?: number;
  waitLowerStartedAtMs?: number;
}

export interface ElderlyHoldRepStepInput {
  state: ElderlyHoldRepState;
  activeRomDeg: number | null;
  leftArmRomDeg?: number | null;
  rightArmRomDeg?: number | null;
  exerciseCode?: string;
  exerciseSide?: 'LEFT' | 'RIGHT' | 'BOTH';
  targetRomDeg: number;
  targetHoldSec: number;
  targetReps: number;
  dtSec: number;
  nowMs: number;
  isKneeExercise?: boolean;
  isSmallRomExercise?: boolean;
  armWristLowered?: boolean;
  handsOutOfFrameRaised?: boolean;
  sideViolation: ElderlySideViolationCode;
  wrongSideBannerTh: string | null;
  wrongPlaneRaised: boolean;
  wrongPlaneBannerTh: string | null;
  isCompensatedNow: boolean;
  postureBannerTh: string | null;
  blockCounting?: boolean;
  /** Optional real-rotation verifier gate for forearm_turn, shoulder_rotation, and shoulder_shrug. */
  rotationGate?: {
    required: boolean;
    rotationVerified: boolean;
    returnedToStart: boolean;
    guidanceBannerTh?: string | null;
  };
}

export interface ElderlyHoldRepStepResult {
  nextState: ElderlyHoldRepState;
  holdProgressPct: number;
  holdRemainingSec: number;
  justCompletedRep: boolean;
  holdCompleteWaitLower?: boolean;
  unlockedToReady?: boolean;
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

export interface ElderlyPoseSolveInput {
  landmarks: readonly PoseLandmark[];
  worldLandmarks?: readonly PoseLandmark[];
  handLandmarksBySide?: {
    left?: readonly PoseLandmark[];
    right?: readonly PoseLandmark[];
  };
  handWorldLandmarksBySide?: {
    left?: readonly PoseLandmark[];
    right?: readonly PoseLandmark[];
  };
  targetJoint: 'left_shoulder' | 'right_shoulder' | 'spine' | 'left_knee' | 'right_knee' | 'none';
  exerciseSide?: 'LEFT' | 'RIGHT' | 'BOTH';
  expectedPlane?: string;
  targetRomDeg?: number;
  jointMode: ElderlyJointMode;
  baseline?: ElderlyAnatomyBaseline;
  videoAspect?: number;
}

export interface ElderlyRotationTelemetry {
  mode: 'forearm_turn' | 'shoulder_rotation' | 'shoulder_shrug';
  signedAngleDeg: number | null;
  secondaryRollSignal?: number;
  postureReady: boolean;
}

export interface ElderlyPoseSolveResult {
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
  sideViolation: ElderlySideViolationCode;
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
  overlayHint: ElderlyTargetOverlayHint | null;
  /** High-detail MediaPipe 21-point hand & finger biomechanics when HandLandmarker is active. */
  handBiomechanics?: {
    left: Hand21Biomechanics | null;
    right: Hand21Biomechanics | null;
    active: Hand21Biomechanics | null;
  } | null;
  /** Live signed rotation signal fed into ClinicalRotationCycleTracker for genuine rotation counting. */
  rotationTelemetry?: ElderlyRotationTelemetry | null;
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
): ElderlyAnatomyBaseline {
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
  const restingMidShoulderZ =
    lSh && rSh ? (((lSh.z ?? 0) + (rSh.z ?? 0)) / 2) : 0;

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
    restingMidShoulderZ,
    restingEarShoulderGapRatio,
    sampleCount: 1,
  };
}

/**
 * Automatic Bone-Length Baseline Calibrator capturing patient-specific anatomy
 * across initial neutral frames without forcing a fixed standing pause.
 */
export class AutoBoneBaselineCalibrator {
  private baseline: ElderlyAnatomyBaseline | null = null;

  reset(): void {
    this.baseline = null;
  }

  getBaseline(): ElderlyAnatomyBaseline | null {
    return this.baseline ? { ...this.baseline } : null;
  }

  update(landmarks: readonly PoseLandmark[]): ElderlyAnatomyBaseline {
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
        this.baseline.restingMidShoulderZ =
          (this.baseline.restingMidShoulderZ ?? 0) * (1 - alpha) +
          (snap.restingMidShoulderZ ?? 0) * alpha;
        this.baseline.restingEarShoulderGapRatio =
          this.baseline.restingEarShoulderGapRatio * (1 - alpha) +
          snap.restingEarShoulderGapRatio * alpha;
      }
    }
    this.baseline.sampleCount += 1;
    return { ...this.baseline };
  }
}

/**
 * Pose-only distal fallback when MediaPipe 21-point HandLandmarker is unavailable/occluded.
 * Uses exact anatomical trigonometry (signed forearm roll, wrist pitch, thumb-index span)
 * without artificial lift inflation.
 */
function computePoseDistalFallbackRom(params: {
  landmarks: readonly PoseLandmark[];
  side: 'left' | 'right';
  mode: 'forearm_turn' | 'wrist_finger' | 'thumb_pinch';
  shoulderSpanX: number;
  forearmLen: number;
  targetGoalDeg: number;
}): { deg: number | null; signedRollDeg?: number | null } {
  const { landmarks, side, mode, shoulderSpanX, forearmLen, targetGoalDeg } = params;
  const sh = landmarks[side === 'left' ? POSE_INDEX.LEFT_SHOULDER : POSE_INDEX.RIGHT_SHOULDER];
  const el = landmarks[side === 'left' ? POSE_INDEX.LEFT_ELBOW : POSE_INDEX.RIGHT_ELBOW];
  const wr = landmarks[side === 'left' ? POSE_INDEX.LEFT_WRIST : POSE_INDEX.RIGHT_WRIST];
  const pinky = landmarks[side === 'left' ? POSE_INDEX.LEFT_PINKY : POSE_INDEX.RIGHT_PINKY];
  const index = landmarks[side === 'left' ? POSE_INDEX.LEFT_INDEX : POSE_INDEX.RIGHT_INDEX];
  const thumb = landmarks[side === 'left' ? POSE_INDEX.LEFT_THUMB : POSE_INDEX.RIGHT_THUMB];

  if (!el || !wr || (wr.visibility ?? 0) < 0.25) return { deg: null, signedRollDeg: null };
  const safeFore = Math.max(0.08, forearmLen, shoulderSpanX * 0.65);
  const handLifted =
    wr.y <= el.y + safeFore * 0.42 && (!sh || wr.y <= sh.y + safeFore * 1.12);

  if (mode === 'thumb_pinch') {
    if (thumb && index && (thumb.visibility ?? 0) >= 0.2 && (index.visibility ?? 0) >= 0.2) {
      if (!handLifted) return { deg: 0 };
      const palmRef = Math.max(0.035, shoulderSpanX * 0.30);
      const pinchRatio = dist2D(thumb, index) / palmRef;
      const pinchClosePct = clamp((1.05 - pinchRatio) / (1.05 - 0.28), 0, 1.05);
      return {
        deg: Number(clamp(pinchClosePct * Math.min(72, targetGoalDeg * 1.05), 0, 75).toFixed(1)),
      };
    }
    return { deg: null };
  }

  if (mode === 'forearm_turn') {
    if (index && pinky && (index.visibility ?? 0) >= 0.2 && (pinky.visibility ?? 0) >= 0.2) {
      if (!handLifted) return { deg: 0, signedRollDeg: 0 };
      const sideSign = side === 'right' ? 1 : -1;
      const span = Math.max(0.02, dist2D(index, pinky));
      // Signed transverse polarity flips when palm rotates from pronated <-> supinated
      const signedX = ((index.x - pinky.x) * sideSign) / span;
      const dz = ((index.z ?? 0) - (pinky.z ?? 0)) * sideSign;
      const signedRollDeg = Number(
        clamp((Math.atan2(signedX, Math.max(0.18, Math.abs(dz) / span)) * 180) / Math.PI, -90, 90).toFixed(1),
      );
      return {
        deg: Number(Math.abs(signedRollDeg).toFixed(1)),
        signedRollDeg,
      };
    }
    return { deg: null, signedRollDeg: null };
  }

  // wrist_finger fallback: true wrist dorsiflexion angle between elbow->wrist and wrist->index/pinky
  const handTip = index && (index.visibility ?? 0) >= 0.2 ? index : pinky;
  if (!handTip || !handLifted) {
    return { deg: handLifted ? 0 : 0 };
  }
  const interiorWrist = computeJointAngle3D(el, wr, handTip);
  const wristDevDeg = clamp(Math.abs(180 - interiorWrist), 0, 85);
  return { deg: Number(wristDevDeg.toFixed(1)) };
}

/**
 * Computes clinical 3D joint angles and Elderly compensation detections across all
 * 26 standardized Elderly Care & MOPH upper-body exercises.
 */
export function solveElderlyClinicalJointBiomechanics(
  input: ElderlyPoseSolveInput,
): ElderlyPoseSolveResult {
  const {
    landmarks,
    worldLandmarks,
    handLandmarksBySide,
    handWorldLandmarksBySide,
    targetJoint,
    exerciseSide,
    expectedPlane = 'any',
    jointMode,
  } = input;
  const baseline = input.baseline ?? computeDefaultAnatomyBaseline(landmarks);
  const aspectScale = clamp(input.videoAspect ?? 1.22, 0.55, 2.4);
  const UPPER_MIN_VIS = 0.02;

  const lSh = landmarks[POSE_INDEX.LEFT_SHOULDER];
  const rSh = landmarks[POSE_INDEX.RIGHT_SHOULDER];
  const nose = landmarks[POSE_INDEX.NOSE];
  const lEar = landmarks[POSE_INDEX.LEFT_EAR];
  const rEar = landmarks[POSE_INDEX.RIGHT_EAR];
  const lEye = landmarks[POSE_INDEX.LEFT_EYE];
  const rEye = landmarks[POSE_INDEX.RIGHT_EYE];

  const shoulderSpanX = baseline.shoulderWidth;
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
  const effTorsoSpan = hipsVisible ? Math.max(torsoSpanY, shoulderSpanX * 1.15) : shoulderSpanX * 1.48;

  // Analyze 21-point MediaPipe hand & finger biomechanics whenever HandLandmarker provides hands
  const leftHandBio = hasUsableHand21Landmarks(handLandmarksBySide?.left)
    ? analyzeHand21Biomechanics({
        landmarks2D: handLandmarksBySide.left,
        worldLandmarks3D: handWorldLandmarksBySide?.left,
        side: 'left',
        poseElbow: landmarks[POSE_INDEX.LEFT_ELBOW],
        poseWrist: landmarks[POSE_INDEX.LEFT_WRIST],
      })
    : null;
  const rightHandBio = hasUsableHand21Landmarks(handLandmarksBySide?.right)
    ? analyzeHand21Biomechanics({
        landmarks2D: handLandmarksBySide.right,
        worldLandmarks3D: handWorldLandmarksBySide?.right,
        side: 'right',
        poseElbow: landmarks[POSE_INDEX.RIGHT_ELBOW],
        poseWrist: landmarks[POSE_INDEX.RIGHT_WRIST],
      })
    : null;

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
      if (!pt || (pt.visibility ?? 0) < UPPER_MIN_VIS || !Number.isFinite(pt.x) || !Number.isFinite(pt.y)) {
        return null;
      }
      const dx = (pt.x - sh.x) * aspectScale;
      const dy = pt.y - sh.y;
      const dzSensor = ((Number.isFinite(pt.z) ? (pt.z as number) : 0) - (Number.isFinite(sh.z) ? (sh.z as number) : 0)) * aspectScale;

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
        expectedPlane === 'abduction'
          ? 0
          : dy > 0
            ? Math.max(clamp(Math.abs(dzSensor) / 0.14, 0, 1), foreshortenedForward)
            : 1;
      // Taper sphere depth completion near vertical overhead (|dy| >= 74% refLen) so straight-up arms report true ~172-180 deg
      const overheadTaper =
        dy < 0 ? clamp((0.92 - Math.abs(dy) / Math.max(0.05, refLen)) / 0.18, 0, 1) : 1;
      const absDz = Math.max(
        Math.abs(dzSensor),
        dzSphere * sagittalWeight * downForwardWeight * overheadTaper,
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
        if (
          wSh &&
          wPt &&
          Number.isFinite(wSh.x) &&
          Number.isFinite(wSh.y) &&
          Number.isFinite(wPt.x) &&
          Number.isFinite(wPt.y)
        ) {
          const wDx = wPt.x - wSh.x;
          const wDy = wPt.y - wSh.y;
          const wDz = (Number.isFinite(wPt.z) ? (wPt.z as number) : 0) - (Number.isFinite(wSh.z) ? (wSh.z as number) : 0);
          const wLen3D = Math.hypot(wDx, wDy, wDz);
          if (Number.isFinite(wLen3D) && wLen3D >= 0.08) {
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

    // Exact anatomical goniometry: shoulder elevation is defined by humerus (shoulder->elbow);
    // blend wrist only when elbow is fairly straight so curling the elbow cannot inflate shoulder angle.
    let angleDeg: number | null = null;
    if (elbowMeasure && wristMeasure) {
      const elbowStraight = elbowAngleDeg === null || elbowAngleDeg >= 135;
      angleDeg = elbowStraight
        ? 0.65 * elbowMeasure.angleDeg + 0.35 * wristMeasure.angleDeg
        : elbowMeasure.angleDeg;
    } else if (elbowMeasure) {
      angleDeg = elbowMeasure.angleDeg;
    } else if (wristMeasure) {
      angleDeg = wristMeasure.angleDeg;
    }

    return {
      angleDeg: angleDeg !== null ? Number(clamp(angleDeg, 0, 180).toFixed(1)) : null,
      xyzVector: elbowMeasure?.xyzVector ?? wristMeasure?.xyzVector ?? null,
      elbowAngleDeg,
    };
  };

  const computeCoronalAbduction = (
    side: 'left' | 'right',
  ): {
    angleDeg: number | null;
    forward3dDeg: number | null;
    xyzVector: { x: number; y: number; z: number } | null;
    elbowAngleDeg: number | null;
    lateralDxRatio: number;
  } => {
    const res = computeForwardFlexionWithForeshortening(side);
    const shIdx = side === 'left' ? POSE_INDEX.LEFT_SHOULDER : POSE_INDEX.RIGHT_SHOULDER;
    const wrIdx = side === 'left' ? POSE_INDEX.LEFT_WRIST : POSE_INDEX.RIGHT_WRIST;
    const elIdx = side === 'left' ? POSE_INDEX.LEFT_ELBOW : POSE_INDEX.RIGHT_ELBOW;
    const sh = landmarks[shIdx];
    const el = landmarks[elIdx];
    const wr = landmarks[wrIdx];
    const distal =
      el && (el.visibility ?? 0) >= UPPER_MIN_VIS
        ? el
        : wr && (wr.visibility ?? 0) >= UPPER_MIN_VIS
          ? wr
          : undefined;
    if (!sh || !distal) {
      return { ...res, forward3dDeg: res.angleDeg, lateralDxRatio: 0 };
    }
    const lateralRef = wr && (wr.visibility ?? 0) >= UPPER_MIN_VIS ? wr : distal;
    const lateralDxRatio = Math.abs(lateralRef.x - sh.x) / Math.max(0.08, shoulderSpanX);
    const spineRad = (liveSpineDeg * Math.PI) / 180;
    const spineDownX = -Math.sin(spineRad);
    const spineDownY = Math.cos(spineRad);
    const vx = (distal.x - sh.x) * aspectScale;
    const vy = distal.y - sh.y;
    const vn = Math.hypot(vx, vy);
    const coronalDeg =
      vn >= 0.02
        ? (Math.acos(clamp((vx * spineDownX + vy * spineDownY) / vn, -1, 1)) * 180) / Math.PI
        : 0;
    const blended = res.angleDeg !== null ? 0.65 * coronalDeg + 0.35 * res.angleDeg : coronalDeg;
    return {
      angleDeg: Number(clamp(blended, 0, 180).toFixed(1)),
      forward3dDeg: res.angleDeg,
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
    // Pure exact 3D anatomical elbow flexion angle: 180 deg - interior(shoulder, elbow, wrist)
    const interiorDeg =
      validW && wSh && wEl && wWr ? computeJointAngle3D(wSh, wEl, wWr) : computeJointAngle3D(sh, el, wr);
    const exactFlexDeg = clamp(180 - interiorDeg, 0, 180);

    const dx = wr.x - el.x;
    const dy = wr.y - el.y;
    const dz = (wr.z ?? 0) - (el.z ?? 0);
    const fMag = Math.hypot(dx, dy, dz);
    return {
      flexionDeg: Number(exactFlexDeg.toFixed(1)),
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

  const primarySide: 'left' | 'right' =
    exerciseSide === 'LEFT'
      ? 'left'
      : exerciseSide === 'RIGHT'
        ? 'right'
        : targetJoint === 'left_shoulder'
          ? 'left'
          : 'right';
  const oppositeSide: 'left' | 'right' = primarySide === 'left' ? 'right' : 'left';
  const activeHand21 =
    handLandmarksBySide?.[primarySide] ??
    (exerciseSide === 'BOTH' ? (handLandmarksBySide?.right ?? handLandmarksBySide?.left) : undefined);
  const activeHandBio =
    (primarySide === 'left' ? leftHandBio : rightHandBio) ??
    (exerciseSide === 'BOTH' ? (rightHandBio ?? leftHandBio) : null);
  let rotationTelemetry: ElderlyRotationTelemetry | null = null;

  let romDeg: number | null = null;
  let rawPrimaryRomDeg: number | null = null;
  let leftArmRom: number | null = null;
  let rightArmRom: number | null = null;
  let oppositeRomDeg: number | null = null;
  let sideViolation: ElderlySideViolationCode = 'NONE';
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
  let overlayHint: ElderlyTargetOverlayHint | null = null;

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
      distToOpp <= 0.85
        ? 1.0 + clamp((0.85 - distToOpp) / 0.85, 0, 0.08)
        : clamp((1.45 - distToOpp) / (1.45 - 0.85), 0, 1);
    const reachScore = armLiftedOutOfRest ? rawReach : rawReach * 0.25;
    return Number(clamp(reachScore * Math.max(targetRomGoal, 90), 0, Math.min(180, Math.max(120, targetRomGoal * 1.1))).toFixed(1));
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
    const armLifted = wPt.y <= sPt.y + effTorsoSpan * 0.20;
    const proxScore = armLifted
      ? headDistRatio <= 0.90
        ? 1.0 + clamp((0.90 - headDistRatio) / 0.90, 0, 0.08)
        : clamp((1.48 - headDistRatio) / (1.48 - 0.90), 0, 1)
      : 0;
    const liftRes = computeForwardFlexionWithForeshortening(side);
    const goalScale = Math.max(targetRomGoal, 95);
    return Number(clamp(Math.max(proxScore * goalScale, (liftRes.angleDeg ?? 0) * 0.65), 0, Math.min(180, Math.max(145, goalScale * 1.1))).toFixed(1));
  };

  const computeScapProtractSide = (
    side: 'left' | 'right',
  ): { deg: number | null; elbowDeg: number | null; xyz: { x: number; y: number; z: number } | null } => {
    const res = computeForwardFlexionWithForeshortening(side);
    const sPt = landmarks[side === 'left' ? POSE_INDEX.LEFT_SHOULDER : POSE_INDEX.RIGHT_SHOULDER];
    const elPt = landmarks[side === 'left' ? POSE_INDEX.LEFT_ELBOW : POSE_INDEX.RIGHT_ELBOW];
    const wPt = landmarks[side === 'left' ? POSE_INDEX.LEFT_WRIST : POSE_INDEX.RIGHT_WRIST];
    if (res.angleDeg === null || !sPt) {
      return { deg: null, elbowDeg: res.elbowAngleDeg, xyz: res.xyzVector };
    }
    const distal = wPt ?? elPt;
    const forwardReachDepth = distal ? Math.max(0, (sPt.z ?? 0) - (distal.z ?? 0)) : 0;
    const protractBonus = clamp(forwardReachDepth * 22, 0, 12);
    return {
      deg: Number(clamp(res.angleDeg + protractBonus, 0, 180).toFixed(1)),
      elbowDeg: res.elbowAngleDeg,
      xyz: res.xyzVector,
    };
  };

  // Exact Glenohumeral External/Internal Rotation around vertical humerus axis with 90-deg elbow tuck verification
  const computeRotationSide = (
    side: 'left' | 'right',
  ): {
    deg: number | null;
    signedDeg: number | null;
    driftRatio: number;
    elbowFlexOk: boolean;
    elbowAngleDeg: number | null;
  } => {
    const shIdxS = side === 'left' ? POSE_INDEX.LEFT_SHOULDER : POSE_INDEX.RIGHT_SHOULDER;
    const elIdxS = side === 'left' ? POSE_INDEX.LEFT_ELBOW : POSE_INDEX.RIGHT_ELBOW;
    const wrIdxS = side === 'left' ? POSE_INDEX.LEFT_WRIST : POSE_INDEX.RIGHT_WRIST;
    const sPt = landmarks[shIdxS];
    const elPt = landmarks[elIdxS];
    const wPt = landmarks[wrIdxS];
    if (!sPt || !elPt || !wPt || (elPt.visibility ?? 0) < UPPER_MIN_VIS || (wPt.visibility ?? 0) < UPPER_MIN_VIS) {
      return { deg: null, signedDeg: null, driftRatio: 0, elbowFlexOk: false, elbowAngleDeg: null };
    }
    const driftRatio = Math.abs(elPt.x - sPt.x) / Math.max(0.08, shoulderSpanX);
    const wS = worldLandmarks?.[shIdxS];
    const wE = worldLandmarks?.[elIdxS];
    const wW = worldLandmarks?.[wrIdxS];
    const validW = Boolean(
      wS && wE && wW &&
      Math.hypot(wS.x - wE.x, wS.y - wE.y, (wS.z ?? 0) - (wE.z ?? 0)) >= 0.08 &&
      Math.hypot(wW.x - wE.x, wW.y - wE.y, (wW.z ?? 0) - (wE.z ?? 0)) >= 0.08,
    );
    const elAngle = validW && wS && wE && wW ? computeJointAngle3D(wS, wE, wW) : computeJointAngle3D(sPt, elPt, wPt);
    const dyDown = Math.max(0, wPt.y - elPt.y);
    const safeFore = Math.max(0.07, baseline.forearmLen);

    // Must NOT be a straight relaxed arm hanging vertically down!
    const armNotHanging = dyDown <= safeFore * 0.58;
    const elbowFlexOk = armNotHanging && driftRatio <= 0.48 && elAngle >= 45 && elAngle <= 145;
    if (!armNotHanging) {
      return { deg: 0, signedDeg: 0, driftRatio, elbowFlexOk: false, elbowAngleDeg: elAngle };
    }

    // Outward lateral direction sign for external rotation
    const lIsPositiveX = !lSh || !rSh || lSh.x >= rSh.x;
    const outwardSign = lIsPositiveX ? (side === 'left' ? 1 : -1) : (side === 'left' ? -1 : 1);

    let signedDeg = 0;
    if (validW && wE && wW) {
      const fDx = (wW.x - wE.x) * outwardSign;
      const fDzForward = Math.max(0.03, -(wW.z ?? 0) + (wE.z ?? 0), Math.abs((wW.z ?? 0) - (wE.z ?? 0)));
      signedDeg = (Math.atan2(fDx, fDzForward) * 180) / Math.PI;
    } else {
      const signedDx = (wPt.x - elPt.x) * outwardSign;
      const absDx = Math.abs(signedDx);
      const dy = wPt.y - elPt.y;
      const dzSensor = Math.abs((wPt.z ?? 0) - (elPt.z ?? 0));
      const dzSphere = Math.sqrt(Math.max(0, safeFore * safeFore - absDx * absDx - dy * dy));
      const forwardDepth = Math.max(0.03, dzSensor, dzSphere);
      signedDeg = (Math.atan2(signedDx, forwardDepth) * 180) / Math.PI;
    }

    const clampedSigned = Number(clamp(signedDeg, -65, 90).toFixed(1));
    // External rotation ROM is positive outward sweep (internal rotation < 0 reads as 0 for external-rotation ROM)
    const externalRomDeg = Number(clamp(Math.max(0, clampedSigned), 0, 90).toFixed(1));
    return {
      deg: externalRomDeg,
      signedDeg: clampedSigned,
      driftRatio,
      elbowFlexOk,
      elbowAngleDeg: elAngle,
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
    if (nose && lEar && rEar && lSh && rSh && (lEar.visibility ?? 1) >= 0.25 && (rEar.visibility ?? 1) >= 0.25) {
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
  } else if (jointMode === 'clasped_push') {
    // True bilateral forward elbow push: measures exact elbow extension angle (90 -> 170 deg)
    // gated smoothly by arms actually lifted into chest-push elevation (>= 45 deg shoulder flexion)
    const lElev = computeForwardFlexionWithForeshortening('left');
    const rElev = computeForwardFlexionWithForeshortening('right');
    const getElbowExt = (side: 'left' | 'right'): number | null => {
      const shI = side === 'left' ? POSE_INDEX.LEFT_SHOULDER : POSE_INDEX.RIGHT_SHOULDER;
      const elI = side === 'left' ? POSE_INDEX.LEFT_ELBOW : POSE_INDEX.RIGHT_ELBOW;
      const wrI = side === 'left' ? POSE_INDEX.LEFT_WRIST : POSE_INDEX.RIGHT_WRIST;
      const wSh = worldLandmarks?.[shI];
      const wEl = worldLandmarks?.[elI];
      const wWr = worldLandmarks?.[wrI];
      if (wSh && wEl && wWr && Math.hypot(wSh.x - wEl.x, wSh.y - wEl.y, (wSh.z ?? 0) - (wEl.z ?? 0)) >= 0.08) {
        return computeJointAngle3D(wSh, wEl, wWr);
      }
      if (landmarks[shI] && landmarks[elI] && landmarks[wrI]) {
        return computeJointAngle3D(landmarks[shI], landmarks[elI], landmarks[wrI]);
      }
      return null;
    };
    const lExt = getElbowExt('left');
    const rExt = getElbowExt('right');
    elbowAngleDeg =
      lExt !== null && rExt !== null ? Number((0.5 * (lExt + rExt)).toFixed(1)) : (lExt ?? rExt ?? null);

    const lLiftGate = lElev.angleDeg !== null ? clamp((lElev.angleDeg - 32) / 30, 0, 1) : 0;
    const rLiftGate = rElev.angleDeg !== null ? clamp((rElev.angleDeg - 32) / 30, 0, 1) : 0;
    const lPush = lExt !== null ? Number((lExt * lLiftGate).toFixed(1)) : null;
    const rPush = rExt !== null ? Number((rExt * rLiftGate).toFixed(1)) : null;
    leftArmRom = lPush;
    rightArmRom = rPush;

    if (exerciseSide === 'LEFT') {
      rawPrimaryRomDeg = lPush !== null ? Number(clamp(lPush, 0, 180).toFixed(1)) : null;
      oppositeRomDeg = rPush;
      xyzVector = lElev.xyzVector;
    } else if (exerciseSide === 'RIGHT') {
      rawPrimaryRomDeg = rPush !== null ? Number(clamp(rPush, 0, 180).toFixed(1)) : null;
      oppositeRomDeg = lPush;
      xyzVector = rElev.xyzVector;
    } else if (lPush !== null && rPush !== null) {
      rawPrimaryRomDeg = Number(clamp(0.5 * (lPush + rPush), 0, 180).toFixed(1));
      xyzVector = rElev.xyzVector ?? lElev.xyzVector;
    } else if (lPush !== null || rPush !== null) {
      rawPrimaryRomDeg = Number(clamp(lPush ?? rPush ?? 0, 0, 180).toFixed(1));
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
        inTargetZone: (rawPrimaryRomDeg ?? 0) >= targetRomGoal * 0.85,
        labelTh: 'ประสานมือดันเหยียดศอกไปด้านหน้า',
      };
    }
  } else if (jointMode === 'elbow') {
    const primaryRes = computeElbowFlexionRom(primarySide);
    const oppRes = computeElbowFlexionRom(oppositeSide);
    const lEl = primarySide === 'left' ? primaryRes.flexionDeg : oppRes.flexionDeg;
    const rEl = primarySide === 'right' ? primaryRes.flexionDeg : oppRes.flexionDeg;
    leftArmRom = lEl;
    rightArmRom = rEl;
    if (exerciseSide === 'BOTH') {
      if (lEl !== null && rEl !== null) {
        rawPrimaryRomDeg = Number((0.5 * (lEl + rEl)).toFixed(1));
      } else {
        rawPrimaryRomDeg = Number(((lEl ?? rEl ?? 0)).toFixed(1));
      }
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
    elbowAngleDeg = pRot.elbowAngleDeg;
    if (exerciseSide === 'BOTH') {
      if (lRot.deg !== null && rRot.deg !== null) {
        const bothSigned =
          lRot.signedDeg !== null && rRot.signedDeg !== null
            ? Number((0.5 * (lRot.signedDeg + rRot.signedDeg)).toFixed(1))
            : null;
        rawPrimaryRomDeg = Number((0.5 * (lRot.deg + rRot.deg)).toFixed(1));
        romDeg = rawPrimaryRomDeg;
        rotationTelemetry = {
          mode: 'shoulder_rotation',
          signedAngleDeg: bothSigned,
          postureReady: lRot.elbowFlexOk && rRot.elbowFlexOk,
        };
      } else {
        lowVisibilityAlertTh = 'กรุณาให้เห็นข้อศอกและข้อมือทั้ง 2 ข้างชัดเจนในกรอบกล้อง';
      }
    } else if (pRot.deg !== null && activeSh && activeEl && activeWr) {
      if (pRot.driftRatio > 0.48) {
        postureCompensated = true;
        postureBannerTh = 'กรุณาหนีบข้อศอกให้ชิดข้างลำตัวไว้ครับ';
      }
      rawPrimaryRomDeg = pRot.deg;
      romDeg = rawPrimaryRomDeg;
      rotationTelemetry = {
        mode: 'shoulder_rotation',
        signedAngleDeg: pRot.signedDeg,
        postureReady: pRot.elbowFlexOk,
      };
      const outDir = primarySide === 'left' ? 1 : -1;
      overlayHint = {
        pivotNorm: { x: activeEl.x, y: activeEl.y },
        targetNorm: {
          x: clamp(activeEl.x + outDir * baseline.forearmLen * 0.9, 0.05, 0.95),
          y: activeEl.y,
        },
        currentNorm: { x: activeWr.x, y: activeWr.y },
        inTargetZone: pRot.elbowFlexOk && rawPrimaryRomDeg >= targetRomGoal * 0.82,
        labelTh: `หมุนปลายแขนออกด้านข้าง (${rawPrimaryRomDeg.toFixed(0)}°)`,
      };
    } else {
      lowVisibilityAlertTh = 'กรุณาให้เห็นข้อศอกและข้อมือชัดเจนในกรอบกล้อง';
    }
  } else if (jointMode === 'wrist_finger') {
    if (activeHandBio && hasUsableHand21Landmarks(activeHand21)) {
      const w = activeHand21[HAND21_INDEX.WRIST];
      const midMcp = activeHand21[HAND21_INDEX.MIDDLE_MCP];
      const palmLen = Math.max(0.02, dist2D(w, midMcp));
      // True combined wrist dorsiflexion angle + 5-finger joint extension score (MCP/PIP/DIP)
      const fingerOpenDeg = (activeHandBio.meanFingerExtensionPct / 100) * 58;
      const wristExtContrib = activeHandBio.wristExtensionDeg * 0.42;
      rawPrimaryRomDeg = Number(clamp(fingerOpenDeg + wristExtContrib, 0, 90).toFixed(1));
      romDeg = rawPrimaryRomDeg;
      overlayHint = {
        pivotNorm: { x: w.x, y: w.y },
        targetNorm: { x: midMcp.x, y: Math.max(0.05, w.y - palmLen * 1.65) },
        currentNorm: {
          x: activeHand21[HAND21_INDEX.INDEX_TIP].x,
          y: activeHand21[HAND21_INDEX.INDEX_TIP].y,
        },
        inTargetZone: activeHandBio.extendedFingerCount >= 4 && rawPrimaryRomDeg >= targetRomGoal * 0.82,
        labelTh: `กางนิ้ว ${activeHandBio.extendedFingerCount}/5 นิ้ว • ข้อมือ ${activeHandBio.wristExtensionDeg.toFixed(0)}°`,
      };
    } else {
      const fallbackRes = computePoseDistalFallbackRom({
        landmarks,
        side: primarySide,
        mode: 'wrist_finger',
        shoulderSpanX,
        forearmLen: baseline.forearmLen,
        targetGoalDeg: targetRomGoal,
      });
      if (fallbackRes.deg !== null) {
        rawPrimaryRomDeg = fallbackRes.deg;
        romDeg = rawPrimaryRomDeg;
      } else {
        lowVisibilityAlertTh = 'กรุณายกมือและข้อมือเข้ามาให้กล้องมองเห็นชัดเจน';
      }
    }
  } else if (jointMode === 'forearm_turn') {
    if (activeHandBio && hasUsableHand21Landmarks(activeHand21)) {
      const signedRollDeg = activeHandBio.forearmRollSignedDeg;
      rawPrimaryRomDeg = Number(clamp(Math.abs(signedRollDeg), 0, 90).toFixed(1));
      romDeg = rawPrimaryRomDeg;
      rotationTelemetry = {
        mode: 'forearm_turn',
        signedAngleDeg: signedRollDeg,
        postureReady: true,
      };
      overlayHint = {
        pivotNorm: { x: activeHand21[HAND21_INDEX.WRIST].x, y: activeHand21[HAND21_INDEX.WRIST].y },
        targetNorm: {
          x: activeHand21[HAND21_INDEX.THUMB_MCP].x,
          y: activeHand21[HAND21_INDEX.THUMB_MCP].y,
        },
        inTargetZone: rawPrimaryRomDeg >= targetRomGoal * 0.80,
        labelTh: `พลิกข้อมือ: ${activeHandBio.palmOrientationTh} (${signedRollDeg >= 0 ? '+' : ''}${signedRollDeg.toFixed(0)}°)`,
      };
    } else {
      const fallbackRes = computePoseDistalFallbackRom({
        landmarks,
        side: primarySide,
        mode: 'forearm_turn',
        shoulderSpanX,
        forearmLen: baseline.forearmLen,
        targetGoalDeg: targetRomGoal,
      });
      if (fallbackRes.deg !== null) {
        rawPrimaryRomDeg = fallbackRes.deg;
        romDeg = rawPrimaryRomDeg;
        rotationTelemetry = {
          mode: 'forearm_turn',
          signedAngleDeg: fallbackRes.signedRollDeg ?? fallbackRes.deg,
          postureReady: true,
        };
      } else {
        lowVisibilityAlertTh = 'กรุณายกปลายแขนและข้อมือเข้ามาให้กล้องมองเห็นชัดเจน';
      }
    }
  } else if (jointMode === 'thumb_pinch') {
    if (activeHandBio && hasUsableHand21Landmarks(activeHand21)) {
      const thumbTip = activeHand21[HAND21_INDEX.THUMB_TIP];
      const targetTipIdx =
        activeHandBio.activePinchFinger === 'middle'
          ? HAND21_INDEX.MIDDLE_TIP
          : activeHandBio.activePinchFinger === 'ring'
            ? HAND21_INDEX.RING_TIP
            : activeHandBio.activePinchFinger === 'pinky'
              ? HAND21_INDEX.PINKY_TIP
              : HAND21_INDEX.INDEX_TIP;
      const targetTip = activeHand21[targetTipIdx];
      rawPrimaryRomDeg = activeHandBio.pinchAngleScoreDeg;
      romDeg = rawPrimaryRomDeg;
      overlayHint = {
        pivotNorm: { x: targetTip.x, y: targetTip.y },
        targetNorm: { x: (thumbTip.x + targetTip.x) / 2, y: (thumbTip.y + targetTip.y) / 2 },
        currentNorm: { x: thumbTip.x, y: thumbTip.y },
        inTargetZone: activeHandBio.activePinchFinger !== null && rawPrimaryRomDeg >= targetRomGoal * 0.82,
        labelTh: activeHandBio.activePinchFingerTh
          ? `จีบนิ้วโป้งแตะ${activeHandBio.activePinchFingerTh}`
          : 'จีบนิ้วโป้งแตะปลายนิ้ว',
      };
    } else {
      const fallbackRes = computePoseDistalFallbackRom({
        landmarks,
        side: primarySide,
        mode: 'thumb_pinch',
        shoulderSpanX,
        forearmLen: baseline.forearmLen,
        targetGoalDeg: targetRomGoal,
      });
      if (fallbackRes.deg !== null) {
        rawPrimaryRomDeg = fallbackRes.deg;
        romDeg = rawPrimaryRomDeg;
      } else {
        lowVisibilityAlertTh = 'กรุณายกมือระดับอกให้เห็นข้อมือชัดเจนในกรอบกล้อง';
      }
    }
  } else if (jointMode === 'neck') {
    if (lSh && rSh && (lSh.visibility ?? 0) >= UPPER_MIN_VIS && (rSh.visibility ?? 0) >= UPPER_MIN_VIS) {
      let tiltDeg = 0;
      let eyeTiltDeg = 0;
      const shRad = Math.atan2(rSh.y - lSh.y, lSh.x - rSh.x);
      if (lEye && rEye && (lEye.visibility ?? 0) >= 0.25 && (rEye.visibility ?? 0) >= 0.25) {
        const eyeRad = Math.atan2(rEye.y - lEye.y, lEye.x - rEye.x);
        let diffDeg = ((eyeRad - shRad) * 180) / Math.PI;
        while (diffDeg > 180) diffDeg -= 360;
        while (diffDeg < -180) diffDeg += 360;
        eyeTiltDeg = Math.abs(diffDeg);
        tiltDeg = eyeTiltDeg;
      }
      if (lEar && rEar && (lEar.visibility ?? 0) >= 0.25 && (rEar.visibility ?? 0) >= 0.25) {
        const earRad = Math.atan2(rEar.y - lEar.y, lEar.x - rEar.x);
        let earDiff = ((earRad - shRad) * 180) / Math.PI;
        while (earDiff > 180) earDiff -= 360;
        while (earDiff < -180) earDiff += 360;
        const earTiltDeg = Math.abs(earDiff);
        // Only blend ear-line when ears are genuinely consistent with eye-line (prevents occluded/flat ear from diluting true eye roll)
        if (eyeTiltDeg < 2) {
          tiltDeg = earTiltDeg;
        } else if (Math.abs(earTiltDeg - eyeTiltDeg) <= 10) {
          tiltDeg = 0.7 * eyeTiltDeg + 0.3 * earTiltDeg;
        }
      }
      if (tiltDeg < 2 && nose && (nose.visibility ?? 0) >= UPPER_MIN_VIS) {
        const neckMidX = (lSh.x + rSh.x) / 2;
        const neckMidY = (lSh.y + rSh.y) / 2;
        const lateralTiltRad = Math.atan2(
          (nose.x - neckMidX) * aspectScale,
          Math.max(0.05, neckMidY - nose.y),
        );
        tiltDeg = (Math.abs(lateralTiltRad) * 180) / Math.PI;
      }
      // Enforce directional isolation when staff/selector specifies LEFT or RIGHT neck tilt
      if (
        (exerciseSide === 'LEFT' || exerciseSide === 'RIGHT') &&
        nose &&
        (nose.visibility ?? 0) >= UPPER_MIN_VIS &&
        tiltDeg >= 6
      ) {
        const neckMidX = (lSh.x + rSh.x) / 2;
        const neckMidY = (lSh.y + rSh.y) / 2;
        const shAxisX = lSh.x - rSh.x;
        const shAxisY = lSh.y - rSh.y;
        const lateralProj = (nose.x - neckMidX) * shAxisX + (nose.y - neckMidY) * shAxisY;
        if (exerciseSide === 'LEFT' && lateralProj < -0.002) {
          tiltDeg = 0;
          wrongSideRaised = true;
          sideViolation = 'RIGHT_ARM_RAISED_IN_LEFT_MODE';
          wrongSideBannerTh = 'ท่านี้เอียงคอไปทางซ้าย กรุณาเอียงศีรษะไปข้างซ้ายครับ';
          sideViolationBannerTh = wrongSideBannerTh;
        } else if (exerciseSide === 'RIGHT' && lateralProj > 0.002) {
          tiltDeg = 0;
          wrongSideRaised = true;
          sideViolation = 'LEFT_ARM_RAISED_IN_RIGHT_MODE';
          wrongSideBannerTh = 'ท่านี้เอียงคอไปทางขวา กรุณาเอียงศีรษะไปข้างขวาครับ';
          sideViolationBannerTh = wrongSideBannerTh;
        }
      }
      // Exact true anatomical lateral cervical flexion angle (zero artificial gain!)
      rawPrimaryRomDeg = Number(clamp(tiltDeg, 0, 60).toFixed(1));
      romDeg = rawPrimaryRomDeg;
      if (nose) {
        overlayHint = {
          pivotNorm: { x: (lSh.x + rSh.x) / 2, y: (lSh.y + rSh.y) / 2 },
          targetNorm: { x: nose.x + (primarySide === 'left' ? 0.06 : -0.06), y: nose.y + 0.03 },
          inTargetZone: rawPrimaryRomDeg >= targetRomGoal * 0.82,
          labelTh: `เอียงคอ (${rawPrimaryRomDeg.toFixed(0)}°)`,
        };
      }
    }
  } else if (jointMode === 'neck_turn') {
    if (nose && (nose.visibility ?? 0) >= UPPER_MIN_VIS) {
      let signedOffsetX = 0;
      if (lEar && rEar && (lEar.visibility ?? 0) >= UPPER_MIN_VIS && (rEar.visibility ?? 0) >= UPPER_MIN_VIS) {
        const midEarX = (lEar.x + rEar.x) / 2;
        const earSpan = Math.max(0.04, dist2D(lEar, rEar));
        signedOffsetX = nose.x - midEarX;
        // Exact geometric head yaw via arcsin(noseOffset / halfEarSpan) fused with ear 3D depth yaw
        const sinRatio = clamp(Math.abs(signedOffsetX) / (earSpan * 0.5), 0, 1);
        const yawFromNose = (Math.asin(sinRatio) * 180) / Math.PI;
        const earDz = Math.abs((lEar.z ?? 0) - (rEar.z ?? 0));
        const yawFromDepth = (Math.atan2(earDz, Math.max(0.03, Math.abs(lEar.x - rEar.x))) * 180) / Math.PI;
        const exactYaw = earDz > 0.01 ? 0.7 * yawFromNose + 0.3 * yawFromDepth : yawFromNose;
        rawPrimaryRomDeg = Number(clamp(exactYaw, 0, 80).toFixed(1));
      } else if (lEye && rEye && (lEye.visibility ?? 0) >= 0.25 && (rEye.visibility ?? 0) >= 0.25) {
        const midEyeX = (lEye.x + rEye.x) / 2;
        const eyeSpan = Math.max(0.03, dist2D(lEye, rEye));
        signedOffsetX = nose.x - midEyeX;
        const sinRatio = clamp(Math.abs(signedOffsetX) / (eyeSpan * 0.82), 0, 1);
        rawPrimaryRomDeg = Number(clamp((Math.asin(sinRatio) * 180) / Math.PI, 0, 80).toFixed(1));
      } else if (lSh && rSh) {
        const midShX = (lSh.x + rSh.x) / 2;
        signedOffsetX = nose.x - midShX;
        const sinRatio = clamp(Math.abs(signedOffsetX) / (shoulderSpanX * 0.35), 0, 1);
        rawPrimaryRomDeg = Number(clamp((Math.asin(sinRatio) * 180) / Math.PI, 0, 80).toFixed(1));
      }
      // Enforce directional isolation when staff/selector explicitly specifies LEFT or RIGHT neck turn
      if (
        (exerciseSide === 'LEFT' || exerciseSide === 'RIGHT') &&
        rawPrimaryRomDeg !== null &&
        rawPrimaryRomDeg >= 12 &&
        lSh &&
        rSh
      ) {
        const leftDirSign = lSh.x >= rSh.x ? 1 : -1;
        const towardLeft = signedOffsetX * leftDirSign;
        if (exerciseSide === 'LEFT' && towardLeft < -0.008) {
          rawPrimaryRomDeg = 0;
          wrongSideRaised = true;
          sideViolation = 'RIGHT_ARM_RAISED_IN_LEFT_MODE';
          wrongSideBannerTh = 'ท่านี้หันศีรษะไปทางซ้าย กรุณาหันหน้าไปข้างซ้ายครับ';
          sideViolationBannerTh = wrongSideBannerTh;
        } else if (exerciseSide === 'RIGHT' && towardLeft > 0.008) {
          rawPrimaryRomDeg = 0;
          wrongSideRaised = true;
          sideViolation = 'LEFT_ARM_RAISED_IN_RIGHT_MODE';
          wrongSideBannerTh = 'ท่านี้หันศีรษะไปทางขวา กรุณาหันหน้าไปข้างขวาครับ';
          sideViolationBannerTh = wrongSideBannerTh;
        }
      }
      romDeg = rawPrimaryRomDeg;
      if (lSh && rSh) {
        const leftDirSign = lSh.x >= rSh.x ? 1 : -1;
        const targetDx =
          exerciseSide === 'LEFT'
            ? 0.08 * leftDirSign
            : exerciseSide === 'RIGHT'
              ? -0.08 * leftDirSign
              : 0.08;
        overlayHint = {
          pivotNorm: { x: (lSh.x + rSh.x) / 2, y: nose.y },
          targetNorm: { x: clamp(nose.x + targetDx, 0.1, 0.9), y: nose.y },
          inTargetZone: (rawPrimaryRomDeg ?? 0) >= targetRomGoal * 0.82,
          labelTh: `หันศีรษะ (${(rawPrimaryRomDeg ?? 0).toFixed(0)}°)`,
        };
      }
    }
  } else if (jointMode === 'shoulder_shrug') {
    if (lSh && rSh) {
      const curMidShY = (lSh.y + rSh.y) / 2;
      const curMidShZ = ((lSh.z ?? 0) + (rSh.z ?? 0)) / 2;
      const liftVsBaseline = Math.max(0, baseline.restingMidShoulderY - curMidShY) / Math.max(0.08, shoulderSpanX);
      const earGaps: number[] = [];
      if (lEar && (lEar.visibility ?? 1) >= 0.25) {
        earGaps.push(Math.max(0, lSh.y - lEar.y));
      }
      if (rEar && (rEar.visibility ?? 1) >= 0.25) {
        earGaps.push(Math.max(0, rSh.y - rEar.y));
      }
      const earLift =
        earGaps.length > 0
          ? Math.max(
              0,
              baseline.restingEarShoulderGapRatio -
                (earGaps.reduce((a, b) => a + b, 0) / earGaps.length) / Math.max(0.08, shoulderSpanX),
            )
          : 0;
      // Exact clavicle elevation angle (arctan of vertical shoulder lift over half-clavicle span)
      const exactShrugDeg = (Math.atan(Math.max(liftVsBaseline, earLift) * 2.1) * 180) / Math.PI;
      const rollDepthZ = curMidShZ - (baseline.restingMidShoulderZ ?? 0);
      const rollWidthSwing =
        (shoulderSpanX - baseline.shoulderWidth) / Math.max(0.08, baseline.shoulderWidth);
      const rollCombinedSignal = rollDepthZ + rollWidthSwing * 0.85;
      rawPrimaryRomDeg = Number(clamp(exactShrugDeg, 0, 65).toFixed(1));
      romDeg = rawPrimaryRomDeg;
      rotationTelemetry = {
        mode: 'shoulder_shrug',
        signedAngleDeg: rawPrimaryRomDeg,
        secondaryRollSignal: Number(rollCombinedSignal.toFixed(4)),
        postureReady: true,
      };
      overlayHint = {
        pivotNorm: { x: (lSh.x + rSh.x) / 2, y: curMidShY },
        targetNorm: { x: (lSh.x + rSh.x) / 2, y: Math.max(0.05, curMidShY - shoulderSpanX * 0.18) },
        inTargetZone: rawPrimaryRomDeg >= targetRomGoal * 0.80,
        labelTh: `ยักและหมุนหัวไหล่ (${rawPrimaryRomDeg.toFixed(0)}°)`,
      };
    }
  } else if (jointMode === 'bilateral_raise') {
    const lRes = computeForwardFlexionWithForeshortening('left');
    const rRes = computeForwardFlexionWithForeshortening('right');
    leftArmRom = lRes.angleDeg;
    rightArmRom = rRes.angleDeg;
    if (exerciseSide === 'LEFT') {
      rawPrimaryRomDeg = lRes.angleDeg !== null ? Number(lRes.angleDeg.toFixed(1)) : null;
      oppositeRomDeg = rRes.angleDeg;
      xyzVector = lRes.xyzVector;
    } else if (exerciseSide === 'RIGHT') {
      rawPrimaryRomDeg = rRes.angleDeg !== null ? Number(rRes.angleDeg.toFixed(1)) : null;
      oppositeRomDeg = lRes.angleDeg;
      xyzVector = rRes.xyzVector;
    } else if (lRes.angleDeg !== null && rRes.angleDeg !== null) {
      rawPrimaryRomDeg = Number((0.5 * (lRes.angleDeg + rRes.angleDeg)).toFixed(1));
      xyzVector = rRes.xyzVector ?? lRes.xyzVector;
    } else if (lRes.angleDeg !== null || rRes.angleDeg !== null) {
      rawPrimaryRomDeg = Number((lRes.angleDeg ?? rRes.angleDeg ?? 0).toFixed(1));
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
        inTargetZone: (rawPrimaryRomDeg ?? 0) >= targetRomGoal * 0.85,
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
    rawPrimaryRomDeg = pRes.angleDeg;
    oppositeRomDeg = oRes.angleDeg;
    leftArmRom = primarySide === 'left' ? pRes.angleDeg : oRes.angleDeg;
    rightArmRom = primarySide === 'right' ? pRes.angleDeg : oRes.angleDeg;
    xyzVector = pRes.xyzVector;
    elbowAngleDeg = pRes.elbowAngleDeg;

    if (rawPrimaryRomDeg !== null) {
      const forwardElevDeg =
        'forward3dDeg' in pRes && typeof pRes.forward3dDeg === 'number'
          ? pRes.forward3dDeg
          : rawPrimaryRomDeg;
      const effectivePlaneElevDeg = Math.max(rawPrimaryRomDeg, forwardElevDeg);
      if (expectedPlane === 'abduction' && effectivePlaneElevDeg >= 42 && activeSh) {
        const latRatio =
          'lateralDxRatio' in pRes && typeof pRes.lateralDxRatio === 'number'
            ? pRes.lateralDxRatio
            : 0;
        const distalRef = activeWr && (activeWr.visibility ?? 0) >= UPPER_MIN_VIS ? activeWr : activeEl;
        const notPureOverhead = !distalRef || distalRef.y >= activeSh.y - shoulderSpanX * 0.55;
        if (latRatio < 0.35 && (effectivePlaneElevDeg <= 145 || notPureOverhead)) {
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
        rawPrimaryRomDeg =
          leftArmRom >= 45 && rightArmRom >= 45
            ? Number((0.5 * (leftArmRom + rightArmRom)).toFixed(1))
            : Number(Math.min(leftArmRom, rightArmRom).toFixed(1));
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
  const isDistalFineMotorMode =
    jointMode === 'wrist_finger' ||
    jointMode === 'forearm_turn' ||
    jointMode === 'thumb_pinch';
  const isUpperLimbIsolatedExercise =
    jointMode !== 'neck' &&
    jointMode !== 'neck_turn' &&
    jointMode !== 'shoulder_shrug' &&
    targetJoint !== 'spine' &&
    targetJoint !== 'left_knee' &&
    targetJoint !== 'right_knee' &&
    targetJoint !== 'none';

  if (isUpperLimbIsolatedExercise) {
    if (leftArmRom === null && !isDistalFineMotorMode) {
      leftArmRom = computeForwardFlexionWithForeshortening('left').angleDeg;
    }
    if (rightArmRom === null && !isDistalFineMotorMode) {
      rightArmRom = computeForwardFlexionWithForeshortening('right').angleDeg;
    }
    if (oppositeRomDeg === null && !isDistalFineMotorMode) {
      oppositeRomDeg = primarySide === 'left' ? rightArmRom : leftArmRom;
    }

    const effectiveSideMode: 'LEFT' | 'RIGHT' | 'BOTH' =
      exerciseSide ??
      (jointMode === 'bilateral_raise' || jointMode === 'clasped_push'
        ? 'BOTH'
        : targetJoint === 'left_shoulder'
          ? 'LEFT'
          : 'RIGHT');

    if (isDistalFineMotorMode && effectiveSideMode === 'BOTH') {
      // Distal fine-motor exercises keep hand/wrist ROM intact instead of falsely triggering shoulder-lift BOTH check
      leftArmRom = rawPrimaryRomDeg;
      rightArmRom = rawPrimaryRomDeg;
    }

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
          'ท่านี้ฝึกเฉพาะข้างซ้าย! กรุณาลดแขนขวาลง และยกเฉพาะแขนซ้ายข้างเดียว';
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
          'ท่านี้ฝึกเฉพาะข้างขวา! กรุณาลดแขนซ้ายลง และยกเฉพาะแขนขวาข้างเดียว';
        wrongSideRaised = true;
        wrongSideBannerTh = sideViolationBannerTh;
        romDeg = 0;
      }
    } else if (effectiveSideMode === 'BOTH' && !isDistalFineMotorMode) {
      const lVal = leftArmRom ?? 0;
      const rVal = rightArmRom ?? 0;
      const oneArmRaisedOnly = (lVal >= 65 && rVal < 35) || (rVal >= 65 && lVal < 35);
      if (oneArmRaisedOnly) {
        sideViolation = 'SINGLE_ARM_IN_BOTH_MODE';
        sideViolationBannerTh =
          'ท่านี้ฝึกทั้งสองข้างพร้อมกัน กรุณายกแขนทั้ง 2 ข้างขึ้นพร้อมกันครับ';
        wrongSideRaised = true;
        wrongSideBannerTh = sideViolationBannerTh;
        romDeg = 0;
      }
    }
    if (wrongSideRaised && overlayHint) {
      overlayHint = { ...overlayHint, inTargetZone: false };
    }
  }

  // --- Elderly Compensation Checks (Exact clinical thresholds & user alerts) ---
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
    postureBannerTh = 'กรุณาตั้งลำตัวให้ตรง ไม่เอนตัวช่วยยกแขน';
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
    postureBannerTh = 'อย่ายักไหล่ช่วยครับ ผ่อนคลายหัวไหล่ลง';
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
    postureBannerTh = 'มองตรงข้างหน้าครับ ให้ยกมือขึ้นหาศีรษะแทนการก้มคอ';
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
    handBiomechanics:
      leftHandBio || rightHandBio
        ? {
            left: leftHandBio,
            right: rightHandBio,
            active: activeHandBio,
          }
        : null,
    rotationTelemetry,
  };
}

export const MANDATORY_LOWER_REST_STREAK_FRAMES = 3;
export const HOLD_COMPLETED_WAIT_LOWER_BANNER_TH =
  'ค้างครบเวลาแล้ว! ค่อยๆ เอาแขนลงเพื่อนับเป็น 1 ครั้งครับ';
export const PHASE_2_LOWER_ARM_PROMPT_BANNER_TH =
  'ค้างครบแล้ว! กรุณาเอาแขนลงข้างลำตัวเพื่อนับ +1 ครั้ง';
export const REP_COMPLETED_LOWER_ARM_BANNER_TH = PHASE_2_LOWER_ARM_PROMPT_BANNER_TH;
export const ARM_LOWERED_READY_NEXT_REP_BANNER_TH =
  'ยอดเยี่ยม! นับเพิ่ม +1 ครั้งเรียบร้อย';

export function formatRepScoredBannerTh(completedReps: number, targetReps: number): string {
  return `ได้ +1 แต้มแล้ว! (รวม ${completedReps}/${targetReps} ครั้ง) • กรุณาเอาแขนลงข้างลำตัวเพื่อเริ่มนับรอบถัดไป`;
}

export function formatWaitLowerAfterRepBankBannerTh(completedReps: number, targetReps: number): string {
  return `ได้ +1 แต้มแล้ว! (รวม ${completedReps}/${targetReps} ครั้ง) • กรุณาเอาแขนลงข้างลำตัวเพื่อเริ่มนับรอบถัดไป`;
}

/**
 * Checks anatomically whether active wrist(s) for arm exercises have physically dropped below
 * chest/shoulder level (`wrist.y > shoulder.y + 0.15 * shoulderWidth`).
 */
export function evaluateActiveWristsLowered(params: {
  landmarks: readonly PoseLandmark[] | null | undefined;
  exerciseSide?: 'LEFT' | 'RIGHT' | 'BOTH';
  targetJoint?: string;
  isSmallRomOrKnee?: boolean;
}): boolean {
  if (params.isSmallRomOrKnee) return true;
  const lms = params.landmarks;
  if (!lms || lms.length < 17) return true;
  const lSh = lms[POSE_INDEX.LEFT_SHOULDER];
  const rSh = lms[POSE_INDEX.RIGHT_SHOULDER];
  if (!lSh || !rSh || (lSh.visibility ?? 0) < 0.25 || (rSh.visibility ?? 0) < 0.25) {
    return true;
  }
  const shoulderWidth = Math.max(0.12, Math.hypot(rSh.x - lSh.x, rSh.y - lSh.y));
  const checkSide = (side: 'left' | 'right'): boolean => {
    const sh = side === 'left' ? lSh : rSh;
    const wrist = lms[side === 'left' ? POSE_INDEX.LEFT_WRIST : POSE_INDEX.RIGHT_WRIST];
    if (!wrist || (wrist.visibility ?? 0) < 0.45) return false;
    const thresholdY = sh.y + 0.15 * shoulderWidth;
    return wrist.y > thresholdY;
  };
  const side =
    params.exerciseSide ??
    (params.targetJoint === 'left_shoulder'
      ? 'LEFT'
      : params.targetJoint === 'right_shoulder'
        ? 'RIGHT'
        : undefined);
  if (side === 'BOTH') return checkSide('left') && checkSide('right');
  if (side === 'LEFT') return checkSide('left');
  if (side === 'RIGHT') return checkSide('right');
  return true;
}

/**
 * Option A Clinical State Machine (Immediate Guaranteed Rep Bank + Mandatory Arm-Lower Gate):
 * - **Phase 1 (`PHASE_1_HOLDING`)**: Accumulate hold time when `activeRom >= targetRom * 0.65`
 *   (or while arm is raised even if wrist temporarily exits camera edge). Brief dip below 0.65 decays slowly (`dt * 0.15`).
 *   NEVER resets timer on trunk lean, shoulder hike, or movement speed.
 * - **100% Hold Reached**: Immediately increments `completedReps += 1`, locks bar at `100%`,
 *   and transitions to `PHASE_2_WAIT_LOWER_TO_SCORE` (Wait-Lower Gate).
 * - **Phase 2 (`PHASE_2_WAIT_LOWER_TO_SCORE`)**: Holds `completedReps` locked (no auto-spam!) until
 *   the patient lowers arm(s) (`activeRom <= max(48, targetRom * 0.62)` or `wrist.y > sh.y + 0.15 * shW`)
 *   for 3 consecutive frames (or >= 1.2s for fine-motor wrist/finger exercises), then resets to `0.0s` for next rep!
 */
export function stepElderlyHoldAndRepStateMachine(
  input: ElderlyHoldRepStepInput,
): ElderlyHoldRepStepResult {
  const safeTargetHold = Math.max(0.5, input.targetHoldSec || 3.0);
  const safeTargetRom = Math.max(15, input.targetRomDeg || 110);
  const dt = Math.max(0, Math.min(0.25, input.dtSec));
  const rom = input.activeRomDeg;

  const next: ElderlyHoldRepState = {
    ...input.state,
    loweredFrameStreak: input.state.loweredFrameStreak ?? 0,
  };

  if (next.completedReps >= input.targetReps) {
    return {
      nextState: next,
      holdProgressPct: 100,
      holdRemainingSec: 0,
      justCompletedRep: false,
      bannerTh: 'ฝึกครบเซ็ตเรียบร้อยแล้ว เยี่ยมมากครับ!',
      statusKind: 'SET_FINISHED',
    };
  }

  const isPhase2WaitLower = next.phase === 'PHASE_2_WAIT_LOWER_TO_SCORE';

  // 1. Strict Side Isolation Gate (never wipe Phase 2 wait-lower lock, and never hard-zero accumulated Phase 1 hold)
  if (
    input.sideViolation !== 'NONE' &&
    !(isPhase2WaitLower && input.sideViolation === 'SINGLE_ARM_IN_BOTH_MODE')
  ) {
    if (isPhase2WaitLower) {
      next.holdElapsedSec = safeTargetHold;
      next.loweredFrameStreak = 0;
      return {
        nextState: next,
        holdProgressPct: 100,
        holdRemainingSec: 0,
        justCompletedRep: false,
        holdCompleteWaitLower: true,
        bannerTh: input.wrongSideBannerTh,
        statusKind: 'WRONG_LIMB',
      };
    }
    next.holdElapsedSec = Math.max(0, next.holdElapsedSec - dt * 0.15);
    next.loweredFrameStreak = 0;
    next.phase = 'PHASE_1_HOLDING';
    const pct = Math.min(100, (next.holdElapsedSec / safeTargetHold) * 100);
    const rem = Math.max(0, safeTargetHold - next.holdElapsedSec);
    return {
      nextState: next,
      holdProgressPct: Number(pct.toFixed(2)),
      holdRemainingSec: Number(rem.toFixed(1)),
      justCompletedRep: false,
      bannerTh: input.wrongSideBannerTh,
      statusKind: 'WRONG_LIMB',
    };
  }

  // 2. Wrong Movement Plane Violation -> Pause / slow-decay timer cleanly, preserve Wait-Lower lock
  if (input.wrongPlaneRaised) {
    if (isPhase2WaitLower) {
      next.holdElapsedSec = safeTargetHold;
      next.loweredFrameStreak = 0;
      return {
        nextState: next,
        holdProgressPct: 100,
        holdRemainingSec: 0,
        justCompletedRep: false,
        holdCompleteWaitLower: true,
        bannerTh: input.wrongPlaneBannerTh,
        statusKind: 'WRONG_PLANE',
      };
    }
    next.holdElapsedSec = Math.max(0, next.holdElapsedSec - dt * 0.15);
    next.loweredFrameStreak = 0;
    next.phase = 'PHASE_1_HOLDING';
    const pct = Math.min(100, (next.holdElapsedSec / safeTargetHold) * 100);
    const rem = Math.max(0, safeTargetHold - next.holdElapsedSec);
    return {
      nextState: next,
      holdProgressPct: Number(pct.toFixed(2)),
      holdRemainingSec: Number(rem.toFixed(1)),
      justCompletedRep: false,
      bannerTh: input.wrongPlaneBannerTh,
      statusKind: 'WRONG_PLANE',
    };
  }

  const isKneeFlex = Boolean(input.isKneeExercise && safeTargetRom < 140);
  const loweredRomThreshold = Math.min(Math.max(28, safeTargetRom * 0.50), safeTargetRom * 0.65);
  const rotationGate = input.rotationGate;
  const requiresRealRotation = Boolean(rotationGate?.required);
  const exCodeUpper = String(input.exerciseCode ?? '').toUpperCase();
  const isFineMotorTimed =
    !requiresRealRotation &&
    (exCodeUpper.includes('WRIST_FINGER') ||
      exCodeUpper.includes('THUMB_PINCH') ||
      exCodeUpper.includes('PINCH_BUTTON'));

  // 3. PHASE_2_WAIT_LOWER_TO_SCORE (Rep ALREADY banked! Lock at 100% until lowered/returned before counting next rep!)
  if (isPhase2WaitLower) {
    next.holdElapsedSec = safeTargetHold;
    const waitStarted = next.waitLowerStartedAtMs ?? input.nowMs;
    next.waitLowerStartedAtMs = waitStarted;

    let armsAtRestRom = false;
    if (requiresRealRotation) {
      armsAtRestRom = !input.blockCounting && Boolean(rotationGate?.returnedToStart);
    } else if (!input.blockCounting && !input.handsOutOfFrameRaised && rom !== null && Number.isFinite(rom)) {
      if (isKneeFlex) {
        armsAtRestRom = rom > safeTargetRom + 20;
      } else if (input.exerciseSide === 'BOTH') {
        const l = input.leftArmRomDeg;
        const r = input.rightArmRomDeg;
        if (l !== undefined && l !== null && r !== undefined && r !== null) {
          armsAtRestRom = l <= loweredRomThreshold && r <= loweredRomThreshold;
        } else if (l !== undefined && l !== null) {
          armsAtRestRom = l <= loweredRomThreshold;
        } else if (r !== undefined && r !== null) {
          armsAtRestRom = r <= loweredRomThreshold;
        } else {
          armsAtRestRom = rom <= loweredRomThreshold;
        }
      } else if (input.exerciseSide === 'LEFT') {
        const activeL = input.leftArmRomDeg ?? rom;
        armsAtRestRom = activeL <= loweredRomThreshold;
      } else if (input.exerciseSide === 'RIGHT') {
        const activeR = input.rightArmRomDeg ?? rom;
        armsAtRestRom = activeR <= loweredRomThreshold;
      } else {
        armsAtRestRom = rom <= loweredRomThreshold;
      }
    }

    const wristDroppedOk = !requiresRealRotation && !input.blockCounting && input.armWristLowered === true;
    const fineMotorTimedOut =
      !requiresRealRotation &&
      !input.blockCounting &&
      isFineMotorTimed &&
      input.nowMs - waitStarted >= 1200;

    if (
      !input.blockCounting &&
      !input.handsOutOfFrameRaised &&
      (armsAtRestRom || wristDroppedOk || fineMotorTimedOut)
    ) {
      next.loweredFrameStreak =
        fineMotorTimedOut || (requiresRealRotation && rotationGate?.returnedToStart)
          ? MANDATORY_LOWER_REST_STREAK_FRAMES
          : (next.loweredFrameStreak ?? 0) + 1;
      if (next.loweredFrameStreak >= MANDATORY_LOWER_REST_STREAK_FRAMES) {
        next.phase = 'PHASE_1_HOLDING';
        next.holdElapsedSec = 0;
        next.loweredFrameStreak = 0;
        next.waitLowerStartedAtMs = undefined;
        return {
          nextState: next,
          holdProgressPct: 0,
          holdRemainingSec: Number(safeTargetHold.toFixed(1)),
          justCompletedRep: false,
          unlockedToReady: true,
          bannerTh: null,
          statusKind: 'ACTIVE_REP',
        };
      }
    } else {
      next.loweredFrameStreak = 0;
    }

    return {
      nextState: next,
      holdProgressPct: 100,
      holdRemainingSec: 0,
      justCompletedRep: false,
      holdCompleteWaitLower: true,
      bannerTh:
        (requiresRealRotation && rotationGate?.guidanceBannerTh) ||
        formatWaitLowerAfterRepBankBannerTh(next.completedReps, input.targetReps),
      statusKind: 'HOLD_PHASE',
    };
  }

  if (input.blockCounting) {
    const pct = Math.min(100, (next.holdElapsedSec / safeTargetHold) * 100);
    const rem = Math.max(0, safeTargetHold - next.holdElapsedSec);
    return {
      nextState: next,
      holdProgressPct: Number(pct.toFixed(2)),
      holdRemainingSec: Number(rem.toFixed(1)),
      justCompletedRep: false,
      bannerTh: null,
      statusKind: next.holdElapsedSec > 0 ? 'HOLD_PHASE' : 'ACTIVE_REP',
    };
  }

  // 4. PHASE_1_HOLDING -> Exact Anatomical Angle Gate (>= 82% of target ROM) + Real Rotation Verification
  const romReachedExactTarget =
    Boolean(input.handsOutOfFrameRaised) ||
    (rom !== null &&
      Number.isFinite(rom) &&
      (isKneeFlex ? rom <= safeTargetRom + 12 : rom >= safeTargetRom * 0.82));
  const inTargetHoldZone = requiresRealRotation
    ? Boolean(rotationGate?.rotationVerified) && romReachedExactTarget
    : romReachedExactTarget;

  if (inTargetHoldZone) {
    next.phase = 'PHASE_1_HOLDING';
    next.readyBannerUntilMs = 0;
    next.holdElapsedSec = Math.min(safeTargetHold, next.holdElapsedSec + dt);
  } else if (next.holdElapsedSec > 0) {
    // Gentle slow decay (`dt * 0.15`) on brief dip — NEVER warp from 2.8s to 0.0s!
    next.holdElapsedSec = Math.max(0, next.holdElapsedSec - dt * 0.15);
    if (
      next.phase === 'PHASE_3_SCORED_READY' &&
      next.readyBannerUntilMs &&
      input.nowMs > next.readyBannerUntilMs
    ) {
      next.phase = 'PHASE_1_HOLDING';
    }
  } else if (next.phase === 'PHASE_3_SCORED_READY' && next.holdElapsedSec === 0) {
    next.phase = 'PHASE_1_HOLDING';
  }

  // 5. Hold reached 100% -> IMMEDIATELY bank +1 rep AND lock into Wait-Lower / Return-Rotation Gate!
  if (next.holdElapsedSec >= safeTargetHold - 1e-4) {
    next.holdElapsedSec = safeTargetHold;
    next.completedReps = Math.min(input.targetReps, next.completedReps + 1);
    next.phase = 'PHASE_2_WAIT_LOWER_TO_SCORE';
    next.loweredFrameStreak = 0;
    next.waitLowerStartedAtMs = input.nowMs;
    next.cooldownStartedAtMs = input.nowMs;
    next.repCelebrateUntilMs = input.nowMs + 2500;
    next.readyBannerUntilMs = 0;
    const bankBanner = formatWaitLowerAfterRepBankBannerTh(next.completedReps, input.targetReps);
    return {
      nextState: next,
      holdProgressPct: 100,
      holdRemainingSec: 0,
      justCompletedRep: true,
      holdCompleteWaitLower: true,
      bannerTh: bankBanner,
      statusKind: next.completedReps >= input.targetReps ? 'SET_FINISHED' : 'REP_COMPLETED',
    };
  }

  const pct = Math.min(100, (next.holdElapsedSec / safeTargetHold) * 100);
  const rem = Math.max(0, safeTargetHold - next.holdElapsedSec);
  const holdingBanner =
    inTargetHoldZone || next.holdElapsedSec > 0
      ? `ค้างท่าไว้ในองศาที่ถูกต้อง... (${next.holdElapsedSec.toFixed(1)} / ${safeTargetHold.toFixed(1)} วิ)`
      : requiresRealRotation && rotationGate?.guidanceBannerTh
        ? rotationGate.guidanceBannerTh
        : null;

  return {
    nextState: next,
    holdProgressPct: Number(pct.toFixed(2)),
    holdRemainingSec: Number(rem.toFixed(1)),
    justCompletedRep: false,
    bannerTh: holdingBanner,
    statusKind: inTargetHoldZone || next.holdElapsedSec > 0 ? 'HOLD_PHASE' : 'ACTIVE_REP',
  };
}
