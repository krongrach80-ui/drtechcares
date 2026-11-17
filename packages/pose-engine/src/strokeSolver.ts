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

    // Distance fallback when wrist curls straight toward shoulder in camera view
    const wrToShRatio = dist2D(wr, sh) / Math.max(0.10, armLen);
    const isUpwardCurl = wr.y <= el.y + shoulderSpanX * 0.20;
    const proxFlex = isUpwardCurl
      ? clamp((0.88 - wrToShRatio) / (0.88 - 0.25), 0, 1) * 140
      : 0;

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
  if (jointMode === 'cross_body') {
    if (activeSh && oppSh && activeWr && (activeWr.visibility ?? 0) >= UPPER_MIN_VIS) {
      const distToOpp = dist2D(activeWr, oppSh) / Math.max(0.10, shoulderSpanX);
      const reachScore = clamp((1.35 - distToOpp) / (1.35 - 0.18), 0, 1);
      rawPrimaryRomDeg = Number((reachScore * 115).toFixed(1));
      romDeg = rawPrimaryRomDeg;
      overlayHint = {
        pivotNorm: { x: activeSh.x, y: activeSh.y },
        targetNorm: { x: oppSh.x, y: oppSh.y },
        currentNorm: { x: activeWr.x, y: activeWr.y },
        inTargetZone: rawPrimaryRomDeg >= targetRomGoal * 0.82,
        labelTh: 'แตะไหล่ตรงข้าม',
      };
    } else {
      lowVisibilityAlertTh = 'กรุณาให้เห็นหัวไหล่ทั้ง 2 ข้างและมือในกรอบกล้อง';
    }
  } else if (jointMode === 'hand_to_head') {
    if (activeSh && activeWr && (activeWr.visibility ?? 0) >= UPPER_MIN_VIS) {
      const crownNorm = {
        x: nose ? nose.x : lSh && rSh ? (lSh.x + rSh.x) / 2 : activeSh.x,
        y: nose ? Math.max(0.05, nose.y - shoulderSpanX * 0.28) : Math.max(0.05, activeSh.y - shoulderSpanX * 0.85),
      };
      const distHeadRatio = dist2D(activeWr, crownNorm) / Math.max(0.10, baseline.upperArmLen);
      const proxFactor = clamp((1.65 - distHeadRatio) / 1.15, 0, 1);
      const liftRes = computeForwardFlexionWithForeshortening(primarySide);
      const combinedHeadDeg = clamp(
        0.55 * (liftRes.angleDeg ?? 0) + 0.45 * (proxFactor * 135),
        0,
        145,
      );
      rawPrimaryRomDeg = Number(combinedHeadDeg.toFixed(1));

      // Detect NECK_DROP compensation (> 15 deg head drop downward toward hand)
      if (nose && lEar && rEar && lSh && rSh) {
        const earMidY = (lEar.y + rEar.y) / 2;
        const eyeMidY = lEye && rEye ? (lEye.y + rEye.y) / 2 : earMidY;
        const dropTan = (nose.y - eyeMidY) / Math.max(0.05, shoulderSpanX * 0.35);
        const noseLowerDeg = Math.max(0, (Math.atan(dropTan) * 180) / Math.PI - 12);
        neckDropDeg = Number(noseLowerDeg.toFixed(1));
      }

      romDeg = rawPrimaryRomDeg;
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
    const res = computeForwardFlexionWithForeshortening(primarySide);
    elbowAngleDeg = res.elbowAngleDeg;
    xyzVector = res.xyzVector;
    if (res.angleDeg !== null && activeSh) {
      const wr = activeWr ?? activeEl;
      const depthBoost = wr ? Math.max(0, ((activeSh.z ?? 0) - (wr.z ?? 0)) * 38) : 0;
      rawPrimaryRomDeg = Number(clamp(res.angleDeg * 0.92 + depthBoost, 0, 115).toFixed(1));
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
    leftArmRom = lElev.angleDeg;
    rightArmRom = rElev.angleDeg;
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
    if (lElev.angleDeg !== null && rElev.angleDeg !== null) {
      rawPrimaryRomDeg = Number(clamp(0.5 * (lElev.angleDeg + rElev.angleDeg), 0, 145).toFixed(1));
      xyzVector = rElev.xyzVector ?? lElev.xyzVector;
    } else if (lElev.angleDeg !== null || rElev.angleDeg !== null) {
      rawPrimaryRomDeg = Number(clamp(lElev.angleDeg ?? rElev.angleDeg ?? 0, 0, 145).toFixed(1));
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
    rawPrimaryRomDeg = primaryRes.flexionDeg;
    oppositeRomDeg = oppRes.flexionDeg;
    leftArmRom =
      lRaiseRes.angleDeg !== null || oppRes.flexionDeg !== null || primaryRes.flexionDeg !== null
        ? Number(
            Math.max(
              lRaiseRes.angleDeg ?? 0,
              primarySide === 'left' ? (primaryRes.flexionDeg ?? 0) : (oppRes.flexionDeg ?? 0),
            ).toFixed(1),
          )
        : null;
    rightArmRom =
      rRaiseRes.angleDeg !== null || oppRes.flexionDeg !== null || primaryRes.flexionDeg !== null
        ? Number(
            Math.max(
              rRaiseRes.angleDeg ?? 0,
              primarySide === 'right' ? (primaryRes.flexionDeg ?? 0) : (oppRes.flexionDeg ?? 0),
            ).toFixed(1),
          )
        : null;
    xyzVector = primaryRes.xyzVector;
    romDeg = rawPrimaryRomDeg;
    if (activeSh && activeEl) {
      overlayHint = {
        pivotNorm: { x: activeEl.x, y: activeEl.y },
        targetNorm: { x: activeSh.x, y: activeSh.y + shoulderSpanX * 0.12 },
        currentNorm: activeWr ? { x: activeWr.x, y: activeWr.y } : undefined,
        inTargetZone: (rawPrimaryRomDeg ?? 0) >= targetRomGoal * 0.82,
        labelTh: 'พับศอกเข้าหาไหล่',
      };
    }
  } else if (jointMode === 'shoulder_rotation') {
    if (activeSh && activeEl && activeWr && (activeEl.visibility ?? 0) >= UPPER_MIN_VIS && (activeWr.visibility ?? 0) >= UPPER_MIN_VIS) {
      const elbowDriftRatio = Math.abs(activeEl.x - activeSh.x) / Math.max(0.08, shoulderSpanX);
      if (elbowDriftRatio > 0.45) {
        postureCompensated = true;
        postureBannerTh = 'กรุณาหนีบข้อศอกให้ชิดข้างลำตัวไว้ครับ';
      }
      const dx = Math.abs(activeWr.x - activeEl.x);
      const sweepRatio = clamp(dx / Math.max(0.07, baseline.forearmLen), 0, 1);
      const dz = Math.abs((activeWr.z ?? 0) - (activeEl.z ?? 0));
      const rotBySin = (Math.asin(sweepRatio) * 180) / Math.PI * 0.78;
      const rotByAtan = (Math.atan2(dx, Math.max(0.03, dz)) * 180) / Math.PI * 0.82;
      rawPrimaryRomDeg = Number(clamp(Math.max(rotBySin, rotByAtan), 0, 75).toFixed(1));
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
    if (activeHand21 && activeHand21.length >= 21) {
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
      const spreadScore = clamp((avgRatio - 0.95) / (1.85 - 0.95), 0, 1);
      const dorsiflexBonus = Math.max(0, (w.y - midMcp.y) / palmLen) * 28;
      rawPrimaryRomDeg = Number(clamp(spreadScore * 75 + dorsiflexBonus, 0, 88).toFixed(1));
      romDeg = rawPrimaryRomDeg;
      overlayHint = {
        pivotNorm: { x: w.x, y: w.y },
        targetNorm: { x: midMcp.x, y: Math.max(0.05, w.y - palmLen * 1.65) },
        currentNorm: { x: activeHand21[HAND21_INDEX.INDEX_TIP].x, y: activeHand21[HAND21_INDEX.INDEX_TIP].y },
        inTargetZone: rawPrimaryRomDeg >= targetRomGoal * 0.82,
        labelTh: 'กางนิ้วทั้ง 5 ออกกว้าง',
      };
    } else {
      // Fallback to Pose 33-pt hand keypoints when HandLandmarker isn't active
      const idxTip = landmarks[primarySide === 'left' ? POSE_INDEX.LEFT_INDEX : POSE_INDEX.RIGHT_INDEX];
      const pnkTip = landmarks[primarySide === 'left' ? POSE_INDEX.LEFT_PINKY : POSE_INDEX.RIGHT_PINKY];
      if (activeWr && (activeWr.visibility ?? 0) >= UPPER_MIN_VIS && idxTip) {
        const spreadSpan = pnkTip ? dist2D(idxTip, pnkTip) : 0;
        const liftY = Math.max(0, activeWr.y - idxTip.y);
        const deg = clamp(liftY * 520 + spreadSpan * 360, 0, 85);
        rawPrimaryRomDeg = Number(deg.toFixed(1));
        romDeg = rawPrimaryRomDeg;
      } else {
        lowVisibilityAlertTh = 'กรุณายกมือและข้อมือเข้ามาให้กล้องมองเห็นนิ้วชัดเจน';
      }
    }
  } else if (jointMode === 'forearm_turn') {
    if (activeHand21 && activeHand21.length >= 21) {
      const thumbCmc = activeHand21[HAND21_INDEX.THUMB_CMC];
      const pinkyMcp = activeHand21[HAND21_INDEX.PINKY_MCP];
      const palmSpan = Math.max(0.015, dist2D(thumbCmc, pinkyMcp));
      const horizFlipRatio = clamp(Math.abs(thumbCmc.x - pinkyMcp.x) / palmSpan, 0, 1);
      const deg = (Math.acos(1 - horizFlipRatio) * 180) / Math.PI;
      rawPrimaryRomDeg = Number(clamp(deg, 0, 90).toFixed(1));
      romDeg = rawPrimaryRomDeg;
      overlayHint = {
        pivotNorm: { x: activeHand21[0].x, y: activeHand21[0].y },
        targetNorm: { x: thumbCmc.x, y: thumbCmc.y },
        inTargetZone: rawPrimaryRomDeg >= targetRomGoal * 0.82,
        labelTh: 'พลิกหงาย-คว่ำฝ่ามือ',
      };
    } else {
      const thumb = landmarks[primarySide === 'left' ? POSE_INDEX.LEFT_THUMB : POSE_INDEX.RIGHT_THUMB];
      const pinky = landmarks[primarySide === 'left' ? POSE_INDEX.LEFT_PINKY : POSE_INDEX.RIGHT_PINKY];
      if (activeWr && thumb && pinky && (activeWr.visibility ?? 0) >= UPPER_MIN_VIS) {
        const dx = Math.abs(thumb.x - pinky.x);
        const dy = Math.abs(thumb.y - pinky.y);
        const dz = Math.abs((thumb.z ?? 0) - (pinky.z ?? 0));
        const palmW = Math.max(0.015, Math.hypot(dx, dy, dz));
        const supRatio = clamp((dx + dz * 0.6) / palmW, 0, 1);
        rawPrimaryRomDeg = Number(clamp(supRatio * 88, 0, 90).toFixed(1));
        romDeg = rawPrimaryRomDeg;
      } else {
        lowVisibilityAlertTh = 'กรุณายกปลายแขนและฝ่ามือเข้ามาให้กล้องมองเห็นนิ้วชัดเจน';
      }
    }
  } else if (jointMode === 'thumb_pinch') {
    if (activeHand21 && activeHand21.length >= 21) {
      const thumbTip = activeHand21[HAND21_INDEX.THUMB_TIP];
      const idxTip = activeHand21[HAND21_INDEX.INDEX_TIP];
      const midTip = activeHand21[HAND21_INDEX.MIDDLE_TIP];
      const palmW = Math.max(0.02, dist2D(activeHand21[HAND21_INDEX.INDEX_MCP], activeHand21[HAND21_INDEX.PINKY_MCP]));
      const pinchDist = Math.min(dist2D(thumbTip, idxTip), dist2D(thumbTip, midTip));
      const pinchRatio = pinchDist / palmW;
      const pinchScore = clamp((0.85 - pinchRatio) / (0.85 - 0.22), 0, 1);
      rawPrimaryRomDeg = Number((pinchScore * 68).toFixed(1));
      romDeg = rawPrimaryRomDeg;
      overlayHint = {
        pivotNorm: { x: idxTip.x, y: idxTip.y },
        targetNorm: { x: (thumbTip.x + idxTip.x) / 2, y: (thumbTip.y + idxTip.y) / 2 },
        currentNorm: { x: thumbTip.x, y: thumbTip.y },
        inTargetZone: rawPrimaryRomDeg >= targetRomGoal * 0.82,
        labelTh: 'จีบนิ้วโป้งแตะปลายนิ้ว',
      };
    } else {
      const thumb = landmarks[primarySide === 'left' ? POSE_INDEX.LEFT_THUMB : POSE_INDEX.RIGHT_THUMB];
      const idx = landmarks[primarySide === 'left' ? POSE_INDEX.LEFT_INDEX : POSE_INDEX.RIGHT_INDEX];
      if (activeWr && thumb && idx && (activeWr.visibility ?? 0) >= UPPER_MIN_VIS && (!activeSh || activeWr.y < activeSh.y + shoulderSpanX * 1.1)) {
        const handLen = Math.max(0.025, dist2D(idx, activeWr));
        const tipGapRatio = dist2D(thumb, idx) / handLen;
        const pinchScore = clamp((0.95 - tipGapRatio) / 0.70, 0, 1);
        rawPrimaryRomDeg = Number((pinchScore * 68).toFixed(1));
        romDeg = rawPrimaryRomDeg;
      } else {
        lowVisibilityAlertTh = 'กรุณายกมือระดับอกให้เห็นนิ้วโป้งและนิ้วชี้ชัดเจนในกรอบกล้อง';
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
      rawPrimaryRomDeg = Number(clamp(tiltDeg, 0, 60).toFixed(1));
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
      if (lEar && rEar && (lEar.visibility ?? 0) >= UPPER_MIN_VIS && (rEar.visibility ?? 0) >= UPPER_MIN_VIS) {
        const midEarX = (lEar.x + rEar.x) / 2;
        const earSpan = Math.max(0.04, dist2D(lEar, rEar));
        const yawRatio = Math.abs(nose.x - midEarX) / (earSpan * 0.5);
        rawPrimaryRomDeg = Number(clamp(yawRatio * 58, 0, 70).toFixed(1));
      } else if (lEye && rEye && (lEye.visibility ?? 0) >= 0.25 && (rEye.visibility ?? 0) >= 0.25) {
        const midEyeX = (lEye.x + rEye.x) / 2;
        const eyeSpan = Math.max(0.03, dist2D(lEye, rEye));
        rawPrimaryRomDeg = Number(clamp((Math.abs(nose.x - midEyeX) / (eyeSpan * 0.8)) * 55, 0, 70).toFixed(1));
      } else if (lSh && rSh) {
        const midShX = (lSh.x + rSh.x) / 2;
        rawPrimaryRomDeg = Number(clamp((Math.abs(nose.x - midShX) / (shoulderSpanX * 0.35)) * 55, 0, 70).toFixed(1));
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
      const shrugScore = Math.max(liftVsBaseline * 240, earLift * 210);
      rawPrimaryRomDeg = Number(clamp(shrugScore, 0, 65).toFixed(1));
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
    leftArmRom = lRes.angleDeg;
    rightArmRom = rRes.angleDeg;
    if (lRes.angleDeg !== null && rRes.angleDeg !== null) {
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
    rawPrimaryRomDeg = pRes.angleDeg;
    oppositeRomDeg = oRes.angleDeg;
    leftArmRom = primarySide === 'left' ? rawPrimaryRomDeg : oppositeRomDeg;
    rightArmRom = primarySide === 'right' ? rawPrimaryRomDeg : oppositeRomDeg;
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

    if (rawPrimaryRomDeg === null) {
      lowVisibilityAlertTh = 'มองเห็นข้อต่อแขนไม่ชัดเจน (< 0.70) กรุณาขยับให้อยู่ในกรอบกล้อง';
    }
    romDeg = wrongPlaneRaised ? 0 : rawPrimaryRomDeg;

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
        labelTh: expectedPlane === 'abduction' ? 'กางแขนถึงจุดเป้าหมาย' : 'ยกแขนตรงถึงเป้าหมาย',
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

    const isOppositeWristRaisedAboveChest = (side: 'left' | 'right'): boolean => {
      const shP = landmarks[side === 'left' ? POSE_INDEX.LEFT_SHOULDER : POSE_INDEX.RIGHT_SHOULDER];
      const wrP = landmarks[side === 'left' ? POSE_INDEX.LEFT_WRIST : POSE_INDEX.RIGHT_WRIST];
      const elP = landmarks[side === 'left' ? POSE_INDEX.LEFT_ELBOW : POSE_INDEX.RIGHT_ELBOW];
      if (!shP || !wrP || !elP || (wrP.visibility ?? 0) < UPPER_MIN_VIS || (shP.visibility ?? 0) < UPPER_MIN_VIS) {
        return false;
      }
      // Wrist elevated into chest/shoulder active zone while noticeably higher than neutral elbow hang
      const chestCeilingY = shP.y + shoulderSpanX * 0.45;
      return wrP.y < chestCeilingY && wrP.y < elP.y - 0.025;
    };

    if (effectiveSideMode === 'LEFT') {
      const rRom = rightArmRom ?? 0;
      const rightRaised = rRom > 35 || isOppositeWristRaisedAboveChest('right');
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
      const leftRaised = lRom > 35 || isOppositeWristRaisedAboveChest('left');
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
      const oneArmRaisedOnly = (lVal > 45 && rVal < 25) || (rVal > 45 && lVal < 25);
      const asymmetricSpread = Math.max(lVal, rVal) >= 35 && Math.abs(lVal - rVal) > 28;
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
 * Clinical 60FPS hysteresis hold & rep state machine:
 * - Zero stale closures (mutates/returns explicit state values)
 * - Soft hysteresis band prevents 1-frame tremor from wiping hold progress
 * - Immediate block & reset on side violations (`LEFT`/`RIGHT`/`BOTH`)
 * - Guaranteed `+1 rep` once `holdElapsedSec >= targetHoldSec` and auto-unlocks cooldown
 *   at `< 0.65 * targetRom` OR after `1.2s`.
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

  // 3. COOLDOWN phase -> Unlock smoothly when lowered < 0.65 * targetRom OR after 1.2s
  if (next.phase === 'COOLDOWN') {
    const cooldownElapsedMs = input.nowMs - next.cooldownStartedAtMs;
    const isKneeFlex = Boolean(input.isKneeExercise && safeTargetRom < 140);
    const unlockedByReturn =
      rom !== null &&
      (isKneeFlex ? rom > safeTargetRom + 28 : rom < safeTargetRom * 0.65);

    if (cooldownElapsedMs >= 1200 || (cooldownElapsedMs >= 350 && unlockedByReturn)) {
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
    // Freeze softly without wiping progress on 1-2 dropped pose frames
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

  // 4. Clinical Hysteresis Thresholds (exact tolerance for tremor while holding)
  const isKneeFlex = Boolean(input.isKneeExercise && safeTargetRom < 140);
  const inEnterZone = isKneeFlex ? rom <= safeTargetRom + 15 : rom >= safeTargetRom * 0.85;
  const inMaintainZone = isKneeFlex ? rom <= safeTargetRom + 22 : rom >= safeTargetRom * 0.75;
  const inGraceZone = isKneeFlex ? rom <= safeTargetRom + 35 : rom >= safeTargetRom * 0.60;
  const inResetZone = isKneeFlex ? rom >= 145 : rom < safeTargetRom * 0.50;

  if (input.isCompensatedNow) {
    // Pause hold timer cleanly during trunk lean / shoulder hike / neck drop without resetting
    const pct = Math.min(100, (next.holdElapsedSec / safeTargetHold) * 100);
    const rem = Math.max(0, safeTargetHold - next.holdElapsedSec);
    return {
      nextState: next,
      holdProgressPct: Number(pct.toFixed(2)),
      holdRemainingSec: Number(rem.toFixed(1)),
      justCompletedRep: false,
      bannerTh: input.postureBannerTh,
      statusKind: 'POSTURE_COMPENSATED',
    };
  }

  if (next.phase === 'WAIT_RAISE') {
    if (inEnterZone) {
      next.phase = 'HOLDING';
      next.holdElapsedSec = Math.min(safeTargetHold, next.holdElapsedSec + dt);
    } else if (next.holdElapsedSec > 0 && inGraceZone) {
      next.holdElapsedSec = Math.max(0, next.holdElapsedSec - dt * 0.4);
    } else if (inResetZone) {
      next.holdElapsedSec = 0;
    }
  } else if (next.phase === 'HOLDING') {
    if (inMaintainZone) {
      next.holdElapsedSec = Math.min(safeTargetHold, next.holdElapsedSec + dt);
    } else if (inGraceZone) {
      // Micro-dip grace zone (0.60..0.75 * targetRom): never wipe hold progress to 0!
      next.holdElapsedSec = Math.max(0, next.holdElapsedSec - dt * 0.4);
    } else if (inResetZone) {
      next.holdElapsedSec = 0;
      next.phase = 'WAIT_RAISE';
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
    bannerTh: null,
    statusKind: next.phase === 'HOLDING' ? 'HOLD_PHASE' : 'ACTIVE_REP',
  };
}
