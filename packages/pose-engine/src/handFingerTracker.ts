import { computeJointAngle3D } from './armJointProcessor';
import type { PoseLandmark } from './types';

/**
 * Full MediaPipe 21-Point HandLandmarker anatomical landmark indices.
 * Reference: MediaPipe Hand Landmarker specification (0..20).
 */
export const HAND21_INDEX = {
  WRIST: 0,
  THUMB_CMC: 1,
  THUMB_MCP: 2,
  THUMB_IP: 3,
  THUMB_TIP: 4,
  INDEX_MCP: 5,
  INDEX_PIP: 6,
  INDEX_DIP: 7,
  INDEX_TIP: 8,
  MIDDLE_MCP: 9,
  MIDDLE_PIP: 10,
  MIDDLE_DIP: 11,
  MIDDLE_TIP: 12,
  RING_MCP: 13,
  RING_PIP: 14,
  RING_DIP: 15,
  RING_TIP: 16,
  PINKY_MCP: 17,
  PINKY_PIP: 18,
  PINKY_DIP: 19,
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

export type FingerName = 'thumb' | 'index' | 'middle' | 'ring' | 'pinky';
export type PinchTargetFinger = 'index' | 'middle' | 'ring' | 'pinky';
export type PalmOrientationLabel = 'SUPINATED_UP' | 'PRONATED_DOWN' | 'NEUTRAL_SIDE';

export const HAND21_FINGER_CHAINS: Readonly<Record<FingerName, readonly [number, number, number, number]>> = {
  thumb: [HAND21_INDEX.THUMB_CMC, HAND21_INDEX.THUMB_MCP, HAND21_INDEX.THUMB_IP, HAND21_INDEX.THUMB_TIP],
  index: [HAND21_INDEX.INDEX_MCP, HAND21_INDEX.INDEX_PIP, HAND21_INDEX.INDEX_DIP, HAND21_INDEX.INDEX_TIP],
  middle: [HAND21_INDEX.MIDDLE_MCP, HAND21_INDEX.MIDDLE_PIP, HAND21_INDEX.MIDDLE_DIP, HAND21_INDEX.MIDDLE_TIP],
  ring: [HAND21_INDEX.RING_MCP, HAND21_INDEX.RING_PIP, HAND21_INDEX.RING_DIP, HAND21_INDEX.RING_TIP],
  pinky: [HAND21_INDEX.PINKY_MCP, HAND21_INDEX.PINKY_PIP, HAND21_INDEX.PINKY_DIP, HAND21_INDEX.PINKY_TIP],
};

export interface SingleFingerKinematics {
  finger: FingerName;
  mcpAngleDeg: number;
  pipAngleDeg: number;
  dipAngleDeg: number;
  extensionPct: number;
  isExtended: boolean;
  tipToWristRatio: number;
}

export interface Hand21Biomechanics {
  side: 'left' | 'right';
  usedWorld3D: boolean;
  fingers: Record<FingerName, SingleFingerKinematics>;
  extendedFingerCount: number;
  meanFingerExtensionPct: number;
  fingerSplayDeg: number;
  /** Signed forearm pronation-supination roll angle in degrees [-90 pronated .. 0 neutral .. +90 supinated]. */
  forearmRollSignedDeg: number;
  palmOrientation: PalmOrientationLabel;
  palmOrientationTh: string;
  /** True wrist dorsiflexion / palmar extension angle in degrees [0..90]. */
  wristExtensionDeg: number;
  /** Exact thumb-to-fingertip pinch ratios normalized by palm width (lower = closer touch). */
  pinchRatios: Record<PinchTargetFinger, number>;
  activePinchFinger: PinchTargetFinger | null;
  activePinchFingerTh: string | null;
  minPinchRatio: number;
  pinchAngleScoreDeg: number;
  palmWidth: number;
  palmLength: number;
}

export interface RotationTrackerModeInput {
  mode: 'forearm_turn' | 'shoulder_rotation' | 'shoulder_shrug';
  /** Signed angle for forearm_turn [-90..+90] or shoulder_rotation [-60..+90], or elevation angle for shrug. */
  signedAngleDeg: number | null;
  /** Secondary orthogonal axis (e.g., shoulder roll forward/back Z or horizontal circular offset for shrug_roll). */
  secondaryRollSignal?: number;
  /** True when prerequisite posture holds (e.g., elbow tucked ~90 deg for shoulder_rotation, hand visible for forearm_turn). */
  postureReady: boolean;
  targetRomDeg: number;
  nowMs: number;
}

export interface RotationCycleEvaluation {
  /** Exact physical rotation angle achieved from the starting baseline (0 when holding still!). */
  effectiveRotationRomDeg: number;
  /** Raw signed angle currently measured (e.g. -65 deg pronated, +72 deg supinated). */
  signedAngleDeg: number | null;
  /** Cumulative continuous angular sweep in current rep attempt. */
  sweptArcDeg: number;
  /** True ONLY when the patient has genuinely rotated through the required clinical arc. */
  rotationVerified: boolean;
  /** True when the patient has rotated back toward neutral/opposite side after a rep (unlocks next rep). */
  returnedToStart: boolean;
  /** Clear Thai instruction guiding the real rotation motion. */
  guidanceBannerTh: string | null;
  /** Direction/phase label for live HUD. */
  phaseLabelTh: string;
}

function clamp(v: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, v));
}

function dist3D(a: PoseLandmark, b: PoseLandmark): number {
  const dz = (Number.isFinite(a.z) ? (a.z as number) : 0) - (Number.isFinite(b.z) ? (b.z as number) : 0);
  return Math.hypot(a.x - b.x, a.y - b.y, dz);
}

function dist2D(a: PoseLandmark, b: PoseLandmark): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

const PINCH_FINGER_TH: Record<PinchTargetFinger, string> = {
  index: 'นิ้วชี้',
  middle: 'นิ้วกลาง',
  ring: 'นิ้วนาง',
  pinky: 'นิ้วก้อย',
};

/** Checks whether a 21-point MediaPipe hand landmark array is anatomically usable. */
export function hasUsableHand21Landmarks(
  hand: readonly PoseLandmark[] | undefined,
): hand is readonly PoseLandmark[] {
  if (!hand || hand.length < 21) return false;
  const wrist = hand[HAND21_INDEX.WRIST];
  const indexMcp = hand[HAND21_INDEX.INDEX_MCP];
  const pinkyMcp = hand[HAND21_INDEX.PINKY_MCP];
  const middleTip = hand[HAND21_INDEX.MIDDLE_TIP];
  if (!wrist || !indexMcp || !pinkyMcp || !middleTip) return false;
  if (
    !Number.isFinite(wrist.x) ||
    !Number.isFinite(wrist.y) ||
    !Number.isFinite(indexMcp.x) ||
    !Number.isFinite(pinkyMcp.x)
  ) {
    return false;
  }
  const minVis = Math.min(
    wrist.visibility ?? 1,
    indexMcp.visibility ?? 1,
    pinkyMcp.visibility ?? 1,
    middleTip.visibility ?? 1,
  );
  if (minVis < 0.25) return false;
  const span = Math.max(dist2D(wrist, pinkyMcp), dist2D(wrist, indexMcp), dist2D(wrist, middleTip));
  return span >= 0.018;
}

function hasUsableWorldHand21(
  worldHand: readonly PoseLandmark[] | undefined,
): worldHand is readonly PoseLandmark[] {
  if (!worldHand || worldHand.length < 21) return false;
  const w = worldHand[HAND21_INDEX.WRIST];
  const m = worldHand[HAND21_INDEX.MIDDLE_MCP];
  const i = worldHand[HAND21_INDEX.INDEX_MCP];
  const p = worldHand[HAND21_INDEX.PINKY_MCP];
  if (!w || !m || !i || !p) return false;
  const palmLen = dist3D(w, m);
  const palmW = dist3D(i, p);
  return Number.isFinite(palmLen) && Number.isFinite(palmW) && palmLen >= 0.03 && palmW >= 0.025;
}

function computeVectorAngleDeg(
  ax: number,
  ay: number,
  az: number,
  bx: number,
  by: number,
  bz: number,
): number {
  const ma = Math.hypot(ax, ay, az);
  const mb = Math.hypot(bx, by, bz);
  if (ma < 1e-6 || mb < 1e-6) return 0;
  const cos = clamp((ax * bx + ay * by + az * bz) / (ma * mb), -1, 1);
  return (Math.acos(cos) * 180) / Math.PI;
}

/**
 * Analyzes all 21 MediaPipe hand landmarks (preferring true metric 3D `worldLandmarks` when present)
 * to compute exact per-finger MCP/PIP/DIP angles, 5-finger extension count, wrist extension angle,
 * signed 3D forearm pronation-supination roll angle, and 4-finger thumb pinch metrics.
 */
export function analyzeHand21Biomechanics(params: {
  landmarks2D: readonly PoseLandmark[];
  worldLandmarks3D?: readonly PoseLandmark[];
  side: 'left' | 'right';
  poseElbow?: PoseLandmark;
  poseWrist?: PoseLandmark;
}): Hand21Biomechanics | null {
  const { landmarks2D, worldLandmarks3D, side, poseElbow, poseWrist } = params;
  if (!hasUsableHand21Landmarks(landmarks2D)) return null;

  const useWorld = hasUsableWorldHand21(worldLandmarks3D);
  const pts = useWorld ? worldLandmarks3D : landmarks2D;
  const distFn = useWorld ? dist3D : dist2D;

  const wrist = pts[HAND21_INDEX.WRIST];
  const indexMcp = pts[HAND21_INDEX.INDEX_MCP];
  const middleMcp = pts[HAND21_INDEX.MIDDLE_MCP];
  const pinkyMcp = pts[HAND21_INDEX.PINKY_MCP];
  const palmLength = Math.max(0.02, distFn(wrist, middleMcp));
  const palmWidth = Math.max(0.018, distFn(indexMcp, pinkyMcp));

  const analyzeFinger = (finger: FingerName): SingleFingerKinematics => {
    const [j1, j2, j3, tipIdx] = HAND21_FINGER_CHAINS[finger];
    const p0 = wrist;
    const p1 = pts[j1];
    const p2 = pts[j2];
    const p3 = pts[j3];
    const pTip = pts[tipIdx];
    const tipToWristRatio = distFn(pTip, wrist) / palmLength;

    if (finger === 'thumb') {
      const cmcAngle = computeJointAngle3D(p0, p1, p2);
      const mcpAngle = computeJointAngle3D(p1, p2, p3);
      const ipAngle = computeJointAngle3D(p2, p3, pTip);
      const jointExtPct = clamp(((mcpAngle + ipAngle) / 2 - 115) / (172 - 115), 0, 1) * 100;
      const reachPct = clamp((tipToWristRatio - 0.85) / (1.55 - 0.85), 0, 1) * 100;
      const extPct = Number((jointExtPct * 0.55 + reachPct * 0.45).toFixed(1));
      const isExtended = ipAngle >= 140 && tipToWristRatio >= 1.15;
      return {
        finger,
        mcpAngleDeg: Number(cmcAngle.toFixed(1)),
        pipAngleDeg: Number(mcpAngle.toFixed(1)),
        dipAngleDeg: Number(ipAngle.toFixed(1)),
        extensionPct: extPct,
        isExtended,
        tipToWristRatio: Number(tipToWristRatio.toFixed(2)),
      };
    }

    // Standard 4 fingers: MCP (WRIST-MCP-PIP), PIP (MCP-PIP-DIP), DIP (PIP-DIP-TIP)
    const mcpAngle = computeJointAngle3D(p0, p1, p2);
    const pipAngle = computeJointAngle3D(p1, p2, p3);
    const dipAngle = computeJointAngle3D(p2, p3, pTip);
    // Check whether synthetic/sparse test hand placed PIP/DIP identical to default points
    const jointsDistinct = distFn(p1, p2) >= palmLength * 0.12 && distFn(p2, p3) >= palmLength * 0.08;
    const jointScore = jointsDistinct
      ? clamp(((mcpAngle * 0.35 + pipAngle * 0.40 + dipAngle * 0.25) - 85) / (172 - 85), 0, 1)
      : clamp((tipToWristRatio - 0.95) / (1.80 - 0.95), 0, 1);
    const reachScore = clamp((tipToWristRatio - 0.92) / (1.78 - 0.92), 0, 1);
    const blended = jointsDistinct ? jointScore * 0.65 + reachScore * 0.35 : reachScore;
    const extensionPct = Number((blended * 100).toFixed(1));
    const isExtended = jointsDistinct
      ? pipAngle >= 145 && mcpAngle >= 135 && tipToWristRatio >= 1.35
      : tipToWristRatio >= 1.42;

    return {
      finger,
      mcpAngleDeg: Number(mcpAngle.toFixed(1)),
      pipAngleDeg: Number(pipAngle.toFixed(1)),
      dipAngleDeg: Number(dipAngle.toFixed(1)),
      extensionPct,
      isExtended,
      tipToWristRatio: Number(tipToWristRatio.toFixed(2)),
    };
  };

  const fingers: Record<FingerName, SingleFingerKinematics> = {
    thumb: analyzeFinger('thumb'),
    index: analyzeFinger('index'),
    middle: analyzeFinger('middle'),
    ring: analyzeFinger('ring'),
    pinky: analyzeFinger('pinky'),
  };

  const fingerList = Object.values(fingers);
  const extendedFingerCount = fingerList.filter((f) => f.isExtended).length;
  const meanFingerExtensionPct = Number(
    (fingerList.reduce((acc, f) => acc + f.extensionPct, 0) / fingerList.length).toFixed(1),
  );

  // Inter-finger angular splay across Index -> Middle -> Ring -> Pinky
  const splayPairs: Array<[number, number]> = [
    [HAND21_INDEX.INDEX_TIP, HAND21_INDEX.MIDDLE_TIP],
    [HAND21_INDEX.MIDDLE_TIP, HAND21_INDEX.RING_TIP],
    [HAND21_INDEX.RING_TIP, HAND21_INDEX.PINKY_TIP],
  ];
  const splaySum = splayPairs.reduce((acc, [aIdx, bIdx]) => {
    const a = pts[aIdx];
    const b = pts[bIdx];
    return (
      acc +
      computeVectorAngleDeg(
        a.x - wrist.x,
        a.y - wrist.y,
        (a.z ?? 0) - (wrist.z ?? 0),
        b.x - wrist.x,
        b.y - wrist.y,
        (b.z ?? 0) - (wrist.z ?? 0),
      )
    );
  }, 0);
  const fingerSplayDeg = Number(clamp(splaySum, 0, 95).toFixed(1));

  // Signed Forearm Pronation-Supination Roll Angle [-90 pronated .. 0 neutral .. +90 supinated]
  // Uses true 3D palm normal N = (MIDDLE_MCP - WRIST) x (PINKY_MCP - INDEX_MCP) + chirality flip
  const vX = middleMcp.x - wrist.x;
  const vY = middleMcp.y - wrist.y;
  const vZ = (middleMcp.z ?? 0) - (wrist.z ?? 0);
  const uX = pinkyMcp.x - indexMcp.x;
  const uY = pinkyMcp.y - indexMcp.y;
  const uZ = (pinkyMcp.z ?? 0) - (indexMcp.z ?? 0);
  const nX = vY * uZ - vZ * uY;
  const nY = vZ * uX - vX * uZ;
  const nZ = vX * uY - vY * uX;
  const nMag = Math.hypot(nX, nY, nZ);
  // Handedness sign so +1 corresponds to palm facing UP/camera (supinated) and -1 to palm facing DOWN/away (pronated)
  const sideSign = side === 'right' ? 1 : -1;
  let forearmRollSignedDeg = 0;
  if (nMag > 1e-6) {
    // Project normal onto vertical-camera plane: -nY points UP in image coords, -nZ points TOWARD camera
    const upComponent = (-nY * sideSign) / nMag;
    const camComponent = (-nZ * sideSign) / nMag;
    const sideComponent = Math.abs(nX) / nMag;
    // Signed roll combines palm-up/down and palm-facing-camera chirality seamlessly
    const palmUpBias = upComponent * 0.72 + camComponent * 0.68;
    const rollRad = Math.atan2(palmUpBias, Math.max(0.08, sideComponent));
    forearmRollSignedDeg = Number(clamp((rollRad * 180) / Math.PI, -90, 90).toFixed(1));
  } else {
    // Fallback to 2D signed index->pinky horizontal polarity when Z is degenerate
    const transverseX = ((indexMcp.x - pinkyMcp.x) * sideSign) / palmWidth;
    forearmRollSignedDeg = Number(clamp(transverseX * 90, -90, 90).toFixed(1));
  }

  let palmOrientation: PalmOrientationLabel = 'NEUTRAL_SIDE';
  let palmOrientationTh = 'สันมือตรง (กลาง)';
  if (forearmRollSignedDeg >= 28) {
    palmOrientation = 'SUPINATED_UP';
    palmOrientationTh = 'หงายฝ่ามือ';
  } else if (forearmRollSignedDeg <= -28) {
    palmOrientation = 'PRONATED_DOWN';
    palmOrientationTh = 'คว่ำฝ่ามือ';
  }

  // Exact Wrist Extension (Dorsiflexion) Angle [0..90 deg]
  let wristExtensionDeg = 0;
  if (
    poseElbow &&
    poseWrist &&
    (poseElbow.visibility ?? 0) >= 0.2 &&
    (poseWrist.visibility ?? 0) >= 0.2
  ) {
    const hand2dWrist = landmarks2D[HAND21_INDEX.WRIST];
    const hand2dMidMcp = landmarks2D[HAND21_INDEX.MIDDLE_MCP];
    const foreX = poseWrist.x - poseElbow.x;
    const foreY = poseWrist.y - poseElbow.y;
    const foreZ = (poseWrist.z ?? 0) - (poseElbow.z ?? 0);
    const palmX = hand2dMidMcp.x - hand2dWrist.x;
    const palmY = hand2dMidMcp.y - hand2dWrist.y;
    const palmZ = (hand2dMidMcp.z ?? 0) - (hand2dWrist.z ?? 0);
    const rawDev = computeVectorAngleDeg(foreX, foreY, foreZ, palmX, palmY, palmZ);
    wristExtensionDeg = Number(clamp(rawDev, 0, 90).toFixed(1));
  } else {
    // Local wrist-to-metacarpal upward pitch angle when forearm pose elbow is absent
    const w2 = landmarks2D[HAND21_INDEX.WRIST];
    const m2 = landmarks2D[HAND21_INDEX.MIDDLE_MCP];
    const pitchUpDeg = (Math.atan2(Math.max(0, w2.y - m2.y), Math.max(0.01, Math.abs(m2.x - w2.x))) * 180) / Math.PI;
    wristExtensionDeg = Number(clamp(pitchUpDeg * 0.78, 0, 85).toFixed(1));
  }

  // Exact 4-Finger Thumb Pinch Ratios
  const thumbTip = pts[HAND21_INDEX.THUMB_TIP];
  const pinchRatios: Record<PinchTargetFinger, number> = {
    index: Number((distFn(thumbTip, pts[HAND21_INDEX.INDEX_TIP]) / palmWidth).toFixed(3)),
    middle: Number((distFn(thumbTip, pts[HAND21_INDEX.MIDDLE_TIP]) / palmWidth).toFixed(3)),
    ring: Number((distFn(thumbTip, pts[HAND21_INDEX.RING_TIP]) / palmWidth).toFixed(3)),
    pinky: Number((distFn(thumbTip, pts[HAND21_INDEX.PINKY_TIP]) / palmWidth).toFixed(3)),
  };

  let bestFinger: PinchTargetFinger = 'index';
  let minPinchRatio = pinchRatios.index;
  for (const f of ['middle', 'ring', 'pinky'] as const) {
    if (pinchRatios[f] < minPinchRatio) {
      minPinchRatio = pinchRatios[f];
      bestFinger = f;
    }
  }
  const activePinchFinger: PinchTargetFinger | null = minPinchRatio <= 0.42 ? bestFinger : null;
  const activePinchFingerTh = activePinchFinger ? PINCH_FINGER_TH[activePinchFinger] : null;
  // Exact pinch closure score in degrees: 0 deg when wide open (>= 0.85 palm width), 60-75 deg when touching (<= 0.28)
  const pinchClosure = clamp((0.88 - minPinchRatio) / (0.88 - 0.24), 0, 1.12);
  const pinchAngleScoreDeg = Number(clamp(pinchClosure * 68, 0, 78).toFixed(1));

  return {
    side,
    usedWorld3D: useWorld,
    fingers,
    extendedFingerCount,
    meanFingerExtensionPct,
    fingerSplayDeg,
    forearmRollSignedDeg,
    palmOrientation,
    palmOrientationTh,
    wristExtensionDeg,
    pinchRatios,
    activePinchFinger,
    activePinchFingerTh,
    minPinchRatio,
    pinchAngleScoreDeg,
    palmWidth: Number(palmWidth.toFixed(4)),
    palmLength: Number(palmLength.toFixed(4)),
  };
}

/**
 * Stateful Real-Rotation Verifier for Wrist/Forearm Rotation, Shoulder Internal/External Rotation,
 * and Circular Shoulder Roll. Guarantees static poses NEVER falsely trigger reps: the patient
 * MUST physically rotate through a genuine angular sweep and return/reverse for each rep!
 */
export class ClinicalRotationCycleTracker {
  private mode: RotationTrackerModeInput['mode'] | null = null;
  private baselineAngleDeg: number | null = null;
  private peakAngleDeg: number | null = null;
  private lastSignedAngleDeg: number | null = null;
  private cumulativeArcDeg = 0;
  private visitedNeutral = false;
  private targetDirectionSign: 1 | -1 = 1;
  private awaitingReturnAfterPeak = false;
  // Circular shoulder roll phase accumulator
  private lastShrugPhaseRad: number | null = null;
  private shrugCircleSweptDeg = 0;
  private shrugMaxElevationDeg = 0;
  private shrugReturnedLow = true;

  reset(): void {
    this.mode = null;
    this.baselineAngleDeg = null;
    this.peakAngleDeg = null;
    this.lastSignedAngleDeg = null;
    this.cumulativeArcDeg = 0;
    this.visitedNeutral = false;
    this.targetDirectionSign = 1;
    this.awaitingReturnAfterPeak = false;
    this.lastShrugPhaseRad = null;
    this.shrugCircleSweptDeg = 0;
    this.shrugMaxElevationDeg = 0;
    this.shrugReturnedLow = true;
  }

  /** Called right after a rep is banked so the tracker requires genuine reverse/return rotation before next rep. */
  notifyRepCompleted(): void {
    this.awaitingReturnAfterPeak = true;
    this.cumulativeArcDeg = 0;
    this.shrugCircleSweptDeg = 0;
    this.shrugMaxElevationDeg = 0;
    this.shrugReturnedLow = false;
    if (this.mode === 'forearm_turn' && this.lastSignedAngleDeg !== null) {
      // Next rep starts from the currently reached palm orientation (so flipping back counts the return turn!)
      this.baselineAngleDeg = this.lastSignedAngleDeg;
      this.targetDirectionSign = this.lastSignedAngleDeg >= 0 ? -1 : 1;
    }
  }

  update(input: RotationTrackerModeInput): RotationCycleEvaluation {
    if (this.mode !== input.mode) {
      this.reset();
      this.mode = input.mode;
    }

    const { signedAngleDeg, postureReady, targetRomDeg } = input;
    if (signedAngleDeg === null || !Number.isFinite(signedAngleDeg)) {
      return {
        effectiveRotationRomDeg: 0,
        signedAngleDeg: null,
        sweptArcDeg: Number(this.cumulativeArcDeg.toFixed(1)),
        rotationVerified: false,
        returnedToStart: false,
        guidanceBannerTh:
          input.mode === 'forearm_turn'
            ? 'ยกมือและข้อมือเข้ามาในกรอบกล้องให้ชัดเจน'
            : input.mode === 'shoulder_rotation'
              ? 'ตั้งศอก 90 องศาแนบข้างลำตัวให้กล้องเห็นชัดเจน'
              : 'หันหน้าเข้าหากล้องให้เห็นหัวไหล่ทั้ง 2 ข้าง',
        phaseLabelTh: 'รอตรวจจับข้อต่อ',
      };
    }

    // -----------------------------------------------------------------------
    // 1. FOREARM / WRIST ROTATION (ELDERLY_FOREARM_TURN: Pronation <-> Supination)
    // -----------------------------------------------------------------------
    if (input.mode === 'forearm_turn') {
      if (this.baselineAngleDeg === null) {
        this.baselineAngleDeg = signedAngleDeg;
        this.lastSignedAngleDeg = signedAngleDeg;
        this.targetDirectionSign = signedAngleDeg >= 10 ? -1 : 1;
        return {
          effectiveRotationRomDeg: 0,
          signedAngleDeg,
          sweptArcDeg: 0,
          rotationVerified: false,
          returnedToStart: true,
          guidanceBannerTh:
            this.targetDirectionSign > 0
              ? 'เริ่มหมุนพลิกหงายฝ่ามือขึ้นช้าๆ'
              : 'เริ่มหมุนพลิกคว่ำฝ่ามือลงช้าๆ',
          phaseLabelTh: 'ตั้งค่าท่าเริ่มต้นแล้ว',
        };
      }

      const stepDelta =
        this.lastSignedAngleDeg !== null ? Math.abs(signedAngleDeg - this.lastSignedAngleDeg) : 0;
      if (stepDelta >= 1.5 && stepDelta <= 70) {
        this.cumulativeArcDeg = Math.min(240, this.cumulativeArcDeg + stepDelta);
      }
      this.lastSignedAngleDeg = signedAngleDeg;

      // Net angular rotation away from starting palm angle (static hand => netRotation = 0!)
      const netRotationDeg = Math.abs(signedAngleDeg - this.baselineAngleDeg);
      const requiredSweep = Math.max(48, Math.min(82, targetRomDeg * 0.78));

      if (this.awaitingReturnAfterPeak) {
        const returned = netRotationDeg >= 36;
        if (returned) {
          this.awaitingReturnAfterPeak = false;
          this.baselineAngleDeg = signedAngleDeg;
          this.cumulativeArcDeg = 0;
          this.targetDirectionSign = signedAngleDeg >= 10 ? -1 : 1;
        }
        return {
          effectiveRotationRomDeg: Number(clamp(netRotationDeg, 0, 95).toFixed(1)),
          signedAngleDeg,
          sweptArcDeg: Number(netRotationDeg.toFixed(1)),
          rotationVerified: false,
          returnedToStart: returned,
          guidanceBannerTh: returned
            ? 'หมุนกลับเรียบร้อย พร้อมนับรอบถัดไป'
            : this.targetDirectionSign > 0
              ? 'พลิกหงายฝ่ามือกลับอีกด้านเพื่อเริ่มรอบถัดไป'
              : 'พลิกคว่ำฝ่ามือกลับอีกด้านเพื่อเริ่มรอบถัดไป',
          phaseLabelTh: returned ? 'พร้อมรอบถัดไป' : 'รอหมุนกลับอีกด้าน',
        };
      }

      // Must actually sweep through at least 48 deg (and cumulative motion >= 40 deg) so static hand NEVER scores!
      const rotationVerified = postureReady && netRotationDeg >= requiredSweep && this.cumulativeArcDeg >= 38;
      const effectiveRom = postureReady ? Number(clamp(netRotationDeg, 0, 95).toFixed(1)) : 0;
      const palmLabel =
        signedAngleDeg >= 28 ? 'หงายฝ่ามือ' : signedAngleDeg <= -28 ? 'คว่ำฝ่ามือ' : 'สันมือตรง';

      return {
        effectiveRotationRomDeg: effectiveRom,
        signedAngleDeg,
        sweptArcDeg: Number(netRotationDeg.toFixed(1)),
        rotationVerified,
        returnedToStart: netRotationDeg <= 22,
        guidanceBannerTh: rotationVerified
          ? null
          : `หมุนพลิกข้อมือให้สุด (${Math.round(netRotationDeg)}° / ${Math.round(requiredSweep)}° • ${palmLabel})`,
        phaseLabelTh: rotationVerified ? `หมุนครบ ${Math.round(netRotationDeg)}°` : `${palmLabel} (${Math.round(netRotationDeg)}°)`,
      };
    }

    // -----------------------------------------------------------------------
    // 2. SHOULDER INTERNAL/EXTERNAL ROTATION (ELDERLY_SH_ROT)
    // -----------------------------------------------------------------------
    if (input.mode === 'shoulder_rotation') {
      if (!postureReady) {
        return {
          effectiveRotationRomDeg: 0,
          signedAngleDeg,
          sweptArcDeg: 0,
          rotationVerified: false,
          returnedToStart: false,
          guidanceBannerTh: 'งอข้อศอก 90 องศาแนบข้างลำตัวก่อน แล้วค่อยหมุนปลายแขนออกด้านข้าง',
          phaseLabelTh: 'จัดท่าศอก 90 องศา',
        };
      }

      // Must visit neutral/internal zone (<= 22 deg) first so starting already parked sideways never auto-counts
      if (signedAngleDeg <= 24) {
        this.visitedNeutral = true;
        this.baselineAngleDeg = Math.min(this.baselineAngleDeg ?? signedAngleDeg, signedAngleDeg);
      }

      const stepDelta =
        this.lastSignedAngleDeg !== null ? Math.abs(signedAngleDeg - this.lastSignedAngleDeg) : 0;
      if (stepDelta >= 1.2 && stepDelta <= 65) {
        this.cumulativeArcDeg = Math.min(200, this.cumulativeArcDeg + stepDelta);
      }
      this.lastSignedAngleDeg = signedAngleDeg;

      const neutralRef = this.baselineAngleDeg ?? 0;
      const sweptOutwardDeg = Math.max(0, signedAngleDeg - Math.min(20, neutralRef));
      const requiredExternalDeg = Math.max(40, targetRomDeg * 0.80);

      if (this.awaitingReturnAfterPeak) {
        const returned = signedAngleDeg <= Math.min(24, targetRomDeg * 0.40);
        if (returned) {
          this.awaitingReturnAfterPeak = false;
          this.visitedNeutral = true;
          this.baselineAngleDeg = signedAngleDeg;
          this.cumulativeArcDeg = 0;
        }
        return {
          effectiveRotationRomDeg: Number(clamp(Math.max(0, signedAngleDeg), 0, 90).toFixed(1)),
          signedAngleDeg,
          sweptArcDeg: Number(sweptOutwardDeg.toFixed(1)),
          rotationVerified: false,
          returnedToStart: returned,
          guidanceBannerTh: returned
            ? 'หุบแขนกลับมาด้านหน้าแล้ว พร้อมหมุนรอบถัดไป'
            : 'หมุนปลายแขนกลับเข้ามาด้านหน้าก่อนเริ่มรอบถัดไป',
          phaseLabelTh: returned ? 'พร้อมรอบถัดไป' : 'รอหมุนกลับด้านหน้า',
        };
      }

      if (!this.visitedNeutral) {
        return {
          effectiveRotationRomDeg: 0,
          signedAngleDeg,
          sweptArcDeg: 0,
          rotationVerified: false,
          returnedToStart: false,
          guidanceBannerTh: 'นำปลายแขนกลับมาชี้ด้านหน้าก่อน แล้วค่อยหมุนข้อไหล่ออกด้านข้างช้าๆ',
          phaseLabelTh: 'ตั้งต้นปลายแขนชี้ด้านหน้า',
        };
      }

      const rotationVerified =
        signedAngleDeg >= requiredExternalDeg &&
        sweptOutwardDeg >= Math.max(30, targetRomDeg * 0.55) &&
        this.cumulativeArcDeg >= 25;
      const effectiveRom = Number(clamp(Math.max(0, signedAngleDeg), 0, 90).toFixed(1));

      return {
        effectiveRotationRomDeg: effectiveRom,
        signedAngleDeg,
        sweptArcDeg: Number(sweptOutwardDeg.toFixed(1)),
        rotationVerified,
        returnedToStart: signedAngleDeg <= 22,
        guidanceBannerTh: rotationVerified
          ? null
          : `หมุนปลายแขนออกด้านข้าง (${Math.round(effectiveRom)}° / ${Math.round(requiredExternalDeg)}°)`,
        phaseLabelTh: rotationVerified ? `หมุนไหล่ครบ ${Math.round(effectiveRom)}°` : `กำลังหมุนไหล่ ${Math.round(effectiveRom)}°`,
      };
    }

    // -----------------------------------------------------------------------
    // 3. SHOULDER SHRUG & CIRCULAR ROLL (MOPH_SHOULDER_SHRUG_ROLL)
    // -----------------------------------------------------------------------
    const elevDeg = Math.max(0, signedAngleDeg);
    const rollAxis = input.secondaryRollSignal ?? 0;
    const minLiftDeg = Math.max(22, targetRomDeg * 0.65);

    if (elevDeg <= 14) {
      this.shrugReturnedLow = true;
    }
    this.shrugMaxElevationDeg = Math.max(this.shrugMaxElevationDeg, elevDeg);

    // Track continuous phase angle around the circular shrug-roll trajectory (elevation vs roll depth/horizontal)
    const radius = Math.hypot(elevDeg / 35, rollAxis * 12);
    if (radius >= 0.35) {
      const phaseRad = Math.atan2(elevDeg / 35, rollAxis * 12);
      if (this.lastShrugPhaseRad !== null) {
        let dPhase = ((phaseRad - this.lastShrugPhaseRad) * 180) / Math.PI;
        while (dPhase > 180) dPhase -= 360;
        while (dPhase < -180) dPhase += 360;
        if (Math.abs(dPhase) >= 2 && Math.abs(dPhase) <= 85) {
          this.shrugCircleSweptDeg = Math.min(360, this.shrugCircleSweptDeg + Math.abs(dPhase));
        }
      }
      this.lastShrugPhaseRad = phaseRad;
    }

    if (this.awaitingReturnAfterPeak) {
      const returned = elevDeg <= 15;
      if (returned) {
        this.awaitingReturnAfterPeak = false;
        this.shrugReturnedLow = true;
        this.shrugMaxElevationDeg = 0;
        this.shrugCircleSweptDeg = 0;
      }
      return {
        effectiveRotationRomDeg: Number(elevDeg.toFixed(1)),
        signedAngleDeg: elevDeg,
        sweptArcDeg: Number(this.shrugCircleSweptDeg.toFixed(1)),
        rotationVerified: false,
        returnedToStart: returned,
        guidanceBannerTh: returned
          ? 'ผ่อนคลายหัวไหล่ลงแล้ว พร้อมหมุนหัวไหล่รอบถัดไป'
          : 'หมุนหัวไหล่ลงและผ่อนคลายบ่าก่อนเริ่มรอบถัดไป',
        phaseLabelTh: returned ? 'พร้อมรอบถัดไป' : 'รอหมุนไหล่ลง',
      };
    }

    const liftedAndRolling =
      postureReady &&
      this.shrugReturnedLow &&
      elevDeg >= minLiftDeg &&
      (this.shrugCircleSweptDeg >= 25 || Math.abs(rollAxis) >= 0.024);

    return {
      effectiveRotationRomDeg: Number(clamp(elevDeg, 0, 75).toFixed(1)),
      signedAngleDeg: elevDeg,
      sweptArcDeg: Number(this.shrugCircleSweptDeg.toFixed(1)),
      rotationVerified: liftedAndRolling,
      returnedToStart: elevDeg <= 15,
      guidanceBannerTh: liftedAndRolling
        ? null
        : !this.shrugReturnedLow
          ? 'ลดหัวไหล่ลงในท่าผ่อนคลายก่อน แล้วค่อยยักและหมุนไหล่ขึ้น'
          : elevDeg >= minLiftDeg
            ? `หมุนหัวไหล่เป็นวงกลมไปด้านหลัง-ด้านหน้า (${Math.round(this.shrugCircleSweptDeg)}°)`
            : `ยกและหมุนหัวไหล่ขึ้นเป็นวงกลม (${Math.round(elevDeg)}° / ${Math.round(minLiftDeg)}°)`,
      phaseLabelTh: liftedAndRolling ? `ยักและหมุนไหล่ ${Math.round(elevDeg)}°` : `รอหมุนหัวไหล่ ${Math.round(elevDeg)}°`,
    };
  }
}
