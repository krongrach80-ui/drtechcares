import { POSE_INDEX, type PoseLandmark } from '@drtechcares/pose-engine';

export const RAISE_HANDS_HOLD_MS = 650;
export const RAISE_HANDS_MIN_RATIO = 0.12;
export const RAISE_HANDS_MIN_VIS = 0.35;
export const RAISE_HANDS_MAX_ANCHOR_DRIFT = 0.22;
export const ACCESSIBILITY_FALLBACK_SINGLE_ARM_MS = 3500;
export const ACCESSIBILITY_FALLBACK_FULL_MS = 7000;

export interface RaiseHandsStatus {
  bothRaised: boolean;
  singleRaised: boolean;
  holdProgress: number;
  confirmed: boolean;
  showAccessibilityFallback: boolean;
  anchorDriftExceeded: boolean;
}

function isPoseArmSideRaised(
  landmarks: readonly PoseLandmark[],
  side: 'left' | 'right',
  fallbackShoulderY: number | null,
): boolean {
  const shoulderIdx = side === 'left' ? POSE_INDEX.LEFT_SHOULDER : POSE_INDEX.RIGHT_SHOULDER;
  const elbowIdx = side === 'left' ? POSE_INDEX.LEFT_ELBOW : POSE_INDEX.RIGHT_ELBOW;
  const handPointIndices = side === 'left' ? [15, 17, 19, 21] : [16, 18, 20, 22];

  const shoulder = landmarks[shoulderIdx];
  const shoulderVisible = Boolean(shoulder && (shoulder.visibility ?? 1) >= 0.2);
  const refShoulderY = shoulderVisible ? shoulder!.y : fallbackShoulderY;

  const elbow = landmarks[elbowIdx];
  const elbowVisible = Boolean(elbow && (elbow.visibility ?? 1) >= 0.2);

  for (const idx of handPointIndices) {
    const pt = landmarks[idx];
    if (!pt || (pt.visibility ?? 1) < 0.15) continue;
    if (refShoulderY !== null) {
      if (pt.y <= Math.min(0.94, refShoulderY + 0.16)) {
        return true;
      }
    } else if (pt.y <= 0.82) {
      return true;
    }
  }

  // Fallback for seated users raising hands high above the top edge of the camera frame
  if (elbowVisible && elbow) {
    const limitY = refShoulderY !== null ? refShoulderY + 0.08 : 0.72;
    if (elbow.y <= limitY) {
      return true;
    }
  }

  return false;
}

export function detectRaiseBothHands(
  landmarks: readonly PoseLandmark[],
  anchorCenterX?: number,
): { bothRaised: boolean; leftRaised: boolean; rightRaised: boolean; driftExceeded: boolean } {
  const lShoulder = landmarks[POSE_INDEX.LEFT_SHOULDER];
  const rShoulder = landmarks[POSE_INDEX.RIGHT_SHOULDER];
  const lWrist = landmarks[POSE_INDEX.LEFT_WRIST];
  const rWrist = landmarks[POSE_INDEX.RIGHT_WRIST];
  const lHip = landmarks[POSE_INDEX.LEFT_HIP];
  const rHip = landmarks[POSE_INDEX.RIGHT_HIP];

  const lShoulderVis = lShoulder && (lShoulder.visibility ?? 1) >= 0.2 ? lShoulder : undefined;
  const rShoulderVis = rShoulder && (rShoulder.visibility ?? 1) >= 0.2 ? rShoulder : undefined;

  if (!lShoulderVis && !rShoulderVis) {
    const leftRaised = isPoseArmSideRaised(landmarks, 'left', null);
    const rightRaised = isPoseArmSideRaised(landmarks, 'right', null);
    return {
      bothRaised: leftRaised && rightRaised,
      leftRaised,
      rightRaised,
      driftExceeded: false,
    };
  }

  const fallbackShoulderY = lShoulderVis?.y ?? rShoulderVis?.y ?? null;
  const midShoulderX =
    lShoulderVis && rShoulderVis
      ? (lShoulderVis.x + rShoulderVis.x) / 2
      : (lShoulderVis?.x ?? rShoulderVis?.x ?? 0.5);
  const driftExceeded =
    anchorCenterX !== undefined && Math.abs(midShoulderX - anchorCenterX) > RAISE_HANDS_MAX_ANCHOR_DRIFT;

  const hasValidHips = Boolean(
    lHip && rHip && (lHip.visibility ?? 1) >= 0.2 && (rHip.visibility ?? 1) >= 0.2,
  );
  const torsoHeight = hasValidHips && lShoulderVis && rShoulderVis
    ? Math.max(0.1, (lHip!.y + rHip!.y) / 2 - (lShoulderVis.y + rShoulderVis.y) / 2)
    : Math.max(
        0.18,
        lShoulderVis && rShoulderVis ? Math.abs(rShoulderVis.x - lShoulderVis.x) * 1.2 : 0.25,
      );
  const chestMargin = Math.max(0.08, torsoHeight * RAISE_HANDS_MIN_RATIO);

  const lVisOk = Boolean(lWrist && (lWrist.visibility ?? 1) >= RAISE_HANDS_MIN_VIS);
  const rVisOk = Boolean(rWrist && (rWrist.visibility ?? 1) >= RAISE_HANDS_MIN_VIS);

  const leftRaised =
    (lVisOk && lShoulderVis !== undefined && lWrist!.y <= lShoulderVis.y + Math.min(0.08, chestMargin)) ||
    isPoseArmSideRaised(landmarks, 'left', fallbackShoulderY);
  const rightRaised =
    (rVisOk && rShoulderVis !== undefined && rWrist!.y <= rShoulderVis.y + Math.min(0.08, chestMargin)) ||
    isPoseArmSideRaised(landmarks, 'right', fallbackShoulderY);

  return {
    bothRaised: leftRaised && rightRaised && !driftExceeded,
    leftRaised,
    rightRaised,
    driftExceeded,
  };
}

function filterValidHandLandmarks<T extends ReadonlyArray<{ x: number; y: number }>>(hands: ReadonlyArray<T>): T[] {
  return hands.filter((hand) => {
    if (hand.length < 21) return false;
    const wrist = hand[0];
    return Boolean(wrist && wrist.y <= 0.99);
  });
}

function handsSeparated(a: ReadonlyArray<{ x: number; y: number }>, b: ReadonlyArray<{ x: number; y: number }>): boolean {
  const aRef = a[9] ?? a[0];
  const bRef = b[9] ?? b[0];
  const wristSep = Math.hypot(a[0].x - b[0].x, a[0].y - b[0].y);
  const mcpSep = aRef && bRef ? Math.hypot(aRef.x - bRef.x, aRef.y - bRef.y) : wristSep;
  return Math.max(wristSep, mcpSep) >= 0.04;
}

/**
 * Checks whether MediaPipe HandLandmarker detected two separate raised hands in frame.
 */
export function detectTwoRaisedHandLandmarks(
  hands: ReadonlyArray<ReadonlyArray<{ x: number; y: number }>>,
): boolean {
  const validHands = filterValidHandLandmarks(hands);
  if (validHands.length < 2) return false;
  for (let i = 0; i < validHands.length; i += 1) {
    for (let j = i + 1; j < validHands.length; j += 1) {
      if (handsSeparated(validHands[i], validHands[j])) {
        return true;
      }
    }
  }
  return false;
}

/**
 * Combines MediaPipe HandLandmarker (hands) and PoseLandmarker (wrists/hand-tips/elbows)
 * to robustly detect whether left, right, or both arms/hands are raised in seated webcam views.
 */
export function extractRaisedHandSides(
  hands: ReadonlyArray<ReadonlyArray<{ x: number; y: number }>>,
  poseLandmarks?: readonly PoseLandmark[] | null,
): { leftRaised: boolean; rightRaised: boolean; bothRaised: boolean } {
  if (detectTwoRaisedHandLandmarks(hands)) {
    return { leftRaised: true, rightRaised: true, bothRaised: true };
  }

  let leftRaised = false;
  let rightRaised = false;

  if (poseLandmarks && poseLandmarks.length > 0) {
    const poseDet = detectRaiseBothHands(poseLandmarks);
    leftRaised = poseDet.leftRaised;
    rightRaised = poseDet.rightRaised;
  }

  const validHands = filterValidHandLandmarks(hands);
  if (validHands.length === 1) {
    const singleHand = validHands[0];
    const handAnchorX = (singleHand[9] ?? singleHand[0]).x;
    const lShoulder = poseLandmarks?.[POSE_INDEX.LEFT_SHOULDER];
    const rShoulder = poseLandmarks?.[POSE_INDEX.RIGHT_SHOULDER];
    if (lShoulder && rShoulder && (lShoulder.visibility ?? 1) >= 0.2 && (rShoulder.visibility ?? 1) >= 0.2) {
      if (Math.abs(handAnchorX - lShoulder.x) <= Math.abs(handAnchorX - rShoulder.x)) {
        leftRaised = true;
      } else {
        rightRaised = true;
      }
    } else if (handAnchorX >= 0.5) {
      leftRaised = true;
    } else {
      rightRaised = true;
    }
  }

  return {
    leftRaised,
    rightRaised,
    bothRaised: leftRaised && rightRaised,
  };
}

export class RaiseBothHandsTracker {
  private raisedSinceMs: number | null = null;
  private singleRaisedSinceMs: number | null = null;
  private startedAtMs: number | null = null;
  private emaProgress = 0;

  reset(): void {
    this.raisedSinceMs = null;
    this.singleRaisedSinceMs = null;
    this.startedAtMs = null;
    this.emaProgress = 0;
  }

  update(landmarks: readonly PoseLandmark[], timestampMs: number, anchorCenterX?: number): RaiseHandsStatus {
    if (this.startedAtMs === null) this.startedAtMs = timestampMs;

    const det = detectRaiseBothHands(landmarks, anchorCenterX);
    const singleRaised = (det.leftRaised || det.rightRaised) && !det.bothRaised;

    if (singleRaised) {
      if (this.singleRaisedSinceMs === null) this.singleRaisedSinceMs = timestampMs;
    } else {
      this.singleRaisedSinceMs = null;
    }

    if (det.bothRaised) {
      if (this.raisedSinceMs === null) this.raisedSinceMs = timestampMs;
      const elapsed = timestampMs - this.raisedSinceMs;
      const rawProgress = Math.min(1, elapsed / RAISE_HANDS_HOLD_MS);
      this.emaProgress = 0.35 * rawProgress + 0.65 * this.emaProgress;
      return {
        bothRaised: true,
        singleRaised: false,
        holdProgress: Math.max(rawProgress, this.emaProgress),
        confirmed: elapsed >= RAISE_HANDS_HOLD_MS,
        showAccessibilityFallback: false,
        anchorDriftExceeded: det.driftExceeded,
      };
    }

    this.raisedSinceMs = null;
    this.emaProgress = 0.65 * this.emaProgress;

    const totalWait = timestampMs - this.startedAtMs;
    const singleWait = this.singleRaisedSinceMs ? timestampMs - this.singleRaisedSinceMs : 0;
    const showAccessibilityFallback =
      totalWait >= ACCESSIBILITY_FALLBACK_FULL_MS ||
      singleWait >= ACCESSIBILITY_FALLBACK_SINGLE_ARM_MS;

    return {
      bothRaised: false,
      singleRaised,
      holdProgress: Number(this.emaProgress.toFixed(2)),
      confirmed: false,
      showAccessibilityFallback,
      anchorDriftExceeded: det.driftExceeded,
    };
  }
}

