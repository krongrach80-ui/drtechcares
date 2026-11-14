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

  if (!lShoulder || !rShoulder || !lWrist || !rWrist) {
    return { bothRaised: false, leftRaised: false, rightRaised: false, driftExceeded: false };
  }

  const midShoulderX = (lShoulder.x + rShoulder.x) / 2;
  const driftExceeded =
    anchorCenterX !== undefined && Math.abs(midShoulderX - anchorCenterX) > RAISE_HANDS_MAX_ANCHOR_DRIFT;

  const hasValidHips = Boolean(
    lHip && rHip && (lHip.visibility ?? 1) >= 0.2 && (rHip.visibility ?? 1) >= 0.2,
  );
  const torsoHeight = hasValidHips
    ? Math.max(0.1, (lHip!.y + rHip!.y) / 2 - (lShoulder.y + rShoulder.y) / 2)
    : Math.max(0.18, Math.abs(rShoulder.x - lShoulder.x) * 1.2);
  const chestMargin = Math.max(0.08, torsoHeight * RAISE_HANDS_MIN_RATIO);

  const lVisOk = (lWrist.visibility ?? 1) >= RAISE_HANDS_MIN_VIS;
  const rVisOk = (rWrist.visibility ?? 1) >= RAISE_HANDS_MIN_VIS;

  const leftRaised = lVisOk && lWrist.y <= lShoulder.y + Math.min(0.08, chestMargin);
  const rightRaised = rVisOk && rWrist.y <= rShoulder.y + Math.min(0.08, chestMargin);

  return {
    bothRaised: leftRaised && rightRaised && !driftExceeded,
    leftRaised,
    rightRaised,
    driftExceeded,
  };
}

/**
 * Checks whether MediaPipe HandLandmarker detected two separate raised hands in frame.
 */
export function detectTwoRaisedHandLandmarks(
  hands: ReadonlyArray<ReadonlyArray<{ x: number; y: number }>>,
): boolean {
  const raisedHands = hands.filter((hand) => {
    if (hand.length < 21) return false;
    const wrist = hand[0];
    if (!wrist || wrist.y > 0.98) return false;
    const tips = [hand[4], hand[8], hand[12], hand[16], hand[20]];
    return tips.some((tip) => tip && tip.y <= wrist.y + 0.12);
  });
  return (
    raisedHands.length >= 2 &&
    Math.hypot(
      raisedHands[0][0].x - raisedHands[1][0].x,
      raisedHands[0][0].y - raisedHands[1][0].y,
    ) >= 0.06
  );
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

