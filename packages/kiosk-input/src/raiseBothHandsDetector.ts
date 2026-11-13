import { POSE_INDEX, type PoseLandmark } from '@drtechcares/pose-engine';

export const RAISE_HANDS_HOLD_MS = 1200;
export const RAISE_HANDS_MIN_RATIO = 0.12;
export const RAISE_HANDS_MIN_VIS = 0.6;
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

  if (!lShoulder || !rShoulder || !lWrist || !rWrist || !lHip || !rHip) {
    return { bothRaised: false, leftRaised: false, rightRaised: false, driftExceeded: false };
  }

  const midShoulderX = (lShoulder.x + rShoulder.x) / 2;
  const driftExceeded =
    anchorCenterX !== undefined && Math.abs(midShoulderX - anchorCenterX) > RAISE_HANDS_MAX_ANCHOR_DRIFT;

  const torsoHeight = Math.max(
    0.1,
    ((lHip.y + rHip.y) / 2) - ((lShoulder.y + rShoulder.y) / 2),
  );
  const thresholdOffset = RAISE_HANDS_MIN_RATIO * torsoHeight;

  const lVisOk = (lWrist.visibility ?? 1) >= RAISE_HANDS_MIN_VIS;
  const rVisOk = (rWrist.visibility ?? 1) >= RAISE_HANDS_MIN_VIS;

  const leftRaised = lVisOk && lWrist.y < lShoulder.y - thresholdOffset;
  const rightRaised = rVisOk && rWrist.y < rShoulder.y - thresholdOffset;

  return {
    bothRaised: leftRaised && rightRaised && !driftExceeded,
    leftRaised,
    rightRaised,
    driftExceeded,
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

