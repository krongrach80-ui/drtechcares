import type { BodySide, PoseLandmark } from './types';

export const SMOOTHING_ALPHA = 0.35;
export const MAX_DELTA_DEG_PER_FRAME = 28;
export const SPIKE_WINDOW_MS = 150;
export const MISSING_RESET_MS = 500;

class LowPassFilter {
  private initialized = false;
  private yPrev = 0;

  filter(value: number, alpha: number): number {
    if (!this.initialized) {
      this.initialized = true;
      this.yPrev = value;
      return value;
    }
    const y = alpha * value + (1 - alpha) * this.yPrev;
    this.yPrev = y;
    return y;
  }

  reset(): void {
    this.initialized = false;
    this.yPrev = 0;
  }
}

class OneEuroScalar {
  private readonly xFilter = new LowPassFilter();
  private readonly dxFilter = new LowPassFilter();
  private lastTimestampMs: number | null = null;
  private prevRaw = 0;

  constructor(
    private readonly minCutoff = 1.2,
    private readonly beta = 0.015,
    private readonly dCutoff = 1.0,
  ) {}

  private alpha(cutoff: number, dtSec: number): number {
    const tau = 1.0 / (2 * Math.PI * cutoff);
    return 1.0 / (1.0 + tau / Math.max(dtSec, 1e-4));
  }

  filter(value: number, timestampMs: number): number {
    if (this.lastTimestampMs === null) {
      this.lastTimestampMs = timestampMs;
      this.prevRaw = value;
      return this.xFilter.filter(value, 1);
    }

    const dtSec = Math.max(0.001, (timestampMs - this.lastTimestampMs) / 1000);
    this.lastTimestampMs = timestampMs;

    const dx = (value - this.prevRaw) / dtSec;
    this.prevRaw = value;

    const edx = this.dxFilter.filter(dx, this.alpha(this.dCutoff, dtSec));
    const cutoff = this.minCutoff + this.beta * Math.abs(edx);
    return this.xFilter.filter(value, this.alpha(cutoff, dtSec));
  }

  reset(): void {
    this.xFilter.reset();
    this.dxFilter.reset();
    this.lastTimestampMs = null;
    this.prevRaw = 0;
  }
}

/**
 * 3D/2D Clinical One-Euro Landmark Filter across 33 MediaPipe Pose landmarks.
 * Resets automatically if landmarks disappear for > 500ms.
 */
export class ClinicalOneEuroLandmarkFilter {
  private filters: { x: OneEuroScalar; y: OneEuroScalar; z: OneEuroScalar }[] = [];
  private lastSeenMs: number | null = null;

  constructor(
    private readonly minCutoff = 1.2,
    private readonly beta = 0.015,
    private readonly dCutoff = 1.0,
  ) {}

  reset(): void {
    for (const f of this.filters) {
      f.x.reset();
      f.y.reset();
      f.z.reset();
    }
    this.lastSeenMs = null;
  }

  filterLandmarks(landmarks: readonly PoseLandmark[], timestampMs: number): PoseLandmark[] {
    if (this.lastSeenMs !== null && timestampMs - this.lastSeenMs > MISSING_RESET_MS) {
      this.reset();
    }
    this.lastSeenMs = timestampMs;

    while (this.filters.length < landmarks.length) {
      this.filters.push({
        x: new OneEuroScalar(this.minCutoff, this.beta, this.dCutoff),
        y: new OneEuroScalar(this.minCutoff, this.beta, this.dCutoff),
        z: new OneEuroScalar(this.minCutoff, this.beta, this.dCutoff),
      });
    }

    return landmarks.map((lm, i) => ({
      x: this.filters[i].x.filter(lm.x, timestampMs),
      y: this.filters[i].y.filter(lm.y, timestampMs),
      z: lm.z !== undefined ? this.filters[i].z.filter(lm.z, timestampMs) : undefined,
      visibility: lm.visibility,
    }));
  }
}

interface SideAngleState {
  smoothedAngle: number | null;
  lastUpdateMs: number | null;
  missingSinceMs: number | null;
}

/**
 * Per-side Joint Angle Smoother with EMA (alpha = 0.35) and Spike Clamp
 * (MAX_DELTA_DEG_PER_FRAME = 28° when dt < 150ms).
 * Maintains separate left/right state and clears automatically on exercise switch
 * or landmark absence > 500ms.
 */
export class JointAngleSmoother {
  private exerciseKey = '';
  private readonly sideState: Record<BodySide, SideAngleState> = {
    left: { smoothedAngle: null, lastUpdateMs: null, missingSinceMs: null },
    right: { smoothedAngle: null, lastUpdateMs: null, missingSinceMs: null },
  };

  reset(side?: BodySide): void {
    const sides: BodySide[] = side ? [side] : ['left', 'right'];
    for (const s of sides) {
      this.sideState[s] = { smoothedAngle: null, lastUpdateMs: null, missingSinceMs: null };
    }
  }

  setExercise(exerciseId: string): void {
    if (this.exerciseKey !== exerciseId) {
      this.exerciseKey = exerciseId;
      this.reset();
    }
  }

  /**
   * Updates smoothed angle for a side. If rawAngle is null (occluded/out-of-bounds),
   * holds the last reliable angle (never jumps to 0 or 180) and resets if missing > 500ms.
   */
  update(side: BodySide, rawAngle: number | null, timestampMs: number): {
    smoothed: number | null;
    lastReliable: number | null;
    clampedSpike: boolean;
  } {
    const state = this.sideState[side];

    if (rawAngle === null || !Number.isFinite(rawAngle)) {
      if (state.missingSinceMs === null) {
        state.missingSinceMs = timestampMs;
      } else if (timestampMs - state.missingSinceMs > MISSING_RESET_MS) {
        this.reset(side);
        return { smoothed: null, lastReliable: null, clampedSpike: false };
      }
      return {
        smoothed: null,
        lastReliable: state.smoothedAngle,
        clampedSpike: false,
      };
    }

    state.missingSinceMs = null;

    if (state.smoothedAngle === null || state.lastUpdateMs === null) {
      state.smoothedAngle = rawAngle;
      state.lastUpdateMs = timestampMs;
      return { smoothed: rawAngle, lastReliable: rawAngle, clampedSpike: false };
    }

    const dtMs = Math.max(0, timestampMs - state.lastUpdateMs);
    if (dtMs > MISSING_RESET_MS) {
      state.smoothedAngle = rawAngle;
      state.lastUpdateMs = timestampMs;
      return { smoothed: rawAngle, lastReliable: rawAngle, clampedSpike: false };
    }

    let candidateAngle = rawAngle;
    let clampedSpike = false;

    if (dtMs < SPIKE_WINDOW_MS) {
      const delta = rawAngle - state.smoothedAngle;
      if (Math.abs(delta) > MAX_DELTA_DEG_PER_FRAME) {
        candidateAngle = state.smoothedAngle + Math.sign(delta) * MAX_DELTA_DEG_PER_FRAME;
        clampedSpike = true;
      }
    }

    const nextSmoothed = SMOOTHING_ALPHA * candidateAngle + (1 - SMOOTHING_ALPHA) * state.smoothedAngle;
    state.smoothedAngle = nextSmoothed;
    state.lastUpdateMs = timestampMs;

    return {
      smoothed: Number(nextSmoothed.toFixed(2)),
      lastReliable: Number(nextSmoothed.toFixed(2)),
      clampedSpike,
    };
  }

  getLastReliable(side: BodySide): number | null {
    return this.sideState[side].smoothedAngle;
  }
}

