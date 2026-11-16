export const PRESENCE_WAKE_MS = 1000;
export const PRESENCE_SLEEP_HYSTERESIS_MS = 2000;
// Supports long-range clinic presence up to ~4.5m (~28px+ face width on 1080p)
export const MIN_FACE_SIZE_RATIO = 0.025;

export interface PresenceObservation {
  facePresent: boolean;
  faceSizeRatio: number;
  timestampMs: number;
}

export class FacePresenceTracker {
  private awake = false;
  private presentSinceMs: number | null = null;
  private absentSinceMs: number | null = null;

  get isAwake(): boolean {
    return this.awake;
  }

  reset(): void {
    this.awake = false;
    this.presentSinceMs = null;
    this.absentSinceMs = null;
  }

  update(obs: PresenceObservation): { awake: boolean; progress: number; justWokeUp: boolean; justSlept: boolean } {
    const validFace = obs.facePresent && obs.faceSizeRatio >= MIN_FACE_SIZE_RATIO;
    let justWokeUp = false;
    let justSlept = false;

    if (validFace) {
      this.absentSinceMs = null;
      if (this.presentSinceMs === null) {
        this.presentSinceMs = obs.timestampMs;
      }
      const elapsed = obs.timestampMs - this.presentSinceMs;
      if (!this.awake && elapsed >= PRESENCE_WAKE_MS) {
        this.awake = true;
        justWokeUp = true;
      }
      return {
        awake: this.awake,
        progress: Math.min(1, elapsed / PRESENCE_WAKE_MS),
        justWokeUp,
        justSlept,
      };
    }

    this.presentSinceMs = null;
    if (this.awake) {
      if (this.absentSinceMs === null) {
        this.absentSinceMs = obs.timestampMs;
      }
      if (obs.timestampMs - this.absentSinceMs >= PRESENCE_SLEEP_HYSTERESIS_MS) {
        this.awake = false;
        justSlept = true;
      }
    }

    return {
      awake: this.awake,
      progress: 0,
      justWokeUp,
      justSlept,
    };
  }
}

