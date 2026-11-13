export type AllowedStarterSide = 'left' | 'right' | 'both';

export const STARTER_HOLD_MS = 600;
export const STARTER_TIMEOUT_MS = 15000;

export interface HandDetectionInput {
  landmarks: readonly { x: number; y: number; z?: number }[];
  handedness: 'Left' | 'Right';
  confidence?: number;
}

export class TouchlessGestureStarter {
  private targetSide: 'left' | 'right' = 'right';
  private startedAtMs: number | null = null;
  private holdStartMs: number | null = null;

  constructor(allowedSide: AllowedStarterSide = 'both') {
    this.rollTarget(allowedSide);
  }

  get requiredSide(): 'left' | 'right' {
    return this.targetSide;
  }

  rollTarget(allowedSide: AllowedStarterSide = 'both'): 'left' | 'right' {
    if (allowedSide === 'left' || allowedSide === 'right') {
      this.targetSide = allowedSide;
    } else if (typeof globalThis.crypto !== 'undefined' && globalThis.crypto.getRandomValues) {
      const buf = new Uint32Array(1);
      globalThis.crypto.getRandomValues(buf);
      this.targetSide = buf[0] % 2 === 0 ? 'left' : 'right';
    } else {
      this.targetSide = Math.random() < 0.5 ? 'left' : 'right';
    }
    this.startedAtMs = null;
    this.holdStartMs = null;
    return this.targetSide;
  }

  update(
    hands: readonly HandDetectionInput[],
    timestampMs: number,
  ): {
    requiredSide: 'left' | 'right';
    confirmed: boolean;
    timedOut: boolean;
    bothHandsWarning: boolean;
    holdProgress: number;
  } {
    if (this.startedAtMs === null) this.startedAtMs = timestampMs;
    const timedOut = timestampMs - this.startedAtMs >= STARTER_TIMEOUT_MS;

    // Reject if both hands raised simultaneously
    if (hands.length >= 2) {
      this.holdStartMs = null;
      return {
        requiredSide: this.targetSide,
        confirmed: false,
        timedOut,
        bothHandsWarning: true,
        holdProgress: 0,
      };
    }

    if (hands.length === 1) {
      const hand = hands[0];
      const normalizedSide = hand.handedness.toLowerCase() as 'left' | 'right';
      const wrist = hand.landmarks[0];
      const middleTip = hand.landmarks[12];
      const isRaised = wrist && middleTip && middleTip.y < wrist.y;

      if (normalizedSide === this.targetSide && isRaised) {
        if (this.holdStartMs === null) this.holdStartMs = timestampMs;
        const elapsed = timestampMs - this.holdStartMs;
        return {
          requiredSide: this.targetSide,
          confirmed: elapsed >= STARTER_HOLD_MS,
          timedOut,
          bothHandsWarning: false,
          holdProgress: Math.min(1, elapsed / STARTER_HOLD_MS),
        };
      }
    }

    this.holdStartMs = null;
    return {
      requiredSide: this.targetSide,
      confirmed: false,
      timedOut,
      bothHandsWarning: false,
      holdProgress: 0,
    };
  }
}

