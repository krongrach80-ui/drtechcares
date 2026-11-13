export const HOLD_DWELL_MS = 600;
export const MAX_CROP_RATIO_PORTRAIT = 0.4;
export const MIN_EXTENDED_FINGERS = 3;
export const MAX_HAND_CHEST_Y = 0.88;

export interface HandLandmarkPoint {
  x: number;
  y: number;
  z?: number;
}

export interface DetectedHandFrame {
  landmarks: readonly HandLandmarkPoint[];
  /** Anatomical handedness from MediaPipe ('Left' | 'Right') */
  handedness: 'Left' | 'Right';
  score?: number;
}

export type CognitiveAnswerChoice = 'yes' | 'no' | null;

export interface HandVerificationOutput {
  choice: CognitiveAnswerChoice;
  anatomicalSide: 'left' | 'right' | null;
  holdProgress: number;
  confirmedAnswer: CognitiveAnswerChoice;
  bothHandsRaisedWarning: boolean;
  outsidePortraitCropWarning: boolean;
  gateReason: string;
}

function dist2(a: HandLandmarkPoint, b: HandLandmarkPoint): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

/**
 * Evaluates 4-Layer Hand Gate on a single 21-point hand:
 * Layer 1: 21 landmarks present & score >= 0.55
 * Layer 2: Open palm geometry (>= 3 extended fingers where tipToWrist > 1.25 * mcpToWrist)
 * Layer 3: Fingertips elevated above wrist & hand at chest level or higher (wrist.y <= 0.88)
 * Layer 4: Anatomical handedness mapping -> Right = 'yes', Left = 'no'
 */
export function evaluateAnatomicalHand(hand: DetectedHandFrame): {
  valid: boolean;
  anatomicalSide: 'left' | 'right';
  choice: 'yes' | 'no';
  palmSize: number;
  centerX: number;
  reason: string;
} {
  const anatomicalSide = hand.handedness === 'Right' ? 'right' : 'left';
  const choice: 'yes' | 'no' = anatomicalSide === 'right' ? 'yes' : 'no';

  if (hand.landmarks.length < 21 || (hand.score !== undefined && hand.score < 0.55)) {
    return { valid: false, anatomicalSide, choice, palmSize: 0, centerX: 0.5, reason: 'LOW_CONFIDENCE_OR_INCOMPLETE' };
  }

  const wrist = hand.landmarks[0];
  const middleMcp = hand.landmarks[9];
  const palmSize = dist2(wrist, middleMcp);

  if (wrist.y > MAX_HAND_CHEST_Y) {
    return { valid: false, anatomicalSide, choice, palmSize, centerX: wrist.x, reason: 'BELOW_CHEST_LEVEL' };
  }

  // Check 4 finger pairs: [MCP, TIP] -> Index(5,8), Middle(9,12), Ring(13,16), Pinky(17,20)
  const fingerPairs: readonly [number, number][] = [
    [5, 8],
    [9, 12],
    [13, 16],
    [17, 20],
  ];

  let extendedCount = 0;
  let tipsAboveWrist = 0;
  for (const [mcpIdx, tipIdx] of fingerPairs) {
    const mcp = hand.landmarks[mcpIdx];
    const tip = hand.landmarks[tipIdx];
    const tipToWrist = dist2(tip, wrist);
    const mcpToWrist = Math.max(1e-4, dist2(mcp, wrist));
    if (tipToWrist > 1.25 * mcpToWrist) {
      extendedCount += 1;
    }
    if (tip.y < wrist.y) {
      tipsAboveWrist += 1;
    }
  }

  if (extendedCount < MIN_EXTENDED_FINGERS || tipsAboveWrist < 2) {
    return { valid: false, anatomicalSide, choice, palmSize, centerX: wrist.x, reason: 'NOT_OPEN_RAISED_PALM' };
  }

  return { valid: true, anatomicalSide, choice, palmSize, centerX: wrist.x, reason: 'OK' };
}

/**
 * Verifies Cognitive Minigame hand gestures with:
 * - Clean camera view (no skeleton overlays)
 * - Dual-hand simultaneous raise warning (blocks answer)
 * - Largest-palm primary player filter
 * - 600ms dwell hold progress
 * - Portrait crop margin guard (MAX_CROP_RATIO_PORTRAIT = 0.4)
 */
export class HandGestureVerifier {
  private activeChoice: CognitiveAnswerChoice = null;
  private holdStartMs: number | null = null;
  private lockedAfterConfirm = false;

  reset(): void {
    this.activeChoice = null;
    this.holdStartMs = null;
    this.lockedAfterConfirm = false;
  }

  update(
    hands: readonly DetectedHandFrame[],
    timestampMs: number,
    options: { isPortrait?: boolean; cropRatio?: number } = {},
  ): HandVerificationOutput {
    const evaluated = hands.map((h) => evaluateAnatomicalHand(h));
    const validHands = evaluated.filter((e) => e.valid);

    // Check portrait crop margin
    const cropRatio = Math.min(MAX_CROP_RATIO_PORTRAIT, options.cropRatio ?? 0);
    const leftBound = options.isPortrait ? cropRatio / 2 : 0.02;
    const rightBound = options.isPortrait ? 1 - cropRatio / 2 : 0.98;
    const outsideCrop = evaluated.some(
      (e) => e.centerX < leftBound || e.centerX > rightBound,
    );

    const hasValidLeft = validHands.some((e) => e.anatomicalSide === 'left');
    const hasValidRight = validHands.some((e) => e.anatomicalSide === 'right');

    // Reject if both hands are raised simultaneously
    if (hasValidLeft && hasValidRight) {
      this.activeChoice = null;
      this.holdStartMs = null;
      return {
        choice: null,
        anatomicalSide: null,
        holdProgress: 0,
        confirmedAnswer: null,
        bothHandsRaisedWarning: true,
        outsidePortraitCropWarning: outsideCrop,
        gateReason: 'BOTH_HANDS_RAISED',
      };
    }

    if (validHands.length === 0) {
      this.activeChoice = null;
      this.holdStartMs = null;
      this.lockedAfterConfirm = false;
      return {
        choice: null,
        anatomicalSide: null,
        holdProgress: 0,
        confirmedAnswer: null,
        bothHandsRaisedWarning: false,
        outsidePortraitCropWarning: outsideCrop,
        gateReason: evaluated[0]?.reason ?? 'NO_VALID_PALM',
      };
    }

    // Select largest palm (primary user in front of kiosk)
    const primary = validHands.reduce((a, b) => (b.palmSize > a.palmSize ? b : a));

    if (this.activeChoice !== primary.choice || this.holdStartMs === null) {
      this.activeChoice = primary.choice;
      this.holdStartMs = timestampMs;
      this.lockedAfterConfirm = false;
    }

    const elapsed = timestampMs - this.holdStartMs;
    const holdProgress = Math.min(1, elapsed / HOLD_DWELL_MS);
    let confirmedAnswer: CognitiveAnswerChoice = null;

    if (elapsed >= HOLD_DWELL_MS && !this.lockedAfterConfirm) {
      confirmedAnswer = primary.choice;
      this.lockedAfterConfirm = true;
    }

    return {
      choice: primary.choice,
      anatomicalSide: primary.anatomicalSide,
      holdProgress,
      confirmedAnswer,
      bothHandsRaisedWarning: false,
      outsidePortraitCropWarning: outsideCrop,
      gateReason: 'OK',
    };
  }
}

