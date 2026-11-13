import {
  BURST_FRAME_COUNT,
  NOT_FOUND_CONFIRM_ROUNDS,
  computeOutlierRejectedCentroid,
  l2Normalize,
  type BiometricBurstFrame,
  type BiometricPose,
} from '@drtechcares/smart-biometric-core';

import type { Detected3DPose, FaceIdScanStage } from './biometricAssistant';

export interface EnrollmentShot {
  pose: 'front' | 'left' | 'right' | 'nod';
  vector: number[];
  qualityScore: number;
  capturedAt: string;
}

export interface EnrollmentPayload10Shot {
  frontal_centroid: number[];
  overall_centroid: number[];
  raw_frames: EnrollmentShot[];
}

export const TEN_SHOT_SEQUENCE: readonly ('front' | 'left' | 'right' | 'nod')[] = [
  'front',
  'front',
  'front',
  'front',
  'left',
  'left',
  'right',
  'right',
  'nod',
  'nod',
];

export const CIRCULAR_3D_POSE_SEQUENCE: readonly Detected3DPose[] = [
  'CENTER',
  'CENTER',
  'TURN_LEFT',
  'TILT_UP',
  'TURN_RIGHT',
  'CENTER',
  'CENTER',
  'TURN_LEFT',
  'TURN_RIGHT',
  'TILT_DOWN',
];

export function mapBackendPoseTo3DPose(
  pose: 'front' | 'left' | 'right' | 'nod' | null,
  stepIndex: number,
): Detected3DPose | undefined {
  if (!pose) return undefined;
  return CIRCULAR_3D_POSE_SEQUENCE[stepIndex] ?? (pose === 'front' ? 'CENTER' : pose === 'left' ? 'TURN_LEFT' : pose === 'right' ? 'TURN_RIGHT' : 'TILT_DOWN');
}

/**
 * Controls the 10-shot multi-angle enrollment sequence:
 * front(4) -> left(2) -> right(2) -> nod(2)
 */
export class TenShotEnrollmentController {
  private shots: EnrollmentShot[] = [];

  get currentStepIndex(): number {
    return this.shots.length;
  }

  get isComplete(): boolean {
    return this.shots.length >= TEN_SHOT_SEQUENCE.length;
  }

  get expectedPose(): 'front' | 'left' | 'right' | 'nod' | null {
    return TEN_SHOT_SEQUENCE[this.shots.length] ?? null;
  }

  get capturedShots(): readonly EnrollmentShot[] {
    return this.shots;
  }

  get scanStage(): FaceIdScanStage {
    if (this.isComplete) return 'completed';
    if (this.shots.length === 0) return 'initial';
    if (this.shots.length < 5) return 'round1';
    return 'round2';
  }

  get expected3DPose(): Detected3DPose | undefined {
    if (this.isComplete) return undefined;
    return mapBackendPoseTo3DPose(this.expectedPose, this.shots.length);
  }

  reset(): void {
    this.shots = [];
  }

  acceptShot(pose: 'front' | 'left' | 'right' | 'nod', vector: readonly number[], qualityScore: number): boolean {
    if (this.isComplete) return false;
    if (pose !== this.expectedPose) return false;
    this.shots.push({
      pose,
      vector: l2Normalize(vector),
      qualityScore,
      capturedAt: new Date().toISOString(),
    });
    return true;
  }

  buildPayload(): EnrollmentPayload10Shot {
    if (!this.isComplete) {
      throw new Error(`10-shot enrollment incomplete (${this.shots.length}/10)`);
    }
    const frontalShots = this.shots.filter((s) => s.pose === 'front');
    const frontalCentroid = computeOutlierRejectedCentroid(
      frontalShots.map((s) => ({
        vector: s.vector,
        qualityScore: s.qualityScore,
        capturedAfterCountdown: true,
        pose: 'front' as BiometricPose,
      })),
      frontalShots.length,
    ).centroid;

    const overallCentroid = computeOutlierRejectedCentroid(
      this.shots.map((s) => ({
        vector: s.vector,
        qualityScore: s.qualityScore,
        capturedAfterCountdown: true,
        pose: s.pose as BiometricPose,
      })),
      this.shots.length,
    ).centroid;

    return {
      frontal_centroid: frontalCentroid,
      overall_centroid: overallCentroid,
      raw_frames: [...this.shots],
    };
  }
}

/**
 * Controls Single-Scan Decisive Kiosk Login:
 * - Waits for countdown completion
 * - Collects ONLY fresh post-countdown burst frames (6 frames)
 * - Uses NOT_FOUND_CONFIRM_ROUNDS = 1 so unknown faces immediately trigger the
 *   "Account Not Found" modal without endless looping.
 */
export class SingleScanLoginController {
  private countdownComplete = false;
  private burstFrames: BiometricBurstFrame[] = [];
  private failedRounds = 0;

  startCountdownComplete(): void {
    this.countdownComplete = true;
    this.burstFrames = [];
  }

  reset(): void {
    this.countdownComplete = false;
    this.burstFrames = [];
    this.failedRounds = 0;
  }

  pushFrame(frame: BiometricBurstFrame): { readyToVerify: boolean; collected: number } {
    if (!this.countdownComplete || !frame.capturedAfterCountdown) {
      // Strictly discard stale pre-countdown frames
      return { readyToVerify: false, collected: 0 };
    }
    this.burstFrames.push(frame);
    return {
      readyToVerify: this.burstFrames.length >= BURST_FRAME_COUNT,
      collected: this.burstFrames.length,
    };
  }

  buildProbeCentroid(): number[] {
    const { centroid } = computeOutlierRejectedCentroid(this.burstFrames, BURST_FRAME_COUNT);
    return centroid;
  }

  recordUnmatchedRound(): { shouldShowNotFoundModal: boolean; rounds: number } {
    this.failedRounds += 1;
    return {
      shouldShowNotFoundModal: this.failedRounds >= NOT_FOUND_CONFIRM_ROUNDS,
      rounds: this.failedRounds,
    };
  }
}

