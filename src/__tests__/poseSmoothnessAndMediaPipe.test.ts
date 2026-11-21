import fs from 'node:fs';
import path from 'node:path';
import type { NormalizedLandmark } from '@mediapipe/tasks-vision';
import { describe, expect, it } from 'vitest';
import {
  CLINICAL_HAND_ONE_EURO_PRESET,
  CLINICAL_POSE_ONE_EURO_PRESET,
  ClinicalHandOneEuroFilter,
  ClinicalOneEuroLandmarkFilter,
  computeDtNormalizedLerpAlpha,
  stabilizeArmIdentity,
} from '@drtechcares/pose-engine';
import {
  getNextMonotonicHandDetectTs,
  getNextMonotonicPoseDetectTs,
  getNextMonotonicRehabFaceDetectTs,
  getNextMonotonicVideoDetectTs,
  resetBiometricVisionCacheForTests,
} from '../lib/biometricVisionCache';
import { drawLivePoseOverlayOnCanvas } from '../components/SmartRehabKioskUI';

function makeStandingPose(
  overrides: Partial<Record<number, Partial<NormalizedLandmark>>> = {},
): NormalizedLandmark[] {
  const base: NormalizedLandmark[] = Array.from({ length: 33 }, () => ({
    x: 0.5,
    y: 0.5,
    z: 0,
    visibility: 0.96,
  }));

  // Symmetric anatomical baseline: left x ~ 0.62, right x ~ 0.38
  base[0] = { x: 0.5, y: 0.18, z: -0.04, visibility: 0.99 };
  base[7] = { x: 0.54, y: 0.18, z: 0.0, visibility: 0.95 };
  base[8] = { x: 0.46, y: 0.18, z: 0.0, visibility: 0.95 };
  base[11] = { x: 0.6, y: 0.34, z: 0.0, visibility: 0.99 };
  base[12] = { x: 0.4, y: 0.34, z: 0.0, visibility: 0.99 };
  base[13] = { x: 0.64, y: 0.49, z: -0.02, visibility: 0.98 };
  base[14] = { x: 0.36, y: 0.49, z: -0.02, visibility: 0.98 };
  base[15] = { x: 0.66, y: 0.62, z: -0.05, visibility: 0.97 };
  base[16] = { x: 0.34, y: 0.62, z: -0.05, visibility: 0.97 };
  // Left palm extremities (17, 19, 21) & Right palm extremities (18, 20, 22)
  base[17] = { x: 0.68, y: 0.64, z: -0.05, visibility: 0.92 };
  base[19] = { x: 0.67, y: 0.65, z: -0.05, visibility: 0.92 };
  base[21] = { x: 0.65, y: 0.63, z: -0.05, visibility: 0.92 };
  base[18] = { x: 0.32, y: 0.64, z: -0.05, visibility: 0.92 };
  base[20] = { x: 0.33, y: 0.65, z: -0.05, visibility: 0.92 };
  base[22] = { x: 0.35, y: 0.63, z: -0.05, visibility: 0.92 };
  base[23] = { x: 0.56, y: 0.64, z: 0.0, visibility: 0.98 };
  base[24] = { x: 0.44, y: 0.64, z: 0.0, visibility: 0.98 };
  base[25] = { x: 0.56, y: 0.81, z: 0.0, visibility: 0.95 };
  base[26] = { x: 0.44, y: 0.81, z: 0.0, visibility: 0.95 };
  base[27] = { x: 0.56, y: 0.95, z: 0.0, visibility: 0.93 };
  base[28] = { x: 0.44, y: 0.95, z: 0.0, visibility: 0.93 };

  for (const [idxStr, patch] of Object.entries(overrides)) {
    const idx = Number(idxStr);
    if (patch && base[idx]) {
      base[idx] = { ...base[idx], ...patch };
    }
  }
  return base;
}

describe('One-Euro preset responsiveness, dt-normalized lerp & L/R arm lock', () => {
  it('CLINICAL_POSE_ONE_EURO_PRESET eliminates rubber-band lag vs legacy (1.15, 0.06) while keeping stationary jitter damped', () => {
    expect(CLINICAL_POSE_ONE_EURO_PRESET).toEqual({
      minCutoff: 1.8,
      beta: 16.0,
      dCutoff: 1.0,
    });
    expect(CLINICAL_HAND_ONE_EURO_PRESET).toEqual({
      minCutoff: 2.2,
      beta: 20.0,
      dCutoff: 1.0,
    });

    const fastPoseFilter = new ClinicalOneEuroLandmarkFilter(
      CLINICAL_POSE_ONE_EURO_PRESET.minCutoff,
      CLINICAL_POSE_ONE_EURO_PRESET.beta,
      CLINICAL_POSE_ONE_EURO_PRESET.dCutoff,
    );
    const legacyPoseFilter = new ClinicalOneEuroLandmarkFilter();
    const dt = 1000 / 60;

    // Warm up at rest
    for (let i = 0; i < 15; i += 1) {
      const t = i * dt;
      const pose = makeStandingPose({ 16: { y: 0.68 } });
      fastPoseFilter.filterLandmarks(pose, t);
      legacyPoseFilter.filterLandmarks(pose, t);
    }

    // Fast upward lift over 12 frames (0.68 -> 0.24)
    let fastLagSum = 0;
    let legacyLagSum = 0;
    for (let step = 1; step <= 12; step += 1) {
      const t = (15 + step) * dt;
      const targetY = 0.68 - (step / 12) * 0.44;
      const pose = makeStandingPose({ 16: { y: targetY } });
      const fastOut = fastPoseFilter.filterLandmarks(pose, t);
      const legacyOut = legacyPoseFilter.filterLandmarks(pose, t);
      fastLagSum += Math.abs(fastOut[16].y - targetY);
      legacyLagSum += Math.abs(legacyOut[16].y - targetY);
    }

    expect(fastLagSum).toBeLessThan(legacyLagSum * 0.65);

    // Stationary jitter damping check (+/- 0.004 amplitude noise at 60Hz)
    const stationaryFilter = new ClinicalOneEuroLandmarkFilter(
      CLINICAL_POSE_ONE_EURO_PRESET.minCutoff,
      CLINICAL_POSE_ONE_EURO_PRESET.beta,
      CLINICAL_POSE_ONE_EURO_PRESET.dCutoff,
    );
    const handFilter = new ClinicalHandOneEuroFilter(
      CLINICAL_HAND_ONE_EURO_PRESET.minCutoff,
      CLINICAL_HAND_ONE_EURO_PRESET.beta,
      CLINICAL_HAND_ONE_EURO_PRESET.dCutoff,
    );
    let maxFilteredDev = 0;
    for (let i = 0; i < 40; i += 1) {
      const noise = i % 2 === 0 ? 0.004 : -0.004;
      const out = stationaryFilter.filterLandmarks(makeStandingPose({ 16: { y: 0.5 + noise } }), i * dt);
      const handOut = handFilter.filterHand(
        'right',
        Array.from({ length: 21 }, () => ({ x: 0.5, y: 0.5 + noise, z: 0, visibility: 1 })),
        i * dt,
      );
      if (i >= 10) {
        maxFilteredDev = Math.max(maxFilteredDev, Math.abs(out[16].y - 0.5));
        expect(Math.abs(handOut[8].y - 0.5)).toBeLessThan(0.003);
      }
    }
    expect(maxFilteredDev).toBeLessThan(0.0025);
  });

  it('computeDtNormalizedLerpAlpha stays frame-rate invariant across 30Hz, 60Hz, and 120Hz', () => {
    const a120 = computeDtNormalizedLerpAlpha(1000 / 120, 16);
    const a60 = computeDtNormalizedLerpAlpha(1000 / 60, 16);
    const a30 = computeDtNormalizedLerpAlpha(1000 / 30, 16);

    expect(a30).toBeGreaterThan(a60);
    expect(a60).toBeGreaterThan(a120);

    // Two 120Hz steps compound identically to one 60Hz step; two 60Hz steps compound to one 30Hz step
    const twoAt120 = 1 - Math.pow(1 - a120, 2);
    const twoAt60 = 1 - Math.pow(1 - a60, 2);
    expect(twoAt120).toBeCloseTo(a60, 6);
    expect(twoAt60).toBeCloseTo(a30, 6);

    // Edge cases stay safely clamped within [0.12, 1]
    expect(computeDtNormalizedLerpAlpha(0, 16)).toBe(0.12);
    expect(computeDtNormalizedLerpAlpha(-5, 16)).toBe(0.12);
    expect(computeDtNormalizedLerpAlpha(1000, 16)).toBeCloseTo(1 - Math.exp(-100 / 16), 6);
    expect(computeDtNormalizedLerpAlpha(Number.NaN, 16)).toBeCloseTo(1 - Math.exp(-16.7 / 16), 6);
    expect(computeDtNormalizedLerpAlpha(16, 0)).toBeCloseTo(1 - Math.exp(-16 / 4), 6);
  });

  it('stabilizeArmIdentity restores swapped L/R wrists together with palm extremities and ignores low-visibility jumps', () => {
    const prev = makeStandingPose({
      13: { x: 0.65, y: 0.45 },
      14: { x: 0.35, y: 0.45 },
      15: { x: 0.68, y: 0.32 },
      16: { x: 0.32, y: 0.32 },
      17: { x: 0.70, y: 0.31 },
      19: { x: 0.69, y: 0.29 },
      21: { x: 0.67, y: 0.30 },
      18: { x: 0.30, y: 0.31 },
      20: { x: 0.31, y: 0.29 },
      22: { x: 0.33, y: 0.30 },
    });

    // MediaPipe accidentally swaps Left (13/15/17/19/21) and Right (14/16/18/20/22) in next frame
    const crossedFrame = makeStandingPose({
      13: { x: 0.35, y: 0.45 },
      14: { x: 0.65, y: 0.45 },
      15: { x: 0.32, y: 0.32 },
      16: { x: 0.68, y: 0.32 },
      17: { x: 0.30, y: 0.31 },
      19: { x: 0.31, y: 0.29 },
      21: { x: 0.33, y: 0.30 },
      18: { x: 0.70, y: 0.31 },
      20: { x: 0.69, y: 0.29 },
      22: { x: 0.67, y: 0.30 },
    });

    const corrected = stabilizeArmIdentity(crossedFrame, prev);
    expect(corrected[15].x).toBeCloseTo(0.68, 3);
    expect(corrected[16].x).toBeCloseTo(0.32, 3);
    // Palm extremities swapped cleanly with wrists
    expect(corrected[17].x).toBeCloseTo(0.70, 3);
    expect(corrected[19].x).toBeCloseTo(0.69, 3);
    expect(corrected[21].x).toBeCloseTo(0.67, 3);
    expect(corrected[18].x).toBeCloseTo(0.30, 3);
    expect(corrected[20].x).toBeCloseTo(0.31, 3);
    expect(corrected[22].x).toBeCloseTo(0.33, 3);

    // Low visibility (< 0.25) must NOT trigger a swap
    const lowVisCrossed = makeStandingPose({
      15: { x: 0.32, y: 0.32, visibility: 0.1 },
      16: { x: 0.68, y: 0.32, visibility: 0.98 },
    });
    const unswapped = stabilizeArmIdentity(lowVisCrossed, prev);
    expect(unswapped[15].x).toBeCloseTo(0.32, 3);
    expect(unswapped[16].x).toBeCloseTo(0.68, 3);
  });
});

describe('Cached MediaPipe monotonic timestamps & GPU canvas zero-shadowBlur renderer', () => {
  it('enforces strictly monotonic Pose, Hand, RehabFace, and Video timestamps across simulated remounts', () => {
    resetBiometricVisionCacheForTests();

    const p1 = getNextMonotonicPoseDetectTs(1000);
    const p2 = getNextMonotonicPoseDetectTs(1000);
    const p3 = getNextMonotonicPoseDetectTs(920);
    const h1 = getNextMonotonicHandDetectTs(500);
    const h2 = getNextMonotonicHandDetectTs(450);
    const f1 = getNextMonotonicRehabFaceDetectTs(750);
    const f2 = getNextMonotonicRehabFaceDetectTs(750);
    const v1 = getNextMonotonicVideoDetectTs(300);
    const v2 = getNextMonotonicVideoDetectTs(250);

    expect(p1).toBe(1000);
    expect(p2).toBeGreaterThan(p1);
    expect(p3).toBeGreaterThan(p2);
    expect(h1).toBe(500);
    expect(h2).toBeGreaterThan(h1);
    expect(f1).toBe(750);
    expect(f2).toBeGreaterThan(f1);
    expect(v1).toBe(300);
    expect(v2).toBeGreaterThan(v1);
  });

  it('drawLivePoseOverlayOnCanvas never sets shadowBlur > 0 and requests desynchronized 2D context', () => {
    const shadowBlurValues: number[] = [];
    const contextOptionsSeen: unknown[] = [];

    const mockCtx = {
      clearRect: () => {},
      save: () => {},
      restore: () => {},
      beginPath: () => {},
      closePath: () => {},
      moveTo: () => {},
      lineTo: () => {},
      arc: () => {},
      ellipse: () => {},
      quadraticCurveTo: () => {},
      bezierCurveTo: () => {},
      rect: () => {},
      roundRect: () => {},
      clip: () => {},
      fillRect: () => {},
      strokeRect: () => {},
      stroke: () => {},
      fill: () => {},
      fillText: () => {},
      strokeText: () => {},
      measureText: (txt: string) => ({ width: txt.length * 8 }),
      setLineDash: () => {},
      translate: () => {},
      scale: () => {},
      resetTransform: () => {},
      setTransform: () => {},
      createLinearGradient: () => ({ addColorStop: () => {} }),
      createRadialGradient: () => ({ addColorStop: () => {} }),
      fillStyle: '#000',
      strokeStyle: '#000',
      lineWidth: 1,
      lineCap: 'round',
      lineJoin: 'round',
      font: '16px sans-serif',
      textAlign: 'left',
      textBaseline: 'alphabetic',
      globalAlpha: 1,
      set shadowBlur(val: number) {
        shadowBlurValues.push(val);
      },
      get shadowBlur() {
        return 0;
      },
      shadowColor: 'transparent',
    };

    const mockCanvas = {
      width: 640,
      height: 360,
      clientWidth: 640,
      clientHeight: 360,
      getContext: (_id: string, opts?: unknown) => {
        contextOptionsSeen.push(opts);
        return mockCtx;
      },
    } as unknown as HTMLCanvasElement;

    const landmarks = makeStandingPose();
    const hand21: NormalizedLandmark[] = Array.from({ length: 21 }, (_, idx) => ({
      x: 0.34 + idx * 0.002,
      y: 0.62 - idx * 0.003,
      z: -0.02,
      visibility: 1,
    }));

    drawLivePoseOverlayOnCanvas(mockCanvas, [landmarks], {
      targetJoint: 'right_shoulder',
      expectedPlane: 'any',
      romDeg: 95,
      targetRomDeg: 120,
      holdRatio: 0.6,
      trunkTiltDeg: 3,
      isMultiPerson: false,
      isImpostorOrEmergency: false,
      handLandmarks21: hand21,
      videoWidth: 640,
      videoHeight: 360,
      canvasWidth: 640,
      canvasHeight: 360,
      showSkeleton: true,
      ownerLock: {
        ownerName: 'Test Patient',
        ownerFaceVerified: true,
        faceCount: 1,
        faceBox: {
          xMin: 0.44,
          xMax: 0.56,
          yMin: 0.10,
          yMax: 0.26,
          centerX: 0.50,
          centerY: 0.18,
          width: 0.12,
          height: 0.16,
          rawPx: {
            xMin: 281.6,
            xMax: 358.4,
            yMin: 36,
            yMax: 93.6,
            centerX: 320,
            centerY: 64.8,
            width: 76.8,
            height: 57.6,
          },
        },
      },
    });

    expect(contextOptionsSeen[0]).toEqual({ desynchronized: true });
    expect(shadowBlurValues.every((v) => v <= 0)).toBe(true);
  });

  it('contains zero Unicode emojis and zero active ctx.shadowBlur > 0 calls across all updated pose/camera modules', () => {
    const emojiRe = /[\p{Extended_Pictographic}\u{FE0F}\u{2600}-\u{27BF}\u{23E9}-\u{23FA}]/u;
    const shadowBlurPosRe = /\.shadowBlur\s*=\s*[1-9]/;

    const touchedFiles = [
      'src/lib/biometricVisionCache.ts',
      'packages/pose-engine/src/oneEuroFilter.ts',
      'packages/pose-engine/src/armIdentityStabilizer.ts',
      'src/components/SmartRehabKioskUI.tsx',
      'src/components/AnimatedRehabCoachModel.tsx',
      'src/App.tsx',
      'src/components/RandomHandStartPrompt.tsx',
      'apps/kiosk/src/DynamicExerciseSessionView.tsx',
      'apps/kiosk/src/PhysicalCardGame.tsx',
      'src/__tests__/poseSmoothnessAndMediaPipe.test.ts',
    ];

    for (const relPath of touchedFiles) {
      const content = fs.readFileSync(path.resolve(process.cwd(), relPath), 'utf8');
      expect(emojiRe.test(content), `Found Unicode emoji in ${relPath}`).toBe(false);
      if (relPath !== 'src/__tests__/poseSmoothnessAndMediaPipe.test.ts') {
        expect(shadowBlurPosRe.test(content), `Found shadowBlur > 0 in ${relPath}`).toBe(false);
      }
    }
  });
});

