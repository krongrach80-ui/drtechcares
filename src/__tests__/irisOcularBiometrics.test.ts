import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  STRICT_CRANIOFACIAL_MAX_SINGLE_DEVIATION,
  STRICT_IRIS_RATIO_MAX_DEVIATION,
  STRICT_OCULAR_IRIS_ZNCC_THRESHOLD,
  STRICT_PATCH_GRAD_ZNCC_THRESHOLD,
  STRICT_RELU6_RAW_COSINE_THRESHOLD,
  computeCraniofacialRatioDeviation,
  computeIrisOcularRatios8,
  computeOcularIrisZnccScore,
  computePatchGradientZnccScore,
  evaluateAppearanceAndCraniofacialGate,
  evaluateOcularIrisGate,
  evaluateTripleGateNeuralCandidate,
  extractOcularIrisPatch256FromRgba,
  flipIrisOcularRatios8Horizontally,
  flipOcularIrisPatch256Horizontally,
  isNonNegativeRelu6Vector,
  l2Normalize,
  normalizeOcularIrisPatch256,
  type GeometricAnthropometricRatios,
} from '@drtechcares/smart-biometric-core';
import { BiologicalLivenessTracker } from '@drtechcares/face-client';
import {
  getNextMonotonicVideoDetectTs,
  resetBiometricVisionCacheForTests,
} from '../lib/biometricVisionCache';

const EMOJI_RE = /[\p{Extended_Pictographic}\u{FE0F}\u{2600}-\u{27BF}\u{23E9}-\u{23FA}]/u;

function buildSynthetic478Face(params?: {
  scale?: number;
  shiftX?: number;
  shiftY?: number;
  gazeOffsetX?: number;
  gazeOffsetY?: number;
  irisRadiusX?: number;
}) {
  const s = params?.scale ?? 1;
  const dx = params?.shiftX ?? 0;
  const dy = params?.shiftY ?? 0;
  const gx = params?.gazeOffsetX ?? 0;
  const gy = params?.gazeOffsetY ?? 0;
  const ir = (params?.irisRadiusX ?? 0.011) * s;

  const lms = Array.from({ length: 478 }, () => ({ x: 0.5 * s + dx, y: 0.5 * s + dy, z: 0 }));

  // Face boundary & symmetry landmarks
  lms[10] = { x: 0.5 * s + dx, y: 0.25 * s + dy, z: 0 };
  lms[152] = { x: 0.5 * s + dx, y: 0.78 * s + dy, z: 0 };
  lms[234] = { x: 0.32 * s + dx, y: 0.5 * s + dy, z: 0.02 };
  lms[454] = { x: 0.68 * s + dx, y: 0.5 * s + dy, z: 0.02 };
  lms[1] = { x: 0.5 * s + dx, y: 0.56 * s + dy, z: -0.05 };
  lms[61] = { x: 0.43 * s + dx, y: 0.67 * s + dy, z: 0 };
  lms[291] = { x: 0.57 * s + dx, y: 0.67 * s + dy, z: 0 };
  lms[13] = { x: 0.5 * s + dx, y: 0.665 * s + dy, z: 0 };
  lms[14] = { x: 0.5 * s + dx, y: 0.68 * s + dy, z: 0 };

  // Left eye corners + lids + brow
  lms[33] = { x: 0.39 * s + dx, y: 0.44 * s + dy, z: 0 };
  lms[133] = { x: 0.46 * s + dx, y: 0.44 * s + dy, z: 0 };
  lms[159] = { x: 0.425 * s + dx, y: 0.428 * s + dy, z: 0 };
  lms[145] = { x: 0.425 * s + dx, y: 0.452 * s + dy, z: 0 };
  lms[105] = { x: 0.425 * s + dx, y: 0.385 * s + dy, z: 0 };

  // Right eye corners + lids + brow
  lms[362] = { x: 0.54 * s + dx, y: 0.44 * s + dy, z: 0 };
  lms[263] = { x: 0.61 * s + dx, y: 0.44 * s + dy, z: 0 };
  lms[386] = { x: 0.575 * s + dx, y: 0.428 * s + dy, z: 0 };
  lms[374] = { x: 0.575 * s + dx, y: 0.452 * s + dy, z: 0 };
  lms[334] = { x: 0.575 * s + dx, y: 0.385 * s + dy, z: 0 };

  // Left iris (468..472) & right iris (473..477)
  const lCx = (0.425 + gx) * s + dx;
  const lCy = (0.44 + gy) * s + dy;
  const rCx = (0.575 + gx) * s + dx;
  const rCy = (0.44 + gy) * s + dy;
  const irY = ir * (16 / 9);

  lms[468] = { x: lCx, y: lCy, z: 0 };
  lms[469] = { x: lCx + ir, y: lCy, z: 0 };
  lms[470] = { x: lCx, y: lCy - irY, z: 0 };
  lms[471] = { x: lCx - ir, y: lCy, z: 0 };
  lms[472] = { x: lCx, y: lCy + irY, z: 0 };

  lms[473] = { x: rCx, y: rCy, z: 0 };
  lms[474] = { x: rCx + ir, y: rCy, z: 0 };
  lms[475] = { x: rCx, y: rCy - irY, z: 0 };
  lms[476] = { x: rCx - ir, y: rCy, z: 0 };
  lms[477] = { x: rCx, y: rCy + irY, z: 0 };

  return lms;
}

function buildSyntheticEyeRgba(
  seedPhase: number,
  options?: { brightnessOffset?: number; contrastScale?: number },
): { width: number; height: number; data: Uint8ClampedArray } {
  const width = 48;
  const height = 32;
  const data = new Uint8ClampedArray(width * height * 4);
  const bright = options?.brightnessOffset ?? 0;
  const contrast = options?.contrastScale ?? 1;
  const cx = (width - 1) * 0.5;
  const cy = (height - 1) * 0.5;

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const dx = x - cx;
      const dy = y - cy;
      const r = Math.hypot(dx, dy);
      const ang = Math.atan2(dy, dx);
      // Iris ring + pupil + eyelid texture unique to seedPhase
      const baseTexture =
        r < 4.5
          ? 28
          : r < 11.5
            ? 88 + Math.sin(ang * 5 + seedPhase) * 26 + Math.cos(r * 0.8 + seedPhase) * 14
            : 182 - Math.abs(dy) * 2.2 + Math.sin(x * 0.25 + seedPhase) * 10;
      const luma = Math.max(0, Math.min(255, baseTexture * contrast + bright));
      const p = (y * width + x) * 4;
      data[p] = Math.round(luma);
      data[p + 1] = Math.round(Math.max(0, Math.min(255, luma * 0.94 + seedPhase * 2)));
      data[p + 2] = Math.round(Math.max(0, Math.min(255, luma * 0.88 - seedPhase * 2)));
      data[p + 3] = 255;
    }
  }
  return { width, height, data };
}

function makeUnitVector(seed: number, dim = 1024): number[] {
  const arr = Array.from({ length: dim }, (_, i) => Math.sin((i + 1) * 0.19 + seed * 3.17));
  return l2Normalize(arr);
}

describe('Dual-Iris & Periocular Precision Biometrics + Vision Cache', () => {
  it('computes scale- and translation-invariant 8-D iris ocular ratios and rejects degenerate frames', () => {
    const base = buildSynthetic478Face({ scale: 1, shiftX: 0, shiftY: 0 });
    const shiftedScaled = buildSynthetic478Face({ scale: 0.62, shiftX: 0.14, shiftY: -0.07 });

    const rBase = computeIrisOcularRatios8(base, 16 / 9);
    const rShifted = computeIrisOcularRatios8(shiftedScaled, 16 / 9);
    expect(rBase).toBeDefined();
    expect(rBase).toHaveLength(8);
    expect(rShifted).toHaveLength(8);
    for (let i = 0; i < 8; i += 1) {
      expect(rBase![i]).toBeCloseTo(rShifted![i], 3);
    }

    // Rejects sub-478 or flat same-point fixtures
    expect(computeIrisOcularRatios8(base.slice(0, 468), 16 / 9)).toBeUndefined();
    const flatIris = base.map((p, idx) => (idx >= 468 ? { x: 0.5, y: 0.5, z: 0 } : p));
    expect(computeIrisOcularRatios8(flatIris, 16 / 9)).toBeUndefined();
  });

  it('extracts lighting-invariant 256-D dual-iris descriptor, preserves ZNCC, and separates distinct eyes', () => {
    const leftUserA = buildSyntheticEyeRgba(0.3);
    const rightUserA = buildSyntheticEyeRgba(0.9);
    const patchUserA = extractOcularIrisPatch256FromRgba(leftUserA, rightUserA);
    expect(patchUserA).not.toBeNull();
    expect(patchUserA!).toHaveLength(256);

    // Lighting shift (darker/brighter + lower contrast) should remain strongly correlated
    const leftLitA = buildSyntheticEyeRgba(0.3, { brightnessOffset: -35, contrastScale: 0.72 });
    const rightLitA = buildSyntheticEyeRgba(0.9, { brightnessOffset: -35, contrastScale: 0.72 });
    const patchLitA = extractOcularIrisPatch256FromRgba(leftLitA, rightLitA);
    expect(patchLitA).not.toBeNull();

    const samePersonZncc = computeOcularIrisZnccScore(patchUserA!, patchLitA!);
    expect(samePersonZncc).toBeGreaterThanOrEqual(0.85);

    // Distinct eye iris pattern (different angular sector & gradient structure) drops sharply
    const leftImposter = buildSyntheticEyeRgba(2.9);
    const rightImposter = buildSyntheticEyeRgba(3.7);
    const patchImposter = extractOcularIrisPatch256FromRgba(leftImposter, rightImposter);
    expect(patchImposter).not.toBeNull();

    const imposterZncc = computeOcularIrisZnccScore(patchUserA!, patchImposter!);
    expect(imposterZncc).toBeLessThan(STRICT_OCULAR_IRIS_ZNCC_THRESHOLD);
    expect(normalizeOcularIrisPatch256(patchUserA!)).toHaveLength(256);
  });

  it('performs exact horizontal involution flip on both 256-D iris patch and 8-D ocular ratios', () => {
    const leftEye = buildSyntheticEyeRgba(0.4);
    const rightEye = buildSyntheticEyeRgba(1.4);
    const patch256 = extractOcularIrisPatch256FromRgba(leftEye, rightEye)!;
    const flippedOnce = flipOcularIrisPatch256Horizontally(patch256);
    const flippedTwice = flipOcularIrisPatch256Horizontally(flippedOnce);

    expect(flippedOnce).toHaveLength(256);
    expect(flippedTwice).toEqual(patch256);

    const ratios8 = computeIrisOcularRatios8(buildSynthetic478Face(), 16 / 9)!;
    const ratiosFlippedOnce = flipIrisOcularRatios8Horizontally(ratios8);
    const ratiosFlippedTwice = flipIrisOcularRatios8Horizontally(ratiosFlippedOnce);
    expect(ratiosFlippedTwice).toEqual(ratios8);
  });

  it('enforces dual-iris gate while preserving 100% backwards-compatibility for legacy enrolled profiles', () => {
    const vecA = makeUnitVector(1.25);
    const candidateRatiosWithIris: GeometricAnthropometricRatios = {
      eyeToNoseRatio: 1.5,
      faceWidthToUpperHeight: 1.9,
      mouthToEyeRatio: 0.78,
      noseToChinRatio: 1.15,
      ocularIrisPatch256: extractOcularIrisPatch256FromRgba(
        buildSyntheticEyeRgba(0.2),
        buildSyntheticEyeRgba(0.7),
      )!,
      irisOcularRatios8: computeIrisOcularRatios8(buildSynthetic478Face(), 16 / 9),
    };

    const legacyCandidate = {
      masterEmbedding: vecA,
      angleEmbeddings: { center: vecA, left: vecA, right: vecA },
      geometricRatios: null,
    };
    // Legacy candidate without iris fields must still pass
    const legacyGate = evaluateTripleGateNeuralCandidate(vecA, legacyCandidate, candidateRatiosWithIris, 'center');
    expect(legacyGate.passed).toBe(true);

    // Same eyes + matching vectors pass additive Ocular-Iris Gate
    const enrolledIrisCandidate = {
      masterEmbedding: vecA,
      angleEmbeddings: { center: vecA, left: vecA, right: vecA },
      geometricRatios: candidateRatiosWithIris,
    };
    const matchRes = evaluateTripleGateNeuralCandidate(
      vecA,
      enrolledIrisCandidate,
      candidateRatiosWithIris,
      'center',
    );
    expect(matchRes.passed).toBe(true);
    expect(matchRes.ocularIrisZncc).toBeGreaterThanOrEqual(STRICT_OCULAR_IRIS_ZNCC_THRESHOLD);
    expect(matchRes.irisRatioDeviation).toBeLessThanOrEqual(STRICT_IRIS_RATIO_MAX_DEVIATION);

    // Different eye texture fails additive Ocular-Iris Gate even when face vector matches
    const eyeImposterRatios: GeometricAnthropometricRatios = {
      ...candidateRatiosWithIris,
      ocularIrisPatch256: extractOcularIrisPatch256FromRgba(
        buildSyntheticEyeRgba(3.1),
        buildSyntheticEyeRgba(3.9),
      )!,
    };
    const gateCheck = evaluateOcularIrisGate(eyeImposterRatios, candidateRatiosWithIris);
    expect(gateCheck.available).toBe(true);
    expect(gateCheck.passed).toBe(false);
    expect(gateCheck.failureReason).toContain('OCULAR_IRIS_ZNCC_BELOW_THRESHOLD');

    const failRes = evaluateTripleGateNeuralCandidate(
      vecA,
      enrolledIrisCandidate,
      eyeImposterRatios,
      'center',
    );
    expect(failRes.passed).toBe(false);
    expect(failRes.failureReasons.some((r) => r.includes('OCULAR_IRIS_ZNCC_BELOW_THRESHOLD'))).toBe(true);
  });

  it('verifies fast Ocular-Iris micro-saccade liveness (< 0.5s) in BiologicalLivenessTracker', () => {
    const tracker = new BiologicalLivenessTracker();
    const gazeOffsets = [0, 0.0012, -0.0011, 0.0015, -0.0009, 0.0013];
    let latestReport = tracker.update({
      landmarks3D: buildSynthetic478Face({ gazeOffsetX: 0, gazeOffsetY: 0 }),
      earLeft: 0.3,
      earRight: 0.3,
      yawDeg: 0,
      pitchDeg: 0,
      timestampMs: 1000,
    });
    for (let i = 1; i < gazeOffsets.length; i += 1) {
      latestReport = tracker.update({
        landmarks3D: buildSynthetic478Face({
          gazeOffsetX: gazeOffsets[i],
          gazeOffsetY: i % 2 === 0 ? 0.0008 : -0.0008,
        }),
        earLeft: 0.3,
        earRight: 0.3,
        yawDeg: 0,
        pitchDeg: 0,
        timestampMs: 1000 + i * 60,
      });
    }
    expect(latestReport.irisLocked).toBe(true);
    expect(latestReport.gazeShiftScore).toBeGreaterThanOrEqual(0.028);
    expect(latestReport.irisLivenessVerified).toBe(true);
    expect(latestReport.biologicalVerified).toBe(true);
    expect(latestReport.spoofDetected).toBe(false);
  });

  it('maintains strictly increasing monotonic video detect timestamps across remounts', () => {
    resetBiometricVisionCacheForTests();
    const t1 = getNextMonotonicVideoDetectTs(500);
    const t2 = getNextMonotonicVideoDetectTs(500);
    const t3 = getNextMonotonicVideoDetectTs(120); // Simulated remount / earlier performance.now()
    expect(t1).toBe(500);
    expect(t2).toBeGreaterThan(t1);
    expect(t3).toBeGreaterThan(t2);
  });

  it('rejects non-negative ReLU6 strangers at 0.86 raw cosine while accepting enrolled owners around 0.94', () => {
    // Construct two orthogonal non-negative unit vectors in positive ReLU6 orthant (negCnt === 0)
    const ownerBase = l2Normalize(
      Array.from({ length: 1024 }, (_, i) => (i < 512 ? 0.02 + Math.abs(Math.sin((i + 1) * 0.23)) : 0)),
    );
    const orthoBase = l2Normalize(
      Array.from({ length: 1024 }, (_, i) => (i >= 512 ? 0.02 + Math.abs(Math.cos((i + 1) * 0.31)) : 0)),
    );
    const sameHalfPerturb = l2Normalize(
      Array.from({ length: 1024 }, (_, i) => (i < 512 ? 0.02 + Math.abs(Math.cos((i + 1) * 0.37)) : 0)),
    );
    expect(isNonNegativeRelu6Vector(ownerBase)).toBe(true);

    // Enrolled owner shifts slightly within own ReLU6 manifold (~0.95+ cosine)
    const ownerLive = l2Normalize(ownerBase.map((v, i) => 0.92 * v + 0.14 * sameHalfPerturb[i]));
    const ownerMaster = l2Normalize(ownerBase.map((v, i) => 0.96 * v + 0.06 * sameHalfPerturb[i]));
    const candidate = {
      masterEmbedding: ownerMaster,
      angleEmbeddings: { center: ownerMaster, left: ownerMaster, right: ownerMaster },
      geometricRatios: null,
    };

    const ownerRes = evaluateTripleGateNeuralCandidate(ownerLive, candidate, null, 'center');
    expect(ownerRes.passed).toBe(true);
    expect(ownerRes.rawCosine).toBeGreaterThanOrEqual(STRICT_RELU6_RAW_COSINE_THRESHOLD);

    // Unregistered stranger blended at exact 0.86 raw cosine against ownerMaster must be rejected
    const strangerWeight = Math.sqrt(1 - 0.86 * 0.86);
    const strangerProbe = l2Normalize(
      ownerMaster.map((v, i) => 0.86 * v + strangerWeight * orthoBase[i]),
    );
    expect(isNonNegativeRelu6Vector(strangerProbe)).toBe(true);

    const strangerRes = evaluateTripleGateNeuralCandidate(strangerProbe, candidate, null, 'center');
    expect(strangerRes.passed).toBe(false);
    expect(strangerRes.rawCosine).toBeCloseTo(0.86, 2);
    expect(strangerRes.rawCosine).toBeLessThan(STRICT_RELU6_RAW_COSINE_THRESHOLD);
    expect(strangerRes.failureReasons.length).toBeGreaterThan(0);
  });

  it('validates lighting-invariant 16x16 high-pass gradient ZNCC and rejects smooth-toned imposters', () => {
    const buildStructuredPatch = (detailSign: number, shift = 0, gain = 1): number[] => {
      const out = new Array<number>(384).fill(0);
      for (let y = 0; y < 16; y += 1) {
        for (let x = 0; x < 16; x += 1) {
          const base = 0.22 + 0.52 * (y / 15);
          const detail = detailSign * 0.065 * Math.sin(x * 2.2) * Math.cos(y * 2.2);
          const val = Math.max(0, Math.min(1, (base + detail) * gain + shift));
          out[y * 16 + x] = Number(val.toFixed(4));
        }
      }
      for (let i = 256; i < 384; i += 1) {
        out[i] = Number((0.45 + 0.08 * Math.sin(i * 0.15)).toFixed(4));
      }
      return out;
    };

    const ownerPatch = buildStructuredPatch(1);
    const litOwnerPatch = buildStructuredPatch(1, 0.08, 0.85);
    const differentStructurePatch = buildStructuredPatch(-1);

    const sameGradZncc = computePatchGradientZnccScore(ownerPatch, litOwnerPatch);
    const diffGradZncc = computePatchGradientZnccScore(ownerPatch, differentStructurePatch);
    expect(sameGradZncc).toBeGreaterThanOrEqual(0.85);
    expect(diffGradZncc).toBeLessThan(STRICT_PATCH_GRAD_ZNCC_THRESHOLD);

    const gateRes = evaluateAppearanceAndCraniofacialGate(
      {
        eyeToNoseRatio: 1.5,
        faceWidthToUpperHeight: 1.9,
        mouthToEyeRatio: 0.78,
        noseToChinRatio: 1.15,
        appearancePatch384: differentStructurePatch,
      },
      {
        eyeToNoseRatio: 1.5,
        faceWidthToUpperHeight: 1.9,
        mouthToEyeRatio: 0.78,
        noseToChinRatio: 1.15,
        appearancePatch384: ownerPatch,
      },
    );
    expect(gateRes.passed).toBe(false);
    expect(gateRes.failureReason).toContain('PATCH_GRAD_ZNCC_BELOW_THRESHOLD');
  });

  it('rejects single-bone craniofacial outlier even when 12-D mean deviation stays below 0.065', () => {
    const enrolledRatios12 = Array.from({ length: 12 }, (_, i) => Number((1.0 + i * 0.04).toFixed(4)));
    const outlierProbe12 = [...enrolledRatios12];
    // Perturb ONLY bone 0 by +0.22: meanDeviation = 0.22/12 (~0.0183 <= 0.065), maxSingleDeviation = 0.22 (> 0.145)
    outlierProbe12[0] = Number((outlierProbe12[0] + 0.22).toFixed(4));

    const outlierGeom: GeometricAnthropometricRatios = {
      eyeToNoseRatio: 1.5,
      faceWidthToUpperHeight: 1.9,
      mouthToEyeRatio: 0.78,
      noseToChinRatio: 1.15,
      craniofacialRatios12: outlierProbe12,
    };
    const enrolledGeom: GeometricAnthropometricRatios = {
      eyeToNoseRatio: 1.5,
      faceWidthToUpperHeight: 1.9,
      mouthToEyeRatio: 0.78,
      noseToChinRatio: 1.15,
      craniofacialRatios12: enrolledRatios12,
    };

    const dev = computeCraniofacialRatioDeviation(outlierGeom, enrolledGeom);
    expect(dev.available).toBe(true);
    expect(dev.meanDeviation).toBeLessThanOrEqual(0.065);
    expect(dev.maxSingleDeviation).toBeGreaterThan(STRICT_CRANIOFACIAL_MAX_SINGLE_DEVIATION);
    expect(dev.passed).toBe(false);

    const gateRes = evaluateAppearanceAndCraniofacialGate(outlierGeom, enrolledGeom);
    expect(gateRes.passed).toBe(false);
    expect(gateRes.failureReason).toContain('CRANIOFACIAL_RATIO_MISMATCH');
  });

  it('audits all iris biometric and cache files for zero Unicode emojis', () => {
    const root = process.cwd();
    const files = [
      'packages/smart-biometric-core/src/vectorMath.ts',
      'packages/face-client/src/biologicalLiveness.ts',
      'packages/api-client/src/cloudRouter.ts',
      'src/lib/biometricEngine.ts',
      'src/lib/biometricVisionCache.ts',
      'src/components/FaceID120Scanner.tsx',
      'src/components/KioskBiometricLoginView.tsx',
      'src/App.tsx',
      'src/__tests__/irisOcularBiometrics.test.ts',
      'src/__tests__/kioskBiometricLogin.test.ts',
    ];
    for (const rel of files) {
      const content = fs.readFileSync(path.join(root, rel), 'utf8');
      expect(EMOJI_RE.test(content), `Unexpected emoji in ${rel}`).toBe(false);
    }
  });
});

