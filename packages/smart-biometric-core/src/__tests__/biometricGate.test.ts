import { describe, expect, it } from 'vitest';
import {
  CANONICAL_HUMAN_FACE_BASELINE_512,
  FACE_EMBEDDING_DIM,
  NOT_FOUND_CONFIRM_ROUNDS,
  computeOutlierRejectedCentroid,
  l2Normalize,
  verifyHybridBiometricIdentity,
  whitenEmbedding,
  type EnrolledTemplateCandidate,
} from '../index';

function buildSyntheticPersonVector(personSeed: number, noiseScale = 0): number[] {
  const vec = new Array<number>(FACE_EMBEDDING_DIM);
  const freqA = 0.13 + (personSeed % 7) * 0.19;
  const freqB = 0.17 + (personSeed % 11) * 0.23;
  for (let i = 0; i < FACE_EMBEDDING_DIM; i += 1) {
    const base = CANONICAL_HUMAN_FACE_BASELINE_512[i];
    let identityResidual = 0;
    if (i < 128) {
      identityResidual = Math.sin(personSeed * 1.7 + i * freqA) * 0.045;
    } else if (i < 256) {
      identityResidual = Math.cos(personSeed * 2.3 + i * freqB) * 0.04;
    } else {
      identityResidual = Math.sin(personSeed * 1.1 + i * (0.09 + (personSeed % 5) * 0.11)) * 0.008;
    }
    const noise = noiseScale > 0 ? Math.sin(i * 13.7 + personSeed) * noiseScale : 0;
    vec[i] = base * 0.92 + identityResidual + noise;
  }
  return l2Normalize(vec);
}

describe('Zero False-Positive Biometric Core', () => {
  it('enforces single-scan decisive constant NOT_FOUND_CONFIRM_ROUNDS = 1', () => {
    expect(NOT_FOUND_CONFIRM_ROUNDS).toBe(1);
  });

  it('accepts genuine enrolled patient and rejects strangers (<50% confidence) despite high common-mode similarity', () => {
    const aliceEnroll = buildSyntheticPersonVector(11);
    const aliceLiveProbe = buildSyntheticPersonVector(11, 0.0015);
    const bobEnroll = buildSyntheticPersonVector(47);
    const strangerProbe = buildSyntheticPersonVector(99);

    const enrolled: EnrolledTemplateCandidate[] = [
      { patientId: 'pat-alice', patientName: 'Alice', pose: 'frontal_centroid', vector: aliceEnroll },
      { patientId: 'pat-bob', patientName: 'Bob', pose: 'frontal_centroid', vector: bobEnroll },
    ];

    const genuineRes = verifyHybridBiometricIdentity(aliceLiveProbe, enrolled);
    expect(genuineRes.matched).toBe(true);
    expect(genuineRes.patientId).toBe('pat-alice');
    expect(genuineRes.confidencePercent).toBeGreaterThanOrEqual(90);

    const strangerRes = verifyHybridBiometricIdentity(strangerProbe, enrolled);
    expect(strangerRes.matched).toBe(false);
    expect(strangerRes.patientId).toBeNull();
    expect(strangerRes.confidencePercent).toBeLessThan(50);
    expect(strangerRes.failureReasons.length).toBeGreaterThan(0);

    // Verify whitening separates distinct individuals cleanly
    const wAlice = whitenEmbedding(aliceEnroll);
    const wStranger = whitenEmbedding(strangerProbe);
    let dot = 0;
    for (let i = 0; i < FACE_EMBEDDING_DIM; i += 1) dot += wAlice[i] * wStranger[i];
    expect(dot).toBeLessThan(0.5);
  });

  it('excludes non-frontal poses (left/right/nod) during frontal login verification', () => {
    const aliceSide = buildSyntheticPersonVector(11);
    const probe = buildSyntheticPersonVector(11);
    const enrolled: EnrolledTemplateCandidate[] = [
      { patientId: 'pat-alice', patientName: 'Alice', pose: 'left', vector: aliceSide },
      { patientId: 'pat-alice', patientName: 'Alice', pose: 'right', vector: aliceSide },
      { patientId: 'pat-alice', patientName: 'Alice', pose: 'nod', vector: aliceSide },
    ];

    const res = verifyHybridBiometricIdentity(probe, enrolled);
    expect(res.matched).toBe(false);
    expect(res.confidencePercent).toBe(0);
  });

  it('rejects pre-countdown stale frames and outlier frames in burst stabilization', () => {
    const cleanVec = buildSyntheticPersonVector(7);
    const outlierVec = buildSyntheticPersonVector(88);

    const res = computeOutlierRejectedCentroid([
      { vector: outlierVec, qualityScore: 10, capturedAfterCountdown: false },
      { vector: cleanVec, qualityScore: 8, capturedAfterCountdown: true },
      { vector: cleanVec, qualityScore: 9, capturedAfterCountdown: true },
      { vector: outlierVec, qualityScore: 4, capturedAfterCountdown: true },
      { vector: cleanVec, qualityScore: 9, capturedAfterCountdown: true },
    ]);

    expect(res.acceptedCount).toBe(3);
    expect(res.rejectedCount).toBe(1);
  });
});
