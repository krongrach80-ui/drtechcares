import { describe, expect, it } from 'vitest';
import type { PatientBiometricIndexEntry } from '@drtechcares/api-client';
import {
  CANONICAL_HUMAN_FACE_BASELINE_512,
  FACE_EMBEDDING_DIM,
  l2Normalize,
} from '@drtechcares/smart-biometric-core';
import {
  TemporalBiometricMatcher,
  computeBoundingBoxFromLandmarks,
  cosineSimilarity,
  matchMultiAngleEmbedding,
} from '../lib/faceMatcher';

function makeRealisticFaceVector(seed: number, angleBias = 0): number[] {
  const raw = new Array<number>(FACE_EMBEDDING_DIM);
  let state = seed >>> 0;
  for (let i = 0; i < FACE_EMBEDDING_DIM; i += 1) {
    state = (state * 1664525 + 1013904223 + i * 17) >>> 0;
    const identityWave = ((state % 2000) / 1000 - 1) * 0.42;
    const anglePerturb = angleBias > 0 && i % 6 === angleBias ? 0.28 : 0;
    raw[i] = CANONICAL_HUMAN_FACE_BASELINE_512[i] * 0.52 + identityWave + anglePerturb;
  }
  return l2Normalize(raw);
}

function makeEnrolledPatient(
  patientId: string,
  hn: string,
  fullName: string,
  seed: number,
): PatientBiometricIndexEntry {
  return {
    patientId,
    hn,
    fullName,
    diagnosis: 'Knee Rehab',
    assignedPtId: 'pt-1',
    nextAppointmentDate: '2026-11-20',
    framesCaptured: 120,
    qualityScore: 0.94,
    masterEmbedding: makeRealisticFaceVector(seed, 0),
    angleEmbeddings: {
      center: makeRealisticFaceVector(seed, 1),
      left: makeRealisticFaceVector(seed, 2),
      right: makeRealisticFaceVector(seed, 3),
      up: makeRealisticFaceVector(seed, 4),
      down: makeRealisticFaceVector(seed, 5),
    },
  };
}

describe('120-Frame 3D Multi-Angle Face Matcher', () => {
  const patientA = makeEnrolledPatient('pat-a', 'HN-10001', 'สมชาย ใจดี', 11111);
  const patientB = makeEnrolledPatient('pat-b', 'HN-10002', 'สมหญิง รักดี', 99999);
  const index = [patientA, patientB];

  it('computes normalized cosine similarity accurately', () => {
    const v1 = makeRealisticFaceVector(42);
    expect(cosineSimilarity(v1, v1)).toBeCloseTo(1, 5);
    expect(cosineSimilarity([1, 0, 0], [0, 1, 0])).toBe(0);
  });

  it('matches tilted left/up probe against 5-angle centroids even when rotated', () => {
    // Probe resembles patientA's left-turned centroid plus minor live sensor noise
    const noisyLeftProbe = l2Normalize(
      patientA.angleEmbeddings.left.map((v, i) => v + (i % 7 === 0 ? 0.02 : -0.01)),
    );

    const res = matchMultiAngleEmbedding(noisyLeftProbe, index);
    expect(res.matched).toBe(true);
    expect(res.bestCandidate?.patientId).toBe('pat-a');
    expect(res.bestAngle).toBe('left');
    expect(res.similarity).toBeGreaterThanOrEqual(0.78);
    expect(res.margin).toBeGreaterThanOrEqual(0.06);
  });

  it('rejects unknown stranger embedding without false positives', () => {
    const strangerProbe = makeRealisticFaceVector(55555);
    const res = matchMultiAngleEmbedding(strangerProbe, index);
    expect(res.matched).toBe(false);
  });

  it('requires 5 consecutive live frames and rejects flat or static photo spoofs', () => {
    const matcher = new TemporalBiometricMatcher(5);
    const probe = patientB.angleEmbeddings.right;

    // 4 frames -> not yet confirmed
    for (let i = 0; i < 4; i += 1) {
      const r = matcher.evaluateFrame(
        {
          probeVector: probe,
          depthScore: 0.62,
          yawDeg: 14 + i * 0.4,
          pitchDeg: -2 + i * 0.2,
          microMotion: 0.002,
          blinkDetected: i === 2,
        },
        index,
      );
      expect(r.confirmed).toBe(false);
      expect(r.streakCount).toBe(i + 1);
    }

    // 5th consecutive live frame -> confirmed!
    const confirmed = matcher.evaluateFrame(
      {
        probeVector: probe,
        depthScore: 0.64,
        yawDeg: 15.8,
        pitchDeg: -1.1,
        microMotion: 0.0024,
      },
      index,
    );
    expect(confirmed.confirmed).toBe(true);
    expect(confirmed.streakCount).toBe(5);
    expect(confirmed.candidate?.patientId).toBe('pat-b');
    expect(confirmed.livenessPassed).toBe(true);

    // Reject flat printout spoof
    const flatMatcher = new TemporalBiometricMatcher(5);
    let lastFlat = flatMatcher.evaluateFrame(
      {
        probeVector: probe,
        depthScore: 0.08,
        yawDeg: 14,
        pitchDeg: 0,
        microMotion: 0.002,
      },
      index,
    );
    for (let i = 1; i < 5; i += 1) {
      lastFlat = flatMatcher.evaluateFrame(
        {
          probeVector: probe,
          depthScore: 0.08,
          yawDeg: 14 + i * 0.5,
          pitchDeg: 0,
          microMotion: 0.002,
        },
        index,
      );
    }
    expect(lastFlat.confirmed).toBe(false);
    expect(lastFlat.livenessPassed).toBe(false);
    expect(lastFlat.livenessReason).toBe('FLAT_SPOOF_SUSPECTED');

    // Reject completely static printed image with zero micro-motion/pose variance
    const staticMatcher = new TemporalBiometricMatcher(5);
    let lastStatic = staticMatcher.evaluateFrame(
      {
        probeVector: probe,
        depthScore: 0.55,
        yawDeg: 14,
        pitchDeg: 0,
        microMotion: 0,
      },
      index,
    );
    for (let i = 1; i < 5; i += 1) {
      lastStatic = staticMatcher.evaluateFrame(
        {
          probeVector: probe,
          depthScore: 0.55,
          yawDeg: 14,
          pitchDeg: 0,
          microMotion: 0,
        },
        index,
      );
    }
    expect(lastStatic.confirmed).toBe(false);
    expect(lastStatic.livenessPassed).toBe(false);
    expect(lastStatic.livenessReason).toBe('STATIC_PHOTO_SUSPECTED');
  });

  it('computes normalized HUD bounding box from landmarks', () => {
    const lms = Array.from({ length: 420 }, (_, i) => ({
      x: 0.3 + (i % 20) * 0.02,
      y: 0.2 + Math.floor(i / 20) * 0.025,
    }));
    const box = computeBoundingBoxFromLandmarks(lms);
    expect(box).not.toBeNull();
    expect(box!.width).toBeGreaterThan(0.35);
    expect(box!.height).toBeGreaterThan(0.45);
    expect(box!.confidence).toBeGreaterThan(0.9);
  });
});
