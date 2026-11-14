import { describe, expect, it } from 'vitest';
import {
  resolveTodayPatientSchedule,
  type PatientBiometricIndexEntry,
} from '@drtechcares/api-client';
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
import { chooseCaptureBucket, classifyDisplayBucket } from '../components/FaceID120Scanner';

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

  it('recognizes the same patient even when clothing/border/lighting indices (0..255) change drastically', () => {
    const buildHybridLikePerson = (seed: number, shirtBias: number, lightingGain: number): number[] => {
      const vec = new Array<number>(FACE_EMBEDDING_DIM).fill(0);
      let state = seed >>> 0;
      const nextRand = (): number => {
        state = (state * 1664525 + 1013904223) >>> 0;
        return (state % 2000) / 1000 - 1;
      };

      // 0..127 DCT: u=0 or v=0 carries heavy shirt/background gradient; interior u>=1,v>=1 carries face structure
      let dctIdx = 0;
      for (let u = 0; u < 12 && dctIdx < 128; u += 1) {
        for (let v = 0; v < 12 && dctIdx < 128; v += 1) {
          if (u === 0 && v === 0) continue;
          const innerWave = nextRand() * 0.32 * lightingGain;
          vec[dctIdx] =
            u === 0 || v === 0
              ? shirtBias * (0.85 - u * 0.04) * (v % 2 === 0 ? 1 : -0.7)
              : CANONICAL_HUMAN_FACE_BASELINE_512[dctIdx] * 0.4 + innerWave;
          dctIdx += 1;
        }
      }

      // 128..255 Patches: outer 12 patches carry shirt/background; inner 4 patches carry inner facial gradients
      for (let py = 0; py < 4; py += 1) {
        for (let px = 0; px < 4; px += 1) {
          const offset = 128 + (py * 4 + px) * 8;
          const isInner = py >= 1 && py <= 2 && px >= 1 && px <= 2;
          for (let slot = 0; slot < 8; slot += 1) {
            const innerWave = nextRand() * 0.25 * lightingGain;
            vec[offset + slot] = isInner && slot <= 2
              ? CANONICAL_HUMAN_FACE_BASELINE_512[offset + slot] * 0.45 + innerWave
              : shirtBias * ((slot % 3) - 1) * 0.9;
          }
        }
      }

      // 256..511: 478-point 3D craniofacial landmark ratios
      for (let i = 256; i < FACE_EMBEDDING_DIM; i += 1) {
        const cranioWave = nextRand() * 0.04;
        vec[i] = 0.65 * CANONICAL_HUMAN_FACE_BASELINE_512[i] + cranioWave;
      }

      return l2Normalize(vec);
    };

    const enrolledPerson = {
      ...makeEnrolledPatient('pat-clothes-a', 'HN-77001', 'ปิยะ ทดสอบชุด', 777),
      masterEmbedding: buildHybridLikePerson(7777, 1.4, 1.0),
      angleEmbeddings: {
        center: buildHybridLikePerson(7777, 1.4, 1.0),
        left: buildHybridLikePerson(7777, 1.35, 1.0),
        right: buildHybridLikePerson(7777, 1.45, 1.0),
        up: buildHybridLikePerson(7777, 1.38, 1.0),
        down: buildHybridLikePerson(7777, 1.42, 1.0),
      },
    };
    const otherEnrolled = {
      ...makeEnrolledPatient('pat-clothes-b', 'HN-77002', 'ผู้ป่วยอีกคน', 888),
      masterEmbedding: buildHybridLikePerson(88888, 1.4, 1.0),
      angleEmbeddings: {
        center: buildHybridLikePerson(88888, 1.4, 1.0),
        left: buildHybridLikePerson(88888, 1.4, 1.0),
        right: buildHybridLikePerson(88888, 1.4, 1.0),
        up: buildHybridLikePerson(88888, 1.4, 1.0),
        down: buildHybridLikePerson(88888, 1.4, 1.0),
      },
    };

    // Same person changes from tank top (+1.4) to bright blue T-shirt (-2.2) and different lighting (1.8x),
    // plus slight 3D landmark capture jitter.
    const samePersonNewShirt = buildHybridLikePerson(7777, -2.2, 1.8).map((v, idx) =>
      idx >= 256 ? v + ((idx % 5) - 2) * 0.0012 : v,
    );
    const sameRes = matchMultiAngleEmbedding(l2Normalize(samePersonNewShirt), [
      enrolledPerson,
      otherEnrolled,
    ]);
    expect(sameRes.matched).toBe(true);
    expect(sameRes.bestCandidate?.patientId).toBe('pat-clothes-a');
    expect(sameRes.confidencePercent).toBeGreaterThanOrEqual(80);

    // Stranger wearing the EXACT same shirt & background (+1.4) must still be rejected
    const strangerSameShirt = buildHybridLikePerson(999999, 1.4, 1.0);
    const strangerRes = matchMultiAngleEmbedding(strangerSameShirt, [enrolledPerson, otherEnrolled]);
    expect(strangerRes.matched).toBe(false);
  });

  it('enforces strict 5-angle pose capture in FaceID120Scanner and blocks frontal-only spillover', () => {
    expect(classifyDisplayBucket(3, 0)).toBe('center');
    expect(classifyDisplayBucket(-10, 0)).toBe('left');
    expect(classifyDisplayBucket(11, 0)).toBe('right');
    expect(classifyDisplayBucket(0, -8)).toBe('up');
    expect(classifyDisplayBucket(0, 8)).toBe('down');

    const afterCenterFull = { center: 40, left: 0, right: 0, up: 0, down: 0 };
    // Staring straight ahead (Yaw 3, Pitch 0) MUST NOT spill into left/right/up/down
    expect(chooseCaptureBucket(0, 0, { center: 10, left: 0, right: 0, up: 0, down: 0 })).toBe('center');
    expect(chooseCaptureBucket(3, 0, afterCenterFull)).toBeNull();
    expect(chooseCaptureBucket(-5, 3, afterCenterFull)).toBeNull();

    // Directional poses only fill their corresponding angle buckets
    expect(chooseCaptureBucket(-9, 0, afterCenterFull)).toBe('left');
    expect(chooseCaptureBucket(9, 0, afterCenterFull)).toBe('right');
    expect(chooseCaptureBucket(0, -7, afterCenterFull)).toBe('up');
    expect(chooseCaptureBucket(0, 7, afterCenterFull)).toBe('down');
  });

  it('resolves today appointment status and exercise list accurately for rest vs scheduled days', () => {
    const fixedNow = new Date(2026, 10, 14, 10, 0, 0); // 2026-11-14 local time (Saturday)

    // 1. Newly enrolled patient with future nextAppointmentDate (2026-11-16) and empty rehabSchedules -> rest day
    const futureOnly = resolveTodayPatientSchedule({
      rehabSchedules: [],
      todayRoutine: {
        dayIndex: 6,
        dayType: 'rest',
        activityType: 'physio_exercise',
        difficulty: 'medium',
        label: 'Rest Day',
        exercises: [],
      },
      nextAppointmentDate: '2026-11-16',
      now: fixedNow,
    });
    expect(futureOnly.todayIso).toBe('2026-11-14');
    expect(futureOnly.hasAppointmentToday).toBe(false);
    expect(futureOnly.todayExercises).toEqual([]);

    // 2. Patient with explicit rehabSchedules only on a future date -> rest day today
    const nonMatchingRehab = resolveTodayPatientSchedule({
      rehabSchedules: [
        {
          id: 'sched-future',
          patientId: 'pat-a',
          exerciseId: 'ex-01',
          scheduledDate: '2026-11-16',
          sets: 3,
          reps: 10,
          status: 'ACTIVE',
        },
      ],
      nextAppointmentDate: '2026-11-16',
      now: fixedNow,
    });
    expect(nonMatchingRehab.hasAppointmentToday).toBe(false);
    expect(nonMatchingRehab.todayExercises).toEqual([]);

    // 3. Patient with ACTIVE rehabSchedule for today (2026-11-14) -> active appointment with exercise list
    const scheduledToday = resolveTodayPatientSchedule({
      rehabSchedules: [
        {
          id: 'sched-today-1',
          patientId: 'pat-a',
          exerciseId: 'ex-01',
          scheduledDate: '2026-11-14',
          sets: 3,
          reps: 10,
          status: 'ACTIVE',
        },
        {
          id: 'sched-rest-ignored',
          patientId: 'pat-a',
          exerciseId: 'ex-02',
          scheduledDate: '2026-11-14',
          sets: 2,
          reps: 8,
          status: 'REST',
        },
      ],
      nextAppointmentDate: '2026-11-14',
      now: fixedNow,
    });
    expect(scheduledToday.hasAppointmentToday).toBe(true);
    expect(scheduledToday.todayExercises).toHaveLength(1);
    expect(scheduledToday.todayExercises[0].exerciseId).toBe('ex-01');
    expect(scheduledToday.todayExercises[0].nameTh).toContain('ยกแขนตรงไปด้านหน้า');
    expect(scheduledToday.todayExercises[0].sets).toBe(3);
    expect(scheduledToday.todayExercises[0].reps).toBe(10);
  });
});
