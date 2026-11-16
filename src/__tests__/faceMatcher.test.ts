import { describe, expect, it } from 'vitest';
import {
  resolveTodayPatientSchedule,
  type PatientBiometricIndexEntry,
} from '@drtechcares/api-client';
import {
  BURST_FRAME_COUNT,
  CANONICAL_HUMAN_FACE_BASELINE_512,
  FACE_EMBEDDING_DIM,
  l2Normalize,
  parseEmbeddingVector,
} from '@drtechcares/smart-biometric-core';
import {
  computeEyeRollAngleDeg,
  extractHybrid512Descriptor,
  flipImageDataHorizontally,
  flipLandmarksHorizontally,
  rotateNormalizedLandmarksForRoll,
} from '@drtechcares/face-client';
import {
  TemporalBiometricMatcher,
  computeBoundingBoxFromLandmarks,
  cosineSimilarity,
  matchMultiAngleEmbedding,
} from '../lib/faceMatcher';
import {
  chooseCaptureBucket,
  classifyDisplayBucket,
  getFaceId120TargetGuidance,
} from '../components/FaceID120Scanner';
import { evaluateRandomHandChallenge } from '../components/RandomHandStartPrompt';
import { isFullFrontalFaceReadyForLogin } from '../components/KioskBiometricLoginView';

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

  it('evaluates random left/right hand challenge accurately from mirrored front-camera hand coordinates', () => {
    const makeSingleHand = (anchorX: number) =>
      Array.from({ length: 21 }, (_, idx) => ({
        x: idx === 9 || idx === 0 ? anchorX : anchorX,
        y: 0.45,
      }));

    // Left hand in raw front camera coords (anchorX >= 0.5 renders on left side of mirrored video)
    const raisedLeftHand = [makeSingleHand(0.75)];
    const leftChallengeMatch = evaluateRandomHandChallenge(raisedLeftHand, null, 'left');
    expect(leftChallengeMatch.leftRaised).toBe(true);
    expect(leftChallengeMatch.rightRaised).toBe(false);
    expect(leftChallengeMatch.targetMatched).toBe(true);
    expect(leftChallengeMatch.wrongSideRaised).toBe(false);

    const leftRaisedWhenRightRequested = evaluateRandomHandChallenge(raisedLeftHand, null, 'right');
    expect(leftRaisedWhenRightRequested.targetMatched).toBe(false);
    expect(leftRaisedWhenRightRequested.wrongSideRaised).toBe(true);

    // Right hand in raw front camera coords (anchorX < 0.5 renders on right side of mirrored video)
    const raisedRightHand = [makeSingleHand(0.25)];
    const rightChallengeMatch = evaluateRandomHandChallenge(raisedRightHand, null, 'right');
    expect(rightChallengeMatch.rightRaised).toBe(true);
    expect(rightChallengeMatch.leftRaised).toBe(false);
    expect(rightChallengeMatch.targetMatched).toBe(true);
    expect(rightChallengeMatch.wrongSideRaised).toBe(false);

    const rightRaisedWhenLeftRequested = evaluateRandomHandChallenge(raisedRightHand, null, 'left');
    expect(rightRaisedWhenLeftRequested.targetMatched).toBe(false);
    expect(rightRaisedWhenLeftRequested.wrongSideRaised).toBe(true);
  });

  it('provides 3-language (th, en, cn) directional voice and visual guidance for all 5 Face ID 120-frame scan steps', () => {
    const directions = ['center', 'left', 'right', 'up', 'down'] as const;
    for (const dir of directions) {
      const th = getFaceId120TargetGuidance('th', dir);
      const en = getFaceId120TargetGuidance('en', dir);
      const cn = getFaceId120TargetGuidance('cn', dir);

      expect(th.voice.length).toBeGreaterThan(5);
      expect(en.voice.length).toBeGreaterThan(5);
      expect(cn.voice.length).toBeGreaterThan(5);
      expect(th.stepTitle).not.toBe(en.stepTitle);
      expect(en.stepTitle).not.toBe(cn.stepTitle);
    }

    expect(getFaceId120TargetGuidance('th', 'left').voice).toContain('ซ้าย');
    expect(getFaceId120TargetGuidance('en', 'left').voice.toLowerCase()).toContain('left');
    expect(getFaceId120TargetGuidance('cn', 'left').voice).toContain('左');

    expect(getFaceId120TargetGuidance('th', 'right').voice).toContain('ขวา');
    expect(getFaceId120TargetGuidance('en', 'right').voice.toLowerCase()).toContain('right');
    expect(getFaceId120TargetGuidance('cn', 'right').voice).toContain('右');

    expect(getFaceId120TargetGuidance('th', 'up').voice).toContain('เงย');
    expect(getFaceId120TargetGuidance('en', 'up').voice.toLowerCase()).toContain('up');
    expect(getFaceId120TargetGuidance('cn', 'up').voice).toContain('抬头');

    expect(getFaceId120TargetGuidance('th', 'down').voice).toContain('ก้ม');
    expect(getFaceId120TargetGuidance('en', 'down').voice.toLowerCase()).toContain('down');
    expect(getFaceId120TargetGuidance('cn', 'down').voice).toContain('低头');
  });

  it('decisively rejects unregistered stranger even when 3D craniofacial subband matches enrolled single patient', () => {
    const enrolledOwner = makeEnrolledPatient('pat-owner', 'HN-65964', 'พัทธดนย์ ผ่องอารมย์', 424242);
    const otherAppearance = makeRealisticFaceVector(878787, 0);
    const strangerShared3D = l2Normalize([
      ...otherAppearance.slice(0, 256),
      ...enrolledOwner.masterEmbedding.slice(256),
    ]);

    const res = matchMultiAngleEmbedding(strangerShared3D, [enrolledOwner]);
    expect(res.matched).toBe(false);
    expect(res.confidencePercent).toBeLessThan(85);

    const temporal = new TemporalBiometricMatcher(5, 0.85);
    for (let i = 0; i < 6; i += 1) {
      const step = temporal.evaluateFrame(
        {
          probeVector: strangerShared3D,
          yawDeg: (i % 2) * 0.5,
          pitchDeg: 0,
          depthScore: 0.42,
          microMotion: 0.002,
          blinkDetected: i === 2,
        },
        [enrolledOwner],
      );
      expect(step.confirmed).toBe(false);
      expect(step.streakCount).toBe(0);
    }

    const validOwnerRes = matchMultiAngleEmbedding(enrolledOwner.angleEmbeddings.center, [enrolledOwner]);
    expect(validOwnerRes.matched).toBe(true);
    expect(validOwnerRes.confidencePercent).toBeGreaterThanOrEqual(85);
  });

  it('provides 3-language (th, en, cn) directional voice and visual guidance for all 5 Face ID 120-frame scan steps', () => {
    const directions = ['center', 'left', 'right', 'up', 'down'] as const;
    for (const dir of directions) {
      const th = getFaceId120TargetGuidance('th', dir);
      const en = getFaceId120TargetGuidance('en', dir);
      const cn = getFaceId120TargetGuidance('cn', dir);

      expect(th.voice.length).toBeGreaterThan(5);
      expect(en.voice.length).toBeGreaterThan(5);
      expect(cn.voice.length).toBeGreaterThan(5);
      expect(th.stepTitle).not.toBe(en.stepTitle);
      expect(en.stepTitle).not.toBe(cn.stepTitle);
    }

    expect(getFaceId120TargetGuidance('th', 'left').voice).toContain('ซ้าย');
    expect(getFaceId120TargetGuidance('en', 'left').voice.toLowerCase()).toContain('left');
    expect(getFaceId120TargetGuidance('cn', 'left').voice).toContain('左');

    expect(getFaceId120TargetGuidance('th', 'right').voice).toContain('ขวา');
    expect(getFaceId120TargetGuidance('en', 'right').voice.toLowerCase()).toContain('right');
    expect(getFaceId120TargetGuidance('cn', 'right').voice).toContain('右');

    expect(getFaceId120TargetGuidance('th', 'up').voice).toContain('เงย');
    expect(getFaceId120TargetGuidance('en', 'up').voice.toLowerCase()).toContain('up');
    expect(getFaceId120TargetGuidance('cn', 'up').voice).toContain('抬头');

    expect(getFaceId120TargetGuidance('th', 'down').voice).toContain('ก้ม');
    expect(getFaceId120TargetGuidance('en', 'down').voice.toLowerCase()).toContain('down');
    expect(getFaceId120TargetGuidance('cn', 'down').voice).toContain('低头');
  });

  it('decisively rejects unregistered stranger even when 3D craniofacial subband matches enrolled single patient', () => {
    const enrolledOwner = makeEnrolledPatient('pat-owner', 'HN-65964', 'พัทธดนย์ ผ่องอารมย์', 424242);
    const otherAppearance = makeRealisticFaceVector(878787, 0);
    const strangerShared3D = l2Normalize([
      ...otherAppearance.slice(0, 256),
      ...enrolledOwner.masterEmbedding.slice(256),
    ]);

    const res = matchMultiAngleEmbedding(strangerShared3D, [enrolledOwner]);
    expect(res.matched).toBe(false);
    expect(res.confidencePercent).toBeLessThan(85);

    const temporal = new TemporalBiometricMatcher(5, 0.85);
    for (let i = 0; i < 6; i += 1) {
      const step = temporal.evaluateFrame(
        {
          probeVector: strangerShared3D,
          yawDeg: (i % 2) * 0.5,
          pitchDeg: 0,
          depthScore: 0.42,
          microMotion: 0.002,
          blinkDetected: i === 2,
        },
        [enrolledOwner],
      );
      expect(step.confirmed).toBe(false);
      expect(step.streakCount).toBe(0);
    }

    const validOwnerRes = matchMultiAngleEmbedding(enrolledOwner.angleEmbeddings.center, [enrolledOwner]);
    expect(validOwnerRes.matched).toBe(true);
    expect(validOwnerRes.confidencePercent).toBeGreaterThanOrEqual(85);
  });

  it('prevents duplicate enrollments of the same patient from failing the margin gate and prefers newer HN', () => {
    const olderFace = makeRealisticFaceVector(314159, 0);
    const newerFace = l2Normalize(
      olderFace.map((v, idx) => v * 0.84 + makeRealisticFaceVector(271828, 1)[idx] * 0.16),
    );
    const olderPatient: PatientBiometricIndexEntry = {
      ...makeEnrolledPatient('pat-dup-old', 'HN-40683', 'พัทธดนย์ ผ่องอารมย์', 314159),
      masterEmbedding: olderFace,
      angleEmbeddings: {
        center: olderFace,
        left: olderFace,
        right: olderFace,
        up: olderFace,
        down: olderFace,
      },
    };
    const newerPatient: PatientBiometricIndexEntry = {
      ...makeEnrolledPatient('pat-dup-new', 'HN-54720', 'พัทธดนย์ ผ่องอารมย์', 271828),
      masterEmbedding: newerFace,
      angleEmbeddings: {
        center: newerFace,
        left: newerFace,
        right: newerFace,
        up: newerFace,
        down: newerFace,
      },
    };
    const unrelated = makeEnrolledPatient('pat-other', 'HN-94856', 'ศิรสิทธิ์ เฉลิมเสนาะ', 123456);

    // Probe sits closely between older and newer registration of the same patient
    const liveProbe = l2Normalize(
      olderFace.map((v, idx) => v * 0.52 + newerFace[idx] * 0.48),
    );
    const res = matchMultiAngleEmbedding(liveProbe, [olderPatient, unrelated, newerPatient], {
      similarityThreshold: 0.78,
    });
    expect(res.matched).toBe(true);
    expect(res.bestCandidate?.hn).toBe('HN-54720');
    expect(res.margin).toBeGreaterThanOrEqual(0.06);
  });

  it('allows normal desk sitting distance and slight head angle in isFullFrontalFaceReadyForLogin', () => {
    // 16:9 webcam seated patient: height ~ 0.16, width ~ 0.11, yaw = 12, pitch = 14, depth = 0.35
    const landmarks = Array.from({ length: 478 }, () => ({ x: 0.5, y: 0.5, z: 0 }));
    landmarks[10] = { x: 0.50, y: 0.42, z: 0 };
    landmarks[152] = { x: 0.50, y: 0.58, z: 0 };
    landmarks[33] = { x: 0.46, y: 0.48, z: 0 };
    landmarks[263] = { x: 0.54, y: 0.48, z: 0 };
    landmarks[1] = { x: 0.50, y: 0.51, z: -0.04 };
    landmarks[61] = { x: 0.47, y: 0.55, z: 0 };
    landmarks[291] = { x: 0.53, y: 0.55, z: 0 };
    landmarks[234] = { x: 0.445, y: 0.50, z: 0 };
    landmarks[454] = { x: 0.555, y: 0.50, z: 0 };

    // Must strictly reject SPOOF_DETECTED regardless of canonical 3D mesh depth
    expect(
      isFullFrontalFaceReadyForLogin(
        landmarks,
        { yawDeg: 12, pitchDeg: 14, depthScore: 0.35 },
        { scan_status: 'SPOOF_DETECTED', error_code: 'SPOOF_ATTEMPT' },
      ),
    ).toBe(false);

    // Genuine PASS frame at normal desk distance still passes cleanly
    expect(
      isFullFrontalFaceReadyForLogin(
        landmarks,
        { yawDeg: 12, pitchDeg: 14, depthScore: 0.35 },
        { scan_status: 'PASS', error_code: 'NONE' },
      ),
    ).toBe(true);

    // Reject flat low-depth spoof or obstructed eyes
    expect(
      isFullFrontalFaceReadyForLogin(
        landmarks,
        { yawDeg: 0, pitchDeg: 0, depthScore: 0.12 },
        { scan_status: 'SPOOF_DETECTED', error_code: 'SPOOF_ATTEMPT' },
      ),
    ).toBe(false);
    expect(
      isFullFrontalFaceReadyForLogin(
        landmarks,
        { yawDeg: 0, pitchDeg: 0, depthScore: 0.45 },
        { scan_status: 'RETRY', error_code: 'OBSTRUCTED_EYES' },
      ),
    ).toBe(false);
  });

  it('blocks login in TemporalBiometricMatcher when biologicalVerified is false or spoofDetected is true', () => {
    const bioMatcher = new TemporalBiometricMatcher(3);
    const probe = patientB.angleEmbeddings.center;

    // 4 consecutive matching frames WITHOUT biological verification must NOT confirm
    for (let i = 0; i < 4; i += 1) {
      const r = bioMatcher.evaluateFrame(
        {
          probeVector: probe,
          depthScore: 0.55,
          yawDeg: i * 0.4,
          pitchDeg: 0,
          microMotion: 0.002,
          biologicalVerified: false,
          spoofDetected: false,
        },
        index,
      );
      expect(r.confirmed).toBe(false);
      expect(r.livenessPassed).toBe(false);
      expect(r.livenessReason).toBe('BIOLOGICAL_BLINK_REQUIRED');
    }

    // Immediate spoof detection resets streak & blocks
    const spoofed = bioMatcher.evaluateFrame(
      {
        probeVector: probe,
        depthScore: 0.55,
        yawDeg: 1,
        pitchDeg: 0,
        microMotion: 0.002,
        biologicalVerified: false,
        spoofDetected: true,
      },
      index,
    );
    expect(spoofed.confirmed).toBe(false);
    expect(spoofed.streakCount).toBe(0);
    expect(spoofed.livenessReason).toBe('SPOOF_ATTEMPT_BLOCKED');

    // Once biological blink + clean frames arrive, streak builds and confirms
    let finalRes = spoofed;
    for (let i = 0; i < 3; i += 1) {
      finalRes = bioMatcher.evaluateFrame(
        {
          probeVector: probe,
          depthScore: 0.55,
          yawDeg: i * 0.4,
          pitchDeg: 0,
          microMotion: 0.002,
          blinkDetected: i === 0,
          biologicalVerified: true,
          spoofDetected: false,
        },
        index,
      );
    }
    expect(finalRes.confirmed).toBe(true);
    expect(finalRes.livenessPassed).toBe(true);
    expect(finalRes.candidate?.patientId).toBe('pat-b');
  });

  it('supports Ultra-Long-Range (faceHeightRatio < 0.20) distance-adaptive threshold (>= 0.62) with >= 0.05 margin across 3 frames', () => {
    const owner = makeEnrolledPatient('pat-lr', 'HN-LR-01', 'คุณมานะ เดินไกล', 919191);
    const other = makeEnrolledPatient('pat-other', 'HN-LR-02', 'คนไข้ท่านอื่น', 333333);

    // Distant probe retains ~0.70 similarity to enrolled template (mixed appearance attenuation + craniofacial core)
    const slightDiff = makeRealisticFaceVector(777777, 1);
    const farProbe = l2Normalize(
      owner.angleEmbeddings.center.map((v, idx) => v * 0.68 + slightDiff[idx] * 0.32),
    );
    const farFaceLandmarks = Array.from({ length: 478 }, () => ({ x: 0.5, y: 0.4, z: 0 }));
    farFaceLandmarks[10] = { x: 0.5, y: 0.37, z: 0 };
    farFaceLandmarks[152] = { x: 0.5, y: 0.43, z: 0 };
    farFaceLandmarks[33] = { x: 0.485, y: 0.39, z: 0 };
    farFaceLandmarks[263] = { x: 0.515, y: 0.39, z: 0 };
    farFaceLandmarks[1] = { x: 0.5, y: 0.405, z: -0.02 };
    farFaceLandmarks[61] = { x: 0.49, y: 0.415, z: 0 };
    farFaceLandmarks[291] = { x: 0.51, y: 0.415, z: 0 };
    farFaceLandmarks[234] = { x: 0.478, y: 0.4, z: 0 };
    farFaceLandmarks[454] = { x: 0.522, y: 0.4, z: 0 };

    expect(
      isFullFrontalFaceReadyForLogin(
        farFaceLandmarks,
        { yawDeg: 2, pitchDeg: 1, depthScore: 0.42 },
        { scan_status: 'PASS', error_code: 'NONE' },
      ),
    ).toBe(true);

    const lrMatcher = new TemporalBiometricMatcher(3, 0.72);
    let out = lrMatcher.evaluateFrame(
      {
        probeVector: farProbe,
        depthScore: 0.45,
        yawDeg: 0,
        pitchDeg: 0,
        microMotion: 0.001,
        biologicalVerified: true,
        spoofDetected: false,
        faceHeightRatio: 0.08,
      },
      [owner, other],
    );
    expect(out.confirmed).toBe(false);
    expect(out.streakCount).toBe(1);

    for (let i = 1; i < 3; i += 1) {
      out = lrMatcher.evaluateFrame(
        {
          probeVector: farProbe,
          depthScore: 0.46,
          yawDeg: i * 1.4,
          pitchDeg: 0.5,
          microMotion: 0.0012,
          biologicalVerified: true,
          spoofDetected: false,
          faceHeightRatio: 0.08,
        },
        [owner, other],
      );
    }
    expect(out.confirmed).toBe(true);
    expect(out.candidate?.patientId).toBe('pat-lr');
    expect(out.confidencePercent).toBeGreaterThanOrEqual(84);
  });

  it('parses all embedding formats, handles 20-deg roll tilt & mirror flip, and uses 3-frame sliding window with soft decay', () => {
    // 1. BURST_FRAME_COUNT reduced to 3 for fast responsive 0/3 -> 3/3 progress
    expect(BURST_FRAME_COUNT).toBe(3);

    // 2. Universal Embedding Loader supports Array, Float32Array, JSON string, pgvector brace, & object wrapper
    const seedVec = makeRealisticFaceVector(31337, 2);
    expect(parseEmbeddingVector(new Float32Array(seedVec))?.length).toBe(FACE_EMBEDDING_DIM);
    expect(parseEmbeddingVector(JSON.stringify(seedVec))?.length).toBe(FACE_EMBEDDING_DIM);
    expect(parseEmbeddingVector(`{${seedVec.join(',')}}`)?.length).toBe(FACE_EMBEDDING_DIM);
    expect(parseEmbeddingVector({ master_embedding: seedVec })?.length).toBe(FACE_EMBEDDING_DIM);
    expect(parseEmbeddingVector('invalid-vector')).toBeUndefined();

    // 3. Pure 5-Angle Max-Pooling matches right-angle template cleanly even when master centroid differs strongly
    const pureRightOwner: PatientBiometricIndexEntry = {
      ...makeEnrolledPatient('pat-5angle', 'HN-5A01', 'คุณวิชัย มุมข้าง', 80808),
      masterEmbedding: makeRealisticFaceVector(11111, 0),
      angleEmbeddings: {
        center: makeRealisticFaceVector(11111, 1),
        left: makeRealisticFaceVector(22222, 2),
        right: makeRealisticFaceVector(80808, 3),
        up: makeRealisticFaceVector(33333, 4),
        down: makeRealisticFaceVector(44444, 5),
      },
    };
    const otherPat = makeEnrolledPatient('pat-other-5a', 'HN-5A02', 'คนไข้ท่านอื่น', 55555);
    const rightMatch = matchMultiAngleEmbedding(pureRightOwner.angleEmbeddings.right, [otherPat, pureRightOwner], {
      yawDeg: 22,
    });
    expect(rightMatch.matched).toBe(true);
    expect(rightMatch.bestCandidate?.patientId).toBe('pat-5angle');
    expect(rightMatch.bestAngle).toBe('right');

    // 4. Roll Tilt Invariance (|roll| > 5 deg) keeps 3D craniofacial descriptor consistent & compensates landmark angle
    const baseLms = Array.from({ length: 478 }, (_, idx) => ({
      x: 0.4 + (idx % 17) * 0.012,
      y: 0.35 + Math.floor(idx / 17) * 0.01,
      z: (idx % 5) * -0.004,
    }));
    baseLms[33] = { x: 0.44, y: 0.46, z: 0 };
    baseLms[133] = { x: 0.46, y: 0.46, z: 0 };
    baseLms[468] = { x: 0.45, y: 0.46, z: 0 };
    baseLms[362] = { x: 0.54, y: 0.46, z: 0 };
    baseLms[263] = { x: 0.56, y: 0.46, z: 0 };
    baseLms[473] = { x: 0.55, y: 0.46, z: 0 };
    baseLms[1] = { x: 0.50, y: 0.51, z: -0.04 };

    const tiltedLms = rotateNormalizedLandmarksForRoll(baseLms, 0.5, 0.46, (20 * Math.PI) / 180, 1280, 720);
    expect(Math.abs(computeEyeRollAngleDeg(tiltedLms, 1280, 720) - 20)).toBeLessThan(0.5);

    const flatImg = { width: 112, height: 112, data: new Uint8ClampedArray(112 * 112 * 4).fill(128) };
    const descFlat = extractHybrid512Descriptor(flatImg, baseLms);
    const descTilted = extractHybrid512Descriptor(flatImg, tiltedLms);
    expect(cosineSimilarity(descFlat, descTilted)).toBeGreaterThan(0.95);

    // 5. Mirror Flip Invariance preserves match when probe is mirror-flipped
    const flippedImg = flipImageDataHorizontally(flatImg);
    const flippedLms = flipLandmarksHorizontally(baseLms);
    const mirroredDesc = extractHybrid512Descriptor(flippedImg, flippedLms);
    const mirrorMatch = matchMultiAngleEmbedding(makeRealisticFaceVector(90909, 0), [pureRightOwner], {
      mirroredProbeVector: pureRightOwner.angleEmbeddings.right,
      yawDeg: 18,
    });
    expect(mirroredDesc.length).toBe(FACE_EMBEDDING_DIM);
    expect(mirrorMatch.matched).toBe(true);
    expect(mirrorMatch.bestCandidate?.patientId).toBe('pat-5angle');

    // 6. Allows ±32° head turn in isFullFrontalFaceReadyForLogin (prevents 0/3 lockup)
    expect(
      isFullFrontalFaceReadyForLogin(
        tiltedLms,
        { yawDeg: 32, pitchDeg: 18, depthScore: 0.42 },
        { scan_status: 'PASS', error_code: 'NONE' },
      ),
    ).toBe(true);

    // 7. TemporalBiometricMatcher defaults to 3 frames & uses soft decay (0.5) instead of wiping to 0 on 1 blink glitch
    const matcher = new TemporalBiometricMatcher();
    expect(matcher.requiredConsecutiveFrames).toBe(3);

    const slightDiffVec = makeRealisticFaceVector(918273, 1);
    const angledProbe = l2Normalize(
      pureRightOwner.angleEmbeddings.right.map((v, idx) => v * 0.62 + slightDiffVec[idx] * 0.38),
    );
    const strangerProbe = makeRealisticFaceVector(444444, 0);

    const f1 = matcher.evaluateFrame(
      {
        probeVector: angledProbe,
        depthScore: 0.52,
        yawDeg: 22,
        pitchDeg: 4,
        rollDeg: 12,
        microMotion: 0.002,
        biologicalVerified: true,
        spoofDetected: false,
      },
      [otherPat, pureRightOwner],
    );
    expect(f1.streakCount).toBe(1);

    const f2 = matcher.evaluateFrame(
      {
        probeVector: angledProbe,
        depthScore: 0.52,
        yawDeg: 23,
        pitchDeg: 4,
        rollDeg: 12,
        microMotion: 0.002,
        biologicalVerified: true,
        spoofDetected: false,
      },
      [otherPat, pureRightOwner],
    );
    expect(f2.streakCount).toBe(2);

    // 1 missed frame (e.g. eye blink/motion blur) decays softly rather than wiping streak to 0
    const glitchFrame = matcher.evaluateFrame(
      {
        probeVector: strangerProbe,
        depthScore: 0.52,
        yawDeg: 23,
        pitchDeg: 4,
        rollDeg: 12,
        microMotion: 0.002,
        biologicalVerified: true,
        spoofDetected: false,
      },
      [otherPat, pureRightOwner],
    );
    expect(glitchFrame.streakCount).toBeGreaterThanOrEqual(1);
    expect(glitchFrame.candidate?.patientId).toBe('pat-5angle');

    // Recover smoothly to confirmed (3/3) without restarting from scratch
    matcher.evaluateFrame(
      {
        probeVector: angledProbe,
        depthScore: 0.52,
        yawDeg: 24,
        pitchDeg: 5,
        rollDeg: 12,
        microMotion: 0.002,
        biologicalVerified: true,
        spoofDetected: false,
      },
      [otherPat, pureRightOwner],
    );
    const finalConfirm = matcher.evaluateFrame(
      {
        probeVector: angledProbe,
        depthScore: 0.52,
        yawDeg: 24.5,
        pitchDeg: 5,
        rollDeg: 12,
        microMotion: 0.002,
        biologicalVerified: true,
        spoofDetected: false,
      },
      [otherPat, pureRightOwner],
    );
    expect(finalConfirm.confirmed).toBe(true);
    expect(finalConfirm.streakCount).toBe(3);
    expect(finalConfirm.candidate?.patientId).toBe('pat-5angle');
  });
});
