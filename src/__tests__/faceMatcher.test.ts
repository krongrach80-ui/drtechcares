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
  extractLegacyV1Hybrid512Descriptor,
  flipImageDataHorizontally,
  flipLandmarksHorizontally,
  rotateNormalizedLandmarksForRoll,
} from '@drtechcares/face-client';
import {
  TemporalBiometricMatcher,
  computeBoundingBoxFromLandmarks,
  cosineSimilarity,
  identifyPatient1toN,
  matchMultiAngleEmbedding,
} from '../lib/faceMatcher';
import {
  computeGeometricAnthropometricRatios,
  isLegacyV1HybridVector,
  trimOutlierVectorsAndAverage,
} from '@drtechcares/smart-biometric-core';
import {
  ENROLLMENT_SEQUENCE,
  buildWrongDirectionCue,
  chooseCaptureBucket,
  classifyDisplayBucket,
  computeNormalizedMirroredHeadPoseDeg,
  computePeakAngleScoreForStep,
  getActiveEnrollmentStage,
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

    const res = matchMultiAngleEmbedding(noisyLeftProbe, index, { yawDeg: -16 });
    expect(res.matched).toBe(true);
    expect(res.bestCandidate?.patientId).toBe('pat-a');
    expect(res.bestAngle).toBe('left');
    expect(res.similarity).toBeGreaterThanOrEqual(0.72);
    expect(res.margin).toBeGreaterThanOrEqual(0.08);
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

      // 0..127 DCT: u=0 or v=0 carries shirt/background gradient; interior u>=1,v>=1 carries face structure
      let dctIdx = 0;
      for (let u = 0; u < 12 && dctIdx < 128; u += 1) {
        for (let v = 0; v < 12 && dctIdx < 128; v += 1) {
          if (u === 0 && v === 0) continue;
          const innerWave = nextRand() * 0.14 * (0.85 + 0.15 * lightingGain);
          vec[dctIdx] =
            u === 0 || v === 0
              ? shirtBias * (0.025 - u * 0.001) * (v % 2 === 0 ? 1 : -0.7)
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
            const innerWave = nextRand() * 0.14 * (0.85 + 0.15 * lightingGain);
            vec[offset + slot] = isInner && slot <= 2
              ? CANONICAL_HUMAN_FACE_BASELINE_512[offset + slot] * 0.45 + innerWave
              : shirtBias * ((slot % 3) - 1) * 0.02;
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

  it('enforces strict 5-stage sequential pose capture (center -> left -> right -> up -> down) and normalized -45..+45 mirror-aware angles', () => {
    expect(ENROLLMENT_SEQUENCE.map((s) => [s.step, s.key, s.target])).toEqual([
      [1, 'center', 40],
      [2, 'left', 20],
      [3, 'right', 20],
      [4, 'up', 20],
      [5, 'down', 20],
    ]);

    expect(classifyDisplayBucket(3, 0)).toBe('center');
    expect(classifyDisplayBucket(-10, 0)).toBe('center');
    expect(classifyDisplayBucket(-12, 0)).toBe('left');
    expect(classifyDisplayBucket(12, 0)).toBe('right');
    expect(classifyDisplayBucket(0, -9)).toBe('up');
    expect(classifyDisplayBucket(0, 9)).toBe('down');

    // Step 1 (center 0..39): strictly requires |deltaYaw| <= 8° and |deltaPitch| <= 9° (rejects turns/tilts!)
    const inStep1 = { center: 10, left: 0, right: 0, up: 0, down: 0 };
    expect(getActiveEnrollmentStage(inStep1)?.key).toBe('center');
    expect(chooseCaptureBucket(0, 0, inStep1)).toBe('center');
    expect(chooseCaptureBucket(-8, 9, inStep1)).toBe('center');
    expect(chooseCaptureBucket(-12, 0, inStep1)).toBeNull();
    expect(chooseCaptureBucket(0, 12, inStep1)).toBeNull();
    expect(buildWrongDirectionCue('center', 12, 0, 'th')).toContain('มองตรงหน้ากล้อง');

    // Step 2 (left 0..19): ZERO fake/auto-pass — staring straight (0°, 0°) keeps left at 0/20 indefinitely until deltaYaw <= -11°!
    const inStep2 = { center: 40, left: 0, right: 0, up: 0, down: 0 };
    expect(getActiveEnrollmentStage(inStep2)?.key).toBe('left');
    for (let tick = 0; tick < 60; tick += 1) {
      expect(chooseCaptureBucket(0, 0, inStep2)).toBeNull();
      expect(chooseCaptureBucket(-8, 0, inStep2)).toBeNull();
    }
    expect(chooseCaptureBucket(-11, 0, inStep2)).toBe('left');
    expect(chooseCaptureBucket(-15, 3, inStep2)).toBe('left');
    expect(chooseCaptureBucket(15, 0, inStep2)).toBeNull();
    expect(chooseCaptureBucket(0, -10, inStep2)).toBeNull();
    expect(buildWrongDirectionCue('left', 14, 0, 'th')).toContain('ซ้าย');
    expect(buildWrongDirectionCue('left', -5, 0, 'th')).toContain('11°');

    // Step 3 (right 0..19): unlocks right ONLY when deltaYaw >= +11°
    const inStep3 = { center: 40, left: 20, right: 0, up: 0, down: 0 };
    expect(getActiveEnrollmentStage(inStep3)?.key).toBe('right');
    expect(chooseCaptureBucket(0, 0, inStep3)).toBeNull();
    expect(chooseCaptureBucket(9, 0, inStep3)).toBeNull();
    expect(chooseCaptureBucket(11, 0, inStep3)).toBe('right');
    expect(chooseCaptureBucket(-15, 0, inStep3)).toBeNull();
    expect(chooseCaptureBucket(0, -10, inStep3)).toBeNull();
    expect(buildWrongDirectionCue('right', 4, 0, 'th')).toContain('11°');

    // Step 4 (up 0..19): unlocks up ONLY when deltaPitch <= -8° and |deltaYaw| <= 16°
    const inStep4 = { center: 40, left: 20, right: 20, up: 0, down: 0 };
    expect(getActiveEnrollmentStage(inStep4)?.key).toBe('up');
    expect(chooseCaptureBucket(0, 0, inStep4)).toBeNull();
    expect(chooseCaptureBucket(0, -6, inStep4)).toBeNull();
    expect(chooseCaptureBucket(0, -8, inStep4)).toBe('up');
    expect(chooseCaptureBucket(20, -12, inStep4)).toBeNull();
    expect(chooseCaptureBucket(0, 8, inStep4)).toBeNull();
    expect(buildWrongDirectionCue('up', 0, -4, 'th')).toContain('8°');

    // Step 5 (down 0..19): unlocks down ONLY when deltaPitch >= +8° and |deltaYaw| <= 16°
    const inStep5 = { center: 40, left: 20, right: 20, up: 20, down: 0 };
    expect(getActiveEnrollmentStage(inStep5)?.key).toBe('down');
    expect(chooseCaptureBucket(0, 0, inStep5)).toBeNull();
    expect(chooseCaptureBucket(0, 6, inStep5)).toBeNull();
    expect(chooseCaptureBucket(0, 8, inStep5)).toBe('down');
    expect(chooseCaptureBucket(-20, 12, inStep5)).toBeNull();
    expect(chooseCaptureBucket(0, -8, inStep5)).toBeNull();
    expect(buildWrongDirectionCue('down', 0, 5, 'th')).toContain('8°');

    // Peak pose scores ensure 5 distinct strongest-angle thumbnails are retained
    expect(computePeakAngleScoreForStep('center', 0, 0)).toBeGreaterThan(
      computePeakAngleScoreForStep('center', 5, 4),
    );
    expect(computePeakAngleScoreForStep('left', -18, 0)).toBeGreaterThan(
      computePeakAngleScoreForStep('left', -11, 0),
    );
    expect(computePeakAngleScoreForStep('right', 18, 0)).toBeGreaterThan(
      computePeakAngleScoreForStep('right', 11, 0),
    );
    expect(computePeakAngleScoreForStep('up', 0, -14)).toBeGreaterThan(
      computePeakAngleScoreForStep('up', 0, -8),
    );
    expect(computePeakAngleScoreForStep('down', 0, 14)).toBeGreaterThan(
      computePeakAngleScoreForStep('down', 0, 8),
    );

    // Anatomical Pitch & Mirror-Aware Yaw (with bounding box + mouth landmarks & eye-only fallback)
    const makeFaceMesh468 = (noseX: number, noseY: number, withMouth = true) => {
      const pts = Array.from({ length: 468 }, () => ({ x: 0.5, y: 0.5, z: 0 }));
      pts[33] = { x: 0.36, y: 0.4, z: 0 };
      pts[133] = { x: 0.44, y: 0.4, z: 0 };
      pts[362] = { x: 0.56, y: 0.4, z: 0 };
      pts[263] = { x: 0.64, y: 0.4, z: 0 }; // eyeMid = (0.5, 0.4), eyeDist(1:1) = 0.20
      pts[1] = { x: noseX, y: noseY, z: -0.05 };
      if (withMouth) {
        // Neutral face: eyeMidY = 0.40, noseY = 0.54, mouthMidY = 0.65 -> (0.54 - 0.40) / 0.25 = 0.56 -> Pitch = 0°
        pts[61] = { x: 0.44, y: 0.65, z: 0 };
        pts[291] = { x: 0.56, y: 0.65, z: 0 };
        pts[13] = { x: 0.5, y: 0.645, z: 0 };
        pts[14] = { x: 0.5, y: 0.655, z: 0 };
      }
      return pts;
    };

    const centerBox = { x: 0.3, y: 0.25, width: 0.4, height: 0.5 };
    const centerPose16x9 = computeNormalizedMirroredHeadPoseDeg(
      makeFaceMesh468(0.5, 0.54, true),
      1280 / 720,
      centerBox,
    );
    expect(Math.abs(centerPose16x9.yawDeg)).toBeLessThanOrEqual(1);
    expect(Math.abs(centerPose16x9.pitchDeg)).toBeLessThanOrEqual(8);
    expect(classifyDisplayBucket(centerPose16x9.yawDeg, centerPose16x9.pitchDeg)).toBe('center');

    // Eye-only fallback (without mouth/box) uses natural 0.68 frontal nose offset
    const centerPoseEyeOnly = computeNormalizedMirroredHeadPoseDeg(makeFaceMesh468(0.5, 0.536, false));
    expect(Math.abs(centerPoseEyeOnly.yawDeg)).toBeLessThanOrEqual(1);
    expect(Math.abs(centerPoseEyeOnly.pitchDeg)).toBeLessThanOrEqual(1);

    // Turning LEFT on mirrored screen -> nose shifts +x in raw sensor -> negative yawDeg <= -11°
    const mirrorLeftPose = computeNormalizedMirroredHeadPoseDeg(makeFaceMesh468(0.54, 0.54, true));
    expect(mirrorLeftPose.yawDeg).toBeLessThanOrEqual(-11);
    expect(mirrorLeftPose.yawDeg).toBeGreaterThanOrEqual(-55);

    // Turning RIGHT on mirrored screen -> nose shifts -x in raw sensor -> positive yawDeg >= +11°
    const mirrorRightPose = computeNormalizedMirroredHeadPoseDeg(makeFaceMesh468(0.46, 0.54, true));
    expect(mirrorRightPose.yawDeg).toBeGreaterThanOrEqual(11);
    expect(mirrorRightPose.yawDeg).toBeLessThanOrEqual(55);

    // Tilting chin UP / DOWN produces clear negative / positive pitch relative to Step 1 baseline
    const tiltUpPose = computeNormalizedMirroredHeadPoseDeg(
      makeFaceMesh468(0.5, 0.49, true),
      1280 / 720,
      centerBox,
    );
    const tiltDownPose = computeNormalizedMirroredHeadPoseDeg(
      makeFaceMesh468(0.5, 0.59, true),
      1280 / 720,
      centerBox,
    );
    expect(tiltUpPose.pitchDeg - centerPose16x9.pitchDeg).toBeLessThanOrEqual(-8);
    expect(tiltDownPose.pitchDeg - centerPose16x9.pitchDeg).toBeGreaterThanOrEqual(8);

    // Extreme outlier offset strictly clamped to [-55, 55]
    const extremeYaw = computeNormalizedMirroredHeadPoseDeg(makeFaceMesh468(0.95, 0.54, true));
    expect(extremeYaw.yawDeg).toBe(-55);
    const extremeRight = computeNormalizedMirroredHeadPoseDeg(makeFaceMesh468(0.05, 0.54, true));
    expect(extremeRight.yawDeg).toBe(55);

    // End-to-end 120-frame sequential progression simulation with real physical poses (center 40 -> left 20 -> right 20 -> up 20 -> down 20)
    const simCounts = { center: 0, left: 0, right: 0, up: 0, down: 0 };
    const simPoses: Array<{ key: keyof typeof simCounts; yaw: number; pitch: number }> = [
      { key: 'center', yaw: 0, pitch: 0 },
      { key: 'left', yaw: -14, pitch: 0 },
      { key: 'right', yaw: 14, pitch: 0 },
      { key: 'up', yaw: 0, pitch: -11 },
      { key: 'down', yaw: 0, pitch: 11 },
    ];
    for (const stage of simPoses) {
      const target = ENROLLMENT_SEQUENCE.find((s) => s.key === stage.key)!.target;
      for (let i = 0; i < target; i += 1) {
        const picked = chooseCaptureBucket(stage.yaw, stage.pitch, simCounts);
        expect(picked).toBe(stage.key);
        simCounts[picked!] += 1;
      }
    }
    expect(simCounts).toEqual({ center: 40, left: 20, right: 20, up: 20, down: 20 });
    expect(getActiveEnrollmentStage(simCounts)).toBeNull();
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
          exerciseId: 'STROKE_SH_FLEX_R',
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
          exerciseId: 'STROKE_SH_FLEX_R',
          scheduledDate: '2026-11-14',
          sets: 3,
          reps: 10,
          status: 'ACTIVE',
        },
        {
          id: 'sched-rest-ignored',
          patientId: 'pat-a',
          exerciseId: 'STROKE_SH_FLEX_L',
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
    expect(scheduledToday.todayExercises[0].exerciseId).toBe('STROKE_SH_FLEX_R');
    expect(scheduledToday.todayExercises[0].exerciseCode).toBe('STROKE_SH_FLEX');
    expect(scheduledToday.todayExercises[0].side).toBe('RIGHT');
    expect(scheduledToday.todayExercises[0].nameTh).toContain('ยกแขนไปด้านหน้า');
    expect(scheduledToday.todayExercises[0].nameTh).toContain('ข้างขวา');
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

  it('enforces calibrated open-set raw cosine (>= 0.64) even at Ultra-Long-Range across 3/3 frames with zero premature name leak', () => {
    const owner = makeEnrolledPatient('pat-lr', 'HN-LR-01', 'คุณมานะ เดินไกล', 919191);
    const other = makeEnrolledPatient('pat-other', 'HN-LR-02', 'คนไข้ท่านอื่น', 333333);

    // 1. Sub-0.60 stranger (~0.52 cosine) MUST be rejected cleanly at long range
    const rawDiff = makeRealisticFaceVector(777777, 1);
    const d = owner.angleEmbeddings.center.reduce((acc, v, idx) => acc + v * rawDiff[idx], 0);
    const lowDiff = l2Normalize(rawDiff.map((v, idx) => v - d * owner.angleEmbeddings.center[idx]));
    const rejectedProbe = l2Normalize(
      owner.angleEmbeddings.center.map((v, idx) => v * 0.52 + lowDiff[idx] * Math.sqrt(1 - 0.52 * 0.52)),
    );
    const strictMatcher = new TemporalBiometricMatcher();
    expect(strictMatcher.requiredConsecutiveFrames).toBe(3);
    const lowOut = strictMatcher.evaluateFrame(
      {
        probeVector: rejectedProbe,
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
    expect(lowOut.confirmed).toBe(false);
    expect(lowOut.streakCount).toBe(0);
    expect(lowOut.candidate).toBeNull();

    // 2. Genuine long-range probe confirms only at 3/3 frames with raw similarity >= 0.64
    const farProbe = l2Normalize(
      owner.angleEmbeddings.center.map((v, idx) => v * 0.92 + lowDiff[idx] * 0.39),
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

    let out = strictMatcher.evaluateFrame(
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
    expect(out.candidate).toBeNull();

    for (let i = 1; i < 3; i += 1) {
      out = strictMatcher.evaluateFrame(
        {
          probeVector: farProbe,
          depthScore: 0.46,
          yawDeg: i * 1.1,
          pitchDeg: 0.5,
          microMotion: 0.0012,
          biologicalVerified: true,
          spoofDetected: false,
          faceHeightRatio: 0.08,
        },
        [owner, other],
      );
      if (i < 2) {
        expect(out.confirmed).toBe(false);
        expect(out.candidate).toBeNull();
      }
    }
    expect(out.confirmed).toBe(true);
    expect(out.streakCount).toBe(3);
    expect(out.candidate?.patientId).toBe('pat-lr');
    expect(out.confidencePercent).toBeGreaterThanOrEqual(64);
  });

  it('parses all embedding formats, handles 20-deg roll tilt & mirror flip, and enforces strict 6/6 unanimous identity lock', () => {
    // 1. BURST_FRAME_COUNT stays 3 for post-countdown burst probe centroid
    expect(BURST_FRAME_COUNT).toBe(3);

    // 2. Universal Embedding Loader supports Array, Float32Array, JSON string, pgvector brace, & object wrapper
    const seedVec = makeRealisticFaceVector(31337, 2);
    expect(parseEmbeddingVector(new Float32Array(seedVec))?.length).toBe(FACE_EMBEDDING_DIM);
    expect(parseEmbeddingVector(JSON.stringify(seedVec))?.length).toBe(FACE_EMBEDDING_DIM);
    expect(parseEmbeddingVector(`{${seedVec.join(',')}}`)?.length).toBe(FACE_EMBEDDING_DIM);
    expect(parseEmbeddingVector({ master_embedding: seedVec })?.length).toBe(FACE_EMBEDDING_DIM);
    expect(parseEmbeddingVector('invalid-vector')).toBeUndefined();

    // 3. Pose-Synchronized Matching matches right-angle probe cleanly using same-owner master + right centroids
    const pureRightOwner = makeEnrolledPatient('pat-5angle', 'HN-5A01', 'คุณวิชัย มุมข้าง', 80808);
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

    // 5. Mirror Flip Invariance maps left/right zone cleanly for horizontally flipped probes
    const flippedImg = flipImageDataHorizontally(flatImg);
    const flippedLms = flipLandmarksHorizontally(baseLms);
    const mirroredDesc = extractHybrid512Descriptor(flippedImg, flippedLms);
    const mirrorMatch = matchMultiAngleEmbedding(makeRealisticFaceVector(90909, 0), [pureRightOwner], {
      mirroredProbeVector: pureRightOwner.angleEmbeddings.right,
      yawDeg: -18,
    });
    expect(mirroredDesc.length).toBe(FACE_EMBEDDING_DIM);
    expect(mirrorMatch.matched).toBe(true);
    expect(mirrorMatch.bestCandidate?.patientId).toBe('pat-5angle');

    // 6. Allows natural desk posture in isFullFrontalFaceReadyForLogin
    expect(
      isFullFrontalFaceReadyForLogin(
        tiltedLms,
        { yawDeg: 32, pitchDeg: 18, depthScore: 0.42 },
        { scan_status: 'PASS', error_code: 'NONE' },
      ),
    ).toBe(true);

    // 7. Strict 3/3 Unanimous Identity Lock: resets immediately to 0 on miss or identity switch
    const matcher = new TemporalBiometricMatcher();
    expect(matcher.requiredConsecutiveFrames).toBe(3);

    const angledProbe = l2Normalize(
      pureRightOwner.angleEmbeddings.right.map((v, idx) => v * 0.94 + pureRightOwner.masterEmbedding[idx] * 0.06),
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
    expect(f1.candidate).toBeNull();

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
    expect(f2.candidate).toBeNull();

    // 1 failed/stranger frame resets streak IMMEDIATELY to 0 (no soft decay bypass!)
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
    expect(glitchFrame.streakCount).toBe(0);
    expect(glitchFrame.confirmed).toBe(false);
    expect(glitchFrame.candidate).toBeNull();

    // Requires full 3/3 consecutive matching frames to confirm & reveal candidate
    for (let i = 0; i < 2; i += 1) {
      const step = matcher.evaluateFrame(
        {
          probeVector: angledProbe,
          depthScore: 0.52,
          yawDeg: 22 + i * 0.4,
          pitchDeg: 4,
          rollDeg: 12,
          microMotion: 0.002,
          biologicalVerified: true,
          spoofDetected: false,
        },
        [otherPat, pureRightOwner],
      );
      expect(step.confirmed).toBe(false);
      expect(step.streakCount).toBe(i + 1);
      expect(step.candidate).toBeNull();
    }
    const thirdConfirm = matcher.evaluateFrame(
      {
        probeVector: angledProbe,
        depthScore: 0.52,
        yawDeg: 24.5,
        pitchDeg: 4.5,
        rollDeg: 12,
        microMotion: 0.002,
        biologicalVerified: true,
        spoofDetected: false,
      },
      [otherPat, pureRightOwner],
    );
    expect(thirdConfirm.confirmed).toBe(true);
    expect(thirdConfirm.streakCount).toBe(3);
    expect(thirdConfirm.candidate?.patientId).toBe('pat-5angle');
  });

  it('enforces Pose-Synchronized matching, >= 0.08 Distinctiveness Margin, excludedPatientIds ("ไม่ใช่ฉัน"), Bottom-20% outlier trimming, and Two-Person Room Separation', () => {
    // 1. Pose-Synchronized Anti-Leakage: frontal probe (yaw=0, pitch=0) MUST NOT blindly match another patient's skewed "up" centroid
    const victimWithSkewedUp: PatientBiometricIndexEntry = {
      ...makeEnrolledPatient('pat-victim', 'HN-S1', 'นายวิชัย ต้นแบบ', 12345),
      angleEmbeddings: {
        ...makeEnrolledPatient('pat-victim', 'HN-S1', 'นายวิชัย ต้นแบบ', 12345).angleEmbeddings,
        up: makeRealisticFaceVector(676767, 0),
      },
    };
    const frontalStranger = makeRealisticFaceVector(676767, 0);
    const crossPoseRes = identifyPatient1toN(frontalStranger, [victimWithSkewedUp], {
      yawDeg: 0,
      pitchDeg: 0,
    });
    expect(crossPoseRes.matched).toBe(false);
    expect(crossPoseRes.livePoseZone).toBe('center');

    // 2. Strict Distinctiveness Margin (>= 0.08): reject when Rank #1 and Rank #2 are within < 0.08
    const twinA = makeEnrolledPatient('pat-twin-a', 'HN-TW1', 'คุณแฝด หนึ่ง', 50505);
    const closeSeed = l2Normalize(
      twinA.angleEmbeddings.center.map((v, i) => v * 0.96 + makeRealisticFaceVector(60606, 1)[i] * 0.04),
    );
    const twinB: PatientBiometricIndexEntry = {
      ...makeEnrolledPatient('pat-twin-b', 'HN-TW2', 'คุณแฝด สอง', 60606),
      masterEmbedding: closeSeed,
      angleEmbeddings: {
        center: closeSeed,
        left: closeSeed,
        right: closeSeed,
        up: closeSeed,
        down: closeSeed,
      },
    };
    const ambiguousRes = identifyPatient1toN(twinA.angleEmbeddings.center, [twinA, twinB], {
      yawDeg: 0,
      pitchDeg: 0,
    });
    expect(ambiguousRes.matched).toBe(false);
    expect(ambiguousRes.reason).toBe('AMBIGUOUS_CLOSE_CANDIDATES');
    expect(ambiguousRes.margin).toBeLessThan(0.08);

    // 3. "ไม่ใช่ฉัน (สแกนใหม่)" excludedPatientIds excludes misidentified patient completely
    const excludedRetry = identifyPatient1toN(twinA.angleEmbeddings.center, [twinA, twinB], {
      yawDeg: 0,
      pitchDeg: 0,
      excludedPatientIds: ['pat-twin-a'],
    });
    expect(excludedRetry.bestCandidate?.patientId).not.toBe('pat-twin-a');

    // 4. Bottom-20% outlier trimming on 120-frame centroid calculation
    const cleanVec = makeRealisticFaceVector(19191, 1);
    const badOutlier1 = makeRealisticFaceVector(99911, 0);
    const badOutlier2 = makeRealisticFaceVector(88822, 0);
    const trimmed = trimOutlierVectorsAndAverage(
      [
        ...Array.from({ length: 8 }, () => ({ vec: cleanVec, quality: 95 })),
        { vec: badOutlier1, quality: 20 },
        { vec: badOutlier2, quality: 15 },
      ],
      0.2,
    );
    expect(cosineSimilarity(trimmed, cleanVec)).toBeGreaterThan(0.995);

    // 5. Real-camera 112x112 synthetic inner-face image with IDENTICAL background border but different inner face & bone ratios
    const mkSyntheticRoomFaceImage = (personToneOffset: number) => {
      const data = new Uint8ClampedArray(112 * 112 * 4);
      for (let y = 0; y < 112; y += 1) {
        for (let x = 0; x < 112; x += 1) {
          const idx = (y * 112 + x) * 4;
          const isInner = x >= 26 && x <= 86 && y >= 30 && y <= 92;
          const val = isInner
            ? Math.round(
              110 +
              45 *
              Math.sin((x + personToneOffset) * 0.27) *
              Math.cos((y - personToneOffset) * 0.23),
            )
            : 215; // identical bright white room wall & ceiling for both people!
          data[idx] = val;
          data[idx + 1] = val;
          data[idx + 2] = val;
          data[idx + 3] = 255;
        }
      }
      return { width: 112, height: 112, data };
    };
    const mkLandmarksWithBoneGeometry = (noseY: number, cheekSpread: number) => {
      const lms = Array.from({ length: 478 }, () => ({ x: 0.5, y: 0.5, z: 0 }));
      lms[10] = { x: 0.5, y: 0.30, z: 0 };
      lms[168] = { x: 0.5, y: 0.38, z: -0.02 };
      lms[1] = { x: 0.5, y: noseY, z: -0.05 };
      lms[33] = { x: 0.42, y: 0.42, z: 0 };
      lms[263] = { x: 0.58, y: 0.42, z: 0 };
      lms[234] = { x: 0.5 - cheekSpread, y: 0.50, z: 0.04 };
      lms[454] = { x: 0.5 + cheekSpread, y: 0.50, z: 0.04 };
      lms[61] = { x: 0.45, y: 0.62, z: 0 };
      lms[291] = { x: 0.55, y: 0.62, z: 0 };
      lms[152] = { x: 0.5, y: 0.74, z: 0.01 };
      return lms;
    };

    const lmsPersonA = mkLandmarksWithBoneGeometry(0.49, 0.14);
    const lmsPersonB = mkLandmarksWithBoneGeometry(0.56, 0.21);
    const vecPersonA = extractHybrid512Descriptor(mkSyntheticRoomFaceImage(3), lmsPersonA);
    const vecPersonB = extractHybrid512Descriptor(mkSyntheticRoomFaceImage(19), lmsPersonB);
    const ratiosA = computeGeometricAnthropometricRatios(lmsPersonA);
    const ratiosB = computeGeometricAnthropometricRatios(lmsPersonB);

    const enrolledPersonA: PatientBiometricIndexEntry = {
      ...makeEnrolledPatient('pat-room-a', 'HN-R1', 'คุณกิตติ ห้องเดียวกัน', 1),
      masterEmbedding: vecPersonA,
      angleEmbeddings: {
        center: vecPersonA,
        left: vecPersonA,
        right: vecPersonA,
        up: vecPersonA,
        down: vecPersonA,
      },
      geometricRatios: ratiosA,
    };

    const crossRoomAttempt = identifyPatient1toN(vecPersonB, [enrolledPersonA], {
      yawDeg: 0,
      pitchDeg: 0,
      geometricRatios: ratiosB,
    });
    expect(crossRoomAttempt.matched).toBe(false);
  });

  it('matches V2 across room lighting changes & supports Legacy V1 DB backward compatibility strictly without cross-account impersonation', () => {
    const mkLitFaceImage = (personFreqShift: number, lightingMul: number, lightBias: number, bgVal: number) => {
      const data = new Uint8ClampedArray(112 * 112 * 4);
      for (let y = 0; y < 112; y += 1) {
        for (let x = 0; x < 112; x += 1) {
          const idx = (y * 112 + x) * 4;
          const isInner = x >= 22 && x <= 90 && y >= 22 && y <= 90;
          const texture = isInner
            ? 118 * lightingMul +
            lightBias +
            42 *
            lightingMul *
            (Math.sin((x + personFreqShift * 7) * 0.29) +
              0.65 * Math.cos((y - personFreqShift * 5) * 0.26) +
              0.35 * Math.sin((x * 0.14 + y * 0.19) * (1 + personFreqShift * 0.08)))
            : bgVal;
          const v = Math.max(0, Math.min(255, Math.round(texture)));
          data[idx] = v;
          data[idx + 1] = v;
          data[idx + 2] = v;
          data[idx + 3] = 255;
        }
      }
      return { width: 112, height: 112, data };
    };
    const mkBoneLandmarks = (noseY: number, cheekSpread: number, mouthW: number) => {
      const lms = Array.from({ length: 478 }, () => ({ x: 0.5, y: 0.5, z: 0 }));
      lms[10] = { x: 0.5, y: 0.29, z: 0 };
      lms[168] = { x: 0.5, y: 0.38, z: -0.02 };
      lms[1] = { x: 0.5, y: noseY, z: -0.05 };
      lms[33] = { x: 0.42, y: 0.42, z: 0 };
      lms[263] = { x: 0.58, y: 0.42, z: 0 };
      lms[234] = { x: 0.5 - cheekSpread, y: 0.51, z: 0.04 };
      lms[454] = { x: 0.5 + cheekSpread, y: 0.51, z: 0.04 };
      lms[61] = { x: 0.5 - mouthW, y: 0.63, z: 0 };
      lms[291] = { x: 0.5 + mouthW, y: 0.63, z: 0 };
      lms[152] = { x: 0.5, y: 0.74, z: 0.01 };
      return lms;
    };

    const lmsOwner = mkBoneLandmarks(0.50, 0.15, 0.055);
    const lmsStranger = mkBoneLandmarks(0.57, 0.22, 0.085);
    const ratiosOwner = computeGeometricAnthropometricRatios(lmsOwner);
    const ratiosStranger = computeGeometricAnthropometricRatios(lmsStranger);

    // 1. V2 illumination-invariant real image extraction: dim home room (0.7x, -15, dark bg 45) -> bright lab room (1.45x, +22, bright wall 230)
    const imgOwnerHome = mkLitFaceImage(1, 0.75, -15, 45);
    const imgOwnerLab = mkLitFaceImage(1, 1.35, 20, 230);
    const imgStrangerLab = mkLitFaceImage(4.7, 1.35, 20, 230);

    const v2Enroll = extractHybrid512Descriptor(imgOwnerHome, lmsOwner);
    const v2Probe = extractHybrid512Descriptor(imgOwnerLab, lmsOwner);
    const v2Stranger = extractHybrid512Descriptor(imgStrangerLab, lmsStranger);

    expect(isLegacyV1HybridVector(v2Enroll)).toBe(false);
    expect(isLegacyV1HybridVector(v2Probe)).toBe(false);

    const v2Candidate: PatientBiometricIndexEntry = {
      ...makeEnrolledPatient('pat-v2-room', 'HN-V2-01', 'นายทดสอบ แสงเปลี่ยน', 10),
      masterEmbedding: v2Enroll,
      angleEmbeddings: {
        center: v2Enroll,
        left: v2Enroll,
        right: v2Enroll,
        up: v2Enroll,
        down: v2Enroll,
      },
      geometricRatios: ratiosOwner,
    };

    const v2SelfRes = identifyPatient1toN(v2Probe, [v2Candidate], {
      yawDeg: 0,
      pitchDeg: 0,
      geometricRatios: ratiosOwner,
    });
    expect(v2SelfRes.matched).toBe(true);
    expect(v2SelfRes.bestCandidate?.patientId).toBe('pat-v2-room');

    const v2StrangerRes = identifyPatient1toN(v2Stranger, [v2Candidate], {
      yawDeg: 0,
      pitchDeg: 0,
      geometricRatios: ratiosStranger,
    });
    expect(v2StrangerRes.matched).toBe(false);

    // 2. Legacy V1 DB record backward-compatibility bridge
    const v1Enroll = extractLegacyV1Hybrid512Descriptor(imgOwnerHome, lmsOwner);
    const v1ProbeOwner = extractLegacyV1Hybrid512Descriptor(imgOwnerLab, lmsOwner);
    const v1ProbeStranger = extractLegacyV1Hybrid512Descriptor(imgStrangerLab, lmsStranger);

    expect(isLegacyV1HybridVector(v1Enroll)).toBe(true);

    const v1Candidate: PatientBiometricIndexEntry = {
      ...makeEnrolledPatient('pat-v1-legacy', 'HN-44524', 'Phattadon Phongarrom', 20),
      masterEmbedding: v1Enroll,
      angleEmbeddings: {
        center: v1Enroll,
        left: v1Enroll,
        right: v1Enroll,
        up: v1Enroll,
        down: v1Enroll,
      },
      geometricRatios: ratiosOwner,
    };

    const v1BridgeMatch = identifyPatient1toN(v2Probe, [v1Candidate], {
      yawDeg: 0,
      pitchDeg: 0,
      legacyV1ProbeVector: v1ProbeOwner,
      geometricRatios: ratiosOwner,
    });
    expect(v1BridgeMatch.matched).toBe(true);
    expect(v1BridgeMatch.bestCandidate?.hn).toBe('HN-44524');

    const v1BridgeStrangerReject = identifyPatient1toN(v2Stranger, [v1Candidate], {
      yawDeg: 0,
      pitchDeg: 0,
      legacyV1ProbeVector: v1ProbeStranger,
      geometricRatios: ratiosStranger,
    });
    expect(v1BridgeStrangerReject.matched).toBe(false);
  });

  it('isolates arcface-onnx vs hybrid-512d engines and avoids +0.20 cos3D inflation on Legacy V1 strangers', () => {
    const mkArcFaceUnit = (seed: number): number[] => {
      const vec = new Array<number>(FACE_EMBEDDING_DIM);
      let s = seed >>> 0;
      for (let i = 0; i < FACE_EMBEDDING_DIM; i += 1) {
        s = (s * 1664525 + 1013904223 + i * 53) >>> 0;
        vec[i] = ((s % 2000) / 1000 - 1) * (i < 256 ? 1.0 : 1.35);
      }
      return l2Normalize(vec);
    };

    const arcOwnerVec = mkArcFaceUnit(10101);
    const arcStrangerVec = mkArcFaceUnit(20202);
    const arcOwnerCandidate: PatientBiometricIndexEntry = {
      ...makeEnrolledPatient('pat-arc-1', 'HN-ARC-01', 'นายอาร์คเฟซ ตรงรุ่น', 1),
      biometricEngine: 'arcface-onnx',
      masterEmbedding: arcOwnerVec,
      angleEmbeddings: {
        center: arcOwnerVec,
        left: arcOwnerVec,
        right: arcOwnerVec,
        up: arcOwnerVec,
        down: arcOwnerVec,
      },
    };
    const hybridCandidate: PatientBiometricIndexEntry = {
      ...makeEnrolledPatient('pat-hyb-1', 'HN-HYB-01', 'นายไฮบริด คนละระบบ', 2),
      biometricEngine: 'hybrid-512d',
    };

    // 1. ArcFace probe with >= 0.54 neural cosine (even if ~0.62 — below hybrid's 0.75!) matches ArcFace account cleanly!
    const litArcProbe = l2Normalize(arcOwnerVec.map((v, i) => v * 0.65 + mkArcFaceUnit(777)[i] * 0.42));
    const arcMatch = identifyPatient1toN(litArcProbe, [hybridCandidate, arcOwnerCandidate], {
      engine: 'arcface-onnx',
      yawDeg: 0,
      pitchDeg: 0,
    });
    expect(arcMatch.matched).toBe(true);
    expect(arcMatch.bestCandidate?.patientId).toBe('pat-arc-1');

    // 2. Hybrid probe MUST NEVER match an arcface-onnx enrolled account even if same vector passed cleanly without arcface engine tag!
    const crossEngineReject = identifyPatient1toN(arcOwnerVec, [arcOwnerCandidate], {
      engine: 'hybrid-512d',
      yawDeg: 0,
      pitchDeg: 0,
    });
    expect(crossEngineReject.matched).toBe(false);

    // 3. ArcFace stranger must reject cleanly
    const arcStrangerReject = identifyPatient1toN(arcStrangerVec, [arcOwnerCandidate], {
      engine: 'arcface-onnx',
      yawDeg: 0,
      pitchDeg: 0,
    });
    expect(arcStrangerReject.matched).toBe(false);
  });

  it('allows full-range kiosk login (0.25m close-up border clip, 5m far tiny face, and off-center left/right)', () => {
    const mockPose = { yawDeg: 4, pitchDeg: 0, depthScore: 0.65 };
    const passInspect = { scan_status: 'PASS', error_code: 'NONE' } as const;

    // 1. Very close (~0.25m): forehead (10) & chin (152) clipped near edge, core 5pt inside frame
    const closeLm = Array.from({ length: 478 }, () => ({ x: 0.5, y: 0.5, z: 0 }));
    closeLm[10] = { x: 0.5, y: 0.001, z: 0 };
    closeLm[152] = { x: 0.5, y: 0.999, z: 0 };
    closeLm[234] = { x: 0.14, y: 0.5, z: 0 };
    closeLm[454] = { x: 0.86, y: 0.5, z: 0 };
    closeLm[33] = { x: 0.34, y: 0.38, z: 0 };
    closeLm[263] = { x: 0.66, y: 0.38, z: 0 };
    closeLm[1] = { x: 0.5, y: 0.54, z: -0.05 };
    closeLm[61] = { x: 0.39, y: 0.72, z: 0 };
    closeLm[291] = { x: 0.61, y: 0.72, z: 0 };
    expect(isFullFrontalFaceReadyForLogin(closeLm, mockPose, passInspect)).toBe(true);

    // 2. Far off-center left (~4.5m, ~2.8% height, x around 0.18)
    const farLeftLm = Array.from({ length: 478 }, () => ({ x: 0.18, y: 0.45, z: 0 }));
    farLeftLm[10] = { x: 0.18, y: 0.436, z: 0 };
    farLeftLm[152] = { x: 0.18, y: 0.464, z: 0 };
    farLeftLm[234] = { x: 0.17, y: 0.45, z: 0 };
    farLeftLm[454] = { x: 0.19, y: 0.45, z: 0 };
    farLeftLm[33] = { x: 0.174, y: 0.445, z: 0 };
    farLeftLm[263] = { x: 0.186, y: 0.445, z: 0 };
    farLeftLm[1] = { x: 0.18, y: 0.452, z: -0.01 };
    farLeftLm[61] = { x: 0.176, y: 0.458, z: 0 };
    farLeftLm[291] = { x: 0.184, y: 0.458, z: 0 };
    expect(isFullFrontalFaceReadyForLogin(farLeftLm, mockPose, passInspect)).toBe(true);
  });
});
