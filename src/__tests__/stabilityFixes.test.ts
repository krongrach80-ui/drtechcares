import fs from 'node:fs';
import path from 'node:path';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import {
  assertZeroGhostLocalCache,
  evaluateTripleGateNeuralCandidate,
  l2Normalize,
  purgeLegacyBiometricCache,
} from '@drtechcares/smart-biometric-core';
import {
  clampTtsText,
  resolveTodayPatientSchedule,
  createCloudDelegatedRouter,
} from '@drtechcares/api-client';
import {
  ClinicalBiomechanicsAgent,
  ClinicalOneEuroLandmarkFilter,
  POSE_INDEX,
  computeJointAngle3D,
  evaluateArmKinematics,
  evaluateActiveWristsLowered,
  solveStrokeClinicalJointBiomechanics,
  type PoseLandmark,
} from '@drtechcares/pose-engine';
import { RomGauge } from '../components/RomGauge';
import { classifyExerciseJointMode } from '../components/SmartRehabKioskUI';
import {
  evaluateLiveMatch1toN,
  normalizeImageDataLuminance,
  type EnrolledBiometricCandidate,
} from '../lib/biometricEngine';
import {
  evaluatePoseCanvasSkeletonSafety,
  mapVideoPointToCoverCanvas,
} from '../lib/poseSkeletonSafety';

const EMOJI_RE = /[\p{Extended_Pictographic}\u{FE0F}\u{2600}-\u{27BF}\u{23E9}-\u{23FA}]/u;

function makeBasePose(overrides: Partial<Record<number, Partial<PoseLandmark>>> = {}): PoseLandmark[] {
  const base: PoseLandmark[] = Array.from({ length: 33 }, () => ({
    x: 0.5,
    y: 0.5,
    z: 0,
    visibility: 0.99,
  }));
  base[POSE_INDEX.NOSE] = { x: 0.5, y: 0.2, z: -0.05, visibility: 0.99 };
  base[POSE_INDEX.LEFT_SHOULDER] = { x: 0.58, y: 0.35, z: 0, visibility: 0.99 };
  base[POSE_INDEX.RIGHT_SHOULDER] = { x: 0.42, y: 0.35, z: 0, visibility: 0.99 };
  base[POSE_INDEX.LEFT_ELBOW] = { x: 0.60, y: 0.52, z: 0, visibility: 0.99 };
  base[POSE_INDEX.RIGHT_ELBOW] = { x: 0.40, y: 0.52, z: 0, visibility: 0.99 };
  base[POSE_INDEX.LEFT_WRIST] = { x: 0.62, y: 0.68, z: 0, visibility: 0.99 };
  base[POSE_INDEX.RIGHT_WRIST] = { x: 0.38, y: 0.68, z: 0, visibility: 0.99 };
  base[POSE_INDEX.LEFT_HIP] = { x: 0.56, y: 0.65, z: 0, visibility: 0.99 };
  base[POSE_INDEX.RIGHT_HIP] = { x: 0.44, y: 0.65, z: 0, visibility: 0.99 };

  for (const [k, v] of Object.entries(overrides)) {
    base[Number(k)] = { ...base[Number(k)], ...v };
  }
  return base;
}

describe('Project Stability, Face Scan & Pose Detection Hardening', () => {
  it('preserves active v3 biometric vault keys during legacy cache purge while wiping legacy keys', () => {
    const store = new Map<string, string>([
      ['drtech_biometric_vault_v3', '[{"patientId":"p1"}]'],
      ['drtech_face_templates', '[]'],
      ['legacy_biometric_cache_v1', 'stale'],
      ['old_face_embeddings', 'stale'],
    ]);
    const origStorage = globalThis.localStorage;
    Object.defineProperty(globalThis, 'localStorage', {
      configurable: true,
      value: {
        get length() {
          return store.size;
        },
        key: (i: number) => Array.from(store.keys())[i] ?? null,
        getItem: (k: string) => store.get(k) ?? null,
        setItem: (k: string, v: string) => {
          store.set(k, v);
        },
        removeItem: (k: string) => {
          store.delete(k);
        },
        clear: () => store.clear(),
      },
    });
    try {
      const purged = purgeLegacyBiometricCache();
      expect(purged).toContain('legacy_biometric_cache_v1');
      expect(purged).toContain('old_face_embeddings');
      expect(store.has('drtech_biometric_vault_v3')).toBe(true);
      expect(store.has('drtech_face_templates')).toBe(true);
      expect(assertZeroGhostLocalCache().clean).toBe(true);
    } finally {
      Object.defineProperty(globalThis, 'localStorage', {
        configurable: true,
        value: origStorage,
      });
    }
  });

  it('does not falsely lock multi-exercise patients as completedToday after completing only 1 of 2 exercises', () => {
    const now = new Date();
    const todayIso = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}`;
    const res = resolveTodayPatientSchedule({
      now,
      rehabSchedules: [
        {
          id: 's1',
          patientId: 'p1',
          ptId: 'pt1',
          exerciseId: 'STROKE_SH_FLEX_R',
          exerciseCode: 'STROKE_SH_FLEX',
          side: 'RIGHT',
          scheduledDate: todayIso,
          sets: 1,
          reps: 5,
          status: 'ACTIVE',
        },
        {
          id: 's2',
          patientId: 'p1',
          ptId: 'pt1',
          exerciseId: 'STROKE_SH_ABD_R',
          exerciseCode: 'STROKE_SH_ABD',
          side: 'RIGHT',
          scheduledDate: todayIso,
          sets: 1,
          reps: 5,
          status: 'ACTIVE',
        },
      ],
      rehabSessions: [
        {
          id: 'sess1',
          patientId: 'p1',
          activityType: 'physio_exercise',
          exerciseId: 'STROKE_SH_FLEX_R',
          exerciseName: 'Flexion',
          completedSets: 1,
          completedReps: 5,
          accuracyPercent: 95,
          maxRomDeg: 115,
          painStopTriggered: false,
          aiFeedback: 'OK',
          kinematicsReplay: [],
          createdAt: now.toISOString(),
        },
      ],
    });
    expect(res.hasAppointmentToday).toBe(true);
    expect(res.completedToday).toBe(false);
    expect(res.todayExercises[0]?.status).toBe('COMPLETED');
    expect(res.todayExercises[1]?.status).toBe('ACTIVE');
  });

  it('guards NaN coordinates in computeJointAngle3D and RomGauge, and supports seated cropped-hip upper-body mode', () => {
    expect(
      computeJointAngle3D(
        { x: Number.NaN, y: 0.5, z: 0 },
        { x: 0.4, y: 0.4, z: 0 },
        { x: 0.2, y: 0.3, z: 0 },
      ),
    ).toBe(0);

    const seatedPose = makeBasePose({
      [POSE_INDEX.RIGHT_ELBOW]: { x: 0.36, y: 0.25, z: -0.05, visibility: 0.99 },
      [POSE_INDEX.RIGHT_WRIST]: { x: 0.30, y: 0.14, z: -0.10, visibility: 0.99 },
      [POSE_INDEX.RIGHT_HIP]: { x: 0.44, y: 1.05, z: 0, visibility: 0.1 },
    });
    const armRes = evaluateArmKinematics(seatedPose, 'right', { jointGroup: 'shoulder' });
    expect(armRes.gate.valid).toBe(true);
    expect(armRes.angleDeg).not.toBeNull();
    expect(armRes.angleDeg!).toBeGreaterThan(100);

    const gaugeHtml = renderToStaticMarkup(
      React.createElement(RomGauge, {
        currentRomDeg: Number.NaN,
        targetRomDeg: Number.NaN,
        completionPct: Number.POSITIVE_INFINITY,
        kioskState: 'ACTIVE_REP',
        highlightColor: 'GREEN',
        holdRemainingSec: Number.NaN,
        holdTargetSec: 3,
      }),
    );
    expect(gaugeHtml).not.toContain('NaN');
    expect(gaugeHtml).toContain('0°');
  });

  it('prevents One-Euro filter poisoning on occluded landmarks and preserves forward depth foreshortening', () => {
    const flt = new ClinicalOneEuroLandmarkFilter();
    const valid = makeBasePose();
    flt.filterLandmarks(valid, 100);

    // Occluded right wrist with extreme garbage coordinate must NOT pollute filter velocity state
    const occluded = makeBasePose({
      [POSE_INDEX.RIGHT_WRIST]: { x: 99, y: -99, z: 0, visibility: 0 },
    });
    const outOccluded = flt.filterLandmarks(occluded, 130);
    expect(outOccluded[POSE_INDEX.RIGHT_WRIST].visibility).toBe(0);

    const recovered = makeBasePose({
      [POSE_INDEX.RIGHT_WRIST]: { x: 0.38, y: 0.67, z: 0, visibility: 0.99 },
    });
    const outRecovered = flt.filterLandmarks(recovered, 160);
    expect(outRecovered[POSE_INDEX.RIGHT_WRIST].x).toBeCloseTo(0.38, 2);
    expect(outRecovered[POSE_INDEX.RIGHT_WRIST].y).toBeCloseTo(0.67, 2);

    // Depth-foreshortened forward arm raise pointing toward camera lens (short 2D projection, |dz| >= 0.14)
    const forwardFlex = makeBasePose({
      [POSE_INDEX.RIGHT_ELBOW]: { x: 0.42, y: 0.36, z: -0.18, visibility: 0.95 },
      [POSE_INDEX.RIGHT_WRIST]: { x: 0.42, y: 0.35, z: -0.34, visibility: 0.92 },
    });
    const safety = evaluatePoseCanvasSkeletonSafety({
      landmarks: forwardFlex,
      videoW: 640,
      videoH: 480,
      canvasW: 640,
      canvasH: 480,
    });
    expect(safety.rightUpperArmValid).toBe(true);
    expect(safety.rightForearmValid).toBe(true);
  });

  it('prevents false EMERGENCY_HALT on target+26 deg, prevents distal hand BOTH false errors, and gates invisible wrists on lower check', () => {
    const agent = new ClinicalBiomechanicsAgent();
    const res = agent.evaluate({
      auth_status: { verified: true, confidence: 0.95, liveness: true },
      spatial_framing: { distance_m: 2.0, full_body_visible: true, feet_visible: true },
      exercise_prescription: {
        exercise_id: 'STROKE_SH_ROT_R',
        target_joint: 'right_shoulder',
        target_rom_deg: 65,
        hold_sec: 3,
        target_reps: 5,
      },
      live_telemetry: {
        primary_angle_deg: 95,
        trunk_tilt_deg: 2,
        angular_velocity_deg_per_sec: 15,
        dt_sec: 0.25,
      },
    });
    expect(res.kiosk_state).not.toBe('EMERGENCY_HALT');

    const neutral = makeBasePose();
    const distalBoth = solveStrokeClinicalJointBiomechanics({
      landmarks: neutral,
      targetJoint: 'right_shoulder',
      exerciseSide: 'BOTH',
      expectedPlane: 'wrist_finger',
      jointMode: 'wrist_finger',
      targetRomDeg: 70,
    });
    expect(distalBoth.sideViolation).not.toBe('SINGLE_ARM_IN_BOTH_MODE');
    expect(distalBoth.wrongSideRaised).toBe(false);

    const lowVisWristLowY = makeBasePose({
      [POSE_INDEX.RIGHT_WRIST]: { x: 0.38, y: 0.85, z: 0, visibility: 0.1 },
    });
    expect(
      evaluateActiveWristsLowered({
        landmarks: lowVisWristLowY,
        exerciseSide: 'RIGHT',
        targetJoint: 'right_shoulder',
      }),
    ).toBe(false);
  });

  it('returns standard empty-body 204 CORS OPTIONS preflight and clamps Thai text safely', async () => {
    const router = createCloudDelegatedRouter();
    const res = await router(
      new Request('https://example.com/api/health', { method: 'OPTIONS' }),
    );
    expect(res.status).toBe(204);
    expect(res.body).toBeNull();

    const longThai = 'สวัสดีครับวันนี้เรามาฝึกกายภาพบำบัดร่วมกันนะครับขอให้ยกแขนขึ้นช้าๆอย่างปลอดภัยและมั่นคงตลอดการฝึกครับผม';
    const clamped = clampTtsText(longThai, 60);
    expect(clamped.length).toBeLessThanOrEqual(60);
    expect(clamped.length).toBeGreaterThan(20);

    const testContent = fs.readFileSync(
      path.resolve(process.cwd(), 'src/__tests__/stabilityFixes.test.ts'),
      'utf8',
    );
    expect(EMOJI_RE.test(testContent)).toBe(false);
  });

  it('classifies STROKE_BOBATH_FORWARD_PUSH as bobath_push, prevents hanging rest false reps, and keeps BOTH boost symmetric', () => {
    expect(
      classifyExerciseJointMode({
        exercise_id: 'STROKE_BOBATH_FORWARD_PUSH',
        name_en: 'Stroke ADL: Bobath Bilateral Forward Elbow Push',
        name_th: '[Stroke ADL] ประสานมือสองข้างดันเหยียดแขนไปด้านหน้า',
      }),
    ).toBe('bobath_push');

    // Relaxed hanging arm with slight horizontal slant must NOT trigger shoulder_rotation or thumb_pinch hold
    const hanging = makeBasePose({
      [POSE_INDEX.RIGHT_ELBOW]: { x: 0.40, y: 0.52, z: 0, visibility: 0.99 },
      [POSE_INDEX.RIGHT_WRIST]: { x: 0.36, y: 0.68, z: 0.01, visibility: 0.99 },
      [POSE_INDEX.RIGHT_THUMB]: { x: 0.36, y: 0.70, z: 0.01, visibility: 0.95 },
      [POSE_INDEX.RIGHT_INDEX]: { x: 0.37, y: 0.71, z: 0.01, visibility: 0.95 },
    });
    const rotRest = solveStrokeClinicalJointBiomechanics({
      landmarks: hanging,
      targetJoint: 'right_shoulder',
      exerciseSide: 'RIGHT',
      expectedPlane: 'rotation',
      jointMode: 'shoulder_rotation',
      targetRomDeg: 65,
    });
    expect(rotRest.romDeg ?? 0).toBeLessThan(20);

    const pinchRest = solveStrokeClinicalJointBiomechanics({
      landmarks: hanging,
      targetJoint: 'right_shoulder',
      exerciseSide: 'RIGHT',
      expectedPlane: 'thumb_pinch',
      jointMode: 'thumb_pinch',
      targetRomDeg: 60,
    });
    expect(pinchRest.romDeg ?? 0).toBeLessThan(20);

    // Single-side elbow mode must report curled elbow angle on rightArmRom (NOT resting vertical shoulder ~10 deg)
    const curledRight = makeBasePose({
      [POSE_INDEX.RIGHT_ELBOW]: { x: 0.42, y: 0.52, z: -0.02, visibility: 0.99 },
      [POSE_INDEX.RIGHT_WRIST]: { x: 0.42, y: 0.38, z: -0.06, visibility: 0.99 },
    });
    const elbowRes = solveStrokeClinicalJointBiomechanics({
      landmarks: curledRight,
      targetJoint: 'right_shoulder',
      exerciseSide: 'RIGHT',
      expectedPlane: 'any',
      jointMode: 'elbow',
      targetRomDeg: 110,
    });
    expect(elbowRes.rightArmRom).not.toBeNull();
    expect(elbowRes.rightArmRom!).toBeGreaterThan(90);

    // Symmetric BOTH-mode elevation boost for equal left & right raises
    const bothRaised = makeBasePose({
      [POSE_INDEX.LEFT_ELBOW]: { x: 0.60, y: 0.35, z: -0.14, visibility: 0.99 },
      [POSE_INDEX.LEFT_WRIST]: { x: 0.60, y: 0.32, z: -0.26, visibility: 0.99 },
      [POSE_INDEX.RIGHT_ELBOW]: { x: 0.40, y: 0.35, z: -0.14, visibility: 0.99 },
      [POSE_INDEX.RIGHT_WRIST]: { x: 0.40, y: 0.32, z: -0.26, visibility: 0.99 },
    });
    const bothRes = solveStrokeClinicalJointBiomechanics({
      landmarks: bothRaised,
      targetJoint: 'right_shoulder',
      exerciseSide: 'BOTH',
      expectedPlane: 'flexion',
      jointMode: 'default',
      targetRomDeg: 110,
    });
    expect(bothRes.leftArmRom).not.toBeNull();
    expect(bothRes.rightArmRom).not.toBeNull();
    expect(Math.abs((bothRes.leftArmRom ?? 0) - (bothRes.rightArmRom ?? 0))).toBeLessThan(2);

    // mapVideoPointToCoverCanvas must NOT teleport normalized landmarks slightly outside [0..1]
    const edgePt = mapVideoPointToCoverCanvas(1.12, 0.5, 640, 360, 640, 360, false);
    expect(edgePt.x).toBeGreaterThan(640);

    // One-Euro filter handles duplicate timestamps cleanly without velocity spike
    const flt = new ClinicalOneEuroLandmarkFilter();
    const p0 = makeBasePose();
    flt.filterLandmarks(p0, 1000);
    const pDup = flt.filterLandmarks(
      makeBasePose({ [POSE_INDEX.RIGHT_WRIST]: { x: 0.9, y: 0.1, z: 0, visibility: 0.99 } }),
      1000,
    );
    expect(pDup[POSE_INDEX.RIGHT_WRIST].x).toBeCloseTo(0.38, 2);
  });

  it('supports realistic non-negative 1024-D ReLU vectors without patch, ranks Gate-passing candidates first, and preserves chroma on luminance normalization', () => {
    // Realistic non-negative 1024-D ReLU6 unit vector (std ~0.0155, < 0.02 but well above 0.008)
    const rawRelu = Array.from({ length: 1024 }, (_, i) => 0.45 + 0.38 * Math.sin(i * 0.37) + 0.15 * Math.cos(i * 1.13));
    const vecRelu = l2Normalize(rawRelu.map((v) => Math.max(0, v)));
    const tg = evaluateTripleGateNeuralCandidate(vecRelu, {
      masterEmbedding: vecRelu,
      angleEmbeddings: {
        center: vecRelu,
        left: vecRelu,
        right: vecRelu,
        up: vecRelu,
        down: vecRelu,
      },
    });
    expect(tg.passed).toBe(true);

    // Gate-passing candidate must beat a slightly higher raw-cosine candidate that fails Gate 2
    const goodPatch = Array.from({ length: 384 }, (_, i) => Math.sin(i * 0.25));
    const badPatch = Array.from({ length: 384 }, (_, i) => -Math.sin(i * 0.25));
    const otherVec = l2Normalize(Array.from({ length: 1024 }, (_, i) => Math.max(0, 0.5 + 0.4 * Math.cos(i * 0.71))));
    const makeCand = (id: string, name: string, vec: number[], patch: number[]): EnrolledBiometricCandidate => ({
      patientId: id,
      hn: `HN-${id}`,
      fullName: name,
      diagnosis: 'Stroke',
      assignedPtId: 'pt-1',
      nextAppointmentDate: '',
      framesCaptured: 120,
      qualityScore: 0.96,
      dimension: 1024,
      master_embedding: vec,
      angle_embeddings: { center: vec, left: vec, right: vec, up: vec, down: vec },
      exemplar_embeddings: [vec],
      geometricRatios: {
        eyeToNoseRatio: 1,
        faceWidthToUpperHeight: 1,
        mouthToEyeRatio: 1,
        noseToChinRatio: 1,
        appearancePatch384: patch,
      },
      sourceEntry: {
        patientId: id,
        hn: `HN-${id}`,
        fullName: name,
        diagnosis: 'Stroke',
        assignedPtId: 'pt-1',
        nextAppointmentDate: '',
        framesCaptured: 120,
        qualityScore: 0.96,
        masterEmbedding: vec,
        angleEmbeddings: { center: vec, left: vec, right: vec, up: vec, down: vec },
      },
    });
    const probeSlightlyNoisy = l2Normalize(vecRelu.map((v, i) => v + (i % 7 === 0 ? 0.002 : 0)));
    const eval1n = evaluateLiveMatch1toN(
      probeSlightlyNoisy,
      [
        makeCand('fail-gate2', 'Fail Gate2', probeSlightlyNoisy, badPatch),
        makeCand('pass-all', 'Pass All', vecRelu, goodPatch),
        makeCand('other', 'Other Person', otherVec, badPatch),
      ],
      [],
      undefined,
      'center',
      {
        eyeToNoseRatio: 1,
        faceWidthToUpperHeight: 1,
        mouthToEyeRatio: 1,
        noseToChinRatio: 1,
        appearancePatch384: goodPatch,
      },
    );
    // Gate-passing candidate ('pass-all') is ranked #1 ahead of 'fail-gate2' (higher rawCos but badPatch),
    // yielding topPatchZncc > 0.95 & AMBIGUOUS_CLOSE_CANDIDATES (rather than BELOW_THRESHOLD from fail-gate2).
    expect(eval1n.topPatchZncc).toBeGreaterThan(0.95);
    expect(eval1n.reason).toBe('AMBIGUOUS_CLOSE_CANDIDATES');
    const eval1nClean = evaluateLiveMatch1toN(
      probeSlightlyNoisy,
      [
        makeCand('pass-all', 'Pass All', vecRelu, goodPatch),
        makeCand('other', 'Other Person', otherVec, badPatch),
      ],
      [],
      undefined,
      'center',
      {
        eyeToNoseRatio: 1,
        faceWidthToUpperHeight: 1,
        mouthToEyeRatio: 1,
        noseToChinRatio: 1,
        appearancePatch384: goodPatch,
      },
    );
    expect(eval1nClean.matched).toBe(true);
    expect(eval1nClean.topCandidate?.patientId).toBe('pass-all');

    // Luminance normalization centers luma around 128 without exploding skin chroma saturation
    const pixels = new Uint8ClampedArray(16 * 16 * 4);
    for (let i = 0; i < 16 * 16; i += 1) {
      const base = i % 2 === 0 ? 60 : 75;
      pixels[i * 4] = base + 25;
      pixels[i * 4 + 1] = base + 8;
      pixels[i * 4 + 2] = base - 5;
      pixels[i * 4 + 3] = 255;
    }
    const normImg = normalizeImageDataLuminance({ width: 16, height: 16, data: pixels } as ImageData, 128);
    let rMinusBSum = 0;
    let ySum = 0;
    for (let i = 0; i < 16 * 16; i += 1) {
      const r = normImg.data[i * 4];
      const g = normImg.data[i * 4 + 1];
      const b = normImg.data[i * 4 + 2];
      rMinusBSum += r - b;
      ySum += 0.299 * r + 0.587 * g + 0.114 * b;
    }
    expect(ySum / (16 * 16)).toBeGreaterThan(120);
    expect(ySum / (16 * 16)).toBeLessThan(136);
    // Original R - B was 30; chroma-preserving normalization must stay well below 2.3x explosion (69)
    expect(rMinusBSum / (16 * 16)).toBeLessThan(40);
  });
});
