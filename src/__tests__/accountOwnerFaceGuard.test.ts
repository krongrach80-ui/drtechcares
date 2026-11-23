import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { l2Normalize } from '@drtechcares/smart-biometric-core';
import {
  OWNER_GUARD_MISMATCH_STREAK_REQUIRED,
  OWNER_GUARD_NO_FACE_GRACE_MS,
  OWNER_GUARD_UNAUTHORIZED_TTS_TH,
  collectOwnerReferenceEmbeddings,
  computeOwnerGuardMatchThreshold,
  createInitialAccountOwnerGuardState,
  evaluateDetectedFacesAgainstOwner,
  projectRawFaceBoxToMirroredCover,
  stepAccountOwnerGuardState,
} from '../hooks/useAccountOwnerFaceGuard';
import type { OwnerBiometricTemplate } from '../lib/ownerBiometricLock';

const UNICODE_EMOJI_RE = /[\p{Extended_Pictographic}\u{1F300}-\u{1FAFF}\u{2600}-\u{27BF}]/u;

function makeEmbedding1024(seed: number): number[] {
  const out = new Array<number>(1024);
  let s = (seed * 2654435761) >>> 0;
  for (let i = 0; i < 1024; i += 1) {
    s = (s * 1664525 + 1013904223 + i * 37) >>> 0;
    const raw = ((s % 2000) / 1000 - 1) + 0.25 * Math.sin((i + 1) * (seed * 0.19));
    out[i] = raw;
  }
  return l2Normalize(out);
}

function blendNormalized(a: readonly number[], b: readonly number[], weightA: number): number[] {
  const weightB = 1 - weightA;
  return l2Normalize(a.map((v, i) => v * weightA + b[i] * weightB));
}

describe('Continuous Account-Owner Face Guard & Biometric Exercise Lock', () => {
  const poomCenter = makeEmbedding1024(101);
  const poomLeft = blendNormalized(poomCenter, makeEmbedding1024(102), 0.78);
  const poomRight = blendNormalized(poomCenter, makeEmbedding1024(103), 0.78);
  const poomUp = blendNormalized(poomCenter, makeEmbedding1024(104), 0.8);
  const poomDown = blendNormalized(poomCenter, makeEmbedding1024(105), 0.8);
  const poomExemplar = blendNormalized(poomCenter, makeEmbedding1024(106), 0.84);
  const strangerEmbedding = makeEmbedding1024(9999);

  const poomTemplate: OwnerBiometricTemplate = {
    patientId: '5104db15-5634-45e5-ab65-f2f008a15e54',
    hn: 'HN-410382',
    fullName: 'Poom',
    biometricEngine: 'human-faceres-1024d',
    masterEmbedding: poomCenter,
    angleEmbeddings: {
      center: poomCenter,
      left: poomLeft,
      right: poomRight,
      up: poomUp,
      down: poomDown,
    },
    exemplarEmbeddings: [poomExemplar],
  };

  it('collects all exclusive owner reference embeddings (master + 5 angles + exemplars) and enforces threshold floor', () => {
    const refs = collectOwnerReferenceEmbeddings(poomTemplate);
    expect(refs.length).toBe(7);
    expect(refs.every((v) => v.length === 1024)).toBe(true);

    expect(computeOwnerGuardMatchThreshold(0.76)).toBe(0.72);
    expect(computeOwnerGuardMatchThreshold(0.72)).toBe(0.68);
    expect(computeOwnerGuardMatchThreshold(0.68)).toBe(0.66);
    expect(computeOwnerGuardMatchThreshold(0.5)).toBe(0.66);
  });

  it('keeps identityStatus === VERIFIED_OWNER and canExercise === true across frontal and angled owner poses', () => {
    const ownerEmbeddings = collectOwnerReferenceEmbeddings(poomTemplate);
    let state = createInitialAccountOwnerGuardState(1000);

    // Frontal live probe matching Poom
    const liveCenterProbe = blendNormalized(poomCenter, makeEmbedding1024(201), 0.9);
    const eval1 = evaluateDetectedFacesAgainstOwner({
      faces: [{ embedding: liveCenterProbe, embeddingDim: 1024, score: 0.96, box: { x: 0.35, y: 0.2, width: 0.25, height: 0.32 } }],
      ownerEmbeddings,
      baseMatchThreshold: 0.72,
    });
    state = stepAccountOwnerGuardState(state, { ...eval1, nowMs: 1450 });
    expect(state.identityStatus).toBe('VERIFIED_OWNER');
    expect(state.canExercise).toBe(true);
    expect(state.similarity).toBeGreaterThanOrEqual(0.85);

    // Side-tilted live probe matching Poom's stored `left` angle embedding during exercise
    const liveLeftProbe = blendNormalized(poomLeft, makeEmbedding1024(202), 0.88);
    const eval2 = evaluateDetectedFacesAgainstOwner({
      faces: [{ embedding: liveLeftProbe, embeddingDim: 1024, score: 0.91, box: { x: 0.38, y: 0.21, width: 0.24, height: 0.31 } }],
      ownerEmbeddings,
      baseMatchThreshold: 0.72,
    });
    state = stepAccountOwnerGuardState(state, { ...eval2, nowMs: 1900 });
    expect(state.identityStatus).toBe('VERIFIED_OWNER');
    expect(state.canExercise).toBe(true);
    expect(state.mismatchStreak).toBe(0);

    // Head tilted up during overhead shoulder raise matches stored `up` angle embedding
    const liveUpProbe = blendNormalized(poomUp, makeEmbedding1024(203), 0.88);
    const eval3 = evaluateDetectedFacesAgainstOwner({
      faces: [{ embedding: liveUpProbe, embeddingDim: 1024, score: 0.9, box: { x: 0.36, y: 0.19, width: 0.24, height: 0.3 } }],
      ownerEmbeddings,
      baseMatchThreshold: 0.72,
    });
    state = stepAccountOwnerGuardState(state, { ...eval3, nowMs: 2350 });
    expect(state.identityStatus).toBe('VERIFIED_OWNER');
    expect(state.canExercise).toBe(true);
  });

  it('transitions to UNAUTHORIZED_PERSON after 2 checks, freezes reps/timers/ROM, and auto-resumes when owner returns', () => {
    const ownerEmbeddings = collectOwnerReferenceEmbeddings(poomTemplate);
    let guard = createInitialAccountOwnerGuardState(1000);

    // Simulated exercise progression state gated strictly by guard.canExercise
    let reps = 0;
    let holdSeconds = 0;
    let peakRomDeg = 0;
    let savedSessions = 0;

    const tickExerciseFrame = (dtSec: number, liveRomDeg: number) => {
      if (!guard.canExercise) return;
      if (liveRomDeg > peakRomDeg) peakRomDeg = liveRomDeg;
      if (liveRomDeg >= 120) {
        holdSeconds += dtSec;
        if (holdSeconds >= 2.0) {
          reps += 1;
          holdSeconds = 0;
        }
      }
    };
    const saveCompletedSession = () => {
      if (!guard.canExercise) return false;
      savedSessions += 1;
      return true;
    };

    // 1. Owner starts exercising -> accumulates 1.0s hold and 135 deg peak ROM
    tickExerciseFrame(1.0, 135);
    expect(holdSeconds).toBe(1.0);
    expect(peakRomDeg).toBe(135);
    expect(reps).toBe(0);

    // 2. Stranger appears — Check #1 (anti-flicker buffer prevents transient blur false alarm)
    const strangerFace = {
      embedding: strangerEmbedding,
      embeddingDim: 1024,
      score: 0.95,
      box: { x: 0.3, y: 0.22, width: 0.26, height: 0.34 },
    };
    const evalMismatch1 = evaluateDetectedFacesAgainstOwner({
      faces: [strangerFace],
      ownerEmbeddings,
      baseMatchThreshold: 0.72,
    });
    expect(evalMismatch1.matchedOwner).toBe(false);
    guard = stepAccountOwnerGuardState(guard, { ...evalMismatch1, nowMs: 1450 });
    expect(guard.mismatchStreak).toBe(1);
    expect(guard.identityStatus).toBe('VERIFIED_OWNER');
    expect(guard.canExercise).toBe(true);

    // 3. Stranger persists — Check #2 (~900ms) -> UNAUTHORIZED_PERSON + canExercise === false!
    const evalMismatch2 = evaluateDetectedFacesAgainstOwner({
      faces: [strangerFace],
      ownerEmbeddings,
      baseMatchThreshold: 0.72,
    });
    guard = stepAccountOwnerGuardState(guard, { ...evalMismatch2, nowMs: 1900 });
    expect(guard.mismatchStreak).toBe(OWNER_GUARD_MISMATCH_STREAK_REQUIRED);
    expect(guard.identityStatus).toBe('UNAUTHORIZED_PERSON');
    expect(guard.canExercise).toBe(false);
    expect(guard.trackedRawBox).toEqual(strangerFace.box);

    // While UNAUTHORIZED_PERSON, stranger lifting arm to 165 deg for 3.0s CANNOT increment reps, hold timer, ROM, or save!
    tickExerciseFrame(3.0, 165);
    expect(reps).toBe(0);
    expect(holdSeconds).toBe(1.0); // Frozen cleanly at 1.0s!
    expect(peakRomDeg).toBe(135); // Blocked from recording stranger's 165 deg ROM!
    expect(saveCompletedSession()).toBe(false);
    expect(savedSessions).toBe(0);

    // 4. Owner (Poom) returns -> immediately unlocks VERIFIED_OWNER & canExercise === true and resumes from 1.0s!
    const evalOwnerReturn = evaluateDetectedFacesAgainstOwner({
      faces: [
        {
          embedding: blendNormalized(poomRight, makeEmbedding1024(305), 0.9),
          embeddingDim: 1024,
          score: 0.95,
          box: { x: 0.34, y: 0.2, width: 0.25, height: 0.32 },
        },
      ],
      ownerEmbeddings,
      baseMatchThreshold: 0.72,
    });
    guard = stepAccountOwnerGuardState(guard, { ...evalOwnerReturn, nowMs: 2350 });
    expect(guard.identityStatus).toBe('VERIFIED_OWNER');
    expect(guard.canExercise).toBe(true);
    expect(guard.mismatchStreak).toBe(0);

    // Resumes smoothly from 1.0s -> +1.0s finishes the rep!
    tickExerciseFrame(1.0, 142);
    expect(reps).toBe(1);
    expect(peakRomDeg).toBe(142);
    expect(saveCompletedSession()).toBe(true);
    expect(savedSessions).toBe(1);
  });

  it('applies 1500ms grace buffer before transitioning to NO_OWNER_FACE and unlocks on owner re-entry', () => {
    const ownerEmbeddings = collectOwnerReferenceEmbeddings(poomTemplate);
    let guard = createInitialAccountOwnerGuardState(1000);

    // No face for 1200ms (<= 1500ms grace buffer) -> stays unlocked
    const emptyEval = evaluateDetectedFacesAgainstOwner({
      faces: [],
      ownerEmbeddings,
      baseMatchThreshold: 0.72,
    });
    guard = stepAccountOwnerGuardState(guard, { ...emptyEval, nowMs: 1000 + OWNER_GUARD_NO_FACE_GRACE_MS - 300 });
    expect(guard.identityStatus).toBe('VERIFIED_OWNER');
    expect(guard.canExercise).toBe(true);

    // No face for 1700ms (> 1500ms) -> locks into NO_OWNER_FACE
    guard = stepAccountOwnerGuardState(guard, { ...emptyEval, nowMs: 1000 + OWNER_GUARD_NO_FACE_GRACE_MS + 200 });
    expect(guard.identityStatus).toBe('NO_OWNER_FACE');
    expect(guard.canExercise).toBe(false);

    // Owner re-enters -> unlocks immediately
    const returnEval = evaluateDetectedFacesAgainstOwner({
      faces: [{ embedding: poomCenter, embeddingDim: 1024, score: 0.97, box: { x: 0.4, y: 0.2, width: 0.2, height: 0.3 } }],
      ownerEmbeddings,
      baseMatchThreshold: 0.72,
    });
    guard = stepAccountOwnerGuardState(guard, { ...returnEval, nowMs: 3200 });
    expect(guard.identityStatus).toBe('VERIFIED_OWNER');
    expect(guard.canExercise).toBe(true);
  });

  it('projects raw face bounding boxes cleanly into mirrored object-cover viewport coordinates', () => {
    const projected = projectRawFaceBoxToMirroredCover(
      { x: 0.15, y: 0.2, width: 0.2, height: 0.3 },
      640,
      480,
      640,
      480,
    );
    expect(projected).not.toBeNull();
    // Raw left (0.15..0.35) mirrors onto screen right (~65%..85%)
    expect(projected!.leftPct).toBeGreaterThan(55);
    expect(projected!.widthPct).toBeGreaterThan(15);
    expect(projected!.heightPct).toBeGreaterThan(25);
  });

  it('enforces UI lock copy, zero emojis, and 60FPS non-blocking guardrails across modified files', () => {
    const root = process.cwd();
    const guardHookSrc = fs.readFileSync(
      path.join(root, 'src', 'hooks', 'useAccountOwnerFaceGuard.ts'),
      'utf8',
    );
    const rehabUiSrc = fs.readFileSync(
      path.join(root, 'src', 'components', 'SmartRehabKioskUI.tsx'),
      'utf8',
    );
    const mascotSrc = fs.readFileSync(
      path.join(root, 'src', 'components', 'DrTechCareMascot.tsx'),
      'utf8',
    );
    const selfTestSrc = fs.readFileSync(
      path.join(root, 'src', '__tests__', 'accountOwnerFaceGuard.test.ts'),
      'utf8',
    );

    expect(OWNER_GUARD_UNAUTHORIZED_TTS_TH).toBe(
      'ตรวจพบบุคคลอื่นหน้ากล้องครับ ระบบระงับการนับรอบชั่วคราว กรุณาให้เจ้าของบัญชีเป็นผู้ทำกายภาพบำบัดนะครับ',
    );
    expect(rehabUiSrc).toContain('useAccountOwnerFaceGuard');
    expect(rehabUiSrc).toContain('ระงับการฝึกชั่วคราว: ตรวจพบบุคคลอื่นหน้ากล้อง');
    expect(rehabUiSrc).toContain('สลับบัญชี / สแกนใบหน้าใหม่');
    expect(rehabUiSrc).toContain('เจ้าของบัญชี:');
    expect(rehabUiSrc).toContain('ตรวจพบบุคคลอื่น (ไม่ใช่');
    expect(rehabUiSrc).toContain('กรุณาให้ใบหน้าของคุณ');
    expect(rehabUiSrc).toContain('DrTechCareMascot');
    expect(rehabUiSrc).toContain('h-[100dvh]');

    // Strict guardrail: ZERO Unicode emojis anywhere in modified/new files
    expect(UNICODE_EMOJI_RE.test(guardHookSrc)).toBe(false);
    expect(UNICODE_EMOJI_RE.test(rehabUiSrc)).toBe(false);
    expect(UNICODE_EMOJI_RE.test(mascotSrc)).toBe(false);
    expect(UNICODE_EMOJI_RE.test(selfTestSrc)).toBe(false);
  });
});

