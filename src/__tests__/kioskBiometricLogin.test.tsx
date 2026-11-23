import fs from 'node:fs';
import path from 'node:path';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it, vi } from 'vitest';
import { POSE_INDEX } from '@drtechcares/pose-engine';
import { l2Normalize } from '@drtechcares/smart-biometric-core';
import { computeBestMultiAngleSimilarity } from '../lib/biometricEngine';
import {
  AsyncInferenceMutex,
  HAND_GLITCH_GRACE_MS,
  KIOSK_DETECT_MAX_LONG_SIDE_PX,
  KIOSK_FAST_MATCH_STREAK,
  KIOSK_INFERENCE_CANVAS_MAX_WIDTH,
  KIOSK_INFERENCE_INTERVAL_MS,
  KIOSK_STREAK_GRACE_MS,
  KioskVerifiedConfirmationCard,
  PASSIVE_LIVENESS_MIN_PIXEL_SHIFT,
  VERIFIED_HAND_HOLD_MS,
  VERIFIED_MODAL_TTS_STALL_TIMEOUT_MS,
  buildKioskLoginSuccessPayload,
  createInitialFastScanStreakState,
  createInitialPostMatchGestureGateState,
  evaluatePassiveWebcamLiveness,
  formatKioskSupabaseSyncBadgeLabel,
  getKioskVerifiedModalCopy,
  isClearFaceForFastScan,
  matchLocalEnrolledPatientsStrict,
  mergeEnrolledPatientIntoKioskList,
  resolveKioskFaceBoundingBoxOverlay,
  resolveKioskUnifiedScanProgress,
  resolvePostMatchActiveVideoElement,
  shouldTriggerFastNotFoundVerdict,
  stepFastScanStreakWithGrace,
  stepPostMatchGestureGateFrame,
  type KioskLoginSuccessPayload,
} from '../components/KioskBiometricLoginView';
import type { PatientRecord } from '@drtechcares/api-client';

const EMOJI_RE = /[\p{Extended_Pictographic}\u{FE0F}\u{2600}-\u{27BF}\u{23E9}-\u{23FA}]/u;

function makeUnitVec(seed: number, dim = 1024): number[] {
  const out = new Array<number>(dim);
  for (let i = 0; i < dim; i += 1) {
    out[i] = Math.sin((i + 1) * seed * 0.17) + 0.5 * Math.cos((i + 3) * seed * 0.29);
  }
  return l2Normalize(out);
}

function mixUnitVec(a: number[], b: number[], alpha: number): number[] {
  return l2Normalize(a.map((v, i) => v * (1 - alpha) + b[i] * alpha));
}

function makeFaceLandmarks(rawCenterX: number, rawCenterY: number, halfW = 0.08, halfH = 0.1) {
  return [
    { x: rawCenterX - halfW, y: rawCenterY - halfH },
    { x: rawCenterX + halfW, y: rawCenterY - halfH },
    { x: rawCenterX, y: rawCenterY },
    { x: rawCenterX - halfW * 0.9, y: rawCenterY + halfH },
    { x: rawCenterX + halfW * 0.9, y: rawCenterY + halfH },
    { x: rawCenterX - halfW, y: rawCenterY },
    { x: rawCenterX + halfW, y: rawCenterY },
    { x: rawCenterX - halfW * 0.45, y: rawCenterY - halfH * 0.4 },
    { x: rawCenterX + halfW * 0.45, y: rawCenterY - halfH * 0.4 },
    { x: rawCenterX, y: rawCenterY - halfH * 0.85 },
    { x: rawCenterX, y: rawCenterY + halfH * 0.85 },
  ];
}

function makeHandLandmarks(rawX: number, rawY: number) {
  return Array.from({ length: 21 }, (_, idx) => ({
    x: rawX + (idx % 3) * 0.004,
    y: rawY + Math.floor(idx / 3) * 0.003,
  }));
}

function makeUpperBodyPose(params: {
  headRawX: number;
  headRawY: number;
  leftWristRaw: { x: number; y: number; visibility: number };
  rightWristRaw: { x: number; y: number; visibility: number };
  shoulderY?: number;
}) {
  const shY = params.shoulderY ?? 0.62;
  const pose = Array.from({ length: 33 }, () => ({ x: params.headRawX, y: shY, visibility: 0.9 }));
  pose[0] = { x: params.headRawX, y: params.headRawY, visibility: 0.95 };
  pose[2] = { x: params.headRawX - 0.03, y: params.headRawY - 0.02, visibility: 0.95 };
  pose[5] = { x: params.headRawX + 0.03, y: params.headRawY - 0.02, visibility: 0.95 };
  pose[POSE_INDEX.LEFT_SHOULDER] = { x: params.headRawX + 0.14, y: shY, visibility: 0.9 };
  pose[POSE_INDEX.RIGHT_SHOULDER] = { x: params.headRawX - 0.14, y: shY, visibility: 0.9 };
  pose[POSE_INDEX.LEFT_WRIST] = params.leftWristRaw;
  pose[POSE_INDEX.RIGHT_WRIST] = params.rightWristRaw;
  return pose;
}

function makeMatchedPayload(): KioskLoginSuccessPayload {
  return buildKioskLoginSuccessPayload({
    patient: {
      id: 'pat-sdwa-01',
      hn: 'HN-344304',
      fullName: 'sdwa',
      nextAppointmentDate: '2026-11-23',
    },
    todayRoutine: null,
    confidencePercent: 96.6,
  });
}

function makeCardRef<T>(): React.RefObject<T | null> {
  return { current: null };
}

describe('Kiosk Fast 60FPS Biometric Login, Multi-Angle 2-Frame Lock & 2.0s Two-Hand Gate', () => {
  it('enforces async inference mutex (zero overlap + ~190ms throttle) and resolves registered patients across all angles in 2 frames (< 0.5s)', async () => {
    expect(KIOSK_INFERENCE_CANVAS_MAX_WIDTH).toBe(320);
    expect(KIOSK_DETECT_MAX_LONG_SIDE_PX).toBe(320);
    expect(KIOSK_INFERENCE_INTERVAL_MS).toBeGreaterThanOrEqual(180);
    expect(KIOSK_INFERENCE_INTERVAL_MS).toBeLessThanOrEqual(220);
    expect(KIOSK_FAST_MATCH_STREAK).toBe(2);
    expect(KIOSK_STREAK_GRACE_MS).toBe(500);

    // 1. Forgiving clear-face check accepts normal webcam frames without iris/stillness gates
    expect(isClearFaceForFastScan({ faceScore: 0.72, boxWidthRatio: 0.22, isSpoof: false })).toBe(true);
    expect(isClearFaceForFastScan({ faceScore: 0.50, boxWidthRatio: 0.22, isSpoof: false })).toBe(false);
    expect(isClearFaceForFastScan({ faceScore: 0.85, boxWidthRatio: 0.12, isSpoof: false })).toBe(false);
    expect(isClearFaceForFastScan({ faceScore: 0.95, boxWidthRatio: 0.30, isSpoof: true })).toBe(false);

    // 2. Async inference mutex prevents overlapping detect() calls and throttles within ~190ms
    const mutex = new AsyncInferenceMutex(KIOSK_INFERENCE_INTERVAL_MS);
    let resolveFirst!: () => void;
    const holdFirstGate = new Promise<void>((r) => {
      resolveFirst = r;
    });
    let activeCalls = 0;
    let maxConcurrentCalls = 0;
    const executedTimestamps: number[] = [];

    const firstPromise = mutex.tryRun(1000, async () => {
      activeCalls += 1;
      maxConcurrentCalls = Math.max(maxConcurrentCalls, activeCalls);
      executedTimestamps.push(1000);
      await holdFirstGate;
      activeCalls -= 1;
      return 'frame-1';
    });
    expect(firstPromise).not.toBeNull();
    expect(mutex.isBusy).toBe(true);

    // Attempt overlapping call while first inference is still in-flight -> MUST be rejected (null)
    const overlapping = mutex.tryRun(1250, async () => {
      activeCalls += 1;
      return 'overlap';
    });
    expect(overlapping).toBeNull();

    resolveFirst();
    await expect(firstPromise).resolves.toBe('frame-1');
    expect(mutex.isBusy).toBe(false);

    // Attempt call too soon after previous dispatch (100ms < 190ms) -> throttled (null)
    expect(mutex.tryRun(1100, async () => 'too-soon')).toBeNull();

    // Next call after 200ms (>= 190ms) runs cleanly -> 2 frames complete in ~200ms (< 0.5s!)
    const secondPromise = mutex.tryRun(1200, async () => {
      activeCalls += 1;
      maxConcurrentCalls = Math.max(maxConcurrentCalls, activeCalls);
      executedTimestamps.push(1200);
      activeCalls -= 1;
      return 'frame-2';
    });
    await expect(secondPromise).resolves.toBe('frame-2');
    expect(maxConcurrentCalls).toBe(1);
    expect(executedTimestamps).toEqual([1000, 1200]);
    expect(executedTimestamps[1] - executedTimestamps[0]).toBeLessThan(500);

    // 3. Best-of-All-Angles similarity matches tilted probes against masterEmbedding + angleEmbeddings + exemplars
    const centerVec = makeUnitVec(1.3);
    const leftVec = makeUnitVec(2.7);
    const rightVec = makeUnitVec(3.9);
    const upVec = makeUnitVec(5.1);
    const downVec = makeUnitVec(6.3);
    const masterVec = l2Normalize(centerVec.map((v, i) => v * 0.5 + (leftVec[i] + rightVec[i]) * 0.25));
    const candidate = {
      masterEmbedding: masterVec,
      angleEmbeddings: {
        center: centerVec,
        left: leftVec,
        right: rightVec,
        up: upVec,
        down: downVec,
      },
      exemplarEmbeddings: [mixUnitVec(rightVec, centerVec, 0.2)],
    };

    // Slightly tilted right probe matches `angleEmbeddings.right` effortlessly at >= 0.98
    const tiltedProbe = mixUnitVec(rightVec, makeUnitVec(9.4), 0.1);
    const bestRight = computeBestMultiAngleSimilarity(tiltedProbe, candidate);
    expect(bestRight.evaluatedCount).toBe(7);
    expect(bestRight.bestSource).toBe('right');
    expect(bestRight.maxSimilarity).toBeGreaterThanOrEqual(0.95);

    // Unregistered stranger stays well below threshold across ALL stored angles
    const strangerProbe = makeUnitVec(77.7);
    const strangerScore = computeBestMultiAngleSimilarity(strangerProbe, candidate);
    expect(strangerScore.maxSimilarity).toBeLessThan(0.35);
  });

  it('preserves streak across transient single-frame dropouts within the 500ms grace window and locks on 2/2', () => {
    let state = createInitialFastScanStreakState();

    // Frame 1 match at t=1000ms -> 1/2
    const m1 = stepFastScanStreakWithGrace(state, {
      type: 'match',
      patientId: 'pat-sdwa-01',
      score: 0.95,
      nowMs: 1000,
    });
    state = m1;
    expect(m1.streak).toBe(1);
    expect(m1.confirmed).toBe(false);

    // Transient single-frame dropout at t=1190ms (0ms into dropout) -> MUST NOT reset streak!
    const d1 = stepFastScanStreakWithGrace(state, { type: 'dropout', nowMs: 1190 });
    state = d1;
    expect(d1.streak).toBe(1);
    expect(d1.didReset).toBe(false);

    // Still within 500ms grace window at t=1550ms (360ms elapsed) -> STILL holds 1/2
    const d2 = stepFastScanStreakWithGrace(state, { type: 'dropout', nowMs: 1550 });
    state = d2;
    expect(d2.streak).toBe(1);
    expect(d2.didReset).toBe(false);

    // Frame 2 recovers at t=1600ms -> locks immediately at 2/2!
    const m2 = stepFastScanStreakWithGrace(state, {
      type: 'match',
      patientId: 'pat-sdwa-01',
      score: 0.96,
      nowMs: 1600,
    });
    expect(m2.streak).toBe(2);
    expect(m2.confirmed).toBe(true);

    // Sustained dropout > 500ms cleanly resets streak to 0
    const s1 = stepFastScanStreakWithGrace(createInitialFastScanStreakState(), {
      type: 'match',
      patientId: 'pat-sdwa-01',
      score: 0.95,
      nowMs: 2000,
    });
    const dStart = stepFastScanStreakWithGrace(s1, { type: 'dropout', nowMs: 2200 });
    expect(dStart.streak).toBe(1);
    const dExpired = stepFastScanStreakWithGrace(dStart, { type: 'dropout', nowMs: 2750 });
    expect(dExpired.streak).toBe(0);
    expect(dExpired.didReset).toBe(true);
  });

  it('prefers the live playing confirmVideoRef over a paused scan videoRef and continuously tracks moving faces (never frozen)', () => {
    const pausedScanVideo = {
      readyState: 4,
      videoWidth: 1280,
      videoHeight: 720,
      paused: true,
      ended: false,
      currentTime: 3.12,
    };
    const liveConfirmVideo = {
      readyState: 4,
      videoWidth: 1280,
      videoHeight: 720,
      paused: false,
      ended: false,
      currentTime: 5.44,
    };

    expect(resolvePostMatchActiveVideoElement(liveConfirmVideo, pausedScanVideo)).toBe(
      liveConfirmVideo,
    );
    const warmingConfirmVideo = { readyState: 0, videoWidth: 0, paused: true, ended: false };
    const playingScanVideo = { ...pausedScanVideo, paused: false };
    expect(resolvePostMatchActiveVideoElement(warmingConfirmVideo, playingScanVideo)).toBe(
      playingScanVideo,
    );

    let gate = createInitialPostMatchGestureGateState();
    const rawPositions = [0.5, 0.55, 0.61, 0.67, 0.72];
    const trackedScreenCentersX: number[] = [];

    for (let i = 0; i < rawPositions.length; i += 1) {
      const next = stepPostMatchGestureGateFrame(gate, {
        nowMs: 1000 + i * 33,
        dtMs: 33,
        speechBlocked: i < 2,
        ownerName: 'sdwa',
        ownerFaceVerified: true,
        faceLandmarks: makeFaceLandmarks(rawPositions[i], 0.46),
        faceCount: 1,
        hands: [],
        pose: null,
        videoW: 1280,
        videoH: 720,
        canvasW: 400,
        canvasH: 225,
      });
      gate = next;
      expect(next.smoothedFaceBox).not.toBeNull();
      trackedScreenCentersX.push(next.smoothedFaceBox!.centerX);
    }

    for (let i = 1; i < trackedScreenCentersX.length; i += 1) {
      expect(trackedScreenCentersX[i]).toBeLessThan(trackedScreenCentersX[i - 1]);
    }
    expect(trackedScreenCentersX[0]).toBeCloseTo(0.5, 2);
    expect(trackedScreenCentersX[trackedScreenCentersX.length - 1]).toBeLessThan(0.36);
  });

  it('detects both hands via HandLandmarker OR PoseLandmarker wrists, completes 2.0s hold smoothly, and auto-proceeds', () => {
    expect(VERIFIED_HAND_HOLD_MS).toBe(2000);
    expect(HAND_GLITCH_GRACE_MS).toBe(250);
    expect(VERIFIED_MODAL_TTS_STALL_TIMEOUT_MS).toBe(2500);

    const payload = makeMatchedPayload();
    const proceedToExerciseSession = vi.fn();

    // 1. Pose wrist dual-detector fallback triggers BOTH hands even when HandLandmarker returns 0 hands
    const poseOnlyFrame = stepPostMatchGestureGateFrame(createInitialPostMatchGestureGateState(), {
      nowMs: 1000,
      dtMs: 16,
      speechBlocked: false,
      ownerName: payload.patient.fullName,
      ownerFaceVerified: true,
      faceLandmarks: makeFaceLandmarks(0.5, 0.42),
      faceCount: 1,
      hands: [],
      pose: makeUpperBodyPose({
        headRawX: 0.5,
        headRawY: 0.42,
        shoulderY: 0.62,
        leftWristRaw: { x: 0.72, y: 0.52, visibility: 0.38 },
        rightWristRaw: { x: 0.28, y: 0.51, visibility: 0.41 },
      }),
      poseCount: 1,
      videoW: 1280,
      videoH: 720,
      canvasW: 400,
      canvasH: 225,
    });
    expect(poseOnlyFrame.leftHandRaised).toBe(true);
    expect(poseOnlyFrame.rightHandRaised).toBe(true);
    expect(poseOnlyFrame.bothHandsRaised).toBe(true);
    expect(poseOnlyFrame.holdSeconds).toBeGreaterThan(0);

    // 2. Full 60FPS HandLandmarker hold progression from 0.0s -> 2.0s with brief <250ms drop grace buffer
    let gate = createInitialPostMatchGestureGateState();
    let nowMs = 2000;
    const faceLms = makeFaceLandmarks(0.5, 0.44);
    const leftHand = makeHandLandmarks(0.72, 0.46);
    const rightHand = makeHandLandmarks(0.28, 0.46);

    // Hold both hands for 1.0 second (50 frames * 20ms) -> 50% (1000ms)
    for (let i = 0; i < 50; i += 1) {
      nowMs += 20;
      const step = stepPostMatchGestureGateFrame(gate, {
        nowMs,
        dtMs: 20,
        speechBlocked: false,
        ownerName: payload.patient.fullName,
        ownerFaceVerified: true,
        faceLandmarks: faceLms,
        faceCount: 1,
        hands: [leftHand, rightHand],
        pose: null,
        videoW: 1280,
        videoH: 720,
        canvasW: 400,
        canvasH: 225,
      });
      gate = step;
      if (step.shouldProceedToExercise) proceedToExerciseSession(payload);
    }
    expect(gate.leftHandRaised).toBe(true);
    expect(gate.rightHandRaised).toBe(true);
    expect(gate.holdMs).toBe(1000);
    expect(proceedToExerciseSession).not.toHaveBeenCalled();

    // Brief 180ms single-hand drop (< 250ms grace buffer): timer holds steady at 1.0s
    for (let i = 0; i < 9; i += 1) {
      nowMs += 20;
      const step = stepPostMatchGestureGateFrame(gate, {
        nowMs,
        dtMs: 20,
        speechBlocked: false,
        ownerName: payload.patient.fullName,
        ownerFaceVerified: true,
        faceLandmarks: faceLms,
        faceCount: 1,
        hands: [leftHand],
        pose: null,
        videoW: 1280,
        videoH: 720,
        canvasW: 400,
        canvasH: 225,
      });
      gate = step;
    }
    expect(gate.holdMs).toBe(1000);

    // Hold remaining 1.0 second -> reaches 2.0s (100%) and automatically triggers navigation!
    let finalProgress = 0;
    for (let i = 0; i < 50; i += 1) {
      nowMs += 20;
      const step = stepPostMatchGestureGateFrame(gate, {
        nowMs,
        dtMs: 20,
        speechBlocked: false,
        ownerName: payload.patient.fullName,
        ownerFaceVerified: true,
        faceLandmarks: faceLms,
        faceCount: 1,
        hands: [leftHand, rightHand],
        pose: null,
        videoW: 1280,
        videoH: 720,
        canvasW: 400,
        canvasH: 225,
      });
      gate = step;
      finalProgress = step.progress;
      if (step.shouldProceedToExercise) {
        proceedToExerciseSession(payload);
      }
    }
    expect(gate.holdMs).toBe(2000);
    expect(finalProgress).toBe(1);
    expect(gate.confirmed).toBe(true);
    expect(proceedToExerciseSession).toHaveBeenCalledTimes(1);
    expect(proceedToExerciseSession).toHaveBeenCalledWith(payload);

    // 3. Render KioskVerifiedConfirmationCard in waiting vs both-hands-raised states
    const waitingCopy = getKioskVerifiedModalCopy('th', {
      fullName: payload.patient.fullName,
      hn: payload.patient.hn,
      nextAppointmentDate: payload.patient.nextAppointmentDate,
      confidencePercent: payload.confidencePercent,
      twoHandProgressPct: 0,
    });
    const waitingHtml = renderToStaticMarkup(
      <KioskVerifiedConfirmationCard
        lang="th"
        verifiedModalPayload={payload}
        modalCopy={waitingCopy}
        scanGeneration={1}
        verifiedAnimDone={true}
        voiceMuted={false}
        isWelcomeSpeechDone={true}
        raisedDuringSpeech={false}
        twoHandCameraReady={true}
        twoHandDetectorReady={true}
        twoHandDetected={false}
        twoHandProgress={0}
        leftHandUpState={false}
        rightHandUpState={false}
        ownerGuardBanner={null}
        confirmVideoRef={makeCardRef<HTMLVideoElement>()}
        confirmCanvasRef={makeCardRef<HTMLCanvasElement>()}
        confirmProgressFillRef={makeCardRef<HTMLDivElement>()}
        confirmProgressTextRef={makeCardRef<HTMLSpanElement>()}
        confirmPillLabelRef={makeCardRef<HTMLSpanElement>()}
        confirmBtnFillRef={makeCardRef<HTMLSpanElement>()}
        confirmBtnLabelRef={makeCardRef<HTMLSpanElement>()}
        onSelectLang={() => {}}
        onToggleVoiceMute={() => {}}
        onProceedToExercise={() => {}}
        onRejectWrongMatch={() => {}}
      />,
    );
    expect(waitingHtml).toContain('aspect-video w-full');
    expect(waitingHtml).toContain('รอมือซ้าย');
    expect(waitingHtml).toContain('รอมือขวา');
    expect(waitingHtml).toContain('bg-slate-800/80 text-white');
    expect(waitingHtml).toContain('0.0 / 2.0 วิ (0%)');
    expect(waitingHtml).toContain('ยกมือทั้ง 2 ข้างค้าง 2 วิ หรือ กดเพื่อเริ่มฝึกทันที');
    expect(EMOJI_RE.test(waitingHtml)).toBe(false);

    const raisedCopy = getKioskVerifiedModalCopy('th', {
      fullName: payload.patient.fullName,
      hn: payload.patient.hn,
      nextAppointmentDate: payload.patient.nextAppointmentDate,
      confidencePercent: payload.confidencePercent,
      twoHandProgressPct: 100,
    });
    const raisedHtml = renderToStaticMarkup(
      <KioskVerifiedConfirmationCard
        lang="th"
        verifiedModalPayload={payload}
        modalCopy={raisedCopy}
        scanGeneration={1}
        verifiedAnimDone={true}
        voiceMuted={false}
        isWelcomeSpeechDone={true}
        raisedDuringSpeech={false}
        twoHandCameraReady={true}
        twoHandDetectorReady={true}
        twoHandDetected={true}
        twoHandProgress={1}
        leftHandUpState={true}
        rightHandUpState={true}
        ownerGuardBanner={null}
        confirmVideoRef={makeCardRef<HTMLVideoElement>()}
        confirmCanvasRef={makeCardRef<HTMLCanvasElement>()}
        confirmProgressFillRef={makeCardRef<HTMLDivElement>()}
        confirmProgressTextRef={makeCardRef<HTMLSpanElement>()}
        confirmPillLabelRef={makeCardRef<HTMLSpanElement>()}
        confirmBtnFillRef={makeCardRef<HTMLSpanElement>()}
        confirmBtnLabelRef={makeCardRef<HTMLSpanElement>()}
        onSelectLang={() => {}}
        onToggleVoiceMute={() => {}}
        onProceedToExercise={() => {}}
        onRejectWrongMatch={() => {}}
      />,
    );
    expect(raisedHtml).toContain('ยกมือซ้ายแล้ว');
    expect(raisedHtml).toContain('ยกมือขวาแล้ว');
    expect(raisedHtml).toContain('bg-emerald-600 text-white');
    expect(raisedHtml).toContain('2.0 / 2.0 วิ (100%)');
    expect(raisedHtml).toContain('scaleX(1.0000)');
    expect(EMOJI_RE.test(raisedHtml)).toBe(false);
  });

  it('invokes proceedToExerciseSession immediately on manual bypass tap and audits all modified files for zero Unicode emojis', () => {
    const payload = makeMatchedPayload();
    const onProceed = vi.fn();
    const copy = getKioskVerifiedModalCopy('th', {
      fullName: payload.patient.fullName,
      hn: payload.patient.hn,
      nextAppointmentDate: payload.patient.nextAppointmentDate,
      confidencePercent: payload.confidencePercent,
      twoHandProgressPct: 0,
    });

    const tree = KioskVerifiedConfirmationCard({
      lang: 'th',
      verifiedModalPayload: payload,
      modalCopy: copy,
      scanGeneration: 1,
      verifiedAnimDone: true,
      voiceMuted: false,
      isWelcomeSpeechDone: false,
      raisedDuringSpeech: false,
      twoHandCameraReady: true,
      twoHandDetectorReady: true,
      twoHandDetected: false,
      twoHandProgress: 0,
      leftHandUpState: false,
      rightHandUpState: false,
      ownerGuardBanner: null,
      confirmVideoRef: makeCardRef<HTMLVideoElement>(),
      confirmCanvasRef: makeCardRef<HTMLCanvasElement>(),
      confirmProgressFillRef: makeCardRef<HTMLDivElement>(),
      confirmProgressTextRef: makeCardRef<HTMLSpanElement>(),
      confirmPillLabelRef: makeCardRef<HTMLSpanElement>(),
      confirmBtnFillRef: makeCardRef<HTMLSpanElement>(),
      confirmBtnLabelRef: makeCardRef<HTMLSpanElement>(),
      onSelectLang: () => {},
      onToggleVoiceMute: () => {},
      onProceedToExercise: onProceed,
      onRejectWrongMatch: () => {},
    });

    const buttons: React.ReactElement<{ 'data-testid'?: string; onClick?: () => void }>[] = [];
    const walk = (node: unknown): void => {
      if (!node || typeof node !== 'object') return;
      if (Array.isArray(node)) {
        node.forEach(walk);
        return;
      }
      const el = node as React.ReactElement<{
        'data-testid'?: string;
        onClick?: () => void;
        children?: unknown;
      }>;
      if (el.type === 'button' && typeof el.props?.onClick === 'function') {
        buttons.push(el);
      }
      if (el.props?.children) {
        walk(el.props.children);
      }
    };
    walk(tree);

    const bypassBtn = buttons.find((b) => b.props['data-testid'] === 'kiosk-confirm-start-btn');
    expect(bypassBtn).toBeDefined();
    bypassBtn?.props.onClick?.();
    expect(onProceed).toHaveBeenCalledTimes(1);

    // Structural regression checks & zero-emoji audit across all modified files
    const root = process.cwd();
    const kioskSrc = fs.readFileSync(
      path.join(root, 'src/components/KioskBiometricLoginView.tsx'),
      'utf8',
    );
    expect(kioskSrc).toContain('isInferringRef');
    expect(kioskSrc).toContain('KIOSK_INFERENCE_CANVAS_MAX_WIDTH = 320');
    expect(kioskSrc).toContain('KIOSK_FAST_MATCH_STREAK = 2');
    expect(kioskSrc).toContain('KIOSK_STREAK_GRACE_MS = 500');
    expect(kioskSrc).toContain('VERIFIED_HAND_HOLD_MS = 2000');
    expect(kioskSrc).toContain('กรุณามองกล้องในระยะพอดี');
    expect(kioskSrc).not.toContain('v.pause();');

    const auditedFiles = [
      'src/components/KioskBiometricLoginView.tsx',
      'src/components/StaffAssistedEnrollmentModal.tsx',
      'src/components/DrTechCareMascot.tsx',
      'src/App.tsx',
      'src/lib/biometricEngine.ts',
      'src/lib/biometricVisionCache.ts',
      'packages/smart-biometric-core/src/vectorMath.ts',
      'packages/api-client/src/cloudRouter.ts',
      'apps/api/src/app.ts',
      'src/__tests__/kioskBiometricLogin.test.tsx',
      'src/__tests__/kioskBiometricLogin.test.ts',
    ];
    for (const rel of auditedFiles) {
      const content = fs.readFileSync(path.join(root, rel), 'utf8');
      expect(EMOJI_RE.test(content), `Unexpected Unicode emoji in ${rel}`).toBe(false);
    }
  });

  it('immediately injects newly registered patients into Kiosk scan list, updates synced count, and allows instant face login without reload', () => {
    const existingPatient: PatientRecord = {
      id: 'pat-existing-1',
      hn: 'HN-410382',
      fullName: 'Poom',
      age: 42,
      phone: '0811111111',
      diagnosis: 'Stroke Rehab',
      assignedPtId: 'pt-1',
      nextAppointmentDate: '2026-11-23',
      registrationStatus: 'active',
      enrolledAt: '2026-11-20T08:00:00.000Z',
      faceBiometricStatus: 'VERIFIED',
      faceFramesCount: 120,
      masterEmbedding: makeUnitVec(11.1),
      angleEmbeddings: {
        center: makeUnitVec(11.1),
        left: makeUnitVec(11.2),
        right: makeUnitVec(11.3),
        up: makeUnitVec(11.4),
        down: makeUnitVec(11.5),
      },
    };
    const newCenter = makeUnitVec(22.1);
    const newLeft = makeUnitVec(22.2);
    const newlyRegistered: PatientRecord = {
      id: 'pat-new-2',
      hn: 'HN-990123',
      fullName: 'Somchai NewlyEnrolled',
      age: 58,
      phone: '0892223333',
      diagnosis: 'Post-Stroke Upper Limb',
      assignedPtId: 'pt-1',
      nextAppointmentDate: '2026-11-23',
      registrationStatus: 'active',
      enrolledAt: '2026-11-23T07:00:00.000Z',
      // Even if incoming status was stale PENDING while vectors exist, merge normalizes to VERIFIED
      faceBiometricStatus: 'PENDING',
      faceFramesCount: 0,
      masterEmbedding: newCenter,
      angleEmbeddings: {
        center: newCenter,
        left: newLeft,
        right: makeUnitVec(22.3),
        up: makeUnitVec(22.4),
        down: makeUnitVec(22.5),
      },
      exemplarEmbeddings: [mixUnitVec(newCenter, newLeft, 0.25)],
    };

    const merged = mergeEnrolledPatientIntoKioskList([existingPatient], newlyRegistered);
    expect(merged).toHaveLength(2);
    expect(merged[0].id).toBe('pat-new-2');
    expect(merged[0].faceBiometricStatus).toBe('VERIFIED');
    expect(merged[0].faceFramesCount).toBe(120);
    expect(formatKioskSupabaseSyncBadgeLabel(merged.length, merged.length)).toBe(
      'ซิงก์ฐานข้อมูลใบหน้าจาก Supabase สำเร็จ: 2 บัญชี',
    );

    // Live tilted probe of newly registered user matches immediately without page reload
    const liveProbe = mixUnitVec(newLeft, makeUnitVec(99.1), 0.1);
    const matched = matchLocalEnrolledPatientsStrict(liveProbe, merged);
    expect(matched).not.toBeNull();
    expect(matched?.patient.id).toBe('pat-new-2');
    expect(matched?.patient.fullName).toBe('Somchai NewlyEnrolled');
    expect(matched?.confidencePercent).toBeGreaterThanOrEqual(90);

    // Unregistered stranger is STILL strictly rejected
    const strangerProbe = makeUnitVec(777.7);
    expect(matchLocalEnrolledPatientsStrict(strangerProbe, merged)).toBeNull();
  });

  it('uses passive live-webcam liveness without blink trap, keeps top & bottom (X/2) counters synced, and transitions directly to Success with greeting mascot', () => {
    // 1. Passive live-webcam liveness passes cleanly on >= 0.8px natural micro-movement with blinkCount=0
    expect(PASSIVE_LIVENESS_MIN_PIXEL_SHIFT).toBe(0.8);
    const livePass = evaluatePassiveWebcamLiveness({
      prevAnchors: { boxCenterX: 320, boxCenterY: 240, noseX: 321, noseY: 245, yawDeg: 1.0, pitchDeg: 0.5 },
      nextAnchors: { boxCenterX: 321.1, boxCenterY: 240.3, noseX: 322.2, noseY: 245.5, yawDeg: 1.1, pitchDeg: 0.5 },
      bioBiologicalVerified: false,
      blinkCount: 0,
      isSpoof: false,
    });
    expect(livePass.liveVerified).toBe(true);
    expect(livePass.signalReason).toBe('micro_motion');

    // Spoofed static photo is STILL blocked
    const spoofBlocked = evaluatePassiveWebcamLiveness({
      prevAnchors: { boxCenterX: 320, boxCenterY: 240 },
      nextAnchors: { boxCenterX: 322, boxCenterY: 240 },
      bioBiologicalVerified: false,
      blinkCount: 0,
      isSpoof: true,
    });
    expect(spoofBlocked.liveVerified).toBe(false);
    expect(spoofBlocked.signalReason).toBe('spoof');

    // 2. Top status badge and bottom-right counter read from the exact same scanMatchCount (never 2/2 vs 0/2)
    const step0 = resolveKioskUnifiedScanProgress({ scanMatchCount: 0, isMatched: false });
    expect(step0.topLabel).toBe('กำลังตรวจสอบใบหน้า... (0/2)');
    expect(step0.counterLabel).toBe('กำลังตรวจสอบใบหน้า... (0/2)');
    expect(step0.isComplete).toBe(false);

    const step1 = resolveKioskUnifiedScanProgress({ scanMatchCount: 1, isMatched: false });
    expect(step1.topLabel).toBe('กำลังตรวจสอบใบหน้า... (1/2)');
    expect(step1.counterLabel).toBe('กำลังตรวจสอบใบหน้า... (1/2)');
    expect(step1.isComplete).toBe(false);

    const step2 = resolveKioskUnifiedScanProgress({ scanMatchCount: 2, isMatched: true });
    expect(step2.topLabel).toBe('ยืนยันสำเร็จ! (2/2)');
    expect(step2.counterLabel).toBe('กำลังตรวจสอบใบหน้า... (2/2)');
    expect(step2.isComplete).toBe(true);
    expect(step2.progressPct).toBe(100);

    // 3. Once matched or matching, NEVER fall back to "not_found" and NEVER show forced blink trap
    expect(
      shouldTriggerFastNotFoundVerdict({
        candidatesLoaded: true,
        humanWarmupReady: true,
        documentHidden: false,
        isAmbiguous: false,
        isSpoof: false,
        bioVerified: true,
        matchStreak: 2,
        hasMatchedPatient: true,
        unmatchedClearFrames: 10,
        unmatchedElapsedMs: 5000,
        inRescanCooldown: false,
        hasZeroCandidates: false,
        totalEvaluatedFrames: 10,
        totalDecisionElapsedMs: 5000,
      }),
    ).toBe(false);

    const calmLivenessOverlay = resolveKioskFaceBoundingBoxOverlay({
      isSpoof: false,
      isBioVerified: false,
      rawCosine: 0.92,
      streak: 1,
      requiredStreak: 2,
      confirmedCandidateName: null,
      distanceM: 1.3,
    });
    expect(calmLivenessOverlay.label).toBe('กำลังยืนยันใบหน้าจริง...');
    expect(calmLivenessOverlay.label).not.toContain('กะพริบตา');

    // 4. Success confirmation card renders DrTechCareMascot in 'greeting' pose with zero emojis
    const payload = makeMatchedPayload();
    const copy = getKioskVerifiedModalCopy('th', {
      fullName: payload.patient.fullName,
      hn: payload.patient.hn,
      nextAppointmentDate: payload.patient.nextAppointmentDate,
      confidencePercent: payload.confidencePercent,
      twoHandProgressPct: 0,
    });
    const successHtml = renderToStaticMarkup(
      <KioskVerifiedConfirmationCard
        lang="th"
        verifiedModalPayload={payload}
        modalCopy={copy}
        scanGeneration={2}
        verifiedAnimDone={true}
        voiceMuted={false}
        isWelcomeSpeechDone={false}
        raisedDuringSpeech={false}
        twoHandCameraReady={true}
        twoHandDetectorReady={true}
        twoHandDetected={false}
        twoHandProgress={0}
        leftHandUpState={false}
        rightHandUpState={false}
        ownerGuardBanner={null}
        confirmVideoRef={makeCardRef<HTMLVideoElement>()}
        confirmCanvasRef={makeCardRef<HTMLCanvasElement>()}
        confirmProgressFillRef={makeCardRef<HTMLDivElement>()}
        confirmProgressTextRef={makeCardRef<HTMLSpanElement>()}
        confirmPillLabelRef={makeCardRef<HTMLSpanElement>()}
        confirmBtnFillRef={makeCardRef<HTMLSpanElement>()}
        confirmBtnLabelRef={makeCardRef<HTMLSpanElement>()}
        onSelectLang={() => {}}
        onToggleVoiceMute={() => {}}
        onProceedToExercise={() => {}}
        onRejectWrongMatch={() => {}}
      />,
    );
    expect(successHtml).toContain('data-testid="drtechcare-mascot"');
    expect(successHtml).toContain('data-pose="greeting"');
    expect(EMOJI_RE.test(successHtml)).toBe(false);
  });
});
