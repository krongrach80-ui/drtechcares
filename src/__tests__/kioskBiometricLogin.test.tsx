import fs from 'node:fs';
import path from 'node:path';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it, vi } from 'vitest';
import { POSE_INDEX } from '@drtechcares/pose-engine';
import {
  HAND_GLITCH_GRACE_MS,
  KioskVerifiedConfirmationCard,
  VERIFIED_HAND_HOLD_MS,
  VERIFIED_MODAL_TTS_STALL_TIMEOUT_MS,
  buildKioskLoginSuccessPayload,
  createInitialPostMatchGestureGateState,
  getKioskVerifiedModalCopy,
  resolvePostMatchActiveVideoElement,
  stepPostMatchGestureGateFrame,
  type KioskLoginSuccessPayload,
} from '../components/KioskBiometricLoginView';

const EMOJI_RE = /[\p{Extended_Pictographic}\u{FE0F}\u{2600}-\u{27BF}\u{23E9}-\u{23FA}]/u;

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

describe('Kiosk Success Confirmation Card — Live 60FPS Face Tracking & Two-Hand Raise Gate', () => {
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

    // Must ALWAYS pick the actively playing confirmVideo rather than a paused scan video
    expect(resolvePostMatchActiveVideoElement(liveConfirmVideo, pausedScanVideo)).toBe(
      liveConfirmVideo,
    );
    // While confirmVideo is still warming up on Frame 0, safely falls back to a live playing scan video
    const warmingConfirmVideo = { readyState: 0, videoWidth: 0, paused: true, ended: false };
    const playingScanVideo = { ...pausedScanVideo, paused: false };
    expect(resolvePostMatchActiveVideoElement(warmingConfirmVideo, playingScanVideo)).toBe(
      playingScanVideo,
    );

    // Simulate user moving from center (rawX = 0.50 -> screenX = 0.50) toward the left
    // (rawX = 0.72 -> screenX = 0.28) across frames BOTH during speech and after speech.
    let gate = createInitialPostMatchGestureGateState();
    const rawPositions = [0.5, 0.55, 0.61, 0.67, 0.72];
    const trackedScreenCentersX: number[] = [];

    for (let i = 0; i < rawPositions.length; i += 1) {
      const next = stepPostMatchGestureGateFrame(gate, {
        nowMs: 1000 + i * 33,
        dtMs: 33,
        speechBlocked: i < 2, // first 2 frames during welcome speech, remaining after speech
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

    // Face bounding box MUST move continuously leftward (decreasing mirrored centerX) and NEVER freeze
    for (let i = 1; i < trackedScreenCentersX.length; i += 1) {
      expect(trackedScreenCentersX[i]).toBeLessThan(trackedScreenCentersX[i - 1]);
    }
    expect(trackedScreenCentersX[0]).toBeCloseTo(0.5, 2);
    expect(trackedScreenCentersX[trackedScreenCentersX.length - 1]).toBeLessThan(0.36);
  });

  it('detects both hands via HandLandmarker OR PoseLandmarker wrists, updates badges, advances 0.0 -> 4.0s smoothly, and auto-proceeds', () => {
    expect(VERIFIED_HAND_HOLD_MS).toBe(4000);
    expect(HAND_GLITCH_GRACE_MS).toBe(250);
    expect(VERIFIED_MODAL_TTS_STALL_TIMEOUT_MS).toBe(2500);

    const payload = makeMatchedPayload();
    const proceedToExerciseSession = vi.fn();

    // 1. Verify Pose wrist dual-detector fallback triggers BOTH hands even when HandLandmarker returns 0 hands
    const poseOnlyFrame = stepPostMatchGestureGateFrame(createInitialPostMatchGestureGateState(), {
      nowMs: 1000,
      dtMs: 16,
      speechBlocked: false,
      ownerName: payload.patient.fullName,
      ownerFaceVerified: true,
      faceLandmarks: makeFaceLandmarks(0.5, 0.42),
      faceCount: 1,
      hands: [], // 0 HandLandmarker results!
      pose: makeUpperBodyPose({
        headRawX: 0.5,
        headRawY: 0.42,
        shoulderY: 0.62,
        // Confidence >= 0.35 and wrist.y < shoulder.y + 0.06 (both raised near shoulders)
        leftWristRaw: { x: 0.72, y: 0.52, visibility: 0.38 }, // mirrored left (x = 0.28)
        rightWristRaw: { x: 0.28, y: 0.51, visibility: 0.41 }, // mirrored right (x = 0.72)
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

    // 2. Full 60FPS HandLandmarker hold progression from 0.0s -> 4.0s with brief <250ms drop grace buffer
    let gate = createInitialPostMatchGestureGateState();
    let nowMs = 2000;
    const faceLms = makeFaceLandmarks(0.5, 0.44);
    const leftHand = makeHandLandmarks(0.72, 0.46); // mirrored x = 0.28 (left), y < 0.72
    const rightHand = makeHandLandmarks(0.28, 0.46); // mirrored x = 0.72 (right), y < 0.72

    // Hold both hands for 2.0 seconds (100 frames * 20ms)
    for (let i = 0; i < 100; i += 1) {
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
    expect(gate.holdMs).toBe(2000);
    expect(proceedToExerciseSession).not.toHaveBeenCalled();

    // Brief 180ms single-hand drop (< 250ms grace buffer): timer MUST hold steady at 2.0s
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
    expect(gate.holdMs).toBe(2000);

    // Hold remaining 2.0 seconds -> reaches 4.0s (100%) and automatically triggers navigation!
    let finalProgress = 0;
    for (let i = 0; i < 100; i += 1) {
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
    expect(gate.holdMs).toBe(4000);
    expect(finalProgress).toBe(1);
    expect(gate.confirmed).toBe(true);
    expect(proceedToExerciseSession).toHaveBeenCalledTimes(1);
    expect(proceedToExerciseSession).toHaveBeenCalledWith(payload);

    // 3. Render KioskVerifiedConfirmationCard in waiting vs both-hands-raised states and verify UI copy & classes
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
    expect(waitingHtml).toContain('0.0 / 4.0 วิ (0%)');
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
    expect(raisedHtml).toContain('4.0 / 4.0 วิ (100%)');
    expect(raisedHtml).toContain('scaleX(1.0000)');
    expect(EMOJI_RE.test(raisedHtml)).toBe(false);
  });

  it('invokes proceedToExerciseSession immediately when clicking the manual bypass button and contains zero Unicode emojis across modified files', () => {
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
      // Even if speech is still playing and progress is 0%, manual bypass button MUST work immediately!
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

    // Regression checks & zero-emoji audit on all modified files
    const root = process.cwd();
    const kioskSrc = fs.readFileSync(
      path.join(root, 'src/components/KioskBiometricLoginView.tsx'),
      'utf8',
    );
    expect(kioskSrc).toContain('resolvePostMatchActiveVideoElement(');
    expect(kioskSrc).toContain('detectedHands.length < 2');
    expect(kioskSrc).toContain('VERIFIED_MODAL_TTS_STALL_TIMEOUT_MS');
    expect(kioskSrc).not.toContain('v.pause();');

    const auditedFiles = [
      'src/components/KioskBiometricLoginView.tsx',
      'src/lib/ownerBiometricLock.ts',
      'src/__tests__/kioskBiometricLogin.test.tsx',
    ];
    for (const rel of auditedFiles) {
      const content = fs.readFileSync(path.join(root, rel), 'utf8');
      expect(EMOJI_RE.test(content), `Unexpected Unicode emoji in ${rel}`).toBe(false);
    }
  });
});

