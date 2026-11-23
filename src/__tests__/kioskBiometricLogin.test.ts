import fs from 'node:fs';
import path from 'node:path';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it, vi } from 'vitest';
import {
  BIOMETRIC_CENTERED_COSINE_THRESHOLD,
  BIOMETRIC_MATCH_THRESHOLD,
  BIOMETRIC_MAX_EUCLIDEAN_DISTANCE,
  BIOMETRIC_REQUIRED_STREAK,
  LiveIdentityStreakTracker,
  computeImageDataSharpness,
  normalizeImageDataLuminance,
  type EnrolledBiometricCandidate,
  type Live1toNMatchEvaluation,
} from '../lib/biometricEngine';
import {
  computeContrastNormalizedSharpness,
  resolveKioskReadinessGuidance,
  smoothBiometricBox,
  smoothFacialLandmarks,
} from '../lib/biometricTrackerSmoothing';
import {
  MIN_SHARPNESS_LAPLACIAN,
  evaluateTripleGateNeuralCandidate,
  l2Normalize,
} from '@drtechcares/smart-biometric-core';
import {
  KIOSK_DETECT_MAX_LONG_SIDE_PX,
  KIOSK_FAST_MATCH_STREAK,
  KIOSK_FAST_UNREGISTERED_FRAMES,
  KIOSK_FAST_UNREGISTERED_TIMEOUT_MS,
  KIOSK_MAX_DECISION_EVAL_FRAMES,
  KIOSK_MAX_DECISION_WINDOW_MS,
  KIOSK_MAX_EXPOSURE_SETTLE_MS,
  KIOSK_NO_FACE_GUIDANCE_DELAY_MS,
  KIOSK_NOT_FOUND_VOICE_PROMPT_TH,
  KIOSK_OVERLAY_MIN_INTERVAL_MS,
  KIOSK_RESCAN_COOLDOWN_MS,
  KioskSystemErrorPopup,
  UnregisteredFaceSignupPopup,
  shouldAutoReturnToStandby,
  shouldTriggerFastNotFoundVerdict,
} from '../components/KioskBiometricLoginView';
import {
  KioskBiometricScanProfiler,
  computeTimingPercentiles,
} from '../lib/biometricScanProfiler';
import { openBestAvailableCameraStream } from '@drtechcares/face-client';

const EMOJI_RE = /[\p{Extended_Pictographic}\u{FE0F}\u{2600}-\u{27BF}\u{23E9}-\u{23FA}]/u;

describe('Fast ~1s Face Scan Decision & Automatic Unregistered Signup Popup', () => {
  it('keeps strict biometric thresholds untouched while locking genuine matches cleanly within <= 3 frames', () => {
    expect(BIOMETRIC_MATCH_THRESHOLD).toBe(0.84);
    expect(BIOMETRIC_CENTERED_COSINE_THRESHOLD).toBe(0.8);
    expect(BIOMETRIC_MAX_EUCLIDEAN_DISTANCE).toBe(0.56);
    expect(BIOMETRIC_REQUIRED_STREAK).toBe(6);

    expect(KIOSK_FAST_MATCH_STREAK).toBeLessThanOrEqual(3);
    expect(KIOSK_FAST_MATCH_STREAK).toBeGreaterThanOrEqual(2);
    expect(KIOSK_FAST_UNREGISTERED_FRAMES).toBeLessThanOrEqual(3);
    expect(KIOSK_FAST_UNREGISTERED_TIMEOUT_MS).toBeLessThanOrEqual(1200);
    expect(KIOSK_NO_FACE_GUIDANCE_DELAY_MS).toBe(1500);
    expect(KIOSK_RESCAN_COOLDOWN_MS).toBe(1500);

    const tracker = new LiveIdentityStreakTracker(KIOSK_FAST_MATCH_STREAK);
    const dummyVec = new Array<number>(1024).fill(1 / Math.sqrt(1024));
    const candidate: EnrolledBiometricCandidate = {
      patientId: 'pat-fast-1',
      hn: 'HN-FAST-1',
      fullName: 'คุณสมศรี แข็งแรง',
      diagnosis: 'Stroke',
      assignedPtId: 'pt-1',
      nextAppointmentDate: '2026-12-01',
      framesCaptured: 120,
      qualityScore: 0.96,
      dimension: 1024,
      master_embedding: dummyVec,
      angle_embeddings: {
        center: dummyVec,
        left: dummyVec,
        right: dummyVec,
        up: dummyVec,
        down: dummyVec,
      },
      exemplar_embeddings: [dummyVec],
      sourceEntry: {
        patientId: 'pat-fast-1',
        hn: 'HN-FAST-1',
        fullName: 'คุณสมศรี แข็งแรง',
        diagnosis: 'Stroke',
        assignedPtId: 'pt-1',
        nextAppointmentDate: '2026-12-01',
        framesCaptured: 120,
        qualityScore: 0.96,
        masterEmbedding: dummyVec,
        angleEmbeddings: {
          center: dummyVec,
          left: dummyVec,
          right: dummyVec,
          up: dummyVec,
          down: dummyVec,
        },
      },
    };

    const subThresholdEval: Live1toNMatchEvaluation = {
      matched: false,
      topCandidate: null,
      runnerUpCandidate: null,
      topSimilarity: 0.81,
      topCenteredSimilarity: 0.78,
      topCentroidSimilarity: 0.8,
      topExemplarSimilarity: 0.8,
      topHumanSimilarity: 0.81,
      topFinalScore: 0.81,
      topEuclideanDistance: 0.62,
      topConsensusTop3Avg: 0.8,
      topPatchZncc: null,
      topCraniofacialDeviation: null,
      cohortMargin: 0.12,
      runnerUpSimilarity: 0,
      runnerUpFinalScore: 0,
      margin: 1,
      confidencePercent: 81,
      bestSource: 'master',
      reason: 'BELOW_THRESHOLD',
      validAccounts: 1,
      totalAccounts: 1,
    };
    expect(
      tracker.step(subThresholdEval, { biologicalVerified: true, spoofDetected: false }).confirmed,
    ).toBe(false);

    const strictPassEval: Live1toNMatchEvaluation = {
      ...subThresholdEval,
      matched: true,
      topCandidate: candidate,
      topSimilarity: 0.91,
      topCenteredSimilarity: 0.89,
      topFinalScore: 0.91,
      topEuclideanDistance: 0.42,
      confidencePercent: 91,
      reason: 'OK',
    };
    const f1 = tracker.step(strictPassEval, { biologicalVerified: true, spoofDetected: false });
    expect(f1.streak).toBe(1);
    expect(f1.confirmed).toBe(false);

    const f2 = tracker.step(strictPassEval, { biologicalVerified: true, spoofDetected: false });
    expect(f2.streak).toBe(2);
    expect(f2.confirmed).toBe(false);

    const f3 = tracker.step(strictPassEval, { biologicalVerified: true, spoofDetected: false });
    expect(f3.streak).toBe(3);
    expect(f3.confirmed).toBe(true);
    expect(f3.candidate?.patientId).toBe('pat-fast-1');
  });

  it('triggers fast not_found verdict within <= 3 clear frames / ~1.1s while respecting rescan cooldown and guards', () => {
    const base = {
      candidatesLoaded: true,
      humanWarmupReady: true,
      documentHidden: false,
      isAmbiguous: false,
      isSpoof: false,
      bioVerified: true,
      matchStreak: 0,
      unmatchedClearFrames: 1,
      unmatchedElapsedMs: 250,
      inRescanCooldown: false,
      hasZeroCandidates: false,
    };

    // Frame 1: still scanning
    expect(shouldTriggerFastNotFoundVerdict(base)).toBe(false);

    // Frame 3 (or >= 1100ms): triggers immediate not_found popup
    expect(
      shouldTriggerFastNotFoundVerdict({
        ...base,
        unmatchedClearFrames: 3,
      }),
    ).toBe(true);
    expect(
      shouldTriggerFastNotFoundVerdict({
        ...base,
        bioVerified: false,
        unmatchedClearFrames: 3,
      }),
    ).toBe(true);
    expect(
      shouldTriggerFastNotFoundVerdict({
        ...base,
        unmatchedClearFrames: 2,
        unmatchedElapsedMs: 1100,
      }),
    ).toBe(true);

    // Rescan cooldown / spoof / active streak / ambiguous candidates block not_found trigger
    expect(
      shouldTriggerFastNotFoundVerdict({
        ...base,
        unmatchedClearFrames: 3,
        inRescanCooldown: true,
      }),
    ).toBe(false);
    expect(
      shouldTriggerFastNotFoundVerdict({
        ...base,
        unmatchedClearFrames: 3,
        isSpoof: true,
      }),
    ).toBe(false);
    expect(
      shouldTriggerFastNotFoundVerdict({
        ...base,
        unmatchedClearFrames: 3,
        matchStreak: 1,
      }),
    ).toBe(false);
    expect(
      shouldTriggerFastNotFoundVerdict({
        ...base,
        unmatchedClearFrames: 3,
        isAmbiguous: true,
      }),
    ).toBe(false);

    // Explicit server rejection triggers immediate decisive not_found even with prior streak/ambiguity
    expect(
      shouldTriggerFastNotFoundVerdict({
        ...base,
        unmatchedClearFrames: 0,
        unmatchedElapsedMs: 200,
        matchStreak: 2,
        serverRejectedCount: 1,
      }),
    ).toBe(true);
    expect(
      shouldTriggerFastNotFoundVerdict({
        ...base,
        unmatchedClearFrames: 0,
        unmatchedElapsedMs: 200,
        isAmbiguous: true,
        serverRejectedCount: 1,
      }),
    ).toBe(true);

    // Anti-flicker hard ceiling breaks out cleanly when flickering/ambiguous frames exceed decision window
    expect(
      shouldTriggerFastNotFoundVerdict({
        ...base,
        unmatchedClearFrames: 0,
        unmatchedElapsedMs: 0,
        matchStreak: 1,
        totalEvaluatedFrames: KIOSK_MAX_DECISION_EVAL_FRAMES,
        totalDecisionElapsedMs: KIOSK_MAX_DECISION_WINDOW_MS,
      }),
    ).toBe(true);
    expect(
      shouldTriggerFastNotFoundVerdict({
        ...base,
        unmatchedClearFrames: 0,
        unmatchedElapsedMs: 0,
        isAmbiguous: true,
        totalEvaluatedFrames: KIOSK_MAX_DECISION_EVAL_FRAMES,
        totalDecisionElapsedMs: KIOSK_MAX_DECISION_WINDOW_MS,
      }),
    ).toBe(true);
  });

  it('renders UnregisteredFaceSignupPopup with mascot, Thai headings, buttons, and wires action handlers cleanly', () => {
    const onRegisterNewPatient = vi.fn();
    const onRescan = vi.fn();
    const onCallStaff = vi.fn();

    const element = React.createElement(UnregisteredFaceSignupPopup, {
      scanKey: 1,
      isSpeaking: true,
      onRegisterNewPatient,
      onRescan,
      onCallStaff,
    });
    const html = renderToStaticMarkup(element);

    expect(html).toContain('ไม่พบบัญชีผู้ป่วยในระบบ');
    expect(html).toContain(
      'ใบหน้านี้ยังไม่ได้ลงทะเบียนใช้งานตู้กายภาพบำบัด Dr.Tech.Care กรุณาสมัครบัญชีใหม่ก่อนเริ่มฝึก',
    );
    expect(html).toContain('สมัครบัญชี / ลงทะเบียนคนไข้ใหม่');
    expect(html).toContain('สแกนใหม่อีกครั้ง');
    expect(html).toContain('เรียกเจ้าหน้าที่ช่วยเหลือ');
    expect(html).toContain('data-testid="drtechcare-mascot"');
    expect(html).toContain('data-testid="kiosk-not-found-feedback-panel"');
    expect(EMOJI_RE.test(html)).toBe(false);
    expect(EMOJI_RE.test(KIOSK_NOT_FOUND_VOICE_PROMPT_TH)).toBe(false);

    // Verify rendered React tree action buttons invoke the exact provided callbacks
    const renderedTree = UnregisteredFaceSignupPopup({
      scanKey: 1,
      isSpeaking: false,
      onRegisterNewPatient,
      onRescan,
      onCallStaff,
    });
    const buttons: React.ReactElement<{ onClick?: () => void }>[] = [];
    const walk = (node: unknown): void => {
      if (!node || typeof node !== 'object') return;
      if (Array.isArray(node)) {
        node.forEach(walk);
        return;
      }
      const el = node as React.ReactElement<{ onClick?: () => void; children?: unknown }>;
      if (el.type === 'button' && typeof el.props?.onClick === 'function') {
        buttons.push(el);
      }
      if (el.props?.children) {
        walk(el.props.children);
      }
    };
    walk(renderedTree);

    const textOf = (node: unknown): string => {
      if (typeof node === 'string') return node;
      if (Array.isArray(node)) return node.map(textOf).join('');
      if (node && typeof node === 'object') {
        const el = node as React.ReactElement<{ children?: unknown }>;
        return textOf(el.props?.children);
      }
      return '';
    };

    const signupBtn = buttons.find((b) => textOf(b).includes('สมัครบัญชี / ลงทะเบียนคนไข้ใหม่'));
    const rescanBtn = buttons.find((b) => textOf(b).includes('สแกนใหม่อีกครั้ง'));
    const callStaffBtn = buttons.find((b) => textOf(b).includes('เรียกเจ้าหน้าที่ช่วยเหลือ'));

    expect(signupBtn).toBeDefined();
    expect(rescanBtn).toBeDefined();
    expect(callStaffBtn).toBeDefined();

    signupBtn?.props.onClick?.();
    rescanBtn?.props.onClick?.();
    callStaffBtn?.props.onClick?.();

    expect(onRegisterNewPatient).toHaveBeenCalledTimes(1);
    expect(onRescan).toHaveBeenCalledTimes(1);
    expect(onCallStaff).toHaveBeenCalledTimes(1);
  });

  it('normalizes backlit side-lit face crops cleanly without border window contamination and distinguishes sharp dim faces from real blur', () => {
    const w = 128;
    const h = 128;
    const backlitData = new Uint8ClampedArray(w * h * 4);
    const cx = (w - 1) * 0.5;
    const cy = (h - 1) * 0.5;
    for (let y = 0; y < h; y += 1) {
      for (let x = 0; x < w; x += 1) {
        const idx = (y * w + x) * 4;
        const ellip = ((x - cx) / (w * 0.37)) ** 2 + ((y - cy) / (h * 0.41)) ** 2;
        if (ellip > 1.0) {
          // Blinding sunlit window behind user's head in outer crop corners
          backlitData[idx] = 250;
          backlitData[idx + 1] = 250;
          backlitData[idx + 2] = 250;
          backlitData[idx + 3] = 255;
        } else {
          // Strong side-lit shadow across face (left cheek ~135, right cheek ~52) + fine facial edge detail
          const sideRamp = 135 - ((x - (cx - w * 0.37)) / (w * 0.74)) * 83;
          const texture = Math.sin(x * 0.65) * 9 + Math.cos(y * 0.55) * 9;
          const v = Math.max(15, Math.min(230, Math.round(sideRamp + texture)));
          backlitData[idx] = Math.min(255, v + 8);
          backlitData[idx + 1] = v;
          backlitData[idx + 2] = Math.max(0, v - 10);
          backlitData[idx + 3] = 255;
        }
      }
    }

    const normalized = normalizeImageDataLuminance(
      { width: w, height: h, data: backlitData, colorSpace: 'srgb' } as ImageData,
      128,
    );

    let leftSum = 0;
    let leftCnt = 0;
    let rightSum = 0;
    let rightCnt = 0;
    let coreSum = 0;
    let coreSq = 0;
    let coreCnt = 0;
    for (let y = 0; y < h; y += 1) {
      for (let x = 0; x < w; x += 1) {
        const ellip = ((x - cx) / (w * 0.35)) ** 2 + ((y - cy) / (h * 0.39)) ** 2;
        if (ellip > 1.0) continue;
        const idx = (y * w + x) * 4;
        const yVal =
          0.299 * normalized.data[idx] +
          0.587 * normalized.data[idx + 1] +
          0.114 * normalized.data[idx + 2];
        coreSum += yVal;
        coreSq += yVal * yVal;
        coreCnt += 1;
        if (x < cx) {
          leftSum += yVal;
          leftCnt += 1;
        } else {
          rightSum += yVal;
          rightCnt += 1;
        }
      }
    }
    const coreMean = coreSum / coreCnt;
    const coreStd = Math.sqrt(Math.max(0, coreSq / coreCnt - coreMean * coreMean));
    const balancedSideDiff = Math.abs(leftSum / leftCnt - rightSum / rightCnt);

    expect(coreMean).toBeGreaterThan(112);
    expect(coreMean).toBeLessThan(144);
    // Pure facial texture (originally std ~11 under shadow) is boosted cleanly without window-induced crushing
    expect(coreStd).toBeGreaterThan(22);
    expect(balancedSideDiff).toBeLessThan(35);
    expect(computeImageDataSharpness(normalized)).toBeGreaterThan(8);

    // Contrast-normalized sharpness passes sharp low-contrast shadowed faces while rejecting genuinely blurry frames
    const sharpDim = new Uint8ClampedArray(112 * 112 * 4);
    const flatBlurry = new Uint8ClampedArray(112 * 112 * 4);
    for (let y = 0; y < 112; y += 1) {
      for (let x = 0; x < 112; x += 1) {
        const idx = (y * 112 + x) * 4;
        const sharpVal = 60 + (((x >> 1) + (y >> 1)) % 2 === 0 ? 11 : -11);
        sharpDim[idx] = sharpVal;
        sharpDim[idx + 1] = sharpVal;
        sharpDim[idx + 2] = sharpVal;
        sharpDim[idx + 3] = 255;

        const blurVal = Math.round(90 + Math.sin(x * 0.08) * 14);
        flatBlurry[idx] = blurVal;
        flatBlurry[idx + 1] = blurVal;
        flatBlurry[idx + 2] = blurVal;
        flatBlurry[idx + 3] = 255;
      }
    }
    expect(computeContrastNormalizedSharpness({ width: 112, height: 112, data: sharpDim })).toBeGreaterThanOrEqual(
      MIN_SHARPNESS_LAPLACIAN,
    );
    expect(computeContrastNormalizedSharpness({ width: 112, height: 112, data: flatBlurry })).toBeLessThan(
      MIN_SHARPNESS_LAPLACIAN,
    );
  });

  it('smooths 60FPS bounding boxes & 478-pt iris landmarks without lag and provides actionable Thai readiness cues', () => {
    const baseBox = { left: 200, top: 120, width: 180, height: 220 };
    // Tiny 0.8px MediaPipe micro-jitter is damped smoothly
    const damped = smoothBiometricBox(baseBox, { left: 200.8, top: 120.6, width: 180, height: 220 }, 16);
    expect(damped.left - baseBox.left).toBeGreaterThan(0.05);
    expect(damped.left - baseBox.left).toBeLessThan(0.35);

    // Large sudden movement snaps cleanly without sluggish drag
    const snapped = smoothBiometricBox(baseBox, { left: 360, top: 140, width: 180, height: 220 }, 16);
    expect(snapped.left).toBe(360);

    // 478 landmarks + dual-iris points smooth cleanly
    const lmsA = Array.from({ length: 478 }, () => ({ x: 0.5, y: 0.5, z: 0 }));
    const lmsB = Array.from({ length: 478 }, (_, idx) => ({
      x: 0.5 + (idx >= 468 ? 0.002 : 0.001),
      y: 0.5,
      z: 0,
    }));
    const smoothLms = smoothFacialLandmarks(lmsA, lmsB, 16);
    expect(smoothLms).toHaveLength(478);
    expect(smoothLms[1].x).toBeGreaterThan(0.5);
    expect(smoothLms[1].x).toBeLessThan(0.501);
    expect(smoothLms[468].x).toBeGreaterThan(0.5);
    expect(smoothLms[468].x).toBeLessThan(0.502);

    // Actionable zero-emoji guidance prevents silent (0/3) stalls
    const yawCue = resolveKioskReadinessGuidance({
      isSpoof: false,
      isBioVerified: true,
      yawDeg: 20,
      pitchDeg: 0,
      inspection: null,
      blurryFrameStreak: 0,
      lang: 'th',
    });
    const pitchUpCue = resolveKioskReadinessGuidance({
      isSpoof: false,
      isBioVerified: true,
      yawDeg: 0,
      pitchDeg: -19,
      inspection: null,
      blurryFrameStreak: 0,
      lang: 'th',
    });
    const blurCue = resolveKioskReadinessGuidance({
      isSpoof: false,
      isBioVerified: true,
      yawDeg: 0,
      pitchDeg: 0,
      inspection: { scan_status: 'RETRY', error_code: 'BLURRY' },
      blurryFrameStreak: 2,
      lang: 'th',
    });
    const readyCue = resolveKioskReadinessGuidance({
      isSpoof: false,
      isBioVerified: true,
      yawDeg: 3,
      pitchDeg: -2,
      inspection: { scan_status: 'PASS', error_code: 'NONE' },
      blurryFrameStreak: 0,
      lang: 'th',
    });
    expect(yawCue).toContain('หันหน้าตรง');
    expect(pitchUpCue).toContain('ก้มหน้า');
    expect(blurCue).toContain('หยุดนิ่ง');
    expect(readyCue).toBeNull();
    expect(EMOJI_RE.test(yawCue ?? '')).toBe(false);
    expect(EMOJI_RE.test(pitchUpCue ?? '')).toBe(false);
    expect(EMOJI_RE.test(blurCue ?? '')).toBe(false);
  });

  it('records 20-round Step 0 telemetry percentiles (avg/p50/p95) cleanly without storing PII or face vectors', () => {
    expect(KIOSK_MAX_EXPOSURE_SETTLE_MS).toBeLessThanOrEqual(800);
    expect(KIOSK_DETECT_MAX_LONG_SIDE_PX).toBe(1280);
    expect(KIOSK_OVERLAY_MIN_INTERVAL_MS).toBeGreaterThanOrEqual(30);

    const summary = computeTimingPercentiles(Array.from({ length: 20 }, (_, i) => (i + 1) * 10));
    expect(summary.count).toBe(20);
    expect(summary.avgMs).toBe(105);
    expect(summary.p50Ms).toBe(100);
    expect(summary.p95Ms).toBe(190);
    expect(summary.lastMs).toBe(200);

    const profiler = new KioskBiometricScanProfiler();
    profiler.clearStoredRounds();

    for (let round = 0; round < 20; round += 1) {
      const t0 = 1000 + round * 5000;
      profiler.startSession(t0);
      profiler.recordCameraOpen({
        openDurationMs: 120 + round * 5,
        probeCount: 1,
        usedRememberedDevice: true,
      });
      profiler.recordModelsReady({
        modelWarmupMs: round === 0 ? 420 : 0,
        wasPrewarmed: round > 0,
        faceDelegate: 'GPU',
        humanBackend: 'webgl',
      });
      profiler.recordFirstVideoFrame(t0 + 180);
      profiler.recordExposureSettled(40);
      profiler.markFaceSeen(t0 + 300);
      profiler.recordDetectFrame({
        now: t0 + 320,
        detectMs: 12 + (round % 4),
        roiMs: 0,
        livenessMs: 2.5,
        inputWidth: 480,
        inputHeight: 270,
      });
      profiler.recordOverlayDraw(t0 + 325, 1.8);
      profiler.markFaceReady(t0 + 360);
      profiler.markLivenessVerified(t0 + 520, 'iris_saccade');
      profiler.recordEmbedStep({
        irisMs: 3.1,
        humanEmbedMs: 85 + round * 2,
        localMatchMs: 0.4,
      });
      profiler.recordServerVerify(160 + round * 3, 110);
      profiler.commitRound('matched', t0 + 950 + round * 10, 45);
    }

    const snap = profiler.getSnapshot(110_000);
    expect(snap.roundsCount).toBe(20);
    expect(snap.stats.cameraOpenMs.count).toBe(20);
    expect(snap.stats.cameraOpenMs.p50Ms).toBeGreaterThanOrEqual(160);
    expect(snap.stats.humanEmbedAvgMs.p95Ms).toBeGreaterThan(snap.stats.humanEmbedAvgMs.p50Ms);
    expect(snap.stats.serverVerifyMs.count).toBe(20);
    expect(snap.stats.totalFaceToVerifiedMs.count).toBe(20);
    expect(snap.live.pipelinedOverlapSavedMs).toBe(110);
    expect(snap.live.detectInputLabel).toBe('480x270');
    expect(snap.topBottlenecks.length).toBeGreaterThan(0);

    // Strictly numeric telemetry only: no patient names, HNs, or base64 data URLs
    const serialized = JSON.stringify(snap);
    expect(serialized).not.toContain('data:image');
    expect(serialized).not.toContain('fullName');
    expect(serialized).not.toContain('embedding');
  });

  it('opens remembered camera directly in 1 probe without scanning all 5 cameras, and falls back if unplugged', async () => {
    const origNav = globalThis.navigator;
    const enumerateDevices = vi.fn(async () => [
      { kind: 'videoinput', deviceId: 'cam-1', label: 'Logitech USB HD' },
      { kind: 'videoinput', deviceId: 'cam-2', label: 'USB Camera 2' },
      { kind: 'videoinput', deviceId: 'cam-3', label: 'USB Camera 3' },
      { kind: 'videoinput', deviceId: 'cam-4', label: 'Integrated Webcam' },
      { kind: 'videoinput', deviceId: 'cam-5', label: 'Virtual Camera' },
    ]);
    const makeStream = (id: string) =>
      ({
        getVideoTracks: () => [{ getSettings: () => ({ deviceId: id }), stop: vi.fn() }],
        getTracks: () => [{ stop: vi.fn() }],
      }) as unknown as MediaStream;

    const getUserMedia = vi.fn(async (constraints: MediaStreamConstraints) => {
      const video = constraints.video as MediaTrackConstraints;
      const exactId = (video?.deviceId as { exact?: string } | undefined)?.exact;
      if (exactId === 'cam-unplugged') {
        const err = new Error('Requested device not found');
        err.name = 'NotFoundError';
        throw err;
      }
      return makeStream(exactId ?? 'cam-1');
    });

    Object.defineProperty(globalThis, 'navigator', {
      configurable: true,
      value: {
        mediaDevices: {
          enumerateDevices,
          getUserMedia,
        },
      },
    });

    try {
      // 1. Remembered camera opens directly on attempt 1 with ZERO enumerateDevices blocking calls!
      const direct = await openBestAvailableCameraStream('cam-1', { skipPostEnumerate: true });
      expect(direct.usedRememberedDevice).toBe(true);
      expect(direct.probeCount).toBe(1);
      expect(direct.activeDeviceId).toBe('cam-1');
      expect(enumerateDevices).toHaveBeenCalledTimes(0);

      // 2. If remembered camera was unplugged (NotFoundError), immediately falls back to ranked camera
      const fallback = await openBestAvailableCameraStream('cam-unplugged', {
        skipPostEnumerate: true,
      });
      expect(fallback.usedRememberedDevice).toBe(false);
      expect(fallback.activeDeviceId).toBe('cam-1');
      expect(fallback.probeCount).toBe(2);
      expect(enumerateDevices).toHaveBeenCalledTimes(1);
    } finally {
      Object.defineProperty(globalThis, 'navigator', {
        configurable: true,
        value: origNav,
      });
    }
  });

  it('passes realistic 1024-D ReLU6 frontal self-match when turned centroids are ~0.77 while still rejecting strangers', () => {
    const makeRelu6Vec = (seed: number, dim = 1024): number[] => {
      const raw = new Array<number>(dim);
      for (let i = 0; i < dim; i += 1) {
        const v = Math.sin((i + 1) * seed * 0.13) + 0.55 * Math.cos((i + 7) * seed * 0.29) + 0.35;
        raw[i] = v > 0 ? v : 0;
      }
      return l2Normalize(raw);
    };
    const mixRelu6 = (a: number[], b: number[], alpha: number): number[] =>
      l2Normalize(a.map((v, i) => v * (1 - alpha) + b[i] * alpha));

    const center = makeRelu6Vec(1.7);
    const leftTurn = mixRelu6(center, makeRelu6Vec(9.1), 0.55);
    const rightTurn = mixRelu6(center, makeRelu6Vec(11.3), 0.55);
    const upTurn = mixRelu6(center, makeRelu6Vec(13.7), 0.55);
    const downTurn = mixRelu6(center, makeRelu6Vec(15.9), 0.55);
    // Master is the mean across all 5 poses (distinct from center, matching FaceID120Scanner)
    const master = l2Normalize(
      center.map((_, i) => center[i] * 0.4 + (leftTurn[i] + rightTurn[i] + upTurn[i] + downTurn[i]) * 0.15),
    );
    const candidate = {
      masterEmbedding: master,
      angleEmbeddings: {
        center,
        left: leftTurn,
        right: rightTurn,
        up: upTurn,
        down: downTurn,
      },
      exemplarEmbeddings: [center, leftTurn, rightTurn, upTurn, downTurn],
    };

    // Genuine live frontal probe (slight webcam lighting/pose perturbation around center)
    const genuineProbe = mixRelu6(center, makeRelu6Vec(4.2), 0.12);
    const genuineEval = evaluateTripleGateNeuralCandidate(genuineProbe, candidate, null, 'center');
    expect(genuineEval.passed).toBe(true);
    expect(genuineEval.rawCosine).toBeGreaterThanOrEqual(0.89);
    expect(genuineEval.centeredCosine).toBeGreaterThanOrEqual(0.85);

    // Imposter stranger must still fail strictly (thresholds >= 0.89 raw / >= 0.85 centered untouched!)
    const strangerProbe = makeRelu6Vec(42.5);
    const strangerEval = evaluateTripleGateNeuralCandidate(strangerProbe, candidate, null, 'center');
    expect(strangerEval.passed).toBe(false);

    // Near-miss ReLU6 probe (~0.84 raw, below strict 0.89 ReLU6 threshold) must still fail!
    const nearMissProbe = mixRelu6(center, makeRelu6Vec(31.4), 0.42);
    const nearMissEval = evaluateTripleGateNeuralCandidate(nearMissProbe, candidate, null, 'center');
    expect(nearMissEval.passed).toBe(false);
  });

  it('separates KioskSystemErrorPopup from UnregisteredFaceSignupPopup and enforces zero-vector-download privacy in Kiosk login', () => {
    // 1. System error blocks not_found verdict cleanly
    expect(
      shouldTriggerFastNotFoundVerdict({
        candidatesLoaded: true,
        humanWarmupReady: true,
        documentHidden: false,
        isAmbiguous: false,
        isSpoof: false,
        bioVerified: true,
        matchStreak: 0,
        unmatchedClearFrames: 5,
        unmatchedElapsedMs: 3000,
        inRescanCooldown: false,
        hasZeroCandidates: true,
        serverRejectedCount: 2,
        systemError: true,
      }),
    ).toBe(false);

    expect(
      shouldAutoReturnToStandby({
        hasCallback: true,
        staffModalOpen: false,
        hasVerifiedModal: false,
        isFaceDetected: false,
        scanStage: 'system_error',
        idleElapsedMs: 65_000,
      }),
    ).toBe(true);

    // 2. KioskSystemErrorPopup shows polite Thai system error copy (NEVER "ไม่พบบัญชี" or raw codes/scores/emojis)
    const onRescan = vi.fn();
    const onUsePin = vi.fn();
    const onCallStaff = vi.fn();
    const sysHtml = renderToStaticMarkup(
      React.createElement(KioskSystemErrorPopup, {
        reasonCode: 'TIMEOUT',
        onRescan,
        onUsePin,
        onCallStaff,
      }),
    );
    expect(sysHtml).toContain('ระบบตรวจสอบขัดข้องชั่วคราว กรุณาลองใหม่หรือเรียกเจ้าหน้าที่');
    expect(sysHtml).not.toContain('ไม่พบบัญชี');
    expect(sysHtml).not.toContain('TIMEOUT');
    expect(EMOJI_RE.test(sysHtml)).toBe(false);

    const rateHtml = renderToStaticMarkup(
      React.createElement(KioskSystemErrorPopup, {
        reasonCode: 'RATE_LIMITED',
        onRescan,
        onUsePin,
        onCallStaff,
      }),
    );
    expect(rateHtml).toContain('มีการลองสแกนไม่ผ่านหลายครั้งเกินกำหนด');
    expect(rateHtml).not.toContain('ไม่พบบัญชี');
    expect(rateHtml).not.toContain('RATE_LIMITED');
    expect(EMOJI_RE.test(rateHtml)).toBe(false);

    // 3. KioskBiometricLoginView never downloads other patients' face vectors to the browser
    //    and uses atomic 3-frame server consensus + background ?warm=1 + camera persistence
    const root = process.cwd();
    const kioskSrc = fs.readFileSync(path.join(root, 'src/components/KioskBiometricLoginView.tsx'), 'utf8');
    expect(kioskSrc).toContain('getBiometricLibraryStats');
    expect(kioskSrc).not.toContain('fetchLiveBiometricsFromSupabase');
    expect(kioskSrc).not.toContain('getBiometricIndex');
    expect(kioskSrc).not.toContain('evaluateLiveMatch1toN');
    expect(kioskSrc).toContain('data-testid="staff-biometric-diagnostics"');
    expect(kioskSrc).toContain('data-testid="kiosk-system-error-modal"');
    expect(kioskSrc).toContain('frameVectors');
    expect(kioskSrc).toContain('frameTimestamps');
    expect(kioskSrc).toContain('consensusFramesVerified');
    expect(kioskSrc).toContain("window.localStorage.setItem('drtech_preferred_camera_id', opened.activeDeviceId)");
    // Regression guards for enroll-to-login parity (no single-call false not_found, exposure + open-eye + distinct-frame gating)
    expect(kioskSrc).toContain('unmatchedClearFramesRef.current += 1;');
    expect(kioskSrc).not.toContain('unmatchedClearFramesRef.current += evaluatedBatchFrames');
    expect(kioskSrc).toContain('exposureSettled &&');
    expect(kioskSrc).toContain('lastEmbeddedVideoTime');
    expect(kioskSrc).toContain('excludedPatientIdsRef.current = [];');

    // 4. StaffAssistedEnrollmentModal enforces duplicate-face check and warning UI
    const enrollSrc = fs.readFileSync(path.join(root, 'src/components/StaffAssistedEnrollmentModal.tsx'), 'utf8');
    expect(enrollSrc).toContain('checkDuplicateFace');
    expect(enrollSrc).toContain('DUPLICATE_FACE_ENROLLMENT');
    expect(enrollSrc).toContain('data-testid="duplicate-face-warning-banner"');

    // 5. FaceID120Scanner uses the same remembered camera ID as Kiosk login on multi-camera PCs
    const scannerSrc = fs.readFileSync(path.join(root, 'src/components/FaceID120Scanner.tsx'), 'utf8');
    expect(scannerSrc).toContain("window.localStorage.getItem('drtech_preferred_camera_id')");
    expect(scannerSrc).toContain('selectClinicalCameraDeviceId(storedPreferredCameraId)');
  });

  it('audits all mascot, kiosk login, and biometric matcher files for zero Unicode emojis', () => {
    const root = process.cwd();
    const files = [
      'src/components/DrTechCareMascot.tsx',
      'src/components/KioskBiometricLoginView.tsx',
      'src/components/FaceID120Scanner.tsx',
      'src/components/ClinicalStaffPortalShell.tsx',
      'src/components/StaffAssistedEnrollmentModal.tsx',
      'src/lib/biometricEngine.ts',
      'src/lib/biometricScanProfiler.ts',
      'src/lib/biometricTrackerSmoothing.ts',
      'src/lib/biometricVisionCache.ts',
      'packages/smart-biometric-core/src/vectorMath.ts',
      'packages/api-client/src/index.ts',
      'packages/api-client/src/cloudRouter.ts',
      'packages/api-client/src/supabasePersistence.ts',
      'apps/api/src/app.ts',
      'packages/face-client/src/cameraSelector.ts',
      'src/__tests__/kioskBiometricLogin.test.ts',
    ];
    for (const rel of files) {
      const content = fs.readFileSync(path.join(root, rel), 'utf8');
      expect(EMOJI_RE.test(content), `Unexpected emoji in ${rel}`).toBe(false);
    }
  });
});

