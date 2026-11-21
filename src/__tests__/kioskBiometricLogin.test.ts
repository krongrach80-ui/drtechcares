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
  type EnrolledBiometricCandidate,
  type Live1toNMatchEvaluation,
} from '../lib/biometricEngine';
import {
  KIOSK_FAST_MATCH_STREAK,
  KIOSK_FAST_UNREGISTERED_FRAMES,
  KIOSK_FAST_UNREGISTERED_TIMEOUT_MS,
  KIOSK_MAX_DECISION_EVAL_FRAMES,
  KIOSK_MAX_DECISION_WINDOW_MS,
  KIOSK_NO_FACE_GUIDANCE_DELAY_MS,
  KIOSK_NOT_FOUND_VOICE_PROMPT_TH,
  KIOSK_RESCAN_COOLDOWN_MS,
  UnregisteredFaceSignupPopup,
  shouldTriggerFastNotFoundVerdict,
} from '../components/KioskBiometricLoginView';

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

  it('audits all mascot and kiosk login files for zero Unicode emojis', () => {
    const root = process.cwd();
    const files = [
      'src/components/DrTechCareMascot.tsx',
      'src/components/KioskBiometricLoginView.tsx',
      'src/__tests__/kioskBiometricLogin.test.ts',
    ];
    for (const rel of files) {
      const content = fs.readFileSync(path.join(root, rel), 'utf8');
      expect(EMOJI_RE.test(content), `Unexpected emoji in ${rel}`).toBe(false);
    }
  });
});

