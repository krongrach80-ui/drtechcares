import fs from 'node:fs';
import path from 'node:path';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it, vi } from 'vitest';
import {
  INITIAL_STANDBY_HAND_HOLD_STATE,
  KioskBrandIntroOverlay,
  KioskStandbyScreen,
  KioskStandbyView,
  STANDBY_HAND_WAKE_HOLD_MS,
  STANDBY_WELCOME_COPY,
  SevenSegmentClock,
  advanceStandbyHandHold,
  computeEcgSweepProgress,
  computeHeartbeatScale,
  computeStandbyFloatTransform,
  createSingleFireStandbyWake,
  evaluateTwoHandsRaised,
  formatStandbyClockHHMM,
  sampleStandbySilkWaveY,
  smoothVisualProgress,
} from '../components/KioskStandbyScreen';
import {
  KIOSK_IDLE_STANDBY_TIMEOUT_MS,
  KIOSK_NOT_FOUND_IDLE_STANDBY_MS,
  shouldAutoReturnToStandby,
} from '../components/KioskBiometricLoginView';

const EMOJI_RE = /[\p{Extended_Pictographic}\u{FE0F}\u{2600}-\u{27BF}\u{23E9}-\u{23FA}]/u;

describe('Dr.Tech.Care AI Kiosk Standby Screen & Auto-Idle Return', () => {
  it('formats live clock HH:MM with zero padding', () => {
    expect(formatStandbyClockHHMM(new Date(2026, 10, 21, 20, 54, 0))).toBe('20:54');
    expect(formatStandbyClockHHMM(new Date(2026, 10, 21, 8, 5, 9))).toBe('08:05');
    expect(formatStandbyClockHHMM(new Date(2026, 10, 21, 0, 0, 0))).toBe('00:00');
  });

  it('renders the full visual standby layout matching the clinical design without emojis', () => {
    const html = renderToStaticMarkup(
      React.createElement(KioskStandbyScreen, {
        onStart: vi.fn(),
        initialDate: new Date(2026, 10, 21, 20, 54, 0),
      }),
    );

    expect(html).toContain('Dr.Tech.Care');
    expect(html).toContain('AI Kiosk');
    expect(html).toContain('ยกมือ 2 ข้าง ค้างไว้ 3 วินาทีเพื่อเริ่มต้น');
    expect(html).toContain('ระบบฟื้นฟูกายภาพบำบัดผู้สูงอายุอัจฉริยะ');
    expect(html).toContain('data-testid="standby-hand-status-strip"');
    expect(html).toContain('data-testid="standby-left-hand-badge"');
    expect(html).toContain('data-testid="standby-right-hand-badge"');
    expect(html).toContain('data-testid="kiosk-standby-clock"');
    expect(html).toContain('data-time="20:54"');
    expect(html).toContain('data-testid="kiosk-standby-start-btn"');
    expect(html).toContain('data-testid="standby-fluid-wave-canvas"');
    expect(html).toContain('data-testid="standby-ecg-waveform-top"');
    expect(html).toContain('data-testid="standby-ecg-waveform-bottom"');
    expect(html).toContain('data-testid="standby-ecg-comet-top"');
    expect(html).toContain('data-testid="standby-ecg-comet-bottom"');

    const plusMatches = html.match(/data-testid="standby-glow-plus"/g) ?? [];
    expect(plusMatches).toHaveLength(4);

    const digitMatches = html.match(/data-testid="seven-segment-digit"/g) ?? [];
    expect(digitMatches).toHaveLength(4);

    expect(EMOJI_RE.test(html)).toBe(false);
  });

  it('renders localized EN/CN copy, cinematic brand intro overlay, and clean 7-segment clock independently', () => {
    const enHtml = renderToStaticMarkup(
      React.createElement(KioskStandbyView, {
        timeText: '09:12',
        lang: 'en',
        onWake: vi.fn(),
      }),
    );
    expect(enHtml).toContain('Raise Both Hands for 3 Seconds to Start');
    expect(enHtml).toContain('Intelligent Elderly Physical Rehabilitation System');

    const introHtml = renderToStaticMarkup(
      React.createElement(KioskBrandIntroOverlay, {
        lang: 'th',
        isExiting: false,
        canSkip: true,
        onSkip: vi.fn(),
      }),
    );
    expect(introHtml).toContain('data-testid="kiosk-brand-intro-overlay"');
    expect(introHtml).toContain('data-testid="kiosk-intro-logo"');
    expect(introHtml).toContain('data-testid="kiosk-intro-wordmark"');
    expect(introHtml).toContain('Dr.Tech.Care');
    expect(introHtml).toContain(STANDBY_WELCOME_COPY.th.headline);
    expect(introHtml).toContain(STANDBY_WELCOME_COPY.th.subline);
    expect(introHtml).toContain('ยินดีต้อนรับสู่ Dr.Tech.Care นะคะ');
    expect(introHtml).toContain('ขอให้ทุกท่านได้ใช้เราอย่างมีความสุขค่ะ');
    expect(EMOJI_RE.test(introHtml)).toBe(false);

    const clockHtml = renderToStaticMarkup(
      React.createElement(SevenSegmentClock, { timeText: '20:54' }),
    );
    expect(clockHtml).toContain('data-digit="2"');
    expect(clockHtml).toContain('data-digit="0"');
    expect(clockHtml).toContain('data-digit="5"');
    expect(clockHtml).toContain('data-digit="4"');
  });

  it('detects smooth 2-hand raise states and drives a glitch-free 3-second hold timer', () => {
    const makePose = (lwY: number, leY: number, rwY: number, reY: number) => {
      const lms = Array.from({ length: 17 }, () => ({ x: 0.5, y: 0.5, visibility: 0.95 }));
      lms[11] = { x: 0.62, y: 0.48, visibility: 0.96 }; // left shoulder
      lms[12] = { x: 0.38, y: 0.48, visibility: 0.96 }; // right shoulder
      lms[13] = { x: 0.68, y: leY, visibility: 0.95 }; // left elbow
      lms[14] = { x: 0.32, y: reY, visibility: 0.95 }; // right elbow
      lms[15] = { x: 0.7, y: lwY, visibility: 0.95 }; // left wrist
      lms[16] = { x: 0.3, y: rwY, visibility: 0.95 }; // right wrist
      return lms;
    };

    // Resting arms at sides -> neither raised
    const resting = evaluateTwoHandsRaised(makePose(0.78, 0.64, 0.78, 0.64));
    expect(resting.personDetected).toBe(true);
    expect(resting.bothRaised).toBe(false);
    expect(resting.raisedCount).toBe(0);

    // Only left hand raised -> raisedCount === 1, not bothRaised
    const oneHand = evaluateTwoHandsRaised(makePose(0.28, 0.42, 0.78, 0.64));
    expect(oneHand.leftRaised).toBe(true);
    expect(oneHand.rightRaised).toBe(false);
    expect(oneHand.bothRaised).toBe(false);
    expect(oneHand.raisedCount).toBe(1);

    // Both hands raised comfortably above shoulders -> bothRaised === true
    const bothUp = evaluateTwoHandsRaised(makePose(0.25, 0.42, 0.26, 0.43));
    expect(bothUp.leftRaised).toBe(true);
    expect(bothUp.rightRaised).toBe(true);
    expect(bothUp.bothRaised).toBe(true);
    expect(bothUp.raisedCount).toBe(2);

    // Hold for 3 seconds cleanly in 200ms steps -> counts 3 -> 2 -> 1 -> completed
    let hold = { ...INITIAL_STANDBY_HAND_HOLD_STATE };
    expect(STANDBY_HAND_WAKE_HOLD_MS).toBe(3000);
    hold = advanceStandbyHandHold(hold, { bothRaised: true, dtMs: 200 });
    expect(hold.remainingSec).toBe(3);
    expect(hold.completed).toBe(false);

    // Brief 180ms flicker drop stays protected by grace buffer (zero rollback!)
    const afterFlicker = advanceStandbyHandHold(hold, { bothRaised: false, dtMs: 180 });
    expect(afterFlicker.holdMs).toBe(hold.holdMs);

    // Continue holding until full 3.0s completion
    let cur = afterFlicker;
    for (let i = 0; i < 14; i++) {
      cur = advanceStandbyHandHold(cur, { bothRaised: true, dtMs: 200 });
    }
    expect(cur.completed).toBe(true);
    expect(cur.progress).toBe(1);
    expect(cur.remainingSec).toBe(0);

    // Lowering hands past grace period smoothly decays progress instead of snapping to 0
    const mid = advanceStandbyHandHold(
      { holdMs: 1800, missingMs: 0, progress: 0.6, remainingSec: 2, completed: false },
      { bothRaised: false, dtMs: 200 },
    );
    const decaying = advanceStandbyHandHold(mid, { bothRaised: false, dtMs: 240 });
    expect(decaying.holdMs).toBeLessThan(1800);
    expect(decaying.holdMs).toBeGreaterThan(1200);

    // Smooth visual lerp approaches target monotonically
    const lerped = smoothVisualProgress(0, 1, 16.6);
    expect(lerped).toBeGreaterThan(0.1);
    expect(lerped).toBeLessThan(0.4);
  });

  it('triggers onStart once cleanly when CTA button or screen container is tapped', () => {
    const onStartCta = vi.fn();
    const wakeCta = createSingleFireStandbyWake(onStartCta);
    const tree = KioskStandbyView({
      timeText: '20:54',
      lang: 'th',
      onWake: wakeCta,
    });

    const buttons: React.ReactElement<{ onClick?: (e: { stopPropagation: () => void }) => void }>[] =
      [];
    const walk = (node: unknown): void => {
      if (!node || typeof node !== 'object') return;
      if (Array.isArray(node)) {
        node.forEach(walk);
        return;
      }
      const el = node as React.ReactElement<{
        onClick?: (e: { stopPropagation: () => void }) => void;
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

    expect(buttons).toHaveLength(1);
    const stopPropagation = vi.fn();
    buttons[0].props.onClick?.({ stopPropagation });
    // Even if outer div handler also invoked, single-fire guard prevents duplicate wake
    (tree.props as { onClick?: () => void }).onClick?.();

    expect(stopPropagation).toHaveBeenCalledTimes(1);
    expect(onStartCta).toHaveBeenCalledTimes(1);

    const onStartScreen = vi.fn();
    const wakeScreen = createSingleFireStandbyWake(onStartScreen);
    const tree2 = KioskStandbyView({
      timeText: '20:54',
      lang: 'th',
      onWake: wakeScreen,
    });
    (tree2.props as { onClick?: () => void }).onClick?.();
    expect(onStartScreen).toHaveBeenCalledTimes(1);
  });

  it('enforces safe idle auto-return thresholds without sleeping mid-scan', () => {
    expect(KIOSK_IDLE_STANDBY_TIMEOUT_MS).toBe(45000);
    expect(KIOSK_NOT_FOUND_IDLE_STANDBY_MS).toBe(60000);

    // Active face in camera -> stay awake even after timeout
    expect(
      shouldAutoReturnToStandby({
        hasCallback: true,
        staffModalOpen: false,
        hasVerifiedModal: false,
        isFaceDetected: true,
        scanStage: 'scanning',
        idleElapsedMs: 120000,
      }),
    ).toBe(false);

    // Staff modal or verified welcome modal open -> stay awake
    expect(
      shouldAutoReturnToStandby({
        hasCallback: true,
        staffModalOpen: true,
        hasVerifiedModal: false,
        isFaceDetected: false,
        scanStage: 'scanning',
        idleElapsedMs: 120000,
      }),
    ).toBe(false);
    expect(
      shouldAutoReturnToStandby({
        hasCallback: true,
        staffModalOpen: false,
        hasVerifiedModal: true,
        isFaceDetected: false,
        scanStage: 'scanning',
        idleElapsedMs: 120000,
      }),
    ).toBe(false);

    // Below 45s idle with no face -> stay awake
    expect(
      shouldAutoReturnToStandby({
        hasCallback: true,
        staffModalOpen: false,
        hasVerifiedModal: false,
        isFaceDetected: false,
        scanStage: 'scanning',
        idleElapsedMs: 44999,
      }),
    ).toBe(false);

    // At 45s idle with no face -> return to standby
    expect(
      shouldAutoReturnToStandby({
        hasCallback: true,
        staffModalOpen: false,
        hasVerifiedModal: false,
        isFaceDetected: false,
        scanStage: 'scanning',
        idleElapsedMs: 45000,
      }),
    ).toBe(true);

    // On unregistered not_found modal -> wait full 60s before auto-return
    expect(
      shouldAutoReturnToStandby({
        hasCallback: true,
        staffModalOpen: false,
        hasVerifiedModal: false,
        isFaceDetected: false,
        scanStage: 'not_found',
        idleElapsedMs: 45000,
      }),
    ).toBe(false);
    expect(
      shouldAutoReturnToStandby({
        hasCallback: true,
        staffModalOpen: false,
        hasVerifiedModal: false,
        isFaceDetected: false,
        scanStage: 'not_found',
        idleElapsedMs: 60000,
      }),
    ).toBe(true);
  });

  it('computes smooth bounded 60FPS fluid wave, float, heartbeat, and ECG sweep kinematics', () => {
    const y0 = sampleStandbySilkWaveY(0.5, 4, 16, 0, 320, 180, false);
    const y1 = sampleStandbySilkWaveY(0.5, 4, 16, 1.25, 320, 180, false);
    const y2 = sampleStandbySilkWaveY(0.5, 4, 16, 1.25, 320, 180, true);
    expect(Number.isFinite(y0)).toBe(true);
    expect(Number.isFinite(y1)).toBe(true);
    expect(y0).not.toBe(y1);
    expect(y1).not.toBe(y2);
    expect(Math.abs(y0 - 320)).toBeLessThan(180);

    const f0 = computeStandbyFloatTransform(0, {
      ampX: 8,
      ampY: 12,
      ampRotDeg: 5,
      ampScale: 0.05,
      freqX: 0.4,
      freqY: 0.55,
      phase: 1,
    });
    const f1 = computeStandbyFloatTransform(1.4, {
      ampX: 8,
      ampY: 12,
      ampRotDeg: 5,
      ampScale: 0.05,
      freqX: 0.4,
      freqY: 0.55,
      phase: 1,
    });
    expect(Math.abs(f0.x)).toBeLessThanOrEqual(8);
    expect(Math.abs(f0.y)).toBeLessThanOrEqual(12);
    expect(Math.abs(f0.rotDeg)).toBeLessThanOrEqual(5);
    expect(f0.scale).toBeGreaterThanOrEqual(0.95);
    expect(f0.scale).toBeLessThanOrEqual(1.05);
    expect(f0.opacity).toBeGreaterThanOrEqual(0.82);
    expect(f0.opacity).toBeLessThanOrEqual(1);
    expect(f0.y).not.toBe(f1.y);

    const hbBase = computeHeartbeatScale(0, 2.4);
    const hbPeak = computeHeartbeatScale(2.4 * 0.14, 2.4);
    expect(hbBase).toBeGreaterThanOrEqual(1);
    expect(hbPeak).toBeGreaterThan(hbBase);
    expect(hbPeak).toBeLessThanOrEqual(1.08);

    const ecgA = computeEcgSweepProgress(0.5, 4.0, 0.25);
    const ecgB = computeEcgSweepProgress(4.5, 4.0, 0.25);
    expect(ecgA).toBeGreaterThanOrEqual(0);
    expect(ecgA).toBeLessThan(1);
    expect(ecgA).toBeCloseTo(ecgB, 6);
  });

  it('keeps standby animations active even under Windows prefers-reduced-motion: reduce', () => {
    const css = fs.readFileSync(path.join(process.cwd(), 'src/index.css'), 'utf8');
    const reducedMotionStart = css.indexOf('@media (prefers-reduced-motion: reduce)');
    expect(reducedMotionStart).toBeGreaterThan(-1);
    const reducedMotionSection = css.slice(reducedMotionStart);
    expect(reducedMotionSection).not.toContain('.animate-standby-');
  });

  it('audits all standby-related files for zero Unicode emojis', () => {
    const root = process.cwd();
    const files = [
      'src/components/KioskStandbyScreen.tsx',
      'src/utils/standbyHandWakeHelpers.ts',
      'src/components/KioskBiometricLoginView.tsx',
      'src/App.tsx',
      'src/index.css',
      'src/__tests__/kioskStandbyScreen.test.ts',
    ];
    for (const rel of files) {
      const content = fs.readFileSync(path.join(root, rel), 'utf8');
      expect(EMOJI_RE.test(content), `Unexpected emoji in ${rel}`).toBe(false);
    }
  });
});

