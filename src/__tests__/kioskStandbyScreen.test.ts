import fs from 'node:fs';
import path from 'node:path';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it, vi } from 'vitest';
import {
  KioskStandbyScreen,
  KioskStandbyView,
  SevenSegmentClock,
  createSingleFireStandbyWake,
  formatStandbyClockHHMM,
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
    expect(html).toContain('แตะหน้าจอเพื่อเริ่มต้นใช้งาน');
    expect(html).toContain('ระบบฟื้นฟูกายภาพบำบัดอัจฉริยะ');
    expect(html).toContain('data-testid="kiosk-standby-clock"');
    expect(html).toContain('data-time="20:54"');
    expect(html).toContain('data-testid="kiosk-standby-start-btn"');
    expect(html).toContain('data-testid="standby-ecg-waveform-top"');
    expect(html).toContain('data-testid="standby-ecg-waveform-bottom"');

    const plusMatches = html.match(/data-testid="standby-glow-plus"/g) ?? [];
    expect(plusMatches).toHaveLength(4);

    const digitMatches = html.match(/data-testid="seven-segment-digit"/g) ?? [];
    expect(digitMatches).toHaveLength(4);

    expect(EMOJI_RE.test(html)).toBe(false);
  });

  it('renders localized EN/CN copy and clean 7-segment clock independently', () => {
    const enHtml = renderToStaticMarkup(
      React.createElement(KioskStandbyView, {
        timeText: '09:12',
        lang: 'en',
        onWake: vi.fn(),
      }),
    );
    expect(enHtml).toContain('Tap Screen to Get Started');
    expect(enHtml).toContain('Intelligent Physical Rehabilitation System');

    const clockHtml = renderToStaticMarkup(
      React.createElement(SevenSegmentClock, { timeText: '20:54' }),
    );
    expect(clockHtml).toContain('data-digit="2"');
    expect(clockHtml).toContain('data-digit="0"');
    expect(clockHtml).toContain('data-digit="5"');
    expect(clockHtml).toContain('data-digit="4"');
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

  it('audits all standby-related files for zero Unicode emojis', () => {
    const root = process.cwd();
    const files = [
      'src/components/KioskStandbyScreen.tsx',
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

