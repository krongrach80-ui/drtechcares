import fs from 'node:fs';
import path from 'node:path';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it, vi } from 'vitest';
import { DrTechCaresApiClient, type UserAccount } from '@drtechcares/api-client';
import { ClinicalI18nProvider } from '@drtechcares/clinical-i18n';
import {
  KINETIC_AUTH_HOLD_DELAY_MS,
  KINETIC_SUCCESS_WORD,
  KINETIC_TYPEWRITER_START_DELAY_MS,
  KINETIC_TYPEWRITER_STEP_MS,
  KineticAuthStatusBadge,
  KineticEmblemSvg,
  resolveKineticTypewriterText,
} from '../components/KineticAuthStatusBadge';
import {
  StaffLoginModalOrPage,
  executeStaffAuthSubmission,
  type StaffLoginAnimStatus,
} from '../StaffLoginModalOrPage';

const EMOJI_RE = /[\p{Extended_Pictographic}\u{FE0F}\u{2600}-\u{27BF}\u{23E9}-\u{23FA}]/u;

describe('4-Stage Kinetic SVG Auth Badge & Staff/Kiosk Login Feedback Choreography', () => {
  it('defines all 4-stage kinetic SVG keyframes, error-shake, and reduced-motion fallbacks in src/index.css', () => {
    const css = fs.readFileSync(path.resolve(process.cwd(), 'src', 'index.css'), 'utf8');
    for (const token of [
      '@keyframes kineticDotBounceIn',
      '@keyframes kineticCheckDraw',
      '@keyframes kineticCrossDraw',
      '@keyframes kineticTipDotPop',
      '@keyframes kineticSparkBurst',
      '@keyframes kineticRingSweep',
      '@keyframes kineticOrbitSweep',
      '@keyframes errorShake',
      '.animate-kinetic-dot-bounce',
      '.animate-kinetic-check',
      '.animate-kinetic-cross-1',
      '.animate-kinetic-cross-2',
      '.animate-kinetic-tip-dot',
      '.animate-kinetic-tip-spark',
      '.animate-kinetic-ring',
      '.animate-kinetic-orbit',
      '.animate-kinetic-closure-spark',
      '.animate-error-shake',
      'stroke-dashoffset: 0 !important',
    ]) {
      expect(css).toContain(token);
    }
    expect(EMOJI_RE.test(css)).toBe(false);
  });

  it('transitions checking -> success, holds 1400ms for ring closure + typewriter "Success" (750ms delay, 55ms/char)', async () => {
    vi.useFakeTimers();
    try {
      const mockAdmin: UserAccount = {
        id: 'user-admin-1',
        username: 'admin',
        role: 'super_admin',
        fullName: 'ผอ.สมชาย แอดมิน',
        phone: '0810000001',
        active: true,
        createdAt: '2026-01-01T00:00:00.000Z',
      };

      const statuses: StaffLoginAnimStatus[] = [];
      const loadingStates: boolean[] = [];
      const onSuccess = vi.fn();
      const setError = vi.fn();

      const status = await executeStaffAuthSubmission({
        runRequest: async () => mockAdmin,
        setAnimStatus: (next) => statuses.push(next),
        setLoading: (busy) => loadingStates.push(busy),
        setError,
        onSuccess,
        scheduleTimer: (fn, ms) => setTimeout(fn, ms) as unknown as number,
      });

      expect(status).toBe('success');
      expect(statuses).toEqual(['checking', 'success']);
      expect(loadingStates).toEqual([true, false]);
      expect(onSuccess).not.toHaveBeenCalled();

      expect(KINETIC_TYPEWRITER_START_DELAY_MS).toBe(750);
      expect(KINETIC_TYPEWRITER_STEP_MS).toBe(55);
      expect(KINETIC_AUTH_HOLD_DELAY_MS).toBe(1400);
      expect(resolveKineticTypewriterText(0)).toBe('');
      expect(resolveKineticTypewriterText(1)).toBe('S');
      expect(resolveKineticTypewriterText(2)).toBe('Su');
      expect(resolveKineticTypewriterText(3)).toBe('Suc');
      expect(resolveKineticTypewriterText(4)).toBe('Succ');
      expect(resolveKineticTypewriterText(5)).toBe('Succe');
      expect(resolveKineticTypewriterText(6)).toBe('Succes');
      expect(resolveKineticTypewriterText(7)).toBe(KINETIC_SUCCESS_WORD);

      vi.advanceTimersByTime(1399);
      expect(onSuccess).not.toHaveBeenCalled();
      vi.advanceTimersByTime(1);
      expect(onSuccess).toHaveBeenCalledWith(mockAdmin);
    } finally {
      vi.useRealTimers();
    }
  });

  it('transitions checking -> not_found when username/password is invalid and never calls onSuccess', async () => {
    const statuses: StaffLoginAnimStatus[] = [];
    const onSuccess = vi.fn();

    const status = await executeStaffAuthSubmission({
      runRequest: async () => {
        throw new Error('Invalid credentials');
      },
      setAnimStatus: (next) => statuses.push(next),
      setLoading: () => {},
      setError: () => {},
      onSuccess,
      scheduleTimer: (fn, ms) => setTimeout(fn, ms) as unknown as number,
    });

    expect(status).toBe('not_found');
    expect(statuses).toEqual(['checking', 'not_found']);
    expect(onSuccess).not.toHaveBeenCalled();
  });

  it('renders KineticAuthStatusBadge status="success" (dot bounce, M32 52 L45 65 L69 37 checkmark, orbit r=38 ring, micro-sparks, typewriter)', () => {
    const successBadgeHtml = renderToStaticMarkup(
      React.createElement(KineticAuthStatusBadge, {
        status: 'success',
        testId: 'test-kinetic-success',
      }),
    );
    expect(successBadgeHtml).toContain('test-kinetic-success');
    expect(successBadgeHtml).toContain('viewBox="0 0 100 100"');
    expect(successBadgeHtml).toContain('animate-kinetic-dot-bounce');
    expect(successBadgeHtml).toContain('d="M32 52 L45 65 L69 37"');
    expect(successBadgeHtml).toContain('animate-kinetic-check');
    expect(successBadgeHtml).toContain('cx="69" cy="29" r="4.5"');
    expect(successBadgeHtml).toContain('animate-kinetic-tip-spark');
    expect(successBadgeHtml).toContain('r="38"');
    expect(successBadgeHtml).toContain('transform="rotate(-45 50 50)"');
    expect(successBadgeHtml).toContain('animate-kinetic-ring');
    expect(successBadgeHtml).toContain('animate-kinetic-orbit');
    expect(successBadgeHtml).toContain('animate-kinetic-closure-spark');
    expect(successBadgeHtml).toContain('kinetic-success-typewriter');
    expect(successBadgeHtml).toContain('aria-label="Success"');
    expect(successBadgeHtml).toContain('เข้าสู่ระบบสำเร็จ กำลังเปิดหน้าจัดการคลินิก...');
    expect(EMOJI_RE.test(successBadgeHtml)).toBe(false);
  });

  it('renders KineticAuthStatusBadge status="not_found" (red dot, M36 36 L64 64 / M64 36 L36 64 cross, orbit ring, shake, ไม่พบบัญชี, ไม่มีบัญชี button)', () => {
    const notFoundHtml = renderToStaticMarkup(
      React.createElement(KineticAuthStatusBadge, {
        status: 'not_found',
        testId: 'test-kinetic-not-found',
        secondaryLabel: 'สมัครสมาชิก (Admin / PT)',
        onNoAccountClick: () => {},
        onSecondaryClick: () => {},
      }),
    );
    expect(notFoundHtml).toContain('test-kinetic-not-found');
    expect(notFoundHtml).toContain('animate-error-shake');
    expect(notFoundHtml).toContain('animate-kinetic-dot-bounce');
    expect(notFoundHtml).toContain('d="M36 36 L64 64"');
    expect(notFoundHtml).toContain('d="M64 36 L36 64"');
    expect(notFoundHtml).toContain('animate-kinetic-cross-1');
    expect(notFoundHtml).toContain('animate-kinetic-cross-2');
    expect(notFoundHtml).toContain('animate-kinetic-ring');
    expect(notFoundHtml).toContain('animate-kinetic-orbit');
    expect(notFoundHtml).toContain('ไม่พบบัญชี');
    expect(notFoundHtml).toContain('ไม่พบชื่อผู้ใช้หรือรหัสผ่านนี้ในระบบ');
    expect(notFoundHtml).toContain('ไม่มีบัญชี');
    expect(notFoundHtml).toContain('สมัครสมาชิก (Admin / PT)');
    expect(EMOJI_RE.test(notFoundHtml)).toBe(false);
  });

  it('wires KineticAuthStatusBadge cleanly inside StaffLoginModalOrPage across checking, success, and not_found states with zero emojis', () => {
    const apiClient = new DrTechCaresApiClient();

    const checkingHtml = renderToStaticMarkup(
      React.createElement(
        ClinicalI18nProvider,
        null,
        React.createElement(StaffLoginModalOrPage, {
          apiClient,
          onSuccess: () => {},
          initialLoginAnimStatus: 'checking',
        }),
      ),
    );
    expect(checkingHtml).toContain('กำลังตรวจสอบบัญชี...');
    expect(checkingHtml).toContain('animate-spin');
    expect(EMOJI_RE.test(checkingHtml)).toBe(false);

    const successModalHtml = renderToStaticMarkup(
      React.createElement(
        ClinicalI18nProvider,
        null,
        React.createElement(StaffLoginModalOrPage, {
          apiClient,
          onSuccess: () => {},
          initialLoginAnimStatus: 'success',
        }),
      ),
    );
    expect(successModalHtml).toContain('staff-login-success-panel');
    expect(successModalHtml).toContain('animate-kinetic-check');
    expect(successModalHtml).toContain('animate-kinetic-orbit');
    expect(EMOJI_RE.test(successModalHtml)).toBe(false);

    const notFoundModalHtml = renderToStaticMarkup(
      React.createElement(
        ClinicalI18nProvider,
        null,
        React.createElement(StaffLoginModalOrPage, {
          apiClient,
          onSuccess: () => {},
          initialLoginAnimStatus: 'not_found',
        }),
      ),
    );
    expect(notFoundModalHtml).toContain('staff-login-not-found-panel');
    expect(notFoundModalHtml).toContain('animate-error-shake');
    expect(notFoundModalHtml).toContain('ไม่พบบัญชี');
    expect(notFoundModalHtml).toContain('ไม่มีบัญชี');
    expect(EMOJI_RE.test(notFoundModalHtml)).toBe(false);

    const emblemMini = renderToStaticMarkup(
      React.createElement(KineticEmblemSvg, { status: 'success', className: 'w-6 h-6' }),
    );
    expect(emblemMini).toContain('animate-kinetic-orbit');

    for (const relPath of [
      'src/StaffLoginModalOrPage.tsx',
      'src/components/KineticAuthStatusBadge.tsx',
      'src/components/KioskBiometricLoginView.tsx',
      'src/index.css',
    ]) {
      const content = fs.readFileSync(path.resolve(process.cwd(), relPath), 'utf8');
      expect(EMOJI_RE.test(content)).toBe(false);
    }
  });
});
