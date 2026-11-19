import fs from 'node:fs';
import path from 'node:path';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it, vi } from 'vitest';
import { DrTechCaresApiClient, type UserAccount } from '@drtechcares/api-client';
import { ClinicalI18nProvider } from '@drtechcares/clinical-i18n';
import {
  AUTH_SUCCESS_REDIRECT_DELAY_MS,
  AUTH_SUCCESS_WORD,
  AUTH_TYPEWRITER_STEP_MS,
  AuthNotFoundFeedbackPanel,
  AuthSuccessFeedbackPanel,
  resolveSuccessTypewriterText,
} from '../components/AuthStatusAnimationPanel';
import {
  StaffLoginModalOrPage,
  executeStaffAuthSubmission,
  type StaffLoginAnimStatus,
} from '../StaffLoginModalOrPage';

const EMOJI_RE = /[\p{Extended_Pictographic}\u{FE0F}\u{2600}-\u{27BF}\u{23E9}-\u{23FA}]/u;

describe('Staff & Kiosk Login Feedback Animations (Circle Checkmark Success vs Circle Cross ไม่พบบัญชี)', () => {
  it('defines SVG circle/check/cross keyframes and reduced-motion fallbacks in src/index.css', () => {
    const css = fs.readFileSync(path.resolve(process.cwd(), 'src', 'index.css'), 'utf8');
    for (const token of [
      '@keyframes svgCircleDraw',
      '@keyframes svgCheckDraw',
      '@keyframes svgCrossDraw',
      '@keyframes statusCirclePop',
      '.animate-circle-pop',
      '.animate-svg-circle',
      '.animate-svg-check',
      '.animate-svg-cross-1',
      '.animate-svg-cross-2',
      'stroke-dashoffset: 0 !important',
    ]) {
      expect(css).toContain(token);
    }
    expect(EMOJI_RE.test(css)).toBe(false);
  });

  it('transitions checking -> success on valid staff login, waits 1150ms before completing login, and types "Success"', async () => {
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

      // Letter-by-letter typewriter progression across 65ms ticks
      expect(AUTH_TYPEWRITER_STEP_MS).toBe(65);
      expect(resolveSuccessTypewriterText(1)).toBe('S');
      expect(resolveSuccessTypewriterText(2)).toBe('Su');
      expect(resolveSuccessTypewriterText(3)).toBe('Suc');
      expect(resolveSuccessTypewriterText(4)).toBe('Succ');
      expect(resolveSuccessTypewriterText(5)).toBe('Succe');
      expect(resolveSuccessTypewriterText(6)).toBe('Succes');
      expect(resolveSuccessTypewriterText(7)).toBe(AUTH_SUCCESS_WORD);

      vi.advanceTimersByTime(AUTH_SUCCESS_REDIRECT_DELAY_MS - 1);
      expect(onSuccess).not.toHaveBeenCalled();
      vi.advanceTimersByTime(1);
      expect(onSuccess).toHaveBeenCalledWith(mockAdmin);
    } finally {
      vi.useRealTimers();
    }
  });

  it('transitions checking -> not_found when username/password does not exist and never calls onSuccess', async () => {
    const statuses: StaffLoginAnimStatus[] = [];
    const onSuccess = vi.fn();
    const setError = vi.fn();

    const status = await executeStaffAuthSubmission({
      runRequest: async () => {
        throw new Error('ชื่อผู้ใช้หรือรหัสผ่านไม่ถูกต้อง');
      },
      setAnimStatus: (next) => statuses.push(next),
      setLoading: () => {},
      setError,
      onSuccess,
      scheduleTimer: (fn, ms) => window.setTimeout(fn, ms) as unknown as number,
    });

    expect(status).toBe('not_found');
    expect(statuses).toEqual(['checking', 'not_found']);
    expect(onSuccess).not.toHaveBeenCalled();
  });

  it('renders StaffLoginModalOrPage checking, success (SVG circle checkmark + Success), and not_found (SVG cross + ไม่พบบัญชี + ไม่มีบัญชี) states', () => {
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

    const successHtml = renderToStaticMarkup(
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
    expect(successHtml).toContain('staff-login-success-panel');
    expect(successHtml).toContain('animate-svg-circle');
    expect(successHtml).toContain('animate-svg-check');
    expect(successHtml).toContain('aria-label="Success"');
    expect(successHtml).toContain('เข้าสู่ระบบสำเร็จ กำลังเปิดหน้าจัดการคลินิก...');
    expect(EMOJI_RE.test(successHtml)).toBe(false);

    const notFoundHtml = renderToStaticMarkup(
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
    expect(notFoundHtml).toContain('staff-login-not-found-panel');
    expect(notFoundHtml).toContain('animate-svg-circle');
    expect(notFoundHtml).toContain('animate-svg-cross-1');
    expect(notFoundHtml).toContain('animate-svg-cross-2');
    expect(notFoundHtml).toContain('ไม่พบบัญชี');
    expect(notFoundHtml).toContain('ไม่พบชื่อผู้ใช้หรือรหัสผ่านนี้ในระบบคลินิก');
    expect(notFoundHtml).toContain('ไม่มีบัญชี');
    expect(notFoundHtml).toContain('สมัครสมาชิก (Admin / PT)');
    expect(EMOJI_RE.test(notFoundHtml)).toBe(false);
  });

  it('mirrors compact animated checkmark Success and crimson cross ไม่พบบัญชี panels cleanly with zero emojis', () => {
    const kioskOkHtml = renderToStaticMarkup(
      React.createElement(AuthSuccessFeedbackPanel, {
        compact: true,
        testId: 'kiosk-face-success-panel',
        subtitle: 'คุณสมชาย ทดสอบ • 97%',
      }),
    );
    expect(kioskOkHtml).toContain('kiosk-face-success-panel');
    expect(kioskOkHtml).toContain('animate-svg-circle');
    expect(kioskOkHtml).toContain('animate-svg-check');
    expect(kioskOkHtml).toContain('aria-label="Success"');
    expect(EMOJI_RE.test(kioskOkHtml)).toBe(false);

    const kioskNotFoundHtml = renderToStaticMarkup(
      React.createElement(AuthNotFoundFeedbackPanel, {
        compact: true,
        testId: 'kiosk-not-found-feedback-panel',
        heading: 'ไม่พบบัญชี',
        noAccountLabel: 'ไม่มีบัญชี',
      }),
    );
    expect(kioskNotFoundHtml).toContain('kiosk-not-found-feedback-panel');
    expect(kioskNotFoundHtml).toContain('animate-svg-cross-1');
    expect(kioskNotFoundHtml).toContain('animate-svg-cross-2');
    expect(kioskNotFoundHtml).toContain('ไม่พบบัญชี');
    expect(kioskNotFoundHtml).toContain('ไม่มีบัญชี');
    expect(EMOJI_RE.test(kioskNotFoundHtml)).toBe(false);

    for (const relPath of [
      'src/StaffLoginModalOrPage.tsx',
      'src/components/AuthStatusAnimationPanel.tsx',
      'src/components/KioskBiometricLoginView.tsx',
      'src/index.css',
    ]) {
      const content = fs.readFileSync(path.resolve(process.cwd(), relPath), 'utf8');
      expect(EMOJI_RE.test(content)).toBe(false);
    }
  });
});
