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
  KINETIC_TOTAL_DURATION_MS,
  KINETIC_TYPEWRITER_START_DELAY_MS,
  KINETIC_TYPEWRITER_STEP_MS,
  KIOSK_AUTH_HOLD_DELAY_MS,
  KineticAuthStatusBadge,
  KineticEmblemSvg,
  computeKineticAuthFrame,
  getKineticTypewriterText,
} from '../components/KineticAuthStatusBadge';
import {
  StaffLoginModalOrPage,
  executeStaffAuthSubmission,
  type StaffLoginAnimStatus,
} from '../StaffLoginModalOrPage';

const EMOJI_RE = /[\p{Extended_Pictographic}\u{FE0F}\u{2600}-\u{27BF}\u{23E9}-\u{23FA}]/u;

describe('Frame-Accurate 60FPS Kinetic SVG Auth Badge & Staff/Kiosk Login Choreography', () => {
  it('computes all 5 rAF choreography phases cleanly with two-tone contrast (success & not_found)', () => {
    expect(KINETIC_TOTAL_DURATION_MS).toBe(2350);
    expect(KINETIC_AUTH_HOLD_DELAY_MS).toBe(2800);
    expect(KIOSK_AUTH_HOLD_DELAY_MS).toBe(2400);
    expect(KINETIC_TYPEWRITER_START_DELAY_MS).toBe(1450);
    expect(KINETIC_TYPEWRITER_STEP_MS).toBe(55);
    expect(KINETIC_SUCCESS_WORD).toBe('Success');

    // Phase 1 (0 - 450ms): hop-up from (50, 56) -> (42, 35) then straight drop to (50, 50)
    const f0 = computeKineticAuthFrame(0, 'success');
    expect(f0.phase).toBe(1);
    expect(f0.innerColor).toBe('#10b981');
    expect(f0.ringColor).toBe('#0ea5e9');
    expect(f0.innerOpacity).toBe(0);
    expect(f0.dotX).toBeCloseTo(50, 1);
    expect(f0.dotY).toBeCloseTo(56, 1);

    const fPeak = computeKineticAuthFrame(240, 'success');
    expect(fPeak.phase).toBe(1);
    expect(fPeak.dotX).toBeCloseTo(42, 1);
    expect(fPeak.dotY).toBeCloseTo(35, 1);
    expect(fPeak.hopTailOpacity).toBeGreaterThan(0);

    // Phase 2 (430 - 680ms): checkmark draws smoothly & dot lifts above tip toward (68, 26)
    const fImpact = computeKineticAuthFrame(430, 'success');
    expect(fImpact.phase).toBe(2);
    expect(fImpact.sparks.length).toBeGreaterThanOrEqual(1);

    const fDrawMid = computeKineticAuthFrame(540, 'success');
    expect(fDrawMid.innerOpacity).toBe(1);
    expect(fDrawMid.checkDashOffset).toBeLessThan(100);
    expect(fDrawMid.checkDashOffset).toBeGreaterThanOrEqual(0);

    // Phase 3 (680 - 1020ms): hover bounce around (68, 25) + 2 tip micro-sparks
    const fHover = computeKineticAuthFrame(850, 'success');
    expect(fHover.phase).toBe(3);
    expect(fHover.checkDashOffset).toBe(0);
    expect(fHover.dotX).toBe(68);
    expect(fHover.dotY).toBeCloseTo(24, 0.6);
    expect(fHover.sparks.length).toBe(2);

    // Phase 4 (1020 - 1920ms): leading dot dives to -42deg & sweeps 360deg clockwise pulling r=34 ring
    const fSweepStart = computeKineticAuthFrame(1140, 'success');
    expect(fSweepStart.phase).toBe(4);
    expect(Math.hypot(fSweepStart.dotX - 50, fSweepStart.dotY - 50)).toBeCloseTo(34, 1);
    expect(fSweepStart.sweepProgress).toBeCloseTo(0, 3);

    const fSweepMid = computeKineticAuthFrame(1530, 'success');
    expect(fSweepMid.phase).toBe(4);
    expect(Math.hypot(fSweepMid.dotX - 50, fSweepMid.dotY - 50)).toBeCloseTo(34, 1);
    expect(fSweepMid.sweepProgress).toBeGreaterThan(0.4);
    expect(fSweepMid.sweepProgress).toBeLessThan(0.6);
    expect(fSweepMid.ringDashOffset).toBeGreaterThan(35);
    expect(fSweepMid.ringDashOffset).toBeLessThan(65);
    expect(fSweepMid.typedText.length).toBeGreaterThanOrEqual(2);

    // Phase 5 (1920 - 2350ms): ring closes clockwise 100%, dot shrinks & merges, 3 celebration sparks pop, "Success" done
    const fCloseStart = computeKineticAuthFrame(1920, 'success');
    expect(fCloseStart.phase).toBe(5);
    expect(fCloseStart.sweepProgress).toBe(1);
    expect(fCloseStart.ringDashOffset).toBe(0);
    expect(fCloseStart.sparks.length).toBe(3);
    expect(fCloseStart.typedText).toBe('Success');

    const fDone = computeKineticAuthFrame(2350, 'success');
    expect(fDone.phase).toBe(5);
    expect(fDone.dotVisible).toBe(false);
    expect(fDone.ringDashOffset).toBe(0);
    expect(fDone.typedText).toBe('Success');

    // Account Not Found two-tone palette + subtle impact shake
    const fErr = computeKineticAuthFrame(540, 'not_found');
    expect(fErr.innerColor).toBe('#e11d48');
    expect(fErr.ringColor).toBe('#ef4444');
    expect(fErr.crossDashOffset1).toBeLessThan(100);
    expect(Math.abs(fErr.shakeX)).toBeGreaterThan(0);
    expect(fErr.typedText).toBe('');
  });

  it('reveals visible typewriter "Success" from 1450ms at 55ms per character finishing by 1835ms', () => {
    expect(getKineticTypewriterText(0).typedText).toBe('');
    expect(getKineticTypewriterText(1449).typedText).toBe('');
    expect(getKineticTypewriterText(1450).typedText).toBe('S');
    expect(getKineticTypewriterText(1505).typedText).toBe('Su');
    expect(getKineticTypewriterText(1560).typedText).toBe('Suc');
    expect(getKineticTypewriterText(1615).typedText).toBe('Succ');
    expect(getKineticTypewriterText(1670).typedText).toBe('Succe');
    expect(getKineticTypewriterText(1725).typedText).toBe('Succes');
    expect(getKineticTypewriterText(1780).typedText).toBe('Success');
    expect(getKineticTypewriterText(1835).typedText).toBe('Success');
    expect(getKineticTypewriterText(2350).typedText).toBe('Success');
  });

  it('transitions checking -> success and holds KINETIC_AUTH_HOLD_DELAY_MS (2800ms) before closing modal', async () => {
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

      vi.advanceTimersByTime(2799);
      expect(onSuccess).not.toHaveBeenCalled();
      vi.advanceTimersByTime(1);
      expect(onSuccess).toHaveBeenCalledTimes(1);
      expect(onSuccess).toHaveBeenCalledWith(mockAdmin);
    } finally {
      vi.useRealTimers();
    }
  });

  it('transitions checking -> not_found on invalid credentials without calling onSuccess', async () => {
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

  it('renders KineticAuthStatusBadge SVG phases & visible emerald #047857 "Success" vs crimson "ไม่พบบัญชี" UI', () => {
    const completedSuccessHtml = renderToStaticMarkup(
      React.createElement(KineticAuthStatusBadge, {
        status: 'success',
        testId: 'test-kinetic-success',
        initialElapsedMs: 1950,
        animate: false,
      }),
    );
    expect(completedSuccessHtml).toContain('test-kinetic-success');
    expect(completedSuccessHtml).toContain('viewBox="0 0 100 100"');
    expect(completedSuccessHtml).toContain('stroke="#10b981"');
    expect(completedSuccessHtml).toContain('stroke="#0ea5e9"');
    expect(completedSuccessHtml).toContain('d="M33 51 L45 63 L68 38"');
    expect(completedSuccessHtml).toContain('stroke-width="8.5"');
    expect(completedSuccessHtml).toContain('stroke-width="4.5"');
    expect(completedSuccessHtml).toContain('r="34"');
    expect(completedSuccessHtml).toContain('rotate(-42 50 50)');
    expect(completedSuccessHtml).toContain('data-testid="kinetic-success-word"');
    expect(completedSuccessHtml).toContain('color:#047857');
    expect(completedSuccessHtml).toContain('>Success<');
    expect(completedSuccessHtml).toContain('เข้าสู่ระบบสำเร็จ กำลังเปิดหน้าจัดการคลินิก...');
    expect(EMOJI_RE.test(completedSuccessHtml)).toBe(false);

    const notFoundHtml = renderToStaticMarkup(
      React.createElement(KineticAuthStatusBadge, {
        status: 'not_found',
        testId: 'test-kinetic-not-found',
        secondaryLabel: 'สมัครสมาชิก (Admin / PT)',
        onNoAccountClick: () => {},
        onSecondaryClick: () => {},
        initialElapsedMs: 1950,
        animate: false,
      }),
    );
    expect(notFoundHtml).toContain('test-kinetic-not-found');
    expect(notFoundHtml).toContain('stroke="#e11d48"');
    expect(notFoundHtml).toContain('stroke="#ef4444"');
    expect(notFoundHtml).toContain('d="M36 36 L64 64"');
    expect(notFoundHtml).toContain('d="M64 36 L36 64"');
    expect(notFoundHtml).toContain('ไม่พบบัญชี');
    expect(notFoundHtml).toContain('ไม่พบชื่อผู้ใช้หรือรหัสผ่านนี้ในระบบคลินิก');
    expect(notFoundHtml).toContain('ไม่มีบัญชี');
    expect(notFoundHtml).toContain('สมัครสมาชิก (Admin / PT)');
    expect(EMOJI_RE.test(notFoundHtml)).toBe(false);
  });

  it('wires rAF KineticAuthStatusBadge inside StaffLoginModalOrPage & maintains zero emojis project-wide', () => {
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
    expect(successModalHtml).toContain('data-testid="staff-login-success-stage"');
    expect(successModalHtml).toContain('data-testid="staff-login-confetti-burst"');
    expect(successModalHtml).toContain('staff-login-success-panel');
    expect(successModalHtml).toContain('data-testid="kinetic-svg-success"');
    expect(successModalHtml).toContain('data-testid="kinetic-success-word"');
    expect(successModalHtml).toContain('color:#047857');
    expect(successModalHtml).toContain('bg-white');
    expect(successModalHtml).not.toContain('เข้าสู่ระบบบุคลากร Dr.Tech.Care');
    expect(successModalHtml).not.toContain('เข้าสู่ระบบสำเร็จ กำลังเปิดหน้าจัดการคลินิก...');
    expect(successModalHtml).not.toContain('Hospital Clinical Grade Security');
    expect(EMOJI_RE.test(successModalHtml)).toBe(false);

    const welcomeStageHtml = renderToStaticMarkup(
      React.createElement(
        ClinicalI18nProvider,
        null,
        React.createElement(StaffLoginModalOrPage, {
          apiClient,
          onSuccess: () => {},
          initialLoginAnimStatus: 'success',
          initialSuccessStage: 'welcome',
        }),
      ),
    );
    expect(welcomeStageHtml).toContain('data-testid="staff-login-welcome-panel"');
    expect(welcomeStageHtml).toContain('data-testid="staff-welcome-project-logo"');
    expect(welcomeStageHtml).toContain('data-testid="staff-welcome-project-name"');
    expect(welcomeStageHtml).toContain('Dr.Tech.Care');
    expect(welcomeStageHtml).toContain('ยินดีต้อนรับสู่บุคลากรของเรา');
    expect(EMOJI_RE.test(welcomeStageHtml)).toBe(false);

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
    expect(notFoundModalHtml).toContain('ไม่พบบัญชี');
    expect(notFoundModalHtml).toContain('ไม่พบชื่อผู้ใช้หรือรหัสผ่านนี้ในระบบคลินิก');
    expect(notFoundModalHtml).toContain('ไม่มีบัญชี');
    expect(EMOJI_RE.test(notFoundModalHtml)).toBe(false);

    const emblemMini = renderToStaticMarkup(
      React.createElement(KineticEmblemSvg, {
        status: 'success',
        className: 'w-6 h-6',
        initialElapsedMs: 2100,
        animate: false,
      }),
    );
    expect(emblemMini).toContain('stroke="#10b981"');
    expect(emblemMini).toContain('stroke="#0ea5e9"');

    const badgeSource = fs.readFileSync(
      path.resolve(process.cwd(), 'src/components/KineticAuthStatusBadge.tsx'),
      'utf8',
    );
    expect(badgeSource).toContain('requestAnimationFrame');

    for (const relPath of [
      'src/StaffLoginModalOrPage.tsx',
      'src/components/KineticAuthStatusBadge.tsx',
      'src/components/KioskBiometricLoginView.tsx',
      'src/index.css',
      'src/__tests__/loginAuthAnimation.test.ts',
    ]) {
      const content = fs.readFileSync(path.resolve(process.cwd(), relPath), 'utf8');
      expect(EMOJI_RE.test(content)).toBe(false);
    }
  });

  it('prevents Vite dev full-page reload on data/drtechcares-db.json writes and blocks accidental dismissal during login redirect', () => {
    const viteCfg = fs.readFileSync(path.resolve(process.cwd(), 'vite.config.ts'), 'utf8');
    expect(viteCfg).toContain('ignoreRuntimeDataReloadPlugin');
    expect(viteCfg).toContain("**/data/**");
    expect(viteCfg).toContain('/data/drtechcares-db.json');

    const staffLoginSrc = fs.readFileSync(
      path.resolve(process.cwd(), 'src/StaffLoginModalOrPage.tsx'),
      'utf8',
    );
    expect(staffLoginSrc).toContain(
      "if (e.key === 'Escape' && loginAnimStatus !== 'checking' && loginAnimStatus !== 'success')",
    );
    expect(staffLoginSrc).toContain(
      'if (asModal && !isCheckingOrSuccess && e.target === e.currentTarget) handleReturn();',
    );
  });
});
