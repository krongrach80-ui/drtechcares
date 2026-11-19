import React, { useEffect, useState } from 'react';

export type StaffLoginAnimStatus = 'idle' | 'checking' | 'success' | 'not_found';

export const AUTH_SUCCESS_WORD = 'Success';
export const AUTH_TYPEWRITER_STEP_MS = 65;
export const AUTH_SUCCESS_REDIRECT_DELAY_MS = 1150;

/** Returns the typewriter substring for a given tick index (`stepCount >= 0`). */
export function resolveSuccessTypewriterText(stepCount: number, targetWord = AUTH_SUCCESS_WORD): string {
  const safeCount = Math.max(0, Math.min(targetWord.length, Math.floor(stepCount)));
  return targetWord.slice(0, safeCount);
}

/** Shared hook driving letter-by-letter typewriter animation for `"Success"`. */
export function useSuccessTypewriter(active: boolean, intervalMs = AUTH_TYPEWRITER_STEP_MS): string {
  const [charCount, setCharCount] = useState<number>(active ? 1 : 0);

  useEffect(() => {
    if (!active) {
      setCharCount(0);
      return;
    }
    setCharCount(1);
    const id = window.setInterval(() => {
      setCharCount((prev) => {
        if (prev >= AUTH_SUCCESS_WORD.length) {
          window.clearInterval(id);
          return AUTH_SUCCESS_WORD.length;
        }
        return prev + 1;
      });
    }, intervalMs);

    return () => window.clearInterval(id);
  }, [active, intervalMs]);

  return resolveSuccessTypewriterText(charCount);
}

export function AnimatedCircleCheckmarkSvg({ className = 'w-20 h-20 mx-auto animate-circle-pop' }: { className?: string }) {
  return (
    <svg viewBox="0 0 52 52" className={className} aria-hidden="true">
      <circle
        cx="26"
        cy="26"
        r="24"
        fill="none"
        stroke="#10b981"
        strokeWidth="3.5"
        className="animate-svg-circle"
      />
      <path
        fill="none"
        stroke="#059669"
        strokeWidth="4"
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M14.5 27.2l7.1 7.2 16.7-16.8"
        className="animate-svg-check"
      />
    </svg>
  );
}

export function AnimatedCircleCrossSvg({ className = 'w-20 h-20 mx-auto animate-circle-pop' }: { className?: string }) {
  return (
    <svg viewBox="0 0 52 52" className={className} aria-hidden="true">
      <circle
        cx="26"
        cy="26"
        r="24"
        fill="none"
        stroke="#ef4444"
        strokeWidth="3.5"
        className="animate-svg-circle"
      />
      <path
        fill="none"
        stroke="#dc2626"
        strokeWidth="4"
        strokeLinecap="round"
        d="M17 17 L35 35"
        className="animate-svg-cross-1"
      />
      <path
        fill="none"
        stroke="#dc2626"
        strokeWidth="4"
        strokeLinecap="round"
        d="M35 17 L17 35"
        className="animate-svg-cross-2"
      />
    </svg>
  );
}

export interface AuthSuccessFeedbackPanelProps {
  subtitle?: string;
  compact?: boolean;
  testId?: string;
}

/** Centered animated emerald circle-checkmark + typing `"Success"` clinical feedback card. */
export function AuthSuccessFeedbackPanel({
  subtitle = 'เข้าสู่ระบบสำเร็จ กำลังเปิดหน้าจัดการคลินิก...',
  compact = false,
  testId = 'auth-success-feedback-panel',
}: AuthSuccessFeedbackPanelProps) {
  const typedText = useSuccessTypewriter(true);

  return (
    <div
      role="status"
      aria-label="Success"
      data-testid={testId}
      className={`animate-popup-spring bg-emerald-50/95 border-2 border-emerald-400 rounded-2xl text-center shadow-lg ${
        compact ? 'p-4' : 'p-6'
      }`}
    >
      <AnimatedCircleCheckmarkSvg
        className={compact ? 'w-16 h-16 mx-auto animate-circle-pop' : 'w-20 h-20 mx-auto animate-circle-pop'}
      />
      <div className={compact ? 'mt-2' : 'mt-3'}>
        <span
          data-testid="auth-success-typewriter"
          className={`inline-block font-black text-emerald-700 tracking-wide border-r-2 border-emerald-600 pr-1 animate-auth-cursor ${
            compact ? 'text-xl md:text-2xl' : 'text-2xl md:text-3xl'
          }`}
        >
          {typedText}
        </span>
      </div>
      {subtitle && (
        <p className="text-sm font-bold text-emerald-900 mt-1">{subtitle}</p>
      )}
    </div>
  );
}

export interface AuthNotFoundFeedbackPanelProps {
  heading?: string;
  subtext?: string;
  noAccountLabel?: string;
  secondaryLabel?: string;
  onNoAccountClick?: () => void;
  onSecondaryClick?: () => void;
  compact?: boolean;
  id?: string;
  testId?: string;
}

/** Prominent animated crimson circle-cross (`X`) + `"ไม่พบบัญชี"` + `"ไม่มีบัญชี"` pill button. */
export function AuthNotFoundFeedbackPanel({
  heading = 'ไม่พบบัญชี',
  subtext = 'ไม่พบชื่อผู้ใช้หรือรหัสผ่านนี้ในระบบคลินิก',
  noAccountLabel = 'ไม่มีบัญชี',
  secondaryLabel,
  onNoAccountClick,
  onSecondaryClick,
  compact = false,
  id,
  testId = 'auth-not-found-feedback-panel',
}: AuthNotFoundFeedbackPanelProps) {
  return (
    <div
      id={id}
      role="alert"
      data-testid={testId}
      className={`animate-popup-spring animate-gentle-shake bg-red-50/95 border-2 border-red-400 rounded-2xl text-center shadow-lg ${
        compact ? 'p-4' : 'p-5'
      }`}
    >
      <AnimatedCircleCrossSvg
        className={compact ? 'w-16 h-16 mx-auto animate-circle-pop' : 'w-20 h-20 mx-auto animate-circle-pop'}
      />
      <h3 className={`font-black text-red-700 ${compact ? 'text-xl mt-2' : 'text-2xl mt-3'}`}>
        {heading}
      </h3>
      {subtext && (
        <p className="text-sm font-semibold text-red-800 mt-1 leading-relaxed">{subtext}</p>
      )}
      <div className="mt-3 flex flex-wrap items-center justify-center gap-2">
        <button
          type="button"
          data-testid="auth-no-account-btn"
          onClick={onNoAccountClick}
          className="bg-red-600 text-white font-extrabold px-5 py-2.5 rounded-xl shadow-md cursor-pointer hover:bg-red-700 active:scale-95 transition-all"
        >
          {noAccountLabel}
        </button>
        {secondaryLabel && onSecondaryClick && (
          <button
            type="button"
            data-testid="auth-not-found-secondary-btn"
            onClick={onSecondaryClick}
            className="bg-white border border-red-300 text-red-700 font-bold px-4 py-2.5 rounded-xl shadow-xs cursor-pointer hover:bg-red-100/70 active:scale-95 transition-all"
          >
            {secondaryLabel}
          </button>
        )}
      </div>
    </div>
  );
}
