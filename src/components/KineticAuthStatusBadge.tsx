import React, { useEffect, useId, useState } from 'react';

export type KineticAuthStatus = 'success' | 'not_found';

export const KINETIC_SUCCESS_WORD = 'Success';
export const KINETIC_TYPEWRITER_START_DELAY_MS = 750;
export const KINETIC_TYPEWRITER_STEP_MS = 55;
export const KINETIC_AUTH_HOLD_DELAY_MS = 1400;

/** Computes progressive typewriter slices (`0..word.length`) for `"Success"`. */
export function resolveKineticTypewriterText(stepCount: number, targetWord = KINETIC_SUCCESS_WORD): string {
  const clamped = Math.max(0, Math.min(targetWord.length, Math.floor(stepCount)));
  return targetWord.slice(0, clamped);
}

/**
 * 4-stage typewriter controller: begins at `startDelayMs` (750ms) once Stage 3 ring orbit nears completion,
 * stepping letter-by-letter every `stepMs` (55ms) through `"Success"`.
 */
export function useKineticTypewriter(
  active: boolean,
  options?: { word?: string; startDelayMs?: number; stepMs?: number },
): string {
  const word = options?.word ?? KINETIC_SUCCESS_WORD;
  const startDelayMs = options?.startDelayMs ?? KINETIC_TYPEWRITER_START_DELAY_MS;
  const stepMs = options?.stepMs ?? KINETIC_TYPEWRITER_STEP_MS;
  const [charCount, setCharCount] = useState<number>(0);

  useEffect(() => {
    if (!active) {
      setCharCount(0);
      return;
    }
    setCharCount(0);
    let intervalId: number | null = null;
    const delayId = window.setTimeout(() => {
      setCharCount(1);
      intervalId = window.setInterval(() => {
        setCharCount((prev) => {
          if (prev >= word.length) {
            if (intervalId !== null) window.clearInterval(intervalId);
            return word.length;
          }
          return prev + 1;
        });
      }, stepMs);
    }, startDelayMs);

    return () => {
      window.clearTimeout(delayId);
      if (intervalId !== null) window.clearInterval(intervalId);
    };
  }, [active, word, startDelayMs, stepMs]);

  return resolveKineticTypewriterText(charCount, word);
}

const TIP_SPARK_VECTORS = [
  { sx: '-7px', sy: '-9px', r: '1.7' },
  { sx: '9px', sy: '-8px', r: '2.0' },
  { sx: '10px', sy: '4px', r: '1.5' },
] as const;

const CLOSURE_SPARK_VECTORS = [
  { sx: '-6px', sy: '-10px', r: '1.8' },
  { sx: '9px', sy: '-9px', r: '2.1' },
  { sx: '11px', sy: '3px', r: '1.6' },
] as const;

export interface KineticEmblemSvgProps {
  status: KineticAuthStatus;
  className?: string;
}

/**
 * Pure GPU-accelerated 4-stage kinetic SVG emblem (`viewBox="0 0 100 100"`):
 * Stage 1: Dot bounce-in (`0-220ms`)
 * Stage 2: Bold rounded checkmark / cross stroke draw (`200-440ms`) + accent dot pop at `(69, 29)` + 3 sparks (`420ms`)
 * Stage 3: Orbiting dot sweeps `360deg` clockwise from `-45deg` around `r="38"` trailing outer ring (`440-880ms`) + closure sparks at `(78, 24)` (`860ms`)
 */
export function KineticEmblemSvg({
  status,
  className = 'w-24 h-24 mx-auto overflow-visible',
}: KineticEmblemSvgProps) {
  const uid = useId().replace(/:/g, '');
  const isSuccess = status === 'success';
  const gradId = `kineticGrad-${uid}-${status}`;
  const glowId = `kineticGlow-${uid}-${status}`;
  const dotPrimary = isSuccess ? '#10b981' : '#ef4444';
  const dotAccent = isSuccess ? '#06b6d4' : '#dc2626';
  const strokeMain = isSuccess ? '#059669' : '#dc2626';
  const sparkColor = isSuccess ? '#34d399' : '#f87171';

  return (
    <svg viewBox="0 0 100 100" className={className} aria-hidden="true">
      <defs>
        <linearGradient id={gradId} x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor={isSuccess ? '#10b981' : '#ef4444'} />
          <stop offset="100%" stopColor={isSuccess ? '#06b6d4' : '#dc2626'} />
        </linearGradient>
        <filter id={glowId} x="-35%" y="-35%" width="170%" height="170%">
          <feGaussianBlur stdDeviation="2.2" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      {/* Subtle clinical halo disc */}
      <circle
        cx="50"
        cy="50"
        r="38"
        fill={isSuccess ? 'rgba(16, 185, 129, 0.08)' : 'rgba(239, 68, 68, 0.08)'}
      />

      {/* Stage 1 (0-220ms): Glowing dot bounces from +12px below into stroke origin */}
      <circle
        cx={isSuccess ? 34 : 36}
        cy={isSuccess ? 52 : 36}
        r="4.5"
        fill={dotPrimary}
        filter={`url(#${glowId})`}
        style={{ transformOrigin: isSuccess ? '34px 52px' : '36px 36px' }}
        className="animate-kinetic-dot-bounce"
      />

      {/* Stage 2 (200-440ms): Thick rounded Checkmark or Dual-Stroke Cross */}
      {isSuccess ? (
        <path
          d="M32 52 L45 65 L69 37"
          fill="none"
          stroke={`url(#${gradId})`}
          strokeWidth="6.5"
          strokeLinecap="round"
          strokeLinejoin="round"
          className="animate-kinetic-check"
        />
      ) : (
        <>
          <path
            d="M36 36 L64 64"
            fill="none"
            stroke={strokeMain}
            strokeWidth="6.5"
            strokeLinecap="round"
            className="animate-kinetic-cross-1"
          />
          <path
            d="M64 36 L36 64"
            fill="none"
            stroke={dotPrimary}
            strokeWidth="6.5"
            strokeLinecap="round"
            className="animate-kinetic-cross-2"
          />
        </>
      )}

      {/* Stage 2 Accent Dot Pop (420ms) at (69, 29) + 3 outward micro-sparks */}
      <circle
        cx="69"
        cy="29"
        r="4.5"
        fill={dotAccent}
        filter={`url(#${glowId})`}
        style={{ transformOrigin: '69px 29px' }}
        className="animate-kinetic-tip-dot"
      />
      {TIP_SPARK_VECTORS.map((spark, idx) => (
        <circle
          key={`tip-spark-${idx}`}
          cx="69"
          cy="29"
          r={spark.r}
          fill={sparkColor}
          className="animate-kinetic-tip-spark"
          style={
            {
              transformOrigin: '69px 29px',
              '--sx': spark.sx,
              '--sy': spark.sy,
            } as React.CSSProperties
          }
        />
      ))}

      {/* Stage 3 (440-880ms): Outer circle ring sweeps 360deg clockwise starting at -45deg */}
      <circle
        cx="50"
        cy="50"
        r="38"
        fill="none"
        stroke={`url(#${gradId})`}
        strokeWidth="5.5"
        strokeLinecap="round"
        strokeDasharray="239"
        transform="rotate(-45 50 50)"
        className="animate-kinetic-ring"
      />

      {/* Orbiting accent dot tracking leading tip of outer ring from -45deg (76.87, 23.13) clockwise 360deg */}
      <g
        style={{ transformOrigin: '50px 50px' }}
        className="animate-kinetic-orbit"
      >
        <circle
          cx="76.87"
          cy="23.13"
          r="4.5"
          fill={dotAccent}
          filter={`url(#${glowId})`}
        />
      </g>

      {/* Stage 3 Closure Sparks (860ms) twinkling near (78, 24) */}
      {CLOSURE_SPARK_VECTORS.map((spark, idx) => (
        <circle
          key={`close-spark-${idx}`}
          cx="78"
          cy="24"
          r={spark.r}
          fill={idx === 1 ? dotAccent : sparkColor}
          className="animate-kinetic-closure-spark"
          style={
            {
              transformOrigin: '78px 24px',
              '--sx': spark.sx,
              '--sy': spark.sy,
            } as React.CSSProperties
          }
        />
      ))}
    </svg>
  );
}

export interface KineticAuthStatusBadgeProps {
  status: KineticAuthStatus;
  subtitle?: string;
  heading?: string;
  noAccountLabel?: string;
  secondaryLabel?: string;
  onNoAccountClick?: () => void;
  onSecondaryClick?: () => void;
  compact?: boolean;
  id?: string;
  testId?: string;
}

/**
 * Reusable clinical 4-stage kinetic status badge for Staff Portal Login and Kiosk Patient Face Login.
 */
export function KineticAuthStatusBadge({
  status,
  subtitle,
  heading = 'ไม่พบบัญชี',
  noAccountLabel = 'ไม่มีบัญชี',
  secondaryLabel,
  onNoAccountClick,
  onSecondaryClick,
  compact = false,
  id,
  testId,
}: KineticAuthStatusBadgeProps) {
  const isSuccess = status === 'success';
  const typedSuccess = useKineticTypewriter(isSuccess, {
    word: KINETIC_SUCCESS_WORD,
    startDelayMs: KINETIC_TYPEWRITER_START_DELAY_MS,
    stepMs: KINETIC_TYPEWRITER_STEP_MS,
  });

  if (isSuccess) {
    const resolvedSubtitle = subtitle ?? 'เข้าสู่ระบบสำเร็จ กำลังเปิดหน้าจัดการคลินิก...';
    return (
      <div
        id={id}
        role="status"
        aria-label="Success"
        data-testid={testId ?? 'kinetic-auth-success-badge'}
        className={`animate-popup-spring bg-gradient-to-b from-white via-emerald-50/90 to-sky-50/70 border-2 border-emerald-400 rounded-2xl text-center shadow-lg ${
          compact ? 'p-4' : 'p-6'
        }`}
      >
        <KineticEmblemSvg
          status="success"
          className={compact ? 'w-20 h-20 mx-auto overflow-visible' : 'w-24 h-24 mx-auto overflow-visible'}
        />
        <div className={compact ? 'mt-2 min-h-[2rem]' : 'mt-3 min-h-[2.4rem]'}>
          <span
            data-testid="kinetic-success-typewriter"
            className={`inline-block font-black text-emerald-700 tracking-wide border-r-2 border-emerald-600 pr-1 animate-auth-cursor ${
              compact ? 'text-xl md:text-2xl' : 'text-2xl md:text-3xl'
            }`}
          >
            {typedSuccess || '\u00A0'}
          </span>
        </div>
        {resolvedSubtitle && (
          <p className="text-sm font-bold text-[#0F4C81] mt-1">{resolvedSubtitle}</p>
        )}
      </div>
    );
  }

  const resolvedErrorSubtitle = subtitle ?? 'ไม่พบชื่อผู้ใช้หรือรหัสผ่านนี้ในระบบ';
  return (
    <div
      id={id}
      role="alert"
      data-testid={testId ?? 'kinetic-auth-not-found-badge'}
      className={`animate-popup-spring animate-error-shake bg-red-50/95 border-2 border-red-400 rounded-2xl text-center shadow-lg ${
        compact ? 'p-4' : 'p-5'
      }`}
    >
      <KineticEmblemSvg
        status="not_found"
        className={compact ? 'w-20 h-20 mx-auto overflow-visible' : 'w-24 h-24 mx-auto overflow-visible'}
      />
      <h3 className={`font-black text-red-700 ${compact ? 'text-xl mt-2' : 'text-2xl mt-3'}`}>
        {heading}
      </h3>
      {resolvedErrorSubtitle && (
        <p className="text-sm font-semibold text-red-800 mt-1 leading-relaxed">
          {resolvedErrorSubtitle}
        </p>
      )}
      <div className="mt-3 flex flex-wrap items-center justify-center gap-2">
        <button
          type="button"
          data-testid="kinetic-no-account-btn"
          onClick={onNoAccountClick}
          className="bg-red-600 hover:bg-red-700 active:scale-95 text-white font-extrabold px-6 py-2.5 rounded-xl shadow-md transition-all cursor-pointer"
        >
          {noAccountLabel}
        </button>
        {secondaryLabel && onSecondaryClick && (
          <button
            type="button"
            data-testid="kinetic-not-found-secondary-btn"
            onClick={onSecondaryClick}
            className="bg-white border border-red-300 text-red-700 hover:bg-red-100/70 active:scale-95 font-bold px-4 py-2.5 rounded-xl shadow-xs transition-all cursor-pointer"
          >
            {secondaryLabel}
          </button>
        )}
      </div>
    </div>
  );
}

