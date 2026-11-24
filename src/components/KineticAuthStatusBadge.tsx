import React, { useEffect, useState } from 'react';

export type KineticAuthStatus = 'success' | 'not_found';

export const KINETIC_SUCCESS_WORD = 'Success';
export const KINETIC_TYPEWRITER_START_DELAY_MS = 1450;
export const KINETIC_TYPEWRITER_STEP_MS = 55;
export const KINETIC_TOTAL_DURATION_MS = 2350;
export const KINETIC_CLOCK_MAX_MS = 2400;
export const KINETIC_AUTH_HOLD_DELAY_MS = 2800;
export const KIOSK_AUTH_HOLD_DELAY_MS = 2400;

export interface KineticSparkFrame {
  cx: number;
  cy: number;
  r: number;
  opacity: number;
  color: string;
}

export interface KineticAuthFrame {
  elapsedMs: number;
  phase: 1 | 2 | 3 | 4 | 5;
  innerColor: string;
  ringColor: string;
  innerScale: number;
  innerOpacity: number;
  checkDashOffset: number;
  crossDashOffset1: number;
  crossDashOffset2: number;
  shakeX: number;
  dotVisible: boolean;
  dotX: number;
  dotY: number;
  dotRadius: number;
  hopTailOpacity: number;
  hopTailDashOffset: number;
  sweepProgress: number;
  ringOpacity: number;
  ringDashOffset: number;
  ringScale: number;
  sparks: KineticSparkFrame[];
  typedText: string;
  isTypingActive: boolean;
}

export interface KineticEmblemSvgProps {
  status: KineticAuthStatus;
  className?: string;
  initialElapsedMs?: number;
  animate?: boolean;
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
  hero?: boolean;
  id?: string;
  testId?: string;
  initialElapsedMs?: number;
  animate?: boolean;
}

function clamp01(value: number): number {
  if (value <= 0) return 0;
  if (value >= 1) return 1;
  return value;
}

function clampElapsed(value: number): number {
  if (!Number.isFinite(value) || value <= 0) return 0;
  if (value >= KINETIC_CLOCK_MAX_MS) return KINETIC_CLOCK_MAX_MS;
  return value;
}

function lerp(start: number, end: number, t: number): number {
  return start + (end - start) * t;
}

function easeOutCubic(t: number): number {
  const k = clamp01(t);
  return 1 - Math.pow(1 - k, 3);
}

function easeInOutCubic(t: number): number {
  const k = clamp01(t);
  return k < 0.5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2;
}

function quadBezier(
  p0: { x: number; y: number },
  p1: { x: number; y: number },
  p2: { x: number; y: number },
  t: number
): { x: number; y: number } {
  const u = clamp01(t);
  const mt = 1 - u;
  return {
    x: mt * mt * p0.x + 2 * mt * u * p1.x + u * u * p2.x,
    y: mt * mt * p0.y + 2 * mt * u * p1.y + u * u * p2.y,
  };
}

/**
 * Computes typed prefix for `"Success"` starting at `1450ms` (`55ms` per character, finishing at `1835ms`).
 */
export function getKineticTypewriterText(
  elapsedMs: number,
  word: string = KINETIC_SUCCESS_WORD
): { typedText: string; isTypingActive: boolean } {
  if (!Number.isFinite(elapsedMs) || elapsedMs < KINETIC_TYPEWRITER_START_DELAY_MS) {
    return { typedText: '', isTypingActive: false };
  }
  const offset = elapsedMs - KINETIC_TYPEWRITER_START_DELAY_MS;
  const charCount = Math.min(
    word.length,
    Math.floor(offset / KINETIC_TYPEWRITER_STEP_MS) + 1
  );
  const finishTime = KINETIC_TYPEWRITER_START_DELAY_MS + word.length * KINETIC_TYPEWRITER_STEP_MS;
  return {
    typedText: word.slice(0, charCount),
    isTypingActive: elapsedMs >= KINETIC_TYPEWRITER_START_DELAY_MS && elapsedMs <= finishTime + 350,
  };
}

/**
 * Pure deterministic 60FPS choreography frame solver (`0ms` to `2400ms`),
 * completely driven by `requestAnimationFrame` timestamps so OS `prefers-reduced-motion`
 * cannot freeze SVG geometry or stroke dash offsets.
 */
export function computeKineticAuthFrame(
  rawElapsedMs: number,
  status: KineticAuthStatus
): KineticAuthFrame {
  const t = clampElapsed(rawElapsedMs);
  const isSuccess = status === 'success';

  const innerColor = isSuccess ? '#10b981' : '#e11d48';
  const ringColor = isSuccess ? '#0ea5e9' : '#ef4444';
  const sparkPrimary = isSuccess ? '#38bdf8' : '#f87171';
  const sparkSecondary = isSuccess ? '#10b981' : '#e11d48';

  const startAngle = (-42 * Math.PI) / 180;
  const perimStart = {
    x: 50 + 34 * Math.cos(startAngle),
    y: 50 + 34 * Math.sin(startAngle),
  };

  let phase: 1 | 2 | 3 | 4 | 5 = 1;
  if (t >= 1920) phase = 5;
  else if (t >= 1020) phase = 4;
  else if (t >= 680) phase = 3;
  else if (t >= 430) phase = 2;

  // Inner symbol stroke progress (430ms - 650ms for check, 430ms - 660ms staggered for cross)
  const pCheck = clamp01((t - 430) / 220);
  const checkDashOffset = 100 * (1 - easeOutCubic(pCheck));
  const crossDashOffset1 = 100 * (1 - easeOutCubic(clamp01((t - 430) / 130)));
  const crossDashOffset2 = 100 * (1 - easeOutCubic(clamp01((t - 530) / 130)));
  const innerOpacity = t >= 430 ? 1 : 0;

  let innerScale = 1;
  if (t >= 430 && t < 760) {
    const pPop = clamp01((t - 430) / 330);
    innerScale = pPop < 0.45 ? lerp(0.92, 1.06, pPop / 0.45) : lerp(1.06, 1.0, (pPop - 0.45) / 0.55);
  }

  let shakeX = 0;
  if (!isSuccess && t >= 460 && t <= 840) {
    const pShake = clamp01((t - 460) / 380);
    shakeX = Math.sin(pShake * Math.PI * 5) * (1 - pShake) * 3.6;
  }

  // Hop tail (curved launch trail during Phase 1)
  let hopTailOpacity = 0;
  let hopTailDashOffset = 100;
  if (t < 330) {
    const pTailDraw = clamp01(t / 190);
    hopTailDashOffset = 100 * (1 - easeOutCubic(pTailDraw));
    if (t < 70) hopTailOpacity = lerp(0, 0.85, clamp01(t / 70));
    else if (t < 190) hopTailOpacity = 0.85;
    else hopTailOpacity = lerp(0.85, 0, clamp01((t - 190) / 140));
  }

  // Leading dot kinematics across Phases 1..5
  let dotVisible = true;
  let dotX = 50;
  let dotY = 56;
  let dotRadius = 4.6;
  const sparks: KineticSparkFrame[] = [];

  if (t < 240) {
    // Phase 1a (0 - 240ms): curved hop-up from (50, 56) -> control (43, 47) -> (42, 35)
    const pHop = easeOutCubic(t / 240);
    const pos = quadBezier({ x: 50, y: 56 }, { x: 43, y: 47 }, { x: 42, y: 35 }, pHop);
    dotX = pos.x;
    dotY = pos.y;
    dotRadius = lerp(3.4, 4.6, pHop);
  } else if (t < 450) {
    // Phase 1b (240 - 450ms): straight drop from (42, 35) to center (50, 50) with subtle squash
    const pDrop = easeInOutCubic((t - 240) / 210);
    dotX = lerp(42, 50, pDrop);
    dotY = lerp(35, 50, pDrop);
    const bounceWave = t > 380 ? Math.sin(((t - 380) / 70) * Math.PI) * 0.7 : 0;
    dotRadius = 4.6 + bounceWave;
  } else if (t < 680) {
    // Phase 2 (450 - 680ms): sweeps across tip (68, 38) then pops upward to hover at (68, 26)
    const pLift = clamp01((t - 450) / 230);
    const tipRef = isSuccess ? { x: 68, y: 38 } : { x: 65, y: 36 };
    if (pLift < 0.42) {
      const sub = easeOutCubic(pLift / 0.42);
      dotX = lerp(50, tipRef.x, sub);
      dotY = lerp(50, tipRef.y, sub);
    } else {
      const sub = easeOutCubic((pLift - 0.42) / 0.58);
      dotX = lerp(tipRef.x, 68, sub);
      dotY = lerp(tipRef.y, 26, sub);
    }
    dotRadius = lerp(4.6, 4.8, pLift);
  } else if (t < 1020) {
    // Phase 3 (680 - 1020ms): hover bounce around (68, 25)
    const pHover = clamp01((t - 680) / 340);
    dotX = 68;
    dotY = 26 - Math.sin(pHover * Math.PI) * 2.0;
    dotRadius = 4.8;
  } else if (t < 1140) {
    // Phase 4a (1020 - 1140ms): curved dive from (68, 26) onto circle perimeter at -42deg
    const pDive = easeOutCubic((t - 1020) / 120);
    const pos = quadBezier({ x: 68, y: 26 }, { x: 72, y: 22 }, perimStart, pDive);
    dotX = pos.x;
    dotY = pos.y;
    dotRadius = 4.8;
  } else if (t < 1920) {
    // Phase 4b (1140 - 1920ms): clockwise 360-degree sweep leading the outer circle arc
    const sweepProgress = easeInOutCubic((t - 1140) / 780);
    const angle = startAngle + sweepProgress * Math.PI * 2;
    dotX = 50 + 34 * Math.cos(angle);
    dotY = 50 + 34 * Math.sin(angle);
    dotRadius = 4.8;
  } else {
    // Phase 5 (1920 - 2350ms): leading dot shrinks and merges into closed outer ring
    const pMerge = clamp01((t - 1920) / 140);
    dotX = perimStart.x;
    dotY = perimStart.y;
    dotRadius = lerp(4.8, 0, easeOutCubic(pMerge));
    dotVisible = dotRadius > 0.08;
  }

  // Impact micro-spark (370ms - 530ms) near (42, 52)
  if (t >= 370 && t <= 530) {
    const pImp = clamp01((t - 370) / 160);
    const op = pImp < 0.35 ? pImp / 0.35 : 1 - (pImp - 0.35) / 0.65;
    sparks.push({
      cx: lerp(45, 42, easeOutCubic(pImp)),
      cy: lerp(51, 52, easeOutCubic(pImp)),
      r: lerp(0.8, 1.7, clamp01(pImp * 1.6)),
      opacity: clamp01(op * 0.9),
      color: sparkPrimary,
    });
  }

  // Hover burst micro-sparks (690ms - 1020ms) at (73, 19) & (77, 24)
  if (t >= 690 && t <= 1020) {
    const pHoverSparks = clamp01((t - 690) / 330);
    const travel = easeOutCubic(Math.min(1, pHoverSparks * 1.45));
    const fade =
      pHoverSparks < 0.22
        ? pHoverSparks / 0.22
        : pHoverSparks > 0.6
          ? 1 - (pHoverSparks - 0.6) / 0.4
          : 1;
    const sparkOpacity = clamp01(fade);
    sparks.push(
      {
        cx: lerp(69, 73, travel),
        cy: lerp(25, 19, travel),
        r: 2.3,
        opacity: sparkOpacity,
        color: sparkPrimary,
      },
      {
        cx: lerp(70, 77, travel),
        cy: lerp(26, 24, travel),
        r: 1.6,
        opacity: sparkOpacity * 0.9,
        color: sparkSecondary,
      }
    );
  }

  // Closure celebration micro-sparks (1920ms - 2350ms) at (79, 18), (84, 24), (74, 14)
  if (t >= 1920 && t <= 2350) {
    const pClose = clamp01((t - 1920) / 430);
    const travel = easeOutCubic(Math.min(1, pClose * 1.35));
    const fade =
      pClose < 0.18
        ? pClose / 0.18
        : pClose > 0.55
          ? 1 - (pClose - 0.55) / 0.45
          : 1;
    const sparkOpacity = clamp01(fade);
    sparks.push(
      {
        cx: lerp(perimStart.x, 79, travel),
        cy: lerp(perimStart.y, 18, travel),
        r: 2.3,
        opacity: sparkOpacity,
        color: sparkPrimary,
      },
      {
        cx: lerp(perimStart.x, 84, travel),
        cy: lerp(perimStart.y, 24, travel),
        r: 1.7,
        opacity: sparkOpacity * 0.92,
        color: sparkSecondary,
      },
      {
        cx: lerp(perimStart.x, 74, travel),
        cy: lerp(perimStart.y, 14, travel),
        r: 1.4,
        opacity: sparkOpacity * 0.85,
        color: sparkPrimary,
      }
    );
  }

  // Outer circle sweep state
  const sweepProgress = t < 1140 ? 0 : t >= 1920 ? 1 : easeInOutCubic((t - 1140) / 780);
  const ringDashOffset = 100 * (1 - sweepProgress);
  const ringOpacity = sweepProgress > 0.002 ? 1 : 0;

  let ringScale = 1;
  if (t >= 1920 && t < 2240) {
    const pSettle = clamp01((t - 1920) / 320);
    ringScale = 1 + Math.sin(pSettle * Math.PI) * 0.035;
  }

  const typewriter = isSuccess
    ? getKineticTypewriterText(t, KINETIC_SUCCESS_WORD)
    : { typedText: '', isTypingActive: false };

  return {
    elapsedMs: t,
    phase,
    innerColor,
    ringColor,
    innerScale,
    innerOpacity,
    checkDashOffset,
    crossDashOffset1,
    crossDashOffset2,
    shakeX,
    dotVisible,
    dotX,
    dotY,
    dotRadius,
    hopTailOpacity,
    hopTailDashOffset,
    sweepProgress,
    ringOpacity,
    ringDashOffset,
    ringScale,
    sparks,
    typedText: typewriter.typedText,
    isTypingActive: typewriter.isTypingActive,
  };
}

/**
 * Shared 60FPS `requestAnimationFrame` clock driving SVG geometry deterministically (`0 -> 2400ms`).
 */
export function useKineticFrameClock(
  animate: boolean = true,
  initialElapsedMs?: number,
  resetKey?: string | number
): number {
  const initialClamped = clampElapsed(initialElapsedMs ?? 0);
  const [elapsedMs, setElapsedMs] = useState<number>(initialClamped);

  useEffect(() => {
    if (!animate) {
      return;
    }

    const startOffset = clampElapsed(initialElapsedMs ?? 0);
    setElapsedMs(startOffset);
    if (startOffset >= KINETIC_CLOCK_MAX_MS || typeof window === 'undefined') {
      return;
    }

    let rafId = 0;
    const startedAt = performance.now() - startOffset;

    const tick = (now: number) => {
      const nextElapsed = Math.min(KINETIC_CLOCK_MAX_MS, Math.max(0, now - startedAt));
      setElapsedMs(nextElapsed);
      if (nextElapsed < KINETIC_CLOCK_MAX_MS) {
        rafId = window.requestAnimationFrame(tick);
      }
    };

    rafId = window.requestAnimationFrame(tick);
    return () => {
      if (rafId) window.cancelAnimationFrame(rafId);
    };
  }, [animate, initialElapsedMs, resetKey]);

  if (!animate) {
    return initialElapsedMs !== undefined ? initialClamped : KINETIC_TOTAL_DURATION_MS;
  }

  return elapsedMs;
}

export const KineticEmblemSvg: React.FC<KineticEmblemSvgProps> = ({
  status,
  className = 'w-28 h-28 md:w-32 md:h-32 mx-auto overflow-visible',
  initialElapsedMs,
  animate = true,
}) => {
  const elapsedMs = useKineticFrameClock(animate, initialElapsedMs, status);
  const frame = computeKineticAuthFrame(elapsedMs, status);
  const isSuccess = status === 'success';

  const groupTransform = [
    frame.shakeX !== 0 ? `translate(${frame.shakeX.toFixed(2)}, 0)` : '',
    frame.innerScale !== 1
      ? `translate(50, 50) scale(${frame.innerScale.toFixed(4)}) translate(-50, -50)`
      : '',
  ]
    .filter(Boolean)
    .join(' ');

  const ringTransform =
    frame.ringScale !== 1
      ? `translate(50, 50) scale(${frame.ringScale.toFixed(4)}) translate(-50, -50) rotate(-42 50 50)`
      : 'rotate(-42 50 50)';

  return (
    <svg
      viewBox="0 0 100 100"
      className={className}
      fill="none"
      aria-hidden="true"
      data-testid={`kinetic-svg-${status}`}
      data-phase={frame.phase}
      data-elapsed-ms={Math.round(frame.elapsedMs)}
    >
      {/* Subtle circular glow halo */}
      <circle
        cx="50"
        cy="50"
        r="34"
        fill={isSuccess ? 'rgba(16, 185, 129, 0.08)' : 'rgba(225, 29, 72, 0.08)'}
        opacity={frame.ringOpacity * 0.9 + 0.1}
      />

      {/* Curved hop-up launch trail in Phase 1 */}
      {frame.hopTailOpacity > 0.01 && (
        <path
          d="M50 58 Q43 47 42 35"
          stroke={frame.ringColor}
          strokeWidth="3.2"
          strokeLinecap="round"
          pathLength="100"
          strokeDasharray="100"
          strokeDashoffset={frame.hopTailDashOffset}
          opacity={frame.hopTailOpacity}
        />
      )}

      {/* Thick Inner Checkmark / Cross */}
      {frame.innerOpacity > 0 && (
        <g transform={groupTransform || undefined} opacity={frame.innerOpacity}>
          {isSuccess ? (
            <path
              d="M33 51 L45 63 L68 38"
              stroke={frame.innerColor}
              strokeWidth="8.5"
              strokeLinecap="round"
              strokeLinejoin="round"
              pathLength="100"
              strokeDasharray="100"
              strokeDashoffset={frame.checkDashOffset}
            />
          ) : (
            <>
              <path
                d="M36 36 L64 64"
                stroke={frame.innerColor}
                strokeWidth="8.5"
                strokeLinecap="round"
                pathLength="100"
                strokeDasharray="100"
                strokeDashoffset={frame.crossDashOffset1}
              />
              {frame.crossDashOffset2 < 99.9 && (
                <path
                  d="M64 36 L36 64"
                  stroke={frame.innerColor}
                  strokeWidth="8.5"
                  strokeLinecap="round"
                  pathLength="100"
                  strokeDasharray="100"
                  strokeDashoffset={frame.crossDashOffset2}
                />
              )}
            </>
          )}
        </g>
      )}

      {/* Outer swept 360-degree ring */}
      {frame.ringOpacity > 0 && (
        <circle
          cx="50"
          cy="50"
          r="34"
          stroke={frame.ringColor}
          strokeWidth="4.5"
          strokeLinecap="round"
          pathLength="100"
          strokeDasharray="100"
          strokeDashoffset={frame.ringDashOffset}
          transform={ringTransform}
          opacity={frame.ringOpacity}
        />
      )}

      {/* Leading Orbiting Dot */}
      {frame.dotVisible && (
        <circle
          cx={Number(frame.dotX.toFixed(2))}
          cy={Number(frame.dotY.toFixed(2))}
          r={Number(frame.dotRadius.toFixed(2))}
          fill={frame.ringColor}
        />
      )}

      {/* Micro-Sparks (Impact, Hover, Closure Celebration) */}
      {frame.sparks.map((spark, index) => (
        <circle
          key={index}
          cx={Number(spark.cx.toFixed(2))}
          cy={Number(spark.cy.toFixed(2))}
          r={Number(spark.r.toFixed(2))}
          fill={spark.color}
          opacity={Number(spark.opacity.toFixed(3))}
        />
      ))}
    </svg>
  );
};

export const KineticAuthStatusBadge: React.FC<KineticAuthStatusBadgeProps> = ({
  status,
  subtitle,
  heading,
  noAccountLabel = 'ไม่มีบัญชี',
  secondaryLabel,
  onNoAccountClick,
  onSecondaryClick,
  compact = false,
  hero = false,
  id,
  testId,
  initialElapsedMs,
  animate = true,
}) => {
  const isSuccess = status === 'success';
  const elapsedMs = useKineticFrameClock(animate, initialElapsedMs, status);
  const frame = computeKineticAuthFrame(elapsedMs, status);
  const resolvedSubtitle =
    subtitle ??
    (isSuccess
      ? 'เข้าสู่ระบบสำเร็จ กำลังเปิดหน้าจัดการคลินิก...'
      : 'ไม่พบชื่อผู้ใช้หรือรหัสผ่านนี้ในระบบคลินิก');

  const svgSizeClass = hero
    ? 'w-44 h-44 sm:w-52 sm:h-52 mx-auto overflow-visible'
    : compact
      ? 'w-24 h-24 mx-auto overflow-visible'
      : 'w-28 h-28 md:w-32 md:h-32 mx-auto overflow-visible';

  return (
    <div
      id={id}
      data-testid={testId}
      data-elapsed-ms={Math.round(frame.elapsedMs)}
      role={isSuccess ? 'status' : 'alert'}
      aria-live={isSuccess ? 'polite' : 'assertive'}
      className={
        isSuccess
          ? `flex flex-col items-center justify-center text-center ${
              hero ? 'py-2 px-4' : compact ? 'py-3 px-4' : 'py-6 px-4 sm:py-8'
            }`
          : `flex flex-col items-center justify-center text-center rounded-2xl border-2 border-red-200 bg-gradient-to-b from-red-50/90 to-white shadow-sm ${
              compact ? 'p-4' : 'p-5 sm:p-6'
            }`
      }
    >
      <div className="relative inline-flex items-center justify-center">
        <KineticEmblemSvg
          status={status}
          className={svgSizeClass}
          initialElapsedMs={elapsedMs}
          animate={false}
        />
      </div>

      {isSuccess ? (
        <div className={hero ? 'mt-3 flex flex-col items-center' : 'mt-2 flex flex-col items-center'}>
          <div
            className={`${
              hero ? 'mt-2 min-h-[4rem]' : 'mt-3 min-h-[2.75rem]'
            } flex items-center justify-center`}
          >
            <span
              data-testid="kinetic-success-word"
              aria-label={KINETIC_SUCCESS_WORD}
              style={{
                color: '#047857',
                textShadow: hero ? '0 8px 24px rgba(16, 185, 129, 0.24)' : undefined,
              }}
              className={`${
                hero
                  ? 'text-5xl sm:text-6xl tracking-tight'
                  : compact
                    ? 'text-2xl sm:text-3xl tracking-wide'
                    : 'text-3xl md:text-4xl tracking-wide'
              } font-black text-emerald-700`}
            >
              {frame.typedText}
            </span>
            {frame.isTypingActive && (
              <span
                aria-hidden="true"
                className={`inline-block w-1 ${
                  hero ? 'h-11 sm:h-12 ml-1.5' : 'w-0.5 h-7 ml-1'
                } rounded-full bg-emerald-500 animate-pulse`}
              />
            )}
          </div>
          {resolvedSubtitle && (
            <p className="mt-1.5 text-sm font-bold text-slate-600">{resolvedSubtitle}</p>
          )}
        </div>
      ) : (
        <div className="mt-2 flex flex-col items-center">
          <h3
            data-testid="kinetic-not-found-heading"
            style={{ color: '#b91c1c' }}
            className={`${
              compact ? 'text-xl sm:text-2xl' : 'text-2xl md:text-3xl'
            } font-black text-red-700 tracking-tight`}
          >
            {heading ?? 'ไม่พบบัญชี'}
          </h3>
          {resolvedSubtitle && (
            <p className="mt-1 text-xs sm:text-sm font-semibold text-red-600/90 max-w-sm">
              {resolvedSubtitle}
            </p>
          )}

          {(onNoAccountClick || onSecondaryClick) && (
            <div className="mt-4 flex flex-wrap items-center justify-center gap-2.5">
              {onNoAccountClick && (
                <button
                  type="button"
                  onClick={onNoAccountClick}
                  className="inline-flex items-center gap-2 rounded-2xl bg-red-600 hover:bg-red-700 active:scale-95 px-6 py-3 text-sm font-black text-white shadow-lg shadow-red-600/25 transition-all"
                >
                  <span>{noAccountLabel}</span>
                </button>
              )}
              {onSecondaryClick && secondaryLabel && (
                <button
                  type="button"
                  onClick={onSecondaryClick}
                  className="inline-flex items-center gap-2 rounded-2xl border-2 border-red-200 bg-white hover:bg-red-50 active:scale-95 px-5 py-2.5 text-sm font-black text-red-700 shadow-sm transition-all"
                >
                  <span>{secondaryLabel}</span>
                </button>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
};
