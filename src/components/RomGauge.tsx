import React from 'react';
import type { HighlightColor, KioskState } from '../hooks/useKioskSocket';

interface RomGaugeProps {
  currentRomDeg: number;
  targetRomDeg: number;
  completionPct: number;
  kioskState: KioskState;
  highlightColor: HighlightColor;
  holdRemainingSec: number;
  holdTargetSec: number;
  embedded?: boolean;
}

export function RomGauge({
  currentRomDeg,
  targetRomDeg,
  completionPct,
  kioskState,
  highlightColor,
  holdRemainingSec,
  holdTargetSec,
  embedded = false,
}: RomGaugeProps) {
  const safeCurrentRom = Number.isFinite(currentRomDeg) ? currentRomDeg : 0;
  const safeTargetRom = Number.isFinite(targetRomDeg) ? targetRomDeg : 90;
  const safeCompletionPct = Number.isFinite(completionPct) ? completionPct : 0;
  const safeHoldRemaining = Number.isFinite(holdRemainingSec) ? holdRemainingSec : 0;
  const safeHoldTarget = Number.isFinite(holdTargetSec) ? holdTargetSec : 0;

  const clampedInputRom = Math.max(0, Math.min(180, safeCurrentRom));
  const displayedRomInt = Math.round(clampedInputRom);

  const radius = 104;
  const semiCircumference = Math.PI * radius;
  const clampedRatio = Math.max(0, Math.min(1, clampedInputRom / Math.max(10, safeTargetRom)));
  const dashOffset = semiCircumference * (1 - clampedRatio);
  const exactPct = Math.max(
    safeCompletionPct,
    Math.round((displayedRomInt / Math.max(10, safeTargetRom)) * 100),
  );
  const targetAchieved = exactPct >= 100 && highlightColor !== 'RED';

  const outerRadius = 122;
  const outerSemi = Math.PI * outerRadius;
  const holdRatio =
    kioskState === 'HOLD_PHASE' && safeHoldTarget > 0
      ? Math.max(0, Math.min(1, (safeHoldTarget - safeHoldRemaining) / safeHoldTarget))
      : safeHoldTarget > 0 && safeHoldRemaining > 0 && safeHoldRemaining < safeHoldTarget
        ? Math.max(0, Math.min(1, (safeHoldTarget - safeHoldRemaining) / safeHoldTarget))
        : 0;

  const barColorClass =
    highlightColor === 'RED'
      ? 'from-rose-500 to-red-600'
      : highlightColor === 'AMBER'
        ? 'from-amber-400 to-amber-500'
        : targetAchieved
          ? 'from-emerald-500 via-teal-400 to-amber-400'
          : 'from-[#0284C7] to-[#10B981]';

  return (
    <div
      className={
        embedded
          ? ''
          : `glass-card game-card-surface rounded-3xl p-4 sm:p-5 transition-all ${
              targetAchieved ? 'ring-2 ring-emerald-300/80' : ''
            }`
      }
    >
      <div className="flex items-center justify-between gap-2">
        <div>
          <p className="text-xs font-bold uppercase tracking-wider text-[#0284C7]">Range of Motion</p>
          <h3 className="text-base font-bold text-[#0F4C81]">มุมการเคลื่อนไหวข้อต่อ</h3>
        </div>
        <span
          className={`tabular-mono relative overflow-hidden inline-flex items-center gap-1 rounded-full px-3 py-1 text-xs font-black transition-colors ${
            highlightColor === 'AMBER'
              ? 'bg-amber-100 text-amber-800 border border-amber-300'
              : highlightColor === 'RED'
                ? 'bg-rose-100 text-rose-700 border border-rose-300'
                : targetAchieved
                  ? 'game-badge-shimmer bg-gradient-to-r from-emerald-500 to-teal-500 text-white shadow-sm'
                  : 'bg-emerald-50 text-[#0D9488] border border-emerald-200'
          }`}
        >
          {targetAchieved && (
            <svg viewBox="0 0 16 16" className="h-3.5 w-3.5 text-amber-200 shrink-0" fill="currentColor" aria-hidden="true">
              <path d="M8 1.2 9.9 5.5 14.6 6 11 9.1 12.1 13.8 8 11.3 3.9 13.8 5 9.1 1.4 6 6.1 5.5 Z" />
            </svg>
          )}
          <span>{Math.min(100, exactPct)}% TARGET</span>
        </span>
      </div>

      {/* Semi-Circular Radial Gauge (120Hz Direct GPU Render) */}
      <div className="relative mt-2 flex flex-col items-center">
        <svg viewBox="0 0 280 165" className="w-full max-w-[270px] overflow-visible">
          <defs>
            <linearGradient id="romGaugeGrad" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor={highlightColor === 'AMBER' ? '#F59E0B' : '#0284C7'} />
              <stop
                offset="100%"
                stopColor={
                  highlightColor === 'RED'
                    ? '#EF4444'
                    : highlightColor === 'AMBER'
                      ? '#FBBF24'
                      : '#10B981'
                }
              />
            </linearGradient>
          </defs>

          {/* Outer Hold-Phase Countdown Ring */}
          {(kioskState === 'HOLD_PHASE' || holdRatio > 0) && (
            <>
              <path
                d="M 18 144 A 122 122 0 0 1 262 144"
                fill="none"
                stroke="#CCFBF1"
                strokeWidth="7"
                strokeLinecap="round"
              />
              <path
                d="M 18 144 A 122 122 0 0 1 262 144"
                fill="none"
                stroke="#10B981"
                strokeWidth="7"
                strokeLinecap="round"
                strokeDasharray={outerSemi}
                className="transition-[stroke-dashoffset,stroke] duration-200 ease-out"
                style={{
                  strokeDashoffset: outerSemi * (1 - holdRatio),
                  willChange: 'stroke-dashoffset',
                }}
              />
            </>
          )}

          {/* Track Arc */}
          <path
            d="M 36 144 A 104 104 0 0 1 244 144"
            fill="none"
            stroke="#E0F2FE"
            strokeWidth="18"
            strokeLinecap="round"
          />

          {/* Progress Arc */}
          <path
            d="M 36 144 A 104 104 0 0 1 244 144"
            fill="none"
            stroke="url(#romGaugeGrad)"
            strokeWidth="18"
            strokeLinecap="round"
            strokeDasharray={semiCircumference}
            className="transition-[stroke-dashoffset,stroke] duration-200 ease-out"
            style={{
              strokeDashoffset: dashOffset,
              willChange: 'stroke-dashoffset',
            }}
          />

          {/* Glossy Highlight Arc Overlay */}
          <path
            d="M 36 144 A 104 104 0 0 1 244 144"
            fill="none"
            stroke="rgba(255,255,255,0.34)"
            strokeWidth="4.5"
            strokeLinecap="round"
            strokeDasharray={semiCircumference}
            className="transition-[stroke-dashoffset] duration-200 ease-out"
            style={{
              strokeDashoffset: dashOffset,
            }}
          />

          {/* Target Checkpoint Star Badge at 100% Tip */}
          <g transform="translate(244, 144)" aria-hidden="true">
            <circle
              r={targetAchieved ? 13 : 9}
              fill={targetAchieved ? '#fef3c7' : '#ffffff'}
              stroke={targetAchieved ? '#f59e0b' : '#cbd5e1'}
              strokeWidth="2.2"
              className="transition-all duration-300"
            />
            <path
              d="M0 -6.2 L1.7 -2 L6.1 -1.5 L2.8 1.5 L3.7 5.8 L0 3.6 L-3.7 5.8 L-2.8 1.5 L-6.1 -1.5 L-1.7 -2 Z"
              fill={targetAchieved ? '#f59e0b' : '#94a3b8'}
              className={targetAchieved ? 'game-star-twinkle' : ''}
            />
          </g>
        </svg>

        {/* Center Numeric Readout */}
        <div className="-mt-16 flex flex-col items-center text-center">
          <div
            className={`tabular-mono text-4xl font-extrabold tracking-tight sm:text-5xl transition-colors ${
              targetAchieved ? 'text-emerald-600' : 'text-[#0F4C81]'
            }`}
          >
            {displayedRomInt}°
          </div>
          <p className="mt-1 text-xs font-semibold text-[#0F4C81]/70">
            เป้าหมาย <span className="tabular-mono font-bold text-[#0D9488]">{safeTargetRom.toFixed(0)}°</span>
            {targetAchieved ? (
              <span className="ml-1.5 inline-flex items-center rounded-full bg-emerald-100 px-2 py-0.5 text-[10px] font-black text-emerald-700">
                ถึงเป้าแล้ว!
              </span>
            ) : null}
          </p>
        </div>
      </div>

      {/* Completion Bar (GPU scaleX transform for zero-lag 120 FPS response) */}
      <div className="mt-4">
        <div className="h-3 w-full overflow-hidden rounded-full bg-sky-100 border border-sky-200/70 shadow-inner">
          <div
            className={`game-progress-stripes h-full w-full origin-left rounded-full bg-gradient-to-r ${barColorClass}`}
            style={{
              transform: `scaleX(${clampedRatio})`,
              willChange: 'transform',
            }}
          />
        </div>
      </div>
    </div>
  );
}
