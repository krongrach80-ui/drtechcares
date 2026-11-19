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
        : 'from-[#0284C7] to-[#10B981]';

  return (
    <div className={embedded ? '' : 'glass-card rounded-3xl p-4 sm:p-5'}>
      <div className="flex items-center justify-between">
        <div>
          <p className="text-xs font-bold uppercase tracking-wider text-[#0284C7]">Range of Motion</p>
          <h3 className="text-base font-bold text-[#0F4C81]">มุมการเคลื่อนไหวข้อต่อ</h3>
        </div>
        <span
          className={`tabular-mono rounded-full px-3 py-1 text-xs font-bold ${
            highlightColor === 'AMBER'
              ? 'bg-amber-100 text-amber-800'
              : highlightColor === 'RED'
                ? 'bg-rose-100 text-rose-700'
                : 'bg-emerald-50 text-[#0D9488]'
          }`}
        >
          {Math.min(100, exactPct)}% TARGET
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
        </svg>

        {/* Center Numeric Readout */}
        <div className="-mt-16 flex flex-col items-center text-center">
          <div className="tabular-mono text-4xl font-extrabold tracking-tight text-[#0F4C81] sm:text-5xl">
            {displayedRomInt}°
          </div>
          <p className="mt-1 text-xs font-semibold text-[#0F4C81]/70">
            เป้าหมาย <span className="tabular-mono font-bold text-[#0D9488]">{safeTargetRom.toFixed(0)}°</span>
          </p>
        </div>
      </div>

      {/* Completion Bar (GPU scaleX transform for zero-lag 120 FPS response) */}
      <div className="mt-4">
        <div className="h-2.5 w-full overflow-hidden rounded-full bg-sky-100">
          <div
            className={`h-full w-full origin-left rounded-full bg-gradient-to-r ${barColorClass}`}
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

