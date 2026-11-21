import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useClinicalI18n, type ClinicalLang } from '@drtechcares/clinical-i18n';

export function formatStandbyClockHHMM(date: Date): string {
  const hours = String(date.getHours()).padStart(2, '0');
  const minutes = String(date.getMinutes()).padStart(2, '0');
  return `${hours}:${minutes}`;
}

export function createSingleFireStandbyWake(onStart: () => void): () => void {
  let fired = false;
  return () => {
    if (fired) return;
    fired = true;
    onStart();
  };
}

export function sampleStandbySilkWaveY(
  xNorm: number,
  strandIndex: number,
  strandCount: number,
  timeSec: number,
  bandCenterY: number,
  bandHeight: number,
  secondary = false,
): number {
  const safeCount = Math.max(2, strandCount);
  const ratio = strandIndex / (safeCount - 1) - 0.5;
  const speed = secondary ? -0.78 : 0.94;
  const freq1 = secondary ? 5.1 : 6.2;
  const freq2 = secondary ? 10.6 : 12.1;
  const phase = timeSec * speed + ratio * (secondary ? 2.4 : 2.9);

  const primaryWave = Math.sin(xNorm * freq1 + phase) * (0.28 * bandHeight);
  const secondaryWave =
    Math.sin(xNorm * freq2 - timeSec * (speed * 0.62) + ratio * 4.2) * (0.11 * bandHeight);
  const swellWave =
    Math.cos(xNorm * 3.1 + timeSec * (secondary ? -0.44 : 0.48) + ratio * 1.4) * (0.09 * bandHeight);
  const strandSpread =
    ratio * bandHeight * (0.56 + 0.16 * Math.sin(xNorm * Math.PI + timeSec * 0.65));

  return bandCenterY + strandSpread + primaryWave + secondaryWave + swellWave;
}

export function computeStandbyFloatTransform(
  timeSec: number,
  cfg: {
    ampX: number;
    ampY: number;
    ampRotDeg: number;
    ampScale: number;
    freqX: number;
    freqY: number;
    phase: number;
  },
): { x: number; y: number; rotDeg: number; scale: number; opacity: number } {
  const x = Math.sin(timeSec * cfg.freqX + cfg.phase) * cfg.ampX;
  const y = Math.cos(timeSec * cfg.freqY + cfg.phase * 1.37) * cfg.ampY;
  const rotDeg = Math.sin(timeSec * (cfg.freqX * 0.72) + cfg.phase * 0.8) * cfg.ampRotDeg;
  const breathWave = Math.sin(timeSec * (cfg.freqY * 1.15) + cfg.phase * 1.9);
  const scale = 1 + breathWave * cfg.ampScale;
  const opacity = 0.82 + 0.18 * (0.5 + 0.5 * breathWave);

  return { x, y, rotDeg, scale, opacity };
}

export function computeHeartbeatScale(timeSec: number, cycleSec = 2.4): number {
  const safeCycle = Math.max(0.5, cycleSec);
  const p = (((timeSec % safeCycle) + safeCycle) % safeCycle) / safeCycle;
  const peak1 = Math.exp(-Math.pow((p - 0.14) / 0.045, 2)) * 0.075;
  const peak2 = Math.exp(-Math.pow((p - 0.28) / 0.055, 2)) * 0.045;
  return 1 + peak1 + peak2;
}

export function computeEcgSweepProgress(
  timeSec: number,
  cycleSec: number,
  phaseOffset = 0,
): number {
  const safeCycle = Math.max(0.5, cycleSec);
  const shifted = timeSec / safeCycle + phaseOffset;
  return ((shifted % 1) + 1) % 1;
}

type SegmentKey = 'a' | 'b' | 'c' | 'd' | 'e' | 'f' | 'g';

const SEGMENT_MAP: Record<string, readonly SegmentKey[]> = {
  '0': ['a', 'b', 'c', 'd', 'e', 'f'],
  '1': ['b', 'c'],
  '2': ['a', 'b', 'g', 'e', 'd'],
  '3': ['a', 'b', 'g', 'c', 'd'],
  '4': ['f', 'g', 'b', 'c'],
  '5': ['a', 'f', 'g', 'c', 'd'],
  '6': ['a', 'f', 'g', 'e', 'c', 'd'],
  '7': ['a', 'b', 'c'],
  '8': ['a', 'b', 'c', 'd', 'e', 'f', 'g'],
  '9': ['a', 'b', 'c', 'd', 'f', 'g'],
};

const SEGMENT_POLYGONS: Record<SegmentKey, string> = {
  a: '9,4 35,4 31,10 13,10',
  b: '37,6 41,10 41,34 37,37 33,33 33,12',
  c: '37,40 41,43 41,66 37,70 33,65 33,44',
  d: '13,66 31,66 35,72 9,72',
  e: '7,40 11,44 11,65 7,70 3,66 3,43',
  f: '7,6 11,12 11,33 7,37 3,34 3,10',
  g: '11,34 33,34 36,38 33,42 11,42 8,38',
};

export function SevenSegmentDigit({
  digit,
  className = '',
}: {
  digit: string;
  className?: string;
}) {
  const activeSegments = SEGMENT_MAP[digit] ?? SEGMENT_MAP['0'];

  return (
    <svg
      viewBox="0 0 44 76"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
      data-testid="seven-segment-digit"
      data-digit={digit}
      className={`h-11 w-6 sm:h-12 sm:w-7 md:h-14 md:w-8 ${className}`}
      style={{
        filter: 'drop-shadow(0 0 8px rgba(165, 243, 252, 0.55))',
      }}
    >
      {activeSegments.map((key) => (
        <polygon key={key} points={SEGMENT_POLYGONS[key]} fill="#ffffff" />
      ))}
    </svg>
  );
}

export function SevenSegmentClock({
  timeText,
  colonRef,
}: {
  timeText: string;
  colonRef?: React.RefObject<HTMLDivElement | null>;
}) {
  const digits = timeText.replace(':', '').slice(0, 4).padEnd(4, '0').split('');

  return (
    <div
      data-testid="kiosk-standby-clock"
      data-time={timeText}
      aria-label={timeText}
      className="inline-flex items-center justify-center gap-1.5 sm:gap-2 select-none"
    >
      <SevenSegmentDigit digit={digits[0]} />
      <SevenSegmentDigit digit={digits[1]} />
      <div
        ref={colonRef}
        aria-hidden="true"
        className="standby-gpu-layer flex flex-col items-center justify-center gap-2.5 sm:gap-3 px-0.5 sm:px-1"
        style={{
          filter: 'drop-shadow(0 0 8px rgba(165, 243, 252, 0.6))',
        }}
      >
        <span className="block w-1.5 h-1.5 sm:w-2 sm:h-2 rounded-[2px] bg-white" />
        <span className="block w-1.5 h-1.5 sm:w-2 sm:h-2 rounded-[2px] bg-white" />
      </div>
      <SevenSegmentDigit digit={digits[2]} />
      <SevenSegmentDigit digit={digits[3]} />
    </div>
  );
}

const STANDBY_COPY: Record<
  ClinicalLang,
  {
    ctaLabel: string;
    tagline: string;
    ariaWake: string;
  }
> = {
  th: {
    ctaLabel: 'แตะหน้าจอเพื่อเริ่มต้นใช้งาน',
    tagline: 'ระบบฟื้นฟูกายภาพบำบัดอัจฉริยะ',
    ariaWake: 'แตะหน้าจอเพื่อเริ่มต้นใช้งานระบบสแกนใบหน้า Dr.Tech.Care',
  },
  en: {
    ctaLabel: 'Tap Screen to Get Started',
    tagline: 'Intelligent Physical Rehabilitation System',
    ariaWake: 'Tap anywhere to start Dr.Tech.Care face scan',
  },
  cn: {
    ctaLabel: '点击屏幕开始使用',
    tagline: '智能物理康复治疗系统',
    ariaWake: '点击屏幕开始 Dr.Tech.Care 人脸识别',
  },
};

function buildPrimarySilkWaves() {
  return Array.from({ length: 14 }, (_, idx) => {
    const offset = idx * 14 - 84;
    const ctrlA = 320 + idx * 16;
    const ctrlB = 820 - idx * 12;
    const yStart = 240 + offset * 0.45;
    const yMid1 = 90 + offset * 0.85;
    const yMid2 = 415 - offset * 0.72;
    const yEnd = 225 + offset * 0.52;
    const opacity = (0.04 + (1 - Math.abs(idx - 6.5) / 7) * 0.1).toFixed(3);
    return {
      key: idx,
      d: `M -60 ${yStart} C ${ctrlA} ${yMid1}, ${ctrlB} ${yMid2}, 1260 ${yEnd}`,
      opacity,
    };
  });
}

function buildSecondarySilkWaves() {
  return Array.from({ length: 8 }, (_, idx) => {
    const offset = idx * 18 - 54;
    const opacity = (0.03 + (1 - Math.abs(idx - 3.5) / 4) * 0.08).toFixed(3);
    return {
      key: idx,
      d: `M -60 ${310 + offset * 0.5} C 360 ${440 - offset * 0.75}, 780 ${110 + offset * 0.8}, 1260 ${290 - offset * 0.45}`,
      opacity,
    };
  });
}

const DEFAULT_PRIMARY_SILK_WAVES = buildPrimarySilkWaves();
const DEFAULT_SECONDARY_SILK_WAVES = buildSecondarySilkWaves();
const ECG_TOP_PATH =
  'M 0 64 L 78 64 L 94 46 L 106 72 L 122 14 L 138 96 L 154 52 L 166 64 L 194 64 L 208 48 L 218 70 L 232 28 L 246 86 L 260 56 L 272 64 L 360 64';
const ECG_BOTTOM_PATH =
  'M 0 70 L 42 70 L 58 54 L 70 76 L 86 24 L 102 102 L 118 58 L 132 70 L 188 70 L 204 54 L 218 76 L 236 16 L 254 108 L 272 56 L 288 70 L 460 70';

const PLUS_FLOAT_CONFIGS = [
  { ampX: 7, ampY: 13, ampRotDeg: 4.5, ampScale: 0.055, freqX: 0.42, freqY: 0.56, phase: 0.2 },
  { ampX: 9, ampY: 11, ampRotDeg: 5.2, ampScale: 0.048, freqX: 0.36, freqY: 0.49, phase: 1.7 },
  { ampX: 6, ampY: 12, ampRotDeg: 4.8, ampScale: 0.052, freqX: 0.46, freqY: 0.61, phase: 3.1 },
  { ampX: 8, ampY: 14, ampRotDeg: 3.8, ampScale: 0.058, freqX: 0.39, freqY: 0.52, phase: 4.4 },
] as const;

interface BokehSeed {
  xPhase: number;
  yOffset: number;
  speedY: number;
  speedX: number;
  swayPx: number;
  radius: number;
  alpha: number;
}

const BOKEH_SEEDS: readonly BokehSeed[] = Array.from({ length: 26 }, (_, i) => ({
  xPhase: ((i * 37) % 100) / 100,
  yOffset: ((i * 53) % 100) / 100,
  speedY: 0.022 + (i % 5) * 0.0055,
  speedX: 0.35 + (i % 4) * 0.14,
  swayPx: 14 + (i % 6) * 4.5,
  radius: 1.3 + (i % 4) * 0.85,
  alpha: 0.22 + (i % 5) * 0.09,
}));

export interface KioskStandbyMotionRefs {
  waveCanvasRef: React.RefObject<HTMLCanvasElement | null>;
  orbTopRef: React.RefObject<HTMLDivElement | null>;
  orbBottomRef: React.RefObject<HTMLDivElement | null>;
  plusRefs: [
    React.RefObject<HTMLDivElement | null>,
    React.RefObject<HTMLDivElement | null>,
    React.RefObject<HTMLDivElement | null>,
    React.RefObject<HTMLDivElement | null>,
  ];
  ecgTopPathRef: React.RefObject<SVGPathElement | null>;
  ecgTopGlowPathRef: React.RefObject<SVGPathElement | null>;
  ecgTopCometGroupRef: React.RefObject<SVGGElement | null>;
  ecgBottomPathRef: React.RefObject<SVGPathElement | null>;
  ecgBottomGlowPathRef: React.RefObject<SVGPathElement | null>;
  ecgBottomCometGroupRef: React.RefObject<SVGGElement | null>;
  brandLockupRef: React.RefObject<HTMLElement | null>;
  emblemRef: React.RefObject<HTMLDivElement | null>;
  emblemSheenRef: React.RefObject<HTMLDivElement | null>;
  ctaWrapperRef: React.RefObject<HTMLDivElement | null>;
  ctaGlowRef: React.RefObject<HTMLDivElement | null>;
  ctaSheenRef: React.RefObject<HTMLDivElement | null>;
  handSvgRef: React.RefObject<SVGSVGElement | null>;
  handRippleInnerRef: React.RefObject<SVGPathElement | null>;
  handRippleOuterRef: React.RefObject<SVGPathElement | null>;
  colonRef: React.RefObject<HTMLDivElement | null>;
}

function GlowingMedicalPlusSvg({
  idSuffix,
  className,
}: {
  idSuffix: string;
  className?: string;
}) {
  const fillGradId = `plusFill-${idSuffix}`;
  const glowFilterId = `plusGlow-${idSuffix}`;

  return (
    <svg
      viewBox="0 0 120 120"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
      data-testid="standby-glow-plus"
      className={className}
    >
      <defs>
        <linearGradient id={fillGradId} x1="20" y1="16" x2="100" y2="104" gradientUnits="userSpaceOnUse">
          <stop offset="0%" stopColor="rgba(186, 242, 255, 0.44)" />
          <stop offset="100%" stopColor="rgba(103, 232, 249, 0.16)" />
        </linearGradient>
        <filter id={glowFilterId} x="-35%" y="-35%" width="170%" height="170%">
          <feGaussianBlur stdDeviation="4.5" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>
      <path
        d="M45 18 C45 14.5 47.5 12 51 12 L69 12 C72.5 12 75 14.5 75 18 L75 45 L102 45 C105.5 45 108 47.5 108 51 L108 69 C108 72.5 105.5 75 102 75 L75 75 L75 102 C75 105.5 72.5 108 69 108 L51 108 C47.5 108 45 105.5 45 102 L45 75 L18 75 C14.5 75 12 72.5 12 69 L12 51 C12 47.5 14.5 45 18 45 L45 45 Z"
        fill={`url(#${fillGradId})`}
        stroke="#a5f3fc"
        strokeWidth="2.5"
        filter={`url(#${glowFilterId})`}
      />
    </svg>
  );
}

function TappingHandOutlineSvg({
  className,
  svgRef,
  rippleInnerRef,
  rippleOuterRef,
}: {
  className?: string;
  svgRef?: React.RefObject<SVGSVGElement | null>;
  rippleInnerRef?: React.RefObject<SVGPathElement | null>;
  rippleOuterRef?: React.RefObject<SVGPathElement | null>;
}) {
  return (
    <svg
      ref={svgRef}
      viewBox="0 0 64 64"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
      className={`standby-gpu-layer ${className ?? ''}`}
    >
      <g stroke="#ffffff" strokeWidth="2.2" strokeLinecap="round">
        <path
          ref={rippleInnerRef}
          d="M20.5 14.5 A8.5 8.5 0 0 1 36.5 14.5"
          opacity="0.92"
        />
        <path
          ref={rippleOuterRef}
          d="M16 10.5 A13.8 13.8 0 0 1 41 10.5"
          opacity="0.62"
        />
      </g>
      <path
        d="M28.5 14 C26.3 14 24.5 15.8 24.5 18 V35.2 L20.6 31.6 C18.9 30.1 16.3 30.3 15 32.1 C13.8 33.8 14.2 36.2 15.9 37.6 L26 47.4 C28 49.4 30.8 50.5 33.8 50.5 H41 C46.2 50.5 49.5 46.7 49.5 41.2 V28.5 C49.5 26.4 47.9 24.8 45.8 24.8 C44.5 24.8 43.3 25.4 42.6 26.4 V25.2 C42.6 23.1 40.9 21.5 38.8 21.5 C37.5 21.5 36.3 22.1 35.6 23.1 V22.4 C35.6 20.3 34 18.8 32.5 18.8 V18 C32.5 15.8 30.7 14 28.5 14 Z"
        stroke="#ffffff"
        strokeWidth="2.4"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <line x1="32.5" y1="22.4" x2="32.5" y2="32" stroke="#ffffff" strokeWidth="2.1" strokeLinecap="round" />
      <line x1="38.8" y1="25.2" x2="38.8" y2="33.2" stroke="#ffffff" strokeWidth="2.1" strokeLinecap="round" />
      <line x1="44.8" y1="26.8" x2="44.8" y2="34.5" stroke="#ffffff" strokeWidth="2.1" strokeLinecap="round" />
    </svg>
  );
}

export function KioskStandbyView({
  timeText,
  lang = 'th',
  onWake,
  wakeRipple = null,
  isWaking = false,
  motionRefs,
}: {
  timeText: string;
  lang?: ClinicalLang;
  onWake: (e?: { clientX?: number; clientY?: number }) => void;
  wakeRipple?: { xPct: number; yPct: number } | null;
  isWaking?: boolean;
  motionRefs?: KioskStandbyMotionRefs;
}) {
  const copy = STANDBY_COPY[lang] ?? STANDBY_COPY.th;

  return (
    <div
      data-testid="kiosk-standby-screen"
      data-no-ripple="true"
      data-no-lift="true"
      onClick={(e) => onWake(e)}
      aria-label={copy.ariaWake}
      className={`animate-view-enter standby-gpu-layer min-h-screen w-full relative overflow-hidden flex flex-col items-center justify-between px-6 py-10 sm:py-12 select-none cursor-pointer text-white transition-all duration-250 ease-out ${
        isWaking ? 'opacity-0 scale-[1.035]' : 'opacity-100 scale-100'
      }`}
      style={{
        background: 'linear-gradient(108deg, #052e6e 0%, #094b8f 34%, #0b749c 68%, #10a6ab 100%)',
      }}
    >
      {/* Ambient Radial Soft Lighting */}
      <div
        ref={motionRefs?.orbTopRef}
        aria-hidden="true"
        className="standby-gpu-layer pointer-events-none absolute -top-24 -right-24 w-[440px] h-[440px] rounded-full opacity-35 blur-3xl"
        style={{ background: 'radial-gradient(circle, #38bdf8 0%, transparent 72%)' }}
      />
      <div
        ref={motionRefs?.orbBottomRef}
        aria-hidden="true"
        className="standby-gpu-layer pointer-events-none absolute -bottom-28 -left-24 w-[480px] h-[480px] rounded-full opacity-30 blur-3xl"
        style={{ background: 'radial-gradient(circle, #2dd4bf 0%, transparent 72%)' }}
      />

      {/* Live 60FPS Fluid Silk Ribbons + Bokeh Micro-Particles Canvas */}
      <canvas
        ref={motionRefs?.waveCanvasRef}
        data-testid="standby-fluid-wave-canvas"
        aria-hidden="true"
        className="pointer-events-none absolute inset-0 h-full w-full z-[1]"
      />

      {/* Subtle SSR Fallback Silk Waves */}
      <svg
        aria-hidden="true"
        viewBox="0 0 1200 540"
        preserveAspectRatio="none"
        className="pointer-events-none absolute inset-x-[-4%] top-[24%] h-[56%] w-[108%]"
      >
        <defs>
          <linearGradient id="standbySilkGradA" x1="0" y1="0" x2="1200" y2="540" gradientUnits="userSpaceOnUse">
            <stop offset="0%" stopColor="#93c5fd" />
            <stop offset="50%" stopColor="#e0f2fe" />
            <stop offset="100%" stopColor="#99f6e4" />
          </linearGradient>
        </defs>
        {DEFAULT_PRIMARY_SILK_WAVES.map((w) => (
          <path
            key={w.key}
            d={w.d}
            stroke="url(#standbySilkGradA)"
            strokeWidth="1.2"
            strokeOpacity={w.opacity}
            fill="none"
          />
        ))}
      </svg>

      <svg
        aria-hidden="true"
        viewBox="0 0 1200 540"
        preserveAspectRatio="none"
        className="pointer-events-none absolute inset-x-[-4%] top-[28%] h-[52%] w-[108%]"
      >
        {DEFAULT_SECONDARY_SILK_WAVES.map((w) => (
          <path
            key={w.key}
            d={w.d}
            stroke="#bae6fd"
            strokeWidth="1.05"
            strokeOpacity={w.opacity}
            fill="none"
          />
        ))}
      </svg>

      {/* Layer 2a: Upper-Right Sweeping Telemetry ECG Waveform */}
      <svg
        aria-hidden="true"
        data-testid="standby-ecg-waveform-top"
        viewBox="0 0 360 110"
        fill="none"
        className="pointer-events-none absolute right-[2%] top-[33%] w-48 sm:w-64 md:w-72 z-[2] overflow-visible"
      >
        <path
          ref={motionRefs?.ecgTopPathRef}
          d={ECG_TOP_PATH}
          stroke="#7dd3fc"
          strokeOpacity="0.28"
          strokeWidth="2.2"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <path
          ref={motionRefs?.ecgTopGlowPathRef}
          d={ECG_TOP_PATH}
          stroke="#e0f2fe"
          strokeWidth="2.8"
          strokeLinecap="round"
          strokeLinejoin="round"
          style={{ filter: 'drop-shadow(0 0 10px rgba(125, 211, 252, 0.9))' }}
        />
        <g
          ref={motionRefs?.ecgTopCometGroupRef}
          data-testid="standby-ecg-comet-top"
          transform="translate(0, 64)"
        >
          <circle cx="0" cy="0" r="8" fill="rgba(165, 243, 252, 0.32)" />
          <circle cx="0" cy="0" r="3.4" fill="#ffffff" />
        </g>
      </svg>

      {/* Layer 2b: Lower-Left Sweeping Telemetry ECG Waveform */}
      <svg
        aria-hidden="true"
        data-testid="standby-ecg-waveform-bottom"
        viewBox="0 0 460 120"
        fill="none"
        className="pointer-events-none absolute left-0 bottom-[14%] w-72 sm:w-96 md:w-[440px] z-[2] overflow-visible"
      >
        <path
          ref={motionRefs?.ecgBottomPathRef}
          d={ECG_BOTTOM_PATH}
          stroke="#a5f3fc"
          strokeOpacity="0.28"
          strokeWidth="2.4"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <path
          ref={motionRefs?.ecgBottomGlowPathRef}
          d={ECG_BOTTOM_PATH}
          stroke="#ecfeff"
          strokeWidth="3"
          strokeLinecap="round"
          strokeLinejoin="round"
          style={{ filter: 'drop-shadow(0 0 12px rgba(165, 243, 252, 0.95))' }}
        />
        <g
          ref={motionRefs?.ecgBottomCometGroupRef}
          data-testid="standby-ecg-comet-bottom"
          transform="translate(0, 70)"
        >
          <circle cx="0" cy="0" r="9" fill="rgba(165, 243, 252, 0.35)" />
          <circle cx="0" cy="0" r="3.6" fill="#ffffff" />
        </g>
      </svg>

      {/* Layer 3: Four Weightless Floating Holographic Medical Plus Symbols */}
      <div
        ref={motionRefs?.plusRefs[0]}
        aria-hidden="true"
        className="standby-gpu-layer pointer-events-none absolute top-[5%] right-[5%] sm:right-[8%] z-[3]"
      >
        <GlowingMedicalPlusSvg idSuffix="top-right" className="w-20 h-20 sm:w-24 sm:h-24 md:w-28 md:h-28" />
      </div>
      <div
        ref={motionRefs?.plusRefs[1]}
        aria-hidden="true"
        className="standby-gpu-layer pointer-events-none absolute top-[28%] left-[10%] sm:left-[18%] z-[3]"
      >
        <GlowingMedicalPlusSvg idSuffix="upper-left" className="w-14 h-14 sm:w-16 sm:h-16 md:w-[72px] md:h-[72px]" />
      </div>
      <div
        ref={motionRefs?.plusRefs[2]}
        aria-hidden="true"
        className="standby-gpu-layer pointer-events-none absolute top-[61%] right-[9%] sm:right-[16%] z-[3]"
      >
        <GlowingMedicalPlusSvg idSuffix="lower-right" className="w-12 h-12 sm:w-14 sm:h-14 md:w-16 md:h-16" />
      </div>
      <div
        ref={motionRefs?.plusRefs[3]}
        aria-hidden="true"
        className="standby-gpu-layer pointer-events-none absolute bottom-[15%] left-[10%] sm:left-[16%] z-[3]"
      >
        <GlowingMedicalPlusSvg idSuffix="bottom-left" className="w-20 h-20 sm:w-24 sm:h-24 md:w-28 md:h-28" />
      </div>

      {/* Tap-to-Wake Expanding Ripple Overlay */}
      {wakeRipple && (
        <div
          aria-hidden="true"
          data-testid="standby-wake-ripple"
          className="animate-standby-wake-ripple pointer-events-none absolute z-30 h-56 w-56 rounded-full border-2 border-cyan-100/80"
          style={{
            left: `${wakeRipple.xPct}%`,
            top: `${wakeRipple.yPct}%`,
            background:
              'radial-gradient(circle, rgba(165, 243, 252, 0.34) 0%, rgba(56, 189, 248, 0.14) 55%, transparent 72%)',
          }}
        />
      )}

      {/* Foreground Vertical Composition */}
      {/* 1. Top Brand Lockup */}
      <header
        ref={motionRefs?.brandLockupRef}
        className="standby-gpu-layer relative z-10 mt-[8vh] sm:mt-[7vh] flex flex-wrap items-center justify-center gap-2.5 sm:gap-4 max-w-full"
      >
        <div
          ref={motionRefs?.emblemRef}
          className="standby-gpu-layer relative overflow-hidden w-12 h-12 sm:w-16 sm:h-16 md:w-[70px] md:h-[70px] rounded-2xl flex items-center justify-center ring-1 ring-white/40 shadow-[0_12px_28px_rgba(3,37,76,0.38)] shrink-0"
          style={{
            background: 'linear-gradient(135deg, #2596e8 0%, #1bc4b8 100%)',
          }}
        >
          <div
            ref={motionRefs?.emblemSheenRef}
            aria-hidden="true"
            className="standby-gpu-layer pointer-events-none absolute inset-y-0 -left-1/2 w-1/2"
            style={{
              background:
                'linear-gradient(105deg, transparent 15%, rgba(255,255,255,0.42) 50%, transparent 85%)',
            }}
          />
          <svg
            viewBox="0 0 48 48"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
            aria-hidden="true"
            className="relative z-10 w-7 h-7 sm:w-10 sm:h-10 md:w-11 md:h-11"
          >
            <path
              d="M24 40 C24 40 7.5 29.8 7.5 17.8 C7.5 12.2 11.9 8 17.2 8 C20.4 8 23.1 9.6 24 12.1 C24.9 9.6 27.6 8 30.8 8 C36.1 8 40.5 12.2 40.5 17.8 C40.5 29.8 24 40 24 40 Z"
              stroke="#ffffff"
              strokeWidth="2.6"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
            <path
              d="M11 23 H17.5 L20.5 16.5 L24.5 29.5 L28 20.5 L30.5 23 H37"
              stroke="#ffffff"
              strokeWidth="2.4"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </div>

        <h1 className="text-[26px] sm:text-4xl md:text-[46px] font-extrabold tracking-tight text-white drop-shadow-[0_2px_12px_rgba(3,28,66,0.3)] leading-none whitespace-nowrap">
          Dr.Tech.Care
        </h1>

        <span className="px-3 py-1 sm:px-4 sm:py-1.5 rounded-full bg-white/18 border border-white/30 backdrop-blur-md text-xs sm:text-base md:text-lg font-semibold text-white/95 whitespace-nowrap">
          AI Kiosk
        </span>
      </header>

      {/* 2. Middle Stack: Frosted Glass Tap CTA Pill + Live 7-Segment Digital Clock */}
      <main className="relative z-10 flex flex-col items-center justify-center w-full max-w-2xl my-auto py-8">
        <div ref={motionRefs?.ctaWrapperRef} className="standby-gpu-layer relative max-w-full">
          <div
            ref={motionRefs?.ctaGlowRef}
            aria-hidden="true"
            className="standby-gpu-layer pointer-events-none -inset-2 absolute rounded-full blur-xl opacity-55"
            style={{
              background:
                'linear-gradient(90deg, rgba(56, 189, 248, 0.42), rgba(165, 243, 252, 0.58), rgba(45, 212, 191, 0.42))',
            }}
          />
          <button
            type="button"
            data-testid="kiosk-standby-start-btn"
            onClick={(e) => {
              e.stopPropagation();
              onWake(e);
            }}
            className="group relative overflow-hidden inline-flex items-center justify-center gap-2.5 sm:gap-4 max-w-full rounded-full border-[1.5px] border-cyan-100/75 px-5 py-3.5 sm:px-10 sm:py-5 backdrop-blur-md transition-transform duration-150 hover:scale-[1.02] active:scale-[0.97] cursor-pointer shadow-[0_14px_38px_rgba(2,132,199,0.35)]"
            style={{
              background:
                'linear-gradient(180deg, rgba(145, 224, 240, 0.30) 0%, rgba(71, 168, 206, 0.24) 100%)',
            }}
          >
            <div
              ref={motionRefs?.ctaSheenRef}
              aria-hidden="true"
              className="standby-gpu-layer pointer-events-none absolute inset-y-0 -left-1/3 w-1/3"
              style={{
                background:
                  'linear-gradient(105deg, transparent 10%, rgba(255,255,255,0.28) 50%, transparent 90%)',
              }}
            />
            <TappingHandOutlineSvg
              className="relative z-10 w-8 h-8 sm:w-12 sm:h-12 shrink-0"
              svgRef={motionRefs?.handSvgRef}
              rippleInnerRef={motionRefs?.handRippleInnerRef}
              rippleOuterRef={motionRefs?.handRippleOuterRef}
            />
            <span
              className="relative z-10 text-base min-[390px]:text-lg sm:text-2xl md:text-[27px] font-bold tracking-wide text-white drop-shadow-[0_1px_6px_rgba(4,34,74,0.35)] whitespace-nowrap"
              style={{ fontFamily: "'Prompt', 'IBM Plex Sans Thai', sans-serif" }}
            >
              {copy.ctaLabel}
            </span>
          </button>
        </div>

        <div className="mt-9 sm:mt-10">
          <SevenSegmentClock timeText={timeText} colonRef={motionRefs?.colonRef} />
        </div>
      </main>

      {/* 3. Bottom System Tagline */}
      <footer className="relative z-10 mb-3 sm:mb-6">
        <p
          data-testid="kiosk-standby-tagline"
          className="text-lg sm:text-2xl md:text-[26px] font-bold tracking-wide text-white text-center drop-shadow-[0_2px_10px_rgba(4,34,74,0.45)]"
          style={{ fontFamily: "'Prompt', 'IBM Plex Sans Thai', sans-serif" }}
        >
          {copy.tagline}
        </p>
      </footer>
    </div>
  );
}

export interface KioskStandbyScreenProps {
  onStart: () => void;
  initialDate?: Date;
}

export function KioskStandbyScreen({ onStart, initialDate }: KioskStandbyScreenProps) {
  const { lang } = useClinicalI18n();
  const [timeText, setTimeText] = useState<string>(() =>
    formatStandbyClockHHMM(initialDate ?? new Date()),
  );
  const [wakeRipple, setWakeRipple] = useState<{ xPct: number; yPct: number } | null>(null);
  const [isWaking, setIsWaking] = useState(false);

  const onStartRef = useRef(onStart);
  onStartRef.current = onStart;
  const wakeTimerRef = useRef<number | null>(null);

  const singleWake = useMemo(() => createSingleFireStandbyWake(() => onStartRef.current()), []);

  const handleWake = useCallback(
    (e?: { clientX?: number; clientY?: number }) => {
      if (isWaking) return;
      if (
        typeof window !== 'undefined' &&
        e &&
        typeof e.clientX === 'number' &&
        typeof e.clientY === 'number' &&
        (e.clientX > 0 || e.clientY > 0)
      ) {
        const vw = Math.max(1, window.innerWidth);
        const vh = Math.max(1, window.innerHeight);
        setWakeRipple({
          xPct: Math.min(95, Math.max(5, (e.clientX / vw) * 100)),
          yPct: Math.min(95, Math.max(5, (e.clientY / vh) * 100)),
        });
      } else {
        setWakeRipple({ xPct: 50, yPct: 52 });
      }
      setIsWaking(true);
      if (typeof window === 'undefined') {
        singleWake();
        return;
      }
      wakeTimerRef.current = window.setTimeout(() => {
        wakeTimerRef.current = null;
        singleWake();
      }, 240);
    },
    [isWaking, singleWake],
  );

  useEffect(() => {
    return () => {
      if (wakeTimerRef.current !== null) {
        window.clearTimeout(wakeTimerRef.current);
      }
    };
  }, []);

  const waveCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const orbTopRef = useRef<HTMLDivElement | null>(null);
  const orbBottomRef = useRef<HTMLDivElement | null>(null);
  const plusRef0 = useRef<HTMLDivElement | null>(null);
  const plusRef1 = useRef<HTMLDivElement | null>(null);
  const plusRef2 = useRef<HTMLDivElement | null>(null);
  const plusRef3 = useRef<HTMLDivElement | null>(null);
  const ecgTopPathRef = useRef<SVGPathElement | null>(null);
  const ecgTopGlowPathRef = useRef<SVGPathElement | null>(null);
  const ecgTopCometGroupRef = useRef<SVGGElement | null>(null);
  const ecgBottomPathRef = useRef<SVGPathElement | null>(null);
  const ecgBottomGlowPathRef = useRef<SVGPathElement | null>(null);
  const ecgBottomCometGroupRef = useRef<SVGGElement | null>(null);
  const brandLockupRef = useRef<HTMLElement | null>(null);
  const emblemRef = useRef<HTMLDivElement | null>(null);
  const emblemSheenRef = useRef<HTMLDivElement | null>(null);
  const ctaWrapperRef = useRef<HTMLDivElement | null>(null);
  const ctaGlowRef = useRef<HTMLDivElement | null>(null);
  const ctaSheenRef = useRef<HTMLDivElement | null>(null);
  const handSvgRef = useRef<SVGSVGElement | null>(null);
  const handRippleInnerRef = useRef<SVGPathElement | null>(null);
  const handRippleOuterRef = useRef<SVGPathElement | null>(null);
  const colonRef = useRef<HTMLDivElement | null>(null);

  const motionRefs = useMemo<KioskStandbyMotionRefs>(
    () => ({
      waveCanvasRef,
      orbTopRef,
      orbBottomRef,
      plusRefs: [plusRef0, plusRef1, plusRef2, plusRef3],
      ecgTopPathRef,
      ecgTopGlowPathRef,
      ecgTopCometGroupRef,
      ecgBottomPathRef,
      ecgBottomGlowPathRef,
      ecgBottomCometGroupRef,
      brandLockupRef,
      emblemRef,
      emblemSheenRef,
      ctaWrapperRef,
      ctaGlowRef,
      ctaSheenRef,
      handSvgRef,
      handRippleInnerRef,
      handRippleOuterRef,
      colonRef,
    }),
    [],
  );

  // Unified 60FPS RequestAnimationFrame Fluid Motion Engine
  useEffect(() => {
    if (typeof window === 'undefined' || typeof window.requestAnimationFrame !== 'function') {
      return;
    }

    let rafId = 0;
    let width = Math.max(1, window.innerWidth);
    let height = Math.max(1, window.innerHeight);

    const resizeCanvas = () => {
      const canvas = waveCanvasRef.current;
      if (!canvas) return;
      width = Math.max(1, window.innerWidth);
      height = Math.max(1, window.innerHeight);
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      }
    };

    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    const ecgTopLen =
      ecgTopPathRef.current && typeof ecgTopPathRef.current.getTotalLength === 'function'
        ? ecgTopPathRef.current.getTotalLength()
        : 0;
    const ecgBottomLen =
      ecgBottomPathRef.current && typeof ecgBottomPathRef.current.getTotalLength === 'function'
        ? ecgBottomPathRef.current.getTotalLength()
        : 0;

    if (ecgTopGlowPathRef.current && ecgTopLen > 0) {
      ecgTopGlowPathRef.current.style.strokeDasharray = `${(ecgTopLen * 0.34).toFixed(1)} ${ecgTopLen.toFixed(1)}`;
    }
    if (ecgBottomGlowPathRef.current && ecgBottomLen > 0) {
      ecgBottomGlowPathRef.current.style.strokeDasharray = `${(ecgBottomLen * 0.32).toFixed(1)} ${ecgBottomLen.toFixed(1)}`;
    }

    const startMs = typeof performance !== 'undefined' ? performance.now() : Date.now();

    const drawFluidFrame = (nowMs: number) => {
      if (typeof document !== 'undefined' && document.hidden) {
        rafId = window.requestAnimationFrame(drawFluidFrame);
        return;
      }

      const t = (nowMs - startMs) / 1000;

      // 1. Canvas Liquid-Silk Multi-Strand Ribbons + Bokeh Micro-Particles
      const canvas = waveCanvasRef.current;
      const ctx = canvas?.getContext('2d');
      if (canvas && ctx) {
        ctx.clearRect(0, 0, width, height);

        const bandCenterY = height * 0.51;
        const bandHeight = Math.max(170, height * 0.3);
        const stepPx = Math.max(10, Math.round(width / 72));

        const primaryGrad = ctx.createLinearGradient(0, bandCenterY - bandHeight, width, bandCenterY + bandHeight);
        primaryGrad.addColorStop(0, 'rgba(147, 197, 253, 0.25)');
        primaryGrad.addColorStop(0.5, 'rgba(224, 242, 254, 0.65)');
        primaryGrad.addColorStop(1, 'rgba(153, 246, 228, 0.32)');

        ctx.strokeStyle = primaryGrad;
        ctx.lineWidth = 1.25;
        const primaryCount = 16;
        for (let s = 0; s < primaryCount; s++) {
          const bell = 1 - Math.abs(s / (primaryCount - 1) - 0.5) * 2;
          ctx.globalAlpha = 0.08 + bell * 0.28;
          ctx.beginPath();
          for (let x = -stepPx; x <= width + stepPx; x += stepPx) {
            const xNorm = x / width;
            const y = sampleStandbySilkWaveY(xNorm, s, primaryCount, t, bandCenterY, bandHeight, false);
            if (x <= -stepPx) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
          }
          ctx.stroke();
        }

        const secondaryCount = 10;
        ctx.strokeStyle = 'rgba(186, 230, 253, 0.72)';
        ctx.lineWidth = 1.05;
        for (let s = 0; s < secondaryCount; s++) {
          const bell = 1 - Math.abs(s / (secondaryCount - 1) - 0.5) * 2;
          ctx.globalAlpha = 0.06 + bell * 0.19;
          ctx.beginPath();
          for (let x = -stepPx; x <= width + stepPx; x += stepPx) {
            const xNorm = x / width;
            const y = sampleStandbySilkWaveY(
              xNorm,
              s,
              secondaryCount,
              t,
              bandCenterY + height * 0.025,
              bandHeight * 0.92,
              true,
            );
            if (x <= -stepPx) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
          }
          ctx.stroke();
        }

        // Drifting glowing cyan-white micro bokeh
        for (let i = 0; i < BOKEH_SEEDS.length; i++) {
          const b = BOKEH_SEEDS[i];
          const pyNorm = (((b.yOffset - t * b.speedY) % 1) + 1) % 1;
          const edgeFade = Math.sin(pyNorm * Math.PI);
          const px = b.xPhase * width + Math.sin(t * b.speedX + i) * b.swayPx;
          const py = pyNorm * height;
          ctx.globalAlpha = b.alpha * edgeFade;
          ctx.fillStyle = i % 2 === 0 ? '#e0f2fe' : '#a5f3fc';
          ctx.beginPath();
          ctx.arc(px, py, b.radius, 0, Math.PI * 2);
          ctx.fill();
        }
        ctx.globalAlpha = 1;
      }

      // 2. Real-Time Travelling ECG Telemetry Glows & Comets
      if (ecgTopPathRef.current && ecgTopGlowPathRef.current && ecgTopLen > 0) {
        const progTop = computeEcgSweepProgress(t, 3.8, 0.0);
        const dashOffsetTop = ecgTopLen * (1 - progTop);
        ecgTopGlowPathRef.current.style.strokeDashoffset = dashOffsetTop.toFixed(2);
        if (ecgTopCometGroupRef.current && typeof ecgTopPathRef.current.getPointAtLength === 'function') {
          const pt = ecgTopPathRef.current.getPointAtLength(progTop * ecgTopLen);
          ecgTopCometGroupRef.current.setAttribute('transform', `translate(${pt.x.toFixed(1)}, ${pt.y.toFixed(1)})`);
        }
      }

      if (ecgBottomPathRef.current && ecgBottomGlowPathRef.current && ecgBottomLen > 0) {
        const progBottom = computeEcgSweepProgress(t, 4.4, 0.35);
        const dashOffsetBottom = ecgBottomLen * (1 - progBottom);
        ecgBottomGlowPathRef.current.style.strokeDashoffset = dashOffsetBottom.toFixed(2);
        if (ecgBottomCometGroupRef.current && typeof ecgBottomPathRef.current.getPointAtLength === 'function') {
          const pt = ecgBottomPathRef.current.getPointAtLength(progBottom * ecgBottomLen);
          ecgBottomCometGroupRef.current.setAttribute(
            'transform',
            `translate(${pt.x.toFixed(1)}, ${pt.y.toFixed(1)})`,
          );
        }
      }

      // 3. Weightless Holographic Floating '+' Symbols
      PLUS_FLOAT_CONFIGS.forEach((cfg, idx) => {
        const el = motionRefs.plusRefs[idx].current;
        if (!el) return;
        const ft = computeStandbyFloatTransform(t, cfg);
        el.style.transform = `translate3d(${ft.x.toFixed(2)}px, ${ft.y.toFixed(2)}px, 0) rotate(${ft.rotDeg.toFixed(2)}deg) scale(${ft.scale.toFixed(4)})`;
        el.style.opacity = ft.opacity.toFixed(3);
      });

      // 4. Liquid Ambient Aurora Orbs
      if (orbTopRef.current) {
        const ox = Math.sin(t * 0.25) * 28;
        const oy = Math.cos(t * 0.32) * 22;
        const os = 1 + Math.sin(t * 0.4) * 0.08;
        orbTopRef.current.style.transform = `translate3d(${ox.toFixed(1)}px, ${oy.toFixed(1)}px, 0) scale(${os.toFixed(3)})`;
      }
      if (orbBottomRef.current) {
        const ox = Math.cos(t * 0.22) * 32;
        const oy = Math.sin(t * 0.29) * 24;
        const os = 1 + Math.cos(t * 0.35) * 0.08;
        orbBottomRef.current.style.transform = `translate3d(${ox.toFixed(1)}px, ${oy.toFixed(1)}px, 0) scale(${os.toFixed(3)})`;
      }

      // 5. Living Heartbeat Brand Lockup + Gloss Sweep
      if (brandLockupRef.current) {
        const lockupFloat = Math.sin(t * 0.72) * 2.5;
        brandLockupRef.current.style.transform = `translate3d(0, ${lockupFloat.toFixed(2)}px, 0)`;
      }
      if (emblemRef.current) {
        const hbScale = computeHeartbeatScale(t, 2.4);
        emblemRef.current.style.transform = `scale(${hbScale.toFixed(4)})`;
      }
      if (emblemSheenRef.current) {
        const sheenCycle = computeEcgSweepProgress(t, 4.2, 0.15);
        const sheenX = -120 + sheenCycle * 360;
        emblemSheenRef.current.style.transform = `translate3d(${sheenX.toFixed(1)}%, 0, 0) skewX(-18deg)`;
      }

      // 6. Silky Floating CTA Pill + Outer Breathing Halo + Shimmer + Finger Ripple
      if (ctaWrapperRef.current) {
        const ctaFloat = Math.sin(t * 0.95 + 0.6) * 3.5;
        ctaWrapperRef.current.style.transform = `translate3d(0, ${ctaFloat.toFixed(2)}px, 0)`;
      }
      if (ctaGlowRef.current) {
        const glowAlpha = 0.38 + 0.34 * (0.5 + 0.5 * Math.sin(t * 1.9));
        const glowScale = 0.98 + 0.05 * (0.5 + 0.5 * Math.sin(t * 1.9));
        ctaGlowRef.current.style.opacity = glowAlpha.toFixed(3);
        ctaGlowRef.current.style.transform = `scale(${glowScale.toFixed(3)})`;
      }
      if (ctaSheenRef.current) {
        const ctaCycle = computeEcgSweepProgress(t, 3.6, 0.4);
        const ctaX = -140 + ctaCycle * 560;
        ctaSheenRef.current.style.transform = `translate3d(${ctaX.toFixed(1)}%, 0, 0) skewX(-22deg)`;
      }
      if (handSvgRef.current) {
        const handDip = Math.max(0, Math.sin(t * 2.8)) * 2.4;
        const handScale = 1 - Math.max(0, Math.sin(t * 2.8)) * 0.06;
        handSvgRef.current.style.transform = `translate3d(0, ${handDip.toFixed(2)}px, 0) scale(${handScale.toFixed(4)})`;
      }
      if (handRippleInnerRef.current) {
        const pInner = computeEcgSweepProgress(t, 1.8, 0.0);
        const opInner = Math.sin(pInner * Math.PI);
        handRippleInnerRef.current.style.opacity = (0.25 + opInner * 0.75).toFixed(3);
      }
      if (handRippleOuterRef.current) {
        const pOuter = computeEcgSweepProgress(t, 1.8, 0.25);
        const opOuter = Math.sin(pOuter * Math.PI);
        handRippleOuterRef.current.style.opacity = (0.1 + opOuter * 0.72).toFixed(3);
      }

      // 7. Smooth Breathing Digital Colon
      if (colonRef.current) {
        const colonOpacity = 0.35 + 0.65 * (0.5 + 0.5 * Math.sin(t * 3.1));
        colonRef.current.style.opacity = colonOpacity.toFixed(3);
      }

      rafId = window.requestAnimationFrame(drawFluidFrame);
    };

    rafId = window.requestAnimationFrame(drawFluidFrame);

    return () => {
      window.removeEventListener('resize', resizeCanvas);
      if (rafId) window.cancelAnimationFrame(rafId);
    };
  }, [motionRefs]);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    const syncClock = () => {
      setTimeText(formatStandbyClockHHMM(new Date()));
    };
    const timerId = window.setInterval(syncClock, 1000);
    return () => window.clearInterval(timerId);
  }, []);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        handleWake();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleWake]);

  return (
    <KioskStandbyView
      timeText={timeText}
      lang={lang}
      onWake={handleWake}
      wakeRipple={wakeRipple}
      isWaking={isWaking}
      motionRefs={motionRefs}
    />
  );
}
