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

export function SevenSegmentClock({ timeText }: { timeText: string }) {
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
        aria-hidden="true"
        className="animate-standby-colon flex flex-col items-center justify-center gap-2.5 sm:gap-3 px-0.5 sm:px-1"
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
    const opacity = (0.07 + (1 - Math.abs(idx - 6.5) / 7) * 0.21).toFixed(3);
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
    const opacity = (0.06 + (1 - Math.abs(idx - 3.5) / 4) * 0.14).toFixed(3);
    return {
      key: idx,
      d: `M -60 ${310 + offset * 0.5} C 360 ${440 - offset * 0.75}, 780 ${110 + offset * 0.8}, 1260 ${290 - offset * 0.45}`,
      opacity,
    };
  });
}

const DEFAULT_PRIMARY_SILK_WAVES = buildPrimarySilkWaves();
const DEFAULT_SECONDARY_SILK_WAVES = buildSecondarySilkWaves();

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
          <stop offset="0%" stopColor="rgba(186, 242, 255, 0.42)" />
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

function TappingHandOutlineSvg({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 64 64"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
      className={className}
    >
      <g className="animate-standby-tap-ripple" stroke="#ffffff" strokeWidth="2.2" strokeLinecap="round">
        <path d="M20.5 14.5 A8.5 8.5 0 0 1 36.5 14.5" opacity="0.92" />
        <path d="M16 10.5 A13.8 13.8 0 0 1 41 10.5" opacity="0.62" />
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
}: {
  timeText: string;
  lang?: ClinicalLang;
  onWake: () => void;
}) {
  const copy = STANDBY_COPY[lang] ?? STANDBY_COPY.th;

  return (
    <div
      data-testid="kiosk-standby-screen"
      onClick={onWake}
      aria-label={copy.ariaWake}
      className="animate-view-enter min-h-screen w-full relative overflow-hidden flex flex-col items-center justify-between px-6 py-10 sm:py-12 select-none cursor-pointer text-white"
      style={{
        background: 'linear-gradient(108deg, #052e6e 0%, #094b8f 34%, #0b749c 68%, #10a6ab 100%)',
      }}
    >
      {/* Ambient Radial Soft Lighting */}
      <div
        aria-hidden="true"
        className="pointer-events-none absolute -top-24 -right-24 w-[420px] h-[420px] rounded-full opacity-35 blur-3xl"
        style={{ background: 'radial-gradient(circle, #38bdf8 0%, transparent 72%)' }}
      />
      <div
        aria-hidden="true"
        className="pointer-events-none absolute -bottom-28 -left-24 w-[460px] h-[460px] rounded-full opacity-30 blur-3xl"
        style={{ background: 'radial-gradient(circle, #2dd4bf 0%, transparent 72%)' }}
      />

      {/* Layer 1: Multi-Strand Silk Wave Mesh */}
      <svg
        aria-hidden="true"
        viewBox="0 0 1200 540"
        preserveAspectRatio="none"
        className="animate-standby-wave-a pointer-events-none absolute inset-x-[-4%] top-[24%] h-[56%] w-[108%]"
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
            strokeWidth="1.25"
            strokeOpacity={w.opacity}
            fill="none"
          />
        ))}
      </svg>

      <svg
        aria-hidden="true"
        viewBox="0 0 1200 540"
        preserveAspectRatio="none"
        className="animate-standby-wave-b pointer-events-none absolute inset-x-[-4%] top-[28%] h-[52%] w-[108%]"
      >
        {DEFAULT_SECONDARY_SILK_WAVES.map((w) => (
          <path
            key={w.key}
            d={w.d}
            stroke="#bae6fd"
            strokeWidth="1.1"
            strokeOpacity={w.opacity}
            fill="none"
          />
        ))}
      </svg>

      {/* Layer 2: Upper-Right Glowing ECG Waveform */}
      <svg
        aria-hidden="true"
        data-testid="standby-ecg-waveform-top"
        viewBox="0 0 360 110"
        fill="none"
        className="animate-standby-ecg pointer-events-none absolute right-[2%] top-[33%] w-48 sm:w-64 md:w-72 opacity-85"
      >
        <path
          d="M 0 64 L 78 64 L 94 46 L 106 72 L 122 14 L 138 96 L 154 52 L 166 64 L 194 64 L 208 48 L 218 70 L 232 28 L 246 86 L 260 56 L 272 64 L 360 64"
          stroke="#7dd3fc"
          strokeWidth="2.4"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>

      {/* Layer 2b: Lower-Left Glowing ECG Waveform passing behind Bottom-Left Plus */}
      <svg
        aria-hidden="true"
        data-testid="standby-ecg-waveform-bottom"
        viewBox="0 0 460 120"
        fill="none"
        className="animate-standby-ecg pointer-events-none absolute left-0 bottom-[14%] w-72 sm:w-96 md:w-[440px] opacity-90"
      >
        <path
          d="M 0 70 L 42 70 L 58 54 L 70 76 L 86 24 L 102 102 L 118 58 L 132 70 L 188 70 L 204 54 L 218 76 L 236 16 L 254 108 L 272 56 L 288 70 L 460 70"
          stroke="#a5f3fc"
          strokeWidth="2.6"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>

      {/* Layer 3: Four Glowing Holographic Medical Plus Symbols */}
      <div
        aria-hidden="true"
        className="animate-standby-float-1 pointer-events-none absolute top-[5%] right-[5%] sm:right-[8%]"
      >
        <GlowingMedicalPlusSvg idSuffix="top-right" className="w-20 h-20 sm:w-24 sm:h-24 md:w-28 md:h-28" />
      </div>
      <div
        aria-hidden="true"
        className="animate-standby-float-2 pointer-events-none absolute top-[28%] left-[10%] sm:left-[18%]"
      >
        <GlowingMedicalPlusSvg idSuffix="upper-left" className="w-14 h-14 sm:w-16 sm:h-16 md:w-[72px] md:h-[72px]" />
      </div>
      <div
        aria-hidden="true"
        className="animate-standby-float-3 pointer-events-none absolute top-[61%] right-[9%] sm:right-[16%]"
      >
        <GlowingMedicalPlusSvg idSuffix="lower-right" className="w-12 h-12 sm:w-14 sm:h-14 md:w-16 md:h-16" />
      </div>
      <div
        aria-hidden="true"
        className="animate-standby-float-4 pointer-events-none absolute bottom-[15%] left-[10%] sm:left-[16%]"
      >
        <GlowingMedicalPlusSvg idSuffix="bottom-left" className="w-20 h-20 sm:w-24 sm:h-24 md:w-28 md:h-28" />
      </div>

      {/* Foreground Vertical Composition */}
      {/* 1. Top Brand Lockup */}
      <header className="relative z-10 mt-[8vh] sm:mt-[7vh] flex flex-wrap items-center justify-center gap-2.5 sm:gap-4 max-w-full">
        <div
          className="w-12 h-12 sm:w-16 sm:h-16 md:w-[70px] md:h-[70px] rounded-2xl flex items-center justify-center ring-1 ring-white/40 shadow-[0_12px_28px_rgba(3,37,76,0.38)] shrink-0"
          style={{
            background: 'linear-gradient(135deg, #2596e8 0%, #1bc4b8 100%)',
          }}
        >
          <svg
            viewBox="0 0 48 48"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
            aria-hidden="true"
            className="w-7 h-7 sm:w-10 sm:h-10 md:w-11 md:h-11"
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
        <button
          type="button"
          data-testid="kiosk-standby-start-btn"
          onClick={(e) => {
            e.stopPropagation();
            onWake();
          }}
          className="animate-standby-cta-glow group relative inline-flex items-center justify-center gap-2.5 sm:gap-4 max-w-full rounded-full border-[1.5px] border-cyan-100/75 px-5 py-3.5 sm:px-10 sm:py-5 backdrop-blur-md transition-transform duration-150 hover:scale-[1.02] active:scale-[0.97] cursor-pointer"
          style={{
            background:
              'linear-gradient(180deg, rgba(145, 224, 240, 0.30) 0%, rgba(71, 168, 206, 0.24) 100%)',
          }}
        >
          <TappingHandOutlineSvg className="w-8 h-8 sm:w-12 sm:h-12 shrink-0" />
          <span
            className="text-base min-[390px]:text-lg sm:text-2xl md:text-[27px] font-bold tracking-wide text-white drop-shadow-[0_1px_6px_rgba(4,34,74,0.35)] whitespace-nowrap"
            style={{ fontFamily: "'Prompt', 'IBM Plex Sans Thai', sans-serif" }}
          >
            {copy.ctaLabel}
          </span>
        </button>

        <div className="mt-9 sm:mt-10">
          <SevenSegmentClock timeText={timeText} />
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

  const onStartRef = useRef(onStart);
  onStartRef.current = onStart;

  const singleWake = useMemo(() => createSingleFireStandbyWake(() => onStartRef.current()), []);

  const handleWake = useCallback(() => {
    singleWake();
  }, [singleWake]);

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

  return <KioskStandbyView timeText={timeText} lang={lang} onWake={handleWake} />;
}
