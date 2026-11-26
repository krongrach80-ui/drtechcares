import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { PoseLandmarker } from '@mediapipe/tasks-vision';
import { useClinicalI18n, type ClinicalLang } from '@drtechcares/clinical-i18n';
import { selectClinicalCameraDeviceId } from '@drtechcares/face-client';
import { Check } from 'lucide-react';
import {
  getCachedPoseLandmarker,
  getNextMonotonicPoseDetectTs,
} from '../lib/biometricVisionCache';
import { playGameSfx } from '../lib/gameUiSfx';
import { MorTechMascot } from './MorTechMascot';
import {
  INITIAL_STANDBY_HAND_HOLD_STATE,
  STANDBY_HAND_WAKE_HOLD_MS,
  STANDBY_INTRO_EXIT_MS,
  STANDBY_INTRO_MAX_MS,
  STANDBY_INTRO_MIN_MS,
  STANDBY_WELCOME_COPY,
  advanceStandbyHandHold,
  evaluateTwoHandsRaised,
  playStandbyWelcomeVoice,
  smoothVisualProgress,
  type StandbyHandHoldState,
  type TwoHandsRaiseSample,
} from '../utils/standbyHandWakeHelpers';

export {
  INITIAL_STANDBY_HAND_HOLD_STATE,
  STANDBY_HAND_WAKE_HOLD_MS,
  STANDBY_WELCOME_COPY,
  advanceStandbyHandHold,
  evaluateTwoHandsRaised,
  smoothVisualProgress,
};

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
    ctaHoldingPrefix: string;
    ctaHoldingSuffix: string;
    tagline: string;
    ariaWake: string;
    leftHandLabel: string;
    rightHandLabel: string;
    sensorConnecting: string;
    sensorReady: string;
    sensorOneHand: string;
    sensorHolding: string;
    sensorFallback: string;
  }
> = {
  th: {
    ctaLabel: 'ยกมือ 2 ข้าง ค้างไว้ 3 วินาทีเพื่อเริ่มต้น',
    ctaHoldingPrefix: 'ค้างไว้ทั้ง 2 มือ อีก',
    ctaHoldingSuffix: 'วินาที',
    tagline: 'ระบบฟื้นฟูกายภาพบำบัดอัจฉริยะ',
    ariaWake: 'ยกมือทั้ง 2 ข้างค้างไว้ 3 วินาทีเพื่อเริ่มต้นใช้งานระบบ Dr.Tech.Care',
    leftHandLabel: 'มือซ้าย',
    rightHandLabel: 'มือขวา',
    sensorConnecting: 'กำลังเปิดกล้อง AI ตรวจจับการยกมือ...',
    sensorReady: 'ยืนหน้าตู้แล้วยกมือทั้ง 2 ข้างขึ้นพร้อมกัน',
    sensorOneHand: 'พบมือ 1 ข้าง — กรุณายกอีกข้างขึ้นพร้อมกัน',
    sensorHolding: 'ตรวจพบทั้ง 2 มือแล้ว กำลังเริ่มต้นอย่างนุ่มนวล',
    sensorFallback: 'ไม่พบกล้อง — สามารถแตะหน้าจอเพื่อเริ่มต้นได้',
  },
  en: {
    ctaLabel: 'Raise Both Hands for 3 Seconds to Start',
    ctaHoldingPrefix: 'Hold both hands up for',
    ctaHoldingSuffix: 'sec',
    tagline: 'Intelligent Physical Rehabilitation System',
    ariaWake: 'Raise both hands for 3 seconds to start Dr.Tech.Care',
    leftHandLabel: 'Left Hand',
    rightHandLabel: 'Right Hand',
    sensorConnecting: 'Starting AI gesture camera...',
    sensorReady: 'Stand in view and raise both hands together',
    sensorOneHand: 'One hand detected — raise your other hand too',
    sensorHolding: 'Both hands detected — waking smoothly',
    sensorFallback: 'Camera unavailable — tap screen to begin',
  },
  cn: {
    ctaLabel: '举起双手保持3秒开始使用',
    ctaHoldingPrefix: '请保持双手举起',
    ctaHoldingSuffix: '秒',
    tagline: '智能物理康复治疗系统',
    ariaWake: '举起双手保持3秒开始使用 Dr.Tech.Care',
    leftHandLabel: '左手',
    rightHandLabel: '右手',
    sensorConnecting: '正在启动 AI 手势识别摄像头...',
    sensorReady: '请站在屏幕前同时举起双手',
    sensorOneHand: '已检测到一只手 — 请同时举起另一只手',
    sensorHolding: '已检测到双手，正在平滑启动',
    sensorFallback: '未检测到摄像头 — 可点击屏幕开始',
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

const RING_RADIUS = 24;
const RING_CIRCUMFERENCE = 2 * Math.PI * RING_RADIUS;

export type StandbySensorStatus = 'connecting' | 'ready' | 'unavailable';

export interface StandbyGestureUiState {
  sensorStatus: StandbySensorStatus;
  leftRaised: boolean;
  rightRaised: boolean;
  bothRaised: boolean;
  remainingSec: number;
  isHolding: boolean;
}

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
  ctaProgressFillRef: React.RefObject<HTMLDivElement | null>;
  progressRingRef: React.RefObject<SVGCircleElement | null>;
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

/**
 * Clean vector icon showing two raised hands with glowing upward wave arcs
 * and an integrated 60FPS SVG circular hold-progress ring.
 */
function RaisedTwoHandsProgressEmblem({
  leftRaised = false,
  rightRaised = false,
  isHolding = false,
  remainingSec = 3,
  svgRef,
  progressRingRef,
  rippleInnerRef,
  rippleOuterRef,
}: {
  leftRaised?: boolean;
  rightRaised?: boolean;
  isHolding?: boolean;
  remainingSec?: number;
  svgRef?: React.RefObject<SVGSVGElement | null>;
  progressRingRef?: React.RefObject<SVGCircleElement | null>;
  rippleInnerRef?: React.RefObject<SVGPathElement | null>;
  rippleOuterRef?: React.RefObject<SVGPathElement | null>;
}) {
  return (
    <div className="relative flex h-12 w-12 sm:h-14 sm:w-14 shrink-0 items-center justify-center">
      {/* 60FPS SVG Circular Progress Ring */}
      <svg
        viewBox="0 0 56 56"
        className="absolute inset-0 h-full w-full -rotate-90"
        aria-hidden="true"
      >
        <circle
          cx="28"
          cy="28"
          r={RING_RADIUS}
          fill="rgba(255, 255, 255, 0.08)"
          stroke="rgba(224, 242, 254, 0.28)"
          strokeWidth="3"
        />
        <circle
          ref={progressRingRef}
          cx="28"
          cy="28"
          r={RING_RADIUS}
          fill="none"
          stroke={isHolding ? '#6ee7b7' : '#a5f3fc'}
          strokeWidth="3.5"
          strokeLinecap="round"
          strokeDasharray={RING_CIRCUMFERENCE.toFixed(2)}
          strokeDashoffset={RING_CIRCUMFERENCE.toFixed(2)}
          style={{
            filter: isHolding ? 'drop-shadow(0 0 8px rgba(110, 231, 183, 0.95))' : undefined,
          }}
        />
      </svg>

      {isHolding ? (
        <span
          data-testid="standby-hold-countdown"
          className="relative z-10 font-mono text-xl sm:text-2xl font-black text-white drop-shadow-[0_0_10px_rgba(110,231,183,0.9)] game-combo-pop"
          key={remainingSec}
        >
          {remainingSec}
        </span>
      ) : (
        <svg
          ref={svgRef}
          viewBox="0 0 64 64"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
          aria-hidden="true"
          className="standby-gpu-layer relative z-10 h-8 w-8 sm:h-9 sm:w-9"
        >
          {/* Upward motion arcs */}
          <g stroke="#a5f3fc" strokeWidth="2.1" strokeLinecap="round">
            <path ref={rippleInnerRef} d="M22 14 C27 9.5 37 9.5 42 14" opacity="0.9" />
            <path ref={rippleOuterRef} d="M17 10 C24.5 4.2 39.5 4.2 47 10" opacity="0.58" />
          </g>
          {/* Left raised open hand */}
          <g
            stroke={leftRaised ? '#6ee7b7' : '#ffffff'}
            strokeWidth="2.3"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M14 43 V27 C14 25.4 15.2 24.2 16.7 24.2 C18.2 24.2 19.3 25.4 19.3 27 V22.8 C19.3 21.3 20.5 20.2 22 20.2 C23.5 20.2 24.6 21.3 24.6 22.8 V25.5 C24.6 24.2 25.7 23.2 27 23.2 C28.3 23.2 29.3 24.2 29.3 25.5 V39.5 C29.3 44.5 26 48 21.5 48 C17.1 48 14 45.7 14 43 Z" />
          </g>
          {/* Right raised open hand */}
          <g
            stroke={rightRaised ? '#6ee7b7' : '#ffffff'}
            strokeWidth="2.3"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M50 43 V27 C50 25.4 48.8 24.2 47.3 24.2 C45.8 24.2 44.7 25.4 44.7 27 V22.8 C44.7 21.3 43.5 20.2 42 20.2 C40.5 20.2 39.4 21.3 39.4 22.8 V25.5 C39.4 24.2 38.3 23.2 37 23.2 C35.7 23.2 34.7 24.2 34.7 25.5 V39.5 C34.7 44.5 38 48 42.5 48 C46.9 48 50 45.7 50 43 Z" />
          </g>
        </svg>
      )}
    </div>
  );
}

/**
 * Full-Screen Cinematic Dr.Tech.Care Brand Intro Overlay shown right after the 3-second
 * two-hand raise completes, featuring the animated emblem, wordmark typography,
 * and warm Thai welcome voice subtitle.
 */
export function KioskBrandIntroOverlay({
  lang = 'th',
  isExiting = false,
  canSkip = false,
  onSkip,
}: {
  lang?: ClinicalLang;
  isExiting?: boolean;
  canSkip?: boolean;
  onSkip?: () => void;
}) {
  const welcome = STANDBY_WELCOME_COPY[lang] ?? STANDBY_WELCOME_COPY.th;

  return (
    <div
      data-testid="kiosk-brand-intro-overlay"
      data-no-ripple="true"
      data-no-lift="true"
      onClick={() => {
        if (canSkip) onSkip?.();
      }}
      className={`standby-intro-overlay fixed inset-0 z-50 flex flex-col items-center justify-center overflow-hidden px-6 text-center text-white select-none transition-all duration-400 ease-out ${
        isExiting ? 'opacity-0 scale-[1.03]' : 'opacity-100 scale-100'
      }`}
      style={{
        background:
          'radial-gradient(circle at 50% 42%, #0c6f9c 0%, #07447d 46%, #042556 100%)',
      }}
    >
      {/* Ambient Glowing Aurora Orbs */}
      <div
        aria-hidden="true"
        className="pointer-events-none absolute -top-28 left-1/2 -translate-x-1/2 h-[440px] w-[440px] rounded-full opacity-45 blur-3xl"
        style={{
          background: 'radial-gradient(circle, rgba(45, 212, 191, 0.55) 0%, transparent 70%)',
        }}
      />
      <div
        aria-hidden="true"
        className="pointer-events-none absolute -bottom-32 left-1/2 -translate-x-1/2 h-[460px] w-[460px] rounded-full opacity-40 blur-3xl"
        style={{
          background: 'radial-gradient(circle, rgba(56, 189, 248, 0.5) 0%, transparent 72%)',
        }}
      />

      {/* Center Hero Brand Showcase */}
      <div className="relative z-10 flex flex-col items-center max-w-2xl">
        {/* Emblem with Expanding Holographic Halo Rings */}
        <div className="relative flex items-center justify-center mb-7">
          <div
            aria-hidden="true"
            className="standby-intro-ring-1 pointer-events-none absolute h-36 w-36 sm:h-44 sm:w-44 rounded-full border-2 border-cyan-200/55"
          />
          <div
            aria-hidden="true"
            className="standby-intro-ring-2 pointer-events-none absolute h-36 w-36 sm:h-44 sm:w-44 rounded-full border-2 border-emerald-300/45"
          />

          <div
            data-testid="kiosk-intro-logo"
            className="standby-intro-logo relative overflow-hidden flex h-28 w-28 sm:h-36 sm:w-36 items-center justify-center rounded-[32px] border border-white/45 shadow-[0_24px_65px_rgba(2,132,199,0.55)]"
            style={{
              background: 'linear-gradient(135deg, #2596e8 0%, #1bc4b8 65%, #10b981 100%)',
            }}
          >
            <div
              aria-hidden="true"
              className="pointer-events-none absolute inset-0 bg-gradient-to-tr from-transparent via-white/20 to-white/35"
            />
            <svg
              viewBox="0 0 48 48"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
              aria-hidden="true"
              className="relative z-10 h-16 w-16 sm:h-20 sm:w-20"
            >
              <path
                d="M24 40 C24 40 7.5 29.8 7.5 17.8 C7.5 12.2 11.9 8 17.2 8 C20.4 8 23.1 9.6 24 12.1 C24.9 9.6 27.6 8 30.8 8 C36.1 8 40.5 12.2 40.5 17.8 C40.5 29.8 24 40 24 40 Z"
                stroke="#ffffff"
                strokeWidth="2.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
              <path
                className="standby-intro-ecg-line"
                d="M10.5 23 H17.5 L20.5 16 L24.5 30 L28 20.5 L30.5 23 H37.5"
                stroke="#ffffff"
                strokeWidth="2.6"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </div>
        </div>

        {/* Project Signature Font / Wordmark */}
        <div className="standby-intro-wordmark flex flex-col items-center">
          <h1
            data-testid="kiosk-intro-wordmark"
            className="text-4xl sm:text-6xl md:text-7xl font-black tracking-tight text-white drop-shadow-[0_6px_26px_rgba(3,28,66,0.48)] leading-none"
            style={{ fontFamily: "'Prompt', 'Inter', 'IBM Plex Sans Thai', sans-serif" }}
          >
            Dr.Tech.Care
          </h1>

          <div className="mt-4 inline-flex items-center gap-2 rounded-full border border-cyan-200/35 bg-white/14 px-5 py-1.5 backdrop-blur-md">
            <span className="h-2 w-2 rounded-full bg-emerald-300 shadow-[0_0_10px_#6ee7b7]" />
            <span className="text-xs sm:text-sm font-extrabold tracking-[0.22em] text-cyan-50 uppercase">
              {welcome.badgeLabel}
            </span>
          </div>

          <div
            aria-hidden="true"
            className="mt-6 h-0.5 w-44 sm:w-56 rounded-full bg-gradient-to-r from-transparent via-cyan-200/80 to-transparent"
          />
        </div>

        {/* Welcome Message + Live Voice Equalizer Bars */}
        <div className="standby-intro-caption mt-6 space-y-2.5 px-4">
          <p
            data-testid="kiosk-intro-welcome-headline"
            className="text-xl sm:text-3xl font-extrabold text-white drop-shadow-[0_2px_12px_rgba(4,34,74,0.45)]"
            style={{ fontFamily: "'Prompt', 'IBM Plex Sans Thai', sans-serif" }}
          >
            {welcome.headline}
          </p>
          <p
            data-testid="kiosk-intro-welcome-subline"
            className="text-base sm:text-xl font-semibold text-cyan-100/95"
            style={{ fontFamily: "'Prompt', 'IBM Plex Sans Thai', sans-serif" }}
          >
            {welcome.subline}
          </p>

          {/* 5-bar smooth voice wave indicator */}
          <div
            aria-hidden="true"
            className="pt-3 flex items-end justify-center gap-1.5 h-8"
          >
            {[0, 140, 280, 180, 80].map((delayMs, idx) => (
              <span
                key={idx}
                className="standby-intro-voice-bar w-1.5 rounded-full bg-gradient-to-t from-cyan-300 to-emerald-200"
                style={{
                  height: idx === 2 ? '26px' : idx === 1 || idx === 3 ? '20px' : '14px',
                  animationDelay: `${delayMs}ms`,
                }}
              />
            ))}
          </div>
        </div>
      </div>

      {/* Subtle Optional Skip Pill */}
      {canSkip && onSkip && (
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            onSkip();
          }}
          className="absolute bottom-8 z-20 rounded-full border border-white/25 bg-white/10 px-5 py-2 text-xs sm:text-sm font-bold text-cyan-50/90 backdrop-blur-md transition hover:bg-white/20 cursor-pointer"
        >
          {welcome.skipLabel}
        </button>
      )}
    </div>
  );
}

export function KioskStandbyView({
  timeText,
  lang = 'th',
  onWake,
  wakeRipple = null,
  isWaking = false,
  gestureUi,
  motionRefs,
}: {
  timeText: string;
  lang?: ClinicalLang;
  onWake: (e?: { clientX?: number; clientY?: number }) => void;
  wakeRipple?: { xPct: number; yPct: number } | null;
  isWaking?: boolean;
  gestureUi?: StandbyGestureUiState;
  motionRefs?: KioskStandbyMotionRefs;
}) {
  const copy = STANDBY_COPY[lang] ?? STANDBY_COPY.th;
  const activeGesture: StandbyGestureUiState = gestureUi ?? {
    sensorStatus: 'ready',
    leftRaised: false,
    rightRaised: false,
    bothRaised: false,
    remainingSec: 3,
    isHolding: false,
  };

  const statusText = activeGesture.isHolding
    ? copy.sensorHolding
    : activeGesture.leftRaised !== activeGesture.rightRaised
      ? copy.sensorOneHand
      : activeGesture.sensorStatus === 'connecting'
        ? copy.sensorConnecting
        : activeGesture.sensorStatus === 'unavailable'
          ? copy.sensorFallback
          : copy.sensorReady;

  const pillHeadline = activeGesture.isHolding
    ? `${copy.ctaHoldingPrefix} ${activeGesture.remainingSec} ${copy.ctaHoldingSuffix}`
    : copy.ctaLabel;

  return (
    <div
      data-testid="kiosk-standby-screen"
      data-no-ripple="true"
      data-no-lift="true"
      onClick={(e) => onWake(e)}
      aria-label={copy.ariaWake}
      className={`animate-view-enter standby-gpu-layer min-h-screen w-full relative overflow-hidden flex flex-col items-center justify-between px-6 py-10 sm:py-12 select-none cursor-pointer text-white transition-all duration-300 ease-out ${
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

      {/* Wake Expanding Ripple Overlay */}
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
        className="standby-gpu-layer relative z-10 mt-[7vh] sm:mt-[6vh] flex flex-wrap items-center justify-center gap-2.5 sm:gap-4 max-w-full"
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

      {/* 2. Middle Stack: Silky 2-Hand 3s Raise Centerpiece + Hand Indicators + 7-Segment Clock */}
      <main className="relative z-10 flex flex-col items-center justify-center w-full max-w-2xl my-auto py-6">
        <div ref={motionRefs?.ctaWrapperRef} className="standby-gpu-layer relative max-w-full">
          <div
            ref={motionRefs?.ctaGlowRef}
            aria-hidden="true"
            className="standby-gpu-layer pointer-events-none -inset-2.5 absolute rounded-full blur-xl opacity-55 transition-colors duration-300"
            style={{
              background: activeGesture.isHolding
                ? 'linear-gradient(90deg, rgba(16, 185, 129, 0.55), rgba(110, 231, 183, 0.75), rgba(45, 212, 191, 0.55))'
                : 'linear-gradient(90deg, rgba(56, 189, 248, 0.42), rgba(165, 243, 252, 0.58), rgba(45, 212, 191, 0.42))',
            }}
          />
          <button
            type="button"
            data-testid="kiosk-standby-start-btn"
            onClick={(e) => {
              e.stopPropagation();
              onWake(e);
            }}
            className={`group relative overflow-hidden inline-flex items-center justify-center gap-3 sm:gap-4 max-w-full rounded-full border-[1.5px] px-5 py-3 sm:px-8 sm:py-4 backdrop-blur-md transition-all duration-200 hover:scale-[1.02] active:scale-[0.98] cursor-pointer shadow-[0_16px_42px_rgba(2,132,199,0.38)] ${
              activeGesture.isHolding ? 'border-emerald-200/95' : 'border-cyan-100/75'
            }`}
            style={{
              background:
                'linear-gradient(180deg, rgba(145, 224, 240, 0.28) 0%, rgba(71, 168, 206, 0.22) 100%)',
            }}
          >
            {/* 60FPS Liquid Emerald Progress Fill across Pill */}
            <div
              ref={motionRefs?.ctaProgressFillRef}
              aria-hidden="true"
              className="standby-gpu-layer pointer-events-none absolute inset-0 origin-left"
              style={{
                transform: 'scaleX(0)',
                background:
                  'linear-gradient(90deg, rgba(16, 185, 129, 0.42) 0%, rgba(45, 212, 191, 0.55) 65%, rgba(110, 231, 183, 0.62) 100%)',
              }}
            />

            {/* Diagonal Glass Sheen */}
            <div
              ref={motionRefs?.ctaSheenRef}
              aria-hidden="true"
              className="standby-gpu-layer pointer-events-none absolute inset-y-0 -left-1/3 w-1/3"
              style={{
                background:
                  'linear-gradient(105deg, transparent 10%, rgba(255,255,255,0.28) 50%, transparent 90%)',
              }}
            />

            <RaisedTwoHandsProgressEmblem
              leftRaised={activeGesture.leftRaised}
              rightRaised={activeGesture.rightRaised}
              isHolding={activeGesture.isHolding}
              remainingSec={activeGesture.remainingSec}
              svgRef={motionRefs?.handSvgRef}
              progressRingRef={motionRefs?.progressRingRef}
              rippleInnerRef={motionRefs?.handRippleInnerRef}
              rippleOuterRef={motionRefs?.handRippleOuterRef}
            />

            <span
              className="relative z-10 text-base min-[390px]:text-lg sm:text-2xl md:text-[26px] font-bold tracking-wide text-white drop-shadow-[0_1px_6px_rgba(4,34,74,0.38)] whitespace-nowrap"
              style={{ fontFamily: "'Prompt', 'IBM Plex Sans Thai', sans-serif" }}
            >
              {pillHeadline}
            </span>
          </button>
        </div>

        {/* Live Left/Right Hand Raised Status Pills + Guidance Subtext */}
        <div
          data-testid="standby-hand-status-strip"
          className="mt-4 flex flex-col items-center gap-2"
        >
          <div className="flex items-center gap-2.5">
            <span
              data-testid="standby-left-hand-badge"
              className={`inline-flex items-center gap-1.5 rounded-full border px-3.5 py-1 text-xs sm:text-sm font-bold backdrop-blur-md transition-all duration-200 ${
                activeGesture.leftRaised
                  ? 'border-emerald-200/90 bg-emerald-400/28 text-white shadow-[0_0_16px_rgba(110,231,183,0.45)] scale-105'
                  : 'border-white/25 bg-white/10 text-cyan-100/85'
              }`}
            >
              <span
                className={`flex h-4 w-4 items-center justify-center rounded-full ${
                  activeGesture.leftRaised ? 'bg-emerald-300 text-emerald-950' : 'bg-white/25'
                }`}
              >
                {activeGesture.leftRaised && <Check className="h-3 w-3 stroke-[3]" />}
              </span>
              <span>{copy.leftHandLabel}</span>
            </span>

            <span
              data-testid="standby-right-hand-badge"
              className={`inline-flex items-center gap-1.5 rounded-full border px-3.5 py-1 text-xs sm:text-sm font-bold backdrop-blur-md transition-all duration-200 ${
                activeGesture.rightRaised
                  ? 'border-emerald-200/90 bg-emerald-400/28 text-white shadow-[0_0_16px_rgba(110,231,183,0.45)] scale-105'
                  : 'border-white/25 bg-white/10 text-cyan-100/85'
              }`}
            >
              <span
                className={`flex h-4 w-4 items-center justify-center rounded-full ${
                  activeGesture.rightRaised ? 'bg-emerald-300 text-emerald-950' : 'bg-white/25'
                }`}
              >
                {activeGesture.rightRaised && <Check className="h-3 w-3 stroke-[3]" />}
              </span>
              <span>{copy.rightHandLabel}</span>
            </span>
          </div>

          <p className="text-xs sm:text-sm font-semibold text-cyan-100/90 tracking-wide">
            {statusText}
          </p>
        </div>

        <div className="mt-7 sm:mt-8">
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
  const [phase, setPhase] = useState<'standby' | 'intro'>('standby');
  const [isIntroExiting, setIsIntroExiting] = useState(false);
  const [canSkipIntro, setCanSkipIntro] = useState(false);
  const [gestureUi, setGestureUi] = useState<StandbyGestureUiState>({
    sensorStatus: 'connecting',
    leftRaised: false,
    rightRaised: false,
    bothRaised: false,
    remainingSec: 3,
    isHolding: false,
  });

  const onStartRef = useRef(onStart);
  onStartRef.current = onStart;
  const langRef = useRef<ClinicalLang>(lang);
  langRef.current = lang;

  const hiddenVideoRef = useRef<HTMLVideoElement | null>(null);
  const cameraStreamRef = useRef<MediaStream | null>(null);
  const handSampleRef = useRef<TwoHandsRaiseSample>({
    personDetected: false,
    leftRaised: false,
    rightRaised: false,
    bothRaised: false,
    raisedCount: 0,
  });
  const holdStateRef = useRef<StandbyHandHoldState>({ ...INITIAL_STANDBY_HAND_HOLD_STATE });
  const visualProgressRef = useRef(0);
  const lastAnnouncedSecRef = useRef<number | null>(null);
  const introStartedRef = useRef(false);
  const stopVoiceRef = useRef<(() => void) | null>(null);
  const introMinTimerRef = useRef<number | null>(null);
  const introMaxTimerRef = useRef<number | null>(null);
  const introSkipUnlockTimerRef = useRef<number | null>(null);
  const introExitTimerRef = useRef<number | null>(null);
  const introMinDoneRef = useRef(false);
  const introVoiceDoneRef = useRef(false);

  const singleWake = useMemo(() => createSingleFireStandbyWake(() => onStartRef.current()), []);

  const stopStandbyCamera = useCallback(() => {
    const stream = cameraStreamRef.current;
    cameraStreamRef.current = null;
    if (stream) {
      stream.getTracks().forEach((track) => {
        try {
          track.stop();
        } catch {
          // Ignore track stop errors
        }
      });
    }
    if (hiddenVideoRef.current) {
      try {
        hiddenVideoRef.current.srcObject = null;
      } catch {
        // Ignore cleanup errors
      }
    }
  }, []);

  const finishIntroAndEnterKiosk = useCallback(() => {
    if (introExitTimerRef.current !== null) return;
    if (introMinTimerRef.current !== null && typeof window !== 'undefined') {
      window.clearTimeout(introMinTimerRef.current);
      introMinTimerRef.current = null;
    }
    if (introMaxTimerRef.current !== null && typeof window !== 'undefined') {
      window.clearTimeout(introMaxTimerRef.current);
      introMaxTimerRef.current = null;
    }
    if (introSkipUnlockTimerRef.current !== null && typeof window !== 'undefined') {
      window.clearTimeout(introSkipUnlockTimerRef.current);
      introSkipUnlockTimerRef.current = null;
    }
    setIsIntroExiting(true);
    if (typeof window === 'undefined') {
      singleWake();
      return;
    }
    introExitTimerRef.current = window.setTimeout(() => {
      introExitTimerRef.current = null;
      stopVoiceRef.current?.();
      stopVoiceRef.current = null;
      singleWake();
    }, STANDBY_INTRO_EXIT_MS);
  }, [singleWake]);

  const startBrandIntroSequence = useCallback(
    (e?: { clientX?: number; clientY?: number }) => {
      if (introStartedRef.current) return;
      introStartedRef.current = true;
      stopStandbyCamera();

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
        setWakeRipple({ xPct: 50, yPct: 50 });
      }

      playGameSfx('unlock');
      setPhase('intro');

      if (typeof window === 'undefined') {
        singleWake();
        return;
      }

      introMinDoneRef.current = false;
      introVoiceDoneRef.current = false;

      const tryComplete = () => {
        if (introMinDoneRef.current && introVoiceDoneRef.current) {
          finishIntroAndEnterKiosk();
        }
      };

      stopVoiceRef.current = playStandbyWelcomeVoice(langRef.current, {
        onEnd: () => {
          introVoiceDoneRef.current = true;
          tryComplete();
        },
      });

      introSkipUnlockTimerRef.current = window.setTimeout(() => {
        introSkipUnlockTimerRef.current = null;
        setCanSkipIntro(true);
      }, 1400);

      introMinTimerRef.current = window.setTimeout(() => {
        introMinTimerRef.current = null;
        introMinDoneRef.current = true;
        tryComplete();
      }, STANDBY_INTRO_MIN_MS);

      introMaxTimerRef.current = window.setTimeout(() => {
        introMaxTimerRef.current = null;
        introMinDoneRef.current = true;
        introVoiceDoneRef.current = true;
        finishIntroAndEnterKiosk();
      }, STANDBY_INTRO_MAX_MS);
    },
    [finishIntroAndEnterKiosk, singleWake, stopStandbyCamera],
  );

  const startBrandIntroRef = useRef(startBrandIntroSequence);
  startBrandIntroRef.current = startBrandIntroSequence;

  useEffect(() => {
    if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
      try {
        window.speechSynthesis.getVoices();
      } catch {
        // Ignore unavailable voice synthesis warm-up
      }
    }
    return () => {
      stopStandbyCamera();
      stopVoiceRef.current?.();
      stopVoiceRef.current = null;
      if (typeof window !== 'undefined') {
        if (introMinTimerRef.current !== null) window.clearTimeout(introMinTimerRef.current);
        if (introMaxTimerRef.current !== null) window.clearTimeout(introMaxTimerRef.current);
        if (introSkipUnlockTimerRef.current !== null) {
          window.clearTimeout(introSkipUnlockTimerRef.current);
        }
        if (introExitTimerRef.current !== null) window.clearTimeout(introExitTimerRef.current);
      }
    };
  }, [stopStandbyCamera]);

  // Background Touchless 2-Hand-Raise Detector (MediaPipe PoseLandmarker on hidden video)
  useEffect(() => {
    if (
      phase !== 'standby' ||
      typeof window === 'undefined' ||
      typeof navigator === 'undefined' ||
      !navigator.mediaDevices?.getUserMedia
    ) {
      return;
    }

    let cancelled = false;
    let poseLandmarker: PoseLandmarker | null = null;
    let pollTimerId: number | null = null;
    let lastVideoTime = -1;

    async function initGestureCamera() {
      try {
        const deviceId = await selectClinicalCameraDeviceId().catch(() => undefined);
        const constraintChain: MediaTrackConstraints[] = deviceId
          ? [
              { deviceId: { ideal: deviceId }, width: { ideal: 640 }, height: { ideal: 480 } },
              { deviceId: { ideal: deviceId } },
              { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
            ]
          : [
              { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
              { facingMode: 'user' },
            ];

        let stream: MediaStream | null = null;
        for (const vc of constraintChain) {
          try {
            stream = await navigator.mediaDevices.getUserMedia({ video: vc, audio: false });
            break;
          } catch {
            // Try next fallback constraint
          }
        }
        if (!stream) {
          stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        }

        if (cancelled || introStartedRef.current) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }

        cameraStreamRef.current = stream;
        const video = hiddenVideoRef.current;
        if (!video) {
          stopStandbyCamera();
          return;
        }
        video.srcObject = stream;
        await video.play().catch(() => {});
        if (cancelled || introStartedRef.current) {
          stopStandbyCamera();
          return;
        }

        const handle = await getCachedPoseLandmarker({ numPoses: 1 });
        if (cancelled || introStartedRef.current) {
          stopStandbyCamera();
          return;
        }
        poseLandmarker = handle.instance;

        setGestureUi((prev) =>
          prev.sensorStatus === 'ready' ? prev : { ...prev, sensorStatus: 'ready' },
        );

        const detectStep = () => {
          if (cancelled || introStartedRef.current) return;
          const v = hiddenVideoRef.current;
          if (
            v &&
            poseLandmarker &&
            v.readyState >= 2 &&
            v.videoWidth > 0 &&
            v.currentTime !== lastVideoTime &&
            !(typeof document !== 'undefined' && document.hidden)
          ) {
            lastVideoTime = v.currentTime;
            try {
              const nowMs = typeof performance !== 'undefined' ? performance.now() : Date.now();
              const res = poseLandmarker.detectForVideo(v, getNextMonotonicPoseDetectTs(nowMs));
              const landmarks = res.landmarks?.[0];
              const prevSample = handSampleRef.current;
              const nextSample = evaluateTwoHandsRaised(landmarks, prevSample);
              handSampleRef.current = nextSample;

              if (
                nextSample.leftRaised !== prevSample.leftRaised ||
                nextSample.rightRaised !== prevSample.rightRaised ||
                nextSample.bothRaised !== prevSample.bothRaised
              ) {
                setGestureUi((prev) => ({
                  ...prev,
                  leftRaised: nextSample.leftRaised,
                  rightRaised: nextSample.rightRaised,
                  bothRaised: nextSample.bothRaised,
                }));
              }
            } catch {
              // Ignore single-frame detection errors cleanly
            }
          }
          pollTimerId = window.setTimeout(detectStep, 38);
        };

        detectStep();
      } catch {
        if (!cancelled) {
          setGestureUi((prev) => ({ ...prev, sensorStatus: 'unavailable' }));
        }
      }
    }

    void initGestureCamera();

    return () => {
      cancelled = true;
      if (pollTimerId !== null) {
        window.clearTimeout(pollTimerId);
      }
      stopStandbyCamera();
    };
  }, [phase, stopStandbyCamera]);

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
  const ctaProgressFillRef = useRef<HTMLDivElement | null>(null);
  const progressRingRef = useRef<SVGCircleElement | null>(null);
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
      ctaProgressFillRef,
      progressRingRef,
      handSvgRef,
      handRippleInnerRef,
      handRippleOuterRef,
      colonRef,
    }),
    [],
  );

  // Unified 60FPS RequestAnimationFrame Fluid Motion & Silky 3-Second Hold Progress Engine
  useEffect(() => {
    if (
      phase !== 'standby' ||
      typeof window === 'undefined' ||
      typeof window.requestAnimationFrame !== 'function'
    ) {
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
    let prevFrameMs = startMs;

    const drawFluidFrame = (nowMs: number) => {
      if (typeof document !== 'undefined' && document.hidden) {
        prevFrameMs = nowMs;
        rafId = window.requestAnimationFrame(drawFluidFrame);
        return;
      }

      const dtMs = Math.max(0, Math.min(100, nowMs - prevFrameMs));
      prevFrameMs = nowMs;
      const t = (nowMs - startMs) / 1000;

      // 0. Advance 3-Second Two-Hand Hold Timer & 60FPS Silky Progress Visuals
      if (!introStartedRef.current) {
        const prevHold = holdStateRef.current;
        const nextHold = advanceStandbyHandHold(prevHold, {
          bothRaised: handSampleRef.current.bothRaised,
          dtMs,
          targetHoldMs: STANDBY_HAND_WAKE_HOLD_MS,
        });
        holdStateRef.current = nextHold;

        const visualProg = smoothVisualProgress(
          visualProgressRef.current,
          nextHold.progress,
          dtMs,
        );
        visualProgressRef.current = visualProg;

        if (progressRingRef.current) {
          const offset = RING_CIRCUMFERENCE * (1 - visualProg);
          progressRingRef.current.style.strokeDashoffset = offset.toFixed(2);
        }
        if (ctaProgressFillRef.current) {
          ctaProgressFillRef.current.style.transform = `scaleX(${visualProg.toFixed(4)})`;
        }

        const isHoldingNow = nextHold.holdMs > 30;
        const wasHolding = prevHold.holdMs > 30;

        if (isHoldingNow && lastAnnouncedSecRef.current !== nextHold.remainingSec) {
          lastAnnouncedSecRef.current = nextHold.remainingSec;
          if (nextHold.remainingSec === 3) {
            playGameSfx('tap');
          } else if (nextHold.remainingSec === 2) {
            playGameSfx('rep');
          } else if (nextHold.remainingSec === 1) {
            playGameSfx('combo', { comboCount: 3 });
          }
        } else if (!isHoldingNow) {
          lastAnnouncedSecRef.current = null;
        }

        if (isHoldingNow !== wasHolding || prevHold.remainingSec !== nextHold.remainingSec) {
          setGestureUi((prev) => ({
            ...prev,
            isHolding: isHoldingNow,
            remainingSec: nextHold.remainingSec,
          }));
        }

        if (nextHold.completed) {
          startBrandIntroRef.current();
          return;
        }
      }

      const activeProg = visualProgressRef.current;

      // 1. Canvas Liquid-Silk Multi-Strand Ribbons + Bokeh Micro-Particles
      const canvas = waveCanvasRef.current;
      const ctx = canvas?.getContext('2d');
      if (canvas && ctx) {
        ctx.clearRect(0, 0, width, height);

        const bandCenterY = height * 0.51;
        const bandHeight = Math.max(170, height * 0.3) * (1 + activeProg * 0.14);
        const stepPx = Math.max(10, Math.round(width / 72));

        const primaryGrad = ctx.createLinearGradient(0, bandCenterY - bandHeight, width, bandCenterY + bandHeight);
        primaryGrad.addColorStop(0, 'rgba(147, 197, 253, 0.25)');
        primaryGrad.addColorStop(0.5, activeProg > 0.1 ? 'rgba(167, 243, 208, 0.72)' : 'rgba(224, 242, 254, 0.65)');
        primaryGrad.addColorStop(1, 'rgba(153, 246, 228, 0.36)');

        ctx.strokeStyle = primaryGrad;
        ctx.lineWidth = 1.25 + activeProg * 0.45;
        const primaryCount = 16;
        for (let s = 0; s < primaryCount; s++) {
          const bell = 1 - Math.abs(s / (primaryCount - 1) - 0.5) * 2;
          ctx.globalAlpha = 0.08 + bell * (0.28 + activeProg * 0.12);
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
        const os = 1 + Math.sin(t * 0.4) * 0.08 + activeProg * 0.1;
        orbTopRef.current.style.transform = `translate3d(${ox.toFixed(1)}px, ${oy.toFixed(1)}px, 0) scale(${os.toFixed(3)})`;
      }
      if (orbBottomRef.current) {
        const ox = Math.cos(t * 0.22) * 32;
        const oy = Math.sin(t * 0.29) * 24;
        const os = 1 + Math.cos(t * 0.35) * 0.08 + activeProg * 0.1;
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

      // 6. Silky Floating CTA Pill + Outer Breathing Halo + Shimmer + Raised Hands Float
      if (ctaWrapperRef.current) {
        const ctaFloat = Math.sin(t * 0.95 + 0.6) * (3.5 * (1 - activeProg * 0.5));
        const ctaBoostScale = 1 + activeProg * 0.045;
        ctaWrapperRef.current.style.transform = `translate3d(0, ${ctaFloat.toFixed(2)}px, 0) scale(${ctaBoostScale.toFixed(4)})`;
      }
      if (ctaGlowRef.current) {
        const glowAlpha = 0.4 + 0.32 * (0.5 + 0.5 * Math.sin(t * 1.9)) + activeProg * 0.28;
        const glowScale = 0.98 + 0.05 * (0.5 + 0.5 * Math.sin(t * 1.9)) + activeProg * 0.08;
        ctaGlowRef.current.style.opacity = Math.min(1, glowAlpha).toFixed(3);
        ctaGlowRef.current.style.transform = `scale(${glowScale.toFixed(3)})`;
      }
      if (ctaSheenRef.current) {
        const ctaCycle = computeEcgSweepProgress(t, 3.6, 0.4);
        const ctaX = -140 + ctaCycle * 560;
        ctaSheenRef.current.style.transform = `translate3d(${ctaX.toFixed(1)}%, 0, 0) skewX(-22deg)`;
      }
      if (handSvgRef.current) {
        const handLift = -Math.abs(Math.sin(t * 2.2)) * 2.8;
        const handScale = 1 + Math.abs(Math.sin(t * 2.2)) * 0.05;
        handSvgRef.current.style.transform = `translate3d(0, ${handLift.toFixed(2)}px, 0) scale(${handScale.toFixed(4)})`;
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
  }, [motionRefs, phase]);

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
        if (phase === 'standby') {
          startBrandIntroSequence();
        } else if (canSkipIntro) {
          finishIntroAndEnterKiosk();
        }
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [phase, canSkipIntro, startBrandIntroSequence, finishIntroAndEnterKiosk]);

  return (
    <>
      {/* Hidden Low-Bandwidth Camera Feed for Touchless 2-Hand-Raise Standby Wake */}
      {phase === 'standby' && (
        <video
          ref={hiddenVideoRef}
          muted
          playsInline
          autoPlay
          aria-hidden="true"
          className="sr-only pointer-events-none"
        />
      )}

      {phase === 'standby' ? (
        <>
          <KioskStandbyView
            timeText={timeText}
            lang={lang}
            onWake={startBrandIntroSequence}
            wakeRipple={wakeRipple}
            isWaking={false}
            gestureUi={gestureUi}
            motionRefs={motionRefs}
          />
          <MorTechMascot mode="standby" lang={lang} />
        </>
      ) : (
        <KioskBrandIntroOverlay
          lang={lang}
          isExiting={isIntroExiting}
          canSkip={canSkipIntro}
          onSkip={finishIntroAndEnterKiosk}
        />
      )}
    </>
  );
}
