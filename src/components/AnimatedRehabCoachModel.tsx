import React, { useEffect, useState } from 'react';
import { Eye, EyeOff, Maximize2, Play, X } from 'lucide-react';
import type { ExerciseSideCode } from '@drtechcares/pose-engine';
import type { ExercisePrescriptionOption } from '../hooks/useKioskSocket';

export type CoachPhaseId = 'prepare' | 'move' | 'hold' | 'lower';

export interface CoachTimelineFrame {
  cycleElapsedSec: number;
  phase: CoachPhaseId;
  /** 1-based step highlight matching Zone 4 cards (1=เตรียม, 2=ขยับ, 3=ค้างท่า) */
  activeStepNumber: 1 | 2 | 3;
  /** 0.0 -> 1.0 kinematic interpolation progress from neutral to peak ROM */
  motionProgress: number;
  phaseProgress: number;
  phaseLabelTh: string;
  countdownSec: number;
}

export const COACH_PREPARE_END_SEC = 1.2;
export const COACH_MOVE_END_SEC = 3.0;
export const COACH_HOLD_END_SEC = 5.0;
export const COACH_CYCLE_DURATION_SEC = 6.2;

function easeInOutCubic(t: number): number {
  const clamped = Math.max(0, Math.min(1, t));
  return clamped < 0.5
    ? 4 * clamped * clamped * clamped
    : 1 - Math.pow(-2 * clamped + 2, 3) / 2;
}

export function computeCoachTimelineFrame(
  rawElapsedSec: number,
  displayHoldSec = 3,
): CoachTimelineFrame {
  const safeMod =
    ((rawElapsedSec % COACH_CYCLE_DURATION_SEC) + COACH_CYCLE_DURATION_SEC) %
    COACH_CYCLE_DURATION_SEC;

  if (safeMod < COACH_PREPARE_END_SEC) {
    const p = safeMod / COACH_PREPARE_END_SEC;
    return {
      cycleElapsedSec: safeMod,
      phase: 'prepare',
      activeStepNumber: 1,
      motionProgress: 0,
      phaseProgress: p,
      phaseLabelTh: '1. เตรียมท่า',
      countdownSec: Math.max(1, Math.round(displayHoldSec)),
    };
  }

  if (safeMod < COACH_MOVE_END_SEC) {
    const p = (safeMod - COACH_PREPARE_END_SEC) / (COACH_MOVE_END_SEC - COACH_PREPARE_END_SEC);
    return {
      cycleElapsedSec: safeMod,
      phase: 'move',
      activeStepNumber: 2,
      motionProgress: easeInOutCubic(p),
      phaseProgress: p,
      phaseLabelTh: '2. ขยับขึ้น',
      countdownSec: Math.max(1, Math.round(displayHoldSec)),
    };
  }

  if (safeMod < COACH_HOLD_END_SEC) {
    const p = (safeMod - COACH_MOVE_END_SEC) / (COACH_HOLD_END_SEC - COACH_MOVE_END_SEC);
    const visualCountdown = Math.max(1, Math.ceil((1 - p) * Math.max(1, Math.round(displayHoldSec))));
    return {
      cycleElapsedSec: safeMod,
      phase: 'hold',
      activeStepNumber: 3,
      motionProgress: 1,
      phaseProgress: p,
      phaseLabelTh: '3. ค้างท่า',
      countdownSec: visualCountdown,
    };
  }

  const p = (safeMod - COACH_HOLD_END_SEC) / (COACH_CYCLE_DURATION_SEC - COACH_HOLD_END_SEC);
  return {
    cycleElapsedSec: safeMod,
    phase: 'lower',
    activeStepNumber: 3,
    motionProgress: 1 - easeInOutCubic(p),
    phaseProgress: p,
    phaseLabelTh: '4. เอาลงช้าๆ',
    countdownSec: 0,
  };
}

// Shared 60FPS singleton RAF hub so Zone 2 PiP, Zone 4 Coach, Zone 4 Steps, and Expanded Modal
// stay 100% synchronized while never re-rendering the parent MediaPipe container.
type CoachClockListener = (cycleElapsedSec: number) => void;
const coachClockListeners = new Set<CoachClockListener>();
let coachRafHandle = 0;
let coachEpochMs = 0;
let lastCoachEmitMs = 0;

function runSharedCoachClock(nowMs: number) {
  if (coachClockListeners.size === 0) {
    coachRafHandle = 0;
    return;
  }
  if (coachEpochMs === 0) {
    coachEpochMs = nowMs;
  }
  // Cap SVG coach updates to ~30FPS to prevent 60Hz React reconciliation from competing with pose tracking
  if (lastCoachEmitMs === 0 || nowMs - lastCoachEmitMs >= 33) {
    lastCoachEmitMs = nowMs;
    const elapsedSec = ((nowMs - coachEpochMs) / 1000) % COACH_CYCLE_DURATION_SEC;
    coachClockListeners.forEach((listener) => listener(elapsedSec));
  }
  coachRafHandle = window.requestAnimationFrame(runSharedCoachClock);
}

export function resetSharedCoachClockEpoch(): void {
  coachEpochMs = typeof performance !== 'undefined' ? performance.now() : 0;
  coachClockListeners.forEach((listener) => listener(0));
}

export function useSyncedCoachClock(enabled = true): number {
  const [elapsedSec, setElapsedSec] = useState<number>(0);

  useEffect(() => {
    if (!enabled || typeof window === 'undefined') {
      return;
    }
    const listener: CoachClockListener = (sec) => {
      setElapsedSec(sec);
    };
    coachClockListeners.add(listener);
    if (coachRafHandle === 0) {
      coachRafHandle = window.requestAnimationFrame(runSharedCoachClock);
    }
    return () => {
      coachClockListeners.delete(listener);
      if (coachClockListeners.size === 0 && coachRafHandle !== 0) {
        window.cancelAnimationFrame(coachRafHandle);
        coachRafHandle = 0;
      }
    };
  }, [enabled]);

  return elapsedSec;
}

export type CoachMotionKind =
  | 'shoulder_flexion'
  | 'shoulder_abduction'
  | 'shoulder_rotation'
  | 'cross_body_reach'
  | 'scap_protract'
  | 'elbow_flexion'
  | 'hand_to_head'
  | 'forearm_turn'
  | 'wrist_finger'
  | 'thumb_pinch'
  | 'clasped_arm_raise'
  | 'clasped_forward_push'
  | 'neck_tilt'
  | 'neck_turn'
  | 'shoulder_shrug_roll'
  | 'sit_to_stand'
  | 'knee_lift'
  | 'hip_abduction'
  | 'heel_toe_raise'
  | 'knee_extension'
  | 'trunk_reach';

export function resolveCoachMotionKind(exercise: ExercisePrescriptionOption): CoachMotionKind {
  const combined =
    `${exercise.exercise_id} ${exercise.base_code ?? ''} ${exercise.catalog_exercise_id ?? ''} ${exercise.effective_code ?? ''} ${exercise.expected_plane ?? ''} ${exercise.name_th} ${exercise.name_en} ${exercise.target_joint}`.toLowerCase();

  if (combined.includes('sit_to_stand') || combined.includes('ลุกนั่ง') || combined.includes('squat')) {
    return 'sit_to_stand';
  }
  if (combined.includes('knee_lift') || combined.includes('ยกเข่า') || combined.includes('high knee')) {
    return 'knee_lift';
  }
  if (combined.includes('hip_abd') || combined.includes('กางขา')) {
    return 'hip_abduction';
  }
  if (combined.includes('heel') || combined.includes('tiptoe') || combined.includes('เขย่ง')) {
    return 'heel_toe_raise';
  }
  if (combined.includes('knee_ext') || combined.includes('เหยียดเข่า')) {
    return 'knee_extension';
  }
  if (combined.includes('trunk_reach') || combined.includes('เอื้อมตัว') || combined.includes('เอื้อมลำตัว')) {
    return 'trunk_reach';
  }
  if (combined.includes('neck_turn') || combined.includes('หันคอ')) {
    return 'neck_turn';
  }
  if (combined.includes('neck_tilt') || combined.includes('เอียงคอ') || combined.includes('neck')) {
    return 'neck_tilt';
  }
  if (combined.includes('shrug') || combined.includes('ยักไหล่') || combined.includes('หมุนหัวไหล่')) {
    return 'shoulder_shrug_roll';
  }
  if (combined.includes('thumb_pinch') || combined.includes('จีบนิ้ว') || combined.includes('pinch')) {
    return 'thumb_pinch';
  }
  if (combined.includes('forearm_turn') || combined.includes('คว่ำ') || combined.includes('หงายมือ')) {
    return 'forearm_turn';
  }
  if (combined.includes('wrist_finger') || combined.includes('กระดกข้อมือ') || combined.includes('นิ้วมือ') || combined.includes('wrist')) {
    return 'wrist_finger';
  }
  if (combined.includes('forward_push') || combined.includes('bilateral_push') || combined.includes('ดันแขนตรงไปด้านหน้า')) {
    return 'clasped_forward_push';
  }
  if (combined.includes('arm_raise') || combined.includes('ประสานมือยกแขน') || exercise.expected_plane === 'bilateral') {
    return 'clasped_arm_raise';
  }
  if (combined.includes('hand_to_head') || combined.includes('แตะศีรษะ') || combined.includes('หวีผม')) {
    return 'hand_to_head';
  }
  if (combined.includes('elb_flex') || combined.includes('งอ-เหยียดข้อศอก') || combined.includes('ข้อศอก') || combined.includes('elbow')) {
    return 'elbow_flexion';
  }
  if (combined.includes('scap_protract') || combined.includes('ยื่นมือเอื้อมไปด้านหน้า') || combined.includes('protract')) {
    return 'scap_protract';
  }
  if (combined.includes('cross_body') || combined.includes('เอื้อมแขนไขว้')) {
    return 'cross_body_reach';
  }
  if (combined.includes('sh_rot') || combined.includes('หมุนข้อไหล่') || exercise.expected_plane === 'rotation') {
    return 'shoulder_rotation';
  }
  if (combined.includes('sh_abd') || combined.includes('กางแขน') || exercise.expected_plane === 'abduction') {
    return 'shoulder_abduction';
  }
  return 'shoulder_flexion';
}

export function resolveCoachSide(
  exercise: ExercisePrescriptionOption,
  explicitSide?: ExerciseSideCode,
): ExerciseSideCode {
  if (explicitSide === 'LEFT' || explicitSide === 'RIGHT' || explicitSide === 'BOTH') {
    return explicitSide;
  }
  if (exercise.prescribed_side === 'LEFT' || exercise.prescribed_side === 'RIGHT' || exercise.prescribed_side === 'BOTH') {
    return exercise.prescribed_side;
  }
  if (exercise.target_joint === 'left_shoulder') return 'LEFT';
  if (exercise.target_joint === 'right_shoulder') return 'RIGHT';
  return 'BOTH';
}

interface Pt {
  x: number;
  y: number;
}

function polarPt(origin: Pt, length: number, angleDeg: number): Pt {
  // 0° = straight down, positive angles depend on caller direction
  const rad = (angleDeg * Math.PI) / 180;
  return {
    x: origin.x + Math.sin(rad) * length,
    y: origin.y + Math.cos(rad) * length,
  };
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function lerpPt(a: Pt, b: Pt, t: number): Pt {
  return { x: lerp(a.x, b.x, t), y: lerp(a.y, b.y, t) };
}

function makeCurvedArrow(from: Pt, to: Pt, bulge = 20): { d: string; tipAngleDeg: number } {
  const mx = (from.x + to.x) / 2;
  const my = (from.y + to.y) / 2;
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const len = Math.max(1, Math.hypot(dx, dy));
  const nx = -dy / len;
  const ny = dx / len;
  const cx = mx + nx * bulge;
  const cy = my + ny * bulge;
  const tipAngleDeg = (Math.atan2(to.y - cy, to.x - cx) * 180) / Math.PI;
  return {
    d: `M ${from.x.toFixed(1)} ${from.y.toFixed(1)} Q ${cx.toFixed(1)} ${cy.toFixed(1)} ${to.x.toFixed(1)} ${to.y.toFixed(1)}`,
    tipAngleDeg,
  };
}

const EMERALD = '#10b981';
const EMERALD_DARK = '#047857';
const EMERALD_LIGHT = '#34d399';
const OTTER_FUR = '#9C6644';
const OTTER_FUR_LIGHT = '#B8825C';
const OTTER_FUR_DARK = '#7A4B2E';
const OTTER_OUTLINE = '#5C3620';
const OTTER_CREAM = '#F7E7CE';
const OTTER_PAW_PAD = '#F2DFC2';
const MINT_SCARF = '#10B981';
const MINT_DARK = '#065F46';

function MascotEmblem({ cx, cy, scale = 1 }: { cx: number; cy: number; scale?: number }) {
  return (
    <g transform={`translate(${cx.toFixed(1)}, ${cy.toFixed(1)}) scale(${scale})`}>
      <circle cx={0} cy={0} r={7.2} fill="#FFFFFF" stroke="#0D9488" strokeWidth={1.6} />
      <path
        d="M 0 4.5 C 0 4.5 -5 1.7 -5 -1.4 C -5 -3 -3.8 -4.1 -2.3 -4.1 C -1.3 -4.1 -0.5 -3.6 0 -2.8 C 0.5 -3.6 1.3 -4.1 2.3 -4.1 C 3.8 -4.1 5 -3 5 -1.4 C 5 1.7 0 4.5 0 4.5 Z"
        fill="#10B981"
      />
      <rect x={-0.8} y={-2.2} width={1.6} height={4.2} rx={0.6} fill="#FFFFFF" />
      <rect x={-2.1} y={-0.9} width={4.2} height={1.6} rx={0.6} fill="#FFFFFF" />
    </g>
  );
}

function MascotOtterHeadFront({
  cx,
  cy,
  headTiltDeg = 0,
  neckTurnShiftX = 0,
  scale = 1,
}: {
  cx: number;
  cy: number;
  headTiltDeg?: number;
  neckTurnShiftX?: number;
  scale?: number;
}) {
  return (
    <g transform={`translate(${cx.toFixed(1)}, ${cy.toFixed(1)}) rotate(${headTiltDeg}) scale(${scale})`}>
      {/* Rounded Sea Otter Ears */}
      <circle cx={-23} cy={-12} r={7.8} fill={OTTER_FUR_DARK} stroke={OTTER_OUTLINE} strokeWidth={2.1} />
      <circle cx={-23} cy={-11.5} r={3.8} fill={OTTER_PAW_PAD} />
      <circle cx={23} cy={-12} r={7.8} fill={OTTER_FUR_DARK} stroke={OTTER_OUTLINE} strokeWidth={2.1} />
      <circle cx={23} cy={-11.5} r={3.8} fill={OTTER_PAW_PAD} />

      {/* Plump Otter Head */}
      <ellipse cx={0} cy={0} rx={25} ry={21.5} fill={OTTER_FUR} stroke={OTTER_OUTLINE} strokeWidth={2.3} />
      {/* Cream Facial Mask */}
      <ellipse cx={neckTurnShiftX * 0.45} cy={4.5} rx={19.5} ry={15} fill={OTTER_CREAM} />

      {/* Signature Mint Sport Headband + Heart Cross */}
      <path
        d="M -23.5 -6 C -15 -14 15 -14 23.5 -6 C 24 -12 19 -19.5 0 -20 C -19 -19.5 -24 -12 -23.5 -6 Z"
        fill={MINT_SCARF}
        stroke={MINT_DARK}
        strokeWidth={1.8}
        strokeLinejoin="round"
      />
      <MascotEmblem cx={neckTurnShiftX * 0.35} cy={-13} scale={0.85} />

      {/* Expressive Dark Eyes with White Sparkles */}
      <circle cx={-9 + neckTurnShiftX} cy={1.5} r={3.8} fill="#2A1A10" />
      <circle cx={9 + neckTurnShiftX} cy={1.5} r={3.8} fill="#2A1A10" />
      <circle cx={-7.7 + neckTurnShiftX} cy={0.2} r={1.3} fill="#FFFFFF" />
      <circle cx={10.3 + neckTurnShiftX} cy={0.2} r={1.3} fill="#FFFFFF" />

      {/* Rosy Cheeks */}
      <ellipse cx={-15.5 + neckTurnShiftX * 0.7} cy={6.8} rx={3.8} ry={2.1} fill="#FB7185" opacity={0.38} />
      <ellipse cx={15.5 + neckTurnShiftX * 0.7} cy={6.8} rx={3.8} ry={2.1} fill="#FB7185" opacity={0.38} />

      {/* Cute Otter Nose & Happy Smile */}
      <path
        d={`M ${(-4 + neckTurnShiftX).toFixed(1)} 4.6 C ${(-4 + neckTurnShiftX).toFixed(1)} 3.1 ${(4 + neckTurnShiftX).toFixed(1)} 3.1 ${(4 + neckTurnShiftX).toFixed(1)} 4.6 C ${(4 + neckTurnShiftX).toFixed(1)} 6.8 ${(1.8 + neckTurnShiftX).toFixed(1)} 8.6 ${neckTurnShiftX.toFixed(1)} 8.6 C ${(-1.8 + neckTurnShiftX).toFixed(1)} 8.6 ${(-4 + neckTurnShiftX).toFixed(1)} 6.8 ${(-4 + neckTurnShiftX).toFixed(1)} 4.6 Z`}
        fill="#3E2414"
      />
      <path
        d={`M ${(-6 + neckTurnShiftX).toFixed(1)} 10.6 Q ${(-2.5 + neckTurnShiftX).toFixed(1)} 13.6 ${neckTurnShiftX.toFixed(1)} 11.4 Q ${(2.5 + neckTurnShiftX).toFixed(1)} 13.6 ${(6 + neckTurnShiftX).toFixed(1)} 10.6`}
        fill="none"
        stroke={OTTER_OUTLINE}
        strokeWidth={2}
        strokeLinecap="round"
        strokeLinejoin="round"
      />

      {/* Crisp Otter Whiskers */}
      <g stroke={OTTER_FUR_DARK} strokeWidth={1.3} strokeLinecap="round" opacity={0.7}>
        <line x1={-17} y1={6} x2={-24.5} y2={4.5} />
        <line x1={-17} y1={9} x2={-24} y2={10.2} />
        <line x1={17} y1={6} x2={24.5} y2={4.5} />
        <line x1={17} y1={9} x2={24} y2={10.2} />
      </g>
    </g>
  );
}

function MascotOtterHeadSide({ cx, cy, scale = 1 }: { cx: number; cy: number; scale?: number }) {
  return (
    <g transform={`translate(${cx.toFixed(1)}, ${cy.toFixed(1)}) scale(${scale})`}>
      {/* Back Ear */}
      <circle cx={-12} cy={-14} r={7.2} fill={OTTER_FUR_DARK} stroke={OTTER_OUTLINE} strokeWidth={2} />
      <circle cx={-12} cy={-13.5} r={3.4} fill={OTTER_PAW_PAD} />
      {/* Plump Side Otter Head */}
      <ellipse cx={0} cy={0} rx={22.5} ry={20.5} fill={OTTER_FUR} stroke={OTTER_OUTLINE} strokeWidth={2.3} />
      {/* Side Cream Snout & Cheek */}
      <ellipse cx={12.5} cy={5.5} rx={12.5} ry={11} fill={OTTER_CREAM} />
      {/* Signature Mint Headband */}
      <path
        d="M -21 -5 C -8 -13 12 -13 21 -6 C 20 -13 12 -20 -1 -20 C -15 -20 -22 -12 -21 -5 Z"
        fill={MINT_SCARF}
        stroke={MINT_DARK}
        strokeWidth={1.8}
        strokeLinejoin="round"
      />
      <MascotEmblem cx={7} cy={-12.5} scale={0.78} />
      {/* Front Eye */}
      <circle cx={10.5} cy={1} r={3.6} fill="#2A1A10" />
      <circle cx={11.8} cy={-0.3} r={1.3} fill="#FFFFFF" />
      {/* Rosy Cheek */}
      <ellipse cx={8} cy={7.5} rx={3.6} ry={2} fill="#FB7185" opacity={0.38} />
      {/* Side Cute Nose */}
      <ellipse cx={23.2} cy={4} rx={4} ry={3} fill="#3E2414" />
      {/* Happy Side Smile */}
      <path d="M 14 10.5 Q 19 13 22 9.5" fill="none" stroke={OTTER_OUTLINE} strokeWidth={2} strokeLinecap="round" />
      {/* Side Whiskers */}
      <g stroke={OTTER_FUR_DARK} strokeWidth={1.3} strokeLinecap="round" opacity={0.72}>
        <line x1={18} y1={6.5} x2={26} y2={5.5} />
        <line x1={18} y1={9} x2={25.5} y2={10.5} />
      </g>
    </g>
  );
}

function ActiveOrMutedLimb({
  p1,
  p2,
  p3,
  active,
  width = 14,
  showEndpointDot = true,
}: {
  p1: Pt;
  p2: Pt;
  p3?: Pt;
  active: boolean;
  width?: number;
  showEndpointDot?: boolean;
}) {
  const pathD = p3
    ? `M ${p1.x.toFixed(1)} ${p1.y.toFixed(1)} L ${p2.x.toFixed(1)} ${p2.y.toFixed(1)} L ${p3.x.toFixed(1)} ${p3.y.toFixed(1)}`
    : `M ${p1.x.toFixed(1)} ${p1.y.toFixed(1)} L ${p2.x.toFixed(1)} ${p2.y.toFixed(1)}`;
  const endPt = p3 ?? p2;

  return (
    <g>
      {active && (
        <path
          d={pathD}
          fill="none"
          stroke={EMERALD_LIGHT}
          strokeOpacity={0.42}
          strokeWidth={width + 11}
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      )}
      {/* Dark otter fur outline */}
      <path
        d={pathD}
        fill="none"
        stroke={active ? EMERALD_DARK : OTTER_OUTLINE}
        strokeWidth={width + 4}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      {/* Plump otter fur limb core */}
      <path
        d={pathD}
        fill="none"
        stroke={active ? OTTER_FUR_LIGHT : OTTER_FUR}
        strokeWidth={width}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      {/* Active Emerald Rehab Wristband / Sport Sleeve Accent */}
      {active && (
        <path
          d={pathD}
          fill="none"
          stroke={EMERALD}
          strokeOpacity={0.35}
          strokeWidth={width - 3}
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      )}
      {/* Joint cue dot */}
      <circle
        cx={p2.x}
        cy={p2.y}
        r={active ? 5.2 : 3.2}
        fill={active ? '#ffffff' : OTTER_FUR_LIGHT}
        stroke={active ? EMERALD_DARK : OTTER_OUTLINE}
        strokeWidth={active ? 2.2 : 1.4}
      />
      {/* Cute Otter Paw at Endpoint */}
      {showEndpointDot && p3 && (
        <g>
          <circle
            cx={endPt.x}
            cy={endPt.y}
            r={active ? 9.5 : 7.5}
            fill={active ? OTTER_FUR_LIGHT : OTTER_FUR_DARK}
            stroke={active ? EMERALD_DARK : OTTER_OUTLINE}
            strokeWidth={active ? 2.8 : 2}
          />
          <circle cx={endPt.x} cy={endPt.y + 0.5} r={active ? 4.2 : 3.2} fill={OTTER_PAW_PAD} />
          {active && (
            <circle cx={endPt.x} cy={endPt.y} r={11.5} fill="none" stroke={EMERALD} strokeWidth={2} strokeOpacity={0.75} />
          )}
        </g>
      )}
    </g>
  );
}

function TargetGhostLimb({ p1, p2, p3 }: { p1: Pt; p2: Pt; p3?: Pt }) {
  const pathD = p3
    ? `M ${p1.x.toFixed(1)} ${p1.y.toFixed(1)} L ${p2.x.toFixed(1)} ${p2.y.toFixed(1)} L ${p3.x.toFixed(1)} ${p3.y.toFixed(1)}`
    : `M ${p1.x.toFixed(1)} ${p1.y.toFixed(1)} L ${p2.x.toFixed(1)} ${p2.y.toFixed(1)}`;
  const endPt = p3 ?? p2;
  return (
    <g opacity={0.45}>
      <path
        d={pathD}
        fill="none"
        stroke={EMERALD_DARK}
        strokeWidth={7}
        strokeDasharray="6 5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <circle
        cx={endPt.x}
        cy={endPt.y}
        r={7}
        fill="none"
        stroke={EMERALD_DARK}
        strokeWidth={2.5}
        strokeDasharray="3 2"
      />
    </g>
  );
}

function HoldPulseTarget({
  pt,
  phase,
  countdownSec,
  targetHoldSec,
  phaseProgress,
  compact,
  bounds = { minX: 58, maxX: 200, minY: 18 },
}: {
  pt: Pt;
  phase: CoachPhaseId;
  countdownSec: number;
  targetHoldSec: number;
  phaseProgress: number;
  compact?: boolean;
  bounds?: { minX: number; maxX: number; minY: number };
}) {
  if (phase === 'hold') {
    const ringRadius = 11 + Math.sin(phaseProgress * Math.PI * 4) * 3.5;
    // Place badge below the raised hand when hand sits near the top pills so they never overlap
    const safeY = pt.y < bounds.minY + 18 ? Math.min(bounds.minY + 26, pt.y + 22) : Math.max(bounds.minY, pt.y - 18);
    return (
      <g>
        <circle
          cx={pt.x}
          cy={pt.y}
          r={ringRadius}
          fill="none"
          stroke={EMERALD}
          strokeWidth={2.5}
          strokeOpacity={0.65}
        />
        <circle cx={pt.x} cy={pt.y} r={8} fill={EMERALD} fillOpacity={0.22} />
        <g
          transform={`translate(${Math.min(bounds.maxX, Math.max(bounds.minX, pt.x))}, ${safeY})`}
        >
          <rect x={-42} y={-9.5} width={84} height={19} rx={9.5} fill={EMERALD_DARK} />
          <text
            x={0}
            y={3}
            textAnchor="middle"
            fill="#ffffff"
            fontSize={compact ? 9.2 : 9.8}
            fontWeight={800}
          >
            ค้าง {targetHoldSec}วิ ({countdownSec}s)
          </text>
        </g>
      </g>
    );
  }
  if (phase === 'lower') {
    return (
      <g
        transform={`translate(${Math.min(bounds.maxX, Math.max(bounds.minX, pt.x))}, ${Math.max(bounds.minY, pt.y - 14)})`}
      >
        <rect x={-32} y={-10} width={64} height={20} rx={10} fill="#0284c7" />
        <text x={0} y={3.5} textAnchor="middle" fill="#ffffff" fontSize={10.5} fontWeight={900}>
          +1 ครั้ง!
        </text>
      </g>
    );
  }
  return null;
}

function TargetStarBadge({ x, y }: { x: number; y: number }) {
  return (
    <polygon
      points="0,-8 2.4,-2.6 8,-2.4 3.6,1.2 5.2,6.8 0,3.6 -5.2,6.8 -3.6,1.2 -8,-2.4 -2.4,-2.6"
      fill="#f59e0b"
      stroke="#b45309"
      strokeWidth={1.2}
      transform={`translate(${x.toFixed(1)}, ${y.toFixed(1)})`}
    />
  );
}

function DirectionalArrow({ from, to, bulge = -20 }: { from: Pt; to: Pt; bulge?: number }) {
  if (Math.hypot(to.x - from.x, to.y - from.y) < 10) return null;
  const { d, tipAngleDeg } = makeCurvedArrow(from, to, bulge);
  return (
    <g>
      <path d={d} fill="none" stroke="#0284c7" strokeWidth={3} strokeDasharray="5 3" strokeLinecap="round" />
      <polygon
        points="-5,-4 5,0 -5,4"
        fill="#0284c7"
        transform={`translate(${to.x}, ${to.y}) rotate(${tipAngleDeg})`}
      />
    </g>
  );
}

function FrontStandingBase({
  cx = 130,
  headTiltDeg = 0,
  neckTurnShiftX = 0,
  shoulderLiftLeft = 0,
  shoulderLiftRight = 0,
  leftHipHighlight = false,
  rightHipHighlight = false,
  leftKneeLift = 0,
  rightKneeLift = 0,
  leftLegOutDeg = 0,
  rightLegOutDeg = 0,
  tiptoeLift = 0,
}: {
  cx?: number;
  headTiltDeg?: number;
  neckTurnShiftX?: number;
  shoulderLiftLeft?: number;
  shoulderLiftRight?: number;
  leftHipHighlight?: boolean;
  rightHipHighlight?: boolean;
  leftKneeLift?: number;
  rightKneeLift?: number;
  leftLegOutDeg?: number;
  rightLegOutDeg?: number;
  tiptoeLift?: number;
}) {
  const headCenter = { x: cx, y: 36 };
  const lsY = 64 - shoulderLiftLeft;
  const rsY = 64 - shoulderLiftRight;
  const leftShoulder = { x: cx - 24, y: lsY };
  const rightShoulder = { x: cx + 24, y: rsY };
  const leftHip = { x: cx - 15, y: 122 };
  const rightHip = { x: cx + 15, y: 122 };

  const leftKnee = polarPt(leftHip, 42 - leftKneeLift * 0.4, -leftLegOutDeg);
  leftKnee.y -= leftKneeLift * 0.65;
  leftKnee.x -= leftKneeLift * 0.15;
  const leftAnkle = polarPt(leftKnee, 42, -leftLegOutDeg * 0.8);
  leftAnkle.y -= leftKneeLift * 0.45 + tiptoeLift;

  const rightKnee = polarPt(rightHip, 42 - rightKneeLift * 0.4, rightLegOutDeg);
  rightKnee.y -= rightKneeLift * 0.65;
  rightKnee.x += rightKneeLift * 0.15;
  const rightAnkle = polarPt(rightKnee, 42, rightLegOutDeg * 0.8);
  rightAnkle.y -= rightKneeLift * 0.45 + tiptoeLift;

  return (
    <g data-testid="mascot-coach-character">
      {/* Subtle floor shadow */}
      <ellipse cx={cx} cy={210} rx={58} ry={7} fill="#0F4C81" opacity={0.14} />

      {/* Cute Otter Tail peeking behind */}
      <path
        d={`M ${cx - 18} 138 C ${cx - 46} 142, ${cx - 56} 126, ${cx - 48} 118 C ${cx - 42} 112, ${cx - 34} 124, ${cx - 16} 126 Z`}
        fill={OTTER_FUR_DARK}
        stroke={OTTER_OUTLINE}
        strokeWidth={2.2}
        strokeLinejoin="round"
      />

      {/* Otter Hind Legs & Paws */}
      <ActiveOrMutedLimb p1={leftHip} p2={leftKnee} p3={leftAnkle} active={leftHipHighlight} width={14} />
      <ActiveOrMutedLimb p1={rightHip} p2={rightKnee} p3={rightAnkle} active={rightHipHighlight} width={14} />

      {/* Plump Otter Torso + Soft Cream Belly */}
      <ellipse
        cx={cx}
        cy={102 - (shoulderLiftLeft + shoulderLiftRight) * 0.25}
        rx={33}
        ry={40}
        fill={OTTER_FUR}
        stroke={OTTER_OUTLINE}
        strokeWidth={2.4}
      />
      <ellipse
        cx={cx}
        cy={106 - (shoulderLiftLeft + shoulderLiftRight) * 0.25}
        rx={21.5}
        ry={27}
        fill={OTTER_CREAM}
      />

      {/* Signature Mint Clinical Scarf + Flowing Tail + Medical Emblem */}
      <path
        d={`M ${cx - 28} ${68 - shoulderLiftLeft * 0.6} C ${cx - 16} ${79 - shoulderLiftLeft * 0.4}, ${cx + 16} ${79 - shoulderLiftRight * 0.4}, ${cx + 28} ${68 - shoulderLiftRight * 0.6} C ${cx + 30} ${76 - shoulderLiftRight * 0.4}, ${cx + 22} ${84 - shoulderLiftRight * 0.4}, ${cx} ${85 - (shoulderLiftLeft + shoulderLiftRight) * 0.3} C ${cx - 22} ${84 - shoulderLiftLeft * 0.4}, ${cx - 30} ${76 - shoulderLiftLeft * 0.4}, ${cx - 28} ${68 - shoulderLiftLeft * 0.6} Z`}
        fill={MINT_SCARF}
        stroke={MINT_DARK}
        strokeWidth={2}
        strokeLinejoin="round"
      />
      <path
        d={`M ${cx + 10} 79 L ${cx + 22} 80 L ${cx + 26} 102 C ${cx + 20} 104, ${cx + 14} 104, ${cx + 10} 100 Z`}
        fill={MINT_SCARF}
        stroke={MINT_DARK}
        strokeWidth={1.8}
        strokeLinejoin="round"
      />
      <MascotEmblem cx={cx} cy={79 - (shoulderLiftLeft + shoulderLiftRight) * 0.3} scale={0.95} />

      {/* Nong MorTech Otter Head */}
      <MascotOtterHeadFront
        cx={headCenter.x}
        cy={headCenter.y + 8}
        headTiltDeg={headTiltDeg}
        neckTurnShiftX={neckTurnShiftX}
        scale={1.06}
      />
    </g>
  );
}

export function resolveElderlyLimbCueTh(
  exercise: ExercisePrescriptionOption,
  targetSide: ExerciseSideCode,
): {
  headlineTh: string;
  shortSideTh: string;
  direction: 'left' | 'right' | 'both';
} {
  const kind = resolveCoachMotionKind(exercise);
  const isLeg =
    kind === 'knee_lift' ||
    kind === 'hip_abduction' ||
    kind === 'knee_extension' ||
    kind === 'heel_toe_raise' ||
    kind === 'sit_to_stand';
  const isNeck = kind === 'neck_tilt' || kind === 'neck_turn';

  if (targetSide === 'BOTH') {
    return {
      headlineTh: isNeck ? 'ทำซ้าย - ขวา' : isLeg ? 'ทำทั้ง 2 ข้าง' : 'ทำทั้ง 2 แขน',
      shortSideTh: 'ทั้ง 2 ข้าง',
      direction: 'both',
    };
  }
  if (targetSide === 'LEFT') {
    return {
      headlineTh: isNeck ? 'ทำข้างซ้าย' : isLeg ? 'ทำขาซ้าย' : 'ทำแขนซ้าย',
      shortSideTh: 'แขนซ้าย',
      direction: 'left',
    };
  }
  return {
    headlineTh: isNeck ? 'ทำข้างขวา' : isLeg ? 'ทำขาขวา' : 'ทำแขนขวา',
    shortSideTh: 'แขนขวา',
    direction: 'right',
  };
}

export function formatConciseExerciseNameTh(rawNameTh: string): string {
  const cleaned = rawNameTh
    .replace(/^\[[^\]]+\]\s*/g, '')
    .replace(/\s*—\s*ข้าง.*$/g, '')
    .replace(/\s*\([^)]*\)\s*/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
  return cleaned || rawNameTh;
}

export interface AnimatedRehabCoachModelProps {
  exercise: ExercisePrescriptionOption;
  targetSide?: ExerciseSideCode;
  /** Optional explicit cycle time; if omitted, subscribes to 60FPS shared clock cleanly */
  cycleElapsedSec?: number;
  compact?: boolean;
  className?: string;
}

function AnimatedRehabCoachModelInner({
  exercise,
  targetSide,
  cycleElapsedSec,
  compact = false,
  className = 'w-full h-full',
}: AnimatedRehabCoachModelProps) {
  const internalElapsed = useSyncedCoachClock(cycleElapsedSec === undefined);
  const effectiveElapsed = cycleElapsedSec ?? internalElapsed;
  const targetHoldSec = Math.max(1, Math.round(exercise.hold_sec || 3));
  const targetRom = Math.max(20, Math.round(exercise.target_rom_deg || 110));
  const frame = computeCoachTimelineFrame(effectiveElapsed, targetHoldSec);
  const kind = resolveCoachMotionKind(exercise);
  const resolvedSide = resolveCoachSide(exercise, targetSide);
  const activeLeft = resolvedSide === 'LEFT' || resolvedSide === 'BOTH';
  const activeRight = resolvedSide === 'RIGHT' || resolvedSide === 'BOTH';

  const currentDeg = Math.round(targetRom * frame.motionProgress);
  const isMoving = frame.phase === 'move';
  const isHoldOrLower = frame.phase === 'hold' || frame.phase === 'lower';
  const sideLabel =
    resolvedSide === 'BOTH' ? 'ทั้ง 2 ข้าง' : resolvedSide === 'LEFT' ? 'ซ้าย (L)' : 'ขวา (R)';

  // Dynamic tight framing per exercise category so Nong MorTech fills the stage ~2x larger
  // without empty side margins or unused lower-body whitespace on upper-body exercises.
  const stageFrame = (() => {
    if (
      kind === 'shoulder_flexion' ||
      kind === 'scap_protract' ||
      kind === 'clasped_forward_push'
    ) {
      return {
        x: 36,
        y: -6,
        w: 184,
        h: 152,
        leftPillX: 73,
        rightPillX: 180,
        pillY: 5,
        holdBounds: { minX: 78, maxX: 174, minY: 22 },
      };
    }
    if (kind === 'wrist_finger' || kind === 'thumb_pinch' || kind === 'forearm_turn') {
      return {
        x: 20,
        y: 22,
        w: 210,
        h: 176,
        leftPillX: 59,
        rightPillX: 190,
        pillY: 33,
        holdBounds: { minX: 68, maxX: 184, minY: 48 },
      };
    }
    if (
      kind === 'sit_to_stand' ||
      kind === 'knee_lift' ||
      kind === 'hip_abduction' ||
      kind === 'heel_toe_raise' ||
      kind === 'knee_extension' ||
      kind === 'trunk_reach'
    ) {
      return {
        x: 24,
        y: 2,
        w: 212,
        h: 212,
        leftPillX: 64,
        rightPillX: 196,
        pillY: 13,
        holdBounds: { minX: 66, maxX: 192, minY: 28 },
      };
    }
    return {
      x: 22,
      y: -8,
      w: 216,
      h: 154,
      leftPillX: 61,
      rightPillX: 198,
      pillY: 3,
      holdBounds: { minX: 66, maxX: 192, minY: 20 },
    };
  })();

  const renderScene = () => {
    // 1. SIDE / 3/4 VIEW: SHOULDER FLEXION / SCAP PROTRACT / CLASPED FORWARD PUSH / SIT-TO-STAND
    if (
      kind === 'shoulder_flexion' ||
      kind === 'scap_protract' ||
      kind === 'clasped_forward_push'
    ) {
      // Face right (+x is forward)
      const cx = 104;
      const shoulder = { x: cx + 2, y: 66 };
      const hip = { x: cx - 2, y: 124 };
      const backArmEnd = { x: cx - 10, y: 122 };
      const knee = { x: cx + 2, y: 164 };
      const ankle = { x: cx - 2, y: 204 };

      const reachMaxDeg = kind === 'shoulder_flexion' ? Math.min(176, Math.max(60, targetRom)) : 90;
      const currentArmAngle =
        kind === 'shoulder_flexion' ? reachMaxDeg * frame.motionProgress : 90;
      const forwardPushPx =
        kind === 'scap_protract' || kind === 'clasped_forward_push'
          ? 24 * frame.motionProgress
          : 0;

      const curElbow = polarPt(
        { x: shoulder.x + forwardPushPx * 0.45, y: shoulder.y },
        36,
        currentArmAngle,
      );
      const curWrist = polarPt(curElbow, 32, currentArmAngle);
      const targetElbow = polarPt(
        { x: shoulder.x + (kind === 'shoulder_flexion' ? 0 : 11), y: shoulder.y },
        36,
        reachMaxDeg,
      );
      const targetWrist = polarPt(targetElbow, 32, reachMaxDeg);
      const restWrist = polarPt(shoulder, 68, kind === 'shoulder_flexion' ? 6 : 90);

      return (
        <g data-testid="mascot-coach-character">
          <ellipse cx={cx} cy={208} rx={56} ry={7} fill="#0F4C81" opacity={0.14} />
          {/* Cute Otter Tail on Back */}
          <path
            d={`M ${cx - 16} 130 C ${cx - 46} 136, ${cx - 56} 118, ${cx - 46} 109 C ${cx - 40} 103, ${cx - 28} 116, ${cx - 14} 118 Z`}
            fill={OTTER_FUR_DARK}
            stroke={OTTER_OUTLINE}
            strokeWidth={2.2}
            strokeLinejoin="round"
          />
          {/* Standing Otter Legs */}
          <ActiveOrMutedLimb p1={hip} p2={knee} p3={ankle} active={false} width={14} />
          {/* Background Arm (muted if single side, or parallel active if BOTH) */}
          {kind !== 'clasped_forward_push' &&
            (resolvedSide === 'BOTH' ? (
              <ActiveOrMutedLimb
                p1={{ x: shoulder.x - 10, y: shoulder.y - 3 }}
                p2={{ x: curElbow.x - 10, y: curElbow.y - 4 }}
                p3={{ x: curWrist.x - 10, y: curWrist.y - 5 }}
                active
                width={12}
              />
            ) : (
              <ActiveOrMutedLimb
                p1={shoulder}
                p2={{ x: cx - 6, y: 98 }}
                p3={backArmEnd}
                active={false}
                width={12}
              />
            ))}
          {/* Plump Side Otter Torso + Cream Belly */}
          <ellipse cx={cx} cy={104} rx={24} ry={38} fill={OTTER_FUR} stroke={OTTER_OUTLINE} strokeWidth={2.4} />
          <ellipse cx={cx + 10} cy={106} rx={12.5} ry={25} fill={OTTER_CREAM} />
          {/* Side Mint Clinical Scarf + Badge */}
          <path
            d={`M ${cx - 20} 70 C ${cx - 6} 79, ${cx + 14} 79, ${cx + 22} 70 C ${cx + 24} 78, ${cx + 16} 85, ${cx + 2} 86 C ${cx - 12} 85, ${cx - 22} 78, ${cx - 20} 70 Z`}
            fill={MINT_SCARF}
            stroke={MINT_DARK}
            strokeWidth={1.9}
            strokeLinejoin="round"
          />
          <MascotEmblem cx={cx + 11} cy={79} scale={0.85} />
          {/* Nong MorTech Side Otter Head */}
          <MascotOtterHeadSide cx={cx + 2} cy={46} scale={1.06} />

          {/* Dashed Target Ghost + Star */}
          <TargetGhostLimb p1={shoulder} p2={targetElbow} p3={targetWrist} />
          {(kind === 'scap_protract' || kind === 'clasped_forward_push') && (
            <TargetStarBadge x={targetWrist.x + 18} y={targetWrist.y} />
          )}
          {/* Active Forearm/Upper Arm */}
          <ActiveOrMutedLimb
            p1={{ x: shoulder.x + forwardPushPx * 0.45, y: shoulder.y }}
            p2={curElbow}
            p3={curWrist}
            active
            width={14}
          />
          {kind === 'clasped_forward_push' && (
            <circle cx={curWrist.x} cy={curWrist.y} r={11} fill="#10b981" fillOpacity={0.28} stroke="#047857" strokeWidth={2} />
          )}
          {isMoving && <DirectionalArrow from={restWrist} to={targetWrist} bulge={-22} />}
          {isHoldOrLower && (
            <HoldPulseTarget
              pt={curWrist}
              phase={frame.phase}
              countdownSec={frame.countdownSec}
              targetHoldSec={targetHoldSec}
              phaseProgress={frame.phaseProgress}
              compact={compact}
              bounds={stageFrame.holdBounds}
            />
          )}
        </g>
      );
    }

    // 2. ZOOMED WRIST EXTENSION & FINGER SPREAD (#8) OR THUMB PINCH (#9) OR FOREARM TURN (#10)
    if (kind === 'wrist_finger' || kind === 'thumb_pinch' || kind === 'forearm_turn') {
      const cx = 130;
      const wrist = { x: cx - 14, y: 124 };
      const forearmStart = { x: 46, y: 152 };

      if (kind === 'forearm_turn') {
        const flipScaleY = lerp(1, -1, frame.motionProgress);
        const palmLabel = frame.motionProgress > 0.5 ? 'หงายมือ (Up)' : 'คว่ำมือ (Down)';
        return (
          <g data-testid="mascot-coach-character">
            <MascotOtterHeadFront cx={62} cy={72} scale={0.78} />
            {/* Otter Forearm */}
            <ActiveOrMutedLimb p1={forearmStart} p2={wrist} active width={16} showEndpointDot={false} />
            {/* Rotating Otter Paw + Thumb */}
            <g transform={`translate(${wrist.x + 38}, ${wrist.y - 8}) scale(1, ${flipScaleY.toFixed(2)})`}>
              <ellipse cx={0} cy={0} rx={36} ry={18} fill={OTTER_FUR_LIGHT} stroke={EMERALD_DARK} strokeWidth={3} />
              <ellipse cx={2} cy={0} rx={22} ry={10} fill={OTTER_PAW_PAD} />
              <line x1={-6} y1={-12} x2={-14} y2={-28} stroke={EMERALD} strokeWidth={8} strokeLinecap="round" />
              <circle cx={0} cy={0} r={5} fill={EMERALD} />
            </g>
            {/* Curved rotation indicator */}
            <path
              d="M 160 72 A 28 28 0 0 1 198 110"
              fill="none"
              stroke="#0284c7"
              strokeWidth={3.5}
              strokeDasharray="5 3"
            />
            <text x={130} y={182} textAnchor="middle" fill="#0f172a" fontSize={12} fontWeight={800}>
              {palmLabel} • {sideLabel}
            </text>
            {isHoldOrLower && (
              <HoldPulseTarget
                pt={{ x: wrist.x + 36, y: wrist.y - 16 }}
                phase={frame.phase}
                countdownSec={frame.countdownSec}
                targetHoldSec={targetHoldSec}
                phaseProgress={frame.phaseProgress}
                compact={compact}
                bounds={stageFrame.holdBounds}
              />
            )}
          </g>
        );
      }

      if (kind === 'thumb_pinch') {
        const openGap = lerp(34, 4, frame.motionProgress);
        const fingerBase = { x: wrist.x + 28, y: wrist.y - 8 };
        const indexTip = { x: wrist.x + 68, y: wrist.y - 8 - openGap * 0.55 };
        const thumbTip = { x: wrist.x + 66, y: wrist.y - 4 + openGap * 0.55 };
        return (
          <g data-testid="mascot-coach-character">
            <MascotOtterHeadFront cx={60} cy={72} scale={0.76} />
            <ActiveOrMutedLimb p1={forearmStart} p2={wrist} active width={15} showEndpointDot={false} />
            <circle cx={fingerBase.x - 4} cy={fingerBase.y + 4} r={19} fill={OTTER_FUR_LIGHT} stroke={EMERALD_DARK} strokeWidth={2.5} />
            <circle cx={fingerBase.x - 4} cy={fingerBase.y + 4} r={10} fill={OTTER_PAW_PAD} />
            {/* Index finger */}
            <ActiveOrMutedLimb p1={fingerBase} p2={{ x: wrist.x + 50, y: wrist.y - 18 }} p3={indexTip} active width={9} />
            {/* Thumb */}
            <ActiveOrMutedLimb p1={{ x: wrist.x + 18, y: wrist.y + 10 }} p2={{ x: wrist.x + 48, y: wrist.y + 14 }} p3={thumbTip} active width={9} />
            {/* Pinch Star */}
            {frame.motionProgress > 0.78 && (
              <circle cx={(indexTip.x + thumbTip.x) / 2} cy={(indexTip.y + thumbTip.y) / 2} r={12} fill={EMERALD} fillOpacity={0.35} stroke={EMERALD_DARK} strokeWidth={2.5} />
            )}
            <text x={130} y={188} textAnchor="middle" fill="#0f172a" fontSize={12} fontWeight={800}>
              {frame.motionProgress > 0.75 ? 'จีบนิ้วโป้ง-ชี้' : 'กางนิ้วออก'} • {sideLabel}
            </text>
            {isHoldOrLower && (
              <HoldPulseTarget
                pt={{ x: wrist.x + 68, y: wrist.y - 6 }}
                phase={frame.phase}
                countdownSec={frame.countdownSec}
                targetHoldSec={targetHoldSec}
                phaseProgress={frame.phaseProgress}
                compact={compact}
                bounds={stageFrame.holdBounds}
              />
            )}
          </g>
        );
      }

      // WRIST EXTENSION & 5-FINGER SPREAD / FIST
      const wristTiltDeg = lerp(-12, Math.min(75, targetRom), frame.motionProgress);
      const spreadRatio = frame.motionProgress;
      const targetWristTilt = Math.min(75, targetRom);
      const fingers = [-36, -18, 0, 18, 36];
      const wristJoint = { x: 102, y: 128 };
      const palmOrigin = { x: wristJoint.x, y: wristJoint.y };

      return (
        <g data-testid="mascot-coach-character">
          <MascotOtterHeadFront cx={58} cy={72} scale={0.76} />
          {/* Zoomed Otter Forearm */}
          <ActiveOrMutedLimb p1={{ x: 38, y: 158 }} p2={wristJoint} active width={16} showEndpointDot={false} />
          {/* Ghost Wrist */}
          <g transform={`translate(${palmOrigin.x}, ${palmOrigin.y}) rotate(${-targetWristTilt})`} opacity={0.35}>
            <rect x={4} y={-14} width={44} height={28} rx={12} fill="none" stroke={EMERALD_DARK} strokeWidth={2.5} strokeDasharray="5 3" />
          </g>
          {/* Active Articulated Otter Paw + 5 Fingers */}
          <g transform={`translate(${palmOrigin.x}, ${palmOrigin.y}) rotate(${-wristTiltDeg})`}>
            <rect x={2} y={-15} width={44} height={30} rx={13} fill={OTTER_FUR_LIGHT} stroke={EMERALD_DARK} strokeWidth={3} />
            <ellipse cx={22} cy={0} rx={13} ry={9} fill={OTTER_PAW_PAD} />
            {fingers.map((baseAngle, idx) => {
              const splayedAngle = baseAngle * lerp(0.18, 1.0, spreadRatio);
              const fingerLen = idx === 0 ? lerp(20, 32, spreadRatio) : lerp(22, 46, spreadRatio);
              const angleOffset = idx === 0 ? splayedAngle - 24 : splayedAngle - 2;
              const startF = { x: idx === 0 ? 24 : 42, y: baseAngle * 0.25 };
              const rad = (angleOffset * Math.PI) / 180;
              const tipF = { x: startF.x + Math.cos(rad) * fingerLen, y: startF.y + Math.sin(rad) * fingerLen };
              return (
                <g key={baseAngle}>
                  <line
                    x1={startF.x}
                    y1={startF.y}
                    x2={tipF.x}
                    y2={tipF.y}
                    stroke={EMERALD}
                    strokeWidth={idx === 0 ? 8 : 7}
                    strokeLinecap="round"
                  />
                  <circle cx={tipF.x} cy={tipF.y} r={4} fill="#ffffff" stroke={EMERALD_DARK} strokeWidth={2} />
                </g>
              );
            })}
          </g>
          {isMoving && (
            <DirectionalArrow
              from={{ x: 162, y: 132 }}
              to={{ x: 152, y: 66 }}
              bulge={18}
            />
          )}
          <text x={130} y={190} textAnchor="middle" fill="#0f172a" fontSize={12} fontWeight={800}>
            {spreadRatio > 0.5 ? 'กระดกข้อมือ + กาง 5 นิ้ว' : 'ผ่อนข้อมือ / กำหลวมๆ'} • {sideLabel}
          </text>
          {isHoldOrLower && (
            <HoldPulseTarget
              pt={{ x: 145, y: 74 }}
              phase={frame.phase}
              countdownSec={frame.countdownSec}
              targetHoldSec={targetHoldSec}
              phaseProgress={frame.phaseProgress}
              compact={compact}
              bounds={stageFrame.holdBounds}
            />
          )}
        </g>
      );
    }

    // 3. SIT-TO-STAND OR SEATED KNEE EXTENSION / TRUNK REACH (#10, #14, #15)
    if (kind === 'sit_to_stand' || kind === 'knee_extension' || kind === 'trunk_reach') {
      const chairX = 88;
      if (kind === 'sit_to_stand') {
        const standT = frame.motionProgress;
        const hip = { x: lerp(96, 134, standT), y: lerp(136, 114, standT) };
        const leanForward = standT < 0.45 ? standT * 18 : lerp(8, 0, (standT - 0.45) / 0.55);
        const shoulder = { x: hip.x + leanForward, y: hip.y - 52 };
        const knee = { x: lerp(136, 136, standT), y: lerp(142, 160, standT) };
        const ankle = { x: 134, y: 202 };
        const hand = { x: shoulder.x + lerp(28, 10, standT), y: shoulder.y + lerp(26, 48, standT) };

        return (
          <g data-testid="mascot-coach-character">
            {/* Chair */}
            <path d="M 62 78 L 68 142 L 116 142 L 116 202 M 68 142 L 68 202" fill="none" stroke="#64748b" strokeWidth={6} strokeLinecap="round" strokeLinejoin="round" />
            <ActiveOrMutedLimb p1={hip} p2={knee} p3={ankle} active width={14} />
            {/* Otter Torso + Scarf */}
            <line x1={hip.x} y1={hip.y} x2={shoulder.x} y2={shoulder.y} stroke={OTTER_OUTLINE} strokeWidth={28} strokeLinecap="round" />
            <line x1={hip.x} y1={hip.y} x2={shoulder.x} y2={shoulder.y} stroke={OTTER_FUR} strokeWidth={23} strokeLinecap="round" />
            <line x1={hip.x + 4} y1={hip.y - 4} x2={shoulder.x + 5} y2={shoulder.y + 8} stroke={OTTER_CREAM} strokeWidth={11} strokeLinecap="round" />
            <MascotOtterHeadSide cx={shoulder.x + 2} cy={shoulder.y - 22} scale={0.95} />
            <ActiveOrMutedLimb p1={shoulder} p2={lerpPt(shoulder, hand, 0.55)} p3={hand} active={false} width={11} />
            {isMoving && <DirectionalArrow from={{ x: 112, y: 120 }} to={{ x: 148, y: 58 }} bulge={-16} />}
            {isHoldOrLower && (
              <HoldPulseTarget
                pt={{ x: shoulder.x, y: shoulder.y - 36 }}
                phase={frame.phase}
                countdownSec={frame.countdownSec}
                targetHoldSec={targetHoldSec}
                phaseProgress={frame.phaseProgress}
                compact={compact}
                bounds={stageFrame.holdBounds}
              />
            )}
          </g>
        );
      }

      // Seated Knee Extension / Trunk Reach
      const hip = { x: 100, y: 132 };
      const trunkLeanDeg = kind === 'trunk_reach' ? 35 * frame.motionProgress : 4;
      const shoulder = {
        x: hip.x + Math.sin((trunkLeanDeg * Math.PI) / 180) * 52,
        y: hip.y - Math.cos((trunkLeanDeg * Math.PI) / 180) * 52,
      };
      const knee = { x: 142, y: 136 };
      const ankleAngle = kind === 'knee_extension' ? lerp(8, 86, frame.motionProgress) : 8;
      const ankle = polarPt(knee, 46, ankleAngle);
      const reachHand =
        kind === 'trunk_reach'
          ? { x: shoulder.x + lerp(18, 62, frame.motionProgress), y: shoulder.y + lerp(28, 12, frame.motionProgress) }
          : { x: hip.x + 18, y: hip.y - 6 };

      return (
        <g data-testid="mascot-coach-character">
          {/* Chair */}
          <path d={`M ${chairX - 24} 78 L ${chairX - 18} 142 L 136 142 L 136 202 M ${chairX - 18} 142 L ${chairX - 18} 202`} fill="none" stroke="#64748b" strokeWidth={6} strokeLinecap="round" strokeLinejoin="round" />
          <ActiveOrMutedLimb
            p1={hip}
            p2={knee}
            p3={ankle}
            active={kind === 'knee_extension'}
            width={14}
          />
          <line x1={hip.x} y1={hip.y} x2={shoulder.x} y2={shoulder.y} stroke={OTTER_OUTLINE} strokeWidth={28} strokeLinecap="round" />
          <line x1={hip.x} y1={hip.y} x2={shoulder.x} y2={shoulder.y} stroke={OTTER_FUR} strokeWidth={23} strokeLinecap="round" />
          <line x1={hip.x + 4} y1={hip.y - 4} x2={shoulder.x + 5} y2={shoulder.y + 8} stroke={OTTER_CREAM} strokeWidth={11} strokeLinecap="round" />
          <MascotOtterHeadSide cx={shoulder.x + 2} cy={shoulder.y - 20} scale={0.95} />
          <ActiveOrMutedLimb
            p1={shoulder}
            p2={lerpPt(shoulder, reachHand, 0.5)}
            p3={reachHand}
            active={kind === 'trunk_reach'}
            width={12}
          />
          {kind === 'trunk_reach' && <TargetStarBadge x={218} y={88} />}
          {kind === 'knee_extension' && isMoving && (
            <DirectionalArrow from={{ x: 146, y: 184 }} to={{ x: 194, y: 140 }} bulge={16} />
          )}
          {isHoldOrLower && (
            <HoldPulseTarget
              pt={kind === 'knee_extension' ? ankle : reachHand}
              phase={frame.phase}
              countdownSec={frame.countdownSec}
              targetHoldSec={targetHoldSec}
              phaseProgress={frame.phaseProgress}
              compact={compact}
              bounds={stageFrame.holdBounds}
            />
          )}
        </g>
      );
    }

    // 4. FRONT MIRROR FULL-BODY VIEW (All remaining shoulder, elbow, clasped, neck, shrug, leg exercises)
    const cx = 130;
    const isLeftArm = activeLeft;
    const isRightArm = activeRight;
    const shoulderLiftLeft =
      kind === 'shoulder_shrug_roll' ? Math.sin(frame.motionProgress * Math.PI) * 14 : 0;
    const shoulderLiftRight =
      kind === 'shoulder_shrug_roll' ? Math.sin(frame.motionProgress * Math.PI) * 14 : 0;
    const neckTiltDeg =
      kind === 'neck_tilt'
        ? Math.sin(frame.motionProgress * Math.PI) * (resolvedSide === 'LEFT' ? -30 : 30)
        : 0;
    const neckShiftX =
      kind === 'neck_turn'
        ? Math.sin(frame.motionProgress * Math.PI) * (resolvedSide === 'LEFT' ? -8 : 8)
        : 0;

    const ls: Pt = { x: cx - 24, y: 64 - shoulderLiftLeft };
    const rs: Pt = { x: cx + 24, y: 64 - shoulderLiftRight };

    // Rest arm poses
    let leftElbow = { x: ls.x - 8, y: ls.y + 34 };
    let leftWrist = { x: ls.x - 10, y: ls.y + 68 };
    let rightElbow = { x: rs.x + 8, y: rs.y + 34 };
    let rightWrist = { x: rs.x + 10, y: rs.y + 68 };

    let targetGhostLeft: [Pt, Pt, Pt] | null = null;
    let targetGhostRight: [Pt, Pt, Pt] | null = null;
    let activeEndpoint: Pt = rs;
    let arrowFrom: Pt | null = null;
    let arrowTo: Pt | null = null;

    if (kind === 'shoulder_abduction') {
      const maxAbd = Math.min(172, Math.max(45, targetRom));
      const ang = maxAbd * frame.motionProgress;
      if (isLeftArm) {
        leftElbow = polarPt(ls, 34, -ang - 8);
        leftWrist = polarPt(leftElbow, 32, -ang - 8);
        const ge = polarPt(ls, 34, -maxAbd - 8);
        const gw = polarPt(ge, 32, -maxAbd - 8);
        targetGhostLeft = [ls, ge, gw];
        arrowFrom = polarPt(ls, 66, -12);
        arrowTo = gw;
        activeEndpoint = leftWrist;
      }
      if (isRightArm) {
        rightElbow = polarPt(rs, 34, ang + 8);
        rightWrist = polarPt(rightElbow, 32, ang + 8);
        const ge = polarPt(rs, 34, maxAbd + 8);
        const gw = polarPt(ge, 32, maxAbd + 8);
        targetGhostRight = [rs, ge, gw];
        arrowFrom = polarPt(rs, 66, 12);
        arrowTo = gw;
        activeEndpoint = rightWrist;
      }
    } else if (kind === 'elbow_flexion') {
      const curlT = frame.motionProgress;
      if (isLeftArm) {
        leftElbow = { x: ls.x - 6, y: ls.y + 34 };
        const faAng = lerp(-8, -148, curlT);
        leftWrist = polarPt(leftElbow, 32, faAng);
        const gw = polarPt(leftElbow, 32, -148);
        targetGhostLeft = [ls, leftElbow, gw];
        arrowFrom = polarPt(leftElbow, 32, -12);
        arrowTo = gw;
        activeEndpoint = leftWrist;
      }
      if (isRightArm) {
        rightElbow = { x: rs.x + 6, y: rs.y + 34 };
        const faAng = lerp(8, 148, curlT);
        rightWrist = polarPt(rightElbow, 32, faAng);
        const gw = polarPt(rightElbow, 32, 148);
        targetGhostRight = [rs, rightElbow, gw];
        arrowFrom = polarPt(rightElbow, 32, 12);
        arrowTo = gw;
        activeEndpoint = rightWrist;
      }
    } else if (kind === 'shoulder_rotation') {
      const rotT = frame.motionProgress;
      if (isLeftArm) {
        leftElbow = { x: ls.x - 10, y: ls.y + 32 };
        const rotAng = lerp(-90, -162, rotT);
        leftWrist = polarPt(leftElbow, 32, rotAng);
        const gw = polarPt(leftElbow, 32, -162);
        targetGhostLeft = [ls, leftElbow, gw];
        arrowFrom = polarPt(leftElbow, 32, -92);
        arrowTo = gw;
        activeEndpoint = leftWrist;
      }
      if (isRightArm) {
        rightElbow = { x: rs.x + 10, y: rs.y + 32 };
        const rotAng = lerp(90, 162, rotT);
        rightWrist = polarPt(rightElbow, 32, rotAng);
        const gw = polarPt(rightElbow, 32, 162);
        targetGhostRight = [rs, rightElbow, gw];
        arrowFrom = polarPt(rightElbow, 32, 92);
        arrowTo = gw;
        activeEndpoint = rightWrist;
      }
    } else if (kind === 'hand_to_head') {
      const t = frame.motionProgress;
      if (isLeftArm) {
        const tgtE = { x: ls.x - 28, y: 48 };
        const tgtW = { x: cx - 10, y: 20 };
        leftElbow = lerpPt(leftElbow, tgtE, t);
        leftWrist = lerpPt(leftWrist, tgtW, t);
        targetGhostLeft = [ls, tgtE, tgtW];
        arrowFrom = { x: ls.x - 12, y: 120 };
        arrowTo = tgtW;
        activeEndpoint = leftWrist;
      }
      if (isRightArm) {
        const tgtE = { x: rs.x + 28, y: 48 };
        const tgtW = { x: cx + 10, y: 20 };
        rightElbow = lerpPt(rightElbow, tgtE, t);
        rightWrist = lerpPt(rightWrist, tgtW, t);
        targetGhostRight = [rs, tgtE, tgtW];
        arrowFrom = { x: rs.x + 12, y: 120 };
        arrowTo = tgtW;
        activeEndpoint = rightWrist;
      }
    } else if (kind === 'cross_body_reach') {
      const t = frame.motionProgress;
      if (isLeftArm) {
        const tgtE = { x: cx - 2, y: 80 };
        const tgtW = { x: rs.x + 6, y: 66 };
        leftElbow = lerpPt(leftElbow, tgtE, t);
        leftWrist = lerpPt(leftWrist, tgtW, t);
        targetGhostLeft = [ls, tgtE, tgtW];
        arrowFrom = { x: ls.x - 10, y: 120 };
        arrowTo = tgtW;
        activeEndpoint = leftWrist;
      }
      if (isRightArm) {
        const tgtE = { x: cx + 2, y: 80 };
        const tgtW = { x: ls.x - 6, y: 66 };
        rightElbow = lerpPt(rightElbow, tgtE, t);
        rightWrist = lerpPt(rightWrist, tgtW, t);
        targetGhostRight = [rs, tgtE, tgtW];
        arrowFrom = { x: rs.x + 10, y: 120 };
        arrowTo = tgtW;
        activeEndpoint = rightWrist;
      }
    } else if (kind === 'clasped_arm_raise') {
      const t = frame.motionProgress;
      const claspPeak = { x: cx, y: 16 };
      const claspRest = { x: cx, y: 118 };
      const claspCur = lerpPt(claspRest, claspPeak, t);
      leftElbow = lerpPt({ x: ls.x - 14, y: 94 }, { x: ls.x - 14, y: 40 }, t);
      rightElbow = lerpPt({ x: rs.x + 14, y: 94 }, { x: rs.x + 14, y: 40 }, t);
      leftWrist = { x: claspCur.x - 4, y: claspCur.y };
      rightWrist = { x: claspCur.x + 4, y: claspCur.y };
      targetGhostLeft = [ls, { x: ls.x - 14, y: 40 }, { x: claspPeak.x - 4, y: claspPeak.y }];
      targetGhostRight = [rs, { x: rs.x + 14, y: 40 }, { x: claspPeak.x + 4, y: claspPeak.y }];
      arrowFrom = claspRest;
      arrowTo = claspPeak;
      activeEndpoint = claspCur;
    } else if (kind === 'neck_tilt' || kind === 'neck_turn') {
      activeEndpoint = { x: cx, y: 36 };
    } else if (kind === 'shoulder_shrug_roll') {
      activeEndpoint = isLeftArm ? ls : rs;
    } else if (kind === 'knee_lift') {
      activeEndpoint = { x: cx + (isLeftArm ? -20 : 20), y: 146 };
    } else if (kind === 'hip_abduction') {
      activeEndpoint = { x: cx + (isLeftArm ? -44 : 44), y: 192 };
    } else if (kind === 'heel_toe_raise') {
      activeEndpoint = { x: cx, y: 196 };
    }

    return (
      <g>
        <FrontStandingBase
          cx={cx}
          headTiltDeg={neckTiltDeg}
          neckTurnShiftX={neckShiftX}
          shoulderLiftLeft={shoulderLiftLeft}
          shoulderLiftRight={shoulderLiftRight}
          leftHipHighlight={
            (kind === 'knee_lift' || kind === 'hip_abduction' || kind === 'heel_toe_raise') &&
            isLeftArm
          }
          rightHipHighlight={
            (kind === 'knee_lift' || kind === 'hip_abduction' || kind === 'heel_toe_raise') &&
            isRightArm
          }
          leftKneeLift={kind === 'knee_lift' && isLeftArm ? 34 * frame.motionProgress : 0}
          rightKneeLift={kind === 'knee_lift' && isRightArm ? 34 * frame.motionProgress : 0}
          leftLegOutDeg={kind === 'hip_abduction' && isLeftArm ? 36 * frame.motionProgress : 0}
          rightLegOutDeg={kind === 'hip_abduction' && isRightArm ? 36 * frame.motionProgress : 0}
          tiptoeLift={kind === 'heel_toe_raise' ? 12 * frame.motionProgress : 0}
        />

        {targetGhostLeft && (
          <TargetGhostLimb p1={targetGhostLeft[0]} p2={targetGhostLeft[1]} p3={targetGhostLeft[2]} />
        )}
        {targetGhostRight && (
          <TargetGhostLimb p1={targetGhostRight[0]} p2={targetGhostRight[1]} p3={targetGhostRight[2]} />
        )}

        <ActiveOrMutedLimb
          p1={ls}
          p2={leftElbow}
          p3={leftWrist}
          active={
            kind !== 'neck_tilt' &&
            kind !== 'neck_turn' &&
            kind !== 'knee_lift' &&
            kind !== 'hip_abduction' &&
            kind !== 'heel_toe_raise' &&
            isLeftArm
          }
          width={12}
        />
        <ActiveOrMutedLimb
          p1={rs}
          p2={rightElbow}
          p3={rightWrist}
          active={
            kind !== 'neck_tilt' &&
            kind !== 'neck_turn' &&
            kind !== 'knee_lift' &&
            kind !== 'hip_abduction' &&
            kind !== 'heel_toe_raise' &&
            isRightArm
          }
          width={12}
        />

        {kind === 'clasped_arm_raise' && (
          <circle
            cx={(leftWrist.x + rightWrist.x) / 2}
            cy={leftWrist.y}
            r={9}
            fill={EMERALD}
            fillOpacity={0.25}
            stroke={EMERALD_DARK}
            strokeWidth={2}
          />
        )}

        {isMoving && arrowFrom && arrowTo && (
          <DirectionalArrow from={arrowFrom} to={arrowTo} bulge={isLeftArm && !isRightArm ? 18 : -18} />
        )}

        {isHoldOrLower && (
          <HoldPulseTarget
            pt={activeEndpoint}
            phase={frame.phase}
            countdownSec={frame.countdownSec}
            targetHoldSec={targetHoldSec}
            phaseProgress={frame.phaseProgress}
            compact={compact}
            bounds={stageFrame.holdBounds}
          />
        )}
      </g>
    );
  };

  return (
    <div
      data-testid="animated-rehab-coach-model"
      data-coach-phase={frame.phase}
      className={`relative flex flex-col items-center justify-center overflow-hidden ${className}`}
    >
      <svg
        viewBox={`${stageFrame.x} ${stageFrame.y} ${stageFrame.w} ${stageFrame.h}`}
        preserveAspectRatio="xMidYMid meet"
        className="w-full h-full max-h-full select-none"
        role="img"
        aria-label={`น้องหมอเทคมาสคอตสาธิต ${exercise.name_th} (${sideLabel})`}
      >
        <defs>
          <linearGradient id="coachBgGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#f0f9ff" />
            <stop offset="100%" stopColor="#ecfdf5" />
          </linearGradient>
        </defs>
        <rect
          x={stageFrame.x}
          y={stageFrame.y}
          width={stageFrame.w}
          height={stageFrame.h}
          rx={12}
          fill="url(#coachBgGrad)"
        />

        {renderScene()}

        {/* Compact Active Side Pill Top-Left (clear of otter head) */}
        <g transform={`translate(${stageFrame.leftPillX}, ${stageFrame.pillY})`}>
          <rect x={-35} y={-9} width={70} height={18} rx={9} fill="#10b981" />
          <circle cx={-26} cy={0} r={2.6} fill="#ffffff" />
          <text x={4} y={3.2} textAnchor="middle" fill="#ffffff" fontSize={9.5} fontWeight={900}>
            {sideLabel}
          </text>
        </g>

        {/* Compact Live Angle Pill Top-Right (clear of otter head) */}
        <g transform={`translate(${stageFrame.rightPillX}, ${stageFrame.pillY})`}>
          <rect
            x={-37}
            y={-9}
            width={74}
            height={18}
            rx={9}
            fill="#ffffff"
            fillOpacity={0.94}
            stroke="#0284c7"
            strokeWidth={1.6}
          />
          <text x={0} y={3.2} textAnchor="middle" fill="#0f172a" fontSize={9.5} fontWeight={900}>
            {currentDeg}° → {targetRom}°
          </text>
        </g>
      </svg>
    </div>
  );
}

export const AnimatedRehabCoachModel = React.memo(AnimatedRehabCoachModelInner);

export interface Zone2PipCoachOverlayProps {
  exercise: ExercisePrescriptionOption;
  targetSide: ExerciseSideCode;
  visible: boolean;
  onToggleVisible: () => void;
}

export function Zone2PipCoachOverlay({
  exercise,
  targetSide,
  visible,
  onToggleVisible,
}: Zone2PipCoachOverlayProps) {
  const elapsedSec = useSyncedCoachClock(visible);
  const frame = computeCoachTimelineFrame(elapsedSec, Math.max(1, Math.round(exercise.hold_sec || 3)));
  const limbCue = resolveElderlyLimbCueTh(exercise, targetSide);

  if (!visible) {
    return (
      <div className="absolute top-20 left-3 z-20 animate-toast-in transition-all duration-250 ease-out">
        <button
          type="button"
          onClick={onToggleVisible}
          data-testid="kiosk-pip-coach-toggle"
          aria-label="ซ่อน/แสดงหุ่นโค้ชบนจอ"
          className="inline-flex items-center gap-1.5 bg-white/95 border-2 border-emerald-500 text-emerald-900 font-extrabold text-sm px-3.5 py-1.5 rounded-full shadow-lg hover:bg-emerald-50 cursor-pointer active:scale-95 transition-all duration-250 ease-out"
        >
          <Eye className="w-4 h-4 shrink-0" />
          <span>แสดงหุ่นตัวอย่าง</span>
        </button>
      </div>
    );
  }

  return (
    <div
      data-testid="kiosk-pip-coach-box"
      className="absolute top-20 left-3 z-20 w-48 h-56 sm:w-60 sm:h-72 md:w-64 md:h-76 bg-white/96 border-[3px] border-emerald-500 rounded-3xl shadow-2xl p-2 flex flex-col items-center justify-between gap-1.5 animate-popup-spring transition-all duration-250 ease-out"
    >
      <div className="w-full flex items-center justify-between gap-1.5 shrink-0">
        <span
          className={`flex-1 inline-flex items-center justify-center gap-1.5 rounded-xl px-2 py-1 text-sm sm:text-base font-black text-white shadow-xs truncate ${
            limbCue.direction === 'left'
              ? 'bg-gradient-to-r from-indigo-700 to-blue-600'
              : limbCue.direction === 'both'
                ? 'bg-gradient-to-r from-emerald-600 to-teal-600'
                : 'bg-gradient-to-r from-[#0F4C81] to-emerald-600'
          }`}
        >
          <span className="truncate">{limbCue.headlineTh}</span>
        </span>
        <button
          type="button"
          onClick={onToggleVisible}
          data-testid="kiosk-pip-coach-toggle"
          title="ซ่อน/แสดงหุ่นโค้ชบนจอ"
          aria-label="ซ่อน/แสดงหุ่นโค้ชบนจอ"
          className="inline-flex items-center justify-center rounded-xl bg-slate-100 hover:bg-slate-200 p-1.5 text-slate-700 shrink-0 cursor-pointer"
        >
          <EyeOff className="w-4 h-4 shrink-0" />
        </button>
      </div>

      <AnimatedRehabCoachModel
        exercise={exercise}
        targetSide={targetSide}
        cycleElapsedSec={elapsedSec}
        className="w-full flex-1 min-h-0"
      />

      <div className="w-full text-center rounded-xl bg-emerald-50 border border-emerald-200 py-0.5 px-2 text-xs sm:text-sm font-black text-emerald-900 shrink-0 truncate">
        {frame.phaseLabelTh}
      </div>
    </div>
  );
}

export interface Zone4CoachStageAndStepsProps {
  exercise: ExercisePrescriptionOption;
  targetSide: ExerciseSideCode;
  steps: [string, string, string];
  holdTargetSec: number;
  targetRomDeg: number;
  onExpandPreview: () => void;
}

export function Zone4CoachStageAndSteps({
  exercise,
  targetSide,
  holdTargetSec,
  targetRomDeg,
  onExpandPreview,
}: Zone4CoachStageAndStepsProps) {
  const elapsedSec = useSyncedCoachClock(true);
  const frame = computeCoachTimelineFrame(elapsedSec, holdTargetSec);
  const limbCue = resolveElderlyLimbCueTh(exercise, targetSide);
  const conciseName = formatConciseExerciseNameTh(exercise.name_th);

  useEffect(() => {
    resetSharedCoachClockEpoch();
  }, [exercise.exercise_id, targetSide]);

  const stepItems = [
    { stepNum: 1 as const, title: '1. เตรียม', shortCue: 'ท่าเริ่มต้น' },
    { stepNum: 2 as const, title: '2. ขยับ', shortCue: `ถึง ${targetRomDeg}°` },
    { stepNum: 3 as const, title: '3. ค้างท่า', shortCue: `ค้าง ${Math.round(holdTargetSec)} วิ` },
  ];

  return (
    <>
      {/* Left 7 Columns: EXTRA-LARGE Zoomed Nong MorTech Mascot Coach Stage */}
      <div className="col-span-7 bg-gradient-to-b from-sky-50 via-white to-emerald-50/70 border-2 border-emerald-400 rounded-2xl p-1.5 flex flex-col items-center justify-between relative overflow-hidden card-hover-lift">
        <div className="w-full flex items-center justify-between gap-1 shrink-0 z-10 mb-1">
          <span className="inline-flex items-center gap-1 bg-emerald-600 text-white text-[11px] sm:text-xs font-black px-2 py-0.5 rounded-full truncate">
            <span className="inline-block w-1.5 h-1.5 rounded-full bg-white animate-pulse shrink-0" />
            <span className="truncate">น้องหมอเทคสาธิต • {frame.phaseLabelTh}</span>
          </span>
          <button
            type="button"
            onClick={onExpandPreview}
            data-testid="kiosk-coach-expand-btn"
            className="inline-flex items-center gap-1 bg-white hover:bg-sky-50 border border-blue-300 text-blue-800 text-[11px] font-extrabold px-2 py-0.5 rounded-lg shrink-0 cursor-pointer shadow-xs active:scale-95 transition"
          >
            <Maximize2 className="w-3 h-3 shrink-0" />
            <span>ขยายใหญ่</span>
          </button>
        </div>

        <AnimatedRehabCoachModel
          exercise={exercise}
          targetSide={targetSide}
          cycleElapsedSec={elapsedSec}
          className="w-full h-64 sm:h-72 flex-1"
        />
      </div>

      {/* Right 5 Columns: Compact Arm Banner + 3 Sleek Horizontal Step Rows */}
      <div className="col-span-5 flex flex-col justify-between gap-1.5">
        {/* Compact Side / Arm Callout Card */}
        <div
          data-testid="zone4-elderly-arm-callout"
          className={`shrink-0 rounded-xl border-2 px-2.5 py-2 flex flex-col justify-center gap-1 shadow-xs ${
            limbCue.direction === 'left'
              ? 'border-blue-500 bg-gradient-to-r from-indigo-700 via-blue-600 to-sky-600 text-white'
              : limbCue.direction === 'both'
                ? 'border-emerald-500 bg-gradient-to-r from-emerald-600 via-teal-600 to-cyan-600 text-white'
                : 'border-emerald-400 bg-gradient-to-r from-[#0F4C81] via-[#0284C7] to-[#059669] text-white'
          }`}
        >
          <div className="flex items-center justify-between gap-1.5">
            <span className="text-xs sm:text-sm font-extrabold text-white/95 truncate">
              {conciseName}
            </span>
            <span className="shrink-0 rounded-full bg-white/20 border border-white/40 px-2 py-0.5 text-[11px] font-black text-white whitespace-nowrap">
              เป้า {targetRomDeg}°
            </span>
          </div>

          <div className="flex items-center justify-center gap-2">
            {limbCue.direction === 'left' && (
              <svg viewBox="0 0 36 36" className="w-6 h-6 sm:w-7 sm:h-7 shrink-0 text-amber-300" fill="none" stroke="currentColor" strokeWidth="4" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <path d="M30 18H6" />
                <path d="M16 8L6 18l10 10" />
              </svg>
            )}
            {limbCue.direction === 'both' && (
              <svg viewBox="0 0 36 36" className="w-6 h-6 sm:w-7 sm:h-7 shrink-0 text-amber-300" fill="none" stroke="currentColor" strokeWidth="3.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <path d="M13 8L5 18l8 10" />
                <path d="M23 8l8 10-8 10" />
                <path d="M7 18h22" />
              </svg>
            )}
            <span className="text-xl sm:text-2xl md:text-3xl font-black tracking-tight leading-none drop-shadow-sm truncate">
              {limbCue.headlineTh}
            </span>
            {limbCue.direction === 'right' && (
              <svg viewBox="0 0 36 36" className="w-6 h-6 sm:w-7 sm:h-7 shrink-0 text-amber-300" fill="none" stroke="currentColor" strokeWidth="4" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <path d="M6 18h24" />
                <path d="M20 8l10 10-10 10" />
              </svg>
            )}
          </div>
        </div>

        {/* 3 Clear, Tactile Elderly-Readable Step Rows */}
        <div className="flex-1 flex flex-col justify-between gap-1.5">
          {stepItems.map((item) => {
            const isActiveStep = frame.activeStepNumber === item.stepNum;
            return (
              <div
                key={item.stepNum}
                data-testid={`kiosk-step-card-${item.stepNum}`}
                data-active-step={isActiveStep ? 'true' : 'false'}
                className={`flex-1 rounded-xl px-3 py-2 flex items-center justify-between gap-2 transition-all duration-300 ${
                  isActiveStep
                    ? 'shadow-sm border-2 border-emerald-500 bg-emerald-50 ring-2 ring-emerald-300/60'
                    : 'bg-slate-50 border border-sky-200 opacity-95'
                }`}
              >
                <div className="flex items-center gap-2 min-w-0">
                  <span
                    className={`text-xs sm:text-sm font-black px-2.5 py-0.5 rounded-full shrink-0 ${
                      isActiveStep ? 'bg-blue-700 text-white' : 'bg-sky-100 text-blue-900'
                    }`}
                  >
                    {item.title}
                  </span>
                  <p className="text-sm sm:text-base font-black text-slate-900 leading-tight truncate">
                    {item.shortCue}
                  </p>
                </div>
                {isActiveStep && (
                  <span className="inline-flex items-center gap-1 bg-emerald-600 text-white text-[11px] font-black px-2.5 py-1 rounded-full animate-toast-in animate-status-halo whitespace-nowrap shrink-0">
                    <Play className="w-2.5 h-2.5 fill-current shrink-0" />
                    <span>กำลังทำขั้นนี้</span>
                  </span>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </>
  );
}

export interface ExpandedCoachModalProps {
  open: boolean;
  exercise: ExercisePrescriptionOption;
  targetSide: ExerciseSideCode;
  targetSideLabel: string;
  steps: [string, string, string];
  onClose: () => void;
}

export function ExpandedCoachModal({
  open,
  exercise,
  targetSide,
  targetSideLabel,
  steps,
  onClose,
}: ExpandedCoachModalProps) {
  const elapsedSec = useSyncedCoachClock(open);
  if (!open) return null;
  const frame = computeCoachTimelineFrame(elapsedSec, Math.max(1, Math.round(exercise.hold_sec || 3)));

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label={`ตัวอย่างหุ่นโค้ชท่า ${exercise.name_th}`}
      className="fixed inset-0 z-50 bg-slate-900/65 backdrop-blur-sm flex items-center justify-center p-4 animate-backdrop-fade"
    >
      <div className="w-full max-w-2xl bg-white border-2 border-emerald-400 rounded-3xl p-5 shadow-2xl space-y-3 animate-popup-spring">
        <div className="flex items-center justify-between gap-3 border-b border-slate-200 pb-2.5">
          <div className="min-w-0">
            <p className="text-xs font-extrabold text-emerald-700">
              หุ่นโค้ชสาธิต 60FPS ({frame.phaseLabelTh})
            </p>
            <h3 className="text-lg md:text-xl font-black text-blue-950 truncate">
              {exercise.name_th} — {targetSideLabel}
            </h3>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="inline-flex items-center gap-1.5 bg-blue-600 hover:bg-blue-700 text-white font-extrabold px-4 py-2 rounded-xl text-sm shrink-0 cursor-pointer"
          >
            <span>ปิดหน้าต่าง</span>
            <X className="w-4 h-4 shrink-0" />
          </button>
        </div>

        <div className="w-full h-72 sm:h-80 rounded-2xl border-2 border-sky-200 overflow-hidden bg-sky-50">
          <AnimatedRehabCoachModel
            exercise={exercise}
            targetSide={targetSide}
            cycleElapsedSec={elapsedSec}
            className="w-full h-full"
          />
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
          {steps.map((text, idx) => {
            const active = frame.activeStepNumber === idx + 1;
            return (
              <div
                key={idx}
                className={`rounded-xl p-2.5 border transition-all duration-300 ${
                  active
                    ? '-translate-y-0.5 bg-emerald-50/70 border-emerald-500 ring-2 ring-emerald-300 shadow-md'
                    : 'bg-slate-50 border-sky-200'
                }`}
              >
                <p className="text-xs font-black text-blue-900">
                  {idx === 0 ? '1. เตรียม' : idx === 1 ? '2. ขยับ' : '3. ค้างท่า'}
                </p>
                <p className="mt-1 text-xs md:text-sm font-bold text-slate-800 leading-snug">
                  {text}
                </p>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
