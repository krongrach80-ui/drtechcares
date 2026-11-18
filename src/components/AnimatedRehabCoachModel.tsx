import React, { useEffect, useState } from 'react';
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

function runSharedCoachClock(nowMs: number) {
  if (coachClockListeners.size === 0) {
    coachRafHandle = 0;
    return;
  }
  if (coachEpochMs === 0) {
    coachEpochMs = nowMs;
  }
  const elapsedSec = ((nowMs - coachEpochMs) / 1000) % COACH_CYCLE_DURATION_SEC;
  coachClockListeners.forEach((listener) => listener(elapsedSec));
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
  | 'bobath_arm_raise'
  | 'bobath_forward_push'
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
    return 'bobath_forward_push';
  }
  if (combined.includes('arm_raise') || combined.includes('ประสานมือยกแขน') || exercise.expected_plane === 'bilateral') {
    return 'bobath_arm_raise';
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
const INACTIVE = '#94a3b8';
const INACTIVE_DARK = '#64748b';
const TORSO_FILL = '#1e3a8a';
const TORSO_STROKE = '#0f172a';
const SKIN_FILL = '#fed7aa';

function ActiveOrMutedLimb({
  p1,
  p2,
  p3,
  active,
  width = 11,
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
          strokeOpacity={0.35}
          strokeWidth={width + 10}
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      )}
      <path
        d={pathD}
        fill="none"
        stroke={active ? EMERALD : INACTIVE}
        strokeWidth={active ? width + 1 : width - 1}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <circle
        cx={p1.x}
        cy={p1.y}
        r={active ? 6.5 : 4.5}
        fill={active ? '#ffffff' : '#cbd5e1'}
        stroke={active ? EMERALD_DARK : INACTIVE_DARK}
        strokeWidth={2.5}
      />
      <circle
        cx={p2.x}
        cy={p2.y}
        r={active ? 6 : 4}
        fill={active ? '#ffffff' : '#cbd5e1'}
        stroke={active ? EMERALD_DARK : INACTIVE_DARK}
        strokeWidth={2.2}
      />
      {showEndpointDot && p3 && (
        <circle
          cx={endPt.x}
          cy={endPt.y}
          r={active ? 7.5 : 4.5}
          fill={active ? EMERALD : '#cbd5e1'}
          stroke={active ? '#ffffff' : INACTIVE_DARK}
          strokeWidth={2.5}
        />
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
}: {
  pt: Pt;
  phase: CoachPhaseId;
  countdownSec: number;
  targetHoldSec: number;
  phaseProgress: number;
  compact?: boolean;
}) {
  if (phase === 'hold') {
    const ringRadius = 11 + Math.sin(phaseProgress * Math.PI * 4) * 3.5;
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
        <g transform={`translate(${Math.min(200, Math.max(58, pt.x))}, ${Math.max(18, pt.y - 20)})`}>
          <rect x={-46} y={-12} width={92} height={22} rx={11} fill={EMERALD_DARK} />
          <text
            x={0}
            y={3}
            textAnchor="middle"
            fill="#ffffff"
            fontSize={compact ? 10 : 10.5}
            fontWeight={800}
          >
            ⭕ ค้าง {targetHoldSec}วิ ({countdownSec}s)
          </text>
        </g>
      </g>
    );
  }
  if (phase === 'lower') {
    return (
      <g transform={`translate(${Math.min(205, Math.max(55, pt.x))}, ${Math.max(18, pt.y - 16)})`}>
        <rect x={-34} y={-11} width={68} height={22} rx={11} fill="#0284c7" />
        <text x={0} y={4} textAnchor="middle" fill="#ffffff" fontSize={11.5} fontWeight={900}>
          +1 ครั้ง!
        </text>
      </g>
    );
  }
  return null;
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
    <g>
      {/* Subtle floor shadow */}
      <ellipse cx={cx} cy={210} rx={52} ry={6} fill="#cbd5e1" opacity={0.6} />

      {/* Legs */}
      <ActiveOrMutedLimb p1={leftHip} p2={leftKnee} p3={leftAnkle} active={leftHipHighlight} width={10} />
      <ActiveOrMutedLimb p1={rightHip} p2={rightKnee} p3={rightAnkle} active={rightHipHighlight} width={10} />

      {/* Torso */}
      <path
        d={`M ${leftShoulder.x} ${leftShoulder.y} L ${rightShoulder.x} ${rightShoulder.y} L ${rightHip.x} ${rightHip.y} L ${leftHip.x} ${leftHip.y} Z`}
        fill={TORSO_FILL}
        stroke={TORSO_STROKE}
        strokeWidth={2}
        strokeLinejoin="round"
      />
      {/* Neck */}
      <line x1={cx} y1={52} x2={cx} y2={64} stroke={SKIN_FILL} strokeWidth={9} strokeLinecap="round" />

      {/* Head with tilt & friendly face orientation */}
      <g transform={`translate(${headCenter.x}, ${headCenter.y}) rotate(${headTiltDeg})`}>
        <circle cx={0} cy={0} r={17} fill={SKIN_FILL} stroke={TORSO_STROKE} strokeWidth={2.2} />
        <path d="M -16 -5 C -14 -17, 14 -17, 16 -5 C 10 -10, -10 -10, -16 -5 Z" fill="#334155" />
        <circle cx={-5 + neckTurnShiftX} cy={-1} r={2} fill="#0f172a" />
        <circle cx={5 + neckTurnShiftX} cy={-1} r={2} fill="#0f172a" />
        <path
          d={`M ${(-4 + neckTurnShiftX).toFixed(1)} 6 Q ${neckTurnShiftX.toFixed(1)} 10 ${(4 + neckTurnShiftX).toFixed(1)} 6`}
          fill="none"
          stroke="#0f172a"
          strokeWidth={1.8}
          strokeLinecap="round"
        />
      </g>
    </g>
  );
}

export interface AnimatedRehabCoachModelProps {
  exercise: ExercisePrescriptionOption;
  targetSide?: ExerciseSideCode;
  /** Optional explicit cycle time; if omitted, subscribes to 60FPS shared clock cleanly */
  cycleElapsedSec?: number;
  compact?: boolean;
  className?: string;
}

export function AnimatedRehabCoachModel({
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

  const renderScene = () => {
    // 1. SIDE / 3/4 VIEW: SHOULDER FLEXION / SCAP PROTRACT / BOBATH FORWARD PUSH / SIT-TO-STAND
    if (
      kind === 'shoulder_flexion' ||
      kind === 'scap_protract' ||
      kind === 'bobath_forward_push'
    ) {
      // Face right (+x is forward)
      const cx = 104;
      const shoulder = { x: cx + 2, y: 66 };
      const hip = { x: cx - 2, y: 124 };
      const backArmEnd = { x: cx - 10, y: 122 };
      const knee = { x: cx + 2, y: 164 };
      const ankle = { x: cx - 2, y: 204 };

      const reachMaxDeg = kind === 'shoulder_flexion' ? Math.min(165, Math.max(60, targetRom)) : 90;
      const currentArmAngle =
        kind === 'shoulder_flexion' ? reachMaxDeg * frame.motionProgress : 90;
      const forwardPushPx =
        kind === 'scap_protract' || kind === 'bobath_forward_push'
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
        <g>
          <ellipse cx={cx} cy={208} rx={48} ry={6} fill="#cbd5e1" opacity={0.65} />
          {/* Standing Legs */}
          <ActiveOrMutedLimb p1={hip} p2={knee} p3={ankle} active={false} width={10} />
          {/* Background Arm (muted if single side, or parallel active if BOTH) */}
          {kind !== 'bobath_forward_push' &&
            (resolvedSide === 'BOTH' ? (
              <ActiveOrMutedLimb
                p1={{ x: shoulder.x - 10, y: shoulder.y - 3 }}
                p2={{ x: curElbow.x - 10, y: curElbow.y - 4 }}
                p3={{ x: curWrist.x - 10, y: curWrist.y - 5 }}
                active
                width={9}
              />
            ) : (
              <ActiveOrMutedLimb
                p1={shoulder}
                p2={{ x: cx - 6, y: 95 }}
                p3={backArmEnd}
                active={false}
                width={9}
              />
            ))}
          {/* Torso */}
          <rect
            x={cx - 12}
            y={60}
            width={24}
            height={65}
            rx={11}
            fill={TORSO_FILL}
            stroke={TORSO_STROKE}
            strokeWidth={2}
          />
          {/* Head looking forward (+x) */}
          <circle cx={cx + 2} cy={38} r={16} fill={SKIN_FILL} stroke={TORSO_STROKE} strokeWidth={2.2} />
          <path d={`M ${(cx - 14)} 34 C ${(cx - 12)} 19, ${(cx + 16)} 19, ${(cx + 15)} 30 Z`} fill="#334155" />
          <circle cx={cx + 11} cy={36} r={2} fill="#0f172a" />
          <path d={`M ${cx + 17} 38 L ${cx + 21} 40 L ${cx + 17} 42`} fill={SKIN_FILL} stroke={TORSO_STROKE} strokeWidth={1.6} />

          {/* Dashed Target Ghost + Star */}
          <TargetGhostLimb p1={shoulder} p2={targetElbow} p3={targetWrist} />
          {(kind === 'scap_protract' || kind === 'bobath_forward_push') && (
            <text x={targetWrist.x + 12} y={targetWrist.y + 5} fontSize={14}>
              ⭐
            </text>
          )}
          {/* Active Forearm/Upper Arm */}
          <ActiveOrMutedLimb
            p1={{ x: shoulder.x + forwardPushPx * 0.45, y: shoulder.y }}
            p2={curElbow}
            p3={curWrist}
            active
            width={11}
          />
          {kind === 'bobath_forward_push' && (
            <circle cx={curWrist.x} cy={curWrist.y} r={9} fill="#10b981" fillOpacity={0.28} stroke="#047857" strokeWidth={2} />
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
          <g>
            {/* Elbow & Forearm */}
            <ActiveOrMutedLimb p1={forearmStart} p2={wrist} active width={14} showEndpointDot={false} />
            {/* Rotating Palm Oval + Thumb */}
            <g transform={`translate(${wrist.x + 36}, ${wrist.y - 8}) scale(1, ${flipScaleY.toFixed(2)})`}>
              <ellipse cx={0} cy={0} rx={34} ry={17} fill={SKIN_FILL} stroke={EMERALD_DARK} strokeWidth={3} />
              <line x1={-6} y1={-12} x2={-14} y2={-26} stroke={EMERALD} strokeWidth={7} strokeLinecap="round" />
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
          <g>
            <ActiveOrMutedLimb p1={forearmStart} p2={wrist} active width={13} showEndpointDot={false} />
            <circle cx={fingerBase.x - 4} cy={fingerBase.y + 4} r={18} fill={SKIN_FILL} stroke={EMERALD_DARK} strokeWidth={2.5} />
            {/* Index finger */}
            <ActiveOrMutedLimb p1={fingerBase} p2={{ x: wrist.x + 50, y: wrist.y - 18 }} p3={indexTip} active width={9} />
            {/* Thumb */}
            <ActiveOrMutedLimb p1={{ x: wrist.x + 18, y: wrist.y + 10 }} p2={{ x: wrist.x + 48, y: wrist.y + 14 }} p3={thumbTip} active width={9} />
            {/* Pinch Star */}
            {frame.motionProgress > 0.78 && (
              <circle cx={(indexTip.x + thumbTip.x) / 2} cy={(indexTip.y + thumbTip.y) / 2} r={12} fill={EMERALD} fillOpacity={0.35} stroke={EMERALD_DARK} strokeWidth={2.5} />
            )}
            <text x={130} y={188} textAnchor="middle" fill="#0f172a" fontSize={12} fontWeight={800}>
              {frame.motionProgress > 0.75 ? '👌 จีบนิ้วโป้ง-ชี้' : '🖐 กางนิ้วออก'} • {sideLabel}
            </text>
            {isHoldOrLower && (
              <HoldPulseTarget
                pt={{ x: wrist.x + 68, y: wrist.y - 6 }}
                phase={frame.phase}
                countdownSec={frame.countdownSec}
                targetHoldSec={targetHoldSec}
                phaseProgress={frame.phaseProgress}
                compact={compact}
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
        <g>
          {/* Zoomed Forearm */}
          <ActiveOrMutedLimb p1={{ x: 38, y: 158 }} p2={wristJoint} active width={15} showEndpointDot={false} />
          {/* Ghost Wrist */}
          <g transform={`translate(${palmOrigin.x}, ${palmOrigin.y}) rotate(${-targetWristTilt})`} opacity={0.35}>
            <rect x={4} y={-14} width={44} height={28} rx={12} fill="none" stroke={EMERALD_DARK} strokeWidth={2.5} strokeDasharray="5 3" />
          </g>
          {/* Active Articulated Wrist + 5 Fingers */}
          <g transform={`translate(${palmOrigin.x}, ${palmOrigin.y}) rotate(${-wristTiltDeg})`}>
            <rect x={2} y={-15} width={44} height={30} rx={12} fill={SKIN_FILL} stroke={EMERALD_DARK} strokeWidth={3} />
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
            {spreadRatio > 0.5 ? '🖐 กระดกข้อมือ + กาง 5 นิ้ว' : '✊ ผ่อนข้อมือ / กำหลวมๆ'} • {sideLabel}
          </text>
          {isHoldOrLower && (
            <HoldPulseTarget
              pt={{ x: 145, y: 74 }}
              phase={frame.phase}
              countdownSec={frame.countdownSec}
              targetHoldSec={targetHoldSec}
              phaseProgress={frame.phaseProgress}
              compact={compact}
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
          <g>
            {/* Chair */}
            <path d="M 62 78 L 68 142 L 116 142 L 116 202 M 68 142 L 68 202" fill="none" stroke="#64748b" strokeWidth={6} strokeLinecap="round" strokeLinejoin="round" />
            <ActiveOrMutedLimb p1={hip} p2={knee} p3={ankle} active width={11} />
            {/* Torso */}
            <line x1={hip.x} y1={hip.y} x2={shoulder.x} y2={shoulder.y} stroke={TORSO_FILL} strokeWidth={20} strokeLinecap="round" />
            <circle cx={shoulder.x + 2} cy={shoulder.y - 22} r={15} fill={SKIN_FILL} stroke={TORSO_STROKE} strokeWidth={2} />
            <ActiveOrMutedLimb p1={shoulder} p2={lerpPt(shoulder, hand, 0.55)} p3={hand} active={false} width={8} />
            {isMoving && <DirectionalArrow from={{ x: 112, y: 120 }} to={{ x: 148, y: 58 }} bulge={-16} />}
            {isHoldOrLower && (
              <HoldPulseTarget
                pt={{ x: shoulder.x, y: shoulder.y - 36 }}
                phase={frame.phase}
                countdownSec={frame.countdownSec}
                targetHoldSec={targetHoldSec}
                phaseProgress={frame.phaseProgress}
                compact={compact}
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
        <g>
          {/* Chair */}
          <path d={`M ${chairX - 24} 78 L ${chairX - 18} 142 L 136 142 L 136 202 M ${chairX - 18} 142 L ${chairX - 18} 202`} fill="none" stroke="#64748b" strokeWidth={6} strokeLinecap="round" strokeLinejoin="round" />
          <ActiveOrMutedLimb
            p1={hip}
            p2={knee}
            p3={ankle}
            active={kind === 'knee_extension'}
            width={11}
          />
          <line x1={hip.x} y1={hip.y} x2={shoulder.x} y2={shoulder.y} stroke={TORSO_FILL} strokeWidth={20} strokeLinecap="round" />
          <circle cx={shoulder.x + 2} cy={shoulder.y - 20} r={15} fill={SKIN_FILL} stroke={TORSO_STROKE} strokeWidth={2} />
          <ActiveOrMutedLimb
            p1={shoulder}
            p2={lerpPt(shoulder, reachHand, 0.5)}
            p3={reachHand}
            active={kind === 'trunk_reach'}
            width={9}
          />
          {kind === 'trunk_reach' && (
            <text x={212} y={94} fontSize={16}>
              ⭐
            </text>
          )}
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
            />
          )}
        </g>
      );
    }

    // 4. FRONT MIRROR FULL-BODY VIEW (All remaining shoulder, elbow, bobath, neck, shrug, leg exercises)
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
      const maxAbd = Math.min(165, Math.max(45, targetRom));
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
    } else if (kind === 'bobath_arm_raise') {
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
          width={10}
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
          width={10}
        />

        {kind === 'bobath_arm_raise' && (
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
        viewBox="0 0 260 220"
        className="w-full h-full max-h-full select-none"
        role="img"
        aria-label={`หุ่นโค้ชสาธิต ${exercise.name_th} (${sideLabel})`}
      >
        <defs>
          <linearGradient id="coachBgGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#f0f9ff" />
            <stop offset="100%" stopColor="#ecfdf5" />
          </linearGradient>
        </defs>
        <rect x={0} y={0} width={260} height={220} rx={16} fill="url(#coachBgGrad)" />

        {/* Live Angle Pill Top-Right */}
        <g transform="translate(192, 16)">
          <rect x={-52} y={-11} width={104} height={20} rx={10} fill="#ffffff" stroke="#0284c7" strokeWidth={1.6} />
          <text x={0} y={3} textAnchor="middle" fill="#0f172a" fontSize={10} fontWeight={900}>
            {currentDeg}° ➔ {targetRom}°
          </text>
        </g>

        {/* Active Side Pill Top-Left */}
        <g transform="translate(64, 16)">
          <rect x={-54} y={-11} width={108} height={20} rx={10} fill="#10b981" />
          <text x={0} y={3} textAnchor="middle" fill="#ffffff" fontSize={10} fontWeight={900}>
            🎯 {sideLabel}
          </text>
        </g>

        {renderScene()}
      </svg>
    </div>
  );
}

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

  if (!visible) {
    return (
      <div className="absolute top-14 left-3 z-20">
        <button
          type="button"
          onClick={onToggleVisible}
          data-testid="kiosk-pip-coach-toggle"
          aria-label="ซ่อน/แสดงหุ่นโค้ชบนจอ"
          className="bg-white/90 backdrop-blur-md border-2 border-emerald-500 text-emerald-900 font-extrabold text-xs px-3 py-1 rounded-full shadow-lg hover:bg-emerald-50 cursor-pointer active:scale-95 transition"
        >
          👁️ ซ่อน/แสดงหุ่นโค้ชบนจอ
        </button>
      </div>
    );
  }

  return (
    <div
      data-testid="kiosk-pip-coach-box"
      className="absolute top-14 left-3 z-20 w-28 h-32 md:w-36 md:h-40 bg-white/90 backdrop-blur-md border-2 border-emerald-500 rounded-2xl shadow-xl p-1.5 flex flex-col items-center justify-between"
    >
      <div className="w-full flex items-center justify-between gap-1 shrink-0">
        <span className="text-[10px] font-black text-emerald-900 truncate">
          🎬 {frame.phaseLabelTh}
        </span>
        <button
          type="button"
          onClick={onToggleVisible}
          data-testid="kiosk-pip-coach-toggle"
          title="ซ่อน/แสดงหุ่นโค้ชบนจอ"
          aria-label="ซ่อน/แสดงหุ่นโค้ชบนจอ"
          className="rounded-md bg-emerald-100 hover:bg-emerald-200 px-1.5 py-0.5 text-[10px] font-extrabold text-emerald-950 shrink-0 cursor-pointer"
        >
          👁️ ซ่อน
        </button>
      </div>

      <AnimatedRehabCoachModel
        exercise={exercise}
        targetSide={targetSide}
        cycleElapsedSec={elapsedSec}
        compact
        className="w-full flex-1 min-h-0"
      />
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
  steps,
  holdTargetSec,
  targetRomDeg,
  onExpandPreview,
}: Zone4CoachStageAndStepsProps) {
  const elapsedSec = useSyncedCoachClock(true);
  const frame = computeCoachTimelineFrame(elapsedSec, holdTargetSec);

  useEffect(() => {
    resetSharedCoachClockEpoch();
  }, [exercise.exercise_id, targetSide]);

  const stepItems = [
    { stepNum: 1 as const, title: '1. เตรียม', text: steps[0] || 'เตรียมท่าตั้งต้นให้มั่นคง' },
    { stepNum: 2 as const, title: '2. ขยับ', text: steps[1] || `เคลื่อนไหวช้าๆ ถึง ${targetRomDeg}°` },
    { stepNum: 3 as const, title: '3. ค้างท่า', text: steps[2] || `ค้างท่าไว้ ${holdTargetSec.toFixed(1)} วินาที` },
  ];

  return (
    <>
      {/* Left 3 Columns: Live Animated Coach Stage */}
      <div className="col-span-3 bg-gradient-to-b from-sky-50 to-emerald-50/70 border-2 border-emerald-400 rounded-2xl p-2 flex flex-col items-center justify-between relative overflow-hidden">
        <div className="w-full flex items-center justify-between gap-1 shrink-0 z-10">
          <span className="bg-emerald-600 text-white text-[10px] md:text-xs font-black px-2 py-0.5 rounded-full truncate">
            🎬 หุ่นสาธิต ({frame.phaseLabelTh})
          </span>
          <button
            type="button"
            onClick={onExpandPreview}
            data-testid="kiosk-coach-expand-btn"
            className="bg-white hover:bg-sky-50 border border-blue-300 text-blue-800 text-[10px] md:text-xs font-extrabold px-2 py-0.5 rounded-lg shrink-0 cursor-pointer shadow-xs active:scale-95 transition"
          >
            🔍 ขยายใหญ่
          </button>
        </div>

        <AnimatedRehabCoachModel
          exercise={exercise}
          targetSide={targetSide}
          cycleElapsedSec={elapsedSec}
          className="w-full h-24 md:h-28"
        />
      </div>

      {/* Middle 6 Columns: 3 Synchronized Step Cards (Full Text, Zero Truncation ...) */}
      <div className="col-span-6 grid grid-cols-3 gap-2">
        {stepItems.map((item) => {
          const isActiveStep = frame.activeStepNumber === item.stepNum;
          return (
            <div
              key={item.stepNum}
              data-testid={`kiosk-step-card-${item.stepNum}`}
              data-active-step={isActiveStep ? 'true' : 'false'}
              className={`rounded-2xl p-2 flex flex-col justify-between transition-all duration-200 ${
                isActiveStep
                  ? 'bg-gradient-to-b from-emerald-50 to-sky-50 border-2 border-emerald-500 ring-2 ring-emerald-300/60 shadow-md scale-[1.01]'
                  : 'bg-slate-50 border border-sky-200 opacity-90'
              }`}
            >
              <div className="flex flex-wrap items-center justify-between gap-1">
                <span
                  className={`text-xs font-black px-2 py-0.5 rounded-full ${
                    isActiveStep ? 'bg-blue-700 text-white' : 'bg-sky-100 text-blue-900'
                  }`}
                >
                  {item.title}
                </span>
                {isActiveStep && (
                  <span className="bg-emerald-600 text-white text-[10px] font-black px-1.5 py-0.5 rounded-full animate-pulse whitespace-nowrap">
                    ▶ กำลังทำขั้นนี้
                  </span>
                )}
              </div>

              <p className="mt-1 leading-snug text-[11px] md:text-xs font-bold text-slate-800 break-words">
                {item.text}
              </p>
            </div>
          );
        })}
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
      className="fixed inset-0 z-50 bg-slate-900/65 backdrop-blur-sm flex items-center justify-center p-4"
    >
      <div className="w-full max-w-2xl bg-white border-2 border-emerald-400 rounded-3xl p-5 shadow-2xl space-y-3">
        <div className="flex items-center justify-between gap-3 border-b border-slate-200 pb-2.5">
          <div className="min-w-0">
            <p className="text-xs font-extrabold text-emerald-700">
              🎬 หุ่นโค้ชสาธิต 60FPS ({frame.phaseLabelTh})
            </p>
            <h3 className="text-lg md:text-xl font-black text-blue-950 truncate">
              {exercise.name_th} — {targetSideLabel}
            </h3>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="bg-blue-600 hover:bg-blue-700 text-white font-extrabold px-4 py-2 rounded-xl text-sm shrink-0 cursor-pointer"
          >
            ปิดหน้าต่าง ✕
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
                className={`rounded-xl p-2.5 border ${
                  active
                    ? 'bg-emerald-50 border-emerald-500 ring-2 ring-emerald-300'
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
