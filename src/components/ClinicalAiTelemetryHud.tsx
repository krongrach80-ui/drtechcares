import React from 'react';
import type {
  Hand21Biomechanics,
  RotationCycleEvaluation,
} from '@drtechcares/pose-engine';
import { Activity, CheckCircle2, Cpu, Hand, RotateCw, ShieldAlert, X } from 'lucide-react';

export interface ClinicalAiHudKinematics {
  elbowAngleDeg?: number | null;
  shoulderHikeRatio?: number;
  compensations?: string[];
  xyzVector?: { x: number; y: number; z: number } | null;
  handBiomechanics?: {
    left: Hand21Biomechanics | null;
    right: Hand21Biomechanics | null;
    active: Hand21Biomechanics | null;
  } | null;
}

export interface ClinicalAiTelemetryHudProps {
  visible: boolean;
  onClose: () => void;
  exerciseNameTh: string;
  targetRomDeg: number;
  currentRomDeg: number;
  kinematics: ClinicalAiHudKinematics | null;
  rotationEval: RotationCycleEvaluation | null;
  distanceM: number;
  wrongSideRaised: boolean;
  wrongPlaneRaised: boolean;
  trunkTiltDeg: number;
}

const FINGER_LABELS_TH: ReadonlyArray<{
  key: keyof Hand21Biomechanics['fingers'];
  shortTh: string;
  barClass: string;
}> = [
  { key: 'thumb', shortTh: 'โป้ง', barClass: 'bg-amber-400' },
  { key: 'index', shortTh: 'ชี้', barClass: 'bg-sky-400' },
  { key: 'middle', shortTh: 'กลาง', barClass: 'bg-emerald-400' },
  { key: 'ring', shortTh: 'นาง', barClass: 'bg-violet-400' },
  { key: 'pinky', shortTh: 'ก้อย', barClass: 'bg-pink-400' },
];

/**
 * Toggleable clinical biomechanics HUD for competition judges and physical therapists.
 * Displays live 3D joint angles, 21-point MediaPipe finger kinematics, wrist roll/extension,
 * rotation-cycle verification, and real-time posture compensation guard flags.
 */
export function ClinicalAiTelemetryHud({
  visible,
  onClose,
  exerciseNameTh,
  targetRomDeg,
  currentRomDeg,
  kinematics,
  rotationEval,
  distanceM,
  wrongSideRaised,
  wrongPlaneRaised,
  trunkTiltDeg,
}: ClinicalAiTelemetryHudProps) {
  if (!visible) return null;

  const activeHand: Hand21Biomechanics | null =
    kinematics?.handBiomechanics?.active ??
    kinematics?.handBiomechanics?.right ??
    kinematics?.handBiomechanics?.left ??
    null;
  const bothHandsDetected = Boolean(
    kinematics?.handBiomechanics?.left && kinematics?.handBiomechanics?.right,
  );
  const elbowDeg = kinematics?.elbowAngleDeg ?? null;
  const shoulderHike = kinematics?.shoulderHikeRatio ?? 0;
  const compensations = kinematics?.compensations ?? [];
  const hasTrunkLean = trunkTiltDeg > 12 || compensations.includes('TRUNK_LEAN');
  const hasShoulderHike = shoulderHike > 0.14 || compensations.includes('SHOULDER_HIKE');
  const hasElbowCheat = compensations.includes('ELBOW_FLEXION_CHEAT');
  const sideOk = !wrongSideRaised && !wrongPlaneRaised;

  const guardItems = [
    { label: 'ข้าง/ระนาบ', ok: sideOk, detail: wrongSideRaised ? 'ผิดข้าง' : wrongPlaneRaised ? 'ผิดระนาบ' : 'ถูกต้อง' },
    { label: 'ลำตัวตรง', ok: !hasTrunkLean, detail: `${trunkTiltDeg.toFixed(1)}°` },
    { label: 'ไม่ยักไหล่', ok: !hasShoulderHike, detail: hasShoulderHike ? 'ยกไหล่' : 'ผ่อนคลาย' },
    { label: 'ข้อศอก', ok: !hasElbowCheat, detail: elbowDeg !== null ? `${Math.round(elbowDeg)}°` : '-' },
  ];

  return (
    <aside
      data-testid="clinical-ai-pro-telemetry-hud"
      aria-label="แผงวิเคราะห์ชีวกลศาสตร์ AI เชิงลึก"
      className="pointer-events-auto absolute top-14 left-3 z-25 w-72 sm:w-80 rounded-2xl border border-cyan-400/50 bg-slate-950/86 text-slate-100 p-3 shadow-2xl backdrop-blur-md space-y-2.5 animate-popup-spring"
    >
      {/* Header */}
      <div className="flex items-center justify-between gap-2 border-b border-slate-700/80 pb-2">
        <div className="flex items-center gap-1.5 min-w-0">
          <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-lg bg-cyan-500/20 border border-cyan-400/40 text-cyan-300">
            <Cpu className="h-3.5 w-3.5" />
          </span>
          <div className="min-w-0">
            <div className="text-xs font-black tracking-wide text-cyan-300 leading-none">
              AI BIOMECHANICS PRO HUD
            </div>
            <div className="text-[10.5px] font-semibold text-slate-400 truncate mt-0.5">
              {exerciseNameTh}
            </div>
          </div>
        </div>
        <button
          type="button"
          onClick={onClose}
          aria-label="ปิด AI Pro HUD"
          className="flex h-6 w-6 shrink-0 items-center justify-center rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 cursor-pointer"
        >
          <X className="h-3.5 w-3.5" />
        </button>
      </div>

      {/* 3D Primary Kinematics Grid */}
      <div className="grid grid-cols-3 gap-1.5">
        <div className="rounded-xl bg-slate-900/90 border border-slate-700/80 px-2 py-1.5">
          <div className="text-[10px] font-bold text-slate-400">3D ROM</div>
          <div className="text-base font-black text-emerald-400 tabular-nums leading-tight">
            {currentRomDeg.toFixed(1)}°
          </div>
          <div className="text-[10px] font-semibold text-slate-500">เป้า {Math.round(targetRomDeg)}°</div>
        </div>
        <div className="rounded-xl bg-slate-900/90 border border-slate-700/80 px-2 py-1.5">
          <div className="text-[10px] font-bold text-slate-400">มุมข้อศอก</div>
          <div className="text-base font-black text-sky-300 tabular-nums leading-tight">
            {elbowDeg !== null ? `${Math.round(elbowDeg)}°` : '--'}
          </div>
          <div className="text-[10px] font-semibold text-slate-500">
            {elbowDeg !== null ? (elbowDeg >= 150 ? 'เหยียดตรง' : 'งอข้อศอก') : 'รอข้อมูล'}
          </div>
        </div>
        <div className="rounded-xl bg-slate-900/90 border border-slate-700/80 px-2 py-1.5">
          <div className="text-[10px] font-bold text-slate-400">ระยะกล้อง</div>
          <div className="text-base font-black text-amber-300 tabular-nums leading-tight">
            {distanceM > 0 ? `${distanceM.toFixed(1)} ม.` : '--'}
          </div>
          <div className="text-[10px] font-semibold text-slate-500">
            {kinematics?.xyzVector ? `Z:${kinematics.xyzVector.z.toFixed(2)}` : 'HD 1280p'}
          </div>
        </div>
      </div>

      {/* 21-Point MediaPipe Hand & 5-Finger Live Matrix */}
      <div className="rounded-xl bg-slate-900/90 border border-slate-700/80 p-2 space-y-1.5">
        <div className="flex items-center justify-between gap-1">
          <span className="inline-flex items-center gap-1 text-[11px] font-black text-cyan-300">
            <Hand className="h-3.5 w-3.5" />
            <span>MediaPipe 21-Pt Hand ROI</span>
          </span>
          <span
            className={`rounded-full px-2 py-0.5 text-[10px] font-black ${
              activeHand
                ? 'bg-emerald-500/20 border border-emerald-400/50 text-emerald-300'
                : 'bg-slate-800 text-slate-400'
            }`}
          >
            {bothHandsDetected
              ? 'จับได้ 2 มือ (42 จุด)'
              : activeHand
                ? `กาง ${activeHand.extendedFingerCount}/5 นิ้ว`
                : 'รอตรวจจับมือ'}
          </span>
        </div>

        <div className="grid grid-cols-5 gap-1">
          {FINGER_LABELS_TH.map((f) => {
            const fState = activeHand?.fingers[f.key];
            const pct = Math.round(fState?.extensionPct ?? 0);
            return (
              <div key={f.key} className="text-center space-y-0.5">
                <div className="h-8 w-full rounded-md bg-slate-800 border border-slate-700/70 flex items-end overflow-hidden p-0.5">
                  <div
                    className={`w-full rounded-sm transition-all duration-150 ${f.barClass}`}
                    style={{ height: `${Math.max(8, Math.min(100, pct))}%` }}
                  />
                </div>
                <div className="text-[9.5px] font-extrabold text-slate-300">{f.shortTh}</div>
              </div>
            );
          })}
        </div>

        <div className="flex items-center justify-between gap-1 text-[10.5px] font-bold text-slate-300 pt-0.5">
          <span>
            ข้อมือ: <strong className="text-white">{activeHand ? `${Math.round(activeHand.wristExtensionDeg)}°` : '--'}</strong>
          </span>
          <span>
            พลิกมือ:{' '}
            <strong className="text-cyan-300">
              {activeHand
                ? `${activeHand.palmOrientationTh} (${activeHand.forearmRollSignedDeg >= 0 ? '+' : ''}${Math.round(activeHand.forearmRollSignedDeg)}°)`
                : '--'}
            </strong>
          </span>
        </div>
        {activeHand?.activePinchFingerTh && (
          <div className="rounded-lg bg-emerald-500/20 border border-emerald-400/40 px-2 py-0.5 text-[10.5px] font-black text-emerald-300 text-center">
            ตรวจพบการจีบนิ้วโป้งแตะ{activeHand.activePinchFingerTh}
          </div>
        )}
      </div>

      {/* Real-Rotation Cycle Gate Indicator (Shown when active on rotation/shrug/forearm exercises) */}
      {rotationEval && (
        <div className="rounded-xl bg-slate-900/90 border border-sky-500/40 px-2.5 py-1.5 flex items-center justify-between gap-2">
          <div className="flex items-center gap-1.5 min-w-0">
            <RotateCw className="h-3.5 w-3.5 text-sky-400 shrink-0" />
            <span className="text-[11px] font-extrabold text-slate-200 truncate">
              วงหมุนสะสม {Math.round(rotationEval.sweptArcDeg)}°
            </span>
          </div>
          <span
            className={`shrink-0 rounded-full px-2 py-0.5 text-[10px] font-black ${
              rotationEval.rotationVerified
                ? 'bg-emerald-500 text-slate-950'
                : 'bg-amber-500/20 text-amber-300 border border-amber-400/40'
            }`}
          >
            {rotationEval.rotationVerified ? 'หมุนจริงผ่าน' : 'กำลังหมุน'}
          </span>
        </div>
      )}

      {/* 4 Real-Time Posture Compensation Guard Lights */}
      <div className="space-y-1">
        <div className="flex items-center gap-1 text-[10.5px] font-black text-slate-400">
          <Activity className="h-3 w-3 text-emerald-400" />
          <span>ระบบตรวจจับการโกงท่า (Compensation Guards)</span>
        </div>
        <div className="grid grid-cols-2 gap-1.5">
          {guardItems.map((g) => (
            <div
              key={g.label}
              className={`rounded-lg border px-2 py-1 flex items-center justify-between gap-1 text-[10.5px] font-extrabold ${
                g.ok
                  ? 'border-emerald-500/40 bg-emerald-500/10 text-emerald-300'
                  : 'border-rose-500/60 bg-rose-500/20 text-rose-200'
              }`}
            >
              <span className="inline-flex items-center gap-1 truncate">
                {g.ok ? (
                  <CheckCircle2 className="h-3 w-3 shrink-0" />
                ) : (
                  <ShieldAlert className="h-3 w-3 shrink-0" />
                )}
                <span className="truncate">{g.label}</span>
              </span>
              <span className="tabular-nums shrink-0">{g.detail}</span>
            </div>
          ))}
        </div>
      </div>
    </aside>
  );
}
