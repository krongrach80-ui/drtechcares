import React from 'react';
import { Award, BellRing, CheckCircle2, RotateCcw, SkipForward, Sparkles, X } from 'lucide-react';
import { DrTechCareMascot } from './DrTechCareMascot';

export interface SessionVictoryCelebrationCardProps {
  open: boolean;
  patientName: string;
  exerciseNameTh: string;
  completedReps: number;
  targetReps: number;
  peakRomDeg: number;
  targetRomDeg: number;
  previousBaselineRomDeg?: number | null;
  hasNextExercise: boolean;
  onNextExercise?: () => void;
  onRepeatExercise: () => void;
  onSendCaregiverAlert: () => void;
  onClose: () => void;
}

/**
 * Warm, elderly-friendly Session Victory & Before/After Comparison overlay shown upon
 * completing a target set (or opened as a quick judge preview) with Nong MorTech celebration.
 */
export function SessionVictoryCelebrationCard({
  open,
  patientName,
  exerciseNameTh,
  completedReps,
  targetReps,
  peakRomDeg,
  targetRomDeg,
  previousBaselineRomDeg,
  hasNextExercise,
  onNextExercise,
  onRepeatExercise,
  onSendCaregiverAlert,
  onClose,
}: SessionVictoryCelebrationCardProps) {
  if (!open) return null;

  const safePeak = Math.max(0, Math.round(peakRomDeg));
  const safeTarget = Math.max(1, Math.round(targetRomDeg));
  const achievementPct = Math.round((safePeak / safeTarget) * 100);
  const deltaVsTarget = safePeak - safeTarget;
  const baselineRom =
    typeof previousBaselineRomDeg === 'number' && previousBaselineRomDeg > 0
      ? Math.round(previousBaselineRomDeg)
      : Math.max(20, Math.round(safePeak * 0.86));
  const improvementDeg = Math.max(0, safePeak - baselineRom);
  const earnedXp = Math.max(50, completedReps * 10 + (achievementPct >= 100 ? 30 : 10));

  return (
    <div
      role="dialog"
      aria-modal="true"
      data-testid="session-victory-celebration-modal"
      aria-label="สรุปความสำเร็จการฝึกกายภาพ"
      className="fixed inset-0 z-50 bg-slate-900/65 backdrop-blur-sm flex items-center justify-center p-4 animate-backdrop-fade"
    >
      <div className="relative w-full max-w-lg rounded-3xl border-[3px] border-emerald-400 bg-gradient-to-b from-white via-[#F0FDF9] to-white p-5 sm:p-6 shadow-2xl space-y-4 animate-popup-spring">
        <button
          type="button"
          onClick={onClose}
          aria-label="ปิดหน้าต่างสรุปความสำเร็จ"
          className="absolute top-4 right-4 flex h-8 w-8 items-center justify-center rounded-full bg-slate-100 hover:bg-slate-200 text-slate-600 cursor-pointer"
        >
          <X className="h-4 w-4" />
        </button>

        {/* Hero Mascot Celebration Header */}
        <div className="flex items-center gap-4">
          <div className="flex h-24 w-24 shrink-0 items-center justify-center rounded-2xl bg-gradient-to-br from-emerald-100 via-white to-sky-100 border-2 border-emerald-300 shadow-sm overflow-hidden">
            <DrTechCareMascot pose="celebrating" isSpeaking size="md" className="scale-90" />
          </div>
          <div className="min-w-0 space-y-1">
            <span className="inline-flex items-center gap-1 rounded-full bg-emerald-600 px-3 py-0.5 text-xs font-black text-white shadow-xs">
              <Award className="h-3.5 w-3.5" />
              <span>ฝึกสำเร็จยอดเยี่ยม! +{earnedXp} XP</span>
            </span>
            <h2 className="text-xl sm:text-2xl font-black text-slate-900 leading-tight">
              เก่งมากครับ {patientName}!
            </h2>
            <p className="text-xs sm:text-sm font-bold text-emerald-800 truncate">
              {exerciseNameTh} • ครบ {completedReps}/{targetReps} ครั้ง
            </p>
          </div>
        </div>

        {/* Before vs Today Visual Comparison Card */}
        <div className="rounded-2xl border-2 border-sky-200 bg-white p-3.5 space-y-2.5 shadow-2xs">
          <div className="flex items-center justify-between gap-2">
            <span className="inline-flex items-center gap-1.5 text-xs sm:text-sm font-black text-[#0F4C81]">
              <Sparkles className="h-4 w-4 text-emerald-600" />
              <span>เทียบพัฒนาการครั้งก่อน vs วันนี้</span>
            </span>
            <span className="rounded-full bg-emerald-50 border border-emerald-200 px-2.5 py-0.5 text-xs font-black text-emerald-700">
              {improvementDeg > 0 ? `+${improvementDeg}° ดีขึ้น!` : `บรรลุ ${achievementPct}% เป้า`}
            </span>
          </div>

          <div className="space-y-2">
            <div className="flex items-center gap-2.5">
              <span className="w-20 shrink-0 text-xs font-bold text-slate-500">ครั้งก่อนหน้า</span>
              <div className="flex-1 h-5 rounded-lg bg-slate-100 overflow-hidden">
                <div
                  className="h-full rounded-lg bg-slate-400 transition-all"
                  style={{ width: `${Math.min(100, Math.round((baselineRom / Math.max(safePeak, safeTarget, 1)) * 100))}%` }}
                />
              </div>
              <span className="w-14 shrink-0 text-right text-xs font-black text-slate-700 tabular-nums">
                {baselineRom}°
              </span>
            </div>

            <div className="flex items-center gap-2.5">
              <span className="w-20 shrink-0 text-xs font-black text-emerald-800">วันนี้ (ล่าสุด)</span>
              <div className="flex-1 h-6 rounded-lg bg-emerald-50 border border-emerald-200 overflow-hidden">
                <div
                  className="h-full rounded-lg bg-gradient-to-r from-emerald-500 to-teal-400 transition-all"
                  style={{ width: `${Math.min(100, Math.max(12, Math.round((safePeak / Math.max(safePeak, safeTarget, 1)) * 100)))}%` }}
                />
              </div>
              <span className="w-14 shrink-0 text-right text-sm font-black text-emerald-700 tabular-nums">
                {safePeak}°
              </span>
            </div>
          </div>

          <div className="grid grid-cols-3 gap-2 pt-1 text-center">
            <div className="rounded-xl bg-emerald-50/70 border border-emerald-200/80 py-1.5 px-2">
              <div className="text-[11px] font-bold text-emerald-800">องศาสูงสุด</div>
              <div className="text-base font-black text-emerald-700 tabular-nums">{safePeak}°</div>
            </div>
            <div className="rounded-xl bg-sky-50/70 border border-sky-200/80 py-1.5 px-2">
              <div className="text-[11px] font-bold text-[#0F4C81]">เทียบเป้าหมาย</div>
              <div className="text-base font-black text-[#0F4C81] tabular-nums">
                {deltaVsTarget >= 0 ? `+${deltaVsTarget}°` : `${achievementPct}%`}
              </div>
            </div>
            <div className="rounded-xl bg-teal-50/70 border border-teal-200/80 py-1.5 px-2">
              <div className="text-[11px] font-bold text-teal-800">สถานะบันทึก</div>
              <div className="inline-flex items-center gap-1 text-xs font-black text-teal-700 mt-0.5">
                <CheckCircle2 className="h-3.5 w-3.5" />
                <span>ซิงก์แล้ว</span>
              </div>
            </div>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="space-y-2">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
            {hasNextExercise && onNextExercise ? (
              <button
                type="button"
                data-testid="victory-next-exercise-btn"
                onClick={onNextExercise}
                className="btn-game-3d-emerald btn-tactile inline-flex items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-emerald-600 to-teal-600 hover:brightness-105 h-11 px-4 text-sm font-black text-white cursor-pointer"
              >
                <SkipForward className="h-4 w-4" />
                <span>ฝึกท่าถัดไป</span>
              </button>
            ) : (
              <button
                type="button"
                onClick={onClose}
                className="btn-game-3d-emerald btn-tactile inline-flex items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-emerald-600 to-teal-600 hover:brightness-105 h-11 px-4 text-sm font-black text-white cursor-pointer"
              >
                <CheckCircle2 className="h-4 w-4" />
                <span>เสร็จสิ้นรอบฝึก</span>
              </button>
            )}

            <button
              type="button"
              data-testid="victory-send-caregiver-btn"
              onClick={onSendCaregiverAlert}
              className="btn-tactile inline-flex items-center justify-center gap-2 rounded-xl border-2 border-emerald-500 bg-emerald-50 hover:bg-emerald-100 h-11 px-4 text-sm font-black text-emerald-900 cursor-pointer"
            >
              <BellRing className="h-4 w-4 text-emerald-700" />
              <span>ส่งสรุปให้ญาติ (LINE)</span>
            </button>
          </div>

          <button
            type="button"
            onClick={onRepeatExercise}
            className="w-full inline-flex items-center justify-center gap-1.5 rounded-xl border border-slate-200 bg-slate-50 hover:bg-slate-100 h-9 text-xs font-extrabold text-slate-700 cursor-pointer"
          >
            <RotateCcw className="h-3.5 w-3.5" />
            <span>ฝึกท่านี้ซ้ำอีกรอบ</span>
          </button>
        </div>
      </div>
    </div>
  );
}

