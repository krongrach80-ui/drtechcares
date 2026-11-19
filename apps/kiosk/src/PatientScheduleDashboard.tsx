import React, { useState } from 'react';
import { useClinicalI18n } from '@drtechcares/clinical-i18n';
import type {
  DrTechCaresApiClient,
  PatientCourseRecord,
  PatientRecord,
} from '@drtechcares/api-client';
import { CLINICAL_EXERCISES, parseExerciseCodeAndSide } from '@drtechcares/pose-engine';
import { Activity, Calendar, Coffee, LogOut, PhoneCall, Sparkles } from 'lucide-react';

interface PatientScheduleDashboardProps {
  patient: PatientRecord;
  course: PatientCourseRecord | null;
  apiClient: DrTechCaresApiClient;
  onStartRehab: () => void;
  onStartMinigame: () => void;
  onLogout: () => void;
}

const DAY_NAMES = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];

export function PatientScheduleDashboard({
  patient,
  course,
  apiClient,
  onStartRehab,
  onStartMinigame,
  onLogout,
}: PatientScheduleDashboardProps) {
  const { t, pick } = useClinicalI18n();
  const [staffCalled, setStaffCalled] = useState(false);

  const todayIndex = new Date().getDay();
  const schedule = course?.weeklySchedule ?? [];
  const todayPlan = schedule.find((d) => d.dayIndex === todayIndex) ?? {
    dayIndex: todayIndex,
    dayType: 'training' as const,
    label: 'Daily Rehabilitation Session',
    exercises: [
      {
        exerciseId: 'STROKE_SH_FLEX_RIGHT',
        side: 'right' as const,
        sets: 2,
        reps: 5,
        holdSec: 3,
        targetMinDeg: 95,
        targetMaxDeg: 135,
        allowedStarterSide: 'right' as const,
      },
    ],
  };

  const handleStaffCall = async () => {
    await apiClient.createStaffCall('KIOSK-OPD-01', `Patient ${patient.fullName} requested help on dashboard`, patient.id);
    setStaffCalled(true);
  };

  return (
    <div className="mx-auto max-w-5xl space-y-6 animate-view-enter">
      {/* Patient Welcome Banner */}
      <div className="flex flex-wrap items-center justify-between gap-4 rounded-3xl bg-white p-6 shadow-lg border border-slate-200">
        <div>
          <span className="inline-block rounded-full bg-teal-50 px-3 py-1 text-xs font-semibold text-teal-700">
            HN: {patient.hn} • {patient.diagnosis}
          </span>
          <h1 className="mt-2 text-2xl font-bold text-slate-900">{patient.fullName}</h1>
          <p className="text-sm text-slate-500">{t('raiseBothHandsPrompt')}</p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <button
            onClick={handleStaffCall}
            className="inline-flex items-center gap-1.5 rounded-xl border border-amber-300 bg-amber-50 px-4 py-2.5 text-sm font-semibold text-amber-800 hover:bg-amber-100"
          >
            <PhoneCall className="h-4 w-4" />
            {t('callStaffBtn')}
          </button>
          <button
            onClick={onLogout}
            className="inline-flex items-center gap-1.5 rounded-xl border border-slate-200 px-4 py-2.5 text-sm font-semibold text-slate-600 hover:bg-slate-50"
          >
            <LogOut className="h-4 w-4" />
            {t('logoutBtn')}
          </button>
        </div>
      </div>

      {staffCalled && (
        <div className="rounded-2xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm font-semibold text-amber-900 animate-toast-in">
          {t('staffCalledConfirm')}
        </div>
      )}

      {/* Weekly Calendar: Separating Training Days vs Rest Days */}
      <div className="rounded-3xl bg-white p-6 shadow-lg border border-slate-200">
        <h2 className="text-base font-bold text-slate-800 flex items-center gap-2">
          <Calendar className="h-5 w-5 text-teal-600" />
          Weekly Rehabilitation Schedule
        </h2>
        <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4 lg:grid-cols-7">
          {DAY_NAMES.map((dayName, idx) => {
            const entry = schedule.find((d) => d.dayIndex === idx);
            const isTraining = entry ? entry.dayType === 'training' : [1, 3, 5].includes(idx);
            const isToday = idx === todayIndex;
            return (
              <div
                key={dayName}
                style={{ animationDelay: `${idx * 45}ms` }}
                className={`calendar-day-tile animate-card-enter rounded-2xl border p-3 text-center ${
                  isToday
                    ? 'border-teal-500 bg-teal-50/70 ring-2 ring-teal-200 animate-status-halo'
                    : isTraining
                      ? 'border-sky-200 bg-sky-50/40'
                      : 'border-slate-200 bg-slate-50/60'
                }`}
              >
                <p className="text-xs font-bold uppercase text-slate-500">
                  {dayName} {isToday ? '• TODAY' : ''}
                </p>
                <div className="mt-2 flex justify-center">
                  {isTraining ? (
                    <span className="inline-flex items-center gap-1 rounded-full bg-teal-600 px-2.5 py-0.5 text-[11px] font-bold text-white">
                      <Activity className="h-3 w-3" /> {t('trainingDayBadge')}
                    </span>
                  ) : (
                    <span className="inline-flex items-center gap-1 rounded-full bg-slate-200 px-2.5 py-0.5 text-[11px] font-semibold text-slate-700">
                      <Coffee className="h-3 w-3" /> {t('restDayBadge')}
                    </span>
                  )}
                </div>
                <p className="mt-2 text-xs text-slate-600">
                  {isTraining ? `${entry?.exercises.length ?? 2} Exercises` : 'Light Mobility'}
                </p>
              </div>
            );
          })}
        </div>
      </div>

      {/* Action Cards */}
      <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
        <div className="rounded-3xl bg-gradient-to-br from-teal-600 to-emerald-700 p-6 text-white shadow-xl">
          <div className="flex items-center justify-between">
            <span className="rounded-full bg-white/20 px-3 py-1 text-xs font-bold">
              {todayPlan.dayType === 'training' ? t('trainingDayBadge') : t('restDayBadge')}
            </span>
            <Activity className="h-7 w-7 text-teal-100" />
          </div>
          <h3 className="mt-4 text-xl font-bold">{t('startRehabBtn')}</h3>
          <ul className="mt-3 space-y-1.5 text-sm text-teal-50">
            {(todayPlan.exercises.length > 0 ? todayPlan.exercises : [{ exerciseId: 'STROKE_SH_FLEX_RIGHT', side: 'right', sets: 2, reps: 5, holdSec: 3 }]).map(
              (item, idx) => {
                const parsed = parseExerciseCodeAndSide(item.exerciseId, (item as { side?: string }).side);
                const ex = CLINICAL_EXERCISES.find((e) => e.id === parsed.baseCode || e.code === parsed.baseCode) ?? CLINICAL_EXERCISES[0];
                const sideBadge =
                  parsed.side === 'LEFT' ? 'ซ้าย' : parsed.side === 'BOTH' ? 'ทั้งสองข้าง' : 'ขวา';
                return (
                  <li key={`${item.exerciseId}-${idx}`}>
                    • {pick(ex.name)} [{sideBadge}] — {item.sets} sets × {item.reps} reps ({item.holdSec}s hold)
                  </li>
                );
              },
            )}
          </ul>
          <button
            onClick={onStartRehab}
            className="mt-6 w-full rounded-2xl bg-white py-3.5 text-base font-bold text-teal-800 shadow hover:bg-teal-50"
          >
            {t('startRehabBtn')}
          </button>
        </div>

        <div className="rounded-3xl bg-gradient-to-br from-indigo-600 to-violet-700 p-6 text-white shadow-xl">
          <div className="flex items-center justify-between">
            <span className="rounded-full bg-white/20 px-3 py-1 text-xs font-bold">99-Card Bank • Touchless Hand</span>
            <Sparkles className="h-7 w-7 text-indigo-100" />
          </div>
          <h3 className="mt-4 text-xl font-bold">{t('startCognitiveGameBtn')}</h3>
          <p className="mt-3 text-sm text-indigo-100">
            {t('minigameRightYes')} / {t('minigameLeftNo')}
          </p>
          <button
            onClick={onStartMinigame}
            className="mt-10 w-full rounded-2xl bg-white py-3.5 text-base font-bold text-indigo-800 shadow hover:bg-indigo-50"
          >
            {t('startCognitiveGameBtn')}
          </button>
        </div>
      </div>
    </div>
  );
}

