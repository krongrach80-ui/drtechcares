import React, { useEffect, useMemo, useState } from 'react';
import { AdminPortalApp } from '@apps/admin';
import { KioskPortalApp, PhysicalCardGame, type PhysicalCardGameSummary } from '@apps/kiosk';
import { PtPortalApp } from '@apps/pt';
import {
  DrTechCaresApiClient,
  isPatientCompletedTodayInStorage,
  markPatientCompletedTodayInStorage,
  resolveTodayPatientSchedule,
  type UserAccount,
} from '@drtechcares/api-client';
import { useClinicalI18n, type ClinicalLang } from '@drtechcares/clinical-i18n';
import { buildTtsProxyUrl } from './components/VoiceCoachBanner';
import { PatientPersonalPortal } from './PatientPersonalPortal';
import { StaffLoginModalOrPage, type StaffAuthIntent } from './StaffLoginModalOrPage';
import {
  KioskBiometricLoginView,
  type KioskLoginSuccessPayload,
} from './components/KioskBiometricLoginView';
import { CheckCircle2, Shuffle } from 'lucide-react';
import {
  RandomHandStartPrompt,
  pickRandomHandSide,
  type HandChallengeSide,
} from './components/RandomHandStartPrompt';
import { SmartRehabKioskUI } from './components/SmartRehabKioskUI';
import {
  mapTodayExercisesToPrescriptions,
  type ExercisePrescriptionOption,
} from './hooks/useKioskSocket';

type AppRoute = '/' | '/login' | '/admin' | '/pt' | '/patient';
type KioskViewMode = 'login' | 'training' | 'rehab' | 'classic';

function resolveRoute(pathname: string): AppRoute {
  if (pathname.startsWith('/admin')) return '/admin';
  if (pathname.startsWith('/pt')) return '/pt';
  if (pathname.startsWith('/patient')) return '/patient';
  if (pathname.startsWith('/login')) return '/login';
  if (pathname.startsWith('/kiosk')) return '/';
  return '/';
}

const KIOSK_LANG_PILLS: readonly { code: ClinicalLang; label: string; ttsLang: 'th' | 'en' | 'zh' }[] = [
  { code: 'th', label: '1. ไทย', ttsLang: 'th' },
  { code: 'en', label: '2. อังกฤษ (EN)', ttsLang: 'en' },
  { code: 'cn', label: '3. จีน (中文)', ttsLang: 'zh' },
];

const KIOSK_TRAINING_COPY: Record<
  ClinicalLang,
  {
    backToLogin: string;
    patientLabel: string;
    smartRehabUiBtn: string;
    telemetryModeLabel: string;
    completedHeaderBadge: string;
    completedDailyLimitBadge: (date: string) => string;
    completedHeading: (name: string) => string;
    completedDescPart1: string;
    completedDescLimit: string;
    completedDescPart2: string;
    completedNextAppt: (date: string) => string;
    doneTodayTag: string;
    restHeaderBadge: string;
    scheduleCheckBadge: (date: string) => string;
    restHeading: (name: string) => string;
    restDesc: (date: string) => string;
    nextApptLabel: (date: string) => string;
    restPracticeBtn: string;
    scheduledHeaderBadge: string;
    scheduledCountBadge: (date: string, count: number) => string;
    scheduledHeading: (name: string) => string;
    scheduledSub: (hn: string, date: string) => string;
    startLeftBadge: string;
    startRightBadge: string;
    startLeftDesc: string;
    startRightDesc: string;
    reshuffleBtn: string;
    holdSecTag: (sec: number) => string;
    setsRepsTag: (sets: number, reps: number) => string;
    doneTag: string;
    finalStepBanner: string;
    step1Badge: string;
    backToOverviewBtn: string;
    skipToMinigameBtn: string;
    minigameHeaderBadge: string;
    backToExercisesBtn: string;
    viewSummaryBtn: string;
    minigameBanner: string;
  }
> = {
  th: {
    backToLogin: '← กลับหน้าสแกนใบหน้า (Kiosk Login)',
    patientLabel: 'ผู้ป่วย:',
    smartRehabUiBtn: 'หน้าวิเคราะห์ชีวกลศาสตร์ (Smart Rehab UI)',
    telemetryModeLabel: 'โหมดวิเคราะห์ชีวกลศาสตร์กายภาพบำบัด',
    completedHeaderBadge: 'Completed Today • ทำกายภาพวันนี้เรียบร้อยแล้ว',
    completedDailyLimitBadge: (date) => `จำกัดการทำกายภาพบำบัด 1 ครั้งต่อ 1 วัน • ประจำวันที่ ${date}`,
    completedHeading: (name) => `คุณได้ทำการกายภาพของวันนี้ไปแล้ว ( ${name} )`,
    completedDescPart1: 'ระบบอนุญาตให้ทำกายภาพบำบัดได้',
    completedDescLimit: '1 ครั้ง ต่อ 1 วันเท่านั้น',
    completedDescPart2: 'และได้บันทึกผลการกายภาพของคุณในวันนี้เรียบร้อยแล้ว',
    completedNextAppt: (date) =>
      `กรุณากลับมาฝึกอีกครั้งในวันนัดหมายถัดไป${date ? ` (${date})` : ''}`,
    doneTodayTag: 'ทำครบแล้ววันนี้ ✓',
    restHeaderBadge: 'Rest Day • วันพักผ่อน',
    scheduleCheckBadge: (date) => `ตรวจสอบตารางนัดหมายวันนี้ (${date})`,
    restHeading: (name) => `วันนี้คือวันพักผ่อนของคุณ ( ${name} )`,
    restDesc: (date) =>
      `ระบบตรวจสอบตารางนัดหมายแล้ว วันนี้ (${date}) ไม่มีรายการนัดทำกายภาพบำบัดของคุณ`,
    nextApptLabel: (date) => `นัดถัดไป: ${date || '-'}`,
    restPracticeBtn: 'ฝึกทบทวนเบาๆ ด้วยตัวเอง (ไม่บังคับ)',
    scheduledHeaderBadge: 'Scheduled Today • มีนัดกายภาพวันนี้',
    scheduledCountBadge: (date, count) =>
      `ตรวจสอบตารางนัดหมายวันนี้ (${date}) • มีนัด ${count} รายการ`,
    scheduledHeading: (name) => `รายการกายภาพบำบัดวันนี้ของคุณ ( ${name} )`,
    scheduledSub: (hn, date) => `HN: ${hn} • แผนฝึกกายภาพบำบัดประจำวันที่ ${date}`,
    startLeftBadge: 'เริ่มการกายภาพ: ให้ยก มือซ้าย (Left Hand)',
    startRightBadge: 'เริ่มการกายภาพ: ให้ยก มือขวา (Right Hand)',
    startLeftDesc: 'กรุณาชู มือซ้าย ค้างไว้หน้ากล้องด้านล่างเพื่อเข้าสู่กายภาพอัตโนมัติ',
    startRightDesc: 'กรุณาชู มือขวา ค้างไว้หน้ากล้องด้านล่างเพื่อเข้าสู่กายภาพอัตโนมัติ',
    reshuffleBtn: 'สุ่มใหม่',
    holdSecTag: (sec) => `ค้าง ${sec} วินาที`,
    setsRepsTag: (sets, reps) => `${sets} เซต × ${reps} ครั้ง`,
    doneTag: 'ทำครบแล้ว',
    finalStepBanner:
      '🎮 ขั้นตอนสุดท้าย: มินิเกมฝึกสมอง (จะเปิดอัตโนมัติหลังจากทำท่ากายภาพบำบัดครบทุกท่าแล้ว)',
    step1Badge: 'ขั้นตอนที่ 1/2: ฝึกท่ากายภาพบำบัดตามกำหนด',
    backToOverviewBtn: '← กลับหน้ารายการกายภาพวันนี้',
    skipToMinigameBtn: 'ข้ามไปมินิเกม (ขั้นตอนสุดท้าย) ➔',
    minigameHeaderBadge: 'AI Kiosk Training • ขั้นตอนสุดท้าย (มินิเกม)',
    backToExercisesBtn: '← กลับหน้าท่ากายภาพบำบัด',
    viewSummaryBtn: '← ดูสรุปผลการกายภาพวันนี้',
    minigameBanner:
      '🎉 ยอดเยี่ยมมาก! คุณทำท่ากายภาพบำบัดครบตามกำหนดแล้ว — ขั้นตอนสุดท้าย: มินิเกมฝึกสมองและการ์ดกายภาพ',
  },
  en: {
    backToLogin: '← Back to Face Scan (Kiosk Login)',
    patientLabel: 'Patient:',
    smartRehabUiBtn: 'Biomechanics Telemetry (Smart Rehab UI)',
    telemetryModeLabel: 'Biomechanics Physical Therapy Mode',
    completedHeaderBadge: 'Completed Today • Session Finished',
    completedDailyLimitBadge: (date) =>
      `Limited to 1 physical therapy session per day • Date: ${date}`,
    completedHeading: (name) => `You have already completed today's physical therapy ( ${name} )`,
    completedDescPart1: 'The system permits',
    completedDescLimit: 'only 1 session per day',
    completedDescPart2: 'and your physical therapy results for today have been saved.',
    completedNextAppt: (date) =>
      `Please return for your next scheduled appointment${date ? ` (${date})` : ''}`,
    doneTodayTag: 'Completed Today ✓',
    restHeaderBadge: 'Rest Day • Recovery Day',
    scheduleCheckBadge: (date) => `Checked Today's Schedule (${date})`,
    restHeading: (name) => `Today is your rest day ( ${name} )`,
    restDesc: (date) =>
      `Schedule verified: there are no physical therapy sessions scheduled for today (${date}).`,
    nextApptLabel: (date) => `Next Appointment: ${date || '-'}`,
    restPracticeBtn: 'Light Self-Practice Review (Optional)',
    scheduledHeaderBadge: 'Scheduled Today • Appointment Active',
    scheduledCountBadge: (date, count) =>
      `Today's Schedule (${date}) • ${count} Prescribed`,
    scheduledHeading: (name) => `Your Physical Therapy Program Today ( ${name} )`,
    scheduledSub: (hn, date) => `HN: ${hn} • Physical Therapy Plan for ${date}`,
    startLeftBadge: 'Start Rehab: Raise LEFT HAND',
    startRightBadge: 'Start Rehab: Raise RIGHT HAND',
    startLeftDesc: 'Hold your LEFT HAND up in front of the camera below to start automatically.',
    startRightDesc: 'Hold your RIGHT HAND up in front of the camera below to start automatically.',
    reshuffleBtn: 'Reshuffle',
    holdSecTag: (sec) => `Hold ${sec}s`,
    setsRepsTag: (sets, reps) => `${sets} sets × ${reps} reps`,
    doneTag: 'Completed',
    finalStepBanner:
      '🎮 Final Step: Cognitive Memory Mini-Game (opens automatically after completing all exercises)',
    step1Badge: 'Step 1/2: Prescribed Physical Therapy',
    backToOverviewBtn: "← Back to Today's Program",
    skipToMinigameBtn: 'Skip to Mini-Game (Final Step) ➔',
    minigameHeaderBadge: 'AI Kiosk Training • Final Step (Mini-Game)',
    backToExercisesBtn: '← Back to Exercises',
    viewSummaryBtn: "← View Today's Summary",
    minigameBanner:
      '🎉 Great job! All prescribed exercises are complete — Final step: Cognitive Rehab Card Mini-Game.',
  },
  cn: {
    backToLogin: '← 返回人脸识别首页 (Kiosk Login)',
    patientLabel: '患者:',
    smartRehabUiBtn: '生物力学分析界面 (Smart Rehab UI)',
    telemetryModeLabel: '生物力学康复训练模式',
    completedHeaderBadge: '今日已完成 • 康复训练已完成',
    completedDailyLimitBadge: (date) => `每日限做 1 次物理康复训练 • 日期: ${date}`,
    completedHeading: (name) => `您今天已完成物理康复训练 ( ${name} )`,
    completedDescPart1: '系统规定每位患者每日仅可进行',
    completedDescLimit: '1 次物理康复训练',
    completedDescPart2: '，且您今日的康复训练记录已保存完毕。',
    completedNextAppt: (date) => `请于下次预约日再来继续训练${date ? ` (${date})` : ''}`,
    doneTodayTag: '今日已完成 ✓',
    restHeaderBadge: '休息日 • 肌肉恢复日',
    scheduleCheckBadge: (date) => `已核对今日预约安排 (${date})`,
    restHeading: (name) => `今天是您的休息日 ( ${name} )`,
    restDesc: (date) => `系统已核对预约表：今日 (${date}) 暂无物理康复训练安排。`,
    nextApptLabel: (date) => `下次预约: ${date || '-'}`,
    restPracticeBtn: '自主轻量复习训练（非强制）',
    scheduledHeaderBadge: '今日预约 • 有康复训练安排',
    scheduledCountBadge: (date, count) => `核对今日日程 (${date}) • 共 ${count} 项训练`,
    scheduledHeading: (name) => `您今日的物理康复训练计划 ( ${name} )`,
    scheduledSub: (hn, date) => `病历号 HN: ${hn} • ${date} 康复训练计划`,
    startLeftBadge: '开始训练：请举起 左手 (Left Hand)',
    startRightBadge: '开始训练：请举起 右手 (Right Hand)',
    startLeftDesc: '请在下方镜头前举起左手并保持，以自动开始康复训练。',
    startRightDesc: '请在下方镜头前举起右手并保持，以自动开始康复训练。',
    reshuffleBtn: '重新随机',
    holdSecTag: (sec) => `保持 ${sec} 秒`,
    setsRepsTag: (sets, reps) => `${sets} 组 × ${reps} 次`,
    doneTag: '已完成',
    finalStepBanner: '🎮 最后一步：认知记忆小游戏（完成全部康复动作后将自动开启）',
    step1Badge: '步骤 1/2：按处方进行物理康复训练',
    backToOverviewBtn: '← 返回今日康复计划',
    skipToMinigameBtn: '跳至小游戏（最后一步）➔',
    minigameHeaderBadge: 'AI Kiosk 训练 • 最后一步（小游戏）',
    backToExercisesBtn: '← 返回康复动作训练',
    viewSummaryBtn: '← 查看今日康复总结',
    minigameBanner: '🎉 太棒了！您已完成所有规定康复动作 — 最后一步：认知康复卡片小游戏',
  },
};

export function App() {
  const { lang, setLang } = useClinicalI18n();
  const kc = KIOSK_TRAINING_COPY[lang];

  const handleSelectKioskLang = (nextLang: ClinicalLang, spokenText?: string) => {
    setLang(nextLang);
    if (typeof window !== 'undefined') {
      try {
        window.localStorage.setItem(
          'drtechcares_voice_coach_lang_v1',
          nextLang === 'cn' ? 'zh' : nextLang,
        );
      } catch {
        // Ignore storage error
      }
      if (spokenText && typeof Audio !== 'undefined') {
        try {
          const opt = KIOSK_LANG_PILLS.find((p) => p.code === nextLang) ?? KIOSK_LANG_PILLS[0];
          const audio = new Audio(buildTtsProxyUrl(spokenText, opt.ttsLang));
          void audio.play().catch(() => {});
        } catch {
          // Ignore audio playback errors
        }
      }
    }
  };

  const renderKioskLangSwitcher = (spokenTextForLang?: (next: ClinicalLang) => string) => (
    <div className="inline-flex items-center gap-1 rounded-xl border border-white/30 bg-white/15 p-0.5">
      {KIOSK_LANG_PILLS.map((pill) => {
        const active = lang === pill.code;
        return (
          <button
            key={pill.code}
            type="button"
            onClick={() => handleSelectKioskLang(pill.code, spokenTextForLang?.(pill.code))}
            className={`rounded-lg px-2 py-1 text-[11px] font-extrabold transition cursor-pointer ${
              active ? 'bg-white text-[#0F4C81] shadow-sm' : 'text-white/90 hover:bg-white/20'
            }`}
          >
            {pill.label}
          </button>
        );
      })}
    </div>
  );
  const [route, setRoute] = useState<AppRoute>(() =>
    typeof window !== 'undefined' ? resolveRoute(window.location.pathname) : '/',
  );
  const [kioskMode, setKioskMode] = useState<KioskViewMode>('login');
  const [staffModalIntent, setStaffModalIntent] = useState<StaffAuthIntent | null>(null);
  const [pendingAutoOpenEnroll, setPendingAutoOpenEnroll] = useState(false);
  const [authenticatedPatient, setAuthenticatedPatient] = useState<KioskLoginSuccessPayload | null>(null);
  const [allowRestDayPractice, setAllowRestDayPractice] = useState(false);
  const [trainingStage, setTrainingStage] = useState<'overview' | 'rehab_exercises' | 'minigame'>('overview');
  const [randomHandSide, setRandomHandSide] = useState<HandChallengeSide>(() => pickRandomHandSide());
  const [completedTodaySession, setCompletedTodaySession] = useState(false);
  const [completedTodayExerciseIds, setCompletedTodayExerciseIds] = useState<string[]>([]);

  useEffect(() => {
    setAllowRestDayPractice(false);
    setTrainingStage('overview');
    setRandomHandSide(pickRandomHandSide());
    setCompletedTodaySession(false);
    setCompletedTodayExerciseIds([]);
  }, [authenticatedPatient?.patient.id]);

  const apiClient = useMemo(
    () =>
      new DrTechCaresApiClient({
        supabaseUrl: import.meta.env.VITE_SUPABASE_URL,
        supabaseAnonKey: import.meta.env.VITE_SUPABASE_ANON_KEY,
      }),
    [],
  );

  const [currentUser, setCurrentUser] = useState<UserAccount | null>(() => apiClient.getStoredUser());
  const [authChecked, setAuthChecked] = useState(false);

  const navigate = (next: AppRoute) => {
    setRoute(next);
    if (typeof window !== 'undefined' && window.location.pathname !== next) {
      window.history.pushState({}, '', next);
    }
  };

  useEffect(() => {
    const onPop = () => setRoute(resolveRoute(window.location.pathname));
    const onAuthExpired = () => {
      setCurrentUser(null);
      setAuthChecked(true);
    };
    window.addEventListener('popstate', onPop);
    window.addEventListener('drtech:auth-expired', onAuthExpired);
    return () => {
      window.removeEventListener('popstate', onPop);
      window.removeEventListener('drtech:auth-expired', onAuthExpired);
    };
  }, []);

  useEffect(() => {
    if (route === '/') {
      setAuthChecked(true);
      return;
    }
    let cancelled = false;
    apiClient.getMe().then((u) => {
      if (!cancelled) {
        setCurrentUser(u);
        setAuthChecked(true);
      }
    });
    return () => {
      cancelled = true;
    };
  }, [route, apiClient]);

  // Locked Kiosk Mode (/) — Exact Kiosk Biometric Login UI by default, transitioning into Rehab / Classic Kiosk
  if (route === '/') {
    if (kioskMode === 'classic') {
      return (
        <div className="min-h-screen bg-gradient-to-br from-[#EAF8F5] via-[#EFF6FF] to-[#F8FAFC] p-4">
          <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
            <button
              type="button"
              onClick={() => {
                setAuthenticatedPatient(null);
                setKioskMode('login');
              }}
              className="rounded-xl border border-emerald-300 bg-white px-4 py-1.5 text-xs font-bold text-[#0F4C81] shadow-xs hover:bg-emerald-50 cursor-pointer"
            >
              {kc.backToLogin}
            </button>
            <button
              type="button"
              onClick={() => setKioskMode('rehab')}
              className="rounded-xl border border-sky-300 bg-white px-4 py-1.5 text-xs font-bold text-[#0284C7] shadow-xs hover:bg-sky-50 cursor-pointer"
            >
              {kc.smartRehabUiBtn}
            </button>
          </div>
          <KioskPortalApp apiClient={apiClient} />
        </div>
      );
    }

    if (kioskMode === 'training' && authenticatedPatient) {
      const todaySchedule = resolveTodayPatientSchedule({
        rehabSchedules: authenticatedPatient.rehabSchedules,
        todayRoutine: authenticatedPatient.todayRoutine,
        nextAppointmentDate: authenticatedPatient.patient.nextAppointmentDate,
        serverTodayExercises: authenticatedPatient.todayExercises,
      });
      const hasAppointmentToday =
        authenticatedPatient.hasAppointmentToday ?? todaySchedule.hasAppointmentToday;
      const todayExercises =
        authenticatedPatient.todayExercises && authenticatedPatient.todayExercises.length > 0
          ? authenticatedPatient.todayExercises
          : todaySchedule.todayExercises;
      const hasCompletedRehabToday = Boolean(
        completedTodaySession ||
          authenticatedPatient.completedToday ||
          todaySchedule.completedToday ||
          isPatientCompletedTodayInStorage(authenticatedPatient.patient.id, todaySchedule.todayIso) ||
          isPatientCompletedTodayInStorage(authenticatedPatient.patient.hn, todaySchedule.todayIso),
      );

      const prescribedOptions = mapTodayExercisesToPrescriptions(todayExercises);

      const markDailyRehabDone = () => {
        markPatientCompletedTodayInStorage(
          [authenticatedPatient.patient.id, authenticatedPatient.patient.hn],
          todaySchedule.todayIso,
        );
        setCompletedTodaySession(true);
        setAuthenticatedPatient((prev) => {
          if (!prev) return prev;
          return {
            ...prev,
            completedToday: true,
            rehabSchedules: (prev.rehabSchedules ?? []).map((item) =>
              item.scheduledDate === todaySchedule.todayIso
                ? { ...item, status: 'COMPLETED' as const }
                : item,
            ),
            todayExercises: (prev.todayExercises ?? []).map((ex) => ({
              ...ex,
              status: 'COMPLETED' as const,
            })),
          };
        });
      };

      const handleExitToLogin = () => {
        setAllowRestDayPractice(false);
        setTrainingStage('overview');
        setCompletedTodaySession(false);
        setCompletedTodayExerciseIds([]);
        setAuthenticatedPatient(null);
        setKioskMode('login');
      };

      const handleExerciseCompleted = async (ex: ExercisePrescriptionOption) => {
        const nextIds = completedTodayExerciseIds.includes(ex.exercise_id)
          ? completedTodayExerciseIds
          : [...completedTodayExerciseIds, ex.exercise_id];
        setCompletedTodayExerciseIds(nextIds);
        const allDoneNow =
          prescribedOptions.length <= 1 || nextIds.length >= prescribedOptions.length;
        if (allDoneNow) {
          markDailyRehabDone();
        }
        try {
          await apiClient.recordRehabSession({
            patientId: authenticatedPatient.patient.id,
            activityType: 'physio_exercise',
            exerciseId: ex.catalog_exercise_id ?? ex.exercise_id,
            exerciseName: ex.name_th,
            completedSets: ex.target_sets ?? 1,
            completedReps: ex.target_reps,
            accuracyPercent: 100,
            maxRomDeg: ex.target_rom_deg,
            painStopTriggered: false,
            aiFeedback: `Completed prescribed ${ex.name_th} (${ex.target_reps} reps)`,
            kinematicsReplay: [],
            scheduledDate: todaySchedule.todayIso,
            markDailyCompleted: allDoneNow,
          });
        } catch {
          // Keep Kiosk flow responsive even if offline
        }
      };

      const saveTrainingSession = async (summary: PhysicalCardGameSummary) => {
        markDailyRehabDone();
        const accuracy = Math.round((summary.correct / Math.max(1, summary.total)) * 100);
        try {
          await apiClient.recordRehabSession({
            patientId: authenticatedPatient.patient.id,
            activityType: 'minigame',
            exerciseId: 'cognitive-card-game',
            exerciseName: 'มินิเกมฝึกสมอง (การ์ดกายภาพ)',
            completedSets: 1,
            completedReps: summary.total,
            accuracyPercent: accuracy,
            maxRomDeg: 0,
            painStopTriggered: false,
            aiFeedback: `Cognitive Accuracy ${accuracy}% (${summary.correct}/${summary.total})`,
            kinematicsReplay: [],
            minigameScore: summary,
            scheduledDate: todaySchedule.todayIso,
            markDailyCompleted: true,
          });
        } catch {
          // Allow patient summary screen to remain responsive even if offline
        }
      };

      if (hasCompletedRehabToday && trainingStage === 'overview') {
        return (
          <div className="min-h-screen w-full bg-gradient-to-br from-[#EAF8F5] via-[#EFF6FF] to-[#F8FAFC] flex flex-col">
            <div className="bg-gradient-to-r from-[#0F4C81] via-[#0284C7] to-[#0D9488] px-4 py-2.5 text-white flex flex-wrap items-center justify-between gap-2 text-xs shadow-sm border-b border-white/20">
              <div className="flex items-center gap-2 font-bold">
                <span className="px-2.5 py-0.5 rounded-full bg-emerald-400/25 border border-emerald-200/40 text-white">
                  {kc.completedHeaderBadge}
                </span>
                <span>
                  {kc.patientLabel} {authenticatedPatient.patient.fullName} (HN: {authenticatedPatient.patient.hn})
                </span>
              </div>
              <div className="flex flex-wrap items-center gap-2">
                {renderKioskLangSwitcher((next) =>
                  KIOSK_TRAINING_COPY[next].completedHeading(authenticatedPatient.patient.fullName),
                )}
                <button
                  type="button"
                  onClick={handleExitToLogin}
                  className="rounded-xl border border-white/30 bg-white/15 px-3 py-1.5 font-bold text-white hover:bg-white/25 cursor-pointer"
                >
                  {kc.backToLogin}
                </button>
              </div>
            </div>
            <div className="flex-1 flex items-center justify-center p-4 sm:p-8">
              <div className="w-full max-w-2xl rounded-3xl bg-white border-2 border-emerald-400 shadow-2xl p-6 sm:p-10 text-center">
                <div className="mx-auto w-20 h-20 rounded-full bg-emerald-100 text-emerald-600 flex items-center justify-center shadow-inner mb-5">
                  <CheckCircle2 className="h-11 w-11" />
                </div>
                <span className="inline-block px-3.5 py-1 rounded-full bg-sky-50 border border-sky-200 text-[#0F4C81] text-xs font-extrabold mb-3">
                  {kc.completedDailyLimitBadge(todaySchedule.todayIso)}
                </span>
                <h1 className="text-2xl sm:text-3xl font-black text-[#0F4C81]">
                  {kc.completedHeading(authenticatedPatient.patient.fullName)}
                </h1>
                <p className="text-sm sm:text-base text-slate-600 mt-3 leading-relaxed">
                  {kc.completedDescPart1}{' '}
                  <span className="font-extrabold text-[#0284C7]">{kc.completedDescLimit}</span>{' '}
                  {kc.completedDescPart2}
                  <br />
                  <span className="font-semibold text-emerald-700">
                    {kc.completedNextAppt(authenticatedPatient.patient.nextAppointmentDate)}
                  </span>
                </p>

                {todayExercises.length > 0 && (
                  <div className="mt-6 space-y-2.5 text-left max-h-60 overflow-y-auto pr-1">
                    {todayExercises.map((item, idx) => (
                      <div
                        key={item.id}
                        className="rounded-2xl border border-emerald-200 bg-emerald-50/70 px-4 py-3 flex items-center justify-between gap-3"
                      >
                        <div className="flex items-center gap-3 min-w-0">
                          <span className="shrink-0 w-7 h-7 rounded-full bg-emerald-600 text-white text-xs font-black flex items-center justify-center">
                            {idx + 1}
                          </span>
                          <div className="min-w-0">
                            <p className="text-sm font-extrabold text-slate-900 truncate">
                              {lang === 'en' ? item.nameEn : item.nameTh}
                            </p>
                            <p className="text-[11px] text-slate-500 truncate">
                              {lang === 'en' ? item.nameTh : item.nameEn}
                            </p>
                          </div>
                        </div>
                        <span className="shrink-0 rounded-xl bg-white border border-emerald-300 px-3 py-1 text-xs font-extrabold text-emerald-700">
                          {kc.doneTodayTag}
                        </span>
                      </div>
                    ))}
                  </div>
                )}

                <div className="mt-7">
                  <button
                    type="button"
                    onClick={handleExitToLogin}
                    className="w-full py-4 px-6 rounded-2xl bg-gradient-to-r from-[#0F4C81] via-[#0284C7] to-[#10B981] hover:opacity-95 text-white font-black text-sm sm:text-base shadow-lg cursor-pointer transition"
                  >
                    {kc.backToLogin}
                  </button>
                </div>
              </div>
            </div>
          </div>
        );
      }

      if (!hasAppointmentToday && !allowRestDayPractice) {
        return (
          <div className="min-h-screen w-full bg-gradient-to-br from-[#EAF8F5] via-[#EFF6FF] to-[#F8FAFC] flex flex-col">
            <div className="bg-gradient-to-r from-[#0F4C81] via-[#0284C7] to-[#0D9488] px-4 py-2.5 text-white flex flex-wrap items-center justify-between gap-2 text-xs shadow-sm border-b border-white/20">
              <div className="flex items-center gap-2 font-bold">
                <span className="px-2.5 py-0.5 rounded-full bg-white/20 text-white border border-white/25">
                  {kc.restHeaderBadge}
                </span>
                <span>
                  {kc.patientLabel} {authenticatedPatient.patient.fullName} (HN: {authenticatedPatient.patient.hn})
                </span>
              </div>
              <div className="flex flex-wrap items-center gap-2">
                {renderKioskLangSwitcher((next) =>
                  KIOSK_TRAINING_COPY[next].restHeading(authenticatedPatient.patient.fullName),
                )}
                <button
                  type="button"
                  onClick={handleExitToLogin}
                  className="rounded-xl border border-white/30 bg-white/15 px-3 py-1.5 font-bold text-white hover:bg-white/25 cursor-pointer"
                >
                  {kc.backToLogin}
                </button>
              </div>
            </div>
            <div className="flex-1 flex items-center justify-center p-4 sm:p-8">
              <div className="w-full max-w-xl rounded-3xl bg-white border-2 border-amber-300 shadow-2xl p-6 sm:p-8 text-center">
                <div className="mx-auto w-16 h-16 rounded-full bg-amber-100 text-amber-700 flex items-center justify-center text-3xl mb-4">
                  🌿
                </div>
                <span className="inline-block px-3 py-1 rounded-full bg-amber-50 border border-amber-200 text-amber-800 text-xs font-extrabold mb-3">
                  {kc.scheduleCheckBadge(todaySchedule.todayIso)}
                </span>
                <h1 className="text-2xl sm:text-3xl font-extrabold text-slate-900">
                  {kc.restHeading(authenticatedPatient.patient.fullName)}
                </h1>
                <p className="text-sm text-slate-600 mt-3 leading-relaxed">
                  {kc.restDesc(todaySchedule.todayIso)}
                  <br />
                  <span className="font-semibold text-emerald-700">
                    {kc.nextApptLabel(authenticatedPatient.patient.nextAppointmentDate)}
                  </span>
                </p>
                <div className="mt-6 flex flex-col gap-3">
                  <button
                    type="button"
                    onClick={handleExitToLogin}
                    className="w-full py-3.5 px-5 rounded-2xl bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white font-extrabold text-sm shadow-lg cursor-pointer"
                  >
                    {kc.backToLogin}
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setAllowRestDayPractice(true);
                      setTrainingStage('rehab_exercises');
                    }}
                    className="w-full py-2.5 px-4 rounded-xl bg-slate-100 hover:bg-slate-200 border border-slate-300 text-slate-700 text-xs font-bold cursor-pointer"
                  >
                    {kc.restPracticeBtn}
                  </button>
                </div>
              </div>
            </div>
          </div>
        );
      }

      if (hasAppointmentToday && trainingStage === 'overview') {
        return (
          <div className="min-h-screen w-full bg-gradient-to-br from-[#EAF8F5] via-[#EFF6FF] to-[#F8FAFC] flex flex-col">
            <div className="bg-gradient-to-r from-[#0F4C81] via-[#0284C7] to-[#0D9488] px-4 py-2.5 text-white flex flex-wrap items-center justify-between gap-2 text-xs shadow-sm border-b border-white/20">
              <div className="flex items-center gap-2 font-bold">
                <span className="px-2.5 py-0.5 rounded-full bg-emerald-400/25 border border-emerald-200/40 text-white">
                  {kc.scheduledHeaderBadge}
                </span>
                <span>
                  {kc.patientLabel} {authenticatedPatient.patient.fullName} (HN: {authenticatedPatient.patient.hn})
                </span>
              </div>
              <div className="flex flex-wrap items-center gap-2">
                {renderKioskLangSwitcher((next) =>
                  KIOSK_TRAINING_COPY[next].scheduledHeading(authenticatedPatient.patient.fullName),
                )}
                <button
                  type="button"
                  onClick={handleExitToLogin}
                  className="rounded-xl border border-white/30 bg-white/15 px-3 py-1.5 font-bold text-white hover:bg-white/25 cursor-pointer"
                >
                  {kc.backToLogin}
                </button>
              </div>
            </div>
            <div className="flex-1 flex items-center justify-center p-4 sm:p-8">
              <div className="w-full max-w-3xl rounded-3xl bg-white border-2 border-emerald-400 shadow-2xl p-6 sm:p-8 text-center">
                <div className="mx-auto w-16 h-16 rounded-full bg-emerald-100 text-emerald-700 flex items-center justify-center text-3xl mb-4">
                  📅
                </div>
                <span className="inline-block px-3 py-1 rounded-full bg-emerald-50 border border-emerald-200 text-emerald-800 text-xs font-extrabold mb-3">
                  {kc.scheduledCountBadge(todaySchedule.todayIso, todayExercises.length)}
                </span>
                <h1 className="text-2xl sm:text-3xl font-extrabold text-slate-900">
                  {kc.scheduledHeading(authenticatedPatient.patient.fullName)}
                </h1>
                <p className="text-xs text-slate-500 mt-1.5 font-mono">
                  {kc.scheduledSub(authenticatedPatient.patient.hn, todaySchedule.todayIso)}
                </p>

                <div
                  className={`mt-4 rounded-2xl border-2 px-4 py-3.5 shadow-sm flex flex-wrap items-center justify-between gap-3 text-left transition-colors ${
                    randomHandSide === 'left'
                      ? 'border-orange-400 bg-orange-50 text-orange-950'
                      : 'border-emerald-400 bg-emerald-50 text-emerald-950'
                  }`}
                >
                  <div className="space-y-1.5">
                    <span
                      className={`inline-flex items-center gap-2 rounded-xl px-3.5 py-1.5 text-sm sm:text-base font-extrabold text-white shadow ${
                        randomHandSide === 'left' ? 'bg-orange-500' : 'bg-emerald-600'
                      }`}
                    >
                      <span>{randomHandSide === 'left' ? '🫲' : '🫱'}</span>
                      <span>
                        {randomHandSide === 'left' ? kc.startLeftBadge : kc.startRightBadge}
                      </span>
                    </span>
                    <p
                      className={`text-xs font-bold ${
                        randomHandSide === 'left' ? 'text-orange-900' : 'text-emerald-900'
                      }`}
                    >
                      {randomHandSide === 'left' ? kc.startLeftDesc : kc.startRightDesc}
                    </p>
                  </div>

                  <button
                    type="button"
                    onClick={() => setRandomHandSide((prev) => (prev === 'left' ? 'right' : 'left'))}
                    className={`inline-flex items-center gap-1.5 rounded-xl border bg-white px-3 py-1.5 text-xs font-bold shadow-sm cursor-pointer transition-colors ${
                      randomHandSide === 'left'
                        ? 'border-orange-300 text-orange-800 hover:bg-orange-100'
                        : 'border-emerald-300 text-emerald-800 hover:bg-emerald-100'
                    }`}
                  >
                    <Shuffle className="h-3.5 w-3.5" />
                    {kc.reshuffleBtn}
                  </button>
                </div>

                <div className="mt-5 space-y-2.5 text-left max-h-72 overflow-y-auto pr-1">
                  {todayExercises.map((item, idx) => (
                    <div
                      key={item.id}
                      className="rounded-2xl border border-emerald-200 bg-emerald-50/70 px-4 py-3 flex items-center justify-between gap-3"
                    >
                      <div className="flex items-start gap-3 min-w-0">
                        <span className="shrink-0 w-7 h-7 rounded-full bg-emerald-600 text-white text-xs font-black flex items-center justify-center mt-0.5">
                          {idx + 1}
                        </span>
                        <div className="min-w-0">
                          <p className="text-sm font-extrabold text-slate-900">
                            {lang === 'en' ? item.nameEn : item.nameTh}
                          </p>
                          <p className="text-[11px] text-slate-500">
                            {lang === 'en' ? item.nameTh : item.nameEn}
                          </p>
                          <div className="flex flex-wrap items-center gap-1.5 mt-1">
                            <span className="inline-block px-2 py-0.5 rounded-md bg-teal-50 text-teal-800 border border-teal-200 text-[11px] font-semibold">
                              {item.targetJoint}
                            </span>
                            <span className="inline-block px-2 py-0.5 rounded-md bg-white text-slate-600 border border-slate-200 text-[11px] font-semibold">
                              {kc.holdSecTag(item.holdSec)}
                            </span>
                            {item.status === 'COMPLETED' && (
                              <span className="inline-block px-2 py-0.5 rounded-md bg-sky-50 text-sky-700 border border-sky-200 text-[11px] font-bold">
                                {kc.doneTag}
                              </span>
                            )}
                          </div>
                        </div>
                      </div>
                      <span className="shrink-0 rounded-xl bg-white border border-emerald-300 px-3 py-1.5 text-xs font-extrabold text-emerald-700 shadow-sm">
                        {kc.setsRepsTag(item.sets, item.reps)}
                      </span>
                    </div>
                  ))}
                </div>

                <div className="mt-4 rounded-xl border border-sky-200 bg-sky-50/80 px-3.5 py-2 text-xs font-bold text-sky-900">
                  {kc.finalStepBanner}
                </div>

                <RandomHandStartPrompt
                  targetSide={randomHandSide}
                  onConfirmStart={() => setTrainingStage('rehab_exercises')}
                />

                <div className="mt-4 flex flex-col gap-2">
                  <button
                    type="button"
                    onClick={handleExitToLogin}
                    className="w-full py-2.5 px-4 rounded-xl bg-slate-100 hover:bg-slate-200 border border-slate-300 text-slate-700 text-xs font-bold cursor-pointer"
                  >
                    {kc.backToLogin}
                  </button>
                </div>
              </div>
            </div>
          </div>
        );
      }

      if (trainingStage === 'rehab_exercises') {
        return (
          <div className="min-h-screen w-full bg-gradient-to-br from-[#EAF8F5] via-[#EFF6FF] to-[#F8FAFC] flex flex-col">
            <div className="bg-gradient-to-r from-[#0F4C81] via-[#0284C7] to-[#0D9488] px-4 py-2.5 text-white flex flex-wrap items-center justify-between gap-2 text-xs shadow-sm border-b border-white/20">
              <div className="flex flex-wrap items-center gap-2 font-bold">
                <span className="px-2.5 py-0.5 rounded-full bg-white/20 border border-white/25 text-white">
                  AI Kiosk Session
                </span>
                <span>
                  {kc.patientLabel} {authenticatedPatient.patient.fullName} (HN: {authenticatedPatient.patient.hn})
                </span>
                <span className="px-2.5 py-0.5 rounded-full bg-emerald-400/25 border border-emerald-200/35 text-white">
                  {kc.step1Badge}
                </span>
              </div>
              <div className="flex flex-wrap items-center gap-2">
                {renderKioskLangSwitcher()}
                {hasAppointmentToday && (
                  <button
                    type="button"
                    onClick={() => setTrainingStage('overview')}
                    className="rounded-xl border border-white/30 bg-white/15 px-3 py-1.5 font-bold text-white hover:bg-white/25 cursor-pointer"
                  >
                    {kc.backToOverviewBtn}
                  </button>
                )}
                <button
                  type="button"
                  onClick={() => setTrainingStage('minigame')}
                  className="rounded-xl border border-white/30 bg-white/15 px-3 py-1.5 font-bold text-white hover:bg-white/25 cursor-pointer"
                >
                  {kc.skipToMinigameBtn}
                </button>
                <button
                  type="button"
                  onClick={handleExitToLogin}
                  className="rounded-xl border border-emerald-200/40 bg-emerald-500/30 px-3 py-1.5 font-bold text-white hover:bg-emerald-500/45 cursor-pointer"
                >
                  {kc.backToLogin}
                </button>
              </div>
            </div>
            <SmartRehabKioskUI
              patient={{
                id: authenticatedPatient.patient.hn,
                name: authenticatedPatient.patient.fullName,
              }}
              prescribedExercises={prescribedOptions}
              onExerciseCompleted={handleExerciseCompleted}
              onAllExercisesCompleted={() => {
                markDailyRehabDone();
                setTrainingStage('minigame');
              }}
              onEmergencyStop={({ reason }) => {
                void apiClient
                  .createStaffCall(
                    'KIOSK-OPD-01',
                    `${authenticatedPatient.patient.fullName} (${authenticatedPatient.patient.hn}) • ${reason}`,
                    authenticatedPatient.patient.id,
                  )
                  .catch(() => {});
              }}
              onOpenStaffPortal={() => navigate('/login')}
              onOpenLegacyKiosk={() => setKioskMode('classic')}
            />
          </div>
        );
      }

      return (
        <div className="min-h-screen w-full bg-gradient-to-br from-[#EAF8F5] via-[#EFF6FF] to-[#F8FAFC] flex flex-col">
          <div className="bg-gradient-to-r from-[#0F4C81] via-[#0284C7] to-[#0D9488] px-4 py-2.5 text-white flex flex-wrap items-center justify-between gap-2 text-xs shadow-sm border-b border-white/20">
            <div className="flex items-center gap-2 font-bold">
              <span className="px-2.5 py-0.5 rounded-full bg-white/20 border border-white/25 text-white">
                {kc.minigameHeaderBadge}
              </span>
              <span>
                {kc.patientLabel} {authenticatedPatient.patient.fullName} (HN: {authenticatedPatient.patient.hn})
              </span>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              {renderKioskLangSwitcher()}
              {!hasCompletedRehabToday ? (
                <button
                  type="button"
                  onClick={() => setTrainingStage('rehab_exercises')}
                  className="rounded-xl border border-white/30 bg-white/15 px-3 py-1.5 font-bold text-white hover:bg-white/25 cursor-pointer"
                >
                  {kc.backToExercisesBtn}
                </button>
              ) : (
                <button
                  type="button"
                  onClick={() => setTrainingStage('overview')}
                  className="rounded-xl border border-white/30 bg-white/15 px-3 py-1.5 font-bold text-white hover:bg-white/25 cursor-pointer"
                >
                  {kc.viewSummaryBtn}
                </button>
              )}
              <button
                type="button"
                onClick={handleExitToLogin}
                className="rounded-xl border border-emerald-200/40 bg-emerald-500/30 px-3 py-1.5 font-bold text-white hover:bg-emerald-500/45 cursor-pointer"
              >
                {kc.backToLogin}
              </button>
            </div>
          </div>
          <div className="flex-1 flex flex-col gap-4 p-3 sm:p-5">
            <div className="rounded-2xl border border-emerald-300 bg-gradient-to-r from-emerald-50 via-teal-50 to-sky-50 px-4 py-3 text-[#0F4C81] text-xs sm:text-sm font-bold flex flex-wrap items-center justify-between gap-2 shadow-sm">
              <span>{kc.minigameBanner}</span>
            </div>
            <PhysicalCardGame
              patientName={`${authenticatedPatient.patient.fullName} (${authenticatedPatient.patient.hn})`}
              allowReplay={false}
              onSessionFinished={saveTrainingSession}
              onComplete={() => {
                markDailyRehabDone();
                setTrainingStage('overview');
              }}
            />
          </div>
        </div>
      );
    }

    if (kioskMode === 'rehab') {
      return (
        <div className="relative min-h-screen bg-gradient-to-br from-[#EAF8F5] via-[#EFF6FF] to-[#F8FAFC]">
          <div className="bg-gradient-to-r from-[#0F4C81] via-[#0284C7] to-[#0D9488] px-4 py-2 text-white flex flex-wrap items-center justify-between gap-2 text-xs shadow-sm">
            <div className="flex items-center gap-2 font-bold">
              <span className="px-2 py-0.5 rounded-full bg-white/20 text-white">AI Kiosk Session</span>
              {authenticatedPatient ? (
                <span>
                  {kc.patientLabel} {authenticatedPatient.patient.fullName} (HN: {authenticatedPatient.patient.hn})
                </span>
              ) : (
                <span>{kc.telemetryModeLabel}</span>
              )}
            </div>
            <div className="flex flex-wrap items-center gap-2">
              {renderKioskLangSwitcher()}
              <button
                type="button"
                onClick={() => {
                  setAuthenticatedPatient(null);
                  setKioskMode('login');
                }}
                className="rounded-xl border border-emerald-400/50 bg-emerald-600/25 px-3 py-1 font-bold text-emerald-100 hover:bg-emerald-600/40 cursor-pointer"
              >
                {kc.backToLogin}
              </button>
            </div>
          </div>
          <SmartRehabKioskUI
            onEmergencyStop={({ reason }) => {
              const label = authenticatedPatient
                ? `${authenticatedPatient.patient.fullName} (${authenticatedPatient.patient.hn})`
                : 'โหมดฝึกกายภาพหน้าตู้ Kiosk';
              void apiClient
                .createStaffCall(
                  'KIOSK-OPD-01',
                  `${label} • ${reason}`,
                  authenticatedPatient?.patient.id ?? null,
                )
                .catch(() => {});
            }}
            onOpenStaffPortal={() => navigate('/login')}
            onOpenLegacyKiosk={() => setKioskMode('classic')}
          />
        </div>
      );
    }

    return (
      <>
        <KioskBiometricLoginView
          apiClient={apiClient}
          kioskId="kiosk-01"
          staffModalOpen={staffModalIntent !== null}
          onPatientAuthenticated={(payload) => {
            setAuthenticatedPatient(payload);
            setKioskMode('training');
          }}
          onOpenStaffLogin={() => setStaffModalIntent('general')}
          onOpenStaffRegister={() => {
            setStaffModalIntent('kiosk_patient_registration');
          }}
          onOpenTelemetrySimulator={() => setKioskMode('rehab')}
        />
        {staffModalIntent !== null && (
          <StaffLoginModalOrPage
            apiClient={apiClient}
            asModal
            authIntent={staffModalIntent}
            onReturnToKiosk={() => setStaffModalIntent(null)}
            onSuccess={(user) => {
              const intent = staffModalIntent;
              setStaffModalIntent(null);
              setCurrentUser(user);
              if (intent === 'kiosk_patient_registration' && (user.role === 'pt' || user.role === 'super_admin')) {
                setPendingAutoOpenEnroll(true);
                navigate(user.role === 'super_admin' ? '/admin' : '/pt');
              } else if (user.role === 'super_admin') {
                navigate('/admin');
              } else if (user.role === 'pt') {
                navigate('/pt');
              } else {
                navigate('/patient');
              }
            }}
          />
        )}
      </>
    );
  }

  if (!authChecked) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-slate-100 text-sm font-bold text-slate-600">
        กำลังตรวจสอบสิทธิ์การเข้าใช้งาน...
      </div>
    );
  }

  // Require login for /login, /admin, /pt, /patient
  if (!currentUser || route === '/login') {
    return (
      <StaffLoginModalOrPage
        apiClient={apiClient}
        onReturnToKiosk={() => navigate('/')}
        onSuccess={(user) => {
          setCurrentUser(user);
          if (user.role === 'super_admin') navigate('/admin');
          else if (user.role === 'pt') navigate('/pt');
          else navigate('/patient');
        }}
      />
    );
  }

  const handleLogout = async () => {
    await apiClient.logout();
    setCurrentUser(null);
    navigate('/login');
  };

  // Strict Role Routing Guard
  if (route === '/admin') {
    if (currentUser.role !== 'super_admin') {
      return (
        <div className="min-h-screen flex flex-col items-center justify-center bg-rose-50 p-6 text-center">
          <h1 className="text-3xl font-black text-rose-900">403 Forbidden</h1>
          <p className="mt-2 text-sm text-rose-700">เฉพาะผู้อำนวยการโรงพยาบาล (Super Admin) เท่านั้นที่เข้าถึงหน้านี้ได้</p>
          <button
            onClick={() => navigate(currentUser.role === 'pt' ? '/pt' : '/patient')}
            className="mt-4 rounded-xl bg-gradient-to-r from-[#0F4C81] to-[#0D9488] px-4 py-2 text-xs font-bold text-white shadow-sm"
          >
            กลับพอร์ทัลของคุณ
          </button>
        </div>
      );
    }
    return (
      <AdminPortalApp
        apiClient={apiClient}
        currentUser={currentUser}
        onLogout={handleLogout}
        onReturnToKiosk={() => navigate('/')}
        autoOpenEnroll={pendingAutoOpenEnroll}
        onAutoOpenEnrollConsumed={() => setPendingAutoOpenEnroll(false)}
      />
    );
  }

  if (route === '/pt') {
    if (currentUser.role !== 'pt' && currentUser.role !== 'super_admin') {
      return (
        <div className="min-h-screen flex flex-col items-center justify-center bg-rose-50 p-6 text-center">
          <h1 className="text-3xl font-black text-rose-900">403 Forbidden</h1>
          <p className="mt-2 text-sm text-rose-700">ไม่มีสิทธิ์เข้าถึงพอร์ทัลนักกายภาพบำบัด</p>
        </div>
      );
    }
    return (
      <PtPortalApp
        apiClient={apiClient}
        currentUser={currentUser}
        onLogout={handleLogout}
        onReturnToKiosk={() => navigate('/')}
        autoOpenEnroll={pendingAutoOpenEnroll}
        onAutoOpenEnrollConsumed={() => setPendingAutoOpenEnroll(false)}
      />
    );
  }

  return (
    <div className="min-h-screen bg-slate-100">
      <PatientPersonalPortal apiClient={apiClient} currentUser={currentUser} onLogout={handleLogout} />
    </div>
  );
}
export default App;
