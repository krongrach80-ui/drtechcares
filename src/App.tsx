import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { AdminPortalApp } from '@apps/admin';
import { KioskPortalApp, PhysicalCardGame, type PhysicalCardGameSummary } from '@apps/kiosk';
import { PtPortalApp } from '@apps/pt';
import {
  DrTechCaresApiClient,
  clearPatientCompletedTodayInStorage,
  isPatientCompletedTodayInStorage,
  markPatientCompletedTodayInStorage,
  resolveNextAppointmentDate,
  resolvePrescribedExercisesFromSchedules,
  resolveTodayPatientSchedule,
  type ManagedExerciseRecord,
  type ModpackRecord,
  type UserAccount,
} from '@drtechcares/api-client';
import { useClinicalI18n, type ClinicalLang } from '@drtechcares/clinical-i18n';
import {
  buildTtsProxyUrl,
  pickSpeechVoice,
  readStoredKioskVoiceMuted,
  subscribeKioskVoiceMuted,
  writeStoredKioskVoiceLang,
} from './components/VoiceCoachBanner';
import { PatientPersonalPortal } from './PatientPersonalPortal';
import { StaffLoginModalOrPage, type StaffAuthIntent } from './StaffLoginModalOrPage';
import {
  KioskBiometricLoginView,
  type KioskLoginSuccessPayload,
} from './components/KioskBiometricLoginView';
import { StaffAssistedEnrollmentModal } from './components/StaffAssistedEnrollmentModal';
import { ArrowRight, Calendar, CheckCircle2, Gamepad2, Hand, Leaf, RotateCcw, Shuffle, Volume2 } from 'lucide-react';
import {
  RandomHandStartPrompt,
  pickRandomHandSide,
  type HandChallengeSide,
} from './components/RandomHandStartPrompt';
import { SmartRehabKioskUI } from './components/SmartRehabKioskUI';
import {
  POSE_TEST_PATIENT,
  buildAll15ClinicalExercisePrescriptions,
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

const KIOSK_LANG_PILLS: readonly {
  code: ClinicalLang;
  label: string;
  ttsLang: 'th' | 'en' | 'zh';
  bcp47: string;
}[] = [
  { code: 'th', label: '1. ไทย', ttsLang: 'th', bcp47: 'th-TH' },
  { code: 'en', label: '2. อังกฤษ (EN)', ttsLang: 'en', bcp47: 'en-US' },
  { code: 'cn', label: '3. จีน (中文)', ttsLang: 'zh', bcp47: 'zh-CN' },
];

export type KioskOverviewStatus = 'completed_today' | 'rest_day' | 'scheduled_today';

export function getKioskOverviewVoiceNarration(
  lang: ClinicalLang,
  params: {
    status: KioskOverviewStatus;
    fullName: string;
    handSide?: HandChallengeSide;
  },
): string {
  const name = params.fullName.trim();
  const side = params.handSide ?? 'right';
  if (params.status === 'completed_today') {
    if (lang === 'en') {
      return `You have already completed today's physical therapy, ${name}. Only 1 session per day is allowed, and today's results are saved. See you next visit.`;
    }
    if (lang === 'cn') {
      return `您今天已完成物理康复训练，${name}。每日限做 1 次物理康复训练，今日记录已保存完毕，请于下次预约日再来训练。`;
    }
    return `คุณได้ทำการกายภาพของวันนี้ไปแล้ว คุณ ${name} ระบบอนุญาตให้ทำ 1 ครั้งต่อ 1 วันเท่านั้น และบันทึกผลวันนี้เรียบร้อยแล้วครับ`;
  }
  if (params.status === 'rest_day') {
    if (lang === 'en') {
      return `Today is your rest day, ${name}. There are no physical therapy sessions scheduled for today. Please return on your next appointment.`;
    }
    if (lang === 'cn') {
      return `今天是您的休息日，${name}。今日暂无物理康复训练安排，请于下次预约日再来训练。`;
    }
    return `วันนี้คือวันพักผ่อนของคุณ ${name} ไม่มีรายการนัดทำกายภาพบำบัดวันนี้ กรุณากลับมาฝึกในวันนัดถัดไปครับ`;
  }
  const handTh = side === 'left' ? 'มือซ้าย' : 'มือขวา';
  const handEn = side === 'left' ? 'left hand' : 'right hand';
  const handZh = side === 'left' ? '左手' : '右手';
  if (lang === 'en') {
    return `Your physical therapy program today, ${name}. Please hold your ${handEn} up in front of the camera to start automatically.`;
  }
  if (lang === 'cn') {
    return `您今日的物理康复训练计划，${name}。请在镜头前举起${handZh}并保持，以自动开始康复训练。`;
  }
  return `รายการกายภาพบำบัดวันนี้ของคุณ ${name} กรุณาชู${handTh}ค้างไว้หน้ากล้องเพื่อเริ่มฝึกอัตโนมัติครับ`;
}

const KIOSK_TRAINING_COPY: Record<
  ClinicalLang,
  {
    backToLogin: string;
    replayVoiceBtn: string;
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
    replayVoiceBtn: 'ฟังเสียง',
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
      date
        ? `กรุณากลับมาฝึกอีกครั้งในวันนัดหมายถัดไป (${date})`
        : 'กรุณากลับมาฝึกอีกครั้งในวันนัดหมายถัดไปตามตารางนัดหมาย',
    doneTodayTag: 'ทำครบแล้ววันนี้',
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
      'ขั้นตอนสุดท้าย: มินิเกมฝึกสมอง (จะเปิดอัตโนมัติหลังจากทำท่ากายภาพบำบัดครบทุกท่าแล้ว)',
    step1Badge: 'ขั้นตอนที่ 1/2: ฝึกท่ากายภาพบำบัดตามกำหนด',
    backToOverviewBtn: '← กลับหน้ารายการกายภาพวันนี้',
    skipToMinigameBtn: 'ข้ามไปมินิเกม (ขั้นตอนสุดท้าย) →',
    minigameHeaderBadge: 'AI Kiosk Training • ขั้นตอนสุดท้าย (มินิเกม)',
    backToExercisesBtn: '← กลับหน้าท่ากายภาพบำบัด',
    viewSummaryBtn: '← ดูสรุปผลการกายภาพวันนี้',
    minigameBanner:
      'ยอดเยี่ยมมาก! คุณทำท่ากายภาพบำบัดครบตามกำหนดแล้ว — ขั้นตอนสุดท้าย: มินิเกมฝึกสมองและการ์ดกายภาพ',
  },
  en: {
    backToLogin: '← Back to Face Scan (Kiosk Login)',
    replayVoiceBtn: 'Voice',
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
      date
        ? `Please return on your next scheduled appointment (${date})`
        : 'Please return on your next scheduled appointment',
    doneTodayTag: 'Completed Today',
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
      'Final Step: Cognitive Memory Mini-Game (opens automatically after completing all exercises)',
    step1Badge: 'Step 1/2: Prescribed Physical Therapy',
    backToOverviewBtn: "← Back to Today's Program",
    skipToMinigameBtn: 'Skip to Mini-Game (Final Step) →',
    minigameHeaderBadge: 'AI Kiosk Training • Final Step (Mini-Game)',
    backToExercisesBtn: '← Back to Exercises',
    viewSummaryBtn: "← View Today's Summary",
    minigameBanner:
      'Great job! All prescribed exercises are complete — Final step: Cognitive Rehab Card Mini-Game.',
  },
  cn: {
    backToLogin: '← 返回人脸识别首页 (Kiosk Login)',
    replayVoiceBtn: '语音播报',
    patientLabel: '患者:',
    smartRehabUiBtn: '生物力学分析界面 (Smart Rehab UI)',
    telemetryModeLabel: '生物力学康复训练模式',
    completedHeaderBadge: '今日已完成 • 康复训练已完成',
    completedDailyLimitBadge: (date) => `每日限做 1 次物理康复训练 • 日期: ${date}`,
    completedHeading: (name) => `您今天已完成物理康复训练 ( ${name} )`,
    completedDescPart1: '系统规定每位患者每日仅可进行',
    completedDescLimit: '1 次物理康复训练',
    completedDescPart2: '，且您今日的康复训练记录已保存完毕。',
    completedNextAppt: (date) =>
      date ? `请于下次预约日再来继续训练 (${date})` : '请于下次预约日再来继续训练',
    doneTodayTag: '今日已完成',
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
    finalStepBanner: '最后一步：认知记忆小游戏（完成全部康复动作后将自动开启）',
    step1Badge: '步骤 1/2：按处方进行物理康复训练',
    backToOverviewBtn: '← 返回今日康复计划',
    skipToMinigameBtn: '跳至小游戏（最后一步）→',
    minigameHeaderBadge: 'AI Kiosk 训练 • 最后一步（小游戏）',
    backToExercisesBtn: '← 返回康复动作训练',
    viewSummaryBtn: '← 查看今日康复总结',
    minigameBanner: '太棒了！您已完成所有规定康复动作 — 最后一步：认知康复卡片小游戏',
  },
};

export function getRetestPostureButtonLabel(lang: ClinicalLang): string {
  return lang === 'th'
    ? 'ทดสอบท่าทางกายภาพอีกครั้ง'
    : lang === 'cn'
      ? '再次测试康复动作'
      : 'Retest Rehab Posture';
}

export function shouldShowCompletedTodayScreen(params: {
  hasCompletedRehabToday: boolean;
  bypassDailyLimit: boolean;
  trainingStage: 'overview' | 'rehab_exercises' | 'minigame';
}): boolean {
  return params.hasCompletedRehabToday && !params.bypassDailyLimit && params.trainingStage === 'overview';
}

export function App() {
  const { lang, setLang } = useClinicalI18n();
  const kc = KIOSK_TRAINING_COPY[lang];

  const overviewAudioRef = useRef<HTMLAudioElement | null>(null);
  const overviewSpeakSeqRef = useRef(0);
  const overviewWebSpeechTimerRef = useRef<number | null>(null);
  const overviewEntryTimerRef = useRef<number | null>(null);
  const lastSpokenOverviewKeyRef = useRef<string | null>(null);
  const overviewPendingAudioRef = useRef<{ text: string; lang: ClinicalLang } | null>(null);

  const stopKioskOverviewSpeech = useCallback(() => {
    overviewSpeakSeqRef.current += 1;
    overviewPendingAudioRef.current = null;
    if (typeof window !== 'undefined') {
      if (overviewEntryTimerRef.current !== null) {
        window.clearTimeout(overviewEntryTimerRef.current);
        overviewEntryTimerRef.current = null;
      }
      if (overviewWebSpeechTimerRef.current !== null) {
        window.clearTimeout(overviewWebSpeechTimerRef.current);
        overviewWebSpeechTimerRef.current = null;
      }
      if ('speechSynthesis' in window) {
        try {
          window.speechSynthesis.cancel();
        } catch {
          // Ignore synthesis cancel errors
        }
      }
    }
    const audio = overviewAudioRef.current;
    if (audio) {
      try {
        audio.onplay = null;
        audio.onended = null;
        audio.onerror = null;
        audio.pause();
        audio.currentTime = 0;
      } catch {
        // Ignore audio cleanup errors
      }
    }
  }, []);

  const speakWithWebSpeechFallback = useCallback(
    (text: string, targetLang: ClinicalLang, seq: number) => {
      if (typeof window === 'undefined' || overviewSpeakSeqRef.current !== seq) return;
      if (!('speechSynthesis' in window) || typeof SpeechSynthesisUtterance === 'undefined') return;
      const opt = KIOSK_LANG_PILLS.find((p) => p.code === targetLang) ?? KIOSK_LANG_PILLS[0];
      try {
        const synth = window.speechSynthesis;
        const wasBusy = synth.speaking || synth.pending;
        if (wasBusy) {
          synth.cancel();
        }
        const startUtterance = () => {
          if (overviewSpeakSeqRef.current !== seq) return;
          const voices = synth.getVoices();
          const matchedVoice = pickSpeechVoice(voices, opt.bcp47);
          const utterance = new SpeechSynthesisUtterance(text);
          utterance.lang = matchedVoice?.lang || opt.bcp47;
          if (matchedVoice) utterance.voice = matchedVoice;
          utterance.rate = targetLang === 'th' ? 0.96 : targetLang === 'cn' ? 0.98 : 1.0;
          synth.speak(utterance);
        };
        if (wasBusy) {
          overviewWebSpeechTimerRef.current = window.setTimeout(() => {
            overviewWebSpeechTimerRef.current = null;
            startUtterance();
          }, 60);
        } else {
          startUtterance();
        }
      } catch {
        // Ignore browser speech restriction
      }
    },
    [],
  );

  useEffect(() => {
    return subscribeKioskVoiceMuted((muted) => {
      if (muted) {
        stopKioskOverviewSpeech();
      }
    });
  }, [stopKioskOverviewSpeech]);

  const speakKioskOverview = useCallback(
    (spokenText: string, targetLang: ClinicalLang) => {
      const clean = spokenText.trim();
      if (!clean || typeof window === 'undefined' || readStoredKioskVoiceMuted()) return;
      stopKioskOverviewSpeech();
      const seq = overviewSpeakSeqRef.current;
      lastSpokenOverviewKeyRef.current = `${targetLang}:${clean}`;
      const opt = KIOSK_LANG_PILLS.find((p) => p.code === targetLang) ?? KIOSK_LANG_PILLS[0];

      if (typeof Audio !== 'undefined') {
        try {
          let audio = overviewAudioRef.current;
          if (!audio) {
            audio = new Audio();
            audio.preload = 'auto';
            overviewAudioRef.current = audio;
          }
          let fallbackTriggered = false;
          const triggerFallbackOnce = () => {
            if (fallbackTriggered || overviewSpeakSeqRef.current !== seq) return;
            fallbackTriggered = true;
            speakWithWebSpeechFallback(clean, targetLang, seq);
          };
          audio.playbackRate = targetLang === 'th' ? 1.0 : targetLang === 'cn' ? 0.98 : 1.0;
          audio.onplay = () => {
            if (overviewSpeakSeqRef.current === seq) {
              overviewPendingAudioRef.current = null;
            }
          };
          audio.onerror = triggerFallbackOnce;
          audio.src = buildTtsProxyUrl(clean, opt.ttsLang);
          const playPromise = audio.play();
          if (playPromise && typeof playPromise.catch === 'function') {
            playPromise.catch((err: unknown) => {
              if (overviewSpeakSeqRef.current !== seq) return;
              const errName =
                err && typeof err === 'object' && 'name' in err
                  ? String((err as { name: string }).name)
                  : '';
              if (errName === 'AbortError') return;
              if (errName === 'NotAllowedError') {
                overviewPendingAudioRef.current = { text: clean, lang: targetLang };
              }
              triggerFallbackOnce();
            });
          }
          return;
        } catch {
          // Fall through to Web Speech API
        }
      }
      speakWithWebSpeechFallback(clean, targetLang, seq);
    },
    [stopKioskOverviewSpeech, speakWithWebSpeechFallback],
  );

  const handleSelectKioskLang = (nextLang: ClinicalLang, spokenText?: string) => {
    setLang(nextLang);
    writeStoredKioskVoiceLang(nextLang === 'cn' ? 'zh' : nextLang);
    if (spokenText) {
      speakKioskOverview(spokenText, nextLang);
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
      {spokenTextForLang && (
        <button
          type="button"
          onClick={() => speakKioskOverview(spokenTextForLang(lang), lang)}
          title={kc.replayVoiceBtn}
          aria-label={kc.replayVoiceBtn}
          className="inline-flex items-center gap-1 rounded-lg px-2 py-1 text-[11px] font-extrabold text-white/95 hover:bg-white/25 cursor-pointer"
        >
          <Volume2 className="h-3.5 w-3.5" />
          <span className="hidden sm:inline">{kc.replayVoiceBtn}</span>
        </button>
      )}
    </div>
  );
  const [route, setRoute] = useState<AppRoute>(() =>
    typeof window !== 'undefined' ? resolveRoute(window.location.pathname) : '/',
  );
  const [kioskMode, setKioskMode] = useState<KioskViewMode>('login');
  const [isPoseTestMode, setIsPoseTestMode] = useState(false);
  const [poseTestInitialExerciseCode, setPoseTestInitialExerciseCode] = useState<string | null>(null);
  const [poseTestReturnRoute, setPoseTestReturnRoute] = useState<'/admin' | '/pt'>('/admin');
  const [staffModalIntent, setStaffModalIntent] = useState<StaffAuthIntent | null>(null);
  const [pendingAutoOpenEnroll, setPendingAutoOpenEnroll] = useState(false);
  const [authenticatedPatient, setAuthenticatedPatient] = useState<KioskLoginSuccessPayload | null>(null);
  const [allowRestDayPractice, setAllowRestDayPractice] = useState(false);
  const [bypassDailyLimit, setBypassDailyLimit] = useState(false);
  const [trainingStage, setTrainingStage] = useState<'overview' | 'rehab_exercises' | 'minigame'>('overview');
  const [randomHandSide, setRandomHandSide] = useState<HandChallengeSide>(() => pickRandomHandSide());
  const [completedTodaySession, setCompletedTodaySession] = useState(false);
  const [completedTodayExerciseIds, setCompletedTodayExerciseIds] = useState<string[]>([]);

  const rehabPatientProp = useMemo(() => {
    if (!authenticatedPatient) return undefined;
    const p = authenticatedPatient.patient;
    return {
      id: p.hn,
      name: p.fullName,
      max_safe_rom_deg: p.maxSafeRomDeg,
      affected_side: p.affectedSide,
      stroke_type: p.strokeType,
      motor_assessment_score: p.motorAssessmentScore,
    };
  }, [authenticatedPatient]);

  useEffect(() => {
    setAllowRestDayPractice(false);
    setBypassDailyLimit(false);
    setTrainingStage('overview');
    setRandomHandSide(pickRandomHandSide());
    setCompletedTodaySession(false);
    setCompletedTodayExerciseIds([]);
  }, [authenticatedPatient?.patient.id]);

  const overviewScheduleState = useMemo(() => {
    if (route !== '/' || !authenticatedPatient) {
      return null;
    }
    const todaySchedule = resolveTodayPatientSchedule({
      rehabSchedules: authenticatedPatient.rehabSchedules,
      todayRoutine: authenticatedPatient.todayRoutine,
      nextAppointmentDate: authenticatedPatient.patient.nextAppointmentDate,
      fallbackNextAppointmentDate: authenticatedPatient.patient.nextAppointmentDate,
      serverTodayExercises: authenticatedPatient.todayExercises,
      completedTodayOverride: Boolean(
        !bypassDailyLimit && (completedTodaySession || authenticatedPatient.completedToday),
      ),
    });
    const hasAppointmentToday =
      authenticatedPatient.hasAppointmentToday ?? todaySchedule.hasAppointmentToday;
    const todayExercises =
      todaySchedule.todayExercises.length > 0
        ? todaySchedule.todayExercises
        : (authenticatedPatient.todayExercises ?? []);
    const prescribedFromSchedules = resolvePrescribedExercisesFromSchedules(
      authenticatedPatient.rehabSchedules,
    );
    const effectiveKioskExercises =
      todayExercises.length > 0 ? todayExercises : prescribedFromSchedules;
    const hasCompletedRehabToday = Boolean(
      !bypassDailyLimit &&
        (completedTodaySession ||
          authenticatedPatient.completedToday ||
          todaySchedule.completedToday ||
          isPatientCompletedTodayInStorage(authenticatedPatient.patient.id, todaySchedule.todayIso) ||
          isPatientCompletedTodayInStorage(authenticatedPatient.patient.hn, todaySchedule.todayIso)),
    );
    return {
      todaySchedule,
      hasAppointmentToday,
      todayExercises,
      effectiveKioskExercises,
      hasCompletedRehabToday,
    };
  }, [route, authenticatedPatient, completedTodaySession, bypassDailyLimit]);

  const activeOverviewStatus: KioskOverviewStatus | null =
    overviewScheduleState && trainingStage === 'overview'
      ? overviewScheduleState.hasCompletedRehabToday && !bypassDailyLimit
        ? 'completed_today'
        : !overviewScheduleState.hasAppointmentToday && !allowRestDayPractice && !bypassDailyLimit
          ? 'rest_day'
          : 'scheduled_today'
      : null;

  const activeOverviewNarration =
    activeOverviewStatus && authenticatedPatient
      ? getKioskOverviewVoiceNarration(lang, {
          status: activeOverviewStatus,
          fullName: authenticatedPatient.patient.fullName,
          handSide: randomHandSide,
        })
      : '';

  const isOverviewVisible = Boolean(activeOverviewNarration);

  useEffect(() => {
    if (!isOverviewVisible) {
      lastSpokenOverviewKeyRef.current = null;
      stopKioskOverviewSpeech();
    }
  }, [isOverviewVisible, stopKioskOverviewSpeech]);

  useEffect(() => {
    return () => {
      stopKioskOverviewSpeech();
    };
  }, [stopKioskOverviewSpeech]);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    const unlockOverviewAudio = () => {
      const pending = overviewPendingAudioRef.current;
      if (pending) {
        overviewPendingAudioRef.current = null;
        speakKioskOverview(pending.text, pending.lang);
      }
    };
    window.addEventListener('pointerdown', unlockOverviewAudio, { passive: true });
    window.addEventListener('keydown', unlockOverviewAudio, { passive: true });
    return () => {
      window.removeEventListener('pointerdown', unlockOverviewAudio);
      window.removeEventListener('keydown', unlockOverviewAudio);
    };
  }, [speakKioskOverview]);

  useEffect(() => {
    if (!activeOverviewNarration || typeof window === 'undefined') {
      return;
    }
    const key = `${lang}:${activeOverviewNarration}`;
    if (lastSpokenOverviewKeyRef.current === key) {
      return;
    }
    if (overviewEntryTimerRef.current !== null) {
      window.clearTimeout(overviewEntryTimerRef.current);
    }
    overviewEntryTimerRef.current = window.setTimeout(() => {
      overviewEntryTimerRef.current = null;
      if (lastSpokenOverviewKeyRef.current !== key) {
        speakKioskOverview(activeOverviewNarration, lang);
      }
    }, 140);
    return () => {
      if (overviewEntryTimerRef.current !== null) {
        window.clearTimeout(overviewEntryTimerRef.current);
        overviewEntryTimerRef.current = null;
      }
    };
  }, [activeOverviewNarration, lang, speakKioskOverview]);

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
  const [kioskEnrollModalOpen, setKioskEnrollModalOpen] = useState(false);
  const [kioskPtList, setKioskPtList] = useState<UserAccount[]>([]);
  const [kioskExercises, setKioskExercises] = useState<ManagedExerciseRecord[]>([]);
  const [kioskModpacks, setKioskModpacks] = useState<ModpackRecord[]>([]);

  useEffect(() => {
    if ((!kioskEnrollModalOpen || !currentUser) && !isPoseTestMode) return;
    let cancelled = false;
    if (isPoseTestMode && (!kioskEnrollModalOpen || !currentUser)) {
      void apiClient
        .listExercises()
        .then((exercisesList) => {
          if (!cancelled) setKioskExercises(exercisesList);
        })
        .catch(() => {});
      return () => {
        cancelled = true;
      };
    }
    void Promise.all([
      apiClient.listUsers().catch(() => [] as UserAccount[]),
      apiClient.listExercises().catch(() => [] as ManagedExerciseRecord[]),
      apiClient.listModpacks().catch(() => [] as ModpackRecord[]),
    ]).then(([usersList, exercisesList, modpacksList]) => {
      if (cancelled) return;
      const activePts = usersList.filter((u) => u.role === 'pt' && u.active);
      const fallbackPts = usersList.filter((u) => u.role === 'pt');
      setKioskPtList(
        activePts.length > 0
          ? activePts
          : fallbackPts.length > 0
            ? fallbackPts
            : currentUser && currentUser.role === 'pt'
              ? [currentUser]
              : [],
      );
      setKioskExercises(exercisesList);
      setKioskModpacks(modpacksList);
    });
    return () => {
      cancelled = true;
    };
  }, [kioskEnrollModalOpen, isPoseTestMode, currentUser, apiClient]);

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
    const prescribedOptions = overviewScheduleState
      ? mapTodayExercisesToPrescriptions(overviewScheduleState.effectiveKioskExercises)
      : mapTodayExercisesToPrescriptions([]);

    const markDailyRehabDone = () => {
      if (!authenticatedPatient || !overviewScheduleState) return;
      const { todaySchedule } = overviewScheduleState;
      markPatientCompletedTodayInStorage(
        [authenticatedPatient.patient.id, authenticatedPatient.patient.hn],
        todaySchedule.todayIso,
      );
      setCompletedTodaySession(true);
      setAuthenticatedPatient((prev) => {
        if (!prev) return prev;
        const updatedSchedules = (prev.rehabSchedules ?? []).map((item) =>
          item.scheduledDate === todaySchedule.todayIso
            ? { ...item, status: 'COMPLETED' as const }
            : item,
        );
        const nextAppt = resolveNextAppointmentDate({
          rehabSchedules: updatedSchedules,
          fallbackNextAppointmentDate: prev.patient.nextAppointmentDate,
          todayIso: todaySchedule.todayIso,
          completedToday: true,
          afterTodayOnly: true,
        });
        return {
          ...prev,
          completedToday: true,
          patient: {
            ...prev.patient,
            nextAppointmentDate: nextAppt,
          },
          rehabSchedules: updatedSchedules,
          todayExercises: (prev.todayExercises ?? []).map((ex) => ({
            ...ex,
            status: 'COMPLETED' as const,
          })),
        };
      });
    };

    const handleExitToLogin = () => {
      setIsPoseTestMode(false);
      setPoseTestInitialExerciseCode(null);
      setAllowRestDayPractice(false);
      setBypassDailyLimit(false);
      setTrainingStage('overview');
      setCompletedTodaySession(false);
      setCompletedTodayExerciseIds([]);
      setAuthenticatedPatient(null);
      setKioskMode('login');
    };

    const handleRetestPosture = () => {
      if (!authenticatedPatient || !overviewScheduleState) return;
      stopKioskOverviewSpeech();
      setBypassDailyLimit(true);
      setAllowRestDayPractice(true);
      setCompletedTodaySession(false);
      setCompletedTodayExerciseIds([]);
      clearPatientCompletedTodayInStorage(
        [authenticatedPatient.patient.id, authenticatedPatient.patient.hn],
        overviewScheduleState.todaySchedule.todayIso,
      );
      setAuthenticatedPatient((prev) =>
        prev
          ? {
              ...prev,
              completedToday: false,
            }
          : prev,
      );
      setTrainingStage('rehab_exercises');
    };

    const handleExerciseCompleted = async (ex: ExercisePrescriptionOption) => {
      if (!authenticatedPatient || !overviewScheduleState) return;
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
          scheduledDate: overviewScheduleState.todaySchedule.todayIso,
          markDailyCompleted: allDoneNow,
        });
      } catch {
        // Keep Kiosk flow responsive even if offline
      }
    };

    if (isPoseTestMode) {
      const poseTestPrescriptions = buildAll15ClinicalExercisePrescriptions(kioskExercises);
      return (
        <div className="min-h-screen w-full bg-gradient-to-br from-[#EAF8F5] via-[#EFF6FF] to-[#F8FAFC] flex flex-col">
          <SmartRehabKioskUI
            isPoseTestMode
            initialExerciseCode={poseTestInitialExerciseCode}
            patient={POSE_TEST_PATIENT}
            ownerBiometrics={null}
            prescribedExercises={poseTestPrescriptions}
            isStaffAuthorized
            onBackToAdminPortal={() => {
              setIsPoseTestMode(false);
              setPoseTestInitialExerciseCode(null);
              setKioskMode('login');
              navigate(poseTestReturnRoute);
            }}
            onOpenStaffPortal={() => {
              setIsPoseTestMode(false);
              navigate(poseTestReturnRoute);
            }}
            onOpenLegacyKiosk={() => setKioskMode('classic')}
          />
        </div>
      );
    }

    if (kioskMode === 'classic') {
      return (
        <div className="min-h-screen bg-gradient-to-br from-[#EAF8F5] via-[#EFF6FF] to-[#F8FAFC] p-4">
          <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
            <button
              type="button"
              onClick={handleExitToLogin}
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

    if (kioskMode === 'training' && authenticatedPatient && overviewScheduleState) {
      const { todaySchedule, hasAppointmentToday, todayExercises, hasCompletedRehabToday } =
        overviewScheduleState;

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

      if (
        shouldShowCompletedTodayScreen({
          hasCompletedRehabToday,
          bypassDailyLimit,
          trainingStage,
        })
      ) {
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
                  getKioskOverviewVoiceNarration(next, {
                    status: 'completed_today',
                    fullName: authenticatedPatient.patient.fullName,
                  }),
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
              <div className="w-full max-w-2xl rounded-3xl bg-white border-2 border-emerald-400 shadow-2xl p-6 sm:p-10 text-center animate-popup-spring">
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
                    {kc.completedNextAppt(todaySchedule.nextAppointmentAfterToday)}
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

            <button
              type="button"
              onClick={handleRetestPosture}
              data-testid="completed-today-retest-btn"
              className="fixed bottom-4 right-4 z-50 inline-flex items-center gap-1.5 px-3.5 py-2 rounded-xl bg-teal-900/85 hover:bg-teal-800 text-emerald-100 border border-emerald-400/40 shadow-lg backdrop-blur-sm text-xs md:text-sm font-semibold transition-all active:scale-95 whitespace-nowrap cursor-pointer"
              title="ข้ามการจำกัด 1 ครั้ง/วัน เพื่อทดสอบระบบตรวจจับท่าทางกายภาพอีกครั้ง"
            >
              <RotateCcw className="w-4 h-4 shrink-0" />
              <span>{getRetestPostureButtonLabel(lang)}</span>
            </button>
          </div>
        );
      }

      if (!hasAppointmentToday && !allowRestDayPractice && !bypassDailyLimit) {
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
                  getKioskOverviewVoiceNarration(next, {
                    status: 'rest_day',
                    fullName: authenticatedPatient.patient.fullName,
                  }),
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
              <div className="w-full max-w-xl rounded-3xl bg-white border-2 border-amber-300 shadow-2xl p-6 sm:p-8 text-center animate-popup-spring">
                <div className="mx-auto w-16 h-16 rounded-full bg-amber-100 text-amber-700 flex items-center justify-center mb-4">
                  <Leaf className="w-8 h-8" />
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
                    {kc.nextApptLabel(todaySchedule.nextAppointmentAfterToday || todaySchedule.nextAppointmentDate)}
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

      if (trainingStage === 'overview') {
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
                  getKioskOverviewVoiceNarration(next, {
                    status: 'scheduled_today',
                    fullName: authenticatedPatient.patient.fullName,
                    handSide: randomHandSide,
                  }),
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
              <div className="w-full max-w-3xl rounded-3xl bg-white border-2 border-emerald-400 shadow-2xl p-6 sm:p-8 text-center animate-popup-spring">
                <div className="mx-auto w-16 h-16 rounded-full bg-emerald-100 text-emerald-700 flex items-center justify-center mb-4">
                  <Calendar className="w-8 h-8" />
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
                      <Hand className={`w-4 h-4 shrink-0 ${randomHandSide === 'left' ? '-scale-x-100' : ''}`} />
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
                <button
                  type="button"
                  onClick={() => setTrainingStage('overview')}
                  className="rounded-xl border border-white/30 bg-white/15 px-3 py-1.5 font-bold text-white hover:bg-white/25 cursor-pointer"
                >
                  {kc.backToOverviewBtn}
                </button>
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
              patient={rehabPatientProp}
              ownerBiometrics={authenticatedPatient.biometricTemplate ?? null}
              prescribedExercises={prescribedOptions}
              isStaffAuthorized={Boolean(
                currentUser && (currentUser.role === 'super_admin' || currentUser.role === 'pt'),
              )}
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
          <div className="bg-gradient-to-r from-[#0F4C81] via-[#0284C7] to-[#0D9488] px-4 py-1.5 text-white flex flex-wrap items-center justify-between gap-2 text-xs shadow-xs border-b border-white/20">
            <div className="flex items-center gap-2 font-bold">
              <span className="px-2 py-0.5 rounded-full bg-white/20 border border-white/25 text-white">
                {kc.minigameHeaderBadge}
              </span>
              <span>
                {kc.patientLabel} {authenticatedPatient.patient.fullName} (HN: {authenticatedPatient.patient.hn})
              </span>
            </div>
            <div className="flex flex-wrap items-center gap-1.5">
              {renderKioskLangSwitcher()}
              {!hasCompletedRehabToday ? (
                <button
                  type="button"
                  onClick={() => setTrainingStage('rehab_exercises')}
                  className="rounded-lg border border-white/30 bg-white/15 px-2.5 py-1 font-bold text-white hover:bg-white/25 cursor-pointer"
                >
                  {kc.backToExercisesBtn}
                </button>
              ) : (
                <button
                  type="button"
                  onClick={() => setTrainingStage('overview')}
                  className="rounded-lg border border-white/30 bg-white/15 px-2.5 py-1 font-bold text-white hover:bg-white/25 cursor-pointer"
                >
                  {kc.viewSummaryBtn}
                </button>
              )}
              <button
                type="button"
                onClick={handleExitToLogin}
                className="rounded-lg border border-emerald-200/40 bg-emerald-500/30 px-2.5 py-1 font-bold text-white hover:bg-emerald-500/45 cursor-pointer"
              >
                {kc.backToLogin}
              </button>
            </div>
          </div>
          <div className="flex-1 flex flex-col gap-2 p-2 sm:p-3">
            <div className="rounded-xl border border-emerald-200 bg-gradient-to-r from-emerald-50 via-teal-50 to-sky-50 px-3 py-1.5 text-[#0F4C81] text-xs font-semibold flex flex-wrap items-center justify-between gap-2 animate-toast-in">
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
                onClick={handleExitToLogin}
                className="rounded-xl border border-emerald-400/50 bg-emerald-600/25 px-3 py-1 font-bold text-emerald-100 hover:bg-emerald-600/40 cursor-pointer"
              >
                {kc.backToLogin}
              </button>
            </div>
          </div>
          <SmartRehabKioskUI
            patient={
              authenticatedPatient
                ? {
                    id: authenticatedPatient.patient.hn,
                    name: authenticatedPatient.patient.fullName,
                    max_safe_rom_deg: authenticatedPatient.patient.maxSafeRomDeg,
                    affected_side: authenticatedPatient.patient.affectedSide,
                    stroke_type: authenticatedPatient.patient.strokeType,
                    motor_assessment_score: authenticatedPatient.patient.motorAssessmentScore,
                  }
                : undefined
            }
            ownerBiometrics={authenticatedPatient?.biometricTemplate ?? null}
            prescribedExercises={authenticatedPatient ? prescribedOptions : undefined}
            isStaffAuthorized={Boolean(
              currentUser && (currentUser.role === 'super_admin' || currentUser.role === 'pt'),
            )}
            onExerciseCompleted={
              authenticatedPatient && overviewScheduleState ? handleExerciseCompleted : undefined
            }
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
          staffModalOpen={staffModalIntent !== null || kioskEnrollModalOpen}
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
                setKioskEnrollModalOpen(true);
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
        {kioskEnrollModalOpen && currentUser && (
          <StaffAssistedEnrollmentModal
            open={kioskEnrollModalOpen}
            apiClient={apiClient}
            currentUser={currentUser}
            ptList={kioskPtList}
            exercises={kioskExercises}
            modpacks={kioskModpacks}
            onClose={() => setKioskEnrollModalOpen(false)}
            onEnrolled={() => {}}
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
        onReturnToKiosk={() => {
          setIsPoseTestMode(false);
          navigate('/');
        }}
        onOpenPoseTestMode={(exerciseCode) => {
          setPoseTestReturnRoute('/admin');
          setPoseTestInitialExerciseCode(exerciseCode ?? null);
          setAuthenticatedPatient(null);
          setIsPoseTestMode(true);
          setKioskMode('rehab');
          navigate('/');
        }}
        autoOpenEnroll={pendingAutoOpenEnroll}
        onAutoOpenEnrollConsumed={() => setPendingAutoOpenEnroll(false)}
      />
    );
  }

  if (route === '/pt') {
    if (currentUser.role !== 'pt' && currentUser.role !== 'super_admin') {
      return (
        <div className="min-h-screen flex flex-col items-center justify-center bg-rose-50 p-6 text-center gap-4">
          <h1 className="text-3xl font-black text-rose-900">403 Forbidden</h1>
          <p className="text-sm text-rose-700">ไม่มีสิทธิ์เข้าถึงพอร์ทัลนักกายภาพบำบัด</p>
          <div className="flex flex-wrap items-center justify-center gap-2">
            <button
              type="button"
              onClick={() => navigate('/')}
              className="rounded-xl bg-[#0F4C81] px-4 py-2 text-xs font-bold text-white hover:bg-[#0284C7] cursor-pointer"
            >
              กลับหน้าตู้ Kiosk
            </button>
            <button
              type="button"
              onClick={handleLogout}
              className="rounded-xl border border-rose-300 bg-white px-4 py-2 text-xs font-bold text-rose-800 hover:bg-rose-100 cursor-pointer"
            >
              ออกจากระบบ
            </button>
          </div>
        </div>
      );
    }
    return (
      <PtPortalApp
        apiClient={apiClient}
        currentUser={currentUser}
        onLogout={handleLogout}
        onReturnToKiosk={() => {
          setIsPoseTestMode(false);
          navigate('/');
        }}
        onOpenPoseTestMode={(exerciseCode) => {
          setPoseTestReturnRoute('/pt');
          setPoseTestInitialExerciseCode(exerciseCode ?? null);
          setAuthenticatedPatient(null);
          setIsPoseTestMode(true);
          setKioskMode('rehab');
          navigate('/');
        }}
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
