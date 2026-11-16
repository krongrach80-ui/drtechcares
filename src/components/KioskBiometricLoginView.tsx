import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  FaceLandmarker,
  FilesetResolver,
  HandLandmarker,
  PoseLandmarker,
  type NormalizedLandmark,
} from '@mediapipe/tasks-vision';
import {
  isPatientCompletedTodayInStorage,
  resolveTodayPatientSchedule,
  type DailyScheduleDay,
  type DrTechCaresApiClient,
  type KioskFaceVerifyResponse,
  type PatientBiometricIndexEntry,
  type PatientRecord,
  type RehabScheduleItem,
  type StrokeType,
  type AffectedSide,
  type TodayPrescribedExerciseItem,
} from '@drtechcares/api-client';
import {
  TemporalBiometricMatcher,
  computeBoundingBoxFromLandmarks,
  subscribeBiometricIndex,
} from '../lib/faceMatcher';
import { useClinicalI18n, type ClinicalLang } from '@drtechcares/clinical-i18n';
import {
  buildTtsProxyUrl,
  pickSpeechVoice,
  type VoiceCoachLanguage,
} from './VoiceCoachBanner';
import {
  alignFaceTo112,
  BiologicalLivenessTracker,
  captureDownscaledContextFrame,
  computeEyeRollAngleDeg,
  computeHeadZoneRoiFromFaceBox,
  computeHeadZoneRoiFromPose,
  evaluateLivenessAndPose,
  extractDualFlipFaceEmbedding,
  inspectBiometricFrame3D,
  remapRoiLandmarksToFrame,
  renderUpscaledRoi256,
  selectClinicalCameraDeviceId,
  selectPrimaryFace,
  SingleScanLoginController,
  type BiologicalLivenessReport,
  type BiometricVerificationJson,
  type LivenessAndPoseReport,
} from '@drtechcares/face-client';
import { FacePresenceTracker } from '@drtechcares/kiosk-input';
import { POSE_INDEX } from '@drtechcares/pose-engine';
import {
  BURST_FRAME_COUNT,
  purgeLegacyBiometricCache,
} from '@drtechcares/smart-biometric-core';
import {
  useEdgeVisionSocket,
  type EdgeKioskTelemetryPayload,
} from '../hooks/useEdgeVisionSocket';
import {
  AlertCircle,
  Bell,
  CameraOff,
  CheckCircle2,
  FlipHorizontal2,
  HeartPulse,
  Info,
  KeyRound,
  RefreshCw,
  RotateCw,
  ShieldCheck,
  Sparkles,
  UserPlus,
  X,
} from 'lucide-react';

export interface KioskLoginSuccessPayload {
  patient: {
    id: string;
    hn: string;
    fullName: string;
    nextAppointmentDate: string;
    strokeType?: StrokeType;
    affectedSide?: AffectedSide;
    motorAssessmentScore?: number;
    adlDressingScore?: number;
    maxSafeRomDeg?: number;
    universalHealthLinkCode?: string;
  };
  todayRoutine: DailyScheduleDay | null;
  confidencePercent: number;
  rehabSchedules?: RehabScheduleItem[];
  todayExercises?: TodayPrescribedExerciseItem[];
  hasAppointmentToday?: boolean;
  completedToday?: boolean;
}

function buildKioskLoginSuccessPayload(params: {
  patient: {
    id: string;
    hn: string;
    fullName: string;
    nextAppointmentDate: string;
    strokeType?: StrokeType;
    affectedSide?: AffectedSide;
    motorAssessmentScore?: number;
    adlDressingScore?: number;
    maxSafeRomDeg?: number;
    universalHealthLinkCode?: string;
  };
  confidencePercent: number;
  todayRoutine?: DailyScheduleDay | null;
  rehabSchedules?: RehabScheduleItem[];
  serverTodayExercises?: TodayPrescribedExerciseItem[];
  serverCompletedToday?: boolean;
  indexEntry?: PatientBiometricIndexEntry | null;
}): KioskLoginSuccessPayload {
  const resolvedSchedules = params.rehabSchedules ?? params.indexEntry?.rehabSchedules;
  const resolvedRoutine = params.todayRoutine ?? params.indexEntry?.todayRoutine ?? null;
  const serverExercises = params.serverTodayExercises ?? params.indexEntry?.todayExercises;
  const storedDone =
    isPatientCompletedTodayInStorage(params.patient.id) ||
    isPatientCompletedTodayInStorage(params.patient.hn);
  const completedOverride = Boolean(
    params.serverCompletedToday || params.indexEntry?.completedToday || storedDone,
  );
  const todaySchedule = resolveTodayPatientSchedule({
    rehabSchedules: resolvedSchedules,
    todayRoutine: resolvedRoutine,
    nextAppointmentDate: params.patient.nextAppointmentDate,
    fallbackNextAppointmentDate: params.patient.nextAppointmentDate,
    serverTodayExercises: serverExercises,
    completedTodayOverride: completedOverride,
    now: new Date(),
  });
  return {
    patient: {
      ...params.patient,
      nextAppointmentDate: todaySchedule.nextAppointmentDate,
    },
    todayRoutine: resolvedRoutine,
    confidencePercent: params.confidencePercent,
    rehabSchedules: resolvedSchedules ?? [],
    todayExercises: todaySchedule.todayExercises,
    hasAppointmentToday: todaySchedule.hasAppointmentToday,
    completedToday: todaySchedule.completedToday,
  };
}

interface KioskBiometricLoginViewProps {
  apiClient: DrTechCaresApiClient;
  kioskId?: string;
  staffModalOpen?: boolean;
  onPatientAuthenticated: (payload: KioskLoginSuccessPayload) => void;
  onOpenStaffLogin: () => void;
  onOpenStaffRegister: () => void;
  onOpenTelemetrySimulator?: () => void;
}

type LangButtonOption = {
  code: ClinicalLang;
  prefix: string;
  label: string;
  numberedLabel: string;
  ttsLang: VoiceCoachLanguage;
  bcp47: string;
};

const LANG_OPTIONS: LangButtonOption[] = [
  { code: 'th', prefix: 'TH', label: 'ไทย', numberedLabel: '1. ไทย', ttsLang: 'th', bcp47: 'th-TH' },
  { code: 'en', prefix: 'EN', label: 'ENG', numberedLabel: '2. อังกฤษ (EN)', ttsLang: 'en', bcp47: 'en-US' },
  { code: 'cn', prefix: 'CN', label: '中文', numberedLabel: '3. จีน (中文)', ttsLang: 'zh', bcp47: 'zh-CN' },
];

export interface KioskVerifiedModalCopy {
  confidenceBadge: string;
  hnAndNextAppointment: string;
  openingCamera: string;
  speechCornerBadge: string;
  speechListenBanner: string;
  speechEarlyRaiseBanner: string;
  speechBarHeader: string;
  gestureDetectedPill: string;
  gestureSingleHandPill: string;
  gestureLoadingPill: string;
  gestureReadyPill: string;
  startBtnHolding: string;
  startBtnIdle: string;
  telemetryBtn: string;
  notMeBtn: string;
  voiceGreeting: string;
}

export const VERIFIED_HAND_HOLD_MS = 4000;
export const HAND_SWITCH_GRACE_MS = 600;
export const HAND_GLITCH_GRACE_MS = 220;
export const VERIFIED_MODAL_SPEECH_RATE = 1.12;

export function computeWelcomeSpeechFallbackMs(greetingText: string): number {
  return Math.max(3500, greetingText.trim().length * 75);
}

export function isVerifiedModalSpeechBlocked(params: {
  speechDone: boolean;
  browserSpeaking?: boolean;
}): boolean {
  return !params.speechDone || Boolean(params.browserSpeaking);
}

export function evaluateStrictTwoHandsForStartModal(
  hands: ReadonlyArray<ReadonlyArray<{ x: number; y: number }>>,
  pose?: ReadonlyArray<{ x: number; y: number; visibility?: number }> | null,
): {
  leftUp: boolean;
  rightUp: boolean;
  bothUp: boolean;
  singleOnly: boolean;
} {
  const raisedHands = hands.filter((h) => h.length >= 21 && (h[0]?.y ?? 1) <= 0.88);
  let leftUp = false;
  let rightUp = false;

  if (raisedHands.length >= 2) {
    for (let i = 0; i < raisedHands.length; i += 1) {
      for (let j = i + 1; j < raisedHands.length; j += 1) {
        const a = raisedHands[i][9] ?? raisedHands[i][0];
        const b = raisedHands[j][9] ?? raisedHands[j][0];
        if (Math.hypot(a.x - b.x, a.y - b.y) >= 0.05) {
          leftUp = true;
          rightUp = true;
        }
      }
    }
  }

  if (!leftUp || !rightUp) {
    for (const h of raisedHands) {
      const anchor = h[9] ?? h[0];
      const mirroredX = 1 - anchor.x;
      if (mirroredX < 0.5) leftUp = true;
      else rightUp = true;
    }
  }

  if (pose && pose.length >= 21) {
    const lSh = pose[POSE_INDEX.LEFT_SHOULDER];
    const rSh = pose[POSE_INDEX.RIGHT_SHOULDER];
    const lWr = pose[POSE_INDEX.LEFT_WRIST];
    const rWr = pose[POSE_INDEX.RIGHT_WRIST];
    const lShY = lSh && (lSh.visibility ?? 1) >= 0.3 ? lSh.y : 0.72;
    const rShY = rSh && (rSh.visibility ?? 1) >= 0.3 ? rSh.y : 0.72;
    if (lWr && (lWr.visibility ?? 0) >= 0.45 && lWr.y <= Math.min(0.82, lShY + 0.08)) {
      if (1 - lWr.x < 0.5) leftUp = true;
      else rightUp = true;
    }
    if (rWr && (rWr.visibility ?? 0) >= 0.45 && rWr.y <= Math.min(0.82, rShY + 0.08)) {
      if (1 - rWr.x < 0.5) leftUp = true;
      else rightUp = true;
    }
  }

  const bothUp = leftUp && rightUp;
  return {
    leftUp,
    rightUp,
    bothUp,
    singleOnly: (leftUp || rightUp) && !bothUp,
  };
}

export function stepVerifiedHandHold(input: {
  bothHandsNow?: boolean;
  anyHandNow?: boolean;
  speechBlocked?: boolean;
  nowMs: number;
  dtMs: number;
  lastRaisedAtMs: number;
  holdMs: number;
}): {
  detected: boolean;
  lastRaisedAtMs: number;
  holdMs: number;
  progress: number;
  confirmed: boolean;
} {
  if (input.speechBlocked) {
    return {
      detected: false,
      lastRaisedAtMs: -Infinity,
      holdMs: 0,
      progress: 0,
      confirmed: false,
    };
  }

  const activeBoth = input.bothHandsNow ?? input.anyHandNow ?? false;
  const nextLastRaised = activeBoth ? input.nowMs : input.lastRaisedAtMs;
  const withinMicroGlitch =
    !activeBoth && Number.isFinite(nextLastRaised) && input.nowMs - nextLastRaised <= HAND_GLITCH_GRACE_MS;

  // Genuine 4.0s hold timer: only increments while BOTH hands are held up simultaneously
  // AND after the welcome speech relay lock has released.
  const nextHoldMs = activeBoth
    ? Math.min(VERIFIED_HAND_HOLD_MS, input.holdMs + input.dtMs)
    : withinMicroGlitch
      ? input.holdMs
      : Math.max(0, input.holdMs - input.dtMs * 1.5);
  const progress = Math.min(1, nextHoldMs / VERIFIED_HAND_HOLD_MS);
  return {
    detected: activeBoth,
    lastRaisedAtMs: nextLastRaised,
    holdMs: nextHoldMs,
    progress,
    confirmed: nextHoldMs >= VERIFIED_HAND_HOLD_MS,
  };
}

export function getKioskVerifiedModalCopy(
  lang: ClinicalLang,
  params: {
    fullName: string;
    hn: string;
    nextAppointmentDate: string;
    confidencePercent: number;
    twoHandProgressPct: number;
  },
): KioskVerifiedModalCopy {
  const nextAppt = params.nextAppointmentDate || '-';
  const totalSec = VERIFIED_HAND_HOLD_MS / 1000;
  const heldSecNum = Math.min(totalSec, (params.twoHandProgressPct / 100) * totalSec);
  const heldSec = heldSecNum.toFixed(1);
  const remainSec = Math.max(0, totalSec - heldSecNum).toFixed(1);

  if (lang === 'en') {
    return {
      confidenceBadge: `Biometric Confidence ${params.confidencePercent}% (Verified)`,
      hnAndNextAppointment: `HN: ${params.hn} • Next Appointment: ${nextAppt}`,
      openingCamera: 'Opening camera for 2-hand detection...',
      speechCornerBadge: '🔊 Speaking guide...',
      speechListenBanner: '🔊 Please listen until the guide finishes before raising both hands...',
      speechEarlyRaiseBanner: '⏳ Please wait for the voice guide to finish before holding hands up',
      speechBarHeader: '🔊 Waiting for voice guide to finish before starting...',
      gestureDetectedPill: `🙌 Both hands detected! Hold steady ${heldSec}/${totalSec.toFixed(0)}.0s (${params.twoHandProgressPct}%)`,
      gestureSingleHandPill: '⚠️ 1 hand only does not count — raise BOTH hands together',
      gestureLoadingPill: '⏳ Preparing AI 2-hand detector...',
      gestureReadyPill: `🙌 Guide finished! Raise BOTH hands & hold ${totalSec.toFixed(0)}s to start`,
      startBtnHolding: `🙌 Holding BOTH hands... ${heldSec}/${totalSec.toFixed(0)}.0s (${remainSec}s left)`,
      startBtnIdle: `🙌 Raise BOTH Hands ${totalSec.toFixed(0)}s (Both Required) or Tap to Start ➔`,
      telemetryBtn: 'Open Biomechanics Telemetry (Smart Rehab Telemetry)',
      notMeBtn: 'Not me (Scan again)',
      voiceGreeting: `Identity verified. Welcome, ${params.fullName}. Please raise both hands together and hold for ${totalSec.toFixed(0)} seconds, or tap the button to start.`,
    };
  }
  if (lang === 'cn') {
    return {
      confidenceBadge: `生物识别准确度 ${params.confidencePercent}%（验证通过）`,
      hnAndNextAppointment: `病历号 HN: ${params.hn} • 下次预约: ${nextAppt}`,
      openingCamera: '正在开启双手举起识别摄像头...',
      speechCornerBadge: '🔊 语音引导中...',
      speechListenBanner: '🔊 请先听完语音引导后再举起双手...',
      speechEarlyRaiseBanner: '⏳ 请等语音引导结束后再举手保持',
      speechBarHeader: '🔊 等待语音引导结束后开始双手检测...',
      gestureDetectedPill: `🙌 已检测到双手举起！请保持 ${heldSec}/${totalSec.toFixed(0)}.0 秒 (${params.twoHandProgressPct}%)`,
      gestureSingleHandPill: '⚠️ 单手不计入 — 请同时举起双手',
      gestureLoadingPill: '⏳ 正在准备 AI 双手识别模块...',
      gestureReadyPill: `🙌 语音引导已结束！请双手举起保持 ${totalSec.toFixed(0)} 秒开始训练`,
      startBtnHolding: `🙌 正在加载确认... ${heldSec}/${totalSec.toFixed(0)}.0 秒（剩余 ${remainSec} 秒）`,
      startBtnIdle: `🙌 同时举起双手保持 ${totalSec.toFixed(0)} 秒 或 点击此处开始训练 ➔`,
      telemetryBtn: '打开生物力学分析界面 (Smart Rehab Telemetry)',
      notMeBtn: '不是我本人（重新扫描）',
      voiceGreeting: `身份验证成功，欢迎您，${params.fullName}。请同时举起双手保持 ${totalSec.toFixed(0)} 秒，或点击按钮开始训练。`,
    };
  }
  return {
    confidenceBadge: `ความแม่นยำชีวมาตร ${params.confidencePercent}% (ผ่านเกณฑ์)`,
    hnAndNextAppointment: `HN: ${params.hn} • นัดถัดไป: ${nextAppt}`,
    openingCamera: 'กำลังเปิดกล้องตรวจจับการยกมือทั้ง 2 ข้าง...',
    speechCornerBadge: '🔊 กำลังพูดแนะนำ...',
    speechListenBanner: '🔊 กรุณาฟังเสียงแนะนำให้จบก่อนยกมือทั้ง 2 ข้าง...',
    speechEarlyRaiseBanner: '⏳ รอให้เสียงพูดจบก่อนนะครับ แล้วค่อยยกมือค้างไว้',
    speechBarHeader: '🔊 กำลังรอเสียงแนะนำพูดจบก่อนเริ่มตรวจจับ (ล็อก 0%)',
    gestureDetectedPill: `🙌 ตรวจพบยกครบ 2 ข้าง! ค้างไว้ ${heldSec}/${totalSec.toFixed(0)}.0 วิ (${params.twoHandProgressPct}%)`,
    gestureSingleHandPill: '⚠️ ยกมือเดียวไม่นับ • กรุณายกขึ้นทั้ง 2 ข้างพร้อมกัน',
    gestureLoadingPill: '⏳ กำลังเตรียม AI ตรวจจับการยกมือ 2 ข้าง...',
    gestureReadyPill: `🙌 เสียงพูดจบแล้ว! ยกมือทั้ง 2 ข้างค้างไว้ ${totalSec.toFixed(0)} วิ เพื่อเริ่มฝึก`,
    startBtnHolding: `🙌 กำลังโหลด 2 ข้าง... ${heldSec}/${totalSec.toFixed(0)}.0 วิ (เหลือ ${remainSec} วิ)`,
    startBtnIdle: `🙌 ยกมือทั้ง 2 ข้างค้าง ${totalSec.toFixed(0)} วิ (ต้อง 2 ข้างเท่านั้น) หรือ กดที่นี่ ➔`,
    telemetryBtn: 'เปิดหน้าจอวิเคราะห์ชีวกลศาสตร์ (Smart Rehab Telemetry)',
    notMeBtn: 'ไม่ใช่ฉัน (สแกนใหม่)',
    voiceGreeting: `ยืนยันตัวตนสำเร็จ ยินดีต้อนรับคุณ ${params.fullName} กรุณายกมือทั้งสองข้างพร้อมกันค้างไว้ ${totalSec.toFixed(0)} วินาที หรือกดปุ่มเพื่อเริ่มฝึกครับ`,
  };
}

const REQUIRED_LOGIN_FACE_LANDMARKS = [10, 152, 33, 263, 1, 61, 291] as const;

export function isFullFrontalFaceReadyForLogin(
  landmarks: readonly { x: number; y: number; z?: number }[],
  liveness: Pick<LivenessAndPoseReport, 'yawDeg' | 'pitchDeg' | 'depthScore'>,
  inspection: Pick<BiometricVerificationJson, 'scan_status' | 'error_code'>,
): boolean {
  if (landmarks.length < 300) return false;
  if (inspection.scan_status === 'SPOOF_DETECTED' || inspection.error_code === 'SPOOF_ATTEMPT') {
    return false;
  }
  if (
    inspection.error_code === 'OBSTRUCTED_EYES' ||
    inspection.error_code === 'OBSTRUCTED_MOUTH'
  ) {
    return false;
  }

  for (const idx of REQUIRED_LOGIN_FACE_LANDMARKS) {
    const pt = landmarks[idx];
    if (
      !pt ||
      !Number.isFinite(pt.x) ||
      !Number.isFinite(pt.y) ||
      pt.x < 0.03 ||
      pt.x > 0.97 ||
      pt.y < 0.03 ||
      pt.y > 0.97
    ) {
      return false;
    }
  }

  const box = computeBoundingBoxFromLandmarks(landmarks);
  // Supports ultra-long-range clinical distance (0.4m - 4.5m, down to ~4% frame height)
  if (!box || box.height < 0.035 || box.width < 0.025) {
    return false;
  }

  // Allow natural slight head turn / tilt (up to ±40° yaw, ±35° pitch) matching 5-angle bank
  if (Math.abs(liveness.yawDeg) > 40 || Math.abs(liveness.pitchDeg) > 35) {
    return false;
  }

  if (liveness.depthScore < 0.22) {
    return false;
  }

  return true;
}

const COPY: Record<
  ClinicalLang,
  {
    subtitle: string;
    registerNewPatient: string;
    callStaff: string;
    staffLogin: string;
    checkingLiveness: string;
    longRangePrompt: string;
    distancePrefix: string;
    distanceUnit: string;
    liveVerified: string;
    spoofBlocked: string;
    alignFace: string;
    verifying: string;
    turnHeadPrefix: string;
    turnHeadSuffix: string;
    rescanBtn: string;
    pinBtn: string;
    enrollSelfBtn: string;
    footerBio: string;
    aboutSystem: string;
    cameraPausedTitle: string;
    cameraPausedDesc: string;
    resumeCameraBtn: string;
    usePinFallbackBtn: string;
    staffCalledToast: string;
    notFoundTitle: string;
    notFoundDesc: string;
    staffAssistBtn: string;
    retryScanBtn: string;
    countdownLabel: string;
    liveMatchVerified: string;
    liveMatchAccuracy: string;
    framesUnit: string;
    scanningDbTitle: string;
    comparingAccountsPrefix: string;
    comparingAccountsSuffix: string;
    pinModalTitle: string;
    pinModalDesc: string;
    pinClearBtn: string;
    pinNotFoundError: string;
    pinAuthError: string;
    aboutBullet1: string;
    aboutBullet2Prefix: string;
    aboutBullet2Suffix: string;
    aboutBullet3: string;
    aboutTelemetryBtn: string;
    aboutCloseBtn: string;
  }
> = {
  th: {
    subtitle: 'ระบบฟื้นฟูกายภาพบำบัดอัจฉริยะ',
    registerNewPatient: 'ลงทะเบียนคนไข้ใหม่',
    callStaff: 'เรียกเจ้าหน้าที่ช่วยเหลือ',
    staffLogin: 'เข้าสู่ระบบเจ้าหน้าที่',
    checkingLiveness: '👁️ กรุณากะพริบตาหรือขยับศีรษะเล็กน้อยเพื่อยืนยันตัวตน',
    longRangePrompt: '🎯 ตรวจพบใบหน้าระยะไกล • มองตรงหรือพยักหน้าเบาๆ เพื่อเข้าสู่ระบบ',
    distancePrefix: 'ระยะห่าง',
    distanceUnit: 'ม.',
    liveVerified: '✓ ยืนยันบุคคลจริง (Liveness Verified)',
    spoofBlocked: '⚠️ ตรวจพบภาพถ่ายหรือหน้าจอมือถือ — กรุณากะพริบตาหรือขยับศีรษะด้วยใบหน้าจริง',
    alignFace: 'ระบบสแกนอัตโนมัติเต็มจอ • มองกล้องเพื่อยืนยันตัวตน',
    verifying: 'กำลังสแกนและเปรียบเทียบกับฐานข้อมูลใบหน้า...',
    turnHeadPrefix: '😊 กรุณาหันหน้าเข้าหากล้องอีกนิด กำลังจับคู่บัญชี... (ความคล้าย',
    turnHeadSuffix: '%)',
    rescanBtn: 'เริ่มสแกนใบหน้าใหม่อีกครั้ง',
    pinBtn: 'ใช้รหัส PIN 4 หลัก',
    enrollSelfBtn: 'สมัครคนไข้ใหม่',
    footerBio: 'มาตรฐานชีวมาตร • เชื่อมต่อ Supabase Cloud สด',
    aboutSystem: 'เกี่ยวกับระบบ',
    cameraPausedTitle: 'หยุดกล้องชั่วคราวหรือกล้องไม่พร้อมใช้งาน',
    cameraPausedDesc: 'กดปุ่มด้านล่างเพื่อเปิดกล้องสแกนใบหน้า หรือใช้รหัส PIN 4 หลักแทน',
    resumeCameraBtn: 'เปิดกล้องเริ่มสแกนใหม่',
    usePinFallbackBtn: 'ใช้รหัส PIN 4 หลักแทน',
    staffCalledToast: 'ส่งสัญญาณเรียกเจ้าหน้าที่เรียบร้อยแล้ว กรุณารอสักครู่',
    notFoundTitle: 'ไม่พบบัญชีในระบบ',
    notFoundDesc:
      'ใบหน้านี้ยังไม่ถูกลงทะเบียนในฐานข้อมูล Supabase Cloud กรุณาให้เจ้าหน้าที่ช่วยลงทะเบียน หรือลองสแกนใหม่อีกครั้ง',
    staffAssistBtn: 'เจ้าหน้าที่ลงทะเบียนให้',
    retryScanBtn: 'ลองสแกนใหม่อีกครั้ง',
    countdownLabel: 'กำลังนับถอยหลัง',
    liveMatchVerified: '✓ ยืนยันตัวตนสำเร็จ',
    liveMatchAccuracy: 'ความแม่นยำ',
    framesUnit: 'เฟรม',
    scanningDbTitle: 'กำลังสแกนใบหน้าเทียบฐานข้อมูล...',
    comparingAccountsPrefix: 'เทียบฐานข้อมูล',
    comparingAccountsSuffix: 'บัญชี • ตรวจจับละเอียด 478 จุดใบหน้า',
    pinModalTitle: 'ยืนยันตัวตนด้วยรหัส PIN',
    pinModalDesc: 'กรอกรหัส PIN 4 หลักประจำตัวผู้ป่วยเพื่อเริ่มฝึก',
    pinClearBtn: 'ล้าง',
    pinNotFoundError: 'ไม่พบข้อมูลผู้ป่วยสำหรับรหัส PIN นี้ (หรือต้องให้เจ้าหน้าที่เข้าสู่ระบบก่อน)',
    pinAuthError: 'การค้นหารหัส PIN 4 หลักต้องใช้สิทธิ์เจ้าหน้าที่ หรือติดต่อเคาน์เตอร์พยาบาล',
    aboutBullet1: '• ระบบยืนยันตัวตนชีวมาตร 512 มิติ พร้อม Anti-Spoofing 3D Liveness Detection',
    aboutBullet2Prefix: '• เชื่อมต่อฐานข้อมูล Supabase Cloud (',
    aboutBullet2Suffix: ') แบบเรียลไทม์',
    aboutBullet3: '• รองรับการใช้งาน 3 ภาษา (ไทย / 中文 / English) และการควบคุมไร้สัมผัส',
    aboutTelemetryBtn: 'เปิดโหมดจำลองกายภาพบำบัด (SmartRehab Telemetry)',
    aboutCloseBtn: 'ปิดหน้าต่าง',
  },
  en: {
    subtitle: 'AI Smart Rehabilitation Platform',
    registerNewPatient: 'Register Patient',
    callStaff: 'Call Staff',
    staffLogin: 'Staff Login',
    checkingLiveness: '👁️ Please blink or gently move your head to verify',
    longRangePrompt: '🎯 Long-Range Face Locked • Look ahead or nod gently to sign in',
    distancePrefix: 'Distance',
    distanceUnit: 'm',
    liveVerified: '✓ Live Person Verified (Liveness Verified)',
    spoofBlocked: '⚠️ Photo or Phone Screen Detected — Blink or move head with a live face',
    alignFace: 'Full-screen auto-scan active • Look at the camera',
    verifying: 'Scanning & matching face against database...',
    turnHeadPrefix: '😊 Turn slightly toward the camera, matching profile... (',
    turnHeadSuffix: '% match)',
    rescanBtn: 'Restart Face Scan',
    pinBtn: 'Use 4-Digit PIN',
    enrollSelfBtn: 'New Patient Sign Up',
    footerBio: 'Biometric Standard • Connected to Live Supabase Cloud',
    aboutSystem: 'About System',
    cameraPausedTitle: 'Camera Paused or Unavailable',
    cameraPausedDesc: 'Tap below to restart camera face scanning or sign in with your 4-digit PIN.',
    resumeCameraBtn: 'Restart Camera Scan',
    usePinFallbackBtn: 'Use 4-Digit PIN Instead',
    staffCalledToast: 'Staff alert dispatched cleanly. Assistance is on the way.',
    notFoundTitle: 'Account Not Found',
    notFoundDesc:
      'No matching biometric profile was found in Supabase Cloud. Please ask staff to enroll you or try scanning again.',
    staffAssistBtn: 'Staff Register Patient',
    retryScanBtn: 'Scan Face Again',
    countdownLabel: 'Hold steady',
    liveMatchVerified: '✓ Identity Verified',
    liveMatchAccuracy: 'Confidence',
    framesUnit: 'frames',
    scanningDbTitle: 'Scanning face against database...',
    comparingAccountsPrefix: 'Matching',
    comparingAccountsSuffix: 'profiles • 478-point 3D facial mesh',
    pinModalTitle: 'Sign In with 4-Digit PIN',
    pinModalDesc: 'Enter your 4-digit patient PIN to begin rehabilitation',
    pinClearBtn: 'Clear',
    pinNotFoundError: 'No patient matched that PIN (or staff login may be required first)',
    pinAuthError: 'PIN lookup requires clinical authorization. Please ask staff for assistance.',
    aboutBullet1: '• 512-D facial biometric authentication with 3D Anti-Spoofing Liveness Detection',
    aboutBullet2Prefix: '• Connected in real time to Supabase Cloud (',
    aboutBullet2Suffix: ')',
    aboutBullet3: '• Full trilingual support (Thai / English / 中文) & touchless gesture control',
    aboutTelemetryBtn: 'Open Physical Therapy Simulator (SmartRehab Telemetry)',
    aboutCloseBtn: 'Close',
  },
  cn: {
    subtitle: '智能物理康复训练系统',
    registerNewPatient: '注册新患者',
    callStaff: '呼叫医护人员',
    staffLogin: '医护系统登录',
    checkingLiveness: '👁️ 请轻眨眼或微转头部以验证真人身份',
    longRangePrompt: '🎯 远距离人脸已锁定 • 请直视镜头或轻点头以登录',
    distancePrefix: '距离',
    distanceUnit: '米',
    liveVerified: '✓ 真人活体已验证 (Liveness Verified)',
    spoofBlocked: '⚠️ 检测到照片或手机屏幕 — 请使用真人眨眼或轻转头部',
    alignFace: '全屏自动扫描已开启 • 请注视摄像头',
    verifying: '正在扫描并与人脸数据库比对...',
    turnHeadPrefix: '😊 请略微转正面向摄像头，正在匹配档案…（相似度',
    turnHeadSuffix: '%）',
    rescanBtn: '重新开始人脸扫描',
    pinBtn: '使用 4 位 PIN 码',
    enrollSelfBtn: '新患者注册',
    footerBio: '医疗级生物识别标准 • 实时连接 Supabase 云端',
    aboutSystem: '关于系统',
    cameraPausedTitle: '摄像头已暂停或未就绪',
    cameraPausedDesc: '点击下方重新开启人脸扫描，或使用 4 位 PIN 码登录。',
    resumeCameraBtn: '开启摄像头重新扫描',
    usePinFallbackBtn: '改用 4 位 PIN 码',
    staffCalledToast: '已呼叫医护人员，请稍候...',
    notFoundTitle: '未找到系统账户',
    notFoundDesc: '云端数据库中未匹配到该人脸档案，请联系医护人员协助建档或重新扫描。',
    staffAssistBtn: '医护人员协助建档',
    retryScanBtn: '重新识别人脸',
    countdownLabel: '倒计时准备中',
    liveMatchVerified: '✓ 身份验证成功',
    liveMatchAccuracy: '匹配准确度',
    framesUnit: '帧',
    scanningDbTitle: '正在扫描并与人脸库比对...',
    comparingAccountsPrefix: '比对数据库共',
    comparingAccountsSuffix: '位档案 • 478 点三维人脸特征追踪',
    pinModalTitle: '使用 4 位 PIN 码验证',
    pinModalDesc: '请输入患者 4 位 PIN 码开始康复训练',
    pinClearBtn: '清空',
    pinNotFoundError: '未找到匹配该 PIN 码的患者档案（或需医护人员先登录）',
    pinAuthError: '查询 4 位 PIN 码需要医护权限，请联系护士台协助。',
    aboutBullet1: '• 512 维面部生物识别认证，配备 3D 防伪活体检测',
    aboutBullet2Prefix: '• 实时连接 Supabase 云端数据库（',
    aboutBullet2Suffix: '）',
    aboutBullet3: '• 支持三语界面（泰语 / 英语 / 中文）及无接触手势控制',
    aboutTelemetryBtn: '打开物理康复分析界面 (SmartRehab Telemetry)',
    aboutCloseBtn: '关闭窗口',
  },
};

export function KioskBiometricLoginView({
  apiClient,
  kioskId = 'kiosk-01',
  staffModalOpen = false,
  onPatientAuthenticated,
  onOpenStaffLogin,
  onOpenStaffRegister,
  onOpenTelemetrySimulator,
}: KioskBiometricLoginViewProps) {
  const { lang, setLang } = useClinicalI18n();
  const c = COPY[lang];

  const [verifiedModalPayload, setVerifiedModalPayload] = useState<KioskLoginSuccessPayload | null>(null);
  const [twoHandProgress, setTwoHandProgress] = useState(0);
  const [twoHandDetected, setTwoHandDetected] = useState(false);
  const [leftHandUpState, setLeftHandUpState] = useState(false);
  const [rightHandUpState, setRightHandUpState] = useState(false);
  const [twoHandCameraReady, setTwoHandCameraReady] = useState(false);
  const [twoHandDetectorReady, setTwoHandDetectorReady] = useState(false);
  const [isWelcomeSpeechDone, setIsWelcomeSpeechDone] = useState(false);
  const [raisedDuringSpeech, setRaisedDuringSpeech] = useState(false);
  const isSpeechDoneRef = useRef<boolean>(false);
  const activeUtteranceRef = useRef<SpeechSynthesisUtterance | null>(null);

  const handleSelectLang = useCallback(
    (nextLang: ClinicalLang) => {
      // Re-lock Speech-First Relay synchronously when patient switches modal language
      isSpeechDoneRef.current = false;
      setIsWelcomeSpeechDone(false);
      setRaisedDuringSpeech(false);
      setTwoHandProgress(0);
      setTwoHandDetected(false);
      setLeftHandUpState(false);
      setRightHandUpState(false);
      setLang(nextLang);
      if (typeof window !== 'undefined') {
        try {
          window.localStorage.setItem(
            'drtechcares_voice_coach_lang_v1',
            nextLang === 'cn' ? 'zh' : nextLang,
          );
        } catch {
          // Ignore storage errors
        }
      }
    },
    [setLang],
  );

  const [cameraPaused, setCameraPaused] = useState(false);
  const [clockText, setClockText] = useState<string>(() =>
    new Date().toLocaleTimeString('th-TH', { hour: '2-digit', minute: '2-digit', hour12: false }),
  );

  useEffect(() => {
    const updateClock = () => {
      setClockText(
        new Date().toLocaleTimeString('th-TH', { hour: '2-digit', minute: '2-digit', hour12: false }),
      );
    };
    updateClock();
    const timer = window.setInterval(updateClock, 15000);
    return () => window.clearInterval(timer);
  }, []);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [rotationDeg, setRotationDeg] = useState<0 | 90 | 180 | 270>(0);
  const [mirrored, setMirrored] = useState(true);
  const [showCalibHud, setShowCalibHud] = useState(false);

  const [scanStage, setScanStage] = useState<'scanning' | 'burst' | 'verifying' | 'not_found'>('scanning');
  const [burstCount, setBurstCount] = useState(0);
  const [livenessInfo, setLivenessInfo] = useState<LivenessAndPoseReport | null>(null);
  const [bioReport, setBioReport] = useState<BiologicalLivenessReport | null>(null);
  const [inspectionInfo, setInspectionInfo] = useState<BiometricVerificationJson | null>(null);
  const [lastVerifyResult, setLastVerifyResult] = useState<KioskFaceVerifyResponse | null>(null);

  const modalCopy = verifiedModalPayload
    ? getKioskVerifiedModalCopy(lang, {
        fullName: verifiedModalPayload.patient.fullName,
        hn: verifiedModalPayload.patient.hn,
        nextAppointmentDate: verifiedModalPayload.patient.nextAppointmentDate,
        confidencePercent: verifiedModalPayload.confidencePercent,
        twoHandProgressPct: Math.round(twoHandProgress * 100),
      })
    : null;
  const modalAudioRef = useRef<HTMLAudioElement | null>(null);
  const modalSpeakSeqRef = useRef<number>(0);

  useEffect(() => {
    const greeting = modalCopy?.voiceGreeting;
    if (!verifiedModalPayload || !greeting || typeof window === 'undefined') {
      isSpeechDoneRef.current = false;
      setIsWelcomeSpeechDone(false);
      setRaisedDuringSpeech(false);
      return;
    }

    const seq = ++modalSpeakSeqRef.current;
    const langOpt = LANG_OPTIONS.find((item) => item.code === lang) ?? LANG_OPTIONS[0];

    // Lock two-hand confirmation strictly at 0% until welcome voice guidance completes
    isSpeechDoneRef.current = false;
    setIsWelcomeSpeechDone(false);
    setRaisedDuringSpeech(false);
    setTwoHandProgress(0);
    setTwoHandDetected(false);
    setLeftHandUpState(false);
    setRightHandUpState(false);

    let finished = false;
    let webFallbackTimer: number | null = null;
    let safetyTimer: number | null = null;
    let speakingPollTimer: number | null = null;
    const fallbackMs = computeWelcomeSpeechFallbackMs(greeting);
    const hardDeadlineAt = Date.now() + fallbackMs + 5000;

    const clearTimers = () => {
      if (webFallbackTimer !== null) {
        window.clearTimeout(webFallbackTimer);
        webFallbackTimer = null;
      }
      if (safetyTimer !== null) {
        window.clearTimeout(safetyTimer);
        safetyTimer = null;
      }
      if (speakingPollTimer !== null) {
        window.clearInterval(speakingPollTimer);
        speakingPollTimer = null;
      }
    };

    const markSpeechDone = () => {
      if (finished || modalSpeakSeqRef.current !== seq) return;
      finished = true;
      clearTimers();
      activeUtteranceRef.current = null;
      isSpeechDoneRef.current = true;
      setIsWelcomeSpeechDone(true);
      setRaisedDuringSpeech(false);
    };

    const stopSpeech = () => {
      modalSpeakSeqRef.current += 1;
      finished = true;
      clearTimers();
      activeUtteranceRef.current = null;
      const audio = modalAudioRef.current;
      if (audio) {
        try {
          audio.onended = null;
          audio.onerror = null;
          audio.pause();
          audio.currentTime = 0;
        } catch {
          // Ignore audio cleanup errors
        }
      }
      if ('speechSynthesis' in window) {
        try {
          window.speechSynthesis.cancel();
        } catch {
          // Ignore synthesis cancel errors
        }
      }
      isSpeechDoneRef.current = false;
      setIsWelcomeSpeechDone(false);
      setRaisedDuringSpeech(false);
    };

    // Cancel any residual utterance before speaking the welcome instructions
    if ('speechSynthesis' in window) {
      try {
        window.speechSynthesis.cancel();
      } catch {
        // Ignore synthesis cancel errors
      }
    }

    const startSafetyMonitor = () => {
      safetyTimer = window.setTimeout(() => {
        safetyTimer = null;
        if (finished || modalSpeakSeqRef.current !== seq) return;
        const checkAndUnlock = () => {
          if (finished || modalSpeakSeqRef.current !== seq) return;
          const synthSpeaking =
            'speechSynthesis' in window &&
            Boolean(window.speechSynthesis.speaking || window.speechSynthesis.pending);
          const audioEl = modalAudioRef.current;
          const audioPlaying = Boolean(
            audioEl && !audioEl.paused && !audioEl.ended && audioEl.currentTime > 0,
          );
          if ((!synthSpeaking && !audioPlaying) || Date.now() >= hardDeadlineAt) {
            markSpeechDone();
          }
        };
        checkAndUnlock();
        if (!finished && modalSpeakSeqRef.current === seq) {
          speakingPollTimer = window.setInterval(checkAndUnlock, 250);
        }
      }, fallbackMs);
    };
    startSafetyMonitor();

    const speakWebFallback = () => {
      if (finished || modalSpeakSeqRef.current !== seq) return;
      if (!('speechSynthesis' in window) || typeof SpeechSynthesisUtterance === 'undefined') {
        return;
      }
      try {
        const synth = window.speechSynthesis;
        const wasBusy = synth.speaking || synth.pending;
        if (wasBusy) synth.cancel();
        const startUtterance = () => {
          if (finished || modalSpeakSeqRef.current !== seq) return;
          const voices = synth.getVoices();
          const matchedVoice = pickSpeechVoice(voices, langOpt.bcp47);
          const utterance = new SpeechSynthesisUtterance(greeting);
          utterance.lang = matchedVoice?.lang || langOpt.bcp47;
          if (matchedVoice) utterance.voice = matchedVoice;
          utterance.rate = VERIFIED_MODAL_SPEECH_RATE;
          utterance.onend = () => markSpeechDone();
          utterance.onerror = () => {
            // Keep locked until fallback timer if speech engine errors/is muted
            activeUtteranceRef.current = null;
          };
          activeUtteranceRef.current = utterance;
          synth.speak(utterance);
        };
        if (wasBusy) {
          webFallbackTimer = window.setTimeout(() => {
            webFallbackTimer = null;
            startUtterance();
          }, 60);
        } else {
          startUtterance();
        }
      } catch {
        // Ignore browser TTS restriction; safety timer will unlock cleanly
      }
    };

    if (typeof Audio !== 'undefined') {
      try {
        let audio = modalAudioRef.current;
        if (!audio) {
          audio = new Audio();
          audio.preload = 'auto';
          modalAudioRef.current = audio;
        }
        audio.onended = () => markSpeechDone();
        audio.onerror = () => speakWebFallback();
        audio.defaultPlaybackRate = VERIFIED_MODAL_SPEECH_RATE;
        audio.src = buildTtsProxyUrl(greeting, langOpt.ttsLang);
        audio.playbackRate = VERIFIED_MODAL_SPEECH_RATE;
        const p = audio.play();
        if (p && typeof p.catch === 'function') {
          p.catch((err: unknown) => {
            const name =
              err && typeof err === 'object' && 'name' in err ? String((err as { name: string }).name) : '';
            if (name !== 'AbortError') {
              speakWebFallback();
            }
          });
        }
      } catch {
        speakWebFallback();
      }
    } else {
      speakWebFallback();
    }

    return stopSpeech;
  }, [verifiedModalPayload?.patient.id, lang, modalCopy?.voiceGreeting]);

  const [showPinModal, setShowPinModal] = useState(false);
  const [pinDigits, setPinDigits] = useState('');
  const [pinError, setPinError] = useState<string | null>(null);
  const [pinSubmitting, setPinSubmitting] = useState(false);

  const [showAboutModal, setShowAboutModal] = useState(false);
  const [toastMsg, setToastMsg] = useState<string | null>(null);
  const [scanGeneration, setScanGeneration] = useState(0);
  const [hudBox, setHudBox] = useState<{ xPct: number; yPct: number; wPct: number; hPct: number } | null>(null);
  const [hudLandmarks, setHudLandmarks] = useState<{ xPct: number; yPct: number }[]>([]);
  const [dbAccountCount, setDbAccountCount] = useState(0);
  const [liveMatchHud, setLiveMatchHud] = useState<{
    candidate: PatientBiometricIndexEntry;
    confidencePercent: number;
    bestAngle: string;
    streak: number;
    confirmed: boolean;
  } | null>(null);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const confirmVideoRef = useRef<HTMLVideoElement | null>(null);
  const confirmCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const confirmProgressFillRef = useRef<HTMLDivElement | null>(null);
  const confirmProgressTextRef = useRef<HTMLSpanElement | null>(null);
  const confirmPillLabelRef = useRef<HTMLSpanElement | null>(null);
  const confirmBtnFillRef = useRef<HTMLSpanElement | null>(null);
  const confirmBtnLabelRef = useRef<HTMLSpanElement | null>(null);
  const alignCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const contextCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const roiCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const bioLivenessRef = useRef(new BiologicalLivenessTracker());
  const bioReportRef = useRef<BiologicalLivenessReport | null>(null);
  bioReportRef.current = bioReport;
  const verifiedModalPayloadRef = useRef<KioskLoginSuccessPayload | null>(null);
  verifiedModalPayloadRef.current = verifiedModalPayload;
  const openVerifiedModal = useCallback((payload: KioskLoginSuccessPayload) => {
    isSpeechDoneRef.current = false;
    setIsWelcomeSpeechDone(false);
    setRaisedDuringSpeech(false);
    setTwoHandProgress(0);
    setTwoHandDetected(false);
    setLeftHandUpState(false);
    setRightHandUpState(false);
    if (confirmProgressFillRef.current) {
      confirmProgressFillRef.current.style.transform = 'scaleX(0)';
    }
    if (confirmBtnFillRef.current) {
      confirmBtnFillRef.current.style.transform = 'scaleX(0)';
    }
    verifiedModalPayloadRef.current = payload;
    setVerifiedModalPayload(payload);
  }, []);
  const onPatientAuthenticatedRef = useRef(onPatientAuthenticated);
  onPatientAuthenticatedRef.current = onPatientAuthenticated;
  const presenceRef = useRef(new FacePresenceTracker());
  const loginCtrlRef = useRef(new SingleScanLoginController());
  const verifyingLockRef = useRef(false);
  const edgeTelemetryRef = useRef<EdgeKioskTelemetryPayload | null>(null);

  const { connected: edgeConnected, telemetry: edgeTelemetry } = useEdgeVisionSocket({
    enabled: !cameraPaused && !staffModalOpen && !showPinModal,
    onPatientMatch: (edgeMatch) => {
      if (verifiedModalPayloadRef.current || verifyingLockRef.current || !edgeMatch.id) return;
      if (!bioReportRef.current?.biologicalVerified || bioReportRef.current?.spoofDetected) return;
      if (typeof edgeMatch.similarity === 'number' && edgeMatch.similarity < 0.85) return;
      const entry = biometricIndexRef.current.find(
        (e) => e.patientId === edgeMatch.id || (edgeMatch.hn && e.hn === edgeMatch.hn),
      );
      if (entry) {
        const nextPayload = buildKioskLoginSuccessPayload({
          patient: {
            id: entry.patientId,
            hn: entry.hn,
            fullName: entry.fullName,
            nextAppointmentDate: entry.nextAppointmentDate,
          },
          todayRoutine: entry.todayRoutine ?? null,
          confidencePercent: Math.round((edgeMatch.similarity || 0.92) * 100),
          indexEntry: entry,
        });
        openVerifiedModal(nextPayload);
        return;
      }
      void apiClient
        .listPatients()
        .then((patients) => {
          const found = patients.find(
            (p) => p.id === edgeMatch.id || (edgeMatch.hn && p.hn === edgeMatch.hn),
          );
          if (!found || verifiedModalPayloadRef.current) return;
          const nextPayload = buildKioskLoginSuccessPayload({
            patient: {
              id: found.id,
              hn: found.hn,
              fullName: found.fullName,
              nextAppointmentDate: found.nextAppointmentDate,
            },
            todayRoutine: null,
            confidencePercent: Math.round((edgeMatch.similarity || 0.92) * 100),
            rehabSchedules: found.rehabSchedules,
          });
          openVerifiedModal(nextPayload);
        })
        .catch(() => {});
    },
  });
  edgeTelemetryRef.current = edgeTelemetry;
  const biometricIndexRef = useRef<PatientBiometricIndexEntry[]>([]);
  const temporalMatcherRef = useRef(new TemporalBiometricMatcher(3));
  const [turningHintPct, setTurningHintPct] = useState<number | null>(null);
  const mirroredRef = useRef(mirrored);
  mirroredRef.current = mirrored;

  const supabaseHost = (() => {
    const raw = import.meta.env.VITE_SUPABASE_URL as string | undefined;
    if (!raw) return 'cloud-api';
    try {
      return new URL(raw).host;
    } catch {
      return raw;
    }
  })();

  const showToast = useCallback((msg: string) => {
    setToastMsg(msg);
    window.setTimeout(() => {
      setToastMsg((prev) => (prev === msg ? null : prev));
    }, 4500);
  }, []);

  useEffect(() => {
    return subscribeBiometricIndex(apiClient, (entries) => {
      biometricIndexRef.current = entries;
      setDbAccountCount(entries.length);
    });
  }, [apiClient]);

  const handleRestartScan = useCallback(() => {
    verifyingLockRef.current = false;
    presenceRef.current.reset();
    loginCtrlRef.current.reset();
    temporalMatcherRef.current.reset();
    bioLivenessRef.current.reset();
    setBurstCount(0);
    setInspectionInfo(null);
    setLivenessInfo(null);
    setBioReport(null);
    setHudBox(null);
    setHudLandmarks([]);
    setLiveMatchHud(null);
    setTurningHintPct(null);
    isSpeechDoneRef.current = false;
    setIsWelcomeSpeechDone(false);
    setRaisedDuringSpeech(false);
    if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
      try {
        window.speechSynthesis.cancel();
      } catch {
        // Ignore synthesis cancel errors
      }
    }
    verifiedModalPayloadRef.current = null;
    setVerifiedModalPayload(null);
    setTwoHandProgress(0);
    setTwoHandDetected(false);
    setLeftHandUpState(false);
    setRightHandUpState(false);
    setTwoHandCameraReady(false);
    setCameraError(null);
    setCameraPaused(false);
    setScanStage('scanning');
    setScanGeneration((g) => g + 1);
    void apiClient
      .getBiometricIndex()
      .then((entries) => {
        biometricIndexRef.current = entries;
        setDbAccountCount(entries.length);
      })
      .catch(() => {});
  }, [apiClient]);

  useEffect(() => {
    purgeLegacyBiometricCache();
  }, []);

  const isNotFoundStage = scanStage === 'not_found';
  const prevStaffModalOpenRef = useRef(staffModalOpen);

  useEffect(() => {
    const wasOpen = prevStaffModalOpenRef.current;
    prevStaffModalOpenRef.current = staffModalOpen;
    if (staffModalOpen && !wasOpen) {
      verifyingLockRef.current = false;
      presenceRef.current.reset();
      loginCtrlRef.current.reset();
      temporalMatcherRef.current.reset();
      bioLivenessRef.current.reset();
      setScanStage('scanning');
      setHudBox(null);
      setHudLandmarks([]);
      setLiveMatchHud(null);
      setTurningHintPct(null);
      setBurstCount(0);
    } else if (!staffModalOpen && wasOpen) {
      handleRestartScan();
    }
  }, [staffModalOpen, handleRestartScan]);

  useEffect(() => {
    if (cameraPaused || staffModalOpen || isNotFoundStage || showPinModal) {
      setTwoHandProgress(0);
      setTwoHandDetected(false);
      setTwoHandCameraReady(false);
      return;
    }

    let active = true;
    let stream: MediaStream | null = null;
    let rafId = 0;
    let landmarker: FaceLandmarker | null = null;
    let roiLandmarker: FaceLandmarker | null = null;
    let handLandmarker: HandLandmarker | null = null;
    let poseLandmarker: PoseLandmarker | null = null;
    let gestureOffscreenCanvas: HTMLCanvasElement | null = null;
    let prevLandmarks: { x: number; y: number; z?: number }[] | null = null;
    let wasEyeClosed = false;
    let burstStarted = false;
    let lastDetectTs = 0;
    let lastPoseDetectTs = 0;
    let lastProcessedVideoTime = -1;
    let lastFaceDetectWallMs = 0;
    let unreadyFrameStreak = 0;
    let lowSimBelow035StartAt = 0;
    let lastGestureVideoTime = -1;
    let lastGestureInferenceAt = 0;
    let lastPoseFallbackAt = 0;
    let lastGestureFrameAt = 0;
    let gestureHoldMs = 0;
    let lastAnyHandRaisedAt = -Infinity;
    let latestAnyHandUp = false;
    let lastReportedHolding = false;
    let lastReportedDetected = false;
    let twoHandCameraReadySent = false;
    let gestureTriggered = false;
    const videoEl = videoRef.current;

    const stopMediaAndDetectors = () => {
      cancelAnimationFrame(rafId);
      landmarker?.close();
      landmarker = null;
      roiLandmarker?.close();
      roiLandmarker = null;
      handLandmarker?.close();
      handLandmarker = null;
      poseLandmarker?.close();
      poseLandmarker = null;
      stream?.getTracks().forEach((track) => track.stop());
      stream = null;
      if (videoEl) {
        videoEl.srcObject = null;
      }
      if (confirmVideoRef.current) {
        confirmVideoRef.current.srcObject = null;
      }
    };

    async function warmupGestureDetectors(vision: Awaited<ReturnType<typeof FilesetResolver.forVisionTasks>>) {
      setTwoHandDetectorReady(false);
      const tryCreateHand = async () => {
        for (const modelAssetPath of [
          '/models/hand_landmarker.task',
          'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
        ]) {
          for (const delegate of ['GPU', 'CPU', undefined] as const) {
            try {
              const created = await HandLandmarker.createFromOptions(vision, {
                baseOptions: delegate ? { modelAssetPath, delegate } : { modelAssetPath },
                runningMode: 'VIDEO',
                numHands: 2,
                minHandDetectionConfidence: 0.35,
                minHandPresenceConfidence: 0.35,
                minTrackingConfidence: 0.35,
              });
              if (!active) {
                created.close();
                return;
              }
              handLandmarker = created;
              setTwoHandDetectorReady(true);
              return;
            } catch {
              // Try next delegate or URL
            }
          }
        }
      };

      const tryCreatePose = async () => {
        for (const modelAssetPath of ['/models/pose_landmarker.task', '/models/pose_landmarker_full.task']) {
          for (const delegate of ['GPU', 'CPU', undefined] as const) {
            try {
              const created = await PoseLandmarker.createFromOptions(vision, {
                baseOptions: delegate ? { modelAssetPath, delegate } : { modelAssetPath },
                runningMode: 'VIDEO',
                numPoses: 1,
                minPoseDetectionConfidence: 0.35,
                minPosePresenceConfidence: 0.35,
                minTrackingConfidence: 0.35,
              });
              if (!active) {
                created.close();
                return;
              }
              poseLandmarker = created;
              setTwoHandDetectorReady(true);
              return;
            } catch {
              // Try next delegate or pose model
            }
          }
        }
      };

      await Promise.all([tryCreateHand(), tryCreatePose()]);
    }

    async function startFacePipeline() {
      try {
        setTwoHandProgress(0);
        setTwoHandDetected(false);
        setTwoHandCameraReady(false);

        void apiClient
          .getBiometricIndex()
          .then((entries) => {
            if (active) {
              biometricIndexRef.current = entries;
              setDbAccountCount(entries.length);
            }
          })
          .catch(() => {});
        if (typeof navigator === 'undefined' || !navigator.mediaDevices?.getUserMedia) {
          setCameraError('Browser does not support camera access');
          setCameraPaused(true);
          return;
        }
        const deviceId = await selectClinicalCameraDeviceId();
        if (!active) return;
        stream = await navigator.mediaDevices.getUserMedia({
          video: deviceId
            ? { deviceId: { exact: deviceId }, width: { ideal: 1920 }, height: { ideal: 1080 }, facingMode: 'user' }
            : { width: { ideal: 1920 }, height: { ideal: 1080 }, facingMode: 'user' },
          audio: false,
        });
        if (!active || !videoRef.current) {
          stream.getTracks().forEach((track) => track.stop());
          stream = null;
          return;
        }
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        if (!active) return;
        setCameraError(null);

        let vision;
        for (const wasmBase of [
          '/mediapipe/wasm',
          '/models/wasm',
          'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304/wasm',
        ]) {
          try {
            vision = await FilesetResolver.forVisionTasks(wasmBase);
            break;
          } catch {
            // Try next WASM base path
          }
        }
        if (!vision || !active) return;

        // Pre-warm HandLandmarker + PoseLandmarker in parallel while face scanning runs
        void warmupGestureDetectors(vision);

        const createdLandmarker = await FaceLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: '/models/face_landmarker.task' },
          runningMode: 'VIDEO',
          numFaces: 2,
          outputFaceBlendshapes: true,
          minFaceDetectionConfidence: 0.45,
          minFacePresenceConfidence: 0.45,
          minTrackingConfidence: 0.45,
        });
        if (!active) {
          createdLandmarker.close();
          return;
        }
        landmarker = createdLandmarker;

        // Stage-2 Smart ROI Landmarker (IMAGE mode on 256x256 upscaled head zone for 1.5-4.5m range)
        try {
          const createdRoiLandmarker = await FaceLandmarker.createFromOptions(vision, {
            baseOptions: { modelAssetPath: '/models/face_landmarker.task' },
            runningMode: 'IMAGE',
            numFaces: 1,
            outputFaceBlendshapes: true,
            minFaceDetectionConfidence: 0.45,
            minFacePresenceConfidence: 0.45,
          });
          if (!active) {
            createdRoiLandmarker.close();
            return;
          }
          roiLandmarker = createdRoiLandmarker;
        } catch {
          roiLandmarker = null;
        }

        let lastEmbedCheckAt = 0;
        const KEY_MESH_LANDMARK_INDICES = [
          // Left & right eyes + iris
          33, 133, 159, 145, 362, 263, 386, 374,
          // Eyebrows
          70, 105, 107, 336, 334, 300,
          // Nose bridge, tip & nostrils
          168, 6, 197, 195, 5, 4, 1, 98, 327,
          // Mouth & lips
          61, 291, 0, 17, 78, 308, 13, 14,
          // Cheekbones & jawline
          234, 454, 172, 397, 152,
        ];

        const loop = async () => {
          if (!active || !videoRef.current || !landmarker) return;
          let shouldContinueLoop = true;
          try {
            const verifiedPayload = verifiedModalPayloadRef.current;
            if (verifiedPayload) {
              if (confirmVideoRef.current && stream) {
                if (confirmVideoRef.current.srcObject !== stream) {
                  confirmVideoRef.current.srcObject = stream;
                  void confirmVideoRef.current.play().catch(() => {});
                }
                if (!twoHandCameraReadySent) {
                  twoHandCameraReadySent = true;
                  setTwoHandCameraReady(true);
                }
              }

              const now = performance.now();
              const dt = lastGestureFrameAt > 0 ? Math.min(250, Math.max(1, now - lastGestureFrameAt)) : 16;
              lastGestureFrameAt = now;
              const v = videoRef.current;

              const browserSpeaking =
                typeof window !== 'undefined' &&
                'speechSynthesis' in window &&
                Boolean(window.speechSynthesis.speaking);
              const speechBlocked = isVerifiedModalSpeechBlocked({
                speechDone: isSpeechDoneRef.current,
                browserSpeaking,
              });

              if (speechBlocked) {
                // Strict Speech-First Relay Lock: freeze 4.0s hold timer at 0%, reset all hand ready states,
                // clear hand overlay dots, and NEVER advance until welcome voice completes 100%.
                gestureHoldMs = 0;
                lastAnyHandRaisedAt = -Infinity;
                latestAnyHandUp = false;
                if (lastReportedHolding) {
                  lastReportedHolding = false;
                  setTwoHandProgress(0);
                }
                if (lastReportedDetected) {
                  lastReportedDetected = false;
                  setTwoHandDetected(false);
                }
                setLeftHandUpState((prev) => (prev ? false : prev));
                setRightHandUpState((prev) => (prev ? false : prev));

                const zeroSecStr = `0.0 / ${(VERIFIED_HAND_HOLD_MS / 1000).toFixed(0)}.0 วิ (0%)`;
                if (confirmProgressFillRef.current) {
                  confirmProgressFillRef.current.style.transform = 'scaleX(0)';
                }
                if (confirmBtnFillRef.current) {
                  confirmBtnFillRef.current.style.transform = 'scaleX(0)';
                }
                if (
                  confirmProgressTextRef.current &&
                  confirmProgressTextRef.current.textContent !== zeroSecStr
                ) {
                  confirmProgressTextRef.current.textContent = zeroSecStr;
                }
                if (confirmCanvasRef.current) {
                  const cCtx = confirmCanvasRef.current.getContext('2d');
                  if (cCtx) {
                    cCtx.clearRect(0, 0, confirmCanvasRef.current.width, confirmCanvasRef.current.height);
                  }
                }

                // Lightly sense early hand raises (~8Hz) purely to warn user to wait for speech to finish
                if (
                  v.readyState >= 2 &&
                  v.videoWidth > 0 &&
                  (handLandmarker || Boolean(edgeTelemetryRef.current?.hand_status)) &&
                  now - lastGestureInferenceAt >= 120
                ) {
                  lastGestureInferenceAt = now;
                  const edgeHands = edgeTelemetryRef.current?.hand_status;
                  let detectedEarlyHands: NormalizedLandmark[][] = [];
                  if (handLandmarker) {
                    try {
                      if (!gestureOffscreenCanvas) {
                        gestureOffscreenCanvas = document.createElement('canvas');
                      }
                      const offW = 192;
                      const offH = Math.max(
                        108,
                        Math.round(offW * ((v.videoHeight || 480) / (v.videoWidth || 640))),
                      );
                      if (
                        gestureOffscreenCanvas.width !== offW ||
                        gestureOffscreenCanvas.height !== offH
                      ) {
                        gestureOffscreenCanvas.width = offW;
                        gestureOffscreenCanvas.height = offH;
                      }
                      const offCtx = gestureOffscreenCanvas.getContext('2d', { alpha: false });
                      if (offCtx) {
                        offCtx.drawImage(v, 0, 0, offW, offH);
                      }
                      const handTs = Math.max(lastDetectTs + 1, Math.round(now));
                      lastDetectTs = handTs;
                      const handRes = handLandmarker.detectForVideo(
                        offCtx ? gestureOffscreenCanvas : v,
                        handTs,
                      );
                      detectedEarlyHands = handRes.landmarks ?? [];
                    } catch {
                      // Ignore transient MediaPipe frame errors while locked for speech
                    }
                  }
                  const earlyStrict = evaluateStrictTwoHandsForStartModal(detectedEarlyHands, null);
                  const earlyRaised = Boolean(
                    edgeHands?.left_raised ||
                      edgeHands?.right_raised ||
                      earlyStrict.leftUp ||
                      earlyStrict.rightUp,
                  );
                  setRaisedDuringSpeech((prev) => (prev === earlyRaised ? prev : earlyRaised));
                }
                return;
              }

              if (
                v.readyState >= 2 &&
                v.videoWidth > 0 &&
                (handLandmarker || poseLandmarker || Boolean(edgeTelemetryRef.current?.hand_status))
              ) {
                const edgeHands = edgeTelemetryRef.current?.hand_status;
                const edgeLeftUp = Boolean(edgeHands?.left_raised);
                const edgeRightUp = Boolean(edgeHands?.right_raised);
                const edgeBothUpNow = Boolean(
                  edgeHands && (edgeHands.active_card_zone === 'BOTH' || (edgeLeftUp && edgeRightUp)),
                );
                let latestLeftUp = edgeLeftUp;
                let latestRightUp = edgeRightUp;

                // Throttle inference smoothly (~22Hz max, >=45ms apart) so 60-120fps animations never stutter
                if (
                  now - lastGestureInferenceAt >= 45 &&
                  (v.currentTime !== lastGestureVideoTime || now - lastGestureInferenceAt >= 80)
                ) {
                  lastGestureVideoTime = v.currentTime;
                  lastGestureInferenceAt = now;
                  if (!gestureOffscreenCanvas) {
                    gestureOffscreenCanvas = document.createElement('canvas');
                  }
                  const offW = 192;
                  const offH = Math.max(108, Math.round(offW * ((v.videoHeight || 480) / (v.videoWidth || 640))));
                  if (gestureOffscreenCanvas.width !== offW || gestureOffscreenCanvas.height !== offH) {
                    gestureOffscreenCanvas.width = offW;
                    gestureOffscreenCanvas.height = offH;
                  }
                  const offCtx = gestureOffscreenCanvas.getContext('2d', { alpha: false });
                  if (offCtx) {
                    offCtx.drawImage(v, 0, 0, offW, offH);
                  }
                  const inferenceSource = offCtx ? gestureOffscreenCanvas : v;

                  let detectedHands: NormalizedLandmark[][] = [];
                  let detectedPose: NormalizedLandmark[] | null = null;
                  if (handLandmarker) {
                    try {
                      const handTs = Math.max(lastDetectTs + 1, Math.round(now));
                      lastDetectTs = handTs;
                      const handRes = handLandmarker.detectForVideo(inferenceSource, handTs);
                      detectedHands = handRes.landmarks ?? [];
                    } catch {
                      // Keep gesture loop running across transient MediaPipe frame errors
                    }
                  }
                  // Run heavier PoseLandmarker only if HandLandmarker is unavailable OR as sparse 8Hz fallback when exactly 1 hand shows
                  if (
                    poseLandmarker &&
                    (!handLandmarker || (detectedHands.length === 1 && now - lastPoseFallbackAt >= 140))
                  ) {
                    lastPoseFallbackAt = now;
                    try {
                      const poseTs = Math.max(lastPoseDetectTs + 1, lastDetectTs + 1, Math.round(now));
                      lastPoseDetectTs = poseTs;
                      lastDetectTs = Math.max(lastDetectTs, poseTs);
                      const poseRes = poseLandmarker.detectForVideo(inferenceSource, poseTs);
                      detectedPose = poseRes.landmarks?.[0] ?? null;
                    } catch {
                      // Keep gesture loop running across transient MediaPipe frame errors
                    }
                  }

                  const strictHands = evaluateStrictTwoHandsForStartModal(detectedHands, detectedPose);
                  latestLeftUp = edgeLeftUp || strictHands.leftUp;
                  latestRightUp = edgeRightUp || strictHands.rightUp;
                  latestAnyHandUp = edgeBothUpNow || strictHands.bothUp;

                  setLeftHandUpState((prev) => (prev === latestLeftUp ? prev : latestLeftUp));
                  setRightHandUpState((prev) => (prev === latestRightUp ? prev : latestRightUp));

                  // Lightweight live hand indicator circles on preview canvas
                  const cCanvas = confirmCanvasRef.current;
                  if (cCanvas) {
                    const cw = cCanvas.clientWidth || 320;
                    const ch = cCanvas.clientHeight || 192;
                    if (cCanvas.width !== cw || cCanvas.height !== ch) {
                      cCanvas.width = cw;
                      cCanvas.height = ch;
                    }
                    const cCtx = cCanvas.getContext('2d');
                    if (cCtx) {
                      cCtx.clearRect(0, 0, cw, ch);
                      if (detectedHands.length > 0) {
                        const dotColor = latestAnyHandUp ? '#10b981' : '#f59e0b';
                        for (const h of detectedHands) {
                          const center = h[9] ?? h[0];
                          if (!center) continue;
                          const px = (1 - center.x) * cw;
                          const py = center.y * ch;
                          cCtx.beginPath();
                          cCtx.arc(px, py, 16, 0, Math.PI * 2);
                          cCtx.fillStyle = latestAnyHandUp ? 'rgba(16,185,129,0.28)' : 'rgba(245,158,11,0.28)';
                          cCtx.fill();
                          cCtx.lineWidth = 3;
                          cCtx.strokeStyle = dotColor;
                          cCtx.stroke();
                        }
                      }
                    }
                  }
                } else if (edgeBothUpNow) {
                  latestAnyHandUp = true;
                }

                const holdStep = stepVerifiedHandHold({
                  bothHandsNow: latestAnyHandUp,
                  speechBlocked,
                  nowMs: now,
                  dtMs: dt,
                  lastRaisedAtMs: lastAnyHandRaisedAt,
                  holdMs: gestureHoldMs,
                });
                lastAnyHandRaisedAt = holdStep.lastRaisedAtMs;
                gestureHoldMs = holdStep.holdMs;
                const nextProgress = holdStep.progress;

                // Zero-reflow 60-120Hz GPU compositor updates (scaleX + synchronized text labels, NO continuous React re-renders)
                const scaleCss = `scaleX(${nextProgress.toFixed(4)})`;
                const heldSecStr = (gestureHoldMs / 1000).toFixed(1);
                const remainSecStr = Math.max(0, (VERIFIED_HAND_HOLD_MS - gestureHoldMs) / 1000).toFixed(1);
                const pctInt = Math.round(nextProgress * 100);

                if (confirmProgressFillRef.current) {
                  confirmProgressFillRef.current.style.transform = scaleCss;
                }
                if (confirmBtnFillRef.current) {
                  confirmBtnFillRef.current.style.transform = scaleCss;
                }
                if (confirmProgressTextRef.current) {
                  confirmProgressTextRef.current.textContent = `${heldSecStr} / ${(VERIFIED_HAND_HOLD_MS / 1000).toFixed(0)}.0 วิ (${pctInt}%)`;
                }
                if (holdStep.detected && confirmPillLabelRef.current) {
                  confirmPillLabelRef.current.textContent =
                    lang === 'en'
                      ? `🙌 Both hands detected! Hold ${heldSecStr}/4.0s (${pctInt}%)`
                      : lang === 'cn'
                        ? `🙌 已检测到双手举起！保持 ${heldSecStr}/4.0 秒 (${pctInt}%)`
                        : `🙌 ยกครบ 2 ข้าง! ค้างไว้ ${heldSecStr}/4.0 วิ (${pctInt}%)`;
                }
                if (nextProgress > 0 && confirmBtnLabelRef.current) {
                  confirmBtnLabelRef.current.textContent =
                    lang === 'en'
                      ? `🙌 Holding BOTH hands... ${heldSecStr}/4.0s (${remainSecStr}s left)`
                      : lang === 'cn'
                        ? `🙌 正在加载确认... ${heldSecStr}/4.0 秒（剩余 ${remainSecStr} 秒）`
                        : `🙌 กำลังโหลด 2 ข้าง... ${heldSecStr}/4.0 วิ (เหลือ ${remainSecStr} วิ)`;
                }

                if (holdStep.detected !== lastReportedDetected) {
                  lastReportedDetected = holdStep.detected;
                  setTwoHandDetected(holdStep.detected);
                }
                const nowHolding = nextProgress > 0;
                if (nowHolding !== lastReportedHolding) {
                  lastReportedHolding = nowHolding;
                  setTwoHandProgress(nowHolding ? 0.01 : 0);
                }

                if (holdStep.confirmed && !gestureTriggered) {
                  gestureTriggered = true;
                  shouldContinueLoop = false;
                  active = false;
                  stopMediaAndDetectors();
                  onPatientAuthenticatedRef.current(verifiedPayload);
                  return;
                }
              }
              return;
            }

            if (videoRef.current.readyState >= 2 && !verifyingLockRef.current) {
              const now = performance.now();
              const currentVideoTime = videoRef.current.currentTime;
              // Skip duplicate 60Hz RAF ticks on identical webcam frames so microMotion stays real
              if (currentVideoTime === lastProcessedVideoTime && now - lastFaceDetectWallMs < 120) {
                return;
              }
              lastProcessedVideoTime = currentVideoTime;
              lastFaceDetectWallMs = now;

              const detectTs = Math.max(lastDetectTs + 1, Math.round(now));
              lastDetectTs = detectTs;
              if (!roiCanvasRef.current) {
                roiCanvasRef.current = document.createElement('canvas');
              }

              const res = landmarker.detectForVideo(videoRef.current, detectTs);
              let selection = selectPrimaryFace(res.faceLandmarks ?? []);
              let activeBlendshapes = res.faceBlendshapes?.[0]?.categories;

              // Stage-1 & Stage-2 Two-Stage Smart ROI Zoom: when no face found at full-frame or face < 22% height,
              // locate Head-Zone via PoseLandmarker/coarse box -> upscale 256x256 with contrast boost -> refine FaceLandmarker
              const coarseHeight = selection.primaryBox?.height ?? 0;
              if (roiLandmarker && (!selection.primaryFace || coarseHeight < 0.22)) {
                let targetRoi = selection.primaryFace && selection.primaryBox
                  ? computeHeadZoneRoiFromFaceBox(
                      {
                        x: Math.max(0, (selection.primaryCenter?.x ?? 0.5) - selection.primaryBox.width / 2),
                        y: Math.max(0, (selection.primaryCenter?.y ?? 0.5) - selection.primaryBox.height / 2),
                        width: selection.primaryBox.width,
                        height: selection.primaryBox.height,
                      },
                      2.25,
                    )
                  : null;

                if (!targetRoi && poseLandmarker) {
                  try {
                    const poseTs = Math.max(lastPoseDetectTs + 1, detectTs);
                    lastPoseDetectTs = poseTs;
                    const poseRes = poseLandmarker.detectForVideo(videoRef.current, poseTs);
                    const poseLms = poseRes.landmarks?.[0];
                    if (poseLms && poseLms.length >= 13) {
                      targetRoi = computeHeadZoneRoiFromPose(poseLms);
                    }
                  } catch {
                    // Pose fallback optional
                  }
                }
                if (!targetRoi) {
                  targetRoi = computeHeadZoneRoiFromPose(null);
                }

                const upscaledRoi = renderUpscaledRoi256(videoRef.current, targetRoi, roiCanvasRef.current, 256);
                if (upscaledRoi) {
                  const roiRes = roiLandmarker.detect(upscaledRoi.canvas);
                  const roiSelection = selectPrimaryFace(roiRes.faceLandmarks ?? []);
                  if (roiSelection.primaryFace && roiSelection.primaryFace.length >= 300) {
                    const remapped = remapRoiLandmarksToFrame(roiSelection.primaryFace, upscaledRoi.cropPx);
                    const remappedSelection = selectPrimaryFace([remapped]);
                    if (remappedSelection.primaryFace) {
                      selection = remappedSelection;
                      if (roiRes.faceBlendshapes?.[0]?.categories) {
                        activeBlendshapes = roiRes.faceBlendshapes[0].categories;
                      }
                    }
                  }
                }
              }

              const faceCount = selection.effectiveFaceCount;
              const landmarks = selection.primaryFace ? [...selection.primaryFace] : [];

              const liveness = evaluateLivenessAndPose(landmarks, prevLandmarks, faceCount, wasEyeClosed);
              const meanEar = (liveness.earLeft + liveness.earRight) / 2;
              wasEyeClosed = meanEar < 0.21;
              setLivenessInfo(liveness);

              const faceRatio =
                landmarks.length > 400 ? Math.abs(landmarks[454].x - landmarks[234].x) : 0;
              presenceRef.current.update({
                facePresent: faceCount === 1,
                faceSizeRatio: faceRatio,
                timestampMs: now,
              });

              if (faceCount !== 1 || landmarks.length < 300) {
                setHudBox(null);
                setHudLandmarks([]);
                unreadyFrameStreak += 1;
                if (unreadyFrameStreak >= 18) {
                  setLiveMatchHud(null);
                  setTurningHintPct(null);
                  temporalMatcherRef.current.reset();
                  loginCtrlRef.current.reset();
                  bioLivenessRef.current.reset();
                  lowSimBelow035StartAt = 0;
                  setBioReport(null);
                  burstStarted = false;
                  setBurstCount(0);
                  setScanStage('scanning');
                }
              }

              if (!alignCanvasRef.current) {
                alignCanvasRef.current = document.createElement('canvas');
              }
              if (!contextCanvasRef.current) {
                contextCanvasRef.current = document.createElement('canvas');
              }
              const crop =
                landmarks.length >= 300
                  ? alignFaceTo112(videoRef.current, landmarks, alignCanvasRef.current, roiCanvasRef.current)
                  : null;
              const contextFrame =
                landmarks.length >= 300
                  ? captureDownscaledContextFrame(videoRef.current, contextCanvasRef.current, 160)
                  : null;
              const rawBoxForBio =
                faceCount === 1 && landmarks.length >= 300
                  ? computeBoundingBoxFromLandmarks(landmarks)
                  : null;
              const bio =
                faceCount === 1 && landmarks.length >= 300
                  ? bioLivenessRef.current.update({
                      landmarks3D: landmarks,
                      blendshapes: activeBlendshapes,
                      earLeft: liveness.earLeft,
                      earRight: liveness.earRight,
                      yawDeg: liveness.yawDeg,
                      pitchDeg: liveness.pitchDeg,
                      frameWidth: videoRef.current.videoWidth || 1920,
                      frameHeight: videoRef.current.videoHeight || 1080,
                      faceCrop: crop,
                      contextFrame,
                      faceBox: rawBoxForBio,
                      timestampMs: now,
                    })
                  : null;
              if (bio) {
                setBioReport(bio);
              }

              const inspection = inspectBiometricFrame3D({
                landmarks3D: landmarks,
                prevLandmarks3D: prevLandmarks,
                faceCount,
                wasEyeClosed,
                image: crop,
                contextImage: contextFrame,
                faceBox: rawBoxForBio,
                biologicalVerified: bio?.biologicalVerified,
                allowLongRange: true,
                requestedPose: 'CENTER',
                metrics: bio
                  ? {
                      allowLongRange: true,
                      biologicalVerified: bio.biologicalVerified,
                      planarRigidSpoof: bio.planarSpoofDetected,
                      moireOrBezelDetected: bio.screenSpoofDetected,
                    }
                  : { allowLongRange: true },
              });
              prevLandmarks = landmarks;
              setInspectionInfo(inspection);
              // Single source of truth: immediate state clear removes spoof whenever biologicalVerified passes
              const isFrameSpoof = bio
                ? bio.spoofDetected && !bio.biologicalVerified
                : inspection.scan_status === 'SPOOF_DETECTED';

              // Real-time 1:N 3D Multi-Angle Recognition & Fine Landmark HUD Tracking
              if (faceCount === 1 && landmarks.length >= 300 && crop && bio) {
                const rawBox = rawBoxForBio ?? computeBoundingBoxFromLandmarks(landmarks);
                if (rawBox && videoRef.current) {
                  const vw = videoRef.current.videoWidth || 1280;
                  const vh = videoRef.current.videoHeight || 720;
                  const cw = videoRef.current.clientWidth || vw;
                  const ch = videoRef.current.clientHeight || vh;
                  const scale = Math.max(cw / vw, ch / vh);
                  const renderedW = vw * scale;
                  const renderedH = vh * scale;
                  const offsetX = (renderedW - cw) / 2;
                  const offsetY = (renderedH - ch) / 2;

                  const leftNorm = mirroredRef.current ? 1 - rawBox.x - rawBox.width : rawBox.x;
                  const pxLeft = leftNorm * renderedW - offsetX;
                  const pxTop = rawBox.y * renderedH - offsetY;
                  const pxW = rawBox.width * renderedW;
                  const pxH = rawBox.height * renderedH;

                  const xPct = Math.max(2, Math.min(92, (pxLeft / cw) * 100));
                  const yPct = Math.max(2, Math.min(92, (pxTop / ch) * 100));
                  const wPct = Math.max(8, Math.min(96 - xPct, (pxW / cw) * 100));
                  const hPct = Math.max(8, Math.min(96 - yPct, (pxH / ch) * 100));

                  setHudBox({ xPct, yPct, wPct, hPct });

                  const mappedPoints = KEY_MESH_LANDMARK_INDICES.flatMap((idx) => {
                    const pt = landmarks[idx];
                    if (!pt) return [];
                    const nx = mirroredRef.current ? 1 - pt.x : pt.x;
                    const lx = ((nx * renderedW - offsetX) / cw) * 100;
                    const ly = ((pt.y * renderedH - offsetY) / ch) * 100;
                    if (lx < 0 || lx > 100 || ly < 0 || ly > 100) return [];
                    return [{ xPct: lx, yPct: ly }];
                  });
                  setHudLandmarks(mappedPoints);
                }

                const faceReadyForLogin =
                  !isFrameSpoof && isFullFrontalFaceReadyForLogin(landmarks, liveness, inspection);
                if (!faceReadyForLogin) {
                  unreadyFrameStreak += 1;
                  if ((isFrameSpoof || unreadyFrameStreak >= 18) && !verifyingLockRef.current) {
                    temporalMatcherRef.current.reset();
                    loginCtrlRef.current.reset();
                    lowSimBelow035StartAt = 0;
                    burstStarted = false;
                    setBurstCount(0);
                    setLiveMatchHud(null);
                    setTurningHintPct(null);
                    setScanStage('scanning');
                  }
                } else {
                  unreadyFrameStreak = 0;
                  if (now - lastEmbedCheckAt >= 120 && !verifyingLockRef.current) {
                    lastEmbedCheckAt = now;
                    const { vector, mirroredVector, engine } = await extractDualFlipFaceEmbedding(
                      crop,
                      landmarks,
                    );
                    if (!active) {
                      shouldContinueLoop = false;
                      return;
                    }

                    const rollDeg = computeEyeRollAngleDeg(
                      landmarks,
                      videoRef.current?.videoWidth || 1280,
                      videoRef.current?.videoHeight || 720,
                    );
                    const currentFaceHeightRatio = rawBox?.height ?? bio.faceScale;
                    const livenessEvidence = {
                      biologicalVerified: bio.biologicalVerified,
                      blinkCount: bio.blinkCount,
                      spoofRisk: bio.spoofScore,
                      nonRigidScore: bio.nonRigidScore,
                      rppgScore: bio.rppgScore,
                      faceHeightRatio: currentFaceHeightRatio,
                      yawDeg: liveness.yawDeg,
                      rollDeg,
                      engine,
                      mirroredProbeVector: mirroredVector,
                    };

                    let latestTemporalSim = 0;
                    if (!verifyingLockRef.current && biometricIndexRef.current.length > 0) {
                      const temporal = temporalMatcherRef.current.evaluateFrame(
                        {
                          probeVector: vector,
                          mirroredProbeVector: mirroredVector,
                          yawDeg: liveness.yawDeg,
                          pitchDeg: liveness.pitchDeg,
                          rollDeg,
                          depthScore: liveness.depthScore,
                          microMotion: liveness.microMotion,
                          blinkDetected:
                            liveness.blinkDetected ||
                            bio.blinkVerified ||
                            bio.headPoseVerified ||
                            bio.lipMotionVerified ||
                            bio.longRangeMotionVerified,
                          nonRigidMotion: bio.nonRigidMotion,
                          biologicalVerified: bio.biologicalVerified,
                          spoofDetected: isFrameSpoof,
                          rppgScore: bio.rppgScore,
                          faceHeightRatio: currentFaceHeightRatio,
                          engine,
                        },
                        biometricIndexRef.current,
                      );
                      latestTemporalSim = temporal.bestRawSimilarity;
                      const isHudMatch = Boolean(temporal.candidate && temporal.streakCount > 0);
                      setLiveMatchHud(
                        isHudMatch && temporal.candidate
                          ? {
                              candidate: temporal.candidate,
                              confidencePercent: temporal.confidencePercent,
                              bestAngle: temporal.bestAngle,
                              streak: temporal.streakCount,
                              confirmed: temporal.confirmed,
                            }
                          : null,
                      );

                      if (latestTemporalSim >= 0.40 && !temporal.confirmed) {
                        lowSimBelow035StartAt = 0;
                        setTurningHintPct(Math.round(latestTemporalSim * 100));
                      } else if (temporal.confirmed) {
                        lowSimBelow035StartAt = 0;
                        setTurningHintPct(null);
                      } else if (latestTemporalSim >= 0.35) {
                        lowSimBelow035StartAt = 0;
                        setTurningHintPct(null);
                      } else if (bio.biologicalVerified) {
                        if (lowSimBelow035StartAt === 0) lowSimBelow035StartAt = now;
                        setTurningHintPct(null);
                      }

                      if (
                        temporal.confirmed &&
                        temporal.candidate &&
                        bio.biologicalVerified &&
                        !isFrameSpoof &&
                        !verifyingLockRef.current
                      ) {
                        verifyingLockRef.current = true;
                        setScanStage('verifying');
                        const matchedConf = Math.round(temporal.confidencePercent);
                        let didAuthenticate = false;
                        let blockedBySpoof = false;
                        try {
                          const verifyRes = await apiClient.verifyFaceLogin(vector, livenessEvidence);
                          if (!active) {
                            shouldContinueLoop = false;
                            return;
                          }
                          setLastVerifyResult(verifyRes);
                          if (verifyRes.failureReasons?.includes('BIOLOGICAL_SPOOF_BLOCKED')) {
                            blockedBySpoof = true;
                          } else if (
                            verifyRes.matched &&
                            verifyRes.patient &&
                            verifyRes.confidencePercent >= 80
                          ) {
                            didAuthenticate = true;
                            const matchedEntry =
                              biometricIndexRef.current.find((e) => e.patientId === verifyRes.patient?.id) ??
                              temporal.candidate;
                            const nextPayload = buildKioskLoginSuccessPayload({
                              patient: verifyRes.patient,
                              todayRoutine: verifyRes.todayRoutine ?? null,
                              confidencePercent: Math.max(verifyRes.confidencePercent, matchedConf),
                              rehabSchedules: verifyRes.rehabSchedules,
                              serverTodayExercises: verifyRes.todayExercises,
                              serverCompletedToday: verifyRes.completedToday,
                              indexEntry: matchedEntry,
                            });
                            openVerifiedModal(nextPayload);
                            return;
                          }
                        } catch {
                          if (!active) {
                            shouldContinueLoop = false;
                            return;
                          }
                        } finally {
                          if (!didAuthenticate) {
                            verifyingLockRef.current = false;
                          }
                        }

                        // Accept confirmed multi-angle client match if server DB replication lags cleanly (e.g. just enrolled)
                        if (
                          !didAuthenticate &&
                          !blockedBySpoof &&
                          bio.biologicalVerified &&
                          !isFrameSpoof &&
                          temporal.confidencePercent >= 85 &&
                          temporal.streakCount >= 3
                        ) {
                          didAuthenticate = true;
                          const cand = temporal.candidate;
                          const nextPayload = buildKioskLoginSuccessPayload({
                            patient: {
                              id: cand.patientId,
                              hn: cand.hn,
                              fullName: cand.fullName,
                              nextAppointmentDate: cand.nextAppointmentDate,
                            },
                            todayRoutine: cand.todayRoutine ?? null,
                            confidencePercent: matchedConf,
                            rehabSchedules: cand.rehabSchedules,
                            serverTodayExercises: cand.todayExercises,
                            serverCompletedToday: cand.completedToday,
                            indexEntry: cand,
                          });
                          openVerifiedModal(nextPayload);
                          return;
                        }

                        temporalMatcherRef.current.reset();
                        setLiveMatchHud(null);
                        void apiClient
                          .getBiometricIndex()
                          .then((entries) => {
                            if (active) {
                              biometricIndexRef.current = entries;
                              setDbAccountCount(entries.length);
                            }
                          })
                          .catch(() => {});
                      }
                    }

                    if (!verifyingLockRef.current) {
                      if (!bio.biologicalVerified || isFrameSpoof) {
                        // Hold cleanly on biological blink challenge; NEVER advance authentication on a static image/phone
                        loginCtrlRef.current.reset();
                        lowSimBelow035StartAt = 0;
                        setTurningHintPct(null);
                        burstStarted = false;
                        setBurstCount(0);
                        setScanStage('scanning');
                      } else {
                        if (!burstStarted) {
                          loginCtrlRef.current.startCountdownComplete();
                          burstStarted = true;
                        }
                        setScanStage('burst');
                        const push = loginCtrlRef.current.pushFrame({
                          vector,
                          qualityScore: 0.95,
                          capturedAfterCountdown: true,
                          pose: 'front',
                          sharpness: 85,
                          yaw: liveness.yawDeg,
                          pitch: liveness.pitchDeg,
                        });
                        setBurstCount(push.collected);

                        if (push.readyToVerify) {
                          verifyingLockRef.current = true;
                          setScanStage('verifying');
                          const probe = loginCtrlRef.current.buildProbeCentroid();
                          let didFinishRound = false;
                          try {
                            const verifyRes = await apiClient.verifyFaceLogin(probe, livenessEvidence);
                            if (!active) {
                              shouldContinueLoop = false;
                              return;
                            }
                            setLastVerifyResult(verifyRes);
                            if (verifyRes.matched && verifyRes.patient && verifyRes.confidencePercent >= 80) {
                              const verifiedPatientId = verifyRes.patient.id;
                              const matchedEntry = biometricIndexRef.current.find(
                                (e) => e.patientId === verifiedPatientId,
                              );
                              didFinishRound = true;
                              const nextPayload = buildKioskLoginSuccessPayload({
                                patient: verifyRes.patient,
                                todayRoutine: verifyRes.todayRoutine ?? null,
                                confidencePercent: verifyRes.confidencePercent,
                                rehabSchedules: verifyRes.rehabSchedules,
                                serverTodayExercises: verifyRes.todayExercises,
                                serverCompletedToday: verifyRes.completedToday,
                                indexEntry: matchedEntry,
                              });
                              openVerifiedModal(nextPayload);
                              return;
                            }
                          } catch {
                            if (!active) {
                              shouldContinueLoop = false;
                              return;
                            }
                          } finally {
                            if (!didFinishRound) {
                              verifyingLockRef.current = false;
                            }
                          }
                          loginCtrlRef.current.recordUnmatchedRound();
                          burstStarted = false;
                          setBurstCount(0);

                          // Only show "Account Not Found" when DB is genuinely empty OR similarity stays < 0.35 for >= 8s continuous scanning
                          const noPatientsInDb = biometricIndexRef.current.length === 0;
                          const lowSimElapsed =
                            lowSimBelow035StartAt > 0 ? now - lowSimBelow035StartAt : 0;
                          if ((noPatientsInDb && lowSimElapsed >= 3000) || lowSimElapsed >= 8000) {
                            shouldContinueLoop = false;
                            setScanStage('not_found');
                            return;
                          }
                          setScanStage('burst');
                        }
                      }
                    }
                  }
                }
              }
            }
          } catch {
            // Swallow transient MediaPipe / async frame errors and keep scanning
            verifyingLockRef.current = false;
          } finally {
            if (active && shouldContinueLoop) {
              rafId = requestAnimationFrame(loop);
            }
          }
        };

        rafId = requestAnimationFrame(loop);
      } catch (err) {
        if (!active) return;
        setCameraError(err instanceof Error ? err.message : 'ไม่สามารถเปิดกล้องได้');
        setCameraPaused(true);
      }
    }

    startFacePipeline();

    return () => {
      active = false;
      stopMediaAndDetectors();
    };
  }, [apiClient, cameraPaused, isNotFoundStage, scanGeneration, showPinModal, staffModalOpen]);

  const handleCallStaff = async () => {
    try {
      await apiClient.createStaffCall(kioskId, 'ผู้ป่วยกดเรียกเจ้าหน้าที่จากหน้าตู้ Kiosk');
      showToast(c.staffCalledToast);
    } catch {
      showToast(c.staffCalledToast);
    }
  };

  const handlePinSubmit = async (digits: string) => {
    setPinSubmitting(true);
    setPinError(null);
    try {
      const patients: PatientRecord[] = await apiClient.listPatients();
      const normalized = digits.trim();
      const matched = patients.find((p) => {
        const cleanPhone = p.phone.replace(/\D/g, '');
        const cleanHn = p.hn.replace(/\D/g, '');
        return (
          cleanPhone.endsWith(normalized) ||
          cleanHn.endsWith(normalized) ||
          p.hn.toLowerCase() === normalized.toLowerCase()
        );
      });
      if (matched) {
        const matchedEntry = biometricIndexRef.current.find((e) => e.patientId === matched.id);
        const pinPayload = buildKioskLoginSuccessPayload({
          patient: {
            id: matched.id,
            hn: matched.hn,
            fullName: matched.fullName,
            nextAppointmentDate: matched.nextAppointmentDate,
          },
          todayRoutine: matchedEntry?.todayRoutine ?? null,
          confidencePercent: 100,
          rehabSchedules: matched.rehabSchedules,
          indexEntry: matchedEntry,
        });
        setShowPinModal(false);
        setCameraPaused(false);
        setPinDigits('');
        openVerifiedModal(pinPayload);
        return;
      }
      setPinError(c.pinNotFoundError);
    } catch {
      setPinError(c.pinAuthError);
    } finally {
      setPinSubmitting(false);
    }
  };

  const pressPinKey = (digit: string) => {
    if (pinSubmitting || pinDigits.length >= 4) return;
    const next = pinDigits + digit;
    setPinDigits(next);
    setPinError(null);
    if (next.length === 4) {
      window.setTimeout(() => {
        void handlePinSubmit(next);
      }, 180);
    }
  };

  const isBioVerified = Boolean(bioReport?.biologicalVerified);
  const isSpoof = !isBioVerified && (Boolean(bioReport?.spoofDetected) || inspectionInfo?.scan_status === 'SPOOF_DETECTED');
  const isFaceDetected = Boolean(hudBox);

  const bottomPillText =
    scanStage === 'verifying'
      ? c.verifying
      : c.alignFace;

  return (
    <div className="min-h-screen flex flex-col bg-gradient-to-br from-[#EAF8F5] via-[#EFF6FF] to-white text-[#0F4C81] font-sans select-none overflow-x-hidden">
      {/* Top Clinical Blue-Teal Header Bar — Balanced, Uncluttered 2-Tier Header */}
      <header className="w-full px-5 py-3 bg-gradient-to-r from-[#0F4C81] via-[#0284C7] to-[#0D9488] border-b border-sky-200/30 shadow-md text-white z-40">
        <div className="max-w-6xl mx-auto w-full flex flex-col gap-2.5">
          {/* Row 1: Brand Identity (Left) + Cloud/Clock & Language Switcher (Right) */}
          <div className="flex items-center justify-between gap-3 w-full">
            {/* Left Brand Identity */}
            <div className="flex items-center gap-3 min-w-0">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-[#0EA5E9] to-[#10B981] flex items-center justify-center text-white ring-1 ring-white/40 shadow-sm shrink-0">
                <HeartPulse className="w-5 h-5" />
              </div>
              <div className="leading-snug min-w-0">
                <div className="flex items-center gap-2">
                  <span className="text-lg sm:text-xl font-extrabold tracking-tight text-white whitespace-nowrap">
                    Dr.Tech.Care
                  </span>
                  <span className="text-xs font-semibold px-2.5 py-0.5 rounded-full bg-white/20 border border-white/30 text-white whitespace-nowrap">
                    AI Kiosk
                  </span>
                </div>
                <p className="text-xs sm:text-[13px] text-sky-100 truncate">
                  {c.subtitle} • Terminal ID: {kioskId}
                </p>
              </div>
            </div>

            {/* Right Controls: Cloud Online + Clock & Compact Language Switcher */}
            <div className="flex items-center gap-2.5 shrink-0">
              {/* Cloud Online + Clock Pill */}
              <div className="hidden sm:inline-flex h-9 px-3 items-center gap-2 rounded-xl bg-white/15 border border-white/25 text-xs font-semibold text-white whitespace-nowrap">
                <span className="w-2 h-2 rounded-full bg-emerald-300 animate-pulse shrink-0 shadow-[0_0_8px_#34d399]" />
                <span>{edgeConnected ? 'Cloud + Edge 3D' : 'Cloud Online'}</span>
                <span className="text-white/60">•</span>
                <span className="font-mono tabular-nums font-bold text-white">{clockText}</span>
              </div>

              {/* Compact Language Switcher */}
              <div
                className="flex items-center gap-1 rounded-xl bg-white/15 border border-white/25 p-1 shrink-0"
                data-testid="language-switcher"
              >
                {LANG_OPTIONS.map((opt) => {
                  const active = lang === opt.code;
                  return (
                    <button
                      key={opt.code}
                      type="button"
                      onClick={() => handleSelectLang(opt.code)}
                      className={`h-7 w-[72px] inline-flex items-center justify-center gap-1 whitespace-nowrap rounded-lg text-xs font-bold transition-all cursor-pointer ${
                        active
                          ? 'bg-white text-[#0F4C81] shadow-sm'
                          : 'text-white/85 hover:bg-white/20 hover:text-white'
                      }`}
                    >
                      <span className="opacity-85">{opt.prefix}</span>
                      <span>{opt.label}</span>
                    </button>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Row 2: 3 Equal-Width Calm Action Buttons */}
          <div className="grid grid-cols-3 gap-2.5 w-full pt-2 border-t border-white/20">
            {/* Button 1: Register New Patient */}
            <button
              type="button"
              id="header-open-new-patient-btn"
              data-testid="header-open-new-patient-btn"
              onClick={onOpenStaffRegister}
              className="h-10 w-full inline-flex items-center justify-center gap-2 px-3 rounded-xl bg-white hover:bg-emerald-50 text-[#0F4C81] text-xs sm:text-sm font-bold whitespace-nowrap active:scale-[0.99] transition shadow-sm cursor-pointer"
            >
              <UserPlus className="w-4 h-4 shrink-0 text-emerald-600" />
              <span>{c.registerNewPatient}</span>
            </button>

            {/* Button 2: Call Staff */}
            <button
              type="button"
              id="header-call-staff-btn"
              data-testid="header-call-staff-btn"
              onClick={handleCallStaff}
              className="h-10 w-full inline-flex items-center justify-center gap-2 px-3 rounded-xl bg-white/15 hover:bg-white/25 border border-white/30 text-white text-xs sm:text-sm font-semibold whitespace-nowrap active:scale-[0.99] transition cursor-pointer"
            >
              <Bell className="w-4 h-4 shrink-0 text-amber-200" />
              <span>{c.callStaff}</span>
            </button>

            {/* Button 3: Staff Login */}
            <button
              type="button"
              id="header-open-staff-login-btn"
              data-testid="open-staff-login-btn"
              onClick={onOpenStaffLogin}
              className="h-10 w-full inline-flex items-center justify-center gap-2 px-3 rounded-xl bg-emerald-500 hover:bg-emerald-400 border border-emerald-300/60 text-white text-xs sm:text-sm font-bold whitespace-nowrap active:scale-[0.99] transition cursor-pointer shadow-xs"
            >
              <KeyRound className="w-4 h-4 text-white shrink-0" />
              <span>{c.staffLogin}</span>
            </button>
          </div>
        </div>
      </header>

      {/* Main Content Stage */}
      <main className="flex-1 w-full max-w-4xl mx-auto my-3 px-2 sm:px-3 flex flex-col relative min-h-0">
        {/* Subtle outer cyan corner brackets */}
        <div className="pointer-events-none absolute top-2 left-2 w-5 h-5 border-t-2 border-l-2 border-cyan-400/45 rounded-tl-md" />
        <div className="pointer-events-none absolute top-2 right-2 w-5 h-5 border-t-2 border-r-2 border-cyan-400/45 rounded-tr-md" />
        <div className="pointer-events-none absolute bottom-2 left-2 w-5 h-5 border-b-2 border-l-2 border-cyan-400/45 rounded-bl-md" />
        <div className="pointer-events-none absolute bottom-2 right-2 w-5 h-5 border-b-2 border-r-2 border-cyan-400/45 rounded-br-md" />

        {toastMsg && (
          <div className="mb-3 w-full rounded-2xl border border-amber-300 bg-amber-50 px-4 py-2.5 text-xs sm:text-sm font-bold text-amber-900 shadow-md flex items-center justify-between">
            <span>{toastMsg}</span>
            <button type="button" onClick={() => setToastMsg(null)} className="text-amber-700 hover:text-amber-950">
              <X className="w-4 h-4" />
            </button>
          </div>
        )}

        {/* White Center Kiosk Card */}
        <div className="w-full flex-1 flex flex-col justify-between bg-white rounded-[28px] border border-slate-200/90 shadow-2xl shadow-emerald-950/10 p-2 sm:p-3 relative z-10 min-h-0">
          {/* Light Green-Blue Camera Viewport */}
          <div className="relative flex-1 w-full h-full min-h-[420px] sm:min-h-[540px] rounded-2xl bg-gradient-to-br from-sky-100 via-emerald-50 to-white overflow-hidden border border-sky-200">
            {!cameraPaused && !staffModalOpen ? (
              <>
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  style={
                    rotationDeg
                      ? {
                          transform: `rotate(${rotationDeg}deg) ${mirrored ? 'scaleX(-1)' : ''}`,
                        }
                      : undefined
                  }
                  className={`absolute inset-0 w-full h-full object-cover ${
                    mirrored ? 'scale-x-[-1]' : ''
                  } transition-transform duration-300`}
                />

                {/* 4 Inner Cyan Corner Brackets */}
                <div className="pointer-events-none absolute top-4 left-4 w-6 h-6 border-t-2 border-l-2 border-cyan-400 rounded-tl-sm z-10" />
                <div className="pointer-events-none absolute top-4 right-4 w-6 h-6 border-t-2 border-r-2 border-cyan-400 rounded-tr-sm z-10" />
                <div className="pointer-events-none absolute bottom-4 left-4 w-6 h-6 border-b-2 border-l-2 border-cyan-400 rounded-bl-sm z-10" />
                <div className="pointer-events-none absolute bottom-4 right-4 w-6 h-6 border-b-2 border-r-2 border-cyan-400 rounded-br-sm z-10" />

                {/* Top-Left Camera Controls */}
                <div className="absolute top-3 left-3 z-20 flex items-center gap-1.5">
                  <button
                    type="button"
                    onClick={() => setRotationDeg((prev) => (((prev + 90) % 360) as 0 | 90 | 180 | 270))}
                    className="px-2.5 py-1 rounded-lg bg-white/90 hover:bg-white border border-sky-200 text-[#0F4C81] text-[11px] font-bold flex items-center gap-1 backdrop-blur-sm shadow-xs cursor-pointer"
                    title="หมุนภาพกล้อง"
                  >
                    <RotateCw className="w-3 h-3 text-[#0284C7]" />
                    <span>{rotationDeg}°</span>
                  </button>
                  <button
                    type="button"
                    onClick={() => setMirrored((m) => !m)}
                    className={`p-1.5 rounded-lg border transition cursor-pointer shadow-xs ${
                      mirrored
                        ? 'bg-emerald-600/90 border-emerald-400 text-white'
                        : 'bg-white/90 border-sky-200 text-[#0F4C81]'
                    }`}
                    title="สลับกระจกซ้าย-ขวา"
                  >
                    <FlipHorizontal2 className="w-3.5 h-3.5" />
                  </button>
                  <button
                    type="button"
                    onClick={() => setShowCalibHud((v) => !v)}
                    className={`px-2.5 py-1 rounded-lg border text-[11px] font-bold flex items-center gap-1 backdrop-blur-sm transition cursor-pointer shadow-xs ${
                      showCalibHud
                        ? 'bg-[#0284C7] border-sky-300 text-white'
                        : 'bg-white/90 hover:bg-white border-sky-200 text-[#0F4C81]'
                    }`}
                    title="โหมดสอบเทียบชีวมาตร (Calibration HUD)"
                  >
                    <Sparkles className={`w-3 h-3 ${showCalibHud ? 'text-white' : 'text-[#0284C7]'}`} />
                    <span>Calib</span>
                  </button>
                </div>

                {/* Top-Right Close/Pause Camera */}
                <button
                  type="button"
                  onClick={() => setCameraPaused(true)}
                  className="absolute top-3 right-3 z-20 w-7 h-7 rounded-full bg-white/90 hover:bg-rose-600 border border-sky-200 text-[#0F4C81] hover:text-white flex items-center justify-center transition cursor-pointer shadow-xs"
                  title="หยุดสแกน / ปิดกล้อง"
                >
                  <X className="w-3.5 h-3.5" />
                </button>

                {/* Optional Calibration Telemetry HUD */}
                {showCalibHud && (
                  <div className="absolute top-12 left-3 right-3 sm:right-auto sm:w-80 z-30 rounded-xl bg-white/95 border border-sky-300 p-3 text-[11px] font-mono text-[#0F4C81] shadow-2xl space-y-1">
                    <div className="flex items-center justify-between border-b border-sky-200 pb-1">
                      <span className="font-bold text-[#0F4C81]">Staff Biometric HUD</span>
                      <button
                        type="button"
                        onClick={() => setShowCalibHud(false)}
                        className="text-slate-500 hover:text-[#0F4C81]"
                      >
                        ✕
                      </button>
                    </div>
                    <div className="text-[10px] text-[#0284C7] truncate">Host: {supabaseHost}</div>
                    <div className="grid grid-cols-3 gap-1 pt-1">
                      <div>3D Depth: {livenessInfo?.depthScore.toFixed(2) ?? '0.00'}</div>
                      <div>EAR: {livenessInfo?.earLeft.toFixed(2) ?? '0.00'}</div>
                      <div>BaseEAR: {bioReport?.openEyeBaseline.toFixed(2) ?? '0.00'}</div>
                      <div>Scale: {bioReport?.faceScale.toFixed(2) ?? '0.00'}</div>
                      <div>Blink: {bioReport?.blinkCount ?? 0}</div>
                      <div>YawΔ: {bioReport?.deltaYawDeg.toFixed(1) ?? '0.0'}°</div>
                      <div>Spoof: {Math.round((bioReport?.spoofScore ?? 0) * 100)}%</div>
                      <div>rPPG: {Math.round((bioReport?.rppgScore ?? 0) * 100)}%</div>
                      <div>Burst: {burstCount}/{BURST_FRAME_COUNT}</div>
                    </div>
                    {lastVerifyResult?.topCandidate && (
                      <div className="text-[10px] text-emerald-300 pt-1 border-t border-slate-800">
                        Top: {lastVerifyResult.topCandidate.patientName ?? lastVerifyResult.topCandidate.patientId} (
                        {lastVerifyResult.confidencePercent}%)
                      </div>
                    )}
                  </div>
                )}

                {/* Fine-Grained Facial Landmark Mesh Keypoints (478-point contour subset) */}
                {hudLandmarks.length > 0 && (
                  <div className="pointer-events-none absolute inset-0 z-20 overflow-hidden">
                    {hudLandmarks.map((pt, idx) => (
                      <span
                        key={idx}
                        style={{ left: `${pt.xPct}%`, top: `${pt.yPct}%` }}
                        className={`absolute h-1.5 w-1.5 -translate-x-1/2 -translate-y-1/2 rounded-full ${
                          isSpoof
                            ? 'bg-rose-400 shadow-[0_0_6px_#fb7185]'
                            : 'bg-[#10B981] shadow-[0_0_6px_#10B981]'
                        }`}
                      />
                    ))}
                  </div>
                )}

                {/* Full-Frame Dynamic Long-Range Face Target Lock (#10B981) & Single Primary Banner */}
                {hudBox && (
                  <div
                    className="pointer-events-none absolute z-25 transition-all duration-100"
                    style={{
                      left: `${hudBox.xPct}%`,
                      top: `${hudBox.yPct}%`,
                      width: `${hudBox.wPct}%`,
                      height: `${hudBox.hPct}%`,
                    }}
                  >
                    <div
                      className={`relative h-full w-full overflow-hidden rounded-2xl border-[3px] transition-colors ${
                        isSpoof
                          ? 'border-rose-500 shadow-[0_0_28px_rgba(244,63,94,0.85)]'
                          : 'border-[#10B981] shadow-[0_0_28px_rgba(16,185,129,0.78)]'
                      }`}
                    >
                      <span className={`absolute top-0 left-0 h-4 w-4 rounded-tl-lg border-t-[3px] border-l-[3px] ${isSpoof ? 'border-rose-400' : 'border-[#10B981]'}`} />
                      <span className={`absolute top-0 right-0 h-4 w-4 rounded-tr-lg border-t-[3px] border-r-[3px] ${isSpoof ? 'border-rose-400' : 'border-[#10B981]'}`} />
                      <span className={`absolute bottom-0 left-0 h-4 w-4 rounded-bl-lg border-b-[3px] border-l-[3px] ${isSpoof ? 'border-rose-400' : 'border-[#10B981]'}`} />
                      <span className={`absolute bottom-0 right-0 h-4 w-4 rounded-br-lg border-b-[3px] border-r-[3px] ${isSpoof ? 'border-rose-400' : 'border-[#10B981]'}`} />
                      {!isBioVerified && (
                        <div className={`absolute inset-x-2 top-1/2 h-[2px] -translate-y-1/2 bg-gradient-to-r from-transparent ${isSpoof ? 'via-rose-400 shadow-[0_0_12px_#fb7185]' : 'via-[#10B981] shadow-[0_0_12px_#10B981]'} to-transparent`} />
                      )}
                    </div>

                    {isSpoof ? (
                      <div className="absolute left-1/2 top-full mt-2.5 w-max max-w-[360px] -translate-x-1/2 rounded-2xl border-2 border-rose-400 bg-rose-50/95 px-4 py-2.5 text-center backdrop-blur-md shadow-2xl">
                        <div className="text-sm font-black text-rose-700 leading-snug">{c.spoofBlocked}</div>
                      </div>
                    ) : !isBioVerified ? (
                      <div className="absolute left-1/2 top-full mt-2.5 w-max max-w-[390px] -translate-x-1/2 rounded-2xl border-2 border-[#10B981] bg-white/95 px-4 py-2.5 text-center backdrop-blur-md shadow-xl">
                        <div className="text-sm font-black text-[#0F4C81] leading-snug">
                          {bioReport?.livenessMode === 'LONG_RANGE' ? c.longRangePrompt : c.checkingLiveness}
                        </div>
                        <div className="mt-0.5 text-xs font-bold text-[#059669]">
                          {c.distancePrefix}: {(bioReport?.estimatedDistanceM ?? 2.2).toFixed(1)} {c.distanceUnit} • {c.verifying}
                        </div>
                      </div>
                    ) : liveMatchHud?.candidate ? (
                      <div className="absolute left-1/2 top-full mt-2.5 w-max max-w-[360px] -translate-x-1/2 rounded-2xl border-2 border-[#10B981] bg-emerald-50/95 px-4 py-2.5 text-center backdrop-blur-md shadow-2xl">
                        <div className="text-sm font-black text-emerald-800">
                          {c.liveMatchVerified}: {liveMatchHud.candidate.fullName} ({liveMatchHud.candidate.hn})
                        </div>
                        <div className="mt-0.5 text-xs font-bold text-[#0F4C81]">
                          {c.distancePrefix}: {(bioReport?.estimatedDistanceM ?? 1.5).toFixed(1)} {c.distanceUnit} • {c.liveMatchAccuracy}: {liveMatchHud.confidencePercent.toFixed(1)}% ({liveMatchHud.streak}/3 {c.framesUnit})
                        </div>
                      </div>
                    ) : turningHintPct !== null ? (
                      <div className="absolute left-1/2 top-full mt-2.5 w-max max-w-[390px] -translate-x-1/2 rounded-2xl border-2 border-amber-400 bg-amber-50/95 px-4 py-2 text-center backdrop-blur-md shadow-xl">
                        <div className="text-sm font-black text-amber-800">
                          {c.turnHeadPrefix} {turningHintPct}{c.turnHeadSuffix}
                        </div>
                        <div className="mt-0.5 text-[11px] font-semibold text-[#0284C7]">
                          {c.liveVerified} ({burstCount}/{BURST_FRAME_COUNT})
                        </div>
                      </div>
                    ) : (
                      <div className="absolute left-1/2 top-full mt-2.5 w-max max-w-[360px] -translate-x-1/2 rounded-2xl border-2 border-[#10B981] bg-emerald-50/95 px-4 py-2 text-center backdrop-blur-md shadow-xl">
                        <div className="text-sm font-black text-emerald-800">
                          {c.distancePrefix}: {(bioReport?.estimatedDistanceM ?? 1.8).toFixed(1)} {c.distanceUnit} • {c.verifying} ({burstCount}/{BURST_FRAME_COUNT})
                        </div>
                        <div className="mt-0.5 text-[11px] font-semibold text-[#0284C7]">
                          {c.liveVerified}
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {/* Single Idle Alignment Pill (shown only when no face box is present yet) */}
                {!isFaceDetected && (
                  <div className="pointer-events-none absolute bottom-4 inset-x-0 flex justify-center z-20 px-4">
                    <div className="px-4 py-2 rounded-full bg-white/95 border border-sky-300 text-[#0F4C81] text-xs sm:text-sm font-bold flex items-center gap-2 shadow-lg max-w-full truncate">
                      <span className="w-2.5 h-2.5 rounded-full bg-[#0284C7] shrink-0" />
                      <span className="truncate">{bottomPillText}</span>
                    </div>
                  </div>
                )}
              </>
            ) : (
              /* Camera Paused / Fallback Overlay inside Viewport */
              <div className="absolute inset-0 w-full h-full flex flex-col items-center justify-center p-6 text-center text-[#0F4C81] bg-gradient-to-br from-emerald-50 via-sky-50 to-white">
                <CameraOff className="w-12 h-12 text-[#0284C7] mb-3" />
                <h3 className="text-lg font-extrabold text-[#0F4C81]">{c.cameraPausedTitle}</h3>
                <p className="text-xs text-slate-600 mt-1 max-w-sm">
                  {cameraError ? `${c.cameraPausedDesc} (${cameraError})` : c.cameraPausedDesc}
                </p>
                <div className="mt-5 flex flex-wrap items-center justify-center gap-2.5">
                  <button
                    type="button"
                    onClick={handleRestartScan}
                    className="px-4 py-2.5 rounded-xl bg-gradient-to-r from-[#10B981] to-[#2563EB] text-white text-xs font-bold flex items-center gap-1.5 shadow-md cursor-pointer"
                  >
                    <RefreshCw className="w-3.5 h-3.5" />
                    <span>{c.resumeCameraBtn}</span>
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setPinDigits('');
                      setPinError(null);
                      setShowPinModal(true);
                    }}
                    className="px-4 py-2.5 rounded-xl bg-white text-slate-800 text-xs font-bold flex items-center gap-1.5 shadow-md cursor-pointer"
                  >
                    <KeyRound className="w-3.5 h-3.5 text-amber-600" />
                    <span>{c.usePinFallbackBtn}</span>
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* Primary Gradient Scan Button */}
          <button
            type="button"
            id="kiosk-scan-face-btn"
            data-testid="kiosk-scan-face-btn"
            onClick={handleRestartScan}
            className="mt-2.5 w-full py-3 px-4 rounded-2xl bg-gradient-to-r from-[#059669] via-[#0D9488] to-[#2563EB] hover:brightness-105 text-white font-extrabold text-sm sm:text-base shadow-lg shadow-emerald-600/20 flex items-center justify-center gap-2 cursor-pointer transition active:scale-[0.99] shrink-0"
          >
            <RefreshCw className={`w-4 h-4 ${scanStage === 'verifying' ? 'animate-spin' : ''}`} />
            <span>{scanStage === 'verifying' ? c.verifying : c.rescanBtn}</span>
          </button>

          {/* Secondary Action Buttons */}
          <div className="mt-2 grid grid-cols-2 gap-2.5 shrink-0">
            <button
              type="button"
              id="kiosk-pin-login-btn"
              data-testid="kiosk-pin-login-btn"
              onClick={() => {
                setPinDigits('');
                setPinError(null);
                setShowPinModal(true);
              }}
              className="py-2.5 px-3 rounded-xl bg-white border border-slate-200 hover:bg-slate-50 text-slate-800 text-xs sm:text-sm font-bold flex items-center justify-center gap-2 shadow-xs cursor-pointer transition active:scale-[0.99]"
            >
              <KeyRound className="w-4 h-4 text-amber-500" />
              <span>{c.pinBtn}</span>
            </button>
            <button
              type="button"
              id="kiosk-enroll-new-btn"
              data-testid="kiosk-enroll-new-btn"
              onClick={onOpenStaffRegister}
              className="py-2.5 px-3 rounded-xl bg-white border border-blue-200 hover:bg-blue-50/70 text-blue-700 text-xs sm:text-sm font-bold flex items-center justify-center gap-2 shadow-xs cursor-pointer transition active:scale-[0.99]"
            >
              <UserPlus className="w-4 h-4 text-blue-600" />
              <span>{c.enrollSelfBtn}</span>
            </button>
          </div>

          {/* Card Footer */}
          <div className="mt-2.5 pt-2 border-t border-slate-100 flex flex-wrap items-center justify-between gap-2 text-[11px] text-slate-500 shrink-0">
            <div className="flex items-center gap-1.5 font-medium">
              <ShieldCheck className="w-3.5 h-3.5 text-rose-500" />
              <span>{c.footerBio}</span>
            </div>
            <button
              type="button"
              onClick={() => setShowAboutModal(true)}
              className="px-3 py-1 rounded-full bg-white border border-slate-200 hover:bg-slate-50 text-slate-600 text-[11px] font-semibold flex items-center gap-1 cursor-pointer transition"
            >
              <Info className="w-3.5 h-3.5 text-slate-400" />
              <span>{c.aboutSystem}</span>
            </button>
          </div>
        </div>
      </main>

      {/* 4-Digit PIN Login Modal */}
      {showPinModal && (
        <div className="fixed inset-0 z-50 bg-[#0F4C81]/35 backdrop-blur-md flex items-center justify-center p-4">
          <div className="w-full max-w-sm rounded-3xl bg-white border border-emerald-500/30 p-6 text-center shadow-2xl relative">
            <button
              type="button"
              onClick={() => setShowPinModal(false)}
              className="absolute top-4 right-4 p-1.5 rounded-full text-slate-400 hover:text-slate-700 bg-slate-100 cursor-pointer"
            >
              <X className="w-4 h-4" />
            </button>
            <div className="w-12 h-12 rounded-2xl bg-emerald-50 border border-emerald-200 text-emerald-700 flex items-center justify-center mx-auto mb-2">
              <KeyRound className="w-6 h-6 text-amber-500" />
            </div>
            <h3 className="text-lg font-extrabold text-slate-900">{c.pinModalTitle}</h3>
            <p className="text-xs text-slate-500 mt-0.5">{c.pinModalDesc}</p>

            <div className="flex justify-center gap-3 my-5">
              {[0, 1, 2, 3].map((idx) => (
                <div
                  key={idx}
                  className={`w-4 h-4 rounded-full border-2 transition-all ${
                    idx < pinDigits.length
                      ? 'bg-emerald-500 border-emerald-500 scale-110 shadow-md shadow-emerald-500/40'
                      : 'border-slate-300 bg-transparent'
                  }`}
                />
              ))}
            </div>

            {pinError && (
              <div className="mb-3 rounded-xl bg-rose-50 border border-rose-200 px-3 py-2 text-xs font-bold text-rose-700">
                {pinError}
              </div>
            )}

            <div className="grid grid-cols-3 gap-2.5 max-w-[240px] mx-auto">
              {['1', '2', '3', '4', '5', '6', '7', '8', '9'].map((digit) => (
                <button
                  key={digit}
                  type="button"
                  onClick={() => pressPinKey(digit)}
                  className="h-12 rounded-xl bg-slate-50 border border-slate-200 text-slate-800 font-bold text-lg hover:bg-slate-100 active:scale-95 transition cursor-pointer"
                >
                  {digit}
                </button>
              ))}
              <button
                type="button"
                onClick={() => {
                  setPinDigits('');
                  setPinError(null);
                }}
                className="h-12 rounded-xl bg-slate-100 border border-slate-200 text-slate-600 font-bold text-xs hover:bg-slate-200 cursor-pointer"
              >
                {c.pinClearBtn}
              </button>
              <button
                type="button"
                onClick={() => pressPinKey('0')}
                className="h-12 rounded-xl bg-slate-50 border border-slate-200 text-slate-800 font-bold text-lg hover:bg-slate-100 active:scale-95 transition cursor-pointer"
              >
                0
              </button>
              <button
                type="button"
                onClick={() => setPinDigits((d) => d.slice(0, -1))}
                className="h-12 rounded-xl bg-slate-100 border border-slate-200 text-slate-600 font-bold text-sm hover:bg-slate-200 cursor-pointer"
              >
                ⌫
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Account Not Found Modal */}
      {scanStage === 'not_found' && !staffModalOpen && (
        <div className="fixed inset-0 z-50 bg-[#0F4C81]/40 backdrop-blur-md flex items-center justify-center p-4">
          <div className="w-full max-w-md rounded-3xl bg-white border-2 border-rose-500/45 shadow-2xl p-6 text-center">
            <div className="w-16 h-16 rounded-full bg-rose-50 border border-rose-200 flex items-center justify-center mx-auto mb-3">
              <AlertCircle className="w-8 h-8 text-rose-500" />
            </div>
            <h3 className="text-xl font-extrabold text-slate-900">{c.notFoundTitle}</h3>
            <p className="text-sm text-slate-600 mt-2 leading-relaxed">{c.notFoundDesc}</p>
            <div className="mt-5 flex flex-col gap-2.5">
              <button
                type="button"
                onClick={onOpenStaffRegister}
                className="w-full py-3.5 px-4 rounded-2xl bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white font-bold text-sm shadow-lg flex items-center justify-center gap-2 cursor-pointer"
              >
                <UserPlus className="w-4 h-4" />
                <span>{c.staffAssistBtn}</span>
              </button>
              <div className="grid grid-cols-2 gap-2">
                <button
                  type="button"
                  onClick={handleRestartScan}
                  className="py-3 px-3 rounded-xl bg-white border border-slate-300 hover:bg-slate-50 text-slate-800 font-bold text-xs flex items-center justify-center gap-1.5 cursor-pointer"
                >
                  <RefreshCw className="w-3.5 h-3.5" />
                  <span>{c.retryScanBtn}</span>
                </button>
                <button
                  type="button"
                  onClick={() => {
                    void handleCallStaff();
                    handleRestartScan();
                  }}
                  className="py-3 px-3 rounded-xl bg-amber-50 hover:bg-amber-100 border border-amber-200 text-amber-900 font-bold text-xs flex items-center justify-center gap-1.5 cursor-pointer"
                >
                  <Bell className="w-3.5 h-3.5 text-amber-600" />
                  <span>{c.callStaff}</span>
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Verified Patient Confirmation Modal */}
      {verifiedModalPayload && modalCopy && (
        <div className="fixed inset-0 z-50 bg-[#0F4C81]/40 backdrop-blur-md flex items-center justify-center p-4">
          <div className="w-full max-w-md rounded-3xl bg-white border-2 border-emerald-500/50 shadow-2xl p-6 text-center">
            {/* 3-Language Selector Bar inside Verified Patient Modal */}
            <div
              className="mb-4 inline-flex items-center gap-1 rounded-2xl border border-sky-200 bg-sky-50/90 p-1"
              data-testid="verified-modal-language-switcher"
            >
              {LANG_OPTIONS.map((opt) => {
                const active = lang === opt.code;
                return (
                  <button
                    key={opt.code}
                    type="button"
                    onClick={() => handleSelectLang(opt.code)}
                    className={`rounded-xl px-2.5 py-1 text-[11px] sm:text-xs font-extrabold transition cursor-pointer ${
                      active
                        ? 'bg-[#0284C7] text-white shadow-sm'
                        : 'text-[#0F4C81] hover:bg-white'
                    }`}
                  >
                    {opt.numberedLabel}
                  </button>
                );
              })}
            </div>

            <div className="w-20 h-20 rounded-full mx-auto bg-gradient-to-tr from-emerald-500 to-teal-400 text-white flex items-center justify-center text-3xl font-black shadow-lg mb-3">
              {verifiedModalPayload.patient.fullName?.[0] || 'P'}
            </div>
            <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-emerald-50 text-emerald-700 text-xs font-bold border border-emerald-200 mb-2">
              <CheckCircle2 className="w-4 h-4 text-emerald-600" />
              <span>{modalCopy.confidenceBadge}</span>
            </div>
            <h2 className="text-2xl font-extrabold text-slate-900">
              {verifiedModalPayload.patient.fullName}
            </h2>
            <p className="text-xs text-slate-500 mt-1 font-mono">
              {modalCopy.hnAndNextAppointment}
            </p>

            {/* Full-Frame Live Camera Preview for STRICT 2-Hand Start Gesture */}
            <div className="mt-4 relative mx-auto w-full h-44 sm:h-48 rounded-2xl overflow-hidden bg-slate-950 border-2 border-emerald-300 shadow-inner">
              <video
                ref={confirmVideoRef}
                autoPlay
                playsInline
                muted
                className="w-full h-full object-contain"
                style={{ transform: 'scaleX(-1)' }}
              />
              <canvas
                ref={confirmCanvasRef}
                className="pointer-events-none absolute inset-0 w-full h-full"
              />
              {!twoHandCameraReady && (
                <div className="absolute inset-0 flex items-center justify-center bg-white/90 text-xs font-bold text-[#0F4C81]">
                  {modalCopy.openingCamera}
                </div>
              )}
              <div className="absolute top-2 inset-x-2 flex items-center justify-between gap-2 pointer-events-none">
                {!isWelcomeSpeechDone ? (
                  <>
                    <span className="px-2.5 py-1 rounded-full text-[11px] font-black shadow bg-sky-600/90 text-white border border-sky-200/70">
                      {modalCopy.speechCornerBadge}
                    </span>
                    <span className="px-2.5 py-1 rounded-full text-[11px] font-black shadow bg-sky-600/90 text-white border border-sky-200/70">
                      {modalCopy.speechCornerBadge}
                    </span>
                  </>
                ) : (
                  <>
                    <span
                      className={`px-2.5 py-1 rounded-full text-[11px] font-black shadow ${
                        leftHandUpState
                          ? 'bg-emerald-600 text-white'
                          : 'bg-slate-900/75 text-slate-200 border border-white/25'
                      }`}
                    >
                      {leftHandUpState ? '✓ มือซ้ายพร้อม' : '○ รอมือซ้าย'}
                    </span>
                    <span
                      className={`px-2.5 py-1 rounded-full text-[11px] font-black shadow ${
                        rightHandUpState
                          ? 'bg-emerald-600 text-white'
                          : 'bg-slate-900/75 text-slate-200 border border-white/25'
                      }`}
                    >
                      {rightHandUpState ? '✓ มือขวาพร้อม' : '○ รอมือขวา'}
                    </span>
                  </>
                )}
              </div>
              <div className="absolute inset-x-2 bottom-2 flex items-center justify-center pointer-events-none">
                {!isWelcomeSpeechDone ? (
                  <span
                    key="pill-speech"
                    className={`px-3 py-1 rounded-full text-[11px] font-extrabold border shadow ${
                      raisedDuringSpeech
                        ? 'bg-amber-500/95 text-white border-amber-200'
                        : 'bg-sky-600/95 text-white border-sky-200'
                    }`}
                  >
                    {raisedDuringSpeech ? modalCopy.speechEarlyRaiseBanner : modalCopy.speechListenBanner}
                  </span>
                ) : twoHandDetected ? (
                  <span
                    key="pill-holding"
                    ref={confirmPillLabelRef}
                    className="px-3 py-1 rounded-full text-[11px] font-extrabold border shadow bg-emerald-600/95 text-white border-emerald-200"
                  >
                    {modalCopy.gestureDetectedPill}
                  </span>
                ) : (
                  <span
                    key="pill-ready"
                    className={`px-3 py-1 rounded-full text-[11px] font-extrabold border shadow ${
                      leftHandUpState !== rightHandUpState
                        ? 'bg-amber-500/95 text-white border-amber-200'
                        : !twoHandDetectorReady
                          ? 'bg-white/95 text-[#0F4C81] border-emerald-300'
                          : 'bg-emerald-600/95 text-white border-emerald-200'
                    }`}
                  >
                    {leftHandUpState !== rightHandUpState
                      ? modalCopy.gestureSingleHandPill
                      : !twoHandDetectorReady
                        ? modalCopy.gestureLoadingPill
                        : modalCopy.gestureReadyPill}
                  </span>
                )}
              </div>
            </div>

            {/* Dedicated 4-Second Loading Bar (locked strictly at 0% until welcome voice completes) */}
            <div
              className={`mt-3 rounded-2xl border p-3 text-left ${
                !isWelcomeSpeechDone
                  ? raisedDuringSpeech
                    ? 'border-amber-200 bg-amber-50/75'
                    : 'border-slate-200 bg-slate-50/90'
                  : 'border-emerald-200 bg-emerald-50/70'
              }`}
            >
              <div
                className={`flex items-center justify-between text-[11px] font-extrabold mb-1.5 ${
                  !isWelcomeSpeechDone
                    ? raisedDuringSpeech
                      ? 'text-amber-900'
                      : 'text-slate-700'
                    : 'text-emerald-950'
                }`}
              >
                <span>
                  {!isWelcomeSpeechDone
                    ? raisedDuringSpeech
                      ? modalCopy.speechEarlyRaiseBanner
                      : modalCopy.speechBarHeader
                    : twoHandDetected
                      ? '⏳ กำลังโหลดยืนยัน 2 มือค้าง 4 วินาที...'
                      : leftHandUpState !== rightHandUpState
                        ? '⚠️ ยกมือเดียวไม่นับ กรุณายกขึ้นทั้ง 2 ข้าง'
                        : 'หลอดโหลดเริ่มฝึก (ยก 2 ข้างค้าง 4 วิ)'}
                </span>
                <span
                  key={isWelcomeSpeechDone ? lang : 'locked'}
                  ref={confirmProgressTextRef}
                  className={`font-mono ${!isWelcomeSpeechDone ? 'text-slate-500' : 'text-emerald-700'}`}
                >
                  0.0 / {(VERIFIED_HAND_HOLD_MS / 1000).toFixed(0)}.0 วิ (0%)
                </span>
              </div>
              <div className="h-4 w-full overflow-hidden rounded-full bg-slate-200 border border-slate-300 shadow-inner">
                <div
                  ref={confirmProgressFillRef}
                  className={`h-full w-full origin-left rounded-full ${
                    !isWelcomeSpeechDone
                      ? 'bg-slate-400'
                      : twoHandDetected
                        ? 'bg-gradient-to-r from-emerald-500 via-teal-400 to-sky-500'
                        : leftHandUpState !== rightHandUpState
                          ? 'bg-amber-400'
                          : 'bg-emerald-500'
                  }`}
                  style={{ transform: 'scaleX(0)', willChange: 'transform' }}
                />
              </div>
            </div>

            <div className="mt-4 flex flex-col gap-2.5">
              <button
                type="button"
                onClick={() => onPatientAuthenticated(verifiedModalPayload)}
                className="relative overflow-hidden w-full py-3.5 px-4 rounded-2xl bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white font-extrabold text-sm shadow-lg cursor-pointer"
              >
                <span
                  ref={confirmBtnFillRef}
                  className="pointer-events-none absolute inset-0 w-full origin-left bg-white/25"
                  style={{ transform: 'scaleX(0)', willChange: 'transform' }}
                />
                <span className="relative z-10 flex items-center justify-center gap-2">
                  {isWelcomeSpeechDone && twoHandProgress > 0 ? (
                    <span key="btn-holding" ref={confirmBtnLabelRef}>
                      {modalCopy.startBtnHolding}
                    </span>
                  ) : (
                    <span key="btn-idle">
                      {modalCopy.startBtnIdle}
                    </span>
                  )}
                </span>
              </button>
              {onOpenTelemetrySimulator && (
                <button
                  type="button"
                  onClick={() => {
                    onPatientAuthenticated(verifiedModalPayload);
                    onOpenTelemetrySimulator();
                  }}
                  className="w-full py-2.5 px-4 rounded-xl bg-sky-50 hover:bg-sky-100 border border-sky-200 text-sky-800 text-xs font-bold cursor-pointer"
                >
                  {modalCopy.telemetryBtn}
                </button>
              )}
              <button
                type="button"
                onClick={handleRestartScan}
                className="text-xs text-slate-500 hover:text-slate-700 underline py-1 cursor-pointer"
              >
                {modalCopy.notMeBtn}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* About System Modal */}
      {showAboutModal && (
        <div className="fixed inset-0 z-50 bg-[#0F4C81]/35 backdrop-blur-md flex items-center justify-center p-4">
          <div className="w-full max-w-md rounded-3xl bg-white border border-emerald-500/30 p-6 shadow-2xl relative">
            <button
              type="button"
              onClick={() => setShowAboutModal(false)}
              className="absolute top-4 right-4 p-1.5 rounded-full text-slate-400 hover:text-slate-700 bg-slate-100 cursor-pointer"
            >
              <X className="w-4 h-4" />
            </button>
            <div className="flex items-center gap-3 mb-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-[#0EA5E9] to-[#10B981] text-white flex items-center justify-center">
                <HeartPulse className="w-5 h-5" />
              </div>
              <div>
                <h3 className="text-lg font-extrabold text-slate-900">Dr.Tech.Care AI Kiosk</h3>
                <p className="text-xs text-slate-500">Clinical Biometrics &amp; Rehab Platform</p>
              </div>
            </div>
            <div className="space-y-2 text-xs text-slate-600 leading-relaxed">
              <p>{c.aboutBullet1}</p>
              <p>
                {c.aboutBullet2Prefix}
                {supabaseHost}
                {c.aboutBullet2Suffix}
              </p>
              <p>{c.aboutBullet3}</p>
            </div>
            <div className="mt-5 flex flex-col gap-2">
              {onOpenTelemetrySimulator && (
                <button
                  type="button"
                  onClick={() => {
                    setShowAboutModal(false);
                    onOpenTelemetrySimulator();
                  }}
                  className="w-full py-2.5 px-4 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-bold cursor-pointer"
                >
                  {c.aboutTelemetryBtn}
                </button>
              )}
              <button
                type="button"
                onClick={() => setShowAboutModal(false)}
                className="w-full py-2.5 px-4 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-700 text-xs font-bold cursor-pointer"
              >
                {c.aboutCloseBtn}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

