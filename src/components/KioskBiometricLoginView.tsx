import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  FaceLandmarker,
  FilesetResolver,
  HandLandmarker,
  PoseLandmarker,
  type NormalizedLandmark,
} from '@mediapipe/tasks-vision';
import {
  ApiError,
  DrTechCaresApiClient,
  isPatientCompletedTodayInStorage,
  resolveTodayPatientSchedule,
  type BiometricLibraryStatsResponse,
  type BiometricVerifyDiagnostics,
  type BiometricVerifyReasonCode,
  type DailyScheduleDay,
  type KioskFaceVerifyResponse,
  type PatientBiometricIndexEntry,
  type PatientRecord,
  type RehabScheduleItem,
  type StrokeType,
  type AffectedSide,
  type TodayPrescribedExerciseItem,
} from '@drtechcares/api-client';
import {
  computeBoundingBoxFromLandmarks,
  subscribeBiometricIndex,
} from '../lib/faceMatcher';
import {
  BIOMETRIC_CENTERED_COSINE_THRESHOLD,
  BIOMETRIC_ENGINE_ID,
  BIOMETRIC_MATCH_THRESHOLD,
  REQUIRED_MATCH_STREAK,
  captureAndExtractOcularIris256,
  captureCanonicalFaceSnapshot,
  captureCanonicalFaceSnapshotWithMetrics,
  captureWideCanonicalFaceFallback,
  cosineSimilarityNormalized,
  createMirroredCanvasSnapshot,
  extractCanonicalFaceDescriptor,
  getHumanEngineTelemetry,
  warmupBiometricEngine,
  type OcularIrisCaptureResult,
} from '../lib/biometricEngine';
import {
  KioskBiometricScanProfiler,
  type BiometricLivenessCue,
  type KioskProfilerLiveSnapshot,
} from '../lib/biometricScanProfiler';
import {
  computeContrastNormalizedSharpness,
  resolveKioskReadinessGuidance,
  smoothBiometricBox,
  smoothFacialLandmarks,
  type SmoothedBoxState,
} from '../lib/biometricTrackerSmoothing';
import {
  getCachedHandLandmarker,
  getCachedPoseLandmarker,
  getCachedPrimaryFaceLandmarker,
  getCachedRoiFaceLandmarker,
  getNextMonotonicHandDetectTs,
  getNextMonotonicPoseDetectTs,
  getNextMonotonicVideoDetectTs,
  getVisionWarmupTelemetry,
  preloadRehabPosePipeline,
} from '../lib/biometricVisionCache';
import {
  computeOwnerFaceBoxMetrics,
  drawOwnerBiometricOverlayOnCanvas,
  evaluateOwnerBoundKinematicHands,
  verifyOwnerFace1to1,
  type OwnerBiometricTemplate,
  type OwnerFaceBoxMetrics,
  type OwnerLockStatus,
} from '../lib/ownerBiometricLock';
import { useClinicalI18n, type ClinicalLang } from '@drtechcares/clinical-i18n';
import {
  buildTtsProxyUrl,
  cancelAllKioskVoiceSpeech,
  getSeniorVoiceToggleLabel,
  pickSpeechVoice,
  readStoredKioskVoiceMuted,
  sanitizeSpeechText,
  subscribeKioskVoiceMuted,
  writeStoredKioskVoiceMuted,
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
  extractGeometricAnthropometricRatios,
  inspectBiometricFrame3D,
  listClinicalCameraDevices,
  openBestAvailableCameraStream,
  remapRoiLandmarksToFrame,
  renderUpscaledRoi256,
  selectClinicalCameraDeviceId,
  selectPrimaryFace,
  type BiologicalLivenessReport,
  type BiometricVerificationJson,
  type ClinicalCameraDevice,
  type LivenessAndPoseReport,
} from '@drtechcares/face-client';
import { FacePresenceTracker } from '@drtechcares/kiosk-input';
import { POSE_INDEX } from '@drtechcares/pose-engine';
import {
  STRICT_LOGIN_MAX_PITCH_DEG,
  STRICT_LOGIN_MAX_YAW_DEG,
  computeCraniofacialRatios12,
  flipIrisOcularRatios8Horizontally,
  flipOcularIrisPatch256Horizontally,
  flipPatch384Horizontally,
  l2Normalize,
  normalizeOcularIrisPatch256,
  purgeLegacyBiometricCache,
  type GeometricAnthropometricRatios,
} from '@drtechcares/smart-biometric-core';
import {
  useEdgeVisionSocket,
  type EdgeKioskTelemetryPayload,
} from '../hooks/useEdgeVisionSocket';
import {
  KIOSK_AUTH_HOLD_DELAY_MS,
  KineticAuthStatusBadge,
  KineticEmblemSvg,
} from './KineticAuthStatusBadge';
import { DrTechCareMascot } from './DrTechCareMascot';
import {
  AlertTriangle,
  Bell,
  Camera,
  CameraOff,
  FlipHorizontal2,
  HeartPulse,
  Info,
  KeyRound,
  Moon,
  RefreshCw,
  RotateCw,
  ShieldCheck,
  Sparkles,
  UserPlus,
  Volume2,
  VolumeX,
  X,
} from 'lucide-react';

export const KIOSK_FAST_MATCH_STREAK = 3;
export const KIOSK_FAST_UNREGISTERED_FRAMES = 3;
export const KIOSK_FAST_UNREGISTERED_TIMEOUT_MS = 1100;
export const KIOSK_MAX_DECISION_WINDOW_MS = 2600;
export const KIOSK_MAX_DECISION_EVAL_FRAMES = 6;
export const KIOSK_NO_FACE_GUIDANCE_DELAY_MS = 1500;
export const KIOSK_RESCAN_COOLDOWN_MS = 1500;
export const KIOSK_MAX_EXPOSURE_SETTLE_MS = 800;
export const KIOSK_DETECT_MAX_LONG_SIDE_PX = 480;
export const KIOSK_OVERLAY_MIN_INTERVAL_MS = 32;
export const KIOSK_IDLE_STANDBY_TIMEOUT_MS = 45000;
export const KIOSK_NOT_FOUND_IDLE_STANDBY_MS = 60000;
export const KIOSK_NOT_FOUND_VOICE_PROMPT_TH =
  'ไม่พบบัญชีในระบบครับ กรุณากดปุ่มสมัครบัญชี เพื่อลงทะเบียนคนไข้ใหม่ก่อนใช้งานนะครับ';

export function shouldAutoReturnToStandby(params: {
  hasCallback: boolean;
  staffModalOpen: boolean;
  hasVerifiedModal: boolean;
  isFaceDetected: boolean;
  scanStage: string;
  idleElapsedMs: number;
}): boolean {
  if (!params.hasCallback || params.staffModalOpen || params.hasVerifiedModal) {
    return false;
  }
  if (params.scanStage === 'not_found' || params.scanStage === 'system_error') {
    return params.idleElapsedMs >= KIOSK_NOT_FOUND_IDLE_STANDBY_MS;
  }
  if (params.isFaceDetected) {
    return false;
  }
  return params.idleElapsedMs >= KIOSK_IDLE_STANDBY_TIMEOUT_MS;
}

export function shouldTriggerFastNotFoundVerdict(params: {
  candidatesLoaded: boolean;
  humanWarmupReady: boolean;
  documentHidden: boolean;
  isAmbiguous: boolean;
  isSpoof: boolean;
  bioVerified: boolean;
  matchStreak: number;
  unmatchedClearFrames: number;
  unmatchedElapsedMs: number;
  inRescanCooldown: boolean;
  hasZeroCandidates: boolean;
  totalEvaluatedFrames?: number;
  totalDecisionElapsedMs?: number;
  serverRejectedCount?: number;
  systemError?: boolean;
}): boolean {
  if (
    !params.candidatesLoaded ||
    !params.humanWarmupReady ||
    params.documentHidden ||
    params.isSpoof ||
    params.inRescanCooldown ||
    params.systemError
  ) {
    return false;
  }
  if ((params.serverRejectedCount ?? 0) >= 1) {
    return true;
  }
  if (
    (params.totalEvaluatedFrames ?? 0) >= KIOSK_MAX_DECISION_EVAL_FRAMES &&
    (params.totalDecisionElapsedMs ?? 0) >= KIOSK_MAX_DECISION_WINDOW_MS
  ) {
    return true;
  }
  if (params.isAmbiguous || params.matchStreak > 0) {
    return false;
  }
  if (params.hasZeroCandidates) {
    return (
      params.unmatchedClearFrames >= 2 ||
      params.unmatchedElapsedMs >= KIOSK_FAST_UNREGISTERED_TIMEOUT_MS
    );
  }
  return (
    params.unmatchedClearFrames >= KIOSK_FAST_UNREGISTERED_FRAMES ||
    params.unmatchedElapsedMs >= KIOSK_FAST_UNREGISTERED_TIMEOUT_MS
  );
}

export interface KioskSystemErrorPopupProps {
  reasonCode?: BiometricVerifyReasonCode | null;
  onRescan: () => void;
  onUsePin: () => void;
  onCallStaff: () => void;
}

export function KioskSystemErrorPopup({
  reasonCode = 'SERVER_ERROR',
  onRescan,
  onUsePin,
  onCallStaff,
}: KioskSystemErrorPopupProps) {
  const isRateLimited = reasonCode === 'RATE_LIMITED';
  return (
    <div
      className="fixed inset-0 z-50 bg-[#0F4C81]/45 backdrop-blur-md flex items-center justify-center p-4 animate-backdrop-fade"
      data-testid="kiosk-system-error-modal"
    >
      <div className="w-full max-w-xl rounded-3xl bg-white border-2 border-amber-400/70 shadow-2xl p-6 sm:p-8 text-center animate-popup-spring">
        <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-2xl border-2 border-amber-300 bg-amber-50 text-amber-600 shadow-sm">
          <AlertTriangle className="h-9 w-9" />
        </div>
        <h2 className="text-2xl md:text-3xl font-black text-amber-800 leading-tight">
          {isRateLimited ? 'กรุณารอสักครู่ก่อนสแกนใหม่' : 'ระบบตรวจสอบขัดข้องชั่วคราว'}
        </h2>
        <p className="text-base md:text-lg font-bold text-slate-700 mt-2 leading-snug">
          {isRateLimited
            ? 'มีการลองสแกนไม่ผ่านหลายครั้งเกินกำหนด กรุณารอสักครู่แล้วลองใหม่ หรือเรียกเจ้าหน้าที่ช่วยเหลือ'
            : 'ระบบตรวจสอบขัดข้องชั่วคราว กรุณาลองใหม่หรือเรียกเจ้าหน้าที่'}
        </p>

        <div className="flex flex-col sm:flex-row gap-3 mt-6">
          <button
            type="button"
            data-testid="system-error-rescan-btn"
            onClick={onRescan}
            className="btn-tactile btn-primary-sheen bg-[#0284C7] hover:bg-[#0369A1] text-white font-black text-lg px-6 py-4 rounded-2xl shadow-lg flex-1 inline-flex items-center justify-center gap-2 cursor-pointer"
          >
            <RefreshCw className="w-5 h-5 shrink-0" />
            <span>ลองสแกนใหม่อีกครั้ง</span>
          </button>
          <button
            type="button"
            data-testid="system-error-pin-btn"
            onClick={onUsePin}
            className="btn-tactile bg-slate-100 hover:bg-slate-200 border-2 border-slate-300 text-slate-800 font-extrabold text-base px-5 py-4 rounded-2xl inline-flex items-center justify-center gap-2 cursor-pointer"
          >
            <KeyRound className="w-5 h-5 text-amber-600 shrink-0" />
            <span>ใช้รหัส PIN 4 หลัก</span>
          </button>
        </div>

        <div className="mt-3 flex justify-center">
          <button
            type="button"
            data-testid="system-error-call-staff-btn"
            onClick={onCallStaff}
            className="btn-tactile inline-flex items-center justify-center gap-1.5 rounded-xl border border-amber-200 bg-amber-50 hover:bg-amber-100 px-4 py-2.5 text-sm font-bold text-amber-900 cursor-pointer"
          >
            <Bell className="w-4 h-4 text-amber-600 shrink-0" />
            <span>เรียกเจ้าหน้าที่ช่วยเหลือ</span>
          </button>
        </div>
      </div>
    </div>
  );
}

export interface UnregisteredFaceSignupPopupProps {
  scanKey: number;
  isSpeaking?: boolean;
  onRegisterNewPatient: () => void;
  onRescan: () => void;
  onCallStaff: () => void;
}

export function UnregisteredFaceSignupPopup({
  scanKey,
  isSpeaking = false,
  onRegisterNewPatient,
  onRescan,
  onCallStaff,
}: UnregisteredFaceSignupPopupProps) {
  return (
    <div
      className="fixed inset-0 z-50 bg-[#0F4C81]/45 backdrop-blur-md flex items-center justify-center p-4 animate-backdrop-fade"
      data-testid="unregistered-face-signup-modal"
    >
      <div className="w-full max-w-xl rounded-3xl bg-white border-2 border-red-500/45 shadow-2xl p-6 sm:p-8 text-center animate-popup-spring">
        <KineticAuthStatusBadge
          key={`not-found-${scanKey}`}
          status="not_found"
          compact
          testId="kiosk-not-found-feedback-panel"
          heading="ไม่พบบัญชี"
          subtitle="ผลการสแกนใบหน้าไม่ตรงกับบัญชีใดในฐานข้อมูล"
        />

        <div className="mt-4 rounded-2xl border border-rose-200 bg-rose-50/75 p-4 flex flex-col sm:flex-row items-center gap-4 text-center sm:text-left">
          <DrTechCareMascot pose="explaining" isSpeaking={isSpeaking} size="sm" />
          <div className="min-w-0 flex-1">
            <h2 className="text-2xl md:text-3xl font-black text-red-700 leading-tight">
              ไม่พบบัญชีผู้ป่วยในระบบ
            </h2>
            <p className="text-base md:text-lg font-bold text-slate-700 mt-1 leading-snug">
              ใบหน้านี้ยังไม่ได้ลงทะเบียนใช้งานตู้กายภาพบำบัด Dr.Tech.Care กรุณาสมัครบัญชีใหม่ก่อนเริ่มฝึก
            </p>
          </div>
        </div>

        <div className="flex flex-col sm:flex-row gap-3 mt-5">
          <button
            type="button"
            data-testid="unregistered-signup-btn"
            onClick={onRegisterNewPatient}
            className="btn-tactile btn-primary-sheen bg-emerald-600 hover:bg-emerald-700 text-white font-black text-lg md:text-xl px-6 py-4 rounded-2xl shadow-lg flex-1 inline-flex items-center justify-center gap-2.5 cursor-pointer"
          >
            <UserPlus className="w-6 h-6 shrink-0" />
            <span>สมัครบัญชี / ลงทะเบียนคนไข้ใหม่</span>
          </button>
          <button
            type="button"
            data-testid="unregistered-rescan-btn"
            onClick={onRescan}
            className="btn-tactile bg-slate-100 hover:bg-slate-200 border-2 border-slate-300 text-slate-800 font-extrabold text-base md:text-lg px-5 py-4 rounded-2xl inline-flex items-center justify-center gap-2 cursor-pointer"
          >
            <RefreshCw className="w-5 h-5 shrink-0" />
            <span>สแกนใหม่อีกครั้ง</span>
          </button>
        </div>

        <div className="mt-3 flex justify-center">
          <button
            type="button"
            data-testid="unregistered-call-staff-btn"
            onClick={onCallStaff}
            className="btn-tactile inline-flex items-center justify-center gap-1.5 rounded-xl border border-amber-200 bg-amber-50 hover:bg-amber-100 px-4 py-2.5 text-sm font-bold text-amber-900 cursor-pointer"
          >
            <Bell className="w-4 h-4 text-amber-600 shrink-0" />
            <span>เรียกเจ้าหน้าที่ช่วยเหลือ</span>
          </button>
        </div>
      </div>
    </div>
  );
}

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
  biometricTemplate?: OwnerBiometricTemplate;
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
  directBiometricTemplate?: OwnerBiometricTemplate;
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
  const biometricTemplate: OwnerBiometricTemplate | undefined =
    params.directBiometricTemplate ??
    (params.indexEntry &&
    Array.isArray(params.indexEntry.masterEmbedding) &&
    params.indexEntry.masterEmbedding.length > 0
      ? {
          patientId: params.patient.id,
          hn: params.patient.hn,
          fullName: params.patient.fullName,
          masterEmbedding: params.indexEntry.masterEmbedding,
          angleEmbeddings: params.indexEntry.angleEmbeddings,
          ...(Array.isArray(params.indexEntry.exemplarEmbeddings) &&
          params.indexEntry.exemplarEmbeddings.length > 0
            ? { exemplarEmbeddings: params.indexEntry.exemplarEmbeddings }
            : {}),
          ...(params.indexEntry.biometricEngine
            ? { biometricEngine: params.indexEntry.biometricEngine }
            : {}),
          geometricRatios: params.indexEntry.geometricRatios,
        }
      : undefined);
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
    ...(biometricTemplate ? { biometricTemplate } : {}),
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
  onReturnToStandby?: () => void;
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
  voiceMuted?: boolean;
}): boolean {
  if (params.voiceMuted) return false;
  return !params.speechDone || Boolean(params.browserSpeaking);
}

export type KioskFaceOverlayVariant = 'SPOOF' | 'LIVENESS' | 'UNREGISTERED' | 'CONFIRMING' | 'VERIFIED';

export interface KioskFaceBoundingBoxOverlay {
  variant: KioskFaceOverlayVariant;
  boxColor: 'rose' | 'amber' | 'blue' | 'emerald';
  label: string;
  subLabel: string | null;
  displayName: string | null;
}

/**
 * Resolves Kiosk face bounding-box visual state with strict zero-leak privacy:
 * NEVER returns a candidate name when rawCosine < 0.84, centeredCosine < 0.80, or before 6/6 streak completes.
 */
export function resolveKioskFaceBoundingBoxOverlay(params: {
  isSpoof: boolean;
  isBioVerified: boolean;
  rawCosine: number;
  centeredCosine?: number;
  hasEvaluatedMatch?: boolean;
  gatePassed?: boolean;
  streak: number;
  requiredStreak: number;
  confirmedCandidateName: string | null;
  confirmedCandidateHn?: string | null;
  distanceM?: number;
  confidencePercent?: number;
  isLongRange?: boolean;
  isAmbiguous?: boolean;
}): KioskFaceBoundingBoxOverlay {
  if (params.isSpoof) {
    return {
      variant: 'SPOOF',
      boxColor: 'rose',
      label: 'ตรวจพบความเสี่ยง Spoof (ภาพถ่าย/หน้าจอมือถือ)',
      subLabel: null,
      displayName: null,
    };
  }
  if (!params.isBioVerified) {
    return {
      variant: 'LIVENESS',
      boxColor: 'blue',
      label: params.isLongRange
        ? 'พบใบหน้าจากระยะไกล — กำลังยืนยันความมีชีวิต...'
        : 'กรุณากะพริบตาหรือขยับศีรษะเพื่อยืนยันใบหน้าจริง',
      subLabel: `ระยะ ${(params.distanceM ?? 1.8).toFixed(1)} ม.`,
      displayName: null,
    };
  }
  const required = Math.max(1, params.requiredStreak || REQUIRED_MATCH_STREAK);
  const clampedStreak = Math.max(0, Math.min(required, params.streak));
  const centeredOk =
    params.centeredCosine === undefined ||
    params.centeredCosine >= BIOMETRIC_CENTERED_COSINE_THRESHOLD;
  const gate1Ok = params.rawCosine >= BIOMETRIC_MATCH_THRESHOLD && centeredOk;

  if (
    gate1Ok &&
    params.gatePassed !== false &&
    clampedStreak >= required &&
    params.confirmedCandidateName &&
    !params.isAmbiguous
  ) {
    const pct = (params.confidencePercent ?? params.rawCosine * 100).toFixed(1);
    const fullLabel = params.confirmedCandidateHn
      ? `ยืนยันเจ้าของบัญชี: ${params.confirmedCandidateName} (${params.confirmedCandidateHn})`
      : `ยืนยันเจ้าของบัญชี: ${params.confirmedCandidateName}`;
    return {
      variant: 'VERIFIED',
      boxColor: 'emerald',
      label: fullLabel,
      subLabel: `ความตรงกัน ${pct}% (${clampedStreak}/${required})`,
      displayName: params.confirmedCandidateName,
    };
  }
  if (gate1Ok && params.gatePassed !== false && clampedStreak >= 1 && !params.isAmbiguous) {
    return {
      variant: 'CONFIRMING',
      boxColor: 'blue',
      label: `กำลังตรวจสอบใบหน้า... (${clampedStreak}/${required})`,
      subLabel: `ระยะ ${(params.distanceM ?? 1.8).toFixed(1)} ม.`,
      displayName: null,
    };
  }
  if (params.isAmbiguous) {
    return {
      variant: 'UNREGISTERED',
      boxColor: 'amber',
      label: 'พบใบหน้าตรงกับหลายบัญชีในระบบ — กรุณามองตรงหรือแจ้งเจ้าหน้าที่ตรวจสอบบัญชีซ้ำ',
      subLabel: null,
      displayName: null,
    };
  }
  if (params.hasEvaluatedMatch && (!gate1Ok || params.gatePassed === false)) {
    return {
      variant: 'UNREGISTERED',
      boxColor: 'rose',
      label: 'ไม่พบบัญชีที่ตรงกับใบหน้านี้ (กรุณาลงทะเบียนคนไข้ใหม่ก่อนใช้งาน)',
      subLabel: null,
      displayName: null,
    };
  }
  return {
    variant: 'UNREGISTERED',
    boxColor: 'amber',
    label: `กำลังตรวจสอบใบหน้า... (${clampedStreak}/${required})`,
    subLabel: null,
    displayName: null,
  };
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
      speechCornerBadge: 'Speaking guide...',
      speechListenBanner: 'Please listen until the guide finishes before raising both hands...',
      speechEarlyRaiseBanner: 'Please wait for the voice guide to finish before holding hands up',
      speechBarHeader: 'Waiting for voice guide to finish before starting...',
      gestureDetectedPill: `Both hands detected! Hold steady ${heldSec}/${totalSec.toFixed(0)}.0s (${params.twoHandProgressPct}%)`,
      gestureSingleHandPill: '1 hand only does not count — raise BOTH hands together',
      gestureLoadingPill: 'Preparing AI 2-hand detector...',
      gestureReadyPill: `Guide finished! Raise BOTH hands & hold ${totalSec.toFixed(0)}s to start`,
      startBtnHolding: `Holding BOTH hands... ${heldSec}/${totalSec.toFixed(0)}.0s (${remainSec}s left)`,
      startBtnIdle: `Raise BOTH Hands ${totalSec.toFixed(0)}s (Both Required) or Tap to Start →`,
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
      speechCornerBadge: '语音引导中...',
      speechListenBanner: '请先听完语音引导后再举起双手...',
      speechEarlyRaiseBanner: '请等语音引导结束后再举手保持',
      speechBarHeader: '等待语音引导结束后开始双手检测...',
      gestureDetectedPill: `已检测到双手举起！请保持 ${heldSec}/${totalSec.toFixed(0)}.0 秒 (${params.twoHandProgressPct}%)`,
      gestureSingleHandPill: '单手不计入 — 请同时举起双手',
      gestureLoadingPill: '正在准备 AI 双手识别模块...',
      gestureReadyPill: `语音引导已结束！请双手举起保持 ${totalSec.toFixed(0)} 秒开始训练`,
      startBtnHolding: `正在加载确认... ${heldSec}/${totalSec.toFixed(0)}.0 秒（剩余 ${remainSec} 秒）`,
      startBtnIdle: `同时举起双手保持 ${totalSec.toFixed(0)} 秒 或 点击此处开始训练 →`,
      telemetryBtn: '打开生物力学分析界面 (Smart Rehab Telemetry)',
      notMeBtn: '不是我本人（重新扫描）',
      voiceGreeting: `身份验证成功，欢迎您，${params.fullName}。请同时举起双手保持 ${totalSec.toFixed(0)} 秒，或点击按钮开始训练。`,
    };
  }
  return {
    confidenceBadge: `ความแม่นยำชีวมาตร ${params.confidencePercent}% (ผ่านเกณฑ์)`,
    hnAndNextAppointment: `HN: ${params.hn} • นัดถัดไป: ${nextAppt}`,
    openingCamera: 'กำลังเปิดกล้องตรวจจับการยกมือทั้ง 2 ข้าง...',
    speechCornerBadge: 'กำลังพูดแนะนำ...',
    speechListenBanner: 'กรุณาฟังเสียงแนะนำให้จบก่อนยกมือทั้ง 2 ข้าง...',
    speechEarlyRaiseBanner: 'รอให้เสียงพูดจบก่อนนะครับ แล้วค่อยยกมือค้างไว้',
    speechBarHeader: 'กำลังรอเสียงแนะนำพูดจบก่อนเริ่มตรวจจับ (ล็อก 0%)',
    gestureDetectedPill: `ตรวจพบยกครบ 2 ข้าง! ค้างไว้ ${heldSec}/${totalSec.toFixed(0)}.0 วิ (${params.twoHandProgressPct}%)`,
    gestureSingleHandPill: 'ยกมือเดียวไม่นับ • กรุณายกขึ้นทั้ง 2 ข้างพร้อมกัน',
    gestureLoadingPill: 'กำลังเตรียม AI ตรวจจับการยกมือ 2 ข้าง...',
    gestureReadyPill: `เสียงพูดจบแล้ว! ยกมือทั้ง 2 ข้างค้างไว้ ${totalSec.toFixed(0)} วิ เพื่อเริ่มฝึก`,
    startBtnHolding: `กำลังโหลด 2 ข้าง... ${heldSec}/${totalSec.toFixed(0)}.0 วิ (เหลือ ${remainSec} วิ)`,
    startBtnIdle: `ยกมือทั้ง 2 ข้างค้าง ${totalSec.toFixed(0)} วิ (ต้อง 2 ข้างเท่านั้น) หรือ กดที่นี่ →`,
    telemetryBtn: 'เปิดหน้าจอวิเคราะห์ชีวกลศาสตร์ (Smart Rehab Telemetry)',
    notMeBtn: 'ไม่ใช่ฉัน (สแกนใหม่)',
    voiceGreeting: `ยืนยันตัวตนสำเร็จ ยินดีต้อนรับคุณ ${params.fullName} กรุณายกมือทั้งสองข้างพร้อมกันค้างไว้ ${totalSec.toFixed(0)} วินาที หรือกดปุ่มเพื่อเริ่มฝึกครับ`,
  };
}

const REQUIRED_LOGIN_FACE_LANDMARKS = [33, 263, 1, 61, 291] as const;

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
    inspection.error_code === 'OBSTRUCTED_MOUTH' ||
    inspection.error_code === 'BLURRY'
  ) {
    return false;
  }

  for (const idx of REQUIRED_LOGIN_FACE_LANDMARKS) {
    const pt = landmarks[idx];
    if (
      !pt ||
      !Number.isFinite(pt.x) ||
      !Number.isFinite(pt.y) ||
      pt.x < 0.01 ||
      pt.x > 0.99 ||
      pt.y < 0.01 ||
      pt.y > 0.99
    ) {
      return false;
    }
  }

  const box = computeBoundingBoxFromLandmarks(landmarks);
  // Supports all distances: close-up (0.25m) to ultra-long-range (5.0m+, down to ~2.2% frame height)
  if (!box || box.height < 0.022 || box.width < 0.018) {
    return false;
  }

  // Strict frontal pose requirement for Kiosk login (|yaw| <= 14 deg, |pitch| <= 14 deg)
  if (
    Math.abs(liveness.yawDeg) > STRICT_LOGIN_MAX_YAW_DEG ||
    Math.abs(liveness.pitchDeg) > STRICT_LOGIN_MAX_PITCH_DEG
  ) {
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
    ambiguousClosePrompt: string;
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
    checkingLiveness: 'กรุณากะพริบตาหรือขยับศีรษะเล็กน้อยเพื่อยืนยันตัวตน',
    longRangePrompt: 'ตรวจพบใบหน้าระยะไกล • มองตรงหรือพยักหน้าเบาๆ เพื่อเข้าสู่ระบบ',
    distancePrefix: 'ระยะห่าง',
    distanceUnit: 'ม.',
    liveVerified: 'ยืนยันบุคคลจริง (Liveness Verified)',
    spoofBlocked: 'ตรวจพบภาพถ่ายหรือหน้าจอมือถือ — กรุณากะพริบตาหรือขยับศีรษะด้วยใบหน้าจริง',
    alignFace: 'ระบบสแกนอัตโนมัติเต็มจอ • มองกล้องเพื่อยืนยันตัวตน',
    verifying: 'กำลังสแกนและเปรียบเทียบกับฐานข้อมูลใบหน้า...',
    turnHeadPrefix: 'กรุณาหันหน้าเข้าหากล้องอีกนิด กำลังจับคู่บัญชี... (ความคล้าย',
    turnHeadSuffix: '%)',
    ambiguousClosePrompt:
      'กำลังแยกแยะความละเอียดใบหน้า กรุณามองตรงเข้าหากล้องและขยับเข้ามาใกล้ขึ้นเล็กน้อย',
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
    notFoundTitle: 'ยังไม่พบบัญชีที่ตรงกับใบหน้านี้',
    notFoundDesc:
      'ระบบตรวจสอบใบหน้าจากคลังผู้ป่วยที่ลงทะเบียนแล้วไม่พบความตรงกันตามเกณฑ์ กรุณาให้เจ้าหน้าที่ช่วยลงทะเบียน หรือลองสแกนใหม่อีกครั้ง',
    staffAssistBtn: 'เจ้าหน้าที่ลงทะเบียนให้',
    retryScanBtn: 'ลองสแกนใหม่อีกครั้ง',
    countdownLabel: 'กำลังนับถอยหลัง',
    liveMatchVerified: 'พบบัญชีตรงกัน',
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
    aboutBullet1: '• ระบบยืนยันตัวตนชีวมาตร 1024 มิติ พร้อม Anti-Spoofing 3D Liveness Detection',
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
    checkingLiveness: 'Please blink or gently move your head to verify',
    longRangePrompt: 'Long-Range Face Locked • Look ahead or nod gently to sign in',
    distancePrefix: 'Distance',
    distanceUnit: 'm',
    liveVerified: 'Live Person Verified (Liveness Verified)',
    spoofBlocked: 'Photo or Phone Screen Detected — Blink or move head with a live face',
    alignFace: 'Full-screen auto-scan active • Look at the camera',
    verifying: 'Scanning & matching face against database...',
    turnHeadPrefix: 'Turn slightly toward the camera, matching profile... (',
    turnHeadSuffix: '% match)',
    ambiguousClosePrompt:
      'Refining facial distinction — please look straight into the camera and move slightly closer',
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
      'No matching biometric profile was found among enrolled patients. Please ask staff to enroll you or try scanning again.',
    staffAssistBtn: 'Staff Register Patient',
    retryScanBtn: 'Scan Face Again',
    countdownLabel: 'Hold steady',
    liveMatchVerified: 'Account Matched',
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
    aboutBullet1: '• 1024-D facial biometric authentication with 3D Anti-Spoofing Liveness Detection',
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
    checkingLiveness: '请轻眨眼或微转头部以验证真人身份',
    longRangePrompt: '远距离人脸已锁定 • 请直视镜头或轻点头以登录',
    distancePrefix: '距离',
    distanceUnit: '米',
    liveVerified: '真人活体已验证 (Liveness Verified)',
    spoofBlocked: '检测到照片或手机屏幕 — 请使用真人眨眼或轻转头部',
    alignFace: '全屏自动扫描已开启 • 请注视摄像头',
    verifying: '正在扫描并与人脸数据库比对...',
    turnHeadPrefix: '请略微转正面向摄像头，正在匹配档案…（相似度',
    turnHeadSuffix: '%）',
    ambiguousClosePrompt:
      '正在精细区分面部特征 — 请直视摄像头并稍微靠近一些',
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
    notFoundTitle: '未找到匹配的生物识别账户',
    notFoundDesc: '当前已建档患者库中未匹配到该人脸档案，请联系医护人员协助建档或重新扫描。',
    staffAssistBtn: '医护人员协助建档',
    retryScanBtn: '重新识别人脸',
    countdownLabel: '倒计时准备中',
    liveMatchVerified: '已匹配账户',
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
    aboutBullet1: '• 1024 维面部生物识别认证，配备 3D 防伪活体检测',
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
  onReturnToStandby,
}: KioskBiometricLoginViewProps) {
  const { lang, setLang } = useClinicalI18n();
  const langRef = useRef<ClinicalLang>(lang);
  langRef.current = lang;
  const c = COPY[lang];

  const [verifiedModalPayload, setVerifiedModalPayload] = useState<KioskLoginSuccessPayload | null>(null);
  const [twoHandProgress, setTwoHandProgress] = useState(0);
  const [twoHandDetected, setTwoHandDetected] = useState(false);
  const [leftHandUpState, setLeftHandUpState] = useState(false);
  const [rightHandUpState, setRightHandUpState] = useState(false);
  const [twoHandCameraReady, setTwoHandCameraReady] = useState(false);
  const [twoHandDetectorReady, setTwoHandDetectorReady] = useState(false);
  const [ownerGuardBanner, setOwnerGuardBanner] = useState<{
    status: OwnerLockStatus;
    text: string;
  } | null>(null);
  const [isWelcomeSpeechDone, setIsWelcomeSpeechDone] = useState(() => readStoredKioskVoiceMuted());
  const [raisedDuringSpeech, setRaisedDuringSpeech] = useState(false);
  const [voiceMuted, setVoiceMuted] = useState<boolean>(() => readStoredKioskVoiceMuted());
  const voiceMutedRef = useRef<boolean>(voiceMuted);
  voiceMutedRef.current = voiceMuted;
  const isSpeechDoneRef = useRef<boolean>(readStoredKioskVoiceMuted());
  const activeUtteranceRef = useRef<SpeechSynthesisUtterance | null>(null);

  const handleToggleVoiceMute = useCallback(() => {
    const nextMuted = writeStoredKioskVoiceMuted(!voiceMutedRef.current);
    voiceMutedRef.current = nextMuted;
    setVoiceMuted(nextMuted);
    if (nextMuted) {
      modalSpeakSeqRef.current += 1;
      cancelAllKioskVoiceSpeech(modalAudioRef.current);
      activeUtteranceRef.current = null;
      isSpeechDoneRef.current = true;
      setIsWelcomeSpeechDone(true);
      setRaisedDuringSpeech(false);
    }
  }, []);

  useEffect(() => {
    return subscribeKioskVoiceMuted((muted) => {
      voiceMutedRef.current = muted;
      setVoiceMuted(muted);
      if (muted) {
        modalSpeakSeqRef.current += 1;
        cancelAllKioskVoiceSpeech(modalAudioRef.current);
        activeUtteranceRef.current = null;
        isSpeechDoneRef.current = true;
        setIsWelcomeSpeechDone(true);
        setRaisedDuringSpeech(false);
      }
    });
  }, []);

  const handleSelectLang = useCallback(
    (nextLang: ClinicalLang) => {
      // Re-lock Speech-First Relay only when voice narration is enabled; keep unlocked when muted
      const muted = voiceMutedRef.current;
      isSpeechDoneRef.current = muted;
      setIsWelcomeSpeechDone(muted);
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
  const [selectedCameraId, setSelectedCameraId] = useState<string | null>(() => {
    try {
      return typeof window !== 'undefined'
        ? window.localStorage.getItem('drtech_preferred_camera_id')
        : null;
    } catch {
      return null;
    }
  });
  const [activeCameraId, setActiveCameraId] = useState<string | null>(selectedCameraId);
  const [availableCameras, setAvailableCameras] = useState<ClinicalCameraDevice[]>([]);
  const [rotationDeg, setRotationDeg] = useState<0 | 90 | 180 | 270>(0);
  const [mirrored, setMirrored] = useState(true);
  const [showCalibHud, setShowCalibHud] = useState(false);
  const showCalibHudRef = useRef(showCalibHud);
  showCalibHudRef.current = showCalibHud;

  useEffect(() => {
    if (!showCalibHud) {
      setStaffPerfSnapshot(null);
      if (typeof window !== 'undefined') {
        delete (window as unknown as Record<string, unknown>).__drtechBiometricPerf;
      }
      return;
    }
    const snap = profilerRef.current.getSnapshot(performance.now());
    setStaffPerfSnapshot(snap);
    if (typeof window !== 'undefined') {
      (window as unknown as Record<string, unknown>).__drtechBiometricPerf = snap;
    }
    return () => {
      if (typeof window !== 'undefined') {
        delete (window as unknown as Record<string, unknown>).__drtechBiometricPerf;
      }
    };
  }, [showCalibHud]);

  useEffect(() => {
    let active = true;
    const syncDevices = async () => {
      const list = await listClinicalCameraDevices();
      if (active) {
        setAvailableCameras(list);
      }
    };
    void syncDevices();
    const md = typeof navigator !== 'undefined' ? navigator.mediaDevices : undefined;
    if (md?.addEventListener) {
      md.addEventListener('devicechange', syncDevices);
    }
    return () => {
      active = false;
      if (md?.removeEventListener) {
        md.removeEventListener('devicechange', syncDevices);
      }
    };
  }, []);

  const handleCycleCamera = useCallback(() => {
    if (availableCameras.length < 2) return;
    const activeId = activeCameraId ?? selectedCameraId;
    const currentIdx = availableCameras.findIndex((d) => d.deviceId === activeId);
    const nextCam = availableCameras[(currentIdx + 1) % availableCameras.length];
    if (!nextCam?.deviceId) return;
    setActiveCameraId(nextCam.deviceId);
    setSelectedCameraId(nextCam.deviceId);
    preferredMirrorModeRef.current = null;
    try {
      if (typeof window !== 'undefined') {
        window.localStorage.setItem('drtech_preferred_camera_id', nextCam.deviceId);
      }
    } catch {
      // Ignore storage write errors
    }
  }, [activeCameraId, availableCameras, selectedCameraId]);

  const [scanStage, setScanStage] = useState<'scanning' | 'burst' | 'verifying' | 'not_found' | 'system_error'>('scanning');
  const [systemErrorReason, setSystemErrorReason] = useState<BiometricVerifyReasonCode | null>(null);
  const [staffDiagnostics, setStaffDiagnostics] = useState<
    (BiometricVerifyDiagnostics & { qualityPassedFrames: number; lastUpdatedAt: string }) | null
  >(null);
  const [burstCount, setBurstCount] = useState(0);
  const [showNoFaceGuidance, setShowNoFaceGuidance] = useState(false);
  const [isNotFoundVoiceSpeaking, setIsNotFoundVoiceSpeaking] = useState(false);
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
      isSpeechDoneRef.current = voiceMuted;
      setIsWelcomeSpeechDone(voiceMuted);
      setRaisedDuringSpeech(false);
      return;
    }

    const seq = ++modalSpeakSeqRef.current;
    const langOpt = LANG_OPTIONS.find((item) => item.code === lang) ?? LANG_OPTIONS[0];

    if (voiceMuted) {
      cancelAllKioskVoiceSpeech(modalAudioRef.current);
      activeUtteranceRef.current = null;
      isSpeechDoneRef.current = true;
      setIsWelcomeSpeechDone(true);
      setRaisedDuringSpeech(false);
      return;
    }

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
      isSpeechDoneRef.current = voiceMutedRef.current;
      setIsWelcomeSpeechDone(voiceMutedRef.current);
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

    let fallbackTriggered = false;
    let startWatchdog: number | null = window.setTimeout(() => {
      if (!finished && modalSpeakSeqRef.current === seq) {
        markSpeechDone();
      }
    }, 2200);
    const clearStartWatchdog = () => {
      if (startWatchdog !== null) {
        window.clearTimeout(startWatchdog);
        startWatchdog = null;
      }
    };

    const speakWebFallback = () => {
      if (fallbackTriggered || finished || modalSpeakSeqRef.current !== seq) return;
      fallbackTriggered = true;
      clearStartWatchdog();
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
          const cleanSpeechText = sanitizeSpeechText(greeting);
          if (!cleanSpeechText) return;
          const utterance = new SpeechSynthesisUtterance(cleanSpeechText);
          utterance.lang = matchedVoice?.lang || langOpt.bcp47;
          if (matchedVoice) utterance.voice = matchedVoice;
          utterance.rate = VERIFIED_MODAL_SPEECH_RATE;
          utterance.onend = () => markSpeechDone();
          utterance.onerror = () => {
            activeUtteranceRef.current = null;
            markSpeechDone();
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
        audio.onplaying = () => clearStartWatchdog();
        audio.onended = () => {
          clearStartWatchdog();
          markSpeechDone();
        };
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
            } else {
              clearStartWatchdog();
            }
          });
        }
      } catch {
        speakWebFallback();
      }
    } else {
      speakWebFallback();
    }

    return () => {
      clearStartWatchdog();
      stopSpeech();
    };
  }, [verifiedModalPayload?.patient.id, lang, modalCopy?.voiceGreeting, voiceMuted]);

  const [showPinModal, setShowPinModal] = useState(false);
  const [pinDigits, setPinDigits] = useState('');
  const [pinError, setPinError] = useState<string | null>(null);
  const [pinSubmitting, setPinSubmitting] = useState(false);

  const [showAboutModal, setShowAboutModal] = useState(false);
  const [toastMsg, setToastMsg] = useState<string | null>(null);
  const [scanGeneration, setScanGeneration] = useState(0);
  const [hudBox, setHudBox] = useState<{ xPct: number; yPct: number; wPct: number; hPct: number } | null>(null);
  const hudBoxElRef = useRef<HTMLDivElement | null>(null);
  const faceMeshCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const clearFaceMeshCanvas = useCallback(() => {
    const canvas = faceMeshCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
  }, []);
  const [dbAccountCount, setDbAccountCount] = useState(0);
  const [totalAccountCount, setTotalAccountCount] = useState(0);
  const [legacyAccountCount, setLegacyAccountCount] = useState(0);
  const [supabaseSyncedFromCloud, setSupabaseSyncedFromCloud] = useState<boolean>(false);
  const [supabaseSyncError, setSupabaseSyncError] = useState<string | null>(null);
  const [liveMatchHud, setLiveMatchHud] = useState<{
    patientId: string;
    hn: string;
    fullName: string;
    confidencePercent: number;
    streak: number;
    confirmed: boolean;
  } | null>(null);
  const [liveOverlay, setLiveOverlay] = useState<KioskFaceBoundingBoxOverlay | null>(null);
  const liveOverlayRef = useRef<KioskFaceBoundingBoxOverlay | null>(null);
  const [liveDebugHud, setLiveDebugHud] = useState<{
    validAccounts: number;
    totalAccounts: number;
    legacyAccounts: number;
    syncedFromSupabase?: boolean;
    hasEvaluatedMatch?: boolean;
    gatePassed?: boolean;
    topName: string | null;
    topPct: number;
    centeredPct?: number;
    streak: number;
    requiredStreak: number;
    reason: string;
  } | null>(null);
  const probeWindowRef = useRef<
    Array<{
      ts: number;
      embedding: number[];
      sharpness: number;
      appearancePatch384?: number[];
      craniofacialRatios12?: number[];
      ocularIrisPatch256?: number[];
      irisOcularRatios8?: number[];
      yawDeg: number;
      pitchDeg: number;
    }>
  >([]);

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
  const matchSnapCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const detectCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const fallbackSnapCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const mirrorSnapCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const preferredMirrorModeRef = useRef<'normal' | 'mirrored' | null>(null);
  const irisLeftCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const irisRightCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const profilerRef = useRef<KioskBiometricScanProfiler>(new KioskBiometricScanProfiler());
  const [staffPerfSnapshot, setStaffPerfSnapshot] = useState<KioskProfilerLiveSnapshot | null>(null);
  const qualityPassedFramesRef = useRef<number>(0);
  const serverStreakRef = useRef<{ patientId: string | null; scores: number[] }>({
    patientId: null,
    scores: [],
  });
  const ambiguousSeenInAttemptRef = useRef<boolean>(false);
  const [readinessHint, setReadinessHint] = useState<string | null>(null);
  const readinessHintRef = useRef<string | null>(null);
  const updateReadinessHint = useCallback((next: string | null) => {
    if (readinessHintRef.current === next) return;
    readinessHintRef.current = next;
    setReadinessHint(next);
  }, []);
  const [irisHudState, setIrisHudState] = useState<{
    irisLocked: boolean;
    ocularMatchPct: number | null;
    ipdPx: number | null;
    latencyMs: number | null;
  }>({
    irisLocked: false,
    ocularMatchPct: null,
    ipdPx: null,
    latencyMs: null,
  });
  const bioLivenessRef = useRef(new BiologicalLivenessTracker());
  const bioReportRef = useRef<BiologicalLivenessReport | null>(null);
  bioReportRef.current = bioReport;
  const verifiedModalPayloadRef = useRef<KioskLoginSuccessPayload | null>(null);
  verifiedModalPayloadRef.current = verifiedModalPayload;
  const openVerifiedModal = useCallback((payload: KioskLoginSuccessPayload) => {
    const startUnlocked = Boolean(voiceMutedRef.current);
    isSpeechDoneRef.current = startUnlocked;
    setIsWelcomeSpeechDone(startUnlocked);
    setRaisedDuringSpeech(false);
    setTwoHandProgress(0);
    setTwoHandDetected(false);
    setLeftHandUpState(false);
    setRightHandUpState(false);
    setOwnerGuardBanner(null);
    probeWindowRef.current = [];
    if (confirmProgressFillRef.current) {
      confirmProgressFillRef.current.style.transform = 'scaleX(0)';
    }
    if (confirmBtnFillRef.current) {
      confirmBtnFillRef.current.style.transform = 'scaleX(0)';
    }
    verifiedModalPayloadRef.current = payload;
    setVerifiedModalPayload(payload);
  }, []);
  useEffect(() => {
    const v = videoRef.current;
    if (!v || !v.srcObject) return;
    if (verifiedModalPayload) {
      try {
        v.pause();
      } catch {
        // Ignore video pause error
      }
    } else if (!cameraPaused) {
      void v.play().catch(() => {});
    }
  }, [verifiedModalPayload, cameraPaused]);
  const [verifiedAnimDone, setVerifiedAnimDone] = useState(false);
  useEffect(() => {
    if (!verifiedModalPayload) {
      setVerifiedAnimDone(false);
      return;
    }
    setVerifiedAnimDone(false);
    const timerId = window.setTimeout(() => {
      setVerifiedAnimDone(true);
    }, KIOSK_AUTH_HOLD_DELAY_MS);
    return () => window.clearTimeout(timerId);
  }, [verifiedModalPayload]);
  const onPatientAuthenticatedRef = useRef(onPatientAuthenticated);
  onPatientAuthenticatedRef.current = onPatientAuthenticated;
  const presenceRef = useRef(new FacePresenceTracker());
  const verifyingLockRef = useRef(false);
  const edgeTelemetryRef = useRef<EdgeKioskTelemetryPayload | null>(null);

  const edgeMatchStreakRef = useRef<{ patientId: string | null; count: number }>({
    patientId: null,
    count: 0,
  });

  const { connected: edgeConnected, telemetry: edgeTelemetry } = useEdgeVisionSocket({
    enabled: !cameraPaused && !staffModalOpen && !showPinModal,
    onPatientMatch: (edgeMatch) => {
      if (verifiedModalPayloadRef.current || verifyingLockRef.current || !edgeMatch.id) return;
      if (
        excludedPatientIdsRef.current.includes(edgeMatch.id) ||
        (edgeMatch.hn && excludedPatientIdsRef.current.includes(edgeMatch.hn))
      ) {
        edgeMatchStreakRef.current = { patientId: null, count: 0 };
        return;
      }
      // Edge Python socket is telemetry-only: NEVER bypass live server /api/biometric/verify
      if (typeof edgeMatch.similarity === 'number' && edgeMatch.similarity >= 0.88) {
        if (edgeMatchStreakRef.current.patientId === edgeMatch.id) {
          edgeMatchStreakRef.current.count += 1;
        } else {
          edgeMatchStreakRef.current = { patientId: edgeMatch.id, count: 1 };
        }
        if (edgeMatchStreakRef.current.count >= 4) {
          edgeMatchStreakRef.current = { patientId: null, count: 0 };
          void refreshBiometricLibraryStats(true);
        }
      }
    },
  });
  edgeTelemetryRef.current = edgeTelemetry;
  const accountCountsRef = useRef<{ valid: number; total: number; legacy: number; filtered: number }>({
    valid: 0,
    total: 0,
    legacy: 0,
    filtered: 0,
  });
  const rescanCooldownUntilRef = useRef<number>(0);
  const unmatchedClearFramesRef = useRef<number>(0);
  const noFaceSinceRef = useRef<number | null>(null);
  const scanDecisionStartAtRef = useRef<number>(0);
  const totalEvaluatedFramesRef = useRef<number>(0);
  const serverRejectCountRef = useRef<number>(0);
  const excludedPatientIdsRef = useRef<string[]>([]);
  const [turningHintPct, setTurningHintPct] = useState<number | null>(null);
  const [ambiguousCloseHint, setAmbiguousCloseHint] = useState(false);
  const mirroredRef = useRef(mirrored);
  mirroredRef.current = mirrored;
  const lastCandidatesSyncAtRef = useRef<number>(0);
  const candidatesLoadedRef = useRef<boolean>(false);
  const humanWarmupReadyRef = useRef<boolean>(false);
  const toastTimerRef = useRef<number | null>(null);
  const pinSubmitTimerRef = useRef<number | null>(null);

  useEffect(() => {
    return () => {
      if (toastTimerRef.current !== null) window.clearTimeout(toastTimerRef.current);
      if (pinSubmitTimerRef.current !== null) window.clearTimeout(pinSubmitTimerRef.current);
    };
  }, []);

  /**
   * Refreshes count-only biometric library telemetry from /api/biometrics/stats.
   * ZERO patient face vectors, names, or HNs are ever downloaded to the Kiosk browser.
   */
  const refreshBiometricLibraryStats = useCallback(async (force = false) => {
    const now = Date.now();
    if (!force && lastCandidatesSyncAtRef.current > 0 && now - lastCandidatesSyncAtRef.current < 15_000) {
      return;
    }
    lastCandidatesSyncAtRef.current = now;
    if (typeof apiClient.getBiometricLibraryStats !== 'function') {
      candidatesLoadedRef.current = true;
      return;
    }
    try {
      const stats: BiometricLibraryStatsResponse = await apiClient.getBiometricLibraryStats(force);
      candidatesLoadedRef.current = true;
      accountCountsRef.current = {
        valid: stats.compatibleTemplatesCount,
        total: stats.totalActivePatients,
        legacy: stats.versionMismatchCount,
        filtered: stats.filteredOutCount,
      };
      setDbAccountCount(stats.compatibleTemplatesCount);
      setTotalAccountCount(stats.totalActivePatients);
      setLegacyAccountCount(stats.versionMismatchCount);
      setSupabaseSyncedFromCloud(stats.dbSynced);
      setSupabaseSyncError(null);
      setLiveDebugHud((prev) => ({
        validAccounts: stats.compatibleTemplatesCount,
        totalAccounts: stats.totalActivePatients,
        legacyAccounts: stats.versionMismatchCount,
        syncedFromSupabase: stats.dbSynced,
        hasEvaluatedMatch: prev?.hasEvaluatedMatch,
        gatePassed: prev?.gatePassed,
        topName: prev?.topName ?? null,
        topPct: prev?.topPct ?? 0,
        centeredPct: prev?.centeredPct,
        streak: prev?.streak ?? 0,
        requiredStreak: KIOSK_FAST_MATCH_STREAK,
        reason: prev?.reason ?? (stats.compatibleTemplatesCount === 0 ? 'NO_TEMPLATES_IN_LIBRARY' : 'READY'),
      }));
    } catch (err) {
      candidatesLoadedRef.current = true;
      const errCode: BiometricVerifyReasonCode =
        err instanceof ApiError && err.status === 403
          ? 'PERMISSION_DENIED'
          : err instanceof ApiError && err.status === 408
            ? 'TIMEOUT'
            : 'SERVER_ERROR';
      setSupabaseSyncedFromCloud(false);
      setSupabaseSyncError(err instanceof Error ? err.message : 'Supabase sync unavailable');
      setStaffDiagnostics((prev) => ({
        reasonCode: errCode,
        evaluatedCandidatesCount: prev?.evaluatedCandidatesCount ?? 0,
        totalActivePatients: prev?.totalActivePatients ?? accountCountsRef.current.total,
        compatibleTemplatesCount: prev?.compatibleTemplatesCount ?? accountCountsRef.current.valid,
        versionMismatchCount: prev?.versionMismatchCount ?? accountCountsRef.current.legacy,
        filteredOutCount: prev?.filteredOutCount ?? accountCountsRef.current.filtered,
        thresholdPercent: prev?.thresholdPercent ?? 89,
        livenessPassed: prev?.livenessPassed ?? false,
        top1Code: prev?.top1Code ?? null,
        top1ScorePercent: prev?.top1ScorePercent ?? 0,
        top1CenteredPercent: prev?.top1CenteredPercent ?? 0,
        top2Code: prev?.top2Code ?? null,
        top2ScorePercent: prev?.top2ScorePercent ?? 0,
        marginPercent: prev?.marginPercent ?? 0,
        gateFailCodes: [errCode],
        probeDim: prev?.probeDim ?? 1024,
        probeEngine: prev?.probeEngine ?? BIOMETRIC_ENGINE_ID,
        dbFresh: false,
        qualityPassedFrames: qualityPassedFramesRef.current,
        lastUpdatedAt: new Date().toISOString(),
      }));
    }
  }, [apiClient]);

  useEffect(() => {
    void refreshBiometricLibraryStats(true);
    if (typeof window === 'undefined') return;
    const onFocus = () => {
      void refreshBiometricLibraryStats(true);
    };
    window.addEventListener('focus', onFocus);
    const pollTimer = window.setInterval(() => {
      if (!document.hidden) {
        void refreshBiometricLibraryStats(false);
      }
    }, 15_000);
    return () => {
      window.removeEventListener('focus', onFocus);
      window.clearInterval(pollTimer);
    };
  }, [refreshBiometricLibraryStats]);

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
    if (toastTimerRef.current !== null) window.clearTimeout(toastTimerRef.current);
    toastTimerRef.current = window.setTimeout(() => {
      toastTimerRef.current = null;
      setToastMsg((prev) => (prev === msg ? null : prev));
    }, 4500);
  }, []);

  useEffect(() => {
    return subscribeBiometricIndex(apiClient, () => {
      void refreshBiometricLibraryStats(true);
    });
  }, [apiClient, refreshBiometricLibraryStats]);

  const handleRestartScan = useCallback(() => {
    verifyingLockRef.current = false;
    serverStreakRef.current = { patientId: null, scores: [] };
    ambiguousSeenInAttemptRef.current = false;
    qualityPassedFramesRef.current = 0;
    profilerRef.current.resetFaceAttempt();
    presenceRef.current.reset();
    bioLivenessRef.current.reset();
    probeWindowRef.current = [];
    unmatchedClearFramesRef.current = 0;
    noFaceSinceRef.current = null;
    scanDecisionStartAtRef.current = 0;
    totalEvaluatedFramesRef.current = 0;
    serverRejectCountRef.current = 0;
    rescanCooldownUntilRef.current = Date.now() + KIOSK_RESCAN_COOLDOWN_MS;
    setSystemErrorReason(null);
    setShowNoFaceGuidance(false);
    setIsNotFoundVoiceSpeaking(false);
    updateReadinessHint(null);
    preferredMirrorModeRef.current = null;
    setIrisHudState({ irisLocked: false, ocularMatchPct: null, ipdPx: null, latencyMs: null });
    cancelAllKioskVoiceSpeech(modalAudioRef.current);
    setBurstCount(0);
    setInspectionInfo(null);
    setLivenessInfo(null);
    setBioReport(null);
    setHudBox(null);
    clearFaceMeshCanvas();
    setLiveMatchHud(null);
    liveOverlayRef.current = null;
    setLiveOverlay(null);
    setLiveDebugHud((prev) =>
      prev
        ? {
            ...prev,
            hasEvaluatedMatch: false,
            gatePassed: false,
            topName: null,
            topPct: 0,
            streak: 0,
          }
        : null,
    );
    setTurningHintPct(null);
    setAmbiguousCloseHint(false);
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
    setOwnerGuardBanner(null);
    setCameraError(null);
    setCameraPaused(false);
    setScanStage('scanning');
    setScanGeneration((g) => g + 1);
    edgeMatchStreakRef.current = { patientId: null, count: 0 };
    void refreshBiometricLibraryStats(true);
  }, [refreshBiometricLibraryStats]);

  const handleRejectWrongMatchAndRescan = useCallback(() => {
    const wrong = verifiedModalPayloadRef.current?.patient;
    if (wrong) {
      const nextExcluded = new Set(excludedPatientIdsRef.current);
      if (wrong.id) nextExcluded.add(wrong.id);
      if (wrong.hn) nextExcluded.add(wrong.hn);
      excludedPatientIdsRef.current = Array.from(nextExcluded);
    }
    handleRestartScan();
  }, [handleRestartScan]);

  useEffect(() => {
    purgeLegacyBiometricCache();
  }, []);

  const isNotFoundStage = scanStage === 'not_found';
  const isSystemErrorStage = scanStage === 'system_error';
  const prevStaffModalOpenRef = useRef(staffModalOpen);

  const onReturnToStandbyRef = useRef(onReturnToStandby);
  onReturnToStandbyRef.current = onReturnToStandby;
  const lastStandbyActivityAtRef = useRef<number>(Date.now());
  const idleStandbyStateRef = useRef({
    hasCallback: Boolean(onReturnToStandby),
    staffModalOpen,
    hasVerifiedModal: Boolean(verifiedModalPayload),
    isFaceDetected: Boolean(hudBox),
    scanStage,
  });
  idleStandbyStateRef.current = {
    hasCallback: Boolean(onReturnToStandby),
    staffModalOpen,
    hasVerifiedModal: Boolean(verifiedModalPayload),
    isFaceDetected: Boolean(hudBox),
    scanStage,
  };
  if (hudBox || staffModalOpen || verifiedModalPayload) {
    lastStandbyActivityAtRef.current = Date.now();
  }

  useEffect(() => {
    if (!onReturnToStandby || typeof window === 'undefined') return;
    lastStandbyActivityAtRef.current = Date.now();
    const touchActivity = () => {
      lastStandbyActivityAtRef.current = Date.now();
    };
    window.addEventListener('pointerdown', touchActivity, { passive: true });
    window.addEventListener('touchstart', touchActivity, { passive: true });
    window.addEventListener('keydown', touchActivity);

    const intervalId = window.setInterval(() => {
      const state = idleStandbyStateRef.current;
      const idleElapsedMs = Date.now() - lastStandbyActivityAtRef.current;
      if (
        shouldAutoReturnToStandby({
          ...state,
          idleElapsedMs,
        })
      ) {
        cancelAllKioskVoiceSpeech(modalAudioRef.current);
        onReturnToStandbyRef.current?.();
      }
    }, 1000);

    return () => {
      window.removeEventListener('pointerdown', touchActivity);
      window.removeEventListener('touchstart', touchActivity);
      window.removeEventListener('keydown', touchActivity);
      window.clearInterval(intervalId);
    };
  }, [Boolean(onReturnToStandby)]);

  useEffect(() => {
    if (!isNotFoundStage || staffModalOpen || voiceMuted || typeof window === 'undefined') {
      setIsNotFoundVoiceSpeaking(false);
      return;
    }

    let cancelled = false;
    const speakText = sanitizeSpeechText(KIOSK_NOT_FOUND_VOICE_PROMPT_TH);
    if (!speakText) return;

    const finishSpeaking = () => {
      if (!cancelled) setIsNotFoundVoiceSpeaking(false);
    };

    const runWebFallback = () => {
      if (cancelled || !('speechSynthesis' in window) || typeof SpeechSynthesisUtterance === 'undefined') {
        finishSpeaking();
        return;
      }
      try {
        const synth = window.speechSynthesis;
        synth.cancel();
        const voices = synth.getVoices();
        const voice = pickSpeechVoice(voices, 'th-TH');
        const utt = new SpeechSynthesisUtterance(speakText);
        utt.lang = voice?.lang || 'th-TH';
        if (voice) utt.voice = voice;
        utt.rate = VERIFIED_MODAL_SPEECH_RATE;
        utt.onstart = () => {
          if (!cancelled) setIsNotFoundVoiceSpeaking(true);
        };
        utt.onend = finishSpeaking;
        utt.onerror = finishSpeaking;
        activeUtteranceRef.current = utt;
        setIsNotFoundVoiceSpeaking(true);
        synth.speak(utt);
      } catch {
        finishSpeaking();
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
        audio.onplaying = () => {
          if (!cancelled) setIsNotFoundVoiceSpeaking(true);
        };
        audio.onended = finishSpeaking;
        audio.onerror = runWebFallback;
        audio.defaultPlaybackRate = VERIFIED_MODAL_SPEECH_RATE;
        audio.src = buildTtsProxyUrl(speakText, 'th');
        audio.playbackRate = VERIFIED_MODAL_SPEECH_RATE;
        const p = audio.play();
        if (p && typeof p.catch === 'function') {
          p.catch(() => {
            if (!cancelled) runWebFallback();
          });
        }
      } catch {
        runWebFallback();
      }
    } else {
      runWebFallback();
    }

    return () => {
      cancelled = true;
      setIsNotFoundVoiceSpeaking(false);
      cancelAllKioskVoiceSpeech(modalAudioRef.current);
    };
  }, [isNotFoundStage, staffModalOpen, voiceMuted]);

  useEffect(() => {
    const wasOpen = prevStaffModalOpenRef.current;
    prevStaffModalOpenRef.current = staffModalOpen;
    if (staffModalOpen && !wasOpen) {
      verifyingLockRef.current = false;
      serverStreakRef.current = { patientId: null, scores: [] };
      qualityPassedFramesRef.current = 0;
      profilerRef.current.resetFaceAttempt();
      presenceRef.current.reset();
      bioLivenessRef.current.reset();
      probeWindowRef.current = [];
      unmatchedClearFramesRef.current = 0;
      scanDecisionStartAtRef.current = 0;
      totalEvaluatedFramesRef.current = 0;
      serverRejectCountRef.current = 0;
      setSystemErrorReason(null);
      setScanStage('scanning');
      setHudBox(null);
      updateReadinessHint(null);
      clearFaceMeshCanvas();
      setLiveMatchHud(null);
      liveOverlayRef.current = null;
      setLiveOverlay(null);
      setLiveDebugHud((prev) =>
        prev
          ? {
              ...prev,
              hasEvaluatedMatch: false,
              gatePassed: false,
              topName: null,
              topPct: 0,
              streak: 0,
            }
          : null,
      );
      setTurningHintPct(null);
      setBurstCount(0);
    } else if (!staffModalOpen && wasOpen) {
      handleRestartScan();
    }
  }, [staffModalOpen, handleRestartScan]);

  useEffect(() => {
    if (cameraPaused || staffModalOpen || isNotFoundStage || isSystemErrorStage || showPinModal) {
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
    let lastDetectTs = 0;
    let lastPoseDetectTs = 0;
    let lastProcessedVideoTime = -1;
    let lastProcessedWallMs = 0;
    let lastTelemetryUiSyncAt = 0;
    let lastBioVerifiedUi = false;
    let lastSpoofUi = false;
    let lastBioModeUi = '';
    let hudBoxMounted = false;
    let asyncMatchInFlight = false;
    let lastKnownRoiBox: { x: number; y: number; width: number; height: number } | null = null;
    let roiSweepIdx = 0;
    let unreadyFrameStreak = 0;
    let blurryFrameStreak = 0;
    let bioVerifiedUnmatchedStartAt = 0;
    let isLastVerifyAmbiguous = false;
    let mirrorProbedForCurrentFace = false;
    let lastAlignSampleAt = 0;
    let cachedAlignCrop: ImageData | null = null;
    let cachedContextFrame: ImageData | null = null;
    let cachedAlignContrastSharpness: number | undefined = undefined;
    let smoothedYawDeg = 0;
    let smoothedPitchDeg = 0;
    let hasSmoothedPose = false;
    let lastVisualRafAt = 0;
    let smoothedVisualBox: SmoothedBoxState | null = null;
    let targetVisualBox: SmoothedBoxState | null = null;
    let smoothedVisualLandmarks: { x: number; y: number; z?: number }[] | null = null;
    let targetVisualLandmarks: { x: number; y: number; z?: number }[] | null = null;
    let lastVisualOverlayColor: 'rose' | 'amber' | 'blue' | 'emerald' = 'blue';
    let lastVisualRenderedW = 0;
    let lastVisualRenderedH = 0;
    let lastVisualOffsetX = 0;
    let lastVisualOffsetY = 0;
    let lastVisualCw = 0;
    let lastVisualCh = 0;
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
    let confirmFaceCycle = 0;
    let lastConfirmedPatientId: string | null = null;
    let cachedConfirmOwnerVerified = false;
    let cachedConfirmFaceCount = 0;
    let cachedConfirmFaceBox: OwnerFaceBoxMetrics | null = null;
    let cachedConfirmExtraFaceBoxes: OwnerFaceBoxMetrics[] = [];
    let confirmSnapCanvas: HTMLCanvasElement | null = null;
    let confirmEmbedInFlight = false;
    let lastOwnerGuardKey = '';
    let gestureWarmupTimerId: number | null = null;
    let gestureWarmupStarted = false;
    let videoFrameCallbackId: number | null = null;
    let hasVideoFrameCallbackApi = false;
    let hasFreshVideoFrame = false;
    let lastOverlayDrawAt = 0;
    let lastPerfHudSyncAt = 0;
    let firstFrameWallMs = 0;
    let exposureSettled = false;
    const videoEl = videoRef.current;

    const stopMediaAndDetectors = () => {
      cancelAnimationFrame(rafId);
      if (videoFrameCallbackId !== null && videoEl) {
        const vWithCancel = videoEl as HTMLVideoElement & {
          cancelVideoFrameCallback?: (id: number) => void;
        };
        if (typeof vWithCancel.cancelVideoFrameCallback === 'function') {
          try {
            vWithCancel.cancelVideoFrameCallback(videoFrameCallbackId);
          } catch {
            // Ignore cancelVideoFrameCallback cleanup errors
          }
        }
        videoFrameCallbackId = null;
      }
      if (gestureWarmupTimerId !== null) {
        window.clearTimeout(gestureWarmupTimerId);
        gestureWarmupTimerId = null;
      }
      // Shared cached Face/ROI/Hand/Pose landmarkers stay alive in RAM for 0ms next-scan resume
      landmarker = null;
      roiLandmarker = null;
      handLandmarker = null;
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

    async function warmupGestureDetectors() {
      setTwoHandDetectorReady(false);
      const [handResult, poseResult] = await Promise.allSettled([
        getCachedHandLandmarker({ numHands: 4, minConfidence: 0.35 }),
        getCachedPoseLandmarker({ numPoses: 1 }),
      ]);
      if (!active) return;
      let anyReady = false;
      if (handResult.status === 'fulfilled') {
        handLandmarker = handResult.value.instance;
        anyReady = true;
      }
      if (poseResult.status === 'fulfilled') {
        poseLandmarker = poseResult.value.instance;
        anyReady = true;
      }
      if (anyReady) {
        setTwoHandDetectorReady(true);
      }
    }

    async function startFacePipeline() {
      try {
        const sessionStartMs = performance.now();
        profilerRef.current.startSession(sessionStartMs);
        const wasPrewarmed =
          getVisionWarmupTelemetry().prewarmCompleted && getHumanEngineTelemetry().ready;

        setTwoHandProgress(0);
        setTwoHandDetected(false);
        setTwoHandCameraReady(false);
        // Eagerly warm up shared @vladmandic/human faceres (1024-D) engine & refresh count-only library stats
        void warmupBiometricEngine().then((ready) => {
          humanWarmupReadyRef.current = Boolean(ready);
          const vTel = getVisionWarmupTelemetry();
          const hTel = getHumanEngineTelemetry();
          profilerRef.current.recordModelsReady({
            modelWarmupMs: wasPrewarmed
              ? 0
              : Math.max(vTel.primaryFaceWarmupMs, hTel.warmupMs),
            wasPrewarmed,
            faceDelegate: vTel.primaryFaceDelegate,
            humanBackend: hTel.backend,
          });
        });
        void refreshBiometricLibraryStats();
        if (typeof navigator === 'undefined' || !navigator.mediaDevices?.getUserMedia) {
          setCameraError('Browser does not support camera access');
          setCameraPaused(true);
          return;
        }
        // Open remembered camera directly (skipPostEnumerate avoids blocking on 5-camera enumeration)
        // in parallel with shared pre-warmed 478-pt FaceLandmarker for fastest Frame-0 start
        const modelAcquireStartMs = performance.now();
        const [opened, primaryResult] = await Promise.all([
          openBestAvailableCameraStream(selectedCameraId, { skipPostEnumerate: true }),
          getCachedPrimaryFaceLandmarker().catch(() => null),
        ]);
        const primaryAcquireMs = performance.now() - modelAcquireStartMs;
        if (!active) {
          opened.stream.getTracks().forEach((track) => track.stop());
          return;
        }
        profilerRef.current.recordCameraOpen({
          openDurationMs: opened.openDurationMs,
          probeCount: opened.probeCount,
          usedRememberedDevice: opened.usedRememberedDevice,
        });
        const visionTel = getVisionWarmupTelemetry();
        const humanTel = getHumanEngineTelemetry();
        profilerRef.current.recordModelsReady({
          modelWarmupMs: wasPrewarmed
            ? 0
            : Math.max(visionTel.primaryFaceWarmupMs, humanTel.warmupMs, primaryAcquireMs),
          wasPrewarmed,
          faceDelegate: visionTel.primaryFaceDelegate,
          humanBackend: humanTel.backend,
        });
        stream = opened.stream;
        if (opened.devices.length > 0) {
          setAvailableCameras(opened.devices);
        }
        if (opened.activeDeviceId) {
          setActiveCameraId(opened.activeDeviceId);
        }
        if (!active || !videoRef.current) {
          stream.getTracks().forEach((track) => track.stop());
          stream = null;
          return;
        }
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        if (!active) return;
        setCameraError(null);

        // Refresh camera list asynchronously AFTER video starts playing so device labels update
        // without adding any startup latency to Frame 1
        void listClinicalCameraDevices().then((list) => {
          if (active && list.length > 0) {
            setAvailableCameras(list);
          }
        });

        // Subscribe to decoded video frames when requestVideoFrameCallback is available
        // so FaceLandmarker processes each real camera frame exactly once without duplicate inference
        const vWithRvfc = videoRef.current as HTMLVideoElement & {
          requestVideoFrameCallback?: (cb: () => void) => number;
        };
        if (typeof vWithRvfc.requestVideoFrameCallback === 'function') {
          hasVideoFrameCallbackApi = true;
          const onVideoFrame = () => {
            if (!active) return;
            if (hasFreshVideoFrame) {
              profilerRef.current.recordSkippedStaleFrame();
            }
            hasFreshVideoFrame = true;
            videoFrameCallbackId = vWithRvfc.requestVideoFrameCallback!(onVideoFrame);
          };
          videoFrameCallbackId = vWithRvfc.requestVideoFrameCallback(onVideoFrame);
        }

        if (!primaryResult) {
          setCameraError('ไม่สามารถเริ่มตัวตรวจจับใบหน้า (FaceLandmarker) ได้');
          stopMediaAndDetectors();
          return;
        }
        landmarker = primaryResult;

        // Warm secondary ROI landmarker asynchronously in background;
        // do NOT warm Hand/Pose gesture detectors mid-scan (they load on verified modal / Frame 2 prefetch)
        // so active Face+Iris scanning has zero WebGL/WASM contention.
        void getCachedRoiFaceLandmarker().then((det) => {
          if (active && det) {
            roiLandmarker = det;
          }
        });

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
        const FACE_MESH_CONTOURS: ReadonlyArray<readonly number[]> = [
          // Face oval contour
          [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10],
          // Left & right eyes
          [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 33],
          [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466, 263],
          // Left & right eyebrows
          [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
          [300, 293, 334, 296, 336, 285, 295, 282, 283, 276],
          // Nose bridge & base
          [168, 6, 197, 195, 5, 4, 1],
          [98, 97, 2, 326, 327],
          // Outer lip contour
          [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61],
        ];

        let shouldContinueLoop = true;
        const ROI_SWEEP_ZONES = [
          { x: 0.24, y: 0.06, width: 0.52, height: 0.62 },
          { x: 0.0, y: 0.06, width: 0.54, height: 0.68 },
          { x: 0.46, y: 0.06, width: 0.54, height: 0.68 },
          { x: 0.12, y: 0.22, width: 0.76, height: 0.74 },
        ] as const;

        const runAsyncBiometricMatchStep = async (params: {
          snapCanvas: HTMLCanvasElement;
          snapSharpness: number;
          snapPatch384: number[] | null;
          snapRawBox: { x: number; y: number; width: number; height: number } | null;
          snapLandmarks: { x: number; y: number; z?: number }[];
          snapOcularIris: OcularIrisCaptureResult | null;
          snapIrisMs?: number;
          snapVideoW: number;
          snapVideoH: number;
          snapYawDeg: number;
          snapPitchDeg: number;
          snapFaceHeightRatio: number;
          snapBioVerified: boolean;
          snapBioNonRigidScore: number;
          snapBioRppgScore: number;
          snapBioSpoofScore: number;
          snapBioBlinkCount: number;
          snapBioIrisLocked: boolean;
          snapIsSpoof: boolean;
          snapNow: number;
        }) => {
          if (!active || !shouldContinueLoop || verifiedModalPayloadRef.current || verifyingLockRef.current) {
            return;
          }
          asyncMatchInFlight = true;
          const stepStartMs = performance.now();
          try {
            const canonicalYawDeg = -params.snapYawDeg;
            // Re-verify strict frontal pose gate & motion sharpness BEFORE spending GPU inference time
            if (
              Math.abs(canonicalYawDeg) > STRICT_LOGIN_MAX_YAW_DEG ||
              Math.abs(params.snapPitchDeg) > STRICT_LOGIN_MAX_PITCH_DEG
            ) {
              probeWindowRef.current = [];
              serverStreakRef.current = { patientId: null, scores: [] };
              setBurstCount(0);
              setLiveMatchHud(null);
              return;
            }
            const frameSharpness = params.snapSharpness;
            if (frameSharpness < 2.8) {
              blurryFrameStreak += 1;
              if (blurryFrameStreak >= 2) {
                updateReadinessHint(
                  resolveKioskReadinessGuidance({
                    isSpoof: params.snapIsSpoof,
                    isBioVerified: params.snapBioVerified,
                    yawDeg: params.snapYawDeg,
                    pitchDeg: params.snapPitchDeg,
                    inspection: null,
                    blurryFrameStreak,
                    lang: langRef.current,
                  }),
                );
              }
              if (blurryFrameStreak >= 4) {
                serverStreakRef.current = { patientId: null, scores: [] };
                setBurstCount(0);
              }
              return;
            }
            blurryFrameStreak = 0;

            const useMirroredFirst = preferredMirrorModeRef.current === 'mirrored';
            const primaryInputCanvas = useMirroredFirst
              ? (createMirroredCanvasSnapshot(params.snapCanvas, mirrorSnapCanvasRef.current) ?? params.snapCanvas)
              : params.snapCanvas;
            if (useMirroredFirst && primaryInputCanvas !== params.snapCanvas) {
              mirrorSnapCanvasRef.current = primaryInputCanvas;
            }

            // Single Shared Neural Extractor (@vladmandic/human faceres 1024-D) — ZERO fake geometric/pixel fallbacks
            const embedStartMs = performance.now();
            let extracted = await extractCanonicalFaceDescriptor(primaryInputCanvas);
            humanWarmupReadyRef.current = true;
            if (!active || !shouldContinueLoop || verifiedModalPayloadRef.current) {
              return;
            }
            if (!extracted.embedding && videoRef.current && videoRef.current.readyState >= 2 && videoRef.current.videoWidth > 0) {
              // Fast bounded 288x288 (1.72x padded, backlight-normalized) fallback instead of slow raw 1080p full-frame stall
              if (!fallbackSnapCanvasRef.current) {
                fallbackSnapCanvasRef.current = document.createElement('canvas');
              }
              const wideCrop = captureWideCanonicalFaceFallback(
                videoRef.current,
                params.snapRawBox,
                fallbackSnapCanvasRef.current,
                params.snapLandmarks,
              );
              const fallbackInput =
                wideCrop && useMirroredFirst
                  ? (createMirroredCanvasSnapshot(wideCrop, mirrorSnapCanvasRef.current) ?? wideCrop)
                  : (wideCrop ?? videoRef.current);
              const fallbackExtracted = await extractCanonicalFaceDescriptor(fallbackInput);
              if (!active || !shouldContinueLoop || verifiedModalPayloadRef.current) {
                return;
              }
              if (fallbackExtracted.embedding) {
                extracted = fallbackExtracted;
              }
            }
            const humanEmbedMs = performance.now() - embedStartMs;
            if (!extracted.embedding) {
              profilerRef.current.recordEmbedStep({
                irisMs: params.snapIrisMs ?? 0,
                humanEmbedMs,
                localMatchMs: 0,
              });
              return;
            }
            qualityPassedFramesRef.current += 1;
            profilerRef.current.recordEmbedStep({
              irisMs: params.snapIrisMs ?? 0,
              humanEmbedMs,
              localMatchMs: 0,
            });

            const rollDeg = computeEyeRollAngleDeg(
              params.snapLandmarks,
              params.snapVideoW,
              params.snapVideoH,
            );
            const effectiveYawDeg = useMirroredFirst ? -canonicalYawDeg : canonicalYawDeg;

            const rawGeom = extractGeometricAnthropometricRatios(params.snapLandmarks);
            const rawPatch384 = params.snapPatch384;
            const patch384 =
              useMirroredFirst && rawPatch384
                ? flipPatch384Horizontally(rawPatch384)
                : rawPatch384;
            const ratios12 = computeCraniofacialRatios12(
              params.snapLandmarks,
              (params.snapVideoW || 1280) / (params.snapVideoH || 720),
            );
            const rawOcularPatch256 = params.snapOcularIris?.patch256 ?? null;
            const ocularPatch256 =
              useMirroredFirst && rawOcularPatch256
                ? flipOcularIrisPatch256Horizontally(rawOcularPatch256)
                : rawOcularPatch256;
            const rawIrisRatios8 = params.snapOcularIris?.ratios8 ?? null;
            const irisRatios8 =
              useMirroredFirst && rawIrisRatios8
                ? flipIrisOcularRatios8Horizontally(rawIrisRatios8)
                : rawIrisRatios8;
            const geometricRatios: GeometricAnthropometricRatios | undefined =
              rawGeom || patch384 || ratios12 || ocularPatch256 || irisRatios8
                ? {
                    eyeToNoseRatio: rawGeom?.eyeToNoseRatio ?? 1,
                    faceWidthToUpperHeight: rawGeom?.faceWidthToUpperHeight ?? 1,
                    mouthToEyeRatio: rawGeom?.mouthToEyeRatio ?? 1,
                    noseToChinRatio: rawGeom?.noseToChinRatio ?? 1,
                    ...(patch384 ? { appearancePatch384: patch384 } : {}),
                    ...(ratios12 ? { craniofacialRatios12: ratios12 } : {}),
                    ...(ocularPatch256 ? { ocularIrisPatch256: ocularPatch256 } : {}),
                    ...(irisRatios8 ? { irisOcularRatios8: irisRatios8 } : {}),
                  }
                : undefined;

            // Sharpness-weighted rolling 4-frame centroid across consecutive same-person frontal frames within 1.8s
            let activeEmbedding = extracted.embedding;
            let activeGeometricRatios = geometricRatios;
            if (params.snapIsSpoof) {
              probeWindowRef.current = [];
            } else {
              // Flush rolling buffer immediately if inter-frame embedding jumps (different person entered camera)
              const prevLatest = probeWindowRef.current[probeWindowRef.current.length - 1];
              const samePersonContinuity =
                !prevLatest ||
                prevLatest.embedding.length !== extracted.embedding.length ||
                cosineSimilarityNormalized(extracted.embedding, prevLatest.embedding) >= 0.86;
              if (!samePersonContinuity) {
                ambiguousSeenInAttemptRef.current = false;
              }
              const baseWindow = samePersonContinuity ? probeWindowRef.current : [];
              const validWindow = baseWindow.filter(
                (item) =>
                  params.snapNow - item.ts <= 1800 &&
                  item.embedding.length === extracted.embedding!.length &&
                  Math.abs(item.yawDeg - effectiveYawDeg) <= 12 &&
                  Math.abs(item.pitchDeg - params.snapPitchDeg) <= 10,
              );
              validWindow.push({
                ts: params.snapNow,
                embedding: extracted.embedding,
                sharpness: frameSharpness,
                appearancePatch384: patch384 ?? undefined,
                craniofacialRatios12: ratios12 ?? undefined,
                ocularIrisPatch256: ocularPatch256 ?? undefined,
                irisOcularRatios8: irisRatios8 ?? undefined,
                yawDeg: effectiveYawDeg,
                pitchDeg: params.snapPitchDeg,
              });
              const rolling = validWindow.slice(-4);
              probeWindowRef.current = rolling;

              if (rolling.length >= 2) {
                const dim = extracted.embedding.length;
                const embSum = new Array<number>(dim).fill(0);
                let totalWeight = 0;
                for (const item of rolling) {
                  const weight = 1 + Math.min(2, Math.max(0, item.sharpness) / 60);
                  totalWeight += weight;
                  for (let i = 0; i < dim; i += 1) {
                    embSum[i] += item.embedding[i] * weight;
                  }
                }
                activeEmbedding = l2Normalize(embSum);

                const p384Len = patch384?.length ?? 0;
                const averagedPatch =
                  p384Len === 384 &&
                  totalWeight > 0 &&
                  rolling.every((item) => item.appearancePatch384?.length === 384)
                    ? rolling[0].appearancePatch384!.map(
                        (_, idx) =>
                          rolling.reduce(
                            (acc, item) =>
                              acc +
                              item.appearancePatch384![idx] *
                                (1 + Math.min(2, Math.max(0, item.sharpness) / 60)),
                            0,
                          ) / totalWeight,
                      )
                    : geometricRatios?.appearancePatch384;
                const r12Len = ratios12?.length ?? 0;
                const averagedRatios12 =
                  r12Len === 12 &&
                  totalWeight > 0 &&
                  rolling.every((item) => item.craniofacialRatios12?.length === 12)
                    ? rolling[0].craniofacialRatios12!.map(
                        (_, idx) =>
                          rolling.reduce(
                            (acc, item) =>
                              acc +
                              item.craniofacialRatios12![idx] *
                                (1 + Math.min(2, Math.max(0, item.sharpness) / 60)),
                            0,
                          ) / totalWeight,
                      )
                    : geometricRatios?.craniofacialRatios12;
                const averagedOcularPatch =
                  ocularPatch256?.length === 256 &&
                  totalWeight > 0 &&
                  rolling.every((item) => item.ocularIrisPatch256?.length === 256)
                    ? (normalizeOcularIrisPatch256(
                        rolling[0].ocularIrisPatch256!.map(
                          (_, idx) =>
                            rolling.reduce(
                              (acc, item) =>
                                acc +
                                item.ocularIrisPatch256![idx] *
                                  (1 + Math.min(2, Math.max(0, item.sharpness) / 60)),
                              0,
                            ) / totalWeight,
                        ),
                      ) ?? geometricRatios?.ocularIrisPatch256)
                    : geometricRatios?.ocularIrisPatch256;
                const averagedIrisRatios8 =
                  irisRatios8?.length === 8 &&
                  totalWeight > 0 &&
                  rolling.every((item) => item.irisOcularRatios8?.length === 8)
                    ? rolling[0].irisOcularRatios8!.map(
                        (_, idx) =>
                          Number(
                            (
                              rolling.reduce(
                                (acc, item) =>
                                  acc +
                                  item.irisOcularRatios8![idx] *
                                    (1 + Math.min(2, Math.max(0, item.sharpness) / 60)),
                                0,
                              ) / totalWeight
                            ).toFixed(4),
                          ),
                      )
                    : geometricRatios?.irisOcularRatios8;
                if (geometricRatios) {
                  activeGeometricRatios = {
                    ...geometricRatios,
                    ...(averagedPatch ? { appearancePatch384: averagedPatch } : {}),
                    ...(averagedRatios12 ? { craniofacialRatios12: averagedRatios12 } : {}),
                    ...(averagedOcularPatch ? { ocularIrisPatch256: averagedOcularPatch } : {}),
                    ...(averagedIrisRatios8 ? { irisOcularRatios8: averagedIrisRatios8 } : {}),
                  };
                }
              }
            }

            setIrisHudState({
              irisLocked: Boolean(params.snapBioIrisLocked || params.snapOcularIris?.patch256),
              ocularMatchPct: null,
              ipdPx: params.snapOcularIris?.ipdPx ?? null,
              latencyMs: Math.max(1, Math.round(performance.now() - stepStartMs)),
            });

            if (params.snapIsSpoof) {
              serverStreakRef.current = { patientId: null, scores: [] };
              bioVerifiedUnmatchedStartAt = 0;
              unmatchedClearFramesRef.current = 0;
              scanDecisionStartAtRef.current = 0;
              totalEvaluatedFramesRef.current = 0;
              setBurstCount(0);
              setLiveMatchHud(null);
              setTurningHintPct(null);
              setScanStage('scanning');
              return;
            }

            // Wait for 3D/EAR/Iris biological liveness BEFORE calling server verify (rolling centroid stays pre-warmed)
            if (!params.snapBioVerified) {
              setStaffDiagnostics((prev) => ({
                reasonCode: 'LIVENESS_FAILED',
                evaluatedCandidatesCount: prev?.evaluatedCandidatesCount ?? 0,
                totalActivePatients: accountCountsRef.current.total,
                compatibleTemplatesCount: accountCountsRef.current.valid,
                versionMismatchCount: accountCountsRef.current.legacy,
                filteredOutCount: accountCountsRef.current.filtered,
                thresholdPercent: prev?.thresholdPercent ?? 89,
                livenessPassed: false,
                top1Code: prev?.top1Code ?? null,
                top1ScorePercent: prev?.top1ScorePercent ?? 0,
                top1CenteredPercent: prev?.top1CenteredPercent ?? 0,
                top2Code: prev?.top2Code ?? null,
                top2ScorePercent: prev?.top2ScorePercent ?? 0,
                marginPercent: prev?.marginPercent ?? 0,
                gateFailCodes: ['LIVENESS_PENDING'],
                probeDim: activeEmbedding.length,
                probeEngine: BIOMETRIC_ENGINE_ID,
                dbFresh: prev?.dbFresh ?? true,
                qualityPassedFrames: qualityPassedFramesRef.current,
                lastUpdatedAt: new Date().toISOString(),
              }));
              return;
            }

            const livenessEvidence = {
              biologicalVerified: params.snapBioVerified,
              blinkCount: params.snapBioBlinkCount,
              spoofRisk: params.snapBioSpoofScore,
              nonRigidScore: params.snapBioNonRigidScore,
              rppgScore: params.snapBioRppgScore,
              faceHeightRatio: params.snapFaceHeightRatio,
              yawDeg: effectiveYawDeg,
              pitchDeg: params.snapPitchDeg,
              rollDeg,
              engine: BIOMETRIC_ENGINE_ID,
              ...(activeGeometricRatios ? { geometricRatios: activeGeometricRatios } : {}),
              ...(excludedPatientIdsRef.current.length > 0
                ? { excludedPatientIds: excludedPatientIdsRef.current }
                : {}),
            };

            // Live Server-Side Verification (ZERO other patients' vectors ever downloaded to the Kiosk browser!)
            if (!verifyingLockRef.current) {
              updateReadinessHint(null);
              let verifyRes: KioskFaceVerifyResponse;
              let usedEmbedding = activeEmbedding;
              let usedGeometricRatios = activeGeometricRatios;
              try {
                const verifyStartMs = performance.now();
                verifyRes = await apiClient.verifyFaceLogin(activeEmbedding, livenessEvidence);
                profilerRef.current.recordServerVerify(performance.now() - verifyStartMs, 0);

                if (!active || !shouldContinueLoop || verifiedModalPayloadRef.current) {
                  return;
                }

                // One-time alternate horizontal orientation probe per face session if near threshold
                const diagTop1 = verifyRes.diagnostics?.top1ScorePercent ?? 0;
                if (
                  !verifyRes.matched &&
                  verifyRes.reasonCode === 'BELOW_THRESHOLD' &&
                  !mirrorProbedForCurrentFace &&
                  diagTop1 >= 72
                ) {
                  mirrorProbedForCurrentFace = true;
                  const altCanvas = useMirroredFirst
                    ? params.snapCanvas
                    : createMirroredCanvasSnapshot(params.snapCanvas, mirrorSnapCanvasRef.current);
                  if (!useMirroredFirst && altCanvas) {
                    mirrorSnapCanvasRef.current = altCanvas;
                  }
                  if (altCanvas) {
                    const mirroredExt = await extractCanonicalFaceDescriptor(altCanvas);
                    if (!active || !shouldContinueLoop || verifiedModalPayloadRef.current) {
                      return;
                    }
                    if (mirroredExt.embedding) {
                      const flippedPatch = activeGeometricRatios?.appearancePatch384
                        ? flipPatch384Horizontally(activeGeometricRatios.appearancePatch384)
                        : undefined;
                      const flippedOcular = activeGeometricRatios?.ocularIrisPatch256
                        ? flipOcularIrisPatch256Horizontally(activeGeometricRatios.ocularIrisPatch256)
                        : undefined;
                      const flippedIris8 = activeGeometricRatios?.irisOcularRatios8
                        ? flipIrisOcularRatios8Horizontally(activeGeometricRatios.irisOcularRatios8)
                        : undefined;
                      const mirroredRatios: GeometricAnthropometricRatios | undefined =
                        activeGeometricRatios
                          ? {
                              ...activeGeometricRatios,
                              ...(flippedPatch ? { appearancePatch384: flippedPatch } : {}),
                              ...(flippedOcular ? { ocularIrisPatch256: flippedOcular } : {}),
                              ...(flippedIris8 ? { irisOcularRatios8: flippedIris8 } : {}),
                            }
                          : undefined;
                      const mirrorVerifyStart = performance.now();
                      const mirrorRes = await apiClient.verifyFaceLogin(mirroredExt.embedding, {
                        ...livenessEvidence,
                        yawDeg: -effectiveYawDeg,
                        ...(mirroredRatios ? { geometricRatios: mirroredRatios } : {}),
                        mirroredProbeVector: activeEmbedding,
                      });
                      profilerRef.current.recordServerVerify(performance.now() - mirrorVerifyStart, 0);
                      if (!active || !shouldContinueLoop || verifiedModalPayloadRef.current) {
                        return;
                      }
                      if (mirrorRes.matched) {
                        preferredMirrorModeRef.current = useMirroredFirst ? 'normal' : 'mirrored';
                        verifyRes = mirrorRes;
                        usedEmbedding = mirroredExt.embedding;
                        usedGeometricRatios = mirroredRatios;
                      }
                    }
                  }
                }
              } catch (err) {
                if (!active || !shouldContinueLoop || verifiedModalPayloadRef.current) {
                  return;
                }
                const errReason: BiometricVerifyReasonCode =
                  err instanceof ApiError && err.status === 403
                    ? 'PERMISSION_DENIED'
                    : err instanceof ApiError && (err.status === 408 || err.code === 'TIMEOUT')
                      ? 'TIMEOUT'
                      : 'SERVER_ERROR';
                serverStreakRef.current = { patientId: null, scores: [] };
                setBurstCount(0);
                setLiveMatchHud(null);
                setAmbiguousCloseHint(false);
                setSystemErrorReason(errReason);
                setStaffDiagnostics((prev) => ({
                  reasonCode: errReason,
                  evaluatedCandidatesCount: prev?.evaluatedCandidatesCount ?? 0,
                  totalActivePatients: accountCountsRef.current.total,
                  compatibleTemplatesCount: accountCountsRef.current.valid,
                  versionMismatchCount: accountCountsRef.current.legacy,
                  filteredOutCount: accountCountsRef.current.filtered,
                  thresholdPercent: prev?.thresholdPercent ?? 89,
                  livenessPassed: params.snapBioVerified,
                  top1Code: prev?.top1Code ?? null,
                  top1ScorePercent: prev?.top1ScorePercent ?? 0,
                  top1CenteredPercent: prev?.top1CenteredPercent ?? 0,
                  top2Code: prev?.top2Code ?? null,
                  top2ScorePercent: prev?.top2ScorePercent ?? 0,
                  marginPercent: prev?.marginPercent ?? 0,
                  gateFailCodes: [errReason],
                  probeDim: activeEmbedding.length,
                  probeEngine: BIOMETRIC_ENGINE_ID,
                  dbFresh: false,
                  qualityPassedFrames: qualityPassedFramesRef.current,
                  lastUpdatedAt: new Date().toISOString(),
                }));
                shouldContinueLoop = false;
                setScanStage('system_error');
                return;
              }

              setLastVerifyResult(verifyRes);

              const resolvedReason: BiometricVerifyReasonCode =
                verifyRes.reasonCode ??
                (verifyRes.matched
                  ? 'MATCHED'
                  : verifyRes.error === 'RATE_LIMITED' || verifyRes.httpStatus === 429
                    ? 'RATE_LIMITED'
                    : verifyRes.error === 'BIOMETRIC_SPOOF_DETECTED' ||
                        verifyRes.failureReasons?.includes('BIOLOGICAL_SPOOF_BLOCKED')
                      ? 'LIVENESS_FAILED'
                      : verifyRes.error === 'AMBIGUOUS_CLOSE_CANDIDATES'
                        ? 'AMBIGUOUS'
                        : 'BELOW_THRESHOLD');

              if (verifyRes.diagnostics) {
                accountCountsRef.current = {
                  valid: verifyRes.diagnostics.compatibleTemplatesCount,
                  total: verifyRes.diagnostics.totalActivePatients,
                  legacy: verifyRes.diagnostics.versionMismatchCount,
                  filtered: verifyRes.diagnostics.filteredOutCount,
                };
                setDbAccountCount(verifyRes.diagnostics.compatibleTemplatesCount);
                setTotalAccountCount(verifyRes.diagnostics.totalActivePatients);
                setLegacyAccountCount(verifyRes.diagnostics.versionMismatchCount);
                setSupabaseSyncedFromCloud(verifyRes.diagnostics.dbFresh);
                setStaffDiagnostics({
                  ...verifyRes.diagnostics,
                  qualityPassedFrames: qualityPassedFramesRef.current,
                  lastUpdatedAt: new Date().toISOString(),
                });
              } else {
                setStaffDiagnostics((prev) => ({
                  reasonCode: resolvedReason,
                  evaluatedCandidatesCount: prev?.evaluatedCandidatesCount ?? accountCountsRef.current.valid,
                  totalActivePatients: accountCountsRef.current.total,
                  compatibleTemplatesCount: accountCountsRef.current.valid,
                  versionMismatchCount: accountCountsRef.current.legacy,
                  filteredOutCount: accountCountsRef.current.filtered,
                  thresholdPercent: prev?.thresholdPercent ?? 89,
                  livenessPassed: params.snapBioVerified,
                  top1Code: prev?.top1Code ?? null,
                  top1ScorePercent: verifyRes.confidencePercent ?? 0,
                  top1CenteredPercent: prev?.top1CenteredPercent ?? 0,
                  top2Code: prev?.top2Code ?? null,
                  top2ScorePercent: prev?.top2ScorePercent ?? 0,
                  marginPercent: prev?.marginPercent ?? 0,
                  gateFailCodes: verifyRes.failureReasons ?? [resolvedReason],
                  probeDim: activeEmbedding.length,
                  probeEngine: BIOMETRIC_ENGINE_ID,
                  dbFresh: true,
                  qualityPassedFrames: qualityPassedFramesRef.current,
                  lastUpdatedAt: new Date().toISOString(),
                }));
              }

              // 1. Rate-Limited / System Error -> NEVER display "Account Not Found"!
              if (
                resolvedReason === 'RATE_LIMITED' ||
                resolvedReason === 'SERVER_ERROR' ||
                resolvedReason === 'TIMEOUT' ||
                resolvedReason === 'PERMISSION_DENIED'
              ) {
                serverStreakRef.current = { patientId: null, scores: [] };
                setBurstCount(0);
                setLiveMatchHud(null);
                setAmbiguousCloseHint(false);
                setSystemErrorReason(resolvedReason);
                shouldContinueLoop = false;
                setScanStage('system_error');
                return;
              }

              // 2. Biological Liveness Failed on server -> show spoof warning, NEVER "not_found"
              if (resolvedReason === 'LIVENESS_FAILED') {
                serverStreakRef.current = { patientId: null, scores: [] };
                setBurstCount(0);
                setLiveMatchHud(null);
                setAmbiguousCloseHint(false);
                const spoofOverlay = resolveKioskFaceBoundingBoxOverlay({
                  isSpoof: true,
                  isBioVerified: false,
                  rawCosine: 0,
                  streak: 0,
                  requiredStreak: KIOSK_FAST_MATCH_STREAK,
                  confirmedCandidateName: null,
                  distanceM: bioReportRef.current?.estimatedDistanceM ?? 1.8,
                });
                liveOverlayRef.current = spoofOverlay;
                setLiveOverlay(spoofOverlay);
                setScanStage('scanning');
                return;
              }

              // 3. Ambiguous Close Candidates -> calm neutral prompt, NEVER jump to "not_found"
              if (resolvedReason === 'AMBIGUOUS') {
                isLastVerifyAmbiguous = true;
                ambiguousSeenInAttemptRef.current = true;
                bioVerifiedUnmatchedStartAt = 0;
                unmatchedClearFramesRef.current = 0;
                serverStreakRef.current = { patientId: null, scores: [] };
                setBurstCount(0);
                setLiveMatchHud(null);
                setAmbiguousCloseHint(true);
                const ambigOverlay = resolveKioskFaceBoundingBoxOverlay({
                  isSpoof: false,
                  isBioVerified: true,
                  rawCosine: (verifyRes.confidencePercent ?? 0) / 100,
                  hasEvaluatedMatch: true,
                  gatePassed: false,
                  streak: 0,
                  requiredStreak: KIOSK_FAST_MATCH_STREAK,
                  confirmedCandidateName: null,
                  distanceM: bioReportRef.current?.estimatedDistanceM ?? 1.8,
                  isAmbiguous: true,
                });
                liveOverlayRef.current = ambigOverlay;
                setLiveOverlay(ambigOverlay);
                setLiveDebugHud({
                  validAccounts: accountCountsRef.current.valid,
                  totalAccounts: accountCountsRef.current.total,
                  legacyAccounts: accountCountsRef.current.legacy,
                  syncedFromSupabase: verifyRes.diagnostics?.dbFresh ?? supabaseSyncedFromCloud,
                  hasEvaluatedMatch: false,
                  gatePassed: false,
                  topName: null,
                  topPct: 0,
                  streak: 0,
                  requiredStreak: KIOSK_FAST_MATCH_STREAK,
                  reason: 'AMBIGUOUS',
                });
                setScanStage('burst');
                return;
              }

              isLastVerifyAmbiguous = false;
              if (!ambiguousSeenInAttemptRef.current) {
                setAmbiguousCloseHint(false);
              }

              // 4. Genuine Server-Verified Match -> Strict 3/3 Consecutive Same-Patient Voting with Low Variance!
              const matchedPat = verifyRes.patient;
              if (
                verifyRes.matched &&
                matchedPat &&
                verifyRes.confidencePercent >= 84 &&
                !excludedPatientIdsRef.current.includes(matchedPat.id) &&
                !excludedPatientIdsRef.current.includes(matchedPat.hn)
              ) {
                preferredMirrorModeRef.current = useMirroredFirst ? 'mirrored' : 'normal';
                const scoreNorm = Math.max(0, Math.min(1, verifyRes.confidencePercent / 100));
                if (serverStreakRef.current.patientId === matchedPat.id) {
                  const candidateScores = [...serverStreakRef.current.scores, scoreNorm];
                  const minScore = Math.min(...candidateScores);
                  const maxScore = Math.max(...candidateScores);
                  if (maxScore - minScore <= 0.08) {
                    serverStreakRef.current = {
                      patientId: matchedPat.id,
                      scores: candidateScores,
                    };
                  } else {
                    serverStreakRef.current = {
                      patientId: matchedPat.id,
                      scores: [scoreNorm],
                    };
                  }
                } else {
                  serverStreakRef.current = {
                    patientId: matchedPat.id,
                    scores: [scoreNorm],
                  };
                }

                const nextStreak = serverStreakRef.current.scores.length;
                const isConfirmed = nextStreak >= KIOSK_FAST_MATCH_STREAK;
                bioVerifiedUnmatchedStartAt = 0;
                unmatchedClearFramesRef.current = 0;
                setBurstCount(nextStreak);
                setTurningHintPct(null);

                if (nextStreak === KIOSK_FAST_MATCH_STREAK - 1) {
                  void preloadRehabPosePipeline();
                }

                const matchOverlay = resolveKioskFaceBoundingBoxOverlay({
                  isSpoof: false,
                  isBioVerified: true,
                  rawCosine: scoreNorm,
                  centeredCosine:
                    verifyRes.diagnostics?.top1CenteredPercent !== undefined
                      ? verifyRes.diagnostics.top1CenteredPercent / 100
                      : BIOMETRIC_CENTERED_COSINE_THRESHOLD,
                  hasEvaluatedMatch: true,
                  gatePassed: true,
                  streak: nextStreak,
                  requiredStreak: KIOSK_FAST_MATCH_STREAK,
                  confirmedCandidateName: isConfirmed ? matchedPat.fullName : null,
                  confirmedCandidateHn: isConfirmed ? matchedPat.hn : null,
                  distanceM: bioReportRef.current?.estimatedDistanceM ?? 1.8,
                  confidencePercent: verifyRes.confidencePercent,
                  isLongRange: bioReportRef.current?.livenessMode === 'LONG_RANGE',
                  isAmbiguous: false,
                });
                liveOverlayRef.current = matchOverlay;
                setLiveOverlay(matchOverlay);
                setLiveMatchHud(
                  isConfirmed
                    ? {
                        patientId: matchedPat.id,
                        hn: matchedPat.hn,
                        fullName: matchedPat.fullName,
                        confidencePercent: verifyRes.confidencePercent,
                        streak: nextStreak,
                        confirmed: true,
                      }
                    : null,
                );
                setLiveDebugHud({
                  validAccounts: accountCountsRef.current.valid,
                  totalAccounts: accountCountsRef.current.total,
                  legacyAccounts: accountCountsRef.current.legacy,
                  syncedFromSupabase: verifyRes.diagnostics?.dbFresh ?? true,
                  hasEvaluatedMatch: true,
                  gatePassed: true,
                  topName: isConfirmed ? matchedPat.fullName : null,
                  topPct: isConfirmed ? verifyRes.confidencePercent : 0,
                  streak: nextStreak,
                  requiredStreak: KIOSK_FAST_MATCH_STREAK,
                  reason: isConfirmed ? 'MATCHED' : 'CONFIRMING',
                });

                if (isConfirmed) {
                  verifyingLockRef.current = true;
                  const lockConfirmedAt = performance.now();
                  setScanStage('verifying');
                  // Build ephemeral own-session 1:1 template strictly from the verified user's live embedding
                  // so zero other patients' templates or stored DB vectors are downloaded to the Kiosk browser.
                  const ownSessionTemplate: OwnerBiometricTemplate = {
                    patientId: matchedPat.id,
                    hn: matchedPat.hn,
                    fullName: matchedPat.fullName,
                    masterEmbedding: usedEmbedding,
                    angleEmbeddings: {
                      center: usedEmbedding,
                      left: usedEmbedding,
                      right: usedEmbedding,
                      up: usedEmbedding,
                      down: usedEmbedding,
                    },
                    biometricEngine: BIOMETRIC_ENGINE_ID,
                    geometricRatios: usedGeometricRatios,
                  };
                  const nextPayload = buildKioskLoginSuccessPayload({
                    patient: matchedPat,
                    todayRoutine: verifyRes.todayRoutine ?? null,
                    confidencePercent: verifyRes.confidencePercent,
                    rehabSchedules: verifyRes.rehabSchedules,
                    serverTodayExercises: verifyRes.todayExercises,
                    serverCompletedToday: verifyRes.completedToday,
                    directBiometricTemplate: ownSessionTemplate,
                  });
                  const commitNow = performance.now();
                  profilerRef.current.commitRound(
                    'matched',
                    commitNow,
                    Math.max(0, commitNow - lockConfirmedAt),
                  );
                  if (showCalibHudRef.current) {
                    const nextSnap = profilerRef.current.getSnapshot(commitNow);
                    setStaffPerfSnapshot(nextSnap);
                    if (typeof window !== 'undefined') {
                      (window as unknown as Record<string, unknown>).__drtechBiometricPerf = nextSnap;
                    }
                  }
                  openVerifiedModal(nextPayload);
                  return;
                }
                setScanStage('burst');
                return;
              }

              // 5. Genuine Unmatched Result (BELOW_THRESHOLD / NO_TEMPLATES_IN_LIBRARY / VERSION_MISMATCH / PATIENT_FILTERED_OUT)
              serverStreakRef.current = { patientId: null, scores: [] };
              setBurstCount(0);
              setLiveMatchHud(null);
              // If the exact same face just collided with multiple close accounts (AMBIGUOUS),
              // do NOT mislabel them as "unregistered / account not found" on a subsequent jitter frame!
              if (resolvedReason === 'BELOW_THRESHOLD' && ambiguousSeenInAttemptRef.current) {
                setAmbiguousCloseHint(true);
                const stayAmbigOverlay = resolveKioskFaceBoundingBoxOverlay({
                  isSpoof: false,
                  isBioVerified: true,
                  rawCosine: (verifyRes.confidencePercent ?? 0) / 100,
                  hasEvaluatedMatch: true,
                  gatePassed: false,
                  streak: 0,
                  requiredStreak: KIOSK_FAST_MATCH_STREAK,
                  confirmedCandidateName: null,
                  distanceM: bioReportRef.current?.estimatedDistanceM ?? 1.8,
                  isAmbiguous: true,
                });
                liveOverlayRef.current = stayAmbigOverlay;
                setLiveOverlay(stayAmbigOverlay);
                setScanStage('burst');
                return;
              }
              const noMatchOverlay = resolveKioskFaceBoundingBoxOverlay({
                isSpoof: false,
                isBioVerified: true,
                rawCosine: 0,
                hasEvaluatedMatch: true,
                gatePassed: false,
                streak: 0,
                requiredStreak: KIOSK_FAST_MATCH_STREAK,
                confirmedCandidateName: null,
                distanceM: bioReportRef.current?.estimatedDistanceM ?? 1.8,
                isLongRange: bioReportRef.current?.livenessMode === 'LONG_RANGE',
                isAmbiguous: false,
              });
              liveOverlayRef.current = noMatchOverlay;
              setLiveOverlay(noMatchOverlay);
              setLiveDebugHud({
                validAccounts: accountCountsRef.current.valid,
                totalAccounts: accountCountsRef.current.total,
                legacyAccounts: accountCountsRef.current.legacy,
                syncedFromSupabase: verifyRes.diagnostics?.dbFresh ?? supabaseSyncedFromCloud,
                hasEvaluatedMatch: true,
                gatePassed: false,
                topName: null,
                topPct: 0,
                streak: 0,
                requiredStreak: KIOSK_FAST_MATCH_STREAK,
                reason: resolvedReason,
              });

              const inCooldown = Date.now() < rescanCooldownUntilRef.current;
              const isHidden = typeof document !== 'undefined' && document.hidden;
              const hasZeroCompatCandidates =
                resolvedReason === 'NO_TEMPLATES_IN_LIBRARY' ||
                resolvedReason === 'VERSION_MISMATCH' ||
                resolvedReason === 'PATIENT_FILTERED_OUT';

              if (!inCooldown && candidatesLoadedRef.current && humanWarmupReadyRef.current && !isHidden) {
                if (bioVerifiedUnmatchedStartAt === 0) {
                  bioVerifiedUnmatchedStartAt = params.snapNow;
                }
                if (scanDecisionStartAtRef.current === 0) {
                  scanDecisionStartAtRef.current = params.snapNow;
                }
                totalEvaluatedFramesRef.current += 1;
                unmatchedClearFramesRef.current += 1;
                const unmatchedElapsed = params.snapNow - bioVerifiedUnmatchedStartAt;
                const totalDecisionElapsed = params.snapNow - scanDecisionStartAtRef.current;

                if (
                  shouldTriggerFastNotFoundVerdict({
                    candidatesLoaded: candidatesLoadedRef.current,
                    humanWarmupReady: humanWarmupReadyRef.current,
                    documentHidden: isHidden,
                    isAmbiguous: false,
                    isSpoof: params.snapIsSpoof,
                    bioVerified: params.snapBioVerified,
                    matchStreak: 0,
                    unmatchedClearFrames: unmatchedClearFramesRef.current,
                    unmatchedElapsedMs: unmatchedElapsed,
                    inRescanCooldown: inCooldown,
                    hasZeroCandidates: hasZeroCompatCandidates,
                    totalEvaluatedFrames: totalEvaluatedFramesRef.current,
                    totalDecisionElapsedMs: totalDecisionElapsed,
                    serverRejectedCount: 0,
                  })
                ) {
                  profilerRef.current.commitRound('not_found', performance.now(), 0);
                  shouldContinueLoop = false;
                  setScanStage('not_found');
                  return;
                }
                setScanStage('burst');
              } else {
                bioVerifiedUnmatchedStartAt = 0;
                unmatchedClearFramesRef.current = 0;
                scanDecisionStartAtRef.current = 0;
                totalEvaluatedFramesRef.current = 0;
                setScanStage('scanning');
              }
            }
          } catch {
            // Keep async matcher resilient across transient detector errors
          } finally {
            asyncMatchInFlight = false;
          }
        };

        const loop = () => {
          if (!active || !shouldContinueLoop || !videoRef.current || !landmarker) return;
          try {
              const verifiedPayload = verifiedModalPayloadRef.current;
            if (verifiedPayload) {
              if (!gestureWarmupStarted && !handLandmarker && !poseLandmarker) {
                gestureWarmupStarted = true;
                if (gestureWarmupTimerId !== null) {
                  window.clearTimeout(gestureWarmupTimerId);
                  gestureWarmupTimerId = null;
                }
                if (active) {
                  void warmupGestureDetectors().catch(() => {});
                }
              }
              if (lastConfirmedPatientId !== verifiedPayload.patient.id) {
                lastConfirmedPatientId = verifiedPayload.patient.id;
                confirmFaceCycle = 0;
                cachedConfirmOwnerVerified = false;
                cachedConfirmFaceCount = 0;
                cachedConfirmFaceBox = null;
                cachedConfirmExtraFaceBoxes = [];
                lastOwnerGuardKey = '';
                gestureHoldMs = 0;
                gestureTriggered = false;
              }
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
                voiceMuted: voiceMutedRef.current,
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
                      const offW = 320;
                      const offH = Math.max(
                        180,
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
                      const handTs = getNextMonotonicHandDetectTs(now);
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
                (landmarker || handLandmarker || poseLandmarker)
              ) {
                let latestLeftUp = false;
                let latestRightUp = false;

                // Throttle inference smoothly (~22Hz max, >=45ms apart) so 60-120fps animations never stutter
                if (
                  now - lastGestureInferenceAt >= 45 &&
                  (v.currentTime !== lastGestureVideoTime || now - lastGestureInferenceAt >= 80)
                ) {
                  lastGestureVideoTime = v.currentTime;
                  lastGestureInferenceAt = now;
                  confirmFaceCycle += 1;

                  const cCanvas = confirmCanvasRef.current;
                  const cw = cCanvas?.clientWidth || 320;
                  const ch = cCanvas?.clientHeight || 200;

                  // 1. Run FaceLandmarker + Continuous 1:1 Account Owner Verification every 4 frames (and immediately on cycle 1)
                  if (landmarker) {
                    try {
                      const faceTs = getNextMonotonicVideoDetectTs(now);
                      const fRes = landmarker.detectForVideo(v, faceTs);
                      const rawFaces = fRes.faceLandmarks ?? [];
                      const faceSelection = selectPrimaryFace(rawFaces);
                      const primaryLm = faceSelection.primaryFace ?? rawFaces[0] ?? null;
                      const effCount =
                        rawFaces.length > 1
                          ? Math.max(1, faceSelection.effectiveFaceCount)
                          : rawFaces.length;
                      cachedConfirmFaceCount = effCount;

                      if (!primaryLm || effCount === 0) {
                        cachedConfirmFaceBox = null;
                        cachedConfirmExtraFaceBoxes = [];
                        cachedConfirmOwnerVerified = false;
                      } else {
                        cachedConfirmFaceBox = computeOwnerFaceBoxMetrics(
                          primaryLm,
                          v.videoWidth,
                          v.videoHeight,
                          cw,
                          ch,
                        );
                        cachedConfirmExtraFaceBoxes = rawFaces
                          .filter((lm) => lm !== primaryLm)
                          .map((lm) =>
                            computeOwnerFaceBoxMetrics(lm, v.videoWidth, v.videoHeight, cw, ch),
                          )
                          .filter((b): b is OwnerFaceBoxMetrics => b !== null);

                        if (effCount > 1) {
                          cachedConfirmOwnerVerified = false;
                        } else if (
                          !verifiedPayload.biometricTemplate ||
                          verifiedPayload.biometricTemplate.masterEmbedding.length === 0
                        ) {
                          // Fallback for PIN-only patient sessions with no stored vector: still require single face
                          cachedConfirmOwnerVerified = true;
                        } else if (
                          !confirmEmbedInFlight &&
                          (confirmFaceCycle === 1 ||
                            confirmFaceCycle % 4 === 0 ||
                            !cachedConfirmOwnerVerified)
                        ) {
                          if (!confirmSnapCanvas) {
                            confirmSnapCanvas = document.createElement('canvas');
                          }
                          const confirmRawBox = computeBoundingBoxFromLandmarks(primaryLm);
                          const snapCanvas = captureCanonicalFaceSnapshot(
                            v,
                            confirmRawBox,
                            confirmSnapCanvas,
                            primaryLm,
                          );
                          if (snapCanvas) {
                            const snapLm = primaryLm.map((p) => ({ x: p.x, y: p.y, z: p.z }));
                            const snapEffCount = effCount;
                            const snapW = v.videoWidth || 1280;
                            const snapH = v.videoHeight || 720;
                            const snapTemplate = verifiedPayload.biometricTemplate;
                            confirmEmbedInFlight = true;
                            void extractCanonicalFaceDescriptor(snapCanvas)
                              .then((extracted) => {
                                if (!active || !verifiedModalPayloadRef.current || !extracted.embedding) {
                                  return;
                                }
                                const livePose = evaluateLivenessAndPose(
                                  snapLm,
                                  null,
                                  snapEffCount,
                                );
                                const rollDeg = computeEyeRollAngleDeg(snapLm, snapW, snapH);
                                const probeRatios = extractGeometricAnthropometricRatios(snapLm);
                                const ownerCheck = verifyOwnerFace1to1({
                                  probeVector: extracted.embedding,
                                  probeEngine: BIOMETRIC_ENGINE_ID,
                                  yawDeg: -livePose.yawDeg,
                                  pitchDeg: livePose.pitchDeg,
                                  rollDeg,
                                  probeRatios,
                                  ownerTemplate: snapTemplate,
                                });
                                if (ownerCheck.hasTemplate) {
                                  cachedConfirmOwnerVerified = ownerCheck.matched;
                                }
                              })
                              .catch(() => {})
                              .finally(() => {
                                confirmEmbedInFlight = false;
                              });
                          }
                        }
                      }
                    } catch {
                      // Keep loop resilient across transient MediaPipe frame errors
                    }
                  }

                  if (!gestureOffscreenCanvas) {
                    gestureOffscreenCanvas = document.createElement('canvas');
                  }
                  const offW = 320;
                  const offH = Math.max(
                    180,
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
                  const inferenceSource = offCtx ? gestureOffscreenCanvas : v;

                  let detectedHands: NormalizedLandmark[][] = [];
                  let detectedPose: NormalizedLandmark[] | null = null;
                  let detectedPoseCount = 0;
                  if (handLandmarker) {
                    try {
                      const handTs = getNextMonotonicHandDetectTs(now);
                      const handRes = handLandmarker.detectForVideo(inferenceSource, handTs);
                      detectedHands = handRes.landmarks ?? [];
                    } catch {
                      // Keep gesture loop running across transient MediaPipe frame errors
                    }
                  }
                  // Run PoseLandmarker for skeleton-shoulder verification when HandLandmarker misses or single hand shows
                  if (
                    poseLandmarker &&
                    (!handLandmarker ||
                      (detectedHands.length >= 1 && now - lastPoseFallbackAt >= 140))
                  ) {
                    lastPoseFallbackAt = now;
                    try {
                      const poseTs = getNextMonotonicPoseDetectTs(now);
                      const poseRes = poseLandmarker.detectForVideo(inferenceSource, poseTs);
                      detectedPoseCount = (poseRes.landmarks ?? []).length;
                      detectedPose = poseRes.landmarks?.[0] ?? null;
                    } catch {
                      // Keep gesture loop running across transient MediaPipe frame errors
                    }
                  }

                  // 2. Owner-Bound Kinematic Hand Filter + Face Lock gate
                  const handEval = evaluateOwnerBoundKinematicHands({
                    ownerFaceVerified: cachedConfirmOwnerVerified,
                    ownerFaceBox: cachedConfirmFaceBox,
                    faceCount: cachedConfirmFaceCount,
                    ownerName: verifiedPayload.patient.fullName,
                    hands: detectedHands,
                    pose: detectedPose,
                    poseCount: detectedPoseCount,
                    videoW: v.videoWidth,
                    videoH: v.videoHeight,
                    canvasW: cw,
                    canvasH: ch,
                  });

                  const isGuardBlocked =
                    handEval.status === 'IMPOSTOR_FACE' ||
                    handEval.status === 'NO_OWNER_FACE' ||
                    handEval.status === 'MULTI_PERSON' ||
                    handEval.status === 'INTRUDER_HAND';

                  if (isGuardBlocked) {
                    latestLeftUp = false;
                    latestRightUp = false;
                    latestAnyHandUp = false;
                    // Force hard lock reset to 0.0 / 4.0s (0%) whenever blocked by Face Lock or Intruder Hand!
                    gestureHoldMs = 0;
                    lastAnyHandRaisedAt = -Infinity;
                    const alertStr =
                      lang === 'en'
                        ? handEval.alertBannerEn ?? handEval.alertBannerTh ?? ''
                        : lang === 'cn'
                          ? handEval.alertBannerCn ?? handEval.alertBannerTh ?? ''
                          : handEval.alertBannerTh ?? '';
                    const guardKey = `${handEval.status}:${alertStr}`;
                    if (lastOwnerGuardKey !== guardKey) {
                      lastOwnerGuardKey = guardKey;
                      setOwnerGuardBanner(
                        alertStr ? { status: handEval.status, text: alertStr } : null,
                      );
                    }
                  } else {
                    if (lastOwnerGuardKey !== '') {
                      lastOwnerGuardKey = '';
                      setOwnerGuardBanner(null);
                    }
                    latestLeftUp = handEval.leftUp;
                    latestRightUp = handEval.rightUp;
                    latestAnyHandUp = handEval.bothUp;
                  }

                  setLeftHandUpState((prev) => (prev === latestLeftUp ? prev : latestLeftUp));
                  setRightHandUpState((prev) => (prev === latestRightUp ? prev : latestRightUp));

                  // 3. Draw Full-Bleed Overlay: Owner Face Box (#10B981 / #EF4444) + Biometric Tether HUD + Intruder Markers
                  if (cCanvas) {
                    if (cCanvas.width !== cw || cCanvas.height !== ch) {
                      cCanvas.width = cw;
                      cCanvas.height = ch;
                    }
                    const cCtx = cCanvas.getContext('2d');
                    if (cCtx) {
                      cCtx.clearRect(0, 0, cw, ch);
                      drawOwnerBiometricOverlayOnCanvas(cCtx, cw, ch, {
                        ownerName: verifiedPayload.patient.fullName,
                        faceBox: cachedConfirmFaceBox,
                        extraFaceBoxes: cachedConfirmExtraFaceBoxes,
                        ownerFaceVerified: cachedConfirmOwnerVerified,
                        faceCount: cachedConfirmFaceCount,
                        handEval,
                      });
                    }
                  }
                }

                const isLockedByOwnerSecurity = lastOwnerGuardKey !== '';
                const holdStep = stepVerifiedHandHold({
                  bothHandsNow: latestAnyHandUp && !isLockedByOwnerSecurity,
                  speechBlocked: speechBlocked || isLockedByOwnerSecurity,
                  nowMs: now,
                  dtMs: dt,
                  lastRaisedAtMs: isLockedByOwnerSecurity ? -Infinity : lastAnyHandRaisedAt,
                  holdMs: isLockedByOwnerSecurity ? 0 : gestureHoldMs,
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
                      ? `Both hands detected! Hold ${heldSecStr}/4.0s (${pctInt}%)`
                      : lang === 'cn'
                        ? `已检测到双手举起！保持 ${heldSecStr}/4.0 秒 (${pctInt}%)`
                        : `ยกครบ 2 ข้าง! ค้างไว้ ${heldSecStr}/4.0 วิ (${pctInt}%)`;
                }
                if (nextProgress > 0 && confirmBtnLabelRef.current) {
                  confirmBtnLabelRef.current.textContent =
                    lang === 'en'
                      ? `Holding BOTH hands... ${heldSecStr}/4.0s (${remainSecStr}s left)`
                      : lang === 'cn'
                        ? `正在加载确认... ${heldSecStr}/4.0 秒（剩余 ${remainSecStr} 秒）`
                        : `กำลังโหลด 2 ข้าง... ${heldSecStr}/4.0 วิ (เหลือ ${remainSecStr} วิ)`;
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
                  excludedPatientIdsRef.current = [];
                  stopMediaAndDetectors();
                  onPatientAuthenticatedRef.current(verifiedPayload);
                  return;
                }
              }
              return;
            }

            if (videoRef.current.readyState >= 2 && videoRef.current.videoWidth > 0 && videoRef.current.videoHeight > 0) {
              const now = performance.now();
              profilerRef.current.recordFirstVideoFrame(now);
              if (firstFrameWallMs === 0) {
                firstFrameWallMs = now;
              }
              const visualDtMs = lastVisualRafAt > 0 ? Math.min(100, Math.max(4, now - lastVisualRafAt)) : 16.67;
              lastVisualRafAt = now;

              // Helper: decoupled ~30 FPS HiDPI canvas overlay renderer (throttled to avoid competing with WebGL inference)
              const drawSmoothBiometricOverlay = (frameNow: number, dtMs: number) => {
                if (!targetVisualBox || !targetVisualLandmarks || lastVisualCw <= 0 || lastVisualCh <= 0) {
                  return;
                }
                if (frameNow - lastOverlayDrawAt < KIOSK_OVERLAY_MIN_INTERVAL_MS) {
                  return;
                }
                lastOverlayDrawAt = frameNow;
                const drawStartMs = performance.now();
                smoothedVisualBox = smoothBiometricBox(smoothedVisualBox, targetVisualBox, dtMs);
                smoothedVisualLandmarks = smoothFacialLandmarks(
                  smoothedVisualLandmarks,
                  targetVisualLandmarks,
                  dtMs,
                );

                const b = smoothedVisualBox;
                const lms = smoothedVisualLandmarks;
                const cw = lastVisualCw;
                const ch = lastVisualCh;
                const renderedW = lastVisualRenderedW;
                const renderedH = lastVisualRenderedH;
                const offsetX = lastVisualOffsetX;
                const offsetY = lastVisualOffsetY;

                const clampedLeft = Math.max(4, Math.min(cw - 24, b.left));
                const clampedTop = Math.max(4, Math.min(ch - 24, b.top));
                const clampedW = Math.max(24, Math.min(cw - clampedLeft - 4, b.width));
                const clampedH = Math.max(24, Math.min(ch - clampedTop - 4, b.height));

                // GPU-only translate3d positioning for floating status pill below face box (ZERO DOM layout reflow!)
                if (hudBoxElRef.current && hudBoxMounted) {
                  const pillAnchorX = Math.max(90, Math.min(cw - 90, clampedLeft + clampedW * 0.5));
                  const pillAnchorY = Math.min(ch - 68, clampedTop + clampedH + 10);
                  hudBoxElRef.current.style.transform = `translate3d(${pillAnchorX.toFixed(1)}px, ${pillAnchorY.toFixed(1)}px, 0)`;
                }

                const mCanvas = faceMeshCanvasRef.current;
                if (!mCanvas) return;
                const dpr = typeof window !== 'undefined' ? Math.min(2, window.devicePixelRatio || 1) : 1;
                const targetW = Math.round(cw * dpr);
                const targetH = Math.round(ch * dpr);
                if (mCanvas.width !== targetW || mCanvas.height !== targetH) {
                  mCanvas.width = targetW;
                  mCanvas.height = targetH;
                }
                const mCtx = mCanvas.getContext('2d');
                if (!mCtx) return;

                mCtx.save();
                mCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
                mCtx.clearRect(0, 0, cw, ch);

                const overlayBoxColor = lastVisualOverlayColor;
                const strokeHex =
                  overlayBoxColor === 'rose'
                    ? '#f43f5e'
                    : overlayBoxColor === 'emerald'
                      ? '#10b981'
                      : overlayBoxColor === 'blue'
                        ? '#38bdf8'
                        : '#f59e0b';
                const accentHex =
                  overlayBoxColor === 'rose'
                    ? '#fb7185'
                    : overlayBoxColor === 'emerald'
                      ? '#34d399'
                      : overlayBoxColor === 'blue'
                        ? '#60a5fa'
                        : '#22d3ee';
                const dotFillRgba =
                  overlayBoxColor === 'rose'
                    ? 'rgba(251, 113, 133, 0.90)'
                    : overlayBoxColor === 'emerald'
                      ? 'rgba(52, 211, 153, 0.92)'
                      : overlayBoxColor === 'blue'
                        ? 'rgba(96, 165, 250, 0.90)'
                        : 'rgba(251, 191, 36, 0.90)';

                // 1. Sleek single rounded biometric targeting frame + luminous corner brackets (NO double border!)
                const boxRadius = Math.min(18, clampedW * 0.12, clampedH * 0.12);
                const drawRoundedRectPath = (x: number, y: number, w: number, h: number, r: number) => {
                  mCtx.beginPath();
                  mCtx.moveTo(x + r, y);
                  mCtx.lineTo(x + w - r, y);
                  mCtx.quadraticCurveTo(x + w, y, x + w, y + r);
                  mCtx.lineTo(x + w, y + h - r);
                  mCtx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
                  mCtx.lineTo(x + r, y + h);
                  mCtx.quadraticCurveTo(x, y + h, x, y + h - r);
                  mCtx.lineTo(x, y + r);
                  mCtx.quadraticCurveTo(x, y, x + r, y);
                  mCtx.closePath();
                };

                mCtx.save();
                drawRoundedRectPath(clampedLeft, clampedTop, clampedW, clampedH, boxRadius);
                mCtx.strokeStyle = strokeHex;
                mCtx.globalAlpha = 0.42;
                mCtx.lineWidth = 1.6;
                mCtx.stroke();

                // Crisp precision corner brackets
                const cornerLen = Math.max(16, Math.min(28, clampedW * 0.15));
                mCtx.globalAlpha = 0.96;
                mCtx.strokeStyle = strokeHex;
                mCtx.lineWidth = 3.0;
                mCtx.lineCap = 'round';
                mCtx.beginPath();
                // Top-left
                mCtx.moveTo(clampedLeft, clampedTop + cornerLen);
                mCtx.lineTo(clampedLeft, clampedTop + boxRadius * 0.55);
                mCtx.quadraticCurveTo(clampedLeft, clampedTop, clampedLeft + boxRadius * 0.55, clampedTop);
                mCtx.lineTo(clampedLeft + cornerLen, clampedTop);
                // Top-right
                mCtx.moveTo(clampedLeft + clampedW - cornerLen, clampedTop);
                mCtx.lineTo(clampedLeft + clampedW - boxRadius * 0.55, clampedTop);
                mCtx.quadraticCurveTo(clampedLeft + clampedW, clampedTop, clampedLeft + clampedW, clampedTop + boxRadius * 0.55);
                mCtx.lineTo(clampedLeft + clampedW, clampedTop + cornerLen);
                // Bottom-left
                mCtx.moveTo(clampedLeft, clampedTop + clampedH - cornerLen);
                mCtx.lineTo(clampedLeft, clampedTop + clampedH - boxRadius * 0.55);
                mCtx.quadraticCurveTo(clampedLeft, clampedTop + clampedH, clampedLeft + boxRadius * 0.55, clampedTop + clampedH);
                mCtx.lineTo(clampedLeft + cornerLen, clampedTop + clampedH);
                // Bottom-right
                mCtx.moveTo(clampedLeft + clampedW - cornerLen, clampedTop + clampedH);
                mCtx.lineTo(clampedLeft + clampedW - boxRadius * 0.55, clampedTop + clampedH);
                mCtx.quadraticCurveTo(clampedLeft + clampedW, clampedTop + clampedH, clampedLeft + clampedW, clampedTop + clampedH - boxRadius * 0.55);
                mCtx.lineTo(clampedLeft + clampedW, clampedTop + clampedH - cornerLen);
                mCtx.stroke();

                // 2. Smooth sinusoidal 60FPS holographic laser scanline beam with subtle gradient trail
                mCtx.save();
                drawRoundedRectPath(clampedLeft + 2, clampedTop + 2, clampedW - 4, clampedH - 4, Math.max(4, boxRadius - 2));
                mCtx.clip();
                const sweepPhase = (frameNow % 2200) / 2200;
                const sweepNorm = 0.5 - 0.5 * Math.cos(sweepPhase * Math.PI * 2);
                const scanLineY = clampedTop + 8 + sweepNorm * Math.max(16, clampedH - 16);
                const trailH = Math.min(34, clampedH * 0.16);
                const goingDown = sweepPhase < 0.5;
                const trailGrad = mCtx.createLinearGradient(
                  0,
                  goingDown ? scanLineY - trailH : scanLineY + trailH,
                  0,
                  scanLineY,
                );
                trailGrad.addColorStop(0, 'rgba(34, 211, 238, 0)');
                trailGrad.addColorStop(
                  1,
                  overlayBoxColor === 'emerald'
                    ? 'rgba(16, 185, 129, 0.18)'
                    : overlayBoxColor === 'rose'
                      ? 'rgba(244, 63, 94, 0.15)'
                      : 'rgba(34, 211, 238, 0.16)',
                );
                mCtx.fillStyle = trailGrad;
                mCtx.fillRect(
                  clampedLeft,
                  goingDown ? scanLineY - trailH : scanLineY,
                  clampedW,
                  trailH,
                );

                const scanBeamGrad = mCtx.createLinearGradient(clampedLeft, 0, clampedLeft + clampedW, 0);
                scanBeamGrad.addColorStop(0, 'rgba(34, 211, 238, 0)');
                scanBeamGrad.addColorStop(0.5, accentHex);
                scanBeamGrad.addColorStop(1, 'rgba(34, 211, 238, 0)');
                mCtx.globalAlpha = 0.78;
                mCtx.strokeStyle = scanBeamGrad;
                mCtx.lineWidth = 1.6;
                mCtx.beginPath();
                mCtx.moveTo(clampedLeft + 4, scanLineY);
                mCtx.lineTo(clampedLeft + clampedW - 4, scanLineY);
                mCtx.stroke();
                mCtx.restore();

                // 3. High-detail 478-point anatomical facial contours + craniofacial symmetry axes
                const projectPt = (pt: { x: number; y: number }) => ({
                  x: (mirroredRef.current ? 1 - pt.x : pt.x) * renderedW - offsetX,
                  y: pt.y * renderedH - offsetY,
                });

                mCtx.strokeStyle = dotFillRgba;
                mCtx.globalAlpha = 0.44;
                mCtx.lineWidth = 1.05;
                mCtx.beginPath();
                for (let cIdx = 0; cIdx < FACE_MESH_CONTOURS.length; cIdx += 1) {
                  const ring = FACE_MESH_CONTOURS[cIdx];
                  let started = false;
                  for (let pIdx = 0; pIdx < ring.length; pIdx += 1) {
                    const pt = lms[ring[pIdx]];
                    if (!pt) continue;
                    const pp = projectPt(pt);
                    if (!started) {
                      mCtx.moveTo(pp.x, pp.y);
                      started = true;
                    } else {
                      mCtx.lineTo(pp.x, pp.y);
                    }
                  }
                }
                mCtx.stroke();

                // Subtle craniofacial triangulation mesh lines (cheek-to-nose-to-brow-to-chin biometric geometry)
                const CRANIO_EDGES: ReadonlyArray<readonly [number, number]> = [
                  [33, 234], [263, 454], [234, 1], [454, 1], [168, 33], [168, 263],
                  [1, 61], [1, 291], [61, 172], [291, 397], [172, 152], [397, 152],
                  [105, 168], [334, 168], [133, 1], [362, 1],
                ];
                mCtx.globalAlpha = 0.24;
                mCtx.strokeStyle = accentHex;
                mCtx.lineWidth = 0.85;
                mCtx.beginPath();
                for (let eIdx = 0; eIdx < CRANIO_EDGES.length; eIdx += 1) {
                  const [aIdx, bIdx] = CRANIO_EDGES[eIdx];
                  const pa = lms[aIdx];
                  const pb = lms[bIdx];
                  if (!pa || !pb) continue;
                  const p1 = projectPt(pa);
                  const p2 = projectPt(pb);
                  mCtx.moveTo(p1.x, p1.y);
                  mCtx.lineTo(p2.x, p2.y);
                }
                mCtx.stroke();

                // 4. Precision Dual-Iris 8-Sector Lock Reticles (468..477 iris rings + counter-rotating arcs + IPD axis)
                if (lms.length >= 478 && lms[468] && lms[473]) {
                  const lC = projectPt(lms[468]);
                  const rC = projectPt(lms[473]);
                  const measureIrisRadius = (cIdx: number, rimIdxs: readonly number[]) => {
                    const cPt = projectPt(lms[cIdx]);
                    let sumDist = 0;
                    let cnt = 0;
                    for (const rIdx of rimIdxs) {
                      const rp = lms[rIdx];
                      if (!rp) continue;
                      const pp = projectPt(rp);
                      sumDist += Math.hypot(pp.x - cPt.x, pp.y - cPt.y);
                      cnt += 1;
                    }
                    const avgR = cnt > 0 ? sumDist / cnt : clampedW * 0.032;
                    return Math.max(4.5, Math.min(18, avgR));
                  };
                  const leftIrisR = measureIrisRadius(468, [469, 470, 471, 472]);
                  const rightIrisR = measureIrisRadius(473, [474, 475, 476, 477]);

                  // Inter-pupillary dashed measurement baseline
                  mCtx.globalAlpha = 0.50;
                  mCtx.strokeStyle = overlayBoxColor === 'emerald' ? '#6ee7b7' : '#38bdf8';
                  mCtx.lineWidth = 0.95;
                  mCtx.setLineDash([3, 3]);
                  mCtx.beginPath();
                  mCtx.moveTo(lC.x, lC.y);
                  mCtx.lineTo(rC.x, rC.y);
                  mCtx.stroke();
                  mCtx.setLineDash([]);

                  const rotOuter = (frameNow * 0.0022) % (Math.PI * 2);
                  const rotInner = (-frameNow * 0.0014) % (Math.PI * 2);
                  const drawIrisTarget = (cx: number, cy: number, radius: number) => {
                    const irisColor =
                      overlayBoxColor === 'rose'
                        ? '#fb7185'
                        : overlayBoxColor === 'emerald'
                          ? '#10b981'
                          : '#22d3ee';
                    // Inner limbus lock ring
                    mCtx.globalAlpha = 0.92;
                    mCtx.strokeStyle = irisColor;
                    mCtx.lineWidth = 1.4;
                    mCtx.beginPath();
                    mCtx.arc(cx, cy, radius, 0, Math.PI * 2);
                    mCtx.stroke();

                    // 8 radial iris-sector micro ticks (matching 8-sector ocular descriptor)
                    mCtx.globalAlpha = 0.62;
                    mCtx.lineWidth = 0.9;
                    mCtx.beginPath();
                    const tickR0 = radius * 0.82;
                    const tickR1 = radius * 1.24;
                    for (let s = 0; s < 8; s += 1) {
                      const ang = rotInner + (s * Math.PI) / 4;
                      const cosA = Math.cos(ang);
                      const sinA = Math.sin(ang);
                      mCtx.moveTo(cx + cosA * tickR0, cy + sinA * tickR0);
                      mCtx.lineTo(cx + cosA * tickR1, cy + sinA * tickR1);
                    }
                    mCtx.stroke();

                    // Outer rotating 4-arc calibration bracket
                    const outerR = radius * 1.68;
                    mCtx.globalAlpha = 0.80;
                    mCtx.lineWidth = 1.25;
                    for (let a = 0; a < 4; a += 1) {
                      const startA = rotOuter + a * (Math.PI * 0.5);
                      mCtx.beginPath();
                      mCtx.arc(cx, cy, outerR, startA, startA + 0.52);
                      mCtx.stroke();
                    }

                    // Precision axis crosshairs & sub-pixel pupil lock dot
                    const tickIn = Math.max(2.2, radius * 0.36);
                    const tickOut = outerR + 3.2;
                    mCtx.globalAlpha = 0.65;
                    mCtx.beginPath();
                    mCtx.moveTo(cx - tickOut, cy);
                    mCtx.lineTo(cx - tickIn, cy);
                    mCtx.moveTo(cx + tickIn, cy);
                    mCtx.lineTo(cx + tickOut, cy);
                    mCtx.moveTo(cx, cy - tickOut);
                    mCtx.lineTo(cx, cy - tickIn);
                    mCtx.moveTo(cx, cy + tickIn);
                    mCtx.lineTo(cx, cy + tickOut);
                    mCtx.stroke();

                    mCtx.globalAlpha = 0.98;
                    mCtx.fillStyle = '#ffffff';
                    mCtx.beginPath();
                    mCtx.arc(cx, cy, 1.4, 0, Math.PI * 2);
                    mCtx.fill();
                  };

                  drawIrisTarget(lC.x, lC.y, leftIrisR);
                  drawIrisTarget(rC.x, rC.y, rightIrisR);
                }

                // 5. Fine anatomical landmark nodes
                const dotRadius = Math.max(1.2, Math.min(2.3, clampedH * 0.013));
                mCtx.globalAlpha = 0.88;
                mCtx.fillStyle = dotFillRgba;
                mCtx.beginPath();
                for (let i = 0; i < KEY_MESH_LANDMARK_INDICES.length; i += 1) {
                  const pt = lms[KEY_MESH_LANDMARK_INDICES[i]];
                  if (!pt) continue;
                  const pp = projectPt(pt);
                  if (pp.x < 0 || pp.x > cw || pp.y < 0 || pp.y > ch) continue;
                  mCtx.moveTo(pp.x + dotRadius, pp.y);
                  mCtx.arc(pp.x, pp.y, dotRadius, 0, Math.PI * 2);
                }
                mCtx.fill();
                mCtx.restore();
                mCtx.restore();
                profilerRef.current.recordOverlayDraw(frameNow, performance.now() - drawStartMs);
              };

              const currentVideoTime = videoRef.current.currentTime;
              const isStaleVideoFrame = hasVideoFrameCallbackApi
                ? !hasFreshVideoFrame && now - lastProcessedWallMs < 95
                : currentVideoTime === lastProcessedVideoTime && now - lastProcessedWallMs < 95;
              // Between decoded camera frames, keep overlay smoothly interpolated at <= 30 FPS without re-running MediaPipe!
              if (isStaleVideoFrame) {
                drawSmoothBiometricOverlay(now, visualDtMs);
                return;
              }
              hasFreshVideoFrame = false;
              lastProcessedVideoTime = currentVideoTime;
              lastProcessedWallMs = now;

              const detectTs = getNextMonotonicVideoDetectTs(now);
              lastDetectTs = detectTs;
              if (!roiCanvasRef.current) {
                roiCanvasRef.current = document.createElement('canvas');
              }

              // Downscale primary MediaPipe FaceLandmarker input to <= 480px long side (preserving exact aspect ratio).
              // MediaPipe returns normalized [0..1] landmarks, while full-resolution videoRef.current is still used for
              // Stage-2 ROI zoom, 1024-D embedding crop, and 256-D ocular/iris extraction (ZERO accuracy loss, 4x-16x faster upload).
              const rawVw = videoRef.current.videoWidth || 1280;
              const rawVh = videoRef.current.videoHeight || 720;
              let detectInput: HTMLVideoElement | HTMLCanvasElement = videoRef.current;
              let detectInputW = rawVw;
              let detectInputH = rawVh;
              if (Math.max(rawVw, rawVh) > KIOSK_DETECT_MAX_LONG_SIDE_PX) {
                const downScale = KIOSK_DETECT_MAX_LONG_SIDE_PX / Math.max(rawVw, rawVh);
                detectInputW = Math.max(160, Math.round(rawVw * downScale));
                detectInputH = Math.max(120, Math.round(rawVh * downScale));
                if (!detectCanvasRef.current) {
                  detectCanvasRef.current = document.createElement('canvas');
                }
                const dCanvas = detectCanvasRef.current;
                if (dCanvas.width !== detectInputW || dCanvas.height !== detectInputH) {
                  dCanvas.width = detectInputW;
                  dCanvas.height = detectInputH;
                }
                const dCtx = dCanvas.getContext('2d');
                if (dCtx) {
                  dCtx.drawImage(videoRef.current, 0, 0, detectInputW, detectInputH);
                  detectInput = dCanvas;
                } else {
                  detectInputW = rawVw;
                  detectInputH = rawVh;
                }
              }

              const detectStartMs = performance.now();
              const res = landmarker.detectForVideo(detectInput, detectTs);
              const detectMs = performance.now() - detectStartMs;
              let selection = selectPrimaryFace(res.faceLandmarks ?? []);
              let activeBlendshapes = res.faceBlendshapes?.[0]?.categories;

              // Stage-1 & Stage-2 Two-Stage Smart ROI Zoom: when no face found at full-frame or distant face < 12% height,
              // lock onto last known ROI / Pose head / horizontal sweep -> upscale 320x320 -> refine FaceLandmarker
              const coarseHeight = selection.primaryBox?.height ?? 0;
              let roiMs = 0;
              if (roiLandmarker && (!selection.primaryFace || coarseHeight < 0.12)) {
                const roiStartMs = performance.now();
                let targetRoi =
                  selection.primaryFace && selection.primaryBox
                    ? computeHeadZoneRoiFromFaceBox(
                        {
                          x: Math.max(0, (selection.primaryCenter?.x ?? 0.5) - selection.primaryBox.width / 2),
                          y: Math.max(0, (selection.primaryCenter?.y ?? 0.5) - selection.primaryBox.height / 2),
                          width: selection.primaryBox.width,
                          height: selection.primaryBox.height,
                        },
                        2.35,
                      )
                    : lastKnownRoiBox
                      ? computeHeadZoneRoiFromFaceBox(lastKnownRoiBox, 2.4)
                      : null;

                if (!targetRoi && poseLandmarker && now - lastPoseDetectTs >= 180) {
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
                  const sweepZone = ROI_SWEEP_ZONES[roiSweepIdx % ROI_SWEEP_ZONES.length];
                  roiSweepIdx += 1;
                  targetRoi = sweepZone ?? computeHeadZoneRoiFromPose(null);
                }

                const upscaledRoi = renderUpscaledRoi256(videoRef.current, targetRoi, roiCanvasRef.current, 320);
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
                roiMs = performance.now() - roiStartMs;
              }

              const faceCount = selection.effectiveFaceCount;
              const landmarks = selection.primaryFace ? [...selection.primaryFace] : [];

              const rawLiveness = evaluateLivenessAndPose(landmarks, prevLandmarks, faceCount, wasEyeClosed);
              if (faceCount === 1 && landmarks.length >= 300) {
                if (!hasSmoothedPose) {
                  smoothedYawDeg = rawLiveness.yawDeg;
                  smoothedPitchDeg = rawLiveness.pitchDeg;
                  hasSmoothedPose = true;
                } else {
                  smoothedYawDeg = smoothedYawDeg * 0.45 + rawLiveness.yawDeg * 0.55;
                  smoothedPitchDeg = smoothedPitchDeg * 0.45 + rawLiveness.pitchDeg * 0.55;
                }
              } else {
                hasSmoothedPose = false;
                smoothedYawDeg = 0;
                smoothedPitchDeg = 0;
              }
              const liveness =
                faceCount === 1 && landmarks.length >= 300
                  ? {
                      ...rawLiveness,
                      yawDeg: Number(smoothedYawDeg.toFixed(1)),
                      pitchDeg: Number(smoothedPitchDeg.toFixed(1)),
                    }
                  : rawLiveness;
              const meanEar = (liveness.earLeft + liveness.earRight) / 2;
              wasEyeClosed = meanEar < 0.21;

              const faceRatio =
                landmarks.length > 400 ? Math.abs(landmarks[454].x - landmarks[234].x) : 0;
              presenceRef.current.update({
                facePresent: faceCount === 1,
                faceSizeRatio: faceRatio,
                timestampMs: now,
              });

              if (faceCount !== 1 || landmarks.length < 300) {
                targetVisualBox = null;
                smoothedVisualBox = null;
                targetVisualLandmarks = null;
                smoothedVisualLandmarks = null;
                cachedAlignCrop = null;
                cachedContextFrame = null;
                cachedAlignContrastSharpness = undefined;
                lastAlignSampleAt = 0;
                clearFaceMeshCanvas();
                updateReadinessHint(null);
                if (noFaceSinceRef.current === null) {
                  noFaceSinceRef.current = now;
                } else if (now - noFaceSinceRef.current >= KIOSK_NO_FACE_GUIDANCE_DELAY_MS) {
                  setShowNoFaceGuidance(true);
                }
                unreadyFrameStreak += 1;
                if (unreadyFrameStreak >= 2 && hudBoxMounted) {
                  hudBoxMounted = false;
                  setHudBox(null);
                }
                if (unreadyFrameStreak >= 4) {
                  lastKnownRoiBox = null;
                  mirrorProbedForCurrentFace = false;
                }
                if (unreadyFrameStreak >= 18 && !verifyingLockRef.current && !asyncMatchInFlight) {
                  serverStreakRef.current = { patientId: null, scores: [] };
                  ambiguousSeenInAttemptRef.current = false;
                  qualityPassedFramesRef.current = 0;
                  profilerRef.current.resetFaceAttempt();
                  setLiveMatchHud(null);
                  setTurningHintPct(null);
                  bioLivenessRef.current.reset();
                  probeWindowRef.current = [];
                  bioVerifiedUnmatchedStartAt = 0;
                  unmatchedClearFramesRef.current = 0;
                  scanDecisionStartAtRef.current = 0;
                  totalEvaluatedFramesRef.current = 0;
                  serverRejectCountRef.current = 0;
                  isLastVerifyAmbiguous = false;
                  setBioReport(null);
                  setBurstCount(0);
                  setScanStage('scanning');
                }
              } else {
                profilerRef.current.markFaceSeen(now);
                noFaceSinceRef.current = null;
                setShowNoFaceGuidance(false);
              }

              if (!alignCanvasRef.current) {
                alignCanvasRef.current = document.createElement('canvas');
              }
              if (!contextCanvasRef.current) {
                contextCanvasRef.current = document.createElement('canvas');
              }
              const rawBoxForBio =
                faceCount === 1 && landmarks.length >= 300
                  ? computeBoundingBoxFromLandmarks(landmarks)
                  : null;
              if (rawBoxForBio) {
                lastKnownRoiBox = rawBoxForBio;
              }

              // Throttle synchronous GPU->CPU getImageData readbacks to ~12.5 Hz (every 80ms)
              // while keeping landmark blinks, iris gaze micro-saccades & smooth mesh tracking full speed
              const livenessStartMs = performance.now();
              const shouldSamplePixelCrops =
                landmarks.length >= 300 &&
                (!cachedAlignCrop || now - lastAlignSampleAt >= 80);
              if (shouldSamplePixelCrops) {
                lastAlignSampleAt = now;
                cachedAlignCrop = alignFaceTo112(
                  videoRef.current,
                  landmarks,
                  alignCanvasRef.current,
                  roiCanvasRef.current,
                );
                cachedContextFrame = captureDownscaledContextFrame(
                  videoRef.current,
                  contextCanvasRef.current,
                  160,
                );
                cachedAlignContrastSharpness = cachedAlignCrop
                  ? computeContrastNormalizedSharpness(cachedAlignCrop)
                  : undefined;
                if (!exposureSettled && cachedAlignCrop) {
                  const data = cachedAlignCrop.data;
                  let lumaSum = 0;
                  let lumaCount = 0;
                  for (let px = 0; px < data.length; px += 64) {
                    lumaSum += 0.299 * data[px] + 0.587 * data[px + 1] + 0.114 * data[px + 2];
                    lumaCount += 1;
                  }
                  const meanLuma = lumaCount > 0 ? lumaSum / lumaCount : 128;
                  const elapsedSinceFirstFrame = Math.max(0, now - firstFrameWallMs);
                  if (
                    (meanLuma >= 65 && meanLuma <= 215) ||
                    elapsedSinceFirstFrame >= KIOSK_MAX_EXPOSURE_SETTLE_MS
                  ) {
                    exposureSettled = true;
                    profilerRef.current.recordExposureSettled(
                      Math.min(KIOSK_MAX_EXPOSURE_SETTLE_MS, elapsedSinceFirstFrame),
                    );
                  }
                }
              } else if (!exposureSettled && firstFrameWallMs > 0 && now - firstFrameWallMs >= KIOSK_MAX_EXPOSURE_SETTLE_MS) {
                exposureSettled = true;
                profilerRef.current.recordExposureSettled(KIOSK_MAX_EXPOSURE_SETTLE_MS);
              }
              const crop = landmarks.length >= 300 ? cachedAlignCrop : null;
              const contextFrame = landmarks.length >= 300 ? cachedContextFrame : null;

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
                      faceCrop: shouldSamplePixelCrops ? crop : null,
                      contextFrame: shouldSamplePixelCrops ? contextFrame : null,
                      faceBox: rawBoxForBio,
                      timestampMs: now,
                    })
                  : null;

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
                      ...(cachedAlignContrastSharpness !== undefined
                        ? { sharpness: cachedAlignContrastSharpness }
                        : {}),
                    }
                  : {
                      allowLongRange: true,
                      ...(cachedAlignContrastSharpness !== undefined
                        ? { sharpness: cachedAlignContrastSharpness }
                        : {}),
                    },
              });
              const livenessMs = performance.now() - livenessStartMs;
              profilerRef.current.recordDetectFrame({
                now,
                detectMs,
                roiMs,
                livenessMs,
                inputWidth: detectInputW,
                inputHeight: detectInputH,
              });
              prevLandmarks = landmarks;
              // Single source of truth: immediate state clear removes spoof whenever biologicalVerified passes
              const isFrameSpoof = bio
                ? bio.spoofDetected && !bio.biologicalVerified
                : inspection.scan_status === 'SPOOF_DETECTED';

              const bioVerifiedNow = Boolean(bio?.biologicalVerified && !isFrameSpoof);
              if (bioVerifiedNow && bio) {
                const livenessCue: BiometricLivenessCue = bio.blinkVerified
                  ? 'blink'
                  : bio.irisLivenessVerified
                    ? 'iris_saccade'
                    : bio.lipMotionVerified
                      ? 'lip'
                      : bio.headPoseVerified
                        ? 'head_pose'
                        : 'long_range';
                profilerRef.current.markLivenessVerified(now, livenessCue);
              }
              const bioModeNow = bio?.livenessMode ?? '';
              const bioModeChanged =
                bioVerifiedNow !== lastBioVerifiedUi ||
                isFrameSpoof !== lastSpoofUi ||
                bioModeNow !== lastBioModeUi;
              if (bioModeChanged || (showCalibHudRef.current && now - lastTelemetryUiSyncAt >= 90)) {
                lastTelemetryUiSyncAt = now;
                lastBioVerifiedUi = bioVerifiedNow;
                lastSpoofUi = isFrameSpoof;
                lastBioModeUi = bioModeNow;
                setLivenessInfo(liveness);
                if (bio) {
                  setBioReport(bio);
                }
                setInspectionInfo(inspection);
              }
              if (showCalibHudRef.current && now - lastPerfHudSyncAt >= 250) {
                lastPerfHudSyncAt = now;
                const perfSnap = profilerRef.current.getSnapshot(now);
                setStaffPerfSnapshot(perfSnap);
                if (typeof window !== 'undefined') {
                  (window as unknown as Record<string, unknown>).__drtechBiometricPerf = perfSnap;
                }
              }

              // Real-time 60-120 FPS Smooth Tracking Target Update & GPU Render
              if (faceCount === 1 && landmarks.length >= 300) {
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

                  lastVisualRenderedW = renderedW;
                  lastVisualRenderedH = renderedH;
                  lastVisualOffsetX = offsetX;
                  lastVisualOffsetY = offsetY;
                  lastVisualCw = cw;
                  lastVisualCh = ch;
                  lastVisualOverlayColor = isFrameSpoof
                    ? 'rose'
                    : !bioVerifiedNow
                      ? 'blue'
                      : (liveOverlayRef.current?.boxColor ?? 'amber');
                  targetVisualBox = { left: pxLeft, top: pxTop, width: pxW, height: pxH };
                  targetVisualLandmarks = landmarks;

                  if (!hudBoxMounted) {
                    const xPct = Math.max(0.5, Math.min(94, (pxLeft / cw) * 100));
                    const yPct = Math.max(0.5, Math.min(94, (pxTop / ch) * 100));
                    const wPct = Math.max(4, Math.min(99 - xPct, (pxW / cw) * 100));
                    const hPct = Math.max(4, Math.min(99 - yPct, (pxH / ch) * 100));
                    hudBoxMounted = true;
                    setHudBox({ xPct, yPct, wPct, hPct });
                  }

                  drawSmoothBiometricOverlay(now, visualDtMs);
                }

                const faceReadyForLogin =
                  !isFrameSpoof && isFullFrontalFaceReadyForLogin(landmarks, liveness, inspection);
                if (!faceReadyForLogin) {
                  unreadyFrameStreak += 1;
                  if (unreadyFrameStreak >= 2 || isFrameSpoof) {
                    const hint = resolveKioskReadinessGuidance({
                      isSpoof: isFrameSpoof,
                      isBioVerified: bioVerifiedNow,
                      yawDeg: liveness.yawDeg,
                      pitchDeg: liveness.pitchDeg,
                      inspection,
                      blurryFrameStreak,
                      lang: langRef.current,
                    });
                    if (unreadyFrameStreak >= 3 || isFrameSpoof) {
                      updateReadinessHint(hint);
                    }
                    // Reset streak after >=2 consecutive unready frames (or immediately on spoof) to avoid 1-frame micro-glitch resets
                    if (!verifyingLockRef.current && !asyncMatchInFlight) {
                      if (isFrameSpoof) {
                        probeWindowRef.current = [];
                      }
                      serverStreakRef.current = { patientId: null, scores: [] };
                      setBurstCount(0);
                      setLiveMatchHud(null);
                      setTurningHintPct(null);
                    }
                  }
                  if (
                    (isFrameSpoof || unreadyFrameStreak >= 18) &&
                    !verifyingLockRef.current &&
                    !asyncMatchInFlight
                  ) {
                    bioVerifiedUnmatchedStartAt = 0;
                    unmatchedClearFramesRef.current = 0;
                    scanDecisionStartAtRef.current = 0;
                    totalEvaluatedFramesRef.current = 0;
                    serverRejectCountRef.current = 0;
                    isLastVerifyAmbiguous = false;
                    liveOverlayRef.current = null;
                    setLiveOverlay(null);
                    setAmbiguousCloseHint(false);
                    setScanStage('scanning');
                  }
                } else {
                  unreadyFrameStreak = 0;
                  profilerRef.current.markFaceReady(now);
                  if (blurryFrameStreak < 2) {
                    updateReadinessHint(null);
                  }
                  // Dispatch single-pass backlit-normalized 1024-D + 256-D Dual-Iris matching asynchronously
                  // so GPU landmark tracking NEVER stalls
                  if (
                    !asyncMatchInFlight &&
                    !verifyingLockRef.current &&
                    bio &&
                    meanEar >= 0.16 &&
                    now - lastEmbedCheckAt >= 20
                  ) {
                    if (!matchSnapCanvasRef.current) {
                      matchSnapCanvasRef.current = document.createElement('canvas');
                    }
                    if (!irisLeftCanvasRef.current) {
                      irisLeftCanvasRef.current = document.createElement('canvas');
                    }
                    if (!irisRightCanvasRef.current) {
                      irisRightCanvasRef.current = document.createElement('canvas');
                    }
                    const snapBundle = captureCanonicalFaceSnapshotWithMetrics(
                      videoRef.current,
                      rawBox ?? rawBoxForBio,
                      matchSnapCanvasRef.current,
                      landmarks,
                    );
                    if (snapBundle) {
                      lastEmbedCheckAt = now;
                      // Skip BOTH 256-D iris extraction AND 1024-D neural embedding early when frame is motion-blurred
                      if (snapBundle.sharpness < 2.8) {
                        blurryFrameStreak += 1;
                        if (blurryFrameStreak >= 2) {
                          updateReadinessHint(
                            resolveKioskReadinessGuidance({
                              isSpoof: isFrameSpoof,
                              isBioVerified: bio.biologicalVerified,
                              yawDeg: liveness.yawDeg,
                              pitchDeg: liveness.pitchDeg,
                              inspection: null,
                              blurryFrameStreak,
                              lang: langRef.current,
                            }),
                          );
                        }
                        if (blurryFrameStreak >= 4) {
                          serverStreakRef.current = { patientId: null, scores: [] };
                          setBurstCount(0);
                        }
                      } else {
                        const snapLandmarks = landmarks.map((p) => ({ x: p.x, y: p.y, z: p.z }));
                        const irisStartMs = performance.now();
                        const snapOcularIris = captureAndExtractOcularIris256(
                          videoRef.current,
                          landmarks,
                          irisLeftCanvasRef.current,
                          irisRightCanvasRef.current,
                        );
                        const snapIrisMs = performance.now() - irisStartMs;
                        void runAsyncBiometricMatchStep({
                          snapCanvas: snapBundle.canvas,
                          snapSharpness: snapBundle.sharpness,
                          snapPatch384: snapBundle.patch384,
                          snapRawBox: rawBox ?? rawBoxForBio ?? null,
                          snapLandmarks,
                          snapOcularIris,
                          snapIrisMs,
                          snapVideoW: videoRef.current.videoWidth || 1280,
                          snapVideoH: videoRef.current.videoHeight || 720,
                          snapYawDeg: liveness.yawDeg,
                          snapPitchDeg: liveness.pitchDeg,
                          snapFaceHeightRatio: rawBox?.height ?? bio.faceScale,
                          snapBioVerified: bio.biologicalVerified,
                          snapBioNonRigidScore: bio.nonRigidScore,
                          snapBioRppgScore: bio.rppgScore,
                          snapBioSpoofScore: bio.spoofScore,
                          snapBioBlinkCount: bio.blinkCount,
                          snapBioIrisLocked: bio.irisLocked,
                          snapIsSpoof: isFrameSpoof,
                          snapNow: now,
                        });
                      }
                    }
                  }
                }
              }
            }
          } catch {
            // Swallow transient MediaPipe / frame errors while async verification manages its own locks
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
  }, [
    apiClient,
    cameraPaused,
    isNotFoundStage,
    isSystemErrorStage,
    refreshBiometricLibraryStats,
    scanGeneration,
    selectedCameraId,
    showPinModal,
    staffModalOpen,
  ]);

  const handleCallStaff = async () => {
    try {
      await apiClient.createStaffCall(kioskId, 'ผู้ป่วยกดเรียกเจ้าหน้าที่จากหน้าตู้ Kiosk');
      showToast(c.staffCalledToast);
    } catch {
      showToast('ไม่สามารถส่งคำขอเรียกเจ้าหน้าที่ได้ กรุณาลองใหม่อีกครั้ง');
    }
  };

  const handlePinSubmit = async (digits: string) => {
    setPinSubmitting(true);
    setPinError(null);
    try {
      const normalized = digits.trim();
      const pinRes = await apiClient.verifyKioskPinLogin(normalized).catch(() => null);
      if (pinRes?.matched && pinRes.patient) {
        const pinPayload = buildKioskLoginSuccessPayload({
          patient: {
            id: pinRes.patient.id,
            hn: pinRes.patient.hn,
            fullName: pinRes.patient.fullName,
            nextAppointmentDate: pinRes.patient.nextAppointmentDate,
          },
          todayRoutine: pinRes.todayRoutine ?? null,
          confidencePercent: 100,
          rehabSchedules: pinRes.rehabSchedules,
          serverTodayExercises: pinRes.todayExercises,
          serverCompletedToday: pinRes.completedToday,
        });
        setShowPinModal(false);
        setCameraPaused(false);
        setPinDigits('');
        openVerifiedModal(pinPayload);
        return;
      }

      // Fallback to authenticated patient list only when staff session already exists in browser
      const patients: PatientRecord[] = await apiClient.listPatients();
      const matched = patients.find((p) => {
        const cleanPhone = (p.phone ?? '').replace(/\D/g, '');
        const cleanHn = (p.hn ?? '').replace(/\D/g, '');
        return (
          cleanPhone.endsWith(normalized) ||
          cleanHn.endsWith(normalized) ||
          (p.hn ?? '').toLowerCase() === normalized.toLowerCase()
        );
      });
      if (matched) {
        const matchedEntry =
          matched.masterEmbedding && matched.masterEmbedding.length > 0
            ? ({
                patientId: matched.id,
                hn: matched.hn,
                fullName: matched.fullName,
                diagnosis: matched.diagnosis,
                assignedPtId: matched.assignedPtId,
                nextAppointmentDate: matched.nextAppointmentDate,
                framesCaptured: matched.faceFramesCount ?? 120,
                qualityScore: 0.96,
                masterEmbedding: matched.masterEmbedding,
                angleEmbeddings: matched.angleEmbeddings ?? {
                  center: matched.masterEmbedding,
                  left: matched.masterEmbedding,
                  right: matched.masterEmbedding,
                  up: matched.masterEmbedding,
                  down: matched.masterEmbedding,
                },
                geometricRatios: matched.geometricRatios,
              } as PatientBiometricIndexEntry)
            : null;
        const pinPayload = buildKioskLoginSuccessPayload({
          patient: {
            id: matched.id,
            hn: matched.hn,
            fullName: matched.fullName,
            nextAppointmentDate: matched.nextAppointmentDate,
          },
          todayRoutine: null,
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
      if (pinSubmitTimerRef.current !== null) window.clearTimeout(pinSubmitTimerRef.current);
      pinSubmitTimerRef.current = window.setTimeout(() => {
        pinSubmitTimerRef.current = null;
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
    <div className="animate-view-enter min-h-screen flex flex-col bg-gradient-to-br from-[#EAF8F5] via-[#EFF6FF] to-white text-[#0F4C81] font-sans select-none overflow-x-hidden">
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

            {/* Right Controls: AI Voice Mute Toggle + Cloud Online + Clock & Compact Language Switcher */}
            <div className="flex flex-wrap items-center justify-end gap-2.5 shrink-0">
              <button
                type="button"
                onClick={handleToggleVoiceMute}
                aria-pressed={voiceMuted}
                data-testid="kiosk-login-voice-mute-toggle"
                className={`inline-flex items-center gap-2 rounded-xl border-2 px-3 py-1.5 text-xs sm:text-sm font-extrabold shadow-sm transition-all cursor-pointer active:scale-95 ${
                  voiceMuted
                    ? 'border-amber-300 bg-amber-100 text-amber-950 hover:bg-amber-200'
                    : 'border-emerald-300 bg-emerald-100 text-emerald-950 hover:bg-emerald-200'
                }`}
              >
                {voiceMuted ? (
                  <VolumeX className="h-4 w-4 shrink-0 text-amber-700" />
                ) : (
                  <Volume2 className="h-4 w-4 shrink-0 text-emerald-700" />
                )}
                <span>{getSeniorVoiceToggleLabel(lang, voiceMuted)}</span>
              </button>

              {onReturnToStandby && (
                <button
                  type="button"
                  data-testid="kiosk-enter-standby-btn"
                  onClick={() => {
                    cancelAllKioskVoiceSpeech(modalAudioRef.current);
                    onReturnToStandby();
                  }}
                  className="inline-flex items-center gap-1.5 rounded-xl border border-white/30 bg-white/15 hover:bg-white/25 px-3 py-1.5 text-xs sm:text-sm font-bold text-white shadow-xs transition-all cursor-pointer active:scale-95 whitespace-nowrap"
                >
                  <Moon className="h-4 w-4 shrink-0 text-cyan-200" />
                  <span>{lang === 'en' ? 'Standby' : lang === 'cn' ? '待机' : 'พักหน้าจอ'}</span>
                </button>
              )}

              {/* Cloud Online + Clock Pill */}
              <div className="hidden sm:inline-flex h-9 px-3 items-center gap-2 rounded-xl bg-white/15 border border-white/25 text-xs font-semibold text-white whitespace-nowrap">
                <span className="w-2 h-2 rounded-full bg-emerald-300 animate-status-halo shrink-0 shadow-[0_0_8px_#34d399]" />
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
              className="btn-primary-sheen h-10 w-full inline-flex items-center justify-center gap-2 px-3 rounded-xl bg-white hover:bg-emerald-50 text-[#0F4C81] text-xs sm:text-sm font-bold whitespace-nowrap active:scale-[0.99] transition shadow-sm cursor-pointer"
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
          <div className="mb-3 w-full rounded-2xl border border-amber-300 bg-amber-50 px-4 py-2.5 text-xs sm:text-sm font-bold text-amber-900 shadow-md flex items-center justify-between animate-toast-in">
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
                  {availableCameras.length >= 2 && (
                    <button
                      type="button"
                      onClick={handleCycleCamera}
                      className="px-2.5 py-1 rounded-lg bg-white/95 hover:bg-white border border-sky-200 text-[#0F4C81] text-[11px] font-bold flex items-center gap-1 shadow-xs cursor-pointer"
                      title={
                        availableCameras.find((d) => d.deviceId === (activeCameraId ?? selectedCameraId))?.label ||
                        'สลับกล้อง'
                      }
                    >
                      <Camera className="w-3 h-3 text-emerald-600" />
                      <span>
                        กล้อง {(availableCameras.findIndex((d) => d.deviceId === (activeCameraId ?? selectedCameraId)) + 1) || 1}/
                        {availableCameras.length}
                      </span>
                    </button>
                  )}
                  <button
                    type="button"
                    onClick={() => setRotationDeg((prev) => (((prev + 90) % 360) as 0 | 90 | 180 | 270))}
                    className="px-2.5 py-1 rounded-lg bg-white/95 hover:bg-white border border-sky-200 text-[#0F4C81] text-[11px] font-bold flex items-center gap-1 shadow-xs cursor-pointer"
                    title="หมุนภาพกล้อง"
                  >
                    <RotateCw className="w-3 h-3 text-[#0284C7]" />
                    <span>{rotationDeg}°</span>
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      preferredMirrorModeRef.current = null;
                      setMirrored((m) => !m);
                    }}
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
                    className={`px-2.5 py-1 rounded-lg border text-[11px] font-bold flex items-center gap-1 transition cursor-pointer shadow-xs ${
                      showCalibHud
                        ? 'bg-[#0284C7] border-sky-300 text-white'
                        : 'bg-white/95 hover:bg-white border-sky-200 text-[#0F4C81]'
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

                {/* Staff-Only Calibration & Step 0 Performance Telemetry HUD (never visible to patients; zero images/PII) */}
                {showCalibHud && (
                  <div
                    data-testid="staff-biometric-perf-hud"
                    className="absolute top-12 left-3 right-3 sm:right-auto sm:w-[370px] max-h-[74%] overflow-y-auto z-30 rounded-xl bg-white/95 border border-sky-300 p-3 text-[10px] font-mono text-[#0F4C81] shadow-2xl space-y-1.5 animate-toast-in"
                  >
                    <div className="flex items-center justify-between border-b border-sky-200 pb-1">
                      <span className="font-bold text-[11px] text-[#0F4C81]">Staff Biometric & Perf HUD</span>
                      <div className="flex items-center gap-1.5">
                        <button
                          type="button"
                          onClick={() => {
                            profilerRef.current.clearStoredRounds();
                            const resetSnap = profilerRef.current.getSnapshot(performance.now());
                            setStaffPerfSnapshot(resetSnap);
                          }}
                          className="px-1.5 py-0.5 rounded bg-sky-100 hover:bg-sky-200 text-[#0F4C81] text-[9px] font-bold cursor-pointer"
                        >
                          Reset
                        </button>
                        <button
                          type="button"
                          onClick={() => setShowCalibHud(false)}
                          aria-label="ปิดแถบข้อมูล"
                          className="text-slate-500 hover:text-[#0F4C81] cursor-pointer"
                        >
                          <X className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    </div>
                    <div className="text-[10px] text-[#0284C7] truncate">Host: {supabaseHost}</div>
                    <div className="grid grid-cols-3 gap-1 pt-0.5">
                      <div>3D Depth: {livenessInfo?.depthScore.toFixed(2) ?? '0.00'}</div>
                      <div>EAR: {livenessInfo?.earLeft.toFixed(2) ?? '0.00'}</div>
                      <div>BaseEAR: {bioReport?.openEyeBaseline.toFixed(2) ?? '0.00'}</div>
                      <div>Scale: {bioReport?.faceScale.toFixed(2) ?? '0.00'}</div>
                      <div>Blink: {bioReport?.blinkCount ?? 0}</div>
                      <div>YawΔ: {bioReport?.deltaYawDeg.toFixed(1) ?? '0.0'}°</div>
                      <div>Spoof: {Math.round((bioReport?.spoofScore ?? 0) * 100)}%</div>
                      <div>rPPG: {Math.round((bioReport?.rppgScore ?? 0) * 100)}%</div>
                      <div>Burst: {burstCount}/{KIOSK_FAST_MATCH_STREAK}</div>
                    </div>
                    {staffPerfSnapshot && (
                      <div className="pt-1.5 border-t border-sky-200 space-y-1">
                        <div className="flex items-center justify-between text-[10px] font-bold text-[#0284C7]">
                          <span>
                            Timeline ({staffPerfSnapshot.roundsCount}/{staffPerfSnapshot.targetRounds} rounds)
                          </span>
                          <span>
                            {staffPerfSnapshot.live.faceDelegate}/{staffPerfSnapshot.live.humanBackend} •{' '}
                            {staffPerfSnapshot.live.detectInputLabel}
                          </span>
                        </div>
                        <div className="grid grid-cols-2 gap-x-2 gap-y-0.5 text-[9.5px] leading-tight">
                          <div>
                            Cam Open: {staffPerfSnapshot.live.cameraOpenMs}ms (p:{staffPerfSnapshot.live.cameraProbeCount}
                            {staffPerfSnapshot.live.usedRememberedCamera ? ' mem' : ' scan'})
                          </div>
                          <div>
                            1st Frame: {staffPerfSnapshot.live.firstFrameMs}ms (AE:{staffPerfSnapshot.live.exposureSettleMs}ms)
                          </div>
                          <div>
                            Warmup: {staffPerfSnapshot.live.modelWarmupMs}ms (
                            {staffPerfSnapshot.live.wasPrewarmed ? 'prewarm' : 'cold'})
                          </div>
                          <div>
                            Thread: {staffPerfSnapshot.live.executionThread} | Skip:{staffPerfSnapshot.live.skippedStaleFrames}
                          </div>
                          <div>
                            Detect478: {staffPerfSnapshot.live.faceDetectMs}ms (ROI:{staffPerfSnapshot.live.roiZoomMs}ms)
                          </div>
                          <div>
                            FPS: scan {staffPerfSnapshot.live.scanFps} / ovl {staffPerfSnapshot.live.overlayFps}
                          </div>
                          <div>
                            Face-&gt;Ready: {staffPerfSnapshot.live.faceToReadyMs}ms
                          </div>
                          <div>
                            Liveness: {staffPerfSnapshot.live.livenessLockMs}ms ({staffPerfSnapshot.live.livenessCue})
                          </div>
                          <div>
                            Embed 1024D: {staffPerfSnapshot.live.humanEmbedMs}ms (Iris:{staffPerfSnapshot.live.irisExtractMs}ms)
                          </div>
                          <div>
                            Local 1:N: {staffPerfSnapshot.live.localMatchMs}ms | Ovl:{staffPerfSnapshot.live.overlayDrawMs}ms
                          </div>
                          <div>
                            Server RTT: {staffPerfSnapshot.live.serverVerifyMs}ms (-{staffPerfSnapshot.live.pipelinedOverlapSavedMs}ms)
                          </div>
                          <div>
                            Post-Lock: {staffPerfSnapshot.live.postLockToModalMs}ms
                          </div>
                        </div>
                        {staffPerfSnapshot.roundsCount > 0 && (
                          <div className="pt-1 border-t border-sky-100 text-[9px] text-slate-700 space-y-0.5">
                            <div className="font-bold text-[#0F4C81]">Rolling avg / p50 / p95 (ms):</div>
                            <div>
                              Cam: {staffPerfSnapshot.stats.cameraOpenMs.avgMs}/{staffPerfSnapshot.stats.cameraOpenMs.p50Ms}/
                              {staffPerfSnapshot.stats.cameraOpenMs.p95Ms} | Det:{' '}
                              {staffPerfSnapshot.stats.faceDetectAvgMs.avgMs}/{staffPerfSnapshot.stats.faceDetectAvgMs.p50Ms}/
                              {staffPerfSnapshot.stats.faceDetectAvgMs.p95Ms}
                            </div>
                            <div>
                              Emb: {staffPerfSnapshot.stats.humanEmbedAvgMs.avgMs}/{staffPerfSnapshot.stats.humanEmbedAvgMs.p50Ms}/
                              {staffPerfSnapshot.stats.humanEmbedAvgMs.p95Ms} | Srv:{' '}
                              {staffPerfSnapshot.stats.serverVerifyMs.avgMs}/{staffPerfSnapshot.stats.serverVerifyMs.p50Ms}/
                              {staffPerfSnapshot.stats.serverVerifyMs.p95Ms}
                            </div>
                            <div>
                              Total Face-&gt;OK: {staffPerfSnapshot.stats.totalFaceToVerifiedMs.avgMs}/
                              {staffPerfSnapshot.stats.totalFaceToVerifiedMs.p50Ms}/
                              {staffPerfSnapshot.stats.totalFaceToVerifiedMs.p95Ms}
                            </div>
                          </div>
                        )}
                        {staffPerfSnapshot.topBottlenecks.length > 0 && (
                          <div className="pt-0.5 text-[9px] font-bold text-amber-800 truncate">
                            Top-3:{' '}
                            {staffPerfSnapshot.topBottlenecks
                              .map((b, idx) => `${idx + 1}.${b.stage}(${b.avgMs}ms)`)
                              .join(' ')}
                          </div>
                        )}
                      </div>
                    )}
                    {staffDiagnostics && (
                      <div
                        data-testid="staff-biometric-diagnostics"
                        className="pt-1.5 border-t border-sky-200 space-y-0.5 text-[9.5px] leading-tight text-slate-800"
                      >
                        <div className="flex items-center justify-between font-bold text-[#0F4C81]">
                          <span>Reason: {staffDiagnostics.reasonCode}</span>
                          <span>
                            {staffDiagnostics.probeEngine} ({staffDiagnostics.probeDim}D) •{' '}
                            {staffDiagnostics.dbFresh ? 'DB:LIVE' : 'DB:CACHE'}
                          </span>
                        </div>
                        <div>
                          Library: {staffDiagnostics.compatibleTemplatesCount} compat /{' '}
                          {staffDiagnostics.versionMismatchCount} mismatch /{' '}
                          {staffDiagnostics.filteredOutCount} filtered (eval:{' '}
                          {staffDiagnostics.evaluatedCandidatesCount})
                        </div>
                        <div>
                          Top-1: {staffDiagnostics.top1Code ?? '-'} {staffDiagnostics.top1ScorePercent}% (cen{' '}
                          {staffDiagnostics.top1CenteredPercent}%) | Top-2: {staffDiagnostics.top2Code ?? '-'}{' '}
                          {staffDiagnostics.top2ScorePercent}% (d:{staffDiagnostics.marginPercent}%)
                        </div>
                        <div>
                          Thresh: &gt;= {staffDiagnostics.thresholdPercent}% | Liveness:{' '}
                          {staffDiagnostics.livenessPassed ? 'PASS' : 'WAIT'} | QFrames:{' '}
                          {staffDiagnostics.qualityPassedFrames}
                        </div>
                        {staffDiagnostics.gateFailCodes.length > 0 && (
                          <div className="text-amber-800 truncate">
                            Gates: {staffDiagnostics.gateFailCodes.join(', ')}
                          </div>
                        )}
                      </div>
                    )}
                    {lastVerifyResult?.matched && lastVerifyResult.confidencePercent >= 84 && (
                      <div className="text-[10px] text-emerald-700 font-bold pt-1 border-t border-sky-200">
                        Verified Match ({lastVerifyResult.confidencePercent}%)
                      </div>
                    )}
                  </div>
                )}

                {/* Top-Center Live Scan Progress Pill / 1.5s No-Face Framing Guidance Pill */}
                {!verifiedModalPayload &&
                  scanStage !== 'not_found' &&
                  scanStage !== 'system_error' &&
                  isFaceDetected &&
                  !isSpoof && (
                  <div
                    className="pointer-events-none absolute top-3 left-1/2 -translate-x-1/2 z-25 max-w-[90%]"
                    data-testid="kiosk-scan-progress-pill"
                  >
                    <div
                      className={`rounded-2xl bg-white/95 border shadow-lg px-4 py-2 text-center transition-colors duration-200 ${
                        readinessHint ? 'border-amber-300' : 'border-emerald-300'
                      }`}
                    >
                      <div className="text-xs sm:text-sm font-black text-[#0F4C81] whitespace-nowrap">
                        {scanStage === 'verifying'
                          ? c.verifying
                          : readinessHint
                            ? readinessHint
                            : `กำลังตรวจสอบใบหน้า & ม่านตา... (${Math.min(burstCount, KIOSK_FAST_MATCH_STREAK)}/${KIOSK_FAST_MATCH_STREAK})`}
                      </div>
                      <div className="mt-1 flex flex-wrap items-center justify-center gap-1.5 text-[10px] font-bold text-[#0284C7] whitespace-nowrap">
                        <span
                          className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 border ${
                            bioReport?.irisLocked || irisHudState.irisLocked
                              ? 'border-cyan-300 bg-cyan-50 text-cyan-800'
                              : 'border-sky-200 bg-sky-50 text-sky-700'
                          }`}
                        >
                          <span
                            className={`h-1.5 w-1.5 rounded-full ${
                              bioReport?.irisLocked || irisHudState.irisLocked
                                ? 'bg-cyan-500 animate-pulse'
                                : 'bg-sky-400'
                            }`}
                          />
                          {bioReport?.irisLocked || irisHudState.irisLocked
                            ? 'IRIS LOCK • 478-PT + 1024D'
                            : 'กำลังตรวจจับม่านตา • 478-PT'}
                        </span>
                        {irisHudState.ipdPx !== null && (
                          <span className="rounded-full border border-sky-200 bg-sky-50 px-2 py-0.5 text-sky-800 font-mono">
                            IPD {Math.round(irisHudState.ipdPx)}px
                          </span>
                        )}
                        {irisHudState.ocularMatchPct !== null && (
                          <span className="rounded-full border border-emerald-200 bg-emerald-50 px-2 py-0.5 text-emerald-700">
                            ม่านตา {irisHudState.ocularMatchPct}%
                          </span>
                        )}
                        {irisHudState.latencyMs !== null && (
                          <span className="rounded-full border border-slate-200 bg-slate-50 px-2 py-0.5 text-slate-600 font-mono">
                            {irisHudState.latencyMs}ms
                          </span>
                        )}
                      </div>
                      <div className="mt-1.5 h-1.5 w-44 max-w-full mx-auto overflow-hidden rounded-full bg-slate-200">
                        <div
                          className={`h-full w-full origin-left rounded-full transition-transform duration-150 ${
                            readinessHint
                              ? 'bg-gradient-to-r from-amber-400 to-sky-400'
                              : 'bg-gradient-to-r from-emerald-500 to-sky-500'
                          }`}
                          style={{
                            transform: `scaleX(${(
                              (scanStage === 'verifying'
                                ? 100
                                : Math.max(
                                    18,
                                    Math.round(
                                      (Math.min(burstCount, KIOSK_FAST_MATCH_STREAK) /
                                        KIOSK_FAST_MATCH_STREAK) *
                                        100,
                                    ),
                                  )) / 100
                            ).toFixed(3)})`,
                          }}
                        />
                      </div>
                    </div>
                  </div>
                )}

                {!verifiedModalPayload &&
                  scanStage !== 'not_found' &&
                  scanStage !== 'system_error' &&
                  !isFaceDetected &&
                  showNoFaceGuidance && (
                  <div
                    className="pointer-events-none absolute top-3 left-1/2 -translate-x-1/2 z-25 max-w-[92%]"
                    data-testid="kiosk-no-face-guidance-pill"
                  >
                    <div className="rounded-full bg-white/95 border-2 border-sky-400 shadow-lg px-4 py-2 text-xs sm:text-sm font-extrabold text-[#0F4C81] whitespace-nowrap animate-toast-in">
                      กรุณามองตรงที่กล้องและจัดใบหน้าให้อยู่ในกรอบ
                    </div>
                  </div>
                )}

                {/* Fine-Grained Facial Landmark Mesh Keypoints (60-120 FPS HiDPI GPU canvas) */}
                <canvas
                  ref={faceMeshCanvasRef}
                  className="pointer-events-none absolute inset-0 h-full w-full z-20"
                />

                {/* GPU-Composited Dynamic Face Status Pill (single canvas frame avoids double-box jitter & layout reflow) */}
                {hudBox && (() => {
                  const overlay =
                    isSpoof || !isBioVerified || !liveOverlay
                      ? resolveKioskFaceBoundingBoxOverlay({
                          isSpoof,
                          isBioVerified,
                          rawCosine: (liveDebugHud?.topPct ?? 0) / 100,
                          centeredCosine:
                            liveDebugHud?.centeredPct !== undefined
                              ? liveDebugHud.centeredPct / 100
                              : undefined,
                          hasEvaluatedMatch: Boolean(liveDebugHud?.hasEvaluatedMatch),
                          gatePassed: Boolean(liveDebugHud?.gatePassed ?? liveDebugHud?.topName),
                          streak: burstCount,
                          requiredStreak: KIOSK_FAST_MATCH_STREAK,
                          confirmedCandidateName:
                            liveMatchHud?.confirmed ? liveMatchHud.fullName : null,
                          confirmedCandidateHn:
                            liveMatchHud?.confirmed ? liveMatchHud.hn : null,
                          distanceM: bioReport?.estimatedDistanceM ?? 1.8,
                          confidencePercent: liveMatchHud?.confidencePercent ?? liveDebugHud?.topPct,
                          isLongRange: bioReport?.livenessMode === 'LONG_RANGE',
                          isAmbiguous: ambiguousCloseHint,
                        })
                      : liveOverlay;
                  const pillClass =
                    overlay.boxColor === 'rose'
                      ? 'border-rose-400 bg-rose-50/95 text-rose-700'
                      : overlay.boxColor === 'emerald'
                        ? 'border-[#10B981] bg-emerald-50/95 text-emerald-800'
                        : overlay.boxColor === 'blue'
                          ? 'border-blue-400 bg-blue-50/95 text-blue-900'
                          : 'border-amber-500 bg-amber-50/95 text-amber-900';
                  const pillDisplayLabel =
                    overlay.variant === 'UNREGISTERED' &&
                    !liveDebugHud?.hasEvaluatedMatch &&
                    readinessHint
                      ? readinessHint
                      : overlay.label;

                  return (
                    <div
                      ref={hudBoxElRef}
                      className="pointer-events-none absolute top-0 left-0 z-25 will-change-transform"
                      style={{
                        transform: `translate3d(${((hudBox.xPct + hudBox.wPct * 0.5) / 100) * (videoRef.current?.clientWidth || 640)}px, ${((hudBox.yPct + hudBox.hPct) / 100) * (videoRef.current?.clientHeight || 480) + 10}px, 0)`,
                      }}
                    >
                      <div className={`w-max max-w-[390px] -translate-x-1/2 rounded-2xl border-2 px-4 py-2.5 text-center shadow-2xl transition-colors duration-150 ${pillClass}`}>
                        <div className="inline-flex items-center justify-center gap-2 text-sm font-black leading-snug">
                          {overlay.boxColor === 'emerald' && (
                            <KineticEmblemSvg key={`hud-success-${scanGeneration}`} status="success" className="w-6 h-6 shrink-0 overflow-visible" />
                          )}
                          {overlay.boxColor === 'rose' && (
                            <KineticEmblemSvg key={`hud-not-found-${scanGeneration}`} status="not_found" className="w-6 h-6 shrink-0 overflow-visible" />
                          )}
                          <span>{pillDisplayLabel}</span>
                        </div>
                        {overlay.subLabel && (
                          <div className="mt-0.5 text-xs font-bold opacity-90">{overlay.subLabel}</div>
                        )}
                      </div>
                    </div>
                  );
                })()}

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

          {/* Patient-Safe Biometric Status Bar (never exposes candidate scores or raw internal error codes) */}
          <div
            data-testid="kiosk-biometric-debug-hud"
            className="mt-2 rounded-xl border border-sky-200/90 bg-slate-900/95 px-3 py-2 text-[11px] sm:text-xs font-mono text-slate-100 shadow-sm flex flex-wrap items-center justify-between gap-x-3 gap-y-1 shrink-0"
          >
            <div className="flex items-center gap-2">
              <span
                className={`inline-block h-2 w-2 rounded-full ${
                  supabaseSyncError && !supabaseSyncedFromCloud ? 'bg-amber-400' : 'bg-emerald-400'
                }`}
              />
              <span
                className={`font-bold ${
                  supabaseSyncError && !supabaseSyncedFromCloud ? 'text-amber-300' : 'text-emerald-300'
                }`}
              >
                {liveDebugHud?.syncedFromSupabase || supabaseSyncedFromCloud
                  ? `ซิงก์ฐานข้อมูลใบหน้าจาก Supabase สำเร็จ: ${liveDebugHud?.validAccounts ?? dbAccountCount} บัญชี`
                  : `คลังใบหน้า: ${liveDebugHud?.validAccounts ?? dbAccountCount} บัญชี`}
              </span>
              {(liveDebugHud?.legacyAccounts ?? legacyAccountCount) > 0 && (
                <span className="text-amber-300">
                  (ทั้งหมด {liveDebugHud?.totalAccounts ?? totalAccountCount} • บัญชีเก่ารออัปเดต{' '}
                  {liveDebugHud?.legacyAccounts ?? legacyAccountCount})
                </span>
              )}
            </div>
            <div className="text-sky-200 truncate max-w-full">
              {scanStage === 'system_error' ? (
                <span className="text-amber-300 font-bold">
                  ระบบตรวจสอบขัดข้องชั่วคราว กรุณาลองใหม่หรือเรียกเจ้าหน้าที่
                </span>
              ) : liveDebugHud?.topName ? (
                <>
                  ยืนยันเจ้าของบัญชี:{' '}
                  <span className="font-bold text-emerald-300">{liveDebugHud.topName}</span>{' '}
                  [{liveDebugHud.streak}/{liveDebugHud.requiredStreak}]
                </>
              ) : ambiguousCloseHint ? (
                <span className="text-amber-200">
                  พบใบหน้าใกล้เคียงหลายคน — กรุณามองตรงเข้าหากล้องและขยับเข้าใกล้ขึ้นเล็กน้อย
                </span>
              ) : (liveDebugHud?.validAccounts ?? dbAccountCount) === 0 ? (
                <span className="text-amber-200">
                  ยังไม่มีข้อมูลใบหน้าในคลัง (กรุณาลงทะเบียนคนไข้ใหม่ก่อนใช้งาน)
                </span>
              ) : liveDebugHud && liveDebugHud.streak > 0 ? (
                <span className="text-blue-200">
                  กำลังตรวจสอบใบหน้า... ({liveDebugHud.streak}/{liveDebugHud.requiredStreak})
                </span>
              ) : liveDebugHud?.hasEvaluatedMatch ? (
                <span className="text-rose-300 font-bold">
                  ไม่พบบัญชีที่ตรงกับใบหน้านี้ (กรุณาลงทะเบียนคนไข้ใหม่ก่อนใช้งาน)
                </span>
              ) : (
                <span className="text-slate-300">
                  กำลังตรวจสอบใบหน้า... (0/{KIOSK_FAST_MATCH_STREAK})
                </span>
              )}
            </div>
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
        <div className="fixed inset-0 z-50 bg-[#0F4C81]/35 backdrop-blur-md flex items-center justify-center p-4 animate-backdrop-fade">
          <div className="w-full max-w-sm rounded-3xl bg-white border border-emerald-500/30 p-6 text-center shadow-2xl relative animate-popup-spring">
            <button
              type="button"
              onClick={() => {
                if (pinSubmitTimerRef.current !== null) {
                  window.clearTimeout(pinSubmitTimerRef.current);
                  pinSubmitTimerRef.current = null;
                }
                setShowPinModal(false);
              }}
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
              <div className="mb-3 rounded-xl bg-rose-50 border border-rose-200 px-3 py-2 text-xs font-bold text-rose-700 animate-toast-in animate-gentle-shake">
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
                  if (pinSubmitTimerRef.current !== null) {
                    window.clearTimeout(pinSubmitTimerRef.current);
                    pinSubmitTimerRef.current = null;
                  }
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
                onClick={() => {
                  if (pinSubmitTimerRef.current !== null) {
                    window.clearTimeout(pinSubmitTimerRef.current);
                    pinSubmitTimerRef.current = null;
                  }
                  setPinDigits((d) => d.slice(0, -1));
                }}
                className="h-12 rounded-xl bg-slate-100 border border-slate-200 text-slate-600 font-bold text-sm hover:bg-slate-200 cursor-pointer"
              >
                ⌫
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Temporary System Error / Rate Limit Popup (strictly separated from Account Not Found) */}
      {scanStage === 'system_error' && !staffModalOpen && (
        <KioskSystemErrorPopup
          reasonCode={systemErrorReason}
          onRescan={handleRestartScan}
          onUsePin={() => {
            setSystemErrorReason(null);
            setScanStage('scanning');
            setPinDigits('');
            setPinError(null);
            setShowPinModal(true);
          }}
          onCallStaff={() => {
            void handleCallStaff();
            handleRestartScan();
          }}
        />
      )}

      {/* Account Not Found -> Automatic Registration Prompt Popup */}
      {scanStage === 'not_found' && !staffModalOpen && (
        <UnregisteredFaceSignupPopup
          scanKey={scanGeneration}
          isSpeaking={isNotFoundVoiceSpeaking}
          onRegisterNewPatient={() => {
            cancelAllKioskVoiceSpeech(modalAudioRef.current);
            setIsNotFoundVoiceSpeaking(false);
            setScanStage('scanning');
            onOpenStaffRegister();
          }}
          onRescan={handleRestartScan}
          onCallStaff={() => {
            void handleCallStaff();
            handleRestartScan();
          }}
        />
      )}

      {/* Verified Patient Confirmation Modal */}
      {verifiedModalPayload && modalCopy && (
        <div className="fixed inset-0 z-50 bg-[#0F4C81]/40 backdrop-blur-md flex items-center justify-center p-4 animate-backdrop-fade">
          <div className="w-full max-w-md rounded-3xl bg-white border-2 border-emerald-500/50 shadow-2xl p-6 text-center animate-popup-spring">
            {/* 3-Language Selector Bar & Senior Voice Toggle inside Verified Patient Modal */}
            <div className="mb-4 flex flex-col items-center gap-2.5">
              <div
                className="inline-flex items-center gap-1 rounded-2xl border border-sky-200 bg-sky-50/90 p-1"
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

              <button
                type="button"
                onClick={handleToggleVoiceMute}
                aria-pressed={voiceMuted}
                data-testid="verified-modal-voice-mute-toggle"
                className={`inline-flex items-center justify-center gap-2 rounded-2xl border-2 px-4 py-2 text-xs sm:text-sm font-extrabold shadow-xs transition-all cursor-pointer active:scale-95 ${
                  voiceMuted
                    ? 'border-amber-400 bg-amber-100 text-amber-950 hover:bg-amber-200'
                    : 'border-emerald-400 bg-emerald-100 text-emerald-950 hover:bg-emerald-200'
                }`}
              >
                {voiceMuted ? (
                  <VolumeX className="h-4 w-4 shrink-0 text-amber-700" />
                ) : (
                  <Volume2 className="h-4 w-4 shrink-0 text-emerald-700" />
                )}
                <span>{getSeniorVoiceToggleLabel(lang, voiceMuted)}</span>
              </button>
            </div>

            <div className="mb-3" data-anim-complete={verifiedAnimDone ? 'true' : 'false'}>
              <KineticAuthStatusBadge
                key={`kiosk-success-${verifiedModalPayload.patient.id}-${scanGeneration}`}
                status="success"
                compact
                testId="kiosk-face-success-panel"
                subtitle={`${verifiedModalPayload.patient.fullName} • ${modalCopy.confidenceBadge}`}
              />
            </div>
            <h2 className="text-2xl font-extrabold text-slate-900">
              {verifiedModalPayload.patient.fullName}
            </h2>
            <p className="text-xs text-slate-500 mt-1 font-mono">
              {modalCopy.hnAndNextAppointment}
            </p>

            {/* Zero-Black-Bar Full-Bleed Camera Preview for Continuous 1:1 Owner Lock & Owner-Bound Hand Raising */}
            <div className="mt-4 relative mx-auto w-full aspect-[16/10] md:aspect-[16/9] overflow-hidden rounded-2xl bg-emerald-950 border-2 border-emerald-300 shadow-inner">
              <video
                ref={confirmVideoRef}
                autoPlay
                playsInline
                muted
                className="absolute inset-0 w-full h-full object-cover scale-x-[-1]"
              />
              <canvas
                ref={confirmCanvasRef}
                className="pointer-events-none absolute inset-0 w-full h-full object-cover scale-x-[-1]"
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
                          : 'bg-emerald-950/75 text-emerald-100 border border-white/25'
                      }`}
                    >
                      {leftHandUpState ? 'มือซ้ายพร้อม' : 'รอมือซ้าย'}
                    </span>
                    <span
                      className={`px-2.5 py-1 rounded-full text-[11px] font-black shadow ${
                        rightHandUpState
                          ? 'bg-emerald-600 text-white'
                          : 'bg-emerald-950/75 text-emerald-100 border border-white/25'
                      }`}
                    >
                      {rightHandUpState ? 'มือขวาพร้อม' : 'รอมือขวา'}
                    </span>
                  </>
                )}
              </div>
              <div className="absolute inset-x-2 bottom-2 flex items-center justify-center pointer-events-none">
                {ownerGuardBanner ? (
                  <span
                    key={`guard-${ownerGuardBanner.status}`}
                    className={`px-3 py-1 rounded-full text-[11px] font-extrabold border shadow-lg ${
                      ownerGuardBanner.status === 'IMPOSTOR_FACE' ||
                      ownerGuardBanner.status === 'INTRUDER_HAND'
                        ? 'bg-rose-600/95 text-white border-rose-200'
                        : 'bg-amber-500/95 text-white border-amber-200'
                    }`}
                  >
                    {ownerGuardBanner.text}
                  </span>
                ) : !isWelcomeSpeechDone ? (
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

            {/* Dedicated 4-Second Loading Bar (locked strictly at 0% until welcome voice completes + Owner 1:1 Lock passes) */}
            <div
              className={`mt-3 rounded-2xl border p-3 text-left ${
                ownerGuardBanner
                  ? ownerGuardBanner.status === 'IMPOSTOR_FACE' ||
                    ownerGuardBanner.status === 'INTRUDER_HAND'
                    ? 'border-rose-300 bg-rose-50/90'
                    : 'border-amber-300 bg-amber-50/85'
                  : !isWelcomeSpeechDone
                    ? raisedDuringSpeech
                      ? 'border-amber-200 bg-amber-50/75'
                      : 'border-slate-200 bg-slate-50/90'
                    : 'border-emerald-200 bg-emerald-50/70'
              }`}
            >
              <div
                className={`flex items-center justify-between gap-2 text-[11px] font-extrabold mb-1.5 ${
                  ownerGuardBanner
                    ? ownerGuardBanner.status === 'IMPOSTOR_FACE' ||
                      ownerGuardBanner.status === 'INTRUDER_HAND'
                      ? 'text-rose-800'
                      : 'text-amber-900'
                    : !isWelcomeSpeechDone
                      ? raisedDuringSpeech
                        ? 'text-amber-900'
                        : 'text-slate-700'
                      : 'text-emerald-950'
                }`}
              >
                <span className="truncate">
                  {ownerGuardBanner
                    ? ownerGuardBanner.text
                    : !isWelcomeSpeechDone
                      ? raisedDuringSpeech
                        ? modalCopy.speechEarlyRaiseBanner
                        : modalCopy.speechBarHeader
                      : twoHandDetected
                        ? 'กำลังโหลดยืนยัน 2 มือค้าง 4 วินาที...'
                        : leftHandUpState !== rightHandUpState
                          ? 'ยกมือเดียวไม่นับ กรุณายกขึ้นทั้ง 2 ข้าง'
                          : 'หลอดโหลดเริ่มฝึก (ยก 2 ข้างค้าง 4 วิ)'}
                </span>
                <span
                  key={isWelcomeSpeechDone ? lang : 'locked'}
                  ref={confirmProgressTextRef}
                  className={`font-mono shrink-0 ${
                    ownerGuardBanner
                      ? ownerGuardBanner.status === 'IMPOSTOR_FACE' ||
                        ownerGuardBanner.status === 'INTRUDER_HAND'
                        ? 'text-rose-700'
                        : 'text-amber-800'
                      : !isWelcomeSpeechDone
                        ? 'text-slate-500'
                        : 'text-emerald-700'
                  }`}
                >
                  0.0 / {(VERIFIED_HAND_HOLD_MS / 1000).toFixed(0)}.0 วิ (0%)
                </span>
              </div>
              <div className="h-4 w-full overflow-hidden rounded-full bg-slate-200 border border-slate-300 shadow-inner">
                <div
                  ref={confirmProgressFillRef}
                  className={`h-full w-full origin-left rounded-full ${
                    ownerGuardBanner
                      ? 'bg-rose-500'
                      : !isWelcomeSpeechDone
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
                onClick={() => {
                  excludedPatientIdsRef.current = [];
                  onPatientAuthenticated(verifiedModalPayload);
                }}
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
                onClick={handleRejectWrongMatchAndRescan}
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
        <div className="fixed inset-0 z-50 bg-[#0F4C81]/35 backdrop-blur-md flex items-center justify-center p-4 animate-backdrop-fade">
          <div className="w-full max-w-md rounded-3xl bg-white border border-emerald-500/30 p-6 shadow-2xl relative animate-popup-spring">
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

