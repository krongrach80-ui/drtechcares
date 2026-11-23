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
import { normalizePatientRecord, normalizePatientRecordList } from '../lib/patientHydration';
import {
  BIOMETRIC_CENTERED_COSINE_THRESHOLD,
  BIOMETRIC_ENGINE_ID,
  BIOMETRIC_MATCH_THRESHOLD,
  REQUIRED_MATCH_STREAK,
  captureAndExtractOcularIris256,
  captureCanonicalFaceSnapshot,
  captureCanonicalFaceSnapshotWithMetrics,
  captureWideCanonicalFaceFallback,
  computeBestMultiAngleSimilarity,
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
  smoothOwnerFaceBoxMetrics,
  verifyOwnerFace1to1,
  type OwnerBiometricTemplate,
  type OwnerBoundHandEvaluation,
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

export const KIOSK_FAST_MATCH_STREAK = 2;
export const KIOSK_STREAK_GRACE_MS = 500;
export const KIOSK_INFERENCE_INTERVAL_MS = 190;
export const KIOSK_INFERENCE_CANVAS_MAX_WIDTH = 320;
export const KIOSK_MIN_CLEAR_FACE_SCORE = 0.65;
export const KIOSK_MIN_CLEAR_FACE_WIDTH_RATIO = 0.18;
export const KIOSK_FAST_UNREGISTERED_FRAMES = 3;
export const KIOSK_FAST_UNREGISTERED_TIMEOUT_MS = 1100;
export const KIOSK_MAX_DECISION_WINDOW_MS = 2600;
export const KIOSK_MAX_DECISION_EVAL_FRAMES = 6;
export const KIOSK_NO_FACE_GUIDANCE_DELAY_MS = 1500;
export const KIOSK_RESCAN_COOLDOWN_MS = 1500;
export const KIOSK_MAX_EXPOSURE_SETTLE_MS = 800;
export const KIOSK_DETECT_MAX_LONG_SIDE_PX = KIOSK_INFERENCE_CANVAS_MAX_WIDTH;

/**
 * Non-overlapping async inference mutex with ~190ms (180-220ms) throttle interval so
 * WebGL/WASM face inference never blocks the 60FPS browser render thread or overlaps.
 */
export class AsyncInferenceMutex {
  private inferring = false;
  private lastDispatchMs = -Infinity;
  public readonly minIntervalMs: number;

  constructor(minIntervalMs = KIOSK_INFERENCE_INTERVAL_MS) {
    this.minIntervalMs = Math.max(100, minIntervalMs);
  }

  get isBusy(): boolean {
    return this.inferring;
  }

  canDispatch(nowMs: number): boolean {
    return !this.inferring && nowMs - this.lastDispatchMs >= this.minIntervalMs;
  }

  tryRun<T>(nowMs: number, task: () => Promise<T>): Promise<T> | null {
    if (!this.canDispatch(nowMs)) {
      return null;
    }
    this.inferring = true;
    this.lastDispatchMs = nowMs;
    return Promise.resolve()
      .then(task)
      .finally(() => {
        this.inferring = false;
      });
  }

  reset(): void {
    this.inferring = false;
    this.lastDispatchMs = -Infinity;
  }
}

/**
 * Forgiving fast-scan face readiness gate: accepts any clear, non-spoof face detection
 * (`faceScore >= 0.65` and bounding box width `>= 18%` of frame width) without fragile
 * iris-focus or micro-stillness locks.
 */
export function isClearFaceForFastScan(input: {
  faceScore: number;
  boxWidthRatio: number;
  isSpoof?: boolean;
}): boolean {
  if (input.isSpoof) return false;
  return (
    Number.isFinite(input.faceScore) &&
    input.faceScore >= KIOSK_MIN_CLEAR_FACE_SCORE &&
    Number.isFinite(input.boxWidthRatio) &&
    input.boxWidthRatio >= KIOSK_MIN_CLEAR_FACE_WIDTH_RATIO
  );
}

export interface FastScanStreakState {
  patientId: string | null;
  streak: number;
  scores: number[];
  lastMatchAtMs: number;
  dropoutStartedAtMs: number | null;
}

export function createInitialFastScanStreakState(): FastScanStreakState {
  return {
    patientId: null,
    streak: 0,
    scores: [],
    lastMatchAtMs: -Infinity,
    dropoutStartedAtMs: null,
  };
}

/**
 * Advances the 2-frame fast confirmation streak (`1/2` -> `2/2`) with a 500ms grace buffer
 * before resetting to `0` on transient single-frame dropouts or flicker.
 */
export function stepFastScanStreakWithGrace(
  prev: FastScanStreakState,
  event:
    | { type: 'match'; patientId: string; score: number; nowMs: number }
    | { type: 'dropout'; nowMs: number; forceReset?: boolean },
  options?: { requiredStreak?: number; graceMs?: number },
): FastScanStreakState & { confirmed: boolean; didReset: boolean } {
  const required = Math.max(1, options?.requiredStreak ?? KIOSK_FAST_MATCH_STREAK);
  const graceMs = Math.max(0, options?.graceMs ?? KIOSK_STREAK_GRACE_MS);

  if (event.type === 'match') {
    const withinWindow =
      prev.patientId === event.patientId &&
      Number.isFinite(prev.lastMatchAtMs) &&
      event.nowMs - prev.lastMatchAtMs <= Math.max(3000, graceMs * 6);
    const nextScores = withinWindow ? [...prev.scores, event.score] : [event.score];
    const minScore = Math.min(...nextScores);
    const maxScore = Math.max(...nextScores);
    const stableScores = maxScore - minScore <= 0.12 ? nextScores : [event.score];
    const streak = Math.min(required, stableScores.length);
    return {
      patientId: event.patientId,
      streak,
      scores: stableScores,
      lastMatchAtMs: event.nowMs,
      dropoutStartedAtMs: null,
      confirmed: streak >= required,
      didReset: false,
    };
  }

  if (event.forceReset || prev.streak === 0 || !prev.patientId) {
    return {
      ...createInitialFastScanStreakState(),
      confirmed: false,
      didReset: prev.streak > 0,
    };
  }

  const dropoutStart = prev.dropoutStartedAtMs ?? event.nowMs;
  if (event.nowMs - dropoutStart <= graceMs) {
    return {
      ...prev,
      dropoutStartedAtMs: dropoutStart,
      confirmed: false,
      didReset: false,
    };
  }

  return {
    ...createInitialFastScanStreakState(),
    confirmed: false,
    didReset: true,
  };
}

/**
 * Formats the Kiosk Supabase sync status pill clearly so staff and patients see both
 * face-verified accounts ready for scanning and total registered patients when PENDING
 * accounts exist in `public.patients`.
 */
export function formatKioskSupabaseSyncBadgeLabel(
  verifiedCount: number,
  totalPatients: number,
): string {
  const safeVerified = Math.max(0, Math.round(Number.isFinite(verifiedCount) ? verifiedCount : 0));
  const safeTotal = Math.max(0, Math.round(Number.isFinite(totalPatients) ? totalPatients : 0));
  if (safeTotal > safeVerified) {
    return `ซิงก์ใบหน้าพร้อมสแกนจาก Supabase: ${safeVerified} บัญชี (คนไข้ในระบบทั้งหมด ${safeTotal} บัญชี)`;
  }
  return `ซิงก์ฐานข้อมูลใบหน้าจาก Supabase สำเร็จ: ${safeVerified} บัญชี`;
}
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
  hasMatchedPatient?: boolean;
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
    params.systemError ||
    params.hasMatchedPatient
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

export const PASSIVE_LIVENESS_MIN_PIXEL_SHIFT = 0.8;

export interface PassiveLivenessAnchorsPx {
  boxCenterX: number;
  boxCenterY: number;
  noseX?: number;
  noseY?: number;
  jawX?: number;
  jawY?: number;
  yawDeg?: number;
  pitchDeg?: number;
}

/**
 * Passive live-webcam liveness check: detects natural sub-pixel webcam micro-movement
 * (>= 0.8px shift in face box center or nose/jaw landmarks across consecutive frames),
 * natural head movement, or biological blink/liveness WITHOUT forcing exaggerated eye blinks
 * after a registered patient has already matched.
 */
export function evaluatePassiveWebcamLiveness(params: {
  prevAnchors: PassiveLivenessAnchorsPx | null;
  nextAnchors: PassiveLivenessAnchorsPx | null;
  bioBiologicalVerified?: boolean;
  blinkCount?: number;
  isSpoof: boolean;
  alreadyLiveVerified?: boolean;
}): {
  liveVerified: boolean;
  pixelShiftPx: number;
  signalReason: 'spoof' | 'bio_verified' | 'micro_motion' | 'head_motion' | 'retained' | 'pending';
} {
  if (params.isSpoof) {
    return { liveVerified: false, pixelShiftPx: 0, signalReason: 'spoof' };
  }
  if (params.bioBiologicalVerified || (params.blinkCount ?? 0) > 0) {
    return { liveVerified: true, pixelShiftPx: 0, signalReason: 'bio_verified' };
  }
  let maxShiftPx = 0;
  let headAngleDelta = 0;
  if (params.prevAnchors && params.nextAnchors) {
    const boxShift = Math.hypot(
      params.nextAnchors.boxCenterX - params.prevAnchors.boxCenterX,
      params.nextAnchors.boxCenterY - params.prevAnchors.boxCenterY,
    );
    maxShiftPx = Math.max(maxShiftPx, boxShift);
    if (
      params.prevAnchors.noseX !== undefined &&
      params.prevAnchors.noseY !== undefined &&
      params.nextAnchors.noseX !== undefined &&
      params.nextAnchors.noseY !== undefined
    ) {
      maxShiftPx = Math.max(
        maxShiftPx,
        Math.hypot(
          params.nextAnchors.noseX - params.prevAnchors.noseX,
          params.nextAnchors.noseY - params.prevAnchors.noseY,
        ),
      );
    }
    if (
      params.prevAnchors.jawX !== undefined &&
      params.prevAnchors.jawY !== undefined &&
      params.nextAnchors.jawX !== undefined &&
      params.nextAnchors.jawY !== undefined
    ) {
      maxShiftPx = Math.max(
        maxShiftPx,
        Math.hypot(
          params.nextAnchors.jawX - params.prevAnchors.jawX,
          params.nextAnchors.jawY - params.prevAnchors.jawY,
        ),
      );
    }
    if (
      params.prevAnchors.yawDeg !== undefined &&
      params.nextAnchors.yawDeg !== undefined &&
      params.prevAnchors.pitchDeg !== undefined &&
      params.nextAnchors.pitchDeg !== undefined
    ) {
      headAngleDelta = Math.max(
        Math.abs(params.nextAnchors.yawDeg - params.prevAnchors.yawDeg),
        Math.abs(params.nextAnchors.pitchDeg - params.prevAnchors.pitchDeg),
      );
    }
  }
  if (maxShiftPx >= PASSIVE_LIVENESS_MIN_PIXEL_SHIFT) {
    return { liveVerified: true, pixelShiftPx: maxShiftPx, signalReason: 'micro_motion' };
  }
  if (headAngleDelta >= 0.35) {
    return { liveVerified: true, pixelShiftPx: maxShiftPx, signalReason: 'head_motion' };
  }
  if (params.alreadyLiveVerified) {
    return { liveVerified: true, pixelShiftPx: maxShiftPx, signalReason: 'retained' };
  }
  return { liveVerified: false, pixelShiftPx: maxShiftPx, signalReason: 'pending' };
}

/**
 * Normalizes and immediately prepends/upserts a newly registered patient into the Kiosk's
 * active in-memory patient list so newly enrolled users can scan and log in immediately
 * without reloading the page.
 */
export function mergeEnrolledPatientIntoKioskList(
  existing: readonly PatientRecord[],
  incoming: PatientRecord,
): PatientRecord[] {
  const normalized = normalizePatientRecord(incoming, {
    requireRealVectorsForVerified: false,
  });
  const hasVectors = Boolean(
    (normalized.masterEmbedding && normalized.masterEmbedding.length >= 128) ||
      (normalized.angleEmbeddings?.center && normalized.angleEmbeddings.center.length >= 128),
  );
  const readyPatient: PatientRecord = {
    ...normalized,
    faceBiometricStatus: hasVectors ? 'VERIFIED' : normalized.faceBiometricStatus,
    faceFramesCount:
      normalized.faceFramesCount && normalized.faceFramesCount > 0
        ? normalized.faceFramesCount
        : hasVectors
          ? 120
          : 0,
  };
  const incomingIdLo = readyPatient.id.trim().toLowerCase();
  const incomingHnLo = readyPatient.hn.trim().toLowerCase();
  const rest = existing.filter((p) => {
    const pidLo = (p.id ?? '').trim().toLowerCase();
    const hnLo = (p.hn ?? '').trim().toLowerCase();
    if (incomingIdLo && pidLo === incomingIdLo) return false;
    if (incomingHnLo && hnLo === incomingHnLo) return false;
    return true;
  });
  return [readyPatient, ...rest];
}

/**
 * Strictly matches a live 1024D embedding against locally injected newly-registered patients
 * (`faceBiometricStatus === 'VERIFIED'`) using best-of-all-angles cosine similarity while
 * preserving strict identity thresholds so unregistered strangers remain rejected.
 */
export function matchLocalEnrolledPatientsStrict(
  probeEmbedding: readonly number[],
  enrolledPatients: readonly PatientRecord[],
  excludedIds: readonly string[] = [],
  matchThreshold = BIOMETRIC_MATCH_THRESHOLD,
): { patient: PatientRecord; similarity: number; confidencePercent: number } | null {
  if (!probeEmbedding || probeEmbedding.length < 128 || enrolledPatients.length === 0) {
    return null;
  }
  const excludedSet = new Set(excludedIds.map((id) => id.trim().toLowerCase()).filter(Boolean));
  let best: { patient: PatientRecord; similarity: number } | null = null;
  for (const pat of enrolledPatients) {
    if (pat.faceBiometricStatus !== 'VERIFIED') continue;
    if (
      excludedSet.has((pat.id ?? '').trim().toLowerCase()) ||
      excludedSet.has((pat.hn ?? '').trim().toLowerCase())
    ) {
      continue;
    }
    const score = computeBestMultiAngleSimilarity(probeEmbedding, {
      masterEmbedding: pat.masterEmbedding,
      angleEmbeddings: pat.angleEmbeddings,
      exemplarEmbeddings: pat.exemplarEmbeddings,
    });
    if (score.evaluatedCount === 0) continue;
    if (score.maxSimilarity >= matchThreshold && (!best || score.maxSimilarity > best.similarity)) {
      best = { patient: pat, similarity: score.maxSimilarity };
    }
  }
  if (!best) return null;
  return {
    patient: best.patient,
    similarity: best.similarity,
    confidencePercent: Number((Math.min(1, best.similarity) * 100).toFixed(1)),
  };
}

/**
 * Single source of truth for Kiosk face-scan progress shared by both the top status pill
 * and the bottom-right counter so they NEVER contradict each other (`2/2` vs `0/2`).
 */
export function resolveKioskUnifiedScanProgress(params: {
  scanMatchCount: number;
  requiredStreak?: number;
  isMatched: boolean;
  readinessHint?: string | null;
}): {
  clampedCount: number;
  required: number;
  isComplete: boolean;
  topLabel: string;
  counterLabel: string;
  progressPct: number;
} {
  const required = Math.max(1, params.requiredStreak ?? KIOSK_FAST_MATCH_STREAK);
  const clampedCount = params.isMatched
    ? required
    : Math.max(0, Math.min(required, Math.floor(params.scanMatchCount || 0)));
  const isComplete = params.isMatched || clampedCount >= required;
  const counterLabel = `กำลังตรวจสอบใบหน้า... (${clampedCount}/${required})`;
  const topLabel = isComplete
    ? `ยืนยันสำเร็จ! (${required}/${required})`
    : params.readinessHint
      ? params.readinessHint
      : counterLabel;
  const progressPct = isComplete
    ? 100
    : Math.max(18, Math.round((clampedCount / required) * 100));
  return {
    clampedCount,
    required,
    isComplete,
    topLabel,
    counterLabel,
    progressPct,
  };
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

export function buildKioskLoginSuccessPayload(params: {
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
  recentlyEnrolledPatient?: PatientRecord | null;
  onRecentlyEnrolledConsumed?: () => void;
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

export const VERIFIED_HAND_HOLD_MS = 2000;
export const HAND_SWITCH_GRACE_MS = 600;
export const HAND_GLITCH_GRACE_MS = 250;
export const VERIFIED_MODAL_SPEECH_RATE = 1.12;
export const VERIFIED_MODAL_TTS_STALL_TIMEOUT_MS = 2500;

export interface ActiveVideoCandidateLike {
  readyState: number;
  videoWidth: number;
  videoHeight?: number;
  paused?: boolean;
  ended?: boolean;
  currentTime?: number;
}

/**
 * Selects the actively playing camera `<video>` element during the post-match `Success`
 * confirmation card so face tracking and dual-hand raise detection NEVER read from a
 * paused or frozen video element.
 */
export function resolvePostMatchActiveVideoElement<T extends ActiveVideoCandidateLike>(
  confirmVideo: T | null | undefined,
  scanVideo: T | null | undefined,
): T | null {
  const isLivePlaying = (el: T | null | undefined): boolean =>
    Boolean(el && el.readyState >= 2 && el.videoWidth > 0 && !el.paused && !el.ended);

  if (confirmVideo && isLivePlaying(confirmVideo)) return confirmVideo;
  if (scanVideo && isLivePlaying(scanVideo)) return scanVideo;
  if (confirmVideo && confirmVideo.readyState >= 2 && confirmVideo.videoWidth > 0) {
    return confirmVideo;
  }
  if (scanVideo && scanVideo.readyState >= 2 && scanVideo.videoWidth > 0) {
    return scanVideo;
  }
  return confirmVideo ?? scanVideo ?? null;
}

export interface PostMatchGestureGateState {
  targetFaceBox: OwnerFaceBoxMetrics | null;
  smoothedFaceBox: OwnerFaceBoxMetrics | null;
  holdMs: number;
  lastRaisedAtMs: number;
  leftHandRaised: boolean;
  rightHandRaised: boolean;
  bothHandsRaised: boolean;
  confirmed: boolean;
  handEval: OwnerBoundHandEvaluation | null;
}

export function createInitialPostMatchGestureGateState(): PostMatchGestureGateState {
  return {
    targetFaceBox: null,
    smoothedFaceBox: null,
    holdMs: 0,
    lastRaisedAtMs: -Infinity,
    leftHandRaised: false,
    rightHandRaised: false,
    bothHandsRaised: false,
    confirmed: false,
    handEval: null,
  };
}

/**
 * Pure 60FPS frame stepper for the `post_match_gesture_gate` mode inside the Kiosk `Success`
 * confirmation card:
 * 1. Smoothly tracks the live owner face bounding box on every frame (including during speech).
 * 2. Evaluates dual-detector hand landmarks + pose wrists to update left/right hand badges.
 * 3. Smoothly accumulates the 4.0-second dual-hand hold timer (`deltaMs / 1000`) with a
 *    250ms brief-drop grace buffer and gentle decay, triggering confirmation at >= 4.0s.
 */
export function stepPostMatchGestureGateFrame(
  prev: PostMatchGestureGateState,
  input: {
    nowMs: number;
    dtMs: number;
    speechBlocked: boolean;
    ownerName: string;
    ownerFaceVerified: boolean;
    faceLandmarks?: ReadonlyArray<{ x: number; y: number }> | null;
    faceCount: number;
    hands: ReadonlyArray<ReadonlyArray<{ x: number; y: number }>>;
    pose?: ReadonlyArray<{ x: number; y: number; visibility?: number }> | null;
    poseCount?: number;
    videoW: number;
    videoH: number;
    canvasW: number;
    canvasH: number;
    runInference?: boolean;
  },
): PostMatchGestureGateState & {
  progress: number;
  holdSeconds: number;
  isGuardBlocked: boolean;
  shouldProceedToExercise: boolean;
} {
  let targetFaceBox = prev.targetFaceBox;
  let handEval = prev.handEval;
  let leftHandRaised = prev.leftHandRaised;
  let rightHandRaised = prev.rightHandRaised;
  let bothHandsRaised = prev.bothHandsRaised;
  let holdMs = prev.holdMs;
  let lastRaisedAtMs = prev.lastRaisedAtMs;

  if (input.runInference !== false) {
    targetFaceBox =
      input.faceLandmarks && input.faceLandmarks.length >= 10 && input.faceCount > 0
        ? computeOwnerFaceBoxMetrics(
            input.faceLandmarks,
            input.videoW,
            input.videoH,
            input.canvasW,
            input.canvasH,
          )
        : null;
  }

  const smoothedFaceBox = smoothOwnerFaceBoxMetrics(prev.smoothedFaceBox, targetFaceBox, input.dtMs);

  if (input.speechBlocked) {
    return {
      targetFaceBox,
      smoothedFaceBox,
      holdMs: 0,
      lastRaisedAtMs: -Infinity,
      leftHandRaised: false,
      rightHandRaised: false,
      bothHandsRaised: false,
      confirmed: false,
      handEval: null,
      progress: 0,
      holdSeconds: 0,
      isGuardBlocked: false,
      shouldProceedToExercise: false,
    };
  }

  let isGuardBlocked = false;
  if (input.runInference !== false) {
    handEval = evaluateOwnerBoundKinematicHands({
      ownerFaceVerified: input.ownerFaceVerified,
      ownerFaceBox: targetFaceBox,
      faceCount: input.faceCount,
      ownerName: input.ownerName,
      hands: input.hands,
      pose: input.pose,
      poseCount: input.poseCount,
      videoW: input.videoW,
      videoH: input.videoH,
      canvasW: input.canvasW,
      canvasH: input.canvasH,
    });
    isGuardBlocked =
      handEval.status === 'IMPOSTOR_FACE' ||
      handEval.status === 'NO_OWNER_FACE' ||
      handEval.status === 'MULTI_PERSON' ||
      handEval.status === 'INTRUDER_HAND';

    if (isGuardBlocked) {
      leftHandRaised = false;
      rightHandRaised = false;
      bothHandsRaised = false;
      holdMs = 0;
      lastRaisedAtMs = -Infinity;
    } else {
      leftHandRaised = handEval.leftUp;
      rightHandRaised = handEval.rightUp;
      bothHandsRaised = handEval.bothUp;
    }
  } else if (handEval) {
    isGuardBlocked =
      handEval.status === 'IMPOSTOR_FACE' ||
      handEval.status === 'NO_OWNER_FACE' ||
      handEval.status === 'MULTI_PERSON' ||
      handEval.status === 'INTRUDER_HAND';
  }

  const holdStep = stepVerifiedHandHold({
    bothHandsNow: bothHandsRaised && !isGuardBlocked,
    speechBlocked: isGuardBlocked,
    nowMs: input.nowMs,
    dtMs: input.dtMs,
    lastRaisedAtMs: isGuardBlocked ? -Infinity : lastRaisedAtMs,
    holdMs: isGuardBlocked ? 0 : holdMs,
  });

  const confirmed = prev.confirmed || holdStep.confirmed;
  return {
    targetFaceBox,
    smoothedFaceBox,
    holdMs: holdStep.holdMs,
    lastRaisedAtMs: holdStep.lastRaisedAtMs,
    leftHandRaised,
    rightHandRaised,
    bothHandsRaised: holdStep.detected,
    confirmed,
    handEval,
    progress: holdStep.progress,
    holdSeconds: holdStep.holdMs / 1000,
    isGuardBlocked,
    shouldProceedToExercise: !prev.confirmed && holdStep.confirmed,
  };
}

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
    const required = Math.max(1, params.requiredStreak || REQUIRED_MATCH_STREAK);
    const clampedStreak = Math.max(0, Math.min(required, params.streak));
    return {
      variant: 'LIVENESS',
      boxColor: 'blue',
      label:
        clampedStreak > 0
          ? 'กำลังยืนยันใบหน้าจริง...'
          : params.isLongRange
            ? 'พบใบหน้าจากระยะไกล — กำลังยืนยันใบหน้าจริง...'
            : 'กำลังยืนยันใบหน้าจริง...',
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
    if (lWr && (lWr.visibility ?? 0) >= 0.35 && lWr.y <= Math.min(0.82, lShY + 0.06)) {
      if (1 - lWr.x < 0.5) leftUp = true;
      else rightUp = true;
    }
    if (rWr && (rWr.visibility ?? 0) >= 0.35 && rWr.y <= Math.min(0.82, rShY + 0.06)) {
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
      startBtnIdle: `Raise BOTH Hands ${totalSec.toFixed(0)}s or Tap to Start Immediately →`,
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
      startBtnIdle: `同时举起双手保持 ${totalSec.toFixed(0)} 秒 或 点击立即开始训练 →`,
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
    startBtnIdle: `ยกมือทั้ง 2 ข้างค้าง ${totalSec.toFixed(0)} วิ หรือ กดเพื่อเริ่มฝึกทันที →`,
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

export interface KioskVerifiedConfirmationCardProps {
  lang: ClinicalLang;
  verifiedModalPayload: KioskLoginSuccessPayload;
  modalCopy: KioskVerifiedModalCopy;
  scanGeneration: number;
  verifiedAnimDone: boolean;
  voiceMuted: boolean;
  isWelcomeSpeechDone: boolean;
  raisedDuringSpeech: boolean;
  twoHandCameraReady: boolean;
  twoHandDetectorReady: boolean;
  twoHandDetected: boolean;
  twoHandProgress: number;
  leftHandUpState: boolean;
  rightHandUpState: boolean;
  ownerGuardBanner: { status: OwnerLockStatus; text: string } | null;
  confirmVideoRef: React.RefObject<HTMLVideoElement | null>;
  confirmCanvasRef: React.RefObject<HTMLCanvasElement | null>;
  confirmProgressFillRef: React.RefObject<HTMLDivElement | null>;
  confirmProgressTextRef: React.RefObject<HTMLSpanElement | null>;
  confirmPillLabelRef: React.RefObject<HTMLSpanElement | null>;
  confirmBtnFillRef: React.RefObject<HTMLSpanElement | null>;
  confirmBtnLabelRef: React.RefObject<HTMLSpanElement | null>;
  onSelectLang: (nextLang: ClinicalLang) => void;
  onToggleVoiceMute: () => void;
  onProceedToExercise: () => void;
  onOpenTelemetrySimulator?: () => void;
  onRejectWrongMatch: () => void;
}

export function KioskVerifiedConfirmationCard({
  lang,
  verifiedModalPayload,
  modalCopy,
  scanGeneration,
  verifiedAnimDone,
  voiceMuted,
  isWelcomeSpeechDone,
  raisedDuringSpeech,
  twoHandCameraReady,
  twoHandDetectorReady,
  twoHandDetected,
  twoHandProgress,
  leftHandUpState,
  rightHandUpState,
  ownerGuardBanner,
  confirmVideoRef,
  confirmCanvasRef,
  confirmProgressFillRef,
  confirmProgressTextRef,
  confirmPillLabelRef,
  confirmBtnFillRef,
  confirmBtnLabelRef,
  onSelectLang,
  onToggleVoiceMute,
  onProceedToExercise,
  onOpenTelemetrySimulator,
  onRejectWrongMatch,
}: KioskVerifiedConfirmationCardProps) {
  const clampedProgress = Math.max(0, Math.min(1, twoHandProgress));
  const totalSec = VERIFIED_HAND_HOLD_MS / 1000;
  const heldSecStr = (clampedProgress * totalSec).toFixed(1);
  const pctInt = Math.round(clampedProgress * 100);
  const progressText = `${heldSecStr} / ${totalSec.toFixed(0)}.0 วิ (${pctInt}%)`;

  return (
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
                  onClick={() => onSelectLang(opt.code)}
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
            onClick={onToggleVoiceMute}
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

        <div className="mb-3 flex items-center justify-center gap-3" data-anim-complete={verifiedAnimDone ? 'true' : 'false'}>
          <DrTechCareMascot
            pose="greeting"
            isSpeaking={!isWelcomeSpeechDone && !voiceMuted}
            size="sm"
          />
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

        {/* Wide 16:9 Upper-Body Camera Preview at 1.0x Zoom for Live 60FPS Face Tracking & Dual-Hand Hold */}
        <div
          data-testid="kiosk-confirm-camera-stage"
          className="mt-4 relative mx-auto aspect-video w-full overflow-hidden rounded-2xl bg-emerald-950 border-2 border-emerald-300 shadow-inner"
        >
          <video
            ref={confirmVideoRef}
            data-testid="kiosk-confirm-video"
            autoPlay
            playsInline
            muted
            className="absolute inset-0 w-full h-full object-cover scale-x-[-1]"
          />
          <canvas
            ref={confirmCanvasRef}
            data-testid="kiosk-confirm-canvas"
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
                  data-testid="kiosk-left-hand-badge"
                  className={`px-2.5 py-1 rounded-full text-[11px] font-black shadow transition-colors duration-150 ${
                    leftHandUpState
                      ? 'bg-emerald-600 text-white'
                      : 'bg-slate-800/80 text-white border border-white/25'
                  }`}
                >
                  {leftHandUpState ? 'ยกมือซ้ายแล้ว' : 'รอมือซ้าย'}
                </span>
                <span
                  data-testid="kiosk-right-hand-badge"
                  className={`px-2.5 py-1 rounded-full text-[11px] font-black shadow transition-colors duration-150 ${
                    rightHandUpState
                      ? 'bg-emerald-600 text-white'
                      : 'bg-slate-800/80 text-white border border-white/25'
                  }`}
                >
                  {rightHandUpState ? 'ยกมือขวาแล้ว' : 'รอมือขวา'}
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

        {/* Dedicated 4-Second Loading Bar (locked at 0% until welcome voice completes + Owner 1:1 Lock passes) */}
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
                    ? `กำลังโหลดยืนยัน 2 มือค้าง ${totalSec.toFixed(0)} วินาที...`
                    : leftHandUpState !== rightHandUpState
                      ? 'ยกมือเดียวไม่นับ กรุณายกขึ้นทั้ง 2 ข้าง'
                      : `หลอดโหลดเริ่มฝึก (ยก 2 ข้างค้าง ${totalSec.toFixed(0)} วิ)`}
            </span>
            <span
              key={isWelcomeSpeechDone ? lang : 'locked'}
              ref={confirmProgressTextRef}
              data-testid="kiosk-hold-progress-text"
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
              {progressText}
            </span>
          </div>
          <div className="h-4 w-full overflow-hidden rounded-full bg-slate-200 border border-slate-300 shadow-inner">
            <div
              ref={confirmProgressFillRef}
              data-testid="kiosk-hold-progress-fill"
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
              style={{ transform: `scaleX(${clampedProgress.toFixed(4)})`, willChange: 'transform' }}
            />
          </div>
        </div>

        <div className="mt-4 flex flex-col gap-2.5">
          <button
            type="button"
            data-testid="kiosk-confirm-start-btn"
            onClick={onProceedToExercise}
            className="btn-tactile relative overflow-hidden w-full py-3.5 px-4 rounded-2xl bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white font-extrabold text-sm shadow-lg cursor-pointer"
          >
            <span
              ref={confirmBtnFillRef}
              className="pointer-events-none absolute inset-0 w-full origin-left bg-white/25"
              style={{ transform: `scaleX(${clampedProgress.toFixed(4)})`, willChange: 'transform' }}
            />
            <span className="relative z-10 flex items-center justify-center gap-2">
              {isWelcomeSpeechDone && clampedProgress > 0 ? (
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
                onProceedToExercise();
                onOpenTelemetrySimulator();
              }}
              className="w-full py-2.5 px-4 rounded-xl bg-sky-50 hover:bg-sky-100 border border-sky-200 text-sky-800 text-xs font-bold cursor-pointer"
            >
              {modalCopy.telemetryBtn}
            </button>
          )}
          <button
            type="button"
            onClick={onRejectWrongMatch}
            className="text-xs text-slate-500 hover:text-slate-700 underline py-1 cursor-pointer"
          >
            {modalCopy.notMeBtn}
          </button>
        </div>
      </div>
    </div>
  );
}

export function KioskBiometricLoginView({
  apiClient,
  kioskId = 'kiosk-01',
  staffModalOpen = false,
  recentlyEnrolledPatient = null,
  onRecentlyEnrolledConsumed,
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
  // Single source of truth for confirmed match progress (0..KIOSK_FAST_MATCH_STREAK) shared by top badge & bottom counter
  const [scanMatchCount, setScanMatchCount] = useState(0);
  const [passiveLivenessVerified, setPassiveLivenessVerified] = useState(false);
  const passiveLivenessVerifiedRef = useRef(false);
  const prevLivenessAnchorsRef = useRef<PassiveLivenessAnchorsPx | null>(null);
  const [enrolledPatients, setEnrolledPatients] = useState<PatientRecord[]>([]);
  const enrolledPatientsRef = useRef<PatientRecord[]>([]);
  enrolledPatientsRef.current = enrolledPatients;
  const forceVerifyFreshDbRef = useRef(false);
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
    let synthStallTimer: number | null = null;
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
      if (synthStallTimer !== null) {
        window.clearTimeout(synthStallTimer);
        synthStallTimer = null;
      }
    };

    const markSpeechDone = () => {
      if (finished || modalSpeakSeqRef.current !== seq) return;
      finished = true;
      clearTimers();
      activeUtteranceRef.current = null;
      // Clear any stalled browser speechSynthesis utterance so `browserSpeaking` cannot stay true
      if ('speechSynthesis' in window) {
        try {
          if (window.speechSynthesis.speaking || window.speechSynthesis.pending) {
            window.speechSynthesis.cancel();
          }
        } catch {
          // Ignore synthesis cancel errors
        }
      }
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
    }, VERIFIED_MODAL_TTS_STALL_TIMEOUT_MS);
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
        markSpeechDone();
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
          if (!cleanSpeechText) {
            markSpeechDone();
            return;
          }
          const utterance = new SpeechSynthesisUtterance(cleanSpeechText);
          utterance.lang = matchedVoice?.lang || langOpt.bcp47;
          if (matchedVoice) utterance.voice = matchedVoice;
          utterance.rate = VERIFIED_MODAL_SPEECH_RATE;
          // 2500ms fallback guard so muted/stalled browser speechSynthesis never blocks the gesture gate
          if (synthStallTimer !== null) {
            window.clearTimeout(synthStallTimer);
          }
          synthStallTimer = window.setTimeout(() => {
            synthStallTimer = null;
            if (!finished && modalSpeakSeqRef.current === seq) {
              markSpeechDone();
            }
          }, VERIFIED_MODAL_TTS_STALL_TIMEOUT_MS);
          utterance.onstart = () => {
            if (synthStallTimer !== null) {
              window.clearTimeout(synthStallTimer);
              synthStallTimer = null;
            }
          };
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
        markSpeechDone();
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
  const activeStreamRef = useRef<MediaStream | null>(null);
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
  const isInferringRef = useRef<boolean>(false);
  const streakGraceRef = useRef<FastScanStreakState>(createInitialFastScanStreakState());
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
    // Keep camera MediaStream live across the Success confirmation card:
    // NEVER pause `videoRef.current` here (pausing froze post-match face & hand detection!).
    // Immediately bind the live stream onto `confirmVideoRef.current` and start playback.
    if (verifiedModalPayload) {
      const cv = confirmVideoRef.current;
      const liveStream = activeStreamRef.current ?? (videoRef.current?.srcObject as MediaStream | null);
      if (cv && liveStream) {
        if (cv.srcObject !== liveStream) {
          cv.srcObject = liveStream;
        }
        void cv.play().catch(() => {});
      }
      return;
    }
    const v = videoRef.current;
    if (v && v.srcObject && !cameraPaused && v.paused) {
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
  const proceedToExerciseSession = useCallback((payload: KioskLoginSuccessPayload) => {
    modalSpeakSeqRef.current += 1;
    cancelAllKioskVoiceSpeech(modalAudioRef.current);
    activeUtteranceRef.current = null;
    excludedPatientIdsRef.current = [];
    onPatientAuthenticatedRef.current(payload);
  }, []);
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
   * When `warmVerifyCache` is true, asks the edge Worker to warm its server-side template RAM cache.
   */
  const refreshBiometricLibraryStats = useCallback(async (force = false, warmVerifyCache = false) => {
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
      const stats: BiometricLibraryStatsResponse = await apiClient.getBiometricLibraryStats(
        force,
        warmVerifyCache,
      );
      const localVerifiedCount = enrolledPatientsRef.current.filter(
        (p) => p.faceBiometricStatus === 'VERIFIED',
      ).length;
      const effectiveValidCount = Math.max(stats.compatibleTemplatesCount, localVerifiedCount);
      const effectiveTotalCount = Math.max(stats.totalActivePatients, enrolledPatientsRef.current.length, effectiveValidCount);
      candidatesLoadedRef.current = true;
      accountCountsRef.current = {
        valid: effectiveValidCount,
        total: effectiveTotalCount,
        legacy: stats.versionMismatchCount,
        filtered: stats.filteredOutCount,
      };
      setDbAccountCount(effectiveValidCount);
      setTotalAccountCount(effectiveTotalCount);
      setLegacyAccountCount(stats.versionMismatchCount);
      setSupabaseSyncedFromCloud(stats.dbSynced);
      setSupabaseSyncError(null);
      setLiveDebugHud((prev) => ({
        validAccounts: effectiveValidCount,
        totalAccounts: effectiveTotalCount,
        legacyAccounts: stats.versionMismatchCount,
        syncedFromSupabase: stats.dbSynced,
        hasEvaluatedMatch: prev?.hasEvaluatedMatch,
        gatePassed: prev?.gatePassed,
        topName: prev?.topName ?? null,
        topPct: prev?.topPct ?? 0,
        centeredPct: prev?.centeredPct,
        streak: prev?.streak ?? 0,
        requiredStreak: KIOSK_FAST_MATCH_STREAK,
        reason: prev?.reason ?? (effectiveValidCount === 0 ? 'NO_TEMPLATES_IN_LIBRARY' : 'READY'),
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
    void refreshBiometricLibraryStats(true, true);
    if (typeof window === 'undefined') return;
    const onFocus = () => {
      void refreshBiometricLibraryStats(true, true);
    };
    window.addEventListener('focus', onFocus);
    const pollTimer = window.setInterval(() => {
      if (!document.hidden) {
        void refreshBiometricLibraryStats(false, true);
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
      void refreshBiometricLibraryStats(true, true);
    });
  }, [apiClient, refreshBiometricLibraryStats]);

  const handleRestartScan = useCallback(() => {
    verifyingLockRef.current = false;
    isInferringRef.current = false;
    streakGraceRef.current = createInitialFastScanStreakState();
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
    setScanMatchCount(0);
    passiveLivenessVerifiedRef.current = false;
    setPassiveLivenessVerified(false);
    prevLivenessAnchorsRef.current = null;
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
    void refreshBiometricLibraryStats(true, true);
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
      setScanMatchCount(0);
      passiveLivenessVerifiedRef.current = false;
      setPassiveLivenessVerified(false);
      prevLivenessAnchorsRef.current = null;
    } else if (!staffModalOpen && wasOpen) {
      excludedPatientIdsRef.current = [];
      handleRestartScan();
    }
  }, [staffModalOpen, handleRestartScan]);

  // Immediate in-memory injection & Supabase sync right after a new patient finishes registration
  const onRecentlyEnrolledConsumedRef = useRef(onRecentlyEnrolledConsumed);
  onRecentlyEnrolledConsumedRef.current = onRecentlyEnrolledConsumed;
  useEffect(() => {
    if (!recentlyEnrolledPatient) return;
    const nextList = mergeEnrolledPatientIntoKioskList(
      enrolledPatientsRef.current,
      recentlyEnrolledPatient,
    );
    enrolledPatientsRef.current = nextList;
    setEnrolledPatients(nextList);

    const localVerifiedCount = nextList.filter((p) => p.faceBiometricStatus === 'VERIFIED').length;
    const prevValid = accountCountsRef.current.valid;
    const nextValid = Math.max(prevValid + 1, localVerifiedCount);
    const nextTotal = Math.max(accountCountsRef.current.total + 1, nextList.length, nextValid);
    accountCountsRef.current = {
      ...accountCountsRef.current,
      valid: nextValid,
      total: nextTotal,
    };
    setDbAccountCount(nextValid);
    setTotalAccountCount(nextTotal);
    setLiveDebugHud((prev) => ({
      validAccounts: nextValid,
      totalAccounts: nextTotal,
      legacyAccounts: prev?.legacyAccounts ?? accountCountsRef.current.legacy,
      syncedFromSupabase: true,
      hasEvaluatedMatch: false,
      gatePassed: false,
      topName: null,
      topPct: 0,
      streak: 0,
      requiredStreak: KIOSK_FAST_MATCH_STREAK,
      reason: 'READY',
    }));

    // Reset stale not_found popup, cooldown, and exclusions so the newly registered user can scan immediately
    excludedPatientIdsRef.current = [];
    rescanCooldownUntilRef.current = 0;
    unmatchedClearFramesRef.current = 0;
    scanDecisionStartAtRef.current = 0;
    totalEvaluatedFramesRef.current = 0;
    serverRejectCountRef.current = 0;
    streakGraceRef.current = createInitialFastScanStreakState();
    serverStreakRef.current = { patientId: null, scores: [] };
    probeWindowRef.current = [];
    passiveLivenessVerifiedRef.current = false;
    setPassiveLivenessVerified(false);
    prevLivenessAnchorsRef.current = null;
    setScanMatchCount(0);
    setScanStage('scanning');
    forceVerifyFreshDbRef.current = true;
    void refreshBiometricLibraryStats(true, true);
    onRecentlyEnrolledConsumedRef.current?.();
  }, [recentlyEnrolledPatient, refreshBiometricLibraryStats]);

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
    let cachedConfirmOwnerVerified = true;
    let cachedConfirmFaceCount = 0;
    let cachedConfirmFaceBox: OwnerFaceBoxMetrics | null = null;
    let smoothedConfirmFaceBox: OwnerFaceBoxMetrics | null = null;
    let cachedConfirmExtraFaceBoxes: OwnerFaceBoxMetrics[] = [];
    let cachedConfirmHandEval: OwnerBoundHandEvaluation | null = null;
    let lastConfirmFaceDetectAt = 0;
    let lastConfirmFaceVideoTime = -1;
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
      activeStreamRef.current = null;
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
      } else {
        gestureWarmupStarted = false;
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
        void refreshBiometricLibraryStats(false, true);
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
        activeStreamRef.current = stream;
        if (opened.devices.length > 0) {
          setAvailableCameras(opened.devices);
        }
        if (opened.activeDeviceId) {
          setActiveCameraId(opened.activeDeviceId);
          if (!selectedCameraId && typeof window !== 'undefined') {
            try {
              if (!window.localStorage.getItem('drtech_preferred_camera_id')) {
                window.localStorage.setItem('drtech_preferred_camera_id', opened.activeDeviceId);
              }
            } catch {
              // Ignore storage quota/privacy errors
            }
          }
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
        isInferringRef.current = false;
        // Disable unused sub-models during 'scanning' mode (no ROI sweep, hand, or pose models loaded
        // until after face match succeeds) so active face detection + 1024D embedding runs at < 60ms.

        let lastEmbedCheckAt = 0;
        let lastEmbeddedVideoTime = -1;
        let lastInferenceDispatchAt = -Infinity;
        let lastFaceSeenAt = 0;
        let shouldContinueLoop = true;

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
              setScanMatchCount(0);
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
                setScanMatchCount(0);
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
                  params.snapNow - item.ts <= 3500 &&
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
              streakGraceRef.current = createInitialFastScanStreakState();
              serverStreakRef.current = { patientId: null, scores: [] };
              bioVerifiedUnmatchedStartAt = 0;
              unmatchedClearFramesRef.current = 0;
              scanDecisionStartAtRef.current = 0;
              totalEvaluatedFramesRef.current = 0;
              setScanMatchCount(0);
              setLiveMatchHud(null);
              setTurningHintPct(null);
              setScanStage('scanning');
              return;
            }

            // Passive live-webcam liveness (no forced exaggerated blink blocking):
            // any clear, non-spoof 3D face proceeds immediately to strict server cosine verification.
            const effectiveBioVerified =
              !params.snapIsSpoof &&
              (params.snapBioVerified || passiveLivenessVerifiedRef.current || params.snapFaceHeightRatio >= 0.08);
            if (!effectiveBioVerified) {
              return;
            }

            const consensusWindow = probeWindowRef.current.slice(-KIOSK_FAST_MATCH_STREAK);
            if (consensusWindow.length < KIOSK_FAST_MATCH_STREAK) {
              // Keep scanMatchCount synchronized cleanly (0 until verified match, or 1 if already matched frame 1)
              setScanStage('burst');
              return;
            }

            const frameVectors = consensusWindow.map((item) => item.embedding);
            const frameTimestamps = consensusWindow.map((item) => Math.round(item.ts));
            const shouldForceFreshDb = forceVerifyFreshDbRef.current;

            const livenessEvidence = {
              biologicalVerified: effectiveBioVerified,
              blinkCount: params.snapBioBlinkCount,
              spoofRisk: params.snapIsSpoof ? Math.max(0.75, params.snapBioSpoofScore) : Math.min(0.45, params.snapBioSpoofScore),
              nonRigidScore: params.snapBioNonRigidScore,
              rppgScore: params.snapBioRppgScore,
              faceHeightRatio: params.snapFaceHeightRatio,
              yawDeg: effectiveYawDeg,
              pitchDeg: params.snapPitchDeg,
              rollDeg,
              engine: BIOMETRIC_ENGINE_ID,
              frameVectors,
              frameTimestamps,
              forceRefresh: shouldForceFreshDb,
              ...(activeGeometricRatios ? { geometricRatios: activeGeometricRatios } : {}),
              ...(excludedPatientIdsRef.current.length > 0
                ? { excludedPatientIds: excludedPatientIdsRef.current }
                : {}),
            };

            // Live Server-Side Verification (NEVER set scanMatchCount=2 prematurely before a match succeeds!)
            if (!verifyingLockRef.current) {
              updateReadinessHint(null);
              void preloadRehabPosePipeline();
              let verifyRes: KioskFaceVerifyResponse;
              let usedEmbedding = activeEmbedding;
              let usedGeometricRatios = activeGeometricRatios;
              try {
                const verifyStartMs = performance.now();
                verifyRes = await apiClient.verifyFaceLogin(activeEmbedding, livenessEvidence);
                if (shouldForceFreshDb) {
                  forceVerifyFreshDbRef.current = false;
                }
                profilerRef.current.recordServerVerify(performance.now() - verifyStartMs, 0);

                if (!active || !shouldContinueLoop || verifiedModalPayloadRef.current) {
                  return;
                }

                // Fallback for brand-new locally enrolled patient on the same Kiosk before cloud propagation settles
                if (
                  !verifyRes.matched &&
                  enrolledPatientsRef.current.length > 0 &&
                  (verifyRes.reasonCode === 'BELOW_THRESHOLD' ||
                    verifyRes.reasonCode === 'NO_TEMPLATES_IN_LIBRARY')
                ) {
                  const localNewMatch = matchLocalEnrolledPatientsStrict(
                    activeEmbedding,
                    enrolledPatientsRef.current,
                    excludedPatientIdsRef.current,
                  );
                  if (localNewMatch) {
                    verifyRes = {
                      ...verifyRes,
                      matched: true,
                      patientId: localNewMatch.patient.id,
                      patientName: localNewMatch.patient.fullName,
                      confidencePercent: localNewMatch.confidencePercent,
                      consensusFramesVerified: KIOSK_FAST_MATCH_STREAK,
                      reasonCode: 'MATCHED',
                      patient: {
                        id: localNewMatch.patient.id,
                        hn: localNewMatch.patient.hn,
                        fullName: localNewMatch.patient.fullName,
                        nextAppointmentDate: localNewMatch.patient.nextAppointmentDate,
                        strokeType: localNewMatch.patient.strokeType,
                        affectedSide: localNewMatch.patient.affectedSide,
                        motorAssessmentScore: localNewMatch.patient.motorAssessmentScore,
                        adlDressingScore: localNewMatch.patient.adlDressingScore,
                        maxSafeRomDeg: localNewMatch.patient.maxSafeRomDeg,
                        universalHealthLinkCode: localNewMatch.patient.universalHealthLinkCode,
                      },
                      rehabSchedules: localNewMatch.patient.rehabSchedules,
                    };
                  }
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
                      const { frameVectors: _fv, frameTimestamps: _ft, ...singleMirrorEvidence } = livenessEvidence;
                      const mirrorRes = await apiClient.verifyFaceLogin(mirroredExt.embedding, {
                        ...singleMirrorEvidence,
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
                        probeWindowRef.current = [];
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
                setScanMatchCount(0);
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
                setScanMatchCount(0);
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
                setScanMatchCount(0);
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
                setScanMatchCount(0);
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
                const atomicConsensusVerified =
                  typeof verifyRes.consensusFramesVerified === 'number' &&
                  verifyRes.consensusFramesVerified >= KIOSK_FAST_MATCH_STREAK;
                if (atomicConsensusVerified) {
                  streakGraceRef.current = {
                    patientId: matchedPat.id,
                    streak: KIOSK_FAST_MATCH_STREAK,
                    scores: Array.from({ length: KIOSK_FAST_MATCH_STREAK }, () => scoreNorm),
                    lastMatchAtMs: params.snapNow,
                    dropoutStartedAtMs: null,
                  };
                  serverStreakRef.current = {
                    patientId: matchedPat.id,
                    scores: streakGraceRef.current.scores,
                  };
                } else {
                  const stepped = stepFastScanStreakWithGrace(
                    streakGraceRef.current,
                    {
                      type: 'match',
                      patientId: matchedPat.id,
                      score: scoreNorm,
                      nowMs: params.snapNow,
                    },
                    { requiredStreak: KIOSK_FAST_MATCH_STREAK, graceMs: KIOSK_STREAK_GRACE_MS },
                  );
                  streakGraceRef.current = stepped;
                  serverStreakRef.current = {
                    patientId: stepped.patientId,
                    scores: stepped.scores,
                  };
                }

                const nextStreak = Math.min(KIOSK_FAST_MATCH_STREAK, serverStreakRef.current.scores.length);
                const isConfirmed = nextStreak >= KIOSK_FAST_MATCH_STREAK;
                bioVerifiedUnmatchedStartAt = 0;
                unmatchedClearFramesRef.current = 0;
                passiveLivenessVerifiedRef.current = true;
                setPassiveLivenessVerified(true);
                setScanMatchCount(nextStreak);
                setTurningHintPct(null);

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
              const dropoutStep = stepFastScanStreakWithGrace(
                streakGraceRef.current,
                { type: 'dropout', nowMs: params.snapNow },
                { requiredStreak: KIOSK_FAST_MATCH_STREAK, graceMs: KIOSK_STREAK_GRACE_MS },
              );
              streakGraceRef.current = dropoutStep;
              if (dropoutStep.didReset || dropoutStep.streak === 0) {
                serverStreakRef.current = { patientId: null, scores: [] };
                setScanMatchCount(0);
              }
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
                // Slide the rolling window by dropping the oldest frame so any stale pre-exposure or
                // post-blink frame is immediately replaced on the next capture instead of causing an
                // instant single-call false "not_found" verdict for a newly registered patient!
                if (probeWindowRef.current.length > 0) {
                  probeWindowRef.current = probeWindowRef.current.slice(1);
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
                    matchStreak: serverStreakRef.current.scores.length,
                    hasMatchedPatient: Boolean(verifiedModalPayloadRef.current || verifyingLockRef.current),
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
          if (
            !active ||
            !shouldContinueLoop ||
            (!videoRef.current && !confirmVideoRef.current) ||
            !landmarker
          ) {
            return;
          }
          try {
            const verifiedPayload = verifiedModalPayloadRef.current;
            if (verifiedPayload) {
              // Dedicated 'post_match_gesture_gate' 60FPS mode:
              // Keeps MediaStream live on `confirmVideoRef`, tracks owner face smoothly on every
              // frame (never freezing), and runs dual-detector (Hands + Pose Wrists) 4.0s hold gate.
              if (!gestureWarmupStarted && !handLandmarker && !poseLandmarker) {
                gestureWarmupStarted = true;
                if (gestureWarmupTimerId !== null) {
                  window.clearTimeout(gestureWarmupTimerId);
                  gestureWarmupTimerId = null;
                }
                if (active) {
                  void warmupGestureDetectors().catch(() => {
                    gestureWarmupStarted = false;
                  });
                }
              }
              if (lastConfirmedPatientId !== verifiedPayload.patient.id) {
                lastConfirmedPatientId = verifiedPayload.patient.id;
                confirmFaceCycle = 0;
                // Patient just passed strict 3-frame server biometric verification; start locked green
                // while continuous 1:1 owner verification monitors every few frames for impersonators.
                cachedConfirmOwnerVerified = true;
                cachedConfirmFaceCount = 0;
                cachedConfirmFaceBox = null;
                smoothedConfirmFaceBox = null;
                cachedConfirmExtraFaceBoxes = [];
                cachedConfirmHandEval = null;
                lastConfirmFaceDetectAt = 0;
                lastConfirmFaceVideoTime = -1;
                lastGestureVideoTime = -1;
                lastOwnerGuardKey = '';
                gestureHoldMs = 0;
                lastAnyHandRaisedAt = -Infinity;
                latestAnyHandUp = false;
                gestureTriggered = false;
              }
              if (confirmVideoRef.current && stream) {
                if (confirmVideoRef.current.srcObject !== stream) {
                  confirmVideoRef.current.srcObject = stream;
                  void confirmVideoRef.current.play().catch(() => {});
                } else if (confirmVideoRef.current.paused) {
                  void confirmVideoRef.current.play().catch(() => {});
                }
                if (!twoHandCameraReadySent) {
                  twoHandCameraReadySent = true;
                  setTwoHandCameraReady(true);
                }
              }

              const now = performance.now();
              const dt = lastGestureFrameAt > 0 ? Math.min(250, Math.max(1, now - lastGestureFrameAt)) : 16.67;
              lastGestureFrameAt = now;

              // CRITICAL: Always read frames from the live playing `confirmVideoRef` (falling back
              // to `videoRef` only during the initial mount frame) so detection NEVER sees a paused frame.
              const activeVideo = resolvePostMatchActiveVideoElement(
                confirmVideoRef.current,
                videoRef.current,
              );

              const browserSpeaking =
                typeof window !== 'undefined' &&
                'speechSynthesis' in window &&
                Boolean(window.speechSynthesis.speaking);
              const speechBlocked = isVerifiedModalSpeechBlocked({
                speechDone: isSpeechDoneRef.current,
                browserSpeaking,
                voiceMuted: voiceMutedRef.current,
              });

              const cCanvas = confirmCanvasRef.current;
              const cw = cCanvas?.clientWidth || 320;
              const ch = cCanvas?.clientHeight || 180;

              // 1. Live 60FPS Face Tracking (runs on EVERY fresh frame, even during welcome speech)
              if (
                activeVideo &&
                activeVideo.readyState >= 2 &&
                activeVideo.videoWidth > 0 &&
                landmarker
              ) {
                if (
                  now - lastConfirmFaceDetectAt >= 32 &&
                  (activeVideo.currentTime !== lastConfirmFaceVideoTime ||
                    now - lastConfirmFaceDetectAt >= 55)
                ) {
                  lastConfirmFaceVideoTime = activeVideo.currentTime;
                  lastConfirmFaceDetectAt = now;
                  confirmFaceCycle += 1;
                  try {
                    const faceTs = getNextMonotonicVideoDetectTs(now);
                    const fRes = landmarker.detectForVideo(activeVideo, faceTs);
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
                        activeVideo.videoWidth,
                        activeVideo.videoHeight,
                        cw,
                        ch,
                      );
                      cachedConfirmExtraFaceBoxes = rawFaces
                        .filter((lm) => lm !== primaryLm)
                        .map((lm) =>
                          computeOwnerFaceBoxMetrics(
                            lm,
                            activeVideo.videoWidth,
                            activeVideo.videoHeight,
                            cw,
                            ch,
                          ),
                        )
                        .filter((b): b is OwnerFaceBoxMetrics => b !== null);

                      if (effCount > 1) {
                        cachedConfirmOwnerVerified = false;
                      } else if (
                        !verifiedPayload.biometricTemplate ||
                        verifiedPayload.biometricTemplate.masterEmbedding.length === 0
                      ) {
                        cachedConfirmOwnerVerified = true;
                      } else if (
                        !confirmEmbedInFlight &&
                        (confirmFaceCycle === 1 ||
                          confirmFaceCycle % 5 === 0 ||
                          !cachedConfirmOwnerVerified)
                      ) {
                        if (!confirmSnapCanvas) {
                          confirmSnapCanvas = document.createElement('canvas');
                        }
                        const confirmRawBox = computeBoundingBoxFromLandmarks(primaryLm);
                        const snapCanvas = captureCanonicalFaceSnapshot(
                          activeVideo,
                          confirmRawBox,
                          confirmSnapCanvas,
                          primaryLm,
                        );
                        if (snapCanvas) {
                          const snapLm = primaryLm.map((p) => ({ x: p.x, y: p.y, z: p.z }));
                          const snapEffCount = effCount;
                          const snapW = activeVideo.videoWidth || 1280;
                          const snapH = activeVideo.videoHeight || 720;
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
              }

              smoothedConfirmFaceBox = smoothOwnerFaceBoxMetrics(
                smoothedConfirmFaceBox,
                cachedConfirmFaceBox,
                dt,
              );

              const drawLiveConfirmOverlay = (activeHandEval: OwnerBoundHandEvaluation | null) => {
                if (!cCanvas) return;
                if (cCanvas.width !== cw || cCanvas.height !== ch) {
                  cCanvas.width = cw;
                  cCanvas.height = ch;
                }
                const cCtx = cCanvas.getContext('2d');
                if (!cCtx) return;
                cCtx.clearRect(0, 0, cw, ch);
                drawOwnerBiometricOverlayOnCanvas(cCtx, cw, ch, {
                  ownerName: verifiedPayload.patient.fullName,
                  faceBox: smoothedConfirmFaceBox,
                  extraFaceBoxes: cachedConfirmExtraFaceBoxes,
                  ownerFaceVerified: cachedConfirmOwnerVerified,
                  faceCount: cachedConfirmFaceCount,
                  handEval: activeHandEval,
                });
              };

              if (speechBlocked) {
                // Speech-First Relay Lock: hold 4.0s gesture timer at 0% until welcome voice
                // completes (or 2500ms stall timeout fires), while STILL rendering the 60FPS live face box!
                gestureHoldMs = 0;
                lastAnyHandRaisedAt = -Infinity;
                latestAnyHandUp = false;
                cachedConfirmHandEval = null;
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
                drawLiveConfirmOverlay(null);

                // Lightly sense early hand raises (~8Hz) purely to remind user to wait for speech
                if (
                  activeVideo &&
                  activeVideo.readyState >= 2 &&
                  activeVideo.videoWidth > 0 &&
                  (handLandmarker || poseLandmarker || Boolean(edgeTelemetryRef.current?.hand_status)) &&
                  now - lastGestureInferenceAt >= 120
                ) {
                  lastGestureInferenceAt = now;
                  const edgeHands = edgeTelemetryRef.current?.hand_status;
                  let detectedEarlyHands: NormalizedLandmark[][] = [];
                  let detectedEarlyPose: NormalizedLandmark[] | null = null;
                  if (handLandmarker || poseLandmarker) {
                    try {
                      if (!gestureOffscreenCanvas) {
                        gestureOffscreenCanvas = document.createElement('canvas');
                      }
                      const offW = 480;
                      const offH = Math.max(
                        270,
                        Math.round(offW * ((activeVideo.videoHeight || 720) / (activeVideo.videoWidth || 1280))),
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
                        offCtx.drawImage(activeVideo, 0, 0, offW, offH);
                      }
                      const earlySource = offCtx ? gestureOffscreenCanvas : activeVideo;
                      if (handLandmarker) {
                        const handRes = handLandmarker.detectForVideo(
                          earlySource,
                          getNextMonotonicHandDetectTs(now),
                        );
                        detectedEarlyHands = handRes.landmarks ?? [];
                      }
                      if (poseLandmarker && detectedEarlyHands.length < 2) {
                        const poseRes = poseLandmarker.detectForVideo(
                          earlySource,
                          getNextMonotonicPoseDetectTs(now),
                        );
                        detectedEarlyPose = poseRes.landmarks?.[0] ?? null;
                      }
                    } catch {
                      // Ignore transient MediaPipe frame errors while locked for speech
                    }
                  }
                  const earlyStrict = evaluateStrictTwoHandsForStartModal(
                    detectedEarlyHands,
                    detectedEarlyPose,
                  );
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
                activeVideo &&
                activeVideo.readyState >= 2 &&
                activeVideo.videoWidth > 0 &&
                (landmarker || handLandmarker || poseLandmarker)
              ) {
                let latestLeftUp = false;
                let latestRightUp = false;

                // 2. Dual-Detector Hand + Pose Wrist inference (~25Hz max, >=40ms apart) on live activeVideo
                if (
                  now - lastGestureInferenceAt >= 40 &&
                  (activeVideo.currentTime !== lastGestureVideoTime ||
                    now - lastGestureInferenceAt >= 65)
                ) {
                  lastGestureVideoTime = activeVideo.currentTime;
                  lastGestureInferenceAt = now;

                  if (!gestureOffscreenCanvas) {
                    gestureOffscreenCanvas = document.createElement('canvas');
                  }
                  const offW = 480;
                  const offH = Math.max(
                    270,
                    Math.round(offW * ((activeVideo.videoHeight || 720) / (activeVideo.videoWidth || 1280))),
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
                    offCtx.drawImage(activeVideo, 0, 0, offW, offH);
                  }
                  const inferenceSource = offCtx ? gestureOffscreenCanvas : activeVideo;

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
                  // Run PoseLandmarker whenever HandLandmarker is unavailable OR detects fewer than 2 hands
                  // (0 or 1 hand) so raised upper-body wrists always provide reliable dual-detector coverage!
                  if (
                    poseLandmarker &&
                    (!handLandmarker ||
                      (detectedHands.length < 2 && now - lastPoseFallbackAt >= 90))
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

                  const handEval = evaluateOwnerBoundKinematicHands({
                    ownerFaceVerified: cachedConfirmOwnerVerified,
                    ownerFaceBox: cachedConfirmFaceBox,
                    faceCount: cachedConfirmFaceCount,
                    ownerName: verifiedPayload.patient.fullName,
                    hands: detectedHands,
                    pose: detectedPose,
                    poseCount: detectedPoseCount,
                    videoW: activeVideo.videoWidth,
                    videoH: activeVideo.videoHeight,
                    canvasW: cw,
                    canvasH: ch,
                  });
                  cachedConfirmHandEval = handEval;

                  const isGuardBlocked =
                    handEval.status === 'IMPOSTOR_FACE' ||
                    handEval.status === 'NO_OWNER_FACE' ||
                    handEval.status === 'MULTI_PERSON' ||
                    handEval.status === 'INTRUDER_HAND';

                  if (isGuardBlocked) {
                    latestLeftUp = false;
                    latestRightUp = false;
                    latestAnyHandUp = false;
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
                }

                // 3. Draw 60FPS Live Face Box + Biometric Tether HUD on EVERY animation frame
                drawLiveConfirmOverlay(cachedConfirmHandEval);

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

                // Zero-reflow 60-120Hz GPU compositor updates (scaleX + synchronized text labels)
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
                const totalSecStr = (VERIFIED_HAND_HOLD_MS / 1000).toFixed(1);
                if (confirmProgressTextRef.current) {
                  confirmProgressTextRef.current.textContent = `${heldSecStr} / ${totalSecStr} วิ (${pctInt}%)`;
                }
                if (holdStep.detected && confirmPillLabelRef.current) {
                  confirmPillLabelRef.current.textContent =
                    lang === 'en'
                      ? `Both hands detected! Hold ${heldSecStr}/${totalSecStr}s (${pctInt}%)`
                      : lang === 'cn'
                        ? `已检测到双手举起！保持 ${heldSecStr}/${totalSecStr} 秒 (${pctInt}%)`
                        : `ยกครบ 2 ข้าง! ค้างไว้ ${heldSecStr}/${totalSecStr} วิ (${pctInt}%)`;
                }
                if (nextProgress > 0 && confirmBtnLabelRef.current) {
                  confirmBtnLabelRef.current.textContent =
                    lang === 'en'
                      ? `Holding BOTH hands... ${heldSecStr}/${totalSecStr}s (${remainSecStr}s left)`
                      : lang === 'cn'
                        ? `正在加载确认... ${heldSecStr}/${totalSecStr} 秒（剩余 ${remainSecStr} 秒）`
                        : `กำลังโหลด 2 ข้าง... ${heldSecStr}/${totalSecStr} วิ (เหลือ ${remainSecStr} วิ)`;
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
                  proceedToExerciseSession(verifiedPayload);
                  return;
                }
              } else {
                drawLiveConfirmOverlay(cachedConfirmHandEval);
              }
              return;
            }

            if (!videoRef.current) return;
            if (videoRef.current.readyState >= 2 && videoRef.current.videoWidth > 0 && videoRef.current.videoHeight > 0) {
              const now = performance.now();
              profilerRef.current.recordFirstVideoFrame(now);
              if (firstFrameWallMs === 0) {
                firstFrameWallMs = now;
              }
              const visualDtMs = lastVisualRafAt > 0 ? Math.min(100, Math.max(4, now - lastVisualRafAt)) : 16.67;
              lastVisualRafAt = now;

              // Lightweight 60FPS GPU-composited targeting frame renderer (ZERO heavy 478-pt contour or iris reticles!)
              const drawSmoothBiometricOverlay = (frameNow: number, dtMs: number) => {
                if (!targetVisualBox || lastVisualCw <= 0 || lastVisualCh <= 0) {
                  return;
                }
                if (frameNow - lastOverlayDrawAt < 16) {
                  return;
                }
                lastOverlayDrawAt = frameNow;
                const drawStartMs = performance.now();
                smoothedVisualBox = smoothBiometricBox(smoothedVisualBox, targetVisualBox, dtMs);

                const b = smoothedVisualBox;
                const cw = lastVisualCw;
                const ch = lastVisualCh;

                const clampedLeft = Math.max(4, Math.min(cw - 24, b.left));
                const clampedTop = Math.max(4, Math.min(ch - 24, b.top));
                const clampedW = Math.max(24, Math.min(cw - clampedLeft - 4, b.width));
                const clampedH = Math.max(24, Math.min(ch - clampedTop - 4, b.height));

                // GPU-only translate3d positioning for floating status pill below face box (ZERO DOM reflow)
                if (hudBoxElRef.current && hudBoxMounted) {
                  const pillAnchorX = Math.max(90, Math.min(cw - 90, clampedLeft + clampedW * 0.5));
                  const pillAnchorY = Math.min(ch - 68, clampedTop + clampedH + 10);
                  hudBoxElRef.current.style.transform = `translate3d(${pillAnchorX.toFixed(1)}px, ${pillAnchorY.toFixed(1)}px, 0)`;
                }

                const mCanvas = faceMeshCanvasRef.current;
                if (!mCanvas) return;
                const dpr = typeof window !== 'undefined' ? Math.min(1.5, window.devicePixelRatio || 1) : 1;
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
                mCtx.moveTo(clampedLeft, clampedTop + cornerLen);
                mCtx.lineTo(clampedLeft, clampedTop + boxRadius * 0.55);
                mCtx.quadraticCurveTo(clampedLeft, clampedTop, clampedLeft + boxRadius * 0.55, clampedTop);
                mCtx.lineTo(clampedLeft + cornerLen, clampedTop);
                mCtx.moveTo(clampedLeft + clampedW - cornerLen, clampedTop);
                mCtx.lineTo(clampedLeft + clampedW - boxRadius * 0.55, clampedTop);
                mCtx.quadraticCurveTo(clampedLeft + clampedW, clampedTop, clampedLeft + clampedW, clampedTop + boxRadius * 0.55);
                mCtx.lineTo(clampedLeft + clampedW, clampedTop + cornerLen);
                mCtx.moveTo(clampedLeft, clampedTop + clampedH - cornerLen);
                mCtx.lineTo(clampedLeft, clampedTop + clampedH - boxRadius * 0.55);
                mCtx.quadraticCurveTo(clampedLeft, clampedTop + clampedH, clampedLeft + boxRadius * 0.55, clampedTop + clampedH);
                mCtx.lineTo(clampedLeft + cornerLen, clampedTop + clampedH);
                mCtx.moveTo(clampedLeft + clampedW - cornerLen, clampedTop + clampedH);
                mCtx.lineTo(clampedLeft + clampedW - boxRadius * 0.55, clampedTop + clampedH);
                mCtx.quadraticCurveTo(clampedLeft + clampedW, clampedTop + clampedH, clampedLeft + clampedW, clampedTop + clampedH - boxRadius * 0.55);
                mCtx.lineTo(clampedLeft + clampedW, clampedTop + clampedH - cornerLen);
                mCtx.stroke();

                // Smooth 60FPS scanline sweep beam
                mCtx.save();
                drawRoundedRectPath(clampedLeft + 2, clampedTop + 2, clampedW - 4, clampedH - 4, Math.max(4, boxRadius - 2));
                mCtx.clip();
                const sweepPhase = (frameNow % 1800) / 1800;
                const sweepNorm = 0.5 - 0.5 * Math.cos(sweepPhase * Math.PI * 2);
                const scanLineY = clampedTop + 8 + sweepNorm * Math.max(16, clampedH - 16);
                const scanBeamGrad = mCtx.createLinearGradient(clampedLeft, 0, clampedLeft + clampedW, 0);
                scanBeamGrad.addColorStop(0, 'rgba(34, 211, 238, 0)');
                scanBeamGrad.addColorStop(0.5, accentHex);
                scanBeamGrad.addColorStop(1, 'rgba(34, 211, 238, 0)');
                mCtx.globalAlpha = 0.82;
                mCtx.strokeStyle = scanBeamGrad;
                mCtx.lineWidth = 1.8;
                mCtx.beginPath();
                mCtx.moveTo(clampedLeft + 4, scanLineY);
                mCtx.lineTo(clampedLeft + clampedW - 4, scanLineY);
                mCtx.stroke();
                mCtx.restore();
                mCtx.restore();
                mCtx.restore();
                profilerRef.current.recordOverlayDraw(frameNow, performance.now() - drawStartMs);
              };

              // Always render the lightweight 60FPS tracking overlay without waiting for AI inference
              drawSmoothBiometricOverlay(now, visualDtMs);

              // Strict Async Inference Mutex + 190ms Throttle (~5 inferences/sec):
              // Never run synchronous full-resolution detection every RAF frame and never allow overlapping calls!
              const currentVideoTime = videoRef.current.currentTime;
              const isStaleVideoFrame = hasVideoFrameCallbackApi
                ? !hasFreshVideoFrame && now - lastProcessedWallMs < KIOSK_INFERENCE_INTERVAL_MS
                : currentVideoTime === lastProcessedVideoTime && now - lastProcessedWallMs < KIOSK_INFERENCE_INTERVAL_MS;
              if (
                isInferringRef.current ||
                asyncMatchInFlight ||
                verifyingLockRef.current ||
                isStaleVideoFrame ||
                now - lastInferenceDispatchAt < KIOSK_INFERENCE_INTERVAL_MS
              ) {
                return;
              }

              hasFreshVideoFrame = false;
              lastProcessedVideoTime = currentVideoTime;
              lastProcessedWallMs = now;
              lastInferenceDispatchAt = now;
              isInferringRef.current = true;

              const snapVideoEl = videoRef.current;
              const rawVw = snapVideoEl.videoWidth || 1280;
              const rawVh = snapVideoEl.videoHeight || 720;

              // Downscale to compact 320x240 (maxWidth: 320) offscreen canvas so WebGL inference drops
              // from ~700ms down to < 60ms while the visible <video> stays crisp at full resolution!
              const downScale = Math.min(
                1,
                KIOSK_INFERENCE_CANVAS_MAX_WIDTH / Math.max(1, rawVw),
                240 / Math.max(1, rawVh),
              );
              const detectInputW = Math.max(160, Math.round(rawVw * downScale));
              const detectInputH = Math.max(120, Math.round(rawVh * downScale));
              if (!detectCanvasRef.current) {
                detectCanvasRef.current = document.createElement('canvas');
              }
              const dCanvas = detectCanvasRef.current;
              if (dCanvas.width !== detectInputW || dCanvas.height !== detectInputH) {
                dCanvas.width = detectInputW;
                dCanvas.height = detectInputH;
              }
              const dCtx = dCanvas.getContext('2d', { alpha: false });
              const detectInput: HTMLVideoElement | HTMLCanvasElement = dCtx ? dCanvas : snapVideoEl;
              if (dCtx) {
                dCtx.drawImage(snapVideoEl, 0, 0, detectInputW, detectInputH);
              }

              // Yield to non-blocking async inference step so RAF returns immediately at 60FPS
              void Promise.resolve()
                .then(async () => {
                  if (!active || !shouldContinueLoop || verifiedModalPayloadRef.current || !landmarker) {
                    return;
                  }
                  const detectTs = getNextMonotonicVideoDetectTs(now);
                  lastDetectTs = detectTs;
                  const detectStartMs = performance.now();
                  const res = landmarker.detectForVideo(detectInput, detectTs);
                  const detectMs = performance.now() - detectStartMs;

                  const selection = selectPrimaryFace(res.faceLandmarks ?? []);
                  const activeBlendshapes = res.faceBlendshapes?.[0]?.categories;
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
                    // Honor 500ms grace buffer before resetting streak on brief dropout/flicker
                    const dropoutRes = stepFastScanStreakWithGrace(
                      streakGraceRef.current,
                      { type: 'dropout', nowMs: now },
                      { requiredStreak: KIOSK_FAST_MATCH_STREAK, graceMs: KIOSK_STREAK_GRACE_MS },
                    );
                    streakGraceRef.current = dropoutRes;
                    if (dropoutRes.didReset) {
                      serverStreakRef.current = { patientId: null, scores: [] };
                      setScanMatchCount(0);
                      setLiveMatchHud(null);
                    }
                    if (lastFaceSeenAt > 0 && now - lastFaceSeenAt >= 1800 && !verifyingLockRef.current) {
                      ambiguousSeenInAttemptRef.current = false;
                      qualityPassedFramesRef.current = 0;
                      profilerRef.current.resetFaceAttempt();
                      setTurningHintPct(null);
                      bioLivenessRef.current.reset();
                      passiveLivenessVerifiedRef.current = false;
                      setPassiveLivenessVerified(false);
                      prevLivenessAnchorsRef.current = null;
                      probeWindowRef.current = [];
                      bioVerifiedUnmatchedStartAt = 0;
                      unmatchedClearFramesRef.current = 0;
                      scanDecisionStartAtRef.current = 0;
                      totalEvaluatedFramesRef.current = 0;
                      serverRejectCountRef.current = 0;
                      isLastVerifyAmbiguous = false;
                      mirrorProbedForCurrentFace = false;
                      setBioReport(null);
                      setScanStage('scanning');
                    }
                    return;
                  }

                  lastFaceSeenAt = now;
                  profilerRef.current.markFaceSeen(now);
                  noFaceSinceRef.current = null;
                  setShowNoFaceGuidance(false);
                  if (Date.now() - lastCandidatesSyncAtRef.current >= 10_000) {
                    void refreshBiometricLibraryStats(false, true);
                  }

                  const rawBoxForBio = computeBoundingBoxFromLandmarks(landmarks);
                  if (!exposureSettled) {
                    const elapsedSinceFirstFrame = Math.max(0, now - firstFrameWallMs);
                    if (elapsedSinceFirstFrame >= 120 || elapsedSinceFirstFrame >= KIOSK_MAX_EXPOSURE_SETTLE_MS) {
                      exposureSettled = true;
                      profilerRef.current.recordExposureSettled(
                        Math.min(KIOSK_MAX_EXPOSURE_SETTLE_MS, elapsedSinceFirstFrame),
                      );
                    }
                  }

                  const livenessStartMs = performance.now();
                  const bio = bioLivenessRef.current.update({
                    landmarks3D: landmarks,
                    blendshapes: activeBlendshapes,
                    earLeft: liveness.earLeft,
                    earRight: liveness.earRight,
                    yawDeg: liveness.yawDeg,
                    pitchDeg: liveness.pitchDeg,
                    frameWidth: rawVw,
                    frameHeight: rawVh,
                    faceCrop: null,
                    contextFrame: null,
                    faceBox: rawBoxForBio,
                    timestampMs: now,
                  });
                  const livenessMs = performance.now() - livenessStartMs;

                  profilerRef.current.recordDetectFrame({
                    now,
                    detectMs,
                    roiMs: 0,
                    livenessMs,
                    inputWidth: detectInputW,
                    inputHeight: detectInputH,
                  });
                  prevLandmarks = landmarks;

                  const isFrameSpoof = Boolean(bio.spoofDetected && !bio.biologicalVerified);
                  const noseLm = landmarks[1];
                  const jawLm = landmarks[152];
                  const nextLivenessAnchors: PassiveLivenessAnchorsPx | null = rawBoxForBio
                    ? {
                        boxCenterX: (rawBoxForBio.x + rawBoxForBio.width * 0.5) * rawVw,
                        boxCenterY: (rawBoxForBio.y + rawBoxForBio.height * 0.5) * rawVh,
                        ...(noseLm ? { noseX: noseLm.x * rawVw, noseY: noseLm.y * rawVh } : {}),
                        ...(jawLm ? { jawX: jawLm.x * rawVw, jawY: jawLm.y * rawVh } : {}),
                        yawDeg: liveness.yawDeg,
                        pitchDeg: liveness.pitchDeg,
                      }
                    : null;
                  const passiveLiveEval = evaluatePassiveWebcamLiveness({
                    prevAnchors: prevLivenessAnchorsRef.current,
                    nextAnchors: nextLivenessAnchors,
                    bioBiologicalVerified: bio.biologicalVerified,
                    blinkCount: bio.blinkCount,
                    isSpoof: isFrameSpoof,
                    alreadyLiveVerified: passiveLivenessVerifiedRef.current,
                  });
                  prevLivenessAnchorsRef.current = nextLivenessAnchors;
                  if (passiveLiveEval.liveVerified !== passiveLivenessVerifiedRef.current) {
                    passiveLivenessVerifiedRef.current = passiveLiveEval.liveVerified;
                    setPassiveLivenessVerified(passiveLiveEval.liveVerified);
                  }
                  const bioVerifiedNow = !isFrameSpoof && (passiveLiveEval.liveVerified || !bio.spoofDetected);
                  if (bioVerifiedNow) {
                    profilerRef.current.markLivenessVerified(now, 'head_pose');
                  }

                  const bioModeNow = bio.livenessMode ?? '';
                  const bioModeChanged =
                    bioVerifiedNow !== lastBioVerifiedUi ||
                    isFrameSpoof !== lastSpoofUi ||
                    bioModeNow !== lastBioModeUi;
                  if (bioModeChanged || (showCalibHudRef.current && now - lastTelemetryUiSyncAt >= 180)) {
                    lastTelemetryUiSyncAt = now;
                    lastBioVerifiedUi = bioVerifiedNow;
                    lastSpoofUi = isFrameSpoof;
                    lastBioModeUi = bioModeNow;
                    setLivenessInfo(liveness);
                    setBioReport(bio);
                  }
                  if (showCalibHudRef.current && now - lastPerfHudSyncAt >= 250) {
                    lastPerfHudSyncAt = now;
                    const perfSnap = profilerRef.current.getSnapshot(now);
                    setStaffPerfSnapshot(perfSnap);
                    if (typeof window !== 'undefined') {
                      (window as unknown as Record<string, unknown>).__drtechBiometricPerf = perfSnap;
                    }
                  }

                  // Update 60FPS tracking box target immediately
                  if (rawBoxForBio && videoRef.current) {
                    const vw = videoRef.current.videoWidth || rawVw;
                    const vh = videoRef.current.videoHeight || rawVh;
                    const cw = videoRef.current.clientWidth || vw;
                    const ch = videoRef.current.clientHeight || vh;
                    const scale = Math.max(cw / vw, ch / vh);
                    const renderedW = vw * scale;
                    const renderedH = vh * scale;
                    const offsetX = (renderedW - cw) / 2;
                    const offsetY = (renderedH - ch) / 2;

                    const leftNorm = mirroredRef.current ? 1 - rawBoxForBio.x - rawBoxForBio.width : rawBoxForBio.x;
                    const pxLeft = leftNorm * renderedW - offsetX;
                    const pxTop = rawBoxForBio.y * renderedH - offsetY;
                    const pxW = rawBoxForBio.width * renderedW;
                    const pxH = rawBoxForBio.height * renderedH;

                    lastVisualRenderedW = renderedW;
                    lastVisualRenderedH = renderedH;
                    lastVisualOffsetX = offsetX;
                    lastVisualOffsetY = offsetY;
                    lastVisualCw = cw;
                    lastVisualCh = ch;
                    lastVisualOverlayColor = isFrameSpoof
                      ? 'rose'
                      : (liveOverlayRef.current?.boxColor ?? 'emerald');
                    targetVisualBox = { left: pxLeft, top: pxTop, width: pxW, height: pxH };

                    if (!hudBoxMounted) {
                      const xPct = Math.max(0.5, Math.min(94, (pxLeft / cw) * 100));
                      const yPct = Math.max(0.5, Math.min(94, (pxTop / ch) * 100));
                      const wPct = Math.max(4, Math.min(99 - xPct, (pxW / cw) * 100));
                      const hPct = Math.max(4, Math.min(99 - yPct, (pxH / ch) * 100));
                      hudBoxMounted = true;
                      setHudBox({ xPct, yPct, wPct, hPct });
                    }
                  }

                  // Forgiving fast-scan face readiness (NO iris-lock or micro-stillness reset!):
                  // Accept natural slight head tilts (<= 28 deg yaw, <= 24 deg pitch) & clear non-spoof face.
                  const faceBoxWidth = rawBoxForBio?.width ?? 0;
                  const faceBoxHeight = rawBoxForBio?.height ?? 0;
                  const faceReadyForFastScan =
                    !isFrameSpoof &&
                    faceBoxWidth >= 0.06 &&
                    faceBoxHeight >= 0.08 &&
                    Math.abs(liveness.yawDeg) <= 28 &&
                    Math.abs(liveness.pitchDeg) <= 24;

                  if (!faceReadyForFastScan) {
                    unreadyFrameStreak += 1;
                    const dropRes = stepFastScanStreakWithGrace(
                      streakGraceRef.current,
                      { type: 'dropout', nowMs: now, forceReset: isFrameSpoof },
                      { requiredStreak: KIOSK_FAST_MATCH_STREAK, graceMs: KIOSK_STREAK_GRACE_MS },
                    );
                    streakGraceRef.current = dropRes;
                    if (isFrameSpoof || dropRes.didReset) {
                      if (isFrameSpoof) {
                        probeWindowRef.current = [];
                      }
                      serverStreakRef.current = { patientId: null, scores: [] };
                      setScanMatchCount(0);
                      setLiveMatchHud(null);
                    }
                    return;
                  }

                  unreadyFrameStreak = 0;
                  profilerRef.current.markFaceReady(now);
                  updateReadinessHint(null);

                  const isDistinctDecodedFrame =
                    currentVideoTime !== lastEmbeddedVideoTime || now - lastEmbedCheckAt >= KIOSK_INFERENCE_INTERVAL_MS;
                  if (
                    !verifyingLockRef.current &&
                    exposureSettled &&
                    isDistinctDecodedFrame &&
                    videoRef.current
                  ) {
                    lastEmbeddedVideoTime = currentVideoTime;
                    if (!matchSnapCanvasRef.current) {
                      matchSnapCanvasRef.current = document.createElement('canvas');
                    }
                    const snapBundle = captureCanonicalFaceSnapshotWithMetrics(
                      videoRef.current,
                      rawBoxForBio,
                      matchSnapCanvasRef.current,
                      landmarks,
                    );
                    if (snapBundle) {
                      lastEmbedCheckAt = now;
                      const snapLandmarks = landmarks.map((p) => ({ x: p.x, y: p.y, z: p.z }));
                      await runAsyncBiometricMatchStep({
                        snapCanvas: snapBundle.canvas,
                        snapSharpness: snapBundle.sharpness,
                        snapPatch384: snapBundle.patch384,
                        snapRawBox: rawBoxForBio ?? null,
                        snapLandmarks,
                        // Skip heavy iris sub-model extraction during fast scanning to eliminate lag & jitter resets
                        snapOcularIris: null,
                        snapIrisMs: 0,
                        snapVideoW: rawVw,
                        snapVideoH: rawVh,
                        snapYawDeg: liveness.yawDeg,
                        snapPitchDeg: liveness.pitchDeg,
                        snapFaceHeightRatio: faceBoxHeight || bio.faceScale,
                        snapBioVerified: bioVerifiedNow,
                        snapBioNonRigidScore: bio.nonRigidScore,
                        snapBioRppgScore: bio.rppgScore,
                        snapBioSpoofScore: bio.spoofScore,
                        snapBioBlinkCount: bio.blinkCount,
                        snapBioIrisLocked: false,
                        snapIsSpoof: isFrameSpoof,
                        snapNow: now,
                      });
                    }
                  }
                })
                .catch(() => {
                  // Keep 60FPS loop resilient across transient inference errors
                })
                .finally(() => {
                  isInferringRef.current = false;
                });
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
      const patients: PatientRecord[] = normalizePatientRecordList(await apiClient.listPatients(), {
        requireRealVectorsForVerified: false,
      });
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

  const isSpoof =
    Boolean(bioReport?.spoofDetected && !bioReport?.biologicalVerified) ||
    inspectionInfo?.scan_status === 'SPOOF_DETECTED';
  const isBioVerified =
    !isSpoof &&
    (Boolean(bioReport?.biologicalVerified) ||
      passiveLivenessVerified ||
      scanMatchCount > 0 ||
      scanStage === 'verifying' ||
      Boolean(verifiedModalPayload));
  const isFaceDetected = Boolean(hudBox);
  const unifiedScanProgress = resolveKioskUnifiedScanProgress({
    scanMatchCount,
    requiredStreak: KIOSK_FAST_MATCH_STREAK,
    isMatched: Boolean(verifiedModalPayload || scanStage === 'verifying' || liveMatchHud?.confirmed),
    readinessHint,
  });

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
                      <div>Burst: {unifiedScanProgress.clampedCount}/{unifiedScanProgress.required}</div>
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
                        {unifiedScanProgress.topLabel}
                      </div>
                      <div className="mt-1 flex flex-wrap items-center justify-center gap-1.5 text-[10px] font-bold text-[#0284C7] whitespace-nowrap">
                        <span className="inline-flex items-center gap-1 rounded-full px-2 py-0.5 border border-emerald-200 bg-emerald-50 text-emerald-800">
                          <span className="h-1.5 w-1.5 rounded-full bg-emerald-500 animate-pulse" />
                          1024D FACE AI • FAST 2-FRAME
                        </span>
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
                            transform: `scaleX(${(unifiedScanProgress.progressPct / 100).toFixed(3)})`,
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
                      กรุณามองกล้องในระยะพอดี
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
                          streak: unifiedScanProgress.clampedCount,
                          requiredStreak: unifiedScanProgress.required,
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
                data-testid="kiosk-supabase-sync-label"
                className={`font-bold ${
                  supabaseSyncError && !supabaseSyncedFromCloud ? 'text-amber-300' : 'text-emerald-300'
                }`}
              >
                {liveDebugHud?.syncedFromSupabase || supabaseSyncedFromCloud
                  ? formatKioskSupabaseSyncBadgeLabel(
                      liveDebugHud?.validAccounts ?? dbAccountCount,
                      liveDebugHud?.totalAccounts ?? totalAccountCount,
                    )
                  : `คลังใบหน้า: ${liveDebugHud?.validAccounts ?? dbAccountCount} บัญชี`}
              </span>
              {(liveDebugHud?.legacyAccounts ?? legacyAccountCount) > 0 && (
                <span className="text-amber-300">
                  (บัญชีเก่ารออัปเดต {liveDebugHud?.legacyAccounts ?? legacyAccountCount})
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
                  ({unifiedScanProgress.clampedCount}/{unifiedScanProgress.required})
                </>
              ) : ambiguousCloseHint ? (
                <span className="text-amber-200">
                  พบใบหน้าใกล้เคียงหลายคน — กรุณามองตรงเข้าหากล้องและขยับเข้าใกล้ขึ้นเล็กน้อย
                </span>
              ) : (liveDebugHud?.validAccounts ?? dbAccountCount) === 0 ? (
                <span className="text-amber-200">
                  ยังไม่มีข้อมูลใบหน้าในคลัง (กรุณาลงทะเบียนคนไข้ใหม่ก่อนใช้งาน)
                </span>
              ) : unifiedScanProgress.clampedCount > 0 ? (
                <span className="text-blue-200" data-testid="kiosk-bottom-scan-counter">
                  {unifiedScanProgress.counterLabel}
                </span>
              ) : liveDebugHud?.hasEvaluatedMatch ? (
                <span className="text-rose-300 font-bold">
                  ไม่พบบัญชีที่ตรงกับใบหน้านี้ (กรุณาลงทะเบียนคนไข้ใหม่ก่อนใช้งาน)
                </span>
              ) : (
                <span className="text-slate-300" data-testid="kiosk-bottom-scan-counter">
                  {unifiedScanProgress.counterLabel}
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
        <KioskVerifiedConfirmationCard
          lang={lang}
          verifiedModalPayload={verifiedModalPayload}
          modalCopy={modalCopy}
          scanGeneration={scanGeneration}
          verifiedAnimDone={verifiedAnimDone}
          voiceMuted={voiceMuted}
          isWelcomeSpeechDone={isWelcomeSpeechDone}
          raisedDuringSpeech={raisedDuringSpeech}
          twoHandCameraReady={twoHandCameraReady}
          twoHandDetectorReady={twoHandDetectorReady}
          twoHandDetected={twoHandDetected}
          twoHandProgress={twoHandProgress}
          leftHandUpState={leftHandUpState}
          rightHandUpState={rightHandUpState}
          ownerGuardBanner={ownerGuardBanner}
          confirmVideoRef={confirmVideoRef}
          confirmCanvasRef={confirmCanvasRef}
          confirmProgressFillRef={confirmProgressFillRef}
          confirmProgressTextRef={confirmProgressTextRef}
          confirmPillLabelRef={confirmPillLabelRef}
          confirmBtnFillRef={confirmBtnFillRef}
          confirmBtnLabelRef={confirmBtnLabelRef}
          onSelectLang={handleSelectLang}
          onToggleVoiceMute={handleToggleVoiceMute}
          onProceedToExercise={() => proceedToExerciseSession(verifiedModalPayload)}
          onOpenTelemetrySimulator={onOpenTelemetrySimulator}
          onRejectWrongMatch={handleRejectWrongMatchAndRescan}
        />
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

