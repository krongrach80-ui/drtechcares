import React, { useCallback, useEffect, useRef, useState } from 'react';
import type { HandLandmarker, NormalizedLandmark } from '@mediapipe/tasks-vision';
import {
  getCachedHandLandmarker,
  getNextMonotonicHandDetectTs,
  type VisionDelegate,
} from '../../../src/lib/biometricVisionCache';
import { useClinicalI18n } from '@drtechcares/clinical-i18n';
import {
  AudioChimeSynthesizer,
  cancelAllKioskVoiceSpeech,
  readStoredKioskVoiceMuted,
  subscribeKioskVoiceLang,
  subscribeKioskVoiceMuted,
  writeStoredKioskVoiceLang,
  writeStoredKioskVoiceMuted,
} from '@drtechcares/kiosk-input';
import { selectClinicalCameraDeviceId } from '@drtechcares/face-client';
import {
  AlertTriangle,
  Award,
  CheckCircle2,
  Clock,
  Hand,
  RotateCcw,
  Sparkles,
  Volume2,
  VolumeX,
  XCircle,
} from 'lucide-react';
import { sampleTenCognitiveCards, type CognitiveCardQuestion } from './PhysicalCardGameData';
import { getCardArtworkInnerSvg } from './PhysicalCardGameIllustrations';

/**
 * Resolves giant senior-accessible Tailwind typography classes for minigame questions
 * across TH / EN / CN so elderly patients can read every prompt immediately at 1-4m.
 */
export function resolveMinigameQuestionSizeClass(questionText: string | undefined): string {
  const qLen = (questionText ?? '').trim().length;
  if (qLen > 45) {
    return 'text-3xl sm:text-4xl md:text-5xl leading-snug';
  }
  if (qLen > 25) {
    return 'text-4xl sm:text-5xl md:text-[3.5rem] leading-tight';
  }
  return 'text-4xl sm:text-5xl md:text-6xl leading-tight';
}

export function formatCognitiveCardBadge(cardId: string | undefined, fallbackIndex: number): string {
  const digits = String(cardId ?? '').replace(/\D+/g, '');
  const num = digits ? Number.parseInt(digits, 10) : fallbackIndex + 1;
  return `#${String(Number.isFinite(num) && num > 0 ? num : fallbackIndex + 1).padStart(3, '0')}`;
}

export type CognitiveCardVisualKind = 'svg_art' | 'image_url' | 'emoji_icon';

export function resolveCognitiveCardVisualKind(
  card: Pick<CognitiveCardQuestion, 'artKey' | 'imageUrl' | 'icon'> & { svgMarkup?: string },
): CognitiveCardVisualKind {
  if (card.svgMarkup || (card.artKey && (!card.imageUrl || card.imageUrl.startsWith('data:image/svg+xml')))) {
    return 'svg_art';
  }
  if (card.imageUrl) {
    return 'image_url';
  }
  return 'emoji_icon';
}

export type MinigameVoiceLang = 'th' | 'en' | 'zh';

export interface MinigameVoiceOption {
  code: MinigameVoiceLang;
  shortLabel: string;
  compactLabel: string;
  badgeLabel: string;
  speechLang: string;
  statusReady: string;
  statusSpeaking: string;
  statusMuted: string;
  replayLabel: string;
  muteLabel: string;
  unmuteLabel: string;
}

export const MINIGAME_VOICE_LANG_OPTIONS: ReadonlyArray<MinigameVoiceOption> = [
  {
    code: 'th',
    shortLabel: '1. ไทย',
    compactLabel: '1.ไทย',
    badgeLabel: 'THAI AI VOICE COACH',
    speechLang: 'th-TH',
    statusReady: 'พร้อมอ่านคำถาม',
    statusSpeaking: 'กำลังพูด',
    statusMuted: 'ปิดเสียง',
    replayLabel: 'พูดซ้ำ',
    muteLabel: 'เสียงปกติ',
    unmuteLabel: 'ปิดเสียง',
  },
  {
    code: 'en',
    shortLabel: '2. อังกฤษ (EN)',
    compactLabel: '2.EN',
    badgeLabel: 'ENGLISH AI VOICE COACH',
    speechLang: 'en-US',
    statusReady: 'READY',
    statusSpeaking: 'SPEAKING',
    statusMuted: 'MUTED',
    replayLabel: 'พูดซ้ำ',
    muteLabel: 'เสียงปกติ',
    unmuteLabel: 'ปิดเสียง',
  },
  {
    code: 'zh',
    shortLabel: '3. จีน (中文)',
    compactLabel: '3.中文',
    badgeLabel: 'CHINESE AI VOICE COACH',
    speechLang: 'zh-CN',
    statusReady: '准备就绪',
    statusSpeaking: '正在播报',
    statusMuted: '已静音',
    replayLabel: 'พูดซ้ำ',
    muteLabel: 'เสียงปกติ',
    unmuteLabel: 'ปิดเสียง',
  },
];

const VOICE_LANG_STORAGE_KEY = 'drtechcares_voice_coach_lang_v1';

export function normalizeClinicalLangToMinigameVoice(clinicalLang?: string | null): MinigameVoiceLang {
  const lower = (clinicalLang ?? 'th').trim().toLowerCase();
  if (lower.startsWith('zh') || lower === 'cn' || lower.startsWith('cmn')) return 'zh';
  if (lower.startsWith('en')) return 'en';
  return 'th';
}

function readStoredMinigameVoiceLang(fallbackClinicalLang?: string): MinigameVoiceLang {
  if (typeof window !== 'undefined') {
    try {
      const raw = window.localStorage.getItem(VOICE_LANG_STORAGE_KEY);
      if (raw === 'th' || raw === 'en' || raw === 'zh') {
        return raw;
      }
    } catch {
      // Ignore storage restriction
    }
  }
  return normalizeClinicalLangToMinigameVoice(fallbackClinicalLang);
}

export function pickCardTextByVoiceLang(
  textMap: { th: string; en: string; cn: string },
  voiceLang: MinigameVoiceLang,
): string {
  if (voiceLang === 'en') return textMap.en || textMap.th;
  if (voiceLang === 'zh') return textMap.cn || textMap.th;
  return textMap.th || textMap.en;
}

export function formatMinigameAnswerSpeech(
  card: CognitiveCardQuestion,
  isCorrect: boolean,
  voiceLang: MinigameVoiceLang,
): string {
  const explanation = pickCardTextByVoiceLang(card.explanation, voiceLang).trim();
  if (voiceLang === 'en') {
    const prefix = isCorrect ? 'Correct!' : 'Not quite.';
    return explanation ? `${prefix} ${explanation}` : prefix;
  }
  if (voiceLang === 'zh') {
    const prefix = isCorrect ? '回答正确！' : '不太对哦。';
    return explanation ? `${prefix} ${explanation}` : prefix;
  }
  const prefix = isCorrect ? 'เก่งมาก ถูกต้องครับ!' : 'ยังไม่ถูกนะครับ!';
  return explanation ? `${prefix} ${explanation}` : prefix;
}

export function formatMinigameSummarySpeech(
  summary: PhysicalCardGameSummary,
  voiceLang: MinigameVoiceLang,
): string {
  if (voiceLang === 'en') {
    return `Congratulations! You scored ${summary.correct} out of ${summary.total}, with ${summary.accuracyPct} percent accuracy in ${summary.durationSec} seconds.`;
  }
  if (voiceLang === 'zh') {
    return `恭喜完成训练！您在${summary.durationSec}秒内答对了${summary.total}题中的${summary.correct}题，准确率${summary.accuracyPct}%。`;
  }
  return `เก่งมากครับ ทำแบบทดสอบครบแล้ว ได้คะแนน ${summary.correct} เต็ม ${summary.total} คะแนน ความแม่นยำ ${summary.accuracyPct} เปอร์เซ็นต์ ใช้เวลา ${summary.durationSec} วินาที`;
}

export function buildMinigameTtsUrl(text: string, voiceLang: MinigameVoiceLang): string {
  const collapsed = text.replace(/\s+/g, ' ').trim();
  const maxLen = 190;
  let clipped = collapsed;
  if (collapsed.length > maxLen) {
    const sliced = collapsed.slice(0, maxLen);
    const lastSpace = sliced.lastIndexOf(' ');
    clipped =
      lastSpace >= Math.floor(maxLen * 0.5) ? sliced.slice(0, lastSpace).trim() : sliced.trim();
  }
  const params = new URLSearchParams({
    lang: voiceLang,
    text: clipped,
  });
  return `/api/tts?${params.toString()}`;
}

function pickMinigameSpeechVoice(
  voices: readonly SpeechSynthesisVoice[],
  bcp47: string,
): SpeechSynthesisVoice | null {
  if (!voices.length) return null;
  const targetLower = bcp47.toLowerCase();
  const langPrefix = targetLower.split('-')[0];

  const exactMatches = voices.filter((v) => v.lang.toLowerCase() === targetLower);
  const chineseExtraMatches =
    langPrefix === 'zh'
      ? voices.filter((v) => {
          const l = v.lang.toLowerCase();
          return l.startsWith('zh') || l.startsWith('cmn') || v.name.toLowerCase().includes('chinese');
        })
      : [];
  const prefixMatches = voices.filter((v) => v.lang.toLowerCase().startsWith(langPrefix));

  const candidates =
    exactMatches.length > 0
      ? exactMatches
      : chineseExtraMatches.length > 0
        ? chineseExtraMatches
        : prefixMatches;

  if (!candidates.length) return null;
  const preferred = candidates.find((v) => {
    const name = v.name.toLowerCase();
    return name.includes('natural') || name.includes('google') || name.includes('microsoft');
  });
  return preferred ?? candidates[0];
}

const HOLD_DURATION_SEC = 0.8;
const HAND_RELEASE_DECAY_RATE = 2.5;
const HAND_SWITCH_DECAY_RATE = 4.0;
const LANDMARK_SMOOTH_SPEED = 38;
const OFF_CANVAS_WIDTH = 512;
export const MINIGAME_MIN_HAND_CONFIDENCE = 0.55;

const HAND_CONNECTIONS: ReadonlyArray<[number, number]> = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],
  [0, 5],
  [5, 6],
  [6, 7],
  [7, 8],
  [5, 9],
  [9, 10],
  [10, 11],
  [11, 12],
  [9, 13],
  [13, 14],
  [14, 15],
  [15, 16],
  [13, 17],
  [17, 18],
  [18, 19],
  [19, 20],
  [0, 17],
];

type ActiveHandSide = 'LEFT' | 'RIGHT' | null;

export const MINIGAME_HAND_RAISE_THRESHOLD_Y = 0.58;
export const MINIGAME_THAI_SPEECH_RATE = 1.18;
export const MINIGAME_SPEECH_KEEPALIVE_INTERVAL_MS = 3000;
export const MINIGAME_QUESTION_LOCK_PILL_TEXT = 'กรุณาฟังคำถามให้จบก่อนยกมือตอบ...';
export const MINIGAME_QUESTION_READY_PILL_TEXT = 'ฟังจบแล้ว! ยกมือค้าง 0.8 วินาทีเพื่อตอบ';
export type QuestionAnswerPhase = 'READING_QUESTION' | 'READY_TO_ANSWER';

/**
 * Computes a generous safe watchdog duration (`>= 8000ms`, `text.length * 180 + 3000`)
 * so long Thai/EN/CN card questions are NEVER cut off mid-sentence.
 * When this fallback timer expires (e.g. muted/unavailable OS audio), it ONLY unlocks
 * answering and NEVER calls `window.speechSynthesis.cancel()`.
 */
export function computeMinigameSafeSpeechWatchdogMs(text: string): number {
  const cleanLen = (text ?? '').trim().length;
  return Math.max(8000, cleanLen * 180 + 3000);
}

/**
 * Builds a stable turn key (`cardIndex-cardId-lang`) so automatic card-question TTS
 * only triggers when the active card, session, or language genuinely changes — NEVER on
 * 60FPS hand-tracking updates (`leftHandRaised`, `rightHandRaised`, `holdProgress`).
 */
export function buildMinigameSpeechTurnKey(
  sessionKey: number,
  cardIndex: number,
  cardId: string | undefined,
  voiceLang: MinigameVoiceLang,
): string {
  return `${sessionKey}-${cardIndex}-${cardId ?? 'none'}-${voiceLang}`;
}

export function resolveMinigameAnswerStatusPill(params: {
  bothHandsWarn: boolean;
  isQuestionSpeaking: boolean;
}): { text: string; variant: 'warn' | 'speaking' | 'ready' } {
  if (params.isQuestionSpeaking) {
    return {
      text: MINIGAME_QUESTION_LOCK_PILL_TEXT,
      variant: 'speaking',
    };
  }
  if (params.bothHandsWarn) {
    return {
      text: 'ห้ามยก 2 มือพร้อมกัน! กรุณาลดมือลง 1 ข้างเพื่อตอบ',
      variant: 'warn',
    };
  }
  return {
    text: MINIGAME_QUESTION_READY_PILL_TEXT,
    variant: 'ready',
  };
}

export interface MinigameSpeechEnginePort {
  speak: (utterance: SpeechSynthesisUtterance) => void;
  cancel: () => void;
  resume: () => void;
  getVoices: () => SpeechSynthesisVoice[];
  readonly speaking: boolean;
  readonly pending: boolean;
  readonly paused: boolean;
}

export interface MinigameSpeechDriverOptions {
  synth: MinigameSpeechEnginePort | null;
  createUtterance: (text: string) => SpeechSynthesisUtterance;
  setTimer: (fn: () => void, ms: number) => number;
  clearTimer: (id: number) => void;
  setIntervalFn: (fn: () => void, ms: number) => number;
  clearIntervalFn: (id: number) => void;
  onStateChange?: (state: {
    isQuestionSpeaking: boolean;
    phase: QuestionAnswerPhase;
    statusText: string;
    activeUtterance: SpeechSynthesisUtterance | null;
  }) => void;
}

/**
 * Self-contained, deterministic speech lifecycle driver used for unit testing and mirroring
 * `PhysicalCardGame`'s exact non-cancelling watchdog, utterance GC retention, 3000ms Chromium
 * keep-alive, turn-key deduplication, explicit replay (`พูดซ้ำ`), and mute unlock rules.
 */
export class MinigameSpeechTurnDriver {
  public activeUtterance: SpeechSynthesisUtterance | null = null;
  public spokenTurnKey = '';
  public isQuestionSpeaking = false;
  public isMuted = false;
  private seq = 0;
  private watchdogId: number | null = null;
  private keepAliveId: number | null = null;
  private readonly opts: MinigameSpeechDriverOptions;

  constructor(opts: MinigameSpeechDriverOptions) {
    this.opts = opts;
  }

  private clearTimers(): void {
    if (this.watchdogId !== null) {
      this.opts.clearTimer(this.watchdogId);
      this.watchdogId = null;
    }
    if (this.keepAliveId !== null) {
      this.opts.clearIntervalFn(this.keepAliveId);
      this.keepAliveId = null;
    }
  }

  private emitState(): void {
    const phase: QuestionAnswerPhase = this.isQuestionSpeaking
      ? 'READING_QUESTION'
      : 'READY_TO_ANSWER';
    this.opts.onStateChange?.({
      isQuestionSpeaking: this.isQuestionSpeaking,
      phase,
      statusText: this.isQuestionSpeaking
        ? MINIGAME_QUESTION_LOCK_PILL_TEXT
        : MINIGAME_QUESTION_READY_PILL_TEXT,
      activeUtterance: this.activeUtterance,
    });
  }

  private startQuestionSpeech(text: string, voiceLang: MinigameVoiceLang, forceCancelPrev: boolean): boolean {
    const clean = text.trim();
    if (this.isMuted || !clean || !this.opts.synth) {
      this.clearTimers();
      this.activeUtterance = null;
      this.isQuestionSpeaking = false;
      this.emitState();
      return false;
    }

    this.seq += 1;
    const seq = this.seq;
    this.clearTimers();

    if (forceCancelPrev) {
      try {
        this.opts.synth.cancel();
      } catch {
        // Ignore cancel failure
      }
    }

    const utterance = this.opts.createUtterance(clean);
    const targetOpt =
      MINIGAME_VOICE_LANG_OPTIONS.find((item) => item.code === voiceLang) ??
      MINIGAME_VOICE_LANG_OPTIONS[0];
    utterance.lang = targetOpt.speechLang;
    utterance.rate = voiceLang === 'th' ? MINIGAME_THAI_SPEECH_RATE : voiceLang === 'zh' ? 0.98 : 1.0;

    // Persist strong reference until onend/onerror so Chromium/Edge GC cannot drop speech mid-sentence
    this.activeUtterance = utterance;
    this.isQuestionSpeaking = true;
    this.emitState();

    utterance.onstart = () => {
      if (this.seq !== seq) return;
      this.isQuestionSpeaking = true;
      this.emitState();
    };

    utterance.onend = () => {
      if (this.activeUtterance === utterance) {
        this.activeUtterance = null;
      }
      if (this.seq !== seq) return;
      this.clearTimers();
      this.isQuestionSpeaking = false;
      this.emitState();
    };

    utterance.onerror = () => {
      if (this.activeUtterance === utterance) {
        this.activeUtterance = null;
      }
      if (this.seq !== seq) return;
      this.clearTimers();
      this.isQuestionSpeaking = false;
      this.emitState();
    };

    // Chromium long-utterance keep-alive (3000ms): resume only if paused
    this.keepAliveId = this.opts.setIntervalFn(() => {
      if (this.seq !== seq || !this.isQuestionSpeaking || !this.opts.synth) return;
      if (this.opts.synth.paused) {
        try {
          this.opts.synth.resume();
        } catch {
          // Ignore resume failure
        }
      }
    }, MINIGAME_SPEECH_KEEPALIVE_INTERVAL_MS);

    // Safe dynamic watchdog: unlocks answering WITHOUT calling speechSynthesis.cancel()
    const safeDurationMs = computeMinigameSafeSpeechWatchdogMs(clean);
    this.watchdogId = this.opts.setTimer(() => {
      this.watchdogId = null;
      if (this.seq !== seq) return;
      if (this.keepAliveId !== null) {
        this.opts.clearIntervalFn(this.keepAliveId);
        this.keepAliveId = null;
      }
      this.isQuestionSpeaking = false;
      this.emitState();
    }, safeDurationMs);

    this.opts.synth.speak(utterance);
    return true;
  }

  /**
   * Deduplicated automatic card-turn speech trigger: ignores repeated calls with identical
   * `turnKey` (e.g. 60FPS hand-tracking updates) so active speech is never cancelled or restarted.
   */
  speakQuestionForTurn(params: {
    sessionKey: number;
    cardIndex: number;
    cardId: string;
    voiceLang: MinigameVoiceLang;
    text: string;
  }): boolean {
    const turnKey = buildMinigameSpeechTurnKey(
      params.sessionKey,
      params.cardIndex,
      params.cardId,
      params.voiceLang,
    );
    if (this.spokenTurnKey === turnKey) {
      return false;
    }
    this.spokenTurnKey = turnKey;
    return this.startQuestionSpeech(params.text, params.voiceLang, false);
  }

  /** Explicit `"พูดซ้ำ"` replay: cancels prior utterance cleanly and replays full question. */
  replayQuestion(text: string, voiceLang: MinigameVoiceLang): boolean {
    return this.startQuestionSpeech(text, voiceLang, true);
  }

  /** Toggling mute ON cancels immediately and unlocks answering so muted mode never blocks gameplay. */
  setMuted(muted: boolean, currentQuestionText?: string, voiceLang: MinigameVoiceLang = 'th'): void {
    this.isMuted = muted;
    if (muted) {
      this.seq += 1;
      this.clearTimers();
      this.activeUtterance = null;
      try {
        this.opts.synth?.cancel();
      } catch {
        // Ignore cancel failure
      }
      this.isQuestionSpeaking = false;
      this.emitState();
    } else if (currentQuestionText?.trim()) {
      this.startQuestionSpeech(currentQuestionText, voiceLang, true);
    }
  }

  dispose(): void {
    this.seq += 1;
    this.clearTimers();
    this.activeUtterance = null;
  }
}

export interface DetectedCardHandSample {
  zone: 'LEFT' | 'RIGHT';
  screenX: number;
  screenY: number;
  wristY?: number;
  minTipY?: number;
  px: number;
  py: number;
  topPx: number;
  landmarks: NormalizedLandmark[];
}

/**
 * Strictly validates that detected landmarks represent an anatomical 21-point human hand
 * (wrist + MCP knuckles + fingertips with coherent scale) so faces/shoulders/background are rejected.
 */
export function isValidStrict21HandLandmarks(
  landmarks: ReadonlyArray<{ x: number; y: number; z?: number }> | null | undefined,
  confidence?: number,
): boolean {
  if (!landmarks || landmarks.length < 21) return false;
  if (typeof confidence === 'number' && confidence < MINIGAME_MIN_HAND_CONFIDENCE) return false;

  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (let i = 0; i < 21; i += 1) {
    const pt = landmarks[i];
    if (
      !pt ||
      !Number.isFinite(pt.x) ||
      !Number.isFinite(pt.y) ||
      pt.x < -0.05 ||
      pt.x > 1.05 ||
      pt.y < -0.05 ||
      pt.y > 1.05
    ) {
      return false;
    }
    if (pt.x < minX) minX = pt.x;
    if (pt.x > maxX) maxX = pt.x;
    if (pt.y < minY) minY = pt.y;
    if (pt.y > maxY) maxY = pt.y;
  }

  const boxW = maxX - minX;
  const boxH = maxY - minY;
  if (boxW < 0.025 || boxH < 0.03 || boxW > 0.85 || boxH > 0.9) return false;

  // Require real wrist -> middle-MCP span and recognizable finger landmarks
  const wrist = landmarks[0];
  const midMcp = landmarks[9];
  const palmSpan = Math.hypot(midMcp.x - wrist.x, midMcp.y - wrist.y);
  if (palmSpan < 0.018) return false;

  let distinctFingers = 0;
  for (const [mcpIdx, tipIdx] of [
    [5, 8],
    [9, 12],
    [13, 16],
    [17, 20],
  ] as const) {
    const d = Math.hypot(landmarks[tipIdx].x - landmarks[mcpIdx].x, landmarks[tipIdx].y - landmarks[mcpIdx].y);
    if (d >= 0.012) distinctFingers += 1;
  }
  return distinctFingers >= 2;
}

export function evaluateTwoHandCardSelection(
  hands: ReadonlyArray<Pick<DetectedCardHandSample, 'zone' | 'screenY' | 'wristY' | 'minTipY'>>,
  thresholdY: number = MINIGAME_HAND_RAISE_THRESHOLD_Y,
): {
  activeSide: 'LEFT' | 'RIGHT' | null;
  leftRaised: boolean;
  rightRaised: boolean;
  raisedCount: number;
  bothHandsBlocked: boolean;
} {
  let leftRaised = false;
  let rightRaised = false;
  let raisedCount = 0;
  for (const hand of hands) {
    const effectiveY = Math.min(hand.screenY, hand.wristY ?? hand.screenY);
    if (effectiveY < thresholdY) {
      raisedCount += 1;
      if (hand.zone === 'LEFT') leftRaised = true;
      if (hand.zone === 'RIGHT') rightRaised = true;
    }
  }
  if (raisedCount >= 2 || (leftRaised && rightRaised)) {
    return {
      activeSide: null,
      leftRaised: true,
      rightRaised: true,
      raisedCount,
      bothHandsBlocked: true,
    };
  }
  if (leftRaised) {
    return { activeSide: 'LEFT', leftRaised: true, rightRaised: false, raisedCount: 1, bothHandsBlocked: false };
  }
  if (rightRaised) {
    return { activeSide: 'RIGHT', leftRaised: false, rightRaised: true, raisedCount: 1, bothHandsBlocked: false };
  }
  return { activeSide: null, leftRaised: false, rightRaised: false, raisedCount: 0, bothHandsBlocked: false };
}

export interface PhysicalCardGameSummary {
  correct: number;
  total: number;
  category: string;
  durationSec: number;
  accuracyPct: number;
}

interface PhysicalCardGameProps {
  patientName: string;
  onComplete: (summary: PhysicalCardGameSummary) => void;
  onSessionFinished?: (summary: PhysicalCardGameSummary) => Promise<void> | void;
  allowReplay?: boolean;
}

interface CoverTransform {
  cw: number;
  ch: number;
  renderedW: number;
  renderedH: number;
  offsetX: number;
  offsetY: number;
}

function computeMirroredCoverTransform(cw: number, ch: number, vw: number, vh: number): CoverTransform {
  const safeVw = vw > 0 ? vw : 640;
  const safeVh = vh > 0 ? vh : 480;
  const safeCw = cw > 0 ? cw : 640;
  const safeCh = ch > 0 ? ch : 480;
  const scale = Math.max(safeCw / safeVw, safeCh / safeVh);
  const renderedW = safeVw * scale;
  const renderedH = safeVh * scale;
  return {
    cw: safeCw,
    ch: safeCh,
    renderedW,
    renderedH,
    offsetX: (safeCw - renderedW) / 2,
    offsetY: (safeCh - renderedH) / 2,
  };
}

function mapMirroredLandmarkToScreen(
  lm: { x: number; y: number },
  tf: CoverTransform,
): { px: number; py: number; screenX: number; screenY: number } {
  const px = tf.offsetX + (1 - lm.x) * tf.renderedW;
  const py = tf.offsetY + lm.y * tf.renderedH;
  return {
    px,
    py,
    screenX: px / tf.cw,
    screenY: py / tf.ch,
  };
}

function classifyMirroredHandZone(screenX: number): 'LEFT' | 'RIGHT' {
  return screenX < 0.5 ? 'LEFT' : 'RIGHT';
}

function lerpLandmarks(
  prev: NormalizedLandmark[] | null,
  target: NormalizedLandmark[],
  factor: number,
): NormalizedLandmark[] {
  if (!prev || prev.length !== target.length) {
    return target.map((lm) => ({ ...lm }));
  }
  const wristPrev = prev[0];
  const wristNext = target[0];
  if (wristPrev && wristNext) {
    const jump = Math.hypot(wristNext.x - wristPrev.x, wristNext.y - wristPrev.y);
    if (jump > 0.35) {
      return target.map((lm) => ({ ...lm }));
    }
  }
  return target.map((tLm, idx) => {
    const pLm = prev[idx];
    if (!pLm) return { ...tLm };
    return {
      x: pLm.x + (tLm.x - pLm.x) * factor,
      y: pLm.y + (tLm.y - pLm.y) * factor,
      z: pLm.z + (tLm.z - pLm.z) * factor,
      visibility: tLm.visibility,
    };
  });
}

function drawCenterZoneDivider(ctx: CanvasRenderingContext2D, cw: number, ch: number) {
  ctx.save();
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.28)';
  ctx.lineWidth = 3;
  ctx.setLineDash([10, 10]);
  ctx.beginPath();
  ctx.moveTo(cw / 2, 16);
  ctx.lineTo(cw / 2, ch - 16);
  ctx.stroke();

  // Horizontal True Arm Elevation Gate Guide Line ("เส้นเกณฑ์ยกมือตอบ")
  const thresholdY = ch * MINIGAME_HAND_RAISE_THRESHOLD_Y;
  ctx.strokeStyle = 'rgba(56, 189, 248, 0.62)';
  ctx.lineWidth = 2.5;
  ctx.setLineDash([8, 8]);
  ctx.beginPath();
  ctx.moveTo(16, thresholdY);
  ctx.lineTo(cw - 16, thresholdY);
  ctx.stroke();
  ctx.setLineDash([]);

  const label = '▲ เส้นเกณฑ์ยกมือตอบ (ยกสูงกว่าเส้นนี้)';
  ctx.font = '800 13px "Sarabun", sans-serif';
  const tw = ctx.measureText(label).width;
  const pillW = Math.min(cw - 40, tw + 24);
  const pillH = 24;
  const pillX = (cw - pillW) / 2;
  const pillY = Math.max(8, thresholdY - pillH - 6);
  ctx.fillStyle = 'rgba(15, 76, 129, 0.72)';
  ctx.beginPath();
  ctx.roundRect(pillX, pillY, pillW, pillH, 12);
  ctx.fill();
  ctx.fillStyle = 'rgba(255, 255, 255, 0.94)';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(label, cw / 2, pillY + pillH / 2);
  ctx.restore();
}

function drawHandAlertBox(
  ctx: CanvasRenderingContext2D,
  landmarks: NormalizedLandmark[],
  tf: CoverTransform,
  color: string = '#ef4444',
) {
  const pts = landmarks.map((lm) => mapMirroredLandmarkToScreen(lm, tf));
  if (pts.length === 0) return;
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const p of pts) {
    minX = Math.min(minX, p.px);
    maxX = Math.max(maxX, p.px);
    minY = Math.min(minY, p.py);
    maxY = Math.max(maxY, p.py);
  }
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 4;
  ctx.beginPath();
  ctx.roundRect(minX - 18, minY - 18, Math.max(48, maxX - minX + 36), Math.max(48, maxY - minY + 36), 14);
  ctx.stroke();
  ctx.restore();
}

function drawHandPillLabel(
  ctx: CanvasRenderingContext2D,
  px: number,
  topPy: number,
  label: string,
  bgColor: string,
  textColor: string = '#ffffff',
  cssWidthPx?: number,
) {
  ctx.save();
  ctx.font = '800 13px "Sarabun", sans-serif';
  const tw = ctx.measureText(label).width;
  const pillW = tw + 20;
  const pillH = 26;
  const maxW = cssWidthPx && cssWidthPx > 0 ? cssWidthPx : ctx.canvas.width;
  const pillX = Math.max(8, Math.min(maxW - pillW - 8, px - pillW / 2));
  const pillY = Math.max(8, topPy - pillH - 12);
  ctx.fillStyle = bgColor;
  ctx.beginPath();
  ctx.roundRect(pillX, pillY, pillW, pillH, 13);
  ctx.fill();
  ctx.fillStyle = textColor;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(label, pillX + pillW / 2, pillY + pillH / 2);
  ctx.restore();
}

function drawDualHandRedBridge(
  ctx: CanvasRenderingContext2D,
  palmA: { px: number; py: number },
  palmB: { px: number; py: number },
) {
  ctx.save();
  ctx.strokeStyle = '#ef4444';
  ctx.lineWidth = 5;
  ctx.setLineDash([12, 8]);
  ctx.beginPath();
  ctx.moveTo(palmA.px, palmA.py);
  ctx.lineTo(palmB.px, palmB.py);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.restore();
}

function drawHandSkeletonOnCanvas(
  ctx: CanvasRenderingContext2D,
  landmarks: NormalizedLandmark[],
  tf: CoverTransform,
  color: string,
) {
  const pts = landmarks.map((lm) => mapMirroredLandmarkToScreen(lm, tf));
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 6;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';

  for (const [startIdx, endIdx] of HAND_CONNECTIONS) {
    const start = pts[startIdx];
    const end = pts[endIdx];
    if (!start || !end) continue;
    ctx.beginPath();
    ctx.moveTo(start.px, start.py);
    ctx.lineTo(end.px, end.py);
    ctx.stroke();
  }

  for (let i = 0; i < pts.length; i += 1) {
    const pt = pts[i];
    if (!pt) continue;
    ctx.fillStyle = i === 0 || i === 9 ? '#ffffff' : color;
    ctx.beginPath();
    ctx.arc(pt.px, pt.py, i === 0 || i === 9 ? 8.5 : 5.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#0f172a';
    ctx.lineWidth = 2;
    ctx.stroke();
  }
  ctx.restore();
}

function drawDwellHalo(
  ctx: CanvasRenderingContext2D,
  px: number,
  py: number,
  progress: number,
  color: string,
) {
  if (progress <= 0.01) return;
  ctx.save();
  ctx.beginPath();
  ctx.strokeStyle = 'rgba(255,255,255,0.32)';
  ctx.lineWidth = 7;
  ctx.arc(px, py, 54, 0, Math.PI * 2);
  ctx.stroke();

  ctx.beginPath();
  ctx.strokeStyle = color;
  ctx.lineWidth = 9;
  ctx.lineCap = 'round';
  ctx.arc(px, py, 54, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * Math.min(1, progress));
  ctx.stroke();
  ctx.restore();
}

interface MinigameVoiceBarProps {
  voiceLang: MinigameVoiceLang;
  voiceMuted: boolean;
  isSpeaking: boolean;
  onSelectLang: (nextLang: MinigameVoiceLang) => void;
  onReplay: () => void;
  onToggleMute: () => void;
}

function MinigameVoiceBar({
  voiceLang,
  voiceMuted,
  isSpeaking,
  onSelectLang,
  onReplay,
  onToggleMute,
}: MinigameVoiceBarProps) {
  return (
    <div
      data-minigame-voice-control="true"
      className="flex flex-wrap items-center gap-1.5 shrink-0"
    >
      <div
        role="group"
        aria-label="เลือกภาษาเสียงพากย์มินิเกม"
        className="inline-flex items-center gap-0.5 rounded-full border border-sky-200 bg-sky-50/70 p-0.5"
      >
        {MINIGAME_VOICE_LANG_OPTIONS.map((opt) => {
          const active = opt.code === voiceLang;
          return (
            <button
              key={opt.code}
              type="button"
              aria-pressed={active}
              onClick={() => onSelectLang(opt.code)}
              className={`rounded-full px-2 py-0.5 text-xs font-bold transition-all cursor-pointer ${
                active
                  ? 'bg-[#0284C7] text-white shadow-xs'
                  : 'text-[#0F4C81] hover:bg-white'
              }`}
            >
              {opt.compactLabel}
            </button>
          );
        })}
      </div>

      <button
        type="button"
        onClick={onReplay}
        disabled={voiceMuted}
        className="inline-flex shrink-0 items-center gap-1 rounded-full border border-sky-200 bg-white px-2.5 py-1 text-xs font-bold text-[#0F4C81] transition-colors hover:bg-sky-50 active:scale-95 disabled:cursor-not-allowed disabled:opacity-50 cursor-pointer"
      >
        <RotateCcw className="h-3 w-3 text-[#0284C7]" />
        <span>พูดซ้ำ</span>
      </button>

      <button
        type="button"
        onClick={onToggleMute}
        aria-pressed={voiceMuted}
        className={`inline-flex shrink-0 items-center gap-1 rounded-full border px-2.5 py-1 text-xs font-bold transition-all active:scale-95 cursor-pointer ${
          voiceMuted
            ? 'border-amber-300 bg-amber-100 text-amber-950 hover:bg-amber-200'
            : 'border-emerald-300 bg-emerald-100 text-emerald-950 hover:bg-emerald-200'
        }`}
      >
        {voiceMuted ? (
          <VolumeX className="h-3.5 w-3.5 shrink-0 text-amber-700" />
        ) : (
          <Volume2 className={`h-3.5 w-3.5 shrink-0 text-emerald-700 ${isSpeaking ? 'animate-pulse' : ''}`} />
        )}
        <span>{voiceMuted ? 'เสียง AI: ปิด' : 'เสียง AI: เปิด'}</span>
      </button>
    </div>
  );
}

export function PhysicalCardGame({
  patientName,
  onComplete,
  onSessionFinished,
  allowReplay = true,
}: PhysicalCardGameProps) {
  const { lang, setLang, t } = useClinicalI18n();
  const [cards, setCards] = useState<CognitiveCardQuestion[]>(() => sampleTenCognitiveCards());
  const [index, setIndex] = useState(0);
  const [score, setScore] = useState(0);
  const [feedback, setFeedback] = useState<{ correct: boolean; message: string } | null>(null);
  const [pendingChoice, setPendingChoice] = useState<'yes' | 'no' | null>(null);
  const [bothHandsWarn, setBothHandsWarn] = useState(false);
  const [cropWarn, setCropWarn] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [detectorReady, setDetectorReady] = useState(false);
  const [completedSummary, setCompletedSummary] = useState<PhysicalCardGameSummary | null>(null);
  const [savingResult, setSavingResult] = useState(false);
  const [sessionKey, setSessionKey] = useState(0);

  const [voiceLang, setVoiceLang] = useState<MinigameVoiceLang>(() =>
    readStoredMinigameVoiceLang(lang),
  );
  const [voiceMuted, setVoiceMuted] = useState<boolean>(() => readStoredKioskVoiceMuted());
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [questionPhase, setQuestionPhase] = useState<QuestionAnswerPhase>(() =>
    readStoredKioskVoiceMuted() ? 'READY_TO_ANSWER' : 'READING_QUESTION',
  );
  const questionPhaseRef = useRef<QuestionAnswerPhase>(questionPhase);
  questionPhaseRef.current = questionPhase;
  const bothHandsWarnSpeechTsRef = useRef<number>(0);
  const detectedHandsRef = useRef<DetectedCardHandSample[]>([]);
  const smoothedHandsRef = useRef<NormalizedLandmark[][]>([]);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const offscreenCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const progressBarFillRef = useRef<HTMLDivElement | null>(null);
  const progressPctRef = useRef<HTMLSpanElement | null>(null);
  const badgePctRef = useRef<HTMLSpanElement | null>(null);

  const chimeRef = useRef(new AudioChimeSynthesizer());
  const ttsAudioRef = useRef<HTMLAudioElement | null>(null);
  const speakSeqRef = useRef(0);
  const isSpeakingRef = useRef(false);
  const queuedSpeechRef = useRef<{ text: string; lang: MinigameVoiceLang } | null>(null);
  const pendingSpeakTimerRef = useRef<number | null>(null);
  const speakingWatchdogRef = useRef<number | null>(null);
  const speechKeepAliveRef = useRef<number | null>(null);
  const spokenTurnKeyRef = useRef<string>('');
  const onMinigameSpeechEndedRef = useRef<(seq: number) => void>(() => {});
  const webSpeechTimerRef = useRef<number | null>(null);
  const activeUtteranceRef = useRef<SpeechSynthesisUtterance | null>(null);
  const voicesRef = useRef<SpeechSynthesisVoice[]>([]);
  const blockedByAutoplayRef = useRef(false);

  const voiceLangRef = useRef(voiceLang);
  voiceLangRef.current = voiceLang;
  const voiceMutedRef = useRef(voiceMuted);
  voiceMutedRef.current = voiceMuted;
  const activeFeedbackMetaRef = useRef<{ card: CognitiveCardQuestion; correct: boolean } | null>(null);

  const activeHandRef = useRef<ActiveHandSide>(null);
  const lockedHandRef = useRef<ActiveHandSide>(null);
  const holdProgressRef = useRef(0);
  const lastFrameTimeRef = useRef(0);
  const lastVideoTimeRef = useRef(-1);
  const lastInferenceTsRef = useRef(0);
  const targetLandmarksRef = useRef<NormalizedLandmark[] | null>(null);
  const smoothedLandmarksRef = useRef<NormalizedLandmark[] | null>(null);
  const palmPxRef = useRef<{ px: number; py: number } | null>(null);
  const pendingChoiceRef = useRef<'yes' | 'no' | null>(null);
  const cropWarnRef = useRef(false);
  const bothHandsWarnRef = useRef(false);

  const startedAtRef = useRef<number>(Date.now());
  const submittingRef = useRef(false);
  const timerRef = useRef<number | null>(null);

  const feedbackRef = useRef(feedback);
  feedbackRef.current = feedback;
  const completedSummaryRef = useRef(completedSummary);
  completedSummaryRef.current = completedSummary;
  const indexRef = useRef(index);
  indexRef.current = index;
  const scoreRef = useRef(score);
  scoreRef.current = score;
  const cardsRef = useRef(cards);
  cardsRef.current = cards;

  const clearPendingSpeakTimer = useCallback(() => {
    if (pendingSpeakTimerRef.current !== null && typeof window !== 'undefined') {
      window.clearTimeout(pendingSpeakTimerRef.current);
      pendingSpeakTimerRef.current = null;
    }
  }, []);

  const clearSpeakingWatchdog = useCallback(() => {
    if (speakingWatchdogRef.current !== null && typeof window !== 'undefined') {
      window.clearTimeout(speakingWatchdogRef.current);
      speakingWatchdogRef.current = null;
    }
  }, []);

  const clearSpeechKeepAlive = useCallback(() => {
    if (speechKeepAliveRef.current !== null && typeof window !== 'undefined') {
      window.clearInterval(speechKeepAliveRef.current);
      speechKeepAliveRef.current = null;
    }
  }, []);

  const stopMinigameSpeech = useCallback(
    (clearQueue = true) => {
      speakSeqRef.current += 1;
      clearSpeakingWatchdog();
      clearSpeechKeepAlive();
      if (clearQueue) {
        queuedSpeechRef.current = null;
        clearPendingSpeakTimer();
      }
      if (webSpeechTimerRef.current !== null && typeof window !== 'undefined') {
        window.clearTimeout(webSpeechTimerRef.current);
        webSpeechTimerRef.current = null;
      }
      const audio = ttsAudioRef.current;
      if (audio) {
        try {
          audio.onplay = null;
          audio.onended = null;
          audio.onerror = null;
          audio.pause();
          audio.currentTime = 0;
        } catch {
          // Ignore audio reset errors
        }
      }
      activeUtteranceRef.current = null;
      if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
        try {
          window.speechSynthesis.cancel();
        } catch {
          // Ignore speechSynthesis cancel failure
        }
      }
      isSpeakingRef.current = false;
      setIsSpeaking(false);
    },
    [clearPendingSpeakTimer, clearSpeechKeepAlive, clearSpeakingWatchdog],
  );

  const speakWithWebSpeechFallback = useCallback(
    (clean: string, targetLang: MinigameVoiceLang, seq: number) => {
      if (
        voiceMutedRef.current ||
        !clean ||
        typeof window === 'undefined' ||
        !('speechSynthesis' in window) ||
        typeof SpeechSynthesisUtterance === 'undefined'
      ) {
        onMinigameSpeechEndedRef.current(seq);
        return;
      }

      const targetOpt =
        MINIGAME_VOICE_LANG_OPTIONS.find((item) => item.code === targetLang) ??
        MINIGAME_VOICE_LANG_OPTIONS[0];

      try {
        const synth = window.speechSynthesis;
        const wasBusy = synth.speaking || synth.pending;
        if (wasBusy) {
          synth.cancel();
        }

        const startUtterance = () => {
          if (speakSeqRef.current !== seq || voiceMutedRef.current) return;
          if (synth.paused) {
            synth.resume();
          }
          const availableVoices =
            voicesRef.current.length > 0 ? voicesRef.current : synth.getVoices();
          const matchedVoice = pickMinigameSpeechVoice(availableVoices, targetOpt.speechLang);

          const utterance = new SpeechSynthesisUtterance(clean);
          utterance.lang = matchedVoice?.lang || targetOpt.speechLang;
          if (matchedVoice) {
            utterance.voice = matchedVoice;
          }
          utterance.rate = targetLang === 'th' ? MINIGAME_THAI_SPEECH_RATE : targetLang === 'zh' ? 0.98 : 1.0;
          utterance.pitch = 1.0;
          utterance.volume = 1.0;
          // Persist strong reference in ref until onend/onerror so Chromium/Edge GC never drops mid-sentence
          activeUtteranceRef.current = utterance;

          utterance.onstart = () => {
            if (speakSeqRef.current !== seq) return;
            blockedByAutoplayRef.current = false;
            isSpeakingRef.current = true;
            setIsSpeaking(true);
            if (!activeFeedbackMetaRef.current && !completedSummaryRef.current) {
              questionPhaseRef.current = 'READING_QUESTION';
              setQuestionPhase('READING_QUESTION');
            }
          };
          utterance.onend = () => {
            if (activeUtteranceRef.current === utterance) {
              activeUtteranceRef.current = null;
            }
            clearSpeechKeepAlive();
            onMinigameSpeechEndedRef.current(seq);
          };
          utterance.onerror = (event: SpeechSynthesisErrorEvent) => {
            if (activeUtteranceRef.current === utterance) {
              activeUtteranceRef.current = null;
            }
            clearSpeechKeepAlive();
            if (event?.error === 'not-allowed') {
              clearSpeakingWatchdog();
              if (speakSeqRef.current === seq) {
                blockedByAutoplayRef.current = true;
                isSpeakingRef.current = false;
                setIsSpeaking(false);
                questionPhaseRef.current = 'READY_TO_ANSWER';
                setQuestionPhase('READY_TO_ANSWER');
              }
              return;
            }
            onMinigameSpeechEndedRef.current(seq);
          };

          isSpeakingRef.current = true;
          setIsSpeaking(true);
          // Chromium long-utterance keep-alive: resume paused speechSynthesis every 3000ms
          clearSpeechKeepAlive();
          speechKeepAliveRef.current = window.setInterval(() => {
            if (speakSeqRef.current !== seq || !isSpeakingRef.current) {
              clearSpeechKeepAlive();
              return;
            }
            if (synth.paused) {
              try {
                synth.resume();
              } catch {
                // Ignore resume failure
              }
            }
          }, MINIGAME_SPEECH_KEEPALIVE_INTERVAL_MS);

          // Safe dynamic watchdog: unlocks answering WITHOUT calling speechSynthesis.cancel()
          clearSpeakingWatchdog();
          const safeDurationMs = computeMinigameSafeSpeechWatchdogMs(clean);
          speakingWatchdogRef.current = window.setTimeout(() => {
            speakingWatchdogRef.current = null;
            clearSpeechKeepAlive();
            onMinigameSpeechEndedRef.current(seq);
          }, safeDurationMs);
          synth.speak(utterance);
        };

        if (wasBusy) {
          webSpeechTimerRef.current = window.setTimeout(() => {
            webSpeechTimerRef.current = null;
            startUtterance();
          }, 50);
        } else {
          startUtterance();
        }
      } catch {
        clearSpeakingWatchdog();
        clearSpeechKeepAlive();
        isSpeakingRef.current = false;
        setIsSpeaking(false);
        onMinigameSpeechEndedRef.current(seq);
      }
    },
    [clearSpeechKeepAlive, clearSpeakingWatchdog],
  );

  const speakMinigameText = useCallback(
    (
      text: string,
      targetLang: MinigameVoiceLang,
      options?: { forceInterrupt?: boolean },
    ) => {
      const clean = text.trim();
      if (voiceMutedRef.current || !clean || typeof window === 'undefined') {
        return;
      }

      if (!options?.forceInterrupt && isSpeakingRef.current) {
        queuedSpeechRef.current = { text: clean, lang: targetLang };
        return;
      }

      clearPendingSpeakTimer();
      stopMinigameSpeech(Boolean(options?.forceInterrupt));
      const seq = speakSeqRef.current;

      if (typeof Audio !== 'undefined') {
        try {
          let audio = ttsAudioRef.current;
          if (!audio) {
            audio = new Audio();
            audio.preload = 'auto';
            ttsAudioRef.current = audio;
          }
          let fallbackTriggered = false;
          const triggerWebFallbackOnce = () => {
            if (fallbackTriggered || speakSeqRef.current !== seq) return;
            fallbackTriggered = true;
            clearSpeakingWatchdog();
            isSpeakingRef.current = false;
            setIsSpeaking(false);
            speakWithWebSpeechFallback(clean, targetLang, seq);
          };
          audio.playbackRate = targetLang === 'th' ? MINIGAME_THAI_SPEECH_RATE : targetLang === 'zh' ? 0.98 : 1.0;

          audio.onplay = () => {
            if (speakSeqRef.current !== seq) return;
            blockedByAutoplayRef.current = false;
            isSpeakingRef.current = true;
            setIsSpeaking(true);
            if (!activeFeedbackMetaRef.current && !completedSummaryRef.current) {
              questionPhaseRef.current = 'READING_QUESTION';
              setQuestionPhase('READING_QUESTION');
            }
          };
          audio.onended = () => {
            onMinigameSpeechEndedRef.current(seq);
          };
          audio.onerror = triggerWebFallbackOnce;

          audio.src = buildMinigameTtsUrl(clean, targetLang);
          const playPromise = audio.play();
          isSpeakingRef.current = true;
          setIsSpeaking(true);
          clearSpeakingWatchdog();
          // Safe dynamic watchdog (>= 8000ms, text.length * 180 + 3000): NEVER cancels active audio/TTS!
          const safeAudioWatchdogMs = computeMinigameSafeSpeechWatchdogMs(clean);
          speakingWatchdogRef.current = window.setTimeout(() => {
            speakingWatchdogRef.current = null;
            onMinigameSpeechEndedRef.current(seq);
          }, safeAudioWatchdogMs);

          if (playPromise && typeof playPromise.catch === 'function') {
            playPromise.catch((err: unknown) => {
              if (speakSeqRef.current !== seq) return;
              const errName =
                err && typeof err === 'object' && 'name' in err
                  ? String((err as { name: string }).name)
                  : '';
              if (errName === 'NotAllowedError') {
                clearSpeakingWatchdog();
                isSpeakingRef.current = false;
                setIsSpeaking(false);
                blockedByAutoplayRef.current = true;
                questionPhaseRef.current = 'READY_TO_ANSWER';
                setQuestionPhase('READY_TO_ANSWER');
                return;
              }
              if (errName === 'AbortError') {
                return;
              }
              triggerWebFallbackOnce();
            });
          }
          return;
        } catch {
          // Fall through to Web Speech API
        }
      }

      speakWithWebSpeechFallback(clean, targetLang, seq);
    },
    [clearPendingSpeakTimer, clearSpeakingWatchdog, speakWithWebSpeechFallback, stopMinigameSpeech],
  );

  onMinigameSpeechEndedRef.current = (seq: number) => {
    if (speakSeqRef.current !== seq) return;
    clearSpeakingWatchdog();
    clearSpeechKeepAlive();
    isSpeakingRef.current = false;
    setIsSpeaking(false);

    if (!activeFeedbackMetaRef.current && !completedSummaryRef.current) {
      questionPhaseRef.current = 'READY_TO_ANSWER';
      setQuestionPhase('READY_TO_ANSWER');
    }

    if (voiceMutedRef.current) {
      queuedSpeechRef.current = null;
      return;
    }

    const queued = queuedSpeechRef.current;
    queuedSpeechRef.current = null;
    if (queued && queued.text) {
      clearPendingSpeakTimer();
      pendingSpeakTimerRef.current = window.setTimeout(() => {
        pendingSpeakTimerRef.current = null;
        if (!voiceMutedRef.current) {
          speakMinigameText(queued.text, queued.lang);
        }
      }, 200);
    }
  };

  const getActiveVoiceText = useCallback((targetLang: MinigameVoiceLang): string => {
    if (completedSummaryRef.current) {
      return formatMinigameSummarySpeech(completedSummaryRef.current, targetLang);
    }
    if (activeFeedbackMetaRef.current) {
      return formatMinigameAnswerSpeech(
        activeFeedbackMetaRef.current.card,
        activeFeedbackMetaRef.current.correct,
        targetLang,
      );
    }
    const activeCard = cardsRef.current[indexRef.current];
    return activeCard ? pickCardTextByVoiceLang(activeCard.question, targetLang) : '';
  }, []);

  useEffect(() => {
    if (typeof window === 'undefined' || !('speechSynthesis' in window)) return;
    const synth = window.speechSynthesis;
    const syncVoices = () => {
      try {
        const list = synth.getVoices();
        if (list.length > 0) {
          voicesRef.current = list;
        }
      } catch {
        // Ignore unavailable voice list
      }
    };
    syncVoices();
    if ('addEventListener' in synth) {
      synth.addEventListener('voiceschanged', syncVoices);
      return () => synth.removeEventListener('voiceschanged', syncVoices);
    }
    return undefined;
  }, []);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    const unlockAutoplay = (event: Event) => {
      try {
        if ('speechSynthesis' in window && window.speechSynthesis.paused) {
          window.speechSynthesis.resume();
        }
        const target = event.target instanceof Element ? event.target : null;
        if (target?.closest('[data-minigame-voice-control="true"]')) {
          return;
        }
        if (blockedByAutoplayRef.current && !voiceMutedRef.current) {
          blockedByAutoplayRef.current = false;
          const activeText = getActiveVoiceText(voiceLangRef.current);
          if (activeText) {
            if (!activeFeedbackMetaRef.current && !completedSummaryRef.current) {
              questionPhaseRef.current = 'READING_QUESTION';
              setQuestionPhase('READING_QUESTION');
            }
            speakMinigameText(activeText, voiceLangRef.current);
          }
        }
      } catch {
        // Ignore audio unlock errors
      }
    };
    window.addEventListener('pointerdown', unlockAutoplay, { passive: true });
    window.addEventListener('keydown', unlockAutoplay, { passive: true });
    return () => {
      window.removeEventListener('pointerdown', unlockAutoplay);
      window.removeEventListener('keydown', unlockAutoplay);
    };
  }, [getActiveVoiceText, speakMinigameText]);

  const stopCameraTracks = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((tr) => tr.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  }, []);

  const resetTrackingRefs = useCallback(() => {
    activeHandRef.current = null;
    lockedHandRef.current = null;
    holdProgressRef.current = 0;
    targetLandmarksRef.current = null;
    smoothedLandmarksRef.current = null;
    detectedHandsRef.current = [];
    smoothedHandsRef.current = [];
    palmPxRef.current = null;
    pendingChoiceRef.current = null;
    bothHandsWarnRef.current = false;
    cropWarnRef.current = false;
    if (progressBarFillRef.current) progressBarFillRef.current.style.transform = 'scaleX(0)';
    if (progressPctRef.current) progressPctRef.current.textContent = '0%';
    if (badgePctRef.current) badgePctRef.current.textContent = '0%';
  }, []);

  const currentCard = cards[index];

  // Auto-speak active card question when card changes or game resets (Speech-First Relay Gate)
  useEffect(() => {
    if (completedSummary || !currentCard || feedbackRef.current) return;
    const turnKey = buildMinigameSpeechTurnKey(
      sessionKey,
      index,
      currentCard.id,
      voiceLangRef.current,
    );
    // Deduplicate automatic card-turn speech so 60FPS hand updates / re-renders never cancel or restart active TTS
    if (spokenTurnKeyRef.current === turnKey) {
      return;
    }
    spokenTurnKeyRef.current = turnKey;
    resetTrackingRefs();
    setPendingChoice(null);
    setBothHandsWarn(false);

    const text = pickCardTextByVoiceLang(currentCard.question, voiceLangRef.current);
    const hasSpeechEngine =
      typeof window !== 'undefined' &&
      (('speechSynthesis' in window && typeof SpeechSynthesisUtterance !== 'undefined') ||
        typeof Audio !== 'undefined');
    if (!voiceMutedRef.current && text.trim() && hasSpeechEngine) {
      questionPhaseRef.current = 'READING_QUESTION';
      setQuestionPhase('READING_QUESTION');
      speakMinigameText(text, voiceLangRef.current, { forceInterrupt: true });
    } else {
      questionPhaseRef.current = 'READY_TO_ANSWER';
      setQuestionPhase('READY_TO_ANSWER');
    }
  }, [completedSummary, currentCard?.id, index, sessionKey, resetTrackingRefs, speakMinigameText]);

  // Auto-speak final summary when all 10 cards are finished
  useEffect(() => {
    if (!completedSummary) return;
    const summarySpeech = formatMinigameSummarySpeech(completedSummary, voiceLangRef.current);
    if (!voiceMutedRef.current && summarySpeech.trim()) {
      speakMinigameText(summarySpeech, voiceLangRef.current);
    }
  }, [completedSummary, speakMinigameText]);

  useEffect(() => {
    return subscribeKioskVoiceLang((next) => {
      voiceLangRef.current = next;
      setVoiceLang(next);
    });
  }, []);

  const handleSelectVoiceLang = useCallback(
    (nextLang: MinigameVoiceLang) => {
      voiceLangRef.current = nextLang;
      setVoiceLang(nextLang);
      setLang(nextLang === 'zh' ? 'cn' : nextLang);
      writeStoredKioskVoiceLang(nextLang);
      if (voiceMutedRef.current) {
        voiceMutedRef.current = false;
        setVoiceMuted(false);
        writeStoredKioskVoiceMuted(false);
      }

      if (activeFeedbackMetaRef.current) {
        const localizedFeedback = formatMinigameAnswerSpeech(
          activeFeedbackMetaRef.current.card,
          activeFeedbackMetaRef.current.correct,
          nextLang,
        );
        setFeedback({
          correct: activeFeedbackMetaRef.current.correct,
          message: localizedFeedback,
        });
        speakMinigameText(localizedFeedback, nextLang, { forceInterrupt: true });
        return;
      }

      const nextText = getActiveVoiceText(nextLang);
      if (nextText) {
        if (!completedSummaryRef.current) {
          const activeCard = cardsRef.current[indexRef.current];
          spokenTurnKeyRef.current = buildMinigameSpeechTurnKey(
            sessionKey,
            indexRef.current,
            activeCard?.id,
            nextLang,
          );
          questionPhaseRef.current = 'READING_QUESTION';
          setQuestionPhase('READING_QUESTION');
          resetTrackingRefs();
        }
        speakMinigameText(nextText, nextLang, { forceInterrupt: true });
      }
    },
    [getActiveVoiceText, resetTrackingRefs, sessionKey, setLang, speakMinigameText],
  );

  const handleReplayVoice = useCallback(() => {
    const text = getActiveVoiceText(voiceLangRef.current);
    if (text && !voiceMutedRef.current) {
      if (!activeFeedbackMetaRef.current && !completedSummaryRef.current) {
        questionPhaseRef.current = 'READING_QUESTION';
        setQuestionPhase('READING_QUESTION');
        resetTrackingRefs();
      }
      speakMinigameText(text, voiceLangRef.current, { forceInterrupt: true });
    }
  }, [getActiveVoiceText, resetTrackingRefs, speakMinigameText]);

  const handleToggleVoiceMute = useCallback(() => {
    const nextMuted = writeStoredKioskVoiceMuted(!voiceMutedRef.current);
    voiceMutedRef.current = nextMuted;
    setVoiceMuted(nextMuted);
    if (nextMuted) {
      stopMinigameSpeech(true);
      cancelAllKioskVoiceSpeech();
      questionPhaseRef.current = 'READY_TO_ANSWER';
      setQuestionPhase('READY_TO_ANSWER');
      return;
    }
    const text = getActiveVoiceText(voiceLangRef.current);
    if (text) {
      if (!activeFeedbackMetaRef.current && !completedSummaryRef.current) {
        const activeCard = cardsRef.current[indexRef.current];
        spokenTurnKeyRef.current = buildMinigameSpeechTurnKey(
          sessionKey,
          indexRef.current,
          activeCard?.id,
          voiceLangRef.current,
        );
        questionPhaseRef.current = 'READING_QUESTION';
        setQuestionPhase('READING_QUESTION');
        resetTrackingRefs();
      }
      speakMinigameText(text, voiceLangRef.current, { forceInterrupt: true });
    }
  }, [getActiveVoiceText, resetTrackingRefs, sessionKey, speakMinigameText, stopMinigameSpeech]);

  useEffect(() => {
    return subscribeKioskVoiceMuted((muted) => {
      voiceMutedRef.current = muted;
      setVoiceMuted(muted);
      if (muted) {
        stopMinigameSpeech(true);
        cancelAllKioskVoiceSpeech();
        questionPhaseRef.current = 'READY_TO_ANSWER';
        setQuestionPhase('READY_TO_ANSWER');
      }
    });
  }, [stopMinigameSpeech]);

  const submitAnswer = useCallback(
    (choice: 'yes' | 'no') => {
      const card = cardsRef.current[indexRef.current];
      if (
        !card ||
        feedbackRef.current ||
        submittingRef.current ||
        questionPhaseRef.current === 'READING_QUESTION'
      ) {
        return;
      }
      submittingRef.current = true;
      resetTrackingRefs();
      setPendingChoice(null);
      setBothHandsWarn(false);
      setCropWarn(false);

      const isCorrect = choice === card.correctAnswer;
      chimeRef.current.playTone(isCorrect ? 'correct' : 'wrong');
      const nextScore = isCorrect ? scoreRef.current + 1 : scoreRef.current;
      if (isCorrect) setScore(nextScore);

      const activeLang = voiceLangRef.current;
      const msg = formatMinigameAnswerSpeech(card, isCorrect, activeLang);
      activeFeedbackMetaRef.current = { card, correct: isCorrect };
      setFeedback({ correct: isCorrect, message: msg });
      if (!voiceMutedRef.current) {
        speakMinigameText(msg, activeLang);
      }

      if (timerRef.current) window.clearTimeout(timerRef.current);
      const answerStartedAt = Date.now();
      const advanceCard = async () => {
        timerRef.current = null;
        resetTrackingRefs();
        activeFeedbackMetaRef.current = null;
        setFeedback(null);
        setPendingChoice(null);
        submittingRef.current = false;

        if (indexRef.current + 1 >= cardsRef.current.length) {
          stopCameraTracks();
          const durationSec = Math.max(1, Math.round((Date.now() - startedAtRef.current) / 1000));
          const total = cardsRef.current.length;
          const accuracyPct = total > 0 ? Math.round((nextScore / total) * 100) : 0;
          const summary: PhysicalCardGameSummary = {
            correct: nextScore,
            total,
            category: 'mixed-99',
            durationSec,
            accuracyPct,
          };
          setCompletedSummary(summary);
          if (onSessionFinished) {
            setSavingResult(true);
            try {
              await onSessionFinished(summary);
            } catch {
              // ignore background sync failures so summary stays responsive
            } finally {
              setSavingResult(false);
            }
          }
        } else {
          spokenTurnKeyRef.current = '';
          setIndex((i) => i + 1);
        }
      };

      const maxFeedbackWaitMs = computeMinigameSafeSpeechWatchdogMs(msg);
      const checkSpeechDoneThenAdvance = () => {
        if (
          (isSpeakingRef.current ||
            Boolean(queuedSpeechRef.current) ||
            Boolean(pendingSpeakTimerRef.current)) &&
          Date.now() - answerStartedAt < maxFeedbackWaitMs
        ) {
          timerRef.current = window.setTimeout(checkSpeechDoneThenAdvance, 250);
          return;
        }
        timerRef.current = window.setTimeout(advanceCard, 250);
      };

      timerRef.current = window.setTimeout(checkSpeechDoneThenAdvance, 2200);
    },
    [onSessionFinished, resetTrackingRefs, speakMinigameText, stopCameraTracks],
  );

  const submitAnswerRef = useRef(submitAnswer);
  submitAnswerRef.current = submitAnswer;

  useEffect(() => {
    if (completedSummary) return;

    let active = true;
    let rafId = 0;
    let handLandmarker: HandLandmarker | null = null;
    let handDelegate: VisionDelegate = 'CPU';
    let edgeWs: WebSocket | null = null;
    type EdgeHandTelemetry = {
      left_hand_landmarks?: Array<{ x: number; y: number; z?: number }>;
      right_hand_landmarks?: Array<{ x: number; y: number; z?: number }>;
    };
    let edgeHandStatus: EdgeHandTelemetry | null = null;
    let edgeHandTs = 0;
    setCameraError(null);
    setDetectorReady(false);
    resetTrackingRefs();

    // Optional Edge Vision WS accepting ONLY strict 21-point hand landmarks
    try {
      const configuredWs = (import.meta.env.VITE_EDGE_VISION_WS as string | undefined)?.trim();
      const wsUrl = configuredWs || 'ws://localhost:8000/ws/kiosk';
      const skipMixedLoopback =
        typeof window !== 'undefined' &&
        window.location.protocol === 'https:' &&
        !configuredWs &&
        /^ws:\/\/(localhost|127\.0\.0\.1)/i.test(wsUrl);
      if (!skipMixedLoopback) {
        edgeWs = new WebSocket(wsUrl);
        edgeWs.onmessage = (evt) => {
          if (!active) return;
          try {
            const msg = JSON.parse(evt.data) as { hand_status?: EdgeHandTelemetry };
            const hs = msg?.hand_status;
            const hasStrictHand =
              (hs?.left_hand_landmarks && isValidStrict21HandLandmarks(hs.left_hand_landmarks)) ||
              (hs?.right_hand_landmarks && isValidStrict21HandLandmarks(hs.right_hand_landmarks));
            if (hs && hasStrictHand) {
              edgeHandStatus = hs;
              edgeHandTs = Date.now();
              setDetectorReady(true);
            }
          } catch {
            // ignore malformed frames
          }
        };
        edgeWs.onerror = () => {
          edgeWs?.close();
        };
      }
    } catch {
      edgeWs = null;
    }

    async function createPureHandDetector() {
      try {
        const cached = await getCachedHandLandmarker();
        if (!active) return 'none' as const;
        handLandmarker = cached.instance;
        handDelegate = cached.delegate;
        return 'hand' as const;
      } catch {
        return 'none' as const;
      }
    }

    async function init() {
      const handDetectorPromise = createPureHandDetector();
      let stream: MediaStream | null = null;
      try {
        const deviceId = await selectClinicalCameraDeviceId();
        const constraints: MediaStreamConstraints = {
          video: deviceId
            ? { deviceId: { exact: deviceId }, width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 120 } }
            : { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 120 } },
          audio: false,
        };
        try {
          stream = await navigator.mediaDevices.getUserMedia(constraints);
        } catch {
          await new Promise((r) => window.setTimeout(r, 200));
          if (!active) return;
          stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
            audio: false,
          });
        }
      } catch {
        edgeWs?.close();
        edgeWs = null;
        if (active) {
          setCameraError('ไม่สามารถเปิดกล้องได้ กรุณาอนุญาตการใช้กล้อง หรือคลิกปุ่มคำตอบด้านล่างได้เลย');
        }
        return;
      }

      if (!active || !videoRef.current) {
        edgeWs?.close();
        edgeWs = null;
        stream.getTracks().forEach((tr) => tr.stop());
        return;
      }

      streamRef.current = stream;
      videoRef.current.srcObject = stream;
      try {
        await videoRef.current.play();
      } catch {
        // video play settles shortly after metadata loads
      }

      const mode = await handDetectorPromise;
      if (!active) {
        stopCameraTracks();
        return;
      }

      const edgeAlive = () => Boolean(edgeHandStatus && Date.now() - edgeHandTs < 450);
      if (mode === 'none' && !edgeAlive()) {
        edgeWs?.close();
        stopCameraTracks();
        setCameraError('ระบบตรวจจับมือ (21 จุด) ไม่พร้อมใช้งาน กรุณาคลิกเลือกคำตอบด้านล่างแทน');
        return;
      }

      if (mode !== 'none' || edgeAlive()) {
        setDetectorReady(true);
      }
      lastFrameTimeRef.current = performance.now();
      lastVideoTimeRef.current = -1;
      lastInferenceTsRef.current = 0;

      const buildHandSample = (rawLms: NormalizedLandmark[]): DetectedCardHandSample => {
        const wrist = rawLms[0];
        const indexMcp = rawLms[5];
        const middleMcp = rawLms[9];
        const pinkyMcp = rawLms[17];
        const middleTip = rawLms[12];
        const palmRaw = {
          x: (wrist.x + indexMcp.x + middleMcp.x + pinkyMcp.x + middleTip.x) / 5,
          y: (wrist.y + indexMcp.y + middleMcp.y + pinkyMcp.y + middleTip.y) / 5,
        };
        const mappedPalm = mapMirroredLandmarkToScreen(palmRaw, tfFromLast);
        const mappedWrist = mapMirroredLandmarkToScreen(wrist, tfFromLast);
        let minPy = Infinity;
        for (const pt of rawLms) {
          const mapped = mapMirroredLandmarkToScreen(pt, tfFromLast);
          if (mapped.py < minPy) minPy = mapped.py;
        }
        return {
          zone: classifyMirroredHandZone(mappedPalm.screenX),
          screenX: mappedPalm.screenX,
          screenY: mappedPalm.screenY,
          wristY: mappedWrist.screenY,
          px: mappedPalm.px,
          py: mappedPalm.py,
          topPx: Number.isFinite(minPy) ? minPy : mappedPalm.py - 30,
          landmarks: rawLms,
        };
      };
      let tfFromLast: CoverTransform = computeMirroredCoverTransform(640, 480, 640, 480);

      const loop = (now: number) => {
        if (!active || !videoRef.current) return;
        const video = videoRef.current;
        const canvas = canvasRef.current;

        const dt = Math.min(0.1, Math.max(0.001, (now - (lastFrameTimeRef.current || now)) / 1000));
        lastFrameTimeRef.current = now;

        const vw = video.videoWidth || 640;
        const vh = video.videoHeight || 480;
        const cw = canvas?.clientWidth || vw;
        const ch = canvas?.clientHeight || vh;
        const dpr = typeof window !== 'undefined' ? Math.min(window.devicePixelRatio || 1, 2) : 1;

        if (canvas) {
          const targetW = Math.max(1, Math.round(cw * dpr));
          const targetH = Math.max(1, Math.round(ch * dpr));
          if (canvas.width !== targetW || canvas.height !== targetH) {
            canvas.width = targetW;
            canvas.height = targetH;
          }
        }

        const tf = computeMirroredCoverTransform(cw, ch, vw, vh);
        tfFromLast = tf;
        const ctx = canvas?.getContext('2d', { desynchronized: true }) ?? null;
        if (ctx) {
          ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
          ctx.clearRect(0, 0, cw, ch);
          drawCenterZoneDivider(ctx, cw, ch);
        }

        const isSpeechLocked = questionPhaseRef.current === 'READING_QUESTION';
        const isAnswerPaused = Boolean(feedbackRef.current) || submittingRef.current;
        const isEdgeFresh = Boolean(edgeHandStatus && Date.now() - edgeHandTs < 350);

        if (isSpeechLocked || isAnswerPaused) {
          holdProgressRef.current = 0;
          activeHandRef.current = null;
          lockedHandRef.current = null;
          palmPxRef.current = null;
          if (pendingChoiceRef.current !== null) {
            pendingChoiceRef.current = null;
            setPendingChoice(null);
          }
          if (progressBarFillRef.current) progressBarFillRef.current.style.transform = 'scaleX(0)';
          if (progressPctRef.current) progressPctRef.current.textContent = '0%';
          if (badgePctRef.current) badgePctRef.current.textContent = '0%';
        }

        if (!isAnswerPaused) {
          if (isEdgeFresh && edgeHandStatus) {
            const eh = edgeHandStatus;
            const rawEdgeHands: NormalizedLandmark[][] = [];
            if (eh.left_hand_landmarks && isValidStrict21HandLandmarks(eh.left_hand_landmarks)) {
              rawEdgeHands.push(
                eh.left_hand_landmarks.map((p) => ({ x: p.x, y: p.y, z: p.z ?? 0, visibility: 1 })),
              );
            }
            if (eh.right_hand_landmarks && isValidStrict21HandLandmarks(eh.right_hand_landmarks)) {
              rawEdgeHands.push(
                eh.right_hand_landmarks.map((p) => ({ x: p.x, y: p.y, z: p.z ?? 0, visibility: 1 })),
              );
            }
            const samples = rawEdgeHands.slice(0, 2).map(buildHandSample);
            detectedHandsRef.current = samples;
            const twoHandResult = evaluateTwoHandCardSelection(samples, MINIGAME_HAND_RAISE_THRESHOLD_Y);

            if (twoHandResult.bothHandsBlocked) {
              holdProgressRef.current = 0;
              lockedHandRef.current = null;
              activeHandRef.current = null;
              palmPxRef.current = null;
              if (!bothHandsWarnRef.current) {
                bothHandsWarnRef.current = true;
                setBothHandsWarn(true);
              }
            } else {
              if (bothHandsWarnRef.current) {
                bothHandsWarnRef.current = false;
                setBothHandsWarn(false);
              }
              if (!isSpeechLocked) {
                activeHandRef.current = twoHandResult.activeSide;
                const raisedSample = twoHandResult.activeSide
                  ? samples.find(
                      (s) =>
                        s.zone === twoHandResult.activeSide &&
                        Math.min(s.screenY, s.wristY ?? s.screenY) < MINIGAME_HAND_RAISE_THRESHOLD_Y,
                    )
                  : undefined;
                palmPxRef.current = raisedSample ? { px: raisedSample.px, py: raisedSample.py } : null;
              }
            }
          } else if (video.readyState >= 2 && mode === 'hand' && handLandmarker) {
            const isNewVideoFrame = video.currentTime !== lastVideoTimeRef.current;
            if (isNewVideoFrame) {
              lastVideoTimeRef.current = video.currentTime;

              let inferenceSource: HTMLVideoElement | HTMLCanvasElement = video;
              if (handDelegate !== 'GPU') {
                let offscreen = offscreenCanvasRef.current;
                if (!offscreen) {
                  offscreen = document.createElement('canvas');
                  offscreenCanvasRef.current = offscreen;
                }
                const offH = Math.max(144, Math.round(OFF_CANVAS_WIDTH * (vh / vw)));
                if (offscreen.width !== OFF_CANVAS_WIDTH || offscreen.height !== offH) {
                  offscreen.width = OFF_CANVAS_WIDTH;
                  offscreen.height = offH;
                }
                const offCtx = offscreen.getContext('2d', { alpha: false });
                if (offCtx) {
                  offCtx.drawImage(video, 0, 0, offscreen.width, offscreen.height);
                  inferenceSource = offscreen;
                }
              }
              const inferenceTs = getNextMonotonicHandDetectTs(now);
              const isPortrait = typeof window !== 'undefined' && window.innerHeight > window.innerWidth;

              let landmarksList: NormalizedLandmark[][] = [];
              let handednessList: Array<Array<{ score?: number }>> = [];
              try {
                const res = handLandmarker.detectForVideo(inferenceSource, inferenceTs);
                landmarksList = res.landmarks ?? [];
                handednessList = (res.handedness as Array<Array<{ score?: number }>> | undefined) ?? [];
              } catch {
                // Keep rAF hand-tracking loop resilient across transient MediaPipe/WebGL frame errors
              }
              // Strictly validate 21-point hand skeleton (NEVER treat face/head/shoulders as hands)
              const validHands = landmarksList
                .slice(0, 2)
                .filter((lms, idx) => {
                  const score = handednessList[idx]?.[0]?.score;
                  return isValidStrict21HandLandmarks(lms, score);
                });

              const samples = validHands.map(buildHandSample);
              detectedHandsRef.current = samples;

              if (samples.length > 0) {
                const nextCropWarn =
                  isPortrait && samples.some((s) => s.screenX < 0.08 || s.screenX > 0.92);
                if (cropWarnRef.current !== nextCropWarn) {
                  cropWarnRef.current = nextCropWarn;
                  setCropWarn(nextCropWarn);
                }

                const twoHandResult = evaluateTwoHandCardSelection(
                  samples,
                  MINIGAME_HAND_RAISE_THRESHOLD_Y,
                );

                if (twoHandResult.bothHandsBlocked) {
                  // Strict Anti-Dual-Raise Lock: immediately halt holdProgress = 0 and alert red
                  holdProgressRef.current = 0;
                  lockedHandRef.current = null;
                  activeHandRef.current = null;
                  palmPxRef.current = null;
                  if (pendingChoiceRef.current !== null) {
                    pendingChoiceRef.current = null;
                    setPendingChoice(null);
                  }
                  if (progressBarFillRef.current) progressBarFillRef.current.style.transform = 'scaleX(0)';
                  if (progressPctRef.current) progressPctRef.current.textContent = '0%';
                  if (badgePctRef.current) badgePctRef.current.textContent = '0%';
                  if (!bothHandsWarnRef.current) {
                    bothHandsWarnRef.current = true;
                    setBothHandsWarn(true);
                  }
                  const nowMs = Date.now();
                  // Never interrupt active question or feedback speech when user accidentally moves both hands
                  if (
                    !isSpeechLocked &&
                    !isSpeakingRef.current &&
                    !voiceMutedRef.current &&
                    nowMs - bothHandsWarnSpeechTsRef.current >= 3600
                  ) {
                    bothHandsWarnSpeechTsRef.current = nowMs;
                    const warnText =
                      voiceLangRef.current === 'en'
                        ? 'Please do not raise both hands at the same time. Lower one hand to answer.'
                        : voiceLangRef.current === 'zh'
                          ? '请勿同时举起双手，请放下一只手来作答。'
                          : 'ห้ามยก 2 มือพร้อมกัน กรุณาลดมือลง 1 ข้างเพื่อตอบ';
                    speakMinigameText(warnText, voiceLangRef.current, {
                      forceInterrupt: true,
                    });
                  }
                } else {
                  if (bothHandsWarnRef.current) {
                    bothHandsWarnRef.current = false;
                    setBothHandsWarn(false);
                  }
                  if (!isSpeechLocked) {
                    activeHandRef.current = twoHandResult.activeSide;
                    const raisedSample = twoHandResult.activeSide
                      ? samples.find(
                          (s) =>
                            s.zone === twoHandResult.activeSide &&
                            Math.min(s.screenY, s.wristY ?? s.screenY) < MINIGAME_HAND_RAISE_THRESHOLD_Y,
                        )
                      : undefined;
                    palmPxRef.current = raisedSample
                      ? { px: raisedSample.px, py: raisedSample.py }
                      : null;
                  }
                }
              } else {
                smoothedHandsRef.current = [];
                palmPxRef.current = null;
                activeHandRef.current = null;
                if (bothHandsWarnRef.current) {
                  bothHandsWarnRef.current = false;
                  setBothHandsWarn(false);
                }
                if (cropWarnRef.current) {
                  cropWarnRef.current = false;
                  setCropWarn(false);
                }
              }
            }
          }

          // Smooth 60fps delta-time hold progress only when READY_TO_ANSWER & no dual-hand violation
          const activeHand = isSpeechLocked || bothHandsWarnRef.current ? null : activeHandRef.current;
          if (activeHand) {
            if (!lockedHandRef.current || holdProgressRef.current <= 0.05) {
              lockedHandRef.current = activeHand;
              holdProgressRef.current = Math.min(1, holdProgressRef.current + dt / HOLD_DURATION_SEC);
            } else if (lockedHandRef.current === activeHand) {
              holdProgressRef.current = Math.min(1, holdProgressRef.current + dt / HOLD_DURATION_SEC);
            } else {
              holdProgressRef.current = Math.max(0, holdProgressRef.current - dt * HAND_SWITCH_DECAY_RATE);
              if (holdProgressRef.current === 0) {
                lockedHandRef.current = activeHand;
              }
            }
          } else {
            holdProgressRef.current = Math.max(0, holdProgressRef.current - dt * HAND_RELEASE_DECAY_RATE);
            if (holdProgressRef.current === 0) {
              lockedHandRef.current = null;
            }
          }

          const nextPendingChoice: 'yes' | 'no' | null =
            bothHandsWarnRef.current || isSpeechLocked
              ? null
              : lockedHandRef.current === 'RIGHT'
                ? 'yes'
                : lockedHandRef.current === 'LEFT'
                  ? 'no'
                  : null;
          if (pendingChoiceRef.current !== nextPendingChoice) {
            pendingChoiceRef.current = nextPendingChoice;
            setPendingChoice(nextPendingChoice);
          }

          const pctFloat = bothHandsWarnRef.current
            ? 0
            : Math.min(100, Math.max(0, holdProgressRef.current * 100));
          const pctRounded = `${Math.round(pctFloat)}%`;
          if (progressBarFillRef.current) {
            progressBarFillRef.current.style.transform = `scaleX(${(pctFloat / 100).toFixed(4)})`;
          }
          if (progressPctRef.current) {
            progressPctRef.current.textContent = pctRounded;
          }
          if (badgePctRef.current) {
            badgePctRef.current.textContent = pctRounded;
          }

          // Render Dual-Hand 21-Point Visualization on canvas (both hands simultaneously)
          if (ctx) {
            const smoothAlpha = 1 - Math.exp(-LANDMARK_SMOOTH_SPEED * dt);
            const currentHands = detectedHandsRef.current.slice(0, 2);

            if (bothHandsWarnRef.current && currentHands.length >= 2) {
              drawDualHandRedBridge(
                ctx,
                { px: currentHands[0].px, py: currentHands[0].py },
                { px: currentHands[1].px, py: currentHands[1].py },
              );
            }

            currentHands.forEach((handSample, handIdx) => {
              const smoothed = lerpLandmarks(
                smoothedHandsRef.current[handIdx] ?? null,
                handSample.landmarks,
                smoothAlpha,
              );
              smoothedHandsRef.current[handIdx] = smoothed;

              const isRaised =
                Math.min(handSample.screenY, handSample.wristY ?? handSample.screenY) <
                MINIGAME_HAND_RAISE_THRESHOLD_Y;

              if (bothHandsWarnRef.current) {
                drawHandSkeletonOnCanvas(ctx, smoothed, tf, '#ef4444');
                drawHandAlertBox(ctx, smoothed, tf, '#ef4444');
                drawHandPillLabel(
                  ctx,
                  handSample.px,
                  handSample.topPx,
                  'ยก 2 มือพร้อมกัน!',
                  'rgba(220, 38, 38, 0.94)',
                  '#ffffff',
                  cw,
                );
              } else if (isRaised && !isSpeechLocked) {
                const raisedColor = handSample.zone === 'RIGHT' ? '#10b981' : '#f43f5e';
                drawHandSkeletonOnCanvas(ctx, smoothed, tf, raisedColor);
                drawHandPillLabel(
                  ctx,
                  handSample.px,
                  handSample.topPx,
                  handSample.zone === 'RIGHT' ? 'มือขวา (ใช่ / YES)' : 'มือซ้าย (ไม่ใช่ / NO)',
                  handSample.zone === 'RIGHT' ? 'rgba(5, 150, 105, 0.92)' : 'rgba(225, 29, 72, 0.92)',
                  '#ffffff',
                  cw,
                );
              } else {
                drawHandSkeletonOnCanvas(ctx, smoothed, tf, '#38bdf8');
                drawHandPillLabel(
                  ctx,
                  handSample.px,
                  handSample.topPx,
                  handSample.zone === 'LEFT' ? 'มือซ้าย (เตรียมพร้อม)' : 'มือขวา (เตรียมพร้อม)',
                  'rgba(2, 132, 199, 0.88)',
                  '#ffffff',
                  cw,
                );
              }
            });

            if (currentHands.length < smoothedHandsRef.current.length) {
              smoothedHandsRef.current.length = currentHands.length;
            }

            if (
              !bothHandsWarnRef.current &&
              !isSpeechLocked &&
              palmPxRef.current &&
              holdProgressRef.current > 0.01 &&
              lockedHandRef.current
            ) {
              drawDwellHalo(
                ctx,
                palmPxRef.current.px,
                palmPxRef.current.py,
                holdProgressRef.current,
                lockedHandRef.current === 'RIGHT' ? '#10b981' : '#f43f5e',
              );
            }
          }

          if (!bothHandsWarnRef.current && !isSpeechLocked && holdProgressRef.current >= 1 && lockedHandRef.current) {
            const confirmed: 'yes' | 'no' = lockedHandRef.current === 'RIGHT' ? 'yes' : 'no';
            submitAnswerRef.current(confirmed);
          }
        }

        rafId = requestAnimationFrame(loop);
      };
      rafId = requestAnimationFrame(loop);
    }

    init();
    return () => {
      active = false;
      cancelAnimationFrame(rafId);
      edgeWs?.close();
      handLandmarker = null;
      stopCameraTracks();
    };
  }, [completedSummary, resetTrackingRefs, sessionKey, stopCameraTracks]);

  useEffect(() => {
    return () => {
      if (timerRef.current) window.clearTimeout(timerRef.current);
      spokenTurnKeyRef.current = '';
      stopCameraTracks();
      stopMinigameSpeech();
    };
  }, [stopCameraTracks, stopMinigameSpeech]);

  const handleReplay = () => {
    if (timerRef.current) window.clearTimeout(timerRef.current);
    spokenTurnKeyRef.current = '';
    stopMinigameSpeech();
    resetTrackingRefs();
    activeFeedbackMetaRef.current = null;
    submittingRef.current = false;
    startedAtRef.current = Date.now();
    setCards(sampleTenCognitiveCards());
    setIndex(0);
    setScore(0);
    setFeedback(null);
    setPendingChoice(null);
    setBothHandsWarn(false);
    setCropWarn(false);
    setCompletedSummary(null);
    setSessionKey((k) => k + 1);
  };

  if (completedSummary) {
    return (
      <div className="w-full flex-1 flex flex-col items-center justify-center rounded-3xl bg-white p-6 sm:p-12 shadow-xl border-2 border-slate-200 text-center space-y-6 animate-popup-spring">
        <div className="flex justify-center">
          <MinigameVoiceBar
            voiceLang={voiceLang}
            voiceMuted={voiceMuted}
            isSpeaking={isSpeaking}
            onSelectLang={handleSelectVoiceLang}
            onReplay={handleReplayVoice}
            onToggleMute={handleToggleVoiceMute}
          />
        </div>

        <div className="mx-auto flex h-32 w-32 items-center justify-center rounded-full bg-emerald-50 text-emerald-600 shadow-inner">
          <Award className="h-20 w-20" />
        </div>
        <div>
          <span className="mg-badge-top inline-flex items-center gap-2.5 rounded-full bg-indigo-50 px-6 py-2.5 font-black text-indigo-700">
            <Sparkles className="h-6 w-6 shrink-0" /> {patientName}
          </span>
          <h3 className="mg-summary-title mt-4 font-black text-slate-900">สรุปผลการเล่นเกมการ์ดกายภาพบำบัด</h3>
          <p className="mg-sub-instruction mt-2 font-bold text-slate-600">
            {savingResult ? 'กำลังบันทึกคะแนนลงฐานข้อมูล Supabase...' : 'บันทึกผลการฝึกครบทั้ง 10 ข้อเรียบร้อยแล้ว'}
          </p>
        </div>

        <div className="w-full max-w-5xl grid grid-cols-1 gap-6 sm:grid-cols-3">
          <div className="rounded-3xl border-2 border-emerald-200 bg-emerald-50 p-6 sm:p-8">
            <p className="mg-summary-stat-label font-extrabold text-emerald-700">คะแนนที่ได้</p>
            <p className="mg-summary-stat-val mt-2 font-black text-emerald-900">
              {completedSummary.correct} / {completedSummary.total}
            </p>
          </div>
          <div className="rounded-3xl border-2 border-indigo-200 bg-indigo-50 p-6 sm:p-8">
            <p className="mg-summary-stat-label font-extrabold text-indigo-700">ความแม่นยำ</p>
            <p className="mg-summary-stat-val mt-2 font-black text-indigo-900">{completedSummary.accuracyPct}%</p>
          </div>
          <div className="rounded-3xl border-2 border-slate-200 bg-slate-50 p-6 sm:p-8">
            <p className="mg-summary-stat-label font-extrabold text-slate-600 inline-flex items-center gap-2">
              <Clock className="h-7 w-7 shrink-0" /> เวลาที่ใช้
            </p>
            <p className="mg-summary-stat-val mt-2 font-black text-slate-900">{completedSummary.durationSec} วินาที</p>
          </div>
        </div>

        <div className="flex flex-wrap items-center justify-center gap-5 pt-2">
          {allowReplay && (
            <button
              type="button"
              onClick={handleReplay}
              className="mg-summary-btn inline-flex items-center gap-3 rounded-3xl border-2 border-indigo-200 bg-indigo-50 min-h-[84px] px-10 py-5 font-black text-indigo-700 hover:bg-indigo-100 active:scale-95 transition cursor-pointer"
            >
              <RotateCcw className="h-8 w-8 shrink-0" /> เล่นใหม่อีกครั้ง
            </button>
          )}
          <button
            type="button"
            onClick={() => {
              stopMinigameSpeech();
              stopCameraTracks();
              onComplete(completedSummary);
            }}
            className="mg-summary-btn inline-flex items-center gap-3 rounded-3xl bg-emerald-600 min-h-[84px] px-10 py-5 font-black text-white shadow-lg shadow-emerald-600/20 hover:bg-emerald-700 active:scale-95 transition cursor-pointer"
          >
            <CheckCircle2 className="h-8 w-8 shrink-0" /> ปิดหน้าต่าง
          </button>
        </div>
      </div>
    );
  }

  if (!currentCard) return null;

  const localizedQuestion = pickCardTextByVoiceLang(currentCard.question, voiceLang);
  const localizedSubject = pickCardTextByVoiceLang(currentCard.subject, voiceLang);
  const questionSizeClass = resolveMinigameQuestionSizeClass(localizedQuestion);
  const cardNumberBadge = formatCognitiveCardBadge(currentCard.id, index);
  const visualKind = resolveCognitiveCardVisualKind(currentCard);

  return (
    <div className="w-full flex-1 flex flex-col justify-between rounded-2xl bg-white p-2.5 sm:p-4 shadow-xl border border-slate-200 gap-2.5 sm:gap-3">
      {/* Ultra-Compact Single-Row Control Strip */}
      <div className="shrink-0 flex flex-wrap items-center justify-between gap-2 border-b border-slate-100 px-3 py-1.5">
        <div className="flex flex-wrap items-center gap-2 min-w-0">
          <span className="inline-flex items-center gap-1 rounded-full bg-indigo-50 px-2.5 py-0.5 text-xs font-medium text-indigo-700 shrink-0">
            <Sparkles className="h-3.5 w-3.5 shrink-0" />
            <span>{t('startCognitiveGameBtn')} • {patientName}</span>
          </span>
          <span className="text-xs font-medium text-slate-600">
            ยกมือค้าง 0.8 วินาทีเพื่อตอบ (ซ้าย = ไม่ใช่ / ขวา = ใช่) หรือคลิกปุ่มคำตอบได้ทันที
          </span>
        </div>

        <div className="flex flex-wrap items-center gap-2 shrink-0">
          <span
            key={score}
            className={`rounded-lg bg-gradient-to-r from-[#0F4C81] via-[#0284C7] to-[#10B981] px-3 py-1 text-xs md:text-sm font-bold text-white shadow-xs whitespace-nowrap ${
              score > 0 ? 'animate-rep-pop' : ''
            }`}
          >
            ข้อที่ {index + 1} / {cards.length} • คะแนน {score}
          </span>
          <MinigameVoiceBar
            voiceLang={voiceLang}
            voiceMuted={voiceMuted}
            isSpeaking={isSpeaking}
            onSelectLang={handleSelectVoiceLang}
            onReplay={handleReplayVoice}
            onToggleMute={handleToggleVoiceMute}
          />
        </div>
      </div>

      {/* Smooth 10-Question Progress Bar */}
      <div className="w-full h-1.5 shrink-0 rounded-full bg-slate-100 overflow-hidden">
        <div
          className="h-full rounded-full bg-gradient-to-r from-[#0284C7] to-[#10B981] transition-[width] duration-300 ease-out"
          style={{
            width: `${Math.min(100, Math.round(((index + (feedback ? 1 : 0)) / Math.max(1, cards.length)) * 100))}%`,
          }}
        />
      </div>

      {/* Main Vertical Stack: Top = Artwork + Giant Question Spotlight, Middle = Camera, Bottom = Answer Buttons */}
      <div className="flex-1 flex flex-col justify-between gap-2.5 sm:gap-3 min-h-0">
        {/* Question Card + Giant Elderly-First Question Spotlight Banner */}
        <div
          key={index}
          className="w-full shrink-0 p-2.5 sm:p-3.5 flex flex-col items-center justify-center gap-2.5 rounded-2xl bg-gradient-to-b from-white to-emerald-50/30 border border-indigo-200 text-center shadow-xs animate-view-enter"
        >
          <div
            data-testid="minigame-card-visual-frame"
            className="relative w-full h-[230px] sm:h-[270px] md:h-[310px] rounded-2xl flex items-center justify-center p-2 sm:p-3 overflow-hidden border-2 border-indigo-200 bg-gradient-to-br from-amber-50/60 via-white to-sky-50/60 shadow-inner"
          >
            <span className="absolute top-2.5 left-3 z-10 text-xs font-extrabold tracking-wider uppercase px-3 py-0.5 rounded-full bg-indigo-50/95 text-indigo-700 border border-indigo-200 shadow-xs backdrop-blur-xs">
              {currentCard.category.replace(/_/g, ' ')}
            </span>
            <span className="absolute top-2.5 right-3 z-10 text-xs font-extrabold px-2.5 py-0.5 rounded-full bg-amber-100/95 text-amber-900 border border-amber-300 shadow-xs backdrop-blur-xs">
              {cardNumberBadge}
            </span>

            {visualKind === 'svg_art' && currentCard.artKey ? (
              <div className="w-full h-full flex items-center justify-center [&_svg]:w-[200px] [&_svg]:h-[200px] sm:[&_svg]:w-[240px] sm:[&_svg]:h-[240px] md:[&_svg]:w-[280px] md:[&_svg]:h-[280px] [&_svg]:max-w-full [&_svg]:max-h-full [&_svg]:object-contain drop-shadow-md">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  viewBox="46 28 308 224"
                  role="img"
                  aria-label={localizedSubject}
                  className="w-full h-full transition-transform duration-300"
                  dangerouslySetInnerHTML={{ __html: getCardArtworkInnerSvg(currentCard.artKey) }}
                />
              </div>
            ) : visualKind === 'image_url' && currentCard.imageUrl ? (
              <img
                src={currentCard.imageUrl}
                alt={localizedSubject}
                className="w-full h-full max-h-[290px] object-contain drop-shadow-md scale-105"
                draggable={false}
              />
            ) : (
              <span
                aria-label={localizedSubject}
                className="text-3xl sm:text-4xl md:text-5xl font-black text-indigo-900 px-6 py-4 rounded-2xl bg-indigo-50 border-2 border-indigo-200 select-none"
              >
                {localizedSubject}
              </span>
            )}
          </div>

          <div
            data-testid="minigame-question-banner"
            className="w-full bg-amber-50/90 border-2 border-amber-300 rounded-2xl px-4 py-3 md:px-6 md:py-4 shadow-sm text-center"
          >
            <h2
              data-testid="minigame-question-title"
              className={`${questionSizeClass} font-black text-slate-950 tracking-tight break-words drop-shadow-sm`}
            >
              {localizedQuestion}
            </h2>
            {voiceLang !== 'th' && (
              <p className="mt-1.5 text-sm sm:text-base font-bold text-slate-700">
                ไทย: {currentCard.question.th}
              </p>
            )}
          </div>
        </div>

        {/* Middle Card: Full-Size Live Camera Viewport (120Hz Hand Tracking) */}
        <div className="flex-1 min-h-[280px] sm:min-h-[360px] relative w-full overflow-hidden rounded-3xl bg-gradient-to-br from-sky-100 via-emerald-50 to-white border-4 border-sky-200 shadow-inner">
          <video ref={videoRef} playsInline muted className="h-full w-full object-cover -scale-x-100" />
          <canvas ref={canvasRef} className="pointer-events-none absolute inset-0 h-full w-full" />

          {!cameraError && (
            <div className="pointer-events-none absolute top-4 inset-x-4 z-10 flex flex-wrap items-center justify-between gap-2">
              {(() => {
                const answerStatusPill = resolveMinigameAnswerStatusPill({
                  bothHandsWarn,
                  isQuestionSpeaking: questionPhase === 'READING_QUESTION',
                });
                return (
                  <span
                    data-testid="minigame-answer-status-pill"
                    className={`mg-cam-status inline-flex items-center gap-2 rounded-full px-5 py-2.5 font-black shadow-xl border-2 transition-colors duration-300 ${
                      answerStatusPill.variant === 'warn'
                        ? 'border-red-200 bg-red-600/95 text-white animate-pulse'
                        : answerStatusPill.variant === 'speaking'
                          ? 'border-indigo-300 bg-indigo-600/95 text-white'
                          : 'border-emerald-300 bg-emerald-600/95 text-white'
                    }`}
                  >
                    {answerStatusPill.text}
                  </span>
                );
              })()}

              {questionPhase === 'READY_TO_ANSWER' && pendingChoice === 'yes' && !bothHandsWarn ? (
                <span className="mg-cam-status inline-flex items-center gap-2.5 rounded-full bg-emerald-600/95 px-6 py-3 font-black text-white shadow-xl">
                  <span className="h-4 w-4 rounded-full bg-white animate-pulse shrink-0" />
                  <span>ตรวจพบมือขวา (ใช่)... </span>
                  <span ref={badgePctRef}>0%</span>
                </span>
              ) : questionPhase === 'READY_TO_ANSWER' && pendingChoice === 'no' && !bothHandsWarn ? (
                <span className="mg-cam-status inline-flex items-center gap-2.5 rounded-full bg-rose-600/95 px-6 py-3 font-black text-white shadow-xl">
                  <span className="h-4 w-4 rounded-full bg-white animate-pulse shrink-0" />
                  <span>ตรวจพบมือซ้าย (ไม่ใช่)... </span>
                  <span ref={badgePctRef}>0%</span>
                </span>
              ) : (
                <span className="mg-cam-status inline-flex items-center gap-2.5 rounded-full bg-white/95 px-5 py-2.5 font-black text-[#0F4C81] shadow-xl border-2 border-emerald-300">
                  <span className={`h-4 w-4 rounded-full shrink-0 ${detectorReady ? 'bg-emerald-500' : 'bg-amber-400 animate-pulse'}`} />
                  {detectorReady ? 'ตรวจจับมือ 21 จุด 2 ข้าง (เกณฑ์ y < 0.58)' : 'กำลังเตรียมระบบตรวจจับมือ...'}
                </span>
              )}
            </div>
          )}

          {cameraError && (
            <div className="absolute inset-0 z-20 flex flex-col items-center justify-center bg-gradient-to-br from-sky-50/95 via-white/95 to-emerald-50/95 p-8 text-center">
              <Hand className="h-24 w-24 mb-5 text-[#0284C7]" />
              <p className="mg-cam-status max-w-2xl font-black text-[#0F4C81]">{cameraError}</p>
            </div>
          )}

          {bothHandsWarn && questionPhase !== 'READING_QUESTION' && (
            <div className="pointer-events-none mg-cam-status absolute inset-x-6 top-1/2 -translate-y-1/2 z-20 flex items-center justify-center gap-3 rounded-3xl bg-red-600/95 border-2 border-white px-7 py-5 font-black text-white shadow-2xl text-center animate-toast-in animate-gentle-shake">
              <AlertTriangle className="h-10 w-10 shrink-0" />
              <span className="text-lg sm:text-xl">ห้ามยก 2 มือพร้อมกัน! กรุณาลดมือลง 1 ข้างเพื่อตอบ</span>
            </div>
          )}

          {cropWarn && !bothHandsWarn && (
            <div className="mg-cam-status absolute top-20 inset-x-4 z-10 flex items-center gap-3 rounded-2xl bg-sky-500/95 px-6 py-3.5 font-black text-white shadow-xl animate-toast-in">
              <AlertTriangle className="h-8 w-8 shrink-0" />
              <span>{t('portraitCropWarning')}</span>
            </div>
          )}

          {!cameraError && !pendingChoice && (
            <div className="pointer-events-none absolute bottom-4 inset-x-4 flex items-center justify-between gap-3">
              <span className="rounded-2xl bg-rose-50/95 px-5 py-2.5 text-base md:text-xl font-extrabold text-rose-700 border-2 border-rose-300 shadow-lg">
                ← ซ้าย: ไม่ใช่ (NO)
              </span>
              <span className="rounded-2xl bg-emerald-50/95 px-5 py-2.5 text-base md:text-xl font-extrabold text-emerald-800 border-2 border-emerald-300 shadow-lg">
                ขวา: ใช่ (YES) →
              </span>
            </div>
          )}

          {pendingChoice && (
            <div className="absolute bottom-4 inset-x-4 z-10 rounded-2xl bg-white/95 p-5 sm:p-6 text-[#0F4C81] border-2 border-sky-300 shadow-2xl">
              <div className="mg-hold-label flex items-center justify-between font-black mb-2.5">
                <span>{pendingChoice === 'yes' ? t('minigameRightYes') : t('minigameLeftNo')}</span>
                <span ref={progressPctRef}>0%</span>
              </div>
              <div className="h-7 sm:h-8 w-full overflow-hidden rounded-full bg-sky-100 border border-sky-200">
                <div
                  ref={progressBarFillRef}
                  className={`h-full w-full origin-left transition-colors duration-150 ${
                    pendingChoice === 'yes' ? 'bg-emerald-400' : 'bg-rose-500'
                  }`}
                  style={{ transform: 'scaleX(0)' }}
                />
              </div>
            </div>
          )}
        </div>

        {/* Bottom Controls: Feedback Banner or Giant YES/NO Answer Buttons */}
        <div className="shrink-0">
          {feedback ? (
            <div
              key={`feedback-${index}-${feedback.correct ? 'ok' : 'err'}`}
              className={`mg-feedback-banner min-h-[108px] sm:min-h-[132px] flex items-center justify-center gap-4 rounded-3xl border-4 p-6 font-black text-center animate-toast-in ${
                feedback.correct
                  ? 'border-emerald-300 bg-emerald-100 text-emerald-950 animate-rep-pop'
                  : 'border-rose-300 bg-rose-100 text-rose-950 animate-gentle-shake'
              }`}
            >
              {feedback.correct ? (
                <CheckCircle2 className="h-12 w-12 shrink-0 text-emerald-700" />
              ) : (
                <XCircle className="h-12 w-12 shrink-0 text-rose-700" />
              )}
              <span>{feedback.message}</span>
            </div>
          ) : (
            <div className="grid grid-cols-2 gap-4 sm:gap-6 text-center">
              <button
                type="button"
                disabled={questionPhase === 'READING_QUESTION'}
                onClick={() => submitAnswer('no')}
                className={`rounded-2xl border-4 py-4 md:py-5 px-5 sm:px-7 text-xl sm:text-2xl md:text-3xl font-extrabold text-rose-700 shadow-sm flex items-center justify-center transition ${
                  questionPhase === 'READING_QUESTION'
                    ? 'border-slate-200 bg-slate-100 opacity-60 cursor-not-allowed'
                    : pendingChoice === 'no'
                      ? 'border-rose-500 bg-rose-100 ring-4 ring-rose-300 shadow-lg active:scale-95 cursor-pointer'
                      : 'border-rose-300 bg-rose-50 hover:bg-rose-100 active:scale-95 cursor-pointer'
                }`}
              >
                <span>{t('minigameLeftNo')}</span>
              </button>
              <button
                type="button"
                disabled={questionPhase === 'READING_QUESTION'}
                onClick={() => submitAnswer('yes')}
                className={`rounded-2xl border-4 py-4 md:py-5 px-5 sm:px-7 text-xl sm:text-2xl md:text-3xl font-extrabold text-emerald-700 shadow-sm flex items-center justify-center transition ${
                  questionPhase === 'READING_QUESTION'
                    ? 'border-slate-200 bg-slate-100 opacity-60 cursor-not-allowed'
                    : pendingChoice === 'yes'
                      ? 'border-emerald-500 bg-emerald-100 ring-4 ring-emerald-300 shadow-lg active:scale-95 cursor-pointer'
                      : 'border-emerald-300 bg-emerald-50 hover:bg-emerald-100 active:scale-95 cursor-pointer'
                }`}
              >
                <span>{t('minigameRightYes')}</span>
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
