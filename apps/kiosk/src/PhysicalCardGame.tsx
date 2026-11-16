import React, { useCallback, useEffect, useRef, useState } from 'react';
import { FilesetResolver, HandLandmarker, PoseLandmarker, type NormalizedLandmark } from '@mediapipe/tasks-vision';
import { useClinicalI18n } from '@drtechcares/clinical-i18n';
import {
  AudioChimeSynthesizer,
} from '@drtechcares/kiosk-input';
import { selectClinicalCameraDeviceId } from '@drtechcares/face-client';
import {
  AlertTriangle,
  Award,
  CheckCircle2,
  Clock,
  Hand,
  Languages,
  RotateCcw,
  Sparkles,
  Volume2,
  VolumeX,
  XCircle,
} from 'lucide-react';
import { sampleTenCognitiveCards, type CognitiveCardQuestion } from './PhysicalCardGameData';

export type MinigameVoiceLang = 'th' | 'en' | 'zh';

export interface MinigameVoiceOption {
  code: MinigameVoiceLang;
  shortLabel: string;
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
const OFF_CANVAS_WIDTH = 256;

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

const POSE_HAND_CONNECTIONS_A: ReadonlyArray<[number, number]> = [
  [11, 13],
  [13, 15],
  [15, 17],
  [15, 19],
  [15, 21],
  [17, 19],
];

const POSE_HAND_CONNECTIONS_B: ReadonlyArray<[number, number]> = [
  [12, 14],
  [14, 16],
  [16, 18],
  [16, 20],
  [16, 22],
  [18, 20],
];

type ActiveHandSide = 'LEFT' | 'RIGHT' | null;

export const MINIGAME_HAND_RAISE_THRESHOLD_Y = 0.58;
export const MINIGAME_THAI_SPEECH_RATE = 1.18;
export type QuestionAnswerPhase = 'READING_QUESTION' | 'READY_TO_ANSWER';

export interface DetectedCardHandSample {
  zone: 'LEFT' | 'RIGHT';
  screenY: number;
  wristY?: number;
  minTipY?: number;
  px?: number;
  py?: number;
  landmarks?: NormalizedLandmark[];
}

export function evaluateTwoHandCardSelection(
  hands: ReadonlyArray<Pick<DetectedCardHandSample, 'zone' | 'screenY' | 'wristY' | 'minTipY'>>,
  thresholdY: number = MINIGAME_HAND_RAISE_THRESHOLD_Y,
): {
  activeSide: 'LEFT' | 'RIGHT' | null;
  leftRaised: boolean;
  rightRaised: boolean;
  bothHandsBlocked: boolean;
} {
  let leftRaised = false;
  let rightRaised = false;
  for (const hand of hands) {
    const effectiveY = Math.min(hand.screenY, hand.wristY ?? hand.screenY);
    if (effectiveY < thresholdY) {
      if (hand.zone === 'LEFT') leftRaised = true;
      if (hand.zone === 'RIGHT') rightRaised = true;
    }
  }
  if (leftRaised && rightRaised) {
    return { activeSide: null, leftRaised: true, rightRaised: true, bothHandsBlocked: true };
  }
  if (leftRaised) {
    return { activeSide: 'LEFT', leftRaised: true, rightRaised: false, bothHandsBlocked: false };
  }
  if (rightRaised) {
    return { activeSide: 'RIGHT', leftRaised: false, rightRaised: true, bothHandsBlocked: false };
  }
  return { activeSide: null, leftRaised: false, rightRaised: false, bothHandsBlocked: false };
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

function classifyMirroredHandZone(screenX: number, lockedHand: ActiveHandSide): 'LEFT' | 'RIGHT' {
  const splitThreshold = lockedHand === 'LEFT' ? 0.52 : lockedHand === 'RIGHT' ? 0.48 : 0.5;
  return screenX < splitThreshold ? 'LEFT' : 'RIGHT';
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

function drawAmberHandWarningBox(
  ctx: CanvasRenderingContext2D,
  landmarks: NormalizedLandmark[],
  tf: CoverTransform,
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
  ctx.strokeStyle = '#f59e0b';
  ctx.lineWidth = 4;
  ctx.strokeRect(minX - 18, minY - 18, Math.max(48, maxX - minX + 36), Math.max(48, maxY - minY + 36));
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

function drawPoseArmSkeletonOnCanvas(
  ctx: CanvasRenderingContext2D,
  landmarks: NormalizedLandmark[],
  connections: ReadonlyArray<[number, number]>,
  tf: CoverTransform,
  color: string,
) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 6;
  ctx.lineCap = 'round';
  const drawnPoints = new Set<number>();

  for (const [a, b] of connections) {
    const start = landmarks[a];
    const end = landmarks[b];
    if (!start || !end) continue;
    if ((start.visibility ?? 1) < 0.25 || (end.visibility ?? 1) < 0.25) continue;
    const pA = mapMirroredLandmarkToScreen(start, tf);
    const pB = mapMirroredLandmarkToScreen(end, tf);
    ctx.beginPath();
    ctx.moveTo(pA.px, pA.py);
    ctx.lineTo(pB.px, pB.py);
    ctx.stroke();
    drawnPoints.add(a);
    drawnPoints.add(b);
  }

  for (const idx of drawnPoints) {
    const lm = landmarks[idx];
    if (!lm) continue;
    const pt = mapMirroredLandmarkToScreen(lm, tf);
    ctx.fillStyle = idx === 15 || idx === 16 ? '#ffffff' : color;
    ctx.beginPath();
    ctx.arc(pt.px, pt.py, 8, 0, Math.PI * 2);
    ctx.fill();
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

async function resolveVisionTasks() {
  try {
    return await FilesetResolver.forVisionTasks('/mediapipe/wasm');
  } catch {
    return await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304/wasm',
    );
  }
}

interface MinigameVoiceBarProps {
  voiceLang: MinigameVoiceLang;
  voiceMuted: boolean;
  isSpeaking: boolean;
  spokenText: string;
  onSelectLang: (nextLang: MinigameVoiceLang) => void;
  onReplay: () => void;
  onToggleMute: () => void;
}

function MinigameVoiceBar({
  voiceLang,
  voiceMuted,
  isSpeaking,
  spokenText,
  onSelectLang,
  onReplay,
  onToggleMute,
}: MinigameVoiceBarProps) {
  const activeOption =
    MINIGAME_VOICE_LANG_OPTIONS.find((item) => item.code === voiceLang) ??
    MINIGAME_VOICE_LANG_OPTIONS[0];

  return (
    <div
      data-minigame-voice-control="true"
      className="w-full rounded-2xl border-2 border-sky-200 bg-gradient-to-r from-sky-50 via-white to-emerald-50/70 px-4 py-3 shadow-sm flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between text-left"
    >
      <div className="flex min-w-0 flex-1 items-center gap-3.5">
        <button
          type="button"
          onClick={onReplay}
          disabled={voiceMuted}
          title={activeOption.replayLabel}
          className={`flex h-12 w-12 shrink-0 items-center justify-center rounded-full shadow-md transition-transform active:scale-95 disabled:cursor-not-allowed disabled:opacity-50 cursor-pointer ${
            voiceMuted
              ? 'bg-slate-200 text-slate-500'
              : 'bg-gradient-to-br from-[#0284C7] to-[#10B981] text-white'
          }`}
        >
          {voiceMuted ? (
            <VolumeX className="h-6 w-6" />
          ) : (
            <Volume2 className={`h-6 w-6 ${isSpeaking ? 'animate-pulse' : ''}`} />
          )}
        </button>

        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2 text-xs font-extrabold uppercase tracking-wider text-[#0284C7]">
            <span>{activeOption.badgeLabel}</span>
            <span className="rounded-full bg-sky-100 px-2.5 py-0.5 text-[11px] font-black text-[#0F4C81]">
              {voiceMuted
                ? activeOption.statusMuted
                : isSpeaking
                  ? activeOption.statusSpeaking
                  : activeOption.statusReady}
            </span>
          </div>
          <p className="mt-0.5 line-clamp-2 text-base sm:text-lg font-extrabold text-[#0F4C81]">
            “{spokenText}”
          </p>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-2 shrink-0">
        <div
          role="group"
          aria-label="เลือกภาษาเสียงพากย์มินิเกม"
          className="inline-flex flex-wrap items-center gap-1 rounded-full border-2 border-sky-200 bg-white p-1 shadow-sm"
        >
          <Languages className="ml-2 mr-0.5 hidden sm:block h-4 w-4 text-[#0284C7]" />
          {MINIGAME_VOICE_LANG_OPTIONS.map((opt) => {
            const active = opt.code === voiceLang;
            return (
              <button
                key={opt.code}
                type="button"
                aria-pressed={active}
                onClick={() => onSelectLang(opt.code)}
                className={`rounded-full px-3.5 py-2 text-xs sm:text-sm font-black transition-all cursor-pointer ${
                  active
                    ? 'bg-[#0284C7] text-white shadow-sm'
                    : 'text-[#0F4C81] hover:bg-sky-50'
                }`}
              >
                {opt.shortLabel}
              </button>
            );
          })}
        </div>

        <button
          type="button"
          onClick={onReplay}
          disabled={voiceMuted}
          className="inline-flex shrink-0 items-center gap-1.5 rounded-full border-2 border-sky-200 bg-white px-3.5 py-2 text-xs sm:text-sm font-black text-[#0F4C81] transition-colors hover:bg-sky-50 active:scale-95 disabled:cursor-not-allowed disabled:opacity-50 cursor-pointer"
        >
          <RotateCcw className="h-4 w-4 text-[#0284C7]" />
          <span>{activeOption.replayLabel}</span>
        </button>

        <button
          type="button"
          onClick={onToggleMute}
          className={`inline-flex shrink-0 items-center gap-1.5 rounded-full border-2 px-3.5 py-2 text-xs sm:text-sm font-black transition-colors active:scale-95 cursor-pointer ${
            voiceMuted
              ? 'border-slate-300 bg-slate-100 text-slate-600 hover:bg-slate-200'
              : 'border-sky-200 bg-sky-50 text-[#0F4C81] hover:bg-sky-100'
          }`}
        >
          {voiceMuted ? (
            <>
              <VolumeX className="h-4 w-4" />
              <span>{activeOption.unmuteLabel}</span>
            </>
          ) : (
            <>
              <Volume2 className="h-4 w-4 text-[#0284C7]" />
              <span>{activeOption.muteLabel}</span>
            </>
          )}
        </button>
      </div>
    </div>
  );
}

export function PhysicalCardGame({
  patientName,
  onComplete,
  onSessionFinished,
  allowReplay = true,
}: PhysicalCardGameProps) {
  const { lang, t } = useClinicalI18n();
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
  const [voiceMuted, setVoiceMuted] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [questionPhase, setQuestionPhase] = useState<QuestionAnswerPhase>('READING_QUESTION');
  const questionPhaseRef = useRef<QuestionAnswerPhase>('READING_QUESTION');
  questionPhaseRef.current = questionPhase;
  const bothHandsWarnSpeechTsRef = useRef<number>(0);
  const allHandsLandmarksRef = useRef<NormalizedLandmark[][]>([]);

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
  const poseLandmarksRef = useRef<NormalizedLandmark[] | null>(null);
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

  const stopMinigameSpeech = useCallback(
    (clearQueue = true) => {
      speakSeqRef.current += 1;
      clearSpeakingWatchdog();
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
    [clearPendingSpeakTimer, clearSpeakingWatchdog],
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
          activeUtteranceRef.current = utterance;

          utterance.onstart = () => {
            if (speakSeqRef.current !== seq) return;
            blockedByAutoplayRef.current = false;
            isSpeakingRef.current = true;
            setIsSpeaking(true);
          };
          utterance.onend = () => {
            if (activeUtteranceRef.current === utterance) {
              activeUtteranceRef.current = null;
            }
            onMinigameSpeechEndedRef.current(seq);
          };
          utterance.onerror = (event: SpeechSynthesisErrorEvent) => {
            if (activeUtteranceRef.current === utterance) {
              activeUtteranceRef.current = null;
            }
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
          clearSpeakingWatchdog();
          const watchdogMs = Math.min(7500, Math.max(2200, clean.length * 80));
          speakingWatchdogRef.current = window.setTimeout(() => {
            speakingWatchdogRef.current = null;
            onMinigameSpeechEndedRef.current(seq);
          }, watchdogMs);
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
        isSpeakingRef.current = false;
        setIsSpeaking(false);
      }
    },
    [clearSpeakingWatchdog],
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
          audio.playbackRate = targetLang === 'th' ? MINIGAME_THAI_SPEECH_RATE : targetLang === 'zh' ? 0.98 : 1.0;

          audio.onplay = () => {
            if (speakSeqRef.current !== seq) return;
            blockedByAutoplayRef.current = false;
            isSpeakingRef.current = true;
            setIsSpeaking(true);
          };
          audio.onended = () => {
            onMinigameSpeechEndedRef.current(seq);
          };
          audio.onerror = () => {
            if (speakSeqRef.current !== seq) return;
            clearSpeakingWatchdog();
            isSpeakingRef.current = false;
            setIsSpeaking(false);
            speakWithWebSpeechFallback(clean, targetLang, seq);
          };

          audio.src = buildMinigameTtsUrl(clean, targetLang);
          const playPromise = audio.play();
          isSpeakingRef.current = true;
          setIsSpeaking(true);
          clearSpeakingWatchdog();
          const audioWatchdogMs = Math.min(7500, Math.max(2200, clean.length * 80));
          speakingWatchdogRef.current = window.setTimeout(() => {
            speakingWatchdogRef.current = null;
            onMinigameSpeechEndedRef.current(seq);
          }, audioWatchdogMs);

          if (playPromise && typeof playPromise.catch === 'function') {
            playPromise.catch((err: unknown) => {
              if (speakSeqRef.current !== seq) return;
              clearSpeakingWatchdog();
              isSpeakingRef.current = false;
              setIsSpeaking(false);
              const errName =
                err && typeof err === 'object' && 'name' in err
                  ? String((err as { name: string }).name)
                  : '';
              if (errName === 'NotAllowedError') {
                blockedByAutoplayRef.current = true;
                questionPhaseRef.current = 'READY_TO_ANSWER';
                setQuestionPhase('READY_TO_ANSWER');
                return;
              }
              if (errName === 'AbortError') {
                return;
              }
              speakWithWebSpeechFallback(clean, targetLang, seq);
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
    poseLandmarksRef.current = null;
    allHandsLandmarksRef.current = [];
    palmPxRef.current = null;
    pendingChoiceRef.current = null;
    bothHandsWarnRef.current = false;
    cropWarnRef.current = false;
    if (progressBarFillRef.current) progressBarFillRef.current.style.width = '0%';
    if (progressPctRef.current) progressPctRef.current.textContent = '0%';
    if (badgePctRef.current) badgePctRef.current.textContent = '0%';
  }, []);

  const currentCard = cards[index];

  // Auto-speak active card question when card changes or game resets (Speech-First Relay Gate)
  useEffect(() => {
    if (completedSummary || !currentCard || feedbackRef.current) return;
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
  }, [completedSummary, currentCard?.id, sessionKey, resetTrackingRefs, speakMinigameText]);

  // Auto-speak final summary when all 10 cards are finished
  useEffect(() => {
    if (!completedSummary) return;
    const summarySpeech = formatMinigameSummarySpeech(completedSummary, voiceLangRef.current);
    if (!voiceMutedRef.current && summarySpeech.trim()) {
      speakMinigameText(summarySpeech, voiceLangRef.current);
    }
  }, [completedSummary, speakMinigameText]);

  const handleSelectVoiceLang = useCallback(
    (nextLang: MinigameVoiceLang) => {
      voiceLangRef.current = nextLang;
      setVoiceLang(nextLang);
      if (voiceMutedRef.current) {
        voiceMutedRef.current = false;
        setVoiceMuted(false);
      }
      try {
        window.localStorage.setItem(VOICE_LANG_STORAGE_KEY, nextLang);
      } catch {
        // Ignore storage failure
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
        speakMinigameText(nextText, nextLang, { forceInterrupt: true });
      }
    },
    [getActiveVoiceText, speakMinigameText],
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
    const nextMuted = !voiceMutedRef.current;
    voiceMutedRef.current = nextMuted;
    setVoiceMuted(nextMuted);
    if (nextMuted) {
      stopMinigameSpeech(true);
      questionPhaseRef.current = 'READY_TO_ANSWER';
      setQuestionPhase('READY_TO_ANSWER');
      return;
    }
    const text = getActiveVoiceText(voiceLangRef.current);
    if (text) {
      speakMinigameText(text, voiceLangRef.current, { forceInterrupt: true });
    }
  }, [getActiveVoiceText, speakMinigameText, stopMinigameSpeech]);

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
          setIndex((i) => i + 1);
        }
      };

      const checkSpeechDoneThenAdvance = () => {
        if (
          (isSpeakingRef.current ||
            Boolean(queuedSpeechRef.current) ||
            Boolean(pendingSpeakTimerRef.current)) &&
          Date.now() - answerStartedAt < 6500
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
    let poseLandmarker: PoseLandmarker | null = null;
    let edgeWs: WebSocket | null = null;
    type EdgeHandTelemetry = {
      left_raised?: boolean;
      right_raised?: boolean;
      active_card_zone?: 'LEFT' | 'RIGHT' | 'BOTH' | 'NONE';
      left_palm_2d?: [number, number] | null;
      right_palm_2d?: [number, number] | null;
      left_hand_landmarks?: Array<{ x: number; y: number; z?: number }>;
      right_hand_landmarks?: Array<{ x: number; y: number; z?: number }>;
    };
    let edgeHandStatus: EdgeHandTelemetry | null = null;
    let edgeHandTs = 0;
    setCameraError(null);
    setDetectorReady(false);
    resetTrackingRefs();

    // Optional Edge Vision WS (RTMW3D-l 133-point Wholebody) with automatic MediaPipe fallback
    try {
      const wsUrl = (import.meta.env.VITE_EDGE_VISION_WS as string | undefined)?.trim() || 'ws://localhost:8000/ws/kiosk';
      edgeWs = new WebSocket(wsUrl);
      edgeWs.onmessage = (evt) => {
        if (!active) return;
        try {
          const msg = JSON.parse(evt.data) as { hand_status?: EdgeHandTelemetry };
          if (msg?.hand_status) {
            edgeHandStatus = msg.hand_status;
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
    } catch {
      edgeWs = null;
    }

    async function createHandOrPoseDetector() {
      const vision = await resolveVisionTasks();
      const handUrls = [
        '/models/hand_landmarker.task',
        'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
      ];
      for (const modelAssetPath of handUrls) {
        for (const delegate of ['CPU', 'GPU'] as const) {
          try {
            handLandmarker = await HandLandmarker.createFromOptions(vision, {
              baseOptions: { modelAssetPath, delegate },
              runningMode: 'VIDEO',
              numHands: 2,
              minHandDetectionConfidence: 0.4,
              minHandPresenceConfidence: 0.4,
              minTrackingConfidence: 0.4,
            });
            return 'hand' as const;
          } catch {
            // try GPU delegate or next model path
          }
        }
      }

      const poseUrls = ['/models/pose_landmarker.task', '/models/pose_landmarker_full.task'];
      for (const modelAssetPath of poseUrls) {
        for (const delegate of ['CPU', 'GPU'] as const) {
          try {
            poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
              baseOptions: { modelAssetPath, delegate },
              runningMode: 'VIDEO',
              numPoses: 1,
              minPoseDetectionConfidence: 0.45,
              minPosePresenceConfidence: 0.45,
              minTrackingConfidence: 0.45,
            });
            return 'pose' as const;
          } catch {
            // try GPU delegate or next pose model path
          }
        }
      }

      return 'none' as const;
    }

    async function init() {
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
        if (active) {
          setCameraError('ไม่สามารถเปิดกล้องได้ กรุณาอนุญาตการใช้กล้อง หรือคลิกปุ่มคำตอบด้านล่างได้เลย');
        }
        return;
      }

      if (!active || !videoRef.current) {
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

      const mode = await createHandOrPoseDetector();
      if (!active) return;

      const edgeAlive = () => Boolean(edgeHandStatus && Date.now() - edgeHandTs < 450);
      if (mode === 'none' && !edgeAlive()) {
        setCameraError('ระบบตรวจจับมือไม่พร้อมใช้งาน กรุณาคลิกเลือกคำตอบด้านล่างแทน');
        return;
      }

      if (mode !== 'none' || edgeAlive()) {
        setDetectorReady(true);
      }
      lastFrameTimeRef.current = performance.now();
      lastVideoTimeRef.current = -1;
      lastInferenceTsRef.current = 0;

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
        const ctx = canvas?.getContext('2d', { desynchronized: true }) ?? null;
        if (ctx) {
          ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
          ctx.clearRect(0, 0, cw, ch);
          drawCenterZoneDivider(ctx, cw, ch);
        }

        const isSpeechLocked = questionPhaseRef.current === 'READING_QUESTION';
        const isTrackingPaused = Boolean(feedbackRef.current) || submittingRef.current || isSpeechLocked;
        const isEdgeFresh = Boolean(edgeHandStatus && Date.now() - edgeHandTs < 350);

        if (isSpeechLocked) {
          holdProgressRef.current = 0;
          activeHandRef.current = null;
          lockedHandRef.current = null;
          palmPxRef.current = null;
          allHandsLandmarksRef.current = [];
          if (pendingChoiceRef.current !== null) {
            pendingChoiceRef.current = null;
            setPendingChoice(null);
          }
          if (progressBarFillRef.current) progressBarFillRef.current.style.width = '0%';
          if (progressPctRef.current) progressPctRef.current.textContent = '0%';
          if (badgePctRef.current) badgePctRef.current.textContent = '0%';
        }

        if (!isTrackingPaused && isEdgeFresh && edgeHandStatus) {
          const eh = edgeHandStatus;
          const nextBothWarn = eh.active_card_zone === 'BOTH';
          if (bothHandsWarnRef.current !== nextBothWarn) {
            bothHandsWarnRef.current = nextBothWarn;
            setBothHandsWarn(nextBothWarn);
          }
          if (nextBothWarn) {
            holdProgressRef.current = 0;
            activeHandRef.current = null;
            lockedHandRef.current = null;
            palmPxRef.current = null;
            targetLandmarksRef.current = null;
          } else if (eh.active_card_zone === 'LEFT' || eh.active_card_zone === 'RIGHT') {
            activeHandRef.current = eh.active_card_zone;
            const rawLms = eh.active_card_zone === 'LEFT' ? eh.left_hand_landmarks : eh.right_hand_landmarks;
            targetLandmarksRef.current =
              rawLms && rawLms.length >= 21
                ? rawLms.map((p) => ({ x: p.x, y: p.y, z: p.z ?? 0, visibility: 1 }))
                : null;
            const palmRaw = eh.active_card_zone === 'LEFT' ? eh.left_palm_2d : eh.right_palm_2d;
            if (palmRaw) {
              const mapped = mapMirroredLandmarkToScreen({ x: palmRaw[0], y: palmRaw[1] }, tf);
              palmPxRef.current = { px: mapped.px, py: mapped.py };
            }
          } else {
            activeHandRef.current = null;
            palmPxRef.current = null;
            targetLandmarksRef.current = null;
            smoothedLandmarksRef.current = null;
          }
        } else if (!isTrackingPaused && video.readyState >= 2) {
          const isNewVideoFrame = video.currentTime !== lastVideoTimeRef.current;
          if (isNewVideoFrame) {
            lastVideoTimeRef.current = video.currentTime;

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
            }
            const inferenceSource = offCtx ? offscreen : video;
            const inferenceTs = Math.max(lastInferenceTsRef.current + 1, Math.round(now));
            lastInferenceTsRef.current = inferenceTs;
            const isPortrait = typeof window !== 'undefined' && window.innerHeight > window.innerWidth;

            if (mode === 'hand' && handLandmarker) {
              const res = handLandmarker.detectForVideo(inferenceSource, inferenceTs);
              const detectedHandsList = (res.landmarks ?? []).filter((lms) => lms.length >= 21);
              allHandsLandmarksRef.current = detectedHandsList;
              if (detectedHandsList.length > 0) {
                const samples: DetectedCardHandSample[] = detectedHandsList.map((handLms) => {
                  const wrist = handLms[0];
                  const indexMcp = handLms[5];
                  const middleMcp = handLms[9];
                  const pinkyMcp = handLms[17];
                  const middleTip = handLms[12];
                  const palmRaw = {
                    x: (wrist.x + indexMcp.x + middleMcp.x + pinkyMcp.x + middleTip.x) / 5,
                    y: (wrist.y + indexMcp.y + middleMcp.y + pinkyMcp.y + middleTip.y) / 5,
                  };
                  const mappedPalm = mapMirroredLandmarkToScreen(palmRaw, tf);
                  const mappedWrist = mapMirroredLandmarkToScreen(wrist, tf);
                  const zone = classifyMirroredHandZone(mappedPalm.screenX, lockedHandRef.current);
                  return {
                    zone,
                    screenY: mappedPalm.screenY,
                    wristY: mappedWrist.screenY,
                    px: mappedPalm.px,
                    py: mappedPalm.py,
                    landmarks: handLms,
                  };
                });

                const twoHandResult = evaluateTwoHandCardSelection(samples, MINIGAME_HAND_RAISE_THRESHOLD_Y);
                const nextCropWarn = isPortrait && samples.some((s) => {
                  const wristScreen = s.landmarks?.[0] ? mapMirroredLandmarkToScreen(s.landmarks[0], tf).screenX : 0.5;
                  return wristScreen < 0.08 || wristScreen > 0.92;
                });
                if (cropWarnRef.current !== nextCropWarn) {
                  cropWarnRef.current = nextCropWarn;
                  setCropWarn(nextCropWarn);
                }

                if (twoHandResult.bothHandsBlocked) {
                  // Mutual Exclusion Rule: immediate hold reset & amber two-hand alert + brief voice warning
                  holdProgressRef.current = 0;
                  lockedHandRef.current = null;
                  activeHandRef.current = null;
                  palmPxRef.current = null;
                  targetLandmarksRef.current = null;
                  if (progressBarFillRef.current) progressBarFillRef.current.style.width = '0%';
                  if (progressPctRef.current) progressPctRef.current.textContent = '0%';
                  if (badgePctRef.current) badgePctRef.current.textContent = '0%';
                  if (!bothHandsWarnRef.current) {
                    bothHandsWarnRef.current = true;
                    setBothHandsWarn(true);
                  }
                  const nowMs = Date.now();
                  if (!voiceMutedRef.current && nowMs - bothHandsWarnSpeechTsRef.current >= 3600) {
                    bothHandsWarnSpeechTsRef.current = nowMs;
                    speakMinigameText('กรุณายกมือเพียงข้างเดียว', voiceLangRef.current, { forceInterrupt: true });
                  }
                } else {
                  if (bothHandsWarnRef.current) {
                    bothHandsWarnRef.current = false;
                    setBothHandsWarn(false);
                  }
                  const selectedSample = twoHandResult.activeSide
                    ? samples.find(
                        (s) =>
                          s.zone === twoHandResult.activeSide &&
                          Math.min(s.screenY, s.wristY ?? s.screenY) < MINIGAME_HAND_RAISE_THRESHOLD_Y,
                      ) ?? samples[0]
                    : samples[0];
                  targetLandmarksRef.current = selectedSample.landmarks ?? null;
                  palmPxRef.current =
                    selectedSample.px !== undefined && selectedSample.py !== undefined
                      ? { px: selectedSample.px, py: selectedSample.py }
                      : null;
                  activeHandRef.current = twoHandResult.activeSide;
                }
              } else {
                targetLandmarksRef.current = null;
                smoothedLandmarksRef.current = null;
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
            } else if (mode === 'pose' && poseLandmarker) {
              const res = poseLandmarker.detectForVideo(inferenceSource, inferenceTs);
              const pose = res.landmarks?.[0] ?? null;
              poseLandmarksRef.current = pose;
              const poseSamples: DetectedCardHandSample[] = [];
              let nextCropWarn = false;

              if (pose) {
                const groups = [
                  { shoulder: pose[11], wrist: pose[15], index: pose[19] },
                  { shoulder: pose[12], wrist: pose[16], index: pose[20] },
                ];
                for (const g of groups) {
                  if (!g.wrist || (g.wrist.visibility ?? 1) < 0.35) continue;
                  const palmRaw = {
                    x: g.index && (g.index.visibility ?? 1) > 0.3 ? (g.wrist.x + g.index.x) / 2 : g.wrist.x,
                    y: g.index && (g.index.visibility ?? 1) > 0.3 ? (g.wrist.y + g.index.y) / 2 : g.wrist.y,
                  };
                  const mappedPalm = mapMirroredLandmarkToScreen(palmRaw, tf);
                  const mappedWrist = mapMirroredLandmarkToScreen(g.wrist, tf);
                  if (isPortrait && (mappedPalm.screenX < 0.08 || mappedPalm.screenX > 0.92)) {
                    nextCropWarn = true;
                  }
                  const zone = classifyMirroredHandZone(mappedPalm.screenX, lockedHandRef.current);
                  poseSamples.push({
                    zone,
                    screenY: mappedPalm.screenY,
                    wristY: mappedWrist.screenY,
                    px: mappedPalm.px,
                    py: mappedPalm.py,
                  });
                }
              }

              const poseEval = evaluateTwoHandCardSelection(poseSamples, MINIGAME_HAND_RAISE_THRESHOLD_Y);
              if (bothHandsWarnRef.current !== poseEval.bothHandsBlocked) {
                bothHandsWarnRef.current = poseEval.bothHandsBlocked;
                setBothHandsWarn(poseEval.bothHandsBlocked);
              }
              if (cropWarnRef.current !== nextCropWarn) {
                cropWarnRef.current = nextCropWarn;
                setCropWarn(nextCropWarn);
              }

              if (poseEval.bothHandsBlocked) {
                holdProgressRef.current = 0;
                lockedHandRef.current = null;
                activeHandRef.current = null;
                palmPxRef.current = null;
                if (progressBarFillRef.current) progressBarFillRef.current.style.width = '0%';
                if (progressPctRef.current) progressPctRef.current.textContent = '0%';
                if (badgePctRef.current) badgePctRef.current.textContent = '0%';
                const nowMs = Date.now();
                if (!voiceMutedRef.current && nowMs - bothHandsWarnSpeechTsRef.current >= 3600) {
                  bothHandsWarnSpeechTsRef.current = nowMs;
                  speakMinigameText('กรุณายกมือเพียงข้างเดียว', voiceLangRef.current, { forceInterrupt: true });
                }
              } else {
                activeHandRef.current = poseEval.activeSide;
                const matchedSample = poseEval.activeSide
                  ? poseSamples.find(
                      (s) =>
                        s.zone === poseEval.activeSide &&
                        Math.min(s.screenY, s.wristY ?? s.screenY) < MINIGAME_HAND_RAISE_THRESHOLD_Y,
                    )
                  : undefined;
                palmPxRef.current =
                  matchedSample?.px !== undefined && matchedSample?.py !== undefined
                    ? { px: matchedSample.px, py: matchedSample.py }
                    : null;
              }
            }
          }

          // Smooth 60fps delta-time hold progress (0.8s confirmation, smooth decay)
          const activeHand = activeHandRef.current;
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
            lockedHandRef.current === 'RIGHT' ? 'yes' : lockedHandRef.current === 'LEFT' ? 'no' : null;
          if (pendingChoiceRef.current !== nextPendingChoice) {
            pendingChoiceRef.current = nextPendingChoice;
            setPendingChoice(nextPendingChoice);
          }

          // Update progress width and percentage directly on DOM at 60fps without React re-renders
          const pctFloat = Math.min(100, Math.max(0, holdProgressRef.current * 100));
          const pctRounded = `${Math.round(pctFloat)}%`;
          if (progressBarFillRef.current) {
            progressBarFillRef.current.style.width = `${pctFloat.toFixed(1)}%`;
          }
          if (progressPctRef.current) {
            progressPctRef.current.textContent = pctRounded;
          }
          if (badgePctRef.current) {
            badgePctRef.current.textContent = pctRounded;
          }

          // Render interpolated 120fps hand or pose skeleton + amber dual-hand boxes + progress halo directly on canvas
          if (ctx) {
            const smoothAlpha = 1 - Math.exp(-LANDMARK_SMOOTH_SPEED * dt);
            if (bothHandsWarnRef.current && allHandsLandmarksRef.current.length >= 2) {
              for (const handSet of allHandsLandmarksRef.current) {
                drawHandSkeletonOnCanvas(ctx, handSet, tf, '#f59e0b');
                drawAmberHandWarningBox(ctx, handSet, tf);
              }
            } else if ((mode === 'hand' || isEdgeFresh) && targetLandmarksRef.current) {
              smoothedLandmarksRef.current = lerpLandmarks(
                smoothedLandmarksRef.current,
                targetLandmarksRef.current,
                smoothAlpha,
              );
              const skelColor =
                (activeHand ?? lockedHandRef.current) === 'RIGHT'
                  ? '#10b981'
                  : (activeHand ?? lockedHandRef.current) === 'LEFT'
                    ? '#f43f5e'
                    : '#38bdf8';
              drawHandSkeletonOnCanvas(ctx, smoothedLandmarksRef.current, tf, skelColor);
            } else if (mode === 'pose' && poseLandmarksRef.current && activeHand) {
              const skelColor = activeHand === 'RIGHT' ? '#10b981' : '#f43f5e';
              drawPoseArmSkeletonOnCanvas(ctx, poseLandmarksRef.current, POSE_HAND_CONNECTIONS_A, tf, skelColor);
              drawPoseArmSkeletonOnCanvas(ctx, poseLandmarksRef.current, POSE_HAND_CONNECTIONS_B, tf, skelColor);
            }

            if (palmPxRef.current && holdProgressRef.current > 0.01 && lockedHandRef.current) {
              drawDwellHalo(
                ctx,
                palmPxRef.current.px,
                palmPxRef.current.py,
                holdProgressRef.current,
                lockedHandRef.current === 'RIGHT' ? '#10b981' : '#f43f5e',
              );
            }
          }

          if (holdProgressRef.current >= 1 && lockedHandRef.current) {
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
      handLandmarker?.close();
      poseLandmarker?.close();
      stopCameraTracks();
      stopMinigameSpeech();
    };
  }, [completedSummary, resetTrackingRefs, sessionKey, stopCameraTracks, stopMinigameSpeech]);

  useEffect(() => {
    return () => {
      if (timerRef.current) window.clearTimeout(timerRef.current);
      stopCameraTracks();
      stopMinigameSpeech();
    };
  }, [stopCameraTracks, stopMinigameSpeech]);

  const handleReplay = () => {
    if (timerRef.current) window.clearTimeout(timerRef.current);
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
    const summarySpeechText = formatMinigameSummarySpeech(completedSummary, voiceLang);
    return (
      <div className="w-full flex-1 flex flex-col items-center justify-center rounded-3xl bg-white p-6 sm:p-12 shadow-xl border-2 border-slate-200 text-center space-y-6">
        <div className="w-full max-w-5xl">
          <MinigameVoiceBar
            voiceLang={voiceLang}
            voiceMuted={voiceMuted}
            isSpeaking={isSpeaking}
            spokenText={summarySpeechText}
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
  const activeVoiceQuote = feedback ? feedback.message : localizedQuestion;

  return (
    <div className="w-full flex-1 flex flex-col justify-between rounded-3xl bg-white p-4 sm:p-7 shadow-xl border-2 border-slate-200 gap-4 sm:gap-5">
      {/* Header Section */}
      <div className="shrink-0 flex flex-col gap-3 border-b-2 border-slate-100 pb-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <span className="mg-badge-top inline-flex items-center gap-2.5 rounded-full bg-indigo-50 px-5 py-2.5 font-black text-indigo-700">
            <Sparkles className="h-6 w-6 shrink-0" /> {t('startCognitiveGameBtn')} • {patientName}
          </span>
          <span className="mg-score-pill rounded-2xl bg-gradient-to-r from-[#0F4C81] via-[#0284C7] to-[#10B981] px-7 py-3.5 font-black text-white shadow-md whitespace-nowrap">
            ข้อที่ {index + 1} / {cards.length} • คะแนน {score}
          </span>
        </div>
        <p className="mg-sub-instruction font-extrabold text-slate-700">
          ยกมือค้าง 0.8 วินาทีเพื่อตอบ (ซ้าย = ไม่ใช่ / ขวา = ใช่) หรือคลิกปุ่มคำตอบได้ทันที
        </p>
        <MinigameVoiceBar
          voiceLang={voiceLang}
          voiceMuted={voiceMuted}
          isSpeaking={isSpeaking}
          spokenText={activeVoiceQuote}
          onSelectLang={handleSelectVoiceLang}
          onReplay={handleReplayVoice}
          onToggleMute={handleToggleVoiceMute}
        />
      </div>

      {/* Main Full-Size Vertical Stack: Top = Question & Illustration, Middle = Camera, Bottom = Answer Buttons */}
      <div className="flex-1 flex flex-col justify-between gap-4 sm:gap-5 min-h-0">
        {/* Top Card: Question & Illustration */}
        <div className="flex-[1] min-h-[360px] sm:min-h-[450px] rounded-3xl bg-gradient-to-br from-indigo-50 via-white to-sky-50 p-6 sm:p-8 border-2 border-indigo-200 text-center flex flex-col items-center justify-center shadow-sm">
          <span className="mg-category-pill inline-block rounded-full bg-white px-7 py-2.5 font-black uppercase tracking-wider text-indigo-700 shadow-sm border-2 border-indigo-100">
            {currentCard.category.replace('_', ' ')}
          </span>
          <div className="my-3 sm:my-5 flex items-center justify-center w-full">
            <img
              src={currentCard.imageUrl}
              alt={localizedSubject}
              className="h-64 w-[26rem] sm:h-80 sm:w-[34rem] lg:h-[22rem] lg:w-[38rem] max-w-full rounded-3xl object-contain shadow-lg border-4 border-indigo-200 bg-white p-2"
              draggable={false}
            />
          </div>
          <h2 className="mg-question-title font-black text-[#0F4C81] max-w-5xl">
            {localizedQuestion}
          </h2>
          {voiceLang !== 'th' && (
            <p className="mt-2 text-lg sm:text-xl font-bold text-slate-500 max-w-4xl">
              ไทย: {currentCard.question.th}
            </p>
          )}
        </div>

        {/* Middle Card: Full-Size Live Camera Viewport (120Hz Hand Tracking) */}
        <div className="flex-[1.4] min-h-[360px] sm:min-h-[480px] relative w-full overflow-hidden rounded-3xl bg-gradient-to-br from-sky-100 via-emerald-50 to-white border-4 border-sky-200 shadow-inner">
          <video ref={videoRef} playsInline muted className="h-full w-full object-cover -scale-x-100" />
          <canvas ref={canvasRef} className="pointer-events-none absolute inset-0 h-full w-full" />

          {!cameraError && (
            <div className="pointer-events-none absolute top-4 inset-x-4 z-10 flex flex-wrap items-center justify-between gap-2">
              <span
                className={`mg-cam-status inline-flex items-center gap-2 rounded-full px-5 py-2.5 font-black shadow-xl backdrop-blur border-2 ${
                  questionPhase === 'READING_QUESTION'
                    ? 'border-indigo-300 bg-indigo-600/95 text-white'
                    : 'border-emerald-300 bg-emerald-600/95 text-white'
                }`}
              >
                {questionPhase === 'READING_QUESTION'
                  ? '🔊 กรุณาฟังคำถามให้จบก่อนยกมือตอบ...'
                  : '✋ ยกมือข้างเดียวเพื่อตอบได้เลย!'}
              </span>

              {questionPhase === 'READY_TO_ANSWER' && pendingChoice === 'yes' ? (
                <span className="mg-cam-status inline-flex items-center gap-2.5 rounded-full bg-emerald-600/95 px-6 py-3 font-black text-white shadow-xl backdrop-blur">
                  <span className="h-4 w-4 rounded-full bg-white animate-pulse shrink-0" />
                  <span>ตรวจพบมือขวา (ใช่)... </span>
                  <span ref={badgePctRef}>0%</span>
                </span>
              ) : questionPhase === 'READY_TO_ANSWER' && pendingChoice === 'no' ? (
                <span className="mg-cam-status inline-flex items-center gap-2.5 rounded-full bg-rose-600/95 px-6 py-3 font-black text-white shadow-xl backdrop-blur">
                  <span className="h-4 w-4 rounded-full bg-white animate-pulse shrink-0" />
                  <span>ตรวจพบมือซ้าย (ไม่ใช่)... </span>
                  <span ref={badgePctRef}>0%</span>
                </span>
              ) : (
                <span className="mg-cam-status inline-flex items-center gap-2.5 rounded-full bg-white/95 px-5 py-2.5 font-black text-[#0F4C81] shadow-xl backdrop-blur border-2 border-emerald-300">
                  <span className={`h-4 w-4 rounded-full shrink-0 ${detectorReady ? 'bg-emerald-500' : 'bg-amber-400 animate-pulse'}`} />
                  {detectorReady ? 'ตรวจจับ 2 มือ 120Hz (เกณฑ์ y < 0.58)' : 'กำลังเตรียมระบบตรวจจับมือ...'}
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

          {bothHandsWarn && (
            <div className="mg-cam-status absolute top-20 inset-x-4 z-10 flex items-center gap-3 rounded-2xl bg-amber-500/95 px-6 py-3.5 font-black text-white shadow-xl">
              <AlertTriangle className="h-8 w-8 shrink-0" />
              <span>⚠️ ห้ามยกมือพร้อมกัน 2 ข้าง กรุณายกมือเพียงข้างเดียวเพื่อตอบ</span>
            </div>
          )}

          {cropWarn && !bothHandsWarn && (
            <div className="mg-cam-status absolute top-20 inset-x-4 z-10 flex items-center gap-3 rounded-2xl bg-sky-500/95 px-6 py-3.5 font-black text-white shadow-xl">
              <AlertTriangle className="h-8 w-8 shrink-0" />
              <span>{t('portraitCropWarning')}</span>
            </div>
          )}

          {!cameraError && !pendingChoice && (
            <div className="pointer-events-none absolute bottom-4 inset-x-4 flex items-center justify-between gap-3">
              <span className="mg-cam-zone rounded-2xl bg-rose-50/95 px-6 py-3 font-black text-rose-700 border-2 border-rose-300 shadow-lg">
                ← ซ้าย: ไม่ใช่ (NO)
              </span>
              <span className="mg-cam-zone rounded-2xl bg-emerald-50/95 px-6 py-3 font-black text-emerald-800 border-2 border-emerald-300 shadow-lg">
                ขวา: ใช่ (YES) →
              </span>
            </div>
          )}

          {pendingChoice && (
            <div className="absolute bottom-4 inset-x-4 z-10 rounded-2xl bg-white/95 p-5 sm:p-6 text-[#0F4C81] backdrop-blur border-2 border-sky-300 shadow-2xl">
              <div className="mg-hold-label flex items-center justify-between font-black mb-2.5">
                <span>{pendingChoice === 'yes' ? t('minigameRightYes') : t('minigameLeftNo')}</span>
                <span ref={progressPctRef}>0%</span>
              </div>
              <div className="h-7 sm:h-8 w-full overflow-hidden rounded-full bg-sky-100 border border-sky-200">
                <div
                  ref={progressBarFillRef}
                  className={`h-full will-change-[width] ${
                    pendingChoice === 'yes' ? 'bg-emerald-400' : 'bg-rose-500'
                  }`}
                  style={{ width: '0%' }}
                />
              </div>
            </div>
          )}
        </div>

        {/* Bottom Controls: Feedback Banner or Giant YES/NO Answer Buttons */}
        <div className="shrink-0">
          {feedback ? (
            <div
              className={`mg-feedback-banner min-h-[108px] sm:min-h-[132px] flex items-center justify-center gap-4 rounded-3xl border-4 p-6 font-black text-center ${
                feedback.correct
                  ? 'border-emerald-300 bg-emerald-100 text-emerald-950'
                  : 'border-rose-300 bg-rose-100 text-rose-950'
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
                className={`min-h-[108px] sm:min-h-[132px] rounded-3xl border-4 p-5 sm:p-7 flex items-center justify-center transition ${
                  questionPhase === 'READING_QUESTION'
                    ? 'border-slate-200 bg-slate-100 opacity-60 cursor-not-allowed'
                    : pendingChoice === 'no'
                      ? 'border-rose-500 bg-rose-100 ring-4 ring-rose-300 shadow-lg active:scale-95 cursor-pointer'
                      : 'border-rose-300 bg-rose-50 hover:bg-rose-100 active:scale-95 cursor-pointer'
                }`}
              >
                <p className="mg-answer-btn font-black text-rose-700">
                  {t('minigameLeftNo')}
                </p>
              </button>
              <button
                type="button"
                disabled={questionPhase === 'READING_QUESTION'}
                onClick={() => submitAnswer('yes')}
                className={`min-h-[108px] sm:min-h-[132px] rounded-3xl border-4 p-5 sm:p-7 flex items-center justify-center transition ${
                  questionPhase === 'READING_QUESTION'
                    ? 'border-slate-200 bg-slate-100 opacity-60 cursor-not-allowed'
                    : pendingChoice === 'yes'
                      ? 'border-emerald-500 bg-emerald-100 ring-4 ring-emerald-300 shadow-lg active:scale-95 cursor-pointer'
                      : 'border-emerald-300 bg-emerald-50 hover:bg-emerald-100 active:scale-95 cursor-pointer'
                }`}
              >
                <p className="mg-answer-btn font-black text-emerald-700">
                  {t('minigameRightYes')}
                </p>
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
