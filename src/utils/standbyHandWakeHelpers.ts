import type { ClinicalLang } from '@drtechcares/clinical-i18n';
import { readStoredKioskVoiceMuted } from '@drtechcares/kiosk-input';
import { buildTtsProxyUrl, pickSpeechVoice, type VoiceCoachLanguage } from '../components/VoiceCoachBanner';

export const STANDBY_HAND_WAKE_HOLD_MS = 3000;
export const STANDBY_HAND_GRACE_MS = 340;
export const STANDBY_INTRO_MIN_MS = 4300;
export const STANDBY_INTRO_MAX_MS = 7400;
export const STANDBY_INTRO_EXIT_MS = 420;

export interface PoseLandmarkLike {
  x: number;
  y: number;
  z?: number;
  visibility?: number;
}

export interface TwoHandsRaiseSample {
  personDetected: boolean;
  leftRaised: boolean;
  rightRaised: boolean;
  bothRaised: boolean;
  raisedCount: 0 | 1 | 2;
}

export interface StandbyHandHoldState {
  holdMs: number;
  missingMs: number;
  progress: number;
  remainingSec: number;
  completed: boolean;
}

export const INITIAL_STANDBY_HAND_HOLD_STATE: StandbyHandHoldState = {
  holdMs: 0,
  missingMs: 0,
  progress: 0,
  remainingSec: 3,
  completed: false,
};

export interface StandbyWelcomeCopy {
  headline: string;
  subline: string;
  spokenText: string;
  fullDisplay: string;
  badgeLabel: string;
  skipLabel: string;
}

export const STANDBY_WELCOME_COPY: Record<ClinicalLang, StandbyWelcomeCopy> = {
  th: {
    headline: 'ยินดีต้อนรับสู่ Dr.Tech.Care นะคะ',
    subline: 'ขอให้ทุกท่านได้ใช้เราอย่างมีความสุขค่ะ',
    spokenText: 'ยินดีต้อนรับสู่ ด็อกเตอร์เทคแคร์ นะคะ ขอให้ทุกท่านได้ใช้เราอย่างมีความสุขค่ะ',
    fullDisplay: 'ยินดีต้อนรับสู่ Dr.Tech.Care นะคะ ขอให้ทุกท่านได้ใช้เราอย่างมีความสุขค่ะ',
    badgeLabel: 'AI REHABILITATION KIOSK',
    skipLabel: 'เข้าสู่ระบบสแกนใบหน้าทันที',
  },
  en: {
    headline: 'Welcome to Dr.Tech.Care',
    subline: 'Wishing everyone a joyful and healthy experience with us.',
    spokenText: 'Welcome to Doctor Tech Care. Wishing everyone a joyful and healthy experience with us.',
    fullDisplay: 'Welcome to Dr.Tech.Care — Wishing everyone a joyful and healthy experience with us.',
    badgeLabel: 'AI REHABILITATION KIOSK',
    skipLabel: 'Continue to Face Scan',
  },
  cn: {
    headline: '欢迎来到 Dr.Tech.Care',
    subline: '祝愿大家在这里享受愉快健康的康复体验',
    spokenText: '欢迎来到 Dr.Tech.Care，祝愿大家在这里享受愉快健康的康复体验。',
    fullDisplay: '欢迎来到 Dr.Tech.Care，祝愿大家在这里享受愉快健康的康复体验',
    badgeLabel: 'AI REHABILITATION KIOSK',
    skipLabel: '立即进入人脸识别',
  },
};

const POSE_LEFT_SHOULDER = 11;
const POSE_RIGHT_SHOULDER = 12;
const POSE_LEFT_ELBOW = 13;
const POSE_RIGHT_ELBOW = 14;
const POSE_LEFT_WRIST = 15;
const POSE_RIGHT_WRIST = 16;

function isLandmarkVisible(lm: PoseLandmarkLike | undefined, minVis = 0.3): lm is PoseLandmarkLike {
  if (!lm) return false;
  if (!Number.isFinite(lm.x) || !Number.isFinite(lm.y)) return false;
  return (lm.visibility ?? 1) >= minVis;
}

function isArmRaised(
  shoulder: PoseLandmarkLike | undefined,
  elbow: PoseLandmarkLike | undefined,
  wrist: PoseLandmarkLike | undefined,
  shoulderSpan: number,
  wasRaised: boolean,
): boolean {
  if (!isLandmarkVisible(shoulder, 0.32)) return false;
  const slack = wasRaised ? 0.055 : 0;

  // Case 1: Wrist visible — either clearly above shoulder OR raised to shoulder/face height with forearm pointing upward.
  if (isLandmarkVisible(wrist, 0.28)) {
    const highAboveShoulder = wrist.y <= shoulder.y - 0.015 * shoulderSpan + slack;
    if (highAboveShoulder) return true;

    const forearmUp =
      isLandmarkVisible(elbow, 0.25) &&
      wrist.y < elbow.y - 0.025 + slack * 0.4 &&
      wrist.y <= shoulder.y + 0.22 * shoulderSpan + slack;
    if (forearmUp) return true;
  }

  // Case 2: Wrist momentarily clipped out of top of frame while elbow is raised above shoulder.
  if (isLandmarkVisible(elbow, 0.35) && elbow.y <= shoulder.y - 0.01 + slack) {
    return true;
  }

  return false;
}

/**
 * Evaluates whether the standing/seated person in front of the Kiosk has raised both hands.
 * Applies hysteresis via `prev` so natural minor tremor never causes progress jitter.
 */
export function evaluateTwoHandsRaised(
  landmarks: readonly PoseLandmarkLike[] | null | undefined,
  prev?: { leftRaised?: boolean; rightRaised?: boolean },
): TwoHandsRaiseSample {
  if (!landmarks || landmarks.length <= POSE_RIGHT_WRIST) {
    return {
      personDetected: false,
      leftRaised: false,
      rightRaised: false,
      bothRaised: false,
      raisedCount: 0,
    };
  }

  const ls = landmarks[POSE_LEFT_SHOULDER];
  const rs = landmarks[POSE_RIGHT_SHOULDER];
  const le = landmarks[POSE_LEFT_ELBOW];
  const re = landmarks[POSE_RIGHT_ELBOW];
  const lw = landmarks[POSE_LEFT_WRIST];
  const rw = landmarks[POSE_RIGHT_WRIST];

  const hasLeftShoulder = isLandmarkVisible(ls, 0.32);
  const hasRightShoulder = isLandmarkVisible(rs, 0.32);
  if (!hasLeftShoulder && !hasRightShoulder) {
    return {
      personDetected: false,
      leftRaised: false,
      rightRaised: false,
      bothRaised: false,
      raisedCount: 0,
    };
  }

  const shoulderSpan =
    hasLeftShoulder && hasRightShoulder
      ? Math.max(0.1, Math.hypot(ls.x - rs.x, ls.y - rs.y))
      : 0.22;

  const leftRaised = isArmRaised(ls, le, lw, shoulderSpan, Boolean(prev?.leftRaised));
  const rightRaised = isArmRaised(rs, re, rw, shoulderSpan, Boolean(prev?.rightRaised));
  const bothRaised = leftRaised && rightRaised;
  const raisedCount: 0 | 1 | 2 = bothRaised ? 2 : leftRaised || rightRaised ? 1 : 0;

  return {
    personDetected: true,
    leftRaised,
    rightRaised,
    bothRaised,
    raisedCount,
  };
}

/**
 * Advances the 3-second two-hand hold timer smoothly:
 * - Accumulates hold time monotonically while both hands are raised.
 * - Preserves progress during brief detection flicker (< STANDBY_HAND_GRACE_MS).
 * - Decays smoothly rather than resetting abruptly when hands are lowered.
 */
export function advanceStandbyHandHold(
  prev: StandbyHandHoldState,
  input: {
    bothRaised: boolean;
    dtMs: number;
    targetHoldMs?: number;
    graceMs?: number;
  },
): StandbyHandHoldState {
  const targetMs = Math.max(500, input.targetHoldMs ?? STANDBY_HAND_WAKE_HOLD_MS);
  const graceMs = Math.max(0, input.graceMs ?? STANDBY_HAND_GRACE_MS);
  const dt = Math.max(0, Math.min(250, Number.isFinite(input.dtMs) ? input.dtMs : 0));

  if (prev.completed) {
    return {
      holdMs: targetMs,
      missingMs: 0,
      progress: 1,
      remainingSec: 0,
      completed: true,
    };
  }

  let nextHoldMs = prev.holdMs;
  let nextMissingMs = prev.missingMs;

  if (input.bothRaised) {
    nextMissingMs = 0;
    nextHoldMs = Math.min(targetMs, prev.holdMs + dt);
  } else if (prev.holdMs > 0) {
    nextMissingMs = prev.missingMs + dt;
    if (nextMissingMs > graceMs) {
      const overflowDt = prev.missingMs >= graceMs ? dt : nextMissingMs - graceMs;
      nextHoldMs = Math.max(0, prev.holdMs - overflowDt * 1.45);
    }
  } else {
    nextMissingMs = 0;
    nextHoldMs = 0;
  }

  const progress = Math.max(0, Math.min(1, nextHoldMs / targetMs));
  const completed = nextHoldMs >= targetMs;
  const remainingSec = completed ? 0 : Math.max(1, Math.ceil((targetMs - nextHoldMs) / 1000));

  return {
    holdMs: nextHoldMs,
    missingMs: nextMissingMs,
    progress,
    remainingSec,
    completed,
  };
}

/**
 * Critically-damped exponential interpolation for liquid-smooth 60FPS progress rings.
 */
export function smoothVisualProgress(current: number, target: number, dtMs: number): number {
  const clampedTarget = Math.max(0, Math.min(1, target));
  const clampedCurrent = Math.max(0, Math.min(1, current));
  if (Math.abs(clampedTarget - clampedCurrent) < 0.001) {
    return clampedTarget;
  }
  const dtSec = Math.max(0, Math.min(0.1, dtMs / 1000));
  const factor = 1 - Math.exp(-dtSec * 14);
  return clampedCurrent + (clampedTarget - clampedCurrent) * factor;
}

function mapClinicalLangToVoiceLang(lang: ClinicalLang): VoiceCoachLanguage {
  if (lang === 'en') return 'en';
  if (lang === 'cn') return 'zh';
  return 'th';
}

function pickPreferredFemaleVoice(
  voices: readonly SpeechSynthesisVoice[],
  bcp47: string,
): SpeechSynthesisVoice | null {
  if (!voices.length) return null;
  const targetPrefix = bcp47.toLowerCase().split('-')[0];
  const langMatches = voices.filter((v) => v.lang.toLowerCase().startsWith(targetPrefix));
  if (langMatches.length > 0) {
    const femalePreferred = langMatches.find((v) => {
      const n = v.name.toLowerCase();
      return (
        n.includes('premwadee') ||
        n.includes('achara') ||
        n.includes('kanya') ||
        n.includes('female') ||
        n.includes('aria') ||
        n.includes('xiaoxiao') ||
        n.includes('google')
      );
    });
    if (femalePreferred) return femalePreferred;
  }
  return pickSpeechVoice(voices, bcp47);
}

/**
 * Speaks the Dr.Tech.Care cinematic welcome intro using same-origin neural `/api/tts`
 * with seamless fallback to Web Speech API (preferring natural Thai female voice).
 * Returns a cleanup stop handle.
 */
export function playStandbyWelcomeVoice(
  lang: ClinicalLang,
  callbacks?: { onEnd?: () => void },
): () => void {
  if (typeof window === 'undefined' || readStoredKioskVoiceMuted()) {
    return () => {};
  }

  const welcome = STANDBY_WELCOME_COPY[lang] ?? STANDBY_WELCOME_COPY.th;
  const voiceLang = mapClinicalLangToVoiceLang(lang);
  const bcp47 = voiceLang === 'th' ? 'th-TH' : voiceLang === 'zh' ? 'zh-CN' : 'en-US';
  let stopped = false;
  let endedFired = false;
  let audioEl: HTMLAudioElement | null = null;

  const notifyEndOnce = () => {
    if (stopped || endedFired) return;
    endedFired = true;
    callbacks?.onEnd?.();
  };

  const speakViaWebSpeech = () => {
    if (stopped || !('speechSynthesis' in window) || typeof SpeechSynthesisUtterance === 'undefined') {
      return;
    }
    try {
      const synth = window.speechSynthesis;
      synth.cancel();
      if (synth.paused) synth.resume();
      const utter = new SpeechSynthesisUtterance(welcome.spokenText);
      const voice = pickPreferredFemaleVoice(synth.getVoices(), bcp47);
      utter.lang = voice?.lang || bcp47;
      if (voice) utter.voice = voice;
      utter.rate = voiceLang === 'th' ? 0.94 : 0.98;
      utter.pitch = 1.04;
      utter.volume = 1.0;
      utter.onend = notifyEndOnce;
      utter.onerror = notifyEndOnce;
      synth.speak(utter);
    } catch {
      // Ignore Web Speech errors in restricted environments
    }
  };

  if (typeof Audio !== 'undefined') {
    try {
      audioEl = new Audio();
      audioEl.preload = 'auto';
      audioEl.playbackRate = voiceLang === 'th' ? 0.98 : 1.0;
      let fellBack = false;
      const fallbackOnce = () => {
        if (fellBack || stopped) return;
        fellBack = true;
        speakViaWebSpeech();
      };
      audioEl.onended = notifyEndOnce;
      audioEl.onerror = fallbackOnce;
      audioEl.src = buildTtsProxyUrl(welcome.spokenText, voiceLang);
      const playPromise = audioEl.play();
      if (playPromise && typeof playPromise.catch === 'function') {
        playPromise.catch(() => {
          fallbackOnce();
        });
      }
    } catch {
      speakViaWebSpeech();
    }
  } else {
    speakViaWebSpeech();
  }

  return () => {
    stopped = true;
    if (audioEl) {
      try {
        audioEl.onended = null;
        audioEl.onerror = null;
        audioEl.pause();
        audioEl.currentTime = 0;
      } catch {
        // Ignore cleanup errors
      }
      audioEl = null;
    }
    if ('speechSynthesis' in window) {
      try {
        window.speechSynthesis.cancel();
      } catch {
        // Ignore cleanup errors
      }
    }
  };
}

