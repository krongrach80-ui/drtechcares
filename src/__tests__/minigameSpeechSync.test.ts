import fs from 'node:fs';
import path from 'node:path';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  MINIGAME_HAND_RAISE_THRESHOLD_Y,
  MINIGAME_QUESTION_LOCK_PILL_TEXT,
  MINIGAME_QUESTION_READY_PILL_TEXT,
  MINIGAME_SPEECH_KEEPALIVE_INTERVAL_MS,
  MINIGAME_THAI_SPEECH_RATE,
  MinigameSpeechTurnDriver,
  buildMinigameSpeechTurnKey,
  computeMinigameSafeSpeechWatchdogMs,
  resolveMinigameAnswerStatusPill,
  type MinigameSpeechEnginePort,
} from '../../apps/kiosk/src/PhysicalCardGame';

const LONG_THAI_QUESTION = 'แว่นตาในภาพนี้ช่วยให้มองเห็นตัวหนังสือได้ชัดเจนขึ้นใช่ไหม?';
const UNICODE_EMOJI_RE = /[\p{Extended_Pictographic}\u{1F300}-\u{1FAFF}\u{2600}-\u{27BF}]/u;

interface MockUtterance {
  text: string;
  lang: string;
  rate: number;
  pitch: number;
  volume: number;
  onstart: ((ev: Event) => void) | null;
  onend: ((ev: Event) => void) | null;
  onerror: ((ev: Event) => void) | null;
}

function createSpeechHarness() {
  const spokenUtterances: MockUtterance[] = [];
  const synthState = {
    speaking: false,
    pending: false,
    paused: false,
  };
  const speakSpy = vi.fn((utterance: SpeechSynthesisUtterance) => {
    const mock = utterance as unknown as MockUtterance;
    spokenUtterances.push(mock);
    synthState.speaking = true;
    synthState.pending = false;
    mock.onstart?.({} as Event);
  });
  const cancelSpy = vi.fn(() => {
    synthState.speaking = false;
    synthState.pending = false;
    synthState.paused = false;
  });
  const resumeSpy = vi.fn(() => {
    synthState.paused = false;
  });

  const synth: MinigameSpeechEnginePort = {
    speak: speakSpy,
    cancel: cancelSpy,
    resume: resumeSpy,
    getVoices: () => [],
    get speaking() {
      return synthState.speaking;
    },
    get pending() {
      return synthState.pending;
    },
    get paused() {
      return synthState.paused;
    },
  };

  const stateHistory: Array<{
    isQuestionSpeaking: boolean;
    phase: string;
    statusText: string;
    hasActiveUtterance: boolean;
  }> = [];

  const driver = new MinigameSpeechTurnDriver({
    synth,
    createUtterance: (text: string) =>
      ({
        text,
        lang: '',
        rate: 1,
        pitch: 1,
        volume: 1,
        onstart: null,
        onend: null,
        onerror: null,
      }) as unknown as SpeechSynthesisUtterance,
    setTimer: (fn, ms) => Number(globalThis.setTimeout(fn, ms)),
    clearTimer: (id) => globalThis.clearTimeout(id),
    setIntervalFn: (fn, ms) => Number(globalThis.setInterval(fn, ms)),
    clearIntervalFn: (id) => globalThis.clearInterval(id),
    onStateChange: (state) => {
      stateHistory.push({
        isQuestionSpeaking: state.isQuestionSpeaking,
        phase: state.phase,
        statusText: state.statusText,
        hasActiveUtterance: state.activeUtterance !== null,
      });
    },
  });

  return {
    driver,
    synthState,
    spokenUtterances,
    speakSpy,
    cancelSpy,
    resumeSpy,
    stateHistory,
  };
}

describe('Minigame AI Voice Speech Sync & Full-Sentence Completion', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('computes a generous dynamic watchdog duration so long Thai sentences are never cut off early', () => {
    const longMs = computeMinigameSafeSpeechWatchdogMs(LONG_THAI_QUESTION);
    expect(longMs).toBe(Math.max(8000, LONG_THAI_QUESTION.trim().length * 180 + 3000));
    expect(longMs).toBeGreaterThanOrEqual(13000);

    const shortMs = computeMinigameSafeSpeechWatchdogMs('ใช่ไหม?');
    expect(shortMs).toBe(8000);
  });

  it('retains activeUtterance, ignores 60FPS hand-tracking updates, keeps Chromium alive, and unlocks only after utterance.onend', () => {
    const { driver, synthState, spokenUtterances, speakSpy, cancelSpy, resumeSpy } = createSpeechHarness();

    const started = driver.speakQuestionForTurn({
      sessionKey: 1,
      cardIndex: 0,
      cardId: 'card-glasses',
      voiceLang: 'th',
      text: LONG_THAI_QUESTION,
    });

    expect(started).toBe(true);
    expect(speakSpy).toHaveBeenCalledTimes(1);
    expect(cancelSpy).toHaveBeenCalledTimes(0);
    expect(driver.activeUtterance).not.toBeNull();
    expect(driver.isQuestionSpeaking).toBe(true);

    // Status pill stays locked on listening prompt even if bothHandsWarn flickers while reading
    expect(
      resolveMinigameAnswerStatusPill({ bothHandsWarn: false, isQuestionSpeaking: driver.isQuestionSpeaking }),
    ).toEqual({
      text: MINIGAME_QUESTION_LOCK_PILL_TEXT,
      variant: 'speaking',
    });
    expect(
      resolveMinigameAnswerStatusPill({ bothHandsWarn: true, isQuestionSpeaking: driver.isQuestionSpeaking }),
    ).toEqual({
      text: MINIGAME_QUESTION_LOCK_PILL_TEXT,
      variant: 'speaking',
    });

    // Simulate 120 frames (2s at 60FPS) of hand-tracking updates with the same card/language turn key
    for (let frame = 0; frame < 120; frame += 1) {
      const restarted = driver.speakQuestionForTurn({
        sessionKey: 1,
        cardIndex: 0,
        cardId: 'card-glasses',
        voiceLang: 'th',
        text: LONG_THAI_QUESTION,
      });
      expect(restarted).toBe(false);
    }
    expect(speakSpy).toHaveBeenCalledTimes(1);
    expect(cancelSpy).toHaveBeenCalledTimes(0);
    expect(driver.isQuestionSpeaking).toBe(true);

    // Simulate Chromium pausing long Thai utterances after 3000ms and running well past the old 4.7s cutoff
    synthState.paused = true;
    vi.advanceTimersByTime(MINIGAME_SPEECH_KEEPALIVE_INTERVAL_MS);
    expect(resumeSpy).toHaveBeenCalledTimes(1);
    expect(synthState.paused).toBe(false);

    vi.advanceTimersByTime(5000); // 8000ms total elapsed (< 13.6s safe watchdog)
    expect(cancelSpy).toHaveBeenCalledTimes(0);
    expect(driver.isQuestionSpeaking).toBe(true);
    expect(driver.activeUtterance).not.toBeNull();

    // Utterance completes naturally via onend
    const active = spokenUtterances[0];
    expect(active.text).toBe(LONG_THAI_QUESTION);
    expect(active.rate).toBe(MINIGAME_THAI_SPEECH_RATE);
    synthState.speaking = false;
    active.onend?.({} as Event);

    expect(driver.activeUtterance).toBeNull();
    expect(driver.isQuestionSpeaking).toBe(false);
    expect(cancelSpy).toHaveBeenCalledTimes(0);
    expect(
      resolveMinigameAnswerStatusPill({ bothHandsWarn: false, isQuestionSpeaking: driver.isQuestionSpeaking }),
    ).toEqual({
      text: MINIGAME_QUESTION_READY_PILL_TEXT,
      variant: 'ready',
    });

    driver.dispose();
  });

  it('unlocks answering safely after generous dynamic watchdog without ever calling speechSynthesis.cancel()', () => {
    const { driver, cancelSpy } = createSpeechHarness();

    driver.speakQuestionForTurn({
      sessionKey: 1,
      cardIndex: 2,
      cardId: 'card-glasses',
      voiceLang: 'th',
      text: LONG_THAI_QUESTION,
    });

    const safeWatchdogMs = computeMinigameSafeSpeechWatchdogMs(LONG_THAI_QUESTION);
    vi.advanceTimersByTime(safeWatchdogMs - 100);
    expect(driver.isQuestionSpeaking).toBe(true);
    expect(cancelSpy).toHaveBeenCalledTimes(0);

    vi.advanceTimersByTime(200);
    expect(driver.isQuestionSpeaking).toBe(false);
    // Critical invariant: watchdog ONLY unlocks UI answering and NEVER cancels speechSynthesis
    expect(cancelSpy).toHaveBeenCalledTimes(0);

    driver.dispose();
  });

  it('handles replay ("พูดซ้ำ"), language change, and mute toggle ("เสียง AI: ปิด") cleanly', () => {
    const { driver, spokenUtterances, speakSpy, cancelSpy } = createSpeechHarness();

    driver.speakQuestionForTurn({
      sessionKey: 1,
      cardIndex: 0,
      cardId: 'card-glasses',
      voiceLang: 'th',
      text: LONG_THAI_QUESTION,
    });
    expect(speakSpy).toHaveBeenCalledTimes(1);

    // Explicit replay cancels prior utterance and re-locks until the new onend fires
    const replayed = driver.replayQuestion(LONG_THAI_QUESTION, 'th');
    expect(replayed).toBe(true);
    expect(cancelSpy).toHaveBeenCalledTimes(1);
    expect(speakSpy).toHaveBeenCalledTimes(2);
    expect(driver.isQuestionSpeaking).toBe(true);

    // Finish replay naturally
    spokenUtterances[1].onend?.({} as Event);
    expect(driver.isQuestionSpeaking).toBe(false);

    // Switching language creates a distinct turn key and reads the full localized question
    const enTurnKey = buildMinigameSpeechTurnKey(1, 0, 'card-glasses', 'en');
    expect(enTurnKey).not.toBe(buildMinigameSpeechTurnKey(1, 0, 'card-glasses', 'th'));
    const switched = driver.speakQuestionForTurn({
      sessionKey: 1,
      cardIndex: 0,
      cardId: 'card-glasses',
      voiceLang: 'en',
      text: 'Do these eyeglasses help you see letters more clearly?',
    });
    expect(switched).toBe(true);
    expect(driver.isQuestionSpeaking).toBe(true);

    // Toggling mute ON cancels immediately and unlocks answering so muted mode never blocks gameplay
    driver.setMuted(true);
    expect(cancelSpy).toHaveBeenCalledTimes(2);
    expect(driver.isQuestionSpeaking).toBe(false);
    expect(driver.activeUtterance).toBeNull();

    // Toggling mute OFF re-locks and reads the active question cleanly until onend
    driver.setMuted(false, LONG_THAI_QUESTION, 'th');
    expect(driver.isQuestionSpeaking).toBe(true);
    expect(speakSpy).toHaveBeenCalledTimes(4);
    spokenUtterances[3].onend?.({} as Event);
    expect(driver.isQuestionSpeaking).toBe(false);

    driver.dispose();
  });

  it('preserves strict guardrails in PhysicalCardGame.tsx and VoiceCoachBanner.tsx (y < 0.58 untouched, zero emojis, safe watchdog)', () => {
    const rootDir = process.cwd();
    const physicalCardPath = path.join(rootDir, 'apps', 'kiosk', 'src', 'PhysicalCardGame.tsx');
    const voiceCoachPath = path.join(rootDir, 'src', 'components', 'VoiceCoachBanner.tsx');
    const selfTestPath = path.join(rootDir, 'src', '__tests__', 'minigameSpeechSync.test.ts');

    const physicalCardSource = fs.readFileSync(physicalCardPath, 'utf8');
    const voiceCoachSource = fs.readFileSync(voiceCoachPath, 'utf8');
    const selfTestSource = fs.readFileSync(selfTestPath, 'utf8');

    // Guardrail: hand raise threshold y < 0.58 and Thai speech rate preserved
    expect(MINIGAME_HAND_RAISE_THRESHOLD_Y).toBe(0.58);
    expect(MINIGAME_THAI_SPEECH_RATE).toBe(1.18);
    expect(MINIGAME_SPEECH_KEEPALIVE_INTERVAL_MS).toBe(3000);

    // Source-level invariants preventing mid-sentence cutoff regressions
    expect(physicalCardSource).toContain('spokenTurnKeyRef');
    expect(physicalCardSource).toContain('speechKeepAliveRef');
    expect(physicalCardSource).toContain('computeMinigameSafeSpeechWatchdogMs');
    expect(physicalCardSource).toContain('resolveMinigameAnswerStatusPill');
    expect(physicalCardSource).toContain('ฟังจบแล้ว! ยกมือค้าง 0.8 วินาทีเพื่อตอบ');
    expect(physicalCardSource).toContain('!isSpeechLocked &&');
    expect(physicalCardSource).toContain('!isSpeakingRef.current &&');

    // Guardrail: ZERO Unicode emojis in modified speech modules or test file
    expect(UNICODE_EMOJI_RE.test(physicalCardSource)).toBe(false);
    expect(UNICODE_EMOJI_RE.test(voiceCoachSource)).toBe(false);
    expect(UNICODE_EMOJI_RE.test(selfTestSource)).toBe(false);
  });
});

