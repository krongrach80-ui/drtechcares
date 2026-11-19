export type VoiceLanguage = 'th' | 'en' | 'cn';

export const MIN_TTS_COOLDOWN_MS = 3500;

const LANG_BCP47: Record<VoiceLanguage, string> = {
  th: 'th-TH',
  en: 'en-US',
  cn: 'zh-CN',
};

/**
 * Output-only TTS Voice Feedback Controller (No microphone / SpeechRecognition).
 * Enforces >= 3,500ms cooldown to prevent overlapping hospital kiosk audio.
 */
export class VoiceFeedbackController {
  private lastSpokenAtMs = -Infinity;
  private lastPhrase = '';

  constructor(private readonly cooldownMs: number = MIN_TTS_COOLDOWN_MS) {}

  speak(text: string, lang: VoiceLanguage, force = false, nowMs: number = Date.now()): boolean {
    if (!text.trim()) return false;
    if (!force && nowMs - this.lastSpokenAtMs < this.cooldownMs) {
      return false;
    }
    if (!force && text === this.lastPhrase && nowMs - this.lastSpokenAtMs < this.cooldownMs * 2) {
      return false;
    }

    this.lastSpokenAtMs = nowMs;
    this.lastPhrase = text;

    if (typeof globalThis.window !== 'undefined' && 'speechSynthesis' in globalThis.window) {
      try {
        globalThis.window.speechSynthesis.cancel();
        const utter = new SpeechSynthesisUtterance(text);
        utter.lang = LANG_BCP47[lang];
        utter.rate = 0.96;
        globalThis.window.speechSynthesis.speak(utter);
      } catch {
        // Ignore TTS failure in headless/sandbox environments
      }
    }
    return true;
  }

  flush(): void {
    this.lastSpokenAtMs = -Infinity;
    this.lastPhrase = '';
    if (typeof globalThis.window !== 'undefined' && 'speechSynthesis' in globalThis.window) {
      try {
        globalThis.window.speechSynthesis.cancel();
      } catch {
        // Ignore
      }
    }
  }
}

/**
 * Dependency-free Web Audio API Chime Synthesizer (no external MP3 downloads).
 */
export class AudioChimeSynthesizer {
  private ctx: AudioContext | null = null;

  private getContext(): AudioContext | null {
    if (typeof globalThis.window === 'undefined') return null;
    const AudioCtx =
      globalThis.window.AudioContext ||
      (globalThis.window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!AudioCtx) return null;
    if (!this.ctx) {
      this.ctx = new AudioCtx();
    }
    return this.ctx;
  }

  playTone(type: 'confirm' | 'rep' | 'correct' | 'wrong' | 'alert'): void {
    const ctx = this.getContext();
    if (!ctx) return;
    try {
      if (ctx.state === 'suspended') {
        void ctx.resume().catch(() => {});
      }
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      const now = ctx.currentTime;

      const freqMap: Record<typeof type, number> = {
        confirm: 660,
        rep: 880,
        correct: 987.77,
        wrong: 260,
        alert: 440,
      };
      osc.type = type === 'wrong' ? 'triangle' : 'sine';
      osc.frequency.setValueAtTime(freqMap[type], now);
      gain.gain.setValueAtTime(0.08, now);
      gain.gain.exponentialRampToValueAtTime(0.001, now + 0.22);
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start(now);
      osc.stop(now + 0.23);
    } catch {
      // Ignore AudioContext restriction errors
    }
  }
}

