/**
 * Lightweight, zero-dependency Web Audio synthesizer for playful clinical-game
 * micro-interaction feedback (tactile taps, rep chimes, ascending combo arpeggios,
 * unlock sparkles, and set/game victory fanfares).
 *
 * - Strictly SSR/Node-safe and exception-safe.
 * - Automatically respects the global Kiosk mute flag (`drtechcares_voice_muted_v1`)
 *   and optional per-call mute override.
 * - Uses pure synthesized oscillators (zero network fetches, zero audio files).
 */

export type GameUiSfxKind =
  | 'tap'
  | 'rep'
  | 'combo'
  | 'correct'
  | 'wrong'
  | 'unlock'
  | 'victory';

export interface PlayGameSfxOptions {
  /** Optional combo streak count (`>= 1`) used to pitch-shift combo chimes upwards. */
  comboCount?: number;
  /** Explicitly mute this playback regardless of stored mute preference. */
  muted?: boolean;
}

const VOICE_MUTE_STORAGE_KEY = 'drtechcares_voice_muted_v1';
const MIN_TAP_INTERVAL_MS = 48;

let sharedAudioCtx: AudioContext | null = null;
let lastTapAtMs = 0;

export function isGameSfxMuted(explicitMuted?: boolean): boolean {
  if (explicitMuted) return true;
  if (typeof window === 'undefined') return true;
  try {
    return window.localStorage.getItem(VOICE_MUTE_STORAGE_KEY) === '1';
  } catch {
    return false;
  }
}

function getAudioContext(): AudioContext | null {
  if (typeof window === 'undefined') return null;
  const AudioCtxCtor =
    window.AudioContext ||
    (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
  if (!AudioCtxCtor) return null;
  try {
    if (!sharedAudioCtx || sharedAudioCtx.state === 'closed') {
      sharedAudioCtx = new AudioCtxCtor();
    }
    if (sharedAudioCtx.state === 'suspended') {
      void sharedAudioCtx.resume().catch(() => {});
    }
    return sharedAudioCtx;
  } catch {
    return null;
  }
}

function scheduleTone(
  ctx: AudioContext,
  opts: {
    freq: number;
    endFreq?: number;
    type?: OscillatorType;
    startOffsetSec: number;
    durationSec: number;
    peakGain: number;
  },
): void {
  const now = ctx.currentTime + opts.startOffsetSec;
  const osc = ctx.createOscillator();
  const gain = ctx.createGain();
  osc.type = opts.type ?? 'sine';
  osc.frequency.setValueAtTime(opts.freq, now);
  if (opts.endFreq && opts.endFreq > 0) {
    osc.frequency.exponentialRampToValueAtTime(opts.endFreq, now + opts.durationSec);
  }
  gain.gain.setValueAtTime(0.0001, now);
  gain.gain.linearRampToValueAtTime(opts.peakGain, now + Math.min(0.012, opts.durationSec * 0.3));
  gain.gain.exponentialRampToValueAtTime(0.0001, now + opts.durationSec);

  osc.connect(gain);
  gain.connect(ctx.destination);
  osc.start(now);
  osc.stop(now + opts.durationSec + 0.015);
}

/**
 * Plays a gentle, crisp synthesized UI sound effect. Returns `true` if audio was
 * scheduled, or `false` when muted / throttled / running in a headless environment.
 */
export function playGameSfx(
  kind: GameUiSfxKind,
  options: PlayGameSfxOptions = {},
): boolean {
  if (isGameSfxMuted(options.muted)) return false;

  if (kind === 'tap') {
    const nowMs = typeof performance !== 'undefined' ? performance.now() : Date.now();
    if (nowMs - lastTapAtMs < MIN_TAP_INTERVAL_MS) {
      return false;
    }
    lastTapAtMs = nowMs;
  }

  const ctx = getAudioContext();
  if (!ctx) return false;

  try {
    switch (kind) {
      case 'tap': {
        scheduleTone(ctx, {
          freq: 520,
          endFreq: 760,
          type: 'triangle',
          startOffsetSec: 0,
          durationSec: 0.042,
          peakGain: 0.036,
        });
        return true;
      }
      case 'rep': {
        scheduleTone(ctx, {
          freq: 523.25,
          type: 'triangle',
          startOffsetSec: 0,
          durationSec: 0.12,
          peakGain: 0.08,
        });
        scheduleTone(ctx, {
          freq: 783.99,
          type: 'sine',
          startOffsetSec: 0.075,
          durationSec: 0.2,
          peakGain: 0.09,
        });
        return true;
      }
      case 'combo': {
        const streak = Math.max(1, Math.min(8, Math.round(options.comboCount ?? 2)));
        const pentatonicSemitones = [0, 2, 4, 7, 9, 12, 14, 16];
        const baseShift = pentatonicSemitones[streak - 1] ?? 0;
        const rootHz = 523.25 * Math.pow(2, baseShift / 12);
        scheduleTone(ctx, {
          freq: rootHz,
          type: 'triangle',
          startOffsetSec: 0,
          durationSec: 0.11,
          peakGain: 0.075,
        });
        scheduleTone(ctx, {
          freq: rootHz * 1.25,
          type: 'sine',
          startOffsetSec: 0.06,
          durationSec: 0.14,
          peakGain: 0.08,
        });
        scheduleTone(ctx, {
          freq: rootHz * 1.5,
          type: 'sine',
          startOffsetSec: 0.12,
          durationSec: 0.22,
          peakGain: 0.085,
        });
        return true;
      }
      case 'correct': {
        scheduleTone(ctx, {
          freq: 659.25,
          type: 'triangle',
          startOffsetSec: 0,
          durationSec: 0.12,
          peakGain: 0.08,
        });
        scheduleTone(ctx, {
          freq: 987.77,
          type: 'sine',
          startOffsetSec: 0.08,
          durationSec: 0.22,
          peakGain: 0.085,
        });
        return true;
      }
      case 'wrong': {
        scheduleTone(ctx, {
          freq: 293.66,
          endFreq: 220,
          type: 'sine',
          startOffsetSec: 0,
          durationSec: 0.18,
          peakGain: 0.065,
        });
        return true;
      }
      case 'unlock': {
        scheduleTone(ctx, {
          freq: 587.33,
          type: 'sine',
          startOffsetSec: 0,
          durationSec: 0.1,
          peakGain: 0.07,
        });
        scheduleTone(ctx, {
          freq: 880,
          type: 'triangle',
          startOffsetSec: 0.07,
          durationSec: 0.14,
          peakGain: 0.075,
        });
        scheduleTone(ctx, {
          freq: 1174.66,
          type: 'sine',
          startOffsetSec: 0.14,
          durationSec: 0.24,
          peakGain: 0.08,
        });
        return true;
      }
      case 'victory': {
        const notes = [523.25, 659.25, 783.99, 1046.5];
        notes.forEach((hz, idx) => {
          scheduleTone(ctx, {
            freq: hz,
            type: idx === notes.length - 1 ? 'triangle' : 'sine',
            startOffsetSec: idx * 0.085,
            durationSec: idx === notes.length - 1 ? 0.36 : 0.15,
            peakGain: idx === notes.length - 1 ? 0.095 : 0.075,
          });
        });
        return true;
      }
      default:
        return false;
    }
  } catch {
    return false;
  }
}

export function resetGameSfxStateForTests(): void {
  lastTapAtMs = 0;
  sharedAudioCtx = null;
}

