export const KIOSK_VOICE_MUTED_STORAGE_KEY = 'drtech_kiosk_voice_muted';
export const KIOSK_VOICE_MUTED_EVENT = 'drtech-kiosk-voice-muted-change';

/** Reads persisted kiosk voice mute state across screens and reloads. */
export function readStoredKioskVoiceMuted(): boolean {
  if (typeof window === 'undefined') return false;
  try {
    return window.localStorage.getItem(KIOSK_VOICE_MUTED_STORAGE_KEY) === 'true';
  } catch {
    return false;
  }
}

/** Persists kiosk voice mute state and notifies mounted kiosk components immediately. */
export function writeStoredKioskVoiceMuted(muted: boolean): boolean {
  const next = Boolean(muted);
  if (typeof window !== 'undefined') {
    try {
      window.localStorage.setItem(KIOSK_VOICE_MUTED_STORAGE_KEY, next ? 'true' : 'false');
    } catch {
      // Ignore storage access errors in private browsing/sandboxes
    }
    try {
      window.dispatchEvent(new CustomEvent(KIOSK_VOICE_MUTED_EVENT, { detail: { muted: next } }));
    } catch {
      // Ignore custom event dispatch failures
    }
  }
  return next;
}

/** Subscribes to same-tab and cross-tab changes to the global kiosk voice mute flag. */
export function subscribeKioskVoiceMuted(onChange: (muted: boolean) => void): () => void {
  if (typeof window === 'undefined') return () => {};
  const handleCustom = (ev: Event) => {
    const customEv = ev as CustomEvent<{ muted?: boolean }>;
    if (typeof customEv.detail?.muted === 'boolean') {
      onChange(customEv.detail.muted);
    } else {
      onChange(readStoredKioskVoiceMuted());
    }
  };
  const handleStorage = (ev: StorageEvent) => {
    if (ev.key === KIOSK_VOICE_MUTED_STORAGE_KEY) {
      onChange(ev.newValue === 'true');
    }
  };
  window.addEventListener(KIOSK_VOICE_MUTED_EVENT, handleCustom);
  window.addEventListener('storage', handleStorage);
  return () => {
    window.removeEventListener(KIOSK_VOICE_MUTED_EVENT, handleCustom);
    window.removeEventListener('storage', handleStorage);
  };
}

/** Immediately halts Web Speech API utterances and any playing HTML5 audio tracks. */
export function cancelAllKioskVoiceSpeech(extraAudio?: HTMLAudioElement | null): void {
  if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
    try {
      window.speechSynthesis.cancel();
    } catch {
      // Ignore browser speech synthesis cancellation errors
    }
  }
  if (extraAudio) {
    try {
      extraAudio.pause();
      extraAudio.currentTime = 0;
    } catch {
      // Ignore audio pause errors
    }
  }
  if (typeof document !== 'undefined') {
    try {
      const audioEls = document.querySelectorAll('audio');
      audioEls.forEach((audioEl) => {
        if (!audioEl.paused) {
          audioEl.pause();
          audioEl.currentTime = 0;
        }
      });
    } catch {
      // Ignore DOM audio lookup failures
    }
  }
}

export interface KioskVoiceMuteCopy {
  unmutedLabel: string;
  mutedLabel: string;
}

/** Strip Unicode emojis, variation selectors, and symbol dingbats before speech synthesis or UI display. */
export function sanitizeSpeechText(rawText: string): string {
  return rawText
    .replace(/[\p{Extended_Pictographic}\u{FE0F}\u{2600}-\u{27BF}\u{23E9}-\u{23FA}]/gu, '')
    .replace(/\s+/g, ' ')
    .trim();
}

/** Accessible high-contrast labels for senior patients across TH / EN / ZH. */
export function getSeniorVoiceToggleLabel(lang: 'th' | 'en' | 'zh' | string, muted: boolean): string {
  const norm = lang.trim().toLowerCase();
  if (norm.startsWith('en')) {
    return muted ? 'AI Voice Muted (Tap to Unmute)' : 'AI Voice On (Tap to Mute)';
  }
  if (norm.startsWith('zh') || norm === 'cn') {
    return muted ? 'AI 语音已静音 (点击开启)' : 'AI 语音播报中 (点击静音)';
  }
  return muted ? 'ปิดเสียงพากย์ AI แล้ว (กดเพื่อเปิดเสียง)' : 'เปิดเสียงพากย์ AI อยู่ (กดเพื่อปิดเสียง)';
}

