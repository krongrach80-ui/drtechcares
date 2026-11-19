export type MascotSpeechPose =
  | 'idle'
  | 'greeting'
  | 'explaining'
  | 'speaking'
  | 'encouraging';

export type MascotSpeechSource =
  | 'voice_coach'
  | 'kiosk_login'
  | 'kiosk_overview'
  | 'face_enroll'
  | 'card_game'
  | 'portal'
  | 'other';

export interface GlobalMascotSpeechState {
  isSpeaking: boolean;
  text: string;
  pose: MascotSpeechPose;
  source: MascotSpeechSource | null;
  updatedAt: number;
}

export const MASCOT_SPEECH_EVENT = 'drtech-mascot-speech-change';
const MASCOT_SPEECH_WATCHDOG_MS = 15000;

const EMOJI_STRIP_RE = /[\p{Extended_Pictographic}\u{FE0F}\u{2600}-\u{27BF}\u{23E9}-\u{23FA}]/gu;

function cleanSpeechTextInternal(rawText: string): string {
  return rawText.replace(EMOJI_STRIP_RE, '').replace(/\s+/g, ' ').trim();
}

const ENCOURAGING_RE =
  /เก่งมาก|เยี่ยมมาก|ยอดเยี่ยม|ดีมาก|สู้ๆ|ถูกต้อง|ยินดีด้วย|ครบแล้ว|ครบเซ็ต|จบเซ็ต|สำเร็จ|เรียบร้อยแล้ว|ได้ \+1|คะแนน|excellent|great job|congratulations|well done|correct|completed|finished|太棒了|非常好|恭喜|答对了|已完成|成功/i;

const GREETING_RE =
  /สวัสดี|ยินดีต้อนรับ|พร้อมเริ่ม|พบกันใหม่|วันพักผ่อน|hello|welcome|good morning|see you|rest day|ready to begin|您好|欢迎|你好|休息日|准备就绪/i;

const EXPLAINING_RE =
  /กรุณา|องศา|ค้าง|วินาที|หันหน้า|มองตรง|ระยะ|ถอยหลัง|เดินเข้า|ยืดตัว|ลำตัว|หัวไหล่|เข่า|ส้นเท้า|สะโพก|ท่านี้|ชูมือ|ยกมือ|คำถาม|ข้อที่|ตารางนัด|เป้าหมาย|ห้าม|please|degrees|hold|step|turn|look straight|question|posture|target|schedule|请|保持|度|问题|姿势|目标|抬头|低头|向左|向右/i;

/**
 * Classifies spoken coaching/navigation text into the matching trimmed mascot animation pose.
 */
export function inferMascotPoseFromSpeechText(
  rawText: string,
  hintPose?: MascotSpeechPose
): MascotSpeechPose {
  const clean = cleanSpeechTextInternal(rawText);
  if (hintPose && hintPose !== 'idle' && clean.length > 0) {
    return hintPose;
  }
  if (!clean) {
    return hintPose ?? 'idle';
  }
  const lower = clean.toLowerCase();
  if (ENCOURAGING_RE.test(lower)) {
    return 'encouraging';
  }
  if (GREETING_RE.test(lower)) {
    return 'greeting';
  }
  if (EXPLAINING_RE.test(lower)) {
    return 'explaining';
  }
  return 'speaking';
}

let currentState: GlobalMascotSpeechState = {
  isSpeaking: false,
  text: '',
  pose: 'idle',
  source: null,
  updatedAt: 0,
};

let watchdogTimer: ReturnType<typeof setTimeout> | null = null;
const listeners = new Set<(state: GlobalMascotSpeechState) => void>();

function clearSpeechWatchdog(): void {
  if (watchdogTimer !== null) {
    clearTimeout(watchdogTimer);
    watchdogTimer = null;
  }
}

function emitMascotSpeechState(nextState: GlobalMascotSpeechState): void {
  currentState = nextState;
  listeners.forEach((listener) => {
    try {
      listener(nextState);
    } catch {
      // Ignore individual subscriber errors
    }
  });
  if (typeof window !== 'undefined') {
    try {
      window.dispatchEvent(new CustomEvent(MASCOT_SPEECH_EVENT, { detail: nextState }));
    } catch {
      // Ignore CustomEvent failures in restricted runtimes
    }
  }
}

/**
 * Marks global speech active with auto-classified or hinted mascot pose and safety watchdog.
 */
export function notifyMascotSpeechStart(payload: {
  text: string;
  source: MascotSpeechSource;
  pose?: MascotSpeechPose;
}): GlobalMascotSpeechState {
  const cleanText = cleanSpeechTextInternal(payload.text);
  const resolvedPose = inferMascotPoseFromSpeechText(cleanText, payload.pose);
  const nextState: GlobalMascotSpeechState = {
    isSpeaking: true,
    text: cleanText,
    pose: resolvedPose,
    source: payload.source,
    updatedAt: Date.now(),
  };

  clearSpeechWatchdog();
  watchdogTimer = setTimeout(() => {
    watchdogTimer = null;
    if (currentState.isSpeaking && currentState.source === payload.source) {
      emitMascotSpeechState({
        ...currentState,
        isSpeaking: false,
        pose: 'idle',
        updatedAt: Date.now(),
      });
    }
  }, MASCOT_SPEECH_WATCHDOG_MS);

  emitMascotSpeechState(nextState);
  return nextState;
}

/**
 * Marks speech finished; if `source` is provided, ignores calls from other inactive sources.
 */
export function notifyMascotSpeechEnd(source?: MascotSpeechSource): GlobalMascotSpeechState {
  if (source !== undefined && currentState.source !== null && currentState.source !== source) {
    return currentState;
  }
  clearSpeechWatchdog();
  if (!currentState.isSpeaking && currentState.pose === 'idle') {
    return currentState;
  }
  const nextState: GlobalMascotSpeechState = {
    ...currentState,
    isSpeaking: false,
    pose: 'idle',
    updatedAt: Date.now(),
  };
  emitMascotSpeechState(nextState);
  return nextState;
}

export function getGlobalMascotSpeechState(): GlobalMascotSpeechState {
  return currentState;
}

export function subscribeGlobalMascotSpeech(
  listener: (state: GlobalMascotSpeechState) => void
): () => void {
  listeners.add(listener);
  const handleWindowEvent = (ev: Event) => {
    const custom = ev as CustomEvent<GlobalMascotSpeechState>;
    if (custom?.detail && !listeners.has(listener)) return;
  };
  if (typeof window !== 'undefined') {
    window.addEventListener(MASCOT_SPEECH_EVENT, handleWindowEvent);
  }
  return () => {
    listeners.delete(listener);
    if (typeof window !== 'undefined') {
      window.removeEventListener(MASCOT_SPEECH_EVENT, handleWindowEvent);
    }
  };
}

