import React, { useEffect, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import { Volume2, VolumeX } from 'lucide-react';
import type { HighlightColor, KioskState } from '../hooks/useKioskSocket';

interface VoiceCoachBannerProps {
  textTh: string;
  kioskState: KioskState;
  highlightColor: HighlightColor;
  muted: boolean;
  onToggleMute: () => void;
}

export function VoiceCoachBanner({
  textTh,
  kioskState,
  highlightColor,
  muted,
  onToggleMute,
}: VoiceCoachBannerProps) {
  const [isSpeaking, setIsSpeaking] = useState(false);
  const lastSpokenTextRef = useRef<string>('');
  const lastSpokenAtRef = useRef<number>(0);

  const speakNow = (text: string) => {
    if (muted || !text.trim() || typeof window === 'undefined' || !('speechSynthesis' in window)) {
      return;
    }
    try {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.lang = 'th-TH';
      utterance.rate = 0.96;
      utterance.pitch = 1.0;
      utterance.onstart = () => setIsSpeaking(true);
      utterance.onend = () => setIsSpeaking(false);
      utterance.onerror = () => setIsSpeaking(false);
      window.speechSynthesis.speak(utterance);
    } catch {
      setIsSpeaking(false);
    }
  };

  useEffect(() => {
    const clean = textTh.trim();
    if (!clean || muted) {
      if (muted && typeof window !== 'undefined' && 'speechSynthesis' in window) {
        window.speechSynthesis.cancel();
      }
      setIsSpeaking(false);
      return;
    }

    const now = Date.now();
    const isEmergency = kioskState === 'EMERGENCY_HALT';
    const isSame = clean === lastSpokenTextRef.current;

    if (!isEmergency && isSame && now - lastSpokenAtRef.current < 3500) {
      return;
    }
    if (!isEmergency && now - lastSpokenAtRef.current < 900) {
      return;
    }

    lastSpokenTextRef.current = clean;
    lastSpokenAtRef.current = now;
    speakNow(clean);
  }, [textTh, kioskState, muted]);

  // Throttle 4-second voice reminders while paused by Multi-Person / Impostor guard
  useEffect(() => {
    if (muted || (kioskState !== 'PAUSED_MULTI_PERSON' && kioskState !== 'PAUSED_IMPOSTOR')) {
      return;
    }
    const timer = window.setInterval(() => {
      const clean = textTh.trim();
      if (!clean) return;
      lastSpokenTextRef.current = clean;
      lastSpokenAtRef.current = Date.now();
      speakNow(clean);
    }, 4000);
    return () => window.clearInterval(timer);
  }, [kioskState, textTh, muted]);

  const badgeTone =
    highlightColor === 'RED'
      ? 'bg-rose-500 text-white'
      : highlightColor === 'AMBER'
        ? 'bg-amber-500 text-white'
        : 'bg-gradient-to-br from-[#0284C7] to-[#10B981] text-white';

  const textTone =
    highlightColor === 'RED'
      ? 'text-rose-700'
      : highlightColor === 'AMBER'
        ? 'text-amber-800'
        : 'text-[#0F4C81]';

  const bars = [0.45, 0.9, 1.0, 0.75, 0.5];

  return (
    <motion.div
      layout
      className="glass-card mx-auto flex w-full items-center justify-between gap-4 rounded-full px-4 py-3 sm:px-6"
    >
      <div className="flex min-w-0 items-center gap-4">
        {/* Animated 5-bar AI Voice Coach Soundwave */}
        <button
          type="button"
          onClick={() => speakNow(textTh)}
          title="ฟังเสียงแนะนำซ้ำ"
          className={`flex h-12 w-14 shrink-0 items-center justify-center gap-1 rounded-full shadow-md transition-transform active:scale-95 ${badgeTone}`}
        >
          {bars.map((peak, idx) => (
            <motion.span
              key={idx}
              className="w-1.5 rounded-full bg-white"
              animate={
                !muted && (isSpeaking || highlightColor !== 'GREEN' || kioskState === 'HOLD_PHASE')
                  ? { height: ['8px', `${Math.round(peak * 24)}px`, '8px'] }
                  : { height: '8px' }
              }
              transition={{
                repeat: Infinity,
                duration: 0.7 + idx * 0.1,
                delay: idx * 0.08,
                ease: 'easeInOut',
              }}
            />
          ))}
        </button>

        <div className="min-w-0">
          <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-[#0284C7]">
            <span>Thai AI Voice Coach</span>
            <span className="rounded-full bg-sky-100 px-2 py-0.5 text-[10px] text-[#0F4C81]">
              {muted ? 'ปิดเสียง' : isSpeaking ? 'กำลังพูด' : 'พร้อมแนะนำ'}
            </span>
          </div>
          <p className={`truncate text-lg font-bold sm:text-2xl ${textTone}`}>
            “{textTh || 'พร้อมเริ่มฝึกกายภาพบำบัดครับ'}”
          </p>
        </div>
      </div>

      <button
        type="button"
        onClick={onToggleMute}
        className={`inline-flex shrink-0 items-center gap-2 rounded-full border px-4 py-2.5 text-sm font-bold transition-colors ${
          muted
            ? 'border-slate-200 bg-slate-100 text-slate-600 hover:bg-slate-200'
            : 'border-sky-200 bg-sky-50 text-[#0F4C81] hover:bg-sky-100'
        }`}
      >
        {muted ? <VolumeX className="h-5 w-5" /> : <Volume2 className="h-5 w-5 text-[#0284C7]" />}
        <span className="hidden sm:inline">{muted ? 'เปิดเสียง AI' : 'เสียงปกติ'}</span>
      </button>
    </motion.div>
  );
}

