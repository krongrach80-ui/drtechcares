import React, { useEffect, useRef, useState } from 'react';
import {
  getGlobalMascotSpeechState,
  subscribeGlobalMascotSpeech,
  type GlobalMascotSpeechState,
  type MascotSpeechPose,
} from '@drtechcares/kiosk-input';
import { DrTechCareMascot } from './DrTechCareMascot';

export const GLOBAL_DOCK_LINGER_MS = 1100;

const POSE_BADGE_LABELS_TH: Record<MascotSpeechPose, string> = {
  greeting: 'ทักทาย',
  explaining: 'กำลังอธิบาย',
  speaking: 'กำลังพูด',
  encouraging: 'ให้กำลังใจ',
  idle: 'พร้อมแนะนำ',
};

export function shouldRenderGlobalSpeechDock(state: GlobalMascotSpeechState): boolean {
  if (state.source === 'voice_coach') {
    return false;
  }
  return Boolean(state.text && state.text.trim().length > 0);
}

export interface GlobalSpeechMascotDockProps {
  initialState?: GlobalMascotSpeechState;
  useVideo?: boolean;
}

export const GlobalSpeechMascotDock: React.FC<GlobalSpeechMascotDockProps> = ({
  initialState,
  useVideo = true,
}) => {
  const [displayState, setDisplayState] = useState<GlobalMascotSpeechState>(() =>
    initialState ?? getGlobalMascotSpeechState()
  );
  const [visible, setVisible] = useState<boolean>(() =>
    initialState ? initialState.isSpeaking && shouldRenderGlobalSpeechDock(initialState) : false
  );
  const hideTimerRef = useRef<number | null>(null);

  useEffect(() => {
    if (initialState) {
      setDisplayState(initialState);
      setVisible(initialState.isSpeaking && shouldRenderGlobalSpeechDock(initialState));
      return;
    }

    const syncFromState = (next: GlobalMascotSpeechState) => {
      if (hideTimerRef.current !== null && typeof window !== 'undefined') {
        window.clearTimeout(hideTimerRef.current);
        hideTimerRef.current = null;
      }

      if (next.source === 'voice_coach') {
        setVisible(false);
        return;
      }

      if (next.isSpeaking && shouldRenderGlobalSpeechDock(next)) {
        setDisplayState(next);
        setVisible(true);
        return;
      }

      setDisplayState((prev) => ({
        ...prev,
        isSpeaking: false,
      }));
      if (typeof window !== 'undefined') {
        hideTimerRef.current = window.setTimeout(() => {
          hideTimerRef.current = null;
          setVisible(false);
        }, GLOBAL_DOCK_LINGER_MS);
      } else {
        setVisible(false);
      }
    };

    const unsubscribe = subscribeGlobalMascotSpeech(syncFromState);
    const snapshot = getGlobalMascotSpeechState();
    if (snapshot.isSpeaking && shouldRenderGlobalSpeechDock(snapshot)) {
      syncFromState(snapshot);
    }

    return () => {
      unsubscribe();
      if (hideTimerRef.current !== null && typeof window !== 'undefined') {
        window.clearTimeout(hideTimerRef.current);
        hideTimerRef.current = null;
      }
    };
  }, [initialState]);

  if (!visible || !shouldRenderGlobalSpeechDock(displayState)) {
    return null;
  }

  const poseLabel = POSE_BADGE_LABELS_TH[displayState.pose] ?? POSE_BADGE_LABELS_TH.speaking;

  return (
    <div
      data-testid="global-speech-mascot-dock"
      role="status"
      aria-live="polite"
      className="pointer-events-none fixed bottom-4 left-4 z-[65] flex max-w-[min(92vw,470px)] items-end gap-3 animate-popup-spring"
    >
      <DrTechCareMascot
        pose={displayState.pose}
        isSpeaking={displayState.isSpeaking}
        videoFit="stage"
        useVideo={useVideo}
        showSpeechBubble={false}
        className="w-32 h-24 sm:w-40 sm:h-28 shrink-0"
        testId="global-speech-mascot"
      />

      <div className="rounded-2xl border-2 border-teal-200 bg-white/95 px-3.5 py-2.5 shadow-xl backdrop-blur-md">
        <div className="mb-1 flex items-center gap-2">
          <span className="text-[11px] font-black tracking-wide text-[#0284C7] uppercase">
            Dr. TechCare
          </span>
          <span className="rounded-full bg-teal-50 border border-teal-200 px-2 py-0.5 text-[10px] font-extrabold text-teal-800">
            {poseLabel}
          </span>
          {displayState.isSpeaking && (
            <span className="inline-flex items-end gap-0.5 h-3" aria-hidden="true">
              <span className="w-0.5 h-2 rounded-full bg-emerald-500 animate-mascot-eq" />
              <span
                className="w-0.5 h-3 rounded-full bg-teal-500 animate-mascot-eq"
                style={{ animationDelay: '120ms' }}
              />
              <span
                className="w-0.5 h-2.5 rounded-full bg-sky-500 animate-mascot-eq"
                style={{ animationDelay: '240ms' }}
              />
            </span>
          )}
        </div>
        <p className="line-clamp-3 text-xs sm:text-sm font-bold leading-snug text-[#0F4C81]">
          {displayState.text}
        </p>
      </div>
    </div>
  );
};

