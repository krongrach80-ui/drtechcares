import React, { useEffect, useRef, useState } from 'react';

export type DrTechCareMascotPose =
  | 'idle'
  | 'greeting'
  | 'explaining'
  | 'speaking'
  | 'encouraging';

export type DrTechCareMascotSize = 'sm' | 'md' | 'lg';

export const MASCOT_ENCOURAGING_DURATION_MS = 1600;
export const MASCOT_GREETING_DURATION_MS = 2600;

export const MASCOT_VIDEO_SRC = '/mascot/dr-techcare-mascot.mp4';
export const MASCOT_POSTER_SRC = '/mascot/dr-techcare-poster.jpg';

export interface MascotVideoSegment {
  startSec: number;
  endSec: number;
  playbackRate: number;
  loop: boolean;
}

/**
 * Verified non-bleed clip windows inside `gemini_generated_video_4603d38c.mp4`
 * (hard scene cuts occur at 2.000s, 4.000s, 6.458s, 10.000s).
 */
export const MASCOT_VIDEO_SEGMENTS: Record<DrTechCareMascotPose, MascotVideoSegment> = {
  greeting: { startSec: 0.06, endSec: 1.9, playbackRate: 1.0, loop: true },
  explaining: { startSec: 2.08, endSec: 3.9, playbackRate: 1.0, loop: true },
  speaking: { startSec: 4.08, endSec: 6.32, playbackRate: 1.08, loop: true },
  encouraging: { startSec: 6.54, endSec: 9.88, playbackRate: 1.0, loop: true },
  idle: { startSec: 4.82, endSec: 5.75, playbackRate: 0.52, loop: true },
};

export const MASCOT_DEFAULT_SPEECH_BUBBLES: Partial<Record<DrTechCareMascotPose, string>> = {
  greeting: 'สวัสดีครับ!',
  speaking: 'ค่อยๆ\nยกแขนขึ้น\nช้าๆ นะครับ',
  encouraging: 'สู้ๆครับ!',
};

export interface DrTechCareMascotProps {
  pose: DrTechCareMascotPose;
  isSpeaking: boolean;
  size?: DrTechCareMascotSize;
  className?: string;
  speechText?: string;
  showSpeechBubble?: boolean;
  useVideo?: boolean;
  videoFit?: 'avatar' | 'stage';
  testId?: string;
}

const SIZE_CLASSES: Record<DrTechCareMascotSize, string> = {
  sm: 'w-12 h-14',
  md: 'w-20 h-24',
  lg: 'w-32 h-36',
};

/**
 * Resolves the highest-priority active mascot pose from live clinical/voice states.
 * Precedence: encouraging > speaking > greeting > explaining > idle.
 */
export function resolveMascotPose(params: {
  isGreeting: boolean;
  isEncouraging: boolean;
  isSpeaking: boolean;
  isExplaining: boolean;
}): DrTechCareMascotPose {
  if (params.isEncouraging) return 'encouraging';
  if (params.isSpeaking) return 'speaking';
  if (params.isGreeting) return 'greeting';
  if (params.isExplaining) return 'explaining';
  return 'idle';
}

/**
 * Computes the SVG mouth vector path across 4 60FPS lip-sync phoneme frames
 * or static pose expressions.
 */
export function getMascotMouthPath(
  pose: DrTechCareMascotPose,
  isSpeaking: boolean,
  speakFrame: number
): { d: string; fill: string; showTongue: boolean } {
  if (isSpeaking || pose === 'speaking') {
    const frameIdx = Math.abs(Math.trunc(speakFrame)) % 4;
    if (frameIdx === 0) {
      // Slightly parted "o" shape
      return {
        d: 'M76.5 78.5 Q80 75.5 83.5 78.5 Q84 82.8 80 83.5 Q76 82.8 76.5 78.5 Z',
        fill: '#d96b78',
        showTongue: false,
      };
    }
    if (frameIdx === 1) {
      // Broad open "a" phoneme with visible tongue
      return {
        d: 'M73.5 77.2 Q80 76.2 86.5 77.2 Q86.8 86.5 80 86.8 Q73.2 86.5 73.5 77.2 Z',
        fill: '#cf5e6d',
        showTongue: true,
      };
    }
    if (frameIdx === 2) {
      // Mid "e" smiling open shape
      return {
        d: 'M74.5 78.0 Q80 76.8 85.5 78.0 Q85.0 83.8 80 84.0 Q75.0 83.8 74.5 78.0 Z',
        fill: '#de6e7c',
        showTongue: true,
      };
    }
    // Small rounded "u" phoneme oval
    return {
      d: 'M77.5 79.0 Q80 77.2 82.5 79.0 Q82.8 82.2 80 82.5 Q77.2 82.2 77.5 79.0 Z',
      fill: '#d96b78',
      showTongue: false,
    };
  }

  if (pose === 'encouraging' || pose === 'greeting') {
    return {
      d: 'M73 76.8 Q80 75.6 87 76.8 Q87 86.6 80 86.8 Q73 86.6 73 76.8 Z',
      fill: '#e07a84',
      showTongue: true,
    };
  }

  if (pose === 'explaining') {
    return {
      d: 'M75.5 79.2 Q80 81.8 84.8 78.8',
      fill: 'none',
      showTongue: false,
    };
  }

  // Idle gentle clinical smile
  return {
    d: 'M75 78.8 Q80 83.0 85 78.8',
    fill: 'none',
    showTongue: false,
  };
}

export const DrTechCareMascot: React.FC<DrTechCareMascotProps> = ({
  pose,
  isSpeaking,
  size = 'md',
  className,
  speechText,
  showSpeechBubble = true,
  useVideo = true,
  videoFit = 'avatar',
  testId = 'dr-techcare-mascot',
}) => {
  const [speakFrame, setSpeakFrame] = useState<number>(0);
  const [videoFailed, setVideoFailed] = useState<boolean>(false);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const activeSegment = MASCOT_VIDEO_SEGMENTS[pose];
  const shouldPlayVideo = useVideo && !videoFailed;
  const lipSyncActive = !shouldPlayVideo && (isSpeaking || pose === 'speaking');

  // 60FPS requestAnimationFrame phoneme cadence when SVG fallback is speaking
  useEffect(() => {
    if (!lipSyncActive || typeof window === 'undefined') {
      setSpeakFrame(0);
      return;
    }
    let rafId = 0;
    const startedAt = performance.now();
    const stepMs = isSpeaking ? 95 : 165;

    const tick = (now: number) => {
      const nextFrame = Math.floor((now - startedAt) / stepMs) % 4;
      setSpeakFrame((prev) => (prev === nextFrame ? prev : nextFrame));
      rafId = window.requestAnimationFrame(tick);
    };

    rafId = window.requestAnimationFrame(tick);
    return () => {
      if (rafId) window.cancelAnimationFrame(rafId);
    };
  }, [lipSyncActive, isSpeaking]);

  // 60FPS requestAnimationFrame video segment trimmer preventing cross-cut bleed
  useEffect(() => {
    if (!shouldPlayVideo || typeof window === 'undefined') {
      return;
    }
    const videoEl = videoRef.current;
    if (!videoEl) return;

    const targetRate = isSpeaking && pose === 'speaking' ? 1.12 : activeSegment.playbackRate;
    videoEl.muted = true;
    videoEl.playbackRate = targetRate;

    if (videoEl.readyState >= 1) {
      if (
        videoEl.currentTime < activeSegment.startSec - 0.06 ||
        videoEl.currentTime >= activeSegment.endSec - 0.04
      ) {
        videoEl.currentTime = activeSegment.startSec;
      }
      if (videoEl.paused) {
        void videoEl.play().catch(() => {});
      }
    }

    let rafId = 0;
    const watchFrame = () => {
      const el = videoRef.current;
      if (el && el.readyState >= 2) {
        if (
          el.currentTime >= activeSegment.endSec - 0.04 ||
          el.currentTime < activeSegment.startSec - 0.06
        ) {
          el.currentTime = activeSegment.startSec;
        }
        if (el.paused) {
          void el.play().catch(() => {});
        }
      }
      rafId = window.requestAnimationFrame(watchFrame);
    };

    rafId = window.requestAnimationFrame(watchFrame);
    return () => {
      if (rafId) window.cancelAnimationFrame(rafId);
    };
  }, [shouldPlayVideo, pose, isSpeaking, activeSegment.startSec, activeSegment.endSec, activeSegment.playbackRate]);

  const mouth = getMascotMouthPath(pose, isSpeaking, speakFrame);
  const resolvedBubbleText =
    speechText !== undefined ? speechText.trim() : (MASCOT_DEFAULT_SPEECH_BUBBLES[pose] ?? '');
  const showWaveOverlay = isSpeaking || pose === 'speaking';
  const bodyMotionClass =
    pose === 'encouraging' ? 'animate-fist-pump' : 'animate-mascot-breath';
  const rootClassName = className
    ? `relative inline-flex items-center justify-center select-none ${className}`
    : `relative inline-flex items-center justify-center select-none ${SIZE_CLASSES[size]}`;

  const videoBorderClass = isSpeaking
    ? 'border-emerald-300 ring-2 ring-emerald-300/60 shadow-emerald-500/25'
    : pose === 'encouraging'
      ? 'border-amber-300 ring-2 ring-amber-300/60 shadow-amber-400/25'
      : 'border-sky-200 shadow-sky-900/15';

  const videoObjectClass =
    videoFit === 'stage'
      ? 'h-full w-full object-cover object-center pointer-events-none'
      : 'h-full w-full object-cover object-[50%_42%] scale-[1.08] pointer-events-none';

  return (
    <div
      className={rootClassName}
      data-testid={testId}
      data-pose={pose}
      data-speaking={isSpeaking ? 'true' : 'false'}
    >
      {showSpeechBubble && resolvedBubbleText.length > 0 && (
        <div
          data-testid="mascot-speech-bubble"
          className="pointer-events-none absolute bottom-full mb-1.5 left-0 z-30 whitespace-pre-line rounded-xl border border-teal-300 bg-white/95 px-2.5 py-1 text-[10px] font-black leading-tight text-[#0F4C81] shadow-md"
        >
          <span>{resolvedBubbleText}</span>
          <span
            aria-hidden="true"
            className="absolute -bottom-1 left-4 h-2 w-2 rotate-45 border-r border-b border-teal-300 bg-white"
          />
        </div>
      )}

      {shouldPlayVideo ? (
        <div
          className={`relative h-full w-full overflow-hidden rounded-2xl border-2 bg-[#eaf6f3] shadow-sm transition-all duration-300 ease-in-out ${videoBorderClass}`}
        >
          <video
            ref={videoRef}
            src={MASCOT_VIDEO_SRC}
            poster={MASCOT_POSTER_SRC}
            muted
            autoPlay
            playsInline
            preload="auto"
            disablePictureInPicture
            onLoadedMetadata={(e) => {
              const el = e.currentTarget;
              el.muted = true;
              el.playbackRate =
                isSpeaking && pose === 'speaking' ? 1.12 : activeSegment.playbackRate;
              el.currentTime = activeSegment.startSec;
              void el.play().catch(() => {});
            }}
            onError={() => setVideoFailed(true)}
            data-testid="mascot-video"
            data-video-start={activeSegment.startSec}
            data-video-end={activeSegment.endSec}
            className={videoObjectClass}
          />
          {showWaveOverlay && (
            <div
              aria-hidden="true"
              data-testid="mascot-soundwaves"
              className="pointer-events-none absolute bottom-1 right-1 flex items-end gap-0.5 rounded-full bg-white/90 px-1.5 py-1 shadow-xs"
            >
              <span className="w-0.5 h-2 rounded-full bg-emerald-500 animate-mascot-eq" />
              <span
                className="w-0.5 h-3 rounded-full bg-teal-500 animate-mascot-eq"
                style={{ animationDelay: '120ms' }}
              />
              <span
                className="w-0.5 h-2.5 rounded-full bg-sky-500 animate-mascot-eq"
                style={{ animationDelay: '240ms' }}
              />
            </div>
          )}
          {pose === 'encouraging' && (
            <svg
              viewBox="0 0 24 24"
              aria-hidden="true"
              data-testid="mascot-encouraging-particles"
              className="pointer-events-none absolute top-0.5 right-0.5 h-4 w-4 animate-mascot-sparkle"
              fill="none"
            >
              <path
                d="M12 2 L14.2 9.2 L21.5 11.5 L14.2 13.8 L12 21 L9.8 13.8 L2.5 11.5 L9.8 9.2 Z"
                fill="#facc15"
                stroke="#ca8a04"
                strokeWidth="1.2"
              />
            </svg>
          )}
        </div>
      ) : (
      <svg
        viewBox="0 0 160 180"
        className="w-full h-full overflow-visible"
        fill="none"
        aria-hidden="true"
      >
        {/* Soft Ground Shadow */}
        <ellipse cx="80" cy="169" rx="34" ry="5.5" fill="rgba(15, 76, 129, 0.14)" />

        {/* Back Hair Mass & Flowing Side Braid (Right Shoulder) */}
        <g>
          {/* Back messy hair tufts */}
          <path
            d="M44 58 Q36 74 44 86 Q50 92 58 90 L102 90 Q112 92 117 82 Q123 68 114 55 Z"
            fill="#e9c46a"
            stroke="#3d2b3d"
            strokeWidth="2.4"
            strokeLinejoin="round"
          />
          {/* Long Side Braid Falling Behind His Left Shoulder */}
          <g>
            <path
              d="M111 76 Q124 90 121 106 Q117 120 124 134"
              stroke="#3d2b3d"
              strokeWidth="12"
              strokeLinecap="round"
            />
            <path
              d="M111 76 Q124 90 121 106 Q117 120 124 134"
              stroke="#e9c46a"
              strokeWidth="7.5"
              strokeLinecap="round"
            />
            {/* Braid knot bumps */}
            <ellipse
              cx="117"
              cy="91"
              rx="4.8"
              ry="3.6"
              transform="rotate(-20 117 91)"
              fill="#f5de92"
              stroke="#3d2b3d"
              strokeWidth="1.8"
            />
            <ellipse
              cx="121"
              cy="104"
              rx="4.6"
              ry="3.5"
              transform="rotate(14 121 104)"
              fill="#f4a261"
              stroke="#3d2b3d"
              strokeWidth="1.8"
            />
            <ellipse
              cx="120"
              cy="118"
              rx="4.2"
              ry="3.3"
              transform="rotate(-18 120 118)"
              fill="#f5de92"
              stroke="#3d2b3d"
              strokeWidth="1.8"
            />
            {/* Teal braid ribbon band */}
            <rect
              x="118"
              y="128"
              width="9"
              height="4.2"
              rx="2"
              transform="rotate(-12 122 130)"
              fill="#2dd4bf"
              stroke="#3d2b3d"
              strokeWidth="1.5"
            />
            {/* Fluffy braid tassel tip */}
            <path
              d="M122 133 Q129 140 133 149 Q124 147 120 143 Q116 149 114 140 Z"
              fill="#f5de92"
              stroke="#3d2b3d"
              strokeWidth="1.8"
              strokeLinejoin="round"
            />
          </g>
        </g>

        {/* Animated Full Character Torso + Head Container */}
        <g className={`transition-transform duration-300 ease-in-out ${bodyMotionClass}`}>
          {/* Navy Scrub Trousers */}
          <path
            d="M61 139 L58 159 L74 159 L78 145 L82 145 L86 159 L102 159 L99 139 Z"
            fill="#28364a"
            stroke="#3d2b3d"
            strokeWidth="2.4"
            strokeLinejoin="round"
          />

          {/* Clinical White Sneakers with Teal Trim */}
          <g>
            <path
              d="M53 158 Q52 167 66 167 L75 167 Q76 160 72 158 Z"
              fill="#ffffff"
              stroke="#3d2b3d"
              strokeWidth="2.2"
              strokeLinejoin="round"
            />
            <path d="M55 164 L73 164" stroke="#2dd4bf" strokeWidth="2.2" strokeLinecap="round" />
            <path
              d="M85 158 Q84 167 93 167 L107 167 Q108 160 102 158 Z"
              fill="#ffffff"
              stroke="#3d2b3d"
              strokeWidth="2.2"
              strokeLinejoin="round"
            />
            <path d="M87 164 L105 164" stroke="#2dd4bf" strokeWidth="2.2" strokeLinecap="round" />
          </g>

          {/* Deep Navy V-Neck Inner Scrubs */}
          <path
            d="M62 96 L98 96 L101 141 L59 141 Z"
            fill="#1e293b"
            stroke="#3d2b3d"
            strokeWidth="2.2"
            strokeLinejoin="round"
          />
          <path
            d="M71 96 L80 109 L89 96"
            fill="none"
            stroke="#2dd4bf"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          />

          {/* Tailored White Medical Lab Coat with Neon Teal-Cyan Accents */}
          <g>
            {/* Left coat panel */}
            <path
              d="M61 96 Q53 98 51 112 L48 143 Q60 146 76 144 L76 105 L66 96 Z"
              fill="#ffffff"
              stroke="#3d2b3d"
              strokeWidth="2.4"
              strokeLinejoin="round"
            />
            {/* Right coat panel */}
            <path
              d="M99 96 Q107 98 109 112 L112 143 Q100 146 84 144 L84 105 L94 96 Z"
              fill="#ffffff"
              stroke="#3d2b3d"
              strokeWidth="2.4"
              strokeLinejoin="round"
            />
            {/* Subtle shading fold inside hem */}
            <path d="M51 139 Q64 142 74 140" stroke="#cbd5e1" strokeWidth="2" strokeLinecap="round" />
            <path d="M86 140 Q96 142 109 139" stroke="#cbd5e1" strokeWidth="2" strokeLinecap="round" />
            {/* Glowing teal-cyan coat lapel & hem neon borders */}
            <path
              d="M65 97 L75 108 L75 142 L51 141"
              stroke="#2dd4bf"
              strokeWidth="2.4"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
            <path
              d="M95 97 L85 108 L85 142 L109 141"
              stroke="#2dd4bf"
              strokeWidth="2.4"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
            <path d="M89 125 L103 125" stroke="#06b6d4" strokeWidth="1.6" strokeLinecap="round" />
          </g>

          {/* Clipped Clinical ID Card Badge (Left Chest / Viewer Right) */}
          <g>
            <rect
              x="89"
              y="112"
              width="15"
              height="20"
              rx="2.5"
              fill="#ffffff"
              stroke="#3d2b3d"
              strokeWidth="1.8"
            />
            <rect x="90" y="113" width="13" height="4.5" rx="1" fill="#0284c7" />
            {/* Tiny ID cross & lines */}
            <path d="M93.5 122 V127 M91 124.5 H96" stroke="#06b6d4" strokeWidth="1.7" strokeLinecap="round" />
            <path d="M98 122.5 H102 M98 126 H101.5" stroke="#64748b" strokeWidth="1.4" strokeLinecap="round" />
            <rect x="94.5" y="109.5" width="4" height="3" rx="1" fill="#0284c7" stroke="#3d2b3d" strokeWidth="1.2" />
          </g>

          {/* Stethoscope Draped Around Neck */}
          <g>
            <path
              d="M64 96 Q58 112 63 124 Q67 132 76 128"
              stroke="#334155"
              strokeWidth="3.6"
              strokeLinecap="round"
            />
            <path
              d="M96 96 Q102 108 98 117"
              stroke="#334155"
              strokeWidth="3.6"
              strokeLinecap="round"
            />
            <circle cx="78" cy="127" r="4.6" fill="#e2e8f0" stroke="#3d2b3d" strokeWidth="1.8" />
            <circle cx="78" cy="127" r="1.9" fill="#06b6d4" />
          </g>

          {/* Pose-Specific Arms & Hands (300ms Smooth Transitions) */}
          {pose === 'idle' && (
            <g className="transition-transform duration-300 ease-in-out">
              {/* Viewer-left arm relaxed */}
              <path
                d="M55 101 Q45 113 46 127 L54 128 Q56 116 60 107 Z"
                fill="#ffffff"
                stroke="#3d2b3d"
                strokeWidth="2.3"
                strokeLinejoin="round"
              />
              <path d="M46 124 L54 125" stroke="#2dd4bf" strokeWidth="2.2" strokeLinecap="round" />
              <circle cx="50" cy="131" r="4.8" fill="#ffe5d4" stroke="#3d2b3d" strokeWidth="2" />

              {/* Viewer-right arm relaxed */}
              <path
                d="M105 101 Q115 113 114 127 L106 128 Q104 116 100 107 Z"
                fill="#ffffff"
                stroke="#3d2b3d"
                strokeWidth="2.3"
                strokeLinejoin="round"
              />
              <path d="M106 125 L114 124" stroke="#2dd4bf" strokeWidth="2.2" strokeLinecap="round" />
              <circle cx="110" cy="131" r="4.8" fill="#ffe5d4" stroke="#3d2b3d" strokeWidth="2" />
            </g>
          )}

          {pose === 'greeting' && (
            <g className="transition-transform duration-300 ease-in-out">
              {/* Viewer-left arm akimbo on hip */}
              <path
                d="M55 101 Q39 109 42 120 Q45 127 54 126 L56 119 Q49 119 50 113 L59 107 Z"
                fill="#ffffff"
                stroke="#3d2b3d"
                strokeWidth="2.3"
                strokeLinejoin="round"
              />
              <path d="M51 119 L55 125" stroke="#2dd4bf" strokeWidth="2.1" strokeLinecap="round" />
              <circle cx="55" cy="123" r="4.5" fill="#ffe5d4" stroke="#3d2b3d" strokeWidth="2" />

              {/* Viewer-right arm waving high with .animate-gentle-wave */}
              <g className="animate-gentle-wave">
                <path
                  d="M103 105 Q120 104 127 89 L119 85 Q113 96 101 97 Z"
                  fill="#ffffff"
                  stroke="#3d2b3d"
                  strokeWidth="2.3"
                  strokeLinejoin="round"
                />
                <path d="M119 87 L127 90" stroke="#2dd4bf" strokeWidth="2.2" strokeLinecap="round" />
                {/* Open waving palm */}
                <path
                  d="M121 86 Q120 77 125 75 Q127 76 127 81 Q129 74 132 76 Q133 78 131 82 Q134 77 136 80 Q135 86 128 89 Q124 90 121 86 Z"
                  fill="#ffe5d4"
                  stroke="#3d2b3d"
                  strokeWidth="2"
                  strokeLinejoin="round"
                />
              </g>
            </g>
          )}

          {pose === 'explaining' && (
            <g className="transition-transform duration-300 ease-in-out">
              {/* Viewer-left arm holding cyan holographic tablet */}
              <path
                d="M56 102 Q42 110 41 124 L49 126 Q52 115 60 108 Z"
                fill="#ffffff"
                stroke="#3d2b3d"
                strokeWidth="2.3"
                strokeLinejoin="round"
              />
              <circle cx="44" cy="126" r="4.6" fill="#ffe5d4" stroke="#3d2b3d" strokeWidth="2" />

              {/* Digital Holographic Screen Clipboard (Cyan Translucent Vector Bounds) */}
              <g>
                <path
                  d="M23 102 L60 108 L55 137 L18 131 Z"
                  fill="rgba(6, 182, 212, 0.24)"
                  stroke="#06b6d4"
                  strokeWidth="2.2"
                  strokeLinejoin="round"
                />
                <path
                  d="M27 107 L56 112"
                  stroke="#2dd4bf"
                  strokeWidth="1.6"
                  strokeLinecap="round"
                />
                {/* Compact clinical checkmark path scaled onto digital tablet */}
                <g transform="translate(27, 111) scale(0.22)">
                  <path
                    d="M33 51 L45 63 L68 38"
                    stroke="#0284c7"
                    strokeWidth="9"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </g>
                <path
                  d="M42 120 H53 M41 125 H51 M26 128 H37"
                  stroke="#0284c7"
                  strokeWidth="1.8"
                  strokeLinecap="round"
                />
              </g>

              {/* Viewer-right arm pointing cleanly at digital tablet */}
              <path
                d="M104 102 Q112 116 96 121 L72 118 L73 111 L96 113 Q102 111 98 104 Z"
                fill="#ffffff"
                stroke="#3d2b3d"
                strokeWidth="2.3"
                strokeLinejoin="round"
              />
              <path d="M75 111 L74 118" stroke="#2dd4bf" strokeWidth="2.1" strokeLinecap="round" />
              {/* Pointing hand */}
              <path
                d="M73 112 L61 112 Q59 114 61 116 L71 117 Q74 118 74 114 Z"
                fill="#ffe5d4"
                stroke="#3d2b3d"
                strokeWidth="2"
                strokeLinejoin="round"
              />
            </g>
          )}

          {pose === 'speaking' && (
            <g className="transition-transform duration-300 ease-in-out">
              {/* Viewer-left arm bent with hand reassuringly on chest */}
              <path
                d="M55 102 Q43 114 55 123 L70 118 L68 111 L57 114 Q53 111 59 106 Z"
                fill="#ffffff"
                stroke="#3d2b3d"
                strokeWidth="2.3"
                strokeLinejoin="round"
              />
              <path d="M68 111 L70 118" stroke="#2dd4bf" strokeWidth="2.1" strokeLinecap="round" />
              <ellipse
                cx="74"
                cy="114"
                rx="5.4"
                ry="4.3"
                fill="#ffe5d4"
                stroke="#3d2b3d"
                strokeWidth="2"
              />

              {/* Viewer-right arm extended outward in clear explaining gesture */}
              <path
                d="M104 104 Q118 110 129 102 L125 94 Q116 101 102 98 Z"
                fill="#ffffff"
                stroke="#3d2b3d"
                strokeWidth="2.3"
                strokeLinejoin="round"
              />
              <path d="M124 95 L129 102" stroke="#2dd4bf" strokeWidth="2.1" strokeLinecap="round" />
              <path
                d="M127 97 Q132 90 137 92 Q139 95 135 99 L139 100 Q138 105 131 104 Q127 103 127 97 Z"
                fill="#ffe5d4"
                stroke="#3d2b3d"
                strokeWidth="2"
                strokeLinejoin="round"
              />
            </g>
          )}

          {pose === 'encouraging' && (
            <g className="transition-transform duration-300 ease-in-out">
              {/* Both sleeves raised up cleanly */}
              <path
                d="M56 104 Q37 104 34 89 L42 87 Q46 96 58 96 Z"
                fill="#ffffff"
                stroke="#3d2b3d"
                strokeWidth="2.3"
                strokeLinejoin="round"
              />
              <path d="M35 89 L42 87" stroke="#2dd4bf" strokeWidth="2.1" strokeLinecap="round" />
              {/* Viewer-left fist raised */}
              <rect
                x="30"
                y="77"
                width="13"
                height="11"
                rx="4.8"
                fill="#ffe5d4"
                stroke="#3d2b3d"
                strokeWidth="2.1"
              />
              <path d="M34 82 H39" stroke="#3d2b3d" strokeWidth="1.5" strokeLinecap="round" />

              <path
                d="M104 104 Q123 104 126 89 L118 87 Q114 96 102 96 Z"
                fill="#ffffff"
                stroke="#3d2b3d"
                strokeWidth="2.3"
                strokeLinejoin="round"
              />
              <path d="M118 87 L125 89" stroke="#2dd4bf" strokeWidth="2.1" strokeLinecap="round" />
              {/* Viewer-right fist raised */}
              <rect
                x="117"
                y="77"
                width="13"
                height="11"
                rx="4.8"
                fill="#ffe5d4"
                stroke="#3d2b3d"
                strokeWidth="2.1"
              />
              <path d="M121 82 H126" stroke="#3d2b3d" strokeWidth="1.5" strokeLinecap="round" />
            </g>
          )}

          {/* Chibi Head */}
          <g>
            {/* Neck shadow */}
            <rect x="74" y="88" width="12" height="11" rx="4" fill="#f7cfb8" stroke="#3d2b3d" strokeWidth="2" />

            {/* Ears & Tiny Gold Hoop Earrings */}
            <circle cx="43" cy="67" r="6.5" fill="#ffe5d4" stroke="#3d2b3d" strokeWidth="2.2" />
            <path d="M41.5 66 Q43 68.5 45 67" stroke="#e5a98b" strokeWidth="1.5" strokeLinecap="round" />
            <circle cx="42.5" cy="73.5" r="2.2" stroke="#eab308" strokeWidth="1.6" />

            <circle cx="117" cy="67" r="6.5" fill="#ffe5d4" stroke="#3d2b3d" strokeWidth="2.2" />
            <path d="M115 67 Q117 68.5 118.5 66" stroke="#e5a98b" strokeWidth="1.5" strokeLinecap="round" />
            <circle cx="117.5" cy="73.5" r="2.2" stroke="#eab308" strokeWidth="1.6" />

            {/* Round Chibi Face Base */}
            <path
              d="M44 56 Q44 31 80 31 Q116 31 116 56 Q116 88 80 91 Q44 88 44 56 Z"
              fill="#ffe5d4"
              stroke="#3d2b3d"
              strokeWidth="2.4"
              strokeLinejoin="round"
            />

            {/* Soft Rosy Cheek Blushes */}
            <ellipse cx="56" cy="74" rx="6.5" ry="3.4" fill="#f9a8b8" opacity="0.55" />
            <ellipse cx="104" cy="74" rx="6.5" ry="3.4" fill="#f9a8b8" opacity="0.55" />

            {/* Cute Small Nose */}
            <path d="M79 72 Q80 73.6 81.2 72" stroke="#d4957b" strokeWidth="1.8" strokeLinecap="round" />

            {/* Expressive Eyes by Pose */}
            {pose === 'encouraging' ? (
              <g className="animate-mascot-sparkle">
                {/* Joyful Golden-Yellow Star Eyes */}
                <path
                  d="M64 54 L67 62 L75 65 L67 68 L64 76 L61 68 L53 65 L61 62 Z"
                  fill="#facc15"
                  stroke="#3d2b3d"
                  strokeWidth="2.1"
                  strokeLinejoin="round"
                />
                <path
                  d="M64 59 L65.5 63.5 L70 65 L65.5 66.5 L64 71 L62.5 66.5 L58 65 L62.5 63.5 Z"
                  fill="#fef08a"
                />
                <path
                  d="M96 54 L99 62 L107 65 L99 68 L96 76 L93 68 L85 65 L93 62 Z"
                  fill="#facc15"
                  stroke="#3d2b3d"
                  strokeWidth="2.1"
                  strokeLinejoin="round"
                />
                <path
                  d="M96 59 L97.5 63.5 L102 65 L97.5 66.5 L96 71 L94.5 66.5 L90 65 L94.5 63.5 Z"
                  fill="#fef08a"
                />
              </g>
            ) : pose === 'greeting' ? (
              <g>
                {/* Left Big Violet Eye */}
                <g className="animate-mascot-blink">
                  <ellipse cx="64" cy="64" rx="8.2" ry="9.2" fill="#ffffff" stroke="#3d2b3d" strokeWidth="2.1" />
                  <ellipse cx="64.5" cy="64.5" rx="6.4" ry="7.4" fill="#9b5de5" stroke="#7c3aed" strokeWidth="1.2" />
                  <circle cx="64.5" cy="64.5" r="3.2" fill="#3b1d5e" />
                  <ellipse cx="64.5" cy="69" rx="3.8" ry="1.8" fill="#e9d5ff" opacity="0.8" />
                  <circle cx="61.5" cy="60.5" r="2.1" fill="#ffffff" />
                  <circle cx="67.5" cy="66.5" r="1.1" fill="#ffffff" />
                </g>
                {/* Right Winking Eye */}
                <path
                  d="M86 65 Q96 58 106 65"
                  stroke="#3d2b3d"
                  strokeWidth="2.8"
                  strokeLinecap="round"
                />
                <path
                  d="M103 63 L107 60.5"
                  stroke="#3d2b3d"
                  strokeWidth="2"
                  strokeLinecap="round"
                />
              </g>
            ) : (
              <g>
                {/* Viewer-Right Big Violet Eye (Animated Blink) */}
                <g className="animate-mascot-blink">
                  <ellipse cx="64" cy="64" rx="8.2" ry="9.2" fill="#ffffff" stroke="#3d2b3d" strokeWidth="2.1" />
                  <ellipse cx="64.5" cy="64.5" rx="6.4" ry="7.4" fill="#9b5de5" stroke="#7c3aed" strokeWidth="1.2" />
                  <circle cx="64.5" cy="64.5" r="3.2" fill="#3b1d5e" />
                  <ellipse cx="64.5" cy="69" rx="3.8" ry="1.8" fill="#e9d5ff" opacity="0.8" />
                  <circle cx="61.5" cy="60.5" r="2.1" fill="#ffffff" />
                  <circle cx="67.5" cy="66.5" r="1.1" fill="#ffffff" />
                </g>
                {/* Viewer-Left Big Violet Eye (Animated Blink, Visible Behind Visor When Explaining) */}
                <g className="animate-mascot-blink">
                  <ellipse cx="96" cy="64" rx="8.2" ry="9.2" fill="#ffffff" stroke="#3d2b3d" strokeWidth="2.1" />
                  <ellipse cx="95.5" cy="64.5" rx="6.4" ry="7.4" fill="#9b5de5" stroke="#7c3aed" strokeWidth="1.2" />
                  <circle cx="95.5" cy="64.5" r="3.2" fill="#3b1d5e" />
                  <ellipse cx="95.5" cy="69" rx="3.8" ry="1.8" fill="#e9d5ff" opacity="0.8" />
                  <circle cx="92.5" cy="60.5" r="2.1" fill="#ffffff" />
                  <circle cx="98.5" cy="66.5" r="1.1" fill="#ffffff" />
                </g>
              </g>
            )}

            {/* Dynamic Lip-Sync / Expression Mouth */}
            <g>
              <path
                d={mouth.d}
                fill={mouth.fill}
                stroke="#3d2b3d"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
              {mouth.showTongue && (
                <path
                  d="M76 82.8 Q80 80.5 84 82.8 Q82.5 85.8 80 85.8 Q77.5 85.8 76 82.8 Z"
                  fill="#fda4af"
                />
              )}
            </g>

            {/* Messy Layered Blonde Hair + Bangs + Forward Side Braids */}
            <g>
              {/* Crown & spiky silhouette */}
              <path
                d="M36 59 Q34 40 46 28 Q54 16 74 16 Q82 6 95 16 Q113 18 121 33 Q127 45 123 62 Q117 50 109 47 Q105 56 97 54 Q90 44 82 54 Q73 45 64 55 Q56 44 48 53 Q42 54 36 59 Z"
                fill="#f5de92"
                stroke="#3d2b3d"
                strokeWidth="2.4"
                strokeLinejoin="round"
              />
              {/* Warm layered shading streaks */}
              <path
                d="M52 29 Q64 21 79 23 Q96 20 109 30 Q95 27 81 30 Q66 26 52 29 Z"
                fill="#fff7d6"
                opacity="0.85"
              />
              <path
                d="M58 42 L64 53 L69 43 M82 40 L85 52 L90 42"
                stroke="#e9c46a"
                strokeWidth="2.4"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
              {/* Left & Right Forward Side Braids Framing Face */}
              <path
                d="M42 56 Q37 72 41 87 Q46 84 48 72 Z"
                fill="#f4a261"
                stroke="#3d2b3d"
                strokeWidth="2"
                strokeLinejoin="round"
              />
              <path
                d="M117 56 Q123 72 119 88 Q113 84 111 72 Z"
                fill="#f4a261"
                stroke="#3d2b3d"
                strokeWidth="2"
                strokeLinejoin="round"
              />
              {/* Central wispy bangs */}
              <path
                d="M62 46 Q67 57 74 52 M85 45 Q90 56 98 51"
                stroke="#3d2b3d"
                strokeWidth="2.1"
                strokeLinecap="round"
              />

              {/* Green Cross / Botanical Hair Clip on His Left Hair Side (Viewer Right) */}
              <g>
                <ellipse
                  cx="109"
                  cy="41"
                  rx="4.6"
                  ry="2.8"
                  transform="rotate(-35 109 41)"
                  fill="#34d399"
                  stroke="#3d2b3d"
                  strokeWidth="1.5"
                />
                <ellipse
                  cx="113"
                  cy="49"
                  rx="4.8"
                  ry="2.9"
                  transform="rotate(-22 113 49)"
                  fill="#10b981"
                  stroke="#3d2b3d"
                  strokeWidth="1.5"
                />
                <ellipse
                  cx="114"
                  cy="57"
                  rx="4.4"
                  ry="2.7"
                  transform="rotate(8 114 57)"
                  fill="#34d399"
                  stroke="#3d2b3d"
                  strokeWidth="1.5"
                />
                {/* Tiny white medical cross emblem on clip */}
                <path
                  d="M113 47 V51 M111 49 H115"
                  stroke="#ffffff"
                  strokeWidth="1.6"
                  strokeLinecap="round"
                />
              </g>
            </g>

            {/* Sporty Cyan-Teal Forehead Visor / Sunglasses (Slides Over Eyes in 'explaining') */}
            <g
              data-testid="mascot-visor"
              className="transition-transform duration-300 ease-in-out"
              style={{
                transform:
                  pose === 'explaining'
                    ? 'translateY(25px) scale(1.03)'
                    : 'translateY(0px) scale(1)',
                transformOrigin: '80px 36px',
              }}
            >
              {/* Gold aviator top bar & nose bridge */}
              <path
                d="M44 27 Q80 22 116 27"
                stroke="#d99b26"
                strokeWidth="2.6"
                strokeLinecap="round"
              />
              {/* Left Cyan-Teal Lens */}
              <rect
                x="44"
                y="26"
                width="31"
                height="18"
                rx="6.5"
                fill="#06b6d4"
                fillOpacity={pose === 'explaining' ? 0.56 : 0.92}
                stroke="#d99b26"
                strokeWidth="2.6"
              />
              {/* Right Cyan-Teal Lens */}
              <rect
                x="85"
                y="26"
                width="31"
                height="18"
                rx="6.5"
                fill="#38bdf8"
                fillOpacity={pose === 'explaining' ? 0.56 : 0.92}
                stroke="#d99b26"
                strokeWidth="2.6"
              />
              {/* Center gold bridge */}
              <path d="M75 32 Q80 29.5 85 32" stroke="#d99b26" strokeWidth="2.8" strokeLinecap="round" />
              {/* Crisp white glass sheen highlights */}
              <path
                d="M50 30 L57 39 M56 29 L61 36 M91 30 L98 39 M97 29 L102 36"
                stroke="#ffffff"
                strokeWidth="2"
                strokeLinecap="round"
                opacity="0.78"
              />
            </g>
          </g>
        </g>

        {/* Dynamic Pose Overlays: Ripples, Cascading Audio Waves, Digital Sparks, Celebration Particles */}
        {pose === 'greeting' && (
          <g aria-hidden="true">
            {/* Soft cyan voice ripples on viewer-left */}
            <path
              d="M27 83 Q21 91 27 99"
              stroke="#2dd4bf"
              strokeWidth="2.4"
              strokeLinecap="round"
              className="animate-audio-waves"
            />
            <path
              d="M21 79 Q13 91 21 103"
              stroke="#06b6d4"
              strokeWidth="2.1"
              strokeLinecap="round"
              className="animate-audio-waves"
              style={{ animationDelay: '180ms' }}
            />
            {/* Greeting vertical sound bars on viewer-right */}
            <g className="animate-audio-waves">
              <rect x="137" y="68" width="2.8" height="10" rx="1.4" fill="#2dd4bf" className="animate-mascot-eq" />
              <rect
                x="142"
                y="63"
                width="2.8"
                height="20"
                rx="1.4"
                fill="#10b981"
                className="animate-mascot-eq"
                style={{ animationDelay: '120ms' }}
              />
              <rect
                x="147"
                y="66"
                width="2.8"
                height="14"
                rx="1.4"
                fill="#06b6d4"
                className="animate-mascot-eq"
                style={{ animationDelay: '240ms' }}
              />
            </g>
          </g>
        )}

        {pose === 'explaining' && (
          <g aria-hidden="true">
            <path
              d="M15 95 L20 100 M11 105 L18 106 M14 115 L20 112"
              stroke="#06b6d4"
              strokeWidth="2.3"
              strokeLinecap="round"
              className="animate-mascot-sparkle"
            />
          </g>
        )}

        {showWaveOverlay && (
          <g aria-hidden="true" data-testid="mascot-soundwaves">
            {/* Cascading Emerald-Teal Soundwave Arches behind/beside his left side */}
            <path
              d="M132 58 Q140 69 132 80"
              stroke="#10b981"
              strokeWidth="2.8"
              strokeLinecap="round"
              className="animate-audio-waves"
              style={{ animationDelay: '0ms' }}
            />
            <path
              d="M138 52 Q149 69 138 86"
              stroke="#2dd4bf"
              strokeWidth="2.6"
              strokeLinecap="round"
              className="animate-audio-waves"
              style={{ animationDelay: '160ms' }}
            />
            <path
              d="M144 46 Q158 69 144 92"
              stroke="#06b6d4"
              strokeWidth="2.3"
              strokeLinecap="round"
              className="animate-audio-waves"
              style={{ animationDelay: '320ms' }}
            />
            {/* Upper concentric ripple icon */}
            <circle
              cx="128"
              cy="33"
              r="7.5"
              stroke="#2dd4bf"
              strokeWidth="2"
              className="animate-audio-waves"
              style={{ animationDelay: '100ms' }}
            />
            <circle cx="128" cy="33" r="3.4" stroke="#10b981" strokeWidth="1.8" />
          </g>
        )}

        {pose === 'encouraging' && (
          <g aria-hidden="true" data-testid="mascot-encouraging-particles">
            {/* Gold sparkling 4-point star particles */}
            <path
              d="M27 42 L29 48 L35 50 L29 52 L27 58 L25 52 L19 50 L25 48 Z"
              fill="#facc15"
              stroke="#ca8a04"
              strokeWidth="1"
              className="animate-mascot-sparkle"
            />
            <path
              d="M136 39 L138 44 L143 46 L138 48 L136 53 L134 48 L129 46 L134 44 Z"
              fill="#facc15"
              stroke="#ca8a04"
              strokeWidth="1"
              className="animate-mascot-sparkle"
              style={{ animationDelay: '220ms' }}
            />
            <path
              d="M88 6 L89.5 10 L93.5 11.5 L89.5 13 L88 17 L86.5 13 L82.5 11.5 L86.5 10 Z"
              fill="#fde047"
              stroke="#ca8a04"
              strokeWidth="0.9"
              className="animate-mascot-sparkle"
              style={{ animationDelay: '350ms' }}
            />
            {/* Floating Teal Medical Cross Indicators */}
            <path
              d="M22 126 V136 M17 131 H27"
              stroke="#14b8a6"
              strokeWidth="3.5"
              strokeLinecap="round"
              className="animate-mascot-sparkle"
              style={{ animationDelay: '140ms' }}
            />
            <path
              d="M140 118 V126 M136 122 H144"
              stroke="#2dd4bf"
              strokeWidth="3"
              strokeLinecap="round"
              className="animate-mascot-sparkle"
              style={{ animationDelay: '300ms' }}
            />
            <circle cx="19" cy="116" r="2.6" fill="#38bdf8" className="animate-audio-waves" />
            <circle
              cx="143"
              cy="132"
              r="2.4"
              fill="#34d399"
              className="animate-audio-waves"
              style={{ animationDelay: '250ms' }}
            />
          </g>
        )}
      </svg>
      )}
    </div>
  );
};

