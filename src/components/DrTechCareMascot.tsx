import React from 'react';

export interface DrTechCareMascotProps {
  pose?: 'idle' | 'explaining' | 'celebrating' | 'encouraging' | 'speaking' | 'greeting';
  isSpeaking?: boolean;
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

const SIZE_DIMENSIONS: Record<NonNullable<DrTechCareMascotProps['size']>, number> = {
  sm: 76,
  md: 104,
  lg: 136,
};

/**
 * Pure inline SVG Clinical Rehab Mascot for Dr.Tech.Care (zero Unicode emojis).
 * Follows the Clinical White, Medical Blue, and Emerald Green palette with
 * playful 60FPS GPU floating, blinking eyes, glowing beacon, and star sparkles.
 */
export function DrTechCareMascot({
  pose = 'explaining',
  isSpeaking = false,
  size = 'sm',
  className = '',
}: DrTechCareMascotProps) {
  const px = SIZE_DIMENSIONS[size];
  const isJoyful = pose === 'celebrating' || pose === 'greeting';
  const leftArmTransform =
    pose === 'celebrating'
      ? 'rotate(38 28 68)'
      : pose === 'greeting'
        ? 'rotate(18 28 68)'
        : 'rotate(12 28 68)';
  const rightArmTransform =
    pose === 'celebrating'
      ? 'rotate(-44 86 68)'
      : pose === 'greeting'
        ? 'rotate(-36 86 68)'
        : pose === 'encouraging'
          ? 'rotate(-26 86 68)'
          : pose === 'explaining' || pose === 'speaking'
            ? 'rotate(-16 86 68)'
            : 'rotate(6 86 68)';

  return (
    <div
      className={`inline-flex items-center justify-center shrink-0 select-none ${className}`}
      data-testid="drtechcare-mascot"
      data-pose={pose}
      data-speaking={isSpeaking ? 'true' : 'false'}
    >
      <svg
        width={px}
        height={px}
        viewBox="0 0 120 120"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        role="img"
        aria-label="ผู้ช่วยแพทย์ดิจิทัล Dr.Tech.Care"
        className="overflow-visible"
      >
        <defs>
          <linearGradient id="mascotCoatGrad" x1="28" y1="60" x2="92" y2="114" gradientUnits="userSpaceOnUse">
            <stop offset="0%" stopColor="#ffffff" />
            <stop offset="100%" stopColor="#e0f2fe" />
          </linearGradient>
          <linearGradient id="mascotVisorGrad" x1="34" y1="28" x2="86" y2="54" gradientUnits="userSpaceOnUse">
            <stop offset="0%" stopColor="#0f4c81" />
            <stop offset="100%" stopColor="#0284c7" />
          </linearGradient>
          <radialGradient id="mascotAuraGrad" cx="50%" cy="48%" r="52%">
            <stop offset="0%" stopColor={isJoyful ? '#fef3c7' : '#ecfdf5'} stopOpacity="0.95" />
            <stop offset="68%" stopColor="#ecfdf5" stopOpacity="0.95" />
            <stop offset="100%" stopColor="#d1fae5" stopOpacity="0.7" />
          </radialGradient>
        </defs>

        {/* Soft clinical aura */}
        <circle
          cx="60"
          cy="62"
          r="54"
          fill="url(#mascotAuraGrad)"
          stroke={isJoyful ? '#34d399' : '#a7f3d0'}
          strokeWidth="2.2"
        />

        {/* Soft floating shadow */}
        <ellipse cx="60" cy="109" rx="22" ry="4" fill="#0f4c81" opacity="0.12" />

        {/* Celebratory / Greeting Sparkle Stars (Pure SVG Geometry) */}
        {isJoyful && (
          <g aria-hidden="true">
            <path
              d="M17 24 L19.2 29.3 L24.5 31.5 L19.2 33.7 L17 39 L14.8 33.7 L9.5 31.5 L14.8 29.3 Z"
              fill="#f59e0b"
              className="game-star-twinkle"
            />
            <path
              d="M103 18 L104.8 22.2 L109 24 L104.8 25.8 L103 30 L101.2 25.8 L97 24 L101.2 22.2 Z"
              fill="#10b981"
              className="game-star-twinkle"
              style={{ animationDelay: '0.55s' }}
            />
            <circle
              cx="104"
              cy="84"
              r="3.2"
              fill="#38bdf8"
              className="game-star-twinkle"
              style={{ animationDelay: '1.1s' }}
            />
            <polygon
              points="15,82 18.5,85.5 15,89 11.5,85.5"
              fill="#10b981"
              className="game-star-twinkle"
              style={{ animationDelay: '0.85s' }}
            />
          </g>
        )}

        {/* Acoustic speaking waves */}
        {isSpeaking && (
          <g stroke="#10b981" strokeWidth="2.5" strokeLinecap="round" className="game-wave-pulse">
            <path d="M98 40 C104 45 104 53 98 58" opacity="0.9">
              <animate attributeName="opacity" values="0.25;1;0.25" dur="0.9s" repeatCount="indefinite" />
            </path>
            <path d="M105 34 C113 43 113 55 105 64" opacity="0.65">
              <animate attributeName="opacity" values="0.15;0.9;0.15" dur="0.9s" begin="0.25s" repeatCount="indefinite" />
            </path>
          </g>
        )}

        {/* Animated Floating Mascot Body */}
        <g className="game-mascot-float">
          {/* Antenna / Sensor with Beacon Glow */}
          <line x1="60" y1="16" x2="60" y2="9" stroke="#0284c7" strokeWidth="3.5" strokeLinecap="round" />
          <circle cx="60" cy="8" r="6.8" fill={isJoyful ? '#f59e0b' : '#10b981'} opacity="0.22" className="game-wave-pulse" />
          <circle cx="60" cy="8" r="4" fill={isJoyful ? '#f59e0b' : '#10b981'} />
          <circle cx="58.8" cy="6.8" r="1.3" fill="#ffffff" opacity="0.85" />

          {/* Left Arm */}
          <g transform={leftArmTransform}>
            <rect x="22" y="68" width="12" height="28" rx="6" fill="#0284c7" />
            {pose === 'celebrating' && (
              <circle cx="28" cy="96" r="5" fill="#ffffff" stroke="#0284c7" strokeWidth="2" />
            )}
          </g>

          {/* Right Gesturing Arm */}
          <g transform={rightArmTransform}>
            <rect x="86" y="66" width="12" height="28" rx="6" fill="#10b981" />
            <circle cx="92" cy="95" r="5.5" fill="#ffffff" stroke="#10b981" strokeWidth="2" />
          </g>

          {/* White Clinical Coat Torso */}
          <rect x="33" y="64" width="54" height="42" rx="18" fill="url(#mascotCoatGrad)" stroke="#0f4c81" strokeWidth="2.5" />
          {/* Glossy Collar Accent */}
          <path d="M46 65 L60 73 L74 65" stroke="#38bdf8" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" fill="none" opacity="0.6" />
          {/* Medical Emerald Cross Emblem on Coat */}
          <rect x="57.5" y="75" width="5" height="15" rx="2" fill="#10b981" />
          <rect x="52.5" y="80" width="15" height="5" rx="2" fill="#10b981" />

          {/* Robot Head */}
          <rect x="28" y="18" width="64" height="46" rx="20" fill="#ffffff" stroke="#0f4c81" strokeWidth="2.8" />
          {/* Top Head Gloss Reflection */}
          <path d="M42 22 H74" stroke="#bae6fd" strokeWidth="2.2" strokeLinecap="round" opacity="0.7" />
          {/* Ear pods */}
          <rect x="21" y="33" width="7" height="16" rx="3.5" fill="#10b981" />
          <rect x="92" y="33" width="7" height="16" rx="3.5" fill="#10b981" />

          {/* Visor Faceplate */}
          <rect x="35" y="26" width="50" height="29" rx="12" fill="url(#mascotVisorGrad)" />
          {/* Subtle visor glass sheen */}
          <path d="M40 30 Q60 27 80 30" stroke="#7dd3fc" strokeWidth="1.5" strokeLinecap="round" opacity="0.35" />

          {/* Warm Playful Cheek Glow */}
          <ellipse cx="42" cy="45" rx="3.4" ry="2" fill="#fb7185" opacity={isJoyful ? '0.45' : '0.26'} />
          <ellipse cx="78" cy="45" rx="3.4" ry="2" fill="#fb7185" opacity={isJoyful ? '0.45' : '0.26'} />

          {/* Friendly Expressive Eyes */}
          {pose === 'celebrating' ? (
            <g stroke="#34d399" strokeWidth="3.2" strokeLinecap="round" fill="none">
              <path d="M43 40 Q48 34 53 40" />
              <path d="M67 40 Q72 34 77 40" />
            </g>
          ) : (
            <g className="game-eye-blink">
              <circle cx="48" cy="39" r="4.2" fill="#34d399" />
              <circle cx="72" cy="39" r="4.2" fill="#34d399" />
              <circle cx="49.2" cy="37.8" r="1.4" fill="#ffffff" />
              <circle cx="73.2" cy="37.8" r="1.4" fill="#ffffff" />
            </g>
          )}

          {/* Mouth (pulses cleanly while speaking) */}
          {isSpeaking ? (
            <ellipse cx="60" cy="48.5" rx="6" ry="3.2" fill="#6ee7b7">
              <animate attributeName="ry" values="1.6;3.8;1.6" dur="0.45s" repeatCount="indefinite" />
            </ellipse>
          ) : isJoyful ? (
            <path d="M51 46.5 Q60 54.5 69 46.5" stroke="#fde68a" strokeWidth="2.8" strokeLinecap="round" fill="none" />
          ) : (
            <path d="M52 47.5 Q60 53 68 47.5" stroke="#6ee7b7" strokeWidth="2.6" strokeLinecap="round" fill="none" />
          )}
        </g>
      </svg>
    </div>
  );
}
