import React from 'react';

export interface DrTechCareMascotProps {
  pose?: 'idle' | 'explaining' | 'celebrating';
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
 * Follows the Clinical White, Medical Blue, and Emerald Green palette.
 */
export function DrTechCareMascot({
  pose = 'explaining',
  isSpeaking = false,
  size = 'sm',
  className = '',
}: DrTechCareMascotProps) {
  const px = SIZE_DIMENSIONS[size];
  const rightArmTransform =
    pose === 'celebrating'
      ? 'rotate(-36 86 68)'
      : pose === 'explaining'
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
        </defs>

        {/* Soft clinical aura */}
        <circle cx="60" cy="62" r="54" fill="#ecfdf5" stroke="#a7f3d0" strokeWidth="2" />

        {/* Acoustic speaking waves */}
        {isSpeaking && (
          <g stroke="#10b981" strokeWidth="2.5" strokeLinecap="round">
            <path d="M98 40 C104 45 104 53 98 58" opacity="0.9">
              <animate attributeName="opacity" values="0.25;1;0.25" dur="0.9s" repeatCount="indefinite" />
            </path>
            <path d="M105 34 C113 43 113 55 105 64" opacity="0.65">
              <animate attributeName="opacity" values="0.15;0.9;0.15" dur="0.9s" begin="0.25s" repeatCount="indefinite" />
            </path>
          </g>
        )}

        {/* Antenna / Sensor */}
        <line x1="60" y1="16" x2="60" y2="9" stroke="#0284c7" strokeWidth="3.5" strokeLinecap="round" />
        <circle cx="60" cy="8" r="4" fill="#10b981" />

        {/* Left Arm */}
        <rect x="22" y="68" width="12" height="28" rx="6" fill="#0284c7" transform="rotate(12 28 68)" />

        {/* Right Gesturing Arm */}
        <g transform={rightArmTransform}>
          <rect x="86" y="66" width="12" height="28" rx="6" fill="#10b981" />
          <circle cx="92" cy="95" r="5.5" fill="#ffffff" stroke="#10b981" strokeWidth="2" />
        </g>

        {/* White Clinical Coat Torso */}
        <rect x="33" y="64" width="54" height="42" rx="18" fill="url(#mascotCoatGrad)" stroke="#0f4c81" strokeWidth="2.5" />
        {/* Medical Emerald Cross Emblem on Coat */}
        <rect x="57.5" y="75" width="5" height="15" rx="2" fill="#10b981" />
        <rect x="52.5" y="80" width="15" height="5" rx="2" fill="#10b981" />

        {/* Robot Head */}
        <rect x="28" y="18" width="64" height="46" rx="20" fill="#ffffff" stroke="#0f4c81" strokeWidth="2.8" />
        {/* Ear pods */}
        <rect x="21" y="33" width="7" height="16" rx="3.5" fill="#10b981" />
        <rect x="92" y="33" width="7" height="16" rx="3.5" fill="#10b981" />

        {/* Visor Faceplate */}
        <rect x="35" y="26" width="50" height="29" rx="12" fill="url(#mascotVisorGrad)" />

        {/* Friendly Expressive Eyes */}
        <circle cx="48" cy="39" r="4.2" fill="#34d399" />
        <circle cx="72" cy="39" r="4.2" fill="#34d399" />
        <circle cx="49.2" cy="37.8" r="1.4" fill="#ffffff" />
        <circle cx="73.2" cy="37.8" r="1.4" fill="#ffffff" />

        {/* Mouth (pulses cleanly while speaking) */}
        {isSpeaking ? (
          <ellipse cx="60" cy="48.5" rx="6" ry="3.2" fill="#6ee7b7">
            <animate attributeName="ry" values="1.6;3.8;1.6" dur="0.45s" repeatCount="indefinite" />
          </ellipse>
        ) : (
          <path d="M52 47.5 Q60 53 68 47.5" stroke="#6ee7b7" strokeWidth="2.6" strokeLinecap="round" fill="none" />
        )}
      </svg>
    </div>
  );
}

