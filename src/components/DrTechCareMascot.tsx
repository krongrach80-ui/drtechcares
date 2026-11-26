import React, { useId } from 'react';

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
 * Pure inline SVG Clinical Rehab Mascot ("Nong MorTech" — Dr.Tech.Care Sea Otter) with zero Unicode emojis.
 * Unified visually with the interactive MorTechMascot companion: warm brown-and-cream sea otter wearing
 * a mint-green headband and clinical scarf with a heart-cross emblem, holding a resistance exercise band,
 * and gesturing according to pose with 60FPS GPU float, blinking eyes, acoustic waves, and sparkles.
 */
export function DrTechCareMascot({
  pose = 'explaining',
  isSpeaking = false,
  size = 'sm',
  className = '',
}: DrTechCareMascotProps) {
  const uid = useId().replace(/:/g, '');
  const px = SIZE_DIMENSIONS[size];
  const isJoyful = pose === 'celebrating' || pose === 'greeting';

  const leftArmTransform =
    pose === 'celebrating'
      ? 'rotate(16 44 80)'
      : pose === 'greeting'
        ? 'rotate(6 44 80)'
        : 'rotate(0 44 80)';

  const rightArmTransform =
    pose === 'celebrating'
      ? 'rotate(-18 88 78)'
      : pose === 'greeting'
        ? 'rotate(-12 88 78)'
        : pose === 'encouraging'
          ? 'rotate(-8 88 78)'
          : pose === 'explaining' || pose === 'speaking'
            ? 'rotate(-4 88 78)'
            : 'rotate(8 88 78)';

  return (
    <div
      className={`inline-flex items-center justify-center shrink-0 select-none ${className}`}
      data-testid="drtechcare-mascot"
      data-character="mortech-sea-otter"
      data-pose={pose}
      data-speaking={isSpeaking ? 'true' : 'false'}
    >
      <svg
        width={px}
        height={px}
        viewBox="0 0 132 132"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        role="img"
        aria-label="น้องหมอเทค นากทะเลผู้ช่วยกายภาพบำบัด Dr.Tech.Care"
        className="overflow-visible"
      >
        <defs>
          <radialGradient id={`aura-${uid}`} cx="50%" cy="48%" r="52%">
            <stop offset="0%" stopColor={isJoyful ? '#fef3c7' : '#ecfdf5'} stopOpacity="0.96" />
            <stop offset="68%" stopColor="#ecfdf5" stopOpacity="0.94" />
            <stop offset="100%" stopColor="#d1fae5" stopOpacity="0.72" />
          </radialGradient>
          <linearGradient id={`fur-${uid}`} x1="66" y1="18" x2="66" y2="116" gradientUnits="userSpaceOnUse">
            <stop offset="0%" stopColor="#B8825C" />
            <stop offset="55%" stopColor="#9C6644" />
            <stop offset="100%" stopColor="#825034" />
          </linearGradient>
          <linearGradient id={`furDark-${uid}`} x1="30" y1="60" x2="100" y2="120" gradientUnits="userSpaceOnUse">
            <stop offset="0%" stopColor="#8A5638" />
            <stop offset="100%" stopColor="#6D4128" />
          </linearGradient>
          <linearGradient id={`cream-${uid}`} x1="66" y1="38" x2="66" y2="108" gradientUnits="userSpaceOnUse">
            <stop offset="0%" stopColor="#FFF8EC" />
            <stop offset="100%" stopColor="#F2DFC2" />
          </linearGradient>
          <linearGradient id={`mint-${uid}`} x1="34" y1="22" x2="98" y2="80" gradientUnits="userSpaceOnUse">
            <stop offset="0%" stopColor="#34D399" />
            <stop offset="55%" stopColor="#10B981" />
            <stop offset="100%" stopColor="#0D9488" />
          </linearGradient>
          <linearGradient id={`band-${uid}`} x1="16" y1="74" x2="46" y2="108" gradientUnits="userSpaceOnUse">
            <stop offset="0%" stopColor="#5EEAD4" />
            <stop offset="100%" stopColor="#0EA5E9" />
          </linearGradient>
        </defs>

        {/* Soft clinical emerald aura */}
        <circle
          cx="66"
          cy="68"
          r="58"
          fill={`url(#aura-${uid})`}
          stroke={isJoyful ? '#34d399' : '#a7f3d0'}
          strokeWidth="2.4"
        />

        {/* Soft floor contact shadow */}
        <ellipse cx="66" cy="121" rx="28" ry="4.8" fill="#0F4C81" opacity="0.13" />

        {/* Celebratory / Greeting Sparkle Stars (Pure SVG Geometry) */}
        {isJoyful && (
          <g aria-hidden="true">
            <path
              d="M17 26 L19.2 31.3 L24.5 33.5 L19.2 35.7 L17 41 L14.8 35.7 L9.5 33.5 L14.8 31.3 Z"
              fill="#f59e0b"
              className="game-star-twinkle"
            />
            <path
              d="M114 20 L115.8 24.2 L120 26 L115.8 27.8 L114 32 L112.2 27.8 L108 26 L112.2 24.2 Z"
              fill="#10b981"
              className="game-star-twinkle"
              style={{ animationDelay: '0.55s' }}
            />
            <circle
              cx="116"
              cy="92"
              r="3.2"
              fill="#38bdf8"
              className="game-star-twinkle"
              style={{ animationDelay: '1.1s' }}
            />
            <polygon
              points="15,92 18.5,95.5 15,99 11.5,95.5"
              fill="#10b981"
              className="game-star-twinkle"
              style={{ animationDelay: '0.85s' }}
            />
          </g>
        )}

        {/* Acoustic speaking waves */}
        {isSpeaking && (
          <g stroke="#10b981" strokeWidth="2.6" strokeLinecap="round" className="game-wave-pulse">
            <path d="M106 41 C112 46 112 54 106 59" opacity="0.9">
              <animate attributeName="opacity" values="0.25;1;0.25" dur="0.9s" repeatCount="indefinite" />
            </path>
            <path d="M113 35 C121 44 121 56 113 65" opacity="0.65">
              <animate attributeName="opacity" values="0.15;0.9;0.15" dur="0.9s" begin="0.25s" repeatCount="indefinite" />
            </path>
          </g>
        )}

        {/* Animated Floating Nong MorTech Sea Otter */}
        <g className="game-mascot-float">
          {/* Otter Tail */}
          <path
            d="M44 96 C26 95 14 88 12 78 C10 71 18 69 25 75 C31 80 37 83 47 85 Z"
            fill={`url(#furDark-${uid})`}
            stroke="#5C3620"
            strokeWidth="2.2"
            strokeLinejoin="round"
          />
          <path
            d="M22 77 C28 81 35 84 42 87"
            stroke="#C89570"
            strokeWidth="2"
            strokeLinecap="round"
            opacity="0.65"
          />

          {/* Hind Paws */}
          <g>
            <ellipse
              cx="52"
              cy="114"
              rx="10.5"
              ry="6.2"
              fill={`url(#furDark-${uid})`}
              stroke="#5C3620"
              strokeWidth="2"
            />
            <circle cx="46.5" cy="115" r="1.4" fill="#E9CFAF" />
            <circle cx="51.5" cy="116.5" r="1.4" fill="#E9CFAF" />
            <circle cx="56" cy="115.5" r="1.4" fill="#E9CFAF" />

            <ellipse
              cx="80"
              cy="114"
              rx="10.5"
              ry="6.2"
              fill={`url(#furDark-${uid})`}
              stroke="#5C3620"
              strokeWidth="2"
            />
            <circle cx="75.5" cy="115.5" r="1.4" fill="#E9CFAF" />
            <circle cx="80" cy="116.5" r="1.4" fill="#E9CFAF" />
            <circle cx="85" cy="115" r="1.4" fill="#E9CFAF" />
          </g>

          {/* Left Arm Holding Resistance Exercise Band */}
          <g transform={leftArmTransform}>
            <path
              d="M32 82 C16 74 12 96 27 104 C36 109 42 98 35 88"
              stroke={`url(#band-${uid})`}
              strokeWidth="4.6"
              strokeLinecap="round"
              fill="none"
            />
            <path
              d="M31 84 C20 79 17 94 27 100"
              stroke="#CCFBF1"
              strokeWidth="1.6"
              strokeLinecap="round"
              fill="none"
              opacity="0.85"
            />
            <rect
              x="23"
              y="80"
              width="11"
              height="5.5"
              rx="2.75"
              transform="rotate(-22 28.5 82.75)"
              fill="#0F4C81"
            />
            <path
              d="M44 76 C36 78 31 82 32 88 C33 93 40 93 46 89"
              fill={`url(#fur-${uid})`}
              stroke="#5C3620"
              strokeWidth="2.1"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
            <ellipse cx="34" cy="87" rx="5.5" ry="4.5" fill="#8A5638" stroke="#5C3620" strokeWidth="1.8" />
          </g>

          {/* Warm Brown Torso & Cream Belly */}
          <ellipse
            cx="66"
            cy="90"
            rx="25"
            ry="23.5"
            fill={`url(#fur-${uid})`}
            stroke="#5C3620"
            strokeWidth="2.3"
          />
          <ellipse cx="66" cy="93" rx="16.5" ry="16.8" fill={`url(#cream-${uid})`} />

          {/* Mint-Green Clinical Scarf + Medical Heart-Cross Emblem */}
          <path
            d="M43 68 C51 76 81 76 89 68 C91 74 87 80 66 81 C45 80 41 74 43 68 Z"
            fill={`url(#mint-${uid})`}
            stroke="#065F46"
            strokeWidth="1.9"
            strokeLinejoin="round"
          />
          <path
            d="M73 76 L83 77 L88 94 C83 96 78 96 74 93 Z"
            fill={`url(#mint-${uid})`}
            stroke="#065F46"
            strokeWidth="1.8"
            strokeLinejoin="round"
          />
          <circle cx="66" cy="76.5" r="7.5" fill="#FFFFFF" stroke="#0D9488" strokeWidth="1.7" />
          <path
            d="M66 81.2 C66 81.2 60.8 78.2 60.8 74.9 C60.8 73.2 62.1 72 63.7 72 C64.7 72 65.5 72.5 66 73.3 C66.5 72.5 67.3 72 68.3 72 C69.9 72 71.2 73.2 71.2 74.9 C71.2 78.2 66 81.2 66 81.2 Z"
            fill="#10B981"
          />
          <rect x="65.15" y="73.9" width="1.7" height="4.4" rx="0.7" fill="#FFFFFF" />
          <rect x="63.8" y="75.25" width="4.4" height="1.7" rx="0.7" fill="#FFFFFF" />

          {/* Right Waving / Gesturing Arm */}
          <g transform={rightArmTransform}>
            <path
              d="M86 76 C95 72 103 65 107 58 C110 54 115 57 113 63 C109 73 99 83 87 86 Z"
              fill={`url(#fur-${uid})`}
              stroke="#5C3620"
              strokeWidth="2.1"
              strokeLinejoin="round"
            />
            <circle cx="110" cy="58" r="5.5" fill="#9C6644" stroke="#5C3620" strokeWidth="1.9" />
            <circle cx="109" cy="57.5" r="2.2" fill="#F2DFC2" />
          </g>

          {/* Rounded Sea Otter Ears */}
          <circle cx="40" cy="34" r="8.5" fill={`url(#furDark-${uid})`} stroke="#5C3620" strokeWidth="2.1" />
          <circle cx="40" cy="34.5" r="4.2" fill="#F2DFC2" />
          <circle cx="92" cy="34" r="8.5" fill={`url(#furDark-${uid})`} stroke="#5C3620" strokeWidth="2.1" />
          <circle cx="92" cy="34.5" r="4.2" fill="#F2DFC2" />

          {/* Sea Otter Head & Cream Facial Mask */}
          <ellipse
            cx="66"
            cy="47"
            rx="28"
            ry="24"
            fill={`url(#fur-${uid})`}
            stroke="#5C3620"
            strokeWidth="2.3"
          />
          <ellipse cx="66" cy="52" rx="22" ry="17" fill={`url(#cream-${uid})`} />

          {/* Mint-Green Headband with Medical Heart-Cross Emblem */}
          <path
            d="M39 39 C48 31 84 31 93 39 C94 34 90 27 66 26 C42 27 38 34 39 39 Z"
            fill={`url(#mint-${uid})`}
            stroke="#065F46"
            strokeWidth="1.9"
            strokeLinejoin="round"
          />
          <circle cx="66" cy="32.5" r="7.4" fill="#FFFFFF" stroke="#0D9488" strokeWidth="1.7" />
          <path
            d="M66 37.1 C66 37.1 61 34.2 61 31.1 C61 29.5 62.2 28.4 63.7 28.4 C64.7 28.4 65.5 28.9 66 29.7 C66.5 28.9 67.3 28.4 68.3 28.4 C69.8 28.4 71 29.5 71 31.1 C71 34.2 66 37.1 66 37.1 Z"
            fill="#10B981"
          />
          <rect x="65.2" y="30.2" width="1.6" height="4.2" rx="0.6" fill="#FFFFFF" />
          <rect x="63.9" y="31.5" width="4.2" height="1.6" rx="0.6" fill="#FFFFFF" />

          {/* Expressive Eyes (Happy Arcs when Celebrating, Blinking Glossy Eyes Otherwise) */}
          {pose === 'celebrating' ? (
            <g stroke="#2A1A10" strokeWidth="3.1" strokeLinecap="round" fill="none">
              <path d="M51 49 Q55.5 43.5 60 49" />
              <path d="M72 49 Q76.5 43.5 81 49" />
            </g>
          ) : (
            <g className="game-eye-blink">
              <circle cx="55.5" cy="48.5" r="4.3" fill="#2A1A10" />
              <circle cx="76.5" cy="48.5" r="4.3" fill="#2A1A10" />
              <circle cx="57" cy="46.8" r="1.5" fill="#FFFFFF" />
              <circle cx="78" cy="46.8" r="1.5" fill="#FFFFFF" />
            </g>
          )}

          {/* Warm Rosy Cheeks */}
          <ellipse cx="47.5" cy="54.5" rx="4.2" ry="2.4" fill="#FB7185" opacity={isJoyful ? '0.48' : '0.34'} />
          <ellipse cx="84.5" cy="54.5" rx="4.2" ry="2.4" fill="#FB7185" opacity={isJoyful ? '0.48' : '0.34'} />

          {/* Cute Otter Nose */}
          <path
            d="M61.5 52.5 C61.5 50.8 70.5 50.8 70.5 52.5 C70.5 54.8 68 56.8 66 56.8 C64 56.8 61.5 54.8 61.5 52.5 Z"
            fill="#3E2414"
          />
          <ellipse cx="64.8" cy="52.4" rx="1.5" ry="0.8" fill="#A67C5E" opacity="0.65" />

          {/* Expressive Mouth (Animated Speaking vs Joyful Smile) */}
          {isSpeaking ? (
            <ellipse cx="66" cy="61" rx="5.2" ry="2.8" fill="#E11D48" opacity="0.82">
              <animate attributeName="ry" values="1.5;3.3;1.5" dur="0.45s" repeatCount="indefinite" />
            </ellipse>
          ) : isJoyful ? (
            <path
              d="M58.5 58.8 Q66 65.5 73.5 58.8"
              stroke="#5C3620"
              strokeWidth="2.3"
              strokeLinecap="round"
              fill="none"
            />
          ) : (
            <path
              d="M59.5 59 C62 62.2 64.5 61.5 66 59.8 C67.5 61.5 70 62.2 72.5 59"
              stroke="#5C3620"
              strokeWidth="2.1"
              strokeLinecap="round"
              strokeLinejoin="round"
              fill="none"
            />
          )}

          {/* Crisp Otter Whiskers */}
          <g stroke="#7A4B2E" strokeWidth="1.5" strokeLinecap="round" opacity="0.72">
            <line x1="45" y1="54" x2="36" y2="52.5" />
            <line x1="45" y1="57" x2="36.5" y2="58.5" />
            <line x1="87" y1="54" x2="96" y2="52.5" />
            <line x1="87" y1="57" x2="95.5" y2="58.5" />
          </g>
        </g>
      </svg>
    </div>
  );
}
