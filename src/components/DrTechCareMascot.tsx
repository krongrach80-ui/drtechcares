import React, { useId } from 'react';
import { MotionConfig, motion } from 'framer-motion';

export interface DrTechCareMascotProps {
  pose?: 'idle' | 'explaining' | 'celebrating' | 'encouraging' | 'speaking' | 'greeting';
  isSpeaking?: boolean;
  size?: 'sm' | 'md' | 'lg';
  /** When true, animates Nong MorTech popping up out of a centered 2.5D burrow hole and waving hello. */
  popFromBurrow?: boolean;
  className?: string;
}

const SIZE_DIMENSIONS: Record<NonNullable<DrTechCareMascotProps['size']>, number> = {
  sm: 76,
  md: 104,
  lg: 136,
};

const BURROW_STAGE_DIMENSIONS: Record<
  NonNullable<DrTechCareMascotProps['size']>,
  { width: number; height: number }
> = {
  sm: { width: 128, height: 104 },
  md: { width: 158, height: 124 },
  lg: { width: 186, height: 146 },
};

/**
 * Pure inline SVG Clinical Rehab Mascot ("Nong MorTech" — Dr.Tech.Care Sea Otter) with zero Unicode emojis.
 * Unified visually with the interactive MorTechMascot companion: warm brown-and-cream sea otter wearing
 * a mint-green headband and clinical scarf with a heart-cross emblem, holding a resistance exercise band,
 * and gesturing according to pose (or popping joyfully out of a burrow hole and waving hello when `popFromBurrow` is enabled).
 */
export function DrTechCareMascot({
  pose = 'explaining',
  isSpeaking = false,
  size = 'sm',
  popFromBurrow = false,
  className = '',
}: DrTechCareMascotProps) {
  const uid = useId().replace(/:/g, '');
  const px = SIZE_DIMENSIONS[size];
  const isJoyful = pose === 'celebrating' || pose === 'greeting';

  if (popFromBurrow) {
    const dims = BURROW_STAGE_DIMENSIONS[size];
    return (
      <MotionConfig reducedMotion="never">
        <div
          className={`inline-flex items-center justify-center shrink-0 select-none mx-auto ${className}`}
          data-testid="drtechcare-mascot"
          data-character="mortech-sea-otter"
          data-pose={pose}
          data-burrow="true"
          data-speaking={isSpeaking ? 'true' : 'false'}
        >
          <svg
            width={dims.width}
            height={dims.height}
            viewBox="0 0 168 136"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
            role="img"
            aria-label="น้องหมอเทคโผล่ขึ้นมาจากหลุมและโบกมือต้อนรับ"
            className="block overflow-visible"
            data-testid="mascot-burrow-popup"
          >
            <defs>
              <linearGradient id={`bfur-${uid}`} x1="84" y1="22" x2="84" y2="120" gradientUnits="userSpaceOnUse">
                <stop offset="0%" stopColor="#B8825C" />
                <stop offset="55%" stopColor="#9C6644" />
                <stop offset="100%" stopColor="#825034" />
              </linearGradient>
              <linearGradient id={`bfurDark-${uid}`} x1="48" y1="62" x2="118" y2="122" gradientUnits="userSpaceOnUse">
                <stop offset="0%" stopColor="#8A5638" />
                <stop offset="100%" stopColor="#6D4128" />
              </linearGradient>
              <linearGradient id={`bcream-${uid}`} x1="84" y1="42" x2="84" y2="112" gradientUnits="userSpaceOnUse">
                <stop offset="0%" stopColor="#FFF8EC" />
                <stop offset="100%" stopColor="#F2DFC2" />
              </linearGradient>
              <linearGradient id={`bmint-${uid}`} x1="52" y1="26" x2="116" y2="86" gradientUnits="userSpaceOnUse">
                <stop offset="0%" stopColor="#34D399" />
                <stop offset="55%" stopColor="#10B981" />
                <stop offset="100%" stopColor="#0D9488" />
              </linearGradient>
              <linearGradient id={`bband-${uid}`} x1="34" y1="78" x2="64" y2="112" gradientUnits="userSpaceOnUse">
                <stop offset="0%" stopColor="#5EEAD4" />
                <stop offset="100%" stopColor="#0EA5E9" />
              </linearGradient>
              <radialGradient id={`bhole-${uid}`} cx="50%" cy="42%" r="58%">
                <stop offset="0%" stopColor="#052E2B" />
                <stop offset="68%" stopColor="#0B4A58" />
                <stop offset="100%" stopColor="#0F766E" />
              </radialGradient>
              <linearGradient id={`brim-${uid}`} x1="24" y1="100" x2="144" y2="124" gradientUnits="userSpaceOnUse">
                <stop offset="0%" stopColor="#A7F3D0" />
                <stop offset="50%" stopColor="#6EE7B7" />
                <stop offset="100%" stopColor="#38BDF8" />
              </linearGradient>
              {/* 2.5D Burrow Opening Mask: clips otter strictly below the curved front lip of the hole */}
              <clipPath id={`bclip-${uid}`}>
                <path d="M0 -24 H168 V110 H137 A53 15 0 0 1 31 110 H0 Z" />
              </clipPath>
            </defs>

            {/* Expanding Welcome Ripple Ring Around Burrow */}
            <motion.ellipse
              cx="84"
              cy="110"
              rx="56"
              ry="16.5"
              stroke="#34D399"
              strokeWidth="2.5"
              fill="none"
              style={{ transformOrigin: '84px 110px', transformBox: 'view-box' }}
              initial={{ scale: 0.45, opacity: 0.85 }}
              animate={{ scale: [0.55, 1.24], opacity: [0.8, 0] }}
              transition={{ duration: 0.95, delay: 0.12, ease: 'easeOut' }}
            />

            {/* Layer 1: Burrow Outer Halo + Deep 3D Cavity */}
            <motion.g
              data-testid="mascot-burrow-hole"
              style={{ transformOrigin: '84px 110px', transformBox: 'view-box' }}
              initial={{ scaleX: 0.2, scaleY: 0.3, opacity: 0 }}
              animate={{ scaleX: 1, scaleY: 1, opacity: 1 }}
              transition={{ duration: 0.3, ease: [0.22, 1, 0.36, 1] }}
            >
              {/* Soft ground shadow around hole */}
              <ellipse cx="84" cy="113" rx="64" ry="18.5" fill="#D1FAE5" opacity="0.85" />
              <ellipse cx="84" cy="111.5" rx="59" ry="16.8" fill={`url(#brim-${uid})`} opacity="0.7" />
              {/* Deep burrow cavity interior */}
              <ellipse
                cx="84"
                cy="110"
                rx="53"
                ry="14.5"
                fill={`url(#bhole-${uid})`}
                stroke="#0D9488"
                strokeWidth="2.2"
              />
              {/* Inner depth ring */}
              <ellipse cx="84" cy="112" rx="44" ry="9.5" fill="#04242A" opacity="0.7" />
            </motion.g>

            {/* Celebratory Sparkle Stars Popping Out When Nong MorTech Emerges */}
            <motion.g
              aria-hidden="true"
              initial={{ opacity: 0, scale: 0.4 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.35, delay: 0.28, ease: 'easeOut' }}
              style={{ transformOrigin: '84px 58px', transformBox: 'view-box' }}
            >
              <path
                d="M27 28 L29.4 33.6 L35 36 L29.4 38.4 L27 44 L24.6 38.4 L19 36 L24.6 33.6 Z"
                fill="#F59E0B"
                className="game-star-twinkle"
              />
              <path
                d="M142 22 L144 26.8 L148.8 28.8 L144 30.8 L142 35.6 L140 30.8 L135.2 28.8 L140 26.8 Z"
                fill="#10B981"
                className="game-star-twinkle"
                style={{ animationDelay: '0.45s' }}
              />
              <circle
                cx="148"
                cy="76"
                r="3.4"
                fill="#38BDF8"
                className="game-star-twinkle"
                style={{ animationDelay: '0.85s' }}
              />
              <polygon
                points="22,78 26,82 22,86 18,82"
                fill="#10B981"
                className="game-star-twinkle"
                style={{ animationDelay: '0.65s' }}
              />
            </motion.g>

            {/* Acoustic speaking waves */}
            {isSpeaking && (
              <g stroke="#10B981" strokeWidth="2.6" strokeLinecap="round" className="game-wave-pulse">
                <path d="M126 43 C132 48 132 56 126 61" opacity="0.9">
                  <animate attributeName="opacity" values="0.25;1;0.25" dur="0.9s" repeatCount="indefinite" />
                </path>
                <path d="M133 37 C141 46 141 58 133 67" opacity="0.65">
                  <animate attributeName="opacity" values="0.15;0.9;0.15" dur="0.9s" begin="0.25s" repeatCount="indefinite" />
                </path>
              </g>
            )}

            {/* Layer 2: Clipped Nong MorTech Sea Otter Emerging From Inside The Burrow */}
            <g clipPath={`url(#bclip-${uid})`}>
              <motion.g
                data-testid="mascot-burrow-otter-pop"
                style={{ transformOrigin: '84px 110px', transformBox: 'view-box' }}
                initial={{ y: 98, scale: 0.84 }}
                animate={{ y: 0, scale: 1 }}
                transition={{
                  type: 'spring',
                  stiffness: 270,
                  damping: 16,
                  mass: 0.9,
                  delay: 0.14,
                }}
              >
                {/* Gentle happy idle bounce after emergence */}
                <g className="game-mascot-float">
                  {/* Left Paw Holding Resistance Exercise Band */}
                  <motion.g
                    style={{ transformOrigin: '62px 84px', transformBox: 'view-box' }}
                    animate={{ rotate: [-4, 7, -4] }}
                    transition={{ delay: 0.4, duration: 1.4, repeat: Infinity, ease: 'easeInOut' }}
                  >
                    <path
                      d="M50 86 C34 78 30 100 45 108 C54 113 60 102 53 92"
                      stroke={`url(#bband-${uid})`}
                      strokeWidth="4.6"
                      strokeLinecap="round"
                      fill="none"
                    />
                    <path
                      d="M49 88 C38 83 35 98 45 104"
                      stroke="#CCFBF1"
                      strokeWidth="1.6"
                      strokeLinecap="round"
                      fill="none"
                      opacity="0.85"
                    />
                    <rect
                      x="41"
                      y="84"
                      width="11"
                      height="5.5"
                      rx="2.75"
                      transform="rotate(-22 46.5 86.75)"
                      fill="#0F4C81"
                    />
                    <path
                      d="M62 80 C54 82 49 86 50 92 C51 97 58 97 64 93"
                      fill={`url(#bfur-${uid})`}
                      stroke="#5C3620"
                      strokeWidth="2.1"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                    <ellipse cx="52" cy="91" rx="5.5" ry="4.5" fill="#8A5638" stroke="#5C3620" strokeWidth="1.8" />
                  </motion.g>

                  {/* Warm Brown Torso & Cream Belly (lower portion tucked naturally inside the burrow) */}
                  <ellipse
                    cx="84"
                    cy="98"
                    rx="26.5"
                    ry="26"
                    fill={`url(#bfur-${uid})`}
                    stroke="#5C3620"
                    strokeWidth="2.3"
                  />
                  <ellipse cx="84" cy="101" rx="17.5" ry="18.5" fill={`url(#bcream-${uid})`} />

                  {/* Mint-Green Clinical Scarf + Medical Heart-Cross Emblem */}
                  <path
                    d="M61 72 C69 80 99 80 107 72 C109 78 105 84 84 85 C63 84 59 78 61 72 Z"
                    fill={`url(#bmint-${uid})`}
                    stroke="#065F46"
                    strokeWidth="1.9"
                    strokeLinejoin="round"
                  />
                  <path
                    d="M91 80 L101 81 L106 98 C101 100 96 100 92 97 Z"
                    fill={`url(#bmint-${uid})`}
                    stroke="#065F46"
                    strokeWidth="1.8"
                    strokeLinejoin="round"
                  />
                  <circle cx="84" cy="80.5" r="7.5" fill="#FFFFFF" stroke="#0D9488" strokeWidth="1.7" />
                  <path
                    d="M84 85.2 C84 85.2 78.8 82.2 78.8 78.9 C78.8 77.2 80.1 76 81.7 76 C82.7 76 83.5 76.5 84 77.3 C84.5 76.5 85.3 76 86.3 76 C87.9 76 89.2 77.2 89.2 78.9 C89.2 82.2 84 85.2 84 85.2 Z"
                    fill="#10B981"
                  />
                  <rect x="83.15" y="77.9" width="1.7" height="4.4" rx="0.7" fill="#FFFFFF" />
                  <rect x="81.8" y="79.25" width="4.4" height="1.7" rx="0.7" fill="#FFFFFF" />

                  {/* Right Arm Raised High & Waving Cheerfully Hello */}
                  <motion.g
                    data-testid="mascot-burrow-waving-arm"
                    style={{ transformOrigin: '106px 82px', transformBox: 'view-box' }}
                    initial={{ rotate: 18 }}
                    animate={{ rotate: [-22, 10, -24, 8, -22] }}
                    transition={{
                      delay: 0.34,
                      duration: 0.92,
                      repeat: Infinity,
                      ease: 'easeInOut',
                    }}
                  >
                    <path
                      d="M104 80 C114 75 123 67 127 58 C130 53 136 56 133 63 C129 75 118 86 105 90 Z"
                      fill={`url(#bfur-${uid})`}
                      stroke="#5C3620"
                      strokeWidth="2.1"
                      strokeLinejoin="round"
                    />
                    <circle cx="130" cy="58" r="5.8" fill="#9C6644" stroke="#5C3620" strokeWidth="1.9" />
                    <circle cx="129" cy="57.5" r="2.3" fill="#F2DFC2" />
                  </motion.g>

                  {/* Rounded Sea Otter Ears */}
                  <circle cx="58" cy="37" r="8.8" fill={`url(#bfurDark-${uid})`} stroke="#5C3620" strokeWidth="2.1" />
                  <circle cx="58" cy="37.5" r="4.3" fill="#F2DFC2" />
                  <circle cx="110" cy="37" r="8.8" fill={`url(#bfurDark-${uid})`} stroke="#5C3620" strokeWidth="2.1" />
                  <circle cx="110" cy="37.5" r="4.3" fill="#F2DFC2" />

                  {/* Sea Otter Head & Cream Facial Mask */}
                  <ellipse
                    cx="84"
                    cy="51"
                    rx="28.5"
                    ry="24.5"
                    fill={`url(#bfur-${uid})`}
                    stroke="#5C3620"
                    strokeWidth="2.3"
                  />
                  <ellipse cx="84" cy="56" rx="22.5" ry="17.2" fill={`url(#bcream-${uid})`} />

                  {/* Mint-Green Sport Headband with Medical Heart-Cross Badge */}
                  <path
                    d="M56.5 43 C66 34.5 102 34.5 111.5 43 C112.5 38 108.5 30.5 84 29.5 C59.5 30.5 55.5 38 56.5 43 Z"
                    fill={`url(#bmint-${uid})`}
                    stroke="#065F46"
                    strokeWidth="1.9"
                    strokeLinejoin="round"
                  />
                  <circle cx="84" cy="36" r="7.5" fill="#FFFFFF" stroke="#0D9488" strokeWidth="1.7" />
                  <path
                    d="M84 40.6 C84 40.6 79 37.7 79 34.6 C79 33 80.2 31.9 81.7 31.9 C82.7 31.9 83.5 32.4 84 33.2 C84.5 32.4 85.3 31.9 86.3 31.9 C87.8 31.9 89 33 89 34.6 C89 37.7 84 40.6 84 40.6 Z"
                    fill="#10B981"
                  />
                  <rect x="83.2" y="33.7" width="1.6" height="4.2" rx="0.6" fill="#FFFFFF" />
                  <rect x="81.9" y="35" width="4.2" height="1.6" rx="0.6" fill="#FFFFFF" />

                  {/* Expressive Bright Eyes */}
                  {pose === 'celebrating' ? (
                    <g stroke="#2A1A10" strokeWidth="3.1" strokeLinecap="round" fill="none">
                      <path d="M69 52.5 Q73.5 47 78 52.5" />
                      <path d="M90 52.5 Q94.5 47 99 52.5" />
                    </g>
                  ) : (
                    <g className="game-eye-blink">
                      <circle cx="73.5" cy="52" r="4.4" fill="#2A1A10" />
                      <circle cx="94.5" cy="52" r="4.4" fill="#2A1A10" />
                      <circle cx="75" cy="50.3" r="1.55" fill="#FFFFFF" />
                      <circle cx="96" cy="50.3" r="1.55" fill="#FFFFFF" />
                    </g>
                  )}

                  {/* Warm Rosy Cheeks */}
                  <ellipse cx="65.5" cy="58.2" rx="4.4" ry="2.5" fill="#FB7185" opacity="0.46" />
                  <ellipse cx="102.5" cy="58.2" rx="4.4" ry="2.5" fill="#FB7185" opacity="0.46" />

                  {/* Cute Otter Nose & Joyful Smile */}
                  <path
                    d="M79.5 56 C79.5 54.3 88.5 54.3 88.5 56 C88.5 58.3 86 60.3 84 60.3 C82 60.3 79.5 58.3 79.5 56 Z"
                    fill="#3E2414"
                  />
                  <ellipse cx="82.8" cy="55.9" rx="1.5" ry="0.8" fill="#A67C5E" opacity="0.65" />
                  {isSpeaking ? (
                    <ellipse cx="84" cy="64.5" rx="5.2" ry="2.8" fill="#E11D48" opacity="0.82">
                      <animate attributeName="ry" values="1.5;3.3;1.5" dur="0.45s" repeatCount="indefinite" />
                    </ellipse>
                  ) : (
                    <path
                      d="M76.5 62.5 Q84 69.2 91.5 62.5"
                      stroke="#5C3620"
                      strokeWidth="2.3"
                      strokeLinecap="round"
                      fill="none"
                    />
                  )}

                  {/* Crisp Otter Whiskers */}
                  <g stroke="#7A4B2E" strokeWidth="1.5" strokeLinecap="round" opacity="0.72">
                    <line x1="63" y1="57.5" x2="54" y2="56" />
                    <line x1="63" y1="60.5" x2="54.5" y2="62" />
                    <line x1="105" y1="57.5" x2="114" y2="56" />
                    <line x1="105" y1="60.5" x2="113.5" y2="62" />
                  </g>
                </g>
              </motion.g>
            </g>

            {/* Layer 3: Front Curved Rim Lip of the Burrow (occludes otter's lower body for true 2.5D depth) */}
            <motion.g
              aria-hidden="true"
              style={{ transformOrigin: '84px 110px', transformBox: 'view-box' }}
              initial={{ scaleX: 0.25, opacity: 0 }}
              animate={{ scaleX: 1, opacity: 1 }}
              transition={{ duration: 0.3, ease: [0.22, 1, 0.36, 1] }}
            >
              <path
                d="M31 110 A53 14.5 0 0 0 137 110"
                stroke="#10B981"
                strokeWidth="4.2"
                strokeLinecap="round"
                fill="none"
              />
              <path
                d="M38 115.5 A47 11.5 0 0 0 130 115.5"
                stroke="#A7F3D0"
                strokeWidth="2.2"
                strokeLinecap="round"
                fill="none"
                opacity="0.9"
              />
            </motion.g>
          </svg>
        </div>
      </MotionConfig>
    );
  }

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
