import React, { memo, useCallback, useEffect, useId, useMemo, useRef, useState } from 'react';
import { AnimatePresence, motion, useReducedMotion } from 'framer-motion';
import { Calendar, Hand, HeartPulse, MessageCircle, Sparkles, X } from 'lucide-react';

export type MorTechMascotMode = 'standby' | 'face-login' | 'rehab' | 'patient-portal';
export type MorTechMascotLang = 'th' | 'en' | 'cn';
export type MorTechMascotPhase = 'walking' | 'waving';
export type MorTechMascotIconKind = 'hand' | 'sparkles' | 'heart-pulse' | 'calendar';

export interface MorTechMascotMessage {
  id: string;
  text: string;
  icon: MorTechMascotIconKind;
}

export interface MorTechMascotProps {
  /** Active screen context that drives Nong MorTech's speech bubble copy and patrol bounds. */
  mode: MorTechMascotMode;
  /** Optional display language matching Dr.Tech.Care clinical i18n (defaults to Thai). */
  lang?: MorTechMascotLang;
  /** Optional override text (e.g., live clinical encouragement) shown inside the speech bubble. */
  customMessage?: string | null;
  /** Hide companion smoothly when blocking modals or full-screen intros are active. */
  hidden?: boolean;
  /** Whether the speech bubble starts expanded (defaults to true). */
  defaultSpeechVisible?: boolean;
  /** Optional extra Tailwind classes appended to the fixed companion container. */
  className?: string;
}

export interface MascotPatrolBounds {
  minX: number;
  midX: number;
  maxX: number;
  travelDistance: number;
  walkDurationSec: number;
  canPatrol: boolean;
}

export const MOR_TECH_MASCOT_WIDTH_PX = 92;
export const MOR_TECH_MASCOT_HEIGHT_PX = 96;

export const MOR_TECH_MASCOT_MESSAGES: Record<MorTechMascotMode, readonly MorTechMascotMessage[]> = {
  standby: [
    {
      id: 'standby-wake-1',
      text: 'สวัสดีครับ ยินดีต้อนรับสู่ หมอเทค ชูมือ 2 ข้างค้างไว้ 3 วินาที เพื่อเริ่มใช้งานได้เลยครับ',
      icon: 'hand',
    },
    {
      id: 'standby-wake-2',
      text: 'พร้อมฟื้นฟูร่างกายวันนี้หรือยังครับ ยกมือทั้ง 2 ข้างหน้ากล้องเพื่อปลุกตู้ได้เลยครับ',
      icon: 'sparkles',
    },
  ],
  'face-login': [
    {
      id: 'face-login-1',
      text: 'มองตรงมาที่กล้องนะครับ ระบบกำลังยืนยันตัวตนให้คุณอยู่ครับ',
      icon: 'sparkles',
    },
    {
      id: 'face-login-2',
      text: 'หันหน้าตรงเข้าหากล้องสักครู่นะครับ น้องหมอเทคกำลังพาเข้าสู่หน้าฝึกครับ',
      icon: 'heart-pulse',
    },
  ],
  rehab: [
    {
      id: 'rehab-1',
      text: 'ทำได้ดีมากครับ ค่อยๆ ขยับตามหุ่นสาธิตนะครับ',
      icon: 'heart-pulse',
    },
    {
      id: 'rehab-2',
      text: 'เก่งมากครับ ขยับช้าๆ สบายๆ ไม่ต้องรีบนะครับ น้องหมอเทคเอาใจช่วยอยู่ครับ',
      icon: 'sparkles',
    },
  ],
  'patient-portal': [
    {
      id: 'portal-1',
      text: 'เช็กผลความก้าวหน้าการฟื้นฟูและดูวันนัดหมายครั้งถัดไปได้ที่หน้านี้นะครับ',
      icon: 'calendar',
    },
    {
      id: 'portal-2',
      text: 'ฝึกอย่างสม่ำเสมอทุกวัน แข็งแรงขึ้นแน่นอนครับ น้องหมอเทคคอยดูแลอยู่นะครับ',
      icon: 'heart-pulse',
    },
  ],
};

const LOCALIZED_MASCOT_MESSAGES: Record<
  Exclude<MorTechMascotLang, 'th'>,
  Record<MorTechMascotMode, readonly MorTechMascotMessage[]>
> = {
  en: {
    standby: [
      {
        id: 'standby-en-1',
        text: 'Hello! Welcome to MorTech. Raise both hands for 3 seconds to start touch-free.',
        icon: 'hand',
      },
      {
        id: 'standby-en-2',
        text: 'Ready for today? Hold both hands up in front of the camera to wake the kiosk.',
        icon: 'sparkles',
      },
    ],
    'face-login': [
      {
        id: 'face-login-en-1',
        text: 'Please look straight at the camera. Verifying your profile now.',
        icon: 'sparkles',
      },
      {
        id: 'face-login-en-2',
        text: 'Hold steady for a moment. Taking you to your rehab plan shortly.',
        icon: 'heart-pulse',
      },
    ],
    rehab: [
      {
        id: 'rehab-en-1',
        text: 'Great job! Follow the demo model slowly and comfortably.',
        icon: 'heart-pulse',
      },
      {
        id: 'rehab-en-2',
        text: 'You are doing wonderfully. Move at your own safe pace.',
        icon: 'sparkles',
      },
    ],
    'patient-portal': [
      {
        id: 'portal-en-1',
        text: 'Check your recovery progress and next hospital appointment here.',
        icon: 'calendar',
      },
      {
        id: 'portal-en-2',
        text: 'Steady daily practice builds strength. Keep up the great work.',
        icon: 'heart-pulse',
      },
    ],
  },
  cn: {
    standby: [
      {
        id: 'standby-cn-1',
        text: '您好，欢迎来到 MorTech。请举起双手保持 3 秒即可免接触启动。',
        icon: 'hand',
      },
      {
        id: 'standby-cn-2',
        text: '准备好今天的康复训练了吗？面向镜头举起双手即可唤醒设备。',
        icon: 'sparkles',
      },
    ],
    'face-login': [
      {
        id: 'face-login-cn-1',
        text: '请直视摄像头，系统正在为您确认身份。',
        icon: 'sparkles',
      },
      {
        id: 'face-login-cn-2',
        text: '请保持片刻，马上带您进入康复训练界面。',
        icon: 'heart-pulse',
      },
    ],
    rehab: [
      {
        id: 'rehab-cn-1',
        text: '做得很好！请跟着示范模型慢慢活动。',
        icon: 'heart-pulse',
      },
      {
        id: 'rehab-cn-2',
        text: '非常棒，请按照舒适安全的节奏慢慢练习。',
        icon: 'sparkles',
      },
    ],
    'patient-portal': [
      {
        id: 'portal-cn-1',
        text: '您可以在这里查看康复进展与下次预约时间。',
        icon: 'calendar',
      },
      {
        id: 'portal-cn-2',
        text: '坚持每日训练会越来越强壮，MorTech 一直陪伴您。',
        icon: 'heart-pulse',
      },
    ],
  },
};

export function getMorTechMascotMessages(
  mode: MorTechMascotMode,
  lang: MorTechMascotLang = 'th',
): readonly MorTechMascotMessage[] {
  if (lang === 'en' || lang === 'cn') {
    return LOCALIZED_MASCOT_MESSAGES[lang][mode];
  }
  return MOR_TECH_MASCOT_MESSAGES[mode];
}

/**
 * Calculates non-intrusive horizontal patrol bounds within the current viewport so
 * Nong MorTech stays clear of side edges, emergency controls, and never introduces scrollbars.
 */
export function computeMascotPatrolBounds(
  viewportWidth: number,
  mode: MorTechMascotMode,
  mascotWidthPx = MOR_TECH_MASCOT_WIDTH_PX,
): MascotPatrolBounds {
  const safeViewport = Number.isFinite(viewportWidth) && viewportWidth > 0 ? viewportWidth : 1024;
  // Reserve extra right-edge safety buffer on rehab & face-login so emergency/settings buttons stay unobstructed.
  const leftMargin = safeViewport < 520 ? 10 : 18;
  const rightReserve =
    mode === 'rehab'
      ? Math.min(180, Math.round(safeViewport * 0.22))
      : mode === 'face-login'
        ? Math.min(140, Math.round(safeViewport * 0.16))
        : safeViewport < 520
          ? 12
          : 22;

  const minX = leftMargin;
  const rawMaxX = safeViewport - mascotWidthPx - rightReserve;
  const maxX = Math.max(minX, Math.round(rawMaxX));
  const midX = Math.round((minX + maxX) / 2);
  const travelDistance = Math.max(0, maxX - minX);
  const canPatrol = travelDistance >= 48;
  // Gentle elderly-friendly stroll speed (~62 px/s), bounded to 2.6s..10s per leg
  const walkDurationSec = canPatrol
    ? Number(Math.max(2.6, Math.min(10, travelDistance / 62)).toFixed(2))
    : 0;

  return {
    minX,
    midX,
    maxX,
    travelDistance,
    walkDurationSec,
    canPatrol,
  };
}

export interface MascotSpeechBubbleLayout {
  bubbleWidthPx: number;
  relativeLeftPx: number;
  tailLeftPx: number;
}

/**
 * Computes mathematically bounded horizontal placement for the speech bubble relative
 * to the mascot's `x` position so `[targetX + relativeLeftPx, targetX + relativeLeftPx + bubbleWidthPx]`
 * always remains strictly inside `[12, viewportWidth - 12]` at every screen width (zero scrollbar).
 */
export function computeSpeechBubbleLayout(
  mascotX: number,
  viewportWidth: number,
  mascotWidthPx = MOR_TECH_MASCOT_WIDTH_PX,
): MascotSpeechBubbleLayout {
  const safeViewport = Number.isFinite(viewportWidth) && viewportWidth > 0 ? viewportWidth : 1024;
  const edgePad = 12;
  const bubbleWidthPx = Math.max(210, Math.min(352, Math.round(safeViewport - edgePad * 2)));
  const otterCenterX = mascotX + mascotWidthPx * 0.5;
  const idealLeftX = otterCenterX - bubbleWidthPx * 0.5;
  const maxAllowedLeftX = Math.max(edgePad, safeViewport - bubbleWidthPx - edgePad);
  const clampedBubbleLeftX = Math.round(Math.max(edgePad, Math.min(maxAllowedLeftX, idealLeftX)));
  const relativeLeftPx = Math.round(clampedBubbleLeftX - mascotX);
  const tailLeftPx = Math.round(
    Math.max(22, Math.min(bubbleWidthPx - 22, otterCenterX - clampedBubbleLeftX)),
  );
  return {
    bubbleWidthPx,
    relativeLeftPx,
    tailLeftPx,
  };
}

function MascotBubbleIcon({ kind }: { kind: MorTechMascotIconKind }) {
  switch (kind) {
    case 'hand':
      return <Hand className="h-5 w-5 text-emerald-700 shrink-0" aria-hidden="true" />;
    case 'heart-pulse':
      return <HeartPulse className="h-5 w-5 text-emerald-700 shrink-0" aria-hidden="true" />;
    case 'calendar':
      return <Calendar className="h-5 w-5 text-emerald-700 shrink-0" aria-hidden="true" />;
    case 'sparkles':
    default:
      return <Sparkles className="h-5 w-5 text-emerald-700 shrink-0" aria-hidden="true" />;
  }
}

interface MorTechSeaOtterSvgProps {
  phase: MorTechMascotPhase;
  reducedMotion: boolean;
}

/**
 * Custom vector SVG illustration of "Nong MorTech" (friendly clinical sea otter):
 * - Warm brown fur & cream belly/cheeks/muzzle
 * - Mint-green athletic headband & medical scarf with heart + medical cross emblem
 * - One paw cheerfully waving hello, other paw holding a resistance exercise band
 * - GPU-composited Framer Motion limb/tail/body motion (zero main-thread layout thrash)
 */
const MorTechSeaOtterSvg = memo(function MorTechSeaOtterSvg({
  phase,
  reducedMotion,
}: MorTechSeaOtterSvgProps) {
  const uid = useId().replace(/:/g, '');
  const isWalking = phase === 'walking' && !reducedMotion;
  const isWaving = phase === 'waving' && !reducedMotion;

  return (
    <svg
      width={MOR_TECH_MASCOT_WIDTH_PX}
      height={MOR_TECH_MASCOT_HEIGHT_PX}
      viewBox="0 0 132 132"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      role="img"
      aria-label="น้องหมอเทค นากทะเลผู้ช่วยกายภาพบำบัด"
      className="block overflow-visible drop-shadow-[0_6px_12px_rgba(15,76,129,0.18)]"
      data-testid="mortech-otter-svg"
    >
      <defs>
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

      {/* Contact shadow under feet */}
      <motion.ellipse
        cx="66"
        cy="122"
        rx="30"
        ry="5.5"
        fill="#0F4C81"
        animate={
          isWalking
            ? { rx: [29, 25, 29], opacity: [0.16, 0.11, 0.16] }
            : { rx: 30, opacity: 0.16 }
        }
        transition={
          isWalking
            ? { duration: 0.56, repeat: Infinity, ease: 'easeInOut' }
            : { duration: 0.25 }
        }
      />

      {/* Wagging Sea Otter Tail */}
      <motion.g
        style={{ transformOrigin: '44px 96px', transformBox: 'view-box' }}
        animate={
          isWalking
            ? { rotate: [-10, 12, -10] }
            : isWaving
              ? { rotate: [-6, 14, -6] }
              : { rotate: 0 }
        }
        transition={
          isWalking || isWaving
            ? { duration: isWalking ? 0.56 : 0.9, repeat: Infinity, ease: 'easeInOut' }
            : { duration: 0.2 }
        }
      >
        <path
          d="M44 96 C26 95 13 88 11 78 C9 71 17 69 24 75 C30 80 37 83 47 85 Z"
          fill={`url(#furDark-${uid})`}
          stroke="#5C3620"
          strokeWidth="2.2"
          strokeLinejoin="round"
        />
        <path
          d="M21 77 C27 81 34 84 42 87"
          stroke="#C89570"
          strokeWidth="2"
          strokeLinecap="round"
          opacity="0.65"
        />
      </motion.g>

      {/* Stepping Hind Paws */}
      <motion.g
        style={{ transformOrigin: '54px 108px', transformBox: 'view-box' }}
        animate={
          isWalking
            ? { rotate: [-18, 16, -18], y: [0, -4, 0] }
            : { rotate: 0, y: 0 }
        }
        transition={
          isWalking
            ? { duration: 0.56, repeat: Infinity, ease: 'easeInOut' }
            : { duration: 0.25 }
        }
      >
        <ellipse
          cx="52"
          cy="115"
          rx="11"
          ry="6.5"
          fill={`url(#furDark-${uid})`}
          stroke="#5C3620"
          strokeWidth="2"
        />
        <circle cx="46" cy="116" r="1.5" fill="#E9CFAF" />
        <circle cx="51" cy="117.5" r="1.5" fill="#E9CFAF" />
        <circle cx="56" cy="116.5" r="1.5" fill="#E9CFAF" />
      </motion.g>

      <motion.g
        style={{ transformOrigin: '78px 108px', transformBox: 'view-box' }}
        animate={
          isWalking
            ? { rotate: [16, -18, 16], y: [-4, 0, -4] }
            : { rotate: 0, y: 0 }
        }
        transition={
          isWalking
            ? { duration: 0.56, repeat: Infinity, ease: 'easeInOut' }
            : { duration: 0.25 }
        }
      >
        <ellipse
          cx="80"
          cy="115"
          rx="11"
          ry="6.5"
          fill={`url(#furDark-${uid})`}
          stroke="#5C3620"
          strokeWidth="2"
        />
        <circle cx="75" cy="116.5" r="1.5" fill="#E9CFAF" />
        <circle cx="80" cy="117.5" r="1.5" fill="#E9CFAF" />
        <circle cx="85" cy="116" r="1.5" fill="#E9CFAF" />
      </motion.g>

      {/* Bobbing / Swaying Torso + Head Group */}
      <motion.g
        style={{ transformOrigin: '66px 98px', transformBox: 'view-box' }}
        animate={
          isWalking
            ? { y: [0, -4, 0, -3.5, 0], rotate: [-3, 3, -3] }
            : isWaving
              ? { y: [0, -2.5, 0], rotate: [0, 2.5, -1.5, 0] }
              : { y: 0, rotate: 0 }
        }
        transition={
          isWalking
            ? { duration: 0.56, repeat: Infinity, ease: 'easeInOut' }
            : isWaving
              ? { duration: 1.4, repeat: Infinity, ease: 'easeInOut' }
              : { duration: 0.25 }
        }
      >
        {/* Left Paw Holding Mint/Sky Resistance Exercise Band */}
        <motion.g
          style={{ transformOrigin: '44px 80px', transformBox: 'view-box' }}
          animate={
            isWalking
              ? { rotate: [-8, 10, -8] }
              : isWaving
                ? { rotate: [-4, 6, -4] }
                : { rotate: 0 }
          }
          transition={
            isWalking || isWaving
              ? { duration: isWalking ? 0.56 : 1.1, repeat: Infinity, ease: 'easeInOut' }
              : { duration: 0.2 }
          }
        >
          {/* Stretched Resistance Exercise Band Loop */}
          <path
            d="M32 82 C16 74 12 96 27 104 C36 109 42 98 35 88"
            stroke={`url(#band-${uid})`}
            strokeWidth="4.8"
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
          {/* Resistance Band Grip Handle */}
          <rect
            x="23"
            y="80"
            width="11"
            height="5.5"
            rx="2.75"
            transform="rotate(-22 28.5 82.75)"
            fill="#0F4C81"
          />
          {/* Left Otter Arm & Paw */}
          <path
            d="M44 76 C36 78 31 82 32 88 C33 93 40 93 46 89"
            fill={`url(#fur-${uid})`}
            stroke="#5C3620"
            strokeWidth="2.1"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          <ellipse cx="34" cy="87" rx="5.5" ry="4.5" fill="#8A5638" stroke="#5C3620" strokeWidth="1.8" />
        </motion.g>

        {/* Warm Brown Torso */}
        <ellipse
          cx="66"
          cy="90"
          rx="25"
          ry="24"
          fill={`url(#fur-${uid})`}
          stroke="#5C3620"
          strokeWidth="2.3"
        />
        {/* Soft Cream Belly Patch */}
        <ellipse cx="66" cy="93" rx="16.5" ry="17" fill={`url(#cream-${uid})`} />

        {/* Mint-Green Clinical Scarf Wrap + Flowing Tail */}
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
        {/* Scarf Emblem: Medical Heart + Cross */}
        <circle cx="66" cy="76.5" r="7.5" fill="#FFFFFF" stroke="#0D9488" strokeWidth="1.7" />
        <path
          d="M66 81.2 C66 81.2 60.8 78.2 60.8 74.9 C60.8 73.2 62.1 72 63.7 72 C64.7 72 65.5 72.5 66 73.3 C66.5 72.5 67.3 72 68.3 72 C69.9 72 71.2 73.2 71.2 74.9 C71.2 78.2 66 81.2 66 81.2 Z"
          fill="#10B981"
        />
        <rect x="65.15" y="73.9" width="1.7" height="4.4" rx="0.7" fill="#FFFFFF" />
        <rect x="63.8" y="75.25" width="4.4" height="1.7" rx="0.7" fill="#FFFFFF" />

        {/* Right Waving Arm */}
        <motion.g
          style={{ transformOrigin: '88px 78px', transformBox: 'view-box' }}
          animate={
            isWaving
              ? { rotate: [-18, 6, -20, 4, -18] }
              : isWalking
                ? { rotate: [-8, 12, -8] }
                : { rotate: -12 }
          }
          transition={
            isWaving
              ? { duration: 0.95, repeat: Infinity, ease: 'easeInOut' }
              : isWalking
                ? { duration: 0.56, repeat: Infinity, ease: 'easeInOut' }
                : { duration: 0.25 }
          }
        >
          <path
            d="M86 76 C95 72 103 65 107 58 C110 54 115 57 113 63 C109 73 99 83 87 86 Z"
            fill={`url(#fur-${uid})`}
            stroke="#5C3620"
            strokeWidth="2.1"
            strokeLinejoin="round"
          />
          <circle cx="110" cy="58" r="5.5" fill="#9C6644" stroke="#5C3620" strokeWidth="1.9" />
          <circle cx="109" cy="57.5" r="2.2" fill="#F2DFC2" />
        </motion.g>

        {/* Rounded Sea Otter Ears */}
        <circle cx="40" cy="34" r="8.5" fill={`url(#furDark-${uid})`} stroke="#5C3620" strokeWidth="2.1" />
        <circle cx="40" cy="34.5" r="4.2" fill="#F2DFC2" />
        <circle cx="92" cy="34" r="8.5" fill={`url(#furDark-${uid})`} stroke="#5C3620" strokeWidth="2.1" />
        <circle cx="92" cy="34.5" r="4.2" fill="#F2DFC2" />

        {/* Sea Otter Head */}
        <ellipse
          cx="66"
          cy="47"
          rx="28"
          ry="24"
          fill={`url(#fur-${uid})`}
          stroke="#5C3620"
          strokeWidth="2.3"
        />

        {/* Soft Cream Facial Mask & Cheeks */}
        <ellipse cx="66" cy="52" rx="22" ry="17" fill={`url(#cream-${uid})`} />

        {/* Mint-Green Sport Headband with Medical Heart + Cross Badge */}
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

        {/* Expressive Blinking Eyes */}
        <motion.g
          style={{ transformOrigin: '66px 48.5px', transformBox: 'view-box' }}
          animate={reducedMotion ? { scaleY: 1 } : { scaleY: [1, 1, 0.12, 1, 1] }}
          transition={
            reducedMotion
              ? { duration: 0 }
              : {
                  duration: 4.2,
                  times: [0, 0.9, 0.94, 0.98, 1],
                  repeat: Infinity,
                  ease: 'easeInOut',
                }
          }
        >
          <circle cx="55.5" cy="48.5" r="4.3" fill="#2A1A10" />
          <circle cx="76.5" cy="48.5" r="4.3" fill="#2A1A10" />
          <circle cx="57" cy="46.8" r="1.5" fill="#FFFFFF" />
          <circle cx="78" cy="46.8" r="1.5" fill="#FFFFFF" />
        </motion.g>

        {/* Warm Rosy Cheeks */}
        <ellipse cx="47.5" cy="54.5" rx="4.2" ry="2.4" fill="#FB7185" opacity="0.34" />
        <ellipse cx="84.5" cy="54.5" rx="4.2" ry="2.4" fill="#FB7185" opacity="0.34" />

        {/* Cute Otter Nose & Friendly Smile */}
        <path
          d="M61.5 52.5 C61.5 50.8 70.5 50.8 70.5 52.5 C70.5 54.8 68 56.8 66 56.8 C64 56.8 61.5 54.8 61.5 52.5 Z"
          fill="#3E2414"
        />
        <ellipse cx="64.8" cy="52.4" rx="1.5" ry="0.8" fill="#A67C5E" opacity="0.65" />
        <path
          d="M59.5 59 C62 62.2 64.5 61.5 66 59.8 C67.5 61.5 70 62.2 72.5 59"
          stroke="#5C3620"
          strokeWidth="2.1"
          strokeLinecap="round"
          strokeLinejoin="round"
          fill="none"
        />

        {/* Crisp Otter Whiskers */}
        <g stroke="#7A4B2E" strokeWidth="1.5" strokeLinecap="round" opacity="0.72">
          <line x1="45" y1="54" x2="36" y2="52.5" />
          <line x1="45" y1="57" x2="36.5" y2="58.5" />
          <line x1="87" y1="54" x2="96" y2="52.5" />
          <line x1="87" y1="57" x2="95.5" y2="58.5" />
        </g>
      </motion.g>
    </svg>
  );
});

/**
 * Interactive Companion Mascot ("Nong MorTech" Sea Otter Rehab Assistant) for Dr.Tech.Care.
 * Patrols horizontally along the bottom edge with smooth Framer Motion transforms, flips direction
 * cleanly via `scaleX`, pauses periodically to wave hello, and renders a high-contrast elderly-friendly
 * context-aware speech bubble with a collapse/restore button. Zero Unicode emojis.
 */
export const MorTechMascot = memo(function MorTechMascot({
  mode,
  lang = 'th',
  customMessage = null,
  hidden = false,
  defaultSpeechVisible = true,
  className = '',
}: MorTechMascotProps) {
  const prefersReducedMotion = Boolean(useReducedMotion());
  const messages = useMemo(() => getMorTechMascotMessages(mode, lang), [mode, lang]);

  const [viewportWidth, setViewportWidth] = useState<number>(() =>
    typeof window !== 'undefined' ? window.innerWidth : 1024,
  );
  const [speechVisible, setSpeechVisible] = useState<boolean>(defaultSpeechVisible);
  const [messageIdx, setMessageIdx] = useState<number>(0);

  const bounds = useMemo(
    () => computeMascotPatrolBounds(viewportWidth, mode, MOR_TECH_MASCOT_WIDTH_PX),
    [viewportWidth, mode],
  );

  const [phase, setPhase] = useState<MorTechMascotPhase>('waving');
  const [facing, setFacing] = useState<1 | -1>(1);
  const [targetX, setTargetX] = useState<number>(() => bounds.minX);
  const [moveDurationSec, setMoveDurationSec] = useState<number>(0);

  const patrolTimerRef = useRef<number | null>(null);
  const currentXRef = useRef<number>(bounds.minX);
  const waypointIdxRef = useRef<number>(0);

  useEffect(() => {
    setMessageIdx(0);
  }, [mode, lang]);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    const handleResize = () => {
      setViewportWidth(window.innerWidth);
    };
    window.addEventListener('resize', handleResize, { passive: true });
    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, []);

  // Clamp position immediately whenever viewport or mode bounds shrink
  useEffect(() => {
    const clamped = Math.max(bounds.minX, Math.min(bounds.maxX, currentXRef.current));
    if (clamped !== currentXRef.current) {
      currentXRef.current = clamped;
      setTargetX(clamped);
      setMoveDurationSec(0.35);
    }
  }, [bounds.minX, bounds.maxX]);

  // Low-frequency patrol state machine across [left, center, right, center] waypoints:
  // alternates smooth strolling with friendly wave pauses; fires React state updates only
  // once every 3–7s so MediaPipe / Face Landmarker inference stays at peak FPS.
  useEffect(() => {
    if (hidden || typeof window === 'undefined') {
      return;
    }

    const clearPatrolTimer = () => {
      if (patrolTimerRef.current !== null) {
        window.clearTimeout(patrolTimerRef.current);
        patrolTimerRef.current = null;
      }
    };

    if (prefersReducedMotion || !bounds.canPatrol) {
      clearPatrolTimer();
      currentXRef.current = bounds.minX;
      setTargetX(bounds.minX);
      setMoveDurationSec(0);
      setFacing(1);
      setPhase('waving');
      return;
    }

    let cancelled = false;
    const waypoints =
      bounds.travelDistance >= 220
        ? [bounds.minX, bounds.midX, bounds.maxX, bounds.midX]
        : [bounds.minX, bounds.maxX];

    const scheduleWaveThenWalk = (pauseMs: number) => {
      clearPatrolTimer();
      setPhase('waving');
      patrolTimerRef.current = window.setTimeout(() => {
        if (cancelled) return;
        const nextIdx = (waypointIdxRef.current + 1) % waypoints.length;
        const nextX = waypoints[nextIdx];
        waypointIdxRef.current = nextIdx;

        const nextFacing: 1 | -1 = nextX >= currentXRef.current ? 1 : -1;
        const dist = Math.abs(nextX - currentXRef.current);
        const legSec = Number(Math.max(2.4, Math.min(7.5, dist / 62)).toFixed(2));

        currentXRef.current = nextX;
        setFacing(nextFacing);
        setMoveDurationSec(legSec);
        setTargetX(nextX);
        setPhase('walking');

        patrolTimerRef.current = window.setTimeout(() => {
          if (cancelled) return;
          scheduleWaveThenWalk(3200);
        }, Math.round(legSec * 1000) + 120);
      }, pauseMs);
    };

    waypointIdxRef.current = 0;
    currentXRef.current = bounds.minX;
    setTargetX(bounds.minX);
    setMoveDurationSec(0);
    scheduleWaveThenWalk(2600);

    return () => {
      cancelled = true;
      clearPatrolTimer();
    };
  }, [bounds.canPatrol, bounds.maxX, bounds.midX, bounds.minX, bounds.travelDistance, hidden, prefersReducedMotion]);

  // Gently rotate helpful tips every 12 seconds when speech bubble is visible
  useEffect(() => {
    if (hidden || !speechVisible || customMessage || messages.length <= 1 || typeof window === 'undefined') {
      return;
    }
    const id = window.setInterval(() => {
      setMessageIdx((prev) => (prev + 1) % messages.length);
    }, 12_000);
    return () => {
      window.clearInterval(id);
    };
  }, [customMessage, hidden, messages.length, speechVisible]);

  const handleTapMascot = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      setPhase('waving');
      if (!speechVisible) {
        setSpeechVisible(true);
      } else if (messages.length > 1) {
        setMessageIdx((prev) => (prev + 1) % messages.length);
      }
    },
    [messages.length, speechVisible],
  );

  const handleCloseSpeech = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    setSpeechVisible(false);
  }, []);

  if (hidden) {
    return null;
  }

  const activeMessage = messages[messageIdx % messages.length] ?? messages[0];
  const displayText = customMessage && customMessage.trim().length > 0 ? customMessage.trim() : activeMessage.text;
  const bubbleLayout = computeSpeechBubbleLayout(targetX, viewportWidth, MOR_TECH_MASCOT_WIDTH_PX);
  const legDuration = prefersReducedMotion ? 0 : moveDurationSec;

  return (
    <motion.div
      data-testid="mortech-mascot-companion"
      data-mode={mode}
      data-phase={phase}
      data-facing={facing === 1 ? 'right' : 'left'}
      initial={false}
      animate={{ x: targetX }}
      transition={{
        x: {
          duration: legDuration,
          ease: 'linear',
        },
      }}
      style={{ width: MOR_TECH_MASCOT_WIDTH_PX, willChange: 'transform' }}
      onClick={(e) => e.stopPropagation()}
      className={`fixed bottom-3 left-0 z-40 pointer-events-auto select-none ${className}`}
    >
      {/* Context-Aware High-Contrast Speech Bubble (Always Upright, Convex-Clamped Inside Viewport) */}
      <AnimatePresence initial={false}>
        {speechVisible && (
          <motion.div
            key={`${mode}-${activeMessage.id}`}
            data-testid="mortech-mascot-speech-bubble"
            initial={
              prefersReducedMotion
                ? { opacity: 1, x: bubbleLayout.relativeLeftPx }
                : { opacity: 0, y: 8, scale: 0.94, x: bubbleLayout.relativeLeftPx }
            }
            animate={{ opacity: 1, y: 0, scale: 1, x: bubbleLayout.relativeLeftPx }}
            exit={
              prefersReducedMotion
                ? { opacity: 0, x: bubbleLayout.relativeLeftPx }
                : { opacity: 0, y: 6, scale: 0.94, x: bubbleLayout.relativeLeftPx }
            }
            transition={{
              opacity: { duration: 0.2, ease: 'easeOut' },
              y: { duration: 0.22, ease: 'easeOut' },
              scale: { duration: 0.22, ease: 'easeOut' },
              x: { duration: legDuration, ease: 'linear' },
            }}
            style={{ width: bubbleLayout.bubbleWidthPx, willChange: 'transform' }}
            className="absolute bottom-full left-0 mb-2 rounded-2xl border-2 border-emerald-400 bg-white/98 px-4 py-3 text-slate-900 shadow-[0_12px_28px_rgba(15,76,129,0.2)] backdrop-blur-md origin-bottom"
          >
            <div className="flex items-start gap-2.5">
              <div className="mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-xl border border-emerald-200 bg-emerald-50">
                <MascotBubbleIcon kind={activeMessage.icon} />
              </div>

              <div className="min-w-0 flex-1 pr-1">
                <p className="text-xs font-extrabold tracking-wide text-emerald-700">
                  น้องหมอเทค
                </p>
                <p
                  data-testid="mortech-mascot-speech-text"
                  className="mt-0.5 text-lg sm:text-xl font-extrabold leading-snug text-slate-900"
                >
                  {displayText}
                </p>
              </div>

              <button
                type="button"
                data-testid="mortech-mascot-close-bubble"
                onClick={handleCloseSpeech}
                aria-label="ซ่อนข้อความน้องหมอเทค"
                title="ซ่อนข้อความ"
                className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl border border-emerald-200 bg-emerald-50/70 text-emerald-800 hover:bg-emerald-100 hover:text-emerald-950 transition-colors cursor-pointer"
              >
                <X className="h-5 w-5" aria-hidden="true" />
              </button>
            </div>

            {/* Speech Bubble Pointer Tail tracking Nong MorTech's head */}
            <motion.div
              aria-hidden="true"
              initial={false}
              animate={{ x: bubbleLayout.tailLeftPx - 7 }}
              transition={{ x: { duration: legDuration, ease: 'linear' } }}
              className="absolute -bottom-2 left-0 h-3.5 w-3.5 rotate-45 border-b-2 border-r-2 border-emerald-400 bg-white"
            />
          </motion.div>
        )}
      </AnimatePresence>

      {/* Interactive Sea Otter Button (Flips smoothly via scaleX on facing change) */}
      <button
        type="button"
        data-testid="mortech-mascot-avatar-btn"
        onClick={handleTapMascot}
        aria-label={
          speechVisible
            ? 'น้องหมอเทค กดเพื่อเปลี่ยนคำแนะนำ'
            : 'น้องหมอเทค กดเพื่อเปิดกล่องข้อความคำแนะนำ'
        }
        title={speechVisible ? 'แตะเพื่อดูคำแนะนำถัดไป' : 'แตะเพื่อเปิดข้อความจากน้องหมอเทค'}
        className="group relative flex items-center justify-center rounded-full focus:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400 cursor-pointer"
      >
        {/* Subtle restore indicator pill when speech bubble is minimized */}
        {!speechVisible && (
          <span
            data-testid="mortech-mascot-minimized-pill"
            className="absolute -top-1 -right-1 z-10 flex h-7 w-7 items-center justify-center rounded-full border-2 border-white bg-emerald-500 text-white shadow-md transition-transform group-hover:scale-110"
          >
            <MessageCircle className="h-4 w-4" aria-hidden="true" />
          </span>
        )}

        <motion.div
          animate={{ scaleX: facing }}
          transition={{ duration: prefersReducedMotion ? 0 : 0.22, ease: 'easeOut' }}
          style={{ willChange: 'transform' }}
          className="transition-transform group-hover:scale-105 group-active:scale-95"
        >
          <MorTechSeaOtterSvg phase={phase} reducedMotion={prefersReducedMotion} />
        </motion.div>
      </button>
    </motion.div>
  );
});

export default MorTechMascot;
