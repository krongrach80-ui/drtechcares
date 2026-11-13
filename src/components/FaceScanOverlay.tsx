import React from 'react';
import { motion } from 'framer-motion';
import { CheckCircle2, ScanFace, Sparkles } from 'lucide-react';

interface FaceScanOverlayProps {
  verified: boolean;
  streakFrames?: number;
  requiredFrames?: number;
  distanceM: number;
  patientName?: string | null;
}

export function FaceScanOverlay({
  verified,
  streakFrames = 5,
  requiredFrames = 5,
  distanceM,
  patientName,
}: FaceScanOverlayProps) {
  const bracketColor = verified ? '#10B981' : '#38BDF8';

  return (
    <div className="pointer-events-none absolute inset-0 z-20 flex flex-col items-center justify-center bg-gradient-to-b from-[#0F4C81]/20 via-[#0F4C81]/10 to-[#0F4C81]/30 backdrop-blur-[2px] px-4">
      {/* Main Circular Radar SVG */}
      <div className="relative flex h-72 w-72 items-center justify-center sm:h-80 sm:w-80">
        <svg viewBox="0 0 360 360" className="h-full w-full overflow-visible">
          <defs>
            <linearGradient id="radarLaserGrad" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="rgba(56, 189, 248, 0)" />
              <stop offset="50%" stopColor={verified ? '#10B981' : '#38BDF8'} />
              <stop offset="100%" stopColor="rgba(16, 185, 129, 0)" />
            </linearGradient>
            <radialGradient id="radarAuraGrad" cx="50%" cy="50%" r="50%">
              <stop offset="0%" stopColor={verified ? 'rgba(16,185,129,0.20)' : 'rgba(56,189,248,0.16)'} />
              <stop offset="100%" stopColor="rgba(56,189,248,0)" />
            </radialGradient>
          </defs>

          <circle cx="180" cy="180" r="154" fill="url(#radarAuraGrad)" />

          {/* Outer counter-rotating dashed ring */}
          <motion.circle
            cx="180"
            cy="180"
            r="146"
            fill="none"
            stroke={verified ? '#10B981' : '#38BDF8'}
            strokeWidth={verified ? 3 : 2}
            strokeDasharray={verified ? 'none' : '12 10'}
            strokeOpacity={verified ? 0.9 : 0.75}
            animate={{ rotate: verified ? 0 : 360 }}
            transition={{ repeat: verified ? 0 : Infinity, duration: 16, ease: 'linear' }}
            style={{ originX: '180px', originY: '180px' }}
          />

          {/* Inner rotating targeting ring */}
          <motion.circle
            cx="180"
            cy="180"
            r="122"
            fill="none"
            stroke={verified ? '#0D9488' : '#0284C7'}
            strokeWidth="2.5"
            strokeDasharray="24 14 6 14"
            strokeOpacity={0.8}
            animate={{ rotate: verified ? 0 : -360 }}
            transition={{ repeat: verified ? 0 : Infinity, duration: 11, ease: 'linear' }}
            style={{ originX: '180px', originY: '180px' }}
          />

          {/* Snap-to-center Corner Brackets */}
          <motion.g
            animate={{ scale: verified ? 0.94 : 1.03 }}
            transition={{ type: 'spring', stiffness: 260, damping: 20 }}
            style={{ originX: '180px', originY: '180px' }}
            stroke={bracketColor}
            strokeWidth="4.5"
            strokeLinecap="round"
            fill="none"
          >
            <path d="M 76 112 L 76 76 L 112 76" />
            <path d="M 248 76 L 284 76 L 284 112" />
            <path d="M 76 248 L 76 284 L 112 284" />
            <path d="M 248 284 L 284 284 L 284 248" />
          </motion.g>

          {/* Stylized human facial contour guide */}
          <g stroke={verified ? '#10B981' : '#FFFFFF'} strokeOpacity={verified ? 0.85 : 0.65} strokeWidth="2" fill="none">
            <ellipse cx="180" cy="176" rx="56" ry="74" strokeDasharray={verified ? 'none' : '5 5'} />
            <circle cx="158" cy="164" r="4" fill={verified ? '#10B981' : '#38BDF8'} stroke="none" />
            <circle cx="202" cy="164" r="4" fill={verified ? '#10B981' : '#38BDF8'} stroke="none" />
            <path d="M 164 205 Q 180 218 196 205" strokeLinecap="round" />
          </g>

          {/* Cyan-Emerald Vertical Laser Scan Beam */}
          {!verified && (
            <motion.g
              animate={{ y: [-68, 68, -68] }}
              transition={{ repeat: Infinity, duration: 2.6, ease: 'easeInOut' }}
            >
              <rect x="88" y="176" width="184" height="5" rx="2.5" fill="url(#radarLaserGrad)" />
              <rect x="96" y="170" width="168" height="16" rx="8" fill="#38BDF8" fillOpacity="0.16" />
            </motion.g>
          )}
        </svg>

        {/* Center Status Pill */}
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          className={`absolute -bottom-3 flex items-center gap-2 rounded-full border px-4 py-1.5 text-sm font-bold shadow-lg ${
            verified
              ? 'border-emerald-200 bg-emerald-500 text-white'
              : 'border-white/90 bg-white/90 text-[#0F4C81] backdrop-blur-md'
          }`}
        >
          {verified ? (
            <>
              <CheckCircle2 className="h-4 w-4" />
              <span>ยืนยันตัวตนสำเร็จ{patientName ? ` • ${patientName}` : ''}</span>
            </>
          ) : (
            <>
              <ScanFace className="h-4 w-4 text-[#0284C7]" />
              <span>กำลังสแกนใบหน้า 512-D ({Math.min(streakFrames, requiredFrames)}/{requiredFrames})</span>
            </>
          )}
        </motion.div>
      </div>

      {/* 5-Frame Match Bar & Distance Guidance */}
      <div className="mt-7 flex flex-col items-center gap-2 rounded-2xl border border-white/80 bg-white/85 px-5 py-3 text-center shadow-lg backdrop-blur-xl">
        <div className="flex items-center gap-2">
          {Array.from({ length: requiredFrames }).map((_, idx) => {
            const active = verified || idx < streakFrames;
            return (
              <motion.span
                key={idx}
                animate={{ scale: active ? 1.15 : 1 }}
                className={`h-2.5 w-7 rounded-full transition-colors ${
                  active ? 'bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.6)]' : 'bg-sky-100'
                }`}
              />
            );
          })}
        </div>
        <p className="flex items-center gap-1.5 text-sm font-semibold text-[#0F4C81]">
          <Sparkles className="h-4 w-4 text-[#0D9488]" />
          มองตรงกึ่งกลางกรอบ (±15%) • ระยะปัจจุบัน{' '}
          <span className="tabular-mono font-bold text-[#0284C7]">{distanceM.toFixed(2)} ม.</span>
        </p>
      </div>
    </div>
  );
}

