import React from 'react';

/**
 * Pure inline-SVG & 60FPS GPU CSS micro-interaction primitives for playful,
 * game-like clinical rehab feedback across Dr.Tech.Care.
 *
 * Strict compliance:
 * - ZERO Unicode emojis (pure inline SVG geometry & clean typography only).
 * - GPU-only transforms & opacity (zero layout thrash).
 */

interface BurstParticleSpec {
  id: number;
  dx: number;
  dy: number;
  rot: number;
  delayMs: number;
  size: number;
  color: string;
  shape: 'star' | 'diamond' | 'ring' | 'ribbon';
}

const BURST_PALETTES: Record<'rep' | 'combo' | 'victory' | 'unlock', readonly string[]> = {
  rep: ['#10b981', '#0284c7', '#38bdf8', '#f59e0b', '#14b8a6'],
  combo: ['#f59e0b', '#10b981', '#0284c7', '#f97316', '#34d399'],
  victory: ['#10b981', '#f59e0b', '#0284c7', '#38bdf8', '#fbbf24', '#14b8a6'],
  unlock: ['#0284c7', '#10b981', '#38bdf8', '#34d399', '#fbbf24'],
};

function buildBurstParticles(variant: 'rep' | 'combo' | 'victory' | 'unlock'): BurstParticleSpec[] {
  const count = variant === 'victory' ? 14 : variant === 'combo' ? 10 : 8;
  const radiusBase = variant === 'victory' ? 92 : variant === 'combo' ? 72 : 58;
  const palette = BURST_PALETTES[variant];
  const shapes: BurstParticleSpec['shape'][] = ['star', 'diamond', 'ring', 'ribbon'];

  return Array.from({ length: count }, (_, idx) => {
    const angle = (idx / count) * Math.PI * 2 - Math.PI / 2;
    const radius = radiusBase * (idx % 2 === 0 ? 1 : 0.72);
    return {
      id: idx,
      dx: Math.round(Math.cos(angle) * radius),
      dy: Math.round(Math.sin(angle) * radius),
      rot: (idx % 2 === 0 ? 1 : -1) * (45 + (idx * 23) % 90),
      delayMs: (idx % 4) * 28,
      size: idx % 3 === 0 ? 14 : 11,
      color: palette[idx % palette.length],
      shape: shapes[idx % shapes.length],
    };
  });
}

export interface GameCelebrationBurstProps {
  /** Change this value (incrementing number or unique string) to replay the burst. */
  triggerKey: number | string;
  variant?: 'rep' | 'combo' | 'victory' | 'unlock';
  className?: string;
}

export const GameCelebrationBurst: React.FC<GameCelebrationBurstProps> = ({
  triggerKey,
  variant = 'rep',
  className = '',
}) => {
  if (!triggerKey || triggerKey === 0) return null;
  const particles = buildBurstParticles(variant);

  return (
    <span
      key={`burst-${variant}-${triggerKey}`}
      aria-hidden="true"
      data-testid="game-celebration-burst"
      className={`pointer-events-none absolute inset-0 flex items-center justify-center overflow-visible z-30 ${className}`}
    >
      {particles.map((p) => (
        <svg
          key={p.id}
          viewBox="0 0 16 16"
          width={p.size}
          height={p.size}
          className="game-particle-burst absolute"
          style={
            {
              '--gx': `${p.dx}px`,
              '--gy': `${p.dy}px`,
              '--grot': `${p.rot}deg`,
              animationDelay: `${p.delayMs}ms`,
            } as React.CSSProperties
          }
        >
          {p.shape === 'star' && (
            <path
              d="M8 1 L9.9 6.1 L15 8 L9.9 9.9 L8 15 L6.1 9.9 L1 8 L6.1 6.1 Z"
              fill={p.color}
            />
          )}
          {p.shape === 'diamond' && (
            <polygon points="8,1.5 14.5,8 8,14.5 1.5,8" fill={p.color} />
          )}
          {p.shape === 'ring' && (
            <circle cx="8" cy="8" r="5" fill="none" stroke={p.color} strokeWidth="2.4" />
          )}
          {p.shape === 'ribbon' && (
            <rect x="3" y="5.5" width="10" height="5" rx="2.5" fill={p.color} />
          )}
        </svg>
      ))}
    </span>
  );
};

export interface FloatingScorePopupProps {
  triggerKey: number | string;
  label: string;
  sublabel?: string;
  tone?: 'emerald' | 'gold' | 'sky' | 'rose';
  className?: string;
}

const POPUP_TONE_CLASSES: Record<'emerald' | 'gold' | 'sky' | 'rose', string> = {
  emerald: 'border-emerald-300 bg-gradient-to-r from-emerald-500 to-teal-500 text-white shadow-emerald-500/35',
  gold: 'border-amber-200 bg-gradient-to-r from-amber-500 to-orange-500 text-white shadow-amber-500/35',
  sky: 'border-sky-200 bg-gradient-to-r from-[#0F4C81] to-[#0284C7] text-white shadow-sky-500/35',
  rose: 'border-rose-200 bg-gradient-to-r from-rose-500 to-red-500 text-white shadow-rose-500/35',
};

export const FloatingScorePopup: React.FC<FloatingScorePopupProps> = ({
  triggerKey,
  label,
  sublabel,
  tone = 'emerald',
  className = '',
}) => {
  if (!triggerKey || triggerKey === 0) return null;

  return (
    <div
      key={`score-pop-${triggerKey}`}
      aria-hidden="true"
      data-testid="game-floating-score"
      className={`game-float-score pointer-events-none absolute left-1/2 top-2 z-30 -translate-x-1/2 whitespace-nowrap rounded-full border-2 px-3 py-1 text-xs sm:text-sm font-black shadow-lg ${POPUP_TONE_CLASSES[tone]} ${className}`}
    >
      <span>{label}</span>
      {sublabel ? <span className="ml-1.5 opacity-95 text-[11px] font-extrabold">{sublabel}</span> : null}
    </div>
  );
};

export interface GameComboBadgeProps {
  count: number;
  labelTh?: string;
  compact?: boolean;
  className?: string;
}

export const GameComboBadge: React.FC<GameComboBadgeProps> = ({
  count,
  labelTh = 'คอมโบ',
  compact = false,
  className = '',
}) => {
  if (count < 2) return null;
  const isHot = count >= 4;

  return (
    <span
      key={`combo-${count}`}
      data-testid="game-combo-badge"
      className={`game-combo-pop game-badge-shimmer relative overflow-hidden inline-flex items-center gap-1 rounded-full border-2 font-black whitespace-nowrap shadow-sm ${
        isHot
          ? 'border-amber-300 bg-gradient-to-r from-amber-500 via-orange-500 to-amber-500 text-white'
          : 'border-emerald-300 bg-gradient-to-r from-emerald-500 to-teal-500 text-white'
      } ${compact ? 'px-2 py-0.5 text-[10px]' : 'px-3 py-1 text-xs'} ${className}`}
    >
      <svg
        viewBox="0 0 16 16"
        className={compact ? 'h-3 w-3 shrink-0' : 'h-3.5 w-3.5 shrink-0'}
        fill="currentColor"
        aria-hidden="true"
      >
        <path d="M9.4 1.2 3.2 9h4.3l-1 5.8L12.8 7H8.5l.9-5.8Z" />
      </svg>
      <span>
        {labelTh} x{count}
      </span>
    </span>
  );
};

export interface GameXpProgressBarProps {
  value: number;
  max: number;
  label?: string;
  badgeText?: string;
  tone?: 'emerald' | 'blue' | 'gold';
  showMilestones?: boolean;
  className?: string;
}

const XP_BAR_GRADIENTS: Record<'emerald' | 'blue' | 'gold', string> = {
  emerald: 'from-[#0284C7] via-[#0D9488] to-[#10B981]',
  blue: 'from-[#0F4C81] via-[#0284C7] to-[#38BDF8]',
  gold: 'from-amber-500 via-yellow-400 to-emerald-500',
};

export const GameXpProgressBar: React.FC<GameXpProgressBarProps> = ({
  value,
  max,
  label,
  badgeText,
  tone = 'emerald',
  showMilestones = true,
  className = '',
}) => {
  const safeMax = Math.max(1, max);
  const pct = Math.max(0, Math.min(100, Math.round((value / safeMax) * 100)));

  return (
    <div data-testid="game-xp-progress-bar" className={`w-full space-y-1.5 ${className}`}>
      {(label || badgeText) && (
        <div className="flex items-center justify-between gap-2 text-xs font-extrabold">
          {label ? <span className="text-[#0F4C81] truncate">{label}</span> : <span />}
          {badgeText ? (
            <span className="inline-flex items-center gap-1 rounded-full border border-emerald-200 bg-emerald-50 px-2.5 py-0.5 text-[11px] font-black text-emerald-700 shrink-0 tabular-nums">
              <svg viewBox="0 0 16 16" className="h-3 w-3 text-amber-500" fill="currentColor" aria-hidden="true">
                <path d="M8 1.2 9.9 5.5 14.6 6 11 9.1 12.1 13.8 8 11.3 3.9 13.8 5 9.1 1.4 6 6.1 5.5 Z" />
              </svg>
              <span>{badgeText}</span>
            </span>
          ) : null}
        </div>
      )}

      <div className="relative h-3.5 w-full overflow-hidden rounded-full border border-sky-200/90 bg-slate-100 shadow-inner">
        <div
          className={`game-progress-stripes relative h-full rounded-full bg-gradient-to-r ${XP_BAR_GRADIENTS[tone]} transition-[width] duration-300 ease-out`}
          style={{ width: `${pct}%` }}
        >
          <span className="pointer-events-none absolute inset-x-0 top-0 h-1.5 rounded-full bg-white/35" />
        </div>

        {showMilestones && (
          <div className="pointer-events-none absolute inset-0 flex items-center justify-between px-[24%]">
            {[25, 50, 75].map((tick) => (
              <span
                key={tick}
                className={`h-2 w-2 rounded-full border transition-transform duration-300 ${
                  pct >= tick
                    ? 'border-white bg-amber-300 scale-110 shadow-xs'
                    : 'border-slate-300 bg-white/80'
                }`}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export interface ClinicalRankTier {
  grade: 'S+' | 'S' | 'A' | 'B' | 'C';
  labelTh: string;
  pillClass: string;
  accentColor: string;
}

export function resolveClinicalRankTier(pct: number): ClinicalRankTier {
  const safePct = Number.isFinite(pct) ? Math.max(0, pct) : 0;
  if (safePct >= 95) {
    return {
      grade: 'S+',
      labelTh: 'ยอดเยี่ยมสูงสุด',
      pillClass: 'border-amber-300 bg-gradient-to-r from-amber-50 via-yellow-50 to-emerald-50 text-amber-900',
      accentColor: '#d97706',
    };
  }
  if (safePct >= 85) {
    return {
      grade: 'S',
      labelTh: 'ดีเยี่ยม',
      pillClass: 'border-emerald-300 bg-gradient-to-r from-emerald-50 to-teal-50 text-emerald-800',
      accentColor: '#059669',
    };
  }
  if (safePct >= 70) {
    return {
      grade: 'A',
      labelTh: 'ดีมาก',
      pillClass: 'border-sky-300 bg-gradient-to-r from-sky-50 to-blue-50 text-[#0F4C81]',
      accentColor: '#0284c7',
    };
  }
  if (safePct >= 50) {
    return {
      grade: 'B',
      labelTh: 'กำลังพัฒนา',
      pillClass: 'border-indigo-200 bg-indigo-50 text-indigo-800',
      accentColor: '#4f46e5',
    };
  }
  return {
    grade: 'C',
    labelTh: 'เริ่มต้นฝึก',
    pillClass: 'border-slate-200 bg-slate-50 text-slate-700',
    accentColor: '#64748b',
  };
}

export interface GameRankBadgeProps {
  pct: number;
  compact?: boolean;
  className?: string;
}

export const GameRankBadge: React.FC<GameRankBadgeProps> = ({
  pct,
  compact = false,
  className = '',
}) => {
  const tier = resolveClinicalRankTier(pct);

  return (
    <span
      data-testid="game-rank-badge"
      className={`game-badge-shimmer relative overflow-hidden inline-flex items-center gap-1.5 rounded-full border font-black whitespace-nowrap shadow-xs ${tier.pillClass} ${
        compact ? 'px-2.5 py-0.5 text-[10px]' : 'px-3 py-1 text-xs'
      } ${className}`}
    >
      <svg
        viewBox="0 0 20 20"
        className={compact ? 'h-3.5 w-3.5 shrink-0' : 'h-4 w-4 shrink-0'}
        fill="none"
        aria-hidden="true"
      >
        <path
          d="M10 1.8 L16.8 4.6 V9.8 C16.8 14 13.9 17.2 10 18.6 C6.1 17.2 3.2 14 3.2 9.8 V4.6 Z"
          fill={tier.accentColor}
          fillOpacity="0.16"
          stroke={tier.accentColor}
          strokeWidth="1.7"
        />
        <path
          d="M10 5.4 L11.3 8.1 L14.2 8.5 L12.1 10.5 L12.6 13.4 L10 12 L7.4 13.4 L7.9 10.5 L5.8 8.5 L8.7 8.1 Z"
          fill={tier.accentColor}
        />
      </svg>
      <span>RANK {tier.grade}</span>
      {!compact && <span className="font-bold opacity-85">• {tier.labelTh}</span>}
    </span>
  );
};

export interface GameStarRatingProps {
  earned: number;
  total?: number;
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

const STAR_SIZE_PX: Record<'sm' | 'md' | 'lg', number> = {
  sm: 20,
  md: 28,
  lg: 42,
};

export const GameStarRating: React.FC<GameStarRatingProps> = ({
  earned,
  total = 3,
  size = 'md',
  className = '',
}) => {
  const safeTotal = Math.max(1, Math.min(5, Math.round(total)));
  const safeEarned = Math.max(0, Math.min(safeTotal, Math.round(earned)));
  const px = STAR_SIZE_PX[size];

  return (
    <div
      data-testid="game-star-rating"
      aria-label={`ได้ ${safeEarned} จาก ${safeTotal} ดาว`}
      className={`inline-flex items-center justify-center gap-1.5 ${className}`}
    >
      {Array.from({ length: safeTotal }, (_, idx) => {
        const filled = idx < safeEarned;
        const isCenterHero = safeTotal === 3 && idx === 1;
        const dim = isCenterHero ? Math.round(px * 1.18) : px;
        return (
          <svg
            key={idx}
            viewBox="0 0 24 24"
            width={dim}
            height={dim}
            className={`${
              filled ? 'game-combo-pop drop-shadow-sm' : 'opacity-40'
            } ${isCenterHero ? '-mt-2' : ''}`}
            style={{ animationDelay: `${idx * 95}ms` }}
            aria-hidden="true"
          >
            <path
              d="M12 2.4 L14.9 8.4 L21.5 9.3 L16.7 13.9 L17.9 20.5 L12 17.3 L6.1 20.5 L7.3 13.9 L2.5 9.3 L9.1 8.4 Z"
              fill={filled ? '#f59e0b' : '#cbd5e1'}
              stroke={filled ? '#d97706' : '#94a3b8'}
              strokeWidth="1.5"
              strokeLinejoin="round"
            />
            {filled && (
              <path
                d="M12 4.8 L13.8 8.7 L17.8 9.2 L12 11.4 Z"
                fill="#fef3c7"
                opacity="0.7"
              />
            )}
          </svg>
        );
      })}
    </div>
  );
};

