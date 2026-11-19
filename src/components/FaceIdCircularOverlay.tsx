import React from 'react';
import type {
  BiometricVerificationJson,
  FaceIdScreenPromptSpec,
} from '@drtechcares/face-client';
import { CheckCircle2, ShieldAlert, AlertTriangle } from 'lucide-react';

interface FaceIdCircularOverlayProps {
  /** 0 to 1 progress around the 64 circular Face ID ticks */
  progressRatio: number;
  inspection: BiometricVerificationJson | null;
  screenPrompt: FaceIdScreenPromptSpec;
  compact?: boolean;
}

const TOTAL_TICKS = 64;

const CHECKPOINT_LABELS: {
  key: keyof BiometricVerificationJson['checks'];
  label: string;
}[] = [
  { key: 'framing_ok', label: '1. Framing' },
  { key: 'lighting_ok', label: '2. Lighting' },
  { key: 'no_obstruction', label: '3. Clear Face' },
  { key: 'liveness_ok', label: '4. 3D Liveness' },
];

export function FaceIdCircularOverlay({
  progressRatio,
  inspection,
  screenPrompt,
  compact = false,
}: FaceIdCircularOverlayProps) {
  const clamped = Math.max(0, Math.min(1, progressRatio));
  const filledTicks = Math.round(clamped * TOTAL_TICKS);
  const isSpoof = inspection?.scan_status === 'SPOOF_DETECTED';
  const isPass = inspection?.scan_status === 'PASS';

  const ringSize = compact ? 220 : 256;
  const cx = ringSize / 2;
  const cy = ringSize / 2;
  const innerR = compact ? 90 : 106;
  const outerR = compact ? 104 : 122;

  return (
    <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-between p-3">
      {/* Top Primary + Sub-Prompt Banner */}
      <div className="w-full max-w-md rounded-2xl bg-white/92 backdrop-blur-sm border border-sky-200 px-4 py-2.5 text-center shadow-lg">
        <div className="flex items-center justify-center gap-2">
          {isSpoof ? (
            <ShieldAlert className="h-4 w-4 text-rose-500 shrink-0" />
          ) : isPass ? (
            <CheckCircle2 className="h-4 w-4 text-emerald-600 shrink-0" />
          ) : (
            <AlertTriangle className="h-4 w-4 text-[#0284C7] shrink-0" />
          )}
          <span className="text-sm font-extrabold text-[#0F4C81] tracking-tight">
            {screenPrompt.primaryPrompt}
          </span>
        </div>
        <p className="mt-0.5 text-[11px] text-[#0284C7]">{screenPrompt.subPrompt}</p>
      </div>

      {/* Center Circular iPhone Face ID Tick Ring */}
      <div className="relative flex items-center justify-center my-1">
        <svg
          width={ringSize}
          height={ringSize}
          viewBox={`0 0 ${ringSize} ${ringSize}`}
          className="drop-shadow-[0_0_14px_rgba(20,184,166,0.35)]"
        >
          {Array.from({ length: TOTAL_TICKS }, (_, idx) => {
            const angle = (idx / TOTAL_TICKS) * Math.PI * 2 - Math.PI / 2;
            const active = idx < filledTicks;
            const x1 = cx + Math.cos(angle) * innerR;
            const y1 = cy + Math.sin(angle) * innerR;
            const x2 = cx + Math.cos(angle) * (active ? outerR + 3 : outerR - 3);
            const y2 = cy + Math.sin(angle) * (active ? outerR + 3 : outerR - 3);

            const stroke = isSpoof
              ? '#f43f5e'
              : active
                ? '#10b981'
                : isPass
                  ? '#2dd4bf'
                  : 'rgba(148,163,184,0.38)';

            return (
              <line
                key={idx}
                x1={x1}
                y1={y1}
                x2={x2}
                y2={y2}
                stroke={stroke}
                strokeWidth={active ? 3 : 2}
                strokeLinecap="round"
              />
            );
          })}
        </svg>
      </div>

      {/* Bottom 4 Checkpoints + Short Instruction Badge */}
      <div className="w-full max-w-lg space-y-1.5">
        <div className="grid grid-cols-4 gap-1.5">
          {CHECKPOINT_LABELS.map((cp) => {
            const ok = inspection ? inspection.checks[cp.key] : false;
            return (
              <div
                key={cp.key}
                className={`rounded-lg px-2 py-1 text-center text-[10px] font-bold border inline-flex items-center justify-center gap-1 transition-all duration-200 ${
                  !inspection
                    ? 'bg-white/90 border-sky-200 text-[#0F4C81]'
                    : ok
                      ? 'bg-emerald-50/95 border-emerald-400 text-emerald-800 scale-[1.02]'
                      : 'bg-rose-50/95 border-rose-400 text-rose-700'
                }`}
              >
                {ok ? <CheckCircle2 className="w-3 h-3 shrink-0" /> : <span className="inline-block w-1.5 h-1.5 rounded-full bg-current shrink-0" />}
                <span>{cp.label}</span>
              </div>
            );
          })}
        </div>
        {inspection && (
          <div
            className={`rounded-xl px-3 py-1.5 text-center text-xs font-bold border ${
              isSpoof
                ? 'bg-rose-50/95 border-rose-400 text-rose-700'
                : isPass
                  ? 'bg-emerald-50/95 border-emerald-400 text-emerald-800'
                  : 'bg-sky-50/95 border-sky-300 text-[#0F4C81]'
            }`}
          >
            {inspection.user_instruction_th} ({inspection.detected_pose} • {inspection.error_code})
          </div>
        )}
      </div>
    </div>
  );
}

