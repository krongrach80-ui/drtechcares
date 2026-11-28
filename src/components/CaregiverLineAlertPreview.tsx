import React from 'react';
import { BellRing, CheckCircle2, ShieldAlert, X } from 'lucide-react';

export interface CaregiverAlertPayload {
  id: string;
  kind: 'SESSION_COMPLETE' | 'SAFETY_ALERT' | 'OWNER_GUARD';
  titleTh: string;
  bodyTh: string;
  patientName: string;
  metricBadgeTh?: string;
  timestampLabel: string;
}

export interface CaregiverLineAlertPreviewProps {
  alert: CaregiverAlertPayload | null;
  onDismiss: () => void;
}

/**
 * Real-time LINE Care / Family & PT Notification Preview card demonstrating how Dr.Tech.Care
 * immediately notifies family caregivers and clinical staff of completed milestones or safety events.
 */
export function CaregiverLineAlertPreview({
  alert,
  onDismiss,
}: CaregiverLineAlertPreviewProps) {
  if (!alert) return null;

  const isSafety = alert.kind === 'SAFETY_ALERT' || alert.kind === 'OWNER_GUARD';

  return (
    <div
      data-testid="caregiver-line-alert-preview"
      role="status"
      aria-live="polite"
      className="pointer-events-auto fixed bottom-16 right-4 z-50 w-80 sm:w-96 rounded-2xl border-2 bg-white shadow-2xl overflow-hidden animate-toast-in"
      style={{ borderColor: isSafety ? '#f43f5e' : '#06C755' }}
    >
      {/* LINE-Inspired Caregiver Bot Header */}
      <div
        className={`px-3.5 py-2 flex items-center justify-between gap-2 text-white ${
          isSafety
            ? 'bg-gradient-to-r from-rose-600 to-amber-600'
            : 'bg-gradient-to-r from-[#06C755] to-emerald-600'
        }`}
      >
        <div className="flex items-center gap-2 min-w-0">
          <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-white/20">
            {isSafety ? <ShieldAlert className="h-3.5 w-3.5" /> : <BellRing className="h-3.5 w-3.5" />}
          </span>
          <div className="min-w-0">
            <div className="text-xs font-black tracking-tight truncate">
              Dr.Tech.Care Family Bot (LINE Care)
            </div>
            <div className="text-[10px] font-bold text-white/90">
              แจ้งเตือนญาติและนักกายภาพ • {alert.timestampLabel}
            </div>
          </div>
        </div>
        <button
          type="button"
          onClick={onDismiss}
          aria-label="ปิดการแจ้งเตือนญาติ"
          className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-white/15 hover:bg-white/25 cursor-pointer"
        >
          <X className="h-3.5 w-3.5" />
        </button>
      </div>

      {/* Message Card Body */}
      <div className="p-3.5 space-y-2 bg-gradient-to-b from-white to-slate-50/70">
        <div className="flex items-center justify-between gap-2">
          <span className="text-xs sm:text-sm font-black text-slate-900 truncate">
            {alert.titleTh}
          </span>
          {alert.metricBadgeTh && (
            <span
              className={`shrink-0 rounded-full px-2 py-0.5 text-[11px] font-black ${
                isSafety
                  ? 'bg-rose-100 text-rose-800 border border-rose-200'
                  : 'bg-emerald-100 text-emerald-800 border border-emerald-200'
              }`}
            >
              {alert.metricBadgeTh}
            </span>
          )}
        </div>

        <p className="text-xs font-semibold text-slate-700 leading-relaxed">
          {alert.bodyTh}
        </p>

        <div className="flex items-center justify-between gap-2 pt-1 border-t border-slate-100 text-[11px] font-bold">
          <span className="text-slate-500 truncate">ผู้สูงอายุ: {alert.patientName}</span>
          <span className="inline-flex items-center gap-1 text-emerald-700 shrink-0">
            <CheckCircle2 className="h-3.5 w-3.5" />
            <span>ส่งถึงญาติและ PT แล้ว</span>
          </span>
        </div>
      </div>
    </div>
  );
}

