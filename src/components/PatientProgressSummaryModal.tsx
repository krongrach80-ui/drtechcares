import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  Activity,
  Award,
  Brain,
  Calendar,
  CheckCircle2,
  Printer,
  TrendingUp,
  Volume2,
  VolumeX,
  X,
} from 'lucide-react';
import {
  buildTtsProxyUrl,
  pickSpeechVoice,
  sanitizeSpeechText,
} from './VoiceCoachBanner';
import type {
  PatientProgressSummaryReport,
  ProgressVerdictTone,
} from '../utils/unifiedRehabMinigameSummary';

export interface PatientProgressSummaryModalProps {
  open?: boolean;
  variant?: 'modal' | 'embedded';
  report: PatientProgressSummaryReport;
  onClose?: () => void;
  closeLabelTh?: string;
}

function resolveVerdictChipClasses(tone: ProgressVerdictTone): string {
  switch (tone) {
    case 'emerald':
      return 'bg-emerald-700 text-white';
    case 'sky':
      return 'bg-[#0F4C81] text-white';
    case 'amber':
      return 'bg-amber-600 text-white';
    case 'indigo':
      return 'bg-indigo-600 text-white';
    default:
      return 'bg-slate-600 text-white';
  }
}

function resolveVerdictBoxClasses(tone: ProgressVerdictTone): string {
  switch (tone) {
    case 'emerald':
      return 'border-emerald-200 bg-emerald-50/75 text-emerald-950';
    case 'sky':
      return 'border-sky-200 bg-sky-50/75 text-slate-900';
    case 'amber':
      return 'border-amber-300 bg-amber-50/85 text-amber-950';
    case 'indigo':
      return 'border-indigo-200 bg-indigo-50/75 text-indigo-950';
    default:
      return 'border-slate-200 bg-slate-50 text-slate-800';
  }
}

export function PatientProgressSummaryModal({
  open = true,
  variant = 'modal',
  report,
  onClose,
  closeLabelTh = 'ปิดหน้าต่าง',
}: PatientProgressSummaryModalProps) {
  const [isSpeaking, setIsSpeaking] = useState(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  const stopVoice = useCallback(() => {
    if (audioRef.current) {
      try {
        audioRef.current.pause();
        audioRef.current.currentTime = 0;
      } catch {
        // Ignore audio reset errors
      }
      audioRef.current = null;
    }
    if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
      try {
        window.speechSynthesis.cancel();
      } catch {
        // Ignore unavailable speechSynthesis
      }
    }
    setIsSpeaking(false);
  }, []);

  useEffect(() => {
    return () => {
      stopVoice();
    };
  }, [stopVoice]);

  const handleToggleReadAloud = () => {
    if (isSpeaking) {
      stopVoice();
      return;
    }
    const cleanText = sanitizeSpeechText(report.voiceSummaryTh);
    if (!cleanText || typeof window === 'undefined') return;
    stopVoice();
    setIsSpeaking(true);

    const fallbackWebSpeech = () => {
      if (!('speechSynthesis' in window) || typeof SpeechSynthesisUtterance === 'undefined') {
        setIsSpeaking(false);
        return;
      }
      try {
        const synth = window.speechSynthesis;
        const utter = new SpeechSynthesisUtterance(cleanText);
        utter.lang = 'th-TH';
        const matched = pickSpeechVoice(synth.getVoices(), 'th-TH');
        if (matched) utter.voice = matched;
        utter.rate = 0.96;
        utter.onend = () => setIsSpeaking(false);
        utter.onerror = () => setIsSpeaking(false);
        synth.speak(utter);
      } catch {
        setIsSpeaking(false);
      }
    };

    if (typeof Audio !== 'undefined') {
      try {
        const audio = new Audio(buildTtsProxyUrl(cleanText, 'th'));
        audioRef.current = audio;
        audio.onended = () => setIsSpeaking(false);
        audio.onerror = fallbackWebSpeech;
        const p = audio.play();
        if (p && typeof p.catch === 'function') {
          p.catch(fallbackWebSpeech);
        }
        return;
      } catch {
        // Fall through to Web Speech API
      }
    }
    fallbackWebSpeech();
  };

  const handlePrint = () => {
    if (typeof window !== 'undefined' && typeof window.print === 'function') {
      window.print();
    }
  };

  if (variant === 'modal' && !open) return null;

  // Build grouped comparison bar chart data (Previous vs Latest for every exercise + optional Minigame)
  const hasRomBars = report.exerciseBars.length > 0;
  const maxBarRom = hasRomBars
    ? Math.max(
        180,
        Math.ceil(
          Math.max(
            ...report.exerciseBars.map((b) =>
              Math.max(b.currentRomDeg, b.previousRomDeg ?? 0, b.targetRomDeg),
            ),
            120,
          ) / 30,
        ) * 30,
      )
    : 180;

  const chartGroups: Array<{
    key: string;
    label: string;
    prevRatio: number | null;
    currRatio: number;
    prevLabel: string | null;
    currLabel: string;
    deltaText: string | null;
    deltaPositive: boolean | null;
    isMinigame: boolean;
  }> = report.exerciseBars.map((bar) => ({
    key: bar.exerciseCode,
    label: bar.shortLabelTh,
    prevRatio:
      bar.previousRomDeg !== null ? Math.max(0, Math.min(1, bar.previousRomDeg / maxBarRom)) : null,
    currRatio: Math.max(0.04, Math.min(1, bar.currentRomDeg / maxBarRom)),
    prevLabel: bar.previousRomDeg !== null ? `${bar.previousRomDeg.toFixed(0)}°` : null,
    currLabel: `${bar.currentRomDeg.toFixed(0)}°`,
    deltaText:
      bar.deltaDeg !== null
        ? bar.deltaDeg === 0
          ? 'เท่าเดิม'
          : `${bar.deltaDeg > 0 ? '+' : ''}${bar.deltaDeg.toFixed(1)}°`
        : `${bar.achievedPct.toFixed(0)}% เป้า`,
    deltaPositive: bar.deltaDeg !== null ? bar.deltaDeg >= 0 : bar.achievedPct >= 80,
    isMinigame: false,
  }));

  if (report.minigameComparison.played && report.minigameComparison.currentAccuracyPct !== null) {
    const prevAcc = report.minigameComparison.previousAccuracyPct;
    const currAcc = report.minigameComparison.currentAccuracyPct;
    const deltaAcc = report.minigameComparison.deltaAccuracyPct;
    chartGroups.push({
      key: 'MINIGAME_COGNITIVE',
      label: 'มินิเกม',
      prevRatio: prevAcc !== null ? Math.max(0, Math.min(1, prevAcc / 100)) : null,
      currRatio: Math.max(0.05, Math.min(1, currAcc / 100)),
      prevLabel: prevAcc !== null ? `${prevAcc}%` : null,
      currLabel: `${currAcc}%`,
      deltaText:
        deltaAcc !== null
          ? deltaAcc === 0
            ? 'เท่าเดิม'
            : `${deltaAcc > 0 ? '+' : ''}${deltaAcc}%`
          : `${report.minigameComparison.currentCorrect}/${report.minigameComparison.currentTotal}`,
      deltaPositive: deltaAcc !== null ? deltaAcc >= 0 : currAcc >= 75,
      isMinigame: true,
    });
  }

  const svgW = Math.max(460, chartGroups.length * 118 + 70);
  const svgH = 216;
  const padL = 44;
  const padR = 16;
  const padT = 26;
  const padB = 46;
  const plotW = svgW - padL - padR;
  const plotH = svgH - padT - padB;
  const groupW = chartGroups.length > 0 ? plotW / chartGroups.length : plotW;
  const barW = Math.min(34, Math.max(20, groupW * 0.28));
  const barGap = 8;

  const romDeltaHeadline =
    report.avgRomDeltaDeg !== null
      ? report.avgRomDeltaDeg >= 1
        ? `เพิ่มขึ้น +${report.avgRomDeltaDeg.toFixed(1)}° เทียบกับครั้งก่อน`
        : report.avgRomDeltaDeg <= -1
          ? `ลดลง ${Math.abs(report.avgRomDeltaDeg).toFixed(1)}° เทียบกับครั้งก่อน`
          : 'เท่าเดิม (0.0°) เทียบกับครั้งก่อน'
      : report.avgRomDeg !== null
        ? `เฉลี่ย ${report.avgRomDeg.toFixed(1)}° (บันทึกฐานครั้งแรก)`
        : 'รอข้อมูลวัดองศา ROM';

  const romDeltaColor =
    report.avgRomDeltaDeg !== null
      ? report.avgRomDeltaDeg >= 1
        ? 'text-emerald-700'
        : report.avgRomDeltaDeg <= -1
          ? 'text-amber-700'
          : 'text-[#0F4C81]'
      : 'text-slate-800';

  const cardBody = (
    <div
      data-testid="patient-progress-summary-card"
      className="w-full max-w-xl rounded-3xl border border-slate-200 bg-white shadow-2xl overflow-hidden text-left animate-popup-spring"
    >
      {/* Top Header matching Weekly Summary Mockup */}
      <div className="px-5 pt-5 pb-3 border-b border-slate-100 flex items-start justify-between gap-3">
        <div className="space-y-1 min-w-0">
          <div className="flex items-center gap-2">
            <Calendar className="h-5 w-5 text-slate-800 shrink-0" />
            <h2 className="text-lg sm:text-xl font-black text-slate-900 leading-tight truncate">
              {report.periodTitleTh}
            </h2>
          </div>
          <div className="flex flex-wrap items-center gap-2 text-xs sm:text-sm font-semibold text-slate-600">
            <span className="inline-flex items-center gap-1.5">
              <Calendar className="h-3.5 w-3.5 text-slate-400 shrink-0" />
              <span>{report.periodRangeTh}</span>
            </span>
            {report.patientName && (
              <span className="rounded-full bg-emerald-50 border border-emerald-200 px-2 py-0.5 text-[11px] font-extrabold text-emerald-800">
                {report.patientName}
                {report.patientHn ? ` (${report.patientHn})` : ''}
              </span>
            )}
          </div>
        </div>

        <div className="flex items-center gap-1.5 shrink-0">
          <button
            type="button"
            onClick={handleToggleReadAloud}
            title={isSpeaking ? 'หยุดเสียงอ่านสรุป' : 'ฟังเสียงอ่านสรุปผล'}
            className={`inline-flex items-center gap-1 rounded-xl border px-2.5 py-1.5 text-xs font-extrabold cursor-pointer transition-colors ${
              isSpeaking
                ? 'border-emerald-500 bg-emerald-600 text-white'
                : 'border-sky-200 bg-sky-50 text-[#0F4C81] hover:bg-sky-100'
            }`}
          >
            {isSpeaking ? <VolumeX className="h-4 w-4" /> : <Volume2 className="h-4 w-4" />}
            <span className="hidden sm:inline">{isSpeaking ? 'หยุดฟัง' : 'ฟังสรุป'}</span>
          </button>
          {variant === 'modal' && onClose && (
            <button
              type="button"
              onClick={() => {
                stopVoice();
                onClose();
              }}
              aria-label="ปิดหน้าต่างสรุปผล"
              className="rounded-xl p-1.5 text-slate-400 hover:bg-slate-100 hover:text-slate-700 cursor-pointer"
            >
              <X className="h-5 w-5" />
            </button>
          )}
        </div>
      </div>

      {/* Body Content */}
      <div className="p-5 space-y-4 max-h-[78vh] overflow-y-auto">
        {/* 1. Overall Rehab & Minigame Progress Bar */}
        <div className="space-y-2">
          <div className="flex items-baseline justify-between gap-2">
            <span className="text-sm sm:text-base font-extrabold text-slate-900">
              ความคืบหน้าภาพรวม (Rehab Progress):
            </span>
            <span
              data-testid="summary-overall-progress-pct"
              className="text-lg sm:text-xl font-black text-slate-900 tabular-nums"
            >
              {report.overallProgressPct}%
            </span>
          </div>
          <div className="h-3.5 w-full overflow-hidden rounded-full bg-slate-200/90">
            <div
              className="h-full rounded-full bg-gradient-to-r from-[#0F766E] via-[#059669] to-[#10B981] transition-all duration-500"
              style={{ width: `${Math.max(4, Math.min(100, report.overallProgressPct))}%` }}
            />
          </div>
        </div>

        {/* 2. Clear Patient Verdict Callout (ดีขึ้น / เท่าเดิม / ท่านทำได้ไม่เลว พยายามมากขึ้นกว่านี้นะครับ) */}
        <div
          data-testid="summary-modal-verdict-banner"
          className={`rounded-2xl border px-4 py-3 space-y-1 ${resolveVerdictBoxClasses(report.overallVerdict.tone)}`}
        >
          <div className="flex flex-wrap items-center gap-2">
            <span
              className={`inline-flex items-center gap-1 rounded-full px-3 py-0.5 text-xs font-black ${resolveVerdictChipClasses(report.overallVerdict.tone)}`}
            >
              <Award className="h-3.5 w-3.5 shrink-0" />
              <span>{report.overallVerdict.badgeTh}</span>
            </span>
            <span className="text-xs font-extrabold opacity-80">
              เปรียบเทียบกับรอบก่อนหน้า
            </span>
          </div>
          <p className="text-sm sm:text-base font-black leading-snug">
            {report.overallVerdict.headlineTh}
          </p>
        </div>

        {/* 3. Crisp Key Metric Rows matching the Mockup */}
        <div className="space-y-2 text-sm border-y border-slate-100 py-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <span className="font-extrabold text-slate-800">จำนวนครั้งที่ฝึก:</span>
            <span className="font-bold text-slate-900 text-right">
              {report.weeklySessionCount} ครั้ง / สัปดาห์{' '}
              <span className={report.weeklyGoalMet ? 'text-emerald-700' : 'text-amber-700'}>
                ({report.weeklyGoalMet ? 'เป็นไปตามเป้าหมาย' : `เป้าหมาย ${report.weeklyGoalCount} ครั้ง`})
              </span>
            </span>
          </div>

          <div className="flex flex-wrap items-center justify-between gap-2">
            <span className="font-extrabold text-slate-800">คะแนนการประเมินเฉลี่ย:</span>
            <span className="font-black text-slate-900 tabular-nums">
              {report.avgScore0To6 !== null ? `${report.avgScore0To6.toFixed(1)} / 6 คะแนน` : '-'}
            </span>
          </div>

          <div className="flex flex-wrap items-center justify-between gap-2">
            <span className="font-extrabold text-slate-800">องศาการเคลื่อนไหว (ROM) เฉลี่ย:</span>
            <span className={`font-black text-right ${romDeltaColor}`}>
              {report.avgRomDeg !== null && report.avgRomDeltaDeg !== null
                ? `${report.avgRomDeg.toFixed(1)}° • ${romDeltaHeadline}`
                : romDeltaHeadline}
            </span>
          </div>

          {report.minigameComparison.played && (
            <div className="flex flex-wrap items-center justify-between gap-2 pt-1 border-t border-slate-100">
              <span className="font-extrabold text-indigo-900 inline-flex items-center gap-1.5">
                <Brain className="h-4 w-4 text-indigo-600 shrink-0" />
                <span>ผลมินิเกมฝึกสมอง:</span>
              </span>
              <span className="font-black text-indigo-800 text-right">
                {report.minigameComparison.currentCorrect}/{report.minigameComparison.currentTotal} ข้อ (
                {report.minigameComparison.currentAccuracyPct}%) •{' '}
                {report.minigameComparison.verdict.badgeTh}
              </span>
            </div>
          )}
        </div>

        {/* 4. Side-by-Side Comparison Bar Chart (Previous vs Latest for Every Exercise + Minigame) */}
        {chartGroups.length > 0 && (
          <div data-testid="summary-comparison-bar-chart" className="space-y-2">
            <div className="flex flex-wrap items-center justify-center gap-4 text-xs font-bold text-slate-700">
              <span className="inline-flex items-center gap-1.5">
                <span className="h-3.5 w-3.5 rounded bg-slate-400/80 inline-block" />
                <span>ครั้งก่อนหน้า</span>
              </span>
              <span className="inline-flex items-center gap-1.5">
                <span className="h-3.5 w-3.5 rounded bg-emerald-700 inline-block" />
                <span>ครั้งนี้ (กายภาพ)</span>
              </span>
              {report.minigameComparison.played && (
                <span className="inline-flex items-center gap-1.5">
                  <span className="h-3.5 w-3.5 rounded bg-indigo-600 inline-block" />
                  <span>ครั้งนี้ (มินิเกม %)</span>
                </span>
              )}
            </div>

            <div className="w-full overflow-x-auto rounded-2xl border border-slate-100 bg-slate-50/45 p-2">
              <svg
                viewBox={`0 0 ${svgW} ${svgH}`}
                className="w-full h-auto min-w-[340px]"
                role="img"
                aria-label="กราฟแท่งเปรียบเทียบองศากายภาพและคะแนนมินิเกมระหว่างครั้งก่อนหน้ากับครั้งนี้"
              >
                {[0, 0.5, 1].map((frac) => {
                  const yVal = padT + plotH * (1 - frac);
                  const degLabel = Math.round(maxBarRom * frac);
                  return (
                    <g key={frac}>
                      <line
                        x1={padL}
                        y1={yVal}
                        x2={svgW - padR}
                        y2={yVal}
                        stroke={frac === 0 ? '#cbd5e1' : '#e2e8f0'}
                        strokeDasharray={frac === 0 ? undefined : '3 3'}
                        strokeWidth="1"
                      />
                      <text
                        x={padL - 6}
                        y={yVal + 3.5}
                        textAnchor="end"
                        fill="#64748b"
                        fontSize="10"
                        fontWeight="700"
                      >
                        {hasRomBars ? `${degLabel}°` : `${Math.round(frac * 100)}%`}
                      </text>
                    </g>
                  );
                })}

                {chartGroups.map((g, idx) => {
                  const cx = padL + groupW * idx + groupW / 2;
                  const currH = Math.max(8, g.currRatio * plotH);
                  const currY = padT + plotH - currH;
                  const hasPrev = g.prevRatio !== null;
                  const prevH = hasPrev ? Math.max(8, (g.prevRatio as number) * plotH) : 0;
                  const prevY = padT + plotH - prevH;

                  const prevX = cx - barW - barGap / 2;
                  const currX = hasPrev ? cx + barGap / 2 : cx - barW / 2;
                  const currFill = g.isMinigame ? '#4f46e5' : '#047857';

                  return (
                    <g key={g.key} data-testid={`summary-bar-${g.key}`}>
                      {hasPrev && (
                        <>
                          <rect
                            x={prevX}
                            y={prevY}
                            width={barW}
                            height={prevH}
                            rx="6"
                            fill="#94a3b8"
                          />
                          <text
                            x={prevX + barW / 2}
                            y={Math.max(12, prevY - 5)}
                            textAnchor="middle"
                            fill="#64748b"
                            fontSize="9.5"
                            fontWeight="800"
                          >
                            {g.prevLabel}
                          </text>
                        </>
                      )}

                      <rect
                        x={currX}
                        y={currY}
                        width={barW}
                        height={currH}
                        rx="6"
                        fill={currFill}
                      />
                      <text
                        x={currX + barW / 2}
                        y={Math.max(12, currY - 5)}
                        textAnchor="middle"
                        fill={g.isMinigame ? '#3730a3' : '#065f46'}
                        fontSize="10"
                        fontWeight="900"
                      >
                        {g.currLabel}
                      </text>

                      <text
                        x={cx}
                        y={padT + plotH + 17}
                        textAnchor="middle"
                        fill="#1e293b"
                        fontSize="11"
                        fontWeight="800"
                      >
                        {g.label}
                      </text>
                      {g.deltaText && (
                        <text
                          x={cx}
                          y={padT + plotH + 33}
                          textAnchor="middle"
                          fill={g.deltaPositive ? '#047857' : '#b45309'}
                          fontSize="9.5"
                          fontWeight="800"
                        >
                          {g.deltaText}
                        </text>
                      )}
                    </g>
                  );
                })}
              </svg>
            </div>
          </div>
        )}

        {/* 5. Personalized Clinical & Encouragement Feedback Section */}
        <div className="rounded-2xl border border-slate-200/90 bg-slate-50/70 p-4 space-y-1.5">
          <div className="flex items-center gap-1.5 text-sm font-black text-slate-900">
            {report.overallVerdict.code === 'IMPROVED' ? (
              <TrendingUp className="h-4 w-4 text-emerald-700 shrink-0" />
            ) : report.overallVerdict.code === 'STABLE' ? (
              <CheckCircle2 className="h-4 w-4 text-[#0F4C81] shrink-0" />
            ) : (
              <Activity className="h-4 w-4 text-amber-700 shrink-0" />
            )}
            <span>ความเห็น/ข้อเสนอแนะนักกายภาพและระบบ AI</span>
          </div>
          <p
            data-testid="summary-modal-recommendation"
            className="text-xs sm:text-sm font-semibold text-slate-700 leading-relaxed"
          >
            {report.recommendationTh}
          </p>
        </div>
      </div>

      {/* 6. Footer Action Buttons matching Mockup */}
      <div className="px-5 py-3.5 border-t border-slate-100 bg-slate-50/60 flex flex-wrap items-center justify-end gap-2.5">
        <button
          type="button"
          onClick={handlePrint}
          className="btn-tactile inline-flex items-center gap-2 rounded-2xl border border-slate-300 bg-white px-4 py-2.5 text-sm font-extrabold text-slate-800 hover:bg-slate-100 cursor-pointer"
        >
          <Printer className="h-4 w-4 text-slate-600" />
          <span>พิมพ์สรุป</span>
        </button>
        {onClose && (
          <button
            type="button"
            data-testid="summary-modal-close-btn"
            onClick={() => {
              stopVoice();
              onClose();
            }}
            className="btn-tactile inline-flex items-center gap-2 rounded-2xl bg-emerald-700 hover:bg-emerald-800 px-5 py-2.5 text-sm font-black text-white shadow-sm cursor-pointer"
          >
            <span>{closeLabelTh}</span>
          </button>
        )}
      </div>
    </div>
  );

  if (variant === 'embedded') {
    return cardBody;
  }

  return (
    <div
      data-testid="patient-progress-summary-modal"
      role="dialog"
      aria-modal="true"
      aria-label="สรุปผลการกายภาพและมินิเกม"
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/55 p-3 sm:p-5 backdrop-blur-[2px]"
      onClick={(e) => {
        if (e.target === e.currentTarget && onClose) {
          stopVoice();
          onClose();
        }
      }}
    >
      {cardBody}
    </div>
  );
}

