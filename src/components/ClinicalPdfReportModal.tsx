import React from 'react';
import type { PatientRecord } from '@drtechcares/api-client';
import { Award, CheckCircle2, FileText, Printer, ShieldCheck, X } from 'lucide-react';
import type { ExerciseComparisonBreakdownItem } from '../types/clinicalAssessment';
import type { DoctorClinicalBrief } from '../utils/doctorClinicalBrief';
import type { RehabStockIndexSummary } from '../utils/rehabStockScoreEngine';
import type { UnifiedRehabMinigameSummary } from '../utils/unifiedRehabMinigameSummary';
import { formatThaiClinicalDate } from '../utils/clinicalChartAnalytics';
import { ClinicalQrCodeSvg } from '../utils/qrCodeSvg';

export interface ClinicalPdfReportModalProps {
  open: boolean;
  onClose: () => void;
  patient: PatientRecord;
  universalLinkCode: string;
  facilityName: string;
  assessorName: string;
  rehabStockIndex: RehabStockIndexSummary;
  unifiedSummary: UnifiedRehabMinigameSummary;
  doctorBrief: DoctorClinicalBrief;
  exerciseBreakdown: readonly ExerciseComparisonBreakdownItem[];
  firstMahidolScore: number | null;
  latestMahidolScore: number | null;
}

/**
 * Hospital-grade Printable Clinical Progress Summary Report Modal with genuine ISO/IEC 18004
 * scannable QR Code for Universal EHR linkage and 1-click browser print / Save-as-PDF.
 */
export function ClinicalPdfReportModal({
  open,
  onClose,
  patient,
  universalLinkCode,
  facilityName,
  assessorName,
  rehabStockIndex,
  unifiedSummary,
  doctorBrief,
  exerciseBreakdown,
  firstMahidolScore,
  latestMahidolScore,
}: ClinicalPdfReportModalProps) {
  if (!open) return null;

  const generatedAtLabel = formatThaiClinicalDate(new Date().toISOString());
  const ehrQrPayload = `https://drtechcares.app/ehr/${universalLinkCode}?hn=${encodeURIComponent(patient.hn)}`;
  const sideText =
    patient.affectedSide === 'LEFT'
      ? 'ข้างซ้าย (Left)'
      : patient.affectedSide === 'RIGHT'
        ? 'ข้างขวา (Right)'
        : patient.affectedSide === 'BOTH'
          ? 'ทั้งสองข้าง (Both)'
          : 'ประเมินทั่วไป';

  const handlePrintPdf = () => {
    if (typeof window !== 'undefined' && typeof window.print === 'function') {
      window.print();
    }
  };

  return (
    <div
      role="dialog"
      aria-modal="true"
      data-testid="clinical-pdf-report-modal"
      data-printable-clinical-report="true"
      aria-label="ใบรายงานสรุปผลฟื้นฟูทางคลินิก (พิมพ์ PDF)"
      className="fixed inset-0 z-50 bg-slate-900/65 backdrop-blur-sm flex items-center justify-center p-3 sm:p-5 overflow-y-auto print:static print:inset-auto print:bg-white print:p-0 print:block"
    >
      <div className="relative w-full max-w-4xl max-h-[92vh] flex flex-col rounded-3xl border-2 border-sky-200 bg-white shadow-2xl overflow-hidden print:max-h-none print:max-w-none print:rounded-none print:border-0 print:shadow-none">
        {/* Top Action Bar (Hidden when printing) */}
        <div className="shrink-0 flex flex-wrap items-center justify-between gap-2 border-b border-slate-200 bg-gradient-to-r from-[#0F4C81] to-teal-700 px-5 py-3 text-white print:hidden">
          <div className="flex items-center gap-2.5 min-w-0">
            <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-white/15">
              <FileText className="h-4 w-4" />
            </span>
            <div className="min-w-0">
              <h2 className="text-sm sm:text-base font-black leading-tight truncate">
                ใบสรุปรายงานฟื้นฟูสมรรถภาพทางคลินิก (Hospital PDF Report)
              </h2>
              <p className="text-[11px] font-semibold text-sky-100 truncate">
                พร้อม QR Code เชื่อมโยงเวชระเบียน Universal EHR และลายเซ็นนักกายภาพบำบัด
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2 shrink-0">
            <button
              type="button"
              data-testid="clinical-pdf-print-btn"
              onClick={handlePrintPdf}
              className="inline-flex items-center gap-1.5 rounded-xl bg-emerald-500 hover:bg-emerald-400 px-4 py-2 text-xs sm:text-sm font-black text-white shadow-sm cursor-pointer active:scale-95 transition"
            >
              <Printer className="h-4 w-4" />
              <span>พิมพ์ / บันทึก PDF</span>
            </button>
            <button
              type="button"
              onClick={onClose}
              aria-label="ปิดหน้าต่างใบสรุปรายงาน"
              className="flex h-9 w-9 items-center justify-center rounded-xl bg-white/15 hover:bg-white/25 text-white cursor-pointer"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        </div>

        {/* Printable A4 Clinical Document Body */}
        <div className="flex-1 overflow-y-auto p-5 sm:p-7 space-y-5 text-slate-900 print:overflow-visible print:p-6">
          {/* Document Letterhead + Real Scannable QR Code */}
          <div className="flex flex-wrap items-start justify-between gap-4 border-b-2 border-[#0F4C81] pb-4">
            <div className="space-y-1.5 min-w-0 flex-1">
              <div className="inline-flex items-center gap-2">
                <span className="inline-flex h-8 w-8 items-center justify-center rounded-xl bg-[#0F4C81] text-white font-black text-sm">
                  DT
                </span>
                <div>
                  <div className="text-base sm:text-lg font-black tracking-tight text-[#0F4C81] leading-none">
                    DR.TECH.CARE • Smart AI Rehabilitation Report
                  </div>
                  <div className="text-xs font-bold text-slate-600 mt-0.5">
                    แบบรายงานสรุปพัฒนาการกายภาพบำบัดและฝึกสมองผู้สูงอายุ (เกณฑ์มหิดล 0-6)
                  </div>
                </div>
              </div>
              <div className="text-xs font-semibold text-slate-600 pt-1">
                สถานพยาบาล / จุดบริการ: <strong className="text-slate-900">{facilityName}</strong>
              </div>
              <div className="text-xs font-semibold text-slate-600">
                วันที่ออกรายงาน: <strong className="text-slate-900">{generatedAtLabel}</strong>
              </div>
            </div>

            <div className="flex items-center gap-3 rounded-2xl border-2 border-slate-200 bg-slate-50/70 p-2.5 shrink-0">
              <ClinicalQrCodeSvg
                value={ehrQrPayload}
                size={88}
                className="rounded-lg border border-slate-200 bg-white p-0.5 shrink-0"
              />
              <div className="space-y-0.5 text-left max-w-[140px]">
                <div className="inline-flex items-center gap-1 text-[10.5px] font-black text-emerald-700">
                  <ShieldCheck className="h-3.5 w-3.5 shrink-0" />
                  <span>Universal EHR QR</span>
                </div>
                <div className="font-mono text-[11px] font-black text-slate-900 break-all">
                  {universalLinkCode}
                </div>
                <div className="text-[10px] font-semibold text-slate-500 leading-snug">
                  สแกนเพื่อเปิดประวัติการฟื้นฟูและส่งต่อ รพ.
                </div>
              </div>
            </div>
          </div>

          {/* Patient Demographic & Diagnosis Summary Grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2.5 rounded-2xl border border-slate-200 bg-slate-50/70 p-3.5 text-xs">
            <div>
              <div className="text-[11px] font-bold text-slate-500">ชื่อ-นามสกุลผู้ป่วย</div>
              <div className="text-sm font-black text-slate-900">{patient.fullName}</div>
            </div>
            <div>
              <div className="text-[11px] font-bold text-slate-500">รหัสผู้ป่วย (HN) / อายุ</div>
              <div className="text-sm font-black text-[#0F4C81]">
                {patient.hn} • {patient.age} ปี ({patient.gender || 'ไม่ระบุ'})
              </div>
            </div>
            <div>
              <div className="text-[11px] font-bold text-slate-500">ข้างที่เน้นฟื้นฟู</div>
              <div className="text-sm font-black text-emerald-800">{sideText}</div>
            </div>
            <div>
              <div className="text-[11px] font-bold text-slate-500">รหัสเชื่อมโยงเวชระเบียน</div>
              <div className="font-mono text-sm font-black text-slate-900">{universalLinkCode}</div>
            </div>
            <div className="sm:col-span-2 lg:col-span-4 border-t border-slate-200 pt-2">
              <span className="font-bold text-slate-500">การวินิจฉัย / เป้าหมายการฟื้นฟู: </span>
              <span className="font-extrabold text-slate-900">
                {patient.diagnosis || 'ฟื้นฟูสมรรถภาพการเคลื่อนไหวข้อต่อและกิจวัตรประจำวันผู้สูงอายุ'}
              </span>
            </div>
          </div>

          {/* 4 Clinical KPI Headline Boxes */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            <div className="rounded-2xl border-2 border-emerald-200 bg-emerald-50/60 p-3 space-y-1">
              <div className="text-[11px] font-bold text-emerald-800">ดัชนีฟื้นฟูรวม (Rehab Index)</div>
              <div className="text-2xl font-black text-emerald-700 tabular-nums">
                {(rehabStockIndex.latestPoint?.smartTotalPct ?? 0).toFixed(1)}%
              </div>
              <div className="text-[11px] font-extrabold text-slate-700">
                {(rehabStockIndex.latestPoint?.smartEarnedScore ?? 0).toFixed(0)} / {rehabStockIndex.maxTotalScore} คะแนน
              </div>
            </div>

            <div className="rounded-2xl border-2 border-sky-200 bg-sky-50/60 p-3 space-y-1">
              <div className="text-[11px] font-bold text-[#0F4C81]">คุณภาพองศา ROM เฉลี่ย</div>
              <div className="text-2xl font-black text-[#0F4C81] tabular-nums">
                {unifiedSummary.rehab.latestTargetPct !== null
                  ? `${unifiedSummary.rehab.latestTargetPct.toFixed(0)}%`
                  : '--'}
              </div>
              <div className="text-[11px] font-extrabold text-slate-700">
                ผ่านการฝึกแล้ว {unifiedSummary.rehab.totalVisits} ครั้ง
              </div>
            </div>

            <div className="rounded-2xl border-2 border-teal-200 bg-teal-50/60 p-3 space-y-1">
              <div className="text-[11px] font-bold text-teal-800">กำลังกล้ามเนื้อ (มหิดล 0-6)</div>
              <div className="text-2xl font-black text-teal-700 tabular-nums">
                {latestMahidolScore !== null ? `${latestMahidolScore}/6` : '--'}
              </div>
              <div className="text-[11px] font-extrabold text-slate-700">
                {firstMahidolScore !== null && latestMahidolScore !== null
                  ? `แรกเข้า ${firstMahidolScore}/6 -> ล่าสุด ${latestMahidolScore}/6`
                  : 'เกณฑ์มาตรฐานคณะกายภาพ มหิดล'}
              </div>
            </div>

            <div className="rounded-2xl border-2 border-indigo-200 bg-indigo-50/60 p-3 space-y-1">
              <div className="text-[11px] font-bold text-indigo-800">มินิเกมฝึกสมอง (Cognitive)</div>
              <div className="text-2xl font-black text-indigo-700 tabular-nums">
                {unifiedSummary.minigame.totalRounds > 0 &&
                unifiedSummary.minigame.averageAccuracyPct !== null
                  ? `${unifiedSummary.minigame.averageAccuracyPct.toFixed(0)}%`
                  : '--'}
              </div>
              <div className="text-[11px] font-extrabold text-slate-700">
                จำนวนฝึก {unifiedSummary.minigame.totalRounds} รอบ
              </div>
            </div>
          </div>

          {/* Executive Clinical Verdict Box */}
          <div className="rounded-2xl border border-emerald-200 bg-gradient-to-r from-emerald-50/80 to-sky-50/60 p-3.5 space-y-1.5">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <span className="inline-flex items-center gap-1.5 text-sm font-black text-slate-900">
                <Award className="h-4 w-4 text-emerald-600" />
                <span>{unifiedSummary.overall.verdict.headlineTh}</span>
              </span>
              <span className="rounded-full bg-emerald-600 px-2.5 py-0.5 text-[11px] font-black text-white">
                {doctorBrief.statusBadgeTh}
              </span>
            </div>
            <p className="text-xs font-semibold text-slate-700 leading-relaxed">
              {doctorBrief.executiveSummaryTh} — {unifiedSummary.overall.verdict.detailTh}
            </p>
          </div>

          {/* Per-Exercise ROM Progress Table */}
          <div className="space-y-2">
            <h3 className="text-sm font-black text-[#0F4C81] border-b border-slate-200 pb-1">
              ตารางเปรียบเทียบพัฒนาการองศาข้อต่อ (ROM) และระดับกำลังกล้ามเนื้อรายท่า
            </h3>
            {exerciseBreakdown.length === 0 ? (
              <div className="rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-xs font-semibold text-slate-500">
                ยังไม่มีประวัติบันทึกผลรายท่าสำหรับการออกตารางเปรียบเทียบ
              </div>
            ) : (
              <div className="overflow-x-auto rounded-2xl border border-slate-200">
                <table className="w-full text-left text-xs border-collapse">
                  <thead>
                    <tr className="bg-slate-100 text-slate-700 font-black">
                      <th className="px-3 py-2">ท่ากายภาพบำบัด</th>
                      <th className="px-2.5 py-2 text-center">ครั้งแรก</th>
                      <th className="px-2.5 py-2 text-center">ล่าสุด / เป้า</th>
                      <th className="px-2.5 py-2 text-center">พัฒนาการ</th>
                      <th className="px-2.5 py-2 text-center">ระดับ 0-6</th>
                      <th className="px-3 py-2 w-36">บรรลุเป้าหมาย</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-200 bg-white">
                    {exerciseBreakdown.map((row) => {
                      const pct = Math.min(100, Math.max(0, Math.round(row.latestTargetAchievementPct)));
                      return (
                        <tr key={row.key}>
                          <td className="px-3 py-2 font-extrabold text-slate-900">
                            {row.exerciseNameTh}
                            <span className="ml-1.5 text-[10.5px] font-bold text-slate-500">
                              ({row.visitCount} ครั้ง)
                            </span>
                          </td>
                          <td className="px-2.5 py-2 text-center font-bold text-slate-600 tabular-nums">
                            {row.firstRomDeg.toFixed(0)}°
                          </td>
                          <td className="px-2.5 py-2 text-center font-black text-[#0F4C81] tabular-nums">
                            {row.latestRomDeg.toFixed(0)}° / {row.targetRomDeg.toFixed(0)}°
                          </td>
                          <td className="px-2.5 py-2 text-center font-black tabular-nums">
                            <span
                              className={
                                row.deltaRomDeg > 0
                                  ? 'text-emerald-700'
                                  : row.deltaRomDeg < 0
                                    ? 'text-rose-600'
                                    : 'text-slate-600'
                              }
                            >
                              {row.deltaRomDeg > 0 ? `+${row.deltaRomDeg.toFixed(0)}°` : `${row.deltaRomDeg.toFixed(0)}°`}
                            </span>
                          </td>
                          <td className="px-2.5 py-2 text-center font-black text-teal-800 tabular-nums">
                            {row.firstScore}/6 -&gt; {row.latestScore}/6
                          </td>
                          <td className="px-3 py-2">
                            <div className="flex items-center gap-2">
                              <div className="flex-1 h-2.5 rounded-full bg-slate-100 overflow-hidden">
                                <div
                                  className="h-full rounded-full bg-emerald-600"
                                  style={{ width: `${pct}%` }}
                                />
                              </div>
                              <span className="w-9 text-right font-black text-emerald-700 tabular-nums">
                                {Math.round(row.latestTargetAchievementPct)}%
                              </span>
                            </div>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          {/* Clinical Recommendations & PT Signature Block */}
          <div className="grid grid-cols-1 md:grid-cols-12 gap-4 pt-1">
            <div className="md:col-span-7 rounded-2xl border border-slate-200 bg-slate-50/70 p-3.5 space-y-1.5">
              <div className="text-xs font-black text-[#0F4C81]">ข้อเสนอแนะเชิงคลินิกสำหรับแพทย์และญาติ</div>
              <ul className="space-y-1 text-xs font-semibold text-slate-700">
                {doctorBrief.doctorRecommendations.slice(0, 3).map((rec: string, idx: number) => (
                  <li key={idx} className="flex items-start gap-1.5">
                    <CheckCircle2 className="h-3.5 w-3.5 shrink-0 text-emerald-600 mt-0.5" />
                    <span>{rec}</span>
                  </li>
                ))}
              </ul>
            </div>

            <div className="md:col-span-5 rounded-2xl border border-slate-200 bg-white p-3.5 flex flex-col justify-between text-xs">
              <div className="text-[11px] font-bold text-slate-500">
                รับรองข้อมูลผลการตรวจประเมินและฟื้นฟูสมรรถภาพ
              </div>
              <div className="pt-6 space-y-1 text-center">
                <div className="border-b border-dashed border-slate-400 mx-4" />
                <div className="font-black text-slate-900">{assessorName}</div>
                <div className="text-[11px] font-semibold text-slate-500">
                  ผู้ตรวจประเมิน / นักกายภาพบำบัดประจำศูนย์ ({generatedAtLabel})
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

