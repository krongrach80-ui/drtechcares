import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { PatientRecord, RehabSessionRecord, UserAccount } from '@drtechcares/api-client';
import { CLINICAL_EXERCISES, parseExerciseCodeAndSide } from '@drtechcares/pose-engine';
import {
  Activity,
  ArrowRight,
  Building2,
  Check,
  ClipboardList,
  Copy,
  FileText,
  History,
  Link2,
  RefreshCw,
  Search,
  Sparkles,
  TrendingUp,
} from 'lucide-react';
import {
  CLINICAL_FACILITY_PRESETS,
  createPatientAssessment,
  listPatientAssessments,
} from '../lib/clinicalAssessmentStore';
import {
  MAHIDOL_ASSESSMENT_SCALE_0_TO_6,
  computeLongitudinalProgressSummary,
  getMahidolRubricItem,
  resolvePatientUniversalLinkCode,
  type AssessmentTargetSide,
  type MahidolScore0To6,
  type PatientClinicalAssessmentRecord,
} from '../types/clinicalAssessment';

export interface PatientClinicalRecordPanelProps {
  patients: PatientRecord[];
  sessions: RehabSessionRecord[];
  currentUser: UserAccount;
  selectedPatientId: string | null;
  onSelectPatientId: (patientId: string) => void;
  onOpenUniversalEhrModal?: (patient: PatientRecord) => void;
}

const SIDE_OPTIONS: { value: AssessmentTargetSide; labelTh: string }[] = [
  { value: 'LEFT', labelTh: 'ซ้าย' },
  { value: 'RIGHT', labelTh: 'ขวา' },
  { value: 'BOTH', labelTh: 'ทั้งสองข้าง' },
];

function formatThaiDateTime(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString('th-TH', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

function normalizeDefaultSide(patient: PatientRecord | null, exCode: string): AssessmentTargetSide {
  const matched = CLINICAL_EXERCISES.find((e) => e.code === exCode || e.id === exCode);
  if (matched?.detection.side === 'both') return 'BOTH';
  if (patient?.affectedSide === 'LEFT') return 'LEFT';
  if (patient?.affectedSide === 'BOTH') return 'BOTH';
  return 'RIGHT';
}

export function PatientClinicalRecordPanel({
  patients,
  sessions,
  currentUser,
  selectedPatientId,
  onSelectPatientId,
  onOpenUniversalEhrModal,
}: PatientClinicalRecordPanelProps) {
  const [patientQuery, setPatientQuery] = useState('');
  const [records, setRecords] = useState<PatientClinicalAssessmentRecord[]>([]);
  const [loadingRecords, setLoadingRecords] = useState(false);
  const [syncingCrossFacility, setSyncingCrossFacility] = useState(false);
  const [copiedCode, setCopiedCode] = useState(false);
  const [exerciseFilter, setExerciseFilter] = useState<string>('ALL');

  // Assessment form states
  const [selectedScore, setSelectedScore] = useState<MahidolScore0To6>(3);
  const [exerciseCode, setExerciseCode] = useState<string>(
    CLINICAL_EXERCISES[0]?.code || 'STROKE_SHOULDER_FLEXION',
  );
  const [targetSide, setTargetSide] = useState<AssessmentTargetSide>('RIGHT');
  const [measuredRomInput, setMeasuredRomInput] = useState<string>('48.0');
  const [targetRomInput, setTargetRomInput] = useState<string>(
    String(CLINICAL_EXERCISES[0]?.detection.targetMinDeg ?? 125),
  );
  const [painVasInput, setPainVasInput] = useState<string>('2');
  const [facilityName, setFacilityName] = useState<string>(CLINICAL_FACILITY_PRESETS[0]);
  const [chiefComplaint, setChiefComplaint] = useState<string>('');
  const [handoverNote, setHandoverNote] = useState<string>('');
  const [nextGoal, setNextGoal] = useState<string>('');
  const [saving, setSaving] = useState(false);
  const [toast, setToast] = useState<{ kind: 'success' | 'error' | 'info'; message: string } | null>(
    null,
  );
  const toastTimerRef = useRef<number | null>(null);
  const copyTimerRef = useRef<number | null>(null);
  const latestLoadReqIdRef = useRef(0);
  const selectedPatientIdRef = useRef<string | null>(selectedPatientId ?? null);

  useEffect(() => {
    return () => {
      if (toastTimerRef.current !== null) window.clearTimeout(toastTimerRef.current);
      if (copyTimerRef.current !== null) window.clearTimeout(copyTimerRef.current);
    };
  }, []);

  const filteredPatients = useMemo(() => {
    const q = patientQuery.trim().toLowerCase();
    if (!q) return patients;
    return patients.filter((p) => {
      const linkCode = resolvePatientUniversalLinkCode(p).toLowerCase();
      return (
        p.fullName.toLowerCase().includes(q) ||
        p.hn.toLowerCase().includes(q) ||
        linkCode.includes(q)
      );
    });
  }, [patients, patientQuery]);

  const selectedPatient = useMemo(() => {
    if (selectedPatientId) {
      const found = patients.find((p) => p.id === selectedPatientId);
      if (found) return found;
    }
    return filteredPatients[0] ?? patients[0] ?? null;
  }, [patients, filteredPatients, selectedPatientId]);

  selectedPatientIdRef.current = selectedPatient?.id ?? null;

  useEffect(() => {
    if (selectedPatient && selectedPatient.id !== selectedPatientId) {
      onSelectPatientId(selectedPatient.id);
    }
  }, [selectedPatient, selectedPatientId, onSelectPatientId]);

  const showToast = useCallback((kind: 'success' | 'error' | 'info', message: string) => {
    if (toastTimerRef.current !== null) {
      window.clearTimeout(toastTimerRef.current);
      toastTimerRef.current = null;
    }
    setToast({ kind, message });
    toastTimerRef.current = window.setTimeout(() => {
      toastTimerRef.current = null;
      setToast((prev) => (prev?.message === message ? null : prev));
    }, 4500);
  }, []);

  const loadPatientHistory = useCallback(
    async (patient: PatientRecord | null, showSyncBanner = false) => {
      const reqId = ++latestLoadReqIdRef.current;
      if (!patient) {
        setRecords([]);
        setLoadingRecords(false);
        setSyncingCrossFacility(false);
        return;
      }
      const targetPatientId = patient.id;
      if (showSyncBanner) {
        setSyncingCrossFacility(true);
      } else {
        setLoadingRecords(true);
      }
      try {
        const list = await listPatientAssessments(patient, {
          includeCrossFacility: true,
          seedDemoIfEmpty: true,
        });
        if (
          reqId !== latestLoadReqIdRef.current ||
          selectedPatientIdRef.current !== targetPatientId
        ) {
          return;
        }
        setRecords(list);
        if (showSyncBanner) {
          const facilitiesCount = new Set(list.map((r) => r.facility_name)).size;
          showToast(
            'success',
            `ซิงก์ประวัติเชื่อมโยงข้ามสาขาสำเร็จ: พบ ${list.length} ครั้ง จาก ${facilitiesCount} สถานที่บริการ`,
          );
        }
      } catch (err) {
        if (
          reqId !== latestLoadReqIdRef.current ||
          selectedPatientIdRef.current !== targetPatientId
        ) {
          return;
        }
        showToast(
          'error',
          err instanceof Error ? err.message : 'ไม่สามารถโหลดเวชระเบียนผู้ป่วยได้',
        );
      } finally {
        if (reqId === latestLoadReqIdRef.current) {
          setLoadingRecords(false);
          setSyncingCrossFacility(false);
        }
      }
    },
    [showToast],
  );

  useEffect(() => {
    void loadPatientHistory(selectedPatient, false);
  }, [selectedPatient?.id, loadPatientHistory]);

  useEffect(() => {
    setRecords([]);
    setExerciseFilter('ALL');
    setChiefComplaint('');
    setHandoverNote('');
    setNextGoal('');
    setMeasuredRomInput('');
    if (!selectedPatient) return;
    setTargetSide(normalizeDefaultSide(selectedPatient, exerciseCode));
    const score = selectedPatient.motorAssessmentScore;
    if (typeof score === 'number' && score >= 0 && score <= 6) {
      setSelectedScore(Math.round(score) as MahidolScore0To6);
    }
  }, [selectedPatient?.id]);

  const handleExerciseChange = (nextCode: string) => {
    setExerciseCode(nextCode);
    const matched = CLINICAL_EXERCISES.find((e) => e.code === nextCode || e.id === nextCode);
    if (matched) {
      setTargetRomInput(String(matched.detection.targetMinDeg));
      setTargetSide(normalizeDefaultSide(selectedPatient, matched.code));
    }
  };

  const latestKioskRomInfo = useMemo(() => {
    if (!selectedPatient) return null;
    const normalizedSelectedBase = parseExerciseCodeAndSide(exerciseCode).baseCode;
    const patientKioskSessions = sessions
      .filter((s) => s.patientId === selectedPatient.id && s.maxRomDeg > 0)
      .slice()
      .sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime());

    const exactMatchSession = patientKioskSessions.find(
      (s) => parseExerciseCodeAndSide(s.exerciseId).baseCode === normalizedSelectedBase,
    );
    const fallbackSession = patientKioskSessions[0];
    if (exactMatchSession) {
      return {
        rom: exactMatchSession.maxRomDeg,
        label: exactMatchSession.exerciseName,
        at: exactMatchSession.createdAt,
      };
    }

    const autoSynced = records
      .filter((r) => r.source_channel === 'KIOSK_SESSION_AUTO_SYNC' && r.measured_rom_deg > 0)
      .slice()
      .reverse();
    const matchSync = autoSynced.find((r) => r.exercise_code === normalizedSelectedBase) ?? autoSynced[0];
    if (matchSync) {
      return {
        rom: matchSync.measured_rom_deg,
        label: matchSync.exercise_name_th,
        at: matchSync.assessed_at,
      };
    }
    if (fallbackSession) {
      return {
        rom: fallbackSession.maxRomDeg,
        label: fallbackSession.exerciseName,
        at: fallbackSession.createdAt,
      };
    }
    return null;
  }, [selectedPatient, sessions, records, exerciseCode]);

  const handlePullLatestKioskRom = () => {
    if (!latestKioskRomInfo) {
      showToast('info', 'ยังไม่พบค่าองศาล่าสุดจากตู้ Kiosk สำหรับผู้ป่วยรายนี้');
      return;
    }
    setMeasuredRomInput(latestKioskRomInfo.rom.toFixed(1));
    showToast(
      'info',
      `ดึงค่าองศาล่าสุด ${latestKioskRomInfo.rom.toFixed(1)}° (${latestKioskRomInfo.label}) เรียบร้อยแล้ว`,
    );
  };

  const universalLinkCode = selectedPatient
    ? resolvePatientUniversalLinkCode(selectedPatient)
    : 'DTC-EHR-PATIENT';

  const handleCopyLinkCode = async () => {
    try {
      if (typeof navigator !== 'undefined' && navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(universalLinkCode);
      }
      setCopiedCode(true);
      showToast('success', `คัดลอกรหัสส่งต่อ ${universalLinkCode} แล้ว`);
      if (copyTimerRef.current !== null) window.clearTimeout(copyTimerRef.current);
      copyTimerRef.current = window.setTimeout(() => {
        copyTimerRef.current = null;
        setCopiedCode(false);
      }, 2000);
    } catch {
      showToast('info', `รหัสส่งต่อ: ${universalLinkCode}`);
    }
  };

  const handleSubmitAssessment = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedPatient) return;
    const measuredRom = Number(measuredRomInput);
    const targetRom = Number(targetRomInput);
    if (!Number.isFinite(measuredRom) || measuredRom < 0 || measuredRom > 220) {
      showToast('error', 'กรุณาระบุองศาข้อต่อ (ROM) ระหว่าง 0 ถึง 220 องศา');
      return;
    }

    const matchedEx =
      CLINICAL_EXERCISES.find((ex) => ex.code === exerciseCode || ex.id === exerciseCode) ??
      CLINICAL_EXERCISES[0];
    const painParsed = painVasInput.trim() === '' ? null : Number(painVasInput);
    const safePain =
      painParsed !== null && Number.isFinite(painParsed)
        ? Math.max(0, Math.min(10, Math.round(painParsed)))
        : null;

    const submitPatientId = selectedPatient.id;
    setSaving(true);
    try {
      const saved = await createPatientAssessment({
        patient_id: submitPatientId,
        patient_hn: selectedPatient.hn,
        universal_link_code: universalLinkCode,
        facility_name: facilityName,
        assessor_staff_id: currentUser.id,
        assessor_staff_name: `${currentUser.fullName} (${currentUser.role === 'super_admin' ? 'ผู้อำนวยการ/Admin' : 'นักกายภาพบำบัด'})`,
        exercise_code: matchedEx.code,
        exercise_name_th: matchedEx.name.th,
        target_side: targetSide,
        initial_score_0_to_6: selectedScore,
        measured_rom_deg: Number(measuredRom.toFixed(1)),
        target_rom_deg: Number((Number.isFinite(targetRom) && targetRom > 0 ? targetRom : matchedEx.detection.targetMinDeg).toFixed(1)),
        pain_vas_0_to_10: safePain,
        chief_complaint: chiefComplaint || `ตรวจประเมินต่อเนื่องท่า ${matchedEx.name.th}`,
        clinical_handover_note:
          handoverNote ||
          `วัดมุม ROM ได้ ${measuredRom.toFixed(1)}° (คะแนนมหิดลระดับ ${selectedScore}/6) ติดตามความสม่ำเสมอการฝึกและระวังอาการชดเชย`,
        next_session_goal: nextGoal || `เพิ่มมุมข้อต่อสู่ ${(measuredRom + 12).toFixed(0)}° ในครั้งถัดไป`,
        source_channel: 'STAFF_MANUAL_ASSESSMENT',
      });

      if (selectedPatientIdRef.current === submitPatientId) {
        const refreshed = await listPatientAssessments(selectedPatient, {
          includeCrossFacility: true,
          seedDemoIfEmpty: false,
        });
        if (selectedPatientIdRef.current === submitPatientId) {
          setRecords(refreshed);
          setChiefComplaint('');
          setHandoverNote('');
          setNextGoal('');
        }
      }
      showToast(
        'success',
        `บันทึกผลประเมินครั้งที่ ${saved.visit_number} (ระดับ ${saved.initial_score_0_to_6}/6, ROM ${saved.measured_rom_deg.toFixed(1)}°) เข้าแฟ้ม ${selectedPatient.fullName} แล้ว`,
      );
    } catch (err) {
      showToast(
        'error',
        err instanceof Error ? err.message : 'ไม่สามารถบันทึกผลประเมินได้',
      );
    } finally {
      setSaving(false);
    }
  };

  const exerciseOptionsInHistory = useMemo(() => {
    const map = new Map<string, string>();
    for (const r of records) {
      if (r.exercise_code && !map.has(r.exercise_code)) {
        map.set(r.exercise_code, r.exercise_name_th || r.exercise_code);
      }
    }
    return Array.from(map.entries());
  }, [records]);

  const summary = useMemo(
    () => computeLongitudinalProgressSummary(records, exerciseFilter),
    [records, exerciseFilter],
  );

  const selectedRubric = getMahidolRubricItem(selectedScore);

  // Pure SVG Dual-Metric Progress Chart geometry
  const chartData = useMemo(() => {
    const pts = summary.visitsWithDelta.slice(-10);
    const width = 680;
    const height = 230;
    const padL = 46;
    const padR = 28;
    const padT = 30;
    const padB = 36;
    const plotW = width - padL - padR;
    const plotH = height - padT - padB;
    const maxRom = Math.max(
      150,
      ...pts.map((p) => Math.max(p.record.measured_rom_deg, p.record.target_rom_deg * 0.9)),
    );

    const nodes = pts.map((item, idx) => {
      const x =
        pts.length === 1
          ? padL + plotW / 2
          : padL + (idx / Math.max(1, pts.length - 1)) * plotW;
      const ratio = Math.max(0, Math.min(1, item.record.measured_rom_deg / maxRom));
      const y = padT + plotH - ratio * plotH;
      return {
        id: item.record.id,
        x,
        y,
        visitNumber: item.record.visit_number,
        rom: item.record.measured_rom_deg,
        score: item.record.initial_score_0_to_6,
        facility: item.record.facility_name,
        source: item.record.source_channel,
      };
    });

    const polylinePoints = nodes.map((n) => `${n.x.toFixed(1)},${n.y.toFixed(1)}`).join(' ');
    const areaPoints =
      nodes.length > 0
        ? `${nodes[0].x.toFixed(1)},${(padT + plotH).toFixed(1)} ${polylinePoints} ${nodes[nodes.length - 1].x.toFixed(1)},${(padT + plotH).toFixed(1)}`
        : '';

    return { width, height, padL, padR, padT, padB, plotW, plotH, maxRom, nodes, polylinePoints, areaPoints };
  }, [summary.visitsWithDelta]);

  if (patients.length === 0) {
    return (
      <div className="rounded-3xl border border-slate-200 bg-white p-10 text-center text-sm text-slate-500 shadow-sm animate-view-enter">
        ยังไม่มีรายชื่อผู้ป่วยในระบบ กรุณาลงทะเบียนผู้ป่วยใหม่ก่อนเปิดแฟ้มประเมิน 0-6 คะแนน
      </div>
    );
  }

  return (
    <div className="space-y-5 animate-view-enter">
      {toast && (
        <div
          className={`rounded-2xl border px-4 py-3 text-xs font-bold shadow-sm animate-toast-in ${
            toast.kind === 'success'
              ? 'border-emerald-200 bg-emerald-50 text-emerald-800'
              : toast.kind === 'error'
                ? 'border-rose-200 bg-rose-50 text-rose-700'
                : 'border-sky-200 bg-sky-50 text-sky-800'
          }`}
        >
          {toast.message}
        </div>
      )}

      {/* SECTION A: Individual Patient Selector & Cross-Facility Link Header */}
      <div className="rounded-3xl border border-sky-200/80 bg-white p-5 shadow-sm space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-start gap-3">
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl bg-gradient-to-br from-[#0F4C81] to-[#0D9488] text-white shadow-sm">
              <ClipboardList className="h-5 w-5" />
            </div>
            <div>
              <h2 className="text-base font-black text-slate-900">
                เวชระเบียนรายบุคคลและเกณฑ์ประเมิน 0-6 คะแนน (ประวัติใครประวัติมัน)
              </h2>
              <p className="text-xs text-slate-500">
                เปรียบเทียบพัฒนาการองศาข้อต่อ (ROM) ข้ามสาขาด้วยรหัสเชื่อมโยงข้อมูลกลาง (Universal Cross-Facility Link)
              </p>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              disabled={!selectedPatient || syncingCrossFacility}
              onClick={() => void loadPatientHistory(selectedPatient, true)}
              className="btn-tactile inline-flex items-center gap-1.5 rounded-xl border border-sky-200 bg-sky-50 px-3.5 py-2 text-xs font-bold text-[#0F4C81] hover:bg-sky-100 disabled:opacity-50 cursor-pointer"
            >
              <RefreshCw className={`h-3.5 w-3.5 ${syncingCrossFacility ? 'animate-spin' : ''}`} />
              <span>ดึงข้อมูลเชื่อมโยงข้ามสาขา</span>
            </button>
            {selectedPatient && onOpenUniversalEhrModal && (
              <button
                type="button"
                onClick={() => onOpenUniversalEhrModal(selectedPatient)}
                className="btn-tactile inline-flex items-center gap-1.5 rounded-xl border border-indigo-200 bg-indigo-50 px-3.5 py-2 text-xs font-bold text-indigo-700 hover:bg-indigo-100 cursor-pointer"
              >
                <Link2 className="h-3.5 w-3.5" />
                <span>นำเข้า/ส่งออกแพ็กเกจ EHR</span>
              </button>
            )}
          </div>
        </div>

        {/* Searchable selector */}
        <div className="grid grid-cols-1 gap-3 lg:grid-cols-12">
          <div className="lg:col-span-5 flex items-center rounded-xl border border-slate-200 bg-slate-50/70 px-3 h-10">
            <Search className="h-4 w-4 text-slate-400 mr-2 shrink-0" />
            <input
              value={patientQuery}
              onChange={(e) => setPatientQuery(e.target.value)}
              placeholder="ค้นหาผู้ป่วยจาก ชื่อ-นามสกุล, HN หรือ รหัสเชื่อมโยงกลาง..."
              className="w-full bg-transparent text-xs text-slate-800 placeholder:text-slate-400 focus:outline-none"
            />
          </div>
          <div className="lg:col-span-7">
            <select
              aria-label="เลือกผู้ป่วยรายบุคคล"
              value={selectedPatient?.id ?? ''}
              onChange={(e) => onSelectPatientId(e.target.value)}
              className="h-10 w-full rounded-xl border border-slate-200 bg-white px-3 text-xs font-bold text-slate-800 focus:border-sky-500 focus:outline-none"
            >
              {filteredPatients.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.fullName} ({p.hn}) — {resolvePatientUniversalLinkCode(p)}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Selected Patient Clinical Identity Card */}
        {selectedPatient && (
          <div className="rounded-2xl border border-emerald-200/80 bg-gradient-to-r from-[#F0FDF9] via-white to-[#EFF6FF] p-4 flex flex-wrap items-center justify-between gap-4">
            <div className="space-y-1 min-w-0">
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-base font-black text-slate-900">
                  {selectedPatient.fullName}
                </span>
                <span className="rounded-full bg-sky-100 border border-sky-200 px-2.5 py-0.5 text-[11px] font-bold text-[#0F4C81]">
                  {selectedPatient.hn}
                </span>
                <span className="rounded-full bg-slate-100 border border-slate-200 px-2.5 py-0.5 text-[11px] font-semibold text-slate-700">
                  อายุ {selectedPatient.age} ปี ({selectedPatient.gender || 'ไม่ระบุ'})
                </span>
              </div>
              <p className="text-xs text-slate-600">
                การวินิจฉัย: <span className="font-bold text-slate-800">{selectedPatient.diagnosis}</span>
                {selectedPatient.affectedSide && selectedPatient.affectedSide !== 'NONE' && (
                  <span>
                    {' '}
                    | ข้างที่ฟื้นฟู:{' '}
                    <span className="font-bold text-emerald-700">
                      {selectedPatient.affectedSide === 'LEFT'
                        ? 'ข้างซ้าย'
                        : selectedPatient.affectedSide === 'RIGHT'
                          ? 'ข้างขวา'
                          : 'ทั้งสองข้าง'}
                    </span>
                  </span>
                )}
              </p>
            </div>

            <div className="flex flex-wrap items-center gap-2">
              <div className="rounded-xl border border-emerald-300 bg-white px-3 py-2 shadow-xs flex items-center gap-2">
                <Link2 className="h-4 w-4 text-emerald-600 shrink-0" />
                <div>
                  <div className="text-[10px] font-bold text-slate-500">
                    รหัสเชื่อมโยงข้อมูลกลาง
                  </div>
                  <div className="text-xs font-black tracking-wide text-emerald-800">
                    {universalLinkCode}
                  </div>
                </div>
                <button
                  type="button"
                  onClick={() => void handleCopyLinkCode()}
                  className="ml-1 inline-flex items-center gap-1 rounded-lg bg-emerald-600 px-2.5 py-1.5 text-[11px] font-bold text-white hover:bg-emerald-500 cursor-pointer"
                >
                  {copiedCode ? <Check className="h-3 w-3" /> : <Copy className="h-3 w-3" />}
                  <span>คัดลอกรหัสส่งต่อ</span>
                </button>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Main Two-Column Clinical Workspace */}
      <div className="grid grid-cols-1 gap-5 xl:grid-cols-12">
        {/* SECTION B: Initial Symptom & 0-6 Mahidol-Adapted Assessment Form */}
        <form
          onSubmit={(e) => void handleSubmitAssessment(e)}
          className="xl:col-span-5 rounded-3xl border border-slate-200 bg-white p-5 shadow-sm space-y-4"
        >
          <div className="flex items-center justify-between gap-2 border-b border-slate-100 pb-3">
            <div className="flex items-center gap-2">
              <FileText className="h-4 w-4 text-[#0F4C81]" />
              <h3 className="text-sm font-black text-slate-900">
                บันทึกการประเมินเบื้องต้น 0-6 คะแนน
              </h3>
            </div>
            <span className="rounded-full bg-emerald-50 border border-emerald-200 px-2.5 py-0.5 text-[10px] font-bold text-emerald-700">
              Mahidol-Adapted Scale
            </span>
          </div>

          {/* Interactive 0-6 Score Picker */}
          <div className="space-y-2">
            <label className="block text-xs font-bold text-slate-700">
              คะแนนประเมินการเคลื่อนไหว (0 ถึง 6 คะแนน)
            </label>
            <div className="grid grid-cols-7 gap-2">
              {MAHIDOL_ASSESSMENT_SCALE_0_TO_6.map((item) => {
                const active = selectedScore === item.score;
                return (
                  <button
                    key={item.score}
                    type="button"
                    onClick={() => setSelectedScore(item.score)}
                    className={`btn-tactile h-11 rounded-xl border text-sm font-black transition-all cursor-pointer ${
                      active
                        ? 'bg-emerald-600 text-white border-emerald-700 shadow-md scale-[1.03]'
                        : 'bg-slate-50 text-slate-700 border-slate-200 hover:border-sky-300 hover:bg-sky-50'
                    }`}
                  >
                    {item.score}
                  </button>
                );
              })}
            </div>
            <div className="rounded-xl border border-emerald-200 bg-emerald-50/70 px-3 py-2 text-xs font-semibold text-emerald-900">
              {selectedRubric.descriptionTh}
            </div>
          </div>

          {/* Exercise + Target Side */}
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-12">
            <div className="sm:col-span-7 space-y-1">
              <label className="block text-[11px] font-bold text-slate-600">
                ท่ากายภาพบำบัด / ข้อต่อที่ประเมิน
              </label>
              <select
                value={exerciseCode}
                onChange={(e) => handleExerciseChange(e.target.value)}
                className="h-10 w-full rounded-xl border border-slate-200 bg-white px-3 text-xs font-semibold text-slate-800 focus:border-sky-500 focus:outline-none"
              >
                {CLINICAL_EXERCISES.map((ex) => (
                  <option key={ex.code} value={ex.code}>
                    {ex.name.th} (เป้า {ex.detection.targetMinDeg}°)
                  </option>
                ))}
              </select>
            </div>
            <div className="sm:col-span-5 space-y-1">
              <label className="block text-[11px] font-bold text-slate-600">ข้างที่ประเมิน</label>
              <div className="grid grid-cols-3 gap-1 h-10">
                {SIDE_OPTIONS.map((opt) => (
                  <button
                    key={opt.value}
                    type="button"
                    onClick={() => setTargetSide(opt.value)}
                    className={`rounded-xl border text-[11px] font-bold cursor-pointer transition-colors ${
                      targetSide === opt.value
                        ? 'border-[#0F4C81] bg-[#0F4C81] text-white'
                        : 'border-slate-200 bg-slate-50 text-slate-700 hover:bg-slate-100'
                    }`}
                  >
                    {opt.labelTh}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Measured ROM + Target ROM + Pull Kiosk ROM button */}
          <div className="rounded-2xl border border-sky-200 bg-sky-50/50 p-3 space-y-2.5">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <label className="text-xs font-bold text-[#0F4C81]">
                องศาข้อต่อที่วัดได้ครั้งนี้ (ROM °)
              </label>
              <button
                type="button"
                onClick={handlePullLatestKioskRom}
                className="btn-tactile inline-flex items-center gap-1 rounded-lg border border-emerald-300 bg-white px-2.5 py-1 text-[11px] font-bold text-emerald-700 hover:bg-emerald-50 cursor-pointer"
              >
                <Sparkles className="h-3 w-3" />
                <span>ดึงค่าองศาล่าสุดจากตู้ Kiosk</span>
              </button>
            </div>
            <div className="grid grid-cols-3 gap-2">
              <div>
                <span className="block text-[10px] font-semibold text-slate-500 mb-1">
                  วัดได้จริง (องศา)
                </span>
                <input
                  type="number"
                  step="0.5"
                  min={0}
                  max={220}
                  required
                  value={measuredRomInput}
                  onChange={(e) => setMeasuredRomInput(e.target.value)}
                  className="h-9 w-full rounded-xl border border-slate-200 bg-white px-2.5 text-sm font-black text-slate-900 focus:border-sky-500 focus:outline-none"
                />
              </div>
              <div>
                <span className="block text-[10px] font-semibold text-slate-500 mb-1">
                  เป้าหมายคลินิก (°)
                </span>
                <input
                  type="number"
                  step="1"
                  min={10}
                  max={220}
                  value={targetRomInput}
                  onChange={(e) => setTargetRomInput(e.target.value)}
                  className="h-9 w-full rounded-xl border border-slate-200 bg-white px-2.5 text-sm font-bold text-slate-700 focus:border-sky-500 focus:outline-none"
                />
              </div>
              <div>
                <span className="block text-[10px] font-semibold text-slate-500 mb-1">
                  ระดับปวด VAS (0-10)
                </span>
                <input
                  type="number"
                  min={0}
                  max={10}
                  value={painVasInput}
                  onChange={(e) => setPainVasInput(e.target.value)}
                  className="h-9 w-full rounded-xl border border-slate-200 bg-white px-2.5 text-sm font-bold text-slate-700 focus:border-sky-500 focus:outline-none"
                />
              </div>
            </div>
          </div>

          {/* Facility Name */}
          <div className="space-y-1.5">
            <label className="block text-[11px] font-bold text-slate-600">
              สถานที่ / สาขาที่ตรวจ (Facility Name)
            </label>
            <input
              type="text"
              value={facilityName}
              onChange={(e) => setFacilityName(e.target.value)}
              className="h-9 w-full rounded-xl border border-slate-200 bg-white px-3 text-xs font-semibold text-slate-800 focus:border-sky-500 focus:outline-none"
            />
            <div className="flex flex-wrap gap-1.5">
              {CLINICAL_FACILITY_PRESETS.map((preset) => (
                <button
                  key={preset}
                  type="button"
                  onClick={() => setFacilityName(preset)}
                  className={`rounded-lg border px-2 py-0.5 text-[10px] font-bold cursor-pointer ${
                    facilityName === preset
                      ? 'border-sky-400 bg-sky-50 text-[#0F4C81]'
                      : 'border-slate-200 bg-slate-50 text-slate-600 hover:bg-slate-100'
                  }`}
                >
                  {preset}
                </button>
              ))}
            </div>
          </div>

          {/* Clinical Handover Notes */}
          <div className="space-y-2.5">
            <div className="space-y-1">
              <label className="block text-[11px] font-bold text-slate-600">
                อาการแรกรับ (Chief Complaint)
              </label>
              <input
                type="text"
                value={chiefComplaint}
                onChange={(e) => setChiefComplaint(e.target.value)}
                placeholder="เช่น ยังตึงช่วงยกไหล่เกิน 60 องศา แต่ออกแรงได้มั่นคงขึ้น..."
                className="h-9 w-full rounded-xl border border-slate-200 bg-white px-3 text-xs text-slate-800 focus:border-sky-500 focus:outline-none"
              />
            </div>

            <div className="space-y-1">
              <label className="block text-[11px] font-bold text-slate-600">
                บันทึกส่งต่อการรักษาสำหรับบุคลากรท่านถัดไป (Handover Note)
              </label>
              <textarea
                rows={3}
                value={handoverNote}
                onChange={(e) => setHandoverNote(e.target.value)}
                placeholder="ระบุท่าที่ฝึก ระยะค้างท่า การช่วยเหลือ ข้อควรระวังสำหรับนักกายภาพรอบถัดไปหรือสาขาเครือข่าย..."
                className="w-full rounded-xl border border-slate-200 bg-white p-2.5 text-xs text-slate-800 focus:border-sky-500 focus:outline-none"
              />
            </div>

            <div className="space-y-1">
              <label className="block text-[11px] font-bold text-slate-600">
                เป้าหมายครั้งถัดไป
              </label>
              <input
                type="text"
                value={nextGoal}
                onChange={(e) => setNextGoal(e.target.value)}
                placeholder="เช่น เพิ่มมุมยกแขนเป็น 85 องศา และลดแรงช่วยประคอง"
                className="h-9 w-full rounded-xl border border-slate-200 bg-white px-3 text-xs text-slate-800 focus:border-sky-500 focus:outline-none"
              />
            </div>
          </div>

          <button
            type="submit"
            disabled={saving || !selectedPatient}
            className="btn-tactile w-full h-11 rounded-2xl bg-gradient-to-r from-[#0F4C81] via-[#0284C7] to-[#0D9488] text-xs font-black text-white shadow-md hover:brightness-105 disabled:opacity-50 cursor-pointer"
          >
            {saving ? 'กำลังบันทึกผลประเมิน...' : 'บันทึกผลประเมินเข้าแฟ้มผู้ป่วย'}
          </button>
        </form>

        {/* SECTION C: Longitudinal Progress Comparison & Dual-Metric Chart */}
        <div className="xl:col-span-7 space-y-5">
          {/* Comparison Summary Cards */}
          <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
            {/* 1. ROM Progression */}
            <div className="rounded-2xl border border-emerald-200 bg-white p-4 shadow-sm space-y-1.5">
              <div className="flex items-center justify-between text-[11px] font-bold text-slate-500">
                <span>พัฒนาการมุมองศาข้อต่อ (ROM)</span>
                <TrendingUp className="h-4 w-4 text-emerald-600" />
              </div>
              {summary.firstRom !== null && summary.latestRom !== null ? (
                <>
                  <div className="flex items-center gap-1.5 text-sm font-black text-slate-900">
                    <span>ครั้งแรก: {summary.firstRom.toFixed(1)}°</span>
                    <ArrowRight className="h-3.5 w-3.5 text-emerald-600 shrink-0" />
                    <span className="text-emerald-700">ล่าสุด: {summary.latestRom.toFixed(1)}°</span>
                  </div>
                  <div className="inline-flex items-center rounded-full bg-emerald-50 border border-emerald-200 px-2.5 py-0.5 text-[11px] font-extrabold text-emerald-700">
                    {(summary.deltaRom ?? 0) >= 0 ? `+${(summary.deltaRom ?? 0).toFixed(1)}°` : `${(summary.deltaRom ?? 0).toFixed(1)}°`}
                    {summary.percentGain !== null &&
                      ` (${summary.percentGain >= 0 ? 'เพิ่มขึ้น' : 'ลดลง'} ${Math.abs(summary.percentGain).toFixed(1)}%)`}
                  </div>
                </>
              ) : (
                <div className="text-xs text-slate-400">ยังไม่มีข้อมูล ROM</div>
              )}
            </div>

            {/* 2. Score 0-6 Progression */}
            <div className="rounded-2xl border border-sky-200 bg-white p-4 shadow-sm space-y-1.5">
              <div className="flex items-center justify-between text-[11px] font-bold text-slate-500">
                <span>พัฒนาการคะแนน (เกณฑ์ 0-6)</span>
                <Activity className="h-4 w-4 text-[#0F4C81]" />
              </div>
              {summary.firstScore !== null && summary.latestScore !== null ? (
                <>
                  <div className="flex items-center gap-1.5 text-sm font-black text-slate-900">
                    <span>แรกรับ: ระดับ {summary.firstScore}/6</span>
                    <ArrowRight className="h-3.5 w-3.5 text-sky-600 shrink-0" />
                    <span className="text-[#0F4C81]">ล่าสุด: ระดับ {summary.latestScore}/6</span>
                  </div>
                  <div className="inline-flex items-center rounded-full bg-sky-50 border border-sky-200 px-2.5 py-0.5 text-[11px] font-extrabold text-[#0F4C81]">
                    {(summary.deltaScore ?? 0) >= 0
                      ? `+${summary.deltaScore ?? 0} ระดับ`
                      : `${summary.deltaScore ?? 0} ระดับ`}
                  </div>
                </>
              ) : (
                <div className="text-xs text-slate-400">ยังไม่มีคะแนนประเมิน</div>
              )}
            </div>

            {/* 3. Visit Count & Facilities */}
            <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm space-y-1.5">
              <div className="flex items-center justify-between text-[11px] font-bold text-slate-500">
                <span>จำนวนครั้งที่ประเมินและฝึก</span>
                <Building2 className="h-4 w-4 text-teal-600" />
              </div>
              <div className="text-2xl font-black text-slate-900">
                {summary.totalVisits}{' '}
                <span className="text-xs font-bold text-slate-500">ครั้ง</span>
              </div>
              <div className="text-[11px] font-semibold text-teal-700">
                เชื่อมโยงข้าม {summary.linkedFacilities.length} สาขา/ตู้ Kiosk
              </div>
            </div>
          </div>

          {/* 60FPS Pure SVG Dual-Metric Progress Chart */}
          <div className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div>
                <h3 className="text-sm font-black text-slate-900">
                  ติดตามผลย้อนหลังและเปรียบเทียบพัฒนาการ (ROM &amp; ระดับ 0-6)
                </h3>
                <p className="text-[11px] text-slate-500">
                  เส้นกราฟสีเขียวแสดงองศาข้อต่อ (ROM °) และป้ายตัวเลขแสดงระดับคะแนน Mahidol (0-6)
                </p>
              </div>
              <select
                value={exerciseFilter}
                onChange={(e) => setExerciseFilter(e.target.value)}
                className="h-9 rounded-xl border border-slate-200 bg-slate-50 px-3 text-xs font-bold text-slate-700"
              >
                <option value="ALL">ทุกท่ากายภาพบำบัด</option>
                {exerciseOptionsInHistory.map(([code, nameTh]) => (
                  <option key={code} value={code}>
                    {nameTh}
                  </option>
                ))}
              </select>
            </div>

            {chartData.nodes.length === 0 ? (
              <div className="py-12 text-center text-xs text-slate-400">
                {loadingRecords
                  ? 'กำลังโหลดกราฟพัฒนาการ...'
                  : 'ยังไม่มีข้อมูลการประเมินสำหรับท่ากายภาพนี้'}
              </div>
            ) : (
              <div className="w-full overflow-x-auto">
                <svg
                  viewBox={`0 0 ${chartData.width} ${chartData.height}`}
                  className="w-full min-w-[500px] h-auto"
                  role="img"
                  aria-label="กราฟเปรียบเทียบพัฒนาการมุมองศาข้อต่อและคะแนน 0-6"
                >
                  <defs>
                    <linearGradient id="romAreaGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#10b981" stopOpacity="0.28" />
                      <stop offset="100%" stopColor="#10b981" stopOpacity="0.02" />
                    </linearGradient>
                  </defs>

                  {[0, 0.25, 0.5, 0.75, 1].map((frac) => {
                    const yVal = chartData.padT + chartData.plotH * (1 - frac);
                    const degLabel = Math.round(chartData.maxRom * frac);
                    return (
                      <g key={frac}>
                        <line
                          x1={chartData.padL}
                          y1={yVal}
                          x2={chartData.width - chartData.padR}
                          y2={yVal}
                          stroke="#e2e8f0"
                          strokeDasharray="3 3"
                          strokeWidth="1"
                        />
                        <text
                          x={chartData.padL - 8}
                          y={yVal + 4}
                          textAnchor="end"
                          fill="#64748b"
                          fontSize="10"
                          fontWeight="700"
                        >
                          {degLabel}°
                        </text>
                      </g>
                    );
                  })}

                  {chartData.areaPoints && (
                    <polygon points={chartData.areaPoints} fill="url(#romAreaGrad)" />
                  )}

                  {chartData.nodes.length >= 2 && (
                    <polyline
                      fill="none"
                      stroke="#059669"
                      strokeWidth="3"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      points={chartData.polylinePoints}
                    />
                  )}

                  {chartData.nodes.map((node) => (
                    <g key={node.id}>
                      <circle
                        cx={node.x}
                        cy={node.y}
                        r="6"
                        fill="#ffffff"
                        stroke="#059669"
                        strokeWidth="3"
                      />
                      <text
                        x={node.x}
                        y={Math.max(14, node.y - 12)}
                        textAnchor="middle"
                        fill="#065f46"
                        fontSize="11"
                        fontWeight="800"
                      >
                        {node.rom.toFixed(1)}°
                      </text>
                      <rect
                        x={node.x - 16}
                        y={chartData.padT + chartData.plotH + 6}
                        width="32"
                        height="16"
                        rx="8"
                        fill="#e0f2fe"
                        stroke="#bae6fd"
                        strokeWidth="1"
                      />
                      <text
                        x={node.x}
                        y={chartData.padT + chartData.plotH + 17}
                        textAnchor="middle"
                        fill="#0f4c81"
                        fontSize="9.5"
                        fontWeight="800"
                      >
                        {node.score}/6
                      </text>
                      <text
                        x={node.x}
                        y={chartData.height - 2}
                        textAnchor="middle"
                        fill="#475569"
                        fontSize="10"
                        fontWeight="700"
                      >
                        ครั้งที่ {node.visitNumber}
                      </text>
                    </g>
                  ))}
                </svg>
              </div>
            )}
          </div>

          {/* Chronological Individual History Timeline */}
          <div className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <History className="h-4 w-4 text-[#0F4C81]" />
                <h3 className="text-sm font-black text-slate-900">
                  ประวัติการรักษารายบุคคลเรียงตามลำดับเวลา ({summary.visitsWithDelta.length} ครั้ง)
                </h3>
              </div>
              <span className="text-[11px] font-semibold text-slate-500">
                แสดงเฉพาะแฟ้มของ {selectedPatient?.fullName ?? '-'}
              </span>
            </div>

            <div className="space-y-3 max-h-[540px] overflow-y-auto pr-1">
              {summary.visitsWithDelta
                .slice()
                .reverse()
                .map(({ record, deltaFromPrevRom }) => {
                  const isKiosk = record.source_channel === 'KIOSK_SESSION_AUTO_SYNC';
                  return (
                    <div
                      key={record.id}
                      className="rounded-2xl border border-slate-200 bg-slate-50/60 p-4 space-y-2 hover:border-sky-300 transition-colors card-hover-lift animate-card-enter"
                    >
                      <div className="flex flex-wrap items-center justify-between gap-2">
                        <div className="flex flex-wrap items-center gap-2">
                          <span className="rounded-full bg-[#0F4C81] px-2.5 py-0.5 text-[11px] font-extrabold text-white">
                            ครั้งที่ {record.visit_number}
                          </span>
                          <span className="text-xs font-bold text-slate-700">
                            {formatThaiDateTime(record.assessed_at)}
                          </span>
                          <span
                            className={`rounded-full border px-2 py-0.5 text-[10px] font-bold ${
                              isKiosk
                                ? 'border-teal-200 bg-teal-50 text-teal-800'
                                : 'border-sky-200 bg-sky-50 text-[#0F4C81]'
                            }`}
                          >
                            {isKiosk ? 'ซิงก์จากตู้ Kiosk' : 'ประเมินโดยบุคลากร'}
                          </span>
                        </div>

                        <div className="flex items-center gap-2">
                          <span className="rounded-full bg-emerald-600 px-2.5 py-0.5 text-[11px] font-extrabold text-white">
                            ระดับ {record.initial_score_0_to_6} / 6
                          </span>
                          <span className="rounded-full border border-emerald-300 bg-white px-2.5 py-0.5 text-[11px] font-black text-emerald-800">
                            ROM {record.measured_rom_deg.toFixed(1)}°
                            {deltaFromPrevRom !== null && (
                              <span
                                className={`ml-1 ${
                                  deltaFromPrevRom >= 0 ? 'text-emerald-600' : 'text-rose-600'
                                }`}
                              >
                                ({deltaFromPrevRom >= 0 ? `+${deltaFromPrevRom.toFixed(1)}°` : `${deltaFromPrevRom.toFixed(1)}°`})
                              </span>
                            )}
                          </span>
                        </div>
                      </div>

                      <div className="grid grid-cols-1 gap-1 text-[11px] text-slate-600 sm:grid-cols-2">
                        <div>
                          <span className="font-bold text-slate-700">สถานที่รับบริการ:</span>{' '}
                          {record.facility_name}
                        </div>
                        <div>
                          <span className="font-bold text-slate-700">ผู้บันทึกและตรวจประเมิน:</span>{' '}
                          {record.assessor_staff_name || 'บุคลากรคลินิก'}
                        </div>
                        <div className="sm:col-span-2">
                          <span className="font-bold text-slate-700">ท่ากายภาพ / ข้าง:</span>{' '}
                          {record.exercise_name_th || record.exercise_code} (
                          {record.target_side === 'LEFT'
                            ? 'ซ้าย'
                            : record.target_side === 'RIGHT'
                              ? 'ขวา'
                              : 'ทั้งสองข้าง'}
                          )
                          {record.pain_vas_0_to_10 !== null &&
                            record.pain_vas_0_to_10 !== undefined && (
                              <span className="ml-2 text-amber-700 font-semibold">
                                • Pain VAS: {record.pain_vas_0_to_10}/10
                              </span>
                            )}
                        </div>
                      </div>

                      {record.chief_complaint && (
                        <div className="rounded-xl bg-white border border-slate-200/80 px-3 py-2 text-xs text-slate-700">
                          <span className="font-bold text-slate-900">อาการแรกรับ/ครั้งนี้:</span>{' '}
                          {record.chief_complaint}
                        </div>
                      )}

                      {record.clinical_handover_note && (
                        <div className="rounded-xl bg-sky-50/70 border border-sky-200/80 px-3 py-2 text-xs text-slate-800">
                          <span className="font-bold text-[#0F4C81]">บันทึกส่งต่อการรักษา:</span>{' '}
                          {record.clinical_handover_note}
                        </div>
                      )}

                      {record.next_session_goal && (
                        <div className="text-[11px] font-semibold text-emerald-700">
                          เป้าหมายถัดไป: {record.next_session_goal}
                        </div>
                      )}
                    </div>
                  );
                })}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

