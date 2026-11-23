import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type {
  DrTechCaresApiClient,
  PatientCourseRecord,
  PatientRecord,
  RehabScheduleItem,
  RehabSessionHistoryEntry,
  RehabSessionRecord,
  UserAccount,
} from '@drtechcares/api-client';
import { CLINICAL_EXERCISES, parseExerciseCodeAndSide } from '@drtechcares/pose-engine';
import {
  Activity,
  ArrowRight,
  Building2,
  Calendar,
  Camera,
  Check,
  ClipboardList,
  Copy,
  Download,
  FileText,
  GitCompare,
  History,
  Link2,
  RefreshCw,
  Search,
  Sparkles,
  Target,
  Trash2,
  TrendingUp,
} from 'lucide-react';
import {
  CLINICAL_FACILITY_PRESETS,
  createPatientAssessment,
  deletePatientAssessment,
  deriveScore0To6FromRomRatio,
  listPatientAssessments,
  upsertCourseAutoAssessment,
} from '../lib/clinicalAssessmentStore';
import {
  MAHIDOL_ASSESSMENT_SCALE_0_TO_6,
  buildLongitudinalCsvReport,
  compareTwoAssessmentVisits,
  computeLongitudinalProgressSummary,
  computeTargetAchievementPct,
  getMahidolRubricItem,
  resolvePatientUniversalLinkCode,
  type AssessmentMeasurementMethod,
  type AssessmentTargetSide,
  type MahidolScore0To6,
  type PatientClinicalAssessmentRecord,
} from '../types/clinicalAssessment';
import {
  buildAssessmentInputFromCourseResult,
  buildPatientCourseAutoAssessments,
  type CourseAutoAssessmentResult,
} from '../utils/courseClinicalScoring';
import { ClinicalRomLiveMeasureModal } from './ClinicalRomLiveMeasureModal';

export interface PatientClinicalRecordPanelProps {
  apiClient?: DrTechCaresApiClient;
  patients: PatientRecord[];
  sessions: RehabSessionRecord[];
  currentUser: UserAccount;
  selectedPatientId: string | null;
  onSelectPatientId: (patientId: string) => void;
  onOpenUniversalEhrModal?: (patient: PatientRecord) => void;
  onAssessmentSaved?: () => Promise<void> | void;
  patientCourseOverride?: PatientCourseRecord | null;
  patientRehabSchedulesOverride?: RehabScheduleItem[];
  patientSessionHistoryOverride?: RehabSessionHistoryEntry[];
}

type DateRangeFilter = 'ALL' | '7D' | '30D' | '90D';
type SourceFilter =
  | 'ALL'
  | 'STAFF_MANUAL_ASSESSMENT'
  | 'KIOSK_SESSION_AUTO_SYNC'
  | 'COURSE_AUTO_CALCULATED';
type ChartMetricMode = 'ROM_DEG' | 'TARGET_PCT';

function getDirectSupabaseRestConfig(): { url: string; key: string } | null {
  try {
    const env = (import.meta as unknown as { env?: Record<string, string | undefined> }).env;
    const isTestRuntime =
      env?.MODE === 'test' ||
      (typeof process !== 'undefined' &&
        Boolean(process.env?.VITEST || process.env?.NODE_ENV === 'test'));
    if (isTestRuntime) return null;
    const url = (env?.VITE_SUPABASE_URL ?? '').trim().replace(/\/+$/, '');
    const key = (env?.VITE_SUPABASE_ANON_KEY ?? '').trim();
    if (!url || !key) return null;
    return { url, key };
  } catch {
    return null;
  }
}

const SIDE_OPTIONS: { value: AssessmentTargetSide; labelTh: string }[] = [
  { value: 'LEFT', labelTh: 'ซ้าย' },
  { value: 'RIGHT', labelTh: 'ขวา' },
  { value: 'BOTH', labelTh: 'ทั้งสองข้าง' },
];

function formatSideTh(side: AssessmentTargetSide): string {
  if (side === 'LEFT') return 'ซ้าย';
  if (side === 'BOTH') return 'ทั้งสองข้าง';
  return 'ขวา';
}

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

function formatShortThaiDate(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return '';
  return d.toLocaleDateString('th-TH', {
    day: 'numeric',
    month: 'short',
  });
}

function toLocalDateTimeInputValue(d: Date = new Date()): string {
  const pad = (n: number) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

function normalizeDefaultSide(patient: PatientRecord | null, exCode: string): AssessmentTargetSide {
  const matched = CLINICAL_EXERCISES.find((e) => e.code === exCode || e.id === exCode);
  if (matched?.detection.side === 'both') return 'BOTH';
  if (patient?.affectedSide === 'LEFT') return 'LEFT';
  if (patient?.affectedSide === 'BOTH') return 'BOTH';
  return 'RIGHT';
}

export function PatientClinicalRecordPanel({
  apiClient,
  patients,
  sessions,
  currentUser,
  selectedPatientId,
  onSelectPatientId,
  onOpenUniversalEhrModal,
  onAssessmentSaved,
  patientCourseOverride,
  patientRehabSchedulesOverride,
  patientSessionHistoryOverride,
}: PatientClinicalRecordPanelProps) {
  const [patientQuery, setPatientQuery] = useState('');
  const [records, setRecords] = useState<PatientClinicalAssessmentRecord[]>([]);
  const [loadingRecords, setLoadingRecords] = useState(false);
  const [syncingCrossFacility, setSyncingCrossFacility] = useState(false);
  const [copiedCode, setCopiedCode] = useState(false);

  // Assigned course & automatic 0-6 calculation state
  const [assignedCourse, setAssignedCourse] = useState<PatientCourseRecord | null>(null);
  const [fetchedSchedules, setFetchedSchedules] = useState<RehabScheduleItem[]>([]);
  const [fetchedSessions, setFetchedSessions] = useState<RehabSessionRecord[]>([]);
  const [fetchedSessionHistory, setFetchedSessionHistory] = useState<RehabSessionHistoryEntry[]>([]);
  const [loadingCourseContext, setLoadingCourseContext] = useState(false);
  const [batchSyncingCourse, setBatchSyncingCourse] = useState(false);
  const [activeAutoCalcResult, setActiveAutoCalcResult] =
    useState<CourseAutoAssessmentResult | null>(null);

  // Retrospective tracking & comparison filters
  const [exerciseFilter, setExerciseFilter] = useState<string>('ALL');
  const [sideFilter, setSideFilter] = useState<'ALL' | AssessmentTargetSide>('ALL');
  const [dateRangeFilter, setDateRangeFilter] = useState<DateRangeFilter>('ALL');
  const [sourceFilter, setSourceFilter] = useState<SourceFilter>('ALL');
  const [chartMetricMode, setChartMetricMode] = useState<ChartMetricMode>('ROM_DEG');
  const [showMahidolLine, setShowMahidolLine] = useState<boolean>(true);
  const [showPainLine, setShowPainLine] = useState<boolean>(false);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);

  // Interactive 2-visit comparison selectors
  const [compareBaselineId, setCompareBaselineId] = useState<string>('');
  const [compareFollowUpId, setCompareFollowUpId] = useState<string>('');

  // Assessment form states
  const [selectedScore, setSelectedScore] = useState<MahidolScore0To6>(3);
  const [exerciseCode, setExerciseCode] = useState<string>(
    CLINICAL_EXERCISES[0]?.code || 'STROKE_SHOULDER_FLEXION',
  );
  const [targetSide, setTargetSide] = useState<AssessmentTargetSide>('RIGHT');
  const [measuredRomInput, setMeasuredRomInput] = useState<string>('');
  const [targetRomInput, setTargetRomInput] = useState<string>(
    String(CLINICAL_EXERCISES[0]?.detection.targetMinDeg ?? 125),
  );
  const [painVasInput, setPainVasInput] = useState<string>('');
  const [assessedAtInput, setAssessedAtInput] = useState<string>(() => toLocalDateTimeInputValue());
  const [measurementMethod, setMeasurementMethod] =
    useState<AssessmentMeasurementMethod>('MANUAL_INPUT');
  const [liveMeasureModalOpen, setLiveMeasureModalOpen] = useState<boolean>(false);
  const [facilityName, setFacilityName] = useState<string>(CLINICAL_FACILITY_PRESETS[0]);
  const [chiefComplaint, setChiefComplaint] = useState<string>('');
  const [handoverNote, setHandoverNote] = useState<string>('');
  const [nextGoal, setNextGoal] = useState<string>('');
  const [saving, setSaving] = useState(false);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null);
  const [toast, setToast] = useState<{ kind: 'success' | 'error' | 'info'; message: string } | null>(
    null,
  );

  const toastTimerRef = useRef<number | null>(null);
  const copyTimerRef = useRef<number | null>(null);
  const latestLoadReqIdRef = useRef(0);
  const latestCourseReqIdRef = useRef(0);
  const formManuallyEditedRef = useRef(false);
  const lastAutoPrefilledKeyRef = useRef<string>('');
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
          seedDemoIfEmpty: false,
          excludeDemoRows: true,
          apiClient,
          sessions,
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
            list.length > 0
              ? `ซิงก์ข้อมูลจริงสำเร็จ: พบประวัติ ${list.length} ครั้ง จาก ${facilitiesCount} สถานที่/ตู้ Kiosk`
              : 'ซิงก์ข้อมูลล่าสุดแล้ว: ผู้ป่วยรายนี้ยังไม่มีบันทึกการประเมินหรือประวัติฝึก ROM',
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
    [apiClient, sessions, showToast],
  );

  const loadPatientCourseContext = useCallback(
    async (patient: PatientRecord | null) => {
      const reqId = ++latestCourseReqIdRef.current;
      if (!patient) {
        setAssignedCourse(null);
        setFetchedSchedules([]);
        setFetchedSessions([]);
        setFetchedSessionHistory([]);
        setLoadingCourseContext(false);
        return;
      }
      const targetPatientId = patient.id;
      setLoadingCourseContext(true);
      try {
        let nextCourse: PatientCourseRecord | null = patientCourseOverride ?? null;
        let nextSchedules: RehabScheduleItem[] = patientRehabSchedulesOverride
          ? [...patientRehabSchedulesOverride]
          : [];
        let nextSessions: RehabSessionRecord[] = [];
        let nextHistory: RehabSessionHistoryEntry[] = patientSessionHistoryOverride
          ? [...patientSessionHistoryOverride]
          : [];

        if (apiClient) {
          const [courseRes, sessRes, histRes] = await Promise.allSettled([
            patientCourseOverride !== undefined
              ? Promise.resolve(patientCourseOverride)
              : apiClient.getPatientCourse(targetPatientId),
            apiClient.listRehabSessions(targetPatientId),
            patientSessionHistoryOverride !== undefined
              ? Promise.resolve(patientSessionHistoryOverride)
              : apiClient.listRehabSessionHistory(targetPatientId),
          ]);
          if (courseRes.status === 'fulfilled' && courseRes.value) {
            nextCourse = courseRes.value;
          }
          if (sessRes.status === 'fulfilled' && Array.isArray(sessRes.value)) {
            nextSessions = sessRes.value.filter((s) => s.patientId === targetPatientId);
          }
          if (histRes.status === 'fulfilled' && Array.isArray(histRes.value)) {
            nextHistory = histRes.value.filter((h) => h.patientId === targetPatientId);
          }
        } else {
          const supa = getDirectSupabaseRestConfig();
          if (supa && typeof fetch === 'function') {
            const pidEnc = encodeURIComponent(targetPatientId);
            const headers = {
              apikey: supa.key,
              Authorization: `Bearer ${supa.key}`,
              'Content-Type': 'application/json',
            };
            const [cRes, scRes, seRes] = await Promise.allSettled([
              fetch(`${supa.url}/rest/v1/patient_courses?patient_id=eq.${pidEnc}&limit=1`, { headers }),
              fetch(`${supa.url}/rest/v1/rehab_schedules?patient_id=eq.${pidEnc}`, { headers }),
              fetch(`${supa.url}/rest/v1/rehab_sessions?patient_id=eq.${pidEnc}&order=created_at.desc`, { headers }),
            ]);
            if (cRes.status === 'fulfilled' && cRes.value.ok) {
              const rows = (await cRes.value.json()) as Record<string, unknown>[];
              const row = Array.isArray(rows) ? rows[0] : null;
              if (row && !nextCourse) {
                nextCourse = {
                  id: String(row.id ?? ''),
                  patientId: String(row.patient_id ?? targetPatientId),
                  title: String(row.title ?? 'คอร์สฟื้นฟูที่นักกายภาพจัดให้'),
                  weeklySchedule: Array.isArray(row.weekly_schedule)
                    ? (row.weekly_schedule as PatientCourseRecord['weeklySchedule'])
                    : [],
                  cognitiveModpackId: String(row.cognitive_modpack_id ?? ''),
                  updatedAt: String(row.updated_at ?? new Date().toISOString()),
                };
              }
            }
            if (scRes.status === 'fulfilled' && scRes.value.ok) {
              const rows = (await scRes.value.json()) as Record<string, unknown>[];
              if (Array.isArray(rows) && nextSchedules.length === 0) {
                nextSchedules = rows.map((r) => ({
                  id: String(r.id ?? ''),
                  patientId: String(r.patient_id ?? targetPatientId),
                  ptId: r.pt_id ? String(r.pt_id) : undefined,
                  exerciseId: r.exercise_id ? String(r.exercise_id) : undefined,
                  modpackId: r.modpack_id ? String(r.modpack_id) : undefined,
                  scheduledDate: String(r.scheduled_date ?? ''),
                  sets: Number(r.sets ?? 2),
                  reps: Number(r.reps ?? 8),
                  status: String(r.status ?? 'ACTIVE').toUpperCase() as RehabScheduleItem['status'],
                  notes: String(r.notes ?? ''),
                }));
              }
            }
            if (seRes.status === 'fulfilled' && seRes.value.ok) {
              const rows = (await seRes.value.json()) as Record<string, unknown>[];
              if (Array.isArray(rows)) {
                nextSessions = rows.map((r) => ({
                  id: String(r.id ?? ''),
                  patientId: String(r.patient_id ?? targetPatientId),
                  activityType: 'physio_exercise',
                  exerciseId: String(r.exercise_id ?? ''),
                  exerciseName: String(r.exercise_id ?? ''),
                  completedSets: Number(r.completed_sets ?? 0),
                  completedReps: Number(r.completed_reps ?? 0),
                  accuracyPercent: Number(r.accuracy_percent ?? 85),
                  maxRomDeg: Number(r.max_rom_deg ?? 0),
                  painStopTriggered: Boolean(r.pain_stop_triggered),
                  aiFeedback: '',
                  kinematicsReplay: [],
                  createdAt: String(r.created_at ?? new Date().toISOString()),
                }));
              }
            }
          }
        }

        if (
          reqId !== latestCourseReqIdRef.current ||
          selectedPatientIdRef.current !== targetPatientId
        ) {
          return;
        }
        setAssignedCourse(nextCourse);
        setFetchedSchedules(nextSchedules);
        setFetchedSessions(nextSessions);
        setFetchedSessionHistory(nextHistory);
      } catch {
        // Gracefully fall back to embedded patient schedules & prop sessions
      } finally {
        if (reqId === latestCourseReqIdRef.current) {
          setLoadingCourseContext(false);
        }
      }
    },
    [
      apiClient,
      patientCourseOverride,
      patientRehabSchedulesOverride,
      patientSessionHistoryOverride,
    ],
  );

  useEffect(() => {
    void loadPatientHistory(selectedPatient, false);
    void loadPatientCourseContext(selectedPatient);
  }, [selectedPatient?.id, sessions.length, loadPatientHistory, loadPatientCourseContext]);

  useEffect(() => {
    formManuallyEditedRef.current = false;
    lastAutoPrefilledKeyRef.current = '';
    setActiveAutoCalcResult(null);
    setAssignedCourse(patientCourseOverride ?? null);
    setFetchedSchedules(patientRehabSchedulesOverride ?? []);
    setFetchedSessions([]);
    setFetchedSessionHistory(patientSessionHistoryOverride ?? []);
    setRecords([]);
    setExerciseFilter('ALL');
    setSideFilter('ALL');
    setDateRangeFilter('ALL');
    setSourceFilter('ALL');
    setSelectedNodeId(null);
    setCompareBaselineId('');
    setCompareFollowUpId('');
    setConfirmDeleteId(null);
    setChiefComplaint('');
    setHandoverNote('');
    setNextGoal('');
    setMeasuredRomInput('');
    setPainVasInput('');
    setMeasurementMethod('MANUAL_INPUT');
    setAssessedAtInput(toLocalDateTimeInputValue());
    if (!selectedPatient) return;
    setTargetSide(normalizeDefaultSide(selectedPatient, exerciseCode));
    const score = selectedPatient.motorAssessmentScore;
    if (typeof score === 'number' && score >= 0 && score <= 6) {
      setSelectedScore(Math.round(score) as MahidolScore0To6);
    }
  }, [selectedPatient?.id]);

  const mergedPatientSessions = useMemo(() => {
    if (!selectedPatient) return [];
    const byId = new Map<string, RehabSessionRecord>();
    for (const s of sessions) {
      if (s.patientId === selectedPatient.id) {
        byId.set(s.id, s);
      }
    }
    for (const s of fetchedSessions) {
      if (s.patientId === selectedPatient.id && !byId.has(s.id)) {
        byId.set(s.id, s);
      }
    }
    return Array.from(byId.values());
  }, [selectedPatient, sessions, fetchedSessions]);

  const courseAutoBundle = useMemo(() => {
    if (!selectedPatient) return null;
    return buildPatientCourseAutoAssessments({
      patient: selectedPatient,
      course: patientCourseOverride ?? assignedCourse,
      rehabSchedules:
        patientRehabSchedulesOverride ?? (fetchedSchedules.length > 0 ? fetchedSchedules : null),
      sessions: mergedPatientSessions,
      sessionHistory:
        patientSessionHistoryOverride ??
        (fetchedSessionHistory.length > 0 ? fetchedSessionHistory : null),
      existingAssessments: records,
    });
  }, [
    selectedPatient,
    patientCourseOverride,
    assignedCourse,
    patientRehabSchedulesOverride,
    fetchedSchedules,
    mergedPatientSessions,
    patientSessionHistoryOverride,
    fetchedSessionHistory,
    records,
  ]);

  const applyCourseResultToForm = useCallback(
    (result: CourseAutoAssessmentResult, showFeedback = false) => {
      const matchedCatalog = CLINICAL_EXERCISES.find(
        (e) => e.code.toUpperCase() === result.exerciseCode.toUpperCase(),
      );
      if (matchedCatalog) {
        setExerciseCode(matchedCatalog.code);
      } else {
        setExerciseCode(result.exerciseCode);
      }
      setTargetSide(result.targetSide);
      setSelectedScore(result.calculatedScore0To6);
      setMeasuredRomInput(result.effectiveRomDeg.toFixed(1));
      setTargetRomInput(String(Math.round(result.targetRomDeg)));
      setMeasurementMethod('COURSE_AUTO_CALCULATED');
      setChiefComplaint(
        `ประเมินอัตโนมัติจาก${result.courseTitle} ท่า ${result.exerciseNameTh} (${result.completedReps}/${result.targetReps} ครั้ง)`,
      );
      setHandoverNote(result.breakdownSummaryTh);
      const nextTargetRom =
        result.effectiveRomDeg >= result.targetRomDeg
          ? Math.round(result.targetRomDeg)
          : Math.min(Math.round(result.targetRomDeg), Math.round(result.effectiveRomDeg + 10));
      setNextGoal(
        result.calculatedScore0To6 >= 6
          ? `คงความสมบูรณ์การเคลื่อนไหวท่า ${result.exerciseNameTh} และฝึกตามคอร์สต่อเนื่อง`
          : `เพิ่มมุมการเคลื่อนไหวท่า ${result.exerciseNameTh} สู่ ${nextTargetRom}° และยกระดับคะแนนมหิดลเป็น ${Math.min(6, result.calculatedScore0To6 + 1)}/6`,
      );
      setActiveAutoCalcResult(result);
      if (showFeedback) {
        showToast(
          'info',
          `นำผลคำนวณอัตโนมัติจากคอร์ส (${result.exerciseNameTh}: ระดับ ${result.calculatedScore0To6}/6, ROM ${result.effectiveRomDeg.toFixed(1)}°) กรอกลงแบบฟอร์มแล้ว`,
        );
      }
    },
    [showToast],
  );

  // Automatically pre-fill assessment form when the selected patient has completed assigned course sessions
  useEffect(() => {
    if (!selectedPatient || !courseAutoBundle?.latestResult) return;
    if (formManuallyEditedRef.current) return;
    const prefillKey = `${selectedPatient.id}::${courseAutoBundle.latestResult.sourceSessionId}`;
    if (lastAutoPrefilledKeyRef.current === prefillKey) return;
    lastAutoPrefilledKeyRef.current = prefillKey;
    applyCourseResultToForm(courseAutoBundle.latestResult, false);
  }, [selectedPatient, courseAutoBundle?.latestResult, applyCourseResultToForm]);

  const handleBatchSyncCourseSessions = async () => {
    if (!selectedPatient || batchSyncingCourse) return;
    if (!courseAutoBundle || courseAutoBundle.autoResults.length === 0) {
      showToast('info', 'ยังไม่พบประวัติการเล่นคอร์สกายภาพที่เสร็จสมบูรณ์สำหรับผู้ป่วยรายนี้');
      return;
    }
    const pending = courseAutoBundle.unrecordedResults;
    if (pending.length === 0) {
      if (courseAutoBundle.latestResult) {
        applyCourseResultToForm(courseAutoBundle.latestResult, false);
      }
      showToast(
        'info',
        `ผลเล่นคอร์สทั้งหมด (${courseAutoBundle.autoResults.length} รอบ) ของ ${selectedPatient.fullName} ถูกบันทึกเข้าแฟ้มประเมินครบถ้วนแล้ว`,
      );
      return;
    }

    const submitPatientId = selectedPatient.id;
    setBatchSyncingCourse(true);
    try {
      let createdCount = 0;
      for (const result of pending) {
        const input = buildAssessmentInputFromCourseResult({
          result,
          patient: selectedPatient,
          assessorStaffId: currentUser.id,
          assessorStaffName: `${currentUser.fullName} (คำนวณอัตโนมัติจากคอร์ส)`,
          facilityName: result.facilityName || facilityName,
        });
        const { created } = await upsertCourseAutoAssessment(input, apiClient);
        if (created) createdCount += 1;
      }

      if (selectedPatientIdRef.current === submitPatientId) {
        const refreshed = await listPatientAssessments(selectedPatient, {
          includeCrossFacility: true,
          seedDemoIfEmpty: false,
          excludeDemoRows: true,
          apiClient,
          sessions: mergedPatientSessions,
        });
        if (selectedPatientIdRef.current === submitPatientId) {
          setRecords(refreshed);
        }
      }

      showToast(
        'success',
        createdCount > 0
          ? `ดึงผลเล่นคอร์สและบันทึกอัตโนมัติสำเร็จ ${createdCount} รอบ (คำนวณคะแนนมหิดล 0-6 และ ROM อัตโนมัติแล้ว)`
          : 'ผลเล่นคอร์สทั้งหมดถูกบันทึกเข้าแฟ้มประเมินไว้เรียบร้อยแล้ว',
      );
      if (onAssessmentSaved) {
        void onAssessmentSaved();
      }
    } catch (err) {
      showToast(
        'error',
        err instanceof Error ? err.message : 'ไม่สามารถดึงผลเล่นคอร์สและบันทึกอัตโนมัติได้',
      );
    } finally {
      setBatchSyncingCourse(false);
    }
  };

  const handleExerciseChange = (nextCode: string) => {
    setExerciseCode(nextCode);
    const matched = CLINICAL_EXERCISES.find((e) => e.code === nextCode || e.id === nextCode);
    if (matched) {
      setTargetRomInput(String(matched.detection.targetMinDeg));
      setTargetSide(normalizeDefaultSide(selectedPatient, matched.code));
    }
  };

  const selectedExerciseMeta = useMemo(
    () =>
      CLINICAL_EXERCISES.find((e) => e.code === exerciseCode || e.id === exerciseCode) ??
      CLINICAL_EXERCISES[0],
    [exerciseCode],
  );

  const latestKioskRomInfo = useMemo(() => {
    if (!selectedPatient) return null;
    const normalizedSelectedBase = parseExerciseCodeAndSide(exerciseCode).baseCode;
    const patientKioskSessions = mergedPatientSessions
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
    const matchSync =
      autoSynced.find((r) => r.exercise_code === normalizedSelectedBase) ?? autoSynced[0];
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
  }, [selectedPatient, mergedPatientSessions, records, exerciseCode]);

  const handlePullLatestKioskRom = () => {
    if (!latestKioskRomInfo) {
      showToast('info', 'ยังไม่พบประวัติการฝึกจากตู้ Kiosk ที่มีค่าองศา ROM สำหรับผู้ป่วยรายนี้');
      return;
    }
    setMeasuredRomInput(latestKioskRomInfo.rom.toFixed(1));
    setMeasurementMethod('KIOSK_SESSION_AUTO_SYNC');
    showToast(
      'info',
      `ดึงค่าองศาล่าสุดจริง ${latestKioskRomInfo.rom.toFixed(1)}° (${latestKioskRomInfo.label}) เรียบร้อยแล้ว`,
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
    const matchedEx = selectedExerciseMeta;
    const parsedTargetRom = Number(targetRomInput);
    const safeTargetRom = Number(
      (Number.isFinite(parsedTargetRom) && parsedTargetRom > 0
        ? parsedTargetRom
        : matchedEx.detection.targetMinDeg
      ).toFixed(1),
    );

    // Allow zero-manual-measurement submission by auto-resolving effective ROM from assigned course or 0-6 level
    let measuredRom: number;
    let resolvedMethod: AssessmentMeasurementMethod = measurementMethod;
    if (measuredRomInput.trim() === '') {
      const matchingCourseSummary = courseAutoBundle?.exerciseSummaries.find(
        (item) =>
          item.definition.exerciseCode === matchedEx.code.toUpperCase() && item.latestResult,
      );
      const fallbackCourseResult =
        matchingCourseSummary?.latestResult ?? courseAutoBundle?.latestResult ?? null;
      if (fallbackCourseResult) {
        measuredRom = fallbackCourseResult.effectiveRomDeg;
        resolvedMethod = 'COURSE_AUTO_CALCULATED';
      } else {
        measuredRom = Number(Math.max(0, Math.min(220, (safeTargetRom * selectedScore) / 6)).toFixed(1));
      }
    } else {
      measuredRom = Number(measuredRomInput);
    }
    if (!Number.isFinite(measuredRom) || measuredRom < 0 || measuredRom > 220) {
      showToast('error', 'กรุณาระบุองศาข้อต่อ (ROM) ระหว่าง 0 ถึง 220 องศา');
      return;
    }

    const painParsed = painVasInput.trim() === '' ? null : Number(painVasInput);
    const safePain =
      painParsed !== null && Number.isFinite(painParsed)
        ? Math.max(0, Math.min(10, Math.round(painParsed)))
        : null;

    let assessedAtIso = new Date().toISOString();
    if (assessedAtInput.trim()) {
      const parsedDt = new Date(assessedAtInput);
      if (!Number.isNaN(parsedDt.getTime())) {
        assessedAtIso = parsedDt.toISOString();
      }
    }

    const isCourseAutoSubmission = resolvedMethod === 'COURSE_AUTO_CALCULATED';
    const submitPatientId = selectedPatient.id;
    setSaving(true);
    try {
      const saved = await createPatientAssessment(
        {
          patient_id: submitPatientId,
          patient_hn: selectedPatient.hn,
          universal_link_code: universalLinkCode,
          assessed_at: assessedAtIso,
          facility_name: facilityName,
          assessor_staff_id: currentUser.id,
          assessor_staff_name: `${currentUser.fullName} (${currentUser.role === 'super_admin' ? 'ผู้อำนวยการ/Admin' : 'นักกายภาพบำบัด'})`,
          exercise_code: matchedEx.code,
          exercise_name_th: matchedEx.name.th,
          target_side: targetSide,
          initial_score_0_to_6: selectedScore,
          measured_rom_deg: Number(measuredRom.toFixed(1)),
          target_rom_deg: safeTargetRom,
          pain_vas_0_to_10: safePain,
          chief_complaint: chiefComplaint || `ตรวจประเมินต่อเนื่องท่า ${matchedEx.name.th}`,
          clinical_handover_note:
            handoverNote ||
            (isCourseAutoSubmission && activeAutoCalcResult
              ? activeAutoCalcResult.breakdownSummaryTh
              : `ประเมินการเคลื่อนไหว ROM ${measuredRom.toFixed(1)}° (คะแนนมหิดลระดับ ${selectedScore}/6) ติดตามความสม่ำเสมอการฝึกและระวังอาการชดเชย`),
          next_session_goal:
            nextGoal || `เพิ่มมุมข้อต่อสู่ ${Math.min(180, measuredRom + 12).toFixed(0)}° ในครั้งถัดไป`,
          source_channel: isCourseAutoSubmission
            ? 'COURSE_AUTO_CALCULATED'
            : 'STAFF_MANUAL_ASSESSMENT',
          measurement_method: resolvedMethod,
          completed_reps: isCourseAutoSubmission ? (activeAutoCalcResult?.completedReps ?? null) : null,
          accuracy_percent: isCourseAutoSubmission ? (activeAutoCalcResult?.accuracyScore ?? null) : null,
          session_id: isCourseAutoSubmission ? (activeAutoCalcResult?.sourceSessionId ?? null) : null,
          source_session_id: isCourseAutoSubmission
            ? (activeAutoCalcResult?.sourceSessionId ?? null)
            : null,
        },
        apiClient,
      );

      if (selectedPatientIdRef.current === submitPatientId) {
        const refreshed = await listPatientAssessments(selectedPatient, {
          includeCrossFacility: true,
          seedDemoIfEmpty: false,
          excludeDemoRows: true,
          apiClient,
          sessions: mergedPatientSessions,
        });
        if (selectedPatientIdRef.current === submitPatientId) {
          setRecords(refreshed);
          formManuallyEditedRef.current = false;
          setActiveAutoCalcResult(null);
          setMeasuredRomInput('');
          setChiefComplaint('');
          setHandoverNote('');
          setNextGoal('');
          setMeasurementMethod('MANUAL_INPUT');
          setAssessedAtInput(toLocalDateTimeInputValue());
        }
      }
      showToast(
        'success',
        `บันทึกผลประเมินจริงครั้งที่ ${saved.visit_number} (ระดับ ${saved.initial_score_0_to_6}/6, ROM ${saved.measured_rom_deg.toFixed(1)}°) เข้าฐานข้อมูลของ ${selectedPatient.fullName} แล้ว`,
      );
      if (onAssessmentSaved) {
        void onAssessmentSaved();
      }
    } catch (err) {
      showToast(
        'error',
        err instanceof Error ? err.message : 'ไม่สามารถบันทึกผลประเมินได้',
      );
    } finally {
      setSaving(false);
    }
  };

  const handleDeleteAssessment = async (assessmentId: string) => {
    if (!selectedPatient || deletingId) return;
    setDeletingId(assessmentId);
    try {
      await deletePatientAssessment(selectedPatient.id, assessmentId, apiClient);
      setRecords((prev) => prev.filter((r) => r.id !== assessmentId));
      setConfirmDeleteId(null);
      if (selectedNodeId === assessmentId) setSelectedNodeId(null);
      showToast('success', 'ลบรายการประเมินที่เลือกเรียบร้อยแล้ว');
      if (onAssessmentSaved) {
        void onAssessmentSaved();
      }
    } catch (err) {
      showToast(
        'error',
        err instanceof Error ? err.message : 'ไม่สามารถลบรายการประเมินได้',
      );
    } finally {
      setDeletingId(null);
    }
  };

  const exerciseOptionsInHistory = useMemo(() => {
    const map = new Map<string, string>();
    for (const r of records) {
      const code = (r.exercise_code || '').toUpperCase();
      if (code && !map.has(code)) {
        map.set(code, r.exercise_name_th || code);
      }
    }
    return Array.from(map.entries());
  }, [records]);

  const filteredRecords = useMemo(() => {
    const nowMs = Date.now();
    const rangeCutoffMs =
      dateRangeFilter === '7D'
        ? nowMs - 7 * 86_400_000
        : dateRangeFilter === '30D'
          ? nowMs - 30 * 86_400_000
          : dateRangeFilter === '90D'
            ? nowMs - 90 * 86_400_000
            : 0;

    return records.filter((rec) => {
      if (exerciseFilter !== 'ALL' && rec.exercise_code.toUpperCase() !== exerciseFilter) {
        return false;
      }
      if (sideFilter !== 'ALL' && rec.target_side !== sideFilter) {
        return false;
      }
      if (sourceFilter !== 'ALL' && rec.source_channel !== sourceFilter) {
        return false;
      }
      if (rangeCutoffMs > 0) {
        const ts = new Date(rec.assessed_at).getTime();
        if (Number.isFinite(ts) && ts < rangeCutoffMs) return false;
      }
      return true;
    });
  }, [records, exerciseFilter, sideFilter, dateRangeFilter, sourceFilter]);

  const summary = useMemo(
    () => computeLongitudinalProgressSummary(filteredRecords, exerciseFilter),
    [filteredRecords, exerciseFilter],
  );

  // Full-patient per-exercise breakdown (always visible even when filtering 1 exercise, filtered by date/source)
  const overallBreakdown = useMemo(() => {
    const base = records.filter((rec) => {
      if (sourceFilter !== 'ALL' && rec.source_channel !== sourceFilter) return false;
      if (dateRangeFilter !== 'ALL') {
        const days = dateRangeFilter === '7D' ? 7 : dateRangeFilter === '30D' ? 30 : 90;
        const ts = new Date(rec.assessed_at).getTime();
        if (Number.isFinite(ts) && ts < Date.now() - days * 86_400_000) return false;
      }
      return true;
    });
    return computeLongitudinalProgressSummary(base, 'ALL').exerciseBreakdown;
  }, [records, sourceFilter, dateRangeFilter]);

  // Keep 2-visit comparison selectors synchronized with filtered real visits
  useEffect(() => {
    const visits = summary.visitsWithDelta;
    if (visits.length < 2) {
      setCompareBaselineId('');
      setCompareFollowUpId('');
      return;
    }
    const ids = new Set(visits.map((v) => v.record.id));
    setCompareBaselineId((prev) => (prev && ids.has(prev) ? prev : visits[0].record.id));
    setCompareFollowUpId((prev) =>
      prev && ids.has(prev) ? prev : visits[visits.length - 1].record.id,
    );
  }, [summary.visitsWithDelta]);

  const pairComparison = useMemo(() => {
    if (summary.visitsWithDelta.length < 2) return null;
    const baseItem =
      summary.visitsWithDelta.find((v) => v.record.id === compareBaselineId) ??
      summary.visitsWithDelta[0];
    const followItem =
      summary.visitsWithDelta.find((v) => v.record.id === compareFollowUpId) ??
      summary.visitsWithDelta[summary.visitsWithDelta.length - 1];
    if (!baseItem || !followItem) return null;
    return compareTwoAssessmentVisits(baseItem.record, followItem.record);
  }, [summary.visitsWithDelta, compareBaselineId, compareFollowUpId]);

  const effectiveChartMode: ChartMetricMode =
    summary.hasMixedExercises && exerciseFilter === 'ALL' ? chartMetricMode : chartMetricMode;

  const selectedRubric = getMahidolRubricItem(selectedScore);
  const liveFormMeasuredNum = Number(measuredRomInput);
  const liveFormTargetNum = Number(targetRomInput) || selectedExerciseMeta.detection.targetMinDeg;
  const liveFormPreviewPct =
    measuredRomInput.trim() !== '' && Number.isFinite(liveFormMeasuredNum) && liveFormMeasuredNum >= 0
      ? computeTargetAchievementPct(liveFormMeasuredNum, liveFormTargetNum)
      : null;
  const liveFormSuggestedScore =
    liveFormPreviewPct !== null
      ? deriveScore0To6FromRomRatio(liveFormMeasuredNum, liveFormTargetNum)
      : null;

  // Collision-free Pure SVG Dual-Metric Progress Chart geometry
  const chartData = useMemo(() => {
    const pts = summary.visitsWithDelta.slice(-12);
    const width = 700;
    const height = 268;
    const padL = 56;
    const padR = 44;
    const padT = 34;
    const padB = 54;
    const innerPadX = 34;
    const plotW = width - padL - padR;
    const plotH = height - padT - padB;
    const usableW = Math.max(40, plotW - innerPadX * 2);

    const isPctMode = effectiveChartMode === 'TARGET_PCT';
    const maxY = isPctMode
      ? Math.max(100, Math.ceil(Math.max(...pts.map((p) => p.targetAchievementPct), 100) / 20) * 20)
      : Math.max(
          150,
          Math.ceil(
            Math.max(
              ...pts.map((p) => Math.max(p.record.measured_rom_deg, p.record.target_rom_deg)),
              120,
            ) / 30,
          ) * 30,
        );

    const nodes = pts.map((item, idx) => {
      const x =
        pts.length === 1
          ? padL + plotW / 2
          : padL + innerPadX + (idx / Math.max(1, pts.length - 1)) * usableW;
      const primaryVal = isPctMode ? item.targetAchievementPct : item.record.measured_rom_deg;
      const ratio = Math.max(0, Math.min(1, primaryVal / maxY));
      const y = padT + plotH - ratio * plotH;
      const scoreY = padT + plotH - (item.record.initial_score_0_to_6 / 6) * plotH;
      const painVal =
        typeof item.record.pain_vas_0_to_10 === 'number' && Number.isFinite(item.record.pain_vas_0_to_10)
          ? item.record.pain_vas_0_to_10
          : null;
      const painY = painVal !== null ? padT + plotH - (painVal / 10) * plotH : null;

      return {
        id: item.record.id,
        record: item.record,
        deltaFromPrevRom: item.deltaFromPrevRom,
        deltaFromFirstRom: item.deltaFromFirstRom,
        targetAchievementPct: item.targetAchievementPct,
        x,
        y,
        scoreY,
        painVal,
        painY,
        primaryVal,
        visitNumber: item.record.visit_number,
        shortDate: formatShortThaiDate(item.record.assessed_at),
      };
    });

    const polylinePoints = nodes.map((n) => `${n.x.toFixed(1)},${n.y.toFixed(1)}`).join(' ');
    const scorePolylinePoints = nodes.map((n) => `${n.x.toFixed(1)},${n.scoreY.toFixed(1)}`).join(' ');
    const painNodes = nodes.filter((n) => n.painY !== null);
    const painPolylinePoints = painNodes
      .map((n) => `${n.x.toFixed(1)},${(n.painY as number).toFixed(1)}`)
      .join(' ');
    const areaPoints =
      nodes.length > 0
        ? `${nodes[0].x.toFixed(1)},${(padT + plotH).toFixed(1)} ${polylinePoints} ${nodes[nodes.length - 1].x.toFixed(1)},${(padT + plotH).toFixed(1)}`
        : '';

    const targetRefValue = isPctMode ? 100 : summary.targetRomDeg;
    const targetRefY =
      targetRefValue !== null && targetRefValue > 0 && targetRefValue <= maxY
        ? padT + plotH - (targetRefValue / maxY) * plotH
        : null;

    return {
      width,
      height,
      padL,
      padR,
      padT,
      padB,
      plotW,
      plotH,
      maxY,
      isPctMode,
      nodes,
      polylinePoints,
      scorePolylinePoints,
      painNodes,
      painPolylinePoints,
      areaPoints,
      targetRefValue,
      targetRefY,
    };
  }, [summary.visitsWithDelta, summary.targetRomDeg, effectiveChartMode]);

  const activeSelectedNode = useMemo(() => {
    if (chartData.nodes.length === 0) return null;
    if (selectedNodeId) {
      const found = chartData.nodes.find((n) => n.id === selectedNodeId);
      if (found) return found;
    }
    return chartData.nodes[chartData.nodes.length - 1];
  }, [chartData.nodes, selectedNodeId]);

  const handleExportCsv = () => {
    if (!selectedPatient || summary.totalVisits === 0) {
      showToast('info', 'ยังไม่มีข้อมูลการประเมินสำหรับส่งออกรายงาน CSV');
      return;
    }
    const csv = buildLongitudinalCsvReport(selectedPatient, summary);
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `drtechcares-progress-${selectedPatient.hn || selectedPatient.id}.csv`;
    a.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    showToast('success', `ส่งออกรายงานพัฒนาการ ${summary.totalVisits} ครั้ง เป็นไฟล์ CSV แล้ว`);
  };

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
                เวชระเบียนรายบุคคล การวัดผลจริง และเปรียบเทียบพัฒนาการ (0-6 &amp; ROM)
              </h2>
              <p className="text-xs text-slate-500">
                บันทึกจริงลงฐานข้อมูลกลาง ผสานผลฝึกจากตู้ Kiosk และวัดองศาสดผ่านกล้อง AI เพื่อเปรียบเทียบพัฒนาการย้อนหลังได้จริง
              </p>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              disabled={!selectedPatient || syncingCrossFacility}
              onClick={() => {
                void loadPatientHistory(selectedPatient, true);
                void loadPatientCourseContext(selectedPatient);
              }}
              className="btn-tactile inline-flex items-center gap-1.5 rounded-xl border border-sky-200 bg-sky-50 px-3.5 py-2 text-xs font-bold text-[#0F4C81] hover:bg-sky-100 disabled:opacity-50 cursor-pointer"
            >
              <RefreshCw className={`h-3.5 w-3.5 ${syncingCrossFacility ? 'animate-spin' : ''}`} />
              <span>ซิงก์ข้อมูลล่าสุด / ข้ามสาขา</span>
            </button>
            <button
              type="button"
              disabled={!selectedPatient || summary.totalVisits === 0}
              onClick={handleExportCsv}
              className="btn-tactile inline-flex items-center gap-1.5 rounded-xl border border-emerald-200 bg-emerald-50 px-3.5 py-2 text-xs font-bold text-emerald-700 hover:bg-emerald-100 disabled:opacity-45 cursor-pointer"
            >
              <Download className="h-3.5 w-3.5" />
              <span>ส่งออก CSV</span>
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
        {/* SECTION B: Assigned Rehab Courses Auto-Score Card + 0-6 Mahidol-Adapted Assessment Form */}
        <div className="xl:col-span-5 space-y-4 self-start">
          {/* Assigned Course & Automatic 0-6 Mahidol Scoring Card */}
          <div
            data-testid="assigned-course-auto-score-card"
            className="rounded-3xl border border-emerald-200/90 bg-gradient-to-br from-white via-[#F0FDF9] to-[#EFF6FF] p-4 shadow-sm space-y-3 animate-view-enter"
          >
            <div className="flex flex-wrap items-start justify-between gap-2 border-b border-emerald-100 pb-2.5">
              <div className="flex items-start gap-2.5 min-w-0">
                <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-gradient-to-br from-[#0F4C81] to-emerald-600 text-white shadow-xs">
                  <svg
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2.2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    className="h-4 w-4"
                    aria-hidden="true"
                  >
                    <path d="M9 11l3 3L22 4" />
                    <path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11" />
                  </svg>
                </div>
                <div className="min-w-0">
                  <h3 className="text-xs font-black text-slate-900 leading-snug">
                    คอร์สกายภาพที่นักกายภาพจัดให้ &amp; คำนวณคะแนนอัตโนมัติ
                  </h3>
                  <p className="text-[11px] text-slate-600 truncate">
                    {courseAutoBundle?.courseTitle
                      ? `${courseAutoBundle.courseTitle} • ${courseAutoBundle.exerciseSummaries.length} ท่าที่จัดให้`
                      : `ดึงผลฝึกจากคอร์สของ ${selectedPatient?.fullName ?? '-'} เพื่อคำนวณคะแนน 0-6 และ ROM โดยไม่ต้องวัดมุมเอง`}
                  </p>
                </div>
              </div>
              {courseAutoBundle && courseAutoBundle.unrecordedResults.length > 0 ? (
                <span className="shrink-0 rounded-full bg-emerald-600 px-2.5 py-0.5 text-[10px] font-black text-white shadow-xs">
                  รอบันทึก {courseAutoBundle.unrecordedResults.length} รอบ
                </span>
              ) : courseAutoBundle && courseAutoBundle.autoResults.length > 0 ? (
                <span className="shrink-0 rounded-full border border-emerald-200 bg-emerald-50 px-2.5 py-0.5 text-[10px] font-extrabold text-emerald-700">
                  ซิงก์ครบ {courseAutoBundle.autoResults.length} รอบแล้ว
                </span>
              ) : null}
            </div>

            {loadingCourseContext && (!courseAutoBundle || courseAutoBundle.exerciseSummaries.length === 0) ? (
              <div className="rounded-xl border border-slate-200/80 bg-white/80 px-3 py-3 text-[11px] font-semibold text-slate-500">
                กำลังดึงข้อมูลคอร์สที่นักกายภาพจัดให้และประวัติการฝึก...
              </div>
            ) : courseAutoBundle && courseAutoBundle.exerciseSummaries.length > 0 ? (
              <div className="space-y-2">
                <div className="space-y-1.5 max-h-48 overflow-y-auto pr-0.5">
                  {courseAutoBundle.exerciseSummaries.map((item) => {
                    const latest = item.latestResult;
                    const isSelectedInForm =
                      activeAutoCalcResult &&
                      latest &&
                      activeAutoCalcResult.sourceSessionId === latest.sourceSessionId;
                    return (
                      <div
                        key={item.definition.key}
                        className={`rounded-2xl border p-2.5 transition-colors ${
                          isSelectedInForm
                            ? 'border-emerald-400 bg-white shadow-xs'
                            : 'border-slate-200/80 bg-white/90 hover:border-sky-300'
                        }`}
                      >
                        <div className="flex flex-wrap items-center justify-between gap-2">
                          <div className="min-w-0">
                            <div className="flex flex-wrap items-center gap-1.5">
                              <span className="text-[11px] font-black text-slate-900">
                                {item.definition.exerciseNameTh}
                              </span>
                              <span className="rounded-md bg-sky-100 px-1.5 py-0.5 text-[10px] font-bold text-[#0F4C81]">
                                ข้าง{formatSideTh(item.definition.targetSide)}
                              </span>
                            </div>
                            <div className="mt-0.5 text-[10px] font-semibold text-slate-500">
                              เป้า {Math.round(item.definition.targetRomDeg)}° •{' '}
                              {item.definition.targetReps} ครั้ง/เซต • เล่นสำเร็จ{' '}
                              <span className="font-bold text-slate-700">
                                {item.completedSessionsCount} รอบ
                              </span>
                            </div>
                          </div>

                          <div className="flex flex-wrap items-center gap-1.5">
                            {latest ? (
                              <>
                                <span className="rounded-full bg-emerald-600 px-2 py-0.5 text-[10px] font-black text-white">
                                  ระดับ {latest.calculatedScore0To6}/6
                                </span>
                                <span className="rounded-full border border-emerald-200 bg-emerald-50 px-2 py-0.5 text-[10px] font-extrabold text-emerald-800">
                                  ROM {latest.effectiveRomDeg.toFixed(1)}°
                                </span>
                                <button
                                  type="button"
                                  onClick={() => {
                                    formManuallyEditedRef.current = false;
                                    applyCourseResultToForm(latest, true);
                                  }}
                                  className="btn-tactile rounded-lg border border-sky-200 bg-sky-50 px-2 py-0.5 text-[10px] font-extrabold text-[#0F4C81] hover:bg-sky-100 cursor-pointer"
                                >
                                  ใช้ผลนี้
                                </button>
                              </>
                            ) : (
                              <span className="rounded-full border border-slate-200 bg-slate-100 px-2 py-0.5 text-[10px] font-bold text-slate-500">
                                รอผู้ป่วยเล่นคอร์ส
                              </span>
                            )}
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>

                {courseAutoBundle.latestResult && (
                  <div
                    data-testid="course-auto-score-breakdown"
                    className="rounded-2xl border border-sky-200 bg-white/95 px-3 py-2.5 text-[11px] space-y-1 animate-toast-in"
                  >
                    <div className="flex flex-wrap items-center justify-between gap-1.5">
                      <span className="font-black text-[#0F4C81]">
                        สรุปสูตรคำนวณคะแนนมหิดล 0-6 อัตโนมัติ (เติมเข้าแบบฟอร์มแล้ว)
                      </span>
                      <span className="rounded-md bg-sky-50 border border-sky-200 px-1.5 py-0.5 text-[10px] font-extrabold text-[#0F4C81]">
                        คะแนนรวม {courseAutoBundle.latestResult.compositePct}%
                      </span>
                    </div>
                    <p className="text-[11px] font-semibold text-slate-700 leading-relaxed">
                      {courseAutoBundle.latestResult.breakdownSummaryTh}
                    </p>
                    <div className="text-[10px] text-slate-500 font-medium">
                      เกณฑ์ถ่วงน้ำหนัก: ช่วงองศา ROM 50% • จำนวนครั้งครบตามคอร์ส 30% • ความแม่นยำท่า 20%
                    </div>
                  </div>
                )}

                <div className="flex flex-wrap items-center justify-between gap-2 pt-0.5">
                  <span className="text-[10px] font-semibold text-slate-500">
                    {courseAutoBundle.unrecordedResults.length > 0
                      ? `พบผลเล่นคอร์สใหม่ ${courseAutoBundle.unrecordedResults.length} รอบที่ยังไม่บันทึกลงแฟ้มประเมิน`
                      : courseAutoBundle.autoResults.length > 0
                        ? 'บันทึกผลเล่นคอร์สครบทุกรอบแล้ว (ป้องกันข้อมูลซ้ำอัตโนมัติ)'
                        : 'เมื่อผู้ป่วยเล่นคอร์สที่จัดไว้ครบ ระบบจะคำนวณ 0-6 และ ROM ให้อัตโนมัติ'}
                  </span>
                  <button
                    type="button"
                    data-testid="batch-sync-course-card-btn"
                    disabled={
                      batchSyncingCourse ||
                      !selectedPatient ||
                      courseAutoBundle.autoResults.length === 0
                    }
                    onClick={() => void handleBatchSyncCourseSessions()}
                    className="btn-tactile inline-flex items-center gap-1.5 rounded-xl bg-emerald-600 hover:bg-emerald-700 px-3 py-1.5 text-[11px] font-extrabold text-white shadow-xs disabled:opacity-45 cursor-pointer"
                  >
                    <RefreshCw
                      className={`h-3.5 w-3.5 ${batchSyncingCourse ? 'animate-spin' : ''}`}
                    />
                    <span>
                      {batchSyncingCourse
                        ? 'กำลังบันทึกผลคอร์ส...'
                        : 'ดึงผลเล่นคอร์สและบันทึกอัตโนมัติ'}
                    </span>
                  </button>
                </div>
              </div>
            ) : (
              <div className="rounded-2xl border border-dashed border-emerald-200 bg-white/80 px-3 py-3 text-[11px] text-slate-600">
                ยังไม่มีคอร์สกายภาพหรือรอบฝึกที่เสร็จสมบูรณ์สำหรับผู้ป่วยรายนี้ เมื่อนักกายภาพจัดคอร์สและผู้ป่วยเล่นผ่านตู้ Kiosk ระบบจะดึงผลมาคำนวณคะแนน 0-6 และ ROM ให้อัตโนมัติ
              </div>
            )}
          </div>

          <form
            onSubmit={(e) => void handleSubmitAssessment(e)}
            className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm space-y-4"
          >
            <div className="flex items-center justify-between gap-2 border-b border-slate-100 pb-3">
              <div className="flex items-center gap-2">
                <FileText className="h-4 w-4 text-[#0F4C81]" />
                <h3 className="text-sm font-black text-slate-900">
                  บันทึกการประเมินจริง 0-6 คะแนน
                </h3>
              </div>
              <span className="rounded-full bg-emerald-50 border border-emerald-200 px-2.5 py-0.5 text-[10px] font-bold text-emerald-700">
                {activeAutoCalcResult ? 'คำนวณอัตโนมัติจากคอร์ส' : 'Mahidol-Adapted Scale'}
              </span>
            </div>

            {activeAutoCalcResult && (
              <div className="rounded-2xl border border-emerald-200 bg-emerald-50/75 px-3 py-2 text-[11px] font-semibold text-emerald-900 animate-toast-in">
                <div className="font-black text-emerald-800">
                  เติมค่าอัตโนมัติจากคอร์สกายภาพ (ไม่ต้องวัดองศาด้วยตนเอง)
                </div>
                <div className="mt-0.5 leading-relaxed">
                  {activeAutoCalcResult.breakdownSummaryTh}
                </div>
              </div>
            )}

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
                      onClick={() => {
                        formManuallyEditedRef.current = true;
                        setSelectedScore(item.score);
                      }}
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
                  onChange={(e) => {
                    formManuallyEditedRef.current = true;
                    handleExerciseChange(e.target.value);
                  }}
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
                      onClick={() => {
                        formManuallyEditedRef.current = true;
                        setTargetSide(opt.value);
                      }}
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

            {/* Measured ROM (Optional / Auto-Filled From Course) + One-Click Course Sync + Live Camera AI */}
            <div className="rounded-2xl border border-sky-200 bg-sky-50/50 p-3.5 space-y-3">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div>
                  <label className="text-xs font-bold text-[#0F4C81]">
                    องศาข้อต่อ (ROM °)
                  </label>
                  <span className="ml-1.5 text-[10px] font-semibold text-emerald-700">
                    (เติมอัตโนมัติจากคอร์สได้ / ไม่บังคับวัดเอง)
                  </span>
                </div>
                <div className="flex flex-wrap items-center gap-1.5">
                  <button
                    type="button"
                    data-testid="batch-sync-course-inline-btn"
                    disabled={batchSyncingCourse || !selectedPatient}
                    onClick={() => void handleBatchSyncCourseSessions()}
                    className="btn-tactile inline-flex items-center gap-1 rounded-lg bg-emerald-600 hover:bg-emerald-700 px-2.5 py-1.5 text-[11px] font-extrabold text-white shadow-xs disabled:opacity-50 cursor-pointer"
                  >
                    <RefreshCw
                      className={`h-3 w-3 ${batchSyncingCourse ? 'animate-spin' : ''}`}
                    />
                    <span>ดึงผลเล่นคอร์สและบันทึกอัตโนมัติ</span>
                  </button>
                  <button
                    type="button"
                    onClick={handlePullLatestKioskRom}
                    className="btn-tactile inline-flex items-center gap-1 rounded-lg border border-emerald-300 bg-white px-2.5 py-1.5 text-[11px] font-bold text-emerald-700 hover:bg-emerald-50 cursor-pointer"
                  >
                    <Sparkles className="h-3 w-3" />
                    <span>ดึงจากตู้ Kiosk</span>
                  </button>
                  <button
                    type="button"
                    onClick={() => setLiveMeasureModalOpen(true)}
                    className="btn-tactile inline-flex items-center gap-1 rounded-lg border border-sky-300 bg-white px-2.5 py-1.5 text-[11px] font-bold text-[#0F4C81] hover:bg-sky-50 cursor-pointer"
                  >
                    <Camera className="h-3.5 w-3.5" />
                    <span>วัดองศาจริงด้วยกล้อง AI</span>
                  </button>
                </div>
              </div>

              <div className="grid grid-cols-3 gap-2">
                <div>
                  <span className="block text-[10px] font-semibold text-slate-500 mb-1">
                    ROM (องศา / เว้นว่างได้)
                  </span>
                  <input
                    type="number"
                    step="0.5"
                    min={0}
                    max={220}
                    placeholder="คำนวณจากคอร์สอัตโนมัติ"
                    value={measuredRomInput}
                    onChange={(e) => {
                      formManuallyEditedRef.current = true;
                      setMeasuredRomInput(e.target.value);
                      setMeasurementMethod('MANUAL_INPUT');
                      setActiveAutoCalcResult(null);
                    }}
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
                    onChange={(e) => {
                      formManuallyEditedRef.current = true;
                      setTargetRomInput(e.target.value);
                    }}
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
                    placeholder="ไม่ระบุ"
                    value={painVasInput}
                    onChange={(e) => {
                      formManuallyEditedRef.current = true;
                      setPainVasInput(e.target.value);
                    }}
                    className="h-9 w-full rounded-xl border border-slate-200 bg-white px-2.5 text-sm font-bold text-slate-700 focus:border-sky-500 focus:outline-none"
                  />
                </div>
              </div>

              {liveFormPreviewPct !== null && liveFormSuggestedScore !== null && (
                <div className="flex flex-wrap items-center justify-between gap-2 rounded-xl border border-emerald-200 bg-white px-3 py-2 text-[11px]">
                  <div className="font-bold text-slate-700">
                    เทียบเป้าหมายคลินิก:{' '}
                    <span className="text-emerald-700 font-black">{liveFormPreviewPct.toFixed(1)}%</span>
                    {measurementMethod === 'COURSE_AUTO_CALCULATED' && (
                      <span className="ml-2 rounded-full bg-emerald-100 px-2 py-0.5 text-[10px] font-extrabold text-emerald-800">
                        คำนวณอัตโนมัติจากคอร์ส
                      </span>
                    )}
                    {measurementMethod === 'CAMERA_AI_LIVE' && (
                      <span className="ml-2 rounded-full bg-indigo-100 px-2 py-0.5 text-[10px] font-extrabold text-indigo-700">
                        วัดจากกล้อง AI
                      </span>
                    )}
                  </div>
                  {liveFormSuggestedScore !== selectedScore && (
                    <button
                      type="button"
                      onClick={() => {
                        formManuallyEditedRef.current = true;
                        setSelectedScore(liveFormSuggestedScore);
                      }}
                      className="rounded-lg border border-sky-200 bg-sky-50 px-2 py-0.5 text-[10px] font-extrabold text-[#0F4C81] hover:bg-sky-100 cursor-pointer"
                    >
                      ใช้คะแนนแนะนำ ({liveFormSuggestedScore}/6)
                    </button>
                  )}
                </div>
              )}
            </div>

            {/* Visit Date/Time & Facility */}
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
              <div className="space-y-1">
                <label className="flex items-center gap-1 text-[11px] font-bold text-slate-600">
                  <Calendar className="h-3.5 w-3.5 text-[#0F4C81]" />
                  <span>วันที่และเวลาที่ประเมิน</span>
                </label>
                <input
                  type="datetime-local"
                  value={assessedAtInput}
                  onChange={(e) => setAssessedAtInput(e.target.value)}
                  className="h-9 w-full rounded-xl border border-slate-200 bg-white px-2.5 text-xs font-semibold text-slate-800 focus:border-sky-500 focus:outline-none"
                />
              </div>
              <div className="space-y-1">
                <label className="block text-[11px] font-bold text-slate-600">
                  สถานที่ / สาขาที่ตรวจ
                </label>
                <input
                  type="text"
                  value={facilityName}
                  onChange={(e) => setFacilityName(e.target.value)}
                  className="h-9 w-full rounded-xl border border-slate-200 bg-white px-3 text-xs font-semibold text-slate-800 focus:border-sky-500 focus:outline-none"
                />
              </div>
            </div>
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

            {/* Clinical Handover Notes */}
            <div className="space-y-2.5">
              <div className="space-y-1">
                <label className="block text-[11px] font-bold text-slate-600">
                  อาการแรกรับ / อาการรอบนี้ (Chief Complaint)
                </label>
                <input
                  type="text"
                  value={chiefComplaint}
                  onChange={(e) => {
                    formManuallyEditedRef.current = true;
                    setChiefComplaint(e.target.value);
                  }}
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
                  onChange={(e) => {
                    formManuallyEditedRef.current = true;
                    setHandoverNote(e.target.value);
                  }}
                  placeholder="ระบุท่าที่ฝึก ระยะค้างท่า การช่วยเหลือ ข้อควรระวังสำหรับนักกายภาพรอบถัดไปหรือสาขาเครือข่าย..."
                  className="w-full rounded-xl border border-slate-200 bg-white p-2.5 text-xs text-slate-800 focus:border-sky-500 focus:outline-none"
                />
              </div>

              <div className="space-y-1">
                <label className="block text-[11px] font-bold text-slate-600">เป้าหมายครั้งถัดไป</label>
                <input
                  type="text"
                  value={nextGoal}
                  onChange={(e) => {
                    formManuallyEditedRef.current = true;
                    setNextGoal(e.target.value);
                  }}
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
              {saving ? 'กำลังบันทึกผลประเมินลงฐานข้อมูล...' : 'บันทึกผลประเมินจริงเข้าแฟ้มผู้ป่วย'}
            </button>
          </form>
        </div>

        {/* SECTION C: Genuine Retrospective Tracking & Longitudinal Progress Comparison */}
        <div className="xl:col-span-7 space-y-5">
          {/* Multi-Dimensional Filter Bar */}
          <div className="rounded-2xl border border-slate-200 bg-white p-3.5 shadow-sm flex flex-wrap items-center justify-between gap-2.5">
            <div className="flex flex-wrap items-center gap-2">
              <select
                aria-label="กรองตามท่ากายภาพ"
                value={exerciseFilter}
                onChange={(e) => setExerciseFilter(e.target.value)}
                className="h-9 rounded-xl border border-slate-200 bg-slate-50 px-2.5 text-xs font-bold text-slate-700"
              >
                <option value="ALL">ทุกท่ากายภาพบำบัด ({records.length})</option>
                {exerciseOptionsInHistory.map(([code, nameTh]) => (
                  <option key={code} value={code}>
                    {nameTh}
                  </option>
                ))}
              </select>

              <select
                aria-label="กรองตามข้าง"
                value={sideFilter}
                onChange={(e) => setSideFilter(e.target.value as 'ALL' | AssessmentTargetSide)}
                className="h-9 rounded-xl border border-slate-200 bg-slate-50 px-2.5 text-xs font-bold text-slate-700"
              >
                <option value="ALL">ทุกข้าง</option>
                <option value="LEFT">ข้างซ้าย</option>
                <option value="RIGHT">ข้างขวา</option>
                <option value="BOTH">ทั้งสองข้าง</option>
              </select>

              <select
                aria-label="กรองตามช่วงเวลา"
                value={dateRangeFilter}
                onChange={(e) => setDateRangeFilter(e.target.value as DateRangeFilter)}
                className="h-9 rounded-xl border border-slate-200 bg-slate-50 px-2.5 text-xs font-bold text-slate-700"
              >
                <option value="ALL">ช่วงเวลาทั้งหมด</option>
                <option value="7D">7 วันล่าสุด</option>
                <option value="30D">30 วันล่าสุด</option>
                <option value="90D">90 วันล่าสุด</option>
              </select>

              <select
                aria-label="กรองตามแหล่งข้อมูล"
                value={sourceFilter}
                onChange={(e) => setSourceFilter(e.target.value as SourceFilter)}
                className="h-9 rounded-xl border border-slate-200 bg-slate-50 px-2.5 text-xs font-bold text-slate-700"
              >
                <option value="ALL">ทุกแหล่งข้อมูล</option>
                <option value="COURSE_AUTO_CALCULATED">คำนวณอัตโนมัติจากคอร์ส</option>
                <option value="STAFF_MANUAL_ASSESSMENT">ประเมินโดยบุคลากร</option>
                <option value="KIOSK_SESSION_AUTO_SYNC">ซิงก์จากตู้ Kiosk</option>
              </select>
            </div>

            {(exerciseFilter !== 'ALL' ||
              sideFilter !== 'ALL' ||
              dateRangeFilter !== 'ALL' ||
              sourceFilter !== 'ALL') && (
              <button
                type="button"
                onClick={() => {
                  setExerciseFilter('ALL');
                  setSideFilter('ALL');
                  setDateRangeFilter('ALL');
                  setSourceFilter('ALL');
                }}
                className="rounded-lg border border-slate-200 bg-white px-2.5 py-1 text-[11px] font-bold text-slate-600 hover:bg-slate-50 cursor-pointer"
              >
                ล้างตัวกรอง
              </button>
            )}
          </div>

          {/* 4 Real Clinical KPI Cards */}
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            {/* 1. ROM Progression */}
            <div className="rounded-2xl border border-emerald-200 bg-white p-4 shadow-sm space-y-1.5">
              <div className="flex items-center justify-between text-[11px] font-bold text-slate-500">
                <span>
                  {summary.hasMixedExercises
                    ? 'พัฒนาการเทียบเป้าหมายรวมทุกท่า (%)'
                    : 'พัฒนาการมุมองศาข้อต่อจริง (ROM)'}
                </span>
                <TrendingUp className="h-4 w-4 text-emerald-600" />
              </div>
              {summary.firstRom !== null && summary.latestRom !== null ? (
                summary.hasMixedExercises ? (
                  <>
                    <div className="flex items-center gap-1.5 text-sm font-black text-slate-900">
                      <span>ครั้งแรก: {(summary.firstTargetAchievementPct ?? 0).toFixed(1)}%</span>
                      <ArrowRight className="h-3.5 w-3.5 text-emerald-600 shrink-0" />
                      <span className="text-emerald-700">
                        ล่าสุด: {(summary.latestTargetAchievementPct ?? 0).toFixed(1)}%
                      </span>
                    </div>
                    <div className="flex flex-wrap items-center gap-1.5">
                      <span
                        className={`inline-flex items-center rounded-full border px-2.5 py-0.5 text-[11px] font-extrabold ${
                          (summary.deltaTargetAchievementPct ?? 0) >= 0
                            ? 'bg-emerald-50 border-emerald-200 text-emerald-700'
                            : 'bg-rose-50 border-rose-200 text-rose-700'
                        }`}
                      >
                        {(summary.deltaTargetAchievementPct ?? 0) >= 0
                          ? `+${(summary.deltaTargetAchievementPct ?? 0).toFixed(1)}% เทียบเป้าหมาย`
                          : `${(summary.deltaTargetAchievementPct ?? 0).toFixed(1)}% เทียบเป้าหมาย`}
                      </span>
                      <span className="text-[10px] font-semibold text-slate-500">
                        เลือก 1 ท่าเพื่อดูผลต่างองศาตรง
                      </span>
                    </div>
                  </>
                ) : (
                  <>
                    <div className="flex items-center gap-1.5 text-sm font-black text-slate-900">
                      <span>ครั้งแรก: {summary.firstRom.toFixed(1)}°</span>
                      <ArrowRight className="h-3.5 w-3.5 text-emerald-600 shrink-0" />
                      <span className="text-emerald-700">ล่าสุด: {summary.latestRom.toFixed(1)}°</span>
                    </div>
                    <div className="flex flex-wrap items-center gap-1.5">
                      <span
                        className={`inline-flex items-center rounded-full border px-2.5 py-0.5 text-[11px] font-extrabold ${
                          (summary.deltaRom ?? 0) >= 0
                            ? 'bg-emerald-50 border-emerald-200 text-emerald-700'
                            : 'bg-rose-50 border-rose-200 text-rose-700'
                        }`}
                      >
                        {(summary.deltaRom ?? 0) >= 0
                          ? `+${(summary.deltaRom ?? 0).toFixed(1)}°`
                          : `${(summary.deltaRom ?? 0).toFixed(1)}°`}
                        {summary.percentGain !== null &&
                          ` (${summary.percentGain >= 0 ? 'เพิ่มขึ้น' : 'ลดลง'} ${Math.abs(summary.percentGain).toFixed(1)}%)`}
                      </span>
                      {summary.romGainPerWeek !== null && (
                        <span className="text-[10px] font-bold text-slate-500">
                          เฉลี่ย {summary.romGainPerWeek >= 0 ? '+' : ''}
                          {summary.romGainPerWeek.toFixed(1)}°/สัปดาห์
                        </span>
                      )}
                    </div>
                  </>
                )
              ) : (
                <div className="text-xs text-slate-400">ยังไม่มีข้อมูลการวัด ROM จริง</div>
              )}
            </div>

            {/* 2. Clinical Target ROM Achievement */}
            <div className="rounded-2xl border border-teal-200 bg-white p-4 shadow-sm space-y-1.5">
              <div className="flex items-center justify-between text-[11px] font-bold text-slate-500">
                <span>ความสำเร็จเทียบเป้าหมายคลินิก</span>
                <Target className="h-4 w-4 text-teal-600" />
              </div>
              {summary.latestTargetAchievementPct !== null ? (
                <>
                  <div className="flex items-baseline justify-between gap-2">
                    <span className="text-xl font-black text-teal-700">
                      {summary.latestTargetAchievementPct.toFixed(1)}%
                    </span>
                    {summary.targetRomDeg !== null && summary.latestRom !== null && (
                      <span className="text-[11px] font-bold text-slate-600">
                        {summary.latestRom.toFixed(1)}° / เป้า {summary.targetRomDeg.toFixed(1)}°
                      </span>
                    )}
                  </div>
                  <div className="h-2 w-full overflow-hidden rounded-full bg-slate-100">
                    <div
                      className="h-full rounded-full bg-gradient-to-r from-[#0284C7] to-[#10B981]"
                      style={{ width: `${Math.min(100, summary.latestTargetAchievementPct)}%` }}
                    />
                  </div>
                  <div className="text-[10px] font-semibold text-slate-500">
                    {summary.remainingToTargetDeg !== null
                      ? summary.remainingToTargetDeg <= 0
                        ? 'บรรลุเป้าหมายองศาคลินิกแล้ว'
                        : `เหลืออีก ${summary.remainingToTargetDeg.toFixed(1)}° จะถึงเป้าหมายคลินิก`
                      : `รวม ${summary.exerciseBreakdown.length} ท่ากายภาพที่วัดผลจริง`}
                  </div>
                </>
              ) : (
                <div className="text-xs text-slate-400">บันทึกการประเมินครั้งแรกเพื่อเริ่มเทียบเป้าหมาย</div>
              )}
            </div>

            {/* 3. Mahidol 0-6 & Pain VAS */}
            <div className="rounded-2xl border border-sky-200 bg-white p-4 shadow-sm space-y-1.5">
              <div className="flex items-center justify-between text-[11px] font-bold text-slate-500">
                <span>คะแนนมหิดล (0-6) &amp; ระดับปวด VAS</span>
                <Activity className="h-4 w-4 text-[#0F4C81]" />
              </div>
              {summary.firstScore !== null && summary.latestScore !== null ? (
                <>
                  <div className="flex items-center gap-1.5 text-sm font-black text-slate-900">
                    <span>แรกรับ: {summary.firstScore}/6</span>
                    <ArrowRight className="h-3.5 w-3.5 text-sky-600 shrink-0" />
                    <span className="text-[#0F4C81]">ล่าสุด: {summary.latestScore}/6</span>
                  </div>
                  <div className="flex flex-wrap items-center gap-1.5">
                    <span
                      className={`inline-flex items-center rounded-full border px-2.5 py-0.5 text-[11px] font-extrabold ${
                        (summary.deltaScore ?? 0) >= 0
                          ? 'bg-sky-50 border-sky-200 text-[#0F4C81]'
                          : 'bg-rose-50 border-rose-200 text-rose-700'
                      }`}
                    >
                      {(summary.deltaScore ?? 0) >= 0
                        ? `+${summary.deltaScore ?? 0} ระดับ`
                        : `${summary.deltaScore ?? 0} ระดับ`}
                    </span>
                    {summary.firstPainVas !== null && summary.latestPainVas !== null && (
                      <span
                        className={`inline-flex items-center rounded-full border px-2 py-0.5 text-[10px] font-extrabold ${
                          (summary.deltaPainVas ?? 0) <= 0
                            ? 'bg-emerald-50 border-emerald-200 text-emerald-700'
                            : 'bg-rose-50 border-rose-200 text-rose-700'
                        }`}
                      >
                        ปวด VAS: {summary.firstPainVas} → {summary.latestPainVas}/10
                      </span>
                    )}
                  </div>
                </>
              ) : (
                <div className="text-xs text-slate-400">ยังไม่มีคะแนนประเมิน 0-6</div>
              )}
            </div>

            {/* 4. Visit Count, Channels & Duration */}
            <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm space-y-1.5">
              <div className="flex items-center justify-between text-[11px] font-bold text-slate-500">
                <span>จำนวนครั้งจริง &amp; ช่วงเวลาติดตาม</span>
                <Building2 className="h-4 w-4 text-teal-600" />
              </div>
              <div className="flex items-baseline justify-between gap-2">
                <div className="text-xl font-black text-slate-900">
                  {summary.totalVisits}{' '}
                  <span className="text-xs font-bold text-slate-500">ครั้งจริง</span>
                </div>
                <span className="text-[11px] font-bold text-slate-600">
                  ช่วงติดตาม {summary.elapsedDays} วัน
                </span>
              </div>
              <div className="flex flex-wrap items-center gap-2 text-[11px] font-semibold">
                <span className="text-emerald-700">คอร์สอัตโนมัติ {summary.courseAutoVisitCount}</span>
                <span className="text-slate-300">|</span>
                <span className="text-[#0F4C81]">บุคลากร {summary.staffVisitCount}</span>
                <span className="text-slate-300">|</span>
                <span className="text-teal-700">ตู้ Kiosk {summary.kioskVisitCount}</span>
                <span className="text-slate-300">|</span>
                <span className="text-slate-600">{summary.linkedFacilities.length} สาขา</span>
              </div>
            </div>
          </div>

          {/* 60FPS Collision-Free Interactive Dual-Metric Progress Chart */}
          <div className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm space-y-3">
            <div className="flex flex-wrap items-start justify-between gap-2">
              <div>
                <h3 className="text-sm font-black text-slate-900">
                  กราฟติดตามผลย้อนหลังและเปรียบเทียบพัฒนาการ (ข้อมูลจริง)
                </h3>
                <p className="text-[11px] text-slate-500">
                  คลิกที่จุดข้อมูลบนกราฟเพื่อตรวจสอบรายละเอียดแต่ละรอบ •{' '}
                  {chartData.isPctMode
                    ? 'แสดงสัดส่วนเทียบเป้าหมายคลินิก (% Target ROM) เพื่อเทียบข้ามท่าอย่างยุติธรรม'
                    : 'แสดงองศาข้อต่อจริง (ROM °) พร้อมเส้นเป้าหมายคลินิก'}
                </p>
              </div>

              <div className="flex flex-wrap items-center gap-1.5">
                <div className="inline-flex rounded-xl border border-slate-200 bg-slate-50 p-0.5 text-[11px] font-bold">
                  <button
                    type="button"
                    onClick={() => setChartMetricMode('ROM_DEG')}
                    className={`rounded-lg px-2.5 py-1 cursor-pointer transition-colors ${
                      chartMetricMode === 'ROM_DEG'
                        ? 'bg-[#0F4C81] text-white shadow-xs'
                        : 'text-slate-600 hover:text-slate-900'
                    }`}
                  >
                    องศาจริง (°)
                  </button>
                  <button
                    type="button"
                    onClick={() => setChartMetricMode('TARGET_PCT')}
                    className={`rounded-lg px-2.5 py-1 cursor-pointer transition-colors ${
                      chartMetricMode === 'TARGET_PCT'
                        ? 'bg-emerald-600 text-white shadow-xs'
                        : 'text-slate-600 hover:text-slate-900'
                    }`}
                  >
                    % เป้าหมาย
                  </button>
                </div>

                <button
                  type="button"
                  onClick={() => setShowMahidolLine((v) => !v)}
                  className={`rounded-lg border px-2.5 py-1 text-[10px] font-extrabold cursor-pointer ${
                    showMahidolLine
                      ? 'border-sky-300 bg-sky-50 text-[#0F4C81]'
                      : 'border-slate-200 bg-white text-slate-500'
                  }`}
                >
                  เส้นมหิดล 0-6
                </button>
                <button
                  type="button"
                  onClick={() => setShowPainLine((v) => !v)}
                  className={`rounded-lg border px-2.5 py-1 text-[10px] font-extrabold cursor-pointer ${
                    showPainLine
                      ? 'border-amber-300 bg-amber-50 text-amber-800'
                      : 'border-slate-200 bg-white text-slate-500'
                  }`}
                >
                  เส้นปวด VAS
                </button>
              </div>
            </div>

            {summary.hasMixedExercises && chartMetricMode === 'ROM_DEG' && (
              <div className="flex flex-wrap items-center justify-between gap-2 rounded-xl border border-amber-200 bg-amber-50 px-3 py-2 text-[11px] font-semibold text-amber-900">
                <span>
                  กราฟนี้รวมหลายท่ากายภาพที่มีองศาเป้าหมายต่างกัน แนะนำสลับเป็นโหมด &quot;% เป้าหมาย&quot; หรือเลือกดูทีละท่า
                </span>
                <button
                  type="button"
                  onClick={() => setChartMetricMode('TARGET_PCT')}
                  className="rounded-lg bg-amber-600 px-2.5 py-1 text-[10px] font-extrabold text-white cursor-pointer"
                >
                  สลับเป็น % เป้าหมาย
                </button>
              </div>
            )}

            {chartData.nodes.length === 0 ? (
              <div className="rounded-2xl border border-dashed border-slate-200 bg-slate-50/70 py-10 px-4 text-center space-y-2">
                <div className="text-xs font-extrabold text-slate-700">
                  {loadingRecords
                    ? 'กำลังโหลดประวัติจริงจากฐานข้อมูล...'
                    : 'ยังไม่มีข้อมูลการประเมินหรือประวัติฝึก ROM จริงสำหรับผู้ป่วยรายนี้'}
                </div>
                {!loadingRecords && (
                  <p className="text-[11px] text-slate-500 max-w-md mx-auto">
                    เริ่มต้นติดตามผลได้ทันทีโดยกดปุ่ม &quot;ดึงผลเล่นคอร์สและบันทึกอัตโนมัติ&quot; ด้านซ้าย หรือกดบันทึกผลประเมิน 0-6 เข้าแฟ้มผู้ป่วย
                  </p>
                )}
              </div>
            ) : (
              <div className="space-y-3">
                <div className="w-full overflow-x-auto">
                  <svg
                    viewBox={`0 0 ${chartData.width} ${chartData.height}`}
                    className="w-full min-w-[540px] h-auto"
                    role="img"
                    aria-label="กราฟเปรียบเทียบพัฒนาการมุมองศาข้อต่อและคะแนน 0-6 จากข้อมูลจริง"
                  >
                    <defs>
                      <linearGradient id="romAreaGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#10b981" stopOpacity="0.26" />
                        <stop offset="100%" stopColor="#10b981" stopOpacity="0.02" />
                      </linearGradient>
                    </defs>

                    {/* Horizontal Grid + Left Axis (ROM or % Target) + Right Axis (Mahidol 0-6) */}
                    {[0, 0.25, 0.5, 0.75, 1].map((frac) => {
                      const yVal = chartData.padT + chartData.plotH * (1 - frac);
                      const leftLabel = Math.round(chartData.maxY * frac);
                      const rightScoreLabel = Math.round(6 * frac);
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
                            x={chartData.padL - 10}
                            y={yVal + 3.5}
                            textAnchor="end"
                            fill="#64748b"
                            fontSize="10"
                            fontWeight="700"
                          >
                            {chartData.isPctMode ? `${leftLabel}%` : `${leftLabel}°`}
                          </text>
                          {showMahidolLine && (
                            <text
                              x={chartData.width - chartData.padR + 8}
                              y={yVal + 3.5}
                              textAnchor="start"
                              fill="#0284c7"
                              fontSize="9.5"
                              fontWeight="700"
                            >
                              {rightScoreLabel}/6
                            </text>
                          )}
                        </g>
                      );
                    })}

                    {/* Target ROM / 100% Clinical Reference Line */}
                    {chartData.targetRefY !== null && chartData.targetRefValue !== null && (
                      <g>
                        <line
                          x1={chartData.padL}
                          y1={chartData.targetRefY}
                          x2={chartData.width - chartData.padR}
                          y2={chartData.targetRefY}
                          stroke="#0284c7"
                          strokeDasharray="6 4"
                          strokeWidth="1.6"
                        />
                        <text
                          x={chartData.width - chartData.padR - 4}
                          y={Math.max(14, chartData.targetRefY - 6)}
                          textAnchor="end"
                          fill="#0369a1"
                          fontSize="9.5"
                          fontWeight="800"
                        >
                          {chartData.isPctMode
                            ? 'เป้าหมายคลินิก 100%'
                            : `เป้าหมายคลินิก ${chartData.targetRefValue.toFixed(0)}°`}
                        </text>
                      </g>
                    )}

                    {/* Primary ROM / % Target Area & Line */}
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

                    {/* Secondary Mahidol 0-6 Trend Line */}
                    {showMahidolLine && chartData.nodes.length >= 2 && (
                      <polyline
                        fill="none"
                        stroke="#0284c7"
                        strokeWidth="2"
                        strokeDasharray="5 4"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        points={chartData.scorePolylinePoints}
                      />
                    )}

                    {/* Optional Pain VAS Trend Line */}
                    {showPainLine && chartData.painNodes.length >= 2 && (
                      <polyline
                        fill="none"
                        stroke="#d97706"
                        strokeWidth="2"
                        strokeDasharray="3 3"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        points={chartData.painPolylinePoints}
                      />
                    )}

                    {/* Interactive Nodes */}
                    {chartData.nodes.map((node) => {
                      const isSelected = activeSelectedNode?.id === node.id;
                      const labelAboveY = node.y < chartData.padT + 20 ? node.y + 20 : node.y - 12;
                      return (
                        <g
                          key={node.id}
                          onClick={() => setSelectedNodeId(node.id)}
                          className="cursor-pointer"
                        >
                          {isSelected && (
                            <line
                              x1={node.x}
                              y1={chartData.padT}
                              x2={node.x}
                              y2={chartData.padT + chartData.plotH}
                              stroke="#10b981"
                              strokeDasharray="2 3"
                              strokeWidth="1.2"
                              opacity="0.6"
                            />
                          )}
                          {showMahidolLine && (
                            <circle
                              cx={node.x}
                              cy={node.scoreY}
                              r="3.8"
                              fill="#0284c7"
                              stroke="#ffffff"
                              strokeWidth="1.5"
                            />
                          )}
                          {showPainLine && node.painY !== null && (
                            <circle
                              cx={node.x}
                              cy={node.painY}
                              r="3.5"
                              fill="#d97706"
                              stroke="#ffffff"
                              strokeWidth="1.5"
                            />
                          )}
                          <circle cx={node.x} cy={node.y} r="15" fill="transparent" />
                          <circle
                            cx={node.x}
                            cy={node.y}
                            r={isSelected ? '7.5' : '5.8'}
                            fill={isSelected ? '#059669' : '#ffffff'}
                            stroke="#059669"
                            strokeWidth="3"
                          />
                          <text
                            x={node.x}
                            y={labelAboveY}
                            textAnchor="middle"
                            fill="#065f46"
                            fontSize="10.5"
                            fontWeight="800"
                          >
                            {chartData.isPctMode
                              ? `${node.targetAchievementPct.toFixed(0)}%`
                              : `${node.record.measured_rom_deg.toFixed(1)}°`}
                          </text>
                          <rect
                            x={node.x - 16}
                            y={chartData.padT + chartData.plotH + 6}
                            width="32"
                            height="16"
                            rx="8"
                            fill={isSelected ? '#0F4C81' : '#e0f2fe'}
                            stroke={isSelected ? '#0F4C81' : '#bae6fd'}
                            strokeWidth="1"
                          />
                          <text
                            x={node.x}
                            y={chartData.padT + chartData.plotH + 17}
                            textAnchor="middle"
                            fill={isSelected ? '#ffffff' : '#0f4c81'}
                            fontSize="9.5"
                            fontWeight="800"
                          >
                            {node.record.initial_score_0_to_6}/6
                          </text>
                          <text
                            x={node.x}
                            y={chartData.padT + chartData.plotH + 34}
                            textAnchor="middle"
                            fill="#334155"
                            fontSize="9.5"
                            fontWeight="700"
                          >
                            ครั้งที่ {node.visitNumber}
                          </text>
                          <text
                            x={node.x}
                            y={chartData.padT + chartData.plotH + 46}
                            textAnchor="middle"
                            fill="#64748b"
                            fontSize="8.5"
                            fontWeight="600"
                          >
                            {node.shortDate}
                          </text>
                        </g>
                      );
                    })}
                  </svg>
                </div>

                {/* Selected Chart Visit Inspector */}
                {activeSelectedNode && (
                  <div className="rounded-2xl border border-emerald-200 bg-emerald-50/45 px-4 py-3 text-xs flex flex-wrap items-center justify-between gap-2">
                    <div className="space-y-0.5 min-w-0">
                      <div className="font-black text-slate-900">
                        รายละเอียดครั้งที่ {activeSelectedNode.visitNumber} •{' '}
                        {formatThaiDateTime(activeSelectedNode.record.assessed_at)}
                      </div>
                      <div className="text-[11px] text-slate-600">
                        {activeSelectedNode.record.exercise_name_th} (ข้าง
                        {formatSideTh(activeSelectedNode.record.target_side)}) •{' '}
                        {activeSelectedNode.record.facility_name}
                      </div>
                    </div>
                    <div className="flex flex-wrap items-center gap-1.5">
                      <span className="rounded-full bg-emerald-600 px-2.5 py-0.5 text-[11px] font-black text-white">
                        ROM {activeSelectedNode.record.measured_rom_deg.toFixed(1)}° ({activeSelectedNode.targetAchievementPct.toFixed(1)}%)
                      </span>
                      <span className="rounded-full bg-[#0F4C81] px-2.5 py-0.5 text-[11px] font-extrabold text-white">
                        มหิดล {activeSelectedNode.record.initial_score_0_to_6}/6
                      </span>
                      {activeSelectedNode.deltaFromPrevRom !== null && (
                        <span
                          className={`rounded-full border px-2 py-0.5 text-[11px] font-extrabold ${
                            activeSelectedNode.deltaFromPrevRom >= 0
                              ? 'border-emerald-300 bg-white text-emerald-700'
                              : 'border-rose-300 bg-white text-rose-700'
                          }`}
                        >
                          เทียบครั้งก่อน {activeSelectedNode.deltaFromPrevRom >= 0 ? '+' : ''}
                          {activeSelectedNode.deltaFromPrevRom.toFixed(1)}°
                        </span>
                      )}
                      {activeSelectedNode.painVal !== null && (
                        <span className="rounded-full border border-amber-300 bg-amber-50 px-2 py-0.5 text-[11px] font-bold text-amber-800">
                          ปวด VAS {activeSelectedNode.painVal}/10
                        </span>
                      )}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Interactive 2-Visit Side-by-Side Deep Comparator */}
          {pairComparison && (
            <div className="rounded-3xl border border-indigo-200/80 bg-white p-5 shadow-sm space-y-3.5">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  <GitCompare className="h-4 w-4 text-indigo-600" />
                  <h3 className="text-sm font-black text-slate-900">
                    เปรียบเทียบพัฒนาการแบบเจาะลึก 2 รอบ (Visit-to-Visit Comparator)
                  </h3>
                </div>
                <span className="rounded-full bg-indigo-50 border border-indigo-200 px-2.5 py-0.5 text-[10px] font-extrabold text-indigo-700">
                  ห่างกัน {pairComparison.elapsedDays} วัน
                  {pairComparison.romGainPerWeek !== null && pairComparison.isSameExercise
                    ? ` • เฉลี่ย ${pairComparison.romGainPerWeek >= 0 ? '+' : ''}${pairComparison.romGainPerWeek.toFixed(1)}°/สัปดาห์`
                    : ''}
                </span>
              </div>

              <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-2">
                <div className="space-y-1">
                  <label className="block text-[10px] font-extrabold text-slate-500 uppercase">
                    รอบอ้างอิง (Baseline A)
                  </label>
                  <select
                    aria-label="เลือกรอบอ้างอิง Baseline A"
                    value={pairComparison.baseline.id}
                    onChange={(e) => setCompareBaselineId(e.target.value)}
                    className="h-9 w-full rounded-xl border border-slate-200 bg-slate-50 px-2.5 text-xs font-bold text-slate-800"
                  >
                    {summary.visitsWithDelta.map(({ record }) => (
                      <option key={record.id} value={record.id}>
                        ครั้งที่ {record.visit_number} ({formatShortThaiDate(record.assessed_at)}) —{' '}
                        {record.measured_rom_deg.toFixed(1)}° | {record.initial_score_0_to_6}/6
                      </option>
                    ))}
                  </select>
                </div>

                <div className="space-y-1">
                  <label className="block text-[10px] font-extrabold text-slate-500 uppercase">
                    รอบติดตามผล (Follow-up B)
                  </label>
                  <select
                    aria-label="เลือกรอบติดตามผล Follow-up B"
                    value={pairComparison.followUp.id}
                    onChange={(e) => setCompareFollowUpId(e.target.value)}
                    className="h-9 w-full rounded-xl border border-indigo-200 bg-indigo-50/40 px-2.5 text-xs font-bold text-slate-800"
                  >
                    {summary.visitsWithDelta.map(({ record }) => (
                      <option key={record.id} value={record.id}>
                        ครั้งที่ {record.visit_number} ({formatShortThaiDate(record.assessed_at)}) —{' '}
                        {record.measured_rom_deg.toFixed(1)}° | {record.initial_score_0_to_6}/6
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              {/* Objective Clinical Outcome Verdict */}
              <div
                className={`rounded-2xl border px-3.5 py-2.5 text-xs font-extrabold ${
                  pairComparison.verdictTone === 'emerald'
                    ? 'border-emerald-200 bg-emerald-50 text-emerald-800'
                    : pairComparison.verdictTone === 'sky'
                      ? 'border-sky-200 bg-sky-50 text-[#0F4C81]'
                      : pairComparison.verdictTone === 'rose'
                        ? 'border-rose-200 bg-rose-50 text-rose-700'
                        : 'border-amber-200 bg-amber-50 text-amber-800'
                }`}
              >
                สรุปผลเปรียบเทียบทางคลินิก: {pairComparison.clinicalVerdictTh}
              </div>

              <div className="grid grid-cols-2 gap-2.5 sm:grid-cols-4">
                <div className="rounded-xl border border-slate-200 bg-slate-50/70 p-3">
                  <div className="text-[10px] font-bold text-slate-500">องศา ROM</div>
                  <div className="mt-0.5 text-xs font-black text-slate-900">
                    {pairComparison.baseline.measured_rom_deg.toFixed(1)}° →{' '}
                    {pairComparison.followUp.measured_rom_deg.toFixed(1)}°
                  </div>
                  <div
                    className={`mt-1 text-[11px] font-black ${
                      pairComparison.romDeltaDeg >= 0 ? 'text-emerald-700' : 'text-rose-700'
                    }`}
                  >
                    {pairComparison.romDeltaDeg >= 0 ? '+' : ''}
                    {pairComparison.romDeltaDeg.toFixed(1)}°
                    {pairComparison.romPercentChange !== null
                      ? ` (${pairComparison.romPercentChange >= 0 ? '+' : ''}${pairComparison.romPercentChange.toFixed(1)}%)`
                      : ''}
                  </div>
                </div>

                <div className="rounded-xl border border-slate-200 bg-slate-50/70 p-3">
                  <div className="text-[10px] font-bold text-slate-500">% เทียบเป้าหมาย</div>
                  <div className="mt-0.5 text-xs font-black text-slate-900">
                    {pairComparison.baselineTargetPct.toFixed(1)}% →{' '}
                    {pairComparison.followUpTargetPct.toFixed(1)}%
                  </div>
                  <div
                    className={`mt-1 text-[11px] font-black ${
                      pairComparison.targetPctDelta >= 0 ? 'text-emerald-700' : 'text-rose-700'
                    }`}
                  >
                    {pairComparison.targetPctDelta >= 0 ? '+' : ''}
                    {pairComparison.targetPctDelta.toFixed(1)}%
                  </div>
                </div>

                <div className="rounded-xl border border-slate-200 bg-slate-50/70 p-3">
                  <div className="text-[10px] font-bold text-slate-500">ระดับมหิดล (0-6)</div>
                  <div className="mt-0.5 text-xs font-black text-slate-900">
                    {pairComparison.baseline.initial_score_0_to_6}/6 →{' '}
                    {pairComparison.followUp.initial_score_0_to_6}/6
                  </div>
                  <div
                    className={`mt-1 text-[11px] font-black ${
                      pairComparison.scoreDelta >= 0 ? 'text-[#0F4C81]' : 'text-rose-700'
                    }`}
                  >
                    {pairComparison.scoreDelta >= 0 ? '+' : ''}
                    {pairComparison.scoreDelta} ระดับ
                  </div>
                </div>

                <div className="rounded-xl border border-slate-200 bg-slate-50/70 p-3">
                  <div className="text-[10px] font-bold text-slate-500">ระดับปวด (VAS)</div>
                  {pairComparison.painDelta !== null ? (
                    <>
                      <div className="mt-0.5 text-xs font-black text-slate-900">
                        {pairComparison.baseline.pain_vas_0_to_10}/10 →{' '}
                        {pairComparison.followUp.pain_vas_0_to_10}/10
                      </div>
                      <div
                        className={`mt-1 text-[11px] font-black ${
                          pairComparison.painDelta <= 0 ? 'text-emerald-700' : 'text-rose-700'
                        }`}
                      >
                        {pairComparison.painDelta === 0
                          ? 'คงที่ (0)'
                          : pairComparison.painDelta < 0
                            ? `ลดลง ${Math.abs(pairComparison.painDelta)} (ดีขึ้น)`
                            : `เพิ่มขึ้น +${pairComparison.painDelta}`}
                      </div>
                    </>
                  ) : (
                    <div className="mt-1 text-[11px] font-semibold text-slate-400">ไม่ได้ระบุ VAS</div>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Per-Exercise & Side Breakdown Comparison Table */}
          {overallBreakdown.length > 0 && (
            <div className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm space-y-3">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <h3 className="text-sm font-black text-slate-900">
                  ตารางเปรียบเทียบพัฒนาการแยกตามรายท่ากายภาพและข้าง ({overallBreakdown.length} กลุ่ม)
                </h3>
                <span className="text-[11px] font-semibold text-slate-500">
                  แยกคำนวณตามข้อต่อจริง ไม่นำองศาต่างท่ามาลบกัน
                </span>
              </div>

              <div className="space-y-2.5">
                {overallBreakdown.map((item) => {
                  const isFocused =
                    exerciseFilter === item.exerciseCode &&
                    (sideFilter === 'ALL' || sideFilter === item.targetSide);
                  return (
                    <div
                      key={item.key}
                      className={`rounded-2xl border p-3.5 transition-colors ${
                        isFocused
                          ? 'border-emerald-400 bg-emerald-50/40'
                          : 'border-slate-200 bg-slate-50/50 hover:border-sky-300'
                      }`}
                    >
                      <div className="flex flex-wrap items-center justify-between gap-2">
                        <div className="min-w-0">
                          <div className="flex flex-wrap items-center gap-1.5">
                            <span className="text-xs font-black text-slate-900">
                              {item.exerciseNameTh}
                            </span>
                            <span className="rounded-md bg-sky-100 px-1.5 py-0.5 text-[10px] font-extrabold text-[#0F4C81]">
                              ข้าง{formatSideTh(item.targetSide)}
                            </span>
                            <span className="rounded-md bg-slate-200/70 px-1.5 py-0.5 text-[10px] font-bold text-slate-700">
                              {item.visitCount} ครั้ง
                            </span>
                          </div>
                          <div className="mt-1 text-[11px] text-slate-600">
                            ครั้งแรก ({formatShortThaiDate(item.firstVisit.assessed_at)}):{' '}
                            <span className="font-bold text-slate-800">{item.firstRomDeg.toFixed(1)}°</span>
                            {' → '}
                            ล่าสุด ({formatShortThaiDate(item.latestVisit.assessed_at)}):{' '}
                            <span className="font-black text-emerald-700">{item.latestRomDeg.toFixed(1)}°</span>
                            {' • '}
                            เป้าหมาย {item.targetRomDeg.toFixed(0)}° ({item.latestTargetAchievementPct.toFixed(1)}%)
                          </div>
                        </div>

                        <div className="flex flex-wrap items-center gap-1.5">
                          <span
                            className={`rounded-full border px-2.5 py-0.5 text-[11px] font-black ${
                              item.trendStatus === 'IMPROVED'
                                ? 'border-emerald-300 bg-emerald-100 text-emerald-800'
                                : item.trendStatus === 'REGRESSED'
                                  ? 'border-rose-300 bg-rose-100 text-rose-800'
                                  : item.trendStatus === 'STABLE'
                                    ? 'border-amber-300 bg-amber-100 text-amber-800'
                                    : 'border-slate-200 bg-white text-slate-600'
                            }`}
                          >
                            {item.visitCount < 2
                              ? 'ค่าตั้งต้น (รอบแรก)'
                              : `${item.deltaRomDeg >= 0 ? '+' : ''}${item.deltaRomDeg.toFixed(1)}°`}
                          </span>
                          <button
                            type="button"
                            onClick={() => {
                              setExerciseFilter(item.exerciseCode);
                              setSideFilter(item.targetSide);
                              setChartMetricMode('ROM_DEG');
                            }}
                            className="rounded-lg border border-sky-200 bg-white px-2.5 py-1 text-[10px] font-extrabold text-[#0F4C81] hover:bg-sky-50 cursor-pointer"
                          >
                            ดูกราฟท่านี้
                          </button>
                        </div>
                      </div>

                      <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-slate-200">
                        <div
                          className="h-full rounded-full bg-gradient-to-r from-[#0284C7] to-[#10B981]"
                          style={{ width: `${Math.min(100, item.latestTargetAchievementPct)}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Chronological Individual History Timeline */}
          <div className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
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

            {summary.visitsWithDelta.length === 0 ? (
              <div className="rounded-2xl border border-dashed border-slate-200 bg-slate-50/60 py-8 px-4 text-center text-xs text-slate-500">
                ยังไม่มีประวัติการประเมินในเงื่อนไขที่เลือก
              </div>
            ) : (
              <div className="space-y-3 max-h-[540px] overflow-y-auto pr-1">
                {summary.visitsWithDelta
                  .slice()
                  .reverse()
                  .map(({ record, deltaFromPrevRom, targetAchievementPct }) => {
                    const isCourseAuto = record.source_channel === 'COURSE_AUTO_CALCULATED';
                    const isKiosk = record.source_channel === 'KIOSK_SESSION_AUTO_SYNC';
                    const isCameraAi = record.measurement_method === 'CAMERA_AI_LIVE';
                    const canDelete =
                      !record.id.startsWith('kiosk-sess-') && !record.id.startsWith('hist-sess-');
                    return (
                      <div
                        key={record.id}
                        className="rounded-2xl border border-slate-200 bg-slate-50/60 p-4 space-y-2 hover:border-sky-300 transition-colors card-hover-lift animate-card-enter"
                      >
                        <div className="flex flex-wrap items-center justify-between gap-2">
                          <div className="flex flex-wrap items-center gap-1.5">
                            <span className="rounded-full bg-[#0F4C81] px-2.5 py-0.5 text-[11px] font-extrabold text-white">
                              ครั้งที่ {record.visit_number}
                            </span>
                            <span className="text-xs font-bold text-slate-700">
                              {formatThaiDateTime(record.assessed_at)}
                            </span>
                            <span
                              className={`rounded-full border px-2 py-0.5 text-[10px] font-bold ${
                                isCourseAuto
                                  ? 'border-emerald-300 bg-emerald-50 text-emerald-800'
                                  : isCameraAi
                                    ? 'border-indigo-200 bg-indigo-50 text-indigo-800'
                                    : isKiosk
                                      ? 'border-teal-200 bg-teal-50 text-teal-800'
                                      : 'border-sky-200 bg-sky-50 text-[#0F4C81]'
                              }`}
                            >
                              {isCourseAuto
                                ? 'คำนวณอัตโนมัติจากคอร์ส'
                                : isCameraAi
                                  ? 'วัดด้วยกล้อง AI'
                                  : isKiosk
                                    ? 'ซิงก์จากตู้ Kiosk'
                                    : 'ประเมินโดยบุคลากร'}
                            </span>
                          </div>

                          <div className="flex flex-wrap items-center gap-1.5">
                            <span className="rounded-full bg-emerald-600 px-2.5 py-0.5 text-[11px] font-extrabold text-white">
                              ระดับ {record.initial_score_0_to_6} / 6
                            </span>
                            <span className="rounded-full border border-emerald-300 bg-white px-2.5 py-0.5 text-[11px] font-black text-emerald-800">
                              ROM {record.measured_rom_deg.toFixed(1)}° ({targetAchievementPct.toFixed(0)}%)
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
                            {canDelete &&
                              (confirmDeleteId === record.id ? (
                                <div className="inline-flex items-center gap-1">
                                  <button
                                    type="button"
                                    disabled={deletingId === record.id}
                                    onClick={() => void handleDeleteAssessment(record.id)}
                                    className="rounded-lg bg-rose-600 px-2 py-0.5 text-[10px] font-extrabold text-white cursor-pointer disabled:opacity-50"
                                  >
                                    {deletingId === record.id ? 'กำลังลบ...' : 'ยืนยันลบ'}
                                  </button>
                                  <button
                                    type="button"
                                    onClick={() => setConfirmDeleteId(null)}
                                    className="rounded-lg border border-slate-200 bg-white px-2 py-0.5 text-[10px] font-bold text-slate-600 cursor-pointer"
                                  >
                                    ยกเลิก
                                  </button>
                                </div>
                              ) : (
                                <button
                                  type="button"
                                  onClick={() => setConfirmDeleteId(record.id)}
                                  className="rounded-lg p-1 text-slate-400 hover:bg-rose-50 hover:text-rose-600 cursor-pointer"
                                  title="ลบรายการประเมินนี้"
                                >
                                  <Trash2 className="h-3.5 w-3.5" />
                                </button>
                              ))}
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
                            {record.exercise_name_th || record.exercise_code} (ข้าง
                            {formatSideTh(record.target_side)}) • เป้าหมาย {record.target_rom_deg.toFixed(0)}°
                            {record.completed_reps ? (
                              <span className="ml-2 text-teal-700 font-semibold">
                                • ทำครบ {record.completed_reps} ครั้ง
                              </span>
                            ) : null}
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
            )}
          </div>
        </div>
      </div>

      {/* Live Camera AI Clinical ROM Measurement Modal */}
      <ClinicalRomLiveMeasureModal
        open={liveMeasureModalOpen}
        exerciseCode={selectedExerciseMeta.code}
        exerciseNameTh={selectedExerciseMeta.name.th}
        targetSide={targetSide}
        targetRomDeg={Number(targetRomInput) || selectedExerciseMeta.detection.targetMinDeg}
        onClose={() => setLiveMeasureModalOpen(false)}
        onConfirmMeasuredRom={(res) => {
          setMeasuredRomInput(res.measuredRomDeg.toFixed(1));
          setMeasurementMethod('CAMERA_AI_LIVE');
          if (res.applySuggestedScore) {
            setSelectedScore(res.suggestedScore0To6);
          }
          showToast(
            'success',
            `นำค่าที่วัดได้จริงจากกล้อง AI ${res.measuredRomDeg.toFixed(1)}° (${res.sampleFrames} เฟรม) กรอกลงแบบประเมินแล้ว`,
          );
        }}
      />
    </div>
  );
}
