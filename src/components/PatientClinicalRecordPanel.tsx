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
import {
  buildMultiExerciseChartSeries,
  computeClinicalMilestones,
  computeMonthlyScoreSummary,
  formatThaiClinicalDate,
  formatThaiMonthYear,
  formatThaiShortDate,
  getCalendarMonthKey,
} from '../utils/clinicalChartAnalytics';
import { buildRehabStockIndex } from '../utils/rehabStockScoreEngine';
import { ClinicalRomLiveMeasureModal } from './ClinicalRomLiveMeasureModal';
import { RehabStockPerformanceCard } from './RehabStockPerformanceCard';

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
  initialRecordsOverride?: PatientClinicalAssessmentRecord[];
}

type DateRangeFilter = 'ALL' | '7D' | '30D' | '90D';
type SourceFilter =
  | 'ALL'
  | 'STAFF_MANUAL_ASSESSMENT'
  | 'KIOSK_SESSION_AUTO_SYNC'
  | 'COURSE_AUTO_CALCULATED';
type ChartMetricMode = 'ROM_DEG' | 'TARGET_PCT';
type ChartViewMode = 'SESSIONS' | 'MONTHLY';

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
  return formatThaiClinicalDate(iso);
}

function formatShortThaiDate(iso: string): string {
  return formatThaiShortDate(iso);
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
  initialRecordsOverride,
}: PatientClinicalRecordPanelProps) {
  const [patientQuery, setPatientQuery] = useState('');
  const [records, setRecords] = useState<PatientClinicalAssessmentRecord[]>(
    () => initialRecordsOverride ?? [],
  );
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
  const [chartViewMode, setChartViewMode] = useState<ChartViewMode>('SESSIONS');
  const [legendFocusExerciseCode, setLegendFocusExerciseCode] = useState<string | null>(null);
  const [selectedMonthKey, setSelectedMonthKey] = useState<string | null>(null);
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
    setRecords(initialRecordsOverride ?? []);
    setExerciseFilter('ALL');
    setSideFilter('ALL');
    setDateRangeFilter('ALL');
    setSourceFilter('ALL');
    setLegendFocusExerciseCode(null);
    setSelectedMonthKey(null);
    setChartViewMode('SESSIONS');
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

  // 3-Milestone Navigator (Previous, Latest, Personal Best)
  const milestones = useMemo(
    () => computeClinicalMilestones(summary.visitsWithDelta),
    [summary.visitsWithDelta],
  );

  // Multi-Exercise Color-Coded Grouping across the active visits
  const multiExerciseGrouping = useMemo(
    () => buildMultiExerciseChartSeries(summary.visitsWithDelta),
    [summary.visitsWithDelta],
  );

  // Monthly Score Summary (grouped by YYYY-MM in Buddhist Era Thai labels)
  const monthlySummary = useMemo(
    () => computeMonthlyScoreSummary(filteredRecords),
    [filteredRecords],
  );

  // Stock-Market-Style Total Rehab Percentage & Full-Score Index across all doctor-assigned exercises
  const rehabStockIndex = useMemo(
    () =>
      buildRehabStockIndex({
        assignedExercises: courseAutoBundle?.assignedExercises ?? [],
        completedSessions: courseAutoBundle?.completedSessions ?? [],
        assessments: records,
      }),
    [courseAutoBundle?.assignedExercises, courseAutoBundle?.completedSessions, records],
  );

  // Reset legend isolation if the focused exercise is no longer present in the filtered set
  useEffect(() => {
    if (
      legendFocusExerciseCode &&
      !multiExerciseGrouping.series.some((s) => s.exerciseCode === legendFocusExerciseCode)
    ) {
      setLegendFocusExerciseCode(null);
    }
  }, [legendFocusExerciseCode, multiExerciseGrouping.series]);

  // Reset month filter if no longer present in monthlySummary
  useEffect(() => {
    if (selectedMonthKey && !monthlySummary.some((m) => m.monthKey === selectedMonthKey)) {
      setSelectedMonthKey(null);
    }
  }, [selectedMonthKey, monthlySummary]);

  const handleJumpToMilestone = useCallback((recordId: string) => {
    setChartViewMode('SESSIONS');
    setLegendFocusExerciseCode(null);
    setSelectedNodeId(recordId);
  }, []);

  // Collision-free Pure SVG Multi-Exercise Progress Chart geometry
  const chartData = useMemo(() => {
    const allVisits = summary.visitsWithDelta;
    let pts = allVisits.slice(-16);
    if (selectedNodeId && !pts.some((v) => v.record.id === selectedNodeId)) {
      const targetIdx = allVisits.findIndex((v) => v.record.id === selectedNodeId);
      if (targetIdx >= 0) {
        const start = Math.max(0, targetIdx - 4);
        pts = allVisits.slice(start, start + 16);
      }
    }

    const width = Math.max(700, pts.length > 10 ? pts.length * 64 + 100 : 700);
    const height = 280;
    const padL = 56;
    const padR = 46;
    const padT = 36;
    const padB = 62;
    const innerPadX = 36;
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

    const isMultiExerciseView =
      exerciseFilter === 'ALL' && multiExerciseGrouping.series.length > 1;

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
      const exCode = (item.record.exercise_code || 'UNKNOWN').trim().toUpperCase();
      const exerciseColor = multiExerciseGrouping.colorByExerciseCode[exCode] ?? '#059669';

      return {
        id: item.record.id,
        record: item.record,
        exerciseCode: exCode,
        exerciseNameTh: item.record.exercise_name_th || exCode,
        exerciseColor,
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
        shortDate: formatThaiShortDate(item.record.assessed_at),
        fullThaiDate: formatThaiClinicalDate(item.record.assessed_at),
        isMilestonePrevious: milestones.previous?.record.id === item.record.id,
        isMilestoneLatest: milestones.latest?.record.id === item.record.id,
        isMilestoneBest: milestones.best?.record.id === item.record.id,
      };
    });

    // Separate color-coded polylines per exercise (never connecting unrelated exercises into 1 line)
    const exerciseSeriesPaths = multiExerciseGrouping.series.map((seriesItem) => {
      const seriesNodes = nodes.filter((n) => n.exerciseCode === seriesItem.exerciseCode);
      const ptsStr = seriesNodes.map((n) => `${n.x.toFixed(1)},${n.y.toFixed(1)}`).join(' ');
      const areaStr =
        seriesNodes.length >= 2
          ? `${seriesNodes[0].x.toFixed(1)},${(padT + plotH).toFixed(1)} ${ptsStr} ${seriesNodes[seriesNodes.length - 1].x.toFixed(1)},${(padT + plotH).toFixed(1)}`
          : '';
      return {
        exerciseCode: seriesItem.exerciseCode,
        exerciseNameTh: seriesItem.exerciseNameTh,
        color: seriesItem.color,
        nodes: seriesNodes,
        polylinePoints: ptsStr,
        areaPoints: areaStr,
      };
    });

    const visibleScoreNodes = legendFocusExerciseCode
      ? nodes.filter((n) => n.exerciseCode === legendFocusExerciseCode)
      : nodes;
    const scorePolylinePoints = visibleScoreNodes
      .map((n) => `${n.x.toFixed(1)},${n.scoreY.toFixed(1)}`)
      .join(' ');
    const painNodes = visibleScoreNodes.filter((n) => n.painY !== null);
    const painPolylinePoints = painNodes
      .map((n) => `${n.x.toFixed(1)},${(n.painY as number).toFixed(1)}`)
      .join(' ');

    const focusedSeriesTargetRom = legendFocusExerciseCode
      ? nodes.find((n) => n.exerciseCode === legendFocusExerciseCode)?.record.target_rom_deg ?? null
      : null;
    const targetRefValue = isPctMode ? 100 : focusedSeriesTargetRom ?? summary.targetRomDeg;
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
      isMultiExerciseView,
      nodes,
      exerciseSeriesPaths,
      scorePolylinePoints,
      painNodes,
      painPolylinePoints,
      targetRefValue,
      targetRefY,
    };
  }, [
    summary.visitsWithDelta,
    summary.targetRomDeg,
    effectiveChartMode,
    exerciseFilter,
    multiExerciseGrouping,
    legendFocusExerciseCode,
    milestones,
    selectedNodeId,
  ]);

  // 60FPS Pure SVG Monthly Score Summary Chart geometry (0-6 Y-axis scale)
  const monthlyChartData = useMemo(() => {
    const items = monthlySummary.slice(-12);
    const width = Math.max(660, items.length > 7 ? items.length * 92 + 96 : 660);
    const height = 256;
    const padL = 52;
    const padR = 32;
    const padT = 36;
    const padB = 64;
    const plotW = width - padL - padR;
    const plotH = height - padT - padB;
    const slotW = items.length > 0 ? plotW / items.length : plotW;
    const barW = Math.min(46, Math.max(22, slotW * 0.44));

    const bars = items.map((m, idx) => {
      const cx = padL + slotW * idx + slotW / 2;
      const barH = Math.max(4, (Math.max(0, Math.min(6, m.avgScore0To6)) / 6) * plotH);
      const y = padT + plotH - barH;
      const maxY = padT + plotH - (Math.max(0, Math.min(6, m.maxScore0To6)) / 6) * plotH;
      return {
        ...m,
        cx,
        x: cx - barW / 2,
        y,
        barW,
        barH,
        maxY,
      };
    });

    const trendPoints = bars.map((b) => `${b.cx.toFixed(1)},${b.y.toFixed(1)}`).join(' ');
    return {
      width,
      height,
      padL,
      padR,
      padT,
      padB,
      plotW,
      plotH,
      bars,
      trendPoints,
    };
  }, [monthlySummary]);

  const activeSelectedNode = useMemo(() => {
    if (chartData.nodes.length === 0) return null;
    if (selectedNodeId) {
      const found = chartData.nodes.find((n) => n.id === selectedNodeId);
      if (found) return found;
    }
    return chartData.nodes[chartData.nodes.length - 1];
  }, [chartData.nodes, selectedNodeId]);

  const timelineVisits = useMemo(() => {
    const base = summary.visitsWithDelta;
    if (!selectedMonthKey) return base;
    return base.filter((v) => getCalendarMonthKey(v.record.assessed_at) === selectedMonthKey);
  }, [summary.visitsWithDelta, selectedMonthKey]);

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
    <div className="space-y-3 animate-view-enter">
      {toast && (
        <div
          className={`rounded-xl border px-3 py-2 text-xs font-bold shadow-xs animate-toast-in ${
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

      {/* SECTION A: Compact Individual Patient Selector & Identity Bar */}
      <div className="rounded-2xl border border-sky-200/80 bg-white p-3 shadow-sm space-y-2">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex flex-wrap items-center gap-2 flex-1 min-w-0">
            <div className="flex items-center rounded-xl border border-slate-200 bg-slate-50/70 px-2.5 h-8 w-44 sm:w-52 shrink-0">
              <Search className="h-3.5 w-3.5 text-slate-400 mr-1.5 shrink-0" />
              <input
                value={patientQuery}
                onChange={(e) => setPatientQuery(e.target.value)}
                placeholder="ค้นหาชื่อ, HN..."
                className="w-full bg-transparent text-xs text-slate-800 placeholder:text-slate-400 focus:outline-none"
              />
            </div>
            <select
              aria-label="เลือกผู้ป่วยรายบุคคล"
              value={selectedPatient?.id ?? ''}
              onChange={(e) => onSelectPatientId(e.target.value)}
              className="h-8 flex-1 min-w-[200px] rounded-xl border border-slate-200 bg-white px-2.5 text-xs font-bold text-slate-800 focus:border-sky-500 focus:outline-none"
            >
              {filteredPatients.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.fullName} ({p.hn}) — {resolvePatientUniversalLinkCode(p)}
                </option>
              ))}
            </select>
          </div>

          <div className="flex flex-wrap items-center gap-1.5 shrink-0">
            <button
              type="button"
              disabled={!selectedPatient || syncingCrossFacility}
              onClick={() => {
                void loadPatientHistory(selectedPatient, true);
                void loadPatientCourseContext(selectedPatient);
              }}
              className="btn-tactile inline-flex items-center gap-1 rounded-xl border border-sky-200 bg-sky-50 px-2.5 py-1.5 text-[11px] font-bold text-[#0F4C81] hover:bg-sky-100 disabled:opacity-50 cursor-pointer"
            >
              <RefreshCw className={`h-3 w-3 ${syncingCrossFacility ? 'animate-spin' : ''}`} />
              <span>ซิงก์</span>
            </button>
            <button
              type="button"
              disabled={!selectedPatient || summary.totalVisits === 0}
              onClick={handleExportCsv}
              className="btn-tactile inline-flex items-center gap-1 rounded-xl border border-emerald-200 bg-emerald-50 px-2.5 py-1.5 text-[11px] font-bold text-emerald-700 hover:bg-emerald-100 disabled:opacity-45 cursor-pointer"
            >
              <Download className="h-3 w-3" />
              <span>CSV</span>
            </button>
            {selectedPatient && onOpenUniversalEhrModal && (
              <button
                type="button"
                onClick={() => onOpenUniversalEhrModal(selectedPatient)}
                className="btn-tactile inline-flex items-center gap-1 rounded-xl border border-indigo-200 bg-indigo-50 px-2.5 py-1.5 text-[11px] font-bold text-indigo-700 hover:bg-indigo-100 cursor-pointer"
              >
                <Link2 className="h-3 w-3" />
                <span>EHR</span>
              </button>
            )}
          </div>
        </div>

        {/* Selected Patient Compact Identity Strip */}
        {selectedPatient && (
          <div className="rounded-xl border border-emerald-200/80 bg-gradient-to-r from-[#F0FDF9] via-white to-[#EFF6FF] px-3 py-2 flex flex-wrap items-center justify-between gap-2">
            <div className="flex flex-wrap items-center gap-1.5 min-w-0 text-xs">
              <span className="font-black text-slate-900">{selectedPatient.fullName}</span>
              <span className="rounded-full bg-sky-100 border border-sky-200 px-2 py-0.5 text-[10.5px] font-bold text-[#0F4C81]">
                {selectedPatient.hn}
              </span>
              <span className="rounded-full bg-slate-100 px-2 py-0.5 text-[10.5px] font-semibold text-slate-700">
                {selectedPatient.age} ปี ({selectedPatient.gender || '-'})
              </span>
              <span className="text-[11px] text-slate-600 truncate">
                • <strong className="text-slate-800">{selectedPatient.diagnosis}</strong>
                {selectedPatient.affectedSide && selectedPatient.affectedSide !== 'NONE' && (
                  <span className="text-emerald-700 font-bold">
                    {' '}({selectedPatient.affectedSide === 'LEFT' ? 'ซ้าย' : selectedPatient.affectedSide === 'RIGHT' ? 'ขวา' : '2 ข้าง'})
                  </span>
                )}
              </span>
            </div>

            <div className="flex items-center gap-1.5 shrink-0">
              <span className="font-mono text-[10.5px] font-black text-emerald-800 bg-white border border-emerald-200 rounded-lg px-2 py-1">
                {universalLinkCode}
              </span>
              <button
                type="button"
                onClick={() => void handleCopyLinkCode()}
                className="inline-flex items-center gap-1 rounded-lg bg-emerald-600 px-2 py-1 text-[10.5px] font-bold text-white hover:bg-emerald-500 cursor-pointer"
              >
                {copiedCode ? <Check className="h-3 w-3" /> : <Copy className="h-3 w-3" />}
                <span>คัดลอก</span>
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Main Two-Column Clinical Workspace (side-by-side at md+) */}
      <div className="grid grid-cols-1 gap-3 md:grid-cols-12">
        {/* SECTION B: Assigned Rehab Courses Auto-Score Card + 0-6 Assessment Form */}
        <div className="md:col-span-5 space-y-2.5 self-start">
          {/* Assigned Course & Automatic 0-6 Mahidol Scoring Card */}
          <div
            data-testid="assigned-course-auto-score-card"
            className="rounded-2xl border border-emerald-200/90 bg-gradient-to-br from-white via-[#F0FDF9] to-[#EFF6FF] p-3 shadow-sm space-y-2 animate-view-enter"
          >
            <div className="flex flex-wrap items-center justify-between gap-1.5 border-b border-emerald-100 pb-2">
              <div className="flex items-center gap-2 min-w-0">
                <ClipboardList className="h-4 w-4 text-[#0F4C81] shrink-0" />
                <h3 className="text-[11.5px] font-black text-slate-900 leading-tight truncate">
                  คอร์สกายภาพที่นักกายภาพจัดให้ &amp; คำนวณคะแนนอัตโนมัติ
                </h3>
              </div>
              {courseAutoBundle && courseAutoBundle.unrecordedResults.length > 0 ? (
                <span className="shrink-0 rounded-full bg-emerald-600 px-2 py-0.5 text-[10px] font-black text-white">
                  รอบันทึก {courseAutoBundle.unrecordedResults.length}
                </span>
              ) : courseAutoBundle && courseAutoBundle.autoResults.length > 0 ? (
                <span className="shrink-0 rounded-full border border-emerald-200 bg-emerald-50 px-2 py-0.5 text-[10px] font-extrabold text-emerald-700">
                  ซิงก์ครบ {courseAutoBundle.autoResults.length} รอบ
                </span>
              ) : null}
            </div>

            {loadingCourseContext && (!courseAutoBundle || courseAutoBundle.exerciseSummaries.length === 0) ? (
              <div className="rounded-xl border border-slate-200/80 bg-white/80 px-2.5 py-2 text-[11px] text-slate-500">
                กำลังโหลดข้อมูลคอร์ส...
              </div>
            ) : courseAutoBundle && courseAutoBundle.exerciseSummaries.length > 0 ? (
              <div className="space-y-1.5">
                <div className="space-y-1 max-h-32 overflow-y-auto pr-0.5">
                  {courseAutoBundle.exerciseSummaries.map((item) => {
                    const latest = item.latestResult;
                    const isSelectedInForm =
                      activeAutoCalcResult &&
                      latest &&
                      activeAutoCalcResult.sourceSessionId === latest.sourceSessionId;
                    return (
                      <div
                        key={item.definition.key}
                        className={`rounded-xl border p-2 transition-colors ${
                          isSelectedInForm
                            ? 'border-emerald-400 bg-white shadow-xs'
                            : 'border-slate-200/80 bg-white/90 hover:border-sky-300'
                        }`}
                      >
                        <div className="flex flex-wrap items-center justify-between gap-1.5">
                          <div className="min-w-0">
                            <span className="text-[11px] font-black text-slate-900">
                              {item.definition.exerciseNameTh}
                            </span>
                            <span className="ml-1 rounded bg-sky-100 px-1 py-0.5 text-[9.5px] font-bold text-[#0F4C81]">
                              {formatSideTh(item.definition.targetSide)}
                            </span>
                          </div>
                          <div className="flex flex-wrap items-center gap-1">
                            {latest ? (
                              <>
                                <span className="rounded-full bg-emerald-600 px-1.5 py-0.5 text-[10px] font-black text-white">
                                  ระดับ {latest.calculatedScore0To6}/6
                                </span>
                                <span className="rounded-full border border-emerald-200 bg-emerald-50 px-1.5 py-0.5 text-[10px] font-extrabold text-emerald-800">
                                  ROM {latest.effectiveRomDeg.toFixed(1)}°
                                </span>
                                <button
                                  type="button"
                                  onClick={() => {
                                    formManuallyEditedRef.current = false;
                                    applyCourseResultToForm(latest, true);
                                  }}
                                  className="btn-tactile rounded-md border border-sky-200 bg-sky-50 px-1.5 py-0.5 text-[10px] font-extrabold text-[#0F4C81] hover:bg-sky-100 cursor-pointer"
                                >
                                  ใช้
                                </button>
                              </>
                            ) : (
                              <span className="rounded-full bg-slate-100 px-2 py-0.5 text-[9.5px] font-bold text-slate-500">
                                รอฝึก
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
                    className="rounded-xl border border-sky-200 bg-white/95 px-2.5 py-1.5 text-[10.5px] flex items-center justify-between gap-2 animate-toast-in"
                  >
                    <span className="font-semibold text-slate-700 truncate">
                      {courseAutoBundle.latestResult.breakdownSummaryTh}
                    </span>
                    <span className="shrink-0 rounded bg-sky-50 border border-sky-200 px-1.5 py-0.5 text-[10px] font-extrabold text-[#0F4C81]">
                      {courseAutoBundle.latestResult.compositePct}%
                    </span>
                  </div>
                )}

                <div className="flex items-center justify-end">
                  <button
                    type="button"
                    data-testid="batch-sync-course-card-btn"
                    disabled={
                      batchSyncingCourse ||
                      !selectedPatient ||
                      courseAutoBundle.autoResults.length === 0
                    }
                    onClick={() => void handleBatchSyncCourseSessions()}
                    className="btn-tactile inline-flex items-center gap-1 rounded-xl bg-emerald-600 hover:bg-emerald-700 px-2.5 py-1 text-[10.5px] font-extrabold text-white shadow-xs disabled:opacity-45 cursor-pointer"
                  >
                    <RefreshCw
                      className={`h-3 w-3 ${batchSyncingCourse ? 'animate-spin' : ''}`}
                    />
                    <span>
                      {batchSyncingCourse
                        ? 'กำลังบันทึก...'
                        : 'ดึงผลเล่นคอร์สและบันทึกอัตโนมัติ'}
                    </span>
                  </button>
                </div>
              </div>
            ) : (
              <div className="rounded-xl border border-dashed border-emerald-200 bg-white/80 px-2.5 py-2 text-[11px] text-slate-500">
                ยังไม่มีผลฝึกจากคอร์สสำหรับผู้ป่วยรายนี้
              </div>
            )}
          </div>

          <form
            onSubmit={(e) => void handleSubmitAssessment(e)}
            className="rounded-2xl border border-slate-200 bg-white p-3.5 shadow-sm space-y-2.5"
          >
            <div className="flex items-center justify-between gap-2 border-b border-slate-100 pb-2">
              <div className="flex items-center gap-1.5">
                <FileText className="h-4 w-4 text-[#0F4C81]" />
                <h3 className="text-xs font-black text-slate-900">
                  บันทึกการประเมิน (0-6)
                </h3>
              </div>
              <span className="rounded-full bg-emerald-50 border border-emerald-200 px-2 py-0.5 text-[10px] font-bold text-emerald-700">
                {activeAutoCalcResult ? 'คำนวณอัตโนมัติ' : 'Mahidol 0-6'}
              </span>
            </div>

            {/* Interactive 0-6 Score Picker */}
            <div className="space-y-1.5">
              <div className="grid grid-cols-7 gap-1.5">
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
                      className={`btn-tactile h-9 rounded-xl border text-xs font-black transition-all cursor-pointer ${
                        active
                          ? 'bg-emerald-600 text-white border-emerald-700 shadow-sm'
                          : 'bg-slate-50 text-slate-700 border-slate-200 hover:border-sky-300 hover:bg-sky-50'
                      }`}
                    >
                      {item.score}
                    </button>
                  );
                })}
              </div>
              <div className="rounded-lg border border-emerald-200 bg-emerald-50/70 px-2.5 py-1 text-[11px] font-semibold text-emerald-900 truncate">
                {selectedRubric.descriptionTh}
              </div>
            </div>

            {/* Exercise + Target Side */}
            <div className="grid grid-cols-1 gap-2 sm:grid-cols-12">
              <div className="sm:col-span-7">
                <select
                  aria-label="ท่ากายภาพบำบัด"
                  value={exerciseCode}
                  onChange={(e) => {
                    formManuallyEditedRef.current = true;
                    handleExerciseChange(e.target.value);
                  }}
                  className="h-8 w-full rounded-lg border border-slate-200 bg-white px-2 text-[11px] font-semibold text-slate-800 focus:border-sky-500 focus:outline-none"
                >
                  {CLINICAL_EXERCISES.map((ex) => (
                    <option key={ex.code} value={ex.code}>
                      {ex.name.th} ({ex.detection.targetMinDeg}°)
                    </option>
                  ))}
                </select>
              </div>
              <div className="sm:col-span-5">
                <div className="grid grid-cols-3 gap-1 h-8">
                  {SIDE_OPTIONS.map((opt) => (
                    <button
                      key={opt.value}
                      type="button"
                      onClick={() => {
                        formManuallyEditedRef.current = true;
                        setTargetSide(opt.value);
                      }}
                      className={`rounded-lg border text-[10.5px] font-bold cursor-pointer transition-colors ${
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

            {/* Measured ROM + Quick Actions */}
            <div className="rounded-xl border border-sky-200 bg-sky-50/50 p-2.5 space-y-2">
              <div className="flex flex-wrap items-center justify-between gap-1">
                <span className="text-[11px] font-bold text-[#0F4C81]">ROM (°) / VAS</span>
                <div className="flex flex-wrap items-center gap-1">
                  <button
                    type="button"
                    data-testid="batch-sync-course-inline-btn"
                    disabled={batchSyncingCourse || !selectedPatient}
                    onClick={() => void handleBatchSyncCourseSessions()}
                    className="btn-tactile inline-flex items-center gap-1 rounded-md bg-emerald-600 hover:bg-emerald-700 px-2 py-1 text-[10px] font-extrabold text-white disabled:opacity-50 cursor-pointer"
                  >
                    <RefreshCw className={`h-2.5 w-2.5 ${batchSyncingCourse ? 'animate-spin' : ''}`} />
                    <span>ดึงคอร์ส</span>
                  </button>
                  <button
                    type="button"
                    onClick={handlePullLatestKioskRom}
                    className="btn-tactile inline-flex items-center gap-1 rounded-md border border-emerald-300 bg-white px-2 py-1 text-[10px] font-bold text-emerald-700 hover:bg-emerald-50 cursor-pointer"
                  >
                    <Sparkles className="h-2.5 w-2.5" />
                    <span>Kiosk</span>
                  </button>
                  <button
                    type="button"
                    onClick={() => setLiveMeasureModalOpen(true)}
                    className="btn-tactile inline-flex items-center gap-1 rounded-md border border-sky-300 bg-white px-2 py-1 text-[10px] font-bold text-[#0F4C81] hover:bg-sky-50 cursor-pointer"
                  >
                    <Camera className="h-3 w-3" />
                    <span>กล้อง AI</span>
                  </button>
                </div>
              </div>

              <div className="grid grid-cols-3 gap-1.5">
                <div>
                  <span className="block text-[10px] font-semibold text-slate-500 mb-0.5">ROM (°)</span>
                  <input
                    type="number"
                    step="0.5"
                    min={0}
                    max={220}
                    placeholder="อัตโนมัติ"
                    value={measuredRomInput}
                    onChange={(e) => {
                      formManuallyEditedRef.current = true;
                      setMeasuredRomInput(e.target.value);
                      setMeasurementMethod('MANUAL_INPUT');
                      setActiveAutoCalcResult(null);
                    }}
                    className="h-8 w-full rounded-lg border border-slate-200 bg-white px-2 text-xs font-black text-slate-900 focus:border-sky-500 focus:outline-none"
                  />
                </div>
                <div>
                  <span className="block text-[10px] font-semibold text-slate-500 mb-0.5">เป้า (°)</span>
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
                    className="h-8 w-full rounded-lg border border-slate-200 bg-white px-2 text-xs font-bold text-slate-700 focus:border-sky-500 focus:outline-none"
                  />
                </div>
                <div>
                  <span className="block text-[10px] font-semibold text-slate-500 mb-0.5">ปวด (0-10)</span>
                  <input
                    type="number"
                    min={0}
                    max={10}
                    placeholder="-"
                    value={painVasInput}
                    onChange={(e) => {
                      formManuallyEditedRef.current = true;
                      setPainVasInput(e.target.value);
                    }}
                    className="h-8 w-full rounded-lg border border-slate-200 bg-white px-2 text-xs font-bold text-slate-700 focus:border-sky-500 focus:outline-none"
                  />
                </div>
              </div>

              {liveFormPreviewPct !== null && liveFormSuggestedScore !== null && (
                <div className="flex flex-wrap items-center justify-between gap-1.5 rounded-lg border border-emerald-200 bg-white px-2.5 py-1 text-[10.5px]">
                  <span className="font-bold text-slate-700">
                    เป้า: <strong className="text-emerald-700">{liveFormPreviewPct.toFixed(1)}%</strong>
                  </span>
                  {liveFormSuggestedScore !== selectedScore && (
                    <button
                      type="button"
                      onClick={() => {
                        formManuallyEditedRef.current = true;
                        setSelectedScore(liveFormSuggestedScore);
                      }}
                      className="rounded border border-sky-200 bg-sky-50 px-1.5 py-0.5 text-[10px] font-extrabold text-[#0F4C81] hover:bg-sky-100 cursor-pointer"
                    >
                      ใช้ {liveFormSuggestedScore}/6
                    </button>
                  )}
                </div>
              )}
            </div>

            {/* Visit Date/Time & Facility */}
            <div className="grid grid-cols-2 gap-2">
              <input
                aria-label="วันที่และเวลาที่ประเมิน"
                type="datetime-local"
                value={assessedAtInput}
                onChange={(e) => setAssessedAtInput(e.target.value)}
                className="h-8 w-full rounded-lg border border-slate-200 bg-white px-2 text-[11px] font-semibold text-slate-800 focus:border-sky-500 focus:outline-none"
              />
              <input
                aria-label="สถานที่ตรวจ"
                type="text"
                value={facilityName}
                onChange={(e) => setFacilityName(e.target.value)}
                className="h-8 w-full rounded-lg border border-slate-200 bg-white px-2 text-[11px] font-semibold text-slate-800 focus:border-sky-500 focus:outline-none"
              />
            </div>

            {/* Clinical Handover Notes (Compact) */}
            <div className="space-y-1.5">
              <input
                type="text"
                value={chiefComplaint}
                onChange={(e) => {
                  formManuallyEditedRef.current = true;
                  setChiefComplaint(e.target.value);
                }}
                placeholder="อาการรอบนี้ (Chief Complaint)..."
                className="h-8 w-full rounded-lg border border-slate-200 bg-white px-2.5 text-xs text-slate-800 focus:border-sky-500 focus:outline-none"
              />
              <textarea
                rows={2}
                value={handoverNote}
                onChange={(e) => {
                  formManuallyEditedRef.current = true;
                  setHandoverNote(e.target.value);
                }}
                placeholder="บันทึกส่งต่อ (Handover Note)..."
                className="w-full rounded-lg border border-slate-200 bg-white p-2 text-xs text-slate-800 focus:border-sky-500 focus:outline-none"
              />
              <input
                type="text"
                value={nextGoal}
                onChange={(e) => {
                  formManuallyEditedRef.current = true;
                  setNextGoal(e.target.value);
                }}
                placeholder="เป้าหมายครั้งถัดไป..."
                className="h-8 w-full rounded-lg border border-slate-200 bg-white px-2.5 text-xs text-slate-800 focus:border-sky-500 focus:outline-none"
              />
            </div>

            <button
              type="submit"
              disabled={saving || !selectedPatient}
              className="btn-primary-sheen btn-game-3d-blue btn-tactile w-full h-9 rounded-xl bg-gradient-to-r from-[#0F4C81] via-[#0284C7] to-[#0D9488] text-xs font-black text-white hover:brightness-105 disabled:opacity-50 cursor-pointer"
            >
              {saving ? 'กำลังบันทึก...' : 'บันทึกผลประเมิน (0-6)'}
            </button>
          </form>
        </div>

        {/* SECTION C: Retrospective Tracking & Charts */}
        <div className="md:col-span-7 space-y-2.5">
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
            <div className="game-card-surface rounded-2xl border border-emerald-200 bg-white p-4 space-y-1.5">
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
            <div className="game-card-surface rounded-2xl border border-slate-200 bg-white p-4 space-y-1.5">
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

          {/* Stock-Market-Style Total Rehab Percentage & Full-Score Index Card */}
          {selectedPatient && (
            <RehabStockPerformanceCard
              patientName={selectedPatient.fullName}
              patientHn={selectedPatient.hn}
              stockIndex={rehabStockIndex}
            />
          )}

          {/* Interactive 3-Milestone Navigator (Previous • Latest • Personal Best) with Thai Dates */}
          <div
            data-testid="clinical-milestone-navigator"
            className="grid grid-cols-1 gap-3 md:grid-cols-3"
          >
            {/* 1. Previous Session */}
            <button
              type="button"
              data-testid="milestone-card-previous"
              disabled={!milestones.previous}
              onClick={() => {
                if (milestones.previous) handleJumpToMilestone(milestones.previous.record.id);
              }}
              className={`btn-tactile game-card-surface text-left rounded-2xl border p-3.5 transition-all space-y-1.5 ${
                milestones.previous
                  ? activeSelectedNode?.id === milestones.previous.record.id
                    ? 'border-[#0F4C81] bg-sky-50/70 ring-2 ring-sky-200 cursor-pointer'
                    : 'border-slate-200 bg-white hover:border-sky-300 cursor-pointer'
                  : 'border-slate-200/70 bg-slate-50/70 opacity-75 cursor-default'
              }`}
            >
              <div className="flex items-center justify-between gap-1.5">
                <span className="inline-flex items-center gap-1 text-[11px] font-black text-slate-700">
                  <svg
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2.2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    className="h-3.5 w-3.5 text-slate-500"
                    aria-hidden="true"
                  >
                    <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
                    <path d="M3 3v5h5" />
                  </svg>
                  <span>ครั้งก่อนหน้า</span>
                </span>
                {milestones.previous && (
                  <span className="rounded-full bg-slate-800 px-2 py-0.5 text-[10px] font-extrabold text-white">
                    ระดับ {milestones.previous.record.initial_score_0_to_6}/6
                  </span>
                )}
              </div>
              {milestones.previous ? (
                <>
                  <div className="text-[11px] font-bold text-[#0F4C81]">
                    {formatThaiClinicalDate(milestones.previous.record.assessed_at)}
                  </div>
                  <div className="text-[11px] font-semibold text-slate-700 truncate">
                    {milestones.previous.record.exercise_name_th}
                  </div>
                  <div className="flex flex-wrap items-center gap-1.5 text-[11px]">
                    <span className="font-black text-slate-900">
                      {milestones.previous.record.measured_rom_deg.toFixed(1)}°
                    </span>
                    <span className="text-slate-400">•</span>
                    <span className="font-bold text-emerald-700">
                      {milestones.previous.targetAchievementPct.toFixed(0)}% เป้าหมาย
                    </span>
                  </div>
                </>
              ) : (
                <div className="text-[11px] text-slate-400 py-1">ยังไม่มีข้อมูลครั้งก่อนหน้า</div>
              )}
            </button>

            {/* 2. Latest Session */}
            <button
              type="button"
              data-testid="milestone-card-latest"
              disabled={!milestones.latest}
              onClick={() => {
                if (milestones.latest) handleJumpToMilestone(milestones.latest.record.id);
              }}
              className={`btn-tactile game-card-surface text-left rounded-2xl border p-3.5 transition-all space-y-1.5 ${
                milestones.latest
                  ? activeSelectedNode?.id === milestones.latest.record.id
                    ? 'border-[#0284C7] bg-sky-50/80 ring-2 ring-sky-200 cursor-pointer'
                    : 'border-sky-200 bg-white hover:border-[#0284C7] cursor-pointer'
                  : 'border-slate-200/70 bg-slate-50/70 opacity-75 cursor-default'
              }`}
            >
              <div className="flex items-center justify-between gap-1.5">
                <span className="inline-flex items-center gap-1 text-[11px] font-black text-[#0F4C81]">
                  <svg
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2.2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    className="h-3.5 w-3.5 text-[#0284C7]"
                    aria-hidden="true"
                  >
                    <circle cx="12" cy="12" r="10" />
                    <polyline points="12 6 12 12 16 14" />
                  </svg>
                  <span>ครั้งล่าสุด</span>
                </span>
                {milestones.latest && (
                  <span className="rounded-full bg-[#0F4C81] px-2 py-0.5 text-[10px] font-extrabold text-white">
                    ระดับ {milestones.latest.record.initial_score_0_to_6}/6
                  </span>
                )}
              </div>
              {milestones.latest ? (
                <>
                  <div className="text-[11px] font-bold text-[#0F4C81]">
                    {formatThaiClinicalDate(milestones.latest.record.assessed_at)}
                  </div>
                  <div className="text-[11px] font-semibold text-slate-700 truncate">
                    {milestones.latest.record.exercise_name_th}
                  </div>
                  <div className="flex flex-wrap items-center gap-1.5 text-[11px]">
                    <span className="font-black text-slate-900">
                      {milestones.latest.record.measured_rom_deg.toFixed(1)}°
                    </span>
                    {milestones.latestDeltaRomFromPrevious !== null && (
                      <span
                        className={`rounded-full px-1.5 py-0.5 text-[10px] font-extrabold ${
                          milestones.latestDeltaRomFromPrevious >= 0
                            ? 'bg-emerald-50 text-emerald-700'
                            : 'bg-rose-50 text-rose-700'
                        }`}
                      >
                        {milestones.latestDeltaRomFromPrevious >= 0 ? '+' : ''}
                        {milestones.latestDeltaRomFromPrevious.toFixed(1)}°
                      </span>
                    )}
                    {milestones.latestDeltaScoreFromPrevious !== null && (
                      <span
                        className={`rounded-full px-1.5 py-0.5 text-[10px] font-extrabold ${
                          milestones.latestDeltaScoreFromPrevious >= 0
                            ? 'bg-sky-50 text-[#0F4C81]'
                            : 'bg-rose-50 text-rose-700'
                        }`}
                      >
                        {milestones.latestDeltaScoreFromPrevious >= 0 ? '+' : ''}
                        {milestones.latestDeltaScoreFromPrevious} ระดับ
                      </span>
                    )}
                  </div>
                </>
              ) : (
                <div className="text-[11px] text-slate-400 py-1">ยังไม่มีข้อมูลครั้งล่าสุด</div>
              )}
            </button>

            {/* 3. Personal Best Session */}
            <button
              type="button"
              data-testid="milestone-card-best"
              disabled={!milestones.best}
              onClick={() => {
                if (milestones.best) handleJumpToMilestone(milestones.best.record.id);
              }}
              className={`btn-tactile game-card-surface text-left rounded-2xl border p-3.5 transition-all space-y-1.5 ${
                milestones.best
                  ? activeSelectedNode?.id === milestones.best.record.id
                    ? 'border-emerald-500 bg-gradient-to-br from-emerald-50 via-white to-sky-50 ring-2 ring-emerald-200 cursor-pointer'
                    : 'border-emerald-300 bg-gradient-to-br from-emerald-50/70 via-white to-sky-50/60 hover:border-emerald-500 cursor-pointer'
                  : 'border-slate-200/70 bg-slate-50/70 opacity-75 cursor-default'
              }`}
            >
              <div className="flex items-center justify-between gap-1.5">
                <span className="inline-flex items-center gap-1 text-[11px] font-black text-emerald-800">
                  <svg
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2.2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    className="h-3.5 w-3.5 text-emerald-600"
                    aria-hidden="true"
                  >
                    <path d="M6 9H4.5a2.5 2.5 0 0 1 0-5H6" />
                    <path d="M18 9h1.5a2.5 2.5 0 0 0 0-5H18" />
                    <path d="M4 22h16" />
                    <path d="M10 14.66V17c0 .55-.47.98-.97 1.21C7.85 18.75 7 20.24 7 22" />
                    <path d="M14 14.66V17c0 .55.47.98.97 1.21C16.15 18.75 17 20.24 17 22" />
                    <path d="M18 2H6v7a6 6 0 0 0 12 0V2Z" />
                  </svg>
                  <span>ครั้งที่ดีที่สุด</span>
                </span>
                {milestones.best && (
                  <span className="rounded-full bg-emerald-600 px-2 py-0.5 text-[10px] font-black text-white">
                    ระดับ {milestones.best.record.initial_score_0_to_6}/6
                  </span>
                )}
              </div>
              {milestones.best ? (
                <>
                  <div className="text-[11px] font-bold text-emerald-800">
                    {formatThaiClinicalDate(milestones.best.record.assessed_at)}
                  </div>
                  <div className="text-[11px] font-semibold text-slate-700 truncate">
                    {milestones.best.record.exercise_name_th}
                  </div>
                  <div className="flex flex-wrap items-center gap-1.5 text-[11px]">
                    <span className="font-black text-emerald-700">
                      {milestones.best.record.measured_rom_deg.toFixed(1)}°
                    </span>
                    <span className="text-slate-400">•</span>
                    <span className="font-bold text-[#0F4C81]">
                      {milestones.best.targetAchievementPct.toFixed(0)}% เป้าหมาย
                    </span>
                  </div>
                </>
              ) : (
                <div className="text-[11px] text-slate-400 py-1">ยังไม่มีข้อมูลสถิติสูงสุด</div>
              )}
            </button>
          </div>

          {/* 60FPS Collision-Free Multi-Exercise & Monthly Score Summary Chart Card */}
          <div className="rounded-2xl border border-slate-200 bg-white p-3.5 shadow-sm space-y-2.5">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <h3 className="text-xs font-black text-slate-900">
                {chartViewMode === 'MONTHLY'
                  ? 'กราฟสรุปรายเดือน (0-6)'
                  : 'กราฟพัฒนาการ (ROM & 0-6)'}
              </h3>

              <div className="flex flex-wrap items-center gap-1.5">
                {/* Primary View Switcher: Sessions vs Monthly Score Summary */}
                <div className="inline-flex rounded-xl border border-sky-200 bg-sky-50/70 p-0.5 text-[10.5px] font-bold">
                  <button
                    type="button"
                    data-testid="chart-tab-sessions"
                    onClick={() => setChartViewMode('SESSIONS')}
                    className={`rounded-lg px-2 py-1 cursor-pointer transition-colors ${
                      chartViewMode === 'SESSIONS'
                        ? 'bg-[#0F4C81] text-white shadow-xs'
                        : 'text-[#0F4C81] hover:bg-white/80'
                    }`}
                  >
                    รายครั้ง
                  </button>
                  <button
                    type="button"
                    data-testid="chart-tab-monthly"
                    onClick={() => setChartViewMode('MONTHLY')}
                    className={`rounded-lg px-2 py-1 cursor-pointer transition-colors ${
                      chartViewMode === 'MONTHLY'
                        ? 'bg-emerald-600 text-white shadow-xs'
                        : 'text-emerald-800 hover:bg-white/80'
                    }`}
                  >
                    กราฟสรุปรายเดือน (ผลคะแนน)
                  </button>
                </div>

                {chartViewMode === 'SESSIONS' && (
                  <>
                    <div className="inline-flex rounded-xl border border-slate-200 bg-slate-50 p-0.5 text-[10.5px] font-bold">
                      <button
                        type="button"
                        onClick={() => setChartMetricMode('ROM_DEG')}
                        className={`rounded-lg px-2 py-1 cursor-pointer transition-colors ${
                          chartMetricMode === 'ROM_DEG'
                            ? 'bg-[#0F4C81] text-white shadow-xs'
                            : 'text-slate-600 hover:text-slate-900'
                        }`}
                      >
                        องศา (°)
                      </button>
                      <button
                        type="button"
                        onClick={() => setChartMetricMode('TARGET_PCT')}
                        className={`rounded-lg px-2 py-1 cursor-pointer transition-colors ${
                          chartMetricMode === 'TARGET_PCT'
                            ? 'bg-emerald-600 text-white shadow-xs'
                            : 'text-slate-600 hover:text-slate-900'
                        }`}
                      >
                        % เป้า
                      </button>
                    </div>

                    <button
                      type="button"
                      onClick={() => setShowMahidolLine((v) => !v)}
                      className={`rounded-lg border px-2 py-1 text-[10px] font-extrabold cursor-pointer ${
                        showMahidolLine
                          ? 'border-sky-300 bg-sky-50 text-[#0F4C81]'
                          : 'border-slate-200 bg-white text-slate-500'
                      }`}
                    >
                      0-6
                    </button>
                    <button
                      type="button"
                      onClick={() => setShowPainLine((v) => !v)}
                      className={`rounded-lg border px-2 py-1 text-[10px] font-extrabold cursor-pointer ${
                        showPainLine
                          ? 'border-amber-300 bg-amber-50 text-amber-800'
                          : 'border-slate-200 bg-white text-slate-500'
                      }`}
                    >
                      VAS
                    </button>
                  </>
                )}
              </div>
            </div>

            {chartViewMode === 'SESSIONS' && (
              <>
                {/* Interactive Multi-Exercise Color Legend */}
                {multiExerciseGrouping.series.length > 0 && (
                  <div
                    data-testid="multi-exercise-chart-legend"
                    className="flex flex-wrap items-center gap-1.5 rounded-2xl border border-slate-200/80 bg-slate-50/70 p-2.5"
                  >
                    {multiExerciseGrouping.series.length > 1 && (
                      <button
                        type="button"
                        onClick={() => setLegendFocusExerciseCode(null)}
                        className={`btn-tactile inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[10px] font-extrabold cursor-pointer transition-colors ${
                          legendFocusExerciseCode === null
                            ? 'border-[#0F4C81] bg-[#0F4C81] text-white shadow-xs'
                            : 'border-slate-200 bg-white text-slate-700 hover:border-sky-300'
                        }`}
                      >
                        <span>แสดงทุกท่าพร้อมกัน ({multiExerciseGrouping.series.length} ท่า)</span>
                      </button>
                    )}
                    {multiExerciseGrouping.series.map((seriesItem) => {
                      const isFocused = legendFocusExerciseCode === seriesItem.exerciseCode;
                      const isDimmed =
                        legendFocusExerciseCode !== null && !isFocused;
                      return (
                        <button
                          key={seriesItem.exerciseCode}
                          type="button"
                          data-testid={`legend-exercise-${seriesItem.exerciseCode}`}
                          onClick={() =>
                            setLegendFocusExerciseCode((prev) =>
                              prev === seriesItem.exerciseCode ? null : seriesItem.exerciseCode,
                            )
                          }
                          className={`btn-tactile inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[10px] font-bold cursor-pointer transition-all ${
                            isFocused
                              ? 'border-slate-900 bg-white text-slate-900 shadow-xs ring-2 ring-sky-200'
                              : isDimmed
                                ? 'border-slate-200 bg-white/60 text-slate-400 opacity-60 hover:opacity-100'
                                : 'border-slate-200 bg-white text-slate-700 hover:border-sky-300'
                          }`}
                        >
                          <span
                            className="h-2.5 w-2.5 rounded-full shrink-0"
                            style={{ backgroundColor: seriesItem.color }}
                          />
                          <span className="max-w-[190px] truncate">{seriesItem.exerciseNameTh}</span>
                          <span className="text-[9.5px] font-extrabold text-slate-500">
                            ({seriesItem.latestRomDeg.toFixed(1)}° • {seriesItem.latestScore0To6}/6 • {seriesItem.sessionCount} ครั้ง)
                          </span>
                        </button>
                      );
                    })}
                  </div>
                )}

                {summary.hasMixedExercises &&
                  chartMetricMode === 'ROM_DEG' &&
                  legendFocusExerciseCode === null && (
                    <div className="flex flex-wrap items-center justify-between gap-2 rounded-xl border border-sky-200 bg-sky-50/70 px-3 py-2 text-[11px] font-semibold text-[#0F4C81]">
                      <span>
                        กำลังแสดงเส้นพัฒนาการแยกสีครบทั้ง {multiExerciseGrouping.series.length} ท่ากายภาพ สามารถสลับเป็น &quot;% เป้าหมาย&quot; เพื่อเทียบสัดส่วนความสำเร็จระหว่างท่าได้
                      </span>
                      <button
                        type="button"
                        onClick={() => setChartMetricMode('TARGET_PCT')}
                        className="rounded-lg bg-[#0F4C81] px-2.5 py-1 text-[10px] font-extrabold text-white cursor-pointer"
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
                    <div className="w-full overflow-hidden">
                      <svg
                        viewBox={`0 0 ${chartData.width} ${chartData.height}`}
                        className="w-full h-auto"
                        role="img"
                        aria-label="กราฟเปรียบเทียบพัฒนาการมุมองศาข้อต่อแยกตามท่ากายภาพและคะแนน 0-6 พร้อมวันเดือนปี"
                      >
                        <defs>
                          <linearGradient id="romAreaGrad" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stopColor="#10b981" stopOpacity="0.22" />
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

                        {/* Separate Color-Coded Area & Polyline Per Physical Therapy Exercise */}
                        {chartData.exerciseSeriesPaths.map((seriesPath) => {
                          const isFocusedSeries =
                            legendFocusExerciseCode === null ||
                            legendFocusExerciseCode === seriesPath.exerciseCode;
                          const showSoftArea =
                            seriesPath.areaPoints &&
                            (!chartData.isMultiExerciseView ||
                              legendFocusExerciseCode === seriesPath.exerciseCode);
                          return (
                            <g
                              key={seriesPath.exerciseCode}
                              data-testid={`exercise-series-${seriesPath.exerciseCode}`}
                              opacity={isFocusedSeries ? 1 : 0.16}
                            >
                              {showSoftArea && (
                                <polygon
                                  points={seriesPath.areaPoints}
                                  fill={seriesPath.color}
                                  fillOpacity="0.12"
                                />
                              )}
                              {seriesPath.nodes.length >= 2 && (
                                <polyline
                                  data-testid={`exercise-polyline-${seriesPath.exerciseCode}`}
                                  fill="none"
                                  stroke={seriesPath.color}
                                  strokeWidth={isFocusedSeries ? '3' : '2'}
                                  strokeLinecap="round"
                                  strokeLinejoin="round"
                                  points={seriesPath.polylinePoints}
                                />
                              )}
                            </g>
                          );
                        })}

                        {/* Secondary Mahidol 0-6 Trend Line */}
                        {showMahidolLine && chartData.scorePolylinePoints && (
                          <polyline
                            fill="none"
                            stroke="#0284c7"
                            strokeWidth="1.8"
                            strokeDasharray="5 4"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            opacity="0.72"
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

                        {/* Interactive Nodes with Color-Coding, Milestone Pulse Ring & Full Thai Short Date */}
                        {chartData.nodes.map((node) => {
                          const isSelected = activeSelectedNode?.id === node.id;
                          const isDimmed =
                            legendFocusExerciseCode !== null &&
                            node.exerciseCode !== legendFocusExerciseCode;
                          const labelAboveY = node.y < chartData.padT + 20 ? node.y + 20 : node.y - 12;
                          return (
                            <g
                              key={node.id}
                              onClick={() => setSelectedNodeId(node.id)}
                              className="cursor-pointer"
                              opacity={isDimmed ? 0.2 : 1}
                            >
                              {isSelected && (
                                <>
                                  <line
                                    x1={node.x}
                                    y1={chartData.padT}
                                    x2={node.x}
                                    y2={chartData.padT + chartData.plotH}
                                    stroke={node.exerciseColor}
                                    strokeDasharray="2 3"
                                    strokeWidth="1.3"
                                    opacity="0.65"
                                  />
                                  <circle
                                    cx={node.x}
                                    cy={node.y}
                                    r="13"
                                    fill={node.exerciseColor}
                                    fillOpacity="0.18"
                                    stroke={node.exerciseColor}
                                    strokeWidth="1.4"
                                  />
                                </>
                              )}
                              {showMahidolLine && (
                                <circle
                                  cx={node.x}
                                  cy={node.scoreY}
                                  r="3.5"
                                  fill="#0284c7"
                                  stroke="#ffffff"
                                  strokeWidth="1.4"
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
                                r={isSelected ? '7.5' : '5.6'}
                                fill={isSelected ? node.exerciseColor : '#ffffff'}
                                stroke={node.exerciseColor}
                                strokeWidth="3"
                              />
                              <text
                                x={node.x}
                                y={labelAboveY}
                                textAnchor="middle"
                                fill={node.exerciseColor}
                                fontSize="10"
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
                                y={chartData.padT + chartData.plotH + 35}
                                textAnchor="middle"
                                fill="#334155"
                                fontSize="9.5"
                                fontWeight="700"
                              >
                                ครั้งที่ {node.visitNumber}
                              </text>
                              <text
                                x={node.x}
                                y={chartData.padT + chartData.plotH + 49}
                                textAnchor="middle"
                                fill="#475569"
                                fontSize="8.5"
                                fontWeight="700"
                              >
                                {node.shortDate}
                              </text>
                            </g>
                          );
                        })}
                      </svg>
                    </div>

                    {/* Selected Chart Visit Inspector with Full Thai Date & Milestone Badges */}
                    {activeSelectedNode && (
                      <div
                        data-testid="selected-chart-visit-inspector"
                        className="rounded-2xl border border-emerald-200 bg-emerald-50/45 px-4 py-3 text-xs flex flex-wrap items-center justify-between gap-2 animate-toast-in"
                      >
                        <div className="space-y-0.5 min-w-0">
                          <div className="flex flex-wrap items-center gap-1.5 font-black text-slate-900">
                            <span
                              className="h-2.5 w-2.5 rounded-full shrink-0"
                              style={{ backgroundColor: activeSelectedNode.exerciseColor }}
                            />
                            <span>
                              รายละเอียดครั้งที่ {activeSelectedNode.visitNumber} •{' '}
                              {activeSelectedNode.fullThaiDate}
                            </span>
                            {activeSelectedNode.isMilestoneBest && (
                              <span className="rounded-full bg-emerald-600 px-2 py-0.5 text-[10px] font-extrabold text-white">
                                ครั้งที่ดีที่สุด
                              </span>
                            )}
                            {activeSelectedNode.isMilestoneLatest && (
                              <span className="rounded-full bg-[#0F4C81] px-2 py-0.5 text-[10px] font-extrabold text-white">
                                ครั้งล่าสุด
                              </span>
                            )}
                            {activeSelectedNode.isMilestonePrevious && (
                              <span className="rounded-full bg-slate-700 px-2 py-0.5 text-[10px] font-extrabold text-white">
                                ครั้งก่อนหน้า
                              </span>
                            )}
                          </div>
                          <div className="text-[11px] text-slate-600">
                            {activeSelectedNode.exerciseNameTh} (ข้าง
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
              </>
            )}

            {/* Dedicated Monthly Score Summary Chart (Tab View OR Always-Accessible Card Below) */}
            {chartViewMode === 'MONTHLY' && (
              <div data-testid="monthly-score-summary-chart" className="space-y-3">
                {monthlyChartData.bars.length === 0 ? (
                  <div className="rounded-2xl border border-dashed border-slate-200 bg-slate-50/70 py-10 px-4 text-center text-xs text-slate-500">
                    ยังไม่มีข้อมูลคะแนนประเมินรายเดือนสำหรับผู้ป่วยรายนี้
                  </div>
                ) : (
                  <div className="w-full overflow-hidden">
                    <svg
                      viewBox={`0 0 ${monthlyChartData.width} ${monthlyChartData.height}`}
                      className="w-full h-auto"
                      role="img"
                      aria-label="กราฟสรุปรายเดือนแสดงคะแนนมหิดลเฉลี่ย 0-6 ต่อเดือน"
                    >
                      <defs>
                        <linearGradient id="monthlyScoreBarGrad" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="#059669" stopOpacity="0.92" />
                          <stop offset="100%" stopColor="#0F4C81" stopOpacity="0.88" />
                        </linearGradient>
                        <linearGradient id="monthlyScoreBarSelected" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="#10b981" stopOpacity="1" />
                          <stop offset="100%" stopColor="#0284c7" stopOpacity="1" />
                        </linearGradient>
                      </defs>

                      {/* 0 to 6 Mahidol Y-Axis Gridlines */}
                      {[0, 1, 2, 3, 4, 5, 6].map((scoreTick) => {
                        const yVal =
                          monthlyChartData.padT +
                          monthlyChartData.plotH * (1 - scoreTick / 6);
                        return (
                          <g key={scoreTick}>
                            <line
                              x1={monthlyChartData.padL}
                              y1={yVal}
                              x2={monthlyChartData.width - monthlyChartData.padR}
                              y2={yVal}
                              stroke={scoreTick === 0 ? '#94a3b8' : '#e2e8f0'}
                              strokeDasharray={scoreTick === 0 ? undefined : '3 3'}
                              strokeWidth="1"
                            />
                            <text
                              x={monthlyChartData.padL - 10}
                              y={yVal + 3.5}
                              textAnchor="end"
                              fill="#475569"
                              fontSize="10"
                              fontWeight="700"
                            >
                              {scoreTick}/6
                            </text>
                          </g>
                        );
                      })}

                      {/* Monthly 0-6 Score Bars */}
                      {monthlyChartData.bars.map((bar) => {
                        const isSelectedMonth = selectedMonthKey === bar.monthKey;
                        return (
                          <g
                            key={bar.monthKey}
                            data-testid={`monthly-bar-${bar.monthKey}`}
                            onClick={() =>
                              setSelectedMonthKey((prev) =>
                                prev === bar.monthKey ? null : bar.monthKey,
                              )
                            }
                            className="cursor-pointer"
                          >
                            <rect
                              x={bar.x}
                              y={bar.y}
                              width={bar.barW}
                              height={bar.barH}
                              rx="8"
                              fill={
                                isSelectedMonth
                                  ? 'url(#monthlyScoreBarSelected)'
                                  : 'url(#monthlyScoreBarGrad)'
                              }
                              stroke={isSelectedMonth ? '#0F4C81' : 'transparent'}
                              strokeWidth={isSelectedMonth ? '2.5' : '0'}
                            />
                            <text
                              x={bar.cx}
                              y={Math.max(14, bar.y - 8)}
                              textAnchor="middle"
                              fill="#065f46"
                              fontSize="10.5"
                              fontWeight="900"
                            >
                              {bar.avgScore0To6.toFixed(1)} / 6 คะแนน
                            </text>
                            <text
                              x={bar.cx}
                              y={monthlyChartData.padT + monthlyChartData.plotH + 17}
                              textAnchor="middle"
                              fill={isSelectedMonth ? '#0F4C81' : '#1e293b'}
                              fontSize="10"
                              fontWeight="800"
                            >
                              {bar.monthLabelShortTh}
                            </text>
                            <text
                              x={bar.cx}
                              y={monthlyChartData.padT + monthlyChartData.plotH + 31}
                              textAnchor="middle"
                              fill="#64748b"
                              fontSize="9"
                              fontWeight="700"
                            >
                              {bar.sessionCount} ครั้ง • สูงสุด {bar.maxScore0To6}/6
                            </text>
                            {bar.monthDeltaScore !== null && (
                              <text
                                x={bar.cx}
                                y={monthlyChartData.padT + monthlyChartData.plotH + 45}
                                textAnchor="middle"
                                fill={bar.monthDeltaScore >= 0 ? '#059669' : '#e11d48'}
                                fontSize="9"
                                fontWeight="800"
                              >
                                {bar.monthDeltaScore >= 0 ? '+' : ''}
                                {bar.monthDeltaScore.toFixed(1)} คะแนน
                              </text>
                            )}
                          </g>
                        );
                      })}

                      {/* Smooth Emerald Trend Line Connecting Monthly Averages */}
                      {monthlyChartData.bars.length >= 2 && (
                        <polyline
                          fill="none"
                          stroke="#059669"
                          strokeWidth="2.6"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          points={monthlyChartData.trendPoints}
                        />
                      )}
                      {monthlyChartData.bars.map((bar) => (
                        <circle
                          key={`dot-${bar.monthKey}`}
                          cx={bar.cx}
                          cy={bar.y}
                          r="4.5"
                          fill="#ffffff"
                          stroke="#059669"
                          strokeWidth="2.5"
                          pointerEvents="none"
                        />
                      ))}
                    </svg>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Always-Accessible Monthly Score Summary Breakdown Card */}
          {monthlySummary.length > 0 && (
            <div
              data-testid="monthly-score-summary-card"
              className="rounded-3xl border border-emerald-200/80 bg-white p-5 shadow-sm space-y-3"
            >
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div>
                  <h3 className="text-sm font-black text-slate-900">
                    สรุปผลคะแนนรายเดือน (Monthly Score Summary — {monthlySummary.length} เดือน)
                  </h3>
                  <p className="text-[11px] text-slate-500">
                    คลิกเลือกเดือนเพื่อกรองประวัติการรักษารายบุคคลด้านล่าง หรือเปิดดูกราฟแท่งรายเดือนเต็มรูปแบบ
                  </p>
                </div>
                <div className="flex flex-wrap items-center gap-1.5">
                  {selectedMonthKey && (
                    <button
                      type="button"
                      onClick={() => setSelectedMonthKey(null)}
                      className="rounded-lg border border-slate-200 bg-slate-50 px-2.5 py-1 text-[10px] font-bold text-slate-700 hover:bg-slate-100 cursor-pointer"
                    >
                      แสดงทุกเดือน
                    </button>
                  )}
                  <button
                    type="button"
                    onClick={() =>
                      setChartViewMode((prev) => (prev === 'MONTHLY' ? 'SESSIONS' : 'MONTHLY'))
                    }
                    className="btn-tactile rounded-lg border border-emerald-300 bg-emerald-50 px-2.5 py-1 text-[10px] font-extrabold text-emerald-800 hover:bg-emerald-100 cursor-pointer"
                  >
                    {chartViewMode === 'MONTHLY'
                      ? 'กลับสู่กราฟรายครั้ง'
                      : 'เปิดกราฟแท่งรายเดือน'}
                  </button>
                </div>
              </div>

              <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-2 lg:grid-cols-3">
                {monthlySummary.map((m) => {
                  const isSelectedMonth = selectedMonthKey === m.monthKey;
                  return (
                    <button
                      key={m.monthKey}
                      type="button"
                      onClick={() =>
                        setSelectedMonthKey((prev) => (prev === m.monthKey ? null : m.monthKey))
                      }
                      className={`btn-tactile text-left rounded-2xl border p-3 transition-all space-y-1.5 cursor-pointer ${
                        isSelectedMonth
                          ? 'border-emerald-500 bg-emerald-50/70 ring-2 ring-emerald-200'
                          : 'border-slate-200 bg-slate-50/60 hover:border-sky-300 hover:bg-white'
                      }`}
                    >
                      <div className="flex items-center justify-between gap-1.5">
                        <span className="text-xs font-black text-slate-900">
                          {m.monthLabelFullTh}
                        </span>
                        <span className="rounded-full bg-[#0F4C81] px-2 py-0.5 text-[10px] font-extrabold text-white">
                          {m.avgScore0To6.toFixed(1)}/6
                        </span>
                      </div>
                      <div className="h-1.5 w-full overflow-hidden rounded-full bg-slate-200">
                        <div
                          className="h-full rounded-full bg-gradient-to-r from-[#0F4C81] to-[#10B981]"
                          style={{ width: `${Math.min(100, (m.avgScore0To6 / 6) * 100)}%` }}
                        />
                      </div>
                      <div className="flex flex-wrap items-center justify-between gap-1 text-[10px] font-semibold text-slate-600">
                        <span>
                          {m.sessionCount} ครั้ง • เป้าหมาย {m.avgTargetPct.toFixed(0)}%
                        </span>
                        <span>ROM เฉลี่ย {m.avgRomDeg.toFixed(1)}°</span>
                      </div>
                      <div className="flex flex-wrap items-center justify-between gap-1 text-[10px]">
                        <span className="font-bold text-emerald-700">
                          สูงสุด {m.maxScore0To6}/6
                        </span>
                        {m.monthDeltaScore !== null ? (
                          <span
                            className={`rounded-full px-1.5 py-0.5 font-extrabold ${
                              m.monthDeltaScore >= 0
                                ? 'bg-emerald-100 text-emerald-800'
                                : 'bg-rose-100 text-rose-700'
                            }`}
                          >
                            {m.monthDeltaScore >= 0 ? '+' : ''}
                            {m.monthDeltaScore.toFixed(1)} คะแนน
                          </span>
                        ) : (
                          <span className="text-slate-400 font-semibold">เดือนตั้งต้น</span>
                        )}
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>
          )}

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
              <div className="flex flex-wrap items-center gap-2">
                <History className="h-4 w-4 text-[#0F4C81]" />
                <h3 className="text-sm font-black text-slate-900">
                  ประวัติการรักษารายบุคคลเรียงตามลำดับเวลา ({timelineVisits.length} ครั้ง)
                </h3>
                {selectedMonthKey && (
                  <button
                    type="button"
                    onClick={() => setSelectedMonthKey(null)}
                    className="inline-flex items-center gap-1 rounded-full border border-emerald-300 bg-emerald-50 px-2.5 py-0.5 text-[10px] font-extrabold text-emerald-800 hover:bg-emerald-100 cursor-pointer"
                  >
                    <span>เดือน {formatThaiMonthYear(selectedMonthKey, 'full')}</span>
                    <span className="text-emerald-600">• แสดงทุกเดือน</span>
                  </button>
                )}
              </div>
              <span className="text-[11px] font-semibold text-slate-500">
                แสดงเฉพาะแฟ้มของ {selectedPatient?.fullName ?? '-'}
              </span>
            </div>

            {timelineVisits.length === 0 ? (
              <div className="rounded-2xl border border-dashed border-slate-200 bg-slate-50/60 py-8 px-4 text-center text-xs text-slate-500">
                ยังไม่มีประวัติการประเมินในเงื่อนไขที่เลือก
              </div>
            ) : (
              <div className="space-y-3 max-h-[540px] overflow-y-auto pr-1">
                {timelineVisits
                  .slice()
                  .reverse()
                  .map(({ record, deltaFromPrevRom, targetAchievementPct }) => {
                    const isCourseAuto = record.source_channel === 'COURSE_AUTO_CALCULATED';
                    const isKiosk = record.source_channel === 'KIOSK_SESSION_AUTO_SYNC';
                    const isCameraAi = record.measurement_method === 'CAMERA_AI_LIVE';
                    const isSelectedCard = activeSelectedNode?.id === record.id;
                    const exerciseDotColor =
                      multiExerciseGrouping.colorByExerciseCode[
                        (record.exercise_code || '').toUpperCase()
                      ] ?? '#059669';
                    const canDelete =
                      !record.id.startsWith('kiosk-sess-') && !record.id.startsWith('hist-sess-');
                    return (
                      <div
                        key={record.id}
                        className={`rounded-2xl border p-4 space-y-2 transition-colors card-hover-lift animate-card-enter ${
                          isSelectedCard
                            ? 'border-emerald-400 bg-emerald-50/35 ring-1 ring-emerald-200'
                            : 'border-slate-200 bg-slate-50/60 hover:border-sky-300'
                        }`}
                      >
                        <div className="flex flex-wrap items-center justify-between gap-2">
                          <div className="flex flex-wrap items-center gap-1.5">
                            <span
                              className="h-2.5 w-2.5 rounded-full shrink-0"
                              style={{ backgroundColor: exerciseDotColor }}
                            />
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
