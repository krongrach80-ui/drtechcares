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
  ArrowDownRight,
  ArrowUpRight,
  Award,
  BarChart3,
  Brain,
  Camera,
  Check,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  ChevronUp,
  ClipboardList,
  Copy,
  Download,
  FileText,
  Filter,
  Gamepad2,
  GitCompare,
  History,
  Link2,
  Minus,
  Printer,
  RefreshCw,
  Search,
  Sparkles,
  Table2,
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
import { buildDoctorClinicalBrief } from '../utils/doctorClinicalBrief';
import {
  buildPatientProgressSummaryReport,
  buildUnifiedRehabMinigameSummary,
  isMinigameRehabSession,
  resolveExerciseRowVerdict,
  type ProgressVerdictTone,
} from '../utils/unifiedRehabMinigameSummary';
import { ClinicalPdfReportModal } from './ClinicalPdfReportModal';
import { ClinicalRomLiveMeasureModal } from './ClinicalRomLiveMeasureModal';
import { DrTechCareMascot } from './DrTechCareMascot';
import { PatientProgressSummaryModal } from './PatientProgressSummaryModal';
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
type ChartViewMode = 'SESSIONS' | 'COMPARE';
type WorkspaceChartTab = 'STOCK' | 'ROM' | 'MINIGAME' | 'MONTHLY' | 'TABLE' | 'TIMELINE';
type TimelineKindFilter = 'ALL' | 'PHYSIO' | 'MINIGAME';

const WORKSPACE_CHART_TABS: readonly WorkspaceChartTab[] = [
  'STOCK',
  'ROM',
  'MINIGAME',
  'MONTHLY',
  'TABLE',
  'TIMELINE',
];

function verdictBadgeClasses(tone: ProgressVerdictTone): string {
  switch (tone) {
    case 'emerald':
      return 'border-emerald-300 bg-emerald-50 text-emerald-800';
    case 'sky':
      return 'border-sky-300 bg-sky-50 text-[#0F4C81]';
    case 'amber':
      return 'border-amber-300 bg-amber-50 text-amber-900';
    case 'indigo':
      return 'border-indigo-300 bg-indigo-50 text-indigo-800';
    default:
      return 'border-slate-200 bg-slate-50 text-slate-600';
  }
}

function verdictBannerClasses(tone: ProgressVerdictTone): string {
  switch (tone) {
    case 'emerald':
      return 'border-emerald-300 bg-gradient-to-r from-emerald-50 via-teal-50/70 to-white';
    case 'sky':
      return 'border-sky-300 bg-gradient-to-r from-sky-50 via-white to-emerald-50/40';
    case 'amber':
      return 'border-amber-300 bg-gradient-to-r from-amber-50 via-orange-50/50 to-white';
    case 'indigo':
      return 'border-indigo-300 bg-gradient-to-r from-indigo-50 via-sky-50/60 to-white';
    default:
      return 'border-slate-200 bg-slate-50/70';
  }
}

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

const FRIENDLY_FALLBACK_EXERCISE_NAMES: Record<string, string> = {
  ELDERLY_SHOULDER_FLEXION: 'ยกแขนไปด้านหน้า',
  ELDERLY_SHOULDER_ABDUCTION: 'กางแขนด้านข้าง',
  ELDERLY_ELBOW_FLEXION: 'งอ-เหยียดข้อศอก',
  ELDERLY_WRIST_FLEXION: 'กระดกข้อมือและนิ้วมือ',
  ELDERLY_SHOULDER_EXTERNAL_ROTATION: 'หมุนข้อไหล่เข้า-ออก',
  ELDERLY_OVERHEAD_REACH: 'ประสานมือชูขึ้นเหนือศีรษะ',
};

function formatFriendlyExerciseLabel(rawName?: string | null, rawCode?: string | null): string {
  const cleanRaw = String(rawName ?? '').trim();
  const isRawCodeLike = !cleanRaw || /^[A-Z0-9_]+$/.test(cleanRaw);
  let resolved = cleanRaw;
  if (isRawCodeLike || cleanRaw.includes('_')) {
    const parsed = parseExerciseCodeAndSide(rawCode || cleanRaw);
    const upperBase = parsed.baseCode.toUpperCase();
    const found = CLINICAL_EXERCISES.find((ex) => ex.code.toUpperCase() === upperBase);
    if (found) {
      resolved = found.name.th;
    } else if (FRIENDLY_FALLBACK_EXERCISE_NAMES[upperBase]) {
      resolved = FRIENDLY_FALLBACK_EXERCISE_NAMES[upperBase];
    }
  }
  return resolved.replace(/^\[[^\]]+\]\s*/g, '').trim() || String(rawCode || 'ท่ากายภาพบำบัด');
}

function formatFriendlyDiagnosis(raw?: string | null): string {
  const clean = String(raw ?? '').trim();
  if (!clean) return 'ดูแลและฟื้นฟูสมรรถภาพผู้สูงอายุ';
  return clean.replace(/\s*\(Shoulder & Arm\)/gi, '').replace(/\bADL\b/gi, 'ผู้สูงอายุ').trim();
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
  const [primaryPanelMode, setPrimaryPanelMode] = useState<'DASHBOARD' | 'ASSESS_AND_COURSE'>('DASHBOARD');
  const [activeWorkspaceTab, setActiveWorkspaceTab] = useState<WorkspaceChartTab>('STOCK');
  const [showFilterBar, setShowFilterBar] = useState<boolean>(false);
  const [coursePage, setCoursePage] = useState<number>(0);
  const [coursePerPage, setCoursePerPage] = useState<number>(5);
  const [monthlyPage, setMonthlyPage] = useState<number>(0);
  const [tablePage, setTablePage] = useState<number>(0);
  const [timelinePage, setTimelinePage] = useState<number>(0);
  const courseListBoxRef = useRef<HTMLDivElement | null>(null);
  const [timelineKindFilter, setTimelineKindFilter] = useState<TimelineKindFilter>('ALL');
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
    CLINICAL_EXERCISES[0]?.code || 'ELDERLY_SHOULDER_FLEXION',
  );
  const [targetSide, setTargetSide] = useState<AssessmentTargetSide>('RIGHT');
  const [measuredRomInput, setMeasuredRomInput] = useState<string>('');
  const [targetRomInput, setTargetRomInput] = useState<string>(
    String(
      CLINICAL_EXERCISES[0]?.detection.targetRomDeg ??
        CLINICAL_EXERCISES[0]?.detection.targetMinDeg ??
        180,
    ),
  );
  const [painVasInput, setPainVasInput] = useState<string>('');
  const [assessedAtInput, setAssessedAtInput] = useState<string>(() => toLocalDateTimeInputValue());
  const [measurementMethod, setMeasurementMethod] =
    useState<AssessmentMeasurementMethod>('MANUAL_INPUT');
  const [liveMeasureModalOpen, setLiveMeasureModalOpen] = useState<boolean>(false);
  const [patientSummaryModalOpen, setPatientSummaryModalOpen] = useState<boolean>(false);
  const [pdfReportModalOpen, setPdfReportModalOpen] = useState<boolean>(false);
  const [showExtraFormNotes, setShowExtraFormNotes] = useState<boolean>(false);
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
          const minigameCount = sessions.filter(
            (s) => s.patientId === targetPatientId && isMinigameRehabSession(s),
          ).length;
          showToast(
            'success',
            list.length > 0 || minigameCount > 0
              ? `ซิงก์ข้อมูลจริงสำเร็จ: กายภาพ ${list.length} ครั้ง • มินิเกม ${minigameCount} รอบ (${Math.max(1, facilitiesCount)} สถานที่/ตู้ Kiosk)`
              : 'ซิงก์ข้อมูลล่าสุดแล้ว: ผู้ป่วยรายนี้ยังไม่มีบันทึกกายภาพหรือประวัติเล่นมินิเกม',
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
                nextSessions = rows.map((r) => {
                  const exId = String(r.exercise_id ?? '');
                  const hasMinigameScore =
                    r.minigame_score !== null && r.minigame_score !== undefined;
                  const isMinigame =
                    hasMinigameScore || /minigame|cognitive-card-game|\bcard\b/i.test(exId);
                  return {
                    id: String(r.id ?? ''),
                    patientId: String(r.patient_id ?? targetPatientId),
                    activityType: isMinigame ? 'minigame' : 'physio_exercise',
                    exerciseId: exId,
                    exerciseName: isMinigame ? 'มินิเกมฝึกสมอง (การ์ดกายภาพ)' : exId,
                    completedSets: Number(r.completed_sets ?? 0),
                    completedReps: Number(r.completed_reps ?? 0),
                    accuracyPercent: Number(r.accuracy_percent ?? 85),
                    maxRomDeg: Number(r.max_rom_deg ?? 0),
                    painStopTriggered: Boolean(r.pain_stop_triggered),
                    aiFeedback: '',
                    kinematicsReplay: [],
                    minigameScore: hasMinigameScore
                      ? (r.minigame_score as RehabSessionRecord['minigameScore'])
                      : undefined,
                    createdAt: String(r.created_at ?? new Date().toISOString()),
                  };
                });
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
    setTimelineKindFilter('ALL');
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
      const friendlyName = formatFriendlyExerciseLabel(result.exerciseNameTh, result.exerciseCode);
      setChiefComplaint(
        `ประเมินอัตโนมัติจาก${result.courseTitle} ท่า ${friendlyName} (${result.completedReps}/${result.targetReps} ครั้ง)`,
      );
      setHandoverNote(result.breakdownSummaryTh);
      const nextTargetRom =
        result.effectiveRomDeg >= result.targetRomDeg
          ? Math.round(result.targetRomDeg)
          : Math.min(Math.round(result.targetRomDeg), Math.round(result.effectiveRomDeg + 10));
      setNextGoal(
        result.calculatedScore0To6 >= 6
          ? `คงความสมบูรณ์การเคลื่อนไหวท่า ${friendlyName} และฝึกตามคอร์สต่อเนื่อง`
          : `เพิ่มมุมการเคลื่อนไหวท่า ${friendlyName} สู่ ${nextTargetRom}° และยกระดับคะแนนเป็น ${Math.min(6, result.calculatedScore0To6 + 1)}/6`,
      );
      setActiveAutoCalcResult(result);
      if (showFeedback) {
        showToast(
          'info',
          `นำผลคำนวณอัตโนมัติจากคอร์ส (${friendlyName}: ระดับ ${result.calculatedScore0To6}/6, ROM ${result.effectiveRomDeg.toFixed(1)}°) กรอกลงแบบฟอร์มแล้ว`,
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
      setTargetRomInput(String(matched.detection.targetRomDeg ?? matched.detection.targetMinDeg));
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

  // Merge saved assessments with unsaved completed Kiosk course sessions so KPIs, Verdict,
  // Milestones, Charts, and Timeline ALWAYS match the live Rehab Index (zero contradictory "0 ครั้ง").
  const effectiveRecords = useMemo<PatientClinicalAssessmentRecord[]>(() => {
    if (!selectedPatient) return records;
    const pending = courseAutoBundle?.unrecordedResults ?? [];
    if (pending.length === 0) return records;
    const baseVisitCount = records.length;
    const synthesized: PatientClinicalAssessmentRecord[] = pending.map((res, idx) => ({
      id: `kiosk-sess-${res.sourceSessionId}`,
      patient_id: selectedPatient.id,
      patient_hn: selectedPatient.hn,
      universal_link_code: universalLinkCode,
      visit_number: baseVisitCount + idx + 1,
      assessed_at: res.assessedAt,
      facility_name: res.facilityName || facilityName,
      assessor_staff_id: currentUser.id,
      assessor_staff_name: 'ซิงก์อัตโนมัติจากตู้ Kiosk',
      exercise_code: res.exerciseCode,
      exercise_name_th: formatFriendlyExerciseLabel(res.exerciseNameTh, res.exerciseCode),
      target_side: res.targetSide,
      initial_score_0_to_6: res.calculatedScore0To6,
      measured_rom_deg: res.effectiveRomDeg,
      target_rom_deg: res.targetRomDeg,
      pain_vas_0_to_10: null,
      chief_complaint: `ฝึกจากตู้ Kiosk (${res.completedReps}/${res.targetReps} ครั้ง)`,
      clinical_handover_note: res.breakdownSummaryTh,
      next_session_goal: '',
      source_channel: 'KIOSK_SESSION_AUTO_SYNC',
      measurement_method: 'COURSE_AUTO_CALCULATED',
      created_at: res.assessedAt,
    }));
    return [...records, ...synthesized].sort(
      (a, b) => new Date(a.assessed_at).getTime() - new Date(b.assessed_at).getTime(),
    );
  }, [
    selectedPatient,
    records,
    courseAutoBundle?.unrecordedResults,
    universalLinkCode,
    facilityName,
    currentUser.id,
  ]);

  const exerciseOptionsInHistory = useMemo(() => {
    const map = new Map<string, string>();
    for (const r of effectiveRecords) {
      const code = (r.exercise_code || '').toUpperCase();
      if (code && !map.has(code)) {
        map.set(code, formatFriendlyExerciseLabel(r.exercise_name_th, code));
      }
    }
    return Array.from(map.entries());
  }, [effectiveRecords]);

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

    return effectiveRecords.filter((rec) => {
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
  }, [effectiveRecords, exerciseFilter, sideFilter, dateRangeFilter, sourceFilter]);

  const summary = useMemo(
    () => computeLongitudinalProgressSummary(filteredRecords, exerciseFilter),
    [filteredRecords, exerciseFilter],
  );

  // Full-patient per-exercise breakdown (always visible even when filtering 1 exercise, filtered by date/source)
  const overallBreakdown = useMemo(() => {
    const base = effectiveRecords.filter((rec) => {
      if (sourceFilter !== 'ALL' && rec.source_channel !== sourceFilter) return false;
      if (dateRangeFilter !== 'ALL') {
        const days = dateRangeFilter === '7D' ? 7 : dateRangeFilter === '30D' ? 30 : 90;
        const ts = new Date(rec.assessed_at).getTime();
        if (Number.isFinite(ts) && ts < Date.now() - days * 86_400_000) return false;
      }
      return true;
    });
    return computeLongitudinalProgressSummary(base, 'ALL').exerciseBreakdown;
  }, [effectiveRecords, sourceFilter, dateRangeFilter]);

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

  // Unified Synchronized Rehab + Minigame Clinical Progress & Encouragement Summary
  const unifiedSummary = useMemo(
    () =>
      buildUnifiedRehabMinigameSummary({
        patientId: selectedPatient?.id ?? '',
        longitudinalSummary: summary,
        stockIndex: rehabStockIndex,
        sessions: mergedPatientSessions,
      }),
    [selectedPatient?.id, summary, rehabStockIndex, mergedPatientSessions],
  );

  const patientSummaryReport = useMemo(
    () =>
      selectedPatient
        ? buildPatientProgressSummaryReport({
            patientName: selectedPatient.fullName,
            patientHn: selectedPatient.hn,
            unifiedSummary,
            assessments: effectiveRecords,
          })
        : null,
    [selectedPatient, unifiedSummary, effectiveRecords],
  );

  // Instant Doctor-Readable Clinical Executive Brief (Detailed yet Scannable in 5 Seconds)
  const doctorBrief = useMemo(
    () =>
      buildDoctorClinicalBrief({
        stockIndex: rehabStockIndex,
        longitudinalSummary: summary,
        unifiedSummary,
        exerciseBreakdown: overallBreakdown,
        assessments: effectiveRecords,
      }),
    [rehabStockIndex, summary, unifiedSummary, overallBreakdown, effectiveRecords],
  );

  // 60FPS Pure SVG Minigame Cognitive Accuracy & Score Trend Chart Geometry
  const minigameChartData = useMemo(() => {
    const items = unifiedSummary.minigame.sessions.slice(-12);
    const width = Math.max(640, items.length > 7 ? items.length * 84 + 96 : 640);
    const height = 228;
    const padL = 48;
    const padR = 26;
    const padT = 28;
    const padB = 54;
    const plotW = width - padL - padR;
    const plotH = height - padT - padB;
    const slotW = items.length > 0 ? plotW / items.length : plotW;
    const barW = Math.min(44, Math.max(20, slotW * 0.44));

    const bars = items.map((s, idx) => {
      const cx = padL + slotW * idx + slotW / 2;
      const barH = Math.max(4, (Math.max(0, Math.min(100, s.accuracyPct)) / 100) * plotH);
      const y = padT + plotH - barH;
      return {
        ...s,
        cx,
        x: cx - barW / 2,
        y,
        barW,
        barH,
      };
    });

    const trendPoints = bars.map((b) => `${b.cx.toFixed(1)},${b.y.toFixed(1)}`).join(' ');
    const target80Y = padT + plotH * 0.2;
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
      target80Y,
    };
  }, [unifiedSummary.minigame.sessions]);

  const filteredUnifiedTimeline = useMemo(() => {
    return unifiedSummary.timeline.filter((item) => {
      if (timelineKindFilter === 'PHYSIO' && item.kind !== 'physio') return false;
      if (timelineKindFilter === 'MINIGAME' && item.kind !== 'minigame') return false;
      if (selectedMonthKey && getCalendarMonthKey(item.timestampIso) !== selectedMonthKey) {
        return false;
      }
      return true;
    });
  }, [unifiedSummary.timeline, timelineKindFilter, selectedMonthKey]);

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
    setActiveWorkspaceTab('ROM');
    setChartViewMode('SESSIONS');
    setLegendFocusExerciseCode(null);
    setSelectedNodeId(recordId);
  }, []);

  const activeWorkspaceTabIndex = WORKSPACE_CHART_TABS.indexOf(activeWorkspaceTab);
  const handleStepWorkspaceTab = useCallback((delta: -1 | 1) => {
    setActiveWorkspaceTab((prev) => {
      const idx = WORKSPACE_CHART_TABS.indexOf(prev);
      const nextIdx = (idx + delta + WORKSPACE_CHART_TABS.length) % WORKSPACE_CHART_TABS.length;
      return WORKSPACE_CHART_TABS[nextIdx];
    });
  }, []);

  const activeFilterCount =
    (exerciseFilter !== 'ALL' ? 1 : 0) +
    (sideFilter !== 'ALL' ? 1 : 0) +
    (dateRangeFilter !== 'ALL' ? 1 : 0) +
    (sourceFilter !== 'ALL' ? 1 : 0);

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

    const width = Math.max(580, pts.length > 8 ? pts.length * 68 + 110 : 580);
    const height = 248;
    const padL = 56;
    const padR = 48;
    const padT = 28;
    const padB = 56;
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
    const width = Math.max(640, items.length > 6 ? items.length * 90 + 96 : 640);
    const height = 212;
    const padL = 48;
    const padR = 26;
    const padT = 26;
    const padB = 52;
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

  useEffect(() => {
    if (typeof ResizeObserver === 'undefined' || !courseListBoxRef.current) return;
    const recalc = () => {
      const h = courseListBoxRef.current?.clientHeight ?? 0;
      if (h > 40) {
        setCoursePerPage(Math.max(2, Math.min(12, Math.floor((h + 6) / 72))));
      }
    };
    recalc();
    const ro = new ResizeObserver(recalc);
    ro.observe(courseListBoxRef.current);
    return () => ro.disconnect();
  }, [selectedPatient?.id, courseAutoBundle?.exerciseSummaries.length]);

  useEffect(() => {
    setCoursePage(0);
    setMonthlyPage(0);
    setTablePage(0);
    setTimelinePage(0);
  }, [selectedPatient?.id, exerciseFilter, sideFilter, dateRangeFilter, sourceFilter, timelineKindFilter, selectedMonthKey]);

  const courseSummariesAll = courseAutoBundle?.exerciseSummaries ?? [];
  const courseTotalPages = Math.max(1, Math.ceil(courseSummariesAll.length / Math.max(1, coursePerPage)));
  const safeCoursePage = Math.min(coursePage, courseTotalPages - 1);
  const visibleCourseSummaries = courseSummariesAll.slice(
    safeCoursePage * coursePerPage,
    (safeCoursePage + 1) * coursePerPage,
  );

  const MONTHLY_PER_PAGE = 6;
  const monthlyTotalPages = Math.max(1, Math.ceil(monthlySummary.length / MONTHLY_PER_PAGE));
  const safeMonthlyPage = Math.min(monthlyPage, monthlyTotalPages - 1);
  const visibleMonthlyCards = monthlySummary.slice(
    safeMonthlyPage * MONTHLY_PER_PAGE,
    (safeMonthlyPage + 1) * MONTHLY_PER_PAGE,
  );

  const TABLE_PER_PAGE = 8;
  const tableTotalPages = Math.max(1, Math.ceil(overallBreakdown.length / TABLE_PER_PAGE));
  const safeTablePage = Math.min(tablePage, tableTotalPages - 1);
  const visibleBreakdownRows = overallBreakdown.slice(
    safeTablePage * TABLE_PER_PAGE,
    (safeTablePage + 1) * TABLE_PER_PAGE,
  );

  const TIMELINE_PER_PAGE = 6;
  const timelineTotalPages = Math.max(1, Math.ceil(filteredUnifiedTimeline.length / TIMELINE_PER_PAGE));
  const safeTimelinePage = Math.min(timelinePage, timelineTotalPages - 1);
  const visibleTimelineItems = filteredUnifiedTimeline.slice(
    safeTimelinePage * TIMELINE_PER_PAGE,
    (safeTimelinePage + 1) * TIMELINE_PER_PAGE,
  );

  if (patients.length === 0) {
    return (
      <div className="rounded-3xl border border-slate-200 bg-white p-10 text-center text-sm text-slate-500 shadow-sm animate-view-enter">
        ยังไม่มีรายชื่อผู้ป่วยในระบบ กรุณาลงทะเบียนผู้ป่วยใหม่ก่อนเปิดแฟ้มประเมิน 0-6 คะแนน
      </div>
    );
  }

  return (
    <div className="flex-1 min-h-0 h-full flex flex-col gap-2 overflow-hidden animate-view-enter">
      {toast && (
        <div
          className={`shrink-0 rounded-xl border px-3 py-1.5 text-xs font-bold shadow-xs animate-toast-in ${
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

      {/* SECTION A: Refined Executive Patient Selector, Identity Bar & Primary 2-Mode Switcher */}
      <div className="shrink-0 rounded-2xl border border-slate-200/90 bg-white p-3 shadow-[0_2px_12px_rgba(15,76,129,0.05)] space-y-2.5">
        {/* Row 1: Search + Patient Picker + Primary & Secondary Action Cluster */}
        <div className="flex flex-col gap-2 xl:flex-row xl:items-center xl:justify-between">
          <div className="flex items-center gap-2 flex-1 min-w-0">
            <div className="flex items-center rounded-xl border border-slate-200 bg-slate-50/80 px-2.5 h-9 w-36 sm:w-44 shrink-0 focus-within:border-[#0F4C81] focus-within:bg-white transition-colors">
              <Search className="h-3.5 w-3.5 text-slate-400 mr-1.5 shrink-0" />
              <input
                value={patientQuery}
                onChange={(e) => setPatientQuery(e.target.value)}
                placeholder="ค้นหาชื่อ, HN..."
                className="w-full bg-transparent text-xs sm:text-sm font-semibold text-slate-800 placeholder:text-slate-400 focus:outline-none"
              />
            </div>
            <select
              aria-label="เลือกผู้ป่วยรายบุคคล"
              value={selectedPatient?.id ?? ''}
              onChange={(e) => onSelectPatientId(e.target.value)}
              className="h-9 flex-1 min-w-0 rounded-xl border border-slate-200 bg-white px-3 text-xs sm:text-sm font-bold text-slate-900 focus:border-[#0F4C81] focus:outline-none shadow-2xs truncate"
            >
              {filteredPatients.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.fullName} ({p.hn}) — {resolvePatientUniversalLinkCode(p)}
                </option>
              ))}
            </select>
          </div>

          <div className="flex flex-wrap items-center justify-between sm:justify-end gap-1.5 shrink-0">
            <button
              type="button"
              data-testid="ehr-open-pdf-report-btn"
              disabled={!selectedPatient}
              onClick={() => setPdfReportModalOpen(true)}
              className="btn-tactile inline-flex items-center gap-1.5 rounded-xl bg-[#0F4C81] hover:bg-[#0c3f6b] px-3 h-9 text-xs font-extrabold text-white shadow-2xs disabled:opacity-45 cursor-pointer transition-colors"
              title="เปิดและพิมพ์ใบรายงานสรุปผลการฟื้นฟูระดับโรงพยาบาลพร้อม QR Code (PDF)"
            >
              <Printer className="h-3.5 w-3.5 shrink-0" />
              <span>พิมพ์ PDF</span>
            </button>
            <button
              type="button"
              data-testid="ehr-open-patient-summary-btn"
              disabled={!selectedPatient}
              onClick={() => setPatientSummaryModalOpen(true)}
              className="btn-tactile inline-flex items-center gap-1.5 rounded-xl bg-emerald-600 hover:bg-emerald-700 px-3 h-9 text-xs font-extrabold text-white shadow-2xs disabled:opacity-45 cursor-pointer transition-colors"
            >
              <Award className="h-3.5 w-3.5 shrink-0" />
              <span>สรุปผลคนไข้</span>
            </button>
            <div className="inline-flex items-center rounded-xl border border-slate-200 bg-slate-50/80 p-0.5 gap-0.5">
              <button
                type="button"
                disabled={!selectedPatient || syncingCrossFacility}
                onClick={() => {
                  void loadPatientHistory(selectedPatient, true);
                  void loadPatientCourseContext(selectedPatient);
                }}
                className="inline-flex items-center gap-1 rounded-lg px-2.5 h-8 text-xs font-bold text-slate-700 hover:bg-white hover:text-[#0F4C81] hover:shadow-2xs disabled:opacity-50 cursor-pointer transition-all"
              >
                <RefreshCw className={`h-3.5 w-3.5 ${syncingCrossFacility ? 'animate-spin' : ''}`} />
                <span>ซิงก์</span>
              </button>
              <button
                type="button"
                disabled={!selectedPatient || summary.totalVisits === 0}
                onClick={handleExportCsv}
                className="inline-flex items-center gap-1 rounded-lg px-2.5 h-8 text-xs font-bold text-slate-700 hover:bg-white hover:text-emerald-700 hover:shadow-2xs disabled:opacity-45 cursor-pointer transition-all"
              >
                <Download className="h-3.5 w-3.5" />
                <span>CSV</span>
              </button>
              {selectedPatient && onOpenUniversalEhrModal && (
                <button
                  type="button"
                  onClick={() => onOpenUniversalEhrModal(selectedPatient)}
                  className="inline-flex items-center gap-1 rounded-lg px-2.5 h-8 text-xs font-bold text-slate-700 hover:bg-white hover:text-indigo-700 hover:shadow-2xs cursor-pointer transition-all"
                >
                  <Link2 className="h-3.5 w-3.5" />
                  <span>EHR</span>
                </button>
              )}
            </div>
          </div>
        </div>

        {/* Row 2: Cohesive Patient Identity Strip + Sleek Segmented Mode Switcher */}
        <div className="flex flex-col gap-2 2xl:flex-row 2xl:items-center 2xl:justify-between border-t border-slate-100 pt-2">
          {selectedPatient && (
            <div className="flex-1 min-w-0 rounded-xl border border-slate-200/80 bg-slate-50/70 px-3 py-1.5 flex flex-wrap items-center justify-between gap-2">
              <div className="flex flex-wrap items-center gap-2 min-w-0">
                <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-emerald-500 to-[#0F4C81] text-xs font-black text-white shadow-2xs">
                  {selectedPatient.fullName.trim().slice(0, 1) || 'ผ'}
                </div>
                <span className="text-sm font-black text-slate-900 truncate">
                  {selectedPatient.fullName}
                </span>
                <span className="rounded-md bg-sky-100/90 px-2 py-0.5 text-[11px] font-black text-[#0F4C81]">
                  {selectedPatient.hn}
                </span>
                <span className="text-xs font-semibold text-slate-500">
                  อายุ {selectedPatient.age} ปี • {selectedPatient.gender || 'ไม่ระบุ'}
                </span>
                {selectedPatient.affectedSide && selectedPatient.affectedSide !== 'NONE' && (
                  <span className="rounded-md bg-emerald-100/90 px-2 py-0.5 text-[11px] font-extrabold text-emerald-800">
                    เน้นฝึกข้าง{selectedPatient.affectedSide === 'LEFT' ? 'ซ้าย' : selectedPatient.affectedSide === 'RIGHT' ? 'ขวา' : 'ทั้ง 2 ข้าง'}
                  </span>
                )}
              </div>

              <div className="flex items-center gap-1.5 shrink-0">
                <span className="font-mono text-[11px] font-bold text-slate-600 bg-white border border-slate-200 rounded-lg px-2 py-0.5">
                  {universalLinkCode}
                </span>
                <button
                  type="button"
                  onClick={() => void handleCopyLinkCode()}
                  className="inline-flex items-center gap-1 rounded-lg border border-emerald-200 bg-white px-2 py-0.5 text-[11px] font-extrabold text-emerald-700 hover:bg-emerald-50 cursor-pointer transition-colors"
                >
                  {copiedCode ? <Check className="h-3 w-3" /> : <Copy className="h-3 w-3" />}
                  <span>{copiedCode ? 'คัดลอกแล้ว' : 'คัดลอก'}</span>
                </button>
              </div>
            </div>
          )}

          {/* Sleek Segmented 2-Mode Switcher */}
          <div
            data-testid="primary-clinical-mode-bar"
            className="grid grid-cols-2 2xl:inline-flex items-center rounded-xl border border-slate-200/90 bg-slate-100/90 p-1 shrink-0 w-full 2xl:w-auto"
          >
            <button
              type="button"
              data-testid="primary-mode-dashboard-btn"
              onClick={() => setPrimaryPanelMode('DASHBOARD')}
              className={`inline-flex items-center justify-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-black transition-all cursor-pointer ${
                primaryPanelMode === 'DASHBOARD'
                  ? 'bg-[#0F4C81] text-white shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              <BarChart3 className="h-3.5 w-3.5 shrink-0" />
              <span>ภาพรวมและกราฟสถิติ</span>
            </button>
            <button
              type="button"
              data-testid="primary-mode-assess-btn"
              onClick={() => setPrimaryPanelMode('ASSESS_AND_COURSE')}
              className={`inline-flex items-center justify-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-black transition-all cursor-pointer ${
                primaryPanelMode === 'ASSESS_AND_COURSE'
                  ? 'bg-emerald-600 text-white shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              <FileText className="h-3.5 w-3.5 shrink-0" />
              <span>คอร์ส &amp; บันทึกประเมิน (0-6)</span>
              {courseSummariesAll.length > 0 && (
                <span
                  className={`rounded-full px-1.5 py-0.5 text-[10.5px] font-black ${
                    primaryPanelMode === 'ASSESS_AND_COURSE'
                      ? 'bg-white/25 text-white'
                      : 'bg-emerald-100 text-emerald-800'
                  }`}
                >
                  {courseSummariesAll.length} ท่า
                </span>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Main Full-Width Zero-Scroll Clinical Workspace (100% Width per Mode — Zero Cramped Half-Screen Columns!) */}
      <div className="flex-1 min-h-0 flex flex-col overflow-hidden">
        {/* MODE 2: Full-Width Course Quests (Left 6 Cols) + Spacious 0-6 Assessment Form (Right 6 Cols) */}
        <div
          hidden={primaryPanelMode !== 'ASSESS_AND_COURSE'}
          className={
            primaryPanelMode === 'ASSESS_AND_COURSE'
              ? 'flex-1 min-h-0 grid grid-cols-1 gap-3 lg:grid-cols-12 overflow-hidden animate-view-enter'
              : 'hidden'
          }
        >
          {/* Assigned Course & Automatic 0-6 Scoring Quest Card */}
          <div
            data-testid="assigned-course-auto-score-card"
            className="lg:col-span-6 h-full min-h-0 flex flex-col overflow-hidden rounded-2xl border-2 border-emerald-200/90 bg-gradient-to-br from-white via-[#F0FDF9] to-[#EFF6FF] p-3.5 shadow-sm gap-2.5 animate-view-enter"
          >
            <div className="shrink-0 flex flex-wrap items-center justify-between gap-2 border-b border-emerald-100 pb-2">
              <div className="flex items-center gap-2.5 min-w-0 flex-1">
                <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-emerald-600 text-white shadow-xs">
                  <ClipboardList className="h-4 w-4" />
                </span>
                <div className="min-w-0">
                  <h3 className="text-sm font-black text-slate-900 leading-tight">
                    คอร์สกายภาพที่นักกายภาพจัดให้ &amp; คำนวณคะแนนอัตโนมัติ
                  </h3>
                  <p className="text-xs font-semibold text-slate-600">
                    สรุประดับกำลังกล้ามเนื้อ 0-6 และองศา ROM อัตโนมัติ
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-1.5 shrink-0">
                {courseAutoBundle && courseAutoBundle.unrecordedResults.length > 0 ? (
                  <span className="shrink-0 rounded-full bg-emerald-600 px-2.5 py-1 text-xs font-black text-white shadow-xs whitespace-nowrap">
                    รอบันทึก {courseAutoBundle.unrecordedResults.length} รอบ
                  </span>
                ) : courseAutoBundle && courseAutoBundle.autoResults.length > 0 ? (
                  <span className="shrink-0 rounded-full border border-emerald-200 bg-emerald-50 px-2.5 py-1 text-xs font-extrabold text-emerald-700 whitespace-nowrap">
                    ซิงก์ครบ {courseAutoBundle.autoResults.length} รอบ
                  </span>
                ) : null}
                {courseTotalPages > 1 && (
                  <div className="inline-flex items-center rounded-xl border border-emerald-200 bg-white p-0.5">
                    <button
                      type="button"
                      aria-label="หน้าคอร์สก่อนหน้า"
                      onClick={() => setCoursePage((p) => (p - 1 + courseTotalPages) % courseTotalPages)}
                      className="flex h-7 w-7 items-center justify-center rounded-lg text-slate-600 hover:bg-emerald-50 hover:text-emerald-700 cursor-pointer"
                    >
                      <ChevronLeft className="h-4 w-4" />
                    </button>
                    <span className="px-2 text-xs font-black text-slate-700 tabular-nums">
                      {safeCoursePage + 1}/{courseTotalPages}
                    </span>
                    <button
                      type="button"
                      aria-label="หน้าคอร์สถัดไป"
                      onClick={() => setCoursePage((p) => (p + 1) % courseTotalPages)}
                      className="flex h-7 w-7 items-center justify-center rounded-lg text-slate-600 hover:bg-emerald-50 hover:text-emerald-700 cursor-pointer"
                    >
                      <ChevronRight className="h-4 w-4" />
                    </button>
                  </div>
                )}
              </div>
            </div>

            {loadingCourseContext && (!courseAutoBundle || courseAutoBundle.exerciseSummaries.length === 0) ? (
              <div className="flex-1 flex items-center justify-center rounded-xl border border-slate-200/80 bg-white/80 px-4 py-3 text-sm font-semibold text-slate-500">
                กำลังโหลดข้อมูลคอร์สกายภาพ...
              </div>
            ) : courseAutoBundle && courseAutoBundle.exerciseSummaries.length > 0 ? (
              <div className="flex-1 min-h-0 flex flex-col gap-2.5 overflow-hidden">
                <div ref={courseListBoxRef} className="flex-1 min-h-0 flex flex-col gap-2 overflow-hidden">
                  {visibleCourseSummaries.map((item) => {
                    const latest = item.latestResult;
                    const friendlyExName = formatFriendlyExerciseLabel(
                      item.definition.exerciseNameTh,
                      item.definition.exerciseCode,
                    );
                    const pct = latest
                      ? Math.min(100, Math.round((latest.effectiveRomDeg / Math.max(1, latest.targetRomDeg)) * 100))
                      : 0;
                    const isSelectedInForm =
                      activeAutoCalcResult &&
                      latest &&
                      activeAutoCalcResult.sourceSessionId === latest.sourceSessionId;
                    return (
                      <div
                        key={item.definition.key}
                        className={`rounded-xl border px-3 py-2.5 transition-all space-y-1.5 ${
                          isSelectedInForm
                            ? 'border-emerald-400 bg-white shadow-xs ring-2 ring-emerald-200'
                            : 'border-slate-200/80 bg-white/95 hover:border-sky-300'
                        }`}
                      >
                        <div className="flex items-center justify-between gap-2">
                          <div className="flex items-center gap-2 min-w-0">
                            <span className="text-sm font-black text-slate-900 truncate">
                              {friendlyExName}
                            </span>
                            <span className="shrink-0 rounded-full bg-sky-100 px-2 py-0.5 text-xs font-extrabold text-[#0F4C81]">
                              {formatSideTh(item.definition.targetSide)}
                            </span>
                          </div>
                          <div className="flex items-center gap-1.5 shrink-0">
                            {latest ? (
                              <>
                                <span className="rounded-full bg-emerald-600 px-2.5 py-0.5 text-xs font-black text-white">
                                  ระดับ {latest.calculatedScore0To6}/6
                                </span>
                                <button
                                  type="button"
                                  onClick={() => {
                                    formManuallyEditedRef.current = false;
                                    applyCourseResultToForm(latest, true);
                                  }}
                                  className="btn-tactile rounded-lg border border-sky-200 bg-sky-50 px-2.5 py-1 text-xs font-extrabold text-[#0F4C81] hover:bg-sky-100 cursor-pointer"
                                >
                                  เลือก
                                </button>
                              </>
                            ) : (
                              <span className="rounded-full bg-slate-100 px-2.5 py-0.5 text-xs font-bold text-slate-500">
                                รอฝึก
                              </span>
                            )}
                          </div>
                        </div>
                        {latest && (
                          <div className="space-y-1">
                            <div className="h-2 w-full overflow-hidden rounded-full bg-slate-100">
                              <div
                                className="h-full rounded-full bg-gradient-to-r from-emerald-500 to-teal-400 game-progress-stripes"
                                style={{ width: `${pct}%` }}
                              />
                            </div>
                            <div className="flex items-center justify-between text-xs font-bold text-slate-600">
                              <span>
                                ยกได้ <strong className="text-emerald-700">{latest.effectiveRomDeg.toFixed(1)}°</strong> / เป้า {Math.round(latest.targetRomDeg)}°
                              </span>
                              <span className="text-[#0F4C81] font-black">{pct}%</span>
                            </div>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>

                {courseAutoBundle.latestResult && (
                  <div
                    data-testid="course-auto-score-breakdown"
                    className="shrink-0 rounded-xl border border-sky-200 bg-white px-3 py-2 text-xs sm:text-sm flex items-center justify-between gap-2 animate-toast-in"
                  >
                    <span className="font-semibold text-slate-700 truncate">
                      {courseAutoBundle.latestResult.breakdownSummaryTh}
                    </span>
                    <span className="shrink-0 rounded-full bg-sky-100 border border-sky-200 px-2.5 py-0.5 text-xs font-black text-[#0F4C81]">
                      {courseAutoBundle.latestResult.compositePct}%
                    </span>
                  </div>
                )}

                <button
                  type="button"
                  data-testid="batch-sync-course-card-btn"
                  disabled={
                    batchSyncingCourse ||
                    !selectedPatient ||
                    courseAutoBundle.autoResults.length === 0
                  }
                  onClick={() => void handleBatchSyncCourseSessions()}
                  className="shrink-0 btn-game-3d-emerald btn-tactile w-full inline-flex items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-emerald-600 to-teal-600 hover:brightness-105 px-4 h-11 text-sm font-black text-white shadow-sm disabled:opacity-45 cursor-pointer"
                >
                  <RefreshCw
                    className={`h-4 w-4 ${batchSyncingCourse ? 'animate-spin' : ''}`}
                  />
                  <span>
                    {batchSyncingCourse
                      ? 'กำลังบันทึกผลคอร์ส...'
                      : 'ดึงผลเล่นคอร์สและบันทึกอัตโนมัติ'}
                  </span>
                </button>
              </div>
            ) : (
              <div className="flex-1 flex items-center justify-center rounded-xl border-2 border-dashed border-emerald-200 bg-white/80 px-4 py-4 text-center text-sm font-semibold text-slate-500">
                ยังไม่มีผลฝึกจากคอร์สสำหรับผู้สูงอายุรายนี้ — เมื่อเริ่มฝึกหน้าตู้ Kiosk คะแนนจะขึ้นมาอัตโนมัติ
              </div>
            )}
          </div>

          {/* Spacious Large-Font 0-6 Assessment Form (Right 6 Cols of Mode 2) */}
          <form
            onSubmit={(e) => void handleSubmitAssessment(e)}
            className="lg:col-span-6 h-full min-h-0 flex flex-col justify-between rounded-2xl border-2 border-slate-200 bg-white p-4 shadow-sm gap-3 overflow-y-auto"
          >
            <div className="space-y-3">
            <div className="flex items-center justify-between gap-2 border-b border-slate-100 pb-2.5">
              <div className="flex items-center gap-2.5">
                <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-[#0F4C81] text-white shadow-xs">
                  <FileText className="h-4 w-4" />
                </span>
                <div>
                  <h3 className="text-base font-black text-slate-900 leading-tight">
                    บันทึกการประเมิน (0-6)
                  </h3>
                  <p className="text-xs font-semibold text-slate-500">
                    แตะเลือกระดับคะแนนและกรอกองศาที่ยกได้
                  </p>
                </div>
              </div>
              <span className="rounded-full bg-emerald-50 border border-emerald-200 px-3 py-1 text-xs font-extrabold text-emerald-700">
                {activeAutoCalcResult ? 'คำนวณอัตโนมัติ' : 'Mahidol 0-6'}
              </span>
            </div>

            {/* Interactive Large Tactile 0-6 Score Picker */}
            <div className="space-y-2">
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
                      className={`btn-tactile h-11 rounded-xl border-2 text-base font-black transition-all cursor-pointer ${
                        active
                          ? 'bg-emerald-600 text-white border-emerald-700 btn-game-3d-emerald scale-105'
                          : 'bg-slate-50 text-slate-700 border-slate-200 hover:border-sky-400 hover:bg-sky-50'
                      }`}
                    >
                      {item.score}
                    </button>
                  );
                })}
              </div>
              <div className="rounded-xl border border-emerald-200 bg-emerald-50/70 px-3 py-2 text-xs sm:text-sm font-bold text-emerald-950 leading-snug">
                ระดับ {selectedScore}/6 • {selectedRubric.descriptionTh}
              </div>
            </div>

            {/* Exercise + Target Side */}
            <div className="space-y-2">
              <select
                aria-label="ท่ากายภาพบำบัด"
                value={exerciseCode}
                onChange={(e) => {
                  formManuallyEditedRef.current = true;
                  handleExerciseChange(e.target.value);
                }}
                className="h-10 w-full rounded-xl border-2 border-slate-200 bg-slate-50/50 px-3 text-sm font-bold text-slate-800 focus:border-sky-500 focus:bg-white focus:outline-none"
              >
                {CLINICAL_EXERCISES.map((ex) => (
                  <option key={ex.code} value={ex.code}>
                    {formatFriendlyExerciseLabel(ex.name.th, ex.code)} (เป้า {ex.detection.targetMinDeg}°)
                  </option>
                ))}
              </select>
              <div className="grid grid-cols-3 gap-2">
                {SIDE_OPTIONS.map((opt) => (
                  <button
                    key={opt.value}
                    type="button"
                    onClick={() => {
                      formManuallyEditedRef.current = true;
                      setTargetSide(opt.value);
                    }}
                    className={`h-10 rounded-xl border-2 text-sm font-black cursor-pointer transition-all ${
                      targetSide === opt.value
                        ? 'border-[#0F4C81] bg-[#0F4C81] text-white shadow-xs'
                        : 'border-slate-200 bg-slate-50 text-slate-700 hover:bg-sky-50 hover:border-sky-300'
                    }`}
                  >
                    ข้าง{opt.labelTh}
                  </button>
                ))}
              </div>
            </div>

            {/* Measured ROM + Quick Actions Arena */}
            <div className="rounded-xl border-2 border-sky-200/80 bg-gradient-to-br from-sky-50/70 via-white to-emerald-50/40 p-3 space-y-2.5">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <span className="text-sm font-black text-[#0F4C81]">องศาการยกแขน (ROM) &amp; ความปวด</span>
                <div className="flex flex-wrap items-center gap-1.5">
                  <button
                    type="button"
                    data-testid="batch-sync-course-inline-btn"
                    disabled={batchSyncingCourse || !selectedPatient}
                    onClick={() => void handleBatchSyncCourseSessions()}
                    className="btn-tactile inline-flex items-center gap-1 rounded-lg bg-emerald-600 hover:bg-emerald-700 px-2.5 py-1 text-xs font-extrabold text-white disabled:opacity-50 cursor-pointer"
                  >
                    <RefreshCw className={`h-3.5 w-3.5 ${batchSyncingCourse ? 'animate-spin' : ''}`} />
                    <span>ดึงคอร์ส</span>
                  </button>
                  <button
                    type="button"
                    onClick={handlePullLatestKioskRom}
                    className="btn-tactile inline-flex items-center gap-1 rounded-lg border border-emerald-300 bg-white px-2.5 py-1 text-xs font-extrabold text-emerald-700 hover:bg-emerald-50 cursor-pointer"
                  >
                    <Sparkles className="h-3.5 w-3.5" />
                    <span>Kiosk</span>
                  </button>
                  <button
                    type="button"
                    onClick={() => setLiveMeasureModalOpen(true)}
                    className="btn-tactile inline-flex items-center gap-1 rounded-lg border border-sky-300 bg-white px-2.5 py-1 text-xs font-extrabold text-[#0F4C81] hover:bg-sky-50 cursor-pointer"
                  >
                    <Camera className="h-3.5 w-3.5" />
                    <span>กล้อง AI</span>
                  </button>
                </div>
              </div>

              <div className="grid grid-cols-3 gap-2">
                <div>
                  <span className="block text-xs font-bold text-slate-600 mb-1">ยกได้ (°)</span>
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
                    className="h-10 w-full rounded-xl border-2 border-slate-200 bg-white px-3 text-sm font-black text-slate-900 focus:border-sky-500 focus:outline-none"
                  />
                </div>
                <div>
                  <span className="block text-xs font-bold text-slate-600 mb-1">เป้าหมาย (°)</span>
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
                    className="h-10 w-full rounded-xl border-2 border-slate-200 bg-white px-3 text-sm font-bold text-slate-700 focus:border-sky-500 focus:outline-none"
                  />
                </div>
                <div>
                  <span className="block text-xs font-bold text-slate-600 mb-1">ปวด (0-10)</span>
                  <input
                    type="number"
                    min={0}
                    max={10}
                    placeholder="0"
                    value={painVasInput}
                    onChange={(e) => {
                      formManuallyEditedRef.current = true;
                      setPainVasInput(e.target.value);
                    }}
                    className="h-10 w-full rounded-xl border-2 border-slate-200 bg-white px-3 text-sm font-bold text-slate-700 focus:border-sky-500 focus:outline-none"
                  />
                </div>
              </div>

              {liveFormPreviewPct !== null && liveFormSuggestedScore !== null && (
                <div className="rounded-xl border border-emerald-200 bg-white px-3 py-2 space-y-1.5">
                  <div className="flex flex-wrap items-center justify-between gap-1 text-xs sm:text-sm">
                    <span className="font-bold text-slate-700">
                      ความสำเร็จเทียบเป้า:{' '}
                      <strong className="text-emerald-700">{liveFormPreviewPct.toFixed(1)}%</strong>
                    </span>
                    {liveFormSuggestedScore !== selectedScore && (
                      <button
                        type="button"
                        onClick={() => {
                          formManuallyEditedRef.current = true;
                          setSelectedScore(liveFormSuggestedScore);
                        }}
                        className="rounded-lg border border-sky-200 bg-sky-50 px-2.5 py-0.5 text-xs font-black text-[#0F4C81] hover:bg-sky-100 cursor-pointer"
                      >
                        แนะนำระดับ {liveFormSuggestedScore}/6
                      </button>
                    )}
                  </div>
                  <div className="h-2 w-full overflow-hidden rounded-full bg-slate-100">
                    <div
                      className="h-full rounded-full bg-gradient-to-r from-emerald-500 to-teal-400"
                      style={{ width: `${Math.min(100, liveFormPreviewPct)}%` }}
                    />
                  </div>
                </div>
              )}
            </div>

            {/* Collapsible Optional Notes & Details */}
            <div className="rounded-xl border border-slate-200 bg-slate-50/70 px-3 py-2 space-y-2">
              <button
                type="button"
                onClick={() => setShowExtraFormNotes((v) => !v)}
                className="w-full flex items-center justify-between text-xs sm:text-sm font-extrabold text-slate-700 hover:text-[#0F4C81] cursor-pointer px-0.5"
              >
                <span>รายละเอียดเพิ่มเติม (วันเวลา / สถานที่ / บันทึกส่งต่อ)</span>
                {showExtraFormNotes ? (
                  <ChevronUp className="h-4 w-4 text-slate-500" />
                ) : (
                  <ChevronDown className="h-4 w-4 text-slate-500" />
                )}
              </button>

              {showExtraFormNotes && (
                <div className="space-y-2 pt-1">
                  <div className="grid grid-cols-2 gap-2">
                    <input
                      aria-label="วันที่และเวลาที่ประเมิน"
                      type="datetime-local"
                      value={assessedAtInput}
                      onChange={(e) => setAssessedAtInput(e.target.value)}
                      className="h-9 w-full rounded-lg border border-slate-200 bg-white px-2.5 text-xs font-semibold text-slate-800 focus:border-sky-500 focus:outline-none"
                    />
                    <input
                      aria-label="สถานที่ตรวจ"
                      type="text"
                      value={facilityName}
                      onChange={(e) => setFacilityName(e.target.value)}
                      className="h-9 w-full rounded-lg border border-slate-200 bg-white px-2.5 text-xs font-semibold text-slate-800 focus:border-sky-500 focus:outline-none"
                    />
                  </div>
                  <input
                    type="text"
                    value={chiefComplaint}
                    onChange={(e) => {
                      formManuallyEditedRef.current = true;
                      setChiefComplaint(e.target.value);
                    }}
                    placeholder="อาการรอบนี้ (เช่น ยกแขนได้ดีขึ้น ไม่มีอาการปวด)..."
                    className="h-9 w-full rounded-lg border border-slate-200 bg-white px-3 text-xs sm:text-sm text-slate-800 focus:border-sky-500 focus:outline-none"
                  />
                  <textarea
                    rows={2}
                    value={handoverNote}
                    onChange={(e) => {
                      formManuallyEditedRef.current = true;
                      setHandoverNote(e.target.value);
                    }}
                    placeholder="บันทึกส่งต่อนักกายภาพ..."
                    className="w-full rounded-lg border border-slate-200 bg-white p-2.5 text-xs sm:text-sm text-slate-800 focus:border-sky-500 focus:outline-none"
                  />
                  <input
                    type="text"
                    value={nextGoal}
                    onChange={(e) => {
                      formManuallyEditedRef.current = true;
                      setNextGoal(e.target.value);
                    }}
                    placeholder="เป้าหมายครั้งถัดไป..."
                    className="h-9 w-full rounded-lg border border-slate-200 bg-white px-3 text-xs sm:text-sm text-slate-800 focus:border-sky-500 focus:outline-none"
                  />
                </div>
              )}
            </div>
            </div>

            <button
              type="submit"
              disabled={saving || !selectedPatient}
              className="shrink-0 btn-primary-sheen btn-game-3d-blue btn-tactile w-full h-11 rounded-xl bg-gradient-to-r from-[#0F4C81] via-[#0284C7] to-[#0D9488] text-sm font-black text-white hover:brightness-105 disabled:opacity-50 cursor-pointer"
            >
              {saving ? 'กำลังบันทึก...' : 'บันทึกผลประเมิน (0-6)'}
            </button>
          </form>
        </div>

        {/* MODE 1 (DEFAULT): FULL 100% WIDTH Clinical Dashboard & Statistics Workspace */}
        <div
          hidden={primaryPanelMode !== 'DASHBOARD'}
          className={
            primaryPanelMode === 'DASHBOARD'
              ? 'flex-1 min-h-0 flex flex-col gap-2.5 min-w-0 overflow-hidden animate-view-enter'
              : 'hidden'
          }
        >
          {/* Elevated Clean Verdict Banner (Zero cramped truncation!) */}
          <div
            data-testid="unified-progress-verdict-card"
            className={`shrink-0 rounded-2xl border border-l-4 px-3.5 py-2 shadow-2xs ${
              unifiedSummary.overall.verdict.tone === 'emerald'
                ? 'border-emerald-200/90 border-l-emerald-500 bg-gradient-to-r from-emerald-50/70 via-white to-white'
                : unifiedSummary.overall.verdict.tone === 'amber'
                  ? 'border-amber-200/90 border-l-amber-500 bg-gradient-to-r from-amber-50/70 via-white to-white'
                  : 'border-sky-200/90 border-l-[#0F4C81] bg-gradient-to-r from-sky-50/70 via-white to-white'
            }`}
          >
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="flex flex-wrap items-center gap-2 min-w-0 flex-1">
                <span
                  data-testid="unified-verdict-badge"
                  className={`inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-[11px] font-black whitespace-nowrap shrink-0 ${verdictBadgeClasses(unifiedSummary.overall.verdict.tone)}`}
                >
                  {unifiedSummary.overall.verdict.code === 'IMPROVED' ? (
                    <ArrowUpRight className="h-3 w-3 shrink-0" />
                  ) : unifiedSummary.overall.verdict.code === 'TRY_HARDER' ? (
                    <ArrowDownRight className="h-3 w-3 shrink-0" />
                  ) : unifiedSummary.overall.verdict.code === 'STABLE' ? (
                    <Minus className="h-3 w-3 shrink-0" />
                  ) : (
                    <Award className="h-3 w-3 shrink-0" />
                  )}
                  <span>{unifiedSummary.overall.verdict.badgeTh}</span>
                </span>
                <h3
                  data-testid="unified-verdict-headline"
                  className="text-xs sm:text-sm font-black text-slate-900 leading-snug"
                >
                  {unifiedSummary.overall.verdict.headlineTh}
                </h3>
                <span className="sr-only">
                  {unifiedSummary.overall.verdict.detailTh} {unifiedSummary.overall.verdict.adviceTh}
                </span>
              </div>

              <div className="flex items-center gap-1.5 shrink-0">
                <button
                  type="button"
                  data-testid="verdict-domain-rehab"
                  onClick={() => setActiveWorkspaceTab('ROM')}
                  className="rounded-lg border border-slate-200 bg-white hover:border-sky-300 hover:text-[#0F4C81] px-2.5 py-1 text-[11px] font-extrabold text-slate-700 whitespace-nowrap cursor-pointer transition-colors"
                >
                  กายภาพ {unifiedSummary.rehab.totalVisits} ครั้ง
                </button>
                <button
                  type="button"
                  data-testid="verdict-domain-minigame"
                  onClick={() => setActiveWorkspaceTab('MINIGAME')}
                  className="rounded-lg border border-slate-200 bg-white hover:border-indigo-300 hover:text-indigo-700 px-2.5 py-1 text-[11px] font-extrabold text-slate-700 whitespace-nowrap cursor-pointer transition-colors"
                >
                  มินิเกมฝึกสมอง {unifiedSummary.minigame.totalRounds} รอบ
                </button>
                {unifiedSummary.overall.combinedScorePct !== null && (
                  <span
                    data-testid="unified-combined-score-pct"
                    className="rounded-lg bg-emerald-600 px-2.5 py-1 text-[11px] font-black text-white tabular-nums whitespace-nowrap shadow-2xs"
                  >
                    รวม {unifiedSummary.overall.combinedScorePct.toFixed(1)}%
                  </span>
                )}
              </div>
            </div>
          </div>

          {/* 4 Polished Medical-Grade KPI Metric Cards */}
          <div className="shrink-0 grid grid-cols-2 gap-2.5 lg:grid-cols-4">
            {(() => {
              const scorePctVal = rehabStockIndex.latestPoint
                ? rehabStockIndex.latestPoint.smartTotalPct
                : summary.latestTargetAchievementPct !== null
                  ? Math.min(100, summary.latestTargetAchievementPct)
                  : 0;
              return (
                <button
                  type="button"
                  onClick={() => setActiveWorkspaceTab('STOCK')}
                  className={`text-left rounded-2xl border bg-white px-3.5 py-2.5 flex flex-col justify-between gap-2 shadow-2xs transition-all cursor-pointer ${
                    activeWorkspaceTab === 'STOCK'
                      ? 'border-emerald-400 ring-2 ring-emerald-500/15 bg-gradient-to-br from-emerald-50/40 via-white to-white'
                      : 'border-slate-200/85 hover:border-emerald-300 hover:shadow-xs'
                  }`}
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-[11px] font-extrabold text-slate-500 truncate">คะแนนฟื้นฟูรวม</span>
                    <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-lg bg-emerald-100 text-emerald-700">
                      <TrendingUp className="h-3.5 w-3.5" />
                    </span>
                  </div>
                  <div className="flex items-baseline justify-between gap-2">
                    <span className="text-2xl font-black text-emerald-700 tabular-nums leading-none">
                      {scorePctVal.toFixed(1)}%
                    </span>
                    <span className="text-[11px] font-extrabold text-slate-500 tabular-nums">
                      {rehabStockIndex.latestPoint
                        ? `${rehabStockIndex.latestPoint.smartEarnedScore}/${rehabStockIndex.maxTotalScore}`
                        : `${rehabStockIndex.totalPrescribedExercises} ท่า`}
                    </span>
                  </div>
                  <div className="h-1.5 w-full rounded-full bg-slate-100 overflow-hidden">
                    <div
                      className="h-full rounded-full bg-gradient-to-r from-emerald-500 to-teal-400"
                      style={{ width: `${Math.max(4, Math.min(100, scorePctVal))}%` }}
                    />
                  </div>
                </button>
              );
            })()}

            <button
              type="button"
              onClick={() => {
                setActiveWorkspaceTab('ROM');
                setChartViewMode('SESSIONS');
              }}
              className={`text-left rounded-2xl border bg-white px-3.5 py-2.5 flex flex-col justify-between gap-2 shadow-2xs transition-all cursor-pointer ${
                activeWorkspaceTab === 'ROM'
                  ? 'border-sky-400 ring-2 ring-sky-500/15 bg-gradient-to-br from-sky-50/40 via-white to-white'
                  : 'border-slate-200/85 hover:border-sky-300 hover:shadow-xs'
              }`}
            >
              <div className="flex items-center justify-between gap-2">
                <span className="text-[11px] font-extrabold text-slate-500 truncate">องศาเทียบเป้า</span>
                <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-lg bg-sky-100 text-[#0F4C81]">
                  <Target className="h-3.5 w-3.5" />
                </span>
              </div>
              <div className="flex items-baseline justify-between gap-2">
                <span className="text-2xl font-black text-[#0F4C81] tabular-nums leading-none">
                  {summary.latestTargetAchievementPct !== null
                    ? `${summary.latestTargetAchievementPct.toFixed(0)}%`
                    : '-'}
                </span>
                <span className="text-[11px] font-extrabold text-slate-500 tabular-nums">
                  {summary.latestRom !== null ? `ROM ${summary.latestRom.toFixed(0)}°` : '-'}
                </span>
              </div>
              <div className="h-1.5 w-full rounded-full bg-slate-100 overflow-hidden">
                <div
                  className="h-full rounded-full bg-gradient-to-r from-[#0F4C81] to-sky-400"
                  style={{ width: `${Math.max(4, Math.min(100, summary.latestTargetAchievementPct ?? 0))}%` }}
                />
              </div>
            </button>

            <button
              type="button"
              onClick={() => setActiveWorkspaceTab('MONTHLY')}
              className={`text-left rounded-2xl border bg-white px-3.5 py-2.5 flex flex-col justify-between gap-2 shadow-2xs transition-all cursor-pointer ${
                activeWorkspaceTab === 'MONTHLY'
                  ? 'border-teal-400 ring-2 ring-teal-500/15 bg-gradient-to-br from-teal-50/40 via-white to-white'
                  : 'border-slate-200/85 hover:border-teal-300 hover:shadow-xs'
              }`}
            >
              <div className="flex items-center justify-between gap-2">
                <span className="text-[11px] font-extrabold text-slate-500 truncate">กำลังกล้ามเนื้อ</span>
                <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-lg bg-teal-100 text-teal-700">
                  <Activity className="h-3.5 w-3.5" />
                </span>
              </div>
              <div className="flex items-baseline justify-between gap-2">
                <span className="text-2xl font-black text-slate-900 tabular-nums leading-none">
                  {summary.latestScore !== null ? `${summary.latestScore}/6` : '-'}
                </span>
                <span className="text-[11px] font-extrabold text-teal-700">เกณฑ์มหิดล</span>
              </div>
              <div className="grid grid-cols-6 gap-1">
                {[1, 2, 3, 4, 5, 6].map((lvl) => (
                  <span
                    key={lvl}
                    className={`h-1.5 rounded-full ${
                      (summary.latestScore ?? 0) >= lvl ? 'bg-teal-500' : 'bg-slate-100'
                    }`}
                  />
                ))}
              </div>
            </button>

            <button
              type="button"
              onClick={() => setActiveWorkspaceTab(unifiedSummary.minigame.totalRounds > 0 ? 'MINIGAME' : 'TIMELINE')}
              className={`text-left rounded-2xl border bg-white px-3.5 py-2.5 flex flex-col justify-between gap-2 shadow-2xs transition-all cursor-pointer ${
                activeWorkspaceTab === 'MINIGAME' || activeWorkspaceTab === 'TIMELINE'
                  ? 'border-indigo-400 ring-2 ring-indigo-500/15 bg-gradient-to-br from-indigo-50/40 via-white to-white'
                  : 'border-slate-200/85 hover:border-indigo-300 hover:shadow-xs'
              }`}
            >
              <div className="flex items-center justify-between gap-2">
                <span className="text-[11px] font-extrabold text-slate-500 truncate">ประวัติฝึก / มินิเกม</span>
                <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-lg bg-indigo-100 text-indigo-700">
                  <Gamepad2 className="h-3.5 w-3.5" />
                </span>
              </div>
              <div className="flex items-baseline justify-between gap-2">
                <span className="text-2xl font-black text-indigo-700 tabular-nums leading-none">
                  {unifiedSummary.minigame.latestSession
                    ? `${unifiedSummary.minigame.latestSession.accuracyPct}%`
                    : `${summary.totalVisits} ครั้ง`}
                </span>
                <span className="text-[11px] font-extrabold text-slate-500">
                  {unifiedSummary.minigame.totalRounds > 0 ? `${unifiedSummary.minigame.totalRounds} รอบเกม` : 'กายภาพ'}
                </span>
              </div>
              <div className="h-1.5 w-full rounded-full bg-slate-100 overflow-hidden">
                <div
                  className="h-full rounded-full bg-gradient-to-r from-indigo-500 to-violet-400"
                  style={{
                    width: `${
                      unifiedSummary.minigame.latestSession
                        ? Math.max(8, unifiedSummary.minigame.latestSession.accuracyPct)
                        : Math.min(100, Math.max(10, summary.totalVisits * 15))
                    }%`,
                  }}
                />
              </div>
            </button>
          </div>

          {/* Sleek Modern Pill-Dock Graph Switcher */}
          <div
            data-testid="clinical-chart-tab-switcher"
            className="shrink-0 rounded-2xl border border-slate-200/90 bg-slate-100/80 p-1 flex flex-wrap items-center justify-between gap-1.5"
          >
            <div role="tablist" aria-label="สลับมุมมองกราฟและรายงาน" className="flex flex-wrap items-center gap-1">
              <button
                type="button"
                role="tab"
                aria-selected={activeWorkspaceTab === 'STOCK'}
                data-testid="workspace-tab-STOCK"
                onClick={() => setActiveWorkspaceTab('STOCK')}
                className={`inline-flex items-center gap-1.5 rounded-xl px-3 py-1.5 text-xs font-black transition-all cursor-pointer ${
                  activeWorkspaceTab === 'STOCK'
                    ? 'bg-white text-emerald-700 shadow-xs ring-1 ring-emerald-200'
                    : 'text-slate-600 hover:text-slate-900 hover:bg-white/60'
                }`}
              >
                <TrendingUp className="h-3.5 w-3.5 shrink-0" />
                <span>กราฟดัชนีรวม</span>
              </button>

              <button
                type="button"
                role="tab"
                aria-selected={activeWorkspaceTab === 'ROM'}
                data-testid="workspace-tab-ROM"
                onClick={() => setActiveWorkspaceTab('ROM')}
                className={`inline-flex items-center gap-1.5 rounded-xl px-3 py-1.5 text-xs font-black transition-all cursor-pointer ${
                  activeWorkspaceTab === 'ROM'
                    ? 'bg-white text-[#0F4C81] shadow-xs ring-1 ring-sky-200'
                    : 'text-slate-600 hover:text-slate-900 hover:bg-white/60'
                }`}
              >
                <Activity className="h-3.5 w-3.5 shrink-0" />
                <span>กราฟ ROM</span>
              </button>

              <button
                type="button"
                role="tab"
                aria-selected={activeWorkspaceTab === 'MINIGAME'}
                data-testid="workspace-tab-MINIGAME"
                onClick={() => setActiveWorkspaceTab('MINIGAME')}
                className={`inline-flex items-center gap-1.5 rounded-xl px-3 py-1.5 text-xs font-black transition-all cursor-pointer ${
                  activeWorkspaceTab === 'MINIGAME'
                    ? 'bg-white text-indigo-700 shadow-xs ring-1 ring-indigo-200'
                    : 'text-slate-600 hover:text-slate-900 hover:bg-white/60'
                }`}
              >
                <Gamepad2 className="h-3.5 w-3.5 shrink-0" />
                <span>มินิเกม ({unifiedSummary.minigame.totalRounds})</span>
              </button>

              <button
                type="button"
                role="tab"
                aria-selected={activeWorkspaceTab === 'MONTHLY'}
                data-testid="workspace-tab-MONTHLY"
                onClick={() => setActiveWorkspaceTab('MONTHLY')}
                className={`inline-flex items-center gap-1.5 rounded-xl px-3 py-1.5 text-xs font-black transition-all cursor-pointer ${
                  activeWorkspaceTab === 'MONTHLY'
                    ? 'bg-white text-teal-700 shadow-xs ring-1 ring-teal-200'
                    : 'text-slate-600 hover:text-slate-900 hover:bg-white/60'
                }`}
              >
                <BarChart3 className="h-3.5 w-3.5 shrink-0" />
                <span>รายเดือน</span>
              </button>

              <button
                type="button"
                role="tab"
                aria-selected={activeWorkspaceTab === 'TABLE'}
                data-testid="workspace-tab-TABLE"
                onClick={() => setActiveWorkspaceTab('TABLE')}
                className={`inline-flex items-center gap-1.5 rounded-xl px-3 py-1.5 text-xs font-black transition-all cursor-pointer ${
                  activeWorkspaceTab === 'TABLE'
                    ? 'bg-white text-[#0F4C81] shadow-xs ring-1 ring-sky-200'
                    : 'text-slate-600 hover:text-slate-900 hover:bg-white/60'
                }`}
              >
                <Table2 className="h-3.5 w-3.5 shrink-0" />
                <span>ตารางรายท่า</span>
              </button>

              <button
                type="button"
                role="tab"
                aria-selected={activeWorkspaceTab === 'TIMELINE'}
                data-testid="workspace-tab-TIMELINE"
                onClick={() => setActiveWorkspaceTab('TIMELINE')}
                className={`inline-flex items-center gap-1.5 rounded-xl px-3 py-1.5 text-xs font-black transition-all cursor-pointer ${
                  activeWorkspaceTab === 'TIMELINE'
                    ? 'bg-white text-slate-900 shadow-xs ring-1 ring-slate-200'
                    : 'text-slate-600 hover:text-slate-900 hover:bg-white/60'
                }`}
              >
                <History className="h-3.5 w-3.5 shrink-0" />
                <span>ประวัติ ({filteredUnifiedTimeline.length})</span>
              </button>
            </div>

            <div className="flex items-center gap-1.5 shrink-0">
              <button
                type="button"
                data-testid="toggle-clinical-filters-btn"
                onClick={() => setShowFilterBar((v) => !v)}
                className={`inline-flex items-center gap-1 rounded-xl border px-2.5 py-1 text-xs font-extrabold cursor-pointer transition-colors ${
                  showFilterBar || activeFilterCount > 0
                    ? 'border-sky-300 bg-white text-[#0F4C81] shadow-2xs'
                    : 'border-transparent bg-transparent text-slate-600 hover:bg-white/70'
                }`}
              >
                <Filter className="h-3.5 w-3.5" />
                <span>กรอง{activeFilterCount > 0 ? ` (${activeFilterCount})` : ''}</span>
              </button>

              <div className="inline-flex items-center rounded-xl border border-slate-200/80 bg-white p-0.5 shadow-2xs">
                <button
                  type="button"
                  data-testid="workspace-tab-prev-btn"
                  aria-label="กราฟก่อนหน้า"
                  onClick={() => handleStepWorkspaceTab(-1)}
                  className="flex h-6 w-6 items-center justify-center rounded-lg text-slate-600 hover:bg-slate-100 cursor-pointer"
                >
                  <ChevronLeft className="h-3.5 w-3.5" />
                </button>
                <span className="px-1.5 text-[11px] font-black text-slate-700 tabular-nums">
                  {activeWorkspaceTabIndex + 1}/{WORKSPACE_CHART_TABS.length}
                </span>
                <button
                  type="button"
                  data-testid="workspace-tab-next-btn"
                  aria-label="กราฟถัดไป"
                  onClick={() => handleStepWorkspaceTab(1)}
                  className="flex h-6 w-6 items-center justify-center rounded-lg text-slate-600 hover:bg-slate-100 cursor-pointer"
                >
                  <ChevronRight className="h-3.5 w-3.5" />
                </button>
              </div>
            </div>
          </div>

          {/* Collapsible Compact Multi-Dimensional Filter Strip */}
          <div
            hidden={!showFilterBar}
            className={
              showFilterBar
                ? 'shrink-0 rounded-xl border border-sky-200 bg-sky-50/50 px-2.5 py-2 flex flex-wrap items-center justify-between gap-2 animate-view-enter'
                : 'hidden'
            }
          >
            <div className="flex flex-wrap items-center gap-1.5">
              <select
                aria-label="กรองตามท่ากายภาพ"
                value={exerciseFilter}
                onChange={(e) => setExerciseFilter(e.target.value)}
                className="h-8 rounded-lg border border-slate-200 bg-white px-2.5 text-xs font-bold text-slate-700"
              >
                <option value="ALL">ทุกท่ากายภาพบำบัด ({effectiveRecords.length})</option>
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
                className="h-8 rounded-lg border border-slate-200 bg-white px-2.5 text-xs font-bold text-slate-700"
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
                className="h-8 rounded-lg border border-slate-200 bg-white px-2.5 text-xs font-bold text-slate-700"
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
                className="h-8 rounded-lg border border-slate-200 bg-white px-2.5 text-xs font-bold text-slate-700"
              >
                <option value="ALL">ทุกแหล่งข้อมูล</option>
                <option value="COURSE_AUTO_CALCULATED">คำนวณอัตโนมัติจากคอร์ส</option>
                <option value="STAFF_MANUAL_ASSESSMENT">ประเมินโดยบุคลากร</option>
                <option value="KIOSK_SESSION_AUTO_SYNC">ซิงก์จากตู้ Kiosk</option>
              </select>
            </div>

            {activeFilterCount > 0 && (
              <button
                type="button"
                onClick={() => {
                  setExerciseFilter('ALL');
                  setSideFilter('ALL');
                  setDateRangeFilter('ALL');
                  setSourceFilter('ALL');
                }}
                className="rounded-lg border border-slate-200 bg-white px-2.5 py-1 text-xs font-bold text-slate-600 hover:bg-slate-50 cursor-pointer"
              >
                ล้างตัวกรอง
              </button>
            )}
          </div>

          {/* TAB PANEL 1: Stock-Market-Style Total Rehab Index Chart */}
          <div
            role="tabpanel"
            data-testid="workspace-panel-STOCK"
            hidden={activeWorkspaceTab !== 'STOCK'}
            className={
              activeWorkspaceTab === 'STOCK'
                ? 'flex-1 min-h-0 flex flex-col overflow-hidden animate-view-enter'
                : 'hidden'
            }
          >
            {selectedPatient && (
              <RehabStockPerformanceCard
                patientName={selectedPatient.fullName}
                patientHn={selectedPatient.hn}
                stockIndex={rehabStockIndex}
                compact
                fillHeight
                doctorBrief={doctorBrief}
              />
            )}
          </div>

          {/* TAB PANEL 2: Multi-Exercise ROM & 0-6 Progression + 3-Milestone Bar + 2-Visit Comparator */}
          <div
            role="tabpanel"
            data-testid="workspace-panel-ROM"
            hidden={activeWorkspaceTab !== 'ROM'}
            className={
              activeWorkspaceTab === 'ROM'
                ? 'flex-1 min-h-0 flex flex-col gap-2 overflow-hidden animate-view-enter'
                : 'hidden'
            }
          >
            {/* Compact 3-Milestone Navigator (Previous • Latest • Personal Best) */}
            <div
              data-testid="clinical-milestone-navigator"
              className="shrink-0 grid grid-cols-1 gap-2 md:grid-cols-3"
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

          {/* 60FPS Collision-Free Multi-Exercise ROM & 0-6 Progression Card */}
          <div className="flex-1 min-h-0 flex flex-col overflow-hidden rounded-2xl border-2 border-sky-200/80 bg-white p-3 shadow-sm gap-2">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <h3 className="text-xs font-black text-slate-900">
                {chartViewMode === 'COMPARE'
                  ? 'เปรียบเทียบ 2 รอบ (Visit-to-Visit)'
                  : 'กราฟพัฒนาการ (ROM & 0-6)'}
              </h3>

              <div className="flex flex-wrap items-center gap-1.5">
                <div className="inline-flex flex-wrap rounded-xl border border-sky-200 bg-sky-50/70 p-0.5 text-[10.5px] font-bold gap-0.5">
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
                    พัฒนาการ (ROM &amp; 0-6)
                  </button>
                  {pairComparison && (
                    <button
                      type="button"
                      data-testid="chart-tab-compare"
                      onClick={() => setChartViewMode('COMPARE')}
                      className={`rounded-lg px-2 py-1 cursor-pointer transition-colors ${
                        chartViewMode === 'COMPARE'
                          ? 'bg-teal-600 text-white shadow-xs'
                          : 'text-teal-800 hover:bg-white/80'
                      }`}
                    >
                      เปรียบเทียบ 2 รอบ
                    </button>
                  )}
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
                  <div className="flex-1 min-h-0 flex flex-col gap-2 overflow-hidden">
                    <div className="flex-1 min-h-0 flex flex-col xl:grid xl:grid-cols-12 gap-2 overflow-hidden">
                    <div className="flex-1 min-h-[160px] xl:col-span-8 w-full overflow-hidden rounded-2xl border border-slate-200/80 bg-gradient-to-b from-slate-50/40 via-white to-slate-50/30 p-2 flex items-center justify-center">
                      <svg
                        viewBox={`0 0 ${chartData.width} ${chartData.height}`}
                        preserveAspectRatio="xMidYMid meet"
                        className="w-full h-full max-h-full"
                        role="img"
                        aria-label="กราฟเปรียบเทียบพัฒนาการมุมองศาข้อต่อแยกตามท่ากายภาพและคะแนน 0-6 พร้อมวันเดือนปี"
                      >
                        <defs>
                          <linearGradient id="romAreaGrad" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stopColor="#10b981" stopOpacity="0.22" />
                            <stop offset="100%" stopColor="#10b981" stopOpacity="0.02" />
                          </linearGradient>
                        </defs>

                        {/* Shaded Clinical Target Achievement Zone */}
                        {chartData.targetRefY !== null && (
                          <rect
                            x={chartData.padL}
                            y={chartData.padT}
                            width={chartData.plotW}
                            height={Math.max(0, chartData.targetRefY - chartData.padT)}
                            fill="#10b981"
                            fillOpacity="0.07"
                          />
                        )}

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
                                y={yVal + 4}
                                textAnchor="end"
                                fill="#475569"
                                fontSize="12"
                                fontWeight="800"
                              >
                                {chartData.isPctMode ? `${leftLabel}%` : `${leftLabel}°`}
                              </text>
                              {showMahidolLine && (
                                <text
                                  x={chartData.width - chartData.padR + 8}
                                  y={yVal + 4}
                                  textAnchor="start"
                                  fill="#0284c7"
                                  fontSize="11.5"
                                  fontWeight="800"
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
                              y={Math.max(16, chartData.targetRefY - 7)}
                              textAnchor="end"
                              fill="#0369a1"
                              fontSize="11.5"
                              fontWeight="900"
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
                                fontSize="11.5"
                                fontWeight="900"
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

                    {/* Doctor ROM At-A-Glance Clinical Summary Side Panel */}
                    <div
                      data-testid="doctor-rom-brief-card"
                      className="xl:col-span-4 rounded-2xl border-2 border-sky-200/90 bg-gradient-to-br from-[#F8FCFF] via-white to-[#F0FDF9] p-3 flex flex-col justify-between gap-2"
                    >
                      <div className="space-y-1.5">
                        <div className="flex items-center justify-between gap-1.5">
                          <span className="text-[11px] font-black text-[#0F4C81]">
                            สรุปองศาข้อต่อสำหรับแพทย์
                          </span>
                          <span className="rounded-full border border-emerald-200 bg-emerald-50 px-2 py-0.5 text-[10px] font-black text-emerald-800">
                            {summary.totalVisits} ครั้งจริง
                          </span>
                        </div>
                        <div className="grid grid-cols-2 gap-1.5">
                          <div className="rounded-xl border border-sky-200/80 bg-white px-2 py-1.5">
                            <div className="text-[9.5px] font-bold text-slate-500">ROM แรกรับ → ล่าสุด</div>
                            <div className="text-xs font-black text-slate-900 tabular-nums">
                              {summary.firstRom !== null ? `${summary.firstRom.toFixed(1)}°` : '-'} →{' '}
                              {summary.latestRom !== null ? `${summary.latestRom.toFixed(1)}°` : '-'}
                            </div>
                            <div
                              className={`text-[10px] font-extrabold ${
                                (summary.deltaRom ?? 0) >= 0 ? 'text-emerald-700' : 'text-amber-700'
                              }`}
                            >
                              {summary.deltaRom !== null
                                ? `${summary.deltaRom >= 0 ? '+' : ''}${summary.deltaRom.toFixed(1)}°`
                                : 'ครั้งตั้งต้น'}
                            </div>
                          </div>
                          <div className="rounded-xl border border-emerald-200/80 bg-white px-2 py-1.5">
                            <div className="text-[9.5px] font-bold text-slate-500">เฉลี่ยเทียบเป้า / มหิดล</div>
                            <div className="text-xs font-black text-emerald-700 tabular-nums">
                              {doctorBrief.metrics.avgRomTargetPct !== null
                                ? `${doctorBrief.metrics.avgRomTargetPct.toFixed(0)}%`
                                : '-'}{' '}
                              •{' '}
                              {summary.latestScore !== null ? `${summary.latestScore}/6` : '-'}
                            </div>
                            <div className="text-[10px] font-bold text-slate-600">
                              {doctorBrief.metrics.latestPainVas !== null
                                ? `ปวด VAS ${doctorBrief.metrics.latestPainVas}/10`
                                : 'ไม่พบอาการปวด'}
                            </div>
                          </div>
                        </div>
                        <div className="rounded-xl border border-slate-200/80 bg-white/95 px-2.5 py-1.5 text-[10.5px] font-semibold text-slate-700 leading-snug">
                          <div className="text-[10px] font-black text-[#0F4C81] mb-0.5">
                            การแปลผลทางคลินิก
                          </div>
                          <p>{doctorBrief.executiveSummaryTh}</p>
                        </div>
                      </div>
                      <div className="rounded-xl border border-emerald-200/80 bg-emerald-50/50 px-2.5 py-1.5 text-[10.5px] font-bold text-emerald-950 leading-snug">
                        {doctorBrief.doctorRecommendations[0]}
                      </div>
                    </div>
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

            {/* Embedded Visit-to-Visit Comparator inside ROM Tab */}
            {chartViewMode === 'COMPARE' && pairComparison && (
              <div className="space-y-2.5">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="flex items-center gap-2 text-xs font-black text-slate-900">
                    <GitCompare className="h-4 w-4 text-indigo-600" />
                    <span>เปรียบเทียบพัฒนาการแบบเจาะลึก 2 รอบ</span>
                  </div>
                  <span className="rounded-full bg-indigo-50 border border-indigo-200 px-2.5 py-0.5 text-[10px] font-extrabold text-indigo-700">
                    ห่างกัน {pairComparison.elapsedDays} วัน
                  </span>
                </div>
                <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                  <select
                    aria-label="เลือกรอบอ้างอิง Baseline A"
                    value={pairComparison.baseline.id}
                    onChange={(e) => setCompareBaselineId(e.target.value)}
                    className="h-8 w-full rounded-xl border border-slate-200 bg-slate-50 px-2 text-[11px] font-bold text-slate-800"
                  >
                    {summary.visitsWithDelta.map(({ record }) => (
                      <option key={record.id} value={record.id}>
                        ครั้งที่ {record.visit_number} ({formatShortThaiDate(record.assessed_at)}) —{' '}
                        {record.measured_rom_deg.toFixed(1)}° | {record.initial_score_0_to_6}/6
                      </option>
                    ))}
                  </select>
                  <select
                    aria-label="เลือกรอบติดตามผล Follow-up B"
                    value={pairComparison.followUp.id}
                    onChange={(e) => setCompareFollowUpId(e.target.value)}
                    className="h-8 w-full rounded-xl border border-indigo-200 bg-indigo-50/40 px-2 text-[11px] font-bold text-slate-800"
                  >
                    {summary.visitsWithDelta.map(({ record }) => (
                      <option key={record.id} value={record.id}>
                        ครั้งที่ {record.visit_number} ({formatShortThaiDate(record.assessed_at)}) —{' '}
                        {record.measured_rom_deg.toFixed(1)}° | {record.initial_score_0_to_6}/6
                      </option>
                    ))}
                  </select>
                </div>
                <div
                  className={`rounded-xl border px-3 py-2 text-xs font-extrabold ${
                    pairComparison.verdictTone === 'emerald'
                      ? 'border-emerald-200 bg-emerald-50 text-emerald-800'
                      : pairComparison.verdictTone === 'sky'
                        ? 'border-sky-200 bg-sky-50 text-[#0F4C81]'
                        : pairComparison.verdictTone === 'rose'
                          ? 'border-amber-300 bg-amber-50 text-amber-900'
                          : 'border-amber-200 bg-amber-50 text-amber-800'
                  }`}
                >
                  {pairComparison.verdictTone === 'rose'
                    ? `ท่านทำได้ไม่เลว พยายามมากขึ้นกว่านี้นะครับ • ${pairComparison.clinicalVerdictTh}`
                    : `สรุปผลเปรียบเทียบ: ${pairComparison.clinicalVerdictTh}`}
                </div>
                <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                  <div className="rounded-xl border border-slate-200 bg-slate-50/70 p-2.5">
                    <div className="text-[10px] font-bold text-slate-500">องศา ROM</div>
                    <div className="text-xs font-black text-slate-900">
                      {pairComparison.baseline.measured_rom_deg.toFixed(1)}° → {pairComparison.followUp.measured_rom_deg.toFixed(1)}°
                    </div>
                    <div className={`text-[10.5px] font-black ${pairComparison.romDeltaDeg >= 0 ? 'text-emerald-700' : 'text-amber-700'}`}>
                      {pairComparison.romDeltaDeg >= 0 ? '+' : ''}{pairComparison.romDeltaDeg.toFixed(1)}°
                    </div>
                  </div>
                  <div className="rounded-xl border border-slate-200 bg-slate-50/70 p-2.5">
                    <div className="text-[10px] font-bold text-slate-500">% เทียบเป้าหมาย</div>
                    <div className="text-xs font-black text-slate-900">
                      {pairComparison.baselineTargetPct.toFixed(1)}% → {pairComparison.followUpTargetPct.toFixed(1)}%
                    </div>
                    <div className={`text-[10.5px] font-black ${pairComparison.targetPctDelta >= 0 ? 'text-emerald-700' : 'text-amber-700'}`}>
                      {pairComparison.targetPctDelta >= 0 ? '+' : ''}{pairComparison.targetPctDelta.toFixed(1)}%
                    </div>
                  </div>
                  <div className="rounded-xl border border-slate-200 bg-slate-50/70 p-2.5">
                    <div className="text-[10px] font-bold text-slate-500">ระดับมหิดล (0-6)</div>
                    <div className="text-xs font-black text-slate-900">
                      {pairComparison.baseline.initial_score_0_to_6}/6 → {pairComparison.followUp.initial_score_0_to_6}/6
                    </div>
                    <div className={`text-[10.5px] font-black ${pairComparison.scoreDelta >= 0 ? 'text-[#0F4C81]' : 'text-amber-700'}`}>
                      {pairComparison.scoreDelta >= 0 ? '+' : ''}{pairComparison.scoreDelta} ระดับ
                    </div>
                  </div>
                  <div className="rounded-xl border border-slate-200 bg-slate-50/70 p-2.5">
                    <div className="text-[10px] font-bold text-slate-500">ระดับปวด (VAS)</div>
                    <div className="text-xs font-black text-slate-900">
                      {pairComparison.painDelta !== null
                        ? `${pairComparison.baseline.pain_vas_0_to_10}/10 → ${pairComparison.followUp.pain_vas_0_to_10}/10`
                        : 'ไม่ได้ระบุ'}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
          </div>

          {/* TAB PANEL 3: Dedicated Minigame Cognitive Progress Chart */}
          <div
            role="tabpanel"
            data-testid="workspace-panel-MINIGAME"
            hidden={activeWorkspaceTab !== 'MINIGAME'}
            className={
              activeWorkspaceTab === 'MINIGAME'
                ? 'flex-1 min-h-0 flex flex-col overflow-hidden animate-view-enter'
                : 'hidden'
            }
          >
            <div
              data-testid="minigame-progress-chart"
              className="flex-1 min-h-0 flex flex-col overflow-hidden rounded-2xl border-2 border-indigo-200/80 bg-white p-3 shadow-sm gap-2"
            >
              <div className="flex flex-wrap items-center justify-between gap-2 border-b border-indigo-100 pb-2">
                <div className="flex items-center gap-2">
                  <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-indigo-100 text-indigo-700">
                    <Gamepad2 className="h-4 w-4" />
                  </span>
                  <h3 className="text-xs sm:text-sm font-black text-slate-900">
                    กราฟพัฒนาการมินิเกมฝึกสมอง (ความแม่นยำ %)
                  </h3>
                </div>
                <span
                  className={`rounded-full border px-2.5 py-0.5 text-[10.5px] font-black ${verdictBadgeClasses(unifiedSummary.minigame.verdict.tone)}`}
                >
                  {unifiedSummary.minigame.verdict.badgeTh}
                </span>
              </div>

              {minigameChartData.bars.length === 0 ? (
                <div className="rounded-2xl border border-dashed border-indigo-200 bg-indigo-50/40 py-10 px-4 text-center space-y-1.5">
                  <div className="text-xs font-black text-indigo-900">
                    ยังไม่มีประวัติการเล่นมินิเกมฝึกสมองสำหรับผู้ป่วยรายนี้
                  </div>
                  <p className="text-[11px] text-slate-500 max-w-md mx-auto">
                    เมื่อผู้ป่วยเล่นมินิเกมการ์ดกายภาพที่ตู้ Kiosk ระบบจะบันทึกคะแนนและวิเคราะห์ว่าดีขึ้น เท่าเดิม หรือควรพยายามเพิ่มให้อัตโนมัติ
                  </p>
                </div>
              ) : (
                <>
                  <div
                    className={`rounded-xl border px-3 py-2 text-xs flex flex-wrap items-center justify-between gap-2 ${verdictBannerClasses(unifiedSummary.minigame.verdict.tone)}`}
                  >
                    <div>
                      <div className="font-black text-slate-900">
                        {unifiedSummary.minigame.verdict.headlineTh}
                      </div>
                      <div className="text-[11px] font-semibold text-slate-600">
                        {unifiedSummary.minigame.verdict.detailTh}
                      </div>
                    </div>
                  </div>

                  <div className="flex-1 min-h-0 flex flex-col xl:grid xl:grid-cols-12 gap-2 overflow-hidden">
                  <div className="flex-1 min-h-[160px] xl:col-span-8 w-full overflow-hidden rounded-2xl border border-indigo-100 bg-gradient-to-b from-indigo-50/25 via-white to-white p-2 flex items-center justify-center">
                    <svg
                      viewBox={`0 0 ${minigameChartData.width} ${minigameChartData.height}`}
                      preserveAspectRatio="xMidYMid meet"
                      className="w-full h-full max-h-full"
                      role="img"
                      aria-label="กราฟความแม่นยำการเล่นมินิเกมฝึกสมองรายรอบ"
                    >
                      <defs>
                        <linearGradient id="minigameBarGrad" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="#4f46e5" stopOpacity="0.92" />
                          <stop offset="100%" stopColor="#0284c7" stopOpacity="0.86" />
                        </linearGradient>
                      </defs>

                      {/* 80-100% Cognitive Excellence Zone */}
                      <rect
                        x={minigameChartData.padL}
                        y={minigameChartData.padT}
                        width={minigameChartData.plotW}
                        height={minigameChartData.target80Y - minigameChartData.padT}
                        fill="#10b981"
                        fillOpacity="0.08"
                      />

                      {[0, 25, 50, 75, 100].map((tick) => {
                        const yVal =
                          minigameChartData.padT + minigameChartData.plotH * (1 - tick / 100);
                        return (
                          <g key={tick}>
                            <line
                              x1={minigameChartData.padL}
                              y1={yVal}
                              x2={minigameChartData.width - minigameChartData.padR}
                              y2={yVal}
                              stroke={tick === 0 ? '#94a3b8' : '#e2e8f0'}
                              strokeDasharray={tick === 0 ? undefined : '3 3'}
                              strokeWidth="1"
                            />
                            <text
                              x={minigameChartData.padL - 8}
                              y={yVal + 3.5}
                              textAnchor="end"
                              fill="#475569"
                              fontSize="9.5"
                              fontWeight="700"
                            >
                              {tick}%
                            </text>
                          </g>
                        );
                      })}

                      {/* 80% Target Line */}
                      <line
                        x1={minigameChartData.padL}
                        y1={minigameChartData.target80Y}
                        x2={minigameChartData.width - minigameChartData.padR}
                        y2={minigameChartData.target80Y}
                        stroke="#10b981"
                        strokeDasharray="6 4"
                        strokeWidth="1.4"
                      />
                      <text
                        x={minigameChartData.width - minigameChartData.padR - 4}
                        y={minigameChartData.target80Y - 5}
                        textAnchor="end"
                        fill="#047857"
                        fontSize="9"
                        fontWeight="800"
                      >
                        เป้าหมายดีเยี่ยม 80%
                      </text>

                      {minigameChartData.bars.map((bar) => (
                        <g key={bar.id}>
                          <rect
                            x={bar.x}
                            y={bar.y}
                            width={bar.barW}
                            height={bar.barH}
                            rx="8"
                            fill="url(#minigameBarGrad)"
                          />
                          <text
                            x={bar.cx}
                            y={Math.max(14, bar.y - 7)}
                            textAnchor="middle"
                            fill="#3730a3"
                            fontSize="10"
                            fontWeight="900"
                          >
                            {bar.accuracyPct}% ({bar.correct}/{bar.total})
                          </text>
                          <text
                            x={bar.cx}
                            y={minigameChartData.padT + minigameChartData.plotH + 17}
                            textAnchor="middle"
                            fill="#1e293b"
                            fontSize="9.5"
                            fontWeight="800"
                          >
                            รอบที่ {bar.roundNumber}
                          </text>
                          <text
                            x={bar.cx}
                            y={minigameChartData.padT + minigameChartData.plotH + 31}
                            textAnchor="middle"
                            fill="#64748b"
                            fontSize="8.8"
                            fontWeight="700"
                          >
                            {bar.dateShortTh}
                          </text>
                          {bar.deltaAccuracyFromPrev !== null && (
                            <text
                              x={bar.cx}
                              y={minigameChartData.padT + minigameChartData.plotH + 45}
                              textAnchor="middle"
                              fill={bar.deltaAccuracyFromPrev >= 0 ? '#059669' : '#d97706'}
                              fontSize="9"
                              fontWeight="800"
                            >
                              {bar.deltaAccuracyFromPrev >= 0 ? '+' : ''}
                              {bar.deltaAccuracyFromPrev}%
                            </text>
                          )}
                        </g>
                      ))}

                      {minigameChartData.bars.length >= 2 && (
                        <polyline
                          fill="none"
                          stroke="#4f46e5"
                          strokeWidth="2.4"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          points={minigameChartData.trendPoints}
                        />
                      )}
                    </svg>
                  </div>

                  {/* Doctor Cognitive Minigame Brief Side Panel */}
                  <div
                    data-testid="doctor-minigame-brief-card"
                    className="xl:col-span-4 rounded-2xl border-2 border-indigo-200/80 bg-gradient-to-br from-indigo-50/40 via-white to-sky-50/40 p-3 flex flex-col justify-between gap-2"
                  >
                    <div className="space-y-1.5">
                      <div className="text-[11px] font-black text-indigo-950">
                        สรุปสมรรถภาพสมองและสมาธิ
                      </div>
                      <div className="grid grid-cols-2 gap-1.5">
                        <div className="rounded-xl border border-indigo-200/80 bg-white px-2 py-1.5">
                          <div className="text-[9.5px] font-bold text-slate-500">ความแม่นยำล่าสุด</div>
                          <div className="text-sm font-black text-indigo-700 tabular-nums">
                            {unifiedSummary.minigame.latestAccuracyPct ?? 0}%
                          </div>
                          <div className="text-[10px] font-bold text-slate-600">
                            {unifiedSummary.minigame.latestSession
                              ? `${unifiedSummary.minigame.latestSession.correct}/${unifiedSummary.minigame.latestSession.total} ข้อ`
                              : '-'}
                          </div>
                        </div>
                        <div className="rounded-xl border border-sky-200/80 bg-white px-2 py-1.5">
                          <div className="text-[9.5px] font-bold text-slate-500">เฉลี่ย / สูงสุด</div>
                          <div className="text-sm font-black text-[#0F4C81] tabular-nums">
                            {unifiedSummary.minigame.averageAccuracyPct?.toFixed(0) ?? '-'}% /{' '}
                            {unifiedSummary.minigame.bestAccuracyPct ?? '-'}%
                          </div>
                          <div className="text-[10px] font-bold text-slate-600">
                            รวม {unifiedSummary.minigame.totalRounds} รอบ
                          </div>
                        </div>
                      </div>
                      <p className="rounded-xl border border-indigo-100 bg-white/95 px-2.5 py-1.5 text-[10.5px] font-semibold text-slate-700 leading-snug">
                        {unifiedSummary.minigame.verdict.detailTh}
                      </p>
                    </div>
                    <div className="rounded-xl border border-indigo-200 bg-indigo-50/60 px-2.5 py-1.5 text-[10.5px] font-bold text-indigo-950 leading-snug">
                      {unifiedSummary.minigame.verdict.adviceTh}
                    </div>
                  </div>
                  </div>
                </>
              )}
            </div>
          </div>

          {/* TAB PANEL 4: Monthly Score Summary Bar Chart + Monthly Cards */}
          <div
            role="tabpanel"
            data-testid="workspace-panel-MONTHLY"
            hidden={activeWorkspaceTab !== 'MONTHLY'}
            className={
              activeWorkspaceTab === 'MONTHLY'
                ? 'flex-1 min-h-0 flex flex-col overflow-hidden animate-view-enter'
                : 'hidden'
            }
          >
            <div
              data-testid="monthly-score-summary-card"
              className="flex-1 min-h-0 flex flex-col overflow-hidden rounded-2xl border-2 border-teal-200/80 bg-white p-3 shadow-sm gap-2"
            >
              <div className="flex flex-wrap items-center justify-between gap-2 border-b border-teal-100 pb-2">
                <div className="flex items-center gap-2">
                  <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-teal-100 text-teal-700">
                    <BarChart3 className="h-4 w-4" />
                  </span>
                  <div>
                    <h3 className="text-xs sm:text-sm font-black text-slate-900">
                      กราฟสรุปรายเดือน (ผลคะแนน) — {monthlySummary.length} เดือน
                    </h3>
                    <p className="text-[10.5px] font-semibold text-slate-500">
                      คลิกเลือกเดือนเพื่อดูประวัติเฉพาะเดือนนั้นได้ทันที
                    </p>
                  </div>
                </div>
                <div className="flex flex-wrap items-center gap-1.5">
                  {selectedMonthKey && (
                    <>
                      <button
                        type="button"
                        onClick={() => setActiveWorkspaceTab('TIMELINE')}
                        className="rounded-lg bg-emerald-600 px-2.5 py-1 text-[10.5px] font-extrabold text-white hover:bg-emerald-500 cursor-pointer"
                      >
                        ดูประวัติเดือน {formatThaiMonthYear(selectedMonthKey, 'short')}
                      </button>
                      <button
                        type="button"
                        onClick={() => setSelectedMonthKey(null)}
                        className="rounded-lg border border-slate-200 bg-slate-50 px-2.5 py-1 text-[10.5px] font-bold text-slate-700 hover:bg-slate-100 cursor-pointer"
                      >
                        แสดงทุกเดือน
                      </button>
                    </>
                  )}
                </div>
              </div>

              <div data-testid="monthly-score-summary-chart" className="flex-1 min-h-0 flex flex-col overflow-hidden">
                {monthlyChartData.bars.length === 0 ? (
                  <div className="flex-1 flex items-center justify-center rounded-2xl border border-dashed border-slate-200 bg-slate-50/70 py-8 px-4 text-center text-xs text-slate-500">
                    ยังไม่มีข้อมูลคะแนนประเมินรายเดือนสำหรับผู้ป่วยรายนี้
                  </div>
                ) : (
                  <div className="flex-1 min-h-0 flex flex-col xl:grid xl:grid-cols-12 gap-2 overflow-hidden">
                  <div className="flex-1 min-h-[160px] xl:col-span-8 w-full overflow-hidden rounded-2xl border border-teal-100 bg-gradient-to-b from-teal-50/20 via-white to-white p-2 flex items-center justify-center">
                    <svg
                      viewBox={`0 0 ${monthlyChartData.width} ${monthlyChartData.height}`}
                      preserveAspectRatio="xMidYMid meet"
                      className="w-full h-full max-h-full"
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

                      {/* Mahidol 5-6 Excellence Band & 3-4 Progress Band */}
                      <rect
                        x={monthlyChartData.padL}
                        y={monthlyChartData.padT}
                        width={monthlyChartData.plotW}
                        height={monthlyChartData.plotH * (1 / 6)}
                        fill="#10b981"
                        fillOpacity="0.08"
                      />
                      <rect
                        x={monthlyChartData.padL}
                        y={monthlyChartData.padT + monthlyChartData.plotH * (1 / 6)}
                        width={monthlyChartData.plotW}
                        height={monthlyChartData.plotH * (2 / 6)}
                        fill="#0284c7"
                        fillOpacity="0.04"
                      />

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
                              y={Math.max(14, bar.y - 7)}
                              textAnchor="middle"
                              fill="#065f46"
                              fontSize="10"
                              fontWeight="900"
                            >
                              {bar.avgScore0To6.toFixed(1)} / 6 คะแนน
                            </text>
                            <text
                              x={bar.cx}
                              y={monthlyChartData.padT + monthlyChartData.plotH + 16}
                              textAnchor="middle"
                              fill={isSelectedMonth ? '#0F4C81' : '#1e293b'}
                              fontSize="10"
                              fontWeight="800"
                            >
                              {bar.monthLabelShortTh}
                            </text>
                            <text
                              x={bar.cx}
                              y={monthlyChartData.padT + monthlyChartData.plotH + 29}
                              textAnchor="middle"
                              fill="#64748b"
                              fontSize="8.8"
                              fontWeight="700"
                            >
                              {bar.sessionCount} ครั้ง • สูงสุด {bar.maxScore0To6}/6
                            </text>
                            {bar.monthDeltaScore !== null && (
                              <text
                                x={bar.cx}
                                y={monthlyChartData.padT + monthlyChartData.plotH + 42}
                                textAnchor="middle"
                                fill={bar.monthDeltaScore >= 0 ? '#059669' : '#e11d48'}
                                fontSize="8.8"
                                fontWeight="800"
                              >
                                {bar.monthDeltaScore >= 0 ? '+' : ''}
                                {bar.monthDeltaScore.toFixed(1)} คะแนน
                              </text>
                            )}
                          </g>
                        );
                      })}

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

                  {/* Doctor Monthly Trend Executive Summary Side Panel */}
                  <div className="xl:col-span-4 rounded-2xl border-2 border-teal-200/80 bg-gradient-to-br from-teal-50/40 via-white to-sky-50/35 p-3 flex flex-col justify-between gap-2">
                    <div className="space-y-1.5">
                      <div className="text-[11px] font-black text-teal-950">
                        สรุปพัฒนาการรายเดือนสำหรับแพทย์
                      </div>
                      {monthlySummary.length > 0 && (() => {
                        const firstM = monthlySummary[0];
                        const latestM = monthlySummary[monthlySummary.length - 1];
                        const overallMonthDelta =
                          monthlySummary.length >= 2
                            ? Math.round((latestM.avgScore0To6 - firstM.avgScore0To6) * 10) / 10
                            : null;
                        return (
                          <div className="grid grid-cols-2 gap-1.5">
                            <div className="rounded-xl border border-teal-200/80 bg-white px-2 py-1.5">
                              <div className="text-[9.5px] font-bold text-slate-500">เดือนล่าสุด</div>
                              <div className="text-sm font-black text-teal-800 tabular-nums">
                                {latestM.avgScore0To6.toFixed(1)}/6
                              </div>
                              <div className="text-[10px] font-bold text-slate-600 truncate">
                                {latestM.monthLabelShortTh} ({latestM.sessionCount} ครั้ง)
                              </div>
                            </div>
                            <div className="rounded-xl border border-sky-200/80 bg-white px-2 py-1.5">
                              <div className="text-[9.5px] font-bold text-slate-500">เปลี่ยนแปลงรวม</div>
                              <div
                                className={`text-sm font-black tabular-nums ${
                                  (overallMonthDelta ?? 0) >= 0 ? 'text-emerald-700' : 'text-amber-700'
                                }`}
                              >
                                {overallMonthDelta !== null
                                  ? `${overallMonthDelta >= 0 ? '+' : ''}${overallMonthDelta.toFixed(1)}`
                                  : 'เดือนแรก'}
                              </div>
                              <div className="text-[10px] font-bold text-slate-600">
                                ROM เฉลี่ย {latestM.avgRomDeg.toFixed(1)}°
                              </div>
                            </div>
                          </div>
                        );
                      })()}
                      <p className="rounded-xl border border-teal-100 bg-white/95 px-2.5 py-1.5 text-[10.5px] font-semibold text-slate-700 leading-snug">
                        เกณฑ์มหิดล: ระดับ 5-6 = ควบคุมดีมาก, ระดับ 3-4 = ทำได้แต่ต้องฝึกเพิ่มความคล่องตัว, ระดับ 0-2 = ต้องดูแลประคองใกล้ชิด
                      </p>
                    </div>
                    <div className="rounded-xl border border-teal-200 bg-teal-50/60 px-2.5 py-1.5 text-[10.5px] font-bold text-teal-950 leading-snug">
                      {doctorBrief.doctorRecommendations[0]}
                    </div>
                  </div>
                  </div>
                )}
              </div>

              {monthlySummary.length > 0 && (
                <div className="shrink-0 space-y-1.5 pt-1">
                  {monthlyTotalPages > 1 && (
                    <div className="flex items-center justify-end gap-1">
                      <button
                        type="button"
                        onClick={() => setMonthlyPage((p) => (p - 1 + monthlyTotalPages) % monthlyTotalPages)}
                        className="rounded-lg border border-slate-200 bg-white px-2 py-0.5 text-[10px] font-black text-slate-600 hover:bg-teal-50 cursor-pointer"
                      >
                        ก่อนหน้า
                      </button>
                      <span className="text-[10px] font-black text-slate-600 tabular-nums">
                        {safeMonthlyPage + 1}/{monthlyTotalPages}
                      </span>
                      <button
                        type="button"
                        onClick={() => setMonthlyPage((p) => (p + 1) % monthlyTotalPages)}
                        className="rounded-lg border border-slate-200 bg-white px-2 py-0.5 text-[10px] font-black text-slate-600 hover:bg-teal-50 cursor-pointer"
                      >
                        ถัดไป
                      </button>
                    </div>
                  )}
                  <div className="grid grid-cols-2 gap-1.5 sm:grid-cols-3 overflow-hidden">
                    {visibleMonthlyCards.map((m) => {
                    const isSelectedMonth = selectedMonthKey === m.monthKey;
                    return (
                      <button
                        key={m.monthKey}
                        type="button"
                        onClick={() =>
                          setSelectedMonthKey((prev) => (prev === m.monthKey ? null : m.monthKey))
                        }
                        className={`btn-tactile text-left rounded-xl border p-2.5 transition-all space-y-1 cursor-pointer ${
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
                            {m.sessionCount} ครั้ง • เป้า {m.avgTargetPct.toFixed(0)}%
                          </span>
                          <span>ROM {m.avgRomDeg.toFixed(1)}°</span>
                        </div>
                      </button>
                    );
                  })}
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* TAB PANEL 5: Unified Per-Exercise & Minigame Breakdown Comparison Table (Zero-Scroll Paged) */}
          <div
            role="tabpanel"
            data-testid="workspace-panel-TABLE"
            hidden={activeWorkspaceTab !== 'TABLE'}
            className={
              activeWorkspaceTab === 'TABLE'
                ? 'flex-1 min-h-0 flex flex-col overflow-hidden animate-view-enter'
                : 'hidden'
            }
          >
            <div
              data-testid="unified-exercise-minigame-table"
              className="flex-1 min-h-0 flex flex-col overflow-hidden rounded-2xl border-2 border-sky-200/80 bg-white p-3 shadow-sm gap-2"
            >
              <div className="shrink-0 flex flex-wrap items-center justify-between gap-2 border-b border-slate-100 pb-2">
                <div className="flex items-center gap-2">
                  <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-sky-100 text-[#0F4C81]">
                    <Table2 className="h-4 w-4" />
                  </span>
                  <h3 className="text-xs sm:text-sm font-black text-slate-900">
                    สรุปแยกรายท่ากายภาพและมินิเกม ({overallBreakdown.length + (unifiedSummary.minigame.totalRounds > 0 ? 1 : 0)} กลุ่ม)
                  </h3>
                </div>
                <div className="flex items-center gap-2">
                  <span className="hidden xl:inline text-[11px] font-semibold text-slate-500">
                    เทียบแรกรับ vs ล่าสุด • กดปุ่มกราฟเพื่อดูเส้นพัฒนาการ
                  </span>
                  {tableTotalPages > 1 && (
                    <div className="inline-flex items-center rounded-lg border border-slate-200 bg-slate-50 p-0.5">
                      <button
                        type="button"
                        aria-label="หน้าตารางก่อนหน้า"
                        onClick={() => setTablePage((p) => (p - 1 + tableTotalPages) % tableTotalPages)}
                        className="flex h-5 w-5 items-center justify-center rounded text-slate-600 hover:bg-white hover:text-[#0F4C81] cursor-pointer"
                      >
                        <ChevronLeft className="h-3.5 w-3.5" />
                      </button>
                      <span className="px-1.5 text-[10.5px] font-black text-slate-700 tabular-nums">
                        {safeTablePage + 1}/{tableTotalPages}
                      </span>
                      <button
                        type="button"
                        aria-label="หน้าตารางถัดไป"
                        onClick={() => setTablePage((p) => (p + 1) % tableTotalPages)}
                        className="flex h-5 w-5 items-center justify-center rounded text-slate-600 hover:bg-white hover:text-[#0F4C81] cursor-pointer"
                      >
                        <ChevronRight className="h-3.5 w-3.5" />
                      </button>
                    </div>
                  )}
                </div>
              </div>

              {overallBreakdown.length === 0 && unifiedSummary.minigame.totalRounds === 0 ? (
                <div className="flex-1 flex items-center justify-center rounded-2xl border-2 border-dashed border-slate-200 bg-slate-50/70 py-10 px-4 text-center text-xs font-semibold text-slate-500">
                  ยังไม่มีข้อมูลเปรียบเทียบรายท่าสำหรับผู้สูงอายุรายนี้
                </div>
              ) : (
                <div className="flex-1 min-h-0 overflow-hidden">
                  <table className="w-full text-left border-collapse text-xs">
                    <thead>
                      <tr className="border-b border-slate-200 text-[10.5px] font-extrabold text-slate-500 bg-white">
                        <th className="py-1.5 pr-2">ท่า / กิจกรรม</th>
                        <th className="py-1.5 px-2 text-right">แรกรับ → ล่าสุด (Δ)</th>
                        <th className="py-1.5 px-2 text-center">ระดับ 0-6</th>
                        <th className="py-1.5 px-2 text-right">ความสำเร็จเทียบเป้า</th>
                        <th className="py-1.5 pl-2 text-right">สรุปแนวโน้ม</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100">
                      {visibleBreakdownRows.map((item) => {
                        const rowVerdict = resolveExerciseRowVerdict(item);
                        const isFocused =
                          exerciseFilter === item.exerciseCode &&
                          (sideFilter === 'ALL' || sideFilter === item.targetSide);
                        const pctClamped = Math.min(100, Math.max(0, item.latestTargetAchievementPct));
                        return (
                          <tr
                            key={item.key}
                            className={`transition-colors ${
                              isFocused ? 'bg-emerald-50/50' : 'hover:bg-slate-50/80'
                            }`}
                          >
                            <td className="py-2 pr-2 font-bold text-slate-900">
                              <div className="flex items-center gap-1.5">
                                <span className="truncate max-w-[230px]">
                                  {formatFriendlyExerciseLabel(item.exerciseNameTh, item.exerciseCode)}
                                </span>
                                <span className="shrink-0 rounded bg-sky-100 px-1.5 py-0.5 text-[10px] font-extrabold text-[#0F4C81]">
                                  {formatSideTh(item.targetSide)}
                                </span>
                                <span className="shrink-0 text-[10px] font-semibold text-slate-400">
                                  ({item.visitCount} ครั้ง)
                                </span>
                              </div>
                            </td>
                            <td className="py-2 px-2 text-right tabular-nums whitespace-nowrap">
                              <span className="font-semibold text-slate-500">
                                {item.firstRomDeg.toFixed(1)}°
                              </span>
                              <span className="mx-1 text-slate-400">→</span>
                              <span className="font-black text-slate-900">
                                {item.latestRomDeg.toFixed(1)}°
                              </span>
                              <span
                                className={`ml-1.5 rounded-full px-1.5 py-0.5 text-[9.5px] font-extrabold ${
                                  item.deltaRomDeg >= 0
                                    ? 'bg-emerald-50 text-emerald-700'
                                    : 'bg-amber-50 text-amber-800'
                                }`}
                              >
                                {item.deltaRomDeg >= 0 ? '+' : ''}
                                {item.deltaRomDeg.toFixed(1)}°
                              </span>
                            </td>
                            <td className="py-2 px-2 text-center whitespace-nowrap">
                              <span className="rounded-full bg-[#0F4C81] px-2 py-0.5 text-[10px] font-black text-white tabular-nums">
                                {item.latestScore}/6
                              </span>
                            </td>
                            <td className="py-2 px-2 text-right whitespace-nowrap">
                              <div className="inline-flex items-center gap-2">
                                <div className="hidden sm:block w-16 h-2 overflow-hidden rounded-full bg-slate-100">
                                  <div
                                    className={`h-full rounded-full ${
                                      item.latestTargetAchievementPct >= 80
                                        ? 'bg-gradient-to-r from-emerald-500 to-teal-400'
                                        : 'bg-gradient-to-r from-sky-500 to-amber-400'
                                    }`}
                                    style={{ width: `${pctClamped}%` }}
                                  />
                                </div>
                                <span className="font-extrabold text-emerald-700 tabular-nums">
                                  ({item.latestTargetAchievementPct.toFixed(1)}%)
                                </span>
                              </div>
                            </td>
                            <td className="py-2 pl-2 text-right whitespace-nowrap">
                              <div className="inline-flex items-center gap-1.5">
                                <span
                                  className={`rounded-full border px-2 py-0.5 text-[10px] font-black ${verdictBadgeClasses(rowVerdict.tone)}`}
                                >
                                  {rowVerdict.labelTh}
                                </span>
                                <button
                                  type="button"
                                  onClick={() => {
                                    setExerciseFilter(item.exerciseCode);
                                    setSideFilter(item.targetSide);
                                    setChartMetricMode('ROM_DEG');
                                    setChartViewMode('SESSIONS');
                                    setActiveWorkspaceTab('ROM');
                                  }}
                                  title="ดูกราฟท่านี้"
                                  className="rounded-lg border border-sky-200 bg-sky-50 px-2 py-0.5 text-[10px] font-extrabold text-[#0F4C81] hover:bg-sky-100 cursor-pointer"
                                >
                                  กราฟ
                                </button>
                              </div>
                            </td>
                          </tr>
                        );
                      })}

                      {unifiedSummary.minigame.totalRounds > 0 &&
                        unifiedSummary.minigame.firstSession &&
                        unifiedSummary.minigame.latestSession && (
                          <tr
                            data-testid="unified-table-minigame-row"
                            className="bg-indigo-50/35 hover:bg-indigo-50/70 transition-colors"
                          >
                            <td className="py-2 pr-2 font-bold text-indigo-950">
                              <div className="flex items-center gap-1.5">
                                <Gamepad2 className="h-3.5 w-3.5 text-indigo-600 shrink-0" />
                                <span className="truncate max-w-[220px]">
                                  มินิเกมฝึกสมอง ({unifiedSummary.minigame.totalRounds} รอบ)
                                </span>
                              </div>
                            </td>
                            <td className="py-2 px-2 text-right tabular-nums whitespace-nowrap">
                              <span className="font-semibold text-slate-600">
                                {unifiedSummary.minigame.firstSession.accuracyPct}%
                              </span>
                              <span className="mx-1 text-slate-400">→</span>
                              <span className="font-black text-indigo-900">
                                {unifiedSummary.minigame.latestSession.accuracyPct}% (
                                {unifiedSummary.minigame.latestSession.correct}/
                                {unifiedSummary.minigame.latestSession.total})
                              </span>
                            </td>
                            <td className="py-2 px-2 text-center whitespace-nowrap">
                              <span className="rounded-full bg-indigo-600 px-2 py-0.5 text-[10px] font-black text-white">
                                สมอง
                              </span>
                            </td>
                            <td className="py-2 px-2 text-right font-extrabold text-indigo-700 tabular-nums whitespace-nowrap">
                              (เฉลี่ย {unifiedSummary.minigame.averageAccuracyPct?.toFixed(0)}%)
                            </td>
                            <td className="py-2 pl-2 text-right whitespace-nowrap">
                              <div className="inline-flex items-center gap-1.5">
                                <span
                                  className={`rounded-full border px-2 py-0.5 text-[10px] font-black ${verdictBadgeClasses(unifiedSummary.minigame.verdict.tone)}`}
                                >
                                  {unifiedSummary.minigame.verdict.badgeTh}
                                </span>
                                <button
                                  type="button"
                                  onClick={() => setActiveWorkspaceTab('MINIGAME')}
                                  title="ดูกราฟมินิเกม"
                                  className="rounded-lg border border-indigo-200 bg-indigo-50 px-2 py-0.5 text-[10px] font-extrabold text-indigo-800 hover:bg-indigo-100 cursor-pointer"
                                >
                                  กราฟ
                                </button>
                              </div>
                            </td>
                          </tr>
                        )}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </div>

          {/* TAB PANEL 6: Unified Chronological Rehab & Minigame Timeline (Zero-Scroll Paged) */}
          <div
            role="tabpanel"
            data-testid="workspace-panel-TIMELINE"
            hidden={activeWorkspaceTab !== 'TIMELINE'}
            className={
              activeWorkspaceTab === 'TIMELINE'
                ? 'flex-1 min-h-0 flex flex-col overflow-hidden animate-view-enter'
                : 'hidden'
            }
          >
          <div
            data-testid="unified-activity-timeline"
            className="flex-1 min-h-0 flex flex-col overflow-hidden rounded-2xl border-2 border-slate-200 bg-white p-3 shadow-sm gap-2"
          >
            <div className="shrink-0 flex flex-wrap items-center justify-between gap-2 border-b border-slate-100 pb-2">
              <div className="flex items-center gap-2 min-w-0">
                <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-sky-100 text-[#0F4C81]">
                  <History className="h-3.5 w-3.5" />
                </span>
                <div>
                  <h3 className="text-xs sm:text-sm font-black text-slate-900 truncate">
                    ประวัติการฝึก &amp; มินิเกม ({filteredUnifiedTimeline.length} ครั้ง)
                  </h3>
                  <p className="text-[10px] font-semibold text-slate-500">
                    เรียงลำดับจากครั้งล่าสุด พร้อมคะแนนพัฒนาการและองศาที่ทำได้
                  </p>
                </div>
              </div>
              <div className="flex flex-wrap items-center gap-1.5">
                {timelineTotalPages > 1 && (
                  <div className="inline-flex items-center rounded-lg border border-slate-200 bg-slate-50 p-0.5">
                    <button
                      type="button"
                      aria-label="ประวัติหน้าก่อนหน้า"
                      onClick={() => setTimelinePage((p) => (p - 1 + timelineTotalPages) % timelineTotalPages)}
                      className="flex h-6 w-6 items-center justify-center rounded text-slate-600 hover:bg-white hover:text-[#0F4C81] cursor-pointer"
                    >
                      <ChevronLeft className="h-3.5 w-3.5" />
                    </button>
                    <span className="px-1.5 text-[10.5px] font-black text-slate-700 tabular-nums">
                      {safeTimelinePage + 1}/{timelineTotalPages}
                    </span>
                    <button
                      type="button"
                      aria-label="ประวัติหน้าถัดไป"
                      onClick={() => setTimelinePage((p) => (p + 1) % timelineTotalPages)}
                      className="flex h-6 w-6 items-center justify-center rounded text-slate-600 hover:bg-white hover:text-[#0F4C81] cursor-pointer"
                    >
                      <ChevronRight className="h-3.5 w-3.5" />
                    </button>
                  </div>
                )}
                {selectedMonthKey && (
                  <button
                    type="button"
                    onClick={() => setSelectedMonthKey(null)}
                    className="rounded-full border border-emerald-300 bg-emerald-50 px-2.5 py-0.5 text-[10.5px] font-extrabold text-emerald-800 hover:bg-emerald-100 cursor-pointer"
                  >
                    {formatThaiMonthYear(selectedMonthKey, 'short')} • ล้าง
                  </button>
                )}
                <div className="inline-grid grid-cols-3 gap-1 rounded-xl border border-slate-200 bg-slate-50 p-0.5 text-[10.5px] font-extrabold">
                  <button
                    type="button"
                    onClick={() => setTimelineKindFilter('ALL')}
                    className={`rounded-lg px-2 py-1 cursor-pointer transition-colors ${
                      timelineKindFilter === 'ALL'
                        ? 'bg-[#0F4C81] text-white shadow-xs'
                        : 'text-slate-600 hover:bg-white'
                    }`}
                  >
                    ทั้งหมด ({unifiedSummary.timeline.length})
                  </button>
                  <button
                    type="button"
                    onClick={() => setTimelineKindFilter('PHYSIO')}
                    className={`rounded-lg px-2 py-1 cursor-pointer transition-colors ${
                      timelineKindFilter === 'PHYSIO'
                        ? 'bg-emerald-600 text-white shadow-xs'
                        : 'text-slate-600 hover:bg-white'
                    }`}
                  >
                    กายภาพ ({summary.totalVisits})
                  </button>
                  <button
                    type="button"
                    onClick={() => setTimelineKindFilter('MINIGAME')}
                    className={`rounded-lg px-2 py-1 cursor-pointer transition-colors ${
                      timelineKindFilter === 'MINIGAME'
                        ? 'bg-indigo-600 text-white shadow-xs'
                        : 'text-slate-600 hover:bg-white'
                    }`}
                  >
                    มินิเกม ({unifiedSummary.minigame.totalRounds})
                  </button>
                </div>
              </div>
            </div>

            {filteredUnifiedTimeline.length === 0 ? (
              <div className="flex-1 flex items-center justify-center rounded-2xl border-2 border-dashed border-slate-200 bg-slate-50/70 py-8 px-4 text-center text-xs font-semibold text-slate-500">
                ยังไม่มีประวัติกายภาพหรือการเล่นมินิเกมในเงื่อนไขที่เลือก
              </div>
            ) : (
              <div className="flex-1 min-h-0 grid grid-cols-1 gap-2 sm:grid-cols-2 content-start overflow-hidden">
                {visibleTimelineItems.map((item, idx) => {
                  const isMinigameItem = item.kind === 'minigame';
                  const rec = item.physioRecord;
                  const canDelete =
                    Boolean(rec) &&
                    !rec!.id.startsWith('kiosk-sess-') &&
                    !rec!.id.startsWith('hist-sess-');
                  const isSelectedCard = Boolean(rec && activeSelectedNode?.id === rec.id);
                  return (
                    <div
                      key={item.id}
                      className={`rounded-2xl border p-3 space-y-1.5 transition-colors card-hover-lift animate-card-enter ${
                        isMinigameItem
                          ? 'border-indigo-200 bg-gradient-to-br from-indigo-50/60 via-white to-sky-50/40'
                          : isSelectedCard
                            ? 'border-emerald-400 bg-emerald-50/45 ring-1 ring-emerald-200'
                            : idx === 0
                              ? 'border-emerald-300 bg-emerald-50/35'
                              : 'border-slate-200 bg-slate-50/50 hover:border-sky-300'
                      }`}
                    >
                      <div className="flex items-start justify-between gap-1.5">
                        <div className="min-w-0">
                          <div className="flex flex-wrap items-center gap-1">
                            <span
                              className={`rounded-md px-1.5 py-0.5 text-[9.5px] font-black text-white ${
                                isMinigameItem ? 'bg-indigo-600' : 'bg-[#0F4C81]'
                              }`}
                            >
                              {item.sequenceLabel}
                              {idx === 0 ? ' (ล่าสุด)' : ''}
                            </span>
                            <span
                              className={`rounded-full border px-1.5 py-0.5 text-[9.5px] font-black ${verdictBadgeClasses(item.verdict.tone)}`}
                            >
                              {item.verdict.badgeTh}
                            </span>
                          </div>
                          <div className="mt-1 text-[10.5px] font-bold text-slate-600">
                            {item.dateFullTh}
                          </div>
                        </div>

                        <div className="flex items-center gap-1 shrink-0">
                          <span
                            className={`rounded-full px-2 py-0.5 text-[10px] font-black text-white ${
                              isMinigameItem ? 'bg-indigo-600' : 'bg-emerald-600'
                            }`}
                          >
                            {item.secondaryMetricLabel}
                          </span>
                          {canDelete &&
                            rec &&
                            (confirmDeleteId === rec.id ? (
                              <div className="inline-flex items-center gap-1">
                                <button
                                  type="button"
                                  disabled={deletingId === rec.id}
                                  onClick={() => void handleDeleteAssessment(rec.id)}
                                  className="rounded bg-rose-600 px-1.5 py-0.5 text-[9.5px] font-extrabold text-white cursor-pointer"
                                >
                                  ลบ
                                </button>
                                <button
                                  type="button"
                                  onClick={() => setConfirmDeleteId(null)}
                                  className="rounded border border-slate-200 bg-white px-1.5 py-0.5 text-[9.5px] font-bold text-slate-600 cursor-pointer"
                                >
                                  ยกเลิก
                                </button>
                              </div>
                            ) : (
                              <button
                                type="button"
                                onClick={() => setConfirmDeleteId(rec.id)}
                                className="rounded p-0.5 text-slate-400 hover:bg-rose-50 hover:text-rose-600 cursor-pointer"
                                title="ลบรายการประเมินนี้"
                              >
                                <Trash2 className="h-3 w-3" />
                              </button>
                            ))}
                        </div>
                      </div>

                      <div className="text-xs font-black text-slate-900 leading-snug">
                        {formatFriendlyExerciseLabel(item.titleTh, rec?.exercise_code)}
                      </div>

                      <div className="flex flex-wrap items-center justify-between gap-1 text-[10.5px]">
                        <span className="font-extrabold text-[#0F4C81]">
                          {item.primaryMetricLabel}
                        </span>
                        {item.deltaLabel && (
                          <span
                            className={`rounded-full px-1.5 py-0.5 text-[9.5px] font-black ${
                              item.deltaPositive
                                ? 'bg-emerald-100 text-emerald-800'
                                : 'bg-amber-100 text-amber-900'
                            }`}
                          >
                            {item.deltaLabel}
                          </span>
                        )}
                      </div>

                      <div className="text-[10px] font-semibold text-slate-500 truncate">
                        {item.subtitleTh}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}

            {/* Color Legend Footer */}
            <div className="flex flex-wrap items-center justify-between gap-2 border-t border-slate-100 pt-2 text-[11px] font-bold text-slate-600">
              <span className="inline-flex items-center gap-1.5">
                <span className="h-2.5 w-2.5 rounded-sm bg-[#0284C7]" />
                = องศา ROM (°)
              </span>
              <span className="inline-flex items-center gap-1.5">
                <span className="h-2.5 w-2.5 rounded-sm bg-emerald-600" />
                = ระดับกำลังกล้ามเนื้อ (0-6)
              </span>
              <span className="inline-flex items-center gap-1.5">
                <span className="h-2.5 w-2.5 rounded-sm bg-indigo-600" />
                = มินิเกมฝึกสมอง (%)
              </span>
            </div>
          </div>
          </div>
        </div>
      </div>

      {/* Patient-Friendly Weekly / Progress Summary Modal (with Comparison Bar Chart & Print) */}
      {patientSummaryReport && (
        <PatientProgressSummaryModal
          open={patientSummaryModalOpen}
          report={patientSummaryReport}
          onClose={() => setPatientSummaryModalOpen(false)}
        />
      )}

      {/* Hospital-Grade Printable Clinical Progress PDF Report + Scannable QR Code */}
      {selectedPatient && (
        <ClinicalPdfReportModal
          open={pdfReportModalOpen}
          onClose={() => setPdfReportModalOpen(false)}
          patient={selectedPatient}
          universalLinkCode={universalLinkCode}
          facilityName={facilityName}
          assessorName={currentUser.fullName}
          rehabStockIndex={rehabStockIndex}
          unifiedSummary={unifiedSummary}
          doctorBrief={doctorBrief}
          exerciseBreakdown={overallBreakdown}
          firstMahidolScore={summary.visitsWithDelta[0]?.record.initial_score_0_to_6 ?? null}
          latestMahidolScore={
            summary.visitsWithDelta[summary.visitsWithDelta.length - 1]?.record
              .initial_score_0_to_6 ?? null
          }
        />
      )}

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
