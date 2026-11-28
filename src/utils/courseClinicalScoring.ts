import type {
  PatientCourseRecord,
  PatientRecord,
  RehabScheduleItem,
  RehabSessionHistoryEntry,
  RehabSessionRecord,
} from '@drtechcares/api-client';
import { CLINICAL_EXERCISES, parseExerciseCodeAndSide } from '@drtechcares/pose-engine';
import {
  resolvePatientUniversalLinkCode,
  type AssessmentTargetSide,
  type CreatePatientClinicalAssessmentInput,
  type MahidolScore0To6,
  type PatientClinicalAssessmentRecord,
} from '../types/clinicalAssessment';

export interface CourseAutoScoreInput {
  achievedRomDeg?: number | null;
  peakRomDeg?: number | null;
  maxRomDeg?: number | null;
  targetRomDeg: number;
  completedReps: number;
  targetReps: number;
  accuracyScore?: number | null;
  sourceSessionId: string;
  exerciseCode?: string;
  exerciseNameTh?: string;
  targetSide?: AssessmentTargetSide;
  assessedAt?: string;
  facilityName?: string;
  courseTitle?: string;
  patientId?: string;
  patientHn?: string;
}

export interface CourseAutoAssessmentResult {
  calculatedScore0To6: MahidolScore0To6;
  effectiveRomDeg: number;
  targetRomDeg: number;
  romDerivedProportionally: boolean;
  romPct: number;
  completedReps: number;
  targetReps: number;
  accuracyScore: number;
  compositePct: number;
  breakdownSummaryTh: string;
  sourceSessionId: string;
  exerciseCode: string;
  exerciseNameTh: string;
  targetSide: AssessmentTargetSide;
  assessedAt: string;
  facilityName: string;
  courseTitle: string;
  patientId: string;
  patientHn: string;
}

export interface AssignedCourseExerciseDefinition {
  key: string;
  exerciseCode: string;
  exerciseNameTh: string;
  targetSide: AssessmentTargetSide;
  targetRomDeg: number;
  targetReps: number;
  targetSets: number;
  courseTitle: string;
  sources: ('patient_courses' | 'rehab_schedules' | 'patient_history')[];
  totalScheduledCount: number;
  completedScheduledCount: number;
}

export interface NormalizedCompletedCourseSession {
  sourceSessionId: string;
  patientId: string;
  patientHn: string;
  exerciseCode: string;
  exerciseNameTh: string;
  targetSide: AssessmentTargetSide;
  achievedRomDeg: number | null;
  completedReps: number;
  completedSets: number;
  accuracyScore: number | null;
  assessedAt: string;
  facilityName: string;
  origin: 'rehab_sessions' | 'rehab_session_history' | 'completed_schedule';
}

export interface AssignedCourseExerciseSummary {
  definition: AssignedCourseExerciseDefinition;
  completedSessionsCount: number;
  latestResult: CourseAutoAssessmentResult | null;
  bestRomDeg: number | null;
  averageScore0To6: number | null;
}

export interface PatientCourseAutoScoringBundle {
  patientId: string;
  patientHn: string;
  courseTitle: string | null;
  assignedExercises: AssignedCourseExerciseDefinition[];
  exerciseSummaries: AssignedCourseExerciseSummary[];
  completedSessions: NormalizedCompletedCourseSession[];
  autoResults: CourseAutoAssessmentResult[];
  unrecordedResults: CourseAutoAssessmentResult[];
  latestResult: CourseAutoAssessmentResult | null;
}

function clamp(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) return min;
  return Math.max(min, Math.min(max, value));
}

function normalizeTargetSide(raw?: string | null, fallback?: string | null): AssessmentTargetSide {
  const candidates = [raw, fallback];
  for (const cand of candidates) {
    const upper = String(cand ?? '').trim().toUpperCase();
    if (upper === 'LEFT' || upper === 'L') return 'LEFT';
    if (upper === 'RIGHT' || upper === 'R') return 'RIGHT';
    if (upper === 'BOTH' || upper === 'BILATERAL') return 'BOTH';
  }
  return 'RIGHT';
}

function findCatalogExercise(rawCode?: string | null) {
  const clean = String(rawCode ?? '').trim();
  if (!clean) return undefined;
  const parsed = parseExerciseCodeAndSide(clean);
  const upperRaw = clean.toUpperCase();
  const upperBase = parsed.baseCode.toUpperCase();
  return CLINICAL_EXERCISES.find(
    (ex) =>
      ex.code.toUpperCase() === upperBase ||
      ex.id.toUpperCase() === upperRaw ||
      ex.id.toUpperCase() === upperBase ||
      ex.code.toUpperCase() === upperRaw,
  );
}

function resolveExerciseCodeAndMeta(
  rawCode?: string | null,
  rawSide?: string | null,
  patientAffectedSide?: string | null,
): {
  exerciseCode: string;
  exerciseNameTh: string;
  targetSide: AssessmentTargetSide;
  defaultTargetRomDeg: number;
} {
  const parsed = parseExerciseCodeAndSide(rawCode || 'ELDERLY_SHOULDER_FLEXION', rawSide ?? undefined);
  const catalog = findCatalogExercise(rawCode);
  const exerciseCode = (catalog?.code || parsed.baseCode || 'ELDERLY_SHOULDER_FLEXION').toUpperCase();
  const exerciseNameTh = catalog?.name.th || rawCode || exerciseCode;
  const defaultTargetRomDeg = Number((catalog?.detection.targetMinDeg ?? 90).toFixed(1));
  const catalogSide = catalog?.detection.side === 'both' ? 'BOTH' : undefined;
  const targetSide = normalizeTargetSide(
    rawSide || (parsed.side !== 'BOTH' ? parsed.side : undefined) || catalogSide,
    patientAffectedSide,
  );
  return {
    exerciseCode,
    exerciseNameTh,
    targetSide,
    defaultTargetRomDeg,
  };
}

export function mapCompositePctToMahidolScore0To6(
  compositePct: number,
  completedReps: number,
  effectiveRomDeg: number,
): MahidolScore0To6 {
  if (completedReps <= 0 && effectiveRomDeg <= 0) return 0;
  if (!Number.isFinite(compositePct) || compositePct <= 0) return 0;
  if (compositePct <= 19) return 1;
  if (compositePct <= 39) return 2;
  if (compositePct <= 59) return 3;
  if (compositePct <= 79) return 4;
  if (compositePct <= 94) return 5;
  return 6;
}

/**
 * Deterministic 0-6 Mahidol-adapted clinical score & effective ROM calculator
 * derived from PT-assigned rehab course targets and completed course sessions.
 */
export function calculateMahidolScoreFromCourse(
  input: CourseAutoScoreInput,
): CourseAutoAssessmentResult {
  const meta = resolveExerciseCodeAndMeta(input.exerciseCode, input.targetSide);
  const safeTargetRomDeg =
    Number.isFinite(input.targetRomDeg) && input.targetRomDeg > 0
      ? Math.round(clamp(input.targetRomDeg, 10, 220) * 10) / 10
      : meta.defaultTargetRomDeg;
  const safeTargetReps = Math.max(
    1,
    Math.round(Number.isFinite(input.targetReps) && input.targetReps > 0 ? input.targetReps : 10),
  );
  const safeCompletedReps = Math.max(
    0,
    Math.round(Number.isFinite(input.completedReps) ? input.completedReps : 0),
  );

  const rawRecordedRom =
    typeof input.achievedRomDeg === 'number' && Number.isFinite(input.achievedRomDeg)
      ? input.achievedRomDeg
      : typeof input.peakRomDeg === 'number' && Number.isFinite(input.peakRomDeg)
        ? input.peakRomDeg
        : typeof input.maxRomDeg === 'number' && Number.isFinite(input.maxRomDeg)
          ? input.maxRomDeg
          : null;

  const hasExplicitRom = rawRecordedRom !== null && rawRecordedRom > 0;
  const completionRatio = clamp(safeCompletedReps / Math.max(1, safeTargetReps), 0, 1);

  let effectiveRomDeg = 0;
  let romDerivedProportionally = false;
  if (hasExplicitRom) {
    effectiveRomDeg = Math.round(clamp(rawRecordedRom, 0, 220) * 10) / 10;
  } else if (safeCompletedReps > 0) {
    effectiveRomDeg = Math.round(safeTargetRomDeg * completionRatio * 10) / 10;
    romDerivedProportionally = true;
  }

  const safeAccuracyScore =
    typeof input.accuracyScore === 'number' && Number.isFinite(input.accuracyScore)
      ? clamp(input.accuracyScore, 0, 100)
      : 85;

  let compositePct = 0;
  let calculatedScore0To6: MahidolScore0To6 = 0;

  if (safeCompletedReps === 0 && effectiveRomDeg <= 0) {
    compositePct = 0;
    calculatedScore0To6 = 0;
  } else {
    const romRatio = clamp(effectiveRomDeg / Math.max(1, safeTargetRomDeg), 0, 1.15);
    const repRatio = clamp(safeCompletedReps / Math.max(1, safeTargetReps), 0, 1.0);
    const accuracyRatio = clamp(safeAccuracyScore / 100, 0, 1.0);
    const rawComposite = Math.round((romRatio * 0.5 + repRatio * 0.3 + accuracyRatio * 0.2) * 100);
    compositePct = clamp(rawComposite, 0, 100);
    calculatedScore0To6 = mapCompositePctToMahidolScore0To6(
      compositePct,
      safeCompletedReps,
      effectiveRomDeg,
    );
  }

  const romPct = Math.round((effectiveRomDeg / Math.max(1, safeTargetRomDeg)) * 100);
  const proportionalSuffix = romDerivedProportionally
    ? ' (คำนวณองศาตามสัดส่วนความสำเร็จของคอร์ส)'
    : '';
  const breakdownSummaryTh = `คำนวณอัตโนมัติจากคอร์ส: องศาเฉลี่ย/สูงสุด ${effectiveRomDeg.toFixed(1)}° จากเป้า ${Math.round(safeTargetRomDeg)}° (${romPct}%) • ทำครบ ${safeCompletedReps}/${safeTargetReps} ครั้ง • ความแม่นยำ ${Math.round(safeAccuracyScore)}% -> ประเมินได้ระดับ ${calculatedScore0To6}/6${proportionalSuffix}`;

  return {
    calculatedScore0To6,
    effectiveRomDeg,
    targetRomDeg: safeTargetRomDeg,
    romDerivedProportionally,
    romPct,
    completedReps: safeCompletedReps,
    targetReps: safeTargetReps,
    accuracyScore: Math.round(safeAccuracyScore),
    compositePct,
    breakdownSummaryTh,
    sourceSessionId: String(input.sourceSessionId || '').trim(),
    exerciseCode: meta.exerciseCode,
    exerciseNameTh: (input.exerciseNameTh ?? '').trim() || meta.exerciseNameTh,
    targetSide: input.targetSide ? normalizeTargetSide(input.targetSide) : meta.targetSide,
    assessedAt: input.assessedAt || new Date().toISOString(),
    facilityName:
      (input.facilityName ?? '').trim() || 'ศูนย์กายภาพบำบัด Dr.Tech.Care สาขาหลัก',
    courseTitle: (input.courseTitle ?? '').trim() || 'คอร์สฟื้นฟูที่นักกายภาพจัดให้',
    patientId: (input.patientId ?? '').trim(),
    patientHn: (input.patientHn ?? '').trim(),
  };
}

function belongsStrictlyToPatient(
  patient: Pick<PatientRecord, 'id' | 'hn'>,
  recordPatientId?: string | null,
  recordHn?: string | null,
): boolean {
  const targetId = String(patient.id ?? '').trim();
  const targetHn = String(patient.hn ?? '').trim().toUpperCase();
  const recId = String(recordPatientId ?? '').trim();
  const recHn = String(recordHn ?? '').trim().toUpperCase();
  if (recId) {
    return recId === targetId;
  }
  return Boolean(targetHn && recHn && recHn === targetHn);
}

/**
 * Extracts all PT-assigned course exercises for a single patient across
 * `patient_courses`, `rehab_schedules`, and `patient.rehabSchedules` (from `patient.history`).
 */
export function extractAssignedCourseExercisesForPatient(
  patient: Pick<PatientRecord, 'id' | 'hn' | 'affectedSide' | 'rehabSchedules'> & {
    history?: { rehabSchedules?: RehabScheduleItem[] } | null;
  },
  course?: PatientCourseRecord | null,
  rehabSchedules?: readonly RehabScheduleItem[] | null,
): { courseTitle: string | null; exercises: AssignedCourseExerciseDefinition[] } {
  const byKey = new Map<string, AssignedCourseExerciseDefinition>();
  let resolvedCourseTitle: string | null = null;

  const upsertDef = (
    rawCode: string,
    opts: {
      side?: string | null;
      targetRomDeg?: number | null;
      targetReps?: number | null;
      targetSets?: number | null;
      courseTitle?: string | null;
      source: 'patient_courses' | 'rehab_schedules' | 'patient_history';
      isScheduledEntry?: boolean;
      isCompletedSchedule?: boolean;
    },
  ) => {
    if (!rawCode || rawCode === 'CLINIC_CHECKIN') return;
    const meta = resolveExerciseCodeAndMeta(rawCode, opts.side, patient.affectedSide);
    const key = `${meta.exerciseCode}::${meta.targetSide}`;
    const existing = byKey.get(key);
    const nextTargetRom =
      typeof opts.targetRomDeg === 'number' && Number.isFinite(opts.targetRomDeg) && opts.targetRomDeg >= 10
        ? Math.round(opts.targetRomDeg * 10) / 10
        : existing?.targetRomDeg ?? meta.defaultTargetRomDeg;
    const nextTargetReps =
      typeof opts.targetReps === 'number' && Number.isFinite(opts.targetReps) && opts.targetReps > 0
        ? Math.round(opts.targetReps)
        : existing?.targetReps ?? 10;
    const nextTargetSets =
      typeof opts.targetSets === 'number' && Number.isFinite(opts.targetSets) && opts.targetSets > 0
        ? Math.round(opts.targetSets)
        : existing?.targetSets ?? 2;

    if (existing) {
      if (!existing.sources.includes(opts.source)) {
        existing.sources.push(opts.source);
      }
      existing.targetRomDeg = nextTargetRom;
      existing.targetReps = nextTargetReps;
      existing.targetSets = nextTargetSets;
      if (opts.courseTitle && !existing.courseTitle) {
        existing.courseTitle = opts.courseTitle;
      }
      if (opts.isScheduledEntry) {
        existing.totalScheduledCount += 1;
        if (opts.isCompletedSchedule) {
          existing.completedScheduledCount += 1;
        }
      }
      return;
    }

    byKey.set(key, {
      key,
      exerciseCode: meta.exerciseCode,
      exerciseNameTh: meta.exerciseNameTh,
      targetSide: meta.targetSide,
      targetRomDeg: nextTargetRom,
      targetReps: nextTargetReps,
      targetSets: nextTargetSets,
      courseTitle:
        (opts.courseTitle ?? '').trim() || resolvedCourseTitle || 'คอร์สฟื้นฟูที่นักกายภาพจัดให้',
      sources: [opts.source],
      totalScheduledCount: opts.isScheduledEntry ? 1 : 0,
      completedScheduledCount: opts.isCompletedSchedule ? 1 : 0,
    });
  };

  // 1. patient_courses weeklySchedule (strictly matching patient.id)
  if (course && belongsStrictlyToPatient(patient, course.patientId)) {
    resolvedCourseTitle = (course.title ?? '').trim() || null;
    for (const day of course.weeklySchedule ?? []) {
      if (day.dayType === 'rest' || day.activityType === 'minigame') continue;
      for (const ex of day.exercises ?? []) {
        if (!ex?.exerciseId) continue;
        upsertDef(ex.exerciseId, {
          side: ex.side,
          targetRomDeg: ex.targetMinDeg || ex.targetMaxDeg,
          targetReps: ex.reps,
          targetSets: ex.sets,
          courseTitle: resolvedCourseTitle,
          source: 'patient_courses',
        });
      }
    }
  }

  // 2. Deduplicated rehab_schedules from explicit table + patient.rehabSchedules + patient.history.rehabSchedules
  const seenScheduleIds = new Set<string>();
  const combinedSchedules: { item: RehabScheduleItem; source: 'rehab_schedules' | 'patient_history' }[] = [];
  for (const item of rehabSchedules ?? []) {
    if (!item || !belongsStrictlyToPatient(patient, item.patientId)) continue;
    const sid = item.id || `${item.scheduledDate}-${item.exerciseId || item.exerciseCode || ''}`;
    if (seenScheduleIds.has(sid)) continue;
    seenScheduleIds.add(sid);
    combinedSchedules.push({ item, source: 'rehab_schedules' });
  }
  const historySchedules = [
    ...(patient.rehabSchedules ?? []),
    ...(Array.isArray(patient.history?.rehabSchedules) ? patient.history!.rehabSchedules! : []),
  ];
  for (const item of historySchedules) {
    if (!item || !belongsStrictlyToPatient(patient, item.patientId)) continue;
    const sid = item.id || `${item.scheduledDate}-${item.exerciseId || item.exerciseCode || ''}`;
    if (seenScheduleIds.has(sid)) continue;
    seenScheduleIds.add(sid);
    combinedSchedules.push({ item, source: 'patient_history' });
  }

  for (const { item, source } of combinedSchedules) {
    if (item.status === 'REST') continue;
    const rawCode = item.effectiveCode || item.exerciseCode || item.exerciseId || '';
    if (!rawCode) continue;
    upsertDef(rawCode, {
      side: item.side,
      targetReps: item.reps,
      targetSets: item.sets,
      courseTitle: resolvedCourseTitle,
      source,
      isScheduledEntry: true,
      isCompletedSchedule: item.status === 'COMPLETED',
    });
  }

  return {
    courseTitle: resolvedCourseTitle,
    exercises: Array.from(byKey.values()),
  };
}

/**
 * Collects all completed physical-therapy course sessions for a single patient
 * across `rehab_sessions`, `patient.rehabSessionHistory` / `rehab_session_history`,
 * and completed `rehab_schedules` without cross-patient leakage.
 */
export function collectCompletedCourseSessionsForPatient(
  patient: Pick<PatientRecord, 'id' | 'hn' | 'affectedSide' | 'rehabSchedules' | 'rehabSessionHistory'> & {
    history?: {
      rehabSchedules?: RehabScheduleItem[];
      rehabSessionHistory?: RehabSessionHistoryEntry[];
    } | null;
  },
  sessions?: readonly RehabSessionRecord[] | null,
  sessionHistory?: readonly RehabSessionHistoryEntry[] | null,
  rehabSchedules?: readonly RehabScheduleItem[] | null,
): NormalizedCompletedCourseSession[] {
  const out: NormalizedCompletedCourseSession[] = [];
  const seenIds = new Set<string>();

  const hasDuplicateSession = (candidate: NormalizedCompletedCourseSession): boolean => {
    if (seenIds.has(candidate.sourceSessionId)) return true;
    const candTs = new Date(candidate.assessedAt).getTime();
    for (const existing of out) {
      if (existing.exerciseCode !== candidate.exerciseCode) continue;
      const existTs = new Date(existing.assessedAt).getTime();
      if (
        Number.isFinite(candTs) &&
        Number.isFinite(existTs) &&
        Math.abs(candTs - existTs) <= 90_000
      ) {
        return true;
      }
    }
    return false;
  };

  // 1. Live `rehab_sessions` (primary source)
  for (const s of sessions ?? []) {
    if (!s || !belongsStrictlyToPatient(patient, s.patientId)) continue;
    if (s.activityType !== 'physio_exercise' || s.exerciseId === 'CLINIC_CHECKIN') continue;
    const reps = Math.max(0, Math.round(Number(s.completedReps ?? 0)));
    const rom =
      typeof s.maxRomDeg === 'number' && Number.isFinite(s.maxRomDeg) && s.maxRomDeg > 0
        ? Number(s.maxRomDeg.toFixed(1))
        : null;
    if (reps === 0 && rom === null && (s.completedSets ?? 0) <= 0) continue;

    const meta = resolveExerciseCodeAndMeta(s.exerciseId, s.bodySide, patient.affectedSide);
    const normalized: NormalizedCompletedCourseSession = {
      sourceSessionId: s.id || `rehab-sess-${meta.exerciseCode}-${s.createdAt}`,
      patientId: patient.id,
      patientHn: patient.hn || '',
      exerciseCode: meta.exerciseCode,
      exerciseNameTh: (s.exerciseName ?? '').trim() || meta.exerciseNameTh,
      targetSide: meta.targetSide,
      achievedRomDeg: rom,
      completedReps: reps,
      completedSets: Math.max(0, Math.round(Number(s.completedSets ?? 1))),
      accuracyScore:
        typeof s.accuracyPercent === 'number' && Number.isFinite(s.accuracyPercent)
          ? clamp(s.accuracyPercent, 0, 100)
          : null,
      assessedAt: s.createdAt || new Date().toISOString(),
      facilityName: s.facilityName || 'ตู้ Kiosk กายภาพอัจฉริยะ Dr.Tech.Care',
      origin: 'rehab_sessions',
    };
    if (!hasDuplicateSession(normalized)) {
      seenIds.add(normalized.sourceSessionId);
      out.push(normalized);
    }
  }

  // 2. `rehab_session_history` + `patient.rehabSessionHistory` + `patient.history.rehabSessionHistory`
  const mergedHistory: RehabSessionHistoryEntry[] = [
    ...(sessionHistory ?? []),
    ...(patient.rehabSessionHistory ?? []),
    ...(Array.isArray(patient.history?.rehabSessionHistory)
      ? patient.history!.rehabSessionHistory!
      : []),
  ];
  for (const h of mergedHistory) {
    if (!h || !belongsStrictlyToPatient(patient, h.patientId, h.hn)) continue;
    if (!h.exerciseCode || h.exerciseCode === 'CLINIC_CHECKIN') continue;
    const reps = Math.max(0, Math.round(Number(h.validReps ?? 0)));
    const rom =
      typeof h.achievedMaxRomDeg === 'number' &&
      Number.isFinite(h.achievedMaxRomDeg) &&
      h.achievedMaxRomDeg > 0
        ? Number(h.achievedMaxRomDeg.toFixed(1))
        : null;
    if (reps === 0 && rom === null) continue;

    const meta = resolveExerciseCodeAndMeta(h.exerciseCode, h.bodySide, patient.affectedSide);
    const normalized: NormalizedCompletedCourseSession = {
      sourceSessionId: h.id || `hist-${meta.exerciseCode}-${h.sessionDate}`,
      patientId: patient.id,
      patientHn: h.hn || patient.hn || '',
      exerciseCode: meta.exerciseCode,
      exerciseNameTh: (h.exerciseName ?? '').trim() || meta.exerciseNameTh,
      targetSide: meta.targetSide,
      achievedRomDeg: rom,
      completedReps: reps,
      completedSets: 1,
      accuracyScore: null,
      assessedAt: h.sessionDate || new Date().toISOString(),
      facilityName: h.facilityName || 'ศูนย์กายภาพบำบัด Dr.Tech.Care สาขาหลัก',
      origin: 'rehab_session_history',
    };
    if (!hasDuplicateSession(normalized)) {
      seenIds.add(normalized.sourceSessionId);
      out.push(normalized);
    }
  }

  // 3. Completed PT-assigned schedules without raw angle telemetry (proportional ROM path)
  const allSchedules: RehabScheduleItem[] = [
    ...(rehabSchedules ?? []),
    ...(patient.rehabSchedules ?? []),
    ...(Array.isArray(patient.history?.rehabSchedules) ? patient.history!.rehabSchedules! : []),
  ];
  const seenSchedKeys = new Set<string>();
  for (const sched of allSchedules) {
    if (!sched || sched.status !== 'COMPLETED') continue;
    if (!belongsStrictlyToPatient(patient, sched.patientId)) continue;
    const rawCode = sched.effectiveCode || sched.exerciseCode || sched.exerciseId || '';
    if (!rawCode || rawCode === 'CLINIC_CHECKIN') continue;
    const meta = resolveExerciseCodeAndMeta(rawCode, sched.side, patient.affectedSide);
    const datePrefix = String(sched.scheduledDate || '').slice(0, 10);
    const dedupKey = `${meta.exerciseCode}::${datePrefix}`;
    if (seenSchedKeys.has(dedupKey)) continue;
    seenSchedKeys.add(dedupKey);

    // Skip if a real session already exists for the same exercise on that date
    const alreadyCoveredByRealSession = out.some(
      (s) => s.exerciseCode === meta.exerciseCode && s.assessedAt.slice(0, 10) === datePrefix,
    );
    if (alreadyCoveredByRealSession) continue;

    const reps = Math.max(0, Math.round(Number(sched.reps ?? 0)));
    if (reps <= 0) continue;
    const assessedAt = datePrefix
      ? `${datePrefix}T12:00:00.000Z`
      : new Date().toISOString();
    const sourceId = sched.id ? `sched-${sched.id}` : `sched-${meta.exerciseCode}-${datePrefix}`;
    const normalized: NormalizedCompletedCourseSession = {
      sourceSessionId: sourceId,
      patientId: patient.id,
      patientHn: patient.hn || '',
      exerciseCode: meta.exerciseCode,
      exerciseNameTh: meta.exerciseNameTh,
      targetSide: meta.targetSide,
      achievedRomDeg: null,
      completedReps: reps,
      completedSets: Math.max(1, Math.round(Number(sched.sets ?? 1))),
      accuracyScore: 85,
      assessedAt,
      facilityName: 'ศูนย์กายภาพบำบัด Dr.Tech.Care สาขาหลัก',
      origin: 'completed_schedule',
    };
    if (!hasDuplicateSession(normalized)) {
      seenIds.add(normalized.sourceSessionId);
      out.push(normalized);
    }
  }

  return out.sort(
    (a, b) => new Date(a.assessedAt).getTime() - new Date(b.assessedAt).getTime(),
  );
}

/**
 * Checks whether a completed course session has already been persisted as a clinical assessment
 * (idempotent deduplication by `source_session_id` / `session_id`, ignoring non-persisted synthetic rows).
 */
export function isCourseSessionAlreadyRecorded(
  sourceSessionId: string,
  existingAssessments: readonly PatientClinicalAssessmentRecord[],
  candidate?: Pick<CourseAutoAssessmentResult, 'exerciseCode' | 'assessedAt' | 'effectiveRomDeg'>,
): boolean {
  const cleanId = String(sourceSessionId ?? '').trim();
  const candTs = candidate ? new Date(candidate.assessedAt).getTime() : Number.NaN;

  for (const rec of existingAssessments) {
    // Ephemeral synthetic Kiosk/history rows are NOT persisted COURSE_AUTO_CALCULATED records
    if (rec.id.startsWith('kiosk-sess-') || rec.id.startsWith('hist-sess-')) {
      continue;
    }
    if (cleanId) {
      if (
        rec.source_session_id === cleanId ||
        rec.session_id === cleanId ||
        rec.id === cleanId ||
        rec.id === `course-auto-${cleanId}`
      ) {
        return true;
      }
    }
    if (
      candidate &&
      rec.source_channel === 'COURSE_AUTO_CALCULATED' &&
      rec.exercise_code.toUpperCase() === candidate.exerciseCode.toUpperCase()
    ) {
      const existTs = new Date(rec.assessed_at).getTime();
      if (
        Number.isFinite(candTs) &&
        Number.isFinite(existTs) &&
        Math.abs(candTs - existTs) <= 90_000 &&
        Math.abs(rec.measured_rom_deg - candidate.effectiveRomDeg) <= 0.6
      ) {
        return true;
      }
    }
  }

  return false;
}

/**
 * Pairs a single patient's PT-assigned rehab courses with their completed course play sessions
 * to generate deterministic 0-6 Mahidol scores, effective ROMs, per-exercise summaries,
 * and deduplicated unrecorded assessments ready for one-click batch sync.
 */
export function buildPatientCourseAutoAssessments(params: {
  patient: Pick<
    PatientRecord,
    'id' | 'hn' | 'affectedSide' | 'rehabSchedules' | 'rehabSessionHistory'
  > & {
    history?: {
      rehabSchedules?: RehabScheduleItem[];
      rehabSessionHistory?: RehabSessionHistoryEntry[];
    } | null;
  };
  course?: PatientCourseRecord | null;
  rehabSchedules?: readonly RehabScheduleItem[] | null;
  sessions?: readonly RehabSessionRecord[] | null;
  sessionHistory?: readonly RehabSessionHistoryEntry[] | null;
  existingAssessments?: readonly PatientClinicalAssessmentRecord[] | null;
}): PatientCourseAutoScoringBundle {
  const { patient, course, rehabSchedules, sessions, sessionHistory } = params;
  const existingAssessments = (params.existingAssessments ?? []).filter((rec) =>
    belongsStrictlyToPatient(patient, rec.patient_id, rec.patient_hn),
  );

  const { courseTitle, exercises: assignedExercises } = extractAssignedCourseExercisesForPatient(
    patient,
    course,
    rehabSchedules,
  );
  const completedSessions = collectCompletedCourseSessionsForPatient(
    patient,
    sessions,
    sessionHistory,
    rehabSchedules,
  );

  const findMatchingDefinition = (
    exerciseCode: string,
    targetSide: AssessmentTargetSide,
  ): AssignedCourseExerciseDefinition | undefined => {
    const upperCode = exerciseCode.toUpperCase();
    return (
      assignedExercises.find(
        (def) => def.exerciseCode === upperCode && def.targetSide === targetSide,
      ) ?? assignedExercises.find((def) => def.exerciseCode === upperCode)
    );
  };

  const autoResults: CourseAutoAssessmentResult[] = completedSessions.map((sess) => {
    const matchedDef = findMatchingDefinition(sess.exerciseCode, sess.targetSide);
    const catalog = findCatalogExercise(sess.exerciseCode);
    const targetRomDeg = matchedDef?.targetRomDeg ?? catalog?.detection.targetMinDeg ?? 90;
    const targetReps = matchedDef?.targetReps ?? Math.max(1, sess.completedReps || 10);

    return calculateMahidolScoreFromCourse({
      achievedRomDeg: sess.achievedRomDeg,
      targetRomDeg,
      completedReps: sess.completedReps,
      targetReps,
      accuracyScore: sess.accuracyScore,
      sourceSessionId: sess.sourceSessionId,
      exerciseCode: sess.exerciseCode,
      exerciseNameTh: matchedDef?.exerciseNameTh || sess.exerciseNameTh,
      targetSide: matchedDef?.targetSide || sess.targetSide,
      assessedAt: sess.assessedAt,
      facilityName: sess.facilityName,
      courseTitle: matchedDef?.courseTitle || courseTitle || 'คอร์สฟื้นฟูที่นักกายภาพจัดให้',
      patientId: patient.id,
      patientHn: patient.hn || '',
    });
  });

  // Build summaries for all PT-assigned exercises (plus any completed exercises not explicitly listed)
  const summaryDefs: AssignedCourseExerciseDefinition[] = [...assignedExercises];
  for (const res of autoResults) {
    const exists = summaryDefs.some(
      (def) => def.exerciseCode === res.exerciseCode && def.targetSide === res.targetSide,
    );
    if (!exists) {
      summaryDefs.push({
        key: `${res.exerciseCode}::${res.targetSide}`,
        exerciseCode: res.exerciseCode,
        exerciseNameTh: res.exerciseNameTh,
        targetSide: res.targetSide,
        targetRomDeg: res.targetRomDeg,
        targetReps: res.targetReps,
        targetSets: 1,
        courseTitle: res.courseTitle,
        sources: ['patient_history'],
        totalScheduledCount: 0,
        completedScheduledCount: 0,
      });
    }
  }

  const exerciseSummaries: AssignedCourseExerciseSummary[] = summaryDefs.map((definition) => {
    const matchingResults = autoResults.filter(
      (r) =>
        r.exerciseCode === definition.exerciseCode &&
        (r.targetSide === definition.targetSide || definition.targetSide === 'BOTH'),
    );
    const latestResult =
      matchingResults.length > 0 ? matchingResults[matchingResults.length - 1] : null;
    const bestRomDeg =
      matchingResults.length > 0
        ? Math.max(...matchingResults.map((r) => r.effectiveRomDeg))
        : null;
    const averageScore0To6 =
      matchingResults.length > 0
        ? Number(
            (
              matchingResults.reduce((sum, r) => sum + r.calculatedScore0To6, 0) /
              matchingResults.length
            ).toFixed(1),
          )
        : null;
    return {
      definition,
      completedSessionsCount: matchingResults.length,
      latestResult,
      bestRomDeg,
      averageScore0To6,
    };
  });

  const unrecordedResults = autoResults.filter(
    (res) => !isCourseSessionAlreadyRecorded(res.sourceSessionId, existingAssessments, res),
  );

  const latestResult = autoResults.length > 0 ? autoResults[autoResults.length - 1] : null;

  return {
    patientId: patient.id,
    patientHn: patient.hn || '',
    courseTitle,
    assignedExercises,
    exerciseSummaries,
    completedSessions,
    autoResults,
    unrecordedResults,
    latestResult,
  };
}

export function buildAssessmentInputFromCourseResult(params: {
  result: CourseAutoAssessmentResult;
  patient: { id: string; hn?: string; universalHealthLinkCode?: string | null };
  assessorStaffId?: string;
  assessorStaffName?: string;
  facilityName?: string;
}): CreatePatientClinicalAssessmentInput {
  const { result, patient } = params;
  const linkCode = resolvePatientUniversalLinkCode(patient);
  const nextRomTarget =
    result.effectiveRomDeg >= result.targetRomDeg
      ? Math.round(result.targetRomDeg)
      : Math.min(Math.round(result.targetRomDeg), Math.round(result.effectiveRomDeg + 10));

  return {
    patient_id: patient.id,
    patient_hn: patient.hn || result.patientHn || '',
    universal_link_code: linkCode,
    assessed_at: result.assessedAt,
    facility_name:
      (params.facilityName ?? '').trim() ||
      result.facilityName ||
      'ศูนย์กายภาพบำบัด Dr.Tech.Care สาขาหลัก',
    assessor_staff_id: (params.assessorStaffId ?? '').trim() || 'SYSTEM_COURSE_AUTO_CALC',
    assessor_staff_name:
      (params.assessorStaffName ?? '').trim() || 'ระบบคำนวณอัตโนมัติจากคอร์สกายภาพ',
    exercise_code: result.exerciseCode,
    exercise_name_th: result.exerciseNameTh,
    target_side: result.targetSide,
    initial_score_0_to_6: result.calculatedScore0To6,
    measured_rom_deg: result.effectiveRomDeg,
    target_rom_deg: result.targetRomDeg,
    pain_vas_0_to_10: null,
    chief_complaint: `ประเมินอัตโนมัติจาก${result.courseTitle} ท่า ${result.exerciseNameTh} (${result.completedReps}/${result.targetReps} ครั้ง)`,
    clinical_handover_note: result.breakdownSummaryTh,
    next_session_goal:
      result.calculatedScore0To6 >= 6
        ? `คงความสมบูรณ์การเคลื่อนไหวท่า ${result.exerciseNameTh} และฝึกตามคอร์สต่อเนื่อง`
        : `เพิ่มมุมการเคลื่อนไหวท่า ${result.exerciseNameTh} สู่ ${nextRomTarget}° และยกระดับคะแนนมหิดลเป็น ${Math.min(6, result.calculatedScore0To6 + 1)}/6`,
    source_channel: 'COURSE_AUTO_CALCULATED',
    measurement_method: 'COURSE_AUTO_CALCULATED',
    completed_reps: result.completedReps,
    accuracy_percent: result.accuracyScore,
    session_id: result.sourceSessionId,
    source_session_id: result.sourceSessionId,
  };
}
