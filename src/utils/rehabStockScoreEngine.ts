import {
  CLINICAL_EXERCISES,
  parseExerciseCodeAndSide,
} from '@drtechcares/pose-engine';
import type { PatientClinicalAssessmentRecord } from '../types/clinicalAssessment';
import type {
  AssignedCourseExerciseDefinition,
  NormalizedCompletedCourseSession,
} from './courseClinicalScoring';
import {
  formatThaiFullDateOnly,
  formatThaiShortDate,
} from './clinicalChartAnalytics';

export type StockScoreCalculationMode = 'SMART_WEIGHTED' | 'RAW_REPS';

export interface RehabStockExerciseSpec {
  exerciseCode: string;
  exerciseNameTh: string;
  targetReps: number; // Max points for this exercise (1 rep = 1 point, e.g. 10 reps = 10 pts)
  targetRomDeg: number;
  source: 'prescribed' | 'observed';
}

export interface RehabStockExerciseContribution {
  exerciseCode: string;
  exerciseNameTh: string;
  targetReps: number;
  maxPoints: number;
  completedReps: number;
  rawEarnedPoints: number;
  smartEarnedPoints: number;
  achievedRomDeg: number | null;
  targetRomDeg: number;
  romQualityPct: number;
  accuracyPct: number;
  qualityMultiplier: number;
  rawPct: number;
  smartPct: number;
  played: boolean;
}

export interface RehabStockDayPoint {
  dateKey: string; // YYYY-MM-DD
  labelShortTh: string; // e.g. 24 พ.ย. 69
  labelFullTh: string; // e.g. 24 พ.ย. 2569
  timestampIso: string;
  maxTotalScore: number; // e.g. 50 points for 5 exercises x 10 reps
  rawEarnedScore: number; // e.g. 44.0 / 50
  smartEarnedScore: number; // e.g. 41.8 / 50 (ROM + accuracy weighted)
  rawTotalPct: number; // 0.0 - 100.0
  smartTotalPct: number; // 0.0 - 100.0
  // Stock OHLC fields (computed in intelligent % scale, plus raw equivalents)
  openPct: number;
  highPct: number;
  lowPct: number;
  closePct: number;
  rawOpenPct: number;
  rawHighPct: number;
  rawLowPct: number;
  rawClosePct: number;
  // Moving averages & momentum
  ma3SmartPct: number;
  ma3RawPct: number;
  deltaSmartPct: number | null;
  deltaRawPct: number | null;
  deltaSmartPoints: number | null;
  deltaRawPoints: number | null;
  trend: 'UP' | 'DOWN' | 'FLAT';
  volumeReps: number; // Total reps performed this day (for bottom volume histogram)
  completedExercisesCount: number;
  totalExercisesCount: number;
  avgRomQualityPct: number;
  avgAccuracyPct: number;
  exerciseBreakdown: RehabStockExerciseContribution[];
}

export interface RehabStockIndexSummary {
  prescribedExercises: RehabStockExerciseSpec[];
  totalPrescribedExercises: number;
  maxTotalScore: number; // e.g. 5 exercises * 10 reps = 50 points
  points: RehabStockDayPoint[];
  latestPoint: RehabStockDayPoint | null;
  previousPoint: RehabStockDayPoint | null;
  allTimeHighSmartPct: number | null;
  allTimeLowSmartPct: number | null;
  allTimeHighRawPct: number | null;
  allTimeLowRawPct: number | null;
  avgSmartPct: number | null;
  avgRawPct: number | null;
  overallTrend: 'UP' | 'DOWN' | 'FLAT';
  momentumLabelTh: string;
  strongestExercise: RehabStockExerciseContribution | null;
  weakestExercise: RehabStockExerciseContribution | null;
}

function clamp(v: number, min: number, max: number): number {
  if (!Number.isFinite(v)) return min;
  return Math.max(min, Math.min(max, v));
}

function round1(v: number): number {
  if (!Number.isFinite(v)) return 0;
  return Math.round(v * 10) / 10;
}

function toLocalDateKey(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return '';
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, '0');
  const day = String(d.getDate()).padStart(2, '0');
  return `${y}-${m}-${day}`;
}

function lookupExerciseNameTh(code: string, fallbackName?: string): string {
  const cleanFallback = (fallbackName ?? '').trim();
  if (cleanFallback) return cleanFallback;
  const parsed = parseExerciseCodeAndSide(code);
  const catalog = CLINICAL_EXERCISES.find(
    (e) =>
      e.code.toUpperCase() === parsed.baseCode.toUpperCase() ||
      e.id.toUpperCase() === code.toUpperCase(),
  );
  return catalog?.name.th || parsed.baseCode || code;
}

function lookupDefaultTargetRom(code: string): number {
  const parsed = parseExerciseCodeAndSide(code);
  const catalog = CLINICAL_EXERCISES.find(
    (e) =>
      e.code.toUpperCase() === parsed.baseCode.toUpperCase() ||
      e.id.toUpperCase() === code.toUpperCase(),
  );
  return Number((catalog?.detection.targetMinDeg ?? 90).toFixed(1));
}

/**
 * Computes the intelligent quality multiplier (0.50 .. 1.00) for an exercise based on
 * ROM depth relative to target ROM (65% weight) and pose accuracy (35% weight).
 * Full credit (1.00) is awarded when ROM >= 90% of clinical target AND accuracy >= 85%.
 */
export function computeIntelligentExerciseQualityMultiplier(params: {
  achievedRomDeg: number | null;
  targetRomDeg: number;
  accuracyPct: number;
}): {
  romQualityPct: number;
  accuracyPct: number;
  qualityMultiplier: number;
} {
  const safeTargetRom = Math.max(10, Number.isFinite(params.targetRomDeg) ? params.targetRomDeg : 90);
  const safeAccuracy = clamp(
    Number.isFinite(params.accuracyPct) ? params.accuracyPct : 90,
    0,
    100,
  );
  const hasRom =
    typeof params.achievedRomDeg === 'number' &&
    Number.isFinite(params.achievedRomDeg) &&
    params.achievedRomDeg > 0;

  const romRatio = hasRom
    ? clamp((params.achievedRomDeg as number) / safeTargetRom, 0, 1.0)
    : clamp(safeAccuracy / 100, 0.75, 1.0);
  const accRatio = clamp(safeAccuracy / 100, 0, 1.0);

  const romQualityPct = Math.round(romRatio * 100);
  const accuracyRounded = Math.round(safeAccuracy);

  if (romRatio >= 0.9 && accRatio >= 0.85) {
    return {
      romQualityPct,
      accuracyPct: accuracyRounded,
      qualityMultiplier: 1.0,
    };
  }

  const combinedQuality = romRatio * 0.65 + accRatio * 0.35;
  const qualityMultiplier = Math.round(clamp(0.5 + 0.5 * combinedQuality, 0.5, 1.0) * 1000) / 1000;
  return {
    romQualityPct,
    accuracyPct: accuracyRounded,
    qualityMultiplier,
  };
}

interface DailyObservedExerciseStats {
  exerciseCode: string;
  exerciseNameTh: string;
  totalReps: number;
  maxRomDeg: number | null;
  accuracySum: number;
  accuracyCount: number;
  targetRomDeg: number | null;
  latestIso: string;
}

/**
 * Builds the Stock-Market-Style Total Rehab Percentage & Full-Score Index across all
 * doctor/PT-assigned exercises.
 *
 * Example:
 * - 1 assigned exercise x 10 reps => maxTotalScore = 10 pts (10 reps completed = 10/10 = 100%)
 * - 5 assigned exercises x 10 reps => maxTotalScore = 50 pts (each exercise contributes up to 10 pts;
 *   playing only 1 of 5 exercises for 10/10 reps yields 10/50 = 20.0%, preventing false 100% inflation).
 */
export function buildRehabStockIndex(params: {
  assignedExercises?: readonly AssignedCourseExerciseDefinition[] | null;
  completedSessions?: readonly NormalizedCompletedCourseSession[] | null;
  assessments?: readonly PatientClinicalAssessmentRecord[] | null;
}): RehabStockIndexSummary {
  const assignedList = params.assignedExercises ?? [];
  const completedSessions = params.completedSessions ?? [];
  const assessments = params.assessments ?? [];

  // 1. Consolidate the doctor/PT-prescribed exercise pool (unique by base exercise code)
  const specByCode = new Map<string, RehabStockExerciseSpec>();
  for (const def of assignedList) {
    const code = (def.exerciseCode || '').trim().toUpperCase();
    if (!code || code === 'CLINIC_CHECKIN') continue;
    const targetReps = Math.max(1, Math.round(def.targetReps || 10));
    const targetRomDeg =
      Number.isFinite(def.targetRomDeg) && def.targetRomDeg >= 10
        ? round1(def.targetRomDeg)
        : lookupDefaultTargetRom(code);
    const prev = specByCode.get(code);
    if (!prev) {
      specByCode.set(code, {
        exerciseCode: code,
        exerciseNameTh: lookupExerciseNameTh(code, def.exerciseNameTh),
        targetReps,
        targetRomDeg,
        source: 'prescribed',
      });
    } else {
      specByCode.set(code, {
        ...prev,
        targetReps: Math.max(prev.targetReps, targetReps),
        targetRomDeg: Math.max(prev.targetRomDeg, targetRomDeg),
      });
    }
  }

  // 2. Group completed sessions + clinical assessments by local calendar day (YYYY-MM-DD)
  const dayBuckets = new Map<
    string,
    {
      latestIso: string;
      byExercise: Map<string, DailyObservedExerciseStats>;
      seenSessionIds: Set<string>;
    }
  >();

  const ensureDayBucket = (dateKey: string, iso: string) => {
    let bucket = dayBuckets.get(dateKey);
    if (!bucket) {
      bucket = {
        latestIso: iso,
        byExercise: new Map(),
        seenSessionIds: new Set(),
      };
      dayBuckets.set(dateKey, bucket);
    } else if (new Date(iso).getTime() > new Date(bucket.latestIso).getTime()) {
      bucket.latestIso = iso;
    }
    return bucket;
  };

  const ingestObservation = (obs: {
    dateKey: string;
    iso: string;
    exerciseCode: string;
    exerciseNameTh?: string;
    reps: number;
    romDeg: number | null;
    targetRomDeg: number | null;
    accuracyPct: number | null;
    dedupId?: string | null;
  }) => {
    if (!obs.dateKey || !obs.exerciseCode || obs.exerciseCode === 'CLINIC_CHECKIN') return;
    const code = obs.exerciseCode.trim().toUpperCase();
    const bucket = ensureDayBucket(obs.dateKey, obs.iso);
    const cleanDedup = (obs.dedupId ?? '').trim();
    if (cleanDedup) {
      if (bucket.seenSessionIds.has(cleanDedup)) return;
      bucket.seenSessionIds.add(cleanDedup);
    }

    // Ensure observed exercises are represented if the patient has no explicit course pool yet
    // or performed an extra prescribed exercise not listed in weeklySchedule
    if (!specByCode.has(code)) {
      const inferredTargetReps = Math.max(10, Math.round(obs.reps || 10));
      specByCode.set(code, {
        exerciseCode: code,
        exerciseNameTh: lookupExerciseNameTh(code, obs.exerciseNameTh),
        targetReps: inferredTargetReps,
        targetRomDeg:
          obs.targetRomDeg && obs.targetRomDeg >= 10
            ? round1(obs.targetRomDeg)
            : lookupDefaultTargetRom(code),
        source: 'observed',
      });
    }

    const existing = bucket.byExercise.get(code);
    const safeReps = Math.max(0, Math.round(obs.reps));
    const safeRom =
      typeof obs.romDeg === 'number' && Number.isFinite(obs.romDeg) && obs.romDeg > 0
        ? round1(obs.romDeg)
        : null;
    const safeAcc =
      typeof obs.accuracyPct === 'number' && Number.isFinite(obs.accuracyPct)
        ? clamp(obs.accuracyPct, 0, 100)
        : null;

    if (!existing) {
      bucket.byExercise.set(code, {
        exerciseCode: code,
        exerciseNameTh: lookupExerciseNameTh(code, obs.exerciseNameTh),
        totalReps: safeReps,
        maxRomDeg: safeRom,
        accuracySum: safeAcc !== null ? safeAcc : 0,
        accuracyCount: safeAcc !== null ? 1 : 0,
        targetRomDeg: obs.targetRomDeg,
        latestIso: obs.iso,
      });
    } else {
      existing.totalReps += safeReps;
      if (safeRom !== null && (existing.maxRomDeg === null || safeRom > existing.maxRomDeg)) {
        existing.maxRomDeg = safeRom;
      }
      if (safeAcc !== null) {
        existing.accuracySum += safeAcc;
        existing.accuracyCount += 1;
      }
      if (new Date(obs.iso).getTime() > new Date(existing.latestIso).getTime()) {
        existing.latestIso = obs.iso;
      }
    }
  };

  // Ingest completed course sessions first (authoritative rep & accuracy telemetry)
  for (const s of completedSessions) {
    const dateKey = toLocalDateKey(s.assessedAt);
    ingestObservation({
      dateKey,
      iso: s.assessedAt,
      exerciseCode: s.exerciseCode,
      exerciseNameTh: s.exerciseNameTh,
      reps: s.completedReps,
      romDeg: s.achievedRomDeg,
      targetRomDeg: null,
      accuracyPct: s.accuracyScore ?? 90,
      dedupId: s.sourceSessionId,
    });
  }

  // Ingest persisted clinical assessments (skipping any already covered by session_id / source_session_id)
  for (const rec of assessments) {
    const dateKey = toLocalDateKey(rec.assessed_at);
    const code = (rec.exercise_code || '').trim().toUpperCase();
    const spec = specByCode.get(code);
    const targetReps = spec?.targetReps ?? 10;
    const dedupId = rec.source_session_id || rec.session_id || rec.id;

    // If this assessment came from a manual 0-6 evaluation without rep count, infer equivalent rep completion
    let effectiveReps =
      typeof rec.completed_reps === 'number' && Number.isFinite(rec.completed_reps) && rec.completed_reps > 0
        ? Math.round(rec.completed_reps)
        : 0;
    if (effectiveReps === 0 && (rec.measured_rom_deg > 0 || rec.initial_score_0_to_6 > 0)) {
      const targetRom = rec.target_rom_deg > 0 ? rec.target_rom_deg : lookupDefaultTargetRom(code);
      const romRatio = clamp(rec.measured_rom_deg / Math.max(1, targetRom), 0, 1);
      const scoreRatio = clamp(rec.initial_score_0_to_6 / 6, 0, 1);
      effectiveReps = Math.max(1, Math.round(Math.max(romRatio, scoreRatio) * targetReps));
    }

    const inferredAccuracy =
      typeof rec.accuracy_percent === 'number' && Number.isFinite(rec.accuracy_percent)
        ? rec.accuracy_percent
        : clamp(Math.round((rec.initial_score_0_to_6 / 6) * 100), 60, 100);

    ingestObservation({
      dateKey,
      iso: rec.assessed_at,
      exerciseCode: code,
      exerciseNameTh: rec.exercise_name_th,
      reps: effectiveReps,
      romDeg: rec.measured_rom_deg > 0 ? rec.measured_rom_deg : null,
      targetRomDeg: rec.target_rom_deg > 0 ? rec.target_rom_deg : null,
      accuracyPct: inferredAccuracy,
      dedupId,
    });
  }

  const prescribedExercises = Array.from(specByCode.values());
  const maxTotalScore = prescribedExercises.reduce((sum, ex) => sum + ex.targetReps, 0);

  // 3. Build chronological Stock-Market Day Points with OHLC, Volume, MA3 & Momentum
  const sortedDateKeys = Array.from(dayBuckets.keys()).sort((a, b) => a.localeCompare(b));
  const points: RehabStockDayPoint[] = [];

  for (let i = 0; i < sortedDateKeys.length; i += 1) {
    const dateKey = sortedDateKeys[i];
    const bucket = dayBuckets.get(dateKey)!;
    const safePoolMax = Math.max(1, maxTotalScore);

    let rawEarnedTotal = 0;
    let smartEarnedTotal = 0;
    let volumeReps = 0;
    let completedExercisesCount = 0;
    let playedRomQualitySum = 0;
    let playedAccuracySum = 0;
    const playedSmartPcts: number[] = [];
    const playedRawPcts: number[] = [];

    const exerciseBreakdown: RehabStockExerciseContribution[] = prescribedExercises.map((spec) => {
      const obs = bucket.byExercise.get(spec.exerciseCode);
      if (!obs || obs.totalReps <= 0) {
        return {
          exerciseCode: spec.exerciseCode,
          exerciseNameTh: spec.exerciseNameTh,
          targetReps: spec.targetReps,
          maxPoints: spec.targetReps,
          completedReps: 0,
          rawEarnedPoints: 0,
          smartEarnedPoints: 0,
          achievedRomDeg: null,
          targetRomDeg: spec.targetRomDeg,
          romQualityPct: 0,
          accuracyPct: 0,
          qualityMultiplier: 0,
          rawPct: 0,
          smartPct: 0,
          played: false,
        };
      }

      completedExercisesCount += 1;
      volumeReps += obs.totalReps;
      const cappedReps = clamp(obs.totalReps, 0, spec.targetReps);
      const avgAcc =
        obs.accuracyCount > 0 ? obs.accuracySum / obs.accuracyCount : 90;
      const { romQualityPct, accuracyPct, qualityMultiplier } =
        computeIntelligentExerciseQualityMultiplier({
          achievedRomDeg: obs.maxRomDeg,
          targetRomDeg: obs.targetRomDeg ?? spec.targetRomDeg,
          accuracyPct: avgAcc,
        });

      const rawEarnedPoints = round1(cappedReps);
      const smartEarnedPoints = round1(cappedReps * qualityMultiplier);
      const rawPct = round1((rawEarnedPoints / spec.targetReps) * 100);
      const smartPct = round1((smartEarnedPoints / spec.targetReps) * 100);

      rawEarnedTotal += rawEarnedPoints;
      smartEarnedTotal += smartEarnedPoints;
      playedRomQualitySum += romQualityPct;
      playedAccuracySum += accuracyPct;
      playedSmartPcts.push(smartPct);
      playedRawPcts.push(rawPct);

      return {
        exerciseCode: spec.exerciseCode,
        exerciseNameTh: spec.exerciseNameTh,
        targetReps: spec.targetReps,
        maxPoints: spec.targetReps,
        completedReps: obs.totalReps,
        rawEarnedPoints,
        smartEarnedPoints,
        achievedRomDeg: obs.maxRomDeg,
        targetRomDeg: obs.targetRomDeg ?? spec.targetRomDeg,
        romQualityPct,
        accuracyPct,
        qualityMultiplier,
        rawPct,
        smartPct,
        played: true,
      };
    });

    const rawEarnedScore = round1(rawEarnedTotal);
    const smartEarnedScore = round1(smartEarnedTotal);
    const rawTotalPct = round1(clamp((rawEarnedScore / safePoolMax) * 100, 0, 100));
    const smartTotalPct = round1(clamp((smartEarnedScore / safePoolMax) * 100, 0, 100));

    const prev = i > 0 ? points[i - 1] : null;
    const openPct = prev ? prev.closePct : smartTotalPct;
    const closePct = smartTotalPct;
    const highPct = round1(
      clamp(
        Math.max(
          openPct,
          closePct,
          playedSmartPcts.length > 0 ? Math.max(...playedSmartPcts) : closePct,
        ),
        0,
        100,
      ),
    );
    const lowPct = round1(
      clamp(
        Math.min(
          openPct,
          closePct,
          playedSmartPcts.length > 0 ? Math.min(...playedSmartPcts) : closePct,
        ),
        0,
        100,
      ),
    );

    const rawOpenPct = prev ? prev.rawClosePct : rawTotalPct;
    const rawClosePct = rawTotalPct;
    const rawHighPct = round1(
      clamp(
        Math.max(
          rawOpenPct,
          rawClosePct,
          playedRawPcts.length > 0 ? Math.max(...playedRawPcts) : rawClosePct,
        ),
        0,
        100,
      ),
    );
    const rawLowPct = round1(
      clamp(
        Math.min(
          rawOpenPct,
          rawClosePct,
          playedRawPcts.length > 0 ? Math.min(...playedRawPcts) : rawClosePct,
        ),
        0,
        100,
      ),
    );

    const deltaSmartPct = prev ? round1(smartTotalPct - prev.smartTotalPct) : null;
    const deltaRawPct = prev ? round1(rawTotalPct - prev.rawTotalPct) : null;
    const deltaSmartPoints = prev ? round1(smartEarnedScore - prev.smartEarnedScore) : null;
    const deltaRawPoints = prev ? round1(rawEarnedScore - prev.rawEarnedScore) : null;

    const trend: 'UP' | 'DOWN' | 'FLAT' =
      deltaSmartPct === null || Math.abs(deltaSmartPct) < 0.1
        ? 'FLAT'
        : deltaSmartPct > 0
          ? 'UP'
          : 'DOWN';

    // 3-period moving average (MA3)
    const windowSmart = [...points.slice(-2).map((p) => p.smartTotalPct), smartTotalPct];
    const ma3SmartPct = round1(
      windowSmart.reduce((sum, v) => sum + v, 0) / windowSmart.length,
    );
    const windowRaw = [...points.slice(-2).map((p) => p.rawTotalPct), rawTotalPct];
    const ma3RawPct = round1(
      windowRaw.reduce((sum, v) => sum + v, 0) / windowRaw.length,
    );

    points.push({
      dateKey,
      labelShortTh: formatThaiShortDate(bucket.latestIso),
      labelFullTh: formatThaiFullDateOnly(bucket.latestIso),
      timestampIso: bucket.latestIso,
      maxTotalScore,
      rawEarnedScore,
      smartEarnedScore,
      rawTotalPct,
      smartTotalPct,
      openPct,
      highPct,
      lowPct,
      closePct,
      rawOpenPct,
      rawHighPct,
      rawLowPct,
      rawClosePct,
      ma3SmartPct,
      ma3RawPct,
      deltaSmartPct,
      deltaRawPct,
      deltaSmartPoints,
      deltaRawPoints,
      trend,
      volumeReps,
      completedExercisesCount,
      totalExercisesCount: prescribedExercises.length,
      avgRomQualityPct:
        completedExercisesCount > 0
          ? Math.round(playedRomQualitySum / completedExercisesCount)
          : 0,
      avgAccuracyPct:
        completedExercisesCount > 0
          ? Math.round(playedAccuracySum / completedExercisesCount)
          : 0,
      exerciseBreakdown,
    });
  }

  const latestPoint = points.length > 0 ? points[points.length - 1] : null;
  const previousPoint = points.length >= 2 ? points[points.length - 2] : null;

  const allTimeHighSmartPct =
    points.length > 0 ? Math.max(...points.map((p) => p.smartTotalPct)) : null;
  const allTimeLowSmartPct =
    points.length > 0 ? Math.min(...points.map((p) => p.smartTotalPct)) : null;
  const allTimeHighRawPct =
    points.length > 0 ? Math.max(...points.map((p) => p.rawTotalPct)) : null;
  const allTimeLowRawPct =
    points.length > 0 ? Math.min(...points.map((p) => p.rawTotalPct)) : null;
  const avgSmartPct =
    points.length > 0
      ? round1(points.reduce((sum, p) => sum + p.smartTotalPct, 0) / points.length)
      : null;
  const avgRawPct =
    points.length > 0
      ? round1(points.reduce((sum, p) => sum + p.rawTotalPct, 0) / points.length)
      : null;

  const overallTrend: 'UP' | 'DOWN' | 'FLAT' = latestPoint ? latestPoint.trend : 'FLAT';

  let momentumLabelTh = 'รอข้อมูลการฝึกกายภาพรอบแรก';
  if (latestPoint) {
    if (latestPoint.smartTotalPct >= 90 && overallTrend !== 'DOWN') {
      momentumLabelTh = 'ผลงานยอดเยี่ยม ระดับเป้าหมายสูงสุด';
    } else if (overallTrend === 'UP') {
      momentumLabelTh = 'แนวโน้มขาขึ้น ฟื้นตัวดีขึ้นต่อเนื่อง';
    } else if (overallTrend === 'DOWN') {
      momentumLabelTh = 'คะแนนปรับลดลง ควรติดตามท่าที่ยังไม่ครบ';
    } else if (points.length === 1) {
      momentumLabelTh = 'บันทึกฐานดัชนีรอบแรกแล้ว';
    } else {
      momentumLabelTh = 'ทรงตัวสม่ำเสมอ รักษาวินัยการฝึกได้ดี';
    }
  }

  let strongestExercise: RehabStockExerciseContribution | null = null;
  let weakestExercise: RehabStockExerciseContribution | null = null;
  if (latestPoint && latestPoint.exerciseBreakdown.length > 0) {
    const sortedBySmart = latestPoint.exerciseBreakdown
      .slice()
      .sort((a, b) => b.smartPct - a.smartPct || b.rawEarnedPoints - a.rawEarnedPoints);
    strongestExercise = sortedBySmart[0] ?? null;
    weakestExercise = sortedBySmart[sortedBySmart.length - 1] ?? null;
  }

  return {
    prescribedExercises,
    totalPrescribedExercises: prescribedExercises.length,
    maxTotalScore,
    points,
    latestPoint,
    previousPoint,
    allTimeHighSmartPct,
    allTimeLowSmartPct,
    allTimeHighRawPct,
    allTimeLowRawPct,
    avgSmartPct,
    avgRawPct,
    overallTrend,
    momentumLabelTh,
    strongestExercise,
    weakestExercise,
  };
}

