import {
  computeTargetAchievementPct,
  type LongitudinalVisitDelta,
  type MahidolScore0To6,
  type PatientClinicalAssessmentRecord,
} from '../types/clinicalAssessment';

export const THAI_MONTHS_SHORT: readonly string[] = [
  'ม.ค.',
  'ก.พ.',
  'มี.ค.',
  'เม.ย.',
  'พ.ค.',
  'มิ.ย.',
  'ก.ค.',
  'ส.ค.',
  'ก.ย.',
  'ต.ค.',
  'พ.ย.',
  'ธ.ค.',
];

export const THAI_MONTHS_FULL: readonly string[] = [
  'มกราคม',
  'กุมภาพันธ์',
  'มีนาคม',
  'เมษายน',
  'พฤษภาคม',
  'มิถุนายน',
  'กรกฎาคม',
  'สิงหาคม',
  'กันยายน',
  'ตุลาคม',
  'พฤศจิกายน',
  'ธันวาคม',
];

export const CLINICAL_EXERCISE_CHART_PALETTE: readonly string[] = [
  '#0284c7', // Medical Blue
  '#059669', // Emerald Green
  '#4f46e5', // Indigo
  '#0d9488', // Teal
  '#d97706', // Amber
  '#0891b2', // Cyan
  '#7c3aed', // Violet
  '#e11d48', // Rose
];

export interface MultiExerciseSeriesPoint {
  recordId: string;
  chronologicalIndex: number;
  visit: LongitudinalVisitDelta;
}

export interface MultiExerciseSeriesItem {
  exerciseCode: string;
  exerciseNameTh: string;
  color: string;
  sessionCount: number;
  latestRomDeg: number;
  latestScore0To6: MahidolScore0To6;
  latestTargetAchievementPct: number;
  latestAssessedAt: string;
  points: MultiExerciseSeriesPoint[];
}

export interface MultiExerciseChartGrouping {
  series: MultiExerciseSeriesItem[];
  colorByExerciseCode: Record<string, string>;
}

export interface ClinicalMilestonesSummary {
  previous: LongitudinalVisitDelta | null;
  latest: LongitudinalVisitDelta | null;
  best: LongitudinalVisitDelta | null;
  latestDeltaRomFromPrevious: number | null;
  latestDeltaScoreFromPrevious: number | null;
  latestDeltaTargetPctFromPrevious: number | null;
}

export interface MonthlyScoreSummaryItem {
  monthKey: string; // YYYY-MM
  monthLabelShortTh: string; // e.g. พ.ย. 2569
  monthLabelFullTh: string; // e.g. พฤศจิกายน 2569
  avgScore0To6: number; // 0.0 - 6.0 (1 decimal)
  maxScore0To6: MahidolScore0To6;
  avgTargetPct: number; // 1 decimal
  avgRomDeg: number; // 1 decimal
  sessionCount: number;
  exerciseCount: number;
  monthDeltaScore: number | null; // delta vs immediately preceding month
  records: PatientClinicalAssessmentRecord[];
}

function pad2(n: number): string {
  return String(Math.floor(Math.abs(n))).padStart(2, '0');
}

function parseValidDate(isoString: string): Date | null {
  if (!isoString || typeof isoString !== 'string') return null;
  const d = new Date(isoString);
  return Number.isNaN(d.getTime()) ? null : d;
}

/**
 * Full Thai Buddhist Era date & time format: e.g. `23 พ.ย. 2569 • 16:40 น.`
 */
export function formatThaiClinicalDate(isoString: string): string {
  const d = parseValidDate(isoString);
  if (!d) return isoString || '-';
  const day = d.getDate();
  const monthShort = THAI_MONTHS_SHORT[d.getMonth()] ?? '';
  const yearBE = d.getFullYear() + 543;
  const hh = pad2(d.getHours());
  const mm = pad2(d.getMinutes());
  return `${day} ${monthShort} ${yearBE} • ${hh}:${mm} น.`;
}

/**
 * Full Thai Buddhist Era date only: e.g. `23 พ.ย. 2569`
 */
export function formatThaiFullDateOnly(isoString: string): string {
  const d = parseValidDate(isoString);
  if (!d) return isoString || '-';
  const day = d.getDate();
  const monthShort = THAI_MONTHS_SHORT[d.getMonth()] ?? '';
  const yearBE = d.getFullYear() + 543;
  return `${day} ${monthShort} ${yearBE}`;
}

/**
 * Compact Thai Buddhist Era axis date: e.g. `23 พ.ย. 69`
 */
export function formatThaiShortDate(isoString: string): string {
  const d = parseValidDate(isoString);
  if (!d) return '';
  const day = d.getDate();
  const monthShort = THAI_MONTHS_SHORT[d.getMonth()] ?? '';
  const yearBE2 = String(d.getFullYear() + 543).slice(-2);
  return `${day} ${monthShort} ${yearBE2}`;
}

/**
 * Returns local calendar month key `YYYY-MM` for monthly grouping.
 */
export function getCalendarMonthKey(isoString: string): string {
  const d = parseValidDate(isoString);
  if (!d) return '';
  return `${d.getFullYear()}-${pad2(d.getMonth() + 1)}`;
}

/**
 * Formats a `YYYY-MM` month key into Thai Buddhist Era label:
 * - `short`: `พ.ย. 2569`
 * - `full`: `พฤศจิกายน 2569`
 */
export function formatThaiMonthYear(
  monthKey: string,
  style: 'short' | 'full' = 'short',
): string {
  const match = /^(\d{4})-(\d{2})$/.exec(String(monthKey ?? '').trim());
  if (!match) return monthKey || '-';
  const yearAD = Number(match[1]);
  const monthIndex = Number(match[2]) - 1;
  if (monthIndex < 0 || monthIndex > 11 || !Number.isFinite(yearAD)) return monthKey;
  const yearBE = yearAD + 543;
  const monthName =
    style === 'full' ? THAI_MONTHS_FULL[monthIndex] : THAI_MONTHS_SHORT[monthIndex];
  return `${monthName} ${yearBE}`;
}

/**
 * Groups chronological visit points into separate, color-coded series per exercise
 * so multi-exercise charts never connect unrelated joints with a single misleading line.
 */
export function buildMultiExerciseChartSeries(
  visitsWithDelta: readonly LongitudinalVisitDelta[],
): MultiExerciseChartGrouping {
  const byCode = new Map<string, MultiExerciseSeriesItem>();
  const colorByExerciseCode: Record<string, string> = {};

  visitsWithDelta.forEach((visit, chronologicalIndex) => {
    const code = (visit.record.exercise_code || 'UNKNOWN').trim().toUpperCase();
    let entry = byCode.get(code);
    if (!entry) {
      const colorIndex = byCode.size % CLINICAL_EXERCISE_CHART_PALETTE.length;
      const color = CLINICAL_EXERCISE_CHART_PALETTE[colorIndex];
      colorByExerciseCode[code] = color;
      entry = {
        exerciseCode: code,
        exerciseNameTh: (visit.record.exercise_name_th || code).trim(),
        color,
        sessionCount: 0,
        latestRomDeg: Number(visit.record.measured_rom_deg.toFixed(1)),
        latestScore0To6: visit.record.initial_score_0_to_6,
        latestTargetAchievementPct: visit.targetAchievementPct,
        latestAssessedAt: visit.record.assessed_at,
        points: [],
      };
      byCode.set(code, entry);
    }

    entry.sessionCount += 1;
    entry.latestRomDeg = Number(visit.record.measured_rom_deg.toFixed(1));
    entry.latestScore0To6 = visit.record.initial_score_0_to_6;
    entry.latestTargetAchievementPct = visit.targetAchievementPct;
    entry.latestAssessedAt = visit.record.assessed_at;
    if (visit.record.exercise_name_th?.trim()) {
      entry.exerciseNameTh = visit.record.exercise_name_th.trim();
    }
    entry.points.push({
      recordId: visit.record.id,
      chronologicalIndex,
      visit,
    });
  });

  return {
    series: Array.from(byCode.values()),
    colorByExerciseCode,
  };
}

/**
 * Computes the 3 clickable clinical milestones (`ครั้งก่อนหน้า`, `ครั้งล่าสุด`, `ครั้งที่ดีที่สุด`)
 * from chronologically ordered visits.
 * Personal Best tie-breaker: highest `initial_score_0_to_6` -> highest `% target ROM` -> highest `measured_rom_deg` -> latest timestamp.
 */
export function computeClinicalMilestones(
  visitsWithDelta: readonly LongitudinalVisitDelta[],
): ClinicalMilestonesSummary {
  if (visitsWithDelta.length === 0) {
    return {
      previous: null,
      latest: null,
      best: null,
      latestDeltaRomFromPrevious: null,
      latestDeltaScoreFromPrevious: null,
      latestDeltaTargetPctFromPrevious: null,
    };
  }

  const latest = visitsWithDelta[visitsWithDelta.length - 1];
  const previous = visitsWithDelta.length >= 2 ? visitsWithDelta[visitsWithDelta.length - 2] : null;

  let best = visitsWithDelta[0];
  for (let i = 1; i < visitsWithDelta.length; i += 1) {
    const cand = visitsWithDelta[i];
    const scoreDiff = cand.record.initial_score_0_to_6 - best.record.initial_score_0_to_6;
    if (scoreDiff > 0) {
      best = cand;
      continue;
    }
    if (scoreDiff < 0) continue;

    const pctDiff = cand.targetAchievementPct - best.targetAchievementPct;
    if (pctDiff > 0.05) {
      best = cand;
      continue;
    }
    if (pctDiff < -0.05) continue;

    const romDiff = cand.record.measured_rom_deg - best.record.measured_rom_deg;
    if (romDiff > 0.05) {
      best = cand;
      continue;
    }
    if (romDiff < -0.05) continue;

    const candTs = new Date(cand.record.assessed_at).getTime();
    const bestTs = new Date(best.record.assessed_at).getTime();
    if (Number.isFinite(candTs) && Number.isFinite(bestTs) && candTs >= bestTs) {
      best = cand;
    }
  }

  const latestDeltaRomFromPrevious = previous
    ? Number((latest.record.measured_rom_deg - previous.record.measured_rom_deg).toFixed(1))
    : null;
  const latestDeltaScoreFromPrevious = previous
    ? latest.record.initial_score_0_to_6 - previous.record.initial_score_0_to_6
    : null;
  const latestDeltaTargetPctFromPrevious = previous
    ? Number((latest.targetAchievementPct - previous.targetAchievementPct).toFixed(1))
    : null;

  return {
    previous,
    latest,
    best,
    latestDeltaRomFromPrevious,
    latestDeltaScoreFromPrevious,
    latestDeltaTargetPctFromPrevious,
  };
}

/**
 * Aggregates patient assessments by calendar month (`YYYY-MM`) for the Monthly Score Summary Chart.
 */
export function computeMonthlyScoreSummary(
  records: readonly PatientClinicalAssessmentRecord[],
): MonthlyScoreSummaryItem[] {
  const sorted = records.slice().sort((a, b) => {
    const ta = new Date(a.assessed_at).getTime();
    const tb = new Date(b.assessed_at).getTime();
    const va = Number.isFinite(ta) ? ta : 0;
    const vb = Number.isFinite(tb) ? tb : 0;
    if (va !== vb) return va - vb;
    return a.visit_number - b.visit_number;
  });

  const groups = new Map<string, PatientClinicalAssessmentRecord[]>();
  for (const rec of sorted) {
    const key = getCalendarMonthKey(rec.assessed_at);
    if (!key) continue;
    const list = groups.get(key) ?? [];
    list.push(rec);
    groups.set(key, list);
  }

  const sortedMonthKeys = Array.from(groups.keys()).sort((a, b) => a.localeCompare(b));
  const out: MonthlyScoreSummaryItem[] = [];

  sortedMonthKeys.forEach((monthKey, idx) => {
    const list = groups.get(monthKey) ?? [];
    const sessionCount = list.length;
    if (sessionCount === 0) return;

    const scoreSum = list.reduce((sum, r) => sum + r.initial_score_0_to_6, 0);
    const avgScore0To6 = Number((scoreSum / sessionCount).toFixed(1));
    const maxScore0To6 = list.reduce(
      (maxS, r) => (r.initial_score_0_to_6 > maxS ? r.initial_score_0_to_6 : maxS),
      0 as MahidolScore0To6,
    );
    const targetPctSum = list.reduce(
      (sum, r) => sum + computeTargetAchievementPct(r.measured_rom_deg, r.target_rom_deg || 120),
      0,
    );
    const avgTargetPct = Number((targetPctSum / sessionCount).toFixed(1));
    const romSum = list.reduce((sum, r) => sum + r.measured_rom_deg, 0);
    const avgRomDeg = Number((romSum / sessionCount).toFixed(1));
    const exerciseCount = new Set(
      list.map((r) => (r.exercise_code || '').trim().toUpperCase()).filter(Boolean),
    ).size;

    const prevAvg = idx > 0 ? out[idx - 1].avgScore0To6 : null;
    const monthDeltaScore =
      prevAvg !== null ? Number((avgScore0To6 - prevAvg).toFixed(1)) : null;

    out.push({
      monthKey,
      monthLabelShortTh: formatThaiMonthYear(monthKey, 'short'),
      monthLabelFullTh: formatThaiMonthYear(monthKey, 'full'),
      avgScore0To6,
      maxScore0To6,
      avgTargetPct,
      avgRomDeg,
      sessionCount,
      exerciseCount,
      monthDeltaScore,
      records: list,
    });
  });

  return out;
}

