import type { RehabSessionRecord } from '@drtechcares/api-client';
import type {
  ExerciseComparisonBreakdownItem,
  LongitudinalProgressSummary,
  MahidolScore0To6,
  PatientClinicalAssessmentRecord,
} from '../types/clinicalAssessment';
import type { RehabStockIndexSummary } from './rehabStockScoreEngine';
import { formatThaiClinicalDate, formatThaiFullDateOnly, formatThaiShortDate } from './clinicalChartAnalytics';

export type ProgressVerdictCode =
  | 'IMPROVED'
  | 'STABLE'
  | 'TRY_HARDER'
  | 'FIRST_SESSION'
  | 'NO_DATA';

export type ProgressVerdictTone = 'emerald' | 'sky' | 'amber' | 'indigo' | 'slate';

export interface ProgressVerdictSummary {
  code: ProgressVerdictCode;
  tone: ProgressVerdictTone;
  badgeTh: string;
  headlineTh: string;
  detailTh: string;
  adviceTh: string;
}

export interface NormalizedMinigameSession {
  id: string;
  patientId: string;
  playedAt: string;
  dateShortTh: string;
  dateFullTh: string;
  roundNumber: number;
  titleTh: string;
  category: string;
  correct: number;
  total: number;
  accuracyPct: number;
  durationSec: number | null;
  score0To6Equivalent: MahidolScore0To6;
  facilityName: string;
  deltaAccuracyFromPrev: number | null;
  deltaCorrectFromPrev: number | null;
  verdict: ProgressVerdictSummary;
}

export interface MinigameProgressSummary {
  totalRounds: number;
  sessions: NormalizedMinigameSession[];
  firstSession: NormalizedMinigameSession | null;
  previousSession: NormalizedMinigameSession | null;
  latestSession: NormalizedMinigameSession | null;
  bestSession: NormalizedMinigameSession | null;
  averageAccuracyPct: number | null;
  latestAccuracyPct: number | null;
  bestAccuracyPct: number | null;
  deltaAccuracyFromPrevious: number | null;
  deltaAccuracyFromFirst: number | null;
  deltaCorrectFromPrevious: number | null;
  verdict: ProgressVerdictSummary;
}

export interface RehabDomainProgressSummary {
  totalVisits: number;
  latestRomDeg: number | null;
  latestTargetPct: number | null;
  latestScore0To6: MahidolScore0To6 | null;
  latestExerciseNameTh: string | null;
  comparedSameExercise: boolean;
  deltaRomDeg: number | null;
  deltaTargetPct: number | null;
  deltaScore0To6: number | null;
  stockIndexPct: number | null;
  stockDeltaPct: number | null;
  verdict: ProgressVerdictSummary;
}

export interface UnifiedActivityTimelineItem {
  id: string;
  kind: 'physio' | 'minigame';
  timestampIso: string;
  dateFullTh: string;
  dateShortTh: string;
  sequenceLabel: string;
  titleTh: string;
  subtitleTh: string;
  facilityName: string;
  primaryMetricLabel: string;
  secondaryMetricLabel: string;
  deltaLabel: string | null;
  deltaPositive: boolean | null;
  verdict: ProgressVerdictSummary;
  physioRecord?: PatientClinicalAssessmentRecord;
  minigameSession?: NormalizedMinigameSession;
}

export interface UnifiedRehabMinigameSummary {
  patientId: string;
  rehab: RehabDomainProgressSummary;
  minigame: MinigameProgressSummary;
  overall: {
    totalActivities: number;
    combinedScorePct: number | null;
    syncReady: boolean;
    verdict: ProgressVerdictSummary;
  };
  timeline: UnifiedActivityTimelineItem[];
}

export const ENCOURAGE_TRY_HARDER_PHRASE_TH = 'ท่านทำได้ไม่เลว พยายามมากขึ้นกว่านี้นะครับ';

function clamp(v: number, min: number, max: number): number {
  if (!Number.isFinite(v)) return min;
  return Math.max(min, Math.min(max, v));
}

function round1(v: number): number {
  if (!Number.isFinite(v)) return 0;
  return Math.round(v * 10) / 10;
}

/**
 * Returns true if a `RehabSessionRecord` represents a cognitive / card minigame play.
 */
export function isMinigameRehabSession(
  session: Pick<RehabSessionRecord, 'activityType' | 'exerciseId' | 'exerciseName' | 'minigameScore'>,
): boolean {
  if (!session) return false;
  if (session.activityType === 'minigame') return true;
  if (session.minigameScore !== null && session.minigameScore !== undefined) return true;
  const exId = String(session.exerciseId ?? '');
  const exName = String(session.exerciseName ?? '');
  return /minigame|cognitive-card-game|\bcard\b/i.test(exId) || /มินิเกม|เกมการ์ด/.test(exName);
}

/**
 * Maps minigame accuracy percentage (0-100%) to a 0-6 comparable scale for unified display.
 */
export function mapAccuracyPctToScale0To6(accuracyPct: number): MahidolScore0To6 {
  const pct = clamp(accuracyPct, 0, 100);
  if (pct <= 0) return 0;
  if (pct < 25) return 1;
  if (pct < 45) return 2;
  if (pct < 65) return 3;
  if (pct < 80) return 4;
  if (pct < 92) return 5;
  return 6;
}

/**
 * Evaluates a single minigame round against the previous round (or target accuracy if first round).
 */
export function evaluateMinigameRoundVerdict(params: {
  currentAccuracyPct: number;
  currentCorrect: number;
  currentTotal: number;
  currentDurationSec?: number | null;
  previousAccuracyPct?: number | null;
  previousCorrect?: number | null;
  previousDurationSec?: number | null;
}): ProgressVerdictSummary {
  const curAcc = Math.round(clamp(params.currentAccuracyPct, 0, 100));
  const curCorrect = Math.max(0, Math.round(params.currentCorrect));
  const curTotal = Math.max(1, Math.round(params.currentTotal));
  const hasPrev =
    typeof params.previousAccuracyPct === 'number' && Number.isFinite(params.previousAccuracyPct);

  if (!hasPrev) {
    if (curAcc >= 80) {
      return {
        code: 'FIRST_SESSION',
        tone: 'emerald',
        badgeTh: 'เริ่มต้นได้ดีมาก',
        headlineTh: `เริ่มต้นได้ดีมาก! มินิเกมรอบแรกแม่นยำ ${curAcc}% (${curCorrect}/${curTotal})`,
        detailTh: 'บันทึกเป็นคะแนนตั้งต้นสำหรับติดตามพัฒนาการรอบถัดไปแล้ว',
        adviceTh: 'รักษามาตรฐานนี้ไว้และลองฝึกต่อเนื่องในนัดถัดไปครับ',
      };
    }
    if (curAcc >= 65) {
      return {
        code: 'FIRST_SESSION',
        tone: 'sky',
        badgeTh: 'บันทึกครั้งแรก',
        headlineTh: `เริ่มต้นได้ดี มินิเกมรอบแรกได้ ${curAcc}% (${curCorrect}/${curTotal})`,
        detailTh: 'ระบบบันทึกค่าตั้งต้นเรียบร้อยแล้ว พร้อมเปรียบเทียบในรอบถัดไป',
        adviceTh: 'รอบหน้าตั้งเป้าให้ถึง 80% ขึ้นไปนะครับ',
      };
    }
    return {
      code: 'TRY_HARDER',
      tone: 'amber',
      badgeTh: 'พยายามเพิ่มอีกนิด',
      headlineTh: ENCOURAGE_TRY_HARDER_PHRASE_TH,
      detailTh: `มินิเกมรอบนี้ได้ ${curCorrect}/${curTotal} ข้อ (${curAcc}%) ยังต่ำกว่าเกณฑ์เป้าหมาย 80%`,
      adviceTh: 'ลองฟังคำถามให้จบก่อนค่อยๆ ยกมือตอบ รอบหน้าจะแม่นยำขึ้นแน่นอนครับ',
    };
  }

  const prevAcc = Math.round(clamp(params.previousAccuracyPct as number, 0, 100));
  const deltaAcc = curAcc - prevAcc;
  const prevDur =
    typeof params.previousDurationSec === 'number' && params.previousDurationSec > 0
      ? params.previousDurationSec
      : null;
  const curDur =
    typeof params.currentDurationSec === 'number' && params.currentDurationSec > 0
      ? params.currentDurationSec
      : null;
  const fasterSec = prevDur !== null && curDur !== null ? prevDur - curDur : 0;

  if (deltaAcc >= 5 || (curAcc >= 85 && deltaAcc >= 0 && fasterSec >= 5)) {
    const reason =
      deltaAcc >= 5
        ? `ความแม่นยำเพิ่มขึ้น +${deltaAcc}% (${prevAcc}% -> ${curAcc}%)`
        : `คงความแม่นยำสูง ${curAcc}% และตอบได้เร็วขึ้น ${fasterSec} วินาที`;
    return {
      code: 'IMPROVED',
      tone: 'emerald',
      badgeTh: 'ดีขึ้น',
      headlineTh: `ดีขึ้นกว่าครั้งก่อน! มินิเกมได้ ${curCorrect}/${curTotal} (${curAcc}%)`,
      detailTh: reason,
      adviceTh: 'ทำได้ยอดเยี่ยมมากครับ ฝึกต่อเนื่องเพื่อให้สมองและการตอบสนองคมชัดยิ่งขึ้น',
    };
  }

  if (deltaAcc <= -5 && curAcc < 90) {
    return {
      code: 'TRY_HARDER',
      tone: 'amber',
      badgeTh: 'พยายามเพิ่มอีกนิด',
      headlineTh: ENCOURAGE_TRY_HARDER_PHRASE_TH,
      detailTh: `ความแม่นยำมินิเกมลดลง ${Math.abs(deltaAcc)}% จากครั้งก่อน (${prevAcc}% -> ${curAcc}%)`,
      adviceTh: 'ไม่ต้องรีบนะครับ ฟังเสียงพากย์ให้จบก่อนยกมือค้าง รอบหน้าทำคะแนนแซงเดิมได้แน่ครับ',
    };
  }

  if (curAcc < 65) {
    return {
      code: 'TRY_HARDER',
      tone: 'amber',
      badgeTh: 'พยายามเพิ่มอีกนิด',
      headlineTh: ENCOURAGE_TRY_HARDER_PHRASE_TH,
      detailTh: `คะแนนมินิเกมอยู่ที่ ${curAcc}% (${curCorrect}/${curTotal}) ใกล้เคียงรอบก่อน (${prevAcc}%) แต่ยังมีช่องให้พัฒนาได้อีก`,
      adviceTh: 'ค่อยๆ อ่านและฟังคำถามทีละข้อ ตั้งเป้าเพิ่มเป็น 75-80% ในรอบถัดไปนะครับ',
    };
  }

  const isPeakStable = curAcc >= 90;
  return {
    code: 'STABLE',
    tone: 'sky',
    badgeTh: 'เท่าเดิม',
    headlineTh: isPeakStable
      ? `เท่าเดิมในระดับยอดเยี่ยม (${curAcc}% • ${curCorrect}/${curTotal})`
      : `ผลมินิเกมเท่าเดิม (${curAcc}% เทียบครั้งก่อน ${prevAcc}%)`,
    detailTh: isPeakStable
      ? 'รักษาระดับความแม่นยำสูงสุดได้อย่างสม่ำเสมอ'
      : 'คะแนนทรงตัวสม่ำเสมอ ใกล้เคียงรอบก่อนหน้า',
    adviceTh: isPeakStable
      ? 'ยอดเยี่ยมมากครับ รักษามาตรฐานนี้ต่อไปนะครับ'
      : 'อีกนิดเดียวจะขยับขึ้นระดับดีเยี่ยม ลองตั้งสมาธิเพิ่มขึ้นอีกเล็กน้อยครับ',
  };
}

/**
 * Extracts, strictly isolates, deduplicates, and analyzes all minigame sessions for one patient.
 */
export function buildPatientMinigameSummary(
  patientId: string,
  sessions?: readonly RehabSessionRecord[] | null,
): MinigameProgressSummary {
  const targetPid = String(patientId ?? '').trim();
  const rawMatches = (sessions ?? []).filter(
    (s) => s && String(s.patientId ?? '').trim() === targetPid && isMinigameRehabSession(s),
  );

  // Deduplicate by id or near-identical timestamp + score
  const sortedAsc = rawMatches
    .slice()
    .sort((a, b) => new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime());

  const uniqueRaw: RehabSessionRecord[] = [];
  const seenIds = new Set<string>();
  for (const s of sortedAsc) {
    if (s.id && seenIds.has(s.id)) continue;
    const sTs = new Date(s.createdAt).getTime();
    const dupNear = uniqueRaw.some((ex) => {
      const exTs = new Date(ex.createdAt).getTime();
      return (
        Number.isFinite(sTs) &&
        Number.isFinite(exTs) &&
        Math.abs(sTs - exTs) <= 10_000 &&
        ex.accuracyPercent === s.accuracyPercent &&
        ex.completedReps === s.completedReps
      );
    });
    if (dupNear) continue;
    if (s.id) seenIds.add(s.id);
    uniqueRaw.push(s);
  }

  const normalized: NormalizedMinigameSession[] = [];
  uniqueRaw.forEach((s, idx) => {
    const mg = s.minigameScore as
      | { correct?: number; total?: number; category?: string; durationSec?: number; accuracyPct?: number }
      | undefined;
    const total = Math.max(1, Math.round(Number(mg?.total ?? s.completedReps ?? 10) || 10));
    const rawCorrect =
      typeof mg?.correct === 'number' && Number.isFinite(mg.correct)
        ? Math.max(0, Math.min(total, Math.round(mg.correct)))
        : Math.max(0, Math.min(total, Math.round((clamp(s.accuracyPercent ?? 0, 0, 100) / 100) * total)));
    const computedAcc =
      typeof mg?.accuracyPct === 'number' && Number.isFinite(mg.accuracyPct)
        ? Math.round(clamp(mg.accuracyPct, 0, 100))
        : typeof mg?.correct === 'number' && typeof mg?.total === 'number' && mg.total > 0
          ? Math.round(clamp((rawCorrect / total) * 100, 0, 100))
          : Math.round(clamp(Number(s.accuracyPercent ?? 0), 0, 100));
    const durationSec =
      typeof mg?.durationSec === 'number' && Number.isFinite(mg.durationSec) && mg.durationSec > 0
        ? Math.round(mg.durationSec)
        : null;

    const prev = idx > 0 ? normalized[idx - 1] : null;
    const deltaAccuracyFromPrev = prev ? computedAcc - prev.accuracyPct : null;
    const deltaCorrectFromPrev = prev ? rawCorrect - prev.correct : null;

    const verdict = evaluateMinigameRoundVerdict({
      currentAccuracyPct: computedAcc,
      currentCorrect: rawCorrect,
      currentTotal: total,
      currentDurationSec: durationSec,
      previousAccuracyPct: prev ? prev.accuracyPct : null,
      previousCorrect: prev ? prev.correct : null,
      previousDurationSec: prev ? prev.durationSec : null,
    });

    normalized.push({
      id: s.id || `mg-${targetPid}-${idx + 1}`,
      patientId: targetPid,
      playedAt: s.createdAt || new Date().toISOString(),
      dateShortTh: formatThaiShortDate(s.createdAt || new Date().toISOString()),
      dateFullTh: formatThaiClinicalDate(s.createdAt || new Date().toISOString()),
      roundNumber: idx + 1,
      titleTh: (s.exerciseName ?? '').trim() || 'มินิเกมฝึกสมอง (การ์ดกายภาพ)',
      category: String(mg?.category ?? 'cognitive-card'),
      correct: rawCorrect,
      total,
      accuracyPct: computedAcc,
      durationSec,
      score0To6Equivalent: mapAccuracyPctToScale0To6(computedAcc),
      facilityName: s.facilityName || 'ตู้ Kiosk กายภาพอัจฉริยะ Dr.Tech.Care',
      deltaAccuracyFromPrev,
      deltaCorrectFromPrev,
      verdict,
    });
  });

  if (normalized.length === 0) {
    return {
      totalRounds: 0,
      sessions: [],
      firstSession: null,
      previousSession: null,
      latestSession: null,
      bestSession: null,
      averageAccuracyPct: null,
      latestAccuracyPct: null,
      bestAccuracyPct: null,
      deltaAccuracyFromPrevious: null,
      deltaAccuracyFromFirst: null,
      deltaCorrectFromPrevious: null,
      verdict: {
        code: 'NO_DATA',
        tone: 'slate',
        badgeTh: 'รอผลมินิเกม',
        headlineTh: 'ยังไม่มีประวัติการเล่นมินิเกมฝึกสมอง',
        detailTh: 'เมื่อผู้ป่วยเล่นมินิเกมการ์ดกายภาพที่ตู้ Kiosk ระบบจะซิงก์คะแนนและวิเคราะห์พัฒนาการให้อัตโนมัติ',
        adviceTh: 'แนะนำเล่นมินิเกมหลังจบกายภาพเพื่อกระตุ้นความจำและการตัดสินใจครับ',
      },
    };
  }

  const firstSession = normalized[0];
  const latestSession = normalized[normalized.length - 1];
  const previousSession = normalized.length >= 2 ? normalized[normalized.length - 2] : null;
  const bestSession = normalized.reduce((best, cand) => {
    if (cand.accuracyPct > best.accuracyPct) return cand;
    if (cand.accuracyPct === best.accuracyPct && cand.correct > best.correct) return cand;
    return best;
  }, normalized[0]);

  const averageAccuracyPct = round1(
    normalized.reduce((sum, item) => sum + item.accuracyPct, 0) / normalized.length,
  );
  const deltaAccuracyFromPrevious = previousSession
    ? latestSession.accuracyPct - previousSession.accuracyPct
    : null;
  const deltaAccuracyFromFirst =
    normalized.length >= 2 ? latestSession.accuracyPct - firstSession.accuracyPct : null;
  const deltaCorrectFromPrevious = previousSession
    ? latestSession.correct - previousSession.correct
    : null;

  return {
    totalRounds: normalized.length,
    sessions: normalized,
    firstSession,
    previousSession,
    latestSession,
    bestSession,
    averageAccuracyPct,
    latestAccuracyPct: latestSession.accuracyPct,
    bestAccuracyPct: bestSession.accuracyPct,
    deltaAccuracyFromPrevious,
    deltaAccuracyFromFirst,
    deltaCorrectFromPrevious,
    verdict: latestSession.verdict,
  };
}

/**
 * Evaluates physical therapy progression accurately.
 * Compares the latest visit against the previous visit of the SAME exercise/side first
 * (avoiding false regressions when two consecutive visits are different joints),
 * and recognizes peak clinical attainment (6/6 and >= 100% target ROM).
 */
export function evaluateRehabDomainProgress(params: {
  longitudinalSummary: LongitudinalProgressSummary;
  stockIndex?: RehabStockIndexSummary | null;
}): RehabDomainProgressSummary {
  const { longitudinalSummary, stockIndex } = params;
  const visits = longitudinalSummary.visitsWithDelta;
  const stockIndexPct = stockIndex?.latestPoint ? stockIndex.latestPoint.smartTotalPct : null;
  const stockDeltaPct = stockIndex?.latestPoint ? stockIndex.latestPoint.deltaSmartPct : null;

  if (visits.length === 0) {
    return {
      totalVisits: 0,
      latestRomDeg: null,
      latestTargetPct: null,
      latestScore0To6: null,
      latestExerciseNameTh: null,
      comparedSameExercise: false,
      deltaRomDeg: null,
      deltaTargetPct: null,
      deltaScore0To6: null,
      stockIndexPct,
      stockDeltaPct,
      verdict: {
        code: 'NO_DATA',
        tone: 'slate',
        badgeTh: 'รอผลกายภาพ',
        headlineTh: 'ยังไม่มีประวัติการฝึกกายภาพบำบัดหรือประเมิน ROM',
        detailTh: 'เมื่อผู้ป่วยฝึกกายภาพที่ตู้ Kiosk หรือบันทึกผลประเมิน 0-6 ระบบจะคำนวณพัฒนาการให้ทันที',
        adviceTh: 'เริ่มฝึกท่ากายภาพตามคอร์สหรือกดบันทึกผลประเมินครั้งแรกได้เลยครับ',
      },
    };
  }

  const latest = visits[visits.length - 1];
  const latestCode = (latest.record.exercise_code || '').trim().toUpperCase();
  const latestSide = latest.record.target_side;

  // Prefer comparing against the most recent previous visit of the EXACT SAME exercise & side
  let prevSameExercise = null;
  for (let i = visits.length - 2; i >= 0; i -= 1) {
    const cand = visits[i];
    if (
      (cand.record.exercise_code || '').trim().toUpperCase() === latestCode &&
      cand.record.target_side === latestSide
    ) {
      prevSameExercise = cand;
      break;
    }
  }
  const prevFallback = visits.length >= 2 ? visits[visits.length - 2] : null;
  const baselineVisit = prevSameExercise ?? prevFallback;
  const comparedSameExercise = Boolean(prevSameExercise);

  const latestRomDeg = round1(latest.record.measured_rom_deg);
  const latestTargetPct = round1(latest.targetAchievementPct);
  const latestScore0To6 = latest.record.initial_score_0_to_6;
  const latestExerciseNameTh = latest.record.exercise_name_th || latest.record.exercise_code;

  const deltaRomDeg =
    baselineVisit && comparedSameExercise
      ? round1(latest.record.measured_rom_deg - baselineVisit.record.measured_rom_deg)
      : null;
  const deltaTargetPct = baselineVisit
    ? round1(latest.targetAchievementPct - baselineVisit.targetAchievementPct)
    : null;
  const deltaScore0To6 = baselineVisit
    ? latest.record.initial_score_0_to_6 - baselineVisit.record.initial_score_0_to_6
    : null;

  // Single visit (baseline)
  if (!baselineVisit) {
    if (latestScore0To6 >= 5 || latestTargetPct >= 85) {
      return {
        totalVisits: visits.length,
        latestRomDeg,
        latestTargetPct,
        latestScore0To6,
        latestExerciseNameTh,
        comparedSameExercise: false,
        deltaRomDeg: null,
        deltaTargetPct: null,
        deltaScore0To6: null,
        stockIndexPct,
        stockDeltaPct,
        verdict: {
          code: 'FIRST_SESSION',
          tone: 'emerald',
          badgeTh: 'เริ่มต้นได้ดีมาก',
          headlineTh: `เริ่มต้นกายภาพได้ดีมาก! ระดับ ${latestScore0To6}/6 (ROM ${latestRomDeg.toFixed(1)}° • ${latestTargetPct.toFixed(0)}%)`,
          detailTh: `ท่า ${latestExerciseNameTh} บรรลุเป้าหมายคลินิกตั้งแต่รอบแรก`,
          adviceTh: 'รักษาวินัยการฝึกสม่ำเสมอตามตารางเพื่อคงความแข็งแรงนะครับ',
        },
      };
    }
    if (latestScore0To6 >= 3 || latestTargetPct >= 65) {
      return {
        totalVisits: visits.length,
        latestRomDeg,
        latestTargetPct,
        latestScore0To6,
        latestExerciseNameTh,
        comparedSameExercise: false,
        deltaRomDeg: null,
        deltaTargetPct: null,
        deltaScore0To6: null,
        stockIndexPct,
        stockDeltaPct,
        verdict: {
          code: 'FIRST_SESSION',
          tone: 'sky',
          badgeTh: 'บันทึกครั้งแรก',
          headlineTh: `บันทึกผลกายภาพครั้งแรก ระดับ ${latestScore0To6}/6 (ROM ${latestRomDeg.toFixed(1)}° • ${latestTargetPct.toFixed(0)}%)`,
          detailTh: `เก็บค่าตั้งต้นท่า ${latestExerciseNameTh} เรียบร้อยแล้ว`,
          adviceTh: 'ฝึกต่อเนื่องตามรอบนัดเพื่อค่อยๆ ขยายมุมองศาให้ถึงเป้าหมายครับ',
        },
      };
    }
    return {
      totalVisits: visits.length,
      latestRomDeg,
      latestTargetPct,
      latestScore0To6,
      latestExerciseNameTh,
      comparedSameExercise: false,
      deltaRomDeg: null,
      deltaTargetPct: null,
      deltaScore0To6: null,
      stockIndexPct,
      stockDeltaPct,
      verdict: {
        code: 'TRY_HARDER',
        tone: 'amber',
        badgeTh: 'พยายามเพิ่มอีกนิด',
        headlineTh: ENCOURAGE_TRY_HARDER_PHRASE_TH,
        detailTh: `รอบแรกได้ระดับ ${latestScore0To6}/6 (ROM ${latestRomDeg.toFixed(1)}° หรือ ${latestTargetPct.toFixed(0)}% ของเป้าหมาย)`,
        adviceTh: 'ค่อยๆ ขยับข้อต่อตามจังหวะหุ่นสาธิตโดยไม่ฝืนจนปวด แล้วรอบหน้าจะดีขึ้นแน่นอนครับ',
      },
    };
  }

  // Multi-visit comparison
  // Check peak attainment first: if BOTH latest and baseline are at max 6/6 and >= 100% target,
  // never falsely penalize different exercises (e.g. 146% -> 117.9% is still exceeding 100% target!)
  const isPeakAttainment =
    latestScore0To6 >= 6 &&
    latestTargetPct >= 100 &&
    (!comparedSameExercise || (deltaRomDeg !== null && deltaRomDeg > -8));

  const stockImproved = stockDeltaPct !== null && stockDeltaPct >= 3.0;
  const stockDropped = stockDeltaPct !== null && stockDeltaPct <= -5.0 && (stockIndexPct ?? 100) < 90;

  let verdict: ProgressVerdictSummary;

  if (
    (comparedSameExercise && deltaRomDeg !== null && deltaRomDeg >= 3.0) ||
    (deltaScore0To6 !== null && deltaScore0To6 >= 1) ||
    (!comparedSameExercise && !isPeakAttainment && deltaTargetPct !== null && deltaTargetPct >= 4.0) ||
    (stockImproved && (deltaScore0To6 ?? 0) >= 0)
  ) {
    const gainText =
      comparedSameExercise && deltaRomDeg !== null && deltaRomDeg > 0
        ? `มุมองศาเพิ่ม +${deltaRomDeg.toFixed(1)}°`
        : deltaScore0To6 !== null && deltaScore0To6 > 0
          ? `คะแนนมหิดลเพิ่ม +${deltaScore0To6} ระดับ`
          : stockDeltaPct !== null && stockDeltaPct > 0
            ? `ดัชนีรวมเพิ่ม +${stockDeltaPct.toFixed(1)}%`
            : `เทียบเป้าหมายเพิ่ม +${(deltaTargetPct ?? 0).toFixed(1)}%`;
    verdict = {
      code: 'IMPROVED',
      tone: 'emerald',
      badgeTh: 'ดีขึ้น',
      headlineTh: `พัฒนาการกายภาพดีขึ้น! (${gainText})`,
      detailTh: `ล่าสุดท่า ${latestExerciseNameTh} ทำได้ ${latestRomDeg.toFixed(1)}° (${latestTargetPct.toFixed(0)}% เป้าหมาย • ระดับ ${latestScore0To6}/6)`,
      adviceTh: 'ทำได้ดีมากครับ รักษาจังหวะการฝึกและค้างท่าให้สม่ำเสมอต่อไปนะครับ',
    };
  } else if (isPeakAttainment) {
    verdict = {
      code: 'STABLE',
      tone: 'emerald',
      badgeTh: 'เท่าเดิม (ยอดเยี่ยม)',
      headlineTh: `ผลกายภาพเท่าเดิมในระดับยอดเยี่ยมสูงสุด (${latestScore0To6}/6 • ${latestTargetPct.toFixed(0)}% เป้าหมาย)`,
      detailTh: `ทุกท่าที่ฝึกบรรลุเกินเป้าหมายคลินิกแล้ว (ล่าสุด ${latestExerciseNameTh} ${latestRomDeg.toFixed(1)}°)`,
      adviceTh: 'รักษามาตรฐานสูงสุดได้ดีมากครับ สามารถฝึกต่อเนื่องเพื่อคงความยืดหยุ่นของข้อต่อได้เลย',
    };
  } else if (
    (comparedSameExercise && deltaRomDeg !== null && deltaRomDeg <= -3.0) ||
    (deltaScore0To6 !== null && deltaScore0To6 <= -1) ||
    (!comparedSameExercise && deltaTargetPct !== null && deltaTargetPct <= -5.0 && latestTargetPct < 95) ||
    stockDropped ||
    (latestTargetPct < 65 && latestScore0To6 <= 3)
  ) {
    const dipDetail =
      comparedSameExercise && deltaRomDeg !== null && deltaRomDeg < 0
        ? `มุมองศาท่า ${latestExerciseNameTh} ลดลง ${Math.abs(deltaRomDeg).toFixed(1)}° จากครั้งก่อน (ล่าสุด ${latestRomDeg.toFixed(1)}°)`
        : deltaScore0To6 !== null && deltaScore0To6 < 0
          ? `ระดับคะแนนมหิดลลดลงจาก ${baselineVisit.record.initial_score_0_to_6}/6 เป็น ${latestScore0To6}/6`
          : `ความสำเร็จเทียบเป้าหมายอยู่ที่ ${latestTargetPct.toFixed(0)}% (ระดับ ${latestScore0To6}/6)`;
    verdict = {
      code: 'TRY_HARDER',
      tone: 'amber',
      badgeTh: 'พยายามเพิ่มอีกนิด',
      headlineTh: ENCOURAGE_TRY_HARDER_PHRASE_TH,
      detailTh: dipDetail,
      adviceTh: 'ค่อยๆ ยืดเหยียดและยกให้ถึงมุมเป้าหมายพร้อมค้างท่าให้ครบเวลา หากมีอาการตึงหรือปวดแจ้งนักกายภาพได้เลยครับ',
    };
  } else {
    verdict = {
      code: 'STABLE',
      tone: 'sky',
      badgeTh: 'เท่าเดิม',
      headlineTh: `ผลกายภาพเท่าเดิม / ทรงตัว (ระดับ ${latestScore0To6}/6 • ${latestTargetPct.toFixed(0)}% เป้าหมาย)`,
      detailTh: comparedSameExercise && deltaRomDeg !== null
        ? `ท่า ${latestExerciseNameTh} เปลี่ยนแปลง ${deltaRomDeg >= 0 ? '+' : ''}${deltaRomDeg.toFixed(1)}° ใกล้เคียงรอบก่อนหน้า`
        : `รักษาระดับความสามารถได้สม่ำเสมอ ล่าสุด ${latestRomDeg.toFixed(1)}°`,
      adviceTh: 'รักษาวินัยได้ดีครับ ลองเพิ่มมุมการเคลื่อนไหวอีกเล็กน้อยในครั้งถัดไปเพื่อขยับระดับขึ้นครับ',
    };
  }

  return {
    totalVisits: visits.length,
    latestRomDeg,
    latestTargetPct,
    latestScore0To6,
    latestExerciseNameTh,
    comparedSameExercise,
    deltaRomDeg,
    deltaTargetPct,
    deltaScore0To6,
    stockIndexPct,
    stockDeltaPct,
    verdict,
  };
}

/**
 * Formats a concise human-friendly verdict tag for an individual exercise breakdown row.
 */
export function resolveExerciseRowVerdict(item: ExerciseComparisonBreakdownItem): {
  code: ProgressVerdictCode;
  labelTh: string;
  tone: ProgressVerdictTone;
} {
  if (item.visitCount < 2) {
    if (item.latestTargetAchievementPct >= 85 || item.latestScore >= 5) {
      return { code: 'FIRST_SESSION', labelTh: 'ครั้งแรก (ดีมาก)', tone: 'emerald' };
    }
    if (item.latestTargetAchievementPct < 65 && item.latestScore <= 3) {
      return { code: 'TRY_HARDER', labelTh: 'พยายามเพิ่ม', tone: 'amber' };
    }
    return { code: 'FIRST_SESSION', labelTh: 'ค่าตั้งต้น', tone: 'sky' };
  }
  if (item.trendStatus === 'IMPROVED') {
    return {
      code: 'IMPROVED',
      labelTh: `ดีขึ้น (+${item.deltaRomDeg.toFixed(1)}°)`,
      tone: 'emerald',
    };
  }
  if (item.trendStatus === 'REGRESSED') {
    return {
      code: 'TRY_HARDER',
      labelTh: `พยายามเพิ่ม (${item.deltaRomDeg.toFixed(1)}°)`,
      tone: 'amber',
    };
  }
  return {
    code: 'STABLE',
    labelTh: item.latestScore >= 6 && item.latestTargetAchievementPct >= 100 ? 'เท่าเดิม (ยอดเยี่ยม)' : 'เท่าเดิม',
    tone: 'sky',
  };
}

/**
 * Combines Rehab + Minigame evaluations into one unified clinical & patient-facing summary,
 * plus a single chronological activity timeline containing both physical therapy and minigames.
 */
export function buildUnifiedRehabMinigameSummary(params: {
  patientId: string;
  longitudinalSummary: LongitudinalProgressSummary;
  stockIndex?: RehabStockIndexSummary | null;
  sessions?: readonly RehabSessionRecord[] | null;
}): UnifiedRehabMinigameSummary {
  const { patientId, longitudinalSummary, stockIndex, sessions } = params;
  const rehab = evaluateRehabDomainProgress({ longitudinalSummary, stockIndex });
  const minigame = buildPatientMinigameSummary(patientId, sessions);

  const hasRehab = rehab.totalVisits > 0;
  const hasMinigame = minigame.totalRounds > 0;
  const totalActivities = rehab.totalVisits + minigame.totalRounds;

  const rehabEffectivePct = hasRehab
    ? clamp(
        rehab.stockIndexPct !== null
          ? rehab.stockIndexPct
          : Math.min(100, rehab.latestTargetPct ?? ((rehab.latestScore0To6 ?? 0) / 6) * 100),
        0,
        100,
      )
    : null;
  const minigameEffectivePct = hasMinigame ? minigame.latestAccuracyPct : null;

  const combinedScorePct =
    rehabEffectivePct !== null && minigameEffectivePct !== null
      ? round1(rehabEffectivePct * 0.6 + minigameEffectivePct * 0.4)
      : rehabEffectivePct !== null
        ? round1(rehabEffectivePct)
        : minigameEffectivePct !== null
          ? round1(minigameEffectivePct)
          : null;

  let overallVerdict: ProgressVerdictSummary;

  if (!hasRehab && !hasMinigame) {
    overallVerdict = {
      code: 'NO_DATA',
      tone: 'slate',
      badgeTh: 'รอข้อมูลการฝึก',
      headlineTh: 'ยังไม่มีข้อมูลการกายภาพบำบัดหรือการเล่นมินิเกม',
      detailTh: 'เมื่อผู้ป่วยเริ่มฝึกกายภาพหรือเล่นมินิเกมที่ตู้ Kiosk ระบบจะซิงก์และสรุปผลเปรียบเทียบให้แบบเรียลไทม์',
      adviceTh: 'เริ่มฝึกกายภาพหรือเล่นมินิเกมได้เลยครับ',
    };
  } else if (hasRehab && !hasMinigame) {
    overallVerdict = {
      ...rehab.verdict,
      detailTh: `${rehab.verdict.detailTh} • ยังไม่มีประวัติมินิเกม`,
    };
  } else if (!hasRehab && hasMinigame) {
    overallVerdict = {
      ...minigame.verdict,
      detailTh: `${minigame.verdict.detailTh} • ยังไม่มีประวัติกายภาพ ROM`,
    };
  } else {
    // Both Rehab AND Minigame exist -> synthesize unified synchronized verdict
    const rCode = rehab.verdict.code;
    const mCode = minigame.verdict.code;

    if (rCode === 'IMPROVED' && mCode === 'IMPROVED') {
      overallVerdict = {
        code: 'IMPROVED',
        tone: 'emerald',
        badgeTh: 'ดีขึ้นทั้งกายภาพและมินิเกม',
        headlineTh: 'ยอดเยี่ยมมาก! ดีขึ้นชัดเจนทั้งกายภาพบำบัดและมินิเกมฝึกสมอง',
        detailTh: `กายภาพ: ${rehab.verdict.badgeTh} (${rehab.latestScore0To6}/6 • ${rehab.latestTargetPct?.toFixed(0)}%) | มินิเกม: ${minigame.verdict.badgeTh} (${minigame.latestSession?.correct}/${minigame.latestSession?.total} • ${minigame.latestAccuracyPct}%)`,
        adviceTh: 'พัฒนาการสอดคล้องดีเยี่ยมทั้งร่างกายและสมอง รักษาวินัยการฝึกต่อไปนะครับ',
      };
    } else if (rCode === 'TRY_HARDER' && mCode === 'TRY_HARDER') {
      overallVerdict = {
        code: 'TRY_HARDER',
        tone: 'amber',
        badgeTh: 'พยายามเพิ่มอีกนิด',
        headlineTh: ENCOURAGE_TRY_HARDER_PHRASE_TH,
        detailTh: `ทั้งผลกายภาพ (${rehab.latestScore0To6}/6 • ${rehab.latestTargetPct?.toFixed(0)}%) และมินิเกม (${minigame.latestAccuracyPct}%) ยังปรับลดลงหรือต่ำกว่าเป้าเล็กน้อย`,
        adviceTh: 'ค่อยๆ ฝึกทีละจังหวะ ไม่ต้องรีบเร่ง พักให้เพียงพอแล้วมาลุยรอบหน้าด้วยกันนะครับ',
      };
    } else if (rCode === 'TRY_HARDER' || mCode === 'TRY_HARDER') {
      const strongSide = rCode === 'TRY_HARDER' ? 'มินิเกมฝึกสมอง' : 'กายภาพบำบัด';
      const weakSide = rCode === 'TRY_HARDER' ? 'กายภาพบำบัด' : 'มินิเกมฝึกสมอง';
      overallVerdict = {
        code: 'TRY_HARDER',
        tone: 'amber',
        badgeTh: 'พยายามเพิ่มอีกนิด',
        headlineTh: `${ENCOURAGE_TRY_HARDER_PHRASE_TH} (เน้นเสริมด้าน${weakSide})`,
        detailTh: `${strongSide}ทำได้ดี แต่${weakSide}รอบล่าสุดลดลงหรือยังไม่ถึงเป้า • กายภาพ ${rehab.latestScore0To6}/6 (${rehab.latestTargetPct?.toFixed(0)}%) | มินิเกม ${minigame.latestAccuracyPct}%`,
        adviceTh: rCode === 'TRY_HARDER' ? rehab.verdict.adviceTh : minigame.verdict.adviceTh,
      };
    } else if (rCode === 'IMPROVED' || mCode === 'IMPROVED') {
      const improvedSide = rCode === 'IMPROVED' ? 'กายภาพบำบัด' : 'มินิเกมฝึกสมอง';
      overallVerdict = {
        code: 'IMPROVED',
        tone: 'emerald',
        badgeTh: 'ดีขึ้น',
        headlineTh: `พัฒนาการโดยรวมดีขึ้น! โดดเด่นด้าน${improvedSide}`,
        detailTh: `กายภาพ: ${rehab.verdict.badgeTh} (${rehab.latestScore0To6}/6 • ${rehab.latestTargetPct?.toFixed(0)}%) | มินิเกม: ${minigame.verdict.badgeTh} (${minigame.latestSession?.correct}/${minigame.latestSession?.total} • ${minigame.latestAccuracyPct}%)`,
        adviceTh: 'กำลังเดินมาถูกทางครับ ฝึกต่อเนื่องทั้งกายภาพและมินิเกมเพื่อพัฒนาให้ครบทุกด้าน',
      };
    } else if (rCode === 'FIRST_SESSION' && mCode === 'FIRST_SESSION') {
      overallVerdict = {
        code: 'FIRST_SESSION',
        tone: 'emerald',
        badgeTh: 'เริ่มต้นครบทั้งคู่',
        headlineTh: `เริ่มต้นได้ดีครบทั้งกายภาพและมินิเกม (คะแนนรวมซิงก์ ${combinedScorePct?.toFixed(1)}%)`,
        detailTh: `กายภาพรอบแรก ${rehab.latestScore0To6}/6 (${rehab.latestTargetPct?.toFixed(0)}%) | มินิเกมรอบแรก ${minigame.latestSession?.correct}/${minigame.latestSession?.total} (${minigame.latestAccuracyPct}%)`,
        adviceTh: 'ระบบเก็บฐานข้อมูลครบทั้งสองด้านแล้ว รอบถัดไปจะเปรียบเทียบพัฒนาการให้ทันทีครับ',
      };
    } else {
      const peakBoth =
        (rehab.latestScore0To6 ?? 0) >= 6 &&
        (rehab.latestTargetPct ?? 0) >= 100 &&
        (minigame.latestAccuracyPct ?? 0) >= 90;
      overallVerdict = {
        code: 'STABLE',
        tone: peakBoth ? 'emerald' : 'sky',
        badgeTh: peakBoth ? 'เท่าเดิม (ยอดเยี่ยม)' : 'เท่าเดิม',
        headlineTh: peakBoth
          ? `ผลรวมเท่าเดิมในระดับยอดเยี่ยมสูงสุด (${combinedScorePct?.toFixed(1)}%)`
          : `ผลรวมกายภาพและมินิเกมเท่าเดิม / ทรงตัว (${combinedScorePct?.toFixed(1)}%)`,
        detailTh: `กายภาพ ${rehab.latestScore0To6}/6 (${rehab.latestTargetPct?.toFixed(0)}%) | มินิเกม ${minigame.latestSession?.correct}/${minigame.latestSession?.total} (${minigame.latestAccuracyPct}%)`,
        adviceTh: peakBoth
          ? 'รักษามาตรฐานสูงสุดได้ดีมากทั้งร่างกายและสมองครับ'
          : 'รักษาระดับได้ดีสม่ำเสมอ ลองท้าทายเพิ่มอีกเล็กน้อยในรอบถัดไปนะครับ',
      };
    }
  }

  // Build unified chronological timeline (both Physio visits + Minigame rounds)
  const timeline: UnifiedActivityTimelineItem[] = [];

  // Track previous same-exercise physio visit to produce accurate per-card delta & verdict
  const prevByExerciseKey = new Map<string, (typeof longitudinalSummary.visitsWithDelta)[number]>();
  for (const visit of longitudinalSummary.visitsWithDelta) {
    const rec = visit.record;
    const exKey = `${(rec.exercise_code || '').toUpperCase()}::${rec.target_side}`;
    const prevSame = prevByExerciseKey.get(exKey) ?? null;
    prevByExerciseKey.set(exKey, visit);

    const sameDeltaRom = prevSame
      ? round1(rec.measured_rom_deg - prevSame.record.measured_rom_deg)
      : null;
    const sameDeltaScore = prevSame
      ? rec.initial_score_0_to_6 - prevSame.record.initial_score_0_to_6
      : null;

    let itemVerdict: ProgressVerdictSummary;
    if (!prevSame) {
      itemVerdict =
        visit.targetAchievementPct >= 85 || rec.initial_score_0_to_6 >= 5
          ? {
              code: 'FIRST_SESSION',
              tone: 'emerald',
              badgeTh: 'รอบแรก (ดีมาก)',
              headlineTh: 'เริ่มต้นท่านี้ได้ดีมาก',
              detailTh: '',
              adviceTh: '',
            }
          : visit.targetAchievementPct < 65 && rec.initial_score_0_to_6 <= 3
            ? {
                code: 'TRY_HARDER',
                tone: 'amber',
                badgeTh: 'พยายามเพิ่ม',
                headlineTh: ENCOURAGE_TRY_HARDER_PHRASE_TH,
                detailTh: '',
                adviceTh: '',
              }
            : {
                code: 'FIRST_SESSION',
                tone: 'sky',
                badgeTh: 'ค่าตั้งต้น',
                headlineTh: 'บันทึกค่าตั้งต้น',
                detailTh: '',
                adviceTh: '',
              };
    } else if ((sameDeltaRom !== null && sameDeltaRom >= 3) || (sameDeltaScore !== null && sameDeltaScore >= 1)) {
      itemVerdict = {
        code: 'IMPROVED',
        tone: 'emerald',
        badgeTh: 'ดีขึ้น',
        headlineTh: 'ดีขึ้นกว่าครั้งก่อน',
        detailTh: '',
        adviceTh: '',
      };
    } else if ((sameDeltaRom !== null && sameDeltaRom <= -3) || (sameDeltaScore !== null && sameDeltaScore <= -1)) {
      itemVerdict = {
        code: 'TRY_HARDER',
        tone: 'amber',
        badgeTh: 'พยายามเพิ่ม',
        headlineTh: ENCOURAGE_TRY_HARDER_PHRASE_TH,
        detailTh: '',
        adviceTh: '',
      };
    } else {
      itemVerdict = {
        code: 'STABLE',
        tone: 'sky',
        badgeTh: 'เท่าเดิม',
        headlineTh: 'เท่าเดิม (ทรงตัว)',
        detailTh: '',
        adviceTh: '',
      };
    }

    const sourceLabel =
      rec.source_channel === 'COURSE_AUTO_CALCULATED'
        ? 'คำนวณอัตโนมัติจากคอร์ส'
        : rec.source_channel === 'KIOSK_SESSION_AUTO_SYNC'
          ? 'ทำจากตู้ Kiosk'
          : 'ประเมินโดยบุคลากร';

    timeline.push({
      id: `physio-${rec.id}`,
      kind: 'physio',
      timestampIso: rec.assessed_at,
      dateFullTh: formatThaiClinicalDate(rec.assessed_at),
      dateShortTh: formatThaiShortDate(rec.assessed_at),
      sequenceLabel: `กายภาพครั้งที่ ${rec.visit_number}`,
      titleTh: rec.exercise_name_th || rec.exercise_code,
      subtitleTh: `${sourceLabel} • ${rec.facility_name}`,
      facilityName: rec.facility_name,
      primaryMetricLabel: `ROM ${rec.measured_rom_deg.toFixed(1)}° (${visit.targetAchievementPct.toFixed(0)}%)`,
      secondaryMetricLabel: `ระดับ ${rec.initial_score_0_to_6}/6`,
      deltaLabel:
        sameDeltaRom !== null
          ? `${sameDeltaRom >= 0 ? '+' : ''}${sameDeltaRom.toFixed(1)}°`
          : null,
      deltaPositive: sameDeltaRom !== null ? sameDeltaRom >= 0 : null,
      verdict: itemVerdict,
      physioRecord: rec,
    });
  }

  for (const mg of minigame.sessions) {
    timeline.push({
      id: `minigame-${mg.id}`,
      kind: 'minigame',
      timestampIso: mg.playedAt,
      dateFullTh: mg.dateFullTh,
      dateShortTh: mg.dateShortTh,
      sequenceLabel: `มินิเกมรอบที่ ${mg.roundNumber}`,
      titleTh: mg.titleTh,
      subtitleTh: `ทำจากตู้ Kiosk • ${mg.durationSec ? `ใช้เวลา ${mg.durationSec} วิ` : mg.facilityName}`,
      facilityName: mg.facilityName,
      primaryMetricLabel: `คะแนน ${mg.correct}/${mg.total} (${mg.accuracyPct}%)`,
      secondaryMetricLabel: `เทียบระดับ ${mg.score0To6Equivalent}/6`,
      deltaLabel:
        mg.deltaAccuracyFromPrev !== null
          ? `${mg.deltaAccuracyFromPrev >= 0 ? '+' : ''}${mg.deltaAccuracyFromPrev}%`
          : null,
      deltaPositive: mg.deltaAccuracyFromPrev !== null ? mg.deltaAccuracyFromPrev >= 0 : null,
      verdict: mg.verdict,
      minigameSession: mg,
    });
  }

  // Sort timeline chronologically descending (newest first for UI timeline display)
  timeline.sort((a, b) => new Date(b.timestampIso).getTime() - new Date(a.timestampIso).getTime());

  return {
    patientId,
    rehab,
    minigame,
    overall: {
      totalActivities,
      combinedScorePct,
      syncReady: totalActivities > 0,
      verdict: overallVerdict,
    },
    timeline,
  };
}

export interface ExerciseComparisonBarItem {
  exerciseCode: string;
  shortLabelTh: string;
  fullNameTh: string;
  previousRomDeg: number | null;
  currentRomDeg: number;
  targetRomDeg: number;
  deltaDeg: number | null;
  achievedPct: number;
  status: 'IMPROVED' | 'STABLE' | 'TRY_HARDER' | 'FIRST';
}

export interface PatientProgressSummaryReport {
  patientName: string;
  patientHn: string;
  periodTitleTh: string;
  periodRangeTh: string;
  overallProgressPct: number;
  weeklySessionCount: number;
  weeklyGoalCount: number;
  weeklyGoalMet: boolean;
  avgScore0To6: number | null;
  avgRomDeg: number | null;
  avgRomDeltaDeg: number | null;
  exerciseBars: ExerciseComparisonBarItem[];
  minigameComparison: {
    played: boolean;
    totalRounds: number;
    previousAccuracyPct: number | null;
    currentAccuracyPct: number | null;
    currentCorrect: number | null;
    currentTotal: number | null;
    deltaAccuracyPct: number | null;
    verdict: ProgressVerdictSummary;
  };
  overallVerdict: ProgressVerdictSummary;
  recommendationTh: string;
  voiceSummaryTh: string;
}

/**
 * Produces a concise, elderly-friendly Thai label for X-axis bar chart slots
 * (e.g. "[Stroke ADL] ยกแขนไปด้านหน้า" -> "ยกแขน", "กางแขนด้านข้าง" -> "กางแขน").
 */
export function shortenThaiExerciseLabel(rawNameTh: string, fallbackCode?: string): string {
  const clean = String(rawNameTh ?? '')
    .replace(/^\[[^\]]+\]\s*/g, '')
    .replace(/\([^)]*\)/g, '')
    .trim();
  if (!clean) return fallbackCode || 'ท่ากายภาพ';
  if (clean.includes('ประสานมือ')) return 'ประสานมือ';
  if (clean.includes('ยกแขน')) return 'ยกแขน';
  if (clean.includes('กางแขน')) return 'กางแขน';
  if (clean.includes('งอข้อศอก') || clean.includes('งอศอก')) return 'งอศอก';
  if (clean.includes('หมุนข้อไหล่') || clean.includes('หมุนไหล่')) return 'หมุนไหล่';
  if (clean.includes('กระดกข้อมือ') || clean.includes('ข้อมือ')) return 'ข้อมือ';
  if (clean.includes('หันคอ') || clean.includes('เอียงคอ')) return 'บริหารคอ';
  if (clean.includes('เหยียดเข่า') || clean.includes('งอเข่า')) return 'เหยียดเข่า';
  if (clean.includes('ยกขา') || clean.includes('กางขา')) return 'ยกขา';
  if (clean.includes('ลุกยืน') || clean.includes('นั่งยอง')) return 'ลุก-นั่ง';
  if (clean.length <= 12) return clean;
  return `${clean.slice(0, 11)}...`;
}

/**
 * Builds the complete patient-friendly Weekly & Course Completion Summary Report
 * (matching the clean clinical summary modal: overall progress %, weekly sessions,
 * average 0-6 score, ROM delta vs previous, grouped before-vs-now bar chart per exercise
 * + minigame comparison, and personalized Thai clinical advice).
 */
export function buildPatientProgressSummaryReport(params: {
  patientName: string;
  patientHn?: string;
  unifiedSummary: UnifiedRehabMinigameSummary;
  assessments: readonly PatientClinicalAssessmentRecord[];
  weeklyGoalCount?: number;
  referenceNow?: Date;
}): PatientProgressSummaryReport {
  const { patientName, patientHn = '', unifiedSummary, assessments } = params;
  const weeklyGoalCount = Math.max(1, params.weeklyGoalCount ?? 3);
  const now = params.referenceNow ?? new Date();
  const weekStart = new Date(now.getTime() - 6 * 86_400_000);
  const weekStartMs = weekStart.getTime();
  const weekNumberOfMonth = Math.min(5, Math.max(1, Math.ceil(now.getDate() / 7)));
  const periodTitleTh = 'สรุปผลการกายภาพและมินิเกม (Progress Summary)';
  const periodRangeTh = `สัปดาห์ที่ ${weekNumberOfMonth} (${formatThaiFullDateOnly(weekStart.toISOString())} - ${formatThaiFullDateOnly(now.toISOString())})`;

  const sortedAssessments = assessments
    .slice()
    .sort((a, b) => new Date(a.assessed_at).getTime() - new Date(b.assessed_at).getTime());

  // Count real activity sessions in the last 7 days (physio + minigame)
  let weeklyEvents = 0;
  for (const item of unifiedSummary.timeline) {
    const ts = new Date(item.timestampIso).getTime();
    if (Number.isFinite(ts) && ts >= weekStartMs) {
      weeklyEvents += 1;
    }
  }
  const weeklySessionCount =
    weeklyEvents > 0 ? weeklyEvents : unifiedSummary.overall.totalActivities;
  const weeklyGoalMet = weeklySessionCount >= weeklyGoalCount;

  // Group assessments by exercise code to build accurate before-vs-latest comparison bars
  const byExercise = new Map<string, PatientClinicalAssessmentRecord[]>();
  for (const rec of sortedAssessments) {
    const code = (rec.exercise_code || 'UNKNOWN').trim().toUpperCase();
    const list = byExercise.get(code) ?? [];
    list.push(rec);
    byExercise.set(code, list);
  }

  const exerciseBars: ExerciseComparisonBarItem[] = [];
  const sameExerciseDeltas: number[] = [];

  for (const [code, list] of byExercise.entries()) {
    const latest = list[list.length - 1];
    const previous = list.length >= 2 ? list[list.length - 2] : null;
    const currentRomDeg = round1(latest.measured_rom_deg);
    const previousRomDeg = previous ? round1(previous.measured_rom_deg) : null;
    const targetRomDeg = round1(latest.target_rom_deg || 120);
    const deltaDeg = previousRomDeg !== null ? round1(currentRomDeg - previousRomDeg) : null;
    const achievedPct = round1(
      targetRomDeg > 0 ? (currentRomDeg / targetRomDeg) * 100 : 0,
    );

    if (deltaDeg !== null) {
      sameExerciseDeltas.push(deltaDeg);
    }

    let status: ExerciseComparisonBarItem['status'] = 'FIRST';
    if (deltaDeg !== null) {
      if (deltaDeg >= 3.0) {
        status = 'IMPROVED';
      } else if (deltaDeg <= -3.0 && achievedPct < 100) {
        status = 'TRY_HARDER';
      } else {
        status = 'STABLE';
      }
    } else if (achievedPct < 65 && latest.initial_score_0_to_6 <= 3) {
      status = 'TRY_HARDER';
    }

    exerciseBars.push({
      exerciseCode: code,
      shortLabelTh: shortenThaiExerciseLabel(latest.exercise_name_th || code, code),
      fullNameTh: latest.exercise_name_th || code,
      previousRomDeg,
      currentRomDeg,
      targetRomDeg,
      deltaDeg,
      achievedPct,
      status,
    });
  }

  // Average 0-6 score across recent assessments (preferring current week if available)
  const recentWeekAssessments = sortedAssessments.filter((rec) => {
    const ts = new Date(rec.assessed_at).getTime();
    return Number.isFinite(ts) && ts >= weekStartMs;
  });
  const scorePool = recentWeekAssessments.length > 0 ? recentWeekAssessments : sortedAssessments;
  const avgScore0To6 =
    scorePool.length > 0
      ? round1(
          scorePool.reduce((sum, r) => sum + r.initial_score_0_to_6, 0) / scorePool.length,
        )
      : unifiedSummary.minigame.latestSession
        ? unifiedSummary.minigame.latestSession.score0To6Equivalent
        : null;

  const avgRomDeg =
    exerciseBars.length > 0
      ? round1(
          exerciseBars.reduce((sum, item) => sum + item.currentRomDeg, 0) / exerciseBars.length,
        )
      : null;

  const avgRomDeltaDeg =
    sameExerciseDeltas.length > 0
      ? round1(
          sameExerciseDeltas.reduce((sum, d) => sum + d, 0) / sameExerciseDeltas.length,
        )
      : unifiedSummary.rehab.deltaRomDeg;

  const overallProgressPct = Math.round(
    clamp(
      unifiedSummary.overall.combinedScorePct ??
        (avgScore0To6 !== null ? (avgScore0To6 / 6) * 100 : 0),
      0,
      100,
    ),
  );

  const mg = unifiedSummary.minigame;
  const minigameComparison = {
    played: mg.totalRounds > 0,
    totalRounds: mg.totalRounds,
    previousAccuracyPct: mg.previousSession ? mg.previousSession.accuracyPct : null,
    currentAccuracyPct: mg.latestSession ? mg.latestSession.accuracyPct : null,
    currentCorrect: mg.latestSession ? mg.latestSession.correct : null,
    currentTotal: mg.latestSession ? mg.latestSession.total : null,
    deltaAccuracyPct: mg.deltaAccuracyFromPrevious,
    verdict: mg.verdict,
  };

  // Compose personalized, warm, clinically accurate Thai recommendation text
  const improvedNames = exerciseBars
    .filter((b) => b.status === 'IMPROVED')
    .map((b) => `${b.shortLabelTh} (+${(b.deltaDeg ?? 0).toFixed(1)}°)`);
  const focusNames = exerciseBars
    .filter((b) => b.status === 'TRY_HARDER' || b.achievedPct < 85)
    .sort((a, b) => a.achievedPct - b.achievedPct)
    .map((b) => b.shortLabelTh);

  const parts: string[] = [];
  if (unifiedSummary.overall.verdict.code === 'IMPROVED') {
    if (improvedNames.length > 0) {
      parts.push(
        `ผู้ป่วยมีพัฒนาการที่ดีขึ้นอย่างต่อเนื่องในท่า${improvedNames.join(' และ ')} สามารถยกข้อต่อได้สูงขึ้นกว่าครั้งก่อนชัดเจน`,
      );
    } else {
      parts.push(unifiedSummary.overall.verdict.headlineTh);
    }
  } else if (unifiedSummary.overall.verdict.code === 'TRY_HARDER') {
    parts.push(
      `${ENCOURAGE_TRY_HARDER_PHRASE_TH} รอบนี้ผลการฝึกบางส่วนยังปรับลดลงหรือยังไม่เต็มเป้าหมาย`,
    );
  } else if (unifiedSummary.overall.verdict.code === 'STABLE') {
    parts.push(
      avgRomDeg !== null
        ? `ผู้ป่วยรักษาระดับการเคลื่อนไหวได้เท่าเดิมอย่างสม่ำเสมอ (องศาเฉลี่ย ${avgRomDeg.toFixed(1)}°)`
        : 'ผู้ป่วยรักษาระดับผลการฝึกได้เท่าเดิมอย่างสม่ำเสมอ',
    );
  } else if (unifiedSummary.overall.verdict.code === 'FIRST_SESSION') {
    parts.push(
      avgRomDeg !== null
        ? `เริ่มต้นฝึกได้ดีมากครับ บันทึกมุมองศาตั้งต้นเฉลี่ย ${avgRomDeg.toFixed(1)}° ไว้เปรียบเทียบในรอบถัดไปแล้ว`
        : 'บันทึกผลการฝึกรอบแรกเรียบร้อยแล้ว พร้อมเปรียบเทียบพัฒนาการในครั้งถัดไป',
    );
  } else {
    parts.push('ยังไม่มีบันทึกผลการฝึกกายภาพหรือมินิเกม กรุณาเริ่มฝึกตามคอร์สที่กำหนด');
  }

  if (minigameComparison.played && minigameComparison.currentAccuracyPct !== null) {
    if (
      minigameComparison.deltaAccuracyPct !== null &&
      minigameComparison.deltaAccuracyPct >= 5
    ) {
      parts.push(
        `ด้านมินิเกมฝึกสมองทำได้ ${minigameComparison.currentCorrect}/${minigameComparison.currentTotal} ข้อ (${minigameComparison.currentAccuracyPct}%) ดีขึ้น +${minigameComparison.deltaAccuracyPct}% เทียบกับรอบก่อน`,
      );
    } else if (
      minigameComparison.deltaAccuracyPct !== null &&
      minigameComparison.deltaAccuracyPct <= -5 &&
      minigameComparison.currentAccuracyPct < 90
    ) {
      parts.push(
        `ด้านมินิเกมฝึกสมองได้ ${minigameComparison.currentCorrect}/${minigameComparison.currentTotal} ข้อ (${minigameComparison.currentAccuracyPct}%) ลดลงเล็กน้อยจากรอบก่อน ลองฟังคำถามให้จบก่อนยกมือตอบนะครับ`,
      );
    } else {
      parts.push(
        `ด้านมินิเกมฝึกสมองได้ ${minigameComparison.currentCorrect}/${minigameComparison.currentTotal} ข้อ (${minigameComparison.currentAccuracyPct}%) รักษาระดับสมาธิได้ดี`,
      );
    }
  }

  if (focusNames.length > 0) {
    parts.push(`ควรเน้นฝึกท่า${focusNames[0]}เพิ่มเติมในรอบถัดไปเพื่อขยายมุมองศาให้สมบูรณ์ยิ่งขึ้นครับ`);
  } else if (exerciseBars.length > 0) {
    parts.push('สามารถฝึกต่อเนื่องตามตารางนัดเพื่อคงความแข็งแรงของข้อต่อและกล้ามเนื้อครับ');
  }

  const recommendationTh = parts.join(' ');
  const voiceSummaryTh = `สรุปผลการฝึกของคุณ${patientName} ความคืบหน้าภาพรวม ${overallProgressPct} เปอร์เซ็นต์ ${unifiedSummary.overall.verdict.headlineTh} ${recommendationTh}`;

  return {
    patientName,
    patientHn,
    periodTitleTh,
    periodRangeTh,
    overallProgressPct,
    weeklySessionCount,
    weeklyGoalCount,
    weeklyGoalMet,
    avgScore0To6,
    avgRomDeg,
    avgRomDeltaDeg,
    exerciseBars: exerciseBars.slice(0, 6),
    minigameComparison,
    overallVerdict: unifiedSummary.overall.verdict,
    recommendationTh,
    voiceSummaryTh,
  };
}

