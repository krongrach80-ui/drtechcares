import { CLINICAL_EXERCISES, parseExerciseCodeAndSide } from '@drtechcares/pose-engine';
import type {
  LongitudinalProgressSummary,
  PatientClinicalAssessmentRecord,
} from '../types/clinicalAssessment';
import type { RehabStockIndexSummary } from './rehabStockScoreEngine';
import type { UnifiedRehabMinigameSummary } from './unifiedRehabMinigameSummary';

type ExerciseBreakdownItem = LongitudinalProgressSummary['exerciseBreakdown'][number];

export type DoctorBriefTone = 'emerald' | 'sky' | 'amber' | 'slate';

export interface DoctorHighlightExercise {
  exerciseCode: string;
  nameTh: string;
  romDeg: number | null;
  targetRomDeg: number | null;
  achievementPct: number;
  score0To6: number | null;
  completedReps: number | null;
  targetReps: number | null;
  statusLabelTh: string;
}

export interface DoctorClinicalBrief {
  statusTone: DoctorBriefTone;
  statusBadgeTh: string;
  executiveSummaryTh: string;
  completedExercisesCount: number;
  totalPrescribedExercises: number;
  courseCompletionPct: number;
  pendingExerciseNames: string[];
  bestExercise: DoctorHighlightExercise | null;
  attentionExercise: {
    nameTh: string;
    reasonTh: string;
    pct: number;
  } | null;
  metrics: {
    rehabIndexPct: number | null;
    earnedPoints: number;
    maxPoints: number;
    avgRomTargetPct: number | null;
    latestRomDeg: number | null;
    romDeltaDeg: number | null;
    latestScore0To6: number | null;
    firstScore0To6: number | null;
    scoreDelta: number | null;
    avgAccuracyPct: number | null;
    latestPainVas: number | null;
    latestDayReps: number;
    minigameRounds: number;
    minigameAvgPct: number | null;
    minigameLatestPct: number | null;
  };
  doctorRecommendations: string[];
}

const FALLBACK_EXERCISE_TH: Record<string, string> = {
  ELDERLY_SHOULDER_FLEXION: 'ยกแขนไปด้านหน้า',
  ELDERLY_SHOULDER_ABDUCTION: 'กางแขนด้านข้าง',
  ELDERLY_ELBOW_FLEXION: 'งอ-เหยียดข้อศอก',
  ELDERLY_WRIST_FLEXION: 'กระดกข้อมือและนิ้วมือ',
  ELDERLY_SHOULDER_EXTERNAL_ROTATION: 'หมุนข้อไหล่เข้า-ออก',
  ELDERLY_OVERHEAD_REACH: 'ประสานมือชูขึ้นเหนือศีรษะ',
};

function cleanFriendlyExerciseName(rawName?: string | null, rawCode?: string | null): string {
  const cleanRaw = String(rawName ?? '').trim();
  const isCodeLike = !cleanRaw || /^[A-Z0-9_]+$/.test(cleanRaw) || cleanRaw.includes('_');
  let resolved = cleanRaw;
  if (isCodeLike) {
    const parsed = parseExerciseCodeAndSide(rawCode || cleanRaw);
    const upper = parsed.baseCode.toUpperCase();
    const found = CLINICAL_EXERCISES.find((ex) => ex.code.toUpperCase() === upper);
    if (found) {
      resolved = found.name.th;
    } else if (FALLBACK_EXERCISE_TH[upper]) {
      resolved = FALLBACK_EXERCISE_TH[upper];
    }
  }
  return resolved.replace(/^\[[^\]]+\]\s*/g, '').trim() || String(rawCode || 'ท่ากายภาพ');
}

function round1(value: number): number {
  return Math.round(value * 10) / 10;
}

export function buildDoctorClinicalBrief(input: {
  stockIndex: RehabStockIndexSummary;
  longitudinalSummary: LongitudinalProgressSummary;
  unifiedSummary: UnifiedRehabMinigameSummary;
  exerciseBreakdown: readonly ExerciseBreakdownItem[];
  assessments?: readonly PatientClinicalAssessmentRecord[];
}): DoctorClinicalBrief {
  const { stockIndex, longitudinalSummary, unifiedSummary, exerciseBreakdown, assessments = [] } =
    input;

  const latestDay = stockIndex.latestPoint;
  const totalPrescribed = Math.max(
    stockIndex.totalPrescribedExercises,
    latestDay?.totalExercisesCount ?? 0,
    exerciseBreakdown.length,
  );
  const completedCount =
    latestDay?.completedExercisesCount ?? exerciseBreakdown.filter((e) => e.visitCount > 0).length;
  const courseCompletionPct =
    totalPrescribed > 0 ? Math.min(100, Math.round((completedCount / totalPrescribed) * 100)) : 0;

  const dayBreakdown = latestDay?.exerciseBreakdown ?? [];
  const pendingExerciseNames = dayBreakdown
    .filter((ex) => !ex.played)
    .map((ex) => cleanFriendlyExerciseName(ex.exerciseNameTh, ex.exerciseCode));

  // Resolve average ROM achievement % across played exercises or assessments
  const romPctsFromDay = dayBreakdown
    .filter((ex) => ex.played && ex.achievedRomDeg !== null && ex.targetRomDeg > 0)
    .map((ex) => (ex.achievedRomDeg! / ex.targetRomDeg) * 100);
  const romPctsFromBreakdown = exerciseBreakdown
    .filter((b) => Number.isFinite(b.latestTargetAchievementPct) && b.latestTargetAchievementPct > 0)
    .map((b) => b.latestTargetAchievementPct);
  const effectiveRomPcts = romPctsFromDay.length > 0 ? romPctsFromDay : romPctsFromBreakdown;
  const avgRomTargetPct =
    effectiveRomPcts.length > 0
      ? round1(effectiveRomPcts.reduce((acc, v) => acc + v, 0) / effectiveRomPcts.length)
      : longitudinalSummary.latestTargetAchievementPct;

  const latestPainRec = [...assessments]
    .reverse()
    .find((r) => typeof r.pain_vas_0_to_10 === 'number' && Number.isFinite(r.pain_vas_0_to_10));
  const latestPainVas =
    typeof latestPainRec?.pain_vas_0_to_10 === 'number' ? latestPainRec.pain_vas_0_to_10 : null;

  // Identify strongest exercise
  let bestExercise: DoctorHighlightExercise | null = null;
  if (dayBreakdown.some((ex) => ex.played)) {
    const playedSorted = [...dayBreakdown]
      .filter((ex) => ex.played)
      .sort((a, b) => {
        const aRomPct =
          a.achievedRomDeg !== null && a.targetRomDeg > 0
            ? (a.achievedRomDeg / a.targetRomDeg) * 100
            : a.smartPct;
        const bRomPct =
          b.achievedRomDeg !== null && b.targetRomDeg > 0
            ? (b.achievedRomDeg / b.targetRomDeg) * 100
            : b.smartPct;
        return bRomPct - aRomPct;
      });
    const top = playedSorted[0];
    if (top) {
      const topPct =
        top.achievedRomDeg !== null && top.targetRomDeg > 0
          ? round1((top.achievedRomDeg / top.targetRomDeg) * 100)
          : round1(top.smartPct);
      bestExercise = {
        exerciseCode: top.exerciseCode,
        nameTh: cleanFriendlyExerciseName(top.exerciseNameTh, top.exerciseCode),
        romDeg: top.achievedRomDeg !== null ? round1(top.achievedRomDeg) : null,
        targetRomDeg: top.targetRomDeg > 0 ? Math.round(top.targetRomDeg) : null,
        achievementPct: topPct,
        score0To6: longitudinalSummary.latestScore,
        completedReps: top.completedReps,
        targetReps: top.targetReps,
        statusLabelTh: topPct >= 100 ? 'เกินเป้าหมายคลินิก' : topPct >= 80 ? 'ผ่านเกณฑ์ดีเยี่ยม' : 'กำลังพัฒนา',
      };
    }
  } else if (exerciseBreakdown.length > 0) {
    const top = [...exerciseBreakdown].sort(
      (a, b) => b.latestTargetAchievementPct - a.latestTargetAchievementPct,
    )[0];
    if (top) {
      bestExercise = {
        exerciseCode: top.exerciseCode,
        nameTh: cleanFriendlyExerciseName(top.exerciseNameTh, top.exerciseCode),
        romDeg: round1(top.latestRomDeg),
        targetRomDeg: Math.round(top.targetRomDeg),
        achievementPct: round1(top.latestTargetAchievementPct),
        score0To6: top.latestScore,
        completedReps: null,
        targetReps: null,
        statusLabelTh:
          top.latestTargetAchievementPct >= 100 ? 'เกินเป้าหมายคลินิก' : 'ผ่านเกณฑ์ดี',
      };
    }
  }

  // Identify attention / pending exercise for the doctor
  let attentionExercise: DoctorClinicalBrief['attentionExercise'] = null;
  if (pendingExerciseNames.length > 0) {
    attentionExercise = {
      nameTh: pendingExerciseNames[0],
      reasonTh:
        pendingExerciseNames.length === 1
          ? 'ยังไม่ได้ฝึกในรอบล่าสุด (0%)'
          : `ยังรอฝึกอีก ${pendingExerciseNames.length} ท่าตามคอร์ส`,
      pct: 0,
    };
  } else if (dayBreakdown.length > 0) {
    const lowest = [...dayBreakdown].sort((a, b) => a.smartPct - b.smartPct)[0];
    if (lowest && lowest.smartPct < 85) {
      attentionExercise = {
        nameTh: cleanFriendlyExerciseName(lowest.exerciseNameTh, lowest.exerciseCode),
        reasonTh: `ทำได้ ${lowest.smartPct.toFixed(0)}% • ROM ${
          lowest.achievedRomDeg !== null ? `${lowest.achievedRomDeg.toFixed(0)}°` : '-'
        }/${Math.round(lowest.targetRomDeg)}°`,
        pct: round1(lowest.smartPct),
      };
    }
  } else if (exerciseBreakdown.length > 0) {
    const lowest = [...exerciseBreakdown].sort(
      (a, b) => a.latestTargetAchievementPct - b.latestTargetAchievementPct,
    )[0];
    if (lowest && lowest.latestTargetAchievementPct < 85) {
      attentionExercise = {
        nameTh: cleanFriendlyExerciseName(lowest.exerciseNameTh, lowest.exerciseCode),
        reasonTh: `ROM ล่าสุด ${lowest.latestRomDeg.toFixed(1)}° (${lowest.latestTargetAchievementPct.toFixed(0)}% เป้า)`,
        pct: round1(lowest.latestTargetAchievementPct),
      };
    }
  }

  const rehabIndexPct = latestDay ? round1(latestDay.smartTotalPct) : null;
  const avgAccuracyPct = latestDay ? Math.round(latestDay.avgAccuracyPct) : null;

  // Clinical status tone & executive one-liner
  let statusTone: DoctorBriefTone = 'slate';
  let statusBadgeTh = 'รอข้อมูลประเมิน';
  let executiveSummaryTh = 'ยังไม่มีข้อมูลการฝึกหรือประเมินในระบบ — เริ่มฝึกที่ตู้ Kiosk หรือบันทึกผล 0-6 เพื่อวิเคราะห์';

  if (rehabIndexPct !== null || longitudinalSummary.totalVisits > 0) {
    const effectivePct = rehabIndexPct ?? avgRomTargetPct ?? 0;
    const score = longitudinalSummary.latestScore;
    if (effectivePct >= 80 && pendingExerciseNames.length === 0) {
      statusTone = 'emerald';
      statusBadgeTh = 'ฟื้นตัวดีเยี่ยม (ผ่านเกณฑ์)';
      executiveSummaryTh = `ฝึกครบ ${completedCount}/${totalPrescribed} ท่า (${effectivePct.toFixed(1)}%) • ROM เฉลี่ย ${
        avgRomTargetPct !== null ? `${avgRomTargetPct.toFixed(0)}%` : '-'
      } ของเป้า${score !== null ? ` • กำลังกล้ามเนื้อ ${score}/6` : ''}`;
    } else if (effectivePct >= 60 || (avgRomTargetPct !== null && avgRomTargetPct >= 90)) {
      statusTone = 'sky';
      statusBadgeTh =
        pendingExerciseNames.length > 0
          ? `คุณภาพดีเยี่ยม • เหลือฝึก ${pendingExerciseNames.length} ท่า`
          : 'กำลังฟื้นตัวดี';
      executiveSummaryTh =
        pendingExerciseNames.length > 0
          ? `ท่าที่ฝึกแล้ว (${completedCount}/${totalPrescribed} ท่า) ทำ ROM เฉลี่ย ${
              avgRomTargetPct !== null ? `${avgRomTargetPct.toFixed(0)}%` : '-'
            } แม่นยำ ${avgAccuracyPct ?? '-'}% • เหลืออีก ${pendingExerciseNames.length} ท่าจึงจะครบคอร์ส`
          : `ดัชนีฟื้นฟูรวม ${effectivePct.toFixed(1)}% • ROM เฉลี่ย ${
              avgRomTargetPct !== null ? `${avgRomTargetPct.toFixed(0)}%` : '-'
            } ของเป้าหมายคลินิก`;
    } else {
      statusTone = 'amber';
      statusBadgeTh = 'ควรฝึกเพิ่ม / ติดตามใกล้ชิด';
      executiveSummaryTh = `ดัชนีฟื้นฟูรวม ${effectivePct.toFixed(1)}% (${completedCount}/${totalPrescribed} ท่า) • ควรกระตุ้นให้ฝึกต่อเนื่องและติดตามอาการระหว่างทำท่า`;
    }
  }

  // Concrete, actionable recommendations for the doctor / PT
  const doctorRecommendations: string[] = [];

  if (pendingExerciseNames.length > 0) {
    const previewNames = pendingExerciseNames.slice(0, 2).join(', ');
    const moreSuffix =
      pendingExerciseNames.length > 2 ? ` และอีก ${pendingExerciseNames.length - 2} ท่า` : '';
    doctorRecommendations.push(
      `เหลือรอฝึก ${pendingExerciseNames.length} ท่า (${previewNames}${moreSuffix}) — หากฝึกครบ ดัชนีรวมจะขึ้นสู่เกณฑ์ดีเยี่ยม (80%+)`,
    );
  }

  if (avgRomTargetPct !== null && avgRomTargetPct >= 110 && (longitudinalSummary.latestScore ?? 0) >= 5) {
    doctorRecommendations.push(
      `องศา ROM เฉลี่ย ${avgRomTargetPct.toFixed(0)}% สูงกว่าเป้าหมายคลินิก — แพทย์สามารถพิจารณาขยับเป้าองศาหรือเพิ่มจำนวนครั้งในรอบถัดไป`,
    );
  } else if (avgRomTargetPct !== null && avgRomTargetPct < 75) {
    doctorRecommendations.push(
      `องศา ROM เฉลี่ย ${avgRomTargetPct.toFixed(0)}% ยังต่ำกว่าเป้าหมาย — แนะนำเน้นยืดเหยียดช้าๆ และตรวจประเมินอาการติดขัดของข้อต่อ`,
    );
  }

  if (latestPainVas !== null && latestPainVas >= 5) {
    doctorRecommendations.push(
      `พบระดับปวด VAS ${latestPainVas}/10 — ควรลดช่วงองศาเป้าหมายชั่วคราวและตรวจประเมินอาการปวดก่อนเพิ่มความหนัก`,
    );
  } else if (latestPainVas !== null && latestPainVas <= 2 && longitudinalSummary.totalVisits > 0) {
    doctorRecommendations.push(
      `ระดับปวดต่ำ (VAS ${latestPainVas}/10) ผู้สูงอายุทนต่อการฝึกได้ดี ปลอดภัยสำหรับการฝึกต่อเนื่อง`,
    );
  }

  if (unifiedSummary.minigame.totalRounds === 0) {
    doctorRecommendations.push(
      'ยังไม่มีผลมินิเกมฝึกสมอง — แนะนำให้ลองเล่นมินิเกมการ์ดกายภาพควบคู่เพื่อติดตามความจำและสมาธิ',
    );
  } else if (unifiedSummary.minigame.verdict.code === 'TRY_HARDER') {
    doctorRecommendations.push(
      `ความแม่นยำมินิเกมล่าสุด ${unifiedSummary.minigame.latestAccuracyPct ?? 0}% ลดลงจากรอบก่อน — แนะนำพักให้พร้อมก่อนเล่นรอบถัดไป`,
    );
  } else if (unifiedSummary.minigame.latestAccuracyPct !== null) {
    doctorRecommendations.push(
      `มินิเกมฝึกสมองทำได้ดี (${unifiedSummary.minigame.latestAccuracyPct}% เฉลี่ย ${unifiedSummary.minigame.averageAccuracyPct?.toFixed(0) ?? '-'}%) ช่วยกระตุ้นการทำงานประสานสมอง-กล้ามเนื้อได้ดี`,
    );
  }

  if (doctorRecommendations.length === 0) {
    doctorRecommendations.push(
      'ติดตามการฝึกอย่างสม่ำเสมอ 3-5 วัน/สัปดาห์ และบันทึกประเมิน 0-6 เปรียบเทียบทุกสัปดาห์',
    );
  }

  return {
    statusTone,
    statusBadgeTh,
    executiveSummaryTh,
    completedExercisesCount: completedCount,
    totalPrescribedExercises: totalPrescribed,
    courseCompletionPct,
    pendingExerciseNames,
    bestExercise,
    attentionExercise,
    metrics: {
      rehabIndexPct,
      earnedPoints: latestDay ? round1(latestDay.smartEarnedScore) : 0,
      maxPoints: stockIndex.maxTotalScore,
      avgRomTargetPct,
      latestRomDeg: longitudinalSummary.latestRom,
      romDeltaDeg: longitudinalSummary.deltaRom,
      latestScore0To6: longitudinalSummary.latestScore,
      firstScore0To6: longitudinalSummary.firstScore,
      scoreDelta: longitudinalSummary.deltaScore,
      avgAccuracyPct,
      latestPainVas,
      latestDayReps: latestDay?.volumeReps ?? 0,
      minigameRounds: unifiedSummary.minigame.totalRounds,
      minigameAvgPct: unifiedSummary.minigame.averageAccuracyPct,
      minigameLatestPct: unifiedSummary.minigame.latestAccuracyPct,
    },
    doctorRecommendations: doctorRecommendations.slice(0, 3),
  };
}
