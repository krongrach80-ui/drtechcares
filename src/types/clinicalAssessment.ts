export type MahidolScore0To6 = 0 | 1 | 2 | 3 | 4 | 5 | 6;

export type AssessmentTargetSide = 'LEFT' | 'RIGHT' | 'BOTH';

export type AssessmentSourceChannel =
  | 'STAFF_MANUAL_ASSESSMENT'
  | 'KIOSK_SESSION_AUTO_SYNC'
  | 'COURSE_AUTO_CALCULATED';

export type AssessmentMeasurementMethod =
  | 'MANUAL_INPUT'
  | 'CAMERA_AI_LIVE'
  | 'KIOSK_SESSION_AUTO_SYNC'
  | 'COURSE_AUTO_CALCULATED';

export interface MahidolScaleRubricItem {
  score: MahidolScore0To6;
  shortLabelTh: string;
  descriptionTh: string;
}

export const MAHIDOL_ASSESSMENT_SCALE_0_TO_6: readonly MahidolScaleRubricItem[] = [
  {
    score: 0,
    shortLabelTh: 'ระดับ 0 (ไม่มีการหดตัว)',
    descriptionTh: 'ระดับ 0 — ไม่มีการหดตัวของกล้ามเนื้อ ไม่สามารถเริ่มขยับข้อต่อได้',
  },
  {
    score: 1,
    shortLabelTh: 'ระดับ 1 (เกร็งตัวเล็กน้อย)',
    descriptionTh: 'ระดับ 1 — มีการเกร็งตัวของกล้ามเนื้อเล็กน้อย แต่ยังไม่เกิดการเคลื่อนไหวของข้อต่อ',
  },
  {
    score: 2,
    shortLabelTh: 'ระดับ 2 (ขยับแนวราบ)',
    descriptionTh: 'ระดับ 2 — เคลื่อนไหวข้อต่อได้บางส่วนในแนวราบ หรือต้องช่วยพยุงต้านแรงโน้มถ่วง',
  },
  {
    score: 3,
    shortLabelTh: 'ระดับ 3 (ต้านแรงโน้มถ่วงได้)',
    descriptionTh: 'ระดับ 3 — เคลื่อนไหวข้อต่อต้านแรงโน้มถ่วงได้เต็มช่วง แต่ยังไม่สามารถต้านแรงต้านได้',
  },
  {
    score: 4,
    shortLabelTh: 'ระดับ 4 (ต้านแรงปานกลางได้)',
    descriptionTh: 'ระดับ 4 — เคลื่อนไหวต้านแรงต้านระดับปานกลางได้ แต่อาจมีความล้าหรือเกร็งชดเชยเล็กน้อย',
  },
  {
    score: 5,
    shortLabelTh: 'ระดับ 5 (คุมจังหวะได้ดี)',
    descriptionTh: 'ระดับ 5 — เคลื่อนไหวต้านแรงต้านได้ดีใกล้เคียงปกติ ควบคุมจังหวะการเคลื่อนไหวได้มั่นคง',
  },
  {
    score: 6,
    shortLabelTh: 'ระดับ 6 (สมบูรณ์เทียบเท่าปกติ)',
    descriptionTh: 'ระดับ 6 — เคลื่อนไหวได้สมบูรณ์เต็มช่วงองศา ควบคุมความเร็วและความแม่นยำได้เทียบเท่าปกติ (เกณฑ์มหิดลระดับสูงสุด)',
  },
];

export interface PatientClinicalAssessmentRecord {
  id: string;
  patient_id: string;
  patient_hn: string;
  universal_link_code: string;
  visit_number: number;
  assessed_at: string;
  facility_name: string;
  assessor_staff_id: string;
  assessor_staff_name: string;
  exercise_code: string;
  exercise_name_th: string;
  target_side: AssessmentTargetSide;
  initial_score_0_to_6: MahidolScore0To6;
  measured_rom_deg: number;
  target_rom_deg: number;
  pain_vas_0_to_10?: number | null;
  chief_complaint: string;
  clinical_handover_note: string;
  next_session_goal: string;
  source_channel: AssessmentSourceChannel;
  measurement_method?: AssessmentMeasurementMethod;
  completed_reps?: number | null;
  accuracy_percent?: number | null;
  session_id?: string | null;
  source_session_id?: string | null;
  created_at?: string;
}

export interface CreatePatientClinicalAssessmentInput {
  patient_id: string;
  patient_hn: string;
  universal_link_code: string;
  assessed_at?: string;
  facility_name: string;
  assessor_staff_id: string;
  assessor_staff_name: string;
  exercise_code: string;
  exercise_name_th: string;
  target_side: AssessmentTargetSide;
  initial_score_0_to_6: number;
  measured_rom_deg: number;
  target_rom_deg: number;
  pain_vas_0_to_10?: number | null;
  chief_complaint: string;
  clinical_handover_note: string;
  next_session_goal: string;
  source_channel?: AssessmentSourceChannel;
  measurement_method?: AssessmentMeasurementMethod;
  completed_reps?: number | null;
  accuracy_percent?: number | null;
  session_id?: string | null;
  source_session_id?: string | null;
}

export interface LongitudinalVisitDelta {
  record: PatientClinicalAssessmentRecord;
  deltaFromPrevRom: number | null;
  deltaFromFirstRom: number;
  deltaFromPrevScore: number | null;
  targetAchievementPct: number;
  deltaFromPrevTargetPct: number | null;
}

export interface ExerciseComparisonBreakdownItem {
  key: string;
  exerciseCode: string;
  exerciseNameTh: string;
  targetSide: AssessmentTargetSide;
  visitCount: number;
  firstVisit: PatientClinicalAssessmentRecord;
  latestVisit: PatientClinicalAssessmentRecord;
  firstRomDeg: number;
  latestRomDeg: number;
  deltaRomDeg: number;
  percentChange: number | null;
  targetRomDeg: number;
  firstTargetAchievementPct: number;
  latestTargetAchievementPct: number;
  targetAchievementDeltaPct: number;
  remainingToTargetDeg: number;
  firstScore: MahidolScore0To6;
  latestScore: MahidolScore0To6;
  deltaScore: number;
  firstPainVas: number | null;
  latestPainVas: number | null;
  deltaPainVas: number | null;
  elapsedDays: number;
  romGainPerWeek: number | null;
  trendStatus: 'IMPROVED' | 'STABLE' | 'REGRESSED' | 'BASELINE_ONLY';
}

export interface VisitPairComparison {
  baseline: PatientClinicalAssessmentRecord;
  followUp: PatientClinicalAssessmentRecord;
  elapsedDays: number;
  isSameExercise: boolean;
  romDeltaDeg: number;
  romPercentChange: number | null;
  romGainPerWeek: number | null;
  baselineTargetPct: number;
  followUpTargetPct: number;
  targetPctDelta: number;
  scoreDelta: number;
  painDelta: number | null;
  clinicalVerdictTh: string;
  verdictTone: 'emerald' | 'sky' | 'amber' | 'rose';
}

export interface LongitudinalProgressSummary {
  totalVisits: number;
  firstRom: number | null;
  latestRom: number | null;
  deltaRom: number | null;
  percentGain: number | null;
  firstScore: MahidolScore0To6 | null;
  latestScore: MahidolScore0To6 | null;
  deltaScore: number | null;
  linkedFacilities: string[];
  visitsWithDelta: LongitudinalVisitDelta[];
  targetRomDeg: number | null;
  firstTargetAchievementPct: number | null;
  latestTargetAchievementPct: number | null;
  deltaTargetAchievementPct: number | null;
  remainingToTargetDeg: number | null;
  firstPainVas: number | null;
  latestPainVas: number | null;
  deltaPainVas: number | null;
  elapsedDays: number;
  romGainPerWeek: number | null;
  hasMixedExercises: boolean;
  staffVisitCount: number;
  kioskVisitCount: number;
  courseAutoVisitCount: number;
  exerciseBreakdown: ExerciseComparisonBreakdownItem[];
}

export interface PatientIdentityRef {
  id: string;
  hn?: string;
  universalHealthLinkCode?: string | null;
}

export function isValidMahidolScore(value: number): value is MahidolScore0To6 {
  return Number.isInteger(value) && value >= 0 && value <= 6;
}

export function clampMahidolScore(value: number): MahidolScore0To6 {
  if (!Number.isFinite(value)) return 0;
  const rounded = Math.round(value);
  return Math.max(0, Math.min(6, rounded)) as MahidolScore0To6;
}

export function getMahidolRubricItem(score: number): MahidolScaleRubricItem {
  const safeScore = clampMahidolScore(score);
  return MAHIDOL_ASSESSMENT_SCALE_0_TO_6[safeScore];
}

export function isDemoSeedAssessment(rec: Pick<PatientClinicalAssessmentRecord, 'assessor_staff_id'>): boolean {
  return rec.assessor_staff_id === 'pt-demo-01' || rec.assessor_staff_id === 'pt-demo-02';
}

export function computeTargetAchievementPct(measuredRomDeg: number, targetRomDeg: number): number {
  if (!Number.isFinite(measuredRomDeg) || !Number.isFinite(targetRomDeg) || targetRomDeg <= 0) {
    return 0;
  }
  return Number(Math.max(0, (measuredRomDeg / targetRomDeg) * 100).toFixed(1));
}

export function resolvePatientUniversalLinkCode(patient: PatientIdentityRef): string {
  const existing = (patient.universalHealthLinkCode ?? '').trim().toUpperCase();
  if (existing) return existing;
  const cleanId = (patient.id || patient.hn || 'PATIENT')
    .replace(/[^a-zA-Z0-9]/g, '')
    .toUpperCase()
    .slice(0, 8)
    .padEnd(4, '0');
  return `DTC-EHR-${cleanId}`;
}

export function filterAssessmentsForPatient(
  records: readonly PatientClinicalAssessmentRecord[],
  patient: PatientIdentityRef,
): PatientClinicalAssessmentRecord[] {
  const pid = patient.id.trim();
  const explicitLink = (patient.universalHealthLinkCode ?? '').trim().toUpperCase();
  const derivedLink = resolvePatientUniversalLinkCode(patient).toUpperCase();

  return records
    .filter((rec) => {
      if (rec.patient_id === pid) return true;
      const recLink = (rec.universal_link_code ?? '').trim().toUpperCase();
      if (!recLink) return false;
      if (explicitLink && recLink === explicitLink) return true;
      if (!explicitLink && recLink === derivedLink) return true;
      return false;
    })
    .slice()
    .sort((a, b) => {
      const ta = new Date(a.assessed_at).getTime();
      const tb = new Date(b.assessed_at).getTime();
      const validTa = Number.isFinite(ta) ? ta : 0;
      const validTb = Number.isFinite(tb) ? tb : 0;
      if (validTa !== validTb) return validTa - validTb;
      return a.visit_number - b.visit_number;
    });
}

function sortChronologically(
  records: readonly PatientClinicalAssessmentRecord[],
): PatientClinicalAssessmentRecord[] {
  return records.slice().sort((a, b) => {
    const ta = new Date(a.assessed_at).getTime();
    const tb = new Date(b.assessed_at).getTime();
    const validTa = Number.isFinite(ta) ? ta : 0;
    const validTb = Number.isFinite(tb) ? tb : 0;
    if (validTa !== validTb) return validTa - validTb;
    return a.visit_number - b.visit_number;
  });
}

function computeElapsedDays(startIso: string, endIso: string): number {
  const t1 = new Date(startIso).getTime();
  const t2 = new Date(endIso).getTime();
  if (!Number.isFinite(t1) || !Number.isFinite(t2)) return 0;
  return Number((Math.max(0, t2 - t1) / 86_400_000).toFixed(1));
}

export function computeExerciseBreakdown(
  records: readonly PatientClinicalAssessmentRecord[],
): ExerciseComparisonBreakdownItem[] {
  const sorted = sortChronologically(records);
  const groups = new Map<string, PatientClinicalAssessmentRecord[]>();
  for (const rec of sorted) {
    const code = (rec.exercise_code || 'UNKNOWN').trim().toUpperCase();
    const side: AssessmentTargetSide =
      rec.target_side === 'LEFT' ? 'LEFT' : rec.target_side === 'BOTH' ? 'BOTH' : 'RIGHT';
    const key = `${code}::${side}`;
    const list = groups.get(key) ?? [];
    list.push(rec);
    groups.set(key, list);
  }

  const out: ExerciseComparisonBreakdownItem[] = [];
  for (const [key, list] of groups.entries()) {
    const firstVisit = list[0];
    const latestVisit = list[list.length - 1];
    const firstRomDeg = Number(firstVisit.measured_rom_deg.toFixed(1));
    const latestRomDeg = Number(latestVisit.measured_rom_deg.toFixed(1));
    const deltaRomDeg = Number((latestRomDeg - firstRomDeg).toFixed(1));
    const percentChange =
      firstRomDeg > 0 ? Number((((latestRomDeg - firstRomDeg) / firstRomDeg) * 100).toFixed(1)) : null;
    const targetRomDeg = Number((latestVisit.target_rom_deg || firstVisit.target_rom_deg || 120).toFixed(1));
    const firstTargetAchievementPct = computeTargetAchievementPct(
      firstRomDeg,
      firstVisit.target_rom_deg || targetRomDeg,
    );
    const latestTargetAchievementPct = computeTargetAchievementPct(latestRomDeg, targetRomDeg);
    const targetAchievementDeltaPct = Number(
      (latestTargetAchievementPct - firstTargetAchievementPct).toFixed(1),
    );
    const remainingToTargetDeg = Number(Math.max(0, targetRomDeg - latestRomDeg).toFixed(1));
    const firstScore = firstVisit.initial_score_0_to_6;
    const latestScore = latestVisit.initial_score_0_to_6;
    const deltaScore = latestScore - firstScore;

    const painValues = list.filter(
      (r) => typeof r.pain_vas_0_to_10 === 'number' && Number.isFinite(r.pain_vas_0_to_10),
    );
    const firstPainVas = painValues.length > 0 ? (painValues[0].pain_vas_0_to_10 as number) : null;
    const latestPainVas =
      painValues.length > 0 ? (painValues[painValues.length - 1].pain_vas_0_to_10 as number) : null;
    const deltaPainVas =
      firstPainVas !== null && latestPainVas !== null ? latestPainVas - firstPainVas : null;

    const elapsedDays = computeElapsedDays(firstVisit.assessed_at, latestVisit.assessed_at);
    const romGainPerWeek =
      list.length >= 2 && elapsedDays >= 1
        ? Number((deltaRomDeg / (elapsedDays / 7)).toFixed(1))
        : null;

    let trendStatus: ExerciseComparisonBreakdownItem['trendStatus'] = 'BASELINE_ONLY';
    if (list.length >= 2) {
      if (deltaRomDeg <= -4 || deltaScore <= -1 || (deltaPainVas !== null && deltaPainVas >= 2)) {
        trendStatus = 'REGRESSED';
      } else if (deltaRomDeg >= 3 || deltaScore >= 1 || (deltaPainVas !== null && deltaPainVas <= -2)) {
        trendStatus = 'IMPROVED';
      } else {
        trendStatus = 'STABLE';
      }
    }

    out.push({
      key,
      exerciseCode: (latestVisit.exercise_code || 'UNKNOWN').trim().toUpperCase(),
      exerciseNameTh: latestVisit.exercise_name_th || latestVisit.exercise_code,
      targetSide: latestVisit.target_side,
      visitCount: list.length,
      firstVisit,
      latestVisit,
      firstRomDeg,
      latestRomDeg,
      deltaRomDeg,
      percentChange,
      targetRomDeg,
      firstTargetAchievementPct,
      latestTargetAchievementPct,
      targetAchievementDeltaPct,
      remainingToTargetDeg,
      firstScore,
      latestScore,
      deltaScore,
      firstPainVas,
      latestPainVas,
      deltaPainVas,
      elapsedDays,
      romGainPerWeek,
      trendStatus,
    });
  }

  return out.sort(
    (a, b) =>
      new Date(b.latestVisit.assessed_at).getTime() - new Date(a.latestVisit.assessed_at).getTime(),
  );
}

export function compareTwoAssessmentVisits(
  baseline: PatientClinicalAssessmentRecord,
  followUp: PatientClinicalAssessmentRecord,
): VisitPairComparison {
  const isSameExercise =
    (baseline.exercise_code || '').trim().toUpperCase() ===
    (followUp.exercise_code || '').trim().toUpperCase();
  const elapsedDays = computeElapsedDays(baseline.assessed_at, followUp.assessed_at);
  const romDeltaDeg = Number((followUp.measured_rom_deg - baseline.measured_rom_deg).toFixed(1));
  const romPercentChange =
    baseline.measured_rom_deg > 0
      ? Number(
          (((followUp.measured_rom_deg - baseline.measured_rom_deg) / baseline.measured_rom_deg) * 100).toFixed(1),
        )
      : null;
  const romGainPerWeek =
    elapsedDays >= 1 ? Number((romDeltaDeg / (elapsedDays / 7)).toFixed(1)) : null;

  const baselineTargetPct = computeTargetAchievementPct(
    baseline.measured_rom_deg,
    baseline.target_rom_deg || 120,
  );
  const followUpTargetPct = computeTargetAchievementPct(
    followUp.measured_rom_deg,
    followUp.target_rom_deg || 120,
  );
  const targetPctDelta = Number((followUpTargetPct - baselineTargetPct).toFixed(1));
  const scoreDelta = followUp.initial_score_0_to_6 - baseline.initial_score_0_to_6;
  const painDelta =
    typeof baseline.pain_vas_0_to_10 === 'number' &&
    typeof followUp.pain_vas_0_to_10 === 'number'
      ? followUp.pain_vas_0_to_10 - baseline.pain_vas_0_to_10
      : null;

  const primaryDelta = isSameExercise ? romDeltaDeg : targetPctDelta;
  let clinicalVerdictTh = 'ผลทรงตัว แนะนำประเมินซ้ำและปรับแรงต้าน/ช่วงค้างท่าตามความเหมาะสม';
  let verdictTone: VisitPairComparison['verdictTone'] = 'amber';

  if (primaryDelta <= -5 || scoreDelta <= -1 || (painDelta !== null && painDelta >= 2)) {
    clinicalVerdictTh =
      'ควรเฝ้าระวัง: ช่วงการเคลื่อนไหวหรือคะแนนลดลง หรือระดับความปวดเพิ่มขึ้น ควรทบทวนแผนกายภาพ';
    verdictTone = 'rose';
  } else if (primaryDelta >= 5 || scoreDelta >= 1) {
    clinicalVerdictTh = isSameExercise
      ? `พัฒนาการดีขึ้นชัดเจน (+${romDeltaDeg.toFixed(1)}° / เทียบเป้าหมาย ${targetPctDelta >= 0 ? `+${targetPctDelta.toFixed(1)}%` : `${targetPctDelta.toFixed(1)}%`})`
      : `ความสำเร็จเทียบเป้าหมายคลินิกดีขึ้นชัดเจน (${targetPctDelta >= 0 ? `+${targetPctDelta.toFixed(1)}%` : `${targetPctDelta.toFixed(1)}%`})`;
    verdictTone = 'emerald';
  } else if (primaryDelta > 0 || (painDelta !== null && painDelta <= -1)) {
    clinicalVerdictTh = 'แนวโน้มฟื้นตัวดีขึ้นต่อเนื่อง คงความสม่ำเสมอการฝึกและติดตามผลต่อ';
    verdictTone = 'sky';
  }

  return {
    baseline,
    followUp,
    elapsedDays,
    isSameExercise,
    romDeltaDeg,
    romPercentChange,
    romGainPerWeek,
    baselineTargetPct,
    followUpTargetPct,
    targetPctDelta,
    scoreDelta,
    painDelta,
    clinicalVerdictTh,
    verdictTone,
  };
}

export function computeLongitudinalProgressSummary(
  records: readonly PatientClinicalAssessmentRecord[],
  exerciseCodeFilter?: string | null,
): LongitudinalProgressSummary {
  const cleanFilter = (exerciseCodeFilter ?? '').trim().toUpperCase();
  const filtered =
    cleanFilter && cleanFilter !== 'ALL'
      ? records.filter((r) => (r.exercise_code ?? '').trim().toUpperCase() === cleanFilter)
      : records.slice();

  const sorted = sortChronologically(filtered);
  const exerciseBreakdown = computeExerciseBreakdown(sorted);

  if (sorted.length === 0) {
    return {
      totalVisits: 0,
      firstRom: null,
      latestRom: null,
      deltaRom: null,
      percentGain: null,
      firstScore: null,
      latestScore: null,
      deltaScore: null,
      linkedFacilities: [],
      visitsWithDelta: [],
      targetRomDeg: null,
      firstTargetAchievementPct: null,
      latestTargetAchievementPct: null,
      deltaTargetAchievementPct: null,
      remainingToTargetDeg: null,
      firstPainVas: null,
      latestPainVas: null,
      deltaPainVas: null,
      elapsedDays: 0,
      romGainPerWeek: null,
      hasMixedExercises: false,
      staffVisitCount: 0,
      kioskVisitCount: 0,
      courseAutoVisitCount: 0,
      exerciseBreakdown: [],
    };
  }

  const first = sorted[0];
  const latest = sorted[sorted.length - 1];
  const firstRom = Number(first.measured_rom_deg.toFixed(1));
  const latestRom = Number(latest.measured_rom_deg.toFixed(1));
  const deltaRom = Number((latestRom - firstRom).toFixed(1));
  const percentGain =
    firstRom > 0 ? Number((((latestRom - firstRom) / firstRom) * 100).toFixed(1)) : null;

  const firstScore = first.initial_score_0_to_6;
  const latestScore = latest.initial_score_0_to_6;
  const deltaScore = latestScore - firstScore;

  const uniqueExerciseCodes = new Set(
    sorted.map((r) => (r.exercise_code ?? '').trim().toUpperCase()).filter(Boolean),
  );
  const hasMixedExercises = uniqueExerciseCodes.size > 1;
  const targetRomDeg = !hasMixedExercises
    ? Number((latest.target_rom_deg || first.target_rom_deg || 120).toFixed(1))
    : null;

  const firstTargetAchievementPct = computeTargetAchievementPct(
    firstRom,
    first.target_rom_deg || 120,
  );
  const latestTargetAchievementPct = computeTargetAchievementPct(
    latestRom,
    latest.target_rom_deg || 120,
  );
  const deltaTargetAchievementPct = Number(
    (latestTargetAchievementPct - firstTargetAchievementPct).toFixed(1),
  );
  const remainingToTargetDeg =
    targetRomDeg !== null ? Number(Math.max(0, targetRomDeg - latestRom).toFixed(1)) : null;

  const painRecords = sorted.filter(
    (r) => typeof r.pain_vas_0_to_10 === 'number' && Number.isFinite(r.pain_vas_0_to_10),
  );
  const firstPainVas =
    painRecords.length > 0 ? (painRecords[0].pain_vas_0_to_10 as number) : null;
  const latestPainVas =
    painRecords.length > 0 ? (painRecords[painRecords.length - 1].pain_vas_0_to_10 as number) : null;
  const deltaPainVas =
    firstPainVas !== null && latestPainVas !== null ? latestPainVas - firstPainVas : null;

  const elapsedDays = computeElapsedDays(first.assessed_at, latest.assessed_at);
  const romGainPerWeek =
    sorted.length >= 2 && elapsedDays >= 1 && !hasMixedExercises
      ? Number((deltaRom / (elapsedDays / 7)).toFixed(1))
      : null;

  const facilitiesSet = new Set<string>();
  let staffVisitCount = 0;
  let kioskVisitCount = 0;
  let courseAutoVisitCount = 0;
  const visitsWithDelta: LongitudinalVisitDelta[] = sorted.map((rec, idx) => {
    if (rec.facility_name?.trim()) {
      facilitiesSet.add(rec.facility_name.trim());
    }
    if (rec.source_channel === 'COURSE_AUTO_CALCULATED') {
      courseAutoVisitCount += 1;
    } else if (rec.source_channel === 'KIOSK_SESSION_AUTO_SYNC') {
      kioskVisitCount += 1;
    } else {
      staffVisitCount += 1;
    }
    const prev = idx > 0 ? sorted[idx - 1] : null;
    const targetAchievementPct = computeTargetAchievementPct(
      rec.measured_rom_deg,
      rec.target_rom_deg || 120,
    );
    const prevTargetPct = prev
      ? computeTargetAchievementPct(prev.measured_rom_deg, prev.target_rom_deg || 120)
      : null;
    return {
      record: rec,
      deltaFromPrevRom: prev ? Number((rec.measured_rom_deg - prev.measured_rom_deg).toFixed(1)) : null,
      deltaFromFirstRom: Number((rec.measured_rom_deg - firstRom).toFixed(1)),
      deltaFromPrevScore: prev ? rec.initial_score_0_to_6 - prev.initial_score_0_to_6 : null,
      targetAchievementPct,
      deltaFromPrevTargetPct:
        prevTargetPct !== null ? Number((targetAchievementPct - prevTargetPct).toFixed(1)) : null,
    };
  });

  return {
    totalVisits: sorted.length,
    firstRom,
    latestRom,
    deltaRom,
    percentGain,
    firstScore,
    latestScore,
    deltaScore,
    linkedFacilities: Array.from(facilitiesSet),
    visitsWithDelta,
    targetRomDeg,
    firstTargetAchievementPct,
    latestTargetAchievementPct,
    deltaTargetAchievementPct,
    remainingToTargetDeg,
    firstPainVas,
    latestPainVas,
    deltaPainVas,
    elapsedDays,
    romGainPerWeek,
    hasMixedExercises,
    staffVisitCount,
    kioskVisitCount,
    courseAutoVisitCount,
    exerciseBreakdown,
  };
}

function escapeCsvCell(value: unknown): string {
  const str = value === null || value === undefined ? '' : String(value);
  if (/[",\r\n]/.test(str)) {
    return `"${str.replace(/"/g, '""')}"`;
  }
  return str;
}

export function buildLongitudinalCsvReport(
  patient: { fullName?: string; hn?: string; universalHealthLinkCode?: string | null; id: string },
  summary: LongitudinalProgressSummary,
): string {
  const linkCode = resolvePatientUniversalLinkCode(patient);
  const header = [
    'HN',
    'ชื่อผู้ป่วย',
    'รหัสเชื่อมโยงกลาง',
    'ครั้งที่',
    'วันเวลาประเมิน',
    'แหล่งข้อมูล',
    'วิธีวัดผล',
    'รหัสท่ากายภาพ',
    'ชื่อท่ากายภาพ',
    'ข้าง',
    'องศาที่วัดได้ (ROM deg)',
    'เป้าหมายคลินิก (deg)',
    'ความสำเร็จเทียบเป้าหมาย (%)',
    'เปลี่ยนแปลงจากครั้งก่อน (deg)',
    'คะแนนมหิดล (0-6)',
    'ระดับปวด VAS (0-10)',
    'สถานที่/สาขา',
    'ผู้บันทึก',
    'อาการแรกรับ/ครั้งนี้',
    'บันทึกส่งต่อ',
    'เป้าหมายครั้งถัดไป',
  ];

  const rows = summary.visitsWithDelta.map(({ record, deltaFromPrevRom, targetAchievementPct }) => {
    const sourceLabel =
      record.source_channel === 'COURSE_AUTO_CALCULATED'
        ? 'คำนวณอัตโนมัติจากคอร์ส'
        : record.source_channel === 'KIOSK_SESSION_AUTO_SYNC'
          ? 'ซิงก์จากตู้ Kiosk'
          : 'ประเมินโดยบุคลากร';
    const methodLabel =
      record.measurement_method === 'COURSE_AUTO_CALCULATED' ||
      record.source_channel === 'COURSE_AUTO_CALCULATED'
        ? 'คำนวณอัตโนมัติจากคอร์ส'
        : record.measurement_method === 'CAMERA_AI_LIVE'
          ? 'วัดด้วยกล้อง AI'
          : record.measurement_method === 'KIOSK_SESSION_AUTO_SYNC' ||
              record.source_channel === 'KIOSK_SESSION_AUTO_SYNC'
            ? 'ตู้ Kiosk AI'
            : 'บันทึกโดยบุคลากร';
    const sideLabel =
      record.target_side === 'LEFT' ? 'ซ้าย' : record.target_side === 'RIGHT' ? 'ขวา' : 'ทั้งสองข้าง';
    return [
      patient.hn || record.patient_hn,
      patient.fullName || '',
      linkCode,
      record.visit_number,
      record.assessed_at,
      sourceLabel,
      methodLabel,
      record.exercise_code,
      record.exercise_name_th,
      sideLabel,
      record.measured_rom_deg.toFixed(1),
      record.target_rom_deg.toFixed(1),
      targetAchievementPct.toFixed(1),
      deltaFromPrevRom === null ? '' : deltaFromPrevRom.toFixed(1),
      record.initial_score_0_to_6,
      record.pain_vas_0_to_10 === null || record.pain_vas_0_to_10 === undefined
        ? ''
        : record.pain_vas_0_to_10,
      record.facility_name,
      record.assessor_staff_name,
      record.chief_complaint,
      record.clinical_handover_note,
      record.next_session_goal,
    ];
  });

  const lines = [header, ...rows].map((row) => row.map(escapeCsvCell).join(','));
  return `\uFEFF${lines.join('\r\n')}`;
}
