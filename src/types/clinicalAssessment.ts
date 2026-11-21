export type MahidolScore0To6 = 0 | 1 | 2 | 3 | 4 | 5 | 6;

export type AssessmentTargetSide = 'LEFT' | 'RIGHT' | 'BOTH';

export type AssessmentSourceChannel = 'STAFF_MANUAL_ASSESSMENT' | 'KIOSK_SESSION_AUTO_SYNC';

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
}

export interface LongitudinalVisitDelta {
  record: PatientClinicalAssessmentRecord;
  deltaFromPrevRom: number | null;
  deltaFromFirstRom: number;
  deltaFromPrevScore: number | null;
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

export function computeLongitudinalProgressSummary(
  records: readonly PatientClinicalAssessmentRecord[],
  exerciseCodeFilter?: string | null,
): LongitudinalProgressSummary {
  const cleanFilter = (exerciseCodeFilter ?? '').trim().toUpperCase();
  const filtered = cleanFilter && cleanFilter !== 'ALL'
    ? records.filter((r) => (r.exercise_code ?? '').trim().toUpperCase() === cleanFilter)
    : records.slice();

  const sorted = filtered.slice().sort((a, b) => {
    const ta = new Date(a.assessed_at).getTime();
    const tb = new Date(b.assessed_at).getTime();
    const validTa = Number.isFinite(ta) ? ta : 0;
    const validTb = Number.isFinite(tb) ? tb : 0;
    if (validTa !== validTb) return validTa - validTb;
    return a.visit_number - b.visit_number;
  });

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

  const facilitiesSet = new Set<string>();
  const visitsWithDelta: LongitudinalVisitDelta[] = sorted.map((rec, idx) => {
    if (rec.facility_name?.trim()) {
      facilitiesSet.add(rec.facility_name.trim());
    }
    const prev = idx > 0 ? sorted[idx - 1] : null;
    return {
      record: rec,
      deltaFromPrevRom: prev ? Number((rec.measured_rom_deg - prev.measured_rom_deg).toFixed(1)) : null,
      deltaFromFirstRom: Number((rec.measured_rom_deg - firstRom).toFixed(1)),
      deltaFromPrevScore: prev ? rec.initial_score_0_to_6 - prev.initial_score_0_to_6 : null,
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
  };
}

