import type { DrTechCaresApiClient, RehabSessionRecord } from '@drtechcares/api-client';
import { CLINICAL_EXERCISES, parseExerciseCodeAndSide } from '@drtechcares/pose-engine';
import {
  clampMahidolScore,
  filterAssessmentsForPatient,
  isDemoSeedAssessment,
  isValidMahidolScore,
  resolvePatientUniversalLinkCode,
  type AssessmentMeasurementMethod,
  type AssessmentTargetSide,
  type CreatePatientClinicalAssessmentInput,
  type MahidolScore0To6,
  type PatientClinicalAssessmentRecord,
  type PatientIdentityRef,
} from '../types/clinicalAssessment';

const STORAGE_KEY = 'drtech_patient_clinical_assessments_v1';

export const CLINICAL_FACILITY_PRESETS: readonly string[] = [
  'ศูนย์กายภาพบำบัดผู้สูงอายุ Dr.Tech.Care สาขาหลัก',
  'คลินิกเครือข่ายกายภาพชุมชน',
  'ศูนย์ฟื้นฟูสมรรถภาพผู้สูงอายุ OPD',
];

let inMemoryFallbackRecords: PatientClinicalAssessmentRecord[] = [];
const seededDemoPatientIds = new Set<string>();

function readStorageRecords(): PatientClinicalAssessmentRecord[] {
  try {
    if (typeof window !== 'undefined' && window.localStorage) {
      const raw = window.localStorage.getItem(STORAGE_KEY);
      if (!raw) return inMemoryFallbackRecords.slice();
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed)) {
        return parsed as PatientClinicalAssessmentRecord[];
      }
    }
  } catch {
    // Fallback to in-memory store when storage is unavailable
  }
  return inMemoryFallbackRecords.slice();
}

function writeStorageRecords(records: readonly PatientClinicalAssessmentRecord[]): void {
  inMemoryFallbackRecords = records.slice();
  try {
    if (typeof window !== 'undefined' && window.localStorage) {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(records));
    }
  } catch {
    // Ignore storage quota or sandbox errors
  }
}

export function resetClinicalAssessmentStoreForTests(): void {
  inMemoryFallbackRecords = [];
  seededDemoPatientIds.clear();
  try {
    if (typeof window !== 'undefined' && window.localStorage) {
      window.localStorage.removeItem(STORAGE_KEY);
    }
  } catch {
    // Ignore
  }
}

function getSupabaseRestConfig(): { url: string; key: string } | null {
  try {
    const env = (import.meta as unknown as { env?: Record<string, string | undefined> }).env;
    const isTestRuntime =
      env?.MODE === 'test' ||
      (typeof process !== 'undefined' &&
        Boolean(process.env?.VITEST || process.env?.NODE_ENV === 'test'));
    if (isTestRuntime) {
      return null;
    }
    const url = (env?.VITE_SUPABASE_URL ?? '').trim().replace(/\/+$/, '');
    const key = (env?.VITE_SUPABASE_ANON_KEY ?? '').trim();
    if (!url || !key) return null;
    return { url, key };
  } catch {
    return null;
  }
}

function buildStableUuid(seed: string): string {
  let h1 = 0xdeadbeef ^ seed.length;
  let h2 = 0x41c6ce57 ^ seed.length;
  for (let i = 0; i < seed.length; i += 1) {
    const ch = seed.charCodeAt(i);
    h1 = Math.imul(h1 ^ ch, 2654435761);
    h2 = Math.imul(h2 ^ ch, 1597334677);
  }
  const hex = (n: number) => (n >>> 0).toString(16).padStart(8, '0');
  const p1 = hex(h1);
  const p2 = hex(h2).slice(0, 4);
  const p3 = '4' + hex(h1 ^ h2).slice(1, 4);
  const p4 = '8' + hex(h2 ^ 0x9e3779b9).slice(1, 4);
  const p5 = (hex(h1 ^ 0xa5a5a5a5) + hex(h2 ^ 0x5a5a5a5a)).slice(0, 12);
  return `${p1}-${p2}-${p3}-${p4}-${p5}`;
}

export interface DemoSeedPatientInfo extends PatientIdentityRef {
  fullName?: string;
  affectedSide?: string | null;
  diagnosis?: string;
}

/**
 * Seeds deterministic Thai longitudinal visits (2 linked branches, Visit 1 -> 2 -> 3)
 * ONLY when explicitly requested via ensureDemoAssessmentsForPatient or seedDemoIfEmpty: true.
 */
export function ensureDemoAssessmentsForPatient(
  patient: DemoSeedPatientInfo,
): PatientClinicalAssessmentRecord[] {
  const existing = readStorageRecords();
  const ownExisting = filterAssessmentsForPatient(existing, patient);
  if (ownExisting.length > 0 || seededDemoPatientIds.has(patient.id)) {
    return ownExisting;
  }
  seededDemoPatientIds.add(patient.id);

  const linkCode = resolvePatientUniversalLinkCode(patient);
  const hn = patient.hn || 'HN-CLINICAL';
  const side: AssessmentTargetSide =
    patient.affectedSide === 'LEFT'
      ? 'LEFT'
      : patient.affectedSide === 'BOTH'
        ? 'BOTH'
        : 'RIGHT';
  const sideTh = side === 'LEFT' ? 'ซ้าย' : side === 'BOTH' ? 'ทั้งสองข้าง' : 'ขวา';
  const now = Date.now();
  const dayMs = 86_400_000;

  const seeded: PatientClinicalAssessmentRecord[] = [
    {
      id: buildStableUuid(`${patient.id}-demo-visit-1`),
      patient_id: patient.id,
      patient_hn: hn,
      universal_link_code: linkCode,
      visit_number: 1,
      assessed_at: new Date(now - 14 * dayMs).toISOString(),
      facility_name: 'ศูนย์กายภาพบำบัด Dr.Tech.Care สาขาหลัก',
      assessor_staff_id: 'pt-demo-01',
      assessor_staff_name: 'กภ. วิชญ์พล ชำนาญกาย (PT อาวุโส)',
      exercise_code: 'ELDERLY_SHOULDER_FLEXION',
      exercise_name_th: 'ยกแขนตรงไปด้านหน้า (Shoulder Flexion)',
      target_side: side,
      initial_score_0_to_6: 2,
      measured_rom_deg: 30.0,
      target_rom_deg: 125.0,
      pain_vas_0_to_10: 5,
      chief_complaint: `แรกรับ: อ่อนแรงแขนข้าง${sideTh} ยกแขนต้านแรงโน้มถ่วงยังไม่ได้เต็มช่วง ตึงข้อไหล่เล็กน้อย (${patient.diagnosis || 'ฟื้นฟูสมรรถภาพผู้สูงอายุ'})`,
      clinical_handover_note:
        'ประคองข้อไหล่ข้างที่อ่อนแรงระหว่างเริ่มยก แนะนำฝึกยกแนวราบต้านแรงโน้มถ่วงบางส่วน 2 เซต ระวังไม่ให้ยักไหล่หรือเอียงลำตัวช่วย',
      next_session_goal: 'เพิ่มมุมยกแขนด้านหน้าถึง 45 องศา และลดอาการตึงบ่า',
      source_channel: 'STAFF_MANUAL_ASSESSMENT',
      measurement_method: 'MANUAL_INPUT',
      created_at: new Date(now - 14 * dayMs).toISOString(),
    },
    {
      id: buildStableUuid(`${patient.id}-demo-visit-2`),
      patient_id: patient.id,
      patient_hn: hn,
      universal_link_code: linkCode,
      visit_number: 2,
      assessed_at: new Date(now - 7 * dayMs).toISOString(),
      facility_name: 'คลินิกเครือข่ายกายภาพชุมชน',
      assessor_staff_id: 'pt-demo-02',
      assessor_staff_name: 'กภ. ณัฐิดา รักษาดี (PT เครือข่ายชุมชน)',
      exercise_code: 'ELDERLY_SHOULDER_FLEXION',
      exercise_name_th: 'ยกแขนตรงไปด้านหน้า (Shoulder Flexion)',
      target_side: side,
      initial_score_0_to_6: 3,
      measured_rom_deg: 46.5,
      target_rom_deg: 125.0,
      pain_vas_0_to_10: 3,
      chief_complaint: 'รับช่วงต่อจากสาขาหลัก ผู้ป่วยคุมข้อไหล่ได้ดีขึ้น เริ่มยกต้านแรงโน้มถ่วงได้มากขึ้น',
      clinical_handover_note:
        'ดูประวัติส่งต่อผ่าน Universal Link แล้ว เพิ่มการฝึกค้างท่าที่มุม 45 องศา 3 วินาที เตือนให้ผู้ป่วยควบคุมศอกให้เหยียดตรงตลอดจังหวะยก',
      next_session_goal: 'ตั้งเป้าเพิ่มองศาข้อไหล่แตะ 60 องศา ในสัปดาห์ถัดไป',
      source_channel: 'STAFF_MANUAL_ASSESSMENT',
      measurement_method: 'MANUAL_INPUT',
      created_at: new Date(now - 7 * dayMs).toISOString(),
    },
    {
      id: buildStableUuid(`${patient.id}-demo-visit-3`),
      patient_id: patient.id,
      patient_hn: hn,
      universal_link_code: linkCode,
      visit_number: 3,
      assessed_at: new Date(now - 2 * dayMs).toISOString(),
      facility_name: 'ศูนย์กายภาพบำบัด Dr.Tech.Care สาขาหลัก',
      assessor_staff_id: 'pt-demo-01',
      assessor_staff_name: 'กภ. วิชญ์พล ชำนาญกาย (PT อาวุโส)',
      exercise_code: 'ELDERLY_SHOULDER_FLEXION',
      exercise_name_th: 'ยกแขนตรงไปด้านหน้า (Shoulder Flexion)',
      target_side: side,
      initial_score_0_to_6: 4,
      measured_rom_deg: 62.0,
      target_rom_deg: 125.0,
      pain_vas_0_to_10: 2,
      chief_complaint: 'ติดตามผลรอบที่ 3 ยกแขนไปด้านหน้าได้คล่องขึ้นชัดเจน อาการเกร็งลดลง',
      clinical_handover_note:
        'สามารถต้านแรงต้านระดับปานกลางช่วงท้ายได้แล้ว ให้เริ่มฝึกเสริมผ่านตู้ Kiosk และเตรียมขยับสู่กิจกรรมหยิบจับสิ่งของบนชั้นระดับอก',
      next_session_goal: 'ขยับมุมข้อไหล่สู่ 85-90 องศา และประเมินระดับ 5 ในรอบหน้า',
      source_channel: 'STAFF_MANUAL_ASSESSMENT',
      measurement_method: 'MANUAL_INPUT',
      created_at: new Date(now - 2 * dayMs).toISOString(),
    },
  ];

  writeStorageRecords([...existing, ...seeded]);
  return seeded;
}

function normalizeSupabaseRow(row: Record<string, unknown>): PatientClinicalAssessmentRecord {
  const rawScore = Number(row.initial_score_0_to_6 ?? 0);
  const sideRaw = String(row.target_side ?? 'RIGHT').toUpperCase();
  const targetSide: AssessmentTargetSide =
    sideRaw === 'LEFT' ? 'LEFT' : sideRaw === 'BOTH' ? 'BOTH' : 'RIGHT';
  const channelRaw = String(row.source_channel ?? 'STAFF_MANUAL_ASSESSMENT');
  const sourceChannel =
    channelRaw === 'COURSE_AUTO_CALCULATED'
      ? 'COURSE_AUTO_CALCULATED'
      : channelRaw === 'KIOSK_SESSION_AUTO_SYNC'
        ? 'KIOSK_SESSION_AUTO_SYNC'
        : 'STAFF_MANUAL_ASSESSMENT';
  const methodRaw = String(row.measurement_method ?? '');
  const measurementMethod: AssessmentMeasurementMethod =
    methodRaw === 'COURSE_AUTO_CALCULATED' || sourceChannel === 'COURSE_AUTO_CALCULATED'
      ? 'COURSE_AUTO_CALCULATED'
      : methodRaw === 'CAMERA_AI_LIVE'
        ? 'CAMERA_AI_LIVE'
        : methodRaw === 'KIOSK_SESSION_AUTO_SYNC' || sourceChannel === 'KIOSK_SESSION_AUTO_SYNC'
          ? 'KIOSK_SESSION_AUTO_SYNC'
          : 'MANUAL_INPUT';
  const resolvedSourceSessionId =
    typeof row.source_session_id === 'string' && row.source_session_id.trim()
      ? row.source_session_id.trim()
      : typeof row.session_id === 'string' && row.session_id.trim()
        ? row.session_id.trim()
        : null;

  return {
    id: String(row.id ?? `assess-${Date.now()}`),
    patient_id: String(row.patient_id ?? ''),
    patient_hn: String(row.patient_hn ?? ''),
    universal_link_code: String(row.universal_link_code ?? ''),
    visit_number: Math.max(1, Number(row.visit_number ?? 1)),
    assessed_at: String(row.assessed_at ?? row.created_at ?? new Date().toISOString()),
    facility_name: String(row.facility_name ?? 'ศูนย์กายภาพบำบัด Dr.Tech.Care สาขาหลัก'),
    assessor_staff_id: String(row.assessor_staff_id ?? ''),
    assessor_staff_name: String(row.assessor_staff_name ?? ''),
    exercise_code: String(row.exercise_code ?? 'ELDERLY_SHOULDER_FLEXION'),
    exercise_name_th: String(row.exercise_name_th ?? ''),
    target_side: targetSide,
    initial_score_0_to_6: clampMahidolScore(rawScore),
    measured_rom_deg: Number(Number(row.measured_rom_deg ?? 0).toFixed(1)),
    target_rom_deg: Number(Number(row.target_rom_deg ?? 120).toFixed(1)),
    pain_vas_0_to_10:
      row.pain_vas_0_to_10 === null || row.pain_vas_0_to_10 === undefined
        ? null
        : Number(row.pain_vas_0_to_10),
    chief_complaint: String(row.chief_complaint ?? ''),
    clinical_handover_note: String(row.clinical_handover_note ?? ''),
    next_session_goal: String(row.next_session_goal ?? ''),
    source_channel: sourceChannel,
    measurement_method: measurementMethod,
    completed_reps:
      typeof row.completed_reps === 'number' && Number.isFinite(row.completed_reps)
        ? Math.round(row.completed_reps)
        : null,
    accuracy_percent:
      typeof row.accuracy_percent === 'number' && Number.isFinite(row.accuracy_percent)
        ? Math.round(row.accuracy_percent)
        : null,
    session_id: typeof row.session_id === 'string' ? row.session_id : resolvedSourceSessionId,
    source_session_id: resolvedSourceSessionId,
    created_at: row.created_at ? String(row.created_at) : undefined,
  };
}

export function deriveScore0To6FromRomRatio(
  measuredRomDeg: number,
  targetRomDeg: number,
): MahidolScore0To6 {
  const ratio = targetRomDeg > 0 ? measuredRomDeg / targetRomDeg : 0;
  if (ratio >= 0.98) return 6;
  if (ratio >= 0.85) return 5;
  if (ratio >= 0.65) return 4;
  if (ratio >= 0.45) return 3;
  if (ratio >= 0.25) return 2;
  if (ratio > 0.05) return 1;
  return 0;
}

export function convertRehabSessionToAssessment(
  session: RehabSessionRecord,
  patient: DemoSeedPatientInfo,
): PatientClinicalAssessmentRecord | null {
  if (
    session.activityType !== 'physio_exercise' ||
    session.exerciseId === 'CLINIC_CHECKIN' ||
    !(typeof session.maxRomDeg === 'number' && Number.isFinite(session.maxRomDeg) && session.maxRomDeg > 0)
  ) {
    return null;
  }

  const parsed = parseExerciseCodeAndSide(session.exerciseId, session.bodySide);
  const catalog = CLINICAL_EXERCISES.find(
    (e) => e.code === parsed.baseCode || e.id === session.exerciseId,
  );
  const exCode = (catalog?.code || parsed.baseCode || session.exerciseId).toUpperCase();
  const targetRom = Number((catalog?.detection.targetMinDeg ?? 120).toFixed(1));
  const romRounded = Number(session.maxRomDeg.toFixed(1));
  const score =
    typeof session.motorScoreAtSession === 'number' && Number.isFinite(session.motorScoreAtSession)
      ? clampMahidolScore(session.motorScoreAtSession)
      : deriveScore0To6FromRomRatio(romRounded, targetRom);
  const side: AssessmentTargetSide =
    parsed.side === 'LEFT' ? 'LEFT' : parsed.side === 'BOTH' ? 'BOTH' : 'RIGHT';

  return {
    id: `kiosk-sess-${session.id}`,
    patient_id: session.patientId,
    patient_hn: patient.hn ?? '',
    universal_link_code: resolvePatientUniversalLinkCode(patient),
    visit_number: 1,
    assessed_at: session.createdAt,
    facility_name: session.facilityName || 'ตู้ Kiosk กายภาพอัจฉริยะ Dr.Tech.Care',
    assessor_staff_id: 'SYSTEM_KIOSK_AUTO_SYNC',
    assessor_staff_name: session.recordedByPtName || 'ระบบวัดผลอัตโนมัติ (Kiosk AI)',
    exercise_code: exCode,
    exercise_name_th: catalog?.name.th || session.exerciseName || exCode,
    target_side: side,
    initial_score_0_to_6: score,
    measured_rom_deg: romRounded,
    target_rom_deg: targetRom,
    pain_vas_0_to_10: null,
    chief_complaint: `ฝึกท่า ${catalog?.name.th || session.exerciseName || exCode} ผ่านตู้ Kiosk (${session.completedReps} ครั้ง / ความแม่นยำ ${Math.round(session.accuracyPercent ?? 100)}%)`,
    clinical_handover_note:
      session.aiFeedback ||
      `วัดมุมข้อต่อสูงสุดจริงได้ ${romRounded.toFixed(1)}° เทียบเป้าหมาย ${targetRom.toFixed(1)}°`,
    next_session_goal: `ติดตามพัฒนาการท่า ${catalog?.name.th || session.exerciseName || exCode} ต่อเนื่อง`,
    source_channel: 'KIOSK_SESSION_AUTO_SYNC',
    measurement_method: 'KIOSK_SESSION_AUTO_SYNC',
    completed_reps: session.completedReps ?? null,
    accuracy_percent:
      typeof session.accuracyPercent === 'number' ? Math.round(session.accuracyPercent) : null,
    session_id: session.id,
    created_at: session.createdAt,
  };
}

function hasEquivalentAssessment(
  map: Map<string, PatientClinicalAssessmentRecord>,
  candidate: PatientClinicalAssessmentRecord,
): boolean {
  const candTs = new Date(candidate.assessed_at).getTime();
  const candSourceSessionId = candidate.source_session_id || candidate.session_id || null;
  for (const existing of map.values()) {
    if (existing.id === candidate.id) return true;
    if (
      candSourceSessionId &&
      (existing.source_session_id === candSourceSessionId ||
        existing.session_id === candSourceSessionId ||
        existing.id === candSourceSessionId)
    ) {
      return true;
    }
    if (existing.exercise_code.toUpperCase() !== candidate.exercise_code.toUpperCase()) continue;
    const existTs = new Date(existing.assessed_at).getTime();
    if (
      Number.isFinite(candTs) &&
      Number.isFinite(existTs) &&
      Math.abs(candTs - existTs) <= 90_000 &&
      Math.abs(existing.measured_rom_deg - candidate.measured_rom_deg) <= 0.6
    ) {
      return true;
    }
  }
  return false;
}

async function fetchWithTimeout(url: string, init: RequestInit, timeoutMs = 5000): Promise<Response> {
  if (typeof AbortController === 'undefined') {
    return fetch(url, init);
  }
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...init, signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

export async function listPatientAssessments(
  patient: DemoSeedPatientInfo,
  options?: {
    includeCrossFacility?: boolean;
    seedDemoIfEmpty?: boolean;
    excludeDemoRows?: boolean;
    apiClient?: DrTechCaresApiClient;
    sessions?: readonly RehabSessionRecord[];
  },
): Promise<PatientClinicalAssessmentRecord[]> {
  let localAll = readStorageRecords();
  if (options?.excludeDemoRows) {
    const cleanedLocal = localAll.filter((item) => !isDemoSeedAssessment(item));
    if (cleanedLocal.length !== localAll.length) {
      writeStorageRecords(cleanedLocal);
      localAll = cleanedLocal;
    }
  }

  const byId = new Map<string, PatientClinicalAssessmentRecord>();
  for (const item of localAll) {
    if (options?.excludeDemoRows && isDemoSeedAssessment(item)) continue;
    byId.set(item.id, item);
  }

  let loadedFromServerApi = false;
  if (options?.apiClient) {
    try {
      const apiRows = await options.apiClient.listPatientClinicalAssessments(patient.id, {
        includeCrossFacility: options.includeCrossFacility !== false,
      });
      if (Array.isArray(apiRows)) {
        loadedFromServerApi = true;
        const persistableToCache: PatientClinicalAssessmentRecord[] = [];
        for (const row of apiRows) {
          if (options.excludeDemoRows && isDemoSeedAssessment(row)) continue;
          const norm = normalizeSupabaseRow(row as unknown as Record<string, unknown>);
          byId.set(norm.id, norm);
          if (!norm.id.startsWith('kiosk-sess-') && !norm.id.startsWith('hist-sess-')) {
            persistableToCache.push(norm);
          }
        }
        if (persistableToCache.length > 0) {
          const cacheMap = new Map(localAll.map((r) => [r.id, r]));
          for (const rec of persistableToCache) {
            cacheMap.set(rec.id, rec);
          }
          writeStorageRecords(Array.from(cacheMap.values()));
        }
      }
    } catch {
      // Fall back safely to direct Supabase REST / local records
    }
  }

  const supa = !loadedFromServerApi ? getSupabaseRestConfig() : null;
  if (supa && typeof fetch === 'function') {
    try {
      const linkCode = resolvePatientUniversalLinkCode(patient);
      const pidEnc = encodeURIComponent(patient.id);
      const linkEnc = encodeURIComponent(linkCode);
      const filterExpr = options?.includeCrossFacility === false
        ? `patient_id=eq.${pidEnc}`
        : `or=(patient_id.eq.${pidEnc},universal_link_code.eq.${linkEnc})`;
      const res = await fetchWithTimeout(
        `${supa.url}/rest/v1/patient_clinical_assessments?${filterExpr}&order=assessed_at.asc,visit_number.asc`,
        {
          method: 'GET',
          headers: {
            apikey: supa.key,
            Authorization: `Bearer ${supa.key}`,
            'Content-Type': 'application/json',
          },
        },
        5000,
      );
      if (res.ok) {
        const remoteRows = (await res.json()) as Record<string, unknown>[];
        if (Array.isArray(remoteRows)) {
          let updatedLocal = false;
          for (const rawRow of remoteRows) {
            const norm = normalizeSupabaseRow(rawRow);
            if (options?.excludeDemoRows && isDemoSeedAssessment(norm)) continue;
            if (!byId.has(norm.id)) {
              byId.set(norm.id, norm);
              updatedLocal = true;
            }
          }
          if (updatedLocal) {
            writeStorageRecords(
              Array.from(byId.values()).filter(
                (r) => !r.id.startsWith('kiosk-sess-') && !r.id.startsWith('hist-sess-'),
              ),
            );
          }
        }
      }
    } catch {
      // Offline / sandbox fallback safely uses local records
    }
  }

  // Merge real Kiosk physio sessions provided by caller (deduplicated against persisted assessments)
  if (options?.sessions && options.sessions.length > 0) {
    for (const s of options.sessions) {
      if (s.patientId !== patient.id) continue;
      const converted = convertRehabSessionToAssessment(s, patient);
      if (!converted) continue;
      if (!hasEquivalentAssessment(byId, converted)) {
        byId.set(converted.id, converted);
      }
    }
  }

  let merged = Array.from(byId.values());
  if (
    options?.seedDemoIfEmpty &&
    !options?.excludeDemoRows &&
    filterAssessmentsForPatient(merged, patient).length === 0
  ) {
    ensureDemoAssessmentsForPatient(patient);
    merged = readStorageRecords();
  }

  const scoped =
    options?.includeCrossFacility === false
      ? merged
          .filter((r) => r.patient_id === patient.id)
          .sort((a, b) => new Date(a.assessed_at).getTime() - new Date(b.assessed_at).getTime())
      : filterAssessmentsForPatient(merged, patient);

  // Re-index visit numbers sequentially only when synthetic Kiosk session rows are merged in
  const hasSyntheticKioskRows = scoped.some(
    (r) => r.id.startsWith('kiosk-sess-') || r.id.startsWith('hist-sess-'),
  );
  if (hasSyntheticKioskRows) {
    return scoped.map((rec, idx) => ({
      ...rec,
      visit_number: idx + 1,
    }));
  }
  return scoped;
}

export async function createPatientAssessment(
  input: CreatePatientClinicalAssessmentInput,
  apiClient?: DrTechCaresApiClient,
): Promise<PatientClinicalAssessmentRecord> {
  if (!isValidMahidolScore(input.initial_score_0_to_6)) {
    throw new Error(
      `คะแนนประเมินตามเกณฑ์มหิดลต้องเป็นจำนวนเต็ม 0 ถึง 6 เท่านั้น (ได้รับ: ${String(input.initial_score_0_to_6)})`,
    );
  }
  if (!input.patient_id?.trim()) {
    throw new Error('กรุณาเลือกผู้ป่วยก่อนบันทึกผลการประเมิน');
  }
  const rom = Number(input.measured_rom_deg);
  const safeRom = Number.isFinite(rom) ? Number(Math.max(0, Math.min(220, rom)).toFixed(1)) : 0;
  const targetRom = Number.isFinite(Number(input.target_rom_deg))
    ? Number(Math.max(10, Math.min(220, Number(input.target_rom_deg))).toFixed(1))
    : 120.0;

  const existing = readStorageRecords();
  const patientExisting = filterAssessmentsForPatient(existing, {
    id: input.patient_id,
    hn: input.patient_hn,
    universalHealthLinkCode: input.universal_link_code,
  });
  const cleanSourceSessionId = (input.source_session_id || input.session_id || '').trim() || null;
  if (cleanSourceSessionId) {
    const duplicateLocal = patientExisting.find(
      (rec) =>
        !rec.id.startsWith('kiosk-sess-') &&
        !rec.id.startsWith('hist-sess-') &&
        (rec.source_session_id === cleanSourceSessionId || rec.session_id === cleanSourceSessionId),
    );
    if (duplicateLocal) {
      return duplicateLocal;
    }
  }
  const nextVisitNumber =
    patientExisting.reduce((maxV, rec) => Math.max(maxV, rec.visit_number), 0) + 1;

  const nowIso = input.assessed_at || new Date().toISOString();
  const linkCode =
    input.universal_link_code.trim().toUpperCase() ||
    resolvePatientUniversalLinkCode({ id: input.patient_id, hn: input.patient_hn });
  const sourceChannel = input.source_channel ?? 'STAFF_MANUAL_ASSESSMENT';
  const measurementMethod: AssessmentMeasurementMethod =
    input.measurement_method ??
    (sourceChannel === 'COURSE_AUTO_CALCULATED'
      ? 'COURSE_AUTO_CALCULATED'
      : sourceChannel === 'KIOSK_SESSION_AUTO_SYNC'
        ? 'KIOSK_SESSION_AUTO_SYNC'
        : 'MANUAL_INPUT');

  // Prefer server API persistence (Cloudflare Worker + Supabase service role) when available
  if (apiClient) {
    try {
      const savedRemote = await apiClient.createPatientClinicalAssessment(input.patient_id.trim(), {
        patient_hn: (input.patient_hn || '').trim(),
        universal_link_code: linkCode,
        assessed_at: nowIso,
        facility_name: input.facility_name.trim() || CLINICAL_FACILITY_PRESETS[0],
        assessor_staff_id: input.assessor_staff_id.trim(),
        assessor_staff_name: input.assessor_staff_name.trim() || 'บุคลากรคลินิกกายภาพบำบัด',
        exercise_code: input.exercise_code.trim().toUpperCase(),
        exercise_name_th: input.exercise_name_th.trim(),
        target_side: input.target_side,
        initial_score_0_to_6: input.initial_score_0_to_6,
        measured_rom_deg: safeRom,
        target_rom_deg: targetRom,
        pain_vas_0_to_10:
          input.pain_vas_0_to_10 === null || input.pain_vas_0_to_10 === undefined
            ? null
            : Math.max(0, Math.min(10, Math.round(input.pain_vas_0_to_10))),
        chief_complaint: input.chief_complaint.trim(),
        clinical_handover_note: input.clinical_handover_note.trim(),
        next_session_goal: input.next_session_goal.trim(),
        source_channel: sourceChannel,
        measurement_method: measurementMethod,
        completed_reps: input.completed_reps ?? null,
        accuracy_percent: input.accuracy_percent ?? null,
        session_id: cleanSourceSessionId,
        source_session_id: cleanSourceSessionId,
      });
      const normalized = normalizeSupabaseRow(savedRemote as unknown as Record<string, unknown>);
      writeStorageRecords([...existing.filter((r) => r.id !== normalized.id), normalized]);
      return normalized;
    } catch {
      // Fall back to local + direct REST persistence if network/server unreachable
    }
  }

  const newRecord: PatientClinicalAssessmentRecord = {
    id: cleanSourceSessionId
      ? buildStableUuid(`course-auto-${input.patient_id.trim()}-${cleanSourceSessionId}`)
      : typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function'
        ? crypto.randomUUID()
        : buildStableUuid(`${input.patient_id}-${nowIso}-${nextVisitNumber}-${Math.random()}`),
    patient_id: input.patient_id.trim(),
    patient_hn: (input.patient_hn || '').trim(),
    universal_link_code: linkCode,
    visit_number: nextVisitNumber,
    assessed_at: nowIso,
    facility_name: input.facility_name.trim() || CLINICAL_FACILITY_PRESETS[0],
    assessor_staff_id: input.assessor_staff_id.trim(),
    assessor_staff_name: input.assessor_staff_name.trim() || 'บุคลากรคลินิกกายภาพบำบัด',
    exercise_code: input.exercise_code.trim().toUpperCase(),
    exercise_name_th: input.exercise_name_th.trim(),
    target_side: input.target_side,
    initial_score_0_to_6: input.initial_score_0_to_6,
    measured_rom_deg: safeRom,
    target_rom_deg: targetRom,
    pain_vas_0_to_10:
      input.pain_vas_0_to_10 === null || input.pain_vas_0_to_10 === undefined
        ? null
        : Math.max(0, Math.min(10, Math.round(input.pain_vas_0_to_10))),
    chief_complaint: input.chief_complaint.trim(),
    clinical_handover_note: input.clinical_handover_note.trim(),
    next_session_goal: input.next_session_goal.trim(),
    source_channel: sourceChannel,
    measurement_method: measurementMethod,
    completed_reps: input.completed_reps ?? null,
    accuracy_percent: input.accuracy_percent ?? null,
    session_id: cleanSourceSessionId,
    source_session_id: cleanSourceSessionId,
    created_at: nowIso,
  };

  writeStorageRecords([...existing.filter((r) => r.id !== newRecord.id), newRecord]);

  const supa = getSupabaseRestConfig();
  if (supa && typeof fetch === 'function') {
    try {
      const restRow: Record<string, unknown> = {
        id: newRecord.id,
        patient_id: newRecord.patient_id,
        patient_hn: newRecord.patient_hn,
        universal_link_code: newRecord.universal_link_code,
        visit_number: newRecord.visit_number,
        assessed_at: newRecord.assessed_at,
        facility_name: newRecord.facility_name,
        assessor_staff_id: newRecord.assessor_staff_id,
        assessor_staff_name: newRecord.assessor_staff_name,
        exercise_code: newRecord.exercise_code,
        exercise_name_th: newRecord.exercise_name_th,
        target_side: newRecord.target_side,
        initial_score_0_to_6: newRecord.initial_score_0_to_6,
        measured_rom_deg: newRecord.measured_rom_deg,
        target_rom_deg: newRecord.target_rom_deg,
        pain_vas_0_to_10: newRecord.pain_vas_0_to_10 ?? null,
        chief_complaint: newRecord.chief_complaint,
        clinical_handover_note: newRecord.clinical_handover_note,
        next_session_goal: newRecord.next_session_goal,
        source_channel: newRecord.source_channel,
        created_at: newRecord.created_at,
      };
      if (newRecord.source_session_id) {
        restRow.source_session_id = newRecord.source_session_id;
      }
      const res = await fetchWithTimeout(
        `${supa.url}/rest/v1/patient_clinical_assessments`,
        {
          method: 'POST',
          headers: {
            apikey: supa.key,
            Authorization: `Bearer ${supa.key}`,
            'Content-Type': 'application/json',
            Prefer: 'return=minimal,resolution=merge-duplicates',
          },
          body: JSON.stringify(restRow),
        },
        5000,
      );
      if (!res.ok && (newRecord.source_channel === 'COURSE_AUTO_CALCULATED' || restRow.source_session_id)) {
        const legacyRow = { ...restRow };
        delete legacyRow.source_session_id;
        legacyRow.source_channel = 'KIOSK_SESSION_AUTO_SYNC';
        await fetchWithTimeout(
          `${supa.url}/rest/v1/patient_clinical_assessments`,
          {
            method: 'POST',
            headers: {
              apikey: supa.key,
              Authorization: `Bearer ${supa.key}`,
              'Content-Type': 'application/json',
              Prefer: 'return=minimal,resolution=merge-duplicates',
            },
            body: JSON.stringify(legacyRow),
          },
          5000,
        );
      }
    } catch {
      // Local persistence already committed; remote sync failure does not break UI flow
    }
  }

  return newRecord;
}

export async function deletePatientAssessment(
  patientId: string,
  assessmentId: string,
  apiClient?: DrTechCaresApiClient,
): Promise<boolean> {
  if (apiClient) {
    await apiClient.deletePatientClinicalAssessment(patientId, assessmentId);
  }
  const existing = readStorageRecords();
  const next = existing.filter((r) => r.id !== assessmentId);
  writeStorageRecords(next);
  return true;
}

export async function syncKioskSessionToAssessment(params: {
  patientId: string;
  patientHn?: string;
  universalHealthLinkCode?: string | null;
  exerciseCode: string;
  exerciseNameTh: string;
  targetSide: AssessmentTargetSide;
  measuredRomDeg: number;
  targetRomDeg: number;
  completedReps?: number;
  facilityName?: string;
}): Promise<PatientClinicalAssessmentRecord> {
  const linkCode = resolvePatientUniversalLinkCode({
    id: params.patientId,
    hn: params.patientHn,
    universalHealthLinkCode: params.universalHealthLinkCode,
  });
  const existing = filterAssessmentsForPatient(readStorageRecords(), {
    id: params.patientId,
    hn: params.patientHn,
    universalHealthLinkCode: linkCode,
  });
  const latest = existing.length > 0 ? existing[existing.length - 1] : null;
  const safeMeasuredRom =
    params.measuredRomDeg > 0
      ? Number(params.measuredRomDeg.toFixed(1))
      : Number((params.targetRomDeg * 0.85).toFixed(1));
  const derivedScore = deriveScore0To6FromRomRatio(safeMeasuredRom, params.targetRomDeg);
  const effectiveScore: MahidolScore0To6 = latest
    ? clampMahidolScore(Math.max(latest.initial_score_0_to_6, derivedScore))
    : derivedScore;

  return createPatientAssessment({
    patient_id: params.patientId,
    patient_hn: params.patientHn ?? '',
    universal_link_code: linkCode,
    facility_name: params.facilityName ?? 'ตู้ Kiosk กายภาพอัจฉริยะ Dr.Tech.Care',
    assessor_staff_id: 'SYSTEM_KIOSK_AUTO_SYNC',
    assessor_staff_name: 'ระบบบันทึกอัตโนมัติ (Kiosk AI)',
    exercise_code: params.exerciseCode,
    exercise_name_th: params.exerciseNameTh,
    target_side: params.targetSide,
    initial_score_0_to_6: effectiveScore,
    measured_rom_deg: safeMeasuredRom,
    target_rom_deg: params.targetRomDeg,
    pain_vas_0_to_10: null,
    chief_complaint: `ฝึกท่า ${params.exerciseNameTh} ผ่านตู้ Kiosk ครบ ${params.completedReps ?? 0} ครั้ง`,
    clinical_handover_note: `ซิงก์ผลอัตโนมัติจากตู้ Kiosk: วัดมุมข้อต่อสูงสุดได้ ${safeMeasuredRom.toFixed(1)} องศา (เป้าหมาย ${params.targetRomDeg.toFixed(1)} องศา)`,
    next_session_goal: `ติดตามพัฒนาการท่า ${params.exerciseNameTh} และประเมินคะแนนมหิดลโดยนักกายภาพ`,
    source_channel: 'KIOSK_SESSION_AUTO_SYNC',
    measurement_method: 'KIOSK_SESSION_AUTO_SYNC',
    completed_reps: params.completedReps ?? null,
  });
}

export async function upsertCourseAutoAssessment(
  input: CreatePatientClinicalAssessmentInput,
  apiClient?: DrTechCaresApiClient,
): Promise<{ record: PatientClinicalAssessmentRecord; created: boolean }> {
  const cleanSourceSessionId = (input.source_session_id || input.session_id || '').trim();
  if (cleanSourceSessionId) {
    const localExisting = filterAssessmentsForPatient(readStorageRecords(), {
      id: input.patient_id,
      hn: input.patient_hn,
      universalHealthLinkCode: input.universal_link_code,
    }).find(
      (rec) =>
        !rec.id.startsWith('kiosk-sess-') &&
        !rec.id.startsWith('hist-sess-') &&
        (rec.source_session_id === cleanSourceSessionId || rec.session_id === cleanSourceSessionId),
    );
    if (localExisting) {
      return { record: localExisting, created: false };
    }
  }

  const saved = await createPatientAssessment(
    {
      ...input,
      source_channel: 'COURSE_AUTO_CALCULATED',
      measurement_method: 'COURSE_AUTO_CALCULATED',
      source_session_id: cleanSourceSessionId || null,
      session_id: cleanSourceSessionId || null,
    },
    apiClient,
  );
  return { record: saved, created: true };
}
