import type { SupabaseClient } from '@supabase/supabase-js';
import {
  ACTIVE_BIOMETRIC_DIM,
  ACTIVE_BIOMETRIC_PIPELINE_VERSION,
  parseEmbeddingVector,
  type GeometricAnthropometricRatios,
} from '@drtechcares/smart-biometric-core';
import type {
  AffectedSide,
  AngleEmbeddings5D,
  ClinicalAssessmentRecord,
  PatientRecord,
  RehabScheduleItem,
  RehabSessionHistoryEntry,
  StrokeType,
} from './index';
import {
  isSupabaseCircuitOpen,
  recordUnsupportedColumn,
  stripOversizedBase64ForDb,
  stripUnsupportedColumnsForWrite,
} from './supabasePersistence';

const VALID_STROKE_TYPES = new Set<StrokeType>([
  'ISCHEMIC',
  'HEMORRHAGIC',
  'ELDERLY_GENERAL',
  'ORTHOPEDIC',
  'NEURO',
]);

const VALID_AFFECTED_SIDES = new Set<AffectedSide>(['LEFT', 'RIGHT', 'BOTH', 'NONE']);

export interface RawPatientBiometricsSource {
  masterEmbedding?: number[];
  angleEmbeddings?: AngleEmbeddings5D;
  exemplarEmbeddings?: number[][];
  biometricEngine?: 'human-faceres-1024d' | 'arcface-onnx' | 'hybrid-512d';
  framesCaptured?: number;
  avatarUrl?: string;
  geometricRatios?: GeometricAnthropometricRatios;
  updatedAt?: string;
}

export interface NormalizePatientOptions {
  biometricsRow?: RawPatientBiometricsSource | null;
  prevPatient?: PatientRecord | null;
  /**
   * When true, `faceBiometricStatus` is set to `'VERIFIED'` only if embeddings match
   * `ACTIVE_BIOMETRIC_DIM` (1024-D); legacy 512-D vectors receive `biometricStatusDetail: 'REENROLL_REQUIRED'`.
   */
  strictCurrentDimOnly?: boolean;
  /**
   * When true, a `'VERIFIED'` flag without any real embedding vector (>= 128-D) is downgraded to `'PENDING'`.
   * Defaults to true to prevent empty-vector ghosts from pretending to be verified.
   */
  requireRealVectorsForVerified?: boolean;
  /** Default fallback values when clinical profile fields are completely absent from both top-level and history. */
  applyClinicalDefaults?: boolean;
}

export interface PatientHydrationInspection {
  patient: PatientRecord;
  hasRealVectors: boolean;
  needsTopLevelRepair: boolean;
  topLevelRepairPatch: Record<string, unknown> | null;
}

function asRecord(val: unknown): Record<string, unknown> {
  if (val && typeof val === 'object' && !Array.isArray(val)) {
    return val as Record<string, unknown>;
  }
  if (typeof val === 'string' && val.trim().startsWith('{')) {
    try {
      const parsed = JSON.parse(val);
      if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
        return parsed as Record<string, unknown>;
      }
    } catch {
      return {};
    }
  }
  return {};
}

function nonEmptyString(val: unknown): string | undefined {
  if (typeof val !== 'string') return undefined;
  const trimmed = val.trim();
  return trimmed.length > 0 ? trimmed : undefined;
}

function positiveInt(val: unknown): number | undefined {
  if (typeof val === 'number' && Number.isFinite(val) && val > 0) {
    return Math.round(val);
  }
  if (typeof val === 'string' && /^\d+$/.test(val.trim())) {
    const parsed = Number.parseInt(val.trim(), 10);
    if (Number.isFinite(parsed) && parsed > 0) return parsed;
  }
  return undefined;
}

function finiteNumberInRange(val: unknown, min: number, max: number): number | undefined {
  let num: number | undefined;
  if (typeof val === 'number' && Number.isFinite(val)) {
    num = val;
  } else if (typeof val === 'string' && /^-?\d+(\.\d+)?$/.test(val.trim())) {
    const parsed = Number(val.trim());
    if (Number.isFinite(parsed)) num = parsed;
  }
  if (num === undefined || num < min || num > max) return undefined;
  return num;
}

export function parseAngleEmbeddingsLoose(raw: unknown): AngleEmbeddings5D | undefined {
  let obj: Record<string, unknown> | undefined;
  if (raw && typeof raw === 'object' && !Array.isArray(raw)) {
    const rec = raw as Record<string, unknown>;
    obj =
      rec.angle_embeddings && typeof rec.angle_embeddings === 'object'
        ? (rec.angle_embeddings as Record<string, unknown>)
        : rec.angleEmbeddings && typeof rec.angleEmbeddings === 'object'
          ? (rec.angleEmbeddings as Record<string, unknown>)
          : rec;
  } else if (typeof raw === 'string' && raw.trim().startsWith('{')) {
    try {
      const parsed = JSON.parse(raw);
      if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
        const rec = parsed as Record<string, unknown>;
        obj =
          rec.angle_embeddings && typeof rec.angle_embeddings === 'object'
            ? (rec.angle_embeddings as Record<string, unknown>)
            : rec.angleEmbeddings && typeof rec.angleEmbeddings === 'object'
              ? (rec.angleEmbeddings as Record<string, unknown>)
              : rec;
      }
    } catch {
      return undefined;
    }
  }
  if (!obj) return undefined;
  const center = parseEmbeddingVector(obj.center);
  const left = parseEmbeddingVector(obj.left);
  const right = parseEmbeddingVector(obj.right);
  const up = parseEmbeddingVector(obj.up);
  const down = parseEmbeddingVector(obj.down);
  const base = center ?? left ?? right ?? up ?? down;
  if (!base) return undefined;
  return {
    center: center ?? base,
    left: left ?? base,
    right: right ?? base,
    up: up ?? base,
    down: down ?? base,
  };
}

export function parseExemplarEmbeddingsLoose(raw: unknown): number[][] {
  let source: unknown = raw;
  if (typeof source === 'string' && source.trim().startsWith('[')) {
    try {
      source = JSON.parse(source);
    } catch {
      return [];
    }
  } else if (typeof source === 'string' && source.trim().startsWith('{')) {
    try {
      const obj = JSON.parse(source) as Record<string, unknown>;
      source = obj.exemplar_embeddings ?? obj.exemplarEmbeddings;
    } catch {
      return [];
    }
  } else if (source && typeof source === 'object' && !Array.isArray(source)) {
    const obj = source as Record<string, unknown>;
    source = obj.exemplar_embeddings ?? obj.exemplarEmbeddings;
  }
  if (!Array.isArray(source)) return [];
  const out: number[][] = [];
  for (const item of source) {
    const vec = parseEmbeddingVector(item);
    if (vec && vec.length >= 128) out.push(vec);
  }
  return out;
}

export function normalizeBiometricEngineTag(
  raw: unknown,
  vec?: readonly number[],
): 'human-faceres-1024d' | 'arcface-onnx' | 'hybrid-512d' | undefined {
  if (raw === 'human-faceres-1024d' || (vec && vec.length >= 768)) return 'human-faceres-1024d';
  if (raw === 'arcface-onnx' || raw === 'hybrid-512d') return raw;
  return undefined;
}

function normalizeAngleImageUrlsLoose(raw: unknown): PatientRecord['angleImageUrls'] {
  const rec = asRecord(raw);
  const out: Partial<Record<'center' | 'left' | 'right' | 'up' | 'down', string>> = {};
  for (const key of ['center', 'left', 'right', 'up', 'down'] as const) {
    const url = nonEmptyString(rec[key]);
    if (url) out[key] = url;
  }
  return Object.keys(out).length > 0 ? out : undefined;
}

function normalizeStrokeType(raw: unknown): StrokeType | undefined {
  const clean = nonEmptyString(raw)?.toUpperCase() as StrokeType | undefined;
  return clean && VALID_STROKE_TYPES.has(clean) ? clean : undefined;
}

function normalizeAffectedSide(raw: unknown): AffectedSide | undefined {
  const clean = nonEmptyString(raw)?.toUpperCase() as AffectedSide | undefined;
  return clean && VALID_AFFECTED_SIDES.has(clean) ? clean : undefined;
}

/**
 * Protects a verified patient record from being accidentally downgraded to `'PENDING'`
 * by a stale client form or partial update payload when valid embeddings already exist.
 */
export function resolveProtectedBiometricStatus(
  patient: Pick<
    PatientRecord,
    'faceBiometricStatus' | 'faceFramesCount' | 'masterEmbedding' | 'angleEmbeddings'
  >,
  incomingStatus?: 'PENDING' | 'VERIFIED',
): 'PENDING' | 'VERIFIED' {
  const hasValidEmbeddings = Boolean(
    (Array.isArray(patient.masterEmbedding) && patient.masterEmbedding.length >= 128) ||
      (patient.angleEmbeddings?.center && patient.angleEmbeddings.center.length >= 128),
  );
  const hasVerifiedFrames =
    hasValidEmbeddings && (typeof patient.faceFramesCount === 'number' ? patient.faceFramesCount > 0 : true);
  if (hasValidEmbeddings || hasVerifiedFrames) {
    return 'VERIFIED';
  }
  if (incomingStatus === 'VERIFIED' || patient.faceBiometricStatus === 'VERIFIED') {
    return 'VERIFIED';
  }
  return incomingStatus ?? patient.faceBiometricStatus ?? 'PENDING';
}

/**
 * Builds the canonical top-level `public.patients` column payload for dual-write consistency
 * with `patients.history` JSONB. Never emits a stale `'PENDING'` status when valid embeddings exist.
 */
export function buildPatientDualWriteColumns(
  patient: PatientRecord,
  options?: { updatedAtIso?: string; includeProfileFields?: boolean },
): Record<string, unknown> {
  const updatedAtIso = options?.updatedAtIso ?? patient.biometricUpdatedAt ?? new Date().toISOString();
  const hasValidMaster = Array.isArray(patient.masterEmbedding) && patient.masterEmbedding.length >= 128;
  const resolvedAngles =
    patient.angleEmbeddings ??
    (hasValidMaster && patient.masterEmbedding
      ? {
          center: patient.masterEmbedding,
          left: patient.masterEmbedding,
          right: patient.masterEmbedding,
          up: patient.masterEmbedding,
          down: patient.masterEmbedding,
        }
      : undefined);
  const effectiveStatus = resolveProtectedBiometricStatus(patient);
  const effectiveFrames =
    typeof patient.faceFramesCount === 'number' && patient.faceFramesCount > 0
      ? Math.round(patient.faceFramesCount)
      : hasValidMaster || effectiveStatus === 'VERIFIED'
        ? 120
        : 0;
  const effectiveEngine =
    normalizeBiometricEngineTag(patient.biometricEngine, patient.masterEmbedding) ??
    (hasValidMaster
      ? (patient.masterEmbedding!.length >= 768 ? 'human-faceres-1024d' : 'hybrid-512d')
      : undefined);
  const compactExemplars = (patient.exemplarEmbeddings ?? [])
    .filter((v) => Array.isArray(v) && v.length >= 128)
    .slice(0, 9);

  const biometricBlob =
    hasValidMaster && patient.masterEmbedding
      ? {
          engine: effectiveEngine ?? 'human-faceres-1024d',
          dimension: patient.masterEmbedding.length,
          master_embedding: patient.masterEmbedding,
          angle_embeddings: resolvedAngles ?? null,
          exemplar_embeddings: compactExemplars,
          ...(patient.geometricRatios ? { geometric_ratios: patient.geometricRatios } : {}),
          frames_captured: effectiveFrames,
          updated_at: updatedAtIso,
        }
      : null;

  const cols: Record<string, unknown> = {
    face_biometric_status: effectiveStatus,
    face_frames_count: effectiveFrames,
    ...(patient.faceImageUrl ? { face_image_url: patient.faceImageUrl } : {}),
    ...(hasValidMaster && patient.masterEmbedding
      ? {
          master_embedding: patient.masterEmbedding,
          angle_embeddings: resolvedAngles ?? null,
          face_embedding: biometricBlob,
          biometric_data: biometricBlob,
          biometric_updated_at: updatedAtIso,
        }
      : {}),
  };

  if (options?.includeProfileFields !== false) {
    if (nonEmptyString(patient.gender)) cols.gender = patient.gender!.trim();
    if (normalizeStrokeType(patient.strokeType)) cols.stroke_type = normalizeStrokeType(patient.strokeType);
    if (normalizeAffectedSide(patient.affectedSide)) cols.affected_side = normalizeAffectedSide(patient.affectedSide);
    if (finiteNumberInRange(patient.motorAssessmentScore, 0, 6) !== undefined) {
      cols.motor_assessment_score = finiteNumberInRange(patient.motorAssessmentScore, 0, 6);
    }
    if (finiteNumberInRange(patient.adlDressingScore, 0, 5) !== undefined) {
      cols.adl_dressing_score = finiteNumberInRange(patient.adlDressingScore, 0, 5);
    }
    if (finiteNumberInRange(patient.maxSafeRomDeg, 30, 180) !== undefined) {
      cols.max_safe_rom_deg = finiteNumberInRange(patient.maxSafeRomDeg, 30, 180);
    }
    if (nonEmptyString(patient.universalHealthLinkCode)) {
      cols.universal_health_link_code = patient.universalHealthLinkCode!.trim().toUpperCase();
    }
  }

  return cols;
}

/**
 * Hydrates a patient record from raw Supabase snake_case columns, `history` JSONB, and optional
 * `patient_biometrics` row — AND detects whether top-level DB columns are stale/missing so they
 * can be repaired in the background without blocking login or UI rendering.
 */
export function inspectPatientRecordHydration(
  rawInput: Record<string, unknown> | PatientRecord,
  options?: NormalizePatientOptions,
): PatientHydrationInspection {
  const r = rawInput as Record<string, unknown>;
  const history = asRecord(r.history);
  const b120 = options?.biometricsRow ?? null;
  const prev = options?.prevPatient ?? null;
  const bioDataObj = asRecord(r.biometric_data);
  const faceBlobObj = asRecord(r.face_embedding);
  const requireRealVectors = options?.requireRealVectorsForVerified ?? true;
  const strictCurrentDimOnly = options?.strictCurrentDimOnly ?? false;
  const applyClinicalDefaults = options?.applyClinicalDefaults ?? false;

  const pid = String(r.id ?? prev?.id ?? '').trim();
  const hn = String(r.hn ?? prev?.hn ?? '').trim();
  const fullName = String(r.full_name ?? r.fullName ?? prev?.fullName ?? '').trim();

  const topAngles =
    parseAngleEmbeddingsLoose(r.angle_embeddings) ??
    parseAngleEmbeddingsLoose(r.angleEmbeddings) ??
    parseAngleEmbeddingsLoose(r.biometric_data) ??
    parseAngleEmbeddingsLoose(r.face_embedding);
  const histAngles = parseAngleEmbeddingsLoose(history.angleEmbeddings);
  const topMaster =
    parseEmbeddingVector(r.master_embedding) ??
    parseEmbeddingVector(r.masterEmbedding) ??
    parseEmbeddingVector(r.biometric_data) ??
    parseEmbeddingVector(r.face_embedding);
  const histMaster = parseEmbeddingVector(history.masterEmbedding);

  const mergedMaster =
    b120?.masterEmbedding ??
    topMaster ??
    histMaster ??
    b120?.angleEmbeddings?.center ??
    topAngles?.center ??
    histAngles?.center ??
    prev?.masterEmbedding;

  const mergedAngles: AngleEmbeddings5D | undefined =
    b120?.angleEmbeddings ??
    topAngles ??
    histAngles ??
    (mergedMaster
      ? {
          center: mergedMaster,
          left: mergedMaster,
          right: mergedMaster,
          up: mergedMaster,
          down: mergedMaster,
        }
      : prev?.angleEmbeddings);

  const rawExemplars = (() => {
    const fromB120 = b120?.exemplarEmbeddings ?? [];
    const fromTop = parseExemplarEmbeddingsLoose(
      r.exemplar_embeddings ?? r.exemplarEmbeddings ?? r.biometric_data ?? r.face_embedding,
    );
    const fromHist = parseExemplarEmbeddingsLoose(history.exemplarEmbeddings);
    const chosen = fromB120.length > 0 ? fromB120 : fromTop.length > 0 ? fromTop : fromHist;
    if (!mergedMaster || chosen.length === 0) return prev?.exemplarEmbeddings;
    const matchedDim = chosen.filter((v) => v.length === mergedMaster.length);
    return matchedDim.length > 0 ? matchedDim : prev?.exemplarEmbeddings;
  })();

  const hydratedGeomRatios: GeometricAnthropometricRatios | undefined = (() => {
    const candidates = [
      b120?.geometricRatios,
      r.geometricRatios,
      r.geometric_ratios,
      bioDataObj.geometric_ratios,
      faceBlobObj.geometric_ratios,
      history.geometricRatios,
      prev?.geometricRatios,
    ];
    for (const c of candidates) {
      if (c && typeof c === 'object' && !Array.isArray(c)) {
        return c as GeometricAnthropometricRatios;
      }
    }
    return undefined;
  })();

  const hydratedAngleImageUrls =
    normalizeAngleImageUrlsLoose(r.angleImageUrls) ??
    normalizeAngleImageUrlsLoose(r.angle_image_urls) ??
    normalizeAngleImageUrlsLoose(history.angleImageUrls) ??
    prev?.angleImageUrls;

  const hydratedFaceImageUrls: string[] | undefined = (() => {
    const rawList = Array.isArray(r.faceImageUrls)
      ? r.faceImageUrls
      : Array.isArray(history.faceImageUrls)
        ? history.faceImageUrls
        : [];
    const list = rawList.filter((u): u is string => typeof u === 'string' && u.trim().length > 0);
    const angleList = hydratedAngleImageUrls
      ? (['center', 'left', 'right', 'up', 'down'] as const)
          .map((k) => hydratedAngleImageUrls[k])
          .filter((u): u is string => typeof u === 'string' && u.length > 0)
      : [];
    const combined = Array.from(new Set([...list, ...angleList]));
    return combined.length > 0 ? combined : prev?.faceImageUrls;
  })();

  const topFaceImageUrl = nonEmptyString(r.face_image_url) ?? nonEmptyString(r.faceImageUrl);
  const histFaceImageUrl =
    nonEmptyString(r.historyFaceImageUrl) ??
    nonEmptyString(history.faceImageUrl) ??
    hydratedFaceImageUrls?.[0];
  const mergedFaceImageUrl = b120?.avatarUrl ?? topFaceImageUrl ?? histFaceImageUrl ?? prev?.faceImageUrl;

  const topStatusUpper = nonEmptyString(r.face_biometric_status ?? r.faceBiometricStatus)?.toUpperCase();
  const histStatusUpper = nonEmptyString(r.historyFaceStatus ?? history.faceBiometricStatus)?.toUpperCase();
  const topFramesCount = positiveInt(r.face_frames_count ?? r.faceFramesCount);
  const histFramesCount = positiveInt(r.historyFaceFramesCount ?? history.faceFramesCount);
  const hasRealVectors = Boolean(mergedMaster && mergedMaster.length >= 128);
  const embeddingDim = mergedMaster?.length ?? 0;

  const mergedEngine =
    normalizeBiometricEngineTag(b120?.biometricEngine, mergedMaster) ??
    normalizeBiometricEngineTag(r.biometricEngine ?? r.biometric_engine, mergedMaster) ??
    normalizeBiometricEngineTag(bioDataObj.engine ?? faceBlobObj.engine, mergedMaster) ??
    normalizeBiometricEngineTag(history.biometricEngine, mergedMaster) ??
    normalizeBiometricEngineTag(prev?.biometricEngine, mergedMaster);

  const isCompatibleCurrentDim =
    hasRealVectors &&
    embeddingDim >= ACTIVE_BIOMETRIC_DIM &&
    (!mergedEngine || mergedEngine === 'human-faceres-1024d');

  const biometricStatusDetail: 'VERIFIED' | 'PENDING' | 'REENROLL_REQUIRED' = !hasRealVectors
    ? 'PENDING'
    : isCompatibleCurrentDim
      ? 'VERIFIED'
      : 'REENROLL_REQUIRED';

  const biometricPipelineVersion = !hasRealVectors
    ? undefined
    : embeddingDim >= ACTIVE_BIOMETRIC_DIM
      ? hydratedGeomRatios?.ocularIrisPatch256?.length === 256
        ? ACTIVE_BIOMETRIC_PIPELINE_VERSION
        : 'v2-pre-iris-1024d'
      : 'legacy-512d';

  const hasHistoryVerifiedEvidence =
    histStatusUpper === 'VERIFIED' ||
    ((histFramesCount ?? 0) > 0 && hasRealVectors) ||
    hasRealVectors;

  let resolvedFaceStatus: 'PENDING' | 'VERIFIED';
  if (hasRealVectors) {
    resolvedFaceStatus = strictCurrentDimOnly
      ? isCompatibleCurrentDim
        ? 'VERIFIED'
        : 'PENDING'
      : 'VERIFIED';
  } else if (!requireRealVectors && (topStatusUpper === 'VERIFIED' || histStatusUpper === 'VERIFIED')) {
    resolvedFaceStatus = 'VERIFIED';
  } else {
    resolvedFaceStatus = 'PENDING';
  }

  const resolvedFramesCount =
    b120?.framesCaptured && b120.framesCaptured > 0
      ? Math.round(b120.framesCaptured)
      : topFramesCount ??
        histFramesCount ??
        (hasRealVectors || resolvedFaceStatus === 'VERIFIED'
          ? Math.max(prev?.faceFramesCount ?? 0, 120)
          : prev?.faceFramesCount ?? 0);

  const biometricUpdatedAt =
    nonEmptyString(r.biometric_updated_at) ??
    nonEmptyString(r.biometricUpdatedAt) ??
    nonEmptyString(b120?.updatedAt) ??
    nonEmptyString(bioDataObj.updated_at) ??
    nonEmptyString(history.biometricUpdatedAt) ??
    (hasRealVectors || resolvedFaceStatus === 'VERIFIED'
      ? nonEmptyString(r.updated_at) ?? nonEmptyString(r.created_at) ?? prev?.biometricUpdatedAt
      : prev?.biometricUpdatedAt);

  const hydratedGender =
    nonEmptyString(r.gender) ??
    nonEmptyString(r.historyGender) ??
    nonEmptyString(history.gender) ??
    prev?.gender ??
    (applyClinicalDefaults ? 'ชาย' : undefined);

  const hydratedStrokeType =
    normalizeStrokeType(r.stroke_type) ??
    normalizeStrokeType(r.strokeType) ??
    normalizeStrokeType(history.strokeType) ??
    prev?.strokeType ??
    (applyClinicalDefaults ? 'ELDERLY_GENERAL' : undefined);

  const hydratedAffectedSide =
    normalizeAffectedSide(r.affected_side) ??
    normalizeAffectedSide(r.affectedSide) ??
    normalizeAffectedSide(history.affectedSide) ??
    prev?.affectedSide ??
    (applyClinicalDefaults ? 'NONE' : undefined);

  const hydratedMotorScore =
    finiteNumberInRange(r.motor_assessment_score, 0, 6) ??
    finiteNumberInRange(r.motorAssessmentScore, 0, 6) ??
    finiteNumberInRange(history.motorAssessmentScore, 0, 6) ??
    prev?.motorAssessmentScore ??
    (applyClinicalDefaults ? 3 : undefined);

  const hydratedAdlScore =
    finiteNumberInRange(r.adl_dressing_score, 0, 5) ??
    finiteNumberInRange(r.adlDressingScore, 0, 5) ??
    finiteNumberInRange(history.adlDressingScore, 0, 5) ??
    prev?.adlDressingScore ??
    (applyClinicalDefaults ? 3 : undefined);

  const hydratedMaxSafeRom =
    finiteNumberInRange(r.max_safe_rom_deg, 30, 180) ??
    finiteNumberInRange(r.maxSafeRomDeg, 30, 180) ??
    finiteNumberInRange(history.maxSafeRomDeg, 30, 180) ??
    prev?.maxSafeRomDeg ??
    (applyClinicalDefaults ? 150 : undefined);

  const hydratedLinkCode =
    nonEmptyString(r.universal_health_link_code) ??
    nonEmptyString(r.universalHealthLinkCode) ??
    nonEmptyString(history.universalHealthLinkCode) ??
    prev?.universalHealthLinkCode ??
    (applyClinicalDefaults && pid
      ? `DTC-EHR-${pid.replace(/[^a-zA-Z0-9]/g, '').slice(0, 8).toUpperCase()}`
      : undefined);

  const regStatusRaw = String(r.registration_status ?? r.registrationStatus ?? prev?.registrationStatus ?? 'active')
    .trim()
    .toLowerCase();
  const registrationStatus: 'active' | 'archived' = regStatusRaw === 'archived' ? 'archived' : 'active';

  const patient: PatientRecord = {
    ...(prev ?? {
      id: pid,
      hn,
      fullName,
      age: Number(r.age ?? 0),
      phone: String(r.phone ?? ''),
      diagnosis: String(r.diagnosis ?? ''),
      assignedPtId: String(r.assigned_pt_id ?? r.assignedPtId ?? ''),
      delegatedToPtId: null,
      delegationNote: null,
      nextAppointmentDate: String(r.next_appointment_date ?? r.nextAppointmentDate ?? ''),
      registrationStatus,
      enrolledAt: String(r.created_at ?? r.enrolledAt ?? new Date().toISOString()),
    }),
    id: pid,
    hn,
    fullName,
    age: typeof r.age === 'number' && Number.isFinite(r.age) ? r.age : prev?.age ?? 0,
    ...(hydratedGender ? { gender: hydratedGender } : {}),
    phone: String(r.phone ?? prev?.phone ?? ''),
    diagnosis: String(r.diagnosis ?? prev?.diagnosis ?? ''),
    ...(hydratedStrokeType ? { strokeType: hydratedStrokeType } : {}),
    ...(hydratedAffectedSide ? { affectedSide: hydratedAffectedSide } : {}),
    ...(hydratedMotorScore !== undefined ? { motorAssessmentScore: hydratedMotorScore } : {}),
    ...(hydratedAdlScore !== undefined ? { adlDressingScore: hydratedAdlScore } : {}),
    ...(hydratedMaxSafeRom !== undefined ? { maxSafeRomDeg: hydratedMaxSafeRom } : {}),
    ...(hydratedLinkCode ? { universalHealthLinkCode: hydratedLinkCode } : {}),
    ...(Array.isArray(r.rehabSessionHistory)
      ? { rehabSessionHistory: r.rehabSessionHistory as RehabSessionHistoryEntry[] }
      : Array.isArray(history.rehabSessionHistory)
        ? { rehabSessionHistory: history.rehabSessionHistory as RehabSessionHistoryEntry[] }
        : {}),
    ...(Array.isArray(r.clinicalAssessments)
      ? { clinicalAssessments: r.clinicalAssessments as ClinicalAssessmentRecord[] }
      : Array.isArray(history.clinicalAssessments)
        ? { clinicalAssessments: history.clinicalAssessments as ClinicalAssessmentRecord[] }
        : {}),
    assignedPtId: String(r.assigned_pt_id ?? r.assignedPtId ?? prev?.assignedPtId ?? ''),
    delegatedToPtId:
      r.delegated_to_pt_id !== undefined
        ? r.delegated_to_pt_id
          ? String(r.delegated_to_pt_id)
          : null
        : r.delegatedToPtId !== undefined
          ? r.delegatedToPtId
            ? String(r.delegatedToPtId)
            : null
          : prev?.delegatedToPtId ?? null,
    delegationNote:
      r.delegation_note !== undefined
        ? r.delegation_note
          ? String(r.delegation_note)
          : null
        : r.delegationNote !== undefined
          ? r.delegationNote
            ? String(r.delegationNote)
            : null
          : prev?.delegationNote ?? null,
    nextAppointmentDate: String(
      r.next_appointment_date ?? r.nextAppointmentDate ?? prev?.nextAppointmentDate ?? '',
    ),
    registrationStatus,
    enrolledAt: String(r.created_at ?? r.enrolledAt ?? prev?.enrolledAt ?? new Date().toISOString()),
    faceBiometricStatus: resolvedFaceStatus,
    biometricStatusDetail,
    biometricEmbeddingDim: embeddingDim,
    ...(biometricPipelineVersion ? { biometricPipelineVersion } : {}),
    ...(biometricUpdatedAt ? { biometricUpdatedAt } : {}),
    faceFramesCount: hasRealVectors || resolvedFaceStatus === 'VERIFIED' ? resolvedFramesCount : 0,
    ...(mergedFaceImageUrl ? { faceImageUrl: mergedFaceImageUrl } : {}),
    ...(hydratedFaceImageUrls ? { faceImageUrls: hydratedFaceImageUrls } : {}),
    ...(hydratedAngleImageUrls ? { angleImageUrls: hydratedAngleImageUrls } : {}),
    ...(mergedMaster ? { masterEmbedding: mergedMaster } : {}),
    ...(mergedAngles ? { angleEmbeddings: mergedAngles } : {}),
    ...(rawExemplars ? { exemplarEmbeddings: rawExemplars } : {}),
    ...(mergedEngine ? { biometricEngine: mergedEngine } : {}),
    ...(hydratedGeomRatios ? { geometricRatios: hydratedGeomRatios } : {}),
    ...(Array.isArray(r.rehabSchedules)
      ? { rehabSchedules: r.rehabSchedules as RehabScheduleItem[] }
      : Array.isArray(history.rehabSchedules)
        ? { rehabSchedules: history.rehabSchedules as RehabScheduleItem[] }
        : {}),
  };

  // Detect stale/missing top-level DB columns compared to hydrated history/biometrics data
  const repairPatch: Record<string, unknown> = {};
  const isRawDbRow =
    Object.prototype.hasOwnProperty.call(r, 'face_biometric_status') ||
    Object.prototype.hasOwnProperty.call(r, 'face_frames_count') ||
    Object.prototype.hasOwnProperty.call(r, 'history');

  if (isRawDbRow) {
    const dualCols = buildPatientDualWriteColumns(patient, {
      updatedAtIso: biometricUpdatedAt ?? nonEmptyString(r.updated_at) ?? new Date().toISOString(),
      includeProfileFields: false,
    });

    if (
      resolvedFaceStatus === 'VERIFIED' &&
      hasHistoryVerifiedEvidence &&
      topStatusUpper !== 'VERIFIED'
    ) {
      repairPatch.face_biometric_status = 'VERIFIED';
    }
    if (
      (resolvedFramesCount > 0 && (hasRealVectors || resolvedFaceStatus === 'VERIFIED')) &&
      (topFramesCount ?? 0) === 0
    ) {
      repairPatch.face_frames_count = resolvedFramesCount;
    }
    if (!topFaceImageUrl && mergedFaceImageUrl) {
      repairPatch.face_image_url = mergedFaceImageUrl;
    }
    if (hasRealVectors && mergedMaster) {
      const topMetadataStale =
        topStatusUpper !== 'VERIFIED' ||
        (topFramesCount ?? 0) === 0 ||
        !nonEmptyString(r.biometric_updated_at) ||
        (!topFaceImageUrl && Boolean(mergedFaceImageUrl));
      const hasMasterKey = Object.prototype.hasOwnProperty.call(r, 'master_embedding');
      const hasAnglesKey = Object.prototype.hasOwnProperty.call(r, 'angle_embeddings');
      const hasBioDataKey = Object.prototype.hasOwnProperty.call(r, 'biometric_data');
      const hasFaceEmbKey = Object.prototype.hasOwnProperty.call(r, 'face_embedding');

      const topMasterMissing = hasMasterKey ? !parseEmbeddingVector(r.master_embedding) : topMetadataStale;
      const topAnglesMissing = hasAnglesKey ? !parseAngleEmbeddingsLoose(r.angle_embeddings) : topMetadataStale;
      const topBioDataMissing = hasBioDataKey ? !parseEmbeddingVector(r.biometric_data) : topMetadataStale;
      const topFaceEmbMissing = hasFaceEmbKey ? !parseEmbeddingVector(r.face_embedding) : topMetadataStale;
      const topUpdatedAtMissing = !nonEmptyString(r.biometric_updated_at);
      if (topMasterMissing && dualCols.master_embedding) {
        repairPatch.master_embedding = dualCols.master_embedding;
      }
      if (topAnglesMissing && dualCols.angle_embeddings) {
        repairPatch.angle_embeddings = dualCols.angle_embeddings;
      }
      if (topBioDataMissing && dualCols.biometric_data) {
        repairPatch.biometric_data = dualCols.biometric_data;
      }
      if (topFaceEmbMissing && dualCols.face_embedding) {
        repairPatch.face_embedding = dualCols.face_embedding;
      }
      if (topUpdatedAtMissing && dualCols.biometric_updated_at) {
        repairPatch.biometric_updated_at = dualCols.biometric_updated_at;
      }
    }
    if (Object.prototype.hasOwnProperty.call(r, 'gender') && !nonEmptyString(r.gender) && nonEmptyString(history.gender)) {
      repairPatch.gender = nonEmptyString(history.gender);
    } else if (!Object.prototype.hasOwnProperty.call(r, 'gender') && nonEmptyString(history.gender) && Object.keys(repairPatch).length > 0) {
      repairPatch.gender = nonEmptyString(history.gender);
    }
    if (
      (!nonEmptyString(r.stroke_type) && normalizeStrokeType(history.strokeType)) &&
      (Object.prototype.hasOwnProperty.call(r, 'stroke_type') || Object.keys(repairPatch).length > 0)
    ) {
      repairPatch.stroke_type = normalizeStrokeType(history.strokeType);
    }
    if (
      (!nonEmptyString(r.affected_side) && normalizeAffectedSide(history.affectedSide)) &&
      (Object.prototype.hasOwnProperty.call(r, 'affected_side') || Object.keys(repairPatch).length > 0)
    ) {
      repairPatch.affected_side = normalizeAffectedSide(history.affectedSide);
    }
    if (
      finiteNumberInRange(r.motor_assessment_score, 0, 6) === undefined &&
      finiteNumberInRange(history.motorAssessmentScore, 0, 6) !== undefined &&
      (Object.prototype.hasOwnProperty.call(r, 'motor_assessment_score') || Object.keys(repairPatch).length > 0)
    ) {
      repairPatch.motor_assessment_score = finiteNumberInRange(history.motorAssessmentScore, 0, 6);
    }
    if (
      finiteNumberInRange(r.adl_dressing_score, 0, 5) === undefined &&
      finiteNumberInRange(history.adlDressingScore, 0, 5) !== undefined &&
      (Object.prototype.hasOwnProperty.call(r, 'adl_dressing_score') || Object.keys(repairPatch).length > 0)
    ) {
      repairPatch.adl_dressing_score = finiteNumberInRange(history.adlDressingScore, 0, 5);
    }
  }

  const needsTopLevelRepair = Object.keys(repairPatch).length > 0;
  return {
    patient,
    hasRealVectors,
    needsTopLevelRepair,
    topLevelRepairPatch: needsTopLevelRepair ? repairPatch : null,
  };
}

/**
 * Canonical patient record normalizer used across server hydration, API client, and Kiosk/Portal UI.
 * Automatically hydrates `VERIFIED` biometric status, frames count, face image URL, embeddings,
 * and clinical profile fields from `history` JSONB when top-level columns are empty or `PENDING`.
 */
export function normalizePatientRecord(
  rawInput: Record<string, unknown> | PatientRecord,
  options?: NormalizePatientOptions,
): PatientRecord {
  return inspectPatientRecordHydration(rawInput, options).patient;
}

const repairedPatientsByClient = new WeakMap<object, Set<string>>();

export function resetBackgroundPatientRepairTrackerForTests(): void {
  // WeakMap entries are scoped per SupabaseClient instance; tests create fresh mocks per test.
}

/**
 * Persists repaired top-level `public.patients` columns in the background without blocking reads.
 * Deduplicated per `(supabaseClient, patientId)` and strips any unmigrated columns on PGRST204/42703.
 */
export function schedulePatientTopLevelRepairInBackground(
  supabase: SupabaseClient | null | undefined,
  patientId: string,
  patch: Record<string, unknown> | null | undefined,
): Promise<boolean> {
  if (!supabase || !patientId || !patch || Object.keys(patch).length === 0) {
    return Promise.resolve(false);
  }
  if (isSupabaseCircuitOpen(supabase)) {
    return Promise.resolve(false);
  }
  let seen = repairedPatientsByClient.get(supabase);
  if (!seen) {
    seen = new Set<string>();
    repairedPatientsByClient.set(supabase, seen);
  }
  if (seen.has(patientId)) {
    return Promise.resolve(false);
  }
  seen.add(patientId);

  const runRepair = async (): Promise<boolean> => {
    try {
      const tableBuilder = supabase.from('patients') as unknown as {
        update?: (payload: Record<string, unknown>) => {
          eq?: (col: string, val: string) => PromiseLike<{ error?: { code?: string; message?: string } | null }>;
        };
      };
      if (!tableBuilder || typeof tableBuilder.update !== 'function') {
        return false;
      }
      const working = stripUnsupportedColumnsForWrite(
        'patients',
        stripOversizedBase64ForDb({ ...patch }),
      );
      if (Object.keys(working).length === 0) return false;
      for (let attempt = 0; attempt < 10; attempt += 1) {
        if (Object.keys(working).length === 0) return false;
        const query = tableBuilder.update(working);
        if (!query || typeof query.eq !== 'function') return false;
        const res = await query.eq('id', patientId);
        if (!res?.error) return true;
        const missingCol = recordUnsupportedColumn('patients', res.error);
        if (missingCol && missingCol in working) {
          delete working[missingCol];
          continue;
        }
        return false;
      }
      return false;
    } catch {
      return false;
    }
  };

  return runRepair();
}
