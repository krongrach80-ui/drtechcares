import { createClient, type SupabaseClient } from '@supabase/supabase-js';
import {
  CLINICAL_EXERCISES,
  evaluateClinicalBiomechanicsPacket,
} from '@drtechcares/pose-engine';
import {
  COMMON_MODE_ALPHA,
  FACE_EMBEDDING_DIM,
  CANONICAL_HUMAN_FACE_BASELINE_512,
  classifyLivePoseZone,
  computeArcFaceBiometricScore,
  computeDualFactorBiometricScore,
  computeFaceCoreMetrics,
  cosineSimilarity,
  detectBiometricVectorEngine,
  isLegacyV1HybridVector,
  l2Normalize,
  mirrorPoseZone,
  parseEmbeddingVector,
  verifyHybridBiometricIdentity,
  whitenEmbedding,
  type BiometricPose,
  type EnrolledTemplateCandidate,
  type FacePoseZone5,
  type GeometricAnthropometricRatios,
} from '@drtechcares/smart-biometric-core';
import {
  inspectBiometricFrame3D,
  type Detected3DPose,
} from '../../face-client/src/biometricAssistant';

type BiomechanicsEvaluationPacket = Parameters<typeof evaluateClinicalBiomechanicsPacket>[0];
type RepCounterState = NonNullable<Parameters<typeof evaluateClinicalBiomechanicsPacket>[1]>;
type FrameInspectionInput = Parameters<typeof inspectBiometricFrame3D>[0];
import {
  formatLocalIsoDate,
  resolveNextAppointmentDate,
  resolveTodayPatientSchedule,
  type AffectedSide,
  type AngleEmbeddings5D,
  type AuditLogRecord,
  type BannedTargetRecord,
  type CloudClientConfig,
  type DailyScheduleDay,
  type ManagedExerciseRecord,
  type MedicalNoteRecord,
  type ModpackRecord,
  type PatientBiometricIndexEntry,
  type PatientCourseRecord,
  type PatientRecord,
  type RehabScheduleItem,
  type RehabSessionHistoryEntry,
  type RehabSessionRecord,
  type StaffCallRecord,
  type StrokeType,
  type SystemSettingsRecord,
  type TreatmentReportRecord,
  type UniversalEhrPackage,
  type UserAccount,
  type UserRole,
} from './index';
import {
  hydrateStateFromSupabase,
  newRecordId,
  normalizeStaffRole,
  persistToSupabase,
  type CloudRouterState,
  type StoredUserAccount,
} from './supabasePersistence';
import { clampTtsText, synthesizeTtsAudio } from './ttsProxy';

function randomHex(byteLength: number): string {
  const bytes = new Uint8Array(byteLength);
  globalThis.crypto.getRandomValues(bytes);
  return Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('');
}

function timingSafeEqualStr(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i += 1) {
    diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  }
  return diff === 0;
}

function bytesToBase64Url(bytes: Uint8Array): string {
  let binary = '';
  for (let i = 0; i < bytes.length; i += 1) {
    binary += String.fromCharCode(bytes[i]);
  }
  return globalThis.btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/g, '');
}

function base64UrlToBytes(b64u: string): Uint8Array | null {
  try {
    const padded = b64u.replace(/-/g, '+').replace(/_/g, '/') + '='.repeat((4 - (b64u.length % 4)) % 4);
    const binary = globalThis.atob(padded);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i += 1) {
      bytes[i] = binary.charCodeAt(i);
    }
    return bytes;
  } catch {
    return null;
  }
}

interface SignedStaffTokenPayload {
  uid: string;
  usr: string;
  role: UserRole;
  iat: number;
  rnd: string;
}

async function signPayloadEdge(payloadB64: string, secret: string): Promise<string> {
  const data = new TextEncoder().encode(`${payloadB64}.${secret}`);
  const digest = await globalThis.crypto.subtle.digest('SHA-256', data);
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, '0')).join('');
}

async function createStaffToken(user: StoredUserAccount, secret: string): Promise<string> {
  const payload: SignedStaffTokenPayload = {
    uid: user.id,
    usr: user.username,
    role: user.role,
    iat: Date.now(),
    rnd: randomHex(8),
  };
  const payloadB64 = bytesToBase64Url(new TextEncoder().encode(JSON.stringify(payload)));
  const sig = await signPayloadEdge(payloadB64, secret);
  return `stf.${payloadB64}.${sig}`;
}

async function verifyStaffToken(token: string, secret: string): Promise<SignedStaffTokenPayload | null> {
  if (!token.startsWith('stf.')) return null;
  const parts = token.split('.');
  if (parts.length !== 3) return null;
  const [, payloadB64, sigHex] = parts;
  const expected = await signPayloadEdge(payloadB64, secret);
  if (!timingSafeEqualStr(sigHex, expected)) return null;
  const bytes = base64UrlToBytes(payloadB64);
  if (!bytes) return null;
  try {
    const parsed = JSON.parse(new TextDecoder().decode(bytes)) as Partial<SignedStaffTokenPayload>;
    if (!parsed.uid || !parsed.usr || !parsed.role || typeof parsed.iat !== 'number') return null;
    if (Date.now() - parsed.iat > 7 * 24 * 60 * 60 * 1000) return null;
    return parsed as SignedStaffTokenPayload;
  } catch {
    return null;
  }
}

async function hashPasswordEdge(password: string): Promise<string> {
  const salt = randomHex(16);
  const enc = new TextEncoder();
  const keyMaterial = await globalThis.crypto.subtle.importKey(
    'raw',
    enc.encode(password),
    'PBKDF2',
    false,
    ['deriveBits'],
  );
  const bits = await globalThis.crypto.subtle.deriveBits(
    {
      name: 'PBKDF2',
      salt: enc.encode(salt),
      iterations: 10_000,
      hash: 'SHA-256',
    },
    keyMaterial,
    256,
  );
  const hex = Array.from(new Uint8Array(bits), (b) => b.toString(16).padStart(2, '0')).join('');
  return `pbkdf2:${salt}:${hex}`;
}

async function verifyPasswordEdge(password: string, storedHash: string): Promise<boolean> {
  if (storedHash.startsWith('pbkdf2:')) {
    const parts = storedHash.split(':');
    if (parts.length !== 3) return false;
    const [, salt, expectedHex] = parts;
    const enc = new TextEncoder();
    const keyMaterial = await globalThis.crypto.subtle.importKey(
      'raw',
      enc.encode(password),
      'PBKDF2',
      false,
      ['deriveBits'],
    );
    const bits = await globalThis.crypto.subtle.deriveBits(
      {
        name: 'PBKDF2',
        salt: enc.encode(salt),
        iterations: 10_000,
        hash: 'SHA-256',
      },
      keyMaterial,
      256,
    );
    const actualHex = Array.from(new Uint8Array(bits), (b) => b.toString(16).padStart(2, '0')).join('');
    return timingSafeEqualStr(actualHex, expectedHex);
  }
  if (storedHash.startsWith('plain:')) {
    return timingSafeEqualStr(password, storedHash.slice(6));
  }
  return false;
}

function defaultWeeklySchedule(): DailyScheduleDay[] {
  return [0, 1, 2, 3, 4, 5, 6].map((dayIndex) => ({
    dayIndex,
    dayType: 'rest',
    activityType: 'minigame',
    difficulty: 'easy',
    label: 'พักฟื้น',
    exercises: [],
    cognitiveModpackId: 'modpack-core-99',
  }));
}

function createInitialState(useUuidIds = false): CloudRouterState {
  const now = new Date().toISOString();
  return {
    users: [
      {
        id: useUuidIds ? '00000000-0000-4000-8000-000000000001' : 'user-super-1',
        username: 'admin',
        passwordHash: 'plain:Admin@2026',
        role: 'super_admin',
        fullName: 'นพ.สมชาย ผู้อำนวยการโรงพยาบาล (Super Admin)',
        phone: '02-000-0001',
        specialty: 'Hospital Director',
        licenseNo: 'MD-10001',
        active: true,
        forceLoggedOutAt: null,
        createdAt: now,
      },
      {
        id: useUuidIds ? '00000000-0000-4000-8000-000000000002' : 'user-pt-1',
        username: 'pt_somchai',
        passwordHash: 'plain:Pt@2026',
        role: 'pt',
        fullName: 'กภ.สมชาย ใจดี (PT)',
        phone: '081-111-2233',
        specialty: 'Orthopedic & Shoulder Rehab',
        licenseNo: 'PT-44012',
        active: true,
        forceLoggedOutAt: null,
        createdAt: now,
      },
      {
        id: useUuidIds ? '00000000-0000-4000-8000-000000000003' : 'user-pt-2',
        username: 'pt_waranya',
        passwordHash: 'plain:Pt@2026',
        role: 'pt',
        fullName: 'กภ.วรัญญา สุขสันต์ (PT)',
        phone: '089-222-3344',
        specialty: 'Neuro & Cognitive Rehab',
        licenseNo: 'PT-44089',
        active: true,
        forceLoggedOutAt: null,
        createdAt: now,
      },
    ],
    deleted_user_ids: [],
    deleted_patient_ids: [],
    deleted_ban_ids: [],
    deleted_staff_call_ids: [],
    delegation_overrides: {},
    patients: [],
    face_templates: [],
    medical_notes: [],
    patient_courses: [],
    rehab_sessions: [],
    rehab_session_history: [],
    treatment_reports: [],
    exercises: CLINICAL_EXERCISES.map((ex) => ({ ...ex, createdBy: 'system' })),
    modpacks: [
      {
        id: 'modpack-core-99',
        name: 'Clinical Cognitive Core (99 Cards)',
        categories: [
          'food_fruit',
          'objects',
          'animals',
          'counting',
          'safety_routine',
          'color_shape_time',
          'position',
        ],
        questionCount: 99,
        createdBy: 'system',
        active: true,
      },
      {
        id: 'modpack-memory-safety',
        name: 'Fall Prevention & Daily Safety Pack',
        categories: ['safety_routine', 'counting', 'position'],
        questionCount: 42,
        createdBy: 'system',
        active: true,
      },
    ],
    staff_calls: [],
    audit_logs: [
      {
        id: newRecordId(useUuidIds, 'audit'),
        eventType: 'SYSTEM_READY',
        actorRole: 'system',
        actorId: 'drtechcares-edge',
        actorName: 'Dr.Tech.Care Clinical Engine',
        ipAddress: '127.0.0.1',
        deviceId: 'cloud-worker',
        detail: 'Dr.Tech.Care Smart Rehab clinical security and audit trail initialized',
        createdAt: now,
      },
    ],
    banned_targets: [],
    active_sessions: [],
    system_settings: {
      aiFaceScanEnabled: true,
      aiFaceThreshold: 0.925,
      aiPoseDetectionEnabled: false,
      kioskAutoResetSec: 6,
    },
    kiosks: [
      {
        kioskId: 'KIOSK-OPD-01',
        location: 'OPD Rehab Zone A',
        online: true,
        currentState: 'STANDBY',
        lastHeartbeatAt: now,
      },
    ],
  };
}

function isVector512(value: unknown): value is number[] {
  return (
    Array.isArray(value) &&
    value.length === FACE_EMBEDDING_DIM &&
    value.every((n) => typeof n === 'number' && Number.isFinite(n))
  );
}

function buildAngleEmbeddingsFromPayload(
  rawAngles: unknown,
  fallbackFrontal: number[],
  fallbackOverall: number[],
  rawFrames: { pose: string; vector: number[] }[],
): AngleEmbeddings5D {
  const rec = (rawAngles && typeof rawAngles === 'object' ? rawAngles : {}) as Record<string, unknown>;
  const center = isVector512(rec.center)
    ? rec.center
    : rawFrames.find((f) => f.pose === 'front')?.vector ?? fallbackFrontal;
  const left = isVector512(rec.left)
    ? rec.left
    : rawFrames.find((f) => f.pose === 'left')?.vector ?? fallbackOverall;
  const right = isVector512(rec.right)
    ? rec.right
    : rawFrames.find((f) => f.pose === 'right')?.vector ?? fallbackOverall;
  const up = isVector512(rec.up)
    ? rec.up
    : rawFrames.find((f) => f.pose === 'nod')?.vector ?? fallbackOverall;
  const down = isVector512(rec.down)
    ? rec.down
    : rawFrames.find((f) => f.pose === 'nod')?.vector ?? fallbackOverall;
  return { center, left, right, up, down };
}

function normalizeAngleEmbeddingsInput(rawAngles: unknown): AngleEmbeddings5D | null {
  if (!rawAngles || typeof rawAngles !== 'object') return null;
  const rec = rawAngles as Record<string, unknown>;
  if (
    isVector512(rec.center) &&
    isVector512(rec.left) &&
    isVector512(rec.right) &&
    isVector512(rec.up) &&
    isVector512(rec.down)
  ) {
    return {
      center: rec.center,
      left: rec.left,
      right: rec.right,
      up: rec.up,
      down: rec.down,
    };
  }
  return null;
}

type FaceAngleKey = 'center' | 'left' | 'right' | 'up' | 'down';
const FACE_ANGLE_KEYS: readonly FaceAngleKey[] = ['center', 'left', 'right', 'up', 'down'];

function normalizeAngleUrlMap(
  raw: unknown,
): Partial<Record<FaceAngleKey, string>> | undefined {
  if (!raw || typeof raw !== 'object') return undefined;
  const rec = raw as Record<string, unknown>;
  const out: Partial<Record<FaceAngleKey, string>> = {};
  for (const key of FACE_ANGLE_KEYS) {
    const val = rec[key];
    if (typeof val === 'string' && val.trim().length > 0) {
      out[key] = val;
    }
  }
  return Object.keys(out).length > 0 ? out : undefined;
}

function generateUniversalLinkCode(patientId: string): string {
  const cleaned = patientId.replace(/[^a-zA-Z0-9]/g, '').toUpperCase();
  const suffix = (cleaned.slice(-8) || randomHex(4).toUpperCase()).padStart(8, '0');
  return `DTC-EHR-${suffix}`;
}

function clampScore(val: unknown, min: number, max: number, fallback: number): number {
  const num = typeof val === 'number' && Number.isFinite(val) ? val : fallback;
  return Math.max(min, Math.min(max, Math.round(num)));
}

function buildPatientHistoryPayload(patient: PatientRecord): Record<string, unknown> {
  return {
    gender: patient.gender ?? 'ชาย',
    strokeType: patient.strokeType ?? 'ELDERLY_GENERAL',
    affectedSide: patient.affectedSide ?? 'NONE',
    motorAssessmentScore: patient.motorAssessmentScore ?? 3,
    adlDressingScore: patient.adlDressingScore ?? 3,
    maxSafeRomDeg: patient.maxSafeRomDeg ?? 150,
    universalHealthLinkCode: patient.universalHealthLinkCode ?? generateUniversalLinkCode(patient.id),
    rehabSessionHistory: patient.rehabSessionHistory ?? [],
    faceBiometricStatus: patient.faceBiometricStatus ?? 'PENDING',
    faceFramesCount: patient.faceFramesCount ?? 0,
    faceImageUrl: patient.faceImageUrl ?? null,
    faceImageUrls: patient.faceImageUrls ?? [],
    angleImageUrls: patient.angleImageUrls ?? null,
    rehabSchedules: patient.rehabSchedules ?? [],
    delegatedToPtId: patient.delegatedToPtId ?? null,
    delegationDate: patient.delegationDate ?? null,
    delegationNote: patient.delegationNote ?? null,
    masterEmbedding: patient.masterEmbedding ?? null,
    angleEmbeddings: patient.angleEmbeddings ?? null,
    biometricEngine: patient.biometricEngine ?? null,
    geometricRatios: patient.geometricRatios ?? null,
  };
}

/** Writes a `patients` row and gracefully falls back if optional extended columns are not yet migrated. */
async function writePatientRowWithFramesFallback(
  db: SupabaseClient,
  mode: 'insert' | 'update',
  fields: Record<string, unknown>,
  patientId?: string,
): Promise<void> {
  const exec = async (payload: Record<string, unknown>) =>
    (mode === 'insert'
      ? await db.from('patients').insert(payload)
      : await db.from('patients').update(payload).eq('id', patientId!)) as
      | { error?: unknown }
      | undefined;

  const res = await exec(fields);
  if (res?.error) {
    const fallbackPayload = { ...fields };
    for (const optCol of [
      'face_frames_count',
      'stroke_type',
      'affected_side',
      'motor_assessment_score',
      'adl_dressing_score',
      'max_safe_rom_deg',
      'universal_health_link_code',
    ]) {
      delete fallbackPayload[optCol];
    }
    await exec(fallbackPayload);
  }
}

/** Inserts an enrollment audit row, retrying with `shot_count: 10` on legacy `CHECK (shot_count = 10)` schemas. */
async function insertFaceEnrollmentWithCompatShotCount(
  db: SupabaseClient,
  payload: {
    patient_id: string;
    shot_count: number;
    liveness_verified: boolean;
    enrolled_by: string;
  },
): Promise<void> {
  const res = (await db.from('face_enrollments').insert(payload)) as { error?: unknown } | undefined;
  if (res?.error && payload.shot_count !== 10) {
    await db.from('face_enrollments').insert({ ...payload, shot_count: 10 });
  }
}

/** Uploads a base64 face snapshot into Supabase Storage and falls back cleanly on offline/errors. */
async function uploadFaceImageIfNeeded(
  supabase: SupabaseClient | null,
  patientId: string,
  rawUrl: string | undefined,
  tag = 'main',
): Promise<string | undefined> {
  if (!supabase || !rawUrl || !rawUrl.startsWith('data:image/')) {
    return rawUrl;
  }
  const match = rawUrl.match(/^data:image\/([a-zA-Z0-9.+-]+);base64,(.+)$/);
  if (!match) return rawUrl;
  const rawSubtype = match[1].toLowerCase();
  const ext = rawSubtype === 'jpeg' ? 'jpg' : rawSubtype.replace(/[^a-z0-9]/g, '') || 'jpg';
  const contentType = `image/${rawSubtype}`;
  const safeTag = tag.replace(/[^a-zA-Z0-9_-]/g, '') || 'main';
  try {
    const binary = globalThis.atob(match[2]);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i += 1) {
      bytes[i] = binary.charCodeAt(i);
    }
    await supabase.storage.createBucket('patient-faces', { public: true }).catch(() => {});
    const objectPath = `patients/${patientId}-${safeTag}-${Date.now()}.${ext}`;
    const { error } = await supabase.storage
      .from('patient-faces')
      .upload(objectPath, bytes, { contentType, upsert: true });
    if (error) return rawUrl;
    const { data } = supabase.storage.from('patient-faces').getPublicUrl(objectPath);
    return data?.publicUrl || rawUrl;
  } catch {
    return rawUrl;
  }
}

async function uploadMultiAngleFaceBundle(
  supabase: SupabaseClient | null,
  patientId: string,
  input: {
    faceImageUrl?: string;
    angleImageUrls?: unknown;
    faceImageUrls?: unknown;
  },
): Promise<{
  primaryFaceImageUrl?: string;
  angleImageUrls?: Partial<Record<FaceAngleKey, string>>;
  faceImageUrls?: string[];
}> {
  const cleanAngles = normalizeAngleUrlMap(input.angleImageUrls);
  const angleEntries = await Promise.all(
    FACE_ANGLE_KEYS.map(async (key) => {
      const raw = cleanAngles?.[key];
      if (!raw) return [key, undefined] as const;
      const uploaded = await uploadFaceImageIfNeeded(supabase, patientId, raw, key);
      return [key, uploaded] as const;
    }),
  );

  const storedAngleUrls: Partial<Record<FaceAngleKey, string>> = {};
  for (const [key, url] of angleEntries) {
    if (url) storedAngleUrls[key] = url;
  }

  const hasMultipleAngles = Object.keys(storedAngleUrls).length > 1;
  const storedPrimary = input.faceImageUrl
    ? await uploadFaceImageIfNeeded(
        supabase,
        patientId,
        input.faceImageUrl,
        hasMultipleAngles ? 'composite-5angles' : 'main',
      )
    : storedAngleUrls.center;

  const extraUrls: string[] = [];
  if (Array.isArray(input.faceImageUrls)) {
    const seenInputs = new Set<string>(
      [
        input.faceImageUrl ?? '',
        ...FACE_ANGLE_KEYS.map((k) => cleanAngles?.[k] ?? ''),
      ].filter(Boolean),
    );
    for (let i = 0; i < input.faceImageUrls.length; i += 1) {
      const raw = input.faceImageUrls[i];
      if (typeof raw !== 'string' || !raw || seenInputs.has(raw)) continue;
      const uploaded = await uploadFaceImageIfNeeded(supabase, patientId, raw, `extra-${i + 1}`);
      if (uploaded) extraUrls.push(uploaded);
    }
  }

  const combinedUrls = Array.from(
    new Set(
      [
        storedPrimary,
        ...FACE_ANGLE_KEYS.map((k) => storedAngleUrls[k]),
        ...extraUrls,
      ].filter((u): u is string => typeof u === 'string' && u.length > 0),
    ),
  );

  return {
    primaryFaceImageUrl: storedPrimary ?? combinedUrls[0],
    angleImageUrls: Object.keys(storedAngleUrls).length > 0 ? storedAngleUrls : undefined,
    faceImageUrls: combinedUrls.length > 0 ? combinedUrls : undefined,
  };
}

function sortNewestFirst<T extends { createdAt: string }>(items: readonly T[]): T[] {
  return [...items].sort((a, b) => (Date.parse(b.createdAt) || 0) - (Date.parse(a.createdAt) || 0));
}

function publicUser(u: StoredUserAccount): UserAccount {
  const { passwordHash: _omit, ...rest } = u;
  return rest;
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      'content-type': 'application/json; charset=utf-8',
      'access-control-allow-origin': '*',
      'access-control-allow-headers': 'Content-Type, Authorization, X-Staff-Token, x-device-id',
      'access-control-allow-methods': 'GET, POST, PUT, PATCH, DELETE, OPTIONS',
    },
  });
}

async function readJsonBody<T = Record<string, unknown>>(request: Request): Promise<T | null> {
  try {
    return (await request.json()) as T;
  } catch {
    return null;
  }
}

/**
 * Full-parity edge/cloud API router for Cloudflare Workers & Supabase delegation.
 * Mirrors every route, RBAC rule, biometric gate, and clinical guard in apps/api/src/app.ts.
 */
function normalizeInitialRehabSchedules(
  rawSchedules: unknown,
  patientId: string,
  assignedPtId: string,
): RehabScheduleItem[] {
  if (!Array.isArray(rawSchedules)) return [];
  const validStatuses = new Set<RehabScheduleItem['status']>(['ACTIVE', 'REST', 'COMPLETED', 'MISSED']);
  const out: RehabScheduleItem[] = [];
  rawSchedules.forEach((raw, idx) => {
    if (!raw || typeof raw !== 'object') return;
    const item = raw as Record<string, unknown>;
    const scheduledDate = typeof item.scheduledDate === 'string' ? item.scheduledDate.trim() : '';
    if (!scheduledDate) return;
    const rawStatus = typeof item.status === 'string' ? (item.status as RehabScheduleItem['status']) : 'ACTIVE';
    const status = validStatuses.has(rawStatus) ? rawStatus : 'ACTIVE';
    const sets = typeof item.sets === 'number' && Number.isFinite(item.sets) ? Math.max(1, Math.round(item.sets)) : 3;
    const reps = typeof item.reps === 'number' && Number.isFinite(item.reps) ? Math.max(1, Math.round(item.reps)) : 10;
    out.push({
      id: typeof item.id === 'string' && item.id.trim() ? item.id.trim() : `sched-${Date.now()}-${idx + 1}`,
      patientId,
      ptId: typeof item.ptId === 'string' && item.ptId.trim() ? item.ptId.trim() : assignedPtId,
      exerciseId: typeof item.exerciseId === 'string' && item.exerciseId.trim() ? item.exerciseId.trim() : undefined,
      modpackId: typeof item.modpackId === 'string' && item.modpackId.trim() ? item.modpackId.trim() : undefined,
      scheduledDate,
      sets,
      reps,
      status,
      notes: typeof item.notes === 'string' ? item.notes : undefined,
    });
  });
  return out.sort((a, b) => a.scheduledDate.localeCompare(b.scheduledDate));
}

export function createCloudRouter(config: CloudClientConfig = {}) {
  const url = config.supabaseUrl || '';
  const key = config.supabaseServiceKey || config.supabaseAnonKey || '';
  const tokenSecret = config.supabaseServiceKey || config.supabaseAnonKey || 'drtechcares-clinical-edge-token-secret-2026';
  const supabase: SupabaseClient | null =
    url && key
      ? createClient(url, key, { auth: { persistSession: false, autoRefreshToken: false } })
      : null;
  const state = createInitialState(Boolean(supabase));

  let hydrationPromise: Promise<void> | null = null;
  let lastHydratedAt = 0;
  const ensureHydrated = async (force = false, bypassCooldown = false) => {
    if (!supabase) return;
    if (!hydrationPromise || (force && (bypassCooldown || Date.now() - lastHydratedAt > 2500))) {
      hydrationPromise = hydrateStateFromSupabase(supabase, state).then(() => {
        lastHydratedAt = Date.now();
      });
    }
    await hydrationPromise;
  };

  const recordAudit = (entry: Omit<AuditLogRecord, 'id' | 'createdAt'>): AuditLogRecord => {
    const log: AuditLogRecord = {
      id: newRecordId(Boolean(supabase), 'audit'),
      ...entry,
      createdAt: new Date().toISOString(),
    };
    state.audit_logs.unshift(log);
    if (state.audit_logs.length > 1000) {
      state.audit_logs.length = 1000;
    }
    if (supabase) {
      void persistToSupabase(supabase, (db) =>
        db.from('audit_logs').insert({
          id: log.id,
          event_type: log.eventType,
          actor_role: log.actorRole,
          actor_id: log.actorId,
          actor_name: log.actorName,
          ip_address: log.ipAddress,
          device_id: log.deviceId,
          target_id: log.targetId ?? null,
          detail: { message: log.detail },
        }),
      );
    }
    return log;
  };

  const extractToken = (request: Request): string | null => {
    const authHeader = request.headers.get('authorization');
    if (authHeader && authHeader.startsWith('Bearer ')) {
      const t = authHeader.slice(7).trim();
      if (t) return t;
    }
    const staffHeader = request.headers.get('x-staff-token');
    if (staffHeader && staffHeader.trim()) {
      return staffHeader.trim();
    }
    return null;
  };

  const authenticate = async (
    request: Request,
    clientIp = 'unknown',
    clientDeviceId = 'kiosk-browser',
  ): Promise<{ user?: StoredUserAccount; token?: string }> => {
    const token = extractToken(request);
    if (!token) return {};

    // 1. Check existing in-memory / hydrated session row (honors revocations & force logouts)
    const session = state.active_sessions.find((s) => s.token === token);
    if (session) {
      if (session.revoked) return {};
      let user = state.users.find((u) => u.id === session.userId && u.active);
      if (!user && supabase) {
        await ensureHydrated(true, true);
        user = state.users.find((u) => u.id === session.userId && u.active);
      }
      if (!user) return {};
      const forcedBefore =
        user.forceLoggedOutAt && new Date(session.createdAt) <= new Date(user.forceLoggedOutAt);
      if (forcedBefore) return {};
      return { user, token };
    }

    // 2. Stateless HMAC-like verification for stf.<payload>.<sig> tokens across edge isolates
    const signed = await verifyStaffToken(token, tokenSecret);
    if (signed) {
      let user = state.users.find(
        (u) => (u.id === signed.uid || u.username.toLowerCase() === signed.usr.toLowerCase()) && u.active,
      );
      if (!user && supabase) {
        await ensureHydrated(true, true);
        user = state.users.find(
          (u) => (u.id === signed.uid || u.username.toLowerCase() === signed.usr.toLowerCase()) && u.active,
        );
      }
      if (!user) return {};
      const forcedBefore =
        user.forceLoggedOutAt && signed.iat <= new Date(user.forceLoggedOutAt).getTime();
      if (forcedBefore) return {};
      state.active_sessions.unshift({
        token,
        userId: user.id,
        role: user.role,
        ipAddress: clientIp,
        deviceId: clientDeviceId,
        createdAt: new Date(signed.iat).toISOString(),
        revoked: false,
      });
      return { user, token };
    }

    // 3. Fallback: verify Supabase Auth JWT if a Supabase session token was sent directly
    if (supabase && token.includes('.')) {
      try {
        const { data } = await supabase.auth.getUser(token);
        const sbUser = data?.user;
        if (sbUser) {
          const emailPrefix = sbUser.email ? sbUser.email.split('@')[0]?.toLowerCase() : '';
          const metaUsername =
            typeof sbUser.user_metadata?.username === 'string'
              ? sbUser.user_metadata.username.toLowerCase()
              : emailPrefix;
          let matched = state.users.find(
            (u) =>
              (u.id === sbUser.id || (metaUsername && u.username.toLowerCase() === metaUsername)) &&
              u.active,
          );
          if (!matched) {
            await ensureHydrated(true, true);
            matched = state.users.find(
              (u) =>
                (u.id === sbUser.id || (metaUsername && u.username.toLowerCase() === metaUsername)) &&
                u.active,
            );
          }
          if (matched) return { user: matched, token };
        }
      } catch {
        // Ignore invalid JWT
      }
    }

    return {};
  };

  const requireAuth = (
    authUser: StoredUserAccount | undefined,
    roles?: UserRole[],
  ): Response | null => {
    if (!authUser) {
      return jsonResponse({ error: 'UNAUTHORIZED', message: 'Authentication required' }, 401);
    }
    if (roles && !roles.includes(authUser.role)) {
      return jsonResponse({ error: 'FORBIDDEN', message: 'Insufficient role permissions' }, 403);
    }
    return null;
  };

  return async function handleApiRequest(request: Request): Promise<Response> {
    if (request.method === 'OPTIONS') {
      return jsonResponse({ ok: true }, 204);
    }

    const reqUrl = new URL(request.url);
    const pathname = reqUrl.pathname;
    const method = request.method.toUpperCase();

    // Fast-path TTS audio proxy (th / en / zh-CN) before Supabase hydration for zero DB latency
    if (pathname === '/api/tts' && method === 'GET') {
      const cleanText = clampTtsText(reqUrl.searchParams.get('text') || reqUrl.searchParams.get('q') || '');
      if (!cleanText) {
        return jsonResponse({ error: 'VALIDATION_ERROR', message: 'Missing text parameter' }, 400);
      }
      const langParam = reqUrl.searchParams.get('lang') || 'th';
      const synthesized = await synthesizeTtsAudio(langParam, cleanText);
      if (!synthesized) {
        return jsonResponse({ error: 'TTS_UPSTREAM_UNAVAILABLE', message: 'TTS synthesis unavailable' }, 502);
      }
      return new Response(synthesized.audio, {
        status: 200,
        headers: {
          'Content-Type': synthesized.contentType,
          'Content-Length': String(synthesized.audio.byteLength),
          'Cache-Control': 'public, max-age=86400',
          'Access-Control-Allow-Origin': '*',
          'X-TTS-Provider': synthesized.provider,
        },
      });
    }

    const requiresFreshHydration =
      (pathname === '/api/biometrics/index' && method === 'GET') ||
      (pathname === '/api/biometric/verify' && method === 'POST') ||
      (pathname === '/api/patients' && method === 'GET') ||
      (pathname === '/api/staff-calls' && method === 'GET') ||
      (pathname === '/api/audit-logs' && method === 'GET') ||
      (pathname.startsWith('/api/courses/') && method === 'GET') ||
      (pathname.startsWith('/api/patients/') && method === 'DELETE') ||
      (/^\/api\/(?:users|staff)\/[^/]+$/.test(pathname) && method === 'DELETE');

    await ensureHydrated(
      requiresFreshHydration || (method === 'GET' && pathname !== '/api/health'),
      requiresFreshHydration,
    );

    const clientIp =
      request.headers.get('x-forwarded-for')?.split(',')[0]?.trim() ||
      request.headers.get('cf-connecting-ip') ||
      '127.0.0.1';
    const clientDeviceId = request.headers.get('x-device-id') || 'kiosk-browser';

    // Enforce IP / Device Bans
    const banned = state.banned_targets.find(
      (b) =>
        (b.targetType === 'ip' && b.value === clientIp) ||
        (b.targetType === 'device' && b.value === clientDeviceId),
    );
    if (banned) {
      return jsonResponse(
        {
          error: 'BANNED_IP_OR_DEVICE',
          message: `Access denied: ${banned.targetType.toUpperCase()} is banned (${banned.reason})`,
        },
        403,
      );
    }

    const { user: authUser, token: authToken } = await authenticate(request, clientIp, clientDeviceId);

    // Health
    if (pathname === '/api/health' && method === 'GET') {
      let supabaseConnected = false;
      const t0 = Date.now();
      if (supabase) {
        try {
          const { error } = await supabase.from('system_settings').select('id').limit(1);
          supabaseConnected = !error;
        } catch {
          supabaseConnected = false;
        }
      }
      return jsonResponse({
        ok: true,
        service: 'drtechcares-cloud-router',
        persistentDb: Boolean(supabase),
        supabaseConnected,
        latencyMs: Date.now() - t0,
        zeroGhostCache: true,
      });
    }

    // ========================
    // 1. AUTHENTICATION
    // ========================
    if (pathname === '/api/auth/login' && method === 'POST') {
      const body = await readJsonBody<{ username?: string; password?: string }>(request);
      if (!body?.username || !body?.password) {
        return jsonResponse({ error: 'VALIDATION_ERROR' }, 400);
      }
      const rawUsername = body.username.trim().toLowerCase();
      const normalizedUsername = rawUsername === 'director' ? 'admin' : rawUsername;
      let user = state.users.find(
        (u) => u.username.toLowerCase() === normalizedUsername && u.active,
      );
      if (!user && supabase) {
        await ensureHydrated(true, true);
        user = state.users.find(
          (u) => u.username.toLowerCase() === normalizedUsername && u.active,
        );
      }
      if (!user || !(await verifyPasswordEdge(body.password, user.passwordHash))) {
        recordAudit({
          eventType: 'LOGIN_FAILED',
          actorRole: 'anonymous',
          actorId: body.username,
          actorName: body.username,
          ipAddress: clientIp,
          deviceId: clientDeviceId,
          detail: `Failed login attempt for username "${body.username}"`,
        });
        return jsonResponse(
          { error: 'INVALID_CREDENTIALS', message: 'ชื่อผู้ใช้หรือรหัสผ่านไม่ถูกต้อง' },
          401,
        );
      }

      const token = await createStaffToken(user, tokenSecret);
      const sessionRow = {
        token,
        userId: user.id,
        role: user.role,
        ipAddress: clientIp,
        deviceId: clientDeviceId,
        createdAt: new Date().toISOString(),
        revoked: false,
      };
      state.active_sessions.unshift(sessionRow);
      await persistToSupabase(supabase, (db) =>
        db.from('active_sessions').insert({
          token: sessionRow.token,
          user_id: sessionRow.userId,
          role: sessionRow.role,
          ip_address: sessionRow.ipAddress,
          device_id: sessionRow.deviceId,
          revoked: false,
        }),
      );

      recordAudit({
        eventType: 'USER_LOGIN',
        actorRole: user.role,
        actorId: user.id,
        actorName: user.fullName,
        ipAddress: clientIp,
        deviceId: clientDeviceId,
        detail: `Successful login as ${user.role}`,
      });

      return jsonResponse({ token, user: publicUser(user) });
    }

    if (pathname === '/api/auth/register' && method === 'POST') {
      const body = await readJsonBody<{
        username?: string;
        password?: string;
        role?: string;
        fullName?: string;
        phone?: string;
        specialty?: string;
        licenseNo?: string;
      }>(request);
      const cleanUsername = body?.username?.trim();
      const cleanFullName = body?.fullName?.trim();
      const rawPassword = body?.password ?? '';
      if (!cleanUsername || !cleanFullName || rawPassword.length < 4) {
        return jsonResponse(
          {
            error: 'VALIDATION_ERROR',
            message: 'กรุณากรอกชื่อ-นามสกุล, Username และรหัสผ่านอย่างน้อย 4 ตัวอักษร',
          },
          400,
        );
      }
      const targetRole = normalizeStaffRole(body?.role ?? 'pt');
      if (targetRole === 'patient') {
        return jsonResponse({ error: 'FORBIDDEN_ROLE', message: 'Invalid staff registration role' }, 400);
      }
      if (supabase) {
        await ensureHydrated(true, true);
      }
      if (state.users.some((u) => u.username.toLowerCase() === cleanUsername.toLowerCase())) {
        return jsonResponse(
          { error: 'USERNAME_EXISTS', message: `ชื่อผู้ใช้ "@${cleanUsername}" ถูกใช้งานแล้ว` },
          409,
        );
      }

      const created: StoredUserAccount = {
        id: newRecordId(Boolean(supabase), 'user'),
        username: cleanUsername,
        passwordHash: await hashPasswordEdge(rawPassword),
        role: targetRole,
        fullName: cleanFullName,
        phone: body?.phone?.trim() ?? '',
        specialty: body?.specialty?.trim() || (targetRole === 'super_admin' ? 'บริหารโรงพยาบาล' : 'เวชศาสตร์ฟื้นฟูกายภาพบำบัด'),
        licenseNo: body?.licenseNo?.trim() || undefined,
        active: true,
        forceLoggedOutAt: null,
        createdAt: new Date().toISOString(),
      };
      state.users.push(created);

      await persistToSupabase(supabase, (db) =>
        db.from('staff_users').insert({
          id: created.id,
          username: created.username,
          password_hash: created.passwordHash,
          full_name: created.fullName,
          phone: created.phone,
          specialty: created.specialty ?? null,
          license_no: created.licenseNo ?? null,
          patient_id: null,
          role: created.role,
          active: created.active,
        }),
      );

      const token = await createStaffToken(created, tokenSecret);
      const sessionRow = {
        token,
        userId: created.id,
        role: created.role,
        ipAddress: clientIp,
        deviceId: clientDeviceId,
        createdAt: new Date().toISOString(),
        revoked: false,
      };
      state.active_sessions.unshift(sessionRow);
      await persistToSupabase(supabase, (db) =>
        db.from('active_sessions').insert({
          token: sessionRow.token,
          user_id: sessionRow.userId,
          role: sessionRow.role,
          ip_address: sessionRow.ipAddress,
          device_id: sessionRow.deviceId,
          revoked: false,
        }),
      );

      recordAudit({
        eventType: 'STAFF_REGISTERED',
        actorRole: created.role,
        actorId: created.id,
        actorName: created.fullName,
        ipAddress: clientIp,
        deviceId: clientDeviceId,
        targetId: created.id,
        detail: `Self-registered ${created.role} account @${created.username}`,
      });

      return jsonResponse({ token, user: publicUser(created) }, 201);
    }

    if (pathname === '/api/auth/logout' && method === 'POST') {
      const err = requireAuth(authUser);
      if (err) return err;
      const session = state.active_sessions.find((s) => s.token === authToken);
      if (session) {
        session.revoked = true;
        await persistToSupabase(supabase, (db) =>
          db.from('active_sessions').update({ revoked: true }).eq('token', session.token),
        );
      }
      return jsonResponse({ ok: true });
    }

    if (pathname === '/api/auth/me' && method === 'GET') {
      const err = requireAuth(authUser);
      if (err) return err;
      return jsonResponse({ user: publicUser(authUser!) });
    }

    if (pathname === '/api/auth/change-password' && method === 'POST') {
      const err = requireAuth(authUser);
      if (err) return err;
      const body = await readJsonBody<{ currentPassword?: string; newPassword?: string }>(request);
      if (!body?.currentPassword || !body?.newPassword || body.newPassword.length < 6) {
        return jsonResponse(
          { error: 'INVALID_PASSWORD', message: 'New password must be at least 6 characters' },
          400,
        );
      }
      if (!(await verifyPasswordEdge(body.currentPassword, authUser!.passwordHash))) {
        return jsonResponse(
          { error: 'WRONG_CURRENT_PASSWORD', message: 'รหัสผ่านเดิมไม่ถูกต้อง' },
          400,
        );
      }
      authUser!.passwordHash = await hashPasswordEdge(body.newPassword);
      await persistToSupabase(supabase, (db) =>
        db.from('staff_users').update({ password_hash: authUser!.passwordHash }).eq('id', authUser!.id),
      );
      recordAudit({
        eventType: 'PASSWORD_CHANGED',
        actorRole: authUser!.role,
        actorId: authUser!.id,
        actorName: authUser!.fullName,
        ipAddress: clientIp,
        deviceId: clientDeviceId,
        detail: 'User changed own password',
      });
      return jsonResponse({ ok: true });
    }

    // ========================
    // 2. USER & RBAC MANAGEMENT
    // ========================
    if ((pathname === '/api/users' || pathname === '/api/staff') && method === 'GET') {
      const err = requireAuth(authUser, ['super_admin', 'pt']);
      if (err) return err;
      const users =
        authUser!.role === 'super_admin'
          ? state.users
          : state.users.filter((u) => u.role === 'pt' || u.role === 'patient');
      return jsonResponse({ users: users.map(publicUser) });
    }

    if ((pathname === '/api/users' || pathname === '/api/staff') && method === 'POST') {
      const err = requireAuth(authUser, ['super_admin', 'pt']);
      if (err) return err;
      const body = await readJsonBody<{
        username?: string;
        password?: string;
        role?: string;
        fullName?: string;
        phone?: string;
        specialty?: string;
        licenseNo?: string;
        patientId?: string;
      }>(request);
      if (!body?.username || !body?.password || !body?.role || !body?.fullName) {
        return jsonResponse({ error: 'VALIDATION_ERROR' }, 400);
      }
      const normalizedRole = normalizeStaffRole(body.role);
      if (authUser!.role === 'pt' && normalizedRole !== 'patient') {
        return jsonResponse(
          { error: 'FORBIDDEN', message: 'PT cannot create Admin or PT accounts' },
          403,
        );
      }
      if (supabase) {
        await ensureHydrated(true, true);
      }
      if (state.users.some((u) => u.username.toLowerCase() === body.username!.trim().toLowerCase())) {
        return jsonResponse({ error: 'USERNAME_EXISTS', message: 'ชื่อผู้ใช้นี้มีอยู่ในระบบแล้ว' }, 409);
      }
      const created: StoredUserAccount = {
        id: newRecordId(Boolean(supabase), 'user'),
        username: body.username.trim(),
        passwordHash: await hashPasswordEdge(body.password),
        role: normalizedRole,
        fullName: body.fullName.trim(),
        phone: body.phone?.trim() ?? '',
        specialty: body.specialty?.trim(),
        licenseNo: body.licenseNo?.trim(),
        patientId: body.patientId,
        active: true,
        forceLoggedOutAt: null,
        createdAt: new Date().toISOString(),
      };
      state.users.push(created);
      await persistToSupabase(supabase, (db) =>
        db.from('staff_users').insert({
          id: created.id,
          username: created.username,
          password_hash: created.passwordHash,
          full_name: created.fullName,
          phone: created.phone,
          specialty: created.specialty ?? null,
          license_no: created.licenseNo ?? null,
          patient_id: created.patientId ?? null,
          role: created.role,
          active: created.active,
        }),
      );
      recordAudit({
        eventType: 'USER_CREATED',
        actorRole: authUser!.role,
        actorId: authUser!.id,
        actorName: authUser!.fullName,
        ipAddress: clientIp,
        deviceId: clientDeviceId,
        targetId: created.id,
        detail: `Created ${normalizedRole} account @${created.username}`,
      });
      return jsonResponse({ user: publicUser(created) }, 201);
    }

    const userResetMatch = pathname.match(/^\/api\/(?:users|staff)\/([^/]+)\/reset-password$/);
    if (userResetMatch && method === 'POST') {
      const err = requireAuth(authUser, ['super_admin', 'pt']);
      if (err) return err;
      let target = state.users.find((u) => u.id === userResetMatch[1]);
      if (!target && supabase) {
        await ensureHydrated(true, true);
        target = state.users.find((u) => u.id === userResetMatch[1]);
      }
      if (!target) return jsonResponse({ error: 'USER_NOT_FOUND' }, 404);
      if (authUser!.role === 'pt' && target.role !== 'patient') {
        return jsonResponse(
          { error: 'FORBIDDEN', message: 'PT can only reset Patient passwords' },
          403,
        );
      }
      const body = await readJsonBody<{ newPassword?: string }>(request);
      target.passwordHash = await hashPasswordEdge(body?.newPassword || 'Patient@2026');
      await persistToSupabase(supabase, (db) =>
        db.from('staff_users').update({ password_hash: target!.passwordHash }).eq('id', target!.id),
      );
      recordAudit({
        eventType: 'PASSWORD_RESET',
        actorRole: authUser!.role,
        actorId: authUser!.id,
        actorName: authUser!.fullName,
        ipAddress: clientIp,
        deviceId: clientDeviceId,
        targetId: target.id,
        detail: `Reset password for @${target.username}`,
      });
      return jsonResponse({ ok: true });
    }

    const userMatch = pathname.match(/^\/api\/(?:users|staff)\/([^/]+)$/);
    if (userMatch && method === 'PUT') {
      const err = requireAuth(authUser);
      if (err) return err;
      let target = state.users.find((u) => u.id === userMatch[1]);
      if (!target && supabase) {
        await ensureHydrated(true, true);
        target = state.users.find((u) => u.id === userMatch[1]);
      }
      if (!target) return jsonResponse({ error: 'USER_NOT_FOUND' }, 404);
      const actor = authUser!;
      if (actor.role === 'pt' && target.id !== actor.id && target.role !== 'patient') {
        return jsonResponse(
          { error: 'FORBIDDEN', message: 'PT cannot edit Admin or other PT accounts' },
          403,
        );
      }
      if (actor.role === 'patient' && target.id !== actor.id) {
        return jsonResponse({ error: 'FORBIDDEN' }, 403);
      }
      const body = (await readJsonBody<Partial<UserAccount>>(request)) ?? {};
      if (body.fullName !== undefined) target.fullName = body.fullName;
      if (body.phone !== undefined) target.phone = body.phone;
      if (body.specialty !== undefined && actor.role !== 'patient') target.specialty = body.specialty;
      if (body.licenseNo !== undefined && actor.role !== 'patient') target.licenseNo = body.licenseNo;
      if (body.avatarUrl !== undefined) target.avatarUrl = body.avatarUrl;
      if (body.bio !== undefined) target.bio = body.bio;
      if (body.role !== undefined && actor.role === 'super_admin') target.role = normalizeStaffRole(body.role);
      if (body.active !== undefined && actor.role === 'super_admin') target.active = body.active;

      await persistToSupabase(supabase, (db) =>
        db
          .from('staff_users')
          .update({
            full_name: target!.fullName,
            phone: target!.phone,
            specialty: target!.specialty ?? null,
            license_no: target!.licenseNo ?? null,
            role: target!.role,
            active: target!.active,
          })
          .eq('id', target!.id),
      );
      await persistToSupabase(supabase, (db) =>
        db
          .from('staff_users')
          .update({
            avatar_url: target!.avatarUrl ?? null,
            bio: target!.bio ?? null,
          })
          .eq('id', target!.id),
      );

      recordAudit({
        eventType: 'USER_UPDATED',
        actorRole: actor.role,
        actorId: actor.id,
        actorName: actor.fullName,
        ipAddress: clientIp,
        deviceId: clientDeviceId,
        targetId: target.id,
        detail: `Updated user @${target.username}`,
      });
      return jsonResponse({ user: publicUser(target) });
    }

    if (userMatch && method === 'DELETE') {
      const err = requireAuth(authUser, ['super_admin']);
      if (err) return err;
      let idx = state.users.findIndex((u) => u.id === userMatch[1]);
      if (idx === -1 && supabase) {
        await ensureHydrated(true, true);
        idx = state.users.findIndex((u) => u.id === userMatch[1]);
      }
      if (idx === -1) return jsonResponse({ error: 'USER_NOT_FOUND' }, 404);
      const removed = state.users.splice(idx, 1)[0];
      state.deleted_user_ids = Array.from(new Set([...(state.deleted_user_ids ?? []), removed.id]));
      state.active_sessions = state.active_sessions.filter((s) => s.userId !== removed.id);

      const linkedPatient = state.patients.find(
        (p) =>
          (removed.patientId && p.id === removed.patientId) ||
          p.hn.toLowerCase() === removed.username.toLowerCase() ||
          (removed.role === 'patient' && p.fullName.toLowerCase() === removed.fullName.toLowerCase()),
      );
      const linkedPatientId = linkedPatient?.id ?? removed.patientId ?? null;
      if (linkedPatientId) {
        state.deleted_patient_ids = Array.from(new Set([...(state.deleted_patient_ids ?? []), linkedPatientId]));
        state.patients = state.patients.filter((p) => p.id !== linkedPatientId);
        state.face_templates = state.face_templates.filter((t) => t.patientId !== linkedPatientId);
        state.patient_courses = state.patient_courses.filter((c) => c.patientId !== linkedPatientId);
        state.rehab_sessions = state.rehab_sessions.filter((s) => s.patientId !== linkedPatientId);
        state.medical_notes = state.medical_notes.filter((n) => n.patientId !== linkedPatientId);
        state.treatment_reports = state.treatment_reports.filter((r) => r.patientId !== linkedPatientId);
      }

      await persistToSupabase(supabase, async (db) => {
        await db.from('active_sessions').delete().eq('user_id', removed.id);
        await db.from('staff_users').delete().eq('id', removed.id);
        await db.from('staff_users').delete().eq('username', removed.username);
        if (linkedPatientId) {
          await db.from('face_templates').delete().eq('patient_id', linkedPatientId);
          await db.from('patient_biometrics').delete().eq('patient_id', linkedPatientId);
          await db.from('face_enrollments').delete().eq('patient_id', linkedPatientId);
          await db.from('patient_courses').delete().eq('patient_id', linkedPatientId);
          await db.from('rehab_schedules').delete().eq('patient_id', linkedPatientId);
          await db.from('rehab_sessions').delete().eq('patient_id', linkedPatientId);
          await db.from('medical_notes').delete().eq('patient_id', linkedPatientId);
          await db.from('treatment_reports').delete().eq('patient_id', linkedPatientId);
          await db.from('staff_users').delete().eq('patient_id', linkedPatientId);
          await db.from('patients').delete().eq('id', linkedPatientId);
        }
      });
      recordAudit({
        eventType: 'USER_DELETED',
        actorRole: authUser!.role,
        actorId: authUser!.id,
        actorName: authUser!.fullName,
        ipAddress: clientIp,
        deviceId: clientDeviceId,
        targetId: removed.id,
        detail: linkedPatientId
          ? `Deleted user @${removed.username} and cascaded patient ${linkedPatientId}`
          : `Deleted user @${removed.username}`,
      });
      return jsonResponse({ deleted: true, cascadedPatientId: linkedPatientId });
    }

    // ========================
    // 3. BIOMETRICS & KIOSK FACE AUTH
    // ========================
    if (pathname === '/api/biometric/enroll' && method === 'POST') {
      const err = requireAuth(authUser, ['super_admin', 'pt']);
      if (err) return err;
      const body = await readJsonBody<{
        hn?: string;
        fullName?: string;
        age?: number;
        gender?: string;
        phone?: string;
        diagnosis?: string;
        strokeType?: StrokeType;
        affectedSide?: AffectedSide;
        motorAssessmentScore?: number;
        adlDressingScore?: number;
        maxSafeRomDeg?: number;
        universalHealthLinkCode?: string;
        assignedPtId?: string;
        nextAppointmentDate?: string;
        username?: string;
        password?: string;
        skipFaceScan?: boolean;
        faceBiometricStatus?: 'PENDING' | 'VERIFIED';
        faceImageUrl?: string;
        faceImageUrls?: unknown;
        angleImageUrls?: unknown;
        frontalCentroid?: unknown;
        overallCentroid?: unknown;
        rawFrames?: { pose?: string; vector?: unknown; qualityScore?: number }[];
        masterEmbedding?: unknown;
        angleEmbeddings?: unknown;
        biometricEngine?: 'arcface-onnx' | 'hybrid-512d';
        geometricRatios?: GeometricAnthropometricRatios;
        framesCaptured?: number;
        qualityScore?: number;
        rehabSchedules?: unknown;
      }>(request);

      const isPendingSkip = Boolean(body?.skipFaceScan);
      const valid120Payload =
        Boolean(body) &&
        isVector512(body?.masterEmbedding) &&
        normalizeAngleEmbeddingsInput(body?.angleEmbeddings) !== null;
      const valid10ShotPayload =
        Boolean(body) &&
        isVector512(body?.frontalCentroid) &&
        isVector512(body?.overallCentroid) &&
        Array.isArray(body?.rawFrames) &&
        body!.rawFrames.length >= 10 &&
        body!.rawFrames.every(
          (f) =>
            (f.pose === 'front' || f.pose === 'left' || f.pose === 'right' || f.pose === 'nod') &&
            isVector512(f.vector) &&
            typeof f.qualityScore === 'number' &&
            f.qualityScore > 0,
        );

      if (
        !body ||
        typeof body.hn !== 'string' ||
        body.hn.length < 2 ||
        typeof body.fullName !== 'string' ||
        body.fullName.length < 2 ||
        typeof body.age !== 'number' ||
        !Number.isInteger(body.age) ||
        body.age < 1 ||
        body.age > 120 ||
        (!isPendingSkip && !valid10ShotPayload && !valid120Payload)
      ) {
        return jsonResponse({ error: 'VALIDATION_ERROR' }, 400);
      }

      const patientId = newRecordId(Boolean(supabase), 'pat');
      const assignedPtId =
        authUser!.role === 'pt'
          ? authUser!.id
          : body.assignedPtId || state.users.find((u) => u.role === 'pt')?.id || authUser!.id;
      const initialSchedules = normalizeInitialRehabSchedules(body.rehabSchedules, patientId, assignedPtId);
      const nextAppt =
        initialSchedules.length > 0
          ? resolveNextAppointmentDate({
              rehabSchedules: initialSchedules,
              fallbackNextAppointmentDate: body.nextAppointmentDate ?? '',
              todayIso: formatLocalIsoDate(new Date()),
            })
          : body.nextAppointmentDate ?? '';

      const rawFrames = (!isPendingSkip && Array.isArray(body.rawFrames) ? body.rawFrames : []) as {
        pose: 'front' | 'left' | 'right' | 'nod';
        vector: number[];
        qualityScore: number;
      }[];

      const resolvedAngles = !isPendingSkip
        ? normalizeAngleEmbeddingsInput(body.angleEmbeddings) ??
          buildAngleEmbeddingsFromPayload(
            body.angleEmbeddings,
            isVector512(body.frontalCentroid) ? body.frontalCentroid : (body.masterEmbedding as number[]),
            isVector512(body.overallCentroid) ? body.overallCentroid : (body.masterEmbedding as number[]),
            rawFrames,
          )
        : undefined;
      const masterVec = !isPendingSkip
        ? isVector512(body.masterEmbedding)
          ? l2Normalize(body.masterEmbedding)
          : isVector512(body.overallCentroid)
            ? body.overallCentroid
            : resolvedAngles!.center
        : undefined;
      const frontalVec = !isPendingSkip
        ? isVector512(body.frontalCentroid)
          ? body.frontalCentroid
          : resolvedAngles!.center
        : undefined;
      const framesCaptured = isPendingSkip
        ? 0
        : typeof body.framesCaptured === 'number' && body.framesCaptured > 0
          ? body.framesCaptured
          : Math.max(rawFrames.length, 120);
      const qualityScore = typeof body.qualityScore === 'number' ? body.qualityScore : 0.96;

      const faceStatus: 'PENDING' | 'VERIFIED' = isPendingSkip
        ? 'PENDING'
        : body.faceBiometricStatus ?? 'VERIFIED';
      const gender = body.gender || 'ชาย';
      const faceBundle = await uploadMultiAngleFaceBundle(supabase, patientId, {
        faceImageUrl: body.faceImageUrl,
        angleImageUrls: body.angleImageUrls,
        faceImageUrls: body.faceImageUrls,
      });
      const storedFaceImageUrl = faceBundle.primaryFaceImageUrl;

      const patient: PatientRecord = {
        id: patientId,
        hn: body.hn,
        fullName: body.fullName,
        age: body.age,
        gender,
        phone: body.phone ?? '',
        diagnosis: body.diagnosis ?? 'General Rehabilitation',
        strokeType: (body.strokeType ?? 'ELDERLY_GENERAL') as StrokeType,
        affectedSide: (body.affectedSide ?? 'NONE') as AffectedSide,
        motorAssessmentScore: clampScore(body.motorAssessmentScore, 0, 6, 3),
        adlDressingScore: clampScore(body.adlDressingScore, 0, 5, 3),
        maxSafeRomDeg:
          typeof body.maxSafeRomDeg === 'number' && Number.isFinite(body.maxSafeRomDeg)
            ? Math.max(30, Math.min(180, body.maxSafeRomDeg))
            : 150,
        universalHealthLinkCode:
          typeof body.universalHealthLinkCode === 'string' && body.universalHealthLinkCode.trim()
            ? body.universalHealthLinkCode.trim().toUpperCase()
            : generateUniversalLinkCode(patientId),
        assignedPtId,
        delegatedToPtId: null,
        delegationNote: null,
        delegationDate: null,
        nextAppointmentDate: nextAppt,
        registrationStatus: 'active',
        enrolledAt: new Date().toISOString(),
        faceBiometricStatus: faceStatus,
        faceFramesCount: framesCaptured,
        faceImageUrl: storedFaceImageUrl,
        faceImageUrls: faceBundle.faceImageUrls,
        angleImageUrls: faceBundle.angleImageUrls,
        masterEmbedding: masterVec,
        angleEmbeddings: resolvedAngles,
        ...(body.biometricEngine ? { biometricEngine: body.biometricEngine } : {}),
        ...(body.geometricRatios ? { geometricRatios: body.geometricRatios } : {}),
        rehabSchedules: initialSchedules,
        biometricHistory:
          !isPendingSkip && frontalVec && masterVec
            ? {
                frontal_centroid: frontalVec,
                overall_centroid: masterVec,
                raw_frames: rawFrames,
              }
            : undefined,
      };

      state.patients.push(patient);
      await persistToSupabase(supabase, (db) =>
        writePatientRowWithFramesFallback(db, 'insert', {
          id: patient.id,
          hn: patient.hn,
          full_name: patient.fullName,
          age: patient.age,
          phone: patient.phone,
          diagnosis: patient.diagnosis,
          stroke_type: patient.strokeType,
          affected_side: patient.affectedSide,
          motor_assessment_score: patient.motorAssessmentScore,
          adl_dressing_score: patient.adlDressingScore,
          max_safe_rom_deg: patient.maxSafeRomDeg,
          universal_health_link_code: patient.universalHealthLinkCode,
          assigned_pt_id: patient.assignedPtId,
          next_appointment_date: patient.nextAppointmentDate || null,
          registration_status: 'active',
          face_biometric_status: faceStatus,
          face_frames_count: framesCaptured,
          face_image_url: storedFaceImageUrl ?? null,
          history: buildPatientHistoryPayload(patient),
        }),
      );
      if (!isPendingSkip && frontalVec && masterVec && resolvedAngles) {
        const templateRows =
          rawFrames.length > 0
            ? rawFrames.map((f) => ({
                patientId,
                patientName: body.fullName!,
                pose: f.pose as BiometricPose,
                vector: f.vector,
                qualityScore: f.qualityScore,
              }))
            : [
                { patientId, patientName: body.fullName!, pose: 'front' as BiometricPose, vector: resolvedAngles.center, qualityScore },
                { patientId, patientName: body.fullName!, pose: 'left' as BiometricPose, vector: resolvedAngles.left, qualityScore },
                { patientId, patientName: body.fullName!, pose: 'right' as BiometricPose, vector: resolvedAngles.right, qualityScore },
                { patientId, patientName: body.fullName!, pose: 'nod' as BiometricPose, vector: resolvedAngles.down, qualityScore },
              ];
        state.face_templates.push(
          {
            patientId,
            patientName: body.fullName,
            pose: 'frontal_centroid',
            vector: frontalVec,
          },
          {
            patientId,
            patientName: body.fullName,
            pose: 'overall_centroid',
            vector: masterVec,
          },
          ...templateRows.map((f) => ({
            patientId,
            patientName: body.fullName,
            pose: f.pose,
            vector: f.vector,
          })),
        );
        await persistToSupabase(supabase, async (db) => {
          await insertFaceEnrollmentWithCompatShotCount(db, {
            patient_id: patientId,
            shot_count: framesCaptured,
            liveness_verified: true,
            enrolled_by: authUser!.id,
          });
          await db.from('face_templates').insert([
            { patient_id: patientId, pose: 'frontal_centroid', embedding: frontalVec, quality_score: qualityScore },
            { patient_id: patientId, pose: 'overall_centroid', embedding: masterVec, quality_score: qualityScore },
            ...templateRows.map((f) => ({
              patient_id: patientId,
              pose: f.pose,
              embedding: f.vector,
              quality_score: f.qualityScore,
            })),
          ]);
          await db.from('patient_biometrics').upsert(
            {
              patient_id: patientId,
              hn: patient.hn,
              master_embedding: masterVec,
              angle_embeddings: resolvedAngles,
              frames_captured: framesCaptured,
              quality_score: qualityScore,
              avatar_url: storedFaceImageUrl ?? null,
              updated_at: new Date().toISOString(),
            },
            { onConflict: 'patient_id' },
          );
        });
      }

      const defaultCourse: PatientCourseRecord = {
        id: newRecordId(Boolean(supabase), 'course'),
        patientId,
        title: `${body.fullName} — แผนฟื้นฟูสมรรถภาพ`,
        weeklySchedule: defaultWeeklySchedule(),
        cognitiveModpackId: 'modpack-core-99',
        updatedAt: new Date().toISOString(),
      };
      state.patient_courses.push(defaultCourse);
      await persistToSupabase(supabase, (db) =>
        db.from('patient_courses').insert({
          id: defaultCourse.id,
          patient_id: defaultCourse.patientId,
          title: defaultCourse.title,
          weekly_schedule: defaultCourse.weeklySchedule,
          cognitive_modpack_id: defaultCourse.cognitiveModpackId,
        }),
      );

      const patUsername = (body.username || body.hn.replace(/[^a-zA-Z0-9]/g, '')).toLowerCase();
      if (!state.users.some((u) => u.username.toLowerCase() === patUsername)) {
        const patUser: StoredUserAccount = {
          id: newRecordId(Boolean(supabase), 'user-pat'),
          username: patUsername,
          passwordHash: await hashPasswordEdge(body.password || 'Patient@2026'),
          role: 'patient',
          fullName: body.fullName,
          phone: body.phone ?? '',
          patientId,
          active: true,
          forceLoggedOutAt: null,
          createdAt: new Date().toISOString(),
        };
        state.users.push(patUser);
        await persistToSupabase(supabase, (db) =>
          db.from('staff_users').insert({
            id: patUser.id,
            username: patUser.username,
            password_hash: patUser.passwordHash,
            full_name: patUser.fullName,
            phone: patUser.phone,
            patient_id: patientId,
            role: 'patient',
            active: true,
          }),
        );
      }

      recordAudit({
        eventType: isPendingSkip ? 'PATIENT_REGISTERED_PENDING_FACE' : 'PATIENT_ENROLLED_120_FACE_ID',
        actorRole: authUser!.role,
        actorId: authUser!.id,
        actorName: authUser!.fullName,
        ipAddress: clientIp,
        deviceId: clientDeviceId,
        targetId: patientId,
        detail: isPendingSkip
          ? `Registered patient ${body.fullName} (${body.hn}) — Face ID pending`
          : `Enrolled patient ${body.fullName} (${body.hn}) with ${framesCaptured}-frame 3D Face ID`,
      });

      return jsonResponse({ patient }, 201);
    }

    if (pathname === '/api/biometrics/index' && method === 'GET') {
      const todayIdx = new Date().getDay();
      const deletedPatientSet = new Set(state.deleted_patient_ids ?? []);
      const inactiveUserPatientIds = new Set(
        state.users.filter((u) => !u.active && u.patientId).map((u) => u.patientId as string),
      );
      const index: PatientBiometricIndexEntry[] = [];
      for (const p of state.patients) {
        if ((p.registrationStatus ?? 'active') !== 'active' || deletedPatientSet.has(p.id) || inactiveUserPatientIds.has(p.id)) {
          continue;
        }
        const tplFront = parseEmbeddingVector(
          state.face_templates.find((t) => t.patientId === p.id && t.pose === 'frontal_centroid')?.vector,
        );
        const tplOverall = parseEmbeddingVector(
          state.face_templates.find((t) => t.patientId === p.id && t.pose === 'overall_centroid')?.vector,
        );
        const rawMaster =
          parseEmbeddingVector(p.masterEmbedding) ??
          tplOverall ??
          tplFront ??
          parseEmbeddingVector(p.biometricHistory?.overall_centroid) ??
          parseEmbeddingVector(p.biometricHistory?.frontal_centroid);
        if (!rawMaster || rawMaster.length === 0) continue;
        const masterNorm = l2Normalize(rawMaster);
        const angleEmbeddings = p.angleEmbeddings
          ? {
              center: parseEmbeddingVector(p.angleEmbeddings.center) ?? masterNorm,
              left: parseEmbeddingVector(p.angleEmbeddings.left) ?? masterNorm,
              right: parseEmbeddingVector(p.angleEmbeddings.right) ?? masterNorm,
              up: parseEmbeddingVector(p.angleEmbeddings.up) ?? masterNorm,
              down: parseEmbeddingVector(p.angleEmbeddings.down) ?? masterNorm,
            }
          : {
              center: tplFront ?? parseEmbeddingVector(p.biometricHistory?.frontal_centroid) ?? masterNorm,
              left:
                parseEmbeddingVector(state.face_templates.find((t) => t.patientId === p.id && t.pose === 'left')?.vector) ??
                masterNorm,
              right:
                parseEmbeddingVector(state.face_templates.find((t) => t.patientId === p.id && t.pose === 'right')?.vector) ??
                masterNorm,
              up:
                parseEmbeddingVector(state.face_templates.find((t) => t.patientId === p.id && t.pose === 'nod')?.vector) ??
                masterNorm,
              down:
                parseEmbeddingVector(state.face_templates.find((t) => t.patientId === p.id && t.pose === 'nod')?.vector) ??
                masterNorm,
            };
        const extraEmbeddings = [
          tplFront,
          tplOverall,
          parseEmbeddingVector(p.biometricHistory?.frontal_centroid),
          parseEmbeddingVector(p.biometricHistory?.overall_centroid),
        ].filter((v): v is number[] => Array.isArray(v) && v.length === rawMaster.length);
        const course = state.patient_courses.find((c) => c.patientId === p.id) ?? null;
        const todayRoutine = course?.weeklySchedule.find((d) => d.dayIndex === todayIdx) ?? null;
        const todaySchedule = resolveTodayPatientSchedule({
          rehabSchedules: p.rehabSchedules,
          rehabSessions: state.rehab_sessions.filter((s) => s.patientId === p.id),
          todayRoutine,
          weeklySchedule: course?.weeklySchedule,
          nextAppointmentDate: p.nextAppointmentDate,
          fallbackNextAppointmentDate: p.nextAppointmentDate,
          exercisesCatalog: state.exercises,
        });
        index.push({
          patientId: p.id,
          hn: p.hn,
          fullName: p.fullName,
          diagnosis: p.diagnosis,
          assignedPtId: p.assignedPtId,
          nextAppointmentDate: todaySchedule.nextAppointmentDate,
          avatarUrl: p.faceImageUrl,
          framesCaptured: p.faceFramesCount ?? 120,
          qualityScore: 0.96,
          masterEmbedding: masterNorm,
          angleEmbeddings,
          ...(p.biometricEngine ? { biometricEngine: p.biometricEngine } : {}),
          ...(p.geometricRatios ? { geometricRatios: p.geometricRatios } : {}),
          ...(extraEmbeddings.length > 0 ? { extraEmbeddings } : {}),
          todayRoutine,
          rehabSchedules: p.rehabSchedules ?? [],
          todayExercises: todaySchedule.todayExercises,
          hasAppointmentToday: todaySchedule.hasAppointmentToday,
          completedToday: todaySchedule.completedToday,
        });
      }
      return jsonResponse({ index });
    }

    if (pathname === '/api/biometric/inspect-frame' && method === 'POST') {
      const body = await readJsonBody<FrameInspectionInput & { requestedPose?: Detected3DPose }>(
        request,
      );
      if (!body) return jsonResponse({ error: 'VALIDATION_ERROR' }, 400);

      const inspection = inspectBiometricFrame3D({
        landmarks3D: body.landmarks3D,
        faceCount: body.faceCount,
        requestedPose: body.requestedPose,
        metrics: body.metrics,
      });

      if (inspection.scan_status === 'SPOOF_DETECTED') {
        recordAudit({
          eventType: 'BIOMETRIC_SPOOF_DETECTED',
          actorRole: authUser?.role ?? 'patient_kiosk',
          actorId: authUser?.id ?? clientDeviceId,
          actorName: authUser?.fullName ?? 'Kiosk Face Sensor',
          ipAddress: clientIp,
          deviceId: clientDeviceId,
          detail: `Blocked spoof attempt (${inspection.error_code})`,
        });
      }

      return jsonResponse(inspection);
    }

    if (pathname === '/api/biometric/verify' && method === 'POST') {
      if (!state.system_settings.aiFaceScanEnabled) {
        return jsonResponse(
          {
            error: 'FACE_SCAN_DISABLED',
            message: 'ระบบสแกนใบหน้าถูกปิดชั่วคราวโดยผู้ดูแลระบบ',
          },
          503,
        );
      }
      const body = await readJsonBody<{
        probeVector?: unknown;
        mirroredProbeVector?: unknown;
        hybridV2ProbeVector?: unknown;
        mirroredHybridV2ProbeVector?: unknown;
        legacyV1ProbeVector?: unknown;
        mirroredLegacyV1ProbeVector?: unknown;
        yawDeg?: number;
        pitchDeg?: number;
        rollDeg?: number;
        engine?: 'arcface-onnx' | 'hybrid-512d';
        geometricRatios?: GeometricAnthropometricRatios;
        excludedPatientIds?: string[];
        livenessEvidence?: {
          biologicalVerified?: boolean;
          blinkCount?: number;
          spoofRisk?: number;
          nonRigidScore?: number;
          rppgScore?: number;
          faceHeightRatio?: number;
          yawDeg?: number;
          pitchDeg?: number;
          rollDeg?: number;
          engine?: 'arcface-onnx' | 'hybrid-512d';
          mirroredProbeVector?: unknown;
          hybridV2ProbeVector?: unknown;
          mirroredHybridV2ProbeVector?: unknown;
          legacyV1ProbeVector?: unknown;
          mirroredLegacyV1ProbeVector?: unknown;
          geometricRatios?: GeometricAnthropometricRatios;
          excludedPatientIds?: string[];
        };
      }>(request);
      if (!body || !isVector512(body.probeVector)) {
        return jsonResponse({ error: 'VALIDATION_ERROR' }, 400);
      }
      if (
        body.livenessEvidence &&
        (body.livenessEvidence.biologicalVerified !== true ||
          (typeof body.livenessEvidence.spoofRisk === 'number' &&
            body.livenessEvidence.spoofRisk >= 0.62))
      ) {
        recordAudit({
          eventType: 'BIOMETRIC_SPOOF_DETECTED',
          actorRole: authUser?.role ?? 'patient_kiosk',
          actorId: authUser?.id ?? clientDeviceId,
          actorName: authUser?.fullName ?? 'Kiosk Face Sensor',
          ipAddress: clientIp,
          deviceId: clientDeviceId,
          detail: `Blocked face login without valid biological liveness (blinkCount=${body.livenessEvidence.blinkCount ?? 0}, spoofRisk=${body.livenessEvidence.spoofRisk ?? 0})`,
        });
        return jsonResponse(
          {
            matched: false,
            confidencePercent: 0,
            error: 'BIOMETRIC_SPOOF_DETECTED',
            failureReasons: ['BIOLOGICAL_SPOOF_BLOCKED'],
            message: 'ตรวจพบภาพถ่าย/หน้าจอ หรือยังไม่ผ่านการยืนยันชีววิทยา (กรุณากะพริบตา)',
          },
          403,
        );
      }

      const deletedPatientSet = new Set(state.deleted_patient_ids ?? []);
      const excludedPatientList =
        body.excludedPatientIds ?? body.livenessEvidence?.excludedPatientIds ?? [];
      const excludedPatientSet = new Set(
        Array.isArray(excludedPatientList) ? excludedPatientList.map(String) : [],
      );
      const inactiveUserPatientIds = new Set(
        state.users.filter((u) => !u.active && u.patientId).map((u) => u.patientId as string),
      );
      const activePatientIds = new Set(
        state.patients
          .filter(
            (p) =>
              (p.registrationStatus ?? 'active') === 'active' &&
              !deletedPatientSet.has(p.id) &&
              !excludedPatientSet.has(p.id) &&
              !excludedPatientSet.has(p.hn) &&
              !inactiveUserPatientIds.has(p.id),
          )
          .map((p) => p.id),
      );

      const probeNorm = l2Normalize(body.probeVector);

      let candidates: EnrolledTemplateCandidate[] = state.face_templates
        .filter((t) => activePatientIds.has(t.patientId))
        .map((t) => ({ ...t, vector: l2Normalize(t.vector) }));

      // Ensure active patients with master/angle embeddings but no face_templates row are still in candidates
      const candidatePatientIds = new Set(candidates.map((c) => c.patientId));
      for (const p of state.patients) {
        if (!activePatientIds.has(p.id) || candidatePatientIds.has(p.id)) continue;
        if (p.angleEmbeddings?.center) {
          candidates.push({
            patientId: p.id,
            patientName: p.fullName,
            pose: 'frontal_centroid',
            vector: l2Normalize(p.angleEmbeddings.center),
          });
        }
        if (p.masterEmbedding) {
          candidates.push({
            patientId: p.id,
            patientName: p.fullName,
            pose: 'overall_centroid',
            vector: l2Normalize(p.masterEmbedding),
          });
        }
      }
      let totalPatients = activePatientIds.size;

      if (supabase && candidates.length === 0 && activePatientIds.size > 0) {
        const { data, error } = await supabase.rpc('match_patient_face', {
          query_embedding: probeNorm,
          match_count: 5,
        });
        if (!error && Array.isArray(data) && data.length > 0) {
          candidates = data
            .filter(
              (row: Record<string, unknown>) =>
                Array.isArray(row.embedding) && activePatientIds.has(String(row.patient_id)),
            )
            .map((row: Record<string, unknown>) => ({
              patientId: String(row.patient_id),
              patientName: typeof row.full_name === 'string' ? row.full_name : undefined,
              pose: (row.pose as BiometricPose) || 'frontal_centroid',
              vector: l2Normalize(row.embedding as number[]),
            }));
          totalPatients = Math.max(1, new Set(candidates.map((c) => c.patientId)).size);
        }
      }

      const probeEngine = detectBiometricVectorEngine(
        probeNorm,
        body.engine ?? body.livenessEvidence?.engine,
      );
      const nonArcFacePatientIds = new Set(
        state.patients
          .filter(
            (p) =>
              activePatientIds.has(p.id) &&
              p.biometricEngine !== 'arcface-onnx',
          )
          .map((p) => p.id),
      );
      const hybridCandidates = candidates.filter((c) => nonArcFacePatientIds.has(c.patientId));
      const result =
        probeEngine === 'arcface-onnx'
          ? {
              matched: false,
              patientId: null,
              patientName: null,
              confidencePercent: 0,
              rawMargin: 0,
              whitenedMargin: 0,
              topCandidate: null,
              runnerUp: null,
              failureReasons: ['ARCFACE_PRIMARY_POSE_MATCHER'],
            }
          : verifyHybridBiometricIdentity(probeNorm, hybridCandidates, {
              totalEnrolledPatients: totalPatients,
              excludedPatientIds: Array.from(excludedPatientSet),
              thresholds: {
                minRawCosMulti: state.system_settings.aiFaceThreshold,
              },
            });

      let matchedPatientId =
        result.matched && result.patientId && activePatientIds.has(result.patientId)
          ? result.patientId
          : null;
      let matchedPatientName = matchedPatientId ? result.patientName : null;
      let matchedConfidence = result.confidencePercent;
      let verifyErrorCode: string | undefined;

      // Engine-Isolated Pose-Synchronized Multi-Angle Matching (ArcFace ResNet-50 + Hybrid/V1 fallback)
      if (!matchedPatientId) {
        const yawDegVal = body.yawDeg ?? body.livenessEvidence?.yawDeg ?? 0;
        const pitchDegVal = body.pitchDeg ?? body.livenessEvidence?.pitchDeg ?? 0;
        const rollDegVal = body.rollDeg ?? body.livenessEvidence?.rollDeg ?? 0;
        const probeRatios = body.geometricRatios ?? body.livenessEvidence?.geometricRatios ?? null;
        const livePoseZone = classifyLivePoseZone(yawDegVal, pitchDegVal);
        const mirroredZone = mirrorPoseZone(livePoseZone);

        const mirroredProbeNorm = isVector512(body.mirroredProbeVector)
          ? l2Normalize(body.mirroredProbeVector)
          : isVector512(body.livenessEvidence?.mirroredProbeVector)
            ? l2Normalize(body.livenessEvidence!.mirroredProbeVector)
            : null;
        const primaryVariants: { zone: FacePoseZone5; vec: number[] }[] = [
          { zone: livePoseZone, vec: probeNorm },
        ];
        if (mirroredProbeNorm) {
          primaryVariants.push({ zone: mirroredZone, vec: mirroredProbeNorm });
        }

        const hybridV2Raw =
          body.hybridV2ProbeVector ?? body.livenessEvidence?.hybridV2ProbeVector;
        const mirroredHybridV2Raw =
          body.mirroredHybridV2ProbeVector ?? body.livenessEvidence?.mirroredHybridV2ProbeVector;
        const hybridV2Norm = isVector512(hybridV2Raw) ? l2Normalize(hybridV2Raw) : null;
        const mirroredHybridV2Norm = isVector512(mirroredHybridV2Raw)
          ? l2Normalize(mirroredHybridV2Raw)
          : null;
        const hybridV2Variants: { zone: FacePoseZone5; vec: number[] }[] = hybridV2Norm
          ? [
              { zone: livePoseZone, vec: hybridV2Norm },
              ...(mirroredHybridV2Norm
                ? [{ zone: mirroredZone, vec: mirroredHybridV2Norm }]
                : []),
            ]
          : probeEngine === 'hybrid-v2'
            ? primaryVariants
            : [];

        const legacyV1Raw =
          body.legacyV1ProbeVector ?? body.livenessEvidence?.legacyV1ProbeVector;
        const mirroredLegacyV1Raw =
          body.mirroredLegacyV1ProbeVector ?? body.livenessEvidence?.mirroredLegacyV1ProbeVector;
        const legacyV1ProbeNorm = isVector512(legacyV1Raw) ? l2Normalize(legacyV1Raw) : null;
        const mirroredLegacyV1Norm = isVector512(mirroredLegacyV1Raw)
          ? l2Normalize(mirroredLegacyV1Raw)
          : null;
        const legacyV1Variants: { zone: FacePoseZone5; vec: number[] }[] = legacyV1ProbeNorm
          ? [
              { zone: livePoseZone, vec: legacyV1ProbeNorm },
              ...(mirroredLegacyV1Norm
                ? [{ zone: mirroredZone, vec: mirroredLegacyV1Norm }]
                : []),
            ]
          : probeEngine === 'legacy-v1'
            ? primaryVariants
            : [];
        const arcFaceVariants: { zone: FacePoseZone5; vec: number[] }[] =
          probeEngine === 'arcface-onnx' ? primaryVariants : [];

        const normalizePatientName = (name: string) => name.trim().toLowerCase().replace(/\s+/g, ' ');
        const scoresByPatient: {
          patientId: string;
          fullName: string;
          orderIdx: number;
          masterNorm: number[];
          dualScore: number;
          engine: 'arcface-onnx' | 'hybrid-v2' | 'legacy-v1';
          isLegacyV1: boolean;
          raw: number;
          meanCentered: number;
          white: number;
          geometricPenalty: number;
          bestCos3D: number;
          bestCraniofacialCos: number;
          bestInnerDctCos: number;
          bestInnerDctWhite: number;
          bestInnerPatchCos: number;
          bestInnerPatchWhite: number;
          rankScore: number;
        }[] = [];

        for (let orderIdx = 0; orderIdx < state.patients.length; orderIdx += 1) {
          const p = state.patients[orderIdx];
          if (!activePatientIds.has(p.id)) continue;

          const tplFront = parseEmbeddingVector(
            state.face_templates.find((t) => t.patientId === p.id && t.pose === 'frontal_centroid')?.vector,
          );
          const tplOverall = parseEmbeddingVector(
            state.face_templates.find((t) => t.patientId === p.id && t.pose === 'overall_centroid')?.vector,
          );
          const parsedMaster =
            parseEmbeddingVector(p.masterEmbedding) ??
            tplOverall ??
            tplFront ??
            parseEmbeddingVector(p.biometricHistory?.overall_centroid) ??
            parseEmbeddingVector(p.biometricHistory?.frontal_centroid);
          if (!parsedMaster || parsedMaster.length !== FACE_EMBEDDING_DIM) continue;
          const masterNorm = l2Normalize(parsedMaster);

          const candEngine = detectBiometricVectorEngine(masterNorm, p.biometricEngine);
          const activeVariants =
            candEngine === 'arcface-onnx'
              ? arcFaceVariants
              : candEngine === 'legacy-v1'
                ? legacyV1Variants.length > 0
                  ? legacyV1Variants
                  : hybridV2Variants.length > 0
                    ? hybridV2Variants
                    : probeEngine !== 'arcface-onnx'
                      ? primaryVariants
                      : []
                : hybridV2Variants.length > 0
                  ? hybridV2Variants
                  : probeEngine !== 'arcface-onnx'
                    ? primaryVariants
                    : [];
          if (activeVariants.length === 0) continue;

          const angleMap: Record<FacePoseZone5, number[] | undefined> = {
            center:
              parseEmbeddingVector(p.angleEmbeddings?.center) ??
              tplFront ??
              parseEmbeddingVector(p.biometricHistory?.frontal_centroid) ??
              masterNorm,
            left:
              parseEmbeddingVector(p.angleEmbeddings?.left) ??
              parseEmbeddingVector(state.face_templates.find((t) => t.patientId === p.id && t.pose === 'left')?.vector) ??
              masterNorm,
            right:
              parseEmbeddingVector(p.angleEmbeddings?.right) ??
              parseEmbeddingVector(state.face_templates.find((t) => t.patientId === p.id && t.pose === 'right')?.vector) ??
              masterNorm,
            up:
              parseEmbeddingVector(p.angleEmbeddings?.up) ??
              parseEmbeddingVector(state.face_templates.find((t) => t.patientId === p.id && t.pose === 'nod')?.vector) ??
              masterNorm,
            down:
              parseEmbeddingVector(p.angleEmbeddings?.down) ??
              parseEmbeddingVector(state.face_templates.find((t) => t.patientId === p.id && t.pose === 'nod')?.vector) ??
              masterNorm,
          };

          let bestDual = -1;
          let bestRaw = 0;
          let bestMeanCentered = 0;
          let bestWhite = 0;
          let bestGeomPenalty = 0;
          let bestCos3D = 0;
          let bestCraniofacialCos = 0;
          let bestInnerDctCos = 0;
          let bestInnerDctWhite = 0;
          let bestInnerPatchCos = 0;
          let bestInnerPatchWhite = 0;

          if (candEngine === 'arcface-onnx') {
            for (const variant of activeVariants) {
              const sameZoneVec = angleMap[variant.zone] ?? masterNorm;
              const sameZoneArc = computeArcFaceBiometricScore(variant.vec, sameZoneVec, {
                probeRatios,
                candidateRatios: p.geometricRatios ?? null,
              });
              const masterArc = computeArcFaceBiometricScore(variant.vec, masterNorm, {
                probeRatios,
                candidateRatios: p.geometricRatios ?? null,
              });
              const arcScore = 0.55 * sameZoneArc.score + 0.45 * masterArc.score;
              if (arcScore > bestDual) {
                bestDual = arcScore;
                bestRaw = Math.max(sameZoneArc.rawCos, masterArc.rawCos);
                bestMeanCentered = bestRaw;
                bestWhite = bestRaw;
                bestGeomPenalty = Math.max(sameZoneArc.geometricPenalty, masterArc.geometricPenalty);
              }
            }
          } else {
            for (const variant of activeVariants) {
              const sameZoneVec = angleMap[variant.zone] ?? masterNorm;
              const sameZoneDual = computeDualFactorBiometricScore(variant.vec, sameZoneVec, {
                probeRatios,
                candidateRatios: p.geometricRatios ?? null,
              });
              const masterDual = computeDualFactorBiometricScore(variant.vec, masterNorm, {
                probeRatios,
                candidateRatios: p.geometricRatios ?? null,
              });
              const dualScore =
                0.55 * sameZoneDual.dualFactorScore + 0.45 * masterDual.dualFactorScore;
              if (dualScore > bestDual) {
                bestDual = dualScore;
                const dom =
                  sameZoneDual.dualFactorScore >= masterDual.dualFactorScore
                    ? sameZoneDual
                    : masterDual;
                bestRaw = dom.rawCos;
                bestMeanCentered = dom.meanCenteredCos;
                bestWhite = Math.max(sameZoneDual.whitenedCos, masterDual.whitenedCos);
                bestGeomPenalty = Math.max(
                  sameZoneDual.geometricPenalty,
                  masterDual.geometricPenalty,
                );
                if (dom.faceCore) {
                  bestCos3D = dom.faceCore.cos3DNormalized;
                  bestCraniofacialCos = dom.faceCore.craniofacialCos;
                  bestInnerDctCos = dom.faceCore.innerDctCos;
                  bestInnerDctWhite = dom.faceCore.innerDctWhitenedCos;
                  bestInnerPatchCos = dom.faceCore.innerPatchCos;
                  bestInnerPatchWhite = dom.faceCore.innerPatchWhitenedCos;
                }
              }
            }
          }

          if (bestDual >= 0) {
            scoresByPatient.push({
              patientId: p.id,
              fullName: p.fullName,
              orderIdx,
              masterNorm,
              dualScore: bestDual,
              engine: candEngine,
              isLegacyV1: candEngine === 'legacy-v1',
              raw: bestRaw,
              meanCentered: bestMeanCentered,
              white: bestWhite,
              geometricPenalty: bestGeomPenalty,
              bestCos3D,
              bestCraniofacialCos,
              bestInnerDctCos,
              bestInnerDctWhite,
              bestInnerPatchCos,
              bestInnerPatchWhite,
              rankScore: bestDual,
            });
          }
        }

        const sortScores = (list: typeof scoresByPatient) =>
          [...list].sort((a, b) => {
            const nameA = normalizePatientName(a.fullName);
            const nameB = normalizePatientName(b.fullName);
            if (nameA && nameA === nameB && Math.abs(b.rankScore - a.rankScore) < 0.035) {
              return b.orderIdx - a.orderIdx;
            }
            return b.rankScore - a.rankScore;
          });

        const isAngledPose =
          livePoseZone !== 'center' ||
          Math.abs(yawDegVal) > 8 ||
          Math.abs(pitchDegVal) > 6 ||
          Math.abs(rollDegVal) > 10;

        // 1. Prioritize pure ArcFace ResNet-50 candidates (isolated from hybrid/legacy margin pool)
        const arcScores = sortScores(scoresByPatient.filter((s) => s.engine === 'arcface-onnx'));
        const arcTop = arcScores[0];
        if (arcTop) {
          const arcTopName = normalizePatientName(arcTop.fullName);
          const arcRunners = arcScores.slice(1).filter((s) => {
            const sName = normalizePatientName(s.fullName);
            return !(arcTopName && sName && sName === arcTopName);
          });
          const arcRunner = arcRunners[0];
          const arcMargin = arcRunner ? arcTop.dualScore - arcRunner.dualScore : 1;
          const arcMinThresh = isAngledPose ? 0.48 : 0.54;
          const arcPassesGate =
            arcTop.dualScore >= arcMinThresh && arcTop.geometricPenalty <= 0.14;
          const arcPassesMargin = arcRunners.length === 0 || arcMargin >= 0.1;

          if (arcPassesGate && !arcPassesMargin) {
            verifyErrorCode = 'AMBIGUOUS_CLOSE_CANDIDATES';
          } else if (arcPassesGate && arcPassesMargin) {
            matchedConfidence = Number(
              Math.min(99.9, Math.max(85.0, 50 + arcTop.dualScore * 58)).toFixed(1),
            );
            matchedPatientId = arcTop.patientId;
            matchedPatientName = arcTop.fullName;
          }
        }

        // 2. Fallback to isolated Hybrid-V2 / Legacy-V1 pool ONLY if no ArcFace candidate matched/collided
        if (!matchedPatientId && !verifyErrorCode) {
          const legacyScores = sortScores(
            scoresByPatient.filter((s) => s.engine !== 'arcface-onnx'),
          );
          const top = legacyScores[0];
          const topName = top ? normalizePatientName(top.fullName) : '';
          const distinctRunners = top
            ? legacyScores.slice(1).filter((s) => {
                const sName = normalizePatientName(s.fullName);
                return !(topName && sName && sName === topName);
              })
            : [];
          const runner = distinctRunners[0];
          const margin = runner ? top.dualScore - runner.dualScore : 1;

          const requiredMargin = 0.08;
          const passesMargin = distinctRunners.length === 0 || margin >= requiredMargin;
          const topIsLegacyV1 = Boolean(top?.isLegacyV1);
          const minDualThresh = topIsLegacyV1
            ? isAngledPose
              ? 0.75
              : 0.80
            : isAngledPose
              ? 0.70
              : 0.75;
          const minWhiteThresh = topIsLegacyV1
            ? isAngledPose
              ? 0.36
              : 0.42
            : isAngledPose
              ? 0.32
              : 0.36;
          const minMeanCentered = topIsLegacyV1
            ? isAngledPose
              ? 0.56
              : 0.64
            : isAngledPose
              ? 0.52
              : 0.60;

          const passesSubbandFloor = Boolean(
            top &&
              top.bestInnerDctCos >=
                (topIsLegacyV1 ? (isAngledPose ? 0.32 : 0.40) : isAngledPose ? 0.34 : 0.42) &&
              top.bestInnerDctWhite >=
                (topIsLegacyV1 ? (isAngledPose ? 0.14 : 0.20) : isAngledPose ? 0.15 : 0.22) &&
              top.bestInnerPatchCos >=
                (topIsLegacyV1 ? (isAngledPose ? 0.26 : 0.34) : isAngledPose ? 0.30 : 0.38) &&
              top.bestInnerPatchWhite >=
                (topIsLegacyV1 ? (isAngledPose ? 0.10 : 0.16) : isAngledPose ? 0.12 : 0.18) &&
              top.bestCos3D >=
                (topIsLegacyV1 ? (isAngledPose ? 0.60 : 0.68) : isAngledPose ? 0.52 : 0.60) &&
              top.bestCraniofacialCos >=
                (topIsLegacyV1 ? (isAngledPose ? 0.28 : 0.36) : isAngledPose ? 0.28 : 0.35),
          );
          const passesDualGate = Boolean(
            top &&
              top.dualScore >= minDualThresh &&
              top.white >= minWhiteThresh &&
              top.meanCentered >= minMeanCentered &&
              top.geometricPenalty <= 0.14 &&
              passesSubbandFloor,
          );

          if (top && passesDualGate && !passesMargin) {
            verifyErrorCode = 'AMBIGUOUS_CLOSE_CANDIDATES';
          }

          if (top && passesDualGate && passesMargin) {
            matchedConfidence = Number(
              Math.min(99.9, Math.max(85.0, top.dualScore * 100)).toFixed(1),
            );
            matchedPatientId = top.patientId;
            matchedPatientName = top.fullName;
          } else if (top || arcTop) {
            const bestScore = Math.max(top?.dualScore ?? 0, arcTop?.dualScore ?? 0);
            matchedConfidence = Math.min(48.9, Math.max(0, Number((bestScore * 49).toFixed(1))));
          }
        }
      }

      if (!matchedPatientId || matchedConfidence < 85.0) {
        return jsonResponse(
          {
            ...result,
            matched: false,
            patientId: undefined,
            patientName: undefined,
            confidencePercent: matchedConfidence,
            ...(verifyErrorCode ? { error: verifyErrorCode, failureReasons: [verifyErrorCode] } : {}),
          },
          401,
        );
      }

      const patient =
        state.patients.find((p) => p.id === matchedPatientId && activePatientIds.has(p.id)) ?? null;
      if (!patient) {
        return jsonResponse(
          {
            ...result,
            matched: false,
            patientId: undefined,
            patientName: undefined,
            reason: 'PATIENT_NOT_FOUND_OR_DELETED',
          },
          401,
        );
      }

      const course = state.patient_courses.find((c) => c.patientId === matchedPatientId) ?? null;
      const todayIdx = new Date().getDay();
      const todayRoutine = course?.weeklySchedule.find((d) => d.dayIndex === todayIdx) ?? null;
      const todaySchedule = resolveTodayPatientSchedule({
        rehabSchedules: patient.rehabSchedules,
        rehabSessions: state.rehab_sessions.filter((s) => s.patientId === patient.id),
        todayRoutine,
        weeklySchedule: course?.weeklySchedule,
        nextAppointmentDate: patient.nextAppointmentDate,
        fallbackNextAppointmentDate: patient.nextAppointmentDate,
        exercisesCatalog: state.exercises,
      });

      recordAudit({
        eventType: 'BIOMETRIC_FACE_VERIFIED',
        actorRole: 'patient_kiosk',
        actorId: matchedPatientId,
        actorName: matchedPatientName ?? patient.fullName,
        ipAddress: clientIp,
        deviceId: clientDeviceId,
        targetId: matchedPatientId,
        detail: `3D Multi-Angle Face ID matched (${matchedConfidence}%)`,
      });

      return jsonResponse({
        ...result,
        matched: true,
        patientId: matchedPatientId,
        patientName: matchedPatientName ?? patient.fullName,
        confidencePercent: matchedConfidence,
        patient: {
          id: patient.id,
          hn: patient.hn,
          fullName: patient.fullName,
          nextAppointmentDate: todaySchedule.nextAppointmentDate,
          strokeType: patient.strokeType,
          affectedSide: patient.affectedSide,
          motorAssessmentScore: patient.motorAssessmentScore,
          adlDressingScore: patient.adlDressingScore,
          maxSafeRomDeg: patient.maxSafeRomDeg,
          universalHealthLinkCode: patient.universalHealthLinkCode,
        },
        todayRoutine,
        rehabSchedules: patient.rehabSchedules ?? [],
        todayExercises: todaySchedule.todayExercises,
        hasAppointmentToday: todaySchedule.hasAppointmentToday,
        completedToday: todaySchedule.completedToday,
      });
    }

    // ========================
    // 4. PATIENTS, DELEGATION & APPEND-ONLY NOTES
    // ========================
    if (pathname === '/api/patients' && method === 'GET') {
      const err = requireAuth(authUser);
      if (err) return err;
      const actor = authUser!;
      let list = state.patients;
      if (actor.role === 'pt') {
        list = list.filter((p) => p.assignedPtId === actor.id || p.delegatedToPtId === actor.id);
      } else if (actor.role === 'patient') {
        list = list.filter((p) => p.id === actor.patientId);
      }
      return jsonResponse({ patients: list });
    }

    const delegateMatch = pathname.match(/^\/api\/patients\/([^/]+)\/delegate$/);
    if (delegateMatch && method === 'POST') {
      const err = requireAuth(authUser, ['super_admin', 'pt']);
      if (err) return err;
      const patient = state.patients.find((p) => p.id === delegateMatch[1]);
      if (!patient) return jsonResponse({ error: 'PATIENT_NOT_FOUND' }, 404);
      const actor = authUser!;
      if (actor.role === 'pt' && patient.assignedPtId !== actor.id && patient.delegatedToPtId !== actor.id) {
        return jsonResponse({ error: 'FORBIDDEN', message: 'Only primary PT can delegate case' }, 403);
      }
      const body =
        (await readJsonBody<{ targetPtId?: string | null; note?: string; delegationDate?: string }>(request)) ?? {};
      patient.delegatedToPtId = body.targetPtId || null;
      patient.delegationNote = body.targetPtId ? (body.note || null) : null;
      patient.delegationDate = body.targetPtId
        ? (body.delegationDate || new Date().toISOString().slice(0, 10))
        : null;
      const serializedDelegationNote = patient.delegatedToPtId
        ? patient.delegationDate
          ? `[${patient.delegationDate}] ${patient.delegationNote ?? ''}`.trim()
          : patient.delegationNote
        : null;
      state.delegation_overrides = state.delegation_overrides ?? {};
      state.delegation_overrides[patient.id] = {
        delegatedToPtId: patient.delegatedToPtId,
        delegationDate: patient.delegationDate ?? null,
        delegationNote: patient.delegationNote ?? null,
      };
      await persistToSupabase(supabase, async (db) => {
        if (patient.delegatedToPtId) {
          const targetPt = state.users.find((u) => u.id === patient.delegatedToPtId);
          if (targetPt) {
            await db.from('staff_users').upsert({
              id: targetPt.id,
              username: targetPt.username,
              password_hash: targetPt.passwordHash,
              full_name: targetPt.fullName,
              phone: targetPt.phone,
              specialty: targetPt.specialty ?? null,
              license_no: targetPt.licenseNo ?? null,
              patient_id: null,
              role: targetPt.role,
              active: targetPt.active,
            });
          }
        }
        const existingRow = await db
          .from('patients')
          .select('history')
          .eq('id', patient.id)
          .maybeSingle();
        const existingHistory =
          existingRow.data?.history && typeof existingRow.data.history === 'object'
            ? { ...(existingRow.data.history as Record<string, unknown>) }
            : {};
        existingHistory.delegatedToPtId = patient.delegatedToPtId;
        existingHistory.delegationDate = patient.delegationDate ?? null;
        existingHistory.delegationNote = patient.delegationNote ?? null;

        const updateRes = await db
          .from('patients')
          .update({
            delegated_to_pt_id: patient.delegatedToPtId,
            delegation_note: serializedDelegationNote,
            history: existingHistory,
            updated_at: new Date().toISOString(),
          })
          .eq('id', patient.id);

        if (updateRes.error) {
          await db
            .from('patients')
            .update({
              delegation_note: serializedDelegationNote,
              history: existingHistory,
              updated_at: new Date().toISOString(),
            })
            .eq('id', patient.id);
        }
      });
      recordAudit({
        eventType: 'CASE_DELEGATED',
        actorRole: actor.role,
        actorId: actor.id,
        actorName: actor.fullName,
        ipAddress: clientIp,
        deviceId: clientDeviceId,
        targetId: patient.id,
        detail: body.targetPtId
          ? `Delegated case ${patient.fullName} to PT ${body.targetPtId} on ${patient.delegationDate}`
          : `Reclaimed case ${patient.fullName}`,
      });
      return jsonResponse({ patient });
    }

    const noteItemMatch = pathname.match(/^\/api\/patients\/([^/]+)\/notes\/([^/]+)$/);
    if (noteItemMatch && method === 'PUT') {
      const err = requireAuth(authUser, ['super_admin', 'pt']);
      if (err) return err;
      if (authUser!.role === 'pt') {
        return jsonResponse(
          {
            error: 'APPEND_ONLY_NOTES',
            message: 'นักกายภาพบำบัดไม่สามารถแก้ไขโน้ตประวัติเก่าที่เคยเขียนไปแล้วได้',
          },
          403,
        );
      }
      const note = state.medical_notes.find((n) => n.id === noteItemMatch[2]);
      if (!note) return jsonResponse({ error: 'NOTE_NOT_FOUND' }, 404);
      const body = await readJsonBody<{ content?: string }>(request);
      note.content = String(body?.content ?? note.content);
      persistToSupabase(supabase, (db) =>
        db.from('medical_notes').update({ content: note.content }).eq('id', note.id),
      );
      return jsonResponse({ note });
    }

    if (noteItemMatch && method === 'DELETE') {
      const err = requireAuth(authUser, ['super_admin', 'pt']);
      if (err) return err;
      if (authUser!.role === 'pt') {
        return jsonResponse(
          {
            error: 'APPEND_ONLY_NOTES',
            message: 'นักกายภาพบำบัดไม่สามารถลบโน้ตประวัติเก่าที่เคยเขียนไปแล้วได้',
          },
          403,
        );
      }
      state.medical_notes = state.medical_notes.filter((n) => n.id !== noteItemMatch[2]);
      persistToSupabase(supabase, (db) => db.from('medical_notes').delete().eq('id', noteItemMatch[2]));
      return jsonResponse({ deleted: true });
    }

    const notesMatch = pathname.match(/^\/api\/patients\/([^/]+)\/notes$/);
    if (notesMatch && method === 'GET') {
      const err = requireAuth(authUser);
      if (err) return err;
      const notes = state.medical_notes.filter((n) => n.patientId === notesMatch[1]);
      return jsonResponse({ notes });
    }

    if (notesMatch && method === 'POST') {
      const err = requireAuth(authUser, ['super_admin', 'pt']);
      if (err) return err;
      const body = await readJsonBody<{ content?: string }>(request);
      const content = body?.content?.trim();
      if (!content) return jsonResponse({ error: 'EMPTY_NOTE' }, 400);
      const note: MedicalNoteRecord = {
        id: `note-${Date.now()}-${randomHex(2)}`,
        patientId: notesMatch[1],
        authorId: authUser!.id,
        authorName: authUser!.fullName,
        content,
        createdAt: new Date().toISOString(),
      };
      state.medical_notes.push(note);
      persistToSupabase(supabase, (db) =>
        db.from('medical_notes').insert({
          patient_id: note.patientId,
          author_id: note.authorId,
          author_name: note.authorName,
          content: note.content,
        }),
      );
      recordAudit({
        eventType: 'MEDICAL_NOTE_APPENDED',
        actorRole: authUser!.role,
        actorId: authUser!.id,
        actorName: authUser!.fullName,
        ipAddress: clientIp,
        deviceId: clientDeviceId,
        targetId: note.patientId,
        detail: `Appended medical note to patient ${note.patientId}`,
      });
      return jsonResponse({ note }, 201);
    }

    const enrollExistingMatch = pathname.match(/^\/api\/patients\/([^/]+)\/enroll-face$/);
    if (enrollExistingMatch && method === 'POST') {
      const err = requireAuth(authUser, ['super_admin', 'pt']);
      if (err) return err;
      const patient = state.patients.find((p) => p.id === enrollExistingMatch[1]);
      if (!patient) return jsonResponse({ error: 'PATIENT_NOT_FOUND' }, 404);
      const body = await readJsonBody<{
        frontalCentroid?: unknown;
        overallCentroid?: unknown;
        rawFrames?: { pose?: string; vector?: unknown; qualityScore?: number }[];
        faceImageUrl?: string;
        faceImageUrls?: unknown;
        angleImageUrls?: unknown;
        masterEmbedding?: unknown;
        angleEmbeddings?: unknown;
        biometricEngine?: 'arcface-onnx' | 'hybrid-512d';
        geometricRatios?: GeometricAnthropometricRatios;
        framesCaptured?: number;
        qualityScore?: number;
      }>(request);
      const valid120 =
        Boolean(body) &&
        isVector512(body?.masterEmbedding) &&
        normalizeAngleEmbeddingsInput(body?.angleEmbeddings) !== null;
      const valid10 =
        Boolean(body) &&
        isVector512(body?.frontalCentroid) &&
        isVector512(body?.overallCentroid) &&
        Array.isArray(body?.rawFrames) &&
        body!.rawFrames.length >= 10;

      if (!body || (!valid10 && !valid120)) {
        return jsonResponse({ error: 'VALIDATION_ERROR' }, 400);
      }

      const rawFrames = (Array.isArray(body.rawFrames) ? body.rawFrames : []) as {
        pose: 'front' | 'left' | 'right' | 'nod';
        vector: number[];
        qualityScore: number;
      }[];

      const resolvedAngles =
        normalizeAngleEmbeddingsInput(body.angleEmbeddings) ??
        buildAngleEmbeddingsFromPayload(
          body.angleEmbeddings,
          isVector512(body.frontalCentroid) ? body.frontalCentroid : (body.masterEmbedding as number[]),
          isVector512(body.overallCentroid) ? body.overallCentroid : (body.masterEmbedding as number[]),
          rawFrames,
        );
      const masterVec = isVector512(body.masterEmbedding)
        ? l2Normalize(body.masterEmbedding)
        : isVector512(body.overallCentroid)
          ? body.overallCentroid
          : resolvedAngles.center;
      const frontalVec = isVector512(body.frontalCentroid) ? body.frontalCentroid : resolvedAngles.center;
      const framesCaptured =
        typeof body.framesCaptured === 'number' && body.framesCaptured > 0
          ? body.framesCaptured
          : Math.max(rawFrames.length, 120);
      const qualityScore = typeof body.qualityScore === 'number' ? body.qualityScore : 0.96;

      patient.faceBiometricStatus = 'VERIFIED';
      patient.faceFramesCount = framesCaptured;
      patient.masterEmbedding = masterVec;
      patient.angleEmbeddings = resolvedAngles;
      patient.biometricEngine = body.biometricEngine;
      if (body.geometricRatios) {
        patient.geometricRatios = body.geometricRatios;
      }
      if (body.faceImageUrl || body.angleImageUrls || body.faceImageUrls) {
        const faceBundle = await uploadMultiAngleFaceBundle(supabase, patient.id, {
          faceImageUrl: body.faceImageUrl,
          angleImageUrls: body.angleImageUrls,
          faceImageUrls: body.faceImageUrls,
        });
        if (faceBundle.primaryFaceImageUrl) {
          patient.faceImageUrl = faceBundle.primaryFaceImageUrl;
        }
        if (faceBundle.angleImageUrls) {
          patient.angleImageUrls = faceBundle.angleImageUrls;
        }
        if (faceBundle.faceImageUrls) {
          patient.faceImageUrls = faceBundle.faceImageUrls;
        }
      }
      patient.biometricHistory = {
        frontal_centroid: frontalVec,
        overall_centroid: masterVec,
        raw_frames: rawFrames,
      };

      const templateRows =
        rawFrames.length > 0
          ? rawFrames
          : [
              { pose: 'front' as const, vector: resolvedAngles.center, qualityScore },
              { pose: 'left' as const, vector: resolvedAngles.left, qualityScore },
              { pose: 'right' as const, vector: resolvedAngles.right, qualityScore },
              { pose: 'nod' as const, vector: resolvedAngles.down, qualityScore },
            ];

      state.face_templates = state.face_templates.filter((t) => t.patientId !== patient.id);
      state.face_templates.push(
        {
          patientId: patient.id,
          patientName: patient.fullName,
          pose: 'frontal_centroid',
          vector: frontalVec,
        },
        {
          patientId: patient.id,
          patientName: patient.fullName,
          pose: 'overall_centroid',
          vector: masterVec,
        },
        ...templateRows.map((f) => ({
          patientId: patient.id,
          patientName: patient.fullName,
          pose: f.pose as BiometricPose,
          vector: f.vector,
        })),
      );

      await persistToSupabase(supabase, async (db) => {
        await db.from('face_templates').delete().eq('patient_id', patient.id);
        await insertFaceEnrollmentWithCompatShotCount(db, {
          patient_id: patient.id,
          shot_count: framesCaptured,
          liveness_verified: true,
          enrolled_by: authUser!.id,
        });
        await db.from('face_templates').insert([
          { patient_id: patient.id, pose: 'frontal_centroid', embedding: frontalVec, quality_score: qualityScore },
          { patient_id: patient.id, pose: 'overall_centroid', embedding: masterVec, quality_score: qualityScore },
          ...templateRows.map((f) => ({
            patient_id: patient.id,
            pose: f.pose,
            embedding: f.vector,
            quality_score: f.qualityScore,
          })),
        ]);
        await db.from('patient_biometrics').upsert(
          {
            patient_id: patient.id,
            hn: patient.hn,
            master_embedding: masterVec,
            angle_embeddings: resolvedAngles,
            frames_captured: framesCaptured,
            quality_score: qualityScore,
            avatar_url: patient.faceImageUrl ?? null,
            updated_at: new Date().toISOString(),
          },
          { onConflict: 'patient_id' },
        );
        await writePatientRowWithFramesFallback(
          db,
          'update',
          {
            face_biometric_status: 'VERIFIED',
            face_frames_count: framesCaptured,
            face_image_url: patient.faceImageUrl ?? null,
            history: buildPatientHistoryPayload(patient),
            updated_at: new Date().toISOString(),
          },
          patient.id,
        );
      });

      recordAudit({
        eventType: 'PATIENT_FACE_ENROLLED',
        actorRole: authUser!.role,
        actorId: authUser!.id,
        actorName: authUser!.fullName,
        ipAddress: clientIp,
        deviceId: clientDeviceId,
        targetId: patient.id,
        detail: `Captured ${framesCaptured}-frame 3D Face ID for ${patient.fullName} (${patient.hn})`,
      });
      return jsonResponse({ patient });
    }

    const checkinMatch = pathname.match(/^\/api\/patients\/([^/]+)\/checkin$/);
    if (checkinMatch && method === 'POST') {
      const err = requireAuth(authUser, ['super_admin', 'pt']);
      if (err) return err;
      const patient = state.patients.find((p) => p.id === checkinMatch[1]);
      if (!patient) return jsonResponse({ error: 'PATIENT_NOT_FOUND' }, 404);
      const body = await readJsonBody<{ note?: string }>(request);
      const todayIso = new Date().toISOString().slice(0, 10);
      patient.nextAppointmentDate = todayIso;

      const existingSchedules = [...(patient.rehabSchedules ?? [])];
      const todayScheduleIdx = existingSchedules.findIndex((s) => s.scheduledDate === todayIso);
      if (todayScheduleIdx >= 0) {
        existingSchedules[todayScheduleIdx] = { ...existingSchedules[todayScheduleIdx], status: 'ACTIVE' };
      } else {
        existingSchedules.push({
          id: `sched-${Date.now()}`,
          patientId: patient.id,
          ptId: patient.delegatedToPtId || patient.assignedPtId,
          scheduledDate: todayIso,
          sets: 2,
          reps: 8,
          status: 'ACTIVE',
          notes: body?.note || 'Quick Check-in at clinic reception',
        });
      }
      patient.rehabSchedules = existingSchedules;

      const session: RehabSessionRecord = {
        id: `sess-checkin-${Date.now()}`,
        patientId: patient.id,
        activityType: 'physio_exercise',
        exerciseId: 'CLINIC_CHECKIN',
        exerciseName: body?.note?.trim() || 'เช็คอินเข้าฝึกกายภาพบำบัด',
        completedSets: 0,
        completedReps: 0,
        accuracyPercent: 100,
        maxRomDeg: 0,
        painStopTriggered: false,
        aiFeedback: `Checked in by ${authUser!.fullName}`,
        kinematicsReplay: [],
        createdAt: new Date().toISOString(),
      };
      state.rehab_sessions.unshift(session);

      await persistToSupabase(supabase, async (db) => {
        await writePatientRowWithFramesFallback(
          db,
          'update',
          {
            next_appointment_date: todayIso,
            face_frames_count: patient.faceFramesCount ?? 0,
            history: buildPatientHistoryPayload(patient),
            updated_at: new Date().toISOString(),
          },
          patient.id,
        );
        await db.from('rehab_sessions').insert({
          patient_id: session.patientId,
          exercise_id: session.exerciseId,
          completed_sets: 0,
          completed_reps: 0,
          accuracy_percent: 100,
          max_rom_deg: 0,
          pain_stop_triggered: false,
          kinematics_replay: [],
        });
      });

      recordAudit({
        eventType: 'PATIENT_QUICK_CHECKIN',
        actorRole: authUser!.role,
        actorId: authUser!.id,
        actorName: authUser!.fullName,
        ipAddress: clientIp,
        deviceId: clientDeviceId,
        targetId: patient.id,
        detail: `Quick check-in for ${patient.fullName} (${patient.hn})`,
      });
      return jsonResponse({ patient, session });
    }

    const patientMatch = pathname.match(/^\/api\/patients\/([^/]+)$/);
    if (patientMatch && method === 'PUT') {
      const err = requireAuth(authUser);
      if (err) return err;
      const patient = state.patients.find((p) => p.id === patientMatch[1]);
      if (!patient) return jsonResponse({ error: 'PATIENT_NOT_FOUND' }, 404);
      const actor = authUser!;
      if (actor.role === 'pt' && patient.assignedPtId !== actor.id && patient.delegatedToPtId !== actor.id) {
        return jsonResponse(
          { error: 'FORBIDDEN', message: 'PT can only edit own assigned patients' },
          403,
        );
      }
      if (actor.role === 'patient' && patient.id !== actor.patientId) {
        return jsonResponse({ error: 'FORBIDDEN' }, 403);
      }
      const body = (await readJsonBody<Partial<PatientRecord>>(request)) ?? {};
      if (actor.role === 'patient') {
        if (body.phone !== undefined) patient.phone = body.phone;
      } else {
        if (body.fullName !== undefined) patient.fullName = body.fullName;
        if (body.age !== undefined) patient.age = body.age;
        if (body.gender !== undefined) patient.gender = body.gender;
        if (body.phone !== undefined) patient.phone = body.phone;
        if (body.diagnosis !== undefined) patient.diagnosis = body.diagnosis;
        if (body.strokeType !== undefined) patient.strokeType = body.strokeType;
        if (body.affectedSide !== undefined) patient.affectedSide = body.affectedSide;
        if (body.motorAssessmentScore !== undefined) {
          patient.motorAssessmentScore = clampScore(body.motorAssessmentScore, 0, 6, patient.motorAssessmentScore ?? 3);
        }
        if (body.adlDressingScore !== undefined) {
          patient.adlDressingScore = clampScore(body.adlDressingScore, 0, 5, patient.adlDressingScore ?? 3);
        }
        if (body.maxSafeRomDeg !== undefined && Number.isFinite(body.maxSafeRomDeg)) {
          patient.maxSafeRomDeg = Math.max(30, Math.min(180, body.maxSafeRomDeg));
        }
        if (body.universalHealthLinkCode !== undefined && body.universalHealthLinkCode.trim()) {
          patient.universalHealthLinkCode = body.universalHealthLinkCode.trim().toUpperCase();
        }
        if (body.nextAppointmentDate !== undefined) {
          patient.nextAppointmentDate = body.nextAppointmentDate;
        }
        if (body.assignedPtId !== undefined && actor.role === 'super_admin') {
          patient.assignedPtId = body.assignedPtId;
        }
        if (body.faceBiometricStatus !== undefined) {
          patient.faceBiometricStatus = body.faceBiometricStatus;
        }
        if (
          body.faceImageUrl !== undefined ||
          body.angleImageUrls !== undefined ||
          body.faceImageUrls !== undefined
        ) {
          const faceBundle = await uploadMultiAngleFaceBundle(supabase, patient.id, {
            faceImageUrl: body.faceImageUrl,
            angleImageUrls: body.angleImageUrls,
            faceImageUrls: body.faceImageUrls,
          });
          if (faceBundle.primaryFaceImageUrl !== undefined) {
            patient.faceImageUrl = faceBundle.primaryFaceImageUrl;
          }
          if (faceBundle.angleImageUrls !== undefined) {
            patient.angleImageUrls = faceBundle.angleImageUrls;
          }
          if (faceBundle.faceImageUrls !== undefined) {
            patient.faceImageUrls = faceBundle.faceImageUrls;
          }
        }
        if (Array.isArray(body.rehabSchedules)) {
          patient.rehabSchedules = body.rehabSchedules as RehabScheduleItem[];
          const todayStr = formatLocalIsoDate(new Date());
          patient.nextAppointmentDate = resolveNextAppointmentDate({
            rehabSchedules: patient.rehabSchedules,
            fallbackNextAppointmentDate:
              typeof body.nextAppointmentDate === 'string'
                ? body.nextAppointmentDate
                : patient.nextAppointmentDate,
            todayIso: todayStr,
          });
        }
      }
      await persistToSupabase(supabase, (db) =>
        writePatientRowWithFramesFallback(
          db,
          'update',
          {
            full_name: patient.fullName,
            age: patient.age,
            phone: patient.phone,
            diagnosis: patient.diagnosis,
            stroke_type: patient.strokeType ?? 'ELDERLY_GENERAL',
            affected_side: patient.affectedSide ?? 'NONE',
            motor_assessment_score: patient.motorAssessmentScore ?? 3,
            adl_dressing_score: patient.adlDressingScore ?? 3,
            max_safe_rom_deg: patient.maxSafeRomDeg ?? 150,
            universal_health_link_code:
              patient.universalHealthLinkCode ?? generateUniversalLinkCode(patient.id),
            assigned_pt_id: patient.assignedPtId,
            next_appointment_date: patient.nextAppointmentDate || null,
            face_biometric_status: patient.faceBiometricStatus ?? 'PENDING',
            face_frames_count: patient.faceFramesCount ?? 0,
            face_image_url: patient.faceImageUrl ?? null,
            history: buildPatientHistoryPayload(patient),
            updated_at: new Date().toISOString(),
          },
          patient.id,
        ),
      );
      recordAudit({
        eventType: 'PATIENT_UPDATED',
        actorRole: actor.role,
        actorId: actor.id,
        actorName: actor.fullName,
        ipAddress: clientIp,
        deviceId: clientDeviceId,
        targetId: patient.id,
        detail: `Updated patient ${patient.fullName}`,
      });
      return jsonResponse({ patient });
    }

    if (patientMatch && method === 'DELETE') {
      const err = requireAuth(authUser, ['super_admin', 'pt']);
      if (err) return err;
      const pid = patientMatch[1];
      let patient = state.patients.find((p) => p.id === pid);
      if (!patient && supabase) {
        await ensureHydrated(true, true);
        patient = state.patients.find((p) => p.id === pid);
      }
      if (!patient) return jsonResponse({ error: 'PATIENT_NOT_FOUND' }, 404);
      const actor = authUser!;
      if (actor.role === 'pt' && patient.assignedPtId !== actor.id) {
        return jsonResponse(
          { error: 'FORBIDDEN', message: 'PT can only delete own assigned patients' },
          403,
        );
      }

      const linkedUsers = state.users.filter(
        (u) => u.patientId === pid || u.username.toLowerCase() === patient.hn.toLowerCase(),
      );
      const linkedUserIds = linkedUsers.map((u) => u.id);
      state.deleted_patient_ids = Array.from(new Set([...(state.deleted_patient_ids ?? []), pid]));
      if (linkedUserIds.length > 0) {
        state.deleted_user_ids = Array.from(new Set([...(state.deleted_user_ids ?? []), ...linkedUserIds]));
      }

      state.patients = state.patients.filter((p) => p.id !== pid);
      state.face_templates = state.face_templates.filter((t) => t.patientId !== pid);
      state.patient_courses = state.patient_courses.filter((c) => c.patientId !== pid);
      state.rehab_sessions = state.rehab_sessions.filter((s) => s.patientId !== pid);
      state.medical_notes = state.medical_notes.filter((n) => n.patientId !== pid);
      state.treatment_reports = state.treatment_reports.filter((r) => r.patientId !== pid);
      state.users = state.users.filter(
        (u) => u.patientId !== pid && u.username.toLowerCase() !== patient.hn.toLowerCase(),
      );
      if (linkedUserIds.length > 0) {
        const linkedIdSet = new Set(linkedUserIds);
        state.active_sessions = state.active_sessions.filter((s) => !linkedIdSet.has(s.userId));
      }

      await persistToSupabase(supabase, async (db) => {
        for (const uid of linkedUserIds) {
          await db.from('active_sessions').delete().eq('user_id', uid);
        }
        await db.from('face_templates').delete().eq('patient_id', pid);
        await db.from('patient_biometrics').delete().eq('patient_id', pid);
        await db.from('face_enrollments').delete().eq('patient_id', pid);
        await db.from('patient_courses').delete().eq('patient_id', pid);
        await db.from('rehab_schedules').delete().eq('patient_id', pid);
        await db.from('rehab_sessions').delete().eq('patient_id', pid);
        await db.from('medical_notes').delete().eq('patient_id', pid);
        await db.from('treatment_reports').delete().eq('patient_id', pid);
        await db.from('staff_users').delete().eq('patient_id', pid);
        await db.from('staff_users').delete().eq('username', patient.hn);
        await db.from('patients').delete().eq('id', pid);
      });

      recordAudit({
        eventType: 'PATIENT_CASCADING_PURGE',
        actorRole: actor.role,
        actorId: actor.id,
        actorName: actor.fullName,
        ipAddress: clientIp,
        deviceId: clientDeviceId,
        targetId: pid,
        detail: `Cascading purge for patient ${patient.fullName} (${patient.hn})`,
      });
      return jsonResponse({ deleted: true, patientId: pid, cascadingPurge: true });
    }

    // ========================
    // 5. COURSES, SESSIONS, BIOMECHANICS & REPORTS
    // ========================
    const courseMatch = pathname.match(/^\/api\/courses\/([^/]+)$/);
    if (courseMatch && method === 'GET') {
      const err = requireAuth(authUser);
      if (err) return err;
      const course = state.patient_courses.find((c) => c.patientId === courseMatch[1]);
      if (!course) return jsonResponse({ error: 'COURSE_NOT_FOUND' }, 404);
      return jsonResponse({ course });
    }

    if (courseMatch && method === 'PUT') {
      const err = requireAuth(authUser, ['super_admin', 'pt']);
      if (err) return err;
      const body = await readJsonBody<{
        title?: string;
        cognitiveModpackId?: string;
        weeklySchedule?: DailyScheduleDay[];
      }>(request);
      if (!body?.title || !body?.cognitiveModpackId || !Array.isArray(body.weeklySchedule)) {
        return jsonResponse({ error: 'VALIDATION_ERROR' }, 400);
      }
      if (
        authUser!.role === 'pt' &&
        !state.system_settings.aiPoseDetectionEnabled &&
        body.weeklySchedule.some((d) => d.dayType === 'training' && d.activityType === 'physio_exercise')
      ) {
        return jsonResponse(
          {
            error: 'PHYSIO_LOCKED_UNDER_DEVELOPMENT',
            message:
              'ท่ากายภาพบำบัดถูกล็อกสถานะ "อยู่ระหว่างพัฒนา" PT สามารถสั่งจ่ายตารางได้เฉพาะ "มินิเกม"',
          },
          403,
        );
      }

      const pid = courseMatch[1];
      const existing = state.patient_courses.find((c) => c.patientId === pid);
      if (existing) {
        existing.title = body.title;
        existing.cognitiveModpackId = body.cognitiveModpackId;
        existing.weeklySchedule = body.weeklySchedule;
        existing.updatedAt = new Date().toISOString();
        await persistToSupabase(supabase, (db) =>
          db
            .from('patient_courses')
            .update({
              title: existing.title,
              cognitive_modpack_id: existing.cognitiveModpackId,
              weekly_schedule: existing.weeklySchedule,
              updated_at: existing.updatedAt,
            })
            .eq('id', existing.id),
        );
        recordAudit({
          eventType: 'COURSE_SCHEDULE_UPDATED',
          actorRole: authUser!.role,
          actorId: authUser!.id,
          actorName: authUser!.fullName,
          ipAddress: clientIp,
          deviceId: clientDeviceId,
          targetId: pid,
          detail: `Updated weekly course schedule "${existing.title}" for patient ${pid}`,
        });
        return jsonResponse({ course: existing });
      }
      const created: PatientCourseRecord = {
        id: newRecordId(Boolean(supabase), 'course'),
        patientId: pid,
        title: body.title,
        cognitiveModpackId: body.cognitiveModpackId,
        weeklySchedule: body.weeklySchedule,
        updatedAt: new Date().toISOString(),
      };
      state.patient_courses.push(created);
      await persistToSupabase(supabase, (db) =>
        db.from('patient_courses').insert({
          id: created.id,
          patient_id: created.patientId,
          title: created.title,
          weekly_schedule: created.weeklySchedule,
          cognitive_modpack_id: created.cognitiveModpackId,
        }),
      );
      recordAudit({
        eventType: 'COURSE_SCHEDULE_UPDATED',
        actorRole: authUser!.role,
        actorId: authUser!.id,
        actorName: authUser!.fullName,
        ipAddress: clientIp,
        deviceId: clientDeviceId,
        targetId: pid,
        detail: `Created weekly course schedule "${created.title}" for patient ${pid}`,
      });
      return jsonResponse({ course: created });
    }

    if (pathname === '/api/sessions' && method === 'GET') {
      const err = requireAuth(authUser);
      if (err) return err;
      const pid = reqUrl.searchParams.get('patientId') ?? undefined;
      let list = state.rehab_sessions;
      if (authUser!.role === 'patient') {
        list = list.filter((s) => s.patientId === authUser!.patientId);
      } else if (pid) {
        list = list.filter((s) => s.patientId === pid);
      }
      return jsonResponse({ sessions: list });
    }

    if (pathname === '/api/sessions' && method === 'POST') {
      await ensureHydrated();
      const body = await readJsonBody<
        Partial<RehabSessionRecord> & { scheduledDate?: string; markDailyCompleted?: boolean }
      >(request);
      if (!body?.patientId || !body?.exerciseId || !body?.exerciseName) {
        return jsonResponse({ error: 'VALIDATION_ERROR' }, 400);
      }
      const matchedPatient = state.patients.find(
        (p) => p.id === body.patientId || p.hn === body.patientId,
      );
      const resolvedPatientId = matchedPatient?.id ?? body.patientId;
      const bodyWithNotes = body as Partial<RehabSessionRecord> & { clinicalNotes?: string };
      const catalogMatch = state.exercises.find(
        (ex) => ex.id === body.exerciseId || ex.code === body.exerciseId,
      );
      const exCode = catalogMatch?.code || body.exerciseId;
      const priorSameExercise = (state.rehab_session_history ?? [])
        .filter((h) => h.patientId === resolvedPatientId && h.exerciseCode === exCode && h.achievedMaxRomDeg > 0)
        .sort((a, b) => new Date(a.sessionDate).getTime() - new Date(b.sessionDate).getTime());
      const achievedMaxRom = Number((body.maxRomDeg ?? 0).toFixed(1));
      const baselineRom =
        typeof body.baselineRomDeg === 'number' && Number.isFinite(body.baselineRomDeg)
          ? body.baselineRomDeg
          : priorSameExercise[0]?.baselineRomDeg ?? achievedMaxRom;
      const romDelta =
        typeof body.romDeltaDeg === 'number' && Number.isFinite(body.romDeltaDeg)
          ? body.romDeltaDeg
          : Number((achievedMaxRom - baselineRom).toFixed(1));
      const ptUser = matchedPatient
        ? state.users.find((u) => u.id === (matchedPatient.delegatedToPtId || matchedPatient.assignedPtId))
        : undefined;
      const recordedPtId = authUser?.role === 'pt' ? authUser.id : ptUser?.id ?? authUser?.id ?? null;
      const recordedPtName =
        body.recordedByPtName ??
        (authUser && (authUser.role === 'pt' || authUser.role === 'super_admin')
          ? authUser.fullName
          : ptUser?.fullName ?? 'กภ. ประจำคลินิก');
      const motorScoreSnap =
        typeof body.motorScoreAtSession === 'number'
          ? body.motorScoreAtSession
          : matchedPatient?.motorAssessmentScore ?? null;
      const facilityName = body.facilityName || 'Dr.Tech.Care Main Clinic';

      const session: RehabSessionRecord = {
        id: `sess-${Date.now()}-${randomHex(2)}`,
        patientId: resolvedPatientId,
        activityType: body.activityType ?? 'minigame',
        exerciseId: body.exerciseId,
        exerciseName: body.exerciseName,
        jointName: body.jointName || catalogMatch?.detection.jointGroup || 'upper_limb',
        bodySide: (body.bodySide ||
          (catalogMatch?.detection.side ? catalogMatch.detection.side.toUpperCase() : matchedPatient?.affectedSide) ||
          'BOTH') as AffectedSide,
        completedSets: body.completedSets ?? 0,
        completedReps: body.completedReps ?? 0,
        accuracyPercent: body.accuracyPercent ?? 100,
        maxRomDeg: achievedMaxRom,
        baselineRomDeg: baselineRom,
        romDeltaDeg: romDelta,
        motorScoreAtSession: motorScoreSnap,
        compensationsCount: body.compensationsCount ?? 0,
        facilityName,
        recordedByPtName: recordedPtName,
        painStopTriggered: body.painStopTriggered ?? false,
        aiFeedback: body.aiFeedback ?? 'Good posture stability observed.',
        kinematicsReplay: body.kinematicsReplay ?? [],
        minigameScore: body.minigameScore,
        createdAt: new Date().toISOString(),
      };
      state.rehab_sessions.unshift(session);

      let historyEntry: RehabSessionHistoryEntry | null = null;
      if (session.activityType === 'physio_exercise' && session.exerciseId !== 'CLINIC_CHECKIN') {
        historyEntry = {
          id: newRecordId(Boolean(supabase), 'hist'),
          patientId: resolvedPatientId,
          hn: matchedPatient?.hn ?? '',
          sessionDate: session.createdAt,
          exerciseCode: exCode,
          exerciseName: session.exerciseName,
          jointName: session.jointName ?? 'upper_limb',
          bodySide: session.bodySide ?? 'BOTH',
          baselineRomDeg: baselineRom,
          achievedMaxRomDeg: achievedMaxRom,
          romDeltaDeg: romDelta,
          motorScoreAtSession: motorScoreSnap,
          validReps: session.completedReps,
          compensationsCount: session.compensationsCount ?? 0,
          facilityName,
          recordedByPtId: recordedPtId,
          recordedByPtName: recordedPtName,
          clinicalNotes: bodyWithNotes.clinicalNotes || session.aiFeedback,
        };
        state.rehab_session_history = state.rehab_session_history ?? [];
        state.rehab_session_history.unshift(historyEntry);
        if (matchedPatient) {
          matchedPatient.rehabSessionHistory = [
            historyEntry,
            ...(matchedPatient.rehabSessionHistory ?? []),
          ].slice(0, 120);
        }
      }

      if (matchedPatient && session.exerciseId !== 'CLINIC_CHECKIN') {
        const utcIso = new Date().toISOString().slice(0, 10);
        const localIso = formatLocalIsoDate(new Date());
        const targetDates = new Set<string>(
          [body.scheduledDate, utcIso, localIso].filter((d): d is string => Boolean(d && d.trim())),
        );
        const existingSchedules = [...(matchedPatient.rehabSchedules ?? [])];
        let matchedAnyToday = false;
        let updatedSchedules = existingSchedules.map((item) => {
          if (!targetDates.has(item.scheduledDate)) return item;
          matchedAnyToday = true;
          if (
            body.markDailyCompleted === true ||
            session.activityType === 'minigame' ||
            item.exerciseId === session.exerciseId ||
            (!item.exerciseId && item.modpackId === session.exerciseId)
          ) {
            return { ...item, status: 'COMPLETED' as const };
          }
          return item;
        });

        const anyActiveLeft = updatedSchedules.some(
          (item) => targetDates.has(item.scheduledDate) && item.status === 'ACTIVE',
        );
        if (!anyActiveLeft && matchedAnyToday) {
          updatedSchedules = updatedSchedules.map((item) =>
            targetDates.has(item.scheduledDate) ? { ...item, status: 'COMPLETED' as const } : item,
          );
        }

        if (!matchedAnyToday && (body.markDailyCompleted === true || session.activityType === 'minigame')) {
          updatedSchedules.push({
            id: `sched-done-${Date.now()}`,
            patientId: matchedPatient.id,
            ptId: matchedPatient.delegatedToPtId || matchedPatient.assignedPtId,
            scheduledDate: body.scheduledDate || localIso || utcIso,
            exerciseId: session.exerciseId,
            sets: Math.max(1, session.completedSets || 1),
            reps: Math.max(1, session.completedReps || 1),
            status: 'COMPLETED',
            notes: 'Completed daily physical therapy at Kiosk',
          });
        }
        matchedPatient.rehabSchedules = updatedSchedules;
        const referenceTodayIso = body.scheduledDate?.trim() || localIso || utcIso;
        const anyActiveLeftToday = updatedSchedules.some(
          (item) => targetDates.has(item.scheduledDate) && item.status === 'ACTIVE',
        );
        matchedPatient.nextAppointmentDate = resolveNextAppointmentDate({
          rehabSchedules: updatedSchedules,
          fallbackNextAppointmentDate: matchedPatient.nextAppointmentDate,
          todayIso: referenceTodayIso,
          completedToday: Boolean(body.markDailyCompleted || session.activityType === 'minigame' || !anyActiveLeftToday),
        });
      }

      await persistToSupabase(supabase, async (db) => {
        await db.from('rehab_sessions').insert({
          patient_id: session.patientId,
          exercise_id: session.exerciseId,
          completed_sets: session.completedSets,
          completed_reps: session.completedReps,
          accuracy_percent: session.accuracyPercent,
          max_rom_deg: session.maxRomDeg,
          pain_stop_triggered: session.painStopTriggered,
          kinematics_replay: session.kinematicsReplay,
          minigame_score: session.minigameScore ?? null,
        });
        if (historyEntry) {
          try {
            await db.from('rehab_session_history').insert({
              id: historyEntry.id,
              patient_id: historyEntry.patientId,
              hn: historyEntry.hn,
              session_date: historyEntry.sessionDate,
              exercise_code: historyEntry.exerciseCode,
              exercise_name: historyEntry.exerciseName,
              joint_name: historyEntry.jointName,
              body_side: historyEntry.bodySide,
              baseline_rom_deg: historyEntry.baselineRomDeg,
              achieved_max_rom_deg: historyEntry.achievedMaxRomDeg,
              rom_delta_deg: historyEntry.romDeltaDeg,
              motor_score_at_session: historyEntry.motorScoreAtSession ?? null,
              valid_reps: historyEntry.validReps,
              compensations_count: historyEntry.compensationsCount,
              facility_name: historyEntry.facilityName,
              recorded_by_pt_id: historyEntry.recordedByPtId ?? null,
              recorded_by_pt_name: historyEntry.recordedByPtName ?? null,
              clinical_notes: historyEntry.clinicalNotes ?? null,
            });
          } catch {
            // Best-effort if table 004 is not yet migrated; patient.history JSONB already preserves it.
          }
        }
        if (matchedPatient && session.exerciseId !== 'CLINIC_CHECKIN') {
          await writePatientRowWithFramesFallback(
            db,
            'update',
            {
              next_appointment_date: matchedPatient.nextAppointmentDate || null,
              face_frames_count: matchedPatient.faceFramesCount ?? 0,
              history: buildPatientHistoryPayload(matchedPatient),
              updated_at: new Date().toISOString(),
            },
            matchedPatient.id,
          );
        }
      });
      return jsonResponse({ session }, 201);
    }

    const sessionHistoryMatch = pathname.match(/^\/api\/patients\/([^/]+)\/session-history$/);
    if (sessionHistoryMatch && method === 'GET') {
      const err = requireAuth(authUser);
      if (err) return err;
      const pid = sessionHistoryMatch[1];
      const patient = state.patients.find((p) => p.id === pid || p.hn === pid);
      const targetId = patient?.id ?? pid;
      if (authUser!.role === 'patient' && authUser!.patientId !== targetId) {
        return jsonResponse({ error: 'FORBIDDEN' }, 403);
      }

      const explicitHistory = (state.rehab_session_history ?? []).filter(
        (h) => h.patientId === targetId,
      );
      const patientHistoryEmbedded = (patient?.rehabSessionHistory ?? []).filter(Boolean);
      const byId = new Map<string, RehabSessionHistoryEntry>();
      for (const h of [...patientHistoryEmbedded, ...explicitHistory]) {
        byId.set(h.id, h);
      }

      // Derive longitudinal timeline entries from any older physio sessions lacking an explicit history row
      const existingDates = new Set(Array.from(byId.values()).map((h) => `${h.exerciseCode}|${h.sessionDate}`));
      const baselineByCode = new Map<string, number>();
      const orderedSessions = state.rehab_sessions
        .filter((s) => s.patientId === targetId && s.activityType === 'physio_exercise' && s.exerciseId !== 'CLINIC_CHECKIN')
        .slice()
        .sort((a, b) => new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime());
      for (const s of orderedSessions) {
        const cat = state.exercises.find((e) => e.id === s.exerciseId || e.code === s.exerciseId);
        const code = cat?.code || s.exerciseId;
        const base = s.baselineRomDeg ?? baselineByCode.get(code) ?? s.maxRomDeg;
        if (!baselineByCode.has(code) && base > 0) baselineByCode.set(code, base);
        const effBase = baselineByCode.get(code) ?? base;
        if (!existingDates.has(`${code}|${s.createdAt}`) && !byId.has(s.id)) {
          byId.set(s.id, {
            id: s.id,
            patientId: targetId,
            hn: patient?.hn ?? '',
            sessionDate: s.createdAt,
            exerciseCode: code,
            exerciseName: s.exerciseName || cat?.name.th || code,
            jointName: s.jointName || cat?.detection.jointGroup || 'upper_limb',
            bodySide: (s.bodySide || patient?.affectedSide || 'BOTH') as AffectedSide,
            baselineRomDeg: effBase,
            achievedMaxRomDeg: s.maxRomDeg,
            romDeltaDeg: s.romDeltaDeg ?? Number((s.maxRomDeg - effBase).toFixed(1)),
            motorScoreAtSession: s.motorScoreAtSession ?? patient?.motorAssessmentScore ?? null,
            validReps: s.completedReps,
            compensationsCount: s.compensationsCount ?? 0,
            facilityName: s.facilityName || 'Dr.Tech.Care Clinic',
            recordedByPtName: s.recordedByPtName || 'กภ. ประจำคลินิก',
            clinicalNotes: s.aiFeedback,
          });
        }
      }

      const history = Array.from(byId.values()).sort(
        (a, b) => new Date(b.sessionDate).getTime() - new Date(a.sessionDate).getTime(),
      );
      return jsonResponse({ history });
    }

    if (pathname === '/api/ehr/lookup' && method === 'GET') {
      const err = requireAuth(authUser, ['super_admin', 'pt']);
      if (err) return err;
      const code = (reqUrl.searchParams.get('code') ?? '').trim().toUpperCase();
      if (!code) return jsonResponse({ error: 'VALIDATION_ERROR' }, 400);
      const matched = state.patients.find(
        (p) =>
          (p.universalHealthLinkCode ?? '').toUpperCase() === code ||
          p.hn.toUpperCase() === code ||
          p.id.toUpperCase() === code,
      );
      if (!matched) {
        return jsonResponse({ ehrPackage: null }, 200);
      }
      const hist = (state.rehab_session_history ?? [])
        .filter((h) => h.patientId === matched.id)
        .sort((a, b) => new Date(b.sessionDate).getTime() - new Date(a.sessionDate).getTime());
      const ehrPackage: UniversalEhrPackage = {
        version: '1.0',
        exportedAt: new Date().toISOString(),
        universalHealthLinkCode: matched.universalHealthLinkCode ?? generateUniversalLinkCode(matched.id),
        facilityName: 'Dr.Tech.Care Main Clinic',
        patientProfile: {
          hn: matched.hn,
          fullName: matched.fullName,
          age: matched.age,
          gender: matched.gender,
          diagnosis: matched.diagnosis,
          strokeType: matched.strokeType,
          affectedSide: matched.affectedSide,
          motorAssessmentScore: matched.motorAssessmentScore,
          adlDressingScore: matched.adlDressingScore,
          maxSafeRomDeg: matched.maxSafeRomDeg,
        },
        sessionHistory: hist,
      };
      return jsonResponse({ ehrPackage });
    }

    const linkEhrMatch = pathname.match(/^\/api\/patients\/([^/]+)\/link-ehr$/);
    if (linkEhrMatch && method === 'POST') {
      const err = requireAuth(authUser, ['super_admin', 'pt']);
      if (err) return err;
      const patient = state.patients.find((p) => p.id === linkEhrMatch[1]);
      if (!patient) return jsonResponse({ error: 'PATIENT_NOT_FOUND' }, 404);
      const body = await readJsonBody<{ linkCode?: string; ehrPackage?: UniversalEhrPackage }>(request);
      if (!body?.linkCode && !body?.ehrPackage) {
        return jsonResponse({ error: 'VALIDATION_ERROR' }, 400);
      }

      let pkg: UniversalEhrPackage | null = body.ehrPackage ?? null;
      if (!pkg && body.linkCode) {
        const cleanCode = body.linkCode.trim().toUpperCase();
        const sourcePatient = state.patients.find(
          (p) =>
            (p.universalHealthLinkCode ?? '').toUpperCase() === cleanCode ||
            p.hn.toUpperCase() === cleanCode,
        );
        if (!sourcePatient) {
          return jsonResponse({ error: 'EHR_LINK_CODE_NOT_FOUND' }, 404);
        }
        const sourceHistory = (state.rehab_session_history ?? []).filter(
          (h) => h.patientId === sourcePatient.id,
        );
        pkg = {
          version: '1.0',
          exportedAt: new Date().toISOString(),
          universalHealthLinkCode:
            sourcePatient.universalHealthLinkCode ?? generateUniversalLinkCode(sourcePatient.id),
          facilityName: 'External Partner Facility',
          patientProfile: {
            hn: sourcePatient.hn,
            fullName: sourcePatient.fullName,
            age: sourcePatient.age,
            gender: sourcePatient.gender,
            diagnosis: sourcePatient.diagnosis,
            strokeType: sourcePatient.strokeType,
            affectedSide: sourcePatient.affectedSide,
            motorAssessmentScore: sourcePatient.motorAssessmentScore,
            adlDressingScore: sourcePatient.adlDressingScore,
            maxSafeRomDeg: sourcePatient.maxSafeRomDeg,
          },
          sessionHistory: sourceHistory,
        };
      }

      if (!pkg) return jsonResponse({ error: 'VALIDATION_ERROR' }, 400);

      // Update Stroke clinical profile defaults if missing or provided externally
      if (pkg.patientProfile.strokeType && (!patient.strokeType || patient.strokeType === 'ELDERLY_GENERAL')) {
        patient.strokeType = pkg.patientProfile.strokeType;
      }
      if (pkg.patientProfile.affectedSide && (!patient.affectedSide || patient.affectedSide === 'NONE')) {
        patient.affectedSide = pkg.patientProfile.affectedSide;
      }
      if (typeof pkg.patientProfile.motorAssessmentScore === 'number') {
        patient.motorAssessmentScore = pkg.patientProfile.motorAssessmentScore;
      }
      if (typeof pkg.patientProfile.maxSafeRomDeg === 'number') {
        patient.maxSafeRomDeg = pkg.patientProfile.maxSafeRomDeg;
      }
      if (!patient.universalHealthLinkCode) {
        patient.universalHealthLinkCode = generateUniversalLinkCode(patient.id);
      }

      const existingIds = new Set(
        (state.rehab_session_history ?? [])
          .filter((h) => h.patientId === patient.id)
          .map((h) => `${h.exerciseCode}|${h.sessionDate}|${h.achievedMaxRomDeg}`),
      );
      const importedRows: RehabSessionHistoryEntry[] = [];
      for (const item of pkg.sessionHistory ?? []) {
        const sig = `${item.exerciseCode}|${item.sessionDate}|${item.achievedMaxRomDeg}`;
        if (existingIds.has(sig)) continue;
        existingIds.add(sig);
        const imported: RehabSessionHistoryEntry = {
          ...item,
          id: newRecordId(Boolean(supabase), 'hist'),
          patientId: patient.id,
          hn: patient.hn,
          facilityName: item.facilityName || pkg.facilityName || 'External Linked Clinic',
          clinicalNotes: item.clinicalNotes
            ? `[Linked from ${pkg.universalHealthLinkCode}] ${item.clinicalNotes}`
            : `[Linked from ${pkg.universalHealthLinkCode}]`,
        };
        importedRows.push(imported);
      }

      state.rehab_session_history = state.rehab_session_history ?? [];
      state.rehab_session_history.unshift(...importedRows);
      patient.rehabSessionHistory = [
        ...importedRows,
        ...(patient.rehabSessionHistory ?? []),
      ].slice(0, 150);

      await persistToSupabase(supabase, async (db) => {
        if (importedRows.length > 0) {
          try {
            await db.from('rehab_session_history').insert(
              importedRows.map((h) => ({
                id: h.id,
                patient_id: h.patientId,
                hn: h.hn,
                session_date: h.sessionDate,
                exercise_code: h.exerciseCode,
                exercise_name: h.exerciseName,
                joint_name: h.jointName,
                body_side: h.bodySide,
                baseline_rom_deg: h.baselineRomDeg,
                achieved_max_rom_deg: h.achievedMaxRomDeg,
                rom_delta_deg: h.romDeltaDeg,
                motor_score_at_session: h.motorScoreAtSession ?? null,
                valid_reps: h.validReps,
                compensations_count: h.compensationsCount,
                facility_name: h.facilityName,
                recorded_by_pt_id: h.recordedByPtId ?? null,
                recorded_by_pt_name: h.recordedByPtName ?? null,
                clinical_notes: h.clinicalNotes ?? null,
              })),
            );
          } catch {
            // Fallback already stored inside patient.history JSONB
          }
        }
        await writePatientRowWithFramesFallback(
          db,
          'update',
          {
            stroke_type: patient.strokeType ?? 'ELDERLY_GENERAL',
            affected_side: patient.affectedSide ?? 'NONE',
            motor_assessment_score: patient.motorAssessmentScore ?? 3,
            max_safe_rom_deg: patient.maxSafeRomDeg ?? 150,
            universal_health_link_code: patient.universalHealthLinkCode,
            face_frames_count: patient.faceFramesCount ?? 0,
            history: buildPatientHistoryPayload(patient),
            updated_at: new Date().toISOString(),
          },
          patient.id,
        );
      });

      recordAudit({
        eventType: 'PATIENT_EHR_LINKED',
        actorRole: authUser!.role,
        actorId: authUser!.id,
        actorName: authUser!.fullName,
        ipAddress: clientIp,
        deviceId: clientDeviceId,
        targetId: patient.id,
        detail: `Linked EHR (${pkg.universalHealthLinkCode}) importing ${importedRows.length} session history entries`,
      });

      const mergedForPatient = (state.rehab_session_history ?? [])
        .filter((h) => h.patientId === patient.id)
        .sort((a, b) => new Date(b.sessionDate).getTime() - new Date(a.sessionDate).getTime());
      return jsonResponse({ patient, importedCount: importedRows.length, history: mergedForPatient });
    }

    if (pathname === '/api/biomechanics/evaluate' && method === 'POST') {
      const body = await readJsonBody<
        BiomechanicsEvaluationPacket & { counterState?: RepCounterState }
      >(request);
      if (!body?.auth_status || !body?.spatial_framing || !body?.exercise_prescription || !body?.live_telemetry) {
        return jsonResponse({ error: 'VALIDATION_ERROR' }, 400);
      }
      const { counterState, ...packet } = body;
      const { decision } = evaluateClinicalBiomechanicsPacket(
        packet,
        counterState ?? { valid_reps: 0, invalid_reps: 0, hold_elapsed_sec: 0 },
      );

      if (decision.kiosk_state === 'EMERGENCY_HALT') {
        recordAudit({
          eventType: 'BIOMECHANICS_EMERGENCY_HALT',
          actorRole: authUser?.role ?? 'patient_kiosk',
          actorId: packet.auth_status.patient_id ?? authUser?.id ?? clientDeviceId,
          actorName: authUser?.fullName ?? 'Kiosk Biomechanics AI Agent',
          ipAddress: clientIp,
          deviceId: clientDeviceId,
          detail: `Emergency halt during ${packet.exercise_prescription.exercise_name ?? packet.exercise_prescription.exercise_id ?? 'rehab'}: ${decision.tts_voice_th}`,
        });
      }

      return jsonResponse(decision);
    }

    if (pathname === '/api/reports' && method === 'GET') {
      const err = requireAuth(authUser);
      if (err) return err;
      const actor = authUser!;
      let list = state.treatment_reports;
      if (actor.role === 'patient') {
        list = list.filter((r) => r.patientId === actor.patientId);
      } else if (actor.role === 'pt') {
        const allowed = new Set(
          state.patients
            .filter((p) => p.assignedPtId === actor.id || p.delegatedToPtId === actor.id)
            .map((p) => p.id),
        );
        list = list.filter((r) => allowed.has(r.patientId));
      }
      return jsonResponse({ reports: list });
    }

    if (pathname === '/api/reports' && method === 'POST') {
      const err = requireAuth(authUser, ['super_admin', 'pt']);
      if (err) return err;
      const body = await readJsonBody<{
        patientId?: string;
        summary?: string;
        aiRecommendation?: string;
        scoreSummary?: string;
      }>(request);
      if (!body?.patientId || !body?.summary) {
        return jsonResponse({ error: 'VALIDATION_ERROR' }, 400);
      }
      const now = new Date().toISOString();
      const report: TreatmentReportRecord = {
        id: newRecordId(Boolean(supabase), 'rep'),
        patientId: body.patientId,
        ptId: authUser!.id,
        ptName: authUser!.fullName,
        summary: body.summary,
        aiRecommendation: body.aiRecommendation ?? 'Continue daily cognitive & guided rehab routine.',
        scoreSummary: body.scoreSummary ?? 'Stable progress',
        createdAt: now,
        updatedAt: now,
      };
      state.treatment_reports.unshift(report);
      persistToSupabase(supabase, (db) =>
        db.from('treatment_reports').insert({
          id: report.id,
          patient_id: report.patientId,
          pt_id: report.ptId,
          pt_name: report.ptName,
          summary: report.summary,
          ai_recommendation: report.aiRecommendation,
          score_summary: report.scoreSummary,
        }),
      );
      return jsonResponse({ report }, 201);
    }

    const reportMatch = pathname.match(/^\/api\/reports\/([^/]+)$/);
    if (reportMatch && method === 'PUT') {
      const err = requireAuth(authUser, ['super_admin', 'pt']);
      if (err) return err;
      const rep = state.treatment_reports.find((r) => r.id === reportMatch[1]);
      if (!rep) return jsonResponse({ error: 'REPORT_NOT_FOUND' }, 404);
      const body = (await readJsonBody<Partial<TreatmentReportRecord>>(request)) ?? {};
      if (body.summary) rep.summary = String(body.summary);
      if (body.aiRecommendation) rep.aiRecommendation = String(body.aiRecommendation);
      if (body.scoreSummary) rep.scoreSummary = String(body.scoreSummary);
      rep.updatedAt = new Date().toISOString();
      persistToSupabase(supabase, (db) =>
        db
          .from('treatment_reports')
          .update({
            summary: rep.summary,
            ai_recommendation: rep.aiRecommendation,
            score_summary: rep.scoreSummary,
            updated_at: rep.updatedAt,
          })
          .eq('id', rep.id),
      );
      return jsonResponse({ report: rep });
    }

    // ========================
    // 6. EXERCISES & MODPACKS
    // ========================
    if (pathname === '/api/exercises' && method === 'GET') {
      const existingIds = new Set(state.exercises.map((e) => e.id));
      const builtinExtras = CLINICAL_EXERCISES.filter((ex) => !existingIds.has(ex.id)).map((ex) => ({
        ...ex,
        createdBy: 'system',
      }));
      const combined = builtinExtras.length > 0 ? [...state.exercises, ...builtinExtras] : state.exercises;
      return jsonResponse({ exercises: combined });
    }

    if (pathname === '/api/exercises' && method === 'POST') {
      const err = requireAuth(authUser, ['super_admin', 'pt']);
      if (err) return err;
      const body = await readJsonBody<ManagedExerciseRecord>(request);
      if (!body?.code || !body?.name) {
        return jsonResponse({ error: 'VALIDATION_ERROR' }, 400);
      }
      const created: ManagedExerciseRecord = {
        ...body,
        id: body.id || `ex-custom-${Date.now()}`,
        createdBy: authUser!.role === 'super_admin' ? 'system' : authUser!.id,
      };
      state.exercises.push(created);
      await persistToSupabase(supabase, (db) =>
        db.from('exercises').upsert({
          id: created.id,
          code: created.code,
          category: created.category,
          name_i18n: created.name,
          instruction_i18n: created.instruction,
          detection_config: created.detection,
          active: true,
        }),
      );
      recordAudit({
        eventType: 'EXERCISE_CREATED',
        actorRole: authUser!.role,
        actorId: authUser!.id,
        actorName: authUser!.fullName,
        ipAddress: clientIp,
        deviceId: clientDeviceId,
        targetId: created.id,
        detail: `Added exercise ${created.code} (${created.name.th})`,
      });
      return jsonResponse({ exercise: created }, 201);
    }

    const exerciseMatch = pathname.match(/^\/api\/exercises\/([^/]+)$/);
    if (exerciseMatch && method === 'DELETE') {
      const err = requireAuth(authUser, ['super_admin', 'pt']);
      if (err) return err;
      const target = state.exercises.find((e) => e.id === exerciseMatch[1]);
      if (!target) return jsonResponse({ error: 'NOT_FOUND' }, 404);
      if (authUser!.role === 'pt' && target.createdBy !== authUser!.id) {
        return jsonResponse(
          {
            error: 'FORBIDDEN_GLOBAL_LIBRARY',
            message: 'ห้ามลบหรือแก้ไขคลังท่าฝึกส่วนกลางของโรงพยาบาล',
          },
          403,
        );
      }
      state.exercises = state.exercises.filter((e) => e.id !== exerciseMatch[1]);
      await persistToSupabase(supabase, (db) => db.from('exercises').delete().eq('id', exerciseMatch[1]));
      recordAudit({
        eventType: 'EXERCISE_DELETED',
        actorRole: authUser!.role,
        actorId: authUser!.id,
        actorName: authUser!.fullName,
        ipAddress: clientIp,
        deviceId: clientDeviceId,
        targetId: exerciseMatch[1],
        detail: `Deleted exercise ${target.code}`,
      });
      return jsonResponse({ deleted: true });
    }

    if (pathname === '/api/modpacks' && method === 'GET') {
      return jsonResponse({ modpacks: state.modpacks });
    }

    if (pathname === '/api/modpacks' && method === 'POST') {
      const err = requireAuth(authUser, ['super_admin', 'pt']);
      if (err) return err;
      const body = await readJsonBody<{ name?: string; categories?: string[]; questionCount?: number }>(
        request,
      );
      if (!body?.name) return jsonResponse({ error: 'VALIDATION_ERROR' }, 400);
      const created: ModpackRecord = {
        id: `modpack-${Date.now()}`,
        name: body.name,
        categories: body.categories ?? ['food_fruit', 'safety_routine'],
        questionCount: body.questionCount ?? 20,
        createdBy: authUser!.role === 'super_admin' ? 'system' : authUser!.id,
        active: true,
      };
      state.modpacks.push(created);
      await persistToSupabase(supabase, (db) =>
        db.from('modpacks').upsert({
          id: created.id,
          name: created.name,
          categories: created.categories,
          questions: Array.from({ length: created.questionCount }, (_, i) => ({ index: i + 1 })),
          active: true,
        }),
      );
      recordAudit({
        eventType: 'MODPACK_CREATED',
        actorRole: authUser!.role,
        actorId: authUser!.id,
        actorName: authUser!.fullName,
        ipAddress: clientIp,
        deviceId: clientDeviceId,
        targetId: created.id,
        detail: `Created modpack "${created.name}" (${created.questionCount} cards)`,
      });
      return jsonResponse({ modpack: created }, 201);
    }

    const modpackMatch = pathname.match(/^\/api\/modpacks\/([^/]+)$/);
    if (modpackMatch && method === 'DELETE') {
      const err = requireAuth(authUser, ['super_admin', 'pt']);
      if (err) return err;
      const target = state.modpacks.find((m) => m.id === modpackMatch[1]);
      if (!target) return jsonResponse({ error: 'NOT_FOUND' }, 404);
      if (authUser!.role === 'pt' && target.createdBy !== authUser!.id) {
        return jsonResponse(
          {
            error: 'FORBIDDEN_GLOBAL_LIBRARY',
            message: 'ห้ามลบหรือแก้ไขชุดมินิเกมส่วนกลางของโรงพยาบาล',
          },
          403,
        );
      }
      state.modpacks = state.modpacks.filter((m) => m.id !== modpackMatch[1]);
      await persistToSupabase(supabase, (db) => db.from('modpacks').delete().eq('id', modpackMatch[1]));
      recordAudit({
        eventType: 'MODPACK_DELETED',
        actorRole: authUser!.role,
        actorId: authUser!.id,
        actorName: authUser!.fullName,
        ipAddress: clientIp,
        deviceId: clientDeviceId,
        targetId: modpackMatch[1],
        detail: `Deleted modpack "${target.name}"`,
      });
      return jsonResponse({ deleted: true });
    }

    // ========================
    // 7. STAFF CALLS, SECURITY, AUDIT LOGS & SETTINGS
    // ========================
    const isUuid = (val: string | null | undefined): val is string =>
      Boolean(val && /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(val));

    if (pathname === '/api/staff-calls' && method === 'GET') {
      const err = requireAuth(authUser, ['super_admin', 'pt']);
      if (err) return err;
      return jsonResponse({ calls: sortNewestFirst(state.staff_calls) });
    }

    if (pathname === '/api/staff-calls' && method === 'POST') {
      const body = await readJsonBody<{ kioskId?: string; patientId?: string | null; reason?: string }>(
        request,
      );
      const cleanReason = body?.reason?.trim();
      if (!cleanReason || cleanReason.length > 240) {
        return jsonResponse({ error: 'VALIDATION_ERROR' }, 400);
      }
      const kioskId = (body?.kioskId || 'KIOSK-OPD-01').trim() || 'KIOSK-OPD-01';
      const rawPatientRef = body?.patientId?.trim() || null;
      const matchedPatient = rawPatientRef
        ? state.patients.find((p) => p.id === rawPatientRef || p.hn === rawPatientRef)
        : undefined;
      const resolvedPatientId = matchedPatient?.id ?? rawPatientRef;
      const dbPatientId = isUuid(resolvedPatientId) ? resolvedPatientId : null;
      const callId = newRecordId(Boolean(supabase), 'call');
      const call: StaffCallRecord = {
        id: callId,
        kioskId,
        patientId: resolvedPatientId,
        reason: cleanReason,
        status: 'open',
        createdAt: new Date().toISOString(),
      };
      state.deleted_staff_call_ids = (state.deleted_staff_call_ids ?? []).filter((id) => id !== call.id);
      state.staff_calls.unshift(call);
      await persistToSupabase(supabase, (db) =>
        db.from('staff_calls').insert({
          id: call.id,
          kiosk_id: call.kioskId,
          patient_id: dbPatientId,
          reason: call.reason,
          status: call.status,
        }),
      );
      recordAudit({
        eventType: 'KIOSK_STAFF_CALL',
        actorRole: authUser ? authUser.role : 'patient_kiosk',
        actorId: authUser ? authUser.id : kioskId,
        actorName: authUser ? authUser.fullName : kioskId,
        ipAddress: clientIp,
        deviceId: clientDeviceId,
        targetId: call.id,
        detail: cleanReason,
      });
      return jsonResponse({ call }, 201);
    }

    const staffCallAckMatch = pathname.match(/^\/api\/staff-calls\/([^/]+)\/acknowledge$/);
    if (staffCallAckMatch && method === 'PATCH') {
      const err = requireAuth(authUser, ['super_admin', 'pt']);
      if (err) return err;
      const callId = decodeURIComponent(staffCallAckMatch[1]);
      let target = state.staff_calls.find((c) => c.id === callId);
      if (!target) {
        target = {
          id: callId,
          kioskId: 'KIOSK-OPD-01',
          patientId: null,
          reason: 'คำขอความช่วยเหลือจากตู้ Kiosk',
          status: 'acknowledged',
          createdAt: new Date().toISOString(),
        };
        state.staff_calls.unshift(target);
      }
      target.status = 'acknowledged';
      target.reason = target.reason.replace(/^\[ACKNOWLEDGED\]\s*/, '');
      const ackReason = `[ACKNOWLEDGED] ${target.reason}`;
      await persistToSupabase(supabase, (db) =>
        db.from('staff_calls').update({ status: 'open', reason: ackReason }).eq('id', target!.id),
      );
      recordAudit({
        eventType: 'STAFF_CALL_ACKNOWLEDGED',
        actorRole: authUser!.role,
        actorId: authUser!.id,
        actorName: authUser!.fullName,
        ipAddress: clientIp,
        deviceId: clientDeviceId,
        targetId: target.id,
        detail: `Acknowledged Kiosk call from ${target.kioskId}`,
      });
      return jsonResponse({ ok: true, call: target });
    }

    const staffCallResolveMatch = pathname.match(/^\/api\/staff-calls\/([^/]+)\/resolve$/);
    if (staffCallResolveMatch && method === 'PATCH') {
      const err = requireAuth(authUser, ['super_admin', 'pt']);
      if (err) return err;
      const callId = decodeURIComponent(staffCallResolveMatch[1]);
      let target = state.staff_calls.find((c) => c.id === callId);
      if (!target) {
        target = {
          id: callId,
          kioskId: 'KIOSK-OPD-01',
          patientId: null,
          reason: 'คำขอความช่วยเหลือจากตู้ Kiosk',
          status: 'resolved',
          createdAt: new Date().toISOString(),
        };
        state.staff_calls.unshift(target);
      }
      target.status = 'resolved';
      target.reason = target.reason.replace(/^\[ACKNOWLEDGED\]\s*/, '');
      await persistToSupabase(supabase, (db) =>
        db.from('staff_calls').update({ status: 'resolved', reason: target!.reason }).eq('id', target!.id),
      );
      recordAudit({
        eventType: 'STAFF_CALL_RESOLVED',
        actorRole: authUser!.role,
        actorId: authUser!.id,
        actorName: authUser!.fullName,
        ipAddress: clientIp,
        deviceId: clientDeviceId,
        targetId: target.id,
        detail: `Resolved Kiosk call from ${target.kioskId}`,
      });
      return jsonResponse({ ok: true, call: target });
    }

    const staffCallDeleteMatch = pathname.match(/^\/api\/staff-calls\/([^/]+)$/);
    if (staffCallDeleteMatch && method === 'DELETE') {
      const err = requireAuth(authUser, ['super_admin', 'pt']);
      if (err) return err;
      const callId = decodeURIComponent(staffCallDeleteMatch[1]);
      state.staff_calls = state.staff_calls.filter((c) => c.id !== callId);
      state.deleted_staff_call_ids = Array.from(new Set([...(state.deleted_staff_call_ids ?? []), callId]));
      await persistToSupabase(supabase, (db) =>
        db.from('staff_calls').delete().eq('id', callId),
      );
      recordAudit({
        eventType: 'STAFF_CALL_DELETED',
        actorRole: authUser!.role,
        actorId: authUser!.id,
        actorName: authUser!.fullName,
        ipAddress: clientIp,
        deviceId: clientDeviceId,
        targetId: callId,
        detail: `Removed staff call record ${callId}`,
      });
      return jsonResponse({ ok: true });
    }

    if (pathname === '/api/audit-logs' && method === 'GET') {
      const err = requireAuth(authUser, ['super_admin']);
      if (err) return err;
      return jsonResponse({
        logs: sortNewestFirst(state.audit_logs),
        bannedTargets: sortNewestFirst(state.banned_targets),
        activeSessions: sortNewestFirst(state.active_sessions.filter((s) => !s.revoked)),
      });
    }

    if (pathname === '/api/security/ban' && method === 'POST') {
      const err = requireAuth(authUser, ['super_admin']);
      if (err) return err;
      const body = await readJsonBody<{ targetType?: 'ip' | 'device'; value?: string; reason?: string }>(
        request,
      );
      if ((body?.targetType !== 'ip' && body?.targetType !== 'device') || !body?.value) {
        return jsonResponse({ error: 'VALIDATION_ERROR' }, 400);
      }
      const ban: BannedTargetRecord = {
        id: newRecordId(Boolean(supabase), 'ban'),
        targetType: body.targetType,
        value: body.value.trim(),
        reason: body.reason?.trim() || 'Banned by Hospital Super Admin',
        bannedBy: authUser!.fullName,
        createdAt: new Date().toISOString(),
      };
      state.banned_targets.unshift(ban);
      persistToSupabase(supabase, (db) =>
        db.from('banned_targets').insert({
          id: ban.id,
          target_type: ban.targetType,
          value: ban.value,
          reason: ban.reason,
          banned_by: ban.bannedBy,
        }),
      );
      recordAudit({
        eventType: 'SECURITY_TARGET_BANNED',
        actorRole: authUser!.role,
        actorId: authUser!.id,
        actorName: authUser!.fullName,
        ipAddress: clientIp,
        deviceId: clientDeviceId,
        detail: `Banned ${body.targetType}: ${body.value}`,
      });
      return jsonResponse({ ban }, 201);
    }

    const banDeleteMatch = pathname.match(/^\/api\/security\/ban\/([^/]+)$/);
    if (banDeleteMatch && method === 'DELETE') {
      const err = requireAuth(authUser, ['super_admin']);
      if (err) return err;
      const banId = decodeURIComponent(banDeleteMatch[1]);
      state.deleted_ban_ids = Array.from(new Set([...(state.deleted_ban_ids ?? []), banId]));
      state.banned_targets = state.banned_targets.filter((b) => b.id !== banId);
      await persistToSupabase(supabase, (db) =>
        db.from('banned_targets').delete().eq('id', banId),
      );
      recordAudit({
        eventType: 'SECURITY_TARGET_UNBANNED',
        actorRole: authUser!.role,
        actorId: authUser!.id,
        actorName: authUser!.fullName,
        ipAddress: clientIp,
        deviceId: clientDeviceId,
        targetId: banId,
        detail: `Removed ban ${banId}`,
      });
      return jsonResponse({ ok: true });
    }

    if (pathname === '/api/kiosks' && method === 'POST') {
      const err = requireAuth(authUser, ['super_admin']);
      if (err) return err;
      const body = await readJsonBody<{ kioskId?: string; location?: string }>(request);
      if (!body?.kioskId?.trim()) return jsonResponse({ error: 'VALIDATION_ERROR' }, 400);
      const cleanId = body.kioskId.trim().toUpperCase();
      const pairingToken = `KPAIR-${randomHex(3).toUpperCase()}`;
      const kiosk = {
        kioskId: cleanId,
        location: body.location?.trim() || 'OPD Rehab Zone',
        online: true,
        currentState: 'STANDBY',
        pairingToken,
        lastHeartbeatAt: new Date().toISOString(),
      };
      state.kiosks = state.kiosks.filter((k) => k.kioskId !== cleanId);
      state.kiosks.push(kiosk);
      await persistToSupabase(supabase, (db) =>
        db.from('kiosks').upsert({
          kiosk_id: kiosk.kioskId,
          location: kiosk.location,
          online: true,
          current_state: `STANDBY:${pairingToken}`,
          last_heartbeat_at: kiosk.lastHeartbeatAt,
        }),
      );
      recordAudit({
        eventType: 'KIOSK_REGISTERED',
        actorRole: authUser!.role,
        actorId: authUser!.id,
        actorName: authUser!.fullName,
        ipAddress: clientIp,
        deviceId: clientDeviceId,
        targetId: kiosk.kioskId,
        detail: `Registered Kiosk ${kiosk.kioskId} (${kiosk.location})`,
      });
      return jsonResponse({ kiosk }, 201);
    }

    const kioskDeleteMatch = pathname.match(/^\/api\/kiosks\/([^/]+)$/);
    if (kioskDeleteMatch && method === 'DELETE') {
      const err = requireAuth(authUser, ['super_admin']);
      if (err) return err;
      const kid = decodeURIComponent(kioskDeleteMatch[1]);
      state.kiosks = state.kiosks.filter((k) => k.kioskId !== kid);
      await persistToSupabase(supabase, (db) => db.from('kiosks').delete().eq('kiosk_id', kid));
      recordAudit({
        eventType: 'KIOSK_REMOVED',
        actorRole: authUser!.role,
        actorId: authUser!.id,
        actorName: authUser!.fullName,
        ipAddress: clientIp,
        deviceId: clientDeviceId,
        targetId: kid,
        detail: `Removed Kiosk ${kid}`,
      });
      return jsonResponse({ deleted: true });
    }

    const forceLogoutMatch = pathname.match(/^\/api\/security\/force-logout\/([^/]+)$/);
    if (forceLogoutMatch && method === 'POST') {
      const err = requireAuth(authUser, ['super_admin']);
      if (err) return err;
      const user = state.users.find((u) => u.id === forceLogoutMatch[1]);
      if (!user) return jsonResponse({ error: 'USER_NOT_FOUND' }, 404);
      user.forceLoggedOutAt = new Date().toISOString();
      for (const s of state.active_sessions) {
        if (s.userId === user.id) s.revoked = true;
      }
      await persistToSupabase(supabase, async (db) => {
        await db
          .from('staff_users')
          .update({ force_logged_out_at: user.forceLoggedOutAt })
          .eq('id', user.id);
        await db.from('active_sessions').update({ revoked: true }).eq('user_id', user.id);
      });
      recordAudit({
        eventType: 'FORCE_LOGOUT',
        actorRole: authUser!.role,
        actorId: authUser!.id,
        actorName: authUser!.fullName,
        ipAddress: clientIp,
        deviceId: clientDeviceId,
        targetId: user.id,
        detail: `Forced logout for @${user.username}`,
      });
      return jsonResponse({ ok: true });
    }

    if (pathname === '/api/settings' && method === 'GET') {
      return jsonResponse({ settings: state.system_settings, kiosks: state.kiosks });
    }

    if (pathname === '/api/settings' && method === 'PUT') {
      const err = requireAuth(authUser, ['super_admin']);
      if (err) return err;
      const patch = (await readJsonBody<Partial<SystemSettingsRecord>>(request)) ?? {};
      state.system_settings = {
        ...state.system_settings,
        ...patch,
      };
      await persistToSupabase(supabase, (db) =>
        db
          .from('system_settings')
          .upsert({
            id: 1,
            ai_face_scan_enabled: state.system_settings.aiFaceScanEnabled,
            ai_face_threshold: state.system_settings.aiFaceThreshold,
            ai_pose_detection_enabled: state.system_settings.aiPoseDetectionEnabled,
            kiosk_auto_reset_sec: state.system_settings.kioskAutoResetSec,
            updated_at: new Date().toISOString(),
          }),
      );
      recordAudit({
        eventType: 'SYSTEM_SETTINGS_UPDATED',
        actorRole: authUser!.role,
        actorId: authUser!.id,
        actorName: authUser!.fullName,
        ipAddress: clientIp,
        deviceId: clientDeviceId,
        detail: JSON.stringify(state.system_settings),
      });
      return jsonResponse({ settings: state.system_settings });
    }

    return jsonResponse({ error: 'NOT_FOUND', path: pathname }, 404);
  };
}
