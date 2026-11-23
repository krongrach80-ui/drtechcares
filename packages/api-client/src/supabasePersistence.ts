import type { SupabaseClient } from '@supabase/supabase-js';
import {
  CLINICAL_EXERCISES,
  isDeprecatedExerciseRecord,
  mergeDuplicateSideExercises,
  sortExercisesStrokeUpperFirst,
} from '@drtechcares/pose-engine';
import {
  ACTIVE_BIOMETRIC_DIM,
  ACTIVE_BIOMETRIC_PIPELINE_VERSION,
  parseEmbeddingVector,
  type BiometricPose,
  type EnrolledTemplateCandidate,
  type GeometricAnthropometricRatios,
} from '@drtechcares/smart-biometric-core';
import {
  resolveTodayPatientSchedule,
  sanitizeAccidentalDefaultKneePrefix,
  type ActiveSessionRecord,
  type AffectedSide,
  type ClinicalAssessmentRecord,
  AngleEmbeddings5D,
  AuditLogRecord,
  BannedTargetRecord,
  DailyScheduleDay,
  KioskStatusRecord,
  ManagedExerciseRecord,
  MedicalNoteRecord,
  ModpackRecord,
  PatientCourseRecord,
  PatientRecord,
  RehabScheduleItem,
  RehabSessionHistoryEntry,
  RehabSessionRecord,
  StaffCallRecord,
  type StrokeType,
  SystemSettingsRecord,
  TreatmentReportRecord,
  UserAccount,
  UserRole,
} from './index';
import {
  inspectPatientRecordHydration,
  schedulePatientTopLevelRepairInBackground,
} from './patientNormalization';

export interface StoredUserAccount extends UserAccount {
  passwordHash: string;
}

export const SEEDED_DEFAULT_USER_IDS = new Set([
  '00000000-0000-4000-8000-000000000001',
  '00000000-0000-4000-8000-000000000002',
  '00000000-0000-4000-8000-000000000003',
  'user-super-1',
  'user-pt-1',
  'user-pt-2',
]);

export interface DelegationOverrideEntry {
  delegatedToPtId: string | null;
  delegationDate: string | null;
  delegationNote: string | null;
}

export interface CloudRouterState {
  users: StoredUserAccount[];
  deleted_user_ids?: string[];
  deleted_patient_ids?: string[];
  deleted_ban_ids?: string[];
  deleted_staff_call_ids?: string[];
  deleted_assessment_ids?: string[];
  filtered_out_template_patients_count?: number;
  delegation_overrides?: Record<string, DelegationOverrideEntry>;
  patients: PatientRecord[];
  face_templates: EnrolledTemplateCandidate[];
  medical_notes: MedicalNoteRecord[];
  clinical_assessments?: ClinicalAssessmentRecord[];
  patient_courses: PatientCourseRecord[];
  rehab_sessions: RehabSessionRecord[];
  rehab_session_history?: RehabSessionHistoryEntry[];
  treatment_reports: TreatmentReportRecord[];
  exercises: ManagedExerciseRecord[];
  modpacks: ModpackRecord[];
  staff_calls: StaffCallRecord[];
  audit_logs: AuditLogRecord[];
  banned_targets: BannedTargetRecord[];
  active_sessions: ActiveSessionRecord[];
  system_settings: SystemSettingsRecord;
  kiosks: KioskStatusRecord[];
}

export function normalizeStaffRole(rawRole: unknown): UserRole {
  const r = String(rawRole ?? 'pt').trim().toLowerCase();
  if (r === 'super_admin' || r === 'admin' || r === 'director') return 'super_admin';
  if (r === 'patient') return 'patient';
  return 'pt';
}

export function newRecordId(useUuid: boolean, prefix: string): string {
  if (useUuid && typeof globalThis.crypto?.randomUUID === 'function') {
    return globalThis.crypto.randomUUID();
  }
  const bytes = new Uint8Array(4);
  globalThis.crypto.getRandomValues(bytes);
  const hex = Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('');
  return `${prefix}-${Date.now()}-${hex}`;
}

export const SUPABASE_CIRCUIT_COOLDOWN_MS = 20 * 1000;
export const MAX_DB_IMAGE_STRING_LENGTH = 4000;
export const HYDRATE_TTL_MS = 60_000;

/**
 * Permanent per-table cache of unmigrated columns that returned Postgres 42703 / PostgREST PGRST204.
 * Once a missing column is observed, ALL subsequent SELECT / INSERT / UPDATE / UPSERT calls for
 * that table strip it BEFORE executing the query so 42703 retry storms never repeat.
 */
export const unsupportedColumnsByTable = new Map<string, Set<string>>();

export function resetUnsupportedColumnsForTests(): void {
  unsupportedColumnsByTable.clear();
}

export function parseUndefinedColumnError(error: unknown): string | null {
  if (!error) return null;
  const rawObj = typeof error === 'object' ? (error as Record<string, unknown>) : null;
  const msg = rawObj ? String(rawObj.message ?? '') : String(error);
  const code = rawObj ? String(rawObj.code ?? '') : '';
  const m =
    msg.match(/column\s+(?:[a-zA-Z0-9_]+\.)?["']?([a-zA-Z0-9_]+)["']?\s+does not exist/i) ||
    msg.match(/Could not find the ['"]?([a-zA-Z0-9_]+)['"]? column/i);
  if (m?.[1]) return m[1];
  if (code === '42703' || code === 'PGRST204') {
    const quoted = msg.match(/['"]([a-zA-Z0-9_]+)['"]/);
    if (quoted?.[1]) return quoted[1];
  }
  return null;
}

export function markUnsupportedColumnForTable(table: string, column: string): void {
  const cleanCol = column.trim();
  if (!table || !cleanCol) return;
  let set = unsupportedColumnsByTable.get(table);
  if (!set) {
    set = new Set<string>();
    unsupportedColumnsByTable.set(table, set);
  }
  set.add(cleanCol);
}

export function recordUnsupportedColumn(table: string, error: unknown): string | null {
  const col = parseUndefinedColumnError(error);
  if (!col) return null;
  markUnsupportedColumnForTable(table, col);
  return col;
}

export function stripUnsupportedColumnsForWrite<T extends Record<string, unknown>>(
  table: string,
  payload: T,
): Record<string, unknown> {
  const unsupported = unsupportedColumnsByTable.get(table);
  if (!unsupported || unsupported.size === 0) return payload;
  const cleaned: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(payload)) {
    if (!unsupported.has(key)) {
      cleaned[key] = value;
    }
  }
  return cleaned;
}

export function stripUnsupportedColumnsForWriteRows(
  table: string,
  rows: ReadonlyArray<Record<string, unknown>>,
): Record<string, unknown>[] {
  const unsupported = unsupportedColumnsByTable.get(table);
  if (!unsupported || unsupported.size === 0) return rows.map((row) => ({ ...row }));
  return rows.map((row) => stripUnsupportedColumnsForWrite(table, row));
}

export function filterSelectColsForTable(table: string, colsCsv: string): string {
  const cols = colsCsv
    .split(',')
    .map((c) => c.trim())
    .filter(Boolean);
  const unsupported = unsupportedColumnsByTable.get(table);
  if (!unsupported || unsupported.size === 0) return cols.join(',');
  const filtered = cols.filter((c) => !unsupported.has(c));
  return (filtered.length > 0 ? filtered : cols.slice(0, 1)).join(',');
}

const lastHydratedAtByState = new WeakMap<CloudRouterState, number>();
const hydrationInFlightByState = new WeakMap<CloudRouterState, Promise<void>>();

function isCloudflareWorkerRuntime(): boolean {
  return (
    typeof navigator !== 'undefined' &&
    typeof navigator.userAgent === 'string' &&
    navigator.userAgent.includes('Cloudflare-Workers')
  );
}

export function getLastHydratedAt(state: CloudRouterState): number {
  return lastHydratedAtByState.get(state) ?? 0;
}

export function invalidateHydrationCache(state: CloudRouterState): void {
  lastHydratedAtByState.delete(state);
}

export function resetHydrationCacheForTests(state?: CloudRouterState): void {
  if (state) {
    lastHydratedAtByState.delete(state);
    hydrationInFlightByState.delete(state);
  }
}

interface CircuitRecord {
  downUntil: number;
  reason: string | null;
}

const supabaseCircuitByClient = new WeakMap<object, CircuitRecord>();
let globalSupabaseCircuit: CircuitRecord = { downUntil: 0, reason: null };

export function isSupabaseTransientFailure(errOrStatus: unknown): boolean {
  if (typeof errOrStatus === 'number') {
    return (
      errOrStatus === 401 ||
      errOrStatus === 500 ||
      errOrStatus === 502 ||
      errOrStatus === 503 ||
      errOrStatus === 504 ||
      errOrStatus === 520 ||
      errOrStatus === 522 ||
      errOrStatus === 524 ||
      errOrStatus === 544
    );
  }
  if (!errOrStatus) return false;
  const rawObj = typeof errOrStatus === 'object' ? (errOrStatus as Record<string, unknown>) : null;
  const statusNum = rawObj && typeof rawObj.status === 'number' ? rawObj.status : null;
  if (statusNum !== null && isSupabaseTransientFailure(statusNum)) {
    return true;
  }
  const msg = String(
    rawObj ? `${rawObj.code ?? ''} ${rawObj.message ?? ''} ${rawObj.details ?? ''} ${rawObj.hint ?? ''}` : errOrStatus,
  ).toLowerCase();
  return (
    msg.includes('dbconnectiontimeout') ||
    msg.includes('connection terminated') ||
    msg.includes('544') ||
    msg.includes('522') ||
    msg.includes('504') ||
    msg.includes('502') ||
    msg.includes('503') ||
    msg.includes('timeout') ||
    msg.includes('timed out') ||
    msg.includes('aborterror') ||
    msg.includes('aborted') ||
    msg.includes('fetch failed') ||
    msg.includes('network') ||
    msg.includes('57014') ||
    msg.includes('08006') ||
    msg.includes('08001') ||
    msg.includes('53300') ||
    msg.includes('upstream 5')
  );
}

export function markSupabaseDown(
  reason = 'supabase_unavailable',
  client?: object | null,
  cooldownMs = SUPABASE_CIRCUIT_COOLDOWN_MS,
): void {
  const next: CircuitRecord = { downUntil: Date.now() + cooldownMs, reason };
  if (client && typeof client === 'object') {
    supabaseCircuitByClient.set(client, next);
  } else {
    globalSupabaseCircuit = next;
  }
}

export function isSupabaseCircuitOpen(client?: object | null, now = Date.now()): boolean {
  if (client && typeof client === 'object') {
    const rec = supabaseCircuitByClient.get(client);
    if (!rec) return false;
    if (rec.downUntil > now) return true;
    supabaseCircuitByClient.delete(client);
    return false;
  }
  if (globalSupabaseCircuit.downUntil > now) return true;
  globalSupabaseCircuit = { downUntil: 0, reason: null };
  return false;
}

export function getSupabaseCircuitState(
  client?: object | null,
  now = Date.now(),
): { open: boolean; downUntil: number; remainingMs: number; reason: string | null } {
  const rec =
    client && typeof client === 'object'
      ? supabaseCircuitByClient.get(client)
      : globalSupabaseCircuit;
  if (!rec || rec.downUntil <= now) {
    return { open: false, downUntil: 0, remainingMs: 0, reason: null };
  }
  return {
    open: true,
    downUntil: rec.downUntil,
    remainingMs: Math.max(0, rec.downUntil - now),
    reason: rec.reason,
  };
}

export function clearSupabaseCircuit(client?: object | null): void {
  if (client && typeof client === 'object') {
    supabaseCircuitByClient.delete(client);
  }
  globalSupabaseCircuit = { downUntil: 0, reason: null };
}

export function resetSupabaseCircuitForTests(client?: object | null): void {
  clearSupabaseCircuit(client);
}

/**
 * Resets the circuit breaker (`downUntil = 0`) and executes a fast, bounded
 * `SELECT id FROM patients LIMIT 1` probe (up to `timeoutMs`, default 1500ms)
 * so health checks recover immediately after a Supabase restart.
 */
export async function probeSupabaseHealth(
  supabase: SupabaseClient | null,
  timeoutMs = 1500,
): Promise<{ ok: boolean; latencyMs: number; reason?: string }> {
  if (!supabase) {
    return { ok: false, latencyMs: 0, reason: 'no_supabase_client' };
  }
  clearSupabaseCircuit(supabase);
  const startedAt = Date.now();
  const boundedMs = Math.max(100, Math.min(timeoutMs, 1500));
  let timerId: ReturnType<typeof setTimeout> | undefined;
  try {
    const outcome = await Promise.race([
      Promise.resolve(supabase.from('patients').select('id').limit(1)).then(
        (res) => ({ timedOut: false as const, res }),
        (err) => ({ timedOut: false as const, err }),
      ),
      new Promise<{ timedOut: true }>((resolve) => {
        timerId = setTimeout(() => resolve({ timedOut: true }), boundedMs);
      }),
    ]);
    const latencyMs = Date.now() - startedAt;
    if (outcome.timedOut) {
      markSupabaseDown('health_ping_timeout', supabase);
      return { ok: false, latencyMs, reason: 'health_ping_timeout' };
    }
    if ('err' in outcome && outcome.err) {
      const msg = outcome.err instanceof Error ? outcome.err.message : String(outcome.err);
      if (!isSupabaseCircuitOpen(supabase)) {
        markSupabaseDown(`health_ping_error:${msg}`, supabase);
      }
      return { ok: false, latencyMs, reason: msg };
    }
    const rawRes = 'res' in outcome ? (outcome.res as { error?: { message?: string } | null } | null) : null;
    if (rawRes && typeof rawRes === 'object' && rawRes.error) {
      const msg = rawRes.error.message || JSON.stringify(rawRes.error);
      if (!isSupabaseCircuitOpen(supabase)) {
        markSupabaseDown(`health_ping_postgrest:${msg}`, supabase);
      }
      return { ok: false, latencyMs, reason: msg };
    }
    return { ok: true, latencyMs };
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    if (!isSupabaseCircuitOpen(supabase)) {
      markSupabaseDown(`health_ping_exception:${msg}`, supabase);
    }
    return { ok: false, latencyMs: Date.now() - startedAt, reason: msg };
  } finally {
    if (timerId !== undefined) clearTimeout(timerId);
  }
}

/** Zero-Base64 DB sanitizer: strips any data:image/* string > maxLen (default 4000 chars) before Postgres writes. */
export function stripOversizedBase64ForDb<T>(value: T, maxLen = MAX_DB_IMAGE_STRING_LENGTH): T {
  if (typeof value === 'string') {
    const trimmed = value.trim();
    if (/^data:image\//i.test(trimmed) && trimmed.length > maxLen) {
      return null as unknown as T;
    }
    if (/^data:[^;]+;base64,/i.test(trimmed) && trimmed.length > maxLen) {
      return null as unknown as T;
    }
    return value;
  }
  if (Array.isArray(value)) {
    const mapped: unknown[] = [];
    for (const item of value) {
      if (typeof item === 'string' && /^data:(image\/|[^;]+;base64,)/i.test(item.trim()) && item.trim().length > maxLen) {
        continue;
      }
      mapped.push(stripOversizedBase64ForDb(item, maxLen));
    }
    return mapped as unknown as T;
  }
  if (value && typeof value === 'object') {
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
      if (v === undefined) continue;
      out[k] = stripOversizedBase64ForDb(v, maxLen);
    }
    return out as T;
  }
  return value;
}

/** Safely apply `.limit(maxRows)` on real PostgREST builders without breaking minimal unit mocks. */
function withQueryLimit(query: unknown, maxRows: number): PromiseLike<unknown> {
  const q = query as { limit?: (n: number) => PromiseLike<unknown> } | null;
  if (q && typeof q.limit === 'function') {
    return q.limit(maxRows);
  }
  return Promise.resolve(query);
}

/**
 * Run PostgREST query with explicit columns (never SELECT *).
 * Strips permanently cached missing columns BEFORE executing; if Postgres 42703 / PGRST204 occurs,
 * records the missing column into `unsupportedColumnsByTable` and jumps directly to `safeFallbackCols`
 * in at most 1 retry (never loops 5 times per call).
 */
async function selectExplicitWithFallback(
  table: string,
  runQuery: (cols: string) => PromiseLike<unknown>,
  initialCols: string,
  safeFallbackCols = initialCols,
  maxRows = 50,
): Promise<unknown> {
  const firstCols = filterSelectColsForTable(table, initialCols);
  const res = (await withQueryLimit(runQuery(firstCols), maxRows)) as
    | { data?: unknown; error?: { code?: string; message?: string } | null }
    | undefined;
  if (!res?.error) return res;

  const missingCol = recordUnsupportedColumn(table, res.error);
  if (!missingCol) return res;

  const fallbackCols = filterSelectColsForTable(table, safeFallbackCols);
  if (!fallbackCols || fallbackCols === firstCols) {
    const strippedInitial = filterSelectColsForTable(table, firstCols);
    if (!strippedInitial || strippedInitial === firstCols) {
      return res;
    }
    const retryRes = (await withQueryLimit(runQuery(strippedInitial), maxRows)) as
      | { data?: unknown; error?: { code?: string; message?: string } | null }
      | undefined;
    if (retryRes?.error) {
      recordUnsupportedColumn(table, retryRes.error);
    }
    return retryRes;
  }

  const fallbackRes = (await withQueryLimit(runQuery(fallbackCols), maxRows)) as
    | { data?: unknown; error?: { code?: string; message?: string } | null }
    | undefined;
  if (fallbackRes?.error) {
    recordUnsupportedColumn(table, fallbackRes.error);
  }
  return fallbackRes;
}

function rejectedBudget(reason: string): PromiseSettledResult<unknown> {
  return { status: 'rejected', reason: new Error(reason) };
}

/** Run one priority stage of PostgREST queries within a hard deadline while preserving fulfilled results. */
async function settleStageInBatches(
  tasks: ReadonlyArray<() => PromiseLike<unknown>>,
  batchSize: number,
  deadlineMs: number,
): Promise<PromiseSettledResult<unknown>[]> {
  const results: PromiseSettledResult<unknown>[] = [];
  for (let i = 0; i < tasks.length; i += batchSize) {
    const remainingMs = deadlineMs - Date.now();
    if (remainingMs <= 40) {
      for (let j = i; j < tasks.length; j += 1) {
        results.push(rejectedBudget('hydration_budget_exceeded'));
      }
      break;
    }
    const chunk = tasks.slice(i, i + batchSize);
    const wrappedChunk = chunk.map((fn) => {
      let timerId: ReturnType<typeof setTimeout> | undefined;
      return Promise.race([
        Promise.resolve(fn()).then(
          (val) => ({ status: 'fulfilled' as const, value: val }),
          (err) => ({ status: 'rejected' as const, reason: err }),
        ),
        new Promise<PromiseSettledResult<unknown>>((resolve) => {
          timerId = setTimeout(() => resolve(rejectedBudget('hydration_stage_timeout')), remainingMs);
        }),
      ]).finally(() => {
        if (timerId !== undefined) clearTimeout(timerId);
      });
    });
    const chunkRes = await Promise.all(wrappedChunk);
    results.push(...chunkRes);
  }
  return results;
}

export const STAFF_PROFILE_STORAGE_BUCKET = 'patient-faces';
export const STAFF_PROFILE_MANIFEST_PATH = 'staff-profiles/manifest.json';

export interface StaffProfileCloudRecord {
  id: string;
  username: string;
  avatarUrl?: string;
  bio?: string;
  updatedAt: string;
}

function normalizeStaffProfileManifest(raw: unknown): Record<string, StaffProfileCloudRecord> {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return {};
  const out: Record<string, StaffProfileCloudRecord> = {};
  for (const [key, val] of Object.entries(raw as Record<string, unknown>)) {
    if (!val || typeof val !== 'object' || Array.isArray(val)) continue;
    const rec = val as Record<string, unknown>;
    const id = typeof rec.id === 'string' ? rec.id.trim() : '';
    const username = typeof rec.username === 'string' ? rec.username.trim() : '';
    const avatarUrl = typeof rec.avatarUrl === 'string' && rec.avatarUrl.trim() ? rec.avatarUrl.trim() : undefined;
    const bio = typeof rec.bio === 'string' ? rec.bio : undefined;
    const updatedAt = typeof rec.updatedAt === 'string' && rec.updatedAt.trim() ? rec.updatedAt.trim() : '';
    if (!id && !username) continue;
    const normalized: StaffProfileCloudRecord = {
      id: id || key,
      username,
      avatarUrl,
      bio,
      updatedAt,
    };
    out[key] = normalized;
    if (id) out[id] = normalized;
    if (username) out[`usr:${username.toLowerCase()}`] = normalized;
  }
  return out;
}

async function decodeStorageBlobText(data: unknown): Promise<string | null> {
  if (typeof data === 'string') return data;
  if (!data || typeof data !== 'object') return null;
  if ('text' in data && typeof (data as { text?: unknown }).text === 'function') {
    return await (data as { text: () => Promise<string> }).text();
  }
  if ('arrayBuffer' in data && typeof (data as { arrayBuffer?: unknown }).arrayBuffer === 'function') {
    const buf = await (data as { arrayBuffer: () => Promise<ArrayBuffer> }).arrayBuffer();
    return new TextDecoder().decode(new Uint8Array(buf));
  }
  if (data instanceof Uint8Array) {
    return new TextDecoder().decode(data);
  }
  return null;
}

/**
 * Fetches cross-device staff profile metadata (avatarUrl + bio) stored in Supabase Storage.
 * Guarantees instant cross-device profile photo sync even if Postgres `staff_users` has not
 * yet run migration 011 (`avatar_url`, `bio` columns). Never throws or trips circuit breaker.
 */
export async function fetchStaffProfileManifestFromStorage(
  supabase: SupabaseClient | null,
  timeoutMs = 900,
): Promise<Record<string, StaffProfileCloudRecord>> {
  if (
    !supabase ||
    !supabase.storage ||
    typeof supabase.storage.from !== 'function' ||
    isSupabaseCircuitOpen(supabase)
  ) {
    return {};
  }
  const bucket = supabase.storage.from(STAFF_PROFILE_STORAGE_BUCKET) as {
    download?: (path: string) => Promise<{ data: unknown; error: unknown }>;
  };
  if (typeof bucket?.download !== 'function') {
    return {};
  }
  let timerId: ReturnType<typeof setTimeout> | undefined;
  try {
    const outcome = await Promise.race([
      Promise.resolve(bucket.download(STAFF_PROFILE_MANIFEST_PATH)).then(
        (res) => ({ timedOut: false as const, res }),
        () => ({ timedOut: false as const, res: null }),
      ),
      new Promise<{ timedOut: true }>((resolve) => {
        timerId = setTimeout(() => resolve({ timedOut: true }), timeoutMs);
      }),
    ]);
    if (outcome.timedOut || !outcome.res || outcome.res.error || !outcome.res.data) {
      return {};
    }
    const text = await decodeStorageBlobText(outcome.res.data);
    if (!text) return {};
    const parsed = JSON.parse(text) as unknown;
    return normalizeStaffProfileManifest(parsed);
  } catch {
    return {};
  } finally {
    if (timerId !== undefined) clearTimeout(timerId);
  }
}

/**
 * Read-merge-writes a staff member's latest avatarUrl and bio into the shared Supabase Storage
 * manifest (`staff-profiles/manifest.json`) so all devices and Worker isolates see updates immediately.
 */
export async function persistStaffProfileToStorageManifest(
  supabase: SupabaseClient | null,
  record: StaffProfileCloudRecord,
  timeoutMs = 2000,
): Promise<boolean> {
  if (
    !supabase ||
    !supabase.storage ||
    typeof supabase.storage.from !== 'function' ||
    isSupabaseCircuitOpen(supabase)
  ) {
    return false;
  }
  const bucket = supabase.storage.from(STAFF_PROFILE_STORAGE_BUCKET) as {
    upload?: (
      path: string,
      fileBody: Uint8Array,
      opts?: { contentType?: string; upsert?: boolean; cacheControl?: string },
    ) => Promise<{ data?: unknown; error?: { message?: string } | null }>;
  };
  if (typeof bucket?.upload !== 'function') {
    return false;
  }
  let timerId: ReturnType<typeof setTimeout> | undefined;
  try {
    const run = async (): Promise<boolean> => {
      const existing = await fetchStaffProfileManifestFromStorage(supabase, Math.min(800, timeoutMs));
      const cleanRecord: StaffProfileCloudRecord = {
        id: record.id,
        username: record.username,
        ...(record.avatarUrl ? { avatarUrl: record.avatarUrl } : {}),
        ...(record.bio !== undefined ? { bio: record.bio } : {}),
        updatedAt: record.updatedAt || new Date().toISOString(),
      };
      const merged: Record<string, StaffProfileCloudRecord> = {
        ...existing,
        [cleanRecord.id]: cleanRecord,
        ...(cleanRecord.username
          ? { [`usr:${cleanRecord.username.toLowerCase()}`]: cleanRecord }
          : {}),
      };
      const bytes = new TextEncoder().encode(JSON.stringify(merged));
      let uploadRes = await bucket.upload!(STAFF_PROFILE_MANIFEST_PATH, bytes, {
        contentType: 'application/json',
        upsert: true,
        cacheControl: '0',
      });
      if (uploadRes?.error) {
        const msgLower = String(uploadRes.error.message ?? '').toLowerCase();
        if (
          (msgLower.includes('not found') || msgLower.includes('bucket')) &&
          typeof supabase.storage.createBucket === 'function'
        ) {
          await supabase.storage.createBucket(STAFF_PROFILE_STORAGE_BUCKET, { public: true }).catch(() => {});
          uploadRes = await bucket.upload!(STAFF_PROFILE_MANIFEST_PATH, bytes, {
            contentType: 'application/json',
            upsert: true,
            cacheControl: '0',
          });
        }
      }
      return !uploadRes?.error;
    };
    return await Promise.race([
      run(),
      new Promise<boolean>((resolve) => {
        timerId = setTimeout(() => resolve(false), timeoutMs);
      }),
    ]);
  } catch {
    return false;
  } finally {
    if (timerId !== undefined) clearTimeout(timerId);
  }
}

/** Merge authoritative staff_users DB rows + cloud profile manifest into state without resurrecting deleted seeded PT accounts. */
export function mergeStaffUsersFromDbRows(
  rows: Record<string, unknown>[],
  state: CloudRouterState,
  profileManifest?: Record<string, StaffProfileCloudRecord>,
): void {
  const deletedLower = new Set((state.deleted_user_ids ?? []).map((x) => x.toLowerCase()));
  const prevById = new Map(state.users.map((u) => [u.id, u]));
  const prevByUsername = new Map(state.users.map((u) => [u.username.toLowerCase(), u]));
  const dbUsers: StoredUserAccount[] = rows
    .map((r: Record<string, unknown>) => {
      const id = String(r.id);
      const username = String(r.username);
      const prev = prevById.get(id) ?? prevByUsername.get(username.toLowerCase());
      const manifestRec =
        profileManifest?.[id] ?? profileManifest?.[`usr:${username.toLowerCase()}`];
      const dbAvatar =
        typeof r.avatar_url === 'string' && r.avatar_url.trim() ? r.avatar_url.trim() : '';
      const cloudAvatar =
        typeof manifestRec?.avatarUrl === 'string' && manifestRec.avatarUrl.trim()
          ? manifestRec.avatarUrl.trim()
          : '';
      const dbBio = typeof r.bio === 'string' && r.bio.trim() ? r.bio.trim() : '';
      const cloudBio =
        typeof manifestRec?.bio === 'string' && manifestRec.bio.trim()
          ? manifestRec.bio.trim()
          : '';
      const dbUpdatedMs = typeof r.updated_at === 'string' ? Date.parse(r.updated_at) || 0 : 0;
      const cloudUpdatedMs =
        typeof manifestRec?.updatedAt === 'string' ? Date.parse(manifestRec.updatedAt) || 0 : 0;
      const resolvedAvatar =
        cloudAvatar && (!dbAvatar || cloudUpdatedMs >= dbUpdatedMs)
          ? cloudAvatar
          : dbAvatar || cloudAvatar || prev?.avatarUrl;
      const resolvedBio =
        cloudBio && (!dbBio || cloudUpdatedMs >= dbUpdatedMs)
          ? cloudBio
          : dbBio || cloudBio || prev?.bio;
      return {
        id,
        username,
        passwordHash: String(r.password_hash ?? ''),
        role: normalizeStaffRole(r.role),
        fullName: String(r.full_name ?? ''),
        phone: String(r.phone ?? ''),
        specialty: r.specialty ? String(r.specialty) : undefined,
        licenseNo: r.license_no
          ? String(r.license_no)
          : r.license_number
            ? String(r.license_number)
            : undefined,
        avatarUrl: resolvedAvatar,
        bio: resolvedBio,
        patientId: r.patient_id ? String(r.patient_id) : undefined,
        active:
          r.active !== undefined && r.active !== null
            ? Boolean(r.active)
            : String(r.status ?? 'ACTIVE').toUpperCase() !== 'SUSPENDED',
        forceLoggedOutAt: r.force_logged_out_at ? String(r.force_logged_out_at) : null,
        createdAt: String(r.created_at ?? new Date().toISOString()),
      };
    })
    .filter((u) => !deletedLower.has(u.id.toLowerCase()) && !deletedLower.has(u.username.toLowerCase()));
  const dbIds = new Set(dbUsers.map((u) => u.id));
  const dbUsernames = new Set(dbUsers.map((u) => u.username.toLowerCase()));
  // Authoritative Supabase staff list: NEVER resurrect deleted seeded PT accounts (pt_somchai / pt_waranya)
  // across fresh Worker isolates; keep emergency super_admin bootstrap ONLY if DB has zero active admins.
  const hasSuperAdminInDb = dbUsers.some((u) => u.role === 'super_admin' && u.active);
  const bootstrapAdminOnly = hasSuperAdminInDb
    ? []
    : state.users.filter(
        (u) =>
          u.role === 'super_admin' &&
          SEEDED_DEFAULT_USER_IDS.has(u.id) &&
          !deletedLower.has(u.id.toLowerCase()) &&
          !deletedLower.has(u.username.toLowerCase()) &&
          !dbIds.has(u.id) &&
          !dbUsernames.has(u.username.toLowerCase()),
      );
  state.users = [...dbUsers, ...bootstrapAdminOnly];
}

// Exact base schema columns matching supabase/migrations/20261113000001_drtechcares_schema_and_rpc.sql
// Extended staff columns (avatar_url, bio) are selected opportunistically with automatic fallback to base cols.
const STAFF_USERS_BASE_COLS =
  'id,username,password_hash,role,full_name,phone,specialty,license_no,patient_id,active,force_logged_out_at,created_at';
const STAFF_USERS_EXTENDED_COLS = `${STAFF_USERS_BASE_COLS},avatar_url,bio`;

const PATIENTS_BASE_COLS =
  'id,hn,full_name,age,phone,diagnosis,registration_status,assigned_pt_id,delegated_to_pt_id,delegation_note,next_appointment_date,face_biometric_status,face_frames_count,face_image_url,biometric_updated_at,history,created_at,updated_at';
const PATIENTS_BIOMETRIC_LIVE_COLS =
  'id,hn,full_name,age,phone,diagnosis,registration_status,assigned_pt_id,delegated_to_pt_id,delegation_note,next_appointment_date,face_biometric_status,face_frames_count,face_image_url,biometric_updated_at,created_at,updated_at,historyFaceStatus:history->>faceBiometricStatus,historyFaceFramesCount:history->faceFramesCount,historyFaceImageUrl:history->>faceImageUrl,historyGender:history->>gender,geometricRatios:history->geometricRatios,rehabSchedules:history->rehabSchedules,biometricEngine:history->>biometricEngine,strokeType:history->>strokeType,affectedSide:history->>affectedSide,motorAssessmentScore:history->motorAssessmentScore,adlDressingScore:history->adlDressingScore,maxSafeRomDeg:history->maxSafeRomDeg,universalHealthLinkCode:history->>universalHealthLinkCode';
const PATIENTS_MINIMAL_SAFE_COLS =
  'id,hn,full_name,age,phone,diagnosis,registration_status,assigned_pt_id,history,created_at';

const PATIENT_BIOMETRICS_BASE_COLS =
  'patient_id,hn,master_embedding,angle_embeddings,exemplar_embeddings,biometric_engine,model_version,frames_captured,quality_score,updated_at';
const PATIENT_BIOMETRICS_MINIMAL_SAFE_COLS =
  'patient_id,hn,master_embedding,angle_embeddings,updated_at';

const EXERCISES_BASE_COLS =
  'id,code,category,name_i18n,instruction_i18n,detection_config,active';

const MODPACKS_BASE_COLS =
  'id,name,categories,questions,active';

const KIOSKS_BASE_COLS =
  'kiosk_id,location,online,current_state,last_heartbeat_at';

const AUDIT_LOGS_BASE_COLS =
  'id,event_type,actor_role,actor_id,actor_name,ip_address,device_id,target_id,detail,created_at';

/** Fast single-table lookup of staff_users + cloud profile manifest for login & cross-device profile sync. */
export async function hydrateStaffUsersOnly(
  supabase: SupabaseClient,
  state: CloudRouterState,
  timeoutMs = 1500,
): Promise<boolean> {
  if (isSupabaseCircuitOpen(supabase)) {
    return false;
  }
  let timerId: ReturnType<typeof setTimeout> | undefined;
  try {
    const outcome = await Promise.race([
      Promise.all([
        selectExplicitWithFallback(
          'staff_users',
          (cols) => supabase.from('staff_users').select(cols),
          STAFF_USERS_EXTENDED_COLS,
          STAFF_USERS_BASE_COLS,
          250,
        ),
        fetchStaffProfileManifestFromStorage(supabase, Math.min(900, timeoutMs)),
      ]).then(([res, manifest]) => ({
        timedOut: false as const,
        res: res as { data: Record<string, unknown>[] | null; error: unknown },
        manifest,
      })),
      new Promise<{ timedOut: true }>((resolve) => {
        timerId = setTimeout(() => resolve({ timedOut: true }), timeoutMs);
      }),
    ]);
    if (outcome.timedOut) {
      markSupabaseDown('hydrate_staff_timeout', supabase);
      return false;
    }
    if (outcome.res.error || !Array.isArray(outcome.res.data)) {
      if (isSupabaseTransientFailure(outcome.res.error)) {
        markSupabaseDown('hydrate_staff_transient_error', supabase);
      }
      return false;
    }
    mergeStaffUsersFromDbRows(outcome.res.data, state, outcome.manifest);
    return true;
  } catch (err) {
    if (isSupabaseTransientFailure(err)) {
      markSupabaseDown('hydrate_staff_exception', supabase);
    }
    return false;
  } finally {
    if (timerId !== undefined) {
      clearTimeout(timerId);
    }
  }
}

export interface LiveBiometricHydrationResult {
  ok: boolean;
  errorCode?: 'TIMEOUT' | 'PERMISSION_DENIED' | 'SERVER_ERROR';
  filteredOutCount: number;
}

function isSupabasePermissionDeniedError(err: unknown): boolean {
  if (!err || typeof err !== 'object') return false;
  const obj = err as { code?: unknown; status?: unknown; message?: unknown };
  const code = String(obj.code ?? '').trim();
  const status = typeof obj.status === 'number' ? obj.status : 0;
  const msg = String(obj.message ?? '').toLowerCase();
  return (
    code === '42501' ||
    status === 401 ||
    status === 403 ||
    msg.includes('permission denied') ||
    msg.includes('row-level security') ||
    msg.includes('jwt expired') ||
    msg.includes('invalid api key')
  );
}

function isSupabaseTimeoutError(err: unknown): boolean {
  if (!err || typeof err !== 'object') return false;
  const rec = err as Record<string, unknown>;
  const code = String(rec.code ?? '');
  const status = Number(rec.status ?? rec.statusCode ?? 0);
  const msg = String(rec.message ?? rec.details ?? '').toLowerCase();
  return (
    code === '57014' ||
    status === 408 ||
    status === 504 ||
    msg.includes('statement timeout') ||
    msg.includes('timed out') ||
    msg.includes('timeout')
  );
}

export interface LiveBiometricStatsResult {
  ok: boolean;
  errorCode?: 'TIMEOUT' | 'PERMISSION_DENIED' | 'SERVER_ERROR';
  totalActivePatients: number;
  compatibleTemplatesCount: number;
  versionMismatchCount: number;
  missingTemplateCount: number;
  filteredOutCount: number;
}

/**
 * Ultra-lightweight metadata-only query for `GET /api/biometrics/stats` (~1.5 KB instead of ~1.65 MB).
 * Reads ONLY `patients(id,hn,registration_status)` and `patient_biometrics(patient_id,hn,embedding_dim,biometric_engine,model_version)`
 * with ZERO vector downloads, ZERO PII, and ~200-350ms latency so cold Worker isolates never time out!
 */
export async function fetchBiometricLibraryStatsFromSupabase(
  supabase: SupabaseClient,
  state: CloudRouterState,
  timeoutMs = 3500,
): Promise<LiveBiometricStatsResult> {
  if (isSupabaseCircuitOpen(supabase)) {
    return {
      ok: false,
      errorCode: 'SERVER_ERROR',
      totalActivePatients: 0,
      compatibleTemplatesCount: 0,
      versionMismatchCount: 0,
      missingTemplateCount: 0,
      filteredOutCount: state.filtered_out_template_patients_count ?? 0,
    };
  }
  let timerId: ReturnType<typeof setTimeout> | undefined;
  try {
    const fetchMeta = Promise.allSettled([
      selectExplicitWithFallback(
        'patients',
        (cols) => supabase.from('patients').select(cols),
        'id,hn,registration_status',
        'id,hn,registration_status',
        250,
      ),
      selectExplicitWithFallback(
        'patient_biometrics',
        (cols) => supabase.from('patient_biometrics').select(cols),
        'patient_id,hn,embedding_dim,biometric_engine,model_version',
        'patient_id,hn,embedding_dim',
        250,
      ),
    ]);

    const outcome = await Promise.race([
      fetchMeta.then((results) => ({ timedOut: false as const, results })),
      new Promise<{ timedOut: true }>((resolve) => {
        timerId = setTimeout(() => resolve({ timedOut: true }), timeoutMs);
      }),
    ]);

    if (outcome.timedOut) {
      return {
        ok: false,
        errorCode: 'TIMEOUT',
        totalActivePatients: 0,
        compatibleTemplatesCount: 0,
        versionMismatchCount: 0,
        missingTemplateCount: 0,
        filteredOutCount: state.filtered_out_template_patients_count ?? 0,
      };
    }

    const [patientsRes, pbRes] = outcome.results as [
      PromiseSettledResult<{ data: Record<string, unknown>[] | null; error: unknown }>,
      PromiseSettledResult<{ data: Record<string, unknown>[] | null; error: unknown }>,
    ];

    if (patientsRes.status === 'rejected') {
      if (isSupabasePermissionDeniedError(patientsRes.reason)) {
        return {
          ok: false,
          errorCode: 'PERMISSION_DENIED',
          totalActivePatients: 0,
          compatibleTemplatesCount: 0,
          versionMismatchCount: 0,
          missingTemplateCount: 0,
          filteredOutCount: 0,
        };
      }
      return {
        ok: false,
        errorCode: isSupabaseTimeoutError(patientsRes.reason) ? 'TIMEOUT' : 'SERVER_ERROR',
        totalActivePatients: 0,
        compatibleTemplatesCount: 0,
        versionMismatchCount: 0,
        missingTemplateCount: 0,
        filteredOutCount: 0,
      };
    }
    if (patientsRes.value.error || !Array.isArray(patientsRes.value.data)) {
      if (isSupabasePermissionDeniedError(patientsRes.value.error)) {
        return {
          ok: false,
          errorCode: 'PERMISSION_DENIED',
          totalActivePatients: 0,
          compatibleTemplatesCount: 0,
          versionMismatchCount: 0,
          missingTemplateCount: 0,
          filteredOutCount: 0,
        };
      }
      return {
        ok: false,
        errorCode: isSupabaseTimeoutError(patientsRes.value.error) ? 'TIMEOUT' : 'SERVER_ERROR',
        totalActivePatients: 0,
        compatibleTemplatesCount: 0,
        versionMismatchCount: 0,
        missingTemplateCount: 0,
        filteredOutCount: 0,
      };
    }

    const deletedLower = new Set(
      (state.deleted_patient_ids ?? []).map((x) => x.trim().toLowerCase()).filter(Boolean),
    );
    const inactiveUserPatientIds = new Set(
      state.users.filter((u) => !u.active && u.patientId).map((u) => u.patientId as string),
    );

    const pbByPatient = new Map<string, { dim: number; engine: string }>();
    if (pbRes.status === 'fulfilled' && !pbRes.value.error && Array.isArray(pbRes.value.data)) {
      for (const row of pbRes.value.data) {
        const pid = String(row.patient_id ?? '').trim();
        if (!pid) continue;
        const dim = Number(row.embedding_dim ?? 0);
        const engine = String(row.model_version ?? row.biometric_engine ?? '').trim();
        pbByPatient.set(pid, { dim, engine });
      }
    }

    let totalActivePatients = 0;
    let compatibleTemplatesCount = 0;
    let versionMismatchCount = 0;
    let missingTemplateCount = 0;
    let filteredOutCount = 0;

    for (const r of patientsRes.value.data) {
      const pid = String(r.id ?? '').trim();
      const hnLo = String(r.hn ?? '').trim().toLowerCase();
      const regLo = String(r.registration_status ?? '').trim().toLowerCase();
      const isActive =
        !r.deleted_at &&
        (regLo === '' || regLo === 'active') &&
        !deletedLower.has(pid.toLowerCase()) &&
        (!hnLo || !deletedLower.has(hnLo)) &&
        !inactiveUserPatientIds.has(pid);

      const pbMeta = pbByPatient.get(pid);
      if (!isActive) {
        if (pbMeta && pbMeta.dim >= 128) {
          filteredOutCount += 1;
        }
        continue;
      }

      totalActivePatients += 1;
      if (!pbMeta || pbMeta.dim < 128) {
        missingTemplateCount += 1;
        continue;
      }
      const isCompatible =
        pbMeta.dim >= ACTIVE_BIOMETRIC_DIM &&
        (!pbMeta.engine || pbMeta.engine === 'human-faceres-1024d');
      if (isCompatible) {
        compatibleTemplatesCount += 1;
      } else {
        versionMismatchCount += 1;
      }
    }

    state.filtered_out_template_patients_count = filteredOutCount;
    return {
      ok: true,
      totalActivePatients,
      compatibleTemplatesCount,
      versionMismatchCount,
      missingTemplateCount,
      filteredOutCount,
    };
  } catch (err) {
    if (isSupabasePermissionDeniedError(err)) {
      return {
        ok: false,
        errorCode: 'PERMISSION_DENIED',
        totalActivePatients: 0,
        compatibleTemplatesCount: 0,
        versionMismatchCount: 0,
        missingTemplateCount: 0,
        filteredOutCount: 0,
      };
    }
    return {
      ok: false,
      errorCode: isSupabaseTimeoutError(err) ? 'TIMEOUT' : 'SERVER_ERROR',
      totalActivePatients: 0,
      compatibleTemplatesCount: 0,
      versionMismatchCount: 0,
      missingTemplateCount: 0,
      filteredOutCount: 0,
    };
  } finally {
    if (timerId !== undefined) clearTimeout(timerId);
  }
}

/**
 * Lightweight 3-table live refresh (`patients`, `patient_biometrics`, `face_templates`)
 * so server-side `/api/biometric/verify` ALWAYS compares against live Supabase state
 * (new enrollments work immediately without Kiosk refresh; deleted patients are blocked
 * immediately without waiting for the 60s full-state hydration TTL).
 */
export async function hydrateBiometricsLiveFromSupabase(
  supabase: SupabaseClient,
  state: CloudRouterState,
  timeoutMs = 4500,
): Promise<LiveBiometricHydrationResult> {
  if (isSupabaseCircuitOpen(supabase)) {
    return { ok: false, errorCode: 'SERVER_ERROR', filteredOutCount: state.filtered_out_template_patients_count ?? 0 };
  }
  let timerId: ReturnType<typeof setTimeout> | undefined;
  try {
    const fetchAll = Promise.allSettled([
      selectExplicitWithFallback(
        'patients',
        (cols) => supabase.from('patients').select(cols),
        PATIENTS_BIOMETRIC_LIVE_COLS,
        PATIENTS_MINIMAL_SAFE_COLS,
        250,
      ),
      selectExplicitWithFallback(
        'patient_biometrics',
        (cols) => supabase.from('patient_biometrics').select(cols),
        PATIENT_BIOMETRICS_BASE_COLS,
        PATIENT_BIOMETRICS_MINIMAL_SAFE_COLS,
        250,
      ),
      selectExplicitWithFallback(
        'face_templates',
        (cols) => supabase.from('face_templates').select(cols),
        'patient_id,pose,embedding,quality_score',
        'patient_id,pose,embedding',
        250,
      ),
    ]);

    const outcome = await Promise.race([
      fetchAll.then((results) => ({ timedOut: false as const, results })),
      new Promise<{ timedOut: true }>((resolve) => {
        timerId = setTimeout(() => resolve({ timedOut: true }), timeoutMs);
      }),
    ]);

    if (outcome.timedOut) {
      markSupabaseDown('hydrate_biometrics_live_timeout', supabase);
      return { ok: false, errorCode: 'TIMEOUT', filteredOutCount: state.filtered_out_template_patients_count ?? 0 };
    }

    const [patientsRes, biometrics120Res, templatesRes] = outcome.results as [
      PromiseSettledResult<{ data: Record<string, unknown>[] | null; error: unknown }>,
      PromiseSettledResult<{ data: Record<string, unknown>[] | null; error: unknown }>,
      PromiseSettledResult<{ data: Record<string, unknown>[] | null; error: unknown }>,
    ];

    if (patientsRes.status === 'rejected') {
      if (isSupabasePermissionDeniedError(patientsRes.reason)) {
        return { ok: false, errorCode: 'PERMISSION_DENIED', filteredOutCount: 0 };
      }
      if (isSupabaseTimeoutError(patientsRes.reason)) {
        markSupabaseDown('hydrate_biometrics_live_timeout', supabase);
        return { ok: false, errorCode: 'TIMEOUT', filteredOutCount: 0 };
      }
      if (isSupabaseTransientFailure(patientsRes.reason)) {
        markSupabaseDown('hydrate_biometrics_live_rejected', supabase);
      }
      return { ok: false, errorCode: 'SERVER_ERROR', filteredOutCount: 0 };
    }
    if (patientsRes.value.error || !Array.isArray(patientsRes.value.data)) {
      if (isSupabasePermissionDeniedError(patientsRes.value.error)) {
        return { ok: false, errorCode: 'PERMISSION_DENIED', filteredOutCount: 0 };
      }
      if (isSupabaseTimeoutError(patientsRes.value.error)) {
        markSupabaseDown('hydrate_biometrics_live_timeout', supabase);
        return { ok: false, errorCode: 'TIMEOUT', filteredOutCount: 0 };
      }
      if (isSupabaseTransientFailure(patientsRes.value.error)) {
        markSupabaseDown('hydrate_biometrics_live_error', supabase);
      }
      return { ok: false, errorCode: 'SERVER_ERROR', filteredOutCount: 0 };
    }

    const deletedPatientSet = new Set(state.deleted_patient_ids ?? []);
    const deletedPatientLower = new Set(
      (state.deleted_patient_ids ?? []).map((x) => x.trim().toLowerCase()).filter(Boolean),
    );
    const filteredOutPatientIds = new Set<string>();
    const rawPatientIds = new Set<string>();
    for (const r of patientsRes.value.data) {
      const pidRaw = String(r.id ?? '').trim();
      const hnRaw = String(r.hn ?? '').trim();
      const pidLo = pidRaw.toLowerCase();
      const hnLo = hnRaw.toLowerCase();
      if (pidRaw) rawPatientIds.add(pidRaw);
      const rawReg = String(r.registration_status ?? '').trim().toLowerCase();
      const isExcluded =
        Boolean(r.deleted_at) ||
        rawReg === 'deleted' ||
        rawReg === 'rejected' ||
        rawReg === 'inactive' ||
        rawReg === 'archived';
      if (isExcluded) {
        if (pidRaw) filteredOutPatientIds.add(pidRaw);
        state.deleted_patient_ids ??= [];
        if (pidRaw && !deletedPatientSet.has(pidRaw)) {
          deletedPatientSet.add(pidRaw);
          state.deleted_patient_ids.push(pidRaw);
        }
        if (hnRaw && !deletedPatientSet.has(hnRaw)) {
          deletedPatientSet.add(hnRaw);
          state.deleted_patient_ids.push(hnRaw);
        }
        if (pidLo) deletedPatientLower.add(pidLo);
        if (hnLo) deletedPatientLower.add(hnLo);
      }
    }

    const parseVec = (raw: unknown): number[] | undefined => parseEmbeddingVector(raw);
    const parseExemplarList = (raw: unknown): number[][] => {
      let source: unknown = raw;
      if (typeof source === 'string' && source.trim().startsWith('[')) {
        try { source = JSON.parse(source); } catch { return []; }
      } else if (typeof source === 'string' && source.trim().startsWith('{')) {
        try {
          const obj = JSON.parse(source) as Record<string, unknown>;
          source = obj.exemplar_embeddings ?? obj.exemplarEmbeddings;
        } catch { return []; }
      } else if (source && typeof source === 'object' && !Array.isArray(source)) {
        const obj = source as Record<string, unknown>;
        source = obj.exemplar_embeddings ?? obj.exemplarEmbeddings;
      }
      if (!Array.isArray(source)) return [];
      const out: number[][] = [];
      for (const item of source) {
        const vec = parseVec(item);
        if (vec && vec.length >= 128) out.push(vec);
      }
      return out;
    };
    const parseAngleObject = (raw: unknown): AngleEmbeddings5D | undefined => {
      let obj: Record<string, unknown> | undefined;
      if (raw && typeof raw === 'object' && !Array.isArray(raw)) {
        const rec = raw as Record<string, unknown>;
        obj = (rec.angle_embeddings && typeof rec.angle_embeddings === 'object'
          ? rec.angle_embeddings
          : rec.angleEmbeddings && typeof rec.angleEmbeddings === 'object'
            ? rec.angleEmbeddings
            : rec) as Record<string, unknown>;
      } else if (typeof raw === 'string' && raw.trim().startsWith('{')) {
        try {
          const rec = JSON.parse(raw) as Record<string, unknown>;
          obj = (rec.angle_embeddings && typeof rec.angle_embeddings === 'object'
            ? rec.angle_embeddings
            : rec.angleEmbeddings && typeof rec.angleEmbeddings === 'object'
              ? rec.angleEmbeddings
              : rec) as Record<string, unknown>;
        } catch { return undefined; }
      }
      if (!obj) return undefined;
      const center = parseVec(obj.center);
      const left = parseVec(obj.left);
      const right = parseVec(obj.right);
      const up = parseVec(obj.up);
      const down = parseVec(obj.down);
      const base = center ?? left ?? right ?? up ?? down;
      if (!base) return undefined;
      return { center: center ?? base, left: left ?? base, right: right ?? base, up: up ?? base, down: down ?? base };
    };
    const normalizeEngineTag = (
      raw: unknown,
      vec?: readonly number[],
    ): 'human-faceres-1024d' | 'arcface-onnx' | 'hybrid-512d' | undefined => {
      if (raw === 'human-faceres-1024d' || (vec && vec.length >= 768)) return 'human-faceres-1024d';
      if (raw === 'arcface-onnx' || raw === 'hybrid-512d') return raw;
      return undefined;
    };

    const filteredOutTemplateIds = new Set<string>();
    const parsedTemplatesByPatient = new Map<
      string,
      { frontal?: number[]; overall?: number[]; raw: { pose: 'front' | 'left' | 'right' | 'nod'; vector: number[]; qualityScore: number }[] }
    >();
    const parsedTemplateCandidates: EnrolledTemplateCandidate[] = [];
    if (templatesRes.status === 'fulfilled' && !templatesRes.value.error && Array.isArray(templatesRes.value.data)) {
      for (const row of templatesRes.value.data) {
        const vec = parseVec(row.embedding);
        if (!vec) continue;
        const pid = String(row.patient_id);
        if (deletedPatientSet.has(pid) || deletedPatientLower.has(pid.toLowerCase()) || !rawPatientIds.has(pid)) {
          filteredOutTemplateIds.add(pid);
          continue;
        }
        const pose = (row.pose as BiometricPose) ?? 'frontal_centroid';
        parsedTemplateCandidates.push({ patientId: pid, pose, vector: vec });
        const entry = parsedTemplatesByPatient.get(pid) ?? { raw: [] };
        if (pose === 'frontal_centroid') entry.frontal = vec;
        else if (pose === 'overall_centroid') entry.overall = vec;
        else if (pose === 'front' || pose === 'left' || pose === 'right' || pose === 'nod') {
          entry.raw.push({ pose, vector: vec, qualityScore: Number(row.quality_score ?? 1.0) });
        }
        parsedTemplatesByPatient.set(pid, entry);
      }
    }

    const biometrics120ByPatient = new Map<
      string,
      {
        masterEmbedding?: number[];
        angleEmbeddings?: AngleEmbeddings5D;
        exemplarEmbeddings?: number[][];
        biometricEngine?: 'human-faceres-1024d' | 'arcface-onnx' | 'hybrid-512d';
        framesCaptured?: number;
        avatarUrl?: string;
      }
    >();
    if (biometrics120Res.status === 'fulfilled' && !biometrics120Res.value.error && Array.isArray(biometrics120Res.value.data)) {
      for (const row of biometrics120Res.value.data) {
        const pid = String(row.patient_id);
        const hnLo = String(row.hn ?? '').trim().toLowerCase();
        if (
          deletedPatientSet.has(pid) ||
          deletedPatientLower.has(pid.toLowerCase()) ||
          (hnLo && deletedPatientLower.has(hnLo)) ||
          !rawPatientIds.has(pid)
        ) {
          filteredOutTemplateIds.add(pid);
          continue;
        }
        const parsedAngles = parseAngleObject(row.angle_embeddings ?? row.biometric_data);
        const parsedMaster =
          parseVec(row.master_embedding) ??
          parseVec(row.face_embedding) ??
          parseVec(row.biometric_data) ??
          parsedAngles?.center;
        const parsedExemplars = parseExemplarList(row.exemplar_embeddings ?? row.biometric_data);
        biometrics120ByPatient.set(pid, {
          masterEmbedding: parsedMaster,
          angleEmbeddings: parsedAngles,
          exemplarEmbeddings: parsedExemplars.length > 0 ? parsedExemplars : undefined,
          biometricEngine: normalizeEngineTag(row.model_version ?? row.biometric_engine, parsedMaster),
          framesCaptured: Number(row.frame_count ?? row.frames_captured ?? 120),
          avatarUrl: typeof row.avatar_url === 'string' ? row.avatar_url : undefined,
        });
      }
    }

    for (const r of patientsRes.value.data) {
      const pid = String(r.id ?? '').trim();
      if (!filteredOutPatientIds.has(pid)) continue;
      const h = (r.history && typeof r.history === 'object' ? r.history : {}) as Record<string, unknown>;
      if (parseVec(r.master_embedding) || parseVec(r.face_embedding) || parseVec(h.masterEmbedding)) {
        filteredOutTemplateIds.add(pid);
      }
    }
    state.filtered_out_template_patients_count = filteredOutTemplateIds.size;

    const prevPatientsById = new Map(state.patients.map((p) => [p.id, p]));
    const activePatientRows = patientsRes.value.data.filter((r) => {
      const pidRaw = String(r.id ?? '').trim();
      const hnRaw = String(r.hn ?? '').trim();
      const pidLo = pidRaw.toLowerCase();
      const hnLo = hnRaw.toLowerCase();
      const rawReg = String(r.registration_status ?? '').trim().toLowerCase();
      const isActive = !r.deleted_at && (rawReg === '' || rawReg === 'active');
      return isActive && !deletedPatientLower.has(pidLo) && (!hnLo || !deletedPatientLower.has(hnLo));
    });

    // If any active patient has verified history metadata but is missing a patient_biometrics row,
    // fetch its full history/top-level embedding columns in one targeted query so login never misses it.
    const historyFallbackById = new Map<string, Record<string, unknown>>();
    const missingVecIds = activePatientRows
      .filter((r) => {
        const pid = String(r.id ?? '').trim();
        if (!pid) return false;
        const b120 = biometrics120ByPatient.get(pid);
        const tpl = parsedTemplatesByPatient.get(pid);
        const alreadyHasVec = Boolean(
          b120?.masterEmbedding ||
            parseVec(r.master_embedding) ||
            parseVec(r.face_embedding) ||
            parseVec(r.biometric_data) ||
            (r.history && typeof r.history === 'object' && parseVec((r.history as Record<string, unknown>).masterEmbedding)) ||
            tpl?.overall ||
            tpl?.frontal,
        );
        if (alreadyHasVec) return false;
        const histStatus = String(r.historyFaceStatus ?? '').trim().toUpperCase();
        const histFrames = Number(r.historyFaceFramesCount ?? 0);
        const topStatus = String(r.face_biometric_status ?? '').trim().toUpperCase();
        return histStatus === 'VERIFIED' || histFrames > 0 || topStatus === 'VERIFIED';
      })
      .map((r) => String(r.id));

    if (missingVecIds.length > 0) {
      try {
        const q = supabase
          .from('patients')
          .select('id,history,master_embedding,angle_embeddings,biometric_data,face_embedding') as unknown as {
          in?: (col: string, vals: string[]) => PromiseLike<{ data?: Record<string, unknown>[] | null; error?: unknown }>;
        };
        if (typeof q.in === 'function') {
          const histRes = await q.in('id', missingVecIds);
          if (!histRes?.error && Array.isArray(histRes?.data)) {
            for (const hr of histRes.data) {
              const hpid = String(hr.id ?? '').trim();
              if (hpid) historyFallbackById.set(hpid, hr);
            }
          }
        }
      } catch {
        // Best-effort fallback only
      }
    }

    state.patients = activePatientRows.map((rawRow) => {
        const pid = String(rawRow.id);
        const extraHist = historyFallbackById.get(pid);
        const r: Record<string, unknown> = extraHist ? { ...rawRow, ...extraHist } : rawRow;
        const prevPatient = prevPatientsById.get(pid);
        const history = (r.history && typeof r.history === 'object' ? r.history : {}) as Record<string, unknown>;
        const tpl = parsedTemplatesByPatient.get(pid);
        const b120 = biometrics120ByPatient.get(pid);
        const rowMasterVec =
          parseVec(r.master_embedding) ??
          parseVec(r.face_embedding) ??
          parseVec(r.biometric_data);
        const fallbackMaster =
          b120?.masterEmbedding ??
          rowMasterVec ??
          parseVec(history.masterEmbedding) ??
          tpl?.overall ??
          tpl?.frontal;
        const mergedMaster = fallbackMaster ?? prevPatient?.masterEmbedding;
        let fallbackAngles =
          b120?.angleEmbeddings ??
          parseAngleObject(r.angle_embeddings ?? r.face_embedding ?? r.biometric_data) ??
          parseAngleObject(history.angleEmbeddings);
        if (!fallbackAngles && mergedMaster) {
          const frontVec = tpl?.raw.find((f) => f.pose === 'front')?.vector ?? tpl?.frontal ?? mergedMaster;
          const leftVec = tpl?.raw.find((f) => f.pose === 'left')?.vector ?? mergedMaster;
          const rightVec = tpl?.raw.find((f) => f.pose === 'right')?.vector ?? mergedMaster;
          const nodVec = tpl?.raw.find((f) => f.pose === 'nod')?.vector ?? mergedMaster;
          fallbackAngles = { center: frontVec, left: leftVec, right: rightVec, up: nodVec, down: nodVec };
        }
        const mergedAngles = fallbackAngles ?? prevPatient?.angleEmbeddings;
        const fallbackExemplars = (() => {
          const fromB120 = b120?.exemplarEmbeddings ?? [];
          const fromRow = parseExemplarList(r.face_embedding ?? r.biometric_data);
          const fromHist = parseExemplarList(history.exemplarEmbeddings);
          const chosen = fromB120.length > 0 ? fromB120 : fromRow.length > 0 ? fromRow : fromHist;
          if (!mergedMaster || chosen.length === 0) return undefined;
          const matchedDim = chosen.filter((v) => v.length === mergedMaster.length);
          return matchedDim.length > 0 ? matchedDim : undefined;
        })();
        const mergedExemplars = fallbackExemplars ?? prevPatient?.exemplarEmbeddings;
        const resolvedGeom =
          r.geometricRatios && typeof r.geometricRatios === 'object'
            ? (r.geometricRatios as GeometricAnthropometricRatios)
            : history.geometricRatios && typeof history.geometricRatios === 'object'
              ? (history.geometricRatios as GeometricAnthropometricRatios)
              : prevPatient?.geometricRatios;
        const inspected = inspectPatientRecordHydration(r, {
          biometricsRow: {
            masterEmbedding: mergedMaster,
            angleEmbeddings: mergedAngles,
            exemplarEmbeddings: mergedExemplars,
            biometricEngine: b120?.biometricEngine,
            framesCaptured: b120?.framesCaptured,
            avatarUrl: b120?.avatarUrl,
            geometricRatios: resolvedGeom,
          },
          prevPatient,
          strictCurrentDimOnly: true,
          requireRealVectorsForVerified: true,
        });
        if (inspected.needsTopLevelRepair && inspected.topLevelRepairPatch) {
          void schedulePatientTopLevelRepairInBackground(supabase, pid, inspected.topLevelRepairPatch);
        }
        const hasRealVector = Boolean(mergedMaster && mergedMaster.length >= 128);
        const embeddingDim = mergedMaster?.length ?? 0;
        const resolvedEngine =
          normalizeEngineTag(b120?.biometricEngine, mergedMaster) ??
          normalizeEngineTag(r.biometricEngine, mergedMaster) ??
          normalizeEngineTag(history.biometricEngine, mergedMaster) ??
          normalizeEngineTag(prevPatient?.biometricEngine, mergedMaster);
        const isCompatibleCurrent =
          hasRealVector &&
          embeddingDim >= ACTIVE_BIOMETRIC_DIM &&
          (!resolvedEngine || resolvedEngine === 'human-faceres-1024d');
        const biometricStatusDetail: 'VERIFIED' | 'PENDING' | 'REENROLL_REQUIRED' = !hasRealVector
          ? 'PENDING'
          : isCompatibleCurrent
            ? 'VERIFIED'
            : 'REENROLL_REQUIRED';
        const biometricPipelineVersion = !hasRealVector
          ? undefined
          : embeddingDim >= ACTIVE_BIOMETRIC_DIM
            ? resolvedGeom?.ocularIrisPatch256?.length === 256
              ? ACTIVE_BIOMETRIC_PIPELINE_VERSION
              : 'v2-pre-iris-1024d'
            : 'legacy-512d';
        const projectedSchedules = Array.isArray(r.rehabSchedules)
          ? (r.rehabSchedules as RehabScheduleItem[])
          : Array.isArray(history.rehabSchedules)
            ? (history.rehabSchedules as RehabScheduleItem[])
            : null;
        const projectedStrokeType =
          (typeof r.strokeType === 'string' && r.strokeType
            ? (r.strokeType as StrokeType)
            : typeof history.strokeType === 'string' && history.strokeType
              ? (history.strokeType as StrokeType)
              : prevPatient?.strokeType) ?? undefined;
        const projectedAffectedSide =
          (typeof r.affectedSide === 'string' && r.affectedSide
            ? (r.affectedSide as AffectedSide)
            : typeof history.affectedSide === 'string' && history.affectedSide
              ? (history.affectedSide as AffectedSide)
              : prevPatient?.affectedSide) ?? undefined;
        const projectedMotorScore =
          typeof r.motorAssessmentScore === 'number'
            ? r.motorAssessmentScore
            : typeof history.motorAssessmentScore === 'number'
              ? history.motorAssessmentScore
              : prevPatient?.motorAssessmentScore;
        const projectedAdlScore =
          typeof r.adlDressingScore === 'number'
            ? r.adlDressingScore
            : typeof history.adlDressingScore === 'number'
              ? history.adlDressingScore
              : prevPatient?.adlDressingScore;
        const projectedMaxRom =
          typeof r.maxSafeRomDeg === 'number'
            ? r.maxSafeRomDeg
            : typeof history.maxSafeRomDeg === 'number'
              ? history.maxSafeRomDeg
              : prevPatient?.maxSafeRomDeg;
        const projectedLinkCode =
          typeof r.universalHealthLinkCode === 'string' && r.universalHealthLinkCode
            ? r.universalHealthLinkCode
            : typeof history.universalHealthLinkCode === 'string' && history.universalHealthLinkCode
              ? history.universalHealthLinkCode
              : prevPatient?.universalHealthLinkCode;

        return {
          ...(prevPatient ?? {
            id: pid,
            hn: String(r.hn),
            fullName: String(r.full_name),
            age: Number(r.age ?? 0),
            phone: String(r.phone ?? ''),
            diagnosis: String(r.diagnosis ?? ''),
            assignedPtId: String(r.assigned_pt_id ?? ''),
            delegatedToPtId: r.delegated_to_pt_id ? String(r.delegated_to_pt_id) : null,
            delegationNote: r.delegation_note ? String(r.delegation_note) : null,
            nextAppointmentDate: String(r.next_appointment_date ?? ''),
            registrationStatus: 'active' as const,
            enrolledAt: String(r.created_at ?? new Date().toISOString()),
          }),
          id: pid,
          hn: String(r.hn),
          fullName: String(r.full_name),
          ...(inspected.patient.gender ? { gender: inspected.patient.gender } : {}),
          nextAppointmentDate: String(r.next_appointment_date ?? prevPatient?.nextAppointmentDate ?? ''),
          registrationStatus: 'active' as const,
          ...(projectedStrokeType ? { strokeType: projectedStrokeType } : {}),
          ...(projectedAffectedSide ? { affectedSide: projectedAffectedSide } : {}),
          ...(projectedMotorScore !== undefined ? { motorAssessmentScore: projectedMotorScore } : {}),
          ...(projectedAdlScore !== undefined ? { adlDressingScore: projectedAdlScore } : {}),
          ...(projectedMaxRom !== undefined ? { maxSafeRomDeg: projectedMaxRom } : {}),
          ...(projectedLinkCode ? { universalHealthLinkCode: projectedLinkCode } : {}),
          faceBiometricStatus: isCompatibleCurrent ? ('VERIFIED' as const) : ('PENDING' as const),
          biometricStatusDetail,
          biometricEmbeddingDim: embeddingDim,
          biometricPipelineVersion,
          ...(inspected.patient.biometricUpdatedAt ? { biometricUpdatedAt: inspected.patient.biometricUpdatedAt } : {}),
          ...(inspected.patient.faceImageUrl ? { faceImageUrl: inspected.patient.faceImageUrl } : {}),
          faceFramesCount: hasRealVector
            ? Math.max(b120?.framesCaptured ?? 0, Number(r.face_frames_count ?? 0), Number(r.historyFaceFramesCount ?? 0), prevPatient?.faceFramesCount ?? 0, 120)
            : 0,
          masterEmbedding: mergedMaster,
          angleEmbeddings: mergedAngles,
          ...(mergedExemplars ? { exemplarEmbeddings: mergedExemplars } : {}),
          biometricEngine: resolvedEngine,
          geometricRatios: resolvedGeom,
          rehabSchedules: projectedSchedules
            ? sanitizeAccidentalDefaultKneePrefix(projectedSchedules)
            : prevPatient?.rehabSchedules ?? [],
          biometricHistory:
            tpl && tpl.frontal && tpl.overall
              ? { frontal_centroid: tpl.frontal, overall_centroid: tpl.overall, raw_frames: tpl.raw }
              : mergedMaster
                ? { frontal_centroid: mergedAngles?.center ?? mergedMaster, overall_centroid: mergedMaster, raw_frames: [] }
                : undefined,
        };
      });

    const activePatientMap = new Map(state.patients.map((p) => [p.id, p.fullName]));
    if (templatesRes.status === 'fulfilled' && !templatesRes.value.error && Array.isArray(templatesRes.value.data)) {
      state.face_templates = parsedTemplateCandidates
        .filter((t) => activePatientMap.has(t.patientId))
        .map((t) => ({ ...t, patientName: activePatientMap.get(t.patientId) }));
    } else {
      state.face_templates = state.face_templates.filter((t) => activePatientMap.has(t.patientId));
    }

    return { ok: true, filteredOutCount: filteredOutTemplateIds.size };
  } catch (err) {
    if (isSupabasePermissionDeniedError(err)) {
      return { ok: false, errorCode: 'PERMISSION_DENIED', filteredOutCount: 0 };
    }
    if (isSupabaseTransientFailure(err)) {
      markSupabaseDown('hydrate_biometrics_live_exception', supabase);
    }
    return { ok: false, errorCode: 'SERVER_ERROR', filteredOutCount: 0 };
  } finally {
    if (timerId !== undefined) clearTimeout(timerId);
  }
}

export async function hydrateStateFromSupabase(
  supabase: SupabaseClient,
  state: CloudRouterState,
  options?: { force?: boolean },
): Promise<void> {
  if (isSupabaseCircuitOpen(supabase)) {
    return;
  }
  const now = Date.now();
  const lastHydratedAt = lastHydratedAtByState.get(state) ?? 0;
  if (!options?.force && lastHydratedAt > 0 && now - lastHydratedAt < HYDRATE_TTL_MS) {
    return;
  }
  // Never share in-flight I/O Promises across Cloudflare Worker requests (workerd throws
  // "Cannot perform I/O on behalf of a different request" if request B awaits request A's fetch).
  if (isCloudflareWorkerRuntime()) {
    await runHydrateStateFromSupabaseInternal(supabase, state);
    return;
  }
  const existingFlight = hydrationInFlightByState.get(state);
  if (existingFlight) {
    return existingFlight;
  }
  const runPromise = runHydrateStateFromSupabaseInternal(supabase, state).finally(() => {
    if (hydrationInFlightByState.get(state) === runPromise) {
      hydrationInFlightByState.delete(state);
    }
  });
  hydrationInFlightByState.set(state, runPromise);
  return runPromise;
}

async function runHydrateStateFromSupabaseInternal(
  supabase: SupabaseClient,
  state: CloudRouterState,
): Promise<void> {
  if (isSupabaseCircuitOpen(supabase)) {
    return;
  }
  const totalDeadlineMs = Date.now() + 1500;

  // Stage 1 (Priority Clinical Directory): fetch patients, staff_users & cloud profile manifest FIRST
  const stage1Tasks: ReadonlyArray<() => PromiseLike<unknown>> = [
    () =>
      selectExplicitWithFallback(
        'staff_users',
        (cols) => supabase.from('staff_users').select(cols),
        STAFF_USERS_EXTENDED_COLS,
        STAFF_USERS_BASE_COLS,
        250,
      ),
    () => fetchStaffProfileManifestFromStorage(supabase, 850),
    () =>
      selectExplicitWithFallback(
        'patients',
        (cols) => supabase.from('patients').select(cols),
        PATIENTS_BASE_COLS,
        PATIENTS_MINIMAL_SAFE_COLS,
        250,
      ),
    () =>
      supabase
        .from('system_settings')
        .select('id,ai_face_scan_enabled,ai_face_threshold,ai_pose_detection_enabled,kiosk_auto_reset_sec')
        .eq('id', 1)
        .maybeSingle(),
    () =>
      withQueryLimit(
        supabase
          .from('banned_targets')
          .select('id,target_type,value,reason,banned_by,created_at')
          .order('created_at', { ascending: false }),
        100,
      ),
    () =>
      withQueryLimit(
        supabase
          .from('active_sessions')
          .select('token,user_id,role,ip_address,device_id,created_at,revoked')
          .order('created_at', { ascending: false }),
        200,
      ),
  ];

  // Stage 2 (Biometric Embeddings & Clinical Config): explicit base-schema columns
  const stage2Tasks: ReadonlyArray<() => PromiseLike<unknown>> = [
    () =>
      selectExplicitWithFallback(
        'face_templates',
        (cols) => supabase.from('face_templates').select(cols),
        'patient_id,pose,embedding,quality_score',
        'patient_id,pose,embedding',
        250,
      ),
    () =>
      selectExplicitWithFallback(
        'patient_biometrics',
        (cols) => supabase.from('patient_biometrics').select(cols),
        PATIENT_BIOMETRICS_BASE_COLS,
        PATIENT_BIOMETRICS_MINIMAL_SAFE_COLS,
        250,
      ),
    () =>
      withQueryLimit(
        supabase.from('patient_courses').select('id,patient_id,title,weekly_schedule,cognitive_modpack_id,updated_at'),
        250,
      ),
    () =>
      selectExplicitWithFallback(
        'exercises',
        (cols) => supabase.from('exercises').select(cols).eq('active', true),
        EXERCISES_BASE_COLS,
        'id,code,category,active',
        50,
      ),
    () =>
      selectExplicitWithFallback(
        'modpacks',
        (cols) => supabase.from('modpacks').select(cols).eq('active', true),
        MODPACKS_BASE_COLS,
        'id,name,active',
        50,
      ),
    () =>
      selectExplicitWithFallback(
        'kiosks',
        (cols) => supabase.from('kiosks').select(cols),
        KIOSKS_BASE_COLS,
        KIOSKS_BASE_COLS,
        50,
      ),
  ];

  // Stage 3 (Historical Logs & Telemetry): explicit base-schema columns + limit(50)
  const stage3Tasks: ReadonlyArray<() => PromiseLike<unknown>> = [
    () =>
      withQueryLimit(
        supabase
          .from('medical_notes')
          .select('id,patient_id,author_id,author_name,content,created_at')
          .order('created_at', { ascending: false }),
        50,
      ),
    () =>
      withQueryLimit(
        supabase
          .from('rehab_sessions')
          .select('id,patient_id,exercise_id,completed_sets,completed_reps,accuracy_percent,max_rom_deg,pain_stop_triggered,kinematics_replay,minigame_score,created_at')
          .order('created_at', { ascending: false }),
        50,
      ),
    () =>
      withQueryLimit(
        supabase
          .from('treatment_reports')
          .select('id,patient_id,pt_id,pt_name,summary,ai_recommendation,score_summary,created_at,updated_at')
          .order('created_at', { ascending: false }),
        50,
      ),
    () =>
      withQueryLimit(
        supabase
          .from('staff_calls')
          .select('id,kiosk_id,patient_id,reason,status,created_at')
          .order('created_at', { ascending: false }),
        50,
      ),
    () =>
      selectExplicitWithFallback(
        'audit_logs',
        (cols) =>
          supabase
            .from('audit_logs')
            .select(cols)
            .order('created_at', { ascending: false }),
        AUDIT_LOGS_BASE_COLS,
        AUDIT_LOGS_BASE_COLS,
        50,
      ),
    () =>
      selectExplicitWithFallback(
        'rehab_session_history',
        (cols) =>
          supabase
            .from('rehab_session_history')
            .select(cols)
            .order('session_date', { ascending: false }),
        'id,patient_id,hn,session_date,exercise_code,exercise_name,joint_name,body_side,baseline_rom_deg,achieved_max_rom_deg,rom_delta_deg,motor_score_at_session,valid_reps,compensations_count,facility_name,recorded_by_pt_id,recorded_by_pt_name,clinical_notes,created_at',
        'id,patient_id,hn,session_date,exercise_code,exercise_name,achieved_max_rom_deg,valid_reps,created_at',
        50,
      ),
    () =>
      selectExplicitWithFallback(
        'patient_clinical_assessments',
        (cols) =>
          supabase
            .from('patient_clinical_assessments')
            .select(cols)
            .order('assessed_at', { ascending: true }),
        'id,patient_id,patient_hn,universal_link_code,visit_number,assessed_at,facility_name,assessor_staff_id,assessor_staff_name,exercise_code,exercise_name_th,target_side,initial_score_0_to_6,measured_rom_deg,target_rom_deg,pain_vas_0_to_10,chief_complaint,clinical_handover_note,next_session_goal,source_channel,created_at',
        'id,patient_id,patient_hn,universal_link_code,visit_number,assessed_at,exercise_code,initial_score_0_to_6,measured_rom_deg,target_rom_deg',
        120,
      ),
  ];

  const [usersRes, staffManifestRes, patientsRes, settingsRes, bansRes, activeSessionsRes] =
    (await settleStageInBatches(stage1Tasks, 6, Math.min(totalDeadlineMs, Date.now() + 1200))) as [
      PromiseSettledResult<{ data: Record<string, unknown>[] | null; error: unknown }>,
      PromiseSettledResult<Record<string, StaffProfileCloudRecord>>,
      PromiseSettledResult<{ data: Record<string, unknown>[] | null; error: unknown }>,
      PromiseSettledResult<{ data: Record<string, unknown> | null; error: unknown }>,
      PromiseSettledResult<{ data: Record<string, unknown>[] | null; error: unknown }>,
      PromiseSettledResult<{ data: Record<string, unknown>[] | null; error: unknown }>,
    ];

  const [templatesRes, biometrics120Res, coursesRes, exercisesRes, modpacksRes, kiosksRes] =
    (await settleStageInBatches(stage2Tasks, 2, totalDeadlineMs)) as [
      PromiseSettledResult<{ data: Record<string, unknown>[] | null; error: unknown }>,
      PromiseSettledResult<{ data: Record<string, unknown>[] | null; error: unknown }>,
      PromiseSettledResult<{ data: Record<string, unknown>[] | null; error: unknown }>,
      PromiseSettledResult<{ data: Record<string, unknown>[] | null; error: unknown }>,
      PromiseSettledResult<{ data: Record<string, unknown>[] | null; error: unknown }>,
      PromiseSettledResult<{ data: Record<string, unknown>[] | null; error: unknown }>,
    ];

  const [notesRes, sessionsRes, reportsRes, callsRes, auditRes, sessionHistoryRes, assessmentsRes] =
    (await settleStageInBatches(stage3Tasks, 3, totalDeadlineMs)) as [
      PromiseSettledResult<{ data: Record<string, unknown>[] | null; error: unknown }>,
      PromiseSettledResult<{ data: Record<string, unknown>[] | null; error: unknown }>,
      PromiseSettledResult<{ data: Record<string, unknown>[] | null; error: unknown }>,
      PromiseSettledResult<{ data: Record<string, unknown>[] | null; error: unknown }>,
      PromiseSettledResult<{ data: Record<string, unknown>[] | null; error: unknown }>,
      PromiseSettledResult<{ data: Record<string, unknown>[] | null; error: unknown }>,
      PromiseSettledResult<{ data: Record<string, unknown>[] | null; error: unknown }>,
    ];

  const settledAll: PromiseSettledResult<{ error?: unknown } | null>[] = [
    usersRes,
    patientsRes,
    settingsRes,
    bansRes,
    activeSessionsRes,
    templatesRes,
    biometrics120Res,
    coursesRes,
    exercisesRes,
    modpacksRes,
    kiosksRes,
    notesRes,
    sessionsRes,
    reportsRes,
    callsRes,
    auditRes,
    sessionHistoryRes,
  ];
  for (const s of settledAll) {
    if (s.status === 'rejected') {
      if (isSupabaseTransientFailure(s.reason)) {
        markSupabaseDown('hydrate_rejected_transient', supabase);
        break;
      }
    } else if (s.value && typeof s.value === 'object' && 'error' in s.value && s.value.error) {
      if (isSupabaseTransientFailure(s.value.error)) {
        markSupabaseDown('hydrate_error_5xx_or_544', supabase);
        break;
      }
    }
  }

  if (usersRes.status === 'fulfilled' && !usersRes.value.error && Array.isArray(usersRes.value.data)) {
    const manifest =
      staffManifestRes.status === 'fulfilled' && staffManifestRes.value
        ? staffManifestRes.value
        : undefined;
    mergeStaffUsersFromDbRows(usersRes.value.data as Record<string, unknown>[], state, manifest);
  }

  const deletedPatientSet = new Set(state.deleted_patient_ids ?? []);
  const deletedPatientLower = new Set(
    (state.deleted_patient_ids ?? []).map((x) => x.trim().toLowerCase()).filter(Boolean),
  );
  if (patientsRes.status === 'fulfilled' && !patientsRes.value.error && Array.isArray(patientsRes.value.data)) {
    state.deleted_patient_ids ??= [];
    for (const r of patientsRes.value.data as Record<string, unknown>[]) {
      const regStatus = String(r.registration_status ?? '').trim().toLowerCase();
      if (
        Boolean(r.deleted_at) ||
        regStatus === 'deleted' ||
        regStatus === 'rejected' ||
        regStatus === 'inactive' ||
        regStatus === 'archived'
      ) {
        const pidRaw = String(r.id ?? '').trim();
        const hnRaw = String(r.hn ?? '').trim();
        const pidLo = pidRaw.toLowerCase();
        const hnLo = hnRaw.toLowerCase();
        if (pidRaw && !deletedPatientSet.has(pidRaw)) {
          deletedPatientSet.add(pidRaw);
          state.deleted_patient_ids.push(pidRaw);
        }
        if (hnRaw && !deletedPatientSet.has(hnRaw)) {
          deletedPatientSet.add(hnRaw);
          state.deleted_patient_ids.push(hnRaw);
        }
        if (pidLo) deletedPatientLower.add(pidLo);
        if (hnLo) deletedPatientLower.add(hnLo);
      }
    }
  }

  const parsedTemplatesByPatient = new Map<
    string,
    { frontal?: number[]; overall?: number[]; raw: { pose: 'front' | 'left' | 'right' | 'nod'; vector: number[]; qualityScore: number }[] }
  >();
  const parsedTemplateCandidates: EnrolledTemplateCandidate[] = [];

  if (templatesRes.status === 'fulfilled' && !templatesRes.value.error && Array.isArray(templatesRes.value.data)) {
    for (const row of templatesRes.value.data as Record<string, unknown>[]) {
      const vec = parseEmbeddingVector(row.embedding);
      if (!vec) continue;
      const pid = String(row.patient_id);
      if (deletedPatientSet.has(pid) || deletedPatientLower.has(pid.toLowerCase())) continue;
      const pose = (row.pose as BiometricPose) ?? 'frontal_centroid';
      parsedTemplateCandidates.push({
        patientId: pid,
        pose,
        vector: vec,
      });
      const entry = parsedTemplatesByPatient.get(pid) ?? { raw: [] };
      if (pose === 'frontal_centroid') entry.frontal = vec;
      else if (pose === 'overall_centroid') entry.overall = vec;
      else if (pose === 'front' || pose === 'left' || pose === 'right' || pose === 'nod') {
        entry.raw.push({ pose, vector: vec, qualityScore: Number(row.quality_score ?? 1.0) });
      }
      parsedTemplatesByPatient.set(pid, entry);
    }
  }

  const biometrics120ByPatient = new Map<
    string,
    {
      masterEmbedding?: number[];
      angleEmbeddings?: AngleEmbeddings5D;
      exemplarEmbeddings?: number[][];
      biometricEngine?: 'human-faceres-1024d' | 'arcface-onnx' | 'hybrid-512d';
      framesCaptured?: number;
      avatarUrl?: string;
    }
  >();
  const parseVec = (raw: unknown): number[] | undefined => parseEmbeddingVector(raw);

  const parseExemplarList = (raw: unknown): number[][] => {
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
      const vec = parseVec(item);
      if (vec && vec.length >= 128) {
        out.push(vec);
      }
    }
    return out;
  };

  const normalizeEngineTag = (
    raw: unknown,
    vec?: readonly number[],
  ): 'human-faceres-1024d' | 'arcface-onnx' | 'hybrid-512d' | undefined => {
    if (raw === 'human-faceres-1024d' || (vec && vec.length >= 768)) return 'human-faceres-1024d';
    if (raw === 'arcface-onnx' || raw === 'hybrid-512d') return raw;
    return undefined;
  };

  const parseAngleObject = (raw: unknown): AngleEmbeddings5D | undefined => {
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
        if (parsed && typeof parsed === 'object') {
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
    const center = parseVec(obj.center);
    const left = parseVec(obj.left);
    const right = parseVec(obj.right);
    const up = parseVec(obj.up);
    const down = parseVec(obj.down);
    const base = center ?? left ?? right ?? up ?? down;
    if (!base) return undefined;
    return {
      center: center ?? base,
      left: left ?? base,
      right: right ?? base,
      up: up ?? base,
      down: down ?? base,
    };
  };

  if (biometrics120Res.status === 'fulfilled' && !biometrics120Res.value.error && Array.isArray(biometrics120Res.value.data)) {
    for (const row of biometrics120Res.value.data as Record<string, unknown>[]) {
      const pid = String(row.patient_id);
      const hnLo = String(row.hn ?? '').trim().toLowerCase();
      if (
        deletedPatientSet.has(pid) ||
        deletedPatientLower.has(pid.toLowerCase()) ||
        (hnLo && deletedPatientLower.has(hnLo))
      ) {
        continue;
      }
      const parsedAngles = parseAngleObject(row.angle_embeddings ?? row.biometric_data);
      const parsedMaster =
        parseVec(row.master_embedding) ??
        parseVec(row.face_embedding) ??
        parseVec(row.biometric_data) ??
        parsedAngles?.center;
      const parsedExemplars = parseExemplarList(row.exemplar_embeddings ?? row.biometric_data);
      biometrics120ByPatient.set(pid, {
        masterEmbedding: parsedMaster,
        angleEmbeddings: parsedAngles,
        exemplarEmbeddings: parsedExemplars.length > 0 ? parsedExemplars : undefined,
        biometricEngine: normalizeEngineTag(row.model_version ?? row.biometric_engine, parsedMaster),
        framesCaptured: Number(row.frame_count ?? row.frames_captured ?? 120),
        avatarUrl: typeof row.avatar_url === 'string' ? row.avatar_url : undefined,
      });
    }
  }

  if (patientsRes.status === 'fulfilled' && !patientsRes.value.error && Array.isArray(patientsRes.value.data)) {
    const prevPatientsById = new Map(state.patients.map((p) => [p.id, p]));
    state.patients = patientsRes.value.data
      .filter((r: Record<string, unknown>) => {
        const pidRaw = String(r.id ?? '').trim();
        const hnRaw = String(r.hn ?? '').trim();
        const pidLo = pidRaw.toLowerCase();
        const hnLo = hnRaw.toLowerCase();
        const regStatus = String(r.registration_status ?? '').trim().toLowerCase();
        if (
          Boolean(r.deleted_at) ||
          regStatus === 'deleted' ||
          regStatus === 'rejected' ||
          regStatus === 'inactive' ||
          regStatus === 'archived'
        ) {
          state.deleted_patient_ids ??= [];
          if (pidRaw && !deletedPatientLower.has(pidLo)) {
            deletedPatientLower.add(pidLo);
            state.deleted_patient_ids.push(pidRaw);
          }
          if (hnRaw && !deletedPatientLower.has(hnLo)) {
            deletedPatientLower.add(hnLo);
            state.deleted_patient_ids.push(hnRaw);
          }
          return false;
        }
        return !deletedPatientLower.has(pidLo) && (!hnLo || !deletedPatientLower.has(hnLo));
      })
      .map((r: Record<string, unknown>) => {
      const pid = String(r.id);
      const prevPatient = prevPatientsById.get(pid);
      const history = (r.history as Record<string, unknown>) ?? {};
      const tpl = parsedTemplatesByPatient.get(pid);
      const b120 = biometrics120ByPatient.get(pid);
      const rowMasterVec =
        parseVec(r.master_embedding) ??
        parseVec(r.face_embedding) ??
        parseVec(r.biometric_data);

      const fallbackMaster =
        b120?.masterEmbedding ??
        rowMasterVec ??
        parseVec(history.masterEmbedding) ??
        tpl?.overall ??
        tpl?.frontal;
      const hasTemplates = Boolean((tpl && tpl.raw.length > 0) || fallbackMaster);
      const resolvedFaceStatus: 'PENDING' | 'VERIFIED' = hasTemplates ? 'VERIFIED' : 'PENDING';

      let fallbackAngles: AngleEmbeddings5D | undefined =
        b120?.angleEmbeddings ??
        parseAngleObject(r.angle_embeddings ?? r.face_embedding ?? r.biometric_data) ??
        parseAngleObject(history.angleEmbeddings);
      const fallbackExemplars = (() => {
        const fromB120 = b120?.exemplarEmbeddings ?? [];
        const fromRow = parseExemplarList(r.face_embedding ?? r.biometric_data);
        const fromHist = parseExemplarList(history.exemplarEmbeddings);
        const chosen = fromB120.length > 0 ? fromB120 : fromRow.length > 0 ? fromRow : fromHist;
        if (!fallbackMaster || chosen.length === 0) return undefined;
        const matchedDim = chosen.filter((v) => v.length === fallbackMaster.length);
        return matchedDim.length > 0 ? matchedDim : undefined;
      })();
      if (!fallbackAngles && fallbackMaster) {
        const frontVec = tpl?.raw.find((f) => f.pose === 'front')?.vector ?? tpl?.frontal ?? fallbackMaster;
        const leftVec = tpl?.raw.find((f) => f.pose === 'left')?.vector ?? fallbackMaster;
        const rightVec = tpl?.raw.find((f) => f.pose === 'right')?.vector ?? fallbackMaster;
        const nodVec = tpl?.raw.find((f) => f.pose === 'nod')?.vector ?? fallbackMaster;
        fallbackAngles = {
          center: frontVec,
          left: leftVec,
          right: rightVec,
          up: nodVec,
          down: nodVec,
        };
      }

      const resolvedFramesCount =
        (typeof b120?.framesCaptured === 'number' && b120.framesCaptured > 0
          ? b120.framesCaptured
          : undefined) ??
        (typeof r.face_frames_count === 'number' && r.face_frames_count > 0
          ? r.face_frames_count
          : undefined) ??
        (typeof history.faceFramesCount === 'number' && history.faceFramesCount > 0
          ? history.faceFramesCount
          : undefined) ??
        (resolvedFaceStatus === 'VERIFIED' ||
        String(r.face_biometric_status ?? '').toUpperCase() === 'VERIFIED' ||
        String(history.faceBiometricStatus ?? '').toUpperCase() === 'VERIFIED'
          ? 120
          : 0);

      const hydratedAngleImageUrls: PatientRecord['angleImageUrls'] = (() => {
        const raw = history.angleImageUrls;
        if (!raw || typeof raw !== 'object') return undefined;
        const rec = raw as Record<string, unknown>;
        const out: Partial<Record<'center' | 'left' | 'right' | 'up' | 'down', string>> = {};
        for (const k of ['center', 'left', 'right', 'up', 'down'] as const) {
          if (typeof rec[k] === 'string' && rec[k].trim().length > 0) {
            out[k] = rec[k];
          }
        }
        return Object.keys(out).length > 0 ? out : undefined;
      })();

      const hydratedFaceImageUrls: string[] | undefined = (() => {
        const list = Array.isArray(history.faceImageUrls)
          ? history.faceImageUrls.filter(
              (u): u is string => typeof u === 'string' && u.trim().length > 0,
            )
          : [];
        const angleList = hydratedAngleImageUrls
          ? (['center', 'left', 'right', 'up', 'down'] as const)
              .map((k) => hydratedAngleImageUrls[k])
              .filter((u): u is string => typeof u === 'string' && u.length > 0)
          : [];
        const combined = Array.from(new Set([...list, ...angleList]));
        return combined.length > 0 ? combined : undefined;
      })();

      const primaryFaceImageUrl =
        b120?.avatarUrl ??
        (typeof r.face_image_url === 'string' && r.face_image_url.trim().length > 0
          ? r.face_image_url
          : typeof history.faceImageUrl === 'string' && history.faceImageUrl.trim().length > 0
            ? history.faceImageUrl
            : hydratedFaceImageUrls?.[0]);

      const override = state.delegation_overrides?.[pid];
      const rawDelegationNote = r.delegation_note ? String(r.delegation_note) : null;
      const delegationMatch = rawDelegationNote?.match(/^\[(\d{4}-\d{2}-\d{2})\]\s*(.*)$/);
      const parsedColDate = delegationMatch ? delegationMatch[1] : null;
      const parsedColNote = delegationMatch ? (delegationMatch[2].trim() || null) : rawDelegationNote;

      const historyDelegatedToPtId =
        typeof history.delegatedToPtId === 'string' && history.delegatedToPtId
          ? history.delegatedToPtId
          : null;
      const historyDelegationDate =
        typeof history.delegationDate === 'string' && history.delegationDate
          ? history.delegationDate
          : null;
      const historyDelegationNote =
        typeof history.delegationNote === 'string' && history.delegationNote
          ? history.delegationNote
          : null;

      const hasHistoryDelegationKey = Object.prototype.hasOwnProperty.call(history, 'delegatedToPtId');
      const resolvedDelegatedToPtId = override
        ? override.delegatedToPtId
        : r.delegated_to_pt_id
          ? String(r.delegated_to_pt_id)
          : hasHistoryDelegationKey
            ? historyDelegatedToPtId
            : prevPatient?.delegatedToPtId ?? null;

      const resolvedDelegationDate = resolvedDelegatedToPtId
        ? override?.delegationDate ??
          parsedColDate ??
          historyDelegationDate ??
          prevPatient?.delegationDate ??
          null
        : null;

      const resolvedDelegationNote = resolvedDelegatedToPtId
        ? override
          ? override.delegationNote
          : parsedColNote ?? historyDelegationNote ?? prevPatient?.delegationNote ?? null
        : null;

      const hydratedStrokeType = (
        (typeof r.stroke_type === 'string' && r.stroke_type) ||
        (typeof history.strokeType === 'string' && history.strokeType) ||
        prevPatient?.strokeType ||
        'ELDERLY_GENERAL'
      ) as StrokeType;
      const hydratedAffectedSide = (
        (typeof r.affected_side === 'string' && r.affected_side) ||
        (typeof history.affectedSide === 'string' && history.affectedSide) ||
        prevPatient?.affectedSide ||
        'NONE'
      ) as AffectedSide;
      const hydratedMotorScore =
        typeof r.motor_assessment_score === 'number'
          ? r.motor_assessment_score
          : typeof history.motorAssessmentScore === 'number'
            ? history.motorAssessmentScore
            : prevPatient?.motorAssessmentScore ?? 3;
      const hydratedAdlScore =
        typeof r.adl_dressing_score === 'number'
          ? r.adl_dressing_score
          : typeof history.adlDressingScore === 'number'
            ? history.adlDressingScore
            : prevPatient?.adlDressingScore ?? 3;
      const hydratedMaxSafeRom =
        typeof r.max_safe_rom_deg === 'number'
          ? r.max_safe_rom_deg
          : typeof history.maxSafeRomDeg === 'number'
            ? history.maxSafeRomDeg
            : prevPatient?.maxSafeRomDeg ?? 150;
      const hydratedLinkCode =
        (typeof r.universal_health_link_code === 'string' && r.universal_health_link_code.trim()) ||
        (typeof history.universalHealthLinkCode === 'string' && history.universalHealthLinkCode.trim()) ||
        prevPatient?.universalHealthLinkCode ||
        `DTC-EHR-${pid.replace(/[^a-zA-Z0-9]/g, '').slice(0, 8).toUpperCase()}`;

      const mergedPrimaryFaceImageUrl =
        primaryFaceImageUrl ?? prevPatient?.faceImageUrl;
      const mergedFaceImageUrls =
        hydratedFaceImageUrls ?? prevPatient?.faceImageUrls;
      const mergedAngleImageUrls =
        hydratedAngleImageUrls ?? prevPatient?.angleImageUrls;
      const mergedMaster = fallbackMaster ?? prevPatient?.masterEmbedding;
      const mergedAngles = fallbackAngles ?? prevPatient?.angleEmbeddings;
      const mergedExemplars = fallbackExemplars ?? prevPatient?.exemplarEmbeddings;
      const hydratedGeomRatios =
        history.geometricRatios && typeof history.geometricRatios === 'object'
          ? (history.geometricRatios as GeometricAnthropometricRatios)
          : prevPatient?.geometricRatios;
      const hasRealMergedVector = Boolean(mergedMaster && mergedMaster.length >= 128);
      const mergedEmbeddingDim = mergedMaster?.length ?? 0;
      const mergedEngine =
        normalizeEngineTag(b120?.biometricEngine, mergedMaster) ??
        normalizeEngineTag(history.biometricEngine, mergedMaster) ??
        normalizeEngineTag(prevPatient?.biometricEngine, mergedMaster);
      const isCompatibleCurrentDim =
        hasRealMergedVector &&
        mergedEmbeddingDim >= ACTIVE_BIOMETRIC_DIM &&
        (!mergedEngine || mergedEngine === 'human-faceres-1024d');
      const biometricStatusDetail: 'VERIFIED' | 'PENDING' | 'REENROLL_REQUIRED' = !hasRealMergedVector
        ? 'PENDING'
        : isCompatibleCurrentDim
          ? 'VERIFIED'
          : 'REENROLL_REQUIRED';
      const biometricPipelineVersion = !hasRealMergedVector
        ? undefined
        : mergedEmbeddingDim >= ACTIVE_BIOMETRIC_DIM
          ? hydratedGeomRatios?.ocularIrisPatch256?.length === 256
            ? ACTIVE_BIOMETRIC_PIPELINE_VERSION
            : 'v2-pre-iris-1024d'
          : 'legacy-512d';

      const inspectedFull = inspectPatientRecordHydration(r, {
        biometricsRow: {
          masterEmbedding: mergedMaster,
          angleEmbeddings: mergedAngles,
          exemplarEmbeddings: mergedExemplars,
          biometricEngine: mergedEngine,
          framesCaptured: resolvedFramesCount,
          avatarUrl: mergedPrimaryFaceImageUrl,
          geometricRatios: hydratedGeomRatios,
        },
        prevPatient,
        strictCurrentDimOnly: true,
        requireRealVectorsForVerified: true,
        applyClinicalDefaults: true,
      });
      if (inspectedFull.needsTopLevelRepair && inspectedFull.topLevelRepairPatch) {
        void schedulePatientTopLevelRepairInBackground(supabase, pid, inspectedFull.topLevelRepairPatch);
      }

      return {
        id: pid,
        hn: String(r.hn),
        fullName: String(r.full_name),
        age: Number(r.age ?? 0),
        gender: typeof r.gender === 'string' && r.gender.trim() ? r.gender : typeof history.gender === 'string' && history.gender.trim() ? history.gender : 'ชาย',
        phone: String(r.phone ?? ''),
        diagnosis: String(r.diagnosis ?? ''),
        strokeType: hydratedStrokeType,
        affectedSide: hydratedAffectedSide,
        motorAssessmentScore: hydratedMotorScore,
        adlDressingScore: hydratedAdlScore,
        maxSafeRomDeg: hydratedMaxSafeRom,
        universalHealthLinkCode: hydratedLinkCode,
        rehabSessionHistory: Array.isArray(history.rehabSessionHistory)
          ? (history.rehabSessionHistory as RehabSessionHistoryEntry[])
          : prevPatient?.rehabSessionHistory,
        clinicalAssessments: Array.isArray(history.clinicalAssessments)
          ? (history.clinicalAssessments as ClinicalAssessmentRecord[])
          : prevPatient?.clinicalAssessments,
        assignedPtId: String(r.assigned_pt_id ?? ''),
        delegatedToPtId: resolvedDelegatedToPtId,
        delegationNote: resolvedDelegationNote,
        delegationDate: resolvedDelegationDate,
        nextAppointmentDate: String(r.next_appointment_date ?? ''),
        registrationStatus:
          String(r.registration_status ?? '').trim().toLowerCase() === 'archived' ? 'archived' : 'active',
        enrolledAt: String(r.created_at ?? new Date().toISOString()),
        faceBiometricStatus: isCompatibleCurrentDim ? 'VERIFIED' : 'PENDING',
        biometricStatusDetail,
        biometricEmbeddingDim: mergedEmbeddingDim,
        biometricPipelineVersion,
        ...(inspectedFull.patient.biometricUpdatedAt ? { biometricUpdatedAt: inspectedFull.patient.biometricUpdatedAt } : {}),
        faceFramesCount:
          hasRealMergedVector ||
          (mergedAngleImageUrls && Object.keys(mergedAngleImageUrls).length >= 5)
            ? Math.max(resolvedFramesCount, prevPatient?.faceFramesCount ?? 0, 120)
            : 0,
        faceImageUrl: mergedPrimaryFaceImageUrl,
        faceImageUrls: mergedFaceImageUrls,
        angleImageUrls: mergedAngleImageUrls,
        masterEmbedding: mergedMaster,
        angleEmbeddings: mergedAngles,
        ...(mergedExemplars ? { exemplarEmbeddings: mergedExemplars } : {}),
        biometricEngine: mergedEngine,
        geometricRatios: hydratedGeomRatios,
        rehabSchedules: Array.isArray(history.rehabSchedules)
          ? sanitizeAccidentalDefaultKneePrefix(history.rehabSchedules as RehabScheduleItem[])
          : prevPatient?.rehabSchedules ?? [],
        biometricHistory:
          tpl && tpl.frontal && tpl.overall
            ? {
                frontal_centroid: tpl.frontal,
                overall_centroid: tpl.overall,
                raw_frames: tpl.raw,
              }
            : mergedMaster
              ? {
                  frontal_centroid: mergedAngles?.center ?? mergedMaster,
                  overall_centroid: mergedMaster,
                  raw_frames: [],
                }
              : undefined,
      };
    });

    // Preserve ONLY recently created in-memory patients that failed Postgres flush (e.g. HTTP 544 outage),
    // never resurrecting patients deleted on another Worker isolate.
    const nowMs = Date.now();
    const dbSeenIds = new Set(state.patients.map((p) => p.id.trim().toLowerCase()));
    const dbSeenHns = new Set(state.patients.map((p) => p.hn.trim().toLowerCase()).filter(Boolean));
    for (const [prevId, prevP] of prevPatientsById.entries()) {
      const pidLo = prevId.trim().toLowerCase();
      const hnLo = (prevP.hn ?? '').trim().toLowerCase();
      const pendingFlushAt = (prevP as { pendingDbFlushAt?: number }).pendingDbFlushAt;
      const isPendingFlush = typeof pendingFlushAt === 'number' && nowMs - pendingFlushAt < 120_000;
      if (
        !isPendingFlush ||
        (prevP.registrationStatus ?? 'active') !== 'active' ||
        deletedPatientSet.has(prevId) ||
        deletedPatientLower.has(pidLo) ||
        (hnLo && deletedPatientLower.has(hnLo)) ||
        dbSeenIds.has(pidLo) ||
        (hnLo && dbSeenHns.has(hnLo))
      ) {
        continue;
      }
      state.patients.push(prevP);
    }
  } else {
    state.patients = state.patients.filter(
      (p) =>
        !deletedPatientSet.has(p.id) &&
        !deletedPatientLower.has(p.id.toLowerCase()) &&
        (!p.hn || !deletedPatientLower.has(p.hn.toLowerCase())),
    );
  }

  // Strictly only retain templates belonging to active, non-deleted rows present in state.patients
  const activePatientMap = new Map<string, string>();
  for (const p of state.patients) {
    if (
      p.registrationStatus === 'active' &&
      !deletedPatientSet.has(p.id) &&
      !deletedPatientLower.has(p.id.toLowerCase()) &&
      (!p.hn || !deletedPatientLower.has(p.hn.toLowerCase()))
    ) {
      activePatientMap.set(p.id, p.fullName);
    }
  }

  if (templatesRes.status === 'fulfilled' && !templatesRes.value.error && Array.isArray(templatesRes.value.data)) {
    state.face_templates = parsedTemplateCandidates
      .filter((t) => activePatientMap.has(t.patientId))
      .map((t) => ({
        ...t,
        patientName: activePatientMap.get(t.patientId),
      }));
  } else {
    state.face_templates = state.face_templates.filter((t) => activePatientMap.has(t.patientId));
  }

  // Synthesize face_templates for any active patient who lacks rows in face_templates
  const templatedPatientIds = new Set(state.face_templates.map((t) => t.patientId));
  for (const p of state.patients) {
    if (!activePatientMap.has(p.id) || templatedPatientIds.has(p.id)) continue;
    if (p.angleEmbeddings?.center) {
      state.face_templates.push({
        patientId: p.id,
        patientName: p.fullName,
        pose: 'frontal_centroid',
        vector: p.angleEmbeddings.center,
      });
    }
    if (p.masterEmbedding) {
      state.face_templates.push({
        patientId: p.id,
        patientName: p.fullName,
        pose: 'overall_centroid',
        vector: p.masterEmbedding,
      });
    }
  }

  if (notesRes.status === 'fulfilled' && Array.isArray(notesRes.value.data)) {
    state.medical_notes = notesRes.value.data.map((r: Record<string, unknown>) => ({
      id: String(r.id),
      patientId: String(r.patient_id),
      authorId: String(r.author_id),
      authorName: String(r.author_name),
      content: String(r.content),
      createdAt: String(r.created_at),
    }));
  }

  if (coursesRes.status === 'fulfilled' && Array.isArray(coursesRes.value.data)) {
    state.patient_courses = coursesRes.value.data.map((r: Record<string, unknown>) => ({
      id: String(r.id),
      patientId: String(r.patient_id),
      title: String(r.title),
      weeklySchedule: (r.weekly_schedule as DailyScheduleDay[]) ?? [],
      cognitiveModpackId: String(r.cognitive_modpack_id ?? 'modpack-core-99'),
      updatedAt: String(r.updated_at ?? new Date().toISOString()),
    }));
  }

  // Strip legacy auto-generated default training schedules / nextAppointmentDate when unscheduled
  const legacyDefaultLabels = new Set([
    'Cognitive Rehab Card Game',
    'Physical Rehab (Pending AI Unlock)',
    'พักฟื้น',
  ]);
  for (const patient of state.patients) {
    if ((patient.rehabSchedules ?? []).length > 0) continue;
    const course = state.patient_courses.find((c) => c.patientId === patient.id);
    const isLegacyAutoCourse =
      !course ||
      ((course.title.endsWith('— แผนฟื้นฟูสมรรถภาพ') || course.title === 'ล้างตารางนัดชั่วคราว') &&
        course.weeklySchedule.every(
          (d) => (d.exercises ?? []).length === 0 && legacyDefaultLabels.has(d.label),
        ));
    if (isLegacyAutoCourse) {
      patient.nextAppointmentDate = '';
      if (course) {
        course.weeklySchedule = course.weeklySchedule.map((d) => ({
          ...d,
          dayType: 'rest',
          label: 'พักฟื้น',
          exercises: [],
        }));
      }
    }
  }

  if (sessionsRes.status === 'fulfilled' && Array.isArray(sessionsRes.value.data)) {
    state.rehab_sessions = sessionsRes.value.data.map((r: Record<string, unknown>) => {
      const exId = String(r.exercise_id ?? '');
      const hasMinigameScore = r.minigame_score !== null && r.minigame_score !== undefined;
      const isMinigame = hasMinigameScore || /minigame|cognitive-card-game|\bcard\b/i.test(exId);
      const matchedCatalog = CLINICAL_EXERCISES.find(
        (ex) => ex.id === exId || ex.code === exId,
      );
      return {
        id: String(r.id),
        patientId: String(r.patient_id),
        activityType: isMinigame ? 'minigame' : 'physio_exercise',
        exerciseId: exId,
        exerciseName: matchedCatalog?.name?.th ?? (isMinigame ? 'มินิเกมฝึกสมอง (การ์ดกายภาพ)' : exId),
        completedSets: Number(r.completed_sets ?? 0),
        completedReps: Number(r.completed_reps ?? 0),
        accuracyPercent: Number(r.accuracy_percent ?? 100),
        maxRomDeg: Number(r.max_rom_deg ?? 0),
        painStopTriggered: Boolean(r.pain_stop_triggered),
        aiFeedback: 'Synced from Supabase',
        kinematicsReplay: (r.kinematics_replay as { tMs: number; angleDeg: number }[]) ?? [],
        minigameScore: r.minigame_score as { correct: number; total: number; category: string } | undefined,
        createdAt: String(r.created_at),
      };
    });
  }

  const mergedHistoryMap = new Map<string, RehabSessionHistoryEntry>();
  for (const h of state.rehab_session_history ?? []) {
    mergedHistoryMap.set(h.id, h);
  }
  for (const p of state.patients) {
    for (const h of p.rehabSessionHistory ?? []) {
      mergedHistoryMap.set(h.id, h);
    }
  }
  if (
    sessionHistoryRes.status === 'fulfilled' &&
    !sessionHistoryRes.value.error &&
    Array.isArray(sessionHistoryRes.value.data)
  ) {
    for (const r of sessionHistoryRes.value.data as Record<string, unknown>[]) {
      const pid = String(r.patient_id ?? '');
      if (!pid || deletedPatientSet.has(pid)) continue;
      const entry: RehabSessionHistoryEntry = {
        id: String(r.id),
        patientId: pid,
        hn: String(r.hn ?? ''),
        sessionDate: String(r.session_date ?? r.created_at ?? new Date().toISOString()),
        exerciseCode: String(r.exercise_code ?? ''),
        exerciseName: String(r.exercise_name ?? r.exercise_code ?? ''),
        jointName: String(r.joint_name ?? 'upper_limb'),
        bodySide: (typeof r.body_side === 'string' ? r.body_side : 'BOTH') as AffectedSide,
        baselineRomDeg: Number(r.baseline_rom_deg ?? 0),
        achievedMaxRomDeg: Number(r.achieved_max_rom_deg ?? 0),
        romDeltaDeg: Number(r.rom_delta_deg ?? 0),
        motorScoreAtSession:
          typeof r.motor_score_at_session === 'number' ? r.motor_score_at_session : null,
        validReps: Number(r.valid_reps ?? 0),
        compensationsCount: Number(r.compensations_count ?? 0),
        facilityName: String(r.facility_name ?? 'Dr.Tech.Care Clinic'),
        recordedByPtId: r.recorded_by_pt_id ? String(r.recorded_by_pt_id) : null,
        recordedByPtName: r.recorded_by_pt_name ? String(r.recorded_by_pt_name) : null,
        clinicalNotes: r.clinical_notes ? String(r.clinical_notes) : null,
      };
      mergedHistoryMap.set(entry.id, entry);
    }
  }
  state.rehab_session_history = Array.from(mergedHistoryMap.values()).sort(
    (a, b) => new Date(b.sessionDate).getTime() - new Date(a.sessionDate).getTime(),
  );

  // Merge persisted clinical assessments from BOTH `patients.history.clinicalAssessments` JSONB and `patient_clinical_assessments` table
  const deletedAssessmentSet = new Set(state.deleted_assessment_ids ?? []);
  const mergedAssessmentsMap = new Map<string, ClinicalAssessmentRecord>();
  for (const prev of state.clinical_assessments ?? []) {
    if (!prev?.id || deletedAssessmentSet.has(prev.id) || deletedPatientSet.has(prev.patient_id)) continue;
    mergedAssessmentsMap.set(prev.id, prev);
  }
  for (const p of state.patients) {
    for (const embedded of p.clinicalAssessments ?? []) {
      if (!embedded?.id || deletedAssessmentSet.has(embedded.id) || deletedPatientSet.has(p.id)) continue;
      mergedAssessmentsMap.set(embedded.id, {
        ...embedded,
        patient_id: embedded.patient_id || p.id,
        patient_hn: embedded.patient_hn || p.hn,
        universal_link_code: embedded.universal_link_code || p.universalHealthLinkCode || '',
      });
    }
  }
  if (
    assessmentsRes.status === 'fulfilled' &&
    !assessmentsRes.value.error &&
    Array.isArray(assessmentsRes.value.data)
  ) {
    for (const r of assessmentsRes.value.data as Record<string, unknown>[]) {
      const id = String(r.id ?? '');
      const pid = String(r.patient_id ?? '');
      if (!id || !pid || deletedAssessmentSet.has(id) || deletedPatientSet.has(pid)) continue;
      const sideRaw = String(r.target_side ?? 'RIGHT').toUpperCase();
      const targetSide: ClinicalAssessmentRecord['target_side'] =
        sideRaw === 'LEFT' ? 'LEFT' : sideRaw === 'BOTH' ? 'BOTH' : 'RIGHT';
      const channelRaw = String(r.source_channel ?? 'STAFF_MANUAL_ASSESSMENT');
      const sourceChannel: ClinicalAssessmentRecord['source_channel'] =
        channelRaw === 'KIOSK_SESSION_AUTO_SYNC' ? 'KIOSK_SESSION_AUTO_SYNC' : 'STAFF_MANUAL_ASSESSMENT';
      const rawScore = Number(r.initial_score_0_to_6 ?? 0);
      const clampedScore = Math.max(0, Math.min(6, Math.round(Number.isFinite(rawScore) ? rawScore : 0))) as
        | 0
        | 1
        | 2
        | 3
        | 4
        | 5
        | 6;
      const existingRec = mergedAssessmentsMap.get(id);
      mergedAssessmentsMap.set(id, {
        id,
        patient_id: pid,
        patient_hn: String(r.patient_hn ?? existingRec?.patient_hn ?? ''),
        universal_link_code: String(r.universal_link_code ?? existingRec?.universal_link_code ?? ''),
        visit_number: Math.max(1, Number(r.visit_number ?? existingRec?.visit_number ?? 1)),
        assessed_at: String(r.assessed_at ?? r.created_at ?? existingRec?.assessed_at ?? new Date().toISOString()),
        facility_name: String(
          r.facility_name ?? existingRec?.facility_name ?? 'ศูนย์กายภาพบำบัด Dr.Tech.Care สาขาหลัก',
        ),
        assessor_staff_id: String(r.assessor_staff_id ?? existingRec?.assessor_staff_id ?? ''),
        assessor_staff_name: String(r.assessor_staff_name ?? existingRec?.assessor_staff_name ?? ''),
        exercise_code: String(r.exercise_code ?? existingRec?.exercise_code ?? 'STROKE_SHOULDER_FLEXION'),
        exercise_name_th: String(r.exercise_name_th ?? existingRec?.exercise_name_th ?? ''),
        target_side: targetSide,
        initial_score_0_to_6: clampedScore,
        measured_rom_deg: Number(Number(r.measured_rom_deg ?? 0).toFixed(1)),
        target_rom_deg: Number(Number(r.target_rom_deg ?? 120).toFixed(1)),
        pain_vas_0_to_10:
          r.pain_vas_0_to_10 === null || r.pain_vas_0_to_10 === undefined
            ? existingRec?.pain_vas_0_to_10 ?? null
            : Number(r.pain_vas_0_to_10),
        chief_complaint: String(r.chief_complaint ?? existingRec?.chief_complaint ?? ''),
        clinical_handover_note: String(r.clinical_handover_note ?? existingRec?.clinical_handover_note ?? ''),
        next_session_goal: String(r.next_session_goal ?? existingRec?.next_session_goal ?? ''),
        source_channel: sourceChannel,
        measurement_method:
          existingRec?.measurement_method ??
          (sourceChannel === 'KIOSK_SESSION_AUTO_SYNC' ? 'KIOSK_SESSION_AUTO_SYNC' : 'MANUAL_INPUT'),
        completed_reps: existingRec?.completed_reps ?? null,
        accuracy_percent: existingRec?.accuracy_percent ?? null,
        session_id: existingRec?.session_id ?? null,
        created_at: r.created_at ? String(r.created_at) : existingRec?.created_at,
      });
    }
  }
  state.clinical_assessments = Array.from(mergedAssessmentsMap.values()).sort(
    (a, b) => new Date(a.assessed_at).getTime() - new Date(b.assessed_at).getTime(),
  );

  for (const patient of state.patients) {
    if ((patient.rehabSchedules ?? []).length > 0) {
      const course = state.patient_courses.find((c) => c.patientId === patient.id);
      const todaySchedule = resolveTodayPatientSchedule({
        rehabSchedules: patient.rehabSchedules,
        rehabSessions: state.rehab_sessions.filter((s) => s.patientId === patient.id),
        weeklySchedule: course?.weeklySchedule,
        fallbackNextAppointmentDate: patient.nextAppointmentDate,
      });
      patient.nextAppointmentDate = todaySchedule.nextAppointmentDate;
    }
  }

  if (reportsRes.status === 'fulfilled' && Array.isArray(reportsRes.value.data)) {
    state.treatment_reports = reportsRes.value.data.map((r: Record<string, unknown>) => ({
      id: String(r.id),
      patientId: String(r.patient_id),
      ptId: String(r.pt_id),
      ptName: String(r.pt_name),
      summary: String(r.summary),
      aiRecommendation: String(r.ai_recommendation ?? ''),
      scoreSummary: String(r.score_summary ?? ''),
      createdAt: String(r.created_at),
      updatedAt: String(r.updated_at),
    }));
  }

  const canonicalMap = new Map(CLINICAL_EXERCISES.map((c) => [c.code, c]));
  if (exercisesRes.status === 'fulfilled' && Array.isArray(exercisesRes.value.data)) {
    const validCustomOrDb: ManagedExerciseRecord[] = [];
    for (const r of exercisesRes.value.data as Record<string, unknown>[]) {
      const id = String(r.id ?? '');
      const code = String(r.code ?? id);
      const category = (r.category as ManagedExerciseRecord['category']) ?? 'upper_limb';
      if (isDeprecatedExerciseRecord({ id, code, category })) {
        continue;
      }
      if (canonicalMap.has(code)) {
        continue;
      }
      validCustomOrDb.push({
        id,
        code,
        category,
        name: (r.name_i18n as ManagedExerciseRecord['name']) ?? {
          th: String(r.name_th ?? code),
          en: String(r.name_th ?? code),
          cn: String(r.name_th ?? code),
        },
        instruction: (r.instruction_i18n as ManagedExerciseRecord['instruction']) ?? {
          th: String(r.instructions_th ?? ''),
          en: String(r.instructions_th ?? ''),
          cn: String(r.instructions_th ?? ''),
        },
        detection: (r.detection_config as ManagedExerciseRecord['detection']) ?? {
          mode: 'angle_rep_hold',
          jointGroup: 'shoulder',
          side: 'right',
          targetMinDeg: 30,
          targetMaxDeg: 135,
          startRestMaxDeg: 25,
          holdSec: 2,
          defaultSets: 2,
          defaultReps: 8,
        },
        createdBy: 'system',
      });
    }
    const canonicalRecords: ManagedExerciseRecord[] = CLINICAL_EXERCISES.map((ex) => ({
      ...ex,
      createdBy: 'system',
    }));
    state.exercises = mergeDuplicateSideExercises(
      sortExercisesStrokeUpperFirst([...canonicalRecords, ...validCustomOrDb]),
    );
  } else {
    state.exercises = sortExercisesStrokeUpperFirst(
      CLINICAL_EXERCISES.map((ex) => ({ ...ex, createdBy: 'system' })),
    );
  }

  if (modpacksRes.status === 'fulfilled' && Array.isArray(modpacksRes.value.data) && modpacksRes.value.data.length > 0) {
    state.modpacks = modpacksRes.value.data.map((r: Record<string, unknown>) => {
      const id = String(r.id);
      const seededDefaultCount =
        id === 'modpack-core-99' ? 99 : id === 'modpack-memory-safety' ? 42 : 10;
      const rawQuestionsCount =
        Array.isArray(r.questions) && r.questions.length > 0 ? r.questions.length : null;
      return {
        id,
        name: String(r.name),
        categories: Array.isArray(r.categories) ? (r.categories as string[]) : [String(r.category ?? 'cognitive')],
        questionCount: Number(r.card_count ?? rawQuestionsCount ?? seededDefaultCount),
        createdBy: 'system',
        active: Boolean(r.active ?? true),
      };
    });
  }

  const callStatusRank: Record<StaffCallRecord['status'], number> = {
    open: 0,
    acknowledged: 1,
    resolved: 2,
  };

  if (callsRes.status === 'fulfilled' && !callsRes.value.error && Array.isArray(callsRes.value.data)) {
    const deletedCallSet = new Set(state.deleted_staff_call_ids ?? []);
    const mergedCalls = new Map<string, StaffCallRecord>();
    for (const c of state.staff_calls) {
      if (!deletedCallSet.has(c.id)) {
        mergedCalls.set(c.id, c);
      }
    }
    for (const r of callsRes.value.data as Record<string, unknown>[]) {
      const id = String(r.id);
      if (deletedCallSet.has(id)) continue;
      const rawReason = String(r.reason ?? '');
      const rawStatus = String(r.status ?? 'open');
      const isAcked = rawStatus === 'open' && rawReason.startsWith('[ACKNOWLEDGED]');
      const dbStatus: StaffCallRecord['status'] = isAcked
        ? 'acknowledged'
        : rawStatus === 'resolved' || rawStatus === 'acknowledged' || rawStatus === 'open'
          ? rawStatus
          : 'open';
      const cleanReason = isAcked ? rawReason.replace(/^\[ACKNOWLEDGED\]\s*/, '') : rawReason;
      const prev = mergedCalls.get(id);
      const effectiveStatus =
        prev && callStatusRank[prev.status] > callStatusRank[dbStatus] ? prev.status : dbStatus;
      mergedCalls.set(id, {
        id,
        kioskId: String(r.kiosk_id),
        patientId: r.patient_id ? String(r.patient_id) : (prev?.patientId ?? null),
        reason:
          effectiveStatus === 'resolved'
            ? (prev?.reason || cleanReason).replace(/^\[ACKNOWLEDGED\]\s*/, '')
            : cleanReason || prev?.reason || '',
        status: effectiveStatus,
        createdAt: String(r.created_at ?? prev?.createdAt ?? new Date().toISOString()),
      });
    }
    state.staff_calls = Array.from(mergedCalls.values()).sort(
      (a, b) => (Date.parse(b.createdAt) || 0) - (Date.parse(a.createdAt) || 0),
    );
  }

  if (auditRes.status === 'fulfilled' && !auditRes.value.error && Array.isArray(auditRes.value.data)) {
    const mergedLogs = new Map<string, AuditLogRecord>();
    for (const log of state.audit_logs) {
      mergedLogs.set(log.id, log);
    }
    for (const r of auditRes.value.data as Record<string, unknown>[]) {
      const id = String(r.id);
      const detailObj = (r.detail as Record<string, unknown> | null) ?? (r.details as Record<string, unknown> | null);
      const detailMsg =
        typeof detailObj?.message === 'string'
          ? detailObj.message
          : typeof r.action === 'string'
            ? `${r.action} ${r.target ?? ''}`.trim()
            : JSON.stringify(detailObj ?? {});
      mergedLogs.set(id, {
        id,
        eventType: String(r.event_type ?? r.action ?? 'SYSTEM_EVENT'),
        actorRole: String(r.actor_role ?? 'system'),
        actorId: String(r.actor_id ?? ''),
        actorName: String(r.actor_name ?? ''),
        ipAddress: String(r.ip_address ?? 'unknown'),
        deviceId: String(r.device_id ?? 'unknown'),
        targetId: r.target_id ? String(r.target_id) : r.target ? String(r.target) : undefined,
        detail: detailMsg,
        createdAt: String(r.created_at ?? new Date().toISOString()),
      });
    }
    state.audit_logs = Array.from(mergedLogs.values())
      .sort((a, b) => (Date.parse(b.createdAt) || 0) - (Date.parse(a.createdAt) || 0))
      .slice(0, 1000);
  }

  if (bansRes.status === 'fulfilled' && !bansRes.value.error && Array.isArray(bansRes.value.data)) {
    const deletedBanSet = new Set(state.deleted_ban_ids ?? []);
    const mergedBans = new Map<string, BannedTargetRecord>();
    for (const b of state.banned_targets) {
      if (!deletedBanSet.has(b.id)) {
        mergedBans.set(b.id, b);
      }
    }
    for (const r of bansRes.value.data as Record<string, unknown>[]) {
      const id = String(r.id);
      if (deletedBanSet.has(id)) continue;
      mergedBans.set(id, {
        id,
        targetType: (r.target_type as 'ip' | 'device') ?? 'device',
        value: String(r.value),
        reason: String(r.reason ?? ''),
        bannedBy: String(r.banned_by ?? ''),
        createdAt: String(r.created_at ?? new Date().toISOString()),
      });
    }
    state.banned_targets = Array.from(mergedBans.values()).sort(
      (a, b) => (Date.parse(b.createdAt) || 0) - (Date.parse(a.createdAt) || 0),
    );
  }

  if (activeSessionsRes.status === 'fulfilled' && Array.isArray(activeSessionsRes.value.data)) {
    const merged = new Map<string, ActiveSessionRecord>();
    for (const s of state.active_sessions) {
      merged.set(s.token, s);
    }
    for (const r of activeSessionsRes.value.data as Record<string, unknown>[]) {
      const token = String(r.token);
      const prev = merged.get(token);
      merged.set(token, {
        token,
        userId: String(r.user_id),
        role: normalizeStaffRole(r.role),
        ipAddress: String(r.ip_address ?? 'unknown'),
        deviceId: String(r.device_id ?? 'unknown'),
        createdAt: String(r.created_at),
        revoked: Boolean(prev?.revoked) || Boolean(r.revoked),
      });
    }
    state.active_sessions = Array.from(merged.values()).sort(
      (a, b) => (Date.parse(b.createdAt) || 0) - (Date.parse(a.createdAt) || 0),
    );
  }

  if (settingsRes.status === 'fulfilled' && settingsRes.value.data) {
    const s = settingsRes.value.data as Record<string, unknown>;
    state.system_settings = {
      aiFaceScanEnabled: Boolean(s.ai_face_scan_enabled ?? true),
      aiFaceThreshold: Number(s.ai_face_threshold ?? 0.925),
      aiPoseDetectionEnabled: Boolean(s.ai_pose_detection_enabled ?? false),
      kioskAutoResetSec: Number(s.kiosk_auto_reset_sec ?? 6),
    };
  }

  if (kiosksRes.status === 'fulfilled' && Array.isArray(kiosksRes.value.data) && kiosksRes.value.data.length > 0) {
    state.kiosks = kiosksRes.value.data.map((r: Record<string, unknown>) => {
      const rawState = String(r.current_state ?? 'STANDBY');
      const [parsedState, parsedToken] = rawState.includes(':') ? rawState.split(':', 2) : [rawState, undefined];
      return {
        kioskId: String(r.kiosk_id),
        location: String(r.location ?? ''),
        online: Boolean(r.online ?? true),
        currentState: parsedState || 'STANDBY',
        pairingToken: typeof r.pairing_token === 'string' ? r.pairing_token : parsedToken,
        lastHeartbeatAt: String(r.last_heartbeat_at ?? new Date().toISOString()),
      };
    });
  }

  if (!isSupabaseCircuitOpen(supabase)) {
    lastHydratedAtByState.set(state, Date.now());
  }
}

export async function persistToSupabase(
  supabase: SupabaseClient | null,
  task: (client: SupabaseClient) => PromiseLike<unknown>,
): Promise<boolean> {
  if (!supabase || isSupabaseCircuitOpen(supabase)) return false;
  try {
    const res = (await task(supabase)) as { error?: unknown } | undefined;
    if (res && typeof res === 'object' && res.error && isSupabaseTransientFailure(res.error)) {
      markSupabaseDown('persist_transient_error', supabase);
      return false;
    }
    return true;
  } catch (err) {
    if (isSupabaseTransientFailure(err)) {
      markSupabaseDown('persist_exception', supabase);
    }
    // Non-fatal fallback for offline / missing column / constraint edge cases
    return false;
  }
}

/**
 * Bounded best-effort persistence wrapper for write endpoints (`POST`/`PUT`/`PATCH`):
 * awaits Supabase writes up to `timeoutMs` inside try/catch, trips 20s circuit breaker on 5xx/544/timeout,
 * and NEVER throws HTTP 500 back to the client.
 */
export async function persistWithTimeoutBestEffort(
  supabase: SupabaseClient | null,
  task: (client: SupabaseClient) => PromiseLike<unknown>,
  timeoutMs = 1500,
  label = 'supabase_write',
): Promise<{ persisted: boolean; errorMessage?: string }> {
  if (!supabase) return { persisted: false };
  if (isSupabaseCircuitOpen(supabase)) {
    return { persisted: false, errorMessage: 'supabase_circuit_open' };
  }
  const boundedMs = Math.min(timeoutMs, 1500);
  let timerId: ReturnType<typeof setTimeout> | undefined;
  try {
    const outcome = await Promise.race([
      Promise.resolve(task(supabase)).then(
        (res) => ({ timedOut: false as const, res }),
        (err) => ({ timedOut: false as const, err }),
      ),
      new Promise<{ timedOut: true }>((resolve) => {
        timerId = setTimeout(() => resolve({ timedOut: true }), boundedMs);
      }),
    ]);
    if (outcome.timedOut) {
      markSupabaseDown(`${label}_timeout`, supabase);
      console.warn(`[cloudRouter] ${label} timed out after ${boundedMs}ms (circuit opened 20s; kept in-memory state)`);
      return { persisted: false, errorMessage: 'timeout' };
    }
    if ('err' in outcome && outcome.err) {
      const msg = outcome.err instanceof Error ? outcome.err.message : String(outcome.err);
      if (isSupabaseTransientFailure(outcome.err)) {
        markSupabaseDown(`${label}_exception`, supabase);
      }
      console.warn(`[cloudRouter] ${label} failed cleanly (non-fatal):`, msg);
      return { persisted: false, errorMessage: msg };
    }
    const rawRes = 'res' in outcome ? (outcome.res as { error?: { message?: string } | null } | null) : null;
    if (rawRes && typeof rawRes === 'object' && rawRes.error) {
      const msg = rawRes.error.message || JSON.stringify(rawRes.error);
      if (isSupabaseTransientFailure(rawRes.error)) {
        markSupabaseDown(`${label}_postgrest_5xx`, supabase);
      }
      console.warn(`[cloudRouter] ${label} PostgREST warning (non-fatal):`, msg);
      return { persisted: false, errorMessage: msg };
    }
    return { persisted: true };
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    if (isSupabaseTransientFailure(err)) {
      markSupabaseDown(`${label}_unexpected`, supabase);
    }
    console.warn(`[cloudRouter] ${label} unexpected error (non-fatal):`, msg);
    return { persisted: false, errorMessage: msg };
  } finally {
    if (timerId !== undefined) clearTimeout(timerId);
  }
}

/**
 * Strict persistence wrapper for biometric & critical patient mutations:
 * never swallows PostgREST `{ error }` responses or exceptions.
 */
export async function persistBiometricStrictToSupabase(
  supabase: SupabaseClient | null,
  task: (client: SupabaseClient) => PromiseLike<unknown>,
): Promise<void> {
  if (!supabase || isSupabaseCircuitOpen(supabase)) return;
  const result = (await task(supabase)) as { error?: { message?: string } | null } | undefined;
  if (result && typeof result === 'object' && 'error' in result && result.error) {
    if (isSupabaseTransientFailure(result.error)) {
      markSupabaseDown('biometric_strict_5xx', supabase);
    }
    throw new Error(result.error.message || 'Supabase biometric write failed');
  }
}

const UUID_REGEX = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

const KNOWN_SEEDED_UUID_ALIASES: Record<string, string> = {
  'user-super-1': '00000000-0000-4000-8000-000000000001',
  admin: '00000000-0000-4000-8000-000000000001',
  'user-pt-1': '00000000-0000-4000-8000-000000000002',
  pt_somchai: '00000000-0000-4000-8000-000000000002',
  'user-pt-2': '00000000-0000-4000-8000-000000000003',
  pt_waranya: '00000000-0000-4000-8000-000000000003',
};

export function isUuidString(value: string | null | undefined): boolean {
  return typeof value === 'string' && UUID_REGEX.test(value.trim());
}

/**
 * Resolves any user/entity reference to a valid Postgres UUID or `null` so UUID columns
 * (`assigned_pt_id`, `delegated_to_pt_id`, `enrolled_by`, `user_id`) NEVER trigger 22P02 errors.
 */
export function toSafeUuidOrNull(
  value: string | null | undefined,
  staffUsers?: ReadonlyArray<Pick<StoredUserAccount, 'id' | 'username'>>,
): string | null {
  if (typeof value !== 'string') return null;
  const trimmed = value.trim();
  if (!trimmed) return null;
  if (UUID_REGEX.test(trimmed)) return trimmed;
  const mappedAlias = KNOWN_SEEDED_UUID_ALIASES[trimmed.toLowerCase()];
  if (mappedAlias) return mappedAlias;
  if (Array.isArray(staffUsers)) {
    const lower = trimmed.toLowerCase();
    const matched = staffUsers.find(
      (u) =>
        u.id.toLowerCase() === lower ||
        u.username.toLowerCase() === lower,
    );
    if (matched && UUID_REGEX.test(matched.id)) {
      return matched.id;
    }
  }
  return null;
}

/**
 * Caps biometric exemplar vectors to at most `maxCount` (default 9) L2-normalized anchors
 * of matching dimension so Postgres payloads stay compact (< 15KB) and fast.
 */
export function normalizeCompactExemplars(
  vectors: unknown,
  expectedDim: number,
  maxCount = 9,
): number[][] {
  if (!Array.isArray(vectors) || expectedDim < 128) return [];
  const out: number[][] = [];
  for (const rawVec of vectors) {
    if (!Array.isArray(rawVec) || rawVec.length !== expectedDim) continue;
    let sumSq = 0;
    let valid = true;
    for (let i = 0; i < rawVec.length; i += 1) {
      const n = Number(rawVec[i]);
      if (!Number.isFinite(n)) {
        valid = false;
        break;
      }
      sumSq += n * n;
    }
    if (!valid || sumSq <= 0) continue;
    const norm = Math.sqrt(sumSq);
    out.push(rawVec.map((v) => Number((Number(v) / norm).toFixed(6))));
    if (out.length >= maxCount) break;
  }
  return out;
}

/** Safely upserts a `patient_courses` row into Postgres, handling missing modpack FKs (23503) and schema drift. */
export async function upsertCourseSafeToSupabase(
  db: SupabaseClient,
  row: {
    id: string;
    patient_id: string;
    title: string;
    weekly_schedule: unknown;
    cognitive_modpack_id?: string | null;
    updated_at?: string;
  },
): Promise<void> {
  const safePatientUuid = toSafeUuidOrNull(row.patient_id);
  if (!safePatientUuid) return;
  const safeId = isUuidString(row.id) ? row.id.trim() : newRecordId(true, 'course');
  const working: Record<string, unknown> = stripOversizedBase64ForDb({
    id: safeId,
    patient_id: safePatientUuid,
    title: row.title,
    weekly_schedule: row.weekly_schedule ?? [],
    cognitive_modpack_id: row.cognitive_modpack_id || null,
    updated_at: row.updated_at || new Date().toISOString(),
  });
  for (let attempt = 0; attempt < 4; attempt += 1) {
    const res = (await db.from('patient_courses').upsert(working, { onConflict: 'id' })) as
      | { error?: { code?: string; message?: string } | null }
      | undefined;
    if (!res?.error) return;
    const msg = String(res.error.message ?? '');
    if (
      res.error.code === '23503' ||
      msg.toLowerCase().includes('foreign key') ||
      msg.toLowerCase().includes('modpack')
    ) {
      working.cognitive_modpack_id = null;
      continue;
    }
    const missing = msg.match(/Could not find the ['"]?([a-zA-Z0-9_]+)['"]? column/i)?.[1];
    if (missing && missing in working) {
      delete working[missing];
      continue;
    }
    return;
  }
}

async function settleDeleteWithTimeout(
  tasks: ReadonlyArray<() => PromiseLike<unknown>>,
  timeoutMs: number,
): Promise<void> {
  if (tasks.length === 0) return;
  let timerId: ReturnType<typeof setTimeout> | undefined;
  try {
    await Promise.race([
      Promise.allSettled(tasks.map((fn) => Promise.resolve().then(() => fn()).catch(() => undefined))),
      new Promise<void>((resolve) => {
        timerId = setTimeout(resolve, timeoutMs);
      }),
    ]);
  } catch {
    // Best-effort cleanup never throws
  } finally {
    if (timerId !== undefined) clearTimeout(timerId);
  }
}

function buildNormalizedHnAndUsernameVariants(rawValues: ReadonlyArray<string | null | undefined>): {
  hnVariants: string[];
  usernameVariants: string[];
} {
  const hnSet = new Set<string>();
  const userSet = new Set<string>();
  for (const raw of rawValues) {
    const clean = (raw ?? '').trim();
    if (!clean) continue;
    const withoutAt = clean.replace(/^@+/, '').trim();
    if (!withoutAt) continue;
    hnSet.add(withoutAt);
    userSet.add(clean);
    userSet.add(withoutAt);
    userSet.add(withoutAt.toLowerCase());
    userSet.add(`@${withoutAt.toLowerCase()}`);

    const hnMatch = withoutAt.match(/^hn[-_\s]*([a-zA-Z0-9]+)$/i);
    if (hnMatch?.[1]) {
      const digits = hnMatch[1];
      const upperDigits = digits.toUpperCase();
      const lowerDigits = digits.toLowerCase();
      hnSet.add(`HN-${upperDigits}`);
      hnSet.add(`hn-${lowerDigits}`);
      hnSet.add(`HN${upperDigits}`);
      hnSet.add(`hn${lowerDigits}`);
      userSet.add(`hn${lowerDigits}`);
      userSet.add(`@hn${lowerDigits}`);
      userSet.add(`HN-${upperDigits}`);
      userSet.add(`hn-${lowerDigits}`);
      userSet.add(`HN${upperDigits}`);
    }
  }
  return {
    hnVariants: Array.from(hnSet),
    usernameVariants: Array.from(userSet),
  };
}

/**
 * Cascading patient purge in Supabase capped at 1500ms total.
 * Validates UUIDs before querying uuid columns so non-UUID IDs (e.g., HN-75127) never raise 22P02,
 * deletes child tables + matching staff_users (by patient_id, id, normalized username/HN, and full_name)
 * first to prevent FK 23503 errors or deleted-patient resurrection.
 */
export async function purgePatientInSupabaseBestEffort(
  supabase: SupabaseClient | null,
  params: {
    rawId: string;
    patientId?: string | null;
    hn?: string | null;
    fullName?: string | null;
    linkedUserIds?: readonly string[];
    linkedUsernames?: readonly string[];
  },
): Promise<void> {
  if (!supabase || isSupabaseCircuitOpen(supabase)) return;
  try {
    const candidateUuids = Array.from(
      new Set([params.rawId, params.patientId].filter((v): v is string => isUuidString(v))),
    );
    const { hnVariants, usernameVariants } = buildNormalizedHnAndUsernameVariants([
      params.hn,
      !isUuidString(params.rawId) ? params.rawId : '',
      ...(params.linkedUsernames ?? []),
    ]);
    const userUuids = Array.from(new Set((params.linkedUserIds ?? []).filter(isUuidString)));
    const cleanFullName = (params.fullName ?? '').trim();

    const childTasks: Array<() => PromiseLike<unknown>> = [];
    for (const pid of candidateUuids) {
      childTasks.push(
        () => supabase.from('face_templates').delete().eq('patient_id', pid),
        () => supabase.from('patient_biometrics').delete().eq('patient_id', pid),
        () => supabase.from('face_enrollments').delete().eq('patient_id', pid),
        () => supabase.from('patient_courses').delete().eq('patient_id', pid),
        () => supabase.from('patient_clinical_assessments').delete().eq('patient_id', pid),
        () => supabase.from('rehab_schedules').delete().eq('patient_id', pid),
        () => supabase.from('rehab_sessions').delete().eq('patient_id', pid),
        () => supabase.from('rehab_session_history').delete().eq('patient_id', pid),
        () => supabase.from('medical_notes').delete().eq('patient_id', pid),
        () => supabase.from('treatment_reports').delete().eq('patient_id', pid),
        () => supabase.from('staff_calls').delete().eq('patient_id', pid),
        () => supabase.from('staff_users').delete().eq('patient_id', pid),
      );
    }
    for (const h of hnVariants) {
      childTasks.push(
        () => supabase.from('patient_biometrics').delete().eq('hn', h),
        () => supabase.from('patient_clinical_assessments').delete().eq('patient_hn', h),
      );
    }
    for (const uid of userUuids) {
      childTasks.push(
        () => supabase.from('active_sessions').delete().eq('user_id', uid),
        () => supabase.from('staff_users').delete().eq('id', uid),
      );
    }
    for (const uname of usernameVariants) {
      childTasks.push(() => supabase.from('staff_users').delete().eq('username', uname));
    }
    if (cleanFullName) {
      childTasks.push(() =>
        supabase.from('staff_users').delete().eq('role', 'patient').eq('full_name', cleanFullName),
      );
    }
    await settleDeleteWithTimeout(childTasks, 1000);

    const parentTasks: Array<() => PromiseLike<unknown>> = [
      ...candidateUuids.map((pid) => () => supabase.from('patients').delete().eq('id', pid)),
      ...hnVariants.map((h) => () => supabase.from('patients').delete().eq('hn', h)),
    ];
    await settleDeleteWithTimeout(parentTasks, 500);
  } catch {
    // Non-fatal: never turn DB / FK / 22P02 / timeout failures into HTTP 500
  }
}

/**
 * Staff / PT / Admin / Patient user purge in Supabase capped at 1500ms total.
 * Validates UUIDs before querying uuid columns (id, user_id, assigned_pt_id, delegated_to_pt_id),
 * falls back to normalized username matching, and cascades into public.patients for PATIENT accounts.
 */
export async function purgeUserInSupabaseBestEffort(
  supabase: SupabaseClient | null,
  params: {
    rawId: string;
    matchedUserId?: string | null;
    username?: string | null;
    role?: string | null;
    fullName?: string | null;
    linkedPatientId?: string | null;
    linkedHn?: string | null;
  },
): Promise<void> {
  if (!supabase || isSupabaseCircuitOpen(supabase)) return;
  try {
    const rawKeyClean = (params.rawId ?? '').trim();
    const userUuids = Array.from(
      new Set([params.rawId, params.matchedUserId].filter((v): v is string => isUuidString(v))),
    );
    const { hnVariants, usernameVariants } = buildNormalizedHnAndUsernameVariants([
      params.username,
      params.linkedHn,
      isUuidString(rawKeyClean) ? '' : rawKeyClean,
    ]);
    const cleanFullName = (params.fullName ?? '').trim();
    const isPatientRole = String(params.role ?? '').trim().toLowerCase() === 'patient';

    // Stage 1: Clear child foreign-key references BEFORE deleting parent staff_users row
    const childTasks: Array<() => PromiseLike<unknown>> = [];
    for (const uid of userUuids) {
      childTasks.push(
        () => supabase.from('active_sessions').delete().eq('user_id', uid),
        () => supabase.from('patients').update({ assigned_pt_id: null }).eq('assigned_pt_id', uid),
        () => supabase.from('patients').update({ delegated_to_pt_id: null }).eq('delegated_to_pt_id', uid),
        () => supabase.from('medical_notes').update({ author_id: null }).eq('author_id', uid),
        () => supabase.from('treatment_reports').update({ pt_id: null }).eq('pt_id', uid),
        () => supabase.from('face_enrollments').update({ enrolled_by: null }).eq('enrolled_by', uid),
      );
    }
    if (childTasks.length > 0) {
      await settleDeleteWithTimeout(childTasks, 750);
    }

    // Stage 2: Delete parent staff_users rows after FK children are cleared
    const parentTasks: Array<() => PromiseLike<unknown>> = [
      ...userUuids.map((uid) => () => supabase.from('staff_users').delete().eq('id', uid)),
      ...usernameVariants.map((uname) => () => supabase.from('staff_users').delete().eq('username', uname)),
    ];
    if (isPatientRole && cleanFullName) {
      parentTasks.push(() =>
        supabase.from('staff_users').delete().eq('role', 'patient').eq('full_name', cleanFullName),
      );
    }
    if (parentTasks.length > 0) {
      await settleDeleteWithTimeout(parentTasks, 650);
    }

    if (params.linkedPatientId || params.linkedHn || isPatientRole) {
      await purgePatientInSupabaseBestEffort(supabase, {
        rawId: params.linkedPatientId ?? params.linkedHn ?? hnVariants[0] ?? '',
        patientId: params.linkedPatientId,
        hn: params.linkedHn ?? hnVariants[0] ?? null,
        fullName: isPatientRole ? cleanFullName : null,
        linkedUserIds: userUuids,
        linkedUsernames: usernameVariants,
      });
    }
  } catch {
    // Non-fatal: delete succeeds cleanly even if DB times out or reports FK errors
  }
}
