import type { SupabaseClient } from '@supabase/supabase-js';
import {
  CLINICAL_EXERCISES,
  isDeprecatedExerciseRecord,
  mergeDuplicateSideExercises,
  sortExercisesStrokeUpperFirst,
} from '@drtechcares/pose-engine';
import {
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
  delegation_overrides?: Record<string, DelegationOverrideEntry>;
  patients: PatientRecord[];
  face_templates: EnrolledTemplateCandidate[];
  medical_notes: MedicalNoteRecord[];
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

/** Run PostgREST query with explicit columns (never SELECT *), stripping any unmigrated optional column on 42703/PGRST204. */
async function selectExplicitWithFallback(
  runQuery: (cols: string) => PromiseLike<unknown>,
  initialCols: string,
  maxRows = 50,
): Promise<unknown> {
  let cols = initialCols
    .split(',')
    .map((c) => c.trim())
    .filter(Boolean);
  for (let attempt = 0; attempt < 5; attempt += 1) {
    const colStr = cols.join(',');
    const res = (await withQueryLimit(runQuery(colStr), maxRows)) as
      | { data?: unknown; error?: { code?: string; message?: string } | null }
      | undefined;
    if (!res?.error) return res;
    const msg = String(res.error.message ?? '');
    const m =
      msg.match(/column\s+(?:[a-zA-Z0-9_]+\.)?["']?([a-zA-Z0-9_]+)["']?\s+does not exist/i) ||
      msg.match(/Could not find the ['"]?([a-zA-Z0-9_]+)['"]? column/i);
    if (m?.[1] && cols.length > 1) {
      const nextCols = cols.filter((c) => c !== m[1]);
      if (nextCols.length < cols.length) {
        cols = nextCols;
        continue;
      }
    }
    return res;
  }
  return { data: null, error: { message: 'column_fallback_exhausted' } };
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

/** Merge authoritative staff_users DB rows into state without resurrecting deleted seeded PT accounts. */
export function mergeStaffUsersFromDbRows(
  rows: Record<string, unknown>[],
  state: CloudRouterState,
): void {
  const deletedLower = new Set((state.deleted_user_ids ?? []).map((x) => x.toLowerCase()));
  const prevById = new Map(state.users.map((u) => [u.id, u]));
  const prevByUsername = new Map(state.users.map((u) => [u.username.toLowerCase(), u]));
  const dbUsers: StoredUserAccount[] = rows
    .map((r: Record<string, unknown>) => {
      const id = String(r.id);
      const username = String(r.username);
      const prev = prevById.get(id) ?? prevByUsername.get(username.toLowerCase());
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
        avatarUrl: typeof r.avatar_url === 'string' && r.avatar_url ? r.avatar_url : prev?.avatarUrl,
        bio: typeof r.bio === 'string' && r.bio ? r.bio : prev?.bio,
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

const STAFF_USERS_LIGHT_COLS =
  'id,username,password_hash,role,full_name,phone,specialty,license_no,license_number,patient_id,status,active,force_logged_out_at,created_at';

const PATIENTS_LIGHT_COLS =
  'id,hn,full_name,age,gender,phone,diagnosis,registration_status,assigned_pt_id,delegated_to_pt_id,delegation_note,next_appointment_date,face_biometric_status,face_frames_count,face_embedding,biometric_data,history,stroke_type,affected_side,motor_assessment_score,adl_dressing_score,max_safe_rom_deg,universal_health_link_code,created_at,updated_at';

const PATIENT_BIOMETRICS_LIGHT_COLS =
  'patient_id,hn,master_embedding,angle_embeddings,exemplar_embeddings,biometric_engine,model_version,frames_captured,frame_count,quality_score,updated_at';

/** Fast single-table lookup of staff_users for POST /api/auth/login (never blocks on 17-table hydration). */
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
      selectExplicitWithFallback(
        (cols) => supabase.from('staff_users').select(cols),
        STAFF_USERS_LIGHT_COLS,
        50,
      ).then((res) => ({
        timedOut: false as const,
        res: res as { data: Record<string, unknown>[] | null; error: unknown },
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
    mergeStaffUsersFromDbRows(outcome.res.data, state);
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

export async function hydrateStateFromSupabase(
  supabase: SupabaseClient,
  state: CloudRouterState,
): Promise<void> {
  if (isSupabaseCircuitOpen(supabase)) {
    return;
  }
  const totalDeadlineMs = Date.now() + 1500;

  // Stage 1 (Priority Clinical Directory): fetch patients & staff_users FIRST with explicit columns (zero SELECT *, no base64 images) & limit(50)
  const stage1Tasks: ReadonlyArray<() => PromiseLike<unknown>> = [
    () =>
      selectExplicitWithFallback(
        (cols) => supabase.from('staff_users').select(cols),
        STAFF_USERS_LIGHT_COLS,
        50,
      ),
    () =>
      selectExplicitWithFallback(
        (cols) => supabase.from('patients').select(cols),
        PATIENTS_LIGHT_COLS,
        50,
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
        50,
      ),
    () =>
      withQueryLimit(
        supabase
          .from('active_sessions')
          .select('token,user_id,role,ip_address,device_id,created_at,revoked')
          .eq('revoked', false)
          .order('created_at', { ascending: false }),
        50,
      ),
  ];

  // Stage 2 (Biometric Embeddings & Clinical Config): explicit lightweight columns + limit(50), excluding base64 avatar_url/questions blobs.
  const stage2Tasks: ReadonlyArray<() => PromiseLike<unknown>> = [
    () => withQueryLimit(supabase.from('face_templates').select('patient_id,pose,embedding,quality_score'), 50),
    () =>
      selectExplicitWithFallback(
        (cols) => supabase.from('patient_biometrics').select(cols),
        PATIENT_BIOMETRICS_LIGHT_COLS,
        50,
      ),
    () =>
      withQueryLimit(
        supabase.from('patient_courses').select('id,patient_id,title,weekly_schedule,cognitive_modpack_id,updated_at'),
        50,
      ),
    () =>
      withQueryLimit(
        supabase
          .from('exercises')
          .select('id,code,category,name_th,body_side,instructions_th,target_rom_deg,hold_sec,name_i18n,instruction_i18n,detection_config,active')
          .eq('active', true),
        50,
      ),
    () =>
      withQueryLimit(
        supabase.from('modpacks').select('id,name,category,card_count,categories,active').eq('active', true),
        50,
      ),
    () =>
      withQueryLimit(
        supabase.from('kiosks').select('kiosk_id,location,online,current_state,pairing_token,last_heartbeat_at'),
        50,
      ),
  ];

  // Stage 3 (Historical Logs & Telemetry): explicit lightweight columns + limit(50).
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
      withQueryLimit(
        supabase
          .from('audit_logs')
          .select('id,event_type,actor_role,actor_id,actor_name,ip_address,device_id,target_id,detail,action,target,details,created_at')
          .order('created_at', { ascending: false }),
        50,
      ),
    () =>
      withQueryLimit(
        supabase
          .from('rehab_session_history')
          .select('id,patient_id,hn,session_date,exercise_code,exercise_name,joint_name,body_side,baseline_rom_deg,achieved_max_rom_deg,rom_delta_deg,motor_score_at_session,valid_reps,compensations_count,facility_name,recorded_by_pt_id,recorded_by_pt_name,clinical_notes,created_at')
          .order('session_date', { ascending: false }),
        50,
      ),
  ];

  const [usersRes, patientsRes, settingsRes, bansRes, activeSessionsRes] =
    (await settleStageInBatches(stage1Tasks, 5, Math.min(totalDeadlineMs, Date.now() + 1200))) as [
      PromiseSettledResult<{ data: Record<string, unknown>[] | null; error: unknown }>,
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

  const [notesRes, sessionsRes, reportsRes, callsRes, auditRes, sessionHistoryRes] =
    (await settleStageInBatches(stage3Tasks, 3, totalDeadlineMs)) as [
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
    mergeStaffUsersFromDbRows(usersRes.value.data as Record<string, unknown>[], state);
  }

  const deletedPatientSet = new Set(state.deleted_patient_ids ?? []);
  const deletedPatientLower = new Set(
    (state.deleted_patient_ids ?? []).map((x) => x.trim().toLowerCase()).filter(Boolean),
  );
  if (patientsRes.status === 'fulfilled' && !patientsRes.value.error && Array.isArray(patientsRes.value.data)) {
    state.deleted_patient_ids ??= [];
    for (const r of patientsRes.value.data as Record<string, unknown>[]) {
      const regStatus = String(r.registration_status ?? 'active').trim().toLowerCase();
      if (
        Boolean(r.deleted_at) ||
        regStatus === 'deleted' ||
        regStatus === 'rejected' ||
        regStatus === 'inactive'
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
      const vec = Array.isArray(row.embedding)
        ? (row.embedding as number[])
        : typeof row.embedding === 'string' && row.embedding.startsWith('[')
          ? (JSON.parse(row.embedding) as number[])
          : null;
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
        const regStatus = String(r.registration_status ?? 'active').trim().toLowerCase();
        if (
          Boolean(r.deleted_at) ||
          regStatus === 'deleted' ||
          regStatus === 'rejected' ||
          regStatus === 'inactive'
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
      const hasTemplates = Boolean((tpl && tpl.raw.length > 0) || b120?.masterEmbedding || rowMasterVec);
      const historyStatus = history.faceBiometricStatus;
      const colStatus = r.face_biometric_status;
      const resolvedFaceStatus: 'PENDING' | 'VERIFIED' =
        colStatus === 'VERIFIED' || historyStatus === 'VERIFIED' || hasTemplates ? 'VERIFIED' : 'PENDING';

      const fallbackMaster =
        b120?.masterEmbedding ??
        rowMasterVec ??
        parseVec(history.masterEmbedding) ??
        tpl?.overall ??
        tpl?.frontal;

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
        (resolvedFaceStatus === 'VERIFIED' ? 120 : 0);

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

      return {
        id: pid,
        hn: String(r.hn),
        fullName: String(r.full_name),
        age: Number(r.age ?? 0),
        gender: typeof r.gender === 'string' ? r.gender : typeof history.gender === 'string' ? history.gender : 'ชาย',
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
        assignedPtId: String(r.assigned_pt_id ?? ''),
        delegatedToPtId: resolvedDelegatedToPtId,
        delegationNote: resolvedDelegationNote,
        delegationDate: resolvedDelegationDate,
        nextAppointmentDate: String(r.next_appointment_date ?? ''),
        registrationStatus: (r.registration_status as 'active' | 'archived') ?? 'active',
        enrolledAt: String(r.created_at ?? new Date().toISOString()),
        faceBiometricStatus:
          resolvedFaceStatus === 'VERIFIED' || prevPatient?.faceBiometricStatus === 'VERIFIED'
            ? 'VERIFIED'
            : resolvedFaceStatus,
        faceFramesCount: Math.max(resolvedFramesCount, prevPatient?.faceFramesCount ?? 0),
        faceImageUrl: mergedPrimaryFaceImageUrl,
        faceImageUrls: mergedFaceImageUrls,
        angleImageUrls: mergedAngleImageUrls,
        masterEmbedding: mergedMaster,
        angleEmbeddings: mergedAngles,
        ...(mergedExemplars ? { exemplarEmbeddings: mergedExemplars } : {}),
        biometricEngine:
          normalizeEngineTag(b120?.biometricEngine, mergedMaster) ??
          normalizeEngineTag(history.biometricEngine, mergedMaster) ??
          normalizeEngineTag(prevPatient?.biometricEngine, mergedMaster),
        geometricRatios:
          history.geometricRatios && typeof history.geometricRatios === 'object'
            ? (history.geometricRatios as GeometricAnthropometricRatios)
            : prevPatient?.geometricRatios,
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

    // Preserve active in-memory patients that haven't been flushed to Postgres yet (e.g. during HTTP 544/timeout)
    const dbSeenIds = new Set(state.patients.map((p) => p.id.trim().toLowerCase()));
    const dbSeenHns = new Set(state.patients.map((p) => p.hn.trim().toLowerCase()).filter(Boolean));
    for (const [prevId, prevP] of prevPatientsById.entries()) {
      const pidLo = prevId.trim().toLowerCase();
      const hnLo = (prevP.hn ?? '').trim().toLowerCase();
      if (
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
    state.rehab_sessions = sessionsRes.value.data.map((r: Record<string, unknown>) => ({
      id: String(r.id),
      patientId: String(r.patient_id),
      activityType: 'physio_exercise',
      exerciseId: String(r.exercise_id),
      exerciseName: String(r.exercise_id),
      completedSets: Number(r.completed_sets ?? 0),
      completedReps: Number(r.completed_reps ?? 0),
      accuracyPercent: Number(r.accuracy_percent ?? 100),
      maxRomDeg: Number(r.max_rom_deg ?? 0),
      painStopTriggered: Boolean(r.pain_stop_triggered),
      aiFeedback: 'Synced from Supabase',
      kinematicsReplay: (r.kinematics_replay as { tMs: number; angleDeg: number }[]) ?? [],
      minigameScore: r.minigame_score as { correct: number; total: number; category: string } | undefined,
      createdAt: String(r.created_at),
    }));
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
    state.modpacks = modpacksRes.value.data.map((r: Record<string, unknown>) => ({
      id: String(r.id),
      name: String(r.name),
      categories: Array.isArray(r.categories) ? (r.categories as string[]) : [String(r.category ?? 'cognitive')],
      questionCount: Number(r.card_count ?? (Array.isArray(r.questions) && r.questions.length > 0 ? r.questions.length : 10)),
      createdBy: 'system',
      active: Boolean(r.active ?? true),
    }));
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

/**
 * Cascading patient purge in Supabase capped at 1500ms total.
 * Validates UUIDs before querying uuid columns so non-UUID IDs (e.g., HN-75127) never raise 22P02,
 * and deletes child tables first to prevent FK 23503 errors from returning HTTP 500.
 */
export async function purgePatientInSupabaseBestEffort(
  supabase: SupabaseClient | null,
  params: {
    rawId: string;
    patientId?: string | null;
    hn?: string | null;
    linkedUserIds?: readonly string[];
    linkedUsernames?: readonly string[];
  },
): Promise<void> {
  if (!supabase || isSupabaseCircuitOpen(supabase)) return;
  try {
    const candidateUuids = Array.from(
      new Set([params.rawId, params.patientId].filter((v): v is string => isUuidString(v))),
    );
    const candidateHns = Array.from(
      new Set(
        [params.hn, !isUuidString(params.rawId) ? params.rawId : '']
          .map((v) => (v ?? '').trim())
          .filter(Boolean),
      ),
    );
    const userUuids = Array.from(new Set((params.linkedUserIds ?? []).filter(isUuidString)));
    const usernames = Array.from(
      new Set([...(params.linkedUsernames ?? []), ...candidateHns].map((v) => v.trim()).filter(Boolean)),
    );

    const childTasks: Array<() => PromiseLike<unknown>> = [];
    for (const pid of candidateUuids) {
      childTasks.push(
        () => supabase.from('face_templates').delete().eq('patient_id', pid),
        () => supabase.from('patient_biometrics').delete().eq('patient_id', pid),
        () => supabase.from('face_enrollments').delete().eq('patient_id', pid),
        () => supabase.from('patient_courses').delete().eq('patient_id', pid),
        () => supabase.from('rehab_schedules').delete().eq('patient_id', pid),
        () => supabase.from('rehab_sessions').delete().eq('patient_id', pid),
        () => supabase.from('rehab_session_history').delete().eq('patient_id', pid),
        () => supabase.from('medical_notes').delete().eq('patient_id', pid),
        () => supabase.from('treatment_reports').delete().eq('patient_id', pid),
        () => supabase.from('staff_calls').delete().eq('patient_id', pid),
        () => supabase.from('staff_users').delete().eq('patient_id', pid),
      );
    }
    for (const uid of userUuids) {
      childTasks.push(
        () => supabase.from('active_sessions').delete().eq('user_id', uid),
        () => supabase.from('staff_users').delete().eq('id', uid),
      );
    }
    for (const uname of usernames) {
      childTasks.push(() => supabase.from('staff_users').delete().eq('username', uname));
    }
    await settleDeleteWithTimeout(childTasks, 1000);

    const parentTasks: Array<() => PromiseLike<unknown>> = [
      ...candidateUuids.map((pid) => () => supabase.from('patients').delete().eq('id', pid)),
      ...candidateHns.map((h) => () => supabase.from('patients').delete().eq('hn', h)),
    ];
    await settleDeleteWithTimeout(parentTasks, 500);
  } catch {
    // Non-fatal: never turn DB / FK / 22P02 / timeout failures into HTTP 500
  }
}

/**
 * Staff / PT / Admin user purge in Supabase capped at 1500ms total.
 * Validates UUIDs before querying uuid columns (id, user_id, assigned_pt_id, delegated_to_pt_id),
 * falls back to username matching for seeded non-UUID IDs like `user-pt-1`, and swallows FK/timeout errors.
 */
export async function purgeUserInSupabaseBestEffort(
  supabase: SupabaseClient | null,
  params: {
    rawId: string;
    matchedUserId?: string | null;
    username?: string | null;
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
    const usernames = Array.from(
      new Set(
        [params.username, isUuidString(rawKeyClean) ? '' : rawKeyClean.replace(/^@/, '')]
          .map((v) => (v ?? '').trim())
          .filter(Boolean),
      ),
    );

    const tasks: Array<() => PromiseLike<unknown>> = [];
    for (const uid of userUuids) {
      tasks.push(
        () => supabase.from('active_sessions').delete().eq('user_id', uid),
        () => supabase.from('patients').update({ assigned_pt_id: null }).eq('assigned_pt_id', uid),
        () => supabase.from('patients').update({ delegated_to_pt_id: null }).eq('delegated_to_pt_id', uid),
        () => supabase.from('staff_users').delete().eq('id', uid),
      );
    }
    for (const uname of usernames) {
      tasks.push(() => supabase.from('staff_users').delete().eq('username', uname));
    }
    await settleDeleteWithTimeout(tasks, 1000);

    if (params.linkedPatientId || params.linkedHn) {
      await purgePatientInSupabaseBestEffort(supabase, {
        rawId: params.linkedPatientId ?? params.linkedHn ?? '',
        patientId: params.linkedPatientId,
        hn: params.linkedHn,
        linkedUserIds: userUuids,
        linkedUsernames: usernames,
      });
    }
  } catch {
    // Non-fatal: delete succeeds cleanly even if DB times out or reports FK errors
  }
}
