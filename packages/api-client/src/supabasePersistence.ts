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

/** Safely apply `.limit(maxRows)` on real PostgREST builders without breaking minimal unit mocks. */
function withQueryLimit(query: unknown, maxRows: number): PromiseLike<unknown> {
  const q = query as { limit?: (n: number) => PromiseLike<unknown> } | null;
  if (q && typeof q.limit === 'function') {
    return q.limit(maxRows);
  }
  return Promise.resolve(query);
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

/** Fast single-table lookup of staff_users for POST /api/auth/login (never blocks on 17-table hydration). */
export async function hydrateStaffUsersOnly(
  supabase: SupabaseClient,
  state: CloudRouterState,
  timeoutMs = 2500,
): Promise<boolean> {
  let timerId: ReturnType<typeof setTimeout> | undefined;
  try {
    const outcome = await Promise.race([
      Promise.resolve(withQueryLimit(supabase.from('staff_users').select('*'), 200)).then((res) => ({
        timedOut: false as const,
        res: res as { data: Record<string, unknown>[] | null; error: unknown },
      })),
      new Promise<{ timedOut: true }>((resolve) => {
        timerId = setTimeout(() => resolve({ timedOut: true }), timeoutMs);
      }),
    ]);
    if (outcome.timedOut || outcome.res.error || !Array.isArray(outcome.res.data)) {
      return false;
    }
    mergeStaffUsersFromDbRows(outcome.res.data, state);
    return true;
  } catch {
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
  const totalDeadlineMs = Date.now() + 2450;

  // Stage 1 (Priority Clinical Directory): fetch patients & staff_users FIRST so PATIENTS COUNT > 0
  // never contends with heavy Base64 biometrics, 512/1024-D templates, or historical logs on Free Tier.
  const stage1Tasks: ReadonlyArray<() => PromiseLike<unknown>> = [
    () => withQueryLimit(supabase.from('staff_users').select('*'), 200),
    () => withQueryLimit(supabase.from('patients').select('*'), 300),
    () => supabase.from('system_settings').select('*').eq('id', 1).maybeSingle(),
    () => withQueryLimit(supabase.from('banned_targets').select('*').order('created_at', { ascending: false }), 200),
    () => withQueryLimit(supabase.from('active_sessions').select('*').eq('revoked', false).order('created_at', { ascending: false }), 200),
  ];

  // Stage 2 (Biometric Embeddings & Clinical Config): queried sequentially after Stage 1 completes,
  // using small batchSize=2 so patient_biometrics + face_templates never spike Free Tier connections.
  const stage2Tasks: ReadonlyArray<() => PromiseLike<unknown>> = [
    () => withQueryLimit(supabase.from('face_templates').select('patient_id, pose, embedding, quality_score'), 300),
    () => withQueryLimit(supabase.from('patient_biometrics').select('*'), 150),
    () => withQueryLimit(supabase.from('patient_courses').select('*'), 300),
    () => supabase.from('exercises').select('*').eq('active', true),
    () => supabase.from('modpacks').select('*').eq('active', true),
    () => withQueryLimit(supabase.from('kiosks').select('*'), 50),
  ];

  // Stage 3 (Historical Logs & Telemetry): fetched last within any remaining time budget.
  const stage3Tasks: ReadonlyArray<() => PromiseLike<unknown>> = [
    () => withQueryLimit(supabase.from('medical_notes').select('*').order('created_at', { ascending: false }), 150),
    () => withQueryLimit(supabase.from('rehab_sessions').select('*').order('created_at', { ascending: false }), 150),
    () => withQueryLimit(supabase.from('treatment_reports').select('*').order('created_at', { ascending: false }), 150),
    () => withQueryLimit(supabase.from('staff_calls').select('*').order('created_at', { ascending: false }), 150),
    () => withQueryLimit(supabase.from('audit_logs').select('*').order('created_at', { ascending: false }), 150),
    () => withQueryLimit(supabase.from('rehab_session_history').select('*').order('session_date', { ascending: false }), 150),
  ];

  const [usersRes, patientsRes, settingsRes, bansRes, activeSessionsRes] =
    (await settleStageInBatches(stage1Tasks, 5, Math.min(totalDeadlineMs, Date.now() + 1400))) as [
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

  if (usersRes.status === 'fulfilled' && !usersRes.value.error && Array.isArray(usersRes.value.data)) {
    mergeStaffUsersFromDbRows(usersRes.value.data as Record<string, unknown>[], state);
  }

  const deletedPatientSet = new Set(state.deleted_patient_ids ?? []);

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
      if (deletedPatientSet.has(pid)) continue;
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
      if (deletedPatientSet.has(pid)) continue;
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
    const deletedPatientLower = new Set((state.deleted_patient_ids ?? []).map((x) => x.toLowerCase()));
    const prevPatientsById = new Map(state.patients.map((p) => [p.id, p]));
    state.patients = patientsRes.value.data
      .filter((r: Record<string, unknown>) => {
        const pidLo = String(r.id ?? '').toLowerCase();
        const hnLo = String(r.hn ?? '').toLowerCase();
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

      if (
        resolvedFaceStatus === 'VERIFIED' &&
        (typeof r.face_frames_count !== 'number' || r.face_frames_count <= 0) &&
        resolvedFramesCount > 0
      ) {
        try {
          const tableRef = supabase.from('patients') as unknown as {
            update?: (payload: Record<string, unknown>) => {
              eq?: (col: string, val: string) => Promise<unknown>;
            };
          };
          const q = tableRef.update?.({
            face_frames_count: resolvedFramesCount,
            history: {
              ...history,
              faceBiometricStatus: 'VERIFIED',
              faceFramesCount: resolvedFramesCount,
            },
            updated_at: new Date().toISOString(),
          });
          void q?.eq?.('id', pid)?.then(
            () => {},
            () => {},
          );
        } catch {
          // ignore best-effort self-heal errors
        }
      }

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
        faceBiometricStatus: resolvedFaceStatus,
        faceFramesCount: resolvedFramesCount,
        faceImageUrl: primaryFaceImageUrl,
        faceImageUrls: hydratedFaceImageUrls,
        angleImageUrls: hydratedAngleImageUrls,
        masterEmbedding: fallbackMaster,
        angleEmbeddings: fallbackAngles,
        ...(fallbackExemplars ? { exemplarEmbeddings: fallbackExemplars } : {}),
        biometricEngine:
          normalizeEngineTag(b120?.biometricEngine, fallbackMaster) ??
          normalizeEngineTag(history.biometricEngine, fallbackMaster) ??
          normalizeEngineTag(prevPatient?.biometricEngine, fallbackMaster),
        geometricRatios:
          history.geometricRatios && typeof history.geometricRatios === 'object'
            ? (history.geometricRatios as GeometricAnthropometricRatios)
            : prevPatient?.geometricRatios,
        rehabSchedules: Array.isArray(history.rehabSchedules)
          ? sanitizeAccidentalDefaultKneePrefix(history.rehabSchedules as RehabScheduleItem[])
          : [],
        biometricHistory:
          tpl && tpl.frontal && tpl.overall
            ? {
                frontal_centroid: tpl.frontal,
                overall_centroid: tpl.overall,
                raw_frames: tpl.raw,
              }
            : fallbackMaster
              ? {
                  frontal_centroid: fallbackAngles?.center ?? fallbackMaster,
                  overall_centroid: fallbackMaster,
                  raw_frames: [],
                }
              : undefined,
      };
    });
  } else {
    state.patients = state.patients.filter((p) => !deletedPatientSet.has(p.id));
  }

  const activePatientMap = new Map<string, string>();
  for (const p of state.patients) {
    if (p.registrationStatus === 'active' && !deletedPatientSet.has(p.id)) {
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
): Promise<void> {
  if (!supabase) return;
  try {
    await task(supabase);
  } catch {
    // Non-fatal fallback for offline / missing column edge cases
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
  if (!supabase) return;
  const result = (await task(supabase)) as { error?: { message?: string } | null } | undefined;
  if (result && typeof result === 'object' && 'error' in result && result.error) {
    throw new Error(result.error.message || 'Supabase biometric write failed');
  }
}

const UUID_REGEX = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

export function isUuidString(value: string | null | undefined): boolean {
  return typeof value === 'string' && UUID_REGEX.test(value.trim());
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
  if (!supabase) return;
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
  if (!supabase) return;
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
