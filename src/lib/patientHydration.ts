import type { SupabaseClient } from '@supabase/supabase-js';
import {
  DrTechCaresApiClient,
  buildPatientDualWriteColumns,
  inspectPatientRecordHydration,
  normalizePatientRecord,
  resolveProtectedBiometricStatus,
  schedulePatientTopLevelRepairInBackground,
  type NormalizePatientOptions,
  type PatientHydrationInspection,
  type PatientRecord,
} from '@drtechcares/api-client';

export {
  buildPatientDualWriteColumns,
  inspectPatientRecordHydration,
  normalizePatientRecord,
  resolveProtectedBiometricStatus,
  schedulePatientTopLevelRepairInBackground,
  type NormalizePatientOptions,
  type PatientHydrationInspection,
};

export type PatientScopeFilter = 'all' | 'assigned';
export type PatientFaceStatusFilter = 'all' | 'enrolled' | 'pending';

export interface SupabasePatientSyncResult {
  patients: PatientRecord[];
  syncedFromSupabase: boolean;
  source: 'supabase' | 'worker_api' | 'merged' | 'error';
  errorMessage: string | null;
  authError: unknown;
  syncedAtIso: string;
}

/**
 * Hydrates a list of patient records from API, Supabase, or local store and automatically
 * schedules background top-level column repair whenever a raw DB row had stale `PENDING`
 * or empty biometric/profile columns while `history` JSONB contained valid verified data.
 */
export function normalizePatientRecordList(
  records: ReadonlyArray<Record<string, unknown> | PatientRecord>,
  options?: NormalizePatientOptions & { supabase?: SupabaseClient | null },
): PatientRecord[] {
  return records.map((raw) => {
    const inspected = inspectPatientRecordHydration(raw, options);
    if (options?.supabase && inspected.needsTopLevelRepair && inspected.topLevelRepairPatch && inspected.patient.id) {
      void schedulePatientTopLevelRepairInBackground(
        options.supabase,
        inspected.patient.id,
        inspected.topLevelRepairPatch,
      );
    }
    return inspected.patient;
  });
}

function formatQueryErrorMessage(err: unknown): string {
  if (!err) return 'ไม่สามารถเชื่อมต่อฐานข้อมูล Supabase ได้';
  if (typeof err === 'object') {
    const rec = err as { code?: unknown; message?: unknown; details?: unknown; hint?: unknown };
    const code = typeof rec.code === 'string' && rec.code.trim() ? rec.code.trim() : '';
    const message = typeof rec.message === 'string' && rec.message.trim() ? rec.message.trim() : '';
    const hint = typeof rec.hint === 'string' && rec.hint.trim() ? rec.hint.trim() : '';
    if (code === '42501' || message.toLowerCase().includes('row-level security') || message.toLowerCase().includes('permission denied')) {
      return `สิทธิ์การเข้าถึง Supabase (RLS ${code || '42501'}): ${message || 'permission denied for table patients'}`;
    }
    if (code && message) return `[${code}] ${message}${hint ? ` (${hint})` : ''}`;
    if (message) return message;
  }
  return err instanceof Error ? err.message : String(err);
}

function isRowSoftDeletedOrArchived(row: Record<string, unknown>, deletedSet: Set<string>): boolean {
  const pid = String(row.id ?? '').trim().toLowerCase();
  const hn = String(row.hn ?? '').trim().toLowerCase();
  const regStatus = String(row.registration_status ?? '').trim().toLowerCase();
  if (!pid) return true;
  if (
    Boolean(row.deleted_at) ||
    regStatus === 'deleted' ||
    regStatus === 'archived' ||
    regStatus === 'rejected' ||
    regStatus === 'inactive'
  ) {
    return true;
  }
  return deletedSet.has(pid) || (Boolean(hn) && deletedSet.has(hn));
}

/**
 * Unified Supabase + Worker API patient fetcher shared across Admin/Staff Portal tabs
 * (Patient Registry, Individual Clinical Record, Enrollment refresh).
 *
 * 1. Queries `supabase.from('patients').select('*').order('created_at', { ascending: false })`
 *    directly when a Supabase client is configured and hydrates every row via `normalizePatientRecord`
 *    (merging top-level columns + `history` JSONB and scheduling background top-level repair).
 * 2. Merges with `apiClient.listPatients({ scope: 'all', refresh })` without EVER letting an empty
 *    initial array or stale cache overwrite non-empty Supabase records.
 * 3. Surfaces Supabase query / RLS errors cleanly instead of silently returning 0 rows.
 */
export async function fetchAllPatientsUnified(
  apiClient: Pick<DrTechCaresApiClient, 'listPatients' | 'getSupabaseClient'>,
  options?: { forceRefresh?: boolean; skipApi?: boolean },
): Promise<SupabasePatientSyncResult> {
  const syncedAtIso = new Date().toISOString();
  const deletedSet = DrTechCaresApiClient.readDeletedTombstones();
  const supabase = typeof apiClient.getSupabaseClient === 'function' ? apiClient.getSupabaseClient() : null;

  const fetchDirectSupabase = async (): Promise<{
    patients: PatientRecord[] | null;
    errorMsg: string | null;
  }> => {
    if (!supabase) return { patients: null, errorMsg: null };
    try {
      const DIRECTORY_LIGHT_COLS =
        'id,hn,full_name,age,phone,diagnosis,registration_status,assigned_pt_id,delegated_to_pt_id,delegation_note,next_appointment_date,face_biometric_status,face_frames_count,face_image_url,biometric_updated_at,created_at,updated_at,gender,affected_side,motor_assessment_score,adl_dressing_score,max_safe_rom_deg,universal_health_link_code,historyFaceStatus:history->>faceBiometricStatus,historyFaceFramesCount:history->faceFramesCount,historyFaceImageUrl:history->>faceImageUrl,historyGender:history->>gender,rehabSchedules:history->rehabSchedules,careGroup:history->>careGroup,affectedSide:history->>affectedSide,motorAssessmentScore:history->motorAssessmentScore,adlDressingScore:history->adlDressingScore,maxSafeRomDeg:history->maxSafeRomDeg,universalHealthLinkCode:history->>universalHealthLinkCode,rehabSessionHistory:history->rehabSessionHistory,clinicalAssessments:history->clinicalAssessments,angleImageUrls:history->angleImageUrls,faceImageUrls:history->faceImageUrls,delegatedToPtId:history->>delegatedToPtId,delegationDate:history->>delegationDate,delegationNote:history->>delegationNote,biometricEngine:history->>biometricEngine';

      const runPatientsSelect = async (cols: string) => {
        const tableQuery = supabase.from('patients').select(cols) as unknown as {
          order?: (
            col: string,
            opts?: { ascending?: boolean },
          ) => PromiseLike<{ data?: Record<string, unknown>[] | null; error?: unknown }>;
        } & PromiseLike<{ data?: Record<string, unknown>[] | null; error?: unknown }>;
        return typeof tableQuery.order === 'function'
          ? await tableQuery.order('created_at', { ascending: false })
          : await tableQuery;
      };

      let rawRes = await runPatientsSelect(DIRECTORY_LIGHT_COLS);
      if (rawRes?.error) {
        const errCode = String((rawRes.error as { code?: unknown })?.code ?? '');
        const errMsg = String((rawRes.error as { message?: unknown })?.message ?? '').toLowerCase();
        if (
          errCode === '42703' ||
          errCode === 'PGRST204' ||
          errMsg.includes('does not exist') ||
          errMsg.includes('could not find')
        ) {
          rawRes = await runPatientsSelect(
            'id,hn,full_name,age,phone,diagnosis,registration_status,assigned_pt_id,next_appointment_date,face_biometric_status,face_frames_count,face_image_url,created_at,updated_at',
          );
        }
      }

      if (rawRes?.error) {
        return { patients: null, errorMsg: formatQueryErrorMessage(rawRes.error) };
      }
      if (!Array.isArray(rawRes?.data)) {
        return { patients: [], errorMsg: null };
      }

      const activeRows = rawRes.data.filter((row) => !isRowSoftDeletedOrArchived(row, deletedSet));
      const hydrated = activeRows.map((row) => {
        const inspected = inspectPatientRecordHydration(row, {
          requireRealVectorsForVerified: false,
          applyClinicalDefaults: true,
        });
        if (inspected.needsTopLevelRepair && inspected.topLevelRepairPatch && inspected.patient.id) {
          void schedulePatientTopLevelRepairInBackground(
            supabase,
            inspected.patient.id,
            inspected.topLevelRepairPatch,
          );
        }
        return inspected.patient;
      });
      return { patients: hydrated, errorMsg: null };
    } catch (err) {
      return { patients: null, errorMsg: formatQueryErrorMessage(err) };
    }
  };

  const fetchWorkerApi = async (): Promise<{
    patients: PatientRecord[] | null;
    error: unknown;
  }> => {
    if (options?.skipApi || typeof apiClient.listPatients !== 'function') {
      return { patients: null, error: null };
    }
    try {
      const rows = await apiClient.listPatients({
        scope: 'all',
        refresh: options?.forceRefresh,
      });
      if (Array.isArray(rows)) {
        const cleaned = rows
          .map((raw) => normalizePatientRecord(raw, { requireRealVectorsForVerified: false }))
          .filter((p) => {
            const pidLo = p.id.trim().toLowerCase();
            const hnLo = (p.hn ?? '').trim().toLowerCase();
            return (
              (p.registrationStatus ?? 'active') === 'active' &&
              !deletedSet.has(pidLo) &&
              (!hnLo || !deletedSet.has(hnLo))
            );
          });
        return { patients: cleaned, error: null };
      }
      return { patients: [], error: null };
    } catch (err) {
      return { patients: null, error: err };
    }
  };

  const [directRes, apiRes] = await Promise.all([fetchDirectSupabase(), fetchWorkerApi()]);
  const supabasePatients = directRes.patients;
  const supabaseErrorMsg = directRes.errorMsg;
  const apiPatients = apiRes.patients;
  const apiError = apiRes.error;

  // Never overwrite fetched Supabase records with an empty API array or stale cache
  const mergedById = new Map<string, PatientRecord>();
  const orderedIds: string[] = [];

  if (supabasePatients && supabasePatients.length > 0) {
    for (const p of supabasePatients) {
      mergedById.set(p.id, p);
      orderedIds.push(p.id);
    }
    if (apiPatients && apiPatients.length > 0) {
      for (const ap of apiPatients) {
        const existing = mergedById.get(ap.id);
        if (existing) {
          mergedById.set(ap.id, {
            ...ap,
            ...existing,
            rehabSchedules:
              (existing.rehabSchedules && existing.rehabSchedules.length > 0
                ? existing.rehabSchedules
                : ap.rehabSchedules) ?? [],
            delegatedToPtId: existing.delegatedToPtId ?? ap.delegatedToPtId,
            delegationDate: existing.delegationDate ?? ap.delegationDate,
            delegationNote: existing.delegationNote ?? ap.delegationNote,
          });
        } else {
          mergedById.set(ap.id, ap);
          orderedIds.push(ap.id);
        }
      }
    }
    return {
      patients: orderedIds.map((id) => mergedById.get(id)!),
      syncedFromSupabase: true,
      source: apiPatients && apiPatients.length > 0 ? 'merged' : 'supabase',
      errorMessage: null,
      authError: null,
      syncedAtIso,
    };
  }

  if (apiPatients && apiPatients.length > 0) {
    return {
      patients: apiPatients,
      syncedFromSupabase: true,
      source: 'worker_api',
      errorMessage: supabaseErrorMsg,
      authError: null,
      syncedAtIso,
    };
  }

  if (supabasePatients !== null && !supabaseErrorMsg) {
    return {
      patients: [],
      syncedFromSupabase: true,
      source: 'supabase',
      errorMessage: null,
      authError: null,
      syncedAtIso,
    };
  }

  if (apiPatients !== null && !supabaseErrorMsg) {
    return {
      patients: [],
      syncedFromSupabase: true,
      source: 'worker_api',
      errorMessage: null,
      authError: null,
      syncedAtIso,
    };
  }

  return {
    patients: [],
    syncedFromSupabase: false,
    source: 'error',
    errorMessage:
      supabaseErrorMsg ??
      (apiError instanceof Error ? apiError.message : 'ไม่สามารถโหลดรายชื่อผู้ป่วยจาก Supabase ได้'),
    authError: apiError,
    syncedAtIso,
  };
}

export function isHydratedPatientFaceVerified(patient: PatientRecord): boolean {
  if (patient.biometricStatusDetail) {
    return patient.biometricStatusDetail === 'VERIFIED';
  }
  const hasVectors = Boolean(
    (Array.isArray(patient.masterEmbedding) && patient.masterEmbedding.length >= 128) ||
      (patient.angleEmbeddings?.center && patient.angleEmbeddings.center.length >= 128),
  );
  return patient.faceBiometricStatus === 'VERIFIED' || hasVectors;
}

/**
 * Canonical Admin/Staff Portal patient filter.
 * - Default `scopeFilter = 'all'` shows ALL registered clinic patients regardless of `assignedPtId`.
 * - `scopeFilter = 'assigned'` filters strictly to patients assigned or delegated to `currentUserId`.
 * - Default `faceFilter = 'all'` never hides `PENDING` or `active` patients.
 */
export function filterAdminPortalPatients(
  patients: readonly PatientRecord[],
  options: {
    scopeFilter?: PatientScopeFilter;
    faceFilter?: PatientFaceStatusFilter;
    searchQuery?: string;
    currentUserId?: string;
  } = {},
): PatientRecord[] {
  const scopeFilter = options.scopeFilter ?? 'all';
  const faceFilter = options.faceFilter ?? 'all';
  const q = (options.searchQuery ?? '').trim().toLowerCase();
  const uid = (options.currentUserId ?? '').trim();

  return patients.filter((p) => {
    if (scopeFilter === 'assigned' && uid) {
      if (p.assignedPtId !== uid && p.delegatedToPtId !== uid) {
        return false;
      }
    }
    const verified = isHydratedPatientFaceVerified(p);
    if (faceFilter === 'enrolled' && !verified) return false;
    if (faceFilter === 'pending' && verified) return false;
    if (!q) return true;
    return (
      (p.fullName || '').toLowerCase().includes(q) ||
      (p.hn || '').toLowerCase().includes(q) ||
      (p.phone || '').toLowerCase().includes(q) ||
      (p.diagnosis || '').toLowerCase().includes(q)
    );
  });
}
