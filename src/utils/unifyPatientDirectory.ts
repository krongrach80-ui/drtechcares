import type { SupabaseClient } from '@supabase/supabase-js';
import {
  DrTechCaresApiClient,
  normalizePatientRecord,
  type PatientRecord,
  type UserAccount,
} from '@drtechcares/api-client';
import {
  isSupabaseCircuitOpen,
  isUuidString,
} from '../../packages/api-client/src/supabasePersistence';

export interface StaffPatientMirrorCandidate {
  id: string;
  username?: string;
  fullName?: string;
  full_name?: string;
  phone?: string;
  role?: string;
  patientId?: string | null;
  patient_id?: string | null;
  active?: boolean;
  createdAt?: string;
  created_at?: string;
}

export interface UnifyClinicPatientsResult {
  patients: PatientRecord[];
  /** Always empty on read/render cycles so deleted patients are never resurrected from staff_users. */
  synthesizedPatients: PatientRecord[];
}

export interface CascadeDeletePatientTarget {
  id?: string | null;
  patientId?: string | null;
  hn?: string | null;
  username?: string | null;
  fullName?: string | null;
  role?: string | null;
}

/**
 * Strips leading `@`, whitespace, and hyphens and lowercases so `@hn410382`,
 * `hn410382`, `HN410382`, and `HN-410382` all map to the identical canonical key `hn410382`.
 */
export function normalizeHnKey(raw: unknown): string {
  if (typeof raw !== 'string') return '';
  return raw.trim().replace(/^@+/, '').replace(/[\s_-]+/g, '').toLowerCase();
}

/**
 * Formats a raw HN or patient username (e.g. `@hn53309` or `hn410382`) into canonical `HN-53309` / `HN-410382`.
 */
export function formatCanonicalHn(raw: unknown, fallbackId?: string): string {
  const cleaned = typeof raw === 'string' ? raw.trim().replace(/^@+/, '') : '';
  if (!cleaned) {
    const suffix = (fallbackId ?? '00000').replace(/[^a-zA-Z0-9]/g, '').slice(0, 6).toUpperCase();
    return `HN-${suffix || '00000'}`;
  }
  const hnDigitsMatch = cleaned.match(/^hn[-_\s]*([a-zA-Z0-9]+)$/i);
  if (hnDigitsMatch?.[1]) {
    return `HN-${hnDigitsMatch[1].toUpperCase()}`;
  }
  return cleaned.toUpperCase();
}

export function buildNormalizedHnVariants(rawValues: ReadonlyArray<string | null | undefined>): {
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
      const upper = digits.toUpperCase();
      const lower = digits.toLowerCase();
      hnSet.add(`HN-${upper}`);
      hnSet.add(`hn-${lower}`);
      hnSet.add(`HN${upper}`);
      hnSet.add(`hn${lower}`);
      userSet.add(`hn${lower}`);
      userSet.add(`@hn${lower}`);
      userSet.add(`HN-${upper}`);
      userSet.add(`hn-${lower}`);
      userSet.add(`HN${upper}`);
    }
  }
  return {
    hnVariants: Array.from(hnSet),
    usernameVariants: Array.from(userSet),
  };
}

function normalizeFullNameKey(raw: unknown): string {
  if (typeof raw !== 'string') return '';
  return raw.trim().toLowerCase().replace(/\s+/g, ' ');
}

function isPatientRoleAccount(user: StaffPatientMirrorCandidate): boolean {
  const roleUpper = String(user.role ?? '').trim().toUpperCase();
  return roleUpper === 'PATIENT' && user.active !== false;
}

/**
 * Single-source-of-truth Patient Directory hydrator (`public.patients` ONLY).
 * - Hydrates top-level columns and `history` JSON fields cleanly and deduplicates by `id` / normalized `hn`.
 * - May enrich missing `phone` on an already-existing patient row from a matching `staff_users` entry.
 * - NEVER synthesizes, inserts, or resurrects deleted patients from `staff_users` during read/render cycles.
 */
export function unifyClinicPatientsDetailed(
  supabasePatients: ReadonlyArray<Record<string, unknown> | PatientRecord>,
  staffUserAccounts: ReadonlyArray<UserAccount | StaffPatientMirrorCandidate> = [],
): UnifyClinicPatientsResult {
  const unified: PatientRecord[] = [];
  const byId = new Map<string, number>();
  const byHnKey = new Map<string, number>();
  const byNameKey = new Map<string, number>();

  for (const raw of supabasePatients) {
    if (!raw || typeof raw !== 'object') continue;
    const normalized = normalizePatientRecord(raw, {
      requireRealVectorsForVerified: false,
      applyClinicalDefaults: true,
    });
    if (!normalized.id && !normalized.hn && !normalized.fullName) continue;
    if ((normalized.registrationStatus ?? 'active') !== 'active') continue;

    const canonicalPatient: PatientRecord = {
      ...normalized,
      hn: formatCanonicalHn(normalized.hn, normalized.id),
    };

    const idLo = canonicalPatient.id.trim().toLowerCase();
    const hnKey = normalizeHnKey(canonicalPatient.hn);
    const existingIdx =
      (idLo && byId.has(idLo) ? byId.get(idLo) : undefined) ??
      (hnKey && byHnKey.has(hnKey) ? byHnKey.get(hnKey) : undefined);

    if (existingIdx !== undefined) {
      const prev = unified[existingIdx];
      unified[existingIdx] = {
        ...prev,
        ...canonicalPatient,
        faceBiometricStatus:
          prev.faceBiometricStatus === 'VERIFIED' || canonicalPatient.faceBiometricStatus === 'VERIFIED'
            ? 'VERIFIED'
            : canonicalPatient.faceBiometricStatus,
        faceFramesCount: Math.max(prev.faceFramesCount ?? 0, canonicalPatient.faceFramesCount ?? 0),
        masterEmbedding: canonicalPatient.masterEmbedding ?? prev.masterEmbedding,
        angleEmbeddings: canonicalPatient.angleEmbeddings ?? prev.angleEmbeddings,
        faceImageUrl: canonicalPatient.faceImageUrl ?? prev.faceImageUrl,
        phone: canonicalPatient.phone || prev.phone,
      };
      continue;
    }

    const idx = unified.length;
    unified.push(canonicalPatient);
    if (idLo) byId.set(idLo, idx);
    if (hnKey) byHnKey.set(hnKey, idx);
    const nameKey = normalizeFullNameKey(canonicalPatient.fullName);
    if (nameKey) byNameKey.set(nameKey, idx);
  }

  // Enrich phone ONLY for patients already present in public.patients (never resurrect missing rows)
  for (const rawUser of staffUserAccounts) {
    const user = rawUser as StaffPatientMirrorCandidate;
    if (!user || !isPatientRoleAccount(user)) continue;
    const phone = String(user.phone ?? '').trim();
    if (!phone) continue;

    const linkedPatientId = String(user.patientId ?? user.patient_id ?? '').trim().toLowerCase();
    const userId = String(user.id ?? '').trim().toLowerCase();
    const hnKey = normalizeHnKey(user.username);
    const nameKey = normalizeFullNameKey(user.fullName ?? user.full_name);

    const existingIdx =
      (linkedPatientId && byId.has(linkedPatientId) ? byId.get(linkedPatientId) : undefined) ??
      (userId && byId.has(userId) ? byId.get(userId) : undefined) ??
      (hnKey && byHnKey.has(hnKey) ? byHnKey.get(hnKey) : undefined) ??
      (nameKey && byNameKey.has(nameKey) ? byNameKey.get(nameKey) : undefined);

    if (existingIdx !== undefined && !unified[existingIdx].phone) {
      unified[existingIdx] = { ...unified[existingIdx], phone };
    }
  }

  return { patients: unified, synthesizedPatients: [] };
}

/**
 * Deterministic helper returning the hydrated, deduplicated `PatientRecord[]` directory
 * backed strictly by `public.patients` (never resurrecting deleted accounts from `staff_users`).
 */
export function unifyClinicPatients(
  supabasePatients: ReadonlyArray<Record<string, unknown> | PatientRecord>,
  staffUserAccounts: ReadonlyArray<UserAccount | StaffPatientMirrorCandidate> = [],
): PatientRecord[] {
  return unifyClinicPatientsDetailed(supabasePatients, staffUserAccounts).patients;
}

/**
 * Intentionally a no-op during read/render cycles: new patient rows in `public.patients` and
 * `staff_users` must be created exclusively through explicit patient registration flows.
 */
export function mirrorSynthesizedPatientsToSupabase(
  _supabase: SupabaseClient | null | undefined,
  _synthesizedPatients: ReadonlyArray<PatientRecord>,
): Promise<number> {
  return Promise.resolve(0);
}

/**
 * Removes a deleted patient (or PATIENT staff_user) consistently from both React `patients`
 * and `users` lists using `id`, `patientId`, normalized `hn`/`username`, and `fullName`.
 */
export function applyCascadingPatientDeletionToState(
  patients: ReadonlyArray<PatientRecord>,
  users: ReadonlyArray<UserAccount>,
  target: CascadeDeletePatientTarget,
): { patients: PatientRecord[]; users: UserAccount[]; matchedUserIds: string[]; matchedPatientIds: string[] } {
  const idKeys = new Set(
    [target.id, target.patientId]
      .map((v) => (v ?? '').trim().toLowerCase())
      .filter(Boolean),
  );
  const hnKeys = new Set(
    [target.hn, target.username]
      .map((v) => normalizeHnKey(v))
      .filter(Boolean),
  );
  const nameKey = normalizeFullNameKey(target.fullName);
  const targetIsStaffNonPatient =
    Boolean(target.role) && String(target.role).trim().toLowerCase() !== 'patient';

  const matchedPatientIds: string[] = [];
  const nextPatients: PatientRecord[] = [];
  for (const p of patients) {
    const pidLo = p.id.trim().toLowerCase();
    const pHnKey = normalizeHnKey(p.hn);
    const pNameKey = normalizeFullNameKey(p.fullName);
    const isTargetPatient =
      !targetIsStaffNonPatient &&
      (idKeys.has(pidLo) ||
        (Boolean(pHnKey) && hnKeys.has(pHnKey)) ||
        (Boolean(nameKey) && pNameKey === nameKey));

    if (isTargetPatient) {
      matchedPatientIds.push(p.id);
      idKeys.add(pidLo);
      if (pHnKey) hnKeys.add(pHnKey);
      continue;
    }

    // If deleting a PT/staff member, keep the patient and unassign PT references
    if (
      (p.assignedPtId && idKeys.has(p.assignedPtId.trim().toLowerCase())) ||
      (p.delegatedToPtId && idKeys.has(p.delegatedToPtId.trim().toLowerCase()))
    ) {
      nextPatients.push({
        ...p,
        assignedPtId: p.assignedPtId && idKeys.has(p.assignedPtId.trim().toLowerCase()) ? '' : p.assignedPtId,
        delegatedToPtId:
          p.delegatedToPtId && idKeys.has(p.delegatedToPtId.trim().toLowerCase()) ? null : p.delegatedToPtId,
      });
    } else {
      nextPatients.push(p);
    }
  }

  const matchedUserIds: string[] = [];
  const nextUsers: UserAccount[] = [];
  for (const u of users) {
    const uidLo = u.id.trim().toLowerCase();
    const uPidLo = (u.patientId ?? '').trim().toLowerCase();
    const uHnKey = normalizeHnKey(u.username);
    const uNameKey = normalizeFullNameKey(u.fullName);
    const isPatientRole = String(u.role ?? '').trim().toLowerCase() === 'patient';

    const matchesUser =
      idKeys.has(uidLo) ||
      (Boolean(uPidLo) && idKeys.has(uPidLo)) ||
      (Boolean(uHnKey) && hnKeys.has(uHnKey)) ||
      (isPatientRole && Boolean(nameKey) && uNameKey === nameKey);

    if (matchesUser) {
      matchedUserIds.push(u.id);
      continue;
    }
    nextUsers.push(u);
  }

  return { patients: nextPatients, users: nextUsers, matchedUserIds, matchedPatientIds };
}

/**
 * Purges deleted patient & linked PATIENT account records from all browser `localStorage` keys
 * (`drtech_patients`, `drtech_users`, `drtech_biometric_vault_v3`, `drtech_face_templates`,
 * `drtech_courses`, `drtech_case_delegations`, `drtech_clinical_assessments_v1`,
 * `drtechcares_recent_enroll_sync_v1`) and records tombstones in `drtech_deleted_ids`.
 */
export function cleanupDeletedPatientLocalStorage(target: CascadeDeletePatientTarget): void {
  if (typeof globalThis.localStorage === 'undefined') return;
  try {
    const idKeys = new Set(
      [target.id, target.patientId]
        .map((v) => (v ?? '').trim().toLowerCase())
        .filter(Boolean),
    );
    const { hnVariants, usernameVariants } = buildNormalizedHnVariants([target.hn, target.username]);
    const hnKeySet = new Set(
      [target.hn, target.username, ...hnVariants, ...usernameVariants]
        .map((v) => normalizeHnKey(v))
        .filter(Boolean),
    );
    const exactKeySet = new Set(
      [...idKeys, ...hnVariants.map((v) => v.toLowerCase()), ...usernameVariants.map((v) => v.toLowerCase())],
    );
    const nameKey = normalizeFullNameKey(target.fullName);

    // Record tombstones in drtech_deleted_ids
    const existingTombstones = DrTechCaresApiClient.readDeletedTombstones();
    for (const k of [...exactKeySet, ...hnKeySet]) {
      if (k) existingTombstones.add(k);
    }
    globalThis.localStorage.setItem(
      'drtech_deleted_ids',
      JSON.stringify(Array.from(existingTombstones).slice(-500)),
    );

    const isMatchingEntry = (entry: unknown): boolean => {
      if (!entry || typeof entry !== 'object') return false;
      const rec = entry as Record<string, unknown>;
      const rawIds = [rec.id, rec.patientId, rec.patient_id, rec.userId, rec.user_id]
        .map((v) => (typeof v === 'string' ? v.trim().toLowerCase() : ''))
        .filter(Boolean);
      if (rawIds.some((id) => exactKeySet.has(id))) return true;

      const rawHns = [rec.hn, rec.patient_hn, rec.username]
        .map((v) => normalizeHnKey(v))
        .filter(Boolean);
      if (rawHns.some((hk) => hnKeySet.has(hk))) return true;

      const entryRole = typeof rec.role === 'string' ? rec.role.trim().toLowerCase() : '';
      const entryName = normalizeFullNameKey(rec.fullName ?? rec.full_name);
      if (nameKey && entryName === nameKey && (!entryRole || entryRole === 'patient')) {
        return true;
      }
      return false;
    };

    const arrayStorageKeys = [
      'drtech_patients',
      'drtech_users',
      'drtech_biometric_vault_v3',
      'drtech_face_templates',
      'drtech_courses',
      'drtech_clinical_assessments_v1',
      'drtechcares_recent_enroll_sync_v1',
    ];
    for (const key of arrayStorageKeys) {
      const raw = globalThis.localStorage.getItem(key);
      if (!raw) continue;
      try {
        const parsed = JSON.parse(raw) as unknown;
        if (Array.isArray(parsed)) {
          const filtered = parsed.filter((item) => !isMatchingEntry(item));
          globalThis.localStorage.setItem(key, JSON.stringify(filtered));
        }
      } catch {
        // Ignore malformed localStorage cache entry
      }
    }

    // Clean map-keyed delegation overrides (`drtech_case_delegations`)
    const delRaw = globalThis.localStorage.getItem('drtech_case_delegations');
    if (delRaw) {
      try {
        const parsed = JSON.parse(delRaw) as Record<string, unknown>;
        if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
          let changed = false;
          for (const k of Object.keys(parsed)) {
            if (exactKeySet.has(k.toLowerCase()) || hnKeySet.has(normalizeHnKey(k))) {
              delete parsed[k];
              changed = true;
            }
          }
          if (changed) {
            globalThis.localStorage.setItem('drtech_case_delegations', JSON.stringify(parsed));
          }
        }
      } catch {
        // Ignore malformed map
      }
    }
  } catch {
    // Ignore storage access errors
  }
}

/**
 * Executes a direct best-effort cascading delete in Supabase across `public.patients`,
 * `public.staff_users`, `patient_clinical_assessments`, `rehab_schedules`, and child tables
 * matching by `id`, `patient_id`, normalized `hn` / `username` (`HN-XXXXX` / `hnXXXXX` / `@hnXXXXX`),
 * and `full_name` (restricted to `role = 'patient'`).
 */
export async function cascadeDeletePatientInSupabase(
  supabase: SupabaseClient | null | undefined,
  target: CascadeDeletePatientTarget,
): Promise<void> {
  if (!supabase || isSupabaseCircuitOpen(supabase)) return;
  try {
    const candidateUuids = Array.from(
      new Set([target.id, target.patientId].filter((v): v is string => isUuidString(v))),
    );
    const { hnVariants, usernameVariants } = buildNormalizedHnVariants([target.hn, target.username]);
    const cleanFullName = (target.fullName ?? '').trim();

    const safeDeleteEq = async (table: string, col: string, val: string): Promise<void> => {
      try {
        const builder = supabase.from(table) as unknown as {
          delete?: () => { eq?: (c: string, v: string) => PromiseLike<unknown> };
        };
        const del = builder?.delete?.();
        if (del && typeof del.eq === 'function') {
          await del.eq(col, val);
        }
      } catch {
        // Non-fatal best-effort delete
      }
    };

    const childOps: Promise<void>[] = [];
    for (const pid of candidateUuids) {
      childOps.push(
        safeDeleteEq('patient_clinical_assessments', 'patient_id', pid),
        safeDeleteEq('rehab_schedules', 'patient_id', pid),
        safeDeleteEq('patient_biometrics', 'patient_id', pid),
        safeDeleteEq('face_templates', 'patient_id', pid),
        safeDeleteEq('face_enrollments', 'patient_id', pid),
        safeDeleteEq('patient_courses', 'patient_id', pid),
        safeDeleteEq('rehab_sessions', 'patient_id', pid),
        safeDeleteEq('rehab_session_history', 'patient_id', pid),
        safeDeleteEq('medical_notes', 'patient_id', pid),
        safeDeleteEq('treatment_reports', 'patient_id', pid),
        safeDeleteEq('staff_calls', 'patient_id', pid),
        safeDeleteEq('active_sessions', 'user_id', pid),
        safeDeleteEq('staff_users', 'patient_id', pid),
        safeDeleteEq('staff_users', 'id', pid),
      );
    }
    for (const h of hnVariants) {
      childOps.push(
        safeDeleteEq('patient_clinical_assessments', 'patient_hn', h),
        safeDeleteEq('patient_biometrics', 'hn', h),
      );
    }
    for (const uname of usernameVariants) {
      childOps.push(safeDeleteEq('staff_users', 'username', uname));
    }
    if (cleanFullName) {
      childOps.push(
        (async () => {
          try {
            const builder = supabase.from('staff_users') as unknown as {
              delete?: () => {
                eq?: (
                  c: string,
                  v: string,
                ) => { eq?: (c2: string, v2: string) => PromiseLike<unknown> } & PromiseLike<unknown>;
              };
            };
            const q1 = builder?.delete?.()?.eq?.('role', 'patient');
            if (q1 && typeof q1.eq === 'function') {
              await q1.eq('full_name', cleanFullName);
            }
          } catch {
            // Non-fatal
          }
        })(),
      );
    }

    await Promise.allSettled(childOps);

    const parentOps: Promise<void>[] = [
      ...candidateUuids.map((pid) => safeDeleteEq('patients', 'id', pid)),
      ...hnVariants.map((h) => safeDeleteEq('patients', 'hn', h)),
    ];
    await Promise.allSettled(parentOps);
  } catch {
    // Best-effort cascading delete never crashes UI
  }
}
