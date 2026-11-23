import type { SupabaseClient } from '@supabase/supabase-js';
import {
  buildPatientDualWriteColumns,
  normalizePatientRecord,
  type PatientRecord,
  type UserAccount,
} from '@drtechcares/api-client';
import {
  isSupabaseCircuitOpen,
  recordUnsupportedColumn,
  stripOversizedBase64ForDb,
  stripUnsupportedColumnsForWrite,
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
  synthesizedPatients: PatientRecord[];
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

function normalizeFullNameKey(raw: unknown): string {
  if (typeof raw !== 'string') return '';
  return raw.trim().toLowerCase().replace(/\s+/g, ' ');
}

function isPatientRoleAccount(user: StaffPatientMirrorCandidate): boolean {
  const roleUpper = String(user.role ?? '').trim().toUpperCase();
  return roleUpper === 'PATIENT' && user.active !== false;
}

/**
 * Deterministically unifies Supabase `patients` rows with any `role === 'PATIENT'` accounts
 * loaded from `staff_users` (Staff Management tab) so all Admin/Staff tabs (`2. ข้อมูลคนไข้`,
 * `เวชระเบียนรายบุคคล (ประเมิน 0-6)`, and `5. ตารางนัด`) always display the complete,
 * deduplicated clinic roster.
 */
export function unifyClinicPatientsDetailed(
  supabasePatients: ReadonlyArray<Record<string, unknown> | PatientRecord>,
  staffUserAccounts: ReadonlyArray<UserAccount | StaffPatientMirrorCandidate> = [],
): UnifyClinicPatientsResult {
  const unified: PatientRecord[] = [];
  const byId = new Map<string, number>();
  const byHnKey = new Map<string, number>();
  const byNameKey = new Map<string, number>();

  const registerPatient = (patient: PatientRecord): number => {
    const idx = unified.length;
    unified.push(patient);
    const idLo = patient.id.trim().toLowerCase();
    const hnKey = normalizeHnKey(patient.hn);
    const nameKey = normalizeFullNameKey(patient.fullName);
    if (idLo) byId.set(idLo, idx);
    if (hnKey) byHnKey.set(hnKey, idx);
    if (nameKey) byNameKey.set(nameKey, idx);
    return idx;
  };

  // 1. Hydrate all Supabase patients (top-level columns + history JSON) with deterministic deduplication
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

    registerPatient(canonicalPatient);
  }

  // 2. Merge any active role === 'PATIENT' accounts from staff_users
  const synthesizedPatients: PatientRecord[] = [];
  for (const rawUser of staffUserAccounts) {
    const user = rawUser as StaffPatientMirrorCandidate;
    if (!user || !isPatientRoleAccount(user)) continue;

    const linkedPatientId = String(user.patientId ?? user.patient_id ?? '').trim();
    const userId = String(user.id ?? '').trim();
    const fullName = String(user.fullName ?? user.full_name ?? '').trim();
    const username = String(user.username ?? '').trim();
    const phone = String(user.phone ?? '').trim();
    const createdAt = String(user.createdAt ?? user.created_at ?? new Date().toISOString());

    const hnKey = normalizeHnKey(username);
    const nameKey = normalizeFullNameKey(fullName);
    const existingIdx =
      (linkedPatientId && byId.has(linkedPatientId.toLowerCase())
        ? byId.get(linkedPatientId.toLowerCase())
        : undefined) ??
      (userId && byId.has(userId.toLowerCase()) ? byId.get(userId.toLowerCase()) : undefined) ??
      (hnKey && byHnKey.has(hnKey) ? byHnKey.get(hnKey) : undefined) ??
      (nameKey && byNameKey.has(nameKey) ? byNameKey.get(nameKey) : undefined);

    if (existingIdx !== undefined) {
      const existing = unified[existingIdx];
      if (!existing.phone && phone) {
        unified[existingIdx] = { ...existing, phone };
      }
      continue;
    }

    const effectiveId = linkedPatientId || userId;
    if (!effectiveId) continue;
    const canonicalHn = formatCanonicalHn(username, effectiveId);
    const linkSuffix = effectiveId.replace(/[^a-zA-Z0-9]/g, '').slice(0, 8).toUpperCase().padStart(8, '0');

    const synthesized: PatientRecord = {
      id: effectiveId,
      hn: canonicalHn,
      fullName: fullName || canonicalHn,
      age: 0,
      gender: 'ชาย',
      phone,
      diagnosis: 'รอประเมินอาการทางกายภาพบำบัด',
      strokeType: 'ELDERLY_GENERAL',
      affectedSide: 'NONE',
      motorAssessmentScore: 3,
      adlDressingScore: 3,
      maxSafeRomDeg: 150,
      universalHealthLinkCode: `DTC-EHR-${linkSuffix}`,
      assignedPtId: '',
      delegatedToPtId: null,
      delegationNote: null,
      nextAppointmentDate: '',
      registrationStatus: 'active',
      enrolledAt: createdAt,
      faceBiometricStatus: 'PENDING',
      faceFramesCount: 0,
      rehabSchedules: [],
    };

    registerPatient(synthesized);
    synthesizedPatients.push(synthesized);
  }

  return { patients: unified, synthesizedPatients };
}

/**
 * Deterministic helper returning the unified deduplicated `PatientRecord[]` directory
 * combining Supabase `patients` and `role === 'PATIENT'` accounts from `staff_users`.
 */
export function unifyClinicPatients(
  supabasePatients: ReadonlyArray<Record<string, unknown> | PatientRecord>,
  staffUserAccounts: ReadonlyArray<UserAccount | StaffPatientMirrorCandidate> = [],
): PatientRecord[] {
  return unifyClinicPatientsDetailed(supabasePatients, staffUserAccounts).patients;
}

const mirroredPatientIdsByClient = new WeakMap<object, Set<string>>();

/**
 * Triggers a deduplicated, non-blocking background upsert into Supabase `public.patients`
 * for any `PATIENT` user accounts synthesized from `staff_users` so both tables stay permanently synced.
 */
export function mirrorSynthesizedPatientsToSupabase(
  supabase: SupabaseClient | null | undefined,
  synthesizedPatients: ReadonlyArray<PatientRecord>,
): Promise<number> {
  if (!supabase || synthesizedPatients.length === 0 || isSupabaseCircuitOpen(supabase)) {
    return Promise.resolve(0);
  }
  let seen = mirroredPatientIdsByClient.get(supabase);
  if (!seen) {
    seen = new Set<string>();
    mirroredPatientIdsByClient.set(supabase, seen);
  }

  const pending = synthesizedPatients.filter((p) => p.id && !seen!.has(p.id));
  if (pending.length === 0) return Promise.resolve(0);
  for (const p of pending) seen.add(p.id);

  const runMirror = async (): Promise<number> => {
    let mirroredCount = 0;
    for (const p of pending) {
      try {
        const tableBuilder = supabase.from('patients') as unknown as {
          upsert?: (
            payload: Record<string, unknown>,
            opts?: { onConflict?: string },
          ) => PromiseLike<{ error?: { code?: string; message?: string } | null }>;
        };
        if (!tableBuilder || typeof tableBuilder.upsert !== 'function') break;

        const nowIso = new Date().toISOString();
        const dualCols = buildPatientDualWriteColumns(p, {
          updatedAtIso: nowIso,
          includeProfileFields: true,
        });
        const working = stripUnsupportedColumnsForWrite(
          'patients',
          stripOversizedBase64ForDb({
            id: p.id,
            hn: p.hn,
            full_name: p.fullName,
            age: Number.isFinite(p.age) ? p.age : 0,
            phone: p.phone ?? '',
            diagnosis: p.diagnosis ?? '',
            registration_status: 'active',
            ...dualCols,
            history: {
              gender: p.gender ?? 'ชาย',
              strokeType: p.strokeType ?? 'ELDERLY_GENERAL',
              affectedSide: p.affectedSide ?? 'NONE',
              motorAssessmentScore: p.motorAssessmentScore ?? 3,
              adlDressingScore: p.adlDressingScore ?? 3,
              maxSafeRomDeg: p.maxSafeRomDeg ?? 150,
              universalHealthLinkCode: p.universalHealthLinkCode,
              faceBiometricStatus: p.faceBiometricStatus ?? 'PENDING',
              faceFramesCount: p.faceFramesCount ?? 0,
              rehabSchedules: p.rehabSchedules ?? [],
              mirroredFromStaffUsers: true,
            },
            created_at: p.enrolledAt || nowIso,
            updated_at: nowIso,
          }),
        );

        let conflictCol: 'id' | 'hn' = 'id';
        for (let attempt = 0; attempt < 10; attempt += 1) {
          const res = await tableBuilder.upsert(working, { onConflict: conflictCol });
          if (!res?.error) {
            mirroredCount += 1;
            break;
          }
          const code = String(res.error.code ?? '');
          if (code === '23505' && conflictCol === 'id') {
            conflictCol = 'hn';
            delete working.id;
            continue;
          }
          const missingCol = recordUnsupportedColumn('patients', res.error);
          if (missingCol && missingCol in working && missingCol !== 'id' && missingCol !== 'hn') {
            delete working[missingCol];
            continue;
          }
          break;
        }
      } catch {
        // Non-fatal best-effort mirror
      }
    }
    return mirroredCount;
  };

  return runMirror();
}

