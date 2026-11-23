import type { SupabaseClient } from '@supabase/supabase-js';
import {
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

