import fs from 'node:fs';
import path from 'node:path';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it, vi } from 'vitest';
import type { SupabaseClient } from '@supabase/supabase-js';
import { l2Normalize } from '@drtechcares/smart-biometric-core';
import {
  DrTechCaresApiClient,
  createCloudDelegatedRouter,
  type PatientRecord,
  type UserAccount,
} from '@drtechcares/api-client';
import { ClinicalI18nProvider } from '@drtechcares/clinical-i18n';
import { ClinicalStaffPortalShell } from '../components/ClinicalStaffPortalShell';
import { formatKioskSupabaseSyncBadgeLabel } from '../components/KioskBiometricLoginView';
import {
  applyCascadingPatientDeletionToState,
  buildPatientStaffUserAccount,
  cascadeDeletePatientInSupabase,
  formatCanonicalHn,
  formatPatientStaffHandle,
  formatPatientStaffUsername,
  mergePatientsIntoStaffUsersList,
  mirrorSynthesizedPatientsToSupabase,
  normalizeHnKey,
  unifyClinicPatients,
  unifyClinicPatientsDetailed,
  upsertPatientIntoStaffUsersState,
  upsertPatientStaffUserInSupabase,
} from '../utils/unifyPatientDirectory';

const EMOJI_RE = /[\p{Extended_Pictographic}\u{FE0F}\u{2600}-\u{27BF}\u{23E9}-\u{23FA}]/u;

function makeVec1024(seed: number): number[] {
  const raw = new Array<number>(1024);
  for (let i = 0; i < 1024; i += 1) {
    const v = Math.sin((i + 1) * seed * 0.13) + 0.5 * Math.cos((i + 3) * seed * 0.27) + 0.35;
    raw[i] = v > 0 ? v : 0;
  }
  return l2Normalize(raw);
}

const ADMIN_USER: UserAccount = {
  id: '00000000-0000-4000-8000-000000000001',
  username: 'admin',
  role: 'super_admin',
  fullName: 'นพ.สมชาย ผู้อำนวยการโรงพยาบาล',
  phone: '02-000-0001',
  specialty: 'Hospital Director',
  licenseNo: 'MD-10001',
  active: true,
  createdAt: '2026-11-01T00:00:00.000Z',
};

const STAFF_USER_MANOT: UserAccount = {
  id: '4a4f1af2-9b46-43d5-8951-7105f82e5cf9',
  username: 'hn53309',
  role: 'patient',
  fullName: 'มาโนชญ์ อ่อนสิงห์',
  phone: '0821699168',
  patientId: '7c864f61-4cb0-40f8-9336-dc5654ab9e56',
  active: true,
  createdAt: '2026-11-20T08:00:00.000Z',
};

const STAFF_USER_POOM: UserAccount = {
  id: '49bf2cc5-6c9f-4668-b99d-b6724867141d',
  username: 'hn410382',
  role: 'patient',
  fullName: 'Poom',
  phone: '0914448744',
  patientId: '5104db15-5634-45e5-ab65-f2f008a15e54',
  active: true,
  createdAt: '2026-11-23T02:04:11.925523+00:00',
};

function makePoomSupabaseRow() {
  const vec = makeVec1024(4.10382);
  return {
    id: '5104db15-5634-45e5-ab65-f2f008a15e54',
    hn: 'HN-410382',
    full_name: 'Poom',
    age: 26,
    phone: '0914448744',
    diagnosis: 'Stroke Upper Limb Rehabilitation',
    registration_status: 'active',
    assigned_pt_id: '4a2c7d20-8fb9-4e4b-81fc-19a60a9dd1bc',
    next_appointment_date: '2026-11-25',
    created_at: '2026-11-23T02:04:11.925523+00:00',
    updated_at: '2026-11-23T02:04:11.925523+00:00',
    face_biometric_status: 'VERIFIED',
    face_frames_count: 120,
    face_image_url:
      'https://vyjuwjhbdhknjdzyqctz.supabase.co/storage/v1/object/public/patient-faces/5104db15-5634-45e5-ab65-f2f008a15e54/center.jpg',
    history: {
      faceBiometricStatus: 'VERIFIED',
      faceFramesCount: 120,
      faceImageUrl:
        'https://vyjuwjhbdhknjdzyqctz.supabase.co/storage/v1/object/public/patient-faces/5104db15-5634-45e5-ab65-f2f008a15e54/center.jpg',
      masterEmbedding: vec,
      angleEmbeddings: { center: vec, left: vec, right: vec, up: vec, down: vec },
      biometricEngine: 'human-faceres-1024d',
      gender: 'ชาย',
      strokeType: 'ISCHEMIC',
      affectedSide: 'RIGHT',
      motorAssessmentScore: 3,
      adlDressingScore: 3,
    },
  };
}

function makeManotSupabaseRow() {
  return {
    id: 'bf47d4a3-97ab-40fb-8561-05550da1c040',
    hn: 'HN-53309',
    full_name: 'มาโนชญ์ อ่อนสิงห์',
    age: 62,
    phone: '0821699168',
    diagnosis: 'รอประเมินอาการทางกายภาพบำบัด',
    registration_status: 'active',
    assigned_pt_id: '',
    next_appointment_date: '',
    created_at: '2026-11-20T08:00:00.000Z',
    updated_at: '2026-11-20T08:00:00.000Z',
    face_biometric_status: 'PENDING',
    face_frames_count: 0,
    face_image_url: null,
    history: {
      faceBiometricStatus: 'PENDING',
      faceFramesCount: 0,
      gender: 'ชาย',
      strokeType: 'ELDERLY_GENERAL',
      affectedSide: 'NONE',
      motorAssessmentScore: 3,
      adlDressingScore: 3,
    },
  };
}

describe('Cascading Patient Delete (Zero Resurrection) & Unified Kiosk Sync Count', () => {
  it('treats public.patients as single source of truth and NEVER resurrects deleted patients from staff_users on read/render', async () => {
    expect(normalizeHnKey('@hn410382')).toBe('hn410382');
    expect(normalizeHnKey('HN-410382')).toBe('hn410382');
    expect(formatCanonicalHn('@hn53309')).toBe('HN-53309');

    const staffList: UserAccount[] = [ADMIN_USER, STAFF_USER_MANOT, STAFF_USER_POOM];

    // Even if staff_users still had an orphan row for Manot, unifyClinicPatients strictly uses public.patients
    const result = unifyClinicPatientsDetailed([makePoomSupabaseRow()], staffList);
    expect(result.patients).toHaveLength(1);
    expect(result.patients[0]?.hn).toBe('HN-410382');
    expect(result.patients[0]?.fullName).toBe('Poom');
    expect(result.patients.some((p) => p.hn === 'HN-53309')).toBe(false);
    expect(result.synthesizedPatients).toHaveLength(0);

    // And mirrorSynthesizedPatientsToSupabase must never INSERT/UPSERT during read/render cycles
    const upsertSpy = vi.fn();
    const mockSb = {
      from: () => ({ upsert: upsertSpy }),
    } as unknown as SupabaseClient;
    const mirrored = await mirrorSynthesizedPatientsToSupabase(mockSb, result.patients);
    expect(mirrored).toBe(0);
    expect(upsertSpy).not.toHaveBeenCalled();
  });

  it('cascade-deletes a patient from BOTH patients and staff_users (by id, normalized HN/username, and full_name) plus assessments and schedules without resurrection', async () => {
    const initialPatients = unifyClinicPatients([makeManotSupabaseRow(), makePoomSupabaseRow()], []);
    const initialUsers: UserAccount[] = [ADMIN_USER, STAFF_USER_MANOT, STAFF_USER_POOM];
    expect(initialPatients).toHaveLength(2);
    expect(initialUsers).toHaveLength(3);

    // 1. Delete Manot (HN-53309, different UUID from staff_users.patient_id!) from Tab 2 Patients Directory
    const manot = initialPatients.find((p) => p.hn === 'HN-53309')!;
    const afterDelete = applyCascadingPatientDeletionToState(initialPatients, initialUsers, {
      id: manot.id,
      patientId: manot.id,
      hn: manot.hn,
      fullName: manot.fullName,
      role: 'patient',
    });

    expect(afterDelete.patients.map((p) => p.hn)).toEqual(['HN-410382']);
    expect(afterDelete.users.map((u) => u.username)).toEqual(['admin', 'hn410382']);
    expect(afterDelete.users.some((u) => u.username === 'hn53309')).toBe(false);

    // Subsequent directory unification/reload NEVER resurrects Manot
    const afterReload = unifyClinicPatients(afterDelete.patients, afterDelete.users);
    expect(afterReload.map((p) => p.hn)).toEqual(['HN-410382']);
    expect(afterReload.some((p) => p.hn === 'HN-53309')).toBe(false);

    // 2. Delete Poom from Tab 1 Staff Management (role === 'patient') -> also removes Poom from patients!
    const afterDeleteStaffPatient = applyCascadingPatientDeletionToState(
      afterDelete.patients,
      afterDelete.users,
      {
        id: STAFF_USER_POOM.id,
        patientId: STAFF_USER_POOM.patientId,
        username: STAFF_USER_POOM.username,
        fullName: STAFF_USER_POOM.fullName,
        role: STAFF_USER_POOM.role,
      },
    );
    expect(afterDeleteStaffPatient.patients).toHaveLength(0);
    expect(afterDeleteStaffPatient.users.map((u) => u.username)).toEqual(['admin']);

    // 3. Direct Supabase cascading delete cleans patients, staff_users (username hn53309 + full_name), assessments & schedules
    const deletedOps: Array<{ table: string; col: string; val: string }> = [];
    const mockSb = {
      from: (table: string) => ({
        delete: () => ({
          eq: (col: string, val: string) => {
            deletedOps.push({ table, col, val });
            return {
              eq: async (col2: string, val2: string) => {
                deletedOps.push({ table, col: `${col}+${col2}`, val: `${val}+${val2}` });
                return { error: null };
              },
              then: (resolve: (v: unknown) => void) => resolve({ error: null }),
            };
          },
        }),
      }),
    } as unknown as SupabaseClient;

    await cascadeDeletePatientInSupabase(mockSb, {
      id: manot.id,
      patientId: manot.id,
      hn: 'HN-53309',
      username: 'hn53309',
      fullName: 'มาโนชญ์ อ่อนสิงห์',
      role: 'patient',
    });

    expect(deletedOps.some((op) => op.table === 'patients' && op.col === 'id' && op.val === manot.id)).toBe(true);
    expect(deletedOps.some((op) => op.table === 'patients' && op.col === 'hn' && op.val === 'HN-53309')).toBe(true);
    expect(deletedOps.some((op) => op.table === 'staff_users' && op.col === 'username' && op.val === 'hn53309')).toBe(true);
    expect(deletedOps.some((op) => op.table === 'staff_users' && op.col === 'role+full_name' && op.val === 'patient+มาโนชญ์ อ่อนสิงห์')).toBe(true);
    expect(deletedOps.some((op) => op.table === 'patient_clinical_assessments' && op.col === 'patient_id' && op.val === manot.id)).toBe(true);
    expect(deletedOps.some((op) => op.table === 'rehab_schedules' && op.col === 'patient_id' && op.val === manot.id)).toBe(true);
  });

  it('cascade-deletes HN-53309 on Cloud Worker router without resurrection on subsequent GET /api/patients', async () => {
    const dbPatients = [makeManotSupabaseRow(), makePoomSupabaseRow()];
    const dbStaffUsers = [
      {
        id: '00000000-0000-4000-8000-000000000001',
        username: 'admin',
        password_hash:
          'pbkdf2:dceb3f63b5ff16065454e730ac190076:405a025959bb0a88efc4ee2b0d7f1cc91f8c274e286980cb053b1409d17626aa',
        full_name: 'นพ.สมชาย ผู้อำนวยการโรงพยาบาล',
        phone: '02-000-0001',
        role: 'super_admin',
        active: true,
      },
      {
        id: STAFF_USER_MANOT.id,
        username: 'hn53309',
        password_hash: 'pbkdf2:00:00',
        full_name: 'มาโนชญ์ อ่อนสิงห์',
        phone: '0821699168',
        patient_id: '7c864f61-4cb0-40f8-9336-dc5654ab9e56',
        role: 'patient',
        active: true,
      },
      {
        id: STAFF_USER_POOM.id,
        username: 'hn410382',
        password_hash: 'pbkdf2:00:00',
        full_name: 'Poom',
        phone: '0914448744',
        patient_id: '5104db15-5634-45e5-ab65-f2f008a15e54',
        role: 'patient',
        active: true,
      },
    ];

    const deletedStaffUserQueries: string[] = [];
    const origFetch = globalThis.fetch;
    try {
      globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url;
        const method = init?.method ?? (input instanceof Request ? input.method : 'GET');
        if (method === 'DELETE' && url.includes('/rest/v1/patients')) {
          const parsedUrl = new URL(url);
          const idFilter = parsedUrl.searchParams.get('id')?.replace(/^eq\./, '');
          const hnFilter = parsedUrl.searchParams.get('hn')?.replace(/^eq\./, '');
          for (let i = dbPatients.length - 1; i >= 0; i -= 1) {
            if ((idFilter && dbPatients[i].id === idFilter) || (hnFilter && dbPatients[i].hn === hnFilter)) {
              dbPatients.splice(i, 1);
            }
          }
          return new Response(JSON.stringify([]), { status: 200, headers: { 'Content-Type': 'application/json' } });
        }
        if (method === 'DELETE' && url.includes('/rest/v1/staff_users')) {
          deletedStaffUserQueries.push(url);
          const parsedUrl = new URL(url);
          const idFilter = parsedUrl.searchParams.get('id')?.replace(/^eq\./, '');
          const userFilter = parsedUrl.searchParams.get('username')?.replace(/^eq\./, '');
          const nameFilter = parsedUrl.searchParams.get('full_name')?.replace(/^eq\./, '');
          for (let i = dbStaffUsers.length - 1; i >= 0; i -= 1) {
            if (
              (idFilter && dbStaffUsers[i].id === idFilter) ||
              (userFilter && dbStaffUsers[i].username === userFilter) ||
              (nameFilter && dbStaffUsers[i].full_name === nameFilter && dbStaffUsers[i].role === 'patient')
            ) {
              dbStaffUsers.splice(i, 1);
            }
          }
          return new Response(JSON.stringify([]), { status: 200, headers: { 'Content-Type': 'application/json' } });
        }
        if (method === 'GET' && url.includes('/rest/v1/patients')) {
          return new Response(JSON.stringify(dbPatients), { status: 200, headers: { 'Content-Type': 'application/json' } });
        }
        if (method === 'GET' && url.includes('/rest/v1/staff_users')) {
          return new Response(JSON.stringify(dbStaffUsers), { status: 200, headers: { 'Content-Type': 'application/json' } });
        }
        return new Response(JSON.stringify([]), { status: 200, headers: { 'Content-Type': 'application/json' } });
      }) as typeof fetch;

      const fakeJwt =
        'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJyb2xlIjoic2VydmljZV9yb2xlIn0.signaturevaltestsafe5544332211';
      const router = createCloudDelegatedRouter({
        supabaseUrl: 'https://cascade-delete-test.supabase.co',
        supabaseServiceKey: fakeJwt,
      });

      const loginRes = await router(
        new Request('https://drtechcares.workers.dev/api/auth/login', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ username: 'admin', password: 'Admin@2026' }),
        }),
      );
      const { token } = (await loginRes.json()) as { token: string };

      // Load both patients first
      const beforeRes = await router(
        new Request('https://drtechcares.workers.dev/api/patients?scope=all&refresh=1', {
          headers: { Authorization: `Bearer ${token}` },
        }),
      );
      const beforeJson = (await beforeRes.json()) as { patients: PatientRecord[] };
      expect(beforeJson.patients).toHaveLength(2);

      // Delete Manot (HN-53309)
      const delRes = await router(
        new Request('https://drtechcares.workers.dev/api/patients/bf47d4a3-97ab-40fb-8561-05550da1c040', {
          method: 'DELETE',
          headers: { Authorization: `Bearer ${token}` },
        }),
      );
      expect(delRes.status).toBe(200);

      // Verify staff_users row hn53309 was cascade-deleted in DB too!
      expect(deletedStaffUserQueries.some((u) => u.includes('username=eq.hn53309'))).toBe(true);
      expect(dbStaffUsers.some((u) => u.username === 'hn53309')).toBe(false);
      expect(dbPatients.some((p) => p.hn === 'HN-53309')).toBe(false);

      // Refresh GET /api/patients -> ONLY Poom remains (Manot does NOT resurrect!)
      const afterRes = await router(
        new Request('https://drtechcares.workers.dev/api/patients?scope=all&refresh=1', {
          headers: { Authorization: `Bearer ${token}` },
        }),
      );
      const afterJson = (await afterRes.json()) as { patients: PatientRecord[] };
      expect(afterJson.patients.map((p) => p.hn)).toEqual(['HN-410382']);
    } finally {
      globalThis.fetch = origFetch;
    }
  });

  it('formats Kiosk Supabase sync status pill accurately when all patients are VERIFIED vs when PENDING patients exist', () => {
    // All clinic patients have verified face profiles
    const allVerifiedLabel = formatKioskSupabaseSyncBadgeLabel(2, 2);
    expect(allVerifiedLabel).toBe('ซิงก์ฐานข้อมูลใบหน้าจาก Supabase สำเร็จ: 2 บัญชี');
    expect(EMOJI_RE.test(allVerifiedLabel)).toBe(false);

    const singleVerifiedLabel = formatKioskSupabaseSyncBadgeLabel(1, 1);
    expect(singleVerifiedLabel).toBe('ซิงก์ฐานข้อมูลใบหน้าจาก Supabase สำเร็จ: 1 บัญชี');

    // Mixed state: 1 VERIFIED + 1 PENDING (2 total in public.patients)
    const mixedLabel = formatKioskSupabaseSyncBadgeLabel(1, 2);
    expect(mixedLabel).toBe('ซิงก์ใบหน้าพร้อมสแกนจาก Supabase: 1 บัญชี (คนไข้ในระบบทั้งหมด 2 บัญชี)');
    expect(EMOJI_RE.test(mixedLabel)).toBe(false);

    // 0 VERIFIED + 2 PENDING
    const zeroVerifiedLabel = formatKioskSupabaseSyncBadgeLabel(0, 2);
    expect(zeroVerifiedLabel).toBe('ซิงก์ใบหน้าพร้อมสแกนจาก Supabase: 0 บัญชี (คนไข้ในระบบทั้งหมด 2 บัญชี)');
    expect(EMOJI_RE.test(zeroVerifiedLabel)).toBe(false);

    // Also confirm ClinicalStaffPortalShell displays only authoritative public.patients across Tab 2 & Tab 5
    const apiClient = new DrTechCaresApiClient();
    const poomOnly = unifyClinicPatients([makePoomSupabaseRow()], []);
    const portalHtml = renderToStaticMarkup(
      React.createElement(
        ClinicalI18nProvider,
        null,
        React.createElement(ClinicalStaffPortalShell, {
          apiClient,
          currentUser: ADMIN_USER,
          onLogout: () => {},
          initialTab: 'patients',
          initialPatients: poomOnly,
          initialUsers: [ADMIN_USER, STAFF_USER_MANOT, STAFF_USER_POOM],
        }),
      ),
    );
    expect(portalHtml).toContain('ซิงก์ข้อมูลจาก Supabase: 1 บัญชี');
    expect(portalHtml).toContain('HN-410382');
    expect(portalHtml).not.toContain('HN-53309');
    expect(EMOJI_RE.test(portalHtml)).toBe(false);
  });

  it('dual-writes and unifies newly registered patients (ครูอั๋น HN-359954, สมหวัง ยาพขำ HN-663488) into both Tab 2 Patients Directory and Tab 1 Staff Management (Patient role filter)', async () => {
    expect(formatPatientStaffUsername('HN-359954')).toBe('hn359954');
    expect(formatPatientStaffHandle('HN-359954')).toBe('@hn359954');
    expect(formatPatientStaffUsername('HN-663488')).toBe('hn663488');
    expect(formatPatientStaffHandle('HN-663488')).toBe('@hn663488');

    // 1. In-memory unification: active patients in public.patients appear in Tab 1 even if missing from staff_users
    const kruAun = unifyClinicPatients(
      [
        {
          id: '11111111-1111-4111-8111-111111111111',
          hn: 'HN-359954',
          full_name: 'ครูอั๋น',
          age: 42,
          phone: '0631234567',
          diagnosis: 'Stroke ADL (Shoulder & Arm)',
          registration_status: 'active',
          face_biometric_status: 'VERIFIED',
        },
        {
          id: '22222222-2222-4222-8222-222222222222',
          hn: 'HN-663488',
          full_name: 'สมหวัง ยาพขำ',
          age: 26,
          phone: '0896634880',
          diagnosis: 'Stroke Upper Limb',
          registration_status: 'active',
          face_biometric_status: 'VERIFIED',
        },
      ],
      [],
    );
    expect(kruAun).toHaveLength(2);

    const unifiedStaff = mergePatientsIntoStaffUsersList([ADMIN_USER], kruAun);
    expect(unifiedStaff).toHaveLength(3);
    const patientRoleRows = unifiedStaff.filter((u) => u.role === 'patient');
    expect(patientRoleRows).toHaveLength(2);
    expect(patientRoleRows.map((u) => ({ id: u.id, username: u.username, fullName: u.fullName }))).toEqual([
      { id: '11111111-1111-4111-8111-111111111111', username: 'hn359954', fullName: 'ครูอั๋น' },
      { id: '22222222-2222-4222-8222-222222222222', username: 'hn663488', fullName: 'สมหวัง ยาพขำ' },
    ]);

    // Upserting into in-memory staffUsers uses the exact same patient.id UUID and deduplicates cleanly
    const afterUpsert1 = upsertPatientIntoStaffUsersState([ADMIN_USER], kruAun[0]);
    const afterUpsert2 = upsertPatientIntoStaffUsersState(afterUpsert1, {
      ...kruAun[0],
      phone: '0819998888',
    });
    expect(afterUpsert2.filter((u) => u.role === 'patient')).toHaveLength(1);
    expect(afterUpsert2.find((u) => u.id === kruAun[0].id)?.phone).toBe('0819998888');
    expect(buildPatientStaffUserAccount(kruAun[1]).id).toBe(kruAun[1].id);

    // Non-blocking Supabase dual-write upserts staff_users gracefully even on schema drift
    const upsertedPayloads: Record<string, unknown>[] = [];
    const mockSupabase = {
      from: (table: string) => ({
        upsert: async (payload: Record<string, unknown>) => {
          if (table === 'staff_users') upsertedPayloads.push(payload);
          return { error: null };
        },
      }),
    } as unknown as SupabaseClient;
    const syncedOk = await upsertPatientStaffUserInSupabase(mockSupabase, kruAun[0]);
    expect(syncedOk).toBe(true);
    expect(upsertedPayloads).toHaveLength(1);
    expect(upsertedPayloads[0]).toMatchObject({
      id: kruAun[0].id,
      username: 'hn359954',
      full_name: 'ครูอั๋น',
      phone: '0631234567',
      role: 'patient',
      active: true,
    });

    // 2. Render ClinicalStaffPortalShell Tab 1 ('staff') with ONLY [ADMIN_USER] in staff_users + 2 active patients
    const apiClient = new DrTechCaresApiClient();
    const staffTabHtml = renderToStaticMarkup(
      React.createElement(
        ClinicalI18nProvider,
        null,
        React.createElement(ClinicalStaffPortalShell, {
          apiClient,
          currentUser: ADMIN_USER,
          onLogout: () => {},
          initialTab: 'staff',
          initialPatients: kruAun,
          initialUsers: [ADMIN_USER],
        }),
      ),
    );
    expect(staffTabHtml).toContain('ครูอั๋น');
    expect(staffTabHtml).toContain('@hn359954');
    expect(staffTabHtml).toContain('สมหวัง ยาพขำ');
    expect(staffTabHtml).toContain('@hn663488');
    expect(staffTabHtml).not.toContain('@@hn359954');
    expect(EMOJI_RE.test(staffTabHtml)).toBe(false);

    // 3. End-to-end CloudRouter: POST /api/patients dual-writes staff_users, supports suspend/reset-PIN, and cascading delete from either tab leaves zero resurrection
    const router = createCloudDelegatedRouter({});
    const loginRes = await router(
      new Request('https://drtechcares.workers.dev/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: 'admin', password: 'Admin@2026' }),
      }),
    );
    expect(loginRes.status).toBe(200);
    const { token } = (await loginRes.json()) as { token: string };

    // Register ครูอั๋น (HN-359954) & สมหวัง ยาพขำ (HN-663488)
    const createAunRes = await router(
      new Request('https://drtechcares.workers.dev/api/patients', {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
        body: JSON.stringify({
          hn: 'HN-359954',
          fullName: 'ครูอั๋น',
          age: 42,
          phone: '0631234567',
          diagnosis: 'Stroke ADL (Shoulder & Arm)',
        }),
      }),
    );
    expect(createAunRes.status).toBe(201);
    const { patient: createdAun } = (await createAunRes.json()) as { patient: PatientRecord };

    const createSomwangRes = await router(
      new Request('https://drtechcares.workers.dev/api/patients', {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
        body: JSON.stringify({
          hn: 'HN-663488',
          fullName: 'สมหวัง ยาพขำ',
          age: 26,
          phone: '0896634880',
          diagnosis: 'Stroke Upper Limb',
        }),
      }),
    );
    expect(createSomwangRes.status).toBe(201);
    const { patient: createdSomwang } = (await createSomwangRes.json()) as { patient: PatientRecord };

    // Both MUST appear in GET /api/patients AND GET /api/users under role === 'patient'
    const [ptsRes, usersRes] = await Promise.all([
      router(
        new Request('https://drtechcares.workers.dev/api/patients?scope=all', {
          headers: { Authorization: `Bearer ${token}` },
        }),
      ),
      router(
        new Request('https://drtechcares.workers.dev/api/users', {
          headers: { Authorization: `Bearer ${token}` },
        }),
      ),
    ]);
    const ptsJson = (await ptsRes.json()) as { patients: PatientRecord[] };
    const usersJson = (await usersRes.json()) as { users: UserAccount[] };
    expect(ptsJson.patients.some((p) => p.hn === 'HN-359954')).toBe(true);
    expect(ptsJson.patients.some((p) => p.hn === 'HN-663488')).toBe(true);
    const aunUser = usersJson.users.find((u) => u.role === 'patient' && u.username === 'hn359954');
    const somwangUser = usersJson.users.find((u) => u.role === 'patient' && u.username === 'hn663488');
    expect(aunUser).toBeDefined();
    expect(aunUser?.id).toBe(createdAun.id);
    expect(aunUser?.fullName).toBe('ครูอั๋น');
    expect(somwangUser).toBeDefined();
    expect(somwangUser?.id).toBe(createdSomwang.id);
    expect(somwangUser?.fullName).toBe('สมหวัง ยาพขำ');

    // Suspend & Reset PIN on ครูอั๋น from Tab 1 works seamlessly
    const suspendRes = await router(
      new Request(`https://drtechcares.workers.dev/api/users/${createdAun.id}`, {
        method: 'PATCH',
        headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
        body: JSON.stringify({ active: false }),
      }),
    );
    expect(suspendRes.status).toBe(200);
    const suspendJson = (await suspendRes.json()) as { user: UserAccount };
    expect(suspendJson.user.active).toBe(false);

    const resetPinRes = await router(
      new Request(`https://drtechcares.workers.dev/api/users/${createdAun.id}/reset-password`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
        body: JSON.stringify({ newPassword: '5678' }),
      }),
    );
    expect(resetPinRes.status).toBe(200);

    // Delete ครูอั๋น from Tab 1 (DELETE /api/users/:id) -> removes from BOTH patients and users permanently
    const delAunFromTab1 = await router(
      new Request(`https://drtechcares.workers.dev/api/users/${createdAun.id}`, {
        method: 'DELETE',
        headers: { Authorization: `Bearer ${token}` },
      }),
    );
    expect(delAunFromTab1.status).toBe(200);

    // Delete สมหวัง ยาพขำ from Tab 2 (DELETE /api/patients/:id) -> removes from BOTH patients and users permanently
    const delSomwangFromTab2 = await router(
      new Request(`https://drtechcares.workers.dev/api/patients/${createdSomwang.id}`, {
        method: 'DELETE',
        headers: { Authorization: `Bearer ${token}` },
      }),
    );
    expect(delSomwangFromTab2.status).toBe(200);

    // Neither ครูอั๋น nor สมหวัง ยาพขำ resurrects in GET /api/patients OR GET /api/users
    const [ptsAfterRes, usersAfterRes] = await Promise.all([
      router(
        new Request('https://drtechcares.workers.dev/api/patients?scope=all', {
          headers: { Authorization: `Bearer ${token}` },
        }),
      ),
      router(
        new Request('https://drtechcares.workers.dev/api/users', {
          headers: { Authorization: `Bearer ${token}` },
        }),
      ),
    ]);
    const ptsAfter = (await ptsAfterRes.json()) as { patients: PatientRecord[] };
    const usersAfter = (await usersAfterRes.json()) as { users: UserAccount[] };
    expect(ptsAfter.patients.some((p) => p.hn === 'HN-359954' || p.hn === 'HN-663488')).toBe(false);
    expect(
      usersAfter.users.some((u) => u.username === 'hn359954' || u.username === 'hn663488'),
    ).toBe(false);
  });

  it('audits all modified cascading-delete and Kiosk sync files for zero Unicode emojis', () => {
    const root = process.cwd();
    const files = [
      'src/utils/unifyPatientDirectory.ts',
      'src/components/ClinicalStaffPortalShell.tsx',
      'src/components/StaffAssistedEnrollmentModal.tsx',
      'src/components/KioskBiometricLoginView.tsx',
      'src/components/PortalSubModals.tsx',
      'packages/api-client/src/cloudRouter.ts',
      'packages/api-client/src/supabasePersistence.ts',
      'src/__tests__/unifyPatientDirectory.test.ts',
    ];
    for (const rel of files) {
      const content = fs.readFileSync(path.join(root, rel), 'utf8');
      expect(EMOJI_RE.test(content), `Unexpected Unicode emoji in ${rel}`).toBe(false);
    }
  });
});
