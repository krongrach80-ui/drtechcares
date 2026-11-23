import fs from 'node:fs';
import path from 'node:path';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import type { SupabaseClient } from '@supabase/supabase-js';
import { l2Normalize } from '@drtechcares/smart-biometric-core';
import {
  DrTechCaresApiClient,
  type UserAccount,
} from '@drtechcares/api-client';
import { ClinicalI18nProvider } from '@drtechcares/clinical-i18n';
import { ClinicalStaffPortalShell } from '../components/ClinicalStaffPortalShell';
import {
  formatCanonicalHn,
  mirrorSynthesizedPatientsToSupabase,
  normalizeHnKey,
  unifyClinicPatients,
  unifyClinicPatientsDetailed,
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
  username: '@hn53309',
  role: 'patient',
  fullName: 'มาโนชญ์ อ่อนสิงห์',
  phone: '0821699168',
  patientId: '7c864f61-4cb0-40f8-9336-dc5654ab9e56',
  active: true,
  createdAt: '2026-11-20T08:00:00.000Z',
};

const STAFF_USER_POOM: UserAccount = {
  id: '49bf2cc5-6c9f-4668-b99d-b6724867141d',
  username: '@hn410382',
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
    face_biometric_status: 'PENDING',
    face_frames_count: 0,
    face_image_url: null,
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

describe('Unified Clinic Patient Directory (Staff Management + Patients + EHR + Appointments)', () => {
  it('normalizes @hn usernames and merges staff_users PATIENT accounts with 0 or 1 Supabase patient rows without duplicates', () => {
    expect(normalizeHnKey('@hn410382')).toBe('hn410382');
    expect(normalizeHnKey('HN-410382')).toBe('hn410382');
    expect(formatCanonicalHn('@hn53309')).toBe('HN-53309');
    expect(formatCanonicalHn('@hn410382')).toBe('HN-410382');

    const staffList: UserAccount[] = [ADMIN_USER, STAFF_USER_MANOT, STAFF_USER_POOM];

    // Case 1: Supabase patients table returns 0 rows -> both PATIENT accounts synthesized cleanly
    const fromZero = unifyClinicPatientsDetailed([], staffList);
    expect(fromZero.patients).toHaveLength(2);
    expect(fromZero.synthesizedPatients).toHaveLength(2);
    expect(fromZero.patients.map((p) => p.hn)).toEqual(['HN-53309', 'HN-410382']);
    expect(fromZero.patients.map((p) => p.fullName)).toEqual(['มาโนชญ์ อ่อนสิงห์', 'Poom']);

    // Case 2: Supabase patients table returns 1 row (Poom HN-410382) -> deduplicates Poom and synthesizes Manot
    const fromOne = unifyClinicPatientsDetailed([makePoomSupabaseRow()], staffList);
    expect(fromOne.patients).toHaveLength(2);
    expect(fromOne.synthesizedPatients).toHaveLength(1);
    expect(fromOne.synthesizedPatients[0]?.hn).toBe('HN-53309');
    expect(fromOne.synthesizedPatients[0]?.fullName).toBe('มาโนชญ์ อ่อนสิงห์');

    const poom = fromOne.patients.find((p) => p.hn === 'HN-410382');
    const manot = fromOne.patients.find((p) => p.hn === 'HN-53309');
    expect(poom?.fullName).toBe('Poom');
    expect(poom?.faceBiometricStatus).toBe('VERIFIED');
    expect(poom?.faceFramesCount).toBe(120);
    expect(manot?.fullName).toBe('มาโนชญ์ อ่อนสิงห์');
    expect(manot?.phone).toBe('0821699168');

    // Convenience helper returns identical 2-account unified array
    const simpleList = unifyClinicPatients([makePoomSupabaseRow()], staffList);
    expect(simpleList).toHaveLength(2);
    expect(simpleList.map((p) => p.hn).sort()).toEqual(['HN-410382', 'HN-53309']);
  });

  it('triggers background upsert into Supabase public.patients for synthesized PATIENT user accounts', async () => {
    const upsertedRows: Record<string, unknown>[] = [];
    const mockSb = {
      from: (table: string) => ({
        upsert: async (payload: Record<string, unknown>) => {
          if (table === 'patients') {
            upsertedRows.push(payload);
          }
          return { error: null };
        },
      }),
    } as unknown as SupabaseClient;

    const { synthesizedPatients } = unifyClinicPatientsDetailed(
      [makePoomSupabaseRow()],
      [ADMIN_USER, STAFF_USER_MANOT, STAFF_USER_POOM],
    );
    expect(synthesizedPatients).toHaveLength(1);

    const count1 = await mirrorSynthesizedPatientsToSupabase(mockSb, synthesizedPatients);
    expect(count1).toBe(1);
    expect(upsertedRows).toHaveLength(1);
    expect(upsertedRows[0]?.hn).toBe('HN-53309');
    expect(upsertedRows[0]?.full_name).toBe('มาโนชญ์ อ่อนสิงห์');

    // Deduplicated per client: calling again does not spam duplicate upserts
    const count2 = await mirrorSynthesizedPatientsToSupabase(mockSb, synthesizedPatients);
    expect(count2).toBe(0);
    expect(upsertedRows).toHaveLength(1);
  });

  it('renders both Poom (HN-410382) and มาโนชญ์ อ่อนสิงห์ (HN-53309) across Tab 2 Patients Directory, Tab 5 Schedules, and Individual 0-6 Clinical Assessments', () => {
    const apiClient = new DrTechCaresApiClient();
    const staffUsers = [ADMIN_USER, STAFF_USER_MANOT, STAFF_USER_POOM];
    const poomOnlyFromPatients = unifyClinicPatients([makePoomSupabaseRow()], []);

    // Tab 2: ข้อมูลคนไข้ — starts with 1 row in patients + 2 PATIENT users in staff_users -> displays 2 บัญชี!
    const patientsTabHtml = renderToStaticMarkup(
      React.createElement(
        ClinicalI18nProvider,
        null,
        React.createElement(ClinicalStaffPortalShell, {
          apiClient,
          currentUser: ADMIN_USER,
          onLogout: () => {},
          initialTab: 'patients',
          initialPatients: poomOnlyFromPatients,
          initialUsers: staffUsers,
        }),
      ),
    );
    expect(patientsTabHtml).toContain('ซิงก์ข้อมูลจาก Supabase: 2 บัญชี');
    expect(patientsTabHtml).toContain('Poom');
    expect(patientsTabHtml).toContain('HN-410382');
    expect(patientsTabHtml).toContain('มาโนชญ์ อ่อนสิงห์');
    expect(patientsTabHtml).toContain('HN-53309');
    expect(EMOJI_RE.test(patientsTabHtml)).toBe(false);

    // Tab 5: ตารางนัด — renders selectable patient options instead of "ยังไม่มีคนไข้ในระบบ"
    const schedulesTabHtml = renderToStaticMarkup(
      React.createElement(
        ClinicalI18nProvider,
        null,
        React.createElement(ClinicalStaffPortalShell, {
          apiClient,
          currentUser: ADMIN_USER,
          onLogout: () => {},
          initialTab: 'schedules',
          initialPatients: [],
          initialUsers: staffUsers,
        }),
      ),
    );
    expect(schedulesTabHtml).not.toContain('ยังไม่มีคนไข้ในระบบ');
    expect(schedulesTabHtml).toContain('มาโนชญ์ อ่อนสิงห์ (HN-53309)');
    expect(schedulesTabHtml).toContain('Poom (HN-410382)');
    expect(EMOJI_RE.test(schedulesTabHtml)).toBe(false);

    // Tab 2B: เวชระเบียนรายบุคคล (ประเมิน 0-6) — renders both patients selectable for 0-6 assessment
    const assessmentsTabHtml = renderToStaticMarkup(
      React.createElement(
        ClinicalI18nProvider,
        null,
        React.createElement(ClinicalStaffPortalShell, {
          apiClient,
          currentUser: ADMIN_USER,
          onLogout: () => {},
          initialTab: 'assessments',
          initialPatients: poomOnlyFromPatients,
          initialUsers: staffUsers,
        }),
      ),
    );
    expect(assessmentsTabHtml).toContain('HN-410382');
    expect(assessmentsTabHtml).toContain('HN-53309');
    expect(assessmentsTabHtml).toContain('มาโนชญ์ อ่อนสิงห์');
    expect(assessmentsTabHtml).toContain('Poom');
    expect(EMOJI_RE.test(assessmentsTabHtml)).toBe(false);
  });

  it('validates SQL RLS migration 010 and guarantees zero Unicode emojis in all modified files', () => {
    const root = process.cwd();
    const sqlPath = path.join(
      root,
      'supabase/migrations/010_patients_rls_policies_and_staff_user_backfill.sql',
    );
    const sql = fs.readFileSync(sqlPath, 'utf8');
    expect(sql).toContain('FOR SELECT');
    expect(sql).toContain('FOR INSERT');
    expect(sql).toContain('FOR UPDATE');
    expect(sql).toContain('public.patients');
    expect(sql).toContain('public.staff_users');
    expect(sql).toContain("NOTIFY pgrst, 'reload schema'");

    const files = [
      'supabase/migrations/010_patients_rls_policies_and_staff_user_backfill.sql',
      'src/utils/unifyPatientDirectory.ts',
      'src/components/ClinicalStaffPortalShell.tsx',
      'packages/api-client/src/cloudRouter.ts',
      'packages/api-client/src/index.ts',
      'src/__tests__/unifyPatientDirectory.test.ts',
    ];
    for (const rel of files) {
      const content = fs.readFileSync(path.join(root, rel), 'utf8');
      expect(EMOJI_RE.test(content), `Unexpected Unicode emoji in ${rel}`).toBe(false);
    }
  });
});

