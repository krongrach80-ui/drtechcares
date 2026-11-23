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
import {
  fetchAllPatientsUnified,
  filterAdminPortalPatients,
  normalizePatientRecord,
} from '../lib/patientHydration';

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

const POOM_ASSIGNED_PT_ID = '4a2c7d20-8fb9-4e4b-81fc-19a60a9dd1bc';

function makePoomRawSupabaseRow() {
  const master = makeVec1024(4.10382);
  const faceImageUrl =
    'https://vyjuwjhbdhknjdzyqctz.supabase.co/storage/v1/object/public/patient-faces/5104db15-5634-45e5-ab65-f2f008a15e54/center.jpg';
  return {
    id: '5104db15-5634-45e5-ab65-f2f008a15e54',
    hn: 'HN-410382',
    full_name: 'Poom',
    age: 26,
    phone: '0812345678',
    diagnosis: 'Stroke Upper Limb Rehabilitation',
    registration_status: 'active',
    assigned_pt_id: POOM_ASSIGNED_PT_ID,
    delegated_to_pt_id: null,
    delegation_note: null,
    next_appointment_date: '2026-11-25',
    created_at: '2026-11-23T02:04:11.925523+00:00',
    updated_at: '2026-11-23T02:04:11.925523+00:00',
    face_biometric_status: 'PENDING',
    face_frames_count: 0,
    face_image_url: null,
    master_embedding: null,
    angle_embeddings: null,
    gender: null,
    stroke_type: null,
    affected_side: null,
    motor_assessment_score: null,
    adl_dressing_score: null,
    history: {
      faceBiometricStatus: 'VERIFIED',
      faceFramesCount: 120,
      faceImageUrl,
      masterEmbedding: master,
      angleEmbeddings: {
        center: master,
        left: master,
        right: master,
        up: master,
        down: master,
      },
      biometricEngine: 'human-faceres-1024d',
      gender: 'ชาย',
      strokeType: 'ISCHEMIC',
      affectedSide: 'RIGHT',
      motorAssessmentScore: 3,
      adlDressingScore: 3,
    },
  };
}

function makeAdminAssignedPendingPatient(): PatientRecord {
  return {
    id: 'pat-admin-pending-01',
    hn: 'HN-990001',
    fullName: 'คุณสมศรี รอสแกนหน้า',
    age: 64,
    gender: 'หญิง',
    phone: '0891112222',
    diagnosis: 'Post-Stroke Shoulder Rehab',
    strokeType: 'ISCHEMIC',
    affectedSide: 'LEFT',
    motorAssessmentScore: 2,
    adlDressingScore: 2,
    assignedPtId: ADMIN_USER.id,
    delegatedToPtId: null,
    delegationNote: null,
    nextAppointmentDate: '2026-11-26',
    registrationStatus: 'active',
    enrolledAt: '2026-11-22T08:00:00.000Z',
    faceBiometricStatus: 'PENDING',
    faceFramesCount: 0,
  };
}

describe('Admin Portal Supabase Patient List Sync & Clinic Scope Filtering', () => {
  it('queries Supabase patients directly ordered by created_at DESC, hydrates Poom (HN-410382), and never overwrites with empty API response', async () => {
    const poomRow = makePoomRawSupabaseRow();
    const seenOrderArgs: Array<{ col: string; ascending?: boolean }> = [];
    const mockSupabase = {
      from: (table: string) => ({
        select: (_cols: string) => ({
          order: async (col: string, opts?: { ascending?: boolean }) => {
            seenOrderArgs.push({ col, ascending: opts?.ascending });
            if (table === 'patients') {
              return { data: [poomRow], error: null };
            }
            return { data: [], error: null };
          },
        }),
        update: () => ({
          eq: async () => ({ error: null }),
        }),
      }),
    } as unknown as SupabaseClient;

    // Simulate Worker API returning empty [] while Supabase actually has Poom
    const apiClient = {
      getSupabaseClient: () => mockSupabase,
      listPatients: vi.fn(async () => []),
    };

    const res = await fetchAllPatientsUnified(apiClient, { forceRefresh: true });
    expect(res.syncedFromSupabase).toBe(true);
    expect(res.errorMessage).toBeNull();
    expect(seenOrderArgs).toEqual([{ col: 'created_at', ascending: false }]);
    expect(res.patients).toHaveLength(1);
    expect(res.patients[0]?.id).toBe('5104db15-5634-45e5-ab65-f2f008a15e54');
    expect(res.patients[0]?.hn).toBe('HN-410382');
    expect(res.patients[0]?.fullName).toBe('Poom');
    expect(res.patients[0]?.assignedPtId).toBe(POOM_ASSIGNED_PT_ID);
    expect(res.patients[0]?.faceBiometricStatus).toBe('VERIFIED');
    expect(res.patients[0]?.faceFramesCount).toBe(120);
    expect(res.patients[0]?.gender).toBe('ชาย');
    expect(res.patients[0]?.strokeType).toBe('ISCHEMIC');
    expect(res.patients[0]?.affectedSide).toBe('RIGHT');
    expect(res.patients[0]?.motorAssessmentScore).toBe(3);
  });

  it('surfaces exact Supabase RLS/query error message cleanly instead of silently failing to 0 rows', async () => {
    const rlsErrorSupabase = {
      from: () => ({
        select: () => ({
          order: async () => ({
            data: null,
            error: {
              code: '42501',
              message: 'permission denied for table patients',
            },
          }),
        }),
      }),
    } as unknown as SupabaseClient;

    const failingApi = {
      getSupabaseClient: () => rlsErrorSupabase,
      listPatients: vi.fn(async () => {
        throw new Error('HTTP 500');
      }),
    };

    const res = await fetchAllPatientsUnified(failingApi, { forceRefresh: true });
    expect(res.syncedFromSupabase).toBe(false);
    expect(res.source).toBe('error');
    expect(res.errorMessage).toContain('42501');
    expect(res.errorMessage).toContain('permission denied for table patients');
  });

  it('renders Poom (HN-410382) in Admin Portal when logged in as admin and filters accurately when toggling All vs Assigned without losing records', () => {
    const poomHydrated = normalizePatientRecord(makePoomRawSupabaseRow());
    const adminPendingPatient = makeAdminAssignedPendingPatient();
    const allPatients = [poomHydrated, adminPendingPatient];

    // 1. Pure scope & status filter transitions never lose records
    const allScope = filterAdminPortalPatients(allPatients, {
      scopeFilter: 'all',
      faceFilter: 'all',
      searchQuery: '',
      currentUserId: ADMIN_USER.id,
    });
    expect(allScope.map((p) => p.hn)).toEqual(['HN-410382', 'HN-990001']);

    const assignedToAdmin = filterAdminPortalPatients(allPatients, {
      scopeFilter: 'assigned',
      faceFilter: 'all',
      searchQuery: '',
      currentUserId: ADMIN_USER.id,
    });
    expect(assignedToAdmin.map((p) => p.hn)).toEqual(['HN-990001']);

    const assignedToPoomPt = filterAdminPortalPatients(allPatients, {
      scopeFilter: 'assigned',
      faceFilter: 'all',
      searchQuery: '',
      currentUserId: POOM_ASSIGNED_PT_ID,
    });
    expect(assignedToPoomPt.map((p) => p.hn)).toEqual(['HN-410382']);

    // Switching back to 'all' restores full clinic roster intact
    const backToAll = filterAdminPortalPatients(allPatients, {
      scopeFilter: 'all',
      faceFilter: 'all',
      searchQuery: '',
      currentUserId: ADMIN_USER.id,
    });
    expect(backToAll).toHaveLength(2);
    expect(backToAll.some((p) => p.hn === 'HN-410382')).toBe(true);

    // 2. Render real ClinicalStaffPortalShell logged in as admin on 'patients' tab (default All scope)
    const apiClient = new DrTechCaresApiClient();
    const allViewHtml = renderToStaticMarkup(
      React.createElement(
        ClinicalI18nProvider,
        null,
        React.createElement(ClinicalStaffPortalShell, {
          apiClient,
          currentUser: ADMIN_USER,
          onLogout: () => {},
          initialTab: 'patients',
          initialPatients: allPatients,
        }),
      ),
    );

    expect(allViewHtml).toContain('Poom');
    expect(allViewHtml).toContain('HN-410382');
    expect(allViewHtml).toContain('HN-990001');
    expect(allViewHtml).toContain('ซิงก์ข้อมูลจาก Supabase: 2 บัญชี');
    expect(allViewHtml).toContain('รีเฟรชข้อมูล');
    expect(allViewHtml).toContain('คนไข้ทั้งหมดในคลินิก (All)');
    expect(allViewHtml).toContain('คนไข้ในความดูแลของฉัน (Assigned)');
    expect(allViewHtml).toContain('data-testid="admin-supabase-sync-pill"');
    expect(allViewHtml).toContain('data-testid="admin-refresh-patients-btn"');
    expect(allViewHtml).toContain('data-testid="patient-scope-toggle"');
    expect(EMOJI_RE.test(allViewHtml)).toBe(false);

    // 3. Render with Assigned scope as admin: filters table to admin's assigned patient while keeping total sync count = 2
    const assignedViewHtml = renderToStaticMarkup(
      React.createElement(
        ClinicalI18nProvider,
        null,
        React.createElement(ClinicalStaffPortalShell, {
          apiClient,
          currentUser: ADMIN_USER,
          onLogout: () => {},
          initialTab: 'patients',
          initialPatients: allPatients,
          initialPatientScope: 'assigned',
        }),
      ),
    );
    expect(assignedViewHtml).toContain('HN-990001');
    expect(assignedViewHtml).not.toContain('HN-410382');
    expect(assignedViewHtml).toContain('ซิงก์ข้อมูลจาก Supabase: 2 บัญชี');
    expect(EMOJI_RE.test(assignedViewHtml)).toBe(false);

    // 4. Render with Supabase RLS error banner visible
    const errorViewHtml = renderToStaticMarkup(
      React.createElement(
        ClinicalI18nProvider,
        null,
        React.createElement(ClinicalStaffPortalShell, {
          apiClient,
          currentUser: ADMIN_USER,
          onLogout: () => {},
          initialTab: 'patients',
          initialPatients: [],
          initialPatientSyncError: 'สิทธิ์การเข้าถึง Supabase (RLS 42501): permission denied for table patients',
        }),
      ),
    );
    expect(errorViewHtml).toContain('data-testid="patient-sync-error-banner"');
    expect(errorViewHtml).toContain('42501');
    expect(errorViewHtml).toContain('permission denied for table patients');
    expect(EMOJI_RE.test(errorViewHtml)).toBe(false);
  });

  it('supports GET /api/patients?scope=all on Worker router for clinical staff and hydrates directly when cold cache is empty', async () => {
    const poomRow = makePoomRawSupabaseRow();
    const origFetch = globalThis.fetch;
    try {
      globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url;
        const method = init?.method ?? (input instanceof Request ? input.method : 'GET');
        if (method === 'GET' && url.includes('/rest/v1/staff_users')) {
          return new Response(
            JSON.stringify([
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
                id: '00000000-0000-4000-8000-000000000002',
                username: 'pt_somchai',
                password_hash:
                  'pbkdf2:61d9355274f9c2334e628def69572f75:f9129631c2636dd24bccf0efbb017453d4c36b0a0d787c7c4a6886a7c0c91820',
                full_name: 'กภ.สมชาย ใจดี (PT)',
                phone: '081-111-2233',
                role: 'pt',
                active: true,
              },
            ]),
            { status: 200, headers: { 'Content-Type': 'application/json' } },
          );
        }
        if (method === 'GET' && url.includes('/rest/v1/patients')) {
          return new Response(JSON.stringify([poomRow]), {
            status: 200,
            headers: { 'Content-Type': 'application/json' },
          });
        }
        return new Response(JSON.stringify([]), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }) as typeof fetch;

      const fakeJwt =
        'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJyb2xlIjoic2VydmljZV9yb2xlIn0.signaturevaltestsafe7788990011';
      const router = createCloudDelegatedRouter({
        supabaseUrl: 'https://portal-sync-test.supabase.co',
        supabaseServiceKey: fakeJwt,
      });

      const loginPt = await router(
        new Request('https://drtechcares.workers.dev/api/auth/login', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ username: 'pt_somchai', password: 'Pt@2026' }),
        }),
      );
      expect(loginPt.status).toBe(200);
      const { token: ptToken } = (await loginPt.json()) as { token: string };

      const allRes = await router(
        new Request('https://drtechcares.workers.dev/api/patients?scope=all&refresh=1', {
          method: 'GET',
          headers: { Authorization: `Bearer ${ptToken}` },
        }),
      );
      expect(allRes.status).toBe(200);
      const allJson = (await allRes.json()) as { patients: PatientRecord[] };
      expect(allJson.patients.some((p) => p.hn === 'HN-410382' && p.fullName === 'Poom')).toBe(true);

      const loginAdmin = await router(
        new Request('https://drtechcares.workers.dev/api/auth/login', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ username: 'admin', password: 'Admin@2026' }),
        }),
      );
      expect(loginAdmin.status).toBe(200);
      const { token: adminToken } = (await loginAdmin.json()) as { token: string };
      const adminPatientsRes = await router(
        new Request('https://drtechcares.workers.dev/api/patients?scope=all&refresh=1', {
          method: 'GET',
          headers: { Authorization: `Bearer ${adminToken}` },
        }),
      );
      expect(adminPatientsRes.status).toBe(200);
      const adminPatientsJson = (await adminPatientsRes.json()) as { patients: PatientRecord[] };
      expect(adminPatientsJson.patients.some((p) => p.hn === 'HN-410382' && p.fullName === 'Poom')).toBe(true);
    } finally {
      globalThis.fetch = origFetch;
    }
  });

  it('audits all Admin Portal patient sync modified files for zero Unicode emojis', () => {
    const root = process.cwd();
    const files = [
      'src/components/ClinicalStaffPortalShell.tsx',
      'src/lib/patientHydration.ts',
      'packages/api-client/src/cloudRouter.ts',
      'packages/api-client/src/index.ts',
      'src/__tests__/adminPortalPatientSync.test.ts',
    ];
    for (const rel of files) {
      const content = fs.readFileSync(path.join(root, rel), 'utf8');
      expect(EMOJI_RE.test(content), `Unexpected Unicode emoji in ${rel}`).toBe(false);
    }
  });
});

