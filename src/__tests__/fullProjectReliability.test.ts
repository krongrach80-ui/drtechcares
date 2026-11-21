import fs from 'node:fs';
import path from 'node:path';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { createCloudDelegatedRouter } from '../../packages/api-client/src/index';
import { resetSupabaseCircuitForTests } from '../../packages/api-client/src/supabasePersistence';
import {
  createPatientAssessment,
  listPatientAssessments,
  resetClinicalAssessmentStoreForTests,
} from '../lib/clinicalAssessmentStore';
import { resolvePatientUniversalLinkCode } from '../types/clinicalAssessment';

const EMOJI_RE = /[\p{Extended_Pictographic}\u{FE0F}\u{2600}-\u{27BF}\u{23E9}-\u{23FA}]/u;

function readRepoFile(relPath: string): string {
  return fs.readFileSync(path.resolve(process.cwd(), relPath), 'utf8');
}

async function callRouter(
  router: (req: Request) => Promise<Response>,
  routePath: string,
  init: { method?: string; token?: string; deviceId?: string; body?: unknown } = {},
): Promise<Response> {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (init.token) headers.Authorization = `Bearer ${init.token}`;
  if (init.deviceId) headers['x-device-id'] = init.deviceId;
  const req = new Request(`https://drtechcares.workers.dev${routePath}`, {
    method: init.method ?? 'GET',
    headers,
    body: init.body !== undefined ? JSON.stringify(init.body) : undefined,
  });
  return router(req);
}

describe('Full Project Reliability, API Parity & 60FPS Smoothness Regressions', () => {
  beforeEach(() => {
    resetClinicalAssessmentStoreForTests();
    resetSupabaseCircuitForTests();
  });

  afterEach(() => {
    vi.restoreAllMocks();
    resetSupabaseCircuitForTests();
  });

  it('does not trip the Supabase circuit breaker on /auth/v1/ 401 or invalid stf.* tokens', async () => {
    const authFetchCalls: string[] = [];
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (input) => {
      const url = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url;
      if (url.includes('/auth/v1/')) {
        authFetchCalls.push(url);
        return new Response(JSON.stringify({ message: 'Invalid JWT' }), {
          status: 401,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      if (url.includes('/rest/v1/')) {
        return new Response(JSON.stringify([]), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      return new Response(JSON.stringify({}), { status: 200 });
    });

    const router = createCloudDelegatedRouter({
      supabaseUrl: 'https://example-project.supabase.co',
      supabaseServiceKey: 'sb_secret_test_key_valid_len_12345',
      supabaseAnonKey: 'sb_publishable_test_anon_key_12345',
    });

    // 1) Invalid stf.* token must NOT query /auth/v1/ and returns 401
    const stfRes = await callRouter(router, '/api/auth/me', {
      token: 'stf.invalidpayload.invalidsig',
    });
    expect(stfRes.status).toBe(401);
    expect(authFetchCalls).toHaveLength(0);

    // 2) Invalid 3-segment Supabase JWT triggers /auth/v1/ 401 without tripping circuit breaker
    const jwtRes = await callRouter(router, '/api/auth/me', {
      token: 'eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ4In0.invalidsig',
    });
    expect(jwtRes.status).toBe(401);
    expect(authFetchCalls.length).toBeGreaterThanOrEqual(1);

    // Circuit remains healthy (supabaseConnected must remain true)
    const healthRes = await callRouter(router, '/api/health');
    expect(healthRes.status).toBe(200);
    const healthJson = (await healthRes.json()) as { ok: boolean; supabaseConnected?: boolean };
    expect(healthJson.ok).toBe(true);
    expect(healthJson.supabaseConnected).toBe(true);
  });

  it('supports public POST /api/kiosk/pin-login without staff auth while isolating PHI', async () => {
    const router = createCloudDelegatedRouter({});
    const loginRes = await callRouter(router, '/api/auth/login', {
      method: 'POST',
      body: { username: 'admin', password: 'Admin@2026' },
    });
    expect(loginRes.status).toBe(200);
    const { token: adminToken } = (await loginRes.json()) as { token: string };

    // Create an active patient via admin
    const createRes = await callRouter(router, '/api/patients', {
      method: 'POST',
      token: adminToken,
      body: {
        fullName: 'คุณสมชาย รักษาใจ',
        hn: 'HN-2026-8842',
        age: 68,
        gender: 'ชาย',
        phone: '0891238842',
        diagnosis: 'Stroke left hemiparesis',
        strokeType: 'ISCHEMIC_LEFT',
        affectedSide: 'LEFT',
      },
    });
    expect([200, 201]).toContain(createRes.status);
    const { patient: createdPatient } = (await createRes.json()) as {
      patient: {
        id: string;
        hn: string;
        universalHealthLinkCode?: string;
      };
    };
    expect(createdPatient.id).toBeTruthy();

    // Unauthenticated kiosk PIN login using last 4 phone digits
    const pinOk = await callRouter(router, '/api/kiosk/pin-login', {
      method: 'POST',
      body: { pin: '8842' },
    });
    expect(pinOk.status).toBe(200);
    const pinJson = (await pinOk.json()) as {
      matched: boolean;
      patientId: string;
      patient: Record<string, unknown>;
    };
    expect(pinJson.matched).toBe(true);
    expect(pinJson.patientId).toBe(createdPatient.id);
    expect(pinJson.patient.hn).toBe('HN-2026-8842');
    // Must not expose sensitive PHI like phone or passwordHash
    expect(pinJson.patient.phone).toBeUndefined();
    expect(pinJson.patient.passwordHash).toBeUndefined();

    // Non-matching PIN returns 404
    const pinMiss = await callRouter(router, '/api/kiosk/pin-login', {
      method: 'POST',
      body: { pin: '9999' },
    });
    expect(pinMiss.status).toBe(404);
  });

  it('persists revoked=true on POST /api/auth/logout and rejects subsequent use of the revoked token', async () => {
    const upsertedSessions: Record<string, unknown>[] = [];
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (input, init) => {
      const url = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url;
      const method = (init?.method ?? 'GET').toUpperCase();
      if (url.includes('/rest/v1/active_sessions')) {
        if (method === 'POST' || method === 'PATCH') {
          if (typeof init?.body === 'string') {
            upsertedSessions.push(JSON.parse(init.body) as Record<string, unknown>);
          }
          return new Response(JSON.stringify([]), { status: 201, headers: { 'Content-Type': 'application/json' } });
        }
        return new Response(JSON.stringify(upsertedSessions), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      if (url.includes('/rest/v1/')) {
        return new Response(JSON.stringify([]), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      return new Response(JSON.stringify({}), { status: 200 });
    });

    const router = createCloudDelegatedRouter({});
    const loginRes = await callRouter(router, '/api/auth/login', {
      method: 'POST',
      body: { username: 'admin', password: 'Admin@2026' },
    });
    expect(loginRes.status).toBe(200);
    const { token } = (await loginRes.json()) as { token: string };

    const logoutRes = await callRouter(router, '/api/auth/logout', {
      method: 'POST',
      token,
    });
    expect(logoutRes.status).toBe(200);

    // Revoked token must fail on /api/auth/me
    const afterLogoutMe = await callRouter(router, '/api/auth/me', { token });
    expect(afterLogoutMe.status).toBe(401);

    // Verify source persistence uses upsert with revoked: true & onConflict: 'token'
    const routerSrc = readRepoFile('packages/api-client/src/cloudRouter.ts');
    expect(routerSrc).toContain("db.from('active_sessions').upsert(");
    expect(routerSrc).toContain('revoked: true');
    expect(routerSrc).toContain("onConflict: 'token'");
    const persistSrc = readRepoFile('packages/api-client/src/supabasePersistence.ts');
    expect(persistSrc).not.toMatch(/\.from\('active_sessions'\)[\s\S]{0,200}\.eq\('revoked',\s*false\)/);
  });

  it('aligns universalHealthLinkCode prefix (.slice(0, 8)) across client, router, and persistence', async () => {
    const patient = { id: 'pat-alpha998877', hn: 'HN-2026-0099' };
    const expectedCode = resolvePatientUniversalLinkCode(patient);
    expect(expectedCode).toBe('DTC-EHR-PATALPHA');

    const router = createCloudDelegatedRouter({});
    const loginRes = await callRouter(router, '/api/auth/login', {
      method: 'POST',
      body: { username: 'admin', password: 'Admin@2026' },
    });
    const { token } = (await loginRes.json()) as { token: string };
    const createRes = await callRouter(router, '/api/patients', {
      method: 'POST',
      token,
      body: {
        fullName: 'คุณทดสอบ ลิงก์รหัส',
        hn: 'HN-2026-0099',
        age: 70,
        gender: 'ชาย',
        phone: '0811112233',
      },
    });
    const { patient: created } = (await createRes.json()) as {
      patient: { id: string; universalHealthLinkCode?: string };
    };
    expect(created.universalHealthLinkCode).toBeTruthy();
    expect(created.universalHealthLinkCode).toMatch(/^DTC-EHR-[A-Z0-9]{8}$/);
    const derivedFromCreatedId = resolvePatientUniversalLinkCode({ id: created.id, hn: 'HN-2026-0099' });
    expect(created.universalHealthLinkCode).toBe(derivedFromCreatedId);

    const cloudRouterSrc = readRepoFile('packages/api-client/src/cloudRouter.ts');
    const supabasePersistSrc = readRepoFile('packages/api-client/src/supabasePersistence.ts');
    expect(cloudRouterSrc).toContain('cleaned.slice(0, 8)');
    expect(supabasePersistSrc).toContain('.slice(0, 8)');
  });

  it('does not inject fake demo assessments when records already exist even with seedDemoIfEmpty=true', async () => {
    const patient = { id: 'pat-real-001', hn: 'HN-2026-1122', fullName: 'คุณสมหมาย' };
    await createPatientAssessment({
      patient_id: patient.id,
      patient_hn: patient.hn,
      universal_link_code: 'DTC-EHR-PATREAL0',
      facility_name: 'ศูนย์กายภาพบำบัด Dr.Tech.Care สาขาหลัก',
      assessor_staff_id: 'pt-1',
      assessor_staff_name: 'กภ. จริง',
      exercise_code: 'STROKE_SHOULDER_FLEXION',
      exercise_name_th: 'ยกแขนตรงไปด้านหน้า',
      target_side: 'RIGHT',
      initial_score_0_to_6: 4,
      measured_rom_deg: 95,
      target_rom_deg: 125,
      chief_complaint: 'ตรวจติดตามจริง',
      clinical_handover_note: 'บันทึกรายการจริง',
      next_session_goal: '110 องศา',
    });

    const list = await listPatientAssessments(patient, {
      includeCrossFacility: true,
      seedDemoIfEmpty: true,
    });
    expect(list).toHaveLength(1);
    expect(list[0].assessor_staff_name).toBe('กภ. จริง');

    // Also verify store places seeding check AFTER remote fetch
    const storeSrc = readRepoFile('src/lib/clinicalAssessmentStore.ts');
    const fetchIdx = storeSrc.indexOf('fetchWithTimeout');
    const seedCallIdx = storeSrc.indexOf('ensureDemoAssessmentsForPatient(patient);');
    expect(fetchIdx).toBeGreaterThan(-1);
    expect(seedCallIdx).toBeGreaterThan(fetchIdx);
  });

  it('enforces 60FPS rendering guardrails, ErrorBoundary root mount, singleton detectors, and zero emojis', () => {
    const ownerLockSrc = readRepoFile('src/lib/ownerBiometricLock.ts');
    const edgeSocketSrc = readRepoFile('src/hooks/useEdgeVisionSocket.ts');
    const kioskLoginSrc = readRepoFile('src/components/KioskBiometricLoginView.tsx');
    const mainSrc = readRepoFile('src/main.tsx');
    const portalSubModalsSrc = readRepoFile('src/components/PortalSubModals.tsx');
    const faceId120Src = readRepoFile('src/components/FaceID120Scanner.tsx');
    const clinicalShellSrc = readRepoFile('src/components/ClinicalStaffPortalShell.tsx');

    // No active GPU-stalling shadowBlur > 0
    expect(ownerLockSrc).not.toMatch(/shadowBlur\s*=\s*[1-9]/);
    expect(edgeSocketSrc).not.toMatch(/shadowBlur\s*=\s*[1-9]/);
    expect(kioskLoginSrc).not.toMatch(/shadowBlur\s*=\s*[1-9]/);

    // Root ErrorBoundary wrapper
    expect(mainSrc).toContain('<ClinicalErrorBoundary>');
    expect(mainSrc).toContain('<App />');

    // Shared singleton detector in PortalSubModals & runtime FaceLandmarker in FaceID120Scanner
    expect(portalSubModalsSrc).toContain('getCachedPrimaryFaceLandmarker');
    expect(portalSubModalsSrc).not.toContain('landmarker?.close()');
    expect(faceId120Src).toMatch(/import\s*\{\s*FaceLandmarker\s*\}\s*from\s*'@mediapipe\/tasks-vision'/);
    expect(faceId120Src).not.toContain('backdrop-blur-xs');
    expect(faceId120Src).not.toContain('id="emeraldGlow"');

    // ClinicalStaffPortalShell live modal lookup and local ISO date helper
    expect(clinicalShellSrc).toContain('formatLocalIsoDate(');
    expect(clinicalShellSrc).not.toContain('new Date().toISOString().slice(0, 10)');

    // Strict zero Unicode emojis check across touched files
    const touchedFiles = [
      'src/main.tsx',
      'src/components/ClinicalErrorBoundary.tsx',
      'src/components/FaceID120Scanner.tsx',
      'src/components/PatientClinicalRecordPanel.tsx',
      'src/components/PortalSubModals.tsx',
      'src/components/ClinicalStaffPortalShell.tsx',
      'src/components/StaffAssistedEnrollmentModal.tsx',
      'src/components/KioskBiometricLoginView.tsx',
      'src/PatientPersonalPortal.tsx',
      'src/lib/clinicalAssessmentStore.ts',
      'src/lib/ownerBiometricLock.ts',
      'src/lib/biometricVisionCache.ts',
      'src/hooks/useEdgeVisionSocket.ts',
      'src/hooks/useKioskSocket.ts',
      'apps/kiosk/src/PhysicalCardGame.tsx',
      'apps/kiosk/src/SmartLoginView.tsx',
      'apps/kiosk/src/KioskPortalApp.tsx',
      'packages/api-client/src/cloudRouter.ts',
      'packages/api-client/src/supabasePersistence.ts',
      'packages/api-client/src/index.ts',
      'apps/api/src/app.ts',
      'scripts/worker.js',
    ];
    for (const rel of touchedFiles) {
      const text = readRepoFile(rel);
      expect(EMOJI_RE.test(text), `Unexpected emoji found in ${rel}`).toBe(false);
    }
  });
});

