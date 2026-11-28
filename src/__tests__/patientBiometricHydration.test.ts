import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import type { SupabaseClient } from '@supabase/supabase-js';
import { l2Normalize } from '@drtechcares/smart-biometric-core';
import { createCloudDelegatedRouter } from '@drtechcares/api-client';
import {
  hydrateBiometricsLiveFromSupabase,
  hydrateStateFromSupabase,
  resetHydrationCacheForTests,
  resetSupabaseCircuitForTests,
  resetUnsupportedColumnsForTests,
  type CloudRouterState,
} from '../../packages/api-client/src/supabasePersistence';
import {
  buildPatientDualWriteColumns,
  inspectPatientRecordHydration,
  normalizePatientRecord,
  normalizePatientRecordList,
  resolveProtectedBiometricStatus,
} from '../lib/patientHydration';

const EMOJI_RE = /[\u{1F300}-\u{1FAFF}\u{2600}-\u{27BF}]/u;

function makeRelu6Vector(seed: number, dim = 1024): number[] {
  const raw = new Array<number>(dim);
  for (let i = 0; i < dim; i += 1) {
    const v = Math.sin((i + 1) * seed * 0.17) + 0.55 * Math.cos((i + 5) * seed * 0.29) + 0.35;
    raw[i] = v > 0 ? v : 0;
  }
  return l2Normalize(raw);
}

function buildPoomStaleTopLevelRow() {
  const master = makeRelu6Vector(4.10382);
  const center = makeRelu6Vector(4.10383);
  const left = makeRelu6Vector(4.10384);
  const right = makeRelu6Vector(4.10385);
  const up = makeRelu6Vector(4.10386);
  const down = makeRelu6Vector(4.10387);
  const faceImageUrl =
    'https://vyjuwjhbdhknjdzyqctz.supabase.co/storage/v1/object/public/patient-faces/5104db15-5634-45e5-ab65-f2f008a15e54/center.jpg';

  return {
    id: '5104db15-5634-45e5-ab65-f2f008a15e54',
    hn: 'HN-410382',
    full_name: 'Poom',
    age: 26,
    phone: '0812345678',
    diagnosis: 'Elderly Upper Limb Rehabilitation',
    registration_status: 'active',
    assigned_pt_id: '00000000-0000-4000-8000-000000000002',
    delegated_to_pt_id: null,
    delegation_note: null,
    next_appointment_date: '2026-11-24',
    created_at: '2026-11-23T02:04:11.925523+00:00',
    updated_at: '2026-11-23T02:04:11.925523+00:00',
    // Stale / empty top-level columns (exact production bug snapshot)
    face_biometric_status: 'PENDING',
    face_frames_count: 0,
    face_image_url: null,
    master_embedding: null,
    angle_embeddings: null,
    face_embedding: null,
    biometric_data: null,
    biometric_updated_at: null,
    gender: null,
    care_group: null,
    affected_side: null,
    motor_assessment_score: null,
    adl_dressing_score: null,
    // Complete verified data stored inside history JSONB
    history: {
      faceBiometricStatus: 'VERIFIED',
      faceFramesCount: 120,
      faceImageUrl,
      angleImageUrls: {
        center: faceImageUrl,
        left: faceImageUrl.replace('center.jpg', 'left.jpg'),
        right: faceImageUrl.replace('center.jpg', 'right.jpg'),
        up: faceImageUrl.replace('center.jpg', 'up.jpg'),
        down: faceImageUrl.replace('center.jpg', 'down.jpg'),
      },
      masterEmbedding: master,
      angleEmbeddings: { center, left, right, up, down },
      exemplarEmbeddings: [center, left, right],
      biometricEngine: 'human-faceres-1024d',
      geometricRatios: {
        eyeToNoseRatio: 3.9573,
        faceWidthToUpperHeight: 2.7967,
        mouthToEyeRatio: 0.5657,
        noseToChinRatio: 1.1254,
      },
      gender: 'ชาย',
      careGroup: 'MOBILITY_RECOVERY',
      affectedSide: 'RIGHT',
      motorAssessmentScore: 3,
      adlDressingScore: 3,
      maxSafeRomDeg: 140,
      universalHealthLinkCode: 'DTC-EHR-5104DB15',
      rehabSchedules: [],
    },
  };
}

function makeEmptyCloudState(): CloudRouterState {
  return {
    users: [],
    patients: [],
    face_templates: [],
    medical_notes: [],
    patient_courses: [],
    rehab_sessions: [],
    treatment_reports: [],
    exercises: [],
    modpacks: [],
    staff_calls: [],
    audit_logs: [],
    banned_targets: [],
    active_sessions: [],
    system_settings: {
      aiFaceScanEnabled: true,
      aiFaceThreshold: 0.84,
      aiPoseDetectionEnabled: false,
      kioskAutoResetSec: 6,
    },
    kiosks: [],
  };
}

describe('Patient Biometric Persistence & Hydration (Poom HN-410382)', () => {
  it('hydrates Poom (HN-410382) from history JSON as VERIFIED with embeddings, image URL, and profile fields when top-level columns are PENDING/empty', () => {
    const poomRow = buildPoomStaleTopLevelRow();
    const inspected = inspectPatientRecordHydration(poomRow);
    const hydrated = normalizePatientRecord(poomRow);

    expect(hydrated.id).toBe('5104db15-5634-45e5-ab65-f2f008a15e54');
    expect(hydrated.hn).toBe('HN-410382');
    expect(hydrated.fullName).toBe('Poom');
    expect(hydrated.faceBiometricStatus).toBe('VERIFIED');
    expect(hydrated.biometricStatusDetail).toBe('VERIFIED');
    expect(hydrated.faceFramesCount).toBe(120);
    expect(hydrated.faceImageUrl).toBe(poomRow.history.faceImageUrl);
    expect(hydrated.angleImageUrls?.center).toBe(poomRow.history.faceImageUrl);
    expect(hydrated.angleImageUrls?.left).toContain('left.jpg');
    expect(hydrated.masterEmbedding).toHaveLength(1024);
    expect(hydrated.angleEmbeddings?.center).toHaveLength(1024);
    expect(hydrated.angleEmbeddings?.left).toHaveLength(1024);
    expect(hydrated.angleEmbeddings?.right).toHaveLength(1024);
    expect(hydrated.angleEmbeddings?.up).toHaveLength(1024);
    expect(hydrated.angleEmbeddings?.down).toHaveLength(1024);
    expect(hydrated.exemplarEmbeddings).toHaveLength(3);
    expect(hydrated.geometricRatios?.eyeToNoseRatio).toBeCloseTo(3.9573, 4);
    expect(hydrated.biometricUpdatedAt).toBe('2026-11-23T02:04:11.925523+00:00');
    expect(hydrated.gender).toBe('ชาย');
    expect(hydrated.careGroup).toBe('MOBILITY_RECOVERY');
    expect(hydrated.affectedSide).toBe('RIGHT');
    expect(hydrated.motorAssessmentScore).toBe(3);
    expect(hydrated.adlDressingScore).toBe(3);

    // Must flag stale top-level columns and build a complete repair patch
    expect(inspected.needsTopLevelRepair).toBe(true);
    expect(inspected.topLevelRepairPatch).not.toBeNull();
    expect(inspected.topLevelRepairPatch?.face_biometric_status).toBe('VERIFIED');
    expect(inspected.topLevelRepairPatch?.face_frames_count).toBe(120);
    expect(inspected.topLevelRepairPatch?.face_image_url).toBe(poomRow.history.faceImageUrl);
    expect(Array.isArray(inspected.topLevelRepairPatch?.master_embedding)).toBe(true);
    expect((inspected.topLevelRepairPatch?.master_embedding as number[]).length).toBe(1024);
    expect(inspected.topLevelRepairPatch?.angle_embeddings).toBeTruthy();
    expect(inspected.topLevelRepairPatch?.biometric_data).toBeTruthy();
    expect(inspected.topLevelRepairPatch?.biometric_updated_at).toBe('2026-11-23T02:04:11.925523+00:00');
    expect(inspected.topLevelRepairPatch?.gender).toBe('ชาย');
    expect(inspected.topLevelRepairPatch?.care_group).toBe('MOBILITY_RECOVERY');
    expect(inspected.topLevelRepairPatch?.affected_side).toBe('RIGHT');
    expect(inspected.topLevelRepairPatch?.motor_assessment_score).toBe(3);
    expect(inspected.topLevelRepairPatch?.adl_dressing_score).toBe(3);
  });

  it('hydrates Poom during live/full Supabase hydration even when patient_biometrics is empty and persists repaired top-level columns in the background', async () => {
    resetUnsupportedColumnsForTests();
    resetSupabaseCircuitForTests();
    const poomRow = buildPoomStaleTopLevelRow();
    const repairedPatches: Array<{ id: string; patch: Record<string, unknown> }> = [];

    const makeMockSb = () =>
      ({
        from: (table: string) => {
          const q: Record<string, unknown> = {};
          let currentSelect = '';
          q.select = (cols: string) => {
            currentSelect = cols;
            return q;
          };
          q.eq = () => q;
          q.order = () => q;
          q.limit = () => q;
          q.maybeSingle = () => Promise.resolve({ data: null, error: null });
          q.in = (_col: string, ids: string[]) => {
            if (table === 'patients' && ids.includes(poomRow.id)) {
              return Promise.resolve({ data: [poomRow], error: null });
            }
            return Promise.resolve({ data: [], error: null });
          };
          q.update = (patch: Record<string, unknown>) => ({
            eq: async (_col: string, val: string) => {
              if (table === 'patients') {
                repairedPatches.push({ id: val, patch });
              }
              return { error: null };
            },
          });
          q.then = (resolve: (val: unknown) => void) => {
            if (table === 'patients') {
              // Simulate projected live query vs full base query
              if (currentSelect.includes('historyFaceStatus:history->>faceBiometricStatus')) {
                resolve({
                  data: [
                    {
                      id: poomRow.id,
                      hn: poomRow.hn,
                      full_name: poomRow.full_name,
                      age: poomRow.age,
                      phone: poomRow.phone,
                      diagnosis: poomRow.diagnosis,
                      registration_status: poomRow.registration_status,
                      assigned_pt_id: poomRow.assigned_pt_id,
                      next_appointment_date: poomRow.next_appointment_date,
                      face_biometric_status: 'PENDING',
                      face_frames_count: 0,
                      face_image_url: null,
                      biometric_updated_at: null,
                      created_at: poomRow.created_at,
                      updated_at: poomRow.updated_at,
                      historyFaceStatus: 'VERIFIED',
                      historyFaceFramesCount: 120,
                      historyFaceImageUrl: poomRow.history.faceImageUrl,
                      historyGender: 'ชาย',
                      geometricRatios: poomRow.history.geometricRatios,
                      rehabSchedules: [],
                      biometricEngine: 'human-faceres-1024d',
                      careGroup: 'MOBILITY_RECOVERY',
                      affectedSide: 'RIGHT',
                      motorAssessmentScore: 3,
                      adlDressingScore: 3,
                    },
                  ],
                  error: null,
                });
                return;
              }
              resolve({ data: [poomRow], error: null });
              return;
            }
            // Intentionally return empty patient_biometrics to prove history JSON fallback works
            resolve({ data: [], error: null });
          };
          return q;
        },
      }) as unknown as SupabaseClient;

    // 1. Live biometric hydration recovers vectors from history JSON + schedules top-level repair
    const liveSb = makeMockSb();
    const liveState = makeEmptyCloudState();
    const liveRes = await hydrateBiometricsLiveFromSupabase(liveSb, liveState, 2000);
    expect(liveRes.ok).toBe(true);
    expect(liveState.patients).toHaveLength(1);
    expect(liveState.patients[0]?.faceBiometricStatus).toBe('VERIFIED');
    expect(liveState.patients[0]?.faceFramesCount).toBe(120);
    expect(liveState.patients[0]?.faceImageUrl).toBe(poomRow.history.faceImageUrl);
    expect(liveState.patients[0]?.masterEmbedding).toHaveLength(1024);
    expect(liveState.patients[0]?.angleEmbeddings?.center).toHaveLength(1024);
    expect(liveState.patients[0]?.gender).toBe('ชาย');

    // Flush microtasks for background repair
    await new Promise((r) => setTimeout(r, 10));
    expect(repairedPatches.length).toBeGreaterThanOrEqual(1);
    expect(repairedPatches[0]?.id).toBe(poomRow.id);
    expect(repairedPatches[0]?.patch.face_biometric_status).toBe('VERIFIED');
    expect(repairedPatches[0]?.patch.face_frames_count).toBe(120);
    expect(repairedPatches[0]?.patch.face_image_url).toBe(poomRow.history.faceImageUrl);
    expect(repairedPatches[0]?.patch.master_embedding).toHaveLength(1024);
    expect(repairedPatches[0]?.patch.angle_embeddings).toBeTruthy();
    expect(repairedPatches[0]?.patch.biometric_data).toBeTruthy();

    // 2. Full state hydration + normalizePatientRecordList also repair top-level columns cleanly
    repairedPatches.length = 0;
    const fullSb = makeMockSb();
    const fullState = makeEmptyCloudState();
    resetHydrationCacheForTests(fullState);
    await hydrateStateFromSupabase(fullSb, fullState, { force: true });
    expect(fullState.patients[0]?.faceBiometricStatus).toBe('VERIFIED');
    expect(fullState.patients[0]?.faceFramesCount).toBe(120);
    expect(fullState.patients[0]?.masterEmbedding).toHaveLength(1024);
    await new Promise((r) => setTimeout(r, 10));
    expect(repairedPatches.length).toBe(1);

    // 3. Frontend list normalizer also repairs when given a connected supabase client
    repairedPatches.length = 0;
    const listSb = makeMockSb();
    const normalizedList = normalizePatientRecordList([poomRow], { supabase: listSb });
    expect(normalizedList[0]?.faceBiometricStatus).toBe('VERIFIED');
    await new Promise((r) => setTimeout(r, 10));
    expect(repairedPatches.length).toBe(1);
  });

  it('dual-writes top-level biometric and profile columns on enrollment and blocks stale PENDING overwrite on PUT /api/patients/:id', async () => {
    resetUnsupportedColumnsForTests();
    resetSupabaseCircuitForTests();

    const master = makeRelu6Vector(9.12);
    const angles = {
      center: makeRelu6Vector(9.13),
      left: makeRelu6Vector(9.14),
      right: makeRelu6Vector(9.15),
      up: makeRelu6Vector(9.16),
      down: makeRelu6Vector(9.17),
    };
    const writtenPatientPayloads: Array<{ method: string; body: Record<string, unknown> }> = [];
    const origFetch = globalThis.fetch;

    try {
      globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url;
        const method = init?.method ?? (input instanceof Request ? input.method : 'GET');
        if (url.includes('/rest/v1/patients') && (method === 'POST' || method === 'PATCH')) {
          const parsed = JSON.parse(String(init?.body ?? '{}')) as Record<string, unknown>;
          const row = Array.isArray(parsed) ? (parsed[0] as Record<string, unknown>) : parsed;
          // Simulate production schema where max_safe_rom_deg and universal_health_link_code do not exist
          for (const missing of ['max_safe_rom_deg', 'universal_health_link_code']) {
            if (missing in row) {
              return new Response(
                JSON.stringify({
                  code: 'PGRST204',
                  message: `Could not find the '${missing}' column of 'patients' in the schema cache`,
                }),
                { status: 400, headers: { 'Content-Type': 'application/json' } },
              );
            }
          }
          writtenPatientPayloads.push({ method, body: row });
          return new Response(JSON.stringify([row]), {
            status: method === 'POST' ? 201 : 200,
            headers: { 'Content-Type': 'application/json' },
          });
        }
        return new Response(JSON.stringify([]), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }) as typeof fetch;

      const fakeJwt =
        'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJyb2xlIjoic2VydmljZV9yb2xlIn0.signaturevaltestsafe1122334455';
      const router = createCloudDelegatedRouter({
        supabaseUrl: 'https://dual-write-test.supabase.co',
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

      const faceUrl = 'https://vyjuwjhbdhknjdzyqctz.supabase.co/storage/v1/object/public/patient-faces/poom/center.jpg';
      const enrollRes = await router(
        new Request('https://drtechcares.workers.dev/api/biometric/enroll', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${token}`,
          },
          body: JSON.stringify({
            hn: 'HN-410382',
            fullName: 'Poom',
            age: 26,
            gender: 'ชาย',
            diagnosis: 'Elderly Rehab',
            careGroup: 'MOBILITY_RECOVERY',
            affectedSide: 'RIGHT',
            motorAssessmentScore: 3,
            adlDressingScore: 3,
            faceImageUrl: faceUrl,
            masterEmbedding: master,
            frontalCentroid: angles.center,
            overallCentroid: master,
            angleEmbeddings: angles,
            biometricEngine: 'human-faceres-1024d',
            framesCaptured: 120,
            qualityScore: 0.94,
          }),
        }),
      );
      expect(enrollRes.status).toBe(201);
      const { patient: enrolled } = (await enrollRes.json()) as { patient: { id: string; faceBiometricStatus: string } };
      expect(enrolled.faceBiometricStatus).toBe('VERIFIED');

      const insertPayload = writtenPatientPayloads.find((w) => w.method === 'POST')?.body;
      expect(insertPayload).toBeDefined();
      expect(insertPayload?.face_biometric_status).toBe('VERIFIED');
      expect(insertPayload?.face_frames_count).toBe(120);
      expect(insertPayload?.face_image_url).toBe(faceUrl);
      expect(Array.isArray(insertPayload?.master_embedding)).toBe(true);
      expect((insertPayload?.master_embedding as number[]).length).toBe(1024);
      expect(insertPayload?.angle_embeddings).toBeTruthy();
      expect(insertPayload?.biometric_data).toBeTruthy();
      expect(insertPayload?.biometric_updated_at).toBeTruthy();
      expect(insertPayload?.gender).toBe('ชาย');
      expect(insertPayload?.care_group).toBe('MOBILITY_RECOVERY');
      expect(insertPayload?.affected_side).toBe('RIGHT');
      expect(insertPayload?.motor_assessment_score).toBe(3);
      expect(insertPayload?.adl_dressing_score).toBe(3);

      // Now send a stale PUT /api/patients/:id attempting to overwrite faceBiometricStatus with 'PENDING'
      const updateRes = await router(
        new Request(`https://drtechcares.workers.dev/api/patients/${enrolled.id}`, {
          method: 'PUT',
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${token}`,
          },
          body: JSON.stringify({
            phone: '0899998888',
            faceBiometricStatus: 'PENDING',
          }),
        }),
      );
      expect(updateRes.status).toBe(200);
      const { patient: updated } = (await updateRes.json()) as { patient: { faceBiometricStatus: string } };
      expect(updated.faceBiometricStatus).toBe('VERIFIED');

      const patchPayload = writtenPatientPayloads.find((w) => w.method === 'PATCH')?.body;
      expect(patchPayload).toBeDefined();
      expect(patchPayload?.face_biometric_status).toBe('VERIFIED');
      expect(patchPayload?.face_frames_count).toBe(120);
      expect((patchPayload?.history as Record<string, unknown>)?.faceBiometricStatus).toBe('VERIFIED');

      // Direct helper check for stale PENDING protection
      expect(
        resolveProtectedBiometricStatus(
          { faceBiometricStatus: 'VERIFIED', faceFramesCount: 120, masterEmbedding: master },
          'PENDING',
        ),
      ).toBe('VERIFIED');

      const dualCols = buildPatientDualWriteColumns({
        id: enrolled.id,
        hn: 'HN-410382',
        fullName: 'Poom',
        age: 26,
        gender: 'ชาย',
        phone: '0899998888',
        diagnosis: 'Elderly Rehab',
        careGroup: 'MOBILITY_RECOVERY',
        affectedSide: 'RIGHT',
        motorAssessmentScore: 3,
        adlDressingScore: 3,
        assignedPtId: 'pt-1',
        delegatedToPtId: null,
        delegationNote: null,
        nextAppointmentDate: '',
        registrationStatus: 'active',
        enrolledAt: '2026-11-23T02:04:11.925523+00:00',
        faceBiometricStatus: 'PENDING',
        faceFramesCount: 0,
        faceImageUrl: faceUrl,
        masterEmbedding: master,
        angleEmbeddings: angles,
      });
      expect(dualCols.face_biometric_status).toBe('VERIFIED');
      expect(dualCols.face_frames_count).toBe(120);
    } finally {
      globalThis.fetch = origFetch;
      resetUnsupportedColumnsForTests();
      resetSupabaseCircuitForTests();
    }
  });

  it('validates SQL backfill migration coverage and guarantees zero Unicode emojis across all modified files', () => {
    const root = process.cwd();
    const migrationPath = path.join(
      root,
      'supabase/migrations/009_backfill_patient_biometrics_and_profile_from_history.sql',
    );
    const sql = fs.readFileSync(migrationPath, 'utf8');

    for (const token of [
      'face_biometric_status',
      'face_frames_count',
      'face_image_url',
      'master_embedding',
      'angle_embeddings',
      'biometric_data',
      'biometric_updated_at',
      'gender',
      'care_group',
      'affected_side',
      'motor_assessment_score',
      'adl_dressing_score',
      "NOTIFY pgrst, 'reload schema'",
    ]) {
      expect(sql).toContain(token);
    }

    const modifiedFiles = [
      'supabase/migrations/009_backfill_patient_biometrics_and_profile_from_history.sql',
      'packages/api-client/src/patientNormalization.ts',
      'packages/api-client/src/supabasePersistence.ts',
      'packages/api-client/src/cloudRouter.ts',
      'packages/api-client/src/index.ts',
      'src/lib/patientHydration.ts',
      'src/lib/biometricEngine.ts',
      'src/components/StaffAssistedEnrollmentModal.tsx',
      'src/components/KioskBiometricLoginView.tsx',
      'src/__tests__/patientBiometricHydration.test.ts',
    ];
    for (const rel of modifiedFiles) {
      const content = fs.readFileSync(path.join(root, rel), 'utf8');
      expect(EMOJI_RE.test(content), `Unexpected Unicode emoji in ${rel}`).toBe(false);
    }
  });
});
