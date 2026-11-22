import fs from 'node:fs';
import path from 'node:path';
import { beforeEach, describe, expect, it } from 'vitest';
import { createCloudDelegatedRouter } from '../../packages/api-client/src/index';
import {
  ensureDemoAssessmentsForPatient,
  listPatientAssessments,
  resetClinicalAssessmentStoreForTests,
} from '../lib/clinicalAssessmentStore';
import {
  buildLongitudinalCsvReport,
  compareTwoAssessmentVisits,
  computeExerciseBreakdown,
  computeLongitudinalProgressSummary,
  isDemoSeedAssessment,
  type PatientClinicalAssessmentRecord,
} from '../types/clinicalAssessment';

const EMOJI_RE = /[\p{Extended_Pictographic}\u{FE0F}\u{2600}-\u{27BF}\u{23E9}-\u{23FA}]/u;

function readRepoFile(relPath: string): string {
  return fs.readFileSync(path.resolve(process.cwd(), relPath), 'utf8');
}

async function callRouter(
  router: (req: Request) => Promise<Response>,
  routePath: string,
  init: { method?: string; token?: string; body?: unknown } = {},
): Promise<Response> {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (init.token) headers.Authorization = `Bearer ${init.token}`;
  const req = new Request(`https://drtechcares.workers.dev${routePath}`, {
    method: init.method ?? 'GET',
    headers,
    body: init.body !== undefined ? JSON.stringify(init.body) : undefined,
  });
  return router(req);
}

describe('Real Retrospective Clinical Tracking, Live ROM Measurement & Progress Comparison', () => {
  beforeEach(() => {
    resetClinicalAssessmentStoreForTests();
  });

  it('persists real assessments via Worker CRUD routes, merges real Kiosk sessions without duplicates, and never seeds fake demo rows', async () => {
    const router = createCloudDelegatedRouter({});
    const loginRes = await callRouter(router, '/api/auth/login', {
      method: 'POST',
      body: { username: 'admin', password: 'Admin@2026' },
    });
    expect(loginRes.status).toBe(200);
    const { token } = (await loginRes.json()) as { token: string };

    // Create a brand-new real patient
    const createPatRes = await callRouter(router, '/api/patients', {
      method: 'POST',
      token,
      body: {
        fullName: 'ภัทรดนย์ ผ่องอารมย์',
        hn: 'HN-50397',
        age: 17,
        gender: 'ชาย',
        phone: '0890005039',
        diagnosis: 'Stroke ADL Shoulder & Arm',
        affectedSide: 'RIGHT',
      },
    });
    expect(createPatRes.status).toBe(201);
    const { patient } = (await createPatRes.json()) as {
      patient: { id: string; hn: string; universalHealthLinkCode: string };
    };

    // 1) Brand-new patient MUST have 0 assessments (NO fabricated 30 -> 46.5 -> 62 demo rows!)
    const initialRes = await callRouter(
      router,
      `/api/patients/${encodeURIComponent(patient.id)}/assessments`,
      { token },
    );
    expect(initialRes.status).toBe(200);
    const initialJson = (await initialRes.json()) as { assessments: PatientClinicalAssessmentRecord[] };
    expect(initialJson.assessments).toHaveLength(0);

    // 2) Record Visit 1 (Baseline) via POST /api/patients/:id/assessments
    const post1 = await callRouter(
      router,
      `/api/patients/${encodeURIComponent(patient.id)}/assessments`,
      {
        method: 'POST',
        token,
        body: {
          assessed_at: '2026-11-01T09:00:00.000Z',
          facility_name: 'ศูนย์กายภาพบำบัด Dr.Tech.Care สาขาหลัก',
          exercise_code: 'STROKE_SHOULDER_FLEXION',
          exercise_name_th: 'ยกแขนตรงไปด้านหน้า',
          target_side: 'RIGHT',
          initial_score_0_to_6: 2,
          measured_rom_deg: 42.0,
          target_rom_deg: 120.0,
          pain_vas_0_to_10: 6,
          chief_complaint: 'ยกไหล่ขวาตึงช่วง 40 องศา',
          clinical_handover_note: 'บันทึกแรกรับจริง',
          next_session_goal: 'เพิ่มเป็น 60 องศา',
          measurement_method: 'CAMERA_AI_LIVE',
        },
      },
    );
    expect(post1.status).toBe(201);
    const { assessment: visit1 } = (await post1.json()) as { assessment: PatientClinicalAssessmentRecord };
    expect(visit1.measured_rom_deg).toBe(42.0);
    expect(visit1.measurement_method).toBe('CAMERA_AI_LIVE');

    // 3) Record a real Kiosk physio workout via POST /api/sessions
    const sessionRes = await callRouter(router, '/api/sessions', {
      method: 'POST',
      token,
      body: {
        patientId: patient.id,
        activityType: 'physio_exercise',
        exerciseId: 'STROKE_SHOULDER_FLEXION',
        exerciseName: 'ยกแขนตรงไปด้านหน้า',
        bodySide: 'RIGHT',
        completedSets: 2,
        completedReps: 10,
        accuracyPercent: 92,
        maxRomDeg: 78.5,
        painStopTriggered: false,
        aiFeedback: 'ยกแขนได้สูงขึ้นและคุมลำตัวนิ่งดี',
      },
    });
    expect(sessionRes.status).toBe(201);

    // 4) GET assessments must include BOTH Visit 1 AND the real Kiosk workout (2 total, no duplicates)
    const mergedRes = await callRouter(
      router,
      `/api/patients/${encodeURIComponent(patient.id)}/assessments`,
      { token },
    );
    expect(mergedRes.status).toBe(200);
    const mergedJson = (await mergedRes.json()) as { assessments: PatientClinicalAssessmentRecord[] };
    expect(mergedJson.assessments).toHaveLength(2);
    expect(mergedJson.assessments[0].measured_rom_deg).toBe(42.0);
    expect(mergedJson.assessments[1].measured_rom_deg).toBe(78.5);
    expect(mergedJson.assessments[1].source_channel).toBe('KIOSK_SESSION_AUTO_SYNC');

    // 5) Attempting to delete an auto-derived Kiosk session returns 400
    const delKiosk = await callRouter(
      router,
      `/api/patients/${encodeURIComponent(patient.id)}/assessments/${encodeURIComponent(mergedJson.assessments[1].id)}`,
      { method: 'DELETE', token },
    );
    expect(delKiosk.status).toBe(400);

    // 6) Deleting staff-created Visit 1 succeeds and leaves only the Kiosk session
    const delVisit1 = await callRouter(
      router,
      `/api/patients/${encodeURIComponent(patient.id)}/assessments/${encodeURIComponent(visit1.id)}`,
      { method: 'DELETE', token },
    );
    expect(delVisit1.status).toBe(200);

    const afterDelRes = await callRouter(
      router,
      `/api/patients/${encodeURIComponent(patient.id)}/assessments`,
      { token },
    );
    const afterDelJson = (await afterDelRes.json()) as { assessments: PatientClinicalAssessmentRecord[] };
    expect(afterDelJson.assessments).toHaveLength(1);
    expect(afterDelJson.assessments[0].measured_rom_deg).toBe(78.5);
  });

  it('computes accurate per-exercise breakdowns, visit-to-visit velocity/pain/verdict, and UTF-8 BOM CSV', () => {
    const records: PatientClinicalAssessmentRecord[] = [
      {
        id: 'r1',
        patient_id: 'p1',
        patient_hn: 'HN-50397',
        universal_link_code: 'DTC-EHR-AC756700',
        visit_number: 1,
        assessed_at: '2026-11-01T09:00:00.000Z',
        facility_name: 'สาขาหลัก',
        assessor_staff_id: 'pt-1',
        assessor_staff_name: 'กภ. สมชาย',
        exercise_code: 'STROKE_SHOULDER_FLEXION',
        exercise_name_th: 'ยกแขนตรงไปด้านหน้า',
        target_side: 'RIGHT',
        initial_score_0_to_6: 2,
        measured_rom_deg: 40.0,
        target_rom_deg: 120.0,
        pain_vas_0_to_10: 6,
        chief_complaint: 'ยกแขนตึง',
        clinical_handover_note: 'เริ่มฝึก',
        next_session_goal: '60 องศา',
        source_channel: 'STAFF_MANUAL_ASSESSMENT',
        measurement_method: 'CAMERA_AI_LIVE',
      },
      {
        id: 'r2',
        patient_id: 'p1',
        patient_hn: 'HN-50397',
        universal_link_code: 'DTC-EHR-AC756700',
        visit_number: 2,
        assessed_at: '2026-11-08T09:00:00.000Z',
        facility_name: 'สาขาหลัก',
        assessor_staff_id: 'pt-1',
        assessor_staff_name: 'กภ. สมชาย',
        exercise_code: 'MOPH_NECK_ROTATION',
        exercise_name_th: 'หันคอซ้าย-ขวา',
        target_side: 'BOTH',
        initial_score_0_to_6: 4,
        measured_rom_deg: 45.0,
        target_rom_deg: 60.0,
        pain_vas_0_to_10: 2,
        chief_complaint: 'ตึงต้นคอเล็กน้อย',
        clinical_handover_note: 'หันได้คล่องขึ้น',
        next_session_goal: '55 องศา',
        source_channel: 'STAFF_MANUAL_ASSESSMENT',
        measurement_method: 'MANUAL_INPUT',
      },
      {
        id: 'r3',
        patient_id: 'p1',
        patient_hn: 'HN-50397',
        universal_link_code: 'DTC-EHR-AC756700',
        visit_number: 3,
        assessed_at: '2026-11-15T09:00:00.000Z',
        facility_name: 'คลินิกเครือข่ายกายภาพชุมชน',
        assessor_staff_id: 'pt-2',
        assessor_staff_name: 'กภ. อรทัย',
        exercise_code: 'STROKE_SHOULDER_FLEXION',
        exercise_name_th: 'ยกแขนตรงไปด้านหน้า',
        target_side: 'RIGHT',
        initial_score_0_to_6: 4,
        measured_rom_deg: 82.0,
        target_rom_deg: 120.0,
        pain_vas_0_to_10: 2,
        chief_complaint: 'ยกแขนได้สูงขึ้นมาก',
        clinical_handover_note: 'ลดอาการเกร็งไหล่ได้ชัดเจน',
        next_session_goal: '100 องศา',
        source_channel: 'STAFF_MANUAL_ASSESSMENT',
        measurement_method: 'CAMERA_AI_LIVE',
      },
    ];

    // Mixed ALL summary marks hasMixedExercises = true
    const allSummary = computeLongitudinalProgressSummary(records, 'ALL');
    expect(allSummary.hasMixedExercises).toBe(true);
    expect(allSummary.totalVisits).toBe(3);

    // Breakdown separates Shoulder Flexion (2 visits) from Neck Rotation (1 visit)
    const breakdown = computeExerciseBreakdown(records);
    expect(breakdown).toHaveLength(2);
    const shoulder = breakdown.find((b) => b.exerciseCode === 'STROKE_SHOULDER_FLEXION')!;
    expect(shoulder.visitCount).toBe(2);
    expect(shoulder.firstRomDeg).toBe(40.0);
    expect(shoulder.latestRomDeg).toBe(82.0);
    expect(shoulder.deltaRomDeg).toBe(42.0);
    expect(shoulder.elapsedDays).toBe(14);
    expect(shoulder.romGainPerWeek).toBe(21.0);
    expect(shoulder.deltaPainVas).toBe(-4);
    expect(shoulder.trendStatus).toBe('IMPROVED');

    // Pair comparison between Visit 1 and Visit 3
    const pair = compareTwoAssessmentVisits(records[0], records[2]);
    expect(pair.isSameExercise).toBe(true);
    expect(pair.elapsedDays).toBe(14);
    expect(pair.romDeltaDeg).toBe(42.0);
    expect(pair.romGainPerWeek).toBe(21.0);
    expect(pair.scoreDelta).toBe(2);
    expect(pair.painDelta).toBe(-4);
    expect(pair.verdictTone).toBe('emerald');

    // CSV report includes UTF-8 BOM and Thai headers
    const csv = buildLongitudinalCsvReport(
      { id: 'p1', hn: 'HN-50397', fullName: 'ภัทรดนย์ ผ่องอารมย์', universalHealthLinkCode: 'DTC-EHR-AC756700' },
      allSummary,
    );
    expect(csv.startsWith('\uFEFF')).toBe(true);
    expect(csv).toContain('องศาที่วัดได้ (ROM deg)');
    expect(csv).toContain('วัดด้วยกล้อง AI');
    expect(csv).toContain('82.0');
  });

  it('purges legacy pt-demo-01/pt-demo-02 synthetic records when excludeDemoRows=true', async () => {
    const patient = { id: 'pat-cleanup-01', hn: 'HN-50397', fullName: 'ภัทรดนย์ ผ่องอารมย์' };
    const seeded = ensureDemoAssessmentsForPatient(patient);
    expect(seeded).toHaveLength(3);
    expect(seeded.every((r) => isDemoSeedAssessment(r))).toBe(true);

    const realOnly = await listPatientAssessments(patient, {
      includeCrossFacility: true,
      seedDemoIfEmpty: false,
      excludeDemoRows: true,
    });
    expect(realOnly).toHaveLength(0);
  });

  it('maintains 60FPS constraints and zero Unicode emojis across all tracking & measurement files', () => {
    const modalSrc = readRepoFile('src/components/ClinicalRomLiveMeasureModal.tsx');
    expect(modalSrc).not.toMatch(/shadowBlur\s*=\s*[1-9]/);
    expect(modalSrc).not.toContain('backdrop-blur');
    expect(modalSrc).not.toContain('poseLandmarker?.close()');
    expect(modalSrc).not.toContain('handLandmarker?.close()');

    const files = [
      'src/types/clinicalAssessment.ts',
      'src/lib/clinicalAssessmentStore.ts',
      'src/components/ClinicalRomLiveMeasureModal.tsx',
      'src/components/PatientClinicalRecordPanel.tsx',
      'src/components/ClinicalStaffPortalShell.tsx',
      'packages/api-client/src/index.ts',
      'packages/api-client/src/supabasePersistence.ts',
      'packages/api-client/src/cloudRouter.ts',
      'apps/api/src/db.ts',
      'apps/api/src/app.ts',
    ];
    for (const rel of files) {
      expect(EMOJI_RE.test(readRepoFile(rel)), `Unexpected emoji in ${rel}`).toBe(false);
    }
  });
});

