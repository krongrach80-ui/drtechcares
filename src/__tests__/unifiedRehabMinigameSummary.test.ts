import fs from 'node:fs';
import path from 'node:path';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import {
  createCloudDelegatedRouter,
  type PatientRecord,
  type RehabSessionRecord,
  type UserAccount,
} from '@drtechcares/api-client';
import {
  computeLongitudinalProgressSummary,
  type PatientClinicalAssessmentRecord,
} from '../types/clinicalAssessment';
import {
  ENCOURAGE_TRY_HARDER_PHRASE_TH,
  buildPatientMinigameSummary,
  buildPatientProgressSummaryReport,
  buildUnifiedRehabMinigameSummary,
  evaluateMinigameRoundVerdict,
  isMinigameRehabSession,
  shortenThaiExerciseLabel,
} from '../utils/unifiedRehabMinigameSummary';
import { PatientClinicalRecordPanel } from '../components/PatientClinicalRecordPanel';
import { PatientProgressSummaryModal } from '../components/PatientProgressSummaryModal';

const EMOJI_RE = /[\p{Extended_Pictographic}\u{FE0F}\u{2600}-\u{27BF}\u{23E9}-\u{23FA}]/u;

function makeAssessment(
  overrides: Partial<PatientClinicalAssessmentRecord>,
): PatientClinicalAssessmentRecord {
  return {
    id: overrides.id ?? 'assess-1',
    patient_id: overrides.patient_id ?? 'pat-sync-1',
    patient_hn: overrides.patient_hn ?? 'HN-837042',
    universal_link_code: overrides.universal_link_code ?? 'DTC-EHR-E74E95BC',
    visit_number: overrides.visit_number ?? 1,
    assessed_at: overrides.assessed_at ?? '2026-11-20T09:00:00.000Z',
    facility_name: overrides.facility_name ?? 'ตู้ Kiosk กายภาพอัจฉริยะ Dr.Tech.Care',
    assessor_staff_id: overrides.assessor_staff_id ?? 'SYSTEM_KIOSK_AUTO_SYNC',
    assessor_staff_name: overrides.assessor_staff_name ?? 'ระบบวัดผลอัตโนมัติ (Kiosk AI)',
    exercise_code: overrides.exercise_code ?? 'STROKE_SHOULDER_FLEXION',
    exercise_name_th: overrides.exercise_name_th ?? '[Stroke ADL] ยกแขนไปด้านหน้า',
    target_side: overrides.target_side ?? 'RIGHT',
    initial_score_0_to_6: overrides.initial_score_0_to_6 ?? 4,
    measured_rom_deg: overrides.measured_rom_deg ?? 95,
    target_rom_deg: overrides.target_rom_deg ?? 125,
    pain_vas_0_to_10: overrides.pain_vas_0_to_10 ?? null,
    chief_complaint: overrides.chief_complaint ?? 'ฝึกกายภาพผ่านตู้ Kiosk',
    clinical_handover_note: overrides.clinical_handover_note ?? 'บันทึกอัตโนมัติ',
    next_session_goal: overrides.next_session_goal ?? 'ฝึกต่อเนื่อง',
    source_channel: overrides.source_channel ?? 'KIOSK_SESSION_AUTO_SYNC',
    measurement_method: overrides.measurement_method ?? 'KIOSK_SESSION_AUTO_SYNC',
  };
}

function makeMinigameSession(
  overrides: Partial<RehabSessionRecord> & {
    correct?: number;
    total?: number;
    durationSec?: number;
  },
): RehabSessionRecord {
  const total = overrides.total ?? 10;
  const correct = overrides.correct ?? 8;
  const accuracyPercent =
    typeof overrides.accuracyPercent === 'number'
      ? overrides.accuracyPercent
      : Math.round((correct / Math.max(1, total)) * 100);
  return {
    id: overrides.id ?? `mg-${Date.now()}`,
    patientId: overrides.patientId ?? 'pat-sync-1',
    activityType: 'minigame',
    exerciseId: overrides.exerciseId ?? 'cognitive-card-game',
    exerciseName: overrides.exerciseName ?? 'มินิเกมฝึกสมอง (การ์ดกายภาพ)',
    completedSets: 1,
    completedReps: total,
    accuracyPercent,
    maxRomDeg: 0,
    painStopTriggered: false,
    aiFeedback: `Cognitive Accuracy ${accuracyPercent}% (${correct}/${total})`,
    kinematicsReplay: [],
    minigameScore: {
      correct,
      total,
      category: 'mixed-99',
      durationSec: overrides.durationSec ?? 48,
      accuracyPct: accuracyPercent,
    },
    createdAt: overrides.createdAt ?? '2026-11-20T10:00:00.000Z',
  };
}

describe('Unified Physical Therapy + Minigame Summary & Clinical Verdict Engine', () => {
  it('classifies IMPROVED, STABLE, TRY_HARDER (with encouraging Thai phrase), and FIRST_SESSION accurately for minigames', () => {
    const improved = evaluateMinigameRoundVerdict({
      currentAccuracyPct: 90,
      currentCorrect: 9,
      currentTotal: 10,
      previousAccuracyPct: 70,
      previousCorrect: 7,
    });
    expect(improved.code).toBe('IMPROVED');
    expect(improved.badgeTh).toBe('ดีขึ้น');

    const stable = evaluateMinigameRoundVerdict({
      currentAccuracyPct: 80,
      currentCorrect: 8,
      currentTotal: 10,
      previousAccuracyPct: 80,
      previousCorrect: 8,
    });
    expect(stable.code).toBe('STABLE');
    expect(stable.badgeTh).toBe('เท่าเดิม');

    const tryHarder = evaluateMinigameRoundVerdict({
      currentAccuracyPct: 60,
      currentCorrect: 6,
      currentTotal: 10,
      previousAccuracyPct: 80,
      previousCorrect: 8,
    });
    expect(tryHarder.code).toBe('TRY_HARDER');
    expect(tryHarder.headlineTh).toBe(ENCOURAGE_TRY_HARDER_PHRASE_TH);

    const firstGood = evaluateMinigameRoundVerdict({
      currentAccuracyPct: 90,
      currentCorrect: 9,
      currentTotal: 10,
      previousAccuracyPct: null,
    });
    expect(firstGood.code).toBe('FIRST_SESSION');

    const firstLow = evaluateMinigameRoundVerdict({
      currentAccuracyPct: 50,
      currentCorrect: 5,
      currentTotal: 10,
      previousAccuracyPct: null,
    });
    expect(firstLow.code).toBe('TRY_HARDER');
    expect(firstLow.headlineTh).toBe(ENCOURAGE_TRY_HARDER_PHRASE_TH);
  });

  it('recognizes peak clinical attainment across different exercises without false regression and strictly isolates per patient', () => {
    // 3 different exercises all at 6/6 and > 100% target (mirroring the mockup: 140%, 146.7%, 117.9%)
    const peakRecords: PatientClinicalAssessmentRecord[] = [
      makeAssessment({
        id: 'a1',
        visit_number: 1,
        assessed_at: '2026-11-26T10:57:00.000Z',
        exercise_code: 'STROKE_SHOULDER_FLEXION',
        exercise_name_th: '[Stroke ADL] ยกแขนไปด้านหน้า',
        initial_score_0_to_6: 6,
        measured_rom_deg: 175.0,
        target_rom_deg: 125.0,
      }),
      makeAssessment({
        id: 'a2',
        visit_number: 2,
        assessed_at: '2026-11-26T10:58:00.000Z',
        exercise_code: 'STROKE_SHOULDER_ABDUCTION',
        exercise_name_th: '[Stroke ADL] กางแขนด้านข้าง',
        initial_score_0_to_6: 6,
        measured_rom_deg: 168.7,
        target_rom_deg: 115.0,
      }),
      makeAssessment({
        id: 'a3',
        visit_number: 3,
        assessed_at: '2026-11-26T10:59:00.000Z',
        exercise_code: 'STROKE_OVERHEAD_REACH',
        exercise_name_th: '[Stroke & สร.] ประสานมือชูขึ้นเหนือศีรษะ',
        initial_score_0_to_6: 6,
        measured_rom_deg: 159.2,
        target_rom_deg: 135.0,
      }),
    ];

    const sessions: RehabSessionRecord[] = [
      makeMinigameSession({
        id: 'mg-1',
        patientId: 'pat-sync-1',
        correct: 7,
        total: 10,
        createdAt: '2026-11-25T09:00:00.000Z',
      }),
      makeMinigameSession({
        id: 'mg-2',
        patientId: 'pat-sync-1',
        correct: 9,
        total: 10,
        createdAt: '2026-11-26T11:05:00.000Z',
      }),
      // Another patient's minigame MUST be ignored (strict patient isolation)
      makeMinigameSession({
        id: 'mg-leak',
        patientId: 'pat-other-999',
        correct: 2,
        total: 10,
        createdAt: '2026-11-26T11:10:00.000Z',
      }),
    ];

    const longitudinal = computeLongitudinalProgressSummary(peakRecords, 'ALL');
    const unified = buildUnifiedRehabMinigameSummary({
      patientId: 'pat-sync-1',
      longitudinalSummary: longitudinal,
      sessions,
    });

    // Peak 6/6 across 3 exercises must be STABLE (peak excellence), NOT TRY_HARDER
    expect(unified.rehab.verdict.code).toBe('STABLE');
    // Minigame improved 70% -> 90% (+20%)
    expect(unified.minigame.totalRounds).toBe(2);
    expect(unified.minigame.verdict.code).toBe('IMPROVED');
    expect(unified.minigame.deltaAccuracyFromPrevious).toBe(20);
    // Combined overall verdict is IMPROVED (stable peak rehab + improved minigame)
    expect(unified.overall.verdict.code).toBe('IMPROVED');
    expect(unified.overall.totalActivities).toBe(5);
    expect(unified.timeline).toHaveLength(5);
    // Newest item in timeline is minigame round 2 (11:05)
    expect(unified.timeline[0]?.kind).toBe('minigame');
    expect(isMinigameRehabSession(sessions[0])).toBe(true);
  });

  it('produces encouraging TRY_HARDER verdict when performance dips from previous session', () => {
    const dipRecords: PatientClinicalAssessmentRecord[] = [
      makeAssessment({
        id: 'd1',
        visit_number: 1,
        assessed_at: '2026-11-20T09:00:00.000Z',
        initial_score_0_to_6: 4,
        measured_rom_deg: 105.0,
        target_rom_deg: 125.0,
      }),
      makeAssessment({
        id: 'd2',
        visit_number: 2,
        assessed_at: '2026-11-24T09:00:00.000Z',
        initial_score_0_to_6: 3,
        measured_rom_deg: 82.0,
        target_rom_deg: 125.0,
      }),
    ];
    const dipMinigames: RehabSessionRecord[] = [
      makeMinigameSession({
        id: 'mg-d1',
        patientId: 'pat-sync-1',
        correct: 8,
        total: 10,
        createdAt: '2026-11-20T09:30:00.000Z',
      }),
      makeMinigameSession({
        id: 'mg-d2',
        patientId: 'pat-sync-1',
        correct: 6,
        total: 10,
        createdAt: '2026-11-24T09:30:00.000Z',
      }),
    ];

    const longitudinal = computeLongitudinalProgressSummary(dipRecords, 'ALL');
    const unified = buildUnifiedRehabMinigameSummary({
      patientId: 'pat-sync-1',
      longitudinalSummary: longitudinal,
      sessions: dipMinigames,
    });

    expect(unified.rehab.verdict.code).toBe('TRY_HARDER');
    expect(unified.rehab.verdict.headlineTh).toBe(ENCOURAGE_TRY_HARDER_PHRASE_TH);
    expect(unified.minigame.verdict.code).toBe('TRY_HARDER');
    expect(unified.overall.verdict.code).toBe('TRY_HARDER');
    expect(unified.overall.verdict.headlineTh).toBe(ENCOURAGE_TRY_HARDER_PHRASE_TH);
  });

  it('syncs both physio and minigame sessions end-to-end through Cloudflare Worker /api/sessions and /api/kiosk/pin-login', async () => {
    const router = createCloudDelegatedRouter({});
    const loginRes = await router(
      new Request('https://drtechcares.workers.dev/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: 'admin', password: 'Admin@2026' }),
      }),
    );
    const { token } = (await loginRes.json()) as { token: string };

    const createRes = await router(
      new Request('https://drtechcares.workers.dev/api/patients', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify({
          fullName: 'คุณสมหวัง ฝึกดี',
          hn: 'HN-778899',
          age: 64,
          gender: 'ชาย',
          phone: '0817788990',
          diagnosis: 'Stroke ADL (Shoulder & Arm)',
          affectedSide: 'RIGHT',
        }),
      }),
    );
    const { patient } = (await createRes.json()) as { patient: { id: string; hn: string } };

    // Record 1 physio session + 2 minigame sessions
    await router(
      new Request('https://drtechcares.workers.dev/api/sessions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify({
          patientId: patient.id,
          activityType: 'physio_exercise',
          exerciseId: 'STROKE_SHOULDER_FLEXION',
          exerciseName: 'ยกแขนไปด้านหน้า',
          completedSets: 1,
          completedReps: 10,
          accuracyPercent: 95,
          maxRomDeg: 132.5,
          painStopTriggered: false,
        }),
      }),
    );
    await router(
      new Request('https://drtechcares.workers.dev/api/sessions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify({
          patientId: patient.id,
          activityType: 'minigame',
          exerciseId: 'cognitive-card-game',
          exerciseName: 'มินิเกมฝึกสมอง (การ์ดกายภาพ)',
          completedSets: 1,
          completedReps: 10,
          accuracyPercent: 70,
          maxRomDeg: 0,
          minigameScore: { correct: 7, total: 10, category: 'mixed-99', durationSec: 52, accuracyPct: 70 },
        }),
      }),
    );
    await router(
      new Request('https://drtechcares.workers.dev/api/sessions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify({
          patientId: patient.id,
          activityType: 'minigame',
          exerciseId: 'cognitive-card-game',
          exerciseName: 'มินิเกมฝึกสมอง (การ์ดกายภาพ)',
          completedSets: 1,
          completedReps: 10,
          accuracyPercent: 90,
          maxRomDeg: 0,
          minigameScore: { correct: 9, total: 10, category: 'mixed-99', durationSec: 40, accuracyPct: 90 },
        }),
      }),
    );

    // Verify GET /api/sessions returns all 3 sessions accurately
    const sessionsRes = await router(
      new Request(`https://drtechcares.workers.dev/api/sessions?patientId=${encodeURIComponent(patient.id)}`, {
        headers: { Authorization: `Bearer ${token}` },
      }),
    );
    const sessionsJson = (await sessionsRes.json()) as { sessions: RehabSessionRecord[] };
    expect(sessionsJson.sessions).toHaveLength(3);

    const mgSummary = buildPatientMinigameSummary(patient.id, sessionsJson.sessions);
    expect(mgSummary.totalRounds).toBe(2);
    expect(mgSummary.latestAccuracyPct).toBe(90);
    expect(mgSummary.verdict.code).toBe('IMPROVED');

    // Verify Kiosk PIN login also receives recentSessions so Kiosk summary syncs automatically
    const pinRes = await router(
      new Request('https://drtechcares.workers.dev/api/kiosk/pin-login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pin: '8990' }),
      }),
    );
    expect(pinRes.status).toBe(200);
    const pinJson = (await pinRes.json()) as { matched: boolean; recentSessions?: RehabSessionRecord[] };
    expect(pinJson.matched).toBe(true);
    expect(pinJson.recentSessions?.length).toBe(3);
  });

  it('renders 3-column PatientClinicalRecordPanel with unified Rehab + Minigame verdict, table, timeline, and zero emojis', () => {
    const patient: PatientRecord = {
      id: 'pat-ui-sync',
      hn: 'HN-837042',
      fullName: 'นายภัทรนนท์ ผ่องอารมณ์',
      age: 45,
      gender: 'ชาย',
      phone: '0891112233',
      diagnosis: 'Stroke ADL (Shoulder & Arm)',
      affectedSide: 'RIGHT',
      assignedPtId: 'pt-1',
      delegatedToPtId: null,
      delegationNote: null,
      nextAppointmentDate: '2026-12-01',
      registrationStatus: 'active',
      enrolledAt: '2026-11-01T00:00:00.000Z',
    };
    const admin: UserAccount = {
      id: 'admin-1',
      username: 'admin',
      role: 'super_admin',
      fullName: 'นพ.สมชาย',
      phone: '02-000-0001',
      active: true,
      createdAt: '2026-11-01T00:00:00.000Z',
    };
    const records: PatientClinicalAssessmentRecord[] = [
      makeAssessment({
        id: 'ui-r1',
        patient_id: patient.id,
        visit_number: 1,
        assessed_at: '2026-11-24T09:00:00.000Z',
        initial_score_0_to_6: 4,
        measured_rom_deg: 100.0,
        target_rom_deg: 125.0,
      }),
      makeAssessment({
        id: 'ui-r2',
        patient_id: patient.id,
        visit_number: 2,
        assessed_at: '2026-11-26T09:00:00.000Z',
        initial_score_0_to_6: 5,
        measured_rom_deg: 122.0,
        target_rom_deg: 125.0,
      }),
    ];
    const sessions: RehabSessionRecord[] = [
      makeMinigameSession({
        id: 'ui-mg-1',
        patientId: patient.id,
        correct: 7,
        total: 10,
        createdAt: '2026-11-24T09:20:00.000Z',
      }),
      makeMinigameSession({
        id: 'ui-mg-2',
        patientId: patient.id,
        correct: 9,
        total: 10,
        createdAt: '2026-11-26T09:20:00.000Z',
      }),
    ];

    const html = renderToStaticMarkup(
      React.createElement(PatientClinicalRecordPanel, {
        patients: [patient],
        sessions,
        currentUser: admin,
        selectedPatientId: patient.id,
        onSelectPatientId: () => {},
        initialRecordsOverride: records,
      }),
    );

    expect(html).toContain('data-testid="unified-progress-verdict-card"');
    expect(html).toContain('data-testid="verdict-domain-rehab"');
    expect(html).toContain('data-testid="verdict-domain-minigame"');
    expect(html).toContain('data-testid="unified-exercise-minigame-table"');
    expect(html).toContain('data-testid="unified-table-minigame-row"');
    expect(html).toContain('data-testid="unified-activity-timeline"');
    expect(html).toContain('ดีขึ้นทั้งกายภาพและมินิเกม');
    expect(html).toContain('มินิเกมฝึกสมอง');
    expect(EMOJI_RE.test(html)).toBe(false);

    const root = process.cwd();
    const auditFiles = [
      'src/utils/unifiedRehabMinigameSummary.ts',
      'src/components/PatientProgressSummaryModal.tsx',
      'src/components/PatientClinicalRecordPanel.tsx',
      'src/PatientPersonalPortal.tsx',
      'src/App.tsx',
      'apps/kiosk/src/PhysicalCardGame.tsx',
      'src/__tests__/unifiedRehabMinigameSummary.test.ts',
    ];
    for (const rel of auditFiles) {
      const content = fs.readFileSync(path.join(root, rel), 'utf8');
      expect(EMOJI_RE.test(content), `Unexpected Unicode emoji in ${rel}`).toBe(false);
    }
  });

  it('builds and renders PatientProgressSummaryModal with before-vs-now ROM & minigame comparison bars and zero emojis', () => {
    expect(shortenThaiExerciseLabel('[Stroke ADL] ยกแขนไปด้านหน้า')).toBe('ยกแขน');
    expect(shortenThaiExerciseLabel('[Stroke ADL] กางแขนด้านข้าง')).toBe('กางแขน');
    expect(shortenThaiExerciseLabel('[Stroke & สร.] ประสานมือชูขึ้นเหนือศีรษะ')).toBe('ประสานมือ');

    const records: PatientClinicalAssessmentRecord[] = [
      makeAssessment({
        id: 'w1-flex',
        visit_number: 1,
        assessed_at: '2026-11-18T09:00:00.000Z',
        exercise_code: 'STROKE_SHOULDER_FLEXION',
        exercise_name_th: '[Stroke ADL] ยกแขนไปด้านหน้า',
        initial_score_0_to_6: 4,
        measured_rom_deg: 105.0,
        target_rom_deg: 125.0,
      }),
      makeAssessment({
        id: 'w1-abd',
        visit_number: 2,
        assessed_at: '2026-11-18T09:10:00.000Z',
        exercise_code: 'STROKE_SHOULDER_ABDUCTION',
        exercise_name_th: '[Stroke ADL] กางแขนด้านข้าง',
        initial_score_0_to_6: 4,
        measured_rom_deg: 98.0,
        target_rom_deg: 115.0,
      }),
      makeAssessment({
        id: 'w2-flex',
        visit_number: 3,
        assessed_at: '2026-11-26T09:00:00.000Z',
        exercise_code: 'STROKE_SHOULDER_FLEXION',
        exercise_name_th: '[Stroke ADL] ยกแขนไปด้านหน้า',
        initial_score_0_to_6: 6,
        measured_rom_deg: 122.0,
        target_rom_deg: 125.0,
      }),
      makeAssessment({
        id: 'w2-abd',
        visit_number: 4,
        assessed_at: '2026-11-26T09:10:00.000Z',
        exercise_code: 'STROKE_SHOULDER_ABDUCTION',
        exercise_name_th: '[Stroke ADL] กางแขนด้านข้าง',
        initial_score_0_to_6: 6,
        measured_rom_deg: 112.0,
        target_rom_deg: 115.0,
      }),
    ];

    const sessions: RehabSessionRecord[] = [
      makeMinigameSession({
        id: 'mg-w1',
        patientId: 'pat-sync-1',
        correct: 7,
        total: 10,
        createdAt: '2026-11-18T09:20:00.000Z',
      }),
      makeMinigameSession({
        id: 'mg-w2',
        patientId: 'pat-sync-1',
        correct: 9,
        total: 10,
        createdAt: '2026-11-26T09:20:00.000Z',
      }),
    ];

    const longitudinal = computeLongitudinalProgressSummary(records, 'ALL');
    const unified = buildUnifiedRehabMinigameSummary({
      patientId: 'pat-sync-1',
      longitudinalSummary: longitudinal,
      sessions,
    });
    const report = buildPatientProgressSummaryReport({
      patientName: 'นายภัทรนนท์ ผ่องอารมณ์',
      patientHn: 'HN-837042',
      unifiedSummary: unified,
      assessments: records,
      referenceNow: new Date('2026-11-26T12:00:00.000Z'),
    });

    expect(report.exerciseBars).toHaveLength(2);
    expect(report.exerciseBars[0]).toMatchObject({
      shortLabelTh: 'ยกแขน',
      previousRomDeg: 105,
      currentRomDeg: 122,
      deltaDeg: 17,
      status: 'IMPROVED',
    });
    expect(report.avgRomDeltaDeg).toBe(15.5);
    expect(report.minigameComparison.played).toBe(true);
    expect(report.minigameComparison.deltaAccuracyPct).toBe(20);
    expect(report.overallVerdict.code).toBe('IMPROVED');

    const html = renderToStaticMarkup(
      React.createElement(PatientProgressSummaryModal, {
        open: true,
        report,
        onClose: () => {},
      }),
    );

    expect(html).toContain('data-testid="patient-progress-summary-modal"');
    expect(html).toContain('data-testid="summary-comparison-bar-chart"');
    expect(html).toContain('data-testid="summary-bar-STROKE_SHOULDER_FLEXION"');
    expect(html).toContain('data-testid="summary-bar-MINIGAME_COGNITIVE"');
    expect(html).toContain('ความคืบหน้าภาพรวม');
    expect(html).toContain('จำนวนครั้งที่ฝึก:');
    expect(html).toContain('คะแนนการประเมินเฉลี่ย:');
    expect(html).toContain('องศาการเคลื่อนไหว (ROM) เฉลี่ย:');
    expect(html).toContain('เพิ่มขึ้น +15.5° เทียบกับครั้งก่อน');
    expect(html).toContain('พิมพ์สรุป');
    expect(html).toContain('ปิดหน้าต่าง');
    expect(EMOJI_RE.test(html)).toBe(false);
  });
});

