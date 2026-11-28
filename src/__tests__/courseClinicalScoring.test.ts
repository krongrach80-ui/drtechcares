import fs from 'node:fs';
import path from 'node:path';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { beforeEach, describe, expect, it } from 'vitest';
import {
  createCloudDelegatedRouter,
  type PatientCourseRecord,
  type PatientRecord,
  type RehabScheduleItem,
  type RehabSessionHistoryEntry,
  type RehabSessionRecord,
  type UserAccount,
} from '@drtechcares/api-client';
import { PatientClinicalRecordPanel } from '../components/PatientClinicalRecordPanel';
import {
  buildAssessmentInputFromCourseResult,
  buildPatientCourseAutoAssessments,
  calculateMahidolScoreFromCourse,
} from '../utils/courseClinicalScoring';
import {
  listPatientAssessments,
  resetClinicalAssessmentStoreForTests,
  upsertCourseAutoAssessment,
} from '../lib/clinicalAssessmentStore';
import { computeLongitudinalProgressSummary } from '../types/clinicalAssessment';

const EMOJI_RE = /[\p{Extended_Pictographic}\u{FE0F}\u{2600}-\u{27BF}\u{23E9}-\u{23FA}]/u;

function makePatient(overrides: Partial<PatientRecord> & { id: string; hn: string; fullName: string }): PatientRecord {
  return {
    age: 42,
    gender: 'ชาย',
    phone: '0810000000',
    diagnosis: 'ฟื้นฟูสมรรถภาพผู้สูงอายุ',
    affectedSide: 'RIGHT',
    assignedPtId: 'pt-01',
    delegatedToPtId: null,
    delegationNote: null,
    nextAppointmentDate: '2026-11-30',
    registrationStatus: 'active',
    enrolledAt: '2026-11-01T00:00:00.000Z',
    ...overrides,
  };
}

describe('Automatic 0-6 Mahidol Score & ROM Calculation From PT-Assigned Rehab Courses', () => {
  beforeEach(() => {
    resetClinicalAssessmentStoreForTests();
  });

  it('accurately calculates all 7 Mahidol levels (0 through 6) from course performance with explicit ROM', () => {
    // Level 0: zero reps and zero ROM -> composite 0, score 0 (never inflated by default accuracy)
    const level0 = calculateMahidolScoreFromCourse({
      achievedRomDeg: 0,
      targetRomDeg: 90,
      completedReps: 0,
      targetReps: 10,
      sourceSessionId: 'sess-lvl-0',
      exerciseCode: 'ELDERLY_SHOULDER_FLEXION',
    });
    expect(level0.calculatedScore0To6).toBe(0);
    expect(level0.compositePct).toBe(0);
    expect(level0.effectiveRomDeg).toBe(0);

    // Level 1: composite 1..19 (rom=10/100 -> 5, rep=1/10 -> 3, acc=40 -> 8 => 16%)
    const level1 = calculateMahidolScoreFromCourse({
      achievedRomDeg: 10,
      targetRomDeg: 100,
      completedReps: 1,
      targetReps: 10,
      accuracyScore: 40,
      sourceSessionId: 'sess-lvl-1',
      exerciseCode: 'ELDERLY_SHOULDER_FLEXION',
    });
    expect(level1.compositePct).toBe(16);
    expect(level1.calculatedScore0To6).toBe(1);

    // Level 2: composite 20..39 (rom=30/100 -> 15, rep=3/10 -> 9, acc=50 -> 10 => 34%)
    const level2 = calculateMahidolScoreFromCourse({
      achievedRomDeg: 30,
      targetRomDeg: 100,
      completedReps: 3,
      targetReps: 10,
      accuracyScore: 50,
      sourceSessionId: 'sess-lvl-2',
      exerciseCode: 'ELDERLY_SHOULDER_FLEXION',
    });
    expect(level2.compositePct).toBe(34);
    expect(level2.calculatedScore0To6).toBe(2);

    // Level 3: composite 40..59 (rom=50/100 -> 25, rep=5/10 -> 15, acc=60 -> 12 => 52%)
    const level3 = calculateMahidolScoreFromCourse({
      achievedRomDeg: 50,
      targetRomDeg: 100,
      completedReps: 5,
      targetReps: 10,
      accuracyScore: 60,
      sourceSessionId: 'sess-lvl-3',
      exerciseCode: 'ELDERLY_SHOULDER_FLEXION',
    });
    expect(level3.compositePct).toBe(52);
    expect(level3.calculatedScore0To6).toBe(3);

    // Level 4: composite 60..79 (rom=70/100 -> 35, rep=7/10 -> 21, acc=70 -> 14 => 70%)
    const level4 = calculateMahidolScoreFromCourse({
      achievedRomDeg: 70,
      targetRomDeg: 100,
      completedReps: 7,
      targetReps: 10,
      accuracyScore: 70,
      sourceSessionId: 'sess-lvl-4',
      exerciseCode: 'ELDERLY_SHOULDER_FLEXION',
    });
    expect(level4.compositePct).toBe(70);
    expect(level4.calculatedScore0To6).toBe(4);

    // Level 5: prompt specification example (76/90 deg, 10/10 reps, 90% accuracy => composite 90%, level 5/6)
    const level5 = calculateMahidolScoreFromCourse({
      achievedRomDeg: 76,
      targetRomDeg: 90,
      completedReps: 10,
      targetReps: 10,
      accuracyScore: 90,
      sourceSessionId: 'sess-lvl-5',
      exerciseCode: 'ELDERLY_SHOULDER_FLEXION',
    });
    expect(level5.effectiveRomDeg).toBe(76.0);
    expect(level5.targetRomDeg).toBe(90);
    expect(level5.romPct).toBe(84);
    expect(level5.compositePct).toBe(90);
    expect(level5.calculatedScore0To6).toBe(5);
    expect(level5.breakdownSummaryTh).toBe(
      'คำนวณอัตโนมัติจากคอร์ส: องศาเฉลี่ย/สูงสุด 76.0° จากเป้า 90° (84%) • ทำครบ 10/10 ครั้ง • ความแม่นยำ 90% -> ประเมินได้ระดับ 5/6',
    );
    expect(EMOJI_RE.test(level5.breakdownSummaryTh)).toBe(false);

    // Level 6: composite >= 95 (full range & target completion achieved)
    const level6 = calculateMahidolScoreFromCourse({
      achievedRomDeg: 90,
      targetRomDeg: 90,
      completedReps: 10,
      targetReps: 10,
      accuracyScore: 95,
      sourceSessionId: 'sess-lvl-6',
      exerciseCode: 'ELDERLY_SHOULDER_FLEXION',
    });
    expect(level6.compositePct).toBe(99);
    expect(level6.calculatedScore0To6).toBe(6);
  });

  it('derives proportional effective ROM automatically when raw angle telemetry is missing', () => {
    // 7/10 reps on target 90 deg with no raw ROM telemetry -> effectiveRomDeg = 63.0 deg
    const proportional = calculateMahidolScoreFromCourse({
      maxRomDeg: 0,
      targetRomDeg: 90,
      completedReps: 7,
      targetReps: 10,
      accuracyScore: 90,
      sourceSessionId: 'sess-prop-1',
      exerciseCode: 'ELDERLY_SHOULDER_FLEXION',
    });
    expect(proportional.romDerivedProportionally).toBe(true);
    expect(proportional.effectiveRomDeg).toBe(63.0);
    // romRatio=0.7 (35%) + repRatio=0.7 (21%) + accRatio=0.9 (18%) = 74% -> level 4/6
    expect(proportional.compositePct).toBe(74);
    expect(proportional.calculatedScore0To6).toBe(4);
    expect(proportional.breakdownSummaryTh).toContain('63.0°');
    expect(proportional.breakdownSummaryTh).toContain('7/10 ครั้ง');
    expect(proportional.breakdownSummaryTh).toContain('ระดับ 4/6');

    // Full 10/10 reps without raw angle telemetry and default 85% accuracy -> 90.0 deg, composite 97% -> level 6/6
    const fullProportional = calculateMahidolScoreFromCourse({
      maxRomDeg: 0,
      targetRomDeg: 90,
      completedReps: 10,
      targetReps: 10,
      sourceSessionId: 'sess-prop-2',
      exerciseCode: 'ELDERLY_SHOULDER_FLEXION',
    });
    expect(fullProportional.romDerivedProportionally).toBe(true);
    expect(fullProportional.effectiveRomDeg).toBe(90.0);
    expect(fullProportional.compositePct).toBe(97);
    expect(fullProportional.calculatedScore0To6).toBe(6);
  });

  it('syncs course sessions idempotently by sourceSessionId without creating duplicate timeline rows', async () => {
    const patient = makePatient({
      id: 'pat-kru-aun',
      hn: 'HN-359954',
      fullName: 'ครูอั๋น',
      universalHealthLinkCode: 'DTC-EHR-55624217',
    });

    const course: PatientCourseRecord = {
      id: 'course-aun-1',
      patientId: patient.id,
      title: 'คอร์สฟื้นฟูหัวไหล่และแขนผู้สูงอายุ',
      cognitiveModpackId: '',
      updatedAt: '2026-11-20T08:00:00.000Z',
      weeklySchedule: [
        {
          dayIndex: 1,
          dayType: 'training',
          activityType: 'physio_exercise',
          difficulty: 'medium',
          label: 'วันจันทร์',
          exercises: [
            {
              exerciseId: 'ELDERLY_SHOULDER_FLEXION',
              side: 'right',
              sets: 2,
              reps: 10,
              holdSec: 3,
              targetMinDeg: 90,
              targetMaxDeg: 110,
              allowedStarterSide: 'right',
            },
          ],
        },
      ],
    };

    const sessions: RehabSessionRecord[] = [
      {
        id: 'sess-aun-day1',
        patientId: patient.id,
        activityType: 'physio_exercise',
        exerciseId: 'ELDERLY_SHOULDER_FLEXION',
        exerciseName: 'งอข้อไหล่ด้านหน้า (Shoulder Flexion)',
        bodySide: 'RIGHT',
        completedSets: 2,
        completedReps: 8,
        accuracyPercent: 84,
        maxRomDeg: 62.0,
        painStopTriggered: false,
        aiFeedback: 'ควบคุมลำตัวได้ดี',
        kinematicsReplay: [],
        createdAt: '2026-11-20T09:00:00.000Z',
      },
      {
        id: 'sess-aun-day2',
        patientId: patient.id,
        activityType: 'physio_exercise',
        exerciseId: 'ELDERLY_SHOULDER_FLEXION',
        exerciseName: 'งอข้อไหล่ด้านหน้า (Shoulder Flexion)',
        bodySide: 'RIGHT',
        completedSets: 2,
        completedReps: 10,
        accuracyPercent: 92,
        maxRomDeg: 81.0,
        painStopTriggered: false,
        aiFeedback: 'ยกแขนได้สูงขึ้นชัดเจน',
        kinematicsReplay: [],
        createdAt: '2026-11-23T09:00:00.000Z',
      },
    ];

    // First sync pass: both completed course sessions are unrecorded
    const initialBundle = buildPatientCourseAutoAssessments({
      patient,
      course,
      sessions,
      existingAssessments: [],
    });
    expect(initialBundle.autoResults).toHaveLength(2);
    expect(initialBundle.unrecordedResults).toHaveLength(2);
    expect(initialBundle.latestResult?.sourceSessionId).toBe('sess-aun-day2');

    for (const res of initialBundle.unrecordedResults) {
      const input = buildAssessmentInputFromCourseResult({ result: res, patient });
      const { created, record } = await upsertCourseAutoAssessment(input);
      expect(created).toBe(true);
      expect(record.source_channel).toBe('COURSE_AUTO_CALCULATED');
      expect(record.source_session_id).toBe(res.sourceSessionId);
    }

    const persistedAfterFirst = await listPatientAssessments(patient, {
      includeCrossFacility: true,
      excludeDemoRows: true,
      sessions,
    });
    expect(persistedAfterFirst).toHaveLength(2);
    expect(persistedAfterFirst.every((r) => r.source_channel === 'COURSE_AUTO_CALCULATED')).toBe(true);

    // Second sync pass: unrecordedResults MUST be empty and upsert MUST NOT duplicate
    const secondBundle = buildPatientCourseAutoAssessments({
      patient,
      course,
      sessions,
      existingAssessments: persistedAfterFirst,
    });
    expect(secondBundle.unrecordedResults).toHaveLength(0);

    for (const res of secondBundle.autoResults) {
      const input = buildAssessmentInputFromCourseResult({ result: res, patient });
      const { created } = await upsertCourseAutoAssessment(input);
      expect(created).toBe(false);
    }

    const persistedAfterSecond = await listPatientAssessments(patient, {
      includeCrossFacility: true,
      excludeDemoRows: true,
      sessions,
    });
    expect(persistedAfterSecond).toHaveLength(2);

    const summary = computeLongitudinalProgressSummary(persistedAfterSecond);
    expect(summary.totalVisits).toBe(2);
    expect(summary.courseAutoVisitCount).toBe(2);
    expect(summary.firstRom).toBe(62.0);
    expect(summary.latestRom).toBe(81.0);
    expect(summary.deltaRom).toBe(19.0);
  });

  it('enforces strict per-patient isolation for ครูอั๋น (HN-359954) across courses, schedules, and sessions', () => {
    const kruAun = makePatient({
      id: 'pat-359954',
      hn: 'HN-359954',
      fullName: 'ครูอั๋น',
      affectedSide: 'RIGHT',
      rehabSchedules: [
        {
          id: 'sched-aun-1',
          patientId: 'pat-359954',
          exerciseId: 'ELDERLY_SHOULDER_FLEXION',
          side: 'RIGHT',
          scheduledDate: '2026-11-22',
          sets: 2,
          reps: 10,
          status: 'COMPLETED',
        },
        {
          id: 'sched-leak-other',
          patientId: 'pat-other-999',
          exerciseId: 'ELDERLY_ELBOW_FLEXION',
          side: 'LEFT',
          scheduledDate: '2026-11-22',
          sets: 3,
          reps: 12,
          status: 'COMPLETED',
        },
      ],
    });

    const otherPatientCourse: PatientCourseRecord = {
      id: 'course-other-999',
      patientId: 'pat-other-999',
      title: 'คอร์สของผู้ป่วยรายอื่น (ห้ามรั่วไหล)',
      cognitiveModpackId: '',
      updatedAt: '2026-11-22T08:00:00.000Z',
      weeklySchedule: [
        {
          dayIndex: 1,
          dayType: 'training',
          activityType: 'physio_exercise',
          difficulty: 'hard',
          label: 'วันจันทร์',
          exercises: [
            {
              exerciseId: 'ELDERLY_SHOULDER_ABDUCTION',
              side: 'left',
              sets: 3,
              reps: 15,
              holdSec: 5,
              targetMinDeg: 120,
              targetMaxDeg: 140,
              allowedStarterSide: 'left',
            },
          ],
        },
      ],
    };

    const mixedSchedules: RehabScheduleItem[] = [
      {
        id: 'sched-other-table',
        patientId: 'pat-other-999',
        exerciseId: 'ELDERLY_WRIST_FLEXION',
        side: 'LEFT',
        scheduledDate: '2026-11-23',
        sets: 2,
        reps: 15,
        status: 'COMPLETED',
      },
    ];

    const mixedSessions: RehabSessionRecord[] = [
      {
        id: 'sess-aun-real',
        patientId: 'pat-359954',
        activityType: 'physio_exercise',
        exerciseId: 'ELDERLY_SHOULDER_FLEXION',
        exerciseName: 'งอข้อไหล่ด้านหน้า',
        bodySide: 'RIGHT',
        completedSets: 2,
        completedReps: 10,
        accuracyPercent: 90,
        maxRomDeg: 76.0,
        painStopTriggered: false,
        aiFeedback: 'ฝึกครบตามคอร์สครูอั๋น',
        kinematicsReplay: [],
        createdAt: '2026-11-23T09:15:00.000Z',
      },
      {
        id: 'sess-other-patient',
        patientId: 'pat-other-999',
        activityType: 'physio_exercise',
        exerciseId: 'ELDERLY_SHOULDER_ABDUCTION',
        exerciseName: 'กางแขนของผู้ป่วยคนอื่น',
        bodySide: 'LEFT',
        completedSets: 3,
        completedReps: 15,
        accuracyPercent: 100,
        maxRomDeg: 135.0,
        painStopTriggered: false,
        aiFeedback: 'ห้ามโผล่ในแฟ้มครูอั๋น',
        kinematicsReplay: [],
        createdAt: '2026-11-23T09:20:00.000Z',
      },
    ];

    const mixedHistory: RehabSessionHistoryEntry[] = [
      {
        id: 'hist-other-patient',
        patientId: 'pat-other-999',
        hn: 'HN-999111',
        sessionDate: '2026-11-21T10:00:00.000Z',
        exerciseCode: 'ELDERLY_ELBOW_FLEXION',
        exerciseName: 'งอศอกของผู้ป่วยอื่น',
        jointName: 'elbow',
        bodySide: 'LEFT',
        baselineRomDeg: 40,
        achievedMaxRomDeg: 110,
        romDeltaDeg: 70,
        validReps: 12,
        compensationsCount: 0,
        facilityName: 'สาขาอื่น',
      },
    ];

    const bundle = buildPatientCourseAutoAssessments({
      patient: kruAun,
      course: otherPatientCourse,
      rehabSchedules: mixedSchedules,
      sessions: mixedSessions,
      sessionHistory: mixedHistory,
      existingAssessments: [],
    });

    expect(bundle.patientId).toBe('pat-359954');
    expect(bundle.patientHn).toBe('HN-359954');
    expect(bundle.courseTitle).toBeNull();
    expect(bundle.assignedExercises.every((ex) => ex.exerciseCode === 'ELDERLY_SHOULDER_FLEXION')).toBe(
      true,
    );
    expect(bundle.completedSessions.every((s) => s.patientId === 'pat-359954')).toBe(true);
    expect(bundle.autoResults.every((r) => r.patientId === 'pat-359954')).toBe(true);
    expect(bundle.autoResults.some((r) => r.sourceSessionId === 'sess-other-patient')).toBe(false);
    expect(bundle.autoResults.some((r) => r.sourceSessionId === 'hist-other-patient')).toBe(false);
    expect(bundle.latestResult?.sourceSessionId).toBe('sess-aun-real');
  });

  it('auto-records COURSE_AUTO_CALCULATED assessment in CloudRouter POST /api/sessions with sourceSessionId deduplication and renders EHR auto-score card', async () => {
    const router = createCloudDelegatedRouter({});
    const loginRes = await router(
      new Request('https://drtechcares.local/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: 'admin', password: 'Admin@2026' }),
      }),
    );
    expect(loginRes.status).toBe(200);
    const { token } = (await loginRes.json()) as { token: string };

    // Create a dedicated test patient (ครูอั๋น HN-359954)
    const createPatRes = await router(
      new Request('https://drtechcares.local/api/patients', {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          fullName: 'ครูอั๋น',
          hn: 'HN-359954',
          age: 42,
          gender: 'ชาย',
          phone: '0893599540',
          diagnosis: 'Elderly Rehab Shoulder & Arm',
          affectedSide: 'RIGHT',
        }),
      }),
    );
    expect(createPatRes.status).toBe(201);
    const { patient: targetPatient } = (await createPatRes.json()) as { patient: PatientRecord };
    expect(targetPatient).toBeDefined();

    // Assign PT rehab course with target 90 deg and 10 reps
    const putCourseRes = await router(
      new Request(`https://drtechcares.local/api/courses/${targetPatient.id}`, {
        method: 'PUT',
        headers: {
          Authorization: `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          patientId: targetPatient.id,
          title: 'คอร์สฟื้นฟูหัวไหล่มาตรฐาน',
          cognitiveModpackId: 'pack-elderly-adl',
          weeklySchedule: [
            {
              dayIndex: 1,
              dayType: 'training',
              activityType: 'physio_exercise',
              difficulty: 'medium',
              label: 'วันจันทร์',
              exercises: [
                {
                  exerciseId: 'ELDERLY_SHOULDER_FLEXION',
                  side: 'right',
                  sets: 2,
                  reps: 10,
                  holdSec: 3,
                  targetMinDeg: 90,
                  targetMaxDeg: 110,
                  allowedStarterSide: 'right',
                },
              ],
            },
          ],
        }),
      }),
    );
    expect(putCourseRes.status).toBe(200);

    // Kiosk completes the assigned course session (no staff auth required on /api/sessions)
    const sessionBody = {
      patientId: targetPatient.id,
      activityType: 'physio_exercise',
      exerciseId: 'ELDERLY_SHOULDER_FLEXION',
      exerciseName: 'งอข้อไหล่ด้านหน้า (Shoulder Flexion)',
      bodySide: 'RIGHT',
      completedSets: 2,
      completedReps: 10,
      accuracyPercent: 90,
      maxRomDeg: 76.0,
      targetReps: 10,
      targetRomDeg: 90,
      sourceSessionId: 'kiosk-course-unique-001',
      painStopTriggered: false,
      aiFeedback: 'ทำครบตามคอร์ส',
      kinematicsReplay: [],
    };

    const sessRes1 = await router(
      new Request('https://drtechcares.local/api/sessions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(sessionBody),
      }),
    );
    expect(sessRes1.status).toBe(201);

    // Submitting duplicate sourceSessionId must NOT create a second assessment row
    const sessRes2 = await router(
      new Request('https://drtechcares.local/api/sessions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(sessionBody),
      }),
    );
    expect(sessRes2.status).toBe(201);

    const assessRes = await router(
      new Request(`https://drtechcares.local/api/patients/${targetPatient.id}/assessments`, {
        headers: { Authorization: `Bearer ${token}` },
      }),
    );
    expect(assessRes.status).toBe(200);
    const { assessments } = (await assessRes.json()) as {
      assessments: {
        source_channel: string;
        source_session_id?: string | null;
        initial_score_0_to_6: number;
        measured_rom_deg: number;
        clinical_handover_note: string;
      }[];
    };
    const autoRows = assessments.filter((a) => a.source_session_id === 'kiosk-course-unique-001');
    expect(autoRows).toHaveLength(1);
    expect(autoRows[0]?.source_channel).toBe('COURSE_AUTO_CALCULATED');
    expect(autoRows[0]?.initial_score_0_to_6).toBe(5);
    expect(autoRows[0]?.measured_rom_deg).toBe(76.0);
    expect(autoRows[0]?.clinical_handover_note).toContain('ระดับ 5/6');

    // Render PatientClinicalRecordPanel statically with ครูอั๋น's assigned course & session
    const kruAun = makePatient({
      id: 'pat-359954-render',
      hn: 'HN-359954',
      fullName: 'ครูอั๋น',
      affectedSide: 'RIGHT',
    });
    const ptUser: UserAccount = {
      id: 'pt-admin-1',
      username: 'admin',
      role: 'super_admin',
      fullName: 'นพ.สมชาย',
      phone: '0811111111',
      active: true,
      createdAt: '2026-11-01T00:00:00.000Z',
    };
    const html = renderToStaticMarkup(
      React.createElement(PatientClinicalRecordPanel, {
        patients: [kruAun],
        sessions: [
          {
            id: 'sess-render-1',
            patientId: kruAun.id,
            activityType: 'physio_exercise',
            exerciseId: 'ELDERLY_SHOULDER_FLEXION',
            exerciseName: 'งอข้อไหล่ด้านหน้า',
            bodySide: 'RIGHT',
            completedSets: 2,
            completedReps: 10,
            accuracyPercent: 90,
            maxRomDeg: 76.0,
            painStopTriggered: false,
            aiFeedback: 'ควบคุมได้ดี',
            kinematicsReplay: [],
            createdAt: '2026-11-23T09:00:00.000Z',
          },
        ],
        currentUser: ptUser,
        selectedPatientId: kruAun.id,
        onSelectPatientId: () => {},
        patientCourseOverride: {
          id: 'course-render-1',
          patientId: kruAun.id,
          title: 'คอร์สฟื้นฟูหัวไหล่ครูอั๋น',
          cognitiveModpackId: '',
          updatedAt: '2026-11-23T08:00:00.000Z',
          weeklySchedule: [
            {
              dayIndex: 1,
              dayType: 'training',
              activityType: 'physio_exercise',
              difficulty: 'medium',
              label: 'วันจันทร์',
              exercises: [
                {
                  exerciseId: 'ELDERLY_SHOULDER_FLEXION',
                  side: 'right',
                  sets: 2,
                  reps: 10,
                  holdSec: 3,
                  targetMinDeg: 90,
                  targetMaxDeg: 110,
                  allowedStarterSide: 'right',
                },
              ],
            },
          ],
        },
      }),
    );

    expect(html).toContain('คอร์สกายภาพที่นักกายภาพจัดให้ &amp; คำนวณคะแนนอัตโนมัติ');
    expect(html).toContain('ดึงผลเล่นคอร์สและบันทึกอัตโนมัติ');
    expect(html).toContain('ระดับ 5/6');
    expect(html).toContain('ROM 76.0°');
    expect(EMOJI_RE.test(html)).toBe(false);
  });

  it('contains zero Unicode emojis in all new and modified course-scoring files', () => {
    const root = process.cwd();
    const filesToAudit = [
      'src/utils/courseClinicalScoring.ts',
      'src/components/PatientClinicalRecordPanel.tsx',
      'src/lib/clinicalAssessmentStore.ts',
      'src/types/clinicalAssessment.ts',
      'src/components/SmartRehabKioskUI.tsx',
      'src/App.tsx',
      'packages/api-client/src/index.ts',
      'packages/api-client/src/cloudRouter.ts',
      'packages/api-client/src/supabasePersistence.ts',
      'supabase/migrations/012_course_auto_clinical_assessments.sql',
      'src/__tests__/courseClinicalScoring.test.ts',
    ];

    for (const relPath of filesToAudit) {
      const content = fs.readFileSync(path.join(root, relPath), 'utf8');
      expect(EMOJI_RE.test(content), `Found forbidden Unicode emoji in ${relPath}`).toBe(false);
    }
  });
});
