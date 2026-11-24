import fs from 'node:fs';
import path from 'node:path';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import type { PatientCourseRecord, PatientRecord, RehabSessionRecord, UserAccount } from '@drtechcares/api-client';
import {
  buildRehabStockIndex,
  computeIntelligentExerciseQualityMultiplier,
} from '../utils/rehabStockScoreEngine';
import type {
  AssignedCourseExerciseDefinition,
  NormalizedCompletedCourseSession,
} from '../utils/courseClinicalScoring';
import { PatientClinicalRecordPanel } from '../components/PatientClinicalRecordPanel';

const EMOJI_RE = /[\p{Extended_Pictographic}\u{FE0F}\u{2600}-\u{27BF}\u{23E9}-\u{23FA}]/u;

function makeAssignedExercise(
  code: string,
  nameTh: string,
  targetReps = 10,
  targetRomDeg = 90,
): AssignedCourseExerciseDefinition {
  return {
    key: `${code}::RIGHT`,
    exerciseCode: code,
    exerciseNameTh: nameTh,
    targetSide: 'RIGHT',
    targetRomDeg,
    targetReps,
    targetSets: 1,
    courseTitle: 'คอร์สฟื้นฟูหลังโรคหลอดเลือดสมอง',
    sources: ['patient_courses'],
    totalScheduledCount: 1,
    completedScheduledCount: 1,
  };
}

function makeCompletedSession(
  id: string,
  code: string,
  nameTh: string,
  assessedAt: string,
  completedReps: number,
  achievedRomDeg: number,
  accuracyScore = 95,
): NormalizedCompletedCourseSession {
  return {
    sourceSessionId: id,
    patientId: 'pat-stock-01',
    patientHn: 'HN-382121',
    exerciseCode: code,
    exerciseNameTh: nameTh,
    targetSide: 'RIGHT',
    achievedRomDeg,
    completedReps,
    completedSets: 1,
    accuracyScore,
    assessedAt,
    facilityName: 'ศูนย์กายภาพบำบัด Dr.Tech.Care สาขาหลัก',
    origin: 'rehab_sessions',
  };
}

describe('Intelligent Stock-Market-Style Total Rehab Percentage & Full-Score Engine', () => {
  it('calculates exact full score for 1 exercise (10 reps = 10 pts) and 5 doctor-assigned exercises (5 x 10 reps = 50 pts)', () => {
    // Case A: 1 exercise x 10 reps -> full score = 10 points -> completing 10/10 = 10/10 (100.0%)
    const oneExercise = [
      makeAssignedExercise('STROKE_SHOULDER_FLEXION', 'ยกแขนไปด้านหน้า', 10, 90),
    ];
    const oneSession = [
      makeCompletedSession(
        's-1ex-d1',
        'STROKE_SHOULDER_FLEXION',
        'ยกแขนไปด้านหน้า',
        '2026-11-20T09:00:00.000Z',
        10,
        90,
        96,
      ),
    ];
    const singleIndex = buildRehabStockIndex({
      assignedExercises: oneExercise,
      completedSessions: oneSession,
      assessments: [],
    });
    expect(singleIndex.totalPrescribedExercises).toBe(1);
    expect(singleIndex.maxTotalScore).toBe(10);
    expect(singleIndex.latestPoint?.rawEarnedScore).toBe(10);
    expect(singleIndex.latestPoint?.smartEarnedScore).toBe(10);
    expect(singleIndex.latestPoint?.rawTotalPct).toBe(100);
    expect(singleIndex.latestPoint?.smartTotalPct).toBe(100);

    // Case B: 5 doctor-assigned exercises x 10 reps -> full score = 50 points!
    const fiveExercises: AssignedCourseExerciseDefinition[] = [
      makeAssignedExercise('STROKE_SHOULDER_FLEXION', 'ยกแขนไปด้านหน้า', 10, 90),
      makeAssignedExercise('STROKE_SHOULDER_ABDUCTION', 'กางแขนออกด้านข้าง', 10, 85),
      makeAssignedExercise('STROKE_ELBOW_FLEXION', 'งอข้อศอก', 10, 110),
      makeAssignedExercise('STROKE_WRIST_FLEXION', 'กระดกข้อมือ', 10, 60),
      makeAssignedExercise('STROKE_SHOULDER_EXTERNAL_ROTATION', 'หมุนข้อไหล่ออกด้านนอก', 10, 75),
    ];

    const multiDaySessions: NormalizedCompletedCourseSession[] = [
      // Day 1 (Nov 20): plays ONLY 1 of 5 exercises (10/10 reps) -> 10 / 50 = 20.0%
      makeCompletedSession(
        's-d1-1',
        'STROKE_SHOULDER_FLEXION',
        'ยกแขนไปด้านหน้า',
        '2026-11-20T09:00:00.000Z',
        10,
        90,
        95,
      ),
      // Day 2 (Nov 22): plays 3 of 5 exercises (10/10 reps each) -> 30 / 50 = 60.0% (+40.0%)
      makeCompletedSession(
        's-d2-1',
        'STROKE_SHOULDER_FLEXION',
        'ยกแขนไปด้านหน้า',
        '2026-11-22T09:00:00.000Z',
        10,
        90,
        95,
      ),
      makeCompletedSession(
        's-d2-2',
        'STROKE_SHOULDER_ABDUCTION',
        'กางแขนออกด้านข้าง',
        '2026-11-22T09:10:00.000Z',
        10,
        85,
        92,
      ),
      makeCompletedSession(
        's-d2-3',
        'STROKE_ELBOW_FLEXION',
        'งอข้อศอก',
        '2026-11-22T09:20:00.000Z',
        10,
        110,
        94,
      ),
      // Day 3 (Nov 24): plays ALL 5 exercises (10/10 reps each) -> 50 / 50 = 100.0% (+40.0%)
      makeCompletedSession(
        's-d3-1',
        'STROKE_SHOULDER_FLEXION',
        'ยกแขนไปด้านหน้า',
        '2026-11-24T09:00:00.000Z',
        10,
        92,
        98,
      ),
      makeCompletedSession(
        's-d3-2',
        'STROKE_SHOULDER_ABDUCTION',
        'กางแขนออกด้านข้าง',
        '2026-11-24T09:05:00.000Z',
        10,
        88,
        96,
      ),
      makeCompletedSession(
        's-d3-3',
        'STROKE_ELBOW_FLEXION',
        'งอข้อศอก',
        '2026-11-24T09:10:00.000Z',
        10,
        112,
        95,
      ),
      makeCompletedSession(
        's-d3-4',
        'STROKE_WRIST_FLEXION',
        'กระดกข้อมือ',
        '2026-11-24T09:15:00.000Z',
        10,
        62,
        94,
      ),
      makeCompletedSession(
        's-d3-5',
        'STROKE_SHOULDER_EXTERNAL_ROTATION',
        'หมุนข้อไหล่ออกด้านนอก',
        '2026-11-24T09:20:00.000Z',
        10,
        78,
        95,
      ),
    ];

    const stockIndex = buildRehabStockIndex({
      assignedExercises: fiveExercises,
      completedSessions: multiDaySessions,
      assessments: [],
    });

    expect(stockIndex.totalPrescribedExercises).toBe(5);
    expect(stockIndex.maxTotalScore).toBe(50);
    expect(stockIndex.points).toHaveLength(3);

    // Day 1: 10 / 50 = 20.0%
    expect(stockIndex.points[0].smartEarnedScore).toBe(10);
    expect(stockIndex.points[0].smartTotalPct).toBe(20);
    expect(stockIndex.points[0].completedExercisesCount).toBe(1);
    expect(stockIndex.points[0].volumeReps).toBe(10);

    // Day 2: 30 / 50 = 60.0%, delta = +40.0% (+20 pts), trend UP
    expect(stockIndex.points[1].smartEarnedScore).toBe(30);
    expect(stockIndex.points[1].smartTotalPct).toBe(60);
    expect(stockIndex.points[1].deltaSmartPct).toBe(40);
    expect(stockIndex.points[1].deltaSmartPoints).toBe(20);
    expect(stockIndex.points[1].trend).toBe('UP');
    expect(stockIndex.points[1].completedExercisesCount).toBe(3);

    // Day 3: 50 / 50 = 100.0%, MA3 = (20 + 60 + 100)/3 = 60.0%
    expect(stockIndex.points[2].smartEarnedScore).toBe(50);
    expect(stockIndex.points[2].smartTotalPct).toBe(100);
    expect(stockIndex.points[2].ma3SmartPct).toBe(60);
    expect(stockIndex.points[2].volumeReps).toBe(50);
    expect(stockIndex.allTimeHighSmartPct).toBe(100);
    expect(stockIndex.allTimeLowSmartPct).toBe(20);
    expect(stockIndex.overallTrend).toBe('UP');
  });

  it('intelligently distinguishes shallow ROM / low-accuracy reps from full clinical execution', () => {
    const fullQuality = computeIntelligentExerciseQualityMultiplier({
      achievedRomDeg: 86,
      targetRomDeg: 90, // 95.5% >= 90%
      accuracyPct: 92, // >= 85%
    });
    expect(fullQuality.qualityMultiplier).toBe(1.0);

    const shallowQuality = computeIntelligentExerciseQualityMultiplier({
      achievedRomDeg: 45, // 50% ROM depth
      targetRomDeg: 90,
      accuracyPct: 60,
    });
    expect(shallowQuality.qualityMultiplier).toBeLessThan(0.85);
    expect(shallowQuality.qualityMultiplier).toBeGreaterThanOrEqual(0.5);

    const stock = buildRehabStockIndex({
      assignedExercises: [
        makeAssignedExercise('STROKE_SHOULDER_FLEXION', 'ยกแขนไปด้านหน้า', 10, 90),
      ],
      completedSessions: [
        makeCompletedSession(
          's-shallow',
          'STROKE_SHOULDER_FLEXION',
          'ยกแขนไปด้านหน้า',
          '2026-11-24T08:00:00.000Z',
          10,
          45,
          60,
        ),
      ],
      assessments: [],
    });
    // Raw mode counts 10/10 reps = 100%, while intelligent mode reflects 50% ROM depth & 60% accuracy
    expect(stock.latestPoint?.rawEarnedScore).toBe(10);
    expect(stock.latestPoint?.rawTotalPct).toBe(100);
    expect(stock.latestPoint?.smartEarnedScore).toBeLessThan(8.5);
    expect(stock.latestPoint?.smartTotalPct).toBeLessThan(85);
  });

  it('renders RehabStockPerformanceCard inside PatientClinicalRecordPanel with 50-point course pool and zero emojis', () => {
    const patient: PatientRecord = {
      id: 'pat-bankj',
      hn: 'HN-382121',
      fullName: 'แบงค์เจ',
      age: 45,
      gender: 'ชาย',
      phone: '0812345678',
      diagnosis: 'Stroke ADL (Shoulder & Arm)',
      affectedSide: 'RIGHT',
      assignedPtId: 'pt-1',
      delegatedToPtId: null,
      delegationNote: null,
      nextAppointmentDate: '2026-11-30',
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
    const course5: PatientCourseRecord = {
      id: 'course-5ex',
      patientId: patient.id,
      title: 'คอร์สฟื้นฟู 5 ท่ามาตรฐาน',
      cognitiveModpackId: '',
      updatedAt: '2026-11-20T00:00:00.000Z',
      weeklySchedule: [
        {
          dayIndex: 1,
          dayType: 'training',
          activityType: 'physio_exercise',
          difficulty: 'medium',
          label: 'วันจันทร์',
          exercises: [
            { exerciseId: 'STROKE_SHOULDER_FLEXION', side: 'right', sets: 1, reps: 10, holdSec: 3, targetMinDeg: 90, targetMaxDeg: 110, allowedStarterSide: 'right' },
            { exerciseId: 'STROKE_SHOULDER_ABDUCTION', side: 'right', sets: 1, reps: 10, holdSec: 3, targetMinDeg: 85, targetMaxDeg: 105, allowedStarterSide: 'right' },
            { exerciseId: 'STROKE_ELBOW_FLEXION', side: 'right', sets: 1, reps: 10, holdSec: 3, targetMinDeg: 110, targetMaxDeg: 130, allowedStarterSide: 'right' },
            { exerciseId: 'STROKE_WRIST_FLEXION', side: 'right', sets: 1, reps: 10, holdSec: 3, targetMinDeg: 60, targetMaxDeg: 75, allowedStarterSide: 'right' },
            { exerciseId: 'STROKE_SHOULDER_EXTERNAL_ROTATION', side: 'right', sets: 1, reps: 10, holdSec: 3, targetMinDeg: 75, targetMaxDeg: 90, allowedStarterSide: 'right' },
          ],
        },
      ],
    };
    const sessions: RehabSessionRecord[] = [
      {
        id: 'sess-b1',
        patientId: patient.id,
        activityType: 'physio_exercise',
        exerciseId: 'STROKE_SHOULDER_FLEXION',
        exerciseName: 'ยกแขนไปด้านหน้า',
        bodySide: 'RIGHT',
        completedSets: 1,
        completedReps: 10,
        accuracyPercent: 95,
        maxRomDeg: 90,
        painStopTriggered: false,
        aiFeedback: 'ดีมาก',
        kinematicsReplay: [],
        createdAt: '2026-11-22T09:00:00.000Z',
      },
      {
        id: 'sess-b2',
        patientId: patient.id,
        activityType: 'physio_exercise',
        exerciseId: 'STROKE_SHOULDER_FLEXION',
        exerciseName: 'ยกแขนไปด้านหน้า',
        bodySide: 'RIGHT',
        completedSets: 1,
        completedReps: 10,
        accuracyPercent: 96,
        maxRomDeg: 92,
        painStopTriggered: false,
        aiFeedback: 'ดีมาก',
        kinematicsReplay: [],
        createdAt: '2026-11-24T09:00:00.000Z',
      },
      {
        id: 'sess-b3',
        patientId: patient.id,
        activityType: 'physio_exercise',
        exerciseId: 'STROKE_SHOULDER_ABDUCTION',
        exerciseName: 'กางแขนออกด้านข้าง',
        bodySide: 'RIGHT',
        completedSets: 1,
        completedReps: 10,
        accuracyPercent: 92,
        maxRomDeg: 85,
        painStopTriggered: false,
        aiFeedback: 'ดีมาก',
        kinematicsReplay: [],
        createdAt: '2026-11-24T09:10:00.000Z',
      },
    ];

    const html = renderToStaticMarkup(
      React.createElement(PatientClinicalRecordPanel, {
        patients: [patient],
        sessions,
        currentUser: admin,
        selectedPatientId: patient.id,
        onSelectPatientId: () => {},
        patientCourseOverride: course5,
      }),
    );

    expect(html).toContain('data-testid="rehab-stock-performance-card"');
    expect(html).toContain('REHAB INDEX • HN-382121');
    expect(html).toContain('คะแนนเต็มรวม 50 คะแนน (5 ท่าที่แพทย์/PT กำหนด)');
    expect(html).toContain('40.0%');
    expect(html).toContain('20 / 50 คะแนน');
    expect(html).toContain('กราฟพื้นที่ (Area)');
    expect(html).toContain('แท่งเทียน (Candle)');
    expect(html).toContain('อัจฉริยะ (ROM+แม่นยำ)');
    expect(html).toContain('นับครั้งตรง (1 ครั้ง=1 แต้ม)');
    expect(html).toContain('data-testid="stock-area-polyline"');
    expect(html).toContain('data-testid="stock-ma3-polyline"');
    expect(EMOJI_RE.test(html)).toBe(false);

    const root = process.cwd();
    const filesToAudit = [
      'src/utils/rehabStockScoreEngine.ts',
      'src/components/RehabStockPerformanceCard.tsx',
      'src/components/PatientClinicalRecordPanel.tsx',
      'src/__tests__/rehabStockScoreEngine.test.ts',
    ];
    for (const rel of filesToAudit) {
      const content = fs.readFileSync(path.join(root, rel), 'utf8');
      expect(EMOJI_RE.test(content), `Unexpected Unicode emoji in ${rel}`).toBe(false);
    }
  });
});

