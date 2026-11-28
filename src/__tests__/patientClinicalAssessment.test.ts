import fs from 'node:fs';
import path from 'node:path';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { beforeEach, describe, expect, it } from 'vitest';
import type { PatientRecord, UserAccount } from '@drtechcares/api-client';
import {
  MAHIDOL_ASSESSMENT_SCALE_0_TO_6,
  computeLongitudinalProgressSummary,
  filterAssessmentsForPatient,
  isValidMahidolScore,
  resolvePatientUniversalLinkCode,
  type PatientClinicalAssessmentRecord,
} from '../types/clinicalAssessment';
import {
  createPatientAssessment,
  ensureDemoAssessmentsForPatient,
  listPatientAssessments,
  resetClinicalAssessmentStoreForTests,
  syncKioskSessionToAssessment,
} from '../lib/clinicalAssessmentStore';
import {
  CLINICAL_EXERCISE_CHART_PALETTE,
  buildMultiExerciseChartSeries,
  computeClinicalMilestones,
  computeMonthlyScoreSummary,
  formatThaiClinicalDate,
  formatThaiFullDateOnly,
  formatThaiMonthYear,
  formatThaiShortDate,
} from '../utils/clinicalChartAnalytics';
import { PatientClinicalRecordPanel } from '../components/PatientClinicalRecordPanel';

const EMOJI_RE = /[\p{Extended_Pictographic}\u{FE0F}\u{2600}-\u{27BF}\u{23E9}-\u{23FA}]/u;

describe('Individual Patient Clinical Assessment (0-6 Scale), ROM Comparison & Cross-Facility Linking', () => {
  beforeEach(() => {
    resetClinicalAssessmentStoreForTests();
  });

  it('defines all 7 Mahidol-adapted levels (0-6) and strictly validates 0..6 integer bounds', () => {
    expect(MAHIDOL_ASSESSMENT_SCALE_0_TO_6).toHaveLength(7);
    expect(MAHIDOL_ASSESSMENT_SCALE_0_TO_6.map((item) => item.score)).toEqual([0, 1, 2, 3, 4, 5, 6]);
    for (const item of MAHIDOL_ASSESSMENT_SCALE_0_TO_6) {
      expect(item.descriptionTh).toContain(`ระดับ ${item.score}`);
      expect(EMOJI_RE.test(item.descriptionTh)).toBe(false);
    }

    expect(isValidMahidolScore(0)).toBe(true);
    expect(isValidMahidolScore(6)).toBe(true);
    expect(isValidMahidolScore(-1)).toBe(false);
    expect(isValidMahidolScore(7)).toBe(false);
    expect(isValidMahidolScore(3.5)).toBe(false);
    expect(isValidMahidolScore(Number.NaN)).toBe(false);
  });

  it('rejects out-of-range (<0 or >6) scores in createPatientAssessment', async () => {
    await expect(
      createPatientAssessment({
        patient_id: 'pat-invalid',
        patient_hn: 'HN-9901',
        universal_link_code: 'DTC-EHR-9901',
        facility_name: 'ศูนย์กายภาพบำบัด Dr.Tech.Care สาขาหลัก',
        assessor_staff_id: 'pt-1',
        assessor_staff_name: 'กภ. ทดสอบ',
        exercise_code: 'ELDERLY_SHOULDER_FLEXION',
        exercise_name_th: 'ยกแขนตรงไปด้านหน้า',
        target_side: 'RIGHT',
        initial_score_0_to_6: -1,
        measured_rom_deg: 30,
        target_rom_deg: 125,
        chief_complaint: 'อ่อนแรง',
        clinical_handover_note: 'บันทึก',
        next_session_goal: '45 องศา',
      }),
    ).rejects.toThrow();

    await expect(
      createPatientAssessment({
        patient_id: 'pat-invalid',
        patient_hn: 'HN-9901',
        universal_link_code: 'DTC-EHR-9901',
        facility_name: 'ศูนย์กายภาพบำบัด Dr.Tech.Care สาขาหลัก',
        assessor_staff_id: 'pt-1',
        assessor_staff_name: 'กภ. ทดสอบ',
        exercise_code: 'ELDERLY_SHOULDER_FLEXION',
        exercise_name_th: 'ยกแขนตรงไปด้านหน้า',
        target_side: 'RIGHT',
        initial_score_0_to_6: 7,
        measured_rom_deg: 30,
        target_rom_deg: 125,
        chief_complaint: 'อ่อนแรง',
        clinical_handover_note: 'บันทึก',
        next_session_goal: '45 องศา',
      }),
    ).rejects.toThrow();
  });

  it('enforces strict per-patient record isolation (no cross-patient leakage)', async () => {
    const patientA = {
      id: 'pat-alpha',
      hn: 'HN-2026-0001',
      universalHealthLinkCode: 'DTC-EHR-ALPHA01',
    };
    const patientB = {
      id: 'pat-beta',
      hn: 'HN-2026-0002',
      universalHealthLinkCode: 'DTC-EHR-BETA02',
    };

    await createPatientAssessment({
      patient_id: patientA.id,
      patient_hn: patientA.hn,
      universal_link_code: patientA.universalHealthLinkCode,
      facility_name: 'สาขาหลัก',
      assessor_staff_id: 'pt-1',
      assessor_staff_name: 'กภ. สมชาย',
      exercise_code: 'ELDERLY_SHOULDER_FLEXION',
      exercise_name_th: 'ยกแขนตรงไปด้านหน้า',
      target_side: 'RIGHT',
      initial_score_0_to_6: 2,
      measured_rom_deg: 35,
      target_rom_deg: 125,
      chief_complaint: 'อาการของคนไข้ A',
      clinical_handover_note: 'ส่งต่อเฉพาะ A',
      next_session_goal: '50 องศา',
    });

    await createPatientAssessment({
      patient_id: patientB.id,
      patient_hn: patientB.hn,
      universal_link_code: patientB.universalHealthLinkCode,
      facility_name: 'สาขาชุมชน',
      assessor_staff_id: 'pt-2',
      assessor_staff_name: 'กภ. สุภา',
      exercise_code: 'ELDERLY_SHOULDER_ABDUCTION',
      exercise_name_th: 'กางแขนออกด้านข้าง',
      target_side: 'LEFT',
      initial_score_0_to_6: 4,
      measured_rom_deg: 82,
      target_rom_deg: 110,
      chief_complaint: 'อาการของคนไข้ B',
      clinical_handover_note: 'ส่งต่อเฉพาะ B',
      next_session_goal: '95 องศา',
    });

    const listA = await listPatientAssessments(patientA);
    const listB = await listPatientAssessments(patientB);

    expect(listA).toHaveLength(1);
    expect(listA[0]?.patient_id).toBe('pat-alpha');
    expect(listA[0]?.clinical_handover_note).toBe('ส่งต่อเฉพาะ A');

    expect(listB).toHaveLength(1);
    expect(listB[0]?.patient_id).toBe('pat-beta');
    expect(listB[0]?.clinical_handover_note).toBe('ส่งต่อเฉพาะ B');
  });

  it('computes accurate ROM delta (+25.0 deg) and aggregates cross-facility records sharing universal_link_code', async () => {
    const universalCode = 'DTC-EHR-XBRANCH9';
    const branch1Patient = {
      id: 'pat-branch1-id',
      hn: 'HN-1010',
      universalHealthLinkCode: universalCode,
    };

    await createPatientAssessment({
      patient_id: 'pat-branch1-id',
      patient_hn: 'HN-1010',
      universal_link_code: universalCode,
      assessed_at: '2026-11-01T09:00:00.000Z',
      facility_name: 'ศูนย์กายภาพบำบัด Dr.Tech.Care สาขาหลัก',
      assessor_staff_id: 'pt-main',
      assessor_staff_name: 'กภ. สาขาหลัก',
      exercise_code: 'ELDERLY_SHOULDER_FLEXION',
      exercise_name_th: 'ยกแขนตรงไปด้านหน้า',
      target_side: 'RIGHT',
      initial_score_0_to_6: 2,
      measured_rom_deg: 30.0,
      target_rom_deg: 125.0,
      chief_complaint: 'แรกรับที่สาขาหลัก',
      clinical_handover_note: 'ประคองข้อไหล่',
      next_session_goal: '45 องศา',
    });

    // Visit 2 recorded under a different local branch patient_id, linked via universal_link_code
    await createPatientAssessment({
      patient_id: 'pat-branch2-local-alias',
      patient_hn: 'HN-COMM-88',
      universal_link_code: universalCode,
      assessed_at: '2026-11-15T10:30:00.000Z',
      facility_name: 'คลินิกเครือข่ายกายภาพชุมชน',
      assessor_staff_id: 'pt-comm',
      assessor_staff_name: 'กภ. สาขาชุมชน',
      exercise_code: 'ELDERLY_SHOULDER_FLEXION',
      exercise_name_th: 'ยกแขนตรงไปด้านหน้า',
      target_side: 'RIGHT',
      initial_score_0_to_6: 4,
      measured_rom_deg: 55.0,
      target_rom_deg: 125.0,
      chief_complaint: 'ตรวจติดตามที่สาขาชุมชน',
      clinical_handover_note: 'ควบคุมศอกได้ดีขึ้น',
      next_session_goal: '70 องศา',
    });

    const unified = await listPatientAssessments(branch1Patient, { includeCrossFacility: true });
    expect(unified).toHaveLength(2);

    const summary = computeLongitudinalProgressSummary(unified);
    expect(summary.totalVisits).toBe(2);
    expect(summary.firstRom).toBe(30.0);
    expect(summary.latestRom).toBe(55.0);
    expect(summary.deltaRom).toBe(25.0);
    expect(summary.percentGain).toBe(83.3);
    expect(summary.firstScore).toBe(2);
    expect(summary.latestScore).toBe(4);
    expect(summary.deltaScore).toBe(2);
    expect(summary.linkedFacilities).toEqual([
      'ศูนย์กายภาพบำบัด Dr.Tech.Care สาขาหลัก',
      'คลินิกเครือข่ายกายภาพชุมชน',
    ]);
    expect(summary.visitsWithDelta[1]?.deltaFromPrevRom).toBe(25.0);

    // Kiosk auto-sync appends Visit 3 and reflects new peak ROM
    await syncKioskSessionToAssessment({
      patientId: branch1Patient.id,
      patientHn: branch1Patient.hn,
      universalHealthLinkCode: universalCode,
      exerciseCode: 'ELDERLY_SHOULDER_FLEXION',
      exerciseNameTh: 'ยกแขนตรงไปด้านหน้า',
      targetSide: 'RIGHT',
      measuredRomDeg: 72.5,
      targetRomDeg: 125.0,
      completedReps: 10,
    });

    const afterKiosk = await listPatientAssessments(branch1Patient, { includeCrossFacility: true });
    expect(afterKiosk).toHaveLength(3);
    expect(afterKiosk[2]?.source_channel).toBe('KIOSK_SESSION_AUTO_SYNC');
    expect(computeLongitudinalProgressSummary(afterKiosk).deltaRom).toBe(42.5);
  });

  it('seeds deterministic Thai demo progression across 2 facilities when requested and maintains zero emojis', () => {
    const demoPatient = {
      id: 'pat-demo-thai',
      hn: 'HN-2026-0014',
      fullName: 'คุณสมชาย ใจดี',
      affectedSide: 'RIGHT',
      diagnosis: 'Elderly Upper Limb Weakness',
    };
    const seeded = ensureDemoAssessmentsForPatient(demoPatient);
    expect(seeded).toHaveLength(3);
    const summary = computeLongitudinalProgressSummary(
      filterAssessmentsForPatient(seeded, demoPatient),
    );
    expect(summary.firstRom).toBe(30.0);
    expect(summary.latestRom).toBe(62.0);
    expect(summary.deltaRom).toBe(32.0);
    expect(summary.linkedFacilities.length).toBeGreaterThanOrEqual(2);
    expect(resolvePatientUniversalLinkCode(demoPatient)).toMatch(/^DTC-EHR-/);

    const root = process.cwd();
    const filesToAudit = [
      'supabase/migrations/008_patient_clinical_assessments.sql',
      'src/types/clinicalAssessment.ts',
      'src/lib/clinicalAssessmentStore.ts',
      'src/utils/clinicalChartAnalytics.ts',
      'src/components/PatientClinicalRecordPanel.tsx',
      'src/components/ClinicalStaffPortalShell.tsx',
      'src/components/SmartRehabKioskUI.tsx',
      'src/App.tsx',
      'src/__tests__/patientClinicalAssessment.test.ts',
    ];
    for (const rel of filesToAudit) {
      const content = fs.readFileSync(path.join(root, rel), 'utf8');
      expect(EMOJI_RE.test(content), `Unexpected Unicode emoji in ${rel}`).toBe(false);
    }
  });

  it('groups multi-exercise sessions into distinct color-coded series, identifies Previous/Latest/Best milestones with Thai dates, and computes Monthly Score Summary accurately', () => {
    // 1. Thai Buddhist Era date formatting (วัน/เดือน/ปี พ.ศ.)
    const sampleIso = '2026-11-23T09:40:00.000Z';
    expect(formatThaiFullDateOnly(sampleIso)).toBe('23 พ.ย. 2569');
    expect(formatThaiShortDate(sampleIso)).toBe('23 พ.ย. 69');
    expect(formatThaiClinicalDate(sampleIso)).toMatch(/^23 พ\.ย\. 2569 • \d{2}:\d{2} น\.$/);
    expect(formatThaiMonthYear('2026-09', 'short')).toBe('ก.ย. 2569');
    expect(formatThaiMonthYear('2026-10', 'short')).toBe('ต.ค. 2569');
    expect(formatThaiMonthYear('2026-11', 'full')).toBe('พฤศจิกายน 2569');

    const baseRec = {
      patient_id: 'pat-multi-01',
      patient_hn: 'HN-796334',
      universal_link_code: 'DTC-EHR-9862D8DD',
      facility_name: 'ศูนย์กายภาพบำบัด Dr.Tech.Care สาขาหลัก',
      assessor_staff_id: 'SYSTEM_COURSE_AUTO_CALC',
      assessor_staff_name: 'ระบบคำนวณอัตโนมัติจากคอร์สกายภาพ',
      target_side: 'RIGHT' as const,
      pain_vas_0_to_10: null,
      chief_complaint: 'ฝึกตามคอร์สกายภาพ',
      clinical_handover_note: 'บันทึกอัตโนมัติ',
      next_session_goal: 'ฝึกต่อเนื่อง',
      source_channel: 'COURSE_AUTO_CALCULATED' as const,
      measurement_method: 'COURSE_AUTO_CALCULATED' as const,
    };

    const multiExerciseRecords: PatientClinicalAssessmentRecord[] = [
      // Sep 2026: 2 visits (avg score = 3.0)
      {
        ...baseRec,
        id: 'rec-1',
        visit_number: 1,
        assessed_at: '2026-09-10T09:00:00.000Z',
        exercise_code: 'ELDERLY_SHOULDER_FLEXION',
        exercise_name_th: 'ยกแขนไปด้านหน้า',
        initial_score_0_to_6: 3,
        measured_rom_deg: 55.0,
        target_rom_deg: 90.0,
      },
      {
        ...baseRec,
        id: 'rec-2',
        visit_number: 2,
        assessed_at: '2026-09-20T09:00:00.000Z',
        exercise_code: 'ELDERLY_SHOULDER_ABDUCTION',
        exercise_name_th: 'กางแขนออกด้านข้าง',
        initial_score_0_to_6: 3,
        measured_rom_deg: 51.0,
        target_rom_deg: 85.0,
      },
      // Oct 2026: 2 visits (avg score = 4.5, delta = +1.5)
      {
        ...baseRec,
        id: 'rec-3',
        visit_number: 3,
        assessed_at: '2026-10-05T10:00:00.000Z',
        exercise_code: 'ELDERLY_SHOULDER_FLEXION',
        exercise_name_th: 'ยกแขนไปด้านหน้า',
        initial_score_0_to_6: 4,
        measured_rom_deg: 72.0,
        target_rom_deg: 90.0,
      },
      {
        ...baseRec,
        id: 'rec-4',
        visit_number: 4,
        assessed_at: '2026-10-18T10:00:00.000Z',
        exercise_code: 'ELDERLY_ELBOW_FLEXION',
        exercise_name_th: 'งอข้อศอก',
        initial_score_0_to_6: 5,
        measured_rom_deg: 96.0,
        target_rom_deg: 110.0,
      },
      // Nov 2026: 3 visits -> rec-5 is Personal Best (6/6, 120/90=133.3%), rec-6 is Previous (5/6), rec-7 is Latest (5/6)
      // Avg score in Nov = (6 + 5 + 5) / 3 = 5.3, delta vs Oct = +0.8
      {
        ...baseRec,
        id: 'rec-5',
        visit_number: 5,
        assessed_at: '2026-11-08T09:15:00.000Z',
        exercise_code: 'ELDERLY_SHOULDER_FLEXION',
        exercise_name_th: 'ยกแขนไปด้านหน้า',
        initial_score_0_to_6: 6,
        measured_rom_deg: 120.0,
        target_rom_deg: 90.0,
      },
      {
        ...baseRec,
        id: 'rec-6',
        visit_number: 6,
        assessed_at: '2026-11-18T09:30:00.000Z',
        exercise_code: 'ELDERLY_SHOULDER_ABDUCTION',
        exercise_name_th: 'กางแขนออกด้านข้าง',
        initial_score_0_to_6: 5,
        measured_rom_deg: 78.0,
        target_rom_deg: 85.0,
      },
      {
        ...baseRec,
        id: 'rec-7',
        visit_number: 7,
        assessed_at: '2026-11-23T09:40:00.000Z',
        exercise_code: 'ELDERLY_ELBOW_FLEXION',
        exercise_name_th: 'งอข้อศอก',
        initial_score_0_to_6: 5,
        measured_rom_deg: 102.0,
        target_rom_deg: 110.0,
      },
    ];

    const summary = computeLongitudinalProgressSummary(multiExerciseRecords, 'ALL');

    // 2. Multi-exercise color-coded series grouping (3 distinct exercises -> 3 separate series)
    const grouping = buildMultiExerciseChartSeries(summary.visitsWithDelta);
    expect(grouping.series).toHaveLength(3);
    expect(grouping.series.map((s) => s.exerciseCode)).toEqual([
      'ELDERLY_SHOULDER_FLEXION',
      'ELDERLY_SHOULDER_ABDUCTION',
      'ELDERLY_ELBOW_FLEXION',
    ]);
    expect(grouping.series[0].color).toBe(CLINICAL_EXERCISE_CHART_PALETTE[0]);
    expect(grouping.series[1].color).toBe(CLINICAL_EXERCISE_CHART_PALETTE[1]);
    expect(grouping.series[2].color).toBe(CLINICAL_EXERCISE_CHART_PALETTE[2]);
    expect(grouping.series[0].points.map((p) => p.recordId)).toEqual(['rec-1', 'rec-3', 'rec-5']);
    expect(grouping.series[1].points.map((p) => p.recordId)).toEqual(['rec-2', 'rec-6']);
    expect(grouping.series[2].points.map((p) => p.recordId)).toEqual(['rec-4', 'rec-7']);

    // 3. 3-Milestone identification (Previous, Latest, Personal Best)
    const milestones = computeClinicalMilestones(summary.visitsWithDelta);
    expect(milestones.previous?.record.id).toBe('rec-6');
    expect(milestones.latest?.record.id).toBe('rec-7');
    expect(milestones.best?.record.id).toBe('rec-5'); // 6/6 beats latest 5/6!
    expect(milestones.latestDeltaRomFromPrevious).toBe(24.0);
    expect(milestones.latestDeltaScoreFromPrevious).toBe(0);

    // 4. Monthly Score Summary aggregation
    const monthly = computeMonthlyScoreSummary(multiExerciseRecords);
    expect(monthly).toHaveLength(3);
    expect(monthly[0]).toMatchObject({
      monthKey: '2026-09',
      monthLabelShortTh: 'ก.ย. 2569',
      monthLabelFullTh: 'กันยายน 2569',
      avgScore0To6: 3.0,
      maxScore0To6: 3,
      sessionCount: 2,
      monthDeltaScore: null,
    });
    expect(monthly[1]).toMatchObject({
      monthKey: '2026-10',
      monthLabelShortTh: 'ต.ค. 2569',
      monthLabelFullTh: 'ตุลาคม 2569',
      avgScore0To6: 4.5,
      maxScore0To6: 5,
      sessionCount: 2,
      monthDeltaScore: 1.5,
    });
    expect(monthly[2]).toMatchObject({
      monthKey: '2026-11',
      monthLabelShortTh: 'พ.ย. 2569',
      monthLabelFullTh: 'พฤศจิกายน 2569',
      avgScore0To6: 5.3,
      maxScore0To6: 6,
      sessionCount: 3,
      monthDeltaScore: 0.8,
    });
    expect(monthly[2].avgTargetPct).toBeGreaterThan(100);

    // 5. Render PatientClinicalRecordPanel statically and verify all 3 upgraded visual modules + zero emojis
    const testPatient: PatientRecord = {
      id: 'pat-multi-01',
      hn: 'HN-796334',
      fullName: 'คุณทดสอบ หลายท่า',
      age: 45,
      gender: 'ชาย',
      phone: '0890001234',
      diagnosis: 'Elderly Rehab (Shoulder & Arm)',
      affectedSide: 'RIGHT',
      assignedPtId: 'pt-01',
      delegatedToPtId: null,
      delegationNote: null,
      nextAppointmentDate: '2026-11-30',
      registrationStatus: 'active',
      enrolledAt: '2026-09-01T00:00:00.000Z',
    };
    const adminUser: UserAccount = {
      id: 'admin-01',
      username: 'admin',
      role: 'super_admin',
      fullName: 'นพ.สมชาย',
      phone: '02-000-0001',
      active: true,
      createdAt: '2026-09-01T00:00:00.000Z',
    };

    const html = renderToStaticMarkup(
      React.createElement(PatientClinicalRecordPanel, {
        patients: [testPatient],
        sessions: [],
        currentUser: adminUser,
        selectedPatientId: testPatient.id,
        onSelectPatientId: () => {},
        initialRecordsOverride: multiExerciseRecords,
      }),
    );

    // Milestone cards with Thai dates
    expect(html).toContain('data-testid="clinical-milestone-navigator"');
    expect(html).toContain('ครั้งก่อนหน้า');
    expect(html).toContain('ครั้งล่าสุด');
    expect(html).toContain('ครั้งที่ดีที่สุด');
    expect(html).toContain('23 พ.ย. 2569');
    expect(html).toContain('23 พ.ย. 69');

    // Separate color-coded polylines for each of the 3 exercises + interactive legend
    expect(html).toContain('data-testid="multi-exercise-chart-legend"');
    expect(html).toContain('แสดงทุกท่าพร้อมกัน (3 ท่า)');
    expect(html).toContain('data-testid="exercise-polyline-ELDERLY_SHOULDER_FLEXION"');
    expect(html).toContain('data-testid="exercise-polyline-ELDERLY_SHOULDER_ABDUCTION"');
    expect(html).toContain('data-testid="exercise-polyline-ELDERLY_ELBOW_FLEXION"');

    // Monthly Score Summary tab & breakdown card
    expect(html).toContain('กราฟสรุปรายเดือน (ผลคะแนน)');
    expect(html).toContain('data-testid="monthly-score-summary-card"');
    expect(html).toContain('กันยายน 2569');
    expect(html).toContain('ตุลาคม 2569');
    expect(html).toContain('พฤศจิกายน 2569');
    expect(EMOJI_RE.test(html)).toBe(false);
  });
});

