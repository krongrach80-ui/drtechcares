import fs from 'node:fs';
import path from 'node:path';
import { beforeEach, describe, expect, it } from 'vitest';
import {
  MAHIDOL_ASSESSMENT_SCALE_0_TO_6,
  computeLongitudinalProgressSummary,
  filterAssessmentsForPatient,
  isValidMahidolScore,
  resolvePatientUniversalLinkCode,
} from '../types/clinicalAssessment';
import {
  createPatientAssessment,
  ensureDemoAssessmentsForPatient,
  listPatientAssessments,
  resetClinicalAssessmentStoreForTests,
  syncKioskSessionToAssessment,
} from '../lib/clinicalAssessmentStore';

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
        exercise_code: 'STROKE_SHOULDER_FLEXION',
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
        exercise_code: 'STROKE_SHOULDER_FLEXION',
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
      exercise_code: 'STROKE_SHOULDER_FLEXION',
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
      exercise_code: 'STROKE_SHOULDER_ABDUCTION',
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
      exercise_code: 'STROKE_SHOULDER_FLEXION',
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
      exercise_code: 'STROKE_SHOULDER_FLEXION',
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
      exerciseCode: 'STROKE_SHOULDER_FLEXION',
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
      diagnosis: 'Stroke Hemiparesis',
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
      'src/components/PatientClinicalRecordPanel.tsx',
      'src/components/ClinicalStaffPortalShell.tsx',
      'src/components/SmartRehabKioskUI.tsx',
      'src/App.tsx',
    ];
    for (const rel of filesToAudit) {
      const content = fs.readFileSync(path.join(root, rel), 'utf8');
      expect(EMOJI_RE.test(content), `Unexpected Unicode emoji in ${rel}`).toBe(false);
    }
  });
});

