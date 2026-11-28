import fs from 'node:fs';
import path from 'node:path';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import type { PatientCourseRecord, PatientRecord, RehabSessionRecord } from '@drtechcares/api-client';
import { CLINICAL_EXERCISES } from '@drtechcares/pose-engine';
import { ClinicalQrCodeSvg, generateQrMatrix } from '../utils/qrCodeSvg';
import { buildPatientCourseAutoAssessments } from '../utils/courseClinicalScoring';
import { buildRehabStockIndex } from '../utils/rehabStockScoreEngine';
import { buildDoctorClinicalBrief } from '../utils/doctorClinicalBrief';
import {
  computeLongitudinalProgressSummary,
  type PatientClinicalAssessmentRecord,
} from '../types/clinicalAssessment';
import { buildUnifiedRehabMinigameSummary } from '../utils/unifiedRehabMinigameSummary';
import { ClinicalAiTelemetryHud } from '../components/ClinicalAiTelemetryHud';
import { SessionVictoryCelebrationCard } from '../components/SessionVictoryCelebrationCard';
import { CaregiverLineAlertPreview } from '../components/CaregiverLineAlertPreview';
import { ClinicalPdfReportModal } from '../components/ClinicalPdfReportModal';
import { PatientClinicalRecordPanel } from '../components/PatientClinicalRecordPanel';
import { RehabStockPerformanceCard } from '../components/RehabStockPerformanceCard';
import { buildAll15ClinicalExercisePrescriptions } from '../hooks/useKioskSocket';

const EMOJI_PATTERN = /[\u{1F300}-\u{1FAFF}\u{2600}-\u{26FF}\u{2700}-\u{27BF}]/u;

function buildSamplePatientFixture(): {
  patient: PatientRecord;
  course: PatientCourseRecord;
  singleDaySessions: RehabSessionRecord[];
  singleDayAssessments: PatientClinicalAssessmentRecord[];
} {
  const patient: PatientRecord = {
    id: 'pat-real-01',
    hn: 'HN-823960',
    fullName: 'นางสมศรี ใจสุข',
    age: 68,
    gender: 'หญิง',
    phone: '089-456-7812',
    diagnosis: 'ฟื้นฟูข้อไหล่และแขนขวา',
    affectedSide: 'RIGHT',
    motorAssessmentScore: 6,
    universalHealthLinkCode: 'DTC-EHR-823960',
    assignedPtId: 'pt-1',
    delegatedToPtId: null,
    delegationNote: null,
    nextAppointmentDate: '2026-12-05',
    registrationStatus: 'active',
    enrolledAt: '2026-11-27T08:00:00.000Z',
  };

  const course: PatientCourseRecord = {
    id: 'course-real-01',
    patientId: patient.id,
    title: 'คอร์สฟื้นฟูแขน 4 ท่า',
    cognitiveModpackId: '',
    updatedAt: '2026-11-27T09:00:00.000Z',
    weeklySchedule: [
      {
        dayIndex: 1,
        dayType: 'training',
        activityType: 'physio_exercise',
        difficulty: 'medium',
        label: 'วันฝึกหลัก',
        exercises: [
          { exerciseId: 'ELDERLY_SH_FLEX', side: 'right', sets: 1, reps: 8, holdSec: 3, targetMinDeg: 90, targetMaxDeg: 180, allowedStarterSide: 'right' },
          { exerciseId: 'ELDERLY_SH_ABD', side: 'right', sets: 1, reps: 8, holdSec: 3, targetMinDeg: 85, targetMaxDeg: 180, allowedStarterSide: 'right' },
          { exerciseId: 'ELDERLY_ELB_FLEX_EXT', side: 'right', sets: 1, reps: 8, holdSec: 2, targetMinDeg: 95, targetMaxDeg: 180, allowedStarterSide: 'right' },
          { exerciseId: 'ELDERLY_HAND_TO_HEAD', side: 'right', sets: 1, reps: 8, holdSec: 3, targetMinDeg: 100, targetMaxDeg: 180, allowedStarterSide: 'right' },
        ],
      },
    ],
  };

  const singleDaySessions: RehabSessionRecord[] = [
    {
      id: 'sess-real-1',
      patientId: patient.id,
      activityType: 'physio_exercise',
      exerciseId: 'ELDERLY_SH_FLEX_RIGHT',
      exerciseName: 'ยกแขนไปด้านหน้า',
      bodySide: 'RIGHT',
      completedSets: 1,
      completedReps: 8,
      accuracyPercent: 94,
      maxRomDeg: 165,
      painStopTriggered: false,
      aiFeedback: 'ดีมาก',
      kinematicsReplay: [],
      createdAt: '2026-11-27T09:05:00.000Z',
    },
    {
      id: 'sess-real-2',
      patientId: patient.id,
      activityType: 'physio_exercise',
      exerciseId: 'ELDERLY_SH_ABD_RIGHT',
      exerciseName: 'กางแขนด้านข้าง',
      bodySide: 'RIGHT',
      completedSets: 1,
      completedReps: 8,
      accuracyPercent: 92,
      maxRomDeg: 160,
      painStopTriggered: false,
      aiFeedback: 'ดีมาก',
      kinematicsReplay: [],
      createdAt: '2026-11-27T09:10:00.000Z',
    },
    {
      id: 'sess-real-3',
      patientId: patient.id,
      activityType: 'physio_exercise',
      exerciseId: 'ELDERLY_ELB_FLEX_EXT_RIGHT',
      exerciseName: 'งอ-เหยียดข้อศอก',
      bodySide: 'RIGHT',
      completedSets: 1,
      completedReps: 8,
      accuracyPercent: 95,
      maxRomDeg: 155,
      painStopTriggered: false,
      aiFeedback: 'ดีมาก',
      kinematicsReplay: [],
      createdAt: '2026-11-27T09:15:00.000Z',
    },
  ];

  const singleDayAssessments: PatientClinicalAssessmentRecord[] = singleDaySessions.map((s, idx) => ({
    id: `assess-real-${idx + 1}`,
    patient_id: patient.id,
    patient_hn: patient.hn,
    universal_link_code: 'DTC-EHR-823960',
    visit_number: idx + 1,
    assessed_at: s.createdAt,
    facility_name: 'คลินิกกายภาพบำบัด รพ.ศูนย์ฟื้นฟูผู้สูงอายุ',
    assessor_staff_id: 'pt-1',
    assessor_staff_name: 'กภ. พิมพ์ชนก สุขสบาย',
    exercise_code: s.exerciseId.replace(/_RIGHT$/, ''),
    exercise_name_th: s.exerciseName,
    target_side: 'RIGHT',
    initial_score_0_to_6: 6,
    measured_rom_deg: s.maxRomDeg,
    target_rom_deg: 180,
    chief_complaint: 'ติดตามผลจริง',
    clinical_handover_note: 'บันทึกจากตู้ Kiosk',
    next_session_goal: 'คงมาตรฐานต่อเนื่อง',
    source_channel: 'KIOSK_SESSION_AUTO_SYNC',
    completed_reps: s.completedReps,
    accuracy_percent: s.accuracyPercent,
    session_id: s.id,
    source_session_id: s.id,
    created_at: s.createdAt,
  }));

  return { patient, course, singleDaySessions, singleDayAssessments };
}

describe('Competition Judge-Wow Features (QR, PDF, Real Stock Trend, AI Pro HUD, Victory, LINE Care & 180° Arms)', () => {
  it('generates a valid ISO/IEC 18004 QR Code matrix with 3 intact 7x7 finder patterns and crisp SVG', () => {
    const url = 'https://drtechcares.app/ehr/DTC-EHR-823960?hn=HN-823960';
    const matrix = generateQrMatrix(url);
    const n = matrix.length;
    expect(n).toBeGreaterThanOrEqual(29);
    expect(matrix.every((row) => row.length === n)).toBe(true);

    const checkFinder = (topR: number, leftC: number) => {
      for (let dr = 0; dr < 7; dr++) {
        for (let dc = 0; dc < 7; dc++) {
          const dist = Math.max(Math.abs(dr - 3), Math.abs(dc - 3));
          const expectedDark = dist !== 2;
          expect(matrix[topR + dr][leftC + dc]).toBe(expectedDark);
        }
      }
    };
    checkFinder(0, 0);
    checkFinder(0, n - 7);
    checkFinder(n - 7, 0);

    const svgHtml = renderToStaticMarkup(
      React.createElement(ClinicalQrCodeSvg, { value: url, size: 96 }),
    );
    expect(svgHtml).toContain('data-testid="clinical-qr-code-svg"');
    expect(svgHtml).toContain('<path');
    expect(svgHtml).toContain('shape-rendering="crispEdges"');
  });

  it('renders the real-data Stock Area/Line Trend Chart even on 1 training day (cumulative real exercises) and prints PDF + QR without any 7-day demo mode', () => {
    const { patient, course, singleDaySessions, singleDayAssessments } = buildSamplePatientFixture();
    const courseAuto = buildPatientCourseAutoAssessments({
      patient,
      course,
      rehabSchedules: null,
      sessions: singleDaySessions,
      sessionHistory: null,
      existingAssessments: singleDayAssessments,
    });
    const stock = buildRehabStockIndex({
      assignedExercises: courseAuto.assignedExercises,
      completedSessions: courseAuto.completedSessions,
      assessments: singleDayAssessments,
    });
    expect(stock.points).toHaveLength(1);

    // Even with 1 real training day, RehabStockPerformanceCard defaults to the real Stock Area/Line Chart (zero 2-bar fallback!)
    const stockHtml = renderToStaticMarkup(
      React.createElement(RehabStockPerformanceCard, {
        patientName: patient.fullName,
        patientHn: patient.hn,
        stockIndex: stock,
      }),
    );
    expect(stockHtml).toContain('data-testid="stock-area-polyline"');
    expect(stockHtml).toContain('data-testid="stock-ma3-polyline"');
    expect(stockHtml).toContain('เริ่มฝึก');
    expect(stockHtml).toContain('ท่าที่ 1');
    expect(stockHtml).toContain('ท่าที่ 3');

    const longitudinal = computeLongitudinalProgressSummary(singleDayAssessments, 'ALL');
    const unified = buildUnifiedRehabMinigameSummary({
      patientId: patient.id,
      longitudinalSummary: longitudinal,
      stockIndex: stock,
      sessions: singleDaySessions,
    });
    const brief = buildDoctorClinicalBrief({
      stockIndex: stock,
      longitudinalSummary: longitudinal,
      unifiedSummary: unified,
      exerciseBreakdown: longitudinal.exerciseBreakdown,
      assessments: singleDayAssessments,
    });

    const pdfHtml = renderToStaticMarkup(
      React.createElement(ClinicalPdfReportModal, {
        open: true,
        onClose: () => {},
        patient,
        universalLinkCode: 'DTC-EHR-823960',
        facilityName: 'คลินิกกายภาพบำบัด รพ.ศูนย์ฟื้นฟูผู้สูงอายุ',
        assessorName: 'กภ. พิมพ์ชนก สุขสบาย',
        rehabStockIndex: stock,
        unifiedSummary: unified,
        doctorBrief: brief,
        exerciseBreakdown: longitudinal.exerciseBreakdown,
        firstMahidolScore: 2,
        latestMahidolScore: 6,
      }),
    );
    expect(pdfHtml).toContain('data-testid="clinical-pdf-report-modal"');
    expect(pdfHtml).toContain('data-testid="clinical-pdf-print-btn"');
    expect(pdfHtml).toContain('data-testid="clinical-qr-code-svg"');
    expect(pdfHtml).toContain('HN-823960');
    expect(pdfHtml).toContain('แรกเข้า 2/6 -&gt; ล่าสุด 6/6');

    const panelHtml = renderToStaticMarkup(
      React.createElement(PatientClinicalRecordPanel, {
        patients: [patient],
        sessions: singleDaySessions,
        currentUser: {
          id: 'pt-1',
          username: 'pt1',
          role: 'pt',
          fullName: 'กภ. พิมพ์ชนก สุขสบาย',
          phone: '081-000-0000',
          active: true,
          createdAt: '2026-01-01T00:00:00.000Z',
        },
        selectedPatientId: patient.id,
        onSelectPatientId: () => {},
        initialRecordsOverride: singleDayAssessments,
        patientCourseOverride: course,
      }),
    );
    expect(panelHtml).toContain('data-testid="ehr-open-pdf-report-btn"');
    expect(panelHtml).toContain('พิมพ์ PDF');
    expect(panelHtml).toContain('data-testid="stock-area-polyline"');
    expect(panelHtml).not.toContain('ehr-toggle-showcase-demo-btn');
    expect(panelHtml).not.toContain('เคสเดโม 7 วัน');
  });

  it('renders AI Biomechanics Pro HUD, Session Victory Celebration Card, and LINE Care Alert Preview cleanly', () => {
    const hudHtml = renderToStaticMarkup(
      React.createElement(ClinicalAiTelemetryHud, {
        visible: true,
        onClose: () => {},
        exerciseNameTh: 'ยกแขนไปด้านหน้า',
        targetRomDeg: 180,
        currentRomDeg: 162.4,
        kinematics: {
          elbowAngleDeg: 171,
          shoulderHikeRatio: 0.05,
          compensations: [],
          xyzVector: { x: 0.12, y: -0.78, z: -0.45 },
          handBiomechanics: {
            left: null,
            right: {
              side: 'right',
              usedWorld3D: true,
              extendedFingerCount: 5,
              meanFingerExtensionPct: 92,
              fingerSplayDeg: 18,
              fingers: {
                thumb: { finger: 'thumb', mcpAngleDeg: 168, pipAngleDeg: 170, dipAngleDeg: 172, extensionPct: 90, isExtended: true, tipToWristRatio: 1.4 },
                index: { finger: 'index', mcpAngleDeg: 172, pipAngleDeg: 174, dipAngleDeg: 175, extensionPct: 95, isExtended: true, tipToWristRatio: 1.85 },
                middle: { finger: 'middle', mcpAngleDeg: 171, pipAngleDeg: 173, dipAngleDeg: 174, extensionPct: 94, isExtended: true, tipToWristRatio: 1.9 },
                ring: { finger: 'ring', mcpAngleDeg: 169, pipAngleDeg: 170, dipAngleDeg: 171, extensionPct: 91, isExtended: true, tipToWristRatio: 1.82 },
                pinky: { finger: 'pinky', mcpAngleDeg: 166, pipAngleDeg: 168, dipAngleDeg: 169, extensionPct: 89, isExtended: true, tipToWristRatio: 1.65 },
              },
              pinchRatios: { index: 0.45, middle: 0.52, ring: 0.61, pinky: 0.68 },
              activePinchFinger: null,
              activePinchFingerTh: null,
              minPinchRatio: 0.45,
              pinchAngleScoreDeg: 22,
              wristExtensionDeg: 42,
              forearmRollSignedDeg: 18,
              palmOrientation: 'SUPINATED_UP',
              palmOrientationTh: 'หงายมือ',
              palmWidth: 0.08,
              palmLength: 0.11,
            },
            active: null,
          },
        },
        rotationEval: {
          effectiveRotationRomDeg: 64,
          signedAngleDeg: 64,
          sweptArcDeg: 82,
          rotationVerified: true,
          returnedToStart: false,
          guidanceBannerTh: null,
          phaseLabelTh: 'หมุนจริงผ่าน',
        },
        distanceM: 2.1,
        wrongSideRaised: false,
        wrongPlaneRaised: false,
        trunkTiltDeg: 3.4,
      }),
    );
    expect(hudHtml).toContain('data-testid="clinical-ai-pro-telemetry-hud"');
    expect(hudHtml).toContain('AI BIOMECHANICS PRO HUD');
    expect(hudHtml).toContain('162.4°');
    expect(hudHtml).toContain('หมุนจริงผ่าน');

    const victoryHtml = renderToStaticMarkup(
      React.createElement(SessionVictoryCelebrationCard, {
        open: true,
        patientName: 'นางสมศรี ใจสุข',
        exerciseNameTh: 'ยกแขนไปด้านหน้า',
        completedReps: 8,
        targetReps: 8,
        peakRomDeg: 174,
        targetRomDeg: 180,
        previousBaselineRomDeg: 150,
        hasNextExercise: true,
        onNextExercise: () => {},
        onRepeatExercise: () => {},
        onSendCaregiverAlert: () => {},
        onClose: () => {},
      }),
    );
    expect(victoryHtml).toContain('data-testid="session-victory-celebration-modal"');
    expect(victoryHtml).toContain('นางสมศรี ใจสุข');
    expect(victoryHtml).toContain('+24° ดีขึ้น!');

    const lineHtml = renderToStaticMarkup(
      React.createElement(CaregiverLineAlertPreview, {
        alert: {
          id: 'alert-1',
          kind: 'SESSION_COMPLETE',
          titleTh: 'ฝึกกายภาพสำเร็จครบเซต!',
          bodyTh: 'คุณสมศรี ทำครบ 8/8 ครั้ง มุมสูงสุด 174°',
          patientName: 'นางสมศรี ใจสุข',
          metricBadgeTh: '174° • ครบเซต',
          timestampLabel: '09:30 น.',
        },
        onDismiss: () => {},
      }),
    );
    expect(lineHtml).toContain('data-testid="caregiver-line-alert-preview"');
    expect(lineHtml).toContain('Dr.Tech.Care Family Bot (LINE Care)');
  });

  it('enforces strict ZERO Unicode emojis across all newly added modules and ZERO ctx.shadowBlur > 0 in Kiosk', () => {
    const filesToAudit = [
      'src/utils/qrCodeSvg.ts',
      'src/components/ClinicalAiTelemetryHud.tsx',
      'src/components/SessionVictoryCelebrationCard.tsx',
      'src/components/CaregiverLineAlertPreview.tsx',
      'src/components/ClinicalPdfReportModal.tsx',
      'src/components/RehabStockPerformanceCard.tsx',
    ];
    for (const rel of filesToAudit) {
      const content = fs.readFileSync(path.resolve(process.cwd(), rel), 'utf8');
      expect(EMOJI_PATTERN.test(content), `Found forbidden Unicode emoji in ${rel}`).toBe(false);
    }

    const kioskSrc = fs.readFileSync(
      path.resolve(process.cwd(), 'src/components/SmartRehabKioskUI.tsx'),
      'utf8',
    );
    expect(/ctx\.shadowBlur\s*=\s*[1-9]/.test(kioskSrc)).toBe(false);
  });

  it('sets target ROM to 180 deg for all 8 arm exercises while preserving natural angles for wrist, shoulder-only, and neck exercises', () => {
    const armExerciseCodes = new Set([
      'ELDERLY_SH_FLEX',
      'ELDERLY_SH_ABD',
      'ELDERLY_CROSS_BODY',
      'ELDERLY_SCAP_PROTRACT',
      'ELDERLY_ELB_FLEX_EXT',
      'ELDERLY_HAND_TO_HEAD',
      'ELDERLY_CLASPED_ARM_RAISE',
      'ELDERLY_CLASPED_FORWARD_PUSH',
    ]);
    const nonArmCodes = new Set([
      'ELDERLY_SH_ROT',
      'ELDERLY_FOREARM_TURN',
      'ELDERLY_WRIST_FINGER',
      'ELDERLY_THUMB_PINCH',
      'MOPH_NECK_TILT',
      'MOPH_NECK_TURN',
      'MOPH_SHOULDER_SHRUG_ROLL',
    ]);

    for (const ex of CLINICAL_EXERCISES) {
      if (armExerciseCodes.has(ex.code)) {
        expect(ex.detection.targetRomDeg, `${ex.code} targetRomDeg`).toBe(180);
        expect(ex.detection.targetMaxDeg, `${ex.code} targetMaxDeg`).toBe(180);
      } else if (nonArmCodes.has(ex.code)) {
        expect(ex.detection.targetRomDeg ?? ex.detection.targetMaxDeg, `${ex.code} non-arm`).toBeLessThan(100);
      }
    }

    const all15 = buildAll15ClinicalExercisePrescriptions();
    for (const rx of all15) {
      const code = rx.base_code ?? rx.catalog_exercise_id ?? '';
      if (armExerciseCodes.has(code)) {
        expect(rx.target_rom_deg, `${code} kiosk prescription`).toBe(180);
      } else if (nonArmCodes.has(code)) {
        expect(rx.target_rom_deg, `${code} non-arm kiosk prescription`).toBeLessThan(100);
      }
    }
  });
});
