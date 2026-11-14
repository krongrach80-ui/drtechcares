import { describe, expect, it } from 'vitest';
import { POSE_INDEX } from '@drtechcares/pose-engine';
import type { NormalizedLandmark } from '@mediapipe/tasks-vision';
import {
  EXERCISE_OPTIONS,
  getPrescribedLimbPromptTh,
  mapTodayExercisesToPrescriptions,
  synthesizeDemoPayload,
} from '../hooks/useKioskSocket';
import { measureLiveJointAngleAndTilt } from '../components/SmartRehabKioskUI';

function createNeutralLandmarks(
  overrides: Partial<Record<number, NormalizedLandmark>> = {},
): NormalizedLandmark[] {
  const pts: NormalizedLandmark[] = Array.from({ length: 33 }, () => ({
    x: 0.5,
    y: 0.5,
    z: 0,
    visibility: 0.99,
  }));
  pts[POSE_INDEX.LEFT_SHOULDER] = { x: 0.58, y: 0.35, z: 0, visibility: 0.99 };
  pts[POSE_INDEX.RIGHT_SHOULDER] = { x: 0.42, y: 0.35, z: 0, visibility: 0.99 };
  pts[POSE_INDEX.LEFT_ELBOW] = { x: 0.60, y: 0.52, z: 0, visibility: 0.99 };
  pts[POSE_INDEX.RIGHT_ELBOW] = { x: 0.40, y: 0.52, z: 0, visibility: 0.99 };
  pts[POSE_INDEX.LEFT_WRIST] = { x: 0.61, y: 0.68, z: 0, visibility: 0.99 };
  pts[POSE_INDEX.RIGHT_WRIST] = { x: 0.39, y: 0.68, z: 0, visibility: 0.99 };
  pts[POSE_INDEX.LEFT_HIP] = { x: 0.56, y: 0.65, z: 0, visibility: 0.99 };
  pts[POSE_INDEX.RIGHT_HIP] = { x: 0.44, y: 0.65, z: 0, visibility: 0.99 };
  for (const [idxStr, lm] of Object.entries(overrides)) {
    if (lm) pts[Number(idxStr)] = lm;
  }
  return pts;
}

describe('synthesizeDemoPayload (Smart Rehabilitation Kiosk Simulator)', () => {
  const rightShoulder = EXERCISE_OPTIONS[0];

  it('produces AUTH_FRAMING during FACE_SCAN before and after match', () => {
    const scanning = synthesizeDemoPayload({
      preset: 'FACE_SCAN',
      romDeg: 10,
      trunkTiltDeg: 1,
      distanceM: 2.0,
      exercise: rightShoulder,
      validReps: 0,
      invalidReps: 0,
      holdRemainingSec: 3,
      faceMatched: false,
    });
    expect(scanning.kiosk_state).toBe('AUTH_FRAMING');
    expect(scanning.patient).toBeNull();
    expect(scanning.ui_directives.highlight_color).toBe('AMBER');

    const verified = synthesizeDemoPayload({
      preset: 'FACE_SCAN',
      romDeg: 10,
      trunkTiltDeg: 1,
      distanceM: 2.0,
      exercise: rightShoulder,
      validReps: 0,
      invalidReps: 0,
      holdRemainingSec: 3,
      faceMatched: true,
    });
    expect(verified.patient?.id).toBe('PT-8841');
    expect(verified.ui_directives.highlight_color).toBe('GREEN');
  });

  it('triggers TRUNK_LEAN warm AMBER warning when trunk tilt > 10 deg', () => {
    const res = synthesizeDemoPayload({
      preset: 'ACTIVE_REP',
      romDeg: 120,
      trunkTiltDeg: 15.5,
      distanceM: 2.0,
      exercise: rightShoulder,
      validReps: 2,
      invalidReps: 0,
      holdRemainingSec: 3,
    });
    expect(res.kiosk_state).toBe('ACTIVE_REP');
    expect(res.telemetry_analysis.is_form_valid).toBe(false);
    expect(res.telemetry_analysis.compensation_detected).toContain('TRUNK_LEAN');
    expect(res.ui_directives.skeleton_highlight_joint).toBe('spine');
    expect(res.ui_directives.highlight_color).toBe('AMBER');
  });

  it('enters HOLD_PHASE at target ROM, celebrates REP_COMPLETED / SET_FINISHED, and halts on EMERGENCY_HALT', () => {
    const hold = synthesizeDemoPayload({
      preset: 'HOLD_PHASE',
      romDeg: 150,
      trunkTiltDeg: 2,
      distanceM: 2.0,
      exercise: rightShoulder,
      validReps: 4,
      invalidReps: 0,
      holdRemainingSec: 2.1,
    });
    expect(hold.kiosk_state).toBe('HOLD_PHASE');
    expect(hold.ui_directives.highlight_color).toBe('GREEN');
    expect(hold.rep_counter.hold_remaining_sec).toBe(2.1);

    const completed = synthesizeDemoPayload({
      preset: 'REP_COMPLETED',
      romDeg: 150,
      trunkTiltDeg: 1,
      distanceM: 2.0,
      exercise: rightShoulder,
      validReps: 5,
      invalidReps: 0,
      holdRemainingSec: 0,
    });
    expect(completed.kiosk_state).toBe('REP_COMPLETED');
    expect(completed.rep_counter.valid_reps).toBe(5);

    const finished = synthesizeDemoPayload({
      preset: 'SET_FINISHED',
      romDeg: 150,
      trunkTiltDeg: 1,
      distanceM: 2.0,
      exercise: rightShoulder,
      validReps: 10,
      invalidReps: 0,
      holdRemainingSec: 0,
    });
    expect(finished.kiosk_state).toBe('SET_FINISHED');
    expect(finished.rep_counter.valid_reps).toBe(10);

    const halt = synthesizeDemoPayload({
      preset: 'EMERGENCY_HALT',
      romDeg: 60,
      trunkTiltDeg: 4,
      distanceM: 2.0,
      exercise: rightShoulder,
      validReps: 4,
      invalidReps: 0,
      holdRemainingSec: 3,
    });
    expect(halt.kiosk_state).toBe('EMERGENCY_HALT');
    expect(halt.telemetry_analysis.pain_alert).toBe(true);
    expect(halt.ui_directives.highlight_color).toBe('RED');
  });

  it('maps today prescribed physio exercises and applies authenticated patient override', () => {
    const mapped = mapTodayExercisesToPrescriptions([
      {
        id: 'sch-1',
        exerciseId: 'ex-01',
        nameTh: 'ยกแขนตรงไปด้านหน้า (ขวา)',
        nameEn: 'Right Shoulder Flexion',
        targetJoint: 'ข้อไหล่ (Shoulder)',
        activityType: 'physio_exercise',
        sets: 3,
        reps: 10,
        holdSec: 3,
        status: 'ACTIVE',
      },
      {
        id: 'sch-2',
        exerciseId: 'ex-03',
        nameTh: 'กางแขนตรงออกด้านข้าง (ขวา)',
        nameEn: 'Right Shoulder Abduction',
        targetJoint: 'ข้อไหล่ (Shoulder)',
        activityType: 'physio_exercise',
        sets: 3,
        reps: 8,
        holdSec: 3,
        status: 'ACTIVE',
      },
      {
        id: 'sch-mini',
        exerciseId: 'cognitive-card-game',
        nameTh: 'มินิเกมฝึกสมอง',
        nameEn: 'Card Mini-Game',
        targetJoint: 'Cognitive',
        activityType: 'minigame',
        sets: 1,
        reps: 5,
        holdSec: 3,
        status: 'ACTIVE',
      },
    ]);

    expect(mapped).toHaveLength(2);
    expect(mapped[0].exercise_id).toBe('sch-1');
    expect(mapped[0].catalog_exercise_id).toBe('ex-01');
    expect(mapped[0].target_joint).toBe('right_shoulder');
    expect(mapped[0].target_reps).toBe(10);
    expect(mapped[0].target_sets).toBe(3);
    expect(mapped[1].exercise_id).toBe('sch-2');
    expect(mapped[1].target_joint).toBe('right_shoulder');
    expect(mapped[1].target_reps).toBe(8);

    const patientPayload = synthesizeDemoPayload({
      preset: 'SET_FINISHED',
      romDeg: 145,
      trunkTiltDeg: 1.5,
      distanceM: 2.0,
      exercise: mapped[0],
      validReps: 10,
      invalidReps: 0,
      holdRemainingSec: 0,
      patientOverride: { id: 'HN-61716', name: 'Pdasdwa' },
    });
    expect(patientPayload.patient).toEqual({ id: 'HN-61716', name: 'Pdasdwa' });
    expect(patientPayload.tts_voice_th).toContain('Pdasdwa');
  });

  it('enforces prescribed arm strictly: raising wrong arm does not measure high ROM and alerts user', () => {
    // User raises LEFT arm high (~170 deg) while RIGHT arm hangs down (~6 deg) during RIGHT_SHOULDER exercise
    const leftRaisedOnly = createNeutralLandmarks({
      [POSE_INDEX.LEFT_ELBOW]: { x: 0.60, y: 0.20, z: 0, visibility: 0.99 },
      [POSE_INDEX.LEFT_WRIST]: { x: 0.61, y: 0.05, z: 0, visibility: 0.99 },
    });

    const rightCheck = measureLiveJointAngleAndTilt(leftRaisedOnly, 'right_shoulder');
    expect(rightCheck.wrongSideRaised).toBe(true);
    expect(rightCheck.romDeg).not.toBeNull();
    expect(rightCheck.romDeg!).toBeLessThan(30);
    expect(rightCheck.oppositeRomDeg!).toBeGreaterThan(150);
    expect(rightCheck.wrongSideBannerTh).toBe('กรุณายกแขนขวา (ท่านี้ต้องทำด้วยแขนขวา)');

    // When the same landmarks are used for LEFT_SHOULDER exercise, LEFT arm IS valid
    const leftCheck = measureLiveJointAngleAndTilt(leftRaisedOnly, 'left_shoulder');
    expect(leftCheck.wrongSideRaised).toBe(false);
    expect(leftCheck.romDeg!).toBeGreaterThan(150);
    expect(leftCheck.wrongSideBannerTh).toBeNull();

    // User raises RIGHT arm high during LEFT_SHOULDER exercise -> blocked with left-arm warning
    const rightRaisedOnly = createNeutralLandmarks({
      [POSE_INDEX.RIGHT_ELBOW]: { x: 0.40, y: 0.20, z: 0, visibility: 0.99 },
      [POSE_INDEX.RIGHT_WRIST]: { x: 0.39, y: 0.05, z: 0, visibility: 0.99 },
    });
    const wrongForLeft = measureLiveJointAngleAndTilt(rightRaisedOnly, 'left_shoulder');
    expect(wrongForLeft.wrongSideRaised).toBe(true);
    expect(wrongForLeft.romDeg!).toBeLessThan(30);
    expect(wrongForLeft.wrongSideBannerTh).toBe('กรุณายกแขนซ้าย (ท่านี้ต้องทำด้วยแขนซ้าย)');

    // synthesizeDemoPayload WRONG_LIMB outputs WRONG_LIMB compensation and Thai voice warning
    const wrongLimbPayload = synthesizeDemoPayload({
      preset: 'WRONG_LIMB',
      romDeg: 15,
      trunkTiltDeg: 2,
      distanceM: 2.0,
      exercise: rightShoulder,
      validReps: 3,
      invalidReps: 0,
      holdRemainingSec: 3,
    });
    expect(wrongLimbPayload.telemetry_analysis.is_form_valid).toBe(false);
    expect(wrongLimbPayload.telemetry_analysis.compensation_detected).toContain('WRONG_LIMB');
    expect(wrongLimbPayload.rep_counter.valid_reps).toBe(3);
    expect(wrongLimbPayload.tts_voice_th).toContain(getPrescribedLimbPromptTh('right_shoulder').requiredLimbTh);
  });
});

