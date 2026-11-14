import { describe, expect, it } from 'vitest';
import {
  EXERCISE_OPTIONS,
  mapTodayExercisesToPrescriptions,
  synthesizeDemoPayload,
} from '../hooks/useKioskSocket';

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
});

