import { describe, expect, it } from 'vitest';
import {
  CLINICAL_EXERCISES,
  ClinicalBiomechanicsAgent,
  JointAngleSmoother,
  POSE_INDEX,
  RehabExercisePipeline,
  evaluateArmKinematics,
  type ClinicalTelemetryPacket,
  type PoseLandmark,
} from '../index';

function makeStandingPose(overrides: Partial<Record<number, Partial<PoseLandmark>>> = {}): PoseLandmark[] {
  const base: PoseLandmark[] = Array.from({ length: 33 }, () => ({
    x: 0.5,
    y: 0.5,
    z: 0,
    visibility: 0.99,
  }));

  base[POSE_INDEX.LEFT_SHOULDER] = { x: 0.58, y: 0.35, z: 0, visibility: 0.99 };
  base[POSE_INDEX.RIGHT_SHOULDER] = { x: 0.42, y: 0.35, z: 0, visibility: 0.99 };
  base[POSE_INDEX.LEFT_ELBOW] = { x: 0.60, y: 0.52, z: 0, visibility: 0.99 };
  base[POSE_INDEX.RIGHT_ELBOW] = { x: 0.40, y: 0.52, z: 0, visibility: 0.99 };
  base[POSE_INDEX.LEFT_WRIST] = { x: 0.62, y: 0.68, z: 0, visibility: 0.99 };
  base[POSE_INDEX.RIGHT_WRIST] = { x: 0.38, y: 0.68, z: 0, visibility: 0.99 };
  base[POSE_INDEX.LEFT_HIP] = { x: 0.56, y: 0.65, z: 0, visibility: 0.99 };
  base[POSE_INDEX.RIGHT_HIP] = { x: 0.44, y: 0.65, z: 0, visibility: 0.99 };

  for (const [idxStr, patch] of Object.entries(overrides)) {
    const idx = Number(idxStr);
    base[idx] = { ...base[idx], ...patch };
  }
  return base;
}

describe('Clinical Pose Estimation Engine', () => {
  it('computes accurate straight-arm kinematics when landmarks are visible and in-frame', () => {
    const pose = makeStandingPose({
      // Right arm raised forward/up ~145° with straight elbow
      [POSE_INDEX.RIGHT_ELBOW]: { x: 0.36, y: 0.22, z: -0.05, visibility: 0.99 },
      [POSE_INDEX.RIGHT_WRIST]: { x: 0.30, y: 0.10, z: -0.10, visibility: 0.99 },
    });

    const res = evaluateArmKinematics(pose, 'right', {
      jointGroup: 'shoulder',
      requireStraightElbow: true,
    });
    expect(res.gate.valid).toBe(true);
    expect(res.angleDeg).not.toBeNull();
    expect(res.angleDeg!).toBeGreaterThan(120);
    expect(res.elbowAngleDeg!).toBeGreaterThanOrEqual(135);
    expect(res.elbowGuardFailed).toBe(false);
  });

  it('blocks angle calculation and returns null + targeted warning when wrist/elbow is out-of-frame or invisible', () => {
    const outOfFramePose = makeStandingPose({
      [POSE_INDEX.RIGHT_WRIST]: { x: 0.995, y: 0.45, z: 0, visibility: 0.95 },
    });
    const outRes = evaluateArmKinematics(outOfFramePose, 'right');
    expect(outRes.gate.valid).toBe(false);
    expect(outRes.angleDeg).toBeNull();
    expect(outRes.gate.warningMessage?.th).toContain('อยู่นอกกรอบกล้อง');

    const occludedLeftWrist = makeStandingPose({
      [POSE_INDEX.LEFT_WRIST]: { x: 0.62, y: 0.55, z: 0, visibility: 0.25 },
    });
    const visRes = evaluateArmKinematics(occludedLeftWrist, 'left');
    expect(visRes.gate.valid).toBe(false);
    expect(visRes.angleDeg).toBeNull();
    expect(visRes.gate.warningMessage?.th).toContain('ข้อมือซ้าย');
  });

  it('applies EMA smoothing (0.35) and spike clamping (28 deg per <150ms frame) independently per side', () => {
    const smoother = new JointAngleSmoother();
    const r1 = smoother.update('right', 40, 100);
    expect(r1.smoothed).toBe(40);

    // Sudden impossible spike +90° within 50ms -> clamped to +28° (40+28=68), EMA = 0.35*68 + 0.65*40 = 49.8
    const r2 = smoother.update('right', 130, 150);
    expect(r2.clampedSpike).toBe(true);
    expect(r2.smoothed).toBeCloseTo(49.8, 1);

    // Left side stays independent
    const l1 = smoother.update('left', 120, 150);
    expect(l1.smoothed).toBe(120);

    // Missing landmark holds last reliable angle, then clears after >500ms
    const rHold = smoother.update('right', null, 300);
    expect(rHold.smoothed).toBeNull();
    expect(rHold.lastReliable).toBeCloseTo(49.8, 1);

    const rCleared = smoother.update('right', null, 900);
    expect(rCleared.lastReliable).toBeNull();
  });

  it('enforces straight-elbow guard (<135 deg) and completes full rep state machine cycle', () => {
    const ex =
      CLINICAL_EXERCISES.find((item) => item.code === 'STROKE_SCAP_PROTRACT_R') ?? CLINICAL_EXERCISES[0];
    const pipeline = new RehabExercisePipeline(ex, 'right', 1, 1, 1);

    const restPose = makeStandingPose();
    const raisedBentElbow = makeStandingPose({
      [POSE_INDEX.RIGHT_ELBOW]: { x: 0.30, y: 0.32, z: 0, visibility: 0.99 },
      [POSE_INDEX.RIGHT_WRIST]: { x: 0.36, y: 0.44, z: 0, visibility: 0.99 }, // sharply bent elbow (~60°) at ~105° shoulder
    });
    const raisedStraight = makeStandingPose({
      [POSE_INDEX.RIGHT_ELBOW]: { x: 0.32, y: 0.30, z: 0, visibility: 0.99 },
      [POSE_INDEX.RIGHT_WRIST]: { x: 0.22, y: 0.25, z: 0, visibility: 0.99 }, // straight elbow ~180° inside [70, 120] target zone
    });

    expect(pipeline.step(restPose, 0).state).toBe('IDLE');

    const bentSnap = pipeline.step(raisedBentElbow, 200);
    expect(bentSnap.holdPaused).toBe(true);
    expect(bentSnap.warningMessage?.th).toContain('เหยียดข้อศอก');
    expect(bentSnap.currentRep).toBe(0);

    // Smoothly move into straight-arm target zone (multiple ticks for EMA to cross 110°)
    for (let t = 400; t <= 1600; t += 150) {
      pipeline.step(raisedStraight, t);
    }
    // Hold for >= 1s
    for (let t = 1750; t <= 3100; t += 200) {
      pipeline.step(raisedStraight, t);
    }
    // Return to rest zone
    let finalSnap = pipeline.step(restPose, 3500);
    for (let t = 3650; t <= 4800; t += 150) {
      finalSnap = pipeline.step(restPose, t);
    }

    expect(finalSnap.currentRep).toBe(1);
    expect(finalSnap.sessionCompleted).toBe(true);
  });

  it('evaluates Clinical Biomechanics AI Agent: EMERGENCY_HALT, AUTH_FRAMING (1.8-2.2m), anti-cheating, and SET_FINISHED', () => {
    const agent = new ClinicalBiomechanicsAgent();
    const basePacket = (
      overrideTelemetry: Partial<ClinicalTelemetryPacket['live_telemetry']> = {},
      overrideFraming: Partial<ClinicalTelemetryPacket['spatial_framing']> = {},
      overrideAuth: Partial<ClinicalTelemetryPacket['auth_status']> = {},
    ): ClinicalTelemetryPacket => ({
      auth_status: { verified: true, confidence: 0.94, liveness: true, ...overrideAuth },
      spatial_framing: { distance_m: 2.0, full_body_visible: true, feet_visible: true, lighting_score: 0.9, ...overrideFraming },
      exercise_prescription: {
        exercise_id: 'STROKE_SH_FLEX_R',
        target_joint: 'right_shoulder',
        target_rom_deg: 100,
        hold_sec: 1.0,
        target_reps: 1,
      },
      live_telemetry: {
        primary_angle_deg: 20,
        trunk_tilt_deg: 2,
        contralateral_shoulder_elevation_cm: 0.5,
        knee_valgus_deg: 1,
        angular_velocity_deg_per_sec: 20,
        dt_sec: 0.5,
        ...overrideTelemetry,
      },
    });

    // 1. SAFETY_OVERRIDE: Fall / rapid vertical drop -> EMERGENCY_HALT
    const haltRes = agent.evaluate(basePacket({ rapid_vertical_drop_detected: true }));
    expect(haltRes.kiosk_state).toBe('EMERGENCY_HALT');
    expect(haltRes.telemetry_analysis.pain_alert).toBe(true);
    expect(haltRes.ui_directives.highlight_color).toBe('RED');

    // 2. AUTH_AND_FRAMING: Distance outside 1.8m-2.2m -> AUTH_FRAMING
    const tooCloseRes = agent.evaluate(basePacket({}, { distance_m: 1.4 }));
    expect(tooCloseRes.kiosk_state).toBe('AUTH_FRAMING');
    expect(tooCloseRes.ui_directives.highlight_color).toBe('AMBER');

    const badAuthRes = agent.evaluate(basePacket({}, {}, { confidence: 0.72 }));
    expect(badAuthRes.kiosk_state).toBe('AUTH_FRAMING');

    // 3. ANTI-CHEATING: TRUNK_LEAN (>10 deg) & SHOULDER_HIKE (>3 cm)
    const cheatRes = agent.evaluate(
      basePacket({ primary_angle_deg: 98, trunk_tilt_deg: 14, contralateral_shoulder_elevation_cm: 4.5 }),
    );
    expect(cheatRes.telemetry_analysis.is_form_valid).toBe(false);
    expect(cheatRes.telemetry_analysis.compensation_detected).toContain('TRUNK_LEAN');
    expect(cheatRes.telemetry_analysis.compensation_detected).toContain('SHOULDER_HIKE');
    expect(cheatRes.rep_counter.invalid_reps).toBe(1);
    expect(cheatRes.ui_directives.highlight_color).toBe('AMBER');

    // Reset back to rest before clean rep
    agent.reset();

    // 4. Clean ROM Gate (>= 95% of 100° = 95°) + Hold (1.0s ± 0.2s)
    const hold1 = agent.evaluate(basePacket({ primary_angle_deg: 97, dt_sec: 0.5 }));
    expect(hold1.kiosk_state).toBe('HOLD_PHASE');
    expect(hold1.telemetry_analysis.is_form_valid).toBe(true);

    const doneRes = agent.evaluate(basePacket({ primary_angle_deg: 98, dt_sec: 0.5 }));
    expect(doneRes.kiosk_state).toBe('SET_FINISHED');
    expect(doneRes.rep_counter.valid_reps).toBe(1);
    expect(doneRes.ui_directives.highlight_color).toBe('GREEN');
  });

  it('evaluates exact RTMPose/ZED SDK input payload schema seamlessly', () => {
    const agent = new ClinicalBiomechanicsAgent();
    const res = agent.evaluate({
      auth_status: { verified: true, patient_id: 'PT-8841', confidence: 0.94, liveness: true },
      spatial_framing: { distance_m: 2.05, full_body_visible: true },
      exercise_prescription: {
        exercise_name: 'Right Shoulder Abduction (กางแขนขวาด้านข้าง)',
        target_rom_deg: 150.0,
        hold_sec: 3.0,
        target_reps: 10,
      },
      live_telemetry: {
        primary_angle_deg: 138.5,
        trunk_tilt_deg: 14.2,
        shoulder_elevation_cm: 1.1,
        angular_velocity_deg_s: 28.0,
        hold_elapsed_sec: 0.0,
        pain_score: 0.1,
      },
    });
    expect(res.kiosk_state).toBe('ACTIVE_REP');
    expect(res.telemetry_analysis.rom_completion_pct).toBe(92);
    expect(res.telemetry_analysis.is_form_valid).toBe(false);
    expect(res.telemetry_analysis.compensation_detected).toEqual(['TRUNK_LEAN', 'INCOMPLETE_ROM']);
    expect(res.ui_directives.skeleton_highlight_joint).toBe('spine');
    expect(res.ui_directives.highlight_color).toBe('AMBER');
  });

  it('provides exactly 26 standardized Stroke & MOPH upper-body exercises with zero deprecated leg/duplicate codes', () => {
    expect(CLINICAL_EXERCISES).toHaveLength(26);
    expect(
      CLINICAL_EXERCISES.every((ex) => ex.category === 'upper_limb' || ex.category === 'neck_trunk'),
    ).toBe(true);
    const codes = CLINICAL_EXERCISES.map((ex) => ex.code);
    expect(codes).toContain('STROKE_CROSS_BODY_R');
    expect(codes).toContain('STROKE_SCAP_PROTRACT_R');
    expect(codes).toContain('STROKE_BOBATH_FORWARD_PUSH');
    expect(codes).toContain('STROKE_HAND_TO_HEAD_R');
    expect(codes).toContain('STROKE_FOREARM_TURN_R');
    expect(codes).toContain('STROKE_THUMB_PINCH_R');
    expect(codes).toContain('MOPH_NECK_TURN_BILATERAL');
    expect(codes).not.toContain('BAL_STAND_R');
    expect(codes).not.toContain('KNEE_EXT_R');
    expect(codes).not.toContain('SIT_TO_STAND');
    expect(codes).not.toContain('SH_WALL_SLIDE_R');
    expect(codes).not.toContain('SH_PEND_R');
  });
});

