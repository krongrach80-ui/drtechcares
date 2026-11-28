import { describe, expect, it } from 'vitest';
import {
  CLINICAL_EXERCISES,
  ClinicalBiomechanicsAgent,
  ClinicalRotationCycleTracker,
  HAND21_INDEX,
  JointAngleSmoother,
  POSE_INDEX,
  RehabExercisePipeline,
  analyzeHand21Biomechanics,
  evaluateArmKinematics,
  solveElderlyClinicalJointBiomechanics,
  stepElderlyHoldAndRepStateMachine,
  type ClinicalTelemetryPacket,
  type ElderlyHoldRepState,
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
      CLINICAL_EXERCISES.find((item) => item.code === 'ELDERLY_SCAP_PROTRACT') ?? CLINICAL_EXERCISES[0];
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
        exercise_id: 'ELDERLY_SH_FLEX_R',
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

  it('provides exactly 15 unified master Elderly & MOPH upper-body exercises with zero duplicate L/R or deprecated codes', () => {
    expect(CLINICAL_EXERCISES).toHaveLength(15);
    expect(
      CLINICAL_EXERCISES.every((ex) => ex.category === 'upper_limb' || ex.category === 'neck_trunk'),
    ).toBe(true);
    const codes = CLINICAL_EXERCISES.map((ex) => ex.code);
    expect(codes).toContain('ELDERLY_SH_FLEX');
    expect(codes).toContain('ELDERLY_SH_ABD');
    expect(codes).toContain('ELDERLY_CROSS_BODY');
    expect(codes).toContain('ELDERLY_SCAP_PROTRACT');
    expect(codes).toContain('ELDERLY_CLASPED_FORWARD_PUSH');
    expect(codes).toContain('ELDERLY_HAND_TO_HEAD');
    expect(codes).toContain('ELDERLY_FOREARM_TURN');
    expect(codes).toContain('ELDERLY_THUMB_PINCH');
    expect(codes).toContain('MOPH_NECK_TILT');
    expect(codes).toContain('MOPH_NECK_TURN');
    expect(codes).not.toContain('ELDERLY_SH_FLEX_R');
    expect(codes).not.toContain('ELDERLY_SH_FLEX_L');
    expect(codes).not.toContain('MOPH_NECK_TURN_BILATERAL');
    expect(codes).not.toContain('BAL_STAND_R');
    expect(codes).not.toContain('KNEE_EXT_R');
    expect(codes).not.toContain('SIT_TO_STAND');
    expect(codes).not.toContain('SH_WALL_SLIDE_R');
    expect(codes).not.toContain('SH_PEND_R');

    // Every trilingual master exercise name must reflect Elderly Care branding
    for (const ex of CLINICAL_EXERCISES) {
      expect(ex.name.th.startsWith('[ผู้สูงอายุ]') || ex.name.th.startsWith('[สธ. ผู้สูงอายุ]')).toBe(true);
      expect(ex.name.en.startsWith('Elderly Care:') || ex.name.en.startsWith('MOPH Elderly:')).toBe(true);
      expect(ex.name.cn.startsWith('老年康复训练：') || ex.name.cn.startsWith('卫生部老年保健：')).toBe(true);
    }
  });

  it('computes high-detail 21-point MediaPipe hand biomechanics (5 fingers, 4-finger pinch, true 3D supination/pronation roll)', () => {
    const buildDetailedWorldHand = (opts: {
      curled?: boolean;
      pronated?: boolean;
      pinchFinger?: 'index' | 'middle' | 'ring' | 'pinky';
    } = {}): PoseLandmark[] => {
      const pts: PoseLandmark[] = Array.from({ length: 21 }, () => ({ x: 0, y: 0, z: 0, visibility: 1 }));
      const flipX = opts.pronated ? -1 : 1;
      pts[HAND21_INDEX.WRIST] = { x: 0, y: 0, z: 0, visibility: 1 };
      // Palm MCPs (in meters; right hand supinated faces camera/up with index at +x, pinky at -x)
      pts[HAND21_INDEX.INDEX_MCP] = { x: 0.035 * flipX, y: -0.09, z: 0, visibility: 1 };
      pts[HAND21_INDEX.MIDDLE_MCP] = { x: 0.01 * flipX, y: -0.095, z: 0, visibility: 1 };
      pts[HAND21_INDEX.RING_MCP] = { x: -0.015 * flipX, y: -0.09, z: 0, visibility: 1 };
      pts[HAND21_INDEX.PINKY_MCP] = { x: -0.04 * flipX, y: -0.082, z: 0, visibility: 1 };

      // Thumb chain (CMC=1, MCP=2, IP=3, TIP=4)
      pts[HAND21_INDEX.THUMB_CMC] = { x: 0.032 * flipX, y: -0.025, z: -0.01, visibility: 1 };
      pts[HAND21_INDEX.THUMB_MCP] = { x: 0.056 * flipX, y: -0.055, z: -0.015, visibility: 1 };
      pts[HAND21_INDEX.THUMB_IP] = opts.curled
        ? { x: 0.038 * flipX, y: -0.068, z: -0.045, visibility: 1 }
        : { x: 0.078 * flipX, y: -0.082, z: -0.018, visibility: 1 };
      pts[HAND21_INDEX.THUMB_TIP] = opts.curled
        ? { x: 0.015 * flipX, y: -0.058, z: -0.052, visibility: 1 }
        : { x: 0.098 * flipX, y: -0.105, z: -0.02, visibility: 1 };

      const fingers: Array<{
        name: 'index' | 'middle' | 'ring' | 'pinky';
        mcp: number;
        pip: number;
        dip: number;
        tip: number;
        xBase: number;
        len: number;
      }> = [
        { name: 'index', mcp: 5, pip: 6, dip: 7, tip: 8, xBase: 0.035 * flipX, len: 0.085 },
        { name: 'middle', mcp: 9, pip: 10, dip: 11, tip: 12, xBase: 0.01 * flipX, len: 0.095 },
        { name: 'ring', mcp: 13, pip: 14, dip: 15, tip: 16, xBase: -0.015 * flipX, len: 0.088 },
        { name: 'pinky', mcp: 17, pip: 18, dip: 19, tip: 20, xBase: -0.04 * flipX, len: 0.072 },
      ];

      for (const f of fingers) {
        const mcpPt = pts[f.mcp];
        if (opts.curled) {
          pts[f.pip] = { x: f.xBase, y: mcpPt.y - f.len * 0.35, z: -0.025, visibility: 1 };
          pts[f.dip] = { x: f.xBase, y: mcpPt.y - f.len * 0.15, z: -0.048, visibility: 1 };
          pts[f.tip] = { x: f.xBase, y: mcpPt.y + f.len * 0.12, z: -0.036, visibility: 1 };
        } else {
          pts[f.pip] = { x: f.xBase, y: mcpPt.y - f.len * 0.42, z: 0, visibility: 1 };
          pts[f.dip] = { x: f.xBase, y: mcpPt.y - f.len * 0.72, z: 0, visibility: 1 };
          pts[f.tip] = { x: f.xBase, y: mcpPt.y - f.len, z: 0, visibility: 1 };
        }
      }

      if (opts.pinchFinger) {
        const tipIdx =
          opts.pinchFinger === 'index'
            ? HAND21_INDEX.INDEX_TIP
            : opts.pinchFinger === 'middle'
              ? HAND21_INDEX.MIDDLE_TIP
              : opts.pinchFinger === 'ring'
                ? HAND21_INDEX.RING_TIP
                : HAND21_INDEX.PINKY_TIP;
        pts[HAND21_INDEX.THUMB_TIP] = {
          x: pts[tipIdx].x + 0.004,
          y: pts[tipIdx].y + 0.004,
          z: pts[tipIdx].z,
          visibility: 1,
        };
      }
      return pts;
    };

    const worldOpenSupinated = buildDetailedWorldHand();
    const hand2dProxy = worldOpenSupinated.map((p) => ({
      x: 0.5 + p.x * 1.4,
      y: 0.65 + p.y * 1.4,
      z: p.z,
      visibility: 1,
    }));
    const openBio = analyzeHand21Biomechanics({
      landmarks2D: hand2dProxy,
      worldLandmarks3D: worldOpenSupinated,
      side: 'right',
    });
    expect(openBio).not.toBeNull();
    expect(openBio!.usedWorld3D).toBe(true);
    expect(openBio!.extendedFingerCount).toBe(5);
    expect(openBio!.meanFingerExtensionPct).toBeGreaterThanOrEqual(85);
    expect(openBio!.forearmRollSignedDeg).toBeGreaterThanOrEqual(45);
    expect(openBio!.palmOrientation).toBe('SUPINATED_UP');

    const worldPronated = buildDetailedWorldHand({ pronated: true });
    const pronatedBio = analyzeHand21Biomechanics({
      landmarks2D: hand2dProxy,
      worldLandmarks3D: worldPronated,
      side: 'right',
    });
    expect(pronatedBio!.forearmRollSignedDeg).toBeLessThanOrEqual(-45);
    expect(pronatedBio!.palmOrientation).toBe('PRONATED_DOWN');

    const worldFist = buildDetailedWorldHand({ curled: true });
    const fistBio = analyzeHand21Biomechanics({
      landmarks2D: hand2dProxy,
      worldLandmarks3D: worldFist,
      side: 'right',
    });
    expect(fistBio!.extendedFingerCount).toBe(0);
    expect(fistBio!.meanFingerExtensionPct).toBeLessThanOrEqual(35);

    for (const targetFinger of ['index', 'middle', 'ring', 'pinky'] as const) {
      const pinchWorld = buildDetailedWorldHand({ pinchFinger: targetFinger });
      const pinchBio = analyzeHand21Biomechanics({
        landmarks2D: hand2dProxy,
        worldLandmarks3D: pinchWorld,
        side: 'right',
      });
      expect(pinchBio!.activePinchFinger).toBe(targetFinger);
      expect(pinchBio!.pinchAngleScoreDeg).toBeGreaterThanOrEqual(58);
    }
  });

  it('enforces genuine physical rotation cycles for wrist-turn, shoulder-rotation, and shoulder-roll (static poses never score)', () => {
    const initialRepState = (): ElderlyHoldRepState => ({
      phase: 'PHASE_1_HOLDING',
      holdElapsedSec: 0,
      completedReps: 0,
      cooldownStartedAtMs: 0,
      repCelebrateUntilMs: 0,
      loweredFrameStreak: 0,
    });

    // 1. FOREARM_TURN: Static flat palm held at +75 deg for 5s -> 0 deg effective ROM, 0 reps!
    const wristTracker = new ClinicalRotationCycleTracker();
    let wristState = initialRepState();
    for (let i = 0; i < 50; i += 1) {
      const ev = wristTracker.update({
        mode: 'forearm_turn',
        signedAngleDeg: 75,
        postureReady: true,
        targetRomDeg: 75,
        nowMs: i * 100,
      });
      expect(ev.effectiveRotationRomDeg).toBe(0);
      expect(ev.rotationVerified).toBe(false);
      const step = stepElderlyHoldAndRepStateMachine({
        state: wristState,
        activeRomDeg: ev.effectiveRotationRomDeg,
        exerciseCode: 'ELDERLY_FOREARM_TURN',
        exerciseSide: 'RIGHT',
        targetRomDeg: 75,
        targetHoldSec: 1.0,
        targetReps: 5,
        dtSec: 0.1,
        nowMs: i * 100,
        sideViolation: 'NONE',
        wrongSideBannerTh: null,
        wrongPlaneRaised: false,
        wrongPlaneBannerTh: null,
        isCompensatedNow: false,
        postureBannerTh: null,
        blockCounting: false,
        rotationGate: {
          required: true,
          rotationVerified: ev.rotationVerified,
          returnedToStart: ev.returnedToStart,
          guidanceBannerTh: ev.guidanceBannerTh,
        },
      });
      wristState = step.nextState;
    }
    expect(wristState.completedReps).toBe(0);

    // Now genuinely rotate from +75 deg -> -50 deg (125 deg flip!) and hold for 1.0s -> +1 rep!
    let tMs = 6000;
    for (const angle of [55, 30, 5, -20, -45, -50]) {
      const ev = wristTracker.update({
        mode: 'forearm_turn',
        signedAngleDeg: angle,
        postureReady: true,
        targetRomDeg: 75,
        nowMs: (tMs += 100),
      });
      const step = stepElderlyHoldAndRepStateMachine({
        state: wristState,
        activeRomDeg: ev.effectiveRotationRomDeg,
        exerciseCode: 'ELDERLY_FOREARM_TURN',
        exerciseSide: 'RIGHT',
        targetRomDeg: 75,
        targetHoldSec: 1.0,
        targetReps: 5,
        dtSec: 0.1,
        nowMs: tMs,
        sideViolation: 'NONE',
        wrongSideBannerTh: null,
        wrongPlaneRaised: false,
        wrongPlaneBannerTh: null,
        isCompensatedNow: false,
        postureBannerTh: null,
        blockCounting: false,
        rotationGate: {
          required: true,
          rotationVerified: ev.rotationVerified,
          returnedToStart: ev.returnedToStart,
          guidanceBannerTh: ev.guidanceBannerTh,
        },
      });
      wristState = step.nextState;
    }
    for (let i = 0; i < 12; i += 1) {
      const ev = wristTracker.update({
        mode: 'forearm_turn',
        signedAngleDeg: -50,
        postureReady: true,
        targetRomDeg: 75,
        nowMs: (tMs += 100),
      });
      const step = stepElderlyHoldAndRepStateMachine({
        state: wristState,
        activeRomDeg: ev.effectiveRotationRomDeg,
        exerciseCode: 'ELDERLY_FOREARM_TURN',
        exerciseSide: 'RIGHT',
        targetRomDeg: 75,
        targetHoldSec: 1.0,
        targetReps: 5,
        dtSec: 0.1,
        nowMs: tMs,
        sideViolation: 'NONE',
        wrongSideBannerTh: null,
        wrongPlaneRaised: false,
        wrongPlaneBannerTh: null,
        isCompensatedNow: false,
        postureBannerTh: null,
        blockCounting: false,
        rotationGate: {
          required: true,
          rotationVerified: ev.rotationVerified,
          returnedToStart: ev.returnedToStart,
          guidanceBannerTh: ev.guidanceBannerTh,
        },
      });
      wristState = step.nextState;
      if (step.justCompletedRep) {
        wristTracker.notifyRepCompleted();
        break;
      }
    }
    expect(wristState.completedReps).toBe(1);
    expect(wristState.phase).toBe('PHASE_2_WAIT_LOWER_TO_SCORE');

    // Staying frozen at -50 deg MUST NOT unlock or count rep #2
    for (let i = 0; i < 20; i += 1) {
      const ev = wristTracker.update({
        mode: 'forearm_turn',
        signedAngleDeg: -50,
        postureReady: true,
        targetRomDeg: 75,
        nowMs: (tMs += 100),
      });
      const step = stepElderlyHoldAndRepStateMachine({
        state: wristState,
        activeRomDeg: ev.effectiveRotationRomDeg,
        exerciseCode: 'ELDERLY_FOREARM_TURN',
        exerciseSide: 'RIGHT',
        targetRomDeg: 75,
        targetHoldSec: 1.0,
        targetReps: 5,
        dtSec: 0.1,
        nowMs: tMs,
        sideViolation: 'NONE',
        wrongSideBannerTh: null,
        wrongPlaneRaised: false,
        wrongPlaneBannerTh: null,
        isCompensatedNow: false,
        postureBannerTh: null,
        blockCounting: false,
        rotationGate: {
          required: true,
          rotationVerified: ev.rotationVerified,
          returnedToStart: ev.returnedToStart,
          guidanceBannerTh: ev.guidanceBannerTh,
        },
      });
      wristState = step.nextState;
    }
    expect(wristState.completedReps).toBe(1);
    expect(wristState.phase).toBe('PHASE_2_WAIT_LOWER_TO_SCORE');

    // Rotating back >= 36 deg unlocks back to PHASE_1_HOLDING!
    const returnEv = wristTracker.update({
      mode: 'forearm_turn',
      signedAngleDeg: 0,
      postureReady: true,
      targetRomDeg: 75,
      nowMs: (tMs += 100),
    });
    expect(returnEv.returnedToStart).toBe(true);

    // 2. SHOULDER_ROTATION: Parking arm already out at +60 deg without visiting neutral -> 0 reps!
    const shRotTracker = new ClinicalRotationCycleTracker();
    const parkedEv = shRotTracker.update({
      mode: 'shoulder_rotation',
      signedAngleDeg: 60,
      postureReady: true,
      targetRomDeg: 55,
      nowMs: 100,
    });
    expect(parkedEv.rotationVerified).toBe(false);
    expect(parkedEv.effectiveRotationRomDeg).toBe(0);

    // Start from neutral (5 deg) with 90-deg elbow tucked, then sweep outward to 55 deg -> verified!
    shRotTracker.update({ mode: 'shoulder_rotation', signedAngleDeg: 5, postureReady: true, targetRomDeg: 55, nowMs: 200 });
    shRotTracker.update({ mode: 'shoulder_rotation', signedAngleDeg: 25, postureReady: true, targetRomDeg: 55, nowMs: 300 });
    shRotTracker.update({ mode: 'shoulder_rotation', signedAngleDeg: 45, postureReady: true, targetRomDeg: 55, nowMs: 400 });
    const shVerified = shRotTracker.update({ mode: 'shoulder_rotation', signedAngleDeg: 56, postureReady: true, targetRomDeg: 55, nowMs: 500 });
    expect(shVerified.rotationVerified).toBe(true);
    expect(shVerified.effectiveRotationRomDeg).toBeCloseTo(56, 1);

    // 3. SHOULDER_SHRUG_ROLL: Static vertical lift with zero roll stays unverified; circular lift+roll verifies!
    const shrugTracker = new ClinicalRotationCycleTracker();
    shrugTracker.update({ mode: 'shoulder_shrug', signedAngleDeg: 5, secondaryRollSignal: 0, postureReady: true, targetRomDeg: 35, nowMs: 100 });
    const staticShrug = shrugTracker.update({ mode: 'shoulder_shrug', signedAngleDeg: 32, secondaryRollSignal: 0, postureReady: true, targetRomDeg: 35, nowMs: 200 });
    expect(staticShrug.rotationVerified).toBe(false);
    shrugTracker.update({ mode: 'shoulder_shrug', signedAngleDeg: 24, secondaryRollSignal: -0.04, postureReady: true, targetRomDeg: 35, nowMs: 300 });
    const rolledShrug = shrugTracker.update({ mode: 'shoulder_shrug', signedAngleDeg: 32, secondaryRollSignal: 0.04, postureReady: true, targetRomDeg: 35, nowMs: 400 });
    expect(rolledShrug.rotationVerified).toBe(true);

    // 4. Exact anatomical joint angle verification (zero artificial gain on neck tilt & elbow curl)
    const exactElbowPose = makeStandingPose({
      [POSE_INDEX.RIGHT_SHOULDER]: { x: 0.42, y: 0.35, z: 0, visibility: 0.99 },
      [POSE_INDEX.RIGHT_ELBOW]: { x: 0.42, y: 0.55, z: 0, visibility: 0.99 },
      [POSE_INDEX.RIGHT_WRIST]: { x: 0.22, y: 0.55, z: 0, visibility: 0.99 }, // exact 90-deg elbow flexion!
    });
    const exactElbowRes = solveElderlyClinicalJointBiomechanics({
      landmarks: exactElbowPose,
      targetJoint: 'right_shoulder',
      exerciseSide: 'RIGHT',
      jointMode: 'elbow',
      targetRomDeg: 110,
    });
    expect(exactElbowRes.romDeg!).toBeGreaterThanOrEqual(85);
    expect(exactElbowRes.romDeg!).toBeLessThanOrEqual(95);
  });
});

