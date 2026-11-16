import { describe, expect, it } from 'vitest';
import {
  resolvePrescribedExercisesFromSchedules,
  resolveTodayPatientSchedule,
  sanitizeAccidentalDefaultKneePrefix,
} from '@drtechcares/api-client';
import { POSE_INDEX } from '@drtechcares/pose-engine';
import type { NormalizedLandmark } from '@mediapipe/tasks-vision';
import {
  EXERCISE_OPTIONS,
  getPrescribedLimbPromptTh,
  mapTodayExercisesToPrescriptions,
  resolveExpectedPlaneForExercise,
  resolveKioskWsCandidates,
  synthesizeDemoPayload,
} from '../hooks/useKioskSocket';
import {
  DEFAULT_FORM_ASSESSMENT_RULES,
  LandmarkSmoother,
  interpolatePoseLandmarks,
  isFormRuleActive,
  measureLiveJointAngleAndTilt,
  sanitizeAndStabilizePoseLandmarks,
  stepSmoothRomAngle,
} from '../components/SmartRehabKioskUI';
import {
  VOICE_COACH_LANGUAGES,
  buildTtsProxyUrl,
  decideVoiceCoachSpeechTurn,
  pickSpeechVoice,
  translateVoiceCoachText,
} from '../components/VoiceCoachBanner';
import {
  HAND_GLITCH_GRACE_MS,
  HAND_SWITCH_GRACE_MS,
  VERIFIED_HAND_HOLD_MS,
  evaluateStrictTwoHandsForStartModal,
  getKioskVerifiedModalCopy,
  stepVerifiedHandHold,
} from '../components/KioskBiometricLoginView';
import { getKioskOverviewVoiceNarration } from '../App';
import {
  computeContainRect,
  drawMediaPipe33OnCanvas,
  drawWholeBody133OnCanvas,
  isUpperBodyVisibleInFrame,
  type Keypoint3DPoint,
} from '../hooks/useEdgeVisionSocket';

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
    expect(patientPayload.tts_voice_th).toContain('ครบ 3 วินาทีแล้ว');
    expect(patientPayload.tts_voice_th).toContain('ค่อยๆ เอาแขนลงช้าๆ ครับ');

    const repCompletedPayload = synthesizeDemoPayload({
      preset: 'REP_COMPLETED',
      romDeg: 145,
      trunkTiltDeg: 1.2,
      distanceM: 2.0,
      exercise: mapped[0],
      validReps: 4,
      invalidReps: 0,
      holdRemainingSec: 0,
    });
    expect(repCompletedPayload.kiosk_state).toBe('REP_COMPLETED');
    expect(repCompletedPayload.tts_voice_th).toBe(
      'ครบ 3 วินาทีแล้ว ดีมากครับ ครบ 4 ครั้ง ค่อยๆ เอาแขนลงช้าๆ ครับ',
    );
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

  it('translates AI Voice Coach prompts across 3 languages (Thai, English, Chinese) and matches voices', () => {
    expect(VOICE_COACH_LANGUAGES.map((l) => l.id)).toEqual(['th', 'en', 'zh']);
    expect(VOICE_COACH_LANGUAGES.map((l) => l.bcp47)).toEqual(['th-TH', 'en-US', 'zh-CN']);

    // 1. Dynamic target ROM angle
    const romTh = 'ค่อยๆ เคลื่อนไหวไปยังเป้าหมาย 145 องศาครับ';
    expect(translateVoiceCoachText(romTh, 'th')).toBe(romTh);
    expect(translateVoiceCoachText(romTh, 'en')).toBe(
      'Slowly move toward the target angle of 145 degrees.',
    );
    expect(translateVoiceCoachText(romTh, 'zh')).toBe('请慢慢移动至目标角度 145 度。');

    // 2. Dynamic hold phase seconds
    const holdTh = 'ดีมากครับ ค้างท่าไว้ 3 วินาที';
    expect(translateVoiceCoachText(holdTh, 'en')).toBe('Very good! Hold this position for 3 seconds.');
    expect(translateVoiceCoachText(holdTh, 'zh')).toBe('很好！请保持这个姿势 3 秒。');

    // 3. Dynamic rep completion and set finished with patient name & hold-done lower-limb guidance
    const repTh = 'ครบ 3 วินาทีแล้ว ดีมากครับ ครบ 5 ครั้ง ค่อยๆ เอาแขนลงช้าๆ ครับ';
    expect(translateVoiceCoachText(repTh, 'en')).toBe(
      '3 seconds completed! Very good! 5 reps completed. Slowly lower your arm back down.',
    );
    expect(translateVoiceCoachText(repTh, 'zh')).toBe(
      '已保持 3 秒！很好！已完成 5 次，请慢慢将手臂放下来。',
    );

    const setTh = 'ครบ 3 วินาทีแล้ว เยี่ยมมากครับ คุณสมชาย ใจดี ฝึกครบ 10 ครั้ง จบเซ็ตแล้ว ค่อยๆ เอาแขนลงช้าๆ ครับ';
    expect(translateVoiceCoachText(setTh, 'en')).toBe(
      '3 seconds completed! Great job, คุณสมชาย ใจดี! You completed 10 reps and finished the set. Slowly lower your arm back down.',
    );
    expect(translateVoiceCoachText(setTh, 'zh')).toBe(
      '已保持 3 秒！太棒了，คุณสมชาย ใจดี！已完成全部 10 次，本组训练结束。请慢慢把手臂放下来。',
    );

    // 4. Static alerts: wrong limb, trunk lean, multi-person, impostor, emergency halt
    const wrongRightArm = getPrescribedLimbPromptTh('right_shoulder').voicePromptTh;
    expect(translateVoiceCoachText(wrongRightArm, 'en')).toContain('right arm');
    expect(translateVoiceCoachText(wrongRightArm, 'zh')).toContain('右臂');

    expect(translateVoiceCoachText('ยืดตัวตรงครับ อย่าเอียงลำตัวช่วยยกนะครับ', 'en')).toContain(
      'back straight',
    );
    expect(translateVoiceCoachText('ยืดตัวตรงครับ อย่าเอียงลำตัวช่วยยกนะครับ', 'zh')).toContain(
      '挺直背部',
    );

    expect(
      translateVoiceCoachText(
        'ตรวจพบมีบุคคลอยู่หลายคนในเฟรม กรุณาให้คนไข้ยืนหน้ากล้องเพียงลำพัง ระบบจึงจะนับรอบการฝึกต่อครับ',
        'zh',
      ),
    ).toContain('检测到多人');
    expect(
      translateVoiceCoachText(
        'คุณไม่ใช่คนไข้ที่ลงทะเบียนไว้ ระบบจะไม่นับรอบการฝึก กรุณาให้คนไข้ตัวจริงเข้าฝึกครับ',
        'en',
      ),
    ).toContain('Identity mismatch');
    expect(translateVoiceCoachText('หยุดพักทันทีนะครับ ระบบกำลังเรียกเจ้าหน้าที่ดูแล', 'en')).toContain(
      'stop and rest immediately',
    );
    expect(translateVoiceCoachText('หยุดพักทันทีนะครับ ระบบกำลังเรียกเจ้าหน้าที่ดูแล', 'zh')).toContain(
      '停止并休息',
    );

    // 5. Voice selection prioritizes matching language tags
    const fakeVoices = [
      { lang: 'en-GB', name: 'English UK' },
      { lang: 'en-US', name: 'Microsoft Aria Natural - English (United States)' },
      { lang: 'th-TH', name: 'Microsoft Premwadee Natural - Thai (Thailand)' },
      { lang: 'zh-CN', name: 'Microsoft Xiaoxiao Natural - Chinese (Simplified)' },
    ] as unknown as SpeechSynthesisVoice[];

    expect(pickSpeechVoice(fakeVoices, 'th-TH')?.lang).toBe('th-TH');
    expect(pickSpeechVoice(fakeVoices, 'en-US')?.lang).toBe('en-US');
    expect(pickSpeechVoice(fakeVoices, 'zh-CN')?.lang).toBe('zh-CN');

    // 6. Proxy TTS URL builder encodes language & clamps long prompts
    const thProxyUrl = buildTtsProxyUrl('  ค่อยๆ เคลื่อนไหวไปยังเป้าหมาย 145 องศาครับ  ', 'th');
    expect(thProxyUrl.startsWith('/api/tts?')).toBe(true);
    const thParams = new URLSearchParams(thProxyUrl.split('?')[1]);
    expect(thParams.get('lang')).toBe('th');
    expect(thParams.get('text')).toBe('ค่อยๆ เคลื่อนไหวไปยังเป้าหมาย 145 องศาครับ');

    const longText = 'คำแนะนำ '.repeat(40);
    const clampedParams = new URLSearchParams(buildTtsProxyUrl(longText, 'zh').split('?')[1]);
    expect(clampedParams.get('lang')).toBe('zh');
    expect((clampedParams.get('text') || '').length).toBeLessThanOrEqual(190);
  });

  it('smoothly tracks real-time ROM angles with anti-jitter deadband and responsive slew rate', () => {
    // 1. Ignores micro-jitter within < 0.35 deg deadband
    expect(stepSmoothRomAngle(45, 45.3, 16)).toBe(45);
    expect(stepSmoothRomAngle(45, 44.7, 16)).toBe(45);

    // 2. Bounds single-frame spikes (15 -> 150 deg in 16ms) to <= 1800 deg/sec
    const singleStep = stepSmoothRomAngle(15, 150, 16);
    expect(singleStep).toBeGreaterThan(15);
    expect(singleStep).toBeLessThanOrEqual(15 + (1800 * 16) / 1000 + 1e-6);

    // 3. Converges to >= 149 deg within <= 16 frames (~0.27s at 60Hz) for fast arm raises
    let rom = 15;
    for (let i = 0; i < 16; i += 1) {
      rom = stepSmoothRomAngle(rom, 150, 1000 / 60);
    }
    expect(rom).toBeGreaterThanOrEqual(149);
    expect(rom).toBeLessThanOrEqual(150);

    // 4. Caps downward return speed to <= 1800 deg/sec
    const downStep = stepSmoothRomAngle(150, 15, 20);
    expect(downStep).toBeLessThan(150);
    expect(downStep).toBeGreaterThanOrEqual(150 - (1800 * 20) / 1000 - 1e-6);

    // 5. Still measures low-visibility close-up upper-body landmarks (visibility: 0.08)
    const closeUpRightArm = createNeutralLandmarks({
      [POSE_INDEX.RIGHT_SHOULDER]: { x: 0.42, y: 0.72, z: 0, visibility: 0.08 },
      [POSE_INDEX.RIGHT_ELBOW]: { x: 0.25, y: 0.42, z: 0, visibility: 0.08 },
      [POSE_INDEX.RIGHT_WRIST]: { x: 0.28, y: 0.14, z: 0, visibility: 0.08 },
    });
    const lowVisMeasure = measureLiveJointAngleAndTilt(closeUpRightArm, 'right_shoulder');
    expect(lowVisMeasure.wrongSideRaised).toBe(false);
    expect(lowVisMeasure.romDeg).not.toBeNull();
    expect(lowVisMeasure.romDeg!).toBeGreaterThan(145);
  });

  it('interpolates pose landmarks with stationary jitter damping and instant 1:1 tracking on active movement', () => {
    const base = createNeutralLandmarks({
      [POSE_INDEX.RIGHT_WRIST]: { x: 0.40, y: 0.65, z: 0, visibility: 0.99 },
    });

    // 1. Stationary micro-jitter (< 0.0015 norm units) is damped rather than snapping 100%
    const jitterTarget = createNeutralLandmarks({
      [POSE_INDEX.RIGHT_WRIST]: { x: 0.4008, y: 0.6505, z: 0, visibility: 0.99 },
    });
    const damped = interpolatePoseLandmarks(base, jitterTarget, 16);
    const dampedStep = damped[POSE_INDEX.RIGHT_WRIST].x - 0.40;
    expect(dampedStep).toBeGreaterThan(0);
    expect(dampedStep).toBeLessThan(0.0006);

    // 2. Active arm movement (dist >= 0.004 in 16ms) tracks 100% (1:1) with zero delay
    const fastSwingTarget = createNeutralLandmarks({
      [POSE_INDEX.RIGHT_WRIST]: { x: 0.46, y: 0.30, z: -0.25, visibility: 0.99 },
    });
    const pursued = interpolatePoseLandmarks(base, fastSwingTarget, 16);
    const yProgressRatio = (0.65 - pursued[POSE_INDEX.RIGHT_WRIST].y) / (0.65 - 0.30);
    expect(yProgressRatio).toBe(1);
    expect(pursued[POSE_INDEX.RIGHT_WRIST].x).toBe(0.46);
  });

  it('localizes the Verified Patient Confirmation Modal across Thai, English, and Chinese', () => {
    const sampleParams = {
      fullName: 'Pdasdwa',
      hn: 'HN-61716',
      nextAppointmentDate: '2026-11-20',
      confidencePercent: 77,
      twoHandProgressPct: 50,
    };

    const th = getKioskVerifiedModalCopy('th', sampleParams);
    expect(th.confidenceBadge).toContain('ความแม่นยำชีวมาตร 77%');
    expect(th.hnAndNextAppointment).toContain('นัดถัดไป: 2026-11-20');
    expect(th.startBtnIdle).toContain('4 วิ');
    expect(th.startBtnIdle).toContain('ต้อง 2 ข้างเท่านั้น');
    expect(th.gestureSingleHandPill).toContain('ยกมือเดียวไม่นับ');
    expect(th.gestureDetectedPill).toContain('50%');
    expect(th.gestureDetectedPill).toContain('2.0/4.0');
    expect(th.notMeBtn).toContain('ไม่ใช่ฉัน');
    expect(th.voiceGreeting).toContain('Pdasdwa');

    const en = getKioskVerifiedModalCopy('en', sampleParams);
    expect(en.confidenceBadge).toContain('Biometric Confidence 77%');
    expect(en.hnAndNextAppointment).toContain('Next Appointment: 2026-11-20');
    expect(en.startBtnIdle).toContain('Raise BOTH Hands 4s');
    expect(en.gestureDetectedPill).toContain('50%');
    expect(en.notMeBtn).toContain('Not me');
    expect(en.voiceGreeting).toContain('Welcome, Pdasdwa');

    const cn = getKioskVerifiedModalCopy('cn', sampleParams);
    expect(cn.confidenceBadge).toContain('生物识别准确度 77%');
    expect(cn.hnAndNextAppointment).toContain('下次预约: 2026-11-20');
    expect(cn.startBtnIdle).toContain('4 秒');
    expect(cn.gestureDetectedPill).toContain('50%');
    expect(cn.notMeBtn).toContain('不是我本人');
    expect(cn.voiceGreeting).toContain('Pdasdwa');
  });

  it('rejects single-hand raises and requires BOTH hands held for exact 4.0s to confirm', () => {
    expect(VERIFIED_HAND_HOLD_MS).toBe(4000);
    expect(HAND_SWITCH_GRACE_MS).toBeGreaterThanOrEqual(500);
    expect(HAND_GLITCH_GRACE_MS).toBeGreaterThanOrEqual(150);

    // 1. Single hand only MUST NOT count as bothUp
    const fakeHandL = Array.from({ length: 21 }, (_, idx) => ({
      x: 0.28 + (idx === 9 ? 0.01 : 0),
      y: 0.45,
    }));
    const singleHandRes = evaluateStrictTwoHandsForStartModal([fakeHandL], null);
    expect(singleHandRes.bothUp).toBe(false);
    expect(singleHandRes.singleOnly).toBe(true);

    // 2. Two separated hands raised cleanly -> bothUp === true
    const fakeHandR = Array.from({ length: 21 }, (_, idx) => ({
      x: 0.72 + (idx === 9 ? 0.01 : 0),
      y: 0.44,
    }));
    const bothHandsRes = evaluateStrictTwoHandsForStartModal([fakeHandL, fakeHandR], null);
    expect(bothHandsRes.leftUp).toBe(true);
    expect(bothHandsRes.rightUp).toBe(true);
    expect(bothHandsRes.bothUp).toBe(true);
    expect(bothHandsRes.singleOnly).toBe(false);

    // 3. Loading bar advances strictly when bothHandsNow=true over 4000ms (and never advances while dropped)
    let holdMs = 0;
    let lastRaisedAtMs = -Infinity;
    let nowMs = 1000;

    // Hold BOTH hands for 2.0s (40 ticks * 50ms) -> exact 50% (2000ms), not confirmed
    for (let i = 0; i < 40; i += 1) {
      nowMs += 50;
      const step = stepVerifiedHandHold({
        bothHandsNow: true,
        nowMs,
        dtMs: 50,
        lastRaisedAtMs,
        holdMs,
      });
      holdMs = step.holdMs;
      lastRaisedAtMs = step.lastRaisedAtMs;
      expect(step.detected).toBe(true);
      expect(step.confirmed).toBe(false);
    }
    expect(holdMs).toBe(2000);
    expect(holdMs / VERIFIED_HAND_HOLD_MS).toBe(0.5);

    // Drop one hand for 100ms (micro-glitch freeze: holds steady at 2000ms without adding fake time)
    for (let i = 0; i < 2; i += 1) {
      nowMs += 50;
      const step = stepVerifiedHandHold({
        bothHandsNow: false,
        nowMs,
        dtMs: 50,
        lastRaisedAtMs,
        holdMs,
      });
      holdMs = step.holdMs;
      lastRaisedAtMs = step.lastRaisedAtMs;
      expect(step.detected).toBe(false);
    }
    expect(holdMs).toBe(2000);

    // Hold BOTH hands for remaining 2.0s (40 * 50ms) -> completes exact 4.0s and confirms
    let finalConfirmed = false;
    for (let i = 0; i < 40; i += 1) {
      nowMs += 50;
      const step = stepVerifiedHandHold({
        bothHandsNow: true,
        nowMs,
        dtMs: 50,
        lastRaisedAtMs,
        holdMs,
      });
      holdMs = step.holdMs;
      lastRaisedAtMs = step.lastRaisedAtMs;
      finalConfirmed = step.confirmed;
    }
    expect(holdMs).toBe(4000);
    expect(finalConfirmed).toBe(true);
  });

  it('enforces movement plane strictly: forward flexion rejects side abduction and side abduction rejects forward flexion', () => {
    expect(
      resolveExpectedPlaneForExercise({
        name_th: 'ยกแขนตรงไปด้านหน้า (ขวา)',
        name_en: 'Right Shoulder Flexion',
        target_joint: 'right_shoulder',
      }),
    ).toBe('flexion');

    expect(
      resolveExpectedPlaneForExercise({
        name_th: 'กางแขนตรงออกด้านข้าง (ขวา)',
        name_en: 'Right Shoulder Abduction',
        target_joint: 'right_shoulder',
      }),
    ).toBe('abduction');

    // High forward flexion raise: valid for 'flexion'
    const forwardHigh = createNeutralLandmarks({
      [POSE_INDEX.RIGHT_ELBOW]: { x: 0.41, y: 0.22, z: 0, visibility: 0.99 },
      [POSE_INDEX.RIGHT_WRIST]: { x: 0.40, y: 0.08, z: 0, visibility: 0.99 },
    });
    const flexionHighCheck = measureLiveJointAngleAndTilt(forwardHigh, 'right_shoulder', 'flexion');
    expect(flexionHighCheck.wrongPlaneRaised).toBe(false);
    expect(flexionHighCheck.wrongPlaneBannerTh).toBeNull();
    expect(flexionHighCheck.romDeg).not.toBeNull();
    expect(flexionHighCheck.romDeg!).toBeGreaterThan(140);

    // Mid forward raise (~90 deg foreshortened in 2D webcam view): valid for 'flexion'
    const forwardMid = createNeutralLandmarks({
      [POSE_INDEX.RIGHT_ELBOW]: { x: 0.41, y: 0.34, z: 0, visibility: 0.99 },
      [POSE_INDEX.RIGHT_WRIST]: { x: 0.40, y: 0.34, z: 0, visibility: 0.99 },
    });
    const flexionMidCheck = measureLiveJointAngleAndTilt(forwardMid, 'right_shoulder', 'flexion');
    expect(flexionMidCheck.wrongPlaneRaised).toBe(false);
    expect(flexionMidCheck.romDeg).not.toBeNull();
    expect(flexionMidCheck.romDeg!).toBeGreaterThanOrEqual(85);
    expect(flexionMidCheck.romDeg!).toBeLessThanOrEqual(98);

    // Side abduction raise: rejected for 'flexion'
    const sideAbduction = createNeutralLandmarks({
      [POSE_INDEX.RIGHT_ELBOW]: { x: 0.24, y: 0.34, z: 0, visibility: 0.99 },
      [POSE_INDEX.RIGHT_WRIST]: { x: 0.12, y: 0.32, z: 0, visibility: 0.99 },
    });
    const flexionRejectsSide = measureLiveJointAngleAndTilt(sideAbduction, 'right_shoulder', 'flexion');
    expect(flexionRejectsSide.wrongPlaneRaised).toBe(true);
    expect(flexionRejectsSide.romDeg).toBe(0);
    expect(flexionRejectsSide.wrongPlaneBannerTh).toContain('ยกแขนตรงไปด้านหน้า');

    // Side abduction raise: valid for 'abduction'
    const abductionCheck = measureLiveJointAngleAndTilt(sideAbduction, 'right_shoulder', 'abduction');
    expect(abductionCheck.wrongPlaneRaised).toBe(false);
    expect(abductionCheck.wrongPlaneBannerTh).toBeNull();
    expect(abductionCheck.romDeg).not.toBeNull();
    expect(abductionCheck.romDeg!).toBeGreaterThan(90);

    // Forward raise at 90 deg: rejected for 'abduction'
    const abductionRejectsForward = measureLiveJointAngleAndTilt(forwardMid, 'right_shoulder', 'abduction');
    expect(abductionRejectsForward.wrongPlaneRaised).toBe(true);
    expect(abductionRejectsForward.romDeg).toBe(0);
    expect(abductionRejectsForward.wrongPlaneBannerTh).toContain('กางแขนออกด้านข้าง');

    // synthesizeDemoPayload WRONG_PLANE + 3-language Voice Coach translation
    const flexionExercise = {
      ...rightShoulder,
      exercise_id: 'ex-01',
      name_th: 'ยกแขนตรงไปด้านหน้า (ขวา)',
      name_en: 'Right Shoulder Flexion',
      expected_plane: 'flexion' as const,
    };
    const wrongPlanePayload = synthesizeDemoPayload({
      preset: 'WRONG_PLANE',
      romDeg: 0,
      trunkTiltDeg: 2,
      distanceM: 2.0,
      exercise: flexionExercise,
      validReps: 2,
      invalidReps: 0,
      holdRemainingSec: 3,
    });
    expect(wrongPlanePayload.telemetry_analysis.is_form_valid).toBe(false);
    expect(wrongPlanePayload.telemetry_analysis.compensation_detected).toContain('WRONG_PLANE');
    expect(translateVoiceCoachText(wrongPlanePayload.tts_voice_th, 'th')).toContain('ยกแขนตรงไปด้านหน้า');
    expect(translateVoiceCoachText(wrongPlanePayload.tts_voice_th, 'en')).toContain('straight forward');
    expect(translateVoiceCoachText(wrongPlanePayload.tts_voice_th, 'zh')).toContain('向正前方抬起');

    const abductionWrongPayload = synthesizeDemoPayload({
      preset: 'WRONG_PLANE',
      romDeg: 0,
      trunkTiltDeg: 2,
      distanceM: 2.0,
      exercise: {
        ...rightShoulder,
        expected_plane: 'abduction' as const,
      },
      validReps: 2,
      invalidReps: 0,
      holdRemainingSec: 3,
    });
    expect(translateVoiceCoachText(abductionWrongPayload.tts_voice_th, 'th')).toContain('กางแขนออกด้านข้าง');
    expect(translateVoiceCoachText(abductionWrongPayload.tts_voice_th, 'en')).toContain('out to the side');
    expect(translateVoiceCoachText(abductionWrongPayload.tts_voice_th, 'zh')).toContain('向身体侧面外展抬起');
  });

  it('measures forward flexion progressively across 3D X-Y-Z axes without jumping prematurely to 180 deg', () => {
    // 1. Close-up 16:9 webcam framing (hips occluded below frame, shoulderSpanX = 0.28, sh.y = 0.82)
    const eyeLevelCloseUp = createNeutralLandmarks({
      [POSE_INDEX.LEFT_SHOULDER]: { x: 0.64, y: 0.82, z: 0, visibility: 0.99 },
      [POSE_INDEX.RIGHT_SHOULDER]: { x: 0.36, y: 0.82, z: 0, visibility: 0.99 },
      [POSE_INDEX.LEFT_HIP]: { x: 0.60, y: 1.15, z: 0, visibility: 0.05 },
      [POSE_INDEX.RIGHT_HIP]: { x: 0.40, y: 1.15, z: 0, visibility: 0.05 },
      [POSE_INDEX.RIGHT_ELBOW]: { x: 0.36, y: 0.58, z: -0.18, visibility: 0.99 },
      [POSE_INDEX.RIGHT_WRIST]: { x: 0.36, y: 0.37, z: -0.35, visibility: 0.99 },
    });
    const eyeCheck = measureLiveJointAngleAndTilt(eyeLevelCloseUp, 'right_shoulder', 'flexion');
    expect(eyeCheck.wrongPlaneRaised).toBe(false);
    expect(eyeCheck.romDeg).not.toBeNull();
    expect(eyeCheck.romDeg!).toBeGreaterThanOrEqual(115);
    expect(eyeCheck.romDeg!).toBeLessThanOrEqual(128);
    expect(eyeCheck.xyzVector).not.toBeNull();
    expect(eyeCheck.xyzVector!.z).toBeLessThan(-0.3);

    // 2. Target-level forward raise matching screenshot (wrist.y = 0.16, elbow.y = 0.45) -> ~145..160 deg, NOT 180 deg
    const targetLevelCloseUp = createNeutralLandmarks({
      [POSE_INDEX.LEFT_SHOULDER]: { x: 0.64, y: 0.82, z: 0, visibility: 0.99 },
      [POSE_INDEX.RIGHT_SHOULDER]: { x: 0.36, y: 0.82, z: 0, visibility: 0.99 },
      [POSE_INDEX.LEFT_HIP]: { x: 0.60, y: 1.15, z: 0, visibility: 0.05 },
      [POSE_INDEX.RIGHT_HIP]: { x: 0.40, y: 1.15, z: 0, visibility: 0.05 },
      [POSE_INDEX.RIGHT_ELBOW]: { x: 0.36, y: 0.45, z: -0.12, visibility: 0.99 },
      [POSE_INDEX.RIGHT_WRIST]: { x: 0.36, y: 0.16, z: -0.22, visibility: 0.99 },
    });
    const targetCheck = measureLiveJointAngleAndTilt(targetLevelCloseUp, 'right_shoulder', 'flexion');
    expect(targetCheck.wrongPlaneRaised).toBe(false);
    expect(targetCheck.romDeg).not.toBeNull();
    expect(targetCheck.romDeg!).toBeGreaterThanOrEqual(145);
    expect(targetCheck.romDeg!).toBeLessThanOrEqual(160);
    expect(targetCheck.xyzVector).not.toBeNull();
    expect(targetCheck.xyzVector!.z).toBeLessThan(-0.3);

    // 3. Explicit 3D worldLandmarks Z depth progression (45 deg -> 90 deg -> 135 deg)
    const world45 = createNeutralLandmarks({
      [POSE_INDEX.RIGHT_SHOULDER]: { x: 0, y: 0, z: 0, visibility: 0.99 },
      [POSE_INDEX.RIGHT_ELBOW]: { x: 0, y: 0.212, z: -0.212, visibility: 0.99 },
      [POSE_INDEX.RIGHT_WRIST]: { x: 0, y: 0.424, z: -0.424, visibility: 0.99 },
    });
    const pose45 = createNeutralLandmarks({
      [POSE_INDEX.RIGHT_ELBOW]: { x: 0.42, y: 0.46, z: -0.11, visibility: 0.99 },
      [POSE_INDEX.RIGHT_WRIST]: { x: 0.42, y: 0.56, z: -0.22, visibility: 0.99 },
    });
    const check45 = measureLiveJointAngleAndTilt(pose45, 'right_shoulder', 'flexion', world45);
    expect(check45.romDeg!).toBeGreaterThanOrEqual(38);
    expect(check45.romDeg!).toBeLessThanOrEqual(52);
    expect(check45.xyzVector!.z).toBeLessThan(-0.4);

    const world90 = createNeutralLandmarks({
      [POSE_INDEX.RIGHT_SHOULDER]: { x: 0, y: 0, z: 0, visibility: 0.99 },
      [POSE_INDEX.RIGHT_ELBOW]: { x: 0, y: 0, z: -0.30, visibility: 0.99 },
      [POSE_INDEX.RIGHT_WRIST]: { x: 0, y: 0, z: -0.60, visibility: 0.99 },
    });
    const pose90 = createNeutralLandmarks({
      [POSE_INDEX.RIGHT_ELBOW]: { x: 0.42, y: 0.35, z: -0.18, visibility: 0.99 },
      [POSE_INDEX.RIGHT_WRIST]: { x: 0.42, y: 0.35, z: -0.35, visibility: 0.99 },
    });
    const check90 = measureLiveJointAngleAndTilt(pose90, 'right_shoulder', 'flexion', world90);
    expect(check90.romDeg!).toBeGreaterThanOrEqual(85);
    expect(check90.romDeg!).toBeLessThanOrEqual(95);
    expect(check90.xyzVector!.z).toBeLessThan(-0.9);
  });

  it('waits for current Voice Coach utterance to finish before speaking next prompt and preserves rep completion speech', () => {
    // 1. Active utterance is speaking -> incoming HOLD_PHASE queues after current sentence finishes
    const holdWhileSpeaking = decideVoiceCoachSpeechTurn({
      isCurrentlySpeaking: true,
      incomingText: 'คงท่าค้างไว้ 3 วินาทีครับ',
      incomingLang: 'th',
      incomingState: 'HOLD_PHASE',
      lastSpokenKey: 'th:ค่อยๆ เคลื่อนไหวไปยังเป้าหมาย 145 องศาครับ',
      lastSpokenState: 'ACTIVE_REP',
      lastSpokenLang: 'th',
      elapsedSinceLastSpokenMs: 900,
    });
    expect(holdWhileSpeaking.action).toBe('QUEUE_AFTER_CURRENT');
    expect(holdWhileSpeaking.queuedItem?.text).toBe('คงท่าค้างไว้ 3 วินาทีครับ');
    expect(holdWhileSpeaking.queuedItem?.isCompletion).toBe(false);

    // 2. While HOLD_PHASE is speaking, REP_COMPLETED arrives and queues with isCompletion: true;
    // a subsequent ACTIVE_REP frame must NOT overwrite the queued REP_COMPLETED speech.
    const completionQueued = decideVoiceCoachSpeechTurn({
      isCurrentlySpeaking: true,
      incomingText: 'ครบ 3 วินาทีแล้ว ดีมากครับ ครบ 1 ครั้ง ค่อยๆ เอาแขนลงช้าๆ ครับ',
      incomingLang: 'th',
      incomingState: 'REP_COMPLETED',
      lastSpokenKey: 'th:คงท่าค้างไว้ 3 วินาทีครับ',
      lastSpokenState: 'HOLD_PHASE',
      lastSpokenLang: 'th',
      elapsedSinceLastSpokenMs: 1200,
    });
    expect(completionQueued.action).toBe('QUEUE_AFTER_CURRENT');
    expect(completionQueued.queuedItem?.isCompletion).toBe(true);

    const activeRepAfterCompletion = decideVoiceCoachSpeechTurn({
      isCurrentlySpeaking: true,
      incomingText: 'ค่อยๆ เคลื่อนไหวไปยังเป้าหมาย 145 องศาครับ',
      incomingLang: 'th',
      incomingState: 'ACTIVE_REP',
      lastSpokenKey: 'th:คงท่าค้างไว้ 3 วินาทีครับ',
      lastSpokenState: 'HOLD_PHASE',
      lastSpokenLang: 'th',
      elapsedSinceLastSpokenMs: 1400,
      existingQueued: completionQueued.queuedItem,
    });
    expect(activeRepAfterCompletion.action).toBe('QUEUE_AFTER_CURRENT');
    expect(activeRepAfterCompletion.queuedItem?.isCompletion).toBe(true);
    expect(activeRepAfterCompletion.queuedItem?.text).toContain('ครบ 3 วินาทีแล้ว');

    // 3. While REP_COMPLETED is actively speaking, incoming ACTIVE_REP queues rather than cutting off
    const activeWhileCompletionSpeaking = decideVoiceCoachSpeechTurn({
      isCurrentlySpeaking: true,
      incomingText: 'ค่อยๆ เคลื่อนไหวไปยังเป้าหมาย 145 องศาครับ',
      incomingLang: 'th',
      incomingState: 'ACTIVE_REP',
      lastSpokenKey: 'th:ครบ 3 วินาทีแล้ว ดีมากครับ ครบ 1 ครั้ง ค่อยๆ เอาแขนลงช้าๆ ครับ',
      lastSpokenState: 'REP_COMPLETED',
      lastSpokenLang: 'th',
      elapsedSinceLastSpokenMs: 800,
    });
    expect(activeWhileCompletionSpeaking.action).toBe('QUEUE_AFTER_CURRENT');

    // 4. Duplicate HOLD_PHASE 1s countdown ticks within 4500ms are skipped even when idle
    const holdTickSkipped = decideVoiceCoachSpeechTurn({
      isCurrentlySpeaking: false,
      incomingText: 'ดีมากครับ ค้างท่าไว้อีก 2 วินาที',
      incomingLang: 'th',
      incomingState: 'HOLD_PHASE',
      lastSpokenKey: 'th:คงท่าค้างไว้ 3 วินาทีครับ',
      lastSpokenState: 'HOLD_PHASE',
      lastSpokenLang: 'th',
      elapsedSinceLastSpokenMs: 2100,
    });
    expect(holdTickSkipped.action).toBe('SKIP');

    // 5. EMERGENCY_HALT and explicit forceInterrupt bypass the wait immediately
    const emergencyDecision = decideVoiceCoachSpeechTurn({
      isCurrentlySpeaking: true,
      incomingText: 'หยุดฉุกเฉินครับ กรุณานั่งพัก',
      incomingLang: 'th',
      incomingState: 'EMERGENCY_HALT',
      lastSpokenKey: 'th:ค่อยๆ เคลื่อนไหวไปยังเป้าหมาย 145 องศาครับ',
      lastSpokenState: 'ACTIVE_REP',
      lastSpokenLang: 'th',
      elapsedSinceLastSpokenMs: 400,
    });
    expect(emergencyDecision.action).toBe('SPEAK_IMMEDIATELY');

    const manualReplayDecision = decideVoiceCoachSpeechTurn({
      isCurrentlySpeaking: true,
      forceInterrupt: true,
      incomingText: 'Please raise your arm straight forward.',
      incomingLang: 'en',
      incomingState: 'ACTIVE_REP',
      lastSpokenKey: 'th:ค่อยๆ เคลื่อนไหวไปยังเป้าหมาย 145 องศาครับ',
      lastSpokenState: 'ACTIVE_REP',
      lastSpokenLang: 'th',
      elapsedSinceLastSpokenMs: 400,
    });
    expect(manualReplayDecision.action).toBe('SPEAK_IMMEDIATELY');
  });

  it('supports toggling real-time form assessment master switch and individual rules ON/OFF', () => {
    expect(DEFAULT_FORM_ASSESSMENT_RULES).toEqual({
      enabled: true,
      limb: true,
      plane: true,
      trunk: true,
      posture: true,
      velocity: true,
      identity: true,
    });

    const ruleKeys = ['limb', 'plane', 'trunk', 'posture', 'velocity', 'identity'] as const;
    for (const key of ruleKeys) {
      expect(isFormRuleActive(DEFAULT_FORM_ASSESSMENT_RULES, key)).toBe(true);
    }

    // Master switch OFF disables every rule immediately without mutating per-rule preferences
    const masterDisabled = { ...DEFAULT_FORM_ASSESSMENT_RULES, enabled: false };
    for (const key of ruleKeys) {
      expect(isFormRuleActive(masterDisabled, key)).toBe(false);
    }

    // Individual rule toggle turns OFF only the targeted rule while keeping others ON
    const planeOnlyDisabled = { ...DEFAULT_FORM_ASSESSMENT_RULES, plane: false, trunk: false };
    expect(isFormRuleActive(planeOnlyDisabled, 'plane')).toBe(false);
    expect(isFormRuleActive(planeOnlyDisabled, 'trunk')).toBe(false);
    expect(isFormRuleActive(planeOnlyDisabled, 'limb')).toBe(true);
    expect(isFormRuleActive(planeOnlyDisabled, 'posture')).toBe(true);
    expect(isFormRuleActive(planeOnlyDisabled, 'velocity')).toBe(true);
    expect(isFormRuleActive(planeOnlyDisabled, 'identity')).toBe(true);
  });

  it('builds 3-language entry voice narrations within 190 chars for completed-today, rest-day, and scheduled overview screens', () => {
    const langs = ['th', 'en', 'cn'] as const;
    const statuses = ['completed_today', 'rest_day', 'scheduled_today'] as const;
    for (const status of statuses) {
      for (const lang of langs) {
        const text = getKioskOverviewVoiceNarration(lang, {
          status,
          fullName: 'Pize',
          handSide: 'left',
        });
        expect(text).toContain('Pize');
        expect(text.length).toBeGreaterThan(10);
        expect(text.length).toBeLessThanOrEqual(190);
      }
    }
    expect(
      getKioskOverviewVoiceNarration('th', { status: 'completed_today', fullName: 'Pize' }),
    ).toContain('คุณได้ทำการกายภาพของวันนี้ไปแล้ว');
    expect(
      getKioskOverviewVoiceNarration('en', { status: 'completed_today', fullName: 'Pize' }),
    ).toContain("already completed today's physical therapy");
    expect(
      getKioskOverviewVoiceNarration('cn', { status: 'completed_today', fullName: 'Pize' }),
    ).toContain('您今天已完成物理康复训练');
    expect(
      getKioskOverviewVoiceNarration('th', {
        status: 'scheduled_today',
        fullName: 'Pize',
        handSide: 'right',
      }),
    ).toContain('มือขวา');
  });

  it('strips accidental knee default prefix when followed by chosen exercises while keeping explicit knee programs', () => {
    const baseItem = { patientId: 'p1', ptId: 'pt1', scheduledDate: '2026-11-15', sets: 3, reps: 10, status: 'ACTIVE' as const };
    const contaminated = [
      { ...baseItem, id: 's1', exerciseId: 'ex-13' },
      { ...baseItem, id: 's2', exerciseId: 'ex-14' },
      { ...baseItem, id: 's3', exerciseId: 'ex-26' },
      { ...baseItem, id: 's4', exerciseId: 'ex-01' },
      { ...baseItem, id: 's5', exerciseId: 'ex-03' },
      { ...baseItem, id: 's6', exerciseId: 'ex-02' },
      { ...baseItem, id: 's7', exerciseId: 'ex-04' },
    ];
    const cleaned = sanitizeAccidentalDefaultKneePrefix(contaminated);
    expect(cleaned.map((x) => x.exerciseId)).toEqual(['ex-01', 'ex-03', 'ex-02', 'ex-04']);

    const pureKnee = contaminated.slice(0, 3);
    expect(sanitizeAccidentalDefaultKneePrefix(pureKnee).map((x) => x.exerciseId)).toEqual([
      'ex-13',
      'ex-14',
      'ex-26',
    ]);

    const todayResolved = resolveTodayPatientSchedule({
      rehabSchedules: contaminated,
      now: new Date('2026-11-15T10:00:00'),
    });
    expect(todayResolved.todayExercises.map((e) => e.exerciseId)).toEqual([
      'ex-01',
      'ex-03',
      'ex-02',
      'ex-04',
    ]);

    const restDayFallback = resolvePrescribedExercisesFromSchedules([
      { ...contaminated[3], scheduledDate: '2026-11-20' },
      { ...contaminated[5], scheduledDate: '2026-11-20' },
    ]);
    expect(restDayFallback.map((e) => e.exerciseId)).toEqual(['ex-01', 'ex-02']);
  });

  it('tracks full 0..180 deg shoulder range without overhead ceiling and returns null for out-of-frame knees', () => {
    // Seated distance (~1.8m, shoulderSpanX = 0.17, hips below frame)
    const seatedRest = createNeutralLandmarks({
      [POSE_INDEX.LEFT_SHOULDER]: { x: 0.585, y: 0.82, z: 0, visibility: 0.99 },
      [POSE_INDEX.RIGHT_SHOULDER]: { x: 0.415, y: 0.82, z: 0, visibility: 0.99 },
      [POSE_INDEX.LEFT_HIP]: { x: 0.56, y: 1.25, z: 0, visibility: 0.02 },
      [POSE_INDEX.RIGHT_HIP]: { x: 0.44, y: 1.25, z: 0, visibility: 0.02 },
      [POSE_INDEX.RIGHT_ELBOW]: { x: 0.41, y: 0.96, z: 0, visibility: 0.99 },
      [POSE_INDEX.RIGHT_WRIST]: { x: 0.41, y: 0.99, z: 0, visibility: 0.99 },
    });
    const restCheck = measureLiveJointAngleAndTilt(seatedRest, 'right_shoulder', 'flexion');
    expect(restCheck.romDeg).not.toBeNull();
    expect(restCheck.romDeg!).toBeLessThan(12);

    const seatedOverhead = createNeutralLandmarks({
      [POSE_INDEX.LEFT_SHOULDER]: { x: 0.585, y: 0.86, z: 0, visibility: 0.99 },
      [POSE_INDEX.RIGHT_SHOULDER]: { x: 0.415, y: 0.86, z: 0, visibility: 0.99 },
      [POSE_INDEX.LEFT_HIP]: { x: 0.56, y: 1.25, z: 0, visibility: 0.02 },
      [POSE_INDEX.RIGHT_HIP]: { x: 0.44, y: 1.25, z: 0, visibility: 0.02 },
      [POSE_INDEX.RIGHT_ELBOW]: { x: 0.415, y: 0.63, z: -0.02, visibility: 0.99 },
      [POSE_INDEX.RIGHT_WRIST]: { x: 0.415, y: 0.40, z: -0.03, visibility: 0.99 },
    });
    const overhead2dCheck = measureLiveJointAngleAndTilt(seatedOverhead, 'right_shoulder', 'flexion');
    expect(overhead2dCheck.romDeg).not.toBeNull();
    expect(overhead2dCheck.romDeg!).toBeGreaterThanOrEqual(168);

    // With worldLandmarks straight overhead -> ~172..180 deg even under extreme 2D foreshortening
    const worldOverhead = createNeutralLandmarks({
      [POSE_INDEX.RIGHT_SHOULDER]: { x: 0, y: 0, z: 0, visibility: 0.99 },
      [POSE_INDEX.RIGHT_ELBOW]: { x: 0, y: -0.28, z: -0.02, visibility: 0.99 },
      [POSE_INDEX.RIGHT_WRIST]: { x: 0, y: -0.56, z: -0.04, visibility: 0.99 },
    });
    const worldOverheadCheck = measureLiveJointAngleAndTilt(
      seatedOverhead,
      'right_shoulder',
      'flexion',
      worldOverhead,
    );
    expect(worldOverheadCheck.romDeg).not.toBeNull();
    expect(worldOverheadCheck.romDeg!).toBeGreaterThanOrEqual(170);

    // Out-of-frame knees (only head/shoulders visible) return null instead of static 170 deg
    const upperBodyOnly = createNeutralLandmarks({
      [POSE_INDEX.RIGHT_HIP]: { x: 0.44, y: 1.1, z: 0, visibility: 0.3 },
      [POSE_INDEX.LEFT_HIP]: { x: 0.56, y: 1.1, z: 0, visibility: 0.3 },
      [POSE_INDEX.RIGHT_KNEE]: { x: 0.44, y: 1.35, z: 0, visibility: 0.04 },
      [POSE_INDEX.RIGHT_ANKLE]: { x: 0.44, y: 1.55, z: 0, visibility: 0.02 },
    });
    const kneeOutCheck = measureLiveJointAngleAndTilt(upperBodyOnly, 'right_knee');
    expect(kneeOutCheck.romDeg).toBeNull();
  });

  it('stabilizes torso anchor when hand occlusion collapses shoulders and rescues foreshortened forward reach without false WRONG_PLANE', () => {
    // 1. Valid baseline frame establishes anatomical torso anchor
    const validTorsoFrame = createNeutralLandmarks({
      [POSE_INDEX.LEFT_SHOULDER]: { x: 0.62, y: 0.55, z: 0, visibility: 0.96 },
      [POSE_INDEX.RIGHT_SHOULDER]: { x: 0.38, y: 0.55, z: 0, visibility: 0.96 },
      [POSE_INDEX.NOSE]: { x: 0.5, y: 0.38, z: -0.05, visibility: 0.96 },
      [POSE_INDEX.RIGHT_ELBOW]: { x: 0.41, y: 0.56, z: -0.15, visibility: 0.92 },
      [POSE_INDEX.RIGHT_WRIST]: { x: 0.44, y: 0.58, z: -0.32, visibility: 0.92 },
    });
    const baseStab = sanitizeAndStabilizePoseLandmarks(validTorsoFrame, null, 'right_shoulder');
    expect(baseStab.occludedFallback).toBe(false);
    expect(baseStab.anchor).not.toBeNull();

    // 2. Occluded frame where MediaPipe collapses shoulders/hips/knees onto outstretched hand at center
    const collapsedHandFrame = createNeutralLandmarks({
      [POSE_INDEX.LEFT_SHOULDER]: { x: 0.49, y: 0.52, z: 0, visibility: 0.72 },
      [POSE_INDEX.RIGHT_SHOULDER]: { x: 0.44, y: 0.56, z: 0, visibility: 0.75 },
      [POSE_INDEX.RIGHT_ELBOW]: { x: 0.46, y: 0.56, z: -0.08, visibility: 0.85 },
      [POSE_INDEX.RIGHT_WRIST]: { x: 0.45, y: 0.56, z: -0.12, visibility: 0.95 },
      [POSE_INDEX.LEFT_HIP]: { x: 0.5, y: 0.58, z: 0, visibility: 0.45 },
      [POSE_INDEX.RIGHT_HIP]: { x: 0.45, y: 0.58, z: 0, visibility: 0.45 },
      [POSE_INDEX.RIGHT_KNEE]: { x: 0.46, y: 0.61, z: 0, visibility: 0.4 },
    });
    const rescuedStab = sanitizeAndStabilizePoseLandmarks(
      collapsedHandFrame,
      baseStab.anchor,
      'right_shoulder',
    );
    expect(rescuedStab.occludedFallback).toBe(true);
    expect(rescuedStab.landmarks[POSE_INDEX.LEFT_SHOULDER].x).toBe(0.62);
    expect(rescuedStab.landmarks[POSE_INDEX.RIGHT_SHOULDER].x).toBe(0.38);
    expect(rescuedStab.landmarks[POSE_INDEX.LEFT_HIP].visibility).toBe(0);
    expect(rescuedStab.landmarks[POSE_INDEX.RIGHT_KNEE].visibility).toBe(0);

    // 3. Even if worldLandmarks under-reports 3D depth (~34 deg) under hand self-occlusion,
    // foreshortened forward reach measures ~75..98 deg and NEVER triggers WRONG_PLANE or TRUNK_LEAN
    const noisyWorldOcclusion = createNeutralLandmarks({
      [POSE_INDEX.RIGHT_SHOULDER]: { x: 0, y: 0, z: 0, visibility: 0.95 },
      [POSE_INDEX.RIGHT_ELBOW]: { x: 0.02, y: 0.16, z: -0.11, visibility: 0.85 },
      [POSE_INDEX.RIGHT_WRIST]: { x: 0.03, y: 0.30, z: -0.20, visibility: 0.9 },
    });
    const stabilizedForwardFrame = sanitizeAndStabilizePoseLandmarks(
      createNeutralLandmarks({
        [POSE_INDEX.LEFT_SHOULDER]: { x: 0.62, y: 0.55, z: 0, visibility: 0.96 },
        [POSE_INDEX.RIGHT_SHOULDER]: { x: 0.38, y: 0.55, z: 0, visibility: 0.96 },
        [POSE_INDEX.LEFT_HIP]: { x: 0.58, y: 1.05, z: 0, visibility: 0.05 },
        [POSE_INDEX.RIGHT_HIP]: { x: 0.42, y: 1.05, z: 0, visibility: 0.05 },
        [POSE_INDEX.RIGHT_ELBOW]: { x: 0.41, y: 0.57, z: -0.05, visibility: 0.9 },
        [POSE_INDEX.RIGHT_WRIST]: { x: 0.44, y: 0.58, z: -0.06, visibility: 0.95 },
      }),
      baseStab.anchor,
      'right_shoulder',
    ).landmarks;
    const forwardOcclusionCheck = measureLiveJointAngleAndTilt(
      stabilizedForwardFrame,
      'right_shoulder',
      'flexion',
      noisyWorldOcclusion,
    );
    expect(forwardOcclusionCheck.wrongPlaneRaised).toBe(false);
    expect(forwardOcclusionCheck.trunkTiltDeg).toBeLessThan(10);
    expect(forwardOcclusionCheck.romDeg).not.toBeNull();
    expect(forwardOcclusionCheck.romDeg!).toBeGreaterThanOrEqual(75);
    expect(forwardOcclusionCheck.romDeg!).toBeLessThanOrEqual(100);
  });

  it('tracks rapid bending down without freezing shoulders, repairs out-of-frame overhead wrists, and detects elbow flexion/neck/hip-march accurately', () => {
    const baseline = sanitizeAndStabilizePoseLandmarks(
      createNeutralLandmarks({
        [POSE_INDEX.LEFT_SHOULDER]: { x: 0.62, y: 0.45, z: 0, visibility: 0.96 },
        [POSE_INDEX.RIGHT_SHOULDER]: { x: 0.38, y: 0.45, z: 0, visibility: 0.96 },
      }),
      null,
      'right_shoulder',
    );
    expect(baseline.occludedFallback).toBe(false);

    // Rapid bend downward with tilted shoulders (midJump > 0.20) must follow live without freezing
    const bentDown = sanitizeAndStabilizePoseLandmarks(
      createNeutralLandmarks({
        [POSE_INDEX.LEFT_SHOULDER]: { x: 0.63, y: 0.68, z: -0.04, visibility: 0.9 },
        [POSE_INDEX.RIGHT_SHOULDER]: { x: 0.39, y: 0.78, z: -0.04, visibility: 0.88 },
      }),
      baseline.anchor,
      'right_shoulder',
    );
    expect(bentDown.occludedFallback).toBe(false);
    expect(bentDown.landmarks[POSE_INDEX.RIGHT_SHOULDER].y).toBeCloseTo(0.78, 2);

    // Collapsed torso span (< 0.085) falls back to anchor
    const collapsed = sanitizeAndStabilizePoseLandmarks(
      createNeutralLandmarks({
        [POSE_INDEX.LEFT_SHOULDER]: { x: 0.52, y: 0.6, z: 0, visibility: 0.8 },
        [POSE_INDEX.RIGHT_SHOULDER]: { x: 0.46, y: 0.6, z: 0, visibility: 0.8 },
      }),
      bentDown.anchor,
      'right_shoulder',
    );
    expect(collapsed.occludedFallback).toBe(true);

    // Overhead out-of-frame wrist foldback repair
    const overheadFoldback = sanitizeAndStabilizePoseLandmarks(
      createNeutralLandmarks({
        [POSE_INDEX.LEFT_SHOULDER]: { x: 0.62, y: 0.55, z: 0, visibility: 0.95 },
        [POSE_INDEX.RIGHT_SHOULDER]: { x: 0.38, y: 0.55, z: 0, visibility: 0.95 },
        [POSE_INDEX.RIGHT_ELBOW]: { x: 0.38, y: 0.32, z: 0, visibility: 0.88 },
        [POSE_INDEX.RIGHT_WRIST]: { x: 0.42, y: 0.75, z: 0, visibility: 0.35 },
      }),
      baseline.anchor,
      'right_shoulder',
    );
    expect(overheadFoldback.landmarks[POSE_INDEX.RIGHT_WRIST].y).toBeLessThan(
      overheadFoldback.landmarks[POSE_INDEX.RIGHT_ELBOW].y,
    );

    // Elbow flexion exercise (ex-05): hanging arm < 20 deg vs flexed curl >= 85 deg
    const elbowMeta = {
      exercise_id: 'ex-05',
      name_th: 'งอ-เหยียดข้อศอก (ขวา)',
      name_en: 'Right Elbow Flexion',
    };
    const hangingElbow = createNeutralLandmarks({
      [POSE_INDEX.LEFT_SHOULDER]: { x: 0.62, y: 0.4, z: 0, visibility: 0.95 },
      [POSE_INDEX.RIGHT_SHOULDER]: { x: 0.38, y: 0.4, z: 0, visibility: 0.95 },
      [POSE_INDEX.RIGHT_ELBOW]: { x: 0.36, y: 0.58, z: 0, visibility: 0.95 },
      [POSE_INDEX.RIGHT_WRIST]: { x: 0.35, y: 0.76, z: 0, visibility: 0.95 },
    });
    const hangingCheck = measureLiveJointAngleAndTilt(
      hangingElbow,
      'right_shoulder',
      'any',
      undefined,
      elbowMeta,
    );
    expect(hangingCheck.romDeg).not.toBeNull();
    expect(hangingCheck.romDeg!).toBeLessThan(20);

    const flexedElbow = createNeutralLandmarks({
      [POSE_INDEX.LEFT_SHOULDER]: { x: 0.62, y: 0.4, z: 0, visibility: 0.95 },
      [POSE_INDEX.RIGHT_SHOULDER]: { x: 0.38, y: 0.4, z: 0, visibility: 0.95 },
      [POSE_INDEX.RIGHT_ELBOW]: { x: 0.37, y: 0.6, z: 0, visibility: 0.95 },
      [POSE_INDEX.RIGHT_WRIST]: { x: 0.38, y: 0.44, z: -0.05, visibility: 0.95 },
    });
    const flexedCheck = measureLiveJointAngleAndTilt(
      flexedElbow,
      'right_shoulder',
      'any',
      undefined,
      elbowMeta,
    );
    expect(flexedCheck.wrongPlaneRaised).toBe(false);
    expect(flexedCheck.romDeg).not.toBeNull();
    expect(flexedCheck.romDeg!).toBeGreaterThanOrEqual(85);

    // Neck tilt (ex-19) & standing knee march (ex-15)
    const neckTilted = createNeutralLandmarks({
      [POSE_INDEX.NOSE]: { x: 0.43, y: 0.28, z: 0, visibility: 0.95 },
      [POSE_INDEX.LEFT_SHOULDER]: { x: 0.62, y: 0.45, z: 0, visibility: 0.95 },
      [POSE_INDEX.RIGHT_SHOULDER]: { x: 0.38, y: 0.45, z: 0, visibility: 0.95 },
    });
    const neckCheck = measureLiveJointAngleAndTilt(neckTilted, 'spine', 'any', undefined, {
      exercise_id: 'ex-19',
      name_th: 'เอียงคอช้าๆ ด้านขวา',
    });
    expect(neckCheck.romDeg).not.toBeNull();
    expect(neckCheck.romDeg!).toBeGreaterThanOrEqual(25);

    const kneeMarched = createNeutralLandmarks({
      [POSE_INDEX.RIGHT_HIP]: { x: 0.42, y: 0.55, z: 0, visibility: 0.95 },
      [POSE_INDEX.RIGHT_KNEE]: { x: 0.42, y: 0.55, z: -0.25, visibility: 0.95 },
      [POSE_INDEX.RIGHT_ANKLE]: { x: 0.42, y: 0.85, z: -0.25, visibility: 0.95 },
    });
    const marchCheck = measureLiveJointAngleAndTilt(kneeMarched, 'right_knee', 'any', undefined, {
      exercise_id: 'ex-15',
      name_th: 'ยกเข่าสูงยืนทรงตัว (ขวา)',
    });
    expect(marchCheck.romDeg).not.toBeNull();
    expect(marchCheck.romDeg!).toBeGreaterThanOrEqual(45);
  });

  it('smooths landmarks via Fixed EMA and velocity-based Adaptive EMA and resets cleanly', () => {
    const fixedSmoother = new LandmarkSmoother({ alpha: 0.3, adaptive: false });
    const frame0 = createNeutralLandmarks({
      [POSE_INDEX.RIGHT_WRIST]: { x: 0.4, y: 0.5, z: 0, visibility: 0.95 },
    });
    const out0 = fixedSmoother.apply(frame0);
    expect(out0[POSE_INDEX.RIGHT_WRIST].y).toBeCloseTo(0.5, 5);

    const frame1 = createNeutralLandmarks({
      [POSE_INDEX.RIGHT_WRIST]: { x: 0.4, y: 0.6, z: 0, visibility: 0.95 },
    });
    const out1 = fixedSmoother.apply(frame1);
    // S_1 = 0.3 * 0.6 + 0.7 * 0.5 = 0.53
    expect(out1[POSE_INDEX.RIGHT_WRIST].y).toBeCloseTo(0.53, 5);

    const adaptiveSmoother = new LandmarkSmoother({
      alpha: 0.25,
      adaptive: true,
      minAlpha: 0.1,
      maxAlpha: 0.8,
      velocityScale: 15.0,
    });
    adaptiveSmoother.apply(frame0);
    // Small stationary jitter (delta = 0.001 -> alpha = 0.1 + 0.015 = 0.115)
    const jitterFrame = createNeutralLandmarks({
      [POSE_INDEX.RIGHT_WRIST]: { x: 0.4, y: 0.501, z: 0, visibility: 0.95 },
    });
    const jitterOut = adaptiveSmoother.apply(jitterFrame);
    expect(jitterOut[POSE_INDEX.RIGHT_WRIST].y).toBeCloseTo(0.5 + 0.001 * 0.115, 5);

    // Fast movement (delta ~ 0.10 -> velocity*scale = 1.5 -> clamped to maxAlpha = 0.8)
    const fastFrame = createNeutralLandmarks({
      [POSE_INDEX.RIGHT_WRIST]: { x: 0.4, y: 0.600115, z: 0, visibility: 0.95 },
    });
    const fastOut = adaptiveSmoother.apply(fastFrame);
    const prevY = jitterOut[POSE_INDEX.RIGHT_WRIST].y;
    expect(fastOut[POSE_INDEX.RIGHT_WRIST].y).toBeCloseTo(
      prevY + 0.8 * (0.600115 - prevY),
      5,
    );

    adaptiveSmoother.reset();
    const afterReset = createNeutralLandmarks({
      [POSE_INDEX.RIGHT_WRIST]: { x: 0.2, y: 0.25, z: 0, visibility: 0.95 },
    });
    expect(adaptiveSmoother.apply(afterReset)[POSE_INDEX.RIGHT_WRIST].y).toBeCloseTo(0.25, 5);
  });

  it('rejects out-of-frame or anatomically hallucinated knees even with worldLandmarks and flags bent-elbow or shoulder-shrug compensations', () => {
    const fakeWorldLeg = createNeutralLandmarks({
      [POSE_INDEX.RIGHT_HIP]: { x: 0.1, y: 0, z: 0, visibility: 0.95 },
      [POSE_INDEX.RIGHT_KNEE]: { x: 0.1, y: 0.4, z: 0, visibility: 0.95 },
      [POSE_INDEX.RIGHT_ANKLE]: { x: 0.1, y: 0.8, z: -0.05, visibility: 0.95 },
    });

    // 1a. Low visibility (< 0.55) leg landmarks + straight worldLandmarks must NOT produce ~170 deg
    const lowVisLeg = createNeutralLandmarks({
      [POSE_INDEX.RIGHT_HIP]: { x: 0.44, y: 0.65, z: 0, visibility: 0.35 },
      [POSE_INDEX.RIGHT_KNEE]: { x: 0.44, y: 0.8, z: 0, visibility: 0.35 },
      [POSE_INDEX.RIGHT_ANKLE]: { x: 0.44, y: 0.95, z: 0, visibility: 0.35 },
    });
    const lowVisResult = measureLiveJointAngleAndTilt(
      lowVisLeg,
      'right_knee',
      'any',
      fakeWorldLeg,
    );
    expect(lowVisResult.romDeg).toBeNull();
    expect(lowVisResult.lowVisibilityAlertTh).toContain('มองไม่เห็นสะโพก-เข่า-ข้อเท้าชัดเจน');

    // 1b. High visibility but anatomically invalid vertical order (hip <= maxShY + 0.10) must also return null
    const hallucinatedChestHip = createNeutralLandmarks({
      [POSE_INDEX.LEFT_SHOULDER]: { x: 0.62, y: 0.72, z: 0, visibility: 0.96 },
      [POSE_INDEX.RIGHT_SHOULDER]: { x: 0.38, y: 0.72, z: 0, visibility: 0.96 },
      [POSE_INDEX.RIGHT_HIP]: { x: 0.44, y: 0.78, z: 0, visibility: 0.85 },
      [POSE_INDEX.RIGHT_KNEE]: { x: 0.44, y: 0.86, z: 0, visibility: 0.85 },
      [POSE_INDEX.RIGHT_ANKLE]: { x: 0.44, y: 0.94, z: 0, visibility: 0.85 },
    });
    const chestHipResult = measureLiveJointAngleAndTilt(
      hallucinatedChestHip,
      'right_knee',
      'any',
      fakeWorldLeg,
    );
    expect(chestHipResult.romDeg).toBeNull();
    expect(chestHipResult.lowVisibilityAlertTh).toContain('มองไม่เห็นสะโพก-เข่า-ข้อเท้าชัดเจน');

    // 2a. Shoulder abduction with bent elbow (< 138 deg) triggers postureCompensated & elbow banner
    const bentElbow2d = createNeutralLandmarks({
      [POSE_INDEX.LEFT_SHOULDER]: { x: 0.58, y: 0.4, z: 0, visibility: 0.98 },
      [POSE_INDEX.RIGHT_SHOULDER]: { x: 0.42, y: 0.4, z: 0, visibility: 0.98 },
      [POSE_INDEX.RIGHT_ELBOW]: { x: 0.24, y: 0.4, z: 0, visibility: 0.98 },
      [POSE_INDEX.RIGHT_WRIST]: { x: 0.24, y: 0.22, z: 0, visibility: 0.98 },
    });
    const bentElbow3d = createNeutralLandmarks({
      [POSE_INDEX.RIGHT_SHOULDER]: { x: 0, y: 0, z: 0, visibility: 0.98 },
      [POSE_INDEX.RIGHT_ELBOW]: { x: -0.28, y: 0, z: 0, visibility: 0.98 },
      [POSE_INDEX.RIGHT_WRIST]: { x: -0.28, y: -0.25, z: 0, visibility: 0.98 },
    });
    const bentElbowRes = measureLiveJointAngleAndTilt(
      bentElbow2d,
      'right_shoulder',
      'abduction',
      bentElbow3d,
    );
    expect(bentElbowRes.postureCompensated).toBe(true);
    expect(bentElbowRes.elbowAngleDeg).toBeLessThan(138);
    expect(bentElbowRes.postureBannerTh).toContain('กรุณาเหยียดข้อศอกให้ตรง');

    // 2b. Shoulder abduction with shoulder shrug/tilt (> 13.5 deg) triggers shrug banner
    const shrug2d = createNeutralLandmarks({
      [POSE_INDEX.LEFT_SHOULDER]: { x: 0.58, y: 0.44, z: 0, visibility: 0.98 },
      [POSE_INDEX.RIGHT_SHOULDER]: { x: 0.42, y: 0.36, z: 0, visibility: 0.98 },
      [POSE_INDEX.RIGHT_ELBOW]: { x: 0.24, y: 0.36, z: 0, visibility: 0.98 },
      [POSE_INDEX.RIGHT_WRIST]: { x: 0.08, y: 0.36, z: 0, visibility: 0.98 },
    });
    const shrug3d = createNeutralLandmarks({
      [POSE_INDEX.RIGHT_SHOULDER]: { x: 0, y: 0, z: 0, visibility: 0.98 },
      [POSE_INDEX.RIGHT_ELBOW]: { x: -0.28, y: 0, z: 0, visibility: 0.98 },
      [POSE_INDEX.RIGHT_WRIST]: { x: -0.54, y: 0, z: 0, visibility: 0.98 },
    });
    const shrugRes = measureLiveJointAngleAndTilt(
      shrug2d,
      'right_shoulder',
      'abduction',
      shrug3d,
    );
    expect(shrugRes.postureCompensated).toBe(true);
    expect(shrugRes.shoulderTiltDeg).toBeGreaterThan(13.5);
    expect(shrugRes.postureBannerTh).toContain('ห้ามยกไหล่หรือเอียงไหล่ช่วย');
  });

  it('rescues close seated forward shoulder flexion when worldLandmarks compress depth (~34 deg) without false WRONG_PLANE', () => {
    const seatedPose2d = createNeutralLandmarks({
      [POSE_INDEX.LEFT_SHOULDER]: { x: 0.64, y: 0.78, z: 0, visibility: 0.98 },
      [POSE_INDEX.RIGHT_SHOULDER]: { x: 0.36, y: 0.78, z: 0, visibility: 0.98 },
      [POSE_INDEX.LEFT_HIP]: { x: 0.60, y: 1.15, z: 0, visibility: 0.05 },
      [POSE_INDEX.RIGHT_HIP]: { x: 0.40, y: 1.15, z: 0, visibility: 0.05 },
      [POSE_INDEX.RIGHT_ELBOW]: { x: 0.31, y: 0.78, z: -0.18, visibility: 0.98 },
      [POSE_INDEX.RIGHT_WRIST]: { x: 0.24, y: 0.77, z: -0.35, visibility: 0.98 },
    });
    const compressedWorld3d = createNeutralLandmarks({
      [POSE_INDEX.RIGHT_SHOULDER]: { x: 0, y: 0, z: 0, visibility: 0.98 },
      [POSE_INDEX.RIGHT_ELBOW]: { x: -0.05, y: 0.21, z: -0.13, visibility: 0.98 },
      [POSE_INDEX.RIGHT_WRIST]: { x: -0.10, y: 0.42, z: -0.26, visibility: 0.98 },
    });

    const result = measureLiveJointAngleAndTilt(
      seatedPose2d,
      'right_shoulder',
      'flexion',
      compressedWorld3d,
    );
    expect(result.wrongSideRaised).toBe(false);
    expect(result.wrongPlaneRaised).toBe(false);
    expect(result.wrongPlaneBannerTh).toBeNull();
    expect(result.postureCompensated).toBe(false);
    expect(result.romDeg).not.toBeNull();
    expect(result.romDeg!).toBeGreaterThanOrEqual(80);
    expect(result.romDeg!).toBeLessThanOrEqual(98);
  });

  it('builds deduplicated Edge Vision WebSocket endpoint candidates with loopback fallback', () => {
    const candidates = resolveKioskWsCandidates('ws://localhost:8000/ws/kiosk');
    expect(candidates).toContain('ws://localhost:8000/ws/kiosk');
    expect(candidates).toContain('ws://127.0.0.1:8000/ws/kiosk');
    expect(new Set(candidates).size).toBe(candidates.length);
  });

  it('draws a clean Stickman head circle while hiding the 5 facial dots and keeping hand/foot keypoints', () => {
    const arcCalls: { x: number; y: number; r: number }[] = [];
    const mockCtx = {
      save: () => {},
      restore: () => {},
      clearRect: () => {},
      beginPath: () => {},
      moveTo: () => {},
      lineTo: () => {},
      stroke: () => {},
      fill: () => {},
      strokeRect: () => {},
      fillRect: () => {},
      fillText: () => {},
      arc: (x: number, y: number, r: number) => {
        arcCalls.push({ x, y, r });
      },
      measureText: () => ({ width: 40 }),
      lineCap: 'round',
      lineWidth: 2,
      strokeStyle: '#10B981',
      fillStyle: '#FFFFFF',
      font: 'bold 12px sans-serif',
    } as unknown as CanvasRenderingContext2D;

    const kp: Keypoint3DPoint[] = Array.from({ length: 133 }, (_, index) => ({
      index,
      x: 0.5,
      y: 0.5,
      z: 0,
      score: 0.99,
    }));
    // 5 facial points (0..4)
    kp[0] = { index: 0, x: 0.5, y: 0.2, z: 0, score: 0.99 };
    kp[1] = { index: 1, x: 0.48, y: 0.17, z: 0, score: 0.99 };
    kp[2] = { index: 2, x: 0.52, y: 0.17, z: 0, score: 0.99 };
    kp[3] = { index: 3, x: 0.44, y: 0.19, z: 0, score: 0.99 };
    kp[4] = { index: 4, x: 0.56, y: 0.19, z: 0, score: 0.99 };
    // Shoulders & torso
    kp[5] = { index: 5, x: 0.4, y: 0.36, z: 0, score: 0.99 };
    kp[6] = { index: 6, x: 0.6, y: 0.36, z: 0, score: 0.99 };
    // Feet (17..22)
    kp[17] = { index: 17, x: 0.42, y: 0.92, z: 0, score: 0.99 };
    kp[20] = { index: 20, x: 0.58, y: 0.92, z: 0, score: 0.99 };
    // Left & right hand tips (e.g. 95, 116)
    kp[95] = { index: 95, x: 0.25, y: 0.45, z: 0, score: 0.99 };
    kp[116] = { index: 116, x: 0.75, y: 0.45, z: 0, score: 0.99 };

    drawWholeBody133OnCanvas(mockCtx, 800, 600, kp);

    // Head circle exists at nose anchor (400, 120) with large Stickman radius
    const headCircles = arcCalls.filter(
      (c) => Math.abs(c.x - 400) < 1 && Math.abs(c.y - 120) < 1 && c.r >= 16,
    );
    expect(headCircles).toHaveLength(1);

    // No small facial landmark dots drawn at eye/ear/nose coordinates
    const faceCoords = [
      { x: 400, y: 120 },
      { x: 384, y: 102 },
      { x: 416, y: 102 },
      { x: 352, y: 114 },
      { x: 448, y: 114 },
    ];
    for (const fc of faceCoords) {
      const smallDotAtFace = arcCalls.filter(
        (c) => Math.abs(c.x - fc.x) < 1 && Math.abs(c.y - fc.y) < 1 && c.r < 12,
      );
      expect(smallDotAtFace).toHaveLength(0);
    }

    // Foot landmarks & hand landmarks are still rendered
    expect(arcCalls.some((c) => Math.abs(c.x - 0.42 * 800) < 1 && Math.abs(c.y - 0.92 * 600) < 1)).toBe(true);
    expect(arcCalls.some((c) => Math.abs(c.x - 0.25 * 800) < 1 && Math.abs(c.y - 0.45 * 600) < 1)).toBe(true);
    expect(arcCalls.some((c) => Math.abs(c.x - 0.75 * 800) < 1 && Math.abs(c.y - 0.45 * 600) < 1)).toBe(true);
  });
});

describe('computeContainRect & drawMediaPipe33OnCanvas (1:1 Video Overlay)', () => {
  it('computes exact CSS object-contain pillarbox and letterbox rectangles', () => {
    const pillar = computeContainRect(1000, 500, 640, 480);
    expect(pillar.offsetY).toBe(0);
    expect(pillar.renderH).toBe(500);
    expect(pillar.renderW).toBeCloseTo(666.667, 2);
    expect(pillar.offsetX).toBeCloseTo(166.667, 2);

    const letter = computeContainRect(400, 600, 640, 480);
    expect(letter.offsetX).toBe(0);
    expect(letter.renderW).toBe(400);
    expect(letter.renderH).toBe(300);
    expect(letter.offsetY).toBe(150);

    const fallback = computeContainRect(800, 600);
    expect(fallback).toEqual({ offsetX: 0, offsetY: 0, renderW: 800, renderH: 600 });
  });

  it('draws 33-point MediaPipe skeleton aligned 1:1 with video aspect ratio and omits face dots', () => {
    const arcCalls: Array<{ x: number; y: number; r: number }> = [];
    const mockCtx = {
      clearRect: () => {},
      save: () => {},
      restore: () => {},
      setLineDash: () => {},
      beginPath: () => {},
      moveTo: () => {},
      lineTo: () => {},
      stroke: () => {},
      fill: () => {},
      strokeRect: () => {},
      fillRect: () => {},
      fillText: () => {},
      arc: (x: number, y: number, r: number) => {
        arcCalls.push({ x, y, r });
      },
      lineCap: 'round',
      lineJoin: 'round',
      lineWidth: 2,
      strokeStyle: '#10B981',
      fillStyle: '#6EE7B7',
      shadowBlur: 0,
      shadowColor: 'transparent',
    } as unknown as CanvasRenderingContext2D;

    const mp33: Keypoint3DPoint[] = Array.from({ length: 33 }, (_, index) => ({
      index,
      x: 0.5,
      y: 0.5,
      z: 0,
      score: 0.99,
    }));
    mp33[0] = { index: 0, x: 0.5, y: 0.2, z: 0, score: 0.99 };
    mp33[7] = { index: 7, x: 0.45, y: 0.2, z: 0, score: 0.99 };
    mp33[8] = { index: 8, x: 0.55, y: 0.2, z: 0, score: 0.99 };
    mp33[11] = { index: 11, x: 0.6, y: 0.35, z: 0, score: 0.99 };
    mp33[12] = { index: 12, x: 0.4, y: 0.35, z: 0, score: 0.99 };
    mp33[14] = { index: 14, x: 0.3, y: 0.45, z: 0, score: 0.99 };
    mp33[16] = { index: 16, x: 0.2, y: 0.5, z: 0, score: 0.99 };
    mp33[23] = { index: 23, x: 0.56, y: 0.65, z: 0, score: 0.99 };
    mp33[24] = { index: 24, x: 0.44, y: 0.65, z: 0, score: 0.99 };
    mp33[31] = { index: 31, x: 0.56, y: 0.92, z: 0, score: 0.99 };

    // Container 1000x500 with 640x480 video + mirrorX=true + isPrivacyMode=true:
    // offsetX = 166.67, renderW = 666.67 -> mirrored x=0.2 maps to 166.67 + 0.8 * 666.67 = 700
    drawMediaPipe33OnCanvas(mockCtx, 1000, 500, mp33, {
      mirrorX: true,
      videoWidth: 640,
      videoHeight: 480,
      highlightColor: 'GREEN',
      isPrivacyMode: true,
    });

    // Head circle rendered at mirrored nose (500, 100) with radius >= 14 in Privacy Mode
    expect(
      arcCalls.some((c) => Math.abs(c.x - 500) < 1 && Math.abs(c.y - 100) < 1 && c.r >= 14),
    ).toBe(true);
    // Small face dots at nose (500, 100) must NOT be drawn
    expect(
      arcCalls.filter((c) => Math.abs(c.x - 500) < 1 && Math.abs(c.y - 100) < 1 && c.r < 12),
    ).toHaveLength(0);
    // Mirrored right wrist (index 16 at x=0.2 -> 700, y=0.5 -> 250) aligned to contain rect
    expect(
      arcCalls.some((c) => Math.abs(c.x - 700) < 1 && Math.abs(c.y - 250) < 1),
    ).toBe(true);

    // Live real-camera mode (isPrivacyMode=false) omits the Stickman head circle over the real face
    arcCalls.length = 0;
    drawMediaPipe33OnCanvas(mockCtx, 1000, 500, mp33, {
      mirrorX: true,
      videoWidth: 640,
      videoHeight: 480,
      highlightColor: 'GREEN',
      isPrivacyMode: false,
    });
    expect(
      arcCalls.filter((c) => Math.abs(c.x - 500) < 1 && Math.abs(c.y - 100) < 1 && c.r >= 14),
    ).toHaveLength(0);
    expect(
      arcCalls.some((c) => Math.abs(c.x - 700) < 1 && Math.abs(c.y - 250) < 1),
    ).toBe(true);

    // Toggling showSkeleton=false clears and draws nothing
    arcCalls.length = 0;
    drawMediaPipe33OnCanvas(mockCtx, 1000, 500, mp33, {
      mirrorX: true,
      videoWidth: 640,
      videoHeight: 480,
      showSkeleton: false,
    });
    expect(arcCalls).toHaveLength(0);
  });

  it('validates upper-body framing and suppresses close-up partial skeletons', () => {
    const arcCalls: Array<{ x: number; y: number; r: number }> = [];
    const mockCtx = {
      clearRect: () => {},
      save: () => {},
      restore: () => {},
      setLineDash: () => {},
      beginPath: () => {},
      moveTo: () => {},
      lineTo: () => {},
      stroke: () => {},
      fill: () => {},
      strokeRect: () => {},
      fillRect: () => {},
      fillText: () => {},
      arc: (x: number, y: number, r: number) => {
        arcCalls.push({ x, y, r });
      },
      lineCap: 'round',
      lineJoin: 'round',
      lineWidth: 2,
      strokeStyle: '#10B981',
      fillStyle: '#6EE7B7',
      shadowBlur: 0,
      shadowColor: 'transparent',
    } as unknown as CanvasRenderingContext2D;

    // Close-up headshot: shoulders at very bottom edge (y=0.88), no visible elbows or hips
    const closeUpMp33: Keypoint3DPoint[] = Array.from({ length: 33 }, (_, index) => ({
      index,
      x: 0.5,
      y: 0.99,
      z: 0,
      score: 0.1,
    }));
    closeUpMp33[0] = { index: 0, x: 0.5, y: 0.55, z: 0, score: 0.99 };
    closeUpMp33[11] = { index: 11, x: 0.72, y: 0.88, z: 0, score: 0.9 };
    closeUpMp33[12] = { index: 12, x: 0.28, y: 0.88, z: 0, score: 0.9 };

    expect(isUpperBodyVisibleInFrame(closeUpMp33)).toBe(false);
    drawMediaPipe33OnCanvas(mockCtx, 800, 600, closeUpMp33, {
      mirrorX: true,
      isPrivacyMode: false,
    });
    expect(arcCalls).toHaveLength(0);

    // Properly framed upper body returns true
    const framedMp33: Keypoint3DPoint[] = Array.from({ length: 33 }, (_, index) => ({
      index,
      x: 0.5,
      y: 0.5,
      z: 0,
      score: 0.95,
    }));
    framedMp33[11] = { index: 11, x: 0.6, y: 0.42, z: 0, score: 0.98 };
    framedMp33[12] = { index: 12, x: 0.4, y: 0.42, z: 0, score: 0.98 };
    framedMp33[13] = { index: 13, x: 0.64, y: 0.62, z: 0, score: 0.95 };
    framedMp33[23] = { index: 23, x: 0.57, y: 0.78, z: 0, score: 0.95 };
    framedMp33[24] = { index: 24, x: 0.43, y: 0.78, z: 0, score: 0.95 };
    expect(isUpperBodyVisibleInFrame(framedMp33)).toBe(true);
  });
});

