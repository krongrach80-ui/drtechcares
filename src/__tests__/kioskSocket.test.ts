import { describe, expect, it } from 'vitest';
import { POSE_INDEX } from '@drtechcares/pose-engine';
import type { NormalizedLandmark } from '@mediapipe/tasks-vision';
import {
  EXERCISE_OPTIONS,
  getPrescribedLimbPromptTh,
  mapTodayExercisesToPrescriptions,
  resolveExpectedPlaneForExercise,
  synthesizeDemoPayload,
} from '../hooks/useKioskSocket';
import {
  DEFAULT_FORM_ASSESSMENT_RULES,
  isFormRuleActive,
  measureLiveJointAngleAndTilt,
  stepSmoothRomAngle,
} from '../components/SmartRehabKioskUI';
import {
  VOICE_COACH_LANGUAGES,
  buildTtsProxyUrl,
  decideVoiceCoachSpeechTurn,
  pickSpeechVoice,
  translateVoiceCoachText,
} from '../components/VoiceCoachBanner';
import { getKioskVerifiedModalCopy } from '../components/KioskBiometricLoginView';
import { getKioskOverviewVoiceNarration } from '../App';

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
    // 1. Ignores micro-jitter within < 0.8 deg deadband
    expect(stepSmoothRomAngle(45, 45.6, 16)).toBe(45);
    expect(stepSmoothRomAngle(45, 44.3, 16)).toBe(45);

    // 2. Bounds single-frame spikes (15 -> 150 deg in 16ms) to <= 260 deg/sec
    const singleStep = stepSmoothRomAngle(15, 150, 16);
    expect(singleStep).toBeGreaterThan(15);
    expect(singleStep).toBeLessThanOrEqual(15 + (260 * 16) / 1000 + 1e-6);

    // 3. Converges to >= 148 deg within ~0.65 seconds (39 frames at 60Hz) matching real human arm speed
    let rom = 15;
    for (let i = 0; i < 39; i += 1) {
      rom = stepSmoothRomAngle(rom, 150, 1000 / 60);
    }
    expect(rom).toBeGreaterThanOrEqual(148);
    expect(rom).toBeLessThanOrEqual(150);

    // 4. Caps downward return speed to <= 280 deg/sec
    const downStep = stepSmoothRomAngle(150, 15, 20);
    expect(downStep).toBeLessThan(150);
    expect(downStep).toBeGreaterThanOrEqual(150 - (280 * 20) / 1000 - 1e-6);

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
    expect(th.startBtnIdle).toContain('ยก 2 มือขึ้น');
    expect(th.notMeBtn).toContain('ไม่ใช่ฉัน');
    expect(th.voiceGreeting).toContain('Pdasdwa');

    const en = getKioskVerifiedModalCopy('en', sampleParams);
    expect(en.confidenceBadge).toContain('Biometric Confidence 77%');
    expect(en.hnAndNextAppointment).toContain('Next Appointment: 2026-11-20');
    expect(en.startBtnIdle).toContain('Raise Both Hands');
    expect(en.gestureDetectedPill).toContain('50%');
    expect(en.notMeBtn).toContain('Not me');
    expect(en.voiceGreeting).toContain('Welcome, Pdasdwa');

    const cn = getKioskVerifiedModalCopy('cn', sampleParams);
    expect(cn.confidenceBadge).toContain('生物识别准确度 77%');
    expect(cn.hnAndNextAppointment).toContain('下次预约: 2026-11-20');
    expect(cn.startBtnIdle).toContain('举起双手');
    expect(cn.gestureDetectedPill).toContain('50%');
    expect(cn.notMeBtn).toContain('不是我本人');
    expect(cn.voiceGreeting).toContain('Pdasdwa');
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
});

