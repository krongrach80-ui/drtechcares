import { describe, expect, it } from 'vitest';
import { POSE_INDEX } from '@drtechcares/pose-engine';
import type { NormalizedLandmark } from '@mediapipe/tasks-vision';
import {
  EXERCISE_OPTIONS,
  getPrescribedLimbPromptTh,
  mapTodayExercisesToPrescriptions,
  synthesizeDemoPayload,
} from '../hooks/useKioskSocket';
import { measureLiveJointAngleAndTilt, stepSmoothRomAngle } from '../components/SmartRehabKioskUI';
import {
  VOICE_COACH_LANGUAGES,
  buildTtsProxyUrl,
  pickSpeechVoice,
  translateVoiceCoachText,
} from '../components/VoiceCoachBanner';

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

  it('smoothly ramps ROM angles with deadband and human-paced slew rate instead of jumping rapidly', () => {
    // 1. Ignores micro-jitter within < 1.2 deg deadband
    expect(stepSmoothRomAngle(45, 45.9, 16)).toBe(45);
    expect(stepSmoothRomAngle(45, 44.0, 16)).toBe(45);

    // 2. Caps single-frame jump (15 -> 150 deg) to <= 36 deg/sec
    const singleStep = stepSmoothRomAngle(15, 150, 16);
    expect(singleStep).toBeGreaterThan(15);
    expect(singleStep).toBeLessThanOrEqual(15 + (36 * 16) / 1000 + 1e-6);

    // 3. Climbs gradually over 1 second rather than jumping straight to 150 deg
    let rom = 15;
    for (let i = 0; i < 60; i += 1) {
      rom = stepSmoothRomAngle(rom, 150, 1000 / 60);
    }
    expect(rom).toBeGreaterThan(45);
    expect(rom).toBeLessThanOrEqual(52);

    // 4. Converges smoothly within deadband over ~5 seconds
    for (let i = 0; i < 240; i += 1) {
      rom = stepSmoothRomAngle(rom, 150, 1000 / 60);
    }
    expect(rom).toBeGreaterThanOrEqual(148.5);
    expect(rom).toBeLessThanOrEqual(150);

    // 5. Caps downward return speed to <= 48 deg/sec
    const downStep = stepSmoothRomAngle(150, 15, 20);
    expect(downStep).toBeLessThan(150);
    expect(downStep).toBeGreaterThanOrEqual(150 - (48 * 20) / 1000 - 1e-6);
  });
});

