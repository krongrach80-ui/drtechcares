import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import { Languages, RotateCcw, Volume2, VolumeX } from 'lucide-react';
import {
  KIOSK_VOICE_LANG_EVENT,
  KIOSK_VOICE_LANG_STORAGE_KEY,
  KIOSK_VOICE_MUTED_EVENT,
  KIOSK_VOICE_MUTED_STORAGE_KEY,
  cancelAllKioskVoiceSpeech,
  getSeniorVoiceToggleLabel,
  readStoredKioskVoiceLang,
  readStoredKioskVoiceMuted,
  sanitizeSpeechText,
  subscribeKioskVoiceLang,
  subscribeKioskVoiceMuted,
  writeStoredKioskVoiceLang,
  writeStoredKioskVoiceMuted,
} from '@drtechcares/kiosk-input';
import type { HighlightColor, KioskState } from '../hooks/useKioskSocket';

export {
  KIOSK_VOICE_LANG_EVENT,
  KIOSK_VOICE_LANG_STORAGE_KEY,
  KIOSK_VOICE_MUTED_EVENT,
  KIOSK_VOICE_MUTED_STORAGE_KEY,
  cancelAllKioskVoiceSpeech,
  getSeniorVoiceToggleLabel,
  readStoredKioskVoiceLang,
  readStoredKioskVoiceMuted,
  sanitizeSpeechText,
  subscribeKioskVoiceLang,
  subscribeKioskVoiceMuted,
  writeStoredKioskVoiceLang,
  writeStoredKioskVoiceMuted,
};

export type VoiceCoachLanguage = 'th' | 'en' | 'zh';

export interface VoiceCoachLanguageOption {
  id: VoiceCoachLanguage;
  numberLabel: string;
  shortLabel: string;
  badgeTitle: string;
  bcp47: string;
  statusReady: string;
  statusSpeaking: string;
  statusMuted: string;
  replayTitle: string;
  muteLabel: string;
  unmuteLabel: string;
  defaultPrompt: string;
}

export const VOICE_COACH_LANGUAGES: readonly VoiceCoachLanguageOption[] = [
  {
    id: 'th',
    numberLabel: '1. ไทย',
    shortLabel: 'TH ไทย',
    badgeTitle: 'TH AI VOICE COACH',
    bcp47: 'th-TH',
    statusReady: 'พร้อมแนะนำ',
    statusSpeaking: 'กำลังพูด',
    statusMuted: 'ปิดเสียง',
    replayTitle: 'พูดซ้ำข้อความนี้',
    muteLabel: 'เสียงปกติ',
    unmuteLabel: 'เปิดเสียง AI',
    defaultPrompt: 'พร้อมเริ่มฝึกกายภาพบำบัดครับ',
  },
  {
    id: 'en',
    numberLabel: '2. อังกฤษ (EN)',
    shortLabel: 'EN',
    badgeTitle: 'EN AI VOICE COACH',
    bcp47: 'en-US',
    statusReady: 'READY',
    statusSpeaking: 'SPEAKING',
    statusMuted: 'MUTED',
    replayTitle: 'Replay voice prompt',
    muteLabel: 'Voice On',
    unmuteLabel: 'Unmute AI',
    defaultPrompt: 'Ready to begin your physical therapy session.',
  },
  {
    id: 'zh',
    numberLabel: '3. จีน (中文)',
    shortLabel: '中文',
    badgeTitle: 'CN AI VOICE COACH',
    bcp47: 'zh-CN',
    statusReady: '准备就绪',
    statusSpeaking: '正在播报',
    statusMuted: '已静音',
    replayTitle: '重播语音提示',
    muteLabel: '语音正常',
    unmuteLabel: '开启语音',
    defaultPrompt: '已准备好开始物理治疗训练。',
  },
];

const VOICE_LANG_STORAGE_KEY = 'drtechcares_voice_coach_lang_v1';

const EXACT_TRANSLATIONS: Record<string, { en: string; zh: string }> = {
  'พร้อมเริ่มฝึกกายภาพบำบัดครับ': {
    en: 'Ready to begin your physical therapy session.',
    zh: '已准备好开始物理治疗训练。',
  },
  'ตรวจพบมีบุคคลอยู่หลายคนในเฟรม กรุณาให้คนไข้ยืนหน้ากล้องเพียงลำพัง ระบบจึงจะนับรอบการฝึกต่อครับ': {
    en: 'Multiple people detected in frame. Please stand alone in front of the camera to continue counting reps.',
    zh: '画面中检测到多人，请让患者单独站在镜头前，系统才会继续计数。',
  },
  'คุณไม่ใช่คนไข้ที่ลงทะเบียนไว้ ระบบจะไม่นับรอบการฝึก กรุณาให้คนไข้ตัวจริงเข้าฝึกครับ': {
    en: 'Identity mismatch detected. Rep counting is paused. Please ask the registered patient to step in front of the camera.',
    zh: '检测到非本人患者，系统已暂停计数，请由本人患者站在镜头前训练。',
  },
  'ยืนยันตัวตนคนไข้ถูกต้องแล้ว สามารถฝึกต่อได้ครับ': {
    en: 'Patient identity verified. You may continue your exercise.',
    zh: '患者身份验证成功，可以继续训练。',
  },
  'หยุดพักทันทีนะครับ ระบบกำลังเรียกเจ้าหน้าที่ดูแล': {
    en: 'Please stop and rest immediately. The system is calling clinical staff to assist you.',
    zh: '请立即停止并休息，系统正在呼叫医护人员前来协助。',
  },
  'หยุดพักก่อนนะครับ เจ้าหน้าที่กำลังเข้าไปดูแลครับ': {
    en: 'Please pause and rest for a moment. Clinical staff are coming to assist you.',
    zh: '请先停下来休息，医护人员正在赶来协助您。',
  },
  'หยุดพักทันทีนะครับ รู้สึกเจ็บตรงไหนไหมครับ': {
    en: 'Please stop and rest immediately. Do you feel any pain or discomfort?',
    zh: '请立即停止并休息，您感觉哪里疼痛或不舒服吗？',
  },
  'กรุณามองตรงเข้ากรอบวงกลม เพื่อยืนยันตัวตนอัตโนมัติครับ': {
    en: 'Please look straight into the circle frame for automatic face verification.',
    zh: '请直视圆框内摄像头，以进行自动人脸身份验证。',
  },
  'กรุณาสแกนใบหน้ายืนยันตัวตนก่อนเริ่มฝึกครับ': {
    en: 'Please scan your face to verify your identity before starting your exercise.',
    zh: '开始训练前，请先进行人脸扫描验证身份。',
  },
  'กรุณามองกล้องเพื่อยืนยันตัวตนด้วยใบหน้า': {
    en: 'Please look at the camera for face verification.',
    zh: '请看向摄像头进行人脸身份验证。',
  },
  'ยืนยันตัวตนสำเร็จ กรุณาถอยหลังเล็กน้อยให้อยู่ระยะ 2 เมตรครับ': {
    en: 'Identity verified. Please step back slightly to stand at a 2-meter distance.',
    zh: '身份验证成功，请稍微向后退一步，保持在两米距离。',
  },
  'ยืนยันตัวตนสำเร็จ กรุณาเดินเข้ามาอีกนิดให้อยู่ระยะ 2 เมตรครับ': {
    en: 'Identity verified. Please step forward slightly to stand at a 2-meter distance.',
    zh: '身份验证成功，请向前走近一点，保持在两米距离。',
  },
  'ยืนยันตัวตนสำเร็จ ระยะยืนเหมาะสม พร้อมเริ่มฝึกได้เลยครับ': {
    en: 'Identity verified and distance is optimal. You are ready to start.',
    zh: '身份验证成功，站立距离合适，现在可以开始训练。',
  },
  'ถอยหลังอีกนิด ให้อยู่ระยะ 2 เมตรครับ': {
    en: 'Please step back slightly to stay at 2 meters.',
    zh: '请向后退一点，保持在两米距离。',
  },
  'เดินเข้ามาอีกนิด ให้อยู่ระยะ 2 เมตรครับ': {
    en: 'Please step forward slightly to stay at 2 meters.',
    zh: '请向前走近一点，保持在两米距离。',
  },
  'ขยับให้เห็นทั้งตัวและเท้าในกรอบครับ': {
    en: 'Please adjust your position so your full body and feet are visible in frame.',
    zh: '请调整站位，让全身和双脚都清晰显示在画面框内。',
  },
  'เก่งมากครับ ฝึกครบเซ็ตเรียบร้อยแล้ว': {
    en: 'Excellent work! You have completed the entire exercise set.',
    zh: '非常棒！您已顺利完成整组训练。',
  },
  'กรุณายกแขนซ้ายนะครับ ท่านี้ต้องทำด้วยแขนซ้าย ระบบจะไม่นับรอบหากยกแขนขวาครับ': {
    en: 'Please raise your left arm. This exercise must be performed with your left arm; raising your right arm will not be counted.',
    zh: '请抬起左臂。此动作必须使用左臂，抬起右臂将不计入次数。',
  },
  'ท่านี้ฝึกเฉพาะข้างซ้าย กรุณาลดแขนขวาลง และยกเฉพาะแขนซ้ายข้างเดียวนะครับ': {
    en: 'Please raise your left arm only and lower your right arm.',
    zh: '请仅使用左臂训练，请放下右臂并只抬起左臂。',
  },
  'กรุณายกแขนขวานะครับ ท่านี้ต้องทำด้วยแขนขวา ระบบจะไม่นับรอบหากยกแขนซ้ายครับ': {
    en: 'Please raise your right arm. This exercise must be performed with your right arm; raising your left arm will not be counted.',
    zh: '请抬起右臂。此动作必须使用右臂，抬起左臂将不计入次数。',
  },
  'ท่านี้ฝึกเฉพาะข้างขวา กรุณาลดแขนซ้ายลง และยกเฉพาะแขนขวาข้างเดียวนะครับ': {
    en: 'Please raise your right arm only and lower your left arm.',
    zh: '请仅使用右臂训练，请放下左臂并只抬起右臂。',
  },
  'ท่านี้ต้องทำพร้อมกันทั้ง 2 ข้าง กรุณายกแขนทั้งซ้ายและขวาขึ้นพร้อมกันนะครับ': {
    en: 'This exercise requires both arms together. Please lift your left and right arms together.',
    zh: '此动作需要双臂同时完成，请同时抬起左臂和右臂。',
  },
  'กรุณายกแขนตรงไปด้านหน้านะครับ ท่านี้ห้ามกางแขนออกด้านข้าง ระบบจะไม่นับรอบครับ': {
    en: 'Please raise your arm straight forward. Do not raise it out to the side; side raises will not be counted.',
    zh: '请将手臂向正前方抬起，此动作不要向侧面展开，向侧面抬起将不计入次数。',
  },
  'กรุณากางแขนออกด้านข้างนะครับ ท่านี้ห้ามยกแขนไปด้านหน้า ระบบจะไม่นับรอบครับ': {
    en: 'Please raise your arm out to the side. Do not raise it forward; forward raises will not be counted.',
    zh: '请将手臂向身体侧面外展抬起，不要向前方抬起，向前抬起将不计入次数。',
  },
  'กรุณาทำท่าด้วยขาซ้ายตามที่กำหนดนะครับ': {
    en: 'Please perform this exercise using your left leg as prescribed.',
    zh: '请按照规定使用左腿完成此动作。',
  },
  'กรุณาทำท่าตามข้างที่กำหนดนะครับ': {
    en: 'Please perform this exercise using the prescribed side.',
    zh: '请按照处方规定的一侧肢体完成此动作。',
  },
  'ยืดตัวตรงครับ อย่าเอียงลำตัวช่วยยกนะครับ': {
    en: 'Keep your back straight. Please do not lean your torso to assist the lift.',
    zh: '请挺直背部，不要倾斜身体借力抬起。',
  },
  'ยืดตัวตรงครับ อย่าเอียงตัวช่วยยก': {
    en: 'Keep your back straight and avoid leaning to the side.',
    zh: '请挺直身体，不要向一侧倾斜借力。',
  },
  'ผ่อนไหล่ลงครับ อย่ายักไหล่ขึ้น': {
    en: 'Relax your shoulders down and avoid shrugging.',
    zh: '请放松肩膀向下沉，不要耸肩。',
  },
  'ระวังเข่าบิดเข้าด้านในครับ': {
    en: 'Keep your knees aligned and avoid letting them cave inward.',
    zh: '请注意膝盖保持对齐，不要向内扣。',
  },
  'ค่อยๆ เคลื่อนไหวช้าๆ อย่างควบคุมครับ': {
    en: 'Please move slowly and keep the motion under control.',
    zh: '请缓慢且平稳地控制动作。',
  },
  'กดส้นเท้าติดพื้นไว้ครับ': {
    en: 'Keep your heels pressed flat on the floor.',
    zh: '请将脚后跟稳稳踩在地面上。',
  },
  'รักษาระดับสะโพกให้ตรงครับ': {
    en: 'Keep your hips level and steady.',
    zh: '请保持髋部水平稳定。',
  },
  'ยกให้สุดมุมแล้วค้างไว้สักครู่นะครับ': {
    en: 'Lift all the way to the target angle and hold briefly.',
    zh: '请抬至目标角度并保持片刻。',
  },
  'แขน/ขาตกลงเล็กน้อย กรุณายกกลับเข้าสู่เป้าหมาย': {
    en: 'Your limb lowered slightly. Please raise it back up to the target zone.',
    zh: '手臂或腿部略微下降，请重新抬回目标位置。',
  },
  'ค่อยๆ ยกขึ้นจนถึงเส้นเป้าหมายครับ': {
    en: 'Slowly lift up until you reach the target line.',
    zh: '请慢慢抬起直至达到目标线。',
  },
  'ปรับท่าทางให้ถูกต้อง': {
    en: 'Please adjust your posture into proper alignment.',
    zh: '请将姿势调整回正确位置。',
  },
  'รักษาท่าทางให้ตรงระหว่างค้างท่า': {
    en: 'Maintain straight posture while holding the position.',
    zh: '在保持动作期间，请维持身体挺直。',
  },
};

function translateExerciseNameIfKnown(
  nameTh: string,
  lang: Exclude<VoiceCoachLanguage, 'th'>,
): string {
  const map: Record<string, { en: string; zh: string }> = {
    'กางแขนขวาด้านข้าง': { en: 'Right Shoulder Abduction', zh: '右肩外展动作' },
    'กางแขนซ้ายด้านข้าง': { en: 'Left Shoulder Abduction', zh: '左肩外展动作' },
    'ยกแขนตรงไปด้านหน้า (ขวา)': { en: 'Right Shoulder Flexion', zh: '右肩前屈上举动作' },
    'ยกแขนตรงไปด้านหน้า (ซ้าย)': { en: 'Left Shoulder Flexion', zh: '左肩前屈上举动作' },
    'ลุกนั่งฝึกกล้ามเนื้อขา': { en: 'Chair Squat Exercise', zh: '坐站腿部康复动作' },
  };
  const trimmed = nameTh.trim();
  const hit = map[trimmed];
  if (hit) return hit[lang];
  return lang === 'en' ? 'the prescribed exercise' : '康复训练动作';
}

export function translateVoiceCoachText(
  textTh: string,
  lang: VoiceCoachLanguage,
): string {
  const clean = textTh.trim();
  const langOption =
    VOICE_COACH_LANGUAGES.find((item) => item.id === lang) ?? VOICE_COACH_LANGUAGES[0];

  if (!clean) {
    return langOption.defaultPrompt;
  }
  if (lang === 'th') {
    return clean;
  }

  const exact = EXACT_TRANSLATIONS[clean];
  if (exact) {
    return exact[lang];
  }

  // Dynamic pattern 1: Target ROM instruction
  // e.g. "ค่อยๆ เคลื่อนไหวไปยังเป้าหมาย 145 องศาครับ"
  const romTargetMatch = clean.match(/^ค่อยๆ\s*เคลื่อนไหวไปยังเป้าหมาย\s+(\d+(?:\.\d+)?)\s+องศาครับ$/);
  if (romTargetMatch) {
    const deg = romTargetMatch[1];
    return lang === 'en'
      ? `Slowly move toward the target angle of ${deg} degrees.`
      : `请慢慢移动至目标角度 ${deg} 度。`;
  }

  // Dynamic pattern 2: Hold phase countdown
  // e.g. "ดีมากครับ ค้างท่าไว้ 3 วินาที" | "ดีมากครับ! ค้างท่าไว้ 3 วินาที" | "ค้างท่าไว้ 2.1 วินาที"
  const holdMatch = clean.match(
    /^(?:ดีมากครับ!?\s*)?ค้าง(?:ท่า)?ไว้\s+([\d.]+)\s+วินาที$/,
  );
  if (holdMatch) {
    const sec = holdMatch[1];
    const hasPraise = clean.startsWith('ดีมากครับ');
    if (lang === 'en') {
      const unit = Number(sec) === 1 ? 'second' : 'seconds';
      return hasPraise
        ? `Very good! Hold this position for ${sec} ${unit}.`
        : `Hold this position for ${sec} ${unit}.`;
    }
    return hasPraise ? `很好！请保持这个姿势 ${sec} 秒。` : `请保持这个姿势 ${sec} 秒。`;
  }

  // Dynamic pattern 3: Set finished with patient name & target reps
  // e.g. "เยี่ยมมากครับ คุณสมชาย ใจดี ฝึกครบ 10 ครั้ง จบเซ็ตแล้วครับ"
  // e.g. "ครบ 3 วินาทีแล้ว เยี่ยมมากครับ คุณสมชาย ใจดี ฝึกครบ 10 ครั้ง จบเซ็ตแล้ว ค่อยๆ เอาแขนลงช้าๆ ครับ"
  const setFinishedNamedMatch = clean.match(
    /^(?:ครบ\s+([\d.]+)\s+วินาทีแล้ว\s+)?เยี่ยมมากครับ\s+(.+?)\s+ฝึกครบ\s+(\d+)\s+ครั้ง\s+จบเซ็ตแล้ว(?:ครับ|\s+ค่อยๆ\s*(?:ผ่อน|เอา)(แขน|ขา)ลงช้าๆ\s*ครับ)?$/,
  );
  if (setFinishedNamedMatch) {
    const [, holdSec, patientName, reps, limb] = setFinishedNamedMatch;
    if (lang === 'en') {
      const holdPrefix = holdSec ? `${holdSec} seconds completed! ` : '';
      const limbSuffix = limb
        ? ` Slowly lower your ${limb === 'ขา' ? 'leg' : 'arm'} back down.`
        : '';
      return `${holdPrefix}Great job, ${patientName}! You completed ${reps} reps and finished the set.${limbSuffix}`;
    }
    const holdPrefixZh = holdSec ? `已保持 ${holdSec} 秒！` : '';
    const limbSuffixZh = limb
      ? `请慢慢把${limb === 'ขา' ? '腿' : '手臂'}放下来。`
      : '';
    return `${holdPrefixZh}太棒了，${patientName}！已完成全部 ${reps} 次，本组训练结束。${limbSuffixZh}`;
  }

  // Dynamic pattern 4: Set finished without patient name
  // e.g. "เยี่ยมมากครับ ครบ 10 ครั้ง จบเซ็ตแล้ว" | "ครบ 3 วินาทีแล้ว เยี่ยมมากครับ ครบ 10 ครั้ง จบเซ็ตแล้ว ค่อยๆ เอาแขนลงช้าๆ ครับ"
  const setFinishedShortMatch = clean.match(
    /^(?:ครบ\s+([\d.]+)\s+วินาทีแล้ว\s+)?(?:เยี่ยมมากครับ|เก่งมากครับ)!?\s*(?:ฝึก)?ครบ\s+(\d+)\s+ครั้ง(?:แล้ว)?(?:\s+(?:จบเซ็ตแล้ว|ค่อยๆ\s*วางลง)(?:ครับ)?)?(?:\s+ค่อยๆ\s*(?:ผ่อน|เอา)(แขน|ขา)ลงช้าๆ\s*ครับ)?$/,
  );
  if (setFinishedShortMatch) {
    const [, holdSec, reps, limb] = setFinishedShortMatch;
    if (lang === 'en') {
      const holdPrefix = holdSec ? `${holdSec} seconds completed! ` : '';
      const limbSuffix = limb
        ? ` Slowly lower your ${limb === 'ขา' ? 'leg' : 'arm'} back down.`
        : '';
      return `${holdPrefix}Great job! All ${reps} reps are complete. You finished the set.${limbSuffix}`;
    }
    const holdPrefixZh = holdSec ? `已保持 ${holdSec} 秒！` : '';
    const limbSuffixZh = limb
      ? `请慢慢把${limb === 'ขา' ? '腿' : '手臂'}放下来。`
      : '';
    return `${holdPrefixZh}太棒了！已完成全部 ${reps} 次，本组训练结束。${limbSuffixZh}`;
  }

  // Dynamic pattern 5: Rep completed
  // e.g. "ครบ 3 วินาทีแล้ว ดีมากครับ ครบ 4 ครั้ง ค่อยๆ เอาแขนลงช้าๆ ครับ" | "ดีมากครับ ครบ 4 ครั้ง ค่อยๆ ผ่อนแขนลงช้าๆ ครับ"
  const repCompletedMatch = clean.match(
    /^(?:ครบ\s+([\d.]+)\s+วินาทีแล้ว\s+)?ดีมากครับ\s+ครบ\s+(\d+)\s+ครั้ง\s+ค่อยๆ\s*(?:ผ่อน|เอา)(แขน|ขา)ลงช้าๆ\s*ครับ$/,
  );
  if (repCompletedMatch) {
    const [, holdSec, reps, limb] = repCompletedMatch;
    const isLeg = limb === 'ขา';
    if (lang === 'en') {
      const holdPrefix = holdSec ? `${holdSec} seconds completed! ` : '';
      return `${holdPrefix}Very good! ${reps} reps completed. Slowly lower your ${isLeg ? 'leg' : 'arm'} back down.`;
    }
    const holdPrefixZh = holdSec ? `已保持 ${holdSec} 秒！` : '';
    return `${holdPrefixZh}很好！已完成 ${reps} 次，请慢慢将${isLeg ? '腿' : '手臂'}放下来。`;
  }

  // Dynamic pattern 6: Edge Vision fractional rep progress
  // e.g. "ครบ 3 วินาทีแล้ว เยี่ยมมากครับ ครบ 4/10 ครั้ง ค่อยๆ เอาแขนลงช้าๆ ครับ"
  const repFractionMatch = clean.match(
    /^(?:ครบ\s+([\d.]+)\s+วินาทีแล้ว\s+)?เยี่ยมมากครับ\s+ครบ\s+(\d+)\s*\/\s*(\d+)\s+ครั้ง(?:\s+(.+))?$/,
  );
  if (repFractionMatch) {
    const [, holdSec, done, total, trailing] = repFractionMatch;
    const isArm = trailing?.includes('แขน');
    const isLeg = trailing?.includes('ขา');
    if (lang === 'en') {
      const holdPrefix = holdSec ? `${holdSec} seconds completed! ` : '';
      const lowerText = isLeg
        ? 'Slowly lower your leg back down.'
        : isArm
          ? 'Slowly lower your arm back down.'
          : 'Slowly return to resting position.';
      return `${holdPrefix}Great! Completed ${done} of ${total} reps. ${lowerText}`;
    }
    const holdPrefixZh = holdSec ? `已保持 ${holdSec} 秒！` : '';
    const lowerTextZh = isLeg
      ? '请慢慢把腿放下来。'
      : isArm
        ? '请慢慢把手臂放下来。'
        : '请慢慢回到起始姿势。';
    return `${holdPrefixZh}太棒了！已完成第 ${done} 次，共 ${total} 次，${lowerTextZh}`;
  }

  // Dynamic pattern 7: "ครบ 3 วินาทีแล้ว ค่อยๆ เอาแขนลงช้าๆ ครบ 3 ครั้งแล้วครับ" / "ผ่อนลงช้าๆ ครบ 3 ครั้งแล้วครับ" / "ดีมากครับ ค้างไว้ครบ 3 ครั้งครับ"
  const lowerSlowMatch = clean.match(
    /^(?:ครบ\s+([\d.]+)\s+วินาทีแล้ว\s+)?(?:ค่อยๆ\s*เอา(แขน|ขา)ลงช้าๆ|ผ่อนลงช้าๆ)\s+ครบ\s+(\d+)\s+ครั้งแล้วครับ$/,
  );
  if (lowerSlowMatch) {
    const [, holdSec, limb, reps] = lowerSlowMatch;
    const limbEn = limb === 'ขา' ? 'your leg ' : limb === 'แขน' ? 'your arm ' : '';
    const limbZh = limb === 'ขา' ? '把腿' : limb === 'แขน' ? '把手臂' : '';
    if (lang === 'en') {
      const holdPrefix = holdSec ? `${holdSec} seconds completed! ` : '';
      return `${holdPrefix}Lower ${limbEn}slowly. You have completed ${reps} reps.`;
    }
    const holdPrefixZh = holdSec ? `已保持 ${holdSec} 秒！` : '';
    return `${holdPrefixZh}请慢慢${limbZh}放下，您已完成 ${reps} 次。`;
  }

  const holdCompletedMatch = clean.match(/^ดีมากครับ\s+ค้างไว้ครบ\s+(\d+)\s+ครั้งครับ$/);
  if (holdCompletedMatch) {
    const reps = holdCompletedMatch[1];
    return lang === 'en'
      ? `Very good! Hold completed for ${reps} reps.`
      : `很好！已完成保持动作共 ${reps} 次。`;
  }

  // Dynamic pattern 8: Edge Vision rep start / exercise start / ROM live readout
  const nextRepMatch = clean.match(/^เริ่มครั้งที่\s+(\d+)\s+ได้เลยครับ$/);
  if (nextRepMatch) {
    const repNum = nextRepMatch[1];
    return lang === 'en'
      ? `You may begin repetition number ${repNum}.`
      : `现在可以开始第 ${repNum} 次训练。`;
  }

  const startExerciseMatch = clean.match(/^เริ่มท่า\s+(.+?)\s+ช้าๆ\s*ให้ถึงเป้าหมายครับ$/);
  if (startExerciseMatch) {
    const exName = translateExerciseNameIfKnown(startExerciseMatch[1], lang);
    return lang === 'en'
      ? `Begin ${exName} slowly until you reach the target angle.`
      : `请缓慢开始${exName}，直至达到目标角度。`;
  }

  const setFinishedNamedExerciseMatch = clean.match(
    /^ยินดีด้วยครับ\s+ทำท่า\s+(.+?)\s+ครบทั้งเซตแล้ว$/,
  );
  if (setFinishedNamedExerciseMatch) {
    const exName = translateExerciseNameIfKnown(setFinishedNamedExerciseMatch[1], lang);
    return lang === 'en'
      ? `Congratulations! You have completed the full set of ${exName}.`
      : `恭喜您！已完成整组${exName}。`;
  }

  const angleProgressMatch = clean.match(/^(.+?)\s*\((\d+)°\s*\/\s*(\d+)°\)$/);
  if (angleProgressMatch) {
    const [, rawName, currentDeg, targetDeg] = angleProgressMatch;
    const exName = translateExerciseNameIfKnown(rawName, lang);
    return lang === 'en'
      ? `${exName}: ${currentDeg} degrees out of ${targetDeg} degrees.`
      : `${exName}：当前 ${currentDeg} 度，目标 ${targetDeg} 度。`;
  }

  // Fallback keyword checks for slight multi-person / wrong-limb / impostor variations
  if (clean.includes('หลายคน') || clean.includes('เพียงลำพัง')) {
    return EXACT_TRANSLATIONS[
      'ตรวจพบมีบุคคลอยู่หลายคนในเฟรม กรุณาให้คนไข้ยืนหน้ากล้องเพียงลำพัง ระบบจึงจะนับรอบการฝึกต่อครับ'
    ][lang];
  }
  if (clean.includes('ไม่ใช่คนไข้ที่ลงทะเบียน') || clean.includes('ตัวจริง')) {
    return EXACT_TRANSLATIONS[
      'คุณไม่ใช่คนไข้ที่ลงทะเบียนไว้ ระบบจะไม่นับรอบการฝึก กรุณาให้คนไข้ตัวจริงเข้าฝึกครับ'
    ][lang];
  }
  if (clean.includes('แขนซ้าย') && clean.includes('ไม่นับรอบ')) {
    return EXACT_TRANSLATIONS[
      'กรุณายกแขนซ้ายนะครับ ท่านี้ต้องทำด้วยแขนซ้าย ระบบจะไม่นับรอบหากยกแขนขวาครับ'
    ][lang];
  }
  if (clean.includes('แขนขวา') && clean.includes('ไม่นับรอบ')) {
    return EXACT_TRANSLATIONS[
      'กรุณายกแขนขวานะครับ ท่านี้ต้องทำด้วยแขนขวา ระบบจะไม่นับรอบหากยกแขนซ้ายครับ'
    ][lang];
  }

  return clean;
}

export function buildTtsProxyUrl(text: string, lang: VoiceCoachLanguage): string {
  const collapsed = sanitizeSpeechText(text);
  const maxLen = 190;
  let clipped = collapsed;
  if (collapsed.length > maxLen) {
    const sliced = collapsed.slice(0, maxLen);
    const lastSpace = sliced.lastIndexOf(' ');
    clipped = lastSpace >= Math.floor(maxLen * 0.5) ? sliced.slice(0, lastSpace).trim() : sliced.trim();
  }
  const params = new URLSearchParams({
    lang,
    text: clipped,
  });
  return `/api/tts?${params.toString()}`;
}

export function pickSpeechVoice(
  voices: readonly SpeechSynthesisVoice[],
  bcp47: string,
): SpeechSynthesisVoice | null {
  if (!voices.length) return null;
  const targetLower = bcp47.toLowerCase();
  const langPrefix = targetLower.split('-')[0];

  const exactMatches = voices.filter((v) => v.lang.toLowerCase() === targetLower);
  const chineseExtraMatches =
    langPrefix === 'zh'
      ? voices.filter((v) => {
          const l = v.lang.toLowerCase();
          return l.startsWith('zh') || l.startsWith('cmn') || v.name.toLowerCase().includes('chinese');
        })
      : [];
  const prefixMatches = voices.filter((v) => v.lang.toLowerCase().startsWith(langPrefix));

  const candidates =
    exactMatches.length > 0
      ? exactMatches
      : chineseExtraMatches.length > 0
        ? chineseExtraMatches
        : prefixMatches;

  if (!candidates.length) return null;

  // Prefer natural neural/cloud or Microsoft/Google voices when available
  const preferred = candidates.find((v) => {
    const name = v.name.toLowerCase();
    return name.includes('natural') || name.includes('google') || name.includes('microsoft');
  });
  return preferred ?? candidates[0];
}

function readStoredVoiceLang(): VoiceCoachLanguage {
  if (typeof window === 'undefined') return 'th';
  try {
    const raw = window.localStorage.getItem(VOICE_LANG_STORAGE_KEY);
    if (raw === 'th' || raw === 'en' || raw === 'zh') {
      return raw;
    }
  } catch {
    // Ignore storage restriction
  }
  return 'th';
}

export interface QueuedCoachUtterance {
  text: string;
  lang: VoiceCoachLanguage;
  kioskState: KioskState;
  isCompletion: boolean;
}

export interface SpeechTurnDecisionInput {
  isCurrentlySpeaking: boolean;
  forceInterrupt?: boolean;
  isEmergency?: boolean;
  incomingText: string;
  incomingLang: VoiceCoachLanguage;
  incomingState: KioskState;
  lastSpokenKey: string;
  lastSpokenState: KioskState | null;
  lastSpokenLang: VoiceCoachLanguage | null;
  elapsedSinceLastSpokenMs: number;
  existingQueued?: QueuedCoachUtterance | null;
}

export interface SpeechTurnDecision {
  action: 'SPEAK_IMMEDIATELY' | 'QUEUE_AFTER_CURRENT' | 'SKIP';
  queuedItem: QueuedCoachUtterance | null;
}

/**
 * Decides whether a new AI Voice Coach message should play right away, queue cleanly
 * behind an in-progress utterance so the current sentence finishes first, or be skipped.
 */
export function decideVoiceCoachSpeechTurn(
  input: SpeechTurnDecisionInput,
): SpeechTurnDecision {
  const clean = input.incomingText.trim();
  if (!clean) {
    return { action: 'SKIP', queuedItem: input.existingQueued ?? null };
  }

  const isEmergency = Boolean(input.isEmergency || input.incomingState === 'EMERGENCY_HALT');
  const isCompletion =
    input.incomingState === 'REP_COMPLETED' || input.incomingState === 'SET_FINISHED';
  const messageKey = `${input.incomingLang}:${clean}`;

  if (input.forceInterrupt || isEmergency) {
    return { action: 'SPEAK_IMMEDIATELY', queuedItem: null };
  }

  if (!isCompletion && messageKey === input.lastSpokenKey && input.elapsedSinceLastSpokenMs < 3500) {
    return { action: 'SKIP', queuedItem: input.existingQueued ?? null };
  }

  if (
    input.incomingState === 'HOLD_PHASE' &&
    input.lastSpokenState === 'HOLD_PHASE' &&
    input.lastSpokenLang === input.incomingLang &&
    input.elapsedSinceLastSpokenMs < 4500
  ) {
    return { action: 'SKIP', queuedItem: input.existingQueued ?? null };
  }

  if (input.isCurrentlySpeaking) {
    if (
      input.existingQueued?.isCompletion &&
      !isCompletion &&
      input.incomingState === 'ACTIVE_REP'
    ) {
      return { action: 'QUEUE_AFTER_CURRENT', queuedItem: input.existingQueued };
    }
    return {
      action: 'QUEUE_AFTER_CURRENT',
      queuedItem: {
        text: clean,
        lang: input.incomingLang,
        kioskState: input.incomingState,
        isCompletion,
      },
    };
  }

  return { action: 'SPEAK_IMMEDIATELY', queuedItem: null };
}

interface VoiceCoachBannerProps {
  textTh: string;
  kioskState: KioskState;
  highlightColor: HighlightColor;
  muted: boolean;
  onToggleMute: () => void;
}

export function VoiceCoachBanner({
  textTh,
  kioskState,
  highlightColor,
  muted,
  onToggleMute,
}: VoiceCoachBannerProps) {
  const [voiceLang, setVoiceLang] = useState<VoiceCoachLanguage>(() => readStoredKioskVoiceLang());
  useEffect(() => subscribeKioskVoiceLang(setVoiceLang), []);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [voices, setVoices] = useState<SpeechSynthesisVoice[]>([]);

  const isSpeakingRef = useRef<boolean>(false);
  const lastSpokenKeyRef = useRef<string>('');
  const lastSpokenAtRef = useRef<number>(0);
  const lastSpokenStateRef = useRef<KioskState | null>(null);
  const lastSpokenLangRef = useRef<VoiceCoachLanguage | null>(null);
  const latestKioskStateRef = useRef<KioskState>(kioskState);
  latestKioskStateRef.current = kioskState;
  const pendingSpeakTimerRef = useRef<number | null>(null);
  const webSpeechTimerRef = useRef<number | null>(null);
  const speakingWatchdogRef = useRef<number | null>(null);
  const queuedSpeechRef = useRef<QueuedCoachUtterance | null>(null);
  const onSpeechEndedRef = useRef<(seq: number) => void>(() => {});
  const activeUtteranceRef = useRef<SpeechSynthesisUtterance | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const speakSeqRef = useRef<number>(0);
  const blockedByAutoplayRef = useRef<boolean>(false);

  const langOption = useMemo(
    () => VOICE_COACH_LANGUAGES.find((item) => item.id === voiceLang) ?? VOICE_COACH_LANGUAGES[0],
    [voiceLang],
  );

  const displayedText = useMemo(
    () => translateVoiceCoachText(textTh, voiceLang),
    [textTh, voiceLang],
  );

  const latestDisplayRef = useRef<{ text: string; lang: VoiceCoachLanguage; muted: boolean }>({
    text: displayedText,
    lang: voiceLang,
    muted,
  });
  latestDisplayRef.current = { text: displayedText, lang: voiceLang, muted };

  const clearPendingSpeakTimer = useCallback(() => {
    if (pendingSpeakTimerRef.current !== null && typeof window !== 'undefined') {
      window.clearTimeout(pendingSpeakTimerRef.current);
      pendingSpeakTimerRef.current = null;
    }
  }, []);

  const clearSpeakingWatchdog = useCallback(() => {
    if (speakingWatchdogRef.current !== null && typeof window !== 'undefined') {
      window.clearTimeout(speakingWatchdogRef.current);
      speakingWatchdogRef.current = null;
    }
  }, []);

  const stopAllSpeech = useCallback(
    (clearQueue = true) => {
      speakSeqRef.current += 1;
      clearSpeakingWatchdog();
      if (clearQueue) {
        queuedSpeechRef.current = null;
        clearPendingSpeakTimer();
      }
      if (webSpeechTimerRef.current !== null && typeof window !== 'undefined') {
        window.clearTimeout(webSpeechTimerRef.current);
        webSpeechTimerRef.current = null;
      }
      const audio = audioRef.current;
      if (audio) {
        try {
          audio.onplay = null;
          audio.onended = null;
          audio.onerror = null;
          audio.pause();
          audio.currentTime = 0;
        } catch {
          // Ignore audio reset errors
        }
      }
      activeUtteranceRef.current = null;
      if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
        try {
          window.speechSynthesis.cancel();
        } catch {
          // Ignore unavailable speechSynthesis cancel
        }
      }
      isSpeakingRef.current = false;
      setIsSpeaking(false);
    },
    [clearPendingSpeakTimer, clearSpeakingWatchdog],
  );

  // Load & subscribe to available OS/browser SpeechSynthesis voices
  useEffect(() => {
    if (typeof window === 'undefined' || !('speechSynthesis' in window)) return;
    const synth = window.speechSynthesis;
    const syncVoices = () => {
      try {
        const list = synth.getVoices();
        if (list.length > 0) {
          setVoices(list);
        }
      } catch {
        // Ignore unavailable voice list
      }
    };
    syncVoices();
    if ('addEventListener' in synth) {
      synth.addEventListener('voiceschanged', syncVoices);
      return () => synth.removeEventListener('voiceschanged', syncVoices);
    }
    return undefined;
  }, []);

  useEffect(() => {
    return () => {
      clearPendingSpeakTimer();
      clearSpeakingWatchdog();
      stopAllSpeech(true);
    };
  }, [clearPendingSpeakTimer, clearSpeakingWatchdog, stopAllSpeech]);

  const speakWithWebSpeech = useCallback(
    (clean: string, targetLang: VoiceCoachLanguage, seq: number) => {
      if (
        muted ||
        !clean ||
        typeof window === 'undefined' ||
        !('speechSynthesis' in window) ||
        typeof SpeechSynthesisUtterance === 'undefined'
      ) {
        return;
      }

      const targetOpt =
        VOICE_COACH_LANGUAGES.find((item) => item.id === targetLang) ?? VOICE_COACH_LANGUAGES[0];

      try {
        const synth = window.speechSynthesis;
        const wasBusy = synth.speaking || synth.pending;
        if (wasBusy) {
          synth.cancel();
        }

        const startUtterance = () => {
          if (speakSeqRef.current !== seq) return;
          if (synth.paused) {
            synth.resume();
          }

          const currentVoices = voices.length > 0 ? voices : synth.getVoices();
          const matchedVoice = pickSpeechVoice(currentVoices, targetOpt.bcp47);
          const cleanSpeechText = sanitizeSpeechText(clean);
          if (!cleanSpeechText) return;

          const utterance = new SpeechSynthesisUtterance(cleanSpeechText);
          utterance.lang = matchedVoice?.lang || targetOpt.bcp47;
          if (matchedVoice) {
            utterance.voice = matchedVoice;
          }
          utterance.rate = targetLang === 'th' ? 0.96 : targetLang === 'zh' ? 0.98 : 1.0;
          utterance.pitch = 1.0;
          utterance.volume = 1.0;

          activeUtteranceRef.current = utterance;

          utterance.onstart = () => {
            if (speakSeqRef.current !== seq) return;
            blockedByAutoplayRef.current = false;
            isSpeakingRef.current = true;
            setIsSpeaking(true);
          };
          utterance.onend = () => {
            if (activeUtteranceRef.current === utterance) {
              activeUtteranceRef.current = null;
            }
            onSpeechEndedRef.current(seq);
          };
          utterance.onerror = (event: SpeechSynthesisErrorEvent) => {
            if (activeUtteranceRef.current === utterance) {
              activeUtteranceRef.current = null;
            }
            if (event?.error === 'not-allowed') {
              clearSpeakingWatchdog();
              if (speakSeqRef.current === seq) {
                blockedByAutoplayRef.current = true;
                isSpeakingRef.current = false;
                setIsSpeaking(false);
              }
              return;
            }
            onSpeechEndedRef.current(seq);
          };

          isSpeakingRef.current = true;
          setIsSpeaking(true);
          clearSpeakingWatchdog();
          const webWatchdogMs = Math.max(11000, clean.length * 180 + 3000);
          speakingWatchdogRef.current = window.setTimeout(() => {
            speakingWatchdogRef.current = null;
            // Unlock cleanly without cancelling active slow TTS mid-sentence
            onSpeechEndedRef.current(seq);
          }, webWatchdogMs);
          synth.speak(utterance);
        };

        // Delay slightly after synth.cancel() to avoid Chromium's cancel+speak drop race
        if (wasBusy) {
          webSpeechTimerRef.current = window.setTimeout(() => {
            webSpeechTimerRef.current = null;
            startUtterance();
          }, 50);
        } else {
          startUtterance();
        }
      } catch {
        clearSpeakingWatchdog();
        isSpeakingRef.current = false;
        setIsSpeaking(false);
      }
    },
    [muted, voices, clearSpeakingWatchdog],
  );

  const speakNow = useCallback(
    (
      text: string,
      targetLang: VoiceCoachLanguage,
      options?: { forceInterrupt?: boolean; stateOverride?: KioskState },
    ) => {
      const clean = text.trim();
      if (muted || !clean || typeof window === 'undefined') {
        return;
      }

      const activeState = options?.stateOverride ?? latestKioskStateRef.current;

      if (!options?.forceInterrupt && isSpeakingRef.current) {
        const decision = decideVoiceCoachSpeechTurn({
          isCurrentlySpeaking: true,
          incomingText: clean,
          incomingLang: targetLang,
          incomingState: activeState,
          lastSpokenKey: lastSpokenKeyRef.current,
          lastSpokenState: lastSpokenStateRef.current,
          lastSpokenLang: lastSpokenLangRef.current,
          elapsedSinceLastSpokenMs: Date.now() - lastSpokenAtRef.current,
          existingQueued: queuedSpeechRef.current,
        });
        if (decision.action === 'QUEUE_AFTER_CURRENT') {
          queuedSpeechRef.current = decision.queuedItem;
        }
        return;
      }

      stopAllSpeech(Boolean(options?.forceInterrupt));
      const seq = speakSeqRef.current;

      lastSpokenKeyRef.current = `${targetLang}:${clean}`;
      lastSpokenAtRef.current = Date.now();
      lastSpokenStateRef.current = activeState;
      lastSpokenLangRef.current = targetLang;

      // Primary path: play real neural MP3 audio via same-origin /api/tts proxy so Thai,
      // English, and Chinese work even when OS/browser lacks local th-TH or zh-CN TTS voices.
      if (typeof Audio !== 'undefined') {
        try {
          let audio = audioRef.current;
          if (!audio) {
            audio = new Audio();
            audio.preload = 'auto';
            audioRef.current = audio;
          }
          let webFallbackTriggered = false;
          const triggerWebFallbackOnce = () => {
            if (webFallbackTriggered || speakSeqRef.current !== seq) return;
            webFallbackTriggered = true;
            clearSpeakingWatchdog();
            isSpeakingRef.current = false;
            setIsSpeaking(false);
            speakWithWebSpeech(clean, targetLang, seq);
          };
          audio.playbackRate = targetLang === 'th' ? 1.0 : targetLang === 'zh' ? 0.98 : 1.0;

          audio.onplay = () => {
            if (speakSeqRef.current !== seq) return;
            blockedByAutoplayRef.current = false;
            isSpeakingRef.current = true;
            setIsSpeaking(true);
          };
          audio.onended = () => {
            onSpeechEndedRef.current(seq);
          };
          audio.onerror = triggerWebFallbackOnce;

          audio.src = buildTtsProxyUrl(clean, targetLang);
          const playPromise = audio.play();
          isSpeakingRef.current = true;
          setIsSpeaking(true);
          clearSpeakingWatchdog();
          const audioWatchdogMs = Math.max(11000, clean.length * 180 + 3000);
          speakingWatchdogRef.current = window.setTimeout(() => {
            speakingWatchdogRef.current = null;
            // Safe watchdog: unlock state without abruptly cutting off an active TTS voice
            onSpeechEndedRef.current(seq);
          }, audioWatchdogMs);

          if (playPromise && typeof playPromise.catch === 'function') {
            playPromise.catch((err: unknown) => {
              if (speakSeqRef.current !== seq) return;
              const errName =
                err && typeof err === 'object' && 'name' in err
                  ? String((err as { name: string }).name)
                  : '';
              if (errName === 'NotAllowedError') {
                clearSpeakingWatchdog();
                isSpeakingRef.current = false;
                setIsSpeaking(false);
                blockedByAutoplayRef.current = true;
                return;
              }
              if (errName === 'AbortError') {
                return;
              }
              triggerWebFallbackOnce();
            });
          }
          return;
        } catch {
          // Fall through to Web Speech API
        }
      }

      speakWithWebSpeech(clean, targetLang, seq);
    },
    [muted, stopAllSpeech, speakWithWebSpeech, clearSpeakingWatchdog],
  );

  onSpeechEndedRef.current = (seq: number) => {
    if (speakSeqRef.current !== seq) return;
    clearSpeakingWatchdog();
    isSpeakingRef.current = false;
    setIsSpeaking(false);

    if (latestDisplayRef.current.muted) {
      queuedSpeechRef.current = null;
      return;
    }

    const queued = queuedSpeechRef.current;
    queuedSpeechRef.current = null;
    if (!queued) return;

    const now = Date.now();
    if (queued.isCompletion && `${queued.lang}:${queued.text}` !== lastSpokenKeyRef.current) {
      clearPendingSpeakTimer();
      pendingSpeakTimerRef.current = window.setTimeout(() => {
        pendingSpeakTimerRef.current = null;
        if (!latestDisplayRef.current.muted) {
          speakNow(queued.text, queued.lang, { stateOverride: queued.kioskState });
        }
      }, 220);
      return;
    }

    const liveText = latestDisplayRef.current.text.trim();
    const liveLang = latestDisplayRef.current.lang;
    const liveState = latestKioskStateRef.current;
    const nextDecision = decideVoiceCoachSpeechTurn({
      isCurrentlySpeaking: false,
      incomingText: liveText,
      incomingLang: liveLang,
      incomingState: liveState,
      lastSpokenKey: lastSpokenKeyRef.current,
      lastSpokenState: lastSpokenStateRef.current,
      lastSpokenLang: lastSpokenLangRef.current,
      elapsedSinceLastSpokenMs: now - lastSpokenAtRef.current,
    });
    if (nextDecision.action === 'SPEAK_IMMEDIATELY') {
      clearPendingSpeakTimer();
      pendingSpeakTimerRef.current = window.setTimeout(() => {
        pendingSpeakTimerRef.current = null;
        if (!latestDisplayRef.current.muted) {
          speakNow(liveText, liveLang, { stateOverride: liveState });
        }
      }, 220);
    }
  };

  // Unlock / resume Web Speech + Audio autoplay on first user interaction
  useEffect(() => {
    if (typeof window === 'undefined') return;
    const unlockSpeech = (event: Event) => {
      try {
        if ('speechSynthesis' in window && window.speechSynthesis.paused) {
          window.speechSynthesis.resume();
        }
        const target = event.target instanceof Element ? event.target : null;
        if (target?.closest('[data-voice-coach-control="true"]')) {
          return;
        }
        if (blockedByAutoplayRef.current && !latestDisplayRef.current.muted) {
          blockedByAutoplayRef.current = false;
          speakNow(latestDisplayRef.current.text, latestDisplayRef.current.lang, {
            forceInterrupt: true,
          });
        }
      } catch {
        // Ignore
      }
    };
    window.addEventListener('pointerdown', unlockSpeech, { passive: true });
    window.addEventListener('keydown', unlockSpeech, { passive: true });
    return () => {
      window.removeEventListener('pointerdown', unlockSpeech);
      window.removeEventListener('keydown', unlockSpeech);
    };
  }, [speakNow]);

  // Speak automatically whenever the displayed coaching message or language changes, waiting for active speech to end
  useEffect(() => {
    const clean = displayedText.trim();

    if (!clean || muted) {
      clearPendingSpeakTimer();
      if (muted) {
        stopAllSpeech(true);
      } else {
        queuedSpeechRef.current = null;
        isSpeakingRef.current = false;
        setIsSpeaking(false);
      }
      return;
    }

    const now = Date.now();
    const decision = decideVoiceCoachSpeechTurn({
      isCurrentlySpeaking: isSpeakingRef.current,
      isEmergency: kioskState === 'EMERGENCY_HALT',
      incomingText: clean,
      incomingLang: voiceLang,
      incomingState: kioskState,
      lastSpokenKey: lastSpokenKeyRef.current,
      lastSpokenState: lastSpokenStateRef.current,
      lastSpokenLang: lastSpokenLangRef.current,
      elapsedSinceLastSpokenMs: now - lastSpokenAtRef.current,
      existingQueued: queuedSpeechRef.current,
    });

    if (decision.action === 'SKIP') {
      return;
    }

    if (decision.action === 'QUEUE_AFTER_CURRENT') {
      queuedSpeechRef.current = decision.queuedItem;
      return;
    }

    clearPendingSpeakTimer();
    speakNow(clean, voiceLang, {
      forceInterrupt: kioskState === 'EMERGENCY_HALT',
      stateOverride: kioskState,
    });
  }, [displayedText, voiceLang, kioskState, muted, speakNow, clearPendingSpeakTimer, stopAllSpeech]);

  // Repeat warning every 4s while paused by Multi-Person / Impostor guard without interrupting active speech
  useEffect(() => {
    if (muted || (kioskState !== 'PAUSED_MULTI_PERSON' && kioskState !== 'PAUSED_IMPOSTOR')) {
      return;
    }
    const timer = window.setInterval(() => {
      const clean = displayedText.trim();
      if (!clean || isSpeakingRef.current) return;
      speakNow(clean, voiceLang);
    }, 4000);
    return () => window.clearInterval(timer);
  }, [kioskState, displayedText, voiceLang, muted, speakNow]);

  const handleSelectLanguage = (nextLang: VoiceCoachLanguage) => {
    setVoiceLang(nextLang);
    writeStoredKioskVoiceLang(nextLang);
    clearPendingSpeakTimer();
    const nextText = translateVoiceCoachText(textTh, nextLang);
    if (!muted) {
      speakNow(nextText, nextLang, { forceInterrupt: true });
    }
  };

  const badgeTone =
    highlightColor === 'RED'
      ? 'bg-rose-500 text-white'
      : highlightColor === 'AMBER'
        ? 'bg-amber-500 text-white'
        : 'bg-gradient-to-br from-[#0284C7] to-[#10B981] text-white';

  const textTone =
    highlightColor === 'RED'
      ? 'text-rose-700'
      : highlightColor === 'AMBER'
        ? 'text-amber-800'
        : 'text-[#0F4C81]';

  const bars = [0.45, 0.9, 1.0, 0.75, 0.5];

  return (
    <motion.div
      layout
      className="glass-card mx-auto flex w-full flex-col justify-between gap-3 rounded-3xl px-4 py-3 sm:flex-row sm:items-center sm:gap-4 sm:rounded-[32px] sm:px-6"
    >
      <div className="flex min-w-0 flex-1 items-center gap-3.5 sm:gap-4">
        {/* Animated 5-bar AI Voice Coach Soundwave + Replay Trigger */}
        <button
          type="button"
          data-voice-coach-control="true"
          onClick={() => speakNow(displayedText, voiceLang, { forceInterrupt: true })}
          title={langOption.replayTitle}
          className={`flex h-12 w-14 shrink-0 items-center justify-center gap-1 rounded-full shadow-md transition-transform active:scale-95 ${badgeTone}`}
        >
          {bars.map((peak, idx) => (
            <motion.span
              key={idx}
              className="w-1.5 rounded-full bg-white"
              animate={
                !muted && (isSpeaking || highlightColor !== 'GREEN' || kioskState === 'HOLD_PHASE')
                  ? { height: ['8px', `${Math.round(peak * 24)}px`, '8px'] }
                  : { height: '8px' }
              }
              transition={{
                repeat: Infinity,
                duration: 0.7 + idx * 0.1,
                delay: idx * 0.08,
                ease: 'easeInOut',
              }}
            />
          ))}
        </button>

        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2 text-xs font-bold uppercase tracking-wider text-[#0284C7]">
            <span>{langOption.badgeTitle}</span>
            <span className="rounded-full bg-sky-100 px-2 py-0.5 text-[10px] text-[#0F4C81]">
              {muted
                ? langOption.statusMuted
                : isSpeaking
                  ? langOption.statusSpeaking
                  : langOption.statusReady}
            </span>
          </div>
          <p className={`line-clamp-2 text-lg font-bold leading-snug sm:text-xl ${textTone}`}>
            “{displayedText}”
          </p>
          {voiceLang !== 'th' && textTh.trim() && (
            <p className="truncate text-xs font-medium text-slate-500">
              ไทย: {textTh.trim()}
            </p>
          )}
        </div>
      </div>

      {/* 3-Language Selector + Replay + Mute/Unmute Controls */}
      <div data-voice-coach-control="true" className="flex flex-wrap items-center justify-end gap-2">
        <div
          role="group"
          aria-label="เลือกภาษาเสียงพากย์ AI Coach"
          className="inline-flex items-center gap-1 rounded-full border border-sky-200 bg-white/90 p-1 shadow-sm"
        >
          <Languages className="ml-1.5 mr-0.5 hidden h-4 w-4 text-[#0284C7] md:block" />
          {VOICE_COACH_LANGUAGES.map((opt) => {
            const active = opt.id === voiceLang;
            return (
              <button
                key={opt.id}
                type="button"
                aria-pressed={active}
                onClick={() => handleSelectLanguage(opt.id)}
                className={`rounded-full px-2.5 py-1.5 text-xs font-extrabold transition-all sm:px-3 ${
                  active
                    ? 'bg-[#0284C7] text-white shadow-sm'
                    : 'text-[#0F4C81] hover:bg-sky-50'
                }`}
              >
                {opt.numberLabel}
              </button>
            );
          })}
        </div>

        <button
          type="button"
          onClick={() => speakNow(displayedText, voiceLang, { forceInterrupt: true })}
          disabled={muted}
          title={langOption.replayTitle}
          className="inline-flex shrink-0 items-center gap-1.5 rounded-full border border-sky-200 bg-white px-3 py-2 text-xs font-bold text-[#0F4C81] transition-colors hover:bg-sky-50 disabled:cursor-not-allowed disabled:opacity-50"
        >
          <RotateCcw className="h-4 w-4 text-[#0284C7]" />
          <span className="hidden lg:inline">พูดซ้ำ</span>
        </button>

        <button
          type="button"
          onClick={() => {
            if (!muted) {
              stopAllSpeech(true);
              cancelAllKioskVoiceSpeech(audioRef.current);
            }
            onToggleMute();
          }}
          aria-pressed={muted}
          className={`inline-flex shrink-0 items-center gap-2 rounded-2xl border-2 px-4 py-2.5 text-sm font-extrabold shadow-sm transition-all cursor-pointer active:scale-95 sm:px-5 ${
            muted
              ? 'border-amber-400 bg-amber-100 text-amber-950 hover:bg-amber-200'
              : 'border-emerald-400 bg-emerald-100 text-emerald-950 hover:bg-emerald-200'
          }`}
        >
          {muted ? (
            <VolumeX className="h-5 w-5 shrink-0 text-amber-700" />
          ) : (
            <Volume2 className="h-5 w-5 shrink-0 text-emerald-700" />
          )}
          <span>{getSeniorVoiceToggleLabel(voiceLang, muted)}</span>
        </button>
      </div>
    </motion.div>
  );
}
