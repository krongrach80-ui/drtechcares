import type { BodySide, ExerciseConfig } from './types';

export type ExerciseSideCode = 'LEFT' | 'RIGHT' | 'BOTH';

/**
 * Legacy / non-stroke / split-side exercise codes permanently retired from Dr.Tech.Care.
 * Canonical library is the 15 Unified Master Exercises (`STROKE_*` & `MOPH_*`) with dynamic L/R/BOTH side selection.
 */
export const DEPRECATED_EXERCISE_CODES: readonly string[] = [
  // Lower-limb / standing fall-risk exercises
  'BAL_STAND_R',
  'BAL_STAND_L',
  'KNEE_EXT_R',
  'KNEE_EXT_L',
  'HIP_FLEX_R',
  'HIP_FLEX_L',
  'HEEL_RAISE_R',
  'HEEL_RAISE_L',
  'SIT_TO_STAND',
  // Orthopedic postures unsuitable for front-facing kiosk stroke therapy
  'SH_PEND_R',
  'SH_PEND_L',
  'SH_WALL_SLIDE_R',
  'SH_WALL_SLIDE_L',
  // Duplicate non-clinical-labeled exercises
  'SH_FLEX_R',
  'SH_FLEX_L',
  'SH_ABD_R',
  'SH_ABD_L',
  'ELB_FLEX_R',
  'ELB_FLEX_L',
  'SH_SCAP_R',
  'SH_SCAP_L',
  'SH_OVH_R',
  'SH_OVH_L',
  'NECK_TILT_R',
  'NECK_TILT_L',
  // Retired legacy trunk/bilateral IDs
  'TRUNK_SIT_STRETCH',
  'MOPH_BILATERAL_ARM_RAISE',
  // Split _R / _L catalog cards merged into the 15 Unified Master Exercises
  'STROKE_SH_FLEX_R',
  'STROKE_SH_FLEX_L',
  'STROKE_SH_ABD_R',
  'STROKE_SH_ABD_L',
  'STROKE_CROSS_BODY_R',
  'STROKE_CROSS_BODY_L',
  'STROKE_SH_ROT_R',
  'STROKE_SH_ROT_L',
  'STROKE_SCAP_PROTRACT_R',
  'STROKE_SCAP_PROTRACT_L',
  'STROKE_ELB_FLEX_EXT_R',
  'STROKE_ELB_FLEX_EXT_L',
  'STROKE_HAND_TO_HEAD_R',
  'STROKE_HAND_TO_HEAD_L',
  'STROKE_FOREARM_TURN_R',
  'STROKE_FOREARM_TURN_L',
  'STROKE_WRIST_FINGER_R',
  'STROKE_WRIST_FINGER_L',
  'STROKE_THUMB_PINCH_R',
  'STROKE_THUMB_PINCH_L',
  'MOPH_NECK_TILT_R',
  'MOPH_NECK_TILT_L',
  'MOPH_NECK_TURN_BILATERAL',
] as const;

export const DEPRECATED_EXERCISE_CODE_SET: ReadonlySet<string> = new Set(
  DEPRECATED_EXERCISE_CODES.map((c) => c.toUpperCase()),
);

/** Legacy `ex-01`..`ex-40` and old `_R`/`_L` codes mapped cleanly to unified master + side info. */
export const LEGACY_EXERCISE_ID_MAP: Readonly<Record<string, string>> = {
  'ex-01': 'STROKE_SH_FLEX_RIGHT',
  'ex-02': 'STROKE_SH_FLEX_LEFT',
  'ex-03': 'STROKE_SH_ABD_RIGHT',
  'ex-04': 'STROKE_SH_ABD_LEFT',
  'ex-05': 'STROKE_ELB_FLEX_EXT_RIGHT',
  'ex-06': 'STROKE_ELB_FLEX_EXT_LEFT',
  'ex-07': 'STROKE_SH_ROT_RIGHT',
  'ex-08': 'STROKE_SH_ROT_LEFT',
  'ex-09': 'STROKE_SCAP_PROTRACT_RIGHT',
  'ex-10': 'STROKE_SCAP_PROTRACT_LEFT',
  'ex-11': 'STROKE_BOBATH_ARM_RAISE',
  'ex-12': 'STROKE_BOBATH_ARM_RAISE',
  'ex-19': 'MOPH_NECK_TILT_RIGHT',
  'ex-20': 'MOPH_NECK_TILT_LEFT',
  'ex-23': 'MOPH_NECK_TURN',
  'ex-24': 'STROKE_HAND_TO_HEAD_RIGHT',
  'ex-25': 'STROKE_HAND_TO_HEAD_LEFT',
  'ex-27': 'STROKE_SH_FLEX_RIGHT',
  'ex-28': 'STROKE_SH_FLEX_LEFT',
  'ex-29': 'STROKE_SH_ABD_RIGHT',
  'ex-30': 'STROKE_SH_ABD_LEFT',
  'ex-31': 'STROKE_ELB_FLEX_EXT_RIGHT',
  'ex-32': 'STROKE_ELB_FLEX_EXT_LEFT',
  'ex-33': 'STROKE_SH_ROT_RIGHT',
  'ex-34': 'STROKE_SH_ROT_LEFT',
  'ex-35': 'STROKE_WRIST_FINGER_RIGHT',
  'ex-36': 'STROKE_WRIST_FINGER_LEFT',
  'ex-37': 'MOPH_NECK_TILT_RIGHT',
  'ex-38': 'MOPH_NECK_TILT_LEFT',
  'ex-39': 'MOPH_SHOULDER_SHRUG_ROLL',
  'ex-40': 'STROKE_BOBATH_ARM_RAISE',
  moph_neck_turn_bilateral: 'MOPH_NECK_TURN',
  moph_bilateral_arm_raise: 'STROKE_BOBATH_ARM_RAISE',
  right_shoulder_flexion: 'STROKE_SH_FLEX_RIGHT',
  left_shoulder_flexion: 'STROKE_SH_FLEX_LEFT',
  right_shoulder_abduction: 'STROKE_SH_ABD_RIGHT',
  left_shoulder_abduction: 'STROKE_SH_ABD_LEFT',
  elbow_flexion_right: 'STROKE_ELB_FLEX_EXT_RIGHT',
  elbow_flexion_left: 'STROKE_ELB_FLEX_EXT_LEFT',
};

/**
 * Unified 15 Clinical Master Rehabilitation Exercises (Stroke ADL & MOPH Elderly):
 * - Category 1 (12): แขน ไหล่ ศอก และมือ (Stroke ADL — 12 ท่าหลัก)
 * - Category 2 (3): คอและบ่า (สธ. ผู้สูงอายุ & Stroke Upper Trunk — 3 ท่าหลัก)
 * Every posture supports side selection (`LEFT` / `RIGHT` / `BOTH`) within a single card.
 */
export const CLINICAL_EXERCISES: readonly ExerciseConfig[] = [
  // =========================================================================
  // หมวด 1: แขน ไหล่ ศอก และมือ (Stroke ADL — 12 ท่าหลัก)
  // =========================================================================
  {
    id: 'STROKE_SH_FLEX',
    code: 'STROKE_SH_FLEX',
    category: 'upper_limb',
    clinicalFocus: 'stroke_adl',
    name: {
      th: '[Stroke ADL] ยกแขนไปด้านหน้า (ฝึกเอื้อม/สอดแขนเสื้อ)',
      en: 'Stroke ADL: Shoulder Forward Flexion',
      cn: '卒中康复 ADL：肩前屈上举（伸手/穿衣袖）',
    },
    instruction: {
      th: 'ยกแขนตรงไปด้านหน้าเท่าที่ไหว (ไม่ฝืน) เพื่อฝึกเอื้อมหยิบของหรือสอดแขนเสื้อ ค้างไว้ 3 วินาที',
      en: 'Raise arm smoothly forward within safe pain-free range for reaching/sleeving and hold 3s',
      cn: '舒适范围内缓慢向前抬起手臂练习取物与穿衣，保持3秒',
    },
    detection: {
      mode: 'angle_rep_hold',
      jointGroup: 'shoulder',
      side: 'right',
      targetMinDeg: 90,
      targetMaxDeg: 165,
      targetRomDeg: 125,
      startRestMaxDeg: 30,
      holdSec: 3,
      defaultSets: 2,
      defaultReps: 6,
      requireStraightElbow: false,
      expectedPlane: 'flexion',
    },
  },
  {
    id: 'STROKE_SH_ABD',
    code: 'STROKE_SH_ABD',
    category: 'upper_limb',
    clinicalFocus: 'stroke_adl',
    name: {
      th: '[Stroke ADL] กางแขนด้านข้าง (ฝึกกางแขนสวมเสื้อ)',
      en: 'Stroke ADL: Shoulder Abduction',
      cn: '卒中康复 ADL：肩外展上举（穿衣训练）',
    },
    instruction: {
      th: 'กางแขนออกด้านข้างช้าๆ โดยไม่ยักไหล่ช่วย ค้างไว้ 3 วินาที แล้วลดลง',
      en: 'Abduct arm outward gently without hiking shoulder, hold 3s, then lower calmly',
      cn: '缓慢向体侧外展手臂，保持肩膀放松不耸肩，维持3秒后放下',
    },
    detection: {
      mode: 'angle_rep_hold',
      jointGroup: 'shoulder',
      side: 'right',
      targetMinDeg: 85,
      targetMaxDeg: 160,
      targetRomDeg: 115,
      startRestMaxDeg: 30,
      holdSec: 3,
      defaultSets: 2,
      defaultReps: 6,
      requireStraightElbow: false,
      expectedPlane: 'abduction',
    },
  },
  {
    id: 'STROKE_CROSS_BODY',
    code: 'STROKE_CROSS_BODY',
    category: 'upper_limb',
    clinicalFocus: 'stroke_adl',
    name: {
      th: '[Stroke ADL] เอื้อมมือแตะไหล่ฝั่งตรงข้าม (ฝึกดึงแขนเสื้อ/ใส่เสื้อคลุม)',
      en: 'Stroke ADL: Cross-Body Shoulder Reach',
      cn: '卒中康复 ADL：横越胸前触碰对侧肩',
    },
    instruction: {
      th: 'ยกมืออ้อมหน้าอกไปแตะหัวไหล่ฝั่งตรงข้ามช้าๆ ค้างไว้ 3 วินาทีแล้วคลายออก',
      en: 'Bring hand smoothly across chest to touch opposite shoulder, hold 3s, then release',
      cn: '缓慢横越胸前轻触对侧肩膀，保持3秒后放松收回',
    },
    detection: {
      mode: 'angle_rep_hold',
      jointGroup: 'shoulder',
      side: 'right',
      targetMinDeg: 75,
      targetMaxDeg: 145,
      targetRomDeg: 110,
      startRestMaxDeg: 25,
      holdSec: 3,
      defaultSets: 2,
      defaultReps: 6,
      requireStraightElbow: false,
      expectedPlane: 'cross_body',
    },
  },
  {
    id: 'STROKE_SH_ROT',
    code: 'STROKE_SH_ROT',
    category: 'upper_limb',
    clinicalFocus: 'stroke_adl',
    name: {
      th: '[Stroke ADL] หมุนข้อไหล่เข้า-ออก (อ้อมใส่เสื้อ/ติดกระดุม)',
      en: 'Stroke ADL: Shoulder External/Internal Rotation',
      cn: '卒中康复 ADL：肩关节内外旋（穿衣与扣纽扣）',
    },
    instruction: {
      th: 'แนบข้อศอกชิดลำตัว หมุนปลายแขนออกด้านข้างช้าๆ ค้างไว้ 3 วินาที',
      en: 'Keep elbow tucked by waist and rotate forearm outward gently, hold 3s',
      cn: '肘部贴紧腰侧，缓慢向外旋转前臂并保持3秒',
    },
    detection: {
      mode: 'angle_rep_hold',
      jointGroup: 'shoulder',
      side: 'right',
      targetMinDeg: 45,
      targetMaxDeg: 80,
      targetRomDeg: 65,
      startRestMaxDeg: 15,
      holdSec: 3,
      defaultSets: 2,
      defaultReps: 6,
      expectedPlane: 'rotation',
    },
  },
  {
    id: 'STROKE_SCAP_PROTRACT',
    code: 'STROKE_SCAP_PROTRACT',
    category: 'upper_limb',
    clinicalFocus: 'stroke_adl',
    name: {
      th: '[Stroke ADL] เหยียดแขนดันไปด้านหน้าระดับอก (ป้องกันข้อไหล่หลุด)',
      en: 'Stroke ADL: Scapular Protraction Forward Punch',
      cn: '卒中康复 ADL：胸前伸直前推（预防肩关节半脱位）',
    },
    instruction: {
      th: 'เหยียดแขนระดับอกแล้วดันสะบักไปด้านหน้าเบาๆ ค้างไว้ 3 วินาที',
      en: 'Extend arm forward at chest level and push shoulder blade smoothly forward, hold 3s',
      cn: '手臂平胸伸直并轻柔前推肩胛骨，维持3秒以稳定肩臼',
    },
    detection: {
      mode: 'angle_rep_hold',
      jointGroup: 'shoulder',
      side: 'right',
      targetMinDeg: 75,
      targetMaxDeg: 120,
      targetRomDeg: 95,
      startRestMaxDeg: 25,
      holdSec: 3,
      defaultSets: 2,
      defaultReps: 6,
      requireStraightElbow: true,
      minElbowDeg: 135,
      expectedPlane: 'scap_protract',
    },
  },
  {
    id: 'STROKE_ELB_FLEX_EXT',
    code: 'STROKE_ELB_FLEX_EXT',
    category: 'upper_limb',
    clinicalFocus: 'stroke_adl',
    name: {
      th: '[Stroke ADL] งอ-เหยียดข้อศอก (ดึงแขนเสื้อ/ป้อนอาหาร)',
      en: 'Stroke ADL: Elbow Flexion & Extension',
      cn: '卒中康复 ADL：肘关节屈伸（拉袖/进食训练）',
    },
    instruction: {
      th: 'งอข้อศอกพับมือเข้าหาไหล่/ปากช้าๆ ค้างไว้ 2 วินาที แล้วเหยียดตรง',
      en: 'Flex elbow smoothly toward shoulder/mouth, hold 2s, then fully extend arm straight',
      cn: '缓慢屈曲手肘靠近肩部保持2秒，再充分伸直缓解痉挛',
    },
    detection: {
      mode: 'angle_rep_hold',
      jointGroup: 'elbow',
      side: 'right',
      targetMinDeg: 95,
      targetMaxDeg: 145,
      targetRomDeg: 125,
      startRestMaxDeg: 40,
      holdSec: 2,
      defaultSets: 2,
      defaultReps: 8,
    },
  },
  {
    id: 'STROKE_HAND_TO_HEAD',
    code: 'STROKE_HAND_TO_HEAD',
    category: 'upper_limb',
    clinicalFocus: 'stroke_adl',
    name: {
      th: '[Stroke ADL] ยกมือแตะศีรษะ (ฝึกสวมเสื้อทางศีรษะ/หวีผม)',
      en: 'Stroke ADL: Hand-to-Head Grooming Reach',
      cn: '卒中康复 ADL：上举触头顶（梳头/穿套头衣）',
    },
    instruction: {
      th: 'ยกมือขึ้นแตะศีรษะช้าๆ โดยไม่ก้มคอหาแขน ค้างไว้ 3 วินาที',
      en: 'Lift hand gently to touch head without bending neck forward, hold 3s',
      cn: '缓慢举手轻触头顶，颈部保持正直不前倾，维持3秒',
    },
    detection: {
      mode: 'angle_rep_hold',
      jointGroup: 'shoulder',
      side: 'right',
      targetMinDeg: 100,
      targetMaxDeg: 150,
      targetRomDeg: 130,
      startRestMaxDeg: 35,
      holdSec: 3,
      defaultSets: 2,
      defaultReps: 6,
      requireStraightElbow: false,
      expectedPlane: 'hand_to_head',
    },
  },
  {
    id: 'STROKE_FOREARM_TURN',
    code: 'STROKE_FOREARM_TURN',
    category: 'upper_limb',
    clinicalFocus: 'stroke_adl',
    name: {
      th: '[Stroke ADL] พลิกหงาย-คว่ำฝ่ามือ (ฝึกถือช้อน/บิดลูกบิดประตู)',
      en: 'Stroke ADL: Forearm Supination & Pronation',
      cn: '卒中康复 ADL：前臂旋前旋后（握勺/转门把手）',
    },
    instruction: {
      th: 'ตั้งศอกข้างลำตัว พลิกหงายฝ่ามือขึ้นค้างไว้ 2.5 วินาที แล้วพลิกคว่ำลง',
      en: 'Tuck elbow at waist, rotate palm face up cleanly for 2.5s, then turn palm face down',
      cn: '肘贴腰侧，掌心向上翻转保持2.5秒后再缓慢向下翻转',
    },
    detection: {
      mode: 'wrist_finger',
      jointGroup: 'wrist',
      side: 'right',
      targetMinDeg: 60,
      targetMaxDeg: 90,
      targetRomDeg: 80,
      startRestMaxDeg: 18,
      holdSec: 2.5,
      defaultSets: 2,
      defaultReps: 8,
      expectedPlane: 'forearm_turn',
    },
  },
  {
    id: 'STROKE_WRIST_FINGER',
    code: 'STROKE_WRIST_FINGER',
    category: 'upper_limb',
    clinicalFocus: 'stroke_adl',
    name: {
      th: '[Stroke ADL] กระดกข้อมือ & กาง-กำนิ้วมือ (ลดเกร็งนิ้วมือ)',
      en: 'Stroke ADL: Wrist Extension & Finger Open/Close',
      cn: '卒中康复 ADL：腕背伸与五指张合减痉',
    },
    instruction: {
      th: 'กระดกข้อมือขึ้นพร้อมกางนิ้วมือทั้ง 5 นิ้วให้สุด ค้างไว้ 2.5 วินาที แล้วกำมือเบาๆ',
      en: 'Extend wrist upward spreading all 5 fingers wide for 2.5s, then softly make a fist',
      cn: '腕部上抬背伸同时用力张开五指保持2.5秒，再轻柔握拳',
    },
    detection: {
      mode: 'wrist_finger',
      jointGroup: 'wrist',
      side: 'right',
      targetMinDeg: 50,
      targetMaxDeg: 85,
      targetRomDeg: 70,
      startRestMaxDeg: 15,
      holdSec: 2.5,
      defaultSets: 2,
      defaultReps: 8,
      expectedPlane: 'wrist_finger',
    },
  },
  {
    id: 'STROKE_THUMB_PINCH',
    code: 'STROKE_THUMB_PINCH',
    category: 'upper_limb',
    clinicalFocus: 'stroke_adl',
    name: {
      th: '[Stroke ADL] จีบนิ้วโป้งแตะปลายนิ้วมือ (ฝึกติดกระดุมเสื้อ/รูดซิป)',
      en: 'Stroke ADL: Thumb-to-Finger Fine Pinch',
      cn: '卒中康复 ADL：拇指对指精细抓捏（系纽扣/拉链）',
    },
    instruction: {
      th: 'แตะปลายนิ้วโป้งเข้ากับปลายนิ้วชี้-กลาง-นาง-ก้อยเรียงตามลำดับแล้วกางออก',
      en: 'Touch thumb tip sequentially to index-middle-ring-pinky fingertips then open cleanly',
      cn: '拇指尖依次对捏食指、中指、无名指与小指尖后再张开',
    },
    detection: {
      mode: 'wrist_finger',
      jointGroup: 'wrist',
      side: 'right',
      targetMinDeg: 45,
      targetMaxDeg: 75,
      targetRomDeg: 60,
      startRestMaxDeg: 14,
      holdSec: 2,
      defaultSets: 2,
      defaultReps: 8,
      expectedPlane: 'thumb_pinch',
    },
  },
  {
    id: 'STROKE_BOBATH_ARM_RAISE',
    code: 'STROKE_BOBATH_ARM_RAISE',
    category: 'upper_limb',
    clinicalFocus: 'stroke_adl',
    name: {
      th: '[Stroke & สธ.] ประสานมือชูขึ้นเหนือศีรษะ (ท่า Bobath พยุงแขนอ่อนแรง)',
      en: 'Stroke & MOPH: Bobath Bilateral Clasped Arm Raise',
      cn: '卒中与老年康复：Bobath 双手交叉上举过头',
    },
    instruction: {
      th: 'ประสานมือใช้แขนข้างปกติช่วยพยุงแขนข้างที่อ่อนแรงชูขึ้นช้าๆ ค้างไว้ 3 วินาที',
      en: 'Clasp fingers together and gently assist the weaker arm upward overhead, hold 3s',
      cn: '双手十指交叉，健侧辅助患侧缓慢高举过头顶保持3秒',
    },
    detection: {
      mode: 'bilateral_raise',
      jointGroup: 'shoulder',
      side: 'both',
      targetMinDeg: 100,
      targetMaxDeg: 160,
      targetRomDeg: 135,
      startRestMaxDeg: 35,
      holdSec: 3,
      defaultSets: 2,
      defaultReps: 6,
      requireStraightElbow: false,
      expectedPlane: 'bilateral',
    },
  },
  {
    id: 'STROKE_BOBATH_FORWARD_PUSH',
    code: 'STROKE_BOBATH_FORWARD_PUSH',
    category: 'upper_limb',
    clinicalFocus: 'stroke_adl',
    name: {
      th: '[Stroke ADL] ประสานมือดันเหยียดแขนไปด้านหน้า (ลดเกร็งพับศอก)',
      en: 'Stroke ADL: Bobath Bilateral Forward Elbow Push',
      cn: '卒中康复 ADL：双手交叉向前伸肘推展（缓解屈肘痉挛）',
    },
    instruction: {
      th: 'ประสานมือระดับอกแล้วดันเหยียดข้อศอกออกไปด้านหน้าตรงๆ ค้างไว้ 3 วินาที',
      en: 'Clasp hands at chest height and extend elbows straight forward, hold 3s',
      cn: '双手胸前交叉握紧，向前伸直双肘推展保持3秒',
    },
    detection: {
      mode: 'bilateral_raise',
      jointGroup: 'elbow',
      side: 'both',
      targetMinDeg: 110,
      targetMaxDeg: 165,
      targetRomDeg: 145,
      startRestMaxDeg: 95,
      holdSec: 3,
      defaultSets: 2,
      defaultReps: 6,
      requireStraightElbow: false,
      expectedPlane: 'bilateral_push',
    },
  },

  // =========================================================================
  // หมวด 2: คอและบ่า (สธ. ผู้สูงอายุ & Stroke Upper Trunk — 3 ท่าหลัก)
  // =========================================================================
  {
    id: 'MOPH_NECK_TILT',
    code: 'MOPH_NECK_TILT',
    category: 'neck_trunk',
    clinicalFocus: 'moph_elderly',
    name: {
      th: '[สธ. ผู้สูงอายุ] ยืดกล้ามเนื้อคอและบ่า เอียงคอช้าๆ',
      en: 'MOPH Elderly: Gentle Lateral Neck Stretch',
      cn: '卫生部老年保健：颈肩缓慢侧倾拉伸',
    },
    instruction: {
      th: 'ค่อยๆ เอียงศีรษะไปด้านข้างช้าๆ เพื่อคลายกล้ามเนื้อบ่าและป้องกันคอตก ค้างไว้ 3 วินาที',
      en: 'Tilt head smoothly toward shoulder to relax neck & trapezius muscles, hold 3s',
      cn: '头部缓慢侧倾放松颈肩肌群，保持3秒',
    },
    detection: {
      mode: 'neck_tilt',
      jointGroup: 'neck',
      side: 'both',
      targetMinDeg: 25,
      targetMaxDeg: 45,
      targetRomDeg: 35,
      startRestMaxDeg: 8,
      holdSec: 3,
      defaultSets: 2,
      defaultReps: 5,
    },
  },
  {
    id: 'MOPH_NECK_TURN',
    code: 'MOPH_NECK_TURN',
    category: 'neck_trunk',
    clinicalFocus: 'moph_elderly',
    name: {
      th: '[Stroke & สธ.] หันศีรษะมองซ้าย-ขวาช้าๆ (กระตุ้นการรับรู้ภาพด้านที่อ่อนแรง)',
      en: 'Stroke & MOPH: Neck Rotation Visual Scanning',
      cn: '卒中与老年康复：缓慢转头视野扫描训练',
    },
    instruction: {
      th: 'หันศีรษะมองด้านข้างช้าๆ ค้าง 2 วินาที เพื่อบริหารคอและลดภาวะละเลยครึ่งซีก',
      en: 'Turn head smoothly sideways holding 2s to improve cervical ROM and visual scanning',
      cn: '头部缓慢向侧方转动保持2秒，改善偏侧视野注意与颈椎活动',
    },
    detection: {
      mode: 'neck_tilt',
      jointGroup: 'neck',
      side: 'both',
      targetMinDeg: 35,
      targetMaxDeg: 65,
      targetRomDeg: 50,
      startRestMaxDeg: 12,
      holdSec: 2,
      defaultSets: 2,
      defaultReps: 6,
      expectedPlane: 'neck_turn',
    },
  },
  {
    id: 'MOPH_SHOULDER_SHRUG_ROLL',
    code: 'MOPH_SHOULDER_SHRUG_ROLL',
    category: 'neck_trunk',
    clinicalFocus: 'moph_elderly',
    name: {
      th: '[สธ. ผู้สูงอายุ] ท่ายักไหล่และหมุนหัวไหล่ผ่อนคลายบ่า',
      en: 'MOPH Elderly: Shoulder Shrug & Scapular Roll',
      cn: '卫生部老年保健：耸肩与绕肩放松训练',
    },
    instruction: {
      th: 'ยกหัวไหล่ขึ้นช้าๆ ค้าง 2 วินาที แล้วหมุนไหล่ลงผ่อนคลาย',
      en: 'Shrug shoulders gently up for 2s then roll back down smoothly to relax upper trapezius',
      cn: '双肩缓慢上耸保持2秒，再向后下沉环绕放松肩颈',
    },
    detection: {
      mode: 'shrug_roll',
      jointGroup: 'shoulder',
      side: 'both',
      targetMinDeg: 30,
      targetMaxDeg: 60,
      targetRomDeg: 45,
      startRestMaxDeg: 12,
      holdSec: 2,
      defaultSets: 2,
      defaultReps: 8,
      expectedPlane: 'shrug_roll',
    },
  },
];

export const STROKE_CANONICAL_EXERCISE_CODES: readonly string[] = CLINICAL_EXERCISES.map((ex) => ex.code);
export const STROKE_CANONICAL_EXERCISE_SET: ReadonlySet<string> = new Set(
  STROKE_CANONICAL_EXERCISE_CODES.map((c) => c.toUpperCase()),
);
export const CLINICAL_MASTER_EXERCISE_MAP: ReadonlyMap<string, ExerciseConfig> = new Map(
  CLINICAL_EXERCISES.map((ex) => [ex.code.toUpperCase(), ex]),
);

const SIDE_SUFFIX_REGEX = /^(.+?)_(LEFT|RIGHT|BOTH|BILATERAL|L|R)$/i;

export function normalizeSideValue(rawSide: string | null | undefined): ExerciseSideCode | null {
  if (!rawSide) return null;
  const up = rawSide.trim().toUpperCase();
  if (up === 'LEFT' || up === 'L') return 'LEFT';
  if (up === 'RIGHT' || up === 'R') return 'RIGHT';
  if (up === 'BOTH' || up === 'BILATERAL') return 'BOTH';
  return null;
}

export function exerciseSideToBodySide(side: ExerciseSideCode): BodySide | 'both' {
  if (side === 'LEFT') return 'left';
  if (side === 'BOTH') return 'both';
  return 'right';
}

/**
 * Resolves any legacy (`STROKE_SH_FLEX_R`, `ex-01`, `MOPH_NECK_TURN_BILATERAL`),
 * effective (`STROKE_SH_FLEX_LEFT`), or unified master code (`STROKE_SH_FLEX`) into
 * `{ baseCode, side, effectiveCode, hasExplicitSide }`.
 */
export function parseExerciseCodeAndSide(
  rawId: string | undefined | null,
  explicitSide?: string | null,
): {
  baseCode: string;
  side: ExerciseSideCode;
  effectiveCode: string;
  hasExplicitSide: boolean;
} {
  const clean = (rawId ?? '').trim();
  const explicit = normalizeSideValue(explicitSide);
  const mappedLegacy = LEGACY_EXERCISE_ID_MAP[clean.toLowerCase()];
  const candidate = (mappedLegacy ?? (clean || 'STROKE_SH_FLEX')).toUpperCase();

  let rawBase = candidate;
  let inferredFromSuffix: ExerciseSideCode | null = null;
  const match = candidate.match(SIDE_SUFFIX_REGEX);
  if (match) {
    rawBase = match[1];
    inferredFromSuffix = normalizeSideValue(match[2]);
  }

  if (rawBase === 'MOPH_NECK_TURN_BILATERAL') rawBase = 'MOPH_NECK_TURN';
  if (rawBase === 'MOPH_BILATERAL_ARM_RAISE') rawBase = 'STROKE_BOBATH_ARM_RAISE';

  const master = CLINICAL_MASTER_EXERCISE_MAP.get(rawBase);
  const baseCode = master ? master.code : rawBase;
  const defaultSide: ExerciseSideCode = master?.detection.side === 'both' ? 'BOTH' : 'RIGHT';
  const side: ExerciseSideCode = explicit ?? inferredFromSuffix ?? defaultSide;

  return {
    baseCode,
    side,
    effectiveCode: `${baseCode}_${side}`,
    hasExplicitSide: Boolean(explicit || inferredFromSuffix),
  };
}

export function extractBaseExerciseCode(rawId: string | undefined | null): string {
  return parseExerciseCodeAndSide(rawId).baseCode;
}

export function formatEffectiveExerciseCode(baseCode: string, side: ExerciseSideCode): string {
  const cleanBase = extractBaseExerciseCode(baseCode);
  return `${cleanBase}_${side}`;
}

export interface ActiveExerciseSideHint {
  exercise_id?: string;
  exerciseId?: string;
  catalog_exercise_id?: string;
  effective_code?: string;
  effectiveCode?: string;
  code?: string;
  exerciseCode?: string;
  prescribed_side?: string | null;
  side?: string | null;
  target_side?: string | null;
  name_th?: string | null;
  name_en?: string | null;
  nameTh?: string | null;
  nameEn?: string | null;
  target_joint?: string | null;
  targetJoint?: string | null;
  expected_plane?: string | null;
}

/**
 * Single canonical side resolver shared across Clinical Kiosk UI, Pose AI solver,
 * SkeletonCanvas overlay, and voice prompt helpers.
 */
export function resolveActiveExerciseSide(
  exercise?: ActiveExerciseSideHint | null,
  scheduleItem?: ActiveExerciseSideHint | null,
  targetJoint?: string | null,
  expectedPlane?: string | null,
): ExerciseSideCode | undefined {
  const explicitSide =
    normalizeSideValue(scheduleItem?.side) ??
    normalizeSideValue(scheduleItem?.prescribed_side) ??
    normalizeSideValue(scheduleItem?.target_side) ??
    normalizeSideValue(exercise?.prescribed_side) ??
    normalizeSideValue(exercise?.side) ??
    normalizeSideValue(exercise?.target_side);
  if (explicitSide) return explicitSide;

  const rawCodes = [
    scheduleItem?.effectiveCode,
    scheduleItem?.effective_code,
    scheduleItem?.exerciseCode,
    scheduleItem?.exerciseId,
    scheduleItem?.exercise_id,
    exercise?.effective_code,
    exercise?.effectiveCode,
    exercise?.code,
    exercise?.exerciseCode,
    exercise?.catalog_exercise_id,
    exercise?.exercise_id,
    exercise?.exerciseId,
  ].filter((v): v is string => typeof v === 'string' && v.trim().length > 0);

  for (const code of rawCodes) {
    const clean = code.trim().toUpperCase();
    const mapped = LEGACY_EXERCISE_ID_MAP[clean.toLowerCase()];
    const candidate = (mapped ?? clean).toUpperCase();
    const match = candidate.match(SIDE_SUFFIX_REGEX);
    if (match) {
      const fromSuffix = normalizeSideValue(match[2]);
      if (fromSuffix) return fromSuffix;
    }
    const master = CLINICAL_MASTER_EXERCISE_MAP.get(candidate);
    if (master?.detection.side === 'both') return 'BOTH';
    if (
      candidate.includes('BOBATH_ARM_RAISE') ||
      candidate.includes('BILATERAL') ||
      candidate.includes('FORWARD_PUSH') ||
      candidate.includes('SHRUG') ||
      candidate.includes('NECK_TURN')
    ) {
      return 'BOTH';
    }
  }

  const effPlane = expectedPlane ?? exercise?.expected_plane ?? scheduleItem?.expected_plane;
  if (effPlane === 'bilateral' || effPlane === 'bilateral_push') {
    return 'BOTH';
  }

  const combinedNames = [
    scheduleItem?.name_th,
    scheduleItem?.nameTh,
    scheduleItem?.name_en,
    scheduleItem?.nameEn,
    exercise?.name_th,
    exercise?.nameTh,
    exercise?.name_en,
    exercise?.nameEn,
  ]
    .filter((v): v is string => typeof v === 'string' && v.trim().length > 0)
    .join(' ')
    .toLowerCase();

  if (
    combinedNames.includes('ทั้งสองข้าง') ||
    combinedNames.includes('ทั้ง 2 ข้าง') ||
    combinedNames.includes('(both)') ||
    combinedNames.includes('both sides') ||
    combinedNames.includes('both arms') ||
    combinedNames.includes('ประสานมือ')
  ) {
    return 'BOTH';
  }
  if (
    combinedNames.includes('ข้างซ้าย') ||
    combinedNames.includes('แขนซ้าย') ||
    combinedNames.includes('(left)')
  ) {
    return 'LEFT';
  }
  if (
    combinedNames.includes('ข้างขวา') ||
    combinedNames.includes('แขนขวา') ||
    combinedNames.includes('(right)')
  ) {
    return 'RIGHT';
  }

  const effJoint =
    targetJoint ??
    exercise?.target_joint ??
    exercise?.targetJoint ??
    scheduleItem?.target_joint ??
    scheduleItem?.targetJoint;
  if (effJoint === 'left_shoulder') return 'LEFT';
  if (effJoint === 'right_shoulder') return 'RIGHT';
  return undefined;
}

export function resolveDefaultSideForExercise(
  ex: { id?: string; code?: string; detection?: { side?: string; expectedPlane?: string; mode?: string } },
  patientAffectedSide?: string | null,
): ExerciseSideCode {
  const affected = normalizeSideValue(patientAffectedSide);
  const base = extractBaseExerciseCode(ex.code || ex.id);
  const master = CLINICAL_MASTER_EXERCISE_MAP.get(base.toUpperCase()) ?? ex;
  const defaultMasterSide = master.detection?.side;
  const isInherentlyBilateral =
    base === 'STROKE_BOBATH_ARM_RAISE' ||
    base === 'STROKE_BOBATH_FORWARD_PUSH' ||
    base === 'MOPH_NECK_TURN' ||
    base === 'MOPH_SHOULDER_SHRUG_ROLL' ||
    master.detection?.mode === 'bilateral_raise' ||
    master.detection?.mode === 'shrug_roll';

  if (isInherentlyBilateral) {
    return 'BOTH';
  }
  if (affected === 'LEFT' || affected === 'RIGHT' || affected === 'BOTH') {
    return affected;
  }
  if (defaultMasterSide === 'both') return 'BOTH';
  if (defaultMasterSide === 'left') return 'LEFT';
  return 'RIGHT';
}

export function resolveSideAdjustedExerciseCopy(
  ex: ExerciseConfig,
  side: ExerciseSideCode,
): {
  nameTh: string;
  nameEn: string;
  nameCn: string;
  instructionTh: string;
  instructionEn: string;
  instructionCn: string;
  sideBadgeTh: string;
  sideBadgeEn: string;
  sideBadgeCn: string;
} {
  const base = extractBaseExerciseCode(ex.code || ex.id);
  const baseTh = ex.name.th;
  const baseEn = ex.name.en;
  const baseCn = ex.name.cn;
  const instTh = ex.instruction.th;
  const instEn = ex.instruction.en;
  const instCn = ex.instruction.cn;

  const sideTagTh =
    side === 'LEFT' ? 'ข้างซ้าย (L)' : side === 'BOTH' ? 'ทั้งสองข้างพร้อมกัน (Both)' : 'ข้างขวา (R)';
  const sideTagEn =
    side === 'LEFT' ? 'Left Side (L)' : side === 'BOTH' ? 'Both Sides Together' : 'Right Side (R)';
  const sideTagCn = side === 'LEFT' ? '左侧 (L)' : side === 'BOTH' ? '双侧同时 (Both)' : '右侧 (R)';

  const cleanWordSideTh = side === 'LEFT' ? 'ซ้าย' : side === 'BOTH' ? 'ทั้งสองข้าง' : 'ขวา';
  const cleanWordSideEn = side === 'LEFT' ? 'Left' : side === 'BOTH' ? 'Both Sides' : 'Right';
  const cleanWordSideCn = side === 'LEFT' ? '左' : side === 'BOTH' ? '双侧' : '右';

  const replaceLimbPh = (text: string, thTarget: string): string => {
    if (text.includes('ซ้าย-ขวา')) {
      return side === 'BOTH' ? text : text.replace('ซ้าย-ขวา', thTarget);
    }
    if (text.includes('แขน') && !text.includes('แขนซ้าย') && !text.includes('แขนขวา') && !text.includes('แขนทั้งสอง')) {
      return text.replace('แขน', `แขน${thTarget}`);
    }
    if (text.includes('มือ') && !text.includes('มือซ้าย') && !text.includes('มือขวา') && !text.includes('มือทั้งสอง')) {
      return text.replace('มือ', `มือ${thTarget}`);
    }
    if (text.includes('ศอก') && !text.includes('ศอกซ้าย') && !text.includes('ศอกขวา') && !text.includes('ศอกทั้งสอง')) {
      return text.replace('ศอก', `ศอก${thTarget}`);
    }
    if (text.includes('ศีรษะ') && base.startsWith('MOPH_NECK') && side !== 'BOTH') {
      return text.replace('ด้านข้าง', `ด้าน${thTarget}`);
    }
    return `${text} (${sideTagTh})`;
  };

  return {
    nameTh: `${baseTh} — ${sideTagTh}`,
    nameEn: `${baseEn} (${sideTagEn})`,
    nameCn: `${baseCn}（${sideTagCn}）`,
    instructionTh:
      side === 'BOTH'
        ? `${replaceLimbPh(instTh, cleanWordSideTh)} (ทำพร้อมกันทั้งสองข้าง)`
        : replaceLimbPh(instTh, cleanWordSideTh),
    instructionEn:
      side === 'BOTH'
        ? `${instEn} (Perform with ${cleanWordSideEn} together)`
        : `${instEn} (${cleanWordSideEn} side)`,
    instructionCn: `${instCn}（${cleanWordSideCn}）`,
    sideBadgeTh: sideTagTh,
    sideBadgeEn: sideTagEn,
    sideBadgeCn: sideTagCn,
  };
}

/**
 * Collapses split `_R` / `_L` / legacy database exercise rows into the 15 Unified Master Cards
 * so admin schedule pickers and the library never show duplicate left/right cards.
 */
export function mergeDuplicateSideExercises<T extends ExerciseConfig>(rawExercises: readonly T[]): T[] {
  const byCode = new Map<string, T>();
  for (const canonical of CLINICAL_EXERCISES) {
    byCode.set(canonical.code.toUpperCase(), canonical as T);
  }
  const customMap = new Map<string, T>();

  for (const raw of rawExercises) {
    if (!raw || isDeprecatedExerciseRecord(raw)) continue;
    const { baseCode } = parseExerciseCodeAndSide(raw.code || raw.id);
    const upBase = baseCode.toUpperCase();
    if (CLINICAL_MASTER_EXERCISE_MAP.has(upBase)) {
      continue;
    }
    if (!customMap.has(upBase)) {
      customMap.set(upBase, {
        ...raw,
        id: baseCode,
        code: baseCode,
      });
    }
  }

  return [...Array.from(byCode.values()), ...Array.from(customMap.values())];
}

export function normalizeStrokeExerciseId(rawId: string | undefined | null): string {
  const clean = (rawId ?? '').trim();
  if (!clean) return 'STROKE_SH_FLEX';
  const up = clean.toUpperCase();
  if (STROKE_CANONICAL_EXERCISE_SET.has(up)) {
    return up;
  }
  if (LEGACY_EXERCISE_ID_MAP[clean.toLowerCase()]) {
    return LEGACY_EXERCISE_ID_MAP[clean.toLowerCase()];
  }
  const match = up.match(SIDE_SUFFIX_REGEX);
  if (match && STROKE_CANONICAL_EXERCISE_SET.has(match[1])) {
    const suffix = match[2].toUpperCase();
    if (suffix === 'R' || suffix === 'L' || suffix === 'LEFT' || suffix === 'RIGHT' || suffix === 'BOTH') {
      return `${match[1]}_${suffix}`;
    }
    return match[1];
  }
  return clean;
}

export function isDeprecatedExerciseRecord(ex: { id?: string; code?: string; category?: string }): boolean {
  const codeUp = (ex.code ?? '').trim().toUpperCase();
  const idLo = (ex.id ?? '').trim().toLowerCase();
  if (DEPRECATED_EXERCISE_CODE_SET.has(codeUp)) return true;
  if (/^ex-(0?[1-9]|[1-3][0-9]|40)$/.test(idLo)) return true;
  if (ex.category === 'lower_limb' || ex.category === 'balance') return true;
  return false;
}

export function sortExercisesStrokeUpperFirst<T extends ExerciseConfig>(exercises: ReadonlyArray<T>): T[] {
  const canonicalOrder = new Map<string, number>(
    STROKE_CANONICAL_EXERCISE_CODES.map((code, idx) => [code, idx]),
  );
  const rank = (ex: T): number => {
    const { baseCode } = parseExerciseCodeAndSide(ex.code || ex.id);
    const idx = canonicalOrder.get(baseCode) ?? canonicalOrder.get(ex.code) ?? canonicalOrder.get(ex.id);
    if (typeof idx === 'number') return idx;
    if (ex.clinicalFocus === 'stroke_adl') return 100;
    if (ex.clinicalFocus === 'moph_elderly') return 200;
    if (ex.category === 'upper_limb' || ex.category === 'neck_trunk') return 300;
    return 900;
  };
  return [...exercises].sort((a, b) => rank(a) - rank(b));
}
