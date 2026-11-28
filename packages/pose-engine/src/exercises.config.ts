import type { BodySide, ExerciseConfig } from './types';

export type ExerciseSideCode = 'LEFT' | 'RIGHT' | 'BOTH';

/**
 * Legacy / split-side exercise codes permanently retired from Dr.Tech.Care.
 * Canonical library is the 15 Unified Master Exercises (`ELDERLY_*` & `MOPH_*`) with dynamic L/R/BOTH side selection.
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
  // Orthopedic postures unsuitable for front-facing kiosk elderly therapy
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
  'ELDERLY_SH_FLEX_R',
  'ELDERLY_SH_FLEX_L',
  'ELDERLY_SH_ABD_R',
  'ELDERLY_SH_ABD_L',
  'ELDERLY_CROSS_BODY_R',
  'ELDERLY_CROSS_BODY_L',
  'ELDERLY_SH_ROT_R',
  'ELDERLY_SH_ROT_L',
  'ELDERLY_SCAP_PROTRACT_R',
  'ELDERLY_SCAP_PROTRACT_L',
  'ELDERLY_ELB_FLEX_EXT_R',
  'ELDERLY_ELB_FLEX_EXT_L',
  'ELDERLY_HAND_TO_HEAD_R',
  'ELDERLY_HAND_TO_HEAD_L',
  'ELDERLY_FOREARM_TURN_R',
  'ELDERLY_FOREARM_TURN_L',
  'ELDERLY_WRIST_FINGER_R',
  'ELDERLY_WRIST_FINGER_L',
  'ELDERLY_THUMB_PINCH_R',
  'ELDERLY_THUMB_PINCH_L',
  'MOPH_NECK_TILT_R',
  'MOPH_NECK_TILT_L',
  'MOPH_NECK_TURN_BILATERAL',
] as const;

export const DEPRECATED_EXERCISE_CODE_SET: ReadonlySet<string> = new Set(
  DEPRECATED_EXERCISE_CODES.map((c) => c.toUpperCase()),
);

/** Legacy `ex-01`..`ex-40` and old `_R`/`_L` codes mapped cleanly to unified master + side info. */
export const LEGACY_EXERCISE_ID_MAP: Readonly<Record<string, string>> = {
  'ex-01': 'ELDERLY_SH_FLEX_RIGHT',
  'ex-02': 'ELDERLY_SH_FLEX_LEFT',
  'ex-03': 'ELDERLY_SH_ABD_RIGHT',
  'ex-04': 'ELDERLY_SH_ABD_LEFT',
  'ex-05': 'ELDERLY_ELB_FLEX_EXT_RIGHT',
  'ex-06': 'ELDERLY_ELB_FLEX_EXT_LEFT',
  'ex-07': 'ELDERLY_SH_ROT_RIGHT',
  'ex-08': 'ELDERLY_SH_ROT_LEFT',
  'ex-09': 'ELDERLY_SCAP_PROTRACT_RIGHT',
  'ex-10': 'ELDERLY_SCAP_PROTRACT_LEFT',
  'ex-11': 'ELDERLY_CLASPED_ARM_RAISE',
  'ex-12': 'ELDERLY_CLASPED_ARM_RAISE',
  'ex-19': 'MOPH_NECK_TILT_RIGHT',
  'ex-20': 'MOPH_NECK_TILT_LEFT',
  'ex-23': 'MOPH_NECK_TURN',
  'ex-24': 'ELDERLY_HAND_TO_HEAD_RIGHT',
  'ex-25': 'ELDERLY_HAND_TO_HEAD_LEFT',
  'ex-27': 'ELDERLY_SH_FLEX_RIGHT',
  'ex-28': 'ELDERLY_SH_FLEX_LEFT',
  'ex-29': 'ELDERLY_SH_ABD_RIGHT',
  'ex-30': 'ELDERLY_SH_ABD_LEFT',
  'ex-31': 'ELDERLY_ELB_FLEX_EXT_RIGHT',
  'ex-32': 'ELDERLY_ELB_FLEX_EXT_LEFT',
  'ex-33': 'ELDERLY_SH_ROT_RIGHT',
  'ex-34': 'ELDERLY_SH_ROT_LEFT',
  'ex-35': 'ELDERLY_WRIST_FINGER_RIGHT',
  'ex-36': 'ELDERLY_WRIST_FINGER_LEFT',
  'ex-37': 'MOPH_NECK_TILT_RIGHT',
  'ex-38': 'MOPH_NECK_TILT_LEFT',
  'ex-39': 'MOPH_SHOULDER_SHRUG_ROLL',
  'ex-40': 'ELDERLY_CLASPED_ARM_RAISE',
  moph_neck_turn_bilateral: 'MOPH_NECK_TURN',
  moph_bilateral_arm_raise: 'ELDERLY_CLASPED_ARM_RAISE',
  right_shoulder_flexion: 'ELDERLY_SH_FLEX_RIGHT',
  left_shoulder_flexion: 'ELDERLY_SH_FLEX_LEFT',
  right_shoulder_abduction: 'ELDERLY_SH_ABD_RIGHT',
  left_shoulder_abduction: 'ELDERLY_SH_ABD_LEFT',
  elbow_flexion_right: 'ELDERLY_ELB_FLEX_EXT_RIGHT',
  elbow_flexion_left: 'ELDERLY_ELB_FLEX_EXT_LEFT',
};

/**
 * Unified 15 Clinical Master Rehabilitation Exercises (Elderly Care & MOPH):
 * - Category 1 (12): แขน ไหล่ ศอก และมือ (กายภาพบำบัดผู้สูงอายุ — 12 ท่าหลัก)
 * - Category 2 (3): คอและบ่า (สธ. ผู้สูงอายุ — 3 ท่าหลัก)
 * Every posture supports side selection (`LEFT` / `RIGHT` / `BOTH`) within a single card.
 */
export const CLINICAL_EXERCISES: readonly ExerciseConfig[] = [
  // =========================================================================
  // หมวด 1: แขน ไหล่ ศอก และมือ (กายภาพบำบัดผู้สูงอายุ — 12 ท่าหลัก)
  // =========================================================================
  {
    id: 'ELDERLY_SH_FLEX',
    code: 'ELDERLY_SH_FLEX',
    category: 'upper_limb',
    clinicalFocus: 'elderly_adl',
    name: {
      th: '[ผู้สูงอายุ] ยกแขนไปด้านหน้า (ฝึกเอื้อมหยิบของ/สวมเสื้อ)',
      en: 'Elderly Care: Shoulder Forward Flexion',
      cn: '老年康复训练：肩前屈上举（伸手取物/穿衣）',
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
      targetMaxDeg: 180,
      targetRomDeg: 180,
      startRestMaxDeg: 30,
      holdSec: 3,
      defaultSets: 2,
      defaultReps: 6,
      requireStraightElbow: false,
      expectedPlane: 'flexion',
    },
  },
  {
    id: 'ELDERLY_SH_ABD',
    code: 'ELDERLY_SH_ABD',
    category: 'upper_limb',
    clinicalFocus: 'elderly_adl',
    name: {
      th: '[ผู้สูงอายุ] กางแขนด้านข้าง (ฝึกกางแขนสวมเสื้อ/ลดไหล่ติด)',
      en: 'Elderly Care: Shoulder Abduction',
      cn: '老年康复训练：肩外展上举（预防肩周僵硬）',
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
      targetMaxDeg: 180,
      targetRomDeg: 180,
      startRestMaxDeg: 30,
      holdSec: 3,
      defaultSets: 2,
      defaultReps: 6,
      requireStraightElbow: false,
      expectedPlane: 'abduction',
    },
  },
  {
    id: 'ELDERLY_CROSS_BODY',
    code: 'ELDERLY_CROSS_BODY',
    category: 'upper_limb',
    clinicalFocus: 'elderly_adl',
    name: {
      th: '[ผู้สูงอายุ] เอื้อมมือแตะไหล่ฝั่งตรงข้าม (ฝึกยืดสะบัก/สวมเสื้อคลุม)',
      en: 'Elderly Care: Cross-Body Shoulder Reach',
      cn: '老年康复训练：横越胸前触碰对侧肩',
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
      targetMaxDeg: 180,
      targetRomDeg: 180,
      startRestMaxDeg: 25,
      holdSec: 3,
      defaultSets: 2,
      defaultReps: 6,
      requireStraightElbow: false,
      expectedPlane: 'cross_body',
    },
  },
  {
    id: 'ELDERLY_SH_ROT',
    code: 'ELDERLY_SH_ROT',
    category: 'upper_limb',
    clinicalFocus: 'elderly_adl',
    name: {
      th: '[ผู้สูงอายุ] หมุนข้อไหล่เข้า-ออก (ฝึกอ้อมสวมเสื้อ/ติดกระดุม)',
      en: 'Elderly Care: Shoulder External/Internal Rotation',
      cn: '老年康复训练：肩关节内外旋（日常穿衣与扣纽扣）',
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
    id: 'ELDERLY_SCAP_PROTRACT',
    code: 'ELDERLY_SCAP_PROTRACT',
    category: 'upper_limb',
    clinicalFocus: 'elderly_adl',
    name: {
      th: '[ผู้สูงอายุ] เหยียดแขนดันไปด้านหน้าระดับอก (เสริมความมั่นคงข้อไหล่)',
      en: 'Elderly Care: Scapular Protraction Forward Reach',
      cn: '老年康复训练：胸前伸直前推（增强肩关节稳定）',
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
      targetMaxDeg: 180,
      targetRomDeg: 180,
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
    id: 'ELDERLY_ELB_FLEX_EXT',
    code: 'ELDERLY_ELB_FLEX_EXT',
    category: 'upper_limb',
    clinicalFocus: 'elderly_adl',
    name: {
      th: '[ผู้สูงอายุ] งอ-เหยียดข้อศอก (ฝึกตักอาหาร/ดึงแขนเสื้อ)',
      en: 'Elderly Care: Elbow Flexion & Extension',
      cn: '老年康复训练：肘关节屈伸（进食与穿衣训练）',
    },
    instruction: {
      th: 'งอข้อศอกพับมือเข้าหาไหล่/ปากช้าๆ ค้างไว้ 2 วินาที แล้วเหยียดตรง',
      en: 'Flex elbow smoothly toward shoulder/mouth, hold 2s, then fully extend arm straight',
      cn: '缓慢屈曲手肘靠近肩部保持2秒，再充分伸直放松肌肉',
    },
    detection: {
      mode: 'angle_rep_hold',
      jointGroup: 'elbow',
      side: 'right',
      targetMinDeg: 95,
      targetMaxDeg: 180,
      targetRomDeg: 180,
      startRestMaxDeg: 40,
      holdSec: 2,
      defaultSets: 2,
      defaultReps: 8,
    },
  },
  {
    id: 'ELDERLY_HAND_TO_HEAD',
    code: 'ELDERLY_HAND_TO_HEAD',
    category: 'upper_limb',
    clinicalFocus: 'elderly_adl',
    name: {
      th: '[ผู้สูงอายุ] ยกมือแตะศีรษะ (ฝึกหวีผม/สวมเสื้อทางศีรษะ)',
      en: 'Elderly Care: Hand-to-Head Grooming Reach',
      cn: '老年康复训练：上举触头顶（梳头与日常洗护）',
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
      targetMaxDeg: 180,
      targetRomDeg: 180,
      startRestMaxDeg: 35,
      holdSec: 3,
      defaultSets: 2,
      defaultReps: 6,
      requireStraightElbow: false,
      expectedPlane: 'hand_to_head',
    },
  },
  {
    id: 'ELDERLY_FOREARM_TURN',
    code: 'ELDERLY_FOREARM_TURN',
    category: 'upper_limb',
    clinicalFocus: 'elderly_adl',
    name: {
      th: '[ผู้สูงอายุ] พลิกหงาย-คว่ำฝ่ามือ (ฝึกถือช้อน/บิดลูกบิดประตู)',
      en: 'Elderly Care: Forearm Supination & Pronation',
      cn: '老年康复训练：前臂旋前旋后（握勺与开门训练）',
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
    id: 'ELDERLY_WRIST_FINGER',
    code: 'ELDERLY_WRIST_FINGER',
    category: 'upper_limb',
    clinicalFocus: 'elderly_adl',
    name: {
      th: '[ผู้สูงอายุ] กระดกข้อมือ & กาง-กำนิ้วมือ (บริหารข้อมือและนิ้วมือ)',
      en: 'Elderly Care: Wrist Extension & Finger Open/Close',
      cn: '老年康复训练：腕背伸与五指张合灵活训练',
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
    id: 'ELDERLY_THUMB_PINCH',
    code: 'ELDERLY_THUMB_PINCH',
    category: 'upper_limb',
    clinicalFocus: 'elderly_adl',
    name: {
      th: '[ผู้สูงอายุ] จีบนิ้วโป้งแตะปลายนิ้วมือ (ฝึกติดกระดุม/หยิบจับสิ่งของ)',
      en: 'Elderly Care: Thumb-to-Finger Fine Pinch',
      cn: '老年康复训练：拇指对指精细抓捏（系纽扣与抓握）',
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
    id: 'ELDERLY_CLASPED_ARM_RAISE',
    code: 'ELDERLY_CLASPED_ARM_RAISE',
    category: 'upper_limb',
    clinicalFocus: 'elderly_adl',
    name: {
      th: '[สธ. ผู้สูงอายุ] ประสานมือชูขึ้นเหนือศีรษะ (ยืดเหยียดแขนและไหล่สองข้าง)',
      en: 'MOPH Elderly: Bilateral Clasped Overhead Arm Raise',
      cn: '老年康复训练：双手交叉上举过头（双侧肩臂伸展）',
    },
    instruction: {
      th: 'ประสานนิ้วมือทั้งสองข้างแล้วค่อยๆ ชูแขนขึ้นเหนือศีรษะพร้อมกันช้าๆ ค้างไว้ 3 วินาที',
      en: 'Clasp both hands together and raise arms gently overhead in unison, hold 3s',
      cn: '双手十指交叉，缓慢平稳向上高举过头顶保持3秒',
    },
    detection: {
      mode: 'bilateral_raise',
      jointGroup: 'shoulder',
      side: 'both',
      targetMinDeg: 100,
      targetMaxDeg: 180,
      targetRomDeg: 180,
      startRestMaxDeg: 35,
      holdSec: 3,
      defaultSets: 2,
      defaultReps: 6,
      requireStraightElbow: false,
      expectedPlane: 'bilateral',
    },
  },
  {
    id: 'ELDERLY_CLASPED_FORWARD_PUSH',
    code: 'ELDERLY_CLASPED_FORWARD_PUSH',
    category: 'upper_limb',
    clinicalFocus: 'elderly_adl',
    name: {
      th: '[ผู้สูงอายุ] ประสานมือดันเหยียดแขนไปด้านหน้า (คลายตึงแขนและข้อศอก)',
      en: 'Elderly Care: Bilateral Clasped Forward Elbow Push',
      cn: '老年康复训练：双手交叉向前伸肘推展（缓解上肢僵硬）',
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
      targetMaxDeg: 180,
      targetRomDeg: 180,
      startRestMaxDeg: 95,
      holdSec: 3,
      defaultSets: 2,
      defaultReps: 6,
      requireStraightElbow: false,
      expectedPlane: 'bilateral_push',
    },
  },

  // =========================================================================
  // หมวด 2: คอและบ่า (สธ. ผู้สูงอายุ — 3 ท่าหลัก)
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
      th: '[สธ. ผู้สูงอายุ] หันศีรษะมองซ้าย-ขวาช้าๆ (บริหารคอและเพิ่มมุมมองรอบตัว)',
      en: 'MOPH Elderly: Gentle Neck Rotation Mobility',
      cn: '卫生部老年保健：缓慢左右转头颈部灵活训练',
    },
    instruction: {
      th: 'หันศีรษะมองด้านข้างช้าๆ ค้าง 2 วินาที เพื่อบริหารข้อต่อคอและลดอาการตึงคอ',
      en: 'Turn head smoothly sideways holding 2s to improve cervical ROM and reduce stiffness',
      cn: '头部缓慢向侧方转动保持2秒，舒缓颈部僵硬并提升转头灵活度',
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

export const ELDERLY_CANONICAL_EXERCISE_CODES: readonly string[] = CLINICAL_EXERCISES.map((ex) => ex.code);
export const ELDERLY_CANONICAL_EXERCISE_SET: ReadonlySet<string> = new Set(
  ELDERLY_CANONICAL_EXERCISE_CODES.map((c) => c.toUpperCase()),
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
 * Resolves any legacy (`ELDERLY_SH_FLEX_R`, `ex-01`, `MOPH_NECK_TURN_BILATERAL`),
 * effective (`ELDERLY_SH_FLEX_LEFT`), or unified master code (`ELDERLY_SH_FLEX`) into
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
  const candidate = (mappedLegacy ?? (clean || 'ELDERLY_SH_FLEX')).toUpperCase();

  let rawBase = candidate;
  let inferredFromSuffix: ExerciseSideCode | null = null;
  const match = candidate.match(SIDE_SUFFIX_REGEX);
  if (match) {
    rawBase = match[1];
    inferredFromSuffix = normalizeSideValue(match[2]);
  }

  if (rawBase === 'MOPH_NECK_TURN_BILATERAL') rawBase = 'MOPH_NECK_TURN';
  if (rawBase === 'MOPH_BILATERAL_ARM_RAISE') rawBase = 'ELDERLY_CLASPED_ARM_RAISE';
  if (!CLINICAL_MASTER_EXERCISE_MAP.has(rawBase)) {
    const legacySuffixMatch = rawBase.match(
      /^[A-Z]+_(SH_FLEX|SH_ABD|CROSS_BODY|SH_ROT|SCAP_PROTRACT|ELB_FLEX_EXT|HAND_TO_HEAD|FOREARM_TURN|WRIST_FINGER|THUMB_PINCH)$/,
    );
    if (legacySuffixMatch) {
      rawBase = `ELDERLY_${legacySuffixMatch[1]}`;
    } else if (rawBase.endsWith('_ARM_RAISE')) {
      rawBase = 'ELDERLY_CLASPED_ARM_RAISE';
    } else if (rawBase.endsWith('_FORWARD_PUSH')) {
      rawBase = 'ELDERLY_CLASPED_FORWARD_PUSH';
    }
  }

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
      candidate.includes('CLASPED_ARM_RAISE') ||
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
    base === 'ELDERLY_CLASPED_ARM_RAISE' ||
    base === 'ELDERLY_CLASPED_FORWARD_PUSH' ||
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

export function normalizeElderlyExerciseId(rawId: string | undefined | null): string {
  const clean = (rawId ?? '').trim();
  if (!clean) return 'ELDERLY_SH_FLEX';
  const up = clean.toUpperCase();
  if (ELDERLY_CANONICAL_EXERCISE_SET.has(up)) {
    return up;
  }
  if (LEGACY_EXERCISE_ID_MAP[clean.toLowerCase()]) {
    return LEGACY_EXERCISE_ID_MAP[clean.toLowerCase()];
  }
  const match = up.match(SIDE_SUFFIX_REGEX);
  if (match && ELDERLY_CANONICAL_EXERCISE_SET.has(match[1])) {
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
  if (/_(R|L)$/i.test(codeUp) && !ELDERLY_CANONICAL_EXERCISE_SET.has(codeUp)) return true;
  if (ex.category === 'lower_limb' || ex.category === 'balance') return true;
  return false;
}

export function sortExercisesElderlyUpperFirst<T extends ExerciseConfig>(exercises: ReadonlyArray<T>): T[] {
  const canonicalOrder = new Map<string, number>(
    ELDERLY_CANONICAL_EXERCISE_CODES.map((code, idx) => [code, idx]),
  );
  const rank = (ex: T): number => {
    const { baseCode } = parseExerciseCodeAndSide(ex.code || ex.id);
    const idx = canonicalOrder.get(baseCode) ?? canonicalOrder.get(ex.code) ?? canonicalOrder.get(ex.id);
    if (typeof idx === 'number') return idx;
    if (ex.clinicalFocus === 'elderly_adl') return 100;
    if (ex.clinicalFocus === 'moph_elderly') return 200;
    if (ex.category === 'upper_limb' || ex.category === 'neck_trunk') return 300;
    return 900;
  };
  return [...exercises].sort((a, b) => rank(a) - rank(b));
}
