import React, { useEffect, useMemo, useState } from 'react';
import {
  CLINICAL_EXERCISES,
  parseExerciseCodeAndSide,
  type ExerciseConfig,
} from '@drtechcares/pose-engine';
import { useClinicalI18n, type ClinicalLang } from '@drtechcares/clinical-i18n';
import { CheckCircle2, Pause, Play, ShieldCheck, Sparkles, Target } from 'lucide-react';
import type { ExercisePrescriptionOption } from '../hooks/useKioskSocket';

export type PostureIllustrationKind =
  | 'shoulder_flexion'
  | 'shoulder_abduction'
  | 'shoulder_scaption'
  | 'shoulder_overhead'
  | 'shoulder_rotation'
  | 'cross_body_reach'
  | 'scapular_protract'
  | 'shoulder_shrug_roll'
  | 'bilateral_arm_raise'
  | 'bobath_forward_push'
  | 'elbow_flexion'
  | 'hand_to_head'
  | 'forearm_turn'
  | 'wrist_finger'
  | 'thumb_pinch'
  | 'neck_tilt'
  | 'neck_turn'
  | 'trunk_align';

export type PostureSide = 'right' | 'left' | 'both';
export type PosturePreviewPhase = 'auto' | 'prepare' | 'move' | 'hold';

export interface ExercisePostureSpec {
  kind: PostureIllustrationKind;
  side: PostureSide;
  clinicalFocus?: ExerciseConfig['clinicalFocus'];
  bodyRegionLabel: string;
  viewAngleLabel: string;
  sideBadgeLabel: string;
  muscleFocus: string;
  adlGoal?: string;
  cautionCue: string;
  steps: [string, string, string];
  primaryColor: string;
  softBgClass: string;
  borderClass: string;
}

const REGION_COPY: Record<
  PostureIllustrationKind,
  {
    region: Record<ClinicalLang, string>;
    view: Record<ClinicalLang, string>;
    muscle: Record<ClinicalLang, string>;
    caution: Record<ClinicalLang, string>;
    adl?: Record<ClinicalLang, string>;
    steps: (side: PostureSide, rom: number, holdSec: number) => Record<ClinicalLang, [string, string, string]>;
  }
> = {
  shoulder_flexion: {
    region: {
      th: 'ข้อไหล่ด้านหน้า • แขนช่วงบน',
      en: 'Anterior Shoulder • Upper Arm',
      cn: '肩部前侧 • 上臂',
    },
    view: {
      th: 'มุมมองด้านข้าง (Sagittal • ยกไปด้านหน้า)',
      en: 'Side profile (Sagittal • Forward raise)',
      cn: '侧面视角（矢状面 • 向前平举）',
    },
    muscle: {
      th: 'ฝึกแรงยกข้อไหล่ด้านหน้า เพื่อการเอื้อมหยิบและสอดแขนเสื้อ',
      en: 'Anterior deltoid control for reaching & sleeve dressing',
      cn: '训练肩前屈肌力，辅助伸手取物与穿衣',
    },
    caution: {
      th: 'หลังตั้งตรง อย่าแอ่นหลังหรือยักไหล่ขึ้นตามแขน',
      en: 'Keep trunk upright; avoid leaning back or hiking shoulder',
      cn: '保持躯干挺直，切勿后仰或耸肩代偿',
    },
    adl: {
      th: '👕 กิจวัตรประจำวัน: เอื้อมหยิบ / สอดแขนเสื้อ',
      en: '👕 Daily ADL: Forward reach / Putting arm through sleeve',
      cn: '👕 日常活动：前伸取物 / 穿套袖',
    },
    steps: (side, rom, holdSec) => {
      const sTh = side === 'left' ? 'ซ้าย' : side === 'both' ? 'ทั้งสองข้าง' : 'ขวา';
      const sEn = side === 'left' ? 'left' : side === 'both' ? 'both sides' : 'right';
      const sCn = side === 'left' ? '左' : side === 'both' ? '双侧' : '右';
      return {
        th: [
          `ยืนหรือนั่งตัวตรง ปล่อยแขน${sTh}ห้อยลงแนวดิ่งข้างลำตัว (0°)`,
          `ยกแขน${sTh}ตรงไปด้านหน้าขึ้นเหนือระดับไหล่จนถึง ${rom}°`,
          `ค้างท่าเป้าหมายไว้ ${holdSec} วินาที แล้วค่อยๆ ลดแขนลง`,
        ],
        en: [
          `Stand or sit upright with ${sEn} arm hanging down vertically (0°)`,
          `Raise your ${sEn} arm smoothly forward and upward toward ${rom}°`,
          `Hold steady at peak for ${holdSec}s, then lower gently`,
        ],
        cn: [
          `挺直站立或坐姿，${sCn}臂自然垂放体侧（0°）`,
          `缓慢将${sCn}臂向前上方平举至约 ${rom}°`,
          `在目标角度保持 ${holdSec} 秒，然后缓慢放下`,
        ],
      };
    },
  },
  shoulder_abduction: {
    region: {
      th: 'ข้อไหล่ด้านข้าง • กล้ามเนื้อหัวไหล่',
      en: 'Lateral Shoulder • Deltoid',
      cn: '肩部外侧 • 三角肌',
    },
    view: {
      th: 'มุมมองหน้าตรง (Coronal • กางออกด้านข้าง)',
      en: 'Front coronal view (Outward to side)',
      cn: '正面冠状位视角（向体侧外展）',
    },
    muscle: {
      th: 'ฝึกกางแขนออกด้านข้าง เพื่อใส่เสื้อคลุมและหวีผม',
      en: 'Lateral shoulder abduction for dressing & grooming',
      cn: '训练肩外展肌群，改善穿衣与抬手动作',
    },
    caution: {
      th: 'กางออกด้านข้างตรงๆ ห้ามเอียงลำตัวไปฝั่งตรงข้าม',
      en: 'Raise outward in side plane without leaning trunk sideways',
      cn: '沿体侧外展，避免向对侧倾斜借力',
    },
    adl: {
      th: '🧥 กิจวัตรประจำวัน: กางแขนสวมเสื้อเชิ้ต/เสื้อคลุม',
      en: '🧥 Daily ADL: Spreading arm into shirt or jacket',
      cn: '🧥 日常活动：张臂穿外套',
    },
    steps: (side, rom, holdSec) => {
      const sTh = side === 'left' ? 'ซ้าย' : side === 'both' ? 'ทั้งสองข้าง' : 'ขวา';
      const sEn = side === 'left' ? 'left' : side === 'both' ? 'both sides' : 'right';
      const sCn = side === 'left' ? '左' : side === 'both' ? '双侧' : '右';
      return {
        th: [
          `ยืนตรงหน้ากล้อง ปล่อยแขน${sTh}ห้อยข้างลำตัว (0°)`,
          `ค่อยๆ กางแขน${sTh}ออกด้านข้างขึ้นเป็นวงโค้งถึง ${rom}°`,
          `ค้างไว้ด้านข้าง ${holdSec} วินาที แล้วลดแขนลงอย่างนุ่มนวล`,
        ],
        en: [
          `Face camera upright with ${sEn} arm hanging at your side (0°)`,
          `Smoothly abduct your ${sEn} arm outward along arc toward ${rom}°`,
          `Hold steady for ${holdSec}s, then return smoothly`,
        ],
        cn: [
          `正对摄像头站直，${sCn}臂自然垂放体侧（0°）`,
          `缓慢将${sCn}臂向体侧外展抬升至约 ${rom}°`,
          `保持 ${holdSec} 秒后平缓收回体侧`,
        ],
      };
    },
  },
  shoulder_scaption: {
    region: {
      th: 'ข้อไหล่แนวสะบักเฉียง 45°',
      en: 'Scapular Plane (45° Diagonal)',
      cn: '肩胛平面 45° 斜上举',
    },
    view: {
      th: 'มุมมองหน้าตรง (ยกเฉียงออก 45°)',
      en: 'Front view (45° diagonal elevation)',
      cn: '正面视角（斜前方 45° 抬举）',
    },
    muscle: {
      th: 'ลดแรงกดทับเส้นเอ็นหัวไหล่ขณะยกแขนมุมสูง',
      en: 'Safe scapular-plane elevation protecting rotator cuff',
      cn: '沿肩胛平面抬举，减少肩袖肌腱挤压',
    },
    caution: {
      th: 'นิ้วหัวแม่มือชี้ขึ้นด้านบนและเหยียดศอกตรงตลอดท่า',
      en: 'Keep thumb pointing up and elbow comfortably straight',
      cn: '拇指朝上，保持手肘自然伸直',
    },
    steps: (side, rom, holdSec) => {
      const sTh = side === 'left' ? 'ซ้าย' : side === 'both' ? 'ทั้งสองข้าง' : 'ขวา';
      const sEn = side === 'left' ? 'left' : side === 'both' ? 'both sides' : 'right';
      const sCn = side === 'left' ? '左' : side === 'both' ? '双侧' : '右';
      return {
        th: [
          `ตั้งลำตัวตรง ปล่อยแขน${sTh}ลงข้างลำตัว (0°)`,
          `ยกแขน${sTh}เฉียงออกด้านหน้าข้าง 45° ถึงมุม ${rom}°`,
          `ค้างท่าที่มุมเป้าหมาย ${holdSec} วินาที แล้วค่อยๆ วางลง`,
        ],
        en: [
          `Stand tall with ${sEn} elbow straight by your side (0°)`,
          `Lift ${sEn} arm diagonally 45° forward-outward to ${rom}°`,
          `Hold for ${holdSec}s at top, then lower with control`,
        ],
        cn: [
          `身体挺直，${sCn}臂自然垂放体侧（0°）`,
          `沿斜前方 45° 角抬起${sCn}臂至 ${rom}°`,
          `保持 ${holdSec} 秒后缓慢放下`,
        ],
      };
    },
  },
  shoulder_overhead: {
    region: {
      th: 'ข้อไหล่ช่วงบนสุด • กล้ามเนื้อสะบัก',
      en: 'Overhead Shoulder • Scapular Upward Reach',
      cn: '肩部高举 • 肩胛上回旋',
    },
    view: {
      th: 'มุมมองหน้าตรง (ชูแขนเหนือศีรษะ)',
      en: 'Front view (Overhead vertical reach)',
      cn: '正面视角（垂直上举过头顶）',
    },
    muscle: {
      th: 'เพิ่มองศาการยืดสุดของหัวไหล่เพื่อหยิบของบนชั้นสูง',
      en: 'Full overhead range for high-shelf reach & dressing',
      cn: '恢复高处取物所需的充分肩部上举幅度',
    },
    caution: {
      th: 'หยุดยกทันทีหากเจ็บจี๊ดข้อไหล่ และอย่าแอ่นเอวช่วย',
      en: 'Stay within pain-free range; do not arch lower back',
      cn: '量力而行，切勿拱腰借力',
    },
    steps: (side, rom, holdSec) => {
      const sTh = side === 'left' ? 'ซ้าย' : side === 'both' ? 'ทั้งสองข้าง' : 'ขวา';
      const sEn = side === 'left' ? 'left' : side === 'both' ? 'both sides' : 'right';
      const sCn = side === 'left' ? '左' : side === 'both' ? '双侧' : '右';
      return {
        th: [
          `ยืนหลังตรง ปล่อยแขน${sTh}ข้างลำตัว (0°)`,
          `ยกแขน${sTh}ชูขึ้นเหนือศีรษะชิดข้างใบหู (${rom}°)`,
          `ค้างไว้ด้านบน ${holdSec} วินาที พร้อมหายใจเข้าลึกๆ`,
        ],
        en: [
          `Stand upright with ${sEn} arm resting downward (0°)`,
          `Reach ${sEn} arm overhead beside ear toward ${rom}°`,
          `Hold overhead for ${holdSec}s while breathing evenly`,
        ],
        cn: [
          `站直准备抬起${sCn}臂（0°）`,
          `伸直${sCn}臂向上高举过头靠近耳侧（${rom}°）`,
          `上方保持 ${holdSec} 秒并保持自然呼吸`,
        ],
      };
    },
  },
  cross_body_reach: {
    region: {
      th: 'ข้อไหล่ด้านหน้า & กอดลำตัว (Cross-Body Reach)',
      en: 'Horizontal Adduction • Opposite Shoulder Reach',
      cn: '肩关节水平内收（交叉触摸对侧肩）',
    },
    view: {
      th: 'มุมมองหน้าตรง (วาดแขนไขว้ผ่านหน้าอกแตะไหล่ตรงข้าม)',
      en: 'Front coronal view (Reach gently across chest)',
      cn: '正面冠状位视角（手臂平缓横跨胸前）',
    },
    muscle: {
      th: 'ยืดเยื่อหุ้มข้อไหล่ด้านหลัง เพื่อฝึกเอื้อมติดกระดุมและหยิบของข้างตัว',
      en: 'Posterior shoulder capsule stretch for cross-chest dressing & reaching',
      cn: '拉伸肩后囊并改善对侧穿衣取物功能',
    },
    caution: {
      th: 'หน้าอกและลำตัวตั้งตรงนิ่ง ไม่บิดเอวตามมือที่เอื้อมไป',
      en: 'Keep trunk steady facing camera; avoid twisting spine with arm',
      cn: '保持躯干正对屏幕稳定，避免扭腰代偿',
    },
    adl: {
      th: '👕 กิจวัตรประจำวัน: เอื้อมแตะไหล่ฝั่งตรงข้าม / คาดเข็มขัดนิรภัย',
      en: '👕 Daily ADL: Reaching across chest / Buckling seatbelt',
      cn: '👕 日常活动：交叉触摸对侧肩 / 系安全带',
    },
    steps: (side, rom, holdSec) => {
      const sTh = side === 'left' ? 'ซ้าย' : side === 'both' ? 'ทั้งสองข้าง' : 'ขวา';
      const oppTh = side === 'left' ? 'ขวา' : side === 'both' ? 'ตรงข้ามสลับ/พร้อมกัน' : 'ซ้าย';
      const sEn = side === 'left' ? 'left' : side === 'both' ? 'both' : 'right';
      const oppEn = side === 'left' ? 'right' : side === 'both' ? 'opposite' : 'left';
      const sCn = side === 'left' ? '左' : side === 'both' ? '双侧' : '右';
      const oppCn = side === 'left' ? '右' : side === 'both' ? '对侧' : '左';
      return {
        th: [
          `นั่งหรือยืนตรง ยกแขน${sTh}ขึ้นระดับหัวไหล่ขนานพื้น`,
          `ค่อยๆ วาดแขน${sTh}ไขว้ผ่านหน้าอกไปแตะหัวไหล่ฝั่ง${oppTh} (~${rom}°)`,
          `ค้างท่าที่จุดตึงสบาย ${holdSec} วินาที แล้วกางแขนกลับท่าเดิม`,
        ],
        en: [
          `Sit or stand upright and raise ${sEn} arm comfortably to chest height`,
          `Sweep ${sEn} arm across chest toward your ${oppEn} shoulder (~${rom}°)`,
          `Hold comfortably for ${holdSec}s, then sweep smoothly back`,
        ],
        cn: [
          `正坐或站直，平抬${sCn}臂至胸前水平高度`,
          `将${sCn}臂横跨胸前缓慢伸向${oppCn}肩（约 ${rom}°）`,
          `保持牵拉 ${holdSec} 秒后缓慢收回`,
        ],
      };
    },
  },
  scapular_protract: {
    region: {
      th: 'ข้อสะบัก & หัวไหล่ด้านหน้า (Scapular Protraction)',
      en: 'Scapular Protraction • Seated Forward Reach',
      cn: '肩胛骨前伸推展控制',
    },
    view: {
      th: 'มุมมองด้านข้าง (ดันข้อสะบักยื่นแขนตรงไปข้างหน้าระดับอก)',
      en: 'Side profile view (Push shoulder blade forward at 90°)',
      cn: '侧面视角（90° 水平前推肩胛骨）',
    },
    muscle: {
      th: 'ปลุกการทำงานของกล้ามเนื้อ Serratus Anterior ลดปัญหาข้อไหล่หลวม/ไหล่ตก',
      en: 'Activates serratus anterior stabilizing post-stroke shoulder subluxation',
      cn: '激活前锯肌，改善卒中后肩关节半脱位与肩胛无力',
    },
    caution: {
      th: 'เหยียดข้อศอกให้ตรงแล้วใช้ข้อสะบักดันมือยื่นไปข้างหน้า ไม่โน้มตัวตาม',
      en: 'Keep elbow straight and glide shoulder blade forward without leaning chest',
      cn: '伸直肘关节并仅前推肩胛，切勿身体大幅前倾',
    },
    adl: {
      th: '🚪 กิจวัตรประจำวัน: เอื้อมเปิดประตู / หยิบของบนโต๊ะด้านหน้า',
      en: '🚪 Daily ADL: Pushing door / Reaching across table',
      cn: '🚪 日常活动：推门与桌面取物',
    },
    steps: (side, rom, holdSec) => {
      const sTh = side === 'left' ? 'ซ้าย' : side === 'both' ? 'ทั้งสองข้าง' : 'ขวา';
      const sEn = side === 'left' ? 'left' : side === 'both' ? 'both sides' : 'right';
      const sCn = side === 'left' ? '左' : side === 'both' ? '双侧' : '右';
      return {
        th: [
          `นั่งหลังตรง ยกแขน${sTh}เหยียดตรงระดับหน้าอก 90°`,
          `ยื่นมือและหัวไหล่${sTh}ไปด้านหน้าเพิ่มเติม (~${rom}°)`,
          `ค้างท่าผลักสะบักไว้ ${holdSec} วินาที แล้วดึงไหล่กลับชิดพนัก`,
        ],
        en: [
          `Sit tall with ${sEn} arm extended straight forward at chest height (90°)`,
          `Glide ${sEn} shoulder blade forward extending reach (~${rom}°)`,
          `Hold protraction steady for ${holdSec}s, then retract smoothly`,
        ],
        cn: [
          `端坐挺直腰背，${sCn}臂前平举 90°`,
          `保持肘直，向前推展${sCn}侧肩胛骨（约 ${rom}°）`,
          `保持前伸 ${holdSec} 秒后缓慢收回肩胛`,
        ],
      };
    },
  },
  bobath_forward_push: {
    region: {
      th: 'แขน 2 ข้างประสานมือเหยียดศอกตรง (Bobath Forward Push)',
      en: 'Bilateral Clasped Forward Elbow Extension (Bobath)',
      cn: '双手交叉前推伸肘（Bobath 模式）',
    },
    view: {
      th: 'มุมมองด้านข้าง (ประสานมือดันเหยียดศอกตรงจากอกไปด้านหน้า)',
      en: 'Side profile view (Push clasped hands straight forward)',
      cn: '侧面视角（双手交叉从胸前用力前推伸直双肘）',
    },
    muscle: {
      th: 'ลดอาการเกร็งงอแขนแบบ Flexor Synergy และฝึกเหยียดศอกคู่ตามหลัก Bobath',
      en: 'Inhibits flexor spasticity via bilateral clasped elbow push',
      cn: '依据 Bobath 理念抑制屈肌痉挛并促进双侧对称伸肘',
    },
    caution: {
      th: 'สอดนิ้วโป้งข้างอ่อนแรงให้อยู่ด้านบนมือที่ประสานเสมอ เหยียดศอกให้สุดเท่าที่ทำได้',
      en: 'Keep affected thumb placed on top of clasp and extend elbows symmetrically',
      cn: '患侧拇指始终置于交叉上方，对称推直双肘',
    },
    adl: {
      th: '🤲 กิจวัตร Bobath: ผลักบานประตู / ยื่นถือถาดอาหาร 2 มือ',
      en: '🤲 Bobath ADL: Symmetric reaching & holding items with both hands',
      cn: '🤲 Bobath 训练：双手推门与协同递物',
    },
    steps: (_side, rom, holdSec) => ({
      th: [
        'ประสานนิ้วมือทั้งสองข้างเข้าด้วยกันไว้ที่กลางหน้าอก',
        `ค่อยๆ ดันมือที่ประสานยืดข้อศอกตรงไปด้านหน้าระดับอกจนถึง ~${rom}°`,
        `ค้างท่าแขนเหยียดตรง ${holdSec} วินาที แล้วดึงมือกลับมาชิดอก`,
      ],
      en: [
        'Clasp all fingers snugly together in front of your chest',
        `Push clasped hands straight forward extending elbows smoothly (~${rom}°)`,
        `Hold full forward push for ${holdSec}s, then draw back to chest`,
      ],
      cn: [
        '双手十指交叉相握置于胸前',
        `沿胸前高度缓慢将双手向前推直（约 ${rom}°）`,
        `伸直保持 ${holdSec} 秒后缓慢收回胸前`,
      ],
    }),
  },
  hand_to_head: {
    region: {
      th: 'ข้อศอก & ไหล่: ยกมือแตะศีรษะ / หวีผม (Functional Reach)',
      en: 'Elbow & Shoulder • Hand-to-Head Grooming Reach',
      cn: '肘肩协同功能训练 • 抬手摸头与梳头',
    },
    view: {
      th: 'มุมมองหน้าตรง (งอศอกยกฝ่ามือขึ้นแตะบริเวณขมับ-ศีรษะ)',
      en: 'Front coronal view (Lift hand smoothly toward head crown)',
      cn: '正面冠状位视角（屈肘抬手轻触头顶）',
    },
    muscle: {
      th: 'ประสานการทำงานข้อไหล่และข้อศอกเพื่อฟื้นฟูการสระผม ล้างหน้า และหวีผม',
      en: 'Coordinated elbow flexion + shoulder lift for self-hygiene & combing',
      cn: '协调肩肘运动，恢复自主洗脸、梳头与个人护理动作',
    },
    caution: {
      th: 'ศีรษะตั้งตรงมองกล้อง ไม่ก้มหน้าลงมาหามือขณะยกมือขึ้น',
      en: 'Keep head upright facing camera; bring hand up rather than ducking head',
      cn: '头部保持直立平视镜头，切勿低头迎合手部',
    },
    adl: {
      th: '🪥 กิจวัตรประจำวัน: ล้างหน้า / หวีผม / สวมหมวกด้วยตนเอง',
      en: '🪥 Daily ADL: Face washing / Hair combing / Putting on hat',
      cn: '🪥 日常活动：自主洗脸 / 梳头 / 戴帽',
    },
    steps: (side, rom, holdSec) => {
      const sTh = side === 'left' ? 'ซ้าย' : side === 'both' ? 'ทั้งสองข้าง' : 'ขวา';
      const sEn = side === 'left' ? 'left' : side === 'both' ? 'both sides' : 'right';
      const sCn = side === 'left' ? '左' : side === 'both' ? '双侧' : '右';
      return {
        th: [
          `ตั้งศีรษะตรง ปล่อยมือ${sTh}ข้างลำตัวสบายๆ`,
          `งอศอกพร้อมยกมือ${sTh}ขึ้นช้าๆ แตะบริเวณข้างศีรษะหรือกระหม่อม (~${rom}°)`,
          `แตะค้างไว้ ${holdSec} วินาที แล้วลดมือกลับลงข้างตัว`,
        ],
        en: [
          `Sit or stand tall with ${sEn} arm resting comfortably`,
          `Bend elbow and lift ${sEn} hand smoothly to touch temple/head crown (~${rom}°)`,
          `Hold lightly at head for ${holdSec}s, then lower gently`,
        ],
        cn: [
          `头部直立平视，${sCn}臂自然垂放体侧`,
          `缓慢屈肘抬起${sCn}手轻触太阳穴或头顶（约 ${rom}°）`,
          `轻触保持 ${holdSec} 秒后缓慢放下手臂`,
        ],
      };
    },
  },
  forearm_turn: {
    region: {
      th: 'ปลายแขน & ข้อมือ: คว่ำมือ-หงายมือ (Pronation/Supination)',
      en: 'Forearm Rotation • Palm Up / Palm Down',
      cn: '前臂旋前旋后训练 • 翻手掌心向上/向下',
    },
    view: {
      th: 'มุมมองระยะใกล้ (พลิกฝ่ามือหงายขึ้นและคว่ำลงสลับกัน)',
      en: 'Close forearm view (Flip palm steadily upward & downward)',
      cn: '前臂近景视角（交替翻转掌心向上与向下）',
    },
    muscle: {
      th: 'ฟื้นฟูข้อต่อกระดูกปลายแขน เพื่อการบิดลูกบิดประตู เปิดขวดน้ำ และรับเหรียญ',
      en: 'Restores radioulnar rotation needed for doorknobs, jars & holding bowls',
      cn: '改善前臂旋转活动度，恢复拧门把手、端碗与接物能力',
    },
    caution: {
      th: 'หนีบศอกให้อยู่นิ่งข้างตัว ห้ามกางศอกเหวี่ยงช่วยพลิกมือ',
      en: 'Pin elbow steady at your waist; rotate only the forearm and palm',
      cn: '保持手肘紧贴腰侧稳定，仅转动前臂与手掌',
    },
    adl: {
      th: '🔑 กิจวัตรประจำวัน: บิดลูกบิดประตู / หงายมือรับของ / เทน้ำ',
      en: '🔑 Daily ADL: Turning doorknob / Receiving coins / Pouring',
      cn: '🔑 日常活动：拧门把手 / 摊手接物 / 倒水',
    },
    steps: (side, rom, holdSec) => {
      const sTh = side === 'left' ? 'ซ้าย' : side === 'both' ? 'ทั้งสองข้าง' : 'ขวา';
      const sEn = side === 'left' ? 'left' : side === 'both' ? 'both sides' : 'right';
      const sCn = side === 'left' ? '左' : side === 'both' ? '双侧' : '右';
      return {
        th: [
          `งอศอก${sTh} 90° แนบลำตัว ยื่นฝ่ามือให้อยู่ในกรอบกล้อง`,
          `พลิกหงายฝ่ามือ${sTh}ขึ้นจนสุด (~${rom}°) แล้วพลิกคว่ำฝ่ามือลงช้าๆ`,
          `ค้างท่าที่จุดหงายหรือคว่ำสุด ${holdSec} วินาที อย่างมั่นคง`,
        ],
        en: [
          `Bend ${sEn} elbow 90° against side and position hand clearly facing camera`,
          `Rotate forearm to turn palm flat up (~${rom}°), then flip smoothly palm down`,
          `Pause and hold each turned position clearly for ${holdSec}s`,
        ],
        cn: [
          `${sCn}肘弯曲90°贴紧体侧，手部清晰面向摄像头`,
          `缓缓翻转${sCn}前臂使掌心朝上（约 ${rom}°），再缓慢翻转为掌心朝下`,
          `每次翻转到位保持 ${holdSec} 秒`,
        ],
      };
    },
  },
  thumb_pinch: {
    region: {
      th: 'นิ้วหัวแม่มือ & กล้ามเนื้อมัดเล็ก (Precision Thumb Pinch)',
      en: 'Fine Motor • Thumb-to-Finger Tip Pinch',
      cn: '手指精细动作 • 拇指对指捏合训练',
    },
    view: {
      th: 'มุมมองระยะใกล้ (จีบปลายนิ้วโป้งแตะนิ้วชี้-ก้อยเป็นรูปวงกลม)',
      en: 'Close hand view (Pinch thumb tip cleanly into an O-ring)',
      cn: '手部特写视角（拇指依次与其余四指指尖捏合呈 O 形）',
    },
    muscle: {
      th: 'ฝึกการควบคุมกล้ามเนื้อมัดเล็กในมือ เพื่อจับช้อน ติดกระดุม และจับปากกาเขียน',
      en: 'Trains intrinsic thumb opposition for buttoning, writing & utensil grip',
      cn: '强化手内在肌对指精细控制，辅助握笔、系扣与夹取小物',
    },
    caution: {
      th: 'ยกมือให้เห็นปลายนิ้วชัดเจนในกรอบกล้อง ค่อยๆ จีบปลายนิ้วแตะทีละนิ้ว',
      en: 'Show fingertips clearly to camera and touch pad-to-pad cleanly into an O-shape',
      cn: '手掌在镜头中保持清晰，轻柔精准完成指腹捏合呈圆形',
    },
    adl: {
      th: '✏️ กิจวัตรประจำวัน: จับปากกา / ติดกระดุมเสื้อ / คีบอาหาร',
      en: '✏️ Daily ADL: Writing pen grip / Buttoning shirt / Picking small items',
      cn: '✏️ 日常活动：握笔写字 / 系衣扣 / 夹取小物件',
    },
    steps: (side, _rom, holdSec) => {
      const sTh = side === 'left' ? 'ซ้าย' : side === 'both' ? 'ทั้งสองข้าง' : 'ขวา';
      const sEn = side === 'left' ? 'left' : side === 'both' ? 'both sides' : 'right';
      const sCn = side === 'left' ? '左' : side === 'both' ? '双侧' : '右';
      return {
        th: [
          `ยกฝ่ามือ${sTh}ขึ้นระดับอก หันเข้าหากล้องอย่างชัดเจน`,
          `จีบปลายนิ้วโป้ง${sTh}เข้าแตะปลายนิ้วชี้เป็นวงกลมรูปตัว O แล้วคลายออก`,
          `ค้างท่าจีบนิ้วไว้ ${holdSec} วินาที ก่อนสลับแตะนิ้วกลาง นาง และก้อย`,
        ],
        en: [
          `Hold ${sEn} palm clearly inside camera frame at chest level`,
          `Bring ${sEn} thumb tip gently to index fingertip forming a crisp O-ring`,
          `Hold pinch contact for ${holdSec}s, then repeat smoothly across remaining fingers`,
        ],
        cn: [
          `将${sCn}手掌平举胸前并清晰面向摄像头`,
          `用${sCn}手拇指指尖轻触食指指尖形成清晰 O 形，然后缓慢张开`,
          `每次对捏保持 ${holdSec} 秒后依次换触中指、无名指与小指`,
        ],
      };
    },
  },
  neck_turn: {
    region: {
      th: 'กระดูกต้นคอ & การเหลียวมอง 2 ข้าง (Cervical Rotation)',
      en: 'Cervical Spine • Gentle Left-Right Head Turn',
      cn: '颈椎水平旋转 • 左右温和转头（卫生部标准）',
    },
    view: {
      th: 'มุมมองหน้าตรง (หันคางมองข้ามหัวไหล่ซ้ายและขวาอย่างช้าๆ)',
      en: 'Front view (Turn chin smoothly left and right over level shoulders)',
      cn: '正面视角（双肩保持不动，下颌水平缓慢转向左右肩方向）',
    },
    muscle: {
      th: 'เพิ่มความคล่องตัวข้อต่อคอ ป้องกันคอตึง-เวียนศีรษะขณะมองรอบตัวในผู้สูงอายุ',
      en: 'Maintains safe cervical rotation for scanning surroundings without dizziness',
      cn: '维持老年人及卒中患者颈部灵活旋转，预防僵硬与转头头晕',
    },
    caution: {
      th: 'หันช้าและนุ่มนวล ห้ามกระชากคอ หากเวียนศีรษะให้หยุดพักตรงกลางทันที',
      en: 'Rotate smoothly; stop immediately at center if any dizziness occurs',
      cn: '转动务必慢而轻柔；若感头晕请立即停于中立位休息',
    },
    adl: {
      th: '🌿 มาตรฐาน สธ.: หันมองสิ่งรอบข้างและลดปวดคอเรื้อรัง',
      en: '🌿 MOPH Elderly Neck Mobility: Safe visual scanning & neck relief',
      cn: '🌿 卫生部推荐：安全环顾四周及预防慢性颈痛',
    },
    steps: (_side, rom, holdSec) => ({
      th: [
        'นั่งหลังตรง มองตรงเข้ากล้อง กดไหล่ทั้ง 2 ข้างลงสบายๆ',
        `ค่อยๆ หันศีรษะมองข้ามหัวไหล่ข้างซ้าย (~${rom}°) แล้วหันสลับไปข้างขวา`,
        `ค้างมุมหันแต่ละข้าง ${holdSec} วินาที อย่างนุ่มนวล`,
      ],
      en: [
        'Sit upright looking straight ahead with both shoulders relaxed',
        `Slowly rotate chin toward left shoulder (~${rom}°), then gently turn right`,
        `Hold comfortably at each side for ${holdSec}s before centering`,
      ],
      cn: [
        '端坐挺直，双眼平视前方，双肩放松下沉',
        `缓慢水平转头望向左肩方向（约 ${rom}°），随后轻柔转向右侧`,
        `每侧柔和保持 ${holdSec} 秒后再回正中位`,
      ],
    }),
  },
  shoulder_rotation: {
    region: {
      th: 'ข้อไหล่หมุนเข้า-ออก (Rotator Cuff)',
      en: 'Shoulder External/Internal Rotation',
      cn: '肩关节内外旋（肩袖肌群）',
    },
    view: {
      th: 'มุมมองหน้าตรง (หนีบศอกข้างเอว หมุนปลายแขนออก)',
      en: 'Front view (Elbow tucked, forearm sweeps outward)',
      cn: '正面视角（肘部贴腰，前臂向外旋转）',
    },
    muscle: {
      th: 'ฟื้นฟูการหมุนหัวไหล่เพื่ออ้อมแขนใส่เสื้อและติดกระดุม',
      en: 'Rotator cuff control for buttoning & backward shirt reach',
      cn: '强化肩袖旋转控制，辅助反手穿衣与系扣',
    },
    caution: {
      th: 'หนีบข้อศอกให้ชิดสีข้างไว้ตลอด ห้ามกางศอกหนีลำตัว',
      en: 'Keep elbow tucked snugly against waist throughout rotation',
      cn: '保持手肘紧贴腰侧，勿让手肘离开身体',
    },
    adl: {
      th: '🔘 กิจวัตรประจำวัน: อ้อมแขนใส่เสื้อ / ติดกระดุม / หวีผม',
      en: '🔘 Daily ADL: Tucking shirt / Buttoning / Grooming',
      cn: '🔘 日常活动：反手穿衣 / 系扣子',
    },
    steps: (side, rom, holdSec) => {
      const sTh = side === 'left' ? 'ซ้าย' : side === 'both' ? 'ทั้งสองข้าง' : 'ขวา';
      const sEn = side === 'left' ? 'left' : side === 'both' ? 'both sides' : 'right';
      const sCn = side === 'left' ? '左' : side === 'both' ? '双侧' : '右';
      return {
        th: [
          `งอข้อศอก${sTh} 90° แล้วแนบข้อศอกชิดเอวให้มั่นคง`,
          `หมุนปลายแขน${sTh}กางออกด้านข้างช้าๆ ถึง ~${rom}°`,
          `ค้างไว้ที่จุดสุด ${holdSec} วินาที แล้วหมุนกลับเข้าด้านใน`,
        ],
        en: [
          `Bend ${sEn} elbow 90° and tuck upper arm close to waist`,
          `Rotate ${sEn} forearm outward gently to ~${rom}°`,
          `Hold for ${holdSec}s, then rotate smoothly back inward`,
        ],
        cn: [
          `${sCn}肘弯曲90°，上臂紧贴体侧`,
          `缓慢向外旋转${sCn}前臂至约 ${rom}°`,
          `保持 ${holdSec} 秒后缓慢旋回腹前`,
        ],
      };
    },
  },
  shoulder_shrug_roll: {
    region: {
      th: 'หัวไหล่ทั้ง 2 ข้าง • กล้ามเนื้อบ่าและสะบัก',
      en: 'Both Shoulders • Upper Trapezius & Scapula',
      cn: '双肩及上斜方肌 • 肩胛环绕',
    },
    view: {
      th: 'มุมมองหน้าตรง (ยกหัวไหล่ขึ้น & หมุนลง)',
      en: 'Front view (Elevate shoulders & roll back)',
      cn: '正面视角（双肩向上耸起并向后绕环）',
    },
    muscle: {
      th: 'ลดอาการไหล่ห่อ ไหล่ติด และกระตุ้นระบบไหลเวียนช่วงบ่าตามมาตรฐาน สธ.',
      en: 'MOPH standard scapular shrug & roll to relieve shoulder stiffness',
      cn: '卫生部标准：放松肩颈僵硬并促进肩部血液循环',
    },
    caution: {
      th: 'คอตั้งตรง มองตรงไปด้านหน้า ไม่ยื่นคางมาช่วยยกไหล่',
      en: 'Keep neck tall and chin tucked while rolling shoulders',
      cn: '保持颈部挺直平视前方，切勿前伸下巴',
    },
    adl: {
      th: '🌿 มาตรฐาน สธ. ผู้สูงอายุ: คลายบ่า-ไหล่ ลดอาการปวดตึง',
      en: '🌿 MOPH Elderly Care: Relieve neck/shoulder tension',
      cn: '🌿 卫生部老年保健：舒缓肩颈僵紧',
    },
    steps: (_side, _rom, holdSec) => ({
      th: [
        'นั่งหรือยืนตัวตรง ปล่อยแขนทั้งสองข้างทิ้งตัวสบายๆ',
        `ยกหัวไหล่ทั้ง 2 ข้างขึ้นหาใบหูช้าๆ แล้วค้างไว้ ${holdSec} วินาที`,
        'หมุนหัวไหล่ไปด้านหลังแล้วกดลงผ่อนคลายเต็มที่',
      ],
      en: [
        'Sit or stand upright with both arms relaxed at your sides',
        `Slowly shrug both shoulders up toward ears and hold ${holdSec}s`,
        'Roll shoulders backward and drop down smoothly to relax',
      ],
      cn: [
        '坐直或站直，双臂自然垂放放松',
        `双肩缓慢向上耸起靠近耳部保持 ${holdSec} 秒`,
        '向后画圈环绕并沉肩彻底放松',
      ],
    }),
  },
  bilateral_arm_raise: {
    region: {
      th: 'แขน & ข้อไหล่ทั้ง 2 ข้าง (ประสานมือช่วยกัน)',
      en: 'Bilateral Arms & Shoulders (Self-Assisted)',
      cn: '双臂协同高举过头（健侧辅助患侧）',
    },
    view: {
      th: 'มุมมองหน้าตรง (ชูแขนทั้ง 2 ข้างขึ้นพร้อมกัน)',
      en: 'Front view (Both arms rising symmetrically)',
      cn: '正面视角（双手同步向上举过头顶）',
    },
    muscle: {
      th: 'ให้แขนข้างแข็งแรงช่วยพยุงแขนข้างอ่อนแรงชูเหนือศีรษะและขยายปอด',
      en: 'Strong arm guides weaker side overhead & promotes chest expansion',
      cn: '由健侧手臂带动患侧上举，同时舒展胸廓',
    },
    caution: {
      th: 'หายใจเข้าลึกขณะยกขึ้น และผ่อนลมหายใจออกช้าๆ ตอนลดแขนลง',
      en: 'Inhale gently while lifting; exhale smoothly on lowering',
      cn: '上举时配合深吸气，放下时缓慢呼气',
    },
    adl: {
      th: '🤝 มาตรฐาน สธ.: แขนดีช่วยพยุงแขนอ่อนแรง + บริหารปอด',
      en: '🤝 MOPH Self-Assisted Raise: Symmetrical reach & breathing',
      cn: '🤝 卫生部推荐：健侧带动患侧上举及深呼吸',
    },
    steps: (_side, rom, holdSec) => ({
      th: [
        'ปล่อยแขนทั้ง 2 ข้างลงสบายๆ เตรียมยกพร้อมกัน (0°)',
        `ค่อยๆ ยกแขนทั้งสองข้างชูขึ้นเหนือศีรษะพร้อมกันถึง ~${rom}°`,
        `ค้างไว้ด้านบน ${holdSec} วินาที พร้อมหายใจลึก แล้วลดลงช้าๆ`,
      ],
      en: [
        'Start upright with both arms relaxed down at sides (0°)',
        `Raise both arms smoothly overhead together toward ~${rom}°`,
        `Hold at top for ${holdSec}s breathing deeply, then lower slowly`,
      ],
      cn: [
        '双臂自然垂放准备同时抬起（0°）',
        `双臂同时缓慢向上举过头顶约 ${rom}°`,
        `在上方保持 ${holdSec} 秒并深呼吸，随后缓慢放下`,
      ],
    }),
  },
  elbow_flexion: {
    region: {
      th: 'ข้อศอก & กล้ามเนื้อต้นแขนด้านหน้า',
      en: 'Elbow Hinge • Biceps Flexion/Extension',
      cn: '肘关节屈伸 • 上臂前侧肌群',
    },
    view: {
      th: 'มุมมองหน้าตรง (พับข้อศอกเข้าหาหัวไหล่)',
      en: 'Front view (Curl forearm smoothly toward shoulder)',
      cn: '正面视角（前臂向肩部弯曲再伸直）',
    },
    muscle: {
      th: 'ฝึกงอ-เหยียดศอก เพื่อตักอาหารดื่มน้ำ ดึงแขนเสื้อ และล้างหน้า',
      en: 'Restores bending power for eating, drinking & sleeve pull',
      cn: '训练屈伸肘力，改善端水进食与拉衣袖动作',
    },
    caution: {
      th: 'ตรึงต้นแขนให้นิ่งแนบข้างลำตัว เคลื่อนไหวเฉพาะช่วงข้อศอกลงไป',
      en: 'Keep upper arm steady by ribs; pivot only at the elbow',
      cn: '保持上臂固定于体侧，仅弯曲伸直前臂',
    },
    adl: {
      th: '🍽️ กิจวัตรประจำวัน: ดึงแขนเสื้อ / ตักอาหารเข้าปาก',
      en: '🍽️ Daily ADL: Pulling sleeves / Feeding hand-to-mouth',
      cn: '🍽️ 日常活动：拉穿衣袖 / 自主进食',
    },
    steps: (side, _rom, holdSec) => {
      const sTh = side === 'left' ? 'ซ้าย' : side === 'both' ? 'ทั้งสองข้าง' : 'ขวา';
      const sEn = side === 'left' ? 'left' : side === 'both' ? 'both sides' : 'right';
      const sCn = side === 'left' ? '左' : side === 'both' ? '双侧' : '右';
      return {
        th: [
          `แนบต้นแขน${sTh}ข้างลำตัว เหยียดปลายแขน${sTh}ลงด้านล่าง`,
          `พับงอข้อศอก${sTh}ดึงฝ่ามือเข้าหาหัวไหล่ช้าๆ`,
          `ค้างไว้ ${holdSec} วินาที แล้วค่อยๆ เหยียดข้อศอกออกจนสุด`,
        ],
        en: [
          `Keep ${sEn} upper arm steady by waist with forearm extended`,
          `Flex ${sEn} elbow slowly curling hand toward shoulder`,
          `Hold for ${holdSec}s, then slowly lower back straight`,
        ],
        cn: [
          `${sCn}上臂贴紧腰侧，前臂自然向下伸展`,
          `缓慢屈曲${sCn}肘，将手掌弯向肩部`,
          `保持 ${holdSec} 秒后缓慢伸直还原`,
        ],
      };
    },
  },
  wrist_finger: {
    region: {
      th: 'ข้อมือ & ข้อนิ้วทั้ง 5 นิ้ว (Fine Motor)',
      en: 'Wrist Extension & 5-Finger Opening',
      cn: '腕关节背伸与五指张合训练',
    },
    view: {
      th: 'มุมมองระยะใกล้ (กระดกข้อมือขึ้น & กาง-กำนิ้วมือ)',
      en: 'Close hand view (Extend wrist & spread fingers wide)',
      cn: '手部特写视角（手腕上抬背伸与张指握拳）',
    },
    muscle: {
      th: 'ลดอาการกำมือเกร็ง (Spasticity) และฝึกเปิดฝ่ามือหยิบจับสิ่งของ',
      en: 'Reduces flexor spasticity & trains open-grip hand release',
      cn: '缓解手部屈肌痉挛，恢复抓握与张手释放能力',
    },
    caution: {
      th: 'ประคองแขนให้อยู่ในกรอบกล้องเพื่อให้ AI เห็นนิ้วทั้ง 5 ชัดเจน',
      en: 'Hold hand clearly inside camera view so all 5 fingers track',
      cn: '将手部置于摄像头清晰视野内以便识别五指伸展',
    },
    adl: {
      th: '✋ กิจวัตรประจำวัน: หยิบช้อน / จับแก้ว / ปล่อยของจากมือ',
      en: '✋ Daily ADL: Holding spoon/cup & releasing objects',
      cn: '✋ 日常活动：握勺取杯 / 自主张手放物',
    },
    steps: (side, _rom, holdSec) => {
      const sTh = side === 'left' ? 'ซ้าย' : side === 'both' ? 'ทั้งสองข้าง' : 'ขวา';
      const sEn = side === 'left' ? 'left' : side === 'both' ? 'both sides' : 'right';
      const sCn = side === 'left' ? '左' : side === 'both' ? '双侧' : '右';
      return {
        th: [
          `วางหรือยกปลายแขน${sTh}ระดับหน้าอก หันฝ่ามือเข้ากล้อง`,
          `กระดกข้อมือ${sTh}ขึ้นพร้อมกางนิ้วทั้ง 5 ออกให้กว้างที่สุด`,
          `ค้างท่ากางนิ้ว ${holdSec} วินาที แล้วค่อยๆ กำมือหลวมๆ สลับกัน`,
        ],
        en: [
          `Hold ${sEn} forearm comfortably at chest height facing camera`,
          `Extend ${sEn} wrist upward while spreading all 5 fingers wide`,
          `Hold open for ${holdSec}s, then gently curl into loose fist`,
        ],
        cn: [
          `将${sCn}前臂平举至胸前正对镜头`,
          `手腕向上背伸，同时用力张开五指`,
          `张开保持 ${holdSec} 秒后轻柔握拳交替练习`,
        ],
      };
    },
  },
  neck_tilt: {
    region: {
      th: 'กระดูกต้นคอ & กล้ามเนื้อบ่า (Cervical)',
      en: 'Cervical Neck & Upper Trapezius',
      cn: '颈椎侧屈与斜方肌伸展',
    },
    view: {
      th: 'มุมมองหน้าตรง (เอียงศีรษะช้าๆ ไหล่ทั้ง 2 ข้างนิ่งตรง)',
      en: 'Front view (Gentle head tilt with level shoulders)',
      cn: '正面视角（头颈轻柔侧倾，双肩保持水平）',
    },
    muscle: {
      th: 'ยืดกล้ามเนื้อคอ-บ่า ลดอาการคอแข็ง คอเอียงในผู้สูงอายุ/Stroke',
      en: 'Gentle lateral stretch relieving stiff neck & postural drop',
      cn: '轻柔牵拉颈肩肌群，改善颈部僵硬与偏斜',
    },
    caution: {
      th: 'เคลื่อนไหวช้ามาก! ห้ามสะบัดคอและกดหัวไหล่ทั้งสองข้างให้นิ่งเสมอ',
      en: 'Move VERY slowly! Keep both shoulders flat and relaxed',
      cn: '动作务必极其缓慢！严禁甩头或代偿耸肩',
    },
    adl: {
      th: '🌿 มาตรฐาน สธ.: ป้องกันคอตก ลดปวดตึงต้นคอและเวียนศีรษะ',
      en: '🌿 MOPH Neck Care: Prevents stiff neck & muscle tightness',
      cn: '🌿 卫生部标准：预防颈部僵硬与肌群劳损',
    },
    steps: (side, rom, holdSec) => {
      const sTh = side === 'left' ? 'ซ้าย' : side === 'both' ? 'ทั้งสองข้าง' : 'ขวา';
      const sEn = side === 'left' ? 'left' : side === 'both' ? 'both sides' : 'right';
      const sCn = side === 'left' ? '左' : side === 'both' ? '双侧' : '右';
      return {
        th: [
          'นั่งหรือยืนหลังตรง มองตรงไปด้านหน้า ลดหัวไหล่ทั้ง 2 ข้างลงต่ำ',
          `ค่อยๆ เอียงใบหูเข้าหาหัวไหล่${sTh}อย่างช้าๆ (~${rom}°)`,
          `ค้างยืดกล้ามเนื้อข้างลำคอ ${holdSec} วินาที แล้วตั้งศีรษะกลับตรง`,
        ],
        en: [
          'Sit or stand tall looking straight forward with shoulders relaxed',
          `Slowly tilt ear toward your ${sEn} shoulder (~${rom}°)`,
          `Hold stretch gently for ${holdSec}s, then return head upright`,
        ],
        cn: [
          '正坐或站直平视前方，双肩放松下沉',
          `缓慢将耳部向${sCn}肩方向轻柔侧倾（约 ${rom}°）`,
          `保持牵拉 ${holdSec} 秒后缓慢回正`,
        ],
      };
    },
  },
  trunk_align: {
    region: {
      th: 'แกนกลางลำตัว & แนวกระดูกสันหลัง (Trunk Core)',
      en: 'Postural Core & Spinal Alignment',
      cn: '躯干核心控制与脊柱直立保持',
    },
    view: {
      th: 'มุมมองหน้าตรง (แกนกลางลำตัวตั้งฉาก 0°)',
      en: 'Front upright posture (0° spinal alignment)',
      cn: '正面视角（躯干与脊柱保持垂直 0°）',
    },
    muscle: {
      th: 'ฝึกการทรงตัวแกนกลางขณะนั่ง ป้องกันลำตัวเอียงล้มข้างอ่อนแรง',
      en: 'Seated trunk balance preventing post-stroke lateral lean',
      cn: '增强坐姿核心稳定，预防卒中后向患侧倾倒',
    },
    caution: {
      th: 'ลงน้ำหนักที่สะโพกทั้ง 2 ข้างเท่าๆ กัน ศีรษะอยู่กึ่งกลางตัว',
      en: 'Distribute weight evenly on both hips; center head over pelvis',
      cn: '双侧臀部均匀承重，头部正对骨盆中心',
    },
    steps: (_side, _rom, holdSec) => ({
      th: [
        'นั่งเก้าอี้มั่นคง เท้าทั้งสองข้างวางราบสนิทบนพื้น',
        'ยืดหลังตรง เปิดหน้าอก เก็บคางเล็กน้อยให้ลำตัวตั้งฉาก',
        `ทรงตัวในแนวตรงนิ่งสม่ำเสมอค้างไว้ ${holdSec} วินาที`,
      ],
      en: [
        'Sit securely on chair with both feet flat on floor',
        'Lengthen spine upright and align shoulders level over hips',
        `Maintain balanced midline posture steadily for ${holdSec}s`,
      ],
      cn: [
        '稳固坐于椅面，双脚平踩地面',
        '挺直腰背微收下颌，保持双肩与骨盆水平对齐',
        `稳定维持中立坐姿 ${holdSec} 秒`,
      ],
    }),
  },
};

function resolveSideFromExercise(
  exercise: ExercisePrescriptionOption,
  catalogMatch?: (typeof CLINICAL_EXERCISES)[number],
): PostureSide {
  if (exercise.prescribed_side === 'LEFT') return 'left';
  if (exercise.prescribed_side === 'RIGHT') return 'right';
  if (exercise.prescribed_side === 'BOTH') return 'both';

  const rawCode = (
    exercise.effective_code ||
    exercise.catalog_exercise_id ||
    exercise.exercise_id ||
    ''
  )
    .trim()
    .toUpperCase();
  if (
    rawCode.endsWith('_LEFT') ||
    rawCode.endsWith('_RIGHT') ||
    rawCode.endsWith('_BOTH') ||
    rawCode.endsWith('_L') ||
    rawCode.endsWith('_R') ||
    rawCode === 'MOPH_NECK_TURN_BILATERAL'
  ) {
    const parsedSide = parseExerciseCodeAndSide(rawCode).side;
    return parsedSide === 'LEFT' ? 'left' : parsedSide === 'BOTH' ? 'both' : 'right';
  }

  const text = `${exercise.exercise_id} ${exercise.name_th} ${exercise.name_en}`.toLowerCase();
  if (
    exercise.expected_plane === 'shrug_roll' ||
    exercise.expected_plane === 'bilateral' ||
    exercise.expected_plane === 'bilateral_push' ||
    exercise.expected_plane === 'neck_turn' ||
    text.includes('ทั้งสองข้าง') ||
    text.includes('ทั้ง 2 ข้าง') ||
    text.includes('2 ข้าง') ||
    text.includes('ประสานมือ') ||
    text.includes('ซ้าย-ขวา') ||
    text.includes('bilateral') ||
    text.includes('both') ||
    text.includes('ลำตัวตรง')
  ) {
    return 'both';
  }
  if (exercise.target_joint === 'left_shoulder') {
    return 'left';
  }
  if (text.includes('ซ้าย') || text.includes('left')) {
    return 'left';
  }
  if (catalogMatch?.detection.side === 'both') return 'both';
  if (catalogMatch?.detection.side === 'left') return 'left';
  if (catalogMatch?.detection.side === 'right') return 'right';
  return 'right';
}

export function resolveExercisePostureSpec(
  exercise: ExercisePrescriptionOption,
  lang: ClinicalLang = 'th',
): ExercisePostureSpec {
  const rawMatchId =
    exercise.base_code ||
    exercise.catalog_exercise_id ||
    exercise.effective_code ||
    exercise.exercise_id;
  const parsedBase = parseExerciseCodeAndSide(rawMatchId, exercise.prescribed_side).baseCode;
  const catalogMatch = CLINICAL_EXERCISES.find(
    (c) =>
      c.id === parsedBase ||
      c.code === parsedBase ||
      c.id === exercise.catalog_exercise_id ||
      c.code === exercise.catalog_exercise_id ||
      c.id === exercise.exercise_id ||
      c.code === exercise.exercise_id,
  );

  const side = resolveSideFromExercise(exercise, catalogMatch);
  const combined = `${exercise.exercise_id} ${exercise.catalog_exercise_id ?? ''} ${exercise.name_th} ${exercise.name_en} ${exercise.target_joint}`.toLowerCase();

  let kind: PostureIllustrationKind = 'shoulder_flexion';
  const group = catalogMatch?.detection.jointGroup;
  const plane = exercise.expected_plane ?? catalogMatch?.detection.expectedPlane;

  if (plane === 'neck_turn' || combined.includes('หันคอ') || combined.includes('neck_turn')) {
    kind = 'neck_turn';
  } else if (group === 'neck' || combined.includes('เอียงคอ') || combined.includes('คอและบ่า') || combined.includes('neck')) {
    kind = 'neck_tilt';
  } else if (group === 'trunk' || combined.includes('ลำตัวตรง') || combined.includes('posture alignment')) {
    kind = 'trunk_align';
  } else if (plane === 'thumb_pinch' || combined.includes('จีบนิ้ว') || combined.includes('thumb_pinch') || combined.includes('pinch')) {
    kind = 'thumb_pinch';
  } else if (plane === 'forearm_turn' || combined.includes('คว่ำ') || combined.includes('หงายมือ') || combined.includes('forearm_turn')) {
    kind = 'forearm_turn';
  } else if (group === 'wrist' || plane === 'wrist_finger' || combined.includes('ข้อมือ') || combined.includes('นิ้วมือ') || combined.includes('wrist')) {
    kind = 'wrist_finger';
  } else if (plane === 'bilateral_push' || combined.includes('forward_push') || combined.includes('ดันแขนตรงไปด้านหน้า')) {
    kind = 'bobath_forward_push';
  } else if (plane === 'hand_to_head' || combined.includes('แตะศีรษะ') || combined.includes('หวีผม') || combined.includes('hand_to_head')) {
    kind = 'hand_to_head';
  } else if (group === 'elbow' || combined.includes('ข้อศอก') || combined.includes('elb_flex') || combined.includes('elbow')) {
    kind = 'elbow_flexion';
  } else if (plane === 'shrug_roll' || combined.includes('ยักไหล่') || combined.includes('หมุนหัวไหล่') || combined.includes('shrug')) {
    kind = 'shoulder_shrug_roll';
  } else if (plane === 'bilateral' || combined.includes('ประสานมือยกแขน') || combined.includes('arm_raise') || combined.includes('bilateral')) {
    kind = 'bilateral_arm_raise';
  } else if (plane === 'scap_protract' || combined.includes('scap_protract') || combined.includes('ยื่นมือเอื้อมไปด้านหน้า') || combined.includes('protract')) {
    kind = 'scapular_protract';
  } else if (plane === 'cross_body' || combined.includes('cross_body') || combined.includes('ไขว้หน้าอก')) {
    kind = 'cross_body_reach';
  } else if (plane === 'rotation' || combined.includes('หมุนข้อไหล่') || combined.includes('sh_rot') || combined.includes('rot')) {
    kind = 'shoulder_rotation';
  } else if (combined.includes('overhead') || combined.includes('เหนือศีรษะ')) {
    kind = 'shoulder_overhead';
  } else if (combined.includes('scaption') || combined.includes('ทแยง') || combined.includes('45°')) {
    kind = 'shoulder_scaption';
  } else if (plane === 'abduction' || combined.includes('กางแขน') || combined.includes('abduct') || combined.includes('sh_abd')) {
    kind = 'shoulder_abduction';
  } else {
    kind = 'shoulder_flexion';
  }

  const meta = REGION_COPY[kind];
  const colorMap: Record<PostureSide, { hex: string; bg: string; border: string; badge: Record<ClinicalLang, string> }> = {
    right: {
      hex: '#10B981',
      bg: 'bg-emerald-50 text-emerald-800',
      border: 'border-emerald-300',
      badge: { th: 'ข้างขวา (R)', en: 'Right Side (R)', cn: '右侧 (R)' },
    },
    left: {
      hex: '#F97316',
      bg: 'bg-orange-50 text-orange-800',
      border: 'border-orange-300',
      badge: { th: 'ข้างซ้าย (L)', en: 'Left Side (L)', cn: '左侧 (L)' },
    },
    both: {
      hex: '#0284C7',
      bg: 'bg-sky-50 text-[#0F4C81]',
      border: 'border-sky-300',
      badge: { th: 'ทั้ง 2 ข้าง / แกนกลาง', en: 'Both Sides / Core', cn: '双侧 / 核心' },
    },
  };

  const palette = colorMap[side];
  const rawRom = Math.max(15, Math.round(exercise.target_rom_deg || 120));
  const rawHold = Number(exercise.hold_sec || 3);

  return {
    kind,
    side,
    clinicalFocus: catalogMatch?.clinicalFocus,
    bodyRegionLabel: meta.region[lang],
    viewAngleLabel: meta.view[lang],
    sideBadgeLabel: palette.badge[lang],
    muscleFocus: catalogMatch?.instruction?.[lang] || meta.muscle[lang],
    adlGoal: meta.adl?.[lang],
    cautionCue: meta.caution[lang],
    steps: meta.steps(side, rawRom, rawHold)[lang],
    primaryColor: palette.hex,
    softBgClass: palette.bg,
    borderClass: palette.border,
  };
}

export interface UpperLimbFkInput {
  shoulderX: number;
  shoulderY: number;
  upperArmLen: number;
  forearmLen: number;
  handLen?: number;
  angleDeg: number;
  elbowBendDeg?: number;
  dirSign?: 1 | -1;
}

export interface UpperLimbFkJoints {
  shoulder: { x: number; y: number };
  elbow: { x: number; y: number };
  wrist: { x: number; y: number };
  hand: { x: number; y: number };
}

/**
 * Hierarchical Forward Kinematics for Clinical Upper-Limb Joint Visualization:
 * 0° = resting vertically straight down beside torso (6 o'clock: sin=0, cos=1).
 * Positive angleDeg sweeps forward (Side-Profile dirSign=+1) or outward (Coronal dirSign=±1) and upward.
 */
export function computeUpperLimbFkJoints(input: UpperLimbFkInput): UpperLimbFkJoints {
  const dir = input.dirSign ?? 1;
  const rad = (input.angleDeg * Math.PI) / 180;
  const elbowBend = input.elbowBendDeg ?? 0;
  const forearmRad = ((input.angleDeg + elbowBend) * Math.PI) / 180;
  const handLen = input.handLen ?? 9;

  const elbowX = input.shoulderX + dir * Math.sin(rad) * input.upperArmLen;
  const elbowY = input.shoulderY + Math.cos(rad) * input.upperArmLen;

  const wristX = elbowX + dir * Math.sin(forearmRad) * input.forearmLen;
  const wristY = elbowY + Math.cos(forearmRad) * input.forearmLen;

  const handX = wristX + dir * Math.sin(forearmRad) * handLen;
  const handY = wristY + Math.cos(forearmRad) * handLen;

  return {
    shoulder: { x: input.shoulderX, y: input.shoulderY },
    elbow: { x: elbowX, y: elbowY },
    wrist: { x: wristX, y: wristY },
    hand: { x: handX, y: handY },
  };
}

function easeInOutCubic(t: number): number {
  return t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2;
}

export interface PostureCycleState {
  progress: number;
  stage: 'prepare' | 'move' | 'hold' | 'lower';
}

/**
 * Deterministic 4-phase anatomical exercise timing:
 * 1. Prepare (0° rest) -> 2. Smooth Move (0 -> targetRom) -> 3. Peak Hold (holdSec) -> 4. Smooth Lower.
 */
export function computePostureCycleState(
  phase: PosturePreviewPhase,
  elapsedMs: number,
  holdSec = 3,
): PostureCycleState {
  if (phase === 'prepare') return { progress: 0, stage: 'prepare' };
  if (phase === 'move') return { progress: 0.62, stage: 'move' };
  if (phase === 'hold') return { progress: 1, stage: 'hold' };

  const prepMs = 700;
  const liftMs = 1350;
  const holdMs = Math.max(1200, Math.min(3500, holdSec * 1000));
  const lowerMs = 1150;
  const totalMs = prepMs + liftMs + holdMs + lowerMs;
  const t = ((elapsedMs % totalMs) + totalMs) % totalMs;

  if (t < prepMs) {
    return { progress: 0, stage: 'prepare' };
  }
  if (t < prepMs + liftMs) {
    const u = (t - prepMs) / liftMs;
    return { progress: easeInOutCubic(u), stage: 'move' };
  }
  if (t < prepMs + liftMs + holdMs) {
    return { progress: 1, stage: 'hold' };
  }
  const d = (t - (prepMs + liftMs + holdMs)) / lowerMs;
  return { progress: 1 - easeInOutCubic(d), stage: 'lower' };
}

function buildRomGuideArc(
  cx: number,
  cy: number,
  radius: number,
  targetDeg: number,
  dirSign: 1 | -1,
): { d: string; endX: number; endY: number; tipDx: number; tipDy: number } {
  const clamped = Math.max(8, Math.min(176, targetDeg));
  const rad = (clamped * Math.PI) / 180;
  const startX = cx;
  const startY = cy + radius;
  const endX = cx + dirSign * Math.sin(rad) * radius;
  const endY = cy + Math.cos(rad) * radius;
  const sweepFlag = dirSign === 1 ? 0 : 1;
  // Tangent unit vector at arc end toward increasing angleDeg
  const tanX = dirSign * Math.cos(rad);
  const tanY = -Math.sin(rad);
  return {
    d: `M ${startX.toFixed(1)} ${startY.toFixed(1)} A ${radius} ${radius} 0 0 ${sweepFlag} ${endX.toFixed(1)} ${endY.toFixed(1)}`,
    endX,
    endY,
    tipDx: tanX,
    tipDy: tanY,
  };
}

function CapsuleArmLimb({
  joints,
  color,
  isActive,
  isGhost = false,
  compact = false,
  wristWidth = 10,
}: {
  joints: UpperLimbFkJoints;
  color: string;
  isActive: boolean;
  isGhost?: boolean;
  compact?: boolean;
  wristWidth?: number;
}) {
  const upperW = compact ? 14 : 12;
  const foreW = compact ? 12 : wristWidth;
  if (isGhost) {
    return (
      <g opacity={0.34}>
        <line
          x1={joints.shoulder.x}
          y1={joints.shoulder.y}
          x2={joints.elbow.x}
          y2={joints.elbow.y}
          stroke={color}
          strokeWidth={upperW}
          strokeLinecap="round"
          strokeDasharray="4 4"
        />
        <line
          x1={joints.elbow.x}
          y1={joints.elbow.y}
          x2={joints.wrist.x}
          y2={joints.wrist.y}
          stroke={color}
          strokeWidth={foreW}
          strokeLinecap="round"
          strokeDasharray="4 4"
        />
        <circle cx={joints.wrist.x} cy={joints.wrist.y} r={5} fill="none" stroke={color} strokeWidth={2} strokeDasharray="3 2" />
      </g>
    );
  }

  return (
    <g opacity={isActive ? 1 : 0.72}>
      {/* Upper Arm Capsule */}
      <line
        x1={joints.shoulder.x}
        y1={joints.shoulder.y}
        x2={joints.elbow.x}
        y2={joints.elbow.y}
        stroke={color}
        strokeWidth={upperW}
        strokeLinecap="round"
      />
      {/* Forearm Capsule */}
      <line
        x1={joints.elbow.x}
        y1={joints.elbow.y}
        x2={joints.wrist.x}
        y2={joints.wrist.y}
        stroke={color}
        strokeWidth={foreW}
        strokeLinecap="round"
      />
      {/* Thin White Skeletal Core Lines */}
      {isActive && !compact && (
        <>
          <line
            x1={joints.shoulder.x}
            y1={joints.shoulder.y}
            x2={joints.elbow.x}
            y2={joints.elbow.y}
            stroke="#FFFFFF"
            strokeWidth={2.2}
            strokeLinecap="round"
            opacity={0.9}
          />
          <line
            x1={joints.elbow.x}
            y1={joints.elbow.y}
            x2={joints.wrist.x}
            y2={joints.wrist.y}
            stroke="#FFFFFF"
            strokeWidth={2}
            strokeLinecap="round"
            opacity={0.9}
          />
        </>
      )}
      {/* Articulated Palm */}
      <line
        x1={joints.wrist.x}
        y1={joints.wrist.y}
        x2={joints.hand.x}
        y2={joints.hand.y}
        stroke={color}
        strokeWidth={isActive ? 7.5 : 6}
        strokeLinecap="round"
      />
      {/* Clinical Joint Pivot Circles: Shoulder, Elbow, Wrist */}
      <circle
        cx={joints.shoulder.x}
        cy={joints.shoulder.y}
        r={isActive ? 5.5 : 4}
        fill="#FFFFFF"
        stroke={color}
        strokeWidth={2.5}
      />
      <circle
        cx={joints.elbow.x}
        cy={joints.elbow.y}
        r={isActive ? 4.5 : 3.2}
        fill="#FFFFFF"
        stroke={color}
        strokeWidth={2.2}
      />
      <circle
        cx={joints.hand.x}
        cy={joints.hand.y}
        r={isActive ? 4.2 : 3}
        fill={isActive ? color : '#94A3B8'}
        stroke="#FFFFFF"
        strokeWidth={1.5}
      />
    </g>
  );
}

export function ExercisePostureSvg({
  exercise,
  phase = 'auto',
  compact = false,
}: {
  exercise: ExercisePrescriptionOption;
  phase?: PosturePreviewPhase;
  compact?: boolean;
}) {
  const { lang } = useClinicalI18n();
  const spec = useMemo(() => resolveExercisePostureSpec(exercise, lang), [exercise, lang]);
  const accent = spec.primaryColor;
  const mutedBody = '#0F4C81';
  const mutedLimb = '#CBD5E1';
  const activeIsLeft = spec.side === 'left';
  const targetRom = Math.min(172, Math.max(20, Math.round(exercise.target_rom_deg || 128)));
  const holdSec = Math.max(1, Number(exercise.hold_sec || 3));

  const [cycle, setCycle] = useState<PostureCycleState>(() =>
    computePostureCycleState(phase, 0, holdSec),
  );

  useEffect(() => {
    if (phase !== 'auto') {
      setCycle(computePostureCycleState(phase, 0, holdSec));
      return;
    }
    let rafId = 0;
    const startAt = performance.now();
    let lastRenderAt = 0;
    const minIntervalMs = compact ? 50 : 16;

    const tick = (now: number) => {
      if (now - lastRenderAt >= minIntervalMs) {
        lastRenderAt = now;
        setCycle(computePostureCycleState('auto', now - startAt, holdSec));
      }
      rafId = requestAnimationFrame(tick);
    };
    rafId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafId);
  }, [phase, holdSec, compact]);

  const p = cycle.progress;

  const renderSideSagittalView = (mode: 'flexion' | 'scapular_protract' | 'bobath_forward_push') => {
    const faceDir: 1 | -1 = activeIsLeft ? -1 : 1;
    const isProtract = mode === 'scapular_protract';
    const isPush = mode === 'bobath_forward_push';
    const protractShiftX = isProtract ? p * 18 : 0;
    const shoulderX = 88 + protractShiftX;
    const shoulderY = 82;
    const currentAngleDeg = isProtract
      ? 88
      : isPush
        ? 26 + p * 64
        : p * targetRom;
    const activeElbowBend = isProtract
      ? 4
      : isPush
        ? -(104 * (1 - p * 0.94))
        : 8;

    const activeJoints = computeUpperLimbFkJoints({
      shoulderX,
      shoulderY,
      upperArmLen: 38,
      forearmLen: 34,
      handLen: 9,
      angleDeg: currentAngleDeg,
      elbowBendDeg: activeElbowBend,
      dirSign: 1,
    });
    const backJoints = computeUpperLimbFkJoints({
      shoulderX: shoulderX - 5,
      shoulderY: shoulderY - 2,
      upperArmLen: 38,
      forearmLen: 34,
      handLen: 9,
      angleDeg: currentAngleDeg,
      elbowBendDeg: activeElbowBend,
      dirSign: 1,
    });
    const ghostJoints = computeUpperLimbFkJoints({
      shoulderX: isProtract ? 106 : 88,
      shoulderY,
      upperArmLen: 38,
      forearmLen: 34,
      handLen: 9,
      angleDeg: isProtract || isPush ? 90 : targetRom,
      elbowBendDeg: isProtract || isPush ? 4 : activeElbowBend,
      dirSign: 1,
    });
    const arc = buildRomGuideArc(88, shoulderY, 50, targetRom, 1);

    const readoutDeg = isProtract
      ? Math.round(p * Math.min(95, targetRom))
      : isPush
        ? Math.round(55 + p * (Math.min(170, targetRom) - 55))
        : Math.round(currentAngleDeg);

    return (
      <g>
        <g transform={faceDir === -1 ? 'translate(200, 0) scale(-1, 1)' : undefined}>
          {/* Ground floor baseline */}
          <line x1="28" y1="186" x2="174" y2="186" stroke="#CBD5E1" strokeWidth="2.5" strokeLinecap="round" />

          {/* Standing Side-Profile Legs (Natural Parallel Profile, NO splayed sticks) */}
          <g>
            <line x1="82" y1="132" x2="77" y2="182" stroke={mutedLimb} strokeWidth="11" strokeLinecap="round" />
            <line x1="77" y1="182" x2="86" y2="182" stroke={mutedLimb} strokeWidth="7" strokeLinecap="round" />
            <line x1="86" y1="132" x2="88" y2="158" stroke="#1E3A5F" strokeWidth="12" strokeLinecap="round" />
            <line x1="88" y1="158" x2="87" y2="182" stroke="#1E3A5F" strokeWidth="10.5" strokeLinecap="round" />
            <line x1="85" y1="182" x2="98" y2="182" stroke="#1E3A5F" strokeWidth="7.5" strokeLinecap="round" />
          </g>

          {/* Torso + Neck + Clear Side-Profile Head (nose & chin facing forward) */}
          <g>
            {!compact && mode === 'flexion' && (
              <line x1="82" y1="36" x2="82" y2="136" stroke="#CBD5E1" strokeWidth="1.8" strokeDasharray="4 4" />
            )}

            {spec.side === 'both' ? (
              <g opacity={0.62}>
                <CapsuleArmLimb joints={backJoints} color={accent} isActive compact={compact} />
              </g>
            ) : (
              <line x1="83" y1="84" x2="81" y2="140" stroke={mutedLimb} strokeWidth="9" strokeLinecap="round" opacity={0.55} />
            )}

            {/* Side-Profile Anatomical Torso Capsule (rounded back + natural chest contour) */}
            <path
              d="M 74 78 C 72 95, 73 118, 76 134 C 82 138, 92 138, 95 132 C 98 116, 102 96, 98 76 C 94 69, 78 69, 74 78 Z"
              fill="#DBEAFE"
              stroke={mutedBody}
              strokeWidth="2.8"
              strokeLinejoin="round"
            />

            {/* Side Neck Capsule */}
            <line x1="85" y1="62" x2="86" y2="75" stroke="#BAE6FD" strokeWidth="11" strokeLinecap="round" />
            <line x1="85" y1="62" x2="86" y2="75" stroke={mutedBody} strokeWidth="2.6" strokeLinecap="round" opacity={0.4} />

            {/* Clear Side-Profile Head with Forehead, Nose, Chin */}
            <path
              d="M 73 46 C 72 34, 82 28, 92 30 C 98 32, 100 38, 100 42 L 105 47 C 106 49, 103 50, 100 50 L 99 54 C 99 58, 94 61, 86 61 C 78 61, 73 54, 73 46 Z"
              fill="#FFFFFF"
              stroke={mutedBody}
              strokeWidth="2.8"
              strokeLinejoin="round"
            />
            <circle cx="93" cy="41" r="2" fill={mutedBody} />
            <path d="M 81 44 C 83 44, 83 50, 81 50" fill="none" stroke="#94A3B8" strokeWidth="1.8" strokeLinecap="round" />

            {/* ROM Guide Arc + Ghost Target Arm */}
            {!compact && mode === 'flexion' && (
              <>
                <CapsuleArmLimb joints={ghostJoints} color={accent} isActive isGhost />
                <path d={arc.d} fill="none" stroke={accent} strokeWidth="2.6" strokeDasharray="5 4" opacity="0.88" />
                <circle cx={arc.endX} cy={arc.endY} r="3.5" fill={accent} />
              </>
            )}

            {/* Protraction & Bobath Forward Push Linear Guides */}
            {!compact && (isProtract || isPush) && (
              <>
                <CapsuleArmLimb joints={ghostJoints} color={accent} isActive isGhost />
                <path
                  d="M 118 68 L 166 68 M 158 62 L 166 68 L 158 74"
                  fill="none"
                  stroke={accent}
                  strokeWidth="2.6"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeDasharray={isPush ? undefined : '4 3'}
                />
              </>
            )}

            {/* Active Capsule Limb driven cleanly by Hierarchical Forward Kinematics */}
            <CapsuleArmLimb
              joints={activeJoints}
              color={accent}
              isActive
              compact={compact}
            />

            {/* Clasped Hands Ring for Bobath Forward Push */}
            {isPush && (
              <circle
                cx={activeJoints.hand.x}
                cy={activeJoints.hand.y}
                r="6.5"
                fill="#FFFFFF"
                fillOpacity="0.85"
                stroke={accent}
                strokeWidth="2.4"
              />
            )}
          </g>
        </g>

        {/* Upright Non-Mirrored Live Angle Pill */}
        {!compact && (
          <g>
            <rect
              x={faceDir === -1 ? 16 : 136}
              y="14"
              width="48"
              height="22"
              rx="11"
              fill="#FFFFFF"
              stroke={accent}
              strokeWidth="1.8"
            />
            <text
              x={faceDir === -1 ? 40 : 160}
              y="29"
              textAnchor="middle"
              fontSize="11"
              fontWeight="800"
              fill={mutedBody}
            >
              {readoutDeg}°
            </text>
          </g>
        )}
      </g>
    );
  };

  const renderCoronalFrontUpperBody = () => {
    const leftShX = 72;
    const rightShX = 128;
    const shoulderY = 82;

    const isBilateral = spec.kind === 'bilateral_arm_raise' || spec.side === 'both';
    const isShrug = spec.kind === 'shoulder_shrug_roll';
    const isElbow = spec.kind === 'elbow_flexion';
    const isRot = spec.kind === 'shoulder_rotation';
    const isTrunk = spec.kind === 'trunk_align';
    const isCrossBody = spec.kind === 'cross_body_reach';
    const isHandToHead = spec.kind === 'hand_to_head';

    const shrugDropOffset = isShrug ? -9 * p : 0;
    const curShY = shoulderY + shrugDropOffset;

    const resolvedPeakDeg =
      spec.kind === 'shoulder_overhead' || isBilateral
        ? Math.min(168, Math.max(125, targetRom))
        : spec.kind === 'shoulder_scaption'
          ? Math.min(145, targetRom)
          : targetRom;

    const noPrimarySwing = isShrug || isElbow || isRot || isTrunk || isCrossBody || isHandToHead;
    const rAngle = isBilateral && !noPrimarySwing
      ? p * resolvedPeakDeg
      : !activeIsLeft && !noPrimarySwing
        ? p * resolvedPeakDeg
        : 6;
    const lAngle = isBilateral && !noPrimarySwing
      ? p * resolvedPeakDeg
      : activeIsLeft && !noPrimarySwing
        ? p * resolvedPeakDeg
        : 6;

    const rElbowBend = isElbow && (!activeIsLeft || isBilateral) ? -p * 145 : !activeIsLeft || isBilateral ? 6 : 4;
    const lElbowBend = isElbow && (activeIsLeft || isBilateral) ? -p * 145 : activeIsLeft || isBilateral ? 6 : 4;

    const rightJoints = computeUpperLimbFkJoints({
      shoulderX: rightShX,
      shoulderY: curShY,
      upperArmLen: 36,
      forearmLen: 32,
      handLen: 9,
      angleDeg: isRot && (!activeIsLeft || isBilateral) ? 5 : rAngle,
      elbowBendDeg: isRot && (!activeIsLeft || isBilateral) ? 88 * (1 - p * 0.45) : rElbowBend,
      dirSign: 1,
    });
    const leftJoints = computeUpperLimbFkJoints({
      shoulderX: leftShX,
      shoulderY: curShY,
      upperArmLen: 36,
      forearmLen: 32,
      handLen: 9,
      angleDeg: isRot && (activeIsLeft || isBilateral) ? 5 : lAngle,
      elbowBendDeg: isRot && (activeIsLeft || isBilateral) ? 88 * (1 - p * 0.45) : lElbowBend,
      dirSign: -1,
    });

    // Override horizontal forearm sweep for shoulder external rotation
    if (isRot) {
      if (!activeIsLeft) {
        rightJoints.elbow = { x: rightShX + 2, y: shoulderY + 34 };
        rightJoints.wrist = { x: rightShX + 2 + p * 32, y: shoulderY + 36 };
        rightJoints.hand = { x: rightJoints.wrist.x + 8, y: rightJoints.wrist.y };
      } else {
        leftJoints.elbow = { x: leftShX - 2, y: shoulderY + 34 };
        leftJoints.wrist = { x: leftShX - 2 - p * 32, y: shoulderY + 36 };
        leftJoints.hand = { x: leftJoints.wrist.x - 8, y: leftJoints.wrist.y };
      }
    }

    // Override horizontal chest sweep for cross-body shoulder reach
    if (isCrossBody) {
      if (!activeIsLeft) {
        rightJoints.elbow = { x: rightShX - p * 22, y: shoulderY + 30 - p * 18 };
        rightJoints.wrist = { x: rightShX + 4 - p * 56, y: shoulderY + 56 - p * 46 };
        rightJoints.hand = { x: rightJoints.wrist.x - 6, y: rightJoints.wrist.y - 2 };
      } else {
        leftJoints.elbow = { x: leftShX + p * 22, y: shoulderY + 30 - p * 18 };
        leftJoints.wrist = { x: leftShX - 4 + p * 56, y: shoulderY + 56 - p * 46 };
        leftJoints.hand = { x: leftJoints.wrist.x + 6, y: leftJoints.wrist.y - 2 };
      }
    }

    // Override grooming upward reach for hand-to-head
    if (isHandToHead) {
      if (!activeIsLeft) {
        rightJoints.elbow = { x: rightShX + 8 + p * 26, y: shoulderY + 32 - p * 42 };
        rightJoints.wrist = { x: rightShX + 10 - p * 26, y: shoulderY + 62 - p * 102 };
        rightJoints.hand = { x: rightJoints.wrist.x - 6, y: rightJoints.wrist.y - 6 };
      } else {
        leftJoints.elbow = { x: leftShX - 8 - p * 26, y: shoulderY + 32 - p * 42 };
        leftJoints.wrist = { x: leftShX - 10 + p * 26, y: shoulderY + 62 - p * 102 };
        leftJoints.hand = { x: leftJoints.wrist.x + 6, y: leftJoints.wrist.y - 6 };
      }
    }

    const rGhostJoints = computeUpperLimbFkJoints({
      shoulderX: rightShX,
      shoulderY,
      upperArmLen: 36,
      forearmLen: 32,
      handLen: 9,
      angleDeg: resolvedPeakDeg,
      elbowBendDeg: 6,
      dirSign: 1,
    });
    const lGhostJoints = computeUpperLimbFkJoints({
      shoulderX: leftShX,
      shoulderY,
      upperArmLen: 36,
      forearmLen: 32,
      handLen: 9,
      angleDeg: resolvedPeakDeg,
      elbowBendDeg: 6,
      dirSign: -1,
    });

    const showRightGhost = (!activeIsLeft || isBilateral) && !noPrimarySwing;
    const showLeftGhost = (activeIsLeft || isBilateral) && !noPrimarySwing;
    const rightArc = buildRomGuideArc(rightShX, shoulderY, 46, resolvedPeakDeg, 1);
    const leftArc = buildRomGuideArc(leftShX, shoulderY, 46, resolvedPeakDeg, -1);

    const activeReadoutDeg = isElbow
      ? Math.round(165 - p * 120)
      : isRot
        ? Math.round(p * Math.min(90, targetRom))
        : isCrossBody
          ? Math.round(p * Math.min(115, targetRom))
          : isHandToHead
            ? Math.round(p * Math.min(135, targetRom))
            : Math.round(p * resolvedPeakDeg);

    return (
      <g>
        <line x1="28" y1="186" x2="172" y2="186" stroke="#CBD5E1" strokeWidth="2.5" strokeLinecap="round" />

        {/* Spine Alignment Axis */}
        {isTrunk && !compact && (
          <line x1="100" y1="18" x2="100" y2="184" stroke={accent} strokeWidth="2.5" strokeDasharray="5 4" />
        )}

        {/* Symmetrical Parallel Capsule Legs */}
        <line x1="89" y1="132" x2="88" y2="181" stroke="#1E3A5F" strokeWidth="11.5" strokeLinecap="round" />
        <line x1="82" y1="182" x2="91" y2="182" stroke="#1E3A5F" strokeWidth="7" strokeLinecap="round" />
        <line x1="111" y1="132" x2="112" y2="181" stroke="#1E3A5F" strokeWidth="11.5" strokeLinecap="round" />
        <line x1="109" y1="182" x2="118" y2="182" stroke="#1E3A5F" strokeWidth="7" strokeLinecap="round" />

        {/* Symmetrical Anatomical Torso Capsule */}
        <path
          d="M 72 78 C 72 72, 128 72, 128 78 L 120 132 C 118 136, 82 136, 80 132 Z"
          fill="#DBEAFE"
          stroke={mutedBody}
          strokeWidth="2.8"
          strokeLinejoin="round"
        />
        {/* Neck + Symmetrical Head */}
        <line x1="100" y1="60" x2="100" y2="75" stroke="#BAE6FD" strokeWidth="11" strokeLinecap="round" />
        <circle cx="100" cy="44" r="15" fill="#FFFFFF" stroke={mutedBody} strokeWidth="2.8" />
        <circle cx="94" cy="42" r="1.8" fill={mutedBody} />
        <circle cx="106" cy="42" r="1.8" fill={mutedBody} />
        <path d="M 95 49 Q 100 52 105 49" fill="none" stroke={mutedBody} strokeWidth="1.8" strokeLinecap="round" />

        {/* Shoulder Clavicle Bridge */}
        <line
          x1={leftShX}
          y1={curShY}
          x2={rightShX}
          y2={curShY}
          stroke={mutedBody}
          strokeWidth="7"
          strokeLinecap="round"
        />

        {/* Ghost Target Arms + ROM Arcs */}
        {!compact && showRightGhost && (
          <>
            <CapsuleArmLimb joints={rGhostJoints} color={accent} isActive isGhost />
            <path d={rightArc.d} fill="none" stroke={accent} strokeWidth="2.4" strokeDasharray="5 4" opacity="0.88" />
            <circle cx={rightArc.endX} cy={rightArc.endY} r="3.2" fill={accent} />
          </>
        )}
        {!compact && showLeftGhost && (
          <>
            <CapsuleArmLimb joints={lGhostJoints} color={accent} isActive isGhost />
            <path d={leftArc.d} fill="none" stroke={accent} strokeWidth="2.4" strokeDasharray="5 4" opacity="0.88" />
            <circle cx={leftArc.endX} cy={leftArc.endY} r="3.2" fill={accent} />
          </>
        )}

        {/* Active vs Inactive Arms via Forward Kinematics */}
        <CapsuleArmLimb
          joints={leftJoints}
          color={activeIsLeft || isBilateral || isShrug ? accent : mutedLimb}
          isActive={activeIsLeft || isBilateral || isShrug}
          compact={compact}
        />
        <CapsuleArmLimb
          joints={rightJoints}
          color={!activeIsLeft || isBilateral || isShrug ? accent : mutedLimb}
          isActive={!activeIsLeft || isBilateral || isShrug}
          compact={compact}
        />

        {/* Shrug Vertical Upward Cues */}
        {!compact && isShrug && (
          <>
            <path d="M 56 76 L 56 54 M 51 60 L 56 54 L 61 60" fill="none" stroke={accent} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
            <path d="M 144 76 L 144 54 M 139 60 L 144 54 L 149 60" fill="none" stroke={accent} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
          </>
        )}

        {/* Cross-Body Chest Sweep Guide */}
        {!compact && isCrossBody && (
          <path
            d={activeIsLeft ? 'M 68 96 Q 100 78 130 84' : 'M 132 96 Q 100 78 70 84'}
            fill="none"
            stroke={accent}
            strokeWidth="2.4"
            strokeDasharray="4 3"
          />
        )}

        {/* Hand-to-Head Grooming Crown Target */}
        {!compact && isHandToHead && (
          <path
            d={activeIsLeft ? 'M 54 114 Q 44 48 94 28' : 'M 146 114 Q 156 48 106 28'}
            fill="none"
            stroke={accent}
            strokeWidth="2.4"
            strokeDasharray="4 3"
          />
        )}

        {/* Live Angle Pill */}
        {!compact && !isTrunk && !isShrug && (
          <g>
            <rect x="142" y="14" width="46" height="22" rx="11" fill="#FFFFFF" stroke={accent} strokeWidth="1.8" />
            <text x="165" y="29" textAnchor="middle" fontSize="11" fontWeight="800" fill={mutedBody}>
              {activeReadoutDeg}°
            </text>
          </g>
        )}

        {/* Mirror Labels ONLY on Coronal Front View */}
        {!compact && (
          <>
            <text
              x="22"
              y="174"
              fontSize="10"
              fontWeight="800"
              fill={spec.side === 'left' || spec.side === 'both' ? '#EA580C' : '#94A3B8'}
            >
              L (ซ้าย)
            </text>
            <text
              x="140"
              y="174"
              fontSize="10"
              fontWeight="800"
              fill={spec.side === 'right' || spec.side === 'both' ? '#059669' : '#94A3B8'}
            >
              R (ขวา)
            </text>
          </>
        )}
      </g>
    );
  };

  const renderHandCloseUp = () => {
    const wristLiftDeg = 16 - p * 42;
    const spreadRatio = p;
    const renderOneHand = (tx: number, ty: number, scale: number, flipX: number, keyId: string) => (
      <g key={keyId} transform={`translate(${tx}, ${ty}) scale(${flipX * scale}, ${scale})`}>
        <line x1="-66" y1="26" x2="-12" y2="12" stroke="#1E3A5F" strokeWidth="16" strokeLinecap="round" />
        <line x1="-66" y1="26" x2="-12" y2="12" stroke="#BAE6FD" strokeWidth="8" strokeLinecap="round" />
        <circle cx="-12" cy="12" r="9" fill={accent} fillOpacity="0.2" stroke={accent} strokeWidth="2.6" />
        <g transform={`translate(-10, 12) rotate(${wristLiftDeg})`}>
          <rect x="0" y="-15" width="36" height="34" rx="12" fill={accent} fillOpacity="0.16" stroke={accent} strokeWidth="3" />
          {[-38, -14, -2, 10, 22].map((openAngle, i) => {
            const curFingerAngle = openAngle * spreadRatio + (1 - spreadRatio) * 52;
            const fx = i === 0 ? 12 : 34;
            const fy = i === 0 ? -12 : -9 + (i - 1) * 7.5;
            const len = i === 0 ? 22 : i === 4 ? 24 : 32;
            const fr = (curFingerAngle * Math.PI) / 180;
            return (
              <line
                key={i}
                x1={fx}
                y1={fy}
                x2={fx + Math.cos(fr) * len}
                y2={fy + Math.sin(fr) * len}
                stroke={accent}
                strokeWidth="5.5"
                strokeLinecap="round"
              />
            );
          })}
        </g>
        {!compact && (
          <path d="M 14 -38 Q 36 -40 46 -20" fill="none" stroke={accent} strokeWidth="2.5" strokeDasharray="4 3" />
        )}
      </g>
    );

    if (spec.side === 'both') {
      return (
        <g>
          {renderOneHand(64, 108, 0.56, -1, 'hand-l')}
          {renderOneHand(136, 108, 0.56, 1, 'hand-r')}
        </g>
      );
    }
    return renderOneHand(100, 108, 1, activeIsLeft ? -1 : 1, 'hand-single');
  };

  const renderForearmTurnCloseUp = () => {
    const rollNorm = Math.sin(p * Math.PI);
    const palmScaleY = 0.25 + 0.75 * Math.abs(Math.cos(p * Math.PI));
    const palmUp = p >= 0.5;
    const readoutDeg = Math.round(rollNorm * Math.min(85, targetRom));
    const renderOneForearm = (tx: number, ty: number, scale: number, flipX: number, keyId: string) => (
      <g key={keyId} transform={`translate(${tx}, ${ty}) scale(${flipX * scale}, ${scale})`}>
        <line x1="-68" y1="16" x2="-10" y2="12" stroke="#1E3A5F" strokeWidth="16" strokeLinecap="round" />
        <line x1="-68" y1="16" x2="-10" y2="12" stroke="#BAE6FD" strokeWidth="8" strokeLinecap="round" />
        <circle cx="-10" cy="12" r="8" fill="#FFFFFF" stroke={accent} strokeWidth="2.6" />
        <g transform={`translate(-2, 12) scale(1, ${palmScaleY.toFixed(2)})`}>
          <rect
            x="0"
            y="-15"
            width="38"
            height="30"
            rx="10"
            fill={palmUp ? accent : '#FFFFFF'}
            fillOpacity={palmUp ? 0.22 : 1}
            stroke={accent}
            strokeWidth="3"
          />
          {[38, 38, 36, 32].map((fx, i) => (
            <line
              key={i}
              x1={fx}
              y1={-10 + i * 7}
              x2={fx + 22}
              y2={-10 + i * 7}
              stroke={accent}
              strokeWidth="5"
              strokeLinecap="round"
            />
          ))}
          <line
            x1="12"
            y1={palmUp ? -14 : 14}
            x2="22"
            y2={palmUp ? -28 : 28}
            stroke={accent}
            strokeWidth="5.5"
            strokeLinecap="round"
          />
        </g>
        {!compact && (
          <path
            d="M 16 -44 A 34 34 0 0 1 16 44 M 10 38 L 16 44 L 24 38"
            fill="none"
            stroke={accent}
            strokeWidth="2.5"
            strokeDasharray="4 3"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        )}
      </g>
    );
    return (
      <g>
        {spec.side === 'both' ? (
          <>
            {renderOneForearm(64, 108, 0.56, -1, 'forearm-l')}
            {renderOneForearm(136, 108, 0.56, 1, 'forearm-r')}
          </>
        ) : (
          renderOneForearm(100, 108, 1, activeIsLeft ? -1 : 1, 'forearm-single')
        )}
        {!compact && (
          <g>
            <rect x="138" y="14" width="46" height="22" rx="11" fill="#FFFFFF" stroke={accent} strokeWidth="1.8" />
            <text x="161" y="29" textAnchor="middle" fontSize="11" fontWeight="800" fill={mutedBody}>
              {readoutDeg}°
            </text>
          </g>
        )}
      </g>
    );
  };

  const renderThumbPinchCloseUp = () => {
    const pinchClose = p;
    const thumbTipX = 30 + pinchClose * 12;
    const thumbTipY = -30 + pinchClose * 12;
    const indexTipX = 48 - pinchClose * 6;
    const indexTipY = -28 + pinchClose * 10;
    const isContact = pinchClose > 0.72;
    const readoutDeg = Math.round(pinchClose * Math.min(60, targetRom));

    const renderOnePinch = (tx: number, ty: number, scale: number, flipX: number, keyId: string) => (
      <g key={keyId} transform={`translate(${tx}, ${ty}) scale(${flipX * scale}, ${scale})`}>
        <line x1="-64" y1="24" x2="-12" y2="12" stroke="#1E3A5F" strokeWidth="15" strokeLinecap="round" />
        <line x1="-64" y1="24" x2="-12" y2="12" stroke="#BAE6FD" strokeWidth="7.5" strokeLinecap="round" />
        <rect x="-8" y="-14" width="36" height="32" rx="11" fill={accent} fillOpacity="0.16" stroke={accent} strokeWidth="2.8" />
        {[
          { y: -2, len: 28 },
          { y: 6, len: 26 },
          { y: 14, len: 21 },
        ].map((f, idx) => (
          <line
            key={idx}
            x1="28"
            y1={f.y}
            x2={28 + f.len}
            y2={f.y + idx * 2}
            stroke={mutedLimb}
            strokeWidth="4.8"
            strokeLinecap="round"
          />
        ))}
        <path
          d={`M 26 -9 Q 42 -24 ${indexTipX.toFixed(1)} ${indexTipY.toFixed(1)}`}
          fill="none"
          stroke={accent}
          strokeWidth="5.5"
          strokeLinecap="round"
        />
        <path
          d={`M 8 -13 Q 18 -30 ${thumbTipX.toFixed(1)} ${thumbTipY.toFixed(1)}`}
          fill="none"
          stroke={accent}
          strokeWidth="6"
          strokeLinecap="round"
        />
        <circle
          cx={(thumbTipX + indexTipX) / 2}
          cy={(thumbTipY + indexTipY) / 2}
          r={isContact ? 7.5 : 4}
          fill={isContact ? accent : '#FFFFFF'}
          fillOpacity={isContact ? 0.25 : 0.8}
          stroke={accent}
          strokeWidth="2.2"
        />
      </g>
    );

    return (
      <g>
        {spec.side === 'both' ? (
          <>
            {renderOnePinch(64, 114, 0.56, -1, 'pinch-l')}
            {renderOnePinch(136, 114, 0.56, 1, 'pinch-r')}
          </>
        ) : (
          renderOnePinch(96, 114, 1, activeIsLeft ? -1 : 1, 'pinch-single')
        )}
        {!compact && (
          <g>
            <rect x="138" y="14" width="46" height="22" rx="11" fill="#FFFFFF" stroke={accent} strokeWidth="1.8" />
            <text x="161" y="29" textAnchor="middle" fontSize="11" fontWeight="800" fill={mutedBody}>
              {readoutDeg}°
            </text>
          </g>
        )}
      </g>
    );
  };

  const renderNeckView = (mode: 'tilt' | 'turn') => {
    const isTurn = mode === 'turn';
    const tiltDir = activeIsLeft ? -1 : 1;
    const tiltDeg = isTurn ? 0 : tiltDir * p * Math.min(35, targetRom);
    const yawPx = isTurn ? Math.sin(p * Math.PI * 2) * 10 : 0;
    const neckAngle = isTurn
      ? Math.round(Math.abs(Math.sin(p * Math.PI * 2)) * Math.min(55, targetRom))
      : Math.round(p * Math.min(35, targetRom));

    return (
      <g>
        {!compact && (
          <line x1="100" y1="18" x2="100" y2="174" stroke="#CBD5E1" strokeWidth="2" strokeDasharray="4 4" />
        )}
        <path
          d="M 56 104 C 56 96, 144 96, 144 104 L 136 176 L 64 176 Z"
          fill="#DBEAFE"
          stroke={mutedBody}
          strokeWidth="2.8"
          strokeLinejoin="round"
        />
        <line x1="56" y1="102" x2="144" y2="102" stroke={mutedBody} strokeWidth="9" strokeLinecap="round" />
        <circle cx="58" cy="102" r="5" fill="#64748B" />
        <circle cx="142" cy="102" r="5" fill="#64748B" />
        <g transform={`translate(100, 98) rotate(${tiltDeg})`}>
          <line x1="0" y1="0" x2="0" y2="-24" stroke={accent} strokeWidth="11" strokeLinecap="round" />
          <circle cx="0" cy="-42" r="20" fill="#FFFFFF" stroke={accent} strokeWidth="3.5" />
          <circle cx={-6 + yawPx} cy="-44" r="2.2" fill={mutedBody} />
          <circle cx={6 + yawPx} cy="-44" r="2.2" fill={mutedBody} />
          <path
            d={`M ${(-5 + yawPx).toFixed(1)} -36 Q ${(yawPx).toFixed(1)} -33 ${(5 + yawPx).toFixed(1)} -36`}
            fill="none"
            stroke={mutedBody}
            strokeWidth="2"
            strokeLinecap="round"
          />
        </g>
        {!compact && (
          <>
            {isTurn ? (
              <path
                d="M 64 40 Q 100 22 136 40 M 69 35 L 63 40 L 70 44 M 131 35 L 137 40 L 130 44"
                fill="none"
                stroke={accent}
                strokeWidth="2.6"
                strokeDasharray="4 3"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            ) : (
              <path
                d={activeIsLeft ? 'M 92 28 Q 66 32 56 54' : 'M 108 28 Q 134 32 144 54'}
                fill="none"
                stroke={accent}
                strokeWidth="2.8"
                strokeDasharray="4 3"
              />
            )}
            <g>
              <rect x="142" y="14" width="44" height="22" rx="11" fill="#FFFFFF" stroke={accent} strokeWidth="1.8" />
              <text x="164" y="29" textAnchor="middle" fontSize="11" fontWeight="800" fill={mutedBody}>
                {neckAngle}°
              </text>
            </g>
            <text x="22" y="176" fontSize="10" fontWeight="800" fill={activeIsLeft || isTurn ? '#EA580C' : '#94A3B8'}>
              L (ซ้าย)
            </text>
            <text x="140" y="176" fontSize="10" fontWeight="800" fill={!activeIsLeft || isTurn ? '#059669' : '#94A3B8'}>
              R (ขวา)
            </text>
          </>
        )}
      </g>
    );
  };

  return (
    <svg
      viewBox="0 0 200 200"
      className={compact ? 'h-12 w-12' : 'h-full w-full'}
      aria-label={exercise.name_th}
    >
      {!compact && (
        <circle cx="100" cy="100" r="90" fill="#F8FAFC" stroke="#E2E8F0" strokeWidth="1.5" />
      )}

      {spec.kind === 'wrist_finger'
        ? renderHandCloseUp()
        : spec.kind === 'forearm_turn'
          ? renderForearmTurnCloseUp()
          : spec.kind === 'thumb_pinch'
            ? renderThumbPinchCloseUp()
            : spec.kind === 'neck_tilt'
              ? renderNeckView('tilt')
              : spec.kind === 'neck_turn'
                ? renderNeckView('turn')
                : spec.kind === 'shoulder_flexion'
                  ? renderSideSagittalView('flexion')
                  : spec.kind === 'scapular_protract'
                    ? renderSideSagittalView('scapular_protract')
                    : spec.kind === 'bobath_forward_push'
                      ? renderSideSagittalView('bobath_forward_push')
                      : renderCoronalFrontUpperBody()}
    </svg>
  );
}

interface ExercisePostureGuideProps {
  exercise: ExercisePrescriptionOption;
  currentRomDeg?: number;
  targetRomDeg?: number;
  kioskState?: string;
}

const GUIDE_UI_TEXT: Record<
  ClinicalLang,
  {
    headerTag: string;
    autoLabel: string;
    step1: string;
    step2: string;
    step3: string;
    liveHoldHint: string;
    targetLabel: string;
  }
> = {
  th: {
    headerTag: 'ท่าทางประกอบตามท่ากายภาพบำบัด',
    autoLabel: 'เล่นอัตโนมัติ',
    step1: '1. เตรียม',
    step2: '2. ขยับ',
    step3: '3. ค้างท่า',
    liveHoldHint: 'ถึงมุมเป้าหมายแล้ว — ค้างท่านิ่งตามแบบ!',
    targetLabel: 'เป้าหมายมุม',
  },
  en: {
    headerTag: 'Animated Posture Guide for This Exercise',
    autoLabel: 'Auto Loop',
    step1: '1. Prepare',
    step2: '2. Move',
    step3: '3. Hold',
    liveHoldHint: 'Target angle reached — hold steady!',
    targetLabel: 'Target ROM',
  },
  cn: {
    headerTag: '康复动作标准姿势图解',
    autoLabel: '自动演示',
    step1: '1. 准备',
    step2: '2. 抬举',
    step3: '3. 保持',
    liveHoldHint: '已达目标角度 — 保持稳定！',
    targetLabel: '目标角度',
  },
};

export function ExercisePostureGuide({
  exercise,
  currentRomDeg,
  targetRomDeg,
  kioskState,
}: ExercisePostureGuideProps) {
  const { lang } = useClinicalI18n();
  const [previewPhase, setPreviewPhase] = useState<PosturePreviewPhase>('auto');
  const spec = useMemo(() => resolveExercisePostureSpec(exercise, lang), [exercise, lang]);
  const ui = GUIDE_UI_TEXT[lang];
  const resolvedRom = Math.round(targetRomDeg || exercise.target_rom_deg || 120);
  const isHoldingNow = kioskState === 'HOLDING' || kioskState === 'HOLD_PHASE';

  return (
    <section
      aria-label={ui.headerTag}
      data-testid="exercise-posture-guide"
      className="glass-card flex-1 flex flex-col justify-between rounded-3xl p-4 sm:p-5"
    >
      {/* Top Header Strip */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-sky-100 pb-3">
        <div className="flex flex-wrap items-center gap-2">
          <span className="inline-flex items-center gap-1.5 rounded-full bg-sky-100/90 px-3 py-1 text-[11px] font-extrabold text-[#0F4C81]">
            <Sparkles className="h-3.5 w-3.5 text-[#0284C7]" />
            {ui.headerTag}
          </span>
          <span
            className={`inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-[11px] font-extrabold ${spec.softBgClass} ${spec.borderClass}`}
          >
            {spec.sideBadgeLabel}
          </span>
          <span className="inline-flex items-center gap-1 rounded-full border border-slate-200 bg-white px-2.5 py-0.5 text-[11px] font-bold text-slate-700">
            {spec.bodyRegionLabel}
          </span>
        </div>

        <div className="flex items-center gap-2">
          {typeof currentRomDeg === 'number' && (
            <span className="inline-flex items-center gap-1 rounded-full border border-sky-200 bg-sky-50 px-2.5 py-1 text-[11px] font-extrabold text-[#0284C7] tabular-nums">
              <Target className="h-3.5 w-3.5" />
              {ui.targetLabel}: {Math.round(currentRomDeg)}° / {resolvedRom}°
            </span>
          )}
        </div>
      </div>

      {/* Main 2-Column Clinical Anatomic Mannequin + Step-by-Step Cues */}
      <div className="mt-3.5 flex-1 flex flex-col md:flex-row items-stretch gap-4">
        <div className="w-full md:w-[240px] lg:w-[260px] min-h-[220px] shrink-0 bg-gradient-to-b from-sky-50 to-emerald-50/40 rounded-2xl p-4 flex flex-col items-center justify-center border border-sky-200">
          <div className="relative flex w-40 h-40 md:w-48 md:h-48 items-center justify-center [&_svg]:w-full [&_svg]:h-full object-contain drop-shadow-sm">
            <ExercisePostureSvg exercise={exercise} phase={previewPhase} />
            {isHoldingNow && (
              <span className="absolute top-1 right-1 rounded-full bg-emerald-600 px-2 py-0.5 text-[10px] font-black text-white shadow animate-pulse">
                HOLD
              </span>
            )}
          </div>
          <span className="mt-2 text-center text-xs font-bold text-[#0F4C81]/80">
            {spec.viewAngleLabel}
          </span>
        </div>

        <div className="flex-1 flex flex-col justify-between gap-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <h3 className="text-base sm:text-lg font-extrabold text-[#0F4C81]">
                {lang === 'en' ? exercise.name_en : exercise.name_th}
              </h3>
              <p className="text-xs sm:text-sm font-medium text-slate-600">{spec.muscleFocus}</p>
            </div>
            <div className="inline-flex flex-wrap items-center gap-1 rounded-2xl border border-sky-200 bg-sky-50/80 p-1">
              {(
                [
                  { id: 'auto', label: ui.autoLabel, icon: previewPhase === 'auto' ? Pause : Play },
                  { id: 'prepare', label: ui.step1 },
                  { id: 'move', label: ui.step2 },
                  { id: 'hold', label: ui.step3 },
                ] as const
              ).map((tab) => {
                const active = previewPhase === tab.id;
                const Icon = 'icon' in tab ? tab.icon : null;
                return (
                  <button
                    key={tab.id}
                    type="button"
                    onClick={() => setPreviewPhase(tab.id)}
                    className={`inline-flex items-center gap-1 rounded-xl px-2.5 py-1 text-[11px] font-extrabold transition cursor-pointer ${
                      active
                        ? 'bg-[#0F4C81] text-white shadow-sm'
                        : 'text-[#0F4C81] hover:bg-white'
                    }`}
                  >
                    {Icon && <Icon className="h-3 w-3" />}
                    <span>{tab.label}</span>
                  </button>
                );
              })}
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 flex-1 items-stretch">
            {spec.steps.map((stepText, idx) => {
              const stepPhase: PosturePreviewPhase = idx === 0 ? 'prepare' : idx === 1 ? 'move' : 'hold';
              const highlighted = previewPhase === stepPhase || (idx === 2 && isHoldingNow);
              return (
                <button
                  key={idx}
                  type="button"
                  onClick={() => setPreviewPhase(previewPhase === stepPhase ? 'auto' : stepPhase)}
                  className={`h-full flex flex-col justify-start rounded-2xl border p-3.5 text-left transition cursor-pointer ${
                    highlighted
                      ? 'border-emerald-400 bg-emerald-50/90 shadow-sm ring-2 ring-emerald-200'
                      : 'border-sky-100 bg-white/85 hover:border-sky-300'
                  }`}
                >
                  <div className="flex items-center gap-2 text-xs sm:text-sm font-black text-[#0284C7]">
                    <span className="inline-flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-[#0F4C81] text-xs font-black text-white">
                      {idx + 1}
                    </span>
                    <span>{idx === 0 ? ui.step1 : idx === 1 ? ui.step2 : ui.step3}</span>
                  </div>
                  <p className="mt-2 text-sm md:text-base font-bold leading-snug text-slate-700">
                    {stepText}
                  </p>
                </button>
              );
            })}
          </div>

          <div className="flex flex-wrap items-center justify-between gap-2 rounded-2xl border border-emerald-200/80 bg-emerald-50/60 px-3 py-2 text-xs">
            <span className="inline-flex items-center gap-1.5 font-bold text-emerald-900">
              <ShieldCheck className="h-4 w-4 shrink-0 text-emerald-600" />
              {isHoldingNow ? ui.liveHoldHint : spec.cautionCue}
            </span>
            {spec.adlGoal && (
              <span className="inline-flex items-center gap-1 rounded-full bg-white px-2.5 py-0.5 text-[11px] font-extrabold text-[#0F4C81] border border-emerald-200">
                <CheckCircle2 className="h-3.5 w-3.5 text-emerald-600" />
                {spec.adlGoal}
              </span>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
