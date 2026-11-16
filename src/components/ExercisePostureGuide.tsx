import React, { useEffect, useMemo, useState } from 'react';
import { CLINICAL_EXERCISES, type ExerciseConfig } from '@drtechcares/pose-engine';
import { useClinicalI18n, type ClinicalLang } from '@drtechcares/clinical-i18n';
import { CheckCircle2, Pause, Play, ShieldCheck, Sparkles, Target } from 'lucide-react';
import type { ExercisePrescriptionOption } from '../hooks/useKioskSocket';

export type PostureIllustrationKind =
  | 'shoulder_flexion'
  | 'shoulder_abduction'
  | 'shoulder_scaption'
  | 'shoulder_overhead'
  | 'shoulder_wall_slide'
  | 'shoulder_pendulum'
  | 'shoulder_rotation'
  | 'shoulder_shrug_roll'
  | 'bilateral_arm_raise'
  | 'elbow_flexion'
  | 'wrist_finger'
  | 'neck_tilt'
  | 'trunk_align'
  | 'chair_squat'
  | 'knee_extension'
  | 'hip_march'
  | 'heel_raise'
  | 'single_leg_balance';

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
      const sTh = side === 'left' ? 'ซ้าย' : 'ขวา';
      const sEn = side === 'left' ? 'left' : 'right';
      const sCn = side === 'left' ? '左' : '右';
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
      const sTh = side === 'left' ? 'ซ้าย' : 'ขวา';
      const sEn = side === 'left' ? 'left' : 'right';
      const sCn = side === 'left' ? '左' : '右';
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
      const sTh = side === 'left' ? 'ซ้าย' : 'ขวา';
      const sEn = side === 'left' ? 'left' : 'right';
      const sCn = side === 'left' ? '左' : '右';
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
      const sTh = side === 'left' ? 'ซ้าย' : 'ขวา';
      const sEn = side === 'left' ? 'left' : 'right';
      const sCn = side === 'left' ? '左' : '右';
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
  shoulder_wall_slide: {
    region: {
      th: 'ข้อไหล่ & สะบัก (ใช้ผนังช่วยประคอง)',
      en: 'Shoulder & Scapula (Wall Support)',
      cn: '肩关节与肩胛（扶墙辅助）',
    },
    view: {
      th: 'มุมมองข้างผนัง (สไลด์มือไต่ขึ้นตามแนวตั้ง)',
      en: 'Wall-guided vertical slide',
      cn: '扶墙垂直向上滑举',
    },
    muscle: {
      th: 'ค่อยๆ เพิ่มพิสัยข้อไหล่โดยมีผนังช่วยพยุงน้ำหนักแขน',
      en: 'Wall-supported progressive overhead mobility',
      cn: '借助墙面支撑安全提升肩关节活动度',
    },
    caution: {
      th: 'ประคองมือให้เลียบผนังขึ้นช้าๆ ไม่เอนตัวถอยหลัง',
      en: 'Slide gently along wall; avoid leaning backward',
      cn: '轻贴墙面缓慢上滑，不要后仰身体',
    },
    steps: (side, rom, holdSec) => {
      const sTh = side === 'left' ? 'ซ้าย' : 'ขวา';
      const sEn = side === 'left' ? 'left' : 'right';
      const sCn = side === 'left' ? '左' : '右';
      return {
        th: [
          `ยืนห่างผนังเล็กน้อย วางฝ่ามือ${sTh}แตะเบาๆ ที่ผนัง`,
          `ค่อยๆ สไลด์มือ${sTh}ไต่ผนังขึ้นสูงสู่ ${rom}°`,
          `ค้างที่จุดสูงสุด ${holdSec} วินาที แล้วสไลด์กลับลงช้าๆ`,
        ],
        en: [
          `Stand near wall with ${sEn} palm lightly touching wall`,
          `Slide ${sEn} hand smoothly upward along wall toward ${rom}°`,
          `Hold at peak for ${holdSec}s, then slide back down slowly`,
        ],
        cn: [
          `面向墙站立，${sCn}手轻贴墙面`,
          `沿着墙面缓慢向上滑举至 ${rom}°`,
          `最高处保持 ${holdSec} 秒后缓慢滑下`,
        ],
      };
    },
  },
  shoulder_pendulum: {
    region: {
      th: 'ข้อไหล่ (แกว่งลดตึงแบบคอดแมน)',
      en: 'Relaxed Shoulder Joint (Codman)',
      cn: '肩关节放松（钟摆减压）',
    },
    view: {
      th: 'มุมมองด้านข้าง (โน้มตัวเล็กน้อยแล้วแกว่งเป็นวงกลม)',
      en: 'Side lean view (Circular hanging swing)',
      cn: '侧面微俯视角（自然下垂画圈摆动）',
    },
    muscle: {
      th: 'คลายความฝืดและเพิ่มน้ำหล่อเลี้ยงข้อไหล่โดยไม่เกร็งกล้ามเนื้อ',
      en: 'Gravity-assisted joint decompression without straining',
      cn: '利用重力自然牵引放松僵硬肩关节',
    },
    caution: {
      th: 'ใช้แขนอีกข้างยันโต๊ะ/เก้าอี้ไว้ให้มั่นคง ปล่อยแขนที่แกว่งให้หนักที่สุด',
      en: 'Support with opposite hand on chair/table; let swinging arm hang loose',
      cn: '健侧手扶稳桌椅，患侧手臂完全放松下垂',
    },
    steps: (side, _rom, holdSec) => {
      const sTh = side === 'left' ? 'ซ้าย' : 'ขวา';
      const otherTh = side === 'left' ? 'ขวา' : 'ซ้าย';
      const sEn = side === 'left' ? 'left' : 'right';
      const sCn = side === 'left' ? '左' : '右';
      return {
        th: [
          `โน้มลำตัวไปด้านหน้าเล็กน้อย ใช้มือ${otherTh}ยันเก้าอี้ประคอง`,
          `ปล่อยแขน${sTh}ห้อยตรงตามแรงโน้มถ่วง อย่าเกร็งหัวไหล่`,
          `แกว่งเบาๆ ต่อเนื่องสม่ำเสมอ ${holdSec} วินาที`,
        ],
        en: [
          `Lean forward gently resting opposite hand on support`,
          `Let your ${sEn} arm hang completely loose like a pendulum`,
          `Swing gently in small controlled arcs for ${holdSec}s`,
        ],
        cn: [
          `身体微前倾，健侧手轻扶椅子支撑稳定`,
          `让${sCn}臂如钟摆般完全放松垂下`,
          `轻柔摆动放松肩关节，持续 ${holdSec} 秒`,
        ],
      };
    },
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
      const sTh = side === 'left' ? 'ซ้าย' : 'ขวา';
      const sEn = side === 'left' ? 'left' : 'right';
      const sCn = side === 'left' ? '左' : '右';
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
      const sTh = side === 'left' ? 'ซ้าย' : 'ขวา';
      const sEn = side === 'left' ? 'left' : 'right';
      const sCn = side === 'left' ? '左' : '右';
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
      const sTh = side === 'left' ? 'ซ้าย' : 'ขวา';
      const sEn = side === 'left' ? 'left' : 'right';
      const sCn = side === 'left' ? '左' : '右';
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
      const sTh = side === 'left' ? 'ซ้าย' : 'ขวา';
      const sEn = side === 'left' ? 'left' : 'right';
      const sCn = side === 'left' ? '左' : '右';
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
  chair_squat: {
    region: {
      th: 'กล้ามเนื้อต้นขา • ข้อเข่า & สะโพก (Sit-to-Stand)',
      en: 'Quadriceps • Knees & Hips (Sit-to-Stand)',
      cn: '大腿股四头肌 • 膝髋关节（坐站训练）',
    },
    view: {
      th: 'มุมมองด้านข้างเก้าอี้ (ย่อ-เหยียดเข่าพร้อมลุกยืน)',
      en: 'Chair side view (Rise upright & lower smoothly)',
      cn: '座椅侧面视角（起立站直与缓慢坐下）',
    },
    muscle: {
      th: 'สร้างกำลังขาสองข้างเพื่อลุกจากเตียง เก้าอี้ และห้องน้ำได้ปลอดภัย',
      en: 'Lower-limb strength for safe chair, bed & toilet transfers',
      cn: '强化下肢站立力量，保障日常起坐安全',
    },
    caution: {
      th: 'เตรียมเก้าอี้มั่นคงด้านหลังเสมอ อย่าให้หัวเข่าบิดหุบเข้าด้านใน',
      en: 'Keep sturdy chair behind you; avoid knees caving inward',
      cn: '身后紧靠稳固座椅，膝盖勿向内扣',
    },
    adl: {
      th: '🪑 กิจวัตรประจำวัน: ลุกจากเก้าอี้ / เตียงนอน อย่างมั่นคง',
      en: '🪑 Daily ADL: Safe chair & bed transfer',
      cn: '🪑 日常活动：安全起坐转移',
    },
    steps: (_side, _rom, holdSec) => ({
      th: [
        'นั่งปลายเก้าอี้ แยกเท้ากว้างเท่าช่วงไหล่ วางเท้ามั่นคงบนพื้น',
        'โน้มตัวเล็กน้อย ดันส้นเท้าลุกยืนขึ้นช้าๆ จนเข่าเกือบตรง',
        `ยืนทรงตัวตรงค้างไว้ ${holdSec} วินาที แล้วค่อยๆ หย่อนสะโพกนั่งลง`,
      ],
      en: [
        'Sit near chair edge with feet flat shoulder-width apart',
        'Shift slightly forward and push through heels to stand tall',
        `Hold upright stance for ${holdSec}s, then sit back smoothly`,
      ],
      cn: [
        '坐在椅子前缘，双脚与肩同宽平踩地面',
        '身体微前倾，脚跟发力缓慢起立至膝关节伸直',
        `直立保持 ${holdSec} 秒后控制重心缓慢坐回`,
      ],
    }),
  },
  knee_extension: {
    region: {
      th: 'กล้ามเนื้อหน้าขา & ข้อเข่า (ท่านั่งเหยียดขา)',
      en: 'Quadriceps & Knee Joint (Seated Extension)',
      cn: '股四头肌与膝关节（坐姿伸膝）',
    },
    view: {
      th: 'มุมมองด้านข้างเก้าอี้ (เตะขาเหยียดเข่าตรงไปด้านหน้า)',
      en: 'Seated side view (Straighten lower leg horizontally)',
      cn: '坐姿侧面视角（小腿向前伸平）',
    },
    muscle: {
      th: 'เสริมความแข็งแรงรอบข้อเข่าเพื่อช่วยการก้าวเดินและขึ้นบันได',
      en: 'Builds knee-locking stability for safer walking & stairs',
      cn: '强化膝关节伸展稳定，辅助平稳迈步',
    },
    caution: {
      th: 'พิงพนักเก้าอี้ให้หลังตรง กระดกปลายเท้าเข้าหาตัวเล็กน้อยขณะเหยียด',
      en: 'Keep back supported; flex toes gently upward at full extension',
      cn: '背部靠稳椅背，伸直时脚尖微勾向自身',
    },
    steps: (side, _rom, holdSec) => {
      const sTh = side === 'left' ? 'ซ้าย' : 'ขวา';
      const sEn = side === 'left' ? 'left' : 'right';
      const sCn = side === 'left' ? '左' : '右';
      return {
        th: [
          `นั่งหลังตรงบนเก้าอี้ พักขา${sTh}งอเข่า 90° สบายๆ`,
          `ค่อยๆ เหยียดข้อเข่า${sTh}ยกปลายเท้าขึ้นจนขาขนานพื้น`,
          `เกร็งหน้าขาค้างไว้ ${holdSec} วินาที แล้วหย่อนเท้าลงช้าๆ`,
        ],
        en: [
          `Sit upright on chair with ${sEn} knee bent 90°`,
          `Slowly straighten your ${sEn} knee until shin is horizontal`,
          `Hold straight for ${holdSec}s, then lower foot slowly`,
        ],
        cn: [
          `端坐于椅子上，${sCn}膝自然弯曲 90°`,
          `缓慢伸直${sCn}膝使小腿与地面平行`,
          `伸直保持 ${holdSec} 秒后缓慢放下`,
        ],
      };
    },
  },
  hip_march: {
    region: {
      th: 'ข้อสะโพกด้านหน้า & กล้ามเนื้อพยุงก้าวเดิน',
      en: 'Hip Flexors & Single-Stance Marching',
      cn: '屈髋肌群与迈步抬腿控制',
    },
    view: {
      th: 'มุมมองหน้าตรง (ยกเข่าสูงระดับสะโพกสลับขา)',
      en: 'Front standing view (Lift knee to hip level)',
      cn: '正面站姿视角（抬膝至髋部高度）',
    },
    muscle: {
      th: 'ฝึกยกขาพ้นพื้นลดโอกาสสะดุดล้มในผู้สูงอายุและผู้ป่วยหลอดเลือดสมอง',
      en: 'Trains foot-clearance height to prevent tripping & falls',
      cn: '强化迈步抬脚高度，防止绊倒跌倒',
    },
    caution: {
      th: 'เตรียมราวจับหรือพนักเก้าอี้ข้างตัวเพื่อความปลอดภัยขณะยกขา',
      en: 'Keep chair/handrail nearby for balance safety while lifting knee',
      cn: '身旁放置椅子或扶手以确保安全平衡',
    },
    steps: (side, _rom, holdSec) => {
      const sTh = side === 'left' ? 'ซ้าย' : 'ขวา';
      const sEn = side === 'left' ? 'left' : 'right';
      const sCn = side === 'left' ? '左' : '右';
      return {
        th: [
          `ยืนตัวตรง แยกเท้าพอประมาณ ใช้มือแตะราวจับหากต้องการ`,
          `ค่อยๆ ยกเข่า${sTh}ขึ้นด้านหน้าจนต้นขาขนานพื้นหรือระดับสะโพก`,
          `ทรงตัวค้างไว้ ${holdSec} วินาที แล้ววางเท้าลงพื้นอย่างมั่นคง`,
        ],
        en: [
          'Stand tall with feet hip-width apart near support rail',
          `Raise your ${sEn} knee smoothly upward toward hip level`,
          `Hold steady for ${holdSec}s, then place foot softly down`,
        ],
        cn: [
          '双脚与肩同宽站立，必要时轻扶扶手',
          `缓慢抬起${sCn}膝至髋部水平高度`,
          `单腿稳定保持 ${holdSec} 秒后轻柔落地`,
        ],
      };
    },
  },
  heel_raise: {
    region: {
      th: 'ข้อเท้า & กล้ามเนื้อน่อง (Calf & Ankle)',
      en: 'Ankle Plantarflexors & Calf Strength',
      cn: '踝关节跖屈与小腿肌力',
    },
    view: {
      th: 'มุมมองด้านข้าง (เขย่งส้นเท้าขึ้นจากพื้นช้าๆ)',
      en: 'Side lower-leg view (Lift heel upward on toes)',
      cn: '小腿侧面视角（脚尖着地缓慢踮起脚跟）',
    },
    muscle: {
      th: 'เพิ่มแรงถีบปลายเท้าขณะเดินและกระตุ้นการไหลเวียนโลหิตช่วงขา',
      en: 'Improves walking push-off power & calf venous return',
      cn: '增强步行蹬伸推力并促进下肢静脉回流',
    },
    caution: {
      th: 'ถ่ายน้ำหนักบนจมูกเท้าอย่างมั่นคง มือแตะพนักเก้าอี้ได้เบาๆ',
      en: 'Rise smoothly onto ball of foot; hold chair lightly if needed',
      cn: '重心稳落前脚掌，必要时可轻扶椅背',
    },
    steps: (side, _rom, holdSec) => {
      const sTh = side === 'left' ? 'ซ้าย' : 'ขวา';
      const sEn = side === 'left' ? 'left' : 'right';
      const sCn = side === 'left' ? '左' : '右';
      return {
        th: [
          `ยืนตรง เท้า${sTh}วางราบพื้น ลำตัวไม่เอนไปด้านหน้า`,
          `ค่อยๆ เขย่งส้นเท้า${sTh}ขึ้นสูงสุดเท่าที่ทรงตัวได้`,
          `ค้างไว้ด้านบน ${holdSec} วินาที แล้ววางส้นเท้าลงช้าๆ`,
        ],
        en: [
          `Stand tall with ${sEn} foot flat on the floor`,
          `Smoothly press onto ball of ${sEn} foot lifting heel upward`,
          `Hold peak height for ${holdSec}s, then lower heel gently`,
        ],
        cn: [
          `身体站直，${sCn}脚平踩地面`,
          `缓慢踮起${sCn}脚跟至舒适最高处`,
          `最高处保持 ${holdSec} 秒后缓慢放回地面`,
        ],
      };
    },
  },
  single_leg_balance: {
    region: {
      th: 'สมดุลข้อสะโพก-เข่า-ข้อเท้า (Static Balance)',
      en: 'Hip-Knee-Ankle Single-Leg Stability',
      cn: '单腿静态平衡与核心控制',
    },
    view: {
      th: 'มุมมองหน้าตรง (ยืนทรงตัวขาเดียวแนวดิ่งตรง)',
      en: 'Front view (Balanced single-leg vertical axis)',
      cn: '正面视角（单腿垂直中轴稳定站立）',
    },
    muscle: {
      th: 'ฝึกการรับรู้ข้อต่อและการทรงตัวป้องกันการล้มขณะก้าวเดิน',
      en: 'Proprioceptive balance training reducing fall risk',
      cn: '训练本体感觉与防跌倒平衡控制',
    },
    caution: {
      th: 'มองจุดนิ่งตรงหน้าและยืนใกล้เก้าอี้พยุงไว้เสมอ เพื่อความปลอดภัย',
      en: 'Fix gaze ahead and keep chair support within arm reach',
      cn: '平视前方固定点，确保随时可触及椅子防跌倒',
    },
    steps: (side, _rom, holdSec) => {
      const stanceTh = side === 'left' ? 'ซ้าย' : 'ขวา';
      const liftTh = side === 'left' ? 'ขวา' : 'ซ้าย';
      const stanceEn = side === 'left' ? 'left' : 'right';
      const liftEn = side === 'left' ? 'right' : 'left';
      const stanceCn = side === 'left' ? '左' : '右';
      return {
        th: [
          `ยืนตัวตรงใกล้เก้าอี้ ค่อยๆ ถ่ายน้ำหนักลงบนขา${stanceTh}`,
          `ยกเท้า${liftTh}ขึ้นพ้นพื้นเล็กน้อย รักษาสะโพกไม่เอียง`,
          `นิ่งทรงตัวบนขา${stanceTh}อย่างมั่นคงต่อเนื่อง ${holdSec} วินาที`,
        ],
        en: [
          `Stand tall near support and shift weight smoothly to ${stanceEn} leg`,
          `Lift ${liftEn} foot slightly off floor while keeping hips level`,
          `Balance steadily on your ${stanceEn} leg for ${holdSec}s`,
        ],
        cn: [
          `站直并将身体重心平稳转移至${stanceCn}腿`,
          `微抬起另一侧脚离开地面，保持骨盆水平`,
          `使用${stanceCn}腿稳定单脚平衡站立 ${holdSec} 秒`,
        ],
      };
    },
  },
};

function resolveSideFromExercise(
  exercise: ExercisePrescriptionOption,
  catalogMatch?: (typeof CLINICAL_EXERCISES)[number],
): PostureSide {
  if (catalogMatch?.detection.side === 'both') return 'both';
  if (catalogMatch?.detection.side === 'left') return 'left';
  if (catalogMatch?.detection.side === 'right') return 'right';

  const text = `${exercise.exercise_id} ${exercise.name_th} ${exercise.name_en}`.toLowerCase();
  if (
    exercise.expected_plane === 'shrug_roll' ||
    exercise.expected_plane === 'bilateral' ||
    text.includes('ทั้งสองข้าง') ||
    text.includes('ทั้ง 2 ข้าง') ||
    text.includes('2 ข้าง') ||
    text.includes('ประสานมือ') ||
    text.includes('bilateral') ||
    text.includes('both') ||
    text.includes('chair_squat') ||
    text.includes('sit_to_stand') ||
    text.includes('ลุกนั่ง') ||
    text.includes('ลุกยืน') ||
    text.includes('ลำตัวตรง')
  ) {
    return 'both';
  }
  if (exercise.target_joint === 'left_shoulder' || exercise.target_joint === 'left_knee') {
    return 'left';
  }
  if (text.includes('ซ้าย') || text.includes('left') || text.endsWith('_l')) {
    return 'left';
  }
  return 'right';
}

export function resolveExercisePostureSpec(
  exercise: ExercisePrescriptionOption,
  lang: ClinicalLang = 'th',
): ExercisePostureSpec {
  const catalogMatch = CLINICAL_EXERCISES.find(
    (c) =>
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

  if (group === 'neck' || combined.includes('เอียงคอ') || combined.includes('คอและบ่า') || combined.includes('neck')) {
    kind = 'neck_tilt';
  } else if (group === 'trunk' || combined.includes('ลำตัวตรง') || combined.includes('trunk_sit') || combined.includes('posture alignment')) {
    kind = 'trunk_align';
  } else if (group === 'wrist' || plane === 'wrist_finger' || combined.includes('ข้อมือ') || combined.includes('นิ้วมือ') || combined.includes('wrist')) {
    kind = 'wrist_finger';
  } else if (group === 'elbow' || combined.includes('ข้อศอก') || combined.includes('elbow')) {
    kind = 'elbow_flexion';
  } else if (plane === 'shrug_roll' || combined.includes('ยักไหล่') || combined.includes('shrug')) {
    kind = 'shoulder_shrug_roll';
  } else if (plane === 'bilateral' || combined.includes('ประสานมือ') || combined.includes('bilateral')) {
    kind = 'bilateral_arm_raise';
  } else if (group === 'balance' || combined.includes('ทรงตัวขาเดียว') || combined.includes('bal_stand') || combined.includes('balance')) {
    kind = 'single_leg_balance';
  } else if (group === 'ankle' || combined.includes('เขย่ง') || combined.includes('heel')) {
    kind = 'heel_raise';
  } else if (group === 'hip' || combined.includes('ยกเข่าสูง') || combined.includes('hip_flex') || combined.includes('march')) {
    kind = 'hip_march';
  } else if (combined.includes('squat') || combined.includes('ลุกนั่ง') || combined.includes('ลุกยืน') || combined.includes('sit-to-stand') || combined.includes('sit_to_stand')) {
    kind = 'chair_squat';
  } else if (group === 'knee' || combined.includes('เหยียดเข่า') || combined.includes('knee_ext')) {
    kind = 'knee_extension';
  } else if (plane === 'rotation' || combined.includes('หมุนข้อไหล่') || combined.includes('rot')) {
    kind = 'shoulder_rotation';
  } else if (combined.includes('pendulum') || combined.includes('คอดแมน') || combined.includes('แกว่งแขน')) {
    kind = 'shoulder_pendulum';
  } else if (combined.includes('wall') || combined.includes('ไต่ผนัง') || combined.includes('สไลด์แขน')) {
    kind = 'shoulder_wall_slide';
  } else if (combined.includes('overhead') || combined.includes('เหนือศีรษะ') || combined.includes('sh_ovh')) {
    kind = 'shoulder_overhead';
  } else if (combined.includes('scaption') || combined.includes('ทแยง') || combined.includes('45°') || combined.includes('scap')) {
    kind = 'shoulder_scaption';
  } else if (plane === 'abduction' || combined.includes('กางแขน') || combined.includes('abduct') || combined.includes('sh_abd')) {
    kind = 'shoulder_abduction';
  } else if (exercise.target_joint === 'right_knee' || exercise.target_joint === 'left_knee') {
    kind = 'chair_squat';
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

  const renderSideSagittalView = (mode: 'flexion' | 'pendulum') => {
    const faceDir: 1 | -1 = activeIsLeft ? -1 : 1;
    const shoulderX = 88;
    const shoulderY = 82;
    const currentAngleDeg =
      mode === 'pendulum'
        ? (Math.sin(p * Math.PI * 2) * 18)
        : p * targetRom;
    const activeElbowBend = mode === 'pendulum' ? 3 : 8;

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
    const ghostJoints = computeUpperLimbFkJoints({
      shoulderX,
      shoulderY,
      upperArmLen: 38,
      forearmLen: 34,
      handLen: 9,
      angleDeg: targetRom,
      elbowBendDeg: activeElbowBend,
      dirSign: 1,
    });
    const arc = buildRomGuideArc(shoulderX, shoulderY, 50, targetRom, 1);

    const leanAngle = mode === 'pendulum' ? 18 : 0;

    return (
      <g>
        <g transform={faceDir === -1 ? 'translate(200, 0) scale(-1, 1)' : undefined}>
          {/* Ground floor baseline */}
          <line x1="28" y1="186" x2="174" y2="186" stroke="#CBD5E1" strokeWidth="2.5" strokeLinecap="round" />

          {/* Support Table/Chair for Codman Pendulum */}
          {mode === 'pendulum' && (
            <path
              d="M 134 186 L 134 126 L 166 126"
              fill="none"
              stroke="#94A3B8"
              strokeWidth="5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          )}

          {/* Standing Side-Profile Legs (Natural Parallel Profile, NO splayed sticks) */}
          <g>
            <line x1="82" y1="132" x2="77" y2="182" stroke={mutedLimb} strokeWidth="11" strokeLinecap="round" />
            <line x1="77" y1="182" x2="86" y2="182" stroke={mutedLimb} strokeWidth="7" strokeLinecap="round" />
            <line x1="86" y1="132" x2="88" y2="158" stroke="#1E3A5F" strokeWidth="12" strokeLinecap="round" />
            <line x1="88" y1="158" x2="87" y2="182" stroke="#1E3A5F" strokeWidth="10.5" strokeLinecap="round" />
            <line x1="85" y1="182" x2="98" y2="182" stroke="#1E3A5F" strokeWidth="7.5" strokeLinecap="round" />
          </g>

          {/* Torso + Neck + Clear Side-Profile Head (nose & chin facing forward) */}
          <g transform={leanAngle ? `rotate(${leanAngle}, 84, 132)` : undefined}>
            {!compact && mode === 'flexion' && (
              <line x1="82" y1="36" x2="82" y2="136" stroke="#CBD5E1" strokeWidth="1.8" strokeDasharray="4 4" />
            )}

            {mode === 'pendulum' ? (
              <line x1="88" y1="84" x2="136" y2="124" stroke="#94A3B8" strokeWidth="8.5" strokeLinecap="round" />
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
                <CapsuleArmLimb
                  joints={ghostJoints}
                  color={accent}
                  isActive
                  isGhost
                />
                <path
                  d={arc.d}
                  fill="none"
                  stroke={accent}
                  strokeWidth="2.6"
                  strokeDasharray="5 4"
                  opacity="0.88"
                />
                <circle cx={arc.endX} cy={arc.endY} r="3.5" fill={accent} />
              </>
            )}

            {/* Active Capsule Limb driven cleanly by Hierarchical Forward Kinematics */}
            <CapsuleArmLimb
              joints={activeJoints}
              color={accent}
              isActive
              compact={compact}
            />

            {!compact && mode === 'pendulum' && (
              <ellipse
                cx={shoulderX}
                cy={shoulderY + 72}
                rx="22"
                ry="7"
                fill="none"
                stroke={accent}
                strokeWidth="2.4"
                strokeDasharray="4 3"
              />
            )}
          </g>
        </g>

        {/* Upright Non-Mirrored Live Angle Pill */}
        {!compact && mode === 'flexion' && (
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
              {Math.round(currentAngleDeg)}°
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

    const isBilateral = spec.kind === 'bilateral_arm_raise';
    const isShrug = spec.kind === 'shoulder_shrug_roll';
    const isElbow = spec.kind === 'elbow_flexion';
    const isRot = spec.kind === 'shoulder_rotation';
    const isTrunk = spec.kind === 'trunk_align';

    const shrugDropOffset = isShrug ? -9 * p : 0;
    const curShY = shoulderY + shrugDropOffset;

    const resolvedPeakDeg =
      spec.kind === 'shoulder_overhead' || spec.kind === 'shoulder_wall_slide' || isBilateral
        ? Math.min(168, Math.max(135, targetRom))
        : spec.kind === 'shoulder_scaption'
          ? Math.min(145, targetRom)
          : targetRom;

    const rAngle = isBilateral
      ? p * resolvedPeakDeg
      : !activeIsLeft && !isShrug && !isElbow && !isRot && !isTrunk
        ? p * resolvedPeakDeg
        : 6;
    const lAngle = isBilateral
      ? p * resolvedPeakDeg
      : activeIsLeft && !isShrug && !isElbow && !isRot && !isTrunk
        ? p * resolvedPeakDeg
        : 6;

    const rElbowBend = isElbow && !activeIsLeft ? -p * 145 : !activeIsLeft ? 6 : 4;
    const lElbowBend = isElbow && activeIsLeft ? -p * 145 : activeIsLeft ? 6 : 4;

    const rightJoints = computeUpperLimbFkJoints({
      shoulderX: rightShX,
      shoulderY: curShY,
      upperArmLen: 36,
      forearmLen: 32,
      handLen: 9,
      angleDeg: isRot && !activeIsLeft ? 5 : rAngle,
      elbowBendDeg: isRot && !activeIsLeft ? 88 * (1 - p * 0.45) : rElbowBend,
      dirSign: 1,
    });
    const leftJoints = computeUpperLimbFkJoints({
      shoulderX: leftShX,
      shoulderY: curShY,
      upperArmLen: 36,
      forearmLen: 32,
      handLen: 9,
      angleDeg: isRot && activeIsLeft ? 5 : lAngle,
      elbowBendDeg: isRot && activeIsLeft ? 88 * (1 - p * 0.45) : lElbowBend,
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

    const showRightGhost = (!activeIsLeft || isBilateral) && !isShrug && !isElbow && !isRot && !isTrunk;
    const showLeftGhost = (activeIsLeft || isBilateral) && !isShrug && !isElbow && !isRot && !isTrunk;
    const rightArc = buildRomGuideArc(rightShX, shoulderY, 46, resolvedPeakDeg, 1);
    const leftArc = buildRomGuideArc(leftShX, shoulderY, 46, resolvedPeakDeg, -1);

    const activeReadoutDeg = isElbow
      ? Math.round(165 - p * 120)
      : isRot
        ? Math.round(p * Math.min(90, targetRom))
        : Math.round(p * resolvedPeakDeg);

    return (
      <g>
        <line x1="28" y1="186" x2="172" y2="186" stroke="#CBD5E1" strokeWidth="2.5" strokeLinecap="round" />

        {/* Wall reference line for wall-slide */}
        {spec.kind === 'shoulder_wall_slide' && !compact && (
          <line
            x1={activeIsLeft ? 32 : 168}
            y1="16"
            x2={activeIsLeft ? 32 : 168}
            y2="184"
            stroke="#94A3B8"
            strokeWidth="4.5"
            strokeDasharray="6 4"
          />
        )}

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
    const flipX = activeIsLeft ? -1 : 1;

    return (
      <g transform={`translate(100, 108) scale(${flipX}, 1)`}>
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
  };

  const renderNeckTilt = () => {
    const tiltDir = activeIsLeft ? -1 : 1;
    const tiltDeg = tiltDir * p * Math.min(35, targetRom);
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
          <circle cx="-6" cy="-44" r="2.2" fill={mutedBody} />
          <circle cx="6" cy="-44" r="2.2" fill={mutedBody} />
          <path d="M -5 -36 Q 0 -33 5 -36" fill="none" stroke={mutedBody} strokeWidth="2" strokeLinecap="round" />
        </g>
        {!compact && (
          <>
            <path
              d={activeIsLeft ? 'M 92 28 Q 66 32 56 54' : 'M 108 28 Q 134 32 144 54'}
              fill="none"
              stroke={accent}
              strokeWidth="2.8"
              strokeDasharray="4 3"
            />
            <text x="22" y="176" fontSize="10" fontWeight="800" fill={activeIsLeft ? '#EA580C' : '#94A3B8'}>
              L (ซ้าย)
            </text>
            <text x="140" y="176" fontSize="10" fontWeight="800" fill={!activeIsLeft ? '#059669' : '#94A3B8'}>
              R (ขวา)
            </text>
          </>
        )}
      </g>
    );
  };

  const renderLowerBodyView = () => {
    if (spec.kind === 'chair_squat') {
      const standRatio = p;
      const hipY = 128 - standRatio * 24;
      const shoulderY = hipY - 46;
      const kneeX = 114 - standRatio * 22;
      const kneeY = 132 + standRatio * 20;
      const armJoints = computeUpperLimbFkJoints({
        shoulderX: 84,
        shoulderY: shoulderY + 4,
        upperArmLen: 26,
        forearmLen: 22,
        handLen: 7,
        angleDeg: 35 + (1 - standRatio) * 45,
        elbowBendDeg: 8,
        dirSign: 1,
      });

      return (
        <g>
          <line x1="26" y1="186" x2="174" y2="186" stroke="#CBD5E1" strokeWidth="2.5" strokeLinecap="round" />
          <path d="M 46 186 L 46 130 L 82 130 L 82 186 M 46 130 L 46 84" fill="none" stroke="#94A3B8" strokeWidth="5" strokeLinecap="round" strokeLinejoin="round" />
          <line x1="82" y1={hipY} x2={kneeX} y2={kneeY} stroke={accent} strokeWidth="12" strokeLinecap="round" />
          <line x1={kneeX} y1={kneeY} x2="96" y2="182" stroke={accent} strokeWidth="10.5" strokeLinecap="round" />
          <line x1="94" y1="182" x2="108" y2="182" stroke={accent} strokeWidth="7" strokeLinecap="round" />
          <line x1="82" y1={hipY} x2="84" y2={shoulderY} stroke={mutedBody} strokeWidth="13" strokeLinecap="round" />
          <circle cx="86" cy={shoulderY - 18} r="13" fill="#FFFFFF" stroke={mutedBody} strokeWidth="2.8" />
          <CapsuleArmLimb joints={armJoints} color={accent} isActive compact={compact} wristWidth={8} />
        </g>
      );
    }

    if (spec.kind === 'knee_extension') {
      const flipDir: 1 | -1 = activeIsLeft ? -1 : 1;
      const shinDeg = -p * 86;
      const rad = (shinDeg * Math.PI) / 180;
      const kneeX = 114;
      const kneeY = 128;
      const ankleX = kneeX + Math.sin(-rad) * 48;
      const ankleY = kneeY + Math.cos(-rad) * 48;
      return (
        <g transform={flipDir === -1 ? 'translate(200, 0) scale(-1, 1)' : undefined}>
          <line x1="26" y1="186" x2="174" y2="186" stroke="#CBD5E1" strokeWidth="2.5" strokeLinecap="round" />
          <path d="M 46 186 L 46 132 L 96 132 L 96 186 M 46 132 L 46 80" fill="none" stroke="#94A3B8" strokeWidth="5" strokeLinecap="round" strokeLinejoin="round" />
          <line x1="74" y1="80" x2="74" y2="128" stroke={mutedBody} strokeWidth="13" strokeLinecap="round" />
          <circle cx="76" cy="60" r="13" fill="#FFFFFF" stroke={mutedBody} strokeWidth="2.8" />
          <line x1="74" y1="128" x2={kneeX} y2={kneeY} stroke={mutedBody} strokeWidth="12" strokeLinecap="round" />
          <line x1={kneeX} y1={kneeY} x2={ankleX} y2={ankleY} stroke={accent} strokeWidth="11" strokeLinecap="round" />
          <circle cx={kneeX} cy={kneeY} r="5.5" fill="#FFFFFF" stroke={accent} strokeWidth="2.5" />
          <circle cx={ankleX} cy={ankleY} r="4.5" fill={accent} />
        </g>
      );
    }

    if (spec.kind === 'heel_raise') {
      const liftY = -p * 18;
      return (
        <g>
          <line x1="26" y1="186" x2="174" y2="186" stroke="#CBD5E1" strokeWidth="2.5" strokeLinecap="round" />
          <path d="M 44 186 L 44 114 L 68 114" fill="none" stroke="#94A3B8" strokeWidth="4.5" strokeLinecap="round" />
          <g transform={`translate(0, ${liftY})`}>
            <circle cx="100" cy="48" r="13" fill="#FFFFFF" stroke={mutedBody} strokeWidth="2.8" />
            <line x1="100" y1="62" x2="100" y2="118" stroke={mutedBody} strokeWidth="13" strokeLinecap="round" />
            <line x1="100" y1="78" x2="68" y2="112" stroke={mutedLimb} strokeWidth="8" strokeLinecap="round" />
            <line x1="92" y1="118" x2="92" y2="180" stroke={activeIsLeft ? accent : mutedLimb} strokeWidth="10.5" strokeLinecap="round" />
            <line x1="110" y1="118" x2="110" y2="180" stroke={activeIsLeft ? mutedLimb : accent} strokeWidth="10.5" strokeLinecap="round" />
          </g>
          {!compact && (
            <path d="M 142 174 L 142 146 M 136 153 L 142 146 L 148 153" fill="none" stroke={accent} strokeWidth="2.8" strokeLinecap="round" strokeLinejoin="round" />
          )}
        </g>
      );
    }

    const isMarch = spec.kind === 'hip_march';
    const kneeLift = isMarch ? p * 36 : 26;
    const isLeftLegActive = activeIsLeft;
    return (
      <g>
        <line x1="28" y1="186" x2="172" y2="186" stroke="#CBD5E1" strokeWidth="2.5" strokeLinecap="round" />
        {!compact && <line x1="100" y1="20" x2="100" y2="186" stroke="#E2E8F0" strokeWidth="2" strokeDasharray="4 4" />}
        <path d="M 74 76 L 126 76 L 118 128 L 82 128 Z" fill="#DBEAFE" stroke={mutedBody} strokeWidth="2.8" strokeLinejoin="round" />
        <circle cx="100" cy="46" r="14" fill="#FFFFFF" stroke={mutedBody} strokeWidth="2.8" />
        <line x1="74" y1="78" x2="62" y2="118" stroke={mutedLimb} strokeWidth="8" strokeLinecap="round" />
        <line x1="126" y1="78" x2="138" y2="118" stroke={mutedLimb} strokeWidth="8" strokeLinecap="round" />
        <line
          x1="88"
          y1="128"
          x2="86"
          y2={isMarch ? (isLeftLegActive ? 180 - kneeLift : 182) : isLeftLegActive ? 182 : 148}
          stroke={isLeftLegActive ? accent : '#1E3A5F'}
          strokeWidth="11"
          strokeLinecap="round"
        />
        <line
          x1="112"
          y1="128"
          x2="114"
          y2={isMarch ? (isLeftLegActive ? 182 : 180 - kneeLift) : isLeftLegActive ? 148 : 182}
          stroke={isLeftLegActive ? '#1E3A5F' : accent}
          strokeWidth="11"
          strokeLinecap="round"
        />
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
        : spec.kind === 'neck_tilt'
          ? renderNeckTilt()
          : spec.kind === 'shoulder_flexion'
            ? renderSideSagittalView('flexion')
            : spec.kind === 'shoulder_pendulum'
              ? renderSideSagittalView('pendulum')
              : spec.kind === 'chair_squat' ||
                  spec.kind === 'knee_extension' ||
                  spec.kind === 'hip_march' ||
                  spec.kind === 'heel_raise' ||
                  spec.kind === 'single_leg_balance'
                ? renderLowerBodyView()
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
      className="glass-card rounded-3xl p-4 sm:p-5"
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
      <div className="mt-3.5 grid grid-cols-1 items-center gap-4 md:grid-cols-12">
        <div className="md:col-span-4 flex flex-col items-center">
          <div className="relative flex h-44 w-44 items-center justify-center rounded-2xl border border-sky-200/80 bg-gradient-to-br from-white via-sky-50/70 to-emerald-50/60 p-1.5 shadow-inner">
            <ExercisePostureSvg exercise={exercise} phase={previewPhase} />
            {isHoldingNow && (
              <span className="absolute top-2 right-2 rounded-full bg-emerald-600 px-2 py-0.5 text-[10px] font-black text-white shadow animate-pulse">
                HOLD
              </span>
            )}
          </div>
          <span className="mt-1.5 text-center text-[11px] font-bold text-slate-600">
            {spec.viewAngleLabel}
          </span>
        </div>

        <div className="md:col-span-8 flex flex-col gap-2.5">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <h3 className="text-sm sm:text-base font-extrabold text-[#0F4C81]">
                {lang === 'en' ? exercise.name_en : exercise.name_th}
              </h3>
              <p className="text-xs font-medium text-slate-600">{spec.muscleFocus}</p>
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

          <div className="grid grid-cols-1 gap-2 sm:grid-cols-3">
            {spec.steps.map((stepText, idx) => {
              const stepPhase: PosturePreviewPhase = idx === 0 ? 'prepare' : idx === 1 ? 'move' : 'hold';
              const highlighted = previewPhase === stepPhase || (idx === 2 && isHoldingNow);
              return (
                <button
                  key={idx}
                  type="button"
                  onClick={() => setPreviewPhase(previewPhase === stepPhase ? 'auto' : stepPhase)}
                  className={`rounded-2xl border p-2.5 text-left transition cursor-pointer ${
                    highlighted
                      ? 'border-emerald-400 bg-emerald-50/90 shadow-sm ring-2 ring-emerald-200'
                      : 'border-sky-100 bg-white/85 hover:border-sky-300'
                  }`}
                >
                  <div className="flex items-center gap-1.5 text-[11px] font-black text-[#0284C7]">
                    <span className="inline-flex h-5 w-5 items-center justify-center rounded-full bg-[#0F4C81] text-[10px] font-black text-white">
                      {idx + 1}
                    </span>
                    <span>{idx === 0 ? ui.step1 : idx === 1 ? ui.step2 : ui.step3}</span>
                  </div>
                  <p className="mt-1.5 text-xs font-semibold leading-snug text-slate-700">
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
