import React, { useMemo, useState } from 'react';
import { motion } from 'framer-motion';
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
      th: 'มุมมองด้านข้าง (ยกแขนตรงไปด้านหน้า)',
      en: 'Side view (Raise straight forward)',
      cn: '侧面视角（向前平举上抬）',
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
          `ยืนหรือนั่งตัวตรง ปล่อยแขน${sTh}ข้างลำตัว ผ่อนคลายหัวไหล่`,
          `ยกแขน${sTh}ตรงไปด้านหน้าช้าๆ จนถึงมุมประมาณ ${rom}°`,
          `ค้างท่าเป้าหมายไว้ ${holdSec} วินาที แล้วค่อยๆ ลดแขนลง`,
        ],
        en: [
          `Stand or sit upright with ${sEn} arm relaxed by your side`,
          `Slowly raise your ${sEn} arm straight forward toward ${rom}°`,
          `Hold steady at peak for ${holdSec}s, then lower gently`,
        ],
        cn: [
          `挺直站立或坐姿，${sCn}臂自然垂放体侧，放松双肩`,
          `缓慢将${sCn}臂向前平举上抬至约 ${rom}°`,
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
      th: 'มุมมองหน้าตรง (กางออกด้านข้างแบบกระจกเงา)',
      en: 'Front mirror view (Outward to the side)',
      cn: '正面镜像视角（向体侧外展）',
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
          `ยืนตรงหน้ากล้อง ปล่อยแขน${sTh}แนบข้างลำตัว`,
          `ค่อยๆ กางแขน${sTh}ออกด้านข้างขึ้นเป็นวงโค้งถึง ${rom}°`,
          `ค้างไว้ด้านข้าง ${holdSec} วินาที แล้วลดแขนลงอย่างนุ่มนวล`,
        ],
        en: [
          `Face the camera upright with ${sEn} arm resting at your side`,
          `Smoothly abduct your ${sEn} arm outward toward ${rom}°`,
          `Hold steady for ${holdSec}s, then return smoothly`,
        ],
        cn: [
          `正对摄像头站直，${sCn}臂放松贴于体侧`,
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
          `ตั้งลำตัวตรง เหยียดข้อศอก${sTh}ตรงข้างลำตัว`,
          `ยกแขน${sTh}เฉียงออกด้านหน้าข้าง 45° ถึงมุม ${rom}°`,
          `ค้างท่าที่มุมเป้าหมาย ${holdSec} วินาที แล้วค่อยๆ วางลง`,
        ],
        en: [
          `Stand tall with ${sEn} elbow straight by your side`,
          `Lift ${sEn} arm diagonally 45° forward-outward to ${rom}°`,
          `Hold for ${holdSec}s at top, then lower with control`,
        ],
        cn: [
          `身体挺直，${sCn}肘自然伸直`,
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
          `ยืนหลังตรง เตรียมยกแขน${sTh}จากข้างลำตัว`,
          `เหยียดแขน${sTh}ชูขึ้นเหนือศีรษะชิดข้างใบหู (${rom}°)`,
          `ค้างไว้ด้านบน ${holdSec} วินาที พร้อมหายใจเข้าลึกๆ`,
        ],
        en: [
          `Stand upright preparing to lift your ${sEn} arm`,
          `Reach ${sEn} arm overhead beside ear toward ${rom}°`,
          `Hold overhead for ${holdSec}s while breathing evenly`,
        ],
        cn: [
          `站直准备抬起${sCn}臂`,
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
          `Stand facing/near wall with ${sEn} hand lightly on wall`,
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
          `แกว่งเป็นวงกลมเล็กๆ ต่อเนื่องสม่ำเสมอ ${holdSec} วินาที`,
        ],
        en: [
          `Lean forward gently resting opposite hand on support`,
          `Let your ${sEn} arm hang completely loose like a pendulum`,
          `Trace small gentle circles smoothly for ${holdSec}s`,
        ],
        cn: [
          `身体微前倾，健侧手轻扶椅子支撑稳定`,
          `让${sCn}臂如钟摆般完全放松垂下`,
          `轻柔划小圈摆动，持续 ${holdSec} 秒`,
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
      th: 'มุมมองหน้าตรงช่วงบน (ยกหัวไหล่ขึ้น & หมุนลง)',
      en: 'Upper-torso view (Elevate up & roll back)',
      cn: '上身正面视角（双肩向上耸起并向后绕环）',
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
        'ยกหัวไหล่ทั้ง 2 ข้างขึ้นหาใบหูช้าๆ แล้วค้างไว้ ' + holdSec + ' วินาที',
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
      en: 'Front view (Both arms rising together symmetrically)',
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
        'ประสานมือทั้ง 2 ข้างไว้ด้านหน้าลำตัว เหยียดแขนให้สบาย',
        `ค่อยๆ ยกแขนทั้งสองข้างชูขึ้นเหนือศีรษะพร้อมกันถึง ~${rom}°`,
        `ค้างไว้ด้านบน ${holdSec} วินาที พร้อมหายใจลึก แล้วลดลงช้าๆ`,
      ],
      en: [
        'Clasp both hands together comfortably in front of torso',
        `Raise both arms smoothly overhead together toward ~${rom}°`,
        `Hold at top for ${holdSec}s breathing deeply, then lower slowly`,
      ],
      cn: [
        '双手十指相扣置于腹前，放松双肩',
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

function getPhaseKeyframe<T>(phase: PosturePreviewPhase, values: [T, T, T, T]): T | T[] {
  if (phase === 'prepare') return values[0];
  if (phase === 'move') return values[1];
  if (phase === 'hold') return values[2];
  return values;
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
  const mutedLimb = '#94A3B8';
  const activeIsLeft = spec.side === 'left';
  const targetRom = Math.min(170, Math.max(20, Math.round(exercise.target_rom_deg || 120)));

  const transition =
    phase === 'auto'
      ? { repeat: Infinity, duration: 3.6, times: [0, 0.35, 0.72, 1], ease: 'easeInOut' as const }
      : { duration: 0.45, ease: 'easeOut' as const };

  // Mirror screen convention (R on right side of viewer, L on left side of viewer)
  const rightPivotX = 110;
  const leftPivotX = 70;
  const shoulderY = 62;

  const renderHandCloseUp = () => {
    const wristRotate = getPhaseKeyframe(phase, [18, -14, -26, 18]);
    const fingerSpread = getPhaseKeyframe(phase, [0, 12, 18, 0]);
    const fingerCurl = getPhaseKeyframe(phase, [28, 6, 0, 28]);
    const flipX = activeIsLeft ? -1 : 1;

    return (
      <g transform={`translate(90, 96) scale(${flipX}, 1)`}>
        {/* Forearm base */}
        <line x1="-58" y1="24" x2="-10" y2="10" stroke={mutedBody} strokeWidth="14" strokeLinecap="round" />
        <line x1="-58" y1="24" x2="-10" y2="10" stroke="#BAE6FD" strokeWidth="8" strokeLinecap="round" />
        {/* Wrist joint halo */}
        <circle cx="-10" cy="10" r="8" fill={accent} fillOpacity="0.22" stroke={accent} strokeWidth="2.5" />
        {/* Articulated Palm + 5 Fingers */}
        <motion.g animate={{ rotate: wristRotate }} transition={transition}>
          <rect x="-8" y="-12" width="32" height="30" rx="10" fill={accent} fillOpacity="0.18" stroke={accent} strokeWidth="3" />
          {/* Thumb + 4 Fingers (separate static translate wrapper so Framer Motion rotate preserves origin) */}
          {[-34, -14, -3, 8, 20].map((baseDeg, i) => {
            const tx = i === 0 ? 6 : 22;
            const ty = i === 0 ? -10 : -8 + (i - 1) * 7;
            return (
              <g key={i} transform={`translate(${tx}, ${ty})`}>
                <motion.g
                  animate={{
                    rotate: Array.isArray(fingerSpread)
                      ? fingerSpread.map(
                          (sp, idx) =>
                            baseDeg * (1 + sp / 30) +
                            (Array.isArray(fingerCurl) ? fingerCurl[idx] : fingerCurl),
                        )
                      : baseDeg * (1 + (fingerSpread as number) / 30) + (fingerCurl as number),
                  }}
                  transition={transition}
                >
                  <line
                    x1="0"
                    y1="0"
                    x2={i === 0 ? 18 : i === 4 ? 22 : 28}
                    y2="0"
                    stroke={accent}
                    strokeWidth="5"
                    strokeLinecap="round"
                  />
                </motion.g>
              </g>
            );
          })}
        </motion.g>
        {!compact && (
          <path d="M 10 -32 Q 30 -36 38 -18" fill="none" stroke={accent} strokeWidth="2.5" strokeDasharray="4 3" />
        )}
      </g>
    );
  };

  const renderNeckTilt = () => {
    const tiltDeg = activeIsLeft ? getPhaseKeyframe(phase, [0, -18, -26, 0]) : getPhaseKeyframe(phase, [0, 18, 26, 0]);
    return (
      <g>
        {/* Reference vertical plumb */}
        {!compact && (
          <line x1="90" y1="22" x2="90" y2="148" stroke="#CBD5E1" strokeWidth="2" strokeDasharray="4 4" />
        )}
        {/* Level flat shoulders bar */}
        <line x1="48" y1="92" x2="132" y2="92" stroke={mutedBody} strokeWidth="9" strokeLinecap="round" />
        <path d="M 64 92 L 66 154 L 114 154 L 116 92 Z" fill="#E0F2FE" stroke={mutedBody} strokeWidth="3.5" strokeLinejoin="round" />
        <circle cx="52" cy="92" r="5" fill="#64748B" />
        <circle cx="128" cy="92" r="5" fill="#64748B" />
        {/* Tilting Neck + Head */}
        <g transform="translate(90, 90)">
          <motion.g animate={{ rotate: tiltDeg }} transition={transition}>
            <line x1="0" y1="0" x2="0" y2="-22" stroke={accent} strokeWidth="10" strokeLinecap="round" />
            <circle cx="0" cy="-38" r="18" fill="#FFFFFF" stroke={accent} strokeWidth="4" />
            {/* Face orientation cues */}
            <circle cx="-6" cy="-40" r="2.2" fill={mutedBody} />
            <circle cx="6" cy="-40" r="2.2" fill={mutedBody} />
            <path d="M -5 -32 Q 0 -29 5 -32" fill="none" stroke={mutedBody} strokeWidth="2" strokeLinecap="round" />
          </motion.g>
        </g>
        {!compact && (
          <path
            d={activeIsLeft ? 'M 84 26 Q 60 28 50 48' : 'M 96 26 Q 120 28 130 48'}
            fill="none"
            stroke={accent}
            strokeWidth="3"
            strokeDasharray="4 3"
          />
        )}
      </g>
    );
  };

  const renderSideProfileRaise = (mode: 'flexion' | 'pendulum') => {
    const faceDir = activeIsLeft ? -1 : 1;
    const armAngles =
      mode === 'pendulum'
        ? getPhaseKeyframe(phase, [-16, 16, -14, -16])
        : getPhaseKeyframe(phase, [0, -Math.min(130, targetRom) * 0.65, -Math.min(145, targetRom), 0]);
    const leanDeg = mode === 'pendulum' ? 20 : 0;
    const targetPeakRad = ((-Math.min(145, targetRom) + 90) * Math.PI) / 180;

    return (
      <g transform={`translate(90, 0) scale(${faceDir}, 1) translate(-90, 0)`}>
        {/* Floor baseline */}
        <line x1="34" y1="158" x2="146" y2="158" stroke="#CBD5E1" strokeWidth="2.5" strokeLinecap="round" />
        {/* Support table/chair for Codman Pendulum */}
        {mode === 'pendulum' && (
          <path d="M 118 158 L 118 108 L 144 108" fill="none" stroke="#94A3B8" strokeWidth="4" strokeLinecap="round" strokeLinejoin="round" />
        )}
        {/* Standing legs */}
        <line x1="84" y1="108" x2="88" y2="156" stroke={mutedBody} strokeWidth="6" strokeLinecap="round" />
        <line x1="84" y1="108" x2="76" y2="156" stroke={mutedLimb} strokeWidth="5" strokeLinecap="round" />
        {/* Torso & Head */}
        <g transform={`translate(84, 108) rotate(${leanDeg})`}>
          {/* Vertical 0° spine reference */}
          {!compact && mode === 'flexion' && (
            <line x1="0" y1="10" x2="0" y2="-68" stroke="#CBD5E1" strokeWidth="2" strokeDasharray="4 4" />
          )}
          <line x1="0" y1="0" x2="0" y2="-48" stroke={mutedBody} strokeWidth="7" strokeLinecap="round" />
          <circle cx="2" cy="-64" r="12" fill={mutedBody} />
          {/* Profile nose cue showing forward direction */}
          <path d="M 12 -66 L 17 -62 L 12 -59" fill={mutedBody} />

          {/* Opposite support arm */}
          {mode === 'pendulum' && (
            <line x1="0" y1="-44" x2="36" y2="-10" stroke={mutedLimb} strokeWidth="5" strokeLinecap="round" />
          )}

          {/* Peak Target Ghost Line */}
          {!compact && mode === 'flexion' && (
            <line
              x1="0"
              y1="-44"
              x2={Math.cos(targetPeakRad) * 44}
              y2={-44 + Math.sin(targetPeakRad) * 44}
              stroke={accent}
              strokeWidth="2.5"
              strokeDasharray="4 4"
              opacity="0.55"
            />
          )}

          {/* Prescribed Moving Arm from Shoulder (0, -44) */}
          <g transform="translate(0, -44)">
            <motion.g animate={{ rotate: armAngles }} transition={transition}>
              <line x1="0" y1="0" x2="0" y2="44" stroke={accent} strokeWidth="7" strokeLinecap="round" />
              <circle cx="0" cy="44" r="5" fill={accent} />
            </motion.g>
            <circle cx="0" cy="0" r="6" fill="#FFFFFF" stroke={accent} strokeWidth="3" />
          </g>

          {/* Pendulum circular sweep cue */}
          {!compact && mode === 'pendulum' && (
            <ellipse cx="0" cy="4" rx="18" ry="6" fill="none" stroke={accent} strokeWidth="2.5" strokeDasharray="4 3" />
          )}
        </g>
      </g>
    );
  };

  const renderSquatOrKnee = () => {
    if (spec.kind === 'chair_squat') {
      const hipDropY = getPhaseKeyframe(phase, [34, 16, 0, 34]);
      const kneeForwardX = getPhaseKeyframe(phase, [28, 16, 2, 28]);
      return (
        <g>
          <line x1="28" y1="158" x2="152" y2="158" stroke="#CBD5E1" strokeWidth="2.5" strokeLinecap="round" />
          {/* Chair silhouette */}
          <path d="M 48 158 L 48 112 L 80 112 L 80 158 M 48 112 L 48 74" fill="none" stroke="#94A3B8" strokeWidth="4.5" strokeLinecap="round" strokeLinejoin="round" />
          {/* Animated Sit-to-Stand figure facing right */}
          <motion.g
            animate={{
              y: hipDropY,
            }}
            transition={transition}
          >
            <line x1="72" y1="74" x2="76" y2="108" stroke={mutedBody} strokeWidth="7.5" strokeLinecap="round" />
            <circle cx="75" cy="56" r="11" fill={mutedBody} />
            <line x1="74" y1="80" x2="106" y2="96" stroke={accent} strokeWidth="5.5" strokeLinecap="round" />
          </motion.g>
          {/* Dynamic Thigh + Shin */}
          <motion.line
            animate={{
              x1: 76,
              y1: Array.isArray(hipDropY) ? hipDropY.map((d) => 108 + d) : 108 + (hipDropY as number),
              x2: Array.isArray(kneeForwardX) ? kneeForwardX.map((k) => 86 + k) : 86 + (kneeForwardX as number),
              y2: Array.isArray(hipDropY) ? hipDropY.map((d) => 128 + d * 0.2) : 128 + (hipDropY as number) * 0.2,
            }}
            transition={transition}
            stroke={accent}
            strokeWidth="7.5"
            strokeLinecap="round"
          />
          <motion.line
            animate={{
              x1: Array.isArray(kneeForwardX) ? kneeForwardX.map((k) => 86 + k) : 86 + (kneeForwardX as number),
              y1: Array.isArray(hipDropY) ? hipDropY.map((d) => 128 + d * 0.2) : 128 + (hipDropY as number) * 0.2,
              x2: 96,
              y2: 156,
            }}
            transition={transition}
            stroke={accent}
            strokeWidth="7"
            strokeLinecap="round"
          />
        </g>
      );
    }

    if (spec.kind === 'knee_extension') {
      const shinAngles = getPhaseKeyframe(phase, [0, -52, -84, 0]);
      return (
        <g transform={`translate(90, 0) scale(${activeIsLeft ? -1 : 1}, 1) translate(-90, 0)`}>
          <line x1="28" y1="158" x2="152" y2="158" stroke="#CBD5E1" strokeWidth="2.5" strokeLinecap="round" />
          {/* Chair */}
          <path d="M 48 158 L 48 114 L 88 114 L 88 158 M 48 114 L 48 70" fill="none" stroke="#94A3B8" strokeWidth="4.5" strokeLinecap="round" strokeLinejoin="round" />
          {/* Seated Upright Body */}
          <line x1="66" y1="68" x2="66" y2="112" stroke={mutedBody} strokeWidth="7.5" strokeLinecap="round" />
          <circle cx="66" cy="50" r="11" fill={mutedBody} />
          {/* Thigh */}
          <line x1="66" y1="112" x2="100" y2="112" stroke={mutedBody} strokeWidth="7.5" strokeLinecap="round" />
          {/* Hinged Lower Leg at Knee (100, 112) */}
          <g transform="translate(100, 112)">
            <motion.g animate={{ rotate: shinAngles }} transition={transition}>
              <line x1="0" y1="0" x2="0" y2="42" stroke={accent} strokeWidth="7.5" strokeLinecap="round" />
              <circle cx="4" cy="42" r="4.5" fill={accent} />
            </motion.g>
            <circle cx="0" cy="0" r="6" fill="#FFFFFF" stroke={accent} strokeWidth="3" />
          </g>
        </g>
      );
    }

    if (spec.kind === 'heel_raise') {
      const heelLiftY = getPhaseKeyframe(phase, [0, -9, -15, 0]);
      return (
        <g>
          <line x1="28" y1="158" x2="152" y2="158" stroke="#CBD5E1" strokeWidth="2.5" strokeLinecap="round" />
          {/* Chair support */}
          <path d="M 42 158 L 42 96 L 62 96" fill="none" stroke="#94A3B8" strokeWidth="4" strokeLinecap="round" />
          <motion.g animate={{ y: heelLiftY }} transition={transition}>
            <circle cx="90" cy="36" r="11" fill={mutedBody} />
            <line x1="90" y1="48" x2="90" y2="92" stroke={mutedBody} strokeWidth="7" strokeLinecap="round" />
            <line x1="90" y1="60" x2="62" y2="94" stroke={mutedLimb} strokeWidth="5" strokeLinecap="round" />
            <line x1="84" y1="92" x2="84" y2="152" stroke={activeIsLeft ? accent : mutedLimb} strokeWidth="6.5" strokeLinecap="round" />
            <line x1="98" y1="92" x2="98" y2="152" stroke={activeIsLeft ? mutedLimb : accent} strokeWidth="6.5" strokeLinecap="round" />
          </motion.g>
          {!compact && (
            <path d="M 122 146 L 122 126 M 117 132 L 122 126 L 127 132" fill="none" stroke={accent} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
          )}
        </g>
      );
    }

    // hip_march & single_leg_balance (Front view)
    const raiseLeg = spec.kind === 'hip_march';
    const liftedY = raiseLeg ? getPhaseKeyframe(phase, [156, 130, 114, 156]) : 128;
    const isLeftActive = activeIsLeft;

    return (
      <g>
        <line x1="34" y1="158" x2="146" y2="158" stroke="#CBD5E1" strokeWidth="2.5" strokeLinecap="round" />
        {!compact && <line x1="90" y1="22" x2="90" y2="158" stroke="#E2E8F0" strokeWidth="2" strokeDasharray="4 4" />}
        <circle cx="90" cy="34" r="11" fill={mutedBody} />
        <line x1="90" y1="46" x2="90" y2="96" stroke={mutedBody} strokeWidth="7" strokeLinecap="round" />
        <line x1="70" y1="60" x2="110" y2="60" stroke={mutedBody} strokeWidth="6" strokeLinecap="round" />
        <line x1="70" y1="60" x2="58" y2="96" stroke={mutedLimb} strokeWidth="5" strokeLinecap="round" />
        <line x1="110" y1="60" x2="122" y2="96" stroke={mutedLimb} strokeWidth="5" strokeLinecap="round" />
        {/* Left leg (screen left) */}
        <motion.line
          x1="82"
          y1="96"
          x2="76"
          animate={{ y2: raiseLeg ? (isLeftActive ? liftedY : 156) : isLeftActive ? 156 : liftedY }}
          transition={transition}
          stroke={isLeftActive ? accent : mutedLimb}
          strokeWidth="6.5"
          strokeLinecap="round"
        />
        {/* Right leg (screen right) */}
        <motion.line
          x1="98"
          y1="96"
          x2="104"
          animate={{ y2: raiseLeg ? (isLeftActive ? 156 : liftedY) : isLeftActive ? liftedY : 156 }}
          transition={transition}
          stroke={isLeftActive ? mutedLimb : accent}
          strokeWidth="6.5"
          strokeLinecap="round"
        />
      </g>
    );
  };

  const renderFrontUpperBody = () => {
    const isBilateral = spec.kind === 'bilateral_arm_raise';
    const isShrug = spec.kind === 'shoulder_shrug_roll';
    const isElbow = spec.kind === 'elbow_flexion';
    const isRot = spec.kind === 'shoulder_rotation';
    const isTrunk = spec.kind === 'trunk_align';

    const shrugY = isShrug ? getPhaseKeyframe(phase, [0, -7, -10, 0]) : 0;

    // Arm peak degrees from straight down: Right sweeps CCW (-deg), Left sweeps CW (+deg)
    const peakArmDeg =
      spec.kind === 'shoulder_overhead' || spec.kind === 'shoulder_wall_slide' || isBilateral
        ? 156
        : spec.kind === 'shoulder_scaption'
          ? 122
          : 128;

    const rArmRotate = isBilateral
      ? getPhaseKeyframe(phase, [0, -90, -peakArmDeg, 0])
      : !activeIsLeft && !isShrug && !isElbow && !isRot && !isTrunk
        ? getPhaseKeyframe(phase, [0, -peakArmDeg * 0.65, -peakArmDeg, 0])
        : 0;

    const lArmRotate = isBilateral
      ? getPhaseKeyframe(phase, [0, 90, peakArmDeg, 0])
      : activeIsLeft && !isShrug && !isElbow && !isRot && !isTrunk
        ? getPhaseKeyframe(phase, [0, peakArmDeg * 0.65, peakArmDeg, 0])
        : 0;

    const forearmBendDeg = isElbow
      ? getPhaseKeyframe(phase, [0, 82, 132, 0])
      : isRot
        ? getPhaseKeyframe(phase, [0, 46, 78, 0])
        : 0;

    return (
      <g>
        <line x1="34" y1="158" x2="146" y2="158" stroke="#CBD5E1" strokeWidth="2.5" strokeLinecap="round" />
        {/* Wall guide for wall slide */}
        {spec.kind === 'shoulder_wall_slide' && !compact && (
          <line
            x1={activeIsLeft ? 34 : 146}
            y1="20"
            x2={activeIsLeft ? 34 : 146}
            y2="156"
            stroke="#94A3B8"
            strokeWidth="4"
            strokeDasharray="6 4"
          />
        )}
        {/* Spine axis for trunk align */}
        {isTrunk && !compact && (
          <line x1="90" y1="16" x2="90" y2="156" stroke={accent} strokeWidth="3" strokeDasharray="5 4" />
        )}
        {/* Standing legs */}
        <line x1="84" y1="110" x2="78" y2="156" stroke={mutedBody} strokeWidth="6" strokeLinecap="round" />
        <line x1="96" y1="110" x2="102" y2="156" stroke={mutedBody} strokeWidth="6" strokeLinecap="round" />
        {/* Pelvis & Torso */}
        <path d="M 74 62 L 106 62 L 100 110 L 80 110 Z" fill="#E0F2FE" stroke={mutedBody} strokeWidth="3.5" strokeLinejoin="round" />
        <circle cx="90" cy="36" r="12" fill={mutedBody} />

        {/* Shoulder Bar (elevates in Shrug) */}
        <motion.g animate={{ y: shrugY }} transition={transition}>
          <line x1={leftPivotX} y1={shoulderY} x2={rightPivotX} y2={shoulderY} stroke={mutedBody} strokeWidth="7" strokeLinecap="round" />

          {/* Left Arm (Screen Left) */}
          <g transform={`translate(${leftPivotX}, ${shoulderY})`}>
            {isElbow && activeIsLeft ? (
              <g>
                <line x1="0" y1="0" x2="-4" y2="24" stroke={accent} strokeWidth="6.5" strokeLinecap="round" />
                <g transform="translate(-4, 24)">
                  <motion.g animate={{ rotate: forearmBendDeg }} transition={transition}>
                    <line x1="0" y1="0" x2="-4" y2="24" stroke={accent} strokeWidth="6.5" strokeLinecap="round" />
                    <circle cx="-4" cy="24" r="4.5" fill={accent} />
                  </motion.g>
                  <circle cx="0" cy="0" r="5" fill="#FFFFFF" stroke={accent} strokeWidth="2.5" />
                </g>
              </g>
            ) : isRot && activeIsLeft ? (
              <g>
                <line x1="0" y1="0" x2="-2" y2="24" stroke={accent} strokeWidth="6" strokeLinecap="round" />
                <motion.line
                  x1="-2"
                  y1="24"
                  animate={{
                    x2: Array.isArray(forearmBendDeg) ? forearmBendDeg.map((d) => -2 - (d / 78) * 28) : -2 - ((forearmBendDeg as number) / 78) * 28,
                    y2: 28,
                  }}
                  transition={transition}
                  stroke={accent}
                  strokeWidth="6"
                  strokeLinecap="round"
                />
              </g>
            ) : (
              <motion.g animate={{ rotate: lArmRotate }} transition={transition}>
                <line
                  x1="0"
                  y1="0"
                  x2="-4"
                  y2="46"
                  stroke={activeIsLeft || isBilateral || isShrug ? accent : mutedLimb}
                  strokeWidth={activeIsLeft || isBilateral ? '7' : '5.5'}
                  strokeLinecap="round"
                />
                <circle cx="-4" cy="46" r="4.5" fill={activeIsLeft || isBilateral || isShrug ? accent : mutedLimb} />
              </motion.g>
            )}
            <circle
              cx="0"
              cy="0"
              r="5.5"
              fill="#FFFFFF"
              stroke={activeIsLeft || isBilateral || isShrug ? accent : mutedLimb}
              strokeWidth="2.5"
            />
          </g>

          {/* Right Arm (Screen Right) */}
          <g transform={`translate(${rightPivotX}, ${shoulderY})`}>
            {isElbow && !activeIsLeft ? (
              <g>
                <line x1="0" y1="0" x2="4" y2="24" stroke={accent} strokeWidth="6.5" strokeLinecap="round" />
                <g transform="translate(4, 24)">
                  <motion.g
                    animate={{
                      rotate: Array.isArray(forearmBendDeg) ? forearmBendDeg.map((d) => -d) : -(forearmBendDeg as number),
                    }}
                    transition={transition}
                  >
                    <line x1="0" y1="0" x2="4" y2="24" stroke={accent} strokeWidth="6.5" strokeLinecap="round" />
                    <circle cx="4" cy="24" r="4.5" fill={accent} />
                  </motion.g>
                  <circle cx="0" cy="0" r="5" fill="#FFFFFF" stroke={accent} strokeWidth="2.5" />
                </g>
              </g>
            ) : isRot && !activeIsLeft ? (
              <g>
                <line x1="0" y1="0" x2="2" y2="24" stroke={accent} strokeWidth="6" strokeLinecap="round" />
                <motion.line
                  x1="2"
                  y1="24"
                  animate={{
                    x2: Array.isArray(forearmBendDeg) ? forearmBendDeg.map((d) => 2 + (d / 78) * 28) : 2 + ((forearmBendDeg as number) / 78) * 28,
                    y2: 28,
                  }}
                  transition={transition}
                  stroke={accent}
                  strokeWidth="6"
                  strokeLinecap="round"
                />
              </g>
            ) : (
              <motion.g animate={{ rotate: rArmRotate }} transition={transition}>
                <line
                  x1="0"
                  y1="0"
                  x2="4"
                  y2="46"
                  stroke={!activeIsLeft || isBilateral || isShrug ? accent : mutedLimb}
                  strokeWidth={!activeIsLeft || isBilateral ? '7' : '5.5'}
                  strokeLinecap="round"
                />
                <circle cx="4" cy="46" r="4.5" fill={!activeIsLeft || isBilateral || isShrug ? accent : mutedLimb} />
              </motion.g>
            )}
            <circle
              cx="0"
              cy="0"
              r="5.5"
              fill="#FFFFFF"
              stroke={!activeIsLeft || isBilateral || isShrug ? accent : mutedLimb}
              strokeWidth="2.5"
            />
          </g>
        </motion.g>

        {/* Directional Dotted Arc */}
        {!compact && !isElbow && !isRot && !isShrug && !isTrunk && (
          <>
            {(spec.side === 'right' || isBilateral) && (
              <path d="M 120 104 Q 148 74 128 30" fill="none" stroke={accent} strokeWidth="2.5" strokeDasharray="4 4" opacity="0.7" />
            )}
            {(spec.side === 'left' || isBilateral) && (
              <path d="M 60 104 Q 32 74 52 30" fill="none" stroke={accent} strokeWidth="2.5" strokeDasharray="4 4" opacity="0.7" />
            )}
          </>
        )}
        {!compact && isShrug && (
          <>
            <path d="M 52 58 L 52 38 M 47 44 L 52 38 L 57 44" fill="none" stroke={accent} strokeWidth="2.5" strokeLinecap="round" />
            <path d="M 128 58 L 128 38 M 123 44 L 128 38 L 133 44" fill="none" stroke={accent} strokeWidth="2.5" strokeLinecap="round" />
          </>
        )}
      </g>
    );
  };

  return (
    <svg
      viewBox="0 0 180 180"
      className={compact ? 'h-12 w-12' : 'h-full w-full'}
      aria-label={exercise.name_th}
    >
      {/* Subtle anatomical grid circle */}
      {!compact && (
        <circle cx="90" cy="90" r="78" fill="#F8FAFC" stroke="#E2E8F0" strokeWidth="1.5" />
      )}

      {spec.kind === 'wrist_finger'
        ? renderHandCloseUp()
        : spec.kind === 'neck_tilt'
          ? renderNeckTilt()
          : spec.kind === 'shoulder_flexion'
            ? renderSideProfileRaise('flexion')
            : spec.kind === 'shoulder_pendulum'
              ? renderSideProfileRaise('pendulum')
              : spec.kind === 'chair_squat' ||
                  spec.kind === 'knee_extension' ||
                  spec.kind === 'hip_march' ||
                  spec.kind === 'heel_raise' ||
                  spec.kind === 'single_leg_balance'
                ? renderSquatOrKnee()
                : renderFrontUpperBody()}

      {/* Clear Mirror L/R Hand Indicators */}
      {!compact && spec.kind !== 'wrist_finger' && spec.kind !== 'chair_squat' && spec.kind !== 'knee_extension' && (
        <>
          <text
            x="22"
            y="168"
            fontSize="10"
            fontWeight="800"
            fill={spec.side === 'left' || spec.side === 'both' ? '#EA580C' : '#94A3B8'}
          >
            L (ซ้าย)
          </text>
          <text
            x="120"
            y="168"
            fontSize="10"
            fontWeight="800"
            fill={spec.side === 'right' || spec.side === 'both' ? '#059669' : '#94A3B8'}
          >
            R (ขวา)
          </text>
        </>
      )}
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

      {/* Main 2-Column Interactive Illustration + Step-by-Step Cues */}
      <div className="mt-3.5 grid grid-cols-1 items-center gap-4 md:grid-cols-12">
        {/* Left: Animated Anatomical Vector Stage */}
        <div className="md:col-span-4 flex flex-col items-center">
          <div className="relative flex h-44 w-44 items-center justify-center rounded-2xl border border-sky-200/80 bg-gradient-to-br from-white via-sky-50/70 to-emerald-50/60 p-2 shadow-inner">
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

        {/* Right: Interactive Phase Controls + 3 Movement Steps + Clinical Safety */}
        <div className="md:col-span-8 flex flex-col gap-2.5">
          {/* Interactive Phase Selector Pills */}
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

          {/* 3-Step Sequential Posture Guide Cards */}
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

          {/* ADL Badge + Posture Caution Footer */}
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
