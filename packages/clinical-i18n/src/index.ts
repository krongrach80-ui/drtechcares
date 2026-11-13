import React, { createContext, useContext, useState, type ReactNode } from 'react';

export type ClinicalLang = 'th' | 'en' | 'cn';

export interface TrilingualEntry {
  th: string;
  en: string;
  cn: string;
}

export const CLINICAL_TRANSLATIONS = {
  appName: { th: 'DRTECHCARES', en: 'DRTECHCARES', cn: 'DRTECHCARES' },
  appSubtitle: {
    th: 'แพลตฟอร์มฟื้นฟูสมรรถภาพกายภาพบำบัดและสมองอัจฉริยะ',
    en: 'Next-Gen AI Smart Hospital Rehabilitation Platform',
    cn: '新一代AI智慧医院康复与认知训练平台',
  },
  portalKiosk: { th: 'ตู้บริการผู้ป่วย (Kiosk)', en: 'AI Smart Kiosk', cn: '智能自助终端' },
  portalPt: { th: 'พอร์ทัลนักกายภาพบำบัด', en: 'PT Portal', cn: '物理治疗师门户' },
  portalAdmin: { th: 'ผู้บริหารและผู้ดูแลระบบ', en: 'Clinical Admin', cn: '临床管理后台' },
  touchlessLoginTitle: {
    th: 'ยืนยันตัวตนด้วยใบหน้าแบบไร้สัมผัส',
    en: 'Touchless AI Face Authentication',
    cn: '无接触AI人脸识别登录',
  },
  touchlessLoginSubtitle: {
    th: 'มองกล้องตรงๆ ระบบตรวจจับความมีชีวิตและสแกนอัตโนมัติในครั้งเดียว',
    en: 'Look straight at the camera for liveness check & single-scan verification',
    cn: '请直视摄像头，系统将自动进行活体检测并完成单次扫脸验证',
  },
  scanningBurst: {
    th: 'กำลังเก็บภาพสด 6 เฟรมเพื่อยืนยันตัวตน...',
    en: 'Capturing 6 live frames for biometric stabilization...',
    cn: '正在采集6帧实时画面进行生物特征稳定校验...',
  },
  accountNotFoundTitle: {
    th: 'ไม่พบบัญชีผู้ป่วยในระบบ',
    en: 'Patient Account Not Found',
    cn: '未找到患者账户',
  },
  accountNotFoundDesc: {
    th: 'ใบหน้านี้ยังไม่ถูกลงทะเบียนในฐานข้อมูลกลาง กรุณาติดต่อเจ้าหน้าที่เพื่อลงทะเบียนใหม่ (ไม่มีการสแกนซ้ำหรือขอรหัสบุคคลอื่น)',
    en: 'This face is not enrolled in the cloud database. Please ask staff to register a new profile.',
    cn: '该人脸尚未在云端数据库中注册，请联系工作人员协助建档。',
  },
  staffRegisterBtn: {
    th: 'เจ้าหน้าที่ลงทะเบียนผู้ป่วยใหม่',
    en: 'Staff: Enroll New Patient',
    cn: '工作人员：新患者建档注册',
  },
  callStaffBtn: {
    th: 'เรียกเจ้าหน้าที่',
    en: 'Call Staff',
    cn: '呼叫工作人员',
  },
  staffCalledConfirm: {
    th: 'ส่งสัญญาณเรียกเจ้าหน้าที่แบบเรียลไทม์แล้ว กรุณารอสักครู่',
    en: 'Realtime staff call dispatched. Assistance is on the way.',
    cn: '已实时呼叫工作人员，请稍候。',
  },
  trainingDayBadge: {
    th: 'วันฝึกกายภาพบำบัด',
    en: 'Training Day',
    cn: '康复训练日',
  },
  restDayBadge: {
    th: 'วันพักฟื้นกล้ามเนื้อ',
    en: 'Recovery / Rest Day',
    cn: '肌肉恢复休息日',
  },
  startRehabBtn: {
    th: 'เริ่มฝึกกายภาพบำบัดหน้ากล้อง',
    en: 'Start AI Camera Rehab',
    cn: '开始AI视觉康复训练',
  },
  startCognitiveGameBtn: {
    th: 'เล่นมินิเกมฝึกสมอง (การ์ดกายภาพ)',
    en: 'Cognitive Rehab Minigame',
    cn: '认知康复小游戏（记忆卡片）',
  },
  raiseBothHandsPrompt: {
    th: 'ยกมือทั้งสองข้างเหนือไหล่ค้าง 1.2 วินาที หรือพยักหน้าเพื่อยืนยันเริ่มฝึก',
    en: 'Raise both hands above shoulders for 1.2s or nod head to confirm',
    cn: '双手举过肩膀保持1.2秒或点头确认开始训练',
  },
  painEmergencyStop: {
    th: 'หยุดฉุกเฉินเมื่อมีอาการปวด',
    en: 'Emergency Pain Stop',
    cn: '疼痛紧急停止',
  },
  cleanCameraBadge: {
    th: 'มุมกล้องสะอาดตา (ไม่วาดโครงกระดูก)',
    en: 'Clean Mirror View (No Skeleton Overlay)',
    cn: '纯净镜像画面（无骨骼线遮挡）',
  },
  minigameRightYes: {
    th: 'ยกมือขวา = ใช่ (YES)',
    en: 'Raise Right Hand = YES',
    cn: '举右手 = 是 (YES)',
  },
  minigameLeftNo: {
    th: 'ยกมือซ้าย = ไม่ใช่ (NO)',
    en: 'Raise Left Hand = NO',
    cn: '举左手 = 否 (NO)',
  },
  bothHandsWarning: {
    th: 'กรุณายกเพียงข้างเดียว! ห้ามยก 2 มือพร้อมกัน',
    en: 'Please raise only ONE hand at a time!',
    cn: '请只举起一只手！切勿同时举起双手',
  },
  portraitCropWarning: {
    th: 'มืออยู่ชิดขอบจอเกินไป กรุณาขยับมือเข้าด้านในกรอบ',
    en: 'Hand near portrait edge; move hand closer to center',
    cn: '手部靠近竖屏边缘，请向画面中心移动',
  },
  ptEnrollTitle: {
    th: 'ลงทะเบียนผู้ป่วยใหม่ & สแกนใบหน้า 10 ช็อต',
    en: 'New Patient & 10-Shot Multi-Angle Enrollment',
    cn: '新患者建档与10角度人脸采集',
  },
  ptPrescribeTitle: {
    th: 'จัดตารางกายภาพบำบัดและชุดมินิเกม',
    en: 'Prescribe Rehab Schedule & Cognitive Modpack',
    cn: '制定康复课程表与认知题库包',
  },
  ptProgressTitle: {
    th: 'ติดตามพัฒนาการผู้ป่วยแบบเรียลไทม์',
    en: 'Realtime Patient Progress & Kinematics',
    cn: '患者康复进度实时追踪与运动学分析',
  },
  deletePatientBtn: {
    th: 'ลบข้อมูลผู้ป่วยถาวร (Cascading Purge)',
    en: 'Delete Patient Permanently',
    cn: '彻底删除患者档案（级联清除）',
  },
  logoutBtn: {
    th: 'ออกจากระบบ',
    en: 'Sign Out',
    cn: '退出登录',
  },
} as const satisfies Record<string, TrilingualEntry>;

export type TranslationKey = keyof typeof CLINICAL_TRANSLATIONS;

interface ClinicalI18nContextValue {
  lang: ClinicalLang;
  setLang: (lang: ClinicalLang) => void;
  t: (key: TranslationKey) => string;
  pick: (entry: TrilingualEntry) => string;
}

const ClinicalI18nContext = createContext<ClinicalI18nContextValue | null>(null);

export function ClinicalI18nProvider({
  children,
  initialLang = 'th',
}: {
  children: ReactNode;
  initialLang?: ClinicalLang;
}) {
  const [lang, setLangState] = useState<ClinicalLang>(() => {
    if (typeof window !== 'undefined') {
      const saved = window.localStorage.getItem('drtechcares_lang');
      if (saved === 'th' || saved === 'en' || saved === 'cn') return saved;
    }
    return initialLang;
  });

  const setLang = (next: ClinicalLang) => {
    setLangState(next);
    if (typeof window !== 'undefined') {
      window.localStorage.setItem('drtechcares_lang', next);
    }
  };

  const value: ClinicalI18nContextValue = {
    lang,
    setLang,
    t: (key) => CLINICAL_TRANSLATIONS[key][lang],
    pick: (entry) => entry[lang],
  };

  return React.createElement(ClinicalI18nContext.Provider, { value }, children);
}

export function useClinicalI18n(): ClinicalI18nContextValue {
  const ctx = useContext(ClinicalI18nContext);
  if (!ctx) {
    return {
      lang: 'th',
      setLang: () => undefined,
      t: (key) => CLINICAL_TRANSLATIONS[key].th,
      pick: (entry) => entry.th,
    };
  }
  return ctx;
}

