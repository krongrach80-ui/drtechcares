import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ApiError } from '@drtechcares/api-client';
import type {
  ActiveSessionRecord,
  AuditLogRecord,
  BannedTargetRecord,
  DailyScheduleDay,
  DrTechCaresApiClient,
  KioskStatusRecord,
  ManagedExerciseRecord,
  ModpackRecord,
  PatientCourseRecord,
  PatientRecord,
  RehabScheduleItem,
  RehabSessionRecord,
  StaffCallRecord,
  SystemSettingsRecord,
  UserAccount,
  UserRole,
} from '@drtechcares/api-client';
import { useClinicalI18n, type ClinicalLang } from '@drtechcares/clinical-i18n';
import type { BodySide, ExerciseConfig } from '@drtechcares/pose-engine';
import {
  Activity,
  Ban,
  Bell,
  Calendar,
  CalendarDays,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  Clock,
  FileText,
  Gamepad2,
  HeartHandshake,
  KeyRound,
  LogOut,
  Monitor,
  PauseCircle,
  Plus,
  RefreshCw,
  ScanFace,
  Search,
  Shield,
  ShieldAlert,
  SlidersHorizontal,
  Trash2,
  UserCheck,
  UserPlus,
  Users,
  Wifi,
  WifiOff,
  X,
  Zap,
} from 'lucide-react';
import { PhysicalCardGame } from '../../apps/kiosk/src/PhysicalCardGame';
import { notifyBiometricIndexChanged } from '../lib/faceMatcher';
import { StaffAssistedEnrollmentModal } from './StaffAssistedEnrollmentModal';
import {
  PatientProfileAndHistoryModal,
  ResetPasswordModal,
  ScheduleCreatorAndDayModal,
  WelcomeFaceScanModal,
} from './PortalSubModals';

export interface ClinicalStaffPortalShellProps {
  apiClient: DrTechCaresApiClient;
  currentUser: UserAccount;
  onLogout: () => Promise<void> | void;
  onReturnToKiosk?: () => void;
  autoOpenEnroll?: boolean;
  onAutoOpenEnrollConsumed?: () => void;
}

type PortalTab = 'staff' | 'patients' | 'pts' | 'exercises' | 'schedules' | 'security';
type SecuritySubTab = 'calls' | 'audit' | 'sessions' | 'bans' | 'kiosks';

interface PortalSectionCopy {
  nav: Record<PortalTab, string>;
  headers: Record<PortalTab, { title: string; subtitle: string }>;
  sidebar: {
    adminRole: string;
    ptRole: string;
    collapseTitle: string;
    dbCloud: string;
    dbNormal: string;
    dbOffline: string;
  };
  actions: {
    welcomeScan: string;
    quickCheckin: string;
    kioskBtn: string;
    returnKiosk: string;
    logout: string;
    cardGame: string;
    cloudConnected: string;
    cloudOffline: string;
  };
  staff: {
    title: string;
    subtitle: string;
    addBtn: string;
    refreshTitle: string;
    searchPlaceholder: string;
    allRoles: string;
    roleAdmin: string;
    rolePt: string;
    rolePatient: string;
    allStatus: string;
    statusActive: string;
    statusSuspended: string;
    colName: string;
    colUsername: string;
    colRole: string;
    colStatus: string;
    colActions: string;
    statusBadgeActive: string;
    statusBadgeSuspended: string;
    suspendBtn: string;
    activateBtn: string;
    resetPwdTitle: string;
    deleteTitle: string;
    emptyStaff: string;
  };
  patients: {
    title: string;
    subtitle: string;
    addBtn: string;
    searchPlaceholder: string;
    searchBtn: string;
    allFace: string;
    faceVerified: string;
    facePending: string;
    colHn: string;
    colPatientName: string;
    colAgeGender: string;
    colPhonePt: string;
    colBiometric: string;
    colActions: string;
    verifiedBadge: string;
    pendingBadge: string;
    emrBtn: string;
    scanFaceBtn: string;
    historyBtn: string;
    deleteTitle: string;
    emptyPatients: string;
    ageUnit: string;
  };
  pts: {
    title: string;
    subtitle: string;
    searchPlaceholder: string;
    licenseLabel: string;
    specialtyLabel: string;
    patientsCountLabel: string;
    patientsUnit: string;
    viewDetailBtn: string;
  };
  exercises: {
    title: string;
    subtitle: string;
    playCardBtn: string;
    aiPoseTitle: string;
    aiPoseEnabled: string;
    aiPoseLocked: string;
    aiPoseDesc: string;
    aiPoseToggleLock: string;
    aiPoseToggleUnlock: string;
    modpackHeader: string;
    addModpackBtn: string;
    exerciseHeader: string;
    addExerciseBtn: string;
  };
  schedules: {
    activeBadge: string;
    noPatient: string;
    diagLabel: string;
    ptLabel: string;
    clearScheduleBtn: string;
    weeklyPlanBtn: string;
    createScheduleBtn: string;
    subIndividual: string;
    subAll: string;
    statTotal: string;
    statActive: string;
    statRest: string;
    statCompleted: string;
    unitAppt: string;
    legendActive: string;
    legendRest: string;
    legendCompleted: string;
    legendMissed: string;
    colHn: string;
    colPatientName: string;
    colPt: string;
    colNextAppt: string;
    colActions: string;
    openCalendarBtn: string;
    statusCompleted: string;
    statusRest: string;
    statusMissed: string;
    statusActive: string;
    restDayLabel: string;
    calendarHeaders: string[];
  };
  security: {
    title: string;
    subtitle: string;
    testAlertBtn: string;
    emergencyBannerTitle: string;
    emergencyBannerDesc: string;
    viewCallsBtn: string;
    tabCalls: string;
    tabAudit: string;
    tabSessions: string;
    tabBans: string;
    tabKiosks: string;
    callsTableTitle: string;
    callsPollNote: string;
    colTime: string;
    colKiosk: string;
    colReason: string;
    colStatus: string;
    colActions: string;
    ackBtn: string;
    resolveBtn: string;
    statusResolved: string;
    statusAck: string;
    statusOpen: string;
    emptyCalls: string;
    registerKioskBtn: string;
    kioskPanelTitle: string;
    securitySettingsTitle: string;
  };
}

const PORTAL_TEXT: Record<ClinicalLang, PortalSectionCopy> = {
  th: {
    nav: {
      staff: '1. จัดการผู้ใช้ staff',
      patients: '2. ข้อมูลคนไข้',
      pts: '3. ข้อมูลนักกายภาพ',
      exercises: '4. ท่ากายภาพและมินิเกม/modpack',
      schedules: '5. ตารางนัด',
      security: '6. ประวัติและความปลอดภัย',
    },
    headers: {
      staff: {
        title: 'จัดการผู้ใช้และบุคลากรโรงพยาบาล',
        subtitle: 'HOSPITAL STAFF & USER MANAGEMENT PORTAL',
      },
      patients: {
        title: 'ทะเบียนประวัติและเวชระเบียนคนไข้',
        subtitle: 'PATIENT DIRECTORY & ELECTRONIC HEALTH RECORDS',
      },
      pts: {
        title: 'ข้อมูลนักกายภาพบำบัดและการกำกับดูแล',
        subtitle: 'PHYSIOTHERAPIST DIRECTORY & CLINICAL GOVERNANCE',
      },
      exercises: {
        title: 'คลังท่ากายภาพบำบัดและมินิเกม Modpack',
        subtitle: 'REHAB EXERCISE LIBRARY & COGNITIVE MODPACKS',
      },
      schedules: {
        title: 'ตารางฝึกกายภาพและปฏิทินการฟื้นฟู',
        subtitle: 'CLINICAL PHYSICAL THERAPY SCHEDULE PLANNER',
      },
      security: {
        title: 'ประวัติการใช้งานและบันทึกความปลอดภัย',
        subtitle: 'TAMPER-EVIDENT SYSTEM AUDIT TRAIL & SECURITY',
      },
    },
    sidebar: {
      adminRole: 'ผู้อำนวยการ (Admin)',
      ptRole: 'นักกายภาพบำบัด (PT)',
      collapseTitle: 'ย่อ/ขยายเมนู',
      dbCloud: 'ฐานข้อมูล: Cloud',
      dbNormal: 'ปกติ',
      dbOffline: 'ออฟไลน์',
    },
    actions: {
      welcomeScan: 'ต้อนรับ & สแกนหน้า',
      quickCheckin: 'เช็คอินเร่งด่วน',
      kioskBtn: 'หน้าตู้ Kiosk',
      returnKiosk: '← หน้าตู้ Kiosk',
      logout: 'ออกจากระบบ',
      cardGame: 'การ์ดกายภาพ (10 ข้อ)',
      cloudConnected: 'Supabase Cloud',
      cloudOffline: 'Offline (ลองใหม่)',
    },
    staff: {
      title: 'จัดการเจ้าหน้าที่โรงพยาบาล (Staff Management)',
      subtitle: 'บัญชีผู้อำนวยการ (Admin) และนักกายภาพบำบัด (PT) ควบคุมสิทธิ์และเซสชันการเข้าถึง',
      addBtn: '+ เพิ่มเจ้าหน้าที่ใหม่',
      refreshTitle: 'รีเฟรช',
      searchPlaceholder: 'ค้นหาชื่อ, username, เบอร์โทร...',
      allRoles: 'ทุกบทบาท (All Roles)',
      roleAdmin: 'ผู้อำนวยการ (Admin)',
      rolePt: 'นักกายภาพบำบัด (PT)',
      rolePatient: 'คนไข้ (Patient)',
      allStatus: 'ทุกสถานะ (All Status)',
      statusActive: 'ปกติ (Active)',
      statusSuspended: 'ระงับใช้งาน (Suspended)',
      colName: 'ชื่อ-นามสกุล',
      colUsername: 'USERNAME / HN',
      colRole: 'บทบาท',
      colStatus: 'สถานะ',
      colActions: 'การจัดการ',
      statusBadgeActive: 'ปกติ',
      statusBadgeSuspended: 'ระงับ',
      suspendBtn: 'ระงับสิทธิ์',
      activateBtn: 'เปิดใช้งาน',
      resetPwdTitle: 'รีเซ็ตรหัสผ่าน',
      deleteTitle: 'ลบผู้ใช้',
      emptyStaff: 'ไม่พบบัญชีผู้ใช้ตามเงื่อนไขที่ค้นหา',
    },
    patients: {
      title: 'ทะเบียนคนไข้ (Patients Directory)',
      subtitle: 'ค้นหาคนไข้ กำหนด PT ผู้รับผิดชอบ ส่งตัวสแกนหน้า และดูประวัติการฝึกย้อนหลัง',
      addBtn: '+ ลงทะเบียนคนไข้ใหม่',
      searchPlaceholder: 'ค้นหาชื่อไทย, นามสกุล, HN, เบอร์โทร..',
      searchBtn: 'ค้นหา',
      allFace: 'ทุกสถานะใบหน้า',
      faceVerified: 'ลงทะเบียนใบหน้าแล้ว (VERIFIED)',
      facePending: 'ยังไม่สแกนใบหน้า (PENDING)',
      colHn: 'รหัส HN',
      colPatientName: 'ชื่อ - นามสกุล',
      colAgeGender: 'อายุ / เพศ',
      colPhonePt: 'เบอร์โทร / PT ผู้ดูแล',
      colBiometric: 'สถานะชีวมาตร',
      colActions: 'การจัดการ',
      verifiedBadge: 'ลงทะเบียนหน้าแล้ว',
      pendingBadge: 'ยังไม่สแกนใบหน้า',
      emrBtn: 'เวชระเบียน',
      scanFaceBtn: 'ส่งตัวสแกนหน้า',
      historyBtn: 'ดูประวัติการฝึก',
      deleteTitle: 'ลบประวัติ (เฉพาะ Admin)',
      emptyPatients: 'ไม่พบข้อมูลคนไข้ กด "+ ลงทะเบียนคนไข้ใหม่" เพื่อเพิ่มผู้ป่วย',
      ageUnit: 'ปี',
    },
    pts: {
      title: 'ทำเนียบนักกายภาพบำบัด (PT Directory)',
      subtitle: 'รายชื่อนักกายภาพบำบัดประจำโรงพยาบาล เลขที่ใบอนุญาตประกอบวิชาชีพ และคนไข้ที่ดูแล',
      searchPlaceholder: 'ค้นหาชื่อ PT, เลขที่ใบอนุญาต, ความเชี่ยวชาญ...',
      licenseLabel: '🏅 ใบประกอบวิชาชีพ:',
      specialtyLabel: 'ความเชี่ยวชาญ:',
      patientsCountLabel: 'คนไข้ที่ดูแล:',
      patientsUnit: 'ท่าน',
      viewDetailBtn: 'ดูรายละเอียดและคนไข้',
    },
    exercises: {
      title: 'ท่ากายภาพบำบัดและมินิเกมกระตุ้นสมอง (Exercises & Modpacks)',
      subtitle: 'จัดการชุดคำถาม Physical Card Modpack และปลดล็อกโหมด AI Pose Detection สำหรับตู้ Kiosk',
      playCardBtn: 'ทดลองเล่นการ์ดกายภาพ 10 ข้อ',
      aiPoseTitle: 'สถานะตรวจจับท่ากายภาพอัตโนมัติ (AI Pose Detection):',
      aiPoseEnabled: 'เปิดใช้งาน AI Pose Detection สำหรับตู้ Kiosk แล้ว',
      aiPoseLocked: 'อยู่ระหว่างพัฒนา (PT ล็อกไว้เฉพาะมินิเกม)',
      aiPoseDesc: 'เฉพาะผู้อำนวยการโรงพยาบาล (ADMIN) สามารถปลดล็อกเปิดใช้ท่ากายภาพ AI บนตู้ Kiosk ได้',
      aiPoseToggleLock: 'ตั้งสถานะเป็น: อยู่ระหว่างพัฒนา',
      aiPoseToggleUnlock: 'ปลดล็อกเปิดใช้ท่ากายภาพ AI',
      modpackHeader: 'ชุดมินิเกม Modpack',
      addModpackBtn: '+ เพิ่ม Modpack มินิเกม',
      exerciseHeader: 'คลังท่ากายภาพบำบัด',
      addExerciseBtn: '+ เพิ่มท่ากายภาพบำบัด',
    },
    schedules: {
      activeBadge: 'กำลังเข้ารับการฟื้นฟู (Active)',
      noPatient: 'ยังไม่มีคนไข้ในระบบ',
      diagLabel: 'การวินิจฉัย:',
      ptLabel: 'PT ผู้รับผิดชอบ:',
      clearScheduleBtn: 'ล้างตารางนัดทั้งหมด',
      weeklyPlanBtn: 'แผนรายสัปดาห์ 7 วัน',
      createScheduleBtn: '+ สร้างตารางฝึกใหม่',
      subIndividual: 'ปฏิทินรายบุคคล',
      subAll: 'ปฏิทินคนไข้รวม',
      statTotal: 'ตารางนัดหมายทั้งหมด',
      statActive: 'กำลังรักษา (Active)',
      statRest: 'พักชั่วคราว (Rest)',
      statCompleted: 'สำเร็จแล้ว (Completed)',
      unitAppt: 'นัด',
      legendActive: 'กำลังรักษา',
      legendRest: 'พักการฝึก',
      legendCompleted: 'สำเร็จครบโปรแกรม',
      legendMissed: 'ขาดการฝึก',
      colHn: 'HN',
      colPatientName: 'ชื่อคนไข้',
      colPt: 'PT ผู้รับผิดชอบ',
      colNextAppt: 'วันนัดถัดไป / จำนวนวันฝึก',
      colActions: 'จัดการ',
      openCalendarBtn: 'เปิดปฏิทิน',
      statusCompleted: 'สำเร็จครบ',
      statusRest: 'พักการฝึก',
      statusMissed: 'ขาดการฝึก',
      statusActive: 'กำลังรักษา',
      restDayLabel: 'พักฟื้น',
      calendarHeaders: ['อา.', 'จ.', 'อ.', 'พ.', 'พฤ.', 'ศ.', 'ส.'],
    },
    security: {
      title: 'ประวัติและความปลอดภัย (Audit, Bans & Security)',
      subtitle: 'คำเรียกเจ้าหน้าที่ฉุกเฉิน (Realtime), บันทึก Audit Log, บังคับตัดเซสชัน, ระบบแบน และจัดการตู้ Kiosk',
      testAlertBtn: '🔔 ทดสอบแจ้งเตือนฉุกเฉิน',
      emergencyBannerTitle: 'มีคนไข้ต้องการความช่วยเหลือที่ตู้ Kiosk',
      emergencyBannerDesc: 'เชื่อมต่อ Realtime Supabase + Polling อัตโนมัติเมื่อคนไข้กดเรียกเจ้าหน้าที่',
      viewCallsBtn: 'ดูรายการเรียกเจ้าหน้าที่ทันที',
      tabCalls: 'คำเรียกเจ้าหน้าที่',
      tabAudit: 'บันทึก Audit Logs',
      tabSessions: 'เซสชันและการเตะออก',
      tabBans: 'การจัดการแบน',
      tabKiosks: 'อุปกรณ์ตู้ Kiosk',
      callsTableTitle: 'รายการคำขอความช่วยเหลือจากตู้ Kiosk',
      callsPollNote: 'Realtime + Polling 10 วินาที',
      colTime: 'เวลาแจ้ง',
      colKiosk: 'ตู้ KIOSK',
      colReason: 'สาเหตุ',
      colStatus: 'สถานะ',
      colActions: 'ดำเนินการ',
      ackBtn: 'รับเรื่อง',
      resolveBtn: 'แก้ไขแล้ว',
      statusResolved: '✓ แก้ไขแล้ว',
      statusAck: '⏳ กำลังตรวจสอบ',
      statusOpen: '🔔 รอเจ้าหน้าที่',
      emptyCalls: 'ยังไม่มีคำเรียกเจ้าหน้าที่ฉุกเฉิน',
      registerKioskBtn: '+ ลงทะเบียนตู้ Kiosk ใหม่',
      kioskPanelTitle: 'อุปกรณ์ตู้ Kiosk ในระบบ',
      securitySettingsTitle: 'ตั้งค่าความปลอดภัย AI หน้าตู้ Kiosk',
    },
  },
  en: {
    nav: {
      staff: '1. Staff Management',
      patients: '2. Patients Directory',
      pts: '3. PT Directory',
      exercises: '4. Exercises & Modpacks',
      schedules: '5. Rehab Schedules',
      security: '6. Audit & Security',
    },
    headers: {
      staff: {
        title: 'Hospital Staff & User Management',
        subtitle: 'HOSPITAL STAFF & USER MANAGEMENT PORTAL',
      },
      patients: {
        title: 'Patients Directory & Medical Records',
        subtitle: 'PATIENT DIRECTORY & ELECTRONIC HEALTH RECORDS',
      },
      pts: {
        title: 'Physiotherapist Directory & Governance',
        subtitle: 'PHYSIOTHERAPIST DIRECTORY & CLINICAL GOVERNANCE',
      },
      exercises: {
        title: 'Rehab Exercises & Cognitive Modpacks',
        subtitle: 'REHAB EXERCISE LIBRARY & COGNITIVE MODPACKS',
      },
      schedules: {
        title: 'Clinical Physical Therapy Schedule Planner',
        subtitle: 'CLINICAL PHYSICAL THERAPY SCHEDULE PLANNER',
      },
      security: {
        title: 'System Audit Trail, Bans & Kiosk Security',
        subtitle: 'TAMPER-EVIDENT SYSTEM AUDIT TRAIL & SECURITY',
      },
    },
    sidebar: {
      adminRole: 'Director (Admin)',
      ptRole: 'Physiotherapist (PT)',
      collapseTitle: 'Toggle Sidebar',
      dbCloud: 'Database: Cloud',
      dbNormal: 'Online',
      dbOffline: 'Offline',
    },
    actions: {
      welcomeScan: 'Welcome & Scan',
      quickCheckin: 'Quick Check-in',
      kioskBtn: 'Kiosk View',
      returnKiosk: '← Back to Kiosk',
      logout: 'Sign Out',
      cardGame: 'Rehab Cards (10 Qs)',
      cloudConnected: 'Supabase Cloud',
      cloudOffline: 'Offline (Retry)',
    },
    staff: {
      title: 'Hospital Staff Management',
      subtitle: 'Manage Admin directors and Physiotherapists (PT), access roles, and active sessions',
      addBtn: '+ Add New Staff',
      refreshTitle: 'Refresh',
      searchPlaceholder: 'Search name, username, phone...',
      allRoles: 'All Roles',
      roleAdmin: 'Director (Admin)',
      rolePt: 'Physiotherapist (PT)',
      rolePatient: 'Patient',
      allStatus: 'All Status',
      statusActive: 'Active',
      statusSuspended: 'Suspended',
      colName: 'FULL NAME',
      colUsername: 'USERNAME / HN',
      colRole: 'ROLE',
      colStatus: 'STATUS',
      colActions: 'ACTIONS',
      statusBadgeActive: 'Active',
      statusBadgeSuspended: 'Suspended',
      suspendBtn: 'Suspend',
      activateBtn: 'Activate',
      resetPwdTitle: 'Reset Password',
      deleteTitle: 'Delete User',
      emptyStaff: 'No matching staff accounts found',
    },
    patients: {
      title: 'Patients Directory',
      subtitle: 'Search patients, assign PTs, dispatch 120-frame Face ID scan, and inspect rehab history',
      addBtn: '+ Enroll New Patient',
      searchPlaceholder: 'Search name, HN, phone...',
      searchBtn: 'Search',
      allFace: 'All Face Status',
      faceVerified: 'Face Enrolled (VERIFIED)',
      facePending: 'Pending Scan (PENDING)',
      colHn: 'HN CODE',
      colPatientName: 'PATIENT NAME',
      colAgeGender: 'AGE / GENDER',
      colPhonePt: 'PHONE / ASSIGNED PT',
      colBiometric: 'BIOMETRIC STATUS',
      colActions: 'ACTIONS',
      verifiedBadge: 'Face Verified',
      pendingBadge: 'Scan Pending',
      emrBtn: 'EMR',
      scanFaceBtn: 'Face Scan',
      historyBtn: 'History',
      deleteTitle: 'Delete Record (Admin Only)',
      emptyPatients: 'No patients found. Click "+ Enroll New Patient" to register.',
      ageUnit: 'yrs',
    },
    pts: {
      title: 'Physiotherapist Directory',
      subtitle: 'Registered hospital physiotherapists, medical license numbers, and assigned patients',
      searchPlaceholder: 'Search PT name, license no., specialty...',
      licenseLabel: '🏅 License No:',
      specialtyLabel: 'Specialty:',
      patientsCountLabel: 'Patients:',
      patientsUnit: 'patients',
      viewDetailBtn: 'View Details & Patients',
    },
    exercises: {
      title: 'Rehab Exercises & Cognitive Modpacks',
      subtitle: 'Manage Physical Card Modpacks and unlock AI Pose Detection mode for Kiosks',
      playCardBtn: 'Preview Rehab Cards (10 Qs)',
      aiPoseTitle: 'AI Pose Detection Status:',
      aiPoseEnabled: 'Enabled on Kiosk Terminals',
      aiPoseLocked: 'In Development (Card Minigames Only)',
      aiPoseDesc: 'Only Hospital Directors (ADMIN) can unlock AI Pose Detection for Kiosk terminals',
      aiPoseToggleLock: 'Set Status: Locked',
      aiPoseToggleUnlock: 'Unlock AI Pose Detection',
      modpackHeader: 'Cognitive Modpacks',
      addModpackBtn: '+ Add Modpack',
      exerciseHeader: 'Rehab Exercises Library',
      addExerciseBtn: '+ Add Rehab Exercise',
    },
    schedules: {
      activeBadge: 'Active Rehabilitation',
      noPatient: 'No Patients Enrolled',
      diagLabel: 'Diagnosis:',
      ptLabel: 'Assigned PT:',
      clearScheduleBtn: 'Clear All Schedules',
      weeklyPlanBtn: '7-Day Weekly Plan',
      createScheduleBtn: '+ New Rehab Schedule',
      subIndividual: 'Individual Calendar',
      subAll: 'All Patients',
      statTotal: 'Total Schedules',
      statActive: 'Active',
      statRest: 'Rest Days',
      statCompleted: 'Completed',
      unitAppt: 'slots',
      legendActive: 'Active',
      legendRest: 'Rest',
      legendCompleted: 'Completed',
      legendMissed: 'Missed',
      colHn: 'HN',
      colPatientName: 'PATIENT NAME',
      colPt: 'ASSIGNED PT',
      colNextAppt: 'NEXT APPT / SLOTS',
      colActions: 'ACTION',
      openCalendarBtn: 'Open Calendar',
      statusCompleted: 'Completed',
      statusRest: 'Rest',
      statusMissed: 'Missed',
      statusActive: 'Active',
      restDayLabel: 'Rest Day',
      calendarHeaders: ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'],
    },
    security: {
      title: 'Audit, Bans & Kiosk Security',
      subtitle: 'Realtime Kiosk emergency alerts, tamper-evident audit logs, session kick, bans & terminal status',
      testAlertBtn: '🔔 Test Emergency Alert',
      emergencyBannerTitle: 'Patient Assistance Requested at Kiosk',
      emergencyBannerDesc: 'Synced via Realtime Supabase + 10s automatic polling',
      viewCallsBtn: 'View Staff Calls Now',
      tabCalls: 'Staff Calls',
      tabAudit: 'Audit Logs',
      tabSessions: 'Active Sessions',
      tabBans: 'Bans Manager',
      tabKiosks: 'Kiosk Terminals',
      callsTableTitle: 'Kiosk Emergency Call Queue',
      callsPollNote: 'Realtime + 10s Polling',
      colTime: 'TIMESTAMP',
      colKiosk: 'KIOSK ID',
      colReason: 'REASON',
      colStatus: 'STATUS',
      colActions: 'ACTION',
      ackBtn: 'Acknowledge',
      resolveBtn: 'Resolve',
      statusResolved: '✓ Resolved',
      statusAck: '⏳ Checking',
      statusOpen: '🔔 Waiting Staff',
      emptyCalls: 'No emergency staff calls',
      registerKioskBtn: '+ Register Kiosk',
      kioskPanelTitle: 'Registered Kiosk Terminals',
      securitySettingsTitle: 'Kiosk AI Security Settings',
    },
  },
  cn: {
    nav: {
      staff: '1. 医护人员管理',
      patients: '2. 患者档案目录',
      pts: '3. 康复师团队',
      exercises: '4. 康复动作与题库',
      schedules: '5. 康复日程安排',
      security: '6. 审计与安全中心',
    },
    headers: {
      staff: {
        title: '医院员工与账户权限管理',
        subtitle: 'HOSPITAL STAFF & USER MANAGEMENT PORTAL',
      },
      patients: {
        title: '患者电子健康档案目录',
        subtitle: 'PATIENT DIRECTORY & ELECTRONIC HEALTH RECORDS',
      },
      pts: {
        title: '物理治疗师名录与临床督导',
        subtitle: 'PHYSIOTHERAPIST DIRECTORY & CLINICAL GOVERNANCE',
      },
      exercises: {
        title: '物理治疗动作库与认知训练包',
        subtitle: 'REHAB EXERCISE LIBRARY & COGNITIVE MODPACKS',
      },
      schedules: {
        title: '临床康复训练日程表',
        subtitle: 'CLINICAL PHYSICAL THERAPY SCHEDULE PLANNER',
      },
      security: {
        title: '系统安全审计日志与终端监控',
        subtitle: 'TAMPER-EVIDENT SYSTEM AUDIT TRAIL & SECURITY',
      },
    },
    sidebar: {
      adminRole: '医院管理员 (Admin)',
      ptRole: '物理治疗师 (PT)',
      collapseTitle: '展开/收起菜单',
      dbCloud: '数据库: 云端',
      dbNormal: '正常',
      dbOffline: '离线',
    },
    actions: {
      welcomeScan: '迎宾与人脸扫描',
      quickCheckin: '快速签到',
      kioskBtn: 'Kiosk 终端',
      returnKiosk: '← 返回终端画面',
      logout: '退出登录',
      cardGame: '康复认知卡 (10题)',
      cloudConnected: 'Supabase 云端在线',
      cloudOffline: '离线 (点击重试)',
    },
    staff: {
      title: '医院工作人员与权限管理 (Staff Management)',
      subtitle: '管理管理员 (Admin) 与物理治疗师 (PT) 账号、权限及在线会话控制',
      addBtn: '+ 新增医护人员',
      refreshTitle: '刷新',
      searchPlaceholder: '搜索姓名、用户名、手机号...',
      allRoles: '全部角色 (All Roles)',
      roleAdmin: '管理员 (Admin)',
      rolePt: '物理治疗师 (PT)',
      rolePatient: '患者 (Patient)',
      allStatus: '全部状态 (All Status)',
      statusActive: '正常 (Active)',
      statusSuspended: '已停用 (Suspended)',
      colName: '姓名',
      colUsername: '用户名 / HN',
      colRole: '角色',
      colStatus: '状态',
      colActions: '操作管理',
      statusBadgeActive: '正常',
      statusBadgeSuspended: '已停用',
      suspendBtn: '停用权限',
      activateBtn: '启用账户',
      resetPwdTitle: '重置密码',
      deleteTitle: '删除账户',
      emptyStaff: '未找到符合条件的账户',
    },
    patients: {
      title: '患者电子档案目录 (Patients Directory)',
      subtitle: '查询患者、指派责任康复师 (PT)、发起 120 帧人脸采集及查看康复记录',
      addBtn: '+ 注册新患者',
      searchPlaceholder: '搜索患者姓名、HN病历号、手机号...',
      searchBtn: '搜索',
      allFace: '全部人脸状态',
      faceVerified: '已完成人脸注册 (VERIFIED)',
      facePending: '待人脸采集 (PENDING)',
      colHn: '病历号 HN',
      colPatientName: '患者姓名',
      colAgeGender: '年龄 / 性别',
      colPhonePt: '联系电话 / 责任PT',
      colBiometric: '生物识别状态',
      colActions: '操作管理',
      verifiedBadge: '人脸已验证',
      pendingBadge: '待扫描人脸',
      emrBtn: '电子病历',
      scanFaceBtn: '人脸采集',
      historyBtn: '训练记录',
      deleteTitle: '删除病历 (仅管理员)',
      emptyPatients: '暂无患者记录，请点击“+ 注册新患者”新增档案',
      ageUnit: '岁',
    },
    pts: {
      title: '物理治疗师团队名录 (PT Directory)',
      subtitle: '在院注册物理治疗师名单、执业资格证号与负责患者数量统计',
      searchPlaceholder: '搜索康复师姓名、执业证号、专业方向...',
      licenseLabel: '🏅 执业证号:',
      specialtyLabel: '临床专长:',
      patientsCountLabel: '负责患者数:',
      patientsUnit: '位',
      viewDetailBtn: '查看详情与患者名单',
    },
    exercises: {
      title: '康复训练动作库与认知训练包 (Exercises & Modpacks)',
      subtitle: '管理康复认知卡片题库 Modpack 及开启自助终端 AI 姿态识别模式',
      playCardBtn: '试玩康复认知卡 (10题)',
      aiPoseTitle: 'AI 自动姿态检测状态 (AI Pose Detection):',
      aiPoseEnabled: '自助终端已启用 AI 姿态识别',
      aiPoseLocked: '开发中（暂限认知游戏模式）',
      aiPoseDesc: '仅医院管理员 (ADMIN) 可为 Kiosk 自助终端解锁 AI 康复动作姿态检测',
      aiPoseToggleLock: '设为锁定维护状态',
      aiPoseToggleUnlock: '解锁开启 AI 姿态识别',
      modpackHeader: '认知训练题库包',
      addModpackBtn: '+ 新增题库 Modpack',
      exerciseHeader: '康复动作标准库',
      addExerciseBtn: '+ 新增康复动作',
    },
    schedules: {
      activeBadge: '正在康复训练 (Active)',
      noPatient: '系统暂无患者',
      diagLabel: '临床诊断:',
      ptLabel: '责任康复师:',
      clearScheduleBtn: '清空全部日程',
      weeklyPlanBtn: '每周 7 天计划',
      createScheduleBtn: '+ 新建康复日程',
      subIndividual: '个人康复日历',
      subAll: '全部患者总览',
      statTotal: '总预约课程',
      statActive: '进行中 (Active)',
      statRest: '休息恢复 (Rest)',
      statCompleted: '已完成 (Completed)',
      unitAppt: '次',
      legendActive: '训练中',
      legendRest: '休息日',
      legendCompleted: '已完成',
      legendMissed: '缺席',
      colHn: 'HN 编号',
      colPatientName: '患者姓名',
      colPt: '责任康复师',
      colNextAppt: '下次预约 / 总次数',
      colActions: '操作',
      openCalendarBtn: '打开日历',
      statusCompleted: '已完成',
      statusRest: '休息',
      statusMissed: '缺席',
      statusActive: '训练中',
      restDayLabel: '休息日',
      calendarHeaders: ['周日', '周一', '周二', '周三', '周四', '周五', '周六'],
    },
    security: {
      title: '安全审计日志、封禁与终端管理 (Security)',
      subtitle: '终端实时紧急呼叫、不可篡改审计日志 (Audit Log)、强制踢出会话、封禁及 Kiosk 管理',
      testAlertBtn: '🔔 测试紧急呼叫警报',
      emergencyBannerTitle: '有患者在 Kiosk 终端请求协助',
      emergencyBannerDesc: '已通过 Supabase Realtime 与 10 秒轮询自动同步警报',
      viewCallsBtn: '立即查看呼叫列表',
      tabCalls: '呼叫请求',
      tabAudit: '审计日志 Audit',
      tabSessions: '在线会话管理',
      tabBans: '封禁名单管理',
      tabKiosks: 'Kiosk 终端设备',
      callsTableTitle: 'Kiosk 自助终端求助队列',
      callsPollNote: 'Realtime + 每10秒同步',
      colTime: '告警时间',
      colKiosk: '终端编号',
      colReason: '呼叫原因',
      colStatus: '当前状态',
      colActions: '处理操作',
      ackBtn: '受理',
      resolveBtn: '已处理',
      statusResolved: '✓ 已完成处理',
      statusAck: '⏳ 核查处理中',
      statusOpen: '🔔 待响应',
      emptyCalls: '暂无终端紧急呼叫记录',
      registerKioskBtn: '+ 注册新 Kiosk 终端',
      kioskPanelTitle: '已注册 Kiosk 设备列表',
      securitySettingsTitle: 'Kiosk AI 安全识别阈值配置',
    },
  },
};

const THAI_MONTHS = [
  'มกราคม',
  'กุมภาพันธ์',
  'มีนาคม',
  'เมษายน',
  'พฤษภาคม',
  'มิถุนายน',
  'กรกฎาคม',
  'สิงหาคม',
  'กันยายน',
  'ตุลาคม',
  'พฤศจิกายน',
  'ธันวาคม',
];

const EN_MONTHS = [
  'January',
  'February',
  'March',
  'April',
  'May',
  'June',
  'July',
  'August',
  'September',
  'October',
  'November',
  'December',
];

const CN_MONTHS = ['1月', '2月', '3月', '4月', '5月', '6月', '7月', '8月', '9月', '10月', '11月', '12月'];

const WEEKDAY_NAMES_TH = ['อาทิตย์', 'จันทร์', 'อังคาร', 'พุธ', 'พฤหัสบดี', 'ศุกร์', 'เสาร์'];

function formatThaiDateTime(iso: string): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  const day = date.getDate().toString().padStart(2, '0');
  const month = ['ม.ค.', 'ก.พ.', 'มี.ค.', 'เม.ย.', 'พ.ค.', 'มิ.ย.', 'ก.ค.', 'ส.ค.', 'ก.ย.', 'ต.ค.', 'พ.ย.', 'ธ.ค.'][
    date.getMonth()
  ];
  const beYear = date.getFullYear() + 543;
  const hh = date.getHours().toString().padStart(2, '0');
  const mm = date.getMinutes().toString().padStart(2, '0');
  return `${day} ${month} ${beYear} • ${hh}:${mm} น.`;
}

function isPatientFaceVerified(p: PatientRecord): boolean {
  if (p.faceBiometricStatus === 'PENDING') return false;
  if (p.faceBiometricStatus === 'VERIFIED') return true;
  return Boolean(p.biometricHistory?.raw_frames?.length);
}

function playEmergencyChime() {
  try {
    const AudioCtx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
    if (!AudioCtx) return;
    const ctx = new AudioCtx();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(880, ctx.currentTime);
    osc.frequency.setValueAtTime(660, ctx.currentTime + 0.14);
    gain.gain.setValueAtTime(0.12, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.35);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start();
    osc.stop(ctx.currentTime + 0.36);
  } catch {
    // ignore audio autoplay restrictions
  }
}

export function ClinicalStaffPortalShell({
  apiClient,
  currentUser,
  onLogout,
  onReturnToKiosk,
  autoOpenEnroll,
  onAutoOpenEnrollConsumed,
}: ClinicalStaffPortalShellProps) {
  const { lang, setLang } = useClinicalI18n();
  const locale = PORTAL_TEXT[lang];
  const isAdmin = currentUser.role === 'super_admin';

  const [activeTab, setActiveTab] = useState<PortalTab>(isAdmin ? 'staff' : 'patients');
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  // Supabase connection health ping state
  const [cloudHealth, setCloudHealth] = useState<{ online: boolean; latencyMs?: number }>({
    online: true,
  });

  // Core data state
  const [users, setUsers] = useState<UserAccount[]>([]);
  const [patients, setPatients] = useState<PatientRecord[]>([]);
  const [exercises, setExercises] = useState<ManagedExerciseRecord[]>([]);
  const [modpacks, setModpacks] = useState<ModpackRecord[]>([]);
  const [sessions, setSessions] = useState<RehabSessionRecord[]>([]);
  const [staffCalls, setStaffCalls] = useState<StaffCallRecord[]>([]);
  const [auditLogs, setAuditLogs] = useState<AuditLogRecord[]>([]);
  const [bannedTargets, setBannedTargets] = useState<BannedTargetRecord[]>([]);
  const [activeSessions, setActiveSessions] = useState<ActiveSessionRecord[]>([]);
  const [kiosks, setKiosks] = useState<KioskStatusRecord[]>([]);
  const [systemSettings, setSystemSettings] = useState<SystemSettingsRecord>({
    aiFaceScanEnabled: true,
    aiFaceThreshold: 0.72,
    aiPoseDetectionEnabled: false,
    kioskAutoResetSec: 30,
  });
  const [loading, setLoading] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [bannerMessage, setBannerMessage] = useState<{ kind: 'success' | 'error'; text: string } | null>(null);

  // Modals
  const [enrollModalOpen, setEnrollModalOpen] = useState(false);
  const [enrollTargetPatient, setEnrollTargetPatient] = useState<PatientRecord | null>(null);
  const [welcomeFaceModalOpen, setWelcomeFaceModalOpen] = useState(false);
  const [addStaffModalOpen, setAddStaffModalOpen] = useState(false);
  const [resetPassUser, setResetPassUser] = useState<UserAccount | null>(null);
  const [quickCheckinModalOpen, setQuickCheckinModalOpen] = useState(false);
  const [cardGameModalOpen, setCardGameModalOpen] = useState(false);
  const [patientModalState, setPatientModalState] = useState<{
    patient: PatientRecord | null;
    tab: 'profile' | 'history';
  }>({ patient: null, tab: 'profile' });
  const [selectedPtDetailModal, setSelectedPtDetailModal] = useState<UserAccount | null>(null);
  const [assignPatientIdForPt, setAssignPatientIdForPt] = useState('');
  const [scheduleModalMode, setScheduleModalMode] = useState<'create' | 'day_detail' | null>(null);
  const [selectedCalendarDateIso, setSelectedCalendarDateIso] = useState<string | null>(null);
  const [weeklyEditorOpen, setWeeklyEditorOpen] = useState(false);
  const [registerKioskModalOpen, setRegisterKioskModalOpen] = useState(false);

  useEffect(() => {
    if (!autoOpenEnroll) return;
    setActiveTab('patients');
    setEnrollTargetPatient(null);
    setEnrollModalOpen(true);
    onAutoOpenEnrollConsumed?.();
  }, [autoOpenEnroll, onAutoOpenEnrollConsumed]);

  // Filters
  const [staffSearch, setStaffSearch] = useState('');
  const [staffRoleFilter, setStaffRoleFilter] = useState<'all' | UserRole>('all');
  const [staffStatusFilter, setStaffStatusFilter] = useState<'all' | 'active' | 'inactive'>('all');

  const [patientSearch, setPatientSearch] = useState('');
  const [patientFaceFilter, setPatientFaceFilter] = useState<'all' | 'enrolled' | 'pending'>('all');

  const [ptSearch, setPtSearch] = useState('');
  const [quickCheckinSearch, setQuickCheckinSearch] = useState('');

  // Tab 5 Schedule state
  const [scheduleSubTab, setScheduleSubTab] = useState<'individual' | 'all'>('individual');
  const [selectedSchedulePatientId, setSelectedSchedulePatientId] = useState<string>('');
  const [patientCourse, setPatientCourse] = useState<PatientCourseRecord | null>(null);
  const [courseTitle, setCourseTitle] = useState('แผนฟื้นฟูรายสัปดาห์เฉพาะบุคคล');
  const [courseModpackId, setCourseModpackId] = useState('');
  const [weeklySchedule, setWeeklySchedule] = useState<DailyScheduleDay[]>([]);
  const [calendarDate, setCalendarDate] = useState(() => new Date());

  // Tab 6 Security sub-tab
  const [securitySubTab, setSecuritySubTab] = useState<SecuritySubTab>('calls');
  const [auditSearch, setAuditSearch] = useState('');
  const [banType, setBanType] = useState<'ip' | 'device'>('ip');
  const [banValue, setBanValue] = useState('');
  const [banReason, setBanReason] = useState('');
  const [newKioskCode, setNewKioskCode] = useState('');
  const [newKioskLocation, setNewKioskLocation] = useState('');

  // Add Staff form
  const [newStaffUsername, setNewStaffUsername] = useState('');
  const [newStaffPassword, setNewStaffPassword] = useState('');
  const [newStaffName, setNewStaffName] = useState('');
  const [newStaffPhone, setNewStaffPhone] = useState('');
  const [newStaffRole, setNewStaffRole] = useState<'pt' | 'super_admin'>('pt');
  const [newStaffSpecialty, setNewStaffSpecialty] = useState('เวชศาสตร์ฟื้นฟูกายภาพบำบัด');
  const [newStaffLicense, setNewStaffLicense] = useState('');

  // Tab 4 Exercise & Modpack creation forms
  const [exCode, setExCode] = useState('');
  const [exNameTh, setExNameTh] = useState('');
  const [exInstructionTh, setExInstructionTh] = useState('');
  const [exCategory, setExCategory] = useState<ExerciseConfig['category']>('upper_limb');
  const [exDefaultSide, setExDefaultSide] = useState<BodySide | 'both'>('right');

  const [mpName, setMpName] = useState('');
  const [mpQuestionCount, setMpQuestionCount] = useState(10);
  const [mpCategory, setMpCategory] = useState('ผลไม้');

  const prevOpenCallCountRef = useRef(0);
  const onLogoutRef = useRef(onLogout);
  onLogoutRef.current = onLogout;

  const isAuthError = (err: unknown): boolean =>
    (err instanceof ApiError && err.status === 401) ||
    (err instanceof Error && err.message === 'Authentication required');

  /** Verify session via /api/auth/me when a 401 occurs and sign out cleanly if invalid. */
  const handleAuthFailureIfExpired = useCallback(
    async (err: unknown): Promise<boolean> => {
      if (!isAuthError(err)) return false;
      const verified = await apiClient.getMe();
      if (!verified) {
        await onLogoutRef.current();
        return true;
      }
      return false;
    },
    [apiClient],
  );

  const flashBanner = useCallback((kind: 'success' | 'error', text: string) => {
    setBannerMessage({ kind, text });
    window.setTimeout(() => {
      setBannerMessage((prev) => (prev?.text === text ? null : prev));
    }, 4500);
  }, []);

  const pingCloudHealth = useCallback(async () => {
    const h = await apiClient.checkHealth();
    setCloudHealth({ online: h.ok && h.supabaseConnected, latencyMs: h.latencyMs });
  }, [apiClient]);

  const refreshAll = useCallback(async () => {
    setLoading(true);
    try {
      let listUsersError: unknown = null;
      const [uRes, pRes, exRes, mpRes, sessRes, callsRes, settingsRes, secRes, healthRes] = await Promise.all([
        apiClient.listUsers().catch((err) => {
          listUsersError = err;
          return null;
        }),
        apiClient.listPatients().catch(() => []),
        apiClient.listExercises().catch(() => []),
        apiClient.listModpacks().catch(() => []),
        apiClient.listRehabSessions().catch(() => []),
        apiClient.listStaffCalls().catch(() => []),
        apiClient.getSystemSettings().catch(() => null),
        isAdmin ? apiClient.listSecurityAudit().catch(() => null) : Promise.resolve(null),
        apiClient.checkHealth().catch(() => ({ ok: false, supabaseConnected: false, latencyMs: 0 })),
      ]);

      if (listUsersError) {
        if (await handleAuthFailureIfExpired(listUsersError)) return;
        flashBanner(
          'error',
          listUsersError instanceof Error ? listUsersError.message : 'ไม่สามารถโหลดข้อมูลบุคลากรได้',
        );
      }

      if (uRes && uRes.length > 0) {
        setUsers(uRes);
      } else if (uRes) {
        setUsers([currentUser]);
      } else {
        setUsers((prev) => (prev.length > 0 ? prev : [currentUser]));
      }
      setPatients(pRes);
      setExercises(exRes);
      setModpacks(mpRes);
      setSessions(sessRes);
      setStaffCalls(callsRes);
      setCloudHealth({ online: healthRes.ok && healthRes.supabaseConnected, latencyMs: healthRes.latencyMs });

      const nextOpenCount = callsRes.filter((c) => c.status === 'open').length;
      if (nextOpenCount > prevOpenCallCountRef.current) {
        playEmergencyChime();
      }
      prevOpenCallCountRef.current = nextOpenCount;

      if (settingsRes) {
        setSystemSettings(settingsRes.settings);
        setKiosks(settingsRes.kiosks);
      }
      if (secRes) {
        setAuditLogs(secRes.logs);
        setBannedTargets(secRes.bannedTargets);
        setActiveSessions(secRes.activeSessions);
      }

      if (!selectedSchedulePatientId && pRes.length > 0) {
        setSelectedSchedulePatientId(pRes[0].id);
      }
      if (!courseModpackId && mpRes.length > 0) {
        setCourseModpackId(mpRes[0].id);
      }
    } finally {
      setLoading(false);
    }
  }, [
    apiClient,
    isAdmin,
    currentUser,
    handleAuthFailureIfExpired,
    flashBanner,
    selectedSchedulePatientId,
    courseModpackId,
  ]);

  useEffect(() => {
    refreshAll();
    const pollTimer = window.setInterval(() => {
      void pingCloudHealth();
      apiClient
        .listStaffCalls()
        .then((calls) => {
          const nextOpen = calls.filter((c) => c.status === 'open').length;
          if (nextOpen > prevOpenCallCountRef.current) playEmergencyChime();
          prevOpenCallCountRef.current = nextOpen;
          setStaffCalls(calls);
        })
        .catch(() => {});
    }, 10000);

    // Supabase Realtime subscription for emergency Kiosk staff calls
    const sb = apiClient.getSupabaseClient();
    const channel = sb
      ? sb
          .channel('drtechcares-staff-calls')
          .on('postgres_changes', { event: '*', schema: 'public', table: 'staff_calls' }, () => {
            apiClient
              .listStaffCalls()
              .then((calls) => {
                playEmergencyChime();
                setStaffCalls(calls);
              })
              .catch(() => {});
          })
          .subscribe()
      : null;

    return () => {
      window.clearInterval(pollTimer);
      if (sb && channel) {
        void sb.removeChannel(channel);
      }
    };
  }, [refreshAll, pingCloudHealth, apiClient]);

  useEffect(() => {
    if (!selectedSchedulePatientId) return;
    apiClient
      .getPatientCourse(selectedSchedulePatientId)
      .then((course) => {
        setPatientCourse(course);
        const fallbackModpackId = modpacks[0]?.id ?? 'modpack-core-99';
        if (course) {
          setCourseTitle(course.title);
          setCourseModpackId(course.cognitiveModpackId || fallbackModpackId);
          setWeeklySchedule(
            course.weeklySchedule.length === 7
              ? course.weeklySchedule
              : Array.from({ length: 7 }, (_, dayIndex) => ({
                  dayIndex,
                  dayType: dayIndex % 2 === 0 ? 'training' : 'rest',
                  activityType: 'minigame',
                  difficulty: 'medium',
                  label: dayIndex % 2 === 0 ? 'มินิเกมกระตุ้นสมอง' : 'วันพักฟื้น',
                  exercises: [],
                  cognitiveModpackId: fallbackModpackId,
                })),
          );
        }
      })
      .catch(() => {});
  }, [selectedSchedulePatientId, apiClient, modpacks]);

  const ptUsers = useMemo(() => users.filter((u) => u.role === 'pt'), [users]);
  const openStaffCalls = useMemo(() => staffCalls.filter((c) => c.status !== 'resolved'), [staffCalls]);

  const ptNameMap = useMemo(() => {
    const map = new Map<string, string>();
    users.forEach((u) => map.set(u.id, u.fullName));
    return map;
  }, [users]);

  const filteredStaff = useMemo(() => {
    const q = staffSearch.trim().toLowerCase();
    return users.filter((u) => {
      if (staffRoleFilter !== 'all' && u.role !== staffRoleFilter) return false;
      if (staffStatusFilter === 'active' && !u.active) return false;
      if (staffStatusFilter === 'inactive' && u.active) return false;
      if (!q) return true;
      return (
        u.fullName.toLowerCase().includes(q) ||
        u.username.toLowerCase().includes(q) ||
        (u.phone || '').toLowerCase().includes(q)
      );
    });
  }, [users, staffSearch, staffRoleFilter, staffStatusFilter]);

  const filteredPatients = useMemo(() => {
    const q = patientSearch.trim().toLowerCase();
    return patients.filter((p) => {
      const verified = isPatientFaceVerified(p);
      if (patientFaceFilter === 'enrolled' && !verified) return false;
      if (patientFaceFilter === 'pending' && verified) return false;
      if (!q) return true;
      return (
        p.fullName.toLowerCase().includes(q) ||
        p.hn.toLowerCase().includes(q) ||
        (p.phone || '').toLowerCase().includes(q) ||
        (p.diagnosis || '').toLowerCase().includes(q)
      );
    });
  }, [patients, patientSearch, patientFaceFilter]);

  const filteredPts = useMemo(() => {
    const q = ptSearch.trim().toLowerCase();
    return ptUsers.filter((pt) => {
      if (!q) return true;
      return (
        pt.fullName.toLowerCase().includes(q) ||
        pt.username.toLowerCase().includes(q) ||
        (pt.licenseNo || '').toLowerCase().includes(q) ||
        (pt.specialty || '').toLowerCase().includes(q)
      );
    });
  }, [ptUsers, ptSearch]);

  const filteredAuditLogs = useMemo(() => {
    const q = auditSearch.trim().toLowerCase();
    if (!q) return auditLogs;
    return auditLogs.filter(
      (l) =>
        l.eventType.toLowerCase().includes(q) ||
        l.actorName.toLowerCase().includes(q) ||
        l.detail.toLowerCase().includes(q) ||
        l.ipAddress.toLowerCase().includes(q),
    );
  }, [auditLogs, auditSearch]);

  const selectedSchedulePatient = useMemo(
    () => patients.find((p) => p.id === selectedSchedulePatientId) ?? patients[0] ?? null,
    [patients, selectedSchedulePatientId],
  );

  const handleCreateStaff = async (e: React.FormEvent) => {
    e.preventDefault();
    if (newStaffRole === 'pt' && !newStaffLicense.trim()) {
      flashBanner('error', 'กรุณาระบุเลขที่ใบประกอบวิชาชีพสำหรับนักกายภาพบำบัด');
      return;
    }
    setSubmitting(true);
    try {
      await apiClient.createUser({
        username: newStaffUsername.trim(),
        password: newStaffPassword,
        role: newStaffRole,
        fullName: newStaffName.trim(),
        phone: newStaffPhone.trim(),
        specialty: newStaffRole === 'pt' ? newStaffSpecialty.trim() : 'บริหารโรงพยาบาล',
        licenseNo: newStaffRole === 'pt' ? newStaffLicense.trim() : undefined,
      });
      setAddStaffModalOpen(false);
      setNewStaffUsername('');
      setNewStaffPassword('');
      setNewStaffName('');
      setNewStaffPhone('');
      setNewStaffLicense('');
      flashBanner('success', 'เพิ่มบัญชีบุคลากรและบันทึก Audit Log เรียบร้อยแล้ว');
      await refreshAll();
    } catch (err) {
      if (await handleAuthFailureIfExpired(err)) return;
      flashBanner('error', err instanceof Error ? err.message : 'เพิ่มบุคลากรไม่สำเร็จ');
    } finally {
      setSubmitting(false);
    }
  };

  const handleSaveWeeklyCourse = async () => {
    if (!selectedSchedulePatient) return;
    setSubmitting(true);
    try {
      const saved = await apiClient.upsertPatientCourse({
        patientId: selectedSchedulePatient.id,
        title: courseTitle.trim() || 'แผนฟื้นฟูรายสัปดาห์',
        cognitiveModpackId: courseModpackId || modpacks[0]?.id || 'modpack-core-99',
        weeklySchedule,
      });
      setPatientCourse(saved);
      setWeeklyEditorOpen(false);
      flashBanner('success', `บันทึกตารางฝึก 7 วันของ ${selectedSchedulePatient.fullName} เรียบร้อย`);
      await refreshAll();
    } catch (err) {
      flashBanner('error', err instanceof Error ? err.message : 'บันทึกตารางฝึกไม่สำเร็จ');
    } finally {
      setSubmitting(false);
    }
  };

  const handleClearSchedule = async () => {
    if (!selectedSchedulePatient) return;
    if (!window.confirm(`ยืนยันล้างตารางนัดทั้งหมดของ ${selectedSchedulePatient.fullName}?`)) return;
    setSubmitting(true);
    try {
      const emptyRest = Array.from({ length: 7 }, (_, dayIndex) => ({
        dayIndex,
        dayType: 'rest' as const,
        activityType: 'minigame' as const,
        difficulty: 'easy' as const,
        label: 'พักฟื้น',
        exercises: [],
        cognitiveModpackId: modpacks[0]?.id || 'modpack-core-99',
      }));
      await apiClient.upsertPatientCourse({
        patientId: selectedSchedulePatient.id,
        title: 'ล้างตารางนัดชั่วคราว',
        cognitiveModpackId: modpacks[0]?.id || 'modpack-core-99',
        weeklySchedule: emptyRest,
      });
      await apiClient.updatePatient(selectedSchedulePatient.id, {
        nextAppointmentDate: '',
        rehabSchedules: [],
      });
      setWeeklySchedule(emptyRest);
      flashBanner('success', `ล้างตารางนัดทั้งหมดของ ${selectedSchedulePatient.fullName} แล้ว`);
      await refreshAll();
    } catch (err) {
      flashBanner('error', err instanceof Error ? err.message : 'ล้างตารางนัดไม่สำเร็จ');
    } finally {
      setSubmitting(false);
    }
  };

  const calendarDays = useMemo(() => {
    const year = calendarDate.getFullYear();
    const month = calendarDate.getMonth();
    const firstDay = new Date(year, month, 1).getDay();
    const daysInMonth = new Date(year, month + 1, 0).getDate();
    const cells: (number | null)[] = [];
    for (let i = 0; i < firstDay; i += 1) cells.push(null);
    for (let d = 1; d <= daysInMonth; d += 1) cells.push(d);
    return cells;
  }, [calendarDate]);

  const scheduleStats = useMemo(() => {
    const customList: RehabScheduleItem[] = selectedSchedulePatient?.rehabSchedules ?? [];
    if (customList.length > 0) {
      return {
        total: customList.length,
        active: customList.filter((s) => s.status === 'ACTIVE').length,
        paused: customList.filter((s) => s.status === 'REST').length,
        completed:
          customList.filter((s) => s.status === 'COMPLETED').length +
          sessions.filter((s) => s.patientId === selectedSchedulePatient?.id).length,
      };
    }
    const activeDays = weeklySchedule.filter((d) => d.dayType === 'training').length;
    const restDays = weeklySchedule.filter((d) => d.dayType === 'rest').length;
    const completed = selectedSchedulePatient
      ? sessions.filter((s) => s.patientId === selectedSchedulePatient.id).length
      : 0;
    return {
      total: activeDays,
      active: activeDays,
      paused: restDays,
      completed,
    };
  }, [weeklySchedule, selectedSchedulePatient, sessions]);

  const currentHeader = locale.headers[activeTab];
  const scheduledBadgeCount = patients.filter(
    (p) => Boolean(p.nextAppointmentDate) || Boolean(p.rehabSchedules?.length),
  ).length;

  return (
    <div className="min-h-screen flex bg-[#f4f7fb] text-slate-900">
      {/* Dark Navy Left Sidebar */}
      <aside
        className={`${
          sidebarCollapsed ? 'w-20 shrink-0' : 'w-64 shrink-0'
        } bg-[#0b1f33] text-slate-100 flex flex-col justify-between border-r border-slate-800/80 transition-all duration-200 select-none`}
      >
        <div>
          <div className="flex h-[72px] items-center justify-between gap-2 px-4 border-b border-white/10">
            <div className="flex items-center gap-3 min-w-0">
              <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-gradient-to-br from-emerald-400 to-cyan-500 text-slate-950 shadow-md">
                <HeartHandshake className="h-5 w-5" />
              </div>
              {!sidebarCollapsed && (
                <div className="min-w-0">
                  <div className="text-sm font-black tracking-tight text-white truncate">Dr.Tech.Care</div>
                  <div className="mt-1 inline-flex items-center gap-1.5 rounded-full bg-emerald-500/20 border border-emerald-400/30 px-2 py-0.5 text-[10px] font-bold text-emerald-300 whitespace-nowrap">
                    <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 shrink-0" />
                    <span className="truncate">{isAdmin ? locale.sidebar.adminRole : locale.sidebar.ptRole}</span>
                  </div>
                </div>
              )}
            </div>
            <button
              type="button"
              onClick={() => setSidebarCollapsed((c) => !c)}
              className="rounded-lg p-1.5 text-slate-400 hover:bg-white/10 hover:text-white cursor-pointer shrink-0"
              title={locale.sidebar.collapseTitle}
            >
              <Monitor className="h-4 w-4" />
            </button>
          </div>

          <nav className="p-3 space-y-1.5 text-xs font-semibold">
            {(
              [
                { id: 'staff', label: locale.nav.staff, icon: Users },
                { id: 'patients', label: locale.nav.patients, icon: Activity },
                { id: 'pts', label: locale.nav.pts, icon: UserCheck },
                { id: 'exercises', label: locale.nav.exercises, icon: Gamepad2 },
                { id: 'schedules', label: locale.nav.schedules, icon: Calendar, badge: scheduledBadgeCount || undefined },
                {
                  id: 'security',
                  label: locale.nav.security,
                  icon: Shield,
                  badge: openStaffCalls.length || undefined,
                },
              ] as Array<{
                id: PortalTab;
                label: string;
                icon: React.ComponentType<{ className?: string }>;
                badge?: number;
              }>
            ).map((item) => {
              const Icon = item.icon;
              const isActive = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => setActiveTab(item.id)}
                  className={`w-full h-11 flex items-center justify-between gap-3 rounded-xl px-3.5 text-left whitespace-nowrap overflow-hidden text-ellipsis transition cursor-pointer ${
                    isActive
                      ? 'bg-gradient-to-r from-emerald-500 to-sky-500 text-white font-bold shadow-md shadow-emerald-950/30'
                      : 'text-slate-300 hover:bg-white/5 hover:text-white'
                  }`}
                >
                  <span className="flex items-center gap-3 min-w-0 overflow-hidden">
                    <Icon className="h-4 w-4 shrink-0" />
                    {!sidebarCollapsed && <span className="truncate">{item.label}</span>}
                  </span>
                  {!sidebarCollapsed && item.badge ? (
                    <span className="flex h-5 min-w-[20px] shrink-0 items-center justify-center rounded-full bg-amber-400 px-1.5 text-[10px] font-black text-slate-950">
                      {item.badge}
                    </span>
                  ) : null}
                </button>
              );
            })}
          </nav>
        </div>

        <div className="p-3 space-y-2 border-t border-white/10 text-xs">
          <button
            type="button"
            onClick={() => setCardGameModalOpen(true)}
            className="w-full h-10 flex items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-emerald-500 to-blue-600 px-3 font-extrabold text-white shadow hover:opacity-95 whitespace-nowrap overflow-hidden text-ellipsis cursor-pointer"
          >
            <Gamepad2 className="h-4 w-4 shrink-0" />
            {!sidebarCollapsed && <span className="truncate">{locale.actions.cardGame}</span>}
          </button>

          {!sidebarCollapsed && (
            <div className="flex h-9 items-center justify-between rounded-xl border border-white/10 bg-slate-900/60 px-3 text-[11px] whitespace-nowrap">
              <span className="text-slate-400 truncate">{locale.sidebar.dbCloud}</span>
              <span
                className={`inline-flex items-center gap-1 font-bold shrink-0 ${
                  cloudHealth.online ? 'text-emerald-400' : 'text-rose-400'
                }`}
              >
                <span className={`h-2 w-2 rounded-full ${cloudHealth.online ? 'bg-emerald-400' : 'bg-rose-400'}`} />
                {cloudHealth.online ? locale.sidebar.dbNormal : locale.sidebar.dbOffline}
              </span>
            </div>
          )}

          {onReturnToKiosk && (
            <button
              type="button"
              onClick={onReturnToKiosk}
              className="w-full h-10 flex items-center justify-center gap-2 rounded-xl border border-white/10 bg-white/5 px-3 font-bold text-slate-200 hover:bg-white/10 whitespace-nowrap overflow-hidden text-ellipsis cursor-pointer"
            >
              <Monitor className="h-4 w-4 shrink-0" />
              {!sidebarCollapsed && <span className="truncate">{locale.actions.returnKiosk}</span>}
            </button>
          )}

          <button
            type="button"
            onClick={() => void onLogout()}
            className="w-full h-10 flex items-center justify-center gap-2 rounded-xl border border-rose-500/20 bg-rose-500/10 px-3 font-bold text-rose-300 hover:bg-rose-500/20 whitespace-nowrap overflow-hidden text-ellipsis cursor-pointer"
          >
            <LogOut className="h-4 w-4 shrink-0" />
            {!sidebarCollapsed && <span className="truncate">{locale.actions.logout}</span>}
          </button>
        </div>
      </aside>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Top White Header */}
        <header className="h-[72px] shrink-0 bg-white border-b border-slate-200 px-5 flex items-center justify-between gap-3">
          <div className="min-h-[56px] flex flex-col justify-center min-w-0">
            <h1 className="text-base sm:text-lg font-black text-slate-900 leading-tight truncate">
              {currentHeader.title}
            </h1>
            <p className="text-[10px] font-bold tracking-wider text-slate-400 uppercase mt-0.5 truncate">
              {currentHeader.subtitle}
            </p>
          </div>

          <div className="flex items-center gap-2 text-xs shrink-0 overflow-x-auto">
            {/* Language Toggle */}
            <div className="flex items-center rounded-full border border-slate-200 bg-slate-50 p-0.5 font-bold shrink-0">
              {(['cn', 'th', 'en'] as const).map((code) => (
                <button
                  key={code}
                  type="button"
                  onClick={() => setLang(code)}
                  className={`w-[68px] h-8 inline-flex items-center justify-center whitespace-nowrap rounded-full text-[11px] transition cursor-pointer ${
                    lang === code ? 'bg-white text-emerald-700 shadow-sm border border-emerald-200' : 'text-slate-500'
                  }`}
                >
                  {code === 'cn' ? 'CN 中文' : code === 'th' ? 'TH ไทย' : 'GB EN'}
                </button>
              ))}
            </div>

            <button
              type="button"
              onClick={() => void pingCloudHealth()}
              title={
                cloudHealth.latencyMs !== undefined
                  ? `Supabase Latency: ${cloudHealth.latencyMs}ms`
                  : 'Test Supabase Cloud'
              }
              className={`whitespace-nowrap h-9 min-w-[140px] inline-flex items-center justify-center gap-1.5 rounded-full border px-3.5 font-bold cursor-pointer transition shrink-0 ${
                cloudHealth.online
                  ? 'border-emerald-200 bg-emerald-50 text-emerald-700 hover:bg-emerald-100'
                  : 'border-rose-200 bg-rose-50 text-rose-700 hover:bg-rose-100'
              }`}
            >
              {cloudHealth.online ? <Wifi className="h-3.5 w-3.5 shrink-0" /> : <WifiOff className="h-3.5 w-3.5 shrink-0" />}
              <span>{cloudHealth.online ? locale.actions.cloudConnected : locale.actions.cloudOffline}</span>
            </button>

            <button
              type="button"
              onClick={() => setWelcomeFaceModalOpen(true)}
              className="whitespace-nowrap h-9 min-w-[135px] inline-flex items-center justify-center gap-1.5 rounded-full bg-emerald-600 px-3.5 font-bold text-white shadow-sm hover:bg-emerald-500 cursor-pointer shrink-0"
            >
              <ScanFace className="h-3.5 w-3.5 shrink-0" />
              <span>{locale.actions.welcomeScan}</span>
            </button>

            <button
              type="button"
              onClick={() => setQuickCheckinModalOpen(true)}
              className="whitespace-nowrap h-9 min-w-[135px] inline-flex items-center justify-center gap-1.5 rounded-full border border-amber-300 bg-amber-50 px-3.5 font-bold text-amber-800 hover:bg-amber-100 cursor-pointer shrink-0"
            >
              <Zap className="h-3.5 w-3.5 text-amber-500 shrink-0" />
              <span>{locale.actions.quickCheckin}</span>
            </button>

            {onReturnToKiosk && (
              <button
                type="button"
                onClick={onReturnToKiosk}
                className="whitespace-nowrap h-9 min-w-[135px] inline-flex items-center justify-center gap-1.5 rounded-full border border-emerald-200 bg-white px-3.5 font-bold text-emerald-700 hover:bg-emerald-50 cursor-pointer shrink-0"
              >
                <Monitor className="h-3.5 w-3.5 shrink-0" />
                <span>{locale.actions.kioskBtn}</span>
              </button>
            )}
          </div>
        </header>

        {/* Scrollable Workspace Body */}
        <main className="flex-1 p-5 sm:p-6 overflow-y-auto space-y-5">
          {bannerMessage && (
            <div
              className={`rounded-2xl border px-4 py-3 text-xs font-bold flex items-center justify-between shadow-sm ${
                bannerMessage.kind === 'success'
                  ? 'border-emerald-200 bg-emerald-50 text-emerald-800'
                  : 'border-rose-200 bg-rose-50 text-rose-800'
              }`}
            >
              <span>{bannerMessage.text}</span>
              <button type="button" onClick={() => setBannerMessage(null)}>
                <X className="h-4 w-4" />
              </button>
            </div>
          )}

          {/* TAB 1: STAFF MANAGEMENT */}
          {activeTab === 'staff' && (
            <div className="space-y-4">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div className="flex items-start gap-2.5 min-w-0">
                  <Users className="h-5 w-5 text-emerald-600 mt-0.5 shrink-0" />
                  <div className="min-w-0">
                    <h2 className="text-lg font-extrabold text-slate-900">
                      {locale.staff.title}
                    </h2>
                    <p className="text-xs text-slate-500">
                      {locale.staff.subtitle}
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-2 shrink-0">
                  <button
                    type="button"
                    onClick={() => void refreshAll()}
                    className="flex h-10 w-10 items-center justify-center rounded-xl border border-slate-200 bg-white text-slate-600 hover:bg-slate-50 cursor-pointer shrink-0"
                    title={locale.staff.refreshTitle}
                  >
                    <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
                  </button>
                  {isAdmin && (
                    <button
                      type="button"
                      onClick={() => setAddStaffModalOpen(true)}
                      className="whitespace-nowrap h-10 px-4 inline-flex items-center justify-center gap-1.5 shrink-0 rounded-xl bg-emerald-600 text-xs font-extrabold text-white shadow-sm hover:bg-emerald-500 cursor-pointer"
                    >
                      <Plus className="h-4 w-4 shrink-0" />
                      {locale.staff.addBtn.replace(/^\+\s*/, '')}
                    </button>
                  )}
                </div>
              </div>

              <div className="flex flex-wrap items-center gap-2.5">
                <div className="relative flex-1 min-w-[240px]">
                  <Search className="h-4 w-4 text-slate-400 absolute left-3.5 top-1/2 -translate-y-1/2" />
                  <input
                    value={staffSearch}
                    onChange={(e) => setStaffSearch(e.target.value)}
                    placeholder={locale.staff.searchPlaceholder}
                    className="w-full h-10 rounded-xl border border-slate-200 bg-white pl-10 pr-4 text-xs text-slate-800 placeholder:text-slate-400 focus:border-emerald-500 focus:outline-none"
                  />
                </div>
                <select
                  value={staffRoleFilter}
                  onChange={(e) => setStaffRoleFilter(e.target.value as 'all' | UserRole)}
                  className="h-10 rounded-xl border border-slate-200 bg-white px-3.5 text-xs font-semibold text-slate-700"
                >
                  <option value="all">{locale.staff.allRoles}</option>
                  <option value="super_admin">{locale.staff.roleAdmin}</option>
                  <option value="pt">{locale.staff.rolePt}</option>
                  <option value="patient">{locale.staff.rolePatient}</option>
                </select>
                <select
                  value={staffStatusFilter}
                  onChange={(e) => setStaffStatusFilter(e.target.value as 'all' | 'active' | 'inactive')}
                  className="h-10 rounded-xl border border-slate-200 bg-white px-3.5 text-xs font-semibold text-slate-700"
                >
                  <option value="all">{locale.staff.allStatus}</option>
                  <option value="active">{locale.staff.statusActive}</option>
                  <option value="inactive">{locale.staff.statusSuspended}</option>
                </select>
              </div>

              <div className="rounded-2xl border border-slate-200 bg-white shadow-sm overflow-x-auto">
                <table className="table-fixed w-full min-w-[820px] text-left text-xs">
                  <thead className="border-b border-slate-100 bg-slate-50/70 text-[11px] font-bold text-slate-500 uppercase">
                    <tr>
                      <th className="w-[28%] px-5 py-3.5">{locale.staff.colName}</th>
                      <th className="w-[18%] px-4 py-3.5">{locale.staff.colUsername}</th>
                      <th className="w-[22%] px-4 py-3.5">{locale.staff.colRole}</th>
                      <th className="w-[14%] px-4 py-3.5">{locale.staff.colStatus}</th>
                      {isAdmin && <th className="w-[18%] px-5 py-3.5 text-right">{locale.staff.colActions}</th>}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {filteredStaff.map((u) => (
                      <tr key={u.id} className="hover:bg-slate-50/80 transition">
                        <td className="px-5 py-3.5">
                          <div className="flex items-center gap-3 min-w-0">
                            <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-indigo-50 text-xs font-extrabold text-indigo-700">
                              {u.fullName.trim().slice(0, 1) || 'U'}
                            </div>
                            <div className="min-w-0">
                              <div className="font-bold text-slate-900 truncate">{u.fullName}</div>
                              <div className="text-[11px] text-slate-400 truncate">{u.phone || '08X-XXX-XXXX'}</div>
                            </div>
                          </div>
                        </td>
                        <td className="px-4 py-3.5 font-mono font-bold text-sky-600 truncate">@{u.username}</td>
                        <td className="px-4 py-3.5">
                          {u.role === 'super_admin' && (
                            <span className="inline-flex items-center gap-1 whitespace-nowrap rounded-full border border-purple-200 bg-purple-50 px-2.5 py-1 text-[11px] font-bold text-purple-700">
                              <Shield className="h-3 w-3 shrink-0" /> {locale.staff.roleAdmin}
                            </span>
                          )}
                          {u.role === 'pt' && (
                            <span className="inline-flex items-center gap-1 whitespace-nowrap rounded-full border border-emerald-200 bg-emerald-50 px-2.5 py-1 text-[11px] font-bold text-emerald-700">
                              <UserCheck className="h-3 w-3 shrink-0" /> {locale.staff.rolePt}
                            </span>
                          )}
                          {u.role === 'patient' && (
                            <span className="inline-flex items-center gap-1 whitespace-nowrap rounded-full border border-sky-200 bg-sky-50 px-2.5 py-1 text-[11px] font-bold text-sky-700">
                              <Users className="h-3 w-3 shrink-0" /> {locale.staff.rolePatient}
                            </span>
                          )}
                        </td>
                        <td className="px-4 py-3.5">
                          <span
                            className={`inline-flex items-center gap-1 whitespace-nowrap rounded-full border px-2.5 py-1 text-[11px] font-bold ${
                              u.active
                                ? 'border-emerald-200 bg-emerald-50 text-emerald-700'
                                : 'border-rose-200 bg-rose-50 text-rose-700'
                            }`}
                          >
                            <span
                              className={`h-1.5 w-1.5 rounded-full shrink-0 ${u.active ? 'bg-emerald-500' : 'bg-rose-500'}`}
                            />
                            {u.active ? locale.staff.statusBadgeActive : locale.staff.statusBadgeSuspended}
                          </span>
                        </td>
                        {isAdmin && (
                          <td className="px-5 py-3.5 text-right">
                            <div className="inline-flex items-center justify-end gap-1.5">
                              {u.id !== currentUser.id && (
                                <button
                                  type="button"
                                  disabled={submitting}
                                  onClick={async () => {
                                    setSubmitting(true);
                                    try {
                                      await apiClient.updateUser(u.id, { active: !u.active });
                                      flashBanner(
                                        'success',
                                        u.active ? `ระงับสิทธิ์ @${u.username} แล้ว` : `เปิดใช้งาน @${u.username} แล้ว`,
                                      );
                                      await refreshAll();
                                    } finally {
                                      setSubmitting(false);
                                    }
                                  }}
                                  className="whitespace-nowrap rounded-lg border border-slate-200 px-2.5 py-1 text-[11px] font-bold text-slate-600 hover:bg-slate-100 cursor-pointer"
                                >
                                  {u.active ? locale.staff.suspendBtn : locale.staff.activateBtn}
                                </button>
                              )}
                              <button
                                type="button"
                                onClick={() => setResetPassUser(u)}
                                className="whitespace-nowrap rounded-lg border border-sky-200 bg-sky-50 px-2.5 py-1 text-[11px] font-bold text-sky-700 hover:bg-sky-100 cursor-pointer"
                                title={locale.staff.resetPwdTitle}
                              >
                                <KeyRound className="h-3.5 w-3.5" />
                              </button>
                              {(u.role !== 'super_admin' || users.filter((acc) => acc.role === 'super_admin').length > 1) && (
                                <button
                                  type="button"
                                  disabled={submitting}
                                  onClick={async () => {
                                    const deletingSelf = u.id === currentUser.id;
                                    const confirmMsg = deletingSelf
                                      ? `ยืนยันลบบัญชีแอดมินของตนเอง (${u.fullName})? ระบบจะออกจากระบบอัตโนมัติ`
                                      : `ยืนยันลบบัญชี ${u.fullName} (@${u.username})?`;
                                    if (!window.confirm(confirmMsg)) return;
                                    setSubmitting(true);
                                    try {
                                      await apiClient.deleteUser(u.id);
                                      notifyBiometricIndexChanged();
                                      if (deletingSelf) {
                                        onLogout();
                                        return;
                                      }
                                      flashBanner('success', `ลบบัญชี @${u.username} เรียบร้อยแล้ว`);
                                      await refreshAll();
                                    } catch (err) {
                                      flashBanner(
                                        'error',
                                        err instanceof Error ? err.message : 'ไม่สามารถลบบัญชีผู้ใช้ได้',
                                      );
                                    } finally {
                                      setSubmitting(false);
                                    }
                                  }}
                                  className="whitespace-nowrap rounded-lg border border-rose-200 bg-rose-50 px-2 py-1 text-rose-600 hover:bg-rose-100 disabled:opacity-50 cursor-pointer"
                                  title={locale.staff.deleteTitle}
                                >
                                  <Trash2 className="h-3.5 w-3.5" />
                                </button>
                              )}
                            </div>
                          </td>
                        )}
                      </tr>
                    ))}
                    {filteredStaff.length === 0 && (
                      <tr>
                        <td colSpan={isAdmin ? 5 : 4} className="px-6 py-10 text-center text-slate-400">
                          {locale.staff.emptyStaff}
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* TAB 2: PATIENTS DIRECTORY */}
          {activeTab === 'patients' && (
            <div className="space-y-4">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div className="flex items-start gap-2.5 min-w-0">
                  <Activity className="h-5 w-5 text-emerald-600 mt-0.5 shrink-0" />
                  <div className="min-w-0">
                    <h2 className="text-lg font-extrabold text-slate-900">{locale.patients.title}</h2>
                    <p className="text-xs text-slate-500">
                      {locale.patients.subtitle}
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-2 shrink-0">
                  <button
                    type="button"
                    onClick={() => void refreshAll()}
                    className="flex h-10 w-10 items-center justify-center rounded-xl border border-slate-200 bg-white text-slate-600 hover:bg-slate-50 cursor-pointer shrink-0"
                    title={locale.staff.refreshTitle}
                  >
                    <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setEnrollTargetPatient(null);
                      setEnrollModalOpen(true);
                    }}
                    className="whitespace-nowrap h-10 px-4 inline-flex items-center justify-center gap-1.5 shrink-0 rounded-xl bg-emerald-600 text-xs font-extrabold text-white shadow-sm hover:bg-emerald-500 cursor-pointer"
                  >
                    <UserPlus className="h-4 w-4 shrink-0" />
                    {locale.patients.addBtn}
                  </button>
                </div>
              </div>

              <div className="flex flex-wrap items-center gap-2.5">
                <div className="flex flex-1 min-w-[260px] h-10 items-center rounded-xl border border-slate-200 bg-white p-1">
                  <Search className="h-4 w-4 text-slate-400 ml-2.5 mr-2 shrink-0" />
                  <input
                    value={patientSearch}
                    onChange={(e) => setPatientSearch(e.target.value)}
                    placeholder={locale.patients.searchPlaceholder}
                    className="w-full bg-transparent px-1 py-1 text-xs text-slate-800 placeholder:text-slate-400 focus:outline-none"
                  />
                  <button
                    type="button"
                    onClick={() => void refreshAll()}
                    className="whitespace-nowrap h-8 px-3.5 inline-flex items-center justify-center shrink-0 rounded-lg bg-emerald-600 text-xs font-bold text-white hover:bg-emerald-500 cursor-pointer"
                  >
                    {locale.patients.searchBtn}
                  </button>
                </div>
                <select
                  value={patientFaceFilter}
                  onChange={(e) => setPatientFaceFilter(e.target.value as 'all' | 'enrolled' | 'pending')}
                  className="h-10 rounded-xl border border-slate-200 bg-white px-4 text-xs font-semibold text-slate-700"
                >
                  <option value="all">{locale.patients.allFace}</option>
                  <option value="enrolled">{locale.patients.faceVerified}</option>
                  <option value="pending">{locale.patients.facePending}</option>
                </select>
              </div>

              <div className="rounded-2xl border border-slate-200 bg-white shadow-sm overflow-x-auto">
                <table className="table-fixed w-full min-w-[940px] text-left text-xs">
                  <thead className="border-b border-slate-100 bg-slate-50/70 text-[11px] font-bold text-slate-500">
                    <tr>
                      <th className="w-[11%] px-5 py-3.5">{locale.patients.colHn}</th>
                      <th className="w-[22%] px-4 py-3.5">{locale.patients.colPatientName}</th>
                      <th className="w-[11%] px-4 py-3.5">{locale.patients.colAgeGender}</th>
                      <th className="w-[18%] px-4 py-3.5">{locale.patients.colPhonePt}</th>
                      <th className="w-[15%] px-4 py-3.5">{locale.patients.colBiometric}</th>
                      <th className="w-[23%] px-5 py-3.5 text-right">{locale.patients.colActions}</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {filteredPatients.map((p) => {
                      const verified = isPatientFaceVerified(p);
                      const ptLabel = ptNameMap.get(p.delegatedToPtId || p.assignedPtId) || 'กภ. ประจำคลินิก';
                      return (
                        <tr key={p.id} className="hover:bg-slate-50/80 transition">
                          <td className="px-5 py-3.5">
                            <span className="inline-block whitespace-nowrap rounded-lg border border-sky-200 bg-sky-50 px-2.5 py-1 font-mono text-[11px] font-extrabold text-sky-700">
                              {p.hn}
                            </span>
                          </td>
                          <td className="px-4 py-3.5">
                            <div className="font-bold text-slate-900 truncate">{p.fullName}</div>
                            <div className="text-[11px] text-slate-400 truncate">{p.diagnosis}</div>
                          </td>
                          <td className="px-4 py-3.5 font-medium text-slate-700 whitespace-nowrap">
                            {p.age} {locale.patients.ageUnit} / {p.gender || 'ชาย'}
                          </td>
                          <td className="px-4 py-3.5">
                            <div className="font-mono text-slate-700 truncate">{p.phone || '-'}</div>
                            <div className="text-[11px] font-semibold text-emerald-700 truncate">{ptLabel}</div>
                          </td>
                          <td className="px-4 py-3.5">
                            {verified ? (
                              <span className="inline-flex items-center gap-1.5 whitespace-nowrap rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1 text-[11px] font-bold text-emerald-700">
                                <CheckCircle2 className="h-3.5 w-3.5 shrink-0" />
                                {locale.patients.verifiedBadge}
                              </span>
                            ) : (
                              <span className="inline-flex items-center gap-1.5 whitespace-nowrap rounded-full border border-amber-200 bg-amber-50 px-3 py-1 text-[11px] font-bold text-amber-800">
                                <Clock className="h-3.5 w-3.5 shrink-0" />
                                {locale.patients.pendingBadge}
                              </span>
                            )}
                          </td>
                          <td className="px-5 py-3.5 text-right">
                            <div className="inline-flex items-center justify-end gap-1.5">
                              <button
                                type="button"
                                onClick={() => setPatientModalState({ patient: p, tab: 'profile' })}
                                className="whitespace-nowrap inline-flex items-center gap-1 rounded-xl bg-blue-600 px-2.5 py-1.5 text-[11px] font-bold text-white hover:bg-blue-500 cursor-pointer"
                              >
                                <FileText className="h-3.5 w-3.5 shrink-0" />
                                {locale.patients.emrBtn}
                              </button>
                              <button
                                type="button"
                                onClick={() => {
                                  setEnrollTargetPatient(p);
                                  setEnrollModalOpen(true);
                                }}
                                className="whitespace-nowrap rounded-xl border border-emerald-200 bg-emerald-50 px-2.5 py-1.5 text-[11px] font-bold text-emerald-700 hover:bg-emerald-100 cursor-pointer"
                              >
                                {locale.patients.scanFaceBtn}
                              </button>
                              <button
                                type="button"
                                onClick={() => setPatientModalState({ patient: p, tab: 'history' })}
                                className="whitespace-nowrap rounded-xl border border-slate-200 bg-white px-2.5 py-1.5 text-[11px] font-bold text-slate-700 hover:bg-slate-50 cursor-pointer"
                              >
                                {locale.patients.historyBtn}
                              </button>
                              {isAdmin && (
                                <button
                                  type="button"
                                  onClick={async () => {
                                    if (!window.confirm(`ยืนยันลบข้อมูลคนไข้ ${p.fullName} (รวม Face ID)?`)) return;
                                    await apiClient.deletePatientCascading(p.id);
                                    notifyBiometricIndexChanged();
                                    flashBanner('success', `ลบข้อมูลคนไข้ ${p.fullName} แล้ว`);
                                    await refreshAll();
                                  }}
                                  className="whitespace-nowrap rounded-xl border border-rose-200 bg-rose-50 p-1.5 text-rose-600 hover:bg-rose-100 cursor-pointer"
                                  title={locale.patients.deleteTitle}
                                >
                                  <Trash2 className="h-3.5 w-3.5" />
                                </button>
                              )}
                            </div>
                          </td>
                        </tr>
                      );
                    })}
                    {filteredPatients.length === 0 && (
                      <tr>
                        <td colSpan={6} className="px-6 py-10 text-center text-slate-400">
                          {locale.patients.emptyPatients}
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* TAB 3: PT DIRECTORY */}
          {activeTab === 'pts' && (
            <div className="space-y-4">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div className="flex items-start gap-2.5 min-w-0">
                  <UserCheck className="h-5 w-5 text-emerald-600 mt-0.5 shrink-0" />
                  <div className="min-w-0">
                    <h2 className="text-lg font-extrabold text-slate-900">{locale.pts.title}</h2>
                    <p className="text-xs text-slate-500">
                      {locale.pts.subtitle}
                    </p>
                  </div>
                </div>
                <button
                  type="button"
                  onClick={() => void refreshAll()}
                  className="flex h-10 w-10 items-center justify-center rounded-xl border border-slate-200 bg-white text-slate-600 hover:bg-slate-50 cursor-pointer shrink-0"
                  title={locale.staff.refreshTitle}
                >
                  <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
                </button>
              </div>

              <div className="relative max-w-md">
                <Search className="h-4 w-4 text-slate-400 absolute left-3.5 top-1/2 -translate-y-1/2" />
                <input
                  value={ptSearch}
                  onChange={(e) => setPtSearch(e.target.value)}
                  placeholder={locale.pts.searchPlaceholder}
                  className="w-full h-10 rounded-xl border border-slate-200 bg-white pl-10 pr-4 text-xs text-slate-800 placeholder:text-slate-400 focus:border-emerald-500 focus:outline-none"
                />
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {filteredPts.map((pt) => {
                  const ptPatients = patients.filter(
                    (p) => p.assignedPtId === pt.id || p.delegatedToPtId === pt.id,
                  );
                  return (
                    <div
                      key={pt.id}
                      className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm flex flex-col justify-between space-y-4"
                    >
                      <div className="space-y-3">
                        <div className="flex items-start justify-between gap-2">
                          <div className="flex items-center gap-3 min-w-0">
                            <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-2xl bg-gradient-to-br from-emerald-500 to-teal-600 text-base font-black text-white shadow-sm">
                              {pt.fullName.trim().slice(0, 1) || 'P'}
                            </div>
                            <div className="min-w-0">
                              <h3 className="text-sm font-extrabold text-slate-900 truncate">{pt.fullName}</h3>
                              <p className="font-mono text-[11px] font-bold text-sky-600 truncate">@{pt.username}</p>
                            </div>
                          </div>
                          <div className="flex items-center gap-1.5 shrink-0">
                            <span className="whitespace-nowrap rounded-full bg-emerald-50 border border-emerald-200 px-2.5 py-0.5 text-[11px] font-bold text-emerald-700">
                              PT
                            </span>
                            {isAdmin && (
                              <button
                                type="button"
                                onClick={async () => {
                                  if (!window.confirm(`ยืนยันลบบัญชี PT ${pt.fullName}?`)) return;
                                  await apiClient.deleteUser(pt.id);
                                  notifyBiometricIndexChanged();
                                  flashBanner('success', `ลบบัญชี PT ${pt.fullName} แล้ว`);
                                  await refreshAll();
                                }}
                                className="rounded-lg p-1 text-slate-400 hover:bg-rose-50 hover:text-rose-600 cursor-pointer"
                                title={locale.staff.deleteTitle}
                              >
                                <Trash2 className="h-4 w-4" />
                              </button>
                            )}
                          </div>
                        </div>

                        <div className="flex items-center justify-between gap-2 rounded-xl border border-amber-200/80 bg-amber-50/70 px-3.5 py-2 text-xs">
                          <span className="font-semibold text-amber-900 whitespace-nowrap">{locale.pts.licenseLabel}</span>
                          <span className="whitespace-nowrap rounded-lg bg-white border border-amber-200 px-2.5 py-0.5 font-mono text-[11px] font-bold text-slate-800">
                            {pt.licenseNo || 'PT-44012'}
                          </span>
                        </div>

                        <div className="space-y-1.5 text-xs text-slate-600">
                          <div className="flex items-center gap-2 min-w-0">
                            <FileText className="h-3.5 w-3.5 text-slate-400 shrink-0" />
                            <span className="whitespace-nowrap">{locale.pts.specialtyLabel}</span>
                            <span className="font-semibold text-slate-900 truncate">{pt.specialty || 'กายภาพบำบัดทั่วไป'}</span>
                          </div>
                          <div className="flex items-center justify-between gap-2">
                            <span className="flex items-center gap-2 whitespace-nowrap">
                              <Users className="h-3.5 w-3.5 text-slate-400 shrink-0" />
                              {locale.pts.patientsCountLabel}
                            </span>
                            <span className="font-extrabold text-sky-600 whitespace-nowrap">
                              {ptPatients.length} {locale.pts.patientsUnit}
                            </span>
                          </div>
                        </div>
                      </div>

                      <button
                        type="button"
                        onClick={() => {
                          setSelectedPtDetailModal(pt);
                          const unassigned = patients.find((p) => p.assignedPtId !== pt.id) ?? patients[0];
                          setAssignPatientIdForPt(unassigned?.id ?? '');
                        }}
                        className="flex w-full items-center justify-between border-t border-slate-100 pt-3 text-xs font-bold text-slate-700 hover:text-emerald-700 cursor-pointer whitespace-nowrap"
                      >
                        <span>{locale.pts.viewDetailBtn}</span>
                        <ChevronRight className="h-4 w-4 shrink-0" />
                      </button>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* TAB 4: EXERCISES & MODPACKS */}
          {activeTab === 'exercises' && (
            <div className="space-y-6">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div className="flex items-start gap-2.5 min-w-0">
                  <Gamepad2 className="h-5 w-5 text-emerald-600 mt-0.5 shrink-0" />
                  <div className="min-w-0">
                    <h2 className="text-lg font-extrabold text-slate-900">
                      {locale.exercises.title}
                    </h2>
                    <p className="text-xs text-slate-500">
                      {locale.exercises.subtitle}
                    </p>
                  </div>
                </div>
                <button
                  type="button"
                  onClick={() => setCardGameModalOpen(true)}
                  className="whitespace-nowrap h-10 px-4 inline-flex items-center justify-center gap-2 shrink-0 rounded-xl bg-gradient-to-r from-emerald-600 to-blue-600 text-xs font-extrabold text-white shadow hover:opacity-95 cursor-pointer"
                >
                  <Gamepad2 className="h-4 w-4 shrink-0" />
                  {locale.exercises.playCardBtn}
                </button>
              </div>

              {/* AI Pose Safety Switch Banner */}
              <div
                className={`rounded-2xl border p-4 flex flex-wrap items-center justify-between gap-3 ${
                  systemSettings.aiPoseDetectionEnabled
                    ? 'border-emerald-200 bg-emerald-50/70'
                    : 'border-amber-200 bg-amber-50'
                }`}
              >
                <div className="text-xs min-w-0 flex-1">
                  <div className="font-extrabold text-slate-900">
                    {locale.exercises.aiPoseTitle}{' '}
                    <span className={systemSettings.aiPoseDetectionEnabled ? 'text-emerald-700' : 'text-amber-700'}>
                      {systemSettings.aiPoseDetectionEnabled
                        ? locale.exercises.aiPoseEnabled
                        : locale.exercises.aiPoseLocked}
                    </span>
                  </div>
                  <p className="text-slate-600 mt-0.5">
                    {locale.exercises.aiPoseDesc}
                  </p>
                </div>
                <button
                  type="button"
                  disabled={!isAdmin || submitting}
                  title={!isAdmin ? 'ADMIN ONLY' : undefined}
                  onClick={async () => {
                    setSubmitting(true);
                    try {
                      const updated = await apiClient.updateSystemSettings({
                        aiPoseDetectionEnabled: !systemSettings.aiPoseDetectionEnabled,
                      });
                      setSystemSettings(updated);
                      flashBanner(
                        'success',
                        updated.aiPoseDetectionEnabled
                          ? locale.exercises.aiPoseEnabled
                          : locale.exercises.aiPoseLocked,
                      );
                    } finally {
                      setSubmitting(false);
                    }
                  }}
                  className={`whitespace-nowrap h-10 px-4 inline-flex items-center justify-center shrink-0 rounded-xl text-xs font-bold text-white shadow-sm disabled:opacity-45 disabled:cursor-not-allowed cursor-pointer ${
                    systemSettings.aiPoseDetectionEnabled
                      ? 'bg-amber-600 hover:bg-amber-500'
                      : 'bg-emerald-600 hover:bg-emerald-500'
                  }`}
                >
                  {systemSettings.aiPoseDetectionEnabled
                    ? locale.exercises.aiPoseToggleLock
                    : locale.exercises.aiPoseToggleUnlock}
                </button>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Cognitive Modpacks */}
                <div className="rounded-2xl border border-slate-200 bg-white p-5 space-y-4">
                  <h3 className="text-sm font-extrabold text-slate-900">
                    {locale.exercises.modpackHeader} ({modpacks.length})
                  </h3>
                  <div className="space-y-2.5 max-h-72 overflow-y-auto pr-1">
                    {modpacks.map((m) => (
                      <div
                        key={m.id}
                        className="flex items-center justify-between rounded-xl border border-slate-200 bg-slate-50/70 px-3.5 py-2.5 text-xs"
                      >
                        <div className="min-w-0">
                          <div className="font-bold text-slate-900 truncate">{m.name}</div>
                          <div className="text-[11px] text-slate-500 truncate">
                            {m.categories.join(', ')} • {m.questionCount} Q
                          </div>
                        </div>
                        {(isAdmin || m.createdBy === currentUser.id) && (
                          <button
                            type="button"
                            onClick={async () => {
                              try {
                                await apiClient.deleteModpack(m.id);
                                flashBanner('success', `ลบ Modpack "${m.name}" แล้ว`);
                                await refreshAll();
                              } catch (err) {
                                flashBanner('error', err instanceof Error ? err.message : 'ไม่สามารถลบ Modpack กลางได้');
                              }
                            }}
                            className="rounded-lg p-1.5 text-slate-400 hover:bg-rose-50 hover:text-rose-600 cursor-pointer shrink-0"
                            title={locale.staff.deleteTitle}
                          >
                            <Trash2 className="h-4 w-4" />
                          </button>
                        )}
                      </div>
                    ))}
                  </div>

                  <form
                    onSubmit={async (e) => {
                      e.preventDefault();
                      if (!mpName.trim()) return;
                      setSubmitting(true);
                      try {
                        await apiClient.createModpack({
                          name: mpName.trim(),
                          categories: [mpCategory],
                          questionCount: mpQuestionCount,
                        });
                        setMpName('');
                        flashBanner('success', 'บันทึกชุดมินิเกม Modpack ลง Supabase เรียบร้อย');
                        await refreshAll();
                      } catch (err) {
                        flashBanner('error', err instanceof Error ? err.message : 'เพิ่ม Modpack ไม่สำเร็จ');
                      } finally {
                        setSubmitting(false);
                      }
                    }}
                    className="border-t border-slate-100 pt-3 grid grid-cols-1 sm:grid-cols-3 gap-2 text-xs"
                  >
                    <input
                      required
                      value={mpName}
                      onChange={(e) => setMpName(e.target.value)}
                      placeholder={
                        lang === 'cn'
                          ? '输入新游戏题目包名称...'
                          : lang === 'en'
                            ? 'New modpack name...'
                            : 'ชื่อ Modpack ใหม่...'
                      }
                      className="rounded-xl border border-slate-200 px-3 py-2 sm:col-span-2"
                    />
                    <select
                      value={mpCategory}
                      onChange={(e) => setMpCategory(e.target.value)}
                      className="rounded-xl border border-slate-200 px-2.5 py-2"
                    >
                      <option value="ผลไม้">ผลไม้</option>
                      <option value="ผัก">ผัก</option>
                      <option value="สัตว์">สัตว์</option>
                      <option value="ความปลอดภัยในบ้าน">ความปลอดภัยในบ้าน</option>
                    </select>
                    <input
                      type="number"
                      min={5}
                      max={99}
                      value={mpQuestionCount}
                      onChange={(e) => setMpQuestionCount(Number(e.target.value))}
                      className="rounded-xl border border-slate-200 px-3 py-2"
                    />
                    <button
                      type="submit"
                      disabled={submitting}
                      className="sm:col-span-2 whitespace-nowrap h-10 px-4 inline-flex items-center justify-center rounded-xl bg-emerald-600 font-bold text-white hover:bg-emerald-500 disabled:opacity-50 cursor-pointer"
                    >
                      {locale.exercises.addModpackBtn}
                    </button>
                  </form>
                </div>

                {/* Rehab Exercises Library */}
                <div className="rounded-2xl border border-slate-200 bg-white p-5 space-y-4">
                  <h3 className="text-sm font-extrabold text-slate-900">
                    {locale.exercises.exerciseHeader} ({exercises.length})
                  </h3>
                  <div className="space-y-2.5 max-h-72 overflow-y-auto pr-1">
                    {exercises.map((ex) => (
                      <div
                        key={ex.id}
                        className="flex items-center justify-between rounded-xl border border-slate-200 bg-slate-50/70 px-3.5 py-2.5 text-xs"
                      >
                        <div className="min-w-0">
                          <div className="font-bold text-slate-900 truncate">
                            {ex.name[lang] || ex.name.th}{' '}
                            <span className="ml-1 rounded bg-sky-100 px-1.5 py-0.5 font-mono text-[10px] text-sky-800">
                              {ex.code}
                            </span>
                            <span className="ml-1 rounded bg-slate-200 px-1.5 py-0.5 text-[10px] font-bold text-slate-700 uppercase">
                              {ex.detection.side}
                            </span>
                          </div>
                          <div className="text-[11px] text-slate-500 truncate">{ex.instruction[lang] || ex.instruction.th}</div>
                        </div>
                        {(isAdmin || ex.createdBy === currentUser.id) && (
                          <button
                            type="button"
                            onClick={async () => {
                              try {
                                await apiClient.deleteExercise(ex.id);
                                flashBanner('success', `ลบท่ากายภาพ "${ex.name.th}" แล้ว`);
                                await refreshAll();
                              } catch (err) {
                                flashBanner('error', err instanceof Error ? err.message : 'ไม่สามารถลบท่ากายภาพกลางได้');
                              }
                            }}
                            className="rounded-lg p-1.5 text-slate-400 hover:bg-rose-50 hover:text-rose-600 cursor-pointer shrink-0"
                            title={locale.staff.deleteTitle}
                          >
                            <Trash2 className="h-4 w-4" />
                          </button>
                        )}
                      </div>
                    ))}
                  </div>

                  <form
                    onSubmit={async (e) => {
                      e.preventDefault();
                      if (!exNameTh.trim()) return;
                      setSubmitting(true);
                      try {
                        const cleanCode = (exCode.trim() || `CUSTOM_${Date.now()}`).toUpperCase();
                        await apiClient.createExercise({
                          id: cleanCode.toLowerCase(),
                          code: cleanCode,
                          category: exCategory,
                          name: {
                            th: exNameTh.trim(),
                            en: exNameTh.trim(),
                            cn: exNameTh.trim(),
                          },
                          instruction: {
                            th: exInstructionTh.trim() || 'ทำตามคำแนะนำนักกายภาพบำบัด',
                            en: exInstructionTh.trim() || 'Follow PT posture guide',
                            cn: '请按照康复师指导完成动作',
                          },
                          detection: {
                            mode: 'angle_rep_hold',
                            jointGroup: exCategory === 'lower_limb' ? 'knee' : 'shoulder',
                            side: exDefaultSide,
                            targetMinDeg: 30,
                            targetMaxDeg: 135,
                            startRestMaxDeg: 25,
                            holdSec: 2,
                            defaultSets: 2,
                            defaultReps: 8,
                          },
                        });
                        setExCode('');
                        setExNameTh('');
                        setExInstructionTh('');
                        flashBanner('success', 'เพิ่มท่ากายภาพบำบัดลง Supabase เรียบร้อย');
                        await refreshAll();
                      } catch (err) {
                        flashBanner('error', err instanceof Error ? err.message : 'เพิ่มท่ากายภาพไม่สำเร็จ');
                      } finally {
                        setSubmitting(false);
                      }
                    }}
                    className="border-t border-slate-100 pt-3 grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs"
                  >
                    <input
                      value={exCode}
                      onChange={(e) => setExCode(e.target.value)}
                      placeholder={
                        lang === 'cn'
                          ? '动作代码 (如 SHOULDER_LIFT)'
                          : lang === 'en'
                            ? 'Code (e.g. SHOULDER_LIFT)'
                            : 'รหัสท่า (เช่น SHOULDER_LIFT)'
                      }
                      className="rounded-xl border border-slate-200 px-3 py-2"
                    />
                    <select
                      value={exCategory}
                      onChange={(e) => setExCategory(e.target.value as ExerciseConfig['category'])}
                      className="rounded-xl border border-slate-200 px-2.5 py-2"
                    >
                      <option value="upper_limb">Upper Limb (แขน/ไหล่)</option>
                      <option value="lower_limb">Lower Limb (ขา/เข่า)</option>
                      <option value="neck_trunk">Neck &amp; Trunk (คอ/ลำตัว)</option>
                      <option value="balance">Balance (การทรงตัว)</option>
                    </select>
                    <input
                      required
                      value={exNameTh}
                      onChange={(e) => setExNameTh(e.target.value)}
                      placeholder={
                        lang === 'cn'
                          ? '康复动作名称...'
                          : lang === 'en'
                            ? 'Exercise name...'
                            : 'ชื่อท่าภาษาไทย...'
                      }
                      className="rounded-xl border border-slate-200 px-3 py-2"
                    />
                    <select
                      value={exDefaultSide}
                      onChange={(e) => setExDefaultSide(e.target.value as BodySide | 'both')}
                      className="rounded-xl border border-slate-200 px-2.5 py-2"
                    >
                      <option value="right">Right</option>
                      <option value="left">Left</option>
                      <option value="both">Both</option>
                    </select>
                    <input
                      value={exInstructionTh}
                      onChange={(e) => setExInstructionTh(e.target.value)}
                      placeholder={
                        lang === 'cn'
                          ? '动作指导说明...'
                          : lang === 'en'
                            ? 'Patient guidance instruction...'
                            : 'คำแนะนำผู้ป่วยขณะทำท่า...'
                      }
                      className="sm:col-span-2 rounded-xl border border-slate-200 px-3 py-2"
                    />
                    <button
                      type="submit"
                      disabled={submitting}
                      className="sm:col-span-2 whitespace-nowrap h-10 px-4 inline-flex items-center justify-center rounded-xl bg-slate-900 font-bold text-white hover:bg-slate-800 disabled:opacity-50 cursor-pointer"
                    >
                      {locale.exercises.addExerciseBtn}
                    </button>
                  </form>
                </div>
              </div>
            </div>
          )}

          {/* TAB 5: SCHEDULES & CALENDAR */}
          {activeTab === 'schedules' && (
            <div className="space-y-4">
              <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm flex flex-wrap items-center justify-between gap-4">
                <div className="flex items-start gap-3.5 min-w-0">
                  <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl bg-emerald-50 border border-emerald-200 text-emerald-700">
                    <Users className="h-6 w-6" />
                  </div>
                  <div className="min-w-0">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="whitespace-nowrap rounded-lg bg-sky-50 border border-sky-200 px-2.5 py-0.5 font-mono text-[11px] font-extrabold text-sky-700">
                        HN: {selectedSchedulePatient?.hn ?? '-'}
                      </span>
                      <span className="inline-flex items-center gap-1 whitespace-nowrap rounded-full bg-emerald-50 border border-emerald-200 px-2.5 py-0.5 text-[11px] font-bold text-emerald-700">
                        <span className="h-1.5 w-1.5 rounded-full bg-emerald-500 shrink-0" />
                        {locale.schedules.activeBadge}
                      </span>
                    </div>
                    <h2 className="mt-1 text-lg font-black text-slate-900 truncate">
                      {selectedSchedulePatient?.fullName ?? locale.schedules.noPatient}
                    </h2>
                    <p className="text-xs text-slate-500 truncate">
                      {locale.schedules.diagLabel} {selectedSchedulePatient?.diagnosis ?? '-'} • {locale.schedules.ptLabel}{' '}
                      <span className="font-semibold text-slate-700">
                        {selectedSchedulePatient
                          ? ptNameMap.get(selectedSchedulePatient.delegatedToPtId || selectedSchedulePatient.assignedPtId) ||
                            'กภ. ประจำคลินิก'
                          : '-'}
                      </span>
                    </p>
                  </div>
                </div>

                <div className="flex flex-col items-stretch sm:items-end gap-2 shrink-0">
                  <div className="flex flex-wrap items-center gap-2">
                    <select
                      value={selectedSchedulePatient?.id ?? ''}
                      onChange={(e) => setSelectedSchedulePatientId(e.target.value)}
                      className="h-9 rounded-xl border border-slate-200 bg-slate-50 px-3 text-xs font-bold text-slate-800"
                    >
                      {patients.map((p) => (
                        <option key={p.id} value={p.id}>
                          {p.fullName} ({p.hn})
                        </option>
                      ))}
                    </select>
                    <button
                      type="button"
                      onClick={() => void handleClearSchedule()}
                      className="whitespace-nowrap h-9 px-3 inline-flex items-center justify-center gap-1 rounded-xl border border-slate-200 bg-white text-xs font-bold text-slate-600 hover:bg-rose-50 hover:text-rose-700 cursor-pointer shrink-0"
                    >
                      <Trash2 className="h-3.5 w-3.5 shrink-0" />
                      {locale.schedules.clearScheduleBtn}
                    </button>
                  </div>
                  <div className="flex flex-wrap items-center gap-2">
                    <button
                      type="button"
                      onClick={() => setWeeklyEditorOpen((v) => !v)}
                      className="whitespace-nowrap h-10 px-3.5 inline-flex items-center justify-center gap-1.5 rounded-xl border border-emerald-200 bg-emerald-50 text-xs font-bold text-emerald-700 hover:bg-emerald-100 cursor-pointer shrink-0"
                    >
                      <SlidersHorizontal className="h-3.5 w-3.5 shrink-0" />
                      {locale.schedules.weeklyPlanBtn}
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        setSelectedCalendarDateIso(new Date().toISOString().slice(0, 10));
                        setScheduleModalMode('create');
                      }}
                      className="whitespace-nowrap h-10 px-4 inline-flex items-center justify-center gap-1.5 rounded-xl bg-gradient-to-r from-emerald-600 to-blue-600 text-xs font-extrabold text-white shadow hover:opacity-95 cursor-pointer shrink-0"
                    >
                      <Plus className="h-4 w-4 shrink-0" />
                      {locale.schedules.createScheduleBtn.replace(/^\+\s*/, '')}
                    </button>
                  </div>
                </div>
              </div>

              <div className="flex items-center justify-between gap-2">
                <div className="inline-flex rounded-xl border border-slate-200 bg-white p-1 text-xs font-bold">
                  <button
                    type="button"
                    onClick={() => setScheduleSubTab('individual')}
                    className={`whitespace-nowrap h-8 px-4 inline-flex items-center justify-center rounded-lg transition cursor-pointer ${
                      scheduleSubTab === 'individual' ? 'bg-emerald-50 text-emerald-700' : 'text-slate-500'
                    }`}
                  >
                    {locale.schedules.subIndividual}
                  </button>
                  <button
                    type="button"
                    onClick={() => setScheduleSubTab('all')}
                    className={`whitespace-nowrap h-8 px-4 inline-flex items-center justify-center rounded-lg transition cursor-pointer ${
                      scheduleSubTab === 'all' ? 'bg-emerald-50 text-emerald-700' : 'text-slate-500'
                    }`}
                  >
                    {locale.schedules.subAll} ({patients.length})
                  </button>
                </div>

                <button
                  type="button"
                  onClick={() => void refreshAll()}
                  className="flex h-9 w-9 items-center justify-center rounded-xl border border-slate-200 bg-white text-slate-600 hover:bg-slate-50 cursor-pointer shrink-0"
                  title={locale.staff.refreshTitle}
                >
                  <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
                </button>
              </div>

              {weeklyEditorOpen && selectedSchedulePatient && (
                <div className="rounded-2xl border-2 border-emerald-200 bg-white p-5 space-y-4 shadow-md">
                  <div className="flex items-center justify-between">
                    <h3 className="text-sm font-extrabold text-slate-900">
                      {locale.schedules.weeklyPlanBtn} — {selectedSchedulePatient.fullName}
                    </h3>
                    <button
                      type="button"
                      onClick={() => setWeeklyEditorOpen(false)}
                      className="text-slate-400 hover:text-slate-700"
                    >
                      <X className="h-4 w-4" />
                    </button>
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
                    <input
                      value={courseTitle}
                      onChange={(e) => setCourseTitle(e.target.value)}
                      placeholder={
                        lang === 'cn' ? '康复计划名称' : lang === 'en' ? 'Rehab plan title' : 'ชื่อแผนฟื้นฟู'
                      }
                      className="rounded-xl border border-slate-200 px-3 py-2 font-semibold"
                    />
                    <select
                      value={courseModpackId}
                      onChange={(e) => setCourseModpackId(e.target.value)}
                      className="rounded-xl border border-slate-200 px-3 py-2 font-semibold"
                    >
                      {modpacks.map((m) => (
                        <option key={m.id} value={m.id}>
                          {m.name} ({m.questionCount} Q)
                        </option>
                      ))}
                    </select>
                  </div>
                  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2.5">
                    {weeklySchedule.map((day, idx) => (
                      <div key={idx} className="rounded-xl border border-slate-200 bg-slate-50/80 p-3 space-y-2 text-xs">
                        <div className="flex items-center justify-between">
                          <span className="font-extrabold text-slate-800">
                            {lang === 'th' ? `วัน${WEEKDAY_NAMES_TH[idx]}` : locale.schedules.calendarHeaders[idx]}
                          </span>
                          <select
                            value={day.dayType}
                            onChange={(e) => {
                              const next = [...weeklySchedule];
                              next[idx] = {
                                ...day,
                                dayType: e.target.value as 'training' | 'rest',
                                label: e.target.value === 'rest' ? 'วันพักฟื้น' : 'ฝึกตามโปรแกรม',
                              };
                              setWeeklySchedule(next);
                            }}
                            className="rounded-lg border border-slate-200 bg-white px-2 py-1 text-[11px] font-bold"
                          >
                            <option value="training">
                              {lang === 'cn' ? '训练日' : lang === 'en' ? 'Training' : 'วันฝึก'}
                            </option>
                            <option value="rest">
                              {lang === 'cn' ? '休息日' : lang === 'en' ? 'Rest' : 'วันพัก'}
                            </option>
                          </select>
                        </div>
                        {day.dayType === 'training' && (
                          <select
                            value={day.activityType}
                            onChange={(e) => {
                              const val = e.target.value as 'minigame' | 'physio_exercise';
                              const next = [...weeklySchedule];
                              next[idx] = { ...day, activityType: val };
                              setWeeklySchedule(next);
                            }}
                            className="w-full rounded-lg border border-slate-200 bg-white px-2 py-1 text-[11px] font-semibold"
                          >
                            <option value="minigame">
                              🎮 {lang === 'cn' ? '认知卡牌小游戏' : lang === 'en' ? 'Cognitive Minigame' : 'มินิเกมการ์ด'}
                            </option>
                            <option
                              value="physio_exercise"
                              disabled={!isAdmin && !systemSettings.aiPoseDetectionEnabled}
                            >
                              🏋️ {lang === 'cn' ? 'AI 康复训练' : lang === 'en' ? 'AI Rehab Exercise' : 'ท่ากายภาพ AI'}{' '}
                              {!isAdmin && !systemSettings.aiPoseDetectionEnabled ? '(Locked)' : ''}
                            </option>
                          </select>
                        )}
                      </div>
                    ))}
                  </div>
                  <div className="flex justify-end gap-2">
                    <button
                      type="button"
                      onClick={() => setWeeklyEditorOpen(false)}
                      className="whitespace-nowrap h-10 px-4 inline-flex items-center justify-center rounded-xl border border-slate-200 text-xs font-bold"
                    >
                      {lang === 'cn' ? '取消' : lang === 'en' ? 'Cancel' : 'ยกเลิก'}
                    </button>
                    <button
                      type="button"
                      disabled={submitting}
                      onClick={() => void handleSaveWeeklyCourse()}
                      className="whitespace-nowrap h-10 px-5 inline-flex items-center justify-center rounded-xl bg-emerald-600 text-xs font-extrabold text-white hover:bg-emerald-500"
                    >
                      {lang === 'cn' ? '保存 7 天计划' : lang === 'en' ? 'Save 7-Day Plan' : 'บันทึกตารางฝึก 7 วัน'}
                    </button>
                  </div>
                </div>
              )}

              {scheduleSubTab === 'individual' ? (
                <>
                  <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
                    <div className="min-h-[84px] rounded-2xl border border-slate-200 bg-white p-4 flex items-center justify-between gap-2">
                      <div className="min-w-0">
                        <div className="text-[11px] font-semibold text-slate-500 truncate">{locale.schedules.statTotal}</div>
                        <div className="mt-1 text-xl font-black text-slate-900 whitespace-nowrap">
                          {scheduleStats.total} {locale.schedules.unitAppt}
                        </div>
                      </div>
                      <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-emerald-50 text-emerald-600">
                        <CalendarDays className="h-5 w-5" />
                      </div>
                    </div>
                    <div className="min-h-[84px] rounded-2xl border border-slate-200 bg-white p-4 flex items-center justify-between gap-2">
                      <div className="min-w-0">
                        <div className="text-[11px] font-semibold text-slate-500 truncate">{locale.schedules.statActive}</div>
                        <div className="mt-1 text-xl font-black text-emerald-600 whitespace-nowrap">
                          {scheduleStats.active} {locale.schedules.unitAppt}
                        </div>
                      </div>
                      <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-emerald-50 text-emerald-600">
                        <Activity className="h-5 w-5" />
                      </div>
                    </div>
                    <div className="min-h-[84px] rounded-2xl border border-slate-200 bg-white p-4 flex items-center justify-between gap-2">
                      <div className="min-w-0">
                        <div className="text-[11px] font-semibold text-slate-500 truncate">{locale.schedules.statRest}</div>
                        <div className="mt-1 text-xl font-black text-amber-600 whitespace-nowrap">
                          {scheduleStats.paused} {locale.schedules.unitAppt}
                        </div>
                      </div>
                      <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-amber-50 text-amber-500">
                        <PauseCircle className="h-5 w-5" />
                      </div>
                    </div>
                    <div className="min-h-[84px] rounded-2xl border border-slate-200 bg-white p-4 flex items-center justify-between gap-2">
                      <div className="min-w-0">
                        <div className="text-[11px] font-semibold text-slate-500 truncate">{locale.schedules.statCompleted}</div>
                        <div className="mt-1 text-xl font-black text-blue-600 whitespace-nowrap">
                          {scheduleStats.completed} {locale.schedules.unitAppt}
                        </div>
                      </div>
                      <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-sky-50 text-sky-600">
                        <CheckCircle2 className="h-5 w-5" />
                      </div>
                    </div>
                  </div>

                  <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm space-y-4">
                    <div className="flex flex-wrap items-center justify-between gap-3">
                      <div className="flex items-center gap-2">
                        <Calendar className="h-4 w-4 text-emerald-600 shrink-0" />
                        <h3 className="text-sm font-black text-slate-900 whitespace-nowrap">
                          {lang === 'cn'
                            ? `${calendarDate.getFullYear()}年 ${CN_MONTHS[calendarDate.getMonth()]}`
                            : lang === 'en'
                              ? `${EN_MONTHS[calendarDate.getMonth()]} ${calendarDate.getFullYear()}`
                              : `${THAI_MONTHS[calendarDate.getMonth()]} ${calendarDate.getFullYear() + 543}`}{' '}
                          {lang === 'th' && (
                            <span className="font-normal text-slate-400">
                              ({EN_MONTHS[calendarDate.getMonth()]} {calendarDate.getFullYear()})
                            </span>
                          )}
                        </h3>
                        <div className="flex items-center gap-1 ml-2">
                          <button
                            type="button"
                            onClick={() =>
                              setCalendarDate(new Date(calendarDate.getFullYear(), calendarDate.getMonth() - 1, 1))
                            }
                            className="rounded-lg border border-slate-200 p-1 hover:bg-slate-50 cursor-pointer"
                          >
                            <ChevronLeft className="h-3.5 w-3.5" />
                          </button>
                          <button
                            type="button"
                            onClick={() =>
                              setCalendarDate(new Date(calendarDate.getFullYear(), calendarDate.getMonth() + 1, 1))
                            }
                            className="rounded-lg border border-slate-200 p-1 hover:bg-slate-50 cursor-pointer"
                          >
                            <ChevronRight className="h-3.5 w-3.5" />
                          </button>
                        </div>
                      </div>

                      <div className="flex flex-wrap items-center gap-3 text-[11px] font-semibold text-slate-600">
                        <span className="inline-flex items-center gap-1 whitespace-nowrap">
                          <span className="h-2 w-2 rounded-full bg-emerald-500 shrink-0" /> {locale.schedules.legendActive}
                        </span>
                        <span className="inline-flex items-center gap-1 whitespace-nowrap">
                          <span className="h-2 w-2 rounded-full bg-amber-400 shrink-0" /> {locale.schedules.legendRest}
                        </span>
                        <span className="inline-flex items-center gap-1 whitespace-nowrap">
                          <span className="h-2 w-2 rounded-full bg-blue-500 shrink-0" /> {locale.schedules.legendCompleted}
                        </span>
                        <span className="inline-flex items-center gap-1 whitespace-nowrap">
                          <span className="h-2 w-2 rounded-full bg-rose-500 shrink-0" /> {locale.schedules.legendMissed}
                        </span>
                      </div>
                    </div>

                    <div className="grid grid-cols-7 gap-2 text-center text-[11px] font-bold text-slate-400">
                      {locale.schedules.calendarHeaders.map((h) => (
                        <div key={h} className="py-1 whitespace-nowrap">
                          {h}
                        </div>
                      ))}
                    </div>

                    <div className="grid grid-cols-7 gap-2">
                      {calendarDays.map((dayNum, idx) => {
                        if (dayNum === null) {
                          return <div key={`empty-${idx}`} className="h-20 rounded-xl bg-slate-50/40" />;
                        }
                        const year = calendarDate.getFullYear();
                        const month = calendarDate.getMonth();
                        const dateIso = `${year}-${String(month + 1).padStart(2, '0')}-${String(dayNum).padStart(2, '0')}`;
                        const weekday = new Date(year, month, dayNum).getDay();
                        const today = new Date();
                        const isToday =
                          dayNum === today.getDate() && month === today.getMonth() && year === today.getFullYear();

                        const customForDay = (selectedSchedulePatient?.rehabSchedules ?? []).filter(
                          (s) => s.scheduledDate === dateIso,
                        );
                        const planDay = weeklySchedule[weekday];
                        const isTraining = planDay?.dayType === 'training' && Boolean(patientCourse);

                        return (
                          <button
                            key={dateIso}
                            type="button"
                            onClick={() => {
                              setSelectedCalendarDateIso(dateIso);
                              setScheduleModalMode('day_detail');
                            }}
                            className={`h-20 rounded-xl border p-2 flex flex-col justify-between text-left text-xs transition cursor-pointer ${
                              isToday
                                ? 'border-emerald-500 bg-emerald-50/30 ring-1 ring-emerald-500/30'
                                : 'border-slate-100 bg-white hover:border-emerald-300'
                            }`}
                          >
                            <span
                              className={`inline-flex h-6 w-6 items-center justify-center rounded-full text-[11px] font-bold ${
                                isToday ? 'bg-emerald-600 text-white' : 'text-slate-700'
                              }`}
                            >
                              {dayNum}
                            </span>
                            <div className="w-full truncate text-[10px]">
                              {customForDay.length > 0 ? (
                                <span
                                  className={`inline-block whitespace-nowrap rounded px-1.5 py-0.5 font-bold ${
                                    customForDay[0].status === 'COMPLETED'
                                      ? 'bg-blue-100 text-blue-800'
                                      : customForDay[0].status === 'REST'
                                        ? 'bg-amber-100 text-amber-800'
                                        : customForDay[0].status === 'MISSED'
                                          ? 'bg-rose-100 text-rose-800'
                                          : 'bg-emerald-100 text-emerald-800'
                                  }`}
                                >
                                  {customForDay[0].status === 'COMPLETED'
                                    ? locale.schedules.statusCompleted
                                    : customForDay[0].status === 'REST'
                                      ? locale.schedules.statusRest
                                      : customForDay[0].status === 'MISSED'
                                        ? locale.schedules.statusMissed
                                        : locale.schedules.statusActive}
                                </span>
                              ) : isTraining ? (
                                <span className="inline-block whitespace-nowrap rounded bg-emerald-100 px-1.5 py-0.5 font-bold text-emerald-800">
                                  {lang === 'cn'
                                    ? planDay.activityType === 'physio_exercise'
                                      ? 'AI 训练'
                                      : '认知游戏'
                                    : lang === 'en'
                                      ? planDay.activityType === 'physio_exercise'
                                        ? 'AI Physio'
                                        : 'Minigame'
                                      : planDay.activityType === 'physio_exercise'
                                        ? 'ฝึก AI'
                                        : 'มินิเกม'}
                                </span>
                              ) : (
                                <span className="text-slate-300">{locale.schedules.restDayLabel}</span>
                              )}
                            </div>
                          </button>
                        );
                      })}
                    </div>
                  </div>
                </>
              ) : (
                <div className="rounded-2xl border border-slate-200 bg-white overflow-x-auto shadow-sm">
                  <table className="table-fixed w-full min-w-[780px] text-left text-xs">
                    <thead className="border-b border-slate-100 bg-slate-50 text-[11px] font-bold text-slate-500">
                      <tr>
                        <th className="w-[14%] px-5 py-3">{locale.schedules.colHn}</th>
                        <th className="w-[26%] px-4 py-3">{locale.schedules.colPatientName}</th>
                        <th className="w-[24%] px-4 py-3">{locale.schedules.colPt}</th>
                        <th className="w-[22%] px-4 py-3">{locale.schedules.colNextAppt}</th>
                        <th className="w-[14%] px-5 py-3 text-right">{locale.schedules.colActions}</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100">
                      {patients.map((p) => (
                        <tr key={p.id}>
                          <td className="px-5 py-3 font-mono font-bold text-sky-700 whitespace-nowrap">{p.hn}</td>
                          <td className="px-4 py-3 font-bold text-slate-900 truncate">{p.fullName}</td>
                          <td className="px-4 py-3 text-slate-600 truncate">
                            {ptNameMap.get(p.delegatedToPtId || p.assignedPtId) || 'กภ. ประจำคลินิก'}
                          </td>
                          <td className="px-4 py-3 font-semibold text-emerald-700 whitespace-nowrap">
                            {p.nextAppointmentDate || '-'} ({p.rehabSchedules?.length ?? 0} {locale.schedules.unitAppt})
                          </td>
                          <td className="px-5 py-3 text-right">
                            <button
                              type="button"
                              onClick={() => {
                                setSelectedSchedulePatientId(p.id);
                                setScheduleSubTab('individual');
                              }}
                              className="whitespace-nowrap rounded-lg bg-emerald-50 px-3 py-1 font-bold text-emerald-700 hover:bg-emerald-100 cursor-pointer"
                            >
                              {locale.schedules.openCalendarBtn}
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}

          {/* TAB 6: SECURITY & AUDIT */}
          {activeTab === 'security' && (
            <div className="space-y-4">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div className="flex items-start gap-2.5 min-w-0">
                  <ShieldAlert className="h-5 w-5 text-rose-600 mt-0.5 shrink-0" />
                  <div className="min-w-0">
                    <h2 className="text-lg font-extrabold text-slate-900">
                      {locale.security.title}
                    </h2>
                    <p className="text-xs text-slate-500">
                      {locale.security.subtitle}
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-2 shrink-0">
                  <button
                    type="button"
                    onClick={async () => {
                      await apiClient.createStaffCall(
                        kiosks[0]?.kioskId || 'KIOSK-OPD-01',
                        'ทดสอบสัญญาณเรียกเจ้าหน้าที่จากหน้าตู้ Kiosk',
                      );
                      await refreshAll();
                      flashBanner('success', 'ส่งสัญญาณทดสอบเรียกเจ้าหน้าที่เรียบร้อย');
                    }}
                    className="whitespace-nowrap h-9 px-3.5 inline-flex items-center justify-center rounded-xl border border-amber-200 bg-amber-50 text-[11px] font-bold text-amber-800 hover:bg-amber-100 cursor-pointer shrink-0"
                  >
                    {locale.security.testAlertBtn}
                  </button>
                  <button
                    type="button"
                    onClick={() => void refreshAll()}
                    className="flex h-9 w-9 items-center justify-center rounded-xl bg-slate-900 text-white hover:bg-slate-800 cursor-pointer shrink-0"
                    title={locale.staff.refreshTitle}
                  >
                    <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
                  </button>
                </div>
              </div>

              {openStaffCalls.length > 0 && (
                <div className="rounded-2xl bg-gradient-to-r from-blue-600 to-indigo-600 p-4 text-white shadow-md flex flex-wrap items-center justify-between gap-3">
                  <div className="flex items-center gap-3 min-w-0">
                    <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-white/15">
                      <Bell className="h-5 w-5 animate-bounce" />
                    </div>
                    <div className="min-w-0">
                      <div className="text-sm font-black">
                        {locale.security.emergencyBannerTitle} ({openStaffCalls.length})!
                      </div>
                      <div className="text-[11px] text-blue-100">
                        {locale.security.emergencyBannerDesc}
                      </div>
                    </div>
                  </div>
                  <button
                    type="button"
                    onClick={() => setSecuritySubTab('calls')}
                    className="whitespace-nowrap h-10 px-4 inline-flex items-center justify-center shrink-0 rounded-xl bg-white text-xs font-extrabold text-blue-700 shadow hover:bg-blue-50 cursor-pointer"
                  >
                    {locale.security.viewCallsBtn}
                  </button>
                </div>
              )}

              <div className="flex flex-wrap items-center gap-2 text-xs font-bold">
                {(
                  [
                    { id: 'calls', label: `🔔 ${locale.security.tabCalls} (${staffCalls.length})` },
                    { id: 'audit', label: `🕒 ${locale.security.tabAudit} (${auditLogs.length})` },
                    { id: 'sessions', label: `👥 ${locale.security.tabSessions} (${activeSessions.length})` },
                    { id: 'bans', label: `⊘ ${locale.security.tabBans} (${bannedTargets.length})` },
                    { id: 'kiosks', label: `🖥 ${locale.security.tabKiosks} (${kiosks.length})` },
                  ] as const
                ).map((sub) => (
                  <button
                    key={sub.id}
                    type="button"
                    onClick={() => setSecuritySubTab(sub.id)}
                    className={`whitespace-nowrap h-9 px-4 inline-flex items-center justify-center rounded-xl transition cursor-pointer ${
                      securitySubTab === sub.id
                        ? 'bg-blue-700 text-white shadow-sm'
                        : 'border border-slate-200 bg-white text-slate-600 hover:bg-slate-50'
                    }`}
                  >
                    {sub.label}
                  </button>
                ))}
              </div>

              {securitySubTab === 'calls' && (
                <div className="rounded-2xl border border-slate-200 bg-white shadow-sm overflow-hidden">
                  <div className="flex items-center justify-between border-b border-slate-100 px-5 py-3.5 text-xs">
                    <span className="font-extrabold text-slate-900">{locale.security.callsTableTitle}</span>
                    <span className="text-[11px] text-slate-400 whitespace-nowrap">{locale.security.callsPollNote}</span>
                  </div>
                  <div className="overflow-x-auto">
                    <table className="table-fixed w-full min-w-[760px] text-left text-xs">
                      <thead className="bg-slate-50 text-[11px] font-bold text-slate-500">
                        <tr>
                          <th className="w-[20%] px-5 py-3">{locale.security.colTime}</th>
                          <th className="w-[16%] px-4 py-3">{locale.security.colKiosk}</th>
                          <th className="w-[30%] px-4 py-3">{locale.security.colReason}</th>
                          <th className="w-[16%] px-4 py-3">{locale.security.colStatus}</th>
                          <th className="w-[18%] px-5 py-3 text-right">{locale.security.colActions}</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100">
                        {staffCalls.map((call) => (
                          <tr key={call.id}>
                            <td className="px-5 py-3 text-slate-600 whitespace-nowrap truncate">
                              {lang === 'th'
                                ? formatThaiDateTime(call.createdAt)
                                : new Date(call.createdAt).toLocaleString(lang === 'cn' ? 'zh-CN' : 'en-US')}
                            </td>
                            <td className="px-4 py-3">
                              <span className="inline-block whitespace-nowrap rounded-lg bg-sky-50 border border-sky-200 px-2.5 py-1 font-bold text-sky-700">
                                {call.kioskId}
                              </span>
                            </td>
                            <td className="px-4 py-3 font-semibold text-rose-600 truncate">{call.reason}</td>
                            <td className="px-4 py-3">
                              <span
                                className={`inline-flex items-center gap-1 whitespace-nowrap rounded-full px-2.5 py-1 text-[11px] font-bold ${
                                  call.status === 'resolved'
                                    ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                                    : call.status === 'acknowledged'
                                      ? 'bg-amber-50 text-amber-800 border border-amber-200'
                                      : 'bg-sky-50 text-sky-700 border border-sky-200'
                                }`}
                              >
                                {call.status === 'resolved'
                                  ? locale.security.statusResolved
                                  : call.status === 'acknowledged'
                                    ? locale.security.statusAck
                                    : locale.security.statusOpen}
                              </span>
                            </td>
                            <td className="px-5 py-3 text-right">
                              <div className="inline-flex items-center justify-end gap-1.5">
                                {call.status === 'open' && (
                                  <button
                                    type="button"
                                    onClick={async () => {
                                      await apiClient.acknowledgeStaffCall(call.id);
                                      flashBanner('success', `รับเรื่องแจ้งเตือนจาก ${call.kioskId} แล้ว`);
                                      await refreshAll();
                                    }}
                                    className="whitespace-nowrap rounded-lg border border-amber-300 bg-amber-50 px-2.5 py-1 text-[11px] font-bold text-amber-800 hover:bg-amber-100 cursor-pointer"
                                  >
                                    {locale.security.ackBtn}
                                  </button>
                                )}
                                {call.status !== 'resolved' && (
                                  <button
                                    type="button"
                                    onClick={async () => {
                                      await apiClient.resolveStaffCall(call.id);
                                      flashBanner('success', `ปิดเคสช่วยเหลือจาก ${call.kioskId} แล้ว`);
                                      await refreshAll();
                                    }}
                                    className="whitespace-nowrap rounded-lg bg-emerald-600 px-3 py-1 text-[11px] font-bold text-white hover:bg-emerald-500 cursor-pointer"
                                  >
                                    {locale.security.resolveBtn}
                                  </button>
                                )}
                              </div>
                            </td>
                          </tr>
                        ))}
                        {staffCalls.length === 0 && (
                          <tr>
                            <td colSpan={5} className="px-6 py-8 text-center text-slate-400">
                              {locale.security.emptyCalls}
                            </td>
                          </tr>
                        )}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {securitySubTab === 'audit' && (
                <div className="rounded-2xl border border-slate-200 bg-white p-5 space-y-3">
                  {!isAdmin ? (
                    <div className="rounded-xl bg-amber-50 border border-amber-200 p-4 text-xs font-bold text-amber-800">
                      403 RBAC Policy:{' '}
                      {lang === 'cn'
                        ? '仅医院院长 (Super Admin) 可查看审计日志'
                        : lang === 'en'
                          ? 'Only Super Admin can access Audit Logs'
                          : 'เฉพาะผู้อำนวยการโรงพยาบาล (Super Admin) เท่านั้นที่เข้าถึงบันทึก Audit Log ได้'}
                    </div>
                  ) : (
                    <>
                      <div className="relative max-w-md">
                        <Search className="h-4 w-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
                        <input
                          value={auditSearch}
                          onChange={(e) => setAuditSearch(e.target.value)}
                          placeholder={
                            lang === 'cn'
                              ? '按操作、执行者或 IP 筛选审计日志...'
                              : lang === 'en'
                                ? 'Filter audit logs by action, actor, IP...'
                                : 'กรอง Audit Log ตาม Action, ชื่อผู้ทำรายการ, IP...'
                          }
                          className="w-full h-10 rounded-xl border border-slate-200 pl-9 pr-3 text-xs"
                        />
                      </div>
                      <div className="space-y-2 max-h-[420px] overflow-y-auto">
                        {filteredAuditLogs.map((log) => (
                          <div
                            key={log.id}
                            className="flex flex-wrap items-center justify-between gap-2 rounded-xl border border-slate-100 bg-slate-50/80 px-4 py-2.5 text-xs"
                          >
                            <div className="min-w-0">
                              <span className="font-mono font-extrabold text-indigo-700 mr-2">{log.eventType}</span>
                              <span className="font-bold text-slate-900">{log.actorName}</span>{' '}
                              <span className="text-slate-500">({log.actorRole})</span>
                              <div className="text-[11px] text-slate-600 mt-0.5">{log.detail}</div>
                            </div>
                            <div className="text-right text-[11px] text-slate-400 font-mono whitespace-nowrap">
                              <div>
                                IP: {log.ipAddress} • Dev: {log.deviceId}
                              </div>
                              <div>
                                {lang === 'th'
                                  ? formatThaiDateTime(log.createdAt)
                                  : new Date(log.createdAt).toLocaleString(lang === 'cn' ? 'zh-CN' : 'en-US')}
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </>
                  )}
                </div>
              )}

              {securitySubTab === 'sessions' && (
                <div className="rounded-2xl border border-slate-200 bg-white p-5 space-y-3">
                  {!isAdmin ? (
                    <div className="rounded-xl bg-amber-50 border border-amber-200 p-4 text-xs font-bold text-amber-800">
                      403 RBAC Policy:{' '}
                      {lang === 'cn'
                        ? '仅医院院长 (Super Admin) 可强制注销会话'
                        : lang === 'en'
                          ? 'Only Super Admin can terminate active sessions'
                          : 'เฉพาะผู้อำนวยการโรงพยาบาล (Super Admin) เท่านั้นที่สั่งตัดเซสชันผู้ใช้ได้'}
                    </div>
                  ) : (
                    <div className="space-y-2">
                      {activeSessions.map((s) => (
                        <div
                          key={s.token}
                          className="flex flex-wrap items-center justify-between gap-2 rounded-xl border border-slate-200 px-4 py-3 text-xs"
                        >
                          <div className="min-w-0">
                            <div className="font-bold text-slate-900">
                              {ptNameMap.get(s.userId) ?? s.userId}{' '}
                              <span className="ml-1 rounded bg-slate-100 px-2 py-0.5 font-mono text-[10px] uppercase">
                                {s.role}
                              </span>
                            </div>
                            <div className="font-mono text-[11px] text-slate-500">
                              IP: {s.ipAddress} • Device: {s.deviceId} • Login:{' '}
                              {lang === 'th'
                                ? formatThaiDateTime(s.createdAt)
                                : new Date(s.createdAt).toLocaleString(lang === 'cn' ? 'zh-CN' : 'en-US')}
                            </div>
                          </div>
                          <button
                            type="button"
                            onClick={async () => {
                              await apiClient.forceLogoutUser(s.userId);
                              flashBanner('success', 'เตะเซสชันออกจากระบบเรียบร้อย');
                              await refreshAll();
                            }}
                            className="whitespace-nowrap h-9 px-3.5 inline-flex items-center justify-center rounded-xl bg-rose-600 text-[11px] font-bold text-white hover:bg-rose-500 cursor-pointer shrink-0"
                          >
                            {lang === 'cn'
                              ? '强制注销 (Force Kick)'
                              : lang === 'en'
                                ? 'Force Kick Session'
                                : 'เตะออกจากระบบ (Force Kick)'}
                          </button>
                        </div>
                      ))}
                      {activeSessions.length === 0 && (
                        <p className="text-center text-xs text-slate-400 py-6">
                          {lang === 'cn'
                            ? '当前暂无在线会话'
                            : lang === 'en'
                              ? 'No active online sessions'
                              : 'ไม่มีเซสชันที่กำลังออนไลน์ในระบบ'}
                        </p>
                      )}
                    </div>
                  )}
                </div>
              )}

              {securitySubTab === 'bans' && (
                <div className="rounded-2xl border border-slate-200 bg-white p-5 space-y-4">
                  {!isAdmin ? (
                    <div className="rounded-xl bg-amber-50 border border-amber-200 p-4 text-xs font-bold text-amber-800">
                      403 RBAC Policy:{' '}
                      {lang === 'cn'
                        ? '仅医院院长 (Super Admin) 可封禁 IP/设备'
                        : lang === 'en'
                          ? 'Only Super Admin can ban IP/Devices'
                          : 'เฉพาะผู้อำนวยการโรงพยาบาล (Super Admin) เท่านั้นที่แบน IP/Device ได้'}
                    </div>
                  ) : (
                    <>
                      <form
                        onSubmit={async (e) => {
                          e.preventDefault();
                          if (!banValue.trim()) return;
                          await apiClient.banTarget(banType, banValue.trim(), banReason.trim() || 'Security block');
                          setBanValue('');
                          setBanReason('');
                          flashBanner('success', 'เพิ่มรายการแบนเรียบร้อยแล้ว');
                          await refreshAll();
                        }}
                        className="grid grid-cols-1 sm:grid-cols-4 gap-2 text-xs"
                      >
                        <select
                          value={banType}
                          onChange={(e) => setBanType(e.target.value as 'ip' | 'device')}
                          className="h-10 rounded-xl border border-slate-200 px-3 font-bold"
                        >
                          <option value="ip">Ban IP Address</option>
                          <option value="device">Ban Device ID</option>
                        </select>
                        <input
                          required
                          value={banValue}
                          onChange={(e) => setBanValue(e.target.value)}
                          placeholder={lang === 'cn' ? '如 192.168.1.50' : lang === 'en' ? 'e.g. 192.168.1.50' : 'เช่น 192.168.1.50'}
                          className="h-10 rounded-xl border border-slate-200 px-3 font-mono"
                        />
                        <input
                          value={banReason}
                          onChange={(e) => setBanReason(e.target.value)}
                          placeholder={lang === 'cn' ? '封禁原因...' : lang === 'en' ? 'Ban reason...' : 'สาเหตุการแบน...'}
                          className="h-10 rounded-xl border border-slate-200 px-3"
                        />
                        <button
                          type="submit"
                          className="whitespace-nowrap h-10 px-4 inline-flex items-center justify-center gap-1 rounded-xl bg-rose-600 font-bold text-white hover:bg-rose-500 cursor-pointer"
                        >
                          <Ban className="h-3.5 w-3.5 shrink-0" />{' '}
                          {lang === 'cn' ? '封禁目标' : lang === 'en' ? 'Ban Target' : 'แบนเป้าหมาย'}
                        </button>
                      </form>

                      <div className="space-y-2">
                        {bannedTargets.map((b) => (
                          <div
                            key={b.id}
                            className="flex items-center justify-between gap-2 rounded-xl border border-rose-200 bg-rose-50/50 px-4 py-2.5 text-xs"
                          >
                            <div className="min-w-0 truncate">
                              <span className="font-mono font-bold text-rose-700 mr-2">
                                [{b.targetType.toUpperCase()}] {b.value}
                              </span>
                              <span className="text-slate-600">{b.reason}</span>
                            </div>
                            <button
                              type="button"
                              onClick={async () => {
                                await apiClient.unbanTarget(b.id);
                                flashBanner('success', `ปลดแบน ${b.value} แล้ว`);
                                await refreshAll();
                              }}
                              className="whitespace-nowrap h-8 px-3 rounded-lg border border-slate-200 bg-white font-bold text-slate-700 hover:bg-slate-100 cursor-pointer shrink-0"
                            >
                              {lang === 'cn' ? '解除封禁 (Unban)' : lang === 'en' ? 'Unban' : 'ปลดแบน (Unban)'}
                            </button>
                          </div>
                        ))}
                      </div>
                    </>
                  )}
                </div>
              )}

              {securitySubTab === 'kiosks' && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="rounded-2xl border border-slate-200 bg-white p-5 space-y-3">
                    <div className="flex items-center justify-between gap-2">
                      <h3 className="text-sm font-extrabold text-slate-900">{locale.security.kioskPanelTitle}</h3>
                      {isAdmin && (
                        <button
                          type="button"
                          onClick={() => {
                            setNewKioskCode(`KIOSK-OPD-0${kiosks.length + 1}`);
                            setNewKioskLocation('OPD Rehab Zone B');
                            setRegisterKioskModalOpen(true);
                          }}
                          className="whitespace-nowrap h-9 px-3.5 inline-flex items-center justify-center shrink-0 rounded-xl bg-emerald-600 text-[11px] font-bold text-white hover:bg-emerald-500 cursor-pointer"
                        >
                          {locale.security.registerKioskBtn}
                        </button>
                      )}
                    </div>
                    {kiosks.map((k) => (
                      <div
                        key={k.kioskId}
                        className="flex items-center justify-between gap-2 rounded-xl border border-slate-200 bg-slate-50/70 px-4 py-3 text-xs"
                      >
                        <div className="min-w-0">
                          <div className="font-bold text-slate-900 truncate">
                            {k.kioskId}{' '}
                            {k.pairingToken && (
                              <span className="ml-1 rounded bg-indigo-50 px-2 py-0.5 font-mono text-[10px] text-indigo-700">
                                Token: {k.pairingToken}
                              </span>
                            )}
                          </div>
                          <div className="text-[11px] text-slate-500 truncate">
                            {k.location} • State: {k.currentState}
                          </div>
                        </div>
                        <div className="flex items-center gap-2 shrink-0">
                          <span
                            className={`whitespace-nowrap rounded-full px-2.5 py-0.5 text-[11px] font-bold ${
                              k.online ? 'bg-emerald-100 text-emerald-700' : 'bg-slate-200 text-slate-600'
                            }`}
                          >
                            {k.online ? '● Online' : 'Offline'}
                          </span>
                          {isAdmin && k.kioskId !== 'KIOSK-OPD-01' && (
                            <button
                              type="button"
                              onClick={async () => {
                                await apiClient.deleteKiosk(k.kioskId);
                                await refreshAll();
                              }}
                              className="text-rose-500 hover:text-rose-700"
                            >
                              <Trash2 className="h-3.5 w-3.5" />
                            </button>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>

                  <div className="rounded-2xl border border-slate-200 bg-white p-5 space-y-3 text-xs">
                    <h3 className="text-sm font-extrabold text-slate-900">{locale.security.securitySettingsTitle}</h3>
                    <label className="flex items-center justify-between rounded-xl border border-slate-200 px-3.5 py-2.5">
                      <span className="font-bold text-slate-700">
                        {lang === 'cn'
                          ? '启用 AI 人脸识别 (512D)'
                          : lang === 'en'
                            ? 'Enable AI Face Scan (512D)'
                            : 'เปิดใช้งาน AI Face Scan (512D)'}
                      </span>
                      <input
                        type="checkbox"
                        disabled={!isAdmin}
                        checked={systemSettings.aiFaceScanEnabled}
                        onChange={async (e) => {
                          const updated = await apiClient.updateSystemSettings({
                            aiFaceScanEnabled: e.target.checked,
                          });
                          setSystemSettings(updated);
                        }}
                      />
                    </label>
                    <div className="rounded-xl border border-slate-200 px-3.5 py-2.5 space-y-1">
                      <div className="flex justify-between font-bold text-slate-700">
                        <span>Face Cosine Similarity Threshold</span>
                        <span className="font-mono text-emerald-700">{systemSettings.aiFaceThreshold.toFixed(2)}</span>
                      </div>
                      <input
                        type="range"
                        disabled={!isAdmin}
                        min={0.5}
                        max={0.95}
                        step={0.01}
                        value={systemSettings.aiFaceThreshold}
                        onChange={async (e) => {
                          const updated = await apiClient.updateSystemSettings({
                            aiFaceThreshold: Number(e.target.value),
                          });
                          setSystemSettings(updated);
                        }}
                        className="w-full"
                      />
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </main>
      </div>

      {/* Staff-Assisted Enrollment / Existing Patient Face Scan Modal */}
      <StaffAssistedEnrollmentModal
        open={enrollModalOpen}
        apiClient={apiClient}
        currentUser={currentUser}
        ptList={ptUsers}
        targetPatient={enrollTargetPatient}
        onClose={() => {
          setEnrollModalOpen(false);
          setEnrollTargetPatient(null);
        }}
        onEnrolled={(p) => {
          flashBanner('success', `ลงทะเบียนคนไข้และบันทึก Face ID 120 เฟรมสำเร็จ! (${p.fullName} • ${p.hn})`);
          void refreshAll();
        }}
      />

      {/* Welcome & Face Verification Modal */}
      <WelcomeFaceScanModal
        open={welcomeFaceModalOpen}
        apiClient={apiClient}
        onClose={() => setWelcomeFaceModalOpen(false)}
        onOpenNewEnrollment={() => {
          setEnrollTargetPatient(null);
          setEnrollModalOpen(true);
        }}
        onCheckInSuccess={(pid, pname) => {
          setSelectedSchedulePatientId(pid);
          setActiveTab('schedules');
          flashBanner('success', `ยืนยันใบหน้าและเช็คอิน ${pname} เข้าฝึกเรียบร้อยแล้ว`);
          void refreshAll();
        }}
      />

      {/* Reset Password Modal */}
      <ResetPasswordModal
        user={resetPassUser}
        apiClient={apiClient}
        onClose={() => setResetPassUser(null)}
        onSuccess={(msg) => flashBanner('success', msg)}
      />

      {/* Patient Profile, Medical Notes & Rehab History Modal */}
      <PatientProfileAndHistoryModal
        patient={patientModalState.patient}
        initialTab={patientModalState.tab}
        apiClient={apiClient}
        currentUser={currentUser}
        ptUsers={ptUsers}
        sessions={sessions}
        onClose={() => setPatientModalState({ patient: null, tab: 'profile' })}
        onRefresh={refreshAll}
        onOpenFaceScan={(p) => {
          setEnrollTargetPatient(p);
          setEnrollModalOpen(true);
        }}
        onFlash={flashBanner}
      />

      {/* Schedule Creator & Day Detail Modal */}
      <ScheduleCreatorAndDayModal
        mode={scheduleModalMode}
        selectedDateIso={selectedCalendarDateIso}
        patient={selectedSchedulePatient}
        patients={patients}
        ptUsers={ptUsers}
        exercises={exercises}
        modpacks={modpacks}
        apiClient={apiClient}
        onClose={() => setScheduleModalMode(null)}
        onSaved={async (msg) => {
          flashBanner('success', msg);
          await refreshAll();
        }}
      />

      {/* Add Staff Modal */}
      {addStaffModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/60 p-4 backdrop-blur-sm">
          <div className="w-full max-w-md rounded-3xl bg-white p-6 shadow-2xl border border-slate-200 space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-base font-extrabold text-slate-900">+ เพิ่มเจ้าหน้าที่ใหม่</h3>
              <button type="button" onClick={() => setAddStaffModalOpen(false)} className="text-slate-400">
                <X className="h-5 w-5" />
              </button>
            </div>
            <form onSubmit={handleCreateStaff} className="space-y-3 text-xs">
              <div>
                <label className="block font-bold text-slate-600 mb-1">ชื่อ-นามสกุล *</label>
                <input
                  required
                  value={newStaffName}
                  onChange={(e) => setNewStaffName(e.target.value)}
                  placeholder="เช่น กภ. วิภาวี สุขใจ"
                  className="w-full rounded-xl border border-slate-200 px-3 py-2"
                />
              </div>
              <div className="grid grid-cols-2 gap-2.5">
                <div>
                  <label className="block font-bold text-slate-600 mb-1">Username *</label>
                  <input
                    required
                    value={newStaffUsername}
                    onChange={(e) => setNewStaffUsername(e.target.value)}
                    placeholder="pt_wipawee"
                    className="w-full rounded-xl border border-slate-200 px-3 py-2"
                  />
                </div>
                <div>
                  <label className="block font-bold text-slate-600 mb-1">รหัสผ่าน *</label>
                  <input
                    required
                    type="password"
                    minLength={6}
                    value={newStaffPassword}
                    onChange={(e) => setNewStaffPassword(e.target.value)}
                    placeholder="อย่างน้อย 6 ตัว"
                    className="w-full rounded-xl border border-slate-200 px-3 py-2"
                  />
                </div>
              </div>
              <div className="grid grid-cols-2 gap-2.5">
                <div>
                  <label className="block font-bold text-slate-600 mb-1">บทบาท</label>
                  <select
                    value={newStaffRole}
                    onChange={(e) => setNewStaffRole(e.target.value as 'pt' | 'super_admin')}
                    className="w-full rounded-xl border border-slate-200 px-3 py-2 font-bold"
                  >
                    <option value="pt">นักกายภาพบำบัด (PT)</option>
                    <option value="super_admin">ผู้อำนวยการ (Admin)</option>
                  </select>
                </div>
                <div>
                  <label className="block font-bold text-slate-600 mb-1">เบอร์โทรศัพท์</label>
                  <input
                    value={newStaffPhone}
                    onChange={(e) => setNewStaffPhone(e.target.value)}
                    placeholder="08X-XXX-XXXX"
                    className="w-full rounded-xl border border-slate-200 px-3 py-2"
                  />
                </div>
              </div>
              {newStaffRole === 'pt' && (
                <div className="grid grid-cols-2 gap-2.5">
                  <div>
                    <label className="block font-bold text-slate-600 mb-1">ความเชี่ยวชาญ</label>
                    <input
                      value={newStaffSpecialty}
                      onChange={(e) => setNewStaffSpecialty(e.target.value)}
                      className="w-full rounded-xl border border-slate-200 px-3 py-2"
                    />
                  </div>
                  <div>
                    <label className="block font-bold text-slate-600 mb-1">เลขที่ใบประกอบวิชาชีพ *</label>
                    <input
                      required
                      value={newStaffLicense}
                      onChange={(e) => setNewStaffLicense(e.target.value)}
                      placeholder="PT-44100"
                      className="w-full rounded-xl border border-slate-200 px-3 py-2"
                    />
                  </div>
                </div>
              )}
              <div className="flex justify-end gap-2 pt-2">
                <button
                  type="button"
                  onClick={() => setAddStaffModalOpen(false)}
                  className="rounded-xl border border-slate-200 px-4 py-2 font-bold text-slate-600"
                >
                  ยกเลิก
                </button>
                <button
                  type="submit"
                  disabled={submitting}
                  className="rounded-xl bg-emerald-600 px-4 py-2 font-extrabold text-white hover:bg-emerald-500 disabled:opacity-50"
                >
                  {submitting ? 'กำลังบันทึก...' : 'สร้างบัญชีเจ้าหน้าที่'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* PT Detail & Assign Patient Modal */}
      {selectedPtDetailModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/60 p-4 backdrop-blur-sm">
          <div className="w-full max-w-lg rounded-3xl bg-white p-6 shadow-2xl border border-slate-200 space-y-4 text-xs">
            <div className="flex items-start justify-between">
              <div>
                <h3 className="text-base font-black text-slate-900">{selectedPtDetailModal.fullName}</h3>
                <p className="text-slate-500">
                  @{selectedPtDetailModal.username} • {selectedPtDetailModal.licenseNo || 'PT-44012'} •{' '}
                  {selectedPtDetailModal.specialty || 'กายภาพบำบัด'}
                </p>
              </div>
              <button type="button" onClick={() => setSelectedPtDetailModal(null)} className="text-slate-400">
                <X className="h-5 w-5" />
              </button>
            </div>

            {isAdmin && patients.length > 0 && (
              <div className="rounded-2xl border border-emerald-200 bg-emerald-50/50 p-3.5 space-y-2">
                <div className="font-bold text-emerald-900">มอบหมายคนไข้ให้ PT ท่านนี้ดูแล</div>
                <div className="flex gap-2">
                  <select
                    value={assignPatientIdForPt}
                    onChange={(e) => setAssignPatientIdForPt(e.target.value)}
                    className="flex-1 rounded-xl border border-slate-200 bg-white px-3 py-2 font-semibold"
                  >
                    {patients.map((p) => (
                      <option key={p.id} value={p.id}>
                        {p.fullName} ({p.hn})
                      </option>
                    ))}
                  </select>
                  <button
                    type="button"
                    disabled={!assignPatientIdForPt || submitting}
                    onClick={async () => {
                      setSubmitting(true);
                      try {
                        await apiClient.updatePatient(assignPatientIdForPt, {
                          assignedPtId: selectedPtDetailModal.id,
                        });
                        flashBanner('success', `มอบหมายคนไข้ให้ ${selectedPtDetailModal.fullName} เรียบร้อย`);
                        await refreshAll();
                      } finally {
                        setSubmitting(false);
                      }
                    }}
                    className="rounded-xl bg-emerald-600 px-3.5 py-2 font-bold text-white hover:bg-emerald-500 cursor-pointer"
                  >
                    มอบหมาย
                  </button>
                </div>
              </div>
            )}

            <div className="space-y-2 max-h-60 overflow-y-auto">
              <div className="font-bold text-slate-700">รายชื่อคนไข้ที่อยู่ภายใต้การดูแล</div>
              {patients
                .filter(
                  (p) => p.assignedPtId === selectedPtDetailModal.id || p.delegatedToPtId === selectedPtDetailModal.id,
                )
                .map((p) => (
                  <div key={p.id} className="flex items-center justify-between rounded-xl border border-slate-200 px-3.5 py-2.5">
                    <div>
                      <span className="font-mono font-bold text-sky-700 mr-2">{p.hn}</span>
                      <span className="font-bold text-slate-900">{p.fullName}</span>
                    </div>
                    <button
                      type="button"
                      onClick={() => {
                        setSelectedPtDetailModal(null);
                        setSelectedSchedulePatientId(p.id);
                        setActiveTab('schedules');
                      }}
                      className="rounded-lg bg-emerald-50 px-2.5 py-1 font-bold text-emerald-700 cursor-pointer"
                    >
                      ดูตารางนัด
                    </button>
                  </div>
                ))}
            </div>
          </div>
        </div>
      )}

      {/* Quick Check-in Modal */}
      {quickCheckinModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/60 p-4 backdrop-blur-sm">
          <div className="w-full max-w-md rounded-3xl bg-white p-6 shadow-2xl border border-slate-200 space-y-4 text-xs">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Zap className="h-5 w-5 text-amber-500" />
                <h3 className="text-base font-black text-slate-900">เช็คอินเร่งด่วน (Quick Check-in)</h3>
              </div>
              <button type="button" onClick={() => setQuickCheckinModalOpen(false)} className="text-slate-400">
                <X className="h-5 w-5" />
              </button>
            </div>
            <input
              value={quickCheckinSearch}
              onChange={(e) => setQuickCheckinSearch(e.target.value)}
              placeholder="พิมพ์ HN, ชื่อ-นามสกุล, หรือเบอร์โทรคนไข้..."
              className="w-full rounded-xl border border-slate-200 px-3.5 py-2.5 text-sm"
            />
            <div className="max-h-60 overflow-y-auto space-y-2">
              {patients
                .filter(
                  (p) =>
                    !quickCheckinSearch.trim() ||
                    p.hn.toLowerCase().includes(quickCheckinSearch.toLowerCase()) ||
                    p.fullName.toLowerCase().includes(quickCheckinSearch.toLowerCase()) ||
                    (p.phone || '').toLowerCase().includes(quickCheckinSearch.toLowerCase()),
                )
                .slice(0, 6)
                .map((p) => (
                  <div key={p.id} className="flex items-center justify-between rounded-xl border border-slate-200 p-3">
                    <div>
                      <div className="font-bold text-slate-900">{p.fullName}</div>
                      <div className="font-mono text-[11px] text-sky-700">
                        {p.hn} • {p.phone || '-'}
                      </div>
                    </div>
                    <button
                      type="button"
                      disabled={submitting}
                      onClick={async () => {
                        setSubmitting(true);
                        try {
                          await apiClient.quickCheckInPatient(p.id, 'เช็คอินเร่งด่วนหน้าเคาน์เตอร์');
                          setQuickCheckinModalOpen(false);
                          setSelectedSchedulePatientId(p.id);
                          setActiveTab('schedules');
                          flashBanner('success', `เช็คอินคนไข้ ${p.fullName} (${p.hn}) และบันทึก Audit Log สำเร็จ`);
                          await refreshAll();
                        } finally {
                          setSubmitting(false);
                        }
                      }}
                      className="rounded-xl bg-emerald-600 px-3 py-1.5 font-bold text-white hover:bg-emerald-500 cursor-pointer"
                    >
                      เช็คอินวันนี้
                    </button>
                  </div>
                ))}
            </div>
          </div>
        </div>
      )}

      {/* Register Kiosk Modal */}
      {registerKioskModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/60 p-4 backdrop-blur-sm">
          <div className="w-full max-w-sm rounded-3xl bg-white p-6 shadow-2xl border border-slate-200 space-y-4 text-xs">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-black text-slate-900">+ ลงทะเบียนตู้ Kiosk ใหม่</h3>
              <button type="button" onClick={() => setRegisterKioskModalOpen(false)} className="text-slate-400">
                <X className="h-4 w-4" />
              </button>
            </div>
            <form
              onSubmit={async (e) => {
                e.preventDefault();
                const created = await apiClient.registerKiosk({
                  kioskId: newKioskCode,
                  location: newKioskLocation,
                });
                setRegisterKioskModalOpen(false);
                flashBanner('success', `ลงทะเบียน ${created.kioskId} สำเร็จ (Token: ${created.pairingToken})`);
                await refreshAll();
              }}
              className="space-y-3"
            >
              <input
                required
                value={newKioskCode}
                onChange={(e) => setNewKioskCode(e.target.value)}
                placeholder="รหัสตู้ (เช่น KIOSK-OPD-02)"
                className="w-full rounded-xl border border-slate-200 px-3 py-2 font-mono font-bold"
              />
              <input
                required
                value={newKioskLocation}
                onChange={(e) => setNewKioskLocation(e.target.value)}
                placeholder="ตำแหน่งติดตั้ง (เช่น ชั้น 2 แผนกฟื้นฟู)"
                className="w-full rounded-xl border border-slate-200 px-3 py-2"
              />
              <div className="flex justify-end gap-2">
                <button
                  type="button"
                  onClick={() => setRegisterKioskModalOpen(false)}
                  className="rounded-xl border border-slate-200 px-4 py-2 font-bold"
                >
                  ยกเลิก
                </button>
                <button type="submit" className="rounded-xl bg-emerald-600 px-4 py-2 font-extrabold text-white">
                  สร้าง Pairing Token
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Physical Card Game Modal — Full-Size Kiosk View */}
      {cardGameModalOpen && (
        <div className="fixed inset-0 z-50 flex flex-col w-screen h-screen bg-slate-100 overflow-y-auto overflow-x-hidden">
          <div className="flex items-center justify-between gap-4 bg-white border-b-2 border-slate-200 px-5 py-3.5 sm:px-8 sm:py-4 shrink-0 shadow-sm">
            <div className="flex items-center gap-3.5 min-w-0">
              <Gamepad2 className="h-10 w-10 text-emerald-600 shrink-0" />
              <h3 className="mg-modal-title font-black text-slate-900 truncate">
                เกมการ์ดกายภาพบำบัด 10 ข้อ (ทดสอบระบบมือ/กล้อง หรือคลิกปุ่มตอบ)
              </h3>
            </div>
            <button
              type="button"
              onClick={() => setCardGameModalOpen(false)}
              className="flex h-14 w-14 items-center justify-center rounded-2xl bg-slate-100 text-slate-600 hover:bg-rose-100 hover:text-rose-700 cursor-pointer shrink-0 transition"
            >
              <X className="h-9 w-9" />
            </button>
          </div>
          <div className="flex-1 flex flex-col w-full min-h-0 p-3 sm:p-5">
            <PhysicalCardGame
              patientName={selectedSchedulePatient?.fullName ?? currentUser.fullName}
              onSessionFinished={async (summary) => {
                const targetPatient = selectedSchedulePatient ?? patients[0];
                if (targetPatient?.id) {
                  try {
                    const created = await apiClient.recordRehabSession({
                      patientId: targetPatient.id,
                      activityType: 'minigame',
                      exerciseId: 'cognitive-card-game',
                      exerciseName: 'มินิเกมฝึกสมอง (การ์ดกายภาพ 10 ข้อ)',
                      completedSets: 1,
                      completedReps: summary.total,
                      accuracyPercent: summary.accuracyPct,
                      maxRomDeg: 0,
                      painStopTriggered: false,
                      aiFeedback: `Cognitive Accuracy ${summary.accuracyPct}% (${summary.correct}/${summary.total}) in ${summary.durationSec}s`,
                      kinematicsReplay: [],
                      minigameScore: {
                        correct: summary.correct,
                        total: summary.total,
                        category: summary.category,
                      },
                    });
                    setSessions((prev) => [created, ...prev]);
                  } catch {
                    // ignore background session sync failure
                  }
                }
                flashBanner(
                  'success',
                  `ทดสอบการ์ดกายภาพเสร็จสิ้น (${summary.correct}/${summary.total} คะแนน • ${summary.accuracyPct}% • ${summary.durationSec} วินาที)`,
                );
              }}
              onComplete={(summary) => {
                setCardGameModalOpen(false);
                flashBanner(
                  'success',
                  `ทดสอบการ์ดกายภาพเสร็จสิ้น (${summary.correct}/${summary.total} คะแนน หมวด ${summary.category})`,
                );
              }}
            />
          </div>
        </div>
      )}
    </div>
  );
}
