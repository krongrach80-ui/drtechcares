import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ApiError, formatLocalIsoDate, isAdminRole, resolveNextAppointmentDate } from '@drtechcares/api-client';
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
import {
  CLINICAL_EXERCISES,
  STROKE_CANONICAL_EXERCISE_SET,
  isDeprecatedExerciseRecord,
  mergeDuplicateSideExercises,
  parseExerciseCodeAndSide,
  sortExercisesStrokeUpperFirst,
  type BodySide,
  type ExerciseConfig,
} from '@drtechcares/pose-engine';
import {
  Activity,
  Award,
  Ban,
  Bell,
  Calendar,
  CalendarDays,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  ClipboardList,
  Clock,
  FileText,
  Gamepad2,
  HeartHandshake,
  KeyRound,
  Link2,
  Lock,
  LogOut,
  Monitor,
  PauseCircle,
  Pencil,
  Phone,
  Play,
  Plus,
  RefreshCw,
  Save,
  ScanFace,
  Search,
  Shield,
  ShieldAlert,
  SlidersHorizontal,
  Trash2,
  Upload,
  User,
  UserCheck,
  UserPlus,
  Users,
  Wifi,
  WifiOff,
  X,
  Zap,
} from 'lucide-react';
import { PhysicalCardGame } from '../../apps/kiosk/src/PhysicalCardGame';
import { POSE_TEST_PATIENT, buildAll15ClinicalExercisePrescriptions } from '../hooks/useKioskSocket';
import { PatientClinicalRecordPanel } from './PatientClinicalRecordPanel';
import { SmartRehabKioskUI } from './SmartRehabKioskUI';
import { notifyBiometricIndexChanged } from '../lib/faceMatcher';
import { removeProfileFromLocalBiometricVault } from '../lib/biometricEngine';
import {
  fetchAllPatientsUnified,
  filterAdminPortalPatients,
  normalizePatientRecord,
  type PatientScopeFilter,
} from '../lib/patientHydration';
import {
  applyCascadingPatientDeletionToState,
  cascadeDeletePatientInSupabase,
  cleanupDeletedPatientLocalStorage,
  formatCanonicalHn,
  mergePatientsIntoStaffUsersList,
  normalizeHnKey,
  unifyClinicPatientsDetailed,
  upsertPatientIntoStaffUsersState,
  upsertPatientStaffUserInSupabase,
} from '../utils/unifyPatientDirectory';
import { StaffAssistedEnrollmentModal } from './StaffAssistedEnrollmentModal';
import {
  PatientProfileAndHistoryModal,
  ResetPasswordModal,
  ScheduleCreatorAndDayModal,
  UniversalEhrLinkModal,
  WelcomeFaceScanModal,
} from './PortalSubModals';

export interface ClinicalStaffPortalShellProps {
  apiClient: DrTechCaresApiClient;
  currentUser: UserAccount;
  onLogout: () => Promise<void> | void;
  onReturnToKiosk?: () => void;
  onOpenPoseTestMode?: (initialExerciseCode?: string) => void;
  initialTab?: PortalTab;
  autoOpenEnroll?: boolean;
  onAutoOpenEnrollConsumed?: () => void;
  initialPatients?: PatientRecord[];
  initialUsers?: UserAccount[];
  initialPatientScope?: PatientScopeFilter;
  initialPatientSyncError?: string | null;
}

type PortalTab = 'staff' | 'patients' | 'assessments' | 'pts' | 'exercises' | 'schedules' | 'security' | 'profile';
type SecuritySubTab = 'calls' | 'audit' | 'sessions' | 'bans' | 'kiosks';

interface PortalSectionCopy {
  nav: Record<PortalTab, string>;
  headers: Record<PortalTab, { title: string; subtitle: string }>;
  profile: {
    sidebarCardTitle: string;
    sidebarEditHint: string;
    previewTitle: string;
    editFormTitle: string;
    avatarSectionLabel: string;
    uploadPhotoBtn: string;
    presetLabel: string;
    fullNameLabel: string;
    phoneLabel: string;
    specialtyLabel: string;
    licenseLabel: string;
    bioLabel: string;
    bioPlaceholder: string;
    saveBtn: string;
    savingBtn: string;
    patientsCountLabel: string;
    savedSuccess: string;
    noPhone: string;
  };
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
    updateFaceBtn: string;
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
      assessments: 'เวชระเบียนรายบุคคล (ประเมิน 0-6)',
      pts: '3. ข้อมูลนักกายภาพ',
      exercises: '4. ท่ากายภาพและมินิเกม/modpack',
      schedules: '5. ตารางนัด',
      security: '6. ประวัติและความปลอดภัย',
      profile: '7. ประวัติส่วนตัว',
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
      assessments: {
        title: 'เวชระเบียนรายบุคคลและเกณฑ์ประเมิน 0-6 คะแนน',
        subtitle: 'LONGITUDINAL ROM PROGRESSION & CROSS-FACILITY CLINICAL ASSESSMENTS',
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
      profile: {
        title: 'ประวัติส่วนตัวนักกายภาพบำบัดและบุคลากร',
        subtitle: 'PHYSIOTHERAPIST PERSONAL PROFILE & CREDENTIALS',
      },
    },
    profile: {
      sidebarCardTitle: 'ประวัติส่วนตัว',
      sidebarEditHint: 'แก้ไขข้อมูล / รูป',
      previewTitle: 'บัตรประวัติส่วนตัว (Profile Preview)',
      editFormTitle: 'แก้ไขข้อมูลประวัติส่วนตัว',
      avatarSectionLabel: '1. รูปประกอบ (ภาพประจำตัว)',
      uploadPhotoBtn: 'อัปโหลดรูปจากเครื่อง',
      presetLabel: 'หรือเลือกรูปประกอบนักกายภาพสำเร็จรูป:',
      fullNameLabel: '2. ชื่อ - นามสกุล',
      phoneLabel: 'เบอร์โทรศัพท์ติดต่อ',
      specialtyLabel: 'ความเชี่ยวชาญเฉพาะทาง',
      licenseLabel: 'เลขที่ใบประกอบวิชาชีพ',
      bioLabel: 'ประวัติส่วนตัวและประสบการณ์การทำงาน',
      bioPlaceholder: 'ระบุประวัติการศึกษา ประสบการณ์ด้านกายภาพบำบัด หรือข้อมูลแนะนำตัวสำหรับคนไข้...',
      saveBtn: 'บันทึกประวัติส่วนตัว',
      savingBtn: 'กำลังบันทึก...',
      patientsCountLabel: 'คนไข้ที่ดูแลอยู่',
      savedSuccess: 'บันทึกประวัติส่วนตัวและรูปประกอบเรียบร้อยแล้ว',
      noPhone: 'ยังไม่ระบุเบอร์โทร',
    },
    sidebar: {
      adminRole: 'ผู้อำนวยการ (Admin)',
      ptRole: 'นักกายภาพบำบัด (PT)',
      collapseTitle: 'ย่อ/ขยายเมนู',
      dbCloud: 'ฐานข้อมูล (Cloud)',
      dbNormal: 'ออนไลน์',
      dbOffline: 'ออฟไลน์',
    },
    actions: {
      welcomeScan: 'ต้อนรับ & สแกนหน้า',
      quickCheckin: 'เช็คอินเร่งด่วน',
      kioskBtn: 'หน้าตู้ Kiosk',
      returnKiosk: '← หน้าตู้ Kiosk',
      logout: 'ออกจากระบบ',
      cardGame: 'การ์ดกายภาพ (10 ข้อ)',
      cloudConnected: 'Online (เชื่อมต่อระบบแล้ว)',
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
      updateFaceBtn: 'สแกนอัปเดตใบหน้าใหม่',
      historyBtn: 'ดูประวัติการฝึก',
      deleteTitle: 'ลบประวัติ (เฉพาะ Admin)',
      emptyPatients: 'ไม่พบข้อมูลคนไข้ กด "+ ลงทะเบียนคนไข้ใหม่" เพื่อเพิ่มผู้ป่วย',
      ageUnit: 'ปี',
    },
    pts: {
      title: 'ทำเนียบนักกายภาพบำบัด (PT Directory)',
      subtitle: 'รายชื่อนักกายภาพบำบัดประจำโรงพยาบาล เลขที่ใบอนุญาตประกอบวิชาชีพ และคนไข้ที่ดูแล',
      searchPlaceholder: 'ค้นหาชื่อ PT, เลขที่ใบอนุญาต, ความเชี่ยวชาญ...',
      licenseLabel: 'ใบประกอบวิชาชีพ:',
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
      testAlertBtn: 'ทดสอบแจ้งเตือนฉุกเฉิน',
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
      statusResolved: 'แก้ไขแล้ว',
      statusAck: 'กำลังตรวจสอบ',
      statusOpen: 'รอเจ้าหน้าที่',
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
      assessments: 'Clinical Assessment (0-6)',
      pts: '3. PT Directory',
      exercises: '4. Exercises & Modpacks',
      schedules: '5. Rehab Schedules',
      security: '6. Audit & Security',
      profile: '7. Personal Profile',
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
      assessments: {
        title: 'Individual Clinical Assessment (0-6 Scale) & ROM Tracking',
        subtitle: 'LONGITUDINAL ROM PROGRESSION & CROSS-FACILITY CLINICAL ASSESSMENTS',
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
      profile: {
        title: 'Physiotherapist Personal Profile',
        subtitle: 'PHYSIOTHERAPIST PERSONAL PROFILE & CREDENTIALS',
      },
    },
    profile: {
      sidebarCardTitle: 'My Profile',
      sidebarEditHint: 'Edit Profile',
      previewTitle: 'Clinician Profile Card',
      editFormTitle: 'Edit Personal Profile',
      avatarSectionLabel: '1. Profile Illustration / Photo',
      uploadPhotoBtn: 'Upload Photo',
      presetLabel: 'Or choose a clinician avatar preset:',
      fullNameLabel: '2. Full Name',
      phoneLabel: 'Phone Number',
      specialtyLabel: 'Clinical Specialty',
      licenseLabel: 'License No.',
      bioLabel: 'Personal Bio & Clinical Background',
      bioPlaceholder: 'Enter your education, clinical rehab experience, or greeting for patients...',
      saveBtn: 'Save Personal Profile',
      savingBtn: 'Saving...',
      patientsCountLabel: 'Assigned Patients',
      savedSuccess: 'Personal profile and avatar saved successfully',
      noPhone: 'Phone not set',
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
      cloudConnected: 'Online (Connected)',
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
      updateFaceBtn: 'Re-enroll Face',
      historyBtn: 'History',
      deleteTitle: 'Delete Record (Admin Only)',
      emptyPatients: 'No patients found. Click "+ Enroll New Patient" to register.',
      ageUnit: 'yrs',
    },
    pts: {
      title: 'Physiotherapist Directory',
      subtitle: 'Registered hospital physiotherapists, medical license numbers, and assigned patients',
      searchPlaceholder: 'Search PT name, license no., specialty...',
      licenseLabel: 'License No:',
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
      testAlertBtn: 'Test Emergency Alert',
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
      statusResolved: 'Resolved',
      statusAck: 'Checking',
      statusOpen: 'Waiting Staff',
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
      assessments: '个人康复评估档案 (0-6分)',
      pts: '3. 康复师团队',
      exercises: '4. 康复动作与题库',
      schedules: '5. 康复日程安排',
      security: '6. 审计与安全中心',
      profile: '7. 个人简介',
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
      assessments: {
        title: '个人临床康复评估档案 (0-6级) 与跨院区随访记录',
        subtitle: 'LONGITUDINAL ROM PROGRESSION & CROSS-FACILITY CLINICAL ASSESSMENTS',
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
      profile: {
        title: '物理治疗师个人简介与资质档案',
        subtitle: 'PHYSIOTHERAPIST PERSONAL PROFILE & CREDENTIALS',
      },
    },
    profile: {
      sidebarCardTitle: '个人简介',
      sidebarEditHint: '编辑资料',
      previewTitle: '个人档案预览',
      editFormTitle: '编辑个人资料与头像',
      avatarSectionLabel: '1. 个人头像 / 肖像插图',
      uploadPhotoBtn: '上传照片',
      presetLabel: '或选择康复师默认插画形象:',
      fullNameLabel: '2. 姓名',
      phoneLabel: '联系电话',
      specialtyLabel: '临床专长',
      licenseLabel: '执业证号',
      bioLabel: '个人简介与从业经历',
      bioPlaceholder: '填写教育背景、物理治疗临床经验或患者寄语...',
      saveBtn: '保存个人简介',
      savingBtn: '保存中...',
      patientsCountLabel: '负责患者数',
      savedSuccess: '个人简介与头像已保存',
      noPhone: '暂未填写电话',
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
      cloudConnected: 'Online (已连接云端)',
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
      updateFaceBtn: '更新人脸',
      historyBtn: '训练记录',
      deleteTitle: '删除病历 (仅管理员)',
      emptyPatients: '暂无患者记录，请点击“+ 注册新患者”新增档案',
      ageUnit: '岁',
    },
    pts: {
      title: '物理治疗师团队名录 (PT Directory)',
      subtitle: '在院注册物理治疗师名单、执业资格证号与负责患者数量统计',
      searchPlaceholder: '搜索康复师姓名、执业证号、专业方向...',
      licenseLabel: '执业证号:',
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
      testAlertBtn: '测试紧急呼叫警报',
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
      statusResolved: '已完成处理',
      statusAck: '核查处理中',
      statusOpen: '待响应',
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

function getPatientBiometricStatusDetail(
  p: PatientRecord,
): 'VERIFIED' | 'PENDING' | 'REENROLL_REQUIRED' {
  if (p.biometricStatusDetail) return p.biometricStatusDetail;
  const dim =
    typeof p.biometricEmbeddingDim === 'number' && p.biometricEmbeddingDim > 0
      ? p.biometricEmbeddingDim
      : Array.isArray(p.masterEmbedding)
        ? p.masterEmbedding.length
        : Array.isArray(p.biometricHistory?.overall_centroid)
          ? p.biometricHistory.overall_centroid.length
          : Array.isArray(p.biometricHistory?.frontal_centroid)
            ? p.biometricHistory.frontal_centroid.length
            : 0;
  if (dim < 128) return 'PENDING';
  if (dim < 1024 || (p.biometricEngine && p.biometricEngine !== 'human-faceres-1024d')) {
    return 'REENROLL_REQUIRED';
  }
  return 'VERIFIED';
}

function isPatientFaceVerified(p: PatientRecord): boolean {
  return getPatientBiometricStatusDetail(p) === 'VERIFIED';
}

let sharedEmergencyAudioCtx: AudioContext | null = null;

function playEmergencyChime() {
  try {
    const AudioCtx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
    if (!AudioCtx) return;
    if (!sharedEmergencyAudioCtx || sharedEmergencyAudioCtx.state === 'closed') {
      sharedEmergencyAudioCtx = new AudioCtx();
    }
    const ctx = sharedEmergencyAudioCtx;
    if (ctx.state === 'suspended') {
      void ctx.resume().catch(() => { });
    }
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

const STAFF_PROFILE_STORAGE_KEY = 'drtechcares_staff_profiles_v1';

interface StoredStaffProfileOverride {
  avatarUrl?: string;
  fullName?: string;
  phone?: string;
  specialty?: string;
  licenseNo?: string;
  bio?: string;
}

function makeClinicianAvatarSvg(opts: {
  bgFrom: string;
  bgTo: string;
  skin: string;
  hair: string;
  coat: string;
  shirt: string;
  glasses?: boolean;
  longHair?: boolean;
}): string {
  const hairBack = opts.longHair
    ? `<path d="M56,64 C52,95 54,132 68,144 L132,144 C146,132 148,95 144,64 C138,34 62,34 56,64 Z" fill="${opts.hair}"/>`
    : '';
  const glassesSvg = opts.glasses
    ? `<rect x="74" y="75" width="21" height="14" rx="5" fill="none" stroke="#1E293B" stroke-width="2.5"/>
       <rect x="105" y="75" width="21" height="14" rx="5" fill="none" stroke="#1E293B" stroke-width="2.5"/>
       <line x1="95" y1="81" x2="105" y2="81" stroke="#1E293B" stroke-width="2.5"/>`
    : '';
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200" width="200" height="200">
    <defs>
      <linearGradient id="g" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stop-color="${opts.bgFrom}"/>
        <stop offset="100%" stop-color="${opts.bgTo}"/>
      </linearGradient>
    </defs>
    <rect width="200" height="200" rx="36" fill="url(#g)"/>
    <circle cx="34" cy="34" r="28" fill="#FFFFFF" opacity="0.16"/>
    <circle cx="172" cy="168" r="36" fill="#FFFFFF" opacity="0.14"/>
    ${hairBack}
    <path d="M36,200 C38,148 64,132 100,132 C136,132 162,148 164,200 Z" fill="${opts.coat}"/>
    <polygon points="84,134 116,134 100,174" fill="${opts.shirt}"/>
    <path d="M74,138 C70,164 84,182 100,182 C116,182 130,164 126,138" fill="none" stroke="#334155" stroke-width="4" stroke-linecap="round"/>
    <circle cx="100" cy="182" r="7" fill="#0F172A" stroke="#CBD5E1" stroke-width="2.5"/>
    <rect x="88" y="108" width="24" height="26" rx="10" fill="${opts.skin}"/>
    <circle cx="100" cy="82" r="34" fill="${opts.skin}"/>
    <path d="M65,78 C64,44 86,34 102,34 C122,34 136,48 135,78 C128,60 112,56 100,56 C86,56 72,62 65,78 Z" fill="${opts.hair}"/>
    <circle cx="85" cy="82" r="3.5" fill="#0F172A"/>
    <circle cx="115" cy="82" r="3.5" fill="#0F172A"/>
    <path d="M90,97 Q100,105 110,97" fill="none" stroke="#9A3412" stroke-width="3" stroke-linecap="round"/>
    ${glassesSvg}
  </svg>`;
  return `data:image/svg+xml;utf8,${encodeURIComponent(svg)}`;
}

const PT_PRESET_AVATARS: readonly { id: string; label: string; uri: string }[] = [
  {
    id: 'pt-male',
    label: 'นักกายภาพ (ชาย)',
    uri: makeClinicianAvatarSvg({
      bgFrom: '#D1FAE5',
      bgTo: '#A7F3D0',
      skin: '#FCD5B5',
      hair: '#1E293B',
      coat: '#FFFFFF',
      shirt: '#0D9488',
    }),
  },
  {
    id: 'pt-female',
    label: 'นักกายภาพ (หญิง)',
    uri: makeClinicianAvatarSvg({
      bgFrom: '#E0F2FE',
      bgTo: '#BAE6FD',
      skin: '#FDE2C8',
      hair: '#3F271E',
      coat: '#FFFFFF',
      shirt: '#0284C7',
      longHair: true,
    }),
  },
  {
    id: 'pt-senior',
    label: 'ผู้เชี่ยวชาญ (สวมแว่น)',
    uri: makeClinicianAvatarSvg({
      bgFrom: '#EDE9FE',
      bgTo: '#DDD6FE',
      skin: '#FCD5B5',
      hair: '#475569',
      coat: '#FFFFFF',
      shirt: '#4F46E5',
      glasses: true,
    }),
  },
  {
    id: 'pt-director',
    label: 'หัวหน้าคลินิก',
    uri: makeClinicianAvatarSvg({
      bgFrom: '#FEF3C7',
      bgTo: '#FDE68A',
      skin: '#FDE2C8',
      hair: '#1F2937',
      coat: '#F8FAFC',
      shirt: '#059669',
      glasses: true,
      longHair: true,
    }),
  },
];

function loadStaffProfileOverrides(): Record<string, StoredStaffProfileOverride> {
  try {
    const raw = window.localStorage.getItem(STAFF_PROFILE_STORAGE_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw) as unknown;
    return parsed && typeof parsed === 'object' ? (parsed as Record<string, StoredStaffProfileOverride>) : {};
  } catch {
    return {};
  }
}

function saveStaffProfileOverrides(map: Record<string, StoredStaffProfileOverride>): void {
  try {
    window.localStorage.setItem(STAFF_PROFILE_STORAGE_KEY, JSON.stringify(map));
  } catch {
    // ignore storage quota errors
  }
}

const CASE_DELEGATION_STORAGE_KEY = 'drtechcares_case_delegations_v1';

interface StoredCaseDelegationOverride {
  delegatedToPtId: string | null;
  delegationDate: string | null;
  delegationNote: string | null;
}

function loadCaseDelegationOverrides(): Record<string, StoredCaseDelegationOverride> {
  try {
    const raw = window.localStorage.getItem(CASE_DELEGATION_STORAGE_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw) as unknown;
    return parsed && typeof parsed === 'object' ? (parsed as Record<string, StoredCaseDelegationOverride>) : {};
  } catch {
    return {};
  }
}

function saveCaseDelegationOverrides(map: Record<string, StoredCaseDelegationOverride>): void {
  try {
    window.localStorage.setItem(CASE_DELEGATION_STORAGE_KEY, JSON.stringify(map));
  } catch {
    // ignore storage quota errors
  }
}

function getDefaultBioForUser(u: UserAccount): string {
  if (u.role === 'super_admin') {
    return 'ผู้บริหารและกำกับดูแลระบบฟื้นฟูสมรรถภาพ Dr.Tech.Care ดูแลมาตรฐานคลินิก ความปลอดภัยข้อมูลผู้ป่วย และการประสานงานทีมสหวิชาชีพ';
  }
  const spec = u.specialty || 'กายภาพบำบัดระบบกระดูก กล้ามเนื้อ และการฟื้นฟูสมอง';
  return `นักกายภาพบำบัดวิชาชีพ (ใบอนุญาต ${u.licenseNo || 'PT-44012'}) เชี่ยวชาญด้าน${spec} ให้คำปรึกษาและติดตามการฟื้นฟูผู้ป่วยอย่างใกล้ชิด`;
}

function getDefaultAvatarForUser(u: UserAccount): string {
  const name = u.fullName || u.username;
  if (/วรัญญา|สุทธิดา|วิภาวี|หญิง|waranya|suthida|wipawee/i.test(name)) {
    return PT_PRESET_AVATARS[1].uri;
  }
  if (u.role === 'super_admin') {
    return PT_PRESET_AVATARS[2].uri;
  }
  return PT_PRESET_AVATARS[0].uri;
}

export function resolveUserProfile(
  u: UserAccount,
  overrides: Record<string, StoredStaffProfileOverride>,
): {
  fullName: string;
  phone: string;
  specialty: string;
  licenseNo: string;
  avatarUrl: string;
  bio: string;
} {
  const local = overrides[u.id] ?? overrides[`usr:${u.username.toLowerCase()}`];
  const serverAvatar =
    typeof u.avatarUrl === 'string' && u.avatarUrl.trim() ? u.avatarUrl.trim() : '';
  const localAvatar =
    typeof local?.avatarUrl === 'string' && local.avatarUrl.trim() ? local.avatarUrl.trim() : '';
  const serverBio = typeof u.bio === 'string' && u.bio.trim() ? u.bio.trim() : '';
  const localBio = typeof local?.bio === 'string' && local.bio.trim() ? local.bio.trim() : '';
  return {
    fullName: u.fullName?.trim() || local?.fullName?.trim() || u.username,
    phone: u.phone !== undefined && u.phone !== '' ? u.phone : local?.phone ?? '',
    specialty: u.specialty?.trim() || local?.specialty?.trim() || 'เวชศาสตร์ฟื้นฟูกายภาพบำบัด',
    licenseNo: u.licenseNo || local?.licenseNo || 'PT-44012',
    // Authoritative cloud avatarUrl always takes precedence over stale per-device localStorage overrides
    avatarUrl: serverAvatar || localAvatar || getDefaultAvatarForUser(u),
    bio: serverBio || localBio || getDefaultBioForUser(u),
  };
}

export function reconcileStaffProfileOverridesFromServer(
  prevOverrides: Record<string, StoredStaffProfileOverride>,
  serverUsers: readonly UserAccount[],
): Record<string, StoredStaffProfileOverride> {
  if (!serverUsers || serverUsers.length === 0) return prevOverrides;
  let changed = false;
  const next: Record<string, StoredStaffProfileOverride> = { ...prevOverrides };
  for (const u of serverUsers) {
    const idKey = u.id;
    const usrKey = `usr:${u.username.toLowerCase()}`;
    const existing = next[idKey] ?? next[usrKey];
    const serverAvatar =
      typeof u.avatarUrl === 'string' && u.avatarUrl.trim() ? u.avatarUrl.trim() : undefined;
    const serverBio = typeof u.bio === 'string' && u.bio.trim() ? u.bio.trim() : undefined;
    if (!existing && !serverAvatar && !serverBio) continue;
    const merged: StoredStaffProfileOverride = {
      ...(existing ?? {}),
      fullName: u.fullName || existing?.fullName,
      phone: u.phone !== undefined ? u.phone : existing?.phone,
      specialty: u.specialty || existing?.specialty,
      licenseNo: u.licenseNo || existing?.licenseNo,
      ...(serverAvatar ? { avatarUrl: serverAvatar } : existing?.avatarUrl ? { avatarUrl: existing.avatarUrl } : {}),
      ...(serverBio ? { bio: serverBio } : existing?.bio ? { bio: existing.bio } : {}),
    };
    const prevIdEntry = next[idKey];
    if (
      !prevIdEntry ||
      prevIdEntry.avatarUrl !== merged.avatarUrl ||
      prevIdEntry.bio !== merged.bio ||
      prevIdEntry.fullName !== merged.fullName ||
      prevIdEntry.phone !== merged.phone ||
      prevIdEntry.specialty !== merged.specialty ||
      prevIdEntry.licenseNo !== merged.licenseNo
    ) {
      next[idKey] = merged;
      next[usrKey] = merged;
      changed = true;
    }
  }
  return changed ? next : prevOverrides;
}

async function resizeImageFileToDataUri(file: File, maxDim = 320): Promise<string> {
  const rawUrl = await new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result ?? ''));
    reader.onerror = () => reject(new Error('ไม่สามารถอ่านไฟล์รูปภาพได้'));
    reader.readAsDataURL(file);
  });
  const img = await new Promise<HTMLImageElement>((resolve, reject) => {
    const el = new Image();
    el.onload = () => resolve(el);
    el.onerror = () => reject(new Error('ไฟล์รูปภาพไม่ถูกต้อง'));
    el.src = rawUrl;
  });
  const scale = Math.min(1, maxDim / Math.max(img.width || maxDim, img.height || maxDim));
  const w = Math.max(1, Math.round((img.width || maxDim) * scale));
  const h = Math.max(1, Math.round((img.height || maxDim) * scale));
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');
  if (!ctx) return rawUrl;
  ctx.drawImage(img, 0, 0, w, h);
  return canvas.toDataURL('image/jpeg', 0.85);
}

function sortNewestFirst<T extends { createdAt: string }>(items: readonly T[]): T[] {
  return [...items].sort((a, b) => (Date.parse(b.createdAt) || 0) - (Date.parse(a.createdAt) || 0));
}

const LEGACY_EXERCISE_STORAGE_KEYS = [
  'drtech_exercises_cache',
  'drtech_rehab_exercises_v1',
  'exercises',
  'drtech_exercises',
] as const;

/**
 * Purges deprecated legacy/lower-limb/ortho exercises from localStorage & Supabase
 * and ensures all 26 standardized Stroke ADL & MOPH upper-body postures are synced cleanly.
 */
export async function syncStrokeExerciseLibrary(
  apiClient: DrTechCaresApiClient,
): Promise<{ exercises: ManagedExerciseRecord[]; purgedCount: number }> {
  if (typeof globalThis.localStorage !== 'undefined') {
    for (const storageKey of LEGACY_EXERCISE_STORAGE_KEYS) {
      try {
        globalThis.localStorage.removeItem(storageKey);
      } catch {
        // Ignore storage errors in SSR/restricted browsers
      }
    }
  }

  let purgedCount = 0;
  let serverExercises: ManagedExerciseRecord[] = [];
  if (typeof (apiClient as Partial<DrTechCaresApiClient>).syncStrokeExerciseLibrary === 'function') {
    try {
      const synced = await apiClient.syncStrokeExerciseLibrary();
      purgedCount += synced.purgedCount ?? 0;
      serverExercises = Array.isArray(synced.exercises) ? synced.exercises : [];
    } catch {
      // Fallback safely to canonical 26 list + local clean filter when offline or unauthenticated
    }
  }

  const canonicalFallback: ManagedExerciseRecord[] = CLINICAL_EXERCISES.map((ex) => ({
    ...ex,
    createdBy: 'system',
  }));
  const cleanCustom = serverExercises.filter(
    (ex) =>
      !isDeprecatedExerciseRecord(ex) &&
      !STROKE_CANONICAL_EXERCISE_SET.has(ex.code) &&
      !STROKE_CANONICAL_EXERCISE_SET.has(ex.id),
  );
  return {
    exercises: mergeDuplicateSideExercises(
      sortExercisesStrokeUpperFirst([...canonicalFallback, ...cleanCustom]),
    ),
    purgedCount,
  };
}

export function ClinicalStaffPortalShell({
  apiClient,
  currentUser,
  onLogout,
  onReturnToKiosk,
  onOpenPoseTestMode,
  initialTab,
  autoOpenEnroll,
  onAutoOpenEnrollConsumed,
  initialPatients,
  initialUsers,
  initialPatientScope,
  initialPatientSyncError,
}: ClinicalStaffPortalShellProps) {
  const { lang, setLang } = useClinicalI18n();
  const locale = PORTAL_TEXT[lang];
  const isAdminUser =
    isAdminRole(currentUser?.role) ||
    currentUser?.role?.toUpperCase() === 'ADMIN' ||
    currentUser?.role?.toUpperCase() === 'SUPER_ADMIN' ||
    (currentUser?.role as string) === 'ผู้ดูแลระบบ';
  const isAdmin = isAdminUser;

  const [activeTab, setActiveTab] = useState<PortalTab>(() => initialTab ?? (isAdmin ? 'staff' : 'patients'));
  const [selectedAssessmentPatientId, setSelectedAssessmentPatientId] = useState<string | null>(null);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  const [profileOverrides, setProfileOverrides] = useState<Record<string, StoredStaffProfileOverride>>(
    () => loadStaffProfileOverrides(),
  );
  const profileDirtyRef = useRef(false);

  // Cloud Worker HTTP API + DB health state (derived purely from HTTP 200 responses & navigator.onLine, never WebSocket Realtime)
  const [cloudHealth, setCloudHealth] = useState<{
    online: boolean;
    dbConnected: boolean;
    latencyMs?: number;
  }>({
    online: typeof navigator === 'undefined' ? true : navigator.onLine !== false,
    dbConnected: typeof navigator === 'undefined' ? true : navigator.onLine !== false,
  });

  // Core data state
  const [users, setUsers] = useState<UserAccount[]>(() => (initialUsers ? [...initialUsers] : []));
  const [rawPatients, setRawPatients] = useState<PatientRecord[]>(() =>
    initialPatients
      ? initialPatients.map((p) => normalizePatientRecord(p, { requireRealVectorsForVerified: false }))
      : [],
  );
  const { patients } = useMemo(
    () => unifyClinicPatientsDetailed(rawPatients, users),
    [rawPatients, users],
  );
  const [patientsSyncing, setPatientsSyncing] = useState(false);
  const [patientSyncError, setPatientSyncError] = useState<string | null>(initialPatientSyncError ?? null);
  const [exercises, setExercises] = useState<ManagedExerciseRecord[]>(() =>
    CLINICAL_EXERCISES.map((ex) => ({ ...ex, createdBy: 'system' })),
  );
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
  const [poseTestOverlayExerciseCode, setPoseTestOverlayExerciseCode] = useState<string | null | undefined>(
    undefined,
  );
  const handleLaunchPoseTest = (initialExerciseCode?: string) => {
    if (onOpenPoseTestMode) {
      onOpenPoseTestMode(initialExerciseCode);
      return;
    }
    setPoseTestOverlayExerciseCode(initialExerciseCode ?? null);
  };
  const [patientModalState, setPatientModalState] = useState<{
    patientId: string | null;
    tab: 'profile' | 'history';
  }>({ patientId: null, tab: 'profile' });
  const [ehrLinkPatient, setEhrLinkPatient] = useState<PatientRecord | null>(null);
  const [selectedPtDetailModal, setSelectedPtDetailModal] = useState<UserAccount | null>(null);
  const [editingStaffUser, setEditingStaffUser] = useState<UserAccount | null>(null);
  const [editStaffFullName, setEditStaffFullName] = useState('');
  const [editStaffPhone, setEditStaffPhone] = useState('');
  const [editStaffSpecialty, setEditStaffSpecialty] = useState('');
  const [editStaffLicenseNo, setEditStaffLicenseNo] = useState('');
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
  const [patientScopeFilter, setPatientScopeFilter] = useState<PatientScopeFilter>(
    initialPatientScope ?? 'all',
  );

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
  const [callKioskId, setCallKioskId] = useState('KIOSK-OPD-01');
  const [callPatientId, setCallPatientId] = useState('');
  const [callReason, setCallReason] = useState('');
  const [creatingCall, setCreatingCall] = useState(false);

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

  const activeUserRecord = useMemo(
    () =>
      users.find(
        (u) => u.id === currentUser.id || u.username.toLowerCase() === currentUser.username.toLowerCase(),
      ) ?? currentUser,
    [users, currentUser],
  );
  const myProfile = useMemo(
    () => resolveUserProfile(activeUserRecord, profileOverrides),
    [activeUserRecord, profileOverrides],
  );

  const [profileFullName, setProfileFullName] = useState(myProfile.fullName);
  const [profilePhone, setProfilePhone] = useState(myProfile.phone);
  const [profileSpecialty, setProfileSpecialty] = useState(myProfile.specialty);
  const [profileLicenseNo, setProfileLicenseNo] = useState(myProfile.licenseNo);
  const [profileAvatarUrl, setProfileAvatarUrl] = useState(myProfile.avatarUrl);
  const [profileBio, setProfileBio] = useState(myProfile.bio);

  // Patient Case Delegation form state (Tab 7: ระบบการฝากเคสคนไข้)
  const [delegationPatientId, setDelegationPatientId] = useState('');
  const [delegationDate, setDelegationDate] = useState(() => formatLocalIsoDate());
  const [delegationTargetPtId, setDelegationTargetPtId] = useState('');
  const [delegationNote, setDelegationNote] = useState('');

  useEffect(() => {
    if (profileDirtyRef.current) return;
    setProfileFullName(myProfile.fullName);
    setProfilePhone(myProfile.phone);
    setProfileSpecialty(myProfile.specialty);
    setProfileLicenseNo(myProfile.licenseNo);
    setProfileAvatarUrl(myProfile.avatarUrl);
    setProfileBio(myProfile.bio);
  }, [myProfile]);

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

  const bannerTimerRef = useRef<number | null>(null);
  const thresholdTimerRef = useRef<number | null>(null);

  useEffect(() => {
    return () => {
      if (bannerTimerRef.current !== null) window.clearTimeout(bannerTimerRef.current);
      if (thresholdTimerRef.current !== null) window.clearTimeout(thresholdTimerRef.current);
    };
  }, []);

  const flashBanner = useCallback((kind: 'success' | 'error', text: string) => {
    setBannerMessage({ kind, text });
    if (bannerTimerRef.current !== null) window.clearTimeout(bannerTimerRef.current);
    bannerTimerRef.current = window.setTimeout(() => {
      bannerTimerRef.current = null;
      setBannerMessage((prev) => (prev?.text === text ? null : prev));
    }, 4500);
  }, []);

  const pingCloudHealth = useCallback(
    async (forceRefresh = false) => {
      const browserOnline = typeof navigator === 'undefined' ? true : navigator.onLine !== false;
      const h = await apiClient.checkHealth({ refresh: forceRefresh });
      const workerOnline = browserOnline && h.ok;
      const dbConnected = workerOnline && h.supabaseConnected;
      setCloudHealth({ online: workerOnline, dbConnected, latencyMs: h.latencyMs });
    },
    [apiClient],
  );

  const applyPatientDelegationOverrides = useCallback((rows: PatientRecord[]): PatientRecord[] => {
    const localDelegations = loadCaseDelegationOverrides();
    return rows.map((p) => {
      const ov = localDelegations[p.id];
      if (ov && ov.delegatedToPtId) {
        return {
          ...p,
          delegatedToPtId: p.delegatedToPtId || ov.delegatedToPtId,
          delegationDate: p.delegationDate || ov.delegationDate,
          delegationNote: p.delegationNote ?? ov.delegationNote,
        };
      }
      return p;
    });
  }, []);

  const syncPatientsFromSupabase = useCallback(
    async (options?: { showSuccessToast?: boolean; silent?: boolean }) => {
      setPatientsSyncing(true);
      try {
        const res = await fetchAllPatientsUnified(apiClient, { forceRefresh: true });
        if (res.authError) {
          if (await handleAuthFailureIfExpired(res.authError)) return res;
        }
        if (res.patients.length > 0) {
          const merged = applyPatientDelegationOverrides(res.patients);
          setRawPatients(merged);
          setSelectedSchedulePatientId((prev) => prev || merged[0].id);
          setPatientSyncError(res.errorMessage);
        } else if (res.errorMessage) {
          setPatientSyncError(res.errorMessage);
        } else if (res.syncedFromSupabase) {
          setRawPatients([]);
          setPatientSyncError(null);
        }
        if (options?.showSuccessToast) {
          const unifiedCount = unifyClinicPatientsDetailed(
            res.patients.length > 0 ? res.patients : rawPatients,
            users,
          ).patients.length;
          if (res.errorMessage && res.patients.length === 0) {
            flashBanner('error', `ไม่สามารถซิงก์ข้อมูลจาก Supabase ได้: ${res.errorMessage}`);
          } else {
            flashBanner('success', `ซิงก์ข้อมูลจาก Supabase สำเร็จ: ${unifiedCount} บัญชี`);
          }
        }
        return res;
      } finally {
        setPatientsSyncing(false);
      }
    },
    [apiClient, applyPatientDelegationOverrides, flashBanner, handleAuthFailureIfExpired, rawPatients, users],
  );

  const refreshAll = useCallback(async () => {
    setLoading(true);
    try {
      let listUsersError: unknown = null;
      const [uRes, patientSyncRes, exRes, mpRes, sessRes, callsRes, settingsRes, secRes, healthRes] = await Promise.all([
        apiClient.listUsers({ forceRefresh: true }).catch((err) => {
          listUsersError = err;
          return null;
        }),
        fetchAllPatientsUnified(apiClient, { forceRefresh: true }),
        apiClient.listExercises().catch(() => []),
        apiClient.listModpacks().catch(() => []),
        apiClient.listRehabSessions().catch(() => []),
        apiClient.listStaffCalls().catch(() => []),
        apiClient.getSystemSettings().catch(() => null),
        isAdmin ? apiClient.listSecurityAudit().catch(() => null) : Promise.resolve(null),
        apiClient
          .checkHealth()
          .catch(() => ({ ok: false, persistentDb: false, supabaseConnected: false, latencyMs: 0 })),
      ]);

      const patientsHttpOk = patientSyncRes.syncedFromSupabase;

      if (listUsersError) {
        if (await handleAuthFailureIfExpired(listUsersError)) return;
        flashBanner(
          'error',
          listUsersError instanceof Error ? listUsersError.message : 'ไม่สามารถโหลดข้อมูลบุคลากรได้',
        );
      }
      if (patientSyncRes.authError) {
        if (await handleAuthFailureIfExpired(patientSyncRes.authError)) return;
      }

      const nextUsers = uRes && uRes.length > 0 ? uRes : uRes ? [currentUser] : null;
      if (nextUsers) {
        setUsers(nextUsers);
        setProfileOverrides((prev) => {
          const reconciled = reconcileStaffProfileOverridesFromServer(prev, nextUsers);
          if (reconciled !== prev) {
            saveStaffProfileOverrides(reconciled);
          }
          return reconciled;
        });
      } else {
        setUsers((prev) => (prev.length > 0 ? prev : [currentUser]));
      }

      if (patientSyncRes.patients.length > 0) {
        const mergedPatients = applyPatientDelegationOverrides(patientSyncRes.patients);
        setRawPatients(mergedPatients);
        setPatientSyncError(patientSyncRes.errorMessage);
        setSelectedSchedulePatientId((prev) => prev || mergedPatients[0].id);
      } else if (patientSyncRes.errorMessage) {
        setPatientSyncError(patientSyncRes.errorMessage);
      } else if (patientSyncRes.syncedFromSupabase) {
        setRawPatients([]);
        setPatientSyncError(null);
      }
      if (nextUsers && patientSyncRes.patients.length === 0) {
        const mirroredOnly = unifyClinicPatientsDetailed([], nextUsers).patients;
        if (mirroredOnly.length > 0) {
          setSelectedSchedulePatientId((prev) => prev || mirroredOnly[0].id);
        }
      }

      const unifiedEx = mergeDuplicateSideExercises(
        sortExercisesStrokeUpperFirst(exRes.filter((ex) => !isDeprecatedExerciseRecord(ex))),
      );
      setExercises(unifiedEx);
      const needsStrokeSync =
        exRes.some((ex) => isDeprecatedExerciseRecord(ex)) ||
        unifiedEx.filter((ex) => STROKE_CANONICAL_EXERCISE_SET.has(ex.code)).length < CLINICAL_EXERCISES.length;
      if (needsStrokeSync) {
        void syncStrokeExerciseLibrary(apiClient).then((synced) => {
          setExercises(synced.exercises);
        });
      }
      setModpacks(mpRes);
      setSessions(sortNewestFirst(sessRes));
      setStaffCalls(sortNewestFirst(callsRes));
      const browserOnline = typeof navigator === 'undefined' ? true : navigator.onLine !== false;
      const isCloudOnline = browserOnline && (healthRes.ok || uRes !== null || patientsHttpOk);
      const dbConnected =
        browserOnline && (healthRes.ok ? healthRes.supabaseConnected : uRes !== null || patientsHttpOk);
      setCloudHealth({ online: isCloudOnline, dbConnected, latencyMs: healthRes.latencyMs });

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
        setAuditLogs(sortNewestFirst(secRes.logs));
        setBannedTargets(sortNewestFirst(secRes.bannedTargets));
        setActiveSessions(sortNewestFirst(secRes.activeSessions));
      }

      if (mpRes.length > 0) {
        setCourseModpackId((prev) => prev || mpRes[0].id);
      }
    } finally {
      setLoading(false);
    }
  }, [
    apiClient,
    isAdmin,
    currentUser,
    applyPatientDelegationOverrides,
    handleAuthFailureIfExpired,
    flashBanner,
  ]);

  const lastUsersCloudSyncAtRef = useRef(0);
  const syncStaffUsersFromCloud = useCallback(
    async (force = false) => {
      const now = Date.now();
      if (!force && now - lastUsersCloudSyncAtRef.current < 4000) return;
      lastUsersCloudSyncAtRef.current = now;
      try {
        const freshUsers = await apiClient.listUsers({ forceRefresh: true });
        if (freshUsers && freshUsers.length > 0) {
          setUsers(freshUsers);
          setProfileOverrides((prev) => {
            const reconciled = reconcileStaffProfileOverridesFromServer(prev, freshUsers);
            if (reconciled !== prev) {
              saveStaffProfileOverrides(reconciled);
            }
            return reconciled;
          });
        }
      } catch {
        // Silent background profile sync — keep existing state on transient errors
      }
    },
    [apiClient],
  );

  useEffect(() => {
    if (activeTab === 'patients' || activeTab === 'assessments' || activeTab === 'schedules') {
      void syncPatientsFromSupabase({ silent: true });
    }
    if (activeTab === 'profile' || activeTab === 'staff' || activeTab === 'pts') {
      void syncStaffUsersFromCloud(true);
    }
  }, [activeTab, syncPatientsFromSupabase, syncStaffUsersFromCloud]);

  useEffect(() => {
    refreshAll();
    const handleOnline = () => {
      void pingCloudHealth(true);
      void syncStaffUsersFromCloud(true);
    };
    const handleOffline = () => {
      setCloudHealth((prev) => ({ ...prev, online: false, dbConnected: false }));
    };
    const handleWindowFocusOrVisible = () => {
      if (typeof document !== 'undefined' && document.visibilityState === 'hidden') return;
      void syncStaffUsersFromCloud(false);
    };
    if (typeof window !== 'undefined') {
      window.addEventListener('online', handleOnline);
      window.addEventListener('offline', handleOffline);
      window.addEventListener('focus', handleWindowFocusOrVisible);
    }
    if (typeof document !== 'undefined') {
      document.addEventListener('visibilitychange', handleWindowFocusOrVisible);
    }
    const pollTimer = window.setInterval(() => {
      void pingCloudHealth();
      void syncStaffUsersFromCloud(false);
      apiClient
        .listStaffCalls()
        .then((calls) => {
          const nextOpen = calls.filter((c) => c.status === 'open').length;
          if (nextOpen > prevOpenCallCountRef.current) playEmergencyChime();
          prevOpenCallCountRef.current = nextOpen;
          setStaffCalls(sortNewestFirst(calls));
        })
        .catch(() => { });
    }, 30_000);

    return () => {
      window.clearInterval(pollTimer);
      if (typeof window !== 'undefined') {
        window.removeEventListener('online', handleOnline);
        window.removeEventListener('offline', handleOffline);
        window.removeEventListener('focus', handleWindowFocusOrVisible);
      }
      if (typeof document !== 'undefined') {
        document.removeEventListener('visibilitychange', handleWindowFocusOrVisible);
      }
    };
  }, [refreshAll, pingCloudHealth, syncStaffUsersFromCloud, apiClient]);

  const exerciseLibrarySyncAttemptedRef = useRef(false);
  useEffect(() => {
    if (activeTab !== 'exercises') return;
    if (exerciseLibrarySyncAttemptedRef.current) return;
    const needsSync =
      exercises.some((ex) => isDeprecatedExerciseRecord(ex)) ||
      mergeDuplicateSideExercises(exercises).filter((ex) => STROKE_CANONICAL_EXERCISE_SET.has(ex.code)).length <
      CLINICAL_EXERCISES.length;
    if (!needsSync) return;
    exerciseLibrarySyncAttemptedRef.current = true;
    let cancelled = false;
    void syncStrokeExerciseLibrary(apiClient).then((synced) => {
      if (!cancelled) {
        setExercises(synced.exercises);
      }
    });
    return () => {
      cancelled = true;
    };
  }, [activeTab, exercises, apiClient]);

  const defaultModpackId = modpacks[0]?.id ?? 'modpack-core-99';

  useEffect(() => {
    if (!selectedSchedulePatientId) {
      setPatientCourse(null);
      return;
    }
    let cancelled = false;
    apiClient
      .getPatientCourse(selectedSchedulePatientId)
      .then((course) => {
        if (cancelled) return;
        setPatientCourse(course);
        if (course) {
          setCourseTitle(course.title);
          setCourseModpackId(course.cognitiveModpackId || defaultModpackId);
          setWeeklySchedule(
            course.weeklySchedule.length === 7
              ? course.weeklySchedule
              : Array.from({ length: 7 }, (_, dayIndex) => ({
                dayIndex,
                dayType: 'rest',
                activityType: 'minigame',
                difficulty: 'easy',
                label: 'พักฟื้น',
                exercises: [],
                cognitiveModpackId: defaultModpackId,
              })),
          );
        }
      })
      .catch(() => {
        if (!cancelled) {
          setPatientCourse(null);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [selectedSchedulePatientId, apiClient, defaultModpackId]);

  const ptUsers = useMemo(() => users.filter((u) => u.role === 'pt'), [users]);
  const superAdminCount = useMemo(
    () => users.filter((acc) => acc.role === 'super_admin').length,
    [users],
  );
  const sortedMergedExercises = useMemo(
    () => mergeDuplicateSideExercises(sortExercisesStrokeUpperFirst(exercises)),
    [exercises],
  );
  const activeModalPatient = useMemo(
    () =>
      patientModalState.patientId
        ? patients.find((p) => p.id === patientModalState.patientId) ?? null
        : null,
    [patientModalState.patientId, patients],
  );
  const activeEhrLinkPatient = useMemo(
    () =>
      ehrLinkPatient
        ? patients.find((p) => p.id === ehrLinkPatient.id) ?? ehrLinkPatient
        : null,
    [ehrLinkPatient, patients],
  );
  const todayIsoStr = formatLocalIsoDate();

  useEffect(() => {
    if (!delegationPatientId && patients.length > 0) {
      setDelegationPatientId(patients[0].id);
    }
  }, [delegationPatientId, patients]);

  useEffect(() => {
    if (!delegationTargetPtId && ptUsers.length > 0) {
      const candidate = ptUsers.find((pt) => pt.id !== activeUserRecord.id) ?? ptUsers[0];
      setDelegationTargetPtId(candidate.id);
    }
  }, [delegationTargetPtId, ptUsers, activeUserRecord.id]);

  const delegatedCases = useMemo(
    () =>
      patients
        .filter((p) => Boolean(p.delegatedToPtId))
        .sort((a, b) => (b.delegationDate || '').localeCompare(a.delegationDate || '')),
    [patients],
  );
  const openStaffCalls = useMemo(() => staffCalls.filter((c) => c.status !== 'resolved'), [staffCalls]);

  const ptNameMap = useMemo(() => {
    const map = new Map<string, string>();
    users.forEach((u) => map.set(u.id, u.fullName));
    return map;
  }, [users]);

  const patientNameMap = useMemo(() => {
    const map = new Map<string, string>();
    patients.forEach((p) => {
      const label = `${p.fullName} (HN: ${p.hn})`;
      map.set(p.id, label);
      map.set(p.hn, label);
    });
    return map;
  }, [patients]);

  const submitStaffCall = useCallback(
    async (kioskId: string, reason: string, patientId?: string | null) => {
      setCreatingCall(true);
      try {
        const created = await apiClient.createStaffCall(
          kioskId || kiosks[0]?.kioskId || 'KIOSK-OPD-01',
          reason,
          patientId || null,
        );
        setSecuritySubTab('calls');
        playEmergencyChime();
        setStaffCalls((prev) => {
          const next = [created, ...prev.filter((c) => c.id !== created.id)];
          prevOpenCallCountRef.current = next.filter((c) => c.status === 'open').length;
          return next;
        });
        flashBanner('success', `ส่งคำเรียกเจ้าหน้าที่จาก ${created.kioskId} เรียบร้อยแล้ว`);
        void refreshAll();
        return true;
      } catch (err) {
        flashBanner(
          'error',
          err instanceof Error ? err.message : 'ไม่สามารถส่งคำเรียกเจ้าหน้าที่ได้',
        );
        return false;
      } finally {
        setCreatingCall(false);
      }
    },
    [apiClient, kiosks, flashBanner, refreshAll],
  );

  const unifiedStaffUsers = useMemo(
    () => mergePatientsIntoStaffUsersList(users, patients),
    [users, patients],
  );

  const filteredStaff = useMemo(() => {
    const q = staffSearch.trim().toLowerCase();
    const qNormHn = normalizeHnKey(q);
    return unifiedStaffUsers.filter((u) => {
      if (staffRoleFilter !== 'all' && u.role !== staffRoleFilter) return false;
      if (staffStatusFilter === 'active' && !u.active) return false;
      if (staffStatusFilter === 'inactive' && u.active) return false;
      if (!q) return true;
      const cleanUsr = u.username.replace(/^@+/, '').toLowerCase();
      return (
        u.fullName.toLowerCase().includes(q) ||
        cleanUsr.includes(q) ||
        `@${cleanUsr}`.includes(q) ||
        (Boolean(qNormHn) && normalizeHnKey(cleanUsr).includes(qNormHn)) ||
        (u.phone || '').toLowerCase().includes(q)
      );
    });
  }, [unifiedStaffUsers, staffSearch, staffRoleFilter, staffStatusFilter]);

  const assignedPatientsCount = useMemo(
    () =>
      patients.filter(
        (p) => p.assignedPtId === currentUser.id || p.delegatedToPtId === currentUser.id,
      ).length,
    [patients, currentUser.id],
  );

  const filteredPatients = useMemo(
    () =>
      filterAdminPortalPatients(patients, {
        scopeFilter: patientScopeFilter,
        faceFilter: patientFaceFilter,
        searchQuery: patientSearch,
        currentUserId: currentUser.id,
      }),
    [patients, patientScopeFilter, patientFaceFilter, patientSearch, currentUser.id],
  );

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
    const ordered = sortNewestFirst(auditLogs);
    if (!q) return ordered;
    return ordered.filter(
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

  const handleAvatarFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const dataUri = await resizeImageFileToDataUri(file, 320);
      profileDirtyRef.current = true;
      setProfileAvatarUrl(dataUri);
    } catch (err) {
      flashBanner('error', err instanceof Error ? err.message : 'ไม่สามารถอัปโหลดรูปภาพได้');
    }
  };

  const handleSavePersonalProfile = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!profileFullName.trim()) {
      flashBanner('error', 'กรุณาระบุชื่อ-นามสกุล');
      return;
    }
    const existingLicense = activeUserRecord.licenseNo ?? myProfile.licenseNo;
    // Strict RBAC Payload Guard: non-admin (PT) can never mutate licenseNo in State, localStorage, or DB.
    const safeLicenseNumber = isAdminUser ? profileLicenseNo.trim() : existingLicense;
    const patch: StoredStaffProfileOverride = {
      fullName: profileFullName.trim(),
      phone: profilePhone.trim(),
      specialty: profileSpecialty.trim(),
      licenseNo: safeLicenseNumber,
      avatarUrl: profileAvatarUrl,
      bio: profileBio.trim(),
    };
    const nextOverrides: Record<string, StoredStaffProfileOverride> = {
      ...profileOverrides,
      [activeUserRecord.id]: patch,
      [`usr:${activeUserRecord.username.toLowerCase()}`]: patch,
    };
    setProfileOverrides(nextOverrides);
    saveStaffProfileOverrides(nextOverrides);
    profileDirtyRef.current = false;
    if (!isAdminUser) {
      setProfileLicenseNo(safeLicenseNumber);
    }

    setSubmitting(true);
    try {
      const updated = await apiClient.updateUser(activeUserRecord.id, {
        fullName: patch.fullName,
        phone: patch.phone,
        specialty: activeUserRecord.role !== 'patient' ? patch.specialty : undefined,
        licenseNo:
          isAdminUser && activeUserRecord.role !== 'patient' ? safeLicenseNumber : undefined,
        avatarUrl: patch.avatarUrl,
        bio: patch.bio,
      });
      const syncedAvatarUrl = updated.avatarUrl || patch.avatarUrl;
      const syncedPatch: StoredStaffProfileOverride = {
        ...patch,
        fullName: updated.fullName || patch.fullName,
        phone: updated.phone !== undefined ? updated.phone : patch.phone,
        specialty: updated.specialty || patch.specialty,
        licenseNo: updated.licenseNo ?? safeLicenseNumber,
        avatarUrl: syncedAvatarUrl,
        bio: updated.bio ?? patch.bio,
      };
      const syncedOverrides: Record<string, StoredStaffProfileOverride> = {
        ...nextOverrides,
        [updated.id]: syncedPatch,
        [`usr:${updated.username.toLowerCase()}`]: syncedPatch,
      };
      setProfileOverrides(syncedOverrides);
      saveStaffProfileOverrides(syncedOverrides);
      if (syncedAvatarUrl) {
        setProfileAvatarUrl(syncedAvatarUrl);
      }
      setUsers((prev) =>
        prev.map((u) => (u.id === updated.id ? { ...u, ...updated, ...syncedPatch, licenseNo: updated.licenseNo ?? safeLicenseNumber } : u)),
      );
      flashBanner('success', locale.profile.savedSuccess);
      await refreshAll();
    } catch (err) {
      if (await handleAuthFailureIfExpired(err)) return;
      // Preserve local profile override cleanly while letting the user know cloud sync failed
      flashBanner('error', 'บันทึกเฉพาะเครื่องนี้ชั่วคราว แต่ยังไม่สามารถซิงก์รูปโปรไฟล์ขึ้นคลาวด์ได้ กรุณาลองกดบันทึกอีกครั้ง');
    } finally {
      setSubmitting(false);
    }
  };

  const openEditStaffModal = (target: UserAccount) => {
    if (!isAdminUser) return;
    const resolved = resolveUserProfile(target, profileOverrides);
    setEditingStaffUser(target);
    setEditStaffFullName(resolved.fullName);
    setEditStaffPhone(resolved.phone);
    setEditStaffSpecialty(resolved.specialty);
    setEditStaffLicenseNo(target.licenseNo ?? resolved.licenseNo);
  };

  const handleSaveEditedStaff = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!editingStaffUser || !isAdminUser) return;
    if (!editStaffFullName.trim()) {
      flashBanner('error', 'กรุณาระบุชื่อ-นามสกุล');
      return;
    }
    const existingLicense = editingStaffUser.licenseNo ?? '';
    const safeLicenseNumber = isAdminUser ? editStaffLicenseNo.trim() : existingLicense;
    const existingResolved = resolveUserProfile(editingStaffUser, profileOverrides);
    const targetPatch: StoredStaffProfileOverride = {
      fullName: editStaffFullName.trim(),
      phone: editStaffPhone.trim(),
      specialty: editStaffSpecialty.trim(),
      licenseNo: safeLicenseNumber,
      avatarUrl: existingResolved.avatarUrl,
      bio: existingResolved.bio,
    };
    const nextOverrides: Record<string, StoredStaffProfileOverride> = {
      ...profileOverrides,
      [editingStaffUser.id]: targetPatch,
      [`usr:${editingStaffUser.username.toLowerCase()}`]: targetPatch,
    };
    setProfileOverrides(nextOverrides);
    saveStaffProfileOverrides(nextOverrides);

    setSubmitting(true);
    try {
      const updated = await apiClient.updateUser(editingStaffUser.id, {
        fullName: targetPatch.fullName,
        phone: targetPatch.phone,
        specialty: editingStaffUser.role !== 'patient' ? targetPatch.specialty : undefined,
        licenseNo:
          isAdminUser && editingStaffUser.role !== 'patient'
            ? safeLicenseNumber
            : undefined,
      });
      setUsers((prev) =>
        prev.map((u) => (u.id === updated.id ? { ...u, ...updated, ...targetPatch } : u)),
      );
      if (selectedPtDetailModal?.id === editingStaffUser.id) {
        setSelectedPtDetailModal({ ...editingStaffUser, ...updated, ...targetPatch });
      }
      setEditingStaffUser(null);
      flashBanner('success', `บันทึกข้อมูลและเลขที่ใบประกอบวิชาชีพของ ${targetPatch.fullName} เรียบร้อยแล้ว`);
      await refreshAll();
    } catch (err) {
      if (await handleAuthFailureIfExpired(err)) return;
      flashBanner('error', err instanceof Error ? err.message : 'ไม่สามารถบันทึกข้อมูลเจ้าหน้าที่ได้');
    } finally {
      setSubmitting(false);
    }
  };

  const handleDeleteStaffAccount = async (target: UserAccount) => {
    const deletingSelf = target.id === currentUser.id;
    const confirmMsg = deletingSelf
      ? `ยืนยันลบบัญชีแอดมินของตนเอง (${target.fullName})? ระบบจะออกจากระบบอัตโนมัติ`
      : `ยืนยันลบบัญชี ${target.fullName} (@${target.username})?`;
    if (!window.confirm(confirmMsg)) return;

    setSubmitting(true);
    try {
      const isPatientRole = String(target.role ?? '').trim().toLowerCase() === 'patient';
      const derivedHn = isPatientRole ? formatCanonicalHn(target.username, target.patientId ?? target.id) : null;
      const linkedPatient = isPatientRole
        ? patients.find(
            (p) =>
              (target.patientId && p.id === target.patientId) ||
              normalizeHnKey(p.hn) === normalizeHnKey(target.username) ||
              p.fullName.trim().toLowerCase() === target.fullName.trim().toLowerCase(),
          )
        : undefined;

      if (isPatientRole) {
        await apiClient.deleteUser(target.id, target.username).catch(() => {});
        const cascadeTarget = {
          id: linkedPatient?.id ?? target.patientId ?? target.id,
          patientId: linkedPatient?.id ?? target.patientId ?? target.id,
          hn: linkedPatient?.hn ?? derivedHn,
          username: target.username,
          fullName: target.fullName,
          role: 'patient',
        };
        await apiClient
          .deletePatientCascading(
            linkedPatient?.id ?? target.patientId ?? derivedHn ?? target.id,
            linkedPatient?.hn ?? derivedHn ?? undefined,
          )
          .catch(() => {});
        await cascadeDeletePatientInSupabase(apiClient.getSupabaseClient(), cascadeTarget);
        cleanupDeletedPatientLocalStorage(cascadeTarget);
      } else {
        await apiClient.deleteUser(target.id, target.username);
      }

      const nextState = applyCascadingPatientDeletionToState(rawPatients, users, {
        id: target.id,
        patientId: linkedPatient?.id ?? target.patientId ?? null,
        hn: linkedPatient?.hn ?? derivedHn,
        username: target.username,
        fullName: target.fullName,
        role: target.role,
      });
      setUsers(nextState.users);
      setRawPatients(nextState.patients);

      const nextOverrides = { ...profileOverrides };
      delete nextOverrides[target.id];
      delete nextOverrides[`usr:${target.username.toLowerCase()}`];
      setProfileOverrides(nextOverrides);
      saveStaffProfileOverrides(nextOverrides);
      if (linkedPatient?.id) removeProfileFromLocalBiometricVault(linkedPatient.id);
      if (target.patientId) removeProfileFromLocalBiometricVault(target.patientId);
      if (linkedPatient?.hn) removeProfileFromLocalBiometricVault(linkedPatient.hn);
      if (derivedHn) removeProfileFromLocalBiometricVault(derivedHn);
      removeProfileFromLocalBiometricVault(target.username);
      notifyBiometricIndexChanged();
      if (deletingSelf) {
        onLogout();
        return;
      }
      flashBanner('success', `ลบบัญชี @${target.username} เรียบร้อยแล้ว`);
      await refreshAll();
    } catch (err) {
      if (await handleAuthFailureIfExpired(err)) return;
      flashBanner('error', err instanceof Error ? err.message : 'ไม่สามารถลบบัญชีผู้ใช้ได้');
    } finally {
      setSubmitting(false);
    }
  };

  const handleDeletePatientAccount = async (target: PatientRecord) => {
    if (!window.confirm(`ยืนยันลบข้อมูลคนไข้ ${target.fullName} (รวม Face ID)?`)) return;
    setSubmitting(true);
    try {
      const targetHnKey = normalizeHnKey(target.hn);
      const targetNameKey = target.fullName.trim().toLowerCase();
      const linkedUsers = users.filter(
        (u) =>
          u.patientId === target.id ||
          u.id === target.id ||
          (targetHnKey && normalizeHnKey(u.username) === targetHnKey) ||
          (u.role === 'patient' && u.fullName.trim().toLowerCase() === targetNameKey),
      );

      await apiClient.deletePatientCascading(target.id, target.hn);
      for (const lu of linkedUsers) {
        await apiClient.deleteUser(lu.id, lu.username).catch(() => {});
      }
      const cascadeTarget = {
        id: target.id,
        patientId: target.id,
        hn: target.hn,
        username: linkedUsers[0]?.username ?? targetHnKey,
        fullName: target.fullName,
        role: 'patient',
      };
      await cascadeDeletePatientInSupabase(apiClient.getSupabaseClient(), cascadeTarget);
      cleanupDeletedPatientLocalStorage(cascadeTarget);

      const nextState = applyCascadingPatientDeletionToState(rawPatients, users, cascadeTarget);
      setRawPatients(nextState.patients);
      setUsers(nextState.users);

      removeProfileFromLocalBiometricVault(target.id);
      removeProfileFromLocalBiometricVault(target.hn);
      for (const lu of linkedUsers) {
        removeProfileFromLocalBiometricVault(lu.username);
      }
      notifyBiometricIndexChanged();
      flashBanner('success', `ลบข้อมูลคนไข้ ${target.fullName} และบัญชีผู้ใช้ที่เชื่อมโยงเรียบร้อยแล้ว`);
      await refreshAll();
    } catch (err) {
      if (await handleAuthFailureIfExpired(err)) return;
      flashBanner('error', err instanceof Error ? err.message : 'ไม่สามารถลบข้อมูลคนไข้ได้');
    } finally {
      setSubmitting(false);
    }
  };

  const handleSubmitCaseDelegation = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!delegationPatientId) {
      flashBanner('error', 'กรุณาเลือกชื่อคนไข้ที่ต้องการฝากเคส');
      return;
    }
    if (!delegationTargetPtId) {
      flashBanner('error', 'กรุณาเลือกนักกายภาพบำบัดผู้รับฝากเคส');
      return;
    }
    const cleanDate = delegationDate || formatLocalIsoDate();
    const cleanNote = delegationNote.trim();
    const targetPatient = patients.find((p) => p.id === delegationPatientId);
    const targetPt = ptUsers.find((pt) => pt.id === delegationTargetPtId);

    setSubmitting(true);
    try {
      const updated = await apiClient.delegatePatientCase(
        delegationPatientId,
        delegationTargetPtId,
        cleanNote || undefined,
        cleanDate,
      );
      const nextOverrides = {
        ...loadCaseDelegationOverrides(),
        [delegationPatientId]: {
          delegatedToPtId: delegationTargetPtId,
          delegationDate: cleanDate,
          delegationNote: cleanNote || null,
        },
      };
      saveCaseDelegationOverrides(nextOverrides);
      setRawPatients((prev) =>
        prev.map((p) =>
          p.id === delegationPatientId
            ? {
              ...p,
              ...updated,
              delegatedToPtId: updated.delegatedToPtId ?? delegationTargetPtId,
              delegationDate: updated.delegationDate ?? cleanDate,
              delegationNote: updated.delegationNote ?? (cleanNote || null),
            }
            : p,
        ),
      );
      setDelegationNote('');
      flashBanner(
        'success',
        `บันทึกการฝากเคส ${targetPatient?.fullName ?? ''} ให้ ${targetPt?.fullName ?? 'นักกายภาพ'} (วันที่ ${cleanDate}) เรียบร้อยแล้ว`,
      );
      await refreshAll();
    } catch (err) {
      if (await handleAuthFailureIfExpired(err)) return;
      flashBanner('error', err instanceof Error ? err.message : 'ไม่สามารถฝากเคสคนไข้ได้');
    } finally {
      setSubmitting(false);
    }
  };

  const handleCancelCaseDelegation = async (patient: PatientRecord) => {
    setSubmitting(true);
    try {
      await apiClient.delegatePatientCase(patient.id, null);
      const nextOverrides = { ...loadCaseDelegationOverrides() };
      delete nextOverrides[patient.id];
      saveCaseDelegationOverrides(nextOverrides);
      setRawPatients((prev) =>
        prev.map((p) =>
          p.id === patient.id
            ? { ...p, delegatedToPtId: null, delegationDate: null, delegationNote: null }
            : p,
        ),
      );
      flashBanner('success', `ยกเลิกการฝากเคสของ ${patient.fullName} เรียบร้อยแล้ว`);
      await refreshAll();
    } catch (err) {
      if (await handleAuthFailureIfExpired(err)) return;
      flashBanner('error', err instanceof Error ? err.message : 'ยกเลิกการฝากเคสไม่สำเร็จ');
    } finally {
      setSubmitting(false);
    }
  };

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
      await apiClient.updatePatient(selectedSchedulePatient.id, {
        rehabSchedules: [],
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
    setSubmitting(true);
    try {
      const targetPatientId = selectedSchedulePatient.id;
      const emptyRest = Array.from({ length: 7 }, (_, dayIndex) => ({
        dayIndex,
        dayType: 'rest' as const,
        activityType: 'minigame' as const,
        difficulty: 'easy' as const,
        label: 'พักฟื้น',
        exercises: [],
        cognitiveModpackId: modpacks[0]?.id || 'modpack-core-99',
      }));
      const clearedCourse = await apiClient.upsertPatientCourse({
        patientId: targetPatientId,
        title: 'ล้างตารางนัดชั่วคราว',
        cognitiveModpackId: modpacks[0]?.id || 'modpack-core-99',
        weeklySchedule: emptyRest,
      });
      await apiClient.updatePatient(targetPatientId, {
        nextAppointmentDate: '',
        rehabSchedules: [],
      });
      setRawPatients((prev) =>
        prev.map((p) =>
          p.id === targetPatientId ? { ...p, nextAppointmentDate: '', rehabSchedules: [] } : p,
        ),
      );
      setPatientCourse(clearedCourse);
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

  const calendarDayStateByDate = useMemo(() => {
    const year = calendarDate.getFullYear();
    const month = calendarDate.getMonth();
    const daysInMonth = new Date(year, month + 1, 0).getDate();
    const customList: RehabScheduleItem[] = selectedSchedulePatient?.rehabSchedules ?? [];
    const hasCustomSchedules = customList.length > 0;
    const completedSessionDates = new Set(
      selectedSchedulePatient
        ? sessions
          .filter(
            (s) =>
              s.patientId === selectedSchedulePatient.id &&
              s.exerciseId !== 'CLINIC_CHECKIN' &&
              !s.painStopTriggered &&
              (s.activityType === 'minigame' || (s.completedReps ?? 0) > 0 || (s.completedSets ?? 0) > 0),
          )
          .map((s) => {
            if (!s.createdAt) return '';
            const d = new Date(s.createdAt);
            return Number.isNaN(d.getTime()) ? (s.createdAt || '').slice(0, 10) : formatLocalIsoDate(d);
          })
          .filter(Boolean)
        : [],
    );

    const map = new Map<
      string,
      {
        dateIso: string;
        customForDay: RehabScheduleItem[];
        firstCustomName: string | null;
        planDay: DailyScheduleDay | undefined;
        isTraining: boolean;
        cellStatus: 'ACTIVE' | 'REST' | 'COMPLETED' | 'MISSED';
      }
    >();

    for (let dayNum = 1; dayNum <= daysInMonth; dayNum += 1) {
      const dateIso = `${year}-${String(month + 1).padStart(2, '0')}-${String(dayNum).padStart(2, '0')}`;
      const weekday = new Date(year, month, dayNum).getDay();
      const customForDay = customList.filter((s) => s.scheduledDate === dateIso);
      const firstCustomBaseCode =
        customForDay.length > 0
          ? parseExerciseCodeAndSide(
            customForDay[0].effectiveCode || customForDay[0].exerciseCode || customForDay[0].exerciseId,
            customForDay[0].side,
          ).baseCode
          : null;
      const firstCustomEx = firstCustomBaseCode
        ? exercises.find(
          (ex) =>
            ex.id === customForDay[0].exerciseId ||
            ex.code === firstCustomBaseCode ||
            parseExerciseCodeAndSide(ex.code || ex.id).baseCode === firstCustomBaseCode,
        )
        : undefined;
      const firstCustomName = firstCustomEx ? firstCustomEx.name[lang] || firstCustomEx.name.th : null;
      const planDay = weeklySchedule[weekday];
      const isTraining =
        !hasCustomSchedules &&
        planDay?.dayType === 'training' &&
        ((planDay.exercises?.length ?? 0) > 0 ||
          (Boolean(patientCourse) &&
            !patientCourse!.title.endsWith('— แผนฟื้นฟูสมรรถภาพ') &&
            patientCourse!.title !== 'ล้างตารางนัดชั่วคราว'));
      const hasCompletedSession = completedSessionDates.has(dateIso);

      let cellStatus: 'ACTIVE' | 'REST' | 'COMPLETED' | 'MISSED' = 'REST';
      if (customForDay.length > 0) {
        if (
          customForDay.every((s) => s.status === 'COMPLETED') ||
          customForDay[0].status === 'COMPLETED' ||
          hasCompletedSession
        ) {
          cellStatus = 'COMPLETED';
        } else if (customForDay[0].status === 'REST') {
          cellStatus = 'REST';
        } else if (customForDay[0].status === 'MISSED') {
          cellStatus = 'MISSED';
        } else {
          cellStatus = 'ACTIVE';
        }
      } else if (isTraining) {
        cellStatus = hasCompletedSession ? 'COMPLETED' : 'ACTIVE';
      }

      map.set(dateIso, {
        dateIso,
        customForDay,
        firstCustomName,
        planDay,
        isTraining,
        cellStatus,
      });
    }

    return map;
  }, [calendarDate, selectedSchedulePatient, sessions, exercises, lang, weeklySchedule, patientCourse]);

  const scheduleStats = useMemo(() => {
    let active = 0;
    let paused = 0;
    let completed = 0;
    let missed = 0;

    calendarDayStateByDate.forEach((day) => {
      if (day.cellStatus === 'COMPLETED') completed += 1;
      else if (day.cellStatus === 'ACTIVE') active += 1;
      else if (day.cellStatus === 'MISSED') missed += 1;
      else if (day.customForDay.some((s) => s.status === 'REST')) paused += 1;
    });

    return {
      total: active + completed + missed,
      active,
      paused,
      completed,
      missed,
    };
  }, [calendarDayStateByDate]);

  const currentHeader = locale.headers[activeTab];
  const totalScheduledAppointmentDays = useMemo(() => {
    return patients.reduce((sum, p) => {
      const distinctDates = new Set(
        (p.rehabSchedules ?? [])
          .filter((s) => s.status !== 'REST')
          .map((s) => s.scheduledDate),
      );
      if (p.id === selectedSchedulePatient?.id) {
        return (
          sum +
          (scheduleStats.total > 0
            ? scheduleStats.total
            : distinctDates.size > 0
              ? distinctDates.size
              : p.nextAppointmentDate
                ? 1
                : 0)
        );
      }
      return sum + (distinctDates.size > 0 ? distinctDates.size : p.nextAppointmentDate ? 1 : 0);
    }, 0);
  }, [patients, selectedSchedulePatient?.id, scheduleStats.total]);

  const scheduledBadgeCount =
    scheduleSubTab === 'individual' && selectedSchedulePatient
      ? scheduleStats.total || totalScheduledAppointmentDays
      : totalScheduledAppointmentDays;

  return (
    <div className="min-h-screen flex bg-gradient-to-br from-[#F0FDF9] via-[#F0F7FF] to-[#F8FAFC] text-[#0F4C81]">
      {/* Clinical Blue-Teal Left Sidebar */}
      <aside
        className={`${sidebarCollapsed ? 'w-20 shrink-0' : 'w-64 shrink-0'
          } bg-gradient-to-b from-[#0F4C81] via-[#0284C7] to-[#0D9488] text-white flex flex-col justify-between border-r border-sky-200/40 transition-all duration-200 select-none shadow-lg`}
      >
        <div className="min-h-0 flex-1 overflow-y-auto">
          <div className="flex h-[72px] items-center justify-between gap-2 px-4 border-b border-white/20">
            <div className="flex items-center gap-3 min-w-0">
              <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-gradient-to-br from-emerald-300 to-white text-[#0F4C81] shadow-md">
                <HeartHandshake className="h-5 w-5" />
              </div>
              {!sidebarCollapsed && (
                <div className="min-w-0">
                  <div className="text-sm font-black tracking-tight text-white truncate">Dr.Tech.Care</div>
                  <div className="mt-1 inline-flex items-center gap-1.5 rounded-full bg-white/20 border border-white/30 px-2 py-0.5 text-[10px] font-bold text-white whitespace-nowrap">
                    <span className="h-1.5 w-1.5 rounded-full bg-emerald-300 shrink-0" />
                    <span className="truncate">{isAdmin ? locale.sidebar.adminRole : locale.sidebar.ptRole}</span>
                  </div>
                </div>
              )}
            </div>
            <button
              type="button"
              onClick={() => setSidebarCollapsed((c) => !c)}
              className="rounded-lg p-1.5 text-white/80 hover:bg-white/15 hover:text-white cursor-pointer shrink-0"
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
                { id: 'assessments', label: locale.nav.assessments, icon: ClipboardList },
                { id: 'pts', label: locale.nav.pts, icon: UserCheck },
                { id: 'exercises', label: locale.nav.exercises, icon: Gamepad2 },
                { id: 'schedules', label: locale.nav.schedules, icon: Calendar, badge: scheduledBadgeCount || undefined },
                {
                  id: 'security',
                  label: locale.nav.security,
                  icon: Shield,
                  badge: openStaffCalls.length || undefined,
                },
                { id: 'profile', label: locale.nav.profile, icon: UserCheck },
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
                  className={`w-full h-11 flex items-center justify-between gap-3 rounded-xl px-3.5 text-left whitespace-nowrap overflow-hidden text-ellipsis transition cursor-pointer ${isActive
                      ? 'bg-white text-[#0F4C81] font-extrabold shadow-md'
                      : 'text-white/90 hover:bg-white/15 hover:text-white'
                    }`}
                >
                  <span className="flex items-center gap-3 min-w-0 overflow-hidden">
                    <Icon className={`h-4 w-4 shrink-0 ${isActive ? 'text-[#0284C7]' : ''}`} />
                    {!sidebarCollapsed && <span className="truncate">{item.label}</span>}
                  </span>
                  {!sidebarCollapsed && item.badge ? (
                    <span className="flex h-5 min-w-[20px] shrink-0 items-center justify-center rounded-full bg-amber-300 px-1.5 text-[10px] font-black text-[#0F4C81]">
                      {item.badge}
                    </span>
                  ) : null}
                </button>
              );
            })}
          </nav>

          {/* Sidebar Personal Profile Summary Card */}
          {!sidebarCollapsed ? (
            <div className="px-3 pb-3">
              <button
                type="button"
                onClick={() => setActiveTab('profile')}
                className={`w-full rounded-2xl border p-3 text-left transition cursor-pointer ${activeTab === 'profile'
                    ? 'border-white/70 bg-white/25 shadow-md'
                    : 'border-white/25 bg-white/15 hover:border-white/50 hover:bg-white/20'
                  }`}
              >
                <div className="flex items-center justify-between gap-2 mb-2">
                  <span className="text-[10px] font-extrabold uppercase tracking-wider text-emerald-200">
                    {locale.profile.sidebarCardTitle}
                  </span>
                  <span className="text-[10px] font-semibold text-white underline decoration-white/60">
                    {locale.profile.sidebarEditHint}
                  </span>
                </div>
                <div className="flex items-center gap-2.5 min-w-0">
                  <img
                    src={myProfile.avatarUrl}
                    alt={myProfile.fullName}
                    className="h-12 w-12 shrink-0 rounded-xl object-cover border-2 border-white/60 bg-white"
                  />
                  <div className="min-w-0 flex-1">
                    <div className="text-xs font-extrabold text-white truncate">{myProfile.fullName}</div>
                    <div className="mt-0.5 flex items-center gap-1 text-[11px] text-emerald-100 font-mono truncate">
                      <Phone className="h-3 w-3 shrink-0" />
                      <span className="truncate">{myProfile.phone || locale.profile.noPhone}</span>
                    </div>
                  </div>
                </div>
                <p className="mt-2.5 rounded-xl bg-white/15 border border-white/20 px-2.5 py-2 text-[11px] leading-relaxed text-white/95 line-clamp-3">
                  {myProfile.bio}
                </p>
              </button>
            </div>
          ) : (
            <div className="px-3 pb-3 flex justify-center">
              <button
                type="button"
                onClick={() => setActiveTab('profile')}
                title={`${myProfile.fullName} • ${myProfile.phone || locale.profile.noPhone}`}
                className="rounded-xl p-1 border border-white/30 bg-white/15 hover:border-white/70 cursor-pointer"
              >
                <img
                  src={myProfile.avatarUrl}
                  alt={myProfile.fullName}
                  className="h-10 w-10 rounded-lg object-cover bg-white"
                />
              </button>
            </div>
          )}
        </div>

        <div className="p-3 space-y-2 border-t border-white/20 text-xs">
          <button
            type="button"
            onClick={() => handleLaunchPoseTest()}
            data-testid="sidebar-pose-test-btn"
            className="btn-primary-sheen w-full h-10 flex items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-emerald-500 to-teal-500 text-white hover:from-emerald-400 hover:to-teal-400 px-3 font-extrabold shadow-md border border-white/25 whitespace-nowrap overflow-hidden text-ellipsis cursor-pointer transition"
          >
            <Activity className="h-4 w-4 shrink-0" />
            {!sidebarCollapsed && <span className="truncate">ทดลองท่าทางกายภาพ (15 ท่า)</span>}
          </button>

          <button
            type="button"
            onClick={() => setCardGameModalOpen(true)}
            className="w-full h-10 flex items-center justify-center gap-2 rounded-xl bg-white text-[#0F4C81] hover:bg-emerald-50 px-3 font-extrabold shadow whitespace-nowrap overflow-hidden text-ellipsis cursor-pointer"
          >
            <Gamepad2 className="h-4 w-4 shrink-0 text-emerald-600" />
            {!sidebarCollapsed && <span className="truncate">{locale.actions.cardGame}</span>}
          </button>

          {!sidebarCollapsed && (
            <div className="flex h-9 items-center justify-between rounded-xl border border-white/25 bg-white/15 px-3 text-[11px] whitespace-nowrap">
              <span className="text-white/90 truncate">{locale.sidebar.dbCloud}</span>
              <span
                className={`inline-flex items-center gap-1 font-bold shrink-0 ${cloudHealth.dbConnected ? 'text-emerald-200' : 'text-rose-200'
                  }`}
              >
                <span
                  className={`h-2 w-2 rounded-full ${cloudHealth.dbConnected ? 'bg-emerald-300 animate-status-halo' : 'bg-rose-300'
                    }`}
                />
                {cloudHealth.dbConnected ? locale.sidebar.dbNormal : locale.sidebar.dbOffline}
              </span>
            </div>
          )}

          {onReturnToKiosk && (
            <button
              type="button"
              onClick={onReturnToKiosk}
              className="w-full h-10 flex items-center justify-center gap-2 rounded-xl border border-white/25 bg-white/15 px-3 font-bold text-white hover:bg-white/25 whitespace-nowrap overflow-hidden text-ellipsis cursor-pointer"
            >
              <Monitor className="h-4 w-4 shrink-0" />
              {!sidebarCollapsed && <span className="truncate">{locale.actions.returnKiosk}</span>}
            </button>
          )}

          <button
            type="button"
            onClick={() => void onLogout()}
            className="w-full h-10 flex items-center justify-center gap-2 rounded-xl border border-rose-200/40 bg-rose-500/25 px-3 font-bold text-white hover:bg-rose-500/40 whitespace-nowrap overflow-hidden text-ellipsis cursor-pointer"
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
                  className={`w-[68px] h-8 inline-flex items-center justify-center whitespace-nowrap rounded-full text-[11px] transition cursor-pointer ${lang === code ? 'bg-white text-emerald-700 shadow-sm border border-emerald-200' : 'text-slate-500'
                    }`}
                >
                  {code === 'cn' ? 'CN 中文' : code === 'th' ? 'TH ไทย' : 'GB EN'}
                </button>
              ))}
            </div>

            <button
              type="button"
              onClick={() => void pingCloudHealth(true)}
              title={
                cloudHealth.latencyMs !== undefined
                  ? `Worker API Latency: ${cloudHealth.latencyMs}ms (คลิกเพื่อตรวจสอบใหม่)`
                  : 'Test Worker API & Cloud Health'
              }
              className={`whitespace-nowrap h-9 min-w-[140px] inline-flex items-center justify-center gap-1.5 rounded-full border px-3.5 font-bold cursor-pointer transition shrink-0 ${cloudHealth.online
                  ? 'border-emerald-200 bg-emerald-50 text-emerald-700 hover:bg-emerald-100'
                  : 'border-rose-200 bg-rose-50 text-rose-700 hover:bg-rose-100'
                }`}
            >
              {cloudHealth.online ? (
                <>
                  <span className="h-2 w-2 rounded-full bg-emerald-500 animate-status-halo shrink-0" />
                  <Wifi className="h-3.5 w-3.5 shrink-0 text-emerald-600" />
                </>
              ) : (
                <WifiOff className="h-3.5 w-3.5 shrink-0" />
              )}
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

            <button
              type="button"
              onClick={() => handleLaunchPoseTest()}
              data-testid="header-pose-test-btn"
              className="whitespace-nowrap h-9 min-w-[145px] inline-flex items-center justify-center gap-1.5 rounded-full bg-gradient-to-r from-emerald-600 to-teal-600 px-3.5 font-extrabold text-white shadow-sm hover:from-emerald-500 hover:to-teal-500 cursor-pointer shrink-0 transition"
            >
              <Activity className="h-3.5 w-3.5 shrink-0" />
              <span>ทดลองท่ากายภาพ</span>
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
              key={bannerMessage.text}
              className={`rounded-2xl border px-4 py-3 text-xs font-bold flex items-center justify-between shadow-sm animate-toast-in ${bannerMessage.kind === 'success'
                  ? 'border-emerald-200 bg-emerald-50 text-emerald-800'
                  : 'border-rose-200 bg-rose-50 text-rose-800 animate-gentle-shake'
                }`}
            >
              <span>{bannerMessage.text}</span>
              <button type="button" onClick={() => setBannerMessage(null)}>
                <X className="h-4 w-4" />
              </button>
            </div>
          )}

          <div key={activeTab} className="animate-view-enter">
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
                        <tr key={u.id} className="hover:bg-slate-50/80 transition-colors duration-150">
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
                          <td className="px-4 py-3.5 font-mono font-bold text-sky-600 truncate">
                            @{u.username.replace(/^@+/, '')}
                          </td>
                          <td className="px-4 py-3.5">
                            {u.role === 'super_admin' && (
                              <span className="inline-flex items-center gap-1 whitespace-nowrap rounded-full border border-purple-200 bg-purple-50 px-2.5 py-1 text-[11px] font-bold text-purple-700">
                                <Shield className="h-3 w-3 shrink-0" /> {locale.staff.roleAdmin}
                              </span>
                            )}
                            {u.role === 'pt' && (
                              <div className="flex flex-wrap items-center gap-1.5">
                                <span className="inline-flex items-center gap-1 whitespace-nowrap rounded-full border border-emerald-200 bg-emerald-50 px-2.5 py-1 text-[11px] font-bold text-emerald-700">
                                  <UserCheck className="h-3 w-3 shrink-0" /> {locale.staff.rolePt}
                                </span>
                                {(u.licenseNo || profileOverrides[u.id]?.licenseNo) && (
                                  <span className="whitespace-nowrap rounded-md border border-amber-200 bg-amber-50 px-2 py-0.5 font-mono text-[10px] font-bold text-amber-900">
                                    {u.licenseNo || profileOverrides[u.id]?.licenseNo}
                                  </span>
                                )}
                              </div>
                            )}
                            {u.role === 'patient' && (
                              <span className="inline-flex items-center gap-1 whitespace-nowrap rounded-full border border-sky-200 bg-sky-50 px-2.5 py-1 text-[11px] font-bold text-sky-700">
                                <Users className="h-3 w-3 shrink-0" /> {locale.staff.rolePatient}
                              </span>
                            )}
                          </td>
                          <td className="px-4 py-3.5">
                            <span
                              className={`inline-flex items-center gap-1 whitespace-nowrap rounded-full border px-2.5 py-1 text-[11px] font-bold ${u.active
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
                                {u.role !== 'patient' && (
                                  <button
                                    type="button"
                                    onClick={() => openEditStaffModal(u)}
                                    className="whitespace-nowrap rounded-lg border border-amber-200 bg-amber-50 px-2.5 py-1 text-[11px] font-bold text-amber-800 hover:bg-amber-100 cursor-pointer inline-flex items-center gap-1"
                                    title="แก้ไขข้อมูลและเลขที่ใบประกอบวิชาชีพ"
                                  >
                                    <Pencil className="h-3 w-3" />
                                    <span>แก้ไข</span>
                                  </button>
                                )}
                                {u.id !== currentUser.id && (
                                  <button
                                    type="button"
                                    disabled={submitting}
                                    onClick={async () => {
                                      setSubmitting(true);
                                      try {
                                        const nextActive = !u.active;
                                        const updated = await apiClient.updateUser(u.id, { active: nextActive });
                                        setUsers((prev) => {
                                          const exists = prev.some((item) => item.id === updated.id || item.id === u.id);
                                          if (exists) {
                                            return prev.map((item) =>
                                              item.id === updated.id || item.id === u.id
                                                ? { ...item, ...updated, active: nextActive }
                                                : item,
                                            );
                                          }
                                          return [...prev, { ...u, ...updated, active: nextActive }];
                                        });
                                        if (u.role === 'patient') {
                                          const linkedPat = patients.find(
                                            (p) =>
                                              p.id === u.id ||
                                              p.id === u.patientId ||
                                              normalizeHnKey(p.hn) === normalizeHnKey(u.username),
                                          );
                                          if (linkedPat) {
                                            void upsertPatientStaffUserInSupabase(
                                              apiClient.getSupabaseClient(),
                                              linkedPat,
                                              { active: nextActive },
                                            );
                                          }
                                        }
                                        flashBanner(
                                          'success',
                                          u.active
                                            ? `ระงับสิทธิ์ @${u.username.replace(/^@+/, '')} แล้ว`
                                            : `เปิดใช้งาน @${u.username.replace(/^@+/, '')} แล้ว`,
                                        );
                                        await refreshAll();
                                      } catch (err) {
                                        flashBanner(
                                          'error',
                                          err instanceof Error ? err.message : 'อัปเดตสถานะบัญชีไม่สำเร็จ',
                                        );
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
                                {(u.role !== 'super_admin' || superAdminCount > 1) && (
                                  <button
                                    type="button"
                                    disabled={submitting}
                                    onClick={() => void handleDeleteStaffAccount(u)}
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
                      <div className="flex flex-wrap items-center gap-2">
                        <h2 className="text-lg font-extrabold text-slate-900">{locale.patients.title}</h2>
                        <span
                          data-testid="admin-supabase-sync-pill"
                          className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-[11px] font-bold transition-colors ${patientSyncError
                              ? 'border-rose-200 bg-rose-50 text-rose-700'
                              : 'border-emerald-200 bg-emerald-50 text-emerald-700'
                            }`}
                        >
                          <span
                            className={`h-2 w-2 rounded-full shrink-0 ${patientSyncError
                                ? 'bg-rose-500'
                                : patientsSyncing
                                  ? 'bg-sky-500 animate-pulse'
                                  : 'bg-emerald-500'
                              }`}
                          />
                          <svg
                            viewBox="0 0 20 20"
                            fill="none"
                            aria-hidden="true"
                            className="h-3.5 w-3.5 shrink-0"
                          >
                            <ellipse cx="10" cy="4.5" rx="6.5" ry="2.5" stroke="currentColor" strokeWidth="1.6" />
                            <path
                              d="M3.5 4.5V10C3.5 11.38 6.41 12.5 10 12.5C13.59 12.5 16.5 11.38 16.5 10V4.5"
                              stroke="currentColor"
                              strokeWidth="1.6"
                            />
                            <path
                              d="M3.5 10V15.5C3.5 16.88 6.41 18 10 18C13.59 18 16.5 16.88 16.5 15.5V10"
                              stroke="currentColor"
                              strokeWidth="1.6"
                            />
                          </svg>
                          <span>ซิงก์ข้อมูลจาก Supabase: {patients.length} บัญชี</span>
                        </span>
                      </div>
                      <p className="text-xs text-slate-500 mt-0.5">
                        {locale.patients.subtitle}
                      </p>
                    </div>
                  </div>
                  <div className="flex flex-wrap items-center gap-2 shrink-0">
                    <button
                      type="button"
                      data-testid="admin-refresh-patients-btn"
                      disabled={patientsSyncing || loading}
                      onClick={() => void syncPatientsFromSupabase({ showSuccessToast: true })}
                      className="btn-tactile whitespace-nowrap h-10 px-3.5 inline-flex items-center justify-center gap-1.5 rounded-xl border border-sky-200 bg-white text-xs font-bold text-sky-700 hover:bg-sky-50 disabled:opacity-60 cursor-pointer shrink-0 shadow-xs"
                      title={locale.staff.refreshTitle}
                    >
                      <svg
                        viewBox="0 0 20 20"
                        fill="none"
                        aria-hidden="true"
                        className={`h-4 w-4 shrink-0 ${patientsSyncing || loading ? 'animate-spin' : ''}`}
                      >
                        <path
                          d="M16.25 10A6.25 6.25 0 1 1 14.42 5.58L16.25 7.25M16.25 7.25V3.75M16.25 7.25H12.75"
                          stroke="currentColor"
                          strokeWidth="1.75"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        />
                      </svg>
                      <span>รีเฟรชข้อมูล</span>
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        setEnrollTargetPatient(null);
                        setEnrollModalOpen(true);
                      }}
                      className="btn-tactile whitespace-nowrap h-10 px-4 inline-flex items-center justify-center gap-1.5 shrink-0 rounded-xl bg-emerald-600 text-xs font-extrabold text-white shadow-sm hover:bg-emerald-500 cursor-pointer"
                    >
                      <UserPlus className="h-4 w-4 shrink-0" />
                      {locale.patients.addBtn}
                    </button>
                  </div>
                </div>

                {/* Segmented Clinic Scope Toggle (All Clinic Patients vs Assigned To Me) */}
                <div className="flex flex-wrap items-center justify-between gap-2.5">
                  <div
                    role="tablist"
                    aria-label="ขอบเขตการแสดงรายชื่อผู้ป่วย"
                    data-testid="patient-scope-toggle"
                    className="inline-flex items-center rounded-xl border border-slate-200 bg-slate-100/80 p-1 gap-1"
                  >
                    <button
                      type="button"
                      role="tab"
                      aria-selected={patientScopeFilter === 'all'}
                      data-testid="patient-scope-all"
                      onClick={() => setPatientScopeFilter('all')}
                      className={`btn-tactile whitespace-nowrap inline-flex items-center gap-1.5 rounded-lg px-3.5 py-1.5 text-xs font-bold transition-all cursor-pointer ${patientScopeFilter === 'all'
                          ? 'bg-white text-sky-800 shadow-xs ring-1 ring-sky-200'
                          : 'text-slate-600 hover:text-slate-900'
                        }`}
                    >
                      <span>คนไข้ทั้งหมดในคลินิก (All)</span>
                      <span
                        className={`rounded-full px-1.5 py-0.5 text-[10px] font-extrabold ${patientScopeFilter === 'all'
                            ? 'bg-emerald-100 text-emerald-700'
                            : 'bg-slate-200 text-slate-600'
                          }`}
                      >
                        {patients.length}
                      </span>
                    </button>
                    <button
                      type="button"
                      role="tab"
                      aria-selected={patientScopeFilter === 'assigned'}
                      data-testid="patient-scope-assigned"
                      onClick={() => setPatientScopeFilter('assigned')}
                      className={`btn-tactile whitespace-nowrap inline-flex items-center gap-1.5 rounded-lg px-3.5 py-1.5 text-xs font-bold transition-all cursor-pointer ${patientScopeFilter === 'assigned'
                          ? 'bg-white text-sky-800 shadow-xs ring-1 ring-sky-200'
                          : 'text-slate-600 hover:text-slate-900'
                        }`}
                    >
                      <span>คนไข้ในความดูแลของฉัน (Assigned)</span>
                      <span
                        className={`rounded-full px-1.5 py-0.5 text-[10px] font-extrabold ${patientScopeFilter === 'assigned'
                            ? 'bg-sky-100 text-sky-700'
                            : 'bg-slate-200 text-slate-600'
                          }`}
                      >
                        {assignedPatientsCount}
                      </span>
                    </button>
                  </div>
                </div>

                {patientSyncError && (
                  <div
                    role="alert"
                    data-testid="patient-sync-error-banner"
                    className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-xs text-rose-800 shadow-xs animate-toast-in"
                  >
                    <div className="flex items-start gap-2.5 min-w-0">
                      <svg
                        viewBox="0 0 20 20"
                        fill="none"
                        aria-hidden="true"
                        className="h-4 w-4 text-rose-600 mt-0.5 shrink-0"
                      >
                        <path
                          d="M10 3.2L17.2 16H2.8L10 3.2Z"
                          stroke="currentColor"
                          strokeWidth="1.6"
                          strokeLinejoin="round"
                        />
                        <path d="M10 8V11.4" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
                        <circle cx="10" cy="13.8" r="0.9" fill="currentColor" />
                      </svg>
                      <div className="min-w-0">
                        <div className="font-extrabold text-rose-900">
                          ไม่สามารถซิงก์รายชื่อผู้ป่วยจาก Supabase ได้
                        </div>
                        <div className="mt-0.5 font-mono text-[11px] text-rose-700 break-words">
                          {patientSyncError}
                        </div>
                      </div>
                    </div>
                    <button
                      type="button"
                      onClick={() => void syncPatientsFromSupabase({ showSuccessToast: true })}
                      className="btn-tactile whitespace-nowrap rounded-lg border border-rose-300 bg-white px-3 py-1.5 text-[11px] font-bold text-rose-700 hover:bg-rose-100 cursor-pointer shrink-0"
                    >
                      ลองซิงก์ใหม่
                    </button>
                  </div>
                )}

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
                      onClick={() => void syncPatientsFromSupabase({ showSuccessToast: true })}
                      className="btn-tactile whitespace-nowrap h-8 px-3.5 inline-flex items-center justify-center shrink-0 rounded-lg bg-emerald-600 text-xs font-bold text-white hover:bg-emerald-500 cursor-pointer"
                    >
                      {locale.patients.searchBtn}
                    </button>
                  </div>
                  <select
                    aria-label="กรองสถานะใบหน้าผู้ป่วย"
                    value={patientFaceFilter}
                    onChange={(e) => setPatientFaceFilter(e.target.value as 'all' | 'enrolled' | 'pending')}
                    className="h-10 rounded-xl border border-slate-200 bg-white px-4 text-xs font-semibold text-slate-700"
                  >
                    <option value="all">{locale.patients.allFace}</option>
                    <option value="enrolled">{locale.patients.faceVerified}</option>
                    <option value="pending">{locale.patients.facePending}</option>
                  </select>
                </div>

                {(() => {
                  const reenrollCount = patients.filter(
                    (item) => getPatientBiometricStatusDetail(item) === 'REENROLL_REQUIRED',
                  ).length;
                  if (reenrollCount <= 0) return null;
                  return (
                    <div
                      data-testid="patients-reenroll-required-banner"
                      className="flex flex-wrap items-center justify-between gap-2 rounded-xl border border-rose-200 bg-rose-50/90 px-4 py-2.5 text-xs font-bold text-rose-800"
                    >
                      <span>
                        พบผู้ป่วยต้องลงทะเบียนใบหน้าใหม่ {reenrollCount} คน (ข้อมูลเวกเตอร์คนละรุ่นกับระบบ 1024-D ปัจจุบัน)
                      </span>
                      <span className="text-[11px] font-semibold text-rose-700">
                        กรุณากดปุ่ม &quot;{locale.patients.scanFaceBtn}&quot; ในรายการผู้ป่วยที่เกี่ยวข้อง
                      </span>
                    </div>
                  );
                })()}

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
                        const bioStatus = getPatientBiometricStatusDetail(p);
                        const verified = bioStatus === 'VERIFIED';
                        const ptLabel = ptNameMap.get(p.delegatedToPtId || p.assignedPtId) || 'กภ. ประจำคลินิก';
                        return (
                          <tr key={p.id} className="hover:bg-slate-50/80 transition-colors duration-150">
                            <td className="px-5 py-3.5">
                              <span className="inline-block whitespace-nowrap rounded-lg border border-sky-200 bg-sky-50 px-2.5 py-1 font-mono text-[11px] font-extrabold text-sky-700">
                                {p.hn}
                              </span>
                            </td>
                            <td className="px-4 py-3.5">
                              <div className="font-bold text-slate-900 truncate">{p.fullName}</div>
                              <div className="text-[11px] text-slate-400 truncate">{p.diagnosis}</div>
                              <div className="mt-1 flex flex-wrap items-center gap-1">
                                <span className="rounded bg-emerald-50 border border-emerald-200 px-1.5 py-0.5 text-[10px] font-extrabold text-emerald-700">
                                  Mahidol {p.motorAssessmentScore ?? 3}/6
                                </span>
                                {p.affectedSide && p.affectedSide !== 'NONE' && (
                                  <span className="rounded bg-amber-50 border border-amber-200 px-1.5 py-0.5 text-[10px] font-bold text-amber-800">
                                    เน้น: {p.affectedSide}
                                  </span>
                                )}
                              </div>
                            </td>
                            <td className="px-4 py-3.5 font-medium text-slate-700 whitespace-nowrap">
                              {p.age} {locale.patients.ageUnit} / {p.gender || 'ชาย'}
                            </td>
                            <td className="px-4 py-3.5">
                              <div className="font-mono text-slate-700 truncate">{p.phone || '-'}</div>
                              <div className="text-[11px] font-semibold text-emerald-700 truncate">{ptLabel}</div>
                              {p.delegatedToPtId && (
                                <div className="mt-1 inline-flex items-center gap-1 rounded-md border border-amber-200 bg-amber-50 px-2 py-0.5 text-[10px] font-bold text-amber-800 max-w-full truncate">
                                  <span>ฝากเคส{p.delegationDate ? ` (${p.delegationDate})` : ''}</span>
                                  {p.delegationNote ? <span>• {p.delegationNote}</span> : null}
                                </div>
                              )}
                            </td>
                            <td className="px-4 py-3.5">
                              {bioStatus === 'VERIFIED' ? (
                                <div className="space-y-1">
                                  <span className="inline-flex items-center gap-1.5 whitespace-nowrap rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1 text-[11px] font-bold text-emerald-700">
                                    <CheckCircle2 className="h-3.5 w-3.5 shrink-0" />
                                    {locale.patients.verifiedBadge}
                                  </span>
                                  {p.biometricPipelineVersion === 'v2-pre-iris-1024d' && (
                                    <div className="text-[10px] font-semibold text-sky-700">
                                      1024-D (รุ่นก่อนอัปเดตม่านตา)
                                    </div>
                                  )}
                                </div>
                              ) : bioStatus === 'REENROLL_REQUIRED' ? (
                                <div className="space-y-1">
                                  <span className="inline-flex items-center gap-1.5 whitespace-nowrap rounded-full border border-rose-200 bg-rose-50 px-3 py-1 text-[11px] font-bold text-rose-700">
                                    <Clock className="h-3.5 w-3.5 shrink-0" />
                                    ต้องลงทะเบียนหน้าใหม่
                                  </span>
                                  <div className="text-[10px] font-semibold text-rose-600">
                                    เวกเตอร์คนละรุ่น ({p.biometricEmbeddingDim || p.masterEmbedding?.length || 512}-D)
                                  </div>
                                </div>
                              ) : (
                                <span className="inline-flex items-center gap-1.5 whitespace-nowrap rounded-full border border-amber-200 bg-amber-50 px-3 py-1 text-[11px] font-bold text-amber-800">
                                  <Clock className="h-3.5 w-3.5 shrink-0" />
                                  {locale.patients.pendingBadge}
                                </span>
                              )}
                            </td>
                            <td className="px-5 py-3.5 text-right">
                              <div className="inline-flex flex-wrap items-center justify-end gap-1.5">
                                <button
                                  type="button"
                                  onClick={() => {
                                    setSelectedAssessmentPatientId(p.id);
                                    setActiveTab('assessments');
                                  }}
                                  className="btn-tactile whitespace-nowrap inline-flex items-center gap-1 rounded-xl bg-emerald-600 px-2.5 py-1.5 text-[11px] font-bold text-white hover:bg-emerald-500 cursor-pointer"
                                >
                                  <ClipboardList className="h-3.5 w-3.5 shrink-0" />
                                  <span>เปิดแฟ้มประวัติและประเมินผล</span>
                                </button>
                                <button
                                  type="button"
                                  onClick={() => setPatientModalState({ patientId: p.id, tab: 'profile' })}
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
                                  title="สแกนอัปเดตใบหน้าใหม่ (Re-enroll Face)"
                                  className={`whitespace-nowrap inline-flex items-center gap-1 rounded-xl border px-2.5 py-1.5 text-[11px] font-bold cursor-pointer ${verified
                                      ? 'border-teal-200 bg-teal-50 text-teal-800 hover:bg-teal-100'
                                      : 'border-emerald-200 bg-emerald-50 text-emerald-700 hover:bg-emerald-100'
                                    }`}
                                >
                                  <ScanFace className="h-3.5 w-3.5 shrink-0" />
                                  <span>{verified ? locale.patients.updateFaceBtn : locale.patients.scanFaceBtn}</span>
                                </button>
                                <button
                                  type="button"
                                  onClick={() => setPatientModalState({ patientId: p.id, tab: 'history' })}
                                  className="whitespace-nowrap rounded-xl border border-slate-200 bg-white px-2.5 py-1.5 text-[11px] font-bold text-slate-700 hover:bg-slate-50 cursor-pointer"
                                >
                                  {locale.patients.historyBtn}
                                </button>
                                <button
                                  type="button"
                                  onClick={() => setEhrLinkPatient(p)}
                                  title="เชื่อมโยง/ส่งต่อประวัติการรักษา (Universal EHR)"
                                  className="whitespace-nowrap inline-flex items-center gap-1 rounded-xl border border-indigo-200 bg-indigo-50 px-2 py-1.5 text-[11px] font-bold text-indigo-700 hover:bg-indigo-100 cursor-pointer"
                                >
                                  <Link2 className="h-3 w-3 shrink-0" />
                                  <span>EHR</span>
                                </button>
                                {isAdmin && (
                                  <button
                                    type="button"
                                    disabled={submitting}
                                    onClick={() => void handleDeletePatientAccount(p)}
                                    className="whitespace-nowrap rounded-xl border border-rose-200 bg-rose-50 p-1.5 text-rose-600 hover:bg-rose-100 disabled:opacity-50 cursor-pointer"
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

            {/* TAB 2B: INDIVIDUAL CLINICAL ASSESSMENT (0-6) & LONGITUDINAL ROM RECORD */}
            {activeTab === 'assessments' && (
              <PatientClinicalRecordPanel
                apiClient={apiClient}
                patients={patients}
                sessions={sessions}
                currentUser={currentUser}
                selectedPatientId={selectedAssessmentPatientId}
                onSelectPatientId={setSelectedAssessmentPatientId}
                onOpenUniversalEhrModal={(pat) => setEhrLinkPatient(pat)}
                onAssessmentSaved={refreshAll}
              />
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

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 animate-stagger-children">
                  {filteredPts.map((pt) => {
                    const ptProfile = resolveUserProfile(pt, profileOverrides);
                    const ptPatients = patients.filter(
                      (p) => p.assignedPtId === pt.id || p.delegatedToPtId === pt.id,
                    );
                    return (
                      <div
                        key={pt.id}
                        className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm flex flex-col justify-between space-y-4 card-hover-lift animate-card-enter hover:border-emerald-200"
                      >
                        <div className="space-y-3">
                          <div className="flex items-start justify-between gap-2">
                            <div className="flex items-center gap-3 min-w-0">
                              <img
                                src={ptProfile.avatarUrl}
                                alt={ptProfile.fullName}
                                className="h-14 w-14 shrink-0 rounded-2xl object-cover border-2 border-emerald-200 bg-white shadow-sm"
                              />
                              <div className="min-w-0">
                                <h3 className="text-sm font-extrabold text-slate-900 truncate">{ptProfile.fullName}</h3>
                                <p className="font-mono text-[11px] font-bold text-sky-600 truncate">@{pt.username}</p>
                                <p className="flex items-center gap-1 text-[11px] font-mono text-slate-500 truncate mt-0.5">
                                  <Phone className="h-3 w-3 text-emerald-600 shrink-0" />
                                  <span className="truncate">{ptProfile.phone || locale.profile.noPhone}</span>
                                </p>
                              </div>
                            </div>
                            <div className="flex items-center gap-1.5 shrink-0">
                              <span className="whitespace-nowrap rounded-full bg-emerald-50 border border-emerald-200 px-2.5 py-0.5 text-[11px] font-bold text-emerald-700">
                                PT
                              </span>
                              {isAdmin && (
                                <button
                                  type="button"
                                  disabled={submitting}
                                  onClick={() => void handleDeleteStaffAccount(pt)}
                                  className="rounded-lg p-1 text-slate-400 hover:bg-rose-50 hover:text-rose-600 disabled:opacity-50 cursor-pointer"
                                  title={locale.staff.deleteTitle}
                                >
                                  <Trash2 className="h-4 w-4" />
                                </button>
                              )}
                            </div>
                          </div>

                          <div className="flex items-center justify-between gap-2 rounded-xl border border-amber-200/80 bg-amber-50/70 px-3.5 py-2 text-xs">
                            <span className="font-semibold text-amber-900 whitespace-nowrap">{locale.pts.licenseLabel}</span>
                            <div className="flex items-center gap-1.5">
                              <span className="whitespace-nowrap rounded-lg bg-white border border-amber-200 px-2.5 py-0.5 font-mono text-[11px] font-bold text-slate-800">
                                {ptProfile.licenseNo}
                              </span>
                              {isAdminUser ? (
                                <button
                                  type="button"
                                  onClick={() => openEditStaffModal(pt)}
                                  className="inline-flex items-center gap-1 rounded-lg border border-amber-300 bg-white px-2 py-0.5 text-[10px] font-extrabold text-amber-800 hover:bg-amber-100 cursor-pointer"
                                  title="แก้ไขเลขที่ใบประกอบวิชาชีพ (Admin)"
                                >
                                  <Pencil className="h-3 w-3" />
                                  <span>แก้ไข</span>
                                </button>
                              ) : (
                                <span
                                  className="inline-flex items-center gap-0.5 text-[10px] font-bold text-slate-500 whitespace-nowrap"
                                  title="ดูได้อย่างเดียว (แก้ไขได้เฉพาะผู้ดูแลระบบ Admin)"
                                >
                                  <Lock className="h-3 w-3" />
                                </span>
                              )}
                            </div>
                          </div>

                          <div className="space-y-1.5 text-xs text-slate-600">
                            <div className="flex items-center gap-2 min-w-0">
                              <FileText className="h-3.5 w-3.5 text-slate-400 shrink-0" />
                              <span className="whitespace-nowrap">{locale.pts.specialtyLabel}</span>
                              <span className="font-semibold text-slate-900 truncate">{ptProfile.specialty}</span>
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
                            <p className="rounded-xl bg-slate-50 border border-slate-100 px-3 py-2 text-[11px] leading-relaxed text-slate-600 line-clamp-2">
                              {ptProfile.bio}
                            </p>
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
                  <div className="flex flex-wrap items-center gap-2 shrink-0">
                    <button
                      type="button"
                      onClick={() => handleLaunchPoseTest()}
                      data-testid="tab4-pose-test-btn"
                      className="whitespace-nowrap h-10 px-4 inline-flex items-center justify-center gap-2 shrink-0 rounded-xl bg-gradient-to-r from-emerald-600 to-teal-600 text-xs font-extrabold text-white shadow hover:opacity-95 cursor-pointer"
                    >
                      <Activity className="h-4 w-4 shrink-0" />
                      <span>ทดลองท่าทางกายภาพ (15 ท่า)</span>
                    </button>
                    <button
                      type="button"
                      onClick={() => setCardGameModalOpen(true)}
                      className="btn-primary-sheen whitespace-nowrap h-10 px-4 inline-flex items-center justify-center gap-2 shrink-0 rounded-xl bg-gradient-to-r from-emerald-600 to-blue-600 text-xs font-extrabold text-white shadow hover:opacity-95 cursor-pointer"
                    >
                      <Gamepad2 className="h-4 w-4 shrink-0" />
                      {locale.exercises.playCardBtn}
                    </button>
                  </div>
                </div>

                {/* AI Pose Safety Switch Banner */}
                <div
                  className={`rounded-2xl border p-4 flex flex-wrap items-center justify-between gap-3 ${systemSettings.aiPoseDetectionEnabled
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
                    className={`whitespace-nowrap h-10 px-4 inline-flex items-center justify-center shrink-0 rounded-xl text-xs font-bold text-white shadow-sm disabled:opacity-45 disabled:cursor-not-allowed cursor-pointer ${systemSettings.aiPoseDetectionEnabled
                        ? 'bg-amber-600 hover:bg-amber-500'
                        : 'bg-emerald-600 hover:bg-emerald-500'
                      }`}
                  >
                    {systemSettings.aiPoseDetectionEnabled
                      ? locale.exercises.aiPoseToggleLock
                      : locale.exercises.aiPoseToggleUnlock}
                  </button>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 animate-stagger-children">
                  {/* Cognitive Modpacks */}
                  <div className="rounded-2xl border border-slate-200 bg-white p-5 space-y-4">
                    <h3 className="text-sm font-extrabold text-slate-900">
                      {locale.exercises.modpackHeader} ({modpacks.length})
                    </h3>
                    <div className="space-y-2.5 max-h-72 overflow-y-auto pr-1">
                      {modpacks.map((m) => (
                        <div
                          key={m.id}
                          className="flex items-center justify-between rounded-xl border border-slate-200 bg-slate-50/70 px-3.5 py-2.5 text-xs transition-colors duration-150 hover:bg-emerald-50/40 hover:border-emerald-200"
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
                        inputMode="numeric"
                        min={5}
                        max={99}
                        value={mpQuestionCount}
                        onKeyDown={(e) => {
                          if (['e', 'E', '+', '-', '.'].includes(e.key)) e.preventDefault();
                        }}
                        onChange={(e) => {
                          const digits = e.target.value.replace(/\D/g, '').slice(0, 2);
                          setMpQuestionCount(digits ? Math.min(99, Math.max(5, Number(digits))) : 5);
                        }}
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
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <h3 className="text-sm font-extrabold text-slate-900">
                        {locale.exercises.exerciseHeader} ({exercises.length})
                      </h3>
                      <button
                        type="button"
                        disabled={submitting}
                        onClick={async () => {
                          setSubmitting(true);
                          try {
                            const synced = await syncStrokeExerciseLibrary(apiClient);
                            setExercises(synced.exercises);
                            flashBanner(
                              'success',
                              `ซิงก์คลังท่ากายภาพมาตรฐาน Stroke / สธ. ${synced.exercises.length} ท่าเรียบร้อย`,
                            );
                          } catch (err) {
                            flashBanner(
                              'error',
                              err instanceof Error ? err.message : 'ซิงก์คลังท่ากายภาพไม่สำเร็จ',
                            );
                          } finally {
                            setSubmitting(false);
                          }
                        }}
                        className="inline-flex items-center gap-1.5 rounded-xl border border-sky-200 bg-sky-50 px-3 py-1.5 text-[11px] font-bold text-sky-800 hover:bg-sky-100 disabled:opacity-50 cursor-pointer"
                      >
                        <RefreshCw className="h-3.5 w-3.5" />
                        ซิงก์คลัง 15 ท่าหลัก Stroke/สธ.
                      </button>
                    </div>
                    <div className="space-y-2.5 max-h-72 overflow-y-auto pr-1">
                      {sortedMergedExercises.map((ex) => (
                        <div
                          key={ex.id}
                          className="flex items-center justify-between rounded-xl border border-slate-200 bg-slate-50/70 px-3.5 py-2.5 text-xs transition-colors duration-150 hover:bg-sky-50/40 hover:border-sky-200"
                        >
                          <div className="min-w-0">
                            <div className="font-bold text-slate-900 truncate">
                              {ex.name[lang] || ex.name.th}{' '}
                              <span className="ml-1 rounded bg-sky-100 px-1.5 py-0.5 font-mono text-[10px] text-sky-800">
                                {ex.code}
                              </span>
                              <span className="ml-1 rounded bg-emerald-100 px-1.5 py-0.5 text-[10px] font-bold text-emerald-800 uppercase">
                                {ex.detection.side === 'both' ||
                                  ex.code === 'STROKE_BOBATH_ARM_RAISE' ||
                                  ex.code === 'STROKE_BOBATH_FORWARD_PUSH' ||
                                  ex.code === 'MOPH_NECK_TURN' ||
                                  ex.code === 'MOPH_SHOULDER_SHRUG_ROLL'
                                  ? 'L+R (BOTH)'
                                  : 'L / R / BOTH'}
                              </span>
                            </div>
                            <div className="text-[11px] text-slate-500 truncate">{ex.instruction[lang] || ex.instruction.th}</div>
                          </div>
                          <div className="flex items-center gap-1.5 shrink-0">
                            <button
                              type="button"
                              onClick={() => handleLaunchPoseTest(ex.code || ex.id)}
                              data-testid={`exercise-card-pose-test-${ex.code || ex.id}`}
                              className="inline-flex items-center gap-1 whitespace-nowrap rounded-xl border border-emerald-300 bg-emerald-50 px-2.5 py-1.5 text-[11px] font-extrabold text-emerald-800 hover:bg-emerald-100 cursor-pointer transition"
                            >
                              <Play className="h-3 w-3 fill-current shrink-0" />
                              <span>ทดลองทำท่านี้ด้วยกล้องทันที</span>
                            </button>
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
                              jointGroup: exCategory === 'neck_trunk' ? 'neck' : 'shoulder',
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
                        <option value="upper_limb">Upper Limb (แขน/ไหล่/ศอก/มือ Stroke)</option>
                        <option value="neck_trunk">Neck &amp; Shoulders (คอ/บ่า สธ.)</option>
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
                        className="sm:col-span-2 whitespace-nowrap h-10 px-4 inline-flex items-center justify-center rounded-xl bg-gradient-to-r from-[#0F4C81] to-[#0D9488] font-bold text-white hover:brightness-105 disabled:opacity-50 cursor-pointer shadow-sm"
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
                          setSelectedCalendarDateIso(formatLocalIsoDate());
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
                      className={`whitespace-nowrap h-8 px-4 inline-flex items-center justify-center rounded-lg transition cursor-pointer ${scheduleSubTab === 'individual' ? 'bg-emerald-50 text-emerald-700' : 'text-slate-500'
                        }`}
                    >
                      {locale.schedules.subIndividual}
                    </button>
                    <button
                      type="button"
                      onClick={() => setScheduleSubTab('all')}
                      className={`whitespace-nowrap h-8 px-4 inline-flex items-center justify-center rounded-lg transition cursor-pointer ${scheduleSubTab === 'all' ? 'bg-emerald-50 text-emerald-700' : 'text-slate-500'
                        }`}
                    >
                      {locale.schedules.subAll} ({totalScheduledAppointmentDays})
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
                  <div className="rounded-2xl border-2 border-emerald-200 bg-white p-5 space-y-4 shadow-md animate-popup-spring">
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
                        <div
                          key={idx}
                          style={{ animationDelay: `${idx * 35}ms` }}
                          className="rounded-xl border border-slate-200 bg-slate-50/80 p-3 space-y-2 text-xs card-hover-lift animate-card-enter"
                        >
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
                                {lang === 'cn' ? '认知卡牌小游戏' : lang === 'en' ? 'Cognitive Minigame' : 'มินิเกมการ์ด'}
                              </option>
                              <option
                                value="physio_exercise"
                                disabled={!isAdmin && !systemSettings.aiPoseDetectionEnabled}
                              >
                                {lang === 'cn' ? 'AI 康复训练' : lang === 'en' ? 'AI Rehab Exercise' : 'ท่ากายภาพ AI'}{' '}
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

                <div key={scheduleSubTab} className="animate-view-enter space-y-4">
                  {scheduleSubTab === 'individual' ? (
                    <>
                      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 animate-stagger-children">
                        <div className="min-h-[84px] rounded-2xl border border-slate-200 bg-white p-4 flex items-center justify-between gap-2 card-hover-lift animate-card-enter">
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
                        <div
                          style={{ animationDelay: '45ms' }}
                          className="min-h-[84px] rounded-2xl border border-slate-200 bg-white p-4 flex items-center justify-between gap-2 card-hover-lift animate-card-enter"
                        >
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
                        <div
                          style={{ animationDelay: '90ms' }}
                          className="min-h-[84px] rounded-2xl border border-slate-200 bg-white p-4 flex items-center justify-between gap-2 card-hover-lift animate-card-enter"
                        >
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
                        <div
                          style={{ animationDelay: '135ms' }}
                          className="min-h-[84px] rounded-2xl border border-slate-200 bg-white p-4 flex items-center justify-between gap-2 card-hover-lift animate-card-enter"
                        >
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
                            <h3
                              key={`cal-title-${calendarDate.getFullYear()}-${calendarDate.getMonth()}`}
                              className="text-sm font-black text-slate-900 whitespace-nowrap animate-toast-in"
                            >
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

                        <div
                          key={`cal-grid-${calendarDate.getFullYear()}-${calendarDate.getMonth()}`}
                          className="grid grid-cols-7 gap-2 animate-calendar-month"
                        >
                          {calendarDays.map((dayNum, idx) => {
                            if (dayNum === null) {
                              return <div key={`empty-${idx}`} className="min-h-[88px] sm:min-h-[96px] rounded-xl bg-slate-50/40" />;
                            }
                            const year = calendarDate.getFullYear();
                            const month = calendarDate.getMonth();
                            const dateIso = `${year}-${String(month + 1).padStart(2, '0')}-${String(dayNum).padStart(2, '0')}`;
                            const today = new Date();
                            const isToday =
                              dayNum === today.getDate() && month === today.getMonth() && year === today.getFullYear();

                            const dayState = calendarDayStateByDate.get(dateIso);
                            const customForDay = dayState?.customForDay ?? [];
                            const firstCustomName = dayState?.firstCustomName ?? null;
                            const planDay = dayState?.planDay;
                            const isTraining = Boolean(dayState?.isTraining);
                            const cellStatus = dayState?.cellStatus ?? 'REST';

                            return (
                              <button
                                key={dateIso}
                                type="button"
                                style={{ animationDelay: `${Math.min(idx, 35) * 14}ms` }}
                                onClick={() => {
                                  setSelectedCalendarDateIso(dateIso);
                                  setScheduleModalMode('day_detail');
                                }}
                                className={`calendar-day-tile animate-card-enter min-h-[88px] sm:min-h-[96px] rounded-xl border p-2 flex flex-col justify-between text-left text-xs cursor-pointer ${isToday
                                    ? 'border-emerald-500 bg-emerald-50/40 ring-1 ring-emerald-500/35'
                                    : 'border-slate-100 bg-white hover:border-emerald-300'
                                  }`}
                              >
                                <div className="w-full flex items-center justify-between gap-1">
                                  <span
                                    className={`inline-flex h-6 w-6 items-center justify-center rounded-full text-[11px] font-bold transition-transform ${isToday ? 'bg-emerald-600 text-white animate-status-halo' : 'text-slate-700'
                                      }`}
                                  >
                                    {dayNum}
                                  </span>
                                  {customForDay.length > 1 && (
                                    <span className="rounded-full bg-slate-100 px-1.5 py-0.5 text-[9px] font-extrabold text-slate-600">
                                      +{customForDay.length - 1} ท่า
                                    </span>
                                  )}
                                </div>
                                <div className="w-full space-y-1">
                                  {customForDay.length > 0 ? (
                                    <>
                                      <span
                                        className={`inline-block whitespace-nowrap rounded px-1.5 py-0.5 text-[10px] font-bold ${cellStatus === 'COMPLETED'
                                            ? 'bg-blue-100 text-blue-800'
                                            : cellStatus === 'REST'
                                              ? 'bg-amber-100 text-amber-800'
                                              : cellStatus === 'MISSED'
                                                ? 'bg-rose-100 text-rose-800'
                                                : 'bg-emerald-100 text-emerald-800'
                                          }`}
                                      >
                                        {cellStatus === 'COMPLETED'
                                          ? locale.schedules.statusCompleted
                                          : cellStatus === 'REST'
                                            ? locale.schedules.statusRest
                                            : cellStatus === 'MISSED'
                                              ? locale.schedules.statusMissed
                                              : locale.schedules.statusActive}
                                      </span>
                                      {firstCustomName && (
                                        <div className="w-full truncate text-[10px] font-bold text-slate-700">
                                          {firstCustomName}
                                        </div>
                                      )}
                                    </>
                                  ) : isTraining && planDay ? (
                                    <span
                                      className={`inline-block whitespace-nowrap rounded px-1.5 py-0.5 text-[10px] font-bold ${cellStatus === 'COMPLETED'
                                          ? 'bg-blue-100 text-blue-800'
                                          : 'bg-emerald-100 text-emerald-800'
                                        }`}
                                    >
                                      {cellStatus === 'COMPLETED'
                                        ? locale.schedules.statusCompleted
                                        : lang === 'cn'
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
                                    <span className="text-[10px] text-slate-300">{locale.schedules.restDayLabel}</span>
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
                          {patients.map((p) => {
                            const scheduledDates = Array.from(
                              new Set(
                                (p.rehabSchedules ?? [])
                                  .filter((s) => s.status !== 'REST')
                                  .map((s) => s.scheduledDate),
                              ),
                            ).sort();
                            const apptCount =
                              scheduledDates.length > 0
                                ? scheduledDates.length
                                : p.id === selectedSchedulePatient?.id
                                  ? scheduleStats.total
                                  : p.nextAppointmentDate
                                    ? 1
                                    : 0;
                            const nextAppt =
                              resolveNextAppointmentDate({
                                rehabSchedules: p.rehabSchedules,
                                fallbackNextAppointmentDate: p.nextAppointmentDate,
                                todayIso: todayIsoStr,
                              }) || '-';
                            return (
                              <tr key={p.id} className="hover:bg-slate-50/80 transition-colors duration-150">
                                <td className="px-5 py-3 font-mono font-bold text-sky-700 whitespace-nowrap">{p.hn}</td>
                                <td className="px-4 py-3 font-bold text-slate-900 truncate">{p.fullName}</td>
                                <td className="px-4 py-3 text-slate-600 truncate">
                                  {ptNameMap.get(p.delegatedToPtId || p.assignedPtId) || 'กภ. ประจำคลินิก'}
                                </td>
                                <td className="px-4 py-3 font-semibold text-emerald-700 whitespace-nowrap">
                                  {nextAppt} ({apptCount} {locale.schedules.unitAppt})
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
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
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
                      disabled={creatingCall}
                      onClick={() => {
                        void submitStaffCall(
                          kiosks[0]?.kioskId || 'KIOSK-OPD-01',
                          'ทดสอบสัญญาณเรียกเจ้าหน้าที่จากหน้าตู้ Kiosk',
                        );
                      }}
                      className="whitespace-nowrap h-9 px-3.5 inline-flex items-center justify-center rounded-xl border border-amber-200 bg-amber-50 text-[11px] font-bold text-amber-800 hover:bg-amber-100 disabled:opacity-50 cursor-pointer shrink-0"
                    >
                      {locale.security.testAlertBtn}
                    </button>
                    <button
                      type="button"
                      onClick={() => void refreshAll()}
                      className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-r from-[#0F4C81] to-[#0D9488] text-white hover:brightness-105 cursor-pointer shrink-0 shadow-xs"
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
                      { id: 'calls', label: `${locale.security.tabCalls} (${staffCalls.length})`, Icon: Bell },
                      { id: 'audit', label: `${locale.security.tabAudit} (${auditLogs.length})`, Icon: Clock },
                      { id: 'sessions', label: `${locale.security.tabSessions} (${activeSessions.length})`, Icon: Users },
                      { id: 'bans', label: `${locale.security.tabBans} (${bannedTargets.length})`, Icon: Ban },
                      { id: 'kiosks', label: `${locale.security.tabKiosks} (${kiosks.length})`, Icon: Monitor },
                    ] as const
                  ).map((sub) => (
                    <button
                      key={sub.id}
                      type="button"
                      onClick={() => setSecuritySubTab(sub.id)}
                      className={`whitespace-nowrap h-9 px-4 inline-flex items-center justify-center gap-1.5 rounded-xl transition cursor-pointer ${securitySubTab === sub.id
                          ? 'bg-blue-700 text-white shadow-sm'
                          : 'border border-slate-200 bg-white text-slate-600 hover:bg-slate-50'
                        }`}
                    >
                      <sub.Icon className="h-3.5 w-3.5 shrink-0" />
                      <span>{sub.label}</span>
                    </button>
                  ))}
                </div>

                {securitySubTab === 'calls' && (
                  <div className="space-y-3">
                    {/* Quick Manual Staff Call / Dispatch Form */}
                    <form
                      onSubmit={async (e) => {
                        e.preventDefault();
                        const trimmed = callReason.trim();
                        if (!trimmed) {
                          flashBanner('error', 'กรุณาระบุสาเหตุหรือข้อความสำหรับเรียกเจ้าหน้าที่');
                          return;
                        }
                        const ok = await submitStaffCall(
                          callKioskId || kiosks[0]?.kioskId || 'KIOSK-OPD-01',
                          trimmed,
                          callPatientId || null,
                        );
                        if (ok) {
                          setCallReason('');
                        }
                      }}
                      className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm space-y-3"
                    >
                      <div className="flex flex-wrap items-center justify-between gap-2">
                        <div>
                          <h3 className="text-xs font-extrabold text-slate-900">
                            {lang === 'cn'
                              ? '发起 / 记录呼叫医护人员'
                              : lang === 'en'
                                ? 'Create / Log Staff Assistance Call'
                                : 'สร้างคำเรียกเจ้าหน้าที่ / บันทึกเหตุขอความช่วยเหลือจากหน้าตู้'}
                          </h3>
                          <p className="text-[11px] text-slate-500">
                            {lang === 'cn'
                              ? '可选择设备与患者并立即向护士站及治疗师推送提醒'
                              : lang === 'en'
                                ? 'Select a kiosk and optional patient to dispatch an alert immediately'
                                : 'เลือกตู้ Kiosk และคนไข้ (ถ้ามี) เพื่อส่งแจ้งเตือนเรียกเจ้าหน้าที่ทันที'}
                          </p>
                        </div>
                        <div className="flex flex-wrap gap-1.5">
                          {[
                            'ผู้ป่วยปวดหรือเวียนศีรษะระหว่างฝึก',
                            'ขอความช่วยเหลือปรับระดับกล้อง / ระยะยืน',
                            'ขอความช่วยเหลือสแกนใบหน้าเข้าใช้งาน',
                          ].map((preset) => (
                            <button
                              key={preset}
                              type="button"
                              onClick={() => setCallReason(preset)}
                              className="rounded-lg border border-sky-200 bg-sky-50 px-2.5 py-1 text-[11px] font-bold text-sky-800 hover:bg-sky-100 cursor-pointer"
                            >
                              + {preset}
                            </button>
                          ))}
                        </div>
                      </div>

                      <div className="grid grid-cols-1 md:grid-cols-12 gap-2.5">
                        <div className="md:col-span-3">
                          <select
                            value={callKioskId}
                            onChange={(e) => setCallKioskId(e.target.value)}
                            className="w-full h-10 rounded-xl border border-slate-200 bg-slate-50 px-3 text-xs font-bold text-slate-800"
                          >
                            {kiosks.length > 0 ? (
                              kiosks.map((k) => (
                                <option key={k.kioskId} value={k.kioskId}>
                                  {k.kioskId} ({k.location})
                                </option>
                              ))
                            ) : (
                              <option value="KIOSK-OPD-01">KIOSK-OPD-01</option>
                            )}
                          </select>
                        </div>
                        <div className="md:col-span-3">
                          <select
                            value={callPatientId}
                            onChange={(e) => setCallPatientId(e.target.value)}
                            className="w-full h-10 rounded-xl border border-slate-200 bg-slate-50 px-3 text-xs font-bold text-slate-800"
                          >
                            <option value="">
                              {lang === 'cn'
                                ? '— 未指定患者 —'
                                : lang === 'en'
                                  ? '— Optional patient —'
                                  : '— ไม่ระบุชื่อคนไข้ —'}
                            </option>
                            {patients.map((p) => (
                              <option key={p.id} value={p.id}>
                                {p.fullName} ({p.hn})
                              </option>
                            ))}
                          </select>
                        </div>
                        <div className="md:col-span-4">
                          <input
                            type="text"
                            maxLength={240}
                            value={callReason}
                            onChange={(e) => setCallReason(e.target.value)}
                            placeholder={
                              lang === 'cn'
                                ? '输入呼叫原因，例如：患者需要协助...'
                                : lang === 'en'
                                  ? 'Enter call reason, e.g. Patient needs assistance...'
                                  : 'ระบุสาเหตุคำเรียกเจ้าหน้าที่ เช่น ผู้ป่วยขอความช่วยเหลือ...'
                            }
                            className="w-full h-10 rounded-xl border border-slate-200 px-3 text-xs"
                          />
                        </div>
                        <div className="md:col-span-2">
                          <button
                            type="submit"
                            disabled={creatingCall}
                            className="w-full h-10 rounded-xl bg-rose-600 px-3 text-xs font-extrabold text-white shadow-sm hover:bg-rose-500 disabled:opacity-50 cursor-pointer"
                          >
                            {creatingCall
                              ? '...'
                              : lang === 'cn'
                                ? '+ 呼叫医护'
                                : lang === 'en'
                                  ? '+ Send Call'
                                  : '+ เรียกเจ้าหน้าที่'}
                          </button>
                        </div>
                      </div>
                    </form>

                    <div className="rounded-2xl border border-slate-200 bg-white shadow-sm overflow-hidden">
                      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-100 px-5 py-3.5 text-xs">
                        <span className="font-extrabold text-slate-900">{locale.security.callsTableTitle}</span>
                        <div className="flex items-center gap-3">
                          {staffCalls.some((c) => c.status === 'resolved') && (
                            <button
                              type="button"
                              onClick={async () => {
                                const resolvedIds = staffCalls
                                  .filter((c) => c.status === 'resolved')
                                  .map((c) => c.id);
                                setStaffCalls((prev) => prev.filter((c) => c.status !== 'resolved'));
                                try {
                                  await Promise.all(resolvedIds.map((id) => apiClient.deleteStaffCall(id)));
                                  flashBanner('success', `ล้างรายการที่แก้ไขแล้วจำนวน ${resolvedIds.length} รายการ`);
                                } catch (err) {
                                  flashBanner(
                                    'error',
                                    err instanceof Error ? err.message : 'ไม่สามารถล้างรายการได้',
                                  );
                                  await refreshAll();
                                }
                              }}
                              className="inline-flex items-center gap-1 rounded-lg border border-slate-200 bg-slate-50 px-2.5 py-1 text-[11px] font-bold text-slate-700 hover:bg-slate-100 cursor-pointer"
                            >
                              <Trash2 className="h-3.5 w-3.5 text-slate-500" />
                              <span>
                                {lang === 'cn'
                                  ? '清除已处理'
                                  : lang === 'en'
                                    ? 'Clear Resolved'
                                    : 'ล้างรายการที่แก้ไขแล้ว'}
                              </span>
                            </button>
                          )}
                          <span className="text-[11px] text-slate-400 whitespace-nowrap">{locale.security.callsPollNote}</span>
                        </div>
                      </div>
                      <div className="overflow-x-auto">
                        <table className="table-fixed w-full min-w-[760px] text-left text-xs">
                          <thead className="bg-slate-50 text-[11px] font-bold text-slate-500">
                            <tr>
                              <th className="w-[18%] px-5 py-3">{locale.security.colTime}</th>
                              <th className="w-[14%] px-4 py-3">{locale.security.colKiosk}</th>
                              <th className="w-[34%] px-4 py-3">{locale.security.colReason}</th>
                              <th className="w-[14%] px-4 py-3">{locale.security.colStatus}</th>
                              <th className="w-[20%] px-5 py-3 text-right">{locale.security.colActions}</th>
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-slate-100">
                            {staffCalls.map((call) => {
                              const patientLabel = call.patientId
                                ? patientNameMap.get(call.patientId) ?? call.patientId
                                : null;
                              return (
                                <tr key={call.id} className="hover:bg-slate-50/80 transition-colors duration-150">
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
                                  <td className="px-4 py-3">
                                    <div className="font-semibold text-rose-600 break-words">{call.reason}</div>
                                    {patientLabel && (
                                      <div className="inline-flex items-center gap-1 text-[11px] font-bold text-slate-500 mt-0.5">
                                        <User className="h-3 w-3 shrink-0" />
                                        <span>{patientLabel}</span>
                                      </div>
                                    )}
                                  </td>
                                  <td className="px-4 py-3">
                                    <span
                                      className={`inline-flex items-center gap-1 whitespace-nowrap rounded-full px-2.5 py-1 text-[11px] font-bold ${call.status === 'resolved'
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
                                            setStaffCalls((prev) =>
                                              prev.map((c) =>
                                                c.id === call.id ? { ...c, status: 'acknowledged' } : c,
                                              ),
                                            );
                                            try {
                                              await apiClient.acknowledgeStaffCall(call.id);
                                              flashBanner('success', `รับเรื่องแจ้งเตือนจาก ${call.kioskId} แล้ว`);
                                              void refreshAll();
                                            } catch (err) {
                                              flashBanner(
                                                'error',
                                                err instanceof Error ? err.message : 'ไม่สามารถรับเรื่องคำเรียกได้',
                                              );
                                              await refreshAll();
                                            }
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
                                            setStaffCalls((prev) =>
                                              prev.map((c) =>
                                                c.id === call.id ? { ...c, status: 'resolved' } : c,
                                              ),
                                            );
                                            try {
                                              await apiClient.resolveStaffCall(call.id);
                                              flashBanner('success', `ปิดเคสช่วยเหลือจาก ${call.kioskId} แล้ว`);
                                              void refreshAll();
                                            } catch (err) {
                                              flashBanner(
                                                'error',
                                                err instanceof Error ? err.message : 'ไม่สามารถปิดเคสคำเรียกได้',
                                              );
                                              await refreshAll();
                                            }
                                          }}
                                          className="whitespace-nowrap rounded-lg bg-emerald-600 px-3 py-1 text-[11px] font-bold text-white hover:bg-emerald-500 cursor-pointer"
                                        >
                                          {locale.security.resolveBtn}
                                        </button>
                                      )}
                                      <button
                                        type="button"
                                        title="ลบรายการคำเรียกเจ้าหน้าที่"
                                        onClick={async () => {
                                          setStaffCalls((prev) => prev.filter((c) => c.id !== call.id));
                                          try {
                                            await apiClient.deleteStaffCall(call.id);
                                            flashBanner('success', 'ลบรายการคำเรียกเจ้าหน้าที่เรียบร้อย');
                                          } catch (err) {
                                            flashBanner(
                                              'error',
                                              err instanceof Error ? err.message : 'ไม่สามารถลบรายการได้',
                                            );
                                            await refreshAll();
                                          }
                                        }}
                                        className="inline-flex h-7 w-7 items-center justify-center rounded-lg border border-slate-200 bg-white text-slate-500 hover:border-rose-200 hover:bg-rose-50 hover:text-rose-600 cursor-pointer"
                                      >
                                        <Trash2 className="h-3.5 w-3.5" />
                                      </button>
                                    </div>
                                  </td>
                                </tr>
                              );
                            })}
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
                              className="flex flex-wrap items-center justify-between gap-2 rounded-xl border border-slate-100 bg-slate-50/80 px-4 py-2.5 text-xs transition-colors duration-150 hover:bg-white hover:border-slate-200"
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
                              disabled={submitting}
                              onClick={async () => {
                                if (submitting) return;
                                setSubmitting(true);
                                try {
                                  await apiClient.forceLogoutUser(s.userId);
                                  flashBanner('success', 'เตะเซสชันออกจากระบบเรียบร้อย');
                                  await refreshAll();
                                } catch (err) {
                                  flashBanner(
                                    'error',
                                    err instanceof Error ? err.message : 'ไม่สามารถเตะเซสชันออกจากระบบได้',
                                  );
                                } finally {
                                  setSubmitting(false);
                                }
                              }}
                              className="whitespace-nowrap h-9 px-3.5 inline-flex items-center justify-center rounded-xl bg-rose-600 text-[11px] font-bold text-white hover:bg-rose-500 disabled:opacity-50 cursor-pointer shrink-0"
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
                            if (!banValue.trim() || submitting) return;
                            setSubmitting(true);
                            try {
                              await apiClient.banTarget(banType, banValue.trim(), banReason.trim() || 'Security block');
                              setBanValue('');
                              setBanReason('');
                              flashBanner('success', 'เพิ่มรายการแบนเรียบร้อยแล้ว');
                              await refreshAll();
                            } catch (err) {
                              flashBanner(
                                'error',
                                err instanceof Error ? err.message : 'เพิ่มรายการแบนไม่สำเร็จ',
                              );
                            } finally {
                              setSubmitting(false);
                            }
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
                            disabled={submitting}
                            className="whitespace-nowrap h-10 px-4 inline-flex items-center justify-center gap-1 rounded-xl bg-rose-600 font-bold text-white hover:bg-rose-500 disabled:opacity-50 cursor-pointer"
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
                                disabled={submitting}
                                onClick={async () => {
                                  if (submitting) return;
                                  setSubmitting(true);
                                  try {
                                    await apiClient.unbanTarget(b.id);
                                    flashBanner('success', `ปลดแบน ${b.value} แล้ว`);
                                    await refreshAll();
                                  } catch (err) {
                                    flashBanner(
                                      'error',
                                      err instanceof Error ? err.message : 'ปลดแบนไม่สำเร็จ',
                                    );
                                  } finally {
                                    setSubmitting(false);
                                  }
                                }}
                                className="whitespace-nowrap h-8 px-3 rounded-lg border border-slate-200 bg-white font-bold text-slate-700 hover:bg-slate-100 disabled:opacity-50 cursor-pointer shrink-0"
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
                              className={`whitespace-nowrap rounded-full px-2.5 py-0.5 text-[11px] font-bold ${k.online ? 'bg-emerald-100 text-emerald-700' : 'bg-slate-200 text-slate-600'
                                }`}
                            >
                              {k.online ? '● Online' : 'Offline'}
                            </span>
                            {isAdmin && k.kioskId !== 'KIOSK-OPD-01' && (
                              <button
                                type="button"
                                disabled={submitting}
                                onClick={async () => {
                                  if (submitting) return;
                                  setSubmitting(true);
                                  try {
                                    await apiClient.deleteKiosk(k.kioskId);
                                    flashBanner('success', `ลบ Kiosk ${k.kioskId} แล้ว`);
                                    await refreshAll();
                                  } catch (err) {
                                    flashBanner(
                                      'error',
                                      err instanceof Error ? err.message : 'ลบ Kiosk ไม่สำเร็จ',
                                    );
                                  } finally {
                                    setSubmitting(false);
                                  }
                                }}
                                className="text-rose-500 hover:text-rose-700 disabled:opacity-50 cursor-pointer"
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
                            ? '启用 AI 人脸识别 (1024D)'
                            : lang === 'en'
                              ? 'Enable AI Face Scan (1024D)'
                              : 'เปิดใช้งาน AI Face Scan (1024D)'}
                        </span>
                        <input
                          type="checkbox"
                          disabled={!isAdmin}
                          checked={systemSettings.aiFaceScanEnabled}
                          onChange={async (e) => {
                            try {
                              const updated = await apiClient.updateSystemSettings({
                                aiFaceScanEnabled: e.target.checked,
                              });
                              setSystemSettings(updated);
                            } catch (err) {
                              flashBanner('error', err instanceof Error ? err.message : 'อัปเดตการตั้งค่าไม่สำเร็จ');
                            }
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
                          onChange={(e) => {
                            const nextVal = Number(e.target.value);
                            setSystemSettings((prev) => ({ ...prev, aiFaceThreshold: nextVal }));
                            if (thresholdTimerRef.current !== null) window.clearTimeout(thresholdTimerRef.current);
                            thresholdTimerRef.current = window.setTimeout(() => {
                              thresholdTimerRef.current = null;
                              void apiClient
                                .updateSystemSettings({ aiFaceThreshold: nextVal })
                                .then((updated) => setSystemSettings(updated))
                                .catch((err) => {
                                  flashBanner('error', err instanceof Error ? err.message : 'บันทึก Threshold ไม่สำเร็จ');
                                });
                            }, 250);
                          }}
                          className="w-full"
                        />
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* TAB 7: PERSONAL PROFILE (รูปประกอบ, ชื่อ, เบอร์โทร, ประวัติส่วนตัว) */}
            {activeTab === 'profile' && (
              <div className="space-y-5">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div className="flex items-start gap-2.5 min-w-0">
                    <User className="h-5 w-5 text-emerald-600 mt-0.5 shrink-0" />
                    <div className="min-w-0">
                      <h2 className="text-lg font-extrabold text-slate-900">{locale.headers.profile.title}</h2>
                      <p className="text-xs text-slate-500">{locale.headers.profile.subtitle}</p>
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-1 xl:grid-cols-12 gap-6 items-start">
                  {/* Left Column: Live Clinician Profile Card */}
                  <div className="xl:col-span-5 rounded-3xl border border-slate-200 bg-white p-6 shadow-sm space-y-5">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-extrabold uppercase tracking-wider text-emerald-700">
                        {locale.profile.previewTitle}
                      </span>
                      <span className="rounded-full bg-emerald-50 border border-emerald-200 px-3 py-0.5 text-[11px] font-bold text-emerald-700">
                        {isAdmin ? locale.sidebar.adminRole : locale.sidebar.ptRole}
                      </span>
                    </div>

                    <div className="flex flex-col sm:flex-row items-center sm:items-start gap-4">
                      <img
                        src={profileAvatarUrl}
                        alt={profileFullName || activeUserRecord.fullName}
                        className="h-36 w-36 shrink-0 rounded-3xl object-cover border-4 border-emerald-100 shadow-md bg-white"
                      />
                      <div className="min-w-0 flex-1 text-center sm:text-left space-y-1.5">
                        <h3 className="text-xl font-black text-slate-900 break-words">
                          {profileFullName || activeUserRecord.fullName}
                        </h3>
                        <p className="font-mono text-xs font-bold text-sky-600">@{activeUserRecord.username}</p>
                        <div className="inline-flex items-center gap-1.5 rounded-xl bg-amber-50 border border-amber-200 px-3 py-1 text-xs font-bold text-amber-900">
                          <Award className="h-3.5 w-3.5 text-amber-600 shrink-0" />
                          <span className="font-mono">{profileLicenseNo || 'PT-44012'}</span>
                        </div>
                      </div>
                    </div>

                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
                      <div className="rounded-2xl border border-slate-200 bg-slate-50/70 p-3.5">
                        <div className="text-[11px] font-bold text-slate-400">{locale.profile.phoneLabel}</div>
                        <div className="mt-1 flex items-center gap-1.5 font-mono text-sm font-extrabold text-slate-900">
                          <Phone className="h-4 w-4 text-emerald-600 shrink-0" />
                          <span>{profilePhone || locale.profile.noPhone}</span>
                        </div>
                      </div>
                      <div className="rounded-2xl border border-slate-200 bg-slate-50/70 p-3.5">
                        <div className="text-[11px] font-bold text-slate-400">{locale.profile.patientsCountLabel}</div>
                        <div className="mt-1 flex items-center gap-1.5 text-sm font-extrabold text-sky-700">
                          <Users className="h-4 w-4 text-sky-600 shrink-0" />
                          <span>
                            {
                              patients.filter(
                                (p) => p.assignedPtId === activeUserRecord.id || p.delegatedToPtId === activeUserRecord.id,
                              ).length
                            }{' '}
                            {locale.pts.patientsUnit}
                          </span>
                        </div>
                      </div>
                    </div>

                    <div className="rounded-2xl border border-slate-200 bg-slate-50/70 p-3.5 text-xs">
                      <div className="text-[11px] font-bold text-slate-400">{locale.profile.specialtyLabel}</div>
                      <div className="mt-1 font-bold text-slate-900">
                        {profileSpecialty || 'เวชศาสตร์ฟื้นฟูกายภาพบำบัด'}
                      </div>
                    </div>

                    <div className="rounded-2xl border border-emerald-100 bg-emerald-50/40 p-4 space-y-1.5">
                      <div className="flex items-center gap-1.5 text-xs font-extrabold text-emerald-900">
                        <FileText className="h-4 w-4 text-emerald-600 shrink-0" />
                        <span>{locale.profile.bioLabel}</span>
                      </div>
                      <p className="text-xs leading-relaxed text-slate-700 whitespace-pre-line">
                        {profileBio || getDefaultBioForUser(activeUserRecord)}
                      </p>
                    </div>
                  </div>

                  {/* Right Column: Edit Personal Profile Form */}
                  <form
                    onSubmit={handleSavePersonalProfile}
                    className="xl:col-span-7 rounded-3xl border border-slate-200 bg-white p-6 shadow-sm space-y-5 text-xs"
                  >
                    <div className="flex items-center justify-between border-b border-slate-100 pb-3">
                      <h3 className="text-sm font-extrabold text-slate-900">{locale.profile.editFormTitle}</h3>
                    </div>

                    {/* 1. รูปประกอบ */}
                    <div className="space-y-3">
                      <label className="block font-extrabold text-slate-700">
                        {locale.profile.avatarSectionLabel}
                      </label>
                      <div className="flex flex-wrap items-center gap-4">
                        <img
                          src={profileAvatarUrl}
                          alt={profileFullName}
                          className="h-20 w-20 rounded-2xl object-cover border-2 border-slate-200 bg-white shrink-0"
                        />
                        <div className="space-y-2 flex-1 min-w-[200px]">
                          <label className="inline-flex items-center gap-2 rounded-xl border border-emerald-200 bg-emerald-50 px-3.5 py-2 font-bold text-emerald-700 hover:bg-emerald-100 cursor-pointer">
                            <Upload className="h-4 w-4 shrink-0" />
                            <span>{locale.profile.uploadPhotoBtn}</span>
                            <input
                              type="file"
                              accept="image/*"
                              onChange={handleAvatarFileChange}
                              className="hidden"
                            />
                          </label>
                          <div className="text-[11px] font-semibold text-slate-500">{locale.profile.presetLabel}</div>
                          <div className="flex flex-wrap items-center gap-2">
                            {PT_PRESET_AVATARS.map((preset) => (
                              <button
                                key={preset.id}
                                type="button"
                                onClick={() => {
                                  profileDirtyRef.current = true;
                                  setProfileAvatarUrl(preset.uri);
                                }}
                                className={`flex items-center gap-2 rounded-xl border p-1.5 pr-2.5 transition cursor-pointer ${profileAvatarUrl === preset.uri
                                    ? 'border-emerald-500 bg-emerald-50 ring-2 ring-emerald-200'
                                    : 'border-slate-200 bg-white hover:bg-slate-50'
                                  }`}
                              >
                                <img
                                  src={preset.uri}
                                  alt={preset.label}
                                  className="h-9 w-9 rounded-lg object-cover bg-white"
                                />
                                <span className="text-[11px] font-bold text-slate-700">{preset.label}</span>
                              </button>
                            ))}
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* 2. ชื่อ + เบอร์โทร */}
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                      <div>
                        <label className="block font-extrabold text-slate-700 mb-1.5">
                          {locale.profile.fullNameLabel} *
                        </label>
                        <input
                          required
                          value={profileFullName}
                          onChange={(e) => {
                            profileDirtyRef.current = true;
                            setProfileFullName(e.target.value);
                          }}
                          className="w-full rounded-xl border border-slate-200 bg-white px-3.5 py-2.5 text-sm font-semibold text-slate-900 focus:border-emerald-500 focus:outline-none"
                        />
                      </div>
                      <div>
                        <label className="block font-extrabold text-slate-700 mb-1.5">
                          {locale.profile.phoneLabel}
                        </label>
                        <input
                          type="tel"
                          inputMode="numeric"
                          pattern="[0-9]*"
                          maxLength={10}
                          value={profilePhone}
                          onChange={(e) => {
                            profileDirtyRef.current = true;
                            setProfilePhone(e.target.value.replace(/\D/g, '').slice(0, 10));
                          }}
                          placeholder="08XXXXXXXX"
                          className="w-full rounded-xl border border-slate-200 bg-white px-3.5 py-2.5 text-sm font-mono font-semibold text-slate-900 focus:border-emerald-500 focus:outline-none"
                        />
                      </div>
                    </div>

                    {activeUserRecord.role !== 'patient' && (
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                        <div>
                          <label className="block font-extrabold text-slate-700 mb-1.5">
                            {locale.profile.specialtyLabel}
                          </label>
                          <input
                            value={profileSpecialty}
                            onChange={(e) => {
                              profileDirtyRef.current = true;
                              setProfileSpecialty(e.target.value);
                            }}
                            className="w-full rounded-xl border border-slate-200 bg-white px-3.5 py-2.5 text-xs font-semibold text-slate-900 focus:border-emerald-500 focus:outline-none"
                          />
                        </div>
                        <div>
                          <label className="flex flex-wrap items-center justify-between gap-1.5 font-extrabold text-slate-700 mb-1.5">
                            <span>{locale.profile.licenseLabel}</span>
                            {!isAdminUser && (
                              <span className="inline-flex items-center gap-1 rounded-full border border-slate-200 bg-slate-100 px-2 py-0.5 text-[10px] font-bold text-slate-600">
                                <Lock className="h-3 w-3 shrink-0" />
                                ดูได้อย่างเดียว (แก้ไขได้เฉพาะผู้ดูแลระบบ Admin)
                              </span>
                            )}
                          </label>
                          <input
                            type="text"
                            value={!isAdminUser ? (activeUserRecord.licenseNo || myProfile.licenseNo) : profileLicenseNo}
                            readOnly={!isAdminUser}
                            disabled={!isAdminUser}
                            onChange={(e) => {
                              if (!isAdminUser) return;
                              profileDirtyRef.current = true;
                              setProfileLicenseNo(e.target.value);
                            }}
                            className={
                              !isAdminUser
                                ? 'w-full rounded-xl border border-slate-200 bg-slate-100 px-3.5 py-2.5 text-xs font-mono font-medium text-slate-600 cursor-not-allowed select-text focus:outline-none'
                                : 'w-full rounded-xl border border-slate-200 bg-white px-3.5 py-2.5 text-xs font-mono font-semibold text-slate-900 focus:border-emerald-500 focus:outline-none'
                            }
                          />
                          {!isAdminUser && (
                            <p className="mt-1 inline-flex items-center gap-1 text-[11px] font-semibold text-slate-500">
                              <Lock className="h-3 w-3 shrink-0" />
                              <span>ดูได้อย่างเดียว (แก้ไขได้เฉพาะผู้ดูแลระบบ Admin)</span>
                            </p>
                          )}
                        </div>
                      </div>
                    )}

                    {/* ประวัติส่วนตัว */}
                    <div>
                      <label className="block font-extrabold text-slate-700 mb-1.5">
                        {locale.profile.bioLabel}
                      </label>
                      <textarea
                        rows={5}
                        value={profileBio}
                        onChange={(e) => {
                          profileDirtyRef.current = true;
                          setProfileBio(e.target.value);
                        }}
                        placeholder={locale.profile.bioPlaceholder}
                        className="w-full rounded-2xl border border-slate-200 bg-white px-3.5 py-3 text-xs leading-relaxed text-slate-900 focus:border-emerald-500 focus:outline-none"
                      />
                    </div>

                    <div className="flex justify-end pt-2">
                      <button
                        type="submit"
                        disabled={submitting}
                        className="inline-flex items-center gap-2 rounded-xl bg-emerald-600 px-5 py-2.5 text-xs font-extrabold text-white shadow-sm hover:bg-emerald-500 disabled:opacity-50 cursor-pointer"
                      >
                        <Save className="h-4 w-4 shrink-0" />
                        <span>{submitting ? locale.profile.savingBtn : locale.profile.saveBtn}</span>
                      </button>
                    </div>
                  </form>
                </div>

                {/* ระบบการฝากเคสคนไข้ (Patient Case Delegation / Handover) */}
                <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm space-y-5">
                  <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-100 pb-4">
                    <div className="flex items-start gap-2.5">
                      <HeartHandshake className="h-5 w-5 text-sky-600 mt-0.5 shrink-0" />
                      <div>
                        <h3 className="text-base font-extrabold text-slate-900">
                          ระบบการฝากเคสคนไข้ (Patient Case Handover)
                        </h3>
                        <p className="text-xs text-slate-500">
                          ระบุชื่อคนไข้ วันที่ฝากเคส นักกายภาพบำบัดผู้รับฝากดูแล และหมายเหตุการส่งต่อเคส
                        </p>
                      </div>
                    </div>
                    <span className="inline-flex items-center gap-1.5 rounded-full border border-sky-200 bg-sky-50 px-3 py-1 text-xs font-bold text-sky-700">
                      <Users className="h-3.5 w-3.5 shrink-0" />
                      <span>เคสที่ฝากอยู่: {delegatedCases.length} เคส</span>
                    </span>
                  </div>

                  <div className="grid grid-cols-1 xl:grid-cols-12 gap-6 items-start">
                    {/* Left: ฟอร์มฝากเคสคนไข้ */}
                    <form
                      onSubmit={handleSubmitCaseDelegation}
                      className="xl:col-span-5 rounded-2xl border border-slate-200 bg-slate-50/70 p-5 space-y-4 text-xs"
                    >
                      <div className="font-extrabold text-sm text-slate-900">บันทึกการฝากเคสคนไข้ใหม่</div>

                      {/* 1. ชื่อคนไข้ให้เลือก */}
                      <div>
                        <label className="block font-extrabold text-slate-700 mb-1.5">
                          1. ชื่อคนไข้ที่ต้องการฝากเคส *
                        </label>
                        <select
                          required
                          value={delegationPatientId}
                          onChange={(e) => setDelegationPatientId(e.target.value)}
                          className="w-full rounded-xl border border-slate-200 bg-white px-3.5 py-2.5 text-sm font-semibold text-slate-900 focus:border-sky-500 focus:outline-none"
                        >
                          {patients.length === 0 ? (
                            <option value="">-- ยังไม่มีรายชื่อคนไข้ --</option>
                          ) : (
                            patients.map((p) => (
                              <option key={p.id} value={p.id}>
                                {p.fullName} ({p.hn}){p.delegatedToPtId ? ' • ฝากเคสอยู่' : ''}
                              </option>
                            ))
                          )}
                        </select>
                        {(() => {
                          const selP = patients.find((p) => p.id === delegationPatientId);
                          if (!selP) return null;
                          const ownerName = ptNameMap.get(selP.assignedPtId) || 'กภ. ประจำคลินิก';
                          return (
                            <div className="mt-2 rounded-xl border border-slate-200 bg-white px-3 py-2 text-[11px] text-slate-600 flex flex-wrap items-center justify-between gap-2">
                              <span className="truncate">อาการ/วินิจฉัย: <strong className="text-slate-800">{selP.diagnosis || 'กายภาพฟื้นฟู'}</strong></span>
                              <span className="text-sky-700 font-semibold">เจ้าของเคสหลัก: {ownerName}</span>
                            </div>
                          );
                        })()}
                      </div>

                      {/* 2. วันที่ฝากเคส */}
                      <div>
                        <label className="block font-extrabold text-slate-700 mb-1.5">
                          2. วันที่ฝากเคส *
                        </label>
                        <div className="relative">
                          <CalendarDays className="h-4 w-4 text-sky-600 absolute left-3.5 top-1/2 -translate-y-1/2 pointer-events-none" />
                          <input
                            required
                            type="date"
                            value={delegationDate}
                            onChange={(e) => setDelegationDate(e.target.value)}
                            className="w-full rounded-xl border border-slate-200 bg-white pl-10 pr-3.5 py-2.5 text-sm font-mono font-semibold text-slate-900 focus:border-sky-500 focus:outline-none"
                          />
                        </div>
                      </div>

                      {/* 3. ฝากเคสกับนักกายภาพคนไหน */}
                      <div>
                        <label className="block font-extrabold text-slate-700 mb-1.5">
                          3. ฝากเคสกับนักกายภาพคนไหน *
                        </label>
                        <select
                          required
                          value={delegationTargetPtId}
                          onChange={(e) => setDelegationTargetPtId(e.target.value)}
                          className="w-full rounded-xl border border-slate-200 bg-white px-3.5 py-2.5 text-sm font-semibold text-slate-900 focus:border-sky-500 focus:outline-none"
                        >
                          {ptUsers.length === 0 ? (
                            <option value="">-- ไม่พบรายชื่อนักกายภาพบำบัด --</option>
                          ) : (
                            ptUsers.map((pt) => {
                              const ptProf = resolveUserProfile(pt, profileOverrides);
                              return (
                                <option key={pt.id} value={pt.id}>
                                  {ptProf.fullName} ({ptProf.licenseNo})
                                </option>
                              );
                            })
                          )}
                        </select>
                        {(() => {
                          const targetPtObj = ptUsers.find((pt) => pt.id === delegationTargetPtId);
                          if (!targetPtObj) return null;
                          const targetPtProf = resolveUserProfile(targetPtObj, profileOverrides);
                          return (
                            <div className="mt-2.5 flex items-center gap-3 rounded-xl border border-emerald-200 bg-emerald-50/70 p-2.5">
                              <img
                                src={targetPtProf.avatarUrl}
                                alt={targetPtProf.fullName}
                                className="h-11 w-11 shrink-0 rounded-xl object-cover border border-emerald-200 bg-white"
                              />
                              <div className="min-w-0 flex-1">
                                <div className="font-bold text-slate-900 truncate">{targetPtProf.fullName}</div>
                                <div className="text-[11px] text-emerald-800 font-mono truncate">
                                  {targetPtProf.phone || locale.profile.noPhone} • {targetPtProf.specialty}
                                </div>
                              </div>
                            </div>
                          );
                        })()}
                      </div>

                      {/* 4. หมายเหตุ */}
                      <div>
                        <label className="block font-extrabold text-slate-700 mb-1.5">
                          4. หมายเหตุการฝากเคส
                        </label>
                        <textarea
                          rows={3}
                          value={delegationNote}
                          onChange={(e) => setDelegationNote(e.target.value)}
                          placeholder="ระบุอาการที่ต้องติดตามเป็นพิเศษ ช่วงเวลาที่ฝาก หรือท่ากายภาพที่ต้องเฝ้าระวัง..."
                          className="w-full rounded-xl border border-slate-200 bg-white px-3.5 py-2.5 text-xs leading-relaxed text-slate-900 focus:border-sky-500 focus:outline-none"
                        />
                      </div>

                      <button
                        type="submit"
                        disabled={submitting || patients.length === 0 || ptUsers.length === 0}
                        className="w-full inline-flex items-center justify-center gap-2 rounded-xl bg-sky-600 px-4 py-2.5 text-xs font-extrabold text-white shadow-sm hover:bg-sky-500 disabled:opacity-50 cursor-pointer"
                      >
                        <CheckCircle2 className="h-4 w-4 shrink-0" />
                        <span>{submitting ? 'กำลังบันทึก...' : 'ยืนยันการฝากเคสคนไข้'}</span>
                      </button>
                    </form>

                    {/* Right: รายการฝากเคสคนไข้ปัจจุบัน */}
                    <div className="xl:col-span-7 space-y-3">
                      <div className="flex items-center justify-between">
                        <h4 className="text-sm font-extrabold text-slate-900">
                          รายการเคสคนไข้ที่ฝากดูแลอยู่ในขณะนี้
                        </h4>
                        <span className="text-[11px] font-semibold text-slate-500">
                          แสดงข้อมูลคนไข้ วันที่ฝาก นักกายภาพผู้รับเคส และหมายเหตุ
                        </span>
                      </div>

                      {delegatedCases.length === 0 ? (
                        <div className="rounded-2xl border-2 border-dashed border-slate-200 bg-slate-50/50 p-8 text-center space-y-2">
                          <HeartHandshake className="h-8 w-8 text-slate-300 mx-auto" />
                          <div className="text-sm font-bold text-slate-600">ยังไม่มีรายการฝากเคสคนไข้ในขณะนี้</div>
                          <p className="text-xs text-slate-400">
                            เลือกชื่อคนไข้ วันที่ฝากเคส นักกายภาพบำบัดผู้รับเคส และหมายเหตุจากฟอร์มด้านซ้ายเพื่อบันทึกการฝากเคส
                          </p>
                        </div>
                      ) : (
                        <div className="space-y-3">
                          {delegatedCases.map((p) => {
                            const targetPtUser = ptUsers.find((pt) => pt.id === p.delegatedToPtId);
                            const targetPtProf = targetPtUser
                              ? resolveUserProfile(targetPtUser, profileOverrides)
                              : null;
                            const targetPtDisplayName =
                              targetPtProf?.fullName ||
                              ptNameMap.get(p.delegatedToPtId || '') ||
                              'กภ. ผู้รับฝากเคส';
                            const primaryPtDisplayName =
                              ptNameMap.get(p.assignedPtId) || 'กภ. ประจำคลินิก';
                            return (
                              <div
                                key={p.id}
                                className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm hover:border-sky-200 transition space-y-3"
                              >
                                <div className="flex flex-wrap items-start justify-between gap-2">
                                  <div className="flex items-center gap-2.5 min-w-0">
                                    <span className="rounded-lg border border-sky-200 bg-sky-50 px-2.5 py-1 font-mono text-[11px] font-extrabold text-sky-700 shrink-0">
                                      {p.hn}
                                    </span>
                                    <div className="min-w-0">
                                      <div className="text-sm font-extrabold text-slate-900 truncate">
                                        {p.fullName}
                                      </div>
                                      <div className="text-[11px] text-slate-500 truncate">
                                        {p.diagnosis || 'กายภาพบำบัดฟื้นฟูสมรรถภาพ'} • เจ้าของเคสหลัก: {primaryPtDisplayName}
                                      </div>
                                    </div>
                                  </div>

                                  <span className="inline-flex items-center gap-1.5 rounded-full border border-amber-200 bg-amber-50 px-3 py-1 text-[11px] font-extrabold text-amber-900 shrink-0">
                                    <CalendarDays className="h-3.5 w-3.5 text-amber-600 shrink-0" />
                                    <span>วันที่ฝากเคส: {p.delegationDate || 'ไม่ระบุวันที่'}</span>
                                  </span>
                                </div>

                                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                                  <div className="flex items-center gap-2.5 rounded-xl border border-emerald-100 bg-emerald-50/60 p-2.5">
                                    {targetPtProf && (
                                      <img
                                        src={targetPtProf.avatarUrl}
                                        alt={targetPtDisplayName}
                                        className="h-10 w-10 shrink-0 rounded-xl object-cover border border-emerald-200 bg-white"
                                      />
                                    )}
                                    <div className="min-w-0 flex-1 text-xs">
                                      <div className="text-[10px] font-bold uppercase tracking-wider text-emerald-700">
                                        ฝากเคสกับนักกายภาพ
                                      </div>
                                      <div className="font-extrabold text-slate-900 truncate">
                                        {targetPtDisplayName}
                                      </div>
                                      {targetPtProf?.phone && (
                                        <div className="flex items-center gap-1 font-mono text-[11px] text-slate-600 truncate">
                                          <Phone className="h-3 w-3 text-emerald-600 shrink-0" />
                                          <span>{targetPtProf.phone}</span>
                                        </div>
                                      )}
                                    </div>
                                  </div>

                                  <div className="rounded-xl border border-slate-100 bg-slate-50/80 p-2.5 text-xs">
                                    <div className="flex items-center gap-1 text-[10px] font-bold uppercase tracking-wider text-slate-500">
                                      <FileText className="h-3 w-3 text-slate-400 shrink-0" />
                                      <span>หมายเหตุ</span>
                                    </div>
                                    <div className="mt-1 text-slate-800 leading-relaxed break-words">
                                      {p.delegationNote || 'ไม่มีหมายเหตุเพิ่มเติม'}
                                    </div>
                                  </div>
                                </div>

                                <div className="flex flex-wrap items-center justify-end gap-2 pt-1 border-t border-slate-100">
                                  <button
                                    type="button"
                                    onClick={() => {
                                      setDelegationPatientId(p.id);
                                      setDelegationDate(p.delegationDate || formatLocalIsoDate());
                                      setDelegationTargetPtId(p.delegatedToPtId || '');
                                      setDelegationNote(p.delegationNote || '');
                                    }}
                                    className="rounded-xl border border-sky-200 bg-sky-50 px-3 py-1.5 text-[11px] font-bold text-sky-700 hover:bg-sky-100 cursor-pointer"
                                  >
                                    แก้ไขข้อมูลฝากเคส
                                  </button>
                                  <button
                                    type="button"
                                    disabled={submitting}
                                    onClick={() => void handleCancelCaseDelegation(p)}
                                    className="inline-flex items-center gap-1 rounded-xl border border-rose-200 bg-rose-50 px-3 py-1.5 text-[11px] font-bold text-rose-700 hover:bg-rose-100 disabled:opacity-50 cursor-pointer"
                                  >
                                    <X className="h-3.5 w-3.5 shrink-0" />
                                    <span>ยกเลิกฝากเคส / รับเคสคืน</span>
                                  </button>
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </main>
      </div>

      {/* Staff-Assisted Enrollment / Existing Patient Face Scan Modal */}
      <StaffAssistedEnrollmentModal
        open={enrollModalOpen}
        apiClient={apiClient}
        currentUser={currentUser}
        ptList={ptUsers}
        exercises={exercises}
        modpacks={modpacks}
        targetPatient={enrollTargetPatient}
        onClose={() => {
          setEnrollModalOpen(false);
          setEnrollTargetPatient(null);
        }}
        onEnrolled={(p) => {
          const hydratedEnrolled = normalizePatientRecord(p, {
            requireRealVectorsForVerified: false,
          });
          setRawPatients((prev) => [
            hydratedEnrolled,
            ...prev.filter(
              (item) =>
                item.id !== hydratedEnrolled.id &&
                normalizeHnKey(item.hn) !== normalizeHnKey(hydratedEnrolled.hn),
            ),
          ]);
          setUsers((prev) => upsertPatientIntoStaffUsersState(prev, hydratedEnrolled));
          void upsertPatientStaffUserInSupabase(apiClient.getSupabaseClient(), hydratedEnrolled);
          setPatientSyncError(null);
          const rxCount = hydratedEnrolled.rehabSchedules?.length ?? 0;
          flashBanner(
            'success',
            rxCount > 0
              ? `ลงทะเบียนคนไข้ ${hydratedEnrolled.fullName} (${hydratedEnrolled.hn}) พร้อมกำหนดท่ากายภาพ ${rxCount} รายการสำเร็จ!`
              : `ลงทะเบียนคนไข้และบันทึก Face ID 120 เฟรมสำเร็จ! (${hydratedEnrolled.fullName} - ${hydratedEnrolled.hn})`,
          );
          void syncPatientsFromSupabase({ silent: true });
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
        onSuccess={(msg) => {
          flashBanner('success', msg);
          void refreshAll();
        }}
      />

      {/* Patient Profile, Medical Notes & Rehab History Modal */}
      <PatientProfileAndHistoryModal
        patient={activeModalPatient}
        initialTab={patientModalState.tab}
        apiClient={apiClient}
        currentUser={currentUser}
        ptUsers={ptUsers}
        sessions={sessions}
        onClose={() => setPatientModalState({ patientId: null, tab: 'profile' })}
        onRefresh={refreshAll}
        onOpenFaceScan={(p) => {
          setEnrollTargetPatient(p);
          setEnrollModalOpen(true);
        }}
        onOpenLinkEhr={(p) => {
          setPatientModalState({ patientId: null, tab: 'profile' });
          setEhrLinkPatient(p);
        }}
        onFlash={flashBanner}
      />

      <UniversalEhrLinkModal
        patient={activeEhrLinkPatient}
        apiClient={apiClient}
        onClose={() => setEhrLinkPatient(null)}
        onLinked={refreshAll}
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
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-[#0F4C81]/35 p-4 backdrop-blur-sm animate-backdrop-fade">
          <div className="w-full max-w-md rounded-3xl bg-white p-6 shadow-2xl border border-slate-200 space-y-4 animate-popup-spring">
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
                    type="tel"
                    inputMode="numeric"
                    pattern="[0-9]*"
                    maxLength={10}
                    value={newStaffPhone}
                    onChange={(e) => setNewStaffPhone(e.target.value.replace(/\D/g, '').slice(0, 10))}
                    placeholder="08XXXXXXXX"
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
      {selectedPtDetailModal && (() => {
        const modalPtProfile = resolveUserProfile(selectedPtDetailModal, profileOverrides);
        return (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-[#0F4C81]/35 p-4 backdrop-blur-sm animate-backdrop-fade">
            <div className="w-full max-w-lg rounded-3xl bg-white p-6 shadow-2xl border border-slate-200 space-y-4 text-xs animate-popup-spring">
              <div className="flex items-start justify-between gap-3">
                <div className="flex items-start gap-3 min-w-0">
                  <img
                    src={modalPtProfile.avatarUrl}
                    alt={modalPtProfile.fullName}
                    className="h-16 w-16 shrink-0 rounded-2xl object-cover border-2 border-emerald-200 bg-white shadow-sm"
                  />
                  <div className="min-w-0">
                    <h3 className="text-base font-black text-slate-900">{modalPtProfile.fullName}</h3>
                    <p className="text-slate-500 flex flex-wrap items-center gap-1.5">
                      <span>@{selectedPtDetailModal.username} • {modalPtProfile.licenseNo} • {modalPtProfile.specialty}</span>
                      {isAdminUser && (
                        <button
                          type="button"
                          onClick={() => openEditStaffModal(selectedPtDetailModal)}
                          className="inline-flex items-center gap-1 rounded-lg border border-amber-200 bg-amber-50 px-2 py-0.5 text-[10px] font-bold text-amber-800 hover:bg-amber-100 cursor-pointer"
                        >
                          <Pencil className="h-3 w-3" />
                          <span>แก้ไขใบอนุญาต / ข้อมูล</span>
                        </button>
                      )}
                    </p>
                    <p className="mt-0.5 flex items-center gap-1 font-mono text-emerald-700 font-semibold">
                      <Phone className="h-3 w-3 shrink-0" />
                      <span>{modalPtProfile.phone || locale.profile.noPhone}</span>
                    </p>
                  </div>
                </div>
                <button type="button" onClick={() => setSelectedPtDetailModal(null)} className="text-slate-400">
                  <X className="h-5 w-5" />
                </button>
              </div>
              <p className="rounded-xl border border-slate-100 bg-slate-50 px-3.5 py-2.5 text-xs leading-relaxed text-slate-700">
                {modalPtProfile.bio}
              </p>

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
                        } catch (err) {
                          flashBanner(
                            'error',
                            err instanceof Error ? err.message : 'มอบหมายคนไข้ไม่สำเร็จ',
                          );
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
        );
      })()}

      {/* Admin Staff Profile & License Edit Modal */}
      {editingStaffUser && isAdminUser && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-[#0F4C81]/35 p-4 backdrop-blur-sm animate-backdrop-fade">
          <div className="w-full max-w-md rounded-3xl bg-white p-6 shadow-2xl border border-slate-200 space-y-4 text-xs animate-popup-spring">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Pencil className="h-4 w-4 text-amber-600" />
                <div>
                  <h3 className="text-sm font-black text-slate-900">
                    แก้ไขข้อมูลบุคลากรและใบประกอบวิชาชีพ
                  </h3>
                  <p className="text-[11px] text-slate-500 font-mono">@{editingStaffUser.username}</p>
                </div>
              </div>
              <button
                type="button"
                onClick={() => setEditingStaffUser(null)}
                className="text-slate-400 hover:text-slate-600 cursor-pointer"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
            <form onSubmit={handleSaveEditedStaff} className="space-y-3">
              <div>
                <label className="block font-bold text-slate-600 mb-1">ชื่อ - นามสกุล *</label>
                <input
                  required
                  value={editStaffFullName}
                  onChange={(e) => setEditStaffFullName(e.target.value)}
                  className="w-full rounded-xl border border-slate-200 px-3 py-2 font-semibold text-slate-900"
                />
              </div>
              <div>
                <label className="block font-bold text-slate-600 mb-1">เบอร์โทรศัพท์ติดต่อ</label>
                <input
                  type="tel"
                  inputMode="numeric"
                  pattern="[0-9]*"
                  maxLength={10}
                  value={editStaffPhone}
                  onChange={(e) => setEditStaffPhone(e.target.value.replace(/\D/g, '').slice(0, 10))}
                  placeholder="08XXXXXXXX"
                  className="w-full rounded-xl border border-slate-200 px-3 py-2 font-mono text-slate-900"
                />
              </div>
              {editingStaffUser.role !== 'patient' && (
                <>
                  <div>
                    <label className="block font-bold text-slate-600 mb-1">ความเชี่ยวชาญเฉพาะทาง</label>
                    <input
                      value={editStaffSpecialty}
                      onChange={(e) => setEditStaffSpecialty(e.target.value)}
                      className="w-full rounded-xl border border-slate-200 px-3 py-2 text-slate-900"
                    />
                  </div>
                  <div>
                    <label className="flex items-center justify-between font-bold text-slate-600 mb-1">
                      <span>เลขที่ใบประกอบวิชาชีพ</span>
                      <span className={isAdminUser ? 'text-[10px] font-semibold text-emerald-700' : 'inline-flex items-center gap-1 text-[10px] font-semibold text-slate-500'}>
                        {!isAdminUser && <Lock className="h-3 w-3 shrink-0" />}
                        {isAdminUser ? 'Admin แก้ไขได้' : 'ดูได้อย่างเดียว (แก้ไขได้เฉพาะผู้ดูแลระบบ Admin)'}
                      </span>
                    </label>
                    <input
                      value={editStaffLicenseNo}
                      readOnly={!isAdminUser}
                      disabled={!isAdminUser}
                      onChange={(e) => {
                        if (!isAdminUser) return;
                        setEditStaffLicenseNo(e.target.value);
                      }}
                      placeholder="เช่น PT-44012"
                      className={
                        !isAdminUser
                          ? 'w-full rounded-xl border border-slate-200 bg-slate-100 px-3 py-2 font-mono font-medium text-slate-600 cursor-not-allowed select-text focus:outline-none'
                          : 'w-full rounded-xl border border-emerald-300 bg-white px-3 py-2 font-mono font-bold text-slate-900 focus:border-emerald-500 focus:outline-none'
                      }
                    />
                  </div>
                </>
              )}
              <div className="flex justify-end gap-2 pt-2">
                <button
                  type="button"
                  onClick={() => setEditingStaffUser(null)}
                  className="rounded-xl border border-slate-200 px-4 py-2 font-bold text-slate-600 cursor-pointer"
                >
                  ยกเลิก
                </button>
                <button
                  type="submit"
                  disabled={submitting}
                  className="rounded-xl bg-emerald-600 px-4 py-2 font-extrabold text-white hover:bg-emerald-500 disabled:opacity-50 cursor-pointer"
                >
                  {submitting ? 'กำลังบันทึก...' : 'บันทึกการแก้ไข'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Quick Check-in Modal */}
      {quickCheckinModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-[#0F4C81]/35 p-4 backdrop-blur-sm animate-backdrop-fade">
          <div className="w-full max-w-md rounded-3xl bg-white p-6 shadow-2xl border border-slate-200 space-y-4 text-xs animate-popup-spring">
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
                        } catch (err) {
                          flashBanner('error', err instanceof Error ? err.message : 'เช็คอินไม่สำเร็จ');
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
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-[#0F4C81]/35 p-4 backdrop-blur-sm animate-backdrop-fade">
          <div className="w-full max-w-sm rounded-3xl bg-white p-6 shadow-2xl border border-slate-200 space-y-4 text-xs animate-popup-spring">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-black text-slate-900">+ ลงทะเบียนตู้ Kiosk ใหม่</h3>
              <button type="button" onClick={() => setRegisterKioskModalOpen(false)} className="text-slate-400">
                <X className="h-4 w-4" />
              </button>
            </div>
            <form
              onSubmit={async (e) => {
                e.preventDefault();
                try {
                  const created = await apiClient.registerKiosk({
                    kioskId: newKioskCode,
                    location: newKioskLocation,
                  });
                  setRegisterKioskModalOpen(false);
                  flashBanner('success', `ลงทะเบียน ${created.kioskId} สำเร็จ (Token: ${created.pairingToken})`);
                  await refreshAll();
                } catch (err) {
                  flashBanner('error', err instanceof Error ? err.message : 'ลงทะเบียนตู้ Kiosk ไม่สำเร็จ');
                }
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

      {/* Physical Card Game Modal — Ultra-Compact Header + Full-Size Kiosk View */}
      {cardGameModalOpen && (
        <div className="fixed inset-0 z-50 flex flex-col w-screen h-screen bg-slate-100 overflow-y-auto overflow-x-hidden animate-backdrop-fade">
          <div className="flex items-center justify-between gap-2 bg-white border-b border-slate-200 px-4 py-2 shrink-0 shadow-xs">
            <div className="flex items-center gap-2 min-w-0">
              <Gamepad2 className="h-5 w-5 text-emerald-600 shrink-0" />
              <h3 className="text-sm md:text-base font-bold text-slate-900 truncate">
                เกมการ์ดกายภาพบำบัด 10 ข้อ (ทดสอบระบบมือ/กล้อง หรือคลิกปุ่มตอบ)
              </h3>
            </div>
            <button
              type="button"
              onClick={() => setCardGameModalOpen(false)}
              aria-label="ปิดหน้าต่าง"
              className="flex h-7 w-7 items-center justify-center rounded-lg bg-slate-100 text-sm text-slate-600 hover:bg-rose-100 hover:text-rose-700 cursor-pointer shrink-0 transition"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
          <div className="flex-1 flex flex-col w-full min-h-0 p-2 sm:p-3">
            <PhysicalCardGame
              patientName={selectedSchedulePatient?.fullName ?? currentUser.fullName}
              onSessionFinished={(summary) => {
                // Staff sandbox test mode: display summary only — NEVER write synthetic test rows into a real patient's clinical record
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

      {/* Standalone Pose Test Sandbox Modal (when ClinicalStaffPortalShell is rendered outside App.tsx router) */}
      {poseTestOverlayExerciseCode !== undefined && (
        <div
          data-testid="standalone-pose-test-modal"
          className="fixed inset-0 z-50 w-screen h-screen bg-slate-100 overflow-y-auto overflow-x-hidden animate-backdrop-fade"
        >
          <SmartRehabKioskUI
            isPoseTestMode
            initialExerciseCode={poseTestOverlayExerciseCode}
            patient={POSE_TEST_PATIENT}
            ownerBiometrics={null}
            prescribedExercises={buildAll15ClinicalExercisePrescriptions(exercises)}
            isStaffAuthorized
            onBackToAdminPortal={() => setPoseTestOverlayExerciseCode(undefined)}
          />
        </div>
      )}
    </div>
  );
}
