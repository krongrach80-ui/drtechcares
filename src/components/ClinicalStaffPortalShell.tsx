import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
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
}

type PortalTab = 'staff' | 'patients' | 'pts' | 'exercises' | 'schedules' | 'security';
type SecuritySubTab = 'calls' | 'audit' | 'sessions' | 'bans' | 'kiosks';

const PORTAL_TEXT: Record<
  ClinicalLang,
  {
    nav: Record<PortalTab, string>;
    headers: Record<PortalTab, { title: string; subtitle: string }>;
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
  }
> = {
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
    actions: {
      welcomeScan: 'ต้อนรับ & สแกนหน้า',
      quickCheckin: 'เช็คอินเร่งด่วน',
      kioskBtn: 'หน้าตู้ Kiosk',
      returnKiosk: '← หน้าตู้ Kiosk',
      logout: 'ออกจากระบบ',
      cardGame: 'การ์ดกายภาพ (10 ข้อ)',
      cloudConnected: 'Supabase Cloud',
      cloudOffline: 'Offline (คลิกเพื่อลองใหม่)',
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
    actions: {
      welcomeScan: 'Welcome & Face Scan',
      quickCheckin: 'Quick Check-in',
      kioskBtn: 'Kiosk View',
      returnKiosk: '← Back to Kiosk',
      logout: 'Sign Out',
      cardGame: 'Rehab Cards (10 Qs)',
      cloudConnected: 'Supabase Cloud',
      cloudOffline: 'Offline (Retry)',
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
    actions: {
      welcomeScan: '迎宾人脸识别',
      quickCheckin: '快速签到',
      kioskBtn: '终端界面',
      returnKiosk: '← 返回终端画面',
      logout: '退出登录',
      cardGame: '康复认知卡 (10题)',
      cloudConnected: 'Supabase 云端在线',
      cloudOffline: '离线 (点击重试)',
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

const WEEKDAY_NAMES_TH = ['อาทิตย์', 'จันทร์', 'อังคาร', 'พุธ', 'พฤหัสบดี', 'ศุกร์', 'เสาร์'];
const CALENDAR_HEADERS_TH = ['อา.', 'จ.', 'อ.', 'พ.', 'พฤ.', 'ศ.', 'ส.'];

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
      const [uRes, pRes, exRes, mpRes, sessRes, callsRes, settingsRes, secRes, healthRes] = await Promise.all([
        apiClient.listUsers().catch(() => []),
        apiClient.listPatients().catch(() => []),
        apiClient.listExercises().catch(() => []),
        apiClient.listModpacks().catch(() => []),
        apiClient.listRehabSessions().catch(() => []),
        apiClient.listStaffCalls().catch(() => []),
        apiClient.getSystemSettings().catch(() => null),
        isAdmin ? apiClient.listSecurityAudit().catch(() => null) : Promise.resolve(null),
        apiClient.checkHealth().catch(() => ({ ok: false, supabaseConnected: false, latencyMs: 0 })),
      ]);

      setUsers(uRes);
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
  }, [apiClient, isAdmin, selectedSchedulePatientId, courseModpackId]);

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
          sidebarCollapsed ? 'w-20' : 'w-[260px]'
        } shrink-0 bg-[#0b1f33] text-slate-100 flex flex-col justify-between border-r border-slate-800/80 transition-all duration-200 select-none`}
      >
        <div>
          <div className="flex items-center justify-between gap-2 px-4 py-5 border-b border-white/10">
            <div className="flex items-center gap-3 min-w-0">
              <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-gradient-to-br from-emerald-400 to-cyan-500 text-slate-950 shadow-md">
                <HeartHandshake className="h-5 w-5" />
              </div>
              {!sidebarCollapsed && (
                <div className="min-w-0">
                  <div className="text-sm font-black tracking-tight text-white truncate">Dr.Tech.Care</div>
                  <div className="mt-1 inline-flex items-center gap-1.5 rounded-full bg-emerald-500/20 border border-emerald-400/30 px-2 py-0.5 text-[10px] font-bold text-emerald-300">
                    <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
                    {isAdmin ? 'ผู้อำนวยการ (Admin)' : 'นักกายภาพบำบัด (PT)'}
                  </div>
                </div>
              )}
            </div>
            <button
              type="button"
              onClick={() => setSidebarCollapsed((c) => !c)}
              className="rounded-lg p-1.5 text-slate-400 hover:bg-white/10 hover:text-white cursor-pointer"
              title="ย่อ/ขยายเมนู"
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
                  className={`w-full flex items-center justify-between gap-2.5 rounded-xl px-3.5 py-2.5 text-left transition cursor-pointer ${
                    isActive
                      ? 'bg-gradient-to-r from-emerald-500 to-sky-500 text-white font-bold shadow-md shadow-emerald-950/30'
                      : 'text-slate-300 hover:bg-white/5 hover:text-white'
                  }`}
                >
                  <span className="flex items-center gap-2.5 min-w-0">
                    <Icon className="h-4 w-4 shrink-0" />
                    {!sidebarCollapsed && <span className="truncate">{item.label}</span>}
                  </span>
                  {!sidebarCollapsed && item.badge ? (
                    <span className="flex h-5 min-w-[20px] items-center justify-center rounded-full bg-amber-400 px-1.5 text-[10px] font-black text-slate-950">
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
            className="w-full flex items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-emerald-500 to-blue-600 px-3 py-2.5 font-extrabold text-white shadow hover:opacity-95 cursor-pointer"
          >
            <Gamepad2 className="h-4 w-4 shrink-0" />
            {!sidebarCollapsed && <span>{locale.actions.cardGame}</span>}
          </button>

          {!sidebarCollapsed && (
            <div className="flex items-center justify-between rounded-xl border border-white/10 bg-slate-900/60 px-3 py-2 text-[11px]">
              <span className="text-slate-400">ฐานข้อมูล: Cloud</span>
              <span
                className={`inline-flex items-center gap-1 font-bold ${
                  cloudHealth.online ? 'text-emerald-400' : 'text-rose-400'
                }`}
              >
                <span className={`h-2 w-2 rounded-full ${cloudHealth.online ? 'bg-emerald-400' : 'bg-rose-400'}`} />
                {cloudHealth.online ? 'ปกติ' : 'ออฟไลน์'}
              </span>
            </div>
          )}

          {onReturnToKiosk && (
            <button
              type="button"
              onClick={onReturnToKiosk}
              className="w-full flex items-center justify-center gap-2 rounded-xl border border-white/10 bg-white/5 px-3 py-2 font-bold text-slate-200 hover:bg-white/10 cursor-pointer"
            >
              <Monitor className="h-4 w-4 shrink-0" />
              {!sidebarCollapsed && <span>{locale.actions.returnKiosk}</span>}
            </button>
          )}

          <button
            type="button"
            onClick={() => void onLogout()}
            className="w-full flex items-center justify-center gap-2 rounded-xl border border-rose-500/20 bg-rose-500/10 px-3 py-2 font-bold text-rose-300 hover:bg-rose-500/20 cursor-pointer"
          >
            <LogOut className="h-4 w-4 shrink-0" />
            {!sidebarCollapsed && <span>{locale.actions.logout}</span>}
          </button>
        </div>
      </aside>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Top White Header */}
        <header className="bg-white border-b border-slate-200 px-5 py-3.5 flex flex-wrap items-center justify-between gap-3">
          <div>
            <h1 className="text-base sm:text-lg font-black text-slate-900 leading-tight">{currentHeader.title}</h1>
            <p className="text-[10px] font-bold tracking-wider text-slate-400 uppercase mt-0.5">
              {currentHeader.subtitle}
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2 text-xs">
            {/* Language Toggle */}
            <div className="flex items-center rounded-full border border-slate-200 bg-slate-50 p-0.5 font-bold">
              {(['cn', 'th', 'en'] as const).map((code) => (
                <button
                  key={code}
                  type="button"
                  onClick={() => setLang(code)}
                  className={`rounded-full px-2.5 py-1 text-[11px] transition cursor-pointer ${
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
                  ? `Supabase Latency: ${cloudHealth.latencyMs}ms (คลิกเพื่อทดสอบการเชื่อมต่อ)`
                  : 'คลิกเพื่อทดสอบ Supabase Cloud'
              }
              className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1.5 font-bold cursor-pointer transition ${
                cloudHealth.online
                  ? 'border-emerald-200 bg-emerald-50 text-emerald-700 hover:bg-emerald-100'
                  : 'border-rose-200 bg-rose-50 text-rose-700 hover:bg-rose-100'
              }`}
            >
              {cloudHealth.online ? <Wifi className="h-3.5 w-3.5" /> : <WifiOff className="h-3.5 w-3.5" />}
              {cloudHealth.online ? locale.actions.cloudConnected : locale.actions.cloudOffline}
            </button>

            <button
              type="button"
              onClick={() => setWelcomeFaceModalOpen(true)}
              className="inline-flex items-center gap-1.5 rounded-full bg-emerald-600 px-3.5 py-1.5 font-bold text-white shadow-sm hover:bg-emerald-500 cursor-pointer"
            >
              <ScanFace className="h-3.5 w-3.5" />
              {locale.actions.welcomeScan}
            </button>

            <button
              type="button"
              onClick={() => setQuickCheckinModalOpen(true)}
              className="inline-flex items-center gap-1.5 rounded-full border border-amber-300 bg-amber-50 px-3.5 py-1.5 font-bold text-amber-800 hover:bg-amber-100 cursor-pointer"
            >
              <Zap className="h-3.5 w-3.5 text-amber-500" />
              {locale.actions.quickCheckin}
            </button>

            {onReturnToKiosk && (
              <button
                type="button"
                onClick={onReturnToKiosk}
                className="inline-flex items-center gap-1.5 rounded-full border border-emerald-200 bg-white px-3.5 py-1.5 font-bold text-emerald-700 hover:bg-emerald-50 cursor-pointer"
              >
                <Monitor className="h-3.5 w-3.5" />
                {locale.actions.kioskBtn}
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
                <div className="flex items-start gap-2.5">
                  <Users className="h-5 w-5 text-emerald-600 mt-0.5 shrink-0" />
                  <div>
                    <h2 className="text-lg font-extrabold text-slate-900">
                      จัดการเจ้าหน้าที่โรงพยาบาล (Staff Management)
                    </h2>
                    <p className="text-xs text-slate-500">
                      บัญชีผู้อำนวยการ (Admin) และนักกายภาพบำบัด (PT) ควบคุมสิทธิ์และเซสชันการเข้าถึง
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() => void refreshAll()}
                    className="flex h-9 w-9 items-center justify-center rounded-xl border border-slate-200 bg-white text-slate-600 hover:bg-slate-50 cursor-pointer"
                    title="รีเฟรช"
                  >
                    <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
                  </button>
                  {isAdmin && (
                    <button
                      type="button"
                      onClick={() => setAddStaffModalOpen(true)}
                      className="inline-flex items-center gap-1.5 rounded-xl bg-emerald-600 px-4 py-2 text-xs font-extrabold text-white shadow-sm hover:bg-emerald-500 cursor-pointer"
                    >
                      <Plus className="h-4 w-4" />
                      เพิ่มเจ้าหน้าที่ใหม่
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
                    placeholder="ค้นหาชื่อ, username, เบอร์โทร..."
                    className="w-full rounded-xl border border-slate-200 bg-white pl-10 pr-4 py-2 text-xs text-slate-800 placeholder:text-slate-400 focus:border-emerald-500 focus:outline-none"
                  />
                </div>
                <select
                  value={staffRoleFilter}
                  onChange={(e) => setStaffRoleFilter(e.target.value as 'all' | UserRole)}
                  className="rounded-xl border border-slate-200 bg-white px-3.5 py-2 text-xs font-semibold text-slate-700"
                >
                  <option value="all">ทุกบทบาท (All Roles)</option>
                  <option value="super_admin">ผู้อำนวยการ (Admin)</option>
                  <option value="pt">นักกายภาพบำบัด (PT)</option>
                  <option value="patient">คนไข้ (Patient)</option>
                </select>
                <select
                  value={staffStatusFilter}
                  onChange={(e) => setStaffStatusFilter(e.target.value as 'all' | 'active' | 'inactive')}
                  className="rounded-xl border border-slate-200 bg-white px-3.5 py-2 text-xs font-semibold text-slate-700"
                >
                  <option value="all">ทุกสถานะ (All Status)</option>
                  <option value="active">ปกติ (Active)</option>
                  <option value="inactive">ระงับใช้งาน (Suspended)</option>
                </select>
              </div>

              <div className="rounded-2xl border border-slate-200 bg-white shadow-sm overflow-x-auto">
                <table className="w-full min-w-[760px] text-left text-xs">
                  <thead className="border-b border-slate-100 bg-slate-50/70 text-[11px] font-bold text-slate-500 uppercase">
                    <tr>
                      <th className="px-5 py-3.5">ชื่อ-นามสกุล</th>
                      <th className="px-4 py-3.5">USERNAME / HN</th>
                      <th className="px-4 py-3.5">บทบาท</th>
                      <th className="px-4 py-3.5">สถานะ</th>
                      {isAdmin && <th className="px-5 py-3.5 text-right">การจัดการ</th>}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {filteredStaff.map((u) => (
                      <tr key={u.id} className="hover:bg-slate-50/80 transition">
                        <td className="px-5 py-3.5">
                          <div className="flex items-center gap-3">
                            <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-indigo-50 text-xs font-extrabold text-indigo-700">
                              {u.fullName.trim().slice(0, 1) || 'U'}
                            </div>
                            <div>
                              <div className="font-bold text-slate-900">{u.fullName}</div>
                              <div className="text-[11px] text-slate-400">{u.phone || '08X-XXX-XXXX'}</div>
                            </div>
                          </div>
                        </td>
                        <td className="px-4 py-3.5 font-mono font-bold text-sky-600">@{u.username}</td>
                        <td className="px-4 py-3.5">
                          {u.role === 'super_admin' && (
                            <span className="inline-flex items-center gap-1 rounded-full border border-purple-200 bg-purple-50 px-2.5 py-1 text-[11px] font-bold text-purple-700">
                              <Shield className="h-3 w-3" /> ผู้อำนวยการ (Admin)
                            </span>
                          )}
                          {u.role === 'pt' && (
                            <span className="inline-flex items-center gap-1 rounded-full border border-emerald-200 bg-emerald-50 px-2.5 py-1 text-[11px] font-bold text-emerald-700">
                              <UserCheck className="h-3 w-3" /> นักกายภาพบำบัด (PT)
                            </span>
                          )}
                          {u.role === 'patient' && (
                            <span className="inline-flex items-center gap-1 rounded-full border border-sky-200 bg-sky-50 px-2.5 py-1 text-[11px] font-bold text-sky-700">
                              <Users className="h-3 w-3" /> คนไข้ (Patient)
                            </span>
                          )}
                        </td>
                        <td className="px-4 py-3.5">
                          <span
                            className={`inline-flex items-center gap-1 rounded-full border px-2.5 py-1 text-[11px] font-bold ${
                              u.active
                                ? 'border-emerald-200 bg-emerald-50 text-emerald-700'
                                : 'border-rose-200 bg-rose-50 text-rose-700'
                            }`}
                          >
                            <span
                              className={`h-1.5 w-1.5 rounded-full ${u.active ? 'bg-emerald-500' : 'bg-rose-500'}`}
                            />
                            {u.active ? 'ปกติ' : 'ระงับ'}
                          </span>
                        </td>
                        {isAdmin && (
                          <td className="px-5 py-3.5 text-right">
                            <div className="inline-flex items-center gap-1.5">
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
                                  className="rounded-lg border border-slate-200 px-2.5 py-1 text-[11px] font-bold text-slate-600 hover:bg-slate-100 cursor-pointer"
                                >
                                  {u.active ? 'ระงับสิทธิ์' : 'เปิดใช้งาน'}
                                </button>
                              )}
                              <button
                                type="button"
                                onClick={() => setResetPassUser(u)}
                                className="rounded-lg border border-sky-200 bg-sky-50 px-2.5 py-1 text-[11px] font-bold text-sky-700 hover:bg-sky-100 cursor-pointer"
                                title="รีเซ็ตรหัสผ่าน"
                              >
                                <KeyRound className="h-3.5 w-3.5" />
                              </button>
                              {u.id !== currentUser.id && u.role !== 'super_admin' && (
                                <button
                                  type="button"
                                  onClick={async () => {
                                    if (!window.confirm(`ยืนยันลบบัญชี ${u.fullName}?`)) return;
                                    await apiClient.deleteUser(u.id);
                                    flashBanner('success', `ลบบัญชี @${u.username} เรียบร้อยแล้ว`);
                                    await refreshAll();
                                  }}
                                  className="rounded-lg border border-rose-200 bg-rose-50 px-2 py-1 text-rose-600 hover:bg-rose-100 cursor-pointer"
                                  title="ลบผู้ใช้"
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
                          ไม่พบบัญชีผู้ใช้ตามเงื่อนไขที่ค้นหา
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
                <div className="flex items-start gap-2.5">
                  <Activity className="h-5 w-5 text-emerald-600 mt-0.5 shrink-0" />
                  <div>
                    <h2 className="text-lg font-extrabold text-slate-900">ทะเบียนคนไข้ (Patients Directory)</h2>
                    <p className="text-xs text-slate-500">
                      ค้นหาคนไข้ กำหนด PT ผู้รับผิดชอบ ส่งตัวสแกนหน้า และดูประวัติการฝึกย้อนหลัง
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() => void refreshAll()}
                    className="flex h-9 w-9 items-center justify-center rounded-xl border border-slate-200 bg-white text-slate-600 hover:bg-slate-50 cursor-pointer"
                    title="รีเฟรช"
                  >
                    <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setEnrollTargetPatient(null);
                      setEnrollModalOpen(true);
                    }}
                    className="inline-flex items-center gap-1.5 rounded-xl bg-emerald-600 px-4 py-2 text-xs font-extrabold text-white shadow-sm hover:bg-emerald-500 cursor-pointer"
                  >
                    <UserPlus className="h-4 w-4" />+ ลงทะเบียนคนไข้ใหม่
                  </button>
                </div>
              </div>

              <div className="flex flex-wrap items-center gap-2.5">
                <div className="flex flex-1 min-w-[260px] items-center rounded-xl border border-slate-200 bg-white p-1">
                  <Search className="h-4 w-4 text-slate-400 ml-2.5 mr-2 shrink-0" />
                  <input
                    value={patientSearch}
                    onChange={(e) => setPatientSearch(e.target.value)}
                    placeholder="ค้นหาชื่อไทย, นามสกุล, HN, เบอร์โทร.."
                    className="w-full bg-transparent px-1 py-1.5 text-xs text-slate-800 placeholder:text-slate-400 focus:outline-none"
                  />
                  <button
                    type="button"
                    onClick={() => void refreshAll()}
                    className="rounded-lg bg-emerald-600 px-3.5 py-1.5 text-xs font-bold text-white hover:bg-emerald-500 cursor-pointer"
                  >
                    ค้นหา
                  </button>
                </div>
                <select
                  value={patientFaceFilter}
                  onChange={(e) => setPatientFaceFilter(e.target.value as 'all' | 'enrolled' | 'pending')}
                  className="rounded-xl border border-slate-200 bg-white px-4 py-2.5 text-xs font-semibold text-slate-700"
                >
                  <option value="all">ทุกสถานะใบหน้า</option>
                  <option value="enrolled">ลงทะเบียนใบหน้าแล้ว (VERIFIED)</option>
                  <option value="pending">ยังไม่สแกนใบหน้า (PENDING)</option>
                </select>
              </div>

              <div className="rounded-2xl border border-slate-200 bg-white shadow-sm overflow-x-auto">
                <table className="w-full min-w-[920px] text-left text-xs">
                  <thead className="border-b border-slate-100 bg-slate-50/70 text-[11px] font-bold text-slate-500">
                    <tr>
                      <th className="px-5 py-3.5">รหัส HN</th>
                      <th className="px-4 py-3.5">ชื่อ - นามสกุล</th>
                      <th className="px-4 py-3.5">อายุ / เพศ</th>
                      <th className="px-4 py-3.5">เบอร์โทร / PT ผู้ดูแล</th>
                      <th className="px-4 py-3.5">สถานะชีวมาตร</th>
                      <th className="px-5 py-3.5 text-right">การจัดการ</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {filteredPatients.map((p) => {
                      const verified = isPatientFaceVerified(p);
                      const ptLabel = ptNameMap.get(p.delegatedToPtId || p.assignedPtId) || 'กภ. ประจำคลินิก';
                      return (
                        <tr key={p.id} className="hover:bg-slate-50/80 transition">
                          <td className="px-5 py-3.5">
                            <span className="inline-block rounded-lg border border-sky-200 bg-sky-50 px-2.5 py-1 font-mono text-[11px] font-extrabold text-sky-700">
                              {p.hn}
                            </span>
                          </td>
                          <td className="px-4 py-3.5">
                            <div className="font-bold text-slate-900">{p.fullName}</div>
                            <div className="text-[11px] text-slate-400 truncate max-w-[210px]">{p.diagnosis}</div>
                          </td>
                          <td className="px-4 py-3.5 font-medium text-slate-700">
                            {p.age} ปี / {p.gender || 'ชาย'}
                          </td>
                          <td className="px-4 py-3.5">
                            <div className="font-mono text-slate-700">{p.phone || '-'}</div>
                            <div className="text-[11px] font-semibold text-emerald-700">{ptLabel}</div>
                          </td>
                          <td className="px-4 py-3.5">
                            {verified ? (
                              <span className="inline-flex items-center gap-1.5 rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1 text-[11px] font-bold text-emerald-700">
                                <CheckCircle2 className="h-3.5 w-3.5" />
                                ลงทะเบียนหน้าแล้ว
                              </span>
                            ) : (
                              <span className="inline-flex items-center gap-1.5 rounded-full border border-amber-200 bg-amber-50 px-3 py-1 text-[11px] font-bold text-amber-800">
                                <Clock className="h-3.5 w-3.5" />
                                ยังไม่สแกนใบหน้า
                              </span>
                            )}
                          </td>
                          <td className="px-5 py-3.5 text-right">
                            <div className="inline-flex items-center gap-1.5">
                              <button
                                type="button"
                                onClick={() => setPatientModalState({ patient: p, tab: 'profile' })}
                                className="inline-flex items-center gap-1 rounded-xl bg-blue-600 px-2.5 py-1.5 text-[11px] font-bold text-white hover:bg-blue-500 cursor-pointer"
                              >
                                <FileText className="h-3.5 w-3.5" />
                                เวชระเบียน
                              </button>
                              <button
                                type="button"
                                onClick={() => {
                                  setEnrollTargetPatient(p);
                                  setEnrollModalOpen(true);
                                }}
                                className="rounded-xl border border-emerald-200 bg-emerald-50 px-2.5 py-1.5 text-[11px] font-bold text-emerald-700 hover:bg-emerald-100 cursor-pointer"
                              >
                                ส่งตัวสแกนหน้า
                              </button>
                              <button
                                type="button"
                                onClick={() => setPatientModalState({ patient: p, tab: 'history' })}
                                className="rounded-xl border border-slate-200 bg-white px-2.5 py-1.5 text-[11px] font-bold text-slate-700 hover:bg-slate-50 cursor-pointer"
                              >
                                ดูประวัติการฝึก
                              </button>
                              {isAdmin && (
                                <button
                                  type="button"
                                  onClick={async () => {
                                    if (!window.confirm(`ยืนยันลบข้อมูลคนไข้ ${p.fullName} (รวม Face ID)?`)) return;
                                    await apiClient.deletePatientCascading(p.id);
                                    flashBanner('success', `ลบข้อมูลคนไข้ ${p.fullName} แล้ว`);
                                    await refreshAll();
                                  }}
                                  className="rounded-xl border border-rose-200 bg-rose-50 p-1.5 text-rose-600 hover:bg-rose-100 cursor-pointer"
                                  title="ลบประวัติ (เฉพาะ Admin)"
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
                          ไม่พบข้อมูลคนไข้ กด &ldquo;+ ลงทะเบียนคนไข้ใหม่&rdquo; เพื่อเพิ่มผู้ป่วย
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
                <div className="flex items-start gap-2.5">
                  <UserCheck className="h-5 w-5 text-emerald-600 mt-0.5 shrink-0" />
                  <div>
                    <h2 className="text-lg font-extrabold text-slate-900">ทำเนียบนักกายภาพบำบัด (PT Directory)</h2>
                    <p className="text-xs text-slate-500">
                      รายชื่อนักกายภาพบำบัดประจำโรงพยาบาล เลขที่ใบอนุญาตประกอบวิชาชีพ และคนไข้ที่ดูแล
                    </p>
                  </div>
                </div>
                <button
                  type="button"
                  onClick={() => void refreshAll()}
                  className="flex h-9 w-9 items-center justify-center rounded-xl border border-slate-200 bg-white text-slate-600 hover:bg-slate-50 cursor-pointer"
                >
                  <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
                </button>
              </div>

              <div className="relative max-w-md">
                <Search className="h-4 w-4 text-slate-400 absolute left-3.5 top-1/2 -translate-y-1/2" />
                <input
                  value={ptSearch}
                  onChange={(e) => setPtSearch(e.target.value)}
                  placeholder="ค้นหาชื่อ PT, เลขที่ใบอนุญาต, ความเชี่ยวชาญ..."
                  className="w-full rounded-xl border border-slate-200 bg-white pl-10 pr-4 py-2 text-xs text-slate-800 placeholder:text-slate-400 focus:border-emerald-500 focus:outline-none"
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
                          <div className="flex items-center gap-3">
                            <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-gradient-to-br from-emerald-500 to-teal-600 text-base font-black text-white shadow-sm">
                              {pt.fullName.trim().slice(0, 1) || 'P'}
                            </div>
                            <div>
                              <h3 className="text-sm font-extrabold text-slate-900">{pt.fullName}</h3>
                              <p className="font-mono text-[11px] font-bold text-sky-600">@{pt.username}</p>
                            </div>
                          </div>
                          <div className="flex items-center gap-1.5">
                            <span className="rounded-full bg-emerald-50 border border-emerald-200 px-2.5 py-0.5 text-[11px] font-bold text-emerald-700">
                              PT
                            </span>
                            {isAdmin && (
                              <button
                                type="button"
                                onClick={async () => {
                                  if (!window.confirm(`ยืนยันลบบัญชี PT ${pt.fullName}?`)) return;
                                  await apiClient.deleteUser(pt.id);
                                  flashBanner('success', `ลบบัญชี PT ${pt.fullName} แล้ว`);
                                  await refreshAll();
                                }}
                                className="rounded-lg p-1 text-slate-400 hover:bg-rose-50 hover:text-rose-600 cursor-pointer"
                                title="ลบบัญชี PT"
                              >
                                <Trash2 className="h-4 w-4" />
                              </button>
                            )}
                          </div>
                        </div>

                        <div className="flex items-center justify-between rounded-xl border border-amber-200/80 bg-amber-50/70 px-3.5 py-2 text-xs">
                          <span className="font-semibold text-amber-900">🏅 ใบประกอบวิชาชีพ:</span>
                          <span className="rounded-lg bg-white border border-amber-200 px-2.5 py-0.5 font-mono text-[11px] font-bold text-slate-800">
                            {pt.licenseNo || 'PT-44012'}
                          </span>
                        </div>

                        <div className="space-y-1.5 text-xs text-slate-600">
                          <div className="flex items-center gap-2">
                            <FileText className="h-3.5 w-3.5 text-slate-400" />
                            <span>ความเชี่ยวชาญ:</span>
                            <span className="font-semibold text-slate-900">{pt.specialty || 'กายภาพบำบัดทั่วไป'}</span>
                          </div>
                          <div className="flex items-center justify-between">
                            <span className="flex items-center gap-2">
                              <Users className="h-3.5 w-3.5 text-slate-400" />
                              คนไข้ที่ดูแล:
                            </span>
                            <span className="font-extrabold text-sky-600">{ptPatients.length} ท่าน</span>
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
                        className="flex w-full items-center justify-between border-t border-slate-100 pt-3 text-xs font-bold text-slate-700 hover:text-emerald-700 cursor-pointer"
                      >
                        <span>ดูรายละเอียดและคนไข้</span>
                        <ChevronRight className="h-4 w-4" />
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
                <div className="flex items-start gap-2.5">
                  <Gamepad2 className="h-5 w-5 text-emerald-600 mt-0.5 shrink-0" />
                  <div>
                    <h2 className="text-lg font-extrabold text-slate-900">
                      ท่ากายภาพบำบัดและมินิเกมกระตุ้นสมอง (Exercises &amp; Modpacks)
                    </h2>
                    <p className="text-xs text-slate-500">
                      จัดการชุดคำถาม Physical Card Modpack และปลดล็อกโหมด AI Pose Detection สำหรับตู้ Kiosk
                    </p>
                  </div>
                </div>
                <button
                  type="button"
                  onClick={() => setCardGameModalOpen(true)}
                  className="inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-emerald-600 to-blue-600 px-4 py-2 text-xs font-extrabold text-white shadow hover:opacity-95 cursor-pointer"
                >
                  <Gamepad2 className="h-4 w-4" />
                  ทดลองเล่นการ์ดกายภาพ 10 ข้อ
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
                <div className="text-xs">
                  <div className="font-extrabold text-slate-900">
                    สถานะตรวจจับท่ากายภาพอัตโนมัติ (AI Pose Detection):{' '}
                    <span className={systemSettings.aiPoseDetectionEnabled ? 'text-emerald-700' : 'text-amber-700'}>
                      {systemSettings.aiPoseDetectionEnabled
                        ? 'เปิดใช้งาน AI Pose Detection สำหรับตู้ Kiosk แล้ว'
                        : 'อยู่ระหว่างพัฒนา (PT ล็อกไว้เฉพาะมินิเกม)'}
                    </span>
                  </div>
                  <p className="text-slate-600 mt-0.5">
                    เฉพาะผู้อำนวยการโรงพยาบาล (ADMIN) สามารถปลดล็อกเปิดใช้ท่ากายภาพ AI บนตู้ Kiosk ได้
                  </p>
                </div>
                <button
                  type="button"
                  disabled={!isAdmin || submitting}
                  title={!isAdmin ? 'เฉพาะ ADMIN เท่านั้นที่สามารถปลดล็อกได้' : undefined}
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
                          ? 'เปิดใช้งาน AI Pose Detection สำหรับตู้ Kiosk แล้ว'
                          : 'ตั้งสถานะท่ากายภาพ AI เป็นอยู่ระหว่างพัฒนาแล้ว',
                      );
                    } finally {
                      setSubmitting(false);
                    }
                  }}
                  className={`rounded-xl px-4 py-2 text-xs font-bold text-white shadow-sm disabled:opacity-45 disabled:cursor-not-allowed cursor-pointer ${
                    systemSettings.aiPoseDetectionEnabled
                      ? 'bg-amber-600 hover:bg-amber-500'
                      : 'bg-emerald-600 hover:bg-emerald-500'
                  }`}
                >
                  {systemSettings.aiPoseDetectionEnabled ? 'ตั้งสถานะเป็น: อยู่ระหว่างพัฒนา' : 'ปลดล็อกเปิดใช้ท่ากายภาพ AI'}
                </button>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Cognitive Modpacks */}
                <div className="rounded-2xl border border-slate-200 bg-white p-5 space-y-4">
                  <h3 className="text-sm font-extrabold text-slate-900">ชุดมินิเกม Modpack ({modpacks.length})</h3>
                  <div className="space-y-2.5 max-h-72 overflow-y-auto pr-1">
                    {modpacks.map((m) => (
                      <div
                        key={m.id}
                        className="flex items-center justify-between rounded-xl border border-slate-200 bg-slate-50/70 px-3.5 py-2.5 text-xs"
                      >
                        <div>
                          <div className="font-bold text-slate-900">{m.name}</div>
                          <div className="text-[11px] text-slate-500">
                            หมวดหมู่: {m.categories.join(', ')} • {m.questionCount} ข้อ
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
                            className="rounded-lg p-1.5 text-slate-400 hover:bg-rose-50 hover:text-rose-600 cursor-pointer"
                            title="ลบ Modpack"
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
                      placeholder="ชื่อ Modpack ใหม่..."
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
                      className="sm:col-span-2 rounded-xl bg-emerald-600 px-3 py-2 font-bold text-white hover:bg-emerald-500 disabled:opacity-50 cursor-pointer"
                    >
                      + เพิ่ม Modpack มินิเกม
                    </button>
                  </form>
                </div>

                {/* Rehab Exercises Library */}
                <div className="rounded-2xl border border-slate-200 bg-white p-5 space-y-4">
                  <h3 className="text-sm font-extrabold text-slate-900">คลังท่ากายภาพบำบัด ({exercises.length})</h3>
                  <div className="space-y-2.5 max-h-72 overflow-y-auto pr-1">
                    {exercises.map((ex) => (
                      <div
                        key={ex.id}
                        className="flex items-center justify-between rounded-xl border border-slate-200 bg-slate-50/70 px-3.5 py-2.5 text-xs"
                      >
                        <div>
                          <div className="font-bold text-slate-900">
                            {ex.name.th}{' '}
                            <span className="ml-1 rounded bg-sky-100 px-1.5 py-0.5 font-mono text-[10px] text-sky-800">
                              {ex.code}
                            </span>
                            <span className="ml-1 rounded bg-slate-200 px-1.5 py-0.5 text-[10px] font-bold text-slate-700 uppercase">
                              {ex.detection.side}
                            </span>
                          </div>
                          <div className="text-[11px] text-slate-500">{ex.instruction.th}</div>
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
                            className="rounded-lg p-1.5 text-slate-400 hover:bg-rose-50 hover:text-rose-600 cursor-pointer"
                            title="ลบท่ากายภาพ"
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
                      placeholder="รหัสท่า (เช่น SHOULDER_LIFT)"
                      className="rounded-xl border border-slate-200 px-3 py-2"
                    />
                    <select
                      value={exCategory}
                      onChange={(e) => setExCategory(e.target.value as ExerciseConfig['category'])}
                      className="rounded-xl border border-slate-200 px-2.5 py-2"
                    >
                      <option value="upper_limb">แขนและหัวไหล่ (Upper Limb)</option>
                      <option value="lower_limb">ขาและเข่า (Lower Limb)</option>
                      <option value="neck_trunk">คอและลำตัว (Neck &amp; Trunk)</option>
                      <option value="balance">การทรงตัว (Balance)</option>
                    </select>
                    <input
                      required
                      value={exNameTh}
                      onChange={(e) => setExNameTh(e.target.value)}
                      placeholder="ชื่อท่าภาษาไทย..."
                      className="rounded-xl border border-slate-200 px-3 py-2"
                    />
                    <select
                      value={exDefaultSide}
                      onChange={(e) => setExDefaultSide(e.target.value as BodySide | 'both')}
                      className="rounded-xl border border-slate-200 px-2.5 py-2"
                    >
                      <option value="right">ข้างขวา (Right)</option>
                      <option value="left">ข้างซ้าย (Left)</option>
                      <option value="both">ทั้งสองข้าง (Both)</option>
                    </select>
                    <input
                      value={exInstructionTh}
                      onChange={(e) => setExInstructionTh(e.target.value)}
                      placeholder="คำแนะนำผู้ป่วยขณะทำท่า..."
                      className="sm:col-span-2 rounded-xl border border-slate-200 px-3 py-2"
                    />
                    <button
                      type="submit"
                      disabled={submitting}
                      className="sm:col-span-2 rounded-xl bg-slate-900 px-3 py-2 font-bold text-white hover:bg-slate-800 disabled:opacity-50 cursor-pointer"
                    >
                      + เพิ่มท่ากายภาพบำบัด
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
                <div className="flex items-start gap-3.5">
                  <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl bg-emerald-50 border border-emerald-200 text-emerald-700">
                    <Users className="h-6 w-6" />
                  </div>
                  <div>
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="rounded-lg bg-sky-50 border border-sky-200 px-2.5 py-0.5 font-mono text-[11px] font-extrabold text-sky-700">
                        HN: {selectedSchedulePatient?.hn ?? '-'}
                      </span>
                      <span className="inline-flex items-center gap-1 rounded-full bg-emerald-50 border border-emerald-200 px-2.5 py-0.5 text-[11px] font-bold text-emerald-700">
                        <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" />
                        กำลังเข้ารับการฟื้นฟู (Active)
                      </span>
                    </div>
                    <h2 className="mt-1 text-lg font-black text-slate-900">
                      {selectedSchedulePatient?.fullName ?? 'ยังไม่มีคนไข้ในระบบ'}
                    </h2>
                    <p className="text-xs text-slate-500">
                      การวินิจฉัย: {selectedSchedulePatient?.diagnosis ?? '-'} • PT ผู้รับผิดชอบ:{' '}
                      <span className="font-semibold text-slate-700">
                        {selectedSchedulePatient
                          ? ptNameMap.get(selectedSchedulePatient.delegatedToPtId || selectedSchedulePatient.assignedPtId) ||
                            'กภ. ประจำคลินิก'
                          : '-'}
                      </span>
                    </p>
                  </div>
                </div>

                <div className="flex flex-col items-stretch sm:items-end gap-2">
                  <div className="flex flex-wrap items-center gap-2">
                    <select
                      value={selectedSchedulePatient?.id ?? ''}
                      onChange={(e) => setSelectedSchedulePatientId(e.target.value)}
                      className="rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-xs font-bold text-slate-800"
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
                      className="inline-flex items-center gap-1 rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs font-bold text-slate-600 hover:bg-rose-50 hover:text-rose-700 cursor-pointer"
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                      ล้างตารางนัดทั้งหมด
                    </button>
                  </div>
                  <div className="flex flex-wrap items-center gap-2">
                    <button
                      type="button"
                      onClick={() => setWeeklyEditorOpen((v) => !v)}
                      className="inline-flex items-center gap-1.5 rounded-xl border border-emerald-200 bg-emerald-50 px-3 py-2 text-xs font-bold text-emerald-700 hover:bg-emerald-100 cursor-pointer"
                    >
                      <SlidersHorizontal className="h-3.5 w-3.5" />
                      แผนรายสัปดาห์ 7 วัน
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        setSelectedCalendarDateIso(new Date().toISOString().slice(0, 10));
                        setScheduleModalMode('create');
                      }}
                      className="inline-flex items-center justify-center gap-1.5 rounded-xl bg-gradient-to-r from-emerald-600 to-blue-600 px-4 py-2 text-xs font-extrabold text-white shadow hover:opacity-95 cursor-pointer"
                    >
                      <Plus className="h-4 w-4" />
                      + สร้างตารางฝึกใหม่
                    </button>
                  </div>
                </div>
              </div>

              <div className="flex items-center justify-between">
                <div className="inline-flex rounded-xl border border-slate-200 bg-white p-1 text-xs font-bold">
                  <button
                    type="button"
                    onClick={() => setScheduleSubTab('individual')}
                    className={`rounded-lg px-4 py-1.5 transition cursor-pointer ${
                      scheduleSubTab === 'individual' ? 'bg-emerald-50 text-emerald-700' : 'text-slate-500'
                    }`}
                  >
                    ปฏิทินรายบุคคล
                  </button>
                  <button
                    type="button"
                    onClick={() => setScheduleSubTab('all')}
                    className={`rounded-lg px-4 py-1.5 transition cursor-pointer ${
                      scheduleSubTab === 'all' ? 'bg-emerald-50 text-emerald-700' : 'text-slate-500'
                    }`}
                  >
                    ปฏิทินคนไข้รวม ({patients.length})
                  </button>
                </div>

                <button
                  type="button"
                  onClick={() => void refreshAll()}
                  className="flex h-8 w-8 items-center justify-center rounded-xl border border-slate-200 bg-white text-slate-600 hover:bg-slate-50 cursor-pointer"
                >
                  <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
                </button>
              </div>

              {weeklyEditorOpen && selectedSchedulePatient && (
                <div className="rounded-2xl border-2 border-emerald-200 bg-white p-5 space-y-4 shadow-md">
                  <div className="flex items-center justify-between">
                    <h3 className="text-sm font-extrabold text-slate-900">
                      กำหนดแผนประจำสัปดาห์ — {selectedSchedulePatient.fullName}
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
                      placeholder="ชื่อแผนฟื้นฟู"
                      className="rounded-xl border border-slate-200 px-3 py-2 font-semibold"
                    />
                    <select
                      value={courseModpackId}
                      onChange={(e) => setCourseModpackId(e.target.value)}
                      className="rounded-xl border border-slate-200 px-3 py-2 font-semibold"
                    >
                      {modpacks.map((m) => (
                        <option key={m.id} value={m.id}>
                          {m.name} ({m.questionCount} ข้อ)
                        </option>
                      ))}
                    </select>
                  </div>
                  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2.5">
                    {weeklySchedule.map((day, idx) => (
                      <div key={idx} className="rounded-xl border border-slate-200 bg-slate-50/80 p-3 space-y-2 text-xs">
                        <div className="flex items-center justify-between">
                          <span className="font-extrabold text-slate-800">วัน{WEEKDAY_NAMES_TH[idx]}</span>
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
                            <option value="training">วันฝึก</option>
                            <option value="rest">วันพัก</option>
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
                            <option value="minigame">🎮 มินิเกมการ์ด</option>
                            <option
                              value="physio_exercise"
                              disabled={!isAdmin && !systemSettings.aiPoseDetectionEnabled}
                            >
                              🏋️ ท่ากายภาพ AI {!isAdmin && !systemSettings.aiPoseDetectionEnabled ? '(ล็อก)' : ''}
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
                      className="rounded-xl border border-slate-200 px-4 py-2 text-xs font-bold"
                    >
                      ยกเลิก
                    </button>
                    <button
                      type="button"
                      disabled={submitting}
                      onClick={() => void handleSaveWeeklyCourse()}
                      className="rounded-xl bg-emerald-600 px-5 py-2 text-xs font-extrabold text-white hover:bg-emerald-500"
                    >
                      บันทึกตารางฝึก 7 วัน
                    </button>
                  </div>
                </div>
              )}

              {scheduleSubTab === 'individual' ? (
                <>
                  <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
                    <div className="rounded-2xl border border-slate-200 bg-white p-4 flex items-center justify-between">
                      <div>
                        <div className="text-[11px] font-semibold text-slate-500">ตารางนัดหมายทั้งหมด</div>
                        <div className="mt-1 text-xl font-black text-slate-900">{scheduleStats.total} นัด</div>
                      </div>
                      <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-emerald-50 text-emerald-600">
                        <CalendarDays className="h-5 w-5" />
                      </div>
                    </div>
                    <div className="rounded-2xl border border-slate-200 bg-white p-4 flex items-center justify-between">
                      <div>
                        <div className="text-[11px] font-semibold text-slate-500">กำลังรักษา (Active)</div>
                        <div className="mt-1 text-xl font-black text-emerald-600">{scheduleStats.active} นัด</div>
                      </div>
                      <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-emerald-50 text-emerald-600">
                        <Activity className="h-5 w-5" />
                      </div>
                    </div>
                    <div className="rounded-2xl border border-slate-200 bg-white p-4 flex items-center justify-between">
                      <div>
                        <div className="text-[11px] font-semibold text-slate-500">พักชั่วคราว (Rest)</div>
                        <div className="mt-1 text-xl font-black text-amber-600">{scheduleStats.paused} นัด</div>
                      </div>
                      <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-amber-50 text-amber-500">
                        <PauseCircle className="h-5 w-5" />
                      </div>
                    </div>
                    <div className="rounded-2xl border border-slate-200 bg-white p-4 flex items-center justify-between">
                      <div>
                        <div className="text-[11px] font-semibold text-slate-500">สำเร็จแล้ว (Completed)</div>
                        <div className="mt-1 text-xl font-black text-blue-600">{scheduleStats.completed} นัด</div>
                      </div>
                      <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-sky-50 text-sky-600">
                        <CheckCircle2 className="h-5 w-5" />
                      </div>
                    </div>
                  </div>

                  <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm space-y-4">
                    <div className="flex flex-wrap items-center justify-between gap-3">
                      <div className="flex items-center gap-2">
                        <Calendar className="h-4 w-4 text-emerald-600" />
                        <h3 className="text-sm font-black text-slate-900">
                          {THAI_MONTHS[calendarDate.getMonth()]} {calendarDate.getFullYear() + 543}{' '}
                          <span className="font-normal text-slate-400">
                            ({EN_MONTHS[calendarDate.getMonth()]} {calendarDate.getFullYear()})
                          </span>
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
                        <span className="inline-flex items-center gap-1">
                          <span className="h-2 w-2 rounded-full bg-emerald-500" /> กำลังรักษา
                        </span>
                        <span className="inline-flex items-center gap-1">
                          <span className="h-2 w-2 rounded-full bg-amber-400" /> พักการฝึก
                        </span>
                        <span className="inline-flex items-center gap-1">
                          <span className="h-2 w-2 rounded-full bg-blue-500" /> สำเร็จครบโปรแกรม
                        </span>
                        <span className="inline-flex items-center gap-1">
                          <span className="h-2 w-2 rounded-full bg-rose-500" /> ขาดการฝึก
                        </span>
                      </div>
                    </div>

                    <div className="grid grid-cols-7 gap-2 text-center text-[11px] font-bold text-slate-400">
                      {CALENDAR_HEADERS_TH.map((h) => (
                        <div key={h} className="py-1">
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
                                  className={`inline-block rounded px-1.5 py-0.5 font-bold ${
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
                                    ? 'สำเร็จครบ'
                                    : customForDay[0].status === 'REST'
                                      ? 'พักการฝึก'
                                      : customForDay[0].status === 'MISSED'
                                        ? 'ขาดการฝึก'
                                        : 'กำลังรักษา'}
                                </span>
                              ) : isTraining ? (
                                <span className="inline-block rounded bg-emerald-100 px-1.5 py-0.5 font-bold text-emerald-800">
                                  {planDay.activityType === 'physio_exercise' ? 'ฝึก AI' : 'มินิเกม'}
                                </span>
                              ) : (
                                <span className="text-slate-300">พักฟื้น</span>
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
                  <table className="w-full text-left text-xs">
                    <thead className="border-b border-slate-100 bg-slate-50 text-[11px] font-bold text-slate-500">
                      <tr>
                        <th className="px-5 py-3">HN</th>
                        <th className="px-4 py-3">ชื่อคนไข้</th>
                        <th className="px-4 py-3">PT ผู้รับผิดชอบ</th>
                        <th className="px-4 py-3">วันนัดถัดไป / จำนวนวันฝึก</th>
                        <th className="px-5 py-3 text-right">จัดการ</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100">
                      {patients.map((p) => (
                        <tr key={p.id}>
                          <td className="px-5 py-3 font-mono font-bold text-sky-700">{p.hn}</td>
                          <td className="px-4 py-3 font-bold text-slate-900">{p.fullName}</td>
                          <td className="px-4 py-3 text-slate-600">
                            {ptNameMap.get(p.delegatedToPtId || p.assignedPtId) || 'กภ. ประจำคลินิก'}
                          </td>
                          <td className="px-4 py-3 font-semibold text-emerald-700">
                            {p.nextAppointmentDate || '-'} ({p.rehabSchedules?.length ?? 0} นัด)
                          </td>
                          <td className="px-5 py-3 text-right">
                            <button
                              type="button"
                              onClick={() => {
                                setSelectedSchedulePatientId(p.id);
                                setScheduleSubTab('individual');
                              }}
                              className="rounded-lg bg-emerald-50 px-3 py-1 font-bold text-emerald-700 hover:bg-emerald-100 cursor-pointer"
                            >
                              เปิดปฏิทิน
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
                <div className="flex items-start gap-2.5">
                  <ShieldAlert className="h-5 w-5 text-rose-600 mt-0.5 shrink-0" />
                  <div>
                    <h2 className="text-lg font-extrabold text-slate-900">
                      ประวัติและความปลอดภัย (Audit, Bans &amp; Security)
                    </h2>
                    <p className="text-xs text-slate-500">
                      คำเรียกเจ้าหน้าที่ฉุกเฉิน (Realtime), บันทึก Audit Log, บังคับตัดเซสชัน, ระบบแบน และจัดการตู้ Kiosk
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
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
                    className="rounded-xl border border-amber-200 bg-amber-50 px-3 py-1.5 text-[11px] font-bold text-amber-800 hover:bg-amber-100 cursor-pointer"
                  >
                    🔔 ทดสอบแจ้งเตือนฉุกเฉิน
                  </button>
                  <button
                    type="button"
                    onClick={() => void refreshAll()}
                    className="flex h-9 w-9 items-center justify-center rounded-xl bg-slate-900 text-white hover:bg-slate-800 cursor-pointer"
                  >
                    <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
                  </button>
                </div>
              </div>

              {openStaffCalls.length > 0 && (
                <div className="rounded-2xl bg-gradient-to-r from-blue-600 to-indigo-600 p-4 text-white shadow-md flex flex-wrap items-center justify-between gap-3">
                  <div className="flex items-center gap-3">
                    <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-white/15">
                      <Bell className="h-5 w-5 animate-bounce" />
                    </div>
                    <div>
                      <div className="text-sm font-black">
                        มีคนไข้ต้องการความช่วยเหลือที่ตู้ Kiosk ({openStaffCalls.length} รายการ)!
                      </div>
                      <div className="text-[11px] text-blue-100">
                        เชื่อมต่อ Realtime Supabase + Polling อัตโนมัติเมื่อคนไข้กดเรียกเจ้าหน้าที่
                      </div>
                    </div>
                  </div>
                  <button
                    type="button"
                    onClick={() => setSecuritySubTab('calls')}
                    className="rounded-xl bg-white px-4 py-2 text-xs font-extrabold text-blue-700 shadow hover:bg-blue-50 cursor-pointer"
                  >
                    ดูรายการเรียกเจ้าหน้าที่ทันที
                  </button>
                </div>
              )}

              <div className="flex flex-wrap items-center gap-2 text-xs font-bold">
                {(
                  [
                    { id: 'calls', label: `🔔 คำเรียกเจ้าหน้าที่ (${staffCalls.length})` },
                    { id: 'audit', label: `🕒 บันทึก Audit Logs (${auditLogs.length})` },
                    { id: 'sessions', label: `👥 เซสชันและการเตะออก (${activeSessions.length})` },
                    { id: 'bans', label: `⊘ การจัดการแบน (${bannedTargets.length})` },
                    { id: 'kiosks', label: `🖥 อุปกรณ์ตู้ Kiosk (${kiosks.length})` },
                  ] as const
                ).map((sub) => (
                  <button
                    key={sub.id}
                    type="button"
                    onClick={() => setSecuritySubTab(sub.id)}
                    className={`rounded-xl px-4 py-2 transition cursor-pointer ${
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
                    <span className="font-extrabold text-slate-900">รายการคำขอความช่วยเหลือจากตู้ Kiosk</span>
                    <span className="text-[11px] text-slate-400">Realtime + Polling 10 วินาที</span>
                  </div>
                  <div className="overflow-x-auto">
                    <table className="w-full min-w-[700px] text-left text-xs">
                      <thead className="bg-slate-50 text-[11px] font-bold text-slate-500">
                        <tr>
                          <th className="px-5 py-3">เวลาแจ้ง</th>
                          <th className="px-4 py-3">ตู้ KIOSK</th>
                          <th className="px-4 py-3">สาเหตุ</th>
                          <th className="px-4 py-3">สถานะ</th>
                          <th className="px-5 py-3 text-right">ดำเนินการ</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100">
                        {staffCalls.map((call) => (
                          <tr key={call.id}>
                            <td className="px-5 py-3 text-slate-600 whitespace-nowrap">
                              {formatThaiDateTime(call.createdAt)}
                            </td>
                            <td className="px-4 py-3">
                              <span className="rounded-lg bg-sky-50 border border-sky-200 px-2.5 py-1 font-bold text-sky-700">
                                {call.kioskId}
                              </span>
                            </td>
                            <td className="px-4 py-3 font-semibold text-rose-600">{call.reason}</td>
                            <td className="px-4 py-3">
                              <span
                                className={`inline-flex items-center gap-1 rounded-full px-2.5 py-1 text-[11px] font-bold ${
                                  call.status === 'resolved'
                                    ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                                    : call.status === 'acknowledged'
                                      ? 'bg-amber-50 text-amber-800 border border-amber-200'
                                      : 'bg-sky-50 text-sky-700 border border-sky-200'
                                }`}
                              >
                                {call.status === 'resolved'
                                  ? '✓ แก้ไขแล้ว'
                                  : call.status === 'acknowledged'
                                    ? '⏳ กำลังตรวจสอบ'
                                    : '🔔 รอเจ้าหน้าที่'}
                              </span>
                            </td>
                            <td className="px-5 py-3 text-right">
                              <div className="inline-flex items-center gap-1.5">
                                {call.status === 'open' && (
                                  <button
                                    type="button"
                                    onClick={async () => {
                                      await apiClient.acknowledgeStaffCall(call.id);
                                      flashBanner('success', `รับเรื่องแจ้งเตือนจาก ${call.kioskId} แล้ว`);
                                      await refreshAll();
                                    }}
                                    className="rounded-lg border border-amber-300 bg-amber-50 px-2.5 py-1 text-[11px] font-bold text-amber-800 hover:bg-amber-100 cursor-pointer"
                                  >
                                    รับเรื่อง
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
                                    className="rounded-lg bg-emerald-600 px-3 py-1 text-[11px] font-bold text-white hover:bg-emerald-500 cursor-pointer"
                                  >
                                    แก้ไขแล้ว
                                  </button>
                                )}
                              </div>
                            </td>
                          </tr>
                        ))}
                        {staffCalls.length === 0 && (
                          <tr>
                            <td colSpan={5} className="px-6 py-8 text-center text-slate-400">
                              ยังไม่มีคำเรียกเจ้าหน้าที่ฉุกเฉิน
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
                      403 RBAC Policy: เฉพาะผู้อำนวยการโรงพยาบาล (Super Admin) เท่านั้นที่เข้าถึงบันทึก Audit Log ได้
                    </div>
                  ) : (
                    <>
                      <div className="relative max-w-md">
                        <Search className="h-4 w-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
                        <input
                          value={auditSearch}
                          onChange={(e) => setAuditSearch(e.target.value)}
                          placeholder="กรอง Audit Log ตาม Action, ชื่อผู้ทำรายการ, IP..."
                          className="w-full rounded-xl border border-slate-200 pl-9 pr-3 py-2 text-xs"
                        />
                      </div>
                      <div className="space-y-2 max-h-[420px] overflow-y-auto">
                        {filteredAuditLogs.map((log) => (
                          <div
                            key={log.id}
                            className="flex flex-wrap items-center justify-between gap-2 rounded-xl border border-slate-100 bg-slate-50/80 px-4 py-2.5 text-xs"
                          >
                            <div>
                              <span className="font-mono font-extrabold text-indigo-700 mr-2">{log.eventType}</span>
                              <span className="font-bold text-slate-900">{log.actorName}</span>{' '}
                              <span className="text-slate-500">({log.actorRole})</span>
                              <div className="text-[11px] text-slate-600 mt-0.5">{log.detail}</div>
                            </div>
                            <div className="text-right text-[11px] text-slate-400 font-mono">
                              <div>
                                IP: {log.ipAddress} • Dev: {log.deviceId}
                              </div>
                              <div>{formatThaiDateTime(log.createdAt)}</div>
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
                      403 RBAC Policy: เฉพาะผู้อำนวยการโรงพยาบาล (Super Admin) เท่านั้นที่สั่งตัดเซสชันผู้ใช้ได้
                    </div>
                  ) : (
                    <div className="space-y-2">
                      {activeSessions.map((s) => (
                        <div
                          key={s.token}
                          className="flex flex-wrap items-center justify-between gap-2 rounded-xl border border-slate-200 px-4 py-3 text-xs"
                        >
                          <div>
                            <div className="font-bold text-slate-900">
                              {ptNameMap.get(s.userId) ?? s.userId}{' '}
                              <span className="ml-1 rounded bg-slate-100 px-2 py-0.5 font-mono text-[10px] uppercase">
                                {s.role}
                              </span>
                            </div>
                            <div className="font-mono text-[11px] text-slate-500">
                              IP: {s.ipAddress} • Device: {s.deviceId} • Login: {formatThaiDateTime(s.createdAt)}
                            </div>
                          </div>
                          <button
                            type="button"
                            onClick={async () => {
                              await apiClient.forceLogoutUser(s.userId);
                              flashBanner('success', 'เตะเซสชันออกจากระบบเรียบร้อย');
                              await refreshAll();
                            }}
                            className="rounded-xl bg-rose-600 px-3 py-1.5 text-[11px] font-bold text-white hover:bg-rose-500 cursor-pointer"
                          >
                            เตะออกจากระบบ (Force Kick)
                          </button>
                        </div>
                      ))}
                      {activeSessions.length === 0 && (
                        <p className="text-center text-xs text-slate-400 py-6">ไม่มีเซสชันที่กำลังออนไลน์ในระบบ</p>
                      )}
                    </div>
                  )}
                </div>
              )}

              {securitySubTab === 'bans' && (
                <div className="rounded-2xl border border-slate-200 bg-white p-5 space-y-4">
                  {!isAdmin ? (
                    <div className="rounded-xl bg-amber-50 border border-amber-200 p-4 text-xs font-bold text-amber-800">
                      403 RBAC Policy: เฉพาะผู้อำนวยการโรงพยาบาล (Super Admin) เท่านั้นที่แบน IP/Device ได้
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
                          className="rounded-xl border border-slate-200 px-3 py-2 font-bold"
                        >
                          <option value="ip">Ban IP Address</option>
                          <option value="device">Ban Device ID</option>
                        </select>
                        <input
                          required
                          value={banValue}
                          onChange={(e) => setBanValue(e.target.value)}
                          placeholder="เช่น 192.168.1.50"
                          className="rounded-xl border border-slate-200 px-3 py-2 font-mono"
                        />
                        <input
                          value={banReason}
                          onChange={(e) => setBanReason(e.target.value)}
                          placeholder="สาเหตุการแบน..."
                          className="rounded-xl border border-slate-200 px-3 py-2"
                        />
                        <button
                          type="submit"
                          className="inline-flex items-center justify-center gap-1 rounded-xl bg-rose-600 px-3 py-2 font-bold text-white hover:bg-rose-500 cursor-pointer"
                        >
                          <Ban className="h-3.5 w-3.5" /> แบนเป้าหมาย
                        </button>
                      </form>

                      <div className="space-y-2">
                        {bannedTargets.map((b) => (
                          <div
                            key={b.id}
                            className="flex items-center justify-between rounded-xl border border-rose-200 bg-rose-50/50 px-4 py-2.5 text-xs"
                          >
                            <div>
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
                              className="rounded-lg border border-slate-200 bg-white px-3 py-1 font-bold text-slate-700 hover:bg-slate-100 cursor-pointer"
                            >
                              ปลดแบน (Unban)
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
                    <div className="flex items-center justify-between">
                      <h3 className="text-sm font-extrabold text-slate-900">อุปกรณ์ตู้ Kiosk ในระบบ</h3>
                      {isAdmin && (
                        <button
                          type="button"
                          onClick={() => {
                            setNewKioskCode(`KIOSK-OPD-0${kiosks.length + 1}`);
                            setNewKioskLocation('OPD Rehab Zone B');
                            setRegisterKioskModalOpen(true);
                          }}
                          className="rounded-xl bg-emerald-600 px-3 py-1.5 text-[11px] font-bold text-white hover:bg-emerald-500 cursor-pointer"
                        >
                          + ลงทะเบียนตู้ Kiosk ใหม่
                        </button>
                      )}
                    </div>
                    {kiosks.map((k) => (
                      <div
                        key={k.kioskId}
                        className="flex items-center justify-between rounded-xl border border-slate-200 bg-slate-50/70 px-4 py-3 text-xs"
                      >
                        <div>
                          <div className="font-bold text-slate-900">
                            {k.kioskId}{' '}
                            {k.pairingToken && (
                              <span className="ml-1 rounded bg-indigo-50 px-2 py-0.5 font-mono text-[10px] text-indigo-700">
                                Token: {k.pairingToken}
                              </span>
                            )}
                          </div>
                          <div className="text-[11px] text-slate-500">
                            {k.location} • State: {k.currentState}
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          <span
                            className={`rounded-full px-2.5 py-0.5 text-[11px] font-bold ${
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
                    <h3 className="text-sm font-extrabold text-slate-900">ตั้งค่าความปลอดภัย AI หน้าตู้ Kiosk</h3>
                    <label className="flex items-center justify-between rounded-xl border border-slate-200 px-3.5 py-2.5">
                      <span className="font-bold text-slate-700">เปิดใช้งาน AI Face Scan (512D)</span>
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
          flashBanner('success', `บันทึกข้อมูลและ Face ID ของ ${p.fullName} (${p.hn}) เรียบร้อยแล้ว`);
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

      {/* Physical Card Game Modal */}
      {cardGameModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/75 p-4 backdrop-blur-sm">
          <div className="w-full max-w-4xl max-h-[92vh] overflow-y-auto rounded-3xl bg-white p-6 shadow-2xl border border-slate-200 space-y-4">
            <div className="flex items-center justify-between border-b border-slate-100 pb-3">
              <div className="flex items-center gap-2">
                <Gamepad2 className="h-5 w-5 text-emerald-600" />
                <h3 className="text-base font-black text-slate-900">เกมการ์ดกายภาพบำบัด 10 ข้อ (ทดสอบระบบมือ/กล้อง หรือคลิกปุ่มตอบ)</h3>
              </div>
              <button
                type="button"
                onClick={() => setCardGameModalOpen(false)}
                className="rounded-xl p-1 text-slate-400 hover:bg-slate-100"
              >
                <X className="h-5 w-5" />
              </button>
            </div>
            <PhysicalCardGame
              patientName={selectedSchedulePatient?.fullName ?? currentUser.fullName}
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
